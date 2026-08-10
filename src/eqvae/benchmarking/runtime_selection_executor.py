# Copyright 2026 HiperMaximus
# ruff: noqa: C901, DOC501, PERF401, PLR0912, PLR0913, PLR0914, PLR0915, PLW0717, RUF100, SLF001
# pyright: reportAny=false, reportArgumentType=false, reportAssignmentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportPrivateUsage=false, reportReturnType=false, reportUnnecessaryCast=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false
"""Kaggle executor for the selected-runtime benchmark slice.

This module deliberately keeps the final write gate in
`eqvae.benchmarking.runtime_selection`. The executor's job is to collect fresh
runtime evidence and hand it to that fail-closed writer; it never promotes v8
pretest rows.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import subprocess  # noqa: S404
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from eqvae.benchmarking import real_data_runtime_pretest as pretest
from eqvae.benchmarking.io import CsvRow, JsonObject, JsonValue, write_json
from eqvae.benchmarking.row_id import compose_selected_row_id
from eqvae.benchmarking.runtime_schema import (
    CORRUPTION_CHECK_COLUMNS,
    DATALOADER_MATRIX_COLUMNS,
    GATE_HEALTH_COLUMNS,
    NUMERICAL_CHECK_COLUMNS,
    validate_efficiency_proof_reference_batch_size,
)
from eqvae.benchmarking.runtime_selection import (
    AMP_CONSERVATIVE,
    AMP_OFF_FP32,
    AMP_SCALAR_GATE_RELAXED,
    BRANCHLESS_ALL,
    COMPILE_MODEL_FORWARD,
    COMPILE_NONE,
    COMPILE_STEP,
    DEFAULT_RUNTIME_POLICY_ID,
    DUAL_T4_DDP,
    EXPECTED_DUAL_T4_COUNT,
    EXPECTED_MACHINE_SHAPE,
    FAIL_STATUS,
    INDEXED_MASKED,
    PASS_STATUS,
    REQUIRED_CORRUPTION_SPLITS,
    RUNTIME_SELECTION_KIND,
    RUNTIME_SELECTION_SOURCE,
    SINGLE_VISIBLE_T4,
    SKIPPED_UNSUPPORTED,
    STAIN_CORRUPTOR_QA_FILENAME,
    RuntimeSelectionArtifactPaths,
    RuntimeSelectionBenchmarkRequest,
    RuntimeSelectionEvidence,
    write_runtime_selection_benchmark,
)
from eqvae.benchmarking.vram_feasibility import (
    NO_OOM,
    OOM,
    headroom_below_margin,
    is_oom_error,
    probe_headroom_bytes,
)
from eqvae.config import ResolvedConfig, resolve_json_config
from eqvae.training.fastpath_precision import (
    build_fastpath_grad_scaler,
    fastpath_autocast_dtype,
    run_fastpath_optimizer_step,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from contextlib import AbstractContextManager

    import torch


@dataclass(frozen=True)
class RuntimeSelectionExecutionRequest:
    """Inputs for the Kaggle selected-runtime executor."""

    config_path: Path
    output_dir: Path
    run_name: str | None = None
    data_root: str | None = None
    v8_artifact_dir: Path | None = None


@dataclass(frozen=True)
class _ChildArgs:
    ddp_row: str | None


@dataclass(frozen=True)
class _DdpRowConfig:
    config_path: Path
    output_dir: Path
    data_root: str
    row_spec: pretest.RowSpec
    proof_reference_per_device_batch_size: int


@dataclass(frozen=True)
class _SelectionStageSettings:
    single_batch_sizes: tuple[int, ...]
    fallback_batch_sizes: tuple[int, ...]
    dual_batch_sizes: tuple[int, ...]
    corruption_strategies: tuple[str, ...]
    efficiency_batch_sizes: tuple[int, ...]
    efficiency_corruption_strategies: tuple[str, ...]
    efficiency_policies: tuple[_RuntimePolicy, ...]
    proof_reference_per_device_batch_size: int


@dataclass(frozen=True)
class _RuntimePolicy:
    runtime_policy_id: str
    precision_policy: str
    compile_scope: str
    memory_format: str = "contiguous"
    autocast_dtype: str = ""
    fp32_loss: bool = True
    grad_scaler_enabled: bool = False
    # Speed-first default (Spec 0011 S17f): the dual-T4 search measures under the same
    # cuDNN autotuning the paper-promotable run uses (matching the compiled probe/FSQ),
    # applied via _apply_backend_policy. Not a searched axis; the config omits it.
    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False
    deterministic_algorithms: bool = False
    tf32_enabled: bool = False
    matmul_precision: str = "highest"
    ddp_static_graph: bool = False
    ddp_gradient_as_bucket_view: bool = False
    optimizer_implementation: str = "adamw_default"
    zero_grad_set_to_none: bool = True
    gradient_clip_foreach: bool = True
    compile_dynamic: bool = False
    # Spec 0011 S14a: compiled fast-path recipe knobs. Eager-v5 defaults keep every
    # existing policy byte-identical; the efficiency search declares a compiled winner
    # policy that overrides them, and _row_spec threads them onto the RowSpec.
    optimize_ddp: str = ""
    compiled_autograd: bool = False
    reorder_compute_comm_overlap: bool = False
    ddp_broadcast_buffers: bool = True
    ddp_find_unused_parameters: bool = False
    ddp_bucket_cap_mb: int | None = None
    fused_optimizer: bool = False
    diagnostic_only: bool = False


@dataclass(frozen=True)
class _AmpPhaseAccounting:
    """Separate proof/timing calibration diagnostics from selection-gated skips."""

    proof_calibration_step_count: int
    proof_calibration_skipped_count: int
    timing_calibration_step_count: int
    timing_calibration_skipped_count: int
    timing_successful_optimizer_update_count: int
    selection_amp_step_skipped_count: int


@dataclass(frozen=True)
class _CollectedEvidence:
    evidence: RuntimeSelectionEvidence
    stain_corruptor_qa: JsonObject


@dataclass(frozen=True)
class _DdpLaunchResult:
    row: CsvRow
    rank_payloads: tuple[JsonObject, ...]
    command_display: str
    returncode: int
    failure_kind: str
    failure_message_hash: str
    reference_row_id: str


@dataclass(frozen=True)
class _TrainStepTelemetry:
    forward: object
    losses: object
    grad_norm: float
    param_update_norm: float
    nonfinite_count: int
    amp_step_skipped: bool


def write_runtime_selection_execution(
    request: RuntimeSelectionExecutionRequest,
) -> RuntimeSelectionArtifactPaths:
    """Run the local selected-runtime executor and write gated artifacts.

    Returns:
        Paths written by the strict selected-runtime writer.

    """
    request.output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_dir = request.output_dir / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    resolved = resolve_json_config(request.config_path)
    try:
        collected = _collect_evidence(request=request, resolved_config=resolved)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        collected = _failed_collection(request=request, failure=exc)
    write_json(
        benchmark_dir / STAIN_CORRUPTOR_QA_FILENAME,
        collected.stain_corruptor_qa,
    )
    return write_runtime_selection_benchmark(
        RuntimeSelectionBenchmarkRequest(
            config_path=request.config_path,
            output_dir=request.output_dir,
            run_name=request.run_name,
            v8_artifact_dir=request.v8_artifact_dir,
            evidence=collected.evidence,
        ),
    )


def _collect_evidence(
    *,
    request: RuntimeSelectionExecutionRequest,
    resolved_config: ResolvedConfig,
) -> _CollectedEvidence:
    settings = pretest._settings(  # noqa: SLF001
        resolved_config,
        data_root_override=request.data_root,
    )
    stage_settings = _selection_stage_settings(
        cast("Mapping[str, JsonValue]", resolved_config.effective_config),
    )
    data_proof = pretest._real_data_identity_and_clean_path_proof(settings)  # noqa: SLF001
    runtime_rows: list[CsvRow] = []

    single_row_specs = _single_row_specs(settings=settings, stage=stage_settings)
    single_rows = _run_single_rows(
        request=request,
        settings=settings,
        row_specs=single_row_specs,
    )
    single_linked = _linked_single_evidence(
        settings=settings,
        data_proof=data_proof,
        rows=single_rows,
    )
    single_rows = _rows_with_selection_scope(
        _rows_with_linked_status(
            rows=single_rows,
            data_proof=data_proof,
            linked_evidence=single_linked,
        ),
    )
    runtime_rows.extend(single_rows)

    dual_results = [
        _run_dual_row(
            request=request,
            settings=settings,
            row_spec=row_spec,
            proof_reference_per_device_batch_size=_row_proof_reference_batch_size(
                row_spec=row_spec,
                stage=stage_settings,
            ),
        )
        for row_spec in _dual_row_specs(settings=settings, stage=stage_settings)
    ]
    dual_rows = _rows_with_selection_scope(result.row for result in dual_results)
    runtime_rows.extend(dual_rows)

    dataloader_rows = [
        *_rows_with_selection_scope(
            pretest._schema_dataloader_rows(  # noqa: SLF001
                settings=settings,
                data_proof=data_proof,
                linked_evidence=single_linked,
            ),
        ),
        *_dual_dataloader_rows(settings=settings, results=dual_results),
    ]
    numerical_rows = [
        *_rows_with_selection_scope(
            pretest._schema_numerical_rows(  # noqa: SLF001
                settings=settings,
                rows=single_rows,
                linked_evidence=single_linked,
            ),
        ),
        *_dual_numerical_rows(settings=settings, results=dual_results),
    ]
    corruption_rows = [
        *_rows_with_selection_scope(
            pretest._schema_corruption_rows(  # noqa: SLF001
                settings=settings,
                rows=single_rows,
                linked_evidence=single_linked,
            ),
        ),
        *_dual_corruption_rows(settings=settings, results=dual_results),
    ]
    gate_rows = [
        *_single_gate_rows(
            gate_rows=pretest._gate_health_rows(  # noqa: SLF001
                settings=settings,
                linked_evidence=single_linked,
            ),
            runtime_rows=single_rows,
        ),
        *_dual_gate_rows(results=dual_results),
    ]
    pass_row_ids = [
        row["row_id"] for row in runtime_rows if row["status"] == PASS_STATUS
    ]
    corruption_rows.extend(
        _clean_validation_corruption_rows(
            settings=settings,
            runtime_rows=runtime_rows,
            dataloader_rows=dataloader_rows,
            dual_results=dual_results,
        ),
    )
    gate_summary = _gate_health_summary(
        gate_rows=gate_rows,
        runtime_rows=runtime_rows,
        single_linked=single_linked,
    )
    runtime_environment = _runtime_environment(results=dual_results)
    stain_corruptor_qa = _stain_corruptor_qa_payload(
        runtime_rows=runtime_rows,
        corruption_rows=corruption_rows,
        pass_row_ids=pass_row_ids,
    )
    return _CollectedEvidence(
        evidence=RuntimeSelectionEvidence(
            runtime_rows=tuple(runtime_rows),
            dataloader_rows=tuple(dataloader_rows),
            numerical_rows=tuple(numerical_rows),
            corruption_rows=tuple(corruption_rows),
            gate_health_rows=tuple(gate_rows),
            gate_health_summary=gate_summary,
            runtime_environment=runtime_environment,
        ),
        stain_corruptor_qa=stain_corruptor_qa,
    )


def _row_proof_reference_batch_size(
    *,
    row_spec: pretest.RowSpec,
    stage: _SelectionStageSettings,
) -> int:
    """Return cross-batch proof control only for the efficiency slice.

    Returns:
        Candidate batch for ordinary rows; bounded proof batch for efficiency rows.

    """
    if row_spec.candidate_role not in {
        "selected_runtime_efficiency_followup",
        "selected_runtime_efficiency_diagnostic",
    }:
        return row_spec.per_device_batch_size
    return min(
        row_spec.per_device_batch_size,
        stage.proof_reference_per_device_batch_size,
    )


def _failed_collection(
    *,
    request: RuntimeSelectionExecutionRequest,
    failure: BaseException,
) -> _CollectedEvidence:
    message = f"{type(failure).__name__}: {failure}"
    evidence = RuntimeSelectionEvidence(
        runtime_rows=(),
        dataloader_rows=(),
        numerical_rows=(),
        corruption_rows=(),
        gate_health_rows=(),
        gate_health_summary={
            "status": FAIL_STATUS,
            "benchmark_kind": RUNTIME_SELECTION_KIND,
            "benchmark_source": RUNTIME_SELECTION_SOURCE,
            "overall_status": FAIL_STATUS,
            "full_run_eligible": False,
            "logged_intervals": 0,
            "module_count": 0,
            "nonfinite_count": None,
            "failing_modules": [],
            "warning_modules": [],
            "candidate_row_ids": [],
            "failure_kind": "runtime_selection_evidence_collection_failed",
            "failure_message_hash": pretest._hash_text(message),  # noqa: SLF001
        },
        runtime_environment={
            "status": FAIL_STATUS,
            "machine_shape": EXPECTED_MACHINE_SHAPE,
            "visible_device_count": 0,
            "cuda_device_count": 0,
            "gpu_names": [],
            "world_size": 0,
            "nproc_per_node": 0,
            "rank_assignments": [],
            "child_process_launch_command": "",
            "failure_kind": "runtime_selection_evidence_collection_failed",
            "failure_message_hash": pretest._hash_text(message),  # noqa: SLF001
            "output_dir": str(request.output_dir),
        },
    )
    return _CollectedEvidence(
        evidence=evidence,
        stain_corruptor_qa={
            "status": FAIL_STATUS,
            "benchmark_kind": RUNTIME_SELECTION_KIND,
            "benchmark_source": RUNTIME_SELECTION_SOURCE,
            "full_run_eligible": False,
            "failure_kind": "runtime_selection_evidence_collection_failed",
            "failure_message_hash": pretest._hash_text(message),  # noqa: SLF001
        },
    )


def _selection_stage_settings(
    effective_config: Mapping[str, JsonValue],
) -> _SelectionStageSettings:
    runtime = _required_object(effective_config, "runtime_matrix")
    selection = _required_object(runtime, "selection_benchmark_slice")
    stages = _required_object_list(selection, "stages")
    first_stage = _stage(stages, "v8_shortlist_fp32_eager_confirmation")
    dual_stage = _stage(stages, "dual_t4_train_step_gate")
    efficiency = _optional_object(selection, "efficiency_followup")
    dual_batch_sizes = _int_tuple(dual_stage, "per_device_batch_sizes")
    efficiency_batch_sizes = (
        () if efficiency is None else _int_tuple(efficiency, "per_device_batch_sizes")
    )
    proof_reference_batch_size = _proof_reference_batch_size(
        efficiency=efficiency,
        dual_batch_sizes=dual_batch_sizes,
        efficiency_batch_sizes=efficiency_batch_sizes,
    )
    return _SelectionStageSettings(
        single_batch_sizes=_int_tuple(first_stage, "per_device_batch_sizes"),
        fallback_batch_sizes=_int_tuple(first_stage, "fallback_per_device_batch_sizes"),
        dual_batch_sizes=dual_batch_sizes,
        corruption_strategies=_str_tuple(dual_stage, "corruption_strategies"),
        efficiency_batch_sizes=efficiency_batch_sizes,
        efficiency_corruption_strategies=()
        if efficiency is None
        else _str_tuple(efficiency, "corruption_strategies"),
        efficiency_policies=()
        if efficiency is None
        else _runtime_policies(_required_object_list(efficiency, "policies")),
        proof_reference_per_device_batch_size=proof_reference_batch_size,
    )


def _proof_reference_batch_size(
    *,
    efficiency: JsonObject | None,
    dual_batch_sizes: Sequence[int],
    efficiency_batch_sizes: Sequence[int],
) -> int:
    """Validate and return the fixed linked-proof batch for efficiency rows.

    Returns:
        Zero when no efficiency stage exists, otherwise its configured proof batch.

    """
    if efficiency is None:
        return 0
    batch_size = pretest._required_int(  # noqa: SLF001
        efficiency,
        "proof_reference_per_device_batch_size",
    )
    return validate_efficiency_proof_reference_batch_size(
        batch_size=batch_size,
        dual_gate_batch_sizes=tuple(dual_batch_sizes),
        efficiency_batch_sizes=tuple(efficiency_batch_sizes),
    )


def _stage(stages: Sequence[JsonObject], name: str) -> JsonObject:
    for stage in stages:
        if stage.get("name") == name:
            return stage
    message = f"Missing selection stage: {name}"
    raise ValueError(message)


def _runtime_policies(items: Sequence[JsonObject]) -> tuple[_RuntimePolicy, ...]:
    policies: list[_RuntimePolicy] = []
    for item in items:
        precision_policy = _required_str(item, "precision_policy")
        compile_scope = _required_str(item, "compile_scope")
        if precision_policy not in {
            AMP_OFF_FP32,
            AMP_CONSERVATIVE,
            AMP_SCALAR_GATE_RELAXED,
        }:
            message = f"Unsupported precision_policy: {precision_policy}"
            raise ValueError(message)
        if compile_scope not in {COMPILE_NONE, COMPILE_MODEL_FORWARD, COMPILE_STEP}:
            message = f"Unsupported compile_scope: {compile_scope}"
            raise ValueError(message)
        optimize_ddp = _optional_str(item, "optimize_ddp") or ""
        diagnostic_only = _optional_bool(item, "diagnostic_only", default=False)
        if (
            compile_scope != COMPILE_NONE
            and optimize_ddp not in _COMPILED_OPTIMIZE_DDP_MODES
            and not diagnostic_only
        ):
            message = (
                "Compiled runtime policies must name optimize_ddp as one of "
                f"{sorted(_COMPILED_OPTIMIZE_DDP_MODES)!r}; got {optimize_ddp!r}"
            )
            raise ValueError(message)
        compiled_autograd = _optional_bool(
            item,
            "compiled_autograd",
            default=False,
        )
        if (
            optimize_ddp
            in {
                "python_reducer",
                "python_reducer_without_compiled_forward",
            }
            and not compiled_autograd
        ):
            message = f"optimize_ddp={optimize_ddp!r} requires compiled_autograd=true"
            raise ValueError(message)
        if (
            optimize_ddp == "no_optimization"
            and compiled_autograd
            and not diagnostic_only
        ):
            message = "optimize_ddp='no_optimization' requires compiled_autograd=false"
            raise ValueError(message)
        policies.append(
            _RuntimePolicy(
                runtime_policy_id=_required_str(item, "runtime_policy_id"),
                precision_policy=precision_policy,
                compile_scope=compile_scope,
                memory_format=_optional_str(item, "memory_format") or "contiguous",
                autocast_dtype=_optional_str(item, "autocast_dtype") or "",
                fp32_loss=_optional_bool(item, "fp32_loss", default=True),
                grad_scaler_enabled=_optional_bool(
                    item,
                    "grad_scaler_enabled",
                    default=precision_policy != AMP_OFF_FP32,
                ),
                cudnn_benchmark=_optional_bool(
                    item,
                    "cudnn_benchmark",
                    default=True,
                ),
                cudnn_deterministic=_optional_bool(
                    item,
                    "cudnn_deterministic",
                    default=False,
                ),
                deterministic_algorithms=_optional_bool(
                    item,
                    "deterministic_algorithms",
                    default=False,
                ),
                tf32_enabled=_optional_bool(item, "tf32_enabled", default=False),
                matmul_precision=_optional_str(item, "matmul_precision") or "highest",
                ddp_static_graph=_optional_bool(
                    item,
                    "ddp_static_graph",
                    default=False,
                ),
                ddp_gradient_as_bucket_view=_optional_bool(
                    item,
                    "ddp_gradient_as_bucket_view",
                    default=False,
                ),
                optimizer_implementation=_optional_str(
                    item,
                    "optimizer_implementation",
                )
                or "adamw_default",
                zero_grad_set_to_none=_optional_bool(
                    item,
                    "zero_grad_set_to_none",
                    default=True,
                ),
                gradient_clip_foreach=_optional_bool(
                    item,
                    "gradient_clip_foreach",
                    default=True,
                ),
                compile_dynamic=_optional_bool(item, "compile_dynamic", default=False),
                optimize_ddp=optimize_ddp,
                compiled_autograd=compiled_autograd,
                reorder_compute_comm_overlap=_optional_bool(
                    item,
                    "reorder_compute_comm_overlap",
                    default=False,
                ),
                ddp_broadcast_buffers=_optional_bool(
                    item,
                    "ddp_broadcast_buffers",
                    default=True,
                ),
                ddp_find_unused_parameters=_optional_bool(
                    item,
                    "ddp_find_unused_parameters",
                    default=False,
                ),
                ddp_bucket_cap_mb=_optional_int(item, "ddp_bucket_cap_mb"),
                fused_optimizer=_optional_bool(
                    item,
                    "fused_optimizer",
                    default=False,
                ),
                diagnostic_only=diagnostic_only,
            ),
        )
    return tuple(policies)


def _single_row_specs(
    *,
    settings: pretest.RealDataRuntimePretestSettings,
    stage: _SelectionStageSettings,
) -> tuple[pretest.RowSpec, ...]:
    batch_sizes = (*stage.fallback_batch_sizes, *stage.single_batch_sizes)
    return tuple(
        _row_spec(
            settings=settings,
            accelerator_mode=SINGLE_VISIBLE_T4,
            batch_size=batch_size,
            corruption_strategy=corruption_strategy,
            candidate_role="v8_shortlist_eager_confirmation",
        )
        for batch_size in batch_sizes
        for corruption_strategy in stage.corruption_strategies
    )


def _efficiency_row_enumerable(
    *,
    policy: _RuntimePolicy,
    batch_size: int,
    stage: _SelectionStageSettings,
) -> bool:
    """Return whether this (policy, batch) efficiency row may be enumerated (S14c).

    Compiled AMP rows use the stage's explicit fp32-eager proof batch rather than
    requiring an eager fp32 allocation at the larger timed batch. This keeps the new
    compiled bs48 candidate enumerable while preserving the prior eager-AMP slice at
    same-batch reference sizes. The parser requires the proof batch to be positive,
    present in ``dual_batch_sizes``, and no larger than the candidate. Non-AMP policies
    need no AMP companion.

    Returns:
        True unless an AMP row lacks its required fp32 proof reference.

    """
    if policy.precision_policy not in {AMP_CONSERVATIVE, AMP_SCALAR_GATE_RELAXED}:
        return True
    if policy.compile_scope != COMPILE_STEP:
        return batch_size in stage.dual_batch_sizes
    return (
        stage.proof_reference_per_device_batch_size in stage.dual_batch_sizes
        and stage.proof_reference_per_device_batch_size <= batch_size
    )


def _dual_row_specs(
    *,
    settings: pretest.RealDataRuntimePretestSettings,
    stage: _SelectionStageSettings,
) -> tuple[pretest.RowSpec, ...]:
    gate_specs = [
        _row_spec(
            settings=settings,
            accelerator_mode=DUAL_T4_DDP,
            batch_size=batch_size,
            corruption_strategy=corruption_strategy,
            candidate_role="dual_t4_train_step_gate",
        )
        for batch_size in stage.dual_batch_sizes
        for corruption_strategy in stage.corruption_strategies
    ]
    efficiency_specs = [
        _row_spec(
            settings=settings,
            accelerator_mode=DUAL_T4_DDP,
            batch_size=batch_size,
            corruption_strategy=corruption_strategy,
            candidate_role=(
                "selected_runtime_efficiency_diagnostic"
                if policy.diagnostic_only
                else "selected_runtime_efficiency_followup"
            ),
            policy=policy,
        )
        for batch_size in stage.efficiency_batch_sizes
        for corruption_strategy in stage.efficiency_corruption_strategies
        for policy in stage.efficiency_policies
        if _efficiency_row_enumerable(policy=policy, batch_size=batch_size, stage=stage)
    ]
    seen: set[str] = set()
    unique: list[pretest.RowSpec] = []
    for spec in (*gate_specs, *efficiency_specs):
        if spec.row_id in seen:
            continue
        seen.add(spec.row_id)
        unique.append(spec)
    return tuple(unique)


def _row_spec(
    *,
    settings: pretest.RealDataRuntimePretestSettings,
    accelerator_mode: str,
    batch_size: int,
    corruption_strategy: str,
    candidate_role: str,
    policy: _RuntimePolicy | None = None,
) -> pretest.RowSpec:
    del settings
    world_size = EXPECTED_DUAL_T4_COUNT if accelerator_mode == DUAL_T4_DDP else 1
    cuda_visible_devices = "0,1" if accelerator_mode == DUAL_T4_DDP else "0"
    runtime_policy = policy or _RuntimePolicy(
        runtime_policy_id=DEFAULT_RUNTIME_POLICY_ID,
        precision_policy=AMP_OFF_FP32,
        compile_scope=COMPILE_NONE,
    )
    return pretest.RowSpec(
        row_id=_row_id(
            accelerator_mode=accelerator_mode,
            batch_size=batch_size,
            precision_policy=runtime_policy.precision_policy,
            compile_scope=runtime_policy.compile_scope,
            corruption_strategy=corruption_strategy,
            runtime_policy_id=runtime_policy.runtime_policy_id,
        ),
        accelerator_mode=accelerator_mode,
        per_device_batch_size=batch_size,
        precision_policy=runtime_policy.precision_policy,
        compile_scope=runtime_policy.compile_scope,
        corruption_strategy=corruption_strategy,
        parent_synthetic_row_id="",
        candidate_role=candidate_role,
        world_size=world_size,
        nproc_per_node=world_size,
        cuda_visible_devices=cuda_visible_devices,
        runtime_policy_id=runtime_policy.runtime_policy_id,
        memory_format=runtime_policy.memory_format,
        autocast_dtype=runtime_policy.autocast_dtype,
        fp32_loss=runtime_policy.fp32_loss,
        grad_scaler_enabled=runtime_policy.grad_scaler_enabled,
        cudnn_benchmark=runtime_policy.cudnn_benchmark,
        cudnn_deterministic=runtime_policy.cudnn_deterministic,
        deterministic_algorithms=runtime_policy.deterministic_algorithms,
        tf32_enabled=runtime_policy.tf32_enabled,
        matmul_precision=runtime_policy.matmul_precision,
        ddp_static_graph=runtime_policy.ddp_static_graph,
        ddp_gradient_as_bucket_view=runtime_policy.ddp_gradient_as_bucket_view,
        optimizer_implementation=runtime_policy.optimizer_implementation,
        zero_grad_set_to_none=runtime_policy.zero_grad_set_to_none,
        gradient_clip_foreach=runtime_policy.gradient_clip_foreach,
        compile_dynamic=runtime_policy.compile_dynamic,
        optimize_ddp=runtime_policy.optimize_ddp,
        compiled_autograd=runtime_policy.compiled_autograd,
        reorder_compute_comm_overlap=runtime_policy.reorder_compute_comm_overlap,
        ddp_broadcast_buffers=runtime_policy.ddp_broadcast_buffers,
        ddp_find_unused_parameters=runtime_policy.ddp_find_unused_parameters,
        ddp_bucket_cap_mb=runtime_policy.ddp_bucket_cap_mb,
        fused_optimizer=runtime_policy.fused_optimizer,
    )


def _run_single_rows(
    *,
    request: RuntimeSelectionExecutionRequest,
    settings: pretest.RealDataRuntimePretestSettings,
    row_specs: Sequence[pretest.RowSpec],
) -> list[CsvRow]:
    rows: list[CsvRow] = []
    for row_spec in row_specs:
        rows.append(
            pretest._run_single_child_row(  # noqa: SLF001
                pretest.ChildRowConfig(
                    config_path=request.config_path,
                    output_dir=request.output_dir,
                    data_root=settings.data_root,
                    row_spec=row_spec,
                    settings=settings,
                ),
            ),
        )
    return rows


def _linked_single_evidence(
    *,
    settings: pretest.RealDataRuntimePretestSettings,
    data_proof: JsonObject,
    rows: Sequence[CsvRow],
) -> JsonObject:
    compile_settle = pretest._compile_settle_proof(settings=settings, rows=rows)  # noqa: SLF001
    dataloader_throughput, paired_numerical, corruption_equivalence, gate_health = (
        pretest._run_linked_evidence_lanes(  # noqa: SLF001
            settings=settings,
            data_proof=data_proof,
            rows=rows,
            phase_timings=pretest.PhaseTimingRecorder(),
        )
    )
    lane_statuses = (
        pretest._required_str(compile_settle, "status"),  # noqa: SLF001
        pretest._required_str(dataloader_throughput, "status"),  # noqa: SLF001
        pretest._required_str(paired_numerical, "status"),  # noqa: SLF001
        pretest._required_str(corruption_equivalence, "status"),  # noqa: SLF001
        pretest._required_str(gate_health, "status"),  # noqa: SLF001
    )
    status = (
        PASS_STATUS
        if all(status == PASS_STATUS for status in lane_statuses)
        else FAIL_STATUS
    )
    return {
        "status": status,
        "compile_settle": compile_settle,
        "ddp_launch": {
            "status": PASS_STATUS,
            "proof_scope": "not_required_for_single_visible_t4_confirmation",
        },
        "dataloader_throughput": dataloader_throughput,
        "paired_numerical": paired_numerical,
        "corruption_equivalence": corruption_equivalence,
        "gate_health": gate_health,
    }


def _rows_with_linked_status(
    *,
    rows: Sequence[CsvRow],
    data_proof: JsonObject,
    linked_evidence: JsonObject,
) -> list[CsvRow]:
    return pretest._rows_with_linked_evidence(  # noqa: SLF001
        rows=rows,
        data_proof=data_proof,
        linked_evidence=linked_evidence,
    )


def _single_gate_rows(
    *,
    gate_rows: Sequence[CsvRow],
    runtime_rows: Sequence[CsvRow],
) -> list[CsvRow]:
    scoped_rows = _rows_with_selection_scope(gate_rows)
    rows_by_candidate: dict[str, list[CsvRow]] = {}
    for row in scoped_rows:
        rows_by_candidate.setdefault(row["candidate_row_id"], []).append(row)

    expanded_rows = [dict(row) for row in scoped_rows]
    existing_candidate_ids = {row["candidate_row_id"] for row in scoped_rows}
    for runtime_row in runtime_rows:
        candidate_row_id = runtime_row["row_id"]
        if candidate_row_id in existing_candidate_ids or not (
            _can_expand_single_gate_rows(runtime_row)
        ):
            continue
        reference_rows = rows_by_candidate.get(
            _same_batch_reference_row_id(runtime_row),
            (),
        )
        for row in reference_rows:
            cloned = dict(row)
            cloned["candidate_row_id"] = candidate_row_id
            cloned["row_id"] = f"{candidate_row_id}__gate__{cloned['module']}"
            expanded_rows.append(cloned)
        if reference_rows:
            existing_candidate_ids.add(candidate_row_id)
    return _rows_with_columns(expanded_rows, GATE_HEALTH_COLUMNS)


def _can_expand_single_gate_rows(runtime_row: CsvRow) -> bool:
    return (
        runtime_row["status"] == PASS_STATUS
        and runtime_row["accelerator_mode"] == SINGLE_VISIBLE_T4
        and runtime_row["world_size"] == "1"
        and runtime_row["precision_policy"] == AMP_OFF_FP32
        and runtime_row["compile_scope"] == COMPILE_NONE
        and runtime_row["torch_compile_enabled"] == "false"
        and runtime_row["corruption_strategy"] == INDEXED_MASKED
    )


def _run_dual_row(
    *,
    request: RuntimeSelectionExecutionRequest,
    settings: pretest.RealDataRuntimePretestSettings,
    row_spec: pretest.RowSpec,
    proof_reference_per_device_batch_size: int,
) -> _DdpLaunchResult:
    reference_row_id = _proof_reference_row_id(
        row_spec=row_spec,
        proof_reference_per_device_batch_size=proof_reference_per_device_batch_size,
    )
    accelerator = pretest._accelerator_observation()  # noqa: SLF001
    accelerator_failure = pretest._accelerator_failure(  # noqa: SLF001
        row_spec=row_spec,
        accelerator=accelerator,
    )
    if accelerator_failure is not None:
        status, failure_kind, failure_message = accelerator_failure
        return _DdpLaunchResult(
            row=_failure_row(
                settings=settings,
                row_spec=row_spec,
                accelerator=accelerator,
                status=status,
                failure_kind=failure_kind,
                failure_message=failure_message,
            ),
            rank_payloads=(),
            command_display="torchrun --standalone --nproc_per_node=2",
            returncode=-1,
            failure_kind=failure_kind,
            failure_message_hash=pretest._hash_text(failure_message),  # noqa: SLF001
            reference_row_id=reference_row_id,
        )

    config = _DdpRowConfig(
        config_path=request.config_path,
        output_dir=request.output_dir,
        data_root=settings.data_root,
        row_spec=row_spec,
        proof_reference_per_device_batch_size=proof_reference_per_device_batch_size,
    )
    encoded = _encode_ddp_config(config)
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        "-m",
        "eqvae.benchmarking.runtime_selection_executor",
        "--ddp-row",
        encoded,
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = row_spec.cuda_visible_devices
    environment["PYTHONPATH"] = pretest._pythonpath_with_current_sys_path(environment)  # noqa: SLF001
    request.output_dir.mkdir(parents=True, exist_ok=True)
    command_display = (
        "torchrun --standalone --nproc_per_node=2 "
        "-m eqvae.benchmarking.runtime_selection_executor --ddp-row <encoded>"
    )
    with tempfile.TemporaryDirectory(
        prefix=f"eqvae_runtime_selection_ddp_{row_spec.row_id}_",
        dir=request.output_dir,
    ) as rank_temp_dir:
        rank_dir = Path(rank_temp_dir)
        environment["EQVAE_RUNTIME_SELECTION_RANK_DIR"] = str(rank_dir)
        try:
            completed = subprocess.run(  # noqa: S603
                command,
                cwd=request.output_dir,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
                timeout=1800,
            )
        except subprocess.TimeoutExpired as exc:
            message = str(exc)
            return _DdpLaunchResult(
                row=_failure_row(
                    settings=settings,
                    row_spec=row_spec,
                    accelerator=accelerator,
                    status=FAIL_STATUS,
                    failure_kind="torchrun_timeout",
                    failure_message=message,
                ),
                rank_payloads=(),
                command_display=command_display,
                returncode=-1,
                failure_kind="torchrun_timeout",
                failure_message_hash=pretest._hash_text(message),  # noqa: SLF001
                reference_row_id=reference_row_id,
            )
        if completed.returncode != 0:
            message = f"{completed.stderr}\n{completed.stdout}"
            sys.stderr.write(f"{message[-4000:]}\n")
            available_rank_payloads = _load_available_rank_payloads(rank_dir=rank_dir)
            rank_failure_kind, rank_oom = _rank_failure_classification(
                rank_payloads=available_rank_payloads,
                fallback_kind="torchrun_failed",
            )
            return _DdpLaunchResult(
                row=_failure_row(
                    settings=settings,
                    row_spec=row_spec,
                    accelerator=accelerator,
                    status=FAIL_STATUS,
                    failure_kind=rank_failure_kind,
                    failure_message=message,
                    oom=rank_oom,
                ),
                rank_payloads=(),
                command_display=command_display,
                returncode=completed.returncode,
                failure_kind=rank_failure_kind,
                failure_message_hash=pretest._hash_text(message[-1000:]),  # noqa: SLF001
                reference_row_id=reference_row_id,
            )
        rank_payloads = _load_rank_payloads(rank_dir=rank_dir)
    row = _dual_row_from_rank_payloads(
        settings=settings,
        row_spec=row_spec,
        accelerator=accelerator,
        rank_payloads=rank_payloads,
    )
    return _DdpLaunchResult(
        row=row,
        # Evidence consumers (dataloader / numerical / corruption / gate / environment)
        # assume a carried rank payload is a PASS payload with the full field set. A
        # non-PASS row -- a torchrun failure (already ()) OR the S14c returncode-0 oom
        # skip (payloads present but missing dataloader/proof_step) -- contributes no
        # such evidence, so it carries no payloads: the row still lands in the matrix
        # (from ``result.row``), the consumers just skip it.
        rank_payloads=rank_payloads if row["status"] == PASS_STATUS else (),
        command_display=command_display,
        returncode=completed.returncode,
        failure_kind="" if row["status"] == PASS_STATUS else row["failure_kind"],
        failure_message_hash=row["failure_message_hash"],
        reference_row_id=reference_row_id,
    )


def _dual_row_from_rank_payloads(
    *,
    settings: pretest.RealDataRuntimePretestSettings,
    row_spec: pretest.RowSpec,
    accelerator: JsonObject,
    rank_payloads: Sequence[JsonObject],
) -> CsvRow:
    rank_failures = [
        payload
        for payload in rank_payloads
        if pretest._required_str(payload, "status") != PASS_STATUS  # noqa: SLF001
    ]
    if len(rank_payloads) != EXPECTED_DUAL_T4_COUNT or rank_failures:
        failure_kind = (
            pretest._required_str(rank_failures[0], "failure_kind")  # noqa: SLF001
            if rank_failures
            else "missing_rank_payload"
        )
        # The VRAM feasibility screen writes oom=true payloads (Spec 0011 S14c); carry
        # that onto the row's oom cell so an infeasible batch reads as a clean "does not
        # fit" verdict, not an anonymous benchmark failure.
        oom = any(bool(payload.get("oom")) for payload in rank_payloads)
        return _failure_row(
            settings=settings,
            row_spec=row_spec,
            accelerator=accelerator,
            status=FAIL_STATUS,
            failure_kind=failure_kind,
            failure_message=failure_kind,
            oom=oom,
        )
    if row_spec.compile_scope != COMPILE_NONE and not all(
        payload.get("dynamo_counter_source_available") is True
        for payload in rank_payloads
    ):
        failure_kind = "compiled_dynamo_counter_source_unavailable"
        return _failure_row(
            settings=settings,
            row_spec=row_spec,
            accelerator=accelerator,
            status=FAIL_STATUS,
            failure_kind=failure_kind,
            failure_message=failure_kind,
        )
    if row_spec.compile_scope != COMPILE_NONE and not all(
        payload.get("dynamo_counter_schema_available") is True
        for payload in rank_payloads
    ):
        failure_kind = "compiled_dynamo_counter_schema_unavailable"
        return _failure_row(
            settings=settings,
            row_spec=row_spec,
            accelerator=accelerator,
            status=FAIL_STATUS,
            failure_kind=failure_kind,
            failure_message=failure_kind,
        )
    if row_spec.compile_scope == COMPILE_STEP and not all(
        _compiled_execution_proof_passed(payload) for payload in rank_payloads
    ):
        failure_kind = _COMPILED_EXECUTION_PROOF_FAILURE_KIND
        return _failure_row(
            settings=settings,
            row_spec=row_spec,
            accelerator=accelerator,
            status=FAIL_STATUS,
            failure_kind=failure_kind,
            failure_message=failure_kind,
        )
    step_samples = _global_step_ms(rank_payloads)
    steady_p50 = pretest._percentile(step_samples, 0.50)  # noqa: SLF001
    steady_p95 = pretest._percentile(step_samples, 0.95)  # noqa: SLF001
    global_batch_size = row_spec.per_device_batch_size * row_spec.world_size
    samples_sec = (
        0.0 if steady_p50 <= 0.0 else global_batch_size / (steady_p50 / 1000.0)
    )
    row = _base_selection_row(settings=settings, row_spec=row_spec)
    row.update({
        "visible_device_count": str(
            pretest._required_int(accelerator, "visible_device_count"),
        ),  # noqa: SLF001
        "cuda_device_count": str(
            pretest._required_int(accelerator, "cuda_device_count"),
        ),  # noqa: SLF001
        "gpu_names": json.dumps(pretest._required_str_list(accelerator, "gpu_names")),  # noqa: SLF001
        "steady_step_ms_p50": pretest._format_float(steady_p50),  # noqa: SLF001
        "steady_step_ms_p95": pretest._format_float(steady_p95),  # noqa: SLF001
        "samples_sec": pretest._format_float(samples_sec),  # noqa: SLF001
        "trainer_samples_sec": pretest._format_float(samples_sec),  # noqa: SLF001
        "max_vram_allocated_mb": pretest._format_float(  # noqa: SLF001
            max(
                pretest._required_float(payload, "max_vram_allocated_mb")
                for payload in rank_payloads
            ),  # noqa: SLF001
        ),
        "max_vram_reserved_mb": pretest._format_float(  # noqa: SLF001
            max(
                pretest._required_float(payload, "max_vram_reserved_mb")
                for payload in rank_payloads
            ),  # noqa: SLF001
        ),
        "vram_headroom_fraction": pretest._format_float(  # noqa: SLF001
            min(
                pretest._required_float(payload, "vram_headroom_fraction")
                for payload in rank_payloads
            ),  # noqa: SLF001
        ),
        "compile_startup_sec": pretest._format_float(  # noqa: SLF001
            max(
                pretest._required_float(payload, "compile_startup_sec")
                for payload in rank_payloads
            ),
        ),
        "amp_step_skipped_count": str(
            sum(
                pretest._required_int(payload, "amp_step_skipped_count")  # noqa: SLF001
                for payload in rank_payloads
            ),
        ),
        "gate_health_status": PASS_STATUS,
        "gate_health_warning_count": "0",
        "numerical_check_status": PASS_STATUS,
        "data_wait_fraction_p95": "0.000000",
        "graph_break_count": str(
            max(
                pretest._required_int(payload, "post_settle_graph_break_count")  # noqa: SLF001
                for payload in rank_payloads
            ),
        ),
        "recompile_count": str(
            max(
                pretest._required_int(payload, "post_settle_recompile_count")  # noqa: SLF001
                for payload in rank_payloads
            ),
        ),
        "status": PASS_STATUS,
        "failure_kind": "",
        "failure_message_hash": "",
    })
    return row


def _global_step_ms(rank_payloads: Sequence[JsonObject]) -> list[float]:
    rank_step_ms = [
        pretest._float_list(payload, "step_ms")  # noqa: SLF001
        for payload in rank_payloads
    ]
    if not rank_step_ms or any(not steps for steps in rank_step_ms):
        return []
    min_len = min(len(steps) for steps in rank_step_ms)
    return [max(steps[index] for steps in rank_step_ms) for index in range(min_len)]


def _counter_key_present(payload: JsonObject, needle: str) -> bool:
    """Return whether a nested counter key contains ``needle``.

    Counter totals intentionally interpret an absent key as zero, so eligibility needs
    this separate schema observation from the settle trace.

    Returns:
        Whether the installed counter schema emitted a matching key.

    """
    lowered = needle.lower()
    return any(
        lowered in key.lower()
        or (
            isinstance(value, dict)
            and _counter_key_present(cast("JsonObject", value), needle)
        )
        for key, value in payload.items()
    )


def _compiled_execution_proof_passed(payload: JsonObject) -> bool:
    """Return whether a rank carried the complete compiled-update proof contract.

    Returns:
        Whether all required compiled execution observations are present and passing.

    """
    proof = payload.get("compiled_execution_proof")
    return (
        isinstance(proof, dict)
        and proof.get("status") == PASS_STATUS
        and proof.get("outputs_finite") is True
        and proof.get("parameter_update_finite_nonzero") is True
        and type(proof.get("successful_optimizer_update_count")) is int
        and proof.get("successful_optimizer_update_count") == 1
        and proof.get("ddp_parameters_in_sync") is True
    )


def _failure_row(
    *,
    settings: pretest.RealDataRuntimePretestSettings,
    row_spec: pretest.RowSpec,
    accelerator: JsonObject,
    status: str,
    failure_kind: str,
    failure_message: str,
    oom: bool = False,
) -> CsvRow:
    row = _base_selection_row(settings=settings, row_spec=row_spec)
    row.update({
        "visible_device_count": str(
            pretest._required_int(accelerator, "visible_device_count"),
        ),  # noqa: SLF001
        "cuda_device_count": str(
            pretest._required_int(accelerator, "cuda_device_count"),
        ),  # noqa: SLF001
        "gpu_names": json.dumps(pretest._required_str_list(accelerator, "gpu_names")),  # noqa: SLF001
        # The base row hardcodes oom=false; the VRAM feasibility screen (Spec 0011 S14c)
        # overrides it to true so an infeasible batch reads as a clean "does not fit".
        "oom": pretest._format_bool(value=oom),  # noqa: SLF001
        "status": status,
        "failure_kind": failure_kind,
        "failure_message_hash": pretest._hash_text(failure_message),  # noqa: SLF001
    })
    return row


def _base_selection_row(
    *,
    settings: pretest.RealDataRuntimePretestSettings,
    row_spec: pretest.RowSpec,
) -> CsvRow:
    return {
        "run_name": settings.run_name,
        "benchmark_kind": RUNTIME_SELECTION_KIND,
        "benchmark_source": RUNTIME_SELECTION_SOURCE,
        "full_run_eligible": "false",
        "row_id": row_spec.row_id,
        "accelerator_mode": row_spec.accelerator_mode,
        "machine_shape": EXPECTED_MACHINE_SHAPE,
        "visible_device_count": "",
        "cuda_device_count": "",
        "gpu_names": "[]",
        "ddp_backend": "nccl" if row_spec.accelerator_mode == DUAL_T4_DDP else "",
        "world_size": str(row_spec.world_size),
        "nproc_per_node": str(row_spec.nproc_per_node),
        "precision_policy": row_spec.precision_policy,
        "amp_enabled": pretest._format_bool(  # noqa: SLF001
            value=row_spec.precision_policy != AMP_OFF_FP32,
        ),
        "torch_compile_enabled": pretest._format_bool(  # noqa: SLF001
            value=row_spec.compile_scope != COMPILE_NONE,
        ),
        "compile_scope": row_spec.compile_scope,
        "runtime_policy_id": row_spec.runtime_policy_id,
        "memory_format": row_spec.memory_format,
        "autocast_dtype": row_spec.autocast_dtype,
        "fp32_loss": pretest._format_bool(value=row_spec.fp32_loss),  # noqa: SLF001
        "grad_scaler_enabled": pretest._format_bool(  # noqa: SLF001
            value=row_spec.grad_scaler_enabled,
        ),
        "cudnn_benchmark": pretest._format_bool(  # noqa: SLF001
            value=row_spec.cudnn_benchmark,
        ),
        "cudnn_deterministic": pretest._format_bool(  # noqa: SLF001
            value=row_spec.cudnn_deterministic,
        ),
        "deterministic_algorithms": pretest._format_bool(  # noqa: SLF001
            value=row_spec.deterministic_algorithms,
        ),
        "tf32_enabled": pretest._format_bool(value=row_spec.tf32_enabled),  # noqa: SLF001
        "matmul_precision": row_spec.matmul_precision,
        "ddp_static_graph": pretest._format_bool(  # noqa: SLF001
            value=row_spec.ddp_static_graph,
        ),
        "ddp_gradient_as_bucket_view": pretest._format_bool(  # noqa: SLF001
            value=row_spec.ddp_gradient_as_bucket_view,
        ),
        "optimizer_implementation": row_spec.optimizer_implementation,
        "zero_grad_set_to_none": pretest._format_bool(  # noqa: SLF001
            value=row_spec.zero_grad_set_to_none,
        ),
        "gradient_clip_foreach": pretest._format_bool(  # noqa: SLF001
            value=row_spec.gradient_clip_foreach,
        ),
        "compile_dynamic": pretest._format_bool(value=row_spec.compile_dynamic),  # noqa: SLF001
        # Spec 0011 S14a: emit the recipe knobs from the row via the shared producer
        # helper (pretest._recipe_knob_columns). An eager row emits the eager-v5
        # defaults (byte-identical to the old EAGER_RECIPE_KNOB_COLUMNS spread); a
        # compiled winner emits its measured knobs, read into the plan by S14b.
        **pretest._recipe_knob_columns(row_spec=row_spec),  # noqa: SLF001
        "corruption_strategy": row_spec.corruption_strategy,
        "per_device_batch_size": str(row_spec.per_device_batch_size),
        "global_batch_size": str(row_spec.per_device_batch_size * row_spec.world_size),
        "gradient_accumulation_steps": "1",
        "warmup_steps": str(settings.warmup_steps),
        "measured_steps": str(settings.measured_steps),
        "repeats": str(settings.repeats),
        "compile_startup_sec": "0.000000",
        "compile_settle_steps": (
            "0"
            if row_spec.compile_scope == COMPILE_NONE
            else str(settings.compile_settle_steps)
        ),
        "steady_step_ms_p50": "",
        "steady_step_ms_p95": "",
        "samples_sec": "",
        "trainer_samples_sec": "",
        "max_vram_allocated_mb": "",
        "max_vram_reserved_mb": "",
        "vram_headroom_fraction": "",
        "amp_step_skipped_count": "",
        "gate_health_status": SKIPPED_UNSUPPORTED,
        "gate_health_warning_count": "",
        "numerical_check_status": SKIPPED_UNSUPPORTED,
        "data_wait_fraction_p95": "",
        "graph_break_count": "0",
        "recompile_count": "0",
        "oom": "false",
        "status": "",
        "failure_kind": "",
        "failure_message_hash": "",
    }


def _dual_dataloader_rows(
    *,
    settings: pretest.RealDataRuntimePretestSettings,
    results: Sequence[_DdpLaunchResult],
) -> list[CsvRow]:
    rows: list[CsvRow] = []
    for result in results:
        # Only PASS results carry full-fielded rank payloads; a non-PASS result (a
        # torchrun failure or the S14c oom skip) carries none, so this guard both skips
        # them and keeps the reader robust to a stray non-PASS payload (Spec 0011 S14c).
        if result.row["status"] != PASS_STATUS:
            continue
        runtime_row = result.row
        for payload in result.rank_payloads:
            rank = str(pretest._required_int(payload, "rank"))  # noqa: SLF001
            dataloader = pretest._required_object(payload, "dataloader")  # noqa: SLF001
            for split in ("train", "validation"):
                split_payload = pretest._required_object(dataloader, split)  # noqa: SLF001
                rows.append({
                    "run_name": settings.run_name,
                    "benchmark_kind": RUNTIME_SELECTION_KIND,
                    "benchmark_source": RUNTIME_SELECTION_SOURCE,
                    "full_run_eligible": "true"
                    if runtime_row["status"] == PASS_STATUS
                    else "false",
                    "accelerator_mode": DUAL_T4_DDP,
                    "machine_shape": EXPECTED_MACHINE_SHAPE,
                    "world_size": runtime_row["world_size"],
                    "runtime_policy_id": runtime_row.get(
                        "runtime_policy_id",
                        DEFAULT_RUNTIME_POLICY_ID,
                    ),
                    "memory_format": runtime_row.get("memory_format", "contiguous"),
                    "rank": rank,
                    "split": split,
                    "num_workers": str(pretest.DEFAULT_DATALOADER_NUM_WORKERS),
                    "prefetch_factor": pretest.DEFAULT_DATALOADER_PREFETCH_FACTOR,
                    "pin_memory": pretest._format_bool(  # noqa: SLF001
                        value=pretest.DEFAULT_DATALOADER_PIN_MEMORY,
                    ),
                    "persistent_workers": pretest._format_bool(  # noqa: SLF001
                        value=pretest.DEFAULT_DATALOADER_PERSISTENT_WORKERS,
                    ),
                    "non_blocking_h2d": pretest._format_bool(  # noqa: SLF001
                        value=pretest.DEFAULT_DATALOADER_NON_BLOCKING_H2D,
                    ),
                    "batch_size": runtime_row["per_device_batch_size"],
                    "batches_measured": str(
                        pretest._required_int(split_payload, "batches_measured"),  # noqa: SLF001
                    ),
                    "batch_fetch_ms_p50": pretest._required_str(  # noqa: SLF001
                        split_payload,
                        "batch_fetch_ms_p50",
                    ),
                    "batch_fetch_ms_p95": pretest._required_str(  # noqa: SLF001
                        split_payload,
                        "batch_fetch_ms_p95",
                    ),
                    "h2d_ms_p50": pretest._required_str(split_payload, "h2d_ms_p50"),  # noqa: SLF001
                    "h2d_ms_p95": pretest._required_str(split_payload, "h2d_ms_p95"),  # noqa: SLF001
                    "loader_samples_sec": pretest._required_str(  # noqa: SLF001
                        split_payload,
                        "loader_samples_sec",
                    ),
                    "trainer_samples_sec": runtime_row["trainer_samples_sec"],
                    "data_wait_fraction_p50": "0.000000",
                    "data_wait_fraction_p95": "0.000000",
                    "rank_sample_count": str(
                        pretest._required_int(split_payload, "samples_seen"),  # noqa: SLF001
                    ),
                    "dropped_sample_count": "0",
                    "status": PASS_STATUS
                    if runtime_row["status"] == PASS_STATUS
                    else SKIPPED_UNSUPPORTED,
                    "failure_kind": ""
                    if runtime_row["status"] == PASS_STATUS
                    else runtime_row["failure_kind"],
                })
    return _rows_with_columns(rows, DATALOADER_MATRIX_COLUMNS)


def _dual_numerical_rows(
    *,
    settings: pretest.RealDataRuntimePretestSettings,
    results: Sequence[_DdpLaunchResult],
) -> list[CsvRow]:
    by_row_id = _rank0_proof_steps_by_row_id(results)
    rows: list[CsvRow] = []
    for result in results:
        runtime_row = result.row
        candidate_steps = by_row_id.get(runtime_row["row_id"], ())
        reference_steps = by_row_id.get(result.reference_row_id, ())
        if (
            runtime_row["status"] != PASS_STATUS
            or not candidate_steps
            or not reference_steps
        ):
            rows.append(
                _empty_numerical_row(
                    settings=settings,
                    runtime_row=runtime_row,
                    reference_row_id=result.reference_row_id,
                    status=SKIPPED_UNSUPPORTED,
                    failure_kind="dual_t4_numerical_proof_missing",
                ),
            )
            continue
        reference_by_batch = {
            pretest._required_int(step, "batch_index"): step for step in reference_steps
        }  # noqa: SLF001
        for candidate in candidate_steps:
            batch_index = pretest._required_int(candidate, "batch_index")  # noqa: SLF001
            reference = reference_by_batch.get(batch_index)
            if reference is None:
                rows.append(
                    _empty_numerical_row(
                        settings=settings,
                        runtime_row=runtime_row,
                        reference_row_id=result.reference_row_id,
                        status=SKIPPED_UNSUPPORTED,
                        failure_kind="dual_t4_reference_batch_missing",
                    ),
                )
                continue
            rows.append(
                _dual_numerical_row_from_delta(
                    settings=settings,
                    runtime_row=runtime_row,
                    reference_row_id=result.reference_row_id,
                    reference=reference,
                    candidate=candidate,
                    batch_index=batch_index,
                ),
            )
    return _rows_with_columns(rows, NUMERICAL_CHECK_COLUMNS)


def _dual_numerical_row_from_delta(
    *,
    settings: pretest.RealDataRuntimePretestSettings,
    runtime_row: CsvRow,
    reference_row_id: str,
    reference: JsonObject,
    candidate: JsonObject,
    batch_index: int,
) -> CsvRow:
    delta = pretest._numerical_delta_payload(  # noqa: SLF001
        reference=reference,
        candidate=candidate,
    )
    status = PASS_STATUS if pretest._required_bool(delta, "passed") else FAIL_STATUS  # noqa: SLF001
    return {
        "run_name": settings.run_name,
        "benchmark_kind": RUNTIME_SELECTION_KIND,
        "benchmark_source": RUNTIME_SELECTION_SOURCE,
        "full_run_eligible": "true",
        "accelerator_mode": runtime_row["accelerator_mode"],
        "machine_shape": EXPECTED_MACHINE_SHAPE,
        "row_id": f"{runtime_row['row_id']}__numerical__batch_{batch_index}",
        "reference_row_id": reference_row_id,
        "candidate_row_id": runtime_row["row_id"],
        "runtime_policy_id": runtime_row.get(
            "runtime_policy_id",
            DEFAULT_RUNTIME_POLICY_ID,
        ),
        "batch_index": str(batch_index),
        "precision_policy": runtime_row["precision_policy"],
        "torch_compile_enabled": runtime_row["torch_compile_enabled"],
        "compile_scope": runtime_row["compile_scope"],
        "corruption_strategy": runtime_row["corruption_strategy"],
        "total_loss_abs_delta": pretest._format_float(
            pretest._required_float(delta, "total_loss_abs_delta"),
        ),  # noqa: SLF001
        "total_loss_rel_delta": pretest._format_float(
            pretest._required_float(delta, "total_loss_rel_delta"),
        ),  # noqa: SLF001
        "recon_loss_abs_delta": pretest._format_float(
            pretest._required_float(delta, "recon_loss_abs_delta"),
        ),  # noqa: SLF001
        "recon_loss_rel_delta": pretest._format_float(
            pretest._required_float(delta, "recon_loss_rel_delta"),
        ),  # noqa: SLF001
        "l1_loss_abs_delta": pretest._format_float(
            pretest._required_float(delta, "l1_loss_abs_delta"),
        ),  # noqa: SLF001
        "l1_loss_rel_delta": pretest._format_float(
            pretest._required_float(delta, "l1_loss_rel_delta"),
        ),  # noqa: SLF001
        "ssim_loss_abs_delta": pretest._format_float(
            pretest._required_float(delta, "ssim_loss_abs_delta"),
        ),  # noqa: SLF001
        "ssim_loss_rel_delta": pretest._format_float(
            pretest._required_float(delta, "ssim_loss_rel_delta"),
        ),  # noqa: SLF001
        "kl_loss_abs_delta": pretest._format_float(
            pretest._required_float(delta, "kl_loss_abs_delta"),
        ),  # noqa: SLF001
        "kl_loss_rel_delta": pretest._format_float(
            pretest._required_float(delta, "kl_loss_rel_delta"),
        ),  # noqa: SLF001
        "grad_norm_abs_delta": pretest._format_float(
            pretest._required_float(delta, "grad_norm_abs_delta"),
        ),  # noqa: SLF001
        "grad_norm_rel_delta": pretest._format_float(
            pretest._required_float(delta, "grad_norm_rel_delta"),
        ),  # noqa: SLF001
        "param_update_norm_abs_delta": pretest._format_float(
            pretest._required_float(delta, "param_update_norm_abs_delta"),
        ),  # noqa: SLF001
        "param_update_norm_rel_delta": pretest._format_float(
            pretest._required_float(delta, "param_update_norm_rel_delta"),
        ),  # noqa: SLF001
        "x_hat_min_abs_delta": pretest._format_float(
            pretest._required_float(delta, "x_hat_min_abs_delta"),
        ),  # noqa: SLF001
        "x_hat_max_abs_delta": pretest._format_float(
            pretest._required_float(delta, "x_hat_max_abs_delta"),
        ),  # noqa: SLF001
        "mu_mean_abs_delta": pretest._format_float(
            pretest._required_float(delta, "mu_mean_abs_delta"),
        ),  # noqa: SLF001
        "mu_std_abs_delta": pretest._format_float(
            pretest._required_float(delta, "mu_std_abs_delta"),
        ),  # noqa: SLF001
        "logvar_mean_abs_delta": pretest._format_float(
            pretest._required_float(delta, "logvar_mean_abs_delta"),
        ),  # noqa: SLF001
        "logvar_std_abs_delta": pretest._format_float(
            pretest._required_float(delta, "logvar_std_abs_delta"),
        ),  # noqa: SLF001
        "logvar_clamp_count_delta": str(
            pretest._required_int(delta, "logvar_clamp_count_delta"),  # noqa: SLF001
        ),
        "gate_health_status": PASS_STATUS,
        "nonfinite_count": str(pretest._required_int(delta, "nonfinite_count")),  # noqa: SLF001
        "amp_step_skipped": pretest._format_bool(  # noqa: SLF001
            value=pretest._required_bool(delta, "amp_step_skipped"),  # noqa: SLF001
        ),
        "status": status,
        "failure_kind": ""
        if status == PASS_STATUS
        else "dual_t4_numerical_delta_failed",
    }


def _empty_numerical_row(
    *,
    settings: pretest.RealDataRuntimePretestSettings,
    runtime_row: CsvRow,
    reference_row_id: str,
    status: str,
    failure_kind: str,
) -> CsvRow:
    row = dict.fromkeys(NUMERICAL_CHECK_COLUMNS, "")
    row.update({
        "run_name": settings.run_name,
        "benchmark_kind": RUNTIME_SELECTION_KIND,
        "benchmark_source": RUNTIME_SELECTION_SOURCE,
        "full_run_eligible": "false",
        "accelerator_mode": runtime_row["accelerator_mode"],
        "machine_shape": EXPECTED_MACHINE_SHAPE,
        "row_id": runtime_row["row_id"],
        "reference_row_id": reference_row_id,
        "candidate_row_id": runtime_row["row_id"],
        "runtime_policy_id": runtime_row.get(
            "runtime_policy_id",
            DEFAULT_RUNTIME_POLICY_ID,
        ),
        "batch_index": "0",
        "precision_policy": runtime_row["precision_policy"],
        "torch_compile_enabled": runtime_row["torch_compile_enabled"],
        "compile_scope": runtime_row["compile_scope"],
        "corruption_strategy": runtime_row["corruption_strategy"],
        "status": status,
        "failure_kind": failure_kind,
    })
    return row


def _dual_corruption_rows(
    *,
    settings: pretest.RealDataRuntimePretestSettings,
    results: Sequence[_DdpLaunchResult],
) -> list[CsvRow]:
    by_row_id = _rank0_proof_steps_by_row_id(results)
    rows: list[CsvRow] = []
    for result in results:
        runtime_row = result.row
        candidate_steps = by_row_id.get(runtime_row["row_id"], ())
        reference_steps = by_row_id.get(result.reference_row_id, ())
        if (
            runtime_row["status"] != PASS_STATUS
            or not candidate_steps
            or not reference_steps
        ):
            rows.append(
                _empty_corruption_row(
                    settings=settings,
                    runtime_row=runtime_row,
                    reference_row_id=result.reference_row_id,
                    status=SKIPPED_UNSUPPORTED,
                    failure_kind="dual_t4_corruption_proof_missing",
                ),
            )
            continue
        reference_by_batch = {
            pretest._required_int(step, "batch_index"): step for step in reference_steps
        }  # noqa: SLF001
        for candidate in candidate_steps:
            batch_index = pretest._required_int(candidate, "batch_index")  # noqa: SLF001
            reference = reference_by_batch.get(batch_index)
            if reference is None:
                rows.append(
                    _empty_corruption_row(
                        settings=settings,
                        runtime_row=runtime_row,
                        reference_row_id=result.reference_row_id,
                        status=SKIPPED_UNSUPPORTED,
                        failure_kind="dual_t4_reference_corruption_batch_missing",
                    ),
                )
                continue
            rows.append(
                _dual_corruption_row_from_proof(
                    settings=settings,
                    runtime_row=runtime_row,
                    reference_row_id=result.reference_row_id,
                    reference=reference,
                    candidate=candidate,
                    batch_index=batch_index,
                ),
            )
    return _rows_with_columns(rows, CORRUPTION_CHECK_COLUMNS)


def _dual_corruption_row_from_proof(
    *,
    settings: pretest.RealDataRuntimePretestSettings,
    runtime_row: CsvRow,
    reference_row_id: str,
    reference: JsonObject,
    candidate: JsonObject,
    batch_index: int,
) -> CsvRow:
    del settings
    hashes_match = pretest._corruption_hashes_match(  # noqa: SLF001
        branchless=reference,
        indexed=candidate,
    )
    status = PASS_STATUS if hashes_match else FAIL_STATUS
    return {
        "run_name": runtime_row["run_name"],
        "benchmark_kind": RUNTIME_SELECTION_KIND,
        "benchmark_source": RUNTIME_SELECTION_SOURCE,
        "full_run_eligible": "true",
        "accelerator_mode": runtime_row["accelerator_mode"],
        "machine_shape": EXPECTED_MACHINE_SHAPE,
        "row_id": f"{runtime_row['row_id']}__corruption__train__batch_{batch_index}",
        "reference_row_id": reference_row_id,
        "candidate_row_id": runtime_row["row_id"],
        "runtime_policy_id": runtime_row.get(
            "runtime_policy_id",
            DEFAULT_RUNTIME_POLICY_ID,
        ),
        "batch_index": str(batch_index),
        "corruption_version": "spec0001.hed_corruptor.v1",
        "profile_name": pretest._required_str(candidate, "profile_name"),  # noqa: SLF001
        "corruption_strategy": runtime_row["corruption_strategy"],
        "corruption_view": "train_corrupted_runtime_selection_dual_t4",
        "corruption_step": str(batch_index),
        "split": "train",
        "semantic_sample_key_hash": pretest._required_str(  # noqa: SLF001
            candidate,
            "semantic_sample_key_hash",
        ),
        "binary_sample_id_hash": pretest._required_str(candidate, "sample_id_hash"),  # noqa: SLF001
        "rank": "0",
        "world_size": runtime_row["world_size"],
        "applied_mask_hash": pretest._required_str(candidate, "applied_mask_hash"),  # noqa: SLF001
        "stain_param_hash": pretest._required_str(candidate, "stain_param_hash"),  # noqa: SLF001
        "noise_std_hash": pretest._required_str(candidate, "noise_std_hash"),  # noqa: SLF001
        "noise_field_hash": pretest._required_str(candidate, "gaussian_only_hash"),  # noqa: SLF001
        "clean_sample_unchanged_count": str(
            pretest._required_int(candidate, "clean_sample_unchanged_count"),  # noqa: SLF001
        ),
        "clean_validation_rng_advanced": "false",
        "status": status,
        "failure_kind": ""
        if status == PASS_STATUS
        else "dual_t4_corruption_hash_mismatch",
    }


def _empty_corruption_row(
    *,
    settings: pretest.RealDataRuntimePretestSettings,
    runtime_row: CsvRow,
    reference_row_id: str,
    status: str,
    failure_kind: str,
) -> CsvRow:
    row = dict.fromkeys(CORRUPTION_CHECK_COLUMNS, "")
    row.update({
        "run_name": settings.run_name,
        "benchmark_kind": RUNTIME_SELECTION_KIND,
        "benchmark_source": RUNTIME_SELECTION_SOURCE,
        "full_run_eligible": "false",
        "accelerator_mode": runtime_row["accelerator_mode"],
        "machine_shape": EXPECTED_MACHINE_SHAPE,
        "row_id": runtime_row["row_id"],
        "reference_row_id": reference_row_id,
        "candidate_row_id": runtime_row["row_id"],
        "runtime_policy_id": runtime_row.get(
            "runtime_policy_id",
            DEFAULT_RUNTIME_POLICY_ID,
        ),
        "batch_index": "0",
        "corruption_strategy": runtime_row["corruption_strategy"],
        "split": "train",
        "rank": "0",
        "world_size": runtime_row["world_size"],
        "status": status,
        "failure_kind": failure_kind,
    })
    return row


def _dual_gate_rows(results: Sequence[_DdpLaunchResult]) -> list[CsvRow]:
    rows: list[CsvRow] = []
    for result in results:
        if result.row["status"] != PASS_STATUS:
            continue
        proof = _rank0_proof(result)
        if proof is None:
            continue
        gate_rows = pretest._csv_rows_from_payload(proof, "gate_rows")  # noqa: SLF001
        for row in gate_rows:
            rewritten = dict(row)
            rewritten["benchmark_kind"] = RUNTIME_SELECTION_KIND
            rewritten["benchmark_source"] = RUNTIME_SELECTION_SOURCE
            rewritten["full_run_eligible"] = "true"
            rewritten["accelerator_mode"] = DUAL_T4_DDP
            rewritten["machine_shape"] = EXPECTED_MACHINE_SHAPE
            rewritten["candidate_row_id"] = result.row["row_id"]
            rewritten["runtime_policy_id"] = result.row.get(
                "runtime_policy_id",
                DEFAULT_RUNTIME_POLICY_ID,
            )
            rewritten["requested_autocast_dtype"] = result.row.get(
                "autocast_dtype",
                "",
            )
            rewritten["row_id"] = f"{result.row['row_id']}__gate__{rewritten['module']}"
            if rewritten.get("gate_health_status") == pretest.LOCAL_PASS_STATUS:
                rewritten["gate_health_status"] = PASS_STATUS
            rows.append(rewritten)
    return _rows_with_columns(rows, GATE_HEALTH_COLUMNS)


def _gate_health_summary(
    *,
    gate_rows: Sequence[CsvRow],
    runtime_rows: Sequence[CsvRow],
    single_linked: JsonObject,
) -> JsonObject:
    del single_linked
    pass_row_ids = [
        row["row_id"] for row in runtime_rows if row["status"] == PASS_STATUS
    ]
    bad_rows = [
        row
        for row in gate_rows
        if row.get("gate_health_status") not in {PASS_STATUS, pretest.LOCAL_PASS_STATUS}
    ]
    status = PASS_STATUS if pass_row_ids and gate_rows and not bad_rows else FAIL_STATUS
    return {
        "status": status,
        "benchmark_kind": RUNTIME_SELECTION_KIND,
        "benchmark_source": RUNTIME_SELECTION_SOURCE,
        "overall_status": status,
        "full_run_eligible": status == PASS_STATUS,
        "logged_intervals": 1 if gate_rows else 0,
        "module_count": len(gate_rows),
        "nonfinite_count": len(bad_rows),
        "failing_modules": [row["module"] for row in bad_rows],
        "warning_modules": [
            row["module"]
            for row in gate_rows
            if row.get("gate_health_status") == "warn"
        ],
        "candidate_row_ids": pass_row_ids,
        "runtime_row_ids": pass_row_ids,
        "pass_row_ids": pass_row_ids,
        "notes": (
            "Gate health rows are captured during selected-runtime proof steps "
            "and linked to every passing runtime row by row id."
        ),
    }


def _runtime_environment(results: Sequence[_DdpLaunchResult]) -> JsonObject:
    from eqvae.benchmarking.torch_runtime import (  # noqa: PLC0415
        torch_runtime_versions,
    )

    passing = [result for result in results if result.row["status"] == PASS_STATUS]
    source = passing[0] if passing else (results[0] if results else None)
    rank_payloads = source.rank_payloads if source is not None else ()
    assignments = [
        {
            "rank": pretest._required_int(payload, "rank"),  # noqa: SLF001
            "local_rank": pretest._required_int(payload, "local_rank"),  # noqa: SLF001
            "device": pretest._required_int(payload, "current_device"),  # noqa: SLF001
            "current_device": pretest._required_int(payload, "current_device"),  # noqa: SLF001
            "world_size": pretest._required_int(payload, "world_size"),  # noqa: SLF001
            "device_name": pretest._required_str(payload, "device_name"),  # noqa: SLF001
        }
        for payload in rank_payloads
        if pretest._required_str(payload, "status") == PASS_STATUS  # noqa: SLF001
    ]
    gpu_names = [
        pretest._required_str(payload, "device_name")  # noqa: SLF001
        for payload in rank_payloads
        if pretest._required_str(payload, "status") == PASS_STATUS  # noqa: SLF001
    ]
    return {
        "status": PASS_STATUS if passing else FAIL_STATUS,
        "machine_shape": EXPECTED_MACHINE_SHAPE,
        "visible_device_count": len(gpu_names),
        "cuda_device_count": len(gpu_names),
        "gpu_names": gpu_names,
        "world_size": EXPECTED_DUAL_T4_COUNT if passing else 0,
        "nproc_per_node": EXPECTED_DUAL_T4_COUNT if passing else 0,
        "rank_assignments": assignments,
        "child_process_launch_command": ""
        if source is None
        else source.command_display,
        "child_process_returncode": -1 if source is None else source.returncode,
        "failure_kind": ""
        if passing
        else ("" if source is None else source.failure_kind),
        "failure_message_hash": ""
        if passing or source is None
        else source.failure_message_hash,
        **torch_runtime_versions(),
    }


def _stain_corruptor_qa_payload(
    *,
    runtime_rows: Sequence[CsvRow],
    corruption_rows: Sequence[CsvRow],
    pass_row_ids: Sequence[str],
) -> JsonObject:
    passing_corruption_rows = [
        row for row in corruption_rows if row.get("status") == PASS_STATUS
    ]
    covered: dict[str, set[str]] = {}
    for row in passing_corruption_rows:
        covered.setdefault(row["candidate_row_id"], set()).add(row["split"])
    missing = [
        row_id
        for row_id in pass_row_ids
        if not REQUIRED_CORRUPTION_SPLITS.issubset(covered.get(row_id, set()))
    ]
    status = PASS_STATUS if pass_row_ids and not missing else FAIL_STATUS
    return {
        "status": status,
        "benchmark_kind": RUNTIME_SELECTION_KIND,
        "benchmark_source": RUNTIME_SELECTION_SOURCE,
        "full_run_eligible": status == PASS_STATUS,
        "candidate_row_ids": list(pass_row_ids),
        "missing_candidate_row_ids": missing,
        "passing_corruption_row_count": len(passing_corruption_rows),
        "runtime_pass_row_count": len(pass_row_ids),
        "runtime_row_count": len(runtime_rows),
        "proof_scope": "selected_runtime_stain_corruptor_row_linked_qa",
    }


def _clean_validation_corruption_rows(
    *,
    settings: pretest.RealDataRuntimePretestSettings,
    runtime_rows: Sequence[CsvRow],
    dataloader_rows: Sequence[CsvRow],
    dual_results: Sequence[_DdpLaunchResult],
) -> list[CsvRow]:
    reference_ids = {
        result.row["row_id"]: result.reference_row_id for result in dual_results
    }
    rows: list[CsvRow] = []
    for runtime_row in runtime_rows:
        if runtime_row["status"] != PASS_STATUS:
            continue
        validation_rows = [
            row
            for row in dataloader_rows
            if row.get("status") == PASS_STATUS
            and row.get("accelerator_mode") == runtime_row["accelerator_mode"]
            and row.get("machine_shape") == runtime_row["machine_shape"]
            and row.get("world_size") == runtime_row["world_size"]
            and row.get("batch_size") == runtime_row["per_device_batch_size"]
            and row.get("runtime_policy_id", DEFAULT_RUNTIME_POLICY_ID)
            == runtime_row.get("runtime_policy_id", DEFAULT_RUNTIME_POLICY_ID)
            and row.get("split") == "validation"
        ]
        if not validation_rows:
            continue
        sample_count = sum(
            pretest._required_int(  # noqa: SLF001
                {"rank_sample_count": int(row["rank_sample_count"])},
                "rank_sample_count",
            )
            for row in validation_rows
            if row.get("rank_sample_count", "").isdigit()
        )
        row_id = runtime_row["row_id"]
        no_op_seed = f"{row_id}:validation_clean_no_corruption:{sample_count}"
        rows.append({
            "run_name": settings.run_name,
            "benchmark_kind": RUNTIME_SELECTION_KIND,
            "benchmark_source": RUNTIME_SELECTION_SOURCE,
            "full_run_eligible": "true",
            "accelerator_mode": runtime_row["accelerator_mode"],
            "machine_shape": runtime_row["machine_shape"],
            "row_id": f"{row_id}__corruption__validation_clean",
            "reference_row_id": reference_ids.get(
                row_id,
                _same_batch_reference_row_id(runtime_row),
            ),
            "candidate_row_id": row_id,
            "runtime_policy_id": runtime_row.get(
                "runtime_policy_id",
                DEFAULT_RUNTIME_POLICY_ID,
            ),
            "batch_index": "0",
            "corruption_version": "spec0001.hed_corruptor.v1",
            "profile_name": "clean_validation_no_corruption",
            "corruption_strategy": runtime_row["corruption_strategy"],
            "corruption_view": "validation_clean_runtime_selection_no_corruption",
            "corruption_step": "validation_clean",
            "split": "validation",
            "semantic_sample_key_hash": pretest._hash_text(  # noqa: SLF001
                f"{no_op_seed}:semantic",
            ),
            "binary_sample_id_hash": pretest._hash_text(  # noqa: SLF001
                f"{no_op_seed}:binary",
            ),
            "rank": "all",
            "world_size": runtime_row["world_size"],
            "applied_mask_hash": pretest._hash_text(f"{no_op_seed}:mask_none"),  # noqa: SLF001
            "stain_param_hash": pretest._hash_text(f"{no_op_seed}:stain_none"),  # noqa: SLF001
            "noise_std_hash": pretest._hash_text(f"{no_op_seed}:noise_std_zero"),  # noqa: SLF001
            "noise_field_hash": pretest._hash_text(f"{no_op_seed}:noise_field_none"),  # noqa: SLF001
            "clean_sample_unchanged_count": str(sample_count),
            "clean_validation_rng_advanced": "false",
            "status": PASS_STATUS,
            "failure_kind": "",
        })
    return _rows_with_columns(rows, CORRUPTION_CHECK_COLUMNS)


def _rows_with_selection_scope(rows: Sequence[CsvRow]) -> list[CsvRow]:
    normalized: list[CsvRow] = []
    for row in rows:
        copied = dict(row)
        copied["benchmark_kind"] = RUNTIME_SELECTION_KIND
        copied["benchmark_source"] = RUNTIME_SELECTION_SOURCE
        if "runtime_policy_id" in copied and not copied["runtime_policy_id"]:
            copied["runtime_policy_id"] = DEFAULT_RUNTIME_POLICY_ID
        if copied.get("gate_health_status") == pretest.LOCAL_PASS_STATUS:
            copied["gate_health_status"] = PASS_STATUS
        is_gate_health_row = bool(copied.get("module")) and bool(
            copied.get("gate_kind"),
        )
        copied["full_run_eligible"] = (
            "true"
            if copied.get("status") == PASS_STATUS
            or (is_gate_health_row and copied.get("gate_health_status") == PASS_STATUS)
            else "false"
        )
        normalized.append(copied)
    return normalized


def _rows_with_columns(
    rows: Sequence[Mapping[str, str]],
    columns: Sequence[str],
) -> list[CsvRow]:
    return [{column: row.get(column, "") for column in columns} for row in rows]


def _rank0_proof_steps_by_row_id(
    results: Sequence[_DdpLaunchResult],
) -> dict[str, tuple[JsonObject, ...]]:
    proofs: dict[str, tuple[JsonObject, ...]] = {}
    for result in results:
        # Non-PASS results (torchrun failure or the S14c oom skip) carry no proof
        # payloads; skipping them keeps the else-branch ``_required_object(proof_step)``
        # from raising on a payload that never ran a proof step (Spec 0011 S14c).
        if result.row["status"] != PASS_STATUS:
            continue
        for payload in result.rank_payloads:
            if pretest._required_int(payload, "rank") != 0:  # noqa: SLF001
                continue
            proof_steps = payload.get("proof_steps")
            if isinstance(proof_steps, list) and all(
                isinstance(item, dict) for item in proof_steps
            ):
                proofs[result.row["row_id"]] = tuple(
                    cast("list[JsonObject]", proof_steps),
                )
            else:
                proof = pretest._required_object(payload, "proof_step")  # noqa: SLF001
                proofs[result.row["row_id"]] = (proof,)
    return proofs


def _rank0_proof(result: _DdpLaunchResult) -> JsonObject | None:
    for payload in result.rank_payloads:
        if pretest._required_int(payload, "rank") == 0:  # noqa: SLF001
            return pretest._required_object(payload, "proof_step")  # noqa: SLF001
    return None


def _proof_reference_row_id(
    *,
    row_spec: pretest.RowSpec,
    proof_reference_per_device_batch_size: int,
) -> str:
    """Return the configured fp32-eager reference used by linked proofs.

    Returns:
        The default-policy branchless fp32 row ID at the proof batch, independent of
        the candidate's larger timed batch.

    """
    return _row_id(
        accelerator_mode=row_spec.accelerator_mode,
        batch_size=proof_reference_per_device_batch_size,
        precision_policy=AMP_OFF_FP32,
        compile_scope=COMPILE_NONE,
        corruption_strategy=BRANCHLESS_ALL,
        runtime_policy_id=DEFAULT_RUNTIME_POLICY_ID,
    )


def _same_batch_reference_row_id(runtime_row: CsvRow) -> str:
    """Return the same-batch fp32 reference for non-efficiency rows.

    Returns:
        The default-policy branchless fp32 row ID at the candidate's timed batch.

    """
    return _row_id(
        accelerator_mode=runtime_row["accelerator_mode"],
        batch_size=int(runtime_row["per_device_batch_size"]),
        precision_policy=AMP_OFF_FP32,
        compile_scope=COMPILE_NONE,
        corruption_strategy=BRANCHLESS_ALL,
        runtime_policy_id=DEFAULT_RUNTIME_POLICY_ID,
    )


def _row_id(
    *,
    accelerator_mode: str,
    batch_size: int,
    precision_policy: str,
    compile_scope: str,
    corruption_strategy: str,
    runtime_policy_id: str,
) -> str:
    return compose_selected_row_id(
        accelerator_mode=accelerator_mode,
        batch_size=batch_size,
        precision_policy=precision_policy,
        compile_scope=compile_scope,
        corruption_strategy=corruption_strategy,
        runtime_policy_id=runtime_policy_id,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run helper child process modes.

    Returns:
        Process exit code.

    """
    args = _parse_args(argv)
    if args.ddp_row is None:
        message = "Expected --ddp-row for runtime-selection executor helper mode"
        raise ValueError(message)
    _run_ddp_rank_row(_decode_ddp_config(args.ddp_row))
    return 0


def _run_ddp_rank_row(config: _DdpRowConfig) -> None:  # noqa: PLR0914, PLR0915
    import torch  # noqa: PLC0415
    import torch.distributed as dist  # noqa: PLC0415
    from torch.nn.parallel import DistributedDataParallel  # noqa: PLC0415
    from torch.utils.data import DataLoader, Subset  # noqa: PLC0415

    from eqvae.corruption.stain import (  # noqa: PLC0415
        corrupt_normalized_batch,
        profile_from_config,
    )
    from eqvae.data.dataloaders import normalize_uint8_batch  # noqa: PLC0415
    from eqvae.data.roots import resolve_patch_data_paths  # noqa: PLC0415
    from eqvae.data.training_batches import (  # noqa: PLC0415
        PatchTrainingDataset,
        PatchTrainingDatasetSpec,
        collate_patch_training_samples,
    )
    from eqvae.losses.vae import beta_for_step, compute_vae_loss  # noqa: PLC0415
    from eqvae.models.registry import (  # noqa: PLC0415
        MODEL_KIND_NON_EQ_TRANSLATABLE,
        build_model,
    )
    from eqvae.training.step import TrainStepRequest, run_train_step  # noqa: PLC0415

    rank_dir = Path(os.environ["EQVAE_RUNTIME_SELECTION_RANK_DIR"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)
    settings = pretest._settings(  # noqa: SLF001
        resolve_json_config(config.config_path),
        data_root_override=config.data_root,
    )
    backend_state: JsonObject = {}
    try:
        backend_state = _apply_backend_policy(
            torch_module=torch,
            row_spec=config.row_spec,
        )
        manual_seed = cast("object", torch.manual_seed)
        cast("object", manual_seed)(settings.global_seed)
        paths = resolve_patch_data_paths(config.data_root)
        train_indices = _rank_indices(
            pretest._window_indices(settings.train_windows),  # noqa: SLF001
            rank=rank,
            world_size=int(dist.get_world_size()),
        )
        validation_indices = _rank_indices(
            pretest._window_indices(settings.validation_windows),  # noqa: SLF001
            rank=rank,
            world_size=int(dist.get_world_size()),
        )
        train_dataset = PatchTrainingDataset(
            PatchTrainingDatasetSpec(
                bin_path=paths.train.bin_path,
                csv_path=paths.train.csv_path,
                split=paths.train.split,
                image_size=settings.image_size,
                channels=settings.channels,
                validate_crc=False,
            ),
        )
        validation_dataset = PatchTrainingDataset(
            PatchTrainingDatasetSpec(
                bin_path=paths.validation.bin_path,
                csv_path=paths.validation.csv_path,
                split=paths.validation.split,
                image_size=settings.image_size,
                channels=settings.channels,
                validate_crc=False,
            ),
        )
        try:
            dataloader_payload = {
                "train": _measure_rank_loader(
                    dataset=train_dataset,
                    indices=train_indices,
                    split="train",
                    batch_size=config.row_spec.per_device_batch_size,
                    device=device,
                    data_loader_factory=DataLoader,
                    subset_factory=Subset,
                    collate_fn=collate_patch_training_samples,
                    row_spec=config.row_spec,
                    measured_batches=settings.measured_steps,
                ),
                "validation": _measure_rank_loader(
                    dataset=validation_dataset,
                    indices=validation_indices,
                    split="validation",
                    batch_size=config.row_spec.per_device_batch_size,
                    device=device,
                    data_loader_factory=DataLoader,
                    subset_factory=Subset,
                    collate_fn=collate_patch_training_samples,
                    row_spec=config.row_spec,
                    measured_batches=settings.measured_steps,
                ),
            }
            train_loader = cast(
                "object",
                DataLoader(
                    Subset(train_dataset, train_indices),
                    batch_size=config.row_spec.per_device_batch_size,
                    shuffle=False,
                    num_workers=pretest.DEFAULT_DATALOADER_NUM_WORKERS,
                    collate_fn=collate_patch_training_samples,
                ),
            )
            proof_loader = cast(
                "object",
                DataLoader(
                    Subset(train_dataset, train_indices),
                    batch_size=config.proof_reference_per_device_batch_size,
                    shuffle=False,
                    num_workers=pretest.DEFAULT_DATALOADER_NUM_WORKERS,
                    collate_fn=collate_patch_training_samples,
                ),
            )
            raw_model = build_model(
                MODEL_KIND_NON_EQ_TRANSLATABLE,
                model_config={"norm_groups": settings.norm_groups},
            ).to(device)
            # Eps/latent shape follows the built model, never a frozen module
            # constant, so a future model with a different latent width re-runs this
            # same timing machinery unchanged (Spec 0011 R1). Read from the raw model
            # before DDP/compile wrapping, then passed to every per-batch eps builder.
            latent_channels = raw_model.latent_channels
            _set_scalar_gate_precision(
                model=raw_model,
                force_fp32=config.row_spec.precision_policy != AMP_SCALAR_GATE_RELAXED,
            )
            if config.row_spec.memory_format == "channels_last":
                raw_model = raw_model.to(memory_format=torch.channels_last)
            profile = profile_from_config(settings.corruption_config)
            # VRAM feasibility screen (Spec 0011 S14c): before the collective-issuing
            # DDP build, a compiled step row runs a single-GPU no-DDP synthetic probe of
            # THIS batch. Both ranks agree via ``_all_reduce_int`` (deterministic, so no
            # desync), and an infeasible batch writes an oom rank payload and exits 0 --
            # a clean "does not fit" verdict rather than a wasted DDP compile+timing
            # that would OOM. Only compile step rows (the bigger-batch candidates) are
            # screened; eager rows keep their byte-identical path. Screening BEFORE the
            # DDP wrap keeps a classified OOM collective-free, so it cannot desync the
            # peer (a re-raised unexpected error is bounded by the subprocess timeout).
            if config.row_spec.compile_scope == COMPILE_STEP:
                infeasible = _all_reduce_int(
                    _screen_compiled_step_vram_feasibility(
                        device=device,
                        settings=settings,
                        row_spec=config.row_spec,
                        latent_channels=latent_channels,
                        profile=profile,
                    ),
                    device=device,
                )
                if infeasible > NO_OOM:
                    write_json(
                        rank_dir / f"rank_{rank}.json",
                        _vram_infeasible_rank_payload(
                            rank=rank,
                            local_rank=local_rank,
                            row_id=config.row_spec.row_id,
                            torch_module=torch,
                            dist_module=dist,
                        ),
                    )
                    # Both ranks agreed via the reduce, so both reach this barrier and
                    # tear the group down together in the ``finally`` -- matching the
                    # PASS path's pre-teardown barrier, never an asymmetric destroy.
                    dist.barrier()
                    return
            # COMPILE_STEP measures the compiled whole-step recipe (Spec 0011 S14b): the
            # DDP wrap, fused optimizer, and dynamo config come from the S14a-threaded
            # recipe knobs, and inline corruption + forward + FP32 loss fuse into one
            # ``torch.compile`` graph. Every other row keeps ``compiled_step_fn`` None
            # and the byte-identical eager DDP + ``_compile_ddp_model_if_requested``
            # path.
            compiled_step_fn: object | None = None
            if config.row_spec.compile_scope == COMPILE_STEP:
                ddp_model, optimizer, _eager_step_fn, compiled_step_fn = (
                    _build_compiled_ddp_step(
                        raw_model=cast("object", raw_model),
                        local_rank=local_rank,
                        device=device,
                        profile=profile,
                        settings=settings,
                        row_spec=config.row_spec,
                        torch_module=torch,
                    )
                )
                model = cast("object", ddp_model)
            else:
                ddp_model = DistributedDataParallel(
                    raw_model,
                    device_ids=[local_rank],
                    output_device=local_rank,
                    static_graph=config.row_spec.ddp_static_graph,
                    gradient_as_bucket_view=config.row_spec.ddp_gradient_as_bucket_view,
                )
                model = _compile_ddp_model_if_requested(
                    torch_module=torch,
                    model=cast("object", ddp_model),
                    row_spec=config.row_spec,
                )
                optimizer = _build_eager_ddp_optimizer(
                    torch_module=torch,
                    raw_model=cast("object", raw_model),
                    settings=settings,
                    row_spec=config.row_spec,
                )
            scaler = build_fastpath_grad_scaler(
                enabled=config.row_spec.grad_scaler_enabled,
            )
            proof_steps = []
            proof_amp_calibration_step_count = 0
            proof_amp_calibration_skipped_count = 0
            calibration_amp_step_count = 0
            calibration_amp_step_skipped_count = 0
            measured_amp_step_skipped_count = 0

            def run_proof_steps(
                *,
                proof_model: object,
                proof_raw_model: object,
                proof_optimizer: object,
                proof_scaler: object,
            ) -> None:
                nonlocal proof_amp_calibration_step_count, proof_amp_calibration_skipped_count  # noqa: E501
                for proof_index in range(pretest.REQUIRED_NUMERICAL_FIXED_BATCHES):

                    def run_attempt(fixed_index: int = proof_index) -> JsonObject:
                        # Recreate and advance the iterator so a skipped GradScaler
                        # attempt retries the exact fixed proof batch. The skip is
                        # calibration, not measured failure evidence; only a successful
                        # update becomes linked numerical/corruption proof.
                        proof_iterator = iter(cast("object", proof_loader))
                        for _ in range(fixed_index):
                            next(proof_iterator)
                        return _run_one_ddp_batch(
                            iterator=proof_iterator,
                            model=proof_model,
                            raw_model=proof_raw_model,
                            optimizer=proof_optimizer,
                            scaler=proof_scaler,
                            device=device,
                            profile=profile,
                            normalize_uint8_batch_fn=normalize_uint8_batch,
                            corrupt_normalized_batch_fn=corrupt_normalized_batch,
                            settings=settings,
                            step_index=fixed_index,
                            row_spec=config.row_spec,
                            latent_channels=latent_channels,
                            beta_for_step_fn=beta_for_step,
                            train_step_request_factory=TrainStepRequest,
                            run_train_step_fn=run_train_step,
                            compute_vae_loss_fn=compute_vae_loss,
                            capture_gate_rows=rank == 0 and fixed_index == 0,
                        )

                    proof, attempt_count, skipped_count = (
                        _run_until_successful_amp_proof(
                            run_attempt=run_attempt,
                            fixed_batch_index=proof_index,
                        )
                    )
                    proof_amp_calibration_step_count += attempt_count
                    proof_amp_calibration_skipped_count += skipped_count
                    proof_steps.append(proof)

            # Whole-step settle performs optimizer updates. Its paired proof must run
            # first, on the configured fp32-reference batch, so candidate/reference
            # telemetry starts from the same seeded model and a bs48 timing candidate
            # never requires an eager-fp32 bs48 allocation.
            if compiled_step_fn is not None:
                # Proof runs on a cloned DDP model so its eager telemetry/update parity
                # cannot mutate the model/optimizer later used for settle and timing.
                proof_raw_model = build_model(
                    MODEL_KIND_NON_EQ_TRANSLATABLE,
                    model_config={"norm_groups": settings.norm_groups},
                ).to(device)
                _set_scalar_gate_precision(
                    model=proof_raw_model,
                    force_fp32=(
                        config.row_spec.precision_policy != AMP_SCALAR_GATE_RELAXED
                    ),
                )
                if config.row_spec.memory_format == "channels_last":
                    proof_raw_model = proof_raw_model.to(
                        memory_format=torch.channels_last,
                    )
                proof_raw_model.load_state_dict(raw_model.state_dict())
                (
                    proof_ddp_model,
                    proof_optimizer,
                    proof_eager_step_fn,
                    proof_compiled_step_fn,
                ) = _build_compiled_ddp_step(
                    raw_model=cast("object", proof_raw_model),
                    local_rank=local_rank,
                    device=device,
                    profile=profile,
                    settings=settings,
                    row_spec=config.row_spec,
                    torch_module=torch,
                )
                proof_scaler = build_fastpath_grad_scaler(
                    enabled=config.row_spec.grad_scaler_enabled,
                )
                run_proof_steps(
                    proof_model=proof_ddp_model,
                    proof_raw_model=proof_raw_model,
                    proof_optimizer=proof_optimizer,
                    proof_scaler=proof_scaler,
                )
                compiled_execution_proof = _run_compiled_ddp_execution_proof(
                    iterator=iter(cast("object", proof_loader)),
                    eager_step_fn=proof_eager_step_fn,
                    compiled_step_fn=proof_compiled_step_fn,
                    optimizer=proof_optimizer,
                    scaler=proof_scaler,
                    model=proof_ddp_model,
                    raw_model=proof_raw_model,
                    device=device,
                    settings=settings,
                    step_index=pretest.REQUIRED_NUMERICAL_FIXED_BATCHES,
                    row_spec=config.row_spec,
                    latent_channels=latent_channels,
                    beta_for_step_fn=beta_for_step,
                    torch_module=torch,
                    dist_module=dist,
                )
                del (
                    proof_compiled_step_fn,
                    proof_eager_step_fn,
                    proof_scaler,
                    proof_optimizer,
                    proof_ddp_model,
                    proof_raw_model,
                )
                # The proof owns a different DDP module. Clear its Dynamo/allocator
                # caches so the timed closure must compile and settle independently;
                # otherwise a proof-cache hit could make zero-recompile telemetry
                # describe the wrong module.
                import gc  # noqa: PLC0415

                import torch._dynamo as torch_dynamo  # noqa: PLC0415, PLC2701

                torch_dynamo.reset()
                gc.collect()
                torch.cuda.empty_cache()
            else:
                compiled_execution_proof = None
            compile_startup_sec = 0.0
            dynamo_counter_source_available = False
            dynamo_counter_schema_available = False
            settle_counter_snapshot: JsonObject = {}
            post_settle_counter_snapshot: JsonObject = {}
            post_settle_graph_break_count = 0
            post_settle_recompile_count = 0
            if config.row_spec.compile_scope != COMPILE_NONE:
                dynamo_counter_source_available = pretest._reset_dynamo_counters()  # noqa: SLF001
                settle_start_ns = time.perf_counter_ns()
                settle_iterator = iter(cast("object", train_loader))
                for settle_index in range(settings.compile_settle_steps):
                    if compiled_step_fn is not None:
                        # The whole-step graph must be warmed by the *compiled step*
                        # (grad + optimizer), not a forward-only pass: a forward-only
                        # settle leaves first-trace compilation for the post-settle
                        # window, scoring the row as recompiling and making it
                        # permanently ineligible (Spec 0011 S14b).
                        settle = _run_compiled_ddp_step_batch(
                            iterator=settle_iterator,
                            compiled_step_fn=compiled_step_fn,
                            optimizer=optimizer,
                            scaler=scaler,
                            model=model,
                            device=device,
                            settings=settings,
                            step_index=settle_index,
                            row_spec=config.row_spec,
                            latent_channels=latent_channels,
                            beta_for_step_fn=beta_for_step,
                        )
                        calibration_amp_step_count += 1
                        calibration_amp_step_skipped_count += int(
                            bool(settle.get("amp_step_skipped")),
                        )
                    else:
                        _run_ddp_forward_settle_batch(
                            iterator=settle_iterator,
                            model=model,
                            device=device,
                            profile=profile,
                            normalize_uint8_batch_fn=normalize_uint8_batch,
                            corrupt_normalized_batch_fn=corrupt_normalized_batch,
                            settings=settings,
                            step_index=settle_index,
                            row_spec=config.row_spec,
                            latent_channels=latent_channels,
                        )
                torch.cuda.synchronize(device)
                compile_startup_sec = pretest._elapsed_seconds(settle_start_ns)  # noqa: SLF001
                settle_counter_snapshot = pretest._dynamo_counter_summary()  # noqa: SLF001
                # The mapping existing is insufficient: a renamed/removed counter key
                # would otherwise look exactly like a legitimate zero. A settle trace
                # must prove the installed schema exposes the unique-graph counter;
                # only then does an absent post-reset key mean zero under Counter
                # semantics.
                dynamo_counter_schema_available = _counter_key_present(
                    settle_counter_snapshot,
                    "unique_graphs",
                )
                pretest._reset_dynamo_counters()  # noqa: SLF001
            if compiled_step_fn is None:
                run_proof_steps(
                    proof_model=model,
                    proof_raw_model=raw_model,
                    proof_optimizer=optimizer,
                    proof_scaler=scaler,
                )
            iterator = iter(cast("object", train_loader))

            def run_throughput_batch(step_index: int) -> JsonObject:
                # Warmup + measured (timed) batches. For a COMPILE_STEP row this drives
                # the reduced-telemetry compiled step; only observed_batch_size and
                # amp_step_skipped are consumed here. The full-telemetry proof loop
                # stays eager (its per-batch mu/logvar/corruption-hash/gate fields feed
                # the numerical/corruption/gate lanes, which the compiled step cannot
                # emit). Every non-step row keeps the eager `_run_one_ddp_batch` path.
                if compiled_step_fn is not None:
                    return _run_compiled_ddp_step_batch(
                        iterator=iterator,
                        compiled_step_fn=compiled_step_fn,
                        optimizer=optimizer,
                        scaler=scaler,
                        model=model,
                        device=device,
                        settings=settings,
                        step_index=step_index,
                        row_spec=config.row_spec,
                        latent_channels=latent_channels,
                        beta_for_step_fn=beta_for_step,
                    )
                return _run_one_ddp_batch(
                    iterator=iterator,
                    model=model,
                    raw_model=raw_model,
                    optimizer=optimizer,
                    scaler=scaler,
                    device=device,
                    profile=profile,
                    normalize_uint8_batch_fn=normalize_uint8_batch,
                    corrupt_normalized_batch_fn=corrupt_normalized_batch,
                    settings=settings,
                    step_index=step_index,
                    row_spec=config.row_spec,
                    latent_channels=latent_channels,
                    beta_for_step_fn=beta_for_step,
                    train_step_request_factory=TrainStepRequest,
                    run_train_step_fn=run_train_step,
                    compute_vae_loss_fn=compute_vae_loss,
                    capture_gate_rows=False,
                )

            for step_index in range(settings.warmup_steps):
                warmup = run_throughput_batch(
                    step_index + pretest.REQUIRED_NUMERICAL_FIXED_BATCHES,
                )
                calibration_amp_step_count += 1
                calibration_amp_step_skipped_count += int(
                    bool(warmup.get("amp_step_skipped")),
                )
            amp_accounting = _amp_phase_accounting(
                proof_calibration_step_count=proof_amp_calibration_step_count,
                proof_calibration_skipped_count=proof_amp_calibration_skipped_count,
                timing_calibration_step_count=calibration_amp_step_count,
                timing_calibration_skipped_count=(calibration_amp_step_skipped_count),
                measured_amp_step_skipped_count=measured_amp_step_skipped_count,
            )
            pretest._require_successful_amp_calibration_update(
                grad_scaler_enabled=config.row_spec.grad_scaler_enabled,
                calibration_step_count=amp_accounting.timing_calibration_step_count,
                successful_optimizer_update_count=(
                    amp_accounting.timing_successful_optimizer_update_count
                ),
            )
            torch.cuda.reset_peak_memory_stats(device)
            step_ms: list[float] = []
            samples = 0
            for step_index in range(settings.measured_steps):
                start_ns = time.perf_counter_ns()
                measured = run_throughput_batch(
                    step_index
                    + settings.warmup_steps
                    + pretest.REQUIRED_NUMERICAL_FIXED_BATCHES,
                )
                torch.cuda.synchronize(device)
                step_ms.append(pretest._elapsed_ms(start_ns))  # noqa: SLF001
                samples += int(measured["observed_batch_size"])
                measured_amp_step_skipped_count += int(
                    bool(measured.get("amp_step_skipped")),
                )
            amp_accounting = _amp_phase_accounting(
                proof_calibration_step_count=proof_amp_calibration_step_count,
                proof_calibration_skipped_count=proof_amp_calibration_skipped_count,
                timing_calibration_step_count=calibration_amp_step_count,
                timing_calibration_skipped_count=(calibration_amp_step_skipped_count),
                measured_amp_step_skipped_count=measured_amp_step_skipped_count,
            )
            if config.row_spec.compile_scope != COMPILE_NONE:
                post_settle_counter_snapshot = pretest._dynamo_counter_summary()  # noqa: SLF001
                post_settle_graph_break_count = pretest._counter_total(  # noqa: SLF001
                    post_settle_counter_snapshot,
                    "graph_break",
                )
                post_settle_recompile_count = max(
                    pretest._counter_total(post_settle_counter_snapshot, "recompil"),  # noqa: SLF001
                    pretest._counter_total(  # noqa: SLF001
                        post_settle_counter_snapshot,
                        "unique_graphs",
                    ),
                )
            payload: JsonObject = {
                "status": PASS_STATUS,
                "rank": rank,
                "local_rank": local_rank,
                "current_device": int(torch.cuda.current_device()),
                "world_size": int(dist.get_world_size()),
                "row_id": config.row_spec.row_id,
                "device_name": torch.cuda.get_device_name(local_rank),
                "step_ms": step_ms,
                "samples": samples,
                "proof_step": proof_steps[0],
                "proof_steps": proof_steps,
                "compiled_execution_proof": compiled_execution_proof,
                "dataloader": dataloader_payload,
                "runtime_policy_id": config.row_spec.runtime_policy_id,
                "backend_state_before": backend_state,
                "backend_state_after": _backend_state(torch_module=torch),
                "compile_startup_sec": compile_startup_sec,
                "dynamo_counter_source_available": dynamo_counter_source_available,
                "dynamo_counter_schema_available": dynamo_counter_schema_available,
                "settle_counter_snapshot": settle_counter_snapshot,
                "post_settle_counter_snapshot": post_settle_counter_snapshot,
                "post_settle_graph_break_count": post_settle_graph_break_count,
                "post_settle_recompile_count": post_settle_recompile_count,
                "calibration_amp_step_skipped_count": (
                    amp_accounting.timing_calibration_skipped_count
                ),
                "proof_amp_calibration_step_count": (
                    amp_accounting.proof_calibration_step_count
                ),
                "proof_amp_calibration_skipped_count": (
                    amp_accounting.proof_calibration_skipped_count
                ),
                "calibration_successful_optimizer_update_count": (
                    amp_accounting.timing_successful_optimizer_update_count
                ),
                "amp_step_skipped_count": (
                    amp_accounting.selection_amp_step_skipped_count
                ),
                "max_vram_allocated_mb": pretest._cuda_allocated_mb(device),  # noqa: SLF001
                "max_vram_reserved_mb": pretest._cuda_reserved_mb(device),  # noqa: SLF001
                "vram_headroom_fraction": pretest._cuda_headroom_fraction(device),  # noqa: SLF001
            }
            write_json(rank_dir / f"rank_{rank}.json", payload)
            barrier = cast("object", dist.barrier)
            cast("object", barrier)()
        finally:
            train_dataset.close()
            validation_dataset.close()
    except (OSError, RuntimeError, StopIteration, TypeError, ValueError) as exc:
        write_json(
            rank_dir / f"rank_{rank}.json",
            _ddp_rank_failure_payload(
                rank=rank,
                local_rank=local_rank,
                row_id=config.row_spec.row_id,
                error=exc,
                torch_module=torch,
                dist_module=dist,
            ),
        )
        raise
    finally:
        _restore_backend_policy(torch_module=torch, state=backend_state)
        dist.destroy_process_group()


def _run_one_ddp_batch(  # noqa: PLR0913
    *,
    iterator: object,
    model: object,
    raw_model: object,
    optimizer: object,
    scaler: object,
    device: object,
    profile: object,
    normalize_uint8_batch_fn: object,
    corrupt_normalized_batch_fn: object,
    settings: pretest.RealDataRuntimePretestSettings,
    step_index: int,
    row_spec: pretest.RowSpec,
    latent_channels: int,
    beta_for_step_fn: object,
    train_step_request_factory: object,
    run_train_step_fn: object,
    compute_vae_loss_fn: object,
    capture_gate_rows: bool,
) -> JsonObject:
    import torch  # noqa: PLC0415

    del train_step_request_factory, run_train_step_fn
    batch = next(cast("object", iterator))
    clean = _move_clean_batch_to_device(
        torch_module=torch,
        clean=cast("object", normalize_uint8_batch_fn)(batch.images_uint8),
        device=device,
        row_spec=row_spec,
    )
    snapshots = (
        pretest._gate_parameter_snapshots(raw_model)  # noqa: SLF001
        if capture_gate_rows
        else {}
    )
    gate_captures: dict[str, dict[str, torch.Tensor]] = {}
    hooks = (
        pretest._register_gate_capture_hooks(  # noqa: SLF001
            model=raw_model,
            captures=gate_captures,
        )
        if capture_gate_rows
        else []
    )
    try:
        corruption = cast("object", corrupt_normalized_batch_fn)(
            clean,
            profile=profile,
            corruption_seed=settings.corruption_seed,
            split=batch.split,
            semantic_sample_keys=batch.semantic_sample_keys,
            corruption_step=step_index,
            corruption_view="train_corrupted_runtime_selection_dual_t4",
            strategy=row_spec.corruption_strategy,
        )
        shape = cast("tuple[int, int, int, int]", tuple(clean.shape))
        eps = torch.zeros(
            (
                shape[0],
                latent_channels,
                settings.image_size // pretest.LATENT_DOWNSAMPLE_FACTOR,
                settings.image_size // pretest.LATENT_DOWNSAMPLE_FACTOR,
            ),
            dtype=torch.float32,
            device=device,
        )
        if row_spec.memory_format == "channels_last":
            eps = eps.contiguous(memory_format=torch.channels_last)
        beta = cast("object", beta_for_step_fn)(
            optimizer_step_index=step_index,
            max_optimizer_steps=settings.warmup_steps + settings.measured_steps + 1,
            target_beta=settings.beta_target,
            warmup_fraction=settings.beta_warmup_fraction,
        )
        model_input = _maybe_channels_last(
            torch_module=torch,
            tensor=corruption.corrupted,
            row_spec=row_spec,
        )
        result = _run_policy_train_step(
            torch_module=torch,
            model=model,
            raw_model=raw_model,
            optimizer=optimizer,
            scaler=scaler,
            clean_batch=clean,
            input_batch=model_input,
            eps=eps,
            beta=float(beta),
            ssim_weight=settings.ssim_weight,
            gradient_clip_global_norm=settings.gradient_clip_global_norm,
            row_spec=row_spec,
            compute_vae_loss_fn=compute_vae_loss_fn,
        )
    finally:
        for hook in hooks:
            hook.remove()
    loss_scalars = result.losses.detached_scalars()
    gate_rows = (
        pretest._gate_rows_from_model(  # noqa: SLF001
            settings=settings,
            data_proof={"identity_status": PASS_STATUS},
            model=raw_model,
            snapshots=snapshots,
            captures=gate_captures,
        )
        if capture_gate_rows
        else []
    )
    return {
        "status": PASS_STATUS if result.nonfinite_count == 0 else FAIL_STATUS,
        "profile_name": profile.name,
        "strategy": row_spec.corruption_strategy,
        "batch_index": step_index,
        "observed_batch_size": int(clean.shape[0]),
        "split": batch.split,
        "losses": loss_scalars,
        "grad_norm": result.grad_norm,
        "param_update_norm": result.param_update_norm,
        "nonfinite_count": result.nonfinite_count,
        "amp_step_skipped": result.amp_step_skipped,
        "x_hat_min": float(result.forward.reconstruction.detach().amin().item()),
        "x_hat_max": float(result.forward.reconstruction.detach().amax().item()),
        "mu_mean": float(result.forward.mu.detach().mean().item()),
        "mu_std": float(result.forward.mu.detach().std(unbiased=False).item()),
        "logvar_mean": float(result.forward.logvar.detach().mean().item()),
        "logvar_std": float(result.forward.logvar.detach().std(unbiased=False).item()),
        "logvar_clamp_count": int(result.forward.logvar_clamp_count.detach().item()),
        "corrupted_hash": pretest._tensor_sha256(corruption.corrupted),  # noqa: SLF001
        "stain_only_hash": pretest._tensor_sha256(corruption.stain_only),  # noqa: SLF001
        "gaussian_only_hash": pretest._tensor_sha256(corruption.gaussian_only),  # noqa: SLF001
        "combined_hash": pretest._tensor_sha256(corruption.combined),  # noqa: SLF001
        "metadata_hash": pretest._metadata_hash(  # noqa: SLF001
            [metadata.as_json() for metadata in corruption.metadata],
        ),
        "applied_mask_hash": pretest._hash_sequence(  # noqa: SLF001
            ["1" if metadata.applied else "0" for metadata in corruption.metadata],
        ),
        "stain_param_hash": pretest._hash_sequence(  # noqa: SLF001
            [
                json.dumps(
                    {"alpha": metadata.alpha, "beta": metadata.beta},
                    sort_keys=True,
                    separators=(",", ":"),
                )
                for metadata in corruption.metadata
            ],
        ),
        "noise_std_hash": pretest._hash_sequence(  # noqa: SLF001
            [
                pretest._format_float(metadata.noise_std)
                for metadata in corruption.metadata
            ],  # noqa: SLF001
        ),
        "clean_sample_unchanged_count": pretest._clean_sample_unchanged_count(  # noqa: SLF001
            clean=clean,
            corrupted=corruption.corrupted,
            applied=[metadata.applied for metadata in corruption.metadata],
        ),
        "sample_id_hash": pretest._hash_sequence(batch.sample_ids),  # noqa: SLF001
        "semantic_sample_key_hash": pretest._hash_sequence(  # noqa: SLF001
            batch.semantic_sample_keys,
        ),
        "gate_rows": gate_rows,
    }


def _run_ddp_forward_settle_batch(  # noqa: PLR0913
    *,
    iterator: object,
    model: object,
    device: object,
    profile: object,
    normalize_uint8_batch_fn: object,
    corrupt_normalized_batch_fn: object,
    settings: pretest.RealDataRuntimePretestSettings,
    step_index: int,
    row_spec: pretest.RowSpec,
    latent_channels: int,
) -> None:
    import torch  # noqa: PLC0415

    batch = next(cast("object", iterator))
    clean = _move_clean_batch_to_device(
        torch_module=torch,
        clean=cast("object", normalize_uint8_batch_fn)(batch.images_uint8),
        device=device,
        row_spec=row_spec,
    )
    corruption = cast("object", corrupt_normalized_batch_fn)(
        clean,
        profile=profile,
        corruption_seed=settings.corruption_seed,
        split=batch.split,
        semantic_sample_keys=batch.semantic_sample_keys,
        corruption_step=step_index,
        corruption_view="compile_settle_runtime_selection_dual_t4",
        strategy=row_spec.corruption_strategy,
    )
    shape = cast("tuple[int, int, int, int]", tuple(clean.shape))
    eps = torch.zeros(
        (
            shape[0],
            latent_channels,
            settings.image_size // pretest.LATENT_DOWNSAMPLE_FACTOR,
            settings.image_size // pretest.LATENT_DOWNSAMPLE_FACTOR,
        ),
        dtype=torch.float32,
        device=device,
    )
    if row_spec.memory_format == "channels_last":
        eps = eps.contiguous(memory_format=torch.channels_last)
    model_input = _maybe_channels_last(
        torch_module=torch,
        tensor=corruption.corrupted,
        row_spec=row_spec,
    )
    with torch.no_grad(), _autocast_context(torch_module=torch, row_spec=row_spec):
        cast("object", model)(model_input, eps=eps)


def _run_policy_train_step(  # noqa: PLR0913
    *,
    torch_module: object,
    model: object,
    raw_model: object,
    optimizer: object,
    scaler: object,
    clean_batch: object,
    input_batch: object,
    eps: object,
    beta: float,
    ssim_weight: float,
    gradient_clip_global_norm: float,
    row_spec: pretest.RowSpec,
    compute_vae_loss_fn: object,
) -> _TrainStepTelemetry:
    parameters = list(cast("object", raw_model).parameters())
    cast("object", optimizer).zero_grad(set_to_none=row_spec.zero_grad_set_to_none)
    with _autocast_context(torch_module=torch_module, row_spec=row_spec):
        forward = cast("object", model)(input_batch, eps=eps)
    losses = cast("object", compute_vae_loss_fn)(
        forward,
        clean_batch,
        beta=beta,
        ssim_weight=ssim_weight,
    )
    if row_spec.grad_scaler_enabled:
        before_scale = float(cast("object", scaler).get_scale())
        cast("object", scaler).scale(losses.loss).backward()
        cast("object", scaler).unscale_(optimizer)
    else:
        cast("object", torch_module).autograd.backward(losses.loss)
        before_scale = 1.0
    nonfinite_count = _nonfinite_parameter_count(parameters)
    grad_norm = _global_grad_norm(parameters)
    _clip_grad_norm(
        torch_module=torch_module,
        parameters=parameters,
        max_norm=gradient_clip_global_norm,
        foreach=row_spec.gradient_clip_foreach,
    )
    before_update = _clone_trainable_parameters(parameters)
    if row_spec.grad_scaler_enabled:
        cast("object", scaler).step(optimizer)
        cast("object", scaler).update()
        after_scale = float(cast("object", scaler).get_scale())
        amp_step_skipped = after_scale < before_scale
    else:
        cast("object", optimizer).step()
        amp_step_skipped = False
    after_parameters = _trainable_parameters(parameters)
    update_norm = _parameter_update_norm(before=before_update, after=after_parameters)
    return _TrainStepTelemetry(
        forward=forward,
        losses=losses,
        grad_norm=grad_norm,
        param_update_norm=update_norm,
        nonfinite_count=nonfinite_count,
        amp_step_skipped=amp_step_skipped,
    )


def _autocast_context(
    *,
    torch_module: object,
    row_spec: pretest.RowSpec,
) -> AbstractContextManager[object]:
    enabled = row_spec.precision_policy != AMP_OFF_FP32
    dtype = fastpath_autocast_dtype(
        row_spec.autocast_dtype,
        amp_enabled=enabled,
    )
    return cast(
        "AbstractContextManager[object]",
        cast("object", torch_module).autocast(
            device_type="cuda",
            dtype=dtype,
            enabled=enabled,
            cache_enabled=False,
        ),
    )


def _move_clean_batch_to_device(
    *,
    torch_module: object,
    clean: object,
    device: object,
    row_spec: pretest.RowSpec,
) -> object:
    moved = cast("object", clean).to(
        device=device,
        non_blocking=pretest.DEFAULT_DATALOADER_NON_BLOCKING_H2D,
    )
    return _maybe_channels_last(
        torch_module=torch_module,
        tensor=moved,
        row_spec=row_spec,
    )


def _maybe_channels_last(
    *,
    torch_module: object,
    tensor: object,
    row_spec: pretest.RowSpec,
) -> object:
    if row_spec.memory_format != "channels_last":
        return tensor
    return cast("object", tensor).contiguous(
        memory_format=cast("object", torch_module).channels_last,
    )


def _compile_ddp_model_if_requested(
    *,
    torch_module: object,
    model: object,
    row_spec: pretest.RowSpec,
) -> object:
    # COMPILE_STEP compiles the whole train-step closure (see
    # `_build_compiled_ddp_step`), not the model object, so the DDP model is returned
    # unchanged here and invoked eagerly in the full-telemetry proof loop.
    if row_spec.compile_scope in {COMPILE_NONE, COMPILE_STEP}:
        return model
    if row_spec.compile_scope != COMPILE_MODEL_FORWARD:
        message = f"Unsupported compile_scope: {row_spec.compile_scope}"
        raise ValueError(message)
    compile_fn = cast("object", torch_module).compile
    return compile_fn(model, dynamic=row_spec.compile_dynamic)


def _build_eager_ddp_optimizer(
    *,
    torch_module: object,
    raw_model: object,
    settings: pretest.RealDataRuntimePretestSettings,
    row_spec: pretest.RowSpec,
) -> object:
    """Build the eager-path AdamW optimizer selected by ``optimizer_implementation``.

    Returns:
        The grouped AdamW optimizer for the dual-T4 timing row.

    """
    from eqvae.training.optim import (  # noqa: PLC0415
        SpecAdamWConfig,
        build_adamw_parameter_groups,
        create_adamw_optimizer,
    )

    optimizer_config = SpecAdamWConfig(
        learning_rate=settings.learning_rate,
        weight_decay=settings.weight_decay,
        gate_lr_multiplier=1.0,
        gradient_clip_global_norm=settings.gradient_clip_global_norm,
        beta1=0.9,
        beta2=0.999,
    )
    if row_spec.optimizer_implementation == "adamw_default":
        optimizer, _summary = create_adamw_optimizer(raw_model, config=optimizer_config)
        return cast("object", optimizer)
    parameter_groups, _summary = build_adamw_parameter_groups(
        raw_model,
        config=optimizer_config,
    )
    optimizer_kwargs: dict[str, object] = {}
    if row_spec.optimizer_implementation == "adamw_foreach":
        optimizer_kwargs["foreach"] = True
    elif row_spec.optimizer_implementation == "adamw_fused":
        optimizer_kwargs["fused"] = True
    else:
        message = (
            f"Unsupported optimizer_implementation: {row_spec.optimizer_implementation}"
        )
        raise ValueError(message)
    return cast("object", torch_module).optim.AdamW(
        cast("list[dict[str, object]]", parameter_groups),
        lr=optimizer_config.learning_rate,
        betas=(optimizer_config.beta1, optimizer_config.beta2),
        eps=optimizer_config.epsilon,
        weight_decay=optimizer_config.weight_decay,
        **optimizer_kwargs,
    )


# The compiled whole-step backend. Matches the generator's derived plan value
# (`runtime_selection._selected_runtime_payload`: any compiled scope -> "inductor"), so
# the measured settle/throughput reflect the backend the runner consumes.
_STEP_COMPILE_BACKEND = "inductor"
_COMPILED_OPTIMIZE_DDP_MODES = frozenset({
    "ddp_optimizer",
    "python_reducer",
    "python_reducer_without_compiled_forward",
    "no_optimization",
})

# The failure_kind stamped on a dual row whose per-device batch is VRAM-infeasible: the
# single-GPU no-DDP screen OOM'd or left under the shared margin. Distinct from a real
# crash so the row carries oom=true (a clean "does not fit", not a benchmark bug).
_VRAM_INFEASIBLE_FAILURE_KIND = "dual_t4_ddp_vram_infeasible_oom"
_DDP_RUNTIME_OOM_FAILURE_KIND = "dual_t4_ddp_runtime_oom"
_COMPILED_EXECUTION_PROOF_FAILURE_KIND = "compiled_execution_proof_missing_or_failed"
_VRAM_FEASIBILITY_AMP_SKIPS_FAILURE_KIND = (
    "compiled_step_vram_probe_no_successful_optimizer_update"
)
_MAX_PROOF_AMP_CALIBRATION_ATTEMPTS = 8
_COMPILED_UPDATE_ABS_THRESHOLD = 1.0e-7
# At least two steps put first-trace and steady allocations in the peak. AMP may need
# more scale-backoff attempts before a real optimizer update allocates fused state.
_MIN_FEASIBILITY_PROBE_STEPS = 2
_MAX_FEASIBILITY_PROBE_STEPS = _MAX_PROOF_AMP_CALIBRATION_ATTEMPTS


class _FeasibilityProbeAmpStepsSkippedError(RuntimeError):
    """Signal that bounded AMP probe steps never allocated optimizer state."""


def _all_reduce_int(value: int, *, device: object) -> int:
    """Sum ``value`` across the DDP ranks so both agree on the reduced flag.

    Used to reduce the per-rank single-GPU feasibility flag: ``NO_OOM`` (0) sums to 0
    only when every rank was feasible, so any rank's ``OOM`` (1) makes the batch
    infeasible for all -- and because both ranks that RETURN a flag call this
    collective, they take the identical skip-or-continue branch (Spec 0011 S14c). The
    screen catches every classified VRAM failure and returns a flag, so the only way a
    rank misses this reduce is a truly-unexpected re-raised error, whose one-sided exit
    the ``_run_dual_row`` subprocess timeout bounds.

    Returns:
        The cross-rank sum of ``value``.

    """
    import torch  # noqa: PLC0415
    import torch.distributed as dist  # noqa: PLC0415

    flag = torch.tensor(value, device=device, dtype=torch.int64)
    dist.all_reduce(flag)
    return int(flag.item())


def _screen_compiled_step_vram_feasibility(
    *,
    device: object,
    settings: pretest.RealDataRuntimePretestSettings,
    row_spec: pretest.RowSpec,
    latent_channels: int,
    profile: object,
) -> int:
    """Return ``OOM``/``NO_OOM`` for a compiled step row via a single-GPU no-DDP probe.

    Builds a FRESH model (so the real-timing ``raw_model`` stays pristine) plus the SAME
    compiled fused whole-step the DDP timing runs, but with NO DDP wrap: a *classified*
    VRAM failure here (see :func:`is_oom_error`) issues no collective, so it is returned
    as a flag the caller reduces (the only cross-rank op) and cannot desync the peer; a
    truly-unexpected error re-raises instead. Synthetic zero tensors drive
    at least ``_MIN_FEASIBILITY_PROBE_STEPS`` full steps and permits bounded AMP
    calibration so the inductor first-trace scratch, activations, gradients, and fused
    optimizer state all land in the physical free-VRAM reading. A batch is infeasible
    if the probe OOMs or leaves less than the
    shared ``VRAM_MARGIN_MB`` -- the DDP-only footprint (buckets, NCCL buffers, split
    graph) the real dual-T4 timing adds on top. The synthetic no-dataset path is reused
    ONLY for this feasibility verdict, never for the throughput number (Spec 0011 S14c).

    On a feasible verdict the fresh model + compiled artifacts are ``del``'d and then
    the dynamo cache + peak memory stats reset, so the following real DDP build on this
    rank compiles from a clean cache and measures a peak untouched by this screen.

    Returns:
        ``OOM`` when the batch OOMs or leaves under the margin, else ``NO_OOM``.

    """
    import gc  # noqa: PLC0415

    import torch  # noqa: PLC0415
    import torch._dynamo as torch_dynamo  # noqa: PLC0415, PLC2701

    from eqvae.corruption.inline_stain import InlineStainCorruptor  # noqa: PLC0415
    from eqvae.models.registry import (  # noqa: PLC0415
        MODEL_KIND_NON_EQ_TRANSLATABLE,
        build_model,
    )
    from eqvae.training.fastpath_recipe import (  # noqa: PLC0415
        apply_fastpath_dynamo_config,
        build_fastpath_optimizer,
        compiled_autograd_context,
    )
    from eqvae.training.fastpath_step import make_fastpath_step_fn  # noqa: PLC0415
    from eqvae.training.optim import SpecAdamWConfig  # noqa: PLC0415

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    channels_last = row_spec.memory_format == "channels_last"
    try:
        apply_fastpath_dynamo_config(
            optimize_ddp=row_spec.optimize_ddp,
            compiled_autograd=row_spec.compiled_autograd,
            reorder_compute_comm_overlap=row_spec.reorder_compute_comm_overlap,
        )
        model = build_model(
            MODEL_KIND_NON_EQ_TRANSLATABLE,
            model_config={"norm_groups": settings.norm_groups},
        ).to(device)
        _set_scalar_gate_precision(
            model=model,
            force_fp32=row_spec.precision_policy != AMP_SCALAR_GATE_RELAXED,
        )
        if channels_last:
            model = model.to(memory_format=torch.channels_last)
        optimizer = build_fastpath_optimizer(
            model,
            config=SpecAdamWConfig(
                learning_rate=settings.learning_rate,
                weight_decay=settings.weight_decay,
                gate_lr_multiplier=1.0,
                gradient_clip_global_norm=settings.gradient_clip_global_norm,
                beta1=0.9,
                beta2=0.999,
                fused=row_spec.fused_optimizer,
            ),
        )
        scaler = build_fastpath_grad_scaler(
            enabled=row_spec.grad_scaler_enabled,
        )
        corruptor = InlineStainCorruptor(profile).to(device=device)
        step_fn = make_fastpath_step_fn(
            model,
            corruptor,
            ssim_weight=settings.ssim_weight,
            autocast_dtype=fastpath_autocast_dtype(
                row_spec.autocast_dtype,
                amp_enabled=row_spec.precision_policy != AMP_OFF_FP32,
            ),
            autocast_enabled=row_spec.precision_policy != AMP_OFF_FP32,
        )
        compiled_step_fn = torch.compile(
            step_fn,
            dynamic=False,
            backend=_STEP_COMPILE_BACKEND,
        )
        latent_size = settings.image_size // pretest.LATENT_DOWNSAMPLE_FACTOR
        x_uint8 = torch.zeros(
            (
                row_spec.per_device_batch_size,
                settings.channels,
                settings.image_size,
                settings.image_size,
            ),
            dtype=torch.uint8,
            device=device,
        )
        eps = torch.zeros(
            (row_spec.per_device_batch_size, latent_channels, latent_size, latent_size),
            dtype=torch.float32,
            device=device,
        )
        if channels_last:
            x_uint8 = x_uint8.contiguous(memory_format=torch.channels_last)
            eps = eps.contiguous(memory_format=torch.channels_last)
        beta = torch.zeros((), device=device)
        # Bound before the loop so the feasible-path ``del output`` below stays defined
        # even to a checker that cannot prove the bounded range iterates;
        # the loop always rebinds it to the last step's real output.
        output = None
        amp_step_skips: list[bool] = []
        for _ in range(_MAX_FEASIBILITY_PROBE_STEPS):
            optimizer.zero_grad(set_to_none=True)
            output = compiled_step_fn(x_uint8, eps, beta)
            amp_step_skips.append(
                run_fastpath_optimizer_step(
                    loss=output.loss,
                    optimizer=optimizer,
                    parameters=model.parameters(),
                    scaler=scaler,
                    grad_scaler_enabled=row_spec.grad_scaler_enabled,
                    gradient_clip_global_norm=settings.gradient_clip_global_norm,
                    gradient_clip_foreach=row_spec.gradient_clip_foreach,
                    backward_context=compiled_autograd_context(
                        enabled=row_spec.compiled_autograd,
                    ),
                ),
            )
            if (
                len(amp_step_skips) >= _MIN_FEASIBILITY_PROBE_STEPS
                and _successful_feasibility_optimizer_updates(
                    amp_step_skips=amp_step_skips,
                )
                > 0
            ):
                break

        successful_optimizer_updates = _successful_feasibility_optimizer_updates(
            amp_step_skips=amp_step_skips,
        )
        headroom = _probe_headroom_after_successful_optimizer_update(
            successful_optimizer_updates=successful_optimizer_updates,
            device=device,
        )
    except torch.cuda.OutOfMemoryError:
        # Infeasible: no real DDP build follows on this rank, so the (partial) locals
        # are freed on return; just reset the process dynamo cache + peak stats.
        return _reset_after_feasibility_probe(
            torch,
            torch_dynamo,
            device=device,
            flag=OOM,
        )
    except RuntimeError as error:
        if not is_oom_error(error):
            raise
        return _reset_after_feasibility_probe(
            torch,
            torch_dynamo,
            device=device,
            flag=OOM,
        )
    flag = OOM if headroom_below_margin(headroom) else NO_OOM
    # Feasible: a real DDP build follows on THIS rank, so drop the frame's references
    # to the fresh model + compiled artifacts BEFORE the reset's empty_cache -- only
    # then can empty_cache release those blocks (and defragment) for the real build.
    # ``output`` (the last step's FastpathStepOutput, holding a full [B, C, H, W]
    # ``reconstruction``) is always bound here since the maximum steps are >= 1, and
    # must be dropped too or its block outlives empty_cache.
    del model, optimizer, scaler, corruptor, step_fn, compiled_step_fn
    del x_uint8, eps, beta, output
    return _reset_after_feasibility_probe(torch, torch_dynamo, device=device, flag=flag)


def _probe_headroom_after_successful_optimizer_update(
    *,
    successful_optimizer_updates: int,
    device: object,
) -> int:
    """Read headroom only after at least one optimizer update allocated its state.

    Returns:
        Physical free-VRAM headroom in bytes.

    Raises:
        _FeasibilityProbeAmpStepsSkippedError: If GradScaler skipped every bounded
            probe step, leaving optimizer-state allocation unproven.

    """
    if successful_optimizer_updates <= 0:
        raise _FeasibilityProbeAmpStepsSkippedError(
            _VRAM_FEASIBILITY_AMP_SKIPS_FAILURE_KIND,
        )
    return probe_headroom_bytes(device)


def _successful_feasibility_optimizer_updates(
    *,
    amp_step_skips: Sequence[bool],
) -> int:
    """Count bounded VRAM-probe updates not skipped by GradScaler.

    Returns:
        The number of probe steps whose optimizer update executed.

    """
    return sum(int(not skipped) for skipped in amp_step_skips)


def _reset_after_feasibility_probe(
    torch_module: object,
    torch_dynamo_module: object,
    *,
    device: object,
    flag: int,
) -> int:
    """Reset the process dynamo cache + CUDA peak stats after a probe; return ``flag``.

    Clears the dynamo compile cache and resets the peak memory stats so the real DDP
    build that follows on a feasible rank compiles from a clean cache and measures a
    peak untouched by the screen. The caller must ``del`` the probe's
    model/optimizer/compiled locals BEFORE calling this on the feasible path, so
    ``empty_cache`` can actually reclaim their blocks (this helper holds no references).

    Returns:
        The feasibility ``flag`` passed in (``OOM`` or ``NO_OOM``).

    """
    import gc  # noqa: PLC0415

    torch_dynamo_module.reset()
    gc.collect()
    torch_module.cuda.empty_cache()
    torch_module.cuda.reset_peak_memory_stats(device)
    return flag


def _vram_infeasible_rank_payload(
    *,
    rank: int,
    local_rank: int,
    row_id: str,
    torch_module: object,
    dist_module: object,
) -> JsonObject:
    """Build the rank payload for a batch the single-GPU screen found VRAM-infeasible.

    Written by BOTH ranks (they agreed via ``_all_reduce_int``) then the child exits 0,
    so ``_run_dual_row`` parses the payloads instead of discarding a non-zero child as a
    generic ``torchrun_failed``. Carries ``oom=true`` so
    ``_dual_row_from_rank_payloads`` stamps the row's ``oom`` cell -- a clean verdict.

    Returns:
        The infeasible rank payload dict.

    """
    return {
        "status": FAIL_STATUS,
        "rank": rank,
        "local_rank": local_rank,
        "current_device": int(torch_module.cuda.current_device()),
        "world_size": int(dist_module.get_world_size()),
        "row_id": row_id,
        "device_name": torch_module.cuda.get_device_name(local_rank),
        "failure_kind": _VRAM_INFEASIBLE_FAILURE_KIND,
        "failure_message_hash": pretest._hash_text(_VRAM_INFEASIBLE_FAILURE_KIND),  # noqa: SLF001
        "oom": True,
    }


def _ddp_rank_failure_payload(
    *,
    rank: int,
    local_rank: int,
    row_id: str,
    error: BaseException,
    torch_module: object,
    dist_module: object,
) -> JsonObject:
    """Build a classified child failure payload before torchrun tears peers down.

    Returns:
        A rank payload that distinguishes runtime VRAM exhaustion, bounded AMP-probe
        skip failure, and unrelated child errors.

    """
    oom = is_oom_error(error)
    if oom:
        failure_kind = _DDP_RUNTIME_OOM_FAILURE_KIND
    elif isinstance(error, _FeasibilityProbeAmpStepsSkippedError):
        failure_kind = _VRAM_FEASIBILITY_AMP_SKIPS_FAILURE_KIND
    else:
        failure_kind = f"ddp_rank_{type(error).__name__}"
    return {
        "status": FAIL_STATUS,
        "rank": rank,
        "local_rank": local_rank,
        "current_device": int(torch_module.cuda.current_device()),
        "world_size": int(dist_module.get_world_size()),
        "row_id": row_id,
        "device_name": torch_module.cuda.get_device_name(local_rank),
        "failure_kind": failure_kind,
        "failure_message_hash": pretest._hash_text(str(error)),  # noqa: SLF001
        "oom": oom,
    }


def _build_compiled_ddp_step(
    *,
    raw_model: object,
    local_rank: int,
    device: object,
    profile: object,
    settings: pretest.RealDataRuntimePretestSettings,
    row_spec: pretest.RowSpec,
    torch_module: object,
) -> tuple[object, object, object, object]:
    """Build the DDP model, fused optimizer, and compiled whole-step closure.

    Mirrors the runner's ``_maybe_build_compiled_step`` (Spec 0011 S16): the DDP wrap,
    fused AdamW, and process-global dynamo config are driven by the S14a-threaded
    recipe knobs, and branchless inline corruption + the AMP forward + the FP32 loss
    island fuse into one ``torch.compile(dynamic=False)`` graph, so the measured
    throughput and the zero-graph-break settle proof match what the runner later
    consumes. The returned DDP model is invoked eagerly in the full-telemetry proof
    loop; the compiled closure drives the settle/warmup/measured loops.

    One fail-closed precondition keeps the measured recipe faithful to what the runner
    consumes (a silently divergent measurement is exactly what Spec 0011 forbids):

    * ``static_graph=False`` only -- a ``step`` row runs the eager numerical-proof
      backward before compiled settle on the same DDP module. ``static_graph=True``
      locks the backward structure on that first eager iteration; changing to the
      compiled closure on the next iteration is not yet proven safe. A4 gives the proof
      its own module before this guard can be removed.

    Returns:
        The DDP model, optimizer, eager closure, and compiled closure. The eager
        closure is retained only for the untimed same-input compiled-execution proof.

    Raises:
        ValueError: If the row requests ``static_graph=True``, which this measurement
            path cannot yet faithfully mirror to the runner.

    """
    if row_spec.ddp_static_graph:
        message = (
            "Compiled whole-step rows run an eager numerical-proof backward before "
            "compiled settle on the same DDP module, so ddp_static_graph must be False "
            "until A4 isolates the proof module."
        )
        raise ValueError(message)
    from eqvae.corruption.inline_stain import InlineStainCorruptor  # noqa: PLC0415
    from eqvae.training.fastpath_recipe import (  # noqa: PLC0415
        FastpathDynamoKnobs,
        build_fastpath_optimizer,
        model_requires_buffer_broadcast,
        wrap_fastpath_ddp,
    )
    from eqvae.training.fastpath_step import make_fastpath_step_fn  # noqa: PLC0415
    from eqvae.training.optim import SpecAdamWConfig  # noqa: PLC0415

    ddp_model = wrap_fastpath_ddp(
        raw_model,
        local_rank=local_rank,
        static_graph=row_spec.ddp_static_graph,
        gradient_as_bucket_view=row_spec.ddp_gradient_as_bucket_view,
        broadcast_buffers=row_spec.ddp_broadcast_buffers
        or model_requires_buffer_broadcast(raw_model),
        find_unused_parameters=row_spec.ddp_find_unused_parameters,
        bucket_cap_mb=row_spec.ddp_bucket_cap_mb,
        # Applied by the wrapper immediately before DDP construction, which latches the
        # dynamo optimize_ddp mode (Spec 0011 S17f) -- a measured row must not diverge
        # from the recipe it claims to measure.
        dynamo=FastpathDynamoKnobs(
            optimize_ddp=row_spec.optimize_ddp,
            compiled_autograd=row_spec.compiled_autograd,
            reorder_compute_comm_overlap=row_spec.reorder_compute_comm_overlap,
        ),
    )
    optimizer = build_fastpath_optimizer(
        raw_model,
        config=SpecAdamWConfig(
            learning_rate=settings.learning_rate,
            weight_decay=settings.weight_decay,
            gate_lr_multiplier=1.0,
            gradient_clip_global_norm=settings.gradient_clip_global_norm,
            beta1=0.9,
            beta2=0.999,
            fused=row_spec.fused_optimizer,
        ),
    )
    corruptor = InlineStainCorruptor(profile).to(device=device)
    step_fn = make_fastpath_step_fn(
        cast("object", ddp_model),
        corruptor,
        ssim_weight=settings.ssim_weight,
        autocast_dtype=fastpath_autocast_dtype(
            row_spec.autocast_dtype,
            amp_enabled=row_spec.precision_policy != AMP_OFF_FP32,
        ),
        autocast_enabled=row_spec.precision_policy != AMP_OFF_FP32,
    )
    compiled_step_fn = cast("object", torch_module).compile(
        step_fn,
        dynamic=False,
        backend=_STEP_COMPILE_BACKEND,
    )
    return (
        cast("object", ddp_model),
        cast("object", optimizer),
        cast("object", step_fn),
        compiled_step_fn,
    )


def _run_compiled_ddp_execution_proof(  # noqa: PLR0913, PLR0914
    *,
    iterator: object,
    eager_step_fn: object,
    compiled_step_fn: object,
    optimizer: object,
    scaler: object,
    model: object,
    raw_model: object,
    device: object,
    settings: pretest.RealDataRuntimePretestSettings,
    step_index: int,
    row_spec: pretest.RowSpec,
    latent_channels: int,
    beta_for_step_fn: object,
    torch_module: object,
    dist_module: object,
) -> JsonObject:
    """Check one real compiled DDP update outside the timed measurement.

    Inductor intentionally uses a different random stream from eager for the inline
    stochastic corruption, so eager/compiled value parity is not meaningful here. The
    useful one-off checks are finite outputs, a finite nonzero optimizer update, and
    synchronized DDP parameters.

    Returns:
        Untimed evidence that the compiled update is healthy and synchronized.

    Raises:
        RuntimeError: If outputs/update are invalid, AMP skips, or ranks diverge.

    """
    from eqvae.training.ddp_sync_guard import (  # noqa: PLC0415
        assert_ddp_parameters_exactly_in_sync,
    )
    from eqvae.training.fastpath_recipe import (  # noqa: PLC0415
        compiled_autograd_context,
    )

    del eager_step_fn
    torch = cast("object", torch_module)
    batch = next(cast("object", iterator))
    x_uint8 = _move_clean_batch_to_device(
        torch_module=torch_module,
        clean=batch.images_uint8,
        device=device,
        row_spec=row_spec,
    )
    shape = cast("tuple[int, int, int, int]", tuple(x_uint8.shape))
    eps = torch.zeros(
        (
            shape[0],
            latent_channels,
            settings.image_size // pretest.LATENT_DOWNSAMPLE_FACTOR,
            settings.image_size // pretest.LATENT_DOWNSAMPLE_FACTOR,
        ),
        dtype=torch.float32,
        device=device,
    )
    if row_spec.memory_format == "channels_last":
        eps = eps.contiguous(memory_format=torch.channels_last)
    beta_value = cast("object", beta_for_step_fn)(
        optimizer_step_index=step_index,
        max_optimizer_steps=settings.warmup_steps + settings.measured_steps + 1,
        target_beta=settings.beta_target,
        warmup_fraction=settings.beta_warmup_fraction,
    )
    beta = torch.tensor(float(beta_value), dtype=torch.float32, device=device)
    cast("object", optimizer).zero_grad(set_to_none=row_spec.zero_grad_set_to_none)
    initial_parameters = _clone_trainable_parameters(
        cast("object", model).parameters(),
    )
    compiled_output = cast("object", compiled_step_fn)(x_uint8, eps, beta)
    checked_fields = (
        "loss",
        "recon_loss",
        "l1_loss",
        "ssim_loss",
        "ssim_metric",
        "kl_loss",
        "reconstruction",
    )
    for field_name in checked_fields:
        compiled_value = getattr(compiled_output, field_name)
        if not bool(torch.isfinite(compiled_value.detach()).all()):
            message = f"compiled execution proof produced nonfinite {field_name}"
            raise RuntimeError(message)
    amp_step_skipped = run_fastpath_optimizer_step(
        loss=cast("object", compiled_output.loss),
        optimizer=cast("object", optimizer),
        parameters=cast("object", model).parameters(),
        scaler=cast("object", scaler),
        grad_scaler_enabled=row_spec.grad_scaler_enabled,
        gradient_clip_global_norm=settings.gradient_clip_global_norm,
        gradient_clip_foreach=row_spec.gradient_clip_foreach,
        backward_context=compiled_autograd_context(
            enabled=row_spec.compiled_autograd,
        ),
    )
    if amp_step_skipped:
        message = "compiled execution proof AMP update was skipped"
        raise RuntimeError(message)
    parameter_update_norm = _parameter_update_norm(
        before=initial_parameters,
        after=_trainable_parameters(cast("object", model).parameters()),
    )
    if not math.isfinite(parameter_update_norm) or parameter_update_norm <= 0.0:
        message = "compiled execution proof produced invalid optimizer update"
        raise RuntimeError(message)
    assert_ddp_parameters_exactly_in_sync(
        cast("object", raw_model),
        world_size=int(cast("object", dist_module).get_world_size()),
    )
    return {
        "status": PASS_STATUS,
        "outputs_finite": True,
        "parameter_update_finite_nonzero": True,
        "parameter_update_norm": parameter_update_norm,
        "successful_optimizer_update_count": 1,
        "ddp_parameters_in_sync": True,
    }


def _nested_execution_state_close(  # noqa: PLR0911
    *,
    left: object,
    right: object,
    torch_module: object,
) -> tuple[bool, float]:
    """Compare nested optimizer state with the accepted numerical drift.

    Returns:
        A pair of ``(close, maximum floating-tensor absolute delta)``.

    """
    torch = cast("object", torch_module)
    if torch.is_tensor(left) or torch.is_tensor(right):
        if not (torch.is_tensor(left) and torch.is_tensor(right)):
            return (False, 0.0)
        left_tensor = cast("torch.Tensor", left)
        right_tensor = cast("torch.Tensor", right)
        if (
            left_tensor.shape != right_tensor.shape
            or left_tensor.dtype != right_tensor.dtype
        ):
            return (False, 0.0)
        if not (left_tensor.is_floating_point() or left_tensor.is_complex()):
            return (bool(torch.equal(left_tensor, right_tensor)), 0.0)
        delta = float(torch.amax(torch.abs(left_tensor - right_tensor)).item())
        return (
            bool(
                torch.allclose(
                    left_tensor,
                    right_tensor,
                    rtol=pretest.NUMERICAL_REL_THRESHOLD,
                    atol=_COMPILED_UPDATE_ABS_THRESHOLD,
                ),
            ),
            delta,
        )
    if isinstance(left, dict) or isinstance(right, dict):
        if not (isinstance(left, dict) and isinstance(right, dict)):
            return (False, 0.0)
        if left.keys() != right.keys():
            return (False, 0.0)
        comparisons = tuple(
            _nested_execution_state_close(
                left=left[key],
                right=right[key],
                torch_module=torch_module,
            )
            for key in left
        )
    elif isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not (
            isinstance(left, (list, tuple))
            and isinstance(right, (list, tuple))
            and len(left) == len(right)
        ):
            return (False, 0.0)
        comparisons = tuple(
            _nested_execution_state_close(
                left=left_value,
                right=right_value,
                torch_module=torch_module,
            )
            for left_value, right_value in zip(left, right, strict=True)
        )
    else:
        return (left == right, 0.0)
    return (
        all(close for close, _delta in comparisons),
        max((delta for _close, delta in comparisons), default=0.0),
    )


def _parameter_update_parity(  # pyright: ignore[reportUnusedFunction]
    *,
    initial_parameters: Sequence[object],
    eager_updated_parameters: Sequence[object],
    compiled_updated_parameters: Sequence[object],
    torch_module: object,
) -> tuple[bool, float]:
    """Compare optimizer update deltas, not tolerance-dominated absolute weights.

    Returns:
        A pair of ``(all update deltas close, maximum delta disagreement)``.

    """
    torch = cast("object", torch_module)
    if not (
        len(initial_parameters)
        == len(eager_updated_parameters)
        == len(compiled_updated_parameters)
    ):
        return (False, 0.0)
    close = True
    max_abs_delta = 0.0
    for initial, eager_updated, compiled_updated in zip(
        initial_parameters,
        eager_updated_parameters,
        compiled_updated_parameters,
        strict=True,
    ):
        initial_tensor = cast("torch.Tensor", initial)
        eager_delta = cast("torch.Tensor", eager_updated) - initial_tensor
        compiled_delta = cast("torch.Tensor", compiled_updated) - initial_tensor
        disagreement = torch.amax(torch.abs(eager_delta - compiled_delta))
        max_abs_delta = max(max_abs_delta, float(disagreement.item()))
        close = close and bool(
            torch.allclose(
                eager_delta,
                compiled_delta,
                rtol=pretest.NUMERICAL_REL_THRESHOLD,
                atol=_COMPILED_UPDATE_ABS_THRESHOLD,
            ),
        )
    return (close, max_abs_delta)


def _run_until_successful_amp_proof(
    *,
    run_attempt: Callable[[], JsonObject],
    fixed_batch_index: int,
) -> tuple[JsonObject, int, int]:
    """Retry one fixed proof batch while AMP calibrates, then return only success.

    Returns:
        The successful proof payload, attempt count, and calibration skip count.

    Raises:
        RuntimeError: If the bounded calibration window produces no optimizer update.

    """
    for attempt_count in range(1, _MAX_PROOF_AMP_CALIBRATION_ATTEMPTS + 1):
        proof = run_attempt()
        if not bool(proof.get("amp_step_skipped")):
            return (proof, attempt_count, attempt_count - 1)
    message = (
        "AMP proof calibration produced no successful optimizer update for fixed "
        f"batch {fixed_batch_index}"
    )
    raise RuntimeError(message)


def _amp_phase_accounting(
    *,
    proof_calibration_step_count: int,
    proof_calibration_skipped_count: int,
    timing_calibration_step_count: int,
    timing_calibration_skipped_count: int,
    measured_amp_step_skipped_count: int,
) -> _AmpPhaseAccounting:
    """Derive timing-scaler success and keep calibration out of selection skips.

    Returns:
        Phase-scoped diagnostics plus the measured-only selection skip count.

    """
    return _AmpPhaseAccounting(
        proof_calibration_step_count=proof_calibration_step_count,
        proof_calibration_skipped_count=proof_calibration_skipped_count,
        timing_calibration_step_count=timing_calibration_step_count,
        timing_calibration_skipped_count=timing_calibration_skipped_count,
        timing_successful_optimizer_update_count=(
            timing_calibration_step_count - timing_calibration_skipped_count
        ),
        selection_amp_step_skipped_count=measured_amp_step_skipped_count,
    )


def _run_compiled_ddp_step_batch(
    *,
    iterator: object,
    compiled_step_fn: object,
    optimizer: object,
    scaler: object,
    model: object,
    device: object,
    settings: pretest.RealDataRuntimePretestSettings,
    step_index: int,
    row_spec: pretest.RowSpec,
    latent_channels: int,
    beta_for_step_fn: object,
) -> JsonObject:
    """Drive one compiled whole-step batch (settle / warmup / measured).

    The compiled closure fuses inline corruption + the forward + the FP32 loss; the
    backward (eager, or compiled-autograd per the recipe), gradient clipping, and the
    optimizer step stay eager here, exactly as the runner drives the recipe. Only the
    uint8 batch (normalized inside the graph), ``eps``, and a 0-dim ``beta`` tensor
    cross the graph boundary. AMP rows use the same persistent GradScaler ordering as
    the runner.

    Returns:
        The minimal telemetry the throughput loops consume: observed batch size and
        whether GradScaler skipped the optimizer update.

    """
    import torch  # noqa: PLC0415

    from eqvae.training.fastpath_recipe import (  # noqa: PLC0415
        compiled_autograd_context,
    )

    batch = next(cast("object", iterator))
    # Transfer uint8 (channels_last applied on-device by _move_clean_batch_to_device);
    # the compiled step folds the uint8->float normalize into the graph, so only uint8
    # crosses H2D.
    x_uint8 = _move_clean_batch_to_device(
        torch_module=torch,
        clean=batch.images_uint8,
        device=device,
        row_spec=row_spec,
    )
    shape = cast("tuple[int, int, int, int]", tuple(x_uint8.shape))
    eps = torch.zeros(
        (
            shape[0],
            latent_channels,
            settings.image_size // pretest.LATENT_DOWNSAMPLE_FACTOR,
            settings.image_size // pretest.LATENT_DOWNSAMPLE_FACTOR,
        ),
        dtype=torch.float32,
        device=device,
    )
    if row_spec.memory_format == "channels_last":
        eps = eps.contiguous(memory_format=torch.channels_last)
    beta_value = cast("object", beta_for_step_fn)(
        optimizer_step_index=step_index,
        max_optimizer_steps=settings.warmup_steps + settings.measured_steps + 1,
        target_beta=settings.beta_target,
        warmup_fraction=settings.beta_warmup_fraction,
    )
    # beta crosses the compiled graph boundary as a 0-dim tensor so the warmup schedule
    # changing its value never forces a ``dynamic=False`` recompile.
    beta = torch.tensor(float(beta_value), dtype=torch.float32, device=device)
    cast("object", optimizer).zero_grad(set_to_none=row_spec.zero_grad_set_to_none)
    output = cast("object", compiled_step_fn)(x_uint8, eps, beta)
    amp_step_skipped = run_fastpath_optimizer_step(
        loss=cast("object", output.loss),
        optimizer=cast("object", optimizer),
        parameters=cast("object", model).parameters(),
        scaler=cast("object", scaler),
        grad_scaler_enabled=row_spec.grad_scaler_enabled,
        gradient_clip_global_norm=settings.gradient_clip_global_norm,
        gradient_clip_foreach=row_spec.gradient_clip_foreach,
        backward_context=compiled_autograd_context(
            enabled=row_spec.compiled_autograd,
        ),
    )
    return {
        "observed_batch_size": int(shape[0]),
        "amp_step_skipped": amp_step_skipped,
    }


def _set_scalar_gate_precision(*, model: object, force_fp32: bool) -> int:
    """Set scalar gate math precision for runtime-policy rows.

    Returns:
        Number of gate modules updated.

    """
    from eqvae.models.activations import GatedScalarActivation  # noqa: PLC0415

    updated = 0
    for module in cast("object", model).modules():
        if isinstance(module, GatedScalarActivation):
            module.force_fp32 = force_fp32
            updated += 1
    return updated


def _backend_state(*, torch_module: object) -> JsonObject:
    backends = cast("object", torch_module).backends
    cuda_backend = getattr(backends, "cuda", None)
    matmul_backend = getattr(cuda_backend, "matmul", None)
    cudnn_backend = backends.cudnn
    state: JsonObject = {
        "cudnn_benchmark": bool(cudnn_backend.benchmark),
        "cudnn_deterministic": bool(cudnn_backend.deterministic),
        "cudnn_allow_tf32": bool(getattr(cudnn_backend, "allow_tf32", False)),
        "matmul_allow_tf32": bool(getattr(matmul_backend, "allow_tf32", False)),
    }
    deterministic_enabled = getattr(
        cast("object", torch_module),
        "are_deterministic_algorithms_enabled",
        None,
    )
    if callable(deterministic_enabled):
        state["deterministic_algorithms"] = bool(deterministic_enabled())
    get_precision = getattr(
        cast("object", torch_module),
        "get_float32_matmul_precision",
        None,
    )
    if callable(get_precision):
        state["matmul_precision"] = str(get_precision())
    return state


def _apply_backend_policy(
    *,
    torch_module: object,
    row_spec: pretest.RowSpec,
) -> JsonObject:
    state = _backend_state(torch_module=torch_module)
    backends = cast("object", torch_module).backends
    cuda_backend = getattr(backends, "cuda", None)
    matmul_backend = getattr(cuda_backend, "matmul", None)
    cudnn_backend = backends.cudnn
    cudnn_backend.benchmark = row_spec.cudnn_benchmark
    cudnn_backend.deterministic = row_spec.cudnn_deterministic
    if hasattr(cudnn_backend, "allow_tf32"):
        cudnn_backend.allow_tf32 = row_spec.tf32_enabled
    if matmul_backend is not None and hasattr(matmul_backend, "allow_tf32"):
        matmul_backend.allow_tf32 = row_spec.tf32_enabled
    set_precision = getattr(
        cast("object", torch_module),
        "set_float32_matmul_precision",
        None,
    )
    if callable(set_precision):
        set_precision(row_spec.matmul_precision)
    cast("object", torch_module).use_deterministic_algorithms(
        row_spec.deterministic_algorithms,
    )
    return state


def _restore_backend_policy(*, torch_module: object, state: JsonObject) -> None:
    if not state:
        return
    backends = cast("object", torch_module).backends
    cuda_backend = getattr(backends, "cuda", None)
    matmul_backend = getattr(cuda_backend, "matmul", None)
    cudnn_backend = backends.cudnn
    cudnn_backend.benchmark = bool(state.get("cudnn_benchmark", False))
    cudnn_backend.deterministic = bool(state.get("cudnn_deterministic", False))
    if hasattr(cudnn_backend, "allow_tf32"):
        cudnn_backend.allow_tf32 = bool(state.get("cudnn_allow_tf32", False))
    if matmul_backend is not None and hasattr(matmul_backend, "allow_tf32"):
        matmul_backend.allow_tf32 = bool(state.get("matmul_allow_tf32", False))
    set_precision = getattr(
        cast("object", torch_module),
        "set_float32_matmul_precision",
        None,
    )
    if callable(set_precision):
        set_precision(str(state.get("matmul_precision", "highest")))
    cast("object", torch_module).use_deterministic_algorithms(
        mode=bool(state.get("deterministic_algorithms", False)),
    )


def _global_grad_norm(parameters: Sequence[object]) -> float:
    import torch  # noqa: PLC0415

    squared_norm = 0.0
    for parameter in parameters:
        gradient = getattr(parameter, "grad", None)
        if gradient is None:
            continue
        gradient_f32 = gradient.detach().to(dtype=torch.float32)
        squared_norm += float(gradient_f32.square().sum().item())
    return math.sqrt(squared_norm)


def _nonfinite_parameter_count(parameters: Sequence[object]) -> int:
    import torch  # noqa: PLC0415

    count = 0
    for parameter in parameters:
        gradient = getattr(parameter, "grad", None)
        if gradient is not None:
            count += int((~torch.isfinite(gradient)).sum().item())
        count += int((~torch.isfinite(parameter.detach())).sum().item())
    return count


def _clip_grad_norm(
    *,
    torch_module: object,
    parameters: Sequence[object],
    max_norm: float,
    foreach: bool,
) -> None:
    try:
        cast("object", torch_module).nn.utils.clip_grad_norm_(
            parameters,
            max_norm,
            foreach=foreach,
        )
    except TypeError:
        cast("object", torch_module).nn.utils.clip_grad_norm_(parameters, max_norm)


def _trainable_parameters(parameters: Sequence[object]) -> list[object]:
    return [parameter for parameter in parameters if bool(parameter.requires_grad)]


def _clone_trainable_parameters(parameters: Sequence[object]) -> list[object]:
    return [
        parameter.detach().clone()
        for parameter in parameters
        if bool(parameter.requires_grad)
    ]


def _parameter_update_norm(*, before: list[object], after: list[object]) -> float:
    import torch  # noqa: PLC0415

    squared_norm = 0.0
    for before_tensor, after_parameter in zip(before, after, strict=True):
        delta = after_parameter.detach().to(dtype=torch.float32) - before_tensor.to(
            dtype=torch.float32,
        )
        squared_norm += float(delta.square().sum().item())
    return math.sqrt(squared_norm)


def _measure_rank_loader(  # noqa: PLR0913
    *,
    dataset: object,
    indices: Sequence[int],
    split: str,
    batch_size: int,
    device: object,
    data_loader_factory: object,
    subset_factory: object,
    collate_fn: object,
    row_spec: pretest.RowSpec,
    measured_batches: int,
) -> JsonObject:
    import torch  # noqa: PLC0415

    if not indices:
        return {
            "status": FAIL_STATUS,
            "split": split,
            "batches_measured": 0,
            "samples_seen": 0,
            "failure_kind": "empty_rank_indices",
        }
    loader = cast("object", data_loader_factory)(
        cast("object", subset_factory)(dataset, list(indices)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=pretest.DEFAULT_DATALOADER_NUM_WORKERS,
        collate_fn=collate_fn,
    )
    fetch_ms: list[float] = []
    h2d_ms: list[float] = []
    samples_seen = 0
    observed_batches = 0
    iterator = iter(cast("object", loader))
    for _ in range(
        min(
            measured_batches,
            max(1, len(indices) // max(1, batch_size)),
        ),
    ):
        start_fetch = time.perf_counter_ns()
        batch = next(iterator)
        fetch_ms.append(pretest._elapsed_ms(start_fetch))  # noqa: SLF001
        # Measure the corrected pipeline: transfer uint8 (4x fewer bytes) + on-device
        # channels_last; the uint8->float normalize is folded into the compiled step
        # (not H2D). Mirrors the pretest ``_measure_dataloader_split`` uint8 H2D timing.
        start_h2d = time.perf_counter_ns()
        _move_clean_batch_to_device(
            torch_module=torch,
            clean=batch.images_uint8,
            device=device,
            row_spec=row_spec,
        )
        torch.cuda.synchronize(device)
        h2d_ms.append(pretest._elapsed_ms(start_h2d))  # noqa: SLF001
        samples_seen += int(batch.images_uint8.shape[0])
        observed_batches += 1
    total_fetch_sec = sum(fetch_ms) / 1000.0
    return {
        "status": PASS_STATUS,
        "split": split,
        "batches_measured": observed_batches,
        "samples_seen": samples_seen,
        "batch_fetch_ms_p50": pretest._format_float(  # noqa: SLF001
            pretest._percentile(fetch_ms, 0.50),  # noqa: SLF001
        ),
        "batch_fetch_ms_p95": pretest._format_float(  # noqa: SLF001
            pretest._percentile(fetch_ms, 0.95),  # noqa: SLF001
        ),
        "h2d_ms_p50": pretest._format_float(pretest._percentile(h2d_ms, 0.50)),  # noqa: SLF001
        "h2d_ms_p95": pretest._format_float(pretest._percentile(h2d_ms, 0.95)),  # noqa: SLF001
        "loader_samples_sec": pretest._format_float(  # noqa: SLF001
            0.0 if total_fetch_sec <= 0.0 else samples_seen / total_fetch_sec,
        ),
    }


def _rank_indices(indices: Sequence[int], *, rank: int, world_size: int) -> list[int]:
    return list(indices[rank::world_size])


def _load_rank_payloads(*, rank_dir: Path) -> tuple[JsonObject, ...]:
    payloads: list[JsonObject] = []
    for rank in range(EXPECTED_DUAL_T4_COUNT):
        path = rank_dir / f"rank_{rank}.json"
        if not path.exists():
            message = f"missing DDP rank payload {path}"
            raise RuntimeError(message)
        payloads.append(
            cast("JsonObject", json.loads(path.read_text(encoding="utf-8"))),
        )
    return tuple(
        sorted(payloads, key=lambda payload: pretest._required_int(payload, "rank")),
    )  # noqa: SLF001


def _load_available_rank_payloads(*, rank_dir: Path) -> tuple[JsonObject, ...]:
    """Load every parseable rank payload left by a nonzero torchrun child.

    Returns:
        Available payloads ordered by rank; missing peer files are expected after
        torchrun terminates the process group on the first child failure.

    """
    payloads: list[JsonObject] = []
    for path in sorted(rank_dir.glob("rank_*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(value, dict):
            payloads.append(cast("JsonObject", value))
    return tuple(sorted(payloads, key=lambda payload: int(payload.get("rank", 0))))


def _rank_failure_classification(
    *,
    rank_payloads: Sequence[JsonObject],
    fallback_kind: str,
) -> tuple[str, bool]:
    """Prefer a child's classified OOM, then any classified child failure.

    Returns:
        ``(failure_kind, oom)`` for the strongest available rank evidence, falling
        back to the parent process failure only when no child classification exists.

    """
    oom_failures = [payload for payload in rank_payloads if payload.get("oom") is True]
    failures = [
        payload for payload in rank_payloads if payload.get("status") != PASS_STATUS
    ]
    classified = oom_failures or failures
    if not classified:
        return fallback_kind, False
    failure_kind = classified[0].get("failure_kind")
    if not isinstance(failure_kind, str) or not failure_kind:
        return fallback_kind, bool(oom_failures)
    return failure_kind, bool(oom_failures)


def _encode_ddp_config(config: _DdpRowConfig) -> str:
    payload: JsonObject = {
        "config_path": str(config.config_path),
        "output_dir": str(config.output_dir),
        "data_root": config.data_root,
        "row_spec": pretest._row_spec_payload(config.row_spec),  # noqa: SLF001
        "proof_reference_per_device_batch_size": (
            config.proof_reference_per_device_batch_size
        ),
    }
    return base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")


def _decode_ddp_config(encoded: str) -> _DdpRowConfig:
    payload = cast(
        "JsonObject",
        json.loads(base64.urlsafe_b64decode(encoded.encode("ascii"))),
    )
    return _DdpRowConfig(
        config_path=Path(pretest._required_str(payload, "config_path")),  # noqa: SLF001
        output_dir=Path(pretest._required_str(payload, "output_dir")),  # noqa: SLF001
        data_root=pretest._required_str(payload, "data_root"),  # noqa: SLF001
        proof_reference_per_device_batch_size=pretest._required_int(  # noqa: SLF001
            payload,
            "proof_reference_per_device_batch_size",
        ),
        row_spec=pretest._row_spec_from_payload(  # noqa: SLF001
            pretest._required_object(payload, "row_spec"),  # noqa: SLF001
        ),
    )


def _parse_args(argv: Sequence[str] | None) -> _ChildArgs:
    parser = argparse.ArgumentParser(description="Runtime-selection executor helper.")
    parser.add_argument("--ddp-row")
    namespace = parser.parse_args(argv)
    value = cast("object", namespace.ddp_row)
    if value is not None and not isinstance(value, str):
        message = "Expected optional string argument: ddp_row"
        raise TypeError(message)
    return _ChildArgs(ddp_row=cast("str | None", value))


def _required_object(payload: Mapping[str, JsonValue], key: str) -> JsonObject:
    value = payload.get(key)
    if isinstance(value, dict):
        return cast("JsonObject", value)
    message = f"Expected object field: {key}"
    raise TypeError(message)


def _optional_object(payload: Mapping[str, JsonValue], key: str) -> JsonObject | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, dict):
        return cast("JsonObject", value)
    message = f"Expected optional object field: {key}"
    raise TypeError(message)


def _required_object_list(
    payload: Mapping[str, JsonValue],
    key: str,
) -> list[JsonObject]:
    value = payload.get(key)
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        return [cast("JsonObject", item) for item in value]
    message = f"Expected object-list field: {key}"
    raise TypeError(message)


def _required_str(payload: Mapping[str, JsonValue], key: str) -> str:
    value = payload.get(key)
    if isinstance(value, str):
        return value
    message = f"Expected string field: {key}"
    raise TypeError(message)


def _optional_str(payload: Mapping[str, JsonValue], key: str) -> str | None:
    value = payload.get(key)
    if value is None or isinstance(value, str):
        return value
    message = f"Expected optional string field: {key}"
    raise TypeError(message)


def _optional_bool(
    payload: Mapping[str, JsonValue],
    key: str,
    *,
    default: bool,
) -> bool:
    value = payload.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    message = f"Expected optional boolean field: {key}"
    raise TypeError(message)


def _optional_int(payload: Mapping[str, JsonValue], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    message = f"Expected optional integer field: {key}"
    raise TypeError(message)


def _int_tuple(payload: Mapping[str, JsonValue], key: str) -> tuple[int, ...]:
    value = payload.get(key)
    if isinstance(value, list) and all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        return tuple(cast("list[int]", value))
    message = f"Expected integer list field: {key}"
    raise TypeError(message)


def _str_tuple(payload: Mapping[str, JsonValue], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return tuple(cast("list[str]", value))
    message = f"Expected string list field: {key}"
    raise TypeError(message)


if __name__ == "__main__":
    raise SystemExit(main())
