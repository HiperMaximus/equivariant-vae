# Copyright 2026 HiperMaximus
"""Runtime-selection benchmark artifact plumbing for the v8 shortlist slice."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from eqvae.benchmarking.io import CsvRow, JsonObject, JsonValue, write_csv, write_json
from eqvae.benchmarking.model_count import write_model_count
from eqvae.benchmarking.row_id import (
    DEFAULT_RUNTIME_POLICY_ID,
    compose_row_id_base,
)
from eqvae.benchmarking.runtime_schema import (
    CORRUPTION_CHECK_COLUMNS,
    DATALOADER_MATRIX_COLUMNS,
    EAGER_RECIPE_KNOB_COLUMNS,
    GATE_HEALTH_COLUMNS,
    NUMERICAL_CHECK_COLUMNS,
    RUNTIME_MATRIX_COLUMNS,
    validate_efficiency_proof_reference_batch_size,
)
from eqvae.benchmarking.schedule import training_steps_per_epoch
from eqvae.config import resolve_json_config
from eqvae.data.roots import REAL_TRAIN_PATCH_COUNT

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

RUNTIME_SELECTION_SCHEMA_VERSION = "spec0001.runtime_selection.v1"
RUNTIME_SELECTION_KIND = "kaggle_runtime_selection"
RUNTIME_SELECTION_SOURCE = "kaggle_runtime_benchmark"
LOCAL_SELECTION_SOURCE = "local_runtime_selection_schema_proof"
MODEL_COUNT_FILENAME = "model_count.json"
RUNTIME_PROOF_FILENAME = "runtime_proof.json"
RUNTIME_MATRIX_FILENAME = "runtime_matrix.csv"
DATALOADER_MATRIX_FILENAME = "dataloader_matrix.csv"
NUMERICAL_CHECKS_FILENAME = "numerical_checks.csv"
CORRUPTION_CHECKS_FILENAME = "corruption_checks.csv"
GATE_HEALTH_FILENAME = "gate_health.csv"
GATE_HEALTH_SUMMARY_FILENAME = "gate_health_summary.json"
SELECTED_RUNTIME_FILENAME = "selected_runtime.json"
STAIN_CORRUPTOR_QA_FILENAME = "stain_corruptor_qa.json"
REQUIRED_DATALOADER_SPLITS = ("train", "validation")
SINGLE_VISIBLE_T4 = "single_visible_t4"
DUAL_T4_DDP = "dual_t4_ddp"
AMP_OFF_FP32 = "amp_off_fp32"
AMP_CONSERVATIVE = "amp_conservative"
AMP_SCALAR_GATE_RELAXED = "amp_scalar_gate_relaxed"
COMPILE_NONE = "none"
COMPILE_MODEL_FORWARD = "model_forward"
COMPILE_STEP = "step"
BRANCHLESS_ALL = "branchless_all"
INDEXED_MASKED = "indexed_masked"
PASS_STATUS = "pass"  # noqa: S105
FAIL_STATUS = "fail"
INELIGIBLE_STATUS = "ineligible"
SKIPPED_UNSUPPORTED = "skipped_unsupported"
EXPECTED_MACHINE_SHAPE = "NvidiaTeslaT4"
EXPECTED_DUAL_T4_COUNT = 2
MAX_DATA_WAIT_FRACTION = 0.20
MIN_LOADER_TRAINER_THROUGHPUT_RATIO = 1.25
# DEFAULT_RUNTIME_POLICY_ID is single-sourced in `row_id.py` (imported above) and
# re-exported here for the existing importers.
V3_BASELINE_RUNTIME_POLICY_ID = "v3_fp32_eager_baseline"
DEFAULT_MATERIAL_SPEEDUP_FRACTION = 0.03
REQUIRED_COMPILE_SETTLE_STEPS = 5
# Compiled scopes eligible for selection once they pass the settle-proof relationship
# (Spec 0011 S12). Whole-step compile is the measured winner recipe's scope.
_STABLE_COMPILE_SCOPES = frozenset({COMPILE_MODEL_FORWARD, COMPILE_STEP})
REQUIRED_NUMERICAL_BATCH_INDICES = frozenset({"0", "1", "2"})
REQUIRED_CORRUPTION_SPLITS = frozenset({"train", "validation"})
MAX_RELAXED_LOSS_ABS_DELTA = 1.0e-2
MAX_RELAXED_LOSS_REL_DELTA = 1.0e-2
MAX_RELAXED_GRAD_REL_DELTA = 1.0e-2
MAX_RELAXED_PARAM_UPDATE_REL_DELTA = 1.0e-2
MAX_RELAXED_STATE_ABS_DELTA = 1.0e-2
RELAXED_NUMERICAL_DELTA_FAILURE_KIND = "dual_t4_numerical_delta_failed"
STAIN_QA_PROOF_SCOPE = "selected_runtime_stain_corruptor_row_linked_qa"
REAL_TRAIN_PATCH_COUNT_DEFAULT = REAL_TRAIN_PATCH_COUNT
V8_REQUIRED_ARTIFACTS = (
    "benchmark/runtime_proof.json",
    "benchmark/runtime_matrix.csv",
    "benchmark/dataloader_matrix.csv",
    "benchmark/numerical_checks.csv",
    "benchmark/corruption_checks.csv",
    "benchmark/gate_health_summary.json",
    "metrics/gate_health.csv",
)
V8_REQUIRED_SHORTLIST_ROWS = (
    "single_visible_t4__bs4__amp_off_fp32__compile_none__branchless_all",
    "single_visible_t4__bs4__amp_off_fp32__compile_none__indexed_masked",
    "single_visible_t4__bs8__amp_off_fp32__compile_none__branchless_all",
    "single_visible_t4__bs8__amp_off_fp32__compile_none__indexed_masked",
    "single_visible_t4__bs12__amp_off_fp32__compile_none__branchless_all",
    "single_visible_t4__bs12__amp_off_fp32__compile_none__indexed_masked",
)


@dataclass(frozen=True)
class RuntimeSelectionBenchmarkRequest:
    """Inputs for the selected-runtime benchmark artifact writer."""

    config_path: Path
    output_dir: Path
    run_name: str | None = None
    v8_artifact_dir: Path | None = None
    evidence: RuntimeSelectionEvidence | None = None


@dataclass(frozen=True)
class RuntimeSelectionEvidence:
    """Explicit evidence supplied by an actual runtime-selection executor."""

    runtime_rows: tuple[CsvRow, ...]
    dataloader_rows: tuple[CsvRow, ...]
    numerical_rows: tuple[CsvRow, ...]
    corruption_rows: tuple[CsvRow, ...]
    gate_health_rows: tuple[CsvRow, ...]
    gate_health_summary: JsonObject
    runtime_environment: JsonObject


@dataclass(frozen=True)
class RuntimeSelectionArtifactPaths:
    """Paths written by the selected-runtime benchmark path."""

    model_count: Path
    runtime_proof: Path
    runtime_matrix: Path
    dataloader_matrix: Path
    numerical_checks: Path
    corruption_checks: Path
    gate_health: Path
    gate_health_summary: Path
    selected_runtime: Path | None


def load_runtime_selection_evidence(artifact_dir: Path) -> RuntimeSelectionEvidence:
    """Load downloaded runtime-selection evidence for local writer replay.

    Args:
        artifact_dir: Root directory containing `benchmark/` and `metrics/`.

    Returns:
        Evidence rows and runtime environment from the downloaded artifacts.

    Raises:
        TypeError: If `runtime_proof.json` does not contain runtime environment.

    """
    benchmark_dir = artifact_dir / "benchmark"
    metrics_dir = artifact_dir / "metrics"
    runtime_proof = _load_json(benchmark_dir / RUNTIME_PROOF_FILENAME)
    runtime_environment = runtime_proof.get("runtime_environment")
    if not isinstance(runtime_environment, dict):
        message = "runtime_proof.json must contain runtime_environment"
        raise TypeError(message)
    return RuntimeSelectionEvidence(
        runtime_rows=tuple(_load_csv(benchmark_dir / RUNTIME_MATRIX_FILENAME)),
        dataloader_rows=tuple(_load_csv(benchmark_dir / DATALOADER_MATRIX_FILENAME)),
        numerical_rows=tuple(_load_csv(benchmark_dir / NUMERICAL_CHECKS_FILENAME)),
        corruption_rows=tuple(_load_csv(benchmark_dir / CORRUPTION_CHECKS_FILENAME)),
        gate_health_rows=tuple(_load_csv(metrics_dir / GATE_HEALTH_FILENAME)),
        gate_health_summary=_load_json(benchmark_dir / GATE_HEALTH_SUMMARY_FILENAME),
        runtime_environment=cast("JsonObject", runtime_environment),
    )


@dataclass(frozen=True)
class _EfficiencyPolicySelection:
    runtime_policy_id: str
    precision_policy: str
    compile_scope: str
    diagnostic_only: bool


@dataclass(frozen=True)
class _SelectionSettings:
    run_name: str
    effective_config_hash: str
    real_train_patch_count: int
    warmup_steps: int
    measured_steps: int
    repeats: int
    v8_artifact_dir: Path
    fp32_batch_sizes: tuple[int, ...]
    fallback_batch_sizes: tuple[int, ...]
    dual_batch_sizes: tuple[int, ...]
    corruption_strategies: tuple[str, ...]
    baseline_selected_runtime_path: Path | None
    baseline_selected_row_id: str
    baseline_runtime_policy_id: str
    minimum_material_speedup_fraction: float
    efficiency_accelerator_modes: tuple[str, ...]
    efficiency_batch_sizes: tuple[int, ...]
    efficiency_proof_reference_batch_size: int
    efficiency_corruption_strategies: tuple[str, ...]
    efficiency_policies: tuple[_EfficiencyPolicySelection, ...]


def write_runtime_selection_benchmark(  # noqa: PLR0914
    request: RuntimeSelectionBenchmarkRequest,
) -> RuntimeSelectionArtifactPaths:
    """Write the v8-shortlist selected-runtime benchmark artifact graph.

    The default local path is intentionally fail-closed: it records v8 hashes and
    local schema/proof plumbing, but it does not write selected_runtime.json
    unless a real evidence provider supplies passing dual-T4 timing and all
    linked safety proofs.

    Returns:
        Paths for the artifacts written by the selected-runtime benchmark.

    """
    resolved = resolve_json_config(request.config_path)
    settings = _settings(
        request=request,
        effective=resolved.effective_config,
        effective_config_hash=resolved.effective_config_hash,
    )
    benchmark_dir = request.output_dir / "benchmark"
    metrics_dir = request.output_dir / "metrics"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    v8_provenance = _v8_provenance_payload(settings.v8_artifact_dir)
    evidence = request.evidence or _blocked_local_evidence(settings=settings)
    runtime_rows = _enforce_compiled_rows_diagnostic_only(evidence.runtime_rows)
    amp_policy = _amp_followup_policy(settings=settings, rows=runtime_rows)
    dual_gate = _dual_gate_payload(
        settings=settings,
        runtime_rows=runtime_rows,
        dataloader_rows=evidence.dataloader_rows,
        numerical_rows=evidence.numerical_rows,
        corruption_rows=evidence.corruption_rows,
        gate_health_rows=evidence.gate_health_rows,
        gate_health_summary=evidence.gate_health_summary,
        runtime_environment=evidence.runtime_environment,
    )

    model_count_path = benchmark_dir / MODEL_COUNT_FILENAME
    model_count_payload = write_model_count(
        config_path=request.config_path,
        output_path=model_count_path,
    )
    write_csv(
        benchmark_dir / RUNTIME_MATRIX_FILENAME,
        RUNTIME_MATRIX_COLUMNS,
        runtime_rows,
    )
    write_csv(
        benchmark_dir / DATALOADER_MATRIX_FILENAME,
        DATALOADER_MATRIX_COLUMNS,
        evidence.dataloader_rows,
    )
    write_csv(
        benchmark_dir / NUMERICAL_CHECKS_FILENAME,
        NUMERICAL_CHECK_COLUMNS,
        evidence.numerical_rows,
    )
    write_csv(
        benchmark_dir / CORRUPTION_CHECKS_FILENAME,
        CORRUPTION_CHECK_COLUMNS,
        evidence.corruption_rows,
    )
    write_csv(
        metrics_dir / GATE_HEALTH_FILENAME,
        GATE_HEALTH_COLUMNS,
        evidence.gate_health_rows,
    )
    write_json(
        benchmark_dir / GATE_HEALTH_SUMMARY_FILENAME,
        evidence.gate_health_summary,
    )

    decision = _selected_runtime_write_decision(
        settings=settings,
        runtime_rows=runtime_rows,
        dataloader_rows=evidence.dataloader_rows,
        numerical_rows=evidence.numerical_rows,
        corruption_rows=evidence.corruption_rows,
        gate_health_summary=evidence.gate_health_summary,
        gate_health_rows=evidence.gate_health_rows,
        model_count_payload=model_count_payload,
        dual_gate=dual_gate,
        amp_policy=amp_policy,
        v8_provenance=v8_provenance,
        stain_corruptor_qa=_stain_corruptor_qa_payload(
            benchmark_dir / STAIN_CORRUPTOR_QA_FILENAME,
        ),
    )
    if not decision["allowed"]:
        _reject_stale_selected_runtime(benchmark_dir)

    runtime_proof = _runtime_proof_payload(
        settings=settings,
        runtime_rows=runtime_rows,
        v8_provenance=v8_provenance,
        runtime_environment=evidence.runtime_environment,
        dual_gate=dual_gate,
        amp_policy=amp_policy,
        decision=decision,
        model_count_payload=model_count_payload,
        stain_corruptor_qa=_stain_corruptor_qa_payload(
            benchmark_dir / STAIN_CORRUPTOR_QA_FILENAME,
        ),
    )
    runtime_proof_path = benchmark_dir / RUNTIME_PROOF_FILENAME
    write_json(runtime_proof_path, runtime_proof)

    selected_runtime_path: Path | None = None
    if decision["allowed"]:
        selected_row = _selected_row(settings=settings, rows=runtime_rows)
        selected_runtime_path = benchmark_dir / SELECTED_RUNTIME_FILENAME
        write_json(
            selected_runtime_path,
            _selected_runtime_payload(
                settings=settings,
                selected_row=selected_row,
                dataloader_rows=evidence.dataloader_rows,
                artifact_hashes=_artifact_hashes(request.output_dir),
            ),
        )

    return RuntimeSelectionArtifactPaths(
        model_count=model_count_path,
        runtime_proof=runtime_proof_path,
        runtime_matrix=benchmark_dir / RUNTIME_MATRIX_FILENAME,
        dataloader_matrix=benchmark_dir / DATALOADER_MATRIX_FILENAME,
        numerical_checks=benchmark_dir / NUMERICAL_CHECKS_FILENAME,
        corruption_checks=benchmark_dir / CORRUPTION_CHECKS_FILENAME,
        gate_health=metrics_dir / GATE_HEALTH_FILENAME,
        gate_health_summary=benchmark_dir / GATE_HEALTH_SUMMARY_FILENAME,
        selected_runtime=selected_runtime_path,
    )


def _settings(
    *,
    request: RuntimeSelectionBenchmarkRequest,
    effective: JsonObject,
    effective_config_hash: str,
) -> _SelectionSettings:
    run = _required_object(effective, "run")
    data = _required_object(effective, "data")
    runtime = _required_object(effective, "runtime_matrix")
    selection = _required_object(runtime, "selection_benchmark_slice")
    stages = _required_object_list(selection, "stages")
    first_stage = _stage(stages, "v8_shortlist_fp32_eager_confirmation")
    dual_stage = _stage(stages, "dual_t4_train_step_gate")
    efficiency = _optional_object(selection, "efficiency_followup") or {}
    dual_batch_sizes = _int_tuple(dual_stage, "per_device_batch_sizes")
    efficiency_batch_sizes = (
        () if not efficiency else _int_tuple(efficiency, "per_device_batch_sizes")
    )
    efficiency_proof_reference_batch_size = (
        0
        if not efficiency
        else validate_efficiency_proof_reference_batch_size(
            batch_size=_required_int(
                efficiency,
                "proof_reference_per_device_batch_size",
            ),
            dual_gate_batch_sizes=dual_batch_sizes,
            efficiency_batch_sizes=efficiency_batch_sizes,
        )
    )
    v8_carry_forward = _required_object(runtime, "v8_carry_forward")
    v8_dir_value = request.v8_artifact_dir or Path(
        _required_str(v8_carry_forward, "artifact_dir"),
    )
    baseline_path = _optional_str(efficiency, "baseline_selected_runtime")
    return _SelectionSettings(
        run_name=request.run_name or _required_str(run, "name"),
        effective_config_hash=effective_config_hash,
        real_train_patch_count=_optional_int(data, "real_train_patch_count")
        or REAL_TRAIN_PATCH_COUNT_DEFAULT,
        warmup_steps=_required_int(runtime, "warmup_steps"),
        measured_steps=_required_int(runtime, "measured_steps"),
        repeats=_required_int(runtime, "repeats"),
        v8_artifact_dir=v8_dir_value,
        fp32_batch_sizes=_int_tuple(first_stage, "per_device_batch_sizes"),
        fallback_batch_sizes=_int_tuple(first_stage, "fallback_per_device_batch_sizes"),
        dual_batch_sizes=dual_batch_sizes,
        corruption_strategies=_str_tuple(dual_stage, "corruption_strategies"),
        baseline_selected_runtime_path=None
        if not baseline_path
        else _resolve_config_relative_path(
            config_path=request.config_path,
            configured_path=baseline_path,
        ),
        baseline_selected_row_id=_optional_str(efficiency, "baseline_row_id")
        or "dual_t4_ddp__bs12__amp_off_fp32__compile_none__indexed_masked",
        baseline_runtime_policy_id=_optional_str(
            efficiency,
            "baseline_runtime_policy_id",
        )
        or V3_BASELINE_RUNTIME_POLICY_ID,
        minimum_material_speedup_fraction=_optional_float(
            efficiency,
            "minimum_material_speedup_fraction",
        )
        or DEFAULT_MATERIAL_SPEEDUP_FRACTION,
        efficiency_accelerator_modes=()
        if not efficiency
        else _str_tuple(efficiency, "accelerator_modes"),
        efficiency_batch_sizes=efficiency_batch_sizes,
        efficiency_proof_reference_batch_size=(efficiency_proof_reference_batch_size),
        efficiency_corruption_strategies=()
        if not efficiency
        else _str_tuple(efficiency, "corruption_strategies"),
        efficiency_policies=()
        if not efficiency
        else _efficiency_policy_selections(
            _required_object_list(efficiency, "policies"),
        ),
    )


def _efficiency_policy_selections(
    policies: Sequence[JsonObject],
) -> tuple[_EfficiencyPolicySelection, ...]:
    return tuple(
        _EfficiencyPolicySelection(
            runtime_policy_id=_required_str(policy, "runtime_policy_id"),
            precision_policy=_required_str(policy, "precision_policy"),
            compile_scope=_required_str(policy, "compile_scope"),
            diagnostic_only=_optional_bool_or_default(
                policy,
                "diagnostic_only",
                default=False,
            ),
        )
        for policy in policies
    )


def _resolve_config_relative_path(*, config_path: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    if path.is_absolute():
        return path
    resolved_config = config_path.resolve()
    for parent in resolved_config.parents:
        candidate = parent / path
        if candidate.exists():
            return candidate
    if (
        resolved_config.parent.name == "spec0001"
        and resolved_config.parent.parent.name == "configs"
    ):
        return resolved_config.parent.parent.parent / path
    return Path.cwd() / path


def _stage(stages: Sequence[JsonObject], name: str) -> JsonObject:
    for stage in stages:
        if stage.get("name") == name:
            return stage
    message = f"Missing selection stage: {name}"
    raise ValueError(message)


def _v8_provenance_payload(v8_artifact_dir: Path) -> JsonObject:
    hashes: JsonObject = {}
    missing: list[str] = []
    for relative in V8_REQUIRED_ARTIFACTS:
        path = v8_artifact_dir / relative
        if path.exists():
            hashes[relative] = _sha256_file(path)
        else:
            missing.append(relative)
    runtime_proof = _load_json(v8_artifact_dir / "benchmark" / RUNTIME_PROOF_FILENAME)
    runtime_rows = _load_csv(v8_artifact_dir / "benchmark" / RUNTIME_MATRIX_FILENAME)
    rows_by_id = {row["row_id"]: row for row in runtime_rows}
    missing_rows = [
        row_id
        for row_id in V8_REQUIRED_SHORTLIST_ROWS
        if rows_by_id.get(row_id, {}).get("status") != PASS_STATUS
    ]
    source_is_non_promotable = (
        runtime_proof.get("status") == "pretest_incomplete"
        and runtime_proof.get("full_run_eligible") is False
        and runtime_proof.get("selected_runtime_written") is False
    )
    status = (
        PASS_STATUS
        if not missing and not missing_rows and source_is_non_promotable
        else FAIL_STATUS
    )
    return cast(
        "JsonObject",
        {
            "status": status,
            "source_artifact_dir": str(v8_artifact_dir),
            "used_for": "candidate_shortlist_only",
            "v8_artifacts_are_promotable": False,
            "source_status": _json_value(runtime_proof.get("status")),
            "source_full_run_eligible": _json_value(
                runtime_proof.get("full_run_eligible"),
            ),
            "source_selected_runtime_written": _json_value(
                runtime_proof.get("selected_runtime_written"),
            ),
            "source_eligible_pass_row_count": _json_value(
                runtime_proof.get("eligible_pass_row_count"),
            ),
            "artifact_hashes": hashes,
            "missing_artifacts": missing,
            "required_shortlist_rows": list(V8_REQUIRED_SHORTLIST_ROWS),
            "missing_or_nonpassing_shortlist_rows": missing_rows,
            "promotion_policy": "hash_provenance_only_do_not_promote_v8_rows",
        },
    )


def _blocked_local_evidence(
    *,
    settings: _SelectionSettings,
) -> RuntimeSelectionEvidence:
    rows = _default_runtime_rows(settings)
    return RuntimeSelectionEvidence(
        runtime_rows=tuple(rows),
        dataloader_rows=tuple(
            _linked_rows_for_runtime(
                settings=settings,
                runtime_rows=rows,
                columns=DATALOADER_MATRIX_COLUMNS,
                status=SKIPPED_UNSUPPORTED,
                failure_kind="local_runtime_selection_dataloader_not_measured",
            ),
        ),
        numerical_rows=tuple(
            _linked_rows_for_runtime(
                settings=settings,
                runtime_rows=rows,
                columns=NUMERICAL_CHECK_COLUMNS,
                status=SKIPPED_UNSUPPORTED,
                failure_kind="local_runtime_selection_numerical_not_measured",
            ),
        ),
        corruption_rows=tuple(
            _linked_rows_for_runtime(
                settings=settings,
                runtime_rows=rows,
                columns=CORRUPTION_CHECK_COLUMNS,
                status=SKIPPED_UNSUPPORTED,
                failure_kind="local_runtime_selection_corruption_not_measured",
            ),
        ),
        gate_health_rows=(),
        gate_health_summary={
            "status": SKIPPED_UNSUPPORTED,
            "benchmark_kind": RUNTIME_SELECTION_KIND,
            "benchmark_source": LOCAL_SELECTION_SOURCE,
            "overall_status": SKIPPED_UNSUPPORTED,
            "full_run_eligible": False,
            "logged_intervals": 0,
            "module_count": 0,
            "nonfinite_count": None,
            "failing_modules": [],
            "warning_modules": [],
            "notes": "Local schema path did not measure gate health.",
        },
        runtime_environment={
            "status": SKIPPED_UNSUPPORTED,
            "machine_shape": "local_cpu",
            "visible_device_count": 0,
            "cuda_device_count": 0,
            "gpu_names": [],
            "world_size": 1,
            "nproc_per_node": 1,
            "rank_assignments": [],
        },
    )


def _default_runtime_rows(settings: _SelectionSettings) -> list[CsvRow]:
    rows: list[CsvRow] = []
    rows.extend(
        _runtime_row(
            settings=settings,
            row_id=_row_id(
                accelerator_mode=SINGLE_VISIBLE_T4,
                batch_size=batch_size,
                precision_policy=AMP_OFF_FP32,
                compile_scope=COMPILE_NONE,
                corruption_strategy=corruption_strategy,
            ),
            accelerator_mode=SINGLE_VISIBLE_T4,
            per_device_batch_size=batch_size,
            precision_policy=AMP_OFF_FP32,
            compile_scope=COMPILE_NONE,
            corruption_strategy=corruption_strategy,
            world_size=1,
            nproc_per_node=1,
            status=SKIPPED_UNSUPPORTED,
            failure_kind="local_runtime_selection_single_t4_confirmation_not_run",
        )
        for batch_size in (*settings.fp32_batch_sizes, *settings.fallback_batch_sizes)
        for corruption_strategy in settings.corruption_strategies
    )
    rows.extend(
        _runtime_row(
            settings=settings,
            row_id=_row_id(
                accelerator_mode=DUAL_T4_DDP,
                batch_size=batch_size,
                precision_policy=AMP_OFF_FP32,
                compile_scope=COMPILE_NONE,
                corruption_strategy=corruption_strategy,
            ),
            accelerator_mode=DUAL_T4_DDP,
            per_device_batch_size=batch_size,
            precision_policy=AMP_OFF_FP32,
            compile_scope=COMPILE_NONE,
            corruption_strategy=corruption_strategy,
            world_size=EXPECTED_DUAL_T4_COUNT,
            nproc_per_node=EXPECTED_DUAL_T4_COUNT,
            status=SKIPPED_UNSUPPORTED,
            failure_kind="missing_real_dual_t4_train_step_timing",
        )
        for batch_size in settings.dual_batch_sizes
        for corruption_strategy in settings.corruption_strategies
    )
    return rows


def _runtime_row(  # noqa: PLR0913
    *,
    settings: _SelectionSettings,
    row_id: str,
    accelerator_mode: str,
    per_device_batch_size: int,
    precision_policy: str,
    compile_scope: str,
    corruption_strategy: str,
    world_size: int,
    nproc_per_node: int,
    status: str,
    failure_kind: str = "",
    samples_sec: float | None = None,
    steady_step_ms_p50: float | None = None,
    steady_step_ms_p95: float | None = None,
) -> CsvRow:
    global_batch_size = per_device_batch_size * world_size
    return {
        "run_name": settings.run_name,
        "benchmark_kind": RUNTIME_SELECTION_KIND,
        "benchmark_source": RUNTIME_SELECTION_SOURCE,
        "full_run_eligible": _bool_text(status == PASS_STATUS),
        "row_id": row_id,
        "accelerator_mode": accelerator_mode,
        "machine_shape": EXPECTED_MACHINE_SHAPE,
        "visible_device_count": str(world_size if status == PASS_STATUS else 0),
        "cuda_device_count": str(world_size if status == PASS_STATUS else 0),
        "gpu_names": json.dumps(
            ["Tesla T4"] * world_size if status == PASS_STATUS else [],
        ),
        "ddp_backend": "nccl" if accelerator_mode == DUAL_T4_DDP else "",
        "world_size": str(world_size),
        "nproc_per_node": str(nproc_per_node),
        "precision_policy": precision_policy,
        "amp_enabled": _bool_text(precision_policy != AMP_OFF_FP32),
        "torch_compile_enabled": _bool_text(compile_scope != COMPILE_NONE),
        "compile_scope": compile_scope,
        "runtime_policy_id": DEFAULT_RUNTIME_POLICY_ID,
        "memory_format": "contiguous",
        "autocast_dtype": "",
        "fp32_loss": "true",
        "grad_scaler_enabled": _bool_text(precision_policy != AMP_OFF_FP32),
        "cudnn_benchmark": "false",
        "cudnn_deterministic": "false",
        "deterministic_algorithms": "false",
        "tf32_enabled": "false",
        "matmul_precision": "highest",
        "ddp_static_graph": "false",
        "ddp_gradient_as_bucket_view": "false",
        "optimizer_implementation": "adamw_default",
        "zero_grad_set_to_none": "true",
        "gradient_clip_foreach": "true",
        "compile_dynamic": "false",
        # Spec 0011 S13: eager recipe knobs (local/default rows carry no measured
        # recipe; S14's real dual-T4 search sources these on the winner row).
        **EAGER_RECIPE_KNOB_COLUMNS,
        "corruption_strategy": corruption_strategy,
        "per_device_batch_size": str(per_device_batch_size),
        "global_batch_size": str(global_batch_size),
        "gradient_accumulation_steps": "1",
        "warmup_steps": str(settings.warmup_steps),
        "measured_steps": str(settings.measured_steps),
        "repeats": str(settings.repeats),
        "compile_startup_sec": "0.000000",
        "compile_settle_steps": "0" if compile_scope == COMPILE_NONE else "5",
        "steady_step_ms_p50": _float_text(steady_step_ms_p50),
        "steady_step_ms_p95": _float_text(steady_step_ms_p95),
        "samples_sec": _float_text(samples_sec),
        "trainer_samples_sec": _float_text(samples_sec),
        "max_vram_allocated_mb": "",
        "max_vram_reserved_mb": "",
        "vram_headroom_fraction": "0.500000" if status == PASS_STATUS else "",
        "amp_step_skipped_count": "0" if status == PASS_STATUS else "",
        "gate_health_status": PASS_STATUS
        if status == PASS_STATUS
        else SKIPPED_UNSUPPORTED,
        "gate_health_warning_count": "0" if status == PASS_STATUS else "",
        "numerical_check_status": PASS_STATUS
        if status == PASS_STATUS
        else SKIPPED_UNSUPPORTED,
        "data_wait_fraction_p95": "0.010000" if status == PASS_STATUS else "",
        "graph_break_count": "0" if compile_scope == COMPILE_NONE else "",
        "recompile_count": "0" if compile_scope == COMPILE_NONE else "",
        "oom": "false",
        "status": status,
        "failure_kind": failure_kind,
        "failure_message_hash": "" if not failure_kind else _hash_text(failure_kind),
    }


def _linked_rows_for_runtime(
    *,
    settings: _SelectionSettings,
    runtime_rows: Sequence[CsvRow],
    columns: Sequence[str],
    status: str,
    failure_kind: str,
) -> list[CsvRow]:
    rows: list[CsvRow] = []
    for runtime_row in runtime_rows:
        rank_count = 1
        splits = ("train",)
        if tuple(columns) == DATALOADER_MATRIX_COLUMNS:
            rank_count = max(1, int(runtime_row["world_size"]))
            splits = REQUIRED_DATALOADER_SPLITS
        rows.extend(
            _linked_row(
                settings=settings,
                runtime_row=runtime_row,
                columns=columns,
                status=status,
                failure_kind=failure_kind,
                rank=rank,
                split=split,
            )
            for rank in range(rank_count)
            for split in splits
        )
    return rows


def _linked_row(  # noqa: C901, PLR0912, PLR0913
    *,
    settings: _SelectionSettings,
    runtime_row: CsvRow,
    columns: Sequence[str],
    status: str,
    failure_kind: str,
    rank: int,
    split: str,
) -> CsvRow:
    row: dict[str, str] = dict.fromkeys(columns, "")
    shared = {
        "run_name": runtime_row["run_name"],
        "benchmark_kind": RUNTIME_SELECTION_KIND,
        "benchmark_source": RUNTIME_SELECTION_SOURCE,
        "full_run_eligible": "false",
        "accelerator_mode": runtime_row["accelerator_mode"],
        "machine_shape": runtime_row["machine_shape"],
        "row_id": runtime_row["row_id"],
        "status": status,
        "failure_kind": failure_kind,
    }
    for key, value in shared.items():
        if key in row:
            row[key] = value
    if "candidate_row_id" in row:
        row["candidate_row_id"] = runtime_row["row_id"]
    if "reference_row_id" in row:
        row["reference_row_id"] = _reference_row_id(
            settings=settings,
            runtime_row=runtime_row,
        )
    if "batch_index" in row:
        row["batch_index"] = "0"
    if "precision_policy" in row:
        row["precision_policy"] = runtime_row["precision_policy"]
    if "torch_compile_enabled" in row:
        row["torch_compile_enabled"] = runtime_row["torch_compile_enabled"]
    if "compile_scope" in row:
        row["compile_scope"] = runtime_row["compile_scope"]
    if "corruption_strategy" in row:
        row["corruption_strategy"] = runtime_row["corruption_strategy"]
    if "gate_health_status" in row:
        row["gate_health_status"] = (
            PASS_STATUS if status == PASS_STATUS else SKIPPED_UNSUPPORTED
        )
    if "world_size" in row:
        row["world_size"] = runtime_row["world_size"]
    if "runtime_policy_id" in row:
        row["runtime_policy_id"] = _runtime_policy_id(runtime_row)
    if "memory_format" in row:
        row["memory_format"] = runtime_row.get("memory_format", "contiguous")
    if "rank" in row:
        row["rank"] = str(rank)
    if "split" in row:
        row["split"] = split
    if "batch_size" in row:
        row["batch_size"] = runtime_row["per_device_batch_size"]
    return row


def _reference_row_id(*, settings: _SelectionSettings, runtime_row: CsvRow) -> str:
    candidate_batch_size = int(runtime_row["per_device_batch_size"])
    proof_batch_size = (
        settings.efficiency_proof_reference_batch_size
        if _is_efficiency_runtime_row(settings=settings, row=runtime_row)
        else candidate_batch_size
    )
    return _row_id(
        accelerator_mode=runtime_row["accelerator_mode"],
        batch_size=min(candidate_batch_size, proof_batch_size),
        precision_policy=AMP_OFF_FP32,
        compile_scope=COMPILE_NONE,
        corruption_strategy=BRANCHLESS_ALL,
    )


def _is_efficiency_runtime_row(
    *,
    settings: _SelectionSettings,
    row: Mapping[str, str],
) -> bool:
    policy_ids = {policy.runtime_policy_id for policy in settings.efficiency_policies}
    return (
        _runtime_policy_id(row) in policy_ids
        and row.get("accelerator_mode") in settings.efficiency_accelerator_modes
        and _optional_csv_int(row.get("per_device_batch_size", ""))
        in settings.efficiency_batch_sizes
        and row.get("corruption_strategy") in settings.efficiency_corruption_strategies
    )


def _enforce_compiled_rows_diagnostic_only(
    rows: Sequence[CsvRow],
) -> tuple[CsvRow, ...]:
    normalized: list[CsvRow] = []
    for row in rows:
        if (
            row["compile_scope"] == COMPILE_NONE
            or row["status"] != PASS_STATUS
            or _compiled_row_stable(row)
        ):
            normalized.append(row)
            continue
        copied = dict(row)
        copied["status"] = INELIGIBLE_STATUS
        copied["full_run_eligible"] = "false"
        copied["failure_kind"] = (
            "compiled_rows_diagnostic_only_until_stable_settle_proof"
        )
        copied["failure_message_hash"] = _hash_text(copied["failure_kind"])
        normalized.append(copied)
    return tuple(normalized)


def _compiled_row_stable(row: CsvRow) -> bool:
    if row["compile_scope"] == COMPILE_NONE:
        return True
    settle_steps = _optional_csv_int(row.get("compile_settle_steps", ""))
    graph_breaks = _optional_csv_int(row.get("graph_break_count", ""))
    recompiles = _optional_csv_int(row.get("recompile_count", ""))
    optimize_ddp = row.get("optimize_ddp", "")
    return (
        row["compile_scope"] in _STABLE_COMPILE_SCOPES
        and _runtime_policy_id(row) != DEFAULT_RUNTIME_POLICY_ID
        and optimize_ddp
        in {
            "ddp_optimizer",
            "python_reducer",
            "python_reducer_without_compiled_forward",
            "no_optimization",
        }
        and settle_steps is not None
        and settle_steps >= REQUIRED_COMPILE_SETTLE_STEPS
        # Graph breaks/partitions are measured telemetry, not a universal failure.
        # DDPOptimizer deliberately partitions around bucket boundaries and current
        # reducer modes may expose different stable overlap structures. Availability
        # still fails closed; only post-settle recompilation is disqualifying.
        and graph_breaks is not None
        and graph_breaks >= 0
        and recompiles == 0
    )


def _amp_followup_policy(
    *,
    settings: _SelectionSettings,
    rows: Sequence[CsvRow],
) -> JsonObject:
    pass_fp32_row_ids = {
        row["row_id"]
        for row in rows
        if row["status"] == PASS_STATUS
        and row["precision_policy"] == AMP_OFF_FP32
        and row["compile_scope"] == COMPILE_NONE
        and row["corruption_strategy"] == BRANCHLESS_ALL
    }
    amp_rows = [
        row
        for row in rows
        if row["precision_policy"] in {AMP_CONSERVATIVE, AMP_SCALAR_GATE_RELAXED}
    ]
    violations = [
        row["row_id"]
        for row in amp_rows
        if _reference_row_id(settings=settings, runtime_row=row)
        not in pass_fp32_row_ids
    ]
    skipped_rows = [
        row["row_id"]
        for row in amp_rows
        if (_optional_csv_int(row.get("amp_step_skipped_count", "")) or 0) > 0
    ]
    # AMP skips are row-level catastrophic blockers. They must prevent that row
    # from selection, but they should not reject a different safe row or the v3
    # baseline remeasure.
    return cast(
        "JsonObject",
        {
            "status": PASS_STATUS if not violations else FAIL_STATUS,
            "confirmed_fp32_eager_row_count": len(pass_fp32_row_ids),
            "amp_followup_row_count": len(amp_rows),
            "violation_row_ids": violations,
            "amp_skipped_row_ids": skipped_rows,
            "policy": "amp_followup_only_after_exact_fp32_eager_reference_row",
        },
    )


def _runtime_policy_id(row: Mapping[str, str]) -> str:
    return row.get("runtime_policy_id", "") or DEFAULT_RUNTIME_POLICY_ID


def _runtime_policy_matches(row: Mapping[str, str], runtime_row: CsvRow) -> bool:
    value = row.get("runtime_policy_id", "")
    return not value or value == _runtime_policy_id(runtime_row)


def _dual_gate_payload(  # noqa: PLR0913
    *,
    settings: _SelectionSettings,
    runtime_rows: Sequence[CsvRow],
    dataloader_rows: Sequence[CsvRow],
    numerical_rows: Sequence[CsvRow],
    corruption_rows: Sequence[CsvRow],
    gate_health_rows: Sequence[CsvRow],
    gate_health_summary: JsonObject,
    runtime_environment: JsonObject,
) -> JsonObject:
    dual_rows = [
        row
        for row in runtime_rows
        if row["accelerator_mode"] == DUAL_T4_DDP
        and row["precision_policy"] == AMP_OFF_FP32
        and row["compile_scope"] == COMPILE_NONE
    ]
    required_row_ids = [
        _row_id(
            accelerator_mode=DUAL_T4_DDP,
            batch_size=batch_size,
            precision_policy=AMP_OFF_FP32,
            compile_scope=COMPILE_NONE,
            corruption_strategy=corruption_strategy,
        )
        for batch_size in settings.dual_batch_sizes
        for corruption_strategy in settings.corruption_strategies
    ]
    rows_by_id = {row["row_id"]: row for row in dual_rows}
    missing_rows = [row_id for row_id in required_row_ids if row_id not in rows_by_id]
    nonpassing_rows = [
        row["row_id"] for row in dual_rows if row["status"] != PASS_STATUS
    ]
    required_rows = [
        rows_by_id[row_id] for row_id in required_row_ids if row_id in rows_by_id
    ]
    linked_failures = _linked_failures(
        settings=settings,
        required_rows=required_rows,
        dataloader_rows=dataloader_rows,
        numerical_rows=numerical_rows,
        corruption_rows=corruption_rows,
        gate_health_rows=gate_health_rows,
        gate_health_summary=gate_health_summary,
    )
    gpu_names = _str_list_value(runtime_environment, "gpu_names")
    rank_assignments = _object_list_value(runtime_environment, "rank_assignments")
    child_launch_pass = _child_launch_pass(runtime_environment)
    runtime_pass = (
        _int_value(runtime_environment, "visible_device_count")
        == EXPECTED_DUAL_T4_COUNT
        and _int_value(runtime_environment, "cuda_device_count")
        == EXPECTED_DUAL_T4_COUNT
        and _int_value(runtime_environment, "world_size") == EXPECTED_DUAL_T4_COUNT
        and _int_value(runtime_environment, "nproc_per_node") == EXPECTED_DUAL_T4_COUNT
        and len(gpu_names) == EXPECTED_DUAL_T4_COUNT
        and all("T4" in name for name in gpu_names)
        and _rank_assignments_pass(rank_assignments)
        and child_launch_pass
    )
    global_projection = _global_projection_payload(settings=settings, rows=dual_rows)
    status = (
        PASS_STATUS
        if (
            runtime_pass
            and not missing_rows
            and not nonpassing_rows
            and not linked_failures
            and global_projection["status"] == PASS_STATUS
        )
        else SKIPPED_UNSUPPORTED
    )
    return cast(
        "JsonObject",
        {
            "status": status,
            "required_before_selected_runtime": True,
            "failure_policy": (
                "do_not_write_selected_runtime_if_missing_failed_or_skipped"
            ),
            "visible_device_count": _json_value(
                runtime_environment.get("visible_device_count"),
            ),
            "cuda_device_count": _json_value(
                runtime_environment.get("cuda_device_count"),
            ),
            "world_size": _json_value(runtime_environment.get("world_size")),
            "nproc_per_node": _json_value(runtime_environment.get("nproc_per_node")),
            "gpu_names": _json_value(runtime_environment.get("gpu_names")),
            "rank_assignments": _json_value(
                runtime_environment.get("rank_assignments"),
            ),
            "child_process_launch_command": _json_value(
                runtime_environment.get("child_process_launch_command"),
            ),
            "child_process_launch_status": PASS_STATUS
            if child_launch_pass
            else SKIPPED_UNSUPPORTED,
            "rank_assignment_status": PASS_STATUS
            if _rank_assignments_pass(rank_assignments)
            else SKIPPED_UNSUPPORTED,
            "required_dual_row_ids": required_row_ids,
            "emitted_dual_row_count": len(dual_rows),
            "missing_dual_row_ids": missing_rows,
            "nonpassing_dual_row_ids": nonpassing_rows,
            "linked_failure_reasons": linked_failures,
            "global_throughput_projection": global_projection,
        },
    )


def _linked_failures(  # noqa: PLR0913
    *,
    settings: _SelectionSettings,
    required_rows: Sequence[CsvRow],
    dataloader_rows: Sequence[CsvRow],
    numerical_rows: Sequence[CsvRow],
    corruption_rows: Sequence[CsvRow],
    gate_health_rows: Sequence[CsvRow],
    gate_health_summary: JsonObject,
) -> list[str]:
    failures: list[str] = []
    for runtime_row in required_rows:
        row_id = runtime_row["row_id"]
        if not _dataloader_pass_for_runtime_row(dataloader_rows, runtime_row):
            failures.append(f"dataloader_matrix:{row_id}")
        if not _numerical_pass_for_runtime_row(
            settings=settings,
            numerical_rows=numerical_rows,
            runtime_row=runtime_row,
        ):
            failures.append(f"numerical_checks:{row_id}")
        if not _corruption_pass_for_runtime_row(
            settings=settings,
            corruption_rows=corruption_rows,
            runtime_row=runtime_row,
        ):
            failures.append(f"corruption_checks:{row_id}")
        if not _gate_health_pass_for_runtime_row(
            gate_health_rows=gate_health_rows,
            gate_health_summary=gate_health_summary,
            runtime_row=runtime_row,
        ):
            failures.append(f"gate_health:{row_id}")
    if gate_health_summary.get("status") != PASS_STATUS:
        failures.append("gate_health_summary:not_pass")
    return failures


def _global_projection_payload(
    *,
    settings: _SelectionSettings,
    rows: Sequence[CsvRow],
) -> JsonObject:
    projections: list[JsonObject] = []
    for row in rows:
        if row["status"] != PASS_STATUS:
            continue
        global_batch_size = int(row["global_batch_size"])
        steady_ms = _float_or_none(row["steady_step_ms_p50"])
        if steady_ms is None or steady_ms <= 0.0:
            continue
        steps_per_epoch = training_steps_per_epoch(
            real_train_patch_count=settings.real_train_patch_count,
            global_batch_size=global_batch_size,
        )
        projections.append({
            "row_id": row["row_id"],
            "runtime_policy_id": _runtime_policy_id(row),
            "real_train_patch_count": settings.real_train_patch_count,
            "global_batch_size": global_batch_size,
            "drop_last": True,
            "steps_per_epoch": steps_per_epoch,
            "effective_samples_per_epoch": steps_per_epoch * global_batch_size,
            "steady_step_ms_p50": steady_ms,
            "estimated_epoch_minutes": steps_per_epoch * steady_ms / 60_000.0,
        })
    return cast(
        "JsonObject",
        {
            "status": PASS_STATUS if projections else SKIPPED_UNSUPPORTED,
            "projection_basis": "floor_steps_times_global_batch_drop_last_true",
            "rows": projections,
        },
    )


def _selected_runtime_write_decision(  # noqa: C901, PLR0912, PLR0913
    *,
    settings: _SelectionSettings,
    runtime_rows: Sequence[CsvRow],
    dataloader_rows: Sequence[CsvRow],
    numerical_rows: Sequence[CsvRow],
    corruption_rows: Sequence[CsvRow],
    gate_health_summary: JsonObject,
    gate_health_rows: Sequence[CsvRow],
    model_count_payload: JsonObject,
    dual_gate: JsonObject,
    amp_policy: JsonObject,
    v8_provenance: JsonObject,
    stain_corruptor_qa: JsonObject,
) -> JsonObject:
    blockers: list[str] = []
    if v8_provenance.get("status") != PASS_STATUS:
        blockers.append("v8_hash_provenance_not_pass")
    single_confirmation = _single_confirmation_payload(
        settings=settings,
        rows=runtime_rows,
    )
    if single_confirmation.get("status") != PASS_STATUS:
        blockers.append("single_visible_t4_eager_fp32_confirmation_not_pass")
    if model_count_payload.get("status") != PASS_STATUS:
        blockers.append("model_count_not_pass")
    if dual_gate.get("status") != PASS_STATUS:
        blockers.append("missing_real_dual_t4_train_step_timing")
    if amp_policy.get("status") != PASS_STATUS:
        blockers.append("amp_followup_policy_not_pass")
    if gate_health_summary.get("status") != PASS_STATUS:
        blockers.append("gate_health_summary_not_pass")
    if stain_corruptor_qa.get("status") != PASS_STATUS:
        blockers.append("stain_corruptor_qa_not_pass")
    missing_stain_qa_rows = _stain_qa_missing_runtime_rows(
        stain_corruptor_qa=stain_corruptor_qa,
        runtime_rows=runtime_rows,
    )
    if missing_stain_qa_rows:
        blockers.append("stain_corruptor_qa_candidate_scope_not_pass")
    candidates = _selection_candidate_rows(settings=settings, rows=runtime_rows)
    if settings.baseline_selected_runtime_path is not None and (
        baseline_blocker := _baseline_snapshot_blocker(settings, candidates)
    ):
        blockers.append(baseline_blocker)
    selected = _selected_row_or_none(settings=settings, rows=runtime_rows)
    linked_pass_row_failures: list[str] = []
    if selected is None:
        blockers.append("runtime_matrix_has_no_selectable_pass_row")
    elif not _row_id_present(runtime_rows, selected["row_id"]):
        blockers.append("selected_runtime_reuses_configured_baseline_no_replacement")
    else:
        linked_pass_row_failures = _linked_pass_row_failures(
            settings=settings,
            runtime_rows=(selected,),
            dataloader_rows=dataloader_rows,
            numerical_rows=numerical_rows,
            corruption_rows=corruption_rows,
            gate_health_rows=gate_health_rows,
            gate_health_summary=gate_health_summary,
        )
        if linked_pass_row_failures:
            blockers.append("runtime_pass_rows_linked_proof_not_pass")
        if not _dataloader_pass_for_runtime_row(dataloader_rows, selected):
            blockers.append("selected_row_dataloader_not_pass")
        if not _numerical_pass_for_runtime_row(
            settings=settings,
            numerical_rows=numerical_rows,
            runtime_row=selected,
        ):
            blockers.append("selected_row_numerical_not_pass")
        if not _corruption_pass_for_runtime_row(
            settings=settings,
            corruption_rows=corruption_rows,
            runtime_row=selected,
        ):
            blockers.append("selected_row_corruption_not_pass")
        if not _gate_health_pass_for_runtime_row(
            gate_health_rows=gate_health_rows,
            gate_health_summary=gate_health_summary,
            runtime_row=selected,
        ):
            blockers.append("selected_row_gate_health_not_pass")
    return cast(
        "JsonObject",
        {
            "allowed": not blockers,
            "blockers": blockers,
            "linked_pass_row_failures": linked_pass_row_failures,
            "stain_corruptor_qa_status": _json_value(stain_corruptor_qa.get("status")),
            "stain_corruptor_qa_missing_candidate_row_ids": missing_stain_qa_rows,
            "selected_row_id": "" if selected is None else selected["row_id"],
            "policy": (
                "write_selected_runtime_only_after_dual_t4_and_all_linked_proofs_pass"
            ),
        },
    )


def _stain_qa_missing_runtime_rows(
    *,
    stain_corruptor_qa: JsonObject,
    runtime_rows: Sequence[CsvRow],
) -> list[str]:
    candidate_ids = stain_corruptor_qa.get("candidate_row_ids")
    if not isinstance(candidate_ids, list) or not all(
        isinstance(item, str) for item in candidate_ids
    ):
        return [row["row_id"] for row in runtime_rows if row["status"] == PASS_STATUS]
    covered = set(cast("list[str]", candidate_ids))
    return [
        row["row_id"]
        for row in runtime_rows
        if row["status"] == PASS_STATUS and row["row_id"] not in covered
    ]


def _selected_row_or_none(
    *,
    settings: _SelectionSettings,
    rows: Sequence[CsvRow],
) -> CsvRow | None:
    candidates = _selection_candidate_rows(settings=settings, rows=rows)
    baseline = _baseline_row(settings=settings, candidates=candidates)
    if not candidates:
        return baseline
    ranked = sorted(
        candidates,
        key=lambda row: (
            -(_float_or_none(row["samples_sec"]) or 0.0),
            _float_or_none(row["steady_step_ms_p95"]) or math.inf,
            0 if row["accelerator_mode"] == SINGLE_VISIBLE_T4 else 1,
        ),
    )
    fastest = ranked[0]
    if baseline is None:
        if settings.baseline_selected_runtime_path is not None:
            return None
        return fastest
    fastest_samples = _float_or_none(fastest["samples_sec"]) or 0.0
    baseline_samples = _float_or_none(baseline["samples_sec"]) or 0.0
    if fastest["row_id"] == baseline["row_id"]:
        return baseline
    if fastest_samples >= baseline_samples * (
        1.0 + settings.minimum_material_speedup_fraction
    ):
        return fastest
    return baseline


def _selection_candidate_rows(
    *,
    settings: _SelectionSettings,
    rows: Sequence[CsvRow],
) -> list[CsvRow]:
    return [
        row
        for row in rows
        if _runtime_row_candidate_pass(row)
        and _selection_candidate_scope_matches(settings=settings, row=row)
    ]


def _selection_candidate_scope_matches(
    *,
    settings: _SelectionSettings,
    row: CsvRow,
) -> bool:
    if row["row_id"] == settings.baseline_selected_row_id:
        return True
    if not settings.efficiency_policies:
        return True
    batch_size = _optional_csv_int(row.get("per_device_batch_size", ""))
    return (
        row["accelerator_mode"] in settings.efficiency_accelerator_modes
        and batch_size in settings.efficiency_batch_sizes
        and row["corruption_strategy"] in settings.efficiency_corruption_strategies
        and any(
            not policy.diagnostic_only
            and policy.runtime_policy_id == _runtime_policy_id(row)
            and policy.precision_policy == row["precision_policy"]
            and policy.compile_scope == row["compile_scope"]
            for policy in settings.efficiency_policies
        )
    )


def _optional_bool_or_default(
    payload: JsonObject,
    key: str,
    *,
    default: bool,
) -> bool:
    value = payload.get(key, default)
    if type(value) is not bool:
        message = f"{key} must be a boolean"
        raise TypeError(message)
    return value


def _runtime_row_candidate_pass(row: CsvRow) -> bool:
    samples_sec = _float_or_none(row.get("samples_sec", ""))
    vram_headroom = _float_or_none(row.get("vram_headroom_fraction", ""))
    return (
        row["status"] == PASS_STATUS
        # A batch the VRAM feasibility screen marked infeasible (Spec 0011 S14c) is
        # never selectable, defensively even if a future path leaves it status=pass:
        # an oom row does not fit the GPU with the DDP-run margin.
        and row.get("oom", "false") != "true"
        and row["precision_policy"]
        in {
            AMP_OFF_FP32,
            AMP_CONSERVATIVE,
            AMP_SCALAR_GATE_RELAXED,
        }
        and (row["compile_scope"] == COMPILE_NONE or _compiled_row_stable(row))
        and samples_sec is not None
        and math.isfinite(samples_sec)
        and samples_sec > 0.0
        and (_optional_csv_int(row.get("amp_step_skipped_count", "")) or 0) == 0
        and vram_headroom is not None
        and math.isfinite(vram_headroom)
        and vram_headroom > 0.0
    )


def _baseline_row(
    *,
    settings: _SelectionSettings,
    candidates: Sequence[CsvRow],
) -> CsvRow | None:
    measured = [
        row for row in candidates if row["row_id"] == settings.baseline_selected_row_id
    ]
    if measured:
        return max(measured, key=lambda row: _float_or_none(row["samples_sec"]) or 0.0)
    return _baseline_snapshot_row(settings)


def _baseline_snapshot_blocker(
    settings: _SelectionSettings,
    candidates: Sequence[CsvRow],
) -> str:
    if settings.baseline_selected_runtime_path is None:
        return ""
    if _measured_baseline_rows(settings=settings, candidates=candidates):
        return ""
    if not settings.baseline_selected_runtime_path.exists():
        return "baseline_selected_runtime_not_available"
    if _baseline_snapshot_row(settings) is None:
        return "baseline_selected_runtime_identity_mismatch"
    return ""


def _measured_baseline_rows(
    *,
    settings: _SelectionSettings,
    candidates: Sequence[CsvRow],
) -> list[CsvRow]:
    return [
        row for row in candidates if row["row_id"] == settings.baseline_selected_row_id
    ]


def _baseline_selected_runtime_payload(
    settings: _SelectionSettings,
) -> JsonObject | None:
    if settings.baseline_selected_runtime_path is None:
        return None
    path = settings.baseline_selected_runtime_path
    if not path.exists():
        return None
    return _load_json(path)


def _baseline_snapshot_row(settings: _SelectionSettings) -> CsvRow | None:
    baseline_payload = _baseline_selected_runtime_payload(settings)
    if baseline_payload is None or baseline_payload.get("status") != PASS_STATUS:
        return None
    snapshot = baseline_payload.get("selected_row_snapshot")
    if not isinstance(snapshot, dict):
        return None
    row = cast("CsvRow", {key: str(value) for key, value in snapshot.items()})
    if (
        baseline_payload.get("selected_row_id") != settings.baseline_selected_row_id
        or baseline_payload.get("runtime_policy_id")
        != settings.baseline_runtime_policy_id
        or row.get("row_id") != settings.baseline_selected_row_id
        or _runtime_policy_id(row) != settings.baseline_runtime_policy_id
        or row.get("status") != PASS_STATUS
    ):
        return None
    return row


def _row_id_present(rows: Sequence[CsvRow], row_id: str) -> bool:
    return any(row["row_id"] == row_id for row in rows)


def _selected_row(settings: _SelectionSettings, rows: Sequence[CsvRow]) -> CsvRow:
    selected = _selected_row_or_none(settings=settings, rows=rows)
    if selected is None:
        message = "No selected runtime row is available"
        raise RuntimeError(message)
    return selected


def _linked_pass_row_failures(  # noqa: PLR0913
    *,
    settings: _SelectionSettings,
    runtime_rows: Sequence[CsvRow],
    dataloader_rows: Sequence[CsvRow],
    numerical_rows: Sequence[CsvRow],
    corruption_rows: Sequence[CsvRow],
    gate_health_rows: Sequence[CsvRow],
    gate_health_summary: JsonObject,
) -> list[str]:
    failures: list[str] = []
    for runtime_row in runtime_rows:
        if not _runtime_row_candidate_pass(runtime_row):
            continue
        row_id = runtime_row["row_id"]
        if not _dataloader_pass_for_runtime_row(dataloader_rows, runtime_row):
            failures.append(f"dataloader_matrix:{row_id}")
        if not _numerical_pass_for_runtime_row(
            settings=settings,
            numerical_rows=numerical_rows,
            runtime_row=runtime_row,
        ):
            failures.append(f"numerical_checks:{row_id}")
        if not _corruption_pass_for_runtime_row(
            settings=settings,
            corruption_rows=corruption_rows,
            runtime_row=runtime_row,
        ):
            failures.append(f"corruption_checks:{row_id}")
        if not _gate_health_pass_for_runtime_row(
            gate_health_rows=gate_health_rows,
            gate_health_summary=gate_health_summary,
            runtime_row=runtime_row,
        ):
            failures.append(f"gate_health:{row_id}")
    return failures


def _dataloader_pass_for_runtime_row(
    dataloader_rows: Sequence[CsvRow],
    runtime_row: CsvRow,
) -> bool:
    expected_ranks = {str(rank) for rank in range(int(runtime_row["world_size"]))}
    for split in REQUIRED_DATALOADER_SPLITS:
        observed_ranks = {
            row["rank"]
            for row in _matching_dataloader_rows(
                dataloader_rows,
                runtime_row,
                split=split,
            )
            if _dataloader_row_pass(row, runtime_row)
        }
        if not expected_ranks.issubset(observed_ranks):
            return False
    return True


def _matching_dataloader_rows(
    dataloader_rows: Sequence[CsvRow],
    runtime_row: CsvRow,
    *,
    split: str | None = None,
) -> list[CsvRow]:
    return [
        row
        for row in dataloader_rows
        if row.get("accelerator_mode") == runtime_row["accelerator_mode"]
        and row.get("machine_shape") == runtime_row["machine_shape"]
        and row.get("world_size") == runtime_row["world_size"]
        and row.get("batch_size") == runtime_row["per_device_batch_size"]
        and _runtime_policy_matches(row, runtime_row)
        and (split is None or row.get("split") == split)
    ]


def _selected_dataloader_payload(
    *,
    dataloader_rows: Sequence[CsvRow],
    selected_row: CsvRow,
) -> JsonObject:
    matches = sorted(
        _matching_dataloader_rows(dataloader_rows, selected_row, split="train"),
        key=lambda row: int(row["rank"] or "0"),
    )
    for row in matches:
        if not _dataloader_row_pass(row, selected_row):
            continue
        return {
            "num_workers": _int_from_csv(row["num_workers"]),
            "prefetch_factor": _optional_int_from_csv(row["prefetch_factor"]),
            "pin_memory": _bool_from_csv(row["pin_memory"]),
            "persistent_workers": _bool_from_csv(row["persistent_workers"]),
            "non_blocking_h2d": _bool_from_csv(row["non_blocking_h2d"]),
        }
    message = f"Missing passing dataloader row for {selected_row['row_id']}"
    raise RuntimeError(message)


def _dataloader_row_pass(row: CsvRow, runtime_row: CsvRow) -> bool:
    batches_measured = _optional_csv_int(row.get("batches_measured", ""))
    measured_steps = _optional_csv_int(runtime_row.get("measured_steps", ""))
    required_batches = measured_steps or len(REQUIRED_NUMERICAL_BATCH_INDICES)
    data_wait_fraction = _float_or_none(row.get("data_wait_fraction_p95", ""))
    loader_samples_sec = _float_or_none(row.get("loader_samples_sec", ""))
    trainer_samples_sec = _float_or_none(row.get("trainer_samples_sec", ""))
    rank_sample_count = _optional_csv_int(row.get("rank_sample_count", ""))
    return (
        row.get("status") == PASS_STATUS
        and row.get("benchmark_kind") == RUNTIME_SELECTION_KIND
        and row.get("benchmark_source") == RUNTIME_SELECTION_SOURCE
        and row.get("full_run_eligible") == "true"
        and row.get("accelerator_mode") == runtime_row["accelerator_mode"]
        and row.get("machine_shape") == runtime_row["machine_shape"]
        and row.get("world_size") == runtime_row["world_size"]
        and row.get("batch_size") == runtime_row["per_device_batch_size"]
        and _runtime_policy_matches(row, runtime_row)
        and row.get("split") in REQUIRED_DATALOADER_SPLITS
        and row.get("rank", "").isdigit()
        and batches_measured is not None
        and batches_measured >= required_batches
        and rank_sample_count is not None
        and rank_sample_count > 0
        and data_wait_fraction is not None
        and math.isfinite(data_wait_fraction)
        and data_wait_fraction <= MAX_DATA_WAIT_FRACTION
        and loader_samples_sec is not None
        and math.isfinite(loader_samples_sec)
        and trainer_samples_sec is not None
        and math.isfinite(trainer_samples_sec)
        and trainer_samples_sec > 0.0
        and loader_samples_sec
        >= MIN_LOADER_TRAINER_THROUGHPUT_RATIO * trainer_samples_sec
    )


def _numerical_pass_for_runtime_row(
    *,
    settings: _SelectionSettings,
    numerical_rows: Sequence[CsvRow],
    runtime_row: CsvRow,
) -> bool:
    passing_batch_indices = {
        row["batch_index"]
        for row in numerical_rows
        if _candidate_scope_matches(
            settings=settings,
            row=row,
            runtime_row=runtime_row,
        )
        and _numerical_values_pass(row)
    }
    return REQUIRED_NUMERICAL_BATCH_INDICES.issubset(passing_batch_indices)


def _corruption_pass_for_runtime_row(
    *,
    settings: _SelectionSettings,
    corruption_rows: Sequence[CsvRow],
    runtime_row: CsvRow,
) -> bool:
    passing_splits = {
        row["split"]
        for row in corruption_rows
        if _common_candidate_scope_pass(
            settings=settings,
            row=row,
            runtime_row=runtime_row,
        )
        and row.get("corruption_strategy") == runtime_row["corruption_strategy"]
        and row.get("world_size") == runtime_row["world_size"]
        and row.get("split") in REQUIRED_CORRUPTION_SPLITS
        and _nonempty_csv(row, "applied_mask_hash")
        and _nonempty_csv(row, "stain_param_hash")
        and _nonempty_csv(row, "noise_std_hash")
        and _nonempty_csv(row, "noise_field_hash")
        and (
            row.get("split") == "train"
            or row.get("clean_validation_rng_advanced") == "false"
        )
    }
    return REQUIRED_CORRUPTION_SPLITS.issubset(passing_splits)


def _candidate_scope_matches(
    *,
    settings: _SelectionSettings,
    row: CsvRow,
    runtime_row: CsvRow,
) -> bool:
    return (
        _common_candidate_scope_matches(
            settings=settings,
            row=row,
            runtime_row=runtime_row,
            require_status=False,
        )
        and row.get("status") in {PASS_STATUS, FAIL_STATUS}
        and row.get("precision_policy") == runtime_row["precision_policy"]
        and row.get("torch_compile_enabled") == runtime_row["torch_compile_enabled"]
        and row.get("compile_scope") == runtime_row["compile_scope"]
        and row.get("corruption_strategy") == runtime_row["corruption_strategy"]
    )


def _nonempty_csv(row: CsvRow, key: str) -> bool:
    return bool(row.get(key, ""))


def _common_candidate_scope_pass(
    *,
    settings: _SelectionSettings,
    row: CsvRow,
    runtime_row: CsvRow,
) -> bool:
    return _common_candidate_scope_matches(
        settings=settings,
        row=row,
        runtime_row=runtime_row,
        require_status=True,
    )


def _common_candidate_scope_matches(
    *,
    settings: _SelectionSettings,
    row: CsvRow,
    runtime_row: CsvRow,
    require_status: bool,
) -> bool:
    return (
        (not require_status or row.get("status") == PASS_STATUS)
        and row.get("benchmark_kind") == RUNTIME_SELECTION_KIND
        and row.get("benchmark_source") == RUNTIME_SELECTION_SOURCE
        and row.get("full_run_eligible") == "true"
        and row.get("candidate_row_id") == runtime_row["row_id"]
        and row.get("reference_row_id")
        == _reference_row_id(settings=settings, runtime_row=runtime_row)
        and row.get("accelerator_mode") == runtime_row["accelerator_mode"]
        and row.get("machine_shape") == runtime_row["machine_shape"]
        and _runtime_policy_matches(row, runtime_row)
    )


def _numerical_values_pass(row: CsvRow) -> bool:
    delta_fields = [key for key in row if key.endswith("_delta")]
    return (
        _numerical_status_policy_pass(row)
        and all(_csv_float_is_finite(row[field]) for field in delta_fields)
        and _relaxed_numerical_deltas_pass(row)
        and _optional_csv_int(row.get("nonfinite_count", "")) == 0
        and row.get("amp_step_skipped") == "false"
        and row.get("gate_health_status") == PASS_STATUS
    )


def _numerical_status_policy_pass(row: CsvRow) -> bool:
    failure_kind = row.get("failure_kind", "")
    if row.get("status") == PASS_STATUS:
        return not failure_kind
    return (
        row.get("status") == FAIL_STATUS
        and failure_kind == RELAXED_NUMERICAL_DELTA_FAILURE_KIND
    )


def _relaxed_numerical_deltas_pass(row: CsvRow) -> bool:
    """Accept small drift while still blocking clearly invalid metrics.

    Returns:
        True when all relaxed numerical deltas remain within policy bounds.

    """
    return (
        _bounded_delta(row, "total_loss_abs_delta", MAX_RELAXED_LOSS_ABS_DELTA)
        and _bounded_delta(row, "total_loss_rel_delta", MAX_RELAXED_LOSS_REL_DELTA)
        and _bounded_delta(row, "recon_loss_abs_delta", MAX_RELAXED_LOSS_ABS_DELTA)
        and _bounded_delta(row, "recon_loss_rel_delta", MAX_RELAXED_LOSS_REL_DELTA)
        and _bounded_delta(row, "l1_loss_abs_delta", MAX_RELAXED_LOSS_ABS_DELTA)
        and _bounded_delta(row, "l1_loss_rel_delta", MAX_RELAXED_LOSS_REL_DELTA)
        and _bounded_delta(row, "ssim_loss_abs_delta", MAX_RELAXED_LOSS_ABS_DELTA)
        and _bounded_delta(row, "ssim_loss_rel_delta", MAX_RELAXED_LOSS_REL_DELTA)
        and _bounded_delta(row, "kl_loss_abs_delta", MAX_RELAXED_LOSS_ABS_DELTA)
        and _bounded_delta(row, "kl_loss_rel_delta", MAX_RELAXED_LOSS_REL_DELTA)
        and _bounded_delta(row, "grad_norm_rel_delta", MAX_RELAXED_GRAD_REL_DELTA)
        and _bounded_delta(
            row,
            "param_update_norm_rel_delta",
            MAX_RELAXED_PARAM_UPDATE_REL_DELTA,
        )
        and _bounded_delta(row, "x_hat_min_abs_delta", MAX_RELAXED_STATE_ABS_DELTA)
        and _bounded_delta(row, "x_hat_max_abs_delta", MAX_RELAXED_STATE_ABS_DELTA)
        and _bounded_delta(row, "mu_mean_abs_delta", MAX_RELAXED_STATE_ABS_DELTA)
        and _bounded_delta(row, "mu_std_abs_delta", MAX_RELAXED_STATE_ABS_DELTA)
        and _bounded_delta(row, "logvar_mean_abs_delta", MAX_RELAXED_STATE_ABS_DELTA)
        and _bounded_delta(row, "logvar_std_abs_delta", MAX_RELAXED_STATE_ABS_DELTA)
        and _bounded_delta(row, "logvar_clamp_count_delta", 0.0)
    )


def _bounded_delta(row: CsvRow, key: str, max_abs: float) -> bool:
    value = _float_or_none(row.get(key, ""))
    return value is not None and math.isfinite(value) and abs(value) <= max_abs


def _gate_health_pass_for_runtime_row(
    *,
    gate_health_rows: Sequence[CsvRow],
    gate_health_summary: JsonObject,
    runtime_row: CsvRow,
) -> bool:
    candidate_ids = _summary_candidate_row_ids(gate_health_summary)
    matching_rows = [
        row
        for row in gate_health_rows
        if _gate_health_row_matches_runtime(row, runtime_row)
    ]
    if runtime_row["precision_policy"] == AMP_SCALAR_GATE_RELAXED:
        return (
            gate_health_summary.get("status") == PASS_STATUS
            and runtime_row["row_id"] in candidate_ids
            and bool(matching_rows)
            and all(_gate_health_row_pass(row, runtime_row) for row in matching_rows)
        )
    return (
        gate_health_summary.get("status") == PASS_STATUS
        and runtime_row["row_id"] in candidate_ids
        and any(_gate_health_row_pass(row, runtime_row) for row in matching_rows)
    )


def _summary_candidate_row_ids(gate_health_summary: JsonObject) -> set[str]:
    for key in ("candidate_row_ids", "runtime_row_ids", "pass_row_ids"):
        value = gate_health_summary.get(key)
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return set(cast("list[str]", value))
    return set()


def _gate_health_row_pass(row: CsvRow, runtime_row: CsvRow) -> bool:
    return (
        row.get("gate_health_status") == PASS_STATUS
        and _gate_health_row_matches_runtime(row, runtime_row)
        and _scalar_gate_precision_proof_pass(row, runtime_row)
    )


def _gate_health_row_matches_runtime(row: CsvRow, runtime_row: CsvRow) -> bool:
    return (
        row.get("benchmark_kind") == RUNTIME_SELECTION_KIND
        and row.get("benchmark_source") == RUNTIME_SELECTION_SOURCE
        and row.get("full_run_eligible") == "true"
        and row.get("accelerator_mode") == runtime_row["accelerator_mode"]
        and row.get("machine_shape") == runtime_row["machine_shape"]
        and row.get("candidate_row_id") == runtime_row["row_id"]
        and _runtime_policy_matches(row, runtime_row)
        and _nonempty_csv(row, "row_id")
    )


def _scalar_gate_precision_proof_pass(row: CsvRow, runtime_row: CsvRow) -> bool:
    if runtime_row["precision_policy"] != AMP_SCALAR_GATE_RELAXED:
        return True
    input_dtype = row.get("input_dtype", "")
    gate_math_dtype = row.get("gate_math_dtype", "")
    gate_tensor_dtype = row.get("gate_tensor_dtype", "")
    output_dtype = row.get("output_dtype", "")
    requested_autocast_dtype = runtime_row.get("autocast_dtype", "") or row.get(
        "requested_autocast_dtype",
        "",
    )
    return (
        row.get("precision_proof_status") == PASS_STATUS
        and row.get("gate_force_fp32") == "false"
        and requested_autocast_dtype in {"float16", "bfloat16"}
        and input_dtype == requested_autocast_dtype
        and gate_math_dtype == requested_autocast_dtype
        and gate_tensor_dtype == requested_autocast_dtype
        and output_dtype == requested_autocast_dtype
        and gate_math_dtype != "float32"
    )


def _runtime_proof_payload(  # noqa: PLR0913
    *,
    settings: _SelectionSettings,
    runtime_rows: Sequence[CsvRow],
    v8_provenance: JsonObject,
    runtime_environment: JsonObject,
    dual_gate: JsonObject,
    amp_policy: JsonObject,
    decision: JsonObject,
    model_count_payload: JsonObject,
    stain_corruptor_qa: JsonObject,
) -> JsonObject:
    return {
        "schema_version": RUNTIME_SELECTION_SCHEMA_VERSION,
        "status": PASS_STATUS if decision["allowed"] else FAIL_STATUS,
        "benchmark_kind": RUNTIME_SELECTION_KIND,
        "benchmark_source": RUNTIME_SELECTION_SOURCE,
        "full_run_eligible": bool(decision["allowed"]),
        "run_name": settings.run_name,
        "machine_shape": EXPECTED_MACHINE_SHAPE,
        "accelerator_modes_checked": [SINGLE_VISIBLE_T4, DUAL_T4_DDP],
        "v8_provenance": v8_provenance,
        "single_visible_t4_confirmation": _single_confirmation_payload(
            settings=settings,
            rows=runtime_rows,
        ),
        "amp_followup_policy": amp_policy,
        "efficiency_followup": _efficiency_followup_payload(
            settings=settings,
            runtime_rows=runtime_rows,
            selected_row_id=_json_value(decision.get("selected_row_id")),
        ),
        "dual_t4_train_step_gate": dual_gate,
        "runtime_environment": runtime_environment,
        "model_count_status": _json_value(model_count_payload.get("status")),
        "stain_corruptor_qa_status": _json_value(stain_corruptor_qa.get("status")),
        "compiled_rows_policy": (
            "selectable_only_with_stable_selected_runtime_compile_settle_proof"
        ),
        "compiled_pass_rows_rewritten_ineligible": [
            row["row_id"]
            for row in runtime_rows
            if row["compile_scope"] != COMPILE_NONE
            and row["failure_kind"].startswith("compiled_rows_diagnostic_only")
        ],
        "selection_ready": bool(decision["allowed"]),
        "selected_runtime_written": bool(decision["allowed"]),
        "selected_runtime_write_decision": decision,
    }


def _efficiency_followup_payload(
    *,
    settings: _SelectionSettings,
    runtime_rows: Sequence[CsvRow],
    selected_row_id: JsonValue,
) -> JsonObject:
    candidates = _selection_candidate_rows(settings=settings, rows=runtime_rows)
    baseline = _baseline_row(settings=settings, candidates=candidates)
    selected = next(
        (row for row in candidates if row["row_id"] == selected_row_id),
        None,
    )
    baseline_samples = (
        None if baseline is None else _float_or_none(baseline.get("samples_sec", ""))
    )
    selected_samples = (
        None if selected is None else _float_or_none(selected.get("samples_sec", ""))
    )
    material_speedup = (
        False
        if baseline_samples is None
        or selected_samples is None
        or baseline_samples <= 0.0
        else selected_samples
        >= baseline_samples * (1.0 + settings.minimum_material_speedup_fraction)
    )
    return {
        "status": PASS_STATUS if selected is not None else SKIPPED_UNSUPPORTED,
        "baseline_selected_runtime": ""
        if settings.baseline_selected_runtime_path is None
        else str(settings.baseline_selected_runtime_path),
        "baseline_row_id": settings.baseline_selected_row_id,
        "baseline_runtime_policy_id": settings.baseline_runtime_policy_id,
        "minimum_material_speedup_fraction": (
            settings.minimum_material_speedup_fraction
        ),
        "proof_reference_per_device_batch_size": (
            settings.efficiency_proof_reference_batch_size
        ),
        "baseline_samples_sec": baseline_samples,
        "selected_samples_sec": selected_samples,
        "selected_row_id": selected_row_id,
        "selected_runtime_policy_id": ""
        if selected is None
        else _runtime_policy_id(selected),
        "material_speedup_over_baseline": material_speedup,
        "candidate_row_count": len(candidates),
        "ignored_candidate_row_count": sum(
            1
            for row in runtime_rows
            if _runtime_row_candidate_pass(row)
            and not _selection_candidate_scope_matches(settings=settings, row=row)
        ),
        "catastrophic_blockers": [
            row["row_id"]
            for row in runtime_rows
            if row["status"] == PASS_STATUS and not _runtime_row_candidate_pass(row)
        ],
        "baseline_available": baseline is not None,
        "selection_policy": (
            "prefer_fastest_material_speedup_over_configured_baseline_else_keep_"
            "baseline"
        ),
    }


def _single_confirmation_payload(
    *,
    settings: _SelectionSettings,
    rows: Sequence[CsvRow],
) -> JsonObject:
    target_rows = [
        row
        for row in rows
        if row["accelerator_mode"] == SINGLE_VISIBLE_T4
        and row["precision_policy"] == AMP_OFF_FP32
        and row["compile_scope"] == COMPILE_NONE
    ]
    pass_rows = [row["row_id"] for row in target_rows if row["status"] == PASS_STATUS]
    required_primary_row_ids = [
        _row_id(
            accelerator_mode=SINGLE_VISIBLE_T4,
            batch_size=batch_size,
            precision_policy=AMP_OFF_FP32,
            compile_scope=COMPILE_NONE,
            corruption_strategy=corruption_strategy,
        )
        for batch_size in settings.fp32_batch_sizes
        for corruption_strategy in settings.corruption_strategies
    ]
    fallback_row_ids = [
        _row_id(
            accelerator_mode=SINGLE_VISIBLE_T4,
            batch_size=batch_size,
            precision_policy=AMP_OFF_FP32,
            compile_scope=COMPILE_NONE,
            corruption_strategy=corruption_strategy,
        )
        for batch_size in settings.fallback_batch_sizes
        for corruption_strategy in settings.corruption_strategies
    ]
    missing_primary = [
        row_id for row_id in required_primary_row_ids if row_id not in pass_rows
    ]
    missing_fallback = [
        row_id for row_id in fallback_row_ids if row_id not in pass_rows
    ]
    return cast(
        "JsonObject",
        {
            "status": PASS_STATUS
            if not missing_primary and not missing_fallback
            else SKIPPED_UNSUPPORTED,
            "required_primary_batch_sizes": list(settings.fp32_batch_sizes),
            "fallback_batch_sizes": list(settings.fallback_batch_sizes),
            "required_primary_row_ids": required_primary_row_ids,
            "fallback_row_ids": fallback_row_ids,
            "missing_primary_row_ids": missing_primary,
            "missing_fallback_row_ids": missing_fallback,
            "pass_row_ids": pass_rows,
            "row_count": len(target_rows),
        },
    )


def _selected_runtime_payload(
    *,
    settings: _SelectionSettings,
    selected_row: CsvRow,
    dataloader_rows: Sequence[CsvRow],
    artifact_hashes: JsonObject,
) -> JsonObject:
    global_batch_size = int(selected_row["global_batch_size"])
    steady_ms = _float_or_none(selected_row["steady_step_ms_p50"]) or 0.0
    steps_per_epoch = training_steps_per_epoch(
        real_train_patch_count=settings.real_train_patch_count,
        global_batch_size=global_batch_size,
    )
    return {
        "status": PASS_STATUS,
        "benchmark_kind": RUNTIME_SELECTION_KIND,
        "benchmark_source": RUNTIME_SELECTION_SOURCE,
        "full_run_eligible": True,
        "full_training_launch_ready": False,
        "launch_blockers": [
            "missing_selected_runtime_debug_proof",
            "missing_checkpoint_resume_proof",
            "missing_tiny_overfit_proof",
        ],
        "selected_row_id": selected_row["row_id"],
        "runtime_policy_id": _runtime_policy_id(selected_row),
        "accelerator_mode": selected_row["accelerator_mode"],
        "machine_shape": selected_row["machine_shape"],
        "world_size": int(selected_row["world_size"]),
        "nproc_per_node": int(selected_row["nproc_per_node"]),
        "gpu_names": _json_array_from_csv(selected_row["gpu_names"]),
        "per_device_batch_size": int(selected_row["per_device_batch_size"]),
        "global_batch_size": global_batch_size,
        "gradient_accumulation_steps": int(selected_row["gradient_accumulation_steps"]),
        "optimizer_updates_per_epoch": steps_per_epoch,
        "lr_warmup_steps": 0,
        "beta_warmup_steps": 0,
        "mixed_precision": {
            "enabled": selected_row["precision_policy"] != AMP_OFF_FP32,
            "policy": selected_row["precision_policy"],
            "autocast_dtype": selected_row.get("autocast_dtype", ""),
            "fp32_loss": _bool_from_csv(selected_row.get("fp32_loss", "true")),
            "grad_scaler_enabled": _bool_from_csv(
                selected_row.get("grad_scaler_enabled", "false"),
            ),
        },
        "torch_compile": {
            "enabled": selected_row["compile_scope"] != COMPILE_NONE,
            "backend": "inductor"
            if selected_row["compile_scope"] != COMPILE_NONE
            else "eager",
            "scope": selected_row["compile_scope"],
            "dynamic": _bool_from_csv(selected_row.get("compile_dynamic", "false")),
            # Spec 0011 S11 dynamo/inductor recipe knobs (home = torch_compile);
            # eager defaults when the measured row has no such column yet (S13).
            "optimize_ddp": selected_row.get("optimize_ddp", ""),
            "compiled_autograd": _bool_from_csv(
                selected_row.get("compiled_autograd", "false"),
            ),
            "reorder_compute_comm_overlap": _bool_from_csv(
                selected_row.get("reorder_compute_comm_overlap", "false"),
            ),
        },
        "runtime_policy": {
            "memory_format": selected_row.get("memory_format", "contiguous"),
            "cudnn_benchmark": _bool_from_csv(
                selected_row.get("cudnn_benchmark", "false"),
            ),
            "cudnn_deterministic": _bool_from_csv(
                selected_row.get("cudnn_deterministic", "false"),
            ),
            "deterministic_algorithms": _bool_from_csv(
                selected_row.get("deterministic_algorithms", "false"),
            ),
            "tf32_enabled": _bool_from_csv(selected_row.get("tf32_enabled", "false")),
            "matmul_precision": selected_row.get("matmul_precision", "highest"),
            "ddp_static_graph": _bool_from_csv(
                selected_row.get("ddp_static_graph", "false"),
            ),
            "ddp_gradient_as_bucket_view": _bool_from_csv(
                selected_row.get("ddp_gradient_as_bucket_view", "false"),
            ),
            # Spec 0011 S11 DDP/optimizer recipe knobs (home = runtime_policy,
            # beside the existing ddp_* fields); eager defaults reproduce the v5
            # DDP wrap (broadcast_buffers/find_unused/bucket_cap are DDP defaults,
            # fused off) until the search measures them (S13/S14).
            "ddp_broadcast_buffers": _bool_from_csv(
                selected_row.get("ddp_broadcast_buffers", "true"),
            ),
            "ddp_find_unused_parameters": _bool_from_csv(
                selected_row.get("ddp_find_unused_parameters", "false"),
            ),
            "ddp_bucket_cap_mb": _optional_int_from_csv(
                selected_row.get("ddp_bucket_cap_mb", ""),
            ),
            "fused_optimizer": _bool_from_csv(
                selected_row.get("fused_optimizer", "false"),
            ),
            "optimizer_implementation": selected_row.get(
                "optimizer_implementation",
                "adamw_default",
            ),
            "zero_grad_set_to_none": _bool_from_csv(
                selected_row.get("zero_grad_set_to_none", "true"),
            ),
            "gradient_clip_foreach": _bool_from_csv(
                selected_row.get("gradient_clip_foreach", "true"),
            ),
            "gradient_clip_foreach_applied": True,
        },
        "relaxed_determinism": {
            "accepted": True,
            "policy": (
                "performance_first_accept_small_numerical_drift_block_catastrophic"
            ),
            "bitwise_determinism_required": False,
        },
        "corruption": {"strategy": selected_row["corruption_strategy"]},
        "dataloader": _selected_dataloader_payload(
            dataloader_rows=dataloader_rows,
            selected_row=selected_row,
        ),
        "throughput": {
            "samples_sec": _float_or_none(selected_row["samples_sec"]) or 0.0,
            "steady_step_ms_p50": steady_ms,
            "compile_startup_sec": _float_or_none(selected_row["compile_startup_sec"])
            or 0.0,
            "estimated_10_epoch_wall_time_sec": steps_per_epoch
            * steady_ms
            / 1000.0
            * 10,
        },
        "safety": {
            "numerical_check_status": PASS_STATUS,
            "corruption_check_status": PASS_STATUS,
            "gate_health_status": PASS_STATUS,
            "dataloader_status": PASS_STATUS,
            "amp_step_skipped_count": int(selected_row["amp_step_skipped_count"]),
        },
        "artifacts": artifact_hashes,
        "selected_row_snapshot": {
            **dict(selected_row),
            "compile_settle_protocol_sha256": _hash_text(
                _compile_settle_protocol_id(selected_row),
            ),
            "post_settle_graph_break_count": int(selected_row["graph_break_count"]),
            "post_settle_recompile_count": int(selected_row["recompile_count"]),
        },
        "resolved_full_run_config_path": (
            "configs/spec0001/non_eq_vae_baseline.resolved.json"
        ),
        "resolved_full_run_config_sha256": settings.effective_config_hash,
    }


def _compile_settle_protocol_id(selected_row: CsvRow) -> str:
    """Return the settle-proof identity actually applied to the selected row.

    Returns:
        Stable protocol identifier whose hash is stored in the selected snapshot.

    """
    if selected_row["compile_scope"] == COMPILE_NONE:
        return "runtime_selection_compile_none_eager_no_settle_v1"
    return (
        f"runtime_selection_{selected_row['compile_scope']}_"
        f"{selected_row.get('optimize_ddp', '')}_settle_"
        f"{selected_row['compile_settle_steps']}_reported_graph_breaks_"
        "zero_recompiles_v1"
    )


def _artifact_hashes(output_dir: Path) -> JsonObject:
    artifacts = {
        "runtime_matrix": "benchmark/runtime_matrix.csv",
        "model_count": "benchmark/model_count.json",
        "runtime_proof": "benchmark/runtime_proof.json",
        "dataloader_matrix": "benchmark/dataloader_matrix.csv",
        "numerical_checks": "benchmark/numerical_checks.csv",
        "corruption_checks": "benchmark/corruption_checks.csv",
        "stain_corruptor_qa": f"benchmark/{STAIN_CORRUPTOR_QA_FILENAME}",
        "gate_health_summary": "benchmark/gate_health_summary.json",
    }
    payload: JsonObject = {}
    for name, relative in artifacts.items():
        payload[name] = relative
        payload[f"{name}_sha256"] = _sha256_file(output_dir / relative)
    return payload


def _reject_stale_selected_runtime(benchmark_dir: Path) -> None:
    selected_runtime = benchmark_dir / SELECTED_RUNTIME_FILENAME
    if selected_runtime.exists():
        message = (
            "runtime-selection benchmark refuses to leave "
            "benchmark/selected_runtime.json while dual timing or linked proof "
            "is missing, failed, or skipped"
        )
        raise RuntimeError(message)


def _row_id(
    *,
    accelerator_mode: str,
    batch_size: int,
    precision_policy: str,
    compile_scope: str,
    corruption_strategy: str,
) -> str:
    return compose_row_id_base(
        accelerator_mode=accelerator_mode,
        batch_size=batch_size,
        precision_policy=precision_policy,
        compile_scope=compile_scope,
        corruption_strategy=corruption_strategy,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> JsonObject:
    payload = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(payload, dict):
        message = f"Expected JSON object at {path}"
        raise TypeError(message)
    return cast("JsonObject", payload)


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        return [{key: value or "" for key, value in row.items()} for row in reader]


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
    message = f"Expected object list field: {key}"
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


def _required_int(payload: Mapping[str, JsonValue], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    message = f"Expected integer field: {key}"
    raise TypeError(message)


def _int_value(payload: Mapping[str, JsonValue], key: str) -> int | None:
    value = payload.get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _optional_int(payload: Mapping[str, JsonValue], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    message = f"Expected optional integer field: {key}"
    raise TypeError(message)


def _optional_float(payload: Mapping[str, JsonValue], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    message = f"Expected optional numeric field: {key}"
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


def _str_list_value(payload: Mapping[str, JsonValue], key: str) -> list[str]:
    value = payload.get(key)
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(cast("list[str]", value))
    return []


def _object_list_value(payload: Mapping[str, JsonValue], key: str) -> list[JsonObject]:
    value = payload.get(key)
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        return [cast("JsonObject", item) for item in value]
    return []


def _rank_assignments_pass(assignments: Sequence[Mapping[str, JsonValue]]) -> bool:
    if len(assignments) != EXPECTED_DUAL_T4_COUNT:
        return False
    ranks: set[int] = set()
    local_ranks: set[int] = set()
    devices: set[int] = set()
    for assignment in assignments:
        rank = _int_value(assignment, "rank")
        local_rank = _int_value(assignment, "local_rank")
        device = _assignment_device(assignment)
        if rank is None or device is None:
            return False
        ranks.add(rank)
        devices.add(device)
        if local_rank is not None:
            local_ranks.add(local_rank)
    expected = set(range(EXPECTED_DUAL_T4_COUNT))
    return (
        ranks == expected
        and devices == expected
        and (not local_ranks or local_ranks == expected)
    )


def _child_launch_pass(runtime_environment: Mapping[str, JsonValue]) -> bool:
    command = runtime_environment.get("child_process_launch_command")
    if not isinstance(command, str):
        return False
    has_launcher = "torchrun" in command or "torch.distributed.run" in command
    has_nproc = (
        "--nproc_per_node=2" in command
        or "--nproc-per-node=2" in command
        or "--nproc_per_node 2" in command
        or "--nproc-per-node 2" in command
    )
    return has_launcher and has_nproc


def _assignment_device(assignment: Mapping[str, JsonValue]) -> int | None:
    for key in ("device_index", "cuda_device_index", "cuda_device", "device"):
        value = assignment.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.removeprefix("cuda:")
            if normalized.isdecimal():
                return int(normalized)
    return _int_value(assignment, "local_rank")


def _int_from_csv(value: str) -> int:
    return int(value)


def _optional_int_from_csv(value: str) -> int | None:
    return int(value) if value else None


def _bool_from_csv(value: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    message = f"Expected CSV boolean true|false, got {value!r}"
    raise ValueError(message)


def _csv_float_is_finite(value: str) -> bool:
    if not value:
        return False
    return math.isfinite(float(value))


def _optional_csv_int(value: str) -> int | None:
    if not value:
        return None
    return int(value)


def _stain_corruptor_qa_payload(path: Path) -> JsonObject:
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    payload = _load_json(path)
    status = payload.get("status")
    candidate_row_ids = payload.get("candidate_row_ids")
    missing_candidate_row_ids = payload.get("missing_candidate_row_ids")
    strict_pass = (
        status == PASS_STATUS
        and payload.get("benchmark_kind") == RUNTIME_SELECTION_KIND
        and payload.get("benchmark_source") == RUNTIME_SELECTION_SOURCE
        and payload.get("proof_scope") == STAIN_QA_PROOF_SCOPE
        and payload.get("full_run_eligible") is True
        and isinstance(candidate_row_ids, list)
        and all(isinstance(item, str) for item in candidate_row_ids)
        and bool(candidate_row_ids)
        and missing_candidate_row_ids == []
    )
    return {
        "status": PASS_STATUS if strict_pass else FAIL_STATUS,
        "path": str(path),
        "source_status": _json_value(status),
        "benchmark_kind": _json_value(payload.get("benchmark_kind")),
        "benchmark_source": _json_value(payload.get("benchmark_source")),
        "proof_scope": _json_value(payload.get("proof_scope")),
        "candidate_row_ids": _json_value(candidate_row_ids),
        "missing_candidate_row_ids": _json_value(missing_candidate_row_ids),
    }


def _json_value(value: object) -> JsonValue:
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, list):
        items = cast("list[object]", value)
        return [_json_value(item) for item in items]
    if isinstance(value, dict):
        mapping = cast("Mapping[object, object]", value)
        return {str(key): _json_value(item) for key, item in mapping.items()}
    return str(value)


def _json_array_from_csv(value: str) -> list[JsonValue]:
    parsed = cast("object", json.loads(value or "[]"))
    if isinstance(parsed, list):
        items = cast("list[object]", parsed)
        return [_json_value(item) for item in items]
    return []


def _bool_text(value: bool) -> str:  # noqa: FBT001
    return "true" if value else "false"


def _float_text(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def _float_or_none(value: str) -> float | None:
    if not value:
        return None
    return float(value)


__all__ = [
    "RuntimeSelectionArtifactPaths",
    "RuntimeSelectionBenchmarkRequest",
    "RuntimeSelectionEvidence",
    "load_runtime_selection_evidence",
    "write_runtime_selection_benchmark",
]
