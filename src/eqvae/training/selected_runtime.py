# Copyright 2026 HiperMaximus
"""Selected-runtime v5 plan parsing and local application proof helpers."""

from __future__ import annotations

import hashlib
import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from eqvae.benchmarking.row_id import compose_selected_row_id
from eqvae.benchmarking.schedule import training_steps_per_epoch
from eqvae.data.roots import REAL_TRAIN_PATCH_COUNT
from eqvae.training.fastpath_precision import (
    EXPECTED_RUNNER_AMP_GRAD_SCALER_INIT_SCALE,
)

if TYPE_CHECKING:
    from eqvae.config import JsonObject, JsonValue

EXPECTED_DATASET_SLUG = "maximusshtefan/patches-pre-shuffled-ubc-ocean"
EXPECTED_MACHINE_SHAPE = "NvidiaTeslaT4"
EXPECTED_SELECTED_ROW_ID = (
    "dual_t4_ddp__bs12__amp_conservative__compile_none__indexed_masked__"
    "policy_amp_fp16_conservative"
)
EXPECTED_RUNTIME_POLICY_ID = "amp_fp16_conservative"
EXPECTED_SELECTED_TORCH_VERSION = "2.13.0+cu130"
EXPECTED_SELECTED_CUDA_VERSION = "13.0"
EXPECTED_DUAL_T4_RANK_COUNT = 2
EXPECTED_RUNTIME_PROOF_WRITE_POLICY = (
    "write_selected_runtime_only_after_dual_t4_and_all_linked_proofs_pass"
)
EXPECTED_DDP_APPLICATION_STATUS = "executed_dual_t4_ddp"
EXPECTED_AMP_APPLICATION_STATUS = "executed_amp_fp16_conservative"
# Spec 0011 S17c -- the compiled bigger-batch winner runs the ``amp_off_fp32`` profile
# (autocast + grad scaler off), so its local-AMP application status is a distinct,
# plan-derived value rather than the eager fallback's fp16-conservative constant.
EXPECTED_AMP_OFF_APPLICATION_STATUS = "executed_amp_off_fp32"
SPEC0011_SELECTED_RUNTIME_SCHEMA = "spec0011.selected_runtime_plan.v1"
_SHA256_HEX_LENGTH = 64
# Spec 0011 S17f -- corruption is a FIXED speed-first property (the vectorized native
# ``InlineStainCorruptor``), no longer a plan-selected axis: the blake2b per-sample
# seeding is retired from every runtime path. The observed corruption label records
# honestly which inline variant ran -- fused seedless in the graph on the compiled fast
# path, through a dedicated checkpoint-continued ``torch.Generator`` on the eager
# training path, or through a per-boundary re-seeded generator on the validation path.
# The plan's declared ``corruption.strategy`` is now informational only (the runtime
# ignores it and always applies inline stain).
COMPILED_FASTPATH_CORRUPTION_STRATEGY = "compiled_fastpath_inline_stain"
EAGER_INLINE_STAIN_CORRUPTION_STRATEGY = "eager_inline_stain"
VALIDATION_INLINE_STAIN_CORRUPTION_STRATEGY = "validation_reseeded_inline_stain"


@dataclass(frozen=True)
class SelectedRuntimePlan:
    """Validated selected-runtime plan for the v5 fallback row."""

    path: Path
    artifact_sha256: str
    selected_row_id: str
    runtime_policy_id: str
    accelerator_mode: str
    machine_shape: str
    world_size: int
    nproc_per_node: int
    torchrun_standalone: bool
    per_device_batch_size: int
    global_batch_size: int
    gradient_accumulation_steps: int
    optimizer_updates_per_epoch: int
    precision_policy: str
    amp_enabled: bool
    autocast_dtype: str
    fp32_loss: bool
    grad_scaler_enabled: bool
    torch_compile_enabled: bool
    compile_scope: str
    dataloader_num_workers: int
    dataloader_prefetch_factor: int | None
    dataloader_pin_memory: bool
    dataloader_persistent_workers: bool
    dataloader_non_blocking_h2d: bool
    corruption_strategy: str
    memory_format: str
    ddp_static_graph: bool
    ddp_gradient_as_bucket_view: bool
    zero_grad_set_to_none: bool
    gradient_clip_foreach: bool = True
    # Spec 0011 S11 -- compiled fast-path recipe knobs. Optional with eager-v5
    # defaults so pre-S11 plans (the committed v5 fallback omits every knob below)
    # parse byte-identically to the eager recipe. Frozen carrier homes mirror
    # `_plan_from_payload`: the dynamo/inductor knobs live in the `torch_compile`
    # block, the DDP/optimizer knobs beside the existing `ddp_*` fields in
    # `runtime_policy`. Spec 0011 S15 wires the runner to *apply* the DDP-wrap and
    # fused-optimizer knobs via `training.fastpath_recipe`; the plan-applied
    # observation mirror for them is a scoped follow-up (the structural
    # `broadcast_buffers` override may legitimately diverge the effective value from
    # the plan, so a naive `observed == plan` mirror would false-flag it). The literal
    # value-validators that would *accept* a compiled plan
    # (`_torch_compile_errors`/`_runtime_policy_errors`) are de-pinned later.
    compile_backend: str = "eager"
    compile_dynamic: bool = False
    optimize_ddp: str = ""
    compiled_autograd: bool = False
    reorder_compute_comm_overlap: bool = False
    ddp_broadcast_buffers: bool = True
    ddp_find_unused_parameters: bool = False
    ddp_bucket_cap_mb: int | None = None
    fused_optimizer: bool = False
    compile_mode: str = "default"
    cudagraphs: str = "mode_default"
    inductor_options_json: str = "{}"
    autocast_cache_enabled: bool = True
    ddp_forward_sync_buffers: bool | None = None
    communication_hook: str = "none"
    nccl_environment_json: str = "{}"
    tf32_enabled: bool = True
    matmul_precision: str = "high"
    torch_version: str = ""
    cuda_version: str | None = None

    def expected_application(self) -> JsonObject:
        """Return the expected values a train run must actually apply.

        Returns:
            JSON-safe expected application values.

        """
        return {
            "selected_row_id": self.selected_row_id,
            "runtime_policy_id": self.runtime_policy_id,
            "accelerator_mode": self.accelerator_mode,
            "machine_shape": self.machine_shape,
            "world_size": self.world_size,
            "nproc_per_node": self.nproc_per_node,
            "torchrun_standalone": self.torchrun_standalone,
            "per_device_batch_size": self.per_device_batch_size,
            "global_batch_size": self.global_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "optimizer_updates_per_epoch": self.optimizer_updates_per_epoch,
            "precision_policy": self.precision_policy,
            "amp_enabled": self.amp_enabled,
            "autocast_dtype": self.autocast_dtype,
            "fp32_loss": self.fp32_loss,
            "grad_scaler_enabled": self.grad_scaler_enabled,
            "torch_compile_enabled": self.torch_compile_enabled,
            "compile_scope": self.compile_scope,
            "dataloader": {
                "num_workers": self.dataloader_num_workers,
                "prefetch_factor": self.dataloader_prefetch_factor,
                "pin_memory": self.dataloader_pin_memory,
                "persistent_workers": self.dataloader_persistent_workers,
                "non_blocking_h2d": self.dataloader_non_blocking_h2d,
            },
            "corruption_strategy": self.corruption_strategy,
            "memory_format": self.memory_format,
            "ddp_static_graph": self.ddp_static_graph,
            "ddp_gradient_as_bucket_view": self.ddp_gradient_as_bucket_view,
            "zero_grad_set_to_none": self.zero_grad_set_to_none,
            "gradient_clip_foreach": self.gradient_clip_foreach,
            "compile_backend": self.compile_backend,
            "compile_dynamic": self.compile_dynamic,
            "optimize_ddp": self.optimize_ddp,
            "compiled_autograd": self.compiled_autograd,
            "reorder_compute_comm_overlap": self.reorder_compute_comm_overlap,
            "ddp_broadcast_buffers": self.ddp_broadcast_buffers,
            "ddp_find_unused_parameters": self.ddp_find_unused_parameters,
            "ddp_bucket_cap_mb": self.ddp_bucket_cap_mb,
            "fused_optimizer": self.fused_optimizer,
            "compile_mode": self.compile_mode,
            "cudagraphs": self.cudagraphs,
            "inductor_options_json": self.inductor_options_json,
            "autocast_cache_enabled": self.autocast_cache_enabled,
            "ddp_forward_sync_buffers": self.ddp_forward_sync_buffers,
            "communication_hook": self.communication_hook,
            "nccl_environment_json": self.nccl_environment_json,
            "tf32_enabled": self.tf32_enabled,
            "matmul_precision": self.matmul_precision,
            "selected_runtime_artifact_sha256": self.artifact_sha256,
            "local_ddp_status": EXPECTED_DDP_APPLICATION_STATUS,
            "local_amp_status": expected_local_amp_status(self),
        }


@dataclass(frozen=True)
class SelectedRuntimeApplicationObservation:
    """Observed local settings used to prove the plan was applied."""

    selected_row_id: str
    runtime_policy_id: str
    accelerator_mode: str
    machine_shape: str
    world_size: int
    nproc_per_node: int
    torchrun_standalone: bool
    batch_size: int
    global_batch_size: int
    optimizer_updates_per_epoch: int
    amp_enabled: bool
    grad_scaler_enabled: bool
    fp32_loss: bool
    autocast_dtype: str
    torch_compile_enabled: bool
    compile_scope: str
    dataloader_num_workers: int
    dataloader_prefetch_factor: int | None
    dataloader_pin_memory: bool
    dataloader_persistent_workers: bool
    dataloader_non_blocking_h2d: bool
    corruption_strategy: str
    memory_format: str
    ddp_static_graph: bool
    ddp_gradient_as_bucket_view: bool
    zero_grad_set_to_none: bool
    # Spec 0011 S17c -- the compiled fast-path recipe knobs, observed so a plan
    # that records a compiled recipe but a run that applies a different one is caught.
    # Every knob but ``ddp_broadcast_buffers`` is checked for equality against the plan;
    # that one tolerates the upward override (see ``_application_mismatches``).
    compile_backend: str
    compile_dynamic: bool
    optimize_ddp: str
    compiled_autograd: bool
    reorder_compute_comm_overlap: bool
    ddp_broadcast_buffers: bool
    ddp_find_unused_parameters: bool
    ddp_bucket_cap_mb: int | None
    fused_optimizer: bool
    local_ddp_status: str
    local_amp_status: str
    gradient_clip_foreach: bool = True
    runner_amp_grad_scaler_init_scale: float | None = None
    compile_mode: str = "default"
    cudagraphs: str = "mode_default"
    inductor_options_json: str = "{}"
    autocast_cache_enabled: bool = True
    ddp_forward_sync_buffers: bool | None = None
    communication_hook: str = "none"
    nccl_environment_json: str = "{}"
    tf32_enabled: bool = True
    matmul_precision: str = "high"

    def as_json(self) -> JsonObject:
        """Return JSON-safe observed values.

        Returns:
            JSON-safe observed application values.

        """
        payload: JsonObject = {
            "selected_row_id": self.selected_row_id,
            "runtime_policy_id": self.runtime_policy_id,
            "accelerator_mode": self.accelerator_mode,
            "machine_shape": self.machine_shape,
            "world_size": self.world_size,
            "nproc_per_node": self.nproc_per_node,
            "torchrun_standalone": self.torchrun_standalone,
            "per_device_batch_size": self.batch_size,
            "global_batch_size": self.global_batch_size,
            "optimizer_updates_per_epoch": self.optimizer_updates_per_epoch,
            "amp_enabled": self.amp_enabled,
            "grad_scaler_enabled": self.grad_scaler_enabled,
            "fp32_loss": self.fp32_loss,
            "autocast_dtype": self.autocast_dtype,
            "torch_compile_enabled": self.torch_compile_enabled,
            "compile_scope": self.compile_scope,
            "dataloader": {
                "num_workers": self.dataloader_num_workers,
                "prefetch_factor": self.dataloader_prefetch_factor,
                "pin_memory": self.dataloader_pin_memory,
                "persistent_workers": self.dataloader_persistent_workers,
                "non_blocking_h2d": self.dataloader_non_blocking_h2d,
            },
            "corruption_strategy": self.corruption_strategy,
            "memory_format": self.memory_format,
            "ddp_static_graph": self.ddp_static_graph,
            "ddp_gradient_as_bucket_view": self.ddp_gradient_as_bucket_view,
            "zero_grad_set_to_none": self.zero_grad_set_to_none,
            "gradient_clip_foreach": self.gradient_clip_foreach,
            "compile_backend": self.compile_backend,
            "compile_dynamic": self.compile_dynamic,
            "optimize_ddp": self.optimize_ddp,
            "compiled_autograd": self.compiled_autograd,
            "reorder_compute_comm_overlap": self.reorder_compute_comm_overlap,
            "ddp_broadcast_buffers": self.ddp_broadcast_buffers,
            "ddp_find_unused_parameters": self.ddp_find_unused_parameters,
            "ddp_bucket_cap_mb": self.ddp_bucket_cap_mb,
            "fused_optimizer": self.fused_optimizer,
            "compile_mode": self.compile_mode,
            "cudagraphs": self.cudagraphs,
            "inductor_options_json": self.inductor_options_json,
            "autocast_cache_enabled": self.autocast_cache_enabled,
            "ddp_forward_sync_buffers": self.ddp_forward_sync_buffers,
            "communication_hook": self.communication_hook,
            "nccl_environment_json": self.nccl_environment_json,
            "tf32_enabled": self.tf32_enabled,
            "matmul_precision": self.matmul_precision,
            "local_ddp_status": self.local_ddp_status,
            "local_amp_status": self.local_amp_status,
        }
        if self.runner_amp_grad_scaler_init_scale is not None:
            payload["runner_amp_extension"] = {
                "grad_scaler_init_scale": self.runner_amp_grad_scaler_init_scale,
            }
        return payload


def expected_local_amp_status(plan: SelectedRuntimePlan) -> str:
    """Return the local-AMP application status a real run applies for this plan.

    The eager fallback runs the fp16-conservative AMP profile; the compiled
    bigger-batch winner runs ``amp_off_fp32`` (autocast and grad scaler off). The runner
    records the matching status when it actually executes on CUDA, so this expectation
    is derived from the plan's AMP toggle instead of a single frozen constant (Spec 0011
    S17c).

    Returns:
        The expected on-CUDA ``local_amp_status`` for ``plan``.

    """
    if plan.amp_enabled:
        return EXPECTED_AMP_APPLICATION_STATUS
    return EXPECTED_AMP_OFF_APPLICATION_STATUS


def expected_corruption_strategy(plan: SelectedRuntimePlan) -> str:
    """Return the corruption label a train run applies for this plan.

    Corruption is a fixed inline-stain property, not a plan choice: the
    compiled whole-step fast path (``torch_compile`` enabled with
    ``compile_scope == "step"``) fuses the seedless inline corruptor into the graph,
    while the eager training path draws it from the ordinary device RNG. Either way the
    plan's declared ``corruption.strategy`` is informational only; this returns the
    label the train steps actually record (Spec 0011 S17f).

    Returns:
        The expected observed ``corruption_strategy`` for a train run of ``plan``.

    """
    if plan.torch_compile_enabled and plan.compile_scope == _COMPILE_SCOPE_STEP:
        return COMPILED_FASTPATH_CORRUPTION_STRATEGY
    return EAGER_INLINE_STAIN_CORRUPTION_STRATEGY


def parse_selected_runtime_plan(path: Path) -> SelectedRuntimePlan:
    """Load and validate the selected-runtime plan.

    Returns:
        Parsed v5 selected-runtime plan.

    Raises:
        ValueError: If the payload is not the v5 fallback selected runtime.

    """
    payload = _load_json(path)
    errors = selected_runtime_plan_errors(payload, selected_runtime_path=path)
    if errors:
        message = "invalid selected runtime plan: " + ", ".join(errors)
        raise ValueError(message)
    return _plan_from_payload(path=path, payload=payload)


def selected_runtime_plan_errors(
    payload: JsonObject,
    *,
    selected_runtime_path: Path | None = None,
) -> tuple[str, ...]:
    """Return stable validation errors for a selected-runtime payload.

    Returns:
        Stable validation error identifiers.

    """
    source_errors = (
        _spec0011_winner_source_errors(
            payload=payload,
            selected_runtime_path=selected_runtime_path,
        )
        if payload.get("schema_version") == SPEC0011_SELECTED_RUNTIME_SCHEMA
        else _runtime_proof_errors(
            payload=payload,
            selected_runtime_path=selected_runtime_path,
        )
    )
    return (
        *_top_level_errors(payload),
        *_launch_errors(payload),
        *_snapshot_errors(payload),
        *_safety_errors(payload.get("safety")),
        *_runtime_policy_errors(payload.get("runtime_policy")),
        *_ddp_recipe_safety_errors(payload),
        *_torch_compile_errors(payload.get("torch_compile")),
        *source_errors,
    )


def _spec0011_winner_source_errors(  # noqa: C901, PLR0911, PLR0914
    *,
    payload: JsonObject,
    selected_runtime_path: Path | None,
) -> tuple[str, ...]:
    """Verify the compact plan against the immutable measured-winner record.

    The plan is a consumer translation, not a relabeled copy of the historical v5
    artifact. Its one provenance dependency is therefore the compact Spec 0011 winner
    JSON. Cross-checking the measured identity, batch, and recipe fields prevents a
    hand-edited plan from silently drifting away from the two final Kaggle confirmations
    while avoiding the retired audit-platform certificate tree.

    Returns:
        Stable source-winner validation errors.

    """
    source = payload.get("source_winner")
    if not isinstance(source, dict):
        return ("selected_runtime_missing_source_winner",)
    source_payload = cast("JsonObject", source)
    path_value = source_payload.get("path")
    digest = source_payload.get("sha256")
    errors: list[str] = []
    if not isinstance(path_value, str) or not path_value:
        errors.append("selected_runtime_source_winner_path_invalid")
    if not isinstance(digest, str) or len(digest) != _SHA256_HEX_LENGTH:
        errors.append("selected_runtime_source_winner_sha256_invalid")
    if errors or selected_runtime_path is None:
        return tuple(errors)
    winner_path = Path(cast("str", path_value))
    if not winner_path.is_absolute() and not winner_path.exists():
        winner_path = selected_runtime_path.parent / winner_path.name
    if not winner_path.exists():
        return ("selected_runtime_source_winner_missing",)
    if _sha256_file(winner_path) != digest:
        return ("selected_runtime_source_winner_sha256_mismatch",)
    try:
        winner = _load_json(winner_path)
    except (OSError, TypeError, ValueError):
        return ("selected_runtime_source_winner_unreadable",)
    selection = winner.get("selection")
    policy = winner.get("runtime_policy")
    if not isinstance(selection, dict) or not isinstance(policy, dict):
        return ("selected_runtime_source_winner_shape_invalid",)
    selection_payload = cast("JsonObject", selection)
    policy_payload = cast("JsonObject", policy)
    cross_checks = {
        "per_device_batch_size": (
            payload.get("per_device_batch_size"),
            selection_payload.get("per_device_batch_size"),
        ),
        "global_batch_size": (
            payload.get("global_batch_size"),
            selection_payload.get("global_batch_size"),
        ),
        "runtime_policy_id": (
            payload.get("runtime_policy_id"),
            policy_payload.get("runtime_policy_id"),
        ),
    }
    errors.extend(
        f"selected_runtime_source_winner_{name}_mismatch"
        for name, (plan_value, winner_value) in cross_checks.items()
        if plan_value != winner_value
    )
    snapshot = payload.get("selected_row_snapshot")
    if not isinstance(snapshot, dict):
        errors.append("selected_runtime_source_winner_snapshot_missing")
    else:
        snapshot_payload = cast("JsonObject", snapshot)
        stack_cross_checks = {
            "torch_version": (
                snapshot_payload.get("torch_version"),
                selection_payload.get("torch_version"),
            ),
            "cuda_version": (
                snapshot_payload.get("torch_cuda_version"),
                selection_payload.get("cuda_version"),
            ),
        }
        errors.extend(
            f"selected_runtime_source_winner_{name}_mismatch"
            for name, (snapshot_value, winner_value) in stack_cross_checks.items()
            if winner_value is not None and snapshot_value != winner_value
        )
    mixed = payload.get("mixed_precision")
    compile_block = payload.get("torch_compile")
    runtime = payload.get("runtime_policy")
    if (
        not isinstance(mixed, dict)
        or not isinstance(compile_block, dict)
        or not isinstance(
            runtime,
            dict,
        )
    ):
        return (*errors, "selected_runtime_source_winner_plan_recipe_missing")
    translated = {
        "precision_policy": cast("JsonObject", mixed).get("policy"),
        "autocast_dtype": cast("JsonObject", mixed).get("autocast_dtype"),
        "fp32_loss": cast("JsonObject", mixed).get("fp32_loss"),
        "grad_scaler_enabled": cast("JsonObject", mixed).get(
            "grad_scaler_enabled",
        ),
        "compile_scope": cast("JsonObject", compile_block).get("scope"),
        "memory_format": cast("JsonObject", runtime).get("memory_format"),
        "cudnn_benchmark": cast("JsonObject", runtime).get("cudnn_benchmark"),
        "cudnn_deterministic": cast("JsonObject", runtime).get(
            "cudnn_deterministic",
        ),
        "tf32_enabled": cast("JsonObject", runtime).get("tf32_enabled"),
        "matmul_precision": cast("JsonObject", runtime).get("matmul_precision"),
        "ddp_static_graph": cast("JsonObject", runtime).get("ddp_static_graph"),
        "ddp_gradient_as_bucket_view": cast("JsonObject", runtime).get(
            "ddp_gradient_as_bucket_view",
        ),
        "optimize_ddp": cast("JsonObject", compile_block).get("optimize_ddp"),
        "compiled_autograd": cast("JsonObject", compile_block).get(
            "compiled_autograd",
        ),
        "reorder_compute_comm_overlap": cast("JsonObject", compile_block).get(
            "reorder_compute_comm_overlap",
        ),
        "fused_optimizer": cast("JsonObject", runtime).get("fused_optimizer"),
        "ddp_broadcast_buffers": cast("JsonObject", runtime).get(
            "ddp_broadcast_buffers",
        ),
        "ddp_find_unused_parameters": cast("JsonObject", runtime).get(
            "ddp_find_unused_parameters",
        ),
        "ddp_bucket_cap_mb": cast("JsonObject", runtime).get("ddp_bucket_cap_mb"),
        "gradient_clip_foreach": cast("JsonObject", runtime).get(
            "gradient_clip_foreach",
        ),
        "zero_grad_set_to_none": cast("JsonObject", runtime).get(
            "zero_grad_set_to_none",
        ),
    }
    errors.extend(
        f"selected_runtime_source_winner_{name}_mismatch"
        for name, plan_value in translated.items()
        if plan_value != policy_payload.get(name)
    )
    return tuple(errors)


def selected_runtime_identity_payload(
    *,
    path: Path,
    payload: JsonObject,
    errors: tuple[str, ...],
) -> JsonObject:
    """Return the JSON identity block shared by train and gate artifacts.

    Returns:
        JSON-safe selected-runtime identity and validation detail.

    """
    snapshot = payload.get("selected_row_snapshot")
    selected_snapshot = snapshot if isinstance(snapshot, dict) else {}
    return cast(
        "JsonObject",
        {
            "path": str(path),
            "sha256": _sha256_file(path),
            "selected_row_id": _string_value(payload.get("selected_row_id")),
            "runtime_policy_id": _string_value(payload.get("runtime_policy_id")),
            "status": _string_value(payload.get("status")),
            "full_run_eligible": payload.get("full_run_eligible") is True,
            "full_training_launch_ready": payload.get("full_training_launch_ready")
            is True,
            "selected_row_snapshot": cast("JsonObject", selected_snapshot),
            "launch_blockers": _string_list(payload.get("launch_blockers")),
            "validation_errors": list(errors),
        },
    )


def build_plan_applied_proof(
    *,
    plan: SelectedRuntimePlan,
    observed: SelectedRuntimeApplicationObservation,
    status_scope: str = "local_selected_runtime_mechanics",
) -> JsonObject:
    """Build a proof that rejects recorded-but-unapplied selected runtime.

    Returns:
        Local non-promotable plan-application proof.

    """
    expected = plan.expected_application()
    observed_payload = observed.as_json()
    if observed.runner_amp_grad_scaler_init_scale is not None:
        expected["runner_amp_extension"] = {
            "grad_scaler_init_scale": EXPECTED_RUNNER_AMP_GRAD_SCALER_INIT_SCALE,
        }
    mismatches = _application_mismatches(
        plan=plan,
        observed=observed,
    )
    return {
        "status": "local_pass" if not mismatches else "fail",
        "status_scope": status_scope,
        "full_run_eligible": False,
        "selected_runtime_artifact_sha256": plan.artifact_sha256,
        "selected_row_id": plan.selected_row_id,
        "runtime_policy_id": plan.runtime_policy_id,
        "expected": expected,
        "observed": observed_payload,
        "plan_applied": not mismatches,
        "mismatches": list(mismatches),
        "failure_kind": "" if not mismatches else "selected_runtime_plan_not_applied",
    }


def fail_closed_plan_applied_proof(
    *,
    path: Path,
    payload: JsonObject,
    errors: tuple[str, ...],
    failure_kind: str,
) -> JsonObject:
    """Return a non-promotable failed plan proof for the gate-only path.

    Returns:
        Fail-closed plan proof without training observations.

    """
    return {
        "status": "fail",
        "status_scope": "fail_closed_real_gate_contract",
        "full_run_eligible": False,
        "selected_runtime": selected_runtime_identity_payload(
            path=path,
            payload=payload,
            errors=errors,
        ),
        "plan_applied": False,
        "mismatches": ["no_train_runner_observation"],
        "failure_kind": failure_kind,
    }


def _plan_from_payload(*, path: Path, payload: JsonObject) -> SelectedRuntimePlan:
    mixed_precision = _object(payload, "mixed_precision")
    dataloader = _object(payload, "dataloader")
    corruption = _object(payload, "corruption")
    runtime_policy = _object(payload, "runtime_policy")
    torch_compile = _object(payload, "torch_compile")
    selected_row_snapshot = _object(payload, "selected_row_snapshot")
    return SelectedRuntimePlan(
        path=path,
        artifact_sha256=_sha256_file(path),
        selected_row_id=_str(payload, "selected_row_id"),
        runtime_policy_id=_str(payload, "runtime_policy_id"),
        accelerator_mode=_str(payload, "accelerator_mode"),
        machine_shape=_str(payload, "machine_shape"),
        world_size=_int(payload, "world_size"),
        nproc_per_node=_int(payload, "nproc_per_node"),
        torchrun_standalone=True,
        per_device_batch_size=_int(payload, "per_device_batch_size"),
        global_batch_size=_int(payload, "global_batch_size"),
        gradient_accumulation_steps=_int(payload, "gradient_accumulation_steps"),
        optimizer_updates_per_epoch=_int(payload, "optimizer_updates_per_epoch"),
        precision_policy=_str(mixed_precision, "policy"),
        amp_enabled=_bool(mixed_precision, "enabled"),
        autocast_dtype=_str(mixed_precision, "autocast_dtype"),
        fp32_loss=_bool(mixed_precision, "fp32_loss"),
        grad_scaler_enabled=_bool(mixed_precision, "grad_scaler_enabled"),
        torch_compile_enabled=_bool(torch_compile, "enabled"),
        compile_scope=_str(torch_compile, "scope"),
        dataloader_num_workers=_int(dataloader, "num_workers"),
        dataloader_prefetch_factor=_optional_int(dataloader, "prefetch_factor"),
        dataloader_pin_memory=_bool(dataloader, "pin_memory"),
        dataloader_persistent_workers=_bool(dataloader, "persistent_workers"),
        dataloader_non_blocking_h2d=_bool(dataloader, "non_blocking_h2d"),
        corruption_strategy=_str(corruption, "strategy"),
        memory_format=_str(runtime_policy, "memory_format"),
        ddp_static_graph=_bool(runtime_policy, "ddp_static_graph"),
        ddp_gradient_as_bucket_view=_bool(
            runtime_policy,
            "ddp_gradient_as_bucket_view",
        ),
        zero_grad_set_to_none=_bool(runtime_policy, "zero_grad_set_to_none"),
        gradient_clip_foreach=(
            _bool(runtime_policy, "gradient_clip_foreach")
            if _bool_or(
                runtime_policy,
                "gradient_clip_foreach_applied",
                default=False,
            )
            else True
        ),
        # Spec 0011 S11 recipe knobs, from their frozen carrier homes with eager
        # defaults (absent on the committed v5 plan).
        compile_backend=_str_or(torch_compile, "backend", "eager"),
        compile_dynamic=_bool_or(torch_compile, "dynamic", default=False),
        optimize_ddp=_str_or(torch_compile, "optimize_ddp", ""),
        compiled_autograd=_bool_or(torch_compile, "compiled_autograd", default=False),
        reorder_compute_comm_overlap=_bool_or(
            torch_compile,
            "reorder_compute_comm_overlap",
            default=False,
        ),
        ddp_broadcast_buffers=_bool_or(
            runtime_policy,
            "ddp_broadcast_buffers",
            default=True,
        ),
        ddp_find_unused_parameters=_bool_or(
            runtime_policy,
            "ddp_find_unused_parameters",
            default=False,
        ),
        ddp_bucket_cap_mb=_optional_int_field(runtime_policy, "ddp_bucket_cap_mb"),
        fused_optimizer=_bool_or(runtime_policy, "fused_optimizer", default=False),
        compile_mode=_str_or(torch_compile, "mode", "default"),
        cudagraphs=_str_or(torch_compile, "cudagraphs", "mode_default"),
        inductor_options_json=_canonical_json_object_string(
            _str_or(torch_compile, "inductor_options_json", "{}"),
        ),
        autocast_cache_enabled=_bool_or(
            mixed_precision,
            "autocast_cache_enabled",
            default=True,
        ),
        ddp_forward_sync_buffers=_optional_bool_field(
            runtime_policy,
            "ddp_forward_sync_buffers",
        ),
        communication_hook=_str_or(runtime_policy, "communication_hook", "none"),
        nccl_environment_json=_canonical_json_object_string(
            _str_or(runtime_policy, "nccl_environment_json", "{}"),
        ),
        tf32_enabled=_bool_or(runtime_policy, "tf32_enabled", default=True),
        matmul_precision=_str_or(runtime_policy, "matmul_precision", "high"),
        torch_version=_str_or(selected_row_snapshot, "torch_version", ""),
        cuda_version=(_str_or(selected_row_snapshot, "torch_cuda_version", "") or None),
    )


def _composed_selected_row_id(payload: JsonObject) -> str | None:
    """Recompose the canonical selected row_id from the plan's own fields.

    The generator EMITS this id from the winning row's shape; the parser recomposes
    it from the plan's parsed fields and requires the recorded id to match, so the
    identity is self-consistent rather than pinned to the eager v5 literal (Spec 0011
    S17b) -- a re-measured winner (a bigger batch, the compiled amp-off recipe) is
    accepted only when its recorded id agrees with its own fields. Returns None when
    any composing field is missing or wrongly typed, so the identity checks fail closed:
    a plan whose own fields cannot compose an id is rejected, never silently accepted.

    Returns:
        The composed selected row_id, or None when a composing field is unusable.

    """
    accelerator_mode = payload.get("accelerator_mode")
    batch_size = payload.get("per_device_batch_size")
    runtime_policy_id = payload.get("runtime_policy_id")
    mixed_precision = payload.get("mixed_precision")
    torch_compile = payload.get("torch_compile")
    corruption = payload.get("corruption")
    if not (isinstance(accelerator_mode, str) and isinstance(runtime_policy_id, str)):
        return None
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        return None
    if not (
        isinstance(mixed_precision, dict)
        and isinstance(torch_compile, dict)
        and isinstance(corruption, dict)
    ):
        return None
    precision_policy = cast("JsonObject", mixed_precision).get("policy")
    compile_scope = cast("JsonObject", torch_compile).get("scope")
    corruption_strategy = cast("JsonObject", corruption).get("strategy")
    if not (
        isinstance(precision_policy, str)
        and isinstance(compile_scope, str)
        and isinstance(corruption_strategy, str)
    ):
        return None
    return compose_selected_row_id(
        accelerator_mode=accelerator_mode,
        batch_size=batch_size,
        precision_policy=precision_policy,
        compile_scope=compile_scope,
        corruption_strategy=corruption_strategy,
        runtime_policy_id=runtime_policy_id,
    )


def _selected_identity_errors(payload: JsonObject) -> tuple[str, ...]:
    """Return structural identity errors for the plan's own row_id / policy id.

    Spec 0011 S17b de-pins the identity from the eager v5 literal: the recorded
    ``selected_row_id`` must equal the id recomposed from the plan's own fields (so a
    re-measured winner is accepted only when its id is self-consistent), and the
    ``runtime_policy_id`` -- the free label the row_id encodes -- must be a non-empty
    string. The hardware/status anchors stay pinned in ``_launch_errors``, so identity
    de-pinning cannot admit a different accelerator or topology.

    Returns:
        Structural identity error identifiers.

    """
    errors: list[str] = []
    composed = _composed_selected_row_id(payload)
    if composed is None or payload.get("selected_row_id") != composed:
        errors.append("selected_runtime_selected_row_id_not_self_consistent")
    runtime_policy_id = payload.get("runtime_policy_id")
    if not isinstance(runtime_policy_id, str) or not runtime_policy_id:
        errors.append("selected_runtime_runtime_policy_id_missing")
    return tuple(errors)


def composed_selected_runtime_identity(
    payload: JsonObject,
) -> tuple[str | None, str | None]:
    """Recompose (selected_row_id, runtime_policy_id) from a plan payload's own fields.

    The single source for what a plan's identity *is*: every cross-check that compares a
    recorded identity against the plan -- the runtime proof here, a downloaded
    ``gate_health.csv`` in the gate -- derives its expectation from this, so a
    re-measured winner (a bigger batch, the compiled amp-off recipe) is accepted without
    a Kaggle re-point while an inconsistent bundle still fails closed (Spec 0011 S17b).
    ``selected_row_id`` is recomposed structurally from the plan's own fields (None when
    any composing field is missing or wrongly typed); ``runtime_policy_id`` is the
    recorded free label (None when missing, empty, or non-string -- an empty label
    identifies nothing, so it must not satisfy a cross-check). A None component makes
    the caller's identity check fail closed rather than silently accept.

    Returns:
        The structurally composed selected row_id (or None) and the recorded runtime
        policy id (or None).

    """
    composed = _composed_selected_row_id(payload)
    runtime_policy_id = payload.get("runtime_policy_id")
    policy = (
        runtime_policy_id
        if isinstance(runtime_policy_id, str) and runtime_policy_id
        else None
    )
    return composed, policy


def _top_level_errors(payload: JsonObject) -> tuple[str, ...]:
    expected = {
        "status": "pass",
        "benchmark_kind": "kaggle_runtime_selection",
        "benchmark_source": "kaggle_runtime_benchmark",
    }
    error_names = {
        "status": "selected_runtime_status_not_pass",
        "benchmark_kind": "selected_runtime_wrong_benchmark_kind",
        "benchmark_source": "selected_runtime_wrong_benchmark_source",
    }
    errors = [
        error_names[key]
        for key, expected_value in expected.items()
        if payload.get(key) != expected_value
    ]
    errors.extend(_selected_identity_errors(payload))
    if payload.get("full_run_eligible") is not True:
        errors.append("selected_runtime_not_full_run_eligible")
    if payload.get("full_training_launch_ready") is not False:
        errors.append("selected_runtime_already_claims_launch_ready")
    return tuple(errors)


def _launch_errors(payload: JsonObject) -> tuple[str, ...]:
    errors: list[str] = []
    # Hardware/policy anchors stay pinned: a run cannot self-declare a different
    # accelerator, topology, or gradient-accumulation policy. The batch and schedule
    # are de-pinned to goal-derived relationships (Spec 0011 S7) in
    # _launch_schedule_errors, so a re-measured non-24 batch is accepted.
    expected = {
        "accelerator_mode": "dual_t4_ddp",
        "machine_shape": EXPECTED_MACHINE_SHAPE,
        "world_size": 2,
        "nproc_per_node": 2,
        "gradient_accumulation_steps": 1,
    }
    error_names = {
        "accelerator_mode": "selected_runtime_top_level_not_dual_t4_ddp",
        "machine_shape": "selected_runtime_top_level_wrong_machine_shape",
        "world_size": "selected_runtime_top_level_wrong_world_size",
        "nproc_per_node": "selected_runtime_top_level_wrong_nproc_per_node",
        "gradient_accumulation_steps": (
            "selected_runtime_top_level_wrong_gradient_accumulation"
        ),
    }
    errors.extend(
        error_names[key]
        for key, expected_value in expected.items()
        if payload.get(key) != expected_value
    )
    errors.extend(_launch_schedule_errors(payload))
    mixed_precision = payload.get("mixed_precision")
    if not isinstance(mixed_precision, dict):
        errors.append("selected_runtime_missing_mixed_precision")
    else:
        errors.extend(_mixed_precision_errors(cast("JsonObject", mixed_precision)))
    dataloader = payload.get("dataloader")
    if not isinstance(dataloader, dict):
        errors.append("selected_runtime_missing_dataloader")
    else:
        errors.extend(_dataloader_errors(cast("JsonObject", dataloader)))
    corruption = payload.get("corruption")
    if not isinstance(corruption, dict):
        errors.append("selected_runtime_missing_corruption")
    else:
        # Corruption is a fixed inline-stain runtime property (Spec 0011 S17f), no
        # longer a selected axis, so the strategy value is informational: require only
        # that the block carries a non-empty string, not a pinned literal.
        strategy = corruption.get("strategy")
        if not isinstance(strategy, str) or not strategy:
            errors.append("selected_runtime_corruption_strategy_not_string")
    return tuple(errors)


def _launch_schedule_errors(payload: JsonObject) -> tuple[str, ...]:
    """Return goal-derived batch/schedule relationship errors (Spec 0011 S7).

    The batch is a measured output of the runtime search, so the parser no longer pins
    it to the reference literals. Instead it validates the relationships every plan must
    satisfy: the per-device batch is a positive integer, the global batch is
    ``per_device_batch_size * world_size``, and the plan's own recorded
    ``optimizer_updates_per_epoch`` equals
    ``floor(REAL_TRAIN_PATCH_COUNT / global_batch_size)`` via the single-sourced
    ``training_steps_per_epoch`` helper. That last check is the plan-recorded schedule
    cross-check deferred from S6d: the parser is where the recorded number is validated
    against the derivation. At the reference global batch 24 these reproduce the
    committed 12/24/12500 literals exactly, so parsing the committed plan is unchanged.

    Returns:
        Stable batch/schedule relationship error identifiers.

    """
    errors: list[str] = []
    per_device = _launch_positive_int_or_none(payload.get("per_device_batch_size"))
    world_size = _launch_positive_int_or_none(payload.get("world_size"))
    global_batch = _launch_positive_int_or_none(payload.get("global_batch_size"))
    updates = _launch_int_or_none(payload.get("optimizer_updates_per_epoch"))
    if per_device is None:
        errors.append("selected_runtime_top_level_wrong_per_device_batch")
    if not _global_batch_matches_product(global_batch, per_device, world_size):
        errors.append("selected_runtime_top_level_wrong_global_batch")
    if not _updates_match_derivation(updates, global_batch):
        errors.append("selected_runtime_top_level_wrong_optimizer_updates_per_epoch")
    return tuple(errors)


def _global_batch_matches_product(
    global_batch: int | None,
    per_device: int | None,
    world_size: int | None,
) -> bool:
    if global_batch is None or per_device is None or world_size is None:
        return False
    return global_batch == per_device * world_size


def _updates_match_derivation(updates: int | None, global_batch: int | None) -> bool:
    if updates is None or global_batch is None:
        return False
    return updates == training_steps_per_epoch(
        real_train_patch_count=REAL_TRAIN_PATCH_COUNT,
        global_batch_size=global_batch,
    )


def _launch_positive_int_or_none(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _launch_int_or_none(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


# Spec 0011 S17a -- coherence-based recipe value validators. The runtime search emits
# the measured winner profile: either the eager AMP small-batch fallback or the compiled
# bigger-batch recipe. So these validators accept any internally coherent member of that
# measured profile space instead of pinning the eager v5 literals. The safety anchors
# stay pinned: the FP32 loss island is always required, ddp_static_graph stays False
# (DDPOptimizer forbids it, and the compiled whole-step interleaves an eager proof
# backward on one DDP module), zero_grad_set_to_none stays True, and the DDPOptimizer
# flag guard (_ddp_optimizer_safety_errors) keeps firing. Identity (row_id /
# runtime_policy_id) and the snapshot batch/precision literals are re-pointed separately
# (Spec 0011 S17b / Kaggle row_id mint), not here.
_AMP_ON_PRECISION_POLICIES = frozenset({"amp_conservative", "amp_scalar_gate_relaxed"})
_AMP_OFF_PRECISION_POLICY = "amp_off_fp32"
_AMP_AUTOCAST_DTYPE = "float16"
# Autocast dtypes only an AMP-on profile may declare; an amp-off plan claiming one is
# internally incoherent (autocast is disabled, so the dtype would never be consumed).
_AMP_AUTOCAST_DTYPES = frozenset({"float16", "bfloat16"})
_COMPILE_EAGER_BACKEND = "eager"
_COMPILE_INDUCTOR_BACKEND = "inductor"
_COMPILE_SCOPE_NONE = "none"
_COMPILE_SCOPE_STEP = "step"
_STABLE_COMPILE_SCOPES = frozenset({"model_forward", "step"})
_ALLOWED_MEMORY_FORMATS = frozenset({"contiguous", "channels_last"})


def _mixed_precision_errors(payload: JsonObject) -> tuple[str, ...]:
    """Return mixed-precision coherence errors (Spec 0011 S17a).

    The runtime search emits the winner precision profile, which is either an AMP
    profile (``amp_conservative`` / ``amp_scalar_gate_relaxed`` -- fp16 autocast with a
    grad scaler) or the compiled winner's ``amp_off_fp32`` (autocast and scaler off).
    Both keep the FP32 loss island, so this validator requires ``fp32_loss`` in every
    profile and checks the AMP fields against the declared policy instead of pinning the
    eager v5 AMP literals. An unknown policy, or AMP fields that contradict the policy,
    fails closed.

    Returns:
        Stable mixed-precision coherence error identifiers.

    """
    errors = list(_autocast_cache_errors(payload))
    if payload.get("fp32_loss") is not True:
        errors.append("selected_runtime_mixed_precision_missing_fp32_loss")
    policy = payload.get("policy")
    if policy in _AMP_ON_PRECISION_POLICIES:
        if payload.get("enabled") is not True:
            errors.append("selected_runtime_mixed_precision_not_enabled")
        if payload.get("autocast_dtype") != _AMP_AUTOCAST_DTYPE:
            errors.append("selected_runtime_mixed_precision_wrong_dtype")
        if payload.get("grad_scaler_enabled") is not True:
            errors.append("selected_runtime_mixed_precision_missing_scaler")
    elif policy == _AMP_OFF_PRECISION_POLICY:
        if payload.get("enabled") is not False:
            errors.append("selected_runtime_mixed_precision_amp_off_not_disabled")
        if payload.get("grad_scaler_enabled") is not False:
            errors.append("selected_runtime_mixed_precision_amp_off_scaler_enabled")
        if payload.get("autocast_dtype") in _AMP_AUTOCAST_DTYPES:
            errors.append("selected_runtime_mixed_precision_amp_off_autocast_dtype")
    else:
        errors.append("selected_runtime_mixed_precision_wrong_policy")
    return tuple(errors)


def _autocast_cache_errors(payload: JsonObject) -> tuple[str, ...]:
    if "autocast_cache_enabled" in payload and not isinstance(
        payload.get("autocast_cache_enabled"),
        bool,
    ):
        return ("selected_runtime_mixed_precision_bad_autocast_cache",)
    return ()


def _dataloader_errors(payload: JsonObject) -> tuple[str, ...]:
    expected = {
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
        "non_blocking_h2d": True,
    }
    error_names = {
        "num_workers": "selected_runtime_dataloader_wrong_num_workers",
        "pin_memory": "selected_runtime_dataloader_wrong_pin_memory",
        "persistent_workers": "selected_runtime_dataloader_wrong_persistent_workers",
        "non_blocking_h2d": "selected_runtime_dataloader_wrong_non_blocking_h2d",
    }
    errors = [
        error_names[key]
        for key, expected_value in expected.items()
        if payload.get(key) != expected_value
    ]
    if payload.get("prefetch_factor") is not None:
        errors.append("selected_runtime_dataloader_wrong_prefetch_factor")
    return tuple(errors)


def _snapshot_errors(payload: JsonObject) -> tuple[str, ...]:
    """Return selected-row snapshot coherence errors (Spec 0011 S17b).

    The snapshot embeds the winning benchmark row for provenance; every cell is a
    string. The identity and batch/precision cells are cross-checked against the plan's
    own parsed fields (so a re-measured winner whose snapshot agrees with its plan is
    accepted), while the hardware/status anchors -- accelerator, machine shape, world
    size, nproc, status -- stay pinned. Corruption is no longer pinned here: it is a
    fixed inline-stain runtime property (Spec 0011 S17f), not a selected axis. A None
    expected value (an unusable plan field) fails that cell closed.

    Returns:
        Stable selected-row snapshot coherence error identifiers.

    """
    snapshot = payload.get("selected_row_snapshot")
    if not isinstance(snapshot, dict):
        return ("selected_runtime_missing_snapshot",)
    snapshot_payload = cast("dict[str, object]", snapshot)
    mixed_precision = payload.get("mixed_precision")
    mixed: JsonObject = (
        cast("JsonObject", mixed_precision) if isinstance(mixed_precision, dict) else {}
    )
    expected: dict[str, object | None] = {
        "row_id": _composed_selected_row_id(payload),
        "runtime_policy_id": _str_or_none(payload.get("runtime_policy_id")),
        "status": "pass",
        "accelerator_mode": "dual_t4_ddp",
        "machine_shape": EXPECTED_MACHINE_SHAPE,
        "precision_policy": _str_or_none(mixed.get("policy")),
        "nproc_per_node": "2",
        "per_device_batch_size": _int_as_snapshot_str(
            payload.get("per_device_batch_size"),
        ),
        "global_batch_size": _int_as_snapshot_str(payload.get("global_batch_size")),
        "grad_scaler_enabled": _bool_as_snapshot_str(mixed.get("grad_scaler_enabled")),
        "autocast_dtype": _str_or_none(mixed.get("autocast_dtype")),
    }
    error_names = {
        "row_id": "selected_runtime_snapshot_row_mismatch",
        "runtime_policy_id": "selected_runtime_snapshot_policy_mismatch",
        "status": "selected_runtime_snapshot_status_not_pass",
        "accelerator_mode": "selected_runtime_snapshot_not_dual_t4_ddp",
        "machine_shape": "selected_runtime_snapshot_wrong_machine_shape",
        "precision_policy": "selected_runtime_snapshot_wrong_precision_policy",
        "nproc_per_node": "selected_runtime_snapshot_wrong_nproc_per_node",
        "per_device_batch_size": "selected_runtime_snapshot_wrong_per_device_batch",
        "global_batch_size": "selected_runtime_snapshot_wrong_global_batch",
        "grad_scaler_enabled": "selected_runtime_snapshot_missing_scaler",
        "autocast_dtype": "selected_runtime_snapshot_wrong_autocast_dtype",
    }
    if payload.get("schema_version") == SPEC0011_SELECTED_RUNTIME_SCHEMA:
        expected.update(
            {
                "torch_version": EXPECTED_SELECTED_TORCH_VERSION,
                "torch_cuda_version": EXPECTED_SELECTED_CUDA_VERSION,
            },
        )
        error_names.update(
            {
                "torch_version": "selected_runtime_snapshot_wrong_torch_version",
                "torch_cuda_version": ("selected_runtime_snapshot_wrong_cuda_version"),
            },
        )
    errors = [
        error_names[key]
        for key, expected_value in expected.items()
        if expected_value is None or snapshot_payload.get(key) != expected_value
    ]
    if snapshot_payload.get("world_size") not in {2, "2"}:
        errors.append("selected_runtime_snapshot_wrong_world_size")
    return tuple(errors)


def _safety_errors(safety: object) -> tuple[str, ...]:
    if not isinstance(safety, dict):
        return ("selected_runtime_missing_safety",)
    safety_payload = cast("dict[str, object]", safety)
    keys = (
        "dataloader_status",
        "numerical_check_status",
        "corruption_check_status",
        "gate_health_status",
    )
    errors = [
        f"selected_runtime_safety_{key}_not_pass"
        for key in keys
        if safety_payload.get(key) != "pass"
    ]
    if safety_payload.get("amp_step_skipped_count") != 0:
        errors.append("selected_runtime_safety_amp_skips_not_zero")
    return tuple(errors)


def _runtime_policy_errors(policy: object) -> tuple[str, ...]:
    """Return runtime-policy coherence errors (Spec 0011 S17a).

    ``memory_format`` may be the eager ``contiguous`` or the compiled winner's
    ``channels_last``, and ``ddp_gradient_as_bucket_view`` is a measured performance
    flag (the winner enables it), so neither is pinned to the eager literal.
    ``ddp_static_graph`` stays pinned False -- DDPOptimizer forbids it and the compiled
    whole-step interleaves an eager proof backward on one DDP module -- and
    ``zero_grad_set_to_none`` stays True.

    Returns:
        Stable runtime-policy coherence error identifiers.

    """
    if not isinstance(policy, dict):
        return ("selected_runtime_missing_runtime_policy",)
    payload = cast("JsonObject", policy)
    errors = list(_runtime_backend_precision_errors(payload))
    if payload.get("memory_format") not in _ALLOWED_MEMORY_FORMATS:
        errors.append("selected_runtime_runtime_policy_memory_format_mismatch")
    if payload.get("ddp_static_graph") is not False:
        errors.append("selected_runtime_runtime_policy_ddp_static_graph_mismatch")
    if payload.get("zero_grad_set_to_none") is not True:
        errors.append(
            "selected_runtime_runtime_policy_zero_grad_set_to_none_mismatch",
        )
    if payload.get("ddp_find_unused_parameters", False) is not False:
        errors.append("selected_runtime_runtime_policy_find_unused_mismatch")
    forward_sync = payload.get("ddp_forward_sync_buffers")
    if forward_sync is not None and not isinstance(forward_sync, bool):
        errors.append("selected_runtime_runtime_policy_forward_sync_mismatch")
    if payload.get("communication_hook", "none") not in {
        "none",
        "fp16_compress_hook",
        "bf16_compress_hook",
    }:
        errors.append("selected_runtime_runtime_policy_communication_hook_mismatch")
    if not _json_string_map_is_valid(
        payload.get("nccl_environment_json", "{}"),
    ):
        errors.append("selected_runtime_runtime_policy_nccl_environment_mismatch")
    return tuple(errors)


def _runtime_backend_precision_errors(payload: JsonObject) -> tuple[str, ...]:
    errors: list[str] = []
    if not isinstance(payload.get("tf32_enabled", True), bool):
        errors.append("selected_runtime_runtime_policy_tf32_mismatch")
    if payload.get("matmul_precision", "high") not in {"highest", "high", "medium"}:
        errors.append("selected_runtime_runtime_policy_matmul_precision_mismatch")
    return tuple(errors)


def _torch_compile_errors(torch_compile: object) -> tuple[str, ...]:
    """Return torch.compile coherence errors (Spec 0011 S17a).

    Accepts the eager profile (disabled, scope ``none``, ``eager`` backend) or a stable
    compiled profile (enabled, scope in the settle-proven set ``{model_forward, step}``,
    ``inductor`` backend). ``dynamic`` must stay False in both because the compiled step
    is traced with static shapes, so a dynamic plan would recompile every batch. The
    enabled/scope/backend fields must agree, so a plan cannot claim compilation while
    carrying an eager backend or a diagnostic scope, or vice versa.

    Returns:
        Stable torch.compile coherence error identifiers.

    """
    if not isinstance(torch_compile, dict):
        return ("selected_runtime_missing_torch_compile",)
    payload = cast("JsonObject", torch_compile)
    errors = list(_compile_invocation_errors(payload))
    if payload.get("dynamic") is not False:
        errors.append("selected_runtime_torch_compile_dynamic_mismatch")
    enabled = payload.get("enabled")
    if enabled is False:
        if payload.get("scope") != _COMPILE_SCOPE_NONE:
            errors.append("selected_runtime_torch_compile_scope_mismatch")
        if payload.get("backend") != _COMPILE_EAGER_BACKEND:
            errors.append("selected_runtime_torch_compile_backend_mismatch")
    elif enabled is True:
        if payload.get("scope") not in _STABLE_COMPILE_SCOPES:
            errors.append("selected_runtime_torch_compile_scope_mismatch")
        if payload.get("backend") != _COMPILE_INDUCTOR_BACKEND:
            errors.append("selected_runtime_torch_compile_backend_mismatch")
        if payload.get("optimize_ddp") not in {
            "ddp_optimizer",
            "python_reducer",
            "python_reducer_without_compiled_forward",
            "no_optimization",
        }:
            errors.append("selected_runtime_torch_compile_optimize_ddp_mismatch")
    else:
        errors.append("selected_runtime_torch_compile_enabled_mismatch")
    return tuple(errors)


def _compile_invocation_errors(payload: JsonObject) -> tuple[str, ...]:
    errors: list[str] = []
    mode = payload.get("mode", "default")
    if not isinstance(mode, str) or not mode:
        errors.append("selected_runtime_torch_compile_mode_mismatch")
    if payload.get("cudagraphs", "mode_default") not in {
        "mode_default",
        "enabled",
        "disabled",
    }:
        errors.append("selected_runtime_torch_compile_cudagraphs_mismatch")
    if not _json_object_string_is_valid(
        payload.get("inductor_options_json", "{}"),
    ):
        errors.append("selected_runtime_torch_compile_options_mismatch")
    return tuple(errors)


_RECIPE_CARRIER_BLOCK_KEYS = ("runtime_policy", "torch_compile")


def _ddp_recipe_safety_errors(payload: JsonObject) -> tuple[str, ...]:
    """Reject incoherent DDP modes and unsafe DDPOptimizer flags.

    ``optimize_ddp="ddp_optimizer"`` (DDPOptimizer) splits the backward at DDP bucket
    boundaries. The measured recipe keeps ``static_graph`` and
    ``find_unused_parameters`` false. PyTorch 2.13 additionally requires
    ``python_reducer`` to use compiled autograd and forbids compiled autograd with
    ``no_optimization``. It does not document a DDPOptimizer/compiled-autograd ban, so
    that combination remains expressible for a future measured row.

    The committed eager fallback has no mode, so these checks are a no-op there.

    Returns:
        Stable DDPOptimizer safety error identifiers.

    """
    optimize_ddp = _recipe_field(payload, "optimize_ddp")
    compiled_autograd = _recipe_flag_enabled(payload, "compiled_autograd")
    errors: list[str] = []
    if (
        optimize_ddp
        in {
            "python_reducer",
            "python_reducer_without_compiled_forward",
        }
        and not compiled_autograd
    ):
        errors.append("selected_runtime_python_reducer_requires_compiled_autograd")
    if optimize_ddp == "no_optimization" and compiled_autograd:
        errors.append("selected_runtime_no_optimization_compiled_autograd_conflict")
    if optimize_ddp != "ddp_optimizer":
        return tuple(errors)
    if _recipe_flag_enabled(payload, "ddp_static_graph", "static_graph"):
        errors.append("selected_runtime_ddp_optimizer_static_graph_conflict")
    if _recipe_flag_enabled(
        payload,
        "ddp_find_unused_parameters",
        "find_unused_parameters",
    ):
        errors.append(
            "selected_runtime_ddp_optimizer_find_unused_parameters_conflict",
        )
    return tuple(errors)


def _recipe_field(payload: JsonObject, key: str) -> JsonValue | None:
    """Read a recipe field from whichever plan block declares it.

    Spec 0011 S11 froze the recipe carrier homes, and this ``runtime_policy`` ->
    ``torch_compile`` -> top-level read order already resolves each knob from its
    frozen home: the DDP knobs the guard reads (``ddp_static_graph``,
    ``ddp_find_unused_parameters``) live in ``runtime_policy`` (matched first), and
    the dynamo knobs (``optimize_ddp``, ``compiled_autograd``) live only in
    ``torch_compile`` (reached on the ``runtime_policy`` miss). ``_plan_from_payload``
    parses the same knobs from the same homes, so the pre-parse guard and the parsed
    plan agree. Returns ``None`` when no block declares the key, which keeps the
    DDPOptimizer safety check a no-op on the eager v5 plan (it declares no knob).

    First-carrier-wins assumes each flag has a single home, which the honest generator
    guarantees (it emits each knob into exactly one block); a plan that contradictorily
    declares the same flag in two blocks is malformed input this safety net does not
    reconcile.

    Returns:
        The declared recipe value, or ``None`` when absent everywhere.

    """
    for block_key in _RECIPE_CARRIER_BLOCK_KEYS:
        block = payload.get(block_key)
        if isinstance(block, dict):
            block_payload = cast("JsonObject", block)
            if key in block_payload:
                return block_payload.get(key)
    return payload.get(key)


def _recipe_flag_enabled(payload: JsonObject, *keys: str) -> bool:
    return any(_recipe_field(payload, key) is True for key in keys)


def _runtime_proof_errors(  # noqa: PLR0911
    *,
    payload: JsonObject,
    selected_runtime_path: Path | None,
) -> tuple[str, ...]:
    if selected_runtime_path is None:
        return ()
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        return ("selected_runtime_missing_artifacts",)
    artifact_payload = cast("JsonObject", artifacts)
    runtime_proof_value = artifact_payload.get("runtime_proof")
    runtime_proof_hash = artifact_payload.get("runtime_proof_sha256")
    if not isinstance(runtime_proof_value, str) or not runtime_proof_value:
        return ("selected_runtime_missing_runtime_proof_artifact",)
    if not isinstance(runtime_proof_hash, str) or not runtime_proof_hash:
        return ("selected_runtime_missing_runtime_proof_sha256",)

    proof_path = _linked_artifact_path(
        selected_runtime_path=selected_runtime_path,
        artifact_path=runtime_proof_value,
    )
    if not proof_path.exists():
        return ("selected_runtime_runtime_proof_missing",)
    if _sha256_file(proof_path) != runtime_proof_hash:
        return ("selected_runtime_runtime_proof_sha256_mismatch",)
    try:
        proof_payload = _load_json(proof_path)
    except (OSError, TypeError, ValueError):
        return ("selected_runtime_runtime_proof_unreadable",)

    expected_row_id, expected_policy_id = composed_selected_runtime_identity(payload)
    proof_errors = _runtime_proof_payload_errors(
        proof_payload,
        expected_row_id=expected_row_id,
        expected_policy_id=expected_policy_id,
    )
    command_errors = _runtime_proof_launch_command_errors(proof_payload)
    return (*proof_errors, *command_errors)


def _runtime_proof_payload_errors(
    payload: JsonObject,
    *,
    expected_row_id: str | None,
    expected_policy_id: str | None,
) -> tuple[str, ...]:
    errors: list[str] = []
    expected_top_level = {
        "schema_version": "spec0001.runtime_selection.v1",
        "benchmark_kind": "kaggle_runtime_selection",
        "benchmark_source": "kaggle_runtime_benchmark",
        "status": "pass",
        "full_run_eligible": True,
        "selection_ready": True,
        "selected_runtime_written": True,
        "machine_shape": EXPECTED_MACHINE_SHAPE,
        "model_count_status": "pass",
        "stain_corruptor_qa_status": "pass",
    }
    errors.extend(
        f"selected_runtime_runtime_proof_{key}_mismatch"
        for key, expected in expected_top_level.items()
        if payload.get(key) != expected
    )

    errors.extend(
        _runtime_proof_dual_gate_errors(payload.get("dual_t4_train_step_gate")),
    )
    errors.extend(
        _runtime_proof_environment_errors(payload.get("runtime_environment")),
    )
    errors.extend(
        _runtime_proof_write_decision_errors(
            payload.get("selected_runtime_write_decision"),
            expected_row_id=expected_row_id,
        ),
    )
    errors.extend(
        _runtime_proof_efficiency_errors(
            payload.get("efficiency_followup"),
            expected_row_id=expected_row_id,
            expected_policy_id=expected_policy_id,
        ),
    )
    errors.extend(
        _runtime_proof_amp_followup_errors(payload.get("amp_followup_policy")),
    )
    return tuple(errors)


def _runtime_proof_dual_gate_errors(dual_gate: object) -> tuple[str, ...]:
    if not isinstance(dual_gate, dict):
        return ("selected_runtime_runtime_proof_missing_dual_t4_gate",)
    payload = cast("JsonObject", dual_gate)
    expected = {
        "status": "pass",
        "child_process_launch_status": "pass",
        "cuda_device_count": 2,
        "visible_device_count": 2,
        "world_size": 2,
        "nproc_per_node": 2,
        "rank_assignment_status": "pass",
        "required_before_selected_runtime": True,
        "failure_policy": "do_not_write_selected_runtime_if_missing_failed_or_skipped",
    }
    errors = [
        f"selected_runtime_runtime_proof_dual_gate_{key}_mismatch"
        for key, expected_value in expected.items()
        if payload.get(key) != expected_value
    ]
    if _object_list_len(payload.get("rank_assignments")) != EXPECTED_DUAL_T4_RANK_COUNT:
        errors.append("selected_runtime_runtime_proof_dual_gate_rank_count_mismatch")
    elif not _rank_assignments_are_dual_t4(payload.get("rank_assignments")):
        errors.append(
            "selected_runtime_runtime_proof_dual_gate_rank_assignment_mismatch",
        )
    errors.extend(
        f"selected_runtime_runtime_proof_dual_gate_{key}_not_empty"
        for key in (
            "linked_failure_reasons",
            "missing_dual_row_ids",
            "nonpassing_dual_row_ids",
        )
        if payload.get(key) != []
    )
    return tuple(errors)


def _runtime_proof_environment_errors(environment: object) -> tuple[str, ...]:
    if not isinstance(environment, dict):
        return ("selected_runtime_runtime_proof_missing_runtime_environment",)
    payload = cast("JsonObject", environment)
    expected = {
        "status": "pass",
        "child_process_returncode": 0,
        "cuda_device_count": 2,
        "visible_device_count": 2,
        "world_size": 2,
        "nproc_per_node": 2,
        "machine_shape": EXPECTED_MACHINE_SHAPE,
        "failure_kind": "",
        "failure_message_hash": "",
    }
    errors = [
        f"selected_runtime_runtime_proof_environment_{key}_mismatch"
        for key, expected_value in expected.items()
        if payload.get(key) != expected_value
    ]
    if _object_list_len(payload.get("rank_assignments")) != EXPECTED_DUAL_T4_RANK_COUNT:
        errors.append("selected_runtime_runtime_proof_environment_rank_count_mismatch")
    elif not _rank_assignments_are_dual_t4(payload.get("rank_assignments")):
        errors.append(
            "selected_runtime_runtime_proof_environment_rank_assignment_mismatch",
        )
    return tuple(errors)


def _runtime_proof_write_decision_errors(
    decision: object,
    *,
    expected_row_id: str | None,
) -> tuple[str, ...]:
    if not isinstance(decision, dict):
        return ("selected_runtime_runtime_proof_missing_write_decision",)
    payload = cast("JsonObject", decision)
    expected = {
        "allowed": True,
        "policy": EXPECTED_RUNTIME_PROOF_WRITE_POLICY,
        "stain_corruptor_qa_status": "pass",
    }
    errors = [
        f"selected_runtime_runtime_proof_write_decision_{key}_mismatch"
        for key, expected_value in expected.items()
        if payload.get(key) != expected_value
    ]
    # Spec 0011 S17b: the proof's selected_row_id must match the id recomposed from the
    # plan's own fields, not the eager v5 literal.
    if expected_row_id is None or payload.get("selected_row_id") != expected_row_id:
        errors.append(
            "selected_runtime_runtime_proof_write_decision_selected_row_id_mismatch",
        )
    errors.extend(
        f"selected_runtime_runtime_proof_write_decision_{key}_not_empty"
        for key in (
            "blockers",
            "linked_pass_row_failures",
            "stain_corruptor_qa_missing_candidate_row_ids",
        )
        if payload.get(key) != []
    )
    return tuple(errors)


def _runtime_proof_efficiency_errors(
    efficiency: object,
    *,
    expected_row_id: str | None,
    expected_policy_id: str | None,
) -> tuple[str, ...]:
    if not isinstance(efficiency, dict):
        return ("selected_runtime_runtime_proof_missing_efficiency_followup",)
    payload = cast("JsonObject", efficiency)
    expected = {
        "status": "pass",
        "material_speedup_over_baseline": True,
    }
    errors = [
        f"selected_runtime_runtime_proof_efficiency_{key}_mismatch"
        for key, expected_value in expected.items()
        if payload.get(key) != expected_value
    ]
    # Spec 0011 S17b: the efficiency block's identity must match the plan's own
    # recomposed row_id and its runtime_policy_id, not the eager v5 literals.
    if expected_row_id is None or payload.get("selected_row_id") != expected_row_id:
        errors.append(
            "selected_runtime_runtime_proof_efficiency_selected_row_id_mismatch",
        )
    if expected_policy_id is None or (
        payload.get("selected_runtime_policy_id") != expected_policy_id
    ):
        errors.append(
            "selected_runtime_runtime_proof_efficiency_"
            "selected_runtime_policy_id_mismatch",
        )
    return tuple(errors)


def _runtime_proof_amp_followup_errors(amp_followup: object) -> tuple[str, ...]:
    if not isinstance(amp_followup, dict):
        return ("selected_runtime_runtime_proof_missing_amp_followup",)
    payload = cast("JsonObject", amp_followup)
    errors: list[str] = []
    if payload.get("status") != "pass":
        errors.append("selected_runtime_runtime_proof_amp_followup_status_mismatch")
    if payload.get("violation_row_ids") != []:
        errors.append("selected_runtime_runtime_proof_amp_followup_violations")
    return tuple(errors)


def _rank_assignments_are_dual_t4(value: object) -> bool:
    if not isinstance(value, list):
        return False
    items = cast("list[object]", value)
    if len(items) != EXPECTED_DUAL_T4_RANK_COUNT:
        return False
    assignments = [
        cast("dict[str, object]", item) for item in items if isinstance(item, dict)
    ]
    if len(assignments) != EXPECTED_DUAL_T4_RANK_COUNT:
        return False
    observed: set[tuple[object, object, object, object, object]] = {
        (
            assignment.get("rank"),
            assignment.get("local_rank"),
            assignment.get("device"),
            assignment.get("current_device"),
            assignment.get("world_size"),
        )
        for assignment in assignments
    }
    return observed == {
        (0, 0, 0, 0, 2),
        (1, 1, 1, 1, 2),
    }


def _object_list_len(value: object) -> int:
    if not isinstance(value, list):
        return -1
    items = cast("list[object]", value)
    return sum(1 for item in items if isinstance(item, dict))


def _runtime_proof_launch_command_errors(payload: JsonObject) -> tuple[str, ...]:
    missing_command = False
    invalid_command = False
    for key in ("dual_t4_train_step_gate", "runtime_environment"):
        value = payload.get(key)
        if not isinstance(value, dict):
            missing_command = True
            continue
        command = value.get("child_process_launch_command")
        if not isinstance(command, str) or not command:
            missing_command = True
        elif not _is_standalone_torchrun_nproc2(command):
            invalid_command = True
    errors: list[str] = []
    if missing_command:
        errors.append("selected_runtime_runtime_proof_missing_torchrun_command")
    if invalid_command:
        errors.append("selected_runtime_runtime_proof_not_standalone_nproc2")
    return tuple(errors)


def _is_standalone_torchrun_nproc2(command: str) -> bool:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return False
    if not tokens or Path(tokens[0]).name != "torchrun":
        return False
    return "--standalone" in tokens and _token_option_equals(
        tokens,
        option="--nproc_per_node",
        expected="2",
    )


def _token_option_equals(
    tokens: list[str],
    *,
    option: str,
    expected: str,
) -> bool:
    prefix = f"{option}="
    values = [
        token.removeprefix(prefix) for token in tokens if token.startswith(prefix)
    ]
    for index, token in enumerate(tokens[:-1]):
        if token == option:
            values.append(tokens[index + 1])
    return values == [expected]


def _linked_artifact_path(
    *,
    selected_runtime_path: Path,
    artifact_path: str,
) -> Path:
    configured = Path(artifact_path)
    if configured.is_absolute():
        return configured
    candidates = (
        selected_runtime_path.parent.parent / configured,
        selected_runtime_path.parent / configured,
        selected_runtime_path.parent / configured.name,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _application_mismatches(
    *,
    plan: SelectedRuntimePlan,
    observed: SelectedRuntimeApplicationObservation,
) -> tuple[str, ...]:
    checks: tuple[tuple[str, object, object], ...] = (
        ("selected_row_id", observed.selected_row_id, plan.selected_row_id),
        ("runtime_policy_id", observed.runtime_policy_id, plan.runtime_policy_id),
        ("accelerator_mode", observed.accelerator_mode, plan.accelerator_mode),
        ("machine_shape", observed.machine_shape, plan.machine_shape),
        ("world_size", observed.world_size, plan.world_size),
        ("nproc_per_node", observed.nproc_per_node, plan.nproc_per_node),
        ("torchrun_standalone", observed.torchrun_standalone, plan.torchrun_standalone),
        ("per_device_batch_size", observed.batch_size, plan.per_device_batch_size),
        ("global_batch_size", observed.global_batch_size, plan.global_batch_size),
        (
            "optimizer_updates_per_epoch",
            observed.optimizer_updates_per_epoch,
            plan.optimizer_updates_per_epoch,
        ),
        ("amp_enabled", observed.amp_enabled, plan.amp_enabled),
        ("grad_scaler_enabled", observed.grad_scaler_enabled, plan.grad_scaler_enabled),
        ("fp32_loss", observed.fp32_loss, plan.fp32_loss),
        ("autocast_dtype", observed.autocast_dtype, plan.autocast_dtype),
        (
            "torch_compile_enabled",
            observed.torch_compile_enabled,
            plan.torch_compile_enabled,
        ),
        ("compile_scope", observed.compile_scope, plan.compile_scope),
        (
            "dataloader_num_workers",
            observed.dataloader_num_workers,
            plan.dataloader_num_workers,
        ),
        (
            "dataloader_prefetch_factor",
            observed.dataloader_prefetch_factor,
            plan.dataloader_prefetch_factor,
        ),
        (
            "dataloader_pin_memory",
            observed.dataloader_pin_memory,
            plan.dataloader_pin_memory,
        ),
        (
            "dataloader_persistent_workers",
            observed.dataloader_persistent_workers,
            plan.dataloader_persistent_workers,
        ),
        (
            "dataloader_non_blocking_h2d",
            observed.dataloader_non_blocking_h2d,
            plan.dataloader_non_blocking_h2d,
        ),
        (
            "corruption_strategy",
            observed.corruption_strategy,
            expected_corruption_strategy(plan),
        ),
        ("memory_format", observed.memory_format, plan.memory_format),
        ("ddp_static_graph", observed.ddp_static_graph, plan.ddp_static_graph),
        (
            "ddp_gradient_as_bucket_view",
            observed.ddp_gradient_as_bucket_view,
            plan.ddp_gradient_as_bucket_view,
        ),
        (
            "zero_grad_set_to_none",
            observed.zero_grad_set_to_none,
            plan.zero_grad_set_to_none,
        ),
        (
            "gradient_clip_foreach",
            observed.gradient_clip_foreach,
            plan.gradient_clip_foreach,
        ),
        # Spec 0011 S17c -- the compiled fast-path recipe knobs. All are exact echoes of
        # the plan except ``ddp_broadcast_buffers``, handled separately below because
        # the runner may structurally force it on.
        ("compile_backend", observed.compile_backend, plan.compile_backend),
        ("compile_dynamic", observed.compile_dynamic, plan.compile_dynamic),
        ("optimize_ddp", observed.optimize_ddp, plan.optimize_ddp),
        ("compiled_autograd", observed.compiled_autograd, plan.compiled_autograd),
        (
            "reorder_compute_comm_overlap",
            observed.reorder_compute_comm_overlap,
            plan.reorder_compute_comm_overlap,
        ),
        (
            "ddp_find_unused_parameters",
            observed.ddp_find_unused_parameters,
            plan.ddp_find_unused_parameters,
        ),
        ("ddp_bucket_cap_mb", observed.ddp_bucket_cap_mb, plan.ddp_bucket_cap_mb),
        ("fused_optimizer", observed.fused_optimizer, plan.fused_optimizer),
        ("compile_mode", observed.compile_mode, plan.compile_mode),
        ("cudagraphs", observed.cudagraphs, plan.cudagraphs),
        (
            "inductor_options_json",
            observed.inductor_options_json,
            plan.inductor_options_json,
        ),
        (
            "autocast_cache_enabled",
            observed.autocast_cache_enabled,
            plan.autocast_cache_enabled,
        ),
        (
            "ddp_forward_sync_buffers",
            observed.ddp_forward_sync_buffers,
            plan.ddp_forward_sync_buffers,
        ),
        (
            "communication_hook",
            observed.communication_hook,
            plan.communication_hook,
        ),
        (
            "nccl_environment_json",
            observed.nccl_environment_json,
            plan.nccl_environment_json,
        ),
        ("tf32_enabled", observed.tf32_enabled, plan.tf32_enabled),
        ("matmul_precision", observed.matmul_precision, plan.matmul_precision),
        # ``ddp_broadcast_buffers`` tolerates the structural UPWARD override: the runner
        # forces broadcasting on when a model carries rank-divergent running-stat
        # buffers (``model_requires_buffer_broadcast``), so observed True with plan
        # False is legitimate. Only the reverse -- the plan requires broadcasting but
        # the run dropped it -- is a real application failure.
        *(
            (
                (
                    "ddp_broadcast_buffers",
                    observed.ddp_broadcast_buffers,
                    plan.ddp_broadcast_buffers,
                ),
            )
            if plan.ddp_broadcast_buffers and not observed.ddp_broadcast_buffers
            else ()
        ),
        (
            "local_ddp_status",
            observed.local_ddp_status,
            EXPECTED_DDP_APPLICATION_STATUS,
        ),
        (
            "local_amp_status",
            observed.local_amp_status,
            expected_local_amp_status(plan),
        ),
        *(
            (
                (
                    "runner_amp_grad_scaler_init_scale",
                    observed.runner_amp_grad_scaler_init_scale,
                    EXPECTED_RUNNER_AMP_GRAD_SCALER_INIT_SCALE,
                ),
            )
            if observed.runner_amp_grad_scaler_init_scale is not None
            else ()
        ),
    )
    return tuple(
        f"{name}: expected {expected!r}, observed {actual!r}"
        for name, actual, expected in checks
        if actual != expected
    )


def _load_json(path: Path) -> JsonObject:
    payload = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(payload, dict):
        message = f"Expected JSON object in {path}"
        raise TypeError(message)
    return cast("JsonObject", payload)


def _object(payload: JsonObject, key: str) -> JsonObject:
    value = payload.get(key)
    if not isinstance(value, dict):
        message = f"Expected selected-runtime object field {key}"
        raise TypeError(message)
    return cast("JsonObject", value)


def _str(payload: JsonObject, key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        message = f"Expected selected-runtime string field {key}"
        raise TypeError(message)
    return value


def _int(payload: JsonObject, key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        message = f"Expected selected-runtime integer field {key}"
        raise TypeError(message)
    return value


def _optional_int(payload: JsonObject, key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        message = f"Expected selected-runtime optional integer field {key}"
        raise TypeError(message)
    return value


def _bool(payload: JsonObject, key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        message = f"Expected selected-runtime boolean field {key}"
        raise TypeError(message)
    return value


def _str_or(payload: JsonObject, key: str, default: str) -> str:
    """Return the optional string ``key``, or ``default`` when absent.

    Spec 0011 S11 recipe knobs are optional so pre-S11 plans that omit them parse
    to the eager default. A present-but-wrong-typed value still fails closed via
    :func:`_str`.

    Returns:
        The parsed string value, or ``default`` when the key is absent.

    """
    if key not in payload:
        return default
    return _str(payload, key)


def _bool_or(payload: JsonObject, key: str, *, default: bool) -> bool:
    """Return the optional boolean ``key``, or ``default`` when absent.

    Returns:
        The parsed boolean value, or ``default`` when the key is absent.

    """
    if key not in payload:
        return default
    return _bool(payload, key)


def _optional_int_field(payload: JsonObject, key: str) -> int | None:
    """Return the optional int-or-null ``key``, or ``None`` when absent.

    Returns:
        The parsed integer, or ``None`` when the key is absent or explicitly null.

    """
    if key not in payload:
        return None
    return _optional_int(payload, key)


def _optional_bool_field(payload: JsonObject, key: str) -> bool | None:
    """Return the optional bool-or-null ``key``, or ``None`` when absent.

    Returns:
        The parsed boolean, or ``None`` when absent or explicitly null.

    """
    if key not in payload or payload.get(key) is None:
        return None
    return _bool(payload, key)


def _json_object_string_is_valid(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        decoded = cast("object", json.loads(value))
    except json.JSONDecodeError:
        return False
    return isinstance(decoded, dict) and all(
        isinstance(key, str) for key in cast("dict[object, object]", decoded)
    )


def _json_string_map_is_valid(value: object) -> bool:
    if not _json_object_string_is_valid(value):
        return False
    decoded = cast("dict[object, object]", json.loads(cast("str", value)))
    return all(isinstance(item, str) for item in decoded.values())


def _canonical_json_object_string(value: str) -> str:
    decoded = cast("dict[str, object]", json.loads(value))
    return json.dumps(decoded, sort_keys=True, separators=(",", ":"))


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    items = cast("list[object]", value)
    return [item for item in items if isinstance(item, str)]


def _string_value(value: object) -> str:
    return value if isinstance(value, str) else ""


def _str_or_none(value: object) -> str | None:
    # Return the string as-is, else None so a cross-check against it fails closed.
    return value if isinstance(value, str) else None


def _int_as_snapshot_str(value: object) -> str | None:
    """Return the CSV-cell string an integer plan field serializes to, else None.

    The selected-row snapshot stores every cell as a string, so a plan integer field
    cross-checks against ``str(value)``. Booleans are rejected (not batch ints).

    Returns:
        ``str(value)`` for a plain integer, else None so the cross-check fails closed.

    """
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return str(value)


def _bool_as_snapshot_str(value: object) -> str | None:
    """Return the CSV-cell string a boolean plan field serializes to, else None.

    Returns:
        ``"true"``/``"false"`` for a boolean, else None so the cross-check fails closed.

    """
    if not isinstance(value, bool):
        return None
    return "true" if value else "false"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "EXPECTED_DATASET_SLUG",
    "EXPECTED_MACHINE_SHAPE",
    "EXPECTED_RUNTIME_POLICY_ID",
    "EXPECTED_SELECTED_ROW_ID",
    "SelectedRuntimeApplicationObservation",
    "SelectedRuntimePlan",
    "build_plan_applied_proof",
    "composed_selected_runtime_identity",
    "fail_closed_plan_applied_proof",
    "parse_selected_runtime_plan",
    "selected_runtime_identity_payload",
    "selected_runtime_plan_errors",
]
