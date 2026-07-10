# Copyright 2026 HiperMaximus
"""Selected-runtime v5 plan parsing and local application proof helpers."""

from __future__ import annotations

import hashlib
import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from eqvae.benchmarking.schedule import training_steps_per_epoch
from eqvae.data.roots import REAL_TRAIN_PATCH_COUNT

if TYPE_CHECKING:
    from eqvae.config import JsonObject, JsonValue

EXPECTED_DATASET_SLUG = "maximusshtefan/patches-pre-shuffled-ubc-ocean"
EXPECTED_MACHINE_SHAPE = "NvidiaTeslaT4"
EXPECTED_SELECTED_ROW_ID = (
    "dual_t4_ddp__bs12__amp_conservative__compile_none__indexed_masked__"
    "policy_amp_fp16_conservative"
)
EXPECTED_RUNTIME_POLICY_ID = "amp_fp16_conservative"
EXPECTED_DUAL_T4_RANK_COUNT = 2
EXPECTED_RUNTIME_PROOF_WRITE_POLICY = (
    "write_selected_runtime_only_after_dual_t4_and_all_linked_proofs_pass"
)
EXPECTED_DDP_APPLICATION_STATUS = "executed_dual_t4_ddp"
EXPECTED_AMP_APPLICATION_STATUS = "executed_amp_fp16_conservative"
EXPECTED_RUNNER_AMP_GRAD_SCALER_INIT_SCALE = 16384.0


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
            "selected_runtime_artifact_sha256": self.artifact_sha256,
            "local_ddp_status": EXPECTED_DDP_APPLICATION_STATUS,
            "local_amp_status": EXPECTED_AMP_APPLICATION_STATUS,
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
    local_ddp_status: str
    local_amp_status: str
    runner_amp_grad_scaler_init_scale: float | None = None

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
            "local_ddp_status": self.local_ddp_status,
            "local_amp_status": self.local_amp_status,
        }
        if self.runner_amp_grad_scaler_init_scale is not None:
            payload["runner_amp_extension"] = {
                "grad_scaler_init_scale": self.runner_amp_grad_scaler_init_scale,
            }
        return payload


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
    return (
        *_top_level_errors(payload),
        *_launch_errors(payload),
        *_snapshot_errors(payload.get("selected_row_snapshot")),
        *_safety_errors(payload.get("safety")),
        *_runtime_policy_errors(payload.get("runtime_policy")),
        *_ddp_optimizer_safety_errors(payload),
        *_torch_compile_errors(payload.get("torch_compile")),
        *_runtime_proof_errors(
            payload=payload,
            selected_runtime_path=selected_runtime_path,
        ),
    )


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
    )


def _top_level_errors(payload: JsonObject) -> tuple[str, ...]:
    expected = {
        "status": "pass",
        "benchmark_kind": "kaggle_runtime_selection",
        "benchmark_source": "kaggle_runtime_benchmark",
        "selected_row_id": EXPECTED_SELECTED_ROW_ID,
        "runtime_policy_id": EXPECTED_RUNTIME_POLICY_ID,
    }
    error_names = {
        "status": "selected_runtime_status_not_pass",
        "benchmark_kind": "selected_runtime_wrong_benchmark_kind",
        "benchmark_source": "selected_runtime_wrong_benchmark_source",
        "selected_row_id": "selected_runtime_row_not_v5_fallback",
        "runtime_policy_id": "selected_runtime_policy_not_v5_fallback",
    }
    errors = [
        error_names[key]
        for key, expected_value in expected.items()
        if payload.get(key) != expected_value
    ]
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
    elif corruption.get("strategy") != "indexed_masked":
        errors.append("selected_runtime_top_level_wrong_corruption_strategy")
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


def _mixed_precision_errors(payload: JsonObject) -> tuple[str, ...]:
    expected = {
        "enabled": True,
        "policy": "amp_conservative",
        "autocast_dtype": "float16",
        "fp32_loss": True,
        "grad_scaler_enabled": True,
    }
    error_names = {
        "enabled": "selected_runtime_mixed_precision_not_enabled",
        "policy": "selected_runtime_mixed_precision_wrong_policy",
        "autocast_dtype": "selected_runtime_mixed_precision_wrong_dtype",
        "fp32_loss": "selected_runtime_mixed_precision_missing_fp32_loss",
        "grad_scaler_enabled": "selected_runtime_mixed_precision_missing_scaler",
    }
    return tuple(
        error_names[key]
        for key, expected_value in expected.items()
        if payload.get(key) != expected_value
    )


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


def _snapshot_errors(snapshot: object) -> tuple[str, ...]:
    if not isinstance(snapshot, dict):
        return ("selected_runtime_missing_snapshot",)
    snapshot_payload = cast("dict[str, object]", snapshot)
    expected = {
        "row_id": EXPECTED_SELECTED_ROW_ID,
        "runtime_policy_id": EXPECTED_RUNTIME_POLICY_ID,
        "status": "pass",
        "accelerator_mode": "dual_t4_ddp",
        "machine_shape": EXPECTED_MACHINE_SHAPE,
        "precision_policy": "amp_conservative",
        "corruption_strategy": "indexed_masked",
        "nproc_per_node": "2",
        "per_device_batch_size": "12",
        "global_batch_size": "24",
        "grad_scaler_enabled": "true",
        "autocast_dtype": "float16",
    }
    error_names = {
        "row_id": "selected_runtime_snapshot_row_mismatch",
        "runtime_policy_id": "selected_runtime_snapshot_policy_mismatch",
        "status": "selected_runtime_snapshot_status_not_pass",
        "accelerator_mode": "selected_runtime_snapshot_not_dual_t4_ddp",
        "machine_shape": "selected_runtime_snapshot_wrong_machine_shape",
        "precision_policy": "selected_runtime_snapshot_wrong_precision_policy",
        "corruption_strategy": "selected_runtime_snapshot_wrong_corruption_strategy",
        "nproc_per_node": "selected_runtime_snapshot_wrong_nproc_per_node",
        "per_device_batch_size": "selected_runtime_snapshot_wrong_per_device_batch",
        "global_batch_size": "selected_runtime_snapshot_wrong_global_batch",
        "grad_scaler_enabled": "selected_runtime_snapshot_missing_scaler",
        "autocast_dtype": "selected_runtime_snapshot_wrong_autocast_dtype",
    }
    errors = [
        error_names[key]
        for key, expected_value in expected.items()
        if snapshot_payload.get(key) != expected_value
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
    if not isinstance(policy, dict):
        return ("selected_runtime_missing_runtime_policy",)
    payload = cast("JsonObject", policy)
    expected = {
        "memory_format": "contiguous",
        "ddp_static_graph": False,
        "ddp_gradient_as_bucket_view": False,
        "zero_grad_set_to_none": True,
    }
    return tuple(
        f"selected_runtime_runtime_policy_{key}_mismatch"
        for key, expected_value in expected.items()
        if payload.get(key) != expected_value
    )


def _torch_compile_errors(torch_compile: object) -> tuple[str, ...]:
    if not isinstance(torch_compile, dict):
        return ("selected_runtime_missing_torch_compile",)
    payload = cast("JsonObject", torch_compile)
    expected = {
        "enabled": False,
        "scope": "none",
        "dynamic": False,
    }
    return tuple(
        f"selected_runtime_torch_compile_{key}_mismatch"
        for key, expected_value in expected.items()
        if payload.get(key) != expected_value
    )


_DDP_OPTIMIZER_RECIPE_VALUE = "ddp_optimizer"
_RECIPE_CARRIER_BLOCK_KEYS = ("runtime_policy", "torch_compile")


def _ddp_optimizer_safety_errors(payload: JsonObject) -> tuple[str, ...]:
    """Reject a DDPOptimizer plan whose flags break cross-rank gradient sync.

    ``optimize_ddp="ddp_optimizer"`` (DDPOptimizer) splits the backward at DDP bucket
    boundaries. Three flag pairings break it, per memory
    ``eqvae-compiled-ddp-optimize-ddp`` and the measured winner ``_DDP_OPTIMIZER_SPEC``
    (``benchmarking/compiled_fastpath_probe.py``), which pairs DDPOptimizer with
    ``compiled_autograd=False``:

    - ``compiled_autograd=True`` traces the backward, so DDP's C++ reducer hooks never
      fire and the grad all_reduce is **silently** dropped -- each rank then trains an
      independent replica (the empirically caught failure);
    - ``static_graph=True`` is a **loud** dynamo #93672 "training graph has changed"
      conflict;
    - ``find_unused_parameters=True`` is incompatible with the bucket split.

    No plan sets ``optimize_ddp`` today, so this is a no-op on the v5 fallback plan
    and a fail-closed guard for the Phase 2 compiled plans that carry the recipe knobs.

    Returns:
        Stable DDPOptimizer safety error identifiers.

    """
    if _recipe_field(payload, "optimize_ddp") != _DDP_OPTIMIZER_RECIPE_VALUE:
        return ()
    errors: list[str] = []
    if _recipe_flag_enabled(payload, "compiled_autograd"):
        errors.append("selected_runtime_ddp_optimizer_compiled_autograd_conflict")
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

    proof_errors = _runtime_proof_payload_errors(proof_payload)
    command_errors = _runtime_proof_launch_command_errors(proof_payload)
    return (*proof_errors, *command_errors)


def _runtime_proof_payload_errors(payload: JsonObject) -> tuple[str, ...]:
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
        ),
    )
    errors.extend(_runtime_proof_efficiency_errors(payload.get("efficiency_followup")))
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


def _runtime_proof_write_decision_errors(decision: object) -> tuple[str, ...]:
    if not isinstance(decision, dict):
        return ("selected_runtime_runtime_proof_missing_write_decision",)
    payload = cast("JsonObject", decision)
    expected = {
        "allowed": True,
        "policy": EXPECTED_RUNTIME_PROOF_WRITE_POLICY,
        "selected_row_id": EXPECTED_SELECTED_ROW_ID,
        "stain_corruptor_qa_status": "pass",
    }
    errors = [
        f"selected_runtime_runtime_proof_write_decision_{key}_mismatch"
        for key, expected_value in expected.items()
        if payload.get(key) != expected_value
    ]
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


def _runtime_proof_efficiency_errors(efficiency: object) -> tuple[str, ...]:
    if not isinstance(efficiency, dict):
        return ("selected_runtime_runtime_proof_missing_efficiency_followup",)
    payload = cast("JsonObject", efficiency)
    expected = {
        "status": "pass",
        "material_speedup_over_baseline": True,
        "selected_row_id": EXPECTED_SELECTED_ROW_ID,
        "selected_runtime_policy_id": EXPECTED_RUNTIME_POLICY_ID,
    }
    return tuple(
        f"selected_runtime_runtime_proof_efficiency_{key}_mismatch"
        for key, expected_value in expected.items()
        if payload.get(key) != expected_value
    )


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
        ("corruption_strategy", observed.corruption_strategy, plan.corruption_strategy),
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
            "local_ddp_status",
            observed.local_ddp_status,
            EXPECTED_DDP_APPLICATION_STATUS,
        ),
        (
            "local_amp_status",
            observed.local_amp_status,
            EXPECTED_AMP_APPLICATION_STATUS,
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


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    items = cast("list[object]", value)
    return [item for item in items if isinstance(item, str)]


def _string_value(value: object) -> str:
    return value if isinstance(value, str) else ""


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
    "fail_closed_plan_applied_proof",
    "parse_selected_runtime_plan",
    "selected_runtime_identity_payload",
    "selected_runtime_plan_errors",
]
