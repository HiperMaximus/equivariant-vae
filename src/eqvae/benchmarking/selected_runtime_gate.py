# Copyright 2026 HiperMaximus
"""Fail-closed selected-runtime debug/resume/tiny gate artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from collections.abc import Sequence

    from eqvae.benchmarking.io import CsvRow

from eqvae.artifacts.fixed25_equivariance import (
    DEGREES_PER_K,
    EQUIVARIANCE_25_COLUMNS,
    FIRST3_PNG,
    FIXED25_DIRNAME,
    GRID_PNG,
    LATENT_MU_PT,
    MANIFEST_JSON,
    MEASURED_K_VALUES,
    ORIGINALS_PT,
    PCA_PNG,
    RECONSTRUCTION_PROGRESS_PT,
    REQUIRED_EQUIVARIANCE_METRICS,
    error_maps_pt_name,
    rotated_pt_name,
)
from eqvae.benchmarking.fixed32_selector_readiness import (
    EXPECTED_TINY_SELECTOR_COUNT,
    LOCAL_SELECTOR_MODE,
    OK_STATUS,
    REMOTE_GENERATE_MODE,
    Fixed32RemoteGenerateReadinessRequest,
    canonical_real_ubc_requirements,
    fixed32_selector_status,
    readiness_blockers,
    write_fixed32_remote_generate_readiness,
)
from eqvae.benchmarking.io import JsonObject, write_csv, write_json
from eqvae.benchmarking.runtime_schema import GATE_HEALTH_COLUMNS
from eqvae.benchmarking.schedule import training_steps_per_epoch
from eqvae.config import ResolvedConfig, resolve_json_config
from eqvae.data.fixed_selectors import (
    FIXED_32_TRAIN_OVERFIT_KIND,
    FIXED_32_TRAIN_OVERFIT_SEED,
    FIXED_SELECTOR_READY_STATUS,
    FixedSelectorDocument,
    load_fixed_selector_document,
)
from eqvae.data.roots import REAL_TRAIN_PATCH_COUNT
from eqvae.training.optim import BatchLrScaling, scaled_learning_rate
from eqvae.training.selected_runtime import (
    EXPECTED_DATASET_SLUG,
    EXPECTED_RUNNER_AMP_GRAD_SCALER_INIT_SCALE,
    EXPECTED_RUNTIME_POLICY_ID,
    EXPECTED_SELECTED_ROW_ID,
    fail_closed_plan_applied_proof,
    selected_runtime_identity_payload,
    selected_runtime_plan_errors,
)

GATE_SCHEMA_VERSION = "spec0001.selected_runtime_debug_gate.v1"
GATE_KIND = "kaggle_selected_runtime_debug_resume_tiny_gate"
GATE_SOURCE = "kaggle_selected_runtime_debug_kernel"
type SelectorGenerationMode = Literal["local_selector", "remote_generate"]
FAIL_STATUS = "fail"
REAL_UBC_SELECTED_RUNTIME_TRAIN_RUNNER_IMPLEMENTED = True
SELECTED_RUNTIME_DEBUG_WRAPPER_WIRED_TO_REAL_RUNNER = True
SELECTED_RUNTIME_PLAN_APPLIED_TO_TRAINING = False
REMOTE_DEBUG_PENDING_BLOCKER = "selected_runtime_debug_remote_proof_pending"
SHA256_HEX_LENGTH = 64
TRAIN_METRIC_COLUMNS = (
    "optimizer_step",
    "loss",
    "recon_loss",
    "l1_loss",
    "ssim_loss",
    "ssim_metric",
    "kl_loss",
    "beta",
    "grad_norm",
    "param_update_norm",
    "nonfinite_count",
    "checkpoint_path",
)
REMOTE_DEBUG_REQUIRED_BENCHMARK_ARTIFACTS = frozenset(
    {
        "artifact_manifest.json",
        "checkpoint_resume_proof.json",
        "fixed32_selector_readiness.json",
        "fixed_32_train_overfit_patches.json",
        "gate_health_summary.json",
        "local_selected_runtime_readiness.json",
        "selected_runtime_plan_applied.json",
        "selected_runtime_debug_summary.json",
        "selected_runtime_gate_summary.json",
        "tiny_overfit_summary.json",
        "training_summary.json",
    },
)
REMOTE_DEBUG_REQUIRED_METRIC_ARTIFACTS = frozenset(
    {
        "gate_health.csv",
        "train_steps.csv",
    },
)
REMOTE_FULL_REQUIRED_BENCHMARK_ARTIFACTS = frozenset(
    {
        "artifact_manifest.json",
        "checkpoint_resume_proof.json",
        "gate_health_summary.json",
        "local_selected_runtime_readiness.json",
        "selected_runtime_full_summary.json",
        "selected_runtime_plan_applied.json",
        "training_summary.json",
    },
)
REMOTE_FULL_REQUIRED_METRIC_ARTIFACTS = frozenset(
    {
        "gate_health.csv",
        "train_steps.csv",
        "validation_metrics.csv",
    },
)
# equivariance_25.csv is required by the deep fixed-25 checks (Spec 0010) rather
# than the early required-artifact gate, so a partial flush that has not yet
# written it still reaches the deep schedule checks. It is allowed here so a real
# full-run output is not flagged as an unexpected extra metric file.
REMOTE_FULL_OPTIONAL_METRIC_ARTIFACTS = frozenset({"equivariance_25.csv"})
_FIXED25_EXPECTED_SAMPLE_COUNT = "25"
_FIXED25_EXPECTED_K_VALUES = [0, 1, 2, 3]
# Policy anchors the gate keeps pinned (a run cannot self-declare a tiny schedule to
# slip past coverage); the schedule sizes (updates/target/half) are re-derived from
# floor(P / global_batch) in _remote_full_expected_schedule, not frozen here, and
# world_size is read from the plan -- all so a different global batch re-runs this gate
# unchanged (Spec 0011).
REMOTE_FULL_EPOCHS = 10
REMOTE_FULL_VALIDATION_BATCHES_PER_VIEW = 20
REMOTE_FULL_VALIDATION_VIEWS = ("clean", "deterministic_denoising")
# best_model.pt must be selected on the denoising view across ranks with a
# sample-weighted reduction (FU-008), and reparameterization eps must be per-rank
# distinct (FU-007); the full summary records both so the gate can assert them.
REMOTE_FULL_CHECKPOINT_SELECTION_VIEW = "deterministic_denoising"
REMOTE_FULL_CHECKPOINT_SELECTION_REDUCTION = "cross_rank_sample_weighted_l1"
REMOTE_FULL_INTERVAL_CHECKPOINT_KEEP_COUNT = 4
REMOTE_DEBUG_FINAL_STEP = 8
REMOTE_DEBUG_RESUME_STEP = 4
REMOTE_DEBUG_REQUIRED_SUCCESSFUL_STEPS = tuple(
    range(REMOTE_DEBUG_RESUME_STEP + 1, REMOTE_DEBUG_FINAL_STEP + 1),
)
REMOTE_TINY_MAX_STEP = 128
REMOTE_TINY_MIN_IMPROVEMENT_FRACTION = 0.01
REMOTE_TINY_FULL_BATCH_SAMPLER_POLICY = "fixed32_tiny_full_batch_repeated"
REMOTE_AMP_GRAD_SCALER_INIT_SCALE = EXPECTED_RUNNER_AMP_GRAD_SCALER_INIT_SCALE
RUNNER_OK_STATUS = "local_pass"
REMOTE_DEBUG_REQUIRED_TRAIN_STEP_COLUMNS = (
    "event_id",
    "rank",
    "optimizer_step_index",
    "optimizer_step",
    "successful_optimizer_update_count",
    "split",
    "loss",
    "recon_loss",
    "l1_loss",
    "ssim_loss",
    "ssim_metric",
    "kl_loss",
    "beta",
    "grad_norm",
    "param_update_norm",
    "nonfinite_count",
    "batch_size",
    "precision_policy",
    "amp_enabled",
    "autocast_dtype",
    "grad_scaler_enabled",
    "fp32_loss",
    "torch_compile_enabled",
    "compile_scope",
    "corruption_strategy",
    "amp_step_skipped",
    "checkpoint_path",
)
REMOTE_GATE_HEALTH_FINITE_COLUMNS = (
    "a_min",
    "a_max",
    "a_mean",
    "a_std",
    "b_min",
    "b_max",
    "b_mean",
    "b_std",
    "max_abs_a",
    "max_abs_b",
    "gate_mean",
    "gate_std",
    "gate_p01",
    "gate_p50",
    "gate_p99",
    "frac_gate_lt_0_01",
    "frac_gate_gt_0_99",
    "worst_channel_frac_gate_lt_0_01",
    "worst_channel_frac_gate_gt_0_99",
    "a_grad_norm",
    "b_grad_norm",
)
REMOTE_GATE_HEALTH_SATURATION_COLUMNS = (
    "frac_gate_lt_0_01",
    "frac_gate_gt_0_99",
    "worst_channel_frac_gate_lt_0_01",
    "worst_channel_frac_gate_gt_0_99",
)
REMOTE_GATE_HEALTH_MAX_SATURATION_FRACTION = 0.99


@dataclass(frozen=True)
class SelectedRuntimeGateRequest:
    """Inputs for the selected-runtime debug/resume/tiny gate."""

    debug_config_path: Path
    tiny_config_path: Path
    selected_runtime_path: Path
    output_dir: Path
    run_name: str
    data_root: str | None = None
    fixed_train_patches: Path | None = None
    selector_generation_mode: SelectorGenerationMode = LOCAL_SELECTOR_MODE


@dataclass(frozen=True)
class SelectedRuntimeGateResult:
    """Artifact paths from the fail-closed gate writer."""

    output_dir: Path
    gate_summary: Path
    training_summary: Path
    selected_runtime_debug_summary: Path
    checkpoint_resume_proof: Path
    selected_runtime_plan_applied: Path
    local_readiness: Path
    tiny_overfit_summary: Path
    artifact_manifest: Path
    gate_health_summary: Path


def write_selected_runtime_gate(
    request: SelectedRuntimeGateRequest,
) -> SelectedRuntimeGateResult:
    """Write selected-runtime gate artifacts without launching long training.

    The current implementation intentionally fails closed for local artifact
    writing. The real `ubc-pre-shuffled` DDP/AMP runner is wired through the
    selected-runtime debug wrapper, but downloaded remote artifacts are still
    required before any pass claim.

    Returns:
        Paths to the gate artifacts.

    """
    output_dir = request.output_dir.resolve()
    benchmark_dir = output_dir / "benchmark"
    metrics_dir = output_dir / "metrics"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    debug_resolved = resolve_json_config(request.debug_config_path)
    tiny_resolved = resolve_json_config(request.tiny_config_path)
    runtime_payload = _load_json(request.selected_runtime_path)
    runtime_errors = _selected_runtime_errors(
        runtime_payload,
        selected_runtime_path=request.selected_runtime_path,
    )
    runtime_identity = _runtime_identity(
        path=request.selected_runtime_path,
        payload=runtime_payload,
        errors=runtime_errors,
    )
    selector_path = request.fixed_train_patches or _selector_path(
        config_path=request.tiny_config_path,
        resolved=tiny_resolved,
    )
    selector_status = fixed32_selector_status(
        selector_path,
        data_root=request.data_root,
    )
    selector_status["selector_generation_mode"] = request.selector_generation_mode
    blockers = _launch_blockers(
        runtime_errors=runtime_errors,
        selector_status=selector_status,
    )

    paths = _artifact_paths(output_dir)
    write_json(
        paths.training_summary,
        _training_summary(
            request=request,
            resolved=debug_resolved,
            runtime_identity=runtime_identity,
            blockers=blockers,
        ),
    )
    write_json(
        paths.selected_runtime_debug_summary,
        _debug_summary(
            request=request,
            runtime_identity=runtime_identity,
            blockers=blockers,
        ),
    )
    write_json(
        paths.checkpoint_resume_proof,
        _resume_summary(runtime_identity=runtime_identity, blockers=blockers),
    )
    write_json(
        paths.selected_runtime_plan_applied,
        fail_closed_plan_applied_proof(
            path=request.selected_runtime_path,
            payload=runtime_payload,
            errors=runtime_errors,
            failure_kind="real_train_runner_observation_missing",
        ),
    )
    write_json(
        paths.tiny_overfit_summary,
        _tiny_summary(
            runtime_identity=runtime_identity,
            selector_status=selector_status,
            blockers=blockers,
        ),
    )
    write_json(
        paths.gate_health_summary,
        _gate_health_summary(runtime_identity=runtime_identity, blockers=blockers),
    )
    write_csv(paths.train_metrics, TRAIN_METRIC_COLUMNS, ())
    write_csv(paths.gate_health_metrics, GATE_HEALTH_COLUMNS, ())
    write_json(
        paths.gate_summary,
        _gate_summary(
            _GateSummaryContext(
                request=request,
                debug_resolved=debug_resolved,
                tiny_resolved=tiny_resolved,
                runtime_identity=runtime_identity,
                selector_status=selector_status,
                blockers=blockers,
            ),
        ),
    )
    write_json(
        paths.local_readiness,
        _local_readiness_summary(
            selector_generation_mode=request.selector_generation_mode,
            runtime_identity=runtime_identity,
            selector_status=selector_status,
            blockers=blockers,
            artifact_manifest=None,
        ),
    )
    artifact_manifest = _artifact_manifest(paths=paths)
    write_json(
        paths.local_readiness,
        _local_readiness_summary(
            selector_generation_mode=request.selector_generation_mode,
            runtime_identity=runtime_identity,
            selector_status=selector_status,
            blockers=blockers,
            artifact_manifest=artifact_manifest,
        ),
    )
    write_json(paths.artifact_manifest, _artifact_manifest(paths=paths))
    return SelectedRuntimeGateResult(
        output_dir=output_dir,
        gate_summary=paths.gate_summary,
        training_summary=paths.training_summary,
        selected_runtime_debug_summary=paths.selected_runtime_debug_summary,
        checkpoint_resume_proof=paths.checkpoint_resume_proof,
        selected_runtime_plan_applied=paths.selected_runtime_plan_applied,
        local_readiness=paths.local_readiness,
        tiny_overfit_summary=paths.tiny_overfit_summary,
        artifact_manifest=paths.artifact_manifest,
        gate_health_summary=paths.gate_health_summary,
    )


def verify_selected_runtime_debug_push_ready(  # noqa: PLR0913
    *,
    debug_config_path: Path,
    tiny_config_path: Path,
    selected_runtime_path: Path,
    selector_generation_mode: SelectorGenerationMode = LOCAL_SELECTOR_MODE,
    data_root: str | None = None,
    fixed_train_patches: Path | None = None,
) -> tuple[str, ...]:
    """Return blockers that make the remote selected-runtime debug push unsafe.

    Returns:
        Stable blocker names. An empty tuple means the local semantic checks do
        not object to a push; it is not by itself permission to run Kaggle.

    """
    debug_resolved = resolve_json_config(debug_config_path)
    tiny_resolved = resolve_json_config(tiny_config_path)
    runtime_payload = _load_json(selected_runtime_path)
    runtime_errors = _selected_runtime_errors(
        runtime_payload,
        selected_runtime_path=selected_runtime_path,
    )
    selector_path = fixed_train_patches or _selector_path(
        config_path=tiny_config_path,
        resolved=tiny_resolved,
    )
    selector_status = fixed32_selector_status(selector_path, data_root=data_root)
    selector_status["selector_generation_mode"] = selector_generation_mode
    blockers = [
        *_push_readiness_blockers(
            runtime_errors=runtime_errors,
            selector_status=selector_status,
            selector_generation_mode=selector_generation_mode,
        ),
        *_structured_readiness_blockers(
            debug_config_path=debug_config_path,
            tiny_config_path=tiny_config_path,
            selected_runtime_path=selected_runtime_path,
            selector_generation_mode=selector_generation_mode,
            data_root=data_root,
            fixed_train_patches=selector_path,
        ),
        *_readiness_config_blockers(
            resolved=debug_resolved,
            gate_key="selected_runtime_debug",
            selector_generation_mode=selector_generation_mode,
        ),
        *_readiness_config_blockers(
            resolved=tiny_resolved,
            gate_key="selected_runtime_debug_gate",
            selector_generation_mode=selector_generation_mode,
        ),
    ]
    return _dedupe_strings(tuple(blockers))


def verify_selected_runtime_debug_output(  # noqa: PLR0914
    *,
    output_dir: Path,
    selected_runtime_path: Path,
) -> tuple[str, ...]:
    """Return blockers for a downloaded selected-runtime debug/tiny output.

    Returns:
        Stable blocker names. Empty means the artifact contract passed locally.

    """
    benchmark_dir = output_dir / "benchmark"
    metrics_dir = output_dir / "metrics"
    blockers: list[str] = []
    if not benchmark_dir.exists():
        return ("selected_runtime_output_benchmark_dir_missing",)
    observed_benchmark: set[str] = {path.name for path in benchmark_dir.iterdir()}
    observed_metrics: set[str] = (
        {path.name for path in metrics_dir.iterdir()} if metrics_dir.exists() else set()
    )
    missing_benchmark = REMOTE_DEBUG_REQUIRED_BENCHMARK_ARTIFACTS - observed_benchmark
    unexpected_benchmark = (
        observed_benchmark - REMOTE_DEBUG_REQUIRED_BENCHMARK_ARTIFACTS
    )
    missing_metrics = REMOTE_DEBUG_REQUIRED_METRIC_ARTIFACTS - observed_metrics
    unexpected_metrics = observed_metrics - REMOTE_DEBUG_REQUIRED_METRIC_ARTIFACTS
    blockers.extend(
        f"selected_runtime_output_missing_{name}" for name in sorted(missing_benchmark)
    )
    blockers.extend(
        f"selected_runtime_output_unexpected_{name}"
        for name in sorted(unexpected_benchmark)
    )
    blockers.extend(
        f"selected_runtime_output_missing_metric_{name}"
        for name in sorted(missing_metrics)
    )
    blockers.extend(
        f"selected_runtime_output_unexpected_metric_{name}"
        for name in sorted(unexpected_metrics)
    )
    if (benchmark_dir / "selected_runtime.json").exists():
        blockers.append("selected_runtime_output_wrote_selected_runtime")
    if blockers:
        return _dedupe_strings(tuple(blockers))

    runtime_payload = _load_json(selected_runtime_path)
    runtime_sha256 = _sha256_file(selected_runtime_path)
    max_batch_size = _int_value(runtime_payload.get("per_device_batch_size"))
    global_batch_size = _int_value(runtime_payload.get("global_batch_size"))
    training_summary = _load_json(benchmark_dir / "training_summary.json")
    debug_summary = _load_json(benchmark_dir / "selected_runtime_debug_summary.json")
    plan_applied = _load_json(benchmark_dir / "selected_runtime_plan_applied.json")
    resume_proof = _load_json(benchmark_dir / "checkpoint_resume_proof.json")
    gate_health = _load_json(benchmark_dir / "gate_health_summary.json")
    artifact_manifest = _load_json(benchmark_dir / "artifact_manifest.json")
    selector_readiness = _load_json(benchmark_dir / "fixed32_selector_readiness.json")
    tiny_summary = _load_json(benchmark_dir / "tiny_overfit_summary.json")
    gate_summary = _load_json(benchmark_dir / "selected_runtime_gate_summary.json")
    blockers.extend(
        _remote_output_json_blockers(
            runtime_sha256=runtime_sha256,
            training_summary=training_summary,
            debug_summary=debug_summary,
            plan_applied=plan_applied,
            resume_proof=resume_proof,
            gate_health=gate_health,
            artifact_manifest=artifact_manifest,
            selector_readiness=selector_readiness,
            tiny_summary=tiny_summary,
            gate_summary=gate_summary,
            max_batch_size=max_batch_size,
            global_batch_size=global_batch_size,
        ),
    )
    blockers.extend(
        _remote_output_selector_blockers(
            selector_path=benchmark_dir / "fixed_32_train_overfit_patches.json",
            selector_readiness=selector_readiness,
        ),
    )
    blockers.extend(
        _remote_output_manifest_blockers(
            output_dir=output_dir,
            artifact_manifest=artifact_manifest,
        ),
    )
    blockers.extend(
        _remote_output_gate_health_blockers(metrics_dir / "gate_health.csv"),
    )
    blockers.extend(
        _remote_output_train_step_blockers(
            metrics_dir / "train_steps.csv",
            max_batch_size=max_batch_size,
        ),
    )
    blockers.extend(
        _remote_output_tiny_train_step_blockers(
            output_dir / "tiny_overfit_phase" / "metrics" / "train_steps.csv",
            max_batch_size=max_batch_size,
            global_batch_size=global_batch_size,
        ),
    )
    return _dedupe_strings(tuple(blockers))


@dataclass(frozen=True)
class _RemoteFullSchedule:
    """Goal-derived full-run schedule the gate independently anchors (Spec 0011).

    ``updates_per_epoch`` is ``floor(REAL_TRAIN_PATCH_COUNT / global_batch_size)`` --
    the gate re-derives it from the immutable patch count and the plan's global batch
    rather than trusting the summary's number (MF2). ``target`` and ``half`` follow
    from it and the pinned epoch policy anchor. ``valid`` is False when the runtime
    payload has a non-positive global batch, in which case the sentinel sizes never
    match a real summary and the gate fails closed.
    """

    global_batch_size: int
    updates_per_epoch: int
    target_updates: int
    half_epoch_interval: int
    valid: bool


def _remote_full_expected_schedule(global_batch_size: int) -> _RemoteFullSchedule:
    updates_per_epoch = (
        training_steps_per_epoch(
            real_train_patch_count=REAL_TRAIN_PATCH_COUNT,
            global_batch_size=global_batch_size,
        )
        if global_batch_size > 0
        else 0
    )
    half_epoch_interval = updates_per_epoch // 2
    # A real full run needs a positive half-epoch boundary (updates_per_epoch >= 2);
    # otherwise the boundary range() would be degenerate. Fail closed with sentinels
    # the summary can never match instead of trusting an impossible schedule.
    if global_batch_size <= 0 or half_epoch_interval < 1:
        return _RemoteFullSchedule(
            global_batch_size=global_batch_size,
            updates_per_epoch=-1,
            target_updates=-1,
            half_epoch_interval=-1,
            valid=False,
        )
    return _RemoteFullSchedule(
        global_batch_size=global_batch_size,
        updates_per_epoch=updates_per_epoch,
        target_updates=REMOTE_FULL_EPOCHS * updates_per_epoch,
        half_epoch_interval=half_epoch_interval,
        valid=True,
    )


def verify_selected_runtime_full_output(  # noqa: PLR0914
    *,
    output_dir: Path,
    selected_runtime_path: Path,
) -> tuple[str, ...]:
    """Return blockers for a downloaded selected-runtime full-run output.

    Returns:
        Stable blocker names. Empty means the artifact contract passed locally.

    """
    benchmark_dir = output_dir / "benchmark"
    metrics_dir = output_dir / "metrics"
    blockers: list[str] = []
    if not benchmark_dir.exists():
        return ("selected_runtime_full_output_benchmark_dir_missing",)
    observed_benchmark: set[str] = {path.name for path in benchmark_dir.iterdir()}
    observed_metrics: set[str] = (
        {path.name for path in metrics_dir.iterdir()} if metrics_dir.exists() else set()
    )
    missing_benchmark = REMOTE_FULL_REQUIRED_BENCHMARK_ARTIFACTS - observed_benchmark
    missing_metrics = REMOTE_FULL_REQUIRED_METRIC_ARTIFACTS - observed_metrics
    blockers.extend(
        f"selected_runtime_full_output_missing_{name}"
        for name in sorted(missing_benchmark)
    )
    blockers.extend(
        f"selected_runtime_full_output_missing_metric_{name}"
        for name in sorted(missing_metrics)
    )
    unexpected_metrics = (
        observed_metrics
        - REMOTE_FULL_REQUIRED_METRIC_ARTIFACTS
        - REMOTE_FULL_OPTIONAL_METRIC_ARTIFACTS
    )
    blockers.extend(
        f"selected_runtime_full_output_unexpected_metric_{name}"
        for name in sorted(unexpected_metrics)
    )
    if (benchmark_dir / "selected_runtime.json").exists():
        blockers.append("selected_runtime_full_output_wrote_selected_runtime")
    if blockers:
        return _dedupe_strings(tuple(blockers))

    runtime_sha256 = _sha256_file(selected_runtime_path)
    training_summary = _load_json(benchmark_dir / "training_summary.json")
    full_summary = _load_json(benchmark_dir / "selected_runtime_full_summary.json")
    plan_applied = _load_json(benchmark_dir / "selected_runtime_plan_applied.json")
    resume_proof = _load_json(benchmark_dir / "checkpoint_resume_proof.json")
    gate_health = _load_json(benchmark_dir / "gate_health_summary.json")
    artifact_manifest = _load_json(benchmark_dir / "artifact_manifest.json")
    runtime_payload = _load_json(selected_runtime_path)
    max_batch_size = _int_value(runtime_payload.get("per_device_batch_size"))
    world_size = _int_value(runtime_payload.get("world_size"))
    # Independent goal-derived schedule (MF2): re-derive updates/target/half from the
    # immutable patch count and the plan's global batch, so a summary that self-reports
    # a wrong schedule cannot slip past the gate on its own numbers.
    schedule = _remote_full_expected_schedule(
        _int_value(runtime_payload.get("global_batch_size")),
    )
    blockers.extend(
        _remote_full_json_blockers(
            runtime_sha256=runtime_sha256,
            training_summary=training_summary,
            full_summary=full_summary,
            plan_applied=plan_applied,
            resume_proof=resume_proof,
            gate_health=gate_health,
            artifact_manifest=artifact_manifest,
            output_dir=output_dir,
            schedule=schedule,
        ),
    )
    blockers.extend(
        _remote_full_lr_blockers(
            training_summary,
            global_batch_size=schedule.global_batch_size,
        ),
    )
    blockers.extend(
        _remote_full_cross_consistency_blockers(
            training_summary=training_summary,
            full_summary=full_summary,
        ),
    )
    blockers.extend(
        _remote_full_manifest_blockers(
            output_dir=output_dir,
            training_summary=training_summary,
            artifact_manifest=artifact_manifest,
        ),
    )
    blockers.extend(
        _remote_output_gate_health_blockers(metrics_dir / "gate_health.csv"),
    )
    blockers.extend(
        _remote_full_train_step_blockers(
            metrics_dir / "train_steps.csv",
            max_batch_size=max_batch_size,
            world_size=world_size,
            target_updates=schedule.target_updates,
        ),
    )
    blockers.extend(
        _remote_full_validation_blockers(
            metrics_dir / "validation_metrics.csv",
            half_epoch_interval=schedule.half_epoch_interval,
            target_updates=schedule.target_updates,
        ),
    )
    blockers.extend(_remote_full_fixed25_blockers(output_dir=output_dir))
    return _dedupe_strings(tuple(blockers))


def _remote_full_json_blockers(  # noqa: PLR0913
    *,
    runtime_sha256: str,
    training_summary: JsonObject,
    full_summary: JsonObject,
    plan_applied: JsonObject,
    resume_proof: JsonObject,
    gate_health: JsonObject,
    artifact_manifest: JsonObject,
    output_dir: Path,
    schedule: _RemoteFullSchedule,
) -> tuple[str, ...]:
    blockers: list[str] = []
    if not schedule.valid:
        blockers.append("selected_runtime_full_output_runtime_global_batch_invalid")
    runtime_config = training_summary.get("runtime_config")
    runtime_payload = runtime_config if isinstance(runtime_config, dict) else {}
    if runtime_payload.get("sha256") != runtime_sha256:
        blockers.append("selected_runtime_full_output_runtime_sha256_mismatch")
    retained_interval_checkpoint_count = _int_value(
        training_summary.get("retained_interval_checkpoint_count"),
    )
    retained_interval_checkpoint_names = _full_retained_interval_checkpoint_names(
        training_summary,
    )
    expected_interval_checkpoint_names = _full_expected_interval_checkpoint_names(
        schedule,
    )
    checks: tuple[tuple[bool, str], ...] = (
        (
            training_summary.get("status") == RUNNER_OK_STATUS,
            "training_summary_not_pass",
        ),
        (
            training_summary.get("run_mode") == "kaggle_selected_runtime_full_train",
            "wrong_run_mode",
        ),
        (
            training_summary.get("target_optimizer_updates") == schedule.target_updates,
            "target_updates_mismatch",
        ),
        (
            training_summary.get("optimizer_steps_completed")
            == schedule.target_updates,
            "completed_updates_mismatch",
        ),
        (
            training_summary.get("requested_epochs") == REMOTE_FULL_EPOCHS,
            "epochs_mismatch",
        ),
        (
            training_summary.get("optimizer_updates_per_epoch")
            == schedule.updates_per_epoch,
            "updates_per_epoch_mismatch",
        ),
        (
            training_summary.get("half_epoch_interval_steps")
            == schedule.half_epoch_interval,
            "half_epoch_interval_mismatch",
        ),
        (
            training_summary.get("validation_batches_per_view")
            == REMOTE_FULL_VALIDATION_BATCHES_PER_VIEW,
            "validation_batch_count_mismatch",
        ),
        (
            training_summary.get("validation_views")
            == list(REMOTE_FULL_VALIDATION_VIEWS),
            "validation_views_mismatch",
        ),
        (
            training_summary.get("train_reparameterization") == "stochastic_seeded",
            "train_reparameterization_not_stochastic",
        ),
        (training_summary.get("amp_step_skipped_count") == 0, "amp_skips_nonzero"),
        (training_summary.get("nonfinite_count") == 0, "nonfinite_nonzero"),
        (
            training_summary.get("checkpoint_retention")
            == "best_final_latest_four_interval",
            "checkpoint_retention_mismatch",
        ),
        (training_summary.get("resume_supported") is True, "resume_not_supported"),
        (
            retained_interval_checkpoint_count
            == REMOTE_FULL_INTERVAL_CHECKPOINT_KEEP_COUNT,
            "retained_interval_checkpoint_count_mismatch",
        ),
        (
            retained_interval_checkpoint_names == expected_interval_checkpoint_names,
            "retained_interval_checkpoints_mismatch",
        ),
        (
            full_summary.get("selected_runtime_full_run_contract_ready") is True,
            "full_summary_contract_not_ready",
        ),
        (full_summary.get("status") == RUNNER_OK_STATUS, "full_summary_not_pass"),
        (
            full_summary.get("target_optimizer_updates") == schedule.target_updates,
            "full_summary_target_mismatch",
        ),
        (
            full_summary.get("stochastic_train_eps_proven") is True,
            "stochastic_eps_not_proven",
        ),
        (
            full_summary.get("per_rank_reparameterization_eps_divergent") is True,
            "per_rank_eps_not_divergent",
        ),
        (
            full_summary.get("best_validation_selection_view")
            == REMOTE_FULL_CHECKPOINT_SELECTION_VIEW,
            "best_validation_selection_view_mismatch",
        ),
        (
            full_summary.get("best_validation_selection_reduction")
            == REMOTE_FULL_CHECKPOINT_SELECTION_REDUCTION,
            "best_validation_selection_reduction_mismatch",
        ),
        (plan_applied.get("status") == RUNNER_OK_STATUS, "plan_applied_not_pass"),
        (plan_applied.get("plan_applied") is True, "plan_applied_false"),
        (resume_proof.get("status") == RUNNER_OK_STATUS, "resume_proof_not_pass"),
        (
            resume_proof.get("grad_scaler_state_restore_attempted") is True,
            "grad_scaler_restore_not_attempted",
        ),
        (
            resume_proof.get("grad_scaler_state_restored") is True,
            "grad_scaler_not_restored",
        ),
        (
            resume_proof.get("cuda_rng_state_restore_attempted") is True,
            "cuda_rng_restore_not_attempted",
        ),
        (
            resume_proof.get("cuda_rng_state_restored") is True,
            "cuda_rng_not_restored",
        ),
        (
            resume_proof.get("sampler_progress_restored") is True,
            "sampler_progress_not_restored",
        ),
        (
            resume_proof.get("optimizer_scheduler_progress_restored") is True,
            "optimizer_progress_not_restored",
        ),
        (
            resume_proof.get("beta_progress_restored") is True,
            "beta_progress_not_restored",
        ),
        (gate_health.get("status") == RUNNER_OK_STATUS, "gate_health_not_pass"),
        (
            artifact_manifest.get("status") == RUNNER_OK_STATUS,
            "artifact_manifest_not_pass",
        ),
        (
            artifact_manifest.get("full_run_eligible") is True,
            "manifest_not_full_run_eligible",
        ),
    )
    blockers.extend(
        f"selected_runtime_full_output_{name}" for passed, name in checks if not passed
    )
    amp_execution = training_summary.get("amp_execution")
    amp_payload = amp_execution if isinstance(amp_execution, dict) else {}
    if (
        _float_value(amp_payload.get("grad_scaler_init_scale"))
        != REMOTE_AMP_GRAD_SCALER_INIT_SCALE
    ):
        blockers.append("selected_runtime_full_output_grad_scaler_init_scale_mismatch")
    blockers.extend(_remote_output_plan_scaler_blockers(plan_applied))
    for key in ("final_checkpoint", "best_checkpoint"):
        checkpoint = training_summary.get(key)
        checkpoint_payload = checkpoint if isinstance(checkpoint, dict) else {}
        rel_path = checkpoint_payload.get("path")
        if not isinstance(rel_path, str) or not (output_dir / rel_path).exists():
            blockers.append(f"selected_runtime_full_output_{key}_missing")
    return tuple(blockers)


def _full_expected_interval_checkpoint_names(
    schedule: _RemoteFullSchedule,
) -> tuple[str, ...]:
    steps = tuple(
        range(
            schedule.half_epoch_interval,
            schedule.target_updates + 1,
            schedule.half_epoch_interval,
        ),
    )
    latest = steps[-REMOTE_FULL_INTERVAL_CHECKPOINT_KEEP_COUNT:]
    return tuple(f"step_{step:06d}.pt" for step in latest)


def _full_retained_interval_checkpoint_names(summary: JsonObject) -> tuple[str, ...]:
    raw_checkpoints = summary.get("retained_interval_checkpoints")
    if not isinstance(raw_checkpoints, list):
        return ()
    names: list[str] = []
    for item in raw_checkpoints:
        if not isinstance(item, dict):
            return ()
        path = item.get("path")
        if not isinstance(path, str):
            return ()
        names.append(Path(path).name)
    return tuple(names)


def _remote_full_train_step_blockers(  # noqa: C901, PLR0912
    path: Path,
    *,
    max_batch_size: int,
    world_size: int,
    target_updates: int,
) -> tuple[str, ...]:
    with path.open(encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
        fieldnames = tuple(reader.fieldnames or ())
    if not rows:
        return ("selected_runtime_full_output_train_steps_empty",)
    required = set(REMOTE_DEBUG_REQUIRED_TRAIN_STEP_COLUMNS) | {
        "train_reparameterization",
        "eps_policy",
        "eps_seed_source",
        "eps_zero_fraction",
        "eps_abs_mean",
        "recon_output_rms",
        "x_hat_min",
        "x_hat_max",
        "frac_x_hat_lt_minus1",
        "frac_x_hat_gt_1",
    }
    blockers: list[str] = []
    if required - set(fieldnames):
        blockers.append("selected_runtime_full_output_train_steps_missing_columns")
    successful_rows = [row for row in rows if row.get("amp_step_skipped") == "0"]
    if not successful_rows:
        blockers.append("selected_runtime_full_output_train_steps_no_successful_rows")
        return tuple(blockers)
    # world_size is sourced from the plan (not a frozen 2); only a >= 1 floor remains.
    expected_world_size = world_size
    if world_size < 1:
        blockers.append("selected_runtime_full_output_runtime_world_size_invalid")
    expected_ranks = set(range(expected_world_size))
    expected_successful_row_count = target_updates * expected_world_size
    if len(successful_rows) != expected_successful_row_count:
        blockers.append("selected_runtime_full_output_train_steps_row_count_mismatch")
    coverage: dict[int, set[int]] = {}
    seen_step_ranks: set[tuple[int, int]] = set()
    for row in successful_rows:
        step = _int_value(row.get("successful_optimizer_update_count"))
        rank = _int_value(row.get("rank"))
        if not (1 <= step <= target_updates) or rank not in expected_ranks:
            blockers.append(
                "selected_runtime_full_output_train_steps_step_or_rank_invalid",
            )
            continue
        step_rank = (step, rank)
        if step_rank in seen_step_ranks:
            blockers.append("selected_runtime_full_output_train_steps_duplicate_rank")
            continue
        seen_step_ranks.add(step_rank)
        coverage.setdefault(step, set()).add(rank)
    if len(coverage) != target_updates or any(
        coverage.get(step) != expected_ranks for step in range(1, target_updates + 1)
    ):
        blockers.append("selected_runtime_full_output_train_steps_schedule_incomplete")
    if any(row.get("amp_step_skipped") != "0" for row in rows):
        blockers.append("selected_runtime_full_output_train_steps_amp_skip")
    if any(_int_value(row.get("nonfinite_count")) != 0 for row in rows):
        blockers.append("selected_runtime_full_output_train_steps_nonfinite")
    if any(
        (batch_size := _int_value(row.get("batch_size"))) <= 0
        or (max_batch_size > 0 and batch_size > max_batch_size)
        for row in successful_rows
    ):
        blockers.append("selected_runtime_full_output_train_steps_batch_size_invalid")
    if any(
        row.get("train_reparameterization") != "stochastic_seeded"
        or row.get("eps_policy") != "stochastic_seeded_train_generator"
        or _float_value(row.get("eps_abs_mean")) <= 0.0
        or _float_value(row.get("eps_zero_fraction")) >= 1.0
        for row in successful_rows
    ):
        blockers.append("selected_runtime_full_output_train_steps_not_stochastic")
    return tuple(blockers)


def _is_positive_finite(value: float) -> bool:
    return math.isfinite(value) and value > 0.0


def _remote_full_expected_effective_lr(
    lr_block: JsonObject,
    *,
    reference_lr: float,
    global_batch_size: int,
) -> float | None:
    scaling_applied = lr_block.get("scaling_applied")
    if scaling_applied is False:
        # A flat lr keeps the reference at any batch (no batch_lr_scaling configured).
        return reference_lr
    if scaling_applied is not True:
        return None
    rule = lr_block.get("rule")
    reference_batch = _int_value(lr_block.get("reference_global_batch_size"))
    if not isinstance(rule, str) or reference_batch <= 0 or global_batch_size <= 0:
        return None
    # Re-derive the effective lr through the SAME primitives the runner used, so the
    # gate verifies the rule->exponent mapping AND the scaling formula rather than
    # trusting the recorded number.
    try:
        scaling = BatchLrScaling(reference_global_batch_size=reference_batch, rule=rule)
    except ValueError:
        return None
    return scaled_learning_rate(
        reference_lr=reference_lr,
        scaling=scaling,
        global_batch_size=global_batch_size,
    )


def _remote_full_lr_blockers(
    training_summary: JsonObject,
    *,
    global_batch_size: int,
) -> tuple[str, ...]:
    block = training_summary.get("optimizer_lr_scaling")
    if not isinstance(block, dict):
        return ("selected_runtime_full_output_lr_scaling_missing",)
    lr_block = cast("JsonObject", block)
    blockers: list[str] = []
    if _int_value(lr_block.get("global_batch_size")) != global_batch_size:
        blockers.append("selected_runtime_full_output_lr_scaling_batch_mismatch")
    reference_lr = _float_value(lr_block.get("reference_learning_rate"))
    effective_lr = _float_value(lr_block.get("effective_learning_rate"))
    if not (_is_positive_finite(reference_lr) and _is_positive_finite(effective_lr)):
        # Both learning rates must be present and positive-finite. _float_value maps a
        # missing/garbage field to 0.0, which would otherwise sail through the
        # relationship check as scaled(0.0) == 0.0 -- a fail-open on a truncated or
        # tampered summary that drops the learning rates.
        blockers.append(
            "selected_runtime_full_output_lr_scaling_learning_rate_invalid",
        )
        return tuple(blockers)
    expected_lr = _remote_full_expected_effective_lr(
        lr_block,
        reference_lr=reference_lr,
        global_batch_size=global_batch_size,
    )
    if expected_lr is None or not math.isclose(effective_lr, expected_lr):
        blockers.append(
            "selected_runtime_full_output_lr_scaling_relationship_mismatch",
        )
    return tuple(blockers)


_REMOTE_FULL_CROSS_CONSISTENCY_KEYS = (
    "target_optimizer_updates",
    "optimizer_steps_completed",
    "requested_epochs",
    "optimizer_updates_per_epoch",
    "half_epoch_interval_steps",
    "validation_batches_per_view",
    "validation_views",
)


def _remote_full_cross_consistency_blockers(
    *,
    training_summary: JsonObject,
    full_summary: JsonObject,
) -> tuple[str, ...]:
    # The full summary must agree with the training summary on every schedule field,
    # not merely the target the gate independently anchors -- so a full summary that
    # drifts on updates_per_epoch / half / epochs / the validation cadence is caught
    # directly rather than slipping through unverified (Spec 0011).
    if any(
        full_summary.get(key) != training_summary.get(key)
        for key in _REMOTE_FULL_CROSS_CONSISTENCY_KEYS
    ):
        return ("selected_runtime_full_output_full_summary_schedule_mismatch",)
    return ()


def _remote_full_validation_blockers(
    path: Path,
    *,
    half_epoch_interval: int,
    target_updates: int,
) -> tuple[str, ...]:
    with path.open(encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
        fieldnames = tuple(reader.fieldnames or ())
    if not rows:
        return ("selected_runtime_full_output_validation_empty",)
    blockers: list[str] = []
    required = {
        "optimizer_step",
        "view",
        "batch_count",
        "l1_loss",
        "deterministic_eps_used",
        "corruption_strategy",
    }
    if required - set(fieldnames):
        blockers.append("selected_runtime_full_output_validation_missing_columns")
    observed = {
        (_int_value(row.get("optimizer_step")), row.get("view", "")) for row in rows
    }
    expected_steps = tuple(
        range(
            half_epoch_interval,
            target_updates + 1,
            half_epoch_interval,
        ),
    )
    for step in expected_steps:
        for view in REMOTE_FULL_VALIDATION_VIEWS:
            if (step, view) not in observed:
                blockers.append(
                    "selected_runtime_full_output_validation_schedule_incomplete",
                )
                break
    if any(
        _int_value(row.get("batch_count")) != REMOTE_FULL_VALIDATION_BATCHES_PER_VIEW
        for row in rows
    ):
        blockers.append("selected_runtime_full_output_validation_batch_count_mismatch")
    if any(row.get("deterministic_eps_used") != "true" for row in rows):
        blockers.append("selected_runtime_full_output_validation_not_deterministic")
    if any(not _is_finite_float(row.get("l1_loss", "")) for row in rows):
        blockers.append("selected_runtime_full_output_validation_nonfinite")
    return tuple(blockers)


def _remote_full_fixed25_blockers(*, output_dir: Path) -> tuple[str, ...]:
    """Return blockers for the Spec 0010 fixed-25 embedding-equivariance artifacts.

    Requires the archived originals, a complete equivariance CSV, a manifest with
    the locked rotation convention and promotability label, and the latest
    boundary directory's reconstruction / rotated / latent / grid / PCA artifacts.

    Returns:
        Stable blocker names; empty when the fixed-25 contract passed locally.

    """
    prefix = "selected_runtime_full_output_fixed25"
    fixed25_dir = output_dir / "artifacts" / FIXED25_DIRNAME
    blockers: list[str] = []
    if not (fixed25_dir / ORIGINALS_PT).exists():
        blockers.append(f"{prefix}_originals_missing")
    blockers.extend(
        _fixed25_equivariance_csv_blockers(
            output_dir / "metrics" / "equivariance_25.csv",
            prefix=prefix,
        ),
    )
    manifest_blockers, boundary_steps = _fixed25_manifest_blockers(
        fixed25_dir / MANIFEST_JSON,
        prefix=prefix,
    )
    blockers.extend(manifest_blockers)
    if boundary_steps:
        blockers.extend(
            _fixed25_boundary_blockers(
                fixed25_dir=fixed25_dir,
                optimizer_step=max(boundary_steps),
                prefix=prefix,
            ),
        )
    return tuple(blockers)


def _fixed25_equivariance_csv_blockers(path: Path, *, prefix: str) -> tuple[str, ...]:
    if not path.exists():
        return (f"{prefix}_equivariance_csv_missing",)
    with path.open(encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
        fieldnames = tuple(reader.fieldnames or ())
    blockers: list[str] = []
    if set(EQUIVARIANCE_25_COLUMNS) - set(fieldnames):
        blockers.append(f"{prefix}_equivariance_csv_missing_columns")
    if not rows:
        blockers.append(f"{prefix}_equivariance_csv_empty")
        return tuple(blockers)
    observed_metrics = {row.get("metric_name", "") for row in rows}
    if set(REQUIRED_EQUIVARIANCE_METRICS) - observed_metrics:
        blockers.append(f"{prefix}_equivariance_csv_missing_metrics")
    measured_angles = {str(DEGREES_PER_K * k) for k in MEASURED_K_VALUES}
    if any(row.get("n") != _FIXED25_EXPECTED_SAMPLE_COUNT for row in rows):
        blockers.append(f"{prefix}_equivariance_csv_bad_sample_count")
    if any(row.get("angle_degrees", "") not in measured_angles for row in rows):
        blockers.append(f"{prefix}_equivariance_csv_bad_angle")
    return tuple(blockers)


def _fixed25_manifest_blockers(
    path: Path,
    *,
    prefix: str,
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    if not path.exists():
        return ((f"{prefix}_manifest_missing",), ())
    manifest = _load_json(path)
    blockers: list[str] = []
    rotation = manifest.get("rotation")
    rotation_payload = rotation if isinstance(rotation, dict) else {}
    if (
        rotation_payload.get("method") != "rot90"
        or rotation_payload.get("k_values") != _FIXED25_EXPECTED_K_VALUES
    ):
        blockers.append(f"{prefix}_manifest_rotation_mismatch")
    if "data_source" not in manifest or "promotable" not in manifest:
        blockers.append(f"{prefix}_manifest_missing_promotability")
    elif not _fixed25_manifest_is_promotable(manifest):
        # A promotable full-run output must carry real fixed-25 evidence;
        # synthetic / non-promotable artifacts are never issue #4/#6 evidence.
        blockers.append(f"{prefix}_manifest_non_promotable")
    steps_raw = manifest.get("boundary_optimizer_steps")
    steps = (
        tuple(step for step in steps_raw if isinstance(step, int))
        if isinstance(steps_raw, list)
        else ()
    )
    if not steps:
        blockers.append(f"{prefix}_manifest_no_boundaries")
    return (tuple(blockers), steps)


def _fixed25_manifest_is_promotable(manifest: JsonObject) -> bool:
    return manifest.get("data_source") == "real" and manifest.get("promotable") is True


def _fixed25_boundary_blockers(
    *,
    fixed25_dir: Path,
    optimizer_step: int,
    prefix: str,
) -> tuple[str, ...]:
    boundary_dir = fixed25_dir / f"boundary_{optimizer_step:06d}"
    required = [
        RECONSTRUCTION_PROGRESS_PT,
        LATENT_MU_PT,
        GRID_PNG,
        PCA_PNG,
        FIRST3_PNG,
        *(rotated_pt_name(DEGREES_PER_K * k) for k in MEASURED_K_VALUES),
        *(error_maps_pt_name(DEGREES_PER_K * k) for k in MEASURED_K_VALUES),
    ]
    if any(not (boundary_dir / name).exists() for name in required):
        return (f"{prefix}_boundary_incomplete",)
    return ()


def _remote_output_plan_scaler_blockers(plan_applied: JsonObject) -> tuple[str, ...]:
    expected = plan_applied.get("expected")
    observed = plan_applied.get("observed")
    expected_payload = expected if isinstance(expected, dict) else {}
    observed_payload = observed if isinstance(observed, dict) else {}
    expected_extension = expected_payload.get("runner_amp_extension")
    observed_extension = observed_payload.get("runner_amp_extension")
    expected_runner_amp = (
        expected_extension if isinstance(expected_extension, dict) else {}
    )
    observed_runner_amp = (
        observed_extension if isinstance(observed_extension, dict) else {}
    )
    blockers: list[str] = []
    if (
        _float_value(expected_runner_amp.get("grad_scaler_init_scale"))
        != REMOTE_AMP_GRAD_SCALER_INIT_SCALE
    ):
        blockers.append("selected_runtime_output_plan_expected_scaler_mismatch")
    if (
        _float_value(observed_runner_amp.get("grad_scaler_init_scale"))
        != REMOTE_AMP_GRAD_SCALER_INIT_SCALE
    ):
        blockers.append("selected_runtime_output_plan_observed_scaler_mismatch")
    return tuple(blockers)


def _remote_output_json_blockers(  # noqa: C901, PLR0912, PLR0913
    *,
    runtime_sha256: str,
    training_summary: JsonObject,
    debug_summary: JsonObject,
    plan_applied: JsonObject,
    resume_proof: JsonObject,
    gate_health: JsonObject,
    artifact_manifest: JsonObject,
    selector_readiness: JsonObject,
    tiny_summary: JsonObject,
    gate_summary: JsonObject,
    max_batch_size: int,
    global_batch_size: int,
) -> tuple[str, ...]:
    blockers: list[str] = []
    runtime_config = training_summary.get("runtime_config")
    runtime_config_payload = runtime_config if isinstance(runtime_config, dict) else {}
    if runtime_config_payload.get("sha256") != runtime_sha256:
        blockers.append("selected_runtime_output_runtime_sha256_mismatch")
    if training_summary.get("status") != RUNNER_OK_STATUS:
        blockers.append("selected_runtime_output_training_summary_not_pass")
    if training_summary.get("optimizer_steps_completed") != REMOTE_DEBUG_FINAL_STEP:
        blockers.append("selected_runtime_output_debug_steps_not_8")
    if training_summary.get("amp_step_skipped_count") != 0:
        blockers.append("selected_runtime_output_amp_skips_nonzero")
    if training_summary.get("nonfinite_count") != 0:
        blockers.append("selected_runtime_output_nonfinite_nonzero")
    amp_execution = training_summary.get("amp_execution")
    amp_execution_payload = amp_execution if isinstance(amp_execution, dict) else {}
    if (
        _float_value(amp_execution_payload.get("grad_scaler_init_scale"))
        != REMOTE_AMP_GRAD_SCALER_INIT_SCALE
    ):
        blockers.append("selected_runtime_output_grad_scaler_init_scale_mismatch")
    if debug_summary.get("remote_pass_ready") is not False:
        blockers.append("selected_runtime_output_debug_claims_remote_pass_ready")
    if plan_applied.get("status") != RUNNER_OK_STATUS:
        blockers.append("selected_runtime_output_plan_applied_not_pass")
    if plan_applied.get("plan_applied") is not True:
        blockers.append("selected_runtime_output_plan_applied_false")
    blockers.extend(_remote_output_plan_scaler_blockers(plan_applied))
    if resume_proof.get("status") != RUNNER_OK_STATUS:
        blockers.append("selected_runtime_output_resume_proof_not_pass")
    if (
        resume_proof.get("loaded_successful_optimizer_update_count")
        != REMOTE_DEBUG_RESUME_STEP
    ):
        blockers.append("selected_runtime_output_resume_not_from_step4")
    if resume_proof.get("additional_optimizer_steps") != REMOTE_DEBUG_RESUME_STEP:
        blockers.append("selected_runtime_output_resume_additional_steps_not_4")
    if gate_health.get("status") != RUNNER_OK_STATUS:
        blockers.append("selected_runtime_output_gate_health_not_pass")
    if artifact_manifest.get("status") != RUNNER_OK_STATUS:
        blockers.append("selected_runtime_output_artifact_manifest_not_pass")
    if artifact_manifest.get("reconstruction_sample_nonblank") is not True:
        blockers.append("selected_runtime_output_reconstruction_blank")
    if selector_readiness.get("fixed_32_selector_real") is not True:
        blockers.append("selected_runtime_output_fixed32_selector_not_real")
    if selector_readiness.get("status") != OK_STATUS:
        blockers.append("selected_runtime_output_fixed32_selector_not_pass")
    blockers.extend(
        _remote_output_tiny_summary_blockers(
            tiny_summary=tiny_summary,
            max_batch_size=max_batch_size,
            global_batch_size=global_batch_size,
        ),
    )
    if gate_summary.get("status") != RUNNER_OK_STATUS:
        blockers.append("selected_runtime_output_gate_summary_not_pass")
    return tuple(blockers)


def _remote_output_tiny_summary_blockers(
    *,
    tiny_summary: JsonObject,
    max_batch_size: int,
    global_batch_size: int,
) -> tuple[str, ...]:
    expected_global_epoch_samples = _expected_tiny_global_epoch_samples(
        global_batch_size=global_batch_size,
    )
    expected_per_rank_epoch_samples = _expected_tiny_per_rank_epoch_samples(
        global_epoch_samples=expected_global_epoch_samples,
        global_batch_size=global_batch_size,
        per_device_batch_size=max_batch_size,
    )
    checks = (
        (
            tiny_summary.get("status") == RUNNER_OK_STATUS,
            "selected_runtime_output_tiny_overfit_not_pass",
        ),
        (
            tiny_summary.get("patch_count") == EXPECTED_TINY_SELECTOR_COUNT,
            "selected_runtime_output_tiny_patch_count_not_32",
        ),
        (
            tiny_summary.get("optimizer_steps") == REMOTE_TINY_MAX_STEP,
            "selected_runtime_output_tiny_steps_not_128",
        ),
        (
            tiny_summary.get("amp_step_skipped_count") == 0,
            "selected_runtime_output_tiny_amp_skips_nonzero",
        ),
        (
            tiny_summary.get("nonfinite_count") == 0,
            "selected_runtime_output_tiny_nonfinite_nonzero",
        ),
        (
            _float_value(tiny_summary.get("grad_scaler_init_scale"))
            == REMOTE_AMP_GRAD_SCALER_INIT_SCALE,
            "selected_runtime_output_tiny_grad_scaler_init_scale_mismatch",
        ),
        (
            tiny_summary.get("train_sampler_policy")
            == REMOTE_TINY_FULL_BATCH_SAMPLER_POLICY,
            "selected_runtime_output_tiny_sampler_policy_mismatch",
        ),
        (
            tiny_summary.get("fixed_train_repeated_to_full_batch") is True,
            "selected_runtime_output_tiny_not_repeated_to_full_batch",
        ),
        (
            tiny_summary.get("train_effective_global_epoch_samples")
            == expected_global_epoch_samples,
            "selected_runtime_output_tiny_global_epoch_samples_mismatch",
        ),
        (
            tiny_summary.get("train_effective_per_rank_epoch_samples")
            == expected_per_rank_epoch_samples,
            "selected_runtime_output_tiny_per_rank_epoch_samples_mismatch",
        ),
        (
            tiny_summary.get("observed_batch_sizes") == [max_batch_size],
            "selected_runtime_output_tiny_batch_sizes_not_full",
        ),
        (
            _float_value(tiny_summary.get("l1_improvement_fraction"))
            >= REMOTE_TINY_MIN_IMPROVEMENT_FRACTION,
            "selected_runtime_output_tiny_l1_improvement_low",
        ),
        (
            _float_value(tiny_summary.get("recon_loss_improvement_fraction"))
            >= REMOTE_TINY_MIN_IMPROVEMENT_FRACTION,
            "selected_runtime_output_tiny_recon_improvement_low",
        ),
    )
    return tuple(blocker for passed, blocker in checks if not passed)


def _expected_tiny_global_epoch_samples(*, global_batch_size: int) -> int:
    if global_batch_size <= 0:
        return 0
    return (
        math.ceil(EXPECTED_TINY_SELECTOR_COUNT / global_batch_size) * global_batch_size
    )


def _expected_tiny_per_rank_epoch_samples(
    *,
    global_epoch_samples: int,
    global_batch_size: int,
    per_device_batch_size: int,
) -> int:
    if (
        global_epoch_samples <= 0
        or global_batch_size <= 0
        or per_device_batch_size <= 0
    ):
        return 0
    world_size = max(1, global_batch_size // per_device_batch_size)
    return global_epoch_samples // world_size


def _remote_output_selector_blockers(
    *,
    selector_path: Path,
    selector_readiness: JsonObject,
) -> tuple[str, ...]:
    blockers: list[str] = []
    selector_sha256 = _sha256_file(selector_path)
    selector_status = selector_readiness.get("selector_status")
    selector_status_payload = (
        selector_status if isinstance(selector_status, dict) else {}
    )
    if selector_status_payload.get("sha256") != selector_sha256:
        blockers.append("selected_runtime_output_fixed32_selector_sha_mismatch")
    if selector_status_payload.get("status") != OK_STATUS:
        blockers.append("selected_runtime_output_fixed32_selector_status_not_pass")
    if selector_status_payload.get("canonical_real_ubc") is not True:
        blockers.append("selected_runtime_output_fixed32_selector_status_not_real")
    if selector_status_payload.get("selector_count") != EXPECTED_TINY_SELECTOR_COUNT:
        blockers.append("selected_runtime_output_fixed32_selector_status_count_not_32")
    try:
        document = load_fixed_selector_document(selector_path)
    except (KeyError, TypeError, ValueError) as error:
        return (
            *blockers,
            "selected_runtime_output_fixed32_selector_schema_invalid",
            f"selected_runtime_output_fixed32_selector_schema_detail_{_hash_text(str(error))}",
        )
    blockers.extend(_remote_selector_document_blockers(document))
    return tuple(blockers)


def _remote_selector_document_blockers(
    document: FixedSelectorDocument,
) -> tuple[str, ...]:
    blockers: list[str] = []
    requirements = canonical_real_ubc_requirements()
    header = document.source.header
    checks: tuple[tuple[str, object, object], ...] = (
        ("selector_kind", document.selector_kind, FIXED_32_TRAIN_OVERFIT_KIND),
        ("status", document.status, FIXED_SELECTOR_READY_STATUS),
        ("source_split", document.source_split, "train"),
        ("expected_count", document.expected_count, EXPECTED_TINY_SELECTOR_COUNT),
        ("selector_seed", document.selector_seed, FIXED_32_TRAIN_OVERFIT_SEED),
        (
            "source.dataset_slug",
            document.source.dataset_slug,
            requirements["dataset_slug"],
        ),
        ("source.source_split", document.source.source_split, "train"),
        (
            "source.csv_path.name",
            document.source.csv_path.name,
            requirements["train_csv_filename"],
        ),
        (
            "source.bin_path.name",
            document.source.bin_path.name,
            requirements["train_bin_filename"],
        ),
        (
            "source.csv_sha256",
            document.source.csv_sha256,
            requirements["train_csv_sha256"],
        ),
        (
            "source.bin_file_size",
            document.source.bin_file_size,
            requirements["train_bin_file_size"],
        ),
        ("source.row_count", document.source.row_count, requirements["row_count"]),
        (
            "source.patch_count",
            document.source.patch_count,
            requirements["patch_count"],
        ),
        ("source.idx_policy", document.source.idx_policy, requirements["idx_policy"]),
        (
            "source.crc_checked",
            document.source.crc_checked,
            requirements["crc_checked"],
        ),
        ("header.crc32", header.crc32, requirements["train_header_crc32"]),
        ("header.patch_count", header.patch_count, requirements["patch_count"]),
        ("header.channels", header.channels, requirements["channels"]),
        ("header.height", header.height, requirements["height"]),
        ("header.width", header.width, requirements["width"]),
        ("header.version", header.version, 1),
        ("header.layout", header.layout.decode("ascii"), requirements["layout"]),
    )
    if any(actual != expected for _, actual, expected in checks):
        blockers.append("selected_runtime_output_fixed32_selector_metadata_mismatch")
    ranks = tuple(selector.rank for selector in document.selectors)
    if ranks != tuple(range(EXPECTED_TINY_SELECTOR_COUNT)):
        blockers.append("selected_runtime_output_fixed32_selector_rank_mismatch")
    if any(selector.source_split != "train" for selector in document.selectors):
        blockers.append("selected_runtime_output_fixed32_selector_row_split_mismatch")
    sample_ids = [selector.sample_id for selector in document.selectors]
    if len(set(sample_ids)) != len(sample_ids):
        blockers.append("selected_runtime_output_fixed32_selector_duplicate_sample")
    if len(document.selectors) != EXPECTED_TINY_SELECTOR_COUNT:
        blockers.append("selected_runtime_output_fixed32_selector_count_not_32")
    return tuple(blockers)


def _remote_full_manifest_blockers(
    *,
    output_dir: Path,
    training_summary: JsonObject,
    artifact_manifest: JsonObject,
) -> tuple[str, ...]:
    hashes = artifact_manifest.get("artifact_hashes")
    if not isinstance(hashes, dict):
        return ("selected_runtime_full_output_manifest_hashes_missing",)
    expected_interval_names = {
        f"checkpoint:{name}"
        for name in _full_retained_interval_checkpoint_names(training_summary)
    }
    expected_names = frozenset(
        {
            "training_summary",
            "selected_runtime_full_summary",
            "selected_runtime_plan_applied",
            "checkpoint_resume_proof",
            "gate_health_summary",
            "local_selected_runtime_readiness",
            "train_steps",
            "validation_metrics",
            "gate_health",
            # Spec 0010: the fixed-25 artifacts replace the retired single-patch
            # reconstruction dump for the full run.
            "equivariance_25",
            "fixed25_originals",
            "fixed25_manifest",
            "checkpoint:final.pt",
            "checkpoint:best_model.pt",
        }
        | expected_interval_names,
    )
    blockers: list[str] = []
    observed_names = set(hashes)
    blockers.extend(
        f"selected_runtime_full_output_manifest_missing_{_blocker_token(name)}"
        for name in sorted(expected_names - observed_names)
    )
    observed_interval_names = {
        name
        for name in observed_names
        if name.startswith("checkpoint:step_") and name.endswith(".pt")
    }
    if observed_interval_names != expected_interval_names:
        blockers.append(
            "selected_runtime_full_output_manifest_interval_checkpoints_mismatch",
        )
    for name, value in sorted(hashes.items()):
        if not isinstance(value, str) or len(value) != SHA256_HEX_LENGTH:
            blockers.append(
                f"selected_runtime_full_output_manifest_invalid_hash_{_blocker_token(name)}",
            )
            continue
        path = _full_manifest_artifact_path(output_dir=output_dir, name=name)
        if path is None:
            blockers.append(
                f"selected_runtime_full_output_manifest_unknown_{_blocker_token(name)}",
            )
            continue
        if not path.exists():
            blockers.append(
                f"selected_runtime_full_output_manifest_artifact_missing_{_blocker_token(name)}",
            )
            continue
        if _sha256_file(path) != value:
            blockers.append(
                f"selected_runtime_full_output_manifest_hash_mismatch_{_blocker_token(name)}",
            )
    return tuple(blockers)


def _full_manifest_artifact_path(*, output_dir: Path, name: str) -> Path | None:
    legacy = {
        "training_summary": output_dir / "benchmark" / "training_summary.json",
        "selected_runtime_full_summary": output_dir
        / "benchmark"
        / "selected_runtime_full_summary.json",
        "selected_runtime_plan_applied": output_dir
        / "benchmark"
        / "selected_runtime_plan_applied.json",
        "checkpoint_resume_proof": output_dir
        / "benchmark"
        / "checkpoint_resume_proof.json",
        "gate_health_summary": output_dir / "benchmark" / "gate_health_summary.json",
        "local_selected_runtime_readiness": output_dir
        / "benchmark"
        / "local_selected_runtime_readiness.json",
        "train_steps": output_dir / "metrics" / "train_steps.csv",
        "validation_metrics": output_dir / "metrics" / "validation_metrics.csv",
        "gate_health": output_dir / "metrics" / "gate_health.csv",
        "reconstruction_samples": output_dir
        / "artifacts"
        / "reconstruction_samples.pt",
        "equivariance_25": output_dir / "metrics" / "equivariance_25.csv",
        "fixed25_originals": output_dir / "artifacts" / FIXED25_DIRNAME / ORIGINALS_PT,
        "fixed25_manifest": output_dir / "artifacts" / FIXED25_DIRNAME / MANIFEST_JSON,
    }
    if name.startswith("checkpoint:"):
        return output_dir / "checkpoints" / name.removeprefix("checkpoint:")
    return legacy.get(name)


def _remote_output_manifest_blockers(
    *,
    output_dir: Path,
    artifact_manifest: JsonObject,
) -> tuple[str, ...]:
    hashes = artifact_manifest.get("artifact_hashes")
    if not isinstance(hashes, dict):
        return ("selected_runtime_output_manifest_hashes_missing",)
    blockers: list[str] = []
    observed_names = set(hashes)
    blockers.extend(
        [
            f"selected_runtime_output_manifest_missing_{_blocker_token(name)}"
            for name in sorted(_expected_remote_manifest_names() - observed_names)
        ],
    )
    for name, value in sorted(hashes.items()):
        if not isinstance(value, str) or len(value) != SHA256_HEX_LENGTH:
            blockers.append(
                f"selected_runtime_output_manifest_invalid_hash_{_blocker_token(name)}",
            )
            continue
        path = _manifest_artifact_path(output_dir=output_dir, name=name)
        if path is None:
            blockers.append(
                f"selected_runtime_output_manifest_unknown_{_blocker_token(name)}",
            )
            continue
        if not path.exists():
            blockers.append(
                f"selected_runtime_output_manifest_artifact_missing_{_blocker_token(name)}",
            )
            continue
        if _sha256_file(path) != value:
            blockers.append(
                f"selected_runtime_output_manifest_hash_mismatch_{_blocker_token(name)}",
            )
    return tuple(blockers)


def _expected_remote_manifest_names() -> frozenset[str]:
    benchmark_names = {
        f"benchmark:{name}"
        for name in REMOTE_DEBUG_REQUIRED_BENCHMARK_ARTIFACTS
        if name != "artifact_manifest.json"
    }
    return frozenset(
        {
            *benchmark_names,
            "metrics:gate_health",
            "metrics:train_steps",
            "metrics:tiny_overfit_train_steps",
            "artifact:reconstruction_samples",
        },
    )


def _manifest_artifact_path(*, output_dir: Path, name: str) -> Path | None:
    if name.startswith("benchmark:"):
        return output_dir / "benchmark" / name.removeprefix("benchmark:")
    if name == "metrics:tiny_overfit_train_steps":
        return output_dir / "tiny_overfit_phase" / "metrics" / "train_steps.csv"
    if name.startswith("metrics:"):
        metric_name = name.removeprefix("metrics:")
        metric_filename = (
            metric_name if metric_name.endswith(".csv") else f"{metric_name}.csv"
        )
        return output_dir / "metrics" / metric_filename
    if name.startswith("artifact:"):
        artifact_name = name.removeprefix("artifact:")
        artifact_filename = (
            "reconstruction_samples.pt"
            if artifact_name == "reconstruction_samples"
            else artifact_name
        )
        return output_dir / "artifacts" / artifact_filename
    legacy = {
        "training_summary": output_dir / "benchmark" / "training_summary.json",
        "selected_runtime_debug_summary": output_dir
        / "benchmark"
        / "selected_runtime_debug_summary.json",
        "selected_runtime_plan_applied": output_dir
        / "benchmark"
        / "selected_runtime_plan_applied.json",
        "checkpoint_resume_proof": output_dir
        / "benchmark"
        / "checkpoint_resume_proof.json",
        "gate_health_summary": output_dir / "benchmark" / "gate_health_summary.json",
        "local_selected_runtime_readiness": output_dir
        / "benchmark"
        / "local_selected_runtime_readiness.json",
        "tiny_overfit_summary": output_dir / "benchmark" / "tiny_overfit_summary.json",
        "train_steps": output_dir / "metrics" / "train_steps.csv",
        "gate_health": output_dir / "metrics" / "gate_health.csv",
        "reconstruction_samples": output_dir
        / "artifacts"
        / "reconstruction_samples.pt",
    }
    if name.startswith("checkpoint:"):
        return output_dir / "checkpoints" / name.removeprefix("checkpoint:")
    return legacy.get(name)


def _remote_output_gate_health_blockers(path: Path) -> tuple[str, ...]:
    with path.open(encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
        fieldnames = tuple(reader.fieldnames or ())
    if not rows:
        return ("selected_runtime_output_gate_health_empty",)
    missing = set(GATE_HEALTH_COLUMNS) - set(fieldnames)
    if missing:
        return ("selected_runtime_output_gate_health_missing_columns",)
    blockers: list[str] = []
    if any(row.get("gate_health_status") != "pass" for row in rows):
        blockers.append("selected_runtime_output_gate_health_row_not_pass")
    if any(row.get("row_id") != EXPECTED_SELECTED_ROW_ID for row in rows):
        blockers.append("selected_runtime_output_gate_health_row_id_mismatch")
    if any(row.get("candidate_row_id") != EXPECTED_SELECTED_ROW_ID for row in rows):
        blockers.append("selected_runtime_output_gate_health_candidate_mismatch")
    if any(row.get("runtime_policy_id") != EXPECTED_RUNTIME_POLICY_ID for row in rows):
        blockers.append("selected_runtime_output_gate_health_policy_mismatch")
    if any(
        not _is_finite_float(row.get(column, ""))
        for row in rows
        for column in REMOTE_GATE_HEALTH_FINITE_COLUMNS
    ):
        blockers.append("selected_runtime_output_gate_health_nonfinite")
    if any(
        _float_value(row.get(column, "")) >= REMOTE_GATE_HEALTH_MAX_SATURATION_FRACTION
        for row in rows
        for column in REMOTE_GATE_HEALTH_SATURATION_COLUMNS
    ):
        blockers.append("selected_runtime_output_gate_health_saturated")
    return tuple(blockers)


def _remote_output_train_step_blockers(
    path: Path,
    *,
    max_batch_size: int,
) -> tuple[str, ...]:
    with path.open(encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
        fieldnames = tuple(reader.fieldnames or ())
    if not rows:
        return ("selected_runtime_output_train_steps_empty",)
    blockers: list[str] = []
    if set(REMOTE_DEBUG_REQUIRED_TRAIN_STEP_COLUMNS) - set(fieldnames):
        blockers.append("selected_runtime_output_train_steps_missing_columns")
    successful_steps = {
        step
        for row in rows
        if row.get("amp_step_skipped") == "0"
        and (step := _int_value(row.get("successful_optimizer_update_count"))) > 0
    }
    if tuple(sorted(successful_steps)) != REMOTE_DEBUG_REQUIRED_SUCCESSFUL_STEPS:
        blockers.append("selected_runtime_output_train_steps_wrong_step_set")
    if max(successful_steps, default=0) != REMOTE_DEBUG_FINAL_STEP:
        blockers.append("selected_runtime_output_train_steps_do_not_reach_8")
    if (
        min(successful_steps, default=REMOTE_DEBUG_FINAL_STEP)
        <= REMOTE_DEBUG_RESUME_STEP
    ):
        blockers.append("selected_runtime_output_train_steps_not_resumed_only")
    if any(row.get("amp_step_skipped") != "0" for row in rows):
        blockers.append("selected_runtime_output_train_steps_amp_skip")
    if any(_int_value(row.get("nonfinite_count")) != 0 for row in rows):
        blockers.append("selected_runtime_output_train_steps_nonfinite")
    if any(
        (batch_size := _int_value(row.get("batch_size"))) <= 0
        or (max_batch_size > 0 and batch_size > max_batch_size)
        for row in rows
    ):
        blockers.append("selected_runtime_output_train_steps_batch_size_invalid")
    return tuple(blockers)


def _remote_output_tiny_train_step_blockers(
    path: Path,
    *,
    max_batch_size: int,
    global_batch_size: int,
) -> tuple[str, ...]:
    if not path.exists():
        return ("selected_runtime_output_tiny_train_steps_missing",)
    with path.open(encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
        fieldnames = tuple(reader.fieldnames or ())
    if not rows:
        return ("selected_runtime_output_tiny_train_steps_empty",)
    blockers: list[str] = []
    if set(REMOTE_DEBUG_REQUIRED_TRAIN_STEP_COLUMNS) - set(fieldnames):
        blockers.append("selected_runtime_output_tiny_train_steps_missing_columns")
    expected_rank_count = _expected_rank_count(
        global_batch_size=global_batch_size,
        per_device_batch_size=max_batch_size,
    )
    successful_rows = [row for row in rows if row.get("amp_step_skipped") == "0"]
    blockers.extend(
        _tiny_train_step_rank_blockers(
            successful_rows=successful_rows,
            expected_rank_count=expected_rank_count,
        ),
    )
    blockers.extend(
        _tiny_train_step_value_blockers(
            rows=rows,
            successful_rows=successful_rows,
            max_batch_size=max_batch_size,
            expected_rank_count=expected_rank_count,
        ),
    )
    return tuple(blockers)


def _tiny_train_step_rank_blockers(
    *,
    successful_rows: Sequence[CsvRow],
    expected_rank_count: int,
) -> tuple[str, ...]:
    blockers: list[str] = []
    expected_steps = tuple(range(1, REMOTE_TINY_MAX_STEP + 1))
    ranks = sorted({row.get("rank", "") for row in successful_rows})
    if ranks != [str(rank) for rank in range(expected_rank_count)]:
        blockers.append("selected_runtime_output_tiny_train_steps_rank_mismatch")
    for rank in ranks:
        rank_steps = tuple(
            sorted(
                _int_value(row.get("successful_optimizer_update_count"))
                for row in successful_rows
                if row.get("rank") == rank
            ),
        )
        if rank_steps != expected_steps:
            blockers.append("selected_runtime_output_tiny_train_steps_wrong_step_set")
            break
    return tuple(blockers)


def _tiny_train_step_value_blockers(
    *,
    rows: Sequence[CsvRow],
    successful_rows: Sequence[CsvRow],
    max_batch_size: int,
    expected_rank_count: int,
) -> tuple[str, ...]:
    blockers: list[str] = []
    if len(successful_rows) != REMOTE_TINY_MAX_STEP * expected_rank_count:
        blockers.append("selected_runtime_output_tiny_train_steps_wrong_row_count")
    if any(row.get("amp_step_skipped") != "0" for row in rows):
        blockers.append("selected_runtime_output_tiny_train_steps_amp_skip")
    if any(_int_value(row.get("nonfinite_count")) != 0 for row in rows):
        blockers.append("selected_runtime_output_tiny_train_steps_nonfinite")
    if any(
        _int_value(row.get("batch_size")) != max_batch_size for row in successful_rows
    ):
        blockers.append("selected_runtime_output_tiny_train_steps_batch_size_not_full")
    if any(not _is_finite_float(row.get("grad_norm", "")) for row in rows):
        blockers.append("selected_runtime_output_tiny_train_steps_grad_norm_nonfinite")
    return tuple(blockers)


def _expected_rank_count(*, global_batch_size: int, per_device_batch_size: int) -> int:
    if global_batch_size <= 0 or per_device_batch_size <= 0:
        return 1
    return max(1, global_batch_size // per_device_batch_size)


def _float_value(value: object) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _is_finite_float(value: object) -> bool:
    if isinstance(value, str) and not value.strip():
        return False
    try:
        parsed = float(cast("str | int | float", value))
    except (TypeError, ValueError):
        return False
    return math.isfinite(parsed)


def _int_value(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _blocker_token(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


def _structured_readiness_blockers(  # noqa: PLR0913
    *,
    debug_config_path: Path,
    tiny_config_path: Path,
    selected_runtime_path: Path,
    selector_generation_mode: SelectorGenerationMode,
    data_root: str | None,
    fixed_train_patches: Path,
) -> tuple[str, ...]:
    if selector_generation_mode == REMOTE_GENERATE_MODE:
        return _remote_generate_structured_readiness_blockers(
            debug_config_path=debug_config_path,
            selected_runtime_path=selected_runtime_path,
        )
    with tempfile.TemporaryDirectory(
        prefix="eqvae_selected_runtime_verify_",
    ) as output_root:
        result = write_selected_runtime_gate(
            SelectedRuntimeGateRequest(
                debug_config_path=debug_config_path,
                tiny_config_path=tiny_config_path,
                selected_runtime_path=selected_runtime_path,
                output_dir=Path(output_root),
                run_name="selected_runtime_push_readiness_probe",
                data_root=data_root,
                fixed_train_patches=fixed_train_patches,
                selector_generation_mode=LOCAL_SELECTOR_MODE,
            ),
        )
        readiness = _load_json(result.local_readiness)
    return _local_readiness_blockers(
        readiness,
        selector_generation_mode=selector_generation_mode,
    )


def _remote_generate_structured_readiness_blockers(
    *,
    debug_config_path: Path,
    selected_runtime_path: Path,
) -> tuple[str, ...]:
    runtime_payload = _load_json(selected_runtime_path)
    runtime_errors = _selected_runtime_errors(
        runtime_payload,
        selected_runtime_path=selected_runtime_path,
    )
    with tempfile.TemporaryDirectory(prefix="eqvae_fixed32_remote_generate_") as root:
        root_path = Path(root)
        result = write_fixed32_remote_generate_readiness(
            Fixed32RemoteGenerateReadinessRequest(
                output_dir=root_path / "readiness",
                synthetic_root=root_path / "synthetic-root",
                config_path=debug_config_path,
                masked_holdout_csv=Path("docs/data/ubc_ocean_masked_holdout_ids.csv"),
            ),
        )
        readiness = _load_json(result.readiness_path)
    blockers = [
        *readiness_blockers(readiness),
        *(("selected_runtime_transport_validation_failed",) if runtime_errors else ()),
    ]
    if not REAL_UBC_SELECTED_RUNTIME_TRAIN_RUNNER_IMPLEMENTED:
        blockers.append("selected_runtime_runner_capability_missing")
    if not SELECTED_RUNTIME_DEBUG_WRAPPER_WIRED_TO_REAL_RUNNER:
        blockers.append("selected_runtime_debug_wrapper_not_wired_to_real_runner")
    return tuple(blockers)


def _local_readiness_blockers(
    readiness: JsonObject,
    *,
    selector_generation_mode: SelectorGenerationMode,
) -> tuple[str, ...]:
    blockers: list[str] = []
    if readiness.get("status") != OK_STATUS:
        blockers.append("local_selected_runtime_readiness_status_not_pass")
    if readiness.get("remote_pass_ready") is not True:
        blockers.append("local_selected_runtime_readiness_remote_pass_ready_not_true")
    if readiness.get("real_train_runner_implemented") is not True:
        blockers.append(
            "local_selected_runtime_readiness_real_train_runner_not_implemented",
        )
    if readiness.get("fixed_32_selector_real") is not True:
        blockers.append("local_selected_runtime_readiness_fixed_32_selector_real_false")
    if readiness.get("selector_generation_mode") != selector_generation_mode:
        blockers.append("local_selected_runtime_readiness_selector_mode_mismatch")

    component_status = readiness.get("component_status")
    if not isinstance(component_status, dict):
        blockers.append("local_selected_runtime_readiness_component_status_missing")
        return tuple(blockers)
    component_payload = cast("JsonObject", component_status)
    for name, raw_status in sorted(component_payload.items()):
        status = _string_value(raw_status)
        if status != OK_STATUS:
            blockers.append(
                "local_selected_runtime_readiness_component_"
                f"{name}_{status or 'missing'}_not_pass",
            )
    return tuple(blockers)


@dataclass(frozen=True)
class _GateArtifactPaths:
    gate_summary: Path
    training_summary: Path
    selected_runtime_debug_summary: Path
    checkpoint_resume_proof: Path
    selected_runtime_plan_applied: Path
    local_readiness: Path
    tiny_overfit_summary: Path
    artifact_manifest: Path
    gate_health_summary: Path
    train_metrics: Path
    gate_health_metrics: Path


@dataclass(frozen=True)
class _GateSummaryContext:
    request: SelectedRuntimeGateRequest
    debug_resolved: ResolvedConfig
    tiny_resolved: ResolvedConfig
    runtime_identity: JsonObject
    selector_status: JsonObject
    blockers: tuple[str, ...]


def _artifact_paths(output_dir: Path) -> _GateArtifactPaths:
    benchmark_dir = output_dir / "benchmark"
    metrics_dir = output_dir / "metrics"
    return _GateArtifactPaths(
        gate_summary=benchmark_dir / "selected_runtime_gate_summary.json",
        training_summary=benchmark_dir / "training_summary.json",
        selected_runtime_debug_summary=(
            benchmark_dir / "selected_runtime_debug_summary.json"
        ),
        checkpoint_resume_proof=benchmark_dir / "checkpoint_resume_proof.json",
        selected_runtime_plan_applied=(
            benchmark_dir / "selected_runtime_plan_applied.json"
        ),
        local_readiness=benchmark_dir / "local_selected_runtime_readiness.json",
        tiny_overfit_summary=benchmark_dir / "tiny_overfit_summary.json",
        artifact_manifest=benchmark_dir / "artifact_manifest.json",
        gate_health_summary=benchmark_dir / "gate_health_summary.json",
        train_metrics=metrics_dir / "train_metrics.csv",
        gate_health_metrics=metrics_dir / "gate_health.csv",
    )


def _gate_summary(
    context: _GateSummaryContext,
) -> JsonObject:
    request = context.request
    return {
        "schema_version": GATE_SCHEMA_VERSION,
        "status": FAIL_STATUS,
        "status_scope": "fail_closed_real_gate_contract",
        "benchmark_kind": GATE_KIND,
        "benchmark_source": GATE_SOURCE,
        "full_run_eligible": False,
        "full_training_launch_ready": False,
        "run_name": request.run_name,
        "data": "ubc-pre-shuffled",
        "data_root": request.data_root or "auto",
        "debug_config": {
            "path": str(request.debug_config_path),
            "invoked_config_hash": context.debug_resolved.invoked_config_hash,
            "effective_config_hash": context.debug_resolved.effective_config_hash,
        },
        "tiny_config": {
            "path": str(request.tiny_config_path),
            "invoked_config_hash": context.tiny_resolved.invoked_config_hash,
            "effective_config_hash": context.tiny_resolved.effective_config_hash,
        },
        "selected_runtime": context.runtime_identity,
        "fixed_train_patches": context.selector_status,
        "component_status": {
            "selected_runtime_transport": (
                OK_STATUS
                if not cast(
                    "list[object]",
                    context.runtime_identity["validation_errors"],
                )
                else FAIL_STATUS
            ),
            "selected_runtime_plan_applied": FAIL_STATUS,
            "real_ubc_debug": FAIL_STATUS,
            "checkpoint_resume": FAIL_STATUS,
            "artifact_manifest": FAIL_STATUS,
            "gate_health": FAIL_STATUS,
            "tiny_overfit": FAIL_STATUS,
            "local_readiness": FAIL_STATUS,
        },
        "launch_blockers_remaining": list(context.blockers),
    }


def _training_summary(
    *,
    request: SelectedRuntimeGateRequest,
    resolved: ResolvedConfig,
    runtime_identity: JsonObject,
    blockers: tuple[str, ...],
) -> JsonObject:
    return {
        "schema_version": GATE_SCHEMA_VERSION,
        "status": FAIL_STATUS,
        "proof_scope": "real_kaggle_gate_fail_closed",
        "benchmark_kind": GATE_KIND,
        "benchmark_source": GATE_SOURCE,
        "full_run_eligible": False,
        "run_name": request.run_name,
        "data": "ubc-pre-shuffled",
        "data_root": request.data_root or "auto",
        "config_path": str(request.debug_config_path),
        "config_sha256": resolved.invoked_config_hash,
        "effective_config_sha256": resolved.effective_config_hash,
        "runtime_config": runtime_identity,
        "optimizer_steps_completed": 0,
        "amp_step_skipped_count": 0,
        "scheduler_advanced_after_amp_skip": False,
        "checkpoint_count": 0,
        "failure_kind": REMOTE_DEBUG_PENDING_BLOCKER,
        "launch_blockers_remaining": list(blockers),
    }


def _debug_summary(
    *,
    request: SelectedRuntimeGateRequest,
    runtime_identity: JsonObject,
    blockers: tuple[str, ...],
) -> JsonObject:
    return {
        "schema_version": GATE_SCHEMA_VERSION,
        "status": FAIL_STATUS,
        "proof_scope": "real_kaggle_gate_fail_closed",
        "full_run_eligible": False,
        "run_name": request.run_name,
        "runtime_config": runtime_identity,
        "optimizer_steps_completed": 0,
        "checkpoint_written": False,
        "checkpoint_resume_proof_status": FAIL_STATUS,
        "artifact_manifest": "benchmark/artifact_manifest.json",
        "real_kaggle_debug_status": FAIL_STATUS,
        "failure_kind": REMOTE_DEBUG_PENDING_BLOCKER,
        "launch_blockers_remaining": list(blockers),
    }


def _resume_summary(
    *,
    runtime_identity: JsonObject,
    blockers: tuple[str, ...],
) -> JsonObject:
    return {
        "schema_version": GATE_SCHEMA_VERSION,
        "status": FAIL_STATUS,
        "proof_scope": "real_kaggle_gate_fail_closed",
        "full_run_eligible": False,
        "runtime_config": runtime_identity,
        "resume_checkpoint": "",
        "optimizer_state_restored": False,
        "model_state_restored": False,
        "python_rng_state_restored": False,
        "numpy_generator_state_restored": False,
        "torch_cpu_rng_state_restored": False,
        "torch_cuda_rng_state_status": "missing_real_gate_checkpoint",
        "lr_scheduler_state_status": "missing_real_gate_checkpoint",
        "beta_schedule_state_status": "missing_real_gate_checkpoint",
        "amp_scaler_state_status": "missing_real_gate_checkpoint",
        "ddp_progress_state_status": "missing_real_gate_checkpoint",
        "failure_kind": "missing_real_checkpoint_resume_proof",
        "launch_blockers_remaining": list(blockers),
    }


def _tiny_summary(
    *,
    runtime_identity: JsonObject,
    selector_status: JsonObject,
    blockers: tuple[str, ...],
) -> JsonObject:
    return {
        "schema_version": GATE_SCHEMA_VERSION,
        "status": FAIL_STATUS,
        "proof_scope": "real_kaggle_gate_fail_closed",
        "full_run_eligible": False,
        "runtime_config": runtime_identity,
        "fixed_train_patches": selector_status["path"],
        "fixed_train_patches_sha256": selector_status["sha256"],
        "patch_count": selector_status["selector_count"],
        "optimizer_steps": 0,
        "smoothing_window_steps": 25,
        "corruption_strategy": "indexed_masked",
        "eval_views": ["train_clean", "train_corrupted_fixed_seed"],
        "initial_smoothed_l1": 0.0,
        "final_smoothed_l1": 0.0,
        "initial_smoothed_recon_loss": 0.0,
        "final_smoothed_recon_loss": 0.0,
        "l1_improvement_fraction": 0.0,
        "recon_loss_improvement_fraction": 0.0,
        "gate_health_status": FAIL_STATUS,
        "real_tiny_overfit_status": FAIL_STATUS,
        "failure_kind": _tiny_failure_kind(selector_status),
        "launch_blockers_remaining": list(blockers),
    }


def _tiny_failure_kind(selector_status: JsonObject) -> str:
    if selector_status.get("status") != OK_STATUS:
        return _string_value(selector_status.get("failure_kind"))
    return "missing_real_tiny_overfit_proof"


def _gate_health_summary(
    *,
    runtime_identity: JsonObject,
    blockers: tuple[str, ...],
) -> JsonObject:
    return {
        "schema_version": GATE_SCHEMA_VERSION,
        "status": FAIL_STATUS,
        "benchmark_kind": GATE_KIND,
        "benchmark_source": GATE_SOURCE,
        "full_run_eligible": False,
        "runtime_config": runtime_identity,
        "logged_intervals": 0,
        "module_count": 0,
        "nonfinite_count": None,
        "failure_kind": "missing_real_gate_health_rows",
        "launch_blockers_remaining": list(blockers),
    }


def _artifact_manifest(*, paths: _GateArtifactPaths) -> JsonObject:
    artifacts = {
        "selected_runtime_gate_summary": paths.gate_summary,
        "training_summary": paths.training_summary,
        "selected_runtime_debug_summary": paths.selected_runtime_debug_summary,
        "checkpoint_resume_proof": paths.checkpoint_resume_proof,
        "selected_runtime_plan_applied": paths.selected_runtime_plan_applied,
        "local_selected_runtime_readiness": paths.local_readiness,
        "tiny_overfit_summary": paths.tiny_overfit_summary,
        "gate_health_summary": paths.gate_health_summary,
        "train_metrics": paths.train_metrics,
        "gate_health_metrics": paths.gate_health_metrics,
    }
    missing = [name for name, path in sorted(artifacts.items()) if not path.exists()]
    return cast(
        "JsonObject",
        {
            "schema_version": GATE_SCHEMA_VERSION,
            "status": FAIL_STATUS,
            "contract_written": not missing,
            "proof_scope": "real_kaggle_gate_fail_closed",
            "full_run_eligible": False,
            "artifact_hashes": {
                name: _sha256_file(path)
                for name, path in sorted(artifacts.items())
                if path.exists()
            },
            "checkpoint_count": 0,
            "metric_row_count": 0,
            "reconstruction_sample_nonblank": False,
            "missing_artifacts": missing,
            "failure_kind": "real_gate_artifact_contract_written_but_not_passed",
        },
    )


def _local_readiness_summary(
    *,
    selector_generation_mode: SelectorGenerationMode,
    runtime_identity: JsonObject,
    selector_status: JsonObject,
    blockers: tuple[str, ...],
    artifact_manifest: JsonObject | None,
) -> JsonObject:
    artifact_manifest_status = (
        "pending"
        if artifact_manifest is None
        else _artifact_manifest_component_status(artifact_manifest)
    )
    component_status = {
        "selected_runtime_plan": (
            OK_STATUS
            if not cast("list[object]", runtime_identity["validation_errors"])
            else FAIL_STATUS
        ),
        "selected_runtime_plan_applied": FAIL_STATUS,
        "ubc_format_mechanics": FAIL_STATUS,
        "amp_progress": FAIL_STATUS,
        "checkpoint_resume": FAIL_STATUS,
        "fixed_32_selector": _string_value(selector_status.get("status")),
        "artifact_manifest": artifact_manifest_status,
        "gate_health": FAIL_STATUS,
    }
    return cast(
        "JsonObject",
        {
            "schema_version": GATE_SCHEMA_VERSION,
            "status": FAIL_STATUS,
            "status_scope": "fail_closed_real_gate_contract",
            "full_run_eligible": False,
            "remote_pass_ready": False,
            "real_train_runner_implemented": (
                REAL_UBC_SELECTED_RUNTIME_TRAIN_RUNNER_IMPLEMENTED
            ),
            "selector_generation_mode": selector_generation_mode,
            "remote_selector_generation_ready": (
                selector_generation_mode == REMOTE_GENERATE_MODE
            ),
            "fixed_32_selector_real": selector_status.get("canonical_real_ubc") is True,
            "component_status": component_status,
            "selected_runtime": runtime_identity,
            "fixed_train_patches": selector_status,
            "failure_kind": "selected_runtime_local_readiness_blocked",
            "launch_blockers_remaining": list(blockers),
        },
    )


def _artifact_manifest_component_status(artifact_manifest: JsonObject) -> str:
    missing = artifact_manifest.get("missing_artifacts")
    status = artifact_manifest.get("status")
    if isinstance(status, str) and isinstance(missing, list) and not missing:
        return status
    return FAIL_STATUS


def _selected_runtime_errors(
    payload: JsonObject,
    *,
    selected_runtime_path: Path,
) -> tuple[str, ...]:
    return selected_runtime_plan_errors(
        payload,
        selected_runtime_path=selected_runtime_path,
    )


def _runtime_identity(
    *,
    path: Path,
    payload: JsonObject,
    errors: tuple[str, ...],
) -> JsonObject:
    return selected_runtime_identity_payload(path=path, payload=payload, errors=errors)


def _selector_path(*, config_path: Path, resolved: ResolvedConfig) -> Path:
    data = resolved.effective_config.get("data")
    if not isinstance(data, dict):
        message = "tiny config data must be an object"
        raise TypeError(message)
    value = data.get("fixed_train_patches")
    if not isinstance(value, str) or not value:
        message = "tiny config must declare data.fixed_train_patches"
        raise ValueError(message)
    path = Path(value)
    if path.is_absolute():
        return path
    for parent in config_path.resolve().parents:
        candidate = parent / path
        if candidate.exists():
            return candidate
    return Path.cwd() / path


def _launch_blockers(
    *,
    runtime_errors: tuple[str, ...],
    selector_status: JsonObject,
) -> tuple[str, ...]:
    blockers = [
        "missing_real_checkpoint_resume_proof",
        "missing_real_gate_health_rows",
        "missing_real_tiny_overfit_proof",
    ]
    if not SELECTED_RUNTIME_DEBUG_WRAPPER_WIRED_TO_REAL_RUNNER:
        blockers.insert(0, "selected_runtime_debug_wrapper_not_wired_to_real_runner")
    if not REAL_UBC_SELECTED_RUNTIME_TRAIN_RUNNER_IMPLEMENTED:
        blockers.insert(0, "selected_runtime_runner_capability_missing")
    if not SELECTED_RUNTIME_PLAN_APPLIED_TO_TRAINING:
        blockers.append("selected_runtime_runtime_plan_not_applied_to_training")
    if runtime_errors:
        blockers.append("selected_runtime_transport_validation_failed")
    if selector_status.get("status") != OK_STATUS:
        blockers.append(_string_value(selector_status.get("failure_kind")))
    return tuple(blocker for blocker in blockers if blocker)


def _push_readiness_blockers(
    *,
    runtime_errors: tuple[str, ...],
    selector_status: JsonObject,
    selector_generation_mode: SelectorGenerationMode,
) -> tuple[str, ...]:
    blockers: list[str] = []
    if not SELECTED_RUNTIME_DEBUG_WRAPPER_WIRED_TO_REAL_RUNNER:
        blockers.append("selected_runtime_debug_wrapper_not_wired_to_real_runner")
    if not REAL_UBC_SELECTED_RUNTIME_TRAIN_RUNNER_IMPLEMENTED:
        blockers.append("selected_runtime_runner_capability_missing")
    if (
        selector_generation_mode != REMOTE_GENERATE_MODE
        and not SELECTED_RUNTIME_PLAN_APPLIED_TO_TRAINING
    ):
        blockers.append("selected_runtime_runtime_plan_not_applied_to_training")
    if runtime_errors:
        blockers.append("selected_runtime_transport_validation_failed")
    if (
        selector_generation_mode != REMOTE_GENERATE_MODE
        and selector_status.get("status") != OK_STATUS
    ):
        blockers.append(_string_value(selector_status.get("failure_kind")))
    return tuple(blocker for blocker in blockers if blocker)


def _readiness_config_blockers(
    *,
    resolved: ResolvedConfig,
    gate_key: str,
    selector_generation_mode: SelectorGenerationMode,
) -> tuple[str, ...]:
    gate = resolved.effective_config.get(gate_key)
    if not isinstance(gate, dict):
        return (f"{gate_key}_missing",)
    gate_payload = cast("JsonObject", gate)
    if selector_generation_mode == REMOTE_GENERATE_MODE:
        blockers: list[str] = []
        if gate_payload.get("selector_generation_mode") != REMOTE_GENERATE_MODE:
            blockers.append(f"{gate_key}_selector_generation_mode_not_remote_generate")
        if gate_payload.get("remote_selector_generation_ready") is not True:
            blockers.append(f"{gate_key}_remote_selector_generation_ready_not_true")
        if gate_payload.get("real_train_runner_implemented") is not True:
            blockers.append(f"{gate_key}_real_train_runner_implemented_not_true")
        if gate_payload.get("fixed_32_selector_real") is not False:
            blockers.append(f"{gate_key}_fixed_32_selector_real_must_remain_false")
        if gate_payload.get("remote_pass_ready") is not False:
            blockers.append(f"{gate_key}_remote_pass_ready_must_remain_false")
        return tuple(blockers)
    flags = (
        "remote_pass_ready",
        "real_train_runner_implemented",
        "fixed_32_selector_real",
    )
    return tuple(
        f"{gate_key}_{flag}_not_true"
        for flag in flags
        if gate_payload.get(flag) is not True
    )


def _dedupe_strings(values: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return tuple(deduped)


def _load_json(path: Path) -> JsonObject:
    payload = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(payload, dict):
        message = f"Expected JSON object in {path}"
        raise TypeError(message)
    return cast("JsonObject", payload)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _string_value(value: object) -> str:
    return value if isinstance(value, str) else ""


__all__ = [
    "EXPECTED_DATASET_SLUG",
    "EXPECTED_RUNTIME_POLICY_ID",
    "EXPECTED_SELECTED_ROW_ID",
    "GATE_KIND",
    "GATE_SOURCE",
    "LOCAL_SELECTOR_MODE",
    "REMOTE_GENERATE_MODE",
    "SelectedRuntimeGateRequest",
    "SelectedRuntimeGateResult",
    "SelectorGenerationMode",
    "verify_selected_runtime_debug_output",
    "verify_selected_runtime_debug_push_ready",
    "write_selected_runtime_gate",
]
