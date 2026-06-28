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
from typing import Literal, cast

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
from eqvae.config import ResolvedConfig, resolve_json_config
from eqvae.data.fixed_selectors import (
    FIXED_32_TRAIN_OVERFIT_KIND,
    FIXED_32_TRAIN_OVERFIT_SEED,
    FIXED_SELECTOR_READY_STATUS,
    FixedSelectorDocument,
    load_fixed_selector_document,
)
from eqvae.training.selected_runtime import (
    EXPECTED_DATASET_SLUG,
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
REMOTE_DEBUG_FINAL_STEP = 8
REMOTE_DEBUG_RESUME_STEP = 4
REMOTE_DEBUG_REQUIRED_SUCCESSFUL_STEPS = tuple(
    range(REMOTE_DEBUG_RESUME_STEP + 1, REMOTE_DEBUG_FINAL_STEP + 1),
)
REMOTE_TINY_MAX_STEP = 128
REMOTE_TINY_MIN_IMPROVEMENT_FRACTION = 0.01
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
    return _dedupe_strings(tuple(blockers))


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
    if debug_summary.get("remote_pass_ready") is not False:
        blockers.append("selected_runtime_output_debug_claims_remote_pass_ready")
    if plan_applied.get("status") != RUNNER_OK_STATUS:
        blockers.append("selected_runtime_output_plan_applied_not_pass")
    if plan_applied.get("plan_applied") is not True:
        blockers.append("selected_runtime_output_plan_applied_false")
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
    if tiny_summary.get("status") != RUNNER_OK_STATUS:
        blockers.append("selected_runtime_output_tiny_overfit_not_pass")
    if tiny_summary.get("patch_count") != EXPECTED_TINY_SELECTOR_COUNT:
        blockers.append("selected_runtime_output_tiny_patch_count_not_32")
    if tiny_summary.get("optimizer_steps") != REMOTE_TINY_MAX_STEP:
        blockers.append("selected_runtime_output_tiny_steps_not_128")
    if (
        _float_value(tiny_summary.get("l1_improvement_fraction"))
        < REMOTE_TINY_MIN_IMPROVEMENT_FRACTION
    ):
        blockers.append("selected_runtime_output_tiny_l1_improvement_low")
    if (
        _float_value(tiny_summary.get("recon_loss_improvement_fraction"))
        < REMOTE_TINY_MIN_IMPROVEMENT_FRACTION
    ):
        blockers.append("selected_runtime_output_tiny_recon_improvement_low")
    if gate_summary.get("status") != RUNNER_OK_STATUS:
        blockers.append("selected_runtime_output_gate_summary_not_pass")
    return tuple(blockers)


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
            "artifact:reconstruction_samples",
        },
    )


def _manifest_artifact_path(*, output_dir: Path, name: str) -> Path | None:
    if name.startswith("benchmark:"):
        return output_dir / "benchmark" / name.removeprefix("benchmark:")
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
