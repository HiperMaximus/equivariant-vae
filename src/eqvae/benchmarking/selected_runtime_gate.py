# Copyright 2026 HiperMaximus
"""Fail-closed selected-runtime debug/resume/tiny gate artifacts."""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from eqvae.benchmarking.io import JsonObject, write_csv, write_json
from eqvae.benchmarking.runtime_schema import GATE_HEALTH_COLUMNS
from eqvae.config import ResolvedConfig, resolve_json_config
from eqvae.data.fixed_selectors import (
    FIXED_32_TRAIN_OVERFIT_KIND,
    FixedSelectorDocument,
    load_fixed_selector_document,
    validate_fixed_selector_document,
)
from eqvae.data.patch_shards import PatchShardSpec
from eqvae.data.roots import TRAIN_BIN_NAME, TRAIN_CSV_NAME, resolve_patch_data_paths
from eqvae.data.splits import load_masked_holdout_wsi_ids
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
EXPECTED_TINY_SELECTOR_COUNT = 32
EXPECTED_REAL_TRAIN_PATCH_COUNT = 300_000
EXPECTED_REAL_TRAIN_CSV_SHA256 = (
    "8fc4959f7de006eed259f818ef2cc4ea03d1f3ec6ba483bf7229c04562f22a52"
)
EXPECTED_REAL_TRAIN_BIN_FILE_SIZE = 58_982_400_064
EXPECTED_REAL_TRAIN_HEADER_CRC32 = 1_289_496_176
FAIL_STATUS = "fail"
OK_STATUS = "pass"
REAL_UBC_SELECTED_RUNTIME_TRAIN_RUNNER_IMPLEMENTED = False
SELECTED_RUNTIME_PLAN_APPLIED_TO_TRAINING = False
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

    The current implementation intentionally fails closed because the real
    `ubc-pre-shuffled` DDP/AMP training runner is not wired yet. It still
    validates and records the selected-runtime transport, config hashes, fixed
    selector readiness, and artifact contract that the Kaggle kernel must
    preserve.

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
    selector_status = _selector_status(
        selector_path,
        data_root=request.data_root,
    )
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


def verify_selected_runtime_debug_push_ready(
    *,
    debug_config_path: Path,
    tiny_config_path: Path,
    selected_runtime_path: Path,
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
    selector_status = _selector_status(selector_path, data_root=data_root)
    blockers = [
        *_push_readiness_blockers(
            runtime_errors=runtime_errors,
            selector_status=selector_status,
        ),
        *_structured_readiness_blockers(
            debug_config_path=debug_config_path,
            tiny_config_path=tiny_config_path,
            selected_runtime_path=selected_runtime_path,
            data_root=data_root,
            fixed_train_patches=selector_path,
        ),
        *_readiness_config_blockers(
            resolved=debug_resolved,
            gate_key="selected_runtime_debug",
        ),
        *_readiness_config_blockers(
            resolved=tiny_resolved,
            gate_key="selected_runtime_debug_gate",
        ),
    ]
    return _dedupe_strings(tuple(blockers))


def _structured_readiness_blockers(
    *,
    debug_config_path: Path,
    tiny_config_path: Path,
    selected_runtime_path: Path,
    data_root: str | None,
    fixed_train_patches: Path,
) -> tuple[str, ...]:
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
            ),
        )
        readiness = _load_json(result.local_readiness)
    return _local_readiness_blockers(readiness)


def _local_readiness_blockers(readiness: JsonObject) -> tuple[str, ...]:
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
        "failure_kind": "real_ubc_selected_runtime_train_runner_not_implemented",
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
        "failure_kind": "real_ubc_selected_runtime_train_runner_not_implemented",
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


def _selector_status(path: Path, *, data_root: str | None) -> JsonObject:
    if not path.exists():
        return {
            "path": str(path),
            "sha256": "",
            "status": FAIL_STATUS,
            "selector_count": 0,
            "expected_count": EXPECTED_TINY_SELECTOR_COUNT,
            "failure_kind": "fixed_32_selector_missing",
            "validation_errors": ["fixed_32_selector_missing"],
            "canonical_real_ubc": False,
        }
    payload = _load_json(path)
    selectors = payload.get("selectors")
    selector_count = len(selectors) if isinstance(selectors, list) else 0
    errors = list(_raw_selector_errors(payload, selector_count=selector_count))
    validation_detail = ""
    if not errors:
        try:
            document = load_fixed_selector_document(path)
        except (KeyError, TypeError, ValueError) as error:
            errors.append("fixed_32_selector_schema_invalid")
            validation_detail = str(error)
        else:
            document_errors, validation_detail = _selector_document_errors(
                path=path,
                data_root=data_root,
                document=document,
            )
            errors.extend(document_errors)
    return cast(
        "JsonObject",
        {
            "path": str(path),
            "sha256": _sha256_file(path),
            "status": OK_STATUS if not errors else FAIL_STATUS,
            "selector_count": selector_count,
            "expected_count": EXPECTED_TINY_SELECTOR_COUNT,
            "failure_kind": "" if not errors else errors[0],
            "validation_errors": errors,
            "validation_detail": validation_detail,
            "canonical_real_ubc": not errors,
        },
    )


def _raw_selector_errors(
    payload: JsonObject,
    *,
    selector_count: int,
) -> tuple[str, ...]:
    errors: list[str] = []
    if payload.get("status") == "requires_real_data_generation":
        errors.append("fixed_32_selector_placeholder")
    if payload.get("selector_kind") != FIXED_32_TRAIN_OVERFIT_KIND:
        errors.append("fixed_32_selector_wrong_kind")
    if payload.get("source_split") != "train":
        errors.append("fixed_32_selector_not_train_split")
    if _selector_dataset_slug(payload) != EXPECTED_DATASET_SLUG:
        errors.append("fixed_32_selector_wrong_dataset")
    if selector_count != EXPECTED_TINY_SELECTOR_COUNT:
        errors.append("fixed_32_selector_count_not_32")
    return tuple(errors)


def _selector_dataset_slug(payload: JsonObject) -> str:
    dataset_slug = payload.get("dataset_slug")
    if isinstance(dataset_slug, str):
        return dataset_slug
    source = payload.get("source")
    if isinstance(source, dict):
        source_slug = source.get("dataset_slug")
        if isinstance(source_slug, str):
            return source_slug
    return ""


def _selector_document_errors(
    *,
    path: Path,
    data_root: str | None,
    document: FixedSelectorDocument,
) -> tuple[tuple[str, ...], str]:
    detail = ""
    errors = list(_selector_document_basic_errors(document))
    if errors:
        return tuple(errors), detail

    document_data_root = (
        None if document.source.data_root is None else str(document.source.data_root)
    )
    resolved_data_root = data_root or document_data_root or "auto"
    try:
        paths = resolve_patch_data_paths(resolved_data_root)
    except FileNotFoundError as error:
        return ("fixed_32_selector_data_unavailable",), str(error)

    holdout_path = _masked_holdout_path(
        selector_path=path,
        selector_value=document.masked_holdout_exclusion,
    )
    try:
        masked_holdout_wsi_ids = load_masked_holdout_wsi_ids(holdout_path)
    except (OSError, ValueError) as error:
        return ("fixed_32_selector_masked_holdout_unavailable",), str(error)

    train_paths = paths.for_split("train")
    shard_spec = PatchShardSpec(
        bin_path=train_paths.bin_path,
        csv_path=train_paths.csv_path,
        image_size=document.source.header.height,
        channels=document.source.header.channels,
        validate_crc=True,
    )
    try:
        validate_fixed_selector_document(
            document=document,
            shard_spec=shard_spec,
            expected_kind=FIXED_32_TRAIN_OVERFIT_KIND,
            masked_holdout_wsi_ids=masked_holdout_wsi_ids,
        )
    except (EOFError, OSError, TypeError, ValueError) as error:
        return ("fixed_32_selector_validation_failed",), str(error)
    canonical_errors = _canonical_real_ubc_selector_errors(document)
    if canonical_errors:
        return ("fixed_32_selector_not_canonical_real_ubc",), "; ".join(
            canonical_errors,
        )
    return (), detail


def _selector_document_basic_errors(
    document: FixedSelectorDocument,
) -> tuple[str, ...]:
    errors: list[str] = []
    if document.selector_kind != FIXED_32_TRAIN_OVERFIT_KIND:
        errors.append("fixed_32_selector_wrong_kind")
    if document.source_split != "train":
        errors.append("fixed_32_selector_not_train_split")
    if document.source.dataset_slug != EXPECTED_DATASET_SLUG:
        errors.append("fixed_32_selector_wrong_dataset")
    if len(document.selectors) != EXPECTED_TINY_SELECTOR_COUNT:
        errors.append("fixed_32_selector_count_not_32")
    if not document.source.crc_checked:
        errors.append("fixed_32_selector_crc_not_checked")
    return tuple(errors)


def _canonical_real_ubc_selector_errors(
    document: FixedSelectorDocument,
) -> tuple[str, ...]:
    source = document.source
    header = source.header
    checks: tuple[tuple[str, object, object], ...] = (
        ("source.dataset_slug", source.dataset_slug, EXPECTED_DATASET_SLUG),
        ("source.source_split", source.source_split, "train"),
        ("source.csv_path.name", source.csv_path.name, TRAIN_CSV_NAME),
        ("source.csv_sha256", source.csv_sha256, EXPECTED_REAL_TRAIN_CSV_SHA256),
        ("source.bin_path.name", source.bin_path.name, TRAIN_BIN_NAME),
        (
            "source.bin_file_size",
            source.bin_file_size,
            EXPECTED_REAL_TRAIN_BIN_FILE_SIZE,
        ),
        ("source.row_count", source.row_count, EXPECTED_REAL_TRAIN_PATCH_COUNT),
        ("source.patch_count", source.patch_count, EXPECTED_REAL_TRAIN_PATCH_COUNT),
        ("source.idx_policy", source.idx_policy, "row_order"),
        ("source.crc_checked", source.crc_checked, True),
        ("header.crc32", header.crc32, EXPECTED_REAL_TRAIN_HEADER_CRC32),
        ("header.patch_count", header.patch_count, EXPECTED_REAL_TRAIN_PATCH_COUNT),
        ("header.channels", header.channels, 3),
        ("header.height", header.height, 256),
        ("header.width", header.width, 256),
        ("header.version", header.version, 1),
        ("header.layout", header.layout, b"CHW"),
    )
    return tuple(
        f"{name}: expected {expected!r}, got {actual!r}"
        for name, actual, expected in checks
        if actual != expected
    )


def _masked_holdout_path(
    *,
    selector_path: Path,
    selector_value: str | None,
) -> Path:
    if selector_value is None or not selector_value:
        return _resolve_relative_to_ancestors(
            base_path=selector_path,
            relative_path=Path("docs/data/ubc_ocean_masked_holdout_ids.csv"),
        )
    configured = Path(selector_value)
    if configured.is_absolute():
        return configured
    return _resolve_relative_to_ancestors(
        base_path=selector_path,
        relative_path=configured,
    )


def _resolve_relative_to_ancestors(*, base_path: Path, relative_path: Path) -> Path:
    for parent in base_path.resolve().parents:
        candidate = parent / relative_path
        if candidate.exists():
            return candidate
    return Path.cwd() / relative_path


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
    if not REAL_UBC_SELECTED_RUNTIME_TRAIN_RUNNER_IMPLEMENTED:
        blockers.insert(0, "real_ubc_selected_runtime_train_runner_not_implemented")
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
) -> tuple[str, ...]:
    blockers: list[str] = []
    if not REAL_UBC_SELECTED_RUNTIME_TRAIN_RUNNER_IMPLEMENTED:
        blockers.append("real_ubc_selected_runtime_train_runner_not_implemented")
    if not SELECTED_RUNTIME_PLAN_APPLIED_TO_TRAINING:
        blockers.append("selected_runtime_runtime_plan_not_applied_to_training")
    if runtime_errors:
        blockers.append("selected_runtime_transport_validation_failed")
    if selector_status.get("status") != OK_STATUS:
        blockers.append(_string_value(selector_status.get("failure_kind")))
    return tuple(blocker for blocker in blockers if blocker)


def _readiness_config_blockers(
    *,
    resolved: ResolvedConfig,
    gate_key: str,
) -> tuple[str, ...]:
    gate = resolved.effective_config.get(gate_key)
    if not isinstance(gate, dict):
        return (f"{gate_key}_missing",)
    gate_payload = cast("JsonObject", gate)
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
    "SelectedRuntimeGateRequest",
    "SelectedRuntimeGateResult",
    "verify_selected_runtime_debug_push_ready",
    "write_selected_runtime_gate",
]
