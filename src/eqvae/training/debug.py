# Copyright 2026 HiperMaximus
"""Short fail-closed training proof runner for spec 0001 debug gates."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import torch

from eqvae.benchmarking.io import CsvRow, JsonObject, JsonValue, write_csv, write_json
from eqvae.checkpointing import (
    CheckpointMetadata,
    CheckpointResumeMetadata,
    LoadedCheckpoint,
    load_training_checkpoint,
    read_training_checkpoint_metadata,
    save_training_checkpoint,
    validate_checkpoint_resume_metadata,
)
from eqvae.config import ResolvedConfig, resolve_json_config
from eqvae.losses.vae import beta_for_step
from eqvae.metrics.reconstruction import reconstruction_metric_summaries
from eqvae.models.non_equivariant_vae import (
    DEFAULT_GROUPNORM_GROUPS,
    LATENT_CHANNELS,
    build_non_equivariant_vae,
)
from eqvae.training.optim import SpecAdamWConfig, create_adamw_optimizer
from eqvae.training.step import TrainStepRequest, TrainStepResult, run_train_step

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.random import Generator

    from eqvae.models.non_equivariant_vae import NonEquivariantVAE


_TRAIN_METRIC_COLUMNS = (
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
_LOCAL_STATUS = "local_pass"
_FAIL_STATUS = "fail"
_LOCAL_SCOPE = "local_synthetic_contract_real_kaggle_proof_pending"
_SUPPORTED_DATA = "synthetic"
_TINY_MODE = "kaggle_tiny_overfit"


@dataclass(frozen=True)
class DebugTrainingRequest:
    """Inputs for a short debug/tiny training proof run."""

    config_path: Path
    output_dir: Path
    run_name: str
    data: str
    runtime_config: Path | None = None
    data_root: str | None = None
    fixed_train_patches: Path | None = None
    resume: Path | None = None
    max_train_steps: int | None = None
    max_val_steps: int | None = None
    save_every_steps: int | None = None


@dataclass(frozen=True)
class DebugTrainingResult:
    """Artifact paths from a short debug/tiny training proof run."""

    output_dir: Path
    training_summary: Path
    metrics: Path
    artifact_manifest: Path
    selected_runtime_debug_summary: Path | None
    checkpoint_resume_proof: Path | None
    tiny_overfit_summary: Path | None


@dataclass(frozen=True)
class _RuntimeConfigProof:
    path: Path | None
    sha256: str
    selected_row_id: str
    runtime_policy_id: str
    launch_blockers: tuple[str, ...]
    consumed: bool
    status: str
    failure_kind: str


@dataclass(frozen=True)
class _TrainingSettings:
    run_name: str
    run_mode: str
    batch_size: int
    image_size: int
    max_train_steps: int
    max_val_steps: int
    save_every_steps: int
    ssim_weight: float
    beta_target: float
    beta_warmup_fraction: float
    optimizer_config: SpecAdamWConfig
    global_seed: int
    data_seed: int
    selected_runtime_required: bool
    fixed_train_patches: Path | None


@dataclass(frozen=True)
class _RunArtifacts:
    training_summary: Path
    train_metrics: Path
    artifact_manifest: Path
    selected_runtime_debug_summary: Path
    checkpoint_resume_proof: Path
    tiny_overfit_summary: Path
    reconstruction_samples: Path


@dataclass(frozen=True)
class _TrainingContext:
    request: DebugTrainingRequest
    resolved: ResolvedConfig
    settings: _TrainingSettings
    artifacts: _RunArtifacts
    runtime_config: _RuntimeConfigProof
    loaded_checkpoint: LoadedCheckpoint | None
    checkpoint_metadata: tuple[CheckpointMetadata, ...]
    final_checkpoint: CheckpointMetadata
    best_checkpoint: CheckpointMetadata
    metric_rows: tuple[CsvRow, ...]
    last_result: TrainStepResult
    initial_metrics: JsonObject
    final_metrics: JsonObject
    reconstruction_sample_nonblank: bool


def write_debug_training_run(  # noqa: PLR0914
    request: DebugTrainingRequest,
) -> DebugTrainingResult:
    """Run a short local training proof and write debug/tiny artifacts.

    Returns:
        Paths to the artifacts produced by the short proof run.

    Raises:
        ValueError: If resume/runtime settings are inconsistent.
        NotImplementedError: If asked to run a non-local data surface.

    """
    if request.data != _SUPPORTED_DATA:
        message = (
            "Only data='synthetic' is implemented for local proof runs; real "
            "UBC selected-runtime debug/tiny-overfit remains Kaggle-gated."
        )
        raise NotImplementedError(message)
    resolved = resolve_json_config(request.config_path)
    settings = _settings(request=request, resolved=resolved)
    runtime_config = _runtime_config_proof(
        runtime_config=request.runtime_config,
        selected_runtime_required=settings.selected_runtime_required,
    )
    resume_metadata = (
        None
        if request.resume is None
        else read_training_checkpoint_metadata(path=request.resume)
    )
    if resume_metadata is not None:
        _validate_resume_metadata(
            metadata=resume_metadata,
            resolved=resolved,
            runtime_config=runtime_config,
        )
        if resume_metadata.successful_optimizer_update_count >= (
            settings.max_train_steps
        ):
            message = (
                "resume checkpoint is already at or beyond requested "
                "max_train_steps: "
                f"{resume_metadata.successful_optimizer_update_count} >= "
                f"{settings.max_train_steps}"
            )
            raise ValueError(message)
    artifacts = _artifact_paths(request.output_dir)
    manual_seed = cast("Callable[[int], torch.Generator]", torch.manual_seed)
    manual_seed(settings.global_seed)
    numpy_generator = np.random.default_rng(settings.global_seed)
    train_data_generator = torch.Generator(device="cpu")
    train_data_generator.manual_seed(settings.data_seed)
    torch_generators = {"train_data": train_data_generator}
    model = build_non_equivariant_vae(norm_groups=_norm_groups(resolved))
    optimizer, _ = create_adamw_optimizer(
        model,
        config=settings.optimizer_config,
    )
    loaded_checkpoint = (
        None
        if request.resume is None
        else load_training_checkpoint(
            path=request.resume,
            model=model,
            optimizer=optimizer,
            numpy_generator=numpy_generator,
            torch_generators=torch_generators,
            expected_effective_config_sha256=resolved.effective_config_hash,
            expected_runtime_config_sha256=runtime_config.sha256,
            expected_selected_row_id=runtime_config.selected_row_id,
            expected_runtime_policy_id=runtime_config.runtime_policy_id,
        )
    )
    start_step = (
        0
        if loaded_checkpoint is None
        else loaded_checkpoint.successful_optimizer_update_count
    )
    if start_step >= settings.max_train_steps:
        message = (
            "resume checkpoint is already at or beyond requested max_train_steps: "
            f"{start_step} >= {settings.max_train_steps}"
        )
        raise ValueError(message)

    initial_metrics = _evaluate_model(
        model=model,
        settings=settings,
        seed_offset=10_000,
    )
    metric_rows, checkpoints, last_result = _run_steps(
        model=model,
        optimizer=optimizer,
        numpy_generator=numpy_generator,
        train_data_generator=train_data_generator,
        runtime_config=runtime_config,
        settings=settings,
        request=request,
        resolved=resolved,
        start_step=start_step,
    )
    final_metrics = _evaluate_model(
        model=model,
        settings=settings,
        seed_offset=20_000,
    )
    final_checkpoint = save_training_checkpoint(
        path=request.output_dir / "checkpoints" / "final.pt",
        model=model,
        optimizer=optimizer,
        numpy_generator=numpy_generator,
        torch_generators=torch_generators,
        runtime_config_sha256=runtime_config.sha256,
        selected_row_id=runtime_config.selected_row_id,
        runtime_policy_id=runtime_config.runtime_policy_id,
        run_name=settings.run_name,
        config_path=request.config_path,
        config_sha256=resolved.invoked_config_hash,
        effective_config_sha256=resolved.effective_config_hash,
        optimizer_step=settings.max_train_steps,
        successful_optimizer_update_count=settings.max_train_steps,
        metric_name="l1_loss",
        metric_value=_json_float(final_metrics, "l1"),
    )
    best_checkpoint = save_training_checkpoint(
        path=request.output_dir / "checkpoints" / "best_model.pt",
        model=model,
        optimizer=optimizer,
        numpy_generator=numpy_generator,
        torch_generators=torch_generators,
        runtime_config_sha256=runtime_config.sha256,
        selected_row_id=runtime_config.selected_row_id,
        runtime_policy_id=runtime_config.runtime_policy_id,
        run_name=settings.run_name,
        config_path=request.config_path,
        config_sha256=resolved.invoked_config_hash,
        effective_config_sha256=resolved.effective_config_hash,
        optimizer_step=settings.max_train_steps,
        successful_optimizer_update_count=settings.max_train_steps,
        metric_name="l1_loss",
        metric_value=_best_l1(metric_rows),
    )
    checkpoint_metadata = (*checkpoints, final_checkpoint, best_checkpoint)
    reconstruction_nonblank = _write_reconstruction_sample(
        path=artifacts.reconstruction_samples,
        model=model,
        settings=settings,
    )
    write_csv(artifacts.train_metrics, _TRAIN_METRIC_COLUMNS, metric_rows)

    context = _TrainingContext(
        request=request,
        resolved=resolved,
        settings=settings,
        artifacts=artifacts,
        runtime_config=runtime_config,
        loaded_checkpoint=loaded_checkpoint,
        checkpoint_metadata=checkpoint_metadata,
        final_checkpoint=final_checkpoint,
        best_checkpoint=best_checkpoint,
        metric_rows=metric_rows,
        last_result=last_result,
        initial_metrics=initial_metrics,
        final_metrics=final_metrics,
        reconstruction_sample_nonblank=reconstruction_nonblank,
    )
    write_json(artifacts.training_summary, _training_summary(context))
    debug_summary_path: Path | None = None
    if runtime_config.consumed:
        write_json(
            artifacts.selected_runtime_debug_summary,
            _selected_runtime_debug_summary(context),
        )
        debug_summary_path = artifacts.selected_runtime_debug_summary

    resume_proof_path: Path | None = None
    if loaded_checkpoint is not None:
        write_json(artifacts.checkpoint_resume_proof, _resume_proof(context))
        resume_proof_path = artifacts.checkpoint_resume_proof

    tiny_summary_path: Path | None = None
    if _writes_tiny_summary(settings):
        write_json(artifacts.tiny_overfit_summary, _tiny_summary(context))
        tiny_summary_path = artifacts.tiny_overfit_summary

    write_json(artifacts.artifact_manifest, _artifact_manifest(context))
    return DebugTrainingResult(
        output_dir=request.output_dir,
        training_summary=artifacts.training_summary,
        metrics=artifacts.train_metrics,
        artifact_manifest=artifacts.artifact_manifest,
        selected_runtime_debug_summary=debug_summary_path,
        checkpoint_resume_proof=resume_proof_path,
        tiny_overfit_summary=tiny_summary_path,
    )


def _run_steps(  # noqa: PLR0913
    *,
    model: NonEquivariantVAE,
    optimizer: torch.optim.Optimizer,
    numpy_generator: Generator,
    train_data_generator: torch.Generator,
    runtime_config: _RuntimeConfigProof,
    settings: _TrainingSettings,
    request: DebugTrainingRequest,
    resolved: ResolvedConfig,
    start_step: int,
) -> tuple[tuple[CsvRow, ...], tuple[CheckpointMetadata, ...], TrainStepResult]:
    rows: list[CsvRow] = []
    checkpoints: list[CheckpointMetadata] = []
    last_result: TrainStepResult | None = None
    for optimizer_step_index in range(start_step, settings.max_train_steps):
        clean_batch = _synthetic_clean_batch(
            batch_size=settings.batch_size,
            image_size=settings.image_size,
            generator=train_data_generator,
        )
        eps = _zero_eps(settings)
        beta = beta_for_step(
            optimizer_step_index=optimizer_step_index,
            max_optimizer_steps=settings.max_train_steps,
            target_beta=settings.beta_target,
            warmup_fraction=settings.beta_warmup_fraction,
        )
        result = run_train_step(
            TrainStepRequest(
                model=model,
                optimizer=optimizer,
                clean_batch=clean_batch,
                eps=eps,
                beta=beta,
                ssim_weight=settings.ssim_weight,
                optimizer_step_index=optimizer_step_index,
                gradient_clip_global_norm=(
                    settings.optimizer_config.gradient_clip_global_norm
                ),
            ),
        )
        last_result = result
        successful_count = result.successful_optimizer_update_count
        checkpoint_path = ""
        if successful_count % settings.save_every_steps == 0:
            checkpoint = _save_step_checkpoint(
                request=request,
                resolved=resolved,
                settings=settings,
                model=model,
                optimizer=optimizer,
                numpy_generator=numpy_generator,
                torch_generators={"train_data": train_data_generator},
                runtime_config=runtime_config,
                step=successful_count,
                metric_value=float(result.losses.l1_loss.detach().item()),
            )
            checkpoints.append(checkpoint)
            checkpoint_path = _relative_to_output(checkpoint.path, request.output_dir)
        rows.append(_metric_row(result=result, checkpoint_path=checkpoint_path))
    if last_result is None:
        message = "No train steps were executed"
        raise RuntimeError(message)
    if not any(
        checkpoint.successful_optimizer_update_count == settings.max_train_steps
        for checkpoint in checkpoints
    ):
        checkpoint = _save_step_checkpoint(
            request=request,
            resolved=resolved,
            settings=settings,
            model=model,
            optimizer=optimizer,
            numpy_generator=numpy_generator,
            torch_generators={"train_data": train_data_generator},
            runtime_config=runtime_config,
            step=settings.max_train_steps,
            metric_value=float(last_result.losses.l1_loss.detach().item()),
        )
        checkpoints.append(checkpoint)
        rows[-1] = {
            **dict(rows[-1]),
            "checkpoint_path": _relative_to_output(
                checkpoint.path,
                request.output_dir,
            ),
        }
    return tuple(rows), tuple(checkpoints), last_result


def _save_step_checkpoint(  # noqa: PLR0913
    *,
    request: DebugTrainingRequest,
    resolved: ResolvedConfig,
    settings: _TrainingSettings,
    model: NonEquivariantVAE,
    optimizer: torch.optim.Optimizer,
    numpy_generator: Generator,
    torch_generators: dict[str, torch.Generator],
    runtime_config: _RuntimeConfigProof,
    step: int,
    metric_value: float,
) -> CheckpointMetadata:
    return save_training_checkpoint(
        path=request.output_dir / "checkpoints" / f"step_{step:06d}.pt",
        model=model,
        optimizer=optimizer,
        numpy_generator=numpy_generator,
        torch_generators=torch_generators,
        runtime_config_sha256=runtime_config.sha256,
        selected_row_id=runtime_config.selected_row_id,
        runtime_policy_id=runtime_config.runtime_policy_id,
        run_name=settings.run_name,
        config_path=request.config_path,
        config_sha256=resolved.invoked_config_hash,
        effective_config_sha256=resolved.effective_config_hash,
        optimizer_step=step,
        successful_optimizer_update_count=step,
        metric_name="l1_loss",
        metric_value=metric_value,
    )


def _metric_row(*, result: TrainStepResult, checkpoint_path: str) -> CsvRow:
    scalars = result.losses.detached_scalars()
    return {
        "optimizer_step": str(result.successful_optimizer_update_count),
        "loss": _format_float(scalars["loss"]),
        "recon_loss": _format_float(scalars["recon_loss"]),
        "l1_loss": _format_float(scalars["l1_loss"]),
        "ssim_loss": _format_float(scalars["ssim_loss"]),
        "ssim_metric": _format_float(scalars["ssim_metric"]),
        "kl_loss": _format_float(scalars["kl_loss"]),
        "beta": _format_float(scalars["beta"]),
        "grad_norm": _format_float(result.grad_norm),
        "param_update_norm": _format_float(result.param_update_norm),
        "nonfinite_count": str(result.nonfinite_count),
        "checkpoint_path": checkpoint_path,
    }


def _training_summary(context: _TrainingContext) -> JsonObject:
    nonfinite_total = sum(int(row["nonfinite_count"]) for row in context.metric_rows)
    return {
        "status": _LOCAL_STATUS if nonfinite_total == 0 else _FAIL_STATUS,
        "proof_scope": _LOCAL_SCOPE,
        "full_run_eligible": False,
        "run_name": context.settings.run_name,
        "run_mode": context.settings.run_mode,
        "data": context.request.data,
        "data_root": context.request.data_root or "",
        "config_path": str(context.request.config_path),
        "config_sha256": context.resolved.invoked_config_hash,
        "effective_config_sha256": context.resolved.effective_config_hash,
        "runtime_config": _runtime_config_payload(context.runtime_config),
        "seeds": {
            "global_seed": context.settings.global_seed,
            "data_seed": context.settings.data_seed,
        },
        "max_train_steps": context.settings.max_train_steps,
        "max_val_steps": context.settings.max_val_steps,
        "save_every_steps": context.settings.save_every_steps,
        "optimizer_steps_completed": len(context.metric_rows),
        "amp_step_skipped_count": 0,
        "scheduler_advanced_after_amp_skip": False,
        "checkpoint_count": len(context.checkpoint_metadata),
        "final_checkpoint": _checkpoint_payload(
            context.final_checkpoint,
            context.request.output_dir,
        ),
        "best_checkpoint": _checkpoint_payload(
            context.best_checkpoint,
            context.request.output_dir,
        ),
        "metrics_csv": "metrics/train_metrics.csv",
        "initial_metrics": context.initial_metrics,
        "final_metrics": context.final_metrics,
        "last_loss": cast("JsonObject", context.last_result.losses.detached_scalars()),
        "nonfinite_count": nonfinite_total,
    }


def _selected_runtime_debug_summary(context: _TrainingContext) -> JsonObject:
    return {
        "status": _LOCAL_STATUS,
        "proof_scope": _LOCAL_SCOPE,
        "full_run_eligible": False,
        "runtime_config": _runtime_config_payload(context.runtime_config),
        "optimizer_steps_completed": len(context.metric_rows),
        "checkpoint_written": True,
        "checkpoint_resume_proof_status": (
            _LOCAL_STATUS if context.loaded_checkpoint is not None else "not_run"
        ),
        "artifact_manifest": "benchmark/artifact_manifest.json",
        "real_kaggle_debug_status": "pending_permission_gated_remote_run",
        "launch_blockers_remaining": [
            "missing_real_selected_runtime_debug_proof",
            "missing_real_checkpoint_resume_proof",
            "missing_real_tiny_overfit_proof",
        ],
    }


def _resume_proof(context: _TrainingContext) -> JsonObject:
    loaded = context.loaded_checkpoint
    if loaded is None:
        message = "resume proof requested without loaded checkpoint"
        raise RuntimeError(message)
    config_match = (
        loaded.effective_config_sha256 == context.resolved.effective_config_hash
    )
    runtime_config_match = loaded.runtime_config_sha256 == context.runtime_config.sha256
    selected_row_match = (
        loaded.selected_row_id == context.runtime_config.selected_row_id
    )
    runtime_policy_match = (
        loaded.runtime_policy_id == context.runtime_config.runtime_policy_id
    )
    if not config_match:
        message = "resume checkpoint effective config hash differs from current config"
        raise ValueError(message)
    return {
        "status": _LOCAL_STATUS,
        "proof_scope": _LOCAL_SCOPE,
        "full_run_eligible": False,
        "resume_checkpoint": str(loaded.path),
        "resume_checkpoint_sha256": _sha256_file(loaded.path),
        "loaded_schema_version": loaded.schema_version,
        "loaded_run_name": loaded.run_name,
        "loaded_runtime_config_sha256": loaded.runtime_config_sha256,
        "current_runtime_config_sha256": context.runtime_config.sha256,
        "loaded_selected_row_id": loaded.selected_row_id,
        "current_selected_row_id": context.runtime_config.selected_row_id,
        "loaded_runtime_policy_id": loaded.runtime_policy_id,
        "current_runtime_policy_id": context.runtime_config.runtime_policy_id,
        "loaded_optimizer_step": loaded.optimizer_step,
        "loaded_successful_optimizer_update_count": (
            loaded.successful_optimizer_update_count
        ),
        "final_optimizer_step": context.settings.max_train_steps,
        "additional_optimizer_steps": (
            context.settings.max_train_steps - loaded.successful_optimizer_update_count
        ),
        "optimizer_state_restored": True,
        "optimizer_state_status": "restored_by_load_state_dict",
        "model_state_restored": True,
        "model_state_status": "restored_by_load_state_dict",
        "python_rng_state_restored": True,
        "numpy_generator_state_restored": True,
        "torch_cpu_rng_state_restored": True,
        "torch_generator_states_restored": True,
        "torch_generator_names_restored": list(loaded.torch_generator_names),
        "torch_cuda_rng_state_status": "not_applicable_local_cpu",
        "lr_scheduler_state_status": "not_applicable_local_debug_no_scheduler",
        "beta_schedule_state_status": (
            "deterministic_from_successful_optimizer_update_count"
        ),
        "amp_scaler_state_status": "not_applicable_local_cpu_amp_disabled",
        "schedule_resumed_from_successful_optimizer_update_count": True,
        "config_sha256_match": config_match,
        "runtime_config_sha256_match": runtime_config_match,
        "selected_row_id_match": selected_row_match,
        "runtime_policy_id_match": runtime_policy_match,
    }


def _tiny_summary(context: _TrainingContext) -> JsonObject:
    l1_values = [_csv_float(row, "l1_loss") for row in context.metric_rows]
    recon_values = [_csv_float(row, "recon_loss") for row in context.metric_rows]
    smoothing_window = min(25, len(l1_values))
    initial_l1 = _mean(l1_values[:smoothing_window])
    final_l1 = _mean(l1_values[-smoothing_window:])
    initial_recon = _mean(recon_values[:smoothing_window])
    final_recon = _mean(recon_values[-smoothing_window:])
    fixed_train_patches = context.settings.fixed_train_patches
    return {
        "status": _LOCAL_STATUS,
        "proof_scope": _LOCAL_SCOPE,
        "full_run_eligible": False,
        "runtime_config": _runtime_config_payload(context.runtime_config),
        "runtime_config_sha256": context.runtime_config.sha256,
        "fixed_train_patches": ""
        if fixed_train_patches is None
        else str(fixed_train_patches),
        "fixed_train_patches_sha256": ""
        if fixed_train_patches is None
        else _sha256_file(fixed_train_patches),
        "patch_count": _fixed_patch_count(fixed_train_patches),
        "optimizer_steps": len(context.metric_rows),
        "smoothing_window_steps": smoothing_window,
        "corruption_strategy": "identity_clean_no_corruption",
        "eval_views": ["train_clean"],
        "initial_smoothed_l1": initial_l1,
        "final_smoothed_l1": final_l1,
        "initial_smoothed_recon_loss": initial_recon,
        "final_smoothed_recon_loss": final_recon,
        "l1_improvement_fraction": _improvement_fraction(initial_l1, final_l1),
        "recon_loss_improvement_fraction": _improvement_fraction(
            initial_recon,
            final_recon,
        ),
        "zero_head_baseline_psnr": _json_float(context.initial_metrics, "psnr"),
        "final_psnr": _json_float(context.final_metrics, "psnr"),
        "zero_head_baseline_ssim": _json_float(context.initial_metrics, "ssim"),
        "final_ssim": _json_float(context.final_metrics, "ssim"),
        "max_logvar_clamp_fraction": 0.0,
        "max_frac_x_hat_lt_minus1": 0.0,
        "max_frac_x_hat_gt_1": 0.0,
        "gate_health_status": "local_not_measured",
        "real_tiny_overfit_status": "pending_permission_gated_remote_run",
    }


def _artifact_manifest(context: _TrainingContext) -> JsonObject:
    artifacts = {
        "training_summary": context.artifacts.training_summary,
        "train_metrics": context.artifacts.train_metrics,
        "reconstruction_samples": context.artifacts.reconstruction_samples,
    }
    if context.runtime_config.consumed:
        artifacts["selected_runtime_debug_summary"] = (
            context.artifacts.selected_runtime_debug_summary
        )
    if context.loaded_checkpoint is not None:
        artifacts["checkpoint_resume_proof"] = context.artifacts.checkpoint_resume_proof
    if _writes_tiny_summary(context.settings):
        artifacts["tiny_overfit_summary"] = context.artifacts.tiny_overfit_summary
    for checkpoint in context.checkpoint_metadata:
        artifacts[f"checkpoint:{checkpoint.path.name}"] = checkpoint.path
    return {
        "status": _LOCAL_STATUS,
        "proof_scope": _LOCAL_SCOPE,
        "full_run_eligible": False,
        "artifact_hashes": cast(
            "JsonObject",
            {name: _sha256_file(path) for name, path in sorted(artifacts.items())},
        ),
        "checkpoint_count": len(context.checkpoint_metadata),
        "metric_row_count": len(context.metric_rows),
        "reconstruction_sample_nonblank": context.reconstruction_sample_nonblank,
        "missing_artifacts": [
            name for name, path in sorted(artifacts.items()) if not path.exists()
        ],
    }


def _runtime_config_proof(
    *,
    runtime_config: Path | None,
    selected_runtime_required: bool,
) -> _RuntimeConfigProof:
    if runtime_config is None:
        return _missing_runtime_config_proof(
            selected_runtime_required=selected_runtime_required,
        )
    payload = _load_json(runtime_config)
    _validate_runtime_payload(payload)
    selected_row_id = _required_str(payload, "selected_row_id")
    runtime_policy_id = _required_str(payload, "runtime_policy_id")
    _validate_selected_runtime_snapshot(
        payload=payload,
        selected_row_id=selected_row_id,
        runtime_policy_id=runtime_policy_id,
    )
    blockers = _str_tuple(payload.get("launch_blockers"))
    return _RuntimeConfigProof(
        path=runtime_config,
        sha256=_sha256_file(runtime_config),
        selected_row_id=selected_row_id,
        runtime_policy_id=runtime_policy_id,
        launch_blockers=blockers,
        consumed=True,
        status=_LOCAL_STATUS,
        failure_kind="",
    )


def _missing_runtime_config_proof(
    *,
    selected_runtime_required: bool,
) -> _RuntimeConfigProof:
    if selected_runtime_required:
        message = "config requires --runtime-config"
        raise ValueError(message)
    return _RuntimeConfigProof(
        path=None,
        sha256="",
        selected_row_id="",
        runtime_policy_id="",
        launch_blockers=(),
        consumed=False,
        status="not_required",
        failure_kind="",
    )


def _validate_runtime_payload(payload: JsonObject) -> None:
    if payload.get("status") != "pass":
        message = "runtime config status must be pass"
        raise ValueError(message)
    if payload.get("benchmark_kind") != "kaggle_runtime_selection":
        message = "runtime config benchmark_kind must be kaggle_runtime_selection"
        raise ValueError(message)
    if payload.get("benchmark_source") != "kaggle_runtime_benchmark":
        message = "runtime config benchmark_source must be kaggle_runtime_benchmark"
        raise ValueError(message)
    if payload.get("full_run_eligible") is not True:
        message = "runtime config must be full_run_eligible"
        raise ValueError(message)
    if payload.get("full_training_launch_ready") is not False:
        message = "runtime config must not already claim full training launch ready"
        raise ValueError(message)
    _validate_runtime_safety(_required_object(payload, "safety"))


def _validate_runtime_safety(safety: JsonObject) -> None:
    for key in (
        "dataloader_status",
        "numerical_check_status",
        "corruption_check_status",
        "gate_health_status",
    ):
        if safety.get(key) != "pass":
            message = f"runtime config safety.{key} must be pass"
            raise ValueError(message)


def _validate_selected_runtime_snapshot(
    *,
    payload: JsonObject,
    selected_row_id: str,
    runtime_policy_id: str,
) -> None:
    selected_snapshot = _required_object(payload, "selected_row_snapshot")
    if selected_snapshot.get("row_id") != selected_row_id:
        message = "runtime config selected_row_snapshot.row_id mismatch"
        raise ValueError(message)
    if selected_snapshot.get("runtime_policy_id") != runtime_policy_id:
        message = "runtime config selected_row_snapshot.runtime_policy_id mismatch"
        raise ValueError(message)
    if selected_snapshot.get("status") != "pass":
        message = "runtime config selected_row_snapshot.status must be pass"
        raise ValueError(message)


def _runtime_config_payload(runtime_config: _RuntimeConfigProof) -> JsonObject:
    return {
        "path": "" if runtime_config.path is None else str(runtime_config.path),
        "sha256": runtime_config.sha256,
        "selected_row_id": runtime_config.selected_row_id,
        "runtime_policy_id": runtime_config.runtime_policy_id,
        "launch_blockers": list(runtime_config.launch_blockers),
        "consumed": runtime_config.consumed,
        "status": runtime_config.status,
        "failure_kind": runtime_config.failure_kind,
    }


def _validate_resume_metadata(
    *,
    metadata: CheckpointResumeMetadata,
    resolved: ResolvedConfig,
    runtime_config: _RuntimeConfigProof,
) -> None:
    validate_checkpoint_resume_metadata(
        metadata,
        expected_effective_config_sha256=resolved.effective_config_hash,
        expected_runtime_config_sha256=runtime_config.sha256,
        expected_selected_row_id=runtime_config.selected_row_id,
        expected_runtime_policy_id=runtime_config.runtime_policy_id,
    )


def _settings(
    *,
    request: DebugTrainingRequest,
    resolved: ResolvedConfig,
) -> _TrainingSettings:
    effective = resolved.effective_config
    run = _required_object(effective, "run")
    data = _required_object(effective, "data")
    runtime = _optional_object(effective, "runtime") or {}
    training = _optional_object(effective, "training") or runtime
    objective = _required_object(effective, "objective")
    beta = _required_object(objective, "beta")
    fixed_train_patches = request.fixed_train_patches or _optional_path(
        data,
        "fixed_train_patches",
        config_path=request.config_path,
    )
    settings = _TrainingSettings(
        run_name=request.run_name or _required_str(run, "name"),
        run_mode=_required_str(run, "mode"),
        batch_size=_first_int(
            _optional_int(runtime, "batch_size"),
            default=2,
        ),
        image_size=_first_int(
            _optional_int(data, "image_size"),
            default=256,
        ),
        max_train_steps=request.max_train_steps
        if request.max_train_steps is not None
        else _first_int(_optional_int(training, "max_train_steps"), default=1),
        max_val_steps=request.max_val_steps
        if request.max_val_steps is not None
        else _first_int(_optional_int(training, "max_val_steps"), default=0),
        save_every_steps=request.save_every_steps
        if request.save_every_steps is not None
        else _first_int(_optional_int(training, "save_every_steps"), default=1),
        ssim_weight=_required_float(objective, "ssim_weight"),
        beta_target=_required_float(beta, "target"),
        beta_warmup_fraction=_required_float(beta, "step_limited_warmup_fraction"),
        optimizer_config=_optimizer_config(effective),
        global_seed=_seed(effective, "global_seed"),
        data_seed=_seed(effective, "data_seed"),
        selected_runtime_required=_optional_bool(
            runtime,
            "selected_runtime_required",
        ),
        fixed_train_patches=fixed_train_patches,
    )
    _validate_settings(settings)
    return settings


def _first_int(value: int | None, *, default: int) -> int:
    return default if value is None else value


def _validate_settings(settings: _TrainingSettings) -> None:
    if settings.batch_size <= 0:
        message = f"batch_size must be positive, got {settings.batch_size}"
        raise ValueError(message)
    if settings.image_size <= 0 or settings.image_size % 8 != 0:
        message = "image_size must be positive and divisible by 8"
        raise ValueError(message)
    if settings.max_train_steps <= 0:
        message = f"max_train_steps must be positive, got {settings.max_train_steps}"
        raise ValueError(message)
    if settings.max_val_steps < 0:
        message = f"max_val_steps must be nonnegative, got {settings.max_val_steps}"
        raise ValueError(message)
    if settings.save_every_steps <= 0:
        message = f"save_every_steps must be positive, got {settings.save_every_steps}"
        raise ValueError(message)


def _optimizer_config(effective_config: JsonObject) -> SpecAdamWConfig:
    optimizer = _required_object(effective_config, "optimizer")
    betas = _required_float_list(optimizer, "betas", expected_len=2)
    return SpecAdamWConfig(
        learning_rate=_required_float(optimizer, "learning_rate"),
        beta1=betas[0],
        beta2=betas[1],
        epsilon=_required_float(optimizer, "epsilon"),
        weight_decay=_required_float(optimizer, "weight_decay"),
        gradient_clip_global_norm=_required_float(
            optimizer,
            "gradient_clip_global_norm",
        ),
        gate_lr_multiplier=_required_float(
            _required_object(
                _required_object(optimizer, "parameter_groups"),
                "gate_no_decay",
            ),
            "lr_multiplier",
        ),
    )


def _artifact_paths(output_dir: Path) -> _RunArtifacts:
    benchmark_dir = output_dir / "benchmark"
    return _RunArtifacts(
        training_summary=benchmark_dir / "training_summary.json",
        train_metrics=output_dir / "metrics" / "train_metrics.csv",
        artifact_manifest=benchmark_dir / "artifact_manifest.json",
        selected_runtime_debug_summary=(
            benchmark_dir / "selected_runtime_debug_summary.json"
        ),
        checkpoint_resume_proof=benchmark_dir / "checkpoint_resume_proof.json",
        tiny_overfit_summary=benchmark_dir / "tiny_overfit_summary.json",
        reconstruction_samples=output_dir / "artifacts" / "reconstruction_samples.pt",
    )


def _evaluate_model(
    *,
    model: NonEquivariantVAE,
    settings: _TrainingSettings,
    seed_offset: int,
) -> JsonObject:
    clean_batch = _synthetic_clean_batch(
        batch_size=settings.batch_size,
        image_size=settings.image_size,
        generator=_seeded_torch_generator(settings.data_seed + seed_offset),
    )
    eps = _zero_eps(settings)
    with torch.no_grad():
        output = model.forward(clean_batch, eps=eps)
    metrics = reconstruction_metric_summaries(output.reconstruction, clean_batch)
    return {
        "l1": float((output.reconstruction - clean_batch).abs().mean().item()),
        "psnr": _metric_mean(metrics["psnr_img"].as_dict()),
        "ssim": _metric_mean(metrics["ssim_img"].as_dict()),
    }


def _write_reconstruction_sample(
    *,
    path: Path,
    model: NonEquivariantVAE,
    settings: _TrainingSettings,
) -> bool:
    clean_batch = _synthetic_clean_batch(
        batch_size=1,
        image_size=settings.image_size,
        generator=_seeded_torch_generator(settings.data_seed + 30_000),
    )
    eps = torch.zeros(
        (
            1,
            LATENT_CHANNELS,
            settings.image_size // 8,
            settings.image_size // 8,
        ),
    )
    with torch.no_grad():
        output = model.forward(clean_batch, eps=eps)
    payload = {
        "target": clean_batch.detach().cpu(),
        "reconstruction": output.reconstruction.detach().cpu(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return bool(clean_batch.numel() > 0 and torch.isfinite(output.reconstruction).all())


def _synthetic_clean_batch(
    *,
    batch_size: int,
    image_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    batch = torch.randint(
        low=0,
        high=256,
        size=(batch_size, 3, image_size, image_size),
        generator=generator,
        dtype=torch.uint8,
    )
    return batch.to(dtype=torch.float32).div(127.5).sub(1.0)


def _seeded_torch_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def _zero_eps(settings: _TrainingSettings) -> torch.Tensor:
    return torch.zeros(
        (
            settings.batch_size,
            LATENT_CHANNELS,
            settings.image_size // 8,
            settings.image_size // 8,
        ),
        dtype=torch.float32,
    )


def _checkpoint_payload(
    checkpoint: CheckpointMetadata,
    output_dir: Path,
) -> JsonObject:
    return {
        "path": _relative_to_output(checkpoint.path, output_dir),
        "sha256": checkpoint.sha256,
        "optimizer_step": checkpoint.optimizer_step,
        "successful_optimizer_update_count": (
            checkpoint.successful_optimizer_update_count
        ),
    }


def _relative_to_output(path: Path, output_dir: Path) -> str:
    try:
        return str(path.relative_to(output_dir))
    except ValueError:
        return str(path)


def _writes_tiny_summary(settings: _TrainingSettings) -> bool:
    return settings.run_mode == _TINY_MODE or settings.fixed_train_patches is not None


def _fixed_patch_count(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    payload = _load_json(path)
    expected = payload.get("expected_count")
    if isinstance(expected, int) and not isinstance(expected, bool):
        return expected
    selectors = payload.get("selectors")
    if isinstance(selectors, list):
        return len(selectors)
    return 0


def _load_json(path: Path) -> JsonObject:
    payload = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(payload, dict):
        message = f"Expected JSON object in {path}"
        raise TypeError(message)
    return cast("JsonObject", payload)


def _norm_groups(resolved: ResolvedConfig) -> int:
    model = _required_object(resolved.effective_config, "model")
    normalization = _required_object(model, "normalization")
    return _optional_int(normalization, "num_groups") or DEFAULT_GROUPNORM_GROUPS


def _seed(effective: JsonObject, name: str) -> int:
    seeds = _optional_object(effective, "seeds") or {}
    value = _optional_int(seeds, name)
    return 20260610 if value is None else value


def _best_l1(rows: Sequence[CsvRow]) -> float:
    return min(_csv_float(row, "l1_loss") for row in rows)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _improvement_fraction(initial: float, final: float) -> float:
    if initial <= 0.0 or not math.isfinite(initial) or not math.isfinite(final):
        return 0.0
    return (initial - final) / initial


def _csv_float(row: CsvRow, key: str) -> float:
    return float(row[key])


def _json_float(payload: JsonObject, key: str) -> float:
    value = payload[key]
    if isinstance(value, bool):
        raise TypeError(key)
    if isinstance(value, int | float):
        return float(value)
    raise TypeError(key)


def _metric_mean(payload: dict[str, int | float | None]) -> float:
    value = payload.get("mean")
    if value is None:
        finite_mean = payload.get("finite_mean")
        if isinstance(finite_mean, int | float):
            return float(finite_mean)
        return float("inf")
    return float(value)


def _format_float(value: float) -> str:
    return f"{value:.10g}"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _required_object(payload: JsonObject, key: str) -> JsonObject:
    value = payload.get(key)
    if isinstance(value, dict):
        return cast("JsonObject", value)
    message = f"Expected object config field {key}"
    raise TypeError(message)


def _optional_object(payload: JsonObject, key: str) -> JsonObject | None:
    value = payload.get(key)
    if value is None or isinstance(value, dict):
        return cast("JsonObject | None", value)
    message = f"Expected optional object config field {key}"
    raise TypeError(message)


def _required_str(payload: JsonObject, key: str) -> str:
    value = payload.get(key)
    if isinstance(value, str):
        return value
    message = f"Expected string config field {key}"
    raise TypeError(message)


def _required_float(payload: JsonObject, key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool):
        raise TypeError(key)
    if isinstance(value, int | float):
        return float(value)
    message = f"Expected numeric config field {key}"
    raise TypeError(message)


def _required_float_list(
    payload: JsonObject,
    key: str,
    *,
    expected_len: int,
) -> tuple[float, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or len(value) != expected_len:
        message = f"Expected list field {key} of length {expected_len}"
        raise TypeError(message)
    return tuple(_json_number(item, key) for item in value)


def _json_number(value: JsonValue, key: str) -> float:
    if isinstance(value, bool):
        raise TypeError(key)
    if isinstance(value, int | float):
        return float(value)
    raise TypeError(key)


def _optional_int(payload: JsonObject, key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(key)
    if isinstance(value, int):
        return value
    message = f"Expected integer config field {key}"
    raise TypeError(message)


def _optional_bool(payload: JsonObject, key: str) -> bool:
    value = payload.get(key)
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    message = f"Expected boolean config field {key}"
    raise TypeError(message)


def _optional_path(
    payload: JsonObject,
    key: str,
    *,
    config_path: Path,
) -> Path | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        message = f"Expected path string config field {key}"
        raise TypeError(message)
    path = Path(value)
    if path.is_absolute():
        return path
    for parent in config_path.resolve().parents:
        candidate = parent / path
        if candidate.exists():
            return candidate
    return Path.cwd() / path


def _str_tuple(value: JsonValue | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        message = "Expected list of strings"
        raise TypeError(message)
    return tuple(cast("list[str]", value))


__all__ = ["DebugTrainingRequest", "DebugTrainingResult", "write_debug_training_run"]
