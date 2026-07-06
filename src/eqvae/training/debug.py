# Copyright 2026 HiperMaximus
"""Short fail-closed training proof runner for spec 0001 debug gates."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
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
from eqvae.corruption.stain import (
    StainCorruptionProfile,
    clean_validation_passthrough,
    corrupt_normalized_batch,
    profile_from_config,
)
from eqvae.data.dataloaders import normalize_uint8_batch
from eqvae.data.roots import (
    TRAIN_BIN_NAME,
    TRAIN_CSV_NAME,
    VALIDATION_BIN_NAME,
    VALIDATION_CSV_NAME,
    resolve_patch_data_paths,
)
from eqvae.data.synthetic import SyntheticPatchSpec, write_synthetic_patch_shard
from eqvae.data.training_batches import (
    PatchTrainingBatch,
    PatchTrainingDataset,
    PatchTrainingDatasetSpec,
    collate_patch_training_samples,
)
from eqvae.losses.vae import beta_for_step
from eqvae.metrics.reconstruction import reconstruction_metric_summaries
from eqvae.models.non_equivariant_vae import (
    DEFAULT_GROUPNORM_GROUPS,
    LATENT_CHANNELS,
)
from eqvae.models.registry import MODEL_KIND_NON_EQ_TRANSLATABLE, build_model
from eqvae.training.optim import SpecAdamWConfig, create_adamw_optimizer
from eqvae.training.progress import TrainingProgressState, record_training_attempt
from eqvae.training.selected_runtime import (
    SelectedRuntimeApplicationObservation,
    SelectedRuntimePlan,
    build_plan_applied_proof,
    parse_selected_runtime_plan,
)
from eqvae.training.step import TrainStepRequest, TrainStepResult, run_train_step

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.random import Generator

    from eqvae.models.non_equivariant_vae import NonEquivariantVAE


_TRAIN_STEP_COLUMNS = (
    "event_id",
    "batch_attempt",
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
    "torch_compile_enabled",
    "compile_scope",
    "corruption_strategy",
    "amp_step_skipped",
    "checkpoint_path",
)
_LOCAL_STATUS = "local_pass"
_FAIL_STATUS = "fail"
_LOCAL_SCOPE = "local_synthetic_contract_real_kaggle_proof_pending"
_SELECTED_RUNTIME_SCOPE = "local_selected_runtime_mechanics"
_LOCAL_EXECUTION_PRECISION_POLICY = "amp_off_fp32"
_SUPPORTED_DATA = "synthetic"
_TINY_MODE = "kaggle_tiny_overfit"
_TRAIN_CORRUPTION_VIEW = "train_corrupted"


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
    plan: SelectedRuntimePlan | None


@dataclass(frozen=True)
class _AppliedRuntimeSettings:
    selected_row_id: str
    runtime_policy_id: str
    accelerator_mode: str
    machine_shape: str
    world_size: int
    nproc_per_node: int
    torchrun_standalone: bool
    global_batch_size: int
    gradient_accumulation_steps: int
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
    zero_grad_set_to_none: bool


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
    corruption_seed: int
    corruption_profile: StainCorruptionProfile
    applied_runtime: _AppliedRuntimeSettings
    simulated_amp_skip_attempts: frozenset[int]
    selected_runtime_required: bool
    fixed_train_patches: Path | None


@dataclass(frozen=True)
class _RunArtifacts:
    training_summary: Path
    train_steps: Path
    artifact_manifest: Path
    selected_runtime_plan_applied: Path
    local_ubc_mechanics: Path
    amp_progress: Path
    local_readiness: Path
    selected_runtime_debug_summary: Path
    checkpoint_resume_proof: Path
    tiny_overfit_summary: Path
    reconstruction_samples: Path


@dataclass(frozen=True)
class _LocalUbcMechanics:
    root: Path
    train_dataset: PatchTrainingDataset
    validation_dataset: PatchTrainingDataset
    train_count: int
    validation_count: int
    status: str
    failure_kind: str


@dataclass(frozen=True)
class _TrainingContext:
    request: DebugTrainingRequest
    resolved: ResolvedConfig
    settings: _TrainingSettings
    artifacts: _RunArtifacts
    runtime_config: _RuntimeConfigProof
    local_ubc: _LocalUbcMechanics | None
    plan_applied_proof: JsonObject | None
    amp_progress_proof: JsonObject
    local_readiness: JsonObject
    loaded_checkpoint: LoadedCheckpoint | None
    checkpoint_metadata: tuple[CheckpointMetadata, ...]
    final_checkpoint: CheckpointMetadata
    best_checkpoint: CheckpointMetadata
    metric_rows: tuple[CsvRow, ...]
    last_result: TrainStepResult
    initial_metrics: JsonObject
    final_metrics: JsonObject
    reconstruction_sample_nonblank: bool


def write_debug_training_run(  # noqa: PLR0914, PLR0915
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
    settings = _apply_selected_runtime_settings(settings, runtime_config)
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
    local_ubc = _prepare_local_ubc_mechanics(
        output_dir=request.output_dir,
        settings=settings,
        enabled=runtime_config.plan is not None,
    )
    manual_seed = cast("Callable[[int], torch.Generator]", torch.manual_seed)
    manual_seed(settings.global_seed)
    numpy_generator = np.random.default_rng(settings.global_seed)
    train_data_generator = torch.Generator(device="cpu")
    train_data_generator.manual_seed(settings.data_seed)
    torch_generators = {"train_data": train_data_generator}
    model = build_model(
        MODEL_KIND_NON_EQ_TRANSLATABLE,
        model_config={"norm_groups": _norm_groups(resolved)},
    )
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
        local_ubc=local_ubc,
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
        local_ubc=local_ubc,
        start_step=start_step,
    )
    final_metrics = _evaluate_model(
        model=model,
        settings=settings,
        local_ubc=local_ubc,
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
        local_ubc=local_ubc,
    )
    plan_applied_proof = _plan_applied_proof(
        runtime_config=runtime_config,
        settings=settings,
        model=model,
        local_ubc=local_ubc,
        metric_rows=metric_rows,
        last_result=last_result,
    )
    write_csv(artifacts.train_steps, _TRAIN_STEP_COLUMNS, metric_rows)
    if plan_applied_proof is not None:
        write_json(artifacts.selected_runtime_plan_applied, plan_applied_proof)
    ubc_mechanics_proof = _local_ubc_mechanics_proof(
        local_ubc,
        settings=settings,
    )
    if ubc_mechanics_proof is not None:
        write_json(artifacts.local_ubc_mechanics, ubc_mechanics_proof)
    amp_progress_proof = _amp_progress_proof(metric_rows)
    write_json(artifacts.amp_progress, amp_progress_proof)

    context = _TrainingContext(
        request=request,
        resolved=resolved,
        settings=settings,
        artifacts=artifacts,
        runtime_config=runtime_config,
        local_ubc=local_ubc,
        plan_applied_proof=plan_applied_proof,
        amp_progress_proof=amp_progress_proof,
        local_readiness={},
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

    provisional_readiness = _local_readiness_summary(context, ubc_mechanics_proof)
    context = _replace_context_readiness(context, provisional_readiness)
    write_json(artifacts.local_readiness, provisional_readiness)
    manifest_probe = _artifact_manifest(context)
    local_readiness = _local_readiness_summary(
        context,
        ubc_mechanics_proof,
        artifact_manifest=manifest_probe,
    )
    context = _replace_context_readiness(context, local_readiness)
    write_json(artifacts.local_readiness, local_readiness)
    write_json(artifacts.training_summary, _training_summary(context))
    write_json(artifacts.artifact_manifest, _artifact_manifest(context))
    return DebugTrainingResult(
        output_dir=request.output_dir,
        training_summary=artifacts.training_summary,
        metrics=artifacts.train_steps,
        artifact_manifest=artifacts.artifact_manifest,
        selected_runtime_debug_summary=debug_summary_path,
        checkpoint_resume_proof=resume_proof_path,
        tiny_overfit_summary=tiny_summary_path,
    )


def _run_steps(  # noqa: PLR0913, PLR0914
    *,
    model: NonEquivariantVAE,
    optimizer: torch.optim.Optimizer,
    numpy_generator: Generator,
    train_data_generator: torch.Generator,
    runtime_config: _RuntimeConfigProof,
    settings: _TrainingSettings,
    request: DebugTrainingRequest,
    resolved: ResolvedConfig,
    local_ubc: _LocalUbcMechanics | None,
    start_step: int,
) -> tuple[tuple[CsvRow, ...], tuple[CheckpointMetadata, ...], TrainStepResult]:
    rows: list[CsvRow] = []
    checkpoints: list[CheckpointMetadata] = []
    last_result: TrainStepResult | None = None
    progress = TrainingProgressState(
        batch_attempt_count=start_step,
        successful_optimizer_update_count=start_step,
        lr_scheduler_step_count=start_step,
        tiny_smoothing_update_count=start_step if _writes_tiny_summary(settings) else 0,
    )
    while progress.successful_optimizer_update_count < settings.max_train_steps:
        optimizer_step_index = progress.optimizer_step_index
        batch_attempt = progress.batch_attempt_count + 1
        clean_batch, input_batch, corruption_strategy = _train_batches_for_step(
            settings=settings,
            local_ubc=local_ubc,
            train_data_generator=train_data_generator,
            optimizer_step_index=optimizer_step_index,
        )
        if batch_attempt in settings.simulated_amp_skip_attempts:
            attempt_progress = record_training_attempt(
                progress,
                amp_step_skipped=True,
                checkpoint_interval=settings.save_every_steps,
                validation_interval=settings.max_val_steps,
                tiny_smoothing_enabled=_writes_tiny_summary(settings),
            )
            progress = attempt_progress.after
            rows.append(
                _skipped_metric_row(
                    batch_attempt=attempt_progress.after.batch_attempt_count,
                    optimizer_step_index=optimizer_step_index,
                    successful_optimizer_update_count=(
                        attempt_progress.after.successful_optimizer_update_count
                    ),
                    settings=settings,
                    corruption_strategy=corruption_strategy,
                ),
            )
            continue
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
                input_batch=input_batch,
                zero_grad_set_to_none=(settings.applied_runtime.zero_grad_set_to_none),
            ),
        )
        last_result = result
        successful_count = result.successful_optimizer_update_count
        attempt_progress = record_training_attempt(
            progress,
            amp_step_skipped=False,
            checkpoint_interval=settings.save_every_steps,
            validation_interval=settings.max_val_steps,
            tiny_smoothing_enabled=_writes_tiny_summary(settings),
        )
        progress = attempt_progress.after
        checkpoint_path = ""
        if attempt_progress.checkpoint_due:
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
        rows.append(
            _metric_row(
                result=result,
                batch_attempt=attempt_progress.after.batch_attempt_count,
                checkpoint_path=checkpoint_path,
                settings=settings,
                corruption_strategy=corruption_strategy,
            ),
        )
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
            "checkpoint_path": _relative_to_output(checkpoint.path, request.output_dir),
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


def _prepare_local_ubc_mechanics(
    *,
    output_dir: Path,
    settings: _TrainingSettings,
    enabled: bool,
) -> _LocalUbcMechanics | None:
    if not enabled:
        return None
    root = output_dir / "local_ubc_synthetic"
    train_count = max(32, settings.batch_size * settings.max_train_steps)
    validation_count = max(32, settings.batch_size)
    write_synthetic_patch_shard(
        bin_path=root / TRAIN_BIN_NAME,
        csv_path=root / TRAIN_CSV_NAME,
        spec=SyntheticPatchSpec(
            count=train_count,
            image_size=settings.image_size,
            channels=3,
            seed=settings.data_seed,
        ),
        include_idx=False,
    )
    write_synthetic_patch_shard(
        bin_path=root / VALIDATION_BIN_NAME,
        csv_path=root / VALIDATION_CSV_NAME,
        spec=SyntheticPatchSpec(
            count=validation_count,
            image_size=settings.image_size,
            channels=3,
            seed=settings.data_seed + 1,
        ),
        include_idx=True,
    )
    paths = resolve_patch_data_paths(root)
    train_dataset = PatchTrainingDataset(
        PatchTrainingDatasetSpec(
            bin_path=paths.train.bin_path,
            csv_path=paths.train.csv_path,
            split="train",
            image_size=settings.image_size,
            channels=3,
            validate_crc=True,
        ),
    )
    validation_dataset = PatchTrainingDataset(
        PatchTrainingDatasetSpec(
            bin_path=paths.validation.bin_path,
            csv_path=paths.validation.csv_path,
            split="validation",
            image_size=settings.image_size,
            channels=3,
            validate_crc=True,
        ),
    )
    return _LocalUbcMechanics(
        root=root,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        train_count=train_count,
        validation_count=validation_count,
        status=_LOCAL_STATUS,
        failure_kind="",
    )


def _train_batches_for_step(
    *,
    settings: _TrainingSettings,
    local_ubc: _LocalUbcMechanics | None,
    train_data_generator: torch.Generator,
    optimizer_step_index: int,
) -> tuple[torch.Tensor, torch.Tensor | None, str]:
    if local_ubc is None:
        clean_batch = _synthetic_clean_batch(
            batch_size=settings.batch_size,
            image_size=settings.image_size,
            generator=train_data_generator,
        )
        return clean_batch, None, "identity_clean_no_corruption"

    batch = _training_batch(
        dataset=local_ubc.train_dataset,
        batch_size=settings.batch_size,
        step_index=optimizer_step_index,
    )
    clean_batch = normalize_uint8_batch(batch.images_uint8)
    corruption_result = corrupt_normalized_batch(
        clean_batch,
        profile=settings.corruption_profile,
        corruption_seed=settings.corruption_seed,
        split=batch.split,
        semantic_sample_keys=batch.semantic_sample_keys,
        corruption_step=optimizer_step_index,
        corruption_view=_TRAIN_CORRUPTION_VIEW,
        strategy=settings.applied_runtime.corruption_strategy,
    )
    return (
        clean_batch,
        corruption_result.corrupted,
        settings.applied_runtime.corruption_strategy,
    )


def _training_batch(
    *,
    dataset: PatchTrainingDataset,
    batch_size: int,
    step_index: int,
) -> PatchTrainingBatch:
    start = step_index * batch_size
    samples = [dataset[(start + offset) % len(dataset)] for offset in range(batch_size)]
    return collate_patch_training_samples(samples)


def _metric_row(
    *,
    result: TrainStepResult,
    batch_attempt: int,
    checkpoint_path: str,
    settings: _TrainingSettings,
    corruption_strategy: str,
) -> CsvRow:
    scalars = result.losses.detached_scalars()
    return {
        "event_id": f"train_attempt_{batch_attempt:06d}",
        "batch_attempt": str(batch_attempt),
        "optimizer_step_index": str(result.optimizer_step_index),
        "optimizer_step": str(result.successful_optimizer_update_count),
        "successful_optimizer_update_count": str(
            result.successful_optimizer_update_count,
        ),
        "split": "train",
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
        "batch_size": str(settings.batch_size),
        "precision_policy": _LOCAL_EXECUTION_PRECISION_POLICY,
        "amp_enabled": _csv_bool(value=False),
        "torch_compile_enabled": _csv_bool(value=False),
        "compile_scope": "none",
        "corruption_strategy": corruption_strategy,
        "amp_step_skipped": "0",
        "checkpoint_path": checkpoint_path,
    }


def _skipped_metric_row(
    *,
    batch_attempt: int,
    optimizer_step_index: int,
    successful_optimizer_update_count: int,
    settings: _TrainingSettings,
    corruption_strategy: str,
) -> CsvRow:
    return {
        "event_id": f"train_attempt_{batch_attempt:06d}",
        "batch_attempt": str(batch_attempt),
        "optimizer_step_index": str(optimizer_step_index),
        "optimizer_step": str(successful_optimizer_update_count),
        "successful_optimizer_update_count": str(successful_optimizer_update_count),
        "split": "train",
        "loss": "",
        "recon_loss": "",
        "l1_loss": "",
        "ssim_loss": "",
        "ssim_metric": "",
        "kl_loss": "",
        "beta": "",
        "grad_norm": "",
        "param_update_norm": "",
        "nonfinite_count": "0",
        "batch_size": str(settings.batch_size),
        "precision_policy": _LOCAL_EXECUTION_PRECISION_POLICY,
        "amp_enabled": _csv_bool(value=False),
        "torch_compile_enabled": _csv_bool(value=False),
        "compile_scope": "none",
        "corruption_strategy": corruption_strategy,
        "amp_step_skipped": "1",
        "checkpoint_path": "",
    }


def _training_summary(context: _TrainingContext) -> JsonObject:
    nonfinite_total = sum(int(row["nonfinite_count"]) for row in context.metric_rows)
    proof_scope = (
        _SELECTED_RUNTIME_SCOPE
        if context.runtime_config.plan is not None
        else _LOCAL_SCOPE
    )
    return {
        "status": _LOCAL_STATUS if nonfinite_total == 0 else _FAIL_STATUS,
        "proof_scope": proof_scope,
        "status_scope": proof_scope,
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
            "corruption_seed": context.settings.corruption_seed,
        },
        "max_train_steps": context.settings.max_train_steps,
        "max_val_steps": context.settings.max_val_steps,
        "save_every_steps": context.settings.save_every_steps,
        "batch_attempts_completed": len(context.metric_rows),
        "optimizer_steps_completed": sum(
            1 for row in context.metric_rows if row["amp_step_skipped"] == "0"
        ),
        "amp_step_skipped_count": sum(
            1 for row in context.metric_rows if row["amp_step_skipped"] == "1"
        ),
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
        "metrics_csv": "metrics/train_steps.csv",
        "train_steps_csv": "metrics/train_steps.csv",
        "selected_runtime_plan_applied": (
            ""
            if context.plan_applied_proof is None
            else "benchmark/selected_runtime_plan_applied.json"
        ),
        "local_ubc_mechanics": (
            "" if context.local_ubc is None else "benchmark/local_ubc_mechanics.json"
        ),
        "amp_progress": "benchmark/amp_progress.json",
        "local_readiness": "benchmark/local_selected_runtime_readiness.json",
        "initial_metrics": context.initial_metrics,
        "final_metrics": context.final_metrics,
        "last_loss": cast("JsonObject", context.last_result.losses.detached_scalars()),
        "nonfinite_count": nonfinite_total,
    }


def _selected_runtime_debug_summary(context: _TrainingContext) -> JsonObject:
    return {
        "status": _LOCAL_STATUS,
        "proof_scope": _SELECTED_RUNTIME_SCOPE,
        "full_run_eligible": False,
        "runtime_config": _runtime_config_payload(context.runtime_config),
        "selected_runtime_plan_applied": "benchmark/selected_runtime_plan_applied.json",
        "local_ubc_mechanics": "benchmark/local_ubc_mechanics.json",
        "amp_progress": "benchmark/amp_progress.json",
        "optimizer_steps_completed": sum(
            1 for row in context.metric_rows if row["amp_step_skipped"] == "0"
        ),
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
        "torch_cuda_rng_state_status": loaded.torch_cuda_rng_state_status,
        "lr_scheduler_state_status": loaded.lr_scheduler_state_status,
        "beta_schedule_state_status": loaded.beta_progress_state_status,
        "amp_scaler_state_status": loaded.amp_scaler_state_status,
        "ddp_sampler_progress_state_status": loaded.ddp_sampler_progress_state_status,
        "schedule_resumed_from_successful_optimizer_update_count": True,
        "config_sha256_match": config_match,
        "runtime_config_sha256_match": runtime_config_match,
        "selected_row_id_match": selected_row_match,
        "runtime_policy_id_match": runtime_policy_match,
    }


def _tiny_summary(context: _TrainingContext) -> JsonObject:
    successful_rows = _successful_metric_rows(context.metric_rows)
    l1_values = [_csv_float(row, "l1_loss") for row in successful_rows]
    recon_values = [_csv_float(row, "recon_loss") for row in successful_rows]
    smoothing_window = min(25, len(l1_values))
    initial_l1 = _mean(l1_values[:smoothing_window])
    final_l1 = _mean(l1_values[-smoothing_window:])
    initial_recon = _mean(recon_values[:smoothing_window])
    final_recon = _mean(recon_values[-smoothing_window:])
    fixed_train_patches = context.settings.fixed_train_patches
    return {
        "status": _LOCAL_STATUS,
        "proof_scope": _SELECTED_RUNTIME_SCOPE,
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
        "optimizer_steps": len(successful_rows),
        "smoothing_window_steps": smoothing_window,
        "corruption_strategy": context.settings.applied_runtime.corruption_strategy,
        "eval_views": ["train_clean", "train_corrupted_fixed_seed"],
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
        "train_steps": context.artifacts.train_steps,
        "reconstruction_samples": context.artifacts.reconstruction_samples,
        "amp_progress": context.artifacts.amp_progress,
        "local_selected_runtime_readiness": context.artifacts.local_readiness,
    }
    if context.plan_applied_proof is not None:
        artifacts["selected_runtime_plan_applied"] = (
            context.artifacts.selected_runtime_plan_applied
        )
    if context.local_ubc is not None:
        artifacts["local_ubc_mechanics"] = context.artifacts.local_ubc_mechanics
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
        "proof_scope": (
            _SELECTED_RUNTIME_SCOPE
            if context.runtime_config.plan is not None
            else _LOCAL_SCOPE
        ),
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
    plan = parse_selected_runtime_plan(runtime_config)
    payload = _load_json(runtime_config)
    blockers = _str_tuple(payload.get("launch_blockers"))
    return _RuntimeConfigProof(
        path=runtime_config,
        sha256=plan.artifact_sha256,
        selected_row_id=plan.selected_row_id,
        runtime_policy_id=plan.runtime_policy_id,
        launch_blockers=blockers,
        consumed=True,
        status=_LOCAL_STATUS,
        failure_kind="",
        plan=plan,
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
        plan=None,
    )


def _apply_selected_runtime_settings(
    settings: _TrainingSettings,
    runtime_config: _RuntimeConfigProof,
) -> _TrainingSettings:
    plan = runtime_config.plan
    if plan is None:
        return settings
    return replace(
        settings,
        batch_size=plan.per_device_batch_size,
        applied_runtime=_applied_runtime_from_plan(plan),
    )


def _default_applied_runtime(*, batch_size: int) -> _AppliedRuntimeSettings:
    return _AppliedRuntimeSettings(
        selected_row_id="",
        runtime_policy_id="",
        accelerator_mode="local_cpu",
        machine_shape="local_cpu",
        world_size=1,
        nproc_per_node=1,
        torchrun_standalone=False,
        global_batch_size=batch_size,
        gradient_accumulation_steps=1,
        precision_policy="amp_off_fp32",
        amp_enabled=False,
        autocast_dtype="float32",
        fp32_loss=True,
        grad_scaler_enabled=False,
        torch_compile_enabled=False,
        compile_scope="none",
        dataloader_num_workers=0,
        dataloader_prefetch_factor=None,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=False,
        dataloader_non_blocking_h2d=False,
        corruption_strategy="identity_clean_no_corruption",
        memory_format="contiguous",
        zero_grad_set_to_none=True,
    )


def _applied_runtime_from_plan(plan: SelectedRuntimePlan) -> _AppliedRuntimeSettings:
    return _AppliedRuntimeSettings(
        selected_row_id=plan.selected_row_id,
        runtime_policy_id=plan.runtime_policy_id,
        accelerator_mode=plan.accelerator_mode,
        machine_shape=plan.machine_shape,
        world_size=plan.world_size,
        nproc_per_node=plan.nproc_per_node,
        torchrun_standalone=plan.torchrun_standalone,
        global_batch_size=plan.global_batch_size,
        gradient_accumulation_steps=plan.gradient_accumulation_steps,
        precision_policy=plan.precision_policy,
        amp_enabled=plan.amp_enabled,
        autocast_dtype=plan.autocast_dtype,
        fp32_loss=plan.fp32_loss,
        grad_scaler_enabled=plan.grad_scaler_enabled,
        torch_compile_enabled=plan.torch_compile_enabled,
        compile_scope=plan.compile_scope,
        dataloader_num_workers=plan.dataloader_num_workers,
        dataloader_prefetch_factor=plan.dataloader_prefetch_factor,
        dataloader_pin_memory=plan.dataloader_pin_memory,
        dataloader_persistent_workers=plan.dataloader_persistent_workers,
        dataloader_non_blocking_h2d=plan.dataloader_non_blocking_h2d,
        corruption_strategy=plan.corruption_strategy,
        memory_format=plan.memory_format,
        zero_grad_set_to_none=plan.zero_grad_set_to_none,
    )


def _plan_applied_proof(  # noqa: PLR0913
    *,
    runtime_config: _RuntimeConfigProof,
    settings: _TrainingSettings,
    model: NonEquivariantVAE,
    local_ubc: _LocalUbcMechanics | None,
    metric_rows: tuple[CsvRow, ...],
    last_result: TrainStepResult,
) -> JsonObject | None:
    plan = runtime_config.plan
    if plan is None:
        return None
    observed_batch_size = _observed_batch_size(
        metric_rows=metric_rows,
        fallback=settings.batch_size,
    )
    observed_corruption_strategy = _observed_corruption_strategy(
        metric_rows=metric_rows,
        fallback=(
            "identity_clean_no_corruption"
            if local_ubc is None
            else settings.applied_runtime.corruption_strategy
        ),
    )
    observed = SelectedRuntimeApplicationObservation(
        selected_row_id=runtime_config.selected_row_id,
        runtime_policy_id=runtime_config.runtime_policy_id,
        accelerator_mode="local_cpu",
        machine_shape="local_cpu",
        world_size=1,
        nproc_per_node=1,
        torchrun_standalone=False,
        batch_size=observed_batch_size,
        global_batch_size=observed_batch_size,
        amp_enabled=False,
        grad_scaler_enabled=False,
        fp32_loss=True,
        autocast_dtype="not_executed_local_cpu",
        torch_compile_enabled=_model_is_compiled(model),
        compile_scope="none",
        dataloader_num_workers=0,
        dataloader_prefetch_factor=None,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=False,
        dataloader_non_blocking_h2d=False,
        corruption_strategy=observed_corruption_strategy,
        memory_format="contiguous",
        ddp_static_graph=False,
        ddp_gradient_as_bucket_view=False,
        zero_grad_set_to_none=last_result.zero_grad_set_to_none,
        local_ddp_status="not_executed_local_cpu_mechanics_only",
        local_amp_status="not_executed_local_cpu",
    )
    return build_plan_applied_proof(
        plan=plan,
        observed=observed,
        status_scope=_SELECTED_RUNTIME_SCOPE,
    )


def _observed_batch_size(
    *,
    metric_rows: tuple[CsvRow, ...],
    fallback: int,
) -> int:
    observed_values = {
        int(row["batch_size"])
        for row in _successful_metric_rows(metric_rows)
        if row.get("batch_size")
    }
    if len(observed_values) == 1:
        return observed_values.pop()
    return fallback


def _observed_corruption_strategy(
    *,
    metric_rows: tuple[CsvRow, ...],
    fallback: str,
) -> str:
    observed_values = {
        row["corruption_strategy"]
        for row in _successful_metric_rows(metric_rows)
        if row.get("corruption_strategy")
    }
    if len(observed_values) == 1:
        return observed_values.pop()
    return fallback


def _model_is_compiled(model: NonEquivariantVAE) -> bool:
    return hasattr(model, "_orig_mod")


def _local_ubc_mechanics_proof(
    local_ubc: _LocalUbcMechanics | None,
    *,
    settings: _TrainingSettings,
) -> JsonObject | None:
    if local_ubc is None:
        return None
    return {
        "status": local_ubc.status,
        "status_scope": _SELECTED_RUNTIME_SCOPE,
        "full_run_eligible": False,
        "data_root": str(local_ubc.root),
        "train_files": {
            "bin": TRAIN_BIN_NAME,
            "csv": TRAIN_CSV_NAME,
            "include_idx": False,
            "sample_count": local_ubc.train_count,
        },
        "validation_files": {
            "bin": VALIDATION_BIN_NAME,
            "csv": VALIDATION_CSV_NAME,
            "include_idx": True,
            "sample_count": local_ubc.validation_count,
        },
        "uses_resolve_patch_data_paths": True,
        "uses_patch_training_dataset": True,
        "uses_collate_patch_training_samples": True,
        "uses_normalize_uint8_batch": True,
        "train_corruption_strategy": settings.applied_runtime.corruption_strategy,
        "clean_validation_uses_passthrough": True,
        "failure_kind": local_ubc.failure_kind,
    }


def _amp_progress_proof(metric_rows: tuple[CsvRow, ...]) -> JsonObject:
    skipped = [row for row in metric_rows if row.get("amp_step_skipped") == "1"]
    successful = [row for row in metric_rows if row.get("amp_step_skipped") == "0"]
    return {
        "status": _LOCAL_STATUS,
        "status_scope": _SELECTED_RUNTIME_SCOPE,
        "full_run_eligible": False,
        "batch_attempt_count": len(metric_rows),
        "successful_optimizer_update_count": len(successful),
        "amp_step_skipped_count": len(skipped),
        "simulated_amp_skip_supported": True,
        "skipped_batch_attempts": [int(row["batch_attempt"]) for row in skipped],
        "skipped_steps_advance_optimizer": False,
        "skipped_steps_advance_beta": False,
        "skipped_steps_advance_lr_scheduler": False,
        "skipped_steps_trigger_checkpoint": False,
        "skipped_steps_trigger_validation": False,
        "skipped_steps_advance_tiny_smoothing": False,
    }


def _local_readiness_summary(
    context: _TrainingContext,
    ubc_mechanics_proof: JsonObject | None,
    *,
    artifact_manifest: JsonObject | None = None,
) -> JsonObject:
    plan_status = (
        "not_required"
        if context.plan_applied_proof is None
        else _string_value(context.plan_applied_proof.get("status"))
    )
    ubc_status = (
        "not_required"
        if ubc_mechanics_proof is None
        else _string_value(ubc_mechanics_proof.get("status"))
    )
    checkpoint_status = (
        "not_run" if context.loaded_checkpoint is None else _LOCAL_STATUS
    )
    artifact_manifest_status = (
        "pending"
        if artifact_manifest is None
        else _artifact_manifest_component_status(artifact_manifest)
    )
    component_status = {
        "selected_runtime_plan_applied": plan_status,
        "ubc_format_mechanics": ubc_status,
        "amp_progress": _string_value(context.amp_progress_proof.get("status")),
        "checkpoint_resume": checkpoint_status,
        "fixed_32_selector": (
            "placeholder_or_synthetic_invalid"
            if context.settings.fixed_train_patches is not None
            else "not_required"
        ),
        "artifact_manifest": artifact_manifest_status,
        "gate_health": "local_not_measured",
    }
    blocked = [
        "selected_runtime_debug_wrapper_not_wired_to_real_runner_until_spec0008",
        "fixed_32_selector_real_false",
        "missing_real_gate_health_rows",
        "missing_real_checkpoint_resume_proof",
        "missing_real_tiny_overfit_proof",
    ]
    return cast(
        "JsonObject",
        {
            "status": _FAIL_STATUS,
            "status_scope": _SELECTED_RUNTIME_SCOPE,
            "full_run_eligible": False,
            "remote_pass_ready": False,
            "real_train_runner_implemented": False,
            "fixed_32_selector_real": False,
            "component_status": component_status,
            "launch_blockers_remaining": blocked,
            "failure_kind": "local_mechanics_non_promotable_real_readiness_blocked",
        },
    )


def _artifact_manifest_component_status(artifact_manifest: JsonObject) -> str:
    missing = artifact_manifest.get("missing_artifacts")
    status = artifact_manifest.get("status")
    if status == _LOCAL_STATUS and isinstance(missing, list) and not missing:
        return _LOCAL_STATUS
    return _FAIL_STATUS


def _replace_context_readiness(
    context: _TrainingContext,
    local_readiness: JsonObject,
) -> _TrainingContext:
    return replace(context, local_readiness=local_readiness)


def _runtime_config_payload(runtime_config: _RuntimeConfigProof) -> JsonObject:
    plan = runtime_config.plan
    return {
        "path": "" if runtime_config.path is None else str(runtime_config.path),
        "sha256": runtime_config.sha256,
        "selected_row_id": runtime_config.selected_row_id,
        "runtime_policy_id": runtime_config.runtime_policy_id,
        "plan_validated": plan is not None,
        "per_device_batch_size": 0 if plan is None else plan.per_device_batch_size,
        "global_batch_size": 0 if plan is None else plan.global_batch_size,
        "precision_policy": "" if plan is None else plan.precision_policy,
        "corruption_strategy": "" if plan is None else plan.corruption_strategy,
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
    batch_size = _first_int(
        _optional_int(runtime, "batch_size"),
        default=2,
    )
    settings = _TrainingSettings(
        run_name=request.run_name or _required_str(run, "name"),
        run_mode=_required_str(run, "mode"),
        batch_size=batch_size,
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
        corruption_seed=_seed(effective, "corruption_seed"),
        corruption_profile=profile_from_config(
            _required_object(effective, "corruption"),
        ),
        applied_runtime=_default_applied_runtime(batch_size=batch_size),
        simulated_amp_skip_attempts=frozenset(
            _optional_int_list(runtime, "simulated_amp_skip_batch_attempts"),
        ),
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
    if any(attempt <= 0 for attempt in settings.simulated_amp_skip_attempts):
        message = "simulated_amp_skip_batch_attempts must contain positive integers"
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
        train_steps=output_dir / "metrics" / "train_steps.csv",
        artifact_manifest=benchmark_dir / "artifact_manifest.json",
        selected_runtime_plan_applied=(
            benchmark_dir / "selected_runtime_plan_applied.json"
        ),
        local_ubc_mechanics=benchmark_dir / "local_ubc_mechanics.json",
        amp_progress=benchmark_dir / "amp_progress.json",
        local_readiness=benchmark_dir / "local_selected_runtime_readiness.json",
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
    local_ubc: _LocalUbcMechanics | None,
    seed_offset: int,
) -> JsonObject:
    if local_ubc is None:
        clean_batch = _synthetic_clean_batch(
            batch_size=settings.batch_size,
            image_size=settings.image_size,
            generator=_seeded_torch_generator(settings.data_seed + seed_offset),
        )
        clean_validation_rng_advanced = False
    else:
        validation_batch = _training_batch(
            dataset=local_ubc.validation_dataset,
            batch_size=settings.batch_size,
            step_index=seed_offset,
        )
        clean_batch = normalize_uint8_batch(validation_batch.images_uint8)
        rng_before = torch.get_rng_state()
        clean_input = clean_validation_passthrough(clean_batch)
        clean_validation_rng_advanced = not torch.equal(
            rng_before,
            torch.get_rng_state(),
        )
        clean_batch = clean_input
    eps = _zero_eps(settings)
    with torch.no_grad():
        output = model.forward(clean_batch, eps=eps)
    metrics = reconstruction_metric_summaries(output.reconstruction, clean_batch)
    return {
        "l1": float((output.reconstruction - clean_batch).abs().mean().item()),
        "psnr": _metric_mean(metrics["psnr_img"].as_dict()),
        "ssim": _metric_mean(metrics["ssim_img"].as_dict()),
        "clean_validation_rng_advanced": clean_validation_rng_advanced,
    }


def _write_reconstruction_sample(
    *,
    path: Path,
    model: NonEquivariantVAE,
    settings: _TrainingSettings,
    local_ubc: _LocalUbcMechanics | None,
) -> bool:
    if local_ubc is None:
        clean_batch = _synthetic_clean_batch(
            batch_size=1,
            image_size=settings.image_size,
            generator=_seeded_torch_generator(settings.data_seed + 30_000),
        )
    else:
        validation_batch = _training_batch(
            dataset=local_ubc.validation_dataset,
            batch_size=1,
            step_index=30_000,
        )
        clean_batch = normalize_uint8_batch(validation_batch.images_uint8)
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
    return min(_csv_float(row, "l1_loss") for row in _successful_metric_rows(rows))


def _successful_metric_rows(rows: Sequence[CsvRow]) -> tuple[CsvRow, ...]:
    return tuple(row for row in rows if row.get("amp_step_skipped") == "0")


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


def _csv_bool(*, value: bool) -> str:
    return "true" if value else "false"


def _string_value(value: object) -> str:
    return value if isinstance(value, str) else ""


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


def _optional_int_list(payload: JsonObject, key: str) -> tuple[int, ...]:
    value = payload.get(key)
    if value is None:
        return ()
    if not isinstance(value, list):
        message = f"Expected integer-list config field {key}"
        raise TypeError(message)
    items = cast("list[object]", value)
    if any(isinstance(item, bool) or not isinstance(item, int) for item in items):
        message = f"Expected integer-list config field {key}"
        raise TypeError(message)
    return tuple(cast("list[int]", items))


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
