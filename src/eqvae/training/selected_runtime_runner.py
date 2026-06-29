# Copyright 2026 HiperMaximus
"""Real selected-runtime train runner with local dry-run proofs."""

from __future__ import annotations

import hashlib
import math
import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import (
    DataLoader,
    Dataset,
    DistributedSampler,
    Sampler,
    SequentialSampler,
)

from eqvae.benchmarking.io import CsvRow, JsonObject, JsonValue, write_csv, write_json
from eqvae.benchmarking.runtime_schema import GATE_HEALTH_COLUMNS
from eqvae.checkpointing import (
    CheckpointMetadata,
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
from eqvae.data.fixed_selectors import (
    FIXED_32_TRAIN_OVERFIT_COUNT,
    FIXED_32_TRAIN_OVERFIT_KIND,
    FixedSelectorDocument,
    load_fixed_selector_document,
)
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
    PatchTrainingSample,
    collate_patch_training_samples,
)
from eqvae.losses.vae import VaeLossComponents, beta_for_step, compute_vae_loss
from eqvae.models.activations import GatedScalarActivation
from eqvae.models.non_equivariant_vae import (
    DEFAULT_GROUPNORM_GROUPS,
    LATENT_CHANNELS,
    NonEquivariantVAE,
    build_non_equivariant_vae,
)
from eqvae.training.optim import SpecAdamWConfig, create_adamw_optimizer
from eqvae.training.selected_runtime import (
    EXPECTED_AMP_APPLICATION_STATUS,
    EXPECTED_DDP_APPLICATION_STATUS,
    EXPECTED_MACHINE_SHAPE,
    EXPECTED_RUNNER_AMP_GRAD_SCALER_INIT_SCALE,
    SelectedRuntimeApplicationObservation,
    SelectedRuntimePlan,
    build_plan_applied_proof,
    parse_selected_runtime_plan,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence

    from numpy.random import Generator


_BENCHMARK_KIND = "kaggle_selected_runtime_real_ubc_runner"
_BENCHMARK_SOURCE = "local_selected_runtime_train_runner"
_STATUS_SCOPE = "local_selected_runtime_runner"
_LOCAL_STATUS = "local_pass"
_FAIL = "fail"
_TRAIN_CORRUPTION_VIEW = "train_corrupted"
_GATE_LOW_SATURATION_THRESHOLD = 0.01
_GATE_HIGH_SATURATION_THRESHOLD = 0.99
_TINY_MAX_OPTIMIZER_STEPS = 128
_TINY_SMOOTHING_WINDOW = 25
_TINY_MIN_IMPROVEMENT_FRACTION = 0.01
_TINY_RUN_MODE = "kaggle_tiny_overfit"
SELECTED_RUNTIME_AMP_GRAD_SCALER_INIT_SCALE = EXPECTED_RUNNER_AMP_GRAD_SCALER_INIT_SCALE
_DEFAULT_SEQUENTIAL_SAMPLER_POLICY = "sequential_sampler"
_DEFAULT_DDP_SAMPLER_POLICY = "distributed_sampler_shuffle_false_drop_last_false"
_FIXED32_TINY_FULL_BATCH_SAMPLER_POLICY = "fixed32_tiny_full_batch_repeated"
_SUPPORTED_DATA = frozenset({"synthetic", "ubc-pre-shuffled"})
_TRAIN_STEP_COLUMNS = (
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


@dataclass(frozen=True)
class SelectedRuntimeTrainRequest:
    """Inputs for the selected-runtime real train runner."""

    config_path: Path
    runtime_config: Path
    output_dir: Path
    run_name: str
    data: str
    data_root: str | None = None
    fixed_train_patches: Path | None = None
    resume: Path | None = None
    max_train_steps: int | None = None
    max_val_steps: int | None = None
    save_every_steps: int | None = None
    dry_run: bool = False


@dataclass(frozen=True)
class SelectedRuntimeTrainResult:
    """Artifact paths written by the selected-runtime runner."""

    output_dir: Path
    training_summary: Path
    metrics: Path
    gate_health: Path
    artifact_manifest: Path
    selected_runtime_plan_applied: Path
    checkpoint_resume_proof: Path
    gate_health_summary: Path


@dataclass(frozen=True)
class SelectedRuntimeLaunchCommand:
    """A validated torchrun command for the selected runtime."""

    tokens: tuple[str, ...]

    @property
    def shell_command(self) -> str:
        """Return a shell-escaped command string for proof artifacts."""
        return shlex.join(self.tokens)


@dataclass(frozen=True)
class RankDeviceAssignment:
    """Observed mapping between one DDP rank and one CUDA device."""

    rank: int
    local_rank: int
    device: int
    current_device: int
    world_size: int
    device_name: str

    def as_json(self) -> JsonObject:
        """Return a JSON-safe assignment payload.

        Returns:
            Rank/device assignment facts.

        """
        return {
            "rank": self.rank,
            "local_rank": self.local_rank,
            "device": self.device,
            "current_device": self.current_device,
            "world_size": self.world_size,
            "device_name": self.device_name,
        }


@dataclass(frozen=True)
class SelectedRuntimeEnvironmentProbe:
    """Runtime facts used to prove or reject a selected-runtime execution."""

    machine_shape: str
    accelerator_mode: str
    cuda_device_count: int
    visible_device_count: int
    gpu_names: tuple[str, ...]
    world_size: int
    nproc_per_node: int
    rank: int
    local_rank: int
    torchrun_standalone: bool
    rank_assignments: tuple[RankDeviceAssignment, ...]
    distributed_initialized: bool

    def as_json(self) -> JsonObject:
        """Return JSON-safe runtime facts.

        Returns:
            Runtime probe facts.

        """
        return {
            "machine_shape": self.machine_shape,
            "accelerator_mode": self.accelerator_mode,
            "cuda_device_count": self.cuda_device_count,
            "visible_device_count": self.visible_device_count,
            "gpu_names": list(self.gpu_names),
            "world_size": self.world_size,
            "nproc_per_node": self.nproc_per_node,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "torchrun_standalone": self.torchrun_standalone,
            "rank_assignments": [
                assignment.as_json() for assignment in self.rank_assignments
            ],
            "distributed_initialized": self.distributed_initialized,
        }


@dataclass(frozen=True)
class _RunnerSettings:
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
    norm_groups: int


@dataclass(frozen=True)
class _RunArtifacts:
    training_summary: Path
    selected_runtime_debug_summary: Path
    selected_runtime_plan_applied: Path
    checkpoint_resume_proof: Path
    gate_health_summary: Path
    tiny_overfit_summary: Path
    local_readiness: Path
    artifact_manifest: Path
    train_steps: Path
    gate_health: Path
    reconstruction_samples: Path


@dataclass(frozen=True)
class _DataSurface:
    root: Path
    train_dataset: PatchTrainingDataset | _SelectedPatchTrainingDataset
    validation_dataset: PatchTrainingDataset
    train_loader: DataLoader[PatchTrainingBatch]
    validation_loader: DataLoader[PatchTrainingBatch]
    source: str
    synthetic_generated: bool
    validate_crc: bool
    fixed_train_patches: Path | None
    fixed_train_patches_sha256: str
    fixed_train_patch_count: int
    train_sampler_policy: str
    train_effective_global_epoch_samples: int
    train_effective_per_rank_epoch_samples: int


@dataclass(frozen=True)
class _TrainSamplerPlan:
    policy: str
    full_batch_repeated: bool
    effective_global_epoch_samples: int
    effective_per_rank_epoch_samples: int


class _SelectedPatchTrainingDataset(Dataset[PatchTrainingSample]):
    """Dataset view restricted to a validated fixed selector."""

    def __init__(
        self,
        base: PatchTrainingDataset,
        *,
        row_indices: Sequence[int],
    ) -> None:
        """Create the selected-row dataset view."""
        self._base = base
        self._row_indices = tuple(row_indices)

    @property
    def split(self) -> str:
        """Return the wrapped split name.

        Returns:
            Canonical split name.

        """
        return self._base.split

    @property
    def records(self) -> tuple[object, ...]:
        """Return records in selector order for audit code.

        Returns:
            Selected records.

        """
        return tuple(self._base.records[index] for index in self._row_indices)

    def __len__(self) -> int:
        """Return the selected patch count.

        Returns:
            Number of selected rows.

        """
        return len(self._row_indices)

    def __getitem__(self, index: int) -> PatchTrainingSample:
        """Return one selected training sample.

        Args:
            index: Selector-order dataset index.

        Returns:
            Wrapped sample.

        """
        return self._base[self._row_indices[index]]

    def close(self) -> None:
        """Close the wrapped dataset."""
        self._base.close()


class _FixedSelectorFullBatchSampler(Sampler[int]):
    """Repeat a fixed selector to full per-rank batches for tiny overfit proof."""

    def __init__(
        self,
        *,
        dataset_size: int,
        batch_size: int,
        world_size: int,
        rank: int,
    ) -> None:
        """Create a finite deterministic sampler epoch."""
        self._indices = fixed_selector_full_batch_indices(
            dataset_size=dataset_size,
            batch_size=batch_size,
            world_size=world_size,
            rank=rank,
        )

    def __iter__(self) -> Iterator[int]:
        """Yield one repeated full-batch sampler epoch.

        Returns:
            Iterator over selected dataset indices for this rank.

        """
        return iter(self._indices)

    def __len__(self) -> int:
        """Return the number of samples emitted on this rank per epoch.

        Returns:
            Per-rank sample count.

        """
        return len(self._indices)


def fixed_selector_full_batch_indices(
    *,
    dataset_size: int,
    batch_size: int,
    world_size: int,
    rank: int,
) -> tuple[int, ...]:
    """Return deterministic per-rank indices padded to full microbatches.

    The canonical selector still has ``dataset_size`` unique patches. Padding
    only repeats selector-order indices so tiny overfit proofs exercise stable
    selected-runtime microbatches instead of accidental tail batches.

    Args:
        dataset_size: Number of unique fixed-selector rows.
        batch_size: Per-rank microbatch size.
        world_size: Number of participating ranks.
        rank: Current rank.

    Returns:
        Per-rank selector indices for one repeated full-batch epoch.

    Raises:
        ValueError: If size arguments are invalid or rank is out of range.

    """
    if dataset_size <= 0:
        message = "fixed selector full-batch sampler requires a nonempty dataset"
        raise ValueError(message)
    if batch_size <= 0:
        message = "fixed selector full-batch sampler requires positive batch_size"
        raise ValueError(message)
    if world_size <= 0:
        message = "fixed selector full-batch sampler requires positive world_size"
        raise ValueError(message)
    if rank < 0 or rank >= world_size:
        message = f"rank must be in [0, {world_size}): {rank}"
        raise ValueError(message)
    global_batch_size = batch_size * world_size
    global_epoch_samples = math.ceil(dataset_size / global_batch_size)
    global_epoch_samples *= global_batch_size
    return tuple(
        index % dataset_size for index in range(rank, global_epoch_samples, world_size)
    )


@dataclass(frozen=True)
class _DistributedContext:
    device: torch.device
    rank: int
    local_rank: int
    world_size: int
    nproc_per_node: int
    should_use_ddp: bool
    initialized_here: bool
    probe: SelectedRuntimeEnvironmentProbe


@dataclass(frozen=True)
class _AmpExecution:
    enabled: bool
    grad_scaler_enabled: bool
    grad_scaler_init_scale: float
    autocast_dtype: str
    requested_autocast_dtype: str
    local_amp_status: str


@dataclass(frozen=True)
class _SelectedRuntimeStepResult:
    optimizer_step_index: int
    successful_optimizer_update_count: int
    losses: VaeLossComponents
    grad_norm: float
    param_update_norm: float
    nonfinite_count: int
    batch_size: int
    amp_step_skipped: bool
    zero_grad_set_to_none: bool


@dataclass(frozen=True)
class _RuntimeIdentity:
    path: Path
    sha256: str
    selected_row_id: str
    runtime_policy_id: str


@dataclass(frozen=True)
class _LocalReadinessComponents:
    plan_applied: JsonObject
    checkpoint_resume_proof: JsonObject
    gate_health_summary: JsonObject
    data_source: str
    ddp_proof: JsonObject
    amp_step_skipped_count: int
    nonfinite_count: int


def build_selected_runtime_torchrun_command(  # noqa: PLR0913
    *,
    config_path: Path,
    runtime_config: Path,
    data: str,
    output_dir: Path,
    run_name: str,
    data_root: str | None = None,
    fixed_train_patches: Path | None = None,
    max_train_steps: int | None = None,
    max_val_steps: int | None = None,
    save_every_steps: int | None = None,
    dry_run: bool = False,
) -> SelectedRuntimeLaunchCommand:
    """Build the selected-runtime dual-rank torchrun command.

    Returns:
        Command tokens. The command is not executed by this helper.

    """
    tokens = [
        "torchrun",
        "--standalone",
        "--nproc_per_node=2",
        "-m",
        "eqvae.cli.selected_runtime_train",
        "--config",
        str(config_path),
        "--runtime-config",
        str(runtime_config),
        "--data",
        data,
        "--output-dir",
        str(output_dir),
        "--run-name",
        run_name,
    ]
    if data_root is not None:
        tokens.extend(["--data-root", data_root])
    if fixed_train_patches is not None:
        tokens.extend(["--fixed-train-patches", str(fixed_train_patches)])
    if max_train_steps is not None:
        tokens.extend(["--max-train-steps", str(max_train_steps)])
    if max_val_steps is not None:
        tokens.extend(["--max-val-steps", str(max_val_steps)])
    if save_every_steps is not None:
        tokens.extend(["--save-every-steps", str(save_every_steps)])
    if dry_run:
        tokens.append("--dry-run")
    return SelectedRuntimeLaunchCommand(tokens=tuple(tokens))


def validate_selected_runtime_torchrun_command(
    tokens: Sequence[str],
    *,
    plan: SelectedRuntimePlan,
) -> tuple[str, ...]:
    """Validate a selected-runtime launch command without executing it.

    Returns:
        Stable failure identifiers.

    """
    errors: list[str] = []
    if not tokens or Path(tokens[0]).name != "torchrun":
        errors.append("selected_runtime_runner_launch_not_torchrun")
    if tokens.count("--standalone") != 1:
        errors.append("selected_runtime_runner_launch_missing_standalone")
    nproc_values = _token_option_values(tokens, "--nproc_per_node")
    expected_nproc = str(plan.nproc_per_node)
    if not nproc_values:
        errors.append("selected_runtime_runner_launch_missing_nproc")
    elif len(nproc_values) != 1:
        errors.append("selected_runtime_runner_launch_duplicate_nproc")
    elif nproc_values[0] != expected_nproc:
        errors.append("selected_runtime_runner_launch_wrong_nproc")
    if "-m" not in tokens:
        errors.append("selected_runtime_runner_launch_missing_module")
    else:
        module_index = list(tokens).index("-m") + 1
        if module_index >= len(tokens):
            errors.append("selected_runtime_runner_launch_missing_module")
        elif tokens[module_index] != "eqvae.cli.selected_runtime_train":
            errors.append("selected_runtime_runner_launch_wrong_module")
    return tuple(errors)


def validate_selected_runtime_environment(
    probe: SelectedRuntimeEnvironmentProbe,
    *,
    plan: SelectedRuntimePlan,
) -> tuple[str, ...]:
    """Return stable failure IDs for runtime environment mismatches.

    Returns:
        Stable failure identifiers.

    """
    errors: list[str] = []
    if probe.machine_shape != plan.machine_shape:
        errors.append("selected_runtime_runner_wrong_accelerator")
    if probe.accelerator_mode != plan.accelerator_mode:
        errors.append("selected_runtime_runner_wrong_accelerator_mode")
    if probe.cuda_device_count != plan.world_size:
        errors.append("selected_runtime_runner_cuda_device_count_mismatch")
    if probe.visible_device_count != plan.world_size:
        if probe.visible_device_count == 1 and _has_t4_name(probe.gpu_names):
            errors.append("selected_runtime_runner_single_visible_t4")
        else:
            errors.append("selected_runtime_runner_visible_device_count_mismatch")
    if probe.world_size != plan.world_size:
        errors.append("selected_runtime_runner_world_size_mismatch")
    if probe.nproc_per_node != plan.nproc_per_node:
        errors.append("selected_runtime_runner_nproc_per_node_mismatch")
    if probe.torchrun_standalone is not plan.torchrun_standalone:
        errors.append("selected_runtime_runner_torchrun_standalone_mismatch")
    errors.extend(_distributed_environment_errors(probe=probe, plan=plan))
    return tuple(errors)


def _distributed_environment_errors(
    *,
    probe: SelectedRuntimeEnvironmentProbe,
    plan: SelectedRuntimePlan,
) -> tuple[str, ...]:
    errors: list[str] = []
    if not probe.distributed_initialized:
        errors.append("selected_runtime_runner_distributed_not_initialized")
    if not _rank_assignments_match_plan(probe.rank_assignments, plan=plan):
        errors.append("selected_runtime_runner_rank_device_mismatch")
    return tuple(errors)


def build_ddp_rank_device_proof(
    *,
    plan: SelectedRuntimePlan,
    probe: SelectedRuntimeEnvironmentProbe,
    launch_command: SelectedRuntimeLaunchCommand,
    dry_run: bool,
) -> JsonObject:
    """Build a JSON proof for the selected-runtime DDP/rank contract.

    Returns:
        JSON-safe proof payload.

    """
    command_errors = validate_selected_runtime_torchrun_command(
        launch_command.tokens,
        plan=plan,
    )
    environment_errors = validate_selected_runtime_environment(probe, plan=plan)
    errors = (*command_errors, *environment_errors)
    return cast(
        "JsonObject",
        {
            "status": _LOCAL_STATUS if not errors else _FAIL,
            "status_scope": _STATUS_SCOPE,
            "full_run_eligible": False,
            "dry_run": dry_run,
            "selected_row_id": plan.selected_row_id,
            "runtime_policy_id": plan.runtime_policy_id,
            "child_process_launch_command": launch_command.shell_command,
            "command_validation_errors": list(command_errors),
            "environment_validation_errors": list(environment_errors),
            "rank_assignment_status": "pass" if not environment_errors else "fail",
            "probe": probe.as_json(),
            "failure_kind": "" if not errors else "selected_runtime_runner_environment",
        },
    )


def write_selected_runtime_training_run(  # noqa: PLR0914, PLR0915
    request: SelectedRuntimeTrainRequest,
) -> SelectedRuntimeTrainResult:
    """Run the selected-runtime train runner and write proof artifacts.

    Returns:
        Paths for the required runner artifacts.

    Raises:
        ValueError: If selected-runtime, data, or resume settings are invalid.

    """
    if request.data not in _SUPPORTED_DATA:
        message = f"Unsupported data surface {request.data!r}"
        raise ValueError(message)
    resolved = resolve_json_config(request.config_path)
    plan = parse_selected_runtime_plan(request.runtime_config)
    settings = _settings(request=request, resolved=resolved, plan=plan)
    artifacts = _artifact_paths(request.output_dir)
    runtime_identity = _runtime_identity(plan)
    launch_command = build_selected_runtime_torchrun_command(
        config_path=request.config_path,
        runtime_config=request.runtime_config,
        data=request.data,
        data_root=request.data_root,
        fixed_train_patches=request.fixed_train_patches,
        output_dir=request.output_dir,
        run_name=settings.run_name,
        max_train_steps=request.max_train_steps,
        max_val_steps=request.max_val_steps,
        save_every_steps=request.save_every_steps,
        dry_run=request.dry_run,
    )
    distributed = _distributed_context(plan=plan, dry_run=request.dry_run)
    ddp_proof = build_ddp_rank_device_proof(
        plan=plan,
        probe=distributed.probe,
        launch_command=launch_command,
        dry_run=request.dry_run,
    )

    manual_seed = cast("Callable[[int], torch.Generator]", torch.manual_seed)
    manual_seed(settings.global_seed)
    numpy_generator = np.random.default_rng(settings.global_seed)
    train_generator = torch.Generator(device="cpu")
    train_generator.manual_seed(settings.data_seed)
    data_surface = _prepare_data_surface(
        request=request,
        settings=settings,
        plan=plan,
        distributed=distributed,
    )

    model = build_non_equivariant_vae(norm_groups=settings.norm_groups)
    model = _place_model(model=model, plan=plan, device=distributed.device)
    optimizer, _ = create_adamw_optimizer(model, config=settings.optimizer_config)
    loaded_checkpoint = _restore_checkpoint_if_requested(
        request=request,
        model=model,
        optimizer=optimizer,
        numpy_generator=numpy_generator,
        train_generator=train_generator,
        runtime_identity=runtime_identity,
        resolved=resolved,
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

    wrapped_model = _maybe_wrap_ddp(
        model=model,
        distributed=distributed,
        plan=plan,
    )
    amp = _amp_execution(plan=plan, distributed=distributed, dry_run=request.dry_run)
    scaler = GradScaler(
        "cuda",
        init_scale=amp.grad_scaler_init_scale,
        enabled=amp.grad_scaler_enabled,
    )
    write_artifacts = _is_primary_rank(distributed)
    metric_rows, checkpoints, last_result = _run_train_steps(
        request=request,
        resolved=resolved,
        settings=settings,
        plan=plan,
        model=wrapped_model,
        checkpoint_model=model,
        optimizer=optimizer,
        scaler=scaler,
        amp=amp,
        data_surface=data_surface,
        distributed=distributed,
        numpy_generator=numpy_generator,
        train_generator=train_generator,
        runtime_identity=runtime_identity,
        start_step=start_step,
        write_checkpoints=write_artifacts,
    )
    metric_rows = _gather_csv_rows(metric_rows, distributed)
    gate_rows = _gather_csv_rows(
        _gate_health_rows(
            run_name=settings.run_name,
            plan=plan,
            probe=distributed.probe,
            amp=amp,
            model=model,
            optimizer_step=settings.max_train_steps,
            rank=distributed.rank,
        ),
        distributed,
    )
    if not write_artifacts:
        _barrier(distributed)
        _close_data_surface(data_surface)
        _cleanup_distributed(distributed)
        return SelectedRuntimeTrainResult(
            output_dir=request.output_dir,
            training_summary=artifacts.training_summary,
            metrics=artifacts.train_steps,
            gate_health=artifacts.gate_health,
            artifact_manifest=artifacts.artifact_manifest,
            selected_runtime_plan_applied=artifacts.selected_runtime_plan_applied,
            checkpoint_resume_proof=artifacts.checkpoint_resume_proof,
            gate_health_summary=artifacts.gate_health_summary,
        )
    final_checkpoint = _save_checkpoint(
        path=request.output_dir / "checkpoints" / "final.pt",
        request=request,
        resolved=resolved,
        settings=settings,
        model=model,
        optimizer=optimizer,
        numpy_generator=numpy_generator,
        train_generator=train_generator,
        runtime_identity=runtime_identity,
        step=settings.max_train_steps,
        metric_value=_best_l1(metric_rows),
        scaler=scaler,
        amp=amp,
        distributed=distributed,
    )
    best_checkpoint = _save_checkpoint(
        path=request.output_dir / "checkpoints" / "best_model.pt",
        request=request,
        resolved=resolved,
        settings=settings,
        model=model,
        optimizer=optimizer,
        numpy_generator=numpy_generator,
        train_generator=train_generator,
        runtime_identity=runtime_identity,
        step=settings.max_train_steps,
        metric_value=_best_l1(metric_rows),
        scaler=scaler,
        amp=amp,
        distributed=distributed,
    )
    all_checkpoints = (*checkpoints, final_checkpoint, best_checkpoint)
    if loaded_checkpoint is None:
        checkpoint_resume_proof = _checkpoint_resume_proof(
            checkpoint=_resume_probe_checkpoint(
                checkpoints=checkpoints,
                target_step=settings.max_train_steps,
            ),
            request=request,
            resolved=resolved,
            settings=settings,
            runtime_identity=runtime_identity,
            train_generator=train_generator,
            amp=amp,
            distributed=distributed,
        )
    else:
        checkpoint_resume_proof = _loaded_checkpoint_resume_proof(
            loaded=loaded_checkpoint,
            request=request,
            resolved=resolved,
            settings=settings,
            runtime_identity=runtime_identity,
            amp=amp,
            distributed=distributed,
        )
    reconstruction_nonblank = _write_reconstruction_sample(
        path=artifacts.reconstruction_samples,
        model=model,
        settings=settings,
        data_surface=data_surface,
        device=distributed.device,
    )
    gate_health_summary = _gate_health_summary(gate_rows)
    plan_applied = _plan_applied_proof(
        plan=plan,
        settings=settings,
        probe=distributed.probe,
        amp=amp,
        ddp_proof=ddp_proof,
        metric_rows=metric_rows,
        last_result=last_result,
    )
    local_readiness = _local_readiness_summary(
        _LocalReadinessComponents(
            plan_applied=plan_applied,
            checkpoint_resume_proof=checkpoint_resume_proof,
            gate_health_summary=gate_health_summary,
            data_source=data_surface.source,
            ddp_proof=ddp_proof,
            amp_step_skipped_count=_amp_step_skipped_count(metric_rows),
            nonfinite_count=_nonfinite_metric_count(metric_rows),
        ),
    )

    write_csv(artifacts.train_steps, _TRAIN_STEP_COLUMNS, metric_rows)
    write_csv(artifacts.gate_health, GATE_HEALTH_COLUMNS, gate_rows)
    write_json(artifacts.selected_runtime_plan_applied, plan_applied)
    write_json(artifacts.checkpoint_resume_proof, checkpoint_resume_proof)
    write_json(artifacts.gate_health_summary, gate_health_summary)
    write_json(artifacts.local_readiness, local_readiness)
    write_json(
        artifacts.training_summary,
        _training_summary(
            request=request,
            resolved=resolved,
            settings=settings,
            plan=plan,
            runtime_identity=runtime_identity,
            launch_command=launch_command,
            ddp_proof=ddp_proof,
            amp=amp,
            data_surface=data_surface,
            metric_rows=metric_rows,
            checkpoints=all_checkpoints,
            final_checkpoint=final_checkpoint,
            best_checkpoint=best_checkpoint,
            last_result=last_result,
            plan_applied=plan_applied,
            checkpoint_resume_proof=checkpoint_resume_proof,
            gate_health_summary=gate_health_summary,
            reconstruction_nonblank=reconstruction_nonblank,
        ),
    )
    write_json(
        artifacts.selected_runtime_debug_summary,
        _selected_runtime_debug_summary(
            plan=plan,
            settings=settings,
            plan_applied=plan_applied,
            ddp_proof=ddp_proof,
            amp=amp,
            checkpoint_resume_proof=checkpoint_resume_proof,
            gate_health_summary=gate_health_summary,
            data_surface=data_surface,
        ),
    )
    if _writes_tiny_summary(settings):
        write_json(
            artifacts.tiny_overfit_summary,
            _tiny_overfit_summary(
                runtime_identity=runtime_identity,
                corruption_strategy=plan.corruption_strategy,
                data_surface=data_surface,
                metric_rows=metric_rows,
                gate_health_summary=gate_health_summary,
            ),
        )
    manifest = _artifact_manifest(
        artifacts=artifacts,
        settings=settings,
        checkpoints=all_checkpoints,
        metric_rows=metric_rows,
        reconstruction_nonblank=reconstruction_nonblank,
    )
    write_json(artifacts.artifact_manifest, manifest)
    _barrier(distributed)
    _close_data_surface(data_surface)
    _cleanup_distributed(distributed)
    return SelectedRuntimeTrainResult(
        output_dir=request.output_dir,
        training_summary=artifacts.training_summary,
        metrics=artifacts.train_steps,
        gate_health=artifacts.gate_health,
        artifact_manifest=artifacts.artifact_manifest,
        selected_runtime_plan_applied=artifacts.selected_runtime_plan_applied,
        checkpoint_resume_proof=artifacts.checkpoint_resume_proof,
        gate_health_summary=artifacts.gate_health_summary,
    )


def _token_option_values(tokens: Sequence[str], option: str) -> list[str]:
    prefix = f"{option}="
    values = [
        token.removeprefix(prefix) for token in tokens if token.startswith(prefix)
    ]
    for index, token in enumerate(tokens[:-1]):
        if token == option:
            values.append(tokens[index + 1])
    return values


def _has_t4_name(gpu_names: Sequence[str]) -> bool:
    return any("t4" in name.lower() for name in gpu_names)


def _rank_assignments_match_plan(
    assignments: Sequence[RankDeviceAssignment],
    *,
    plan: SelectedRuntimePlan,
) -> bool:
    observed = {
        (
            assignment.rank,
            assignment.local_rank,
            assignment.device,
            assignment.current_device,
            assignment.world_size,
        )
        for assignment in assignments
    }
    expected = {
        (rank, rank, rank, rank, plan.world_size) for rank in range(plan.world_size)
    }
    return observed == expected


def _settings(
    *,
    request: SelectedRuntimeTrainRequest,
    resolved: ResolvedConfig,
    plan: SelectedRuntimePlan,
) -> _RunnerSettings:
    effective = resolved.effective_config
    run = _required_object(effective, "run")
    data = _required_object(effective, "data")
    training = _optional_object(effective, "training") or {}
    objective = _required_object(effective, "objective")
    beta = _required_object(objective, "beta")
    settings = _RunnerSettings(
        run_name=request.run_name or _required_str(run, "name"),
        run_mode=_required_str(run, "mode"),
        batch_size=plan.per_device_batch_size,
        image_size=_optional_int(data, "image_size") or 256,
        max_train_steps=(
            request.max_train_steps
            if request.max_train_steps is not None
            else _optional_int(training, "max_train_steps") or 1
        ),
        max_val_steps=(
            request.max_val_steps
            if request.max_val_steps is not None
            else _optional_int(training, "max_val_steps") or 0
        ),
        save_every_steps=(
            request.save_every_steps
            if request.save_every_steps is not None
            else _optional_int(training, "save_every_steps") or 1
        ),
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
        norm_groups=_norm_groups(resolved),
    )
    _validate_settings(settings)
    return settings


def _runtime_identity(plan: SelectedRuntimePlan) -> _RuntimeIdentity:
    return _RuntimeIdentity(
        path=plan.path,
        sha256=plan.artifact_sha256,
        selected_row_id=plan.selected_row_id,
        runtime_policy_id=plan.runtime_policy_id,
    )


def _artifact_paths(output_dir: Path) -> _RunArtifacts:
    benchmark = output_dir / "benchmark"
    metrics = output_dir / "metrics"
    return _RunArtifacts(
        training_summary=benchmark / "training_summary.json",
        selected_runtime_debug_summary=benchmark
        / "selected_runtime_debug_summary.json",
        selected_runtime_plan_applied=benchmark / "selected_runtime_plan_applied.json",
        checkpoint_resume_proof=benchmark / "checkpoint_resume_proof.json",
        gate_health_summary=benchmark / "gate_health_summary.json",
        tiny_overfit_summary=benchmark / "tiny_overfit_summary.json",
        local_readiness=benchmark / "local_selected_runtime_readiness.json",
        artifact_manifest=benchmark / "artifact_manifest.json",
        train_steps=metrics / "train_steps.csv",
        gate_health=metrics / "gate_health.csv",
        reconstruction_samples=output_dir / "artifacts" / "reconstruction_samples.pt",
    )


def _distributed_context(
    *,
    plan: SelectedRuntimePlan,
    dry_run: bool,
) -> _DistributedContext:
    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", 0)
    world_size = _env_int("WORLD_SIZE", 1)
    nproc_per_node = _env_int("LOCAL_WORLD_SIZE", world_size)
    cuda_count = torch.cuda.device_count()
    cuda_available = torch.cuda.is_available()
    should_use_ddp = (
        not dry_run
        and cuda_available
        and world_size == plan.world_size
        and nproc_per_node == plan.nproc_per_node
    )
    device = torch.device("cuda", local_rank) if should_use_ddp else torch.device("cpu")
    initialized_here = False
    if should_use_ddp:
        torch.cuda.set_device(device)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
            initialized_here = True
    gpu_names = _gpu_names(cuda_count)
    assignments = _rank_assignments(
        plan=plan,
        should_use_ddp=should_use_ddp,
        cuda_count=cuda_count,
        gpu_names=gpu_names,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
    )
    probe = SelectedRuntimeEnvironmentProbe(
        machine_shape=EXPECTED_MACHINE_SHAPE
        if _all_t4(gpu_names, plan)
        else "local_cpu",
        accelerator_mode="dual_t4_ddp" if should_use_ddp else "local_cpu",
        cuda_device_count=cuda_count,
        visible_device_count=cuda_count,
        gpu_names=gpu_names,
        world_size=world_size,
        nproc_per_node=nproc_per_node,
        rank=rank,
        local_rank=local_rank,
        torchrun_standalone=("TORCHELASTIC_RUN_ID" in os.environ),
        rank_assignments=assignments,
        distributed_initialized=dist.is_initialized(),
    )
    return _DistributedContext(
        device=device,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        nproc_per_node=nproc_per_node,
        should_use_ddp=should_use_ddp,
        initialized_here=initialized_here,
        probe=probe,
    )


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _gpu_names(cuda_count: int) -> tuple[str, ...]:
    names: list[str] = []
    for index in range(cuda_count):
        try:
            names.append(torch.cuda.get_device_name(index))
        except RuntimeError:
            names.append("unavailable")
    return tuple(names)


def _all_t4(gpu_names: Sequence[str], plan: SelectedRuntimePlan) -> bool:
    return len(gpu_names) == plan.world_size and all(
        "t4" in name.lower() for name in gpu_names
    )


def _rank_assignments(  # noqa: PLR0913
    *,
    plan: SelectedRuntimePlan,
    should_use_ddp: bool,
    cuda_count: int,
    gpu_names: Sequence[str],
    rank: int,
    local_rank: int,
    world_size: int,
) -> tuple[RankDeviceAssignment, ...]:
    if not should_use_ddp:
        return ()
    current_device = torch.cuda.current_device()
    if world_size == plan.world_size and 0 <= rank < plan.world_size:
        device_name = gpu_names[local_rank] if local_rank < len(gpu_names) else ""
        local_assignment = RankDeviceAssignment(
            rank=rank,
            local_rank=local_rank,
            device=local_rank,
            current_device=current_device,
            world_size=world_size,
            device_name=device_name,
        )
        return _gather_rank_assignments(local_assignment, world_size=world_size)
    return tuple(
        RankDeviceAssignment(
            rank=index,
            local_rank=index,
            device=index,
            current_device=index,
            world_size=world_size,
            device_name=gpu_names[index] if index < cuda_count else "",
        )
        for index in range(min(cuda_count, plan.world_size))
    )


def _gather_rank_assignments(
    local_assignment: RankDeviceAssignment,
    *,
    world_size: int,
) -> tuple[RankDeviceAssignment, ...]:
    if not dist.is_initialized():
        return (local_assignment,)
    gathered: list[object] = [None for _ in range(world_size)]
    all_gather_object = cast(
        "Callable[[list[object], object], None]",
        dist.all_gather_object,
    )
    all_gather_object(gathered, local_assignment)
    assignments = [
        assignment
        for assignment in gathered
        if isinstance(assignment, RankDeviceAssignment)
    ]
    return tuple(sorted(assignments, key=lambda assignment: assignment.rank))


def _is_primary_rank(distributed: _DistributedContext) -> bool:
    return not distributed.should_use_ddp or distributed.rank == 0


def _barrier(distributed: _DistributedContext) -> None:
    if distributed.should_use_ddp and dist.is_initialized():
        barrier = cast("Callable[[], object]", dist.barrier)
        barrier()


def _gather_csv_rows(
    local_rows: Sequence[CsvRow],
    distributed: _DistributedContext,
) -> tuple[CsvRow, ...]:
    if not distributed.should_use_ddp or not dist.is_initialized():
        return tuple(local_rows)
    gathered: list[object] = [None for _ in range(distributed.world_size)]
    all_gather_object = cast(
        "Callable[[list[object], object], None]",
        dist.all_gather_object,
    )
    all_gather_object(gathered, tuple(local_rows))
    rows: list[CsvRow] = []
    for rank_rows in gathered:
        if not isinstance(rank_rows, tuple | list):
            continue
        rank_row_sequence = cast("Sequence[object]", rank_rows)
        rows.extend(
            cast("CsvRow", row_object)
            for row_object in rank_row_sequence
            if isinstance(row_object, dict)
        )
    return tuple(rows)


def _prepare_data_surface(
    *,
    request: SelectedRuntimeTrainRequest,
    settings: _RunnerSettings,
    plan: SelectedRuntimePlan,
    distributed: _DistributedContext,
) -> _DataSurface:
    if request.data == "synthetic":
        root = request.output_dir / "local_ubc_synthetic"
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
        validate_crc = True
        source = "synthetic_ubc_format_local"
        synthetic_generated = True
    else:
        root = Path(request.data_root or "auto")
        validate_crc = False
        source = "ubc-pre-shuffled"
        synthetic_generated = False
    paths = resolve_patch_data_paths(root)
    (
        train_dataset,
        fixed_train_patches,
        fixed_train_patches_sha256,
        fixed_train_patch_count,
    ) = _apply_fixed_train_selector(
        dataset=PatchTrainingDataset(
            PatchTrainingDatasetSpec(
                bin_path=paths.train.bin_path,
                csv_path=paths.train.csv_path,
                split="train",
                image_size=settings.image_size,
                channels=3,
                validate_crc=validate_crc,
            ),
        ),
        selector_path=request.fixed_train_patches,
    )
    train_sampler = _train_sampler_plan(
        settings=settings,
        fixed_train_patch_count=fixed_train_patch_count,
        dataset_size=len(train_dataset),
        batch_size=settings.batch_size,
        distributed=distributed,
    )
    validation_dataset = PatchTrainingDataset(
        PatchTrainingDatasetSpec(
            bin_path=paths.validation.bin_path,
            csv_path=paths.validation.csv_path,
            split="validation",
            image_size=settings.image_size,
            channels=3,
            validate_crc=validate_crc,
        ),
    )
    train_loader = _loader(
        dataset=train_dataset,
        batch_size=settings.batch_size,
        plan=plan,
        distributed=distributed,
        full_batch_repeated=train_sampler.full_batch_repeated,
    )
    validation_loader = _loader(
        dataset=validation_dataset,
        batch_size=settings.batch_size,
        plan=plan,
        distributed=distributed,
        full_batch_repeated=False,
    )
    return _DataSurface(
        root=paths.root,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        train_loader=train_loader,
        validation_loader=validation_loader,
        source=source,
        synthetic_generated=synthetic_generated,
        validate_crc=validate_crc,
        fixed_train_patches=fixed_train_patches,
        fixed_train_patches_sha256=fixed_train_patches_sha256,
        fixed_train_patch_count=fixed_train_patch_count,
        train_sampler_policy=train_sampler.policy,
        train_effective_global_epoch_samples=(
            train_sampler.effective_global_epoch_samples
        ),
        train_effective_per_rank_epoch_samples=(
            train_sampler.effective_per_rank_epoch_samples
        ),
    )


def _apply_fixed_train_selector(
    *,
    dataset: PatchTrainingDataset,
    selector_path: Path | None,
) -> tuple[PatchTrainingDataset | _SelectedPatchTrainingDataset, Path | None, str, int]:
    if selector_path is None:
        return dataset, None, "", 0
    document = load_fixed_selector_document(selector_path)
    row_indices = _validated_fixed_train_row_indices(
        document=document,
        dataset=dataset,
    )
    return (
        _SelectedPatchTrainingDataset(dataset, row_indices=row_indices),
        selector_path,
        _sha256_file(selector_path),
        len(row_indices),
    )


def _validated_fixed_train_row_indices(
    *,
    document: FixedSelectorDocument,
    dataset: PatchTrainingDataset,
) -> tuple[int, ...]:
    if document.selector_kind != FIXED_32_TRAIN_OVERFIT_KIND:
        message = (
            "selected-runtime runner only accepts fixed_32_train_overfit selectors"
        )
        raise ValueError(message)
    if document.source_split != "train":
        message = "selected-runtime fixed train selector must target train split"
        raise ValueError(message)
    records = dataset.records
    row_indices: list[int] = []
    for selector in document.selectors:
        if selector.row_index < 0 or selector.row_index >= len(records):
            message = f"selector row_index outside train dataset: {selector.row_index}"
            raise ValueError(message)
        record = records[selector.row_index]
        expected = (
            record.file_index,
            record.row_index,
            record.sample_id("train"),
            record.wsi_id,
            record.label,
            record.x,
            record.y,
        )
        observed = (
            selector.file_index,
            selector.row_index,
            selector.sample_id,
            selector.wsi_id,
            selector.label,
            selector.x,
            selector.y,
        )
        if observed != expected:
            message = f"fixed train selector row mismatch at rank {selector.rank}"
            raise ValueError(message)
        row_indices.append(selector.row_index)
    if len(set(row_indices)) != len(row_indices):
        message = "fixed train selector contains duplicate row indices"
        raise ValueError(message)
    return tuple(row_indices)


def _loader(
    *,
    dataset: PatchTrainingDataset | _SelectedPatchTrainingDataset,
    batch_size: int,
    plan: SelectedRuntimePlan,
    distributed: _DistributedContext,
    full_batch_repeated: bool,
) -> DataLoader[PatchTrainingBatch]:
    sampler: Sampler[int]
    if full_batch_repeated:
        sampler = cast(
            "Sampler[int]",
            _FixedSelectorFullBatchSampler(
                dataset_size=len(dataset),
                batch_size=batch_size,
                world_size=distributed.world_size if distributed.should_use_ddp else 1,
                rank=distributed.rank if distributed.should_use_ddp else 0,
            ),
        )
    elif distributed.should_use_ddp:
        sampler = cast(
            "Sampler[int]",
            DistributedSampler(
                dataset,
                num_replicas=distributed.world_size,
                rank=distributed.rank,
                shuffle=False,
                drop_last=False,
            ),
        )
    else:
        sampler = cast("Sampler[int]", SequentialSampler(dataset))
    prefetch_factor = (
        None if plan.dataloader_num_workers == 0 else plan.dataloader_prefetch_factor
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=plan.dataloader_num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=plan.dataloader_pin_memory,
        persistent_workers=plan.dataloader_persistent_workers,
        collate_fn=collate_patch_training_samples,
        drop_last=False,
    )
    return cast("DataLoader[PatchTrainingBatch]", loader)


def _uses_fixed_tiny_full_batch_sampler(
    *,
    settings: _RunnerSettings,
    fixed_train_patch_count: int,
) -> bool:
    return (
        settings.run_mode == _TINY_RUN_MODE
        and fixed_train_patch_count == FIXED_32_TRAIN_OVERFIT_COUNT
    )


def _train_sampler_plan(
    *,
    settings: _RunnerSettings,
    fixed_train_patch_count: int,
    dataset_size: int,
    batch_size: int,
    distributed: _DistributedContext,
) -> _TrainSamplerPlan:
    full_batch_repeated = _uses_fixed_tiny_full_batch_sampler(
        settings=settings,
        fixed_train_patch_count=fixed_train_patch_count,
    )
    (
        effective_global_epoch_samples,
        effective_per_rank_epoch_samples,
    ) = _effective_train_epoch_samples(
        dataset_size=dataset_size,
        batch_size=batch_size,
        distributed=distributed,
        full_batch_repeated=full_batch_repeated,
    )
    return _TrainSamplerPlan(
        policy=_train_sampler_policy(
            distributed=distributed,
            full_batch_repeated=full_batch_repeated,
        ),
        full_batch_repeated=full_batch_repeated,
        effective_global_epoch_samples=effective_global_epoch_samples,
        effective_per_rank_epoch_samples=effective_per_rank_epoch_samples,
    )


def _train_sampler_policy(
    *,
    distributed: _DistributedContext,
    full_batch_repeated: bool,
) -> str:
    if full_batch_repeated:
        return _FIXED32_TINY_FULL_BATCH_SAMPLER_POLICY
    if distributed.should_use_ddp:
        return _DEFAULT_DDP_SAMPLER_POLICY
    return _DEFAULT_SEQUENTIAL_SAMPLER_POLICY


def _effective_train_epoch_samples(
    *,
    dataset_size: int,
    batch_size: int,
    distributed: _DistributedContext,
    full_batch_repeated: bool,
) -> tuple[int, int]:
    world_size = distributed.world_size if distributed.should_use_ddp else 1
    if full_batch_repeated:
        per_rank = len(
            fixed_selector_full_batch_indices(
                dataset_size=dataset_size,
                batch_size=batch_size,
                world_size=world_size,
                rank=0,
            ),
        )
        return per_rank * world_size, per_rank
    if distributed.should_use_ddp:
        per_rank = math.ceil(dataset_size / world_size)
        return per_rank * world_size, per_rank
    return dataset_size, dataset_size


def _place_model(
    *,
    model: NonEquivariantVAE,
    plan: SelectedRuntimePlan,
    device: torch.device,
) -> NonEquivariantVAE:
    if plan.memory_format == "channels_last":
        model.to(  # pyright: ignore[reportCallIssue]
            device=device,
            memory_format=torch.channels_last,
        )
    else:
        model.to(device=device)
    return model


def _maybe_wrap_ddp(
    *,
    model: NonEquivariantVAE,
    distributed: _DistributedContext,
    plan: SelectedRuntimePlan,
) -> nn.Module:
    if not distributed.should_use_ddp:
        return model
    return DistributedDataParallel(
        model,
        device_ids=[distributed.local_rank],
        output_device=distributed.local_rank,
        static_graph=plan.ddp_static_graph,
        gradient_as_bucket_view=plan.ddp_gradient_as_bucket_view,
    )


def _amp_execution(
    *,
    plan: SelectedRuntimePlan,
    distributed: _DistributedContext,
    dry_run: bool,
) -> _AmpExecution:
    enabled = plan.amp_enabled and distributed.device.type == "cuda" and not dry_run
    scaler_enabled = plan.grad_scaler_enabled and enabled
    return _AmpExecution(
        enabled=enabled,
        grad_scaler_enabled=scaler_enabled,
        grad_scaler_init_scale=SELECTED_RUNTIME_AMP_GRAD_SCALER_INIT_SCALE,
        autocast_dtype=plan.autocast_dtype if enabled else "not_executed_local_cpu",
        requested_autocast_dtype=plan.autocast_dtype,
        local_amp_status=(
            EXPECTED_AMP_APPLICATION_STATUS
            if enabled and scaler_enabled
            else "not_executed_local_cpu"
        ),
    )


def _restore_checkpoint_if_requested(  # noqa: PLR0913
    *,
    request: SelectedRuntimeTrainRequest,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    numpy_generator: Generator,
    train_generator: torch.Generator,
    runtime_identity: _RuntimeIdentity,
    resolved: ResolvedConfig,
) -> LoadedCheckpoint | None:
    if request.resume is None:
        return None
    metadata = read_training_checkpoint_metadata(path=request.resume)
    validate_checkpoint_resume_metadata(
        metadata,
        expected_effective_config_sha256=resolved.effective_config_hash,
        expected_runtime_config_sha256=runtime_identity.sha256,
        expected_selected_row_id=runtime_identity.selected_row_id,
        expected_runtime_policy_id=runtime_identity.runtime_policy_id,
    )
    return load_training_checkpoint(
        path=request.resume,
        model=model,
        optimizer=optimizer,
        numpy_generator=numpy_generator,
        torch_generators={"train_data": train_generator},
        expected_effective_config_sha256=resolved.effective_config_hash,
        expected_runtime_config_sha256=runtime_identity.sha256,
        expected_selected_row_id=runtime_identity.selected_row_id,
        expected_runtime_policy_id=runtime_identity.runtime_policy_id,
    )


def _run_train_steps(  # noqa: PLR0913
    *,
    request: SelectedRuntimeTrainRequest,
    resolved: ResolvedConfig,
    settings: _RunnerSettings,
    plan: SelectedRuntimePlan,
    model: nn.Module,
    checkpoint_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    amp: _AmpExecution,
    data_surface: _DataSurface,
    distributed: _DistributedContext,
    numpy_generator: Generator,
    train_generator: torch.Generator,
    runtime_identity: _RuntimeIdentity,
    start_step: int,
    write_checkpoints: bool,
) -> tuple[
    tuple[CsvRow, ...],
    tuple[CheckpointMetadata, ...],
    _SelectedRuntimeStepResult,
]:
    rows: list[CsvRow] = []
    checkpoints: list[CheckpointMetadata] = []
    train_batches = _cycle_batches(data_surface.train_loader)
    last_result: _SelectedRuntimeStepResult | None = None
    successful_count = start_step
    attempt_count = 0
    max_attempts = (settings.max_train_steps - start_step) + max(
        10,
        settings.max_train_steps * 2,
    )
    while successful_count < settings.max_train_steps:
        attempt_count += 1
        if attempt_count > max_attempts:
            message = (
                "selected-runtime runner exceeded AMP skip retry budget "
                f"after {max_attempts} attempts"
            )
            raise RuntimeError(message)
        batch = next(train_batches)
        result = _run_train_step(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            settings=settings,
            plan=plan,
            amp=amp,
            batch=batch,
            optimizer_step_index=successful_count,
            successful_optimizer_update_count=successful_count + 1,
            device=distributed.device,
        )
        last_result = result
        if not result.amp_step_skipped:
            successful_count = result.successful_optimizer_update_count
        checkpoint_path = ""
        if (
            write_checkpoints
            and not result.amp_step_skipped
            and successful_count > 0
            and successful_count % settings.save_every_steps == 0
        ):
            checkpoint = _save_checkpoint(
                path=request.output_dir
                / "checkpoints"
                / f"step_{successful_count:06d}.pt",
                request=request,
                resolved=resolved,
                settings=settings,
                model=checkpoint_model,
                optimizer=optimizer,
                numpy_generator=numpy_generator,
                train_generator=train_generator,
                runtime_identity=runtime_identity,
                step=successful_count,
                metric_value=float(result.losses.l1_loss.detach().cpu().item()),
                scaler=scaler,
                amp=amp,
                distributed=distributed,
            )
            checkpoints.append(checkpoint)
            checkpoint_path = _relative_to_output(checkpoint.path, request.output_dir)
        rows.append(
            _metric_row(
                result=result,
                rank=distributed.rank,
                plan=plan,
                amp=amp,
                checkpoint_path=checkpoint_path,
                corruption_strategy=plan.corruption_strategy,
            ),
        )
    if last_result is None:
        message = "selected-runtime runner executed no train steps"
        raise RuntimeError(message)
    if write_checkpoints and not checkpoints:
        checkpoints.append(
            _save_checkpoint(
                path=request.output_dir / "checkpoints" / "step_000001.pt",
                request=request,
                resolved=resolved,
                settings=settings,
                model=checkpoint_model,
                optimizer=optimizer,
                numpy_generator=numpy_generator,
                train_generator=train_generator,
                runtime_identity=runtime_identity,
                step=settings.max_train_steps,
                metric_value=float(last_result.losses.l1_loss.detach().cpu().item()),
                scaler=scaler,
                amp=amp,
                distributed=distributed,
            ),
        )
    return tuple(rows), tuple(checkpoints), last_result


def _run_train_step(  # noqa: PLR0913, PLR0914
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    settings: _RunnerSettings,
    plan: SelectedRuntimePlan,
    amp: _AmpExecution,
    batch: PatchTrainingBatch,
    optimizer_step_index: int,
    successful_optimizer_update_count: int,
    device: torch.device,
) -> _SelectedRuntimeStepResult:
    clean_batch_cpu = normalize_uint8_batch(batch.images_uint8)
    corruption = corrupt_normalized_batch(
        clean_batch_cpu,
        profile=settings.corruption_profile,
        corruption_seed=settings.corruption_seed,
        split=batch.split,
        semantic_sample_keys=batch.semantic_sample_keys,
        corruption_step=optimizer_step_index,
        corruption_view=_TRAIN_CORRUPTION_VIEW,
        strategy=plan.corruption_strategy,
    )
    clean_batch = _to_device(clean_batch_cpu, device=device, plan=plan)
    input_batch = _to_device(corruption.corrupted, device=device, plan=plan)
    eps = _zero_eps(
        batch_size=input_batch.shape[0],
        settings=settings,
        device=device,
    )
    beta = beta_for_step(
        optimizer_step_index=optimizer_step_index,
        max_optimizer_steps=settings.max_train_steps,
        target_beta=settings.beta_target,
        warmup_fraction=settings.beta_warmup_fraction,
    )
    optimizer.zero_grad(set_to_none=plan.zero_grad_set_to_none)
    before_params = _clone_trainable_parameters(model)
    dtype = _autocast_dtype(plan.autocast_dtype)
    with torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=amp.enabled,
    ):
        output = cast("NonEquivariantVAE", model).forward(input_batch, eps=eps)
    losses = compute_vae_loss(
        output,
        clean_batch,
        beta=beta,
        ssim_weight=settings.ssim_weight,
    )
    if amp.grad_scaler_enabled:
        old_scale = float(scaler.get_scale())
        scaled_loss = scaler.scale(losses.loss)
        scaled_backward = cast("Callable[[], None]", scaled_loss.backward)
        scaled_backward()
        scaler.unscale_(optimizer)
    else:
        old_scale = 1.0
        backward = cast("Callable[[], None]", losses.loss.backward)
        backward()
    nonfinite_count = _nonfinite_gradient_count(model)
    grad_norm = _global_grad_norm(model)
    if settings.optimizer_config.gradient_clip_global_norm > 0.0:
        nn.utils.clip_grad_norm_(
            list(model.parameters()),
            max_norm=settings.optimizer_config.gradient_clip_global_norm,
            foreach=True,
        )
    if amp.grad_scaler_enabled:
        scaler.step(optimizer)
        scaler.update()
        amp_step_skipped = float(scaler.get_scale()) < old_scale
    else:
        optimizer.step()
        amp_step_skipped = False
    return _SelectedRuntimeStepResult(
        optimizer_step_index=optimizer_step_index,
        successful_optimizer_update_count=(
            optimizer_step_index
            if amp_step_skipped
            else successful_optimizer_update_count
        ),
        losses=losses,
        grad_norm=grad_norm,
        param_update_norm=_parameter_update_norm(model, before_params),
        nonfinite_count=nonfinite_count,
        batch_size=input_batch.shape[0],
        amp_step_skipped=amp_step_skipped,
        zero_grad_set_to_none=plan.zero_grad_set_to_none,
    )


def _to_device(
    tensor: torch.Tensor,
    *,
    device: torch.device,
    plan: SelectedRuntimePlan,
) -> torch.Tensor:
    moved = tensor.to(
        device=device,
        non_blocking=plan.dataloader_non_blocking_h2d and device.type == "cuda",
    )
    if plan.memory_format == "channels_last":
        moved = moved.contiguous(memory_format=torch.channels_last)
    return moved


def _autocast_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def _cycle_batches(
    loader: DataLoader[PatchTrainingBatch],
) -> Iterator[PatchTrainingBatch]:
    while True:
        yield from loader


def _save_checkpoint(  # noqa: PLR0913
    *,
    path: Path,
    request: SelectedRuntimeTrainRequest,
    resolved: ResolvedConfig,
    settings: _RunnerSettings,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    numpy_generator: Generator,
    train_generator: torch.Generator,
    runtime_identity: _RuntimeIdentity,
    step: int,
    metric_value: float,
    scaler: GradScaler,
    amp: _AmpExecution,
    distributed: _DistributedContext,
) -> CheckpointMetadata:
    return save_training_checkpoint(
        path=path,
        model=model,
        optimizer=optimizer,
        numpy_generator=numpy_generator,
        torch_generators={"train_data": train_generator},
        runtime_config_sha256=runtime_identity.sha256,
        selected_row_id=runtime_identity.selected_row_id,
        runtime_policy_id=runtime_identity.runtime_policy_id,
        run_name=settings.run_name,
        config_path=request.config_path,
        config_sha256=resolved.invoked_config_hash,
        effective_config_sha256=resolved.effective_config_hash,
        optimizer_step=step,
        successful_optimizer_update_count=step,
        metric_name="l1_loss",
        metric_value=metric_value,
        amp_scaler_state=_amp_scaler_checkpoint_state(scaler=scaler, amp=amp),
        torch_cuda_rng_state=_cuda_rng_checkpoint_state(distributed),
        ddp_sampler_progress_state=_ddp_progress_checkpoint_state(
            distributed=distributed,
            step=step,
        ),
    )


def _amp_scaler_checkpoint_state(
    *,
    scaler: GradScaler,
    amp: _AmpExecution,
) -> dict[str, object] | None:
    if not amp.grad_scaler_enabled:
        return None
    return {
        "status": "selected_runtime_amp_scaler_state",
        "enabled": True,
        "state_dict": scaler.state_dict(),
    }


def _cuda_rng_checkpoint_state(
    distributed: _DistributedContext,
) -> dict[str, object] | None:
    if distributed.device.type != "cuda":
        return None
    return {
        "status": "selected_runtime_cuda_rng_state",
        "rank": distributed.rank,
        "world_size": distributed.world_size,
        "device_count": torch.cuda.device_count(),
        "states": tuple(
            state.detach().cpu() for state in torch.cuda.get_rng_state_all()
        ),
    }


def _ddp_progress_checkpoint_state(
    *,
    distributed: _DistributedContext,
    step: int,
) -> dict[str, object] | None:
    if not distributed.should_use_ddp:
        return None
    return {
        "status": "selected_runtime_ddp_sampler_progress",
        "rank": distributed.rank,
        "world_size": distributed.world_size,
        "successful_optimizer_update_count": step,
    }


def _checkpoint_resume_proof(  # noqa: PLR0913
    *,
    checkpoint: CheckpointMetadata,
    request: SelectedRuntimeTrainRequest,
    resolved: ResolvedConfig,
    settings: _RunnerSettings,
    runtime_identity: _RuntimeIdentity,
    train_generator: torch.Generator,
    amp: _AmpExecution,
    distributed: _DistributedContext,
) -> JsonObject:
    model = build_non_equivariant_vae(norm_groups=settings.norm_groups)
    optimizer, _ = create_adamw_optimizer(model, config=settings.optimizer_config)
    numpy_generator = np.random.default_rng(settings.global_seed)
    loaded = load_training_checkpoint(
        path=checkpoint.path,
        model=model,
        optimizer=optimizer,
        numpy_generator=numpy_generator,
        torch_generators={"train_data": train_generator},
        expected_effective_config_sha256=resolved.effective_config_hash,
        expected_runtime_config_sha256=runtime_identity.sha256,
        expected_selected_row_id=runtime_identity.selected_row_id,
        expected_runtime_policy_id=runtime_identity.runtime_policy_id,
    )
    amp_status_ok = loaded.amp_scaler_state_status == _expected_amp_scaler_status(amp)
    cuda_status_ok = loaded.torch_cuda_rng_state_status == _expected_cuda_rng_status(
        distributed,
    )
    ddp_status_ok = (
        loaded.ddp_sampler_progress_state_status
        == _expected_ddp_progress_status(distributed)
    )
    additional_steps = (
        settings.max_train_steps - loaded.successful_optimizer_update_count
    )
    return cast(
        "JsonObject",
        {
            "status": _LOCAL_STATUS
            if amp_status_ok and cuda_status_ok and ddp_status_ok
            else _FAIL,
            "status_scope": _STATUS_SCOPE,
            "full_run_eligible": False,
            "resume_checkpoint": _relative_to_output(
                checkpoint.path,
                request.output_dir,
            ),
            "resume_checkpoint_sha256": checkpoint.sha256,
            "loaded_schema_version": loaded.schema_version,
            "loaded_runtime_config_sha256": loaded.runtime_config_sha256,
            "current_runtime_config_sha256": runtime_identity.sha256,
            "loaded_selected_row_id": loaded.selected_row_id,
            "current_selected_row_id": runtime_identity.selected_row_id,
            "loaded_runtime_policy_id": loaded.runtime_policy_id,
            "current_runtime_policy_id": runtime_identity.runtime_policy_id,
            "loaded_successful_optimizer_update_count": (
                loaded.successful_optimizer_update_count
            ),
            "final_optimizer_step": settings.max_train_steps,
            "additional_optimizer_steps": max(0, additional_steps),
            "model_state_restored": True,
            "optimizer_state_restored": True,
            "python_rng_state_restored": True,
            "numpy_generator_state_restored": True,
            "torch_cpu_rng_state_restored": True,
            "torch_generator_states_restored": True,
            "torch_generator_names_restored": list(loaded.torch_generator_names),
            "torch_cuda_rng_state_status": loaded.torch_cuda_rng_state_status,
            "lr_scheduler_state_status": loaded.lr_scheduler_state_status,
            "beta_progress_state_status": loaded.beta_progress_state_status,
            "amp_scaler_state_status": loaded.amp_scaler_state_status,
            "amp_scaler_state_status_match": amp_status_ok,
            "ddp_sampler_progress_state_status": (
                loaded.ddp_sampler_progress_state_status
            ),
            "ddp_sampler_progress_state_status_match": ddp_status_ok,
            "torch_cuda_rng_state_status_match": cuda_status_ok,
            "config_sha256_match": (
                loaded.effective_config_sha256 == resolved.effective_config_hash
            ),
            "runtime_config_sha256_match": (
                loaded.runtime_config_sha256 == runtime_identity.sha256
            ),
            "selected_row_id_match": (
                loaded.selected_row_id == runtime_identity.selected_row_id
            ),
            "runtime_policy_id_match": (
                loaded.runtime_policy_id == runtime_identity.runtime_policy_id
            ),
        },
    )


def _resume_probe_checkpoint(
    *,
    checkpoints: Sequence[CheckpointMetadata],
    target_step: int,
) -> CheckpointMetadata:
    for checkpoint in checkpoints:
        if checkpoint.successful_optimizer_update_count < target_step:
            return checkpoint
    return checkpoints[-1]


def _loaded_checkpoint_resume_proof(  # noqa: PLR0913
    *,
    loaded: LoadedCheckpoint,
    request: SelectedRuntimeTrainRequest,
    resolved: ResolvedConfig,
    settings: _RunnerSettings,
    runtime_identity: _RuntimeIdentity,
    amp: _AmpExecution,
    distributed: _DistributedContext,
) -> JsonObject:
    amp_status_ok = loaded.amp_scaler_state_status == _expected_amp_scaler_status(amp)
    cuda_status_ok = loaded.torch_cuda_rng_state_status == _expected_cuda_rng_status(
        distributed,
    )
    ddp_status_ok = (
        loaded.ddp_sampler_progress_state_status
        == _expected_ddp_progress_status(distributed)
    )
    additional_steps = (
        settings.max_train_steps - loaded.successful_optimizer_update_count
    )
    identity_ok = (
        loaded.effective_config_sha256 == resolved.effective_config_hash
        and loaded.runtime_config_sha256 == runtime_identity.sha256
        and loaded.selected_row_id == runtime_identity.selected_row_id
        and loaded.runtime_policy_id == runtime_identity.runtime_policy_id
    )
    return cast(
        "JsonObject",
        {
            "status": _LOCAL_STATUS
            if amp_status_ok and cuda_status_ok and ddp_status_ok and identity_ok
            else _FAIL,
            "status_scope": _STATUS_SCOPE,
            "full_run_eligible": False,
            "resume_sequence": "loaded_checkpoint_before_training_continued",
            "resume_checkpoint": _relative_to_output(loaded.path, request.output_dir),
            "loaded_schema_version": loaded.schema_version,
            "loaded_runtime_config_sha256": loaded.runtime_config_sha256,
            "current_runtime_config_sha256": runtime_identity.sha256,
            "loaded_selected_row_id": loaded.selected_row_id,
            "current_selected_row_id": runtime_identity.selected_row_id,
            "loaded_runtime_policy_id": loaded.runtime_policy_id,
            "current_runtime_policy_id": runtime_identity.runtime_policy_id,
            "loaded_successful_optimizer_update_count": (
                loaded.successful_optimizer_update_count
            ),
            "final_optimizer_step": settings.max_train_steps,
            "additional_optimizer_steps": max(0, additional_steps),
            "model_state_restored": True,
            "optimizer_state_restored": True,
            "python_rng_state_restored": True,
            "numpy_generator_state_restored": True,
            "torch_cpu_rng_state_restored": True,
            "torch_generator_states_restored": True,
            "torch_generator_names_restored": list(loaded.torch_generator_names),
            "torch_cuda_rng_state_status": loaded.torch_cuda_rng_state_status,
            "lr_scheduler_state_status": loaded.lr_scheduler_state_status,
            "beta_progress_state_status": loaded.beta_progress_state_status,
            "amp_scaler_state_status": loaded.amp_scaler_state_status,
            "amp_scaler_state_status_match": amp_status_ok,
            "ddp_sampler_progress_state_status": (
                loaded.ddp_sampler_progress_state_status
            ),
            "ddp_sampler_progress_state_status_match": ddp_status_ok,
            "torch_cuda_rng_state_status_match": cuda_status_ok,
            "config_sha256_match": (
                loaded.effective_config_sha256 == resolved.effective_config_hash
            ),
            "runtime_config_sha256_match": (
                loaded.runtime_config_sha256 == runtime_identity.sha256
            ),
            "selected_row_id_match": (
                loaded.selected_row_id == runtime_identity.selected_row_id
            ),
            "runtime_policy_id_match": (
                loaded.runtime_policy_id == runtime_identity.runtime_policy_id
            ),
        },
    )


def _expected_amp_scaler_status(amp: _AmpExecution) -> str:
    if amp.grad_scaler_enabled:
        return "selected_runtime_amp_scaler_state"
    return "not_applicable_local_cpu_amp_disabled"


def _expected_cuda_rng_status(distributed: _DistributedContext) -> str:
    if distributed.device.type == "cuda":
        return "selected_runtime_cuda_rng_state"
    return "not_applicable_local_cpu"


def _expected_ddp_progress_status(distributed: _DistributedContext) -> str:
    if distributed.should_use_ddp:
        return "selected_runtime_ddp_sampler_progress"
    return "not_applicable_local_single_process"


def _write_reconstruction_sample(
    *,
    path: Path,
    model: nn.Module,
    settings: _RunnerSettings,
    data_surface: _DataSurface,
    device: torch.device,
) -> bool:
    batch = cast("PatchTrainingBatch", next(iter(data_surface.validation_loader)))
    clean_batch = clean_validation_passthrough(
        normalize_uint8_batch(batch.images_uint8),
    )
    clean_batch = clean_batch[:1].to(device=device)
    eps = torch.zeros(
        (
            clean_batch.shape[0],
            LATENT_CHANNELS,
            settings.image_size // 8,
            settings.image_size // 8,
        ),
        dtype=torch.float32,
        device=device,
    )
    model.eval()
    with torch.no_grad():
        output = cast("NonEquivariantVAE", model).forward(clean_batch, eps=eps)
    model.train()
    payload = {
        "target": clean_batch.detach().cpu(),
        "reconstruction": output.reconstruction.detach().cpu(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return bool(clean_batch.numel() > 0 and torch.isfinite(output.reconstruction).all())


def _gate_health_rows(  # noqa: PLR0913
    *,
    run_name: str,
    plan: SelectedRuntimePlan,
    probe: SelectedRuntimeEnvironmentProbe,
    amp: _AmpExecution,
    model: nn.Module,
    optimizer_step: int,
    rank: int | None = None,
) -> tuple[CsvRow, ...]:
    rows: list[CsvRow] = []
    named_modules = cast("Iterable[tuple[str, nn.Module]]", model.named_modules())
    for module_name, module in named_modules:
        if not isinstance(module, GatedScalarActivation):
            continue
        gate = torch.sigmoid(module.a.detach().float() + module.b.detach().float())
        a_grad = module.a.grad.detach().float() if module.a.grad is not None else None
        b_grad = module.b.grad.detach().float() if module.b.grad is not None else None
        row: CsvRow = {
            "run_name": run_name,
            "benchmark_kind": _BENCHMARK_KIND,
            "benchmark_source": (
                _BENCHMARK_SOURCE if rank is None else f"{_BENCHMARK_SOURCE}_rank{rank}"
            ),
            "full_run_eligible": "false",
            "accelerator_mode": probe.accelerator_mode,
            "machine_shape": probe.machine_shape,
            "row_id": plan.selected_row_id,
            "candidate_row_id": plan.selected_row_id,
            "runtime_policy_id": plan.runtime_policy_id,
            "optimizer_step": str(optimizer_step),
            "module": module_name if rank is None else f"rank{rank}:{module_name}",
            "gate_kind": "gated_scalar_activation",
            "num_channels": str(module.channels),
            "num_elements": str(module.channels),
            "a_min": _format_float(_tensor_stat(module.a, "min")),
            "a_max": _format_float(_tensor_stat(module.a, "max")),
            "a_mean": _format_float(_tensor_stat(module.a, "mean")),
            "a_std": _format_float(_tensor_stat(module.a, "std")),
            "b_min": _format_float(_tensor_stat(module.b, "min")),
            "b_max": _format_float(_tensor_stat(module.b, "max")),
            "b_mean": _format_float(_tensor_stat(module.b, "mean")),
            "b_std": _format_float(_tensor_stat(module.b, "std")),
            "max_abs_a": _format_float(float(module.a.detach().abs().max().item())),
            "max_abs_b": _format_float(float(module.b.detach().abs().max().item())),
            "gate_mean": _format_float(float(gate.mean().item())),
            "gate_std": _format_float(_std(gate)),
            "gate_p01": _format_float(_quantile(gate, 0.01)),
            "gate_p50": _format_float(_quantile(gate, 0.50)),
            "gate_p99": _format_float(_quantile(gate, 0.99)),
            "frac_gate_lt_0_01": _format_float(
                _frac(gate < _GATE_LOW_SATURATION_THRESHOLD),
            ),
            "frac_gate_gt_0_99": _format_float(
                _frac(gate > _GATE_HIGH_SATURATION_THRESHOLD),
            ),
            "worst_channel_frac_gate_lt_0_01": _format_float(
                _frac(gate < _GATE_LOW_SATURATION_THRESHOLD),
            ),
            "worst_channel_frac_gate_gt_0_99": _format_float(
                _frac(gate > _GATE_HIGH_SATURATION_THRESHOLD),
            ),
            "dead_channel_count": str(
                int(
                    torch.count_nonzero(
                        gate < _GATE_LOW_SATURATION_THRESHOLD,
                    ).item(),
                ),
            ),
            "input_rms": "",
            "output_rms": "",
            "output_input_rms_ratio": "",
            "a_grad_norm": "" if a_grad is None else _format_float(_norm(a_grad)),
            "b_grad_norm": "" if b_grad is None else _format_float(_norm(b_grad)),
            "a_update_to_param_norm": "",
            "b_update_to_param_norm": "",
            "gate_force_fp32": _csv_bool(value=module.force_fp32),
            "input_dtype": module.last_input_dtype,
            "gate_math_dtype": module.last_gate_math_dtype,
            "gate_tensor_dtype": module.last_gate_tensor_dtype,
            "output_dtype": module.last_output_dtype,
            "requested_autocast_dtype": amp.requested_autocast_dtype,
            "precision_proof_status": module.last_precision_proof_status,
            "gate_health_status": "pass",
        }
        rows.append(row)
    return tuple(rows)


def _gate_health_summary(gate_rows: Sequence[CsvRow]) -> JsonObject:
    row_count = len(gate_rows)
    pass_count = sum(1 for row in gate_rows if row.get("gate_health_status") == "pass")
    return cast(
        "JsonObject",
        {
            "status": _LOCAL_STATUS
            if row_count > 0 and pass_count == row_count
            else _FAIL,
            "status_scope": _STATUS_SCOPE,
            "full_run_eligible": False,
            "rows_written": row_count,
            "pass_count": pass_count,
            "failure_kind": ""
            if row_count > 0 and pass_count == row_count
            else "no_gate_rows",
        },
    )


def _plan_applied_proof(  # noqa: PLR0913
    *,
    plan: SelectedRuntimePlan,
    settings: _RunnerSettings,
    probe: SelectedRuntimeEnvironmentProbe,
    amp: _AmpExecution,
    ddp_proof: JsonObject,
    metric_rows: Sequence[CsvRow],
    last_result: _SelectedRuntimeStepResult,
) -> JsonObject:
    ddp_pass = ddp_proof.get("status") == _LOCAL_STATUS
    observed = SelectedRuntimeApplicationObservation(
        selected_row_id=plan.selected_row_id,
        runtime_policy_id=plan.runtime_policy_id,
        accelerator_mode=probe.accelerator_mode,
        machine_shape=probe.machine_shape,
        world_size=probe.world_size,
        nproc_per_node=probe.nproc_per_node,
        torchrun_standalone=probe.torchrun_standalone,
        batch_size=_observed_batch_size(metric_rows, fallback=settings.batch_size),
        global_batch_size=(
            settings.batch_size * probe.world_size if ddp_pass else settings.batch_size
        ),
        amp_enabled=amp.enabled,
        grad_scaler_enabled=amp.grad_scaler_enabled,
        fp32_loss=True,
        autocast_dtype=amp.autocast_dtype,
        torch_compile_enabled=False,
        compile_scope="none",
        dataloader_num_workers=plan.dataloader_num_workers,
        dataloader_prefetch_factor=plan.dataloader_prefetch_factor,
        dataloader_pin_memory=plan.dataloader_pin_memory,
        dataloader_persistent_workers=plan.dataloader_persistent_workers,
        dataloader_non_blocking_h2d=plan.dataloader_non_blocking_h2d,
        corruption_strategy=_observed_corruption_strategy(
            metric_rows,
            fallback=plan.corruption_strategy,
        ),
        memory_format=plan.memory_format,
        ddp_static_graph=plan.ddp_static_graph,
        ddp_gradient_as_bucket_view=plan.ddp_gradient_as_bucket_view,
        zero_grad_set_to_none=last_result.zero_grad_set_to_none,
        local_ddp_status=(
            EXPECTED_DDP_APPLICATION_STATUS
            if ddp_pass
            else "not_executed_local_cpu_mechanics_only"
        ),
        local_amp_status=amp.local_amp_status,
        runner_amp_grad_scaler_init_scale=amp.grad_scaler_init_scale,
    )
    return build_plan_applied_proof(
        plan=plan,
        observed=observed,
        status_scope=_STATUS_SCOPE,
    )


def _local_readiness_summary(
    components: _LocalReadinessComponents,
) -> JsonObject:
    remote_ready = (
        components.plan_applied.get("status") == _LOCAL_STATUS
        and components.checkpoint_resume_proof.get("status") == _LOCAL_STATUS
        and components.gate_health_summary.get("status") == _LOCAL_STATUS
        and components.data_source == "ubc-pre-shuffled"
        and components.amp_step_skipped_count == 0
        and components.nonfinite_count == 0
    )
    blockers: list[str] = []
    if components.plan_applied.get("status") != _LOCAL_STATUS:
        blockers.append("missing_real_dual_t4_amp_plan_applied_proof")
    if components.data_source != "ubc-pre-shuffled":
        blockers.append("dry_run_synthetic_data_non_promotable")
    if components.ddp_proof.get("status") != _LOCAL_STATUS:
        blockers.append("missing_dual_t4_ddp_rank_device_proof")
    if components.amp_step_skipped_count > 0:
        blockers.append("amp_step_skip_observed")
    if components.nonfinite_count > 0:
        blockers.append("nonfinite_train_metric_observed")
    blockers.append("fixed_32_selector_real_false_until_spec0008")
    return cast(
        "JsonObject",
        {
            "status": _LOCAL_STATUS if remote_ready else _FAIL,
            "status_scope": _STATUS_SCOPE,
            "full_run_eligible": False,
            "remote_pass_ready": False,
            "real_train_runner_implemented": True,
            "fixed_32_selector_real": False,
            "component_status": {
                "selected_runtime_plan_applied": _string_value(
                    components.plan_applied.get("status"),
                ),
                "checkpoint_resume": _string_value(
                    components.checkpoint_resume_proof.get("status"),
                ),
                "gate_health": _string_value(
                    components.gate_health_summary.get("status"),
                ),
                "ubc_data_surface": components.data_source,
                "ddp_rank_device": _string_value(components.ddp_proof.get("status")),
                "amp_step_skipped_count": components.amp_step_skipped_count,
                "nonfinite_count": components.nonfinite_count,
            },
            "launch_blockers_remaining": blockers,
            "failure_kind": "" if remote_ready else "local_non_promotable",
        },
    )


def _training_summary(  # noqa: PLR0913
    *,
    request: SelectedRuntimeTrainRequest,
    resolved: ResolvedConfig,
    settings: _RunnerSettings,
    plan: SelectedRuntimePlan,
    runtime_identity: _RuntimeIdentity,
    launch_command: SelectedRuntimeLaunchCommand,
    ddp_proof: JsonObject,
    amp: _AmpExecution,
    data_surface: _DataSurface,
    metric_rows: Sequence[CsvRow],
    checkpoints: Sequence[CheckpointMetadata],
    final_checkpoint: CheckpointMetadata,
    best_checkpoint: CheckpointMetadata,
    last_result: _SelectedRuntimeStepResult,
    plan_applied: JsonObject,
    checkpoint_resume_proof: JsonObject,
    gate_health_summary: JsonObject,
    reconstruction_nonblank: bool,
) -> JsonObject:
    return cast(
        "JsonObject",
        {
            "status": _LOCAL_STATUS
            if _nonfinite_metric_count(metric_rows) == 0
            else _FAIL,
            "status_scope": _STATUS_SCOPE,
            "proof_scope": _STATUS_SCOPE,
            "full_run_eligible": False,
            "run_name": settings.run_name,
            "run_mode": settings.run_mode,
            "data": request.data,
            "data_root": str(data_surface.root),
            "synthetic_generated": data_surface.synthetic_generated,
            "config_path": str(request.config_path),
            "config_sha256": resolved.invoked_config_hash,
            "effective_config_sha256": resolved.effective_config_hash,
            "runtime_config": {
                "path": str(runtime_identity.path),
                "sha256": runtime_identity.sha256,
                "selected_row_id": runtime_identity.selected_row_id,
                "runtime_policy_id": runtime_identity.runtime_policy_id,
                "per_device_batch_size": plan.per_device_batch_size,
                "global_batch_size": plan.global_batch_size,
                "precision_policy": plan.precision_policy,
                "corruption_strategy": plan.corruption_strategy,
                "consumed": True,
            },
            "selected_runtime_launch_command": launch_command.shell_command,
            "ddp_rank_device_proof": ddp_proof,
            "amp_execution": {
                "enabled": amp.enabled,
                "grad_scaler_enabled": amp.grad_scaler_enabled,
                "grad_scaler_init_scale": amp.grad_scaler_init_scale,
                "autocast_dtype": amp.autocast_dtype,
                "requested_autocast_dtype": amp.requested_autocast_dtype,
                "local_amp_status": amp.local_amp_status,
                "fp32_objective_island": True,
            },
            "max_train_steps": settings.max_train_steps,
            "max_val_steps": settings.max_val_steps,
            "save_every_steps": settings.save_every_steps,
            "optimizer_steps_completed": _successful_optimizer_update_count(
                metric_rows,
            ),
            "metric_row_count": len(metric_rows),
            "amp_step_skipped_count": sum(
                1 for row in metric_rows if row["amp_step_skipped"] == "1"
            ),
            "fixed_train_patches": ""
            if data_surface.fixed_train_patches is None
            else str(data_surface.fixed_train_patches),
            "fixed_train_patches_sha256": data_surface.fixed_train_patches_sha256,
            "fixed_train_patch_count": data_surface.fixed_train_patch_count,
            "train_sampler_policy": data_surface.train_sampler_policy,
            "train_effective_global_epoch_samples": (
                data_surface.train_effective_global_epoch_samples
            ),
            "train_effective_per_rank_epoch_samples": (
                data_surface.train_effective_per_rank_epoch_samples
            ),
            "fixed_train_repeated_to_full_batch": (
                data_surface.train_sampler_policy
                == _FIXED32_TINY_FULL_BATCH_SAMPLER_POLICY
            ),
            "checkpoint_count": len(checkpoints),
            "final_checkpoint": _checkpoint_payload(
                final_checkpoint,
                request.output_dir,
            ),
            "best_checkpoint": _checkpoint_payload(best_checkpoint, request.output_dir),
            "metrics_csv": "metrics/train_steps.csv",
            "train_steps_csv": "metrics/train_steps.csv",
            "gate_health_csv": "metrics/gate_health.csv",
            "selected_runtime_plan_applied": (
                "benchmark/selected_runtime_plan_applied.json"
            ),
            "checkpoint_resume_proof": "benchmark/checkpoint_resume_proof.json",
            "gate_health_summary": "benchmark/gate_health_summary.json",
            "artifact_manifest": "benchmark/artifact_manifest.json",
            "plan_applied_status": _string_value(plan_applied.get("status")),
            "checkpoint_resume_status": _string_value(
                checkpoint_resume_proof.get("status"),
            ),
            "gate_health_status": _string_value(gate_health_summary.get("status")),
            "reconstruction_sample_nonblank": reconstruction_nonblank,
            "last_loss": cast("JsonObject", last_result.losses.detached_scalars()),
            "nonfinite_count": _nonfinite_metric_count(metric_rows),
        },
    )


def _selected_runtime_debug_summary(  # noqa: PLR0913
    *,
    plan: SelectedRuntimePlan,
    settings: _RunnerSettings,
    plan_applied: JsonObject,
    ddp_proof: JsonObject,
    amp: _AmpExecution,
    checkpoint_resume_proof: JsonObject,
    gate_health_summary: JsonObject,
    data_surface: _DataSurface,
) -> JsonObject:
    remote_blockers: list[str] = []
    if plan_applied.get("status") != _LOCAL_STATUS:
        remote_blockers.append("missing_real_dual_t4_amp_plan_applied_proof")
    if data_surface.source != "ubc-pre-shuffled":
        remote_blockers.append("synthetic_dry_run_non_promotable")
    remote_blockers.append("fixed_32_selector_real_false_until_spec0008")
    return cast(
        "JsonObject",
        {
            "status": _LOCAL_STATUS,
            "status_scope": _STATUS_SCOPE,
            "full_run_eligible": False,
            "real_train_runner_implemented": True,
            "remote_pass_ready": False,
            "fixed_32_selector_real": False,
            "selected_row_id": plan.selected_row_id,
            "runtime_policy_id": plan.runtime_policy_id,
            "per_device_batch_size": settings.batch_size,
            "corruption_strategy": plan.corruption_strategy,
            "selected_runtime_plan_applied_status": _string_value(
                plan_applied.get("status"),
            ),
            "ddp_rank_device_status": _string_value(ddp_proof.get("status")),
            "amp_execution_status": amp.local_amp_status,
            "checkpoint_resume_proof_status": _string_value(
                checkpoint_resume_proof.get("status"),
            ),
            "gate_health_status": _string_value(gate_health_summary.get("status")),
            "fixed_train_patches": ""
            if data_surface.fixed_train_patches is None
            else str(data_surface.fixed_train_patches),
            "fixed_train_patches_sha256": data_surface.fixed_train_patches_sha256,
            "fixed_train_patch_count": data_surface.fixed_train_patch_count,
            "train_sampler_policy": data_surface.train_sampler_policy,
            "train_effective_global_epoch_samples": (
                data_surface.train_effective_global_epoch_samples
            ),
            "train_effective_per_rank_epoch_samples": (
                data_surface.train_effective_per_rank_epoch_samples
            ),
            "fixed_train_repeated_to_full_batch": (
                data_surface.train_sampler_policy
                == _FIXED32_TINY_FULL_BATCH_SAMPLER_POLICY
            ),
            "uses_resolve_patch_data_paths": True,
            "uses_patch_training_dataset": True,
            "uses_collate_patch_training_samples": True,
            "uses_normalize_uint8_batch": True,
            "artifact_manifest": "benchmark/artifact_manifest.json",
            "real_kaggle_debug_status": "pending_permission_gated_remote_run",
            "launch_blockers_remaining": remote_blockers,
        },
    )


def _tiny_overfit_summary(
    *,
    runtime_identity: _RuntimeIdentity,
    corruption_strategy: str,
    data_surface: _DataSurface,
    metric_rows: Sequence[CsvRow],
    gate_health_summary: JsonObject,
) -> JsonObject:
    successful_rows = _successful_metric_rows(metric_rows)
    l1_values = [float(row["l1_loss"]) for row in successful_rows]
    recon_values = [float(row["recon_loss"]) for row in successful_rows]
    batch_size_values = _batch_size_values(successful_rows)
    smoothing_window = min(_TINY_SMOOTHING_WINDOW, len(successful_rows))
    initial_l1 = _mean(l1_values[:smoothing_window])
    final_l1 = _mean(l1_values[-smoothing_window:])
    initial_recon = _mean(recon_values[:smoothing_window])
    final_recon = _mean(recon_values[-smoothing_window:])
    l1_improvement = _improvement_fraction(initial_l1, final_l1)
    recon_improvement = _improvement_fraction(initial_recon, final_recon)
    nonfinite_count = sum(int(row.get("nonfinite_count", "0")) for row in metric_rows)
    amp_skip_count = _amp_step_skipped_count(metric_rows)
    status = (
        _LOCAL_STATUS
        if data_surface.fixed_train_patch_count == FIXED_32_TRAIN_OVERFIT_COUNT
        and _successful_optimizer_update_count(metric_rows) <= _TINY_MAX_OPTIMIZER_STEPS
        and _successful_optimizer_update_count(metric_rows) > 0
        and amp_skip_count == 0
        and nonfinite_count == 0
        and l1_improvement >= _TINY_MIN_IMPROVEMENT_FRACTION
        and recon_improvement >= _TINY_MIN_IMPROVEMENT_FRACTION
        and gate_health_summary.get("status") == _LOCAL_STATUS
        else _FAIL
    )
    return cast(
        "JsonObject",
        {
            "status": status,
            "status_scope": _STATUS_SCOPE,
            "full_run_eligible": False,
            "runtime_config": {
                "path": str(runtime_identity.path),
                "sha256": runtime_identity.sha256,
                "selected_row_id": runtime_identity.selected_row_id,
                "runtime_policy_id": runtime_identity.runtime_policy_id,
            },
            "fixed_train_patches": ""
            if data_surface.fixed_train_patches is None
            else str(data_surface.fixed_train_patches),
            "fixed_train_patches_sha256": data_surface.fixed_train_patches_sha256,
            "patch_count": data_surface.fixed_train_patch_count,
            "train_sampler_policy": data_surface.train_sampler_policy,
            "train_effective_global_epoch_samples": (
                data_surface.train_effective_global_epoch_samples
            ),
            "train_effective_per_rank_epoch_samples": (
                data_surface.train_effective_per_rank_epoch_samples
            ),
            "fixed_train_repeated_to_full_batch": (
                data_surface.train_sampler_policy
                == _FIXED32_TINY_FULL_BATCH_SAMPLER_POLICY
            ),
            "optimizer_steps": _successful_optimizer_update_count(metric_rows),
            "metric_row_count": len(metric_rows),
            "successful_metric_row_count": len(successful_rows),
            "amp_step_skipped_count": amp_skip_count,
            "nonfinite_count": nonfinite_count,
            "grad_scaler_init_scale": SELECTED_RUNTIME_AMP_GRAD_SCALER_INIT_SCALE,
            "observed_batch_sizes": batch_size_values,
            "smoothing_window_steps": smoothing_window,
            "corruption_strategy": _observed_corruption_strategy(
                metric_rows,
                fallback=corruption_strategy,
            ),
            "eval_views": ["train_clean", "train_corrupted_fixed_seed"],
            "initial_smoothed_l1": initial_l1,
            "final_smoothed_l1": final_l1,
            "initial_smoothed_recon_loss": initial_recon,
            "final_smoothed_recon_loss": final_recon,
            "l1_improvement_fraction": l1_improvement,
            "recon_loss_improvement_fraction": recon_improvement,
            "gate_health_status": _string_value(gate_health_summary.get("status")),
            "real_tiny_overfit_status": "pass" if status == _LOCAL_STATUS else "fail",
            "failure_kind": ""
            if status == _LOCAL_STATUS
            else "tiny_overfit_bounds_not_met",
        },
    )


def _writes_tiny_summary(settings: _RunnerSettings) -> bool:
    return settings.run_mode == _TINY_RUN_MODE


def _artifact_manifest(
    *,
    artifacts: _RunArtifacts,
    settings: _RunnerSettings,
    checkpoints: Sequence[CheckpointMetadata],
    metric_rows: Sequence[CsvRow],
    reconstruction_nonblank: bool,
) -> JsonObject:
    artifact_paths = {
        "training_summary": artifacts.training_summary,
        "selected_runtime_debug_summary": artifacts.selected_runtime_debug_summary,
        "selected_runtime_plan_applied": artifacts.selected_runtime_plan_applied,
        "checkpoint_resume_proof": artifacts.checkpoint_resume_proof,
        "gate_health_summary": artifacts.gate_health_summary,
        "local_selected_runtime_readiness": artifacts.local_readiness,
        "train_steps": artifacts.train_steps,
        "gate_health": artifacts.gate_health,
        "reconstruction_samples": artifacts.reconstruction_samples,
    }
    if _writes_tiny_summary(settings):
        artifact_paths["tiny_overfit_summary"] = artifacts.tiny_overfit_summary
    for checkpoint in checkpoints:
        artifact_paths[f"checkpoint:{checkpoint.path.name}"] = checkpoint.path
    missing = [
        name for name, path in sorted(artifact_paths.items()) if not path.exists()
    ]
    return cast(
        "JsonObject",
        {
            "status": _LOCAL_STATUS if not missing else _FAIL,
            "status_scope": _STATUS_SCOPE,
            "full_run_eligible": False,
            "artifact_hashes": cast(
                "JsonObject",
                {
                    name: _sha256_file(path)
                    for name, path in sorted(artifact_paths.items())
                    if path.exists()
                },
            ),
            "missing_artifacts": missing,
            "checkpoint_count": len(checkpoints),
            "metric_row_count": len(metric_rows),
            "reconstruction_sample_nonblank": reconstruction_nonblank,
        },
    )


def _metric_row(  # noqa: PLR0913
    *,
    result: _SelectedRuntimeStepResult,
    rank: int,
    plan: SelectedRuntimePlan,
    amp: _AmpExecution,
    checkpoint_path: str,
    corruption_strategy: str,
) -> CsvRow:
    scalars = result.losses.detached_scalars()
    return {
        "event_id": (
            f"rank{rank}_train_step_{result.successful_optimizer_update_count:06d}"
        ),
        "rank": str(rank),
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
        "batch_size": str(result.batch_size),
        "precision_policy": plan.precision_policy,
        "amp_enabled": _csv_bool(value=amp.enabled),
        "autocast_dtype": amp.autocast_dtype,
        "grad_scaler_enabled": _csv_bool(value=amp.grad_scaler_enabled),
        "fp32_loss": _csv_bool(value=plan.fp32_loss),
        "torch_compile_enabled": _csv_bool(value=plan.torch_compile_enabled),
        "compile_scope": plan.compile_scope,
        "corruption_strategy": corruption_strategy,
        "amp_step_skipped": "1" if result.amp_step_skipped else "0",
        "checkpoint_path": checkpoint_path,
    }


def _observed_batch_size(rows: Sequence[CsvRow], *, fallback: int) -> int:
    values = {int(row["batch_size"]) for row in rows if row.get("batch_size")}
    return values.pop() if len(values) == 1 else fallback


def _observed_corruption_strategy(rows: Sequence[CsvRow], *, fallback: str) -> str:
    values = {
        row["corruption_strategy"] for row in rows if row.get("corruption_strategy")
    }
    return values.pop() if len(values) == 1 else fallback


def _successful_optimizer_update_count(rows: Sequence[CsvRow]) -> int:
    values = [
        int(row["successful_optimizer_update_count"])
        for row in rows
        if row.get("amp_step_skipped") == "0"
        and row.get("successful_optimizer_update_count")
    ]
    return max(values) if values else 0


def _successful_metric_rows(rows: Sequence[CsvRow]) -> tuple[CsvRow, ...]:
    return tuple(row for row in rows if row.get("amp_step_skipped") == "0")


def _amp_step_skipped_count(rows: Sequence[CsvRow]) -> int:
    return sum(1 for row in rows if row.get("amp_step_skipped") == "1")


def _nonfinite_metric_count(rows: Sequence[CsvRow]) -> int:
    return sum(int(row.get("nonfinite_count", "0")) for row in rows)


def _batch_size_values(rows: Sequence[CsvRow]) -> tuple[int, ...]:
    return tuple(
        sorted({int(row["batch_size"]) for row in rows if row.get("batch_size")}),
    )


def _best_l1(rows: Sequence[CsvRow]) -> float:
    values = [float(row["l1_loss"]) for row in rows if row["amp_step_skipped"] == "0"]
    return min(values) if values else 0.0


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _improvement_fraction(initial: float, final: float) -> float:
    if initial <= 0.0:
        return 0.0
    return (initial - final) / initial


def _clone_trainable_parameters(model: nn.Module) -> tuple[torch.Tensor, ...]:
    return tuple(
        parameter.detach().clone()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def _global_grad_norm(model: nn.Module) -> float:
    squared: float = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().float()
        squared += _float_item(torch.sum(grad.square()))
    return math.sqrt(squared)


def _nonfinite_gradient_count(model: nn.Module) -> int:
    count = 0
    for parameter in model.parameters():
        if parameter.grad is not None:
            count += int(torch.count_nonzero(~torch.isfinite(parameter.grad)).item())
    return count


def _parameter_update_norm(
    model: nn.Module,
    before_params: Sequence[torch.Tensor],
) -> float:
    squared: float = 0.0
    index = 0
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        delta = parameter.detach().float().cpu() - before_params[index].float().cpu()
        squared += _float_item(torch.sum(delta.square()))
        index += 1
    return math.sqrt(squared)


def _zero_eps(
    *,
    batch_size: int,
    settings: _RunnerSettings,
    device: torch.device,
) -> torch.Tensor:
    return torch.zeros(
        (
            batch_size,
            LATENT_CHANNELS,
            settings.image_size // 8,
            settings.image_size // 8,
        ),
        dtype=torch.float32,
        device=device,
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


def _cleanup_distributed(distributed: _DistributedContext) -> None:
    if distributed.initialized_here and dist.is_initialized():
        dist.destroy_process_group()


def _close_data_surface(data_surface: _DataSurface) -> None:
    data_surface.train_dataset.close()
    data_surface.validation_dataset.close()


def _relative_to_output(path: Path, output_dir: Path) -> str:
    try:
        return str(path.relative_to(output_dir))
    except ValueError:
        return str(path)


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


def _norm_groups(resolved: ResolvedConfig) -> int:
    model = _required_object(resolved.effective_config, "model")
    normalization = _required_object(model, "normalization")
    return _optional_int(normalization, "num_groups") or DEFAULT_GROUPNORM_GROUPS


def _seed(effective: JsonObject, name: str) -> int:
    seeds = _optional_object(effective, "seeds") or {}
    return _optional_int(seeds, name) or 20260610


def _validate_settings(settings: _RunnerSettings) -> None:
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
    message = f"Expected object config field {key}"
    raise TypeError(message)


def _required_str(payload: JsonObject, key: str) -> str:
    value = payload.get(key)
    if isinstance(value, str):
        return value
    message = f"Expected string config field {key}"
    raise TypeError(message)


def _optional_int(payload: JsonObject, key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        message = f"Expected integer config field {key}"
        raise TypeError(message)
    return value


def _required_float(payload: JsonObject, key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool):
        message = f"Expected numeric config field {key}"
        raise TypeError(message)
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
        message = f"Expected {expected_len} numeric entries in config field {key}"
        raise TypeError(message)
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int | float):
            message = f"Expected numeric entries in config field {key}"
            raise TypeError(message)
        result.append(float(item))
    return tuple(result)


def _tensor_stat(tensor: torch.Tensor, stat: str) -> float:
    values = tensor.detach().float()
    if stat == "min":
        return float(values.min().item())
    if stat == "max":
        return float(values.max().item())
    if stat == "std":
        return _std(values)
    return float(values.mean().item())


def _std(tensor: torch.Tensor) -> float:
    if tensor.numel() <= 1:
        return 0.0
    return float(tensor.detach().float().std(unbiased=False).item())


def _quantile(tensor: torch.Tensor, q: float) -> float:
    return float(torch.quantile(tensor.detach().float(), q).item())


def _frac(mask: torch.Tensor) -> float:
    if mask.numel() == 0:
        return 0.0
    return float(mask.float().mean().item())


def _norm(tensor: torch.Tensor) -> float:
    values = tensor.detach().float().reshape(-1)
    squared: float = 0.0
    for index in range(values.numel()):
        scalar = _float_item(values[index])
        squared += scalar * scalar
    return math.sqrt(squared)


def _float_item(tensor: torch.Tensor) -> float:
    return float(cast("float", tensor.item()))


def _format_float(value: float) -> str:
    return f"{value:.10g}"


def _csv_bool(*, value: bool) -> str:
    return "true" if value else "false"


def _string_value(value: JsonValue | object) -> str:
    return value if isinstance(value, str) else ""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "SELECTED_RUNTIME_AMP_GRAD_SCALER_INIT_SCALE",
    "RankDeviceAssignment",
    "SelectedRuntimeEnvironmentProbe",
    "SelectedRuntimeLaunchCommand",
    "SelectedRuntimeTrainRequest",
    "SelectedRuntimeTrainResult",
    "build_ddp_rank_device_proof",
    "build_selected_runtime_torchrun_command",
    "validate_selected_runtime_environment",
    "validate_selected_runtime_torchrun_command",
    "write_selected_runtime_training_run",
]
