# Copyright 2026 HiperMaximus
"""Real selected-runtime train runner with local dry-run proofs."""

from __future__ import annotations

import csv
import hashlib
import json
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

from eqvae.artifacts.fixed25_equivariance import (
    DEGREES_PER_K,
    EQUIVARIANCE_25_COLUMNS,
    MEASURED_K_VALUES,
    REQUIRED_EQUIVARIANCE_METRICS,
    Fixed25Config,
    Fixed25Patches,
    compute_rot90_exactness,
    evaluate_boundary,
    load_fixed25_patches,
    parse_fixed25_config,
    validation_shard_spec_for,
    write_manifest,
    write_originals,
)
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

    from eqvae.benchmarking.io import CsvRow, JsonObject, JsonValue


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
_FULL_RUN_MODE = "kaggle_selected_runtime_full_train"
_FULL_STATUS_SCOPE = "selected_runtime_full_training_run"
_FULL_EPOCHS = 10
_FULL_UPDATES_PER_EPOCH = 12_500
_FULL_TARGET_UPDATES = 125_000
_FULL_HALF_EPOCH_INTERVAL_STEPS = 6_250
_FULL_VALIDATION_BATCHES_PER_VIEW = 20
_FULL_VALIDATION_VIEWS = ("clean", "deterministic_denoising")
_FULL_CHECKPOINT_RETENTION = "best_final_latest_four_interval"
_STOCHASTIC_REPARAMETERIZATION = "stochastic_seeded"
_DETERMINISTIC_REPARAMETERIZATION = "deterministic_zero"
_FULL_DETERMINISTIC_EPS_ALLOWED_FOR = (
    "debug",
    "tiny",
    "numerical_checks",
    "validation",
    "artifacts",
)
_FULL_INTERVAL_CHECKPOINT_KEEP_COUNT = 4
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
    "train_reparameterization",
    "eps_policy",
    "eps_seed_source",
    "eps_zero_fraction",
    "eps_abs_mean",
    "amp_step_skipped",
    "checkpoint_path",
)
_VALIDATION_METRIC_COLUMNS = (
    "event_id",
    "rank",
    "optimizer_step",
    "validation_boundary",
    "split",
    "view",
    "batch_count",
    "sample_count",
    "loss",
    "recon_loss",
    "l1_loss",
    "ssim_loss",
    "ssim_metric",
    "kl_loss",
    "beta",
    "deterministic_eps_used",
    "corruption_strategy",
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
    fixed_25_validation_patches: Path | None = None
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
    target_train_steps: int
    max_val_steps: int
    save_every_steps: int
    requested_epochs: int
    optimizer_updates_per_epoch: int
    half_epoch_interval_steps: int
    validation_batches_per_view: int
    validation_views: tuple[str, ...]
    train_reparameterization: str
    deterministic_eps_allowed_for: tuple[str, ...]
    checkpoint_retention: str
    resume_supported: bool
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
class _FinalArtifactWriteResult:
    best_checkpoint: CheckpointMetadata
    final_checkpoint: CheckpointMetadata


@dataclass(frozen=True)
class _RunArtifacts:
    training_summary: Path
    selected_runtime_debug_summary: Path
    selected_runtime_full_summary: Path
    selected_runtime_plan_applied: Path
    checkpoint_resume_proof: Path
    gate_health_summary: Path
    tiny_overfit_summary: Path
    local_readiness: Path
    artifact_manifest: Path
    train_steps: Path
    validation_metrics: Path
    gate_health: Path
    reconstruction_samples: Path
    equivariance_25: Path
    fixed25_dir: Path


@dataclass(frozen=True)
class _Fixed25Runtime:
    """Loaded fixed-25 embedding-equivariance evaluation state (Spec 0010)."""

    config: Fixed25Config
    patches: Fixed25Patches
    fixed25_dir: Path
    data_source: str
    promotable: bool
    rot90_exactness_error: float


@dataclass(frozen=True)
class _IntervalFlushContext:
    artifacts: _RunArtifacts
    request: SelectedRuntimeTrainRequest
    settings: _RunnerSettings
    plan: SelectedRuntimePlan
    runtime_identity: _RuntimeIdentity
    launch_command: SelectedRuntimeLaunchCommand
    ddp_proof: JsonObject
    amp: _AmpExecution
    data_surface: _DataSurface
    distributed: _DistributedContext


@dataclass(frozen=True)
class _IntervalFlushState:
    metric_rows: tuple[CsvRow, ...]
    validation_rows: tuple[CsvRow, ...]
    gate_rows: tuple[CsvRow, ...]
    checkpoints: tuple[CheckpointMetadata, ...]
    best_checkpoint: CheckpointMetadata | None
    best_validation_metric: float | None
    last_result: _SelectedRuntimeStepResult
    current_step: int
    # Resume prefix rows are kept separate from the per-rank rows above so the
    # interval flush can prepend them exactly once, after the all-gather, rather
    # than gathering the same prefix from every rank (see
    # _write_interval_artifact_flush).
    resume_metric_rows: tuple[CsvRow, ...] = ()
    resume_validation_rows: tuple[CsvRow, ...] = ()
    # Fixed-25 equivariance rows are canonical global rows produced on rank 0 only
    # (Spec 0010): they are merged with the resume prefix but never all-gathered.
    equivariance_rows: tuple[CsvRow, ...] = ()
    resume_equivariance_rows: tuple[CsvRow, ...] = ()


@dataclass(frozen=True)
class _CheckpointWriteContext:
    request: SelectedRuntimeTrainRequest
    resolved: ResolvedConfig
    settings: _RunnerSettings
    model: nn.Module
    optimizer: torch.optim.Optimizer
    numpy_generator: Generator
    train_generator: torch.Generator
    runtime_identity: _RuntimeIdentity
    scaler: GradScaler
    amp: _AmpExecution
    distributed: _DistributedContext


@dataclass(frozen=True)
class _BoundaryCheckpointWriteResult:
    interval_checkpoint: CheckpointMetadata | None
    best_checkpoint: CheckpointMetadata | None
    best_validation_metric: float | None
    checkpoint_path: str


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
class _EpsProof:
    eps_policy: str
    eps_seed_source: str
    eps_zero_fraction: float
    eps_abs_mean: float


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
    train_reparameterization: str
    eps_policy: str
    eps_seed_source: str
    eps_zero_fraction: float
    eps_abs_mean: float


@dataclass(frozen=True)
class _TrainLoopResult:
    metric_rows: tuple[CsvRow, ...]
    validation_rows: tuple[CsvRow, ...]
    interval_checkpoints: tuple[CheckpointMetadata, ...]
    best_validation_checkpoint: CheckpointMetadata | None
    best_validation_metric: float | None
    last_result: _SelectedRuntimeStepResult
    equivariance_rows: tuple[CsvRow, ...] = ()


@dataclass(frozen=True)
class _ResumeArtifactHistory:
    metric_rows: tuple[CsvRow, ...]
    validation_rows: tuple[CsvRow, ...]
    interval_checkpoints: tuple[CheckpointMetadata, ...]
    best_checkpoint: CheckpointMetadata | None
    best_validation_metric: float | None
    equivariance_rows: tuple[CsvRow, ...] = ()


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


def write_selected_runtime_training_run(  # noqa: PLR0914
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
    amp = _amp_execution(plan=plan, distributed=distributed, dry_run=request.dry_run)
    scaler = GradScaler(
        "cuda",
        init_scale=amp.grad_scaler_init_scale,
        enabled=amp.grad_scaler_enabled,
    )
    loaded_checkpoint = _restore_checkpoint_if_requested(
        request=request,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        amp=amp,
        distributed=distributed,
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
    resume_history = _load_resume_artifact_history(
        artifacts=artifacts,
        settings=settings,
        distributed=distributed,
        start_step=start_step,
    )

    wrapped_model = _maybe_wrap_ddp(
        model=model,
        distributed=distributed,
        plan=plan,
    )
    write_artifacts = _is_primary_rank(distributed)
    fixed25 = _prepare_fixed25_runtime(
        request=request,
        settings=settings,
        resolved=resolved,
        data_surface=data_surface,
        distributed=distributed,
        model=model,
        artifacts=artifacts,
    )
    train_loop = _run_train_steps(
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
        initial_best_validation_metric=resume_history.best_validation_metric,
        resume_history=resume_history,
        write_checkpoints=write_artifacts,
        interval_flush=_IntervalFlushContext(
            artifacts=artifacts,
            request=request,
            settings=settings,
            plan=plan,
            runtime_identity=runtime_identity,
            launch_command=launch_command,
            ddp_proof=ddp_proof,
            amp=amp,
            data_surface=data_surface,
            distributed=distributed,
        )
        if _is_full_run(settings)
        else None,
        fixed25=fixed25,
    )
    metric_rows = _merge_resume_csv_rows(
        prior_rows=resume_history.metric_rows,
        new_rows=_gather_csv_rows(train_loop.metric_rows, distributed),
    )
    validation_rows = _merge_resume_csv_rows(
        prior_rows=resume_history.validation_rows,
        new_rows=_gather_csv_rows(train_loop.validation_rows, distributed),
    )
    # Canonical global fixed-25 rows are produced on rank 0 only: merge the
    # resume prefix but do NOT all-gather (Spec 0010 DDP contract).
    equivariance_rows = _merge_resume_csv_rows(
        prior_rows=resume_history.equivariance_rows,
        new_rows=train_loop.equivariance_rows,
    )
    checkpoints = (
        *resume_history.interval_checkpoints,
        *train_loop.interval_checkpoints,
    )
    last_result = train_loop.last_result
    best_validation_checkpoint = train_loop.best_validation_checkpoint
    best_validation_metric = train_loop.best_validation_metric
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
    _write_final_artifacts(
        artifacts=artifacts,
        request=request,
        resolved=resolved,
        settings=settings,
        plan=plan,
        runtime_identity=runtime_identity,
        launch_command=launch_command,
        ddp_proof=ddp_proof,
        amp=amp,
        data_surface=data_surface,
        distributed=distributed,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        numpy_generator=numpy_generator,
        train_generator=train_generator,
        loaded_checkpoint=loaded_checkpoint,
        resume_history=resume_history,
        metric_rows=metric_rows,
        validation_rows=validation_rows,
        gate_rows=gate_rows,
        equivariance_rows=equivariance_rows,
        checkpoints=checkpoints,
        best_validation_checkpoint=best_validation_checkpoint,
        best_validation_metric=best_validation_metric,
        last_result=last_result,
        fixed25=fixed25,
    )
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


def _write_final_artifacts(  # noqa: PLR0913
    *,
    artifacts: _RunArtifacts,
    request: SelectedRuntimeTrainRequest,
    resolved: ResolvedConfig,
    settings: _RunnerSettings,
    plan: SelectedRuntimePlan,
    runtime_identity: _RuntimeIdentity,
    launch_command: SelectedRuntimeLaunchCommand,
    ddp_proof: JsonObject,
    amp: _AmpExecution,
    data_surface: _DataSurface,
    distributed: _DistributedContext,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    numpy_generator: Generator,
    train_generator: torch.Generator,
    loaded_checkpoint: LoadedCheckpoint | None,
    resume_history: _ResumeArtifactHistory,
    metric_rows: Sequence[CsvRow],
    validation_rows: Sequence[CsvRow],
    gate_rows: Sequence[CsvRow],
    equivariance_rows: Sequence[CsvRow],
    checkpoints: Sequence[CheckpointMetadata],
    best_validation_checkpoint: CheckpointMetadata | None,
    best_validation_metric: float | None,
    last_result: _SelectedRuntimeStepResult,
    fixed25: _Fixed25Runtime | None,
) -> _FinalArtifactWriteResult | None:
    write_error: Exception | None = None
    write_error_message: str | None = None
    result: _FinalArtifactWriteResult | None = None
    if _is_primary_rank(distributed):
        try:
            result = _write_final_artifacts_primary(
                artifacts=artifacts,
                request=request,
                resolved=resolved,
                settings=settings,
                plan=plan,
                runtime_identity=runtime_identity,
                launch_command=launch_command,
                ddp_proof=ddp_proof,
                amp=amp,
                data_surface=data_surface,
                distributed=distributed,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                numpy_generator=numpy_generator,
                train_generator=train_generator,
                loaded_checkpoint=loaded_checkpoint,
                resume_history=resume_history,
                metric_rows=metric_rows,
                validation_rows=validation_rows,
                gate_rows=gate_rows,
                equivariance_rows=equivariance_rows,
                checkpoints=checkpoints,
                best_validation_checkpoint=best_validation_checkpoint,
                best_validation_metric=best_validation_metric,
                last_result=last_result,
                fixed25=fixed25,
            )
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - DDP sync guard
            write_error = exc
            write_error_message = _exception_summary(exc)
    write_error_message = _broadcast_rank0_error(
        error_message=write_error_message,
        distributed=distributed,
    )
    if write_error_message is not None:
        message = (
            "selected-runtime final artifact write failed on rank 0: "
            f"{write_error_message}"
        )
        raise RuntimeError(message) from write_error
    return result


def _write_final_artifacts_primary(  # noqa: PLR0913
    *,
    artifacts: _RunArtifacts,
    request: SelectedRuntimeTrainRequest,
    resolved: ResolvedConfig,
    settings: _RunnerSettings,
    plan: SelectedRuntimePlan,
    runtime_identity: _RuntimeIdentity,
    launch_command: SelectedRuntimeLaunchCommand,
    ddp_proof: JsonObject,
    amp: _AmpExecution,
    data_surface: _DataSurface,
    distributed: _DistributedContext,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    numpy_generator: Generator,
    train_generator: torch.Generator,
    loaded_checkpoint: LoadedCheckpoint | None,
    resume_history: _ResumeArtifactHistory,
    metric_rows: Sequence[CsvRow],
    validation_rows: Sequence[CsvRow],
    gate_rows: Sequence[CsvRow],
    equivariance_rows: Sequence[CsvRow],
    checkpoints: Sequence[CheckpointMetadata],
    best_validation_checkpoint: CheckpointMetadata | None,
    best_validation_metric: float | None,
    last_result: _SelectedRuntimeStepResult,
    fixed25: _Fixed25Runtime | None,
) -> _FinalArtifactWriteResult:
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
    best_checkpoint = (
        best_validation_checkpoint
        or resume_history.best_checkpoint
        or _save_checkpoint(
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
    )
    retained_checkpoints = _apply_checkpoint_retention(
        checkpoints=checkpoints,
        settings=settings,
    )
    all_checkpoints = (*retained_checkpoints, final_checkpoint, best_checkpoint)
    checkpoint_resume_proof = _final_checkpoint_resume_proof(
        loaded_checkpoint=loaded_checkpoint,
        checkpoints=retained_checkpoints,
        request=request,
        resolved=resolved,
        settings=settings,
        runtime_identity=runtime_identity,
        plan=plan,
        train_generator=train_generator,
        amp=amp,
        distributed=distributed,
    )
    # The trivial single-patch dump is retired when the fixed-25 protocol is
    # active (Spec 0010); the debug/tiny path keeps writing it unchanged.
    reconstruction_nonblank = (
        False
        if fixed25 is not None
        else _write_reconstruction_sample(
            path=artifacts.reconstruction_samples,
            model=model,
            settings=settings,
            data_surface=data_surface,
            device=distributed.device,
        )
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
    _write_final_csv_artifacts(
        artifacts=artifacts,
        settings=settings,
        metric_rows=metric_rows,
        validation_rows=validation_rows,
        gate_rows=gate_rows,
        equivariance_rows=equivariance_rows if fixed25 is not None else (),
    )
    _write_json_atomic(artifacts.selected_runtime_plan_applied, plan_applied)
    _write_json_atomic(artifacts.checkpoint_resume_proof, checkpoint_resume_proof)
    _write_json_atomic(artifacts.gate_health_summary, gate_health_summary)
    _write_json_atomic(artifacts.local_readiness, local_readiness)
    _write_json_atomic(
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
            validation_rows=validation_rows,
            checkpoints=retained_checkpoints,
            final_checkpoint=final_checkpoint,
            best_checkpoint=best_checkpoint,
            best_validation_metric=best_validation_metric,
            last_result=last_result,
            plan_applied=plan_applied,
            checkpoint_resume_proof=checkpoint_resume_proof,
            gate_health_summary=gate_health_summary,
            reconstruction_nonblank=reconstruction_nonblank,
        ),
    )
    _write_mode_summary(
        artifacts=artifacts,
        settings=settings,
        plan=plan,
        plan_applied=plan_applied,
        ddp_proof=ddp_proof,
        amp=amp,
        checkpoint_resume_proof=checkpoint_resume_proof,
        gate_health_summary=gate_health_summary,
        data_surface=data_surface,
        metric_rows=metric_rows,
        validation_rows=validation_rows,
        checkpoints=retained_checkpoints,
        best_validation_metric=best_validation_metric,
    )
    if _writes_tiny_summary(settings):
        _write_json_atomic(
            artifacts.tiny_overfit_summary,
            _tiny_overfit_summary(
                runtime_identity=runtime_identity,
                corruption_strategy=plan.corruption_strategy,
                data_surface=data_surface,
                metric_rows=metric_rows,
                gate_health_summary=gate_health_summary,
            ),
        )
    _write_json_atomic(
        artifacts.artifact_manifest,
        _artifact_manifest(
            artifacts=artifacts,
            settings=settings,
            checkpoints=all_checkpoints,
            metric_rows=metric_rows,
            reconstruction_nonblank=reconstruction_nonblank,
            fixed25=fixed25,
        ),
    )
    return _FinalArtifactWriteResult(
        best_checkpoint=best_checkpoint,
        final_checkpoint=final_checkpoint,
    )


def _final_checkpoint_resume_proof(  # noqa: PLR0913
    *,
    loaded_checkpoint: LoadedCheckpoint | None,
    checkpoints: Sequence[CheckpointMetadata],
    request: SelectedRuntimeTrainRequest,
    resolved: ResolvedConfig,
    settings: _RunnerSettings,
    runtime_identity: _RuntimeIdentity,
    plan: SelectedRuntimePlan,
    train_generator: torch.Generator,
    amp: _AmpExecution,
    distributed: _DistributedContext,
) -> JsonObject:
    if loaded_checkpoint is None:
        return _checkpoint_resume_proof(
            checkpoint=_resume_probe_checkpoint(
                checkpoints=checkpoints,
                target_step=settings.max_train_steps,
            ),
            request=request,
            resolved=resolved,
            settings=settings,
            runtime_identity=runtime_identity,
            plan=plan,
            train_generator=train_generator,
            amp=amp,
            distributed=distributed,
        )
    return _loaded_checkpoint_resume_proof(
        loaded=loaded_checkpoint,
        request=request,
        resolved=resolved,
        settings=settings,
        runtime_identity=runtime_identity,
        amp=amp,
        distributed=distributed,
    )


def _write_final_csv_artifacts(  # noqa: PLR0913
    *,
    artifacts: _RunArtifacts,
    settings: _RunnerSettings,
    metric_rows: Sequence[CsvRow],
    validation_rows: Sequence[CsvRow],
    gate_rows: Sequence[CsvRow],
    equivariance_rows: Sequence[CsvRow] = (),
) -> None:
    _write_csv_atomic(artifacts.train_steps, _TRAIN_STEP_COLUMNS, metric_rows)
    if _writes_validation_metrics(settings):
        _write_csv_atomic(
            artifacts.validation_metrics,
            _VALIDATION_METRIC_COLUMNS,
            validation_rows,
        )
    if equivariance_rows:
        _write_csv_atomic(
            artifacts.equivariance_25,
            EQUIVARIANCE_25_COLUMNS,
            equivariance_rows,
        )
    _write_csv_atomic(artifacts.gate_health, GATE_HEALTH_COLUMNS, gate_rows)


def _write_mode_summary(  # noqa: PLR0913
    *,
    artifacts: _RunArtifacts,
    settings: _RunnerSettings,
    plan: SelectedRuntimePlan,
    plan_applied: JsonObject,
    ddp_proof: JsonObject,
    amp: _AmpExecution,
    checkpoint_resume_proof: JsonObject,
    gate_health_summary: JsonObject,
    data_surface: _DataSurface,
    metric_rows: Sequence[CsvRow],
    validation_rows: Sequence[CsvRow],
    checkpoints: Sequence[CheckpointMetadata],
    best_validation_metric: float | None,
) -> None:
    if _is_full_run(settings):
        _write_json_atomic(
            artifacts.selected_runtime_full_summary,
            _selected_runtime_full_summary(
                plan=plan,
                settings=settings,
                plan_applied=plan_applied,
                ddp_proof=ddp_proof,
                amp=amp,
                checkpoint_resume_proof=checkpoint_resume_proof,
                gate_health_summary=gate_health_summary,
                data_surface=data_surface,
                metric_rows=metric_rows,
                validation_rows=validation_rows,
                checkpoints=checkpoints,
                best_validation_metric=best_validation_metric,
            ),
        )
        return
    _write_json_atomic(
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


def _settings(  # noqa: PLR0914
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
    run_mode = _required_str(run, "mode")
    full_mode = run_mode == _FULL_RUN_MODE
    requested_epochs = _optional_int(training, "epochs") or 0
    optimizer_updates_per_epoch = (
        _optional_int(training, "optimizer_updates_per_epoch")
        or plan.optimizer_updates_per_epoch
    )
    configured_max_train_steps = _optional_int(training, "max_train_steps")
    target_train_steps = _target_train_steps(
        training=training,
        requested_epochs=requested_epochs,
        optimizer_updates_per_epoch=optimizer_updates_per_epoch,
        configured_max_train_steps=configured_max_train_steps,
        full_mode=full_mode,
    )
    if not full_mode and request.max_train_steps is not None:
        target_train_steps = request.max_train_steps
    max_train_steps = _execution_train_steps(
        request=request,
        target_train_steps=target_train_steps,
        full_mode=full_mode,
    )
    half_epoch_interval_steps = _half_epoch_interval_steps(
        training=training,
        optimizer_updates_per_epoch=optimizer_updates_per_epoch,
        full_mode=full_mode,
    )
    validation_views = _optional_str_tuple(training, "validation_views")
    validation_batches_per_view = (
        _optional_int(training, "validation_batches_per_view") or 0
    )
    settings = _RunnerSettings(
        run_name=request.run_name or _required_str(run, "name"),
        run_mode=run_mode,
        batch_size=plan.per_device_batch_size,
        image_size=_optional_int(data, "image_size") or 256,
        max_train_steps=max_train_steps,
        target_train_steps=target_train_steps,
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
        requested_epochs=requested_epochs,
        optimizer_updates_per_epoch=optimizer_updates_per_epoch,
        half_epoch_interval_steps=half_epoch_interval_steps,
        validation_batches_per_view=validation_batches_per_view,
        validation_views=validation_views,
        train_reparameterization=(
            _optional_str(training, "train_reparameterization")
            or _DETERMINISTIC_REPARAMETERIZATION
        ),
        deterministic_eps_allowed_for=(
            _optional_str_tuple(training, "deterministic_eps_allowed_for")
            or _FULL_DETERMINISTIC_EPS_ALLOWED_FOR
        ),
        checkpoint_retention=(
            _optional_str(training, "checkpoint_retention") or "keep_all"
        ),
        resume_supported=_optional_bool(training, "resume_supported") or False,
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
    _validate_settings(settings, dry_run=request.dry_run)
    return settings


def _target_train_steps(
    *,
    training: JsonObject,
    requested_epochs: int,
    optimizer_updates_per_epoch: int,
    configured_max_train_steps: int | None,
    full_mode: bool,
) -> int:
    if not full_mode:
        return configured_max_train_steps or 1
    if requested_epochs <= 0:
        message = "full-run config must declare positive training.epochs"
        raise ValueError(message)
    if optimizer_updates_per_epoch <= 0:
        message = "full-run config must declare positive optimizer_updates_per_epoch"
        raise ValueError(message)
    derived = requested_epochs * optimizer_updates_per_epoch
    if configured_max_train_steps is None:
        message = (
            "full-run config must declare max_train_steps; refusing one-step default"
        )
        raise ValueError(message)
    if configured_max_train_steps != derived:
        message = (
            "full-run max_train_steps must equal epochs * "
            "optimizer_updates_per_epoch: "
            f"{configured_max_train_steps} != {derived}"
        )
        raise ValueError(message)
    if training.get("validate_every") == "half_epoch" and derived % 2 != 0:
        message = "full-run half-epoch schedule requires an even update target"
        raise ValueError(message)
    return derived


def _execution_train_steps(
    *,
    request: SelectedRuntimeTrainRequest,
    target_train_steps: int,
    full_mode: bool,
) -> int:
    if request.max_train_steps is None:
        return target_train_steps
    if full_mode and not request.dry_run:
        message = "full-run max_train_steps overrides are allowed only with --dry-run"
        raise ValueError(message)
    if request.max_train_steps > target_train_steps:
        message = "max_train_steps override cannot exceed the configured target"
        raise ValueError(message)
    return request.max_train_steps


def _half_epoch_interval_steps(
    *,
    training: JsonObject,
    optimizer_updates_per_epoch: int,
    full_mode: bool,
) -> int:
    configured = _optional_int(training, "half_epoch_interval_steps")
    if configured is not None:
        return configured
    if full_mode:
        if optimizer_updates_per_epoch % 2 != 0:
            message = (
                "optimizer_updates_per_epoch must be even for half-epoch validation"
            )
            raise ValueError(message)
        return optimizer_updates_per_epoch // 2
    return 0


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
        selected_runtime_full_summary=benchmark / "selected_runtime_full_summary.json",
        selected_runtime_plan_applied=benchmark / "selected_runtime_plan_applied.json",
        checkpoint_resume_proof=benchmark / "checkpoint_resume_proof.json",
        gate_health_summary=benchmark / "gate_health_summary.json",
        tiny_overfit_summary=benchmark / "tiny_overfit_summary.json",
        local_readiness=benchmark / "local_selected_runtime_readiness.json",
        artifact_manifest=benchmark / "artifact_manifest.json",
        train_steps=metrics / "train_steps.csv",
        validation_metrics=metrics / "validation_metrics.csv",
        gate_health=metrics / "gate_health.csv",
        reconstruction_samples=output_dir / "artifacts" / "reconstruction_samples.pt",
        equivariance_25=metrics / "equivariance_25.csv",
        fixed25_dir=output_dir / "artifacts" / "fixed25",
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


def _load_resume_artifact_history(
    *,
    artifacts: _RunArtifacts,
    settings: _RunnerSettings,
    distributed: _DistributedContext,
    start_step: int,
) -> _ResumeArtifactHistory:
    if start_step <= 0:
        return _empty_resume_artifact_history()

    train_rows = _read_resume_csv_prefix(
        path=artifacts.train_steps,
        step_key="successful_optimizer_update_count",
        start_step=start_step,
        required=_is_full_run(settings),
        artifact_name="metrics/train_steps.csv",
    )
    validation_rows = _read_resume_csv_prefix(
        path=artifacts.validation_metrics,
        step_key="optimizer_step",
        start_step=start_step,
        required=False,
        artifact_name="metrics/validation_metrics.csv",
    )
    equivariance_rows = _read_resume_csv_prefix(
        path=artifacts.equivariance_25,
        step_key="optimizer_step",
        start_step=start_step,
        required=False,
        artifact_name="metrics/equivariance_25.csv",
    )
    output_dir = artifacts.train_steps.parent.parent
    interval_checkpoints = _resume_interval_checkpoints(
        output_dir=output_dir,
        start_step=start_step,
    )
    best_checkpoint, best_validation_metric = _resume_best_checkpoint(
        artifacts=artifacts,
        output_dir=output_dir,
        start_step=start_step,
    )
    if _is_full_run(settings):
        _validate_full_resume_train_prefix(
            rows=train_rows,
            start_step=start_step,
            distributed=distributed,
        )
        _validate_full_resume_validation_prefix(
            rows=validation_rows,
            settings=settings,
            start_step=start_step,
        )
        _validate_full_resume_checkpoint_prefix(
            checkpoints=interval_checkpoints,
            settings=settings,
            start_step=start_step,
        )
        _validate_full_resume_equivariance_prefix(
            rows=equivariance_rows,
            settings=settings,
            start_step=start_step,
        )
    return _ResumeArtifactHistory(
        metric_rows=train_rows,
        validation_rows=validation_rows,
        interval_checkpoints=interval_checkpoints,
        best_checkpoint=best_checkpoint,
        best_validation_metric=best_validation_metric,
        equivariance_rows=equivariance_rows,
    )


def _empty_resume_artifact_history() -> _ResumeArtifactHistory:
    return _ResumeArtifactHistory(
        metric_rows=(),
        validation_rows=(),
        interval_checkpoints=(),
        best_checkpoint=None,
        best_validation_metric=None,
        equivariance_rows=(),
    )


def _read_resume_csv_prefix(
    *,
    path: Path,
    step_key: str,
    start_step: int,
    required: bool,
    artifact_name: str,
) -> tuple[CsvRow, ...]:
    if not path.exists():
        if required:
            message = f"full-run resume requires existing {artifact_name}"
            raise ValueError(message)
        return ()
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = [dict(row) for row in reader]
    if required and step_key not in (reader.fieldnames or ()):
        message = f"full-run resume {artifact_name} is missing {step_key}"
        raise ValueError(message)
    return tuple(
        row for row in rows if _csv_int(row, step_key, default=0) <= start_step
    )


def _merge_resume_csv_rows(
    *,
    prior_rows: Sequence[CsvRow],
    new_rows: Sequence[CsvRow],
) -> tuple[CsvRow, ...]:
    if not prior_rows:
        return tuple(new_rows)
    return (*tuple(prior_rows), *tuple(new_rows))


def _resume_interval_checkpoints(
    *,
    output_dir: Path,
    start_step: int,
) -> tuple[CheckpointMetadata, ...]:
    checkpoint_dir = output_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return ()
    checkpoints: list[CheckpointMetadata] = []
    for path in checkpoint_dir.glob("step_*.pt"):
        step = _checkpoint_step_from_name(path.name)
        if step is None or step > start_step:
            continue
        checkpoints.append(
            CheckpointMetadata(
                path=path,
                sha256=_sha256_file(path),
                optimizer_step=step,
                successful_optimizer_update_count=step,
            ),
        )
    return tuple(
        sorted(
            checkpoints,
            key=lambda checkpoint: checkpoint.successful_optimizer_update_count,
        ),
    )


def _resume_best_checkpoint(
    *,
    artifacts: _RunArtifacts,
    output_dir: Path,
    start_step: int,
) -> tuple[CheckpointMetadata | None, float | None]:
    checkpoint_path = output_dir / "checkpoints" / "best_model.pt"
    if not checkpoint_path.exists():
        return None, None
    summary = _read_json_object_if_exists(artifacts.training_summary)
    metric = _json_float(summary.get("best_validation_metric")) if summary else None
    checkpoint_payload = summary.get("best_checkpoint") if summary is not None else None
    checkpoint_step = start_step
    if isinstance(checkpoint_payload, dict):
        checkpoint_step = _json_int(
            checkpoint_payload.get("successful_optimizer_update_count"),
            default=start_step,
        )
    return (
        CheckpointMetadata(
            path=checkpoint_path,
            sha256=_sha256_file(checkpoint_path),
            optimizer_step=checkpoint_step,
            successful_optimizer_update_count=checkpoint_step,
        ),
        metric,
    )


def _validate_full_resume_train_prefix(
    *,
    rows: Sequence[CsvRow],
    start_step: int,
    distributed: _DistributedContext,
) -> None:
    expected_world_size = distributed.world_size if distributed.should_use_ddp else 1
    expected_pairs = {
        (step, rank)
        for step in range(1, start_step + 1)
        for rank in range(expected_world_size)
    }
    observed_pairs = {
        (
            _csv_int(row, "successful_optimizer_update_count", default=-1),
            _csv_int(row, "rank", default=-1),
        )
        for row in _successful_metric_rows(rows)
    }
    missing = expected_pairs - observed_pairs
    if missing:
        sample = sorted(missing)[:5]
        message = (
            "full-run resume history is missing train metric rows before the "
            f"checkpoint; sample missing step/rank pairs: {sample!r}"
        )
        raise ValueError(message)


def _validate_full_resume_validation_prefix(
    *,
    rows: Sequence[CsvRow],
    settings: _RunnerSettings,
    start_step: int,
) -> None:
    expected_steps = tuple(
        range(
            settings.half_epoch_interval_steps,
            min(start_step, settings.target_train_steps) + 1,
            settings.half_epoch_interval_steps,
        ),
    )
    observed = {
        (_csv_int(row, "optimizer_step", default=-1), row.get("view", ""))
        for row in rows
    }
    missing = {
        (step, view)
        for step in expected_steps
        for view in settings.validation_views
        if (step, view) not in observed
    }
    if missing:
        sample = sorted(missing)[:5]
        message = (
            "full-run resume history is missing validation rows before the "
            f"checkpoint; sample missing step/view pairs: {sample!r}"
        )
        raise ValueError(message)


def _validate_full_resume_equivariance_prefix(
    *,
    rows: Sequence[CsvRow],
    settings: _RunnerSettings,
    start_step: int,
) -> None:
    # Fail closed like validation/train/checkpoint (Spec 0010 DDP/resume item c):
    # if the fixed-25 protocol wrote any pre-resume rows, every half-epoch boundary
    # up to start_step must carry all measured angles x required metrics. An empty
    # prefix means the protocol was inactive (skip); a truncated prefix must raise.
    if not rows or settings.half_epoch_interval_steps <= 0:
        return
    expected_steps = tuple(
        range(
            settings.half_epoch_interval_steps,
            min(start_step, settings.target_train_steps) + 1,
            settings.half_epoch_interval_steps,
        ),
    )
    measured_angles = tuple(str(DEGREES_PER_K * k) for k in MEASURED_K_VALUES)
    observed = {
        (
            _csv_int(row, "optimizer_step", default=-1),
            row.get("angle_degrees", ""),
            row.get("metric_name", ""),
        )
        for row in rows
    }
    missing = {
        (step, angle, metric)
        for step in expected_steps
        for angle in measured_angles
        for metric in REQUIRED_EQUIVARIANCE_METRICS
        if (step, angle, metric) not in observed
    }
    if missing:
        sample = sorted(missing)[:5]
        message = (
            "full-run resume history is missing equivariance rows before the "
            f"checkpoint; sample missing step/angle/metric: {sample!r}"
        )
        raise ValueError(message)


def _validate_full_resume_checkpoint_prefix(
    *,
    checkpoints: Sequence[CheckpointMetadata],
    settings: _RunnerSettings,
    start_step: int,
) -> None:
    final_interval_steps = tuple(
        range(
            settings.half_epoch_interval_steps,
            settings.target_train_steps + 1,
            settings.half_epoch_interval_steps,
        ),
    )[-_FULL_INTERVAL_CHECKPOINT_KEEP_COUNT:]
    required_existing_steps = {
        step for step in final_interval_steps if step <= start_step
    }
    observed_steps = {
        checkpoint.successful_optimizer_update_count for checkpoint in checkpoints
    }
    missing = sorted(required_existing_steps - observed_steps)
    if missing:
        message = (
            "full-run resume history is missing retained interval checkpoints "
            f"needed for final verification: {missing!r}"
        )
        raise ValueError(message)


def _checkpoint_step_from_name(name: str) -> int | None:
    prefix = "step_"
    suffix = ".pt"
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    try:
        return int(name[len(prefix) : -len(suffix)])
    except ValueError:
        return None


def _read_json_object_if_exists(path: Path) -> JsonObject | None:
    if not path.exists():
        return None
    payload = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(payload, dict):
        return None
    return cast("JsonObject", payload)


def _json_int(value: object, *, default: int) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _json_float(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _csv_int(row: CsvRow, key: str, *, default: int) -> int:
    try:
        return int(row.get(key, ""))
    except (TypeError, ValueError):
        return default


def _prepare_data_surface(  # noqa: PLR0914
    *,
    request: SelectedRuntimeTrainRequest,
    settings: _RunnerSettings,
    plan: SelectedRuntimePlan,
    distributed: _DistributedContext,
) -> _DataSurface:
    if request.data == "synthetic":
        root = request.output_dir / "local_ubc_synthetic"
        synthetic_train_steps = min(settings.max_train_steps, 32)
        train_count = max(32, settings.batch_size * synthetic_train_steps)
        validation_count = max(
            32,
            settings.batch_size * max(1, settings.validation_batches_per_view),
        )
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
    scaler: GradScaler,
    amp: _AmpExecution,
    distributed: _DistributedContext,
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
        amp_scaler=scaler if amp.grad_scaler_enabled else None,
        restore_cuda_rng=distributed.device.type == "cuda",
        expected_effective_config_sha256=resolved.effective_config_hash,
        expected_runtime_config_sha256=runtime_identity.sha256,
        expected_selected_row_id=runtime_identity.selected_row_id,
        expected_runtime_policy_id=runtime_identity.runtime_policy_id,
    )


def _run_train_steps(  # noqa: C901, PLR0913, PLR0914, PLR0915
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
    initial_best_validation_metric: float | None,
    resume_history: _ResumeArtifactHistory,
    write_checkpoints: bool,
    interval_flush: _IntervalFlushContext | None,
    fixed25: _Fixed25Runtime | None = None,
) -> _TrainLoopResult:
    rows: list[CsvRow] = []
    validation_rows: list[CsvRow] = []
    equivariance_rows: list[CsvRow] = []
    checkpoints: list[CheckpointMetadata] = []
    checkpoint_context = _CheckpointWriteContext(
        request=request,
        resolved=resolved,
        settings=settings,
        model=checkpoint_model,
        optimizer=optimizer,
        numpy_generator=numpy_generator,
        train_generator=train_generator,
        runtime_identity=runtime_identity,
        scaler=scaler,
        amp=amp,
        distributed=distributed,
    )
    train_batches = _cycle_batches(data_surface.train_loader)
    _advance_batches(train_batches, start_step)
    last_result: _SelectedRuntimeStepResult | None = None
    successful_count = start_step
    attempt_count = 0
    best_validation_metric = initial_best_validation_metric
    best_validation_checkpoint: CheckpointMetadata | None = None
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
            train_generator=train_generator,
            device=distributed.device,
        )
        last_result = result
        if not result.amp_step_skipped:
            successful_count = result.successful_optimizer_update_count
        checkpoint_boundary = (
            not result.amp_step_skipped
            and successful_count > 0
            and successful_count % settings.save_every_steps == 0
        )
        scheduled_validation_due = (
            not result.amp_step_skipped
            and _should_run_scheduled_validation(settings, successful_count)
        )
        rows.append(
            _metric_row(
                result=result,
                rank=distributed.rank,
                plan=plan,
                amp=amp,
                checkpoint_path="",
                corruption_strategy=plan.corruption_strategy,
            ),
        )
        if scheduled_validation_due:
            _log_full_boundary_start(
                settings=settings,
                distributed=distributed,
                optimizer_step=successful_count,
            )
            boundary_rows = _run_scheduled_validation(
                model=model,
                settings=settings,
                plan=plan,
                amp=amp,
                data_surface=data_surface,
                optimizer_step=successful_count,
                rank=distributed.rank,
                device=distributed.device,
            )
            validation_rows.extend(boundary_rows)
            boundary_metric = _validation_best_l1(boundary_rows)
            if fixed25 is not None:
                equivariance_rows.extend(
                    _evaluate_fixed25_boundary(
                        fixed25=fixed25,
                        model=checkpoint_model,
                        distributed=distributed,
                        optimizer_step=successful_count,
                    ),
                )
            if interval_flush is not None:
                _write_interval_artifact_flush(
                    context=interval_flush,
                    model=checkpoint_model,
                    local_state=_interval_flush_state(
                        settings=settings,
                        resume_history=resume_history,
                        metric_rows=rows,
                        validation_rows=validation_rows,
                        checkpoints=checkpoints,
                        best_checkpoint=best_validation_checkpoint,
                        best_validation_metric=best_validation_metric,
                        last_result=result,
                        current_step=successful_count,
                        equivariance_rows=equivariance_rows,
                    ),
                )
            checkpoint_write = _write_boundary_checkpoints(
                context=checkpoint_context,
                step=successful_count,
                step_metric_value=float(result.losses.l1_loss.detach().cpu().item()),
                checkpoint_boundary=checkpoint_boundary,
                write_checkpoints=write_checkpoints,
                boundary_metric=boundary_metric,
                best_validation_metric=best_validation_metric,
            )
            best_validation_metric = checkpoint_write.best_validation_metric
            best_validation_checkpoint = (
                checkpoint_write.best_checkpoint or best_validation_checkpoint
            )
            _record_interval_checkpoint_write(
                checkpoint_write=checkpoint_write,
                rows=rows,
                checkpoints=checkpoints,
            )
            if interval_flush is not None:
                _write_interval_artifact_flush(
                    context=interval_flush,
                    model=checkpoint_model,
                    local_state=_interval_flush_state(
                        settings=settings,
                        resume_history=resume_history,
                        metric_rows=rows,
                        validation_rows=validation_rows,
                        checkpoints=checkpoints,
                        best_checkpoint=best_validation_checkpoint,
                        best_validation_metric=best_validation_metric,
                        last_result=result,
                        current_step=successful_count,
                        equivariance_rows=equivariance_rows,
                    ),
                )
            _synchronize_full_boundary_completion(
                settings=settings,
                distributed=distributed,
                optimizer_step=successful_count,
            )
        elif checkpoint_boundary:
            checkpoint_write = _write_boundary_checkpoints(
                context=checkpoint_context,
                step=successful_count,
                step_metric_value=float(result.losses.l1_loss.detach().cpu().item()),
                checkpoint_boundary=checkpoint_boundary,
                write_checkpoints=write_checkpoints,
                boundary_metric=None,
                best_validation_metric=best_validation_metric,
            )
            _record_interval_checkpoint_write(
                checkpoint_write=checkpoint_write,
                rows=rows,
                checkpoints=checkpoints,
            )
            _synchronize_full_boundary_completion(
                settings=settings,
                distributed=distributed,
                optimizer_step=successful_count,
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
    return _TrainLoopResult(
        metric_rows=tuple(rows),
        validation_rows=tuple(validation_rows),
        interval_checkpoints=tuple(checkpoints),
        best_validation_checkpoint=best_validation_checkpoint,
        best_validation_metric=best_validation_metric,
        last_result=last_result,
        equivariance_rows=tuple(equivariance_rows),
    )


def _write_boundary_checkpoints(  # noqa: PLR0913
    *,
    context: _CheckpointWriteContext,
    step: int,
    step_metric_value: float,
    checkpoint_boundary: bool,
    write_checkpoints: bool,
    boundary_metric: float | None,
    best_validation_metric: float | None,
) -> _BoundaryCheckpointWriteResult:
    interval_checkpoint: CheckpointMetadata | None = None
    best_checkpoint: CheckpointMetadata | None = None
    updated_best_metric = best_validation_metric
    checkpoint_error: Exception | None = None
    checkpoint_error_message: str | None = None
    if write_checkpoints:
        try:
            if boundary_metric is not None and _is_better_validation_metric(
                boundary_metric,
                best_validation_metric,
            ):
                updated_best_metric = boundary_metric
                best_checkpoint = _save_best_validation_checkpoint(
                    context=context,
                    step=step,
                    boundary_metric=boundary_metric,
                )
            if checkpoint_boundary:
                interval_checkpoint = _save_interval_checkpoint(
                    context=context,
                    step=step,
                    metric_value=step_metric_value,
                )
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - DDP sync guard
            checkpoint_error = exc
            checkpoint_error_message = _exception_summary(exc)
    checkpoint_error_message = _broadcast_rank0_error(
        error_message=checkpoint_error_message,
        distributed=context.distributed,
    )
    if checkpoint_error_message is not None:
        message = (
            "selected-runtime interval checkpoint save failed on rank 0: "
            f"{checkpoint_error_message}"
        )
        raise RuntimeError(message) from checkpoint_error
    return _BoundaryCheckpointWriteResult(
        interval_checkpoint=interval_checkpoint,
        best_checkpoint=best_checkpoint,
        best_validation_metric=updated_best_metric,
        checkpoint_path=""
        if interval_checkpoint is None
        else _relative_to_output(interval_checkpoint.path, context.request.output_dir),
    )


def _interval_flush_state(  # noqa: PLR0913
    *,
    settings: _RunnerSettings,
    resume_history: _ResumeArtifactHistory,
    metric_rows: Sequence[CsvRow],
    validation_rows: Sequence[CsvRow],
    checkpoints: Sequence[CheckpointMetadata],
    best_checkpoint: CheckpointMetadata | None,
    best_validation_metric: float | None,
    last_result: _SelectedRuntimeStepResult,
    current_step: int,
    equivariance_rows: Sequence[CsvRow] = (),
) -> _IntervalFlushState:
    combined_checkpoints = _apply_checkpoint_retention(
        checkpoints=(*resume_history.interval_checkpoints, *tuple(checkpoints)),
        settings=settings,
    )
    return _IntervalFlushState(
        # Only this rank's new rows go in the all-gathered fields. The resume
        # prefix is carried separately and prepended once after the gather;
        # prepending it here would duplicate it world_size times through
        # _gather_csv_rows on a resumed DDP run.
        metric_rows=tuple(metric_rows),
        validation_rows=tuple(validation_rows),
        gate_rows=(),
        checkpoints=combined_checkpoints,
        best_checkpoint=best_checkpoint or resume_history.best_checkpoint,
        best_validation_metric=best_validation_metric,
        last_result=last_result,
        current_step=current_step,
        resume_metric_rows=tuple(resume_history.metric_rows),
        resume_validation_rows=tuple(resume_history.validation_rows),
        equivariance_rows=tuple(equivariance_rows),
        resume_equivariance_rows=tuple(resume_history.equivariance_rows),
    )


def _record_interval_checkpoint_write(
    *,
    checkpoint_write: _BoundaryCheckpointWriteResult,
    rows: list[CsvRow],
    checkpoints: list[CheckpointMetadata],
) -> None:
    if checkpoint_write.interval_checkpoint is not None:
        checkpoints.append(checkpoint_write.interval_checkpoint)
    if checkpoint_write.checkpoint_path:
        rows[-1] = _replace_metric_checkpoint_path(
            rows[-1],
            checkpoint_write.checkpoint_path,
        )


def _is_better_validation_metric(
    boundary_metric: float,
    best_validation_metric: float | None,
) -> bool:
    return best_validation_metric is None or boundary_metric < best_validation_metric


def _save_best_validation_checkpoint(
    *,
    context: _CheckpointWriteContext,
    step: int,
    boundary_metric: float,
) -> CheckpointMetadata:
    return _save_checkpoint(
        path=context.request.output_dir / "checkpoints" / "best_model.pt",
        request=context.request,
        resolved=context.resolved,
        settings=context.settings,
        model=context.model,
        optimizer=context.optimizer,
        numpy_generator=context.numpy_generator,
        train_generator=context.train_generator,
        runtime_identity=context.runtime_identity,
        step=step,
        metric_name="validation_l1_loss",
        metric_value=boundary_metric,
        scaler=context.scaler,
        amp=context.amp,
        distributed=context.distributed,
    )


def _save_interval_checkpoint(
    *,
    context: _CheckpointWriteContext,
    step: int,
    metric_value: float,
) -> CheckpointMetadata:
    return _save_checkpoint(
        path=context.request.output_dir / "checkpoints" / f"step_{step:06d}.pt",
        request=context.request,
        resolved=context.resolved,
        settings=context.settings,
        model=context.model,
        optimizer=context.optimizer,
        numpy_generator=context.numpy_generator,
        train_generator=context.train_generator,
        runtime_identity=context.runtime_identity,
        step=step,
        metric_value=metric_value,
        scaler=context.scaler,
        amp=context.amp,
        distributed=context.distributed,
    )


def _replace_metric_checkpoint_path(row: CsvRow, checkpoint_path: str) -> CsvRow:
    updated = dict(row)
    updated["checkpoint_path"] = checkpoint_path
    return updated


def _write_interval_artifact_flush(
    *,
    context: _IntervalFlushContext,
    model: nn.Module,
    local_state: _IntervalFlushState,
) -> None:
    """Durably write checkpoint-adjacent progress artifacts for long Kaggle runs.

    Raises:
        RuntimeError: if rank 0 cannot write the partial artifacts.

    """
    gathered_metric_rows = _merge_resume_csv_rows(
        prior_rows=local_state.resume_metric_rows,
        new_rows=_gather_csv_rows(local_state.metric_rows, context.distributed),
    )
    gathered_validation_rows = _merge_resume_csv_rows(
        prior_rows=local_state.resume_validation_rows,
        new_rows=_gather_csv_rows(local_state.validation_rows, context.distributed),
    )
    # Fixed-25 rows are canonical global rank-0 rows: merge the resume prefix but
    # NEVER all-gather them (Spec 0010 DDP contract; gathering would duplicate
    # them world_size times or force a collective around rank-0-only compute).
    merged_equivariance_rows = _merge_resume_csv_rows(
        prior_rows=local_state.resume_equivariance_rows,
        new_rows=local_state.equivariance_rows,
    )
    gate_rows = _gather_csv_rows(
        _gate_health_rows(
            run_name=context.settings.run_name,
            plan=context.plan,
            probe=context.distributed.probe,
            amp=context.amp,
            model=model,
            optimizer_step=local_state.current_step,
            rank=context.distributed.rank,
        ),
        context.distributed,
    )
    write_error: Exception | None = None
    write_error_message: str | None = None
    if _is_primary_rank(context.distributed):
        try:
            _write_partial_interval_artifacts(
                context=context,
                state=_IntervalFlushState(
                    metric_rows=gathered_metric_rows,
                    validation_rows=gathered_validation_rows,
                    gate_rows=gate_rows,
                    checkpoints=local_state.checkpoints,
                    best_checkpoint=local_state.best_checkpoint,
                    best_validation_metric=local_state.best_validation_metric,
                    last_result=local_state.last_result,
                    current_step=local_state.current_step,
                    equivariance_rows=merged_equivariance_rows,
                ),
            )
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - rank sync
            write_error = exc
            write_error_message = _exception_summary(exc)
    write_error_message = _broadcast_rank0_error(
        error_message=write_error_message,
        distributed=context.distributed,
    )
    if write_error_message is not None:
        message = (
            "selected-runtime interval artifact flush failed on rank 0: "
            f"{write_error_message}"
        )
        raise RuntimeError(message) from write_error


def _broadcast_rank0_error(
    *,
    error_message: str | None,
    distributed: _DistributedContext,
) -> str | None:
    if not distributed.should_use_ddp or not dist.is_initialized():
        return error_message
    payload: list[object] = [error_message if distributed.rank == 0 else None]
    broadcast_object_list = cast(
        "Callable[[list[object], int], None]",
        dist.broadcast_object_list,
    )
    broadcast_object_list(payload, 0)
    value = payload[0]
    return value if isinstance(value, str) else None


def _exception_summary(exc: Exception) -> str:
    return f"{exc.__class__.__name__}: {exc}" if str(exc) else exc.__class__.__name__


def _prepare_fixed25_runtime(  # noqa: PLR0913
    *,
    request: SelectedRuntimeTrainRequest,
    settings: _RunnerSettings,
    resolved: ResolvedConfig,
    data_surface: _DataSurface,
    distributed: _DistributedContext,
    model: nn.Module,
    artifacts: _RunArtifacts,
) -> _Fixed25Runtime | None:
    """Load the fixed-25 evaluation runtime for Spec 0010, or ``None`` if off.

    The protocol runs for real data (failing closed if the canonical selector is
    still the placeholder) or when an explicit synthetic selector override is
    provided; a plain synthetic smoke run without an override skips it. Writes the
    immutable originals plus the initial manifest on the primary rank.

    Returns:
        The loaded fixed-25 runtime, or ``None`` when the protocol is inactive.

    Raises:
        RuntimeError: If the rank-0 initial artifact write fails (broadcast so all
            ranks raise together).

    """
    config = parse_fixed25_config(
        resolved.effective_config,
        default_epsilon_seed=_seed(resolved.effective_config, "latent_seed"),
    )
    if config is None or not config.enabled:
        return None
    if data_surface.synthetic_generated and request.fixed_25_validation_patches is None:
        return None
    selector_path = request.fixed_25_validation_patches or Path(config.selector_config)
    validation_paths = resolve_patch_data_paths(data_surface.root).validation
    shard_spec = validation_shard_spec_for(
        validation_bin_path=validation_paths.bin_path,
        validation_csv_path=validation_paths.csv_path,
        image_size=settings.image_size,
        validate_crc=data_surface.validate_crc,
    )
    patches = load_fixed25_patches(
        config=config,
        selector_path=selector_path,
        validation_shard_spec=shard_spec,
        validation_dataset=data_surface.validation_dataset,
    )
    data_source = "synthetic" if data_surface.synthetic_generated else "real"
    promotable = not data_surface.synthetic_generated
    exactness = compute_rot90_exactness(
        model=cast("NonEquivariantVAE", model),
        patches=patches,
        device=distributed.device,
    )
    runtime = _Fixed25Runtime(
        config=config,
        patches=patches,
        fixed25_dir=artifacts.fixed25_dir,
        data_source=data_source,
        promotable=promotable,
        rot90_exactness_error=exactness,
    )
    write_error: Exception | None = None
    write_error_message: str | None = None
    if _is_primary_rank(distributed):
        try:
            write_originals(fixed25_dir=runtime.fixed25_dir, patches=patches)
            write_manifest(
                fixed25_dir=runtime.fixed25_dir,
                config=config,
                patches=patches,
                data_source=data_source,
                promotable=promotable,
                rot90_exactness_error=exactness,
            )
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - DDP sync guard
            write_error = exc
            write_error_message = _exception_summary(exc)
    # Broadcast a rank-0 write failure so all ranks raise together instead of the
    # peers hanging at the first training collective (mirrors the boundary flush).
    write_error_message = _broadcast_rank0_error(
        error_message=write_error_message,
        distributed=distributed,
    )
    if write_error_message is not None:
        message = (
            "selected-runtime fixed-25 initial artifact write failed on rank 0: "
            f"{write_error_message}"
        )
        raise RuntimeError(message) from write_error
    return runtime


def _evaluate_fixed25_boundary(
    *,
    fixed25: _Fixed25Runtime,
    model: nn.Module,
    distributed: _DistributedContext,
    optimizer_step: int,
) -> tuple[CsvRow, ...]:
    """Run the fixed-25 protocol for one boundary on rank 0 and broadcast errors.

    Returns:
        The per-angle equivariance rows (populated only on the primary rank; the
        canonical global rows are merged, never all-gathered).

    Raises:
        RuntimeError: If the rank-0 evaluation fails (broadcast to all ranks).

    """
    rows: tuple[CsvRow, ...] = ()
    eval_error: Exception | None = None
    eval_error_message: str | None = None
    if _is_primary_rank(distributed):
        try:
            rows = evaluate_boundary(
                model=cast("NonEquivariantVAE", model),
                patches=fixed25.patches,
                config=fixed25.config,
                fixed25_dir=fixed25.fixed25_dir,
                optimizer_step=optimizer_step,
                device=distributed.device,
                data_source=fixed25.data_source,
                promotable=fixed25.promotable,
            )
            write_manifest(
                fixed25_dir=fixed25.fixed25_dir,
                config=fixed25.config,
                patches=fixed25.patches,
                data_source=fixed25.data_source,
                promotable=fixed25.promotable,
                rot90_exactness_error=fixed25.rot90_exactness_error,
            )
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - DDP sync guard
            eval_error = exc
            eval_error_message = _exception_summary(exc)
    eval_error_message = _broadcast_rank0_error(
        error_message=eval_error_message,
        distributed=distributed,
    )
    if eval_error_message is not None:
        message = (
            "selected-runtime fixed-25 boundary evaluation failed on rank 0: "
            f"{eval_error_message}"
        )
        raise RuntimeError(message) from eval_error
    return rows


def _write_partial_interval_artifacts(
    *,
    context: _IntervalFlushContext,
    state: _IntervalFlushState,
) -> None:
    latest_checkpoint = state.checkpoints[-1] if state.checkpoints else None
    gate_health_summary = _gate_health_summary(state.gate_rows)
    plan_applied = _plan_applied_proof(
        plan=context.plan,
        settings=context.settings,
        probe=context.distributed.probe,
        amp=context.amp,
        ddp_proof=context.ddp_proof,
        metric_rows=state.metric_rows,
        last_result=state.last_result,
    )
    checkpoint_resume_proof = _partial_checkpoint_resume_proof(
        checkpoint=latest_checkpoint,
        context=context,
        current_step=state.current_step,
    )
    local_readiness = _local_readiness_summary(
        _LocalReadinessComponents(
            plan_applied=plan_applied,
            checkpoint_resume_proof=checkpoint_resume_proof,
            gate_health_summary=gate_health_summary,
            data_source=context.data_surface.source,
            ddp_proof=context.ddp_proof,
            amp_step_skipped_count=_amp_step_skipped_count(state.metric_rows),
            nonfinite_count=_nonfinite_metric_count(state.metric_rows),
        ),
    )

    _write_csv_atomic(
        context.artifacts.train_steps,
        _TRAIN_STEP_COLUMNS,
        state.metric_rows,
    )
    if _writes_validation_metrics(context.settings):
        _write_csv_atomic(
            context.artifacts.validation_metrics,
            _VALIDATION_METRIC_COLUMNS,
            state.validation_rows,
        )
    if state.equivariance_rows:
        _write_csv_atomic(
            context.artifacts.equivariance_25,
            EQUIVARIANCE_25_COLUMNS,
            state.equivariance_rows,
        )
    _write_csv_atomic(
        context.artifacts.gate_health,
        GATE_HEALTH_COLUMNS,
        state.gate_rows,
    )
    _write_json_atomic(context.artifacts.selected_runtime_plan_applied, plan_applied)
    _write_json_atomic(
        context.artifacts.checkpoint_resume_proof,
        checkpoint_resume_proof,
    )
    _write_json_atomic(context.artifacts.gate_health_summary, gate_health_summary)
    _write_json_atomic(context.artifacts.local_readiness, local_readiness)
    _write_json_atomic(
        context.artifacts.training_summary,
        _partial_training_summary(
            context=context,
            state=state,
        ),
    )
    if _is_full_run(context.settings):
        _write_json_atomic(
            context.artifacts.selected_runtime_full_summary,
            _partial_selected_runtime_full_summary(
                context=context,
                state=state,
                plan_applied=plan_applied,
                checkpoint_resume_proof=checkpoint_resume_proof,
                gate_health_summary=gate_health_summary,
            ),
        )
    _write_json_atomic(
        context.artifacts.artifact_manifest,
        _partial_artifact_manifest(
            artifacts=context.artifacts,
            settings=context.settings,
            state=state,
        ),
    )


def _partial_checkpoint_resume_proof(
    *,
    checkpoint: CheckpointMetadata | None,
    context: _IntervalFlushContext,
    current_step: int,
) -> JsonObject:
    checkpoint_step = (
        0 if checkpoint is None else checkpoint.successful_optimizer_update_count
    )
    return cast(
        "JsonObject",
        {
            "status": _FAIL,
            "status_scope": _status_scope(context.settings),
            "full_run_eligible": False,
            "partial_artifact_flush": True,
            "partial_artifact_flush_step": current_step,
            "latest_metric_prefix_step": current_step,
            "latest_checkpoint_step": checkpoint_step,
            "resume_checkpoint": ""
            if checkpoint is None
            else _relative_to_output(checkpoint.path, context.request.output_dir),
            "resume_checkpoint_sha256": "" if checkpoint is None else checkpoint.sha256,
            "loaded_successful_optimizer_update_count": checkpoint_step,
            "final_optimizer_step": context.settings.max_train_steps,
            "additional_optimizer_steps": max(
                0,
                context.settings.max_train_steps - checkpoint_step,
            ),
            "model_state_checkpointed": checkpoint is not None,
            "optimizer_state_checkpointed": checkpoint is not None,
            "grad_scaler_state_checkpointed": context.amp.grad_scaler_enabled,
            "cuda_rng_state_checkpointed": context.distributed.device.type == "cuda",
            "sampler_progress_checkpointed": context.distributed.should_use_ddp,
            "optimizer_scheduler_progress_checkpointed": checkpoint is not None,
            "beta_progress_checkpointed": checkpoint is not None,
            "checkpoint_restore_probe_deferred": True,
            "failure_kind": "partial_interval_checkpoint_not_final_resume_proof",
        },
    )


def _partial_training_summary(
    *,
    context: _IntervalFlushContext,
    state: _IntervalFlushState,
) -> JsonObject:
    payload = cast(
        "JsonObject",
        {
            "status": _FAIL,
            "status_scope": _status_scope(context.settings),
            "proof_scope": _status_scope(context.settings),
            "full_run_eligible": False,
            "partial_artifact_flush": True,
            "partial_artifact_flush_step": state.current_step,
            "failure_kind": "partial_full_run_interval_artifacts",
            "run_name": context.settings.run_name,
            "run_mode": context.settings.run_mode,
            "data": context.request.data,
            "data_root": str(context.data_surface.root),
            "synthetic_generated": context.data_surface.synthetic_generated,
            "config_path": str(context.request.config_path),
            "runtime_config": {
                "path": str(context.runtime_identity.path),
                "sha256": context.runtime_identity.sha256,
                "selected_row_id": context.runtime_identity.selected_row_id,
                "runtime_policy_id": context.runtime_identity.runtime_policy_id,
                "per_device_batch_size": context.plan.per_device_batch_size,
                "global_batch_size": context.plan.global_batch_size,
                "precision_policy": context.plan.precision_policy,
                "corruption_strategy": context.plan.corruption_strategy,
                "consumed": True,
            },
            "selected_runtime_launch_command": context.launch_command.shell_command,
            "ddp_rank_device_proof": context.ddp_proof,
            "amp_execution": {
                "enabled": context.amp.enabled,
                "grad_scaler_enabled": context.amp.grad_scaler_enabled,
                "grad_scaler_init_scale": context.amp.grad_scaler_init_scale,
                "autocast_dtype": context.amp.autocast_dtype,
                "requested_autocast_dtype": context.amp.requested_autocast_dtype,
                "local_amp_status": context.amp.local_amp_status,
                "fp32_objective_island": True,
            },
            "max_train_steps": context.settings.max_train_steps,
            "target_optimizer_updates": context.settings.target_train_steps,
            "requested_epochs": context.settings.requested_epochs,
            "optimizer_updates_per_epoch": context.settings.optimizer_updates_per_epoch,
            "half_epoch_interval_steps": context.settings.half_epoch_interval_steps,
            "current_epoch_fraction": _current_epoch_fraction(
                context.settings,
                state.metric_rows,
            ),
            "max_val_steps": context.settings.max_val_steps,
            "save_every_steps": context.settings.save_every_steps,
            "validation_batches_per_view": context.settings.validation_batches_per_view,
            "validation_views": list(context.settings.validation_views),
            "train_reparameterization": context.settings.train_reparameterization,
            "deterministic_eps_allowed_for": list(
                context.settings.deterministic_eps_allowed_for,
            ),
            "checkpoint_retention": context.settings.checkpoint_retention,
            "resume_supported": context.settings.resume_supported,
            "optimizer_steps_completed": _successful_optimizer_update_count(
                state.metric_rows,
            ),
            "metric_row_count": len(state.metric_rows),
            "validation_metric_row_count": len(state.validation_rows),
            "amp_step_skipped_count": _amp_step_skipped_count(state.metric_rows),
            "nonfinite_count": _nonfinite_metric_count(state.metric_rows),
            "checkpoint_count": len(state.checkpoints),
            "retained_interval_checkpoint_count": _interval_checkpoint_count(
                state.checkpoints,
            ),
            "retained_interval_checkpoints": _checkpoint_payloads(
                state.checkpoints,
                context.request.output_dir,
            ),
            "metrics_csv": "metrics/train_steps.csv",
            "train_steps_csv": "metrics/train_steps.csv",
            "validation_metrics_csv": "metrics/validation_metrics.csv"
            if _writes_validation_metrics(context.settings)
            else "",
            "gate_health_csv": "metrics/gate_health.csv",
            "selected_runtime_plan_applied": (
                "benchmark/selected_runtime_plan_applied.json"
            ),
            "checkpoint_resume_proof": "benchmark/checkpoint_resume_proof.json",
            "gate_health_summary": "benchmark/gate_health_summary.json",
            "artifact_manifest": "benchmark/artifact_manifest.json",
            "reconstruction_sample_nonblank": False,
            "last_loss": cast(
                "JsonObject",
                state.last_result.losses.detached_scalars(),
            ),
            "last_train_eps_policy": state.last_result.eps_policy,
            "last_train_eps_zero_fraction": state.last_result.eps_zero_fraction,
        },
    )
    if state.best_checkpoint is not None:
        payload["best_checkpoint"] = _checkpoint_payload(
            state.best_checkpoint,
            context.request.output_dir,
        )
        payload["best_validation_metric"] = state.best_validation_metric
    return payload


def _partial_selected_runtime_full_summary(
    *,
    context: _IntervalFlushContext,
    state: _IntervalFlushState,
    plan_applied: JsonObject,
    checkpoint_resume_proof: JsonObject,
    gate_health_summary: JsonObject,
) -> JsonObject:
    summary = _selected_runtime_full_summary(
        plan=context.plan,
        settings=context.settings,
        plan_applied=plan_applied,
        ddp_proof=context.ddp_proof,
        amp=context.amp,
        checkpoint_resume_proof=checkpoint_resume_proof,
        gate_health_summary=gate_health_summary,
        data_surface=context.data_surface,
        metric_rows=state.metric_rows,
        validation_rows=state.validation_rows,
        checkpoints=state.checkpoints,
        best_validation_metric=state.best_validation_metric,
    )
    summary["partial_artifact_flush"] = True
    summary["partial_artifact_flush_step"] = state.current_step
    blockers = summary.get("launch_blockers_remaining")
    if isinstance(blockers, list):
        blockers.append("partial_artifact_flush_not_complete")
    else:
        summary["launch_blockers_remaining"] = ["partial_artifact_flush_not_complete"]
    summary["full_run_eligible"] = False
    summary["remote_pass_ready"] = False
    summary["status"] = _FAIL
    summary["failure_kind"] = "partial_full_run_interval_artifacts"
    return summary


def _partial_artifact_manifest(
    *,
    artifacts: _RunArtifacts,
    settings: _RunnerSettings,
    state: _IntervalFlushState,
) -> JsonObject:
    artifact_paths = {
        "training_summary": artifacts.training_summary,
        "selected_runtime_full_summary": artifacts.selected_runtime_full_summary,
        "selected_runtime_plan_applied": artifacts.selected_runtime_plan_applied,
        "checkpoint_resume_proof": artifacts.checkpoint_resume_proof,
        "gate_health_summary": artifacts.gate_health_summary,
        "local_selected_runtime_readiness": artifacts.local_readiness,
        "train_steps": artifacts.train_steps,
        "validation_metrics": artifacts.validation_metrics,
        "gate_health": artifacts.gate_health,
    }
    for checkpoint in state.checkpoints:
        artifact_paths[f"checkpoint:{checkpoint.path.name}"] = checkpoint.path
    if state.best_checkpoint is not None:
        artifact_paths[f"checkpoint:{state.best_checkpoint.path.name}"] = (
            state.best_checkpoint.path
        )
    missing = [
        name for name, path in sorted(artifact_paths.items()) if not path.exists()
    ]
    return cast(
        "JsonObject",
        {
            "status": _FAIL,
            "status_scope": _status_scope(settings),
            "full_run_eligible": False,
            "partial_artifact_flush": True,
            "partial_artifact_flush_step": state.current_step,
            "artifact_hashes": cast(
                "JsonObject",
                {
                    name: _sha256_file(path)
                    for name, path in sorted(artifact_paths.items())
                    if path.exists()
                },
            ),
            "missing_artifacts": missing,
            "checkpoint_count": len(state.checkpoints),
            "metric_row_count": len(state.metric_rows),
            "reconstruction_sample_nonblank": False,
        },
    )


def _write_json_atomic(path: Path, payload: JsonObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    _fsync_file(tmp_path)
    tmp_path.replace(path)
    _fsync_directory(path.parent)


def _write_csv_atomic(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[CsvRow],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
        csv_file.flush()
        os.fsync(csv_file.fileno())
    tmp_path.replace(path)
    _fsync_directory(path.parent)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as file_obj:
        os.fsync(file_obj.fileno())


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _advance_batches(
    batches: Iterator[PatchTrainingBatch],
    count: int,
) -> None:
    for _ in range(count):
        next(batches)


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
    train_generator: torch.Generator,
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
    eps, eps_proof = _train_eps(
        batch_size=input_batch.shape[0],
        settings=settings,
        train_generator=train_generator,
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
        train_reparameterization=settings.train_reparameterization,
        eps_policy=eps_proof.eps_policy,
        eps_seed_source=eps_proof.eps_seed_source,
        eps_zero_fraction=eps_proof.eps_zero_fraction,
        eps_abs_mean=eps_proof.eps_abs_mean,
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


def _should_run_scheduled_validation(
    settings: _RunnerSettings,
    optimizer_step: int,
) -> bool:
    return (
        _writes_validation_metrics(settings)
        and optimizer_step > 0
        and optimizer_step % settings.half_epoch_interval_steps == 0
    )


def _full_boundary_epoch_fraction(
    settings: _RunnerSettings,
    optimizer_step: int,
) -> float:
    if settings.optimizer_updates_per_epoch <= 0:
        return 0.0
    return optimizer_step / float(settings.optimizer_updates_per_epoch)


def _log_full_boundary_start(
    *,
    settings: _RunnerSettings,
    distributed: _DistributedContext,
    optimizer_step: int,
) -> None:
    if not _is_full_run(settings) or not _is_primary_rank(distributed):
        return
    epoch_fraction = _full_boundary_epoch_fraction(settings, optimizer_step)
    validation_views = ",".join(settings.validation_views)
    print(  # noqa: T201 - Kaggle logs need explicit half-epoch breadcrumbs.
        "[RANK 0] selected-runtime full boundary start: "
        f"update {optimizer_step}/{settings.target_train_steps} "
        f"(epoch {epoch_fraction:.1f}); validation_views={validation_views}; "
        "will flush metrics before checkpoint and refresh manifest after checkpoint.",
        flush=True,
    )


def _synchronize_full_boundary_completion(
    *,
    settings: _RunnerSettings,
    distributed: _DistributedContext,
    optimizer_step: int,
) -> None:
    if not _is_full_run(settings):
        return
    epoch_fraction = _full_boundary_epoch_fraction(settings, optimizer_step)
    print(  # noqa: T201 - Kaggle logs need explicit half-epoch breadcrumbs.
        f"[RANK {distributed.rank}] selected-runtime full boundary complete: "
        f"update {optimizer_step}/{settings.target_train_steps} "
        f"(epoch {epoch_fraction:.1f}); waiting at boundary barrier.",
        flush=True,
    )
    _barrier(distributed)
    print(  # noqa: T201 - Kaggle logs need explicit half-epoch breadcrumbs.
        f"[RANK {distributed.rank}] selected-runtime full boundary barrier resolved: "
        f"update {optimizer_step}/{settings.target_train_steps}; resuming training.",
        flush=True,
    )


def _run_scheduled_validation(  # noqa: PLR0913
    *,
    model: nn.Module,
    settings: _RunnerSettings,
    plan: SelectedRuntimePlan,
    amp: _AmpExecution,
    data_surface: _DataSurface,
    optimizer_step: int,
    rank: int,
    device: torch.device,
) -> tuple[CsvRow, ...]:
    was_training = model.training
    model.eval()
    rows: list[CsvRow] = []
    try:
        rows.extend(
            _validation_view_row(
                model=model,
                settings=settings,
                plan=plan,
                amp=amp,
                data_surface=data_surface,
                optimizer_step=optimizer_step,
                view=view,
                rank=rank,
                device=device,
            )
            for view in settings.validation_views
        )
    finally:
        if was_training:
            model.train()
    return tuple(rows)


def _validation_view_row(  # noqa: PLR0913
    *,
    model: nn.Module,
    settings: _RunnerSettings,
    plan: SelectedRuntimePlan,
    amp: _AmpExecution,
    data_surface: _DataSurface,
    optimizer_step: int,
    view: str,
    rank: int,
    device: torch.device,
) -> CsvRow:
    scalars: list[dict[str, float]] = []
    sample_count = 0
    validation_batches = _cycle_batches(data_surface.validation_loader)
    for _batch_index in range(settings.validation_batches_per_view):
        batch = next(validation_batches)
        clean_batch_cpu = normalize_uint8_batch(batch.images_uint8)
        if view == "clean":
            input_batch_cpu = clean_validation_passthrough(clean_batch_cpu)
        elif view == "deterministic_denoising":
            input_batch_cpu = corrupt_normalized_batch(
                clean_batch_cpu,
                profile=settings.corruption_profile,
                corruption_seed=settings.corruption_seed,
                split=batch.split,
                semantic_sample_keys=batch.semantic_sample_keys,
                corruption_step=optimizer_step,
                corruption_view=f"validation_{view}",
                strategy=plan.corruption_strategy,
            ).corrupted
        else:
            message = f"unsupported validation view: {view}"
            raise ValueError(message)
        clean_batch = _to_device(clean_batch_cpu, device=device, plan=plan)
        input_batch = _to_device(input_batch_cpu, device=device, plan=plan)
        eps = _zero_eps(
            batch_size=input_batch.shape[0],
            settings=settings,
            device=device,
        )
        beta = beta_for_step(
            optimizer_step_index=max(0, optimizer_step - 1),
            max_optimizer_steps=settings.target_train_steps,
            target_beta=settings.beta_target,
            warmup_fraction=settings.beta_warmup_fraction,
        )
        dtype = _autocast_dtype(plan.autocast_dtype)
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=device.type,
                dtype=dtype,
                enabled=amp.enabled,
            ),
        ):
            output = cast("NonEquivariantVAE", model).forward(input_batch, eps=eps)
            losses = compute_vae_loss(
                output,
                clean_batch,
                beta=beta,
                ssim_weight=settings.ssim_weight,
            )
        scalars.append(losses.detached_scalars())
        sample_count += int(input_batch.shape[0])
    means = _mean_loss_scalars(scalars)
    return {
        "event_id": f"rank{rank}_validation_{view}_{optimizer_step:06d}",
        "rank": str(rank),
        "optimizer_step": str(optimizer_step),
        "validation_boundary": "half_epoch",
        "split": "validation",
        "view": view,
        "batch_count": str(settings.validation_batches_per_view),
        "sample_count": str(sample_count),
        "loss": _format_float(means["loss"]),
        "recon_loss": _format_float(means["recon_loss"]),
        "l1_loss": _format_float(means["l1_loss"]),
        "ssim_loss": _format_float(means["ssim_loss"]),
        "ssim_metric": _format_float(means["ssim_metric"]),
        "kl_loss": _format_float(means["kl_loss"]),
        "beta": _format_float(means["beta"]),
        "deterministic_eps_used": "true",
        "corruption_strategy": plan.corruption_strategy,
    }


def _mean_loss_scalars(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    keys = (
        "loss",
        "recon_loss",
        "l1_loss",
        "ssim_loss",
        "ssim_metric",
        "kl_loss",
        "beta",
    )
    return {key: _mean([row[key] for row in rows]) for key in keys}


def _validation_best_l1(rows: Sequence[CsvRow]) -> float | None:
    values = [float(row["l1_loss"]) for row in rows if row.get("l1_loss")]
    return min(values) if values else None


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
    metric_name: str = "l1_loss",
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
        metric_name=metric_name,
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
    plan: SelectedRuntimePlan,
    train_generator: torch.Generator,
    amp: _AmpExecution,
    distributed: _DistributedContext,
) -> JsonObject:
    model = build_non_equivariant_vae(norm_groups=settings.norm_groups)
    model = _place_model(model=model, plan=plan, device=distributed.device)
    optimizer, _ = create_adamw_optimizer(model, config=settings.optimizer_config)
    numpy_generator = np.random.default_rng(settings.global_seed)
    probe_scaler = GradScaler(
        "cuda",
        init_scale=amp.grad_scaler_init_scale,
        enabled=amp.grad_scaler_enabled,
    )
    loaded = load_training_checkpoint(
        path=checkpoint.path,
        model=model,
        optimizer=optimizer,
        numpy_generator=numpy_generator,
        torch_generators={"train_data": train_generator},
        amp_scaler=probe_scaler if amp.grad_scaler_enabled else None,
        restore_cuda_rng=distributed.device.type == "cuda",
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
            "status_scope": _status_scope(settings),
            "full_run_eligible": _is_full_run(settings),
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
            "grad_scaler_state_restore_attempted": amp.grad_scaler_enabled,
            "grad_scaler_state_restored": amp_status_ok,
            "cuda_rng_state_restore_attempted": distributed.device.type == "cuda",
            "cuda_rng_state_restored": cuda_status_ok,
            "sampler_progress_restored": ddp_status_ok,
            "sampler_progress_offset_batches": loaded.successful_optimizer_update_count,
            "optimizer_scheduler_progress_restored": True,
            "beta_progress_restored": (
                loaded.beta_progress_state_status
                == "deterministic_from_successful_optimizer_update_count"
            ),
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
            "status_scope": _status_scope(settings),
            "full_run_eligible": _is_full_run(settings),
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
            "grad_scaler_state_restore_attempted": amp.grad_scaler_enabled,
            "grad_scaler_state_restored": amp_status_ok,
            "cuda_rng_state_restore_attempted": distributed.device.type == "cuda",
            "cuda_rng_state_restored": cuda_status_ok,
            "sampler_progress_restored": ddp_status_ok,
            "sampler_progress_offset_batches": loaded.successful_optimizer_update_count,
            "optimizer_scheduler_progress_restored": True,
            "beta_progress_restored": (
                loaded.beta_progress_state_status
                == "deterministic_from_successful_optimizer_update_count"
            ),
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
        status_scope=_status_scope(settings),
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
    validation_rows: Sequence[CsvRow],
    checkpoints: Sequence[CheckpointMetadata],
    final_checkpoint: CheckpointMetadata,
    best_checkpoint: CheckpointMetadata,
    best_validation_metric: float | None,
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
            "status_scope": _status_scope(settings),
            "proof_scope": _status_scope(settings),
            "full_run_eligible": _full_run_artifacts_eligible(
                settings=settings,
                request=request,
                metric_rows=metric_rows,
                validation_rows=validation_rows,
                plan_applied=plan_applied,
                gate_health_summary=gate_health_summary,
            ),
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
            "target_optimizer_updates": settings.target_train_steps,
            "requested_epochs": settings.requested_epochs,
            "optimizer_updates_per_epoch": settings.optimizer_updates_per_epoch,
            "half_epoch_interval_steps": settings.half_epoch_interval_steps,
            "current_epoch_fraction": _current_epoch_fraction(settings, metric_rows),
            "max_val_steps": settings.max_val_steps,
            "save_every_steps": settings.save_every_steps,
            "validation_batches_per_view": settings.validation_batches_per_view,
            "validation_views": list(settings.validation_views),
            "train_reparameterization": settings.train_reparameterization,
            "deterministic_eps_allowed_for": list(
                settings.deterministic_eps_allowed_for,
            ),
            "checkpoint_retention": settings.checkpoint_retention,
            "resume_supported": settings.resume_supported,
            "optimizer_steps_completed": _successful_optimizer_update_count(
                metric_rows,
            ),
            "resumed_optimizer_update_count": _resume_start_step_from_request(request),
            "sampler_progress_offset_batches": _resume_start_step_from_request(request),
            "metric_row_count": len(metric_rows),
            "validation_metric_row_count": len(validation_rows),
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
            "best_validation_metric": best_validation_metric,
            "retained_interval_checkpoint_count": _interval_checkpoint_count(
                checkpoints,
            ),
            "retained_interval_checkpoints": _checkpoint_payloads(
                checkpoints,
                request.output_dir,
            ),
            "metrics_csv": "metrics/train_steps.csv",
            "train_steps_csv": "metrics/train_steps.csv",
            "validation_metrics_csv": "metrics/validation_metrics.csv"
            if _writes_validation_metrics(settings)
            else "",
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
            "last_train_eps_policy": last_result.eps_policy,
            "last_train_eps_zero_fraction": last_result.eps_zero_fraction,
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


def _selected_runtime_full_summary(  # noqa: PLR0913
    *,
    plan: SelectedRuntimePlan,
    settings: _RunnerSettings,
    plan_applied: JsonObject,
    ddp_proof: JsonObject,
    amp: _AmpExecution,
    checkpoint_resume_proof: JsonObject,
    gate_health_summary: JsonObject,
    data_surface: _DataSurface,
    metric_rows: Sequence[CsvRow],
    validation_rows: Sequence[CsvRow],
    checkpoints: Sequence[CheckpointMetadata],
    best_validation_metric: float | None,
) -> JsonObject:
    completed_steps = _successful_optimizer_update_count(metric_rows)
    validation_boundaries = _validation_boundary_steps(validation_rows)
    blockers: list[str] = []
    if completed_steps != settings.target_train_steps:
        blockers.append("target_optimizer_updates_not_completed")
    if _amp_step_skipped_count(metric_rows) != 0:
        blockers.append("amp_step_skip_observed")
    if _nonfinite_metric_count(metric_rows) != 0:
        blockers.append("nonfinite_train_metric_observed")
    if not _stochastic_train_eps_proven(metric_rows):
        blockers.append("stochastic_seeded_train_epsilon_not_proven")
    if not _validation_schedule_complete(settings, validation_rows):
        blockers.append("half_epoch_validation_schedule_incomplete")
    if _interval_checkpoint_count(checkpoints) > _FULL_INTERVAL_CHECKPOINT_KEEP_COUNT:
        blockers.append("too_many_interval_checkpoints_retained")
    if plan_applied.get("status") != _LOCAL_STATUS:
        blockers.append("selected_runtime_plan_not_applied")
    if gate_health_summary.get("status") != _LOCAL_STATUS:
        blockers.append("gate_health_not_pass")
    if data_surface.source != "ubc-pre-shuffled":
        blockers.append("synthetic_or_wrong_data_surface")
    status = _LOCAL_STATUS if not blockers else _FAIL
    return cast(
        "JsonObject",
        {
            "status": status,
            "status_scope": _FULL_STATUS_SCOPE,
            "full_run_eligible": status == _LOCAL_STATUS,
            "real_train_runner_implemented": True,
            "remote_pass_ready": status == _LOCAL_STATUS,
            "selected_row_id": plan.selected_row_id,
            "runtime_policy_id": plan.runtime_policy_id,
            "per_device_batch_size": settings.batch_size,
            "global_batch_size": plan.global_batch_size,
            "target_optimizer_updates": settings.target_train_steps,
            "optimizer_steps_completed": completed_steps,
            "requested_epochs": settings.requested_epochs,
            "optimizer_updates_per_epoch": settings.optimizer_updates_per_epoch,
            "half_epoch_interval_steps": settings.half_epoch_interval_steps,
            "validation_batches_per_view": settings.validation_batches_per_view,
            "validation_views": list(settings.validation_views),
            "validation_boundary_steps": list(validation_boundaries),
            "train_reparameterization": settings.train_reparameterization,
            "stochastic_train_eps_proven": _stochastic_train_eps_proven(metric_rows),
            "checkpoint_retention": settings.checkpoint_retention,
            "retained_interval_checkpoint_count": _interval_checkpoint_count(
                checkpoints,
            ),
            "retained_interval_checkpoints": [
                checkpoint.path.name for checkpoint in checkpoints
            ],
            "best_validation_metric": best_validation_metric,
            "amp_execution_status": amp.local_amp_status,
            "grad_scaler_init_scale": amp.grad_scaler_init_scale,
            "ddp_rank_device_status": _string_value(ddp_proof.get("status")),
            "selected_runtime_plan_applied_status": _string_value(
                plan_applied.get("status"),
            ),
            "checkpoint_resume_proof_status": _string_value(
                checkpoint_resume_proof.get("status"),
            ),
            "gate_health_status": _string_value(gate_health_summary.get("status")),
            "data_source": data_surface.source,
            "launch_blockers_remaining": blockers,
            "failure_kind": "" if not blockers else "full_run_artifacts_not_verified",
            "selected_runtime_full_run_contract_ready": True,
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


def _is_full_run(settings: _RunnerSettings) -> bool:
    return settings.run_mode == _FULL_RUN_MODE


def _status_scope(settings: _RunnerSettings) -> str:
    return _FULL_STATUS_SCOPE if _is_full_run(settings) else _STATUS_SCOPE


def _writes_validation_metrics(settings: _RunnerSettings) -> bool:
    return bool(settings.validation_views)


def _full_run_artifacts_eligible(  # noqa: PLR0913
    *,
    settings: _RunnerSettings,
    request: SelectedRuntimeTrainRequest,
    metric_rows: Sequence[CsvRow],
    validation_rows: Sequence[CsvRow],
    plan_applied: JsonObject,
    gate_health_summary: JsonObject,
) -> bool:
    return (
        _is_full_run(settings)
        and request.data == "ubc-pre-shuffled"
        and _successful_optimizer_update_count(metric_rows)
        == settings.target_train_steps
        and _amp_step_skipped_count(metric_rows) == 0
        and _nonfinite_metric_count(metric_rows) == 0
        and _stochastic_train_eps_proven(metric_rows)
        and _validation_schedule_complete(settings, validation_rows)
        and plan_applied.get("status") == _LOCAL_STATUS
        and gate_health_summary.get("status") == _LOCAL_STATUS
    )


def _current_epoch_fraction(
    settings: _RunnerSettings,
    metric_rows: Sequence[CsvRow],
) -> float:
    if settings.optimizer_updates_per_epoch <= 0:
        return 0.0
    return _successful_optimizer_update_count(metric_rows) / float(
        settings.optimizer_updates_per_epoch,
    )


def _resume_start_step_from_request(request: SelectedRuntimeTrainRequest) -> int:
    if request.resume is None:
        return 0
    metadata = read_training_checkpoint_metadata(path=request.resume)
    return metadata.successful_optimizer_update_count


def _stochastic_train_eps_proven(rows: Sequence[CsvRow]) -> bool:
    successful_rows = _successful_metric_rows(rows)
    return bool(successful_rows) and all(
        row.get("train_reparameterization") == _STOCHASTIC_REPARAMETERIZATION
        and row.get("eps_policy") == "stochastic_seeded_train_generator"
        and float(row.get("eps_abs_mean", "0") or "0") > 0.0
        and float(row.get("eps_zero_fraction", "1") or "1") < 1.0
        for row in successful_rows
    )


def _validation_schedule_complete(
    settings: _RunnerSettings,
    rows: Sequence[CsvRow],
) -> bool:
    if not _writes_validation_metrics(settings):
        return True
    expected_steps = tuple(
        range(
            settings.half_epoch_interval_steps,
            settings.target_train_steps + 1,
            settings.half_epoch_interval_steps,
        ),
    )
    observed = {
        (int(row["optimizer_step"]), row["view"])
        for row in rows
        if row.get("optimizer_step") and row.get("view")
    }
    return all(
        (step, view) in observed
        for step in expected_steps
        for view in settings.validation_views
    )


def _validation_boundary_steps(rows: Sequence[CsvRow]) -> tuple[int, ...]:
    return tuple(
        sorted({
            int(row["optimizer_step"]) for row in rows if row.get("optimizer_step")
        }),
    )


def _checkpoint_payloads(
    checkpoints: Sequence[CheckpointMetadata],
    output_dir: Path,
) -> list[JsonObject]:
    return [_checkpoint_payload(checkpoint, output_dir) for checkpoint in checkpoints]


def _interval_checkpoint_count(checkpoints: Sequence[CheckpointMetadata]) -> int:
    return sum(
        1 for checkpoint in checkpoints if checkpoint.path.name.startswith("step_")
    )


def _writes_tiny_summary(settings: _RunnerSettings) -> bool:
    return settings.run_mode == _TINY_RUN_MODE


def _artifact_manifest(  # noqa: PLR0913
    *,
    artifacts: _RunArtifacts,
    settings: _RunnerSettings,
    checkpoints: Sequence[CheckpointMetadata],
    metric_rows: Sequence[CsvRow],
    reconstruction_nonblank: bool,
    fixed25: _Fixed25Runtime | None = None,
) -> JsonObject:
    artifact_paths = {
        "training_summary": artifacts.training_summary,
        "selected_runtime_plan_applied": artifacts.selected_runtime_plan_applied,
        "checkpoint_resume_proof": artifacts.checkpoint_resume_proof,
        "gate_health_summary": artifacts.gate_health_summary,
        "local_selected_runtime_readiness": artifacts.local_readiness,
        "train_steps": artifacts.train_steps,
        "gate_health": artifacts.gate_health,
    }
    if fixed25 is not None:
        # Spec 0010: the fixed-25 artifacts replace the retired single-patch dump.
        artifact_paths["equivariance_25"] = artifacts.equivariance_25
        artifact_paths["fixed25_originals"] = artifacts.fixed25_dir / "originals.pt"
        artifact_paths["fixed25_manifest"] = artifacts.fixed25_dir / "manifest.json"
    else:
        artifact_paths["reconstruction_samples"] = artifacts.reconstruction_samples
    if _is_full_run(settings):
        artifact_paths["selected_runtime_full_summary"] = (
            artifacts.selected_runtime_full_summary
        )
    else:
        artifact_paths["selected_runtime_debug_summary"] = (
            artifacts.selected_runtime_debug_summary
        )
    if _writes_validation_metrics(settings):
        artifact_paths["validation_metrics"] = artifacts.validation_metrics
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
            "status_scope": _status_scope(settings),
            "full_run_eligible": (
                _is_full_run(settings)
                and not missing
                and _successful_optimizer_update_count(metric_rows)
                == settings.target_train_steps
            ),
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
        "train_reparameterization": result.train_reparameterization,
        "eps_policy": result.eps_policy,
        "eps_seed_source": result.eps_seed_source,
        "eps_zero_fraction": _format_float(result.eps_zero_fraction),
        "eps_abs_mean": _format_float(result.eps_abs_mean),
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


def _train_eps(
    *,
    batch_size: int,
    settings: _RunnerSettings,
    train_generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, _EpsProof]:
    shape = (
        batch_size,
        LATENT_CHANNELS,
        settings.image_size // 8,
        settings.image_size // 8,
    )
    if settings.train_reparameterization == _STOCHASTIC_REPARAMETERIZATION:
        eps_cpu = torch.randn(
            shape,
            generator=train_generator,
            dtype=torch.float32,
            device="cpu",
        )
        eps = eps_cpu.to(device=device)
        return eps, _eps_proof(
            eps_cpu,
            eps_policy="stochastic_seeded_train_generator",
            eps_seed_source="train_data_torch_generator",
        )
    eps = _zero_eps(batch_size=batch_size, settings=settings, device=device)
    return eps, _eps_proof(
        eps.detach().cpu(),
        eps_policy="deterministic_zero",
        eps_seed_source="fixed_zero_tensor",
    )


def _eps_proof(
    eps: torch.Tensor,
    *,
    eps_policy: str,
    eps_seed_source: str,
) -> _EpsProof:
    values = eps.detach().float()
    zero_fraction = float(torch.count_nonzero(values == 0).item()) / float(
        values.numel(),
    )
    return _EpsProof(
        eps_policy=eps_policy,
        eps_seed_source=eps_seed_source,
        eps_zero_fraction=zero_fraction,
        eps_abs_mean=float(values.abs().mean().item()),
    )


def _apply_checkpoint_retention(
    *,
    checkpoints: Sequence[CheckpointMetadata],
    settings: _RunnerSettings,
) -> tuple[CheckpointMetadata, ...]:
    if (
        not _is_full_run(settings)
        or settings.checkpoint_retention != _FULL_CHECKPOINT_RETENTION
    ):
        return tuple(checkpoints)
    retained = tuple(
        sorted(
            checkpoints,
            key=lambda checkpoint: checkpoint.successful_optimizer_update_count,
        )[-_FULL_INTERVAL_CHECKPOINT_KEEP_COUNT:],
    )
    retained_paths = {checkpoint.path for checkpoint in retained}
    for checkpoint in checkpoints:
        if checkpoint.path not in retained_paths and checkpoint.path.exists():
            checkpoint.path.unlink()
    return retained


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


def _validate_settings(  # noqa: C901
    settings: _RunnerSettings,
    *,
    dry_run: bool,
) -> None:
    if settings.batch_size <= 0:
        message = f"batch_size must be positive, got {settings.batch_size}"
        raise ValueError(message)
    if settings.image_size <= 0 or settings.image_size % 8 != 0:
        message = "image_size must be positive and divisible by 8"
        raise ValueError(message)
    if settings.max_train_steps <= 0:
        message = f"max_train_steps must be positive, got {settings.max_train_steps}"
        raise ValueError(message)
    if settings.target_train_steps <= 0:
        message = (
            f"target_train_steps must be positive, got {settings.target_train_steps}"
        )
        raise ValueError(message)
    if settings.max_train_steps > settings.target_train_steps:
        message = "max_train_steps cannot exceed target_train_steps"
        raise ValueError(message)
    if settings.max_val_steps < 0:
        message = f"max_val_steps must be nonnegative, got {settings.max_val_steps}"
        raise ValueError(message)
    if settings.save_every_steps <= 0:
        message = f"save_every_steps must be positive, got {settings.save_every_steps}"
        raise ValueError(message)
    if settings.validation_views and settings.validation_batches_per_view <= 0:
        message = "validation_batches_per_view must be positive when views are set"
        raise ValueError(message)
    if settings.validation_views and settings.half_epoch_interval_steps <= 0:
        message = (
            "half_epoch_interval_steps must be positive when validation is scheduled"
        )
        raise ValueError(message)
    if settings.train_reparameterization not in {
        _DETERMINISTIC_REPARAMETERIZATION,
        _STOCHASTIC_REPARAMETERIZATION,
    }:
        message = (
            "unsupported train_reparameterization: "
            f"{settings.train_reparameterization!r}"
        )
        raise ValueError(message)
    if _is_full_run(settings):
        _validate_full_run_settings(settings, dry_run=dry_run)


def _validate_full_run_settings(
    settings: _RunnerSettings,
    *,
    dry_run: bool,
) -> None:
    expected = {
        "requested_epochs": _FULL_EPOCHS,
        "optimizer_updates_per_epoch": _FULL_UPDATES_PER_EPOCH,
        "target_train_steps": _FULL_TARGET_UPDATES,
        "half_epoch_interval_steps": _FULL_HALF_EPOCH_INTERVAL_STEPS,
        "validation_batches_per_view": _FULL_VALIDATION_BATCHES_PER_VIEW,
    }
    observed = {
        "requested_epochs": settings.requested_epochs,
        "optimizer_updates_per_epoch": settings.optimizer_updates_per_epoch,
        "target_train_steps": settings.target_train_steps,
        "half_epoch_interval_steps": settings.half_epoch_interval_steps,
        "validation_batches_per_view": settings.validation_batches_per_view,
    }
    for field, expected_value in expected.items():
        actual = observed[field]
        if actual != expected_value:
            message = f"full-run {field} must be {expected_value!r}, got {actual!r}"
            raise ValueError(message)
    if settings.validation_views != _FULL_VALIDATION_VIEWS:
        message = "full-run validation_views must be clean and deterministic_denoising"
        raise ValueError(message)
    if settings.train_reparameterization != _STOCHASTIC_REPARAMETERIZATION:
        message = "full-run training must use stochastic_seeded reparameterization"
        raise ValueError(message)
    if settings.deterministic_eps_allowed_for != _FULL_DETERMINISTIC_EPS_ALLOWED_FOR:
        message = "full-run deterministic_eps_allowed_for has unexpected lanes"
        raise ValueError(message)
    if settings.checkpoint_retention != _FULL_CHECKPOINT_RETENTION:
        message = (
            "full-run checkpoint_retention must be best_final_latest_four_interval"
        )
        raise ValueError(message)
    if not settings.resume_supported:
        message = "full-run config must declare resume_supported=true"
        raise ValueError(message)
    if not dry_run and settings.save_every_steps != _FULL_HALF_EPOCH_INTERVAL_STEPS:
        message = "full-run save_every_steps must equal the half-epoch interval"
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


def _optional_str(payload: JsonObject, key: str) -> str | None:
    value = payload.get(key)
    if value is None or isinstance(value, str):
        return value
    message = f"Expected optional string config field {key}"
    raise TypeError(message)


def _optional_str_tuple(payload: JsonObject, key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if value is None:
        return ()
    if not isinstance(value, list):
        message = f"Expected list config field {key}"
        raise TypeError(message)
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            message = f"Expected string entries in config field {key}"
            raise TypeError(message)
        result.append(item)
    return tuple(result)


def _optional_bool(payload: JsonObject, key: str) -> bool | None:
    value = payload.get(key)
    if value is None or isinstance(value, bool):
        return value
    message = f"Expected boolean config field {key}"
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
