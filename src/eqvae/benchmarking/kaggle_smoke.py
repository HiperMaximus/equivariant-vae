# Copyright 2026 HiperMaximus
"""Capped real-data smoke path for Kaggle debug kernels."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from eqvae.benchmarking.io import JsonObject, write_json
from eqvae.config import resolve_json_config
from eqvae.corruption.stain import (
    CORRUPTION_VERSION,
    StainCorruptionMetadata,
    clean_validation_passthrough,
    corrupt_normalized_batch,
    profile_from_config,
)
from eqvae.data.dataloaders import normalize_uint8_batch
from eqvae.data.patch_shards import load_patch_records
from eqvae.data.roots import PatchDataPaths, PatchSplit, resolve_patch_data_paths
from eqvae.data.training_batches import (
    PatchTrainingBatch,
    PatchTrainingDataset,
    PatchTrainingDatasetSpec,
    collate_patch_training_samples,
)
from eqvae.losses.vae import beta_for_step
from eqvae.models.non_equivariant_vae import (
    DEFAULT_GROUPNORM_GROUPS,
    LATENT_CHANNELS,
)
from eqvae.models.registry import MODEL_KIND_NON_EQ_TRANSLATABLE, build_model
from eqvae.training.optim import SpecAdamWConfig, create_adamw_optimizer
from eqvae.training.step import TrainStepRequest, run_train_step

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from eqvae.models.non_equivariant_vae import NonEquivariantVAE

SMOKE_SCHEMA_VERSION = "spec0001.kaggle_smoke.v1"
DEFAULT_SMOKE_KIND = "real_data_kaggle_debug_smoke"
DEFAULT_SMOKE_SOURCE = "kaggle_script_kernel_capped_smoke"
SETUP_SMOKE_KIND = "synthetic_kaggle_setup_smoke"
SETUP_SMOKE_SOURCE = "kaggle_script_kernel_synthetic_setup_smoke"
DEFAULT_CORRUPTION_VIEW = "train_corrupted_kaggle_smoke"
SETUP_CORRUPTION_VIEW = "train_corrupted_kaggle_setup_smoke"
DEFAULT_MAX_TRAIN_STEPS = 3
DEFAULT_MAX_VALIDATION_BATCHES = 1
DEFAULT_BATCH_SIZE = 1
DEFAULT_NUM_WORKERS = 0
EXPECTED_REAL_DATASET_SLUG = "maximusshtefan/patches-pre-shuffled-ubc-ocean"
SETUP_DATA_KIND = "synthetic-ubc-setup-smoke"
ADAM_BETA_COUNT = 2
RGB_CHANNEL_COUNT = 3


@dataclass(frozen=True)
class KaggleSmokeRequest:
    """Inputs for a capped Kaggle/local real-data smoke run."""

    config_path: Path
    output_dir: Path
    data_root: str | None = None


@dataclass(frozen=True)
class KaggleSmokeSettings:
    """Resolved capped smoke settings."""

    benchmark_kind: str
    benchmark_source: str
    full_run_eligible: bool
    batch_size: int
    max_train_steps: int
    max_validation_batches: int
    num_workers: int
    validate_crc: bool
    corruption_view: str
    data_kind: str
    dataset_slug: str
    data_root: str
    image_size: int
    channels: int
    global_seed: int
    data_seed: int
    corruption_seed: int
    latent_seed: int
    corruption_config: JsonObject
    ssim_weight: float
    beta_target: float
    beta_warmup_fraction: float
    optimizer_config: SpecAdamWConfig
    norm_groups: int


@dataclass(frozen=True)
class _TrainSmokeSummary:
    steps_completed: int
    losses: tuple[float, ...]
    applied_counts: tuple[int, ...]
    first_sample_key_hashes: tuple[str, ...]
    nonfinite_counts: tuple[int, ...]
    input_target_delta_maxes: tuple[float, ...]
    input_target_delta_means: tuple[float, ...]
    nonzero_grad_counts: tuple[int, ...]
    nonzero_update_counts: tuple[int, ...]
    update_norms: tuple[float, ...]
    corruption_metadata: tuple[JsonObject, ...]


@dataclass(frozen=True)
class _ValidationSmokeSummary:
    batches_completed: int
    clean_validation_rng_advanced: bool
    first_sample_key_hashes: tuple[str, ...]
    finite_outputs: tuple[bool, ...]


def write_kaggle_smoke(request: KaggleSmokeRequest) -> Path:
    """Run the capped smoke path and write `benchmark/kaggle_smoke.json`.

    Returns:
        Path to the smoke artifact.

    """
    resolved = resolve_json_config(request.config_path)
    effective = resolved.effective_config
    settings = _settings(effective, data_root_override=request.data_root)
    _validate_smoke_settings(settings)
    paths = resolve_patch_data_paths(settings.data_root)
    _seed_runtime(settings)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _validate_runtime(settings=settings, device=device)
    model = build_model(
        MODEL_KIND_NON_EQ_TRANSLATABLE,
        model_config={"norm_groups": settings.norm_groups},
    ).to(device)
    optimizer, optimizer_summary = create_adamw_optimizer(
        model,
        config=settings.optimizer_config,
    )
    train_summary = _run_train_smoke(
        paths=paths,
        settings=settings,
        model=model,
        optimizer=optimizer,
        device=device,
    )
    validation_summary = _run_validation_smoke(
        paths=paths,
        settings=settings,
        model=model,
        device=device,
    )
    status = _status(
        settings=settings,
        train_summary=train_summary,
        validation_summary=validation_summary,
    )
    payload = _payload(
        request=request,
        settings=settings,
        paths=paths,
        invoked_config_hash=resolved.invoked_config_hash,
        effective_config_hash=resolved.effective_config_hash,
        train_summary=train_summary,
        validation_summary=validation_summary,
        payload_manifest=_payload_manifest_for_config(request.config_path),
        device=device,
        optimizer_group_count=optimizer_summary.parameter_group_count,
        status=status,
    )
    output_path = request.output_dir / "benchmark" / _artifact_filename(settings)
    write_json(output_path, payload)
    return output_path


def _run_train_smoke(  # noqa: PLR0914
    *,
    paths: PatchDataPaths,
    settings: KaggleSmokeSettings,
    model: NonEquivariantVAE,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> _TrainSmokeSummary:
    train_loader = _loader(paths=paths, settings=settings, split="train")
    iterator = iter(train_loader)
    profile = profile_from_config(settings.corruption_config)
    losses: list[float] = []
    applied_counts: list[int] = []
    first_hashes: list[str] = []
    nonfinite_counts: list[int] = []
    delta_maxes: list[float] = []
    delta_means: list[float] = []
    grad_counts: list[int] = []
    update_counts: list[int] = []
    update_norms: list[float] = []
    corruption_metadata: list[JsonObject] = []
    for step_index in range(settings.max_train_steps):
        batch = _next_batch(iterator)
        _validate_semantic_keys(batch)
        clean = normalize_uint8_batch(batch.images_uint8).to(device=device)
        corruption = corrupt_normalized_batch(
            clean,
            profile=profile,
            corruption_seed=settings.corruption_seed,
            split=batch.split,
            semantic_sample_keys=batch.semantic_sample_keys,
            corruption_step=step_index,
            corruption_view=settings.corruption_view,
        )
        eps = _zero_eps(clean)
        beta = beta_for_step(
            optimizer_step_index=step_index,
            max_optimizer_steps=settings.max_train_steps,
            target_beta=settings.beta_target,
            warmup_fraction=settings.beta_warmup_fraction,
        )
        result = run_train_step(
            TrainStepRequest(
                model=model,
                optimizer=optimizer,
                clean_batch=clean,
                eps=eps,
                beta=beta,
                ssim_weight=settings.ssim_weight,
                optimizer_step_index=step_index,
                gradient_clip_global_norm=(
                    settings.optimizer_config.gradient_clip_global_norm
                ),
                input_batch=corruption.corrupted,
            ),
        )
        input_target_delta = (corruption.corrupted - clean).detach().abs()
        losses.append(float(result.losses.loss.detach().cpu().item()))
        applied_counts.append(sum(1 for item in corruption.metadata if item.applied))
        first_hashes.append(_hash_text(batch.semantic_sample_keys[0]))
        nonfinite_counts.append(result.nonfinite_count)
        delta_maxes.append(float(input_target_delta.max().cpu().item()))
        delta_means.append(float(input_target_delta.mean().cpu().item()))
        grad_counts.append(result.nonzero_grad_parameter_tensor_count)
        update_counts.append(result.nonzero_update_parameter_tensor_count)
        update_norms.append(result.param_update_norm)
        corruption_metadata.extend(
            _corruption_metadata_payload(item) for item in corruption.metadata
        )
    return _TrainSmokeSummary(
        steps_completed=len(losses),
        losses=tuple(losses),
        applied_counts=tuple(applied_counts),
        first_sample_key_hashes=tuple(first_hashes),
        nonfinite_counts=tuple(nonfinite_counts),
        input_target_delta_maxes=tuple(delta_maxes),
        input_target_delta_means=tuple(delta_means),
        nonzero_grad_counts=tuple(grad_counts),
        nonzero_update_counts=tuple(update_counts),
        update_norms=tuple(update_norms),
        corruption_metadata=tuple(corruption_metadata),
    )


def _run_validation_smoke(
    *,
    paths: PatchDataPaths,
    settings: KaggleSmokeSettings,
    model: NonEquivariantVAE,
    device: torch.device,
) -> _ValidationSmokeSummary:
    validation_loader = _loader(paths=paths, settings=settings, split="validation")
    iterator = iter(validation_loader)
    first_hashes: list[str] = []
    finite_outputs: list[bool] = []
    rng_advanced = False
    model.eval()
    with torch.no_grad():
        for _batch_index in range(settings.max_validation_batches):
            batch = _next_batch(iterator)
            _validate_semantic_keys(batch)
            clean = normalize_uint8_batch(batch.images_uint8).to(device=device)
            state = torch.get_rng_state()
            model_input = clean_validation_passthrough(clean)
            rng_advanced = rng_advanced or not torch.equal(state, torch.get_rng_state())
            output = model.forward(model_input, eps=_zero_eps(clean))
            finite_outputs.append(
                bool(torch.isfinite(output.reconstruction).all().item()),
            )
            first_hashes.append(_hash_text(batch.semantic_sample_keys[0]))
    model.train()
    return _ValidationSmokeSummary(
        batches_completed=len(first_hashes),
        clean_validation_rng_advanced=rng_advanced,
        first_sample_key_hashes=tuple(first_hashes),
        finite_outputs=tuple(finite_outputs),
    )


def _loader(
    *,
    paths: PatchDataPaths,
    settings: KaggleSmokeSettings,
    split: PatchSplit,
) -> DataLoader[PatchTrainingBatch]:
    split_paths = paths.for_split(split)
    dataset = PatchTrainingDataset(
        PatchTrainingDatasetSpec(
            bin_path=split_paths.bin_path,
            csv_path=split_paths.csv_path,
            split=split_paths.split,
            image_size=settings.image_size,
            channels=settings.channels,
            validate_crc=settings.validate_crc,
        ),
    )
    return cast(
        "DataLoader[PatchTrainingBatch]",
        DataLoader(
            dataset,
            batch_size=settings.batch_size,
            shuffle=False,
            num_workers=settings.num_workers,
            collate_fn=collate_patch_training_samples,
        ),
    )


def _next_batch(iterator: Iterator[PatchTrainingBatch]) -> PatchTrainingBatch:
    try:
        return next(iterator)
    except StopIteration as error:
        message = "Kaggle smoke dataset ended before configured cap"
        raise RuntimeError(message) from error


def _zero_eps(clean: Tensor) -> Tensor:
    return torch.zeros(
        (
            clean.shape[0],
            LATENT_CHANNELS,
            clean.shape[2] // 8,
            clean.shape[3] // 8,
        ),
        dtype=torch.float32,
        device=clean.device,
    )


def _status(
    *,
    settings: KaggleSmokeSettings,
    train_summary: _TrainSmokeSummary,
    validation_summary: _ValidationSmokeSummary,
) -> str:
    passed = (
        train_summary.steps_completed == settings.max_train_steps
        and validation_summary.batches_completed == settings.max_validation_batches
        and not validation_summary.clean_validation_rng_advanced
        and all(validation_summary.finite_outputs)
        and sum(train_summary.applied_counts) > 0
        and any(delta > 0.0 for delta in train_summary.input_target_delta_maxes)
        and not any(count != 0 for count in train_summary.nonfinite_counts)
        and all(count > 0 for count in train_summary.nonzero_update_counts)
        and all(torch.isfinite(torch.tensor(train_summary.losses)))
    )
    return "smoke_pass" if passed else "fail"


def _payload(  # noqa: PLR0913
    *,
    request: KaggleSmokeRequest,
    settings: KaggleSmokeSettings,
    paths: PatchDataPaths,
    invoked_config_hash: str,
    effective_config_hash: str,
    train_summary: _TrainSmokeSummary,
    validation_summary: _ValidationSmokeSummary,
    payload_manifest: JsonObject | None,
    device: torch.device,
    optimizer_group_count: int,
    status: str,
) -> JsonObject:
    return cast(
        "JsonObject",
        {
            "schema_version": SMOKE_SCHEMA_VERSION,
            "status": status,
            "status_scope": _status_scope(settings),
            "benchmark_kind": settings.benchmark_kind,
            "benchmark_source": settings.benchmark_source,
            "full_run_eligible": False,
            "config": {
                "path": str(request.config_path),
                "invoked_config_hash": invoked_config_hash,
                "effective_config_hash": effective_config_hash,
            },
            "payload_manifest": payload_manifest,
            "data": {
                "kind": settings.data_kind,
                "dataset_slug": settings.dataset_slug,
                "origin": _data_origin(paths),
                "data_root": str(paths.root),
                "train_bin": str(paths.train.bin_path),
                "train_csv": str(paths.train.csv_path),
                "validation_bin": str(paths.validation.bin_path),
                "validation_csv": str(paths.validation.csv_path),
                "validate_crc": settings.validate_crc,
                "data_integrity_status": (
                    "crc_checked" if settings.validate_crc else "not_checked"
                ),
                "batch_schema": "PatchTrainingBatch.metadata_v1",
                "train_record_count": len(load_patch_records(paths.train.csv_path)),
                "validation_record_count": len(
                    load_patch_records(paths.validation.csv_path),
                ),
                "train_bin_bytes": paths.train.bin_path.stat().st_size,
                "validation_bin_bytes": paths.validation.bin_path.stat().st_size,
            },
            "runtime": {
                "device": str(device),
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count(),
                "gpu_names": _gpu_names(),
                "requires_cuda_t4": _requires_kaggle_t4(settings),
            },
            "seeds": {
                "global_seed": settings.global_seed,
                "data_seed": settings.data_seed,
                "corruption_seed": settings.corruption_seed,
                "latent_seed": settings.latent_seed,
                "latent_eps_policy": "zero_eps_deterministic_smoke",
            },
            "corruption": {
                "version": CORRUPTION_VERSION,
                "strategy": "branchless_all",
                "view": settings.corruption_view,
                "profile": settings.corruption_config,
                "clean_validation_consumes_rng": False,
            },
            "limits": {
                "batch_size": settings.batch_size,
                "max_train_steps": settings.max_train_steps,
                "max_validation_batches": settings.max_validation_batches,
                "num_workers": settings.num_workers,
            },
            "train": {
                "steps_completed": train_summary.steps_completed,
                "losses": list(train_summary.losses),
                "applied_counts": list(train_summary.applied_counts),
                "total_applied_count": sum(train_summary.applied_counts),
                "input_target_delta_maxes": list(
                    train_summary.input_target_delta_maxes,
                ),
                "input_target_delta_means": list(
                    train_summary.input_target_delta_means,
                ),
                "first_sample_key_hashes": list(
                    train_summary.first_sample_key_hashes,
                ),
                "nonfinite_counts": list(train_summary.nonfinite_counts),
                "nonzero_grad_counts": list(train_summary.nonzero_grad_counts),
                "nonzero_update_counts": list(train_summary.nonzero_update_counts),
                "update_norms": list(train_summary.update_norms),
                "corruption_metadata": list(train_summary.corruption_metadata),
                "optimizer_group_count": optimizer_group_count,
            },
            "validation": {
                "batches_completed": validation_summary.batches_completed,
                "clean_validation_rng_advanced": (
                    validation_summary.clean_validation_rng_advanced
                ),
                "first_sample_key_hashes": list(
                    validation_summary.first_sample_key_hashes,
                ),
                "finite_outputs": list(validation_summary.finite_outputs),
            },
        },
    )


def _settings(
    effective_config: JsonObject,
    *,
    data_root_override: str | None,
) -> KaggleSmokeSettings:
    data_config = _required_object(effective_config, "data")
    seeds = _required_object(effective_config, "seeds")
    corruption_config = _required_object(effective_config, "corruption")
    objective = _required_object(effective_config, "objective")
    beta = _required_object(objective, "beta")
    smoke = _optional_object(effective_config, "kaggle_smoke")
    return KaggleSmokeSettings(
        benchmark_kind=_optional_str(
            smoke,
            "benchmark_kind",
            default=DEFAULT_SMOKE_KIND,
        ),
        benchmark_source=_optional_str(
            smoke,
            "benchmark_source",
            default=DEFAULT_SMOKE_SOURCE,
        ),
        full_run_eligible=_optional_bool(
            smoke,
            "full_run_eligible",
            default=False,
        ),
        batch_size=_optional_int(smoke, "batch_size", default=DEFAULT_BATCH_SIZE),
        max_train_steps=_optional_int(
            smoke,
            "max_train_steps",
            default=DEFAULT_MAX_TRAIN_STEPS,
        ),
        max_validation_batches=_optional_int(
            smoke,
            "max_validation_batches",
            default=DEFAULT_MAX_VALIDATION_BATCHES,
        ),
        num_workers=_optional_int(smoke, "num_workers", default=DEFAULT_NUM_WORKERS),
        validate_crc=_optional_bool(smoke, "validate_crc", default=False),
        corruption_view=_optional_str(
            smoke,
            "corruption_view",
            default=DEFAULT_CORRUPTION_VIEW,
        ),
        data_kind=_optional_str(data_config, "kind", default="unknown"),
        dataset_slug=_optional_str(data_config, "dataset_slug", default=""),
        data_root=(
            data_root_override
            if data_root_override is not None
            else _optional_str(data_config, "data_root", default="auto")
        ),
        image_size=_optional_int(data_config, "image_size", default=256),
        channels=_optional_int(data_config, "channels", default=3),
        global_seed=_required_int(seeds, "global_seed"),
        data_seed=_required_int(seeds, "data_seed"),
        corruption_seed=_required_int(seeds, "corruption_seed"),
        latent_seed=_required_int(seeds, "latent_seed"),
        corruption_config=corruption_config,
        ssim_weight=_required_float(objective, "ssim_weight"),
        beta_target=_required_float(beta, "target"),
        beta_warmup_fraction=_required_float(beta, "step_limited_warmup_fraction"),
        optimizer_config=_optimizer_config(effective_config),
        norm_groups=_norm_groups(effective_config),
    )


def _validate_smoke_settings(settings: KaggleSmokeSettings) -> None:
    if settings.full_run_eligible:
        message = "Kaggle smoke config must keep full_run_eligible = false"
        raise ValueError(message)
    if settings.batch_size != DEFAULT_BATCH_SIZE:
        message = f"Kaggle smoke batch_size must be 1, got {settings.batch_size}"
        raise ValueError(message)
    if not 1 <= settings.max_train_steps <= DEFAULT_MAX_TRAIN_STEPS:
        message = (
            "Kaggle smoke max_train_steps must be between 1 and "
            f"{DEFAULT_MAX_TRAIN_STEPS}, got {settings.max_train_steps}"
        )
        raise ValueError(message)
    if settings.max_validation_batches != DEFAULT_MAX_VALIDATION_BATCHES:
        message = (
            "Kaggle smoke max_validation_batches must be exactly "
            f"{DEFAULT_MAX_VALIDATION_BATCHES}, got {settings.max_validation_batches}"
        )
        raise ValueError(message)
    if settings.num_workers != DEFAULT_NUM_WORKERS:
        message = f"Kaggle smoke num_workers must be 0, got {settings.num_workers}"
        raise ValueError(message)
    if _is_setup_smoke(settings):
        _validate_setup_smoke_settings(settings)
        return
    if (
        settings.data_kind == SETUP_DATA_KIND
        or settings.benchmark_kind == SETUP_SMOKE_KIND
        or settings.benchmark_source == SETUP_SMOKE_SOURCE
    ):
        message = (
            "Synthetic Kaggle setup smoke must use the locked setup kind/source "
            "and data contract together"
        )
        raise ValueError(message)
    if (
        settings.data_kind == "ubc-pre-shuffled"
        or settings.dataset_slug == EXPECTED_REAL_DATASET_SLUG
    ) and not _requires_kaggle_t4(settings):
        message = (
            "Real-data Kaggle smoke must use the capped real-data benchmark "
            "kind/source so T4 and dataset checks cannot be bypassed"
        )
        raise ValueError(message)
    if _requires_kaggle_t4(settings) and (
        settings.data_kind != "ubc-pre-shuffled"
        or settings.dataset_slug != EXPECTED_REAL_DATASET_SLUG
    ):
        message = (
            "Real-data Kaggle smoke must record data.kind='ubc-pre-shuffled' "
            f"and dataset_slug={EXPECTED_REAL_DATASET_SLUG!r}"
        )
        raise ValueError(message)


def _validate_setup_smoke_settings(settings: KaggleSmokeSettings) -> None:
    if settings.data_kind != SETUP_DATA_KIND:
        message = (
            f"Setup smoke data.kind must be {SETUP_DATA_KIND!r}, "
            f"got {settings.data_kind!r}"
        )
        raise ValueError(message)
    if settings.dataset_slug:
        message = "Setup smoke must not declare or attach a Kaggle dataset slug"
        raise ValueError(message)
    if settings.data_root == "auto":
        message = "Setup smoke must pass an explicit synthetic data root"
        raise ValueError(message)
    if settings.data_root.startswith("/kaggle/input/"):
        message = "Setup smoke data root must not resolve under /kaggle/input"
        raise ValueError(message)
    if settings.channels != RGB_CHANNEL_COUNT:
        message = (
            f"Setup smoke requires RGB channels={RGB_CHANNEL_COUNT}, "
            f"got {settings.channels}"
        )
        raise ValueError(message)
    if settings.corruption_view != SETUP_CORRUPTION_VIEW:
        message = (
            f"Setup smoke corruption_view must be {SETUP_CORRUPTION_VIEW!r}, "
            f"got {settings.corruption_view!r}"
        )
        raise ValueError(message)
    if settings.corruption_config.get("profile_name") != "conservative_default":
        message = "Setup smoke must use the locked conservative_default profile"
        raise ValueError(message)


def _seed_runtime(settings: KaggleSmokeSettings) -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(settings.global_seed)
    torch.set_rng_state(generator.get_state())
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(settings.global_seed)


def _validate_runtime(*, settings: KaggleSmokeSettings, device: torch.device) -> None:
    if not _requires_kaggle_t4(settings):
        return
    if device.type != "cuda":
        message = "wrong_accelerator: real-data Kaggle smoke requires CUDA T4 runtime"
        raise RuntimeError(message)
    gpu_names = _gpu_names()
    if not gpu_names or not all("T4" in name for name in gpu_names):
        message = f"wrong_accelerator: expected visible T4 GPUs, got {gpu_names}"
        raise RuntimeError(message)


def _requires_kaggle_t4(settings: KaggleSmokeSettings) -> bool:
    return (
        settings.benchmark_kind == DEFAULT_SMOKE_KIND
        and settings.benchmark_source == DEFAULT_SMOKE_SOURCE
    )


def _is_setup_smoke(settings: KaggleSmokeSettings) -> bool:
    return (
        settings.benchmark_kind == SETUP_SMOKE_KIND
        and settings.benchmark_source == SETUP_SMOKE_SOURCE
    )


def _artifact_filename(settings: KaggleSmokeSettings) -> str:
    if _is_setup_smoke(settings):
        return "kaggle_setup_smoke.json"
    return "kaggle_smoke.json"


def _status_scope(settings: KaggleSmokeSettings) -> str:
    if _is_setup_smoke(settings):
        return "non_promotable_setup_smoke"
    return "non_promotable_debug"


def _optimizer_config(config: JsonObject) -> SpecAdamWConfig:
    optimizer = _required_object(config, "optimizer")
    betas = _required_list(optimizer, "betas")
    if len(betas) != ADAM_BETA_COUNT:
        message = "optimizer.betas must contain exactly two values"
        raise ValueError(message)
    return SpecAdamWConfig(
        learning_rate=_required_float(optimizer, "learning_rate"),
        beta1=_float_value(betas[0], key="optimizer.betas[0]"),
        beta2=_float_value(betas[1], key="optimizer.betas[1]"),
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


def _norm_groups(config: JsonObject) -> int:
    model = _required_object(config, "model")
    normalization = _required_object(model, "normalization")
    return _optional_int(
        normalization,
        "num_groups",
        default=DEFAULT_GROUPNORM_GROUPS,
    )


def _gpu_names() -> list[str]:
    return [
        torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
    ]


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_semantic_keys(batch: PatchTrainingBatch) -> None:
    expected_prefix = f"{batch.split}:"
    if any(not key.startswith(expected_prefix) for key in batch.semantic_sample_keys):
        message = "Patch-training batch semantic keys must start with the batch split"
        raise ValueError(message)
    if len(set(batch.semantic_sample_keys)) != len(batch.semantic_sample_keys):
        message = "Patch-training batch semantic keys must be unique"
        raise ValueError(message)


def _corruption_metadata_payload(metadata: StainCorruptionMetadata) -> JsonObject:
    payload = cast("JsonObject", metadata.as_json())
    semantic_key = payload.pop("semantic_sample_key")
    if not isinstance(semantic_key, str):
        message = "Corruption metadata semantic_sample_key must be a string"
        raise TypeError(message)
    payload["semantic_sample_key_hash"] = _hash_text(semantic_key)
    return payload


def _data_origin(paths: PatchDataPaths) -> str:
    root = str(paths.root)
    if root.startswith("/kaggle/input/"):
        return "kaggle_input_mount"
    if root.startswith(("/tmp/", "/kaggle/working/")):  # noqa: S108
        return "synthetic_or_ephemeral_path"
    return "local_or_explicit_path"


def _payload_manifest_for_config(config_path: Path) -> JsonObject | None:
    manifest_path = config_path.parents[2] / "payload_manifest.json"
    if not manifest_path.exists():
        return None
    return cast(
        "JsonObject",
        json.loads(manifest_path.read_text(encoding="utf-8")),
    )


def _required_object(payload: JsonObject, key: str) -> JsonObject:
    value = payload.get(key)
    if isinstance(value, dict):
        return cast("JsonObject", value)
    message = f"Expected object field: {key}"
    raise TypeError(message)


def _optional_object(payload: JsonObject, key: str) -> JsonObject:
    value = payload.get(key)
    if value is None:
        return {}
    if isinstance(value, dict):
        return cast("JsonObject", value)
    message = f"Expected optional object field: {key}"
    raise TypeError(message)


def _required_int(payload: JsonObject, key: str) -> int:
    value = payload.get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    message = f"Expected integer field: {key}"
    raise TypeError(message)


def _optional_int(payload: JsonObject, key: str, *, default: int) -> int:
    value = payload.get(key)
    if value is None:
        return default
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    message = f"Expected optional integer field: {key}"
    raise TypeError(message)


def _required_float(payload: JsonObject, key: str) -> float:
    value = payload.get(key)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    message = f"Expected numeric field: {key}"
    raise TypeError(message)


def _optional_bool(payload: JsonObject, key: str, *, default: bool) -> bool:
    value = payload.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    message = f"Expected optional boolean field: {key}"
    raise TypeError(message)


def _optional_str(payload: JsonObject, key: str, *, default: str) -> str:
    value = payload.get(key)
    if value is None:
        return default
    if isinstance(value, str):
        return value
    message = f"Expected optional string field: {key}"
    raise TypeError(message)


def _required_list(payload: JsonObject, key: str) -> list[object]:
    value = payload.get(key)
    if isinstance(value, list):
        return cast("list[object]", value)
    message = f"Expected list field: {key}"
    raise TypeError(message)


def _float_value(value: object, *, key: str) -> float:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    message = f"Expected numeric value for {key}"
    raise TypeError(message)


__all__ = [
    "KaggleSmokeRequest",
    "KaggleSmokeSettings",
    "write_kaggle_smoke",
]
