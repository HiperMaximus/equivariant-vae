# Copyright 2026 HiperMaximus
"""Capped real-data smoke path for Kaggle debug kernels."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from eqvae.benchmarking.io import JsonObject, write_json
from eqvae.config import resolve_json_config
from eqvae.corruption.stain import (
    CORRUPTION_VERSION,
    clean_validation_passthrough,
    corrupt_normalized_batch,
    profile_from_config,
)
from eqvae.data.dataloaders import normalize_uint8_batch
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
    build_non_equivariant_vae,
)
from eqvae.training.optim import SpecAdamWConfig, create_adamw_optimizer
from eqvae.training.step import TrainStepRequest, run_train_step

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from eqvae.models.non_equivariant_vae import NonEquivariantVAE

SMOKE_SCHEMA_VERSION = "spec0001.kaggle_smoke.v1"
DEFAULT_SMOKE_KIND = "real_data_kaggle_debug_smoke"
DEFAULT_SMOKE_SOURCE = "kaggle_script_kernel_capped_smoke"
DEFAULT_CORRUPTION_VIEW = "train_corrupted_kaggle_smoke"
DEFAULT_MAX_TRAIN_STEPS = 3
DEFAULT_MAX_VALIDATION_BATCHES = 1
DEFAULT_BATCH_SIZE = 1
DEFAULT_NUM_WORKERS = 0
ADAM_BETA_COUNT = 2


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
    data_root: str
    image_size: int
    channels: int
    data_seed: int
    corruption_seed: int
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

    Raises:
        ValueError: If the smoke config is accidentally full-run eligible.

    """
    resolved = resolve_json_config(request.config_path)
    effective = resolved.effective_config
    settings = _settings(effective, data_root_override=request.data_root)
    if settings.full_run_eligible:
        message = "Kaggle smoke config must keep full_run_eligible = false"
        raise ValueError(message)
    paths = resolve_patch_data_paths(settings.data_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_non_equivariant_vae(norm_groups=settings.norm_groups).to(device)
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
        device=device,
        optimizer_group_count=optimizer_summary.parameter_group_count,
        status=status,
    )
    output_path = request.output_dir / "benchmark" / "kaggle_smoke.json"
    write_json(output_path, payload)
    return output_path


def _run_train_smoke(
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
    for step_index in range(settings.max_train_steps):
        batch = _next_batch(iterator)
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
        losses.append(float(result.losses.loss.detach().cpu().item()))
        applied_counts.append(sum(1 for item in corruption.metadata if item.applied))
        first_hashes.append(_hash_text(batch.semantic_sample_keys[0]))
        nonfinite_counts.append(result.nonfinite_count)
    return _TrainSmokeSummary(
        steps_completed=len(losses),
        losses=tuple(losses),
        applied_counts=tuple(applied_counts),
        first_sample_key_hashes=tuple(first_hashes),
        nonfinite_counts=tuple(nonfinite_counts),
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
        and not any(count != 0 for count in train_summary.nonfinite_counts)
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
    device: torch.device,
    optimizer_group_count: int,
    status: str,
) -> JsonObject:
    return cast(
        "JsonObject",
        {
            "schema_version": SMOKE_SCHEMA_VERSION,
            "status": status,
            "benchmark_kind": settings.benchmark_kind,
            "benchmark_source": settings.benchmark_source,
            "full_run_eligible": False,
            "config": {
                "path": str(request.config_path),
                "invoked_config_hash": invoked_config_hash,
                "effective_config_hash": effective_config_hash,
            },
            "data": {
                "data_root": str(paths.root),
                "train_bin": str(paths.train.bin_path),
                "train_csv": str(paths.train.csv_path),
                "validation_bin": str(paths.validation.bin_path),
                "validation_csv": str(paths.validation.csv_path),
                "validate_crc": settings.validate_crc,
                "batch_schema": "PatchTrainingBatch.metadata_v1",
            },
            "runtime": {
                "device": str(device),
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count(),
                "gpu_names": _gpu_names(),
            },
            "corruption": {
                "version": CORRUPTION_VERSION,
                "strategy": "branchless_all",
                "view": settings.corruption_view,
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
                "first_sample_key_hashes": list(
                    train_summary.first_sample_key_hashes,
                ),
                "nonfinite_counts": list(train_summary.nonfinite_counts),
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
        data_root=(
            data_root_override
            if data_root_override is not None
            else _optional_str(data_config, "data_root", default="auto")
        ),
        image_size=_optional_int(data_config, "image_size", default=256),
        channels=_optional_int(data_config, "channels", default=3),
        data_seed=_required_int(seeds, "data_seed"),
        corruption_seed=_required_int(seeds, "corruption_seed"),
        corruption_config=corruption_config,
        ssim_weight=_required_float(objective, "ssim_weight"),
        beta_target=_required_float(beta, "target"),
        beta_warmup_fraction=_required_float(beta, "step_limited_warmup_fraction"),
        optimizer_config=_optimizer_config(effective_config),
        norm_groups=_norm_groups(effective_config),
    )


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
