# Copyright 2026 HiperMaximus
"""Scikit-compatible HED stain corruption in repo-owned Torch code."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from collections.abc import Sequence

    from eqvae.config import JsonObject, JsonValue

CORRUPTION_VERSION = "spec0001.hed_corruptor.v1"
SCIKIT_IMAGE_ORACLE_VERSION = "0.26.0"
OD_EPSILON = 1.0e-6
RGB_CHANNELS = 3
NCHW_NDIM = 4
FLOAT_PAIR_LENGTH = 2

RGB_FROM_HED: tuple[tuple[float, float, float], ...] = (
    (0.65, 0.70, 0.29),
    (0.07, 0.99, 0.11),
    (0.27, 0.57, 0.78),
)
HED_FROM_RGB: tuple[tuple[float, float, float], ...] = (
    (1.8779827368521356, -1.0076786862855642, -0.5561158181996246),
    (-0.06590806222356334, 1.1347303724996625, -0.13552179862837116),
    (-0.6019073634392891, -0.4804141884970579, 1.5735880719641926),
)

CONSERVATIVE_DEFAULT_PROFILE = "conservative_default"
FSQ_LEGACY_WIDE_PROFILE = "fsq_legacy_wide"
NO_CORRUPTION_PROBE_PROFILE = "no_corruption_probe"
BRANCHLESS_ALL_STRATEGY = "branchless_all"
INDEXED_MASKED_STRATEGY = "indexed_masked"
CORRUPTION_STRATEGIES: tuple[str, ...] = (
    BRANCHLESS_ALL_STRATEGY,
    INDEXED_MASKED_STRATEGY,
)
SEMANTIC_SEED_FIELDS: tuple[str, ...] = (
    "corruption_seed",
    "split",
    "semantic_sample_key",
    "corruption_step",
    "corruption_view",
    "corruption_version",
)


@dataclass(frozen=True)
class StainCorruptionProfile:
    """Numeric corruption profile for one HED jitter policy."""

    name: str
    corrupt_prob: float
    he_alpha_range: tuple[float, float]
    he_beta_range: tuple[float, float]
    residual_alpha_range: tuple[float, float]
    residual_beta_range: tuple[float, float]
    noise_std_range: tuple[float, float]

    def as_json(self) -> JsonObject:
        """Return JSON-safe profile fields.

        Returns:
            Profile payload for benchmark artifacts and configs.

        """
        return {
            "name": self.name,
            "corrupt_prob": self.corrupt_prob,
            "he_alpha_range": list(self.he_alpha_range),
            "he_beta_range": list(self.he_beta_range),
            "residual_alpha_range": list(self.residual_alpha_range),
            "residual_beta_range": list(self.residual_beta_range),
            "noise_std_range": list(self.noise_std_range),
        }


@dataclass(frozen=True)
class StainCorruptionParameters:
    """Per-sample branchless-all corruption parameters."""

    applied_mask: Tensor
    alpha: Tensor
    beta: Tensor
    noise_std: Tensor
    noise: Tensor
    sample_seeds: tuple[int, ...]


@dataclass(frozen=True)
class StainCorruptionMetadata:
    """Per-sample corruption metadata for benchmark evidence."""

    applied: bool
    semantic_sample_key: str
    derived_seed: int
    profile_name: str
    alpha: tuple[float, float, float]
    beta: tuple[float, float, float]
    noise_std: float
    finite: bool
    pre_clamp_min: float
    pre_clamp_max: float
    final_min: float
    final_max: float
    lower_clamp_fraction: float
    upper_clamp_fraction: float

    def as_json(self) -> JsonObject:
        """Return JSON-safe metadata fields.

        Returns:
            Per-sample corruption metadata payload.

        """
        return {
            "applied": self.applied,
            "semantic_sample_key": self.semantic_sample_key,
            "derived_seed": self.derived_seed,
            "profile_name": self.profile_name,
            "alpha": list(self.alpha),
            "beta": list(self.beta),
            "noise_std": self.noise_std,
            "finite": self.finite,
            "pre_clamp_min": self.pre_clamp_min,
            "pre_clamp_max": self.pre_clamp_max,
            "final_min": self.final_min,
            "final_max": self.final_max,
            "lower_clamp_fraction": self.lower_clamp_fraction,
            "upper_clamp_fraction": self.upper_clamp_fraction,
        }


@dataclass(frozen=True)
class StainCorruptionResult:
    """Branchless-all corruption outputs plus metadata."""

    corrupted: Tensor
    stain_only: Tensor
    gaussian_only: Tensor
    combined: Tensor
    metadata: tuple[StainCorruptionMetadata, ...]


class StainCorruptor(nn.Module):
    """Torch module containing the compile-friendly HED/RGB tensor math."""

    def __init__(self) -> None:
        """Create fixed HED conversion buffers."""
        super().__init__()
        self.register_buffer(
            "rgb_from_hed",
            torch.tensor(RGB_FROM_HED, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "hed_from_rgb",
            torch.tensor(HED_FROM_RGB, dtype=torch.float32),
            persistent=False,
        )

    def rgb_to_hed(self, rgb: Tensor) -> Tensor:
        """Convert NCHW RGB `[0, 1]` tensors to HED coordinates.

        Returns:
            NCHW HED tensor.

        """
        matrix = cast("Tensor", self.hed_from_rgb)
        return rgb_to_hed(rgb, hed_from_rgb=matrix)

    def hed_to_rgb(self, hed: Tensor) -> Tensor:
        """Convert NCHW HED tensors to RGB `[0, 1]`.

        Returns:
            NCHW RGB tensor.

        """
        matrix = cast("Tensor", self.rgb_from_hed)
        return hed_to_rgb(hed, rgb_from_hed=matrix)

    def apply_with_parameters(
        self,
        images: Tensor,
        parameters: StainCorruptionParameters,
        *,
        semantic_sample_keys: Sequence[str],
        profile_name: str,
        strategy: str = BRANCHLESS_ALL_STRATEGY,
    ) -> StainCorruptionResult:
        """Apply pre-sampled stain/noise parameters.

        `branchless_all` computes stain/noise branches for the full batch and
        selects the public corrupted output at the end. `indexed_masked` uses the
        same sampled parameters but applies the expensive HED path only to rows
        whose Bernoulli mask is true.

        Returns:
            Corrupted batch, diagnostic branches, and per-sample metadata.

        Raises:
            ValueError: If batch dimensions and sample metadata disagree.

        """
        _validate_normalized_batch(images)
        _validate_parameters(images, parameters)
        if len(semantic_sample_keys) != images.shape[0]:
            message = "semantic_sample_keys length must equal batch size"
            raise ValueError(message)
        if strategy == BRANCHLESS_ALL_STRATEGY:
            return self._apply_branchless_all(
                images,
                parameters,
                semantic_sample_keys=semantic_sample_keys,
                profile_name=profile_name,
            )
        if strategy == INDEXED_MASKED_STRATEGY:
            return self._apply_indexed_masked(
                images,
                parameters,
                semantic_sample_keys=semantic_sample_keys,
                profile_name=profile_name,
            )
        message = f"Unknown corruption strategy: {strategy}"
        raise ValueError(message)

    def _apply_branchless_all(  # noqa: PLR0914
        self,
        images: Tensor,
        parameters: StainCorruptionParameters,
        *,
        semantic_sample_keys: Sequence[str],
        profile_name: str,
    ) -> StainCorruptionResult:
        """Apply the full-batch branchless corruption strategy.

        Returns:
            Corrupted batch, diagnostic branches, and per-sample metadata.

        """
        input_dtype = images.dtype
        work = images.detach().to(dtype=torch.float32)
        alpha = parameters.alpha.to(device=work.device, dtype=torch.float32)
        beta = parameters.beta.to(device=work.device, dtype=torch.float32)
        noise = parameters.noise.to(device=work.device, dtype=torch.float32)
        applied_mask = parameters.applied_mask.to(device=work.device)

        rgb = normalized_to_rgb01(work)
        hed = self.rgb_to_hed(rgb)
        jittered_rgb = self.hed_to_rgb((hed * alpha) + beta)
        stain_pre_clamp = rgb01_to_normalized(jittered_rgb)
        gaussian_pre_clamp = work + noise
        combined_pre_clamp = stain_pre_clamp + noise

        stain_only = stain_pre_clamp.clamp(-1.0, 1.0).to(dtype=input_dtype)
        gaussian_only = gaussian_pre_clamp.clamp(-1.0, 1.0).to(dtype=input_dtype)
        combined = combined_pre_clamp.clamp(-1.0, 1.0).to(dtype=input_dtype)
        selected_pre_clamp = torch.where(applied_mask, combined_pre_clamp, work)
        corrupted = selected_pre_clamp.clamp(-1.0, 1.0).to(dtype=input_dtype)
        metadata = _metadata_from_tensors(
            selected_pre_clamp=selected_pre_clamp,
            final=corrupted.to(dtype=torch.float32),
            parameters=parameters,
            semantic_sample_keys=semantic_sample_keys,
            profile_name=profile_name,
        )
        return StainCorruptionResult(
            corrupted=corrupted,
            stain_only=stain_only,
            gaussian_only=gaussian_only,
            combined=combined,
            metadata=metadata,
        )

    def _apply_indexed_masked(  # noqa: PLR0914
        self,
        images: Tensor,
        parameters: StainCorruptionParameters,
        *,
        semantic_sample_keys: Sequence[str],
        profile_name: str,
    ) -> StainCorruptionResult:
        """Apply the indexed masked corruption strategy.

        Returns:
            Corrupted batch, diagnostic branches, and per-sample metadata.

        """
        input_dtype = images.dtype
        work = images.detach().to(dtype=torch.float32)
        alpha = parameters.alpha.to(device=work.device, dtype=torch.float32)
        beta = parameters.beta.to(device=work.device, dtype=torch.float32)
        noise = parameters.noise.to(device=work.device, dtype=torch.float32)
        mask_flat = parameters.applied_mask.to(device=work.device).view(-1)

        stain_pre_clamp = work.clone()
        gaussian_pre_clamp = work.clone()
        combined_pre_clamp = work.clone()
        selected_pre_clamp = work.clone()
        if bool(mask_flat.any().item()):
            masked_work = work[mask_flat]
            masked_noise = noise[mask_flat]
            rgb = normalized_to_rgb01(masked_work)
            hed = self.rgb_to_hed(rgb)
            jittered_rgb = self.hed_to_rgb((hed * alpha[mask_flat]) + beta[mask_flat])
            masked_stain_pre_clamp = rgb01_to_normalized(jittered_rgb)
            masked_gaussian_pre_clamp = masked_work + masked_noise
            masked_combined_pre_clamp = masked_stain_pre_clamp + masked_noise

            stain_pre_clamp[mask_flat] = masked_stain_pre_clamp
            gaussian_pre_clamp[mask_flat] = masked_gaussian_pre_clamp
            combined_pre_clamp[mask_flat] = masked_combined_pre_clamp
            selected_pre_clamp[mask_flat] = masked_combined_pre_clamp

        stain_only = stain_pre_clamp.clamp(-1.0, 1.0).to(dtype=input_dtype)
        gaussian_only = gaussian_pre_clamp.clamp(-1.0, 1.0).to(dtype=input_dtype)
        combined = combined_pre_clamp.clamp(-1.0, 1.0).to(dtype=input_dtype)
        corrupted = selected_pre_clamp.clamp(-1.0, 1.0).to(dtype=input_dtype)
        metadata = _metadata_from_tensors(
            selected_pre_clamp=selected_pre_clamp,
            final=corrupted.to(dtype=torch.float32),
            parameters=parameters,
            semantic_sample_keys=semantic_sample_keys,
            profile_name=profile_name,
        )
        return StainCorruptionResult(
            corrupted=corrupted,
            stain_only=stain_only,
            gaussian_only=gaussian_only,
            combined=combined,
            metadata=metadata,
        )


def normalized_to_rgb01(images: Tensor) -> Tensor:
    """Project normalized RGB `[-1, 1]` tensors into clamped RGB `[0, 1]`.

    Returns:
        Tensor in RGB `[0, 1]`.

    """
    return ((images + 1.0) * 0.5).clamp(0.0, 1.0)


def rgb01_to_normalized(rgb: Tensor) -> Tensor:
    """Project RGB `[0, 1]` tensors into normalized RGB `[-1, 1]`.

    Returns:
        Tensor in normalized RGB `[-1, 1]`.

    """
    return (rgb * 2.0) - 1.0


def rgb_to_hed(rgb: Tensor, *, hed_from_rgb: Tensor | None = None) -> Tensor:
    """Convert NCHW RGB `[0, 1]` to HED with scikit-image semantics.

    Returns:
        NCHW HED tensor after scikit-compatible nonnegative stain clamp.

    """
    _validate_rgb_batch(rgb)
    matrix = _matrix_like(HED_FROM_RGB, rgb, hed_from_rgb)
    values = torch.log(rgb.clamp_min(OD_EPSILON)) / math.log(OD_EPSILON)
    hed = torch.einsum("bchw,cd->bdhw", values, matrix)
    return hed.clamp_min(0.0)


def hed_to_rgb(hed: Tensor, *, rgb_from_hed: Tensor | None = None) -> Tensor:
    """Convert NCHW HED to RGB `[0, 1]` with scikit-image semantics.

    Returns:
        NCHW RGB tensor clamped to `[0, 1]`.

    """
    _validate_hed_batch(hed)
    matrix = _matrix_like(RGB_FROM_HED, hed, rgb_from_hed)
    log_adjust = -math.log(OD_EPSILON)
    log_rgb = torch.einsum("bchw,cd->bdhw", -(hed * log_adjust), matrix)
    return torch.exp(log_rgb).clamp(0.0, 1.0)


def profile_from_name(name: str) -> StainCorruptionProfile:
    """Return a locked stain-corruption profile by name.

    Returns:
        Locked stain-corruption profile.

    Raises:
        ValueError: If the profile name is unknown.

    """
    if name == CONSERVATIVE_DEFAULT_PROFILE:
        return StainCorruptionProfile(
            name=CONSERVATIVE_DEFAULT_PROFILE,
            corrupt_prob=0.3,
            he_alpha_range=(0.80, 1.20),
            he_beta_range=(-0.05, 0.05),
            residual_alpha_range=(0.98, 1.02),
            residual_beta_range=(-0.01, 0.01),
            noise_std_range=(0.0, 0.05),
        )
    if name == FSQ_LEGACY_WIDE_PROFILE:
        return StainCorruptionProfile(
            name=FSQ_LEGACY_WIDE_PROFILE,
            corrupt_prob=0.3,
            he_alpha_range=(0.75, 1.25),
            he_beta_range=(-0.10, 0.10),
            residual_alpha_range=(0.98, 1.02),
            residual_beta_range=(-0.01, 0.01),
            noise_std_range=(0.0, 0.05),
        )
    if name == NO_CORRUPTION_PROBE_PROFILE:
        return StainCorruptionProfile(
            name=NO_CORRUPTION_PROBE_PROFILE,
            corrupt_prob=0.0,
            he_alpha_range=(0.80, 1.20),
            he_beta_range=(-0.05, 0.05),
            residual_alpha_range=(0.98, 1.02),
            residual_beta_range=(-0.01, 0.01),
            noise_std_range=(0.0, 0.05),
        )
    message = f"Unknown stain corruption profile: {name}"
    raise ValueError(message)


def profile_from_config(corruption_config: JsonObject) -> StainCorruptionProfile:
    """Parse and validate a config corruption block against locked profiles.

    Returns:
        Parsed profile, guaranteed to match the named locked profile.

    """
    profile = profile_from_name(_required_str(corruption_config, "profile_name"))
    parsed = StainCorruptionProfile(
        name=profile.name,
        corrupt_prob=_required_float(corruption_config, "corrupt_prob"),
        he_alpha_range=_required_float_pair(corruption_config, "he_alpha_range"),
        he_beta_range=_required_float_pair(corruption_config, "he_beta_range"),
        residual_alpha_range=_required_float_pair(
            corruption_config,
            "residual_alpha_range",
        ),
        residual_beta_range=_required_float_pair(
            corruption_config,
            "residual_beta_range",
        ),
        noise_std_range=_required_float_pair(corruption_config, "noise_std_range"),
    )
    _require_profile_matches_named(parsed, profile)
    return parsed


def semantic_sample_key(
    *,
    split: str,
    wsi_id: str,
    label: int,
    x: int,
    y: int,
) -> str:
    """Return the semantic corruption key that excludes file/rank order.

    Returns:
        Semantic sample key used for corruption RNG.

    """
    return f"{split}:{wsi_id}:{label}:{x}:{y}"


def derive_corruption_seed(  # noqa: PLR0913
    *,
    corruption_seed: int,
    split: str,
    semantic_sample_key: str,
    corruption_step: int,
    corruption_view: str,
    corruption_version: str = CORRUPTION_VERSION,
) -> int:
    """Derive a stable 63-bit per-sample corruption seed.

    Returns:
        Stable nonnegative 63-bit integer seed.

    """
    payload = "\n".join(
        (
            str(corruption_seed),
            split,
            semantic_sample_key,
            str(corruption_step),
            corruption_view,
            corruption_version,
        ),
    ).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=16).digest()
    return int.from_bytes(digest[:8], byteorder="little") & ((1 << 63) - 1)


def sample_corruption_parameters(  # noqa: PLR0913, PLR0914
    *,
    batch_shape: tuple[int, int, int, int],
    profile: StainCorruptionProfile,
    corruption_seed: int,
    split: str,
    semantic_sample_keys: Sequence[str],
    corruption_step: int,
    corruption_view: str,
    device: torch.device | str = "cpu",
) -> StainCorruptionParameters:
    """Sample deterministic per-sample corruption parameters on CPU first.

    Returns:
        Per-sample branchless-all corruption parameters.

    Raises:
        ValueError: If shape, metadata length, or profile fields are invalid.

    """
    batch_size, channels, height, width = batch_shape
    if channels != RGB_CHANNELS:
        message = f"Expected 3 RGB channels, got {channels}"
        raise ValueError(message)
    if len(semantic_sample_keys) != batch_size:
        message = "semantic_sample_keys length must equal batch size"
        raise ValueError(message)
    _validate_profile(profile)

    applied: list[bool] = []
    alphas: list[tuple[float, float, float]] = []
    betas: list[tuple[float, float, float]] = []
    noise_stds: list[float] = []
    noises: list[Tensor] = []
    sample_seeds: list[int] = []
    for key in semantic_sample_keys:
        sample_seed = derive_corruption_seed(
            corruption_seed=corruption_seed,
            split=split,
            semantic_sample_key=key,
            corruption_step=corruption_step,
            corruption_view=corruption_view,
        )
        sample_seeds.append(sample_seed)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(sample_seed)
        applied.append(_uniform_scalar(generator) < profile.corrupt_prob)
        he_alpha = _uniform_tuple(generator, profile.he_alpha_range, count=2)
        residual_alpha = _uniform_tuple(
            generator,
            profile.residual_alpha_range,
            count=1,
        )
        he_beta = _uniform_tuple(generator, profile.he_beta_range, count=2)
        residual_beta = _uniform_tuple(generator, profile.residual_beta_range, count=1)
        alpha = (he_alpha[0], he_alpha[1], residual_alpha[0])
        beta = (he_beta[0], he_beta[1], residual_beta[0])
        noise_std = _uniform_tuple(generator, profile.noise_std_range, count=1)[0]
        noise = (
            torch.randn(
                (channels, height, width),
                generator=generator,
                dtype=torch.float32,
            )
            * noise_std
        )
        alphas.append(alpha)
        betas.append(beta)
        noise_stds.append(noise_std)
        noises.append(noise)

    target_device = torch.device(device)
    return StainCorruptionParameters(
        applied_mask=torch.tensor(
            applied,
            dtype=torch.bool,
            device=target_device,
        ).view(batch_size, 1, 1, 1),
        alpha=torch.tensor(
            alphas,
            dtype=torch.float32,
            device=target_device,
        ).view(batch_size, RGB_CHANNELS, 1, 1),
        beta=torch.tensor(
            betas,
            dtype=torch.float32,
            device=target_device,
        ).view(batch_size, RGB_CHANNELS, 1, 1),
        noise_std=torch.tensor(
            noise_stds,
            dtype=torch.float32,
            device=target_device,
        ).view(batch_size, 1, 1, 1),
        noise=torch.stack(noises, dim=0).to(device=target_device),
        sample_seeds=tuple(sample_seeds),
    )


def corrupt_normalized_batch(  # noqa: PLR0913
    images: Tensor,
    *,
    profile: StainCorruptionProfile,
    corruption_seed: int,
    split: str,
    semantic_sample_keys: Sequence[str],
    corruption_step: int,
    corruption_view: str,
    corruptor: StainCorruptor | None = None,
    strategy: str = BRANCHLESS_ALL_STRATEGY,
) -> StainCorruptionResult:
    """Sample and apply HED corruption to a normalized batch.

    Returns:
        Corrupted batch, diagnostic branches, and per-sample metadata.

    """
    _validate_normalized_batch(images)
    module = StainCorruptor() if corruptor is None else corruptor
    module = module.to(device=images.device)
    parameters = sample_corruption_parameters(
        batch_shape=(
            int(images.shape[0]),
            int(images.shape[1]),
            int(images.shape[2]),
            int(images.shape[3]),
        ),
        profile=profile,
        corruption_seed=corruption_seed,
        split=split,
        semantic_sample_keys=semantic_sample_keys,
        corruption_step=corruption_step,
        corruption_view=corruption_view,
        device=images.device,
    )
    with torch.no_grad():
        return module.apply_with_parameters(
            images,
            parameters,
            semantic_sample_keys=semantic_sample_keys,
            profile_name=profile.name,
            strategy=strategy,
        )


def clean_validation_passthrough(images: Tensor) -> Tensor:
    """Return clean validation/test inputs without consuming corruption RNG.

    Returns:
        The exact input tensor object.

    """
    return images


def _matrix_like(
    values: tuple[tuple[float, float, float], ...],
    reference: Tensor,
    override: Tensor | None,
) -> Tensor:
    if override is not None:
        return override.to(device=reference.device, dtype=reference.dtype)
    return torch.tensor(values, dtype=reference.dtype, device=reference.device)


def _uniform_scalar(generator: torch.Generator) -> float:
    return float(torch.rand((), generator=generator, dtype=torch.float32).item())


def _uniform_tuple(
    generator: torch.Generator,
    value_range: tuple[float, float],
    *,
    count: int,
) -> tuple[float, ...]:
    low, high = value_range
    values = torch.rand((count,), generator=generator, dtype=torch.float32)
    scaled = (values * (high - low)) + low
    return tuple(float(value.item()) for value in scaled)


def _metadata_from_tensors(
    *,
    selected_pre_clamp: Tensor,
    final: Tensor,
    parameters: StainCorruptionParameters,
    semantic_sample_keys: Sequence[str],
    profile_name: str,
) -> tuple[StainCorruptionMetadata, ...]:
    metadata: list[StainCorruptionMetadata] = []
    applied = parameters.applied_mask.detach().cpu().view(-1)
    alpha = parameters.alpha.detach().cpu().view(-1, RGB_CHANNELS)
    beta = parameters.beta.detach().cpu().view(-1, RGB_CHANNELS)
    noise_std = parameters.noise_std.detach().cpu().view(-1)
    pre_clamp = selected_pre_clamp.detach().cpu()
    final_cpu = final.detach().cpu()
    for index, key in enumerate(semantic_sample_keys):
        pre_sample = pre_clamp[index]
        final_sample = final_cpu[index]
        metadata.append(
            StainCorruptionMetadata(
                applied=bool(applied[index].item()),
                semantic_sample_key=key,
                derived_seed=parameters.sample_seeds[index],
                profile_name=profile_name,
                alpha=_float_triplet(alpha[index]),
                beta=_float_triplet(beta[index]),
                noise_std=float(noise_std[index].item()),
                finite=bool(torch.isfinite(final_sample).all().item()),
                pre_clamp_min=float(pre_sample.min().item()),
                pre_clamp_max=float(pre_sample.max().item()),
                final_min=float(final_sample.min().item()),
                final_max=float(final_sample.max().item()),
                lower_clamp_fraction=_fraction(pre_sample < -1.0),
                upper_clamp_fraction=_fraction(pre_sample > 1.0),
            ),
        )
    return tuple(metadata)


def _float_triplet(values: Tensor) -> tuple[float, float, float]:
    return (
        float(values[0].item()),
        float(values[1].item()),
        float(values[2].item()),
    )


def _fraction(mask: Tensor) -> float:
    return float(mask.to(dtype=torch.float32).mean().item())


def _validate_normalized_batch(images: Tensor) -> None:
    _validate_rgb_batch(images)
    if not images.is_floating_point():
        message = "images must be a floating-point normalized RGB tensor"
        raise TypeError(message)


def _validate_rgb_batch(rgb: Tensor) -> None:
    if rgb.ndim != NCHW_NDIM:
        message = f"Expected NCHW tensor with 4 dimensions, got {rgb.ndim}"
        raise ValueError(message)
    if int(rgb.shape[1]) != RGB_CHANNELS:
        message = f"Expected NCHW RGB tensor with 3 channels, got {rgb.shape[1]}"
        raise ValueError(message)


def _validate_hed_batch(hed: Tensor) -> None:
    if hed.ndim != NCHW_NDIM:
        message = f"Expected NCHW tensor with 4 dimensions, got {hed.ndim}"
        raise ValueError(message)
    if int(hed.shape[1]) != RGB_CHANNELS:
        message = f"Expected NCHW HED tensor with 3 channels, got {hed.shape[1]}"
        raise ValueError(message)
    if not hed.is_floating_point():
        message = "HED tensor must be floating point"
        raise TypeError(message)


def _validate_parameters(images: Tensor, parameters: StainCorruptionParameters) -> None:
    batch_size = int(images.shape[0])
    expected_parameter_shape = (batch_size, RGB_CHANNELS, 1, 1)
    if tuple(parameters.alpha.shape) != expected_parameter_shape:
        message = f"alpha must have shape {expected_parameter_shape}"
        raise ValueError(message)
    if tuple(parameters.beta.shape) != expected_parameter_shape:
        message = f"beta must have shape {expected_parameter_shape}"
        raise ValueError(message)
    if tuple(parameters.noise.shape) != tuple(images.shape):
        message = "noise must match image batch shape"
        raise ValueError(message)
    if tuple(parameters.applied_mask.shape) != (batch_size, 1, 1, 1):
        message = "applied_mask must have shape (N, 1, 1, 1)"
        raise ValueError(message)
    if tuple(parameters.noise_std.shape) != (batch_size, 1, 1, 1):
        message = "noise_std must have shape (N, 1, 1, 1)"
        raise ValueError(message)
    if len(parameters.sample_seeds) != batch_size:
        message = "sample_seeds length must equal batch size"
        raise ValueError(message)


def _validate_profile(profile: StainCorruptionProfile) -> None:
    if not 0.0 <= profile.corrupt_prob <= 1.0:
        message = "corrupt_prob must be in [0, 1]"
        raise ValueError(message)
    for name, value_range in (
        ("he_alpha_range", profile.he_alpha_range),
        ("he_beta_range", profile.he_beta_range),
        ("residual_alpha_range", profile.residual_alpha_range),
        ("residual_beta_range", profile.residual_beta_range),
        ("noise_std_range", profile.noise_std_range),
    ):
        if value_range[0] > value_range[1]:
            message = f"{name} lower bound must be <= upper bound"
            raise ValueError(message)
    if profile.noise_std_range[0] < 0.0:
        message = "noise_std_range lower bound must be nonnegative"
        raise ValueError(message)


def _require_profile_matches_named(
    parsed: StainCorruptionProfile,
    named: StainCorruptionProfile,
) -> None:
    if parsed != named:
        message = (
            f"corruption config profile fields do not match locked profile {named.name}"
        )
        raise ValueError(message)


def _required_str(payload: JsonObject, key: str) -> str:
    value = payload.get(key)
    if isinstance(value, str):
        return value
    message = f"Expected string corruption field: {key}"
    raise TypeError(message)


def _required_float(payload: JsonObject, key: str) -> float:
    value = payload.get(key)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    message = f"Expected numeric corruption field: {key}"
    raise TypeError(message)


def _required_float_pair(payload: JsonObject, key: str) -> tuple[float, float]:
    value = payload.get(key)
    if not isinstance(value, list):
        message = f"Expected two-number list corruption field: {key}"
        raise TypeError(message)
    typed_value = cast("list[JsonValue]", value)
    if len(typed_value) != FLOAT_PAIR_LENGTH:
        message = f"Expected exactly two values in corruption field: {key}"
        raise ValueError(message)
    first, second = typed_value
    if (
        isinstance(first, int | float)
        and not isinstance(first, bool)
        and isinstance(second, int | float)
        and not isinstance(second, bool)
    ):
        return (float(first), float(second))
    message = f"Expected numeric pair in corruption field: {key}"
    raise TypeError(message)


__all__ = [
    "BRANCHLESS_ALL_STRATEGY",
    "CONSERVATIVE_DEFAULT_PROFILE",
    "CORRUPTION_STRATEGIES",
    "CORRUPTION_VERSION",
    "FSQ_LEGACY_WIDE_PROFILE",
    "HED_FROM_RGB",
    "INDEXED_MASKED_STRATEGY",
    "OD_EPSILON",
    "RGB_FROM_HED",
    "SCIKIT_IMAGE_ORACLE_VERSION",
    "SEMANTIC_SEED_FIELDS",
    "StainCorruptionMetadata",
    "StainCorruptionParameters",
    "StainCorruptionProfile",
    "StainCorruptionResult",
    "StainCorruptor",
    "clean_validation_passthrough",
    "corrupt_normalized_batch",
    "derive_corruption_seed",
    "hed_to_rgb",
    "normalized_to_rgb01",
    "profile_from_config",
    "profile_from_name",
    "rgb01_to_normalized",
    "rgb_to_hed",
    "sample_corruption_parameters",
    "semantic_sample_key",
]
