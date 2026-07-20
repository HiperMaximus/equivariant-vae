# Copyright 2026 HiperMaximus
"""Compile-friendly inline HED stain corruptor for the training + validation paths.

This module draws corruption parameters with inline `torch.rand`/`torch.randn` so the
whole thing fuses into the compiled train step -- a vectorized native torch RNG draw
(Philox on CUDA) rather than the retired blake2b per-sample seeding (Spec 0011 S17f).
The compiled fast path calls it
with ``generator=None`` (the process-global RNG, fuse-friendly and rank-local,
speed-first). The eager training and validation paths pass an explicit
``torch.Generator`` so their corruption stream can be checkpoint-continued (training)
or re-seeded to a fixed constant each boundary (validation, for stable best-checkpoint
selection); the sampled distributions are identical either way.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from eqvae.corruption.stain import (
    HED_FROM_RGB,
    RGB_CHANNELS,
    RGB_FROM_HED,
    StainCorruptionProfile,
    hed_to_rgb,
    normalized_to_rgb01,
    rgb01_to_normalized,
    rgb_to_hed,
)

_PROB_LOWER = 0.0
_PROB_UPPER = 1.0


class InlineStainCorruptor(nn.Module):
    """Branchless, torch.compile-friendly HED stain + Gaussian-noise corruptor."""

    def __init__(self, profile: StainCorruptionProfile) -> None:
        """Validate the profile and build the fixed HED / jitter-range buffers."""
        super().__init__()
        _validate_profile(profile)
        self.corrupt_prob = profile.corrupt_prob
        self.noise_std_min, self.noise_std_max = profile.noise_std_range
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
        self.register_buffer(
            "alpha_min",
            _channel_bounds(profile.he_alpha_range[0], profile.residual_alpha_range[0]),
            persistent=False,
        )
        self.register_buffer(
            "alpha_max",
            _channel_bounds(profile.he_alpha_range[1], profile.residual_alpha_range[1]),
            persistent=False,
        )
        self.register_buffer(
            "beta_min",
            _channel_bounds(profile.he_beta_range[0], profile.residual_beta_range[0]),
            persistent=False,
        )
        self.register_buffer(
            "beta_max",
            _channel_bounds(profile.he_beta_range[1], profile.residual_beta_range[1]),
            persistent=False,
        )

    def forward(
        self,
        images: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Corrupt a normalized `[-1, 1]` NCHW batch with branchless tensor ops.

        Args:
            images: Normalized `[-1, 1]` NCHW batch to corrupt.
            generator: Optional ``torch.Generator`` (on ``images.device``) supplying the
                corruption randomness. ``None`` (the default, used by the compiled fast
                path) draws from the process-global RNG and keeps ``randn_like`` so the
                graph stays fuse-friendly and preserves the input memory format; the
                eager training / validation paths pass their own generator for a
                checkpoint-continued or re-seeded stream.

        Returns:
            Corrupted batch in the input dtype, clamped to `[-1, 1]`.

        """
        input_dtype = images.dtype
        with torch.no_grad():
            work = images.to(dtype=torch.float32)
            batch = work.shape[0]
            applied = (
                _rand((batch, 1, 1, 1), device=work.device, generator=generator)
                < self.corrupt_prob
            )
            alpha = _sample_range(
                cast("Tensor", self.alpha_min),
                cast("Tensor", self.alpha_max),
                batch=batch,
                channels=RGB_CHANNELS,
                generator=generator,
            )
            beta = _sample_range(
                cast("Tensor", self.beta_min),
                cast("Tensor", self.beta_max),
                batch=batch,
                channels=RGB_CHANNELS,
                generator=generator,
            )
            noise_std = self.noise_std_min + _rand(
                (batch, 1, 1, 1),
                device=work.device,
                generator=generator,
            ) * (self.noise_std_max - self.noise_std_min)
            noise = _randn_like(work, generator=generator) * noise_std
            rgb = normalized_to_rgb01(work)
            hed = rgb_to_hed(rgb, hed_from_rgb=cast("Tensor", self.hed_from_rgb))
            jittered = hed_to_rgb(
                (hed * alpha) + beta,
                rgb_from_hed=cast("Tensor", self.rgb_from_hed),
            )
            stain = rgb01_to_normalized(jittered)
            corrupted = torch.where(applied, stain + noise, work)
            return corrupted.clamp(-1.0, 1.0).to(dtype=input_dtype)


def _rand(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    generator: torch.Generator | None,
) -> Tensor:
    """Draw ``U[0, 1)`` of ``shape`` in float32, emitting the exact seedless overload.

    Passing ``generator=None`` to ``torch.rand`` selects the ``aten.rand.generator``
    overload instead of ``aten.rand.default``; the compiled fast path (which passes
    ``None``) must emit the same op the pre-generator code did, so the None branch omits
    the kwarg. That keeps the measured compiled recipe byte-identical without relying on
    inductor normalizing the two overloads. An explicit generator uses ``.generator``.

    Returns:
        A float32 ``U[0, 1)`` tensor of ``shape`` on ``device``.

    """
    if generator is None:
        return torch.rand(shape, device=device, dtype=torch.float32)
    return torch.rand(shape, device=device, dtype=torch.float32, generator=generator)


def _sample_range(
    lower: Tensor,
    upper: Tensor,
    *,
    batch: int,
    channels: int,
    generator: torch.Generator | None,
) -> Tensor:
    unit = _rand(
        (batch, channels, 1, 1),
        device=lower.device,
        generator=generator,
    )
    return lower + unit * (upper - lower)


def _randn_like(reference: Tensor, *, generator: torch.Generator | None) -> Tensor:
    """Draw standard-normal noise shaped like ``reference``.

    With ``generator=None`` this is ``torch.randn_like`` (the compiled fast path),
    which preserves ``reference``'s memory format so the fused channels_last graph is
    unchanged. With an explicit generator ``randn_like`` cannot take one, so it falls
    back to ``torch.randn`` with the reference's shape/dtype/device.

    Returns:
        A standard-normal tensor matching ``reference``'s shape, dtype, and device.

    """
    if generator is None:
        return torch.randn_like(reference)
    return torch.randn(
        reference.shape,
        device=reference.device,
        dtype=reference.dtype,
        generator=generator,
    )


def _channel_bounds(he_value: float, residual_value: float) -> Tensor:
    return torch.tensor(
        [he_value, he_value, residual_value],
        dtype=torch.float32,
    ).view(1, RGB_CHANNELS, 1, 1)


def _validate_profile(profile: StainCorruptionProfile) -> None:
    if not _PROB_LOWER <= profile.corrupt_prob <= _PROB_UPPER:
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
