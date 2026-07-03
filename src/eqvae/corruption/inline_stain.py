# Copyright 2026 HiperMaximus
"""Compile-friendly inline HED stain corruptor for the fast-path training step.

Unlike the reproducible `corrupt_normalized_batch` (blake2b per-sample seeding, kept
for validation and determinism tests), this module draws corruption parameters with
inline `torch.rand`/`torch.randn` so the whole thing fuses into the compiled train
step. It is intentionally non-deterministic and rank-local (speed-first hot path).
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

    def forward(self, images: Tensor) -> Tensor:
        """Corrupt a normalized `[-1, 1]` NCHW batch with branchless tensor ops.

        Returns:
            Corrupted batch in the input dtype, clamped to `[-1, 1]`.

        """
        input_dtype = images.dtype
        with torch.no_grad():
            work = images.to(dtype=torch.float32)
            batch = work.shape[0]
            applied = (
                torch.rand((batch, 1, 1, 1), device=work.device, dtype=torch.float32)
                < self.corrupt_prob
            )
            alpha = _sample_range(
                cast("Tensor", self.alpha_min),
                cast("Tensor", self.alpha_max),
                batch=batch,
                channels=RGB_CHANNELS,
            )
            beta = _sample_range(
                cast("Tensor", self.beta_min),
                cast("Tensor", self.beta_max),
                batch=batch,
                channels=RGB_CHANNELS,
            )
            noise_std = self.noise_std_min + torch.rand(
                (batch, 1, 1, 1),
                device=work.device,
                dtype=torch.float32,
            ) * (self.noise_std_max - self.noise_std_min)
            noise = torch.randn_like(work) * noise_std
            rgb = normalized_to_rgb01(work)
            hed = rgb_to_hed(rgb, hed_from_rgb=cast("Tensor", self.hed_from_rgb))
            jittered = hed_to_rgb(
                (hed * alpha) + beta,
                rgb_from_hed=cast("Tensor", self.rgb_from_hed),
            )
            stain = rgb01_to_normalized(jittered)
            corrupted = torch.where(applied, stain + noise, work)
            return corrupted.clamp(-1.0, 1.0).to(dtype=input_dtype)


def _sample_range(
    lower: Tensor,
    upper: Tensor,
    *,
    batch: int,
    channels: int,
) -> Tensor:
    unit = torch.rand(
        (batch, channels, 1, 1),
        device=lower.device,
        dtype=torch.float32,
    )
    return lower + unit * (upper - lower)


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
