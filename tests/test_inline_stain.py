# Copyright 2026 HiperMaximus
"""Tests for the compile-friendly inline stain corruptor (fast-path step 2)."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest
import torch

from eqvae.corruption.inline_stain import InlineStainCorruptor
from eqvae.corruption.stain import (
    CONSERVATIVE_DEFAULT_PROFILE,
    StainCorruptionProfile,
    profile_from_name,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def _profile_with_prob(prob: float) -> StainCorruptionProfile:
    base = profile_from_name(CONSERVATIVE_DEFAULT_PROFILE)
    return StainCorruptionProfile(
        name="test",
        corrupt_prob=prob,
        he_alpha_range=base.he_alpha_range,
        he_beta_range=base.he_beta_range,
        residual_alpha_range=base.residual_alpha_range,
        residual_beta_range=base.residual_beta_range,
        noise_std_range=base.noise_std_range,
    )


def _images(batch: int, size: int) -> torch.Tensor:
    return (torch.rand((batch, 3, size, size)) * 2.0) - 1.0


def test_inline_corruptor_forward_is_fullgraph_compilable() -> None:
    """The branchless forward captures as one torch.compile graph (no breaks)."""
    corruptor = InlineStainCorruptor(profile_from_name(CONSERVATIVE_DEFAULT_PROFILE))
    images = _images(2, 32)
    compiled = cast(
        "Callable[..., torch.Tensor]",
        torch.compile(  # pyright: ignore[reportUnknownMemberType]
            corruptor,
            fullgraph=True,
            backend="eager",
        ),
    )

    result = compiled(images)

    assert result.shape == images.shape


def test_inline_corruptor_output_contract() -> None:
    """Output preserves shape/dtype/device and stays clamped to [-1, 1]."""
    corruptor = InlineStainCorruptor(profile_from_name(CONSERVATIVE_DEFAULT_PROFILE))
    images = _images(4, 16)

    result = corruptor.forward(images)

    assert result.shape == images.shape
    assert result.dtype == images.dtype
    assert result.device == images.device
    assert float(result.min().item()) >= -1.0
    assert float(result.max().item()) <= 1.0


def test_inline_corruptor_is_identity_at_zero_probability() -> None:
    """corrupt_prob=0 leaves every sample unchanged (branchless mask off)."""
    corruptor = InlineStainCorruptor(_profile_with_prob(0.0))
    images = _images(4, 16)

    torch.testing.assert_close(corruptor.forward(images), images, atol=0.0, rtol=0.0)


def test_inline_corruptor_changes_input_at_full_probability() -> None:
    """corrupt_prob=1 corrupts every sample, so the output differs from the input."""
    corruptor = InlineStainCorruptor(_profile_with_prob(1.0))
    images = _images(4, 16)

    assert not torch.equal(corruptor.forward(images), images)


def test_inline_corruptor_validates_profile_in_init() -> None:
    """Config validation happens at construction, not in the forward."""
    with pytest.raises(ValueError, match="corrupt_prob"):
        InlineStainCorruptor(_profile_with_prob(2.0))
