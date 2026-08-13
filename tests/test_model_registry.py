# Copyright 2026 HiperMaximus
"""Tests for the model-kind build registry (Spec 0011 S1)."""

from __future__ import annotations

import pytest

from eqvae.models.latent import LATENT_CHANNELS
from eqvae.models.non_equivariant_vae import (
    DEFAULT_GROUPNORM_GROUPS,
    NonEquivariantVAE,
)
from eqvae.models.registry import (
    MODEL_KIND_NON_EQ_TRANSLATABLE,
    MODEL_KIND_SO2_FIXED,
    assert_fixed_so2_model,
    build_model,
)
from eqvae.models.so2_vae import SO2VAE


def test_build_model_returns_non_eq_vae_with_defaults() -> None:
    """The registered kind builds the non-eq VAE with default norm groups."""
    model = build_model(MODEL_KIND_NON_EQ_TRANSLATABLE)
    assert isinstance(model, NonEquivariantVAE)
    assert model.norm_groups == DEFAULT_GROUPNORM_GROUPS
    assert model.latent_channels == LATENT_CHANNELS


def test_build_model_passes_norm_groups_from_config() -> None:
    """Per-kind kwargs are unpacked from the opaque model-config block."""
    requested_groups = 16
    model = build_model(
        MODEL_KIND_NON_EQ_TRANSLATABLE,
        model_config={"norm_groups": requested_groups},
    )
    assert isinstance(model, NonEquivariantVAE)
    assert model.norm_groups == requested_groups


def test_build_model_unknown_kind_raises_keyerror() -> None:
    """An unregistered kind fails closed rather than silently building nothing."""
    with pytest.raises(KeyError, match="unknown model kind"):
        build_model("does_not_exist")


def test_build_model_rejects_non_int_norm_groups() -> None:
    """A malformed norm_groups value is rejected rather than silently coerced."""
    with pytest.raises(TypeError, match="norm_groups"):
        build_model(
            MODEL_KIND_NON_EQ_TRANSLATABLE,
            model_config={"norm_groups": "eight"},
        )


def test_build_model_returns_exact_fixed_so2_vae() -> None:
    """The SO2 kind returns only the locked concrete Spec 0014 model."""
    model = build_model(MODEL_KIND_SO2_FIXED)
    assert type(model) is SO2VAE
    assert assert_fixed_so2_model(model) is model


def test_fixed_so2_registry_kind_rejects_architecture_options() -> None:
    """The fixed SO2 seam cannot become an architecture search surface."""
    with pytest.raises(ValueError, match="accepts no construction options"):
        build_model(MODEL_KIND_SO2_FIXED, model_config={"radial_profile": "other"})
