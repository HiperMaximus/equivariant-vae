# Copyright 2026 HiperMaximus
"""Model scaffolding for the translatable VAE implementation."""

from __future__ import annotations

from eqvae.models.activations import GatedScalarActivation
from eqvae.models.latent import LATENT_CHANNELS
from eqvae.models.non_equivariant_vae import (
    NonEquivariantVAE,
    VaeForwardOutput,
    build_non_equivariant_vae,
)
from eqvae.models.registry import (
    MODEL_KIND_NON_EQ_TRANSLATABLE,
    build_model,
)
from eqvae.models.resampling import (
    FieldwiseBilinearUpsample2x,
    FixedBinomialLowpassDownsample2x,
)

__all__ = [
    "LATENT_CHANNELS",
    "MODEL_KIND_NON_EQ_TRANSLATABLE",
    "FieldwiseBilinearUpsample2x",
    "FixedBinomialLowpassDownsample2x",
    "GatedScalarActivation",
    "NonEquivariantVAE",
    "VaeForwardOutput",
    "build_model",
    "build_non_equivariant_vae",
]
