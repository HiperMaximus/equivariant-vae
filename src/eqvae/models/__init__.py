# Copyright 2026 HiperMaximus
"""Model scaffolding for the translatable VAE implementation."""

from __future__ import annotations

from eqvae.models.activations import GatedScalarActivation
from eqvae.models.non_equivariant_vae import (
    NonEquivariantVAE,
    VaeForwardOutput,
    build_non_equivariant_vae,
)
from eqvae.models.resampling import (
    FieldwiseBilinearUpsample2x,
    FixedBinomialLowpassDownsample2x,
)

__all__ = [
    "FieldwiseBilinearUpsample2x",
    "FixedBinomialLowpassDownsample2x",
    "GatedScalarActivation",
    "NonEquivariantVAE",
    "VaeForwardOutput",
    "build_non_equivariant_vae",
]
