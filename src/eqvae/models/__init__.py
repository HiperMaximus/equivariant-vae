# Copyright 2026 HiperMaximus
"""Model scaffolding for the translatable VAE implementation."""

from __future__ import annotations

from eqvae.models.activations import GatedScalarActivation
from eqvae.models.latent import LATENT_CHANNELS
from eqvae.models.local_global_mil import (
    CLASS_ORDER,
    EXPECTED_PARAMETER_COUNT,
    LOCAL_GLOBAL_MIL_CONFIG,
    LocalAttentionGraph,
    LocalGlobalMILClassifier,
    build_local_attention_graph,
    local_global_mil_adamw_parameter_groups,
    make_paired_local_global_mil_models,
)
from eqvae.models.non_equivariant_vae import (
    NonEquivariantVAE,
    VaeForwardOutput,
    build_non_equivariant_vae,
)
from eqvae.models.registry import (
    MODEL_KIND_NON_EQ_TRANSLATABLE,
    MODEL_KIND_SO2_FIXED,
    assert_fixed_so2_model,
    build_model,
)
from eqvae.models.resampling import (
    FieldwiseBilinearUpsample2x,
    FixedBinomialLowpassDownsample2x,
)
from eqvae.models.so2_vae import SO2VAE, build_so2_vae

__all__ = [
    "CLASS_ORDER",
    "EXPECTED_PARAMETER_COUNT",
    "LATENT_CHANNELS",
    "LOCAL_GLOBAL_MIL_CONFIG",
    "MODEL_KIND_NON_EQ_TRANSLATABLE",
    "MODEL_KIND_SO2_FIXED",
    "SO2VAE",
    "FieldwiseBilinearUpsample2x",
    "FixedBinomialLowpassDownsample2x",
    "GatedScalarActivation",
    "LocalAttentionGraph",
    "LocalGlobalMILClassifier",
    "NonEquivariantVAE",
    "VaeForwardOutput",
    "assert_fixed_so2_model",
    "build_local_attention_graph",
    "build_model",
    "build_non_equivariant_vae",
    "build_so2_vae",
    "local_global_mil_adamw_parameter_groups",
    "make_paired_local_global_mil_models",
]
