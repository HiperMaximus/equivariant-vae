# Copyright 2026 HiperMaximus
"""Model-kind build registry — the reusable per-model seam (Spec 0011 S1/MF4).

``build_model`` is the single entry every construction site uses to build a model.
A future model kind is added by registering one builder here (and setting
``model.kind`` in its config) rather than by editing every construction site, so
the runtime machinery stays model-agnostic.

Per-kind construction kwargs are unpacked opaquely from the model-config block, so
a non-equivariant-only concept (``norm_groups``, a GroupNorm parameter) is never
promoted to a universal signature the field-aware equivariant model has no use for
(Spec 0011 R2).
"""
# pyright: reportPrivateUsage=false

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Literal, cast, overload

from eqvae.models.latent import LATENT_CHANNELS
from eqvae.models.non_equivariant_vae import (
    DEFAULT_GROUPNORM_GROUPS,
    NonEquivariantVAE,
    build_non_equivariant_vae,
)
from eqvae.models.so2_architecture_probe import (
    FixedF01RadialGate,
    _F01ToF01Conv,
    _F01ToScalarConv,
    _ScalarToF01Conv,
)
from eqvae.models.so2_vae import SO2VAE, build_so2_vae

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from torch import nn

MODEL_KIND_NON_EQ_TRANSLATABLE: Final = "non_eq_vae_translatable"
MODEL_KIND_SO2_FIXED: Final = "so2_vae_fixed"
_SO2_LEARNED_CONVOLUTION_COUNT = 43
_SO2_RADIAL_GATE_COUNT = 34
_SO2_PARAMETER_COUNT = 1_180_035

type SupportedVAE = NonEquivariantVAE | SO2VAE


def _build_non_eq_translatable(model_config: Mapping[str, object]) -> NonEquivariantVAE:
    raw_groups = model_config.get("norm_groups", DEFAULT_GROUPNORM_GROUPS)
    if isinstance(raw_groups, bool) or not isinstance(raw_groups, int):
        message = f"model_config 'norm_groups' must be an int, got {raw_groups!r}"
        raise TypeError(message)
    return build_non_equivariant_vae(norm_groups=raw_groups)


def _build_so2_fixed(model_config: Mapping[str, object]) -> SO2VAE:
    if model_config:
        message = (
            "the fixed SO2 model accepts no construction options; "
            f"got {sorted(model_config)}"
        )
        raise ValueError(message)
    return assert_fixed_so2_model(build_so2_vae())


def assert_fixed_so2_model(model: nn.Module) -> SO2VAE:
    """Fail closed unless ``model`` is the singular Spec 0014 architecture.

    Returns:
        The concretely typed fixed model.

    Raises:
        ValueError: If identity, topology, latent width, or count has drifted.

    """
    learned_types = _ScalarToF01Conv | _F01ToF01Conv | _F01ToScalarConv
    facts = {
        "concrete_class": type(model) is SO2VAE,
        "latent_channels": getattr(model, "latent_channels", None) == LATENT_CHANNELS,
        "learned_convolutions": sum(
            isinstance(module, learned_types) for module in model.modules()
        )
        == _SO2_LEARNED_CONVOLUTION_COUNT,
        "radial_gates": sum(
            isinstance(module, FixedF01RadialGate) for module in model.modules()
        )
        == _SO2_RADIAL_GATE_COUNT,
        "learned_parameters": sum(parameter.numel() for parameter in model.parameters())
        == _SO2_PARAMETER_COUNT,
    }
    failures = sorted(name for name, passed in facts.items() if not passed)
    if failures:
        message = f"fixed SO2 model identity drift: {failures}"
        raise ValueError(message)
    return cast("SO2VAE", model)


# The fixed SO2 entry is deliberately singular: it exposes no architecture kwargs.
_MODEL_BUILDERS: dict[str, Callable[[Mapping[str, object]], SupportedVAE]] = {
    MODEL_KIND_NON_EQ_TRANSLATABLE: _build_non_eq_translatable,
    MODEL_KIND_SO2_FIXED: _build_so2_fixed,
}


@overload
def build_model(
    kind: Literal["non_eq_vae_translatable"],
    *,
    model_config: Mapping[str, object] | None = None,
) -> NonEquivariantVAE: ...


@overload
def build_model(
    kind: Literal["so2_vae_fixed"],
    *,
    model_config: Mapping[str, object] | None = None,
) -> SO2VAE: ...


@overload
def build_model(
    kind: str,
    *,
    model_config: Mapping[str, object] | None = None,
) -> SupportedVAE: ...


def build_model(
    kind: str,
    *,
    model_config: Mapping[str, object] | None = None,
) -> SupportedVAE:
    """Build a model by kind, unpacking kind-specific kwargs from ``model_config``.

    Returns:
        The instantiated model for ``kind``.

    Raises:
        KeyError: If ``kind`` is not a registered model kind.

    """
    try:
        builder = _MODEL_BUILDERS[kind]
    except KeyError:
        known = sorted(_MODEL_BUILDERS)
        message = f"unknown model kind {kind!r}; registered kinds: {known}"
        raise KeyError(message) from None
    return builder(model_config if model_config is not None else {})


__all__ = [
    "MODEL_KIND_NON_EQ_TRANSLATABLE",
    "MODEL_KIND_SO2_FIXED",
    "SupportedVAE",
    "assert_fixed_so2_model",
    "build_model",
]
