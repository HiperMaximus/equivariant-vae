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

from __future__ import annotations

from typing import TYPE_CHECKING

from eqvae.models.non_equivariant_vae import (
    DEFAULT_GROUPNORM_GROUPS,
    NonEquivariantVAE,
    build_non_equivariant_vae,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

MODEL_KIND_NON_EQ_TRANSLATABLE = "non_eq_vae_translatable"


def _build_non_eq_translatable(model_config: Mapping[str, object]) -> NonEquivariantVAE:
    raw_groups = model_config.get("norm_groups", DEFAULT_GROUPNORM_GROUPS)
    if isinstance(raw_groups, bool) or not isinstance(raw_groups, int):
        message = f"model_config 'norm_groups' must be an int, got {raw_groups!r}"
        raise TypeError(message)
    return build_non_equivariant_vae(norm_groups=raw_groups)


# One entry today. The eq model registers ``'eq_vae_so2' -> build_eq_vae`` later;
# the shared return type widens to the base VAE contract at that point.
_MODEL_BUILDERS: dict[str, Callable[[Mapping[str, object]], NonEquivariantVAE]] = {
    MODEL_KIND_NON_EQ_TRANSLATABLE: _build_non_eq_translatable,
}


def build_model(
    kind: str,
    *,
    model_config: Mapping[str, object] | None = None,
) -> NonEquivariantVAE:
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
    "build_model",
]
