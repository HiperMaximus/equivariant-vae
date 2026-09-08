# Copyright 2026 HiperMaximus
"""Strict, hash-first loading for the two frozen Spec 0021 encoders."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, cast

import torch

from eqvae.data.latent_shards import EXPECTED_CHECKPOINT_SHA256
from eqvae.models.registry import (
    MODEL_KIND_NON_EQ_TRANSLATABLE,
    MODEL_KIND_SO2_FIXED,
    SupportedVAE,
    build_model,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from torch import Tensor

type FrozenModelName = Literal["normal_vae", "so2_vae"]

CHECKPOINT_SCHEMA_VERSION: Final = "spec0001.checkpoint.v5"
FINAL_OPTIMIZER_STEP: Final = 60_000
_HASH_CHUNK_BYTES: Final = 8 * 1024 * 1024


@dataclass(frozen=True)
class FrozenCheckpointSpec:
    """Immutable identity of one accepted frozen model/checkpoint pair."""

    model_name: FrozenModelName
    model_kind: str
    sha256: str
    learned_parameter_count: int


FROZEN_CHECKPOINT_SPECS: Final[Mapping[FrozenModelName, FrozenCheckpointSpec]] = (
    MappingProxyType({
        "normal_vae": FrozenCheckpointSpec(
            model_name="normal_vae",
            model_kind=MODEL_KIND_NON_EQ_TRANSLATABLE,
            sha256=EXPECTED_CHECKPOINT_SHA256["normal_vae"],
            learned_parameter_count=3_958_435,
        ),
        "so2_vae": FrozenCheckpointSpec(
            model_name="so2_vae",
            model_kind=MODEL_KIND_SO2_FIXED,
            sha256=EXPECTED_CHECKPOINT_SHA256["so2_vae"],
            learned_parameter_count=1_180_035,
        ),
    })
)


def load_frozen_checkpoint(
    path: Path,
    *,
    model_name: FrozenModelName,
) -> SupportedVAE:
    """Hash, validate, strictly load, and freeze one final encoder.

    The file is hashed before it is passed to ``torch.load``. Only the final
    60,000-update checkpoint schema is accepted, and construction always goes
    through the canonical model registry.

    Returns:
        The registry-built model in evaluation mode with gradients disabled.

    Raises:
        TypeError: If the checkpoint payload or state object has the wrong type.
        ValueError: If its hash, schema, steps, or model identity differs.

    """
    spec = FROZEN_CHECKPOINT_SPECS[model_name]
    observed_sha256 = _sha256_file(path)
    if observed_sha256 != spec.sha256:
        message = f"Frozen {model_name} checkpoint SHA-256 mismatch: {observed_sha256}"
        raise ValueError(message)

    payload = cast(
        "object",
        torch.load(path, map_location="cpu", weights_only=False),
    )
    if not isinstance(payload, dict):
        message = "Frozen checkpoint payload must be an object"
        raise TypeError(message)
    checkpoint = cast("Mapping[str, object]", payload)
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        message = "Frozen checkpoint schema version mismatch"
        raise ValueError(message)
    for key in ("optimizer_step", "successful_optimizer_update_count"):
        value = checkpoint.get(key)
        if isinstance(value, bool) or value != FINAL_OPTIMIZER_STEP:
            message = f"Frozen checkpoint {key} must equal {FINAL_OPTIMIZER_STEP}"
            raise ValueError(message)
    raw_state = checkpoint.get("model_state_dict")
    if not isinstance(raw_state, dict):
        message = "Frozen checkpoint has no model_state_dict object"
        raise TypeError(message)
    state = cast("Mapping[str, Tensor]", raw_state)

    model = build_model(spec.model_kind)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != spec.learned_parameter_count:
        message = (
            f"Frozen {model_name} model parameter count drift: "
            f"{parameter_count} != {spec.learned_parameter_count}"
        )
        raise ValueError(message)
    model.load_state_dict(state, strict=True)
    model.requires_grad_(requires_grad=False)
    model.eval()
    return model


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "FINAL_OPTIMIZER_STEP",
    "FROZEN_CHECKPOINT_SPECS",
    "FrozenCheckpointSpec",
    "FrozenModelName",
    "load_frozen_checkpoint",
]
