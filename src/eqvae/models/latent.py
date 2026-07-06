# Copyright 2026 HiperMaximus
"""Shared latent-space constants for every VAE architecture (Spec 0011).

The latent channel width is a fixed global design choice shared by the
non-equivariant baseline AND the future SO(2)-equivariant model -- a project
invariant, not a per-model constant. Generic runtime code still reads the width
from the built model's ``latent_channels`` attribute (which each model sets from
this constant); this module is the single source both architectures build from.
"""

from __future__ import annotations

LATENT_CHANNELS = 16
"""Latent channel width shared by every VAE architecture in this project."""

__all__ = ["LATENT_CHANNELS"]
