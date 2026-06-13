# Copyright 2026 HiperMaximus
"""Loss scaffolding for the future VAE objective."""

from __future__ import annotations

from eqvae.losses.vae import (
    VaeLossComponents,
    beta_for_step,
    beta_warmup_steps,
    compute_vae_loss,
    kl_divergence_loss,
)

__all__ = [
    "VaeLossComponents",
    "beta_for_step",
    "beta_warmup_steps",
    "compute_vae_loss",
    "kl_divergence_loss",
]
