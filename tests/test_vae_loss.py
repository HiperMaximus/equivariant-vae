# Copyright 2026 HiperMaximus
"""Tests for the spec 0001 VAE objective contract."""

from __future__ import annotations

import math

import torch

from eqvae.losses.vae import (
    beta_for_step,
    beta_warmup_steps,
    compute_vae_loss,
    kl_divergence_loss,
)
from eqvae.metrics.reconstruction import normalized_to_image_domain, ssim_per_image
from eqvae.models.non_equivariant_vae import VaeForwardOutput

BATCH = 2
CHANNELS = 3
IMAGE_SIZE = 16
LATENT_CHANNELS = 16
LATENT_SIZE = 4

# FU-002: sane bands plus an independent reduction/formula lock for the mean-reduced
# KL against reconstruction at beta = 1 on a representative early-training posterior.
# The mean reduction gives an O(0.1) KL; a ``.mean() -> .sum()`` regression inflates
# it by the full element count B*C*H*W (2*16*32*32 = 32768; 16384 per sample) and a
# code-level zeroing of KL drives it to 0. This is a loss-contract unit lock; live
# posterior-collapse detection over a run is the runtime CSV's job (FU-002b).
_REPRESENTATIVE_POSTERIOR_SEED = 20260702
_FULL_LATENT_SPATIAL = 32
_FULL_IMAGE_SIZE = 32
_KL_LOSS_LOWER_BOUND = 0.02
_KL_LOSS_UPPER_BOUND = 1.0
_KL_RECON_RATIO_LOWER_BOUND = 0.02
_KL_RECON_RATIO_UPPER_BOUND = 10.0
_KL_REDUCTION_REL_TOL = 1e-5


def test_vae_loss_uses_locked_reductions() -> None:
    """L1 is global, SSIM is per-image mean, and KL is latent global mean."""
    target = _normalized_fixture()
    reconstruction: torch.Tensor = (target * 0.5).detach()
    reconstruction.requires_grad_()
    mu = torch.full((BATCH, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE), 0.25)
    logvar_clamped = torch.zeros_like(mu)
    output = _forward_output(
        reconstruction=reconstruction,
        mu=mu,
        logvar=logvar_clamped,
        logvar_clamped=logvar_clamped,
    )

    components = compute_vae_loss(output, target, beta=0.5, ssim_weight=0.1)

    expected_l1 = (reconstruction - target).abs().mean()
    expected_ssim = ssim_per_image(
        normalized_to_image_domain(reconstruction),
        normalized_to_image_domain(target),
    ).mean()
    expected_kl = kl_divergence_loss(mu=mu, logvar_clamped=logvar_clamped)
    expected_loss = expected_l1 + (0.1 * (1.0 - expected_ssim)) + (0.5 * expected_kl)
    assert torch.allclose(components.l1_loss, expected_l1)
    assert torch.allclose(components.ssim_loss, 1.0 - expected_ssim)
    assert torch.allclose(components.kl_loss, expected_kl)
    assert torch.allclose(components.loss, expected_loss)


def test_kl_uses_clamped_logvar_not_raw_logvar() -> None:
    """A pathological raw logvar does not affect KL once clamped telemetry exists."""
    mu = torch.zeros((1, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE))
    raw_logvar = torch.full_like(mu, 100.0)
    clamped = torch.zeros_like(mu)

    kl = kl_divergence_loss(mu=mu, logvar_clamped=clamped)
    output = _forward_output(
        reconstruction=torch.zeros((1, CHANNELS, IMAGE_SIZE, IMAGE_SIZE)),
        mu=mu,
        logvar=raw_logvar,
        logvar_clamped=clamped,
    )

    components = compute_vae_loss(
        output,
        torch.zeros((1, CHANNELS, IMAGE_SIZE, IMAGE_SIZE)),
        beta=1.0,
    )
    assert torch.equal(kl, torch.zeros_like(kl))
    assert torch.equal(components.kl_loss, torch.zeros_like(components.kl_loss))


def test_step_limited_beta_schedule_is_zero_based() -> None:
    """The first successful optimizer update uses beta zero."""
    assert beta_warmup_steps(8, warmup_fraction=0.1) == 1
    assert math.isclose(
        beta_for_step(optimizer_step_index=0, max_optimizer_steps=8),
        0.0,
        abs_tol=0.0,
    )
    assert math.isclose(
        beta_for_step(optimizer_step_index=1, max_optimizer_steps=8),
        1.0,
        abs_tol=0.0,
    )


def test_kl_recon_balance_stays_in_sane_band_at_beta_one() -> None:
    """FU-002: lock the KL reduction/formula and KL/recon balance at beta = 1.

    An independent recomputation of the diagonal-Gaussian KL from the raw formula
    pins the mean reduction and coefficient exactly (a ``.mean() -> .sum()``
    regression, a per-sample sum, or a dropped ``0.5`` factor all break it); an
    absolute band pins the O(0.1) magnitude; and a ratio band asserts KL neither
    vanishes against nor dominates reconstruction for a representative
    early-training posterior. Live training-time collapse is watched via the run
    CSV (FU-002b), not this unit test.
    """
    generator = torch.Generator().manual_seed(_REPRESENTATIVE_POSTERIOR_SEED)
    latent_shape = (BATCH, LATENT_CHANNELS, _FULL_LATENT_SPATIAL, _FULL_LATENT_SPATIAL)
    image_shape = (BATCH, CHANNELS, _FULL_IMAGE_SIZE, _FULL_IMAGE_SIZE)
    mu = torch.randn(latent_shape, generator=generator) * 0.5
    logvar_clamped = (
        (torch.randn(latent_shape, generator=generator) * 0.2) - 0.5
    ).clamp(-8.0, 4.0)
    target = (torch.rand(image_shape, generator=generator) * 2.0) - 1.0
    reconstruction = (
        target + (torch.randn(image_shape, generator=generator) * 0.3)
    ).clamp(-1.0, 1.0)
    output = _forward_output(
        reconstruction=reconstruction,
        mu=mu,
        logvar=logvar_clamped,
        logvar_clamped=logvar_clamped,
    )

    components = compute_vae_loss(output, target, beta=1.0)

    kl_loss = float(components.kl_loss)
    recon_loss = float(components.recon_loss)
    # Independent reduction+formula lock: recompute the KL from the raw formula and
    # require the mean reduction exactly. A .mean()->.sum() (kl ~ 6072), a
    # per-sample sum, or a coefficient error all break this equality.
    kl_element = -0.5 * (1.0 + logvar_clamped - mu.square() - logvar_clamped.exp())
    expected_mean_kl = float(kl_element.mean())
    assert math.isclose(kl_loss, expected_mean_kl, rel_tol=_KL_REDUCTION_REL_TOL)
    # Sane magnitude band: the mean reduction is O(0.1), never the O(10^3) sum form
    # nor a collapse to 0.
    assert _KL_LOSS_LOWER_BOUND <= kl_loss <= _KL_LOSS_UPPER_BOUND
    # Balance band at beta = 1: KL neither vanishes against nor dominates recon.
    assert kl_loss / recon_loss >= _KL_RECON_RATIO_LOWER_BOUND
    assert kl_loss / recon_loss <= _KL_RECON_RATIO_UPPER_BOUND


def _forward_output(
    *,
    reconstruction: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    logvar_clamped: torch.Tensor,
) -> VaeForwardOutput:
    eps = torch.zeros_like(mu)
    return VaeForwardOutput(
        reconstruction=reconstruction,
        mu=mu,
        logvar=logvar,
        logvar_clamped=logvar_clamped,
        z=mu,
        eps=eps,
        logvar_clamp_count=torch.count_nonzero(logvar != logvar_clamped),
    )


def _normalized_fixture() -> torch.Tensor:
    values = torch.linspace(
        -1.0,
        1.0,
        steps=BATCH * CHANNELS * IMAGE_SIZE * IMAGE_SIZE,
        dtype=torch.float32,
    )
    return values.reshape(BATCH, CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
