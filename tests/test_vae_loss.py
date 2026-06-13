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
