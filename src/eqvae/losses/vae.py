# Copyright 2026 HiperMaximus
"""Spec 0001 VAE loss and beta schedule."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from eqvae.metrics.reconstruction import normalized_to_image_domain, ssim_per_image

if TYPE_CHECKING:
    from eqvae.models.non_equivariant_vae import VaeForwardOutput


@dataclass(frozen=True)
class VaeLossComponents:
    """Scalar tensors for the spec 0001 composite VAE objective."""

    loss: torch.Tensor
    recon_loss: torch.Tensor
    l1_loss: torch.Tensor
    ssim_loss: torch.Tensor
    ssim_metric: torch.Tensor
    kl_loss: torch.Tensor
    beta: float

    def detached_scalars(self) -> dict[str, float]:
        """Return JSON/CSV-safe scalar floats detached from autograd.

        Returns:
            Loss component values.

        """
        return {
            "loss": _tensor_float(self.loss),
            "recon_loss": _tensor_float(self.recon_loss),
            "l1_loss": _tensor_float(self.l1_loss),
            "ssim_loss": _tensor_float(self.ssim_loss),
            "ssim_metric": _tensor_float(self.ssim_metric),
            "kl_loss": _tensor_float(self.kl_loss),
            "beta": self.beta,
        }


def compute_vae_loss(
    output: VaeForwardOutput,
    target: torch.Tensor,
    *,
    beta: float,
    ssim_weight: float = 0.1,
) -> VaeLossComponents:
    """Compute `L1 + ssim_weight * (1 - SSIM) + beta * KL`.

    Returns:
        Scalar loss components with gradients attached where appropriate.

    Raises:
        ValueError: If loss weights or tensor shapes are invalid.

    """
    if beta < 0.0:
        message = f"beta must be nonnegative, got {beta}"
        raise ValueError(message)
    if ssim_weight < 0.0:
        message = f"ssim_weight must be nonnegative, got {ssim_weight}"
        raise ValueError(message)
    reconstruction = output.reconstruction.to(dtype=torch.float32)
    target_f32 = target.to(dtype=torch.float32)
    if reconstruction.shape != target_f32.shape:
        message = (
            "Reconstruction and target shapes differ: "
            f"{reconstruction.shape} vs {target_f32.shape}"
        )
        raise ValueError(message)

    # FU-004: L1 penalizes the raw unbounded output, so it reflects any excess the
    # zero-init head pushes outside [-1, 1] (there is no final tanh). SSIM instead
    # runs on `normalized_to_image_domain`, which clamps to [0, 1], so out-of-range
    # pixels get a zero SSIM gradient and are invisible there; decoder-head
    # saturation is observed via the FU-018 train-step telemetry, not this term.
    l1_loss = (reconstruction - target_f32).abs().mean()
    ssim_metric = ssim_per_image(
        normalized_to_image_domain(reconstruction),
        normalized_to_image_domain(target_f32),
    ).mean()
    ssim_loss = 1.0 - ssim_metric
    recon_loss = l1_loss + (ssim_weight * ssim_loss)
    kl_loss = kl_divergence_loss(mu=output.mu, logvar_clamped=output.logvar_clamped)
    total_loss = recon_loss + (beta * kl_loss)
    return VaeLossComponents(
        loss=total_loss,
        recon_loss=recon_loss,
        l1_loss=l1_loss,
        ssim_loss=ssim_loss,
        ssim_metric=ssim_metric,
        kl_loss=kl_loss,
        beta=beta,
    )


def kl_divergence_loss(
    *,
    mu: torch.Tensor,
    logvar_clamped: torch.Tensor,
) -> torch.Tensor:
    """Return mean diagonal-Gaussian KL over batch, channel, height, and width.

    Returns:
        Scalar KL tensor.

    Raises:
        ValueError: If posterior tensors have different shapes.

    """
    mu_f32 = mu.to(dtype=torch.float32)
    logvar_f32 = logvar_clamped.to(dtype=torch.float32)
    if mu_f32.shape != logvar_f32.shape:
        message = f"mu/logvar shapes differ: {mu_f32.shape} vs {logvar_f32.shape}"
        raise ValueError(message)
    kl_element = -0.5 * (1.0 + logvar_f32 - mu_f32.square() - logvar_f32.exp())
    return kl_element.mean()


def beta_warmup_steps(max_optimizer_steps: int, *, warmup_fraction: float = 0.1) -> int:
    """Return the step-limited debug beta warmup length.

    Returns:
        At least one warmup step.

    Raises:
        ValueError: If the step count or warmup fraction is invalid.

    """
    if max_optimizer_steps <= 0:
        message = f"max_optimizer_steps must be positive, got {max_optimizer_steps}"
        raise ValueError(message)
    if warmup_fraction <= 0.0:
        message = f"warmup_fraction must be positive, got {warmup_fraction}"
        raise ValueError(message)
    return max(1, math.ceil(warmup_fraction * max_optimizer_steps))


def beta_for_step(
    *,
    optimizer_step_index: int,
    max_optimizer_steps: int,
    target_beta: float = 1.0,
    warmup_fraction: float = 0.1,
) -> float:
    """Return beta for a zero-based pre-update optimizer step index.

    Returns:
        Scheduled beta value.

    Raises:
        ValueError: If the step index or beta settings are invalid.

    """
    if optimizer_step_index < 0:
        message = (
            f"optimizer_step_index must be nonnegative, got {optimizer_step_index}"
        )
        raise ValueError(message)
    if target_beta < 0.0:
        message = f"target_beta must be nonnegative, got {target_beta}"
        raise ValueError(message)
    warmup_steps = beta_warmup_steps(
        max_optimizer_steps,
        warmup_fraction=warmup_fraction,
    )
    progress = min(optimizer_step_index / warmup_steps, 1.0)
    return target_beta * progress


def _tensor_float(tensor: torch.Tensor) -> float:
    return float(tensor.detach().to(dtype=torch.float32).item())


__all__ = [
    "VaeLossComponents",
    "beta_for_step",
    "beta_warmup_steps",
    "compute_vae_loss",
    "kl_divergence_loss",
]
