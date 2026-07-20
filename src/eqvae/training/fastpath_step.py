# Copyright 2026 HiperMaximus
"""Shared compiled train-step closure for the FSQ-style fast path.

`make_fastpath_step_fn` builds the single `step_fn(x_uint8, eps, beta)` that both the
synthetic GPU probe and the selected-runtime runner compile with
`torch.compile(dynamic=False)`. The step fuses the uint8->float normalize, branchless
corruption, the AMP-autocast model forward, and the FP32 loss island into one graph, so
the caller transfers uint8 over H2D (4x fewer bytes) and the cast/normalize runs on the
GPU inside the graph; backward, GradScaler, gradient clipping, and the optimizer step
stay eager in the caller (like the FSQ reference).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, cast

import torch
from torch import Tensor

from eqvae.data.dataloaders import normalize_uint8_batch
from eqvae.losses.vae import vae_loss_core

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import nn

    from eqvae.models.non_equivariant_vae import VaeForwardOutput


class FastpathStepOutput(NamedTuple):
    """Compiled-step outputs: the grad-attached loss plus detached telemetry."""

    loss: Tensor
    recon_loss: Tensor
    l1_loss: Tensor
    ssim_loss: Tensor
    ssim_metric: Tensor
    kl_loss: Tensor
    reconstruction: Tensor
    logvar_clamp_count: Tensor


def make_fastpath_step_fn(
    model: nn.Module,
    corruptor: nn.Module,
    *,
    ssim_weight: float,
    autocast_dtype: torch.dtype,
    autocast_enabled: bool = True,
) -> Callable[[Tensor, Tensor, Tensor], FastpathStepOutput]:
    """Return the compile-ready `step_fn(x_uint8, eps, beta)` closure.

    `x_uint8` is the device-resident uint8 NCHW batch (transferred with channels_last
    fused into the H2D ``.to()``); the step normalizes it to FP32 ``[-1, 1]`` inside the
    graph, so the cast runs on the GPU and only uint8 crosses the host boundary. `beta`
    is a 0-dim device tensor (avoids per-step recompiles) and `eps` is a graph input
    produced eagerly per rank by the caller. The model is invoked via ``__call__`` (not
    ``.forward``) so DDP / compiled-autograd hooks fire; corruption runs branchless
    under its own ``no_grad``; the loss island runs in FP32 outside autocast.

    ``autocast_enabled`` gates the forward autocast so a CPU caller (or any run with AMP
    off) forwards in plain FP32 -- matching the eager runner path, which gates its own
    autocast on ``amp.enabled``. The GPU probe (and the real AMP fast path) leaves it at
    the default ``True``, so the measured compiled graph is unchanged.

    Returns:
        The `step_fn` closure returning a `FastpathStepOutput`.

    """

    def step_fn(x_uint8: Tensor, eps: Tensor, beta: Tensor) -> FastpathStepOutput:
        x_clean = normalize_uint8_batch(x_uint8)
        x_in = cast("Tensor", corruptor(x_clean))
        with torch.autocast(
            device_type=x_uint8.device.type,
            dtype=autocast_dtype,
            enabled=autocast_enabled,
            cache_enabled=False,
        ):
            output = cast("VaeForwardOutput", model(x_in, eps=eps))
        losses = vae_loss_core(output, x_clean, beta=beta, ssim_weight=ssim_weight)
        return FastpathStepOutput(
            loss=losses.loss,
            recon_loss=losses.recon_loss.detach(),
            l1_loss=losses.l1_loss.detach(),
            ssim_loss=losses.ssim_loss.detach(),
            ssim_metric=losses.ssim_metric.detach(),
            kl_loss=losses.kl_loss.detach(),
            reconstruction=output.reconstruction.detach(),
            logvar_clamp_count=output.logvar_clamp_count.detach(),
        )

    return step_fn


__all__ = ["FastpathStepOutput", "make_fastpath_step_fn"]
