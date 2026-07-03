# Copyright 2026 HiperMaximus
"""SSIM/VAE-loss torch.compile-cleanliness (fast-path port, steps 1 and 3)."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
import torch._dynamo as torch_dynamo  # noqa: PLC2701

from eqvae.losses.vae import VaeLossTensors, compute_vae_loss, vae_loss_core
from eqvae.metrics.reconstruction import ssim_per_image
from eqvae.models.non_equivariant_vae import build_non_equivariant_vae

if TYPE_CHECKING:
    from collections.abc import Callable

    from eqvae.models.non_equivariant_vae import VaeForwardOutput


def _loss_fixture() -> tuple[VaeForwardOutput, torch.Tensor]:
    model = build_non_equivariant_vae()
    generator = torch.Generator().manual_seed(0)
    inputs = (torch.rand((2, 3, 64, 64), generator=generator) * 2.0) - 1.0
    target = (torch.rand((2, 3, 64, 64), generator=generator) * 2.0) - 1.0
    with torch.no_grad():
        output = model.forward(inputs)
    return output, target


def test_ssim_per_image_traces_to_a_single_fullgraph() -> None:
    """ssim_per_image captures as one torch.compile graph (no .item() breaks)."""
    generator = torch.Generator().manual_seed(0)
    prediction = torch.rand((2, 3, 32, 32), generator=generator)
    target = torch.rand((2, 3, 32, 32), generator=generator)
    compiled = cast(
        "Callable[..., torch.Tensor]",
        torch.compile(  # pyright: ignore[reportUnknownMemberType]
            ssim_per_image,
            fullgraph=True,
            backend="eager",
        ),
    )

    result = compiled(prediction, target)

    torch.testing.assert_close(
        result,
        ssim_per_image(prediction, target),
        atol=0.0,
        rtol=0.0,
    )


def test_vae_loss_core_traces_to_a_single_fullgraph() -> None:
    """vae_loss_core (beta as a 0-dim tensor) captures as one torch.compile graph."""
    output, target = _loss_fixture()
    beta = torch.tensor(1.0)
    compiled = cast(
        "Callable[..., VaeLossTensors]",
        torch.compile(  # pyright: ignore[reportUnknownMemberType]
            vae_loss_core,
            fullgraph=True,
            backend="eager",
        ),
    )

    result = compiled(output, target, beta=beta, ssim_weight=0.1)

    torch.testing.assert_close(
        result.loss,
        vae_loss_core(output, target, beta=beta, ssim_weight=0.1).loss,
        atol=0.0,
        rtol=0.0,
    )


def test_vae_loss_core_does_not_recompile_when_beta_value_changes() -> None:
    """A changing beta value passed as a 0-dim tensor must not force a recompile."""
    output, target = _loss_fixture()
    compiled = cast(
        "Callable[..., VaeLossTensors]",
        torch.compile(  # pyright: ignore[reportUnknownMemberType]
            vae_loss_core,
            backend="eager",
        ),
    )
    torch_dynamo.reset()
    torch_dynamo.config.error_on_recompile = True
    try:
        for value in (0.1, 0.5, 1.0):
            beta = torch.tensor(value, dtype=torch.float32)
            compiled(output, target, beta=beta, ssim_weight=0.1)
    finally:
        torch_dynamo.config.error_on_recompile = False


def test_compute_vae_loss_matches_core() -> None:
    """The eager float-beta wrapper's loss equals the tensor-beta core's."""
    output, target = _loss_fixture()

    wrapper = compute_vae_loss(output, target, beta=0.5)
    core = vae_loss_core(output, target, beta=torch.tensor(0.5), ssim_weight=0.1)

    torch.testing.assert_close(wrapper.loss, core.loss, atol=0.0, rtol=0.0)
