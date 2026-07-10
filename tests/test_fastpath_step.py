# Copyright 2026 HiperMaximus
"""Tests for the shared compiled fast-path step_fn (fast-path port, step 4)."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
import torch._dynamo as torch_dynamo  # noqa: PLC2701

from eqvae.corruption.inline_stain import InlineStainCorruptor
from eqvae.corruption.stain import CONSERVATIVE_DEFAULT_PROFILE, profile_from_name
from eqvae.models.non_equivariant_vae import build_non_equivariant_vae
from eqvae.training.fastpath_step import FastpathStepOutput, make_fastpath_step_fn

if TYPE_CHECKING:
    from collections.abc import Callable


def _fixture() -> tuple[Callable[..., FastpathStepOutput], torch.Tensor, torch.Tensor]:
    model = build_non_equivariant_vae()
    corruptor = InlineStainCorruptor(profile_from_name(CONSERVATIVE_DEFAULT_PROFILE))
    step_fn = make_fastpath_step_fn(
        model,
        corruptor,
        ssim_weight=0.1,
        autocast_dtype=torch.bfloat16,
    )
    generator = torch.Generator().manual_seed(0)
    x_clean = (torch.rand((2, 3, 64, 64), generator=generator) * 2.0) - 1.0
    with torch.no_grad():
        mu_shape = model.forward(x_clean).mu.shape
    eps = torch.randn(mu_shape, generator=generator)
    return step_fn, x_clean, eps


def test_fastpath_step_fn_traces_to_a_single_fullgraph() -> None:
    """Corruption + autocast forward + FP32 loss island captures as one graph."""
    step_fn, x_clean, eps = _fixture()
    compiled = torch.compile(  # pyright: ignore[reportUnknownMemberType]
        step_fn,
        fullgraph=True,
        dynamic=False,
        backend="eager",
    )

    out = compiled(x_clean, eps, torch.tensor(1.0))

    assert out.loss.requires_grad
    assert not out.recon_loss.requires_grad
    assert not out.reconstruction.requires_grad


def test_fastpath_step_fn_does_not_recompile_when_beta_value_changes() -> None:
    """A changing beta value (0-dim tensor) must not force a recompile."""
    step_fn, x_clean, eps = _fixture()
    compiled = torch.compile(  # pyright: ignore[reportUnknownMemberType]
        step_fn,
        dynamic=False,
        backend="eager",
    )
    torch_dynamo.reset()
    torch_dynamo.config.error_on_recompile = True
    try:
        for value in (0.1, 0.5, 1.0):
            compiled(x_clean, eps, torch.tensor(value, dtype=torch.float32))
    finally:
        torch_dynamo.config.error_on_recompile = False


def test_fastpath_step_fn_runs_eagerly_and_backprops() -> None:
    """The eager step yields a finite grad-attached loss whose backward runs."""
    step_fn, x_clean, eps = _fixture()

    out = step_fn(x_clean, eps, torch.tensor(1.0))

    assert bool(torch.isfinite(out.loss).item())
    assert out.loss.requires_grad
    cast("Callable[[], None]", out.loss.backward)()


def test_fastpath_step_fn_autocast_enabled_gates_forward_precision() -> None:
    """`autocast_enabled` gates the forward autocast (Spec 0011 S16).

    With it off the forward runs in FP32 -- what a CPU caller (or AMP-off run) needs,
    matching the eager runner path that gates its own autocast on ``amp.enabled``. The
    GPU probe leaves it at the default ``True``, so the measured graph is unchanged; a
    mutation that ignored the flag would still autocast and produce a low-precision
    reconstruction in the disabled case.
    """
    model = build_non_equivariant_vae()
    corruptor = InlineStainCorruptor(profile_from_name(CONSERVATIVE_DEFAULT_PROFILE))
    generator = torch.Generator().manual_seed(0)
    x_clean = (torch.rand((2, 3, 64, 64), generator=generator) * 2.0) - 1.0
    with torch.no_grad():
        mu_shape = model.forward(x_clean).mu.shape
    eps = torch.randn(mu_shape, generator=generator)

    enabled = make_fastpath_step_fn(
        model,
        corruptor,
        ssim_weight=0.1,
        autocast_dtype=torch.bfloat16,
        autocast_enabled=True,
    )(x_clean, eps, torch.tensor(1.0))
    disabled = make_fastpath_step_fn(
        model,
        corruptor,
        ssim_weight=0.1,
        autocast_dtype=torch.bfloat16,
        autocast_enabled=False,
    )(x_clean, eps, torch.tensor(1.0))

    assert enabled.reconstruction.dtype == torch.bfloat16
    assert disabled.reconstruction.dtype == torch.float32
    assert disabled.loss.dtype == torch.float32
