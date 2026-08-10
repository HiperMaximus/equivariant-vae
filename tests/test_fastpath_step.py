# Copyright 2026 HiperMaximus
"""Tests for the shared compiled fast-path step_fn (fast-path port, step 4).

The step folds the uint8->float normalize into the graph (Spec 0011 S17f), so it takes
the device-resident uint8 batch and casts/normalizes it internally.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, cast

import torch
import torch._dynamo as torch_dynamo  # noqa: PLC2701

from eqvae.corruption.inline_stain import InlineStainCorruptor
from eqvae.corruption.stain import CONSERVATIVE_DEFAULT_PROFILE, profile_from_name
from eqvae.data.dataloaders import normalize_uint8_batch
from eqvae.models.non_equivariant_vae import build_non_equivariant_vae
from eqvae.training.fastpath_step import FastpathStepOutput, make_fastpath_step_fn

if TYPE_CHECKING:
    from collections.abc import Callable


def _uint8_batch(generator: torch.Generator) -> torch.Tensor:
    # Graph structure/precision is shape-independent here; 1x16x16 is the smallest
    # valid three-downsample VAE input and keeps CPU compile/backward checks cheap.
    return torch.randint(0, 256, (1, 3, 16, 16), generator=generator, dtype=torch.uint8)


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
    x_uint8 = _uint8_batch(generator)
    with torch.no_grad():
        mu_shape = model.forward(normalize_uint8_batch(x_uint8)).mu.shape
    eps = torch.randn(mu_shape, generator=generator)
    return step_fn, x_uint8, eps


def test_fastpath_step_fn_traces_to_a_single_fullgraph() -> None:
    """The folded uint8 normalize + corruption + forward + loss trace as one graph."""
    step_fn, x_uint8, eps = _fixture()
    compiled = torch.compile(  # pyright: ignore[reportUnknownMemberType]
        step_fn,
        fullgraph=True,
        dynamic=False,
        backend="eager",
    )

    out = compiled(x_uint8, eps, torch.tensor(1.0))

    assert out.loss.requires_grad
    assert not out.recon_loss.requires_grad
    assert not out.reconstruction.requires_grad


def test_fastpath_step_fn_does_not_recompile_when_beta_value_changes() -> None:
    """A changing beta value (0-dim tensor) must not force a recompile."""
    step_fn, x_uint8, eps = _fixture()
    compiled = torch.compile(  # pyright: ignore[reportUnknownMemberType]
        step_fn,
        dynamic=False,
        backend="eager",
    )
    torch_dynamo.reset()
    torch_dynamo.config.error_on_recompile = True
    try:
        for value in (0.1, 0.5, 1.0):
            compiled(x_uint8, eps, torch.tensor(value, dtype=torch.float32))
    finally:
        torch_dynamo.config.error_on_recompile = False


def test_fastpath_step_fn_runs_eagerly_and_backprops() -> None:
    """The eager step yields a finite grad-attached loss whose backward runs."""
    step_fn, x_uint8, eps = _fixture()

    out = step_fn(x_uint8, eps, torch.tensor(1.0))

    assert bool(torch.isfinite(out.loss).item())
    assert out.loss.requires_grad
    cast("Callable[[], None]", out.loss.backward)()


def test_fastpath_step_fn_folds_the_uint8_normalize() -> None:
    """The step folds the uint8->[-1, 1] normalize into the graph.

    With corruption inert (``corrupt_prob=0``) the folded step reproduces a direct
    forward on ``normalize_uint8_batch(x_uint8)`` -- proving the cast moved into the
    graph without changing the math. Compared with a numerical tolerance (speed-first
    tolerates small drift, not bit-exact); a mutation that skipped or mis-scaled the
    internal normalize would shift the reconstruction outside the tolerance.
    """
    model = build_non_equivariant_vae()
    inert = replace(profile_from_name(CONSERVATIVE_DEFAULT_PROFILE), corrupt_prob=0.0)
    corruptor = InlineStainCorruptor(inert)
    generator = torch.Generator().manual_seed(0)
    x_uint8 = _uint8_batch(generator)
    x_clean = normalize_uint8_batch(x_uint8)
    with torch.no_grad():
        mu_shape = model.forward(x_clean).mu.shape
    eps = torch.randn(mu_shape, generator=generator)
    step_fn = make_fastpath_step_fn(
        model,
        corruptor,
        ssim_weight=0.1,
        autocast_dtype=torch.float32,
        autocast_enabled=False,
    )

    folded = step_fn(x_uint8, eps, torch.tensor(1.0))
    with torch.no_grad():
        reference = model.forward(x_clean, eps=eps)

    assert torch.allclose(folded.reconstruction, reference.reconstruction, atol=1e-5)


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
    x_uint8 = _uint8_batch(generator)
    with torch.no_grad():
        mu_shape = model.forward(normalize_uint8_batch(x_uint8)).mu.shape
    eps = torch.randn(mu_shape, generator=generator)

    enabled = make_fastpath_step_fn(
        model,
        corruptor,
        ssim_weight=0.1,
        autocast_dtype=torch.bfloat16,
        autocast_enabled=True,
    )(x_uint8, eps, torch.tensor(1.0))
    disabled = make_fastpath_step_fn(
        model,
        corruptor,
        ssim_weight=0.1,
        autocast_dtype=torch.bfloat16,
        autocast_enabled=False,
    )(x_uint8, eps, torch.tensor(1.0))

    assert enabled.reconstruction.dtype == torch.bfloat16
    assert disabled.reconstruction.dtype == torch.float32
    assert disabled.loss.dtype == torch.float32
