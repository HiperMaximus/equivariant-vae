# Copyright 2026 HiperMaximus
"""Exact frozen-kernel check for repeated decoder geometry."""

import torch

from eqvae.models.so2_architecture_probe import (  # pyright: ignore[reportPrivateUsage]
    A_LAYOUT,
    _PROFILE_7,
    _ScalarToF01Conv,
)
from eqvae.models.so2_vae import build_so2_vae


def test_materialized_kernel_preserves_output_and_input_gradient() -> None:
    layer = _ScalarToF01Conv(2, A_LAYOUT, _PROFILE_7).eval().requires_grad_(False)
    inputs = torch.randn(2, 2, 8, 8, generator=torch.Generator().manual_seed(53))

    eager_inputs = inputs.clone().requires_grad_(True)
    eager_output = layer(eager_inputs)
    eager_gradient = torch.autograd.grad(eager_output.square().mean(), eager_inputs)[0]

    state_names = tuple(layer.state_dict())
    layer.materialize_frozen_kernel()
    cached_inputs = inputs.clone().requires_grad_(True)
    cached_output = layer(cached_inputs)
    cached_gradient = torch.autograd.grad(
        cached_output.square().mean(),
        cached_inputs,
    )[0]

    torch.testing.assert_close(cached_output, eager_output, rtol=0.0, atol=0.0)
    torch.testing.assert_close(cached_gradient, eager_gradient, rtol=0.0, atol=0.0)
    assert tuple(layer.state_dict()) == state_names


def test_full_decoder_materializes_only_its_21_convolutions() -> None:
    model = build_so2_vae().eval().requires_grad_(False)
    state_names = tuple(model.state_dict())

    assert model.materialize_frozen_decoder_kernels() == 21
    assert tuple(model.state_dict()) == state_names
