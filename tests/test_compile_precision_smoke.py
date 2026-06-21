# Copyright 2026 HiperMaximus
"""Local compile and precision smoke checks for the model/loss slice."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

import pytest
import torch

from eqvae.models.activations import GatedScalarActivation
from eqvae.models.non_equivariant_vae import (
    LATENT_CHANNELS,
    NonEquivariantVAE,
    VaeForwardOutput,
    build_non_equivariant_vae,
)

if TYPE_CHECKING:
    from collections.abc import Callable

SMOKE_IMAGE_SIZE = 64


class CompiledVae(Protocol):
    """Typed call protocol for `torch.compile(model)` in tests."""

    def __call__(
        self,
        inputs: torch.Tensor,
        *,
        eps: torch.Tensor | None = None,
    ) -> VaeForwardOutput:
        """Run the compiled VAE."""
        ...


def test_torch_compile_eager_preserves_forward_contract() -> None:
    """The local compile smoke keeps the explicit-eps output contract."""
    model = build_non_equivariant_vae()
    compiled = _compile_eager(model)
    clean_batch = torch.zeros((1, 3, SMOKE_IMAGE_SIZE, SMOKE_IMAGE_SIZE))
    eps = torch.zeros((
        1,
        LATENT_CHANNELS,
        SMOKE_IMAGE_SIZE // 8,
        SMOKE_IMAGE_SIZE // 8,
    ))

    try:
        output = compiled(clean_batch, eps=eps)
    except RuntimeError as exc:
        pytest.skip(f"local torch.compile eager smoke unsupported: {exc}")

    assert output.reconstruction.shape == clean_batch.shape
    assert torch.equal(output.eps, eps)
    assert torch.isfinite(output.reconstruction).all()


def test_cpu_float16_autocast_forward_is_finite_when_supported() -> None:
    """Try the local CPU FP16 path without requiring it as runtime evidence."""
    model = build_non_equivariant_vae()
    clean_batch = torch.zeros((1, 3, SMOKE_IMAGE_SIZE, SMOKE_IMAGE_SIZE))
    eps = torch.zeros((
        1,
        LATENT_CHANNELS,
        SMOKE_IMAGE_SIZE // 8,
        SMOKE_IMAGE_SIZE // 8,
    ))

    try:
        with torch.autocast(device_type="cpu", dtype=torch.float16):
            output: VaeForwardOutput = model.forward(clean_batch, eps=eps)
    except RuntimeError as exc:
        pytest.skip(f"local CPU float16 autocast unsupported: {exc}")

    assert output.reconstruction.shape == clean_batch.shape
    assert torch.equal(output.eps, eps)
    assert torch.isfinite(output.reconstruction).all()


def test_gated_scalar_activation_fp16_matches_fp32_reference() -> None:
    """The learned gate accepts FP16 inputs while computing the sigmoid in FP32."""
    activation = GatedScalarActivation(channels=2)
    inputs = torch.linspace(-2.0, 2.0, steps=16, dtype=torch.float16).reshape(
        1,
        2,
        2,
        4,
    )

    outputs: torch.Tensor = activation.forward(inputs)
    reference = inputs.to(dtype=torch.float32) * torch.sigmoid(
        inputs.to(dtype=torch.float32),
    )

    assert outputs.dtype == torch.float16
    assert torch.isfinite(outputs).all()
    assert torch.allclose(outputs.to(dtype=torch.float32), reference, atol=1.0e-3)


def test_gated_scalar_activation_relaxed_policy_uses_input_dtype() -> None:
    """The relaxed AMP policy can run scalar gate sigmoid math in FP16."""
    activation = GatedScalarActivation(channels=2, force_fp32=False)
    inputs = torch.linspace(-2.0, 2.0, steps=16, dtype=torch.float16).reshape(
        1,
        2,
        2,
        4,
    )

    outputs: torch.Tensor = activation.forward(inputs)
    reference = inputs * torch.sigmoid(inputs)

    assert activation.force_fp32 is False
    assert outputs.dtype == torch.float16
    assert torch.isfinite(outputs).all()
    assert torch.equal(outputs, reference)


def _compile_eager(model: NonEquivariantVAE) -> CompiledVae:
    compile_fn = cast("Callable[..., CompiledVae]", torch.compile)
    return compile_fn(model, backend="eager")
