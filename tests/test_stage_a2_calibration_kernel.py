# Copyright 2026 HiperMaximus
# pyright: reportAny=false
# pyright: reportPrivateUsage=false
# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
"""Focused regression tests for the Stage A2 calibration rerun."""

from __future__ import annotations

import torch
from experiments import spec0053_stage_a2_calibration as calibration

from eqvae.evaluation.functional_geometry_calibration import (
    affine_chart_latents,
    decoder_path_edge_energy,
    decoder_path_energy,
)


def test_chunked_energy_and_gradient_match_the_monolithic_objective() -> None:
    """Decoder batching changes memory use, not path energy or its gradient."""

    class Decoder:
        @staticmethod
        def decode(latent: torch.Tensor) -> torch.Tensor:
            return torch.stack((latent[:, 0].square(), latent[:, 1].sin()), dim=1)

    left = torch.tensor([[0.0, 0.0]])
    right = torch.tensor([[1.0, 0.7]])
    basis = torch.eye(2)
    secant_norm = torch.linalg.vector_norm(right - left)
    times = torch.linspace(0.0, 1.0, 9)[1:-1, None]
    endpoint = (right - left) / secant_norm
    initial = times * endpoint

    chunk_coordinates = initial.clone().requires_grad_()
    chunked = calibration._chunked_path_energy(  # noqa: SLF001
        Decoder(),
        left,
        right,
        basis=basis,
        interior_coordinates=chunk_coordinates,
        secant_norm=secant_norm,
        total_segments=8,
        chunk_segments=3,
        affine_chart_latents=affine_chart_latents,
        decoder_path_edge_energy=decoder_path_edge_energy,
        backward=True,
        torch=torch,
    )
    assert chunk_coordinates.grad is not None
    chunk_gradient = chunk_coordinates.grad.detach().clone()

    full_coordinates = initial.clone().requires_grad_()
    interior = affine_chart_latents(
        left,
        basis,
        full_coordinates,
        secant_norm=secant_norm,
    )
    decoded = Decoder.decode(torch.cat((left, interior, right), dim=0))
    monolithic = decoder_path_energy(decoded)
    monolithic.backward()

    torch.testing.assert_close(torch.tensor(chunked), monolithic.detach())
    assert full_coordinates.grad is not None
    torch.testing.assert_close(chunk_gradient, full_coordinates.grad)
