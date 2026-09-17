# Copyright 2026 HiperMaximus
"""Focused regression tests for reusable Stage A2 path mechanics."""

import math

import pytest
import torch
from experiments import spec0053_stage_a2_calibration as stage_a2


def test_chunked_energy_and_gradient_match_the_monolithic_objective() -> None:
    """Decoder batching changes memory use, not path energy or its gradient."""

    class Decoder:
        @staticmethod
        def decode(latent: torch.Tensor) -> torch.Tensor:
            return torch.stack((latent[:, 0].square(), latent[:, 1].sin()), dim=1)

    left = torch.tensor([[0.0, 0.0]])
    right = torch.tensor([[1.0, 0.7]])
    initial = torch.linspace(0.0, 1.0, 9)[1:-1, None] * (right - left)
    chunk_interior = initial.clone().requires_grad_()

    def edge_energy(latents: torch.Tensor, segments: torch.Tensor) -> torch.Tensor:
        decoded = Decoder.decode(latents)
        differences = decoded[1:] - decoded[:-1]
        return segments * differences.flatten(1).square().mean(dim=1).sum()

    chunked = stage_a2._chunked_path_energy(
        left,
        chunk_interior,
        right,
        edge_energy=edge_energy,
        total_segments=8,
        chunk_segments=4,
        backward=True,
        torch=torch,
    )
    assert chunk_interior.grad is not None
    chunk_gradient = chunk_interior.grad.detach().clone()

    full_interior = initial.clone().requires_grad_()
    decoded = Decoder.decode(
        stage_a2._path_from_interior(left, full_interior, right, torch=torch)
    )
    differences = decoded[1:] - decoded[:-1]
    monolithic = 8 * differences.flatten(1).square().mean(dim=1).sum()
    monolithic.backward()

    assert chunked == pytest.approx(float(monolithic.detach()))
    assert full_interior.grad is not None
    torch.testing.assert_close(chunk_gradient, full_interior.grad)


def test_full_latent_learning_rate_is_scaled_to_its_dimension() -> None:
    """A full-latent Adam step must not grow with sqrt(number of pixels)."""

    dimension = 16_384
    assert stage_a2._full_latent_learning_rate(0.005, dimension, 32) == pytest.approx(
        0.005 * math.sqrt(32 / dimension)
    )


def test_full_latent_optimizer_returns_the_best_path_not_the_last_path() -> None:
    """Best-iterate retention must preserve coordinates as well as energy."""

    def edge_energy(latents: torch.Tensor, segments: torch.Tensor) -> torch.Tensor:
        decoded = latents.square()
        differences = decoded[1:] - decoded[:-1]
        return segments * differences.flatten(1).square().mean(dim=1).sum()

    result = stage_a2._optimize_full_latent_path(
        torch.tensor([[0.0]]),
        torch.tensor([[1.0]]),
        learning_rate=10.0,
        learning_rate_reference_dimension=1,
        path_segments=2,
        optimizer_steps=1,
        chunk_segments=1,
        eager_edge_energy=edge_energy,
        optimized_edge_energy=edge_energy,
        torch=torch,
    )

    assert result["best_iteration"] == 0
    assert result["best_energy"] < result["final_energy"]
    torch.testing.assert_close(
        result["best_path"],
        torch.tensor([[0.0], [0.5], [1.0]]),
    )
