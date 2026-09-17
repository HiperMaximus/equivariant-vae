# Copyright 2026 HiperMaximus
"""Focused synthetic seams for the fixed Stage A2 chart and IVP numerics."""

import math

import torch

from eqvae.evaluation.functional_geometry_stage_a2 import (
    construct_path_chart,
    estimate_initial_velocity,
    shooting_consistency_rollout,
    transport_rollout,
)


class _LinearDecoder(torch.nn.Module):
    matrix: torch.Tensor

    def __init__(self, matrix: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("matrix", matrix)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return latent @ self.matrix.T


def test_path_chart_keeps_visible_and_inactive_path_directions_separate() -> None:
    """A decoder-null direction in the observed path span is retained as inactive."""
    decoder = _LinearDecoder(torch.tensor(((3.0, 0.0, 0.0), (0.0, 2.0, 0.0))))
    time = torch.linspace(0.0, 1.0, 33)
    path = torch.stack((time, time.square(), time.pow(3)), dim=1)

    chart = construct_path_chart(decoder, path)

    assert chart["path_rank"] == 3
    assert chart["rank"] == 2
    visible = chart["U0"]
    inactive = chart["inactive"]
    knot_singular_values = chart["knot_singular_values"]
    knot_rank_valid = chart["knot_rank_valid"]
    assert isinstance(visible, torch.Tensor)
    assert isinstance(inactive, torch.Tensor)
    assert isinstance(knot_singular_values, torch.Tensor)
    assert isinstance(knot_rank_valid, torch.Tensor)
    assert visible.shape == (2, 3)
    assert inactive.shape == (1, 3)
    assert knot_singular_values.shape == (33, 2)
    assert bool(knot_rank_valid.all())
    torch.testing.assert_close(decoder(inactive), torch.zeros((1, 2)), atol=2e-6, rtol=0)
    assert torch.linalg.vector_norm(decoder(visible), dim=1).min() > 0.1
    for row in torch.cat((visible, inactive)):
        assert row[row.abs().argmax()] > 0

    fully_visible = construct_path_chart(_LinearDecoder(torch.eye(3)), path)
    fully_visible_basis = fully_visible["U0"]
    fully_visible_inactive = fully_visible["inactive"]
    assert isinstance(fully_visible_basis, torch.Tensor)
    assert isinstance(fully_visible_inactive, torch.Tensor)
    assert fully_visible_basis.shape == (3, 3)
    assert fully_visible_inactive.shape == (0, 3)


def test_linear_shooting_and_transport_are_exact() -> None:
    """A constant thin Jacobian gives a straight IVP and identity transport."""
    decoder = _LinearDecoder(torch.diag(torch.tensor((2.0, 3.0))))
    z0 = torch.zeros(2)
    U0 = torch.eye(2)
    c0 = torch.tensor((1.0, -0.5))
    target = z0 + c0
    aggregate_sigma_max = torch.tensor(3.0 / math.sqrt(2.0))

    rollout = shooting_consistency_rollout(
        decoder,
        z0,
        U0,
        c0,
        aggregate_sigma_max,
        target=target,
    )
    primary = rollout["primary"]
    refined = rollout["refined"]
    assert isinstance(primary, dict)
    assert isinstance(refined, dict)
    assert primary["intrinsic_status"] == "defined"
    assert refined["intrinsic_status"] == "defined"
    primary_latents = primary["latents"]
    assert isinstance(primary_latents, torch.Tensor)
    torch.testing.assert_close(primary_latents[8], target)
    torch.testing.assert_close(primary_latents[-1], 4.0 * c0)
    residual = primary["target_residual"]
    assert isinstance(residual, torch.Tensor)
    torch.testing.assert_close(residual, torch.zeros(()))

    transport = transport_rollout(
        decoder,
        primary_latents,
        U0,
        aggregate_sigma_max,
        c0,
    )
    assert transport["intrinsic_status"] == "defined"
    returned_tangent = transport["return_tangent"]
    returned_frame = transport["return_frame"]
    assert isinstance(returned_tangent, torch.Tensor)
    assert isinstance(returned_frame, torch.Tensor)
    frames = transport["frames"]
    singular_values = transport["singular_values"]
    assert isinstance(frames, torch.Tensor)
    assert isinstance(singular_values, torch.Tensor)
    assert singular_values.shape == (33, 2)
    initial_frame = frames[0]
    initial_image_frame = decoder(U0.T @ initial_frame) / math.sqrt(2.0)
    torch.testing.assert_close(initial_image_frame.T @ initial_image_frame, torch.eye(2))
    torch.testing.assert_close(returned_tangent, c0)
    torch.testing.assert_close(returned_frame, initial_frame)


def test_second_directional_term_changes_the_nonlinear_rollout() -> None:
    """For D(z)=z², positive initial motion must decelerate under the IVP."""

    class SquareDecoder(torch.nn.Module):
        def forward(self, latent: torch.Tensor) -> torch.Tensor:
            return latent.square()

    rollout = shooting_consistency_rollout(
        SquareDecoder(),
        torch.tensor((1.0,)),
        torch.tensor(((1.0,),)),
        torch.tensor((1.0,)),
        torch.tensor(2.0),
    )
    primary = rollout["primary"]
    assert isinstance(primary, dict)
    velocities = primary["velocities"]
    latents = primary["latents"]
    assert isinstance(velocities, torch.Tensor)
    assert isinstance(latents, torch.Tensor)
    assert velocities[1, 0] < 1.0
    assert latents[1, 0] < 1.0 + 1.0 / 8.0


def test_forward_difference_recovers_a_quadratic_initial_velocity() -> None:
    """The specified three-knot formula is exact for a quadratic chart path."""
    time = torch.arange(33, dtype=torch.float32) / 32.0
    path = (time + time.square()).unsqueeze(1)

    velocity = estimate_initial_velocity(path, torch.ones((1, 1)))

    torch.testing.assert_close(velocity, torch.ones(1))
