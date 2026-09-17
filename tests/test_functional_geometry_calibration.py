# Copyright 2026 HiperMaximus
"""Focused numerical tests for the Stage A2 calibration helpers."""

import math

import pytest
import torch

from eqvae.evaluation.functional_geometry_calibration import (
    affine_chart_latents,
    decoder_path_edge_energy,
    decoder_path_energy,
    decoder_visible_chart,
    thin_metric_spectra,
)
from eqvae.evaluation.functional_geometry_rla import linearize_decoder


def test_decoder_visible_chart_is_orthonormal_and_contains_secant() -> None:
    """Invariant: the chart contains its secant; tolerance is numerical."""

    def decoder(latent: torch.Tensor) -> torch.Tensor:
        scales = torch.tensor([3.0, 2.0, 1.0], dtype=latent.dtype)
        return latent * scales

    midpoint = torch.zeros(1, 3)
    secant = torch.tensor([[1.0, -2.0, 0.5]])
    operator = linearize_decoder(decoder, midpoint)
    basis = decoder_visible_chart(
        operator,
        secant,
        maximum_dimension=3,
        seed=530201,
        microbatch=2,
    )

    assert torch.allclose(basis @ basis.T, torch.eye(3), atol=1e-6, rtol=1e-6)
    projected = (basis @ secant.T).T @ basis
    assert torch.allclose(projected, secant, atol=1e-6, rtol=1e-6)


def test_thin_metric_spectrum_matches_known_linear_decoder() -> None:
    """Invariant: thin singular values use output-mean metric; values are derived."""

    def decoder(latent: torch.Tensor) -> torch.Tensor:
        scales = torch.tensor([4.0, 2.0, 1.0], dtype=latent.dtype)
        return latent * scales

    operator = linearize_decoder(
        decoder,
        torch.zeros(1, 3),
    )
    spectra = thin_metric_spectra(
        operator,
        torch.eye(3),
        dimensions=(2, 3),
        microbatch=2,
    )

    assert spectra[0].singular_values_descending == pytest.approx(
        (
            4.0 / math.sqrt(3.0),
            2.0 / math.sqrt(3.0),
        )
    )
    assert spectra[0].condition_number == pytest.approx(2.0)
    assert spectra[1].minimum_to_maximum_ratio == pytest.approx(0.25)


def test_thin_metric_preserves_near_threshold_rotated_singular_ratio() -> None:
    """Ensure thin SVD does not square the conditioning policy input."""
    angle = 0.37
    rotation = torch.tensor(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    scales = torch.diag(torch.tensor([1.0, 0.003, 0.0003]))
    matrix = rotation @ scales @ rotation.T

    def decoder(latent: torch.Tensor) -> torch.Tensor:
        return latent @ matrix.T

    operator = linearize_decoder(
        decoder,
        torch.zeros(1, 3),
    )
    spectrum = thin_metric_spectra(
        operator,
        torch.eye(3),
        dimensions=(3,),
        microbatch=2,
    )[0]

    assert spectrum.minimum_to_maximum_ratio == pytest.approx(0.0003, rel=2e-5)


def test_straight_linear_path_has_chord_energy() -> None:
    """Invariant: linear constant-speed energy equals chord energy; this is derived."""
    left = torch.tensor([[1.0, 2.0]])
    basis = torch.eye(2)
    secant = torch.tensor([[3.0, 4.0]])
    secant_norm = torch.linalg.vector_norm(secant)
    endpoint = (basis @ secant.T).flatten() / secant_norm
    times = torch.linspace(0.0, 1.0, 9)
    coordinates = times[:, None] * endpoint[None, :]
    knots = affine_chart_latents(
        left,
        basis,
        coordinates,
        secant_norm=secant_norm,
    )

    assert torch.allclose(knots[-1:], left + secant)
    assert torch.allclose(decoder_path_energy(knots), secant.square().mean())


def test_edge_blocks_sum_to_the_complete_path_energy() -> None:
    """Invariant: chunking edges preserves energy; K=8/chunk=3 are probe policy."""
    decoded = torch.arange(18, dtype=torch.float32).reshape(9, 2).square() / 17
    complete = decoder_path_energy(decoded)
    chunked = torch.stack(
        tuple(
            decoder_path_edge_energy(
                decoded[start : min(start + 3, 8) + 1],
                total_path_segments=8,
            )
            for start in range(0, 8, 3)
        ),
    ).sum()

    assert torch.allclose(chunked, complete)
