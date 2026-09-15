# Copyright 2026 HiperMaximus
# ruff: noqa: DOC201, EM101, PLR0914, PLR2004, TRY003
# pyright: reportAny=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
"""Small numerical helpers for the Stage A2 geometry calibration probe."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from eqvae.evaluation.functional_geometry_slq import decoder_metric_matvec

if TYPE_CHECKING:
    from eqvae.evaluation.functional_geometry_rla import LinearizedDecoder


@dataclass(frozen=True)
class ThinMetricSpectrum:
    """Singular spectrum of the decoder differential on one fixed chart."""

    dimension: int
    singular_values_descending: tuple[float, ...]
    minimum_to_maximum_ratio: float
    condition_number: float | None


def decoder_visible_chart(
    operator: LinearizedDecoder,
    secant: Tensor,
    *,
    maximum_dimension: int,
    seed: int,
    microbatch: int,
) -> Tensor:
    """Build a nested orthonormal chart from a secant and visible probes.

    Raises:
        ValueError: If shapes, dimensions, or numerical ranks are invalid.

    """
    if secant.ndim < 2 or secant.shape[0] != 1:
        raise ValueError("secant must contain exactly one latent row")
    latent_dimension = math.prod(secant.shape[1:])
    if not 1 < maximum_dimension <= latent_dimension:
        raise ValueError("maximum_dimension must lie in (1, latent_dimension]")
    flat_secant = secant.detach().flatten(1)
    secant_norm = torch.linalg.vector_norm(flat_secant)
    if not bool(torch.isfinite(secant_norm)) or float(secant_norm) <= 0.0:
        raise ValueError("secant must have finite positive norm")

    generator = torch.Generator(device=secant.device).manual_seed(seed)
    signs = torch.randint(
        0,
        2,
        (maximum_dimension - 1, *secant.shape[1:]),
        generator=generator,
        device=secant.device,
        dtype=torch.int8,
    )
    probes = (signs.to(dtype=secant.dtype) * 2 - 1) / math.sqrt(latent_dimension)
    visible = decoder_metric_matvec(operator, probes, microbatch=microbatch)
    visible_flat = visible.flatten(1)
    visible_norms = torch.linalg.vector_norm(visible_flat, dim=1)
    if not bool(torch.isfinite(visible_norms).all()) or bool(
        (visible_norms <= 0.0).any(),
    ):
        raise ValueError("decoder-visible probes must have finite positive norm")
    normalized_visible = visible_flat / visible_norms[:, None]
    candidates = torch.cat(
        (flat_secant / secant_norm, normalized_visible),
        dim=0,
    )
    frame, triangular = torch.linalg.qr(candidates.T, mode="reduced")
    diagonal = torch.diagonal(triangular)
    relative_diagonal = diagonal.abs() / diagonal.abs().max().clamp_min(1e-12)
    if bool((relative_diagonal < 1e-6).any()):
        raise ValueError("candidate chart is numerically rank deficient")
    signs_for_frame = torch.where(
        diagonal >= 0,
        torch.ones_like(diagonal),
        -torch.ones_like(diagonal),
    )
    frame *= signs_for_frame[None, :]
    return frame.T.reshape(maximum_dimension, *secant.shape[1:]).detach()


def thin_metric_spectra(
    operator: LinearizedDecoder,
    basis: Tensor,
    *,
    dimensions: tuple[int, ...],
    microbatch: int,
) -> tuple[ThinMetricSpectrum, ...]:
    """Measure exact small pullback metrics for nested chart dimensions.

    Raises:
        ValueError: If the basis cannot provide every requested dimension.

    """
    if basis.ndim < 2 or basis.shape[0] < max(dimensions, default=0):
        raise ValueError("basis does not contain every requested dimension")
    if not dimensions or any(dimension <= 0 for dimension in dimensions):
        raise ValueError("dimensions must be positive")
    tangents = [
        operator.jvp_batch(basis[start : start + microbatch])
        for start in range(0, basis.shape[0], microbatch)
    ]
    flat = torch.cat(tangents, dim=0).flatten(1)
    scaled_transpose = flat.T.to(torch.float64) / math.sqrt(flat.shape[1])
    _, triangular = torch.linalg.qr(scaled_transpose, mode="reduced")
    results = []
    for dimension in dimensions:
        singular_values = torch.linalg.svdvals(triangular[:dimension, :dimension])
        largest = float(singular_values[0])
        smallest = float(singular_values[-1])
        ratio = smallest / max(largest, 1e-30)
        results.append(
            ThinMetricSpectrum(
                dimension=dimension,
                singular_values_descending=tuple(
                    float(value) for value in singular_values.tolist()
                ),
                minimum_to_maximum_ratio=ratio,
                condition_number=(1.0 / ratio if ratio > 0.0 else None),
            ),
        )
    return tuple(results)


def decoder_path_energy(decoded_knots: Tensor) -> Tensor:
    """Return discretized decoder energy under the per-scalar RMS metric.

    Raises:
        ValueError: If fewer than two decoded knots are supplied.

    """
    if decoded_knots.ndim < 2 or decoded_knots.shape[0] < 2:
        raise ValueError("decoded path requires at least two knots")
    return decoder_path_edge_energy(
        decoded_knots,
        total_path_segments=decoded_knots.shape[0] - 1,
    )


def decoder_path_edge_energy(
    decoded_knots: Tensor,
    *,
    total_path_segments: int,
) -> Tensor:
    """Return one contiguous edge block's contribution to full path energy.

    Raises:
        ValueError: If the block or full-path segment count is invalid.

    """
    if decoded_knots.ndim < 2 or decoded_knots.shape[0] < 2:
        raise ValueError("decoded edge block requires at least two knots")
    if total_path_segments < decoded_knots.shape[0] - 1:
        raise ValueError("total_path_segments cannot be smaller than the edge block")
    differences = decoded_knots[1:] - decoded_knots[:-1]
    segment_energy = differences.flatten(1).square().mean(dim=1)
    return total_path_segments * segment_energy.sum()


def affine_chart_latents(
    left: Tensor,
    basis: Tensor,
    normalized_coordinates: Tensor,
    *,
    secant_norm: Tensor,
) -> Tensor:
    """Map dimensionless affine-chart coordinates back to latent rows.

    Raises:
        ValueError: If latent, basis, and coordinate shapes are incompatible.

    """
    if left.shape[0] != 1 or normalized_coordinates.ndim != 2:
        raise ValueError("left and normalized_coordinates have invalid shapes")
    if basis.shape[0] != normalized_coordinates.shape[1]:
        raise ValueError("basis and coordinate dimensions differ")
    offsets = normalized_coordinates @ basis.flatten(1)
    return left + secant_norm * offsets.reshape(
        normalized_coordinates.shape[0],
        *left.shape[1:],
    )


def project_line_deviation_(
    coordinates: Tensor,
    line_coordinates: Tensor,
    *,
    maximum_deviation: float,
) -> None:
    """Project mutable interior coordinates into a line-centered trust tube.

    Raises:
        ValueError: If shapes differ or the trust radius is not positive.

    """
    if coordinates.shape != line_coordinates.shape:
        raise ValueError("coordinates and line coordinates must have equal shape")
    if maximum_deviation <= 0.0:
        raise ValueError("maximum_deviation must be positive")
    with torch.no_grad():
        deviation = coordinates - line_coordinates
        norms = torch.linalg.vector_norm(deviation, dim=1, keepdim=True)
        scales = torch.clamp(maximum_deviation / norms.clamp_min(1e-12), max=1.0)
        coordinates.copy_(line_coordinates + deviation * scales)
