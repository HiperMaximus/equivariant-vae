# pyright: reportAny=false, reportArgumentType=false, reportCallIssue=false, reportUnnecessaryCast=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false
# Copyright 2026 HiperMaximus
# ruff: noqa: COM812, DOC201, DOC501, EM101, EM102, PLR0913, PLR0914, PLR2004, PLR6201, TRY003
"""Numerical contracts for the Spec 0050 frozen rotation-geometry audit."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, cast

import numpy as np
import torch
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Sequence

ArrayF32 = NDArray[np.float32]
ArrayF64 = NDArray[np.float64]
ArrayI64 = NDArray[np.int64]

EPSILON: Final = 1e-8
FIT_RANKS: Final = (23, 22, 1, 8, 4, 24, 9, 20, 11, 10, 0, 14, 5, 16, 7, 17, 12)
HELDOUT_RANKS: Final = (18, 2, 21, 19, 3, 13, 15, 6)
FIT_ANGLES: Final = tuple(range(0, 360, 5))
HELDOUT_ANGLES: Final = tuple(angle for angle in range(360) if angle % 5)


@dataclass(frozen=True)
class PcaOrbit:
    """Patch-local PCA scores and variance accounting."""

    scores: ArrayF64
    explained_fraction: ArrayF64
    eigenvalues: ArrayF64


@dataclass(frozen=True)
class SharedBasis:
    """Fit-only shared rotation basis and centering values."""

    basis: ArrayF64
    global_center: ArrayF64
    patch_centers: ArrayF64
    singular_values: ArrayF64


@dataclass(frozen=True)
class GeneratorFit:
    """A skew generator with design-identifiability diagnostics."""

    matrix: ArrayF64
    design_singular_values: ArrayF64
    design_rank: int
    design_condition: float
    eigenfrequencies: ArrayF64


def masked_orbit_vectors(values: ArrayF32, mask: NDArray[np.bool_]) -> ArrayF64:
    """Flatten ``N x T x C x H x W`` orbits over one spatial mask."""
    if values.ndim != 5:
        raise ValueError(f"expected NxTxCxHxW, got {values.shape}")
    if tuple(mask.shape) != tuple(values.shape[-2:]):
        raise ValueError(f"mask {mask.shape} does not match {values.shape[-2:]}")
    selected = values[:, :, :, mask]
    return cast(
        "ArrayF64",
        selected.reshape(values.shape[0], values.shape[1], -1).astype(np.float64),
    )


def cyclic_geometry(values: ArrayF64, *, step_degrees: int = 1) -> dict[str, object]:
    """Compute patch-first cyclic finite-difference geometry."""
    if values.ndim != 3 or values.shape[1] != 360:
        raise ValueError(f"expected Nx360xD orbit vectors, got {values.shape}")
    if step_degrees not in (1, 2, 5):
        raise ValueError("step_degrees must be 1, 2, or 5")
    sampled = values[:, ::step_degrees]
    delta_radians = math.radians(step_degrees)
    first = np.roll(sampled, -1, axis=1) - sampled
    second = np.roll(first, -1, axis=1) - first
    step_rms = np.sqrt(np.mean(np.square(first), axis=2))
    second_rms = np.sqrt(np.mean(np.square(second), axis=2))
    tangent = (np.roll(sampled, -1, axis=1) - np.roll(sampled, 1, axis=1)) / (
        2.0 * delta_radians
    )
    acceleration = (
        np.roll(sampled, -1, axis=1) - 2.0 * sampled + np.roll(sampled, 1, axis=1)
    ) / delta_radians**2
    tangent_norm = np.sqrt(np.mean(np.square(tangent), axis=2))
    acceleration_norm = np.sqrt(np.mean(np.square(acceleration), axis=2))
    tangent_dot = np.mean(tangent * acceleration, axis=2)
    curvature_numerator = np.maximum(
        acceleration_norm**2 - np.square(tangent_dot / (tangent_norm + EPSILON)),
        0.0,
    )
    curvature = np.sqrt(curvature_numerator) / (np.square(tangent_norm) + EPSILON)
    tangent_next = np.roll(tangent, -1, axis=1)
    cosine = np.sum(tangent * tangent_next, axis=2) / (
        np.linalg.norm(tangent, axis=2) * np.linalg.norm(tangent_next, axis=2) + EPSILON
    )
    turning = np.arccos(np.clip(cosine, -1.0, 1.0))
    mean_step = np.mean(step_rms, axis=1)
    rows: dict[str, object] = {
        "step_degrees": step_degrees,
        "step_rms": step_rms.tolist(),
        "second_rms": second_rms.tolist(),
        "local_linearity_ratio": (
            np.sqrt(np.mean(np.square(second), axis=(1, 2)))
            / (np.sqrt(np.mean(np.square(first), axis=(1, 2))) + EPSILON)
        ).tolist(),
        "step_size_cv": (np.std(step_rms, axis=1) / (mean_step + EPSILON)).tolist(),
        "path_length": np.sum(step_rms, axis=1).tolist(),
        "tangent_norm": tangent_norm.tolist(),
        "acceleration_norm": acceleration_norm.tolist(),
        "turning_angle_radians": turning.tolist(),
        "curvature": curvature.tolist(),
        "tangent_norm_median": np.median(tangent_norm, axis=1).tolist(),
        "turning_angle_median_radians": np.median(turning, axis=1).tolist(),
        "curvature_median": np.median(curvature, axis=1).tolist(),
        "curvature_q90": np.quantile(curvature, 0.9, axis=1).tolist(),
    }
    return rows


def exclude_cardinal_neighborhoods(
    values: ArrayF64,
    *,
    radius_degrees: int = 2,
    step_degrees: int = 1,
) -> dict[str, object]:
    """Score sampled differences whose complete supports avoid cardinals."""
    if values.ndim != 3 or values.shape[1] != 360:
        raise ValueError(f"expected Nx360xD, got {values.shape}")
    if step_degrees not in {1, 2, 5}:
        raise ValueError("step_degrees must be 1, 2, or 5")
    sampled = values[:, ::step_degrees]
    angles = np.arange(0, 360, step_degrees, dtype=np.int64)
    forbidden = np.zeros(angles.size, dtype=np.bool_)
    for cardinal in (0, 90, 180, 270):
        distance = np.abs((angles - cardinal + 180) % 360 - 180)
        forbidden |= distance <= radius_degrees
    keep_first = ~(forbidden | np.roll(forbidden, -1))
    keep_second = ~(forbidden | np.roll(forbidden, -1) | np.roll(forbidden, -2))
    keep_centered = ~(forbidden | np.roll(forbidden, -1) | np.roll(forbidden, 1))
    keep_turning = ~(
        forbidden
        | np.roll(forbidden, 1)
        | np.roll(forbidden, -1)
        | np.roll(forbidden, -2)
    )
    first = np.roll(sampled, -1, axis=1) - sampled
    second = np.roll(first, -1, axis=1) - first
    delta_radians = math.radians(step_degrees)
    tangent = (np.roll(sampled, -1, axis=1) - np.roll(sampled, 1, axis=1)) / (
        2.0 * delta_radians
    )
    acceleration = (
        np.roll(sampled, -1, axis=1) - 2.0 * sampled + np.roll(sampled, 1, axis=1)
    ) / delta_radians**2
    tangent_norm = np.sqrt(np.mean(np.square(tangent), axis=2))
    acceleration_norm = np.sqrt(np.mean(np.square(acceleration), axis=2))
    tangent_dot = np.mean(tangent * acceleration, axis=2)
    curvature = np.sqrt(
        np.maximum(
            acceleration_norm**2 - np.square(tangent_dot / (tangent_norm + EPSILON)),
            0.0,
        )
    ) / (np.square(tangent_norm) + EPSILON)
    tangent_next = np.roll(tangent, -1, axis=1)
    turning = np.arccos(
        np.clip(
            np.sum(tangent * tangent_next, axis=2)
            / (
                np.linalg.norm(tangent, axis=2) * np.linalg.norm(tangent_next, axis=2)
                + EPSILON
            ),
            -1.0,
            1.0,
        )
    )
    first_kept = first[:, keep_first]
    first_second_support = first[:, keep_second]
    second_kept = second[:, keep_second]
    steps = np.sqrt(np.mean(np.square(first_kept), axis=2))
    return {
        "step_degrees": step_degrees,
        "kept_step_count": int(keep_first.sum()),
        "kept_second_count": int(keep_second.sum()),
        "kept_centered_count": int(keep_centered.sum()),
        "kept_turning_count": int(keep_turning.sum()),
        "local_linearity_ratio": (
            np.sqrt(np.mean(np.square(second_kept), axis=(1, 2)))
            / (np.sqrt(np.mean(np.square(first_second_support), axis=(1, 2))) + EPSILON)
        ).tolist(),
        "step_size_cv": (
            np.std(steps, axis=1) / (np.mean(steps, axis=1) + EPSILON)
        ).tolist(),
        "path_length": np.sum(steps, axis=1).tolist(),
        "step_rms": steps.tolist(),
        "second_rms": np.sqrt(np.mean(np.square(second_kept), axis=2)).tolist(),
        "tangent_norm": tangent_norm[:, keep_centered].tolist(),
        "acceleration_norm": acceleration_norm[:, keep_centered].tolist(),
        "turning_angle_radians": turning[:, keep_turning].tolist(),
        "curvature": curvature[:, keep_centered].tolist(),
        "tangent_norm_median": np.median(
            tangent_norm[:, keep_centered], axis=1
        ).tolist(),
        "turning_angle_median_radians": np.median(
            turning[:, keep_turning], axis=1
        ).tolist(),
        "curvature_median": np.median(curvature[:, keep_centered], axis=1).tolist(),
        "curvature_q90": np.quantile(curvature[:, keep_centered], 0.9, axis=1).tolist(),
    }


def local_pca(values: ArrayF64, *, components: int = 6) -> PcaOrbit:
    """Fit one PCA per orbit through the dual ``T x T`` Gram matrix."""
    if values.ndim != 2:
        raise ValueError(f"expected TxD, got {values.shape}")
    centered = values - np.mean(values, axis=0, keepdims=True)
    gram = centered @ centered.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    count = min(components, values.shape[0], values.shape[1])
    scores = eigenvectors[:, :count] * np.sqrt(eigenvalues[:count])[None, :]
    total = float(np.sum(eigenvalues))
    fractions = eigenvalues[:count] / (total + EPSILON)
    return PcaOrbit(
        scores=cast("ArrayF64", scores.astype(np.float64)),
        explained_fraction=cast("ArrayF64", fractions.astype(np.float64)),
        eigenvalues=cast("ArrayF64", eigenvalues[:count].astype(np.float64)),
    )


def harmonic_summary(
    scores: ArrayF64,
    *,
    raw_total_energy: float,
    maximum_harmonic: int = 6,
) -> dict[str, object]:
    """Return basis-invariant real Fourier power for one PCA score subspace."""
    if scores.ndim != 2 or scores.shape[0] != 360:
        raise ValueError(f"expected 360xK scores, got {scores.shape}")
    centered = scores - np.mean(scores, axis=0, keepdims=True)
    coefficients = np.fft.rfft(centered, axis=0) / scores.shape[0]
    power = np.square(np.abs(coefficients))
    power[1:-1] *= 2.0
    frequency_power = np.sum(power, axis=1)
    total = float(np.sum(frequency_power))
    probability = frequency_power / (total + EPSILON)
    positive = probability[1:]
    entropy = float(-np.sum(positive * np.log(positive + EPSILON)))
    m1_matrix = np.column_stack((coefficients[1].real, coefficients[1].imag))
    singular_values = np.linalg.svd(m1_matrix, compute_uv=False)
    displayed = frequency_power[: maximum_harmonic + 1]
    return {
        "harmonic_power_0_to_6": displayed.tolist(),
        "harmonic_fraction_conditional_pc6_0_to_6": (
            displayed / (total + EPSILON)
        ).tolist(),
        "harmonic_fraction_raw_0_to_6": (
            displayed / (raw_total_energy + EPSILON)
        ).tolist(),
        "low_harmonic_fraction_raw": float(
            np.sum(frequency_power[1 : maximum_harmonic + 1])
            / (raw_total_energy + EPSILON)
        ),
        "m1_fraction_raw": float(frequency_power[1] / (raw_total_energy + EPSILON)),
        "spectral_entropy": entropy,
        "effective_frequency_count": float(math.exp(entropy)),
        "m1_plane_singular_values": singular_values.tolist(),
    }


def tangent_summary(values: ArrayF64) -> dict[str, object]:
    """Summarize cyclic tangents and their full-orbit span for one patch."""
    if values.ndim != 2 or values.shape[0] != 360:
        raise ValueError(f"expected 360xD, got {values.shape}")
    delta = math.radians(1.0)
    tangent_rows = (np.roll(values, -1, axis=0) - np.roll(values, 1, axis=0)) / (
        2.0 * delta
    )
    tangent_matrix = tangent_rows.T
    tangent_gram = tangent_rows @ tangent_rows.T
    tangent_eigenvalues = np.maximum(np.linalg.eigvalsh(tangent_gram)[::-1], 0.0)
    singular_values = np.sqrt(tangent_eigenvalues)
    energy = tangent_eigenvalues
    cumulative = np.cumsum(energy) / (np.sum(energy) + EPSILON)
    window = tangent_rows[np.asarray([358, 359, 0, 1, 2])]
    window_singular = np.linalg.svd(window, compute_uv=False)
    return {
        "singular_values": singular_values[:12].tolist(),
        "span_dimension_90": int(np.searchsorted(cumulative, 0.90) + 1),
        "span_dimension_95": int(np.searchsorted(cumulative, 0.95) + 1),
        "span_dimension_99": int(np.searchsorted(cumulative, 0.99) + 1),
        "local_rank1_fraction": float(
            window_singular[0] ** 2 / (np.sum(np.square(window_singular)) + EPSILON)
        ),
        "top2_left_vectors": _top_left_vectors(tangent_matrix, count=2).tolist(),
    }


def _top_left_vectors(values: ArrayF64, *, count: int) -> ArrayF64:
    gram = values.T @ values
    eigenvalues, right = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1][:count]
    scales = np.sqrt(np.maximum(eigenvalues[order], EPSILON))
    left = values @ right[:, order] / scales[None, :]
    return cast("ArrayF64", left.astype(np.float64))


def principal_angles(left: ArrayF64, right: ArrayF64) -> ArrayF64:
    """Return principal angles between two column-orthonormal subspaces."""
    singular = np.linalg.svd(left.T @ right, compute_uv=False)
    return cast("ArrayF64", np.arccos(np.clip(singular, -1.0, 1.0)))


def fit_shared_basis(
    vectors: ArrayF64,
    *,
    fit_ranks: Sequence[int] = FIT_RANKS,
    fit_angles: Sequence[int] = FIT_ANGLES,
    maximum_dimension: int = 6,
    oversampling: int = 6,
    power_iterations: int = 2,
    seed: int = 4158411771,
) -> SharedBasis:
    """Fit the locked randomized shared residual basis using fit data only."""
    fit = vectors[np.asarray(fit_ranks, dtype=np.int64)][:, np.asarray(fit_angles)]
    patch_centers = np.mean(fit, axis=1)
    global_center = np.mean(fit, axis=(0, 1))
    residuals = (fit - patch_centers[:, None, :]).reshape(-1, vectors.shape[2])
    target = maximum_dimension + oversampling
    generator = np.random.default_rng(seed)
    omega = generator.standard_normal((residuals.shape[1], target))
    sketch = residuals @ omega
    for _ in range(power_iterations):
        sketch, _ = np.linalg.qr(sketch, mode="reduced")
        sketch = residuals @ (residuals.T @ sketch)
    q_matrix, _ = np.linalg.qr(sketch, mode="reduced")
    compact = q_matrix.T @ residuals
    _left, singular_values, right_t = np.linalg.svd(compact, full_matrices=False)
    basis = right_t[:maximum_dimension].T
    return SharedBasis(
        basis=cast("ArrayF64", basis.astype(np.float64)),
        global_center=cast("ArrayF64", global_center.astype(np.float64)),
        patch_centers=cast("ArrayF64", patch_centers.astype(np.float64)),
        singular_values=cast("ArrayF64", singular_values.astype(np.float64)),
    )


def fit_skew_generator(states: ArrayF64, derivatives: ArrayF64) -> GeneratorFit:
    """Solve the constrained least-squares problem over skew parameters."""
    if states.shape != derivatives.shape or states.ndim != 2:
        raise ValueError("states and derivatives must have matching NxD shapes")
    dimension = states.shape[1]
    pairs = [
        (row, column)
        for row in range(dimension)
        for column in range(row + 1, dimension)
    ]
    design = np.zeros((states.shape[0] * dimension, len(pairs)), dtype=np.float64)
    for parameter, (row, column) in enumerate(pairs):
        design[row::dimension, parameter] = states[:, column]
        design[column::dimension, parameter] = -states[:, row]
    target = derivatives.reshape(-1)
    if not pairs:
        matrix = np.zeros((dimension, dimension), dtype=np.float64)
        singular_values = np.empty(0, dtype=np.float64)
        rank = 0
        condition = 1.0
    else:
        parameters, _residuals, rank, singular_values = np.linalg.lstsq(
            design,
            target,
            rcond=None,
        )
        matrix = np.zeros((dimension, dimension), dtype=np.float64)
        for value, (row, column) in zip(parameters, pairs, strict=True):
            matrix[row, column] = value
            matrix[column, row] = -value
        condition = float(
            singular_values[0] / singular_values[-1]
            if singular_values.size and singular_values[-1] > EPSILON
            else math.inf
        )
    eigenvalues = np.linalg.eigvals(matrix)
    frequencies = np.sort(np.abs(eigenvalues.imag[eigenvalues.imag > EPSILON]))
    return GeneratorFit(
        matrix=cast("ArrayF64", matrix),
        design_singular_values=cast("ArrayF64", singular_values),
        design_rank=int(rank),
        design_condition=condition,
        eigenfrequencies=cast("ArrayF64", frequencies.astype(np.float64)),
    )


def fit_generator_from_orbits(
    vectors: ArrayF64,
    shared: SharedBasis,
    *,
    dimension: int,
    conditional: bool,
    fit_ranks: Sequence[int] = FIT_RANKS,
    shuffled_seed: int | None = None,
) -> GeneratorFit:
    """Fit the strict-global or patch-centered 5-degree skew generator."""
    ranks = np.asarray(fit_ranks, dtype=np.int64)
    angles = np.asarray(FIT_ANGLES, dtype=np.int64)
    selected = vectors[ranks][:, angles]
    basis = shared.basis[:, :dimension]
    if conditional:
        centers = np.mean(selected, axis=1, keepdims=True)
    else:
        centers = shared.global_center[None, None, :]
    states = (selected - centers) @ basis
    delta = math.radians(5.0)
    derivatives = (np.roll(states, -1, axis=1) - np.roll(states, 1, axis=1)) / (
        2.0 * delta
    )
    flat_states = states.reshape(-1, dimension)
    flat_derivatives = derivatives.reshape(-1, dimension)
    if shuffled_seed is not None:
        permutation = np.random.default_rng(shuffled_seed).permutation(
            flat_derivatives.shape[0]
        )
        flat_derivatives = flat_derivatives[permutation]
    return fit_skew_generator(flat_states, flat_derivatives)


def matrix_exponential(matrix: ArrayF64, radians: float) -> ArrayF64:
    """Evaluate a small matrix exponential deterministically in float64."""
    tensor = torch.from_numpy(matrix).to(dtype=torch.float64)
    result = torch.linalg.matrix_exp(tensor * radians).numpy()
    return cast("ArrayF64", result)


def fit_orthogonal_step(states: ArrayF64, next_states: ArrayF64) -> ArrayF64:
    """Fit ``next ~= Q @ state`` with an orthogonal Procrustes operator."""
    if states.shape != next_states.shape or states.ndim != 2:
        raise ValueError("states and next_states must have matching NxD shapes")
    cross = next_states.T @ states
    left, _singular, right_t = np.linalg.svd(cross, full_matrices=False)
    operator = left @ right_t
    if np.linalg.det(operator) < 0.0:
        left[:, -1] *= -1.0
        operator = left @ right_t
    return cast("ArrayF64", operator.astype(np.float64))


def operator_numerics(
    generator: ArrayF64,
    procrustes_step: ArrayF64,
    *,
    step_radians: float,
) -> dict[str, float]:
    """Cross-check exponential and independently fitted one-step operators."""
    dimension = generator.shape[0]
    if generator.shape != (dimension, dimension):
        raise ValueError("generator must be square")
    if procrustes_step.shape != generator.shape:
        raise ValueError("Procrustes operator must match generator")
    identity = np.eye(dimension, dtype=np.float64)
    exponential_step = matrix_exponential(generator, step_radians)
    angles = (17.0, 31.0, 73.0)
    composition_errors: list[float] = []
    for alpha in angles:
        for beta in angles:
            left = matrix_exponential(generator, math.radians(alpha))
            right = matrix_exponential(generator, math.radians(beta))
            joined = matrix_exponential(generator, math.radians(alpha + beta))
            composition_errors.append(float(np.linalg.norm(left @ right - joined)))
    return {
        "generator_skew_frobenius": float(np.linalg.norm(generator + generator.T)),
        "exponential_orthogonality_frobenius": float(
            np.linalg.norm(exponential_step.T @ exponential_step - identity)
        ),
        "exponential_group_composition_frobenius_max": max(composition_errors),
        "procrustes_orthogonality_frobenius": float(
            np.linalg.norm(procrustes_step.T @ procrustes_step - identity)
        ),
        "procrustes_vs_exponential_step_frobenius": float(
            np.linalg.norm(procrustes_step - exponential_step)
        ),
    }


def evaluate_strict_rollout(
    vectors: ArrayF64,
    shared: SharedBasis,
    fit: GeneratorFit,
    *,
    dimension: int,
    heldout_ranks: Sequence[int] = HELDOUT_RANKS,
) -> dict[str, object]:
    """Evaluate fit-only global action from each held-out patch's 0-degree view."""
    basis = shared.basis[:, :dimension]
    center = shared.global_center
    ranks = np.asarray(heldout_ranks, dtype=np.int64)
    angle_indices = np.asarray(HELDOUT_ANGLES, dtype=np.int64)
    nrmse: list[float] = []
    r2: list[float] = []
    projected_nrmse: list[float] = []
    projected_r2: list[float] = []
    horizons: list[list[float]] = []
    normalized_horizons: list[list[float]] = []
    observed_winding: list[list[float]] = []
    predicted_winding: list[list[float]] = []
    planes = _generator_invariant_planes(fit.matrix)
    for rank in ranks:
        truth = vectors[rank, angle_indices]
        initial = vectors[rank, 0] - center
        projected = basis.T @ initial
        orthogonal = initial - basis @ projected
        predictions = np.stack([
            center
            + basis
            @ (matrix_exponential(fit.matrix, math.radians(float(angle))) @ projected)
            + orthogonal
            for angle in angle_indices
        ])
        error = predictions - truth
        truth_centered = truth - np.mean(truth, axis=0, keepdims=True)
        sse = float(np.sum(np.square(error)))
        sst = float(np.sum(np.square(truth_centered)))
        nrmse.append(
            math.sqrt(sse / error.size) / (math.sqrt(sst / truth.size) + EPSILON)
        )
        r2.append(1.0 - sse / (sst + EPSILON))
        truth_projected = (truth - center) @ basis
        predictions_projected = (predictions - center) @ basis
        projected_error = predictions_projected - truth_projected
        projected_centered = truth_projected - np.mean(
            truth_projected,
            axis=0,
            keepdims=True,
        )
        projected_sse = float(np.sum(np.square(projected_error)))
        projected_sst = float(np.sum(np.square(projected_centered)))
        projected_nrmse.append(
            math.sqrt(projected_sse / projected_error.size)
            / (math.sqrt(projected_sst / truth_projected.size) + EPSILON)
        )
        projected_r2.append(1.0 - projected_sse / (projected_sst + EPSILON))
        observed_winding.append([
            _phase_winding(truth_projected @ plane) for plane in planes
        ])
        predicted_winding.append([
            _phase_winding(predictions_projected @ plane) for plane in planes
        ])
        per_angle = np.sqrt(np.mean(np.square(error), axis=1))
        horizons.append(per_angle.tolist())
        normalized_horizons.append(
            (per_angle / (math.sqrt(sst / truth.size) + EPSILON)).tolist()
        )
    return {
        "patch_ranks": ranks.tolist(),
        "nrmse": nrmse,
        "r2": r2,
        "median_nrmse": float(np.median(nrmse)),
        "median_r2": float(np.median(r2)),
        "projected_nrmse": projected_nrmse,
        "projected_r2": projected_r2,
        "median_projected_nrmse": float(np.median(projected_nrmse)),
        "median_projected_r2": float(np.median(projected_r2)),
        "observed_winding": observed_winding,
        "predicted_winding": predicted_winding,
        "error_by_angle": horizons,
        "normalized_error_by_angle": normalized_horizons,
    }


def _generator_invariant_planes(matrix: ArrayF64) -> list[ArrayF64]:
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    planes: list[ArrayF64] = []
    for index in np.flatnonzero(eigenvalues.imag > EPSILON):
        vector = eigenvectors[:, index]
        plane, _ = np.linalg.qr(np.column_stack((vector.real, -vector.imag)))
        if (plane[:, 1] @ matrix @ plane[:, 0]) < 0:
            plane[:, 1] *= -1.0
        planes.append(cast("ArrayF64", plane.astype(np.float64)))
    return planes


def _phase_winding(coordinates: ArrayF64) -> float:
    if coordinates.shape[1] != 2:
        raise ValueError("phase winding requires two-dimensional coordinates")
    phase = np.unwrap(np.arctan2(coordinates[:, 1], coordinates[:, 0]))
    return float((phase[-1] - phase[0]) / (2.0 * math.pi))


def canonicalization_statistics(
    canonical: ArrayF64,
    identity: ArrayF64,
    *,
    anchor_angles: Sequence[int] = FIT_ANGLES,
    test_angles: Sequence[int] = HELDOUT_ANGLES,
) -> dict[str, object]:
    """Compute exact W/B, retrieval and distance stress for patch orbits."""
    if canonical.shape != identity.shape or canonical.ndim != 3:
        raise ValueError("canonical and identity must have matching Nx360xD shapes")

    def variance_parts(values: ArrayF64) -> tuple[float, float, ArrayF64]:
        patch_centers = np.mean(values, axis=1)
        global_center = np.mean(patch_centers, axis=0)
        within = float(np.mean(np.square(values - patch_centers[:, None, :])))
        between = float(np.mean(np.square(patch_centers - global_center[None, :])))
        return within, between, patch_centers

    within, between, _centers = variance_parts(canonical)
    identity_within, identity_between, _identity_centers = variance_parts(identity)
    patch_within = np.mean(
        np.square(canonical - np.mean(canonical, axis=1, keepdims=True)),
        axis=(1, 2),
    )
    identity_patch_within = np.mean(
        np.square(identity - np.mean(identity, axis=1, keepdims=True)),
        axis=(1, 2),
    )
    patch_within_ratio = patch_within / (identity_patch_within + EPSILON)
    anchors = np.asarray(anchor_angles, dtype=np.int64)
    tests = np.asarray(test_angles, dtype=np.int64)
    prototypes = np.mean(canonical[:, anchors], axis=1)
    retrieval: list[float] = []
    for patch_index in range(canonical.shape[0]):
        distances = np.sum(
            np.square(canonical[patch_index, tests, None, :] - prototypes[None, :, :]),
            axis=2,
        )
        retrieval.append(float(np.mean(np.argmin(distances, axis=1) == patch_index)))
    reference = identity[:, 0]
    reference_distances = _pairwise_distances(reference)
    denominator = float(np.sum(np.square(reference_distances)))
    stresses: list[float] = []
    if denominator > EPSILON:
        for angle in tests:
            observed = _pairwise_distances(canonical[:, angle])
            stresses.append(
                math.sqrt(
                    float(np.sum(np.square(observed - reference_distances)))
                    / denominator
                )
            )
    return {
        "w": within,
        "b": between,
        "f": within / (between + EPSILON),
        "identity_w": identity_within,
        "identity_b": identity_between,
        "identity_f": identity_within / (identity_between + EPSILON),
        "w_ratio": within / (identity_within + EPSILON),
        "patch_w": patch_within.tolist(),
        "identity_patch_w": identity_patch_within.tolist(),
        "patch_w_ratio": patch_within_ratio.tolist(),
        "patch_w_ratio_median": float(np.median(patch_within_ratio)),
        "b_ratio": between / (identity_between + EPSILON),
        "retrieval_per_patch": retrieval,
        "retrieval_mean": float(np.mean(retrieval)),
        "distance_stress_by_angle": stresses,
        "distance_stress_median": float(np.median(stresses)) if stresses else None,
        "distance_stress_defined": bool(stresses),
    }


def _pairwise_distances(values: ArrayF64) -> ArrayF64:
    pairs = [
        float(np.linalg.norm(values[left] - values[right]))
        for left in range(values.shape[0])
        for right in range(left + 1, values.shape[0])
    ]
    return np.asarray(pairs, dtype=np.float64)


def fit_angle_probe(features: ArrayF64, angles_degrees: Sequence[int]) -> ArrayF64:
    """Fit a linear two-output circular probe with an intercept."""
    if features.ndim != 2 or features.shape[0] != len(angles_degrees):
        raise ValueError("angle probe feature and angle rows differ")
    radians = np.deg2rad(np.asarray(angles_degrees, dtype=np.float64))
    targets = np.column_stack((np.cos(radians), np.sin(radians)))
    design = np.column_stack((features, np.ones(features.shape[0], dtype=np.float64)))
    weights, _residuals, _rank, _singular = np.linalg.lstsq(design, targets, rcond=None)
    return cast("ArrayF64", weights.astype(np.float64))


def score_angle_probe(
    features: ArrayF64,
    angles_degrees: Sequence[int],
    weights: ArrayF64,
) -> dict[str, float]:
    """Score circular error and mean cosine alignment for a fitted probe."""
    design = np.column_stack((features, np.ones(features.shape[0], dtype=np.float64)))
    prediction = design @ weights
    predicted_angle = np.arctan2(prediction[:, 1], prediction[:, 0])
    truth = np.deg2rad(np.asarray(angles_degrees, dtype=np.float64))
    error = np.arctan2(np.sin(predicted_angle - truth), np.cos(predicted_angle - truth))
    return {
        "mean_absolute_circular_error_radians": float(np.mean(np.abs(error))),
        "mean_cosine_alignment": float(np.mean(np.cos(error))),
    }


def f1_copy_diagnostics(
    pooled: ArrayF32,
    full_field_residual: ArrayF32,
    *,
    amplitude_threshold: float = 1e-6,
) -> dict[str, object]:
    """Score all patch/copy pooled traces and full-field F1 residuals."""
    if pooled.ndim != 4 or pooled.shape[1:] != (360, 48, 2):
        raise ValueError(f"expected Nx360x48x2 pooled F1, got {pooled.shape}")
    if full_field_residual.shape != pooled.shape[:3]:
        raise ValueError("full-field residual shape differs from pooled traces")
    complex_trace = pooled[..., 0].astype(np.float64) + 1j * pooled[..., 1].astype(
        np.float64
    )
    amplitudes = np.abs(complex_trace)
    coefficients = np.fft.fft(complex_trace, axis=1) / 360.0
    spectral = np.square(np.abs(coefficients))
    purity = (spectral[:, 1, :] + spectral[:, -1, :]) / (
        np.sum(spectral, axis=1) + EPSILON
    )
    angles = np.deg2rad(np.arange(360, dtype=np.float64))
    phase = np.unwrap(np.angle(complex_trace), axis=1)
    centered_angle = angles - np.mean(angles)
    slopes = np.sum(
        (phase - np.mean(phase, axis=1, keepdims=True)) * centered_angle[None, :, None],
        axis=1,
    ) / np.sum(np.square(centered_angle))
    base_phase = np.angle(complex_trace[:, 0, :])
    expected_phase = base_phase[:, None, :] + angles[None, :, None]
    phase_error = np.angle(np.exp(1j * (np.angle(complex_trace) - expected_phase)))
    phase_rmse = np.sqrt(np.mean(np.square(phase_error), axis=1))
    amplitude_cv = np.std(amplitudes, axis=1) / (np.mean(amplitudes, axis=1) + EPSILON)
    near_zero_fraction = np.mean(amplitudes < amplitude_threshold, axis=1)
    valid = (amplitudes[:, 0, :] >= amplitude_threshold) & (near_zero_fraction <= 0.10)
    rows: list[dict[str, object]] = []
    for copy in range(48):
        copy_valid = valid[:, copy]
        valid_count = int(copy_valid.sum())
        selector = (
            copy_valid if valid_count else np.ones(valid.shape[0], dtype=np.bool_)
        )
        row = {
            "copy": copy,
            "valid_patch_count": valid_count,
            "m1_purity_median": float(np.median(purity[selector, copy])),
            "phase_slope_median": float(np.median(slopes[selector, copy])),
            "phase_rmse_median": float(np.median(phase_rmse[selector, copy])),
            "amplitude_cv_median": float(np.median(amplitude_cv[selector, copy])),
            "full_field_residual_median": float(
                np.median(np.median(full_field_residual[:, :, copy], axis=1))
            ),
            "near_zero_fraction_median": float(np.median(near_zero_fraction[:, copy])),
        }
        row["clean"] = bool(
            valid_count >= 18
            and row["m1_purity_median"] >= 0.75
            and abs(cast("float", row["phase_slope_median"]) - 1.0) <= 0.10
            and row["phase_rmse_median"] <= 0.35
            and row["amplitude_cv_median"] <= 0.25
            and row["full_field_residual_median"] <= 0.25
        )
        rows.append(row)
    return {
        "copies": rows,
        "clean_copy_count": sum(bool(row["clean"]) for row in rows),
        "patch_median_phase_rmse": np.median(phase_rmse, axis=1).tolist(),
        "patch_median_amplitude_cv": np.median(amplitude_cv, axis=1).tolist(),
        "patch_median_full_field_residual": np.median(
            full_field_residual, axis=(1, 2)
        ).tolist(),
    }


def paired_bootstrap_median_difference(
    left: Sequence[float],
    right: Sequence[float],
    *,
    draws: int = 10_000,
    seed: int = 3573134496,
) -> dict[str, object]:
    """Return the locked descriptive paired-patch bootstrap interval."""
    differences = np.asarray(left, dtype=np.float64) - np.asarray(
        right, dtype=np.float64
    )
    if differences.ndim != 1 or differences.size == 0:
        raise ValueError("paired bootstrap inputs must be nonempty vectors")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, differences.size, size=(draws, differences.size))
    medians = np.median(differences[indices], axis=1)
    return {
        "paired_values": differences.tolist(),
        "median": float(np.median(differences)),
        "percentile_95": np.quantile(medians, (0.025, 0.975)).tolist(),
        "draws": draws,
        "seed": seed,
    }


__all__ = [
    "EPSILON",
    "FIT_ANGLES",
    "FIT_RANKS",
    "HELDOUT_ANGLES",
    "HELDOUT_RANKS",
    "GeneratorFit",
    "PcaOrbit",
    "SharedBasis",
    "canonicalization_statistics",
    "cyclic_geometry",
    "evaluate_strict_rollout",
    "exclude_cardinal_neighborhoods",
    "f1_copy_diagnostics",
    "fit_angle_probe",
    "fit_generator_from_orbits",
    "fit_orthogonal_step",
    "fit_shared_basis",
    "fit_skew_generator",
    "harmonic_summary",
    "local_pca",
    "masked_orbit_vectors",
    "matrix_exponential",
    "operator_numerics",
    "paired_bootstrap_median_difference",
    "principal_angles",
    "score_angle_probe",
    "tangent_summary",
]
