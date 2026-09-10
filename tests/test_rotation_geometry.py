# pyright: reportAny=false, reportArgumentType=false, reportGeneralTypeIssues=false, reportOperatorIssue=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
# Copyright 2026 HiperMaximus
# ruff: noqa: D103, PLR2004
"""Focused numerical tests for the locked Spec 0050 geometry contract."""

from __future__ import annotations

import math

import numpy as np
import pytest

from eqvae.evaluation.rotation_geometry import (
    FIT_ANGLES,
    HELDOUT_RANKS,
    canonicalization_statistics,
    cyclic_geometry,
    evaluate_strict_rollout,
    exclude_cardinal_neighborhoods,
    f1_copy_diagnostics,
    fit_angle_probe,
    fit_generator_from_orbits,
    fit_orthogonal_step,
    fit_shared_basis,
    fit_skew_generator,
    harmonic_summary,
    local_pca,
    masked_orbit_vectors,
    matrix_exponential,
    operator_numerics,
    paired_bootstrap_median_difference,
    principal_angles,
    score_angle_probe,
    tangent_summary,
)


def _ideal_orbits() -> np.ndarray:
    angles = np.deg2rad(np.arange(360, dtype=np.float64))
    values = np.zeros((25, 360, 6), dtype=np.float64)
    for patch in range(25):
        amplitude = 1.0 + patch / 100.0
        phase = patch * 0.03
        values[patch, :, 0] = amplitude * np.cos(angles + phase)
        values[patch, :, 1] = amplitude * np.sin(angles + phase)
        values[patch, :, 2] = patch / 25.0
        values[patch, :, 3] = (patch / 25.0) ** 2
        values[patch, :, 4] = math.sin(patch)
        values[patch, :, 5] = math.cos(patch)
    return values


def test_masked_vectors_preserve_patch_angle_channel_order() -> None:
    values = np.arange(2 * 3 * 2 * 2 * 2, dtype=np.float32).reshape(2, 3, 2, 2, 2)
    mask = np.asarray([[True, False], [False, True]])
    observed = masked_orbit_vectors(values, mask)
    assert observed.shape == (2, 3, 4)
    assert observed[0, 0].tolist() == [0.0, 3.0, 4.0, 7.0]


def test_cyclic_geometry_and_cardinal_exclusion_score_a_circle() -> None:
    values = _ideal_orbits()[:2, :, :2]
    summary = cyclic_geometry(values)
    ratio = np.asarray(summary["local_linearity_ratio"])
    assert np.allclose(ratio, 2.0 * math.sin(math.pi / 360.0), atol=1e-9)
    assert np.all(np.asarray(summary["step_size_cv"]) < 1e-12)
    excluded = exclude_cardinal_neighborhoods(values)
    assert excluded["kept_step_count"] == 336
    assert np.all(np.asarray(excluded["step_size_cv"]) < 1e-12)


def test_curvature_uses_a_consistent_rms_inner_product() -> None:
    angles = np.deg2rad(np.arange(360, dtype=np.float64))
    ellipse = np.column_stack((2.0 * np.cos(angles), np.sin(angles)))
    values = np.repeat(ellipse[None, :, :], 2, axis=0)
    summary = cyclic_geometry(values)
    curvature = np.asarray(summary["curvature_median"])
    assert np.all(np.isfinite(curvature))
    assert np.all(curvature > 0.0)


def test_local_pca_harmonics_and_tangents_recover_ideal_m1_plane() -> None:
    values = _ideal_orbits()[0]
    pca = local_pca(values, components=6)
    assert pca.scores.shape == (360, 6)
    assert np.sum(pca.explained_fraction[:2]) > 0.999999
    raw_total = float(np.mean(np.sum(np.square(values - values.mean(axis=0)), axis=1)))
    harmonic = harmonic_summary(pca.scores, raw_total_energy=raw_total)
    assert harmonic["m1_fraction_raw"] > 0.999999
    assert harmonic["effective_frequency_count"] == pytest.approx(1.0, abs=1e-6)
    tangent = tangent_summary(values)
    assert tangent["span_dimension_90"] == 2
    assert tangent["local_rank1_fraction"] > 0.999


def test_skew_fit_and_exponential_recover_rotation_frequency() -> None:
    angles = np.linspace(0.0, 2.0 * math.pi, 1000, endpoint=False)
    states = np.column_stack((np.cos(angles), np.sin(angles)))
    derivatives = np.column_stack((-np.sin(angles), np.cos(angles)))
    fit = fit_skew_generator(states, derivatives)
    assert fit.design_rank == 1
    assert np.allclose(fit.matrix, [[0.0, -1.0], [1.0, 0.0]], atol=1e-12)
    assert np.allclose(
        matrix_exponential(fit.matrix, math.pi / 2),
        [[0.0, -1.0], [1.0, 0.0]],
        atol=1e-12,
    )


def test_procrustes_cross_check_recovers_the_same_one_step_action() -> None:
    angles = np.deg2rad(np.arange(0, 360, 5, dtype=np.float64))
    states = np.column_stack((np.cos(angles), np.sin(angles)))
    following = np.roll(states, -1, axis=0)
    observed = fit_orthogonal_step(states, following)
    expected = np.asarray([
        [math.cos(math.radians(5)), -math.sin(math.radians(5))],
        [math.sin(math.radians(5)), math.cos(math.radians(5))],
    ])
    assert np.allclose(observed, expected, atol=1e-12)
    generator = np.asarray([[0.0, -1.0], [1.0, 0.0]])
    diagnostics = operator_numerics(
        generator,
        observed,
        step_radians=math.radians(5),
    )
    assert max(diagnostics.values()) < 1e-12
    reflected = states.copy()
    reflected[:, 0] *= -1.0
    proper = fit_orthogonal_step(states, reflected)
    assert np.linalg.det(proper) == pytest.approx(1.0)


def test_shared_generator_transfers_to_heldout_patches_from_one_view() -> None:
    values = _ideal_orbits()
    shared = fit_shared_basis(values, maximum_dimension=2, oversampling=2)
    fit = fit_generator_from_orbits(values, shared, dimension=2, conditional=False)
    evaluation = evaluate_strict_rollout(values, shared, fit, dimension=2)
    assert evaluation["patch_ranks"] == list(HELDOUT_RANKS)
    assert evaluation["median_nrmse"] < 0.02
    assert evaluation["median_r2"] > 0.999
    assert evaluation["median_projected_nrmse"] < 0.02
    assert evaluation["median_projected_r2"] > 0.999
    assert np.allclose(evaluation["observed_winding"], 1.0, atol=0.03)
    assert np.allclose(evaluation["predicted_winding"], 1.0, atol=0.03)
    assert fit.eigenfrequencies[0] == pytest.approx(0.998731, abs=1e-5)


def test_canonicalization_requires_low_within_and_preserves_content() -> None:
    identity = _ideal_orbits()[:8]
    canonical = identity.copy()
    canonical[:, :, :2] = canonical[:, :1, :2]
    result = canonicalization_statistics(canonical, identity)
    assert result["w_ratio"] < 1e-10
    assert result["patch_w_ratio_median"] < 1e-10
    assert result["b_ratio"] > 0.8
    assert result["retrieval_mean"] == pytest.approx(1.0)
    assert result["distance_stress_defined"] is True


def test_angle_probe_is_refit_and_scores_circular_alignment() -> None:
    angles = np.asarray(FIT_ANGLES, dtype=np.int64)
    radians = np.deg2rad(angles)
    features = np.column_stack((np.cos(radians), np.sin(radians), np.ones(angles.size)))
    weights = fit_angle_probe(features, angles)
    score = score_angle_probe(features, angles, weights)
    assert score["mean_absolute_circular_error_radians"] < 1e-12
    assert score["mean_cosine_alignment"] > 0.999999


def test_all_f1_copies_pass_an_ideal_internal_action() -> None:
    angles = np.deg2rad(np.arange(360, dtype=np.float64))
    pooled = np.zeros((25, 360, 48, 2), dtype=np.float32)
    for patch in range(25):
        for copy in range(48):
            amplitude = 1.0 + 0.01 * patch + 0.001 * copy
            pooled[patch, :, copy, 0] = amplitude * np.cos(angles)
            pooled[patch, :, copy, 1] = amplitude * np.sin(angles)
    residual = np.zeros((25, 360, 48), dtype=np.float32)
    summary = f1_copy_diagnostics(pooled, residual)
    assert summary["clean_copy_count"] == 48
    assert all(row["valid_patch_count"] == 25 for row in summary["copies"])


def test_principal_angles_and_paired_bootstrap_are_deterministic() -> None:
    left = np.eye(4, 2)
    right = left.copy()
    assert np.allclose(principal_angles(left, right), 0.0)
    first = paired_bootstrap_median_difference([1, 2, 3], [0, 1, 1], draws=100)
    second = paired_bootstrap_median_difference([1, 2, 3], [0, 1, 1], draws=100)
    assert first == second
