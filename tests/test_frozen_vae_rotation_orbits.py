# pyright: reportAny=false
# Copyright 2026 HiperMaximus
# ruff: noqa: COM812, PLR2004, TC003
"""Focused contracts for the local-only Spec 0038 visualizer."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from eqvae.artifacts.rotation_orbits import (
    OrbitSweep,
    QuarterResidualSummary,
    centered_disk_mask,
    choose_stem_f1_copy,
    collect_dense_mu_population,
    continuous_rotate,
    disk_mean_f1,
    downsample_uint8_rgb_patches,
    isotropic_plot_bounds,
    masked_relative_rms,
    normalized_f1_phase,
    paper_style_spatial_pca,
    pca_orbit_2d,
    pca_orbit_explained_variance,
    pointwise_rgb_probe,
    project_scalar_pair_means,
    render_dense_orbit_population_png,
    render_f1_phase_png,
    render_kernel_mechanism_png,
    render_paper_style_spatial_pca_png,
    render_pointwise_rgb_probe_png,
    render_spatial_latent_pca_png,
    seeded_scalar_pair_projection,
    select_f1_copies,
    spatial_latent_pca_diagnostic,
    summarize_dense_orbit_population,
    unpack_final_encoder_f1,
    vector_phase_diagnostic,
    write_offline_html,
)
from eqvae.cli.render_frozen_vae_rotation_orbits import verify_pinned_sha256


def _sweep(*, f1: bool) -> OrbitSweep:
    angles = np.arange(4, dtype=np.int64)
    mu = np.zeros((4, 16, 32, 32), dtype=np.float32)
    for index in range(4):
        mu[index] = index
    means = None
    if f1:
        means = np.zeros((4, 48, 2), dtype=np.float32)
        for index, angle in enumerate(angles):
            radians = np.deg2rad(float(angle))
            means[index, 7] = (np.cos(radians), np.sin(radians))
    return OrbitSweep(
        model_label="SO(2) VAE" if f1 else "Normal VAE",
        patch_index=12,
        angles_degrees=angles,
        mu=mu,
        latent_residual=np.zeros(4, dtype=np.float64),
        input_roundtrip_floor=np.zeros(4, dtype=np.float64),
        f1_disk_means=means,
    )


def _quarters() -> QuarterResidualSummary:
    return QuarterResidualSummary(
        angles_degrees=np.asarray([0, 90, 180, 270], dtype=np.int64),
        median=np.asarray([0.0, 0.2, 0.3, 0.2], dtype=np.float64),
        lower_quartile=np.asarray([0.0, 0.1, 0.2, 0.1], dtype=np.float64),
        upper_quartile=np.asarray([0.0, 0.3, 0.4, 0.3], dtype=np.float64),
    )


def _spatial_values() -> np.ndarray:
    """Return a finite, nondegenerate 16-channel fixed25 tensor for spatial PCA.

    Returns:
        A channel-first tensor with the fixed25 latent shape.

    """
    rows, columns = np.mgrid[:32, :32].astype(np.float32)
    values = np.zeros((25, 16, 32, 32), dtype=np.float32)
    for image in range(25):
        for channel in range(16):
            values[image, channel] = (
                (channel + 1.0) * rows / 31.0
                + (16.0 - channel) * columns / 47.0
                + image / 25.0
                + 0.02 * np.sin((channel + 1.0) * rows)
            )
    return values


def _affine_probe_inputs() -> tuple[np.ndarray, np.ndarray]:
    """Create a local affine RGB target that a 1x1 probe can recover.

    Returns:
        Spatial descriptors and their quantized RGB affine target.

    """
    generator = np.random.default_rng(42)
    values = generator.normal(size=(25, 16, 32, 32)).astype(np.float32)
    weights = generator.normal(scale=0.03, size=(16, 3))
    rgb = 0.5 + values.transpose(0, 2, 3, 1) @ weights
    targets = np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    return values, targets


def test_continuous_rotation_preserves_identity_and_exact_quarter_turn() -> None:
    """The visualizer shares the exact quarter-turn branch with fixed25."""
    values = torch.arange(3 * 8 * 8, dtype=torch.float32).reshape(1, 3, 8, 8)
    assert torch.equal(continuous_rotate(values, 0), values)
    assert torch.equal(continuous_rotate(values, 90), torch.rot90(values, 1, (-2, -1)))


def test_disk_masked_residual_is_zero_for_identical_fields() -> None:
    """The central-disk diagnostic has a zero identity baseline."""
    values = torch.randn(2, 16, 32, 32)
    mask = centered_disk_mask(32, radius=14.0)
    result = masked_relative_rms(values, values, mask=mask)
    assert torch.equal(result, torch.zeros_like(result))


def test_pca_uses_one_orbit_fit_and_returns_two_coordinates() -> None:
    """A 2D cyclic feature trajectory remains a nondegenerate two-column trace."""
    angles = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    values = np.zeros((12, 1, 2, 2), dtype=np.float32)
    values[:, 0, 0, 0] = np.cos(angles)
    values[:, 0, 0, 1] = np.sin(angles)
    points = pca_orbit_2d(values)
    assert points.shape == (12, 2)
    assert np.linalg.matrix_rank(points) == 2
    assert pca_orbit_explained_variance(values) == pytest.approx(1.0)


def test_dense_population_collection_batches_patch_angle_pairs() -> None:
    """Population inference preserves patch and angle identity across batches."""

    class MeanEncoder(torch.nn.Module):
        @staticmethod
        def encode(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            pooled = torch.nn.functional.avg_pool2d(values.mean(dim=1, keepdim=True), 8)
            mu = pooled.repeat(1, 16, 1, 1)
            return mu, torch.zeros_like(mu)

    patches = torch.zeros(2, 3, 256, 256)
    patches[1] = 1.0
    values = collect_dense_mu_population(
        model=MeanEncoder(),
        patches=patches,
        angles_degrees=(0, 90, 180, 270),
        batch_size=3,
    )
    assert values.shape == (2, 4, 16, 32, 32)
    assert np.allclose(values[0], 0.0)
    assert np.allclose(values[1], 1.0)


def test_dense_population_summary_and_render_cover_all_fixed25(tmp_path: Path) -> None:
    """The all-patch view distinguishes a smooth cycle from a jagged one."""
    angles = np.arange(0, 360, 10, dtype=np.int64)
    radians = np.deg2rad(angles)
    smooth = np.zeros((25, angles.size, 1, 2, 2), dtype=np.float32)
    smooth[:, :, 0, 0, 0] = np.cos(radians)
    smooth[:, :, 0, 0, 1] = np.sin(radians)
    jagged = smooth.copy()
    jagged[:, 1::2, 0, 0, 0] += 0.4
    mask = torch.ones(2, 2, dtype=torch.bool)
    smooth_summary = summarize_dense_orbit_population(
        smooth,
        angles_degrees=angles,
        mask=mask,
    )
    jagged_summary = summarize_dense_orbit_population(
        jagged,
        angles_degrees=angles,
        mask=mask,
    )
    assert smooth_summary.pca_scores.shape == (25, 36, 2)
    assert np.allclose(smooth_summary.pca_explained_variance, 1.0)
    assert np.all(
        smooth_summary.local_linearity_ratio < jagged_summary.local_linearity_ratio
    )
    path = tmp_path / "all25.png"
    render_dense_orbit_population_png(
        path=path,
        normal=jagged_summary,
        so2=smooth_summary,
    )
    with Image.open(path) as image:
        assert image.size == (3000, 2400)


def test_spatial_pca_preserves_cells_and_relative_edge_rms_is_scale_invariant() -> None:
    """The false-colour map is channel-PCA at each cell, not an orbit PCA."""
    values = _spatial_values()
    mask = centered_disk_mask(32, radius=14.0)
    diagnostic = spatial_latent_pca_diagnostic(values, mask=mask)
    scaled = spatial_latent_pca_diagnostic(
        np.multiply(values, 3.0, dtype=np.float32),
        mask=mask,
    )
    disk = mask.numpy().astype(bool)
    assert diagnostic.rgb.shape == (25, 32, 32, 3)
    assert diagnostic.explained_variance.shape == (3,)
    assert diagnostic.rgb_score_scale > 0.0
    assert np.all(diagnostic.rgb[:, ~disk, :] == np.asarray((238, 241, 246)))
    assert np.all(~diagnostic.degenerate)
    assert np.allclose(
        diagnostic.relative_edge_rms,
        scaled.relative_edge_rms,
        rtol=1e-7,
        atol=1e-9,
    )
    assert np.allclose(scaled.edge_rms, 3.0 * diagnostic.edge_rms)


def test_spatial_pca_flags_a_constant_field_instead_of_dividing_by_epsilon() -> None:
    """A collapsed spatial field remains visibly and numerically degenerate."""
    values = np.ones((25, 16, 32, 32), dtype=np.float32)
    diagnostic = spatial_latent_pca_diagnostic(
        values,
        mask=centered_disk_mask(32, radius=14.0),
    )
    assert np.all(diagnostic.degenerate)
    assert np.allclose(diagnostic.relative_edge_rms, 0.0)
    assert np.all(np.isfinite(diagnostic.rgb))


def test_paper_style_pca_refits_each_image_and_jointly_normalizes_rgb() -> None:
    """Paper-style colour contrast stays local to a tile, not a shared scale."""
    values = _spatial_values()
    values[1, 0] *= 2.0
    diagnostic = paper_style_spatial_pca(values)
    assert diagnostic.rgb.shape == (25, 32, 32, 3)
    assert diagnostic.explained_variance.shape == (25, 3)
    assert np.all(~diagnostic.degenerate)
    assert np.all(diagnostic.rgb.min(axis=(1, 2, 3)) == 0)
    assert np.all(diagnostic.rgb.max(axis=(1, 2, 3)) == 255)
    assert not np.allclose(
        diagnostic.explained_variance[0], diagnostic.explained_variance[1]
    )


def test_paper_style_pca_handles_constant_fields_without_fabricating_texture() -> None:
    """A constant posterior produces a finite black PCA tile, not arbitrary colour."""
    diagnostic = paper_style_spatial_pca(np.ones((25, 16, 32, 32), dtype=np.float32))
    assert np.all(diagnostic.degenerate)
    assert np.all(diagnostic.rgb == 0)


def test_pointwise_probe_is_held_out_and_recovers_a_known_local_affine_target() -> None:
    """A 16-to-3 affine map preserves local cells and never fits held-out rows."""
    values, targets = _affine_probe_inputs()
    probe = pointwise_rgb_probe(
        values,
        targets,
        train_indices=range(20),
        heldout_indices=range(20, 25),
    )
    assert probe.predicted_rgb.shape == (25, 32, 32, 3)
    assert probe.train_indices.tolist() == list(range(20))
    assert probe.heldout_indices.tolist() == list(range(20, 25))
    assert np.all(probe.heldout_r2 > 0.99)
    with pytest.raises(ValueError, match="disjoint valid patches"):
        pointwise_rgb_probe(
            values,
            targets,
            train_indices=range(20),
            heldout_indices=(19, 20, 21, 22, 23),
        )


def test_rgb_probe_downsampling_and_rendering_produce_an_ordinary_png(
    tmp_path: Path,
) -> None:
    """The source target is native-size aligned and the renderer shows five holdsout."""
    originals = torch.full((25, 3, 256, 256), 127, dtype=torch.uint8)
    targets = downsample_uint8_rgb_patches(originals)
    assert targets.shape == (25, 32, 32, 3)
    assert targets.dtype == np.uint8
    values, affine_targets = _affine_probe_inputs()
    probe = pointwise_rgb_probe(
        values,
        affine_targets,
        train_indices=range(20),
        heldout_indices=range(20, 25),
    )
    path = tmp_path / "pointwise.png"
    render_pointwise_rgb_probe_png(path=path, normal=probe, so2=probe)
    with Image.open(path) as image:
        assert image.size == (1800, 1800)


def test_isotropic_plot_bounds_preserve_circle_geometry() -> None:
    """A circular PCA orbit cannot be made elliptical by panel aspect ratio."""
    angles = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
    x_min, x_max, y_min, y_max = isotropic_plot_bounds(
        np.cos(angles),
        np.sin(angles),
        plot_width=700,
        plot_height=300,
    )
    assert (x_max - x_min) / (y_max - y_min) == pytest.approx(700 / 300)


def test_pinned_archive_hash_verification_fails_closed(tmp_path: Path) -> None:
    """All-25 archive evidence has an independent, enforced source identity."""
    archive = tmp_path / "latent_mu.pt"
    archive.write_bytes(b"fixed archive")
    digest = hashlib.sha256(b"fixed archive").hexdigest()
    assert verify_pinned_sha256(archive, expected=digest) == digest
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_pinned_sha256(archive, expected="0" * 64)


def test_f1_unpack_pool_selection_and_phase_are_copy_preserving() -> None:
    """F1 never gets confused with the 16 scalar posterior channels."""
    hidden = torch.zeros(2, 144, 32, 32)
    hidden[:, 48 + 2 * 3, :, :] = 2.0
    unpacked = unpack_final_encoder_f1(hidden)
    assert unpacked.shape == (2, 48, 2, 32, 32)
    means = disk_mean_f1(unpacked, mask=centered_disk_mask(32, radius=14.0))
    assert torch.allclose(means[:, 3, 0], torch.full((2,), 2.0))
    initial = np.zeros((25, 48, 2), dtype=np.float32)
    initial[:, 3, 0] = 4.0
    assert select_f1_copies(initial, count=1).tolist() == [3]
    trace = np.column_stack((np.cos(np.arange(4.0)), np.sin(np.arange(4.0)))).astype(
        np.float32
    )
    normalized = normalized_f1_phase(trace, min_magnitude=1e-6)
    assert np.allclose(normalized, trace, atol=1e-6)


def test_seeded_scalar_pair_null_is_orthonormal_and_shape_preserving() -> None:
    """The normal-model null has fixed, energy-preserving scalar pairings."""
    projection = seeded_scalar_pair_projection(channels=8, pairs=4, seed=30_039)
    repeated = seeded_scalar_pair_projection(channels=8, pairs=4, seed=30_039)
    assert torch.equal(projection, repeated)
    flattened = projection.reshape(8, 8)
    assert torch.allclose(flattened @ flattened.T, torch.eye(8), atol=1e-5)
    means = np.arange(16, dtype=np.float32).reshape(2, 8)
    projected = project_scalar_pair_means(
        means,
        projections=projection.numpy()[None, :, :, :],
    )
    assert projected.shape == (2, 1, 4, 2)


def test_phase_diagnostic_preserves_the_expected_positive_90_degree_rotation() -> None:
    """The displayed complex convention maps a positive quarter turn to +i."""
    angles = np.asarray([0, 90, 180, 270], dtype=np.int64)
    means = np.zeros((4, 1, 2), dtype=np.float32)
    means[:, 0, 0] = np.cos(np.deg2rad(angles))
    means[:, 0, 1] = np.sin(np.deg2rad(angles))
    diagnostic = vector_phase_diagnostic(
        means,
        selected_copies=np.asarray([0], dtype=np.int64),
        angles_degrees=angles,
        min_magnitude=1e-6,
    )
    assert diagnostic.normalized[1, 0, 0] == pytest.approx(0.0, abs=1e-6)
    assert diagnostic.normalized[1, 0, 1] == pytest.approx(1.0, abs=1e-6)
    assert np.allclose(diagnostic.phase_error_degrees, 0.0, atol=1e-5)


def test_kernel_choice_and_rendered_outputs_are_valid(tmp_path: Path) -> None:
    """The static mechanism and phase figures are ordinary nonempty PNGs."""
    kernel = torch.zeros(48, 3, 9, 9)
    kernel[16 + 2 * 5 : 16 + 2 * 5 + 2] = 3.0
    assert choose_stem_f1_copy(kernel) == 5
    kernel_path = tmp_path / "kernel.png"
    assert render_kernel_mechanism_png(path=kernel_path, kernel=kernel) == 5
    with Image.open(kernel_path) as image:
        assert image.size == (1800, 1120)
    phase_path = tmp_path / "phase.png"
    normal_pairs = np.zeros((4, 1, 48, 2), dtype=np.float32)
    normal_pairs[:, 0, 7, 0] = 1.0
    render_f1_phase_png(
        path=phase_path,
        trained_sweep=_sweep(f1=True),
        untrained_sweep=_sweep(f1=True),
        trained_selected_copies=np.asarray([7], dtype=np.int64),
        normal_pair_orbit=normal_pairs,
        normal_selected_pairs=np.asarray([[7]], dtype=np.int64),
        min_magnitude=1e-6,
    )
    with Image.open(phase_path) as image:
        assert image.size == (2400, 1520)
    spatial = spatial_latent_pca_diagnostic(
        _spatial_values(),
        mask=centered_disk_mask(32, radius=14.0),
    )
    spatial_path = tmp_path / "spatial.png"
    render_spatial_latent_pca_png(
        path=spatial_path,
        originals=torch.zeros(25, 3, 256, 256, dtype=torch.uint8),
        display_indices=(0, 8, 12, 24),
        normal=spatial,
        so2=spatial,
    )
    with Image.open(spatial_path) as image:
        assert image.size == (2400, 1500)
    paper_style = paper_style_spatial_pca(_spatial_values())
    paper_style_path = tmp_path / "paper-style-spatial.png"
    render_paper_style_spatial_pca_png(
        path=paper_style_path,
        originals=torch.zeros(25, 3, 256, 256, dtype=torch.uint8),
        display_indices=(0, 8, 12, 24),
        normal=paper_style,
        so2=paper_style,
    )
    with Image.open(paper_style_path) as image:
        assert image.size == (1800, 1500)


def test_offline_html_has_only_precomputed_local_data(tmp_path: Path) -> None:
    """The advisor page keeps controls beside plots and exposes reproducible math.

    The page is evidence for an advisor conversation, so it must remain offline
    while explaining the PCA fit, scale-normalized residual, and F1 null rather
    than relying on an unexplained attractive trace.
    """
    path = tmp_path / "rotation-orbits.html"
    spatial = spatial_latent_pca_diagnostic(
        _spatial_values(),
        mask=centered_disk_mask(32, radius=14.0),
    )
    paper_style = paper_style_spatial_pca(_spatial_values())
    values, targets = _affine_probe_inputs()
    pointwise_probe = pointwise_rgb_probe(
        values,
        targets,
        train_indices=range(20),
        heldout_indices=range(20, 25),
    )
    write_offline_html(
        path=path,
        baseline=_sweep(f1=False),
        so2=_sweep(f1=True),
        untrained_so2=_sweep(f1=True),
        baseline_quarters=_quarters(),
        so2_quarters=_quarters(),
        selected_copies=np.asarray([7], dtype=np.int64),
        normal_pair_orbit=np.ones((4, 1, 48, 2), dtype=np.float32),
        normal_selected_pairs=np.asarray([[7]], dtype=np.int64),
        min_magnitude=1e-6,
        kernel=torch.zeros(48, 3, 9, 9),
        normal_spatial=spatial,
        so2_spatial=spatial,
        normal_paper_style=paper_style,
        so2_paper_style=paper_style,
        normal_pointwise_probe=pointwise_probe,
        so2_pointwise_probe=pointwise_probe,
        spatial_originals=torch.zeros(25, 3, 256, 256, dtype=torch.uint8),
        spatial_display_indices=(0, 8, 12, 24),
    )
    document = path.read_text(encoding="utf-8")
    assert "https://" not in document
    assert "http://" not in document
    assert "already-precomputed" in document
    assert 'max="3"' in document
    assert 'id="normal-chart"' in document
    assert 'id="so2-chart"' in document
    assert "PC 1 (raw score)" in document
    assert '"posterior_rms":' in document
    assert "Archived all-25 exact quarter turns" in document
    assert "9,856" in document
    assert 'id="trained-phase-chart"' in document
    assert 'id="kernel-red"' in document
    assert 'id="orbit-explanation"' in document
    assert 'id="posterior-explanation"' in document
    assert 'id="f1-explanation"' in document
    assert 'id="spatial-pca-explanation"' in document
    assert 'id="pointwise-rgb-probe-explanation"' in document
    assert 'annotation encoding="application/x-tex"' in document
    assert "How to read relative RMS" in document
    assert "the normal VAE is lower at every nonzero angle" in document
    assert "Why the normal-VAE line is only a null" in document
    assert "Why RMS rather than a mean?" in document
    assert "Spatial latent PCA-RGB" in document
    assert "Paper-style visual companion" in document
    assert "Held-out pointwise latent-to-RGB probe" in document
    assert "appendPointwiseProbeSection();" in document
    assert '"pointwise_rgb_probe":' in document
    assert "showXTickLabels:false" in document
    assert "paper_style" in document
    assert "drawRgbCanvas" in document
    assert "appendSpatialPcaSection();" in document
    assert "renderInlineTex(document.querySelector('main'));" in document
    assert "String.fromCharCode(92)" in document
    assert (
        "orbit.replaceChildren(explanation('orbit-explanation'),control,orbitGrid)"
        in document
    )
    assert "<img" not in document
    assert "data:image" not in document
    assert 'id="play"' not in document
