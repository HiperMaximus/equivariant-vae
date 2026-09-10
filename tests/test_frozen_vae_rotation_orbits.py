# pyright: reportAny=false, reportArgumentType=false, reportAttributeAccessIssue=false, reportIndexIssue=false, reportOperatorIssue=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false
# Copyright 2026 HiperMaximus
# ruff: noqa: COM812, PLR2004
"""Focused contracts for the local-only Spec 0038 visualizer."""

from __future__ import annotations

import hashlib
import importlib
import math
import sys
import types
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import torch
from PIL import Image
from torch.nn import functional

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
    rotate_f1_field,
    rotation_fixture_measurements,
    seeded_scalar_pair_projection,
    select_f1_copies,
    spatial_latent_pca_diagnostic,
    summarize_dense_orbit_population,
    unpack_final_encoder_f1,
    vector_phase_diagnostic,
    write_offline_html,
)
from eqvae.cli.render_frozen_vae_rotation_orbits import (
    guard_rotation_output_dir,
    verify_pinned_sha256,
)
from eqvae.models.so2_basis import representation_matrix

if TYPE_CHECKING:
    from collections.abc import Callable


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


def _analytic_gaussian(size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    coordinates = torch.arange(size, dtype=torch.float32)
    normalized = (2.0 * coordinates + 1.0 - size) / size
    rows, columns = torch.meshgrid(normalized, normalized, indexing="ij")
    values = torch.exp(
        -((columns - 0.3125).square() + (rows + 0.1875).square()) / (2.0 * 0.2**2)
    )
    return values[None, None], rows, columns


def _centroid(
    values: torch.Tensor,
    *,
    rows: torch.Tensor,
    columns: torch.Tensor,
) -> tuple[float, float]:
    plane = values[0, 0]
    mass = plane.sum()
    return float((plane * columns).sum() / mass), float((plane * rows).sum() / mass)


def _l_arrow(size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    values = torch.zeros(1, 1, size, size)
    row = round(0.3125 * (size - 1))
    column = round(0.6875 * (size - 1))
    vertical = max(1, round(0.1875 * size))
    horizontal = max(1, round(0.125 * size))
    values[0, 0, row : row + vertical, column] = 1.0
    values[0, 0, row, column - horizontal + 1 : column + 1] = 0.5
    values[0, 0, row, column] = 1.0
    return values, (row, column)


def _boundary_wedge(size: int = 32) -> torch.Tensor:
    coordinates = (2.0 * torch.arange(size, dtype=torch.float32) + 1.0 - size) / size
    rows, columns = torch.meshgrid(coordinates, coordinates, indexing="ij")
    values = ((columns > 0.75) & (rows > -0.35) & (rows < 0.20)) * (
        1.0 - (rows + 0.075).abs() / 0.275
    )
    return values.clamp_min(0.0)[None, None]


def _mutant_spatial_rotate(
    values: torch.Tensor,
    degrees: float,
    mutant: str,
) -> torch.Tensor:
    angle = float(degrees) if mutant == "degrees_as_radians" else math.radians(degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    if mutant == "opposite_spatial_sign":
        matrix = ((cosine, sine, 0.0), (-sine, cosine, 0.0))
    elif mutant == "reversed_matrix_order":
        # Treat the coordinate vector as (row, column), then unpack it as
        # (x, y): a distinct matrix-order/layout error, not another sign flip.
        matrix = ((-sine, cosine, 0.0), (cosine, sine, 0.0))
    else:
        matrix = ((cosine, -sine, 0.0), (sine, cosine, 0.0))
    transform = values.new_tensor(matrix).unsqueeze(0)
    grid_align = mutant == "align_corners_grid_sample_mismatch"
    grid = functional.affine_grid(
        transform, list(values.shape), align_corners=grid_align
    )
    padding = "border" if mutant == "border_or_reflection_padding" else "zeros"
    return functional.grid_sample(
        values,
        grid,
        mode="bilinear",
        padding_mode=padding,
        align_corners=False,
    )


def test_continuous_rotation_uses_one_cardinal_path_with_positive_handedness() -> None:
    """Uniform interpolation converges exactly to positive quarter turns."""
    values = torch.arange(3 * 8 * 8, dtype=torch.float32).reshape(1, 3, 8, 8)
    assert torch.equal(continuous_rotate(values, 0), values)
    for degrees in (90, 180, 270):
        observed = continuous_rotate(values, degrees)
        expected = torch.rot90(values, degrees // 90, (-2, -1))
        assert torch.allclose(observed, expected, rtol=0.0, atol=2e-6)


@pytest.mark.parametrize("size", [32, 256])
def test_continuous_rotation_matches_hand_computed_noncardinal_centroids(
    size: int,
) -> None:
    """An analytic asymmetric fixture catches the original affine sign defect."""
    values, rows, columns = _analytic_gaussian(size)
    source_x, source_y = 0.3125, -0.1875
    source_phase = math.atan2(-source_y, source_x)
    for degrees in (17, 31, 73):
        observed = continuous_rotate(values, degrees)
        actual_x, actual_y = _centroid(observed, rows=rows, columns=columns)
        radians = math.radians(degrees)
        expected_x = math.cos(radians) * source_x + math.sin(radians) * source_y
        expected_y = -math.sin(radians) * source_x + math.cos(radians) * source_y
        assert math.hypot(actual_x - expected_x, actual_y - expected_y) < 0.02
        observed_phase = math.atan2(-actual_y, actual_x)
        phase_error = math.atan2(
            math.sin(observed_phase - source_phase - radians),
            math.cos(observed_phase - source_phase - radians),
        )
        assert abs(math.degrees(phase_error)) < 0.5


@pytest.mark.parametrize("size", [32, 256])
def test_continuous_rotation_cardinal_limits_inverse_and_composition(
    size: int,
) -> None:
    """Interpolation errors obey the locked quantitative fixture contract."""
    values, _rows, _columns = _analytic_gaussian(size)
    for cardinal in (90, 180, 270):
        center = continuous_rotate(values, cardinal)
        expected = torch.rot90(values, cardinal // 90, (-2, -1))
        assert float((center - expected).abs().max()) <= 2e-6
        for side in (-0.001, 0.001):
            rms = (
                (continuous_rotate(values, cardinal + side) - center)
                .square()
                .mean()
                .sqrt()
            )
            assert float(rms) < 5e-4
    scale = values.square().mean().sqrt()
    for degrees in (17, 31, 73):
        inverse = continuous_rotate(continuous_rotate(values, degrees), -degrees)
        assert float((inverse - values).square().mean().sqrt() / scale) < 0.03
    for alpha, beta in ((17, -17), (31, 73), (-73, 31)):
        composed = continuous_rotate(continuous_rotate(values, beta), alpha)
        expected = continuous_rotate(values, alpha + beta)
        expected_scale = expected.square().mean().sqrt()
        error = (composed - expected).square().mean().sqrt() / expected_scale
        assert float(error) < 0.03


@pytest.mark.parametrize("size", [32, 256])
def test_locked_rotation_fixture_measurements_pass_full_and_disk_bounds(
    size: int,
) -> None:
    """The machine-readable fixture records both spatial comparison domains."""
    measurements = rotation_fixture_measurements(size)
    assert max(measurements["centroid_error_normalized"].values()) < 0.02
    assert max(measurements["orientation_error_degrees"].values()) < 0.5
    for sides in measurements["cardinal_sided_rms"].values():
        for variants in sides.values():
            assert set(variants) == {"full", "disk"}
            assert max(variants.values()) < 5e-4
    for variants in measurements["inverse_normalized_rms"].values():
        assert max(variants.values()) < 0.03
    for variants in measurements["composition_normalized_rms"].values():
        assert max(variants.values()) < 0.03
    assert measurements["l_arrow_positive_quarter_max_absolute_error"] <= 2e-6
    assert measurements["boundary_wedge_border_mutant_normalized_rms"] > 1e-4
    if size == 32:
        f1 = measurements["f1_fixture"]
        assert f1["noncardinal_centroid_error_normalized"] < 0.02
        assert f1["noncardinal_phase_error_degrees"] < 0.5
        assert f1["positive_quarter_component_x_max"] <= 2e-6
        assert f1["positive_quarter_component_y_max"] >= 0.90


def test_packed_f1_action_rotates_pixel_and_component_independently() -> None:
    """The D-layout fixture rejects spatial-only and opposite fiber actions."""
    values = torch.zeros(1, 48, 2, 32, 32)
    values[0, 7, 0, 8, 23] = 1.0
    observed = rotate_f1_field(values, 90)
    assert observed[0, 7, 0, 8, 8] == pytest.approx(0.0, abs=2e-6)
    assert observed[0, 7, 1, 8, 8] == pytest.approx(1.0, abs=2e-6)
    assert observed[0, 7, 0].abs().max() < 2e-6


@pytest.mark.parametrize("size", [32, 256])
def test_l_arrow_has_the_hand_indexed_positive_quarter_turn(size: int) -> None:
    """An asymmetric one-pixel-width arrow fixes the exact cardinal direction."""
    values, (row, column) = _l_arrow(size)
    observed = continuous_rotate(values, 90)
    target_row, target_column = size - 1 - column, row
    assert observed[0, 0, target_row, target_column] == pytest.approx(1.0, abs=2e-6)
    assert torch.allclose(observed, torch.rot90(values, 1, (-2, -1)), atol=2e-6)


def test_noncardinal_f1_fixture_checks_spatial_and_internal_action() -> None:
    """A noncardinal vector field must move and rotate its components positively."""
    scalar, rows, columns = _analytic_gaussian(32)
    values = torch.zeros(1, 48, 2, 32, 32)
    values[0, 7, 0] = scalar[0, 0]
    observed = rotate_f1_field(values, 31)
    magnitude = observed[0, 7].square().sum(dim=0).sqrt()[None, None]
    actual_x, actual_y = _centroid(magnitude, rows=rows, columns=columns)
    radians = math.radians(31)
    expected_x = math.cos(radians) * 0.3125 + math.sin(radians) * -0.1875
    expected_y = -math.sin(radians) * 0.3125 + math.cos(radians) * -0.1875
    assert math.hypot(actual_x - expected_x, actual_y - expected_y) < 0.02
    pooled = observed[0, 7].sum(dim=(1, 2))
    phase = math.atan2(float(pooled[1]), float(pooled[0]))
    assert phase == pytest.approx(radians, abs=1e-5)


def _load_pinned_escnn_for_rotation_oracle() -> Any:  # noqa: ANN401
    """Load the ignored pinned escnn checkout without permitting cache writes.

    Returns:
        Imported pinned escnn module.

    """

    class NoCacheMemory:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def cache[**P, R](  # noqa: PLR6301
            self,
            function: Callable[P, R] | None = None,
            **_kwargs: object,
        ) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
            if function is None:
                return lambda wrapped: wrapped
            return function

    joblib = types.ModuleType("joblib")
    joblib.Memory = NoCacheMemory  # type: ignore[attr-defined]
    sys.modules["joblib"] = joblib
    module_names = (
        "lie_learn",
        "lie_learn.representations",
        "lie_learn.representations.SO3",
        "lie_learn.representations.SO3.wigner_d",
    )
    for module_name in module_names:
        sys.modules[module_name] = types.ModuleType(module_name)

    def reject_so3(*_args: object, **_kwargs: object) -> None:
        message = "rotation-oracle test entered an SO(3) path"
        raise RuntimeError(message)

    sys.modules[module_names[-1]].wigner_D_matrix = reject_so3  # type: ignore[attr-defined]
    escnn_root = Path(__file__).resolve().parents[1] / "reference/escnn"
    sys.path.insert(0, str(escnn_root))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return importlib.import_module("escnn")


def test_f1_internal_matrix_matches_repository_and_pinned_escnn_oracles() -> None:
    """Cross-check the already hand-fixed positive F1 convention independently."""
    radians = math.radians(31)
    repository_matrix = torch.from_numpy(representation_matrix(1, radians))
    expected = torch.tensor(
        (
            (math.cos(radians), -math.sin(radians)),
            (math.sin(radians), math.cos(radians)),
        ),
        dtype=torch.float64,
    )
    torch.testing.assert_close(repository_matrix, expected, rtol=0.0, atol=1e-12)

    escnn = _load_pinned_escnn_for_rotation_oracle()
    group_space = escnn.gspaces.rot2dOnR2(N=-1, maximum_frequency=1)
    group = group_space.fibergroup
    field_type = escnn.nn.FieldType(group_space, [group.irrep(1)])
    element = group.element(radians, "radians")
    basis_vectors = torch.eye(2, dtype=torch.float64).reshape(2, 2, 1, 1)
    transformed = field_type.transform_fibers(basis_vectors, element)
    escnn_matrix = transformed[:, :, 0, 0].transpose(0, 1)
    torch.testing.assert_close(escnn_matrix, expected, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize(
    "mutant",
    [
        "opposite_spatial_sign",
        "f1_component_swap",
        "opposite_f1_internal_sign",
        "degrees_as_radians",
        "reversed_matrix_order",
        "align_corners_grid_sample_mismatch",
        "border_or_reflection_padding",
    ],
)
def test_every_required_rotation_mutant_is_rejected(mutant: str) -> None:
    """Every preregistered sign/layout/interpolation mutant changes the fixture."""
    if mutant in {"f1_component_swap", "opposite_f1_internal_sign"}:
        scalar, _rows, _columns = _analytic_gaussian(32)
        values = torch.zeros(1, 48, 2, 32, 32)
        values[0, 7, 0] = scalar[0, 0]
        expected = rotate_f1_field(values, 31)
        if mutant == "f1_component_swap":
            observed = expected[:, :, [1, 0]]
        else:
            spatial = continuous_rotate(values.reshape(1, 96, 32, 32), 31).reshape_as(
                values
            )
            radians = math.radians(-31)
            matrix = values.new_tensor((
                (math.cos(radians), -math.sin(radians)),
                (math.sin(radians), math.cos(radians)),
            ))
            observed = torch.einsum("ab,ncbhw->ncahw", matrix, spatial)
    else:
        values = (
            _boundary_wedge()
            if mutant == "border_or_reflection_padding"
            else _analytic_gaussian(32)[0]
        )
        expected = continuous_rotate(values, 31)
        observed = _mutant_spatial_rotate(values, 31, mutant)
    normalized_error = (
        observed - expected
    ).square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-8)
    assert float(normalized_error) > 1e-4


def test_legacy_rotation_writer_refuses_preserved_or_existing_paths(
    tmp_path: Path,
) -> None:
    """Corrected helpers cannot overwrite the superseded Spec 0038 package."""
    repository_root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="preserved provenance"):
        guard_rotation_output_dir(
            repository_root / "runs/local/frozen_vae_rotation_orbits"
        )
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        guard_rotation_output_dir(existing)
    guard_rotation_output_dir(tmp_path / "new-package")


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
