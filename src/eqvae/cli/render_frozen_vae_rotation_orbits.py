# pyright: reportAny=false, reportArgumentType=false, reportUnnecessaryCast=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false
# Copyright 2026 HiperMaximus
# ruff: noqa: COM812, DOC201, DOC501, E501, EM101, PLR0913, PLR0914, PLR2004, PLR0915, SLF001, TRY003
"""Render the local-only frozen-VAE rotation visualizations (Spec 0038)."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from numpy.typing import NDArray

from eqvae.artifacts.fixed25_equivariance import ORIGINALS_IMAGES_KEY
from eqvae.artifacts.rotation_orbits import (
    LATENT_DISK_RADIUS,
    DenseOrbitPopulation,
    PaperStyleSpatialPca,
    PointwiseRgbProbe,
    SpatialLatentPcaDiagnostic,
    centered_disk_mask,
    collect_dense_mu_population,
    collect_orbit,
    disk_mean_f1,
    disk_mean_scalar,
    downsample_uint8_rgb_patches,
    paper_style_spatial_pca,
    pointwise_rgb_probe,
    project_scalar_pair_means,
    quarter_residual_summary,
    render_dense_orbit_population_png,
    render_f1_phase_png,
    render_kernel_mechanism_png,
    render_latent_orbits_png,
    render_paper_style_spatial_pca_png,
    render_pointwise_rgb_probe_png,
    render_spatial_latent_pca_png,
    seeded_scalar_pair_projection,
    select_f1_copies,
    spatial_latent_pca_diagnostic,
    summarize_dense_orbit_population,
    unpack_final_encoder_f1,
    write_offline_html,
)
from eqvae.inference.checkpoints import load_frozen_checkpoint
from eqvae.models.non_equivariant_vae import BOTTLENECK_CHANNELS
from eqvae.models.so2_vae import SO2VAE

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor


ArrayF32 = NDArray[np.float32]

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_OUTPUT_DIR = _REPOSITORY_ROOT / "runs/local/frozen_vae_rotation_orbits"
_LEGACY_PROVENANCE_DIR = _DEFAULT_OUTPUT_DIR.resolve()
_ORIGINALS_PATH = _REPOSITORY_ROOT / "docs/data/fixed25/originals.pt"
_NORMAL_RUN = _REPOSITORY_ROOT / "runs/kaggle/selected_runtime_full_v4_session3"
_SO2_RUN = (
    _REPOSITORY_ROOT / "runs/kaggle/so2_selected_runtime_full_session7_fresh_v1_retry1"
)
_NORMAL_LATENT_ARCHIVE = _NORMAL_RUN / "artifacts/fixed25/boundary_060000/latent_mu.pt"
_SO2_LATENT_ARCHIVE = _SO2_RUN / "artifacts/fixed25/boundary_060000/latent_mu.pt"
_EXPECTED_ARCHIVE_SHA256 = {
    "normal_fixed25_latent_mu": "d852f8dfee6fff9d7b1eaf03f11165d25fc563b9aa6d47e240ed086897fad3aa",
    "so2_fixed25_latent_mu": "b0cd23feacdf46030a90fc065d4c8b12a44c97f4d417af007c77956619dcbdfb",
}
_MIN_F1_MAGNITUDE = 1e-6
_RANDOM_SO2_SEED = 30_038
_NORMAL_SCALAR_PAIR_SEEDS = tuple(range(30_039, 30_071))
_SPATIAL_PCA_DISPLAY_INDICES = (0, 8, 12, 24)
_POINTWISE_PROBE_TRAIN_INDICES = tuple(range(20))
_POINTWISE_PROBE_HELDOUT_INDICES = tuple(range(20, 25))
_POINTWISE_PROBE_RIDGE = 1e-3


@dataclass(frozen=True)
class RenderArgs:
    """Resolved command-line inputs for the local visualizer."""

    output_dir: Path
    dense_index: int
    max_angle: int
    population_only: bool
    population_step: int
    population_batch_size: int


def main(argv: Sequence[str] | None = None) -> int:
    """Generate a dense-orbit PNG package without changing model state."""
    args = _parse_args(argv)
    guard_rotation_output_dir(args.output_dir)
    originals_uint8 = _load_and_verify_originals()
    originals = originals_uint8.to(torch.float32).div(255.0).mul(2.0).sub(1.0)
    if not 0 <= args.dense_index < originals.shape[0]:
        message = (
            f"dense index {args.dense_index} is outside 0..{originals.shape[0] - 1}"
        )
        raise ValueError(message)
    normal = load_frozen_checkpoint(
        _NORMAL_RUN / "checkpoints/step_060000.pt",
        model_name="normal_vae",
    )
    so2 = cast(
        "SO2VAE",
        load_frozen_checkpoint(
            _SO2_RUN / "checkpoints/step_060000.pt",
            model_name="so2_vae",
        ),
    )
    if args.population_only:
        _render_dense_population(
            output=args.output_dir,
            originals=originals,
            normal=normal,
            so2=so2,
            step=args.population_step,
            batch_size=args.population_batch_size,
        )
        return 0
    angles = list(range(args.max_angle + 1))
    patch = originals[args.dense_index : args.dense_index + 1]
    initial_f1_means = _initial_f1_disk_means(so2, originals)
    selected_f1 = select_f1_copies(initial_f1_means, count=3)
    random_so2 = _seeded_random_so2()
    normal_pair_projections = np.stack([
        seeded_scalar_pair_projection(
            channels=BOTTLENECK_CHANNELS,
            pairs=BOTTLENECK_CHANNELS // 2,
            seed=seed,
        )
        .cpu()
        .numpy()
        .astype(np.float32)
        for seed in _NORMAL_SCALAR_PAIR_SEEDS
    ]).astype(np.float32)
    normal_initial_scalar_means = _initial_normal_scalar_means(
        normal,
        originals,
    )
    normal_initial_pair_means = project_scalar_pair_means(
        normal_initial_scalar_means,
        projections=normal_pair_projections,
    )
    normal_selected_pairs = np.stack([
        select_f1_copies(normal_initial_pair_means[:, index, :, :], count=3)
        for index in range(normal_pair_projections.shape[0])
    ]).astype(np.int64)
    baseline_orbit = collect_orbit(
        model=normal,
        model_label="normal_vae",
        patch=patch,
        patch_index=args.dense_index,
        angles_degrees=angles,
        collect_f1=False,
        collect_scalar=True,
    )
    so2_orbit = collect_orbit(
        model=so2,
        model_label="so2_vae",
        patch=patch,
        patch_index=args.dense_index,
        angles_degrees=angles,
        collect_f1=True,
    )
    if baseline_orbit.scalar_disk_means is None:
        raise RuntimeError("normal scalar disk means were not collected")
    normal_pair_orbit = project_scalar_pair_means(
        baseline_orbit.scalar_disk_means,
        projections=normal_pair_projections,
    )
    random_so2_orbit = collect_orbit(
        model=random_so2,
        model_label="Seeded random SO(2) VAE",
        patch=patch,
        patch_index=args.dense_index,
        angles_degrees=angles,
        collect_f1=True,
    )
    normal_archive_hash = verify_pinned_sha256(
        _NORMAL_LATENT_ARCHIVE,
        expected=_EXPECTED_ARCHIVE_SHA256["normal_fixed25_latent_mu"],
    )
    so2_archive_hash = verify_pinned_sha256(
        _SO2_LATENT_ARCHIVE,
        expected=_EXPECTED_ARCHIVE_SHA256["so2_fixed25_latent_mu"],
    )
    normal_quarters = quarter_residual_summary(saved_latent_path=_NORMAL_LATENT_ARCHIVE)
    so2_quarters = quarter_residual_summary(saved_latent_path=_SO2_LATENT_ARCHIVE)
    spatial_mask = centered_disk_mask(32, radius=LATENT_DISK_RADIUS)
    normal_mu_clean = _load_archived_mu_clean(_NORMAL_LATENT_ARCHIVE)
    so2_mu_clean = _load_archived_mu_clean(_SO2_LATENT_ARCHIVE)
    normal_spatial = spatial_latent_pca_diagnostic(
        normal_mu_clean,
        mask=spatial_mask,
    )
    so2_spatial = spatial_latent_pca_diagnostic(
        so2_mu_clean,
        mask=spatial_mask,
    )
    normal_paper_style = paper_style_spatial_pca(normal_mu_clean)
    so2_paper_style = paper_style_spatial_pca(so2_mu_clean)
    pointwise_targets = downsample_uint8_rgb_patches(originals_uint8)
    normal_pointwise_probe = pointwise_rgb_probe(
        normal_mu_clean,
        pointwise_targets,
        train_indices=_POINTWISE_PROBE_TRAIN_INDICES,
        heldout_indices=_POINTWISE_PROBE_HELDOUT_INDICES,
        ridge=_POINTWISE_PROBE_RIDGE,
    )
    so2_pointwise_probe = pointwise_rgb_probe(
        so2_mu_clean,
        pointwise_targets,
        train_indices=_POINTWISE_PROBE_TRAIN_INDICES,
        heldout_indices=_POINTWISE_PROBE_HELDOUT_INDICES,
        ridge=_POINTWISE_PROBE_RIDGE,
    )
    output = args.output_dir
    render_latent_orbits_png(
        path=output / "01-latent-orbits.png",
        baseline=baseline_orbit,
        so2=so2_orbit,
        baseline_quarters=normal_quarters,
        so2_quarters=so2_quarters,
        original_patch=patch,
    )
    render_f1_phase_png(
        path=output / "02-f1-phase.png",
        trained_sweep=so2_orbit,
        untrained_sweep=random_so2_orbit,
        trained_selected_copies=selected_f1,
        normal_pair_orbit=normal_pair_orbit,
        normal_selected_pairs=normal_selected_pairs,
        min_magnitude=_MIN_F1_MAGNITUDE,
    )
    kernel = so2.stem_conv.expanded_kernel()
    kernel_copy = render_kernel_mechanism_png(
        path=output / "03-kernel-mechanism.png",
        kernel=kernel,
    )
    render_spatial_latent_pca_png(
        path=output / "04-spatial-latent-pca.png",
        originals=originals_uint8,
        display_indices=_SPATIAL_PCA_DISPLAY_INDICES,
        normal=normal_spatial,
        so2=so2_spatial,
    )
    render_paper_style_spatial_pca_png(
        path=output / "05-paper-style-latent-pca.png",
        originals=originals_uint8,
        display_indices=_SPATIAL_PCA_DISPLAY_INDICES,
        normal=normal_paper_style,
        so2=so2_paper_style,
    )
    render_pointwise_rgb_probe_png(
        path=output / "06-pointwise-rgb-probe.png",
        normal=normal_pointwise_probe,
        so2=so2_pointwise_probe,
    )
    write_offline_html(
        path=output / "rotation-orbits.html",
        baseline_quarters=normal_quarters,
        so2_quarters=so2_quarters,
        baseline=baseline_orbit,
        so2=so2_orbit,
        untrained_so2=random_so2_orbit,
        selected_copies=selected_f1,
        normal_pair_orbit=normal_pair_orbit,
        normal_selected_pairs=normal_selected_pairs,
        min_magnitude=_MIN_F1_MAGNITUDE,
        kernel=kernel,
        normal_spatial=normal_spatial,
        so2_spatial=so2_spatial,
        normal_paper_style=normal_paper_style,
        so2_paper_style=so2_paper_style,
        normal_pointwise_probe=normal_pointwise_probe,
        so2_pointwise_probe=so2_pointwise_probe,
        spatial_originals=originals_uint8,
        spatial_display_indices=_SPATIAL_PCA_DISPLAY_INDICES,
    )
    _write_manifest(
        output / "manifest.json",
        args=args,
        selected_f1=selected_f1,
        normal_selected_pairs=normal_selected_pairs,
        normal_pair_projections=normal_pair_projections,
        random_so2=random_so2,
        kernel_copy=kernel_copy,
        originals=originals_uint8,
        normal_spatial=normal_spatial,
        so2_spatial=so2_spatial,
        normal_paper_style=normal_paper_style,
        so2_paper_style=so2_paper_style,
        normal_pointwise_probe=normal_pointwise_probe,
        so2_pointwise_probe=so2_pointwise_probe,
        archive_sha256={
            "normal_fixed25_latent_mu": normal_archive_hash,
            "so2_fixed25_latent_mu": so2_archive_hash,
        },
    )
    return 0


def guard_rotation_output_dir(output_dir: Path) -> None:
    """Refuse to overwrite the superseded Spec 0038 provenance package."""
    resolved = output_dir.resolve()
    if resolved == _LEGACY_PROVENANCE_DIR:
        raise ValueError(
            "the Spec 0038 rotation package is preserved provenance and cannot be rewritten"
        )
    if resolved.exists():
        message = f"rotation output path already exists: {resolved}"
        raise FileExistsError(message)


def _render_dense_population(
    *,
    output: Path,
    originals: Tensor,
    normal: torch.nn.Module,
    so2: torch.nn.Module,
    step: int,
    batch_size: int,
) -> None:
    """Render the all-fixed25 continuous-angle geometry companion."""
    angles = tuple(range(0, 360, step))
    mask = centered_disk_mask(32, radius=LATENT_DISK_RADIUS)
    normal_values = collect_dense_mu_population(
        model=normal,
        patches=originals,
        angles_degrees=angles,
        batch_size=batch_size,
    )
    normal_summary = summarize_dense_orbit_population(
        normal_values,
        angles_degrees=angles,
        mask=mask,
    )
    del normal_values
    so2_values = collect_dense_mu_population(
        model=so2,
        patches=originals,
        angles_degrees=angles,
        batch_size=batch_size,
    )
    so2_summary = summarize_dense_orbit_population(
        so2_values,
        angles_degrees=angles,
        mask=mask,
    )
    del so2_values
    render_dense_orbit_population_png(
        path=output / "07-all25-latent-orbits.png",
        normal=normal_summary,
        so2=so2_summary,
    )
    _write_dense_population_manifest(
        output / "07-all25-latent-orbits.json",
        normal=normal_summary,
        so2=so2_summary,
        batch_size=batch_size,
    )


def _write_dense_population_manifest(
    path: Path,
    *,
    normal: DenseOrbitPopulation,
    so2: DenseOrbitPopulation,
    batch_size: int,
) -> None:
    """Record the exact all-fixed25 grid, metrics, and source identities."""

    def model_payload(summary: DenseOrbitPopulation) -> dict[str, object]:
        return {
            "pca_explained_variance": summary.pca_explained_variance.tolist(),
            "local_linearity_ratio": summary.local_linearity_ratio.tolist(),
            "step_size_cv": summary.step_size_cv.tolist(),
            "median": {
                "pca_explained_variance": float(
                    np.median(summary.pca_explained_variance)
                ),
                "local_linearity_ratio": float(
                    np.median(summary.local_linearity_ratio)
                ),
                "step_size_cv": float(np.median(summary.step_size_cv)),
            },
        }

    document = {
        "schema": "spec0038.fixed25_dense_orbit_population.v1",
        "label": "Fixed validation 25; post-hoc continuous-angle exploratory visualization; not sealed-test evaluation.",
        "angles_degrees": normal.angles_degrees.tolist(),
        "display_endpoint_degrees": 360,
        "batch_size": batch_size,
        "rotation": {
            "non_quarter_method": "affine_grid+grid_sample_bilinear_zero_padding_align_corners_false",
            "quarter_turn_method": "torch.rot90",
            "metric_mask": "central_disk",
            "latent_disk_radius": LATENT_DISK_RADIUS,
        },
        "measurement": {
            "pca": "one raw-score PCA fit per model and patch over the complete sampled cycle; display only",
            "local_linearity_ratio": "cyclic RMS second difference divided by cyclic RMS first difference on disk-masked raw mu; lower is locally smoother at the recorded angular step",
            "step_size_cv": "coefficient of variation of cyclic one-step disk-masked raw-mu RMS; lower is more uniform",
            "inference": "descriptive fixed-validation evidence only; no bootstrap, model selection, or sealed-test claim",
        },
        "normal": model_payload(normal),
        "so2": model_payload(so2),
        "paired_favorable_counts": {
            "so2_lower_local_linearity_ratio": int(
                np.sum(so2.local_linearity_ratio < normal.local_linearity_ratio)
            ),
            "so2_lower_step_size_cv": int(
                np.sum(so2.step_size_cv < normal.step_size_cv)
            ),
            "so2_higher_pca_explained_variance": int(
                np.sum(so2.pca_explained_variance > normal.pca_explained_variance)
            ),
        },
        "source_sha256": {
            "originals": _sha256_file(_ORIGINALS_PATH),
            "normal_checkpoint": _sha256_file(
                _NORMAL_RUN / "checkpoints/step_060000.pt"
            ),
            "so2_checkpoint": _sha256_file(_SO2_RUN / "checkpoints/step_060000.pt"),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_and_verify_originals() -> Tensor:
    """Load canonical uint8 fixed-25 images and enforce archive identity."""
    canonical_bytes = _ORIGINALS_PATH.read_bytes()
    for run in (_NORMAL_RUN, _SO2_RUN):
        archive = run / "artifacts/fixed25/originals.pt"
        if canonical_bytes != archive.read_bytes():
            message = f"fixed25 originals mismatch: {_ORIGINALS_PATH} != {archive}"
            raise ValueError(message)
    payload = torch.load(_ORIGINALS_PATH, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("canonical fixed25 originals payload must be a dictionary")
    images = payload.get(ORIGINALS_IMAGES_KEY)
    if not isinstance(images, torch.Tensor):
        raise TypeError("canonical fixed25 images_uint8 must be a tensor")
    if images.shape != (25, 3, 256, 256) or images.dtype != torch.uint8:
        message = f"canonical fixed25 images must be uint8 [25,3,256,256], got {images.shape}/{images.dtype}"
        raise ValueError(message)
    return images


def _load_archived_mu_clean(path: Path) -> ArrayF32:
    """Load the one archived, finite final-posterior tensor used by the maps."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        message = f"archived latent payload must be a dictionary: {path}"
        raise TypeError(message)
    values = payload.get("mu_clean")
    if not isinstance(values, torch.Tensor):
        message = f"archived latent payload lacks tensor mu_clean: {path}"
        raise TypeError(message)
    if values.shape != (25, 16, 32, 32):
        message = f"archived mu_clean must be [25,16,32,32], got {values.shape}"
        raise ValueError(message)
    array = values.detach().cpu().numpy().astype(np.float32)
    if not np.isfinite(array).all():
        message = f"archived mu_clean contains nonfinite values: {path}"
        raise ValueError(message)
    return array


def _initial_f1_disk_means(model: SO2VAE, originals: Tensor) -> ArrayF32:
    """Obtain θ=0 disk means for reproducible F1-copy selection only."""
    mask = centered_disk_mask(32, radius=LATENT_DISK_RADIUS)
    with torch.inference_mode():
        hidden = model._encode_features(originals)  # pyright: ignore[reportPrivateUsage]
        means = disk_mean_f1(unpack_final_encoder_f1(hidden), mask=mask)
    return means.cpu().numpy().astype(np.float32)


def _initial_normal_scalar_means(model: torch.nn.Module, originals: Tensor) -> ArrayF32:
    """Obtain theta-zero scalar means before applying any arbitrary pairing."""
    mask = centered_disk_mask(32, radius=LATENT_DISK_RADIUS)
    with torch.inference_mode():
        hidden = model._encode_features(originals)  # pyright: ignore[reportAttributeAccessIssue, reportCallIssue]
        means = disk_mean_scalar(hidden, mask=mask)
    return means.cpu().numpy().astype(np.float32)


def _seeded_random_so2() -> SO2VAE:
    """Create a deterministic untrained architecture control without RNG drift."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(_RANDOM_SO2_SEED)
        control = SO2VAE()
    return control.eval()


def _write_manifest(
    path: Path,
    *,
    args: RenderArgs,
    selected_f1: NDArray[np.int64],
    normal_selected_pairs: NDArray[np.int64],
    normal_pair_projections: ArrayF32,
    random_so2: SO2VAE,
    kernel_copy: int,
    originals: Tensor,
    normal_spatial: SpatialLatentPcaDiagnostic,
    so2_spatial: SpatialLatentPcaDiagnostic,
    normal_paper_style: PaperStyleSpatialPca,
    so2_paper_style: PaperStyleSpatialPca,
    normal_pointwise_probe: PointwiseRgbProbe,
    so2_pointwise_probe: PointwiseRgbProbe,
    archive_sha256: dict[str, str],
) -> None:
    """Record all source identities and nonformal visualization settings."""
    paths = {
        "originals": _ORIGINALS_PATH,
        "normal_checkpoint": _NORMAL_RUN / "checkpoints/step_060000.pt",
        "so2_checkpoint": _SO2_RUN / "checkpoints/step_060000.pt",
    }
    document = {
        "schema": "spec0038.frozen_vae_rotation_orbits.v6",
        "label": "Fixed validation 25; continuous-angle exploratory visualization; not sealed-test evaluation.",
        "dense_patch_index": args.dense_index,
        "dense_angle_grid": {
            "start_degrees": 0,
            "stop_degrees": args.max_angle,
            "step_degrees": 1,
            "count": args.max_angle + 1,
        },
        "dense_display_endpoint_degrees": 360 if args.max_angle == 359 else None,
        "rotation": {
            "non_quarter_method": "affine_grid+grid_sample_bilinear_zero_padding_align_corners_false",
            "quarter_turn_method": "torch.rot90",
            "metric_mask": "central_disk",
            "latent_disk_radius": LATENT_DISK_RADIUS,
        },
        "display": {
            "pca": "one raw-score PCA fit per model/patch orbit; equal data units per screen unit; compare within-panel shape only",
            "html": "single offline advisor page; every chart is precomputed data rendered as local SVG, with no embedded raster figures",
        },
        "f1": {
            "feature": "final_encoder_D_before_mu_head",
            "packed_shape": "B x 144 x 32 x 32 = 48 F0 + 48 F1 x 2",
            "selected_copies": selected_f1.tolist(),
            "selection": "top three by median theta-zero disk-mean magnitude over fixed validation 25",
            "near_zero_threshold": _MIN_F1_MAGNITUDE,
            "interpretation": "diagnostic only; not headline equivariance evidence",
            "controls": {
                "untrained_so2": {
                    "role": "seeded untrained architectural control; construction diagnostic, not a learned-equivalence claim",
                    "seed": _RANDOM_SO2_SEED,
                    "state_sha256": _state_dict_sha256(random_so2),
                    "torch_version": torch.__version__,
                    "copies": selected_f1.tolist(),
                },
                "normal_scalar_pair_null": {
                    "role": "arbitrary negative-control null, never a normal-VAE F1 representation",
                    "ensemble_size": len(_NORMAL_SCALAR_PAIR_SEEDS),
                    "seeds": list(_NORMAL_SCALAR_PAIR_SEEDS),
                    "projection_sha256": _sha256_array(normal_pair_projections),
                    "selection": "top three by each projection's median theta-zero disk-mean magnitude over fixed validation 25",
                    "selected_copy_indices_per_projection": normal_selected_pairs.tolist(),
                },
            },
        },
        "kernel": {
            "selected_stem_f1_copy": kernel_copy,
            "interpretation": "architectural mechanism only; not learned steering-orbit evidence",
        },
        "spatial_pca": _spatial_manifest(
            normal=normal_spatial,
            so2=so2_spatial,
            normal_paper_style=normal_paper_style,
            so2_paper_style=so2_paper_style,
        ),
        "pointwise_rgb_probe": _pointwise_rgb_probe_manifest(
            normal=normal_pointwise_probe,
            so2=so2_pointwise_probe,
        ),
        "source_sha256": {
            **{name: _sha256_file(source) for name, source in paths.items()},
            **archive_sha256,
        },
        "original_tensor": {
            "shape": list(originals.shape),
            "stored_domain": "uint8 RGB; converted to uint8/255*2-1 only for encoder inference",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _spatial_manifest(
    *,
    normal: SpatialLatentPcaDiagnostic,
    so2: SpatialLatentPcaDiagnostic,
    normal_paper_style: PaperStyleSpatialPca,
    so2_paper_style: PaperStyleSpatialPca,
) -> dict[str, object]:
    """Serialize the full all-fixed25 scale context without a second fit."""

    def values(diagnostic: SpatialLatentPcaDiagnostic) -> dict[str, object]:
        return {
            "explained_variance": diagnostic.explained_variance.tolist(),
            "rgb_score_scale": diagnostic.rgb_score_scale,
            "edge_rms": diagnostic.edge_rms.tolist(),
            "centered_spatial_rms": diagnostic.centered_spatial_rms.tolist(),
            "posterior_rms": diagnostic.posterior_rms.tolist(),
            "relative_edge_rms": diagnostic.relative_edge_rms.tolist(),
            "degenerate": diagnostic.degenerate.tolist(),
        }

    def paper_values(diagnostic: PaperStyleSpatialPca) -> dict[str, object]:
        return {
            "explained_variance": diagnostic.explained_variance.tolist(),
            "rgb_min": diagnostic.rgb_min.tolist(),
            "rgb_max": diagnostic.rgb_max.tolist(),
            "degenerate": diagnostic.degenerate.tolist(),
        }

    return {
        "source": "common-scale PCA-RGB diagnostic plus paper-style visual companion",
        "mu_key": "mu_clean",
        "mu_shape": [25, 16, 32, 32],
        "fit_scope": "one PCA per model over all 25 x 616 central-disk descriptors",
        "display_indices": list(_SPATIAL_PCA_DISPLAY_INDICES),
        "score_scale": "one symmetric Q99(abs(score)) over all three PCs and fixed25 disk cells per model",
        "pca_sign": "largest-absolute loading positive per component",
        "exterior": "neutral RGB, excluded from fit and metric",
        "metric": "r_edge=edge_RMS/centered_spatial_RMS; r_edge=0 and degenerate=true when denominator is zero",
        "normal": values(normal),
        "so2": values(so2),
        "paper_style": {
            "purpose": "visual convention only; no source-image blend and no cross-tile numeric comparison",
            "fit_scope": "one PCA per model and fixed25 image over its 32x32 spatial descriptors",
            "score_scale": "one joint min/max over all three PCA score planes per image",
            "display": "bilinear 32x32 RGB upsampling",
            "component_order": "released EQ-VAE ordering: third-largest, second-largest, largest eigenvalue to R/G/B; signs unconstrained",
            "normal": paper_values(normal_paper_style),
            "so2": paper_values(so2_paper_style),
        },
    }


def _pointwise_rgb_probe_manifest(
    *,
    normal: PointwiseRgbProbe,
    so2: PointwiseRgbProbe,
) -> dict[str, object]:
    """Record the one declared affine-readout split and held-out scores."""
    if not np.array_equal(
        normal.train_indices, so2.train_indices
    ) or not np.array_equal(
        normal.heldout_indices,
        so2.heldout_indices,
    ):
        raise ValueError("pointwise RGB probes must share the declared fixed25 split")
    return {
        "purpose": "local posterior appearance diagnostic; not equivariance or VAE reconstruction evidence",
        "feature": "mu_clean[n,:,p] at one spatial cell only",
        "target": "canonical RGB source downsampled 256x256 to 32x32 with antialiased bilinear interpolation",
        "model": "one standardized affine 16-to-3 ridge readout plus RGB bias per VAE",
        "parameters_per_model": 51,
        "ridge": _POINTWISE_PROBE_RIDGE,
        "train_indices": normal.train_indices.tolist(),
        "heldout_indices": normal.heldout_indices.tolist(),
        "metric": "per-held-out patch pixel R2 against the training RGB channel mean; MSE is evaluated in RGB [0,1] before display clipping",
        "normal": {
            "heldout_r2": normal.heldout_r2.tolist(),
            "heldout_mse": normal.heldout_mse.tolist(),
        },
        "so2": {
            "heldout_r2": so2.heldout_r2.tolist(),
            "heldout_mse": so2.heldout_mse.tolist(),
        },
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(values: NDArray[np.generic]) -> str:
    """Hash a contiguous array including its machine-readable shape and dtype."""
    normalized = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(normalized.dtype).encode("ascii"))
    digest.update(np.asarray(normalized.shape, dtype=np.int64).tobytes())
    digest.update(normalized.tobytes())
    return digest.hexdigest()


def _state_dict_sha256(model: torch.nn.Module) -> str:
    """Hash a model state in name order for reproducible local-control identity."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        contiguous = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def verify_pinned_sha256(path: Path, *, expected: str) -> str:
    """Fail closed when a frozen archived source differs from its recorded digest."""
    observed = _sha256_file(path)
    if observed != expected:
        message = f"SHA-256 mismatch for {path}: expected {expected}, got {observed}"
        raise ValueError(message)
    return observed


def _parse_args(argv: Sequence[str] | None) -> RenderArgs:
    parser = argparse.ArgumentParser(
        description="Render local-only continuous rotation visualizations for frozen VAEs."
    )
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--dense-index",
        type=int,
        default=12,
        help="One predetermined fixed-25 patch for the expensive dense 1-degree sweep.",
    )
    parser.add_argument(
        "--max-angle",
        type=int,
        default=359,
        help="Highest inclusive dense angle; 359 is the complete nonduplicated orbit.",
    )
    parser.add_argument(
        "--population-only",
        action="store_true",
        help="Render only the all-fixed25 full-cycle population companion.",
    )
    parser.add_argument(
        "--population-step",
        type=int,
        default=10,
        help="Angular step for the all-fixed25 population cycle; must divide 90.",
    )
    parser.add_argument(
        "--population-batch-size",
        type=int,
        default=8,
        help="Bounded inference batch size for patch-angle pairs.",
    )
    parsed = parser.parse_args(argv)
    if not 0 <= parsed.max_angle <= 359:
        raise ValueError("max-angle must be between 0 and 359")
    if parsed.population_step <= 0 or 90 % parsed.population_step != 0:
        raise ValueError("population-step must be a positive divisor of 90")
    if parsed.population_batch_size <= 0:
        raise ValueError("population-batch-size must be positive")
    return RenderArgs(
        output_dir=parsed.output_dir.resolve(),
        dense_index=parsed.dense_index,
        max_angle=parsed.max_angle,
        population_only=parsed.population_only,
        population_step=parsed.population_step,
        population_batch_size=parsed.population_batch_size,
    )


if __name__ == "__main__":
    raise SystemExit(main())
