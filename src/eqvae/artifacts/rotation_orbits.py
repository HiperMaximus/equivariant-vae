# pyright: reportAny=false, reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportPrivateUsage=false, reportReturnType=false, reportUnnecessaryCast=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false
# Copyright 2026 HiperMaximus
# ruff: noqa: COM812, DOC201, DOC501, E501, EM101, EM102, PLR0913, PLR0914, PLR0917, PLR2004, PYI041, RUF001, RUF005, SLF001, TRY003
"""Local-only continuous-rotation visualization helpers (Spec 0038).

These functions intentionally keep their exploratory, interpolation-based
visualization separate from the canonical exact-``rot90`` fixed-25 protocol.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, cast

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont
from torch.nn import functional

from eqvae.models.so2_architecture_probe import A_LAYOUT, D_LAYOUT

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from torch import Tensor

ArrayF32 = NDArray[np.float32]
ArrayF64 = NDArray[np.float64]
ArrayI64 = NDArray[np.int64]
ArrayU8 = NDArray[np.uint8]

EXPLORATORY_LABEL: Final = (
    "Fixed validation 25; continuous-angle exploratory visualization; "
    "not sealed-test evaluation."
)
LATENT_DISK_RADIUS: Final = 14.0
IMAGE_DISK_RADIUS: Final = 112.0
_EPS: Final = 1e-8
_PLOT_COLORS: Final = ("#3a86ff", "#e63946", "#8338ec")


@dataclass(frozen=True)
class OrbitSweep:
    """One dense orbit from one model and one predetermined fixed-25 patch."""

    model_label: str
    patch_index: int
    angles_degrees: ArrayI64
    mu: ArrayF32
    latent_residual: ArrayF64
    input_roundtrip_floor: ArrayF64
    f1_disk_means: ArrayF32 | None = None
    scalar_disk_means: ArrayF32 | None = None


@dataclass(frozen=True)
class DenseOrbitPopulation:
    """All-patch PCA traces and descriptive rotation-regularity summaries."""

    angles_degrees: ArrayI64
    pca_scores: ArrayF64
    pca_explained_variance: ArrayF64
    local_linearity_ratio: ArrayF64
    step_size_cv: ArrayF64


@dataclass(frozen=True)
class QuarterResidualSummary:
    """All-fixed-25 exact-quarter residual statistics for one frozen model."""

    angles_degrees: ArrayI64
    median: ArrayF64
    lower_quartile: ArrayF64
    upper_quartile: ArrayF64


@dataclass(frozen=True)
class VectorPhaseDiagnostic:
    """Phase, gain, and raw-amplitude traces for selected two-vector copies."""

    normalized: ArrayF64
    phase_error_degrees: ArrayF64
    gain: ArrayF64
    amplitude: ArrayF64
    valid: NDArray[np.bool_]


@dataclass(frozen=True)
class SpatialLatentPcaDiagnostic:
    """One model's fixed-25 spatial PCA maps and local-coherence statistics."""

    rgb: ArrayU8
    explained_variance: ArrayF64
    rgb_score_scale: float
    edge_rms: ArrayF64
    centered_spatial_rms: ArrayF64
    posterior_rms: ArrayF64
    relative_edge_rms: ArrayF64
    degenerate: NDArray[np.bool_]


@dataclass(frozen=True)
class PaperStyleSpatialPca:
    """Per-image whole-grid PCA-RGB maps in the released EQ-VAE style."""

    rgb: ArrayU8
    explained_variance: ArrayF64
    rgb_min: ArrayF64
    rgb_max: ArrayF64
    degenerate: NDArray[np.bool_]


@dataclass(frozen=True)
class PointwiseRgbProbe:
    """One held-out affine RGB readout from spatial posterior descriptors."""

    target_rgb: ArrayU8
    predicted_rgb: ArrayU8
    train_indices: ArrayI64
    heldout_indices: ArrayI64
    heldout_r2: ArrayF64
    heldout_mse: ArrayF64


def continuous_rotate(values: Tensor, degrees: int | float) -> Tensor:
    """Rotate a batch with the repository's continuous-angle convention.

    Exact quarter turns stay exact. Other angles use the existing SO(2) test
    convention: inverse sampling, bilinear interpolation, zero padding, and
    ``align_corners=False``.
    """
    rounded = round(float(degrees))
    if (
        math.isclose(float(degrees), float(rounded), abs_tol=1e-12)
        and rounded % 90 == 0
    ):
        return torch.rot90(values, rounded // 90, dims=(-2, -1))
    angle = math.radians(float(degrees))
    cosine = math.cos(angle)
    sine = math.sin(angle)
    transform = values.new_tensor(((cosine, sine, 0.0), (-sine, cosine, 0.0)))
    transform = transform.unsqueeze(0).expand(values.shape[0], -1, -1)
    grid = functional.affine_grid(transform, list(values.shape), align_corners=False)
    return functional.grid_sample(
        values,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )


def centered_disk_mask(
    spatial_size: int,
    *,
    radius: float,
    device: torch.device | None = None,
) -> Tensor:
    """Return a center-symmetric two-dimensional disk mask."""
    coordinates = torch.arange(spatial_size, device=device, dtype=torch.float32)
    centered = coordinates - (spatial_size - 1.0) / 2.0
    rows, columns = torch.meshgrid(centered, centered, indexing="ij")
    return rows.square() + columns.square() <= radius**2


def masked_relative_rms(
    observed: Tensor,
    expected: Tensor,
    *,
    mask: Tensor,
    eps: float = _EPS,
) -> Tensor:
    """Compute per-sample relative RMS over a shared spatial mask."""
    if observed.shape != expected.shape:
        message = f"observed shape {observed.shape} does not match {expected.shape}"
        raise ValueError(message)
    if observed.ndim != 4:
        message = f"expected BxCxHxW tensors, got {observed.shape}"
        raise ValueError(message)
    if mask.shape != observed.shape[-2:]:
        message = f"mask shape {mask.shape} does not match {observed.shape[-2:]}"
        raise ValueError(message)
    weights = mask.to(device=observed.device, dtype=observed.dtype)[None, None, :, :]
    count = weights.sum() * observed.shape[1]
    numerator = ((observed - expected).square() * weights).sum(dim=(1, 2, 3))
    denominator = (expected.square() * weights).sum(dim=(1, 2, 3))
    return torch.sqrt(numerator / count) / (torch.sqrt(denominator / count) + eps)


def pca_orbit_2d(values: ArrayF32, *, mask: Tensor | None = None) -> ArrayF64:
    """Fit one PCA plane to every point of one orbit and return its scores."""
    scores, _explained_variance = _pca_orbit(values, mask=mask)
    return scores


def pca_orbit_explained_variance(
    values: ArrayF32, *, mask: Tensor | None = None
) -> float:
    """Return the fraction of centered orbit variance in its first two PCs."""
    _scores, explained_variance = _pca_orbit(values, mask=mask)
    return explained_variance


def collect_dense_mu_population(
    *,
    model: torch.nn.Module,
    patches: Tensor,
    angles_degrees: Sequence[int],
    batch_size: int,
) -> ArrayF32:
    """Encode every requested patch-angle pair in bounded inference batches."""
    if patches.ndim != 4 or tuple(patches.shape[1:]) != (3, 256, 256):
        message = f"expected Nx3x256x256 normalized patches, got {patches.shape}"
        raise ValueError(message)
    if not angles_degrees or angles_degrees[0] != 0:
        raise ValueError("angles must be nonempty and start at zero")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    angles = tuple(int(angle) for angle in angles_degrees)
    if len(set(angles)) != len(angles) or any(
        angle < 0 or angle >= 360 for angle in angles
    ):
        raise ValueError("angles must be unique integers in [0, 359]")
    work = [
        (patch_index, angle_index, angle)
        for patch_index in range(patches.shape[0])
        for angle_index, angle in enumerate(angles)
    ]
    population: np.ndarray | None = None
    with torch.inference_mode():
        for start in range(0, len(work), batch_size):
            chunk = work[start : start + batch_size]
            inputs = torch.cat(
                [
                    continuous_rotate(
                        patches[patch_index : patch_index + 1],
                        angle,
                    )
                    for patch_index, _angle_index, angle in chunk
                ],
                dim=0,
            )
            encoded = model.encode(inputs)  # pyright: ignore[reportAttributeAccessIssue]
            mu = cast("Tensor", encoded[0]).detach().cpu().numpy().astype(np.float32)
            if population is None:
                population = np.empty(
                    (patches.shape[0], len(angles), *mu.shape[1:]),
                    dtype=np.float32,
                )
            for row, (patch_index, angle_index, _angle) in enumerate(chunk):
                population[patch_index, angle_index] = mu[row]
    if population is None:
        raise RuntimeError("dense population collection produced no values")
    return cast("ArrayF32", population)


def summarize_dense_orbit_population(
    values: ArrayF32,
    *,
    angles_degrees: Sequence[int],
    mask: Tensor,
    eps: float = _EPS,
) -> DenseOrbitPopulation:
    """Summarize full-cycle geometry without selecting attractive patches."""
    if values.ndim != 5:
        message = f"expected NxTxCxHxW values, got {values.shape}"
        raise ValueError(message)
    angles = np.asarray(angles_degrees, dtype=np.int64)
    if values.shape[1] != angles.size:
        message = (
            f"angle count {angles.size} does not match orbit length {values.shape[1]}"
        )
        raise ValueError(message)
    if tuple(mask.shape) != tuple(values.shape[-2:]):
        message = f"mask shape {mask.shape} does not match {values.shape[-2:]}"
        raise ValueError(message)
    if angles.size < 4:
        raise ValueError("at least four angles are required for cyclic geometry")
    scores: list[ArrayF64] = []
    explained: list[float] = []
    linearity: list[float] = []
    step_cv: list[float] = []
    disk = mask.detach().cpu().numpy().astype(bool)
    for orbit in values:
        orbit_scores, orbit_explained = _pca_orbit(orbit, mask=mask)
        features = orbit[:, :, disk].reshape(orbit.shape[0], -1).astype(np.float64)
        first = np.roll(features, -1, axis=0) - features
        second = np.roll(first, -1, axis=0) - first
        first_rms = float(np.sqrt(np.mean(np.square(first))))
        second_rms = float(np.sqrt(np.mean(np.square(second))))
        step_lengths = np.sqrt(np.mean(np.square(first), axis=1))
        mean_step = float(np.mean(step_lengths))
        scores.append(orbit_scores)
        explained.append(orbit_explained)
        linearity.append(second_rms / (first_rms + eps))
        step_cv.append(float(np.std(step_lengths) / (mean_step + eps)))
    return DenseOrbitPopulation(
        angles_degrees=angles,
        pca_scores=np.stack(scores).astype(np.float64),
        pca_explained_variance=np.asarray(explained, dtype=np.float64),
        local_linearity_ratio=np.asarray(linearity, dtype=np.float64),
        step_size_cv=np.asarray(step_cv, dtype=np.float64),
    )


def _pca_orbit(
    values: ArrayF32, *, mask: Tensor | None = None
) -> tuple[ArrayF64, float]:
    """Use the dual Gram form of one PCA fit over all orbit rows."""
    if values.ndim != 4:
        message = f"expected TxCxHxW values, got {values.shape}"
        raise ValueError(message)
    flattened = values
    if mask is not None:
        if tuple(mask.shape) != tuple(values.shape[-2:]):
            message = f"mask shape {mask.shape} does not match {values.shape[-2:]}"
            raise ValueError(message)
        flattened = values[:, :, mask.detach().cpu().numpy().astype(bool)]
    observations = flattened.reshape(values.shape[0], -1).astype(np.float64)
    centered = observations - observations.mean(axis=0, keepdims=True)
    gram = centered @ centered.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    selected = order[:2]
    nonnegative = np.maximum(eigenvalues, 0.0)
    scores = eigenvectors[:, selected] * np.sqrt(nonnegative[selected])
    total = float(np.sum(nonnegative))
    explained = float(np.sum(nonnegative[selected]) / total) if total > 0.0 else 0.0
    return cast("ArrayF64", scores), explained


def spatial_latent_pca_diagnostic(
    values: ArrayF32,
    *,
    mask: Tensor,
    eps: float = _EPS,
) -> SpatialLatentPcaDiagnostic:
    """Fit one all-fixed25 PCA map basis and measure local latent variation.

    The fit is intentionally per model, but shared by all of that model's
    fixed-25 maps. This avoids both per-image colour rotations and an invalid
    assumption that channels in separately trained encoders are aligned.
    """
    if values.ndim != 4 or values.shape[1] != 16:
        message = f"expected Nx16xHxW values, got {values.shape}"
        raise ValueError(message)
    if tuple(mask.shape) != tuple(values.shape[-2:]):
        message = f"mask shape {mask.shape} does not match {values.shape[-2:]}"
        raise ValueError(message)
    if not np.isfinite(values).all():
        raise ValueError("spatial PCA values must be finite")
    disk = mask.detach().cpu().numpy().astype(bool)
    if int(disk.sum()) < 2:
        raise ValueError("spatial PCA mask must contain at least two cells")
    selected = values[:, :, disk].astype(np.float64)
    descriptors = selected.transpose(0, 2, 1).reshape(-1, values.shape[1])
    centered = descriptors - descriptors.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    components = eigenvectors[:, order[:3]].copy()
    for component in range(components.shape[1]):
        dominant = int(np.argmax(np.abs(components[:, component])))
        if components[dominant, component] < 0.0:
            components[:, component] *= -1.0
    scores = centered @ components
    total_variance = float(eigenvalues.sum())
    explained = (
        eigenvalues[:3] / total_variance
        if total_variance > eps
        else np.zeros(3, dtype=np.float64)
    )
    score_scale = float(np.quantile(np.abs(scores), 0.99))
    if score_scale <= eps:
        score_scale = 1.0
    normalized = np.clip(0.5 + scores / (2.0 * score_scale), 0.0, 1.0)
    rgb = np.full(
        (values.shape[0], values.shape[2], values.shape[3], 3),
        fill_value=np.asarray((238, 241, 246), dtype=np.uint8),
        dtype=np.uint8,
    )
    rgb[:, disk, :] = np.rint(
        normalized.reshape(values.shape[0], -1, 3) * 255.0
    ).astype(np.uint8)
    horizontal = disk[:, :-1] & disk[:, 1:]
    vertical = disk[:-1, :] & disk[1:, :]
    edge_differences = np.concatenate(
        (
            (values[:, :, :, 1:] - values[:, :, :, :-1])[:, :, horizontal],
            (values[:, :, 1:, :] - values[:, :, :-1, :])[:, :, vertical],
        ),
        axis=2,
    ).astype(np.float64)
    edge_rms = np.sqrt(np.mean(np.square(edge_differences), axis=(1, 2)))
    centered_spatial = selected - selected.mean(axis=2, keepdims=True)
    centered_spatial_rms = np.sqrt(np.mean(np.square(centered_spatial), axis=(1, 2)))
    posterior_rms = np.sqrt(np.mean(np.square(selected), axis=(1, 2)))
    degenerate = centered_spatial_rms <= eps
    relative_edge_rms = np.divide(
        edge_rms,
        centered_spatial_rms,
        out=np.zeros_like(edge_rms),
        where=~degenerate,
    )
    return SpatialLatentPcaDiagnostic(
        rgb=rgb,
        explained_variance=cast("ArrayF64", explained.astype(np.float64)),
        rgb_score_scale=score_scale,
        edge_rms=cast("ArrayF64", edge_rms.astype(np.float64)),
        centered_spatial_rms=cast("ArrayF64", centered_spatial_rms.astype(np.float64)),
        posterior_rms=cast("ArrayF64", posterior_rms.astype(np.float64)),
        relative_edge_rms=cast("ArrayF64", relative_edge_rms.astype(np.float64)),
        degenerate=degenerate,
    )


def paper_style_spatial_pca(
    values: ArrayF32,
    *,
    eps: float = _EPS,
) -> PaperStyleSpatialPca:
    """Return per-image channel-PCA RGB maps without source-image blending.

    This deliberately follows the released EQ-VAE visualization convention:
    every image receives an independent PCA over its full spatial grid, the
    three returned score planes share one within-image min/max normalization,
    and component signs remain unconstrained.  It is visual context, not a
    cross-image or cross-model numerical comparison.
    """
    if values.ndim != 4 or values.shape[1] != 16:
        message = f"expected Nx16xHxW values, got {values.shape}"
        raise ValueError(message)
    if not np.isfinite(values).all():
        raise ValueError("paper-style spatial PCA values must be finite")
    count, _channels, height, width = values.shape
    rgb = np.empty((count, height, width, 3), dtype=np.uint8)
    explained = np.zeros((count, 3), dtype=np.float64)
    minima = np.zeros(count, dtype=np.float64)
    maxima = np.zeros(count, dtype=np.float64)
    degenerate = np.zeros(count, dtype=bool)
    for image_index in range(count):
        descriptors = values[image_index].reshape(16, -1).T.astype(np.float64)
        centered = descriptors - descriptors.mean(axis=0, keepdims=True)
        covariance = centered.T @ centered / max(descriptors.shape[0] - 1, 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        nonnegative = np.maximum(eigenvalues, 0.0)
        total_variance = float(nonnegative.sum())
        if total_variance > eps:
            explained[image_index] = nonnegative[-3:][::-1] / total_variance
        components = eigenvectors[:, -3:]
        scores = centered @ components
        minimum = float(scores.min())
        maximum = float(scores.max())
        minima[image_index] = minimum
        maxima[image_index] = maximum
        range_ = maximum - minimum
        degenerate[image_index] = range_ <= eps
        normalized = (scores - minimum) / (range_ + eps)
        rgb[image_index] = np.rint(normalized.reshape(height, width, 3) * 255.0).astype(
            np.uint8
        )
    return PaperStyleSpatialPca(
        rgb=rgb,
        explained_variance=cast("ArrayF64", explained),
        rgb_min=cast("ArrayF64", minima),
        rgb_max=cast("ArrayF64", maxima),
        degenerate=degenerate,
    )


def downsample_uint8_rgb_patches(originals: Tensor, *, size: int = 32) -> ArrayU8:
    """Antialiased-downsample canonical fixed25 RGB patches for a local probe."""
    if originals.ndim != 4 or originals.shape[1] != 3 or originals.dtype != torch.uint8:
        message = (
            f"expected Nx3xHxW uint8 originals, got {originals.shape}/{originals.dtype}"
        )
        raise ValueError(message)
    if size < 1:
        raise ValueError("RGB probe output size must be positive")
    resized = functional.interpolate(
        originals.to(torch.float32).div(255.0),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    rgb = torch.round(resized.mul(255.0)).to(torch.uint8).permute(0, 2, 3, 1)
    return cast("ArrayU8", rgb.cpu().numpy())


def pointwise_rgb_probe(
    values: ArrayF32,
    target_rgb: ArrayU8,
    *,
    train_indices: Sequence[int],
    heldout_indices: Sequence[int],
    ridge: float = 1e-3,
) -> PointwiseRgbProbe:
    """Fit one 16-to-3 affine map on patches and report only held-out accuracy."""
    if values.ndim != 4 or values.shape[1] != 16:
        message = f"expected Nx16xHxW values, got {values.shape}"
        raise ValueError(message)
    expected_shape = (values.shape[0], values.shape[2], values.shape[3], 3)
    if target_rgb.shape != expected_shape or target_rgb.dtype != np.uint8:
        message = f"expected uint8 RGB targets with shape {expected_shape}, got {target_rgb.shape}/{target_rgb.dtype}"
        raise ValueError(message)
    if not np.isfinite(values).all():
        raise ValueError("pointwise RGB probe values must be finite")
    train = np.asarray(train_indices, dtype=np.int64)
    heldout = np.asarray(heldout_indices, dtype=np.int64)
    image_count = values.shape[0]
    if train.size != 20 or heldout.size != 5:
        raise ValueError(
            "pointwise RGB probe requires exactly 20 train and 5 held-out patches"
        )
    all_indices = np.concatenate((train, heldout))
    valid_range = np.all((all_indices >= 0) & (all_indices < image_count))
    if np.unique(all_indices).size != all_indices.size or not valid_range:
        raise ValueError(
            "pointwise RGB probe train and held-out indices must be disjoint valid patches"
        )
    descriptors = values.transpose(0, 2, 3, 1).astype(np.float64)
    targets = target_rgb.astype(np.float64) / 255.0
    train_features = descriptors[train].reshape(-1, 16)
    feature_mean = train_features.mean(axis=0, keepdims=True)
    feature_scale = train_features.std(axis=0, keepdims=True)
    feature_scale[feature_scale <= _EPS] = 1.0
    normalized_train = (train_features - feature_mean) / feature_scale
    design = np.concatenate(
        (normalized_train, np.ones((normalized_train.shape[0], 1))),
        axis=1,
    )
    penalty = np.diag(np.concatenate((np.full(16, ridge), np.zeros(1))))
    weights = np.linalg.solve(
        design.T @ design + penalty,
        design.T @ targets[train].reshape(-1, 3),
    )
    normalized_all = (descriptors.reshape(-1, 16) - feature_mean) / feature_scale
    all_design = np.concatenate(
        (normalized_all, np.ones((normalized_all.shape[0], 1))),
        axis=1,
    )
    predictions = (all_design @ weights).reshape(targets.shape)
    target_mean = targets[train].mean(axis=(0, 1, 2), keepdims=True)
    heldout_r2 = []
    heldout_mse = []
    for index in heldout:
        difference = predictions[index] - targets[index]
        sse = float(np.square(difference).sum())
        sst = float(np.square(targets[index] - target_mean).sum())
        heldout_r2.append(1.0 - sse / sst if sst > _EPS else 0.0)
        heldout_mse.append(float(np.mean(np.square(difference))))
    rendered = np.rint(np.clip(predictions, 0.0, 1.0) * 255.0).astype(np.uint8)
    return PointwiseRgbProbe(
        target_rgb=target_rgb,
        predicted_rgb=rendered,
        train_indices=cast("ArrayI64", train),
        heldout_indices=cast("ArrayI64", heldout),
        heldout_r2=np.asarray(heldout_r2, dtype=np.float64),
        heldout_mse=np.asarray(heldout_mse, dtype=np.float64),
    )


def unpack_final_encoder_f1(hidden: Tensor) -> Tensor:
    """Unpack D-layout F1 copies as ``B x 48 x 2 x 32 x 32``."""
    if hidden.ndim != 4:
        message = f"expected BxCxHxW hidden fields, got {hidden.shape}"
        raise ValueError(message)
    expected_channels = D_LAYOUT.channels
    if hidden.shape[1] != expected_channels:
        message = (
            f"expected {expected_channels} D-layout channels, got {hidden.shape[1]}"
        )
        raise ValueError(message)
    f1 = hidden[:, D_LAYOUT.f1_offset :, :, :]
    return f1.reshape(hidden.shape[0], D_LAYOUT.n1, 2, hidden.shape[2], hidden.shape[3])


def disk_mean_f1(f1: Tensor, *, mask: Tensor) -> Tensor:
    """Pool each F1 copy uniformly over a centered disk, never over copies."""
    if f1.ndim != 5 or f1.shape[2] != 2:
        message = f"expected BxCopiesx2xHxW F1 fields, got {f1.shape}"
        raise ValueError(message)
    if mask.shape != f1.shape[-2:]:
        message = f"mask shape {mask.shape} does not match {f1.shape[-2:]}"
        raise ValueError(message)
    weights = mask.to(device=f1.device, dtype=f1.dtype)[None, None, None, :, :]
    return (f1 * weights).sum(dim=(-2, -1)) / weights.sum()


def disk_mean_scalar(hidden: Tensor, *, mask: Tensor) -> Tensor:
    """Pool scalar feature maps over the same disk used for F1 diagnostics."""
    if hidden.ndim != 4:
        message = f"expected BxCxHxW scalar fields, got {hidden.shape}"
        raise ValueError(message)
    if mask.shape != hidden.shape[-2:]:
        message = f"mask shape {mask.shape} does not match {hidden.shape[-2:]}"
        raise ValueError(message)
    weights = mask.to(device=hidden.device, dtype=hidden.dtype)[None, None, :, :]
    return (hidden * weights).sum(dim=(-2, -1)) / weights.sum()


def seeded_scalar_pair_projection(*, channels: int, pairs: int, seed: int) -> Tensor:
    """Make a fixed random orthonormal scalar-pair null basis on CPU."""
    if not 1 <= pairs * 2 <= channels:
        message = f"need 1..{channels // 2} pairs for {channels} channels, got {pairs}"
        raise ValueError(message)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    matrix = torch.randn((channels, channels), generator=generator)
    orthogonal, _ = torch.linalg.qr(matrix)
    return orthogonal[:, : pairs * 2].T.reshape(pairs, 2, channels)


def project_scalar_pair_means(means: ArrayF32, *, projections: ArrayF32) -> ArrayF32:
    """Apply an ensemble of scalar-pair null projections to disk means.

    ``means`` ends in scalar-channel dimension ``C``; ``projections`` has
    ``E x P x 2 x C``.  The result ends in ``E x P x 2``.
    """
    if means.ndim < 2:
        message = f"means must have leading samples and channels, got {means.shape}"
        raise ValueError(message)
    if projections.ndim != 4 or projections.shape[2] != 2:
        message = f"expected E x P x 2 x C projections, got {projections.shape}"
        raise ValueError(message)
    if means.shape[-1] != projections.shape[-1]:
        message = (
            f"mean channels {means.shape[-1]} do not match projection "
            f"channels {projections.shape[-1]}"
        )
        raise ValueError(message)
    projected = np.einsum(
        "...c,epkc->...epk",
        means.astype(np.float64),
        projections.astype(np.float64),
        optimize=True,
    )
    return cast("ArrayF32", projected.astype(np.float32))


def select_f1_copies(initial_means: ArrayF32, *, count: int = 3) -> ArrayI64:
    """Preselect the strongest copies from angle-zero fixed-25 disk means."""
    if initial_means.ndim != 3 or initial_means.shape[-1] != 2:
        message = f"expected patches x copies x 2 means, got {initial_means.shape}"
        raise ValueError(message)
    if not 1 <= count <= initial_means.shape[1]:
        message = f"count {count} is outside 1..{initial_means.shape[1]}"
        raise ValueError(message)
    magnitudes = np.linalg.norm(initial_means.astype(np.float64), axis=-1)
    medians = np.median(magnitudes, axis=0)
    selected = np.argsort(medians)[::-1][:count]
    return cast("ArrayI64", selected.astype(np.int64))


def normalized_f1_phase(trace: ArrayF32, *, min_magnitude: float) -> ArrayF64:
    """Normalize one F1 mean-vector trace by its recorded zero-degree phase."""
    if trace.ndim != 2 or trace.shape[1] != 2:
        message = f"expected angles x 2 trace, got {trace.shape}"
        raise ValueError(message)
    complex_trace = trace[:, 0].astype(np.float64) + 1j * trace[:, 1].astype(np.float64)
    initial = complex_trace[0]
    magnitude = abs(initial)
    if magnitude < min_magnitude:
        message = f"initial F1 magnitude {magnitude:.3e} is below {min_magnitude:.3e}"
        raise ValueError(message)
    return cast(
        "ArrayF64",
        np.column_stack((
            (complex_trace / initial).real,
            (complex_trace / initial).imag,
        )),
    )


def vector_phase_diagnostic(
    means: ArrayF32,
    *,
    selected_copies: ArrayI64,
    angles_degrees: ArrayI64,
    min_magnitude: float,
) -> VectorPhaseDiagnostic:
    """Separate phase error, gain, and raw magnitude for selected vectors."""
    if means.ndim != 3 or means.shape[-1] != 2:
        message = f"expected angles x copies x 2 means, got {means.shape}"
        raise ValueError(message)
    if means.shape[0] != angles_degrees.shape[0]:
        message = "vector means and angles must have the same number of rows"
        raise ValueError(message)
    if selected_copies.ndim != 1 or not len(selected_copies):
        raise ValueError("selected copies must be a nonempty one-dimensional array")
    if np.any(selected_copies < 0) or np.any(selected_copies >= means.shape[1]):
        raise ValueError("selected copy index is outside the available copies")
    selected = means[:, selected_copies, :].astype(np.float64)
    vectors = selected[:, :, 0] + 1j * selected[:, :, 1]
    initial = vectors[0]
    initial_magnitude = np.abs(initial)
    valid = initial_magnitude >= min_magnitude
    normalized = np.full(vectors.shape, np.nan + 1j * np.nan, dtype=np.complex128)
    normalized[:, valid] = vectors[:, valid] / initial[valid]
    expected_angle = np.deg2rad(angles_degrees.astype(np.float64))[:, None]
    phase_error = np.full(vectors.shape, np.nan, dtype=np.float64)
    phase_error[:, valid] = np.rad2deg(
        np.angle(np.exp(1j * (np.angle(normalized[:, valid]) - expected_angle)))
    )
    gain = np.abs(vectors) / (initial_magnitude[None, :] + _EPS)
    return VectorPhaseDiagnostic(
        normalized=cast(
            "ArrayF64", np.stack((normalized.real, normalized.imag), axis=-1)
        ),
        phase_error_degrees=cast("ArrayF64", phase_error),
        gain=cast("ArrayF64", gain),
        amplitude=cast("ArrayF64", np.abs(vectors)),
        valid=valid,
    )


def normal_pair_null_summary(
    pair_orbit: ArrayF32,
    *,
    selected_pairs: ArrayI64,
    angles_degrees: ArrayI64,
    min_magnitude: float,
) -> dict[str, ArrayF64 | int]:
    """Summarize an arbitrary scalar-pair ensemble without inventing F1 fields."""
    if pair_orbit.ndim != 4 or pair_orbit.shape[-1] != 2:
        message = f"expected angles x ensemble x pairs x 2, got {pair_orbit.shape}"
        raise ValueError(message)
    if selected_pairs.ndim != 2 or selected_pairs.shape[0] != pair_orbit.shape[1]:
        message = "selected pairs must be ensemble x selected-copies"
        raise ValueError(message)
    phase_errors: list[ArrayF64] = []
    gains: list[ArrayF64] = []
    amplitudes: list[ArrayF64] = []
    valid_controls = 0
    for ensemble_index, selected in enumerate(selected_pairs):
        diagnostic = vector_phase_diagnostic(
            pair_orbit[:, ensemble_index, :, :],
            selected_copies=selected,
            angles_degrees=angles_degrees,
            min_magnitude=min_magnitude,
        )
        if bool(np.all(diagnostic.valid)):
            valid_controls += 1
        phase_errors.append(
            np.nanmedian(np.abs(diagnostic.phase_error_degrees), axis=1)
        )
        gains.append(np.nanmedian(diagnostic.gain, axis=1))
        amplitudes.append(np.nanmedian(diagnostic.amplitude, axis=1))
    return {
        "phase_abs_error_quantiles": cast(
            "ArrayF64",
            np.nanquantile(np.stack(phase_errors), (0.05, 0.5, 0.95), axis=0),
        ),
        "gain_quantiles": cast(
            "ArrayF64", np.nanquantile(np.stack(gains), (0.05, 0.5, 0.95), axis=0)
        ),
        "amplitude_quantiles": cast(
            "ArrayF64", np.nanquantile(np.stack(amplitudes), (0.05, 0.5, 0.95), axis=0)
        ),
        "valid_controls": valid_controls,
    }


def choose_stem_f1_copy(kernel: Tensor) -> int:
    """Choose the largest-L2 learned scalar-to-F1 stem template reproducibly."""
    if kernel.ndim != 4 or kernel.shape[0] != A_LAYOUT.channels:
        message = f"expected stem kernel [48, 3, 9, 9], got {kernel.shape}"
        raise ValueError(message)
    f1 = kernel[A_LAYOUT.f1_offset :, :, :].reshape(
        A_LAYOUT.n1,
        2,
        *kernel.shape[1:],
    )
    energies = f1.square().sum(dim=(1, 2, 3, 4))
    return int(torch.argmax(energies).item())


def collect_orbit(
    *,
    model: torch.nn.Module,
    model_label: str,
    patch: Tensor,
    patch_index: int,
    angles_degrees: Sequence[int],
    collect_f1: bool,
    collect_scalar: bool = False,
) -> OrbitSweep:
    """Run one fixed patch through a predeclared dense continuous-angle sweep."""
    if patch.shape != (1, 3, 256, 256):
        message = f"expected one normalized 3x256x256 patch, got {patch.shape}"
        raise ValueError(message)
    if not angles_degrees or angles_degrees[0] != 0:
        raise ValueError("angles must be nonempty and start at zero")
    angles = np.asarray(angles_degrees, dtype=np.int64)
    latent_mask = centered_disk_mask(32, radius=LATENT_DISK_RADIUS, device=patch.device)
    image_mask = centered_disk_mask(256, radius=IMAGE_DISK_RADIUS, device=patch.device)
    mu_values: list[ArrayF32] = []
    residuals: list[float] = []
    roundtrip_floors: list[float] = []
    f1_means: list[ArrayF32] = []
    scalar_means: list[ArrayF32] = []
    base_mu: Tensor | None = None
    with torch.inference_mode():
        for angle in angles.tolist():
            rotated = continuous_rotate(patch, angle)
            if collect_f1 or collect_scalar:
                hidden = model._encode_features(rotated)  # pyright: ignore[reportPrivateUsage]
                mu = model.mu_head(hidden)  # pyright: ignore[reportPrivateUsage]
                if collect_f1:
                    f1 = unpack_final_encoder_f1(hidden)
                    f1_means.append(
                        disk_mean_f1(f1, mask=latent_mask)
                        .squeeze(0)
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
                if collect_scalar:
                    scalar_means.append(
                        disk_mean_scalar(hidden, mask=latent_mask)
                        .squeeze(0)
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
            else:
                encoded = model.encode(rotated)  # pyright: ignore[reportAttributeAccessIssue]
                mu = cast("Tensor", encoded[0])
            if base_mu is None:
                base_mu = mu
            expected = continuous_rotate(base_mu, angle)
            residual = masked_relative_rms(mu, expected, mask=latent_mask)
            roundtrip = continuous_rotate(rotated, -angle)
            floor = masked_relative_rms(roundtrip, patch, mask=image_mask)
            mu_values.append(mu.squeeze(0).cpu().numpy().astype(np.float32))
            residuals.append(float(residual.item()))
            roundtrip_floors.append(float(floor.item()))
    f1_array: ArrayF32 | None = None
    if collect_f1:
        f1_array = np.stack(f1_means).astype(np.float32)
    scalar_array: ArrayF32 | None = None
    if collect_scalar:
        scalar_array = np.stack(scalar_means).astype(np.float32)
    return OrbitSweep(
        model_label=model_label,
        patch_index=patch_index,
        angles_degrees=angles,
        mu=np.stack(mu_values).astype(np.float32),
        latent_residual=np.asarray(residuals, dtype=np.float64),
        input_roundtrip_floor=np.asarray(roundtrip_floors, dtype=np.float64),
        f1_disk_means=f1_array,
        scalar_disk_means=scalar_array,
    )


def quarter_residual_summary(
    *,
    saved_latent_path: Path,
    angles_degrees: Sequence[int] = (0, 90, 180, 270),
) -> QuarterResidualSummary:
    """Summarize the archived exact-quarter fixed-25 latent residuals."""
    raw = torch.load(saved_latent_path, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict):
        raise TypeError("saved fixed25 latent payload must be a dictionary")
    payload = cast("dict[str, Tensor]", raw)
    base = payload.get("mu_clean")
    if base is None or base.shape != (25, 16, 32, 32):
        raise ValueError("saved fixed25 mu_clean must have shape [25, 16, 32, 32]")
    mask = centered_disk_mask(32, radius=LATENT_DISK_RADIUS)
    all_values: list[Tensor] = []
    for angle in angles_degrees:
        if angle == 0:
            observed = base
            expected = base
        else:
            observed_key = f"mu_of_rotated_input_{angle}"
            expected_key = f"rotated_latent_of_mu_{angle}"
            observed = payload.get(observed_key)
            expected = payload.get(expected_key)
            if observed is None or expected is None:
                raise ValueError(f"missing archived fixed25 angle {angle}")
        all_values.append(masked_relative_rms(observed, expected, mask=mask).cpu())
    matrix = torch.stack(all_values).numpy().astype(np.float64)
    return QuarterResidualSummary(
        angles_degrees=np.asarray(angles_degrees, dtype=np.int64),
        median=np.median(matrix, axis=1),
        lower_quartile=np.quantile(matrix, 0.25, axis=1),
        upper_quartile=np.quantile(matrix, 0.75, axis=1),
    )


def render_latent_orbits_png(
    *,
    path: Path,
    baseline: OrbitSweep,
    so2: OrbitSweep,
    baseline_quarters: QuarterResidualSummary,
    so2_quarters: QuarterResidualSummary,
    original_patch: Tensor,
) -> None:
    """Render the primary compact latent-orbit PNG with only standard Pillow."""
    canvas = Image.new("RGB", (1800, 1120), "#fcfcfc")
    draw = ImageDraw.Draw(canvas)
    _title(draw, "Rotation response of the frozen spatial posterior mean mu")
    _subtitle(draw, f"Dense 1° orbit: fixed validation patch {baseline.patch_index}")
    _orbit_panel(draw, (70, 145, 820, 760), baseline, "Normal VAE")
    _orbit_panel(draw, (900, 145, 1650, 760), so2, "SO(2) VAE")
    _line_panel(
        draw,
        (70, 805, 940, 1030),
        (
            ("normal (one patch)", baseline.latent_residual, "#e63946"),
            ("SO(2) (one patch)", so2.latent_residual, "#3a86ff"),
        ),
        baseline.angles_degrees,
        "Disk-masked mu residual across the dense orbit",
        "relative RMS",
    )
    _quarter_panel(
        draw,
        (990, 805, 1650, 1030),
        baseline_quarters,
        so2_quarters,
        "All fixed-25, archived exact quarter turns",
    )
    _paste_patch(canvas, original_patch, (1660, 190), side=110)
    _footer(draw, canvas.height)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def render_dense_orbit_population_png(
    *,
    path: Path,
    normal: DenseOrbitPopulation,
    so2: DenseOrbitPopulation,
) -> None:
    """Render paired dense PCA traces and all-patch regularity summaries."""
    if normal.pca_scores.shape != so2.pca_scores.shape:
        raise ValueError("normal and SO(2) population traces must have the same shape")
    if normal.pca_scores.ndim != 3 or normal.pca_scores.shape[2] != 2:
        raise ValueError("population PCA scores must have shape NxTx2")
    if normal.pca_scores.shape[0] != 25:
        raise ValueError("the advisor population figure requires all fixed 25 patches")
    if not np.array_equal(normal.angles_degrees, so2.angles_degrees):
        raise ValueError("normal and SO(2) population angles must match")
    canvas = Image.new("RGB", (3000, 2400), "#fcfcfc")
    draw = ImageDraw.Draw(canvas)
    _title(draw, "Rotation-orbit geometry across all 25 fixed validation patches")
    step = int(normal.angles_degrees[1] - normal.angles_degrees[0])
    _subtitle(
        draw,
        f"Full 0°--360° cycle sampled every {step}°; each mini-panel fits one PCA per model/patch, "
        "uses equal axis scale, and reconnects the final sample to 360° = 0°.",
    )
    draw.text((70, 132), "Normal VAE", fill="#2478d3", font=_font(22))
    draw.text((330, 132), "SO(2) VAE", fill="#df7a27", font=_font(22))
    for patch_index in range(25):
        row, column = divmod(patch_index, 5)
        group_left = 65 + column * 585
        top = 170 + row * 330
        _mini_orbit_panel(
            draw,
            (group_left, top, group_left + 250, top + 285),
            normal.pca_scores[patch_index],
            title=f"patch {patch_index} · normal",
            color="#2478d3",
            angles=normal.angles_degrees,
        )
        _mini_orbit_panel(
            draw,
            (group_left + 270, top, group_left + 520, top + 285),
            so2.pca_scores[patch_index],
            title=f"patch {patch_index} · SO(2)",
            color="#df7a27",
            angles=so2.angles_degrees,
        )
    summary_top = 1845
    _paired_metric_panel(
        draw,
        (65, summary_top, 1000, 2295),
        normal.local_linearity_ratio,
        so2.local_linearity_ratio,
        title="Local linearity error (lower = smoother)",
        value_format=".3f",
    )
    _paired_metric_panel(
        draw,
        (1030, summary_top, 1965, 2295),
        normal.step_size_cv,
        so2.step_size_cv,
        title="One-step speed CV (lower = more uniform)",
        value_format=".3f",
    )
    _paired_metric_panel(
        draw,
        (1995, summary_top, 2930, 2295),
        normal.pca_explained_variance,
        so2.pca_explained_variance,
        title="Variance in first two PCs (higher = more planar)",
        value_format=".1%",
    )
    _footer(draw, canvas.height)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def render_spatial_latent_pca_png(
    *,
    path: Path,
    originals: Tensor,
    display_indices: Sequence[int],
    normal: SpatialLatentPcaDiagnostic,
    so2: SpatialLatentPcaDiagnostic,
) -> None:
    """Render fixed-25 spatial PCA-RGB maps and an all-patch paired summary."""
    if originals.shape != (25, 3, 256, 256) or originals.dtype != torch.uint8:
        message = (
            f"expected fixed25 uint8 originals, got {originals.shape}/{originals.dtype}"
        )
        raise ValueError(message)
    if normal.rgb.shape != so2.rgb.shape or normal.rgb.shape[:3] != (25, 32, 32):
        message = "normal and SO(2) spatial PCA maps must both be [25,32,32,3]"
        raise ValueError(message)
    if any(index < 0 or index >= originals.shape[0] for index in display_indices):
        raise ValueError("spatial PCA display index is outside fixed25")
    canvas = Image.new("RGB", (2400, 1500), "#fcfcfc")
    draw = ImageDraw.Draw(canvas)
    _title(draw, "EQ-VAE-inspired spatial latent PCA-RGB diagnostic")
    _subtitle(
        draw,
        "One disk-masked PCA fit and one RGB score scale per model over all 25 fixed patches; "
        "colours are within-model only, and native 32x32 cells are nearest-neighbour displayed.",
    )
    columns = ((70, "input"), (370, "normal VAE PCA-RGB"), (670, "SO(2) VAE PCA-RGB"))
    for x, label in columns:
        draw.text((x, 135), label, fill="#111827", font=_font(20))
    side = 225
    for row, index in enumerate(display_indices):
        y = 170 + row * 265
        draw.text((25, y + 94), f"patch {index}", fill="#4b5563", font=_font(17))
        _paste_uint8_patch(canvas, originals[index : index + 1], (70, y), side=side)
        _paste_spatial_pca_map(canvas, normal.rgb[index], (370, y), side=side)
        _paste_spatial_pca_map(canvas, so2.rgb[index], (670, y), side=side)
    _paired_coherence_panel(
        draw,
        (1000, 145, 2315, 740),
        normal.relative_edge_rms,
        so2.relative_edge_rms,
    )
    _delta_coherence_panel(
        draw,
        (1000, 790, 2315, 1215),
        so2.relative_edge_rms - normal.relative_edge_rms,
    )
    _multiline(
        draw,
        1000,
        1250,
        [
            _coherence_summary_line("normal", normal),
            _coherence_summary_line("SO(2)", so2),
            "r_edge = edge RMS / centred spatial RMS; lower is locally less rough, not automatically better.",
            "Neutral exterior = excluded disk. PCA-RGB is qualitative; paired r_edge is the shared numerical companion.",
        ],
        fill="#374151",
        font=_font(15),
    )
    _footer(draw, canvas.height)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def render_paper_style_spatial_pca_png(
    *,
    path: Path,
    originals: Tensor,
    display_indices: Sequence[int],
    normal: PaperStyleSpatialPca,
    so2: PaperStyleSpatialPca,
) -> None:
    """Render the per-image EQ-VAE-style PCA view without inventing a blend."""
    if originals.shape != (25, 3, 256, 256) or originals.dtype != torch.uint8:
        message = (
            f"expected fixed25 uint8 originals, got {originals.shape}/{originals.dtype}"
        )
        raise ValueError(message)
    if normal.rgb.shape != so2.rgb.shape or normal.rgb.shape[:3] != (25, 32, 32):
        message = "normal and SO(2) paper-style maps must both be [25,32,32,3]"
        raise ValueError(message)
    if any(index < 0 or index >= originals.shape[0] for index in display_indices):
        raise ValueError("paper-style PCA display index is outside fixed25")
    canvas = Image.new("RGB", (1800, 1500), "#fcfcfc")
    draw = ImageDraw.Draw(canvas)
    _title(draw, "Paper-style per-patch spatial latent PCA-RGB")
    _subtitle(
        draw,
        "Each tile independently PCA-fits its whole 32x32 posterior grid, jointly min-maxes its three scores, "
        "and bilinearly upscales them. The input is a separate tile: there is no image blend.",
    )
    columns = (
        (165, "original input"),
        (665, "normal VAE PCA-RGB"),
        (1165, "SO(2) VAE PCA-RGB"),
    )
    for x, label in columns:
        draw.text((x, 135), label, fill="#111827", font=_font(20))
    side = 250
    for row, index in enumerate(display_indices):
        y = 175 + row * 285
        draw.text((60, y + 108), f"patch {index}", fill="#4b5563", font=_font(18))
        _paste_uint8_patch(canvas, originals[index : index + 1], (165, y), side=side)
        _paste_paper_style_pca_map(canvas, normal.rgb[index], (665, y), side=side)
        _paste_paper_style_pca_map(canvas, so2.rgb[index], (1165, y), side=side)
    _multiline(
        draw,
        165,
        1345,
        [
            "Visual convention only: hue, brightness, PCA directions, and numeric ranges are re-fit separately in every map.",
            "Do not compare colour or apparent smoothness across tiles; use 04-spatial-latent-pca.png and r_edge for the shared-scale companion.",
        ],
        fill="#374151",
        font=_font(16),
    )
    _footer(draw, canvas.height)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def render_pointwise_rgb_probe_png(
    *,
    path: Path,
    normal: PointwiseRgbProbe,
    so2: PointwiseRgbProbe,
) -> None:
    """Render held-out pointwise RGB predictions without overstating the probe."""
    if not np.array_equal(
        normal.train_indices, so2.train_indices
    ) or not np.array_equal(
        normal.heldout_indices,
        so2.heldout_indices,
    ):
        raise ValueError("normal and SO(2) pointwise probes must use the same split")
    if normal.target_rgb.shape != so2.target_rgb.shape or not np.array_equal(
        normal.target_rgb,
        so2.target_rgb,
    ):
        raise ValueError("normal and SO(2) pointwise probes must use the same targets")
    canvas = Image.new("RGB", (1800, 1800), "#fcfcfc")
    draw = ImageDraw.Draw(canvas)
    _title(draw, "Held-out pointwise latent-to-RGB probe")
    _subtitle(
        draw,
        "One affine 16-to-3 map per model, fit once on fixed patches 0--19; rows 20--24 are held out. "
        "No spatial neighbours, decoder, or additional images are used.",
    )
    columns = (
        (165, "held-out source at 32x32"),
        (665, "normal VAE: W mu(p)+b"),
        (1165, "SO(2) VAE: W mu(p)+b"),
    )
    for x, label in columns:
        draw.text((x, 135), label, fill="#111827", font=_font(20))
    side = 250
    for row, index in enumerate(normal.heldout_indices):
        y = 170 + row * 295
        draw.text((48, y + 108), f"patch {int(index)}", fill="#4b5563", font=_font(18))
        _paste_paper_style_pca_map(
            canvas, normal.target_rgb[index], (165, y), side=side
        )
        _paste_paper_style_pca_map(
            canvas, normal.predicted_rgb[index], (665, y), side=side
        )
        _paste_paper_style_pca_map(
            canvas, so2.predicted_rgb[index], (1165, y), side=side
        )
        position = row
        draw.text(
            (665, y + side + 9),
            f"held-out pixel R2={normal.heldout_r2[position]:.3f}; MSE={normal.heldout_mse[position]:.4f}",
            fill="#c23e50",
            font=_font(14),
        )
        draw.text(
            (1165, y + side + 9),
            f"held-out pixel R2={so2.heldout_r2[position]:.3f}; MSE={so2.heldout_mse[position]:.4f}",
            fill="#2478d3",
            font=_font(14),
        )
    _multiline(
        draw,
        165,
        1662,
        [
            f"Mean held-out pixel R2 versus training RGB-mean baseline: normal {np.mean(normal.heldout_r2):.3f}; SO(2) {np.mean(so2.heldout_r2):.3f}.",
            "This measures coarse local appearance retained in mu, not equivariance or VAE reconstruction quality.",
        ],
        fill="#374151",
        font=_font(16),
    )
    _footer(draw, canvas.height)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def render_f1_phase_png(
    *,
    path: Path,
    trained_sweep: OrbitSweep,
    untrained_sweep: OrbitSweep,
    trained_selected_copies: ArrayI64,
    normal_pair_orbit: ArrayF32,
    normal_selected_pairs: ArrayI64,
    min_magnitude: float,
) -> None:
    """Render trained-F1 diagnostics with clearly separate control roles."""
    if trained_sweep.f1_disk_means is None or untrained_sweep.f1_disk_means is None:
        raise ValueError("F1 phase rendering requires both SO(2) F1 orbits")
    trained = vector_phase_diagnostic(
        trained_sweep.f1_disk_means,
        selected_copies=trained_selected_copies,
        angles_degrees=trained_sweep.angles_degrees,
        min_magnitude=min_magnitude,
    )
    untrained = vector_phase_diagnostic(
        untrained_sweep.f1_disk_means,
        selected_copies=trained_selected_copies,
        angles_degrees=untrained_sweep.angles_degrees,
        min_magnitude=min_magnitude,
    )
    null = normal_pair_null_summary(
        normal_pair_orbit,
        selected_pairs=normal_selected_pairs,
        angles_degrees=trained_sweep.angles_degrees,
        min_magnitude=min_magnitude,
    )
    canvas = Image.new("RGB", (2400, 1520), "#fcfcfc")
    draw = ImageDraw.Draw(canvas)
    _title(draw, "Internal two-vector phase diagnostic and explicit controls")
    _subtitle(
        draw,
        "Trained and seeded-untrained SO(2) panels use the same preselected F1 copies; "
        "the normal-VAE trace is a 32-projection scalar-pair null, never normal F1; "
        f"dense 1° orbit for fixed validation patch {trained_sweep.patch_index}",
    )
    _phase_diagnostic_panel(
        draw,
        (70, 150, 800, 760),
        trained,
        "Trained SO(2) F1: phase-normalized disk means",
        tuple(f"F1 copy {copy}" for copy in trained_selected_copies.tolist()),
    )
    _phase_diagnostic_panel(
        draw,
        (835, 150, 1565, 760),
        untrained,
        "Seeded untrained SO(2): same F1 copy indices",
        tuple(f"F1 copy {copy}" for copy in trained_selected_copies.tolist()),
    )
    _line_panel(
        draw,
        (1600, 150, 2330, 760),
        (
            (
                "normal null p05",
                cast("ArrayF64", null["phase_abs_error_quantiles"])[0],
                "#b6becb",
            ),
            (
                "normal null median",
                cast("ArrayF64", null["phase_abs_error_quantiles"])[1],
                "#6b7280",
            ),
            (
                "normal null p95",
                cast("ArrayF64", null["phase_abs_error_quantiles"])[2],
                "#b6becb",
            ),
        ),
        trained_sweep.angles_degrees,
        "Normal VAE scalar-pair null: |phase error| band",
        "degrees",
    )
    _line_panel(
        draw,
        (70, 825, 800, 1375),
        (
            (
                "trained SO(2) median",
                cast(
                    "ArrayF64",
                    np.nanmedian(np.abs(trained.phase_error_degrees), axis=1),
                ),
                "#2478d3",
            ),
            (
                "untrained SO(2) median",
                cast(
                    "ArrayF64",
                    np.nanmedian(np.abs(untrained.phase_error_degrees), axis=1),
                ),
                "#8b5cf6",
            ),
            (
                "normal null median",
                cast("ArrayF64", null["phase_abs_error_quantiles"])[1],
                "#6b7280",
            ),
        ),
        trained_sweep.angles_degrees,
        "Median absolute phase error: selected copies / null ensemble",
        "degrees",
    )
    _line_panel(
        draw,
        (835, 825, 1565, 1375),
        (
            (
                "trained SO(2) median",
                cast("ArrayF64", np.nanmedian(trained.gain, axis=1)),
                "#2478d3",
            ),
            (
                "untrained SO(2) median",
                cast("ArrayF64", np.nanmedian(untrained.gain, axis=1)),
                "#8b5cf6",
            ),
            (
                "normal null median",
                cast("ArrayF64", null["gain_quantiles"])[1],
                "#6b7280",
            ),
        ),
        trained_sweep.angles_degrees,
        "Mean-vector gain |z(theta)| / |z(0)|",
        "ratio",
    )
    _line_panel(
        draw,
        (1600, 825, 2330, 1375),
        (
            (
                "trained SO(2) median",
                cast("ArrayF64", np.nanmedian(trained.amplitude, axis=1)),
                "#2478d3",
            ),
            (
                "untrained SO(2) median",
                cast("ArrayF64", np.nanmedian(untrained.amplitude, axis=1)),
                "#8b5cf6",
            ),
            (
                "normal null median",
                cast("ArrayF64", null["amplitude_quantiles"])[1],
                "#6b7280",
            ),
        ),
        trained_sweep.angles_degrees,
        "Raw disk-mean vector magnitude (scale exposed)",
        "magnitude",
    )
    _multiline(
        draw,
        80,
        1410,
        [
            "Trained copies: top three by median theta=0 disk-mean magnitude over all 25 fixed patches; the untrained panel reuses those indices.",
            f"Normal null: 32 seeded orthonormal scalar pairings, with its own preselected copies; near-zero threshold {min_magnitude:.1e}.",
        ],
        fill="#404040",
    )
    _footer(draw, canvas.height)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def render_kernel_mechanism_png(*, path: Path, kernel: Tensor) -> int:
    """Render the learned stem F1 template as an architectural-mechanism inset."""
    copy = choose_stem_f1_copy(kernel)
    f1 = kernel[16 + 2 * copy : 16 + 2 * copy + 2].detach().cpu().numpy()
    canvas = Image.new("RGB", (1800, 1120), "#fcfcfc")
    draw = ImageDraw.Draw(canvas)
    _title(draw, "Learned scalar-to-F1 stem template: architectural mechanism")
    _subtitle(
        draw,
        f"Largest-L2 F1 copy {copy}; this constrained kernel is jointly rotation-invariant by construction, not an empirical orbit.",
    )
    shared_maximum = float(np.max(np.sqrt(np.square(f1).sum(axis=0)))) or 1.0
    for channel, name in enumerate(("red input", "green input", "blue input")):
        _kernel_vector_panel(
            draw,
            (100 + channel * 530, 170, 570 + channel * 530, 790),
            f1[:, channel, :, :],
            name,
            shared_maximum=shared_maximum,
        )
    magnitudes = np.sqrt(np.square(f1).sum(axis=(0, 2, 3)))
    _bar_panel(
        draw,
        (330, 870, 1470, 1010),
        magnitudes,
        ("red", "green", "blue"),
        "Per-input-color F1 kernel L2 norm",
    )
    _footer(draw, canvas.height)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return copy


def write_offline_html(
    *,
    path: Path,
    baseline: OrbitSweep,
    so2: OrbitSweep,
    untrained_so2: OrbitSweep,
    baseline_quarters: QuarterResidualSummary,
    so2_quarters: QuarterResidualSummary,
    selected_copies: ArrayI64,
    normal_pair_orbit: ArrayF32,
    normal_selected_pairs: ArrayI64,
    min_magnitude: float,
    kernel: Tensor,
    normal_spatial: SpatialLatentPcaDiagnostic,
    so2_spatial: SpatialLatentPcaDiagnostic,
    normal_paper_style: PaperStyleSpatialPca,
    so2_paper_style: PaperStyleSpatialPca,
    normal_pointwise_probe: PointwiseRgbProbe,
    so2_pointwise_probe: PointwiseRgbProbe,
    spatial_originals: Tensor,
    spatial_display_indices: Sequence[int],
) -> None:
    """Write a self-contained advisor page from already-precomputed SVG data."""
    if so2.f1_disk_means is None or untrained_so2.f1_disk_means is None:
        raise ValueError(
            "HTML F1 diagnostics require trained and untrained SO(2) means"
        )
    mask = centered_disk_mask(32, radius=LATENT_DISK_RADIUS)
    baseline_pca = pca_orbit_2d(
        baseline.mu,
        mask=mask,
    )
    so2_pca = pca_orbit_2d(
        so2.mu,
        mask=mask,
    )
    disk = mask.cpu().numpy().astype(bool)
    trained_phase = vector_phase_diagnostic(
        so2.f1_disk_means,
        selected_copies=selected_copies,
        angles_degrees=so2.angles_degrees,
        min_magnitude=min_magnitude,
    )
    untrained_phase = vector_phase_diagnostic(
        untrained_so2.f1_disk_means,
        selected_copies=selected_copies,
        angles_degrees=untrained_so2.angles_degrees,
        min_magnitude=min_magnitude,
    )
    null_summary = normal_pair_null_summary(
        normal_pair_orbit,
        selected_pairs=normal_selected_pairs,
        angles_degrees=baseline.angles_degrees,
        min_magnitude=min_magnitude,
    )
    kernel_copy = choose_stem_f1_copy(kernel)
    kernel_f1 = (
        kernel[
            A_LAYOUT.f1_offset + 2 * kernel_copy : A_LAYOUT.f1_offset
            + 2 * kernel_copy
            + 2
        ]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    if tuple(normal_spatial.rgb.shape) != (25, 32, 32, 3):
        raise ValueError("normal spatial PCA RGB must have shape [25,32,32,3]")
    if tuple(so2_spatial.rgb.shape) != (25, 32, 32, 3):
        raise ValueError("SO(2) spatial PCA RGB must have shape [25,32,32,3]")
    if tuple(normal_paper_style.rgb.shape) != (25, 32, 32, 3):
        raise ValueError("normal paper-style PCA RGB must have shape [25,32,32,3]")
    if tuple(so2_paper_style.rgb.shape) != (25, 32, 32, 3):
        raise ValueError("SO(2) paper-style PCA RGB must have shape [25,32,32,3]")
    pointwise_payload = _pointwise_rgb_probe_payload(
        normal=normal_pointwise_probe,
        so2=so2_pointwise_probe,
    )
    if (
        spatial_originals.shape != (25, 3, 256, 256)
        or spatial_originals.dtype != torch.uint8
    ):
        raise ValueError("spatial originals must be fixed25 uint8 RGB patches")
    if any(index < 0 or index >= 25 for index in spatial_display_indices):
        raise ValueError("spatial PCA display index is outside fixed25")
    payload: dict[str, object] = {
        "label": EXPLORATORY_LABEL,
        "patch_index": baseline.patch_index,
        "angles": baseline.angles_degrees.tolist(),
        "models": {
            "Normal VAE": {
                "pca": baseline_pca.tolist(),
                "residual": baseline.latent_residual.tolist(),
                "posterior_rms": _masked_posterior_rms(baseline.mu, mask=disk).tolist(),
                "pca_explained_variance": pca_orbit_explained_variance(
                    baseline.mu, mask=mask
                ),
            },
            "SO(2) VAE": {
                "pca": so2_pca.tolist(),
                "residual": so2.latent_residual.tolist(),
                "posterior_rms": _masked_posterior_rms(so2.mu, mask=disk).tolist(),
                "pca_explained_variance": pca_orbit_explained_variance(
                    so2.mu, mask=mask
                ),
            },
        },
        "quarters": {
            "angles": baseline_quarters.angles_degrees.tolist(),
            "normal": baseline_quarters.median.tolist(),
            "so2": so2_quarters.median.tolist(),
        },
        "selected_f1_copies": selected_copies.tolist(),
        "min_magnitude": min_magnitude,
        "f1": {
            "trained": _phase_payload(trained_phase),
            "untrained": _phase_payload(untrained_phase),
            "normal_null": {
                "phase_abs_error_quantiles": cast(
                    "ArrayF64", null_summary["phase_abs_error_quantiles"]
                ).tolist(),
                "gain_quantiles": cast(
                    "ArrayF64", null_summary["gain_quantiles"]
                ).tolist(),
                "amplitude_quantiles": cast(
                    "ArrayF64", null_summary["amplitude_quantiles"]
                ).tolist(),
                "valid_controls": int(null_summary["valid_controls"]),
                "ensemble_size": int(normal_pair_orbit.shape[1]),
            },
        },
        "kernel": {"copy": kernel_copy, "vectors": kernel_f1.tolist()},
        "spatial_pca": {
            "display_indices": list(spatial_display_indices),
            "Normal VAE": _spatial_pca_payload(normal_spatial, spatial_display_indices),
            "SO(2) VAE": _spatial_pca_payload(so2_spatial, spatial_display_indices),
            "paper_style": {
                "originals_rgb": _spatial_thumbnails(
                    spatial_originals,
                    spatial_display_indices,
                ).tolist(),
                "Normal VAE": _paper_style_pca_payload(
                    normal_paper_style,
                    spatial_display_indices,
                ),
                "SO(2) VAE": _paper_style_pca_payload(
                    so2_paper_style,
                    spatial_display_indices,
                ),
            },
        },
        "pointwise_rgb_probe": pointwise_payload,
    }
    document = _advisor_html_document(
        json.dumps(payload, separators=(",", ":")),
        max_angle=int(baseline.angles_degrees[-1]),
    )
    if "http://" in document or "https://" in document:
        raise ValueError("offline HTML must not contain remote URLs")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")


def _masked_posterior_rms(values: ArrayF32, *, mask: NDArray[np.bool_]) -> ArrayF64:
    """Return raw posterior RMS over the same disk as the orbit metric."""
    selected = values[:, :, mask].astype(np.float64)
    return cast("ArrayF64", np.sqrt(np.mean(np.square(selected), axis=(1, 2))))


def _phase_payload(diagnostic: VectorPhaseDiagnostic) -> dict[str, object]:
    """Serialize a phase diagnostic using only browser-native numeric arrays."""
    return {
        "normalized": diagnostic.normalized.tolist(),
        "phase_error_degrees": diagnostic.phase_error_degrees.tolist(),
        "gain": diagnostic.gain.tolist(),
        "amplitude": diagnostic.amplitude.tolist(),
        "valid": diagnostic.valid.tolist(),
    }


def _spatial_pca_payload(
    diagnostic: SpatialLatentPcaDiagnostic,
    display_indices: Sequence[int],
) -> dict[str, object]:
    """Serialize compact selected maps plus all-fixed25 scale-aware statistics."""
    return {
        "rgb": diagnostic.rgb[list(display_indices)].tolist(),
        "explained_variance": diagnostic.explained_variance.tolist(),
        "rgb_score_scale": diagnostic.rgb_score_scale,
        "edge_rms": diagnostic.edge_rms.tolist(),
        "centered_spatial_rms": diagnostic.centered_spatial_rms.tolist(),
        "posterior_rms": diagnostic.posterior_rms.tolist(),
        "relative_edge_rms": diagnostic.relative_edge_rms.tolist(),
        "degenerate": diagnostic.degenerate.tolist(),
    }


def _paper_style_pca_payload(
    diagnostic: PaperStyleSpatialPca,
    display_indices: Sequence[int],
) -> dict[str, object]:
    """Serialize visual-only per-image PCA maps with their own score ranges."""
    selected = list(display_indices)
    return {
        "rgb": diagnostic.rgb[selected].tolist(),
        "explained_variance": diagnostic.explained_variance[selected].tolist(),
        "rgb_min": diagnostic.rgb_min[selected].tolist(),
        "rgb_max": diagnostic.rgb_max[selected].tolist(),
        "degenerate": diagnostic.degenerate[selected].tolist(),
    }


def _pointwise_rgb_probe_payload(
    *,
    normal: PointwiseRgbProbe,
    so2: PointwiseRgbProbe,
) -> dict[str, object]:
    """Serialize only held-out pointwise readout maps for the offline page."""
    expected_shape = (25, 32, 32, 3)
    if (
        normal.target_rgb.shape != expected_shape
        or so2.target_rgb.shape != expected_shape
    ):
        raise ValueError("pointwise RGB targets must have shape [25,32,32,3]")
    if (
        normal.predicted_rgb.shape != expected_shape
        or so2.predicted_rgb.shape != expected_shape
    ):
        raise ValueError("pointwise RGB predictions must have shape [25,32,32,3]")
    if not np.array_equal(normal.target_rgb, so2.target_rgb):
        raise ValueError("normal and SO(2) pointwise RGB targets must match")
    if not np.array_equal(
        normal.train_indices, so2.train_indices
    ) or not np.array_equal(
        normal.heldout_indices,
        so2.heldout_indices,
    ):
        raise ValueError("normal and SO(2) pointwise RGB splits must match")
    heldout = normal.heldout_indices
    if heldout.shape != (5,):
        raise ValueError("pointwise RGB page requires exactly five held-out patches")
    return {
        "train_indices": normal.train_indices.tolist(),
        "heldout_indices": heldout.tolist(),
        "target_rgb": normal.target_rgb[heldout].tolist(),
        "Normal VAE": {
            "predicted_rgb": normal.predicted_rgb[heldout].tolist(),
            "heldout_r2": normal.heldout_r2.tolist(),
            "heldout_mse": normal.heldout_mse.tolist(),
        },
        "SO(2) VAE": {
            "predicted_rgb": so2.predicted_rgb[heldout].tolist(),
            "heldout_r2": so2.heldout_r2.tolist(),
            "heldout_mse": so2.heldout_mse.tolist(),
        },
    }


def _spatial_thumbnails(originals: Tensor, display_indices: Sequence[int]) -> ArrayU8:
    """Downsample selected source patches for an offline, array-drawn HTML view."""
    thumbs = []
    for index in display_indices:
        pixels = originals[index].detach().cpu().permute(1, 2, 0).numpy()
        thumb = Image.fromarray(pixels).resize((64, 64), Image.Resampling.LANCZOS)
        thumbs.append(np.asarray(thumb, dtype=np.uint8))
    return cast("ArrayU8", np.stack(thumbs, axis=0))


def _title(draw: ImageDraw.ImageDraw, text: str) -> None:
    draw.text((55, 35), text, fill="#151515", font=_font(34))


def _subtitle(draw: ImageDraw.ImageDraw, text: str) -> None:
    _multiline(draw, 58, 86, [text], fill="#4b5563", font=_font(18))


def _footer(draw: ImageDraw.ImageDraw, height: int) -> None:
    draw.text((55, height - 42), EXPLORATORY_LABEL, fill="#4b5563", font=_font(16))


def _font(size: int) -> ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


def _multiline(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    lines: Sequence[str],
    *,
    fill: str,
    font: ImageFont.ImageFont | None = None,
) -> None:
    active_font = font or _font(18)
    for offset, line in enumerate(lines):
        draw.text((x, y + offset * 26), line, fill=fill, font=active_font)


def _orbit_panel(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    sweep: OrbitSweep,
    title: str,
) -> None:
    pca = pca_orbit_2d(sweep.mu, mask=centered_disk_mask(32, radius=LATENT_DISK_RADIUS))
    _scatter_panel(draw, rect, pca, title, "PC 1", "PC 2", "#3a86ff")


def _mini_orbit_panel(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    points: ArrayF64,
    *,
    title: str,
    color: str,
    angles: ArrayI64,
) -> None:
    left, top, right, bottom = rect
    draw.rounded_rectangle(rect, radius=10, fill="#ffffff", outline="#d1d5db", width=2)
    draw.text((left + 10, top + 8), title, fill=color, font=_font(15))
    plot = (left + 15, top + 34, right - 15, bottom - 14)
    x_min, x_max, y_min, y_max = isotropic_plot_bounds(
        points[:, 0],
        points[:, 1],
        plot_width=plot[2] - plot[0],
        plot_height=plot[3] - plot[1],
    )
    coordinates = [
        _project(float(x), float(y), plot, x_min, x_max, y_min, y_max)
        for x, y in points
    ]
    draw.line(coordinates + [coordinates[0]], fill=color, width=3)
    for angle in (0, 90, 180, 270):
        matches = np.flatnonzero(angles == angle)
        if matches.size == 0:
            continue
        point = coordinates[int(matches[0])]
        radius = 5 if angle == 0 else 3
        fill = "#111827" if angle == 0 else "#ffffff"
        draw.ellipse(
            (
                point[0] - radius,
                point[1] - radius,
                point[0] + radius,
                point[1] + radius,
            ),
            fill=fill,
            outline=color,
            width=2,
        )


def _paired_metric_panel(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    normal: ArrayF64,
    so2: ArrayF64,
    *,
    title: str,
    value_format: str,
) -> None:
    if normal.shape != (25,) or so2.shape != (25,):
        raise ValueError("paired population metrics must contain exactly 25 values")
    left, top, right, bottom = rect
    draw.rounded_rectangle(rect, radius=14, fill="#ffffff", outline="#d1d5db", width=2)
    draw.text((left + 18, top + 15), title, fill="#111827", font=_font(20))
    plot = (left + 70, top + 58, right - 45, bottom - 95)
    y_min, y_max = _range_with_padding(np.concatenate((normal, so2)))
    normal_x = plot[0] + int(0.30 * (plot[2] - plot[0]))
    so2_x = plot[0] + int(0.70 * (plot[2] - plot[0]))
    draw.line((plot[0], plot[3], plot[2], plot[3]), fill="#9ca3af", width=2)
    for value, paired in zip(normal, so2, strict=True):
        normal_y = _project(0.0, float(value), plot, 0.0, 1.0, y_min, y_max)[1]
        so2_y = _project(1.0, float(paired), plot, 0.0, 1.0, y_min, y_max)[1]
        draw.line((normal_x, normal_y, so2_x, so2_y), fill="#d1d5db", width=2)
        draw.ellipse(
            (normal_x - 4, normal_y - 4, normal_x + 4, normal_y + 4),
            fill="#2478d3",
        )
        draw.ellipse(
            (so2_x - 4, so2_y - 4, so2_x + 4, so2_y + 4),
            fill="#df7a27",
        )
    for x, values, color, label in (
        (normal_x, normal, "#2478d3", "normal"),
        (so2_x, so2, "#df7a27", "SO(2)"),
    ):
        median_y = _project(
            0.0,
            float(np.median(values)),
            plot,
            0.0,
            1.0,
            y_min,
            y_max,
        )[1]
        draw.line((x - 22, median_y, x + 22, median_y), fill=color, width=6)
        draw.text((x - 35, plot[3] + 12), label, fill=color, font=_font(16))
    better = (
        int(np.sum(so2 < normal)) if "lower" in title else int(np.sum(so2 > normal))
    )
    summary = (
        f"median normal={format(float(np.median(normal)), value_format)} · "
        f"SO(2)={format(float(np.median(so2)), value_format)} · "
        f"SO(2) favorable in {better}/25"
    )
    draw.text((left + 18, bottom - 47), summary, fill="#374151", font=_font(15))


def _scatter_panel(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    points: ArrayF64,
    title: str,
    x_label: str,
    y_label: str,
    color: str,
) -> None:
    left, top, right, bottom = rect
    draw.rounded_rectangle(rect, radius=16, fill="#ffffff", outline="#d1d5db", width=2)
    draw.text((left + 20, top + 18), title, fill="#111827", font=_font(24))
    pad = 70
    plot = (left + pad, top + 75, right - 35, bottom - 70)
    x_values = points[:, 0]
    y_values = points[:, 1]
    x_min, x_max, y_min, y_max = isotropic_plot_bounds(
        x_values,
        y_values,
        plot_width=plot[2] - plot[0],
        plot_height=plot[3] - plot[1],
    )
    _numeric_axes(
        draw,
        plot,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        x_label=x_label + " (raw score)",
        y_label=y_label + " (raw score)",
    )
    coordinates = [
        _project(float(x), float(y), plot, x_min, x_max, y_min, y_max)
        for x, y in points
    ]
    draw.line(coordinates + [coordinates[0]], fill=color, width=4)
    for index, point in enumerate(coordinates):
        if index % 12 == 0:
            draw.ellipse(
                (point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill=color
            )
    start = coordinates[0]
    draw.ellipse(
        (start[0] - 8, start[1] - 8, start[0] + 8, start[1] + 8), fill="#111827"
    )
    draw.text((start[0] + 10, start[1] - 10), "0°", fill="#111827", font=_font(16))
    draw.text(
        (left + 20, bottom - 20),
        "Separate PCA basis per model; compare path shape within each panel.",
        fill="#4b5563",
        font=_font(12),
    )


def _line_panel(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    series: Sequence[tuple[str, ArrayF64, str]],
    angles: ArrayI64,
    title: str,
    y_label: str,
) -> None:
    left, top, right, bottom = rect
    draw.rounded_rectangle(rect, radius=16, fill="#ffffff", outline="#d1d5db", width=2)
    draw.text((left + 18, top + 16), title, fill="#111827", font=_font(20))
    plot = (left + 65, top + 62, right - 35, bottom - 64)
    combined = np.concatenate([values for _name, values, _color in series])
    y_min, y_max = _range_with_padding(combined, lower_at_zero=True)
    x_min = float(np.min(angles))
    x_max = float(np.max(angles))
    _numeric_axes(
        draw,
        plot,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        x_label="angle (degrees)",
        y_label=y_label,
    )
    for offset, (name, values, color) in enumerate(series):
        points = [
            _project(float(angle), float(value), plot, x_min, x_max, y_min, y_max)
            for angle, value in zip(angles, values, strict=True)
        ]
        draw.line(points, fill=color, width=3)
        draw.text(
            (right - 190, top + 18 + offset * 19), name, fill=color, font=_font(14)
        )


def _quarter_panel(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    baseline: QuarterResidualSummary,
    so2: QuarterResidualSummary,
    title: str,
) -> None:
    left, top, right, bottom = rect
    draw.rounded_rectangle(rect, radius=16, fill="#ffffff", outline="#d1d5db", width=2)
    draw.text((left + 18, top + 16), title, fill="#111827", font=_font(20))
    plot = (left + 65, top + 62, right - 35, bottom - 64)
    combined = np.concatenate((
        baseline.lower_quartile,
        baseline.upper_quartile,
        so2.lower_quartile,
        so2.upper_quartile,
    ))
    y_min, y_max = _range_with_padding(combined, lower_at_zero=True)
    x_min, x_max = 0.0, 270.0
    _numeric_axes(
        draw,
        plot,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        x_label="exact angle (degrees)",
        y_label="relative RMS",
    )
    for name, summary, color in (
        ("normal", baseline, "#e63946"),
        ("SO(2)", so2, "#3a86ff"),
    ):
        lower = [
            _project(float(angle), float(value), plot, x_min, x_max, y_min, y_max)
            for angle, value in zip(
                summary.angles_degrees, summary.lower_quartile, strict=True
            )
        ]
        upper = [
            _project(float(angle), float(value), plot, x_min, x_max, y_min, y_max)
            for angle, value in zip(
                summary.angles_degrees, summary.upper_quartile, strict=True
            )
        ]
        draw.polygon(lower + list(reversed(upper)), fill=_with_alpha(color, 42))
        median = [
            _project(float(angle), float(value), plot, x_min, x_max, y_min, y_max)
            for angle, value in zip(summary.angles_degrees, summary.median, strict=True)
        ]
        draw.line(median, fill=color, width=4)
        draw.text(
            (right - 100, top + 18 + (0 if name == "normal" else 19)),
            name,
            fill=color,
            font=_font(14),
        )


def _phase_diagnostic_panel(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    diagnostic: VectorPhaseDiagnostic,
    title: str,
    labels: Sequence[str],
) -> None:
    left, top, right, bottom = rect
    draw.rounded_rectangle(rect, radius=16, fill="#ffffff", outline="#d1d5db", width=2)
    draw.text(
        (left + 20, top + 18),
        title,
        fill="#111827",
        font=_font(24),
    )
    plot = (left + 90, top + 85, right - 50, bottom - 75)
    _numeric_axes(
        draw,
        plot,
        x_min=-1.4,
        x_max=1.4,
        y_min=-1.4,
        y_max=1.4,
        x_label="real (phase-normalized)",
        y_label="imaginary (phase-normalized)",
    )
    center = _project(0.0, 0.0, plot, -1.4, 1.4, -1.4, 1.4)
    radius = abs(_project(1.0, 0.0, plot, -1.4, 1.4, -1.4, 1.4)[0] - center[0])
    draw.ellipse(
        (
            center[0] - radius,
            center[1] - radius,
            center[0] + radius,
            center[1] + radius,
        ),
        outline="#9ca3af",
        width=2,
    )
    for index, label in enumerate(labels):
        normalized = diagnostic.normalized[:, index, :]
        if not diagnostic.valid[index]:
            draw.text(
                (left + 20, top + 55 + index * 20),
                f"{label}: |z(0)| below threshold",
                fill="#9ca3af",
                font=_font(15),
            )
            continue
        points = [
            _project(float(x), float(y), plot, -1.4, 1.4, -1.4, 1.4)
            for x, y in normalized
        ]
        draw.line(points + [points[0]], fill=_PLOT_COLORS[index], width=3)
        draw.text(
            (left + 20, top + 55 + index * 20),
            label,
            fill=_PLOT_COLORS[index],
            font=_font(15),
        )
    draw.text(
        (left + 20, bottom - 18),
        "Grey unit circle: expected exp(i theta) after theta=0 normalization.",
        fill="#4b5563",
        font=_font(15),
    )


def _kernel_vector_panel(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    vector_field: NDArray[np.float32],
    title: str,
    *,
    shared_maximum: float,
) -> None:
    left, top, right, bottom = rect
    draw.rounded_rectangle(rect, radius=16, fill="#ffffff", outline="#d1d5db", width=2)
    draw.text((left + 18, top + 18), title, fill="#111827", font=_font(22))
    plot_left, plot_top, plot_right, plot_bottom = (
        left + 45,
        top + 80,
        right - 30,
        bottom - 55,
    )
    magnitudes = np.sqrt(vector_field[0] ** 2 + vector_field[1] ** 2)
    cell_width = (plot_right - plot_left) / 9.0
    cell_height = (plot_bottom - plot_top) / 9.0
    for row in range(9):
        for column in range(9):
            magnitude = float(magnitudes[row, column]) / shared_maximum
            fill = _heat_color(magnitude)
            x0 = round(plot_left + column * cell_width)
            y0 = round(plot_top + row * cell_height)
            x1 = round(plot_left + (column + 1) * cell_width)
            y1 = round(plot_top + (row + 1) * cell_height)
            draw.rectangle((x0, y0, x1, y1), fill=fill, outline="#e5e7eb")
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            dx = (
                float(vector_field[0, row, column]) / shared_maximum * cell_width * 0.32
            )
            dy = (
                -float(vector_field[1, row, column])
                / shared_maximum
                * cell_height
                * 0.32
            )
            draw.line((cx, cy, round(cx + dx), round(cy + dy)), fill="#111827", width=2)
    draw.text(
        (left + 18, bottom - 36),
        f"background/arrows share 0-{shared_maximum:.3g} F1 magnitude scale",
        fill="#4b5563",
        font=_font(14),
    )


def _bar_panel(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    values: NDArray[np.float32],
    names: Sequence[str],
    title: str,
) -> None:
    left, top, right, bottom = rect
    draw.rounded_rectangle(rect, radius=16, fill="#ffffff", outline="#d1d5db", width=2)
    draw.text((left + 18, top + 16), title, fill="#111827", font=_font(20))
    maximum = float(np.max(values)) or 1.0
    plot = (left + 75, top + 54, right - 35, bottom - 44)
    y_min, y_max = 0.0, maximum * 1.1
    _numeric_axes(
        draw,
        plot,
        x_min=0.0,
        x_max=float(len(values)),
        y_min=y_min,
        y_max=y_max,
        x_label="input color",
        y_label="L2 norm",
        show_x_ticks=False,
    )
    width = (plot[2] - plot[0]) / len(values)
    colors = ("#ef4444", "#22c55e", "#3b82f6")
    for index, (name, value) in enumerate(zip(names, values, strict=True)):
        x0 = round(plot[0] + (index + 0.22) * width)
        x1 = round(x0 + width * 0.56)
        _, y = _project(0.0, float(value), plot, 0.0, 1.0, y_min, y_max)
        draw.rectangle((x0, y, x1, plot[3]), fill=colors[index])
        draw.text((x0, plot[3] + 8), name, fill="#374151", font=_font(14))
        draw.text(
            (x0, y - 18),
            _format_axis_number(float(value)),
            fill="#374151",
            font=_font(12),
        )


def _paste_patch(
    canvas: Image.Image, patch: Tensor, top_left: tuple[int, int], *, side: int
) -> None:
    values = patch.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    pixels = np.clip((values + 1.0) * 127.5, 0.0, 255.0).astype(np.uint8)
    image = Image.fromarray(pixels).resize((side, side), Image.Resampling.LANCZOS)
    canvas.paste(image, top_left)


def _paste_uint8_patch(
    canvas: Image.Image, patch: Tensor, top_left: tuple[int, int], *, side: int
) -> None:
    """Paste a canonical fixed25 input without changing its uint8 scale."""
    pixels = patch.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    image = Image.fromarray(pixels).resize((side, side), Image.Resampling.LANCZOS)
    canvas.paste(image, top_left)


def _paste_spatial_pca_map(
    canvas: Image.Image, values: ArrayU8, top_left: tuple[int, int], *, side: int
) -> None:
    """Upscale native latent cells without smoothing their false-colour values."""
    image = Image.fromarray(values).resize((side, side), Image.Resampling.NEAREST)
    canvas.paste(image, top_left)


def _paste_paper_style_pca_map(
    canvas: Image.Image, values: ArrayU8, top_left: tuple[int, int], *, side: int
) -> None:
    """Bilinearly display a per-image PCA-RGB field, matching the paper style."""
    image = Image.fromarray(values).resize((side, side), Image.Resampling.BILINEAR)
    canvas.paste(image, top_left)


def _coherence_summary_line(label: str, diagnostic: SpatialLatentPcaDiagnostic) -> str:
    """Summarize all-fixed25 coherence scales without hiding collapse context."""
    return (
        f"{label}: median r_edge={np.median(diagnostic.relative_edge_rms):.4f}; "
        f"edge RMS={np.median(diagnostic.edge_rms):.4f}; "
        f"centred spatial RMS={np.median(diagnostic.centered_spatial_rms):.4f}; "
        f"posterior RMS={np.median(diagnostic.posterior_rms):.4f}; "
        f"PCA EV={100.0 * diagnostic.explained_variance.sum():.1f}%"
    )


def _paired_coherence_panel(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    normal: ArrayF64,
    so2: ArrayF64,
) -> None:
    """Draw all fixed25 paired relative-edge-RMS values on one numeric scale."""
    left, top, right, bottom = rect
    draw.rounded_rectangle(rect, radius=16, fill="#ffffff", outline="#d1d5db", width=2)
    draw.text(
        (left + 18, top + 16),
        "All fixed-25 paired local relative edge RMS",
        fill="#111827",
        font=_font(22),
    )
    plot = (left + 80, top + 65, right - 45, bottom - 82)
    y_min, y_max = _range_with_padding(
        np.concatenate((normal, so2)),
        lower_at_zero=False,
    )
    _numeric_axes(
        draw,
        plot,
        x_min=0.0,
        x_max=1.0,
        y_min=y_min,
        y_max=y_max,
        x_label="model",
        y_label="relative edge RMS",
        show_x_ticks=False,
    )
    normal_x, _ = _project(0.24, y_min, plot, 0.0, 1.0, y_min, y_max)
    so2_x, _ = _project(0.76, y_min, plot, 0.0, 1.0, y_min, y_max)
    for normal_value, so2_value in zip(normal, so2, strict=True):
        _, normal_y = _project(0.24, float(normal_value), plot, 0.0, 1.0, y_min, y_max)
        _, so2_y = _project(0.76, float(so2_value), plot, 0.0, 1.0, y_min, y_max)
        draw.line((normal_x, normal_y, so2_x, so2_y), fill="#cbd5e1", width=2)
        draw.ellipse(
            (normal_x - 3, normal_y - 3, normal_x + 3, normal_y + 3),
            fill="#d94155",
        )
        draw.ellipse(
            (so2_x - 3, so2_y - 3, so2_x + 3, so2_y + 3),
            fill="#2478d3",
        )
    for x, values, colour, label in (
        (normal_x, normal, "#d94155", "normal"),
        (so2_x, so2, "#2478d3", "SO(2)"),
    ):
        median = float(np.median(values))
        _, y = _project(0.5, median, plot, 0.0, 1.0, y_min, y_max)
        draw.line((x - 18, y, x + 18, y), fill=colour, width=5)
        draw.text((x - 26, plot[3] + 10), label, fill=colour, font=_font(16))
    draw.text(
        (left + 18, bottom - 28),
        "Each grey segment is one matched patch. Shared y-axis is printed: do not infer a large effect from a zoomed range.",
        fill="#4b5563",
        font=_font(13),
    )


def _delta_coherence_panel(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    delta: ArrayF64,
) -> None:
    """Draw paired SO(2)-minus-normal coherence differences around zero."""
    left, top, right, bottom = rect
    draw.rounded_rectangle(rect, radius=16, fill="#ffffff", outline="#d1d5db", width=2)
    draw.text(
        (left + 18, top + 16),
        "Paired difference: SO(2) minus normal relative edge RMS",
        fill="#111827",
        font=_font(20),
    )
    plot = (left + 80, top + 62, right - 45, bottom - 72)
    y_min, y_max = _range_with_padding(
        np.concatenate((delta, np.zeros(1, dtype=np.float64))),
        lower_at_zero=False,
    )
    _numeric_axes(
        draw,
        plot,
        x_min=0.0,
        x_max=float(len(delta) - 1),
        y_min=y_min,
        y_max=y_max,
        x_label="fixed25 patch index",
        y_label="delta r_edge",
    )
    x0, y0 = _project(0.0, 0.0, plot, 0.0, float(len(delta) - 1), y_min, y_max)
    x1, _ = _project(
        float(len(delta) - 1), 0.0, plot, 0.0, float(len(delta) - 1), y_min, y_max
    )
    draw.line((x0, y0, x1, y0), fill="#6b7280", width=2)
    for index, value in enumerate(delta):
        x, y = _project(
            float(index),
            float(value),
            plot,
            0.0,
            float(len(delta) - 1),
            y_min,
            y_max,
        )
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="#2478d3")
    positive = int(np.count_nonzero(delta > 0.0))
    draw.text(
        (left + 18, bottom - 27),
        f"mean delta={np.mean(delta):+.5f}; median delta={np.median(delta):+.5f}; positive for {positive}/{len(delta)} patches.",
        fill="#4b5563",
        font=_font(14),
    )


def _axes(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    x_label: str,
    y_label: str,
) -> None:
    left, top, right, bottom = rect
    draw.line((left, bottom, right, bottom), fill="#6b7280", width=2)
    draw.line((left, top, left, bottom), fill="#6b7280", width=2)
    draw.text(
        ((left + right) // 2 - 50, bottom + 37), x_label, fill="#4b5563", font=_font(14)
    )
    draw.text((left, top - 18), y_label, fill="#4b5563", font=_font(14))


def _numeric_axes(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    x_label: str,
    y_label: str,
    show_x_ticks: bool = True,
) -> None:
    """Draw readable raw-value tick marks and a light plotting grid."""
    left, top, right, bottom = rect
    x_ticks = _axis_ticks(x_min, x_max)
    y_ticks = _axis_ticks(y_min, y_max)
    if show_x_ticks:
        for index, value in enumerate(x_ticks):
            x, _ = _project(value, y_min, rect, x_min, x_max, y_min, y_max)
            draw.line((x, top, x, bottom), fill="#e5e7eb", width=1)
            draw.line((x, bottom, x, bottom + 5), fill="#6b7280", width=1)
            label = _format_axis_number(value)
            shift = (
                0
                if index == 0
                else len(label) * 3
                if index == len(x_ticks) - 1
                else len(label) * 2
            )
            draw.text((x - shift, bottom + 8), label, fill="#4b5563", font=_font(12))
    for value in y_ticks:
        _, y = _project(x_min, value, rect, x_min, x_max, y_min, y_max)
        draw.line((left, y, right, y), fill="#e5e7eb", width=1)
        draw.line((left - 5, y, left, y), fill="#6b7280", width=1)
        label = _format_axis_number(value)
        draw.text(
            (left - len(label) * 7 - 8, y - 7), label, fill="#4b5563", font=_font(12)
        )
    _axes(draw, rect, x_label, y_label)


def _axis_ticks(minimum: float, maximum: float, *, count: int = 5) -> list[float]:
    """Use evenly spaced labels so visual scale remains inspectable."""
    return np.linspace(minimum, maximum, count).astype(float).tolist()


def _format_axis_number(value: float) -> str:
    """Keep large and tiny raw scales legible in compact panels."""
    if math.isclose(value, 0.0, abs_tol=1e-12):
        return "0"
    magnitude = abs(value)
    if magnitude >= 1_000.0 or magnitude < 0.01:
        return f"{value:.1e}"
    if magnitude >= 10.0:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _range_with_padding(
    values: NDArray[np.float64], *, lower_at_zero: bool = False
) -> tuple[float, float]:
    minimum = (
        min(float(np.min(values)), 0.0) if lower_at_zero else float(np.min(values))
    )
    maximum = float(np.max(values))
    span = maximum - minimum
    padding = 0.08 * span if span > 0 else max(abs(maximum), 1.0) * 0.2
    return minimum - padding, maximum + padding


def isotropic_plot_bounds(
    x_values: NDArray[np.float64],
    y_values: NDArray[np.float64],
    *,
    plot_width: int,
    plot_height: int,
) -> tuple[float, float, float, float]:
    """Return padded bounds with one data-unit-to-pixel scale in both axes."""
    if plot_width <= 0 or plot_height <= 0:
        raise ValueError("plot dimensions must be positive")
    x_min, x_max = _range_with_padding(x_values)
    y_min, y_max = _range_with_padding(y_values)
    unit_per_pixel = max(
        (x_max - x_min) / plot_width,
        (y_max - y_min) / plot_height,
    )
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    x_half_span = unit_per_pixel * plot_width / 2.0
    y_half_span = unit_per_pixel * plot_height / 2.0
    return (
        x_center - x_half_span,
        x_center + x_half_span,
        y_center - y_half_span,
        y_center + y_half_span,
    )


def _project(
    x: float,
    y: float,
    rect: tuple[int, int, int, int],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> tuple[int, int]:
    left, top, right, bottom = rect
    scaled_x = (x - x_min) / (x_max - x_min)
    scaled_y = (y - y_min) / (y_max - y_min)
    return round(left + scaled_x * (right - left)), round(
        bottom - scaled_y * (bottom - top)
    )


def _with_alpha(color: str, alpha: int) -> tuple[int, int, int, int]:
    color = color.lstrip("#")
    return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16), alpha


def _heat_color(value: float) -> tuple[int, int, int]:
    clamped = min(max(value, 0.0), 1.0)
    return (
        round(245.0 - 20.0 * clamped),
        round(247.0 - 120.0 * clamped),
        round(250.0 - 200.0 * clamped),
    )


def _advisor_html_document(payload_json: str, *, max_angle: int) -> str:
    """Render the advisor page entirely from local numeric arrays and SVG."""
    safe_payload = payload_json.replace("</", "<\\/")
    template = r"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Frozen VAE rotation orbit</title>
<style>
:root{color-scheme:light;--ink:#16233a;--muted:#5a6a80;--line:#d8e1ed;--paper:#fff;--wash:#f3f6fb;--normal:#d94155;--so2:#2478d3;--untrained:#7c55d9;--null:#64748b;--gold:#c48117}
*{box-sizing:border-box}body{margin:0;background:var(--wash);color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif;line-height:1.5}main{max-width:1520px;margin:auto;padding:34px 24px 70px}.hero{padding:32px 36px;background:linear-gradient(135deg,#13233b,#294a79);border-radius:24px;box-shadow:0 18px 42px #13233b28;color:white}.eyebrow{margin:0 0 8px;font-size:.78rem;font-weight:800;letter-spacing:.12em;color:#cbdcff}.hero h1{margin:0;font-size:clamp(2rem,4vw,3.25rem);line-height:1.05;letter-spacing:-.045em}.hero p:last-child{max-width:980px;margin:14px 0 0;color:#e2ecfd}.card{background:var(--paper);border:1px solid var(--line);border-radius:18px;padding:20px;box-shadow:0 8px 24px #17243a0c}.control{display:flex;align-items:center;gap:18px;flex-wrap:wrap;margin-top:22px}.control label{font-weight:750}.control input{width:min(630px,75vw);accent-color:var(--so2)}output{font-size:1.2rem;font-weight:800;color:var(--so2);font-variant-numeric:tabular-nums}.note,.section-lead{color:var(--muted)}section{margin-top:28px}h2{margin:0;font-size:1.48rem;letter-spacing:-.025em}h3{margin:0;font-size:1.08rem}.section-lead{max-width:1040px;margin:9px 0 16px}.grid2,.grid3{display:grid;gap:20px}.grid2{grid-template-columns:repeat(2,minmax(0,1fr))}.grid3{grid-template-columns:repeat(3,minmax(0,1fr))}.panel-top{display:flex;align-items:baseline;justify-content:space-between;gap:12px}.tag{font-size:.78rem;font-weight:800;letter-spacing:.07em;text-transform:uppercase}.normal{color:var(--normal)}.so2{color:var(--so2)}.untrained{color:var(--untrained)}.null{color:var(--null)}svg{display:block;width:100%;height:auto;margin-top:13px;border:1px solid #e1e7f0;background:#fff}.readout,.scale-note{font-variant-numeric:tabular-nums}.readout{padding-top:12px;border-top:1px solid #e6ebf3;margin:12px 0 0}.scale-note{font-size:.89rem;color:var(--muted);min-height:2.8em}.math{display:grid;grid-template-columns:1fr 1fr;gap:16px}.math .formula{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:#f4f7fc;border-left:4px solid #7c98c5;padding:13px 15px;overflow:auto;color:#243755}.warning{border-left:4px solid var(--gold);background:#fff9ec;padding:13px 16px;color:#59400d}.metrics{display:grid;grid-template-columns:1.1fr .9fr;gap:20px}.metrics p{color:var(--muted);margin:9px 0}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;margin-top:10px}th,td{padding:8px 9px;text-align:right;border-bottom:1px solid #e7edf5}th:first-child,td:first-child{text-align:left}th{font-size:.79rem;color:var(--muted)}.caption{font-size:.9rem;color:var(--muted);margin:10px 0 0}.kernel-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}.footer{margin-top:30px;color:var(--muted);font-size:.86rem}@media(max-width:900px){main{padding:18px 13px 48px}.hero{padding:26px}.grid2,.grid3,.metrics,.math,.kernel-grid{grid-template-columns:1fr}.control{align-items:flex-start;flex-direction:column}.control input{width:100%}}
</style>
<style>
.orbit-control{margin:20px 0}.equation{margin:13px 0;padding:12px 14px;background:#f4f7fc;border-left:4px solid #7c98c5;color:#243755;overflow:auto}.equation math{font-size:1.08rem}.steps{margin:10px 0 0;padding-left:1.25rem}.steps li+li{margin-top:7px}
.inline-math{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;color:#243755;white-space:nowrap}.inline-math sub,.inline-math sup{font-size:.78em;line-height:0}
.spatial-map-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}.spatial-map-grid h3{font-size:.94rem}.paper-style-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;margin-top:18px}.paper-style-grid canvas{display:block;width:100%;aspect-ratio:1/1;margin-top:12px;border:1px solid #e1e7f0;background:#fff}.spatial-summary-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.spatial-summary-grid svg{margin-top:8px}.probe-grid{display:grid;gap:16px;margin-top:18px}.probe-row{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}.probe-row h3,.probe-row>.caption{grid-column:1/-1}.probe-row canvas{display:block;width:100%;aspect-ratio:1/1;margin-top:9px;border:1px solid #e1e7f0;background:#fff}.probe-tile h4{margin:0;font-size:.96rem}.probe-tile p{margin:6px 0 0;font-size:.86rem;color:var(--muted);font-variant-numeric:tabular-nums}@media(max-width:900px){.spatial-map-grid,.paper-style-grid,.spatial-summary-grid,.probe-row{grid-template-columns:1fr}}
</style>
<main>
  <header class="hero"><p class="eyebrow">FROZEN CHECKPOINTS · FIXED VALIDATION 25 · EXPLORATORY</p><h1>What rotation does to the spatial posterior</h1><p>This page contains only already-precomputed arrays from a dense one-degree rotation sweep of fixed validation patch <strong id="patch-index"></strong>. Moving the slider changes a highlighted sample in the existing plots; it never runs a model.</p></header>
  <section class="card control"><label for="angle">Input rotation</label><input id="angle" type="range" min="0" max="__MAX_ANGLE__" value="0" step="1"><output id="angle-value" for="angle">0°</output><span class="note">one precomputed sample per degree; the closed curve adds its 360°=0° endpoint only for display</span></section>

  <section><h2>1. The posterior orbit: what the PCA coordinates mean</h2><p class="section-lead">For each rotated input x<sub>θ</sub>=R<sub>θ</sub>x, the frozen encoder produces a spatial posterior mean μ<sub>θ</sub>∈R<sup>16×32×32</sup>. We keep only the central radius-14 disk D: 616 spatial cells, so each vectorized posterior x̃<sub>θ</sub>=vec(μ<sub>θ</sub>[:,D]) has 16×616=9,856 entries. This avoids calling padded/interpolated corners of the continuous rotation part of the latent signal.</p><div class="math"><article class="card"><h3>Fit once, not once per degree</h3><p>We stack all 360 vectors as rows of X and center by the orbit mean x̄. PCA is fitted once for this model and this patch. If v<sub>1</sub>,v<sub>2</sub> are the two leading directions, the plotted point at θ is:</p><div class="formula">sθ = (sθ,1, sθ,2)<br>= ((x̃θ−x̄)ᵀv1, (x̃θ−x̄)ᵀv2).</div><p class="caption">The implementation equivalently eigendecomposes the 360×360 dual Gram matrix X<sub>c</sub>X<sub>c</sub><sup>T</sup>. The plot origin is x̄, not a zero posterior.</p></article><article class="card"><h3>Why show it?</h3><p>PCA gives a legible two-dimensional view of how the full 9,856-dimensional spatial posterior travels as input angle changes. A regular loop is descriptive evidence of a low-dimensional, periodic-looking response; folds and erratic turns indicate a more complicated trajectory.</p><p class="warning"><strong>Important limitation.</strong> The PC bases, signs, origins, and score scales are fitted separately in the two panels. A prettier ring does not prove equivariance or a performance advantage. The direct residual below is the comparable quantity.</p></article></div><div class="grid2" style="margin-top:20px"><article class="card"><div class="panel-top"><h3>Normal VAE</h3><span class="tag normal">non-equivariant</span></div><svg id="normal-chart" viewBox="0 0 620 440" role="img" aria-label="Normal VAE posterior PCA orbit"></svg><p id="normal-scale" class="scale-note"></p><p id="normal-current" class="readout"></p></article><article class="card"><div class="panel-top"><h3>SO(2) VAE</h3><span class="tag so2">equivariant architecture</span></div><svg id="so2-chart" viewBox="0 0 620 440" role="img" aria-label="SO2 VAE posterior PCA orbit"></svg><p id="so2-scale" class="scale-note"></p><p id="so2-current" class="readout"></p></article></div></section>

  <section><h2>2. The direct, scale-aware posterior check</h2><p class="section-lead">The paired exploratory diagnostic is e<sub>μ</sub>(θ)=||μ<sub>θ</sub>−R<sub>θ</sub>μ<sub>0</sub>||<sub>D</sub>/(||R<sub>θ</sub>μ<sub>0</sub>||<sub>D</sub>+ε), evaluated on the same central disk. Unlike PCA position, it has the same definition for both models. Raw posterior RMS is shown alongside it so a small-looking path is not mistaken for a large numerical effect.</p><article class="card"><svg id="posterior-residual-chart" viewBox="0 0 1260 390" role="img" aria-label="Dense posterior residual by rotation angle"></svg><p id="posterior-residual-readout" class="readout"></p></article><div class="metrics" style="margin-top:20px"><article class="card"><h3>Why retain the exact-quarter summary?</h3><p>Dense non-quarter rotations use bilinear interpolation and zero padding. The archived all-25 comparison at 90°, 180°, and 270° instead uses exact torch.rot90. It is the strongest directly comparable evidence on this page.</p><p class="caption">The dense plots are for exploration and explanation, not sealed-test evaluation.</p></article><article class="card"><h3>Archived all-25 exact quarter turns</h3><table><thead><tr><th>angle</th><th class="normal">normal median</th><th class="so2">SO(2) median</th></tr></thead><tbody id="quarter-rows"></tbody></table></article></div></section>

  <section><h2>3. Internal F1 two-vectors: phase, gain, and controls</h2><p class="section-lead">Immediately before the SO(2) μ head, encoder D has 48 scalar F0 copies and 48 two-component F1 copies: 48+48×2=144 spatial channels. For F1 copy j, h<sub>j,θ</sub>(p)∈R² at lattice position p. We pool locations, never copies: h̄<sub>j,θ</sub>=|D|<sup>−1</sup>Σ<sub>p∈D</sub>h<sub>j,θ</sub>(p).</p><div class="math"><article class="card"><h3>Expected F1 behavior</h3><div class="formula">h_j,θ(p) ≈ ρ₁(θ) h_j,0(R₋θp)<br>ρ₁(θ) = [[cosθ, −sinθ], [sinθ, cosθ]]<br><br>z̄_j,θ = h̄_j,θ,1 + i h̄_j,θ,2<br>z̄_j,θ / z̄_j,0 ≈ exp(iθ).</div><p class="caption">A disk is rotation-invariant, which motivates this pooled arrow. But pooling can cancel a spatial field; it is an internal diagnostic, not the headline model metric.</p></article><article class="card"><h3>Why split phase from gain?</h3><p>The phase plane shows z̄<sub>θ</sub>/z̄<sub>0</sub>. We report phase error Δφ=wrap(arg z̄<sub>θ</sub>−arg z̄<sub>0</sub>−θ) and gain g<sub>θ</sub>=|z̄<sub>θ</sub>|/(|z̄<sub>0</sub>|+ε) separately. A single Euclidean residual would mix a wrong angle with a changed magnitude.</p><p class="caption">Trained F1 copies were selected before the sweep: the three largest median θ=0 disk magnitudes over all 25 fixed patches.</p></article></div><div class="grid2" style="margin-top:20px"><article class="card"><div class="panel-top"><h3>Trained SO(2) F1</h3><span class="tag so2">selected copies</span></div><svg id="trained-phase-chart" viewBox="0 0 620 470" role="img" aria-label="Trained SO2 F1 phase plane"></svg><p id="trained-phase-readout" class="readout"></p></article><article class="card"><div class="panel-top"><h3>Seeded untrained SO(2) control</h3><span class="tag untrained">same F1 indices</span></div><svg id="untrained-phase-chart" viewBox="0 0 620 470" role="img" aria-label="Untrained SO2 F1 phase plane"></svg><p id="untrained-phase-readout" class="readout"></p></article></div><article class="card" style="margin-top:20px"><h3>F1 diagnostic curves and the normal-VAE null</h3><p class="caption">The purple control tests architectural construction, not whether training discovered equivariance. The grey band is an arbitrary negative-control null: 32 fixed orthonormal projections that bundle the normal VAE’s 96 scalar maps into pairs. It is <strong>not</strong> an F1 representation and it is shown as a 5th–95th-percentile band to avoid privileging one accidental pairing.</p><div class="grid3"><div><svg id="phase-error-chart" viewBox="0 0 410 305" role="img" aria-label="Phase error comparison"></svg></div><div><svg id="gain-chart" viewBox="0 0 410 305" role="img" aria-label="Gain comparison"></svg></div><div><svg id="amplitude-chart" viewBox="0 0 410 305" role="img" aria-label="Raw amplitude comparison"></svg></div></div><p id="f1-readout" class="readout"></p></article></section>

  <section><h2>4. Learned scalar-to-F1 kernel: mechanism, not an orbit result</h2><p class="section-lead">The input stem maps three scalar image channels to F0/F1 fields. The plots below show the largest-L2 learned F1 copy. Arrow and background magnitudes use one shared scale across red, green, and blue inputs, so their sizes are comparable. This constrained kernel is an architectural mechanism inset; it does not evaluate rotation behavior.</p><article class="card"><div class="kernel-grid"><div><h3 class="normal">red input</h3><svg id="kernel-red" viewBox="0 0 370 370" role="img" aria-label="Red input F1 kernel"></svg></div><div><h3 style="color:#169a5b">green input</h3><svg id="kernel-green" viewBox="0 0 370 370" role="img" aria-label="Green input F1 kernel"></svg></div><div><h3 class="so2">blue input</h3><svg id="kernel-blue" viewBox="0 0 370 370" role="img" aria-label="Blue input F1 kernel"></svg></div></div><svg id="kernel-energy-chart" viewBox="0 0 1260 250" role="img" aria-label="Per channel F1 kernel energy"></svg><p id="kernel-note" class="caption"></p></article></section>
  <footer class="footer">__LABEL__</footer>
</main>
<template id="orbit-explanation">
  <h2>1. The posterior orbit: how a 9,856-dimensional embedding becomes a 2-D curve</h2>
  <p class="section-lead">This plot does <em>not</em> fit PCA separately at each angle. It creates one fixed two-dimensional coordinate system, then shows how one full spatial posterior moves through it as the input rotates by one degree at a time.</p>
  <div class="math">
    <article class="card"><h3>Step 1 — one embedding vector per angle</h3><p>For the original patch \(x\), let \(x_\theta=R_\theta x\) be the rotated patch. The frozen encoder returns its final posterior mean \(\mu_\theta\in\mathbb{R}^{16\times32\times32}\). We retain only the central disk \(D\), which has 616 spatial cells and excludes rotation padding at the corners.</p><div class="equation"><math display="block"><semantics><mrow><msub><mover accent="true"><mi>x</mi><mo>~</mo></mover><mi>θ</mi></msub><mo>=</mo><mtext>vec</mtext><mo>(</mo><msub><mi>μ</mi><mi>θ</mi></msub><mo>[</mo><mo>:</mo><mo>,</mo><mi>D</mi><mo>]</mo><mo>)</mo><mo>∈</mo><msup><mi>ℝ</mi><mrow><mn>16</mn><mo>·</mo><mn>616</mn></mrow></msup><mo>=</mo><msup><mi>ℝ</mi><mn>9856</mn></msup></mrow><annotation encoding="application/x-tex">\tilde{x}_\theta=\operatorname{vec}(\mu_\theta[:,D])\in\mathbb{R}^{16\cdot616}=\mathbb{R}^{9856}</annotation></semantics></math></div><p class="caption">Every angle is thus represented by the same 9,856 ordered numbers: 16 channels times the same 616 locations.</p></article>
    <article class="card"><h3>Step 2 — fit PCA once to all 360 rows</h3><p>Stack these vectors into \(X\in\mathbb{R}^{360\times9856}\), subtract the orbit mean \(\bar{x}\), and compute the two directions \(v_1,v_2\) capturing the most variance among the 360 centred rows.</p><div class="equation"><math display="block"><semantics><mrow><msub><mi>X</mi><mi>c</mi></msub><mo>=</mo><mi>X</mi><mo>−</mo><mn>1</mn><msup><mover accent="true"><mi>x</mi><mo>¯</mo></mover><mi>T</mi></msup><mo>,</mo><mspace width="0.6em"/><msub><mi>v</mi><mn>1</mn></msub><mo>,</mo><msub><mi>v</mi><mn>2</mn></msub><mo>=</mo><mtext>top PCA directions of</mtext><msub><mi>X</mi><mi>c</mi></msub></mrow><annotation encoding="application/x-tex">X_c=X-\mathbf{1}\bar{x}^{T},\qquad v_1,v_2=\text{top PCA directions of }X_c</annotation></semantics></math></div><p class="caption">The origin of either PCA panel is the mean embedding along its own orbit, not a zero latent vector.</p></article>
    <article class="card"><h3>Step 3 — project all angles into those same axes</h3><p>The point at angle \(\theta\) is the pair of dot products below. PC1 and PC2 are raw PCA scores: they are neither spatial image coordinates nor probabilities.</p><div class="equation"><math display="block"><semantics><mrow><msub><mi>s</mi><mi>θ</mi></msub><mo>=</mo><mo>(</mo><mo stretchy="false">(</mo><msub><mover accent="true"><mi>x</mi><mo>~</mo></mover><mi>θ</mi></msub><mo>−</mo><mover accent="true"><mi>x</mi><mo>¯</mo></mover><mo stretchy="false">)</mo><mo>·</mo><msub><mi>v</mi><mn>1</mn></msub><mo>,</mo><mo stretchy="false">(</mo><msub><mover accent="true"><mi>x</mi><mo>~</mo></mover><mi>θ</mi></msub><mo>−</mo><mover accent="true"><mi>x</mi><mo>¯</mo></mover><mo stretchy="false">)</mo><mo>·</mo><msub><mi>v</mi><mn>2</mn></msub><mo>)</mo><mo>∈</mo><msup><mi>ℝ</mi><mn>2</mn></msup></mrow><annotation encoding="application/x-tex">s_\theta=((\tilde{x}_\theta-\bar{x})^Tv_1,(\tilde{x}_\theta-\bar{x})^Tv_2)\in\mathbb{R}^2</annotation></semantics></math></div><p class="caption">The slider selects the white point. The smaller coloured dots are 30° landmarks; the curve closes by drawing 360° as the saved 0° vector.</p></article>
    <article class="card"><h3>How to read it — and its limitation</h3><ol class="steps"><li>A regular loop is descriptive evidence that the dominant PCA variation is periodic-looking as the image rotates.</li><li>Folds, crossings, or abrupt turns mean that the two-dimensional summary of this orbit is more complicated.</li><li>The normal and SO(2) PCA bases, signs, origins, and numeric scales are fitted separately. Therefore one panel cannot be declared “more equivariant” because it looks more circular.</li></ol><p class="warning"><strong>The next plot is the comparable one.</strong> It uses the same direct residual formula for both models and is the evidence to use for rotation consistency.</p></article>
  </div>
</template>
<template id="posterior-explanation">
  <h2>2. Direct posterior equivariance: definition, scale, and what this result says</h2>
  <p class="section-lead">The final posterior \(\mu\) is a 16-channel <em>scalar</em> spatial field in both models. Let \(S_\theta\) rotate every \(32\times32\) channel of \(\mu_0\) spatially with the renderer’s documented convention. This asks whether encoding and rotating approximately commute at this final representation.</p>
  <div class="math">
    <article class="card"><h3>Step 1 — observed field versus expected rotated field</h3><p>Encode the rotated image to obtain \(\mu_\theta=E_\mu(R_\theta x)\). Independently rotate the original posterior to obtain \(S_\theta\mu_0\). If the final scalar field were perfectly rotation-equivariant under this action, these fields would agree on the disk.</p><div class="equation"><math display="block"><semantics><mrow><msub><mi>μ</mi><mi>θ</mi></msub><mover accent="true"><mo>≈</mo><mo>?</mo></mover><msub><mi>S</mi><mi>θ</mi></msub><msub><mi>μ</mi><mn>0</mn></msub></mrow><annotation encoding="application/x-tex">\mu_\theta\stackrel{?}{\approx}S_\theta\mu_0</annotation></semantics></math></div><p class="caption">At 90°, 180°, and 270°, \(S_\theta\) uses exact <code>rot90</code>. The dense one-degree plot uses bilinear interpolation away from quarter turns, so it is explanatory rather than a formal evaluation.</p></article>
    <article class="card"><h3>Step 2 — RMS puts the error on a meaningful scale</h3><p>For any latent field \(A\), RMS is its root-mean-square value across all 16 channels and all retained positions. We divide the RMS difference by the RMS magnitude of the expected field. This prevents a model with smaller raw latent values from appearing better merely because all of its numbers are smaller.</p><div class="equation"><math display="block"><semantics><mrow><mtext>RMS</mtext><msub><mi>D</mi><mrow><mo>(</mo><mi>A</mi><mo>)</mo></mrow></msub><mo>=</mo><msqrt><mfrac><mrow><munderover><mo>∑</mo><mrow><mi>c</mi><mo>=</mo><mn>1</mn></mrow><mn>16</mn></munderover><munderover><mo>∑</mo><mrow><mi>p</mi><mo>∈</mo><mi>D</mi></mrow><mrow></mrow></munderover><msubsup><mi>A</mi><mi>c</mi><mn>2</mn></msubsup><mo>(</mo><mi>p</mi><mo>)</mo></mrow><mrow><mn>16</mn><mo>|</mo><mi>D</mi><mo>|</mo></mrow></mfrac></msqrt></mrow><annotation encoding="application/x-tex">\operatorname{RMS}_D(A)=\sqrt{\frac{1}{16|D|}\sum_{c=1}^{16}\sum_{p\in D}A_c(p)^2}</annotation></semantics></math><math display="block"><semantics><mrow><msub><mi>e</mi><mi>μ</mi></msub><mo>(</mo><mi>θ</mi><mo>)</mo><mo>=</mo><mfrac><mrow><mtext>RMS</mtext><msub><mi>D</mi><mrow><mo>(</mo><msub><mi>μ</mi><mi>θ</mi></msub><mo>−</mo><msub><mi>S</mi><mi>θ</mi></msub><msub><mi>μ</mi><mn>0</mn></msub><mo>)</mo></mrow></msub></mrow><mrow><mtext>RMS</mtext><msub><mi>D</mi><mrow><mo>(</mo><msub><mi>S</mi><mi>θ</mi></msub><msub><mi>μ</mi><mn>0</mn></msub><mo>)</mo></mrow></msub><mo>+</mo><mi>ε</mi></mrow></mfrac></mrow><annotation encoding="application/x-tex">e_\mu(\theta)=\frac{\operatorname{RMS}_D(\mu_\theta-S_\theta\mu_0)}{\operatorname{RMS}_D(S_\theta\mu_0)+\varepsilon}</annotation></semantics></math></div></article>
    <article class="card"><h3>How to read relative RMS</h3><ol class="steps"><li>\(e_\mu(\theta)=0\) means the observed and expected fields match exactly on the disk.</li><li>\(e_\mu(\theta)=1\) means the RMS error is as large as the RMS magnitude of the expected field. It is an amplitude ratio, not “one percent” and not a squared error.</li><li>Lower is better for this rotation-consistency proxy. Raw posterior RMS is shown beside it because a nearly zero field makes any normalized ratio unstable and uninformative.</li></ol><p class="caption">The common formula makes the two model curves directly comparable; it still does not make the result a sealed-test or architecture-wide theorem.</p></article>
    <article class="card"><h3>What these frozen checkpoints show</h3><p>The direct result currently does <strong>not</strong> support saying that the SO(2) final posterior is more equivariant on this diagnostic. In the archived all-25 exact quarter-turn summary, the normal VAE is lower at every nonzero angle:</p><div class="equation"><math display="block"><semantics><mrow><mtext>normal: </mtext><mn>1.327</mn><mo>,</mo><mn>1.303</mn><mo>,</mo><mn>1.327</mn><mspace width="0.5em"/><mtext>vs. SO(2): </mtext><mn>1.473</mn><mo>,</mo><mn>1.418</mn><mo>,</mo><mn>1.487</mn><mspace width="0.5em"/><mtext>at 90°, 180°, 270°</mtext></mrow><annotation encoding="application/x-tex">\text{normal: }1.327,1.303,1.327\quad\text{vs. SO(2): }1.473,1.418,1.487</annotation></semantics></math></div><p class="warning"><strong>This is important, not embarrassing.</strong> The visually cleaner SO(2) PCA loop cannot override the shared residual. The honest reading is mixed evidence: this particular final-posterior proxy ranks the normal VAE lower, and explaining why requires a separate experiment.</p></article>
  </div>
</template>
<template id="f1-explanation">
  <h2>3. Internal F1 two-vectors: phase, gain, and the normal-VAE null</h2>
  <p class="section-lead">This is an internal diagnostic immediately before the SO(2) posterior head. It is not a second measurement of the final posterior in Section 2. Its purpose is to ask whether selected vector-valued \(F_1\) fields display the transformation law built into the architecture.</p>
  <div class="math">
    <article class="card"><h3>Step 1 — what “48 F1 copies” means</h3><p>The final encoder-D tensor has \(144=48+48\times2\) channels: 48 scalar \(F_0\) fields plus 48 \(F_1\) fields with two components each. For one copy \(j\), \(h_{j,\theta}(p)\in\mathbb{R}^2\) is a two-vector at spatial location \(p\). It is not one of the final 16 scalar \(\mu\) channels.</p><div class="equation"><math display="block"><semantics><mrow><msub><mi>h</mi><mrow><mi>j</mi><mo>,</mo><mi>θ</mi></mrow></msub><mo>(</mo><mi>p</mi><mo>)</mo><mover accent="true"><mo>≈</mo><mo>?</mo></mover><msub><mi>ρ</mi><mn>1</mn></msub><mo>(</mo><mi>θ</mi><mo>)</mo><msub><mi>h</mi><mrow><mi>j</mi><mo>,</mo><mn>0</mn></mrow></msub><mo>(</mo><msub><mi>R</mi><mrow><mo>−</mo><mi>θ</mi></mrow></msub><mi>p</mi><mo>)</mo></mrow><annotation encoding="application/x-tex">h_{j,\theta}(p)\stackrel{?}{\approx}\rho_1(\theta)h_{j,0}(R_{-\theta}p)</annotation></semantics></math><math display="block"><semantics><mrow><msub><mi>ρ</mi><mn>1</mn></msub><mo>(</mo><mi>θ</mi><mo>)</mo><mo>=</mo><mfenced><mtable><mtr><mtd><mtext>cos θ</mtext></mtd><mtd><mtext>−sin θ</mtext></mtd></mtr><mtr><mtd><mtext>sin θ</mtext></mtd><mtd><mtext>cos θ</mtext></mtd></mtr></mtable></mfenced></mrow><annotation encoding="application/x-tex">\rho_1(\theta)=\begin{bmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{bmatrix}</annotation></semantics></math></div><p class="caption">The vector components rotate by \(\rho_1\); the spatial field is simultaneously sampled at the rotated location.</p></article>
    <article class="card"><h3>Step 2 — pool positions, not feature copies</h3><p>We average locations in the central disk for each \(j\) separately, then represent its two coordinates as one complex number. Because the disk is rotation-symmetric, an ideal pooled \(F_1\) arrow should rotate by the phase \(e^{i\theta}\).</p><div class="equation"><math display="block"><semantics><mrow><mover accent="true"><mi>h</mi><mo>¯</mo></mover><msub><mi>j</mi><mi>θ</mi></msub><mo>=</mo><mfrac><mn>1</mn><mrow><mo>|</mo><mi>D</mi><mo>|</mo></mrow></mfrac><munderover><mo>∑</mo><mrow><mi>p</mi><mo>∈</mo><mi>D</mi></mrow><mrow></mrow></munderover><msub><mi>h</mi><mrow><mi>j</mi><mo>,</mo><mi>θ</mi></mrow></msub><mo>(</mo><mi>p</mi><mo>)</mo><mo>,</mo><mspace width="0.4em"/><msub><mi>z</mi><mrow><mi>j</mi><mi>θ</mi></mrow></msub><mo>=</mo><mover accent="true"><mi>h</mi><mo>¯</mo></mover><msub><mi>j</mi><mi>θ</mi></msub><msub><mo>,</mo><mn>1</mn></msub><mo>+</mo><mi>i</mi><mover accent="true"><mi>h</mi><mo>¯</mo></mover><msub><mi>j</mi><mi>θ</mi></msub><msub><mo>,</mo><mn>2</mn></msub></mrow><annotation encoding="application/x-tex">\bar h_{j,\theta}=\frac{1}{|D|}\sum_{p\in D}h_{j,\theta}(p),\qquad z_{j,\theta}=\bar h_{j,\theta,1}+i\bar h_{j,\theta,2}</annotation></semantics></math><math display="block"><semantics><mrow><mfrac><msub><mi>z</mi><mrow><mi>j</mi><mi>θ</mi></mrow></msub><msub><mi>z</mi><mrow><mi>j</mi><mn>0</mn></mrow></msub></mfrac><mover accent="true"><mo>≈</mo><mo>?</mo></mover><msup><mi>e</mi><mrow><mi>i</mi><mi>θ</mi></mrow></msup></mrow><annotation encoding="application/x-tex">\frac{z_{j,\theta}}{z_{j,0}}\stackrel{?}{\approx}e^{i\theta}</annotation></semantics></math></div><p class="caption">A small \(|z_{j,\theta}|\) can arise from cancellation of a spatially structured field in the average. Then its phase is inherently unreliable.</p></article>
    <article class="card"><h3>Step 3 — direction, gain, and magnitude are different questions</h3><p>The phase plane shows \(z_{j,\theta}/z_{j,0}\); the ideal is the grey unit circle. The slider marks the current angle. We also compute:</p><div class="equation"><math display="block"><semantics><mrow><mi>Δ</mi><msub><mi>φ</mi><mi>j</mi></msub><mo>(</mo><mi>θ</mi><mo>)</mo><mo>=</mo><mtext>wrap</mtext><mo>(</mo><mtext>arg</mtext><msub><mi>z</mi><mrow><mi>j</mi><mi>θ</mi></mrow></msub><mo>−</mo><mtext>arg</mtext><msub><mi>z</mi><mrow><mi>j</mi><mn>0</mn></mrow></msub><mo>−</mo><mi>θ</mi><mo>)</mo><mo>,</mo><mspace width="0.4em"/><msub><mi>g</mi><mi>j</mi></msub><mo>(</mo><mi>θ</mi><mo>)</mo><mo>=</mo><mfrac><mrow><mo>|</mo><msub><mi>z</mi><mrow><mi>j</mi><mi>θ</mi></mrow></msub><mo>|</mo></mrow><mrow><mo>|</mo><msub><mi>z</mi><mrow><mi>j</mi><mn>0</mn></mrow></msub><mo>|</mo><mo>+</mo><mi>ε</mi></mrow></mfrac></mrow><annotation encoding="application/x-tex">\Delta\phi_j(\theta)=\operatorname{wrap}(\arg z_{j,\theta}-\arg z_{j,0}-\theta),\qquad g_j(\theta)=\frac{|z_{j,\theta}|}{|z_{j,0}|+\varepsilon}</annotation></semantics></math></div><ol class="steps"><li>Ideal absolute phase error is \(0^\circ\).</li><li>Ideal gain is \(1\).</li><li>Raw amplitude should remain well above the displayed near-zero threshold before interpreting phase.</li></ol></article>
    <article class="card"><h3>Why the normal-VAE line is only a null</h3><p>The normal encoder has 96 scalar pre-head maps, not \(F_1\) two-vectors. We create 32 seeded orthonormal maps \(Q_e\) that arbitrarily bundle those scalar disk means into artificial pairs:</p><div class="equation"><math display="block"><semantics><mrow><msub><mi>u</mi><mrow><mi>e</mi><mo>,</mo><mi>j</mi><mo>,</mo><mi>θ</mi></mrow></msub><mo>=</mo><msub><mi>Q</mi><mrow><mi>e</mi><mo>,</mo><mi>j</mi></mrow></msub><msub><mi>a</mi><mi>θ</mi></msub><mo>∈</mo><msup><mi>ℝ</mi><mn>2</mn></msup><mo>,</mo><mspace width="0.4em"/><msub><mi>a</mi><mi>θ</mi></msub><mo>∈</mo><msup><mi>ℝ</mi><mn>96</mn></msup></mrow><annotation encoding="application/x-tex">u_{e,j,\theta}=Q_{e,j}a_\theta\in\mathbb{R}^2,\qquad a_\theta\in\mathbb{R}^{96}</annotation></semantics></math></div><p class="warning"><strong>The grey band answers a narrow question:</strong> could arbitrary scalar pairings accidentally look phase-aligned? It is not “normal F1,” not a normal-VAE equivariance score, and not a trained-model baseline. The trained F1 traces here are not clean unit-circle evidence; that limitation is a result worth reporting.</p></article>
  </div>
</template>
<template id="spatial-pca-explanation">
  <h2>5. Spatial latent PCA-RGB: an EQ-VAE-inspired local-coherence diagnostic</h2>
  <p class="section-lead">This is a different question from the rotation orbit. Rather than treating angles as PCA samples, we treat the 32x32 spatial locations inside each posterior field as samples and its 16 latent channels as features. The result is a false-colour spatial map, not a two-dimensional orbit.</p>
  <div class="math">
    <article class="card"><h3>Step 1 — a local 16-dimensional descriptor</h3><p>For every fixed-25 patch and every central-disk location, collect the 16 posterior values at that location. We fit PCA once per model over all 25 times 616 descriptors, so every map from that model uses the same three colour directions.</p><div class="equation"><math display="block"><mrow><mi>f</mi><mrow><mo>(</mo><mi>p</mi><mo>)</mo></mrow><mo>=</mo><mo>(</mo><msub><mi>μ</mi><mn>1</mn></msub><mo>(</mo><mi>p</mi><mo>)</mo><mo>,</mo><mo>…</mo><mo>,</mo><msub><mi>μ</mi><mn>16</mn></msub><mo>(</mo><mi>p</mi><mo>)</mo><mo>)</mo><mo>∈</mo><msup><mi>ℝ</mi><mn>16</mn></msup></mrow></math></div><p class="caption">This preserves the original spatial position p. PCA chooses a three-number colour for each latent cell; it does not move the cell to a PCA-derived x/y position.</p></article>
    <article class="card"><h3>Step 2 — map three PCA scores to RGB</h3><p>Let W be the three leading PCA directions from the all-fixed25 fit. The score vector is mapped directly to red, green, and blue. One symmetric 99th-percentile score scale is shared by PC1, PC2, and PC3 within a model, so a tiny noisy PC3 cannot be contrast-boosted to look equally important.</p><div class="equation"><math display="block"><mrow><mi>y</mi><mo>(</mo><mi>p</mi><mo>)</mo><mo>=</mo><mo>(</mo><mi>f</mi><mo>(</mo><mi>p</mi><mo>)</mo><mo>−</mo><mover accent="true"><mi>f</mi><mo>¯</mo></mover><mo>)</mo><mi>W</mi><mo>,</mo><mspace width="0.5em"/><msub><mi>RGB</mi><mi>k</mi></msub><mo>=</mo><mtext>clip</mtext><mo>(</mo><mn>1</mn><mo>/</mo><mn>2</mn><mo>+</mo><msub><mi>y</mi><mi>k</mi></msub><mo>/</mo><mn>2</mn><mi>s</mi><mo>,</mo><mn>0</mn><mo>,</mo><mn>1</mn><mo>)</mo></mrow></math></div><p class="caption">The neutral outer ring is excluded from the disk calculation. Maps use enlarged, unsmoothed native 32x32 cells. PCA signs are fixed deterministically, but hues still have no cross-model semantic meaning.</p></article>
    <article class="card"><h3>What “grainy” means here</h3><p>Rapid colour changes indicate rapid changes in the local 16-dimensional descriptor along the displayed PCA directions. That is a useful visual cue for local spatial variation, but it is qualitative and can miss variation outside the top three components.</p><p class="warning"><strong>Not a literal reproduction of the paper's picture.</strong> This version uses one disk-masked PCA fit over the fixed 25 per model, rather than per-image full-square colour fitting. It is designed to make our fixed-25 comparison reproducible and to avoid padded corners.</p></article>
    <article class="card"><h3>Why RMS rather than a mean?</h3><p>A signed mean of neighbour differences can be zero through cancellation: +1 and −1 average to 0 even though both are jumps. RMS squares first, averages, then takes a square root, so it measures a typical jump magnitude in the original latent units and gives rare sharp jumps more weight. Mean absolute difference would also be valid, but it weights all jumps linearly.</p><div class="equation"><math display="block"><mrow><msub><mi>r</mi><mtext>edge</mtext></msub><mo>=</mo><mfrac><mrow><msqrt><mtext>mean over channel-neighbour pairs of</mtext><mo>(</mo><msub><mi>μ</mi><mi>c</mi></msub><mo>(</mo><mi>p</mi><mo>)</mo><mo>−</mo><msub><mi>μ</mi><mi>c</mi></msub><mo>(</mo><mi>q</mi><mo>)</mo><msup><mo>)</mo><mn>2</mn></msup></mrow><mrow><msqrt><mtext>mean over disk cells of</mtext><mo>(</mo><msub><mi>μ</mi><mi>c</mi></msub><mo>(</mo><mi>p</mi><mo>)</mo><mo>−</mo><mover accent="true"><msub><mi>μ</mi><mi>c</mi></msub><mo>¯</mo></mover><msup><mo>)</mo><mn>2</mn></msup></mrow></mfrac></mrow></math></div><p class="caption">The denominator is centred spatial RMS. It asks whether neighbour jumps are large relative to that map's own spatial variation, not merely whether one model has larger raw latent values.</p></article>
  </div>
</template>
<template id="paper-style-pca-explanation">
  <article class="card" style="margin-top:22px"><h3>Paper-style visual companion — per-patch contrast, not a blend</h3><p class="section-lead">The paper-like tiles below answer a visual question: does the spatial arrangement of local posterior descriptors retain coarse structure? The source patch at left is never composited with the PCA colours. It is only a reference for your eye.</p><div class="math">
    <article class="card"><h3>1. One row per spatial location</h3><p>For one selected patch n and one model, retain every location of its 32×32 posterior field. The row at location p contains the 16 final-posterior channel values at that exact lattice coordinate.</p><div class="equation"><math display="block"><semantics><mrow><msub><mi>X</mi><mi>n</mi></msub><mo>∈</mo><msup><mi>ℝ</mi><mrow><mn>1024</mn><mo>×</mo><mn>16</mn></mrow></msup><mo>,</mo><mspace width="0.5em"/><msub><mi>X</mi><mi>n</mi></msub><mo>[</mo><mi>p</mi><mo>,</mo><mo>:</mo><mo>]</mo><mo>=</mo><mo>(</mo><msub><mi>μ</mi><mn>1</mn></msub><mo>(</mo><mi>p</mi><mo>)</mo><mo>,</mo><mo>…</mo><mo>,</mo><msub><mi>μ</mi><mn>16</mn></msub><mo>(</mo><mi>p</mi><mo>)</mo><mo>)</mo></mrow><annotation encoding="application/x-tex">X_n\in\mathbb{R}^{1024\times16},\qquad X_n[p,:]=(\mu_1(p),\ldots,\mu_{16}(p))</annotation></semantics></math></div><p class="caption">Unlike the shared-scale maps above, this fit uses the full native square—not the radius-14 disk—and it is repeated separately for every displayed model/patch tile.</p></article>
    <article class="card"><h3>2. PCA scores become RGB</h3><p>Centre the 1,024 rows, take the top three channel-PCA directions V<sub>n</sub>, and send each location’s three scores back to that same location as red, green, and blue.</p><div class="equation"><math display="block"><semantics><mrow><msub><mi>Z</mi><mi>n</mi></msub><mo>=</mo><mo>(</mo><msub><mi>X</mi><mi>n</mi></msub><mo>−</mo><mn>1</mn><msub><mover accent="true"><mi>x</mi><mo>¯</mo></mover><mi>n</mi></msub><msup><mo>)</mo><mi>T</mi></msup><mo>)</mo><msub><mi>V</mi><mi>n</mi></msub><mo>∈</mo><msup><mi>ℝ</mi><mrow><mn>1024</mn><mo>×</mo><mn>3</mn></mrow></msup></mrow><annotation encoding="application/x-tex">Z_n=(X_n-\mathbf{1}\bar{x}_n^T)V_n\in\mathbb{R}^{1024\times3}</annotation></semantics></math><math display="block"><semantics><mrow><msub><mi>RGB</mi><mi>n</mi></msub><mo>=</mo><mfrac><mrow><msub><mi>Z</mi><mi>n</mi></msub><mo>−</mo><mtext>min</mtext><mo>(</mo><msub><mi>Z</mi><mi>n</mi></msub><mo>)</mo></mrow><mrow><mtext>max</mtext><mo>(</mo><msub><mi>Z</mi><mi>n</mi></msub><mo>)</mo><mo>−</mo><mtext>min</mtext><mo>(</mo><msub><mi>Z</mi><mi>n</mi></msub><mo>)</mo></mrow></mfrac></mrow><annotation encoding="application/x-tex">\operatorname{RGB}_n=\frac{Z_n-\min(Z_n)}{\max(Z_n)-\min(Z_n)}</annotation></semantics></math></div><p class="caption">The minimum and maximum are one joint scalar range across all three score planes of that one tile. A 32×32 RGB result is bilinearly enlarged only to make it easier to see.</p></article>
    <article class="card"><h3>3. How to read it honestly</h3><p>Image-like regions mean nearby spatial locations have similar 16-dimensional descriptors along the dominant PCA directions. They do <em>not</em> mean the latent is reconstructing or copying RGB pixels: PCA sees only the posterior tensor.</p><p class="warning"><strong>Do not compare hues or apparent smoothness between these tiles.</strong> Each tile has its own PCA axes, unconstrained component signs, and min/max contrast. The paper-style view is useful for presentation and hypothesis generation. The shared-fit maps and paired relative-edge RMS immediately below remain the reproducible comparison.</p></article>
  </div></article>
</template>
<template id="pointwise-rgb-probe-explanation">
  <h2>6. Held-out pointwise latent-to-RGB probe: does one local descriptor retain appearance?</h2>
  <p class="section-lead">PCA maps make latent variation visible, but they do not tell us whether a local posterior descriptor contains information about the source appearance at the same location. This deliberately small readout asks that direct question without using the VAE decoder.</p>
  <div class="math">
    <article class="card"><h3>Step 1 — create aligned source targets</h3><p>For source patch n, antialiased bilinear downsampling maps the 256×256 RGB patch to the same 32×32 lattice as the posterior. At each lattice cell p, the target is T<sub>n</sub>(p)∈[0,1]³ and the input is the 16-number descriptor f<sub>m,n</sub>(p)=μ<sub>m,n</sub>[:,p].</p><div class="equation"><math display="block"><semantics><mrow><msub><mi>T</mi><mi>n</mi></msub><mo>(</mo><mi>p</mi><mo>)</mo><mo>∈</mo><msup><mrow><mo>[</mo><mn>0</mn><mo>,</mo><mn>1</mn><mo>]</mo></mrow><mn>3</mn></msup><mo>,</mo><mspace width="0.6em"/><msub><mi>f</mi><mrow><mi>m</mi><mo>,</mo><mi>n</mi></mrow></msub><mo>(</mo><mi>p</mi><mo>)</mo><mo>=</mo><msub><mi>μ</mi><mrow><mi>m</mi><mo>,</mo><mi>n</mi></mrow></msub><mo>[</mo><mo>:</mo><mo>,</mo><mi>p</mi><mo>]</mo><mo>∈</mo><msup><mi>ℝ</mi><mn>16</mn></msup></mrow><annotation encoding="application/x-tex">T_n(p)\in[0,1]^3,\qquad f_{m,n}(p)=\mu_{m,n}[:,p]\in\mathbb{R}^{16}</annotation></semantics></math></div><p class="caption">The target is displayed beside the prediction; it is never blended with it.</p></article>
    <article class="card"><h3>Step 2 — fit one shared 1×1 affine readout</h3><p>For each model independently, standardize the 16 features using only patches 0–19, then fit one ridge-regularized affine map shared by every cell. It has 3×16 weights and 3 colour biases: 51 learned numbers. There are no neighbouring cells, coordinates, decoder features, or patch-specific parameters.</p><div class="equation"><math display="block"><semantics><mrow><mrow><mo>(</mo><msub><mi>W</mi><mi>m</mi></msub><mo>,</mo><msub><mi>b</mi><mi>m</mi></msub><mo>)</mo></mrow><mo>=</mo><munder><mtext>arg min</mtext><mrow><mi>W</mi><mo>,</mo><mi>b</mi></mrow></munder><mrow><munderover><mo>∑</mo><mrow><mi>n</mi><mo>=</mo><mn>0</mn></mrow><mn>19</mn></munderover><munderover><mo>∑</mo><mi>p</mi><mrow></mrow></munderover><msup><mrow><mo>‖</mo><mi>W</mi><mover accent="true"><mi>f</mi><mo>~</mo></mover><msub><mi>m</mi><mi>n</mi></msub><mo>(</mo><mi>p</mi><mo>)</mo><mo>+</mo><mi>b</mi><mo>−</mo><msub><mi>T</mi><mi>n</mi></msub><mo>(</mo><mi>p</mi><mo>)</mo><mo>‖</mo></mrow><mn>2</mn></msup><mo>+</mo><mi>λ</mi><msup><mrow><mo>‖</mo><mi>W</mi><mo>‖</mo></mrow><mn>2</mn></msup><mo>,</mo><mspace width="0.5em"/><mi>λ</mi><mo>=</mo><mn>10</mn><msup><mn>−3</mn></msup></mrow></mrow><annotation encoding="application/x-tex">(W_m,b_m)=\arg\min_{W,b}\sum_{n=0}^{19}\sum_p\lVert W\tilde f_{m,n}(p)+b-T_n(p)\rVert^2+\lambda\lVert W\rVert^2,\quad\lambda=10^{-3}</annotation></semantics></math></div></article>
    <article class="card"><h3>Step 3 — test only on unseen patches</h3><p>Patches 20–24 are never used to fit the weights. For each such patch, R² compares the prediction with the constant RGB baseline fixed from the twenty training patches. R²=0 means no better than that baseline; R²=1 would be perfect. MSE is the mean squared RGB error before display-only clipping.</p><div class="equation"><math display="block"><semantics><mrow><msubsup><mi>R</mi><mrow><mi>m</mi><mo>,</mo><mi>n</mi></mrow><mn>2</mn></msubsup><mo>=</mo><mn>1</mn><mo>−</mo><mfrac><mrow><munderover><mo>∑</mo><mrow><mi>p</mi><mo>,</mo><mi>c</mi></mrow><mrow></mrow></munderover><msup><mrow><mo>(</mo><msub><mover accent="true"><mi>T</mi><mo>^</mo></mover><mrow><mi>m</mi><mo>,</mo><mi>n</mi></mrow></msub><mo>(</mo><mi>p</mi><mo>)</mo><mo>−</mo><msub><mi>T</mi><mi>n</mi></msub><mo>(</mo><mi>p</mi><mo>)</mo><mo>)</mo></mrow><mn>2</mn></msup></mrow><mrow><munderover><mo>∑</mo><mrow><mi>p</mi><mo>,</mo><mi>c</mi></mrow><mrow></mrow></munderover><msup><mrow><mo>(</mo><msub><mover accent="true"><mi>T</mi><mo>¯</mo></mover><mtext>train</mtext></msub><mo>−</mo><msub><mi>T</mi><mi>n</mi></msub><mo>(</mo><mi>p</mi><mo>)</mo><mo>)</mo></mrow><mn>2</mn></msup></mrow></mfrac></mrow><annotation encoding="application/x-tex">R^2_{m,n}=1-\frac{\sum_{p,c}(\hat T_{m,n}(p)-T_n(p))^2}{\sum_{p,c}(\bar T_{\mathrm{train}}-T_n(p))^2}</annotation></semantics></math></div></article>
    <article class="card"><h3>How to read the image rows</h3><ol class="steps"><li>Each row is one held-out source patch; the middle and right tiles predict its downsampled RGB appearance from the descriptor at the <em>same</em> spatial cell.</li><li>Visible tissue boundaries or broad colour regions in a prediction indicate locally decodable appearance information. Blur is expected from a single linear 1×1 map.</li><li>This is not a VAE reconstruction, an equivariance measurement, or a fair model-quality ranking. It is only a tightly controlled test of local linear appearance information.</li></ol><p class="warning"><strong>Read the numerical result modestly.</strong> The normal VAE is higher on this one five-patch split, but the sample is too small for a general claim.</p></article>
  </div>
</template>
<script>
const DATA=__PAYLOAD__;
const NS=['http:','','www.w3.org','2000','svg'].join('/');
const C={normal:'#d94155',so2:'#2478d3',untrained:'#7c55d9',null:'#64748b',nullLight:'#c3ccd9'};
const el=(tag,attrs={},text='')=>{const n=document.createElementNS(NS,tag);for(const [k,v] of Object.entries(attrs))n.setAttribute(k,String(v));n.textContent=text;return n};
const fmt=x=>{const a=Math.abs(x);if(a===0)return'0';if(a>=1000||a<.01)return x.toExponential(2);return a>=10?x.toFixed(1):x.toFixed(3)};
const median=xs=>{const a=[...xs].filter(Number.isFinite).sort((x,y)=>x-y);return a.length?a[Math.floor(a.length/2)]:NaN};
const flat=xs=>xs.flat(Infinity).filter(Number.isFinite);
const bounds=(values,zero=false)=>{let a=Math.min(...values),b=Math.max(...values);if(zero)a=Math.min(0,a);const d=b-a||Math.max(Math.abs(a),1);return[a-.08*d,b+.08*d]};
function axes(svg,{x0,x1,y0,y1,L,T,W,H,xlabel,ylabel,showXTickLabels=true}){const px=x=>L+(x-x0)/(x1-x0)*W,py=y=>T+H-(y-y0)/(y1-y0)*H;for(let i=0;i<5;i++){const x=x0+(x1-x0)*i/4,y=y0+(y1-y0)*i/4;const qx=px(x),qy=py(y);svg.append(el('line',{x1:qx,y1:T,x2:qx,y2:T+H,stroke:'#e7edf5'}));if(showXTickLabels)svg.append(el('text',{x:qx,y:T+H+18,'text-anchor':i===0?'start':i===4?'end':'middle',fill:'#5a6a80','font-size':11},fmt(x)));svg.append(el('line',{x1:L,y1:qy,x2:L+W,y2:qy,stroke:'#e7edf5'}));svg.append(el('text',{x:L-8,y:qy+4,'text-anchor':'end',fill:'#5a6a80','font-size':11},fmt(y)))}svg.append(el('rect',{x:L,y:T,width:W,height:H,fill:'none',stroke:'#6c7b90'}));svg.append(el('text',{x:L+W/2,y:T+H+42,'text-anchor':'middle',fill:'#506077','font-size':12},xlabel));svg.append(el('text',{x:16,y:T+H/2,transform:`rotate(-90 16 ${T+H/2})`,'text-anchor':'middle',fill:'#506077','font-size':12},ylabel));return{px,py}}
function path(points,p,close=false){return points.map((q,i)=>`${i?'L':'M'}${p.px(q[0]).toFixed(2)},${p.py(q[1]).toFixed(2)}`).join(' ')+(close?' Z':'')}
function drawOrbit(id,key,color,index){const svg=document.getElementById(id),m=DATA.models[key],q=m.pca;svg.replaceChildren();const L=70,T=28,W=520,H=350;let [x0,x1]=bounds(q.map(p=>p[0])),[y0,y1]=bounds(q.map(p=>p[1]));const u=Math.max((x1-x0)/W,(y1-y0)/H),xc=(x0+x1)/2,yc=(y0+y1)/2;x0=xc-u*W/2;x1=xc+u*W/2;y0=yc-u*H/2;y1=yc+u*H/2;const p=axes(svg,{x0,x1,y0,y1,L,T,W,H,xlabel:'PC 1 (raw score)',ylabel:'PC 2 (raw score)'});svg.append(el('path',{d:path(q,p,true),fill:'none',stroke:color,'stroke-width':3.2}));q.forEach((v,i)=>{if(i%30===0)svg.append(el('circle',{cx:p.px(v[0]),cy:p.py(v[1]),r:2.4,fill:color}))});const a=q[0],s=q[index];svg.append(el('circle',{cx:p.px(a[0]),cy:p.py(a[1]),r:5,fill:'#16233a'}));svg.append(el('text',{x:p.px(a[0])+8,y:p.py(a[1])-8,fill:'#16233a','font-size':11},'0°'));svg.append(el('circle',{cx:p.px(s[0]),cy:p.py(s[1]),r:6.5,fill:'white',stroke:color,'stroke-width':3}));document.getElementById(key==='Normal VAE'?'normal-scale':'so2-scale').textContent=`Raw PCA ranges — PC 1: [${fmt(Math.min(...q.map(v=>v[0])))}, ${fmt(Math.max(...q.map(v=>v[0])))}]; PC 2: [${fmt(Math.min(...q.map(v=>v[1])))}, ${fmt(Math.max(...q.map(v=>v[1])))}]. PC1+PC2 explain ${(100*m.pca_explained_variance).toFixed(1)}% of this centered orbit's variance.`;document.getElementById(key==='Normal VAE'?'normal-current':'so2-current').innerHTML=`<strong>θ=${DATA.angles[index]}°</strong> &nbsp; relative RMS <b>${fmt(m.residual[index])}</b> &nbsp;|&nbsp; raw posterior RMS <b>${fmt(m.posterior_rms[index])}</b>`}
function lineChart(id,series,{title,ylabel,zero=false,band=null,index=0}){const svg=document.getElementById(id);svg.replaceChildren();const L=58,T=36,W=svg.viewBox.baseVal.width-78,H=svg.viewBox.baseVal.height-92;const values=flat(series.map(s=>s.values).concat(band?[band.lo,band.hi]:[]));const [y0,y1]=bounds(values,zero),x0=DATA.angles[0],x1=DATA.angles.at(-1);const p=axes(svg,{x0,x1,y0,y1,L,T,W,H,xlabel:'angle (degrees)',ylabel});svg.append(el('text',{x:L,y:19,fill:'#16233a','font-size':13,'font-weight':700},title));if(band){const top=band.hi.map((v,i)=>[DATA.angles[i],v]),bot=band.lo.map((v,i)=>[DATA.angles[i],v]).reverse();svg.append(el('path',{d:path(top.concat(bot),p,true),fill:band.fill,stroke:'none'}))}for(const s of series){svg.append(el('path',{d:path(s.values.map((v,i)=>[DATA.angles[i],v]),p),fill:'none',stroke:s.color,'stroke-width':s.width||2.7}));svg.append(el('text',{x:L+8,y:T+18+series.indexOf(s)*16,fill:s.color,'font-size':10},s.name));const v=s.values[index];if(Number.isFinite(v))svg.append(el('circle',{cx:p.px(DATA.angles[index]),cy:p.py(v),r:4.4,fill:'white',stroke:s.color,'stroke-width':2.3}))}return{y0,y1}}
function drawPosterior(index){const n=DATA.models['Normal VAE'],s=DATA.models['SO(2) VAE'];lineChart('posterior-residual-chart',[{name:'normal VAE',values:n.residual,color:C.normal},{name:'SO(2) VAE',values:s.residual,color:C.so2}],{title:'Dense posterior residual: eμ(θ)',ylabel:'relative RMS',zero:true,index});document.getElementById('posterior-residual-readout').innerHTML=`<strong>θ=${DATA.angles[index]}°</strong> normal: <b>${fmt(n.residual[index])}</b> (RMS ${fmt(n.posterior_rms[index])}) &nbsp;|&nbsp; SO(2): <b>${fmt(s.residual[index])}</b> (RMS ${fmt(s.posterior_rms[index])})`}
function drawPhase(id,kind,index){const svg=document.getElementById(id),d=DATA.f1[kind],labels=DATA.selected_f1_copies;svg.replaceChildren();const L=68,T=30,W=520,H=360,p=axes(svg,{x0:-1.4,x1:1.4,y0:-1.4,y1:1.4,L,T,W,H,xlabel:'real: zθ / z0',ylabel:'imaginary: zθ / z0'});const cx=p.px(0),cy=p.py(0),r=Math.abs(p.px(1)-cx);svg.append(el('circle',{cx,cy,r,fill:'none',stroke:'#9aa7b8','stroke-width':1.7}));for(let j=0;j<labels.length;j++){if(!d.valid[j]){svg.append(el('text',{x:L+7,y:T+18+j*16,fill:'#a16b13','font-size':10},`F1 copy ${labels[j]}: |z0| below threshold`));continue}const q=d.normalized.map(row=>row[j]);svg.append(el('path',{d:path(q,p,true),fill:'none',stroke:[C.so2,'#e05b67','#7c55d9'][j], 'stroke-width':2.6}));const v=q[index];svg.append(el('circle',{cx:p.px(v[0]),cy:p.py(v[1]),r:5,fill:'white',stroke:[C.so2,'#e05b67','#7c55d9'][j],'stroke-width':2.5}));svg.append(el('text',{x:L+7,y:T+18+j*16,fill:[C.so2,'#e05b67','#7c55d9'][j],'font-size':10},`F1 copy ${labels[j]}`))}const phase=median(d.phase_error_degrees[index].map(Math.abs)),gain=median(d.gain[index]),amp=median(d.amplitude[index]);document.getElementById(kind==='trained'?'trained-phase-readout':'untrained-phase-readout').innerHTML=`<strong>θ=${DATA.angles[index]}°</strong> median |phase error| <b>${fmt(phase)}°</b> &nbsp;|&nbsp; median gain <b>${fmt(gain)}</b> &nbsp;|&nbsp; raw magnitude <b>${fmt(amp)}</b>`}
function drawF1Curves(index){const t=DATA.f1.trained,u=DATA.f1.untrained,n=DATA.f1.normal_null;const absPhase=d=>d.phase_error_degrees.map(row=>median(row.map(Math.abs)));const med=d=>d.map(row=>median(row));lineChart('phase-error-chart',[{name:'trained SO(2)',values:absPhase(t),color:C.so2},{name:'untrained SO(2)',values:absPhase(u),color:C.untrained},{name:'normal null median',values:n.phase_abs_error_quantiles[1],color:C.null}],{title:'absolute phase error',ylabel:'degrees',zero:true,band:{lo:n.phase_abs_error_quantiles[0],hi:n.phase_abs_error_quantiles[2],fill:'#dce3ed'},index});lineChart('gain-chart',[{name:'trained SO(2)',values:med(t.gain),color:C.so2},{name:'untrained SO(2)',values:med(u.gain),color:C.untrained},{name:'normal null median',values:n.gain_quantiles[1],color:C.null}],{title:'gain: |zθ| / |z0|',ylabel:'ratio',zero:false,band:{lo:n.gain_quantiles[0],hi:n.gain_quantiles[2],fill:'#dce3ed'},index});lineChart('amplitude-chart',[{name:'trained SO(2)',values:med(t.amplitude),color:C.so2},{name:'untrained SO(2)',values:med(u.amplitude),color:C.untrained},{name:'normal null median',values:n.amplitude_quantiles[1],color:C.null}],{title:'raw disk-mean magnitude',ylabel:'magnitude',zero:true,band:{lo:n.amplitude_quantiles[0],hi:n.amplitude_quantiles[2],fill:'#dce3ed'},index});document.getElementById('f1-readout').innerHTML=`Grey band: 5th–95th percentile over ${n.ensemble_size} fixed normal scalar-pair null projections; ${n.valid_controls}/${n.ensemble_size} controls had all three selected θ=0 vectors above the ${fmt(DATA.min_magnitude)} threshold.`}
function drawKernel(id,channel){const svg=document.getElementById(id),f=DATA.kernel.vectors;svg.replaceChildren();let max=0;for(let c=0;c<3;c++)for(let r=0;r<9;r++)for(let q=0;q<9;q++)max=Math.max(max,Math.hypot(f[0][c][r][q],f[1][c][r][q]));const L=30,T=18,S=300,cell=S/9;for(let r=0;r<9;r++)for(let q=0;q<9;q++){const a=f[0][channel][r][q],b=f[1][channel][r][q],m=Math.hypot(a,b)/max,x=L+q*cell,y=T+r*cell;svg.append(el('rect',{x,y,width:cell,height:cell,fill:`rgb(${Math.round(246-28*m)},${Math.round(248-130*m)},${Math.round(251-198*m)})`,stroke:'#e4eaf2'}));const cx=x+cell/2,cy=y+cell/2;svg.append(el('line',{x1:cx,y1:cy,x2:cx+a/max*cell*.31,y2:cy-b/max*cell*.31,stroke:'#17233a','stroke-width':1.6}))}svg.append(el('text',{x:L,y:344,fill:'#5a6a80','font-size':11},`shared vector-magnitude scale: 0–${fmt(max)}`));return max}
function drawKernelEnergy(){const f=DATA.kernel.vectors,energy=[0,1,2].map(c=>Math.sqrt(f.flatMap(k=>k[c].flat()).reduce((a,v)=>a+v*v,0)));const svg=document.getElementById('kernel-energy-chart');svg.replaceChildren();const L=72,T=38,W=1110,H=130,[y0,y1]=[0,Math.max(...energy)*1.13||1],p=axes(svg,{x0:0,x1:3,y0,y1,L,T,W,H,xlabel:'input color',ylabel:'F1 kernel L2 norm'});['red','green','blue'].forEach((name,i)=>{const x=L+(i+.2)*W/3,w=W/3*.6,y=p.py(energy[i]);svg.append(el('rect',{x,y,width:w,height:T+H-y,fill:[C.normal,'#19a765',C.so2][i]}));svg.append(el('text',{x:x+w/2,y:T+H+18,'text-anchor':'middle',fill:'#43526a','font-size':11},name));svg.append(el('text',{x:x+w/2,y:y-7,'text-anchor':'middle',fill:'#43526a','font-size':11},fmt(energy[i])))});document.getElementById('kernel-note').textContent=`Largest-L2 learned stem F1 copy: ${DATA.kernel.copy}. The three numeric bars and every vector field share the displayed raw magnitude context.`}
function fillQuarters(){const b=document.getElementById('quarter-rows');for(let i=0;i<DATA.quarters.angles.length;i++){const r=document.createElement('tr');r.innerHTML=`<td>${DATA.quarters.angles[i]}°</td><td>${fmt(DATA.quarters.normal[i])}</td><td>${fmt(DATA.quarters.so2[i])}</td>`;b.append(r)}}
function update(){const i=Number(document.getElementById('angle').value);document.getElementById('angle-value').textContent=`${DATA.angles[i]}°`;drawOrbit('normal-chart','Normal VAE',C.normal,i);drawOrbit('so2-chart','SO(2) VAE',C.so2,i);drawPosterior(i);drawPhase('trained-phase-chart','trained',i);drawPhase('untrained-phase-chart','untrained',i);drawF1Curves(i)}
function sectionWithHeading(prefix){return [...document.querySelectorAll('main > section')].find(section=>{const heading=section.querySelector(':scope > h2');return heading&&heading.textContent.startsWith(prefix)})}
function explanation(id){return document.getElementById(id).content.cloneNode(true)}
function upgradeExplanations(){const control=document.querySelector('main > section.control'),orbit=sectionWithHeading('1.'),posterior=sectionWithHeading('2.'),f1=sectionWithHeading('3.');if(!control||!orbit||!posterior||!f1)return;const orbitGrid=orbit.querySelector(':scope > .grid2'),posteriorPlot=posterior.querySelector(':scope > article.card'),posteriorMetrics=posterior.querySelector(':scope > .metrics'),f1Grid=f1.querySelector(':scope > .grid2'),f1Curves=f1.querySelector(':scope > article.card');if(!orbitGrid||!posteriorPlot||!posteriorMetrics||!f1Grid||!f1Curves)return;control.classList.add('orbit-control');orbit.replaceChildren(explanation('orbit-explanation'),control,orbitGrid);posterior.replaceChildren(explanation('posterior-explanation'),posteriorPlot,posteriorMetrics);f1Curves.querySelector('h3').textContent='F1 phase, gain, amplitude, and the deliberately artificial normal null';const caption=f1Curves.querySelector('.caption');if(caption)caption.innerHTML='Each coloured curve is the <strong>median across the three preselected copies</strong>. Blue is trained SO(2); purple is the same copy indices in a seeded untrained SO(2) model; grey is the median and 5th–95th-percentile envelope over 32 arbitrary scalar-pair constructions from the normal VAE. Grey is <strong>not</strong> normal F1.';const reading=document.createElement('ol');reading.className='steps';reading.innerHTML='<li><strong>Absolute phase error:</strong> 0° is ideal.</li><li><strong>Gain:</strong> 1 is ideal.</li><li><strong>Raw amplitude:</strong> a small pooled arrow makes phase unreliable because vector contributions can cancel.</li>';f1Curves.insertBefore(reading,f1Curves.querySelector('.grid3'));f1.replaceChildren(explanation('f1-explanation'),f1Grid,f1Curves)}
function inlineTexHtml(tex){const slash=String.fromCharCode(92);let value=tex.trim().replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');for(const [from,to] of [[slash+'mathbb{R}','ℝ'],[slash+'tilde{',''],[slash+'bar{',''],[slash+'theta','θ'],[slash+'mu','μ'],[slash+'rho','ρ'],[slash+'phi','φ'],[slash+'Delta','Δ'],[slash+'varepsilon','ε'],[slash+'in',' ∈ '],[slash+'times','×'],[slash+'cdot','·'],[slash+'approx','≈'],[slash+'circ','°'],[slash+'quad',' '],[slash+',',' ']])value=value.split(from).join(to);value=value.replace(/\^\{([^{}]+)\}/g,'<sup>$1</sup>').replace(/_\{([^{}]+)\}/g,'<sub>$1</sub>').replace(/\^([A-Za-z0-9θμρφΔε])/g,'<sup>$1</sup>').replace(/_([A-Za-z0-9θμρφΔε])/g,'<sub>$1</sub>').replace(/[{}]/g,'');return value}
function renderInlineTex(root){const slash=String.fromCharCode(92),start=slash+'(',end=slash+')',walker=document.createTreeWalker(root,NodeFilter.SHOW_TEXT),nodes=[];let node;while((node=walker.nextNode())){const parent=node.parentElement;if(parent&&!parent.closest('math,script,style,code'))nodes.push(node)}for(const text of nodes){if(!text.data.includes(start))continue;const fragment=document.createDocumentFragment();let cursor=0,open,close;while((open=text.data.indexOf(start,cursor))>=0&&(close=text.data.indexOf(end,open+start.length))>=0){fragment.append(document.createTextNode(text.data.slice(cursor,open)));const span=document.createElement('span');span.className='inline-math';span.innerHTML=inlineTexHtml(text.data.slice(open+start.length,close));fragment.append(span);cursor=close+end.length}fragment.append(document.createTextNode(text.data.slice(cursor)));text.replaceWith(fragment)}}
function drawSpatialMap(target,map){const side=320,cell=side/32,svg=el('svg',{viewBox:`0 0 ${side} ${side}`,role:'img','aria-label':'Spatial latent PCA false-colour map'});for(let row=0;row<32;row++)for(let column=0;column<32;column++){const rgb=map[row][column];svg.append(el('rect',{x:column*cell,y:row*cell,width:cell,height:cell,fill:`rgb(${rgb[0]},${rgb[1]},${rgb[2]})`,stroke:'none'}))}target.append(svg)}
function drawRgbCanvas(target,map,{smooth,label}){const height=map.length,width=map[0].length,source=document.createElement('canvas'),canvas=document.createElement('canvas'),pixels=new Uint8ClampedArray(width*height*4),sourceContext=source.getContext('2d'),context=canvas.getContext('2d');source.width=width;source.height=height;canvas.width=512;canvas.height=512;for(let row=0;row<height;row++)for(let column=0;column<width;column++){const offset=4*(row*width+column),rgb=map[row][column];pixels[offset]=rgb[0];pixels[offset+1]=rgb[1];pixels[offset+2]=rgb[2];pixels[offset+3]=255}sourceContext.putImageData(new ImageData(pixels,width,height),0,0);context.imageSmoothingEnabled=smooth;context.drawImage(source,0,0,canvas.width,canvas.height);canvas.setAttribute('role','img');canvas.setAttribute('aria-label',label);target.append(canvas)}
function drawSpatialPaired(){const n=DATA.spatial_pca['Normal VAE'].relative_edge_rms,s=DATA.spatial_pca['SO(2) VAE'].relative_edge_rms,svg=document.getElementById('spatial-paired-chart');svg.replaceChildren();const L=62,T=34,W=svg.viewBox.baseVal.width-88,H=svg.viewBox.baseVal.height-94,[y0,y1]=bounds(n.concat(s));const p=axes(svg,{x0:0,x1:1,y0,y1,L,T,W,H,xlabel:'same fixed-25 patch, compared by model',ylabel:'relative neighbour-jump ratio',showXTickLabels:false}),nx=p.px(.24),sx=p.px(.76);svg.append(el('text',{x:L,y:19,fill:'#16233a','font-size':13,'font-weight':700},'Each line is one patch; lower means less local roughness on this ratio'));n.forEach((v,i)=>{svg.append(el('line',{x1:nx,y1:p.py(v),x2:sx,y2:p.py(s[i]),stroke:'#cbd5e1','stroke-width':1.5}));svg.append(el('circle',{cx:nx,cy:p.py(v),r:2.8,fill:C.normal}));svg.append(el('circle',{cx:sx,cy:p.py(v),r:2.8,fill:C.so2}))});for(const [x,values,color,label] of [[nx,n,C.normal,'Normal VAE'],[sx,s,C.so2,'SO(2) VAE']]){const y=p.py(median(values));svg.append(el('line',{x1:x-14,y1:y,x2:x+14,y2:y,stroke:color,'stroke-width':4}));svg.append(el('text',{x,y:T+H+18,'text-anchor':'middle',fill:color,'font-size':11},label))}}
function drawSpatialDelta(){const n=DATA.spatial_pca['Normal VAE'].relative_edge_rms,s=DATA.spatial_pca['SO(2) VAE'].relative_edge_rms,d=s.map((v,i)=>v-n[i]),svg=document.getElementById('spatial-delta-chart');svg.replaceChildren();const L=62,T=34,W=svg.viewBox.baseVal.width-88,H=svg.viewBox.baseVal.height-94,[y0,y1]=bounds(d,true),p=axes(svg,{x0:0,x1:d.length-1,y0,y1,L,T,W,H,xlabel:'fixed25 patch index',ylabel:'SO(2) − normal'});svg.append(el('text',{x:L,y:19,fill:'#16233a','font-size':13,'font-weight':700},'Paired difference in relative edge RMS'));svg.append(el('line',{x1:p.px(0),y1:p.py(0),x2:p.px(d.length-1),y2:p.py(0),stroke:'#64748b','stroke-width':1.8}));d.forEach((v,i)=>svg.append(el('circle',{cx:p.px(i),cy:p.py(v),r:3.4,fill:C.so2})))}
function appendSpatialPcaSection(){
  const footer=document.querySelector('main > footer'),section=document.createElement('section'),maps=DATA.spatial_pca,paper=maps.paper_style,paperGrid=document.createElement('div'),commonHeading=document.createElement('article'),grid=document.createElement('div'),summary=document.createElement('article');
  section.append(explanation('spatial-pca-explanation'),explanation('paper-style-pca-explanation'));
  paperGrid.className='paper-style-grid';
  maps.display_indices.forEach((index,position)=>{for(const [label,key,color,map] of [['original input','originals_rgb','#16233a',paper.originals_rgb[position]],['normal VAE PCA-RGB','Normal VAE',C.normal,paper['Normal VAE'].rgb[position]],['SO(2) VAE PCA-RGB','SO(2) VAE',C.so2,paper['SO(2) VAE'].rgb[position]]]){const card=document.createElement('article');card.className='card';card.innerHTML=`<h3 style="color:${color}">patch ${index} · ${label}</h3><p class="caption">${key==='originals_rgb'?'reference only; never blended into the PCA map':'fresh whole-grid PCA; joint within-tile RGB min–max; bilinear display'}</p>`;drawRgbCanvas(card,map,{smooth:true,label:`Patch ${index} ${label}`});paperGrid.append(card)}});
  commonHeading.className='card';commonHeading.style.marginTop='22px';commonHeading.innerHTML='<h3>Comparable common-scale maps and quantitative companion</h3><p class="caption">These retain one disk-masked PCA fit and one Q99 score scale per model over all 25 fixed patches. The hard-edged native cells and the paired r<sub>edge</sub> summary are intentional: unlike the paper-style tiles above, they preserve a common within-model scale.</p>';
  grid.className='spatial-map-grid';
  maps.display_indices.forEach((index,position)=>{for(const [label,key,color] of [['Normal VAE','Normal VAE',C.normal],['SO(2) VAE','SO(2) VAE',C.so2]]){const card=document.createElement('article');card.className='card';card.innerHTML=`<h3 style="color:${color}">patch ${index} · ${label}</h3><p class="caption">native 32×32 cells; neutral exterior excluded</p>`;drawSpatialMap(card,maps[key].rgb[position]);grid.append(card)}});
  summary.className='card';summary.style.marginTop='20px';summary.innerHTML='<h3>Quantitative companion: all 25 patches, paired on a shared raw scale</h3><p class="caption">Left: a red dot and blue dot measure the exact same patch; their light line makes the within-patch change visible. Vertical position is r<sub>edge</sub>, the local neighbour jump divided by that field’s own spatial variation: lower means less locally rough on this ratio, not a better VAE overall. Right: Δr<sub>edge</sub>=SO(2)−normal; above zero means the SO(2) field is rougher for that patch.</p><div class="spatial-summary-grid"><svg id="spatial-paired-chart" viewBox="0 0 620 390" role="img" aria-label="Paired local coherence comparison"></svg><svg id="spatial-delta-chart" viewBox="0 0 620 390" role="img" aria-label="Paired local coherence differences"></svg></div><p id="spatial-readout" class="readout"></p>';
  section.append(paperGrid,commonHeading,grid,summary);footer.before(section);drawSpatialPaired();drawSpatialDelta();
  const n=maps['Normal VAE'],s=maps['SO(2) VAE'],d=s.relative_edge_rms.map((v,i)=>v-n.relative_edge_rms[i]),positive=d.filter(v=>v>0).length;
  document.getElementById('spatial-readout').innerHTML=`<strong>How this sample reads:</strong> the median r<sub>edge</sub> is <b>${fmt(median(n.relative_edge_rms))}</b> for normal and <b>${fmt(median(s.relative_edge_rms))}</b> for SO(2). The mean Δr<sub>edge</sub> is <b>${fmt(d.reduce((a,v)=>a+v,0)/d.length)}</b>, and <b>${positive}/${d.length}</b> dots are above zero. So this local metric does not show an SO(2) coherence advantage here. Raw median edge/centred/posterior RMS are normal ${fmt(median(n.edge_rms))}/${fmt(median(n.centered_spatial_rms))}/${fmt(median(n.posterior_rms))} and SO(2) ${fmt(median(s.edge_rms))}/${fmt(median(s.centered_spatial_rms))}/${fmt(median(s.posterior_rms))}.`;
}
function appendPointwiseProbeSection(){
  const footer=document.querySelector('main > footer'),section=document.createElement('section'),probe=DATA.pointwise_rgb_probe,grid=document.createElement('div'),summary=document.createElement('article'),average=values=>values.reduce((total,value)=>total+value,0)/values.length;
  section.append(explanation('pointwise-rgb-probe-explanation'));
  summary.className='card';summary.style.marginTop='20px';summary.innerHTML=`<h3>What was fitted, what was held out, and what the score means</h3><p class="caption">One 16-to-3 affine map was fitted independently per model using all 32×32 cells of patches ${probe.train_indices[0]}–${probe.train_indices.at(-1)}. These five rows are patches ${probe.heldout_indices[0]}–${probe.heldout_indices.at(-1)}: none contributed a fitted weight. The score is pixel R² versus the RGB mean from the training patches; values above zero improve on that fixed constant-colour baseline.</p><p class="readout"><strong>Mean held-out result:</strong> normal R² <b>${fmt(average(probe['Normal VAE'].heldout_r2))}</b>, MSE <b>${fmt(average(probe['Normal VAE'].heldout_mse))}</b>; SO(2) R² <b>${fmt(average(probe['SO(2) VAE'].heldout_r2))}</b>, MSE <b>${fmt(average(probe['SO(2) VAE'].heldout_mse))}</b>. The predictions retain some broad layout but are blurred, which is exactly the limitation a 1×1 linear readout should expose.</p>`;
  grid.className='probe-grid';
  probe.heldout_indices.forEach((index,position)=>{const row=document.createElement('article');row.className='card probe-row';row.innerHTML=`<h3>held-out patch ${index}</h3><p class="caption">The two predictions use only the local 16-number posterior descriptor at each displayed cell.</p>`;for(const [label,color,map,metric] of [['target: downsampled source','#16233a',probe.target_rgb[position],'reference target; not a prediction'],['normal VAE prediction',C.normal,probe['Normal VAE'].predicted_rgb[position],`R² ${fmt(probe['Normal VAE'].heldout_r2[position])}; MSE ${fmt(probe['Normal VAE'].heldout_mse[position])}`],['SO(2) VAE prediction',C.so2,probe['SO(2) VAE'].predicted_rgb[position],`R² ${fmt(probe['SO(2) VAE'].heldout_r2[position])}; MSE ${fmt(probe['SO(2) VAE'].heldout_mse[position])}`]]){const tile=document.createElement('div');tile.className='probe-tile';tile.innerHTML=`<h4 style="color:${color}">${label}</h4>`;drawRgbCanvas(tile,map,{smooth:true,label:`Held-out patch ${index}: ${label}`});const detail=document.createElement('p');detail.textContent=metric;tile.append(detail);row.append(tile)}grid.append(row)});
  section.append(summary,grid);footer.before(section);
}
upgradeExplanations();
appendSpatialPcaSection();
appendPointwiseProbeSection();
renderInlineTex(document.querySelector('main'));
document.getElementById('patch-index').textContent=DATA.patch_index;fillQuarters();drawKernel('kernel-red',0);drawKernel('kernel-green',1);drawKernel('kernel-blue',2);drawKernelEnergy();document.getElementById('angle').addEventListener('input',update);update();
</script>"""
    return (
        template
        .replace("__PAYLOAD__", safe_payload)
        .replace("__MAX_ANGLE__", str(max_angle))
        .replace("__LABEL__", EXPLORATORY_LABEL)
    )


__all__ = [
    "EXPLORATORY_LABEL",
    "IMAGE_DISK_RADIUS",
    "LATENT_DISK_RADIUS",
    "DenseOrbitPopulation",
    "OrbitSweep",
    "PaperStyleSpatialPca",
    "PointwiseRgbProbe",
    "QuarterResidualSummary",
    "SpatialLatentPcaDiagnostic",
    "centered_disk_mask",
    "choose_stem_f1_copy",
    "collect_dense_mu_population",
    "collect_orbit",
    "continuous_rotate",
    "disk_mean_f1",
    "disk_mean_scalar",
    "downsample_uint8_rgb_patches",
    "isotropic_plot_bounds",
    "masked_relative_rms",
    "normal_pair_null_summary",
    "normalized_f1_phase",
    "paper_style_spatial_pca",
    "pca_orbit_2d",
    "pca_orbit_explained_variance",
    "pointwise_rgb_probe",
    "project_scalar_pair_means",
    "quarter_residual_summary",
    "render_dense_orbit_population_png",
    "render_f1_phase_png",
    "render_kernel_mechanism_png",
    "render_latent_orbits_png",
    "render_paper_style_spatial_pca_png",
    "render_pointwise_rgb_probe_png",
    "render_spatial_latent_pca_png",
    "seeded_scalar_pair_projection",
    "select_f1_copies",
    "spatial_latent_pca_diagnostic",
    "summarize_dense_orbit_population",
    "unpack_final_encoder_f1",
    "vector_phase_diagnostic",
    "write_offline_html",
]
