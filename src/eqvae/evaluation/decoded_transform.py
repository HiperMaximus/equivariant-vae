# Copyright 2026 HiperMaximus
"""Metrics and exact spatial actions for decoded latent-transform audits."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
import torch

TensorTransform = Callable[[torch.Tensor], torch.Tensor]
PAIR_MATRIX_DIMENSIONS = 2
NCHW_DIMENSIONS = 4
EXACT_D4_NAMES = (
    "identity",
    "rot90",
    "rot180",
    "rot270",
    "flip_h",
    "flip_diag",
    "flip_v",
    "flip_anti_diag",
)
EXACT_D4_NONIDENTITY_NAMES = EXACT_D4_NAMES[1:]


def exact_spatial_transform(values: torch.Tensor, name: str) -> torch.Tensor:
    """Apply a locked exact D4 transform to the final two tensor axes.

    Returns:
        Exactly re-indexed tensor.

    Raises:
        ValueError: If ``name`` is not a declared D4 element.

    """
    transforms: dict[str, TensorTransform] = {
        "identity": lambda tensor: tensor,
        "rot90": lambda tensor: torch.rot90(tensor, 1, dims=(-2, -1)),
        "rot180": lambda tensor: torch.rot90(tensor, 2, dims=(-2, -1)),
        "rot270": lambda tensor: torch.rot90(tensor, 3, dims=(-2, -1)),
        "flip_h": lambda tensor: torch.flip(tensor, dims=(-1,)),
        "flip_v": lambda tensor: torch.flip(tensor, dims=(-2,)),
        "flip_diag": lambda tensor: tensor.transpose(-2, -1),
        "flip_anti_diag": lambda tensor: torch.flip(
            tensor.transpose(-2, -1),
            dims=(-2, -1),
        ),
    }
    try:
        return transforms[name](values)
    except KeyError as error:
        message = f"unknown exact spatial transform: {name}"
        raise ValueError(message) from error


def inverse_exact_transform_name(name: str) -> str:
    """Return the inverse name under the locked exact D4 convention.

    Returns:
        Name of the exact inverse operation.

    Raises:
        ValueError: If ``name`` is not a declared D4 element.

    """
    inverse = {
        "identity": "identity",
        "rot90": "rot270",
        "rot180": "rot180",
        "rot270": "rot90",
        "flip_h": "flip_h",
        "flip_v": "flip_v",
        "flip_diag": "flip_diag",
        "flip_anti_diag": "flip_anti_diag",
    }
    try:
        return inverse[name]
    except KeyError as error:
        message = f"unknown exact spatial transform: {name}"
        raise ValueError(message) from error


def masked_mse_per_image(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Return per-image MSE over all channels and selected spatial cells.

    Returns:
        One masked MSE value per image.

    """
    _validate_image_pair_and_mask(prediction, target, mask)
    difference = prediction.to(torch.float32) - target.to(torch.float32)
    return difference[:, :, mask].square().mean(dim=(1, 2))


def masked_mae_per_image(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Return per-image MAE over all channels and selected spatial cells.

    Returns:
        One masked MAE value per image.

    """
    _validate_image_pair_and_mask(prediction, target, mask)
    difference = prediction.to(torch.float32) - target.to(torch.float32)
    return difference[:, :, mask].abs().mean(dim=(1, 2))


def aggregate_rms_ratio(
    numerator_mse: np.ndarray,
    denominator_mse: np.ndarray,
    *,
    angle_indices: Sequence[int] | None = None,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Aggregate repeated-angle MSE per patch, then return the RMS ratio.

    Returns:
        One aggregate RMS ratio per patch.

    Raises:
        ValueError: If matrices are malformed, empty, non-finite, or negative.

    """
    numerator = np.asarray(numerator_mse, dtype=np.float64)
    denominator = np.asarray(denominator_mse, dtype=np.float64)
    if numerator.shape != denominator.shape or numerator.ndim != PAIR_MATRIX_DIMENSIONS:
        message = "numerator and denominator must be equal [patch,angle] matrices"
        raise ValueError(message)
    if angle_indices is not None:
        indices = np.asarray(tuple(angle_indices), dtype=np.int64)
        numerator = numerator[:, indices]
        denominator = denominator[:, indices]
    if numerator.shape[1] == 0:
        message = "at least one angle is required"
        raise ValueError(message)
    if not np.isfinite(numerator).all() or not np.isfinite(denominator).all():
        message = "MSE matrices must be finite"
        raise ValueError(message)
    if (numerator < 0).any() or (denominator < 0).any():
        message = "MSE matrices must be non-negative"
        raise ValueError(message)
    return np.sqrt(numerator.mean(axis=1)) / (
        np.sqrt(denominator.mean(axis=1)) + epsilon
    )


def out_of_range_per_image(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return pre-clamp out-of-range fraction and mean overshoot per image.

    Returns:
        Per-image out-of-range fractions and mean overshoots.

    Raises:
        ValueError: If ``values`` is not a non-empty NCHW tensor.

    """
    if values.ndim != NCHW_DIMENSIONS or values.shape[0] == 0:
        message = "values must be a non-empty NCHW tensor"
        raise ValueError(message)
    values_f32 = values.to(torch.float32)
    magnitude = values_f32.abs()
    return (
        (magnitude > 1.0).to(torch.float32).mean(dim=(1, 2, 3)),
        torch.relu(magnitude - 1.0).mean(dim=(1, 2, 3)),
    )


def gradient_mse_per_image(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Return finite-difference gradient MSE within an eroded spatial mask.

    Returns:
        One gradient-domain MSE value per image.

    """
    _validate_image_pair_and_mask(prediction, target, mask)
    difference = prediction.to(torch.float32) - target.to(torch.float32)
    dx = difference[..., :, 1:] - difference[..., :, :-1]
    dy = difference[..., 1:, :] - difference[..., :-1, :]
    mask_x = mask[:, 1:] & mask[:, :-1]
    mask_y = mask[1:, :] & mask[:-1, :]
    sum_x = dx[:, :, mask_x].square().sum(dim=(1, 2))
    sum_y = dy[:, :, mask_y].square().sum(dim=(1, 2))
    count = prediction.shape[1] * (int(mask_x.sum()) + int(mask_y.sum()))
    return (sum_x + sum_y) / count


def paired_bootstrap_median_difference(  # noqa: PLR0913
    normal: Sequence[float],
    so2: Sequence[float],
    *,
    seed: int,
    draws: int = 10_000,
    favorable_direction: Literal["lower", "higher"] | None = "lower",
    cluster_ids: Sequence[str] | None = None,
) -> dict[str, float | int | None]:
    """Return a WSI-clustered paired interval for median SO2-normal.

    Returns:
        Descriptive paired-bootstrap result.

    Raises:
        ValueError: If paired values or draw count are malformed.

    """
    normal_values = np.asarray(normal, dtype=np.float64)
    so2_values = np.asarray(so2, dtype=np.float64)
    if normal_values.shape != so2_values.shape or normal_values.ndim != 1:
        message = "paired values must be equal one-dimensional arrays"
        raise ValueError(message)
    if normal_values.size == 0 or draws <= 0:
        message = "paired values and draws must be non-empty/positive"
        raise ValueError(message)
    differences = so2_values - normal_values
    labels = (
        tuple(str(value) for value in cluster_ids)
        if cluster_ids is not None
        else tuple(str(index) for index in range(differences.size))
    )
    if len(labels) != differences.size:
        message = "cluster ids must match the paired values"
        raise ValueError(message)
    unique_labels = sorted(set(labels))
    cluster_differences = np.asarray(
        [
            np.median(
                differences[
                    np.asarray(
                        [index for index, value in enumerate(labels) if value == label],
                        dtype=np.int64,
                    )
                ],
            )
            for label in unique_labels
        ],
        dtype=np.float64,
    )
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0,
        cluster_differences.size,
        size=(draws, cluster_differences.size),
    )
    medians = np.median(cluster_differences[indices], axis=1)
    favorable_patch_count = None
    favorable_cluster_count = None
    if favorable_direction == "lower":
        favorable_patch_count = int(np.sum(differences < 0))
        favorable_cluster_count = int(np.sum(cluster_differences < 0))
    elif favorable_direction == "higher":
        favorable_patch_count = int(np.sum(differences > 0))
        favorable_cluster_count = int(np.sum(cluster_differences > 0))
    elif favorable_direction is not None:
        message = f"unknown favorable direction: {favorable_direction}"
        raise ValueError(message)
    return {
        "n_patches": int(differences.size),
        "n_clusters": int(cluster_differences.size),
        "draws": draws,
        "seed": seed,
        "patch_median_difference_so2_minus_normal": float(np.median(differences)),
        "cluster_median_difference_so2_minus_normal": float(
            np.median(cluster_differences),
        ),
        "interval_low": float(np.quantile(medians, 0.025)),
        "interval_high": float(np.quantile(medians, 0.975)),
        "so2_favorable_patch_count": favorable_patch_count,
        "so2_favorable_cluster_count": favorable_cluster_count,
    }


def _validate_image_pair_and_mask(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    if prediction.shape != target.shape or prediction.ndim != NCHW_DIMENSIONS:
        message = "prediction and target must be equal NCHW tensors"
        raise ValueError(message)
    if prediction.device != target.device or mask.device != prediction.device:
        message = "prediction, target and mask must share one device"
        raise ValueError(message)
    if mask.dtype != torch.bool or mask.shape != prediction.shape[-2:]:
        message = "mask must be bool with the image spatial shape"
        raise ValueError(message)
    if not bool(mask.any()):
        message = "mask must select at least one cell"
        raise ValueError(message)
