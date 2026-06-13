# Copyright 2026 HiperMaximus
"""Repo-owned reconstruction metrics for spec 0001."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as nn_functional

SSIM_WINDOW_SIZE = 11
SSIM_SIGMA = 1.5
SSIM_DATA_RANGE = 1.0
SSIM_K1 = 0.01
SSIM_K2 = 0.03
NCHW_DIMENSIONS = 4
IMAGE_DOMAIN_TOLERANCE = 1.0e-6
MIN_MSE_FOR_FINITE_PSNR = 0.0


@dataclass(frozen=True)
class MetricSummary:
    """Population summary for one metric."""

    n: int
    mean: float | None
    std: float | None
    inf_count: int = 0
    finite_mean: float | None = None
    finite_std: float | None = None

    def as_dict(self) -> dict[str, int | float | None]:
        """Return a JSON-ready metric summary.

        Returns:
            JSON-safe metric payload.

        """
        payload: dict[str, int | float | None] = {
            "n": self.n,
            "mean": self.mean,
            "std": self.std,
        }
        if self.inf_count > 0:
            payload.update(
                {
                    "inf_count": self.inf_count,
                    "finite_mean": self.finite_mean,
                    "finite_std": self.finite_std,
                },
            )
        return payload


def normalized_to_image_domain(tensor: torch.Tensor) -> torch.Tensor:
    """Project normalized `[-1, 1]` tensors to clamped image-domain `[0, 1]`.

    Returns:
        FP32 image-domain tensor.

    """
    return tensor.to(dtype=torch.float32).add(1.0).div(2.0).clamp(0.0, 1.0)


def mae_per_image(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return per-image mean absolute error.

    Returns:
        Per-image MAE tensor.

    """
    prediction_f32, target_f32 = _validate_pair(prediction, target)
    return (prediction_f32 - target_f32).abs().mean(dim=(1, 2, 3))


def mse_per_image(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return per-image mean squared error.

    Returns:
        Per-image MSE tensor.

    """
    prediction_f32, target_f32 = _validate_pair(prediction, target)
    return (prediction_f32 - target_f32).square().mean(dim=(1, 2, 3))


def psnr_per_image(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    data_range: float = SSIM_DATA_RANGE,
) -> torch.Tensor:
    """Return per-image PSNR on image-domain tensors.

    Returns:
        Per-image PSNR tensor.

    Raises:
        ValueError: If `data_range` is not positive or tensors are invalid.

    """
    if data_range <= 0.0:
        message = f"data_range must be positive, got {data_range}"
        raise ValueError(message)
    prediction_f32, target_f32 = _validate_pair(prediction, target)
    _validate_image_domain(prediction_f32, name="prediction")
    _validate_image_domain(target_f32, name="target")
    mse_values = mse_per_image(prediction_f32, target_f32)
    range_tensor = torch.tensor(
        data_range,
        dtype=mse_values.dtype,
        device=mse_values.device,
    )
    finite_mse = mse_values.clamp_min(torch.finfo(mse_values.dtype).tiny)
    psnr_values = 20.0 * torch.log10(range_tensor) - 10.0 * torch.log10(finite_mse)
    return torch.where(
        mse_values <= MIN_MSE_FOR_FINITE_PSNR,
        torch.full_like(psnr_values, float("inf")),
        psnr_values,
    )


def ssim_per_image(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    data_range: float = SSIM_DATA_RANGE,
    window_size: int = SSIM_WINDOW_SIZE,
    sigma: float = SSIM_SIGMA,
) -> torch.Tensor:
    """Return standard full-reference SSIM per image.

    The implementation uses the locked spec 0001 convention: FP32 math,
    Gaussian `11x11` window with `sigma=1.5`, grouped per-channel convolutions,
    and reflect padding so the full image contributes to the SSIM map.

    Returns:
        Per-image SSIM tensor.

    """
    _validate_ssim_parameters(
        data_range=data_range,
        window_size=window_size,
        sigma=sigma,
    )
    prediction_f32, target_f32 = _validate_pair(prediction, target)
    _validate_image_domain(prediction_f32, name="prediction")
    _validate_image_domain(target_f32, name="target")
    _validate_ssim_spatial_shape(prediction_f32, window_size=window_size)

    kernel = _ssim_kernel(
        channels=prediction_f32.shape[1],
        window_size=window_size,
        sigma=sigma,
        device=prediction_f32.device,
    )
    return _ssim_map(
        prediction_f32,
        target_f32,
        kernel=kernel,
        padding=window_size // 2,
        data_range=data_range,
    ).mean(dim=(1, 2, 3))


def summarize_metric(values: torch.Tensor, *, allow_inf: bool = False) -> MetricSummary:
    """Summarize per-image metric values with population standard deviation.

    Returns:
        Metric summary.

    Raises:
        ValueError: If values are empty or contain disallowed non-finite values.

    """
    values_f32 = values.to(dtype=torch.float32).flatten()
    if values_f32.numel() == 0:
        message = "Cannot summarize an empty metric tensor"
        raise ValueError(message)
    _validate_finite(values_f32, name="metric values", allow_inf=allow_inf)
    finite_mask = torch.isfinite(values_f32)
    inf_count = int(torch.isinf(values_f32).sum().item())
    if inf_count > 0:
        finite_values = values_f32[finite_mask]
        finite_mean, finite_std = _finite_mean_std_or_none(finite_values)
        return MetricSummary(
            n=values_f32.numel(),
            mean=None,
            std=None,
            inf_count=inf_count,
            finite_mean=finite_mean,
            finite_std=finite_std,
        )
    mean, std = _mean_std(values_f32)
    return MetricSummary(n=values_f32.numel(), mean=mean, std=std)


def reconstruction_metric_summaries(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, MetricSummary]:
    """Summarize spec 0001 reconstruction metrics for normalized tensors.

    Returns:
        Metric summaries keyed by official metric-domain names.

    """
    prediction_img = normalized_to_image_domain(prediction)
    target_img = normalized_to_image_domain(target)
    return {
        "mae_norm": summarize_metric(mae_per_image(prediction, target)),
        "mse_norm": summarize_metric(mse_per_image(prediction, target)),
        "psnr_img": summarize_metric(
            psnr_per_image(prediction_img, target_img),
            allow_inf=True,
        ),
        "ssim_img": summarize_metric(ssim_per_image(prediction_img, target_img)),
    }


def _validate_pair(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if prediction.shape != target.shape:
        message = f"Metric tensor shapes differ: {prediction.shape} vs {target.shape}"
        raise ValueError(message)
    if prediction.device != target.device:
        message = (
            "Metric tensors must be on the same device, got "
            f"{prediction.device} and {target.device}"
        )
        raise ValueError(message)
    if prediction.ndim != NCHW_DIMENSIONS:
        message = f"Expected NCHW tensors, got {prediction.ndim} dimensions"
        raise ValueError(message)
    if prediction.shape[0] <= 0:
        message = "Metric tensors must contain at least one image"
        raise ValueError(message)
    if any(dimension <= 0 for dimension in prediction.shape[1:]):
        message = (
            f"Metric tensors must have positive C/H/W dimensions: {prediction.shape}"
        )
        raise ValueError(message)
    prediction_f32 = prediction.to(dtype=torch.float32)
    target_f32 = target.to(dtype=torch.float32)
    _validate_finite(prediction_f32, name="prediction")
    _validate_finite(target_f32, name="target")
    return prediction_f32, target_f32


def _validate_finite(
    tensor: torch.Tensor,
    *,
    name: str,
    allow_inf: bool = False,
) -> None:
    valid = torch.isfinite(tensor) if not allow_inf else ~torch.isnan(tensor)
    if not bool(valid.all().item()):
        message = f"{name} contains non-finite values"
        raise ValueError(message)


def _validate_image_domain(tensor: torch.Tensor, *, name: str) -> None:
    lower_ok = bool((tensor >= -IMAGE_DOMAIN_TOLERANCE).all().item())
    upper_ok = bool((tensor <= 1.0 + IMAGE_DOMAIN_TOLERANCE).all().item())
    if not lower_ok or not upper_ok:
        message = f"{name} must be in image-domain [0, 1] for PSNR/SSIM"
        raise ValueError(message)


def _validate_ssim_parameters(
    *,
    data_range: float,
    window_size: int,
    sigma: float,
) -> None:
    if data_range <= 0.0:
        message = f"data_range must be positive, got {data_range}"
        raise ValueError(message)
    if window_size % 2 == 0:
        message = f"window_size must be odd, got {window_size}"
        raise ValueError(message)
    if sigma <= 0.0:
        message = f"sigma must be positive, got {sigma}"
        raise ValueError(message)


def _validate_ssim_spatial_shape(tensor: torch.Tensor, *, window_size: int) -> None:
    height = tensor.shape[2]
    width = tensor.shape[3]
    if height < window_size or width < window_size:
        message = (
            "SSIM requires spatial size at least "
            f"{window_size}x{window_size}, got {height}x{width}"
        )
        raise ValueError(message)


def _ssim_map(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    kernel: torch.Tensor,
    padding: int,
    data_range: float,
) -> torch.Tensor:
    mu_x = _windowed_average(prediction, kernel=kernel, padding=padding)
    mu_y = _windowed_average(target, kernel=kernel, padding=padding)
    return _compose_ssim_map(
        means=(mu_x, mu_y),
        variances=_ssim_variance_terms(
            prediction,
            target,
            means=(mu_x, mu_y),
            kernel=kernel,
            padding=padding,
        ),
        data_range=data_range,
    )


def _ssim_variance_terms(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    means: tuple[torch.Tensor, torch.Tensor],
    kernel: torch.Tensor,
    padding: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu_x, mu_y = means
    sigma_x_sq = (
        _windowed_average(prediction.square(), kernel=kernel, padding=padding)
        - mu_x.square()
    )
    sigma_y_sq = (
        _windowed_average(target.square(), kernel=kernel, padding=padding)
        - mu_y.square()
    )
    sigma_xy = (
        _windowed_average(prediction * target, kernel=kernel, padding=padding)
        - mu_x * mu_y
    )
    return sigma_x_sq, sigma_y_sq, sigma_xy


def _compose_ssim_map(
    *,
    means: tuple[torch.Tensor, torch.Tensor],
    variances: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    data_range: float,
) -> torch.Tensor:
    mu_x, mu_y = means
    sigma_x_sq, sigma_y_sq, sigma_xy = variances
    c1 = (SSIM_K1 * data_range) ** 2
    c2 = (SSIM_K2 * data_range) ** 2
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x.square() + mu_y.square() + c1) * (sigma_x_sq + sigma_y_sq + c2)
    return numerator / denominator


def _ssim_kernel(
    *,
    channels: int,
    window_size: int,
    sigma: float,
    device: torch.device,
) -> torch.Tensor:
    radius = window_size // 2
    positions = torch.arange(
        -radius,
        radius + 1,
        dtype=torch.float32,
        device=device,
    )
    kernel_1d = torch.exp(-(positions.square()) / (2.0 * sigma**2))
    kernel_1d /= kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d.reshape(1, 1, window_size, window_size).expand(
        channels,
        1,
        window_size,
        window_size,
    )


def _windowed_average(
    tensor: torch.Tensor,
    *,
    kernel: torch.Tensor,
    padding: int,
) -> torch.Tensor:
    padded = nn_functional.pad(
        tensor,
        (padding, padding, padding, padding),
        mode="reflect",
    )
    return nn_functional.conv2d(padded, kernel, groups=tensor.shape[1])


def _finite_mean_std_or_none(values: torch.Tensor) -> tuple[float | None, float | None]:
    if values.numel() == 0:
        return None, None
    return _mean_std(values)


def _mean_std(values: torch.Tensor) -> tuple[float, float]:
    std_tensor = (
        torch.zeros((), dtype=torch.float32, device=values.device)
        if values.numel() == 1
        else values.std(unbiased=False)
    )
    return float(values.mean()), float(std_tensor)


__all__ = [
    "SSIM_DATA_RANGE",
    "SSIM_K1",
    "SSIM_K2",
    "SSIM_SIGMA",
    "SSIM_WINDOW_SIZE",
    "MetricSummary",
    "mae_per_image",
    "mse_per_image",
    "normalized_to_image_domain",
    "psnr_per_image",
    "reconstruction_metric_summaries",
    "ssim_per_image",
    "summarize_metric",
]
