# Copyright 2026 HiperMaximus
"""Tests for spec 0001 reconstruction metrics."""

from __future__ import annotations

import pytest
import torch

from eqvae.metrics.reconstruction import (
    SSIM_WINDOW_SIZE,
    mae_per_image,
    mse_per_image,
    normalized_to_image_domain,
    psnr_per_image,
    reconstruction_metric_summaries,
    ssim_per_image,
    summarize_metric,
)

BATCH = 2
CHANNELS = 3
IMAGE_SIZE = 16
SMALL_IMAGE_SIZE = 8
NOISE_SCALE = 0.1
IDENTICAL_SSIM = 1.0
ZERO_ERROR = 0.0
PSNR_INF_COUNT = 2
SSIM_TOLERANCE = 1.0e-6


def test_normalized_projection_clamps_to_image_domain() -> None:
    """Image-domain projection is explicit and outside the model forward path."""
    tensor = torch.tensor([[[[-2.0, -1.0, 0.0, 1.0, 2.0]]]])

    projected = normalized_to_image_domain(tensor)
    expected = torch.tensor([[[[0.0, 0.0, 0.5, 1.0, 1.0]]]])

    assert torch.equal(projected, expected)


def test_identical_images_have_zero_error_inf_psnr_and_unit_ssim() -> None:
    """The fixed SSIM fixture for identical images is exactly one."""
    target = _normalized_fixture()
    target_img = normalized_to_image_domain(target)

    mae_values = mae_per_image(target, target)
    mse_values = mse_per_image(target, target)
    psnr_values = psnr_per_image(target_img, target_img)
    ssim_values = ssim_per_image(target_img, target_img)
    summaries = reconstruction_metric_summaries(target, target)

    assert torch.equal(mae_values, torch.zeros_like(mae_values))
    assert torch.equal(mse_values, torch.zeros_like(mse_values))
    assert torch.isinf(psnr_values).all()
    assert torch.allclose(
        ssim_values,
        torch.ones_like(ssim_values),
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    assert summaries["mae_norm"].mean == ZERO_ERROR
    assert summaries["mse_norm"].mean == ZERO_ERROR
    assert summaries["psnr_img"].mean is None
    assert summaries["psnr_img"].std is None
    assert summaries["psnr_img"].inf_count == PSNR_INF_COUNT
    ssim_mean = summaries["ssim_img"].mean
    assert ssim_mean is not None
    assert abs(ssim_mean - IDENTICAL_SSIM) <= SSIM_TOLERANCE


def test_perturbed_images_degrade_metrics() -> None:
    """A deterministic perturbation increases error and lowers SSIM."""
    target = _normalized_fixture()
    prediction = (target + NOISE_SCALE).clamp(-1.0, 1.0)
    target_img = normalized_to_image_domain(target)
    prediction_img = normalized_to_image_domain(prediction)

    mae_values = mae_per_image(prediction, target)
    mse_values = mse_per_image(prediction, target)
    psnr_values = psnr_per_image(prediction_img, target_img)
    ssim_values = ssim_per_image(prediction_img, target_img)

    assert bool((mae_values > ZERO_ERROR).all())
    assert bool((mse_values > ZERO_ERROR).all())
    assert torch.isfinite(psnr_values).all()
    assert bool((ssim_values < IDENTICAL_SSIM).all())


def test_metric_summary_is_json_safe_for_infinite_psnr() -> None:
    """Infinite PSNR values are counted without emitting JSON infinity."""
    values = torch.tensor([float("inf"), float("inf")])

    summary = summarize_metric(values, allow_inf=True)
    payload = summary.as_dict()

    assert summary.mean is None
    assert summary.std is None
    assert summary.inf_count == PSNR_INF_COUNT
    assert payload["mean"] is None
    assert payload["inf_count"] == PSNR_INF_COUNT


def test_ssim_rejects_too_small_images() -> None:
    """SSIM must not silently change the locked 11x11 window."""
    image = torch.zeros((1, CHANNELS, SMALL_IMAGE_SIZE, SMALL_IMAGE_SIZE))

    with pytest.raises(ValueError, match=f"{SSIM_WINDOW_SIZE}x{SSIM_WINDOW_SIZE}"):
        ssim_per_image(image, image)


def test_normalized_to_image_domain_clamps_into_unit_range() -> None:
    """The projection clamps any input into [0, 1] (replaces the runtime guard)."""
    out_of_range = torch.tensor([[-3.0, 3.0]])

    projected = normalized_to_image_domain(out_of_range)

    assert float(projected.min().item()) >= 0.0
    assert float(projected.max().item()) <= 1.0


def test_metrics_reject_empty_channel_or_spatial_dimensions() -> None:
    """Metric validation fails before reductions can produce NaNs."""
    empty_channel = torch.empty((1, 0, IMAGE_SIZE, IMAGE_SIZE))
    empty_height = torch.empty((1, CHANNELS, 0, IMAGE_SIZE))

    with pytest.raises(ValueError, match="positive C/H/W"):
        mse_per_image(empty_channel, empty_channel)
    with pytest.raises(ValueError, match="positive C/H/W"):
        mae_per_image(empty_height, empty_height)


def test_metrics_reject_device_mismatch() -> None:
    """Device mismatches fail with a controlled error at the metric boundary."""
    cpu_tensor = torch.zeros((1, CHANNELS, IMAGE_SIZE, IMAGE_SIZE))
    meta_tensor = torch.empty((1, CHANNELS, IMAGE_SIZE, IMAGE_SIZE), device="meta")

    with pytest.raises(ValueError, match="same device"):
        mse_per_image(cpu_tensor, meta_tensor)


def _normalized_fixture() -> torch.Tensor:
    values = torch.linspace(
        -1.0,
        1.0,
        steps=BATCH * CHANNELS * IMAGE_SIZE * IMAGE_SIZE,
        dtype=torch.float32,
    )
    return values.reshape(BATCH, CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
