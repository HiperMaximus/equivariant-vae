# Copyright 2026 HiperMaximus
"""Fieldwise spatial resampling modules for spec 0001."""

from __future__ import annotations

from typing import Final, cast

import torch
from torch import nn
from torch.nn import functional

BINOMIAL_5X5_KERNEL_1D: Final[tuple[float, ...]] = (
    1.0 / 16.0,
    4.0 / 16.0,
    6.0 / 16.0,
    4.0 / 16.0,
    1.0 / 16.0,
)


class FixedBinomialLowpassDownsample2x(nn.Module):
    """Fixed fieldwise 5x5 binomial low-pass followed by stride-2 decimation."""

    channels: int

    def __init__(self, channels: int) -> None:
        """Store one FP32 scalar kernel shared across all channels."""
        super().__init__()
        self.channels = channels
        kernel_1d = torch.tensor(BINOMIAL_5X5_KERNEL_1D, dtype=torch.float32)
        kernel_2d = torch.outer(kernel_1d, kernel_1d).reshape(1, 1, 5, 5)
        self.register_buffer("kernel", kernel_2d, persistent=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the same scalar blur/decimation to every channel.

        Returns:
            Downsampled tensor with half the input spatial size.

        """
        kernel = cast("torch.Tensor", self.kernel)
        weight = kernel.to(device=inputs.device, dtype=inputs.dtype).expand(
            self.channels,
            1,
            5,
            5,
        )
        return functional.conv2d(
            inputs,
            weight,
            stride=2,
            padding=2,
            groups=self.channels,
        )


class FieldwiseBilinearUpsample2x(nn.Module):
    """Fieldwise bilinear 2x upsampling with fixed `align_corners=False`."""

    channels: int

    def __init__(self, channels: int) -> None:
        """Record the channel count for inventory and proof artifacts."""
        super().__init__()
        self.channels = channels

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Upsample every channel with the same bilinear spatial operator.

        Returns:
            Upsampled tensor with twice the input spatial size.

        Raises:
            ValueError: If the input channel count does not match the module.

        """
        if inputs.shape[1] != self.channels:
            message = f"Expected {self.channels} channels, got {inputs.shape[1]}"
            raise ValueError(message)
        return functional.interpolate(
            inputs,
            scale_factor=2.0,
            mode="bilinear",
            align_corners=False,
        )


__all__ = [
    "BINOMIAL_5X5_KERNEL_1D",
    "FieldwiseBilinearUpsample2x",
    "FixedBinomialLowpassDownsample2x",
]
