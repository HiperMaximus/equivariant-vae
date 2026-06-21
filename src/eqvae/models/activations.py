# Copyright 2026 HiperMaximus
"""Activation modules for the spec 0001 translatable VAE."""

from __future__ import annotations

import torch
from torch import nn


class GatedScalarActivation(nn.Module):
    """Learned scalar gate shared by the Conv2d baseline scalar channels."""

    channels: int
    force_fp32: bool
    a: nn.Parameter
    b: nn.Parameter

    def __init__(
        self,
        channels: int,
        *,
        a_init: float = 1.0,
        b_init: float = 0.0,
        force_fp32: bool = True,
    ) -> None:
        """Initialize per-channel gate parameters."""
        super().__init__()
        self.channels = channels
        self.force_fp32 = force_fp32
        self.a = nn.Parameter(torch.full((channels,), a_init))
        self.b = nn.Parameter(torch.full((channels,), b_init))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply `x * sigmoid(a * x + b)` channelwise.

        Returns:
            Tensor with the same shape as `inputs`.

        """
        gate_dtype = torch.float32 if self.force_fp32 else inputs.dtype
        gate_inputs = inputs.to(dtype=gate_dtype)
        a = self.a.reshape(1, self.channels, 1, 1).to(
            device=inputs.device,
            dtype=gate_dtype,
        )
        b = self.b.reshape(1, self.channels, 1, 1).to(
            device=inputs.device,
            dtype=gate_dtype,
        )
        gate = torch.sigmoid((a * gate_inputs) + b).to(dtype=inputs.dtype)
        return inputs * gate


__all__ = ["GatedScalarActivation"]
