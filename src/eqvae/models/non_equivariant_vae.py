# Copyright 2026 HiperMaximus
"""Spec 0001 translatable non-equivariant Conv2d VAE topology."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from eqvae.models.activations import GatedScalarActivation
from eqvae.models.resampling import (
    FieldwiseBilinearUpsample2x,
    FixedBinomialLowpassDownsample2x,
)

DEFAULT_GROUPNORM_GROUPS = 8
LATENT_CHANNELS = 16
INPUT_CHANNELS = 3
OUTPUT_CHANNELS = 3
STEM_CHANNELS = 32
BOTTLENECK_CHANNELS = 96
DEFAULT_LOGVAR_CLAMP_MIN = -8.0
DEFAULT_LOGVAR_CLAMP_MAX = 4.0


@dataclass(frozen=True)
class VaeForwardOutput:
    """Forward output for training and benchmark code."""

    reconstruction: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor
    logvar_clamped: torch.Tensor
    z: torch.Tensor
    eps: torch.Tensor
    logvar_clamp_count: torch.Tensor


class EncoderResBlock(nn.Module):
    """ResNet-like encoder block with branch-local optional downsampling."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        downsample: bool,
        norm_groups: int,
    ) -> None:
        """Build one locked encoder residual block."""
        super().__init__()
        self.downsample = downsample
        self.has_projection = downsample or in_channels != out_channels
        self.main_conv1 = _conv5(in_channels, out_channels, bias=False)
        self.main_norm1 = _norm(norm_groups, out_channels)
        self.main_gate = GatedScalarActivation(out_channels)
        self.main_downsample = (
            FixedBinomialLowpassDownsample2x(out_channels) if downsample else None
        )
        self.main_conv2 = _conv5(out_channels, out_channels, bias=False)
        self.main_norm2 = _norm(norm_groups, out_channels)
        self.skip_downsample = (
            FixedBinomialLowpassDownsample2x(in_channels) if downsample else None
        )
        self.skip_conv = (
            _conv5(in_channels, out_channels, bias=False)
            if self.has_projection
            else None
        )
        self.skip_norm = (
            _norm(norm_groups, out_channels) if self.has_projection else None
        )
        self.output_gate = GatedScalarActivation(out_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the residual block.

        Returns:
            Block output tensor.

        """
        main = cast("torch.Tensor", self.main_conv1(inputs))
        main = cast("torch.Tensor", self.main_norm1(main))
        main = cast("torch.Tensor", self.main_gate(main))
        if self.main_downsample is not None:
            main = cast("torch.Tensor", self.main_downsample(main))
        main = cast("torch.Tensor", self.main_conv2(main))
        main = cast("torch.Tensor", self.main_norm2(main))

        skip = inputs
        if self.skip_downsample is not None:
            skip = cast("torch.Tensor", self.skip_downsample(skip))
        if self.skip_conv is not None:
            skip = cast("torch.Tensor", self.skip_conv(skip))
        if self.skip_norm is not None:
            skip = cast("torch.Tensor", self.skip_norm(skip))
        return cast("torch.Tensor", self.output_gate(main + skip))


class DecoderUpResBlock(nn.Module):
    """ResNet-like decoder block with branch-local optional upsampling."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        upsample: bool,
        norm_groups: int,
    ) -> None:
        """Build one locked decoder residual block."""
        super().__init__()
        self.upsample = upsample
        self.has_projection = upsample or in_channels != out_channels
        self.main_upsample = (
            FieldwiseBilinearUpsample2x(in_channels) if upsample else None
        )
        self.main_conv1 = _conv5(in_channels, out_channels, bias=False)
        self.main_norm1 = _norm(norm_groups, out_channels)
        self.main_gate = GatedScalarActivation(out_channels)
        self.main_conv2 = _conv5(out_channels, out_channels, bias=False)
        self.main_norm2 = _norm(norm_groups, out_channels)
        self.skip_upsample = (
            FieldwiseBilinearUpsample2x(in_channels) if upsample else None
        )
        self.skip_conv = (
            _conv5(in_channels, out_channels, bias=False)
            if self.has_projection
            else None
        )
        self.skip_norm = (
            _norm(norm_groups, out_channels) if self.has_projection else None
        )
        self.output_gate = GatedScalarActivation(out_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the residual block.

        Returns:
            Block output tensor.

        """
        main = inputs
        if self.main_upsample is not None:
            main = cast("torch.Tensor", self.main_upsample(main))
        main = cast("torch.Tensor", self.main_conv1(main))
        main = cast("torch.Tensor", self.main_norm1(main))
        main = cast("torch.Tensor", self.main_gate(main))
        main = cast("torch.Tensor", self.main_conv2(main))
        main = cast("torch.Tensor", self.main_norm2(main))

        skip = inputs
        if self.skip_upsample is not None:
            skip = cast("torch.Tensor", self.skip_upsample(skip))
        if self.skip_conv is not None:
            skip = cast("torch.Tensor", self.skip_conv(skip))
        if self.skip_norm is not None:
            skip = cast("torch.Tensor", self.skip_norm(skip))
        return cast("torch.Tensor", self.output_gate(main + skip))


class NonEquivariantVAE(nn.Module):
    """Locked spec 0001 Conv2d VAE baseline topology."""

    def __init__(self, *, norm_groups: int = DEFAULT_GROUPNORM_GROUPS) -> None:
        """Initialize the full encoder/decoder topology."""
        super().__init__()
        self.norm_groups = norm_groups
        self.stem_conv = nn.Conv2d(
            INPUT_CHANNELS,
            STEM_CHANNELS,
            kernel_size=7,
            stride=1,
            padding=3,
            bias=False,
        )
        self.stem_norm = _norm(norm_groups, STEM_CHANNELS)
        self.stem_gate = GatedScalarActivation(STEM_CHANNELS)

        self.encoder_blocks = nn.ModuleList(
            [
                EncoderResBlock(
                    in_channels=32,
                    out_channels=32,
                    downsample=False,
                    norm_groups=norm_groups,
                ),
                EncoderResBlock(
                    in_channels=32,
                    out_channels=32,
                    downsample=False,
                    norm_groups=norm_groups,
                ),
                EncoderResBlock(
                    in_channels=32,
                    out_channels=48,
                    downsample=True,
                    norm_groups=norm_groups,
                ),
                EncoderResBlock(
                    in_channels=48,
                    out_channels=48,
                    downsample=False,
                    norm_groups=norm_groups,
                ),
                EncoderResBlock(
                    in_channels=48,
                    out_channels=64,
                    downsample=True,
                    norm_groups=norm_groups,
                ),
                EncoderResBlock(
                    in_channels=64,
                    out_channels=64,
                    downsample=False,
                    norm_groups=norm_groups,
                ),
                EncoderResBlock(
                    in_channels=64,
                    out_channels=96,
                    downsample=True,
                    norm_groups=norm_groups,
                ),
                EncoderResBlock(
                    in_channels=96,
                    out_channels=96,
                    downsample=False,
                    norm_groups=norm_groups,
                ),
            ],
        )
        self.mu_head = _conv5(BOTTLENECK_CHANNELS, LATENT_CHANNELS, bias=True)
        self.logvar_head = _conv5(BOTTLENECK_CHANNELS, LATENT_CHANNELS, bias=True)

        self.latent_projection_conv = _conv5(
            LATENT_CHANNELS,
            BOTTLENECK_CHANNELS,
            bias=False,
        )
        self.latent_projection_norm = _norm(norm_groups, BOTTLENECK_CHANNELS)
        self.latent_projection_gate = GatedScalarActivation(BOTTLENECK_CHANNELS)
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderUpResBlock(
                    in_channels=96,
                    out_channels=96,
                    upsample=False,
                    norm_groups=norm_groups,
                ),
                DecoderUpResBlock(
                    in_channels=96,
                    out_channels=96,
                    upsample=False,
                    norm_groups=norm_groups,
                ),
                DecoderUpResBlock(
                    in_channels=96,
                    out_channels=64,
                    upsample=True,
                    norm_groups=norm_groups,
                ),
                DecoderUpResBlock(
                    in_channels=64,
                    out_channels=64,
                    upsample=False,
                    norm_groups=norm_groups,
                ),
                DecoderUpResBlock(
                    in_channels=64,
                    out_channels=48,
                    upsample=True,
                    norm_groups=norm_groups,
                ),
                DecoderUpResBlock(
                    in_channels=48,
                    out_channels=48,
                    upsample=False,
                    norm_groups=norm_groups,
                ),
                DecoderUpResBlock(
                    in_channels=48,
                    out_channels=32,
                    upsample=True,
                    norm_groups=norm_groups,
                ),
                DecoderUpResBlock(
                    in_channels=32,
                    out_channels=32,
                    upsample=False,
                    norm_groups=norm_groups,
                ),
            ],
        )
        self.output_head = _conv5(STEM_CHANNELS, OUTPUT_CHANNELS, bias=True)
        _zero_initialize(self.output_head)

    def encode(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a clean or corrupted image to posterior parameters.

        Returns:
            Tuple of `mu` and `logvar` tensors.

        """
        hidden = self._encode_features(inputs)
        return (
            cast("torch.Tensor", self.mu_head(hidden)),
            cast("torch.Tensor", self.logvar_head(hidden)),
        )

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode a spatial latent map to raw normalized RGB values.

        Returns:
            Raw normalized RGB reconstruction tensor.

        """
        hidden = cast("torch.Tensor", self.latent_projection_conv(latent))
        hidden = cast("torch.Tensor", self.latent_projection_norm(hidden))
        hidden = cast("torch.Tensor", self.latent_projection_gate(hidden))
        for block in self.decoder_blocks:
            hidden = cast("torch.Tensor", block(hidden))
        return cast("torch.Tensor", self.output_head(hidden))

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        eps: torch.Tensor | None = None,
    ) -> VaeForwardOutput:
        """Run a stochastic VAE forward pass.

        Returns:
            Reconstruction and posterior statistics.

        """
        mu, logvar = self.encode(inputs)
        logvar_clamped = clamp_logvar(logvar)
        latent, used_eps = self.reparameterize(
            mu=mu,
            logvar=logvar_clamped,
            eps=eps,
        )
        return VaeForwardOutput(
            reconstruction=self.decode(latent),
            mu=mu,
            logvar=logvar,
            logvar_clamped=logvar_clamped,
            z=latent,
            eps=used_eps,
            logvar_clamp_count=torch.count_nonzero(logvar != logvar_clamped),
        )

    @staticmethod
    def reparameterize(
        *,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        eps: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample `z = mu + exp(0.5 * logvar) * eps`.

        Returns:
            Spatial latent sample and the epsilon tensor used.

        Raises:
            ValueError: If explicit `eps` has the wrong shape or device.

        """
        used_eps = eps if eps is not None else torch.randn_like(mu)
        if used_eps.shape != mu.shape:
            message = f"eps shape {used_eps.shape} does not match mu shape {mu.shape}"
            raise ValueError(message)
        if used_eps.device != mu.device:
            message = (
                f"eps device {used_eps.device} does not match mu device {mu.device}"
            )
            raise ValueError(message)
        return mu + torch.exp(0.5 * logvar) * used_eps, used_eps

    def _encode_features(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = cast("torch.Tensor", self.stem_conv(inputs))
        hidden = cast("torch.Tensor", self.stem_norm(hidden))
        hidden = cast("torch.Tensor", self.stem_gate(hidden))
        for block in self.encoder_blocks:
            hidden = cast("torch.Tensor", block(hidden))
        return hidden


def build_non_equivariant_vae(
    *,
    norm_groups: int = DEFAULT_GROUPNORM_GROUPS,
) -> NonEquivariantVAE:
    """Create the locked spec 0001 non-equivariant VAE instance.

    Returns:
        Instantiated non-equivariant VAE.

    """
    return NonEquivariantVAE(norm_groups=norm_groups)


def clamp_logvar(
    logvar: torch.Tensor,
    *,
    minimum: float = DEFAULT_LOGVAR_CLAMP_MIN,
    maximum: float = DEFAULT_LOGVAR_CLAMP_MAX,
) -> torch.Tensor:
    """Clamp posterior log-variance for sampling and KL arithmetic.

    Returns:
        Clamped tensor with the same shape as `logvar`.

    """
    return logvar.to(dtype=torch.float32).clamp(minimum, maximum)


def _conv5(in_channels: int, out_channels: int, *, bias: bool) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=5,
        stride=1,
        padding=2,
        bias=bias,
    )


def _norm(norm_groups: int, channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=norm_groups, num_channels=channels, affine=True)


def _zero_initialize(module: nn.Conv2d) -> None:
    nn.init.zeros_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


__all__ = [
    "BOTTLENECK_CHANNELS",
    "DEFAULT_GROUPNORM_GROUPS",
    "DEFAULT_LOGVAR_CLAMP_MAX",
    "DEFAULT_LOGVAR_CLAMP_MIN",
    "INPUT_CHANNELS",
    "LATENT_CHANNELS",
    "OUTPUT_CHANNELS",
    "STEM_CHANNELS",
    "NonEquivariantVAE",
    "VaeForwardOutput",
    "build_non_equivariant_vae",
    "clamp_logvar",
]
