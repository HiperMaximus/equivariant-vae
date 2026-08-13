# Copyright 2026 HiperMaximus
"""Complete fixed continuous-SO(2) VAE selected by Specs 0012-0014."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from eqvae.models.latent import LATENT_CHANNELS
from eqvae.models.non_equivariant_vae import VaeForwardOutput, clamp_logvar
from eqvae.models.so2_architecture_probe import (
    _PROFILE_7,  # pyright: ignore[reportPrivateUsage]
    _PROFILE_9,  # pyright: ignore[reportPrivateUsage]
    A_LAYOUT,
    B_LAYOUT,
    C_LAYOUT,
    D_LAYOUT,
    L_LAYOUT,
    R_LAYOUT,
    FixedF01FieldNorm,
    FixedF01Layout,
    FixedF01RadialGate,
    _F01ToF01Conv,  # pyright: ignore[reportPrivateUsage]
    _F01ToScalarConv,  # pyright: ignore[reportPrivateUsage]
    _FixedF01Downsample2x,  # pyright: ignore[reportPrivateUsage]
    _FixedF01Upsample2x,  # pyright: ignore[reportPrivateUsage]
    _ScalarToF01Conv,  # pyright: ignore[reportPrivateUsage]
)


class _SO2EncoderResBlock(nn.Module):
    """One fixed encoder residual block over the selected F01 layouts."""

    def __init__(
        self,
        input_layout: FixedF01Layout,
        output_layout: FixedF01Layout,
        *,
        downsample: bool,
    ) -> None:
        super().__init__()
        self.main_conv1 = _F01ToF01Conv(input_layout, output_layout)
        self.main_norm1 = FixedF01FieldNorm(output_layout)
        self.main_gate = FixedF01RadialGate(output_layout)
        self.main_downsample = (
            _FixedF01Downsample2x(output_layout.channels) if downsample else None
        )
        self.main_conv2 = _F01ToF01Conv(output_layout, output_layout)
        self.main_norm2 = FixedF01FieldNorm(output_layout)
        self.skip_downsample = (
            _FixedF01Downsample2x(input_layout.channels) if downsample else None
        )
        has_projection = downsample or input_layout != output_layout
        self.skip_conv = (
            _F01ToF01Conv(input_layout, output_layout) if has_projection else None
        )
        self.skip_norm = FixedF01FieldNorm(output_layout) if has_projection else None
        self.output_gate = FixedF01RadialGate(output_layout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the locked encoder main and skip branches.

        Returns:
            Canonically packed output fields.

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


class _SO2DecoderResBlock(nn.Module):
    """One fixed decoder residual block over the selected F01 layouts."""

    def __init__(
        self,
        input_layout: FixedF01Layout,
        output_layout: FixedF01Layout,
        *,
        upsample: bool,
    ) -> None:
        super().__init__()
        self.main_upsample = _FixedF01Upsample2x() if upsample else None
        self.main_conv1 = _F01ToF01Conv(input_layout, output_layout)
        self.main_norm1 = FixedF01FieldNorm(output_layout)
        self.main_gate = FixedF01RadialGate(output_layout)
        self.main_conv2 = _F01ToF01Conv(output_layout, output_layout)
        self.main_norm2 = FixedF01FieldNorm(output_layout)
        self.skip_upsample = _FixedF01Upsample2x() if upsample else None
        has_projection = upsample or input_layout != output_layout
        self.skip_conv = (
            _F01ToF01Conv(input_layout, output_layout) if has_projection else None
        )
        self.skip_norm = FixedF01FieldNorm(output_layout) if has_projection else None
        self.output_gate = FixedF01RadialGate(output_layout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the locked decoder main and skip branches.

        Returns:
            Canonically packed output fields.

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


class SO2VAE(nn.Module):
    """The single fixed 43-convolution continuous-SO(2) VAE."""

    def __init__(self) -> None:
        """Assemble the locked encoder, scalar posterior, and decoder."""
        super().__init__()
        self.latent_channels: int = LATENT_CHANNELS

        self.stem_conv = _ScalarToF01Conv(R_LAYOUT.n0, A_LAYOUT, _PROFILE_9)
        self.stem_norm = FixedF01FieldNorm(A_LAYOUT)
        self.stem_gate = FixedF01RadialGate(A_LAYOUT)
        self.encoder_blocks = nn.ModuleList(
            [
                _SO2EncoderResBlock(A_LAYOUT, A_LAYOUT, downsample=False),
                _SO2EncoderResBlock(A_LAYOUT, A_LAYOUT, downsample=False),
                _SO2EncoderResBlock(A_LAYOUT, B_LAYOUT, downsample=True),
                _SO2EncoderResBlock(B_LAYOUT, B_LAYOUT, downsample=False),
                _SO2EncoderResBlock(B_LAYOUT, C_LAYOUT, downsample=True),
                _SO2EncoderResBlock(C_LAYOUT, C_LAYOUT, downsample=False),
                _SO2EncoderResBlock(C_LAYOUT, D_LAYOUT, downsample=True),
                _SO2EncoderResBlock(D_LAYOUT, D_LAYOUT, downsample=False),
            ],
        )
        self.mu_head = _F01ToScalarConv(
            D_LAYOUT,
            L_LAYOUT.n0,
            zero_initialize=False,
        )
        self.logvar_head = _F01ToScalarConv(
            D_LAYOUT,
            L_LAYOUT.n0,
            zero_initialize=False,
        )

        self.latent_projection_conv = _ScalarToF01Conv(
            L_LAYOUT.n0,
            D_LAYOUT,
            _PROFILE_7,
        )
        self.latent_projection_norm = FixedF01FieldNorm(D_LAYOUT)
        self.latent_projection_gate = FixedF01RadialGate(D_LAYOUT)
        self.decoder_blocks = nn.ModuleList(
            [
                _SO2DecoderResBlock(D_LAYOUT, D_LAYOUT, upsample=False),
                _SO2DecoderResBlock(D_LAYOUT, D_LAYOUT, upsample=False),
                _SO2DecoderResBlock(D_LAYOUT, C_LAYOUT, upsample=True),
                _SO2DecoderResBlock(C_LAYOUT, C_LAYOUT, upsample=False),
                _SO2DecoderResBlock(C_LAYOUT, B_LAYOUT, upsample=True),
                _SO2DecoderResBlock(B_LAYOUT, B_LAYOUT, upsample=False),
                _SO2DecoderResBlock(B_LAYOUT, A_LAYOUT, upsample=True),
                _SO2DecoderResBlock(A_LAYOUT, A_LAYOUT, upsample=False),
            ],
        )
        self.output_head = _F01ToScalarConv(
            A_LAYOUT,
            R_LAYOUT.n0,
            zero_initialize=True,
        )

    def encode(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode RGB into scalar posterior parameters.

        Returns:
            Raw mean and log-variance tensors.

        """
        hidden = self._encode_features(inputs)
        return (
            cast("torch.Tensor", self.mu_head(hidden)),
            cast("torch.Tensor", self.logvar_head(hidden)),
        )

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode a scalar spatial latent into raw RGB.

        Returns:
            Unbounded normalized-domain RGB reconstruction.

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
        """Run the complete stochastic VAE path.

        Returns:
            Reconstruction and scalar posterior state.

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
        """Sample with the baseline's scalar Gaussian policy.

        Returns:
            Spatial latent sample and the epsilon tensor used.

        Raises:
            ValueError: If explicit epsilon has the wrong shape or device.

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


def build_so2_vae() -> SO2VAE:
    """Create the one locked continuous-SO2 VAE.

    Returns:
        Instantiated fixed model.

    """
    return SO2VAE()


__all__ = ["SO2VAE", "build_so2_vae"]
