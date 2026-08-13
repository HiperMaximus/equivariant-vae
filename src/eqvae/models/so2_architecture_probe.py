# Copyright 2026 HiperMaximus
"""Locked F0/F1 mechanics probe for Spec 0013.

This module is deliberately specialized to the selected continuous-SO(2)
experiment. It is not a configurable equivariant-layer library.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Literal, cast

import torch
from torch import nn
from torch.nn import functional

from eqvae.models.non_equivariant_vae import clamp_logvar

_NORM_EPS: Final = 1e-5
_RADIUS_EPS: Final = 1e-4
_F0_GROUPS: Final = 8
_F1_GROUPS: Final = 4


@dataclass(frozen=True, slots=True)
class FixedF01Layout:
    """One hard-coded packed F0/F1 layout used by the selected experiment."""

    name: str
    n0: int
    n1: int

    @property
    def channels(self) -> int:
        """Physical packed width."""
        return self.n0 + 2 * self.n1

    @property
    def f1_offset(self) -> int:
        """First packed F1 component offset."""
        return self.n0


R_LAYOUT: Final = FixedF01Layout("R", 3, 0)
A_LAYOUT: Final = FixedF01Layout("A", 16, 16)
B_LAYOUT: Final = FixedF01Layout("B", 24, 24)
C_LAYOUT: Final = FixedF01Layout("C", 32, 32)
D_LAYOUT: Final = FixedF01Layout("D", 48, 48)
L_LAYOUT: Final = FixedF01Layout("L", 16, 0)
LOCKED_LAYOUTS: Final = (R_LAYOUT, A_LAYOUT, B_LAYOUT, C_LAYOUT, D_LAYOUT, L_LAYOUT)

_PROFILE_7_CENTRES: Final = (1.0, 1.90395977, 2.75)
_PROFILE_7_WIDTHS: Final = (0.3, 0.3, 0.3)
_PROFILE_7_QMAX: Final = (2, 2, 2)
_PROFILE_9_CENTRES: Final = (1.0, 1.99907757, 2.87711643, 3.75)
_PROFILE_9_WIDTHS: Final = (0.3, 0.3, 0.3, 0.3)
_PROFILE_9_QMAX: Final = (2, 2, 2, 4)


@dataclass(frozen=True, slots=True)
class _FixedProfile:
    kernel_size: int
    centres: tuple[float, ...]
    widths: tuple[float, ...]
    qmax: tuple[int, ...]


_PROFILE_7: Final = _FixedProfile(
    7,
    _PROFILE_7_CENTRES,
    _PROFILE_7_WIDTHS,
    _PROFILE_7_QMAX,
)
_PROFILE_9: Final = _FixedProfile(
    9,
    _PROFILE_9_CENTRES,
    _PROFILE_9_WIDTHS,
    _PROFILE_9_QMAX,
)

_I: Final = ((1.0, 0.0), (0.0, 1.0))
_J: Final = ((0.0, -1.0), (1.0, 0.0))
_S: Final = ((1.0, 0.0), (0.0, -1.0))
_T: Final = ((0.0, 1.0), (1.0, 0.0))


def _irrep_dimension(frequency: Literal[0, 1]) -> int:
    return 1 if frequency == 0 else 2


def _rotation_samples(frequency: Literal[0, 1], angles: torch.Tensor) -> torch.Tensor:
    if frequency == 0:
        return torch.ones((*angles.shape, 1, 1), dtype=torch.float64)
    phases = frequency * angles
    cosines = torch.cos(phases)
    sines = torch.sin(phases)
    return torch.stack(
        (
            torch.stack((cosines, -sines), dim=-1),
            torch.stack((sines, cosines), dim=-1),
        ),
        dim=-2,
    )


def _pair_generators(
    input_frequency: Literal[0, 1],
    output_frequency: Literal[0, 1],
) -> tuple[tuple[int, torch.Tensor], ...]:
    if input_frequency == 0 and output_frequency == 0:
        return ((0, torch.ones((1, 1), dtype=torch.float64)),)
    if input_frequency == 0:
        return (
            (1, torch.tensor([[1.0], [0.0]], dtype=torch.float64)),
            (1, torch.tensor([[0.0], [1.0]], dtype=torch.float64)),
        )
    if output_frequency == 0:
        return (
            (1, torch.tensor([[1.0, 0.0]], dtype=torch.float64)),
            (1, torch.tensor([[0.0, 1.0]], dtype=torch.float64)),
        )
    return tuple(
        (order, torch.tensor(generator, dtype=torch.float64))
        for order, generator in ((0, _I), (0, _J), (2, _S), (2, _T))
    )


def _build_pair_bank(  # noqa: PLR0914
    input_frequency: Literal[0, 1],
    output_frequency: Literal[0, 1],
    profile: _FixedProfile,
) -> torch.Tensor:
    """Construct one selected pair bank once, before any training forward.

    Returns:
        Contiguous FP32 bank with one flattened basis element per row.

    """
    centre = (profile.kernel_size - 1) / 2.0
    coordinates = torch.arange(profile.kernel_size, dtype=torch.float64)
    rows, columns = torch.meshgrid(coordinates, coordinates, indexing="ij")
    x_coordinates = columns - centre
    y_coordinates = centre - rows
    radii = torch.hypot(x_coordinates, y_coordinates)
    angles = torch.atan2(y_coordinates, x_coordinates)
    output_rotation = _rotation_samples(output_frequency, angles)
    input_inverse_rotation = _rotation_samples(input_frequency, -angles)
    centre_mask = radii.abs() < torch.finfo(torch.float64).eps
    generators = _pair_generators(input_frequency, output_frequency)
    samples: list[torch.Tensor] = []

    for angular_order, generator in generators:
        if angular_order == 0:
            radial = centre_mask.to(dtype=torch.float64)
            angular = torch.einsum(
                "...oa,ab,...bi->oi...",
                output_rotation,
                generator,
                input_inverse_rotation,
            )
            samples.append(angular * radial.unsqueeze(0).unsqueeze(0))

    for shell_centre, shell_width, shell_qmax in zip(
        profile.centres,
        profile.widths,
        profile.qmax,
        strict=True,
    ):
        radial = torch.exp(-((radii - shell_centre) ** 2) / (2.0 * shell_width**2))
        for angular_order, generator in generators:
            if angular_order > min(shell_qmax, 2):
                continue
            shell_radial = (
                radial.masked_fill(centre_mask, 0.0) if angular_order else radial
            )
            angular = torch.einsum(
                "...oa,ab,...bi->oi...",
                output_rotation,
                generator,
                input_inverse_rotation,
            )
            samples.append(angular * shell_radial.unsqueeze(0).unsqueeze(0))

    sampled = torch.stack(samples)
    columns_flat = sampled.reshape(sampled.shape[0], -1).transpose(0, 1)
    orthonormal = cast(
        "torch.Tensor",
        torch.linalg.qr(  # pyright: ignore[reportUnknownMemberType]
            columns_flat,
            mode="reduced",
        ).Q,
    )
    pivots = orthonormal.abs().argmax(dim=0)
    pivot_values = orthonormal[pivots, torch.arange(orthonormal.shape[1])]
    signs = torch.where(pivot_values < 0.0, -1.0, 1.0)
    orthonormal *= signs.unsqueeze(0)
    scaled = math.sqrt(_irrep_dimension(output_frequency)) * orthonormal
    return scaled.transpose(0, 1).to(dtype=torch.float32).contiguous()


def _new_coefficients(
    *,
    output_copies: int,
    input_copies: int,
    basis_dimension: int,
    present_input_frequencies: int,
    zero: bool = False,
) -> nn.Parameter:
    coefficients = torch.empty(
        output_copies * input_copies,
        basis_dimension,
        dtype=torch.float32,
    )
    if zero:
        nn.init.zeros_(coefficients)
    else:
        deviation = 1.0 / math.sqrt(
            present_input_frequencies * input_copies * basis_dimension,
        )
        nn.init.normal_(coefficients, std=deviation)
    return nn.Parameter(coefficients)


def _expand_pair(  # noqa: PLR0913
    coefficients: torch.Tensor,
    basis: torch.Tensor,
    *,
    output_copies: int,
    input_copies: int,
    output_dimension: int,
    input_dimension: int,
    kernel_size: int,
) -> torch.Tensor:
    flat = torch.mm(coefficients, basis)
    return (
        flat
        .view(
            output_copies,
            input_copies,
            output_dimension,
            input_dimension,
            kernel_size,
            kernel_size,
        )
        .permute(0, 2, 1, 3, 4, 5)
        .reshape(
            output_copies * output_dimension,
            input_copies * input_dimension,
            kernel_size,
            kernel_size,
        )
    )


class _ScalarToF01Conv(nn.Module):
    """Fixed scalar-to-F01 learned convolution with statically unrolled banks."""

    basis00: torch.Tensor
    basis10: torch.Tensor

    def __init__(
        self,
        input_copies: int,
        output_layout: FixedF01Layout,
        profile: _FixedProfile,
    ) -> None:
        super().__init__()
        self.input_copies = input_copies
        self.output_layout = output_layout
        self.kernel_size = profile.kernel_size
        basis00 = _build_pair_bank(0, 0, profile)
        basis10 = _build_pair_bank(0, 1, profile)
        self.register_buffer("basis00", basis00, persistent=True)
        self.register_buffer("basis10", basis10, persistent=True)
        self.coeff00 = _new_coefficients(
            output_copies=output_layout.n0,
            input_copies=input_copies,
            basis_dimension=basis00.shape[0],
            present_input_frequencies=1,
        )
        self.coeff10 = _new_coefficients(
            output_copies=output_layout.n1,
            input_copies=input_copies,
            basis_dimension=basis10.shape[0],
            present_input_frequencies=1,
        )

    def expanded_kernel(self) -> torch.Tensor:
        """Expand the current coefficients into the one dense kernel.

        Returns:
            Dense canonical packed kernel.

        """
        top = _expand_pair(
            self.coeff00,
            self.basis00,
            output_copies=self.output_layout.n0,
            input_copies=self.input_copies,
            output_dimension=1,
            input_dimension=1,
            kernel_size=self.kernel_size,
        )
        bottom = _expand_pair(
            self.coeff10,
            self.basis10,
            output_copies=self.output_layout.n1,
            input_copies=self.input_copies,
            output_dimension=2,
            input_dimension=1,
            kernel_size=self.kernel_size,
        )
        return torch.cat((top, bottom), dim=0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Expand once and execute exactly one learned dense convolution.

        Returns:
            Scalar-to-F01 convolution output.

        """
        return functional.conv2d(
            inputs,
            self.expanded_kernel(),
            bias=None,
            padding=self.kernel_size // 2,
        )


class _F01ToF01Conv(nn.Module):
    """Fixed F01-to-F01 learned convolution with four unrolled contractions."""

    basis00: torch.Tensor
    basis10: torch.Tensor
    basis01: torch.Tensor
    basis11: torch.Tensor

    def __init__(
        self,
        input_layout: FixedF01Layout,
        output_layout: FixedF01Layout,
    ) -> None:
        super().__init__()
        self.input_layout = input_layout
        self.output_layout = output_layout
        self.kernel_size = _PROFILE_7.kernel_size
        basis00 = _build_pair_bank(0, 0, _PROFILE_7)
        basis10 = _build_pair_bank(0, 1, _PROFILE_7)
        basis01 = _build_pair_bank(1, 0, _PROFILE_7)
        basis11 = _build_pair_bank(1, 1, _PROFILE_7)
        self.register_buffer("basis00", basis00, persistent=True)
        self.register_buffer("basis10", basis10, persistent=True)
        self.register_buffer("basis01", basis01, persistent=True)
        self.register_buffer("basis11", basis11, persistent=True)
        self.coeff00 = _new_coefficients(
            output_copies=output_layout.n0,
            input_copies=input_layout.n0,
            basis_dimension=basis00.shape[0],
            present_input_frequencies=2,
        )
        self.coeff10 = _new_coefficients(
            output_copies=output_layout.n1,
            input_copies=input_layout.n0,
            basis_dimension=basis10.shape[0],
            present_input_frequencies=2,
        )
        self.coeff01 = _new_coefficients(
            output_copies=output_layout.n0,
            input_copies=input_layout.n1,
            basis_dimension=basis01.shape[0],
            present_input_frequencies=2,
        )
        self.coeff11 = _new_coefficients(
            output_copies=output_layout.n1,
            input_copies=input_layout.n1,
            basis_dimension=basis11.shape[0],
            present_input_frequencies=2,
        )

    def expanded_kernel(self) -> torch.Tensor:
        """Expand four pair blocks and assemble the canonical packed kernel.

        Returns:
            Dense canonical packed kernel.

        """
        kernel00 = _expand_pair(
            self.coeff00,
            self.basis00,
            output_copies=self.output_layout.n0,
            input_copies=self.input_layout.n0,
            output_dimension=1,
            input_dimension=1,
            kernel_size=self.kernel_size,
        )
        kernel10 = _expand_pair(
            self.coeff10,
            self.basis10,
            output_copies=self.output_layout.n1,
            input_copies=self.input_layout.n0,
            output_dimension=2,
            input_dimension=1,
            kernel_size=self.kernel_size,
        )
        kernel01 = _expand_pair(
            self.coeff01,
            self.basis01,
            output_copies=self.output_layout.n0,
            input_copies=self.input_layout.n1,
            output_dimension=1,
            input_dimension=2,
            kernel_size=self.kernel_size,
        )
        kernel11 = _expand_pair(
            self.coeff11,
            self.basis11,
            output_copies=self.output_layout.n1,
            input_copies=self.input_layout.n1,
            output_dimension=2,
            input_dimension=2,
            kernel_size=self.kernel_size,
        )
        top = torch.cat((kernel00, kernel01), dim=1)
        bottom = torch.cat((kernel10, kernel11), dim=1)
        return torch.cat((top, bottom), dim=0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Expand once and execute exactly one learned dense convolution.

        Returns:
            F01-to-F01 convolution output.

        """
        return functional.conv2d(
            inputs,
            self.expanded_kernel(),
            bias=None,
            padding=self.kernel_size // 2,
        )


class _F01ToScalarConv(nn.Module):
    """Fixed F01-to-scalar convolution for posterior and RGB heads."""

    basis00: torch.Tensor
    basis01: torch.Tensor
    bias: nn.Parameter

    def __init__(
        self,
        input_layout: FixedF01Layout,
        output_copies: int,
        *,
        zero_initialize: bool,
    ) -> None:
        super().__init__()
        self.input_layout = input_layout
        self.output_copies = output_copies
        self.kernel_size = _PROFILE_7.kernel_size
        basis00 = _build_pair_bank(0, 0, _PROFILE_7)
        basis01 = _build_pair_bank(1, 0, _PROFILE_7)
        self.register_buffer("basis00", basis00, persistent=True)
        self.register_buffer("basis01", basis01, persistent=True)
        self.coeff00 = _new_coefficients(
            output_copies=output_copies,
            input_copies=input_layout.n0,
            basis_dimension=basis00.shape[0],
            present_input_frequencies=2,
            zero=zero_initialize,
        )
        self.coeff01 = _new_coefficients(
            output_copies=output_copies,
            input_copies=input_layout.n1,
            basis_dimension=basis01.shape[0],
            present_input_frequencies=2,
            zero=zero_initialize,
        )
        self.bias = nn.Parameter(torch.empty(output_copies, dtype=torch.float32))
        if zero_initialize:
            nn.init.zeros_(self.bias)
        else:
            bound = 1.0 / math.sqrt(input_layout.channels * self.kernel_size**2)
            nn.init.uniform_(self.bias, -bound, bound)

    def expanded_kernel(self) -> torch.Tensor:
        """Expand two input-frequency blocks into one scalar-output kernel.

        Returns:
            Dense canonical packed kernel.

        """
        kernel00 = _expand_pair(
            self.coeff00,
            self.basis00,
            output_copies=self.output_copies,
            input_copies=self.input_layout.n0,
            output_dimension=1,
            input_dimension=1,
            kernel_size=self.kernel_size,
        )
        kernel01 = _expand_pair(
            self.coeff01,
            self.basis01,
            output_copies=self.output_copies,
            input_copies=self.input_layout.n1,
            output_dimension=1,
            input_dimension=2,
            kernel_size=self.kernel_size,
        )
        return torch.cat((kernel00, kernel01), dim=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Expand once and execute exactly one learned dense convolution.

        Returns:
            F01-to-scalar convolution output.

        """
        return functional.conv2d(
            inputs,
            self.expanded_kernel(),
            bias=self.bias,
            padding=self.kernel_size // 2,
        )


class FixedF01FieldNorm(nn.Module):
    """Locked eight-group F0 norm plus four-group invariant F1 RMS norm."""

    def __init__(self, layout: FixedF01Layout) -> None:
        """Build affine parameters for one locked hidden layout."""
        super().__init__()
        self.layout = layout
        self.f0_gamma = nn.Parameter(torch.ones(layout.n0, dtype=torch.float32))
        self.f0_beta = nn.Parameter(torch.zeros(layout.n0, dtype=torch.float32))
        self.f1_gamma = nn.Parameter(torch.ones(layout.n1, dtype=torch.float32))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # noqa: PLR0914
        """Normalize scalar and vector fields using only invariant statistics.

        Returns:
            Canonically packed normalized fields in the input dtype.

        """
        values = inputs.float()
        batch, _channels, height, width = values.shape
        scalar = values[:, : self.layout.n0]
        scalar_grouped = scalar.view(
            batch,
            _F0_GROUPS,
            self.layout.n0 // _F0_GROUPS,
            height,
            width,
        )
        scalar_mean = scalar_grouped.mean(dim=(2, 3, 4), keepdim=True)
        scalar_variance = (
            (scalar_grouped - scalar_mean)
            .square()
            .mean(
                dim=(2, 3, 4),
                keepdim=True,
            )
        )
        scalar_normalized = (
            (scalar_grouped - scalar_mean) * torch.rsqrt(scalar_variance + _NORM_EPS)
        ).view(batch, self.layout.n0, height, width)
        scalar_normalized = scalar_normalized * self.f0_gamma.view(
            1,
            -1,
            1,
            1,
        ) + self.f0_beta.view(1, -1, 1, 1)

        vector = values[:, self.layout.f1_offset :].view(
            batch,
            self.layout.n1,
            2,
            height,
            width,
        )
        vector_grouped = vector.view(
            batch,
            _F1_GROUPS,
            self.layout.n1 // _F1_GROUPS,
            2,
            height,
            width,
        )
        vector_rms = torch.sqrt(
            vector_grouped.square().mean(dim=(2, 3, 4, 5), keepdim=True) + _NORM_EPS,
        )
        vector_normalized = (vector_grouped / vector_rms).view(
            batch,
            self.layout.n1,
            2,
            height,
            width,
        )
        scaled_vector = vector_normalized * self.f1_gamma.view(1, -1, 1, 1, 1)
        packed = torch.cat(
            (
                scalar_normalized,
                scaled_vector.view(batch, 2 * self.layout.n1, height, width),
            ),
            dim=1,
        )
        return packed.to(dtype=inputs.dtype)


class FixedF01RadialGate(nn.Module):
    """Locked scalar gates and shared-component radial F1 gates."""

    def __init__(self, layout: FixedF01Layout) -> None:
        """Build one scalar and one radial gate per field copy."""
        super().__init__()
        self.layout = layout
        self.f0_a = nn.Parameter(torch.ones(layout.n0, dtype=torch.float32))
        self.f0_b = nn.Parameter(torch.zeros(layout.n0, dtype=torch.float32))
        self.f1_a = nn.Parameter(torch.ones(layout.n1, dtype=torch.float32))
        self.f1_b = nn.Parameter(torch.zeros(layout.n1, dtype=torch.float32))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Gate F1 pairs by their shared invariant radius without vector bias.

        Returns:
            Canonically packed gated fields in the input dtype.

        """
        values = inputs.float()
        scalar_gate = torch.sigmoid(
            self.f0_a.view(1, -1, 1, 1) * values[:, : self.layout.n0]
            + self.f0_b.view(1, -1, 1, 1),
        ).to(dtype=inputs.dtype)
        scalar = inputs[:, : self.layout.n0] * scalar_gate

        batch, _channels, height, width = inputs.shape
        vector_values = values[:, self.layout.f1_offset :].view(
            batch,
            self.layout.n1,
            2,
            height,
            width,
        )
        radius = torch.sqrt(vector_values.square().sum(dim=2) + _RADIUS_EPS)
        vector_gate = torch.sigmoid(
            self.f1_a.view(1, -1, 1, 1) * radius + self.f1_b.view(1, -1, 1, 1),
        ).to(dtype=inputs.dtype)
        vector = inputs[:, self.layout.f1_offset :].view_as(vector_values)
        gated_vector = vector * vector_gate.unsqueeze(2)
        return torch.cat(
            (scalar, gated_vector.view(batch, 2 * self.layout.n1, height, width)),
            dim=1,
        )


class _FixedF01Downsample2x(nn.Module):
    """Probe-only pre-expanded FP32 5x5 binomial grouped downsampler."""

    weight: torch.Tensor

    def __init__(self, channels: int) -> None:
        super().__init__()
        kernel_1d = (
            torch.tensor(
                (1.0, 4.0, 6.0, 4.0, 1.0),
                dtype=torch.float32,
            )
            / 16.0
        )
        kernel = torch.outer(kernel_1d, kernel_1d).view(1, 1, 5, 5)
        weight = kernel.expand(channels, 1, 5, 5).contiguous()
        self.channels = channels
        self.register_buffer("weight", weight, persistent=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the fixed fieldwise blur and stride-two decimation.

        Returns:
            Half-resolution packed fields.

        """
        return functional.conv2d(
            inputs,
            self.weight,
            stride=2,
            padding=2,
            groups=self.channels,
        )


class _FixedF01Upsample2x(nn.Module):
    """Probe-only fieldwise bilinear upsampler with the locked grid rule."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # noqa: PLR6301
        """Apply uniform bilinear x2 interpolation to every packed component.

        Returns:
            Double-resolution packed fields.

        """
        return functional.interpolate(
            inputs,
            scale_factor=2.0,
            mode="bilinear",
            align_corners=False,
        )


class SO2IdentityResidualBlockA(nn.Module):
    """One fixed A-to-A identity residual probe block."""

    def __init__(self) -> None:
        """Build the two learned main convolutions and identity skip."""
        super().__init__()
        self.main_conv1 = _F01ToF01Conv(A_LAYOUT, A_LAYOUT)
        self.main_norm1 = FixedF01FieldNorm(A_LAYOUT)
        self.main_gate = FixedF01RadialGate(A_LAYOUT)
        self.main_conv2 = _F01ToF01Conv(A_LAYOUT, A_LAYOUT)
        self.main_norm2 = FixedF01FieldNorm(A_LAYOUT)
        self.output_gate = FixedF01RadialGate(A_LAYOUT)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the fixed identity-skip residual order.

        Returns:
            A-layout residual output.

        """
        main = cast("torch.Tensor", self.main_conv1(inputs))
        main = cast("torch.Tensor", self.main_norm1(main))
        main = cast("torch.Tensor", self.main_gate(main))
        main = cast("torch.Tensor", self.main_conv2(main))
        main = cast("torch.Tensor", self.main_norm2(main))
        return cast("torch.Tensor", self.output_gate(main + inputs))


class SO2LargestDDConv(nn.Module):
    """Isolated fixed D-to-D convolution for the dual-T4 mechanics probe."""

    def __init__(self) -> None:
        """Build the locked largest expansion without exposing its internals."""
        super().__init__()
        self.conv = _F01ToF01Conv(D_LAYOUT, D_LAYOUT)

    def expanded_kernel(self) -> torch.Tensor:
        """Expand the current D-to-D coefficients for isolated timing.

        Returns:
            Dense D-to-D kernel.

        """
        return self.conv.expanded_kernel()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Execute the isolated locked D-to-D convolution.

        Returns:
            D-layout output.

        """
        return cast("torch.Tensor", self.conv(inputs))


class SO2EncoderTransitionAB(nn.Module):
    """One fixed A-to-B encoder transition probe."""

    def __init__(self) -> None:
        """Build the locked A-to-B main and projection branches."""
        super().__init__()
        self.main_conv1 = _F01ToF01Conv(A_LAYOUT, B_LAYOUT)
        self.main_norm1 = FixedF01FieldNorm(B_LAYOUT)
        self.main_gate = FixedF01RadialGate(B_LAYOUT)
        self.main_downsample = _FixedF01Downsample2x(B_LAYOUT.channels)
        self.main_conv2 = _F01ToF01Conv(B_LAYOUT, B_LAYOUT)
        self.main_norm2 = FixedF01FieldNorm(B_LAYOUT)
        self.skip_downsample = _FixedF01Downsample2x(A_LAYOUT.channels)
        self.skip_conv = _F01ToF01Conv(A_LAYOUT, B_LAYOUT)
        self.skip_norm = FixedF01FieldNorm(B_LAYOUT)
        self.output_gate = FixedF01RadialGate(B_LAYOUT)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the locked encoder main and projection-skip branch order.

        Returns:
            Half-resolution B-layout output.

        """
        main = cast("torch.Tensor", self.main_conv1(inputs))
        main = cast("torch.Tensor", self.main_norm1(main))
        main = cast("torch.Tensor", self.main_gate(main))
        main = cast("torch.Tensor", self.main_downsample(main))
        main = cast("torch.Tensor", self.main_conv2(main))
        main = cast("torch.Tensor", self.main_norm2(main))
        skip = cast("torch.Tensor", self.skip_downsample(inputs))
        skip = cast("torch.Tensor", self.skip_conv(skip))
        skip = cast("torch.Tensor", self.skip_norm(skip))
        return cast("torch.Tensor", self.output_gate(main + skip))


class SO2DecoderTransitionBA(nn.Module):
    """One fixed B-to-A decoder transition probe."""

    def __init__(self) -> None:
        """Build the locked B-to-A main and projection branches."""
        super().__init__()
        self.main_upsample = _FixedF01Upsample2x()
        self.main_conv1 = _F01ToF01Conv(B_LAYOUT, A_LAYOUT)
        self.main_norm1 = FixedF01FieldNorm(A_LAYOUT)
        self.main_gate = FixedF01RadialGate(A_LAYOUT)
        self.main_conv2 = _F01ToF01Conv(A_LAYOUT, A_LAYOUT)
        self.main_norm2 = FixedF01FieldNorm(A_LAYOUT)
        self.skip_upsample = _FixedF01Upsample2x()
        self.skip_conv = _F01ToF01Conv(B_LAYOUT, A_LAYOUT)
        self.skip_norm = FixedF01FieldNorm(A_LAYOUT)
        self.output_gate = FixedF01RadialGate(A_LAYOUT)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the locked decoder main and projection-skip branch order.

        Returns:
            Double-resolution A-layout output.

        """
        main = cast("torch.Tensor", self.main_upsample(inputs))
        main = cast("torch.Tensor", self.main_conv1(main))
        main = cast("torch.Tensor", self.main_norm1(main))
        main = cast("torch.Tensor", self.main_gate(main))
        main = cast("torch.Tensor", self.main_conv2(main))
        main = cast("torch.Tensor", self.main_norm2(main))
        skip = cast("torch.Tensor", self.skip_upsample(inputs))
        skip = cast("torch.Tensor", self.skip_conv(skip))
        skip = cast("torch.Tensor", self.skip_norm(skip))
        return cast("torch.Tensor", self.output_gate(main + skip))


class SO2RGBLift(nn.Module):
    """Fixed 9x9 scalar RGB lift followed by A-field norm and gate."""

    def __init__(self) -> None:
        """Build the selected 9x9 lift and A-field postprocessing."""
        super().__init__()
        self.conv = _ScalarToF01Conv(R_LAYOUT.n0, A_LAYOUT, _PROFILE_9)
        self.norm = FixedF01FieldNorm(A_LAYOUT)
        self.gate = FixedF01RadialGate(A_LAYOUT)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Lift scalar RGB into the canonical A layout.

        Returns:
            Normalized and gated A fields.

        """
        hidden = cast("torch.Tensor", self.conv(inputs))
        hidden = cast("torch.Tensor", self.norm(hidden))
        return cast("torch.Tensor", self.gate(hidden))


class SO2ScalarLatentHeads(nn.Module):
    """Fixed scalar posterior heads and baseline-compatible sampling policy."""

    def __init__(self) -> None:
        """Build independent scalar mean and log-variance heads."""
        super().__init__()
        self.mu = _F01ToScalarConv(D_LAYOUT, L_LAYOUT.n0, zero_initialize=False)
        self.logvar = _F01ToScalarConv(D_LAYOUT, L_LAYOUT.n0, zero_initialize=False)

    def forward(
        self,
        inputs: torch.Tensor,
        eps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Produce scalar posterior statistics and a controlled latent sample.

        Returns:
            Mean, raw log-variance, clamped log-variance, and sampled latent.

        """
        mu = cast("torch.Tensor", self.mu(inputs))
        logvar = cast("torch.Tensor", self.logvar(inputs))
        logvar_clamped = clamp_logvar(logvar)
        latent = mu + torch.exp(0.5 * logvar_clamped) * eps
        return mu, logvar, logvar_clamped, latent


class SO2LatentProjection(nn.Module):
    """Fixed scalar-latent to D-field decoder projection."""

    def __init__(self) -> None:
        """Build the selected scalar-to-D projection path."""
        super().__init__()
        self.conv = _ScalarToF01Conv(L_LAYOUT.n0, D_LAYOUT, _PROFILE_7)
        self.norm = FixedF01FieldNorm(D_LAYOUT)
        self.gate = FixedF01RadialGate(D_LAYOUT)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Project the scalar latent into the canonical D layout.

        Returns:
            Normalized and gated D fields.

        """
        hidden = cast("torch.Tensor", self.conv(inputs))
        hidden = cast("torch.Tensor", self.norm(hidden))
        return cast("torch.Tensor", self.gate(hidden))


class SO2RGBHead(nn.Module):
    """Fixed zero-initialized A-to-RGB scalar output head."""

    def __init__(self) -> None:
        """Build the selected zero-initialized scalar RGB projection."""
        super().__init__()
        self.conv = _F01ToScalarConv(A_LAYOUT, R_LAYOUT.n0, zero_initialize=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Project A fields to raw scalar RGB without a final bounding function.

        Returns:
            Raw normalized-domain RGB reconstruction.

        """
        return cast("torch.Tensor", self.conv(inputs))


_CONV_SIGNATURE_OCCURRENCES: Final = {
    "R->A": 1,
    "A->A": 7,
    "A->B": 2,
    "B->B": 6,
    "B->C": 2,
    "C->C": 6,
    "C->D": 2,
    "D->D": 7,
    "D->L": 2,
    "L->D": 1,
    "D->C": 2,
    "C->B": 2,
    "B->A": 2,
    "A->R": 1,
}


def _coefficient_count_for_signature(name: str) -> int:
    layouts = {layout.name: layout for layout in LOCKED_LAYOUTS}
    input_name, output_name = name.split("->")
    input_layout = layouts[input_name]
    output_layout = layouts[output_name]
    profile = _PROFILE_9 if name == "R->A" else _PROFILE_7
    count = 0
    if input_layout.n0 and output_layout.n0:
        count += (
            input_layout.n0
            * output_layout.n0
            * _build_pair_bank(0, 0, profile).shape[0]
        )
    if input_layout.n0 and output_layout.n1:
        count += (
            input_layout.n0
            * output_layout.n1
            * _build_pair_bank(0, 1, profile).shape[0]
        )
    if input_layout.n1 and output_layout.n0:
        count += (
            input_layout.n1
            * output_layout.n0
            * _build_pair_bank(1, 0, profile).shape[0]
        )
    if input_layout.n1 and output_layout.n1:
        count += (
            input_layout.n1
            * output_layout.n1
            * _build_pair_bank(1, 1, profile).shape[0]
        )
    return count


def locked_full_architecture_coefficient_count() -> int:
    """Return the analytic coefficient total without assembling the full VAE.

    Returns:
        Coefficient count for the locked 43 positions.

    """
    return sum(
        occurrences * _coefficient_count_for_signature(signature)
        for signature, occurrences in _CONV_SIGNATURE_OCCURRENCES.items()
    )


__all__ = [
    "A_LAYOUT",
    "B_LAYOUT",
    "C_LAYOUT",
    "D_LAYOUT",
    "LOCKED_LAYOUTS",
    "L_LAYOUT",
    "R_LAYOUT",
    "FixedF01FieldNorm",
    "FixedF01Layout",
    "FixedF01RadialGate",
    "SO2DecoderTransitionBA",
    "SO2EncoderTransitionAB",
    "SO2IdentityResidualBlockA",
    "SO2LargestDDConv",
    "SO2LatentProjection",
    "SO2RGBHead",
    "SO2RGBLift",
    "SO2ScalarLatentHeads",
    "locked_full_architecture_coefficient_count",
]
