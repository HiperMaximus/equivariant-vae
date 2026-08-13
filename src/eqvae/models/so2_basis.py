# Copyright 2026 HiperMaximus
"""Fixed F0/F1/F2 sampled SO(2) kernel bases for Spec 0012's oracle."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, cast

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

LOCKED_FREQUENCIES: Final = (0, 1, 2)
PAIR_FREQUENCIES: Final = tuple(
    (input_frequency, output_frequency)
    for input_frequency in LOCKED_FREQUENCIES
    for output_frequency in LOCKED_FREQUENCIES
)

_I: Final[FloatArray] = np.eye(2, dtype=np.float64)
_J: Final[FloatArray] = np.array([[0.0, -1.0], [1.0, 0.0]])
_S: Final[FloatArray] = np.array([[1.0, 0.0], [0.0, -1.0]])
_T: Final[FloatArray] = np.array([[0.0, 1.0], [1.0, 0.0]])


@dataclass(frozen=True)
class BasisColumn:
    """Identity of one radial/intertwiner basis column."""

    shell: int
    angular_order: int
    generator: int


@dataclass(frozen=True)
class SampledPairBasis:
    """One-copy sampled basis for a locked F0/F1/F2 frequency pair."""

    input_frequency: int
    output_frequency: int
    kernel_size: int
    values: FloatArray
    columns: tuple[BasisColumn, ...]

    def flat_columns(self) -> FloatArray:
        """Flatten basis elements into matrix columns for numerical audits.

        Returns:
            Matrix with one sampled basis element per column.

        """
        flattened = self.values.reshape(len(self.columns), -1).T
        return cast("FloatArray", flattened)


def irrep_dimension(frequency: int) -> int:
    """Return the real dimension of one locked SO(2) irrep.

    Returns:
        One for F0 and two for F1/F2.

    """
    _require_frequency(frequency)
    return 1 if frequency == 0 else 2


def pair_angular_orders(
    input_frequency: int,
    output_frequency: int,
) -> tuple[int, ...]:
    """Return angular order once per legal real generator for a locked pair.

    Returns:
        Generator-aligned tuple of spatial angular orders.

    """
    return tuple(
        angular_order
        for angular_order, _generator in _pair_generators(
            input_frequency,
            output_frequency,
        )
    )


def sample_pair_basis(  # noqa: PLR0913
    *,
    input_frequency: int,
    output_frequency: int,
    kernel_size: int,
    centres: tuple[float, ...],
    widths: tuple[float, ...],
    qmax: tuple[int, ...],
    allowed_orders: frozenset[int] = frozenset({0, 1, 2, 3, 4}),
) -> SampledPairBasis:
    """Sample the exact Spec 0012 basis, including the dedicated q=0 impulse.

    Returns:
        Sampled one-copy pair basis and column identities.

    Raises:
        ValueError: If a value is outside the locked oracle surface.

    """
    _require_frequency(input_frequency)
    _require_frequency(output_frequency)
    if kernel_size not in {7, 9}:
        message = "Spec 0012 samples only 7x7 and 9x9 supports"
        raise ValueError(message)
    if not (len(centres) == len(widths) == len(qmax)):
        message = "centres, widths, and qmax must have equal lengths"
        raise ValueError(message)

    generators = _pair_generators(input_frequency, output_frequency)
    x_coordinates, y_coordinates = _kernel_coordinates(kernel_size)
    radii = np.hypot(x_coordinates, y_coordinates)
    angles = np.arctan2(y_coordinates, x_coordinates)
    output_rotation = _rotation_samples(output_frequency, angles)
    input_inverse_rotation = _rotation_samples(input_frequency, -angles)

    sampled: list[FloatArray] = []
    identities: list[BasisColumn] = []
    centre_mask = np.isclose(radii, 0.0, atol=0.0)
    impulse = centre_mask.astype(np.float64)
    for generator_index, (angular_order, generator) in enumerate(generators):
        if angular_order == 0 and angular_order in allowed_orders:
            sampled.append(
                _sample_generator(
                    radial=impulse,
                    output_rotation=output_rotation,
                    generator=generator,
                    input_inverse_rotation=input_inverse_rotation,
                ),
            )
            identities.append(BasisColumn(0, angular_order, generator_index))

    for shell_index, (centre, width, shell_qmax) in enumerate(
        zip(centres, widths, qmax, strict=True),
        start=1,
    ):
        radial = np.exp(-((radii - centre) ** 2) / (2.0 * width**2))
        for generator_index, (angular_order, generator) in enumerate(generators):
            if angular_order > shell_qmax or angular_order not in allowed_orders:
                continue
            shell_radial = radial.copy()
            if angular_order > 0:
                shell_radial[centre_mask] = 0.0
            sampled.append(
                _sample_generator(
                    radial=shell_radial,
                    output_rotation=output_rotation,
                    generator=generator,
                    input_inverse_rotation=input_inverse_rotation,
                ),
            )
            identities.append(
                BasisColumn(shell_index, angular_order, generator_index),
            )

    if not sampled:
        message = "locked pair/profile combination produced an empty basis"
        raise ValueError(message)
    values = np.stack(sampled, axis=0)
    return SampledPairBasis(
        input_frequency=input_frequency,
        output_frequency=output_frequency,
        kernel_size=kernel_size,
        values=values,
        columns=tuple(identities),
    )


def orthonormalize_columns(matrix: FloatArray) -> FloatArray:
    """Build a reduced QR basis with Spec 0012's deterministic sign rule.

    Returns:
        Euclidean-orthonormal columns with fixed signs.

    """
    orthonormal = np.linalg.qr(matrix, mode="reduced")[0]
    column_count = cast("int", orthonormal.shape[1])
    for column_index in range(column_count):
        column = orthonormal[:, column_index]
        pivot = int(np.argmax(np.abs(column)))
        if column[pivot] < 0.0:
            orthonormal[:, column_index] *= -1.0
    return orthonormal


def stored_pair_basis(sampled: SampledPairBasis) -> FloatArray:
    """Return QR coordinates scaled to escnn's sampled output-irrep norm.

    Returns:
        Basis bank shaped as basis, output, input, height, width.

    """
    orthonormal = orthonormalize_columns(sampled.flat_columns())
    scale = math.sqrt(irrep_dimension(sampled.output_frequency))
    scaled = scale * orthonormal
    output_dimension = irrep_dimension(sampled.output_frequency)
    input_dimension = irrep_dimension(sampled.input_frequency)
    return scaled.T.reshape(
        len(sampled.columns),
        output_dimension,
        input_dimension,
        sampled.kernel_size,
        sampled.kernel_size,
    )


def representation_matrix(frequency: int, angle: float) -> FloatArray:
    """Return the fixed real [cos,sin] SO(2) representation matrix.

    Returns:
        One-by-one or two-by-two real representation matrix.

    """
    _require_frequency(frequency)
    if frequency == 0:
        return np.ones((1, 1), dtype=np.float64)
    phase = frequency * angle
    cosine = math.cos(phase)
    sine = math.sin(phase)
    return np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)


def generalized_he_standard_deviation(
    *,
    present_input_frequency_count: int,
    input_copies: int,
    basis_dimension: int,
) -> float:
    """Return Spec 0012's locked coefficient initialization scale.

    Returns:
        Standard deviation for one input-frequency/output-copy block.

    Raises:
        ValueError: If any count is not positive.

    """
    counts = (
        present_input_frequency_count,
        input_copies,
        basis_dimension,
    )
    if any(count <= 0 for count in counts):
        message = "generalized-He counts must all be positive"
        raise ValueError(message)
    coefficient_count = input_copies * basis_dimension
    return 1.0 / math.sqrt(present_input_frequency_count * coefficient_count)


def _pair_generators(
    input_frequency: int,
    output_frequency: int,
) -> tuple[tuple[int, FloatArray], ...]:
    _require_frequency(input_frequency)
    _require_frequency(output_frequency)
    if input_frequency == 0 and output_frequency == 0:
        return ((0, np.ones((1, 1), dtype=np.float64)),)
    if input_frequency == 0:
        return (
            (output_frequency, np.array([[1.0], [0.0]])),
            (output_frequency, np.array([[0.0], [1.0]])),
        )
    if output_frequency == 0:
        return (
            (input_frequency, np.array([[1.0, 0.0]])),
            (input_frequency, np.array([[0.0, 1.0]])),
        )
    return (
        (abs(output_frequency - input_frequency), _I),
        (abs(output_frequency - input_frequency), _J),
        (output_frequency + input_frequency, _S),
        (output_frequency + input_frequency, _T),
    )


def _kernel_coordinates(kernel_size: int) -> tuple[FloatArray, FloatArray]:
    centre = (kernel_size - 1) / 2.0
    rows, columns = np.meshgrid(
        np.arange(kernel_size, dtype=np.float64),
        np.arange(kernel_size, dtype=np.float64),
        indexing="ij",
    )
    return columns - centre, centre - rows


def _rotation_samples(frequency: int, angles: FloatArray) -> FloatArray:
    if frequency == 0:
        return np.ones((*angles.shape, 1, 1), dtype=np.float64)
    phases = frequency * angles
    cosines = np.cos(phases)
    sines = np.sin(phases)
    rotations = np.empty((*angles.shape, 2, 2), dtype=np.float64)
    rotations[..., 0, 0] = cosines
    rotations[..., 0, 1] = -sines
    rotations[..., 1, 0] = sines
    rotations[..., 1, 1] = cosines
    return rotations


def _sample_generator(
    *,
    radial: FloatArray,
    output_rotation: FloatArray,
    generator: FloatArray,
    input_inverse_rotation: FloatArray,
) -> FloatArray:
    angular = cast(
        "FloatArray",
        np.einsum(
            "...oa,ab,...bi->oi...",
            output_rotation,
            generator,
            input_inverse_rotation,
        ),
    )
    return angular * radial[np.newaxis, np.newaxis, ...]


def _require_frequency(frequency: int) -> None:
    if frequency not in LOCKED_FREQUENCIES:
        message = f"frequency {frequency} is outside locked F0/F1/F2"
        raise ValueError(message)
