# Copyright 2026 HiperMaximus
"""Focused correctness tests for Spec 0013's fixed F0/F1 mechanics probe."""

from __future__ import annotations

import copy
import importlib
import math
import sys
import types
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, cast

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from torch import nn
from torch.nn import functional

from eqvae.models.non_equivariant_vae import clamp_logvar
from eqvae.models.so2_architecture_probe import (
    _PROFILE_7,  # noqa: PLC2701
    _PROFILE_9,  # noqa: PLC2701
    A_LAYOUT,
    B_LAYOUT,
    C_LAYOUT,
    D_LAYOUT,
    L_LAYOUT,
    LOCKED_LAYOUTS,
    R_LAYOUT,
    FixedF01FieldNorm,
    FixedF01Layout,
    FixedF01RadialGate,
    SO2DecoderTransitionBA,
    SO2EncoderTransitionAB,
    SO2IdentityResidualBlockA,
    SO2LatentProjection,
    SO2RGBHead,
    SO2RGBLift,
    SO2ScalarLatentHeads,
    _build_pair_bank,  # noqa: PLC2701
    _expand_pair,  # noqa: PLC2701
    _F01ToF01Conv,  # noqa: PLC2701
    _F01ToScalarConv,  # noqa: PLC2701
    _FixedF01Downsample2x,  # noqa: PLC2701
    _FixedF01Upsample2x,  # noqa: PLC2701
    _FixedProfile,
    _ScalarToF01Conv,  # noqa: PLC2701
    locked_full_architecture_coefficient_count,
)

# pyright: reportPrivateUsage=false

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

FloatArray = NDArray[np.float64]
AttributeValue = int | float | str
_PAIR_FREQUENCIES = ((0, 0), (0, 1), (1, 0), (1, 1))
_PAIR_DIMENSIONS = {
    7: {(0, 0): 4, (0, 1): 6, (1, 0): 6, (1, 1): 14},
    9: {(0, 0): 5, (0, 1): 8, (1, 0): 8, (1, 1): 18},
}
_KERNEL_7 = 7
_KERNEL_9 = 9
_EXPECTED_COEFFICIENTS = 1_172_304
_EXPECTED_PARAMETERS = 1_180_035
_KERNEL_RELATIVE_LIMIT = 5e-5
_OUTPUT_RELATIVE_LIMIT = 1e-4
_GRADIENT_RELATIVE_LIMIT = 1e-6
_CARDINAL_DEGREES = 90
_CARDINAL_EQUIVARIANCE_LIMIT = 5e-4
_ESCNN_EQUIVARIANCE_FACTOR = 1.10


class _Profile(Protocol):
    @property
    def kernel_size(self) -> int: ...

    @property
    def centres(self) -> tuple[float, ...]: ...

    @property
    def widths(self) -> tuple[float, ...]: ...

    @property
    def qmax(self) -> tuple[int, ...]: ...


class _EscnnBasis(Protocol):
    def to(self, *, dtype: torch.dtype) -> _EscnnBasis: ...

    def sample(self, points: torch.Tensor) -> torch.Tensor: ...

    def __iter__(self) -> Iterator[dict[str, AttributeValue]]: ...


class _SO2Group(Protocol):
    def irrep(self, frequency: int) -> object: ...

    def element(self, value: float, parameterization: str) -> object: ...


class _GroupApi(Protocol):
    def so2_group(self, maximum_frequency: int) -> _SO2Group: ...


class _KernelsApi(Protocol):
    def kernels_SO2_act_R2(  # noqa: N802
        self,
        input_representation: object,
        output_representation: object,
        radii: list[float],
        sigma: list[float],
        *,
        maximum_frequency: int,
    ) -> _EscnnBasis: ...


class _GroupSpace(Protocol):
    fibergroup: _SO2Group


class _GSpacesApi(Protocol):
    def rot2dOnR2(  # noqa: N802
        self,
        *,
        N: int,  # noqa: N803
        maximum_frequency: int,
    ) -> _GroupSpace: ...


class _FieldType(Protocol):
    def transform_fibers(
        self,
        values: torch.Tensor,
        element: object,
    ) -> torch.Tensor: ...

    def transform(
        self,
        values: torch.Tensor,
        element: object,
        *,
        order: int,
    ) -> torch.Tensor: ...


class _NNApi(Protocol):
    def FieldType(  # noqa: N802
        self,
        group_space: _GroupSpace,
        representations: list[object],
    ) -> _FieldType: ...


class _Escnn(Protocol):
    group: _GroupApi
    gspaces: _GSpacesApi
    kernels: _KernelsApi
    nn: _NNApi


def _load_local_escnn() -> _Escnn:
    class NoCacheMemory:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def cache[**P, R](  # noqa: PLR6301
            self,
            function: Callable[P, R] | None = None,
            **_kwargs: object,
        ) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
            if function is None:
                return lambda wrapped: wrapped
            return function

    joblib = types.ModuleType("joblib")
    joblib.Memory = NoCacheMemory  # type: ignore[attr-defined]
    sys.modules["joblib"] = joblib
    module_names = (
        "lie_learn",
        "lie_learn.representations",
        "lie_learn.representations.SO3",
        "lie_learn.representations.SO3.wigner_d",
    )
    for module_name in module_names:
        sys.modules[module_name] = types.ModuleType(module_name)

    def reject_so3(*_args: object, **_kwargs: object) -> None:
        message = "Spec 0013 SO(2) probe entered an SO(3) path"
        raise RuntimeError(message)

    sys.modules[module_names[-1]].wigner_D_matrix = reject_so3  # type: ignore[attr-defined]
    escnn_root = Path(__file__).resolve().parents[1] / "reference/escnn"
    sys.path.insert(0, str(escnn_root))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        escnn = importlib.import_module("escnn")
    return cast("_Escnn", escnn)


def _irrep_dimension(frequency: int) -> int:
    return 1 if frequency == 0 else 2


def _pair_orders(input_frequency: int, output_frequency: int) -> tuple[int, ...]:
    if input_frequency == output_frequency == 0:
        return (0,)
    if input_frequency == 0 or output_frequency == 0:
        return (1, 1)
    return (0, 0, 2, 2)


def _escnn_pair_matrix(
    escnn: _Escnn,
    input_frequency: int,
    output_frequency: int,
    *,
    profile: _Profile,
) -> torch.Tensor:
    group = escnn.group.so2_group(4)
    kernel_size = profile.kernel_size
    centre = (kernel_size - 1) / 2.0
    points = torch.tensor(
        [
            (column - centre, centre - row)
            for row in range(kernel_size)
            for column in range(kernel_size)
        ],
        dtype=torch.float64,
    )
    centre_index = kernel_size * kernel_size // 2
    samples: list[torch.Tensor] = []
    shells = (
        (0.0, 0.005, 0),
        *zip(profile.centres, profile.widths, profile.qmax, strict=True),
    )
    for radius, width, cutoff in shells:
        effective_cutoff = min(cutoff, 2)
        if not any(
            order <= effective_cutoff
            for order in _pair_orders(input_frequency, output_frequency)
        ):
            continue
        basis = copy.deepcopy(
            escnn.kernels.kernels_SO2_act_R2(
                group.irrep(input_frequency),
                group.irrep(output_frequency),
                [radius],
                [width],
                maximum_frequency=effective_cutoff,
            ),
        ).to(dtype=torch.float64)
        sampled = basis.sample(points)
        if math.isclose(radius, 0.0, abs_tol=1e-7):
            centre_value = sampled[centre_index].clone()
            sampled = torch.zeros_like(sampled)
            sampled[centre_index] = centre_value
        samples.append(sampled)
    concatenated = torch.cat(samples, dim=1)
    return (
        concatenated
        .permute(2, 3, 0, 1)
        .reshape(
            _irrep_dimension(output_frequency)
            * _irrep_dimension(input_frequency)
            * kernel_size
            * kernel_size,
            concatenated.shape[1],
        )
        .contiguous()
    )


def _reference_projection(
    runtime_bank: torch.Tensor,
    reference_matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    runtime_matrix = runtime_bank.double().transpose(0, 1)
    coordinates = cast(
        "torch.Tensor",
        torch.linalg.lstsq(  # pyright: ignore[reportUnknownMemberType]
            reference_matrix,
            runtime_matrix,
        ).solution,
    )
    projected = reference_matrix @ coordinates
    return projected.transpose(0, 1), coordinates


def _relative_rms(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        (
            torch.sqrt((left - right).double().square().mean())
            / torch.sqrt(right.double().square().mean()).clamp_min(1e-12)
        ).detach(),
    )


def _cropped_relative_rms(
    left: torch.Tensor,
    right: torch.Tensor,
    crop: int,
) -> float:
    return _relative_rms(
        left[..., crop:-crop, crop:-crop],
        right[..., crop:-crop, crop:-crop],
    )


def _direct_expand_loop(
    coefficients: torch.Tensor,
    basis: torch.Tensor,
    *,
    shape: tuple[int, int, int, int, int],
) -> torch.Tensor:
    output_copies, input_copies, output_dimension, input_dimension, kernel_size = shape
    output = torch.empty(
        output_copies * output_dimension,
        input_copies * input_dimension,
        kernel_size,
        kernel_size,
        dtype=coefficients.dtype,
    )
    bank = basis.view(
        basis.shape[0],
        output_dimension,
        input_dimension,
        kernel_size,
        kernel_size,
    )
    for output_copy in range(output_copies):
        for input_copy in range(input_copies):
            row = output_copy * input_copies + input_copy
            block = (coefficients[row].view(-1, 1, 1, 1, 1) * bank).sum(dim=0)
            output[
                output_copy * output_dimension : (output_copy + 1) * output_dimension,
                input_copy * input_dimension : (input_copy + 1) * input_dimension,
            ] = block
    return output


def _projected_pair_bank(
    escnn: _Escnn,
    input_frequency: Literal[0, 1],
    output_frequency: Literal[0, 1],
    profile: _FixedProfile,
) -> torch.Tensor:
    runtime = _build_pair_bank(input_frequency, output_frequency, profile)
    reference = _escnn_pair_matrix(
        escnn,
        input_frequency,
        output_frequency,
        profile=profile,
    )
    projected, _coordinates = _reference_projection(runtime, reference)
    return projected.to(dtype=torch.float32)


def _assert_pair_equivariance_against_escnn(  # noqa: PLR0913
    escnn: _Escnn,
    runtime_kernel: torch.Tensor,
    escnn_kernel: torch.Tensor,
    *,
    input_frequency: int,
    output_frequency: int,
    kernel_size: int,
) -> None:
    input_layout = FixedF01Layout(
        "input",
        int(input_frequency == 0),
        int(input_frequency == 1),
    )
    output_layout = FixedF01Layout(
        "output",
        int(output_frequency == 0),
        int(output_frequency == 1),
    )
    smooth = _smooth_layout_input(input_layout).double()
    crop = kernel_size // 2 + 4
    for degrees in (15, 30, 45, 60, 90):
        transformed_input = _escnn_transform_layout(
            escnn,
            input_layout,
            smooth,
            degrees,
        )
        runtime_left = functional.conv2d(
            transformed_input,
            runtime_kernel,
            padding=kernel_size // 2,
        )
        runtime_right = _escnn_transform_layout(
            escnn,
            output_layout,
            functional.conv2d(smooth, runtime_kernel, padding=kernel_size // 2),
            degrees,
        )
        escnn_left = functional.conv2d(
            transformed_input,
            escnn_kernel,
            padding=kernel_size // 2,
        )
        escnn_right = _escnn_transform_layout(
            escnn,
            output_layout,
            functional.conv2d(smooth, escnn_kernel, padding=kernel_size // 2),
            degrees,
        )
        runtime_error = _cropped_relative_rms(runtime_left, runtime_right, crop)
        escnn_error = _cropped_relative_rms(escnn_left, escnn_right, crop)
        assert runtime_error <= max(
            _CARDINAL_EQUIVARIANCE_LIMIT,
            _ESCNN_EQUIVARIANCE_FACTOR * escnn_error,
        )


def test_locked_layouts_banks_coefficients_and_full_counts() -> None:
    """Pin the chosen physical layout and counts so full-VAE work cannot drift."""
    assert [
        (layout.name, layout.n0, layout.n1, layout.channels, layout.f1_offset)
        for layout in LOCKED_LAYOUTS
    ] == [
        ("R", 3, 0, 3, 3),
        ("A", 16, 16, 48, 16),
        ("B", 24, 24, 72, 24),
        ("C", 32, 32, 96, 32),
        ("D", 48, 48, 144, 48),
        ("L", 16, 0, 16, 16),
    ]
    lift = _ScalarToF01Conv(R_LAYOUT.n0, A_LAYOUT, _PROFILE_9)
    hidden = _F01ToF01Conv(A_LAYOUT, B_LAYOUT)
    head = _F01ToScalarConv(D_LAYOUT, L_LAYOUT.n0, zero_initialize=False)
    assert lift.basis00.shape == (5, 81)
    assert lift.basis10.shape == (8, 162)
    assert lift.coeff00.shape == (48, 5)
    assert lift.coeff10.shape == (48, 8)
    assert hidden.packed_bases.shape == (4, 14, 196)
    expected_banks = (
        _build_pair_bank(0, 0, _PROFILE_7),
        _build_pair_bank(0, 1, _PROFILE_7),
        _build_pair_bank(1, 0, _PROFILE_7),
        _build_pair_bank(1, 1, _PROFILE_7),
    )
    valid_shapes = ((4, 49), (6, 98), (6, 98), (14, 196))
    for index, (basis, shape) in enumerate(
        zip(expected_banks, valid_shapes, strict=True),
    ):
        rows, columns = shape
        assert torch.equal(hidden.packed_bases[index, :rows, :columns], basis)
        assert not bool(hidden.packed_bases[index, rows:].count_nonzero())
        assert not bool(hidden.packed_bases[index, :rows, columns:].count_nonzero())
    assert [
        parameter.shape
        for parameter in (
            hidden.coeff00,
            hidden.coeff10,
            hidden.coeff01,
            hidden.coeff11,
        )
    ] == [
        torch.Size((384, 4)),
        torch.Size((384, 6)),
        torch.Size((384, 6)),
        torch.Size((384, 14)),
    ]
    assert head.coeff00.shape == torch.Size((768, 4))
    assert head.coeff01.shape == torch.Size((768, 6))
    assert all(
        buffer.dtype == torch.float32 and buffer.is_contiguous()
        for buffer in hidden.buffers()
    )
    assert "packed_bases" in hidden.state_dict()
    assert locked_full_architecture_coefficient_count() == _EXPECTED_COEFFICIENTS
    norm_count = 10 * sum(
        2 * layout.n0 + layout.n1 for layout in (A_LAYOUT, B_LAYOUT, C_LAYOUT, D_LAYOUT)
    )
    gate_count = (
        9 * 2 * (A_LAYOUT.n0 + A_LAYOUT.n1)
        + 8 * 2 * (B_LAYOUT.n0 + B_LAYOUT.n1)
        + 8 * 2 * (C_LAYOUT.n0 + C_LAYOUT.n1)
        + 9 * 2 * (D_LAYOUT.n0 + D_LAYOUT.n1)
    )
    assert (norm_count, gate_count, 2 * L_LAYOUT.n0 + R_LAYOUT.n0) == (3_600, 4_096, 35)
    assert _EXPECTED_COEFFICIENTS + norm_count + gate_count + 35 == _EXPECTED_PARAMETERS


def test_full_count_is_derived_from_instantiated_signature_parameters() -> None:
    """Cross-check the analytic total without assembling the forbidden full VAE."""
    layouts = {layout.name: layout for layout in LOCKED_LAYOUTS}
    occurrences = {
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
    measured = 0
    for signature, count in occurrences.items():
        input_name, output_name = signature.split("->")
        input_layout = layouts[input_name]
        output_layout = layouts[output_name]
        if input_layout.n1 == 0:
            module: nn.Module = _ScalarToF01Conv(
                input_layout.n0,
                output_layout,
                _PROFILE_9 if signature == "R->A" else _PROFILE_7,
            )
        elif output_layout.n1 == 0:
            module = _F01ToScalarConv(
                input_layout,
                output_layout.n0,
                zero_initialize=False,
            )
        else:
            module = _F01ToF01Conv(input_layout, output_layout)
        measured += count * sum(
            parameter.numel()
            for name, parameter in module.named_parameters()
            if name.rsplit(".", 1)[-1].startswith("coeff")
        )
    assert measured == _EXPECTED_COEFFICIENTS


def test_generalized_he_scales_and_complete_bias_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pin initializer inputs and prevent hidden/vector biases from appearing."""
    observed_standard_deviations: list[float] = []
    original_normal = nn.init.normal_

    def record_normal(
        tensor: torch.Tensor,
        mean: float = 0.0,
        std: float = 1.0,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        observed_standard_deviations.append(std)
        return original_normal(tensor, mean=mean, std=std, generator=generator)

    monkeypatch.setattr(nn.init, "normal_", record_normal)
    lift = _ScalarToF01Conv(R_LAYOUT.n0, A_LAYOUT, _PROFILE_9)
    hidden = _F01ToF01Conv(A_LAYOUT, B_LAYOUT)
    head = _F01ToScalarConv(D_LAYOUT, L_LAYOUT.n0, zero_initialize=False)
    expected = [
        1.0 / math.sqrt(3 * 5),
        1.0 / math.sqrt(3 * 8),
        1.0 / math.sqrt(2 * 16 * 4),
        1.0 / math.sqrt(2 * 16 * 6),
        1.0 / math.sqrt(2 * 16 * 6),
        1.0 / math.sqrt(2 * 16 * 14),
        1.0 / math.sqrt(2 * 48 * 4),
        1.0 / math.sqrt(2 * 48 * 6),
    ]
    assert observed_standard_deviations == pytest.approx(expected)
    assert not any(name.endswith("bias") for name, _ in lift.named_parameters())
    assert not any(name.endswith("bias") for name, _ in hidden.named_parameters())
    bound = 1.0 / math.sqrt(D_LAYOUT.channels * _KERNEL_7**2)
    assert bool((head.bias.abs() <= bound).all())

    probes: dict[str, nn.Module] = {
        "identity": SO2IdentityResidualBlockA(),
        "encoder": SO2EncoderTransitionAB(),
        "decoder": SO2DecoderTransitionBA(),
        "lift": SO2RGBLift(),
        "latent_projection": SO2LatentProjection(),
        "latent_heads": SO2ScalarLatentHeads(),
        "rgb_head": SO2RGBHead(),
    }
    observed_biases = {
        f"{probe_name}.{parameter_name}"
        for probe_name, probe in probes.items()
        for parameter_name, _ in probe.named_parameters()
        if parameter_name.endswith("bias")
    }
    assert observed_biases == {
        "latent_heads.logvar.bias",
        "latent_heads.mu.bias",
        "rgb_head.conv.bias",
    }


@pytest.mark.parametrize(
    ("input_frequency", "output_frequency"),
    _PAIR_FREQUENCIES,
)
@pytest.mark.parametrize("kernel_size", [_KERNEL_7, _KERNEL_9])
def test_selected_runtime_banks_match_escnn_spans_and_outputs(
    input_frequency: int,
    output_frequency: int,
    kernel_size: int,
) -> None:
    """Require every selected pair bank to remain inside the pinned escnn span."""
    escnn = _load_local_escnn()
    profile = _PROFILE_7 if kernel_size == _KERNEL_7 else _PROFILE_9
    runtime = _build_pair_bank(
        cast("Literal[0, 1]", input_frequency),
        cast("Literal[0, 1]", output_frequency),
        profile,
    )
    reference = _escnn_pair_matrix(
        escnn,
        input_frequency,
        output_frequency,
        profile=profile,
    )
    projected, _coordinates = _reference_projection(runtime, reference)
    assert (
        runtime.shape[0]
        == _PAIR_DIMENSIONS[kernel_size][input_frequency, output_frequency]
    )
    assert _relative_rms(runtime, projected) <= _KERNEL_RELATIVE_LIMIT
    generator = torch.Generator().manual_seed(
        13013 + kernel_size + input_frequency + output_frequency,
    )
    coefficients = torch.randn(runtime.shape[0], generator=generator)
    kernel = coefficients @ runtime
    projected_kernel = coefficients.double() @ projected
    inputs = torch.randn(
        2,
        _irrep_dimension(input_frequency),
        17,
        17,
        generator=generator,
    )
    shape = (
        _irrep_dimension(output_frequency),
        _irrep_dimension(input_frequency),
        kernel_size,
        kernel_size,
    )
    ours = functional.conv2d(inputs, kernel.view(shape).float())
    reference_output = functional.conv2d(inputs, projected_kernel.view(shape).float())
    assert _relative_rms(ours, reference_output) <= _OUTPUT_RELATIVE_LIMIT

    _assert_pair_equivariance_against_escnn(
        escnn,
        kernel.view(shape).double(),
        projected_kernel.view(shape).double(),
        input_frequency=input_frequency,
        output_frequency=output_frequency,
        kernel_size=kernel_size,
    )


def test_mm_expansion_matches_copy_loop_and_uses_independent_rows() -> None:
    """Protect copy-major axes and independent coefficients from silent tying."""
    generator = torch.Generator().manual_seed(23013)
    basis = torch.randn(6, 2 * 7 * 7, generator=generator)
    coefficients = torch.randn(6, 6, generator=generator)
    expanded = _expand_pair(
        coefficients,
        basis,
        output_copies=2,
        input_copies=3,
        output_dimension=1,
        input_dimension=2,
        kernel_size=7,
    )
    direct = _direct_expand_loop(
        coefficients,
        basis,
        shape=(2, 3, 1, 2, 7),
    )
    assert torch.allclose(expanded, direct, atol=1e-6, rtol=1e-6)
    changed = coefficients.clone()
    changed[4, 0] += 1.0
    delta = (
        _expand_pair(
            changed,
            basis,
            output_copies=2,
            input_copies=3,
            output_dimension=1,
            input_dimension=2,
            kernel_size=7,
        )
        - expanded
    )
    nonzero_blocks = delta.view(2, 1, 3, 2, 7, 7).abs().sum(dim=(1, 3, 4, 5)) > 0
    assert torch.equal(
        nonzero_blocks,
        torch.tensor([[False, False, False], [False, True, False]]),
    )


@pytest.mark.parametrize(
    "module",
    [
        _ScalarToF01Conv(R_LAYOUT.n0, A_LAYOUT, _PROFILE_9),
        _F01ToF01Conv(A_LAYOUT, A_LAYOUT),
        _F01ToScalarConv(A_LAYOUT, R_LAYOUT.n0, zero_initialize=True),
    ],
)
def test_each_learned_layer_calls_exactly_one_dense_conv2d(
    module: nn.Module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pin one dense learned convolution so pair banks cannot become copy loops."""
    calls = 0
    original = functional.conv2d

    def counted(*args: object, **kwargs: object) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)  # type: ignore[call-overload]

    monkeypatch.setattr(functional, "conv2d", counted)
    input_channels = cast("int", getattr(module, "input_copies", A_LAYOUT.channels))
    if isinstance(module, _F01ToF01Conv | _F01ToScalarConv):
        input_channels = module.input_layout.channels
    output = cast("torch.Tensor", module(torch.randn(1, input_channels, 5, 5)))
    assert output.shape[-2:] == (5, 5)
    assert calls == 1


def test_representative_multicopy_assembly_matches_escnn_pair_banks() -> None:
    """Validate the three static block layouts with nontrivial copy counts."""
    escnn = _load_local_escnn()
    input_layout = FixedF01Layout("X", 2, 2)
    output_layout = FixedF01Layout("Y", 3, 3)
    scalar_lift = _ScalarToF01Conv(2, input_layout, _PROFILE_7)
    hidden = _F01ToF01Conv(input_layout, output_layout)
    scalar_head = _F01ToScalarConv(input_layout, 3, zero_initialize=False)
    bank00 = _projected_pair_bank(escnn, 0, 0, _PROFILE_7)
    bank10 = _projected_pair_bank(escnn, 0, 1, _PROFILE_7)
    bank01 = _projected_pair_bank(escnn, 1, 0, _PROFILE_7)
    bank11 = _projected_pair_bank(escnn, 1, 1, _PROFILE_7)

    lift_reference = torch.cat(
        (
            _direct_expand_loop(
                scalar_lift.coeff00,
                bank00,
                shape=(2, 2, 1, 1, 7),
            ),
            _direct_expand_loop(
                scalar_lift.coeff10,
                bank10,
                shape=(2, 2, 2, 1, 7),
            ),
        ),
        dim=0,
    )
    hidden_reference = torch.cat(
        (
            torch.cat(
                (
                    _direct_expand_loop(
                        hidden.coeff00,
                        bank00,
                        shape=(3, 2, 1, 1, 7),
                    ),
                    _direct_expand_loop(
                        hidden.coeff01,
                        bank01,
                        shape=(3, 2, 1, 2, 7),
                    ),
                ),
                dim=1,
            ),
            torch.cat(
                (
                    _direct_expand_loop(
                        hidden.coeff10,
                        bank10,
                        shape=(3, 2, 2, 1, 7),
                    ),
                    _direct_expand_loop(
                        hidden.coeff11,
                        bank11,
                        shape=(3, 2, 2, 2, 7),
                    ),
                ),
                dim=1,
            ),
        ),
        dim=0,
    )
    head_reference = torch.cat(
        (
            _direct_expand_loop(
                scalar_head.coeff00,
                bank00,
                shape=(3, 2, 1, 1, 7),
            ),
            _direct_expand_loop(
                scalar_head.coeff01,
                bank01,
                shape=(3, 2, 1, 2, 7),
            ),
        ),
        dim=1,
    )
    for observed, expected in (
        (scalar_lift.expanded_kernel(), lift_reference),
        (hidden.expanded_kernel(), hidden_reference),
        (scalar_head.expanded_kernel(), head_reference),
    ):
        assert _relative_rms(observed, expected) <= _KERNEL_RELATIVE_LIMIT


def test_selected_padded_bmm_gradients_match_four_mm_oracle() -> None:
    """Keep direct assembly differentiable for inputs and every coefficient bank."""
    selected = _F01ToF01Conv(A_LAYOUT, B_LAYOUT).double()
    reference = copy.deepcopy(selected)
    generator = torch.Generator().manual_seed(43013)
    selected_inputs = torch.randn(
        1,
        A_LAYOUT.channels,
        9,
        9,
        generator=generator,
        dtype=torch.float64,
        requires_grad=True,
    )
    reference_inputs = selected_inputs.detach().clone().requires_grad_()

    kernel00 = _expand_pair(
        reference.coeff00,
        _build_pair_bank(0, 0, _PROFILE_7).double(),
        output_copies=B_LAYOUT.n0,
        input_copies=A_LAYOUT.n0,
        output_dimension=1,
        input_dimension=1,
        kernel_size=7,
    )
    kernel10 = _expand_pair(
        reference.coeff10,
        _build_pair_bank(0, 1, _PROFILE_7).double(),
        output_copies=B_LAYOUT.n1,
        input_copies=A_LAYOUT.n0,
        output_dimension=2,
        input_dimension=1,
        kernel_size=7,
    )
    kernel01 = _expand_pair(
        reference.coeff01,
        _build_pair_bank(1, 0, _PROFILE_7).double(),
        output_copies=B_LAYOUT.n0,
        input_copies=A_LAYOUT.n1,
        output_dimension=1,
        input_dimension=2,
        kernel_size=7,
    )
    kernel11 = _expand_pair(
        reference.coeff11,
        _build_pair_bank(1, 1, _PROFILE_7).double(),
        output_copies=B_LAYOUT.n1,
        input_copies=A_LAYOUT.n1,
        output_dimension=2,
        input_dimension=2,
        kernel_size=7,
    )
    reference_kernel = torch.cat(
        (
            torch.cat((kernel00, kernel01), dim=1),
            torch.cat((kernel10, kernel11), dim=1),
        ),
        dim=0,
    )
    selected_output = cast("torch.Tensor", selected(selected_inputs))
    selected_loss = selected_output.square().mean()
    reference_loss = (
        functional
        .conv2d(
            reference_inputs,
            reference_kernel,
            padding=3,
        )
        .square()
        .mean()
    )
    selected_gradients = torch.autograd.grad(
        selected_loss,
        (selected_inputs, *selected.parameters()),
    )
    reference_gradients = torch.autograd.grad(
        reference_loss,
        (reference_inputs, *reference.parameters()),
    )
    assert all(
        _relative_rms(observed, expected) <= _GRADIENT_RELATIVE_LIMIT
        for observed, expected in zip(
            selected_gradients,
            reference_gradients,
            strict=True,
        )
    )


def test_all_locked_oriented_signatures_have_fixed_shapes() -> None:
    """Cover every eventual map shape without assembling the unauthorized VAE."""
    hidden_pairs = (
        (A_LAYOUT, A_LAYOUT),
        (A_LAYOUT, B_LAYOUT),
        (B_LAYOUT, B_LAYOUT),
        (B_LAYOUT, C_LAYOUT),
        (C_LAYOUT, C_LAYOUT),
        (C_LAYOUT, D_LAYOUT),
        (D_LAYOUT, D_LAYOUT),
        (D_LAYOUT, C_LAYOUT),
        (C_LAYOUT, B_LAYOUT),
        (B_LAYOUT, A_LAYOUT),
    )
    for input_layout, output_layout in hidden_pairs:
        module = _F01ToF01Conv(input_layout, output_layout)
        kernel00 = _expand_pair(
            module.coeff00,
            _build_pair_bank(0, 0, _PROFILE_7),
            output_copies=output_layout.n0,
            input_copies=input_layout.n0,
            output_dimension=1,
            input_dimension=1,
            kernel_size=7,
        )
        kernel10 = _expand_pair(
            module.coeff10,
            _build_pair_bank(0, 1, _PROFILE_7),
            output_copies=output_layout.n1,
            input_copies=input_layout.n0,
            output_dimension=2,
            input_dimension=1,
            kernel_size=7,
        )
        kernel01 = _expand_pair(
            module.coeff01,
            _build_pair_bank(1, 0, _PROFILE_7),
            output_copies=output_layout.n0,
            input_copies=input_layout.n1,
            output_dimension=1,
            input_dimension=2,
            kernel_size=7,
        )
        kernel11 = _expand_pair(
            module.coeff11,
            _build_pair_bank(1, 1, _PROFILE_7),
            output_copies=output_layout.n1,
            input_copies=input_layout.n1,
            output_dimension=2,
            input_dimension=2,
            kernel_size=7,
        )
        expected_kernel = torch.cat(
            (
                torch.cat((kernel00, kernel01), dim=1),
                torch.cat((kernel10, kernel11), dim=1),
            ),
            dim=0,
        )
        assert torch.allclose(module.expanded_kernel(), expected_kernel)
        result = cast(
            "torch.Tensor",
            module(torch.randn(1, input_layout.channels, 3, 3)),
        )
        assert result.shape == (1, output_layout.channels, 3, 3)
    assert cast(
        "torch.Tensor",
        _ScalarToF01Conv(R_LAYOUT.n0, A_LAYOUT, _PROFILE_9)(torch.randn(1, 3, 3, 3)),
    ).shape == (1, A_LAYOUT.channels, 3, 3)
    assert cast(
        "torch.Tensor",
        _ScalarToF01Conv(L_LAYOUT.n0, D_LAYOUT, _PROFILE_7)(torch.randn(1, 16, 3, 3)),
    ).shape == (1, D_LAYOUT.channels, 3, 3)
    assert cast(
        "torch.Tensor",
        _F01ToScalarConv(D_LAYOUT, L_LAYOUT.n0, zero_initialize=False)(
            torch.randn(1, D_LAYOUT.channels, 3, 3),
        ),
    ).shape == (1, L_LAYOUT.channels, 3, 3)
    assert cast(
        "torch.Tensor",
        _F01ToScalarConv(A_LAYOUT, R_LAYOUT.n0, zero_initialize=True)(
            torch.randn(1, A_LAYOUT.channels, 3, 3),
        ),
    ).shape == (1, R_LAYOUT.channels, 3, 3)


def _one_copy_gradients(
    coefficient_values: torch.Tensor,
    input_values: torch.Tensor,
    basis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    coefficients = coefficient_values.detach().clone().requires_grad_()
    inputs = input_values.detach().clone().requires_grad_()
    kernel = (coefficients @ basis).view(2, 2, 7, 7)
    loss = functional.conv2d(inputs, kernel, padding=3).square().mean()
    return cast(
        "tuple[torch.Tensor, torch.Tensor]",
        torch.autograd.grad(loss, (coefficients, inputs)),
    )


def test_fp64_runtime_expansion_gradients_match_escnn_coordinates() -> None:
    """Catch contraction errors even when dense kernels happen to look plausible."""
    escnn = _load_local_escnn()
    runtime = _build_pair_bank(1, 1, _PROFILE_7).double()
    reference = _escnn_pair_matrix(
        escnn,
        1,
        1,
        profile=_PROFILE_7,
    )
    _projected, coordinates = _reference_projection(runtime, reference)
    generator = torch.Generator().manual_seed(33013)
    coefficient_values = torch.randn(
        1,
        runtime.shape[0],
        generator=generator,
        dtype=torch.float64,
    )
    input_values = torch.randn(1, 2, 11, 11, generator=generator, dtype=torch.float64)

    ours_coefficient_gradient, ours_input_gradient = _one_copy_gradients(
        coefficient_values,
        input_values,
        runtime,
    )
    reference_basis = coordinates.transpose(0, 1) @ reference.transpose(0, 1)
    reference_coefficient_gradient, reference_input_gradient = _one_copy_gradients(
        coefficient_values,
        input_values,
        reference_basis,
    )
    assert (
        _relative_rms(ours_coefficient_gradient, reference_coefficient_gradient)
        <= _GRADIENT_RELATIVE_LIMIT
    )
    assert (
        _relative_rms(ours_input_gradient, reference_input_gradient)
        <= _GRADIENT_RELATIVE_LIMIT
    )


def _rotate_f1_components(
    inputs: torch.Tensor,
    layout: FixedF01Layout,
    angle: float,
) -> torch.Tensor:
    output = inputs.clone()
    vector = inputs[:, layout.f1_offset :].view(
        inputs.shape[0],
        layout.n1,
        2,
        *inputs.shape[-2:],
    )
    rotation = torch.tensor(
        ((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle))),
        dtype=inputs.dtype,
    )
    rotated = torch.einsum("ij,bnjhw->bnihw", rotation, vector)
    output[:, layout.f1_offset :] = rotated.reshape_as(output[:, layout.f1_offset :])
    return output


def _smooth_layout_input(
    layout: FixedF01Layout,
    *,
    spatial_size: int = 65,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(43013 + layout.channels)
    values = torch.randn(
        1,
        layout.channels,
        spatial_size,
        spatial_size,
        generator=generator,
    )
    for _ in range(4):
        values = functional.avg_pool2d(values, kernel_size=5, stride=1, padding=2)
    return values / values.square().mean().sqrt()


def _escnn_transform_layout(
    escnn: _Escnn,
    layout: FixedF01Layout,
    values: torch.Tensor,
    degrees: int,
) -> torch.Tensor:
    group_space = escnn.gspaces.rot2dOnR2(N=-1, maximum_frequency=1)
    group = group_space.fibergroup
    field_type = escnn.nn.FieldType(
        group_space,
        [group.irrep(0)] * layout.n0 + [group.irrep(1)] * layout.n1,
    )
    element = group.element(math.radians(degrees), "radians")
    if degrees % 90 == 0:
        fibers = field_type.transform_fibers(values, element)
        return torch.rot90(fibers, k=degrees // 90, dims=(-2, -1))
    return field_type.transform(values, element, order=1)


def _equivariance_errors(  # noqa: PLR0913
    escnn: _Escnn,
    module: nn.Module,
    input_layout: FixedF01Layout,
    output_layout: FixedF01Layout,
    degrees: int,
    *,
    spatial_size: int = 65,
) -> tuple[float, float, float]:
    inputs = _smooth_layout_input(input_layout, spatial_size=spatial_size)
    transformed_inputs = _escnn_transform_layout(
        escnn,
        input_layout,
        inputs,
        degrees,
    )
    observed = cast("torch.Tensor", module(transformed_inputs))
    expected = _escnn_transform_layout(
        escnn,
        output_layout,
        cast("torch.Tensor", module(inputs)),
        degrees,
    )
    inverse = _escnn_transform_layout(
        escnn,
        input_layout,
        transformed_inputs,
        -degrees,
    )
    crop = 7
    return (
        _relative_rms(observed, expected),
        _relative_rms(
            observed[..., crop:-crop, crop:-crop],
            expected[..., crop:-crop, crop:-crop],
        ),
        _relative_rms(
            inverse[..., crop:-crop, crop:-crop],
            inputs[..., crop:-crop, crop:-crop],
        ),
    )


def _equivariance_module(name: str) -> nn.Module:
    if name == "primitive":
        return _F01ToF01Conv(A_LAYOUT, A_LAYOUT)
    if name == "identity":
        return SO2IdentityResidualBlockA()
    if name == "encoder":
        return SO2EncoderTransitionAB()
    if name == "decoder":
        return SO2DecoderTransitionBA()
    raise AssertionError(name)


@pytest.mark.parametrize("degrees", [15, 30, 45, 60, 90])
@pytest.mark.parametrize(
    ("module_name", "input_layout", "output_layout", "contains_resampling"),
    [
        ("primitive", A_LAYOUT, A_LAYOUT, False),
        ("identity", A_LAYOUT, A_LAYOUT, False),
        ("encoder", A_LAYOUT, B_LAYOUT, True),
        ("decoder", B_LAYOUT, A_LAYOUT, True),
    ],
)
def test_sampled_primitive_and_block_equivariance_is_reportable(
    module_name: str,
    input_layout: FixedF01Layout,
    output_layout: FixedF01Layout,
    *,
    contains_resampling: bool,
    degrees: int,
) -> None:
    """Exercise every locked angle and keep resampling error separate from kernels."""
    cast("Callable[[int], torch.Generator]", torch.manual_seed)(53013)
    full, cropped, transform_floor = _equivariance_errors(
        _load_local_escnn(),
        _equivariance_module(module_name),
        input_layout,
        output_layout,
        degrees,
        spatial_size=64 if contains_resampling else 65,
    )
    assert all(math.isfinite(value) for value in (full, cropped, transform_floor))
    if degrees == _CARDINAL_DEGREES and not contains_resampling:
        assert cropped <= _CARDINAL_EQUIVARIANCE_LIMIT


@pytest.mark.parametrize("module_type", [FixedF01FieldNorm, FixedF01RadialGate])
def test_field_norm_and_gate_commute_with_f1_component_rotation(
    module_type: type[nn.Module],
) -> None:
    """Reject componentwise vector statistics or gates that break SO(2) fields."""
    module = module_type(A_LAYOUT)
    inputs = torch.randn(2, A_LAYOUT.channels, 7, 7)
    rotated_inputs = _rotate_f1_components(inputs, A_LAYOUT, 0.37)
    expected = _rotate_f1_components(
        cast("torch.Tensor", module(inputs)),
        A_LAYOUT,
        0.37,
    )
    observed = cast("torch.Tensor", module(rotated_inputs))
    assert torch.allclose(observed, expected, atol=2e-6, rtol=2e-6)


def test_norm_gate_fp32_islands_preserve_dtype_and_zero_radius_gradients() -> None:
    """Keep AMP-facing invariant math finite where FP16 squaring would overflow."""
    inputs = torch.zeros(
        1,
        A_LAYOUT.channels,
        3,
        3,
        dtype=torch.float16,
        requires_grad=True,
    )
    with torch.no_grad():
        inputs[:, A_LAYOUT.f1_offset] = 300.0
    norm = FixedF01FieldNorm(A_LAYOUT)
    gate = FixedF01RadialGate(A_LAYOUT)
    normalized = cast("torch.Tensor", norm(inputs))
    output = cast("torch.Tensor", gate(normalized))
    assert output.dtype == torch.float16
    assert torch.isfinite(output).all()
    output.float().sum().backward()  # pyright: ignore[reportUnknownMemberType]
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()


def test_residual_and_resampling_contracts() -> None:
    """Pin the minimal probe topology so it cannot drift into the full VAE."""
    identity = SO2IdentityResidualBlockA()
    encoder = SO2EncoderTransitionAB()
    decoder = SO2DecoderTransitionBA()
    assert cast("torch.Tensor", identity(torch.randn(1, 48, 6, 6))).shape == (
        1,
        48,
        6,
        6,
    )
    assert cast("torch.Tensor", encoder(torch.randn(1, 48, 8, 8))).shape == (
        1,
        72,
        4,
        4,
    )
    assert cast("torch.Tensor", decoder(torch.randn(1, 72, 4, 4))).shape == (
        1,
        48,
        8,
        8,
    )

    downsample = _FixedF01Downsample2x(A_LAYOUT.channels)
    upsample = _FixedF01Upsample2x()
    assert downsample.weight.shape == (48, 1, 5, 5)
    assert downsample.weight.dtype == torch.float32
    assert downsample.weight.is_contiguous()
    assert "weight" in downsample.state_dict()
    assert cast("torch.Tensor", upsample(torch.randn(1, 48, 4, 4))).shape == (
        1,
        48,
        8,
        8,
    )


def test_rgb_and_scalar_latent_contracts() -> None:
    """Keep scalar interfaces and initialization aligned with the control VAE."""
    lift = SO2RGBLift()
    latent_projection = SO2LatentProjection()
    heads = SO2ScalarLatentHeads()
    rgb = SO2RGBHead()
    lifted = cast("torch.Tensor", lift(torch.randn(1, 3, 8, 8)))
    projected = cast("torch.Tensor", latent_projection(torch.randn(1, 16, 4, 4)))
    eps = torch.randn(1, 16, 4, 4)
    mu, logvar, clamped, latent = cast(
        "tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]",
        heads(torch.randn(1, 144, 4, 4), eps),
    )
    assert lifted.shape == (1, 48, 8, 8)
    assert projected.shape == (1, 144, 4, 4)
    assert mu.shape == logvar.shape == clamped.shape == latent.shape == eps.shape
    assert torch.equal(clamped, clamp_logvar(logvar))
    assert torch.allclose(latent, mu + torch.exp(0.5 * clamped) * eps)
    assert torch.count_nonzero(rgb.conv.coeff00) == 0
    assert torch.count_nonzero(rgb.conv.coeff01) == 0
    assert torch.count_nonzero(rgb.conv.bias) == 0
    assert torch.count_nonzero(cast("torch.Tensor", rgb(lifted))) == 0
    assert not torch.equal(heads.mu.bias, torch.zeros_like(heads.mu.bias))
    assert not torch.equal(heads.logvar.bias, torch.zeros_like(heads.logvar.bias))


def test_eager_backward_optimizer_and_cpu_fullgraph_mechanics() -> None:
    """Prove the fixed hot path trains and remains one capturable static graph."""
    module = SO2IdentityResidualBlockA()
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-3)
    inputs = torch.randn(1, A_LAYOUT.channels, 5, 5)
    target = torch.randn_like(inputs)
    before = module.main_conv1.coeff00.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    output = cast("torch.Tensor", module(inputs))
    loss = functional.mse_loss(output, target)
    loss.backward()  # pyright: ignore[reportUnknownMemberType]
    optimizer.step()  # pyright: ignore[reportUnknownMemberType]
    assert torch.isfinite(loss)
    assert not torch.equal(before, module.main_conv1.coeff00)

    compiled = cast(
        "Callable[[torch.Tensor], torch.Tensor]",
        torch.compile(  # pyright: ignore[reportUnknownMemberType]
            module,
            backend="eager",
            fullgraph=True,
            dynamic=False,
        ),
    )
    compiled_output = compiled(inputs)
    eager_output = cast("torch.Tensor", module(inputs))
    assert torch.allclose(compiled_output, eager_output, atol=1e-6, rtol=1e-6)
