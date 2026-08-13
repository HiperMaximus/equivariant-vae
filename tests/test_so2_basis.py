# Copyright 2026 HiperMaximus
"""Regression tests for Spec 0012's locked analytic SO(2) basis."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import sys
import types
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from eqvae.models.so2_basis import (
    PAIR_FREQUENCIES,
    generalized_he_standard_deviation,
    irrep_dimension,
    orthonormalize_columns,
    pair_angular_orders,
    representation_matrix,
    sample_pair_basis,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

EXPECTED_ORDERS = {
    (0, 0): (0,),
    (0, 1): (1, 1),
    (0, 2): (2, 2),
    (1, 0): (1, 1),
    (1, 1): (0, 0, 2, 2),
    (1, 2): (1, 1, 3, 3),
    (2, 0): (2, 2),
    (2, 1): (1, 1, 3, 3),
    (2, 2): (0, 0, 4, 4),
}
SPAN_TOLERANCE = 5e-5
ORIGIN_PROXY_TOLERANCE = 1e-12
EXPECTED_HIGH_ORDER_DIMENSION = 24
EXPECTED_F01_PARAMETERS = 1_180_035
EXPECTED_F01_COEFFICIENTS = 1_172_304
EXPECTED_F01_NORMALIZATION_PARAMETERS = 3_600
EXPECTED_F01_GATE_PARAMETERS = 4_096
EXPECTED_F01_BIASES = 35
EXPECTED_F01_DENSE_MACS = 159_837_585_408
EXPECTED_F01_EXPANSION_MACS = 159_453_168
EXPECTED_INITIALIZATION_TRIALS = 128
EXPECTED_DISTINCT_LAYER_TYPES = 13
INITIALIZATION_RATIO_MINIMUM = 0.9
INITIALIZATION_RATIO_MAXIMUM = 1.1
PRESERVED_AUDIT_HASHES = {
    "search": "05c71843b1f9d83a66c260e887dcd3e5bbee5a59822807bbc02f24cfb8aa32f1",
    "profiles": "0c3965696b96ef79e6e19bd12bc9fd3b0958820e576af43d1a18dd8f47ea9418",
    "escnn_reference": (
        "82d507703ba9f8a40838871f9e19412a771551a38ec4d7ff77be7335248cba08"
    ),
    "high_order_gate": (
        "b69e5fc8a97e9a400e44a03c5080d1c57a8d5e5c5cdb1d27ff31b3f73d2c0805"
    ),
    "locked_premise_findings": (
        "75e17eee78877ed0f48ef5a6cbebe01b7b3c916c017ea7d8bfe883c783709581"
    ),
}
AttributeValue = int | float | str
FloatArray = NDArray[np.float64]


class _EscnnBasis(Protocol):
    def to(self, *, dtype: torch.dtype) -> _EscnnBasis: ...

    def sample(self, points: torch.Tensor) -> torch.Tensor: ...

    def __iter__(self) -> Iterator[dict[str, AttributeValue]]: ...

    def __len__(self) -> int: ...


class _SO2Group(Protocol):
    def irrep(self, frequency: int) -> object: ...


class _GroupApi(Protocol):
    def so2_group(self, maximum_frequency: int) -> _SO2Group: ...


class _KernelsApi(Protocol):
    def kernels_SO2_act_R2(  # noqa: N802, PLR0913
        self,
        input_representation: object,
        output_representation: object,
        radii: list[float],
        sigma: list[float],
        *,
        maximum_frequency: int,
        filter: Callable[[dict[str, AttributeValue]], bool] | None = None,  # noqa: A002
    ) -> _EscnnBasis: ...


class _Escnn(Protocol):
    group: _GroupApi
    kernels: _KernelsApi


def test_generalized_he_scale_uses_every_locked_count() -> None:
    """Catch omission of T_b, input copies, or pair-basis dimension."""
    baseline = generalized_he_standard_deviation(
        present_input_frequency_count=2,
        input_copies=3,
        basis_dimension=5,
    )
    assert baseline == pytest.approx(1.0 / math.sqrt(30.0))
    assert generalized_he_standard_deviation(
        present_input_frequency_count=1,
        input_copies=3,
        basis_dimension=5,
    ) != pytest.approx(baseline)
    assert generalized_he_standard_deviation(
        present_input_frequency_count=2,
        input_copies=1,
        basis_dimension=5,
    ) != pytest.approx(baseline)
    assert generalized_he_standard_deviation(
        present_input_frequency_count=2,
        input_copies=3,
        basis_dimension=1,
    ) != pytest.approx(baseline)


def test_committed_oracle_evidence_guards_locked_decision() -> None:  # noqa: PLR0914
    """Guard the measured fallback decision and non-tautological init evidence."""
    repository = Path(__file__).resolve().parents[1]
    manifest = cast(
        "dict[str, object]",
        json.loads(
            (repository / "configs/spec0012/so2_basis_manifest.json").read_text(
                encoding="utf-8",
            ),
        ),
    )
    audit = cast(
        "dict[str, object]",
        json.loads(
            (repository / "docs/data/spec0012_so2_basis_audit.json").read_text(
                encoding="utf-8",
            ),
        ),
    )
    provisional = cast("dict[str, object]", audit["provisional_candidate"])
    assert provisional == {
        "blockers": [],
        "name": "F01",
        "reference_profiles": ["7-low", "9-low"],
        "status": "pass",
    }
    reference = cast("dict[str, dict[str, object]]", audit["escnn_reference"])
    for profile_name in cast("list[str]", provisional["reference_profiles"]):
        assert reference[profile_name]["status"] == "pass"
    search = cast("dict[str, dict[str, object]]", audit["search"])
    assert search["7-full"]["legal_coarse_start_count"] == 0
    high_order = cast("dict[str, object]", audit["high_order_gate"])
    supports = cast("dict[str, dict[str, object]]", high_order["supports"])
    nine = supports["9"]
    incremental = cast("dict[str, object]", nine["incremental_escnn_reference"])
    assert incremental["status"] == "pass"
    assert cast("int", nine["d_high"]) == EXPECTED_HIGH_ORDER_DIMENSION
    assert cast("float", nine["e_high"]) > cast("float", nine["e_limit"])
    counts = cast("dict[str, dict[str, object]]", audit["architecture_counts"])
    f01_counts = counts["F01"]
    assert f01_counts == {
        "basis_expansion_macs_per_forward": EXPECTED_F01_EXPANSION_MACS,
        "coefficient_parameters": EXPECTED_F01_COEFFICIENTS,
        "dense_convolution_macs_per_sample": EXPECTED_F01_DENSE_MACS,
        "gate_module_count": 34,
        "gate_parameters": EXPECTED_F01_GATE_PARAMETERS,
        "learned_convolution_count": 43,
        "missing_profiles": [],
        "normalization_module_count": 40,
        "normalization_parameters": EXPECTED_F01_NORMALIZATION_PARAMETERS,
        "physical_widths": [48, 72, 96, 144],
        "scalar_bias_parameters": EXPECTED_F01_BIASES,
        "status": "pass",
        "total_learned_parameters": EXPECTED_F01_PARAMETERS,
        "within_parameter_cap": True,
    }
    selected = cast("dict[str, object]", manifest["selected_architecture"])
    assert selected["name"] == "F01_equal_copy"
    assert selected["total_learned_parameters"] == EXPECTED_F01_PARAMETERS
    field_layout = cast("dict[str, dict[str, object]]", selected["field_layout"])
    assert [field_layout[name]["copies"] for name in ("A", "B", "C", "D")] == [
        [16, 16, 0],
        [24, 24, 0],
        [32, 32, 0],
        [48, 48, 0],
    ]
    assert [field_layout[name]["packed_channels"] for name in ("A", "B", "C", "D")] == [
        48,
        72,
        96,
        144,
    ]
    initialization = cast("dict[str, object]", audit["initialization_variance"])
    assert initialization["trial_count"] == EXPECTED_INITIALIZATION_TRIALS
    assert initialization["distinct_layer_type_count"] == EXPECTED_DISTINCT_LAYER_TYPES
    layers = cast("dict[str, dict[str, object]]", initialization["layers"])
    ratios: list[float] = []
    for layer in layers.values():
        ratio_map = cast(
            "dict[str, float]",
            layer["mean_variance_ratio_by_frequency"],
        )
        ratios.extend(ratio_map.values())
    assert all(
        INITIALIZATION_RATIO_MINIMUM <= ratio <= INITIALIZATION_RATIO_MAXIMUM
        for ratio in ratios
    )
    assert any(not math.isclose(ratio, 1.0, abs_tol=1e-6) for ratio in ratios)


def test_pair_generator_table_retains_all_real_so2_paths() -> None:
    """Retain I/J and sum-order paths so the VAE keeps all valid kernels."""
    observed = {pair: pair_angular_orders(*pair) for pair in PAIR_FREQUENCIES}
    assert observed == EXPECTED_ORDERS


def test_layout_refresh_preserves_radial_and_f2_evidence() -> None:
    """Pin layout-independent evidence so a narrow refresh cannot reopen F2."""
    repository = Path(__file__).resolve().parents[1]
    audit = cast(
        "dict[str, object]",
        json.loads(
            (repository / "docs/data/spec0012_so2_basis_audit.json").read_text(
                encoding="utf-8",
            ),
        ),
    )
    observed = {
        key: hashlib.sha256(
            json.dumps(
                audit[key],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            + b"\n",
        ).hexdigest()
        for key in PRESERVED_AUDIT_HASHES
    }
    assert observed == PRESERVED_AUDIT_HASHES


@pytest.mark.parametrize(("input_frequency", "output_frequency"), PAIR_FREQUENCIES)
def test_sampled_columns_obey_exact_cardinal_kernel_constraint(
    input_frequency: int,
    output_frequency: int,
) -> None:
    """Require every analytic column to satisfy the derived cardinal action."""
    sampled = sample_pair_basis(
        input_frequency=input_frequency,
        output_frequency=output_frequency,
        kernel_size=7,
        centres=(1.0, 2.0, 2.75),
        widths=(0.5, 0.6, 0.5),
        qmax=(2, 4, 4),
    )
    output_action = representation_matrix(output_frequency, math.pi / 2.0)
    input_inverse = representation_matrix(input_frequency, -math.pi / 2.0)
    centre = 3
    for row in range(7):
        for column in range(7):
            x_coordinate = column - centre
            y_coordinate = centre - row
            rotated_row = centre - x_coordinate
            rotated_column = centre - y_coordinate
            left = sampled.values[..., rotated_row, rotated_column]
            right = cast(
                "FloatArray",
                np.einsum(
                    "oa,pab,bi->poi",
                    output_action,
                    sampled.values[..., row, column],
                    input_inverse,
                ),
            )
            assert np.allclose(left, right, atol=2e-14, rtol=2e-14)


def test_nonzero_orders_have_zero_centre_but_q0_intertwiners_survive() -> None:
    """Remove origin ambiguity without deleting legal F2 centre maps."""
    sampled = sample_pair_basis(
        input_frequency=2,
        output_frequency=2,
        kernel_size=9,
        centres=(1.0, 2.0, 3.0, 3.75),
        widths=(0.5, 0.5, 0.6, 0.5),
        qmax=(2, 4, 4, 4),
    )
    centre_values = sampled.values[..., 4, 4]
    for index, column in enumerate(sampled.columns):
        if column.angular_order > 0:
            column_values = cast("FloatArray", centre_values[index])
            assert np.count_nonzero(column_values) == 0
    assert any(
        column.angular_order == 0
        and np.count_nonzero(cast("FloatArray", centre_values[index])) > 0
        for index, column in enumerate(sampled.columns)
    )


def test_analytic_spans_match_locked_escnn_reference() -> None:  # noqa: PLR0914
    """Catch a complete but wrongly signed real-basis convention with escnn."""
    escnn = _load_local_escnn()
    group = escnn.group.so2_group(4)
    centres = (1.0, 2.0, 2.75)
    widths = (0.5, 0.6, 0.5)
    qmax = (2, 4, 4)
    points = torch.tensor(
        [(column - 3, 3 - row) for row in range(7) for column in range(7)],
        dtype=torch.float64,
    )
    for input_frequency, output_frequency in PAIR_FREQUENCIES:
        ours = sample_pair_basis(
            input_frequency=input_frequency,
            output_frequency=output_frequency,
            kernel_size=7,
            centres=centres,
            widths=widths,
            qmax=qmax,
        ).flat_columns()

        reference_blocks: list[FloatArray] = []
        reference_dimension = 0
        orders = pair_angular_orders(input_frequency, output_frequency)
        shells = ((0.0, 0.005, 0), *zip(centres, widths, qmax, strict=True))
        for radius, width, cutoff in shells:
            if not any(order <= cutoff for order in orders):
                continue
            basis = copy.deepcopy(
                escnn.kernels.kernels_SO2_act_R2(
                    group.irrep(input_frequency),
                    group.irrep(output_frequency),
                    [radius],
                    [width],
                    maximum_frequency=cutoff,
                ),
            ).to(dtype=torch.float64)
            block = cast("FloatArray", basis.sample(points).detach().numpy())
            reference_dimension += len(basis)
            if math.isclose(radius, 0.0, abs_tol=1e-7):
                off_centre = np.delete(block, 24, axis=0)
                maximum = float(
                    np.max(  # pyright: ignore[reportAny]
                        np.abs(off_centre),
                        initial=0.0,
                    ),
                )
                assert maximum < ORIGIN_PROXY_TOLERANCE
                centre_value = cast(
                    "FloatArray",
                    block[24].copy(),  # pyright: ignore[reportAny]
                )
                block[:] = 0.0
                block[24] = centre_value
            reference_blocks.append(block)
        reference_sample = np.concatenate(reference_blocks, axis=1)
        reference = cast(
            "FloatArray",
            reference_sample.transpose(2, 3, 0, 1).reshape(
                irrep_dimension(output_frequency)
                * irrep_dimension(input_frequency)
                * 49,
                reference_dimension,
            ),
        )
        ours_q = orthonormalize_columns(ours)
        reference_q = orthonormalize_columns(reference)
        span_distance = np.linalg.norm(
            ours_q @ ours_q.T - reference_q @ reference_q.T,
            ord=2,
        )
        pair = (input_frequency, output_frequency)
        assert ours.shape[1] == reference.shape[1], pair
        assert span_distance <= SPAN_TOLERANCE, pair


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
        message = "SO(2) oracle entered an SO(3) path"
        raise RuntimeError(message)

    sys.modules[module_names[-1]].wigner_D_matrix = reject_so3  # type: ignore[attr-defined]
    escnn_root = Path(__file__).resolve().parents[1] / "reference/escnn"
    sys.path.insert(0, str(escnn_root))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        escnn = importlib.import_module("escnn")

    return cast("_Escnn", escnn)
