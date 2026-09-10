# Copyright 2026 HiperMaximus
"""Tests for decoded latent-transform evaluation primitives."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from eqvae.evaluation.decoded_transform import (
    EXACT_D4_NAMES,
    EXACT_D4_NONIDENTITY_NAMES,
    aggregate_rms_ratio,
    exact_spatial_transform,
    gradient_mse_per_image,
    inverse_exact_transform_name,
    masked_mae_per_image,
    masked_mse_per_image,
    out_of_range_per_image,
    paired_bootstrap_median_difference,
)

EXPECTED_D4_ELEMENTS = 8
EXPECTED_D4_NONIDENTITY_ELEMENTS = 7
EXPECTED_FAVORABLE_COUNT = 3
EXPECTED_CLUSTER_TEST_PATCHES = 4
EXPECTED_CLUSTER_TEST_CLUSTERS = 2


def test_exact_d4_transforms_are_invertible_and_flip_product_is_rot180() -> None:
    """The declared operations form the expected exact eight-element set."""
    values = torch.arange(2 * 3 * 4).reshape(1, 2, 3, 4)
    for name in EXACT_D4_NAMES:
        transformed = exact_spatial_transform(values, name)
        restored = exact_spatial_transform(
            transformed,
            inverse_exact_transform_name(name),
        )
        assert torch.equal(restored, values)
    flipped_twice = exact_spatial_transform(
        exact_spatial_transform(values, "flip_h"),
        "flip_v",
    )
    assert torch.equal(flipped_twice, exact_spatial_transform(values, "rot180"))
    assert len(EXACT_D4_NAMES) == EXPECTED_D4_ELEMENTS
    assert len(EXACT_D4_NONIDENTITY_NAMES) == EXPECTED_D4_NONIDENTITY_ELEMENTS
    assert torch.equal(
        exact_spatial_transform(
            exact_spatial_transform(values, "flip_h"),
            "rot90",
        ),
        exact_spatial_transform(values, "flip_diag"),
    )


def test_exact_d4_rejects_unknown_name() -> None:
    """Unknown operation names fail closed."""
    with pytest.raises(ValueError, match="unknown exact spatial transform"):
        exact_spatial_transform(torch.zeros(1, 1, 2, 2), "bad")


def test_exact_d4_is_distinct_closed_and_obeys_dihedral_relation() -> None:
    """An asymmetric square fixture exercises all Cayley products."""
    values = torch.arange(25).reshape(1, 1, 5, 5)
    elements = {name: exact_spatial_transform(values, name) for name in EXACT_D4_NAMES}
    assert len({tensor.numpy().tobytes() for tensor in elements.values()}) == len(
        EXACT_D4_NAMES,
    )
    for left in EXACT_D4_NAMES:
        for right in EXACT_D4_NAMES:
            product = exact_spatial_transform(elements[right], left)
            assert any(
                torch.equal(product, candidate) for candidate in elements.values()
            )
    srs = exact_spatial_transform(
        exact_spatial_transform(
            exact_spatial_transform(values, "flip_h"),
            "rot90",
        ),
        "flip_h",
    )
    assert torch.equal(srs, exact_spatial_transform(values, "rot270"))


def test_masked_errors_and_aggregate_ratio() -> None:
    """Masked scalar errors feed the patch-first aggregate ratio exactly."""
    target = torch.zeros(2, 1, 2, 2)
    prediction = target.clone()
    prediction[0, 0, 0, 0] = 2.0
    prediction[1, 0, 0, 0] = 4.0
    mask = torch.tensor([[True, False], [False, False]])
    assert torch.equal(
        masked_mse_per_image(prediction, target, mask),
        torch.tensor([4.0, 16.0]),
    )
    assert torch.equal(
        masked_mae_per_image(prediction, target, mask),
        torch.tensor([2.0, 4.0]),
    )
    ratio = aggregate_rms_ratio(
        np.array([[1.0, 9.0], [4.0, 4.0]]),
        np.array([[4.0, 16.0], [16.0, 16.0]]),
        epsilon=0.0,
    )
    assert ratio == pytest.approx(np.array([np.sqrt(5.0 / 10.0), 0.5]))


def test_out_of_range_and_gradient_metrics() -> None:
    """Artifact and gradient metrics retain pre-clamp values."""
    values = torch.tensor([[[[-1.5, 0.0], [1.0, 2.0]]]])
    fraction, overshoot = out_of_range_per_image(values)
    assert fraction.item() == pytest.approx(0.5)
    assert overshoot.item() == pytest.approx(0.375)
    mask = torch.ones(2, 2, dtype=torch.bool)
    assert gradient_mse_per_image(values, values, mask).item() == pytest.approx(0.0)


def test_paired_bootstrap_is_deterministic_and_uses_so2_minus_normal() -> None:
    """Bootstrap direction and seed are stable."""
    first = paired_bootstrap_median_difference([2, 3, 4], [1, 2, 3], seed=7, draws=100)
    second = paired_bootstrap_median_difference([2, 3, 4], [1, 2, 3], seed=7, draws=100)
    assert first == second
    assert first["patch_median_difference_so2_minus_normal"] == pytest.approx(-1.0)
    assert first["cluster_median_difference_so2_minus_normal"] == pytest.approx(-1.0)
    assert first["so2_favorable_patch_count"] == EXPECTED_FAVORABLE_COUNT
    higher = paired_bootstrap_median_difference(
        [1, 2, 3],
        [2, 3, 4],
        seed=7,
        draws=100,
        favorable_direction="higher",
    )
    neutral = paired_bootstrap_median_difference(
        [1, 2, 3],
        [2, 3, 4],
        seed=7,
        draws=100,
        favorable_direction=None,
    )
    assert higher["so2_favorable_patch_count"] == EXPECTED_FAVORABLE_COUNT
    assert neutral["so2_favorable_patch_count"] is None


def test_paired_bootstrap_reduces_patch_differences_within_wsi() -> None:
    """Repeated patches are reduced to one median difference per WSI."""
    result = paired_bootstrap_median_difference(
        [0, 0, 0, 0],
        [1, 3, 100, 200],
        cluster_ids=["a", "a", "b", "b"],
        seed=3,
        draws=100,
        favorable_direction="higher",
    )
    assert result["n_patches"] == EXPECTED_CLUSTER_TEST_PATCHES
    assert result["n_clusters"] == EXPECTED_CLUSTER_TEST_CLUSTERS
    assert result["cluster_median_difference_so2_minus_normal"] == pytest.approx(76.0)
