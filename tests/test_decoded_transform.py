# Copyright 2026 HiperMaximus
"""Tests for decoded latent-transform evaluation primitives."""

from __future__ import annotations

import pytest
import torch

from eqvae.evaluation.decoded_transform import (
    EXACT_D4_NAMES,
    EXACT_D4_NONIDENTITY_NAMES,
    exact_spatial_transform,
    inverse_exact_transform_name,
)

EXPECTED_D4_ELEMENTS = 8
EXPECTED_D4_NONIDENTITY_ELEMENTS = 7


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
