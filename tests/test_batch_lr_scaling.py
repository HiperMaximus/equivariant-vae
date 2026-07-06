# Copyright 2026 HiperMaximus
"""Tests for the batch-size to learning-rate scaling primitives (Spec 0011 S2)."""

from __future__ import annotations

import math

import pytest

from eqvae.training.optim import BatchLrScaling, scaled_learning_rate

_REFERENCE_BATCH = 24
_QUADRUPLE_BATCH = 96
_REFERENCE_LR = 5.0e-4
_SQRT_EXPONENT = 0.5
_LINEAR_EXPONENT = 1.0
_SQRT_QUADRUPLE_MULTIPLIER = 2.0
_LINEAR_QUADRUPLE_MULTIPLIER = 4.0


def test_default_rule_is_sqrt() -> None:
    """The default scaling rule is the square-root rule."""
    scaling = BatchLrScaling(reference_global_batch_size=_REFERENCE_BATCH)
    assert scaling.rule == "sqrt"
    assert math.isclose(scaling.exponent, _SQRT_EXPONENT)


def test_exponent_maps_known_rules() -> None:
    """Each known rule maps to its batch-ratio exponent."""
    sqrt = BatchLrScaling(reference_global_batch_size=_REFERENCE_BATCH, rule="sqrt")
    linear = BatchLrScaling(reference_global_batch_size=_REFERENCE_BATCH, rule="linear")
    assert math.isclose(sqrt.exponent, _SQRT_EXPONENT)
    assert math.isclose(linear.exponent, _LINEAR_EXPONENT)


def test_unknown_rule_rejected_at_construction() -> None:
    """An unregistered rule fails closed rather than silently defaulting."""
    with pytest.raises(ValueError, match="unknown batch-lr scaling rule"):
        BatchLrScaling(reference_global_batch_size=_REFERENCE_BATCH, rule="cubic")


def test_nonpositive_reference_batch_rejected() -> None:
    """A non-positive reference batch is rejected (would divide by zero)."""
    with pytest.raises(ValueError, match="reference_global_batch_size must be"):
        BatchLrScaling(reference_global_batch_size=0)


def test_reference_batch_is_behavior_preserving() -> None:
    """At the reference batch the multiplier is exactly 1.0 (lr unchanged)."""
    scaling = BatchLrScaling(reference_global_batch_size=_REFERENCE_BATCH, rule="sqrt")
    lr = scaled_learning_rate(
        reference_lr=_REFERENCE_LR,
        scaling=scaling,
        global_batch_size=_REFERENCE_BATCH,
    )
    assert math.isclose(lr, _REFERENCE_LR)


def test_sqrt_scaling_doubles_lr_at_quadruple_batch() -> None:
    """Square-root scaling multiplies lr by sqrt(4) == 2 at 4x the batch."""
    scaling = BatchLrScaling(reference_global_batch_size=_REFERENCE_BATCH, rule="sqrt")
    lr = scaled_learning_rate(
        reference_lr=_REFERENCE_LR,
        scaling=scaling,
        global_batch_size=_QUADRUPLE_BATCH,
    )
    assert math.isclose(lr, _REFERENCE_LR * _SQRT_QUADRUPLE_MULTIPLIER)


def test_linear_scaling_quadruples_lr_at_quadruple_batch() -> None:
    """Linear scaling multiplies lr by 4 at 4x the batch."""
    scaling = BatchLrScaling(
        reference_global_batch_size=_REFERENCE_BATCH,
        rule="linear",
    )
    lr = scaled_learning_rate(
        reference_lr=_REFERENCE_LR,
        scaling=scaling,
        global_batch_size=_QUADRUPLE_BATCH,
    )
    assert math.isclose(lr, _REFERENCE_LR * _LINEAR_QUADRUPLE_MULTIPLIER)


def test_nonpositive_global_batch_rejected() -> None:
    """A non-positive global batch is rejected rather than scaling to <= 0."""
    scaling = BatchLrScaling(reference_global_batch_size=_REFERENCE_BATCH)
    with pytest.raises(ValueError, match="global_batch_size must be positive"):
        scaled_learning_rate(
            reference_lr=_REFERENCE_LR,
            scaling=scaling,
            global_batch_size=0,
        )
