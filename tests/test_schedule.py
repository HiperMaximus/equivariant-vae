# Copyright 2026 HiperMaximus
"""Tests for the single-sourced training-schedule derivations (Spec 0011)."""

from __future__ import annotations

from eqvae.benchmarking.schedule import training_steps_per_epoch

_REAL_TRAIN_PATCH_COUNT = 300_000
_REFERENCE_GLOBAL_BATCH = 24
_REFERENCE_STEPS_PER_EPOCH = 12_500
_QUADRUPLE_GLOBAL_BATCH = 96
_QUADRUPLE_STEPS_PER_EPOCH = 3_125
_NON_DIVIDING_GLOBAL_BATCH = 7
# floor(300000 / 7) == 42857 (300000 == 7 * 42857 + 1); the former ceil was 42858.
_NON_DIVIDING_FLOOR_STEPS = 42_857
_NON_DIVIDING_CEIL_STEPS = 42_858


def test_dividing_batch_matches_the_exact_quotient() -> None:
    """A batch that divides the patch count evenly yields the exact quotient.

    Both batches used so far (24, 96) divide 300000, so floor equals the old ceil and
    the conversion is behavior-preserving.
    """
    assert (
        training_steps_per_epoch(
            real_train_patch_count=_REAL_TRAIN_PATCH_COUNT,
            global_batch_size=_REFERENCE_GLOBAL_BATCH,
        )
        == _REFERENCE_STEPS_PER_EPOCH
    )
    assert (
        training_steps_per_epoch(
            real_train_patch_count=_REAL_TRAIN_PATCH_COUNT,
            global_batch_size=_QUADRUPLE_GLOBAL_BATCH,
        )
        == _QUADRUPLE_STEPS_PER_EPOCH
    )


def test_non_dividing_batch_floors_down_dropping_the_partial_batch() -> None:
    """A non-dividing batch drops the trailing partial batch (floor, not ceil)."""
    steps = training_steps_per_epoch(
        real_train_patch_count=_REAL_TRAIN_PATCH_COUNT,
        global_batch_size=_NON_DIVIDING_GLOBAL_BATCH,
    )

    assert steps == _NON_DIVIDING_FLOOR_STEPS
    assert steps != _NON_DIVIDING_CEIL_STEPS
