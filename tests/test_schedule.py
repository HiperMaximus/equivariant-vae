# Copyright 2026 HiperMaximus
"""Tests for the single-sourced training-schedule derivations (Spec 0011)."""

from __future__ import annotations

from eqvae.benchmarking.schedule import boundary_steps, training_steps_per_epoch

_REAL_TRAIN_PATCH_COUNT = 300_000
_REFERENCE_GLOBAL_BATCH = 24
_REFERENCE_STEPS_PER_EPOCH = 12_500
_QUADRUPLE_GLOBAL_BATCH = 96
_QUADRUPLE_STEPS_PER_EPOCH = 3_125
_NON_DIVIDING_GLOBAL_BATCH = 7
# floor(300000 / 7) == 42857 (300000 == 7 * 42857 + 1); the former ceil was 42858.
_NON_DIVIDING_FLOOR_STEPS = 42_857
_NON_DIVIDING_CEIL_STEPS = 42_858

# Reference full-run schedule at global batch 24: half=6250, target=10*12500=125000.
# 125000 == 20 * 6250, so the terminal is already on the half-epoch grid.
_REFERENCE_HALF_INTERVAL = 6_250
_REFERENCE_TARGET_UPDATES = 125_000
_REFERENCE_BOUNDARY_COUNT = 20
# Odd schedule (a non-dividing global batch): half=2, target=5. 5 is OFF the {2, 4}
# grid, so the terminal must be force-included by the union.
_ODD_HALF_INTERVAL = 2
_ODD_TARGET_UPDATES = 5
_ON_GRID_TARGET_UPDATES = 6


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


def test_boundary_steps_at_reference_batch_matches_the_former_range() -> None:
    """At global batch 24 the terminal is on-grid, so the union is byte-identical.

    The former producers/consumers open-coded ``range(half, target + 1, half)``; the
    shared generator must return exactly that tuple whenever the terminal already lands
    on the grid, with no duplicated terminal (the set-union dedups it).
    """
    result = boundary_steps(
        interval_steps=_REFERENCE_HALF_INTERVAL,
        target_train_steps=_REFERENCE_TARGET_UPDATES,
    )

    expected = tuple(
        range(
            _REFERENCE_HALF_INTERVAL,
            _REFERENCE_TARGET_UPDATES + 1,
            _REFERENCE_HALF_INTERVAL,
        ),
    )
    assert result == expected
    assert len(result) == _REFERENCE_BOUNDARY_COUNT
    assert result[-1] == _REFERENCE_TARGET_UPDATES
    assert result.count(_REFERENCE_TARGET_UPDATES) == 1


def test_boundary_steps_force_includes_an_off_grid_terminal() -> None:
    """An odd schedule leaves the terminal off the grid; it must still be a boundary."""
    assert boundary_steps(
        interval_steps=_ODD_HALF_INTERVAL,
        target_train_steps=_ODD_TARGET_UPDATES,
    ) == (2, 4, 5)


def test_boundary_steps_does_not_duplicate_an_on_grid_terminal() -> None:
    """A terminal already on the grid appears exactly once (set-union, not append)."""
    assert boundary_steps(
        interval_steps=_ODD_HALF_INTERVAL,
        target_train_steps=_ON_GRID_TARGET_UPDATES,
    ) == (2, 4, 6)


def test_boundary_steps_fails_closed_on_degenerate_inputs() -> None:
    """A sub-1 interval or target yields an empty grid, never a raising ``range``.

    The gate's ``valid=False`` sentinel (half=-1, target=-1) and the runner's inactive
    half-epoch schedule (half<=0) both feed such values; "no boundaries" is their
    fail-closed state.
    """
    assert (
        boundary_steps(interval_steps=0, target_train_steps=_ODD_TARGET_UPDATES) == ()
    )
    assert boundary_steps(interval_steps=_ODD_HALF_INTERVAL, target_train_steps=0) == ()
    assert boundary_steps(interval_steps=-1, target_train_steps=-1) == ()
