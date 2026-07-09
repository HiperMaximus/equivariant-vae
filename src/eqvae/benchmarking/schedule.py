# Copyright 2026 HiperMaximus
"""Single-sourced training-schedule derivations shared by the runtime generators.

Spec 0011 makes the full-run schedule a goal-derived relationship rather than a
frozen number: the optimizer updates per epoch are ``floor(P / G)`` for a training
patch count ``P`` and a global batch ``G``. Both the synthetic-timing projections and
the selected-runtime plan emitter derive that count, so it lives here once instead of
being copied (previously as ``ceil``) across four call sites.
"""

from __future__ import annotations


def training_steps_per_epoch(
    *,
    real_train_patch_count: int,
    global_batch_size: int,
) -> int:
    """Return the optimizer updates in one training epoch (Spec 0011).

    This is the ``floor`` of ``real_train_patch_count / global_batch_size``, matching
    a ``drop_last=True`` train loader (the loader is flipped to drop the trailing
    partial batch in Phase 3). Floor is a no-op whenever the batch divides the patch
    count evenly -- true for every batch used so far (24, 96 both divide 300000) -- so
    converting the former ``ceil`` sites is behavior-preserving at the reference
    batch; it only differs for a future non-dividing batch, where dropping the partial
    batch is the intended behavior.

    Returns:
        The floored number of full global batches per epoch.

    """
    return real_train_patch_count // global_batch_size


def boundary_steps(
    *,
    interval_steps: int,
    target_train_steps: int,
) -> tuple[int, ...]:
    """Return the ascending optimizer steps that are training boundaries (Spec 0011 S9).

    A boundary is a step at which the full-run loop writes an interval checkpoint, runs
    a scheduled validation pass, and becomes eligible for best-model selection. The grid
    is the interval cadence ``range(interval, target + 1, interval)`` unioned with the
    terminal ``target`` itself, so the last step is always a genuine boundary on both
    the PRODUCER side (checkpoint/validation written) and the CONSUMER side (checkpoint
    name expected, validation row required). Without ``| {target}`` an odd
    ``updates_per_epoch`` -- a global batch that does not divide the patch count --
    leaves ``target`` off the interval grid; the producers would silently drop the
    boundary while the consumers still passed, so the final model would never be
    validated, checkpointed, or best-selection-eligible (Spec 0011 MF3). Routing both
    sides through this one generator keeps them in lockstep.

    ``interval_steps`` is the caller's cadence: the scheduled-validation producer passes
    the half-epoch interval, the checkpoint producer passes ``save_every_steps``
    (equal to the half-epoch interval in a real full run, enforced by the runner
    validator). At the reference global batch 24 the terminal is already a multiple of
    the interval (``target=125000`` is ``20 * 6250``), so the union is a no-op and the
    boundary tuple is byte-identical to the former ``range``-based cadence -- the
    set-union dedups the on-grid terminal rather than duplicating it.

    Args:
        interval_steps: the boundary cadence (half-epoch or ``save_every_steps``).
        target_train_steps: the total optimizer updates ``epochs * updates_per_epoch``.

    Returns:
        The ascending tuple of boundary steps. Empty (fail-closed) when either argument
        is below ``1`` -- a degenerate ``range`` step would raise, and the callers that
        pass such values (the gate's ``valid=False`` sentinel, the runner's inactive
        half-epoch schedule) already treat "no boundaries" as their fail-closed state.

    """
    if interval_steps < 1 or target_train_steps < 1:
        return ()
    grid = set(range(interval_steps, target_train_steps + 1, interval_steps))
    grid.add(target_train_steps)
    return tuple(sorted(grid))


__all__ = ["boundary_steps", "training_steps_per_epoch"]
