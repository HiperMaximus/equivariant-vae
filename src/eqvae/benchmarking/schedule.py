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


__all__ = ["training_steps_per_epoch"]
