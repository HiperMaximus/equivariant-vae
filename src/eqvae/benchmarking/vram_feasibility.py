# Copyright 2026 HiperMaximus
"""Shared single-GPU VRAM batch-feasibility seam (Spec 0011 S14c).

Single source for the OOM-safe batch-feasibility primitives used by the compiled
fast-path probe (``benchmarking.compiled_fastpath_probe``, which *discovers* the
feasible batch ceiling of a recipe on synthetic tensors) and, as Spec 0011 S14c
wires it, the runtime-selection executor
(``benchmarking.runtime_selection_executor``, which *screens* each grid-enumerated
compiled efficiency batch for VRAM feasibility with this same absolute-margin rule
before paying for a real-data dual-T4 DDP timing run). Both applying the identical
rule is what lets an infeasible batch be recorded as a clean ``oom`` verdict rather
than surfacing as a hard benchmark failure.

Feasibility is always decided on a SINGLE-GPU (no-DDP) replica: with no DDP wrapper
the OOM-prone forward/backward (and the inductor compile + cuDNN autotune spike it
drives) issues NO collective, so a *classified* VRAM-exhaustion event on one rank (see
:func:`is_oom_error`, which spans "out of memory" AND cuBLAS/cuDNN alloc failures) is
turned into a flag the caller reduces -- the only cross-rank op -- and cannot desync
the peer regardless of where the failure lands. A truly-unexpected (non-VRAM) error is
re-raised instead, and the caller is responsible for bounding the resulting one-sided
exit (the executor relies on a subprocess timeout). A batch is feasible only if
the single-GPU probe neither OOMs nor leaves less than ``VRAM_MARGIN_MB`` of PHYSICAL
free VRAM (read via ``cuda.mem_get_info``, which -- unlike ``max_memory_reserved`` --
already accounts for the CUDA context, cuDNN/Triton modules, and NCCL buffers). That
margin is the extra footprint the real DDP timed run adds on top of the single-GPU
probe (gradient buckets, comm buffers, the compiled split graph, fragmentation), so a
batch that clears the margin single-GPU is guaranteed to fit the DDP run without an
asymmetric mid-stream OOM.

The helpers take plain scalars (a device, a ``feasible`` callable, batch ints) rather
than a probe- or executor-specific spec object, so either caller composes them with
its own model-build machinery without depending on the other's types. The physical
free-VRAM query is the only torch-touching helper; the ladder, the ceiling bisection,
the margin verdict, and the OOM classifier are pure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable

# Byte/megabyte conversion for the VRAM headroom margin and peak-VRAM reporting.
BYTES_PER_MB = 1.0e6
# The substring the canonical CUDA OOM ``RuntimeError`` message carries; used to tell a
# VRAM-pressure event apart from an unrelated ``RuntimeError`` that must re-raise.
OOM_MESSAGE_FRAGMENT = "out of memory"
# A big batch can also exhaust VRAM through the cuBLAS / cuDNN workspace allocator,
# which raises ``*_STATUS_ALLOC_FAILED`` (or "failed to allocate") rather than "out of
# memory". These are the same feasibility signal, so the classifier treats them as OOM
# too -- and, since a probe on the peer rank might trip the "out of memory" path
# instead, matching both keeps the two ranks' verdicts symmetric at the VRAM boundary.
_EXTRA_ALLOC_FAILURE_FRAGMENTS = ("alloc_failed", "failed to allocate")
# Reduced-flag sentinels for the cross-rank feasibility agreement: ``NO_OOM`` (0) sums
# to 0 only when EVERY rank was feasible; any infeasible rank contributes ``OOM`` (1).
NO_OOM = 0
OOM = 1
# Physical free VRAM (MB) a single-GPU probe must leave so the heavier DDP timed run
# still fits. Sized as the DDP-only footprint (buckets, NCCL buffers, split graph).
VRAM_MARGIN_MB = 1024.0
# Stop the ceiling bisection once the feasible/OOM bracket is this wide; a batch delta
# below this is not worth another expensive compile+autotune probe.
CEILING_GRANULARITY = 4
# Hard safety cap on the doubling ladder so a recipe that never OOMs still terminates.
MAX_SWEEP_BATCH = 512


def feasibility_ladder(
    batch_sizes: tuple[int, ...],
    *,
    base_batch: int,
    max_batch: int = MAX_SWEEP_BATCH,
) -> tuple[int, ...]:
    """Return the ascending doubling ladder the sweep walks to find the OOM edge.

    The caller's requested sizes seed the curve; the ladder then keeps doubling past
    the largest requested size (up to ``max_batch``) so the sweep brackets the memory
    ceiling on its own instead of enumerating every size by hand. Falls back to a
    single ``base_batch`` rung when no positive size is requested.

    Args:
        batch_sizes: the caller-requested per-device batch sizes (non-positive dropped).
        base_batch: the sole rung when ``batch_sizes`` has no positive entry.
        max_batch: the inclusive doubling cap that guarantees termination.

    Returns:
        The ascending, de-duplicated ladder of candidate batch sizes.

    """
    requested = sorted({size for size in batch_sizes if size > 0})
    ladder = requested or [base_batch]
    nxt = ladder[-1] * 2
    while nxt <= max_batch:
        ladder.append(nxt)
        nxt *= 2
    return tuple(ladder)


def binary_search_feasible_ceiling(
    feasible: Callable[[int], bool],
    *,
    low_ok: int,
    high_oom: int,
    granularity: int = CEILING_GRANULARITY,
) -> int:
    """Return the largest batch known feasible in ``[low_ok, high_oom)``.

    Invariant on entry: ``feasible(low_ok)`` is True and ``feasible(high_oom)`` is
    False. ``feasible`` MUST return a value both ranks agree on (a cross-rank-reduced
    OOM flag), so both ranks probe the identical midpoint sequence and never diverge
    into mismatched collectives.

    Args:
        feasible: the cross-rank-agreed single-GPU feasibility predicate.
        low_ok: a batch size already proven feasible.
        high_oom: a batch size already proven infeasible.
        granularity: stop once the bracket narrows to at most this width.

    Returns:
        The largest batch size proven feasible.

    """
    while high_oom - low_ok > granularity:
        mid = (low_ok + high_oom) // 2
        if feasible(mid):
            low_ok = mid
        else:
            high_oom = mid
    return low_ok


def probe_headroom_bytes(device: torch.device) -> int:
    """Return the physical free VRAM (bytes) left at the single-GPU probe's peak.

    Take the min of two bounds so it stays safe under either allocator:

    - ``free``: exact when the allocator holds its peak reserved segments at read time
      (``expandable_segments`` OFF, the default). It already subtracts the CUDA context
      and the cuDNN / Triton / NCCL resident set that ``max_memory_reserved`` misses.
    - ``total - reserved``: an allocator-independent bound valid even if
      ``expandable_segments`` returned segments to the OS between the activation peak
      and this read (so ``free`` alone could over-read the steady footprint).

    Returns:
        The smaller of the two free-VRAM bounds, in bytes.

    """
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    return min(free_bytes, total_bytes - peak_reserved)


def headroom_below_margin(
    headroom_bytes: int,
    *,
    margin_mb: float = VRAM_MARGIN_MB,
) -> bool:
    """Return True when the probe's free VRAM is below the safety margin (infeasible).

    Args:
        headroom_bytes: the physical free-VRAM bytes from :func:`probe_headroom_bytes`.
        margin_mb: the required free-VRAM margin, in megabytes.

    Returns:
        True when ``headroom_bytes`` is under the margin, marking the batch infeasible
        even though the probe did not raise an out-of-memory error.

    """
    return headroom_bytes < int(margin_mb * BYTES_PER_MB)


def is_oom_error(error: RuntimeError) -> bool:
    """Return True when ``error`` signals CUDA VRAM exhaustion.

    ``torch.cuda.OutOfMemoryError`` subclasses ``RuntimeError``, but VRAM exhaustion can
    also surface as a bare ``RuntimeError`` -- the canonical "out of memory", or a
    cuBLAS / cuDNN workspace ``*_STATUS_ALLOC_FAILED`` / "failed to allocate". All are
    the same feasibility signal (the batch does not fit), so callers treat any as OOM
    and re-raise anything else (a genuine bug, not a memory limit).

    Returns:
        True when the error message contains a CUDA VRAM-exhaustion fragment.

    """
    message = str(error).lower()
    return OOM_MESSAGE_FRAGMENT in message or any(
        fragment in message for fragment in _EXTRA_ALLOC_FAILURE_FRAGMENTS
    )


__all__ = [
    "BYTES_PER_MB",
    "CEILING_GRANULARITY",
    "MAX_SWEEP_BATCH",
    "NO_OOM",
    "OOM",
    "OOM_MESSAGE_FRAGMENT",
    "VRAM_MARGIN_MB",
    "binary_search_feasible_ceiling",
    "feasibility_ladder",
    "headroom_below_margin",
    "is_oom_error",
    "probe_headroom_bytes",
]
