# Copyright 2026 HiperMaximus
"""CPU tests for the shared single-GPU VRAM batch-feasibility seam (Spec 0011 S14c).

The pure primitives -- the doubling ladder, the ceiling bisection, the margin
verdict, and the OOM classifier -- are exercised here. The one torch-touching
helper (`probe_headroom_bytes`, a `cuda.mem_get_info` read) runs only on GPU
(Kaggle), so it is CUDA-gated and skipped on CPU, mirroring the probe/executor
convention that the physical-VRAM core is a device-only observation.
"""

from __future__ import annotations

import pytest
import torch

from eqvae.benchmarking.vram_feasibility import (
    BYTES_PER_MB,
    CEILING_GRANULARITY,
    MAX_SWEEP_BATCH,
    NO_OOM,
    OOM,
    VRAM_MARGIN_MB,
    binary_search_feasible_ceiling,
    feasibility_ladder,
    headroom_below_margin,
    is_oom_error,
    probe_headroom_bytes,
)

_LADDER_BASE = 12
_LADDER_FIRST_DOUBLED = 48
_CEILING_LOW_OK = 48
_CEILING_HIGH_OOM = 384
_CEILING_TRUE_MAX = 200
_CEILING_MAX_PROBES = 10
_CEILING_TIGHT_MAX = 150
_CEILING_OOM_BOUND = 192


def test_feasibility_ladder_auto_extends_past_requested_until_the_cap() -> None:
    """The derived doubling ladder reaches the last power allowed by its cap.

    Reaching that boundary matters because stopping early can miss the real OOM edge.
    The assertions derive the expected termination from ``MAX_SWEEP_BATCH`` rather
    than freeze a ladder, and catch truncating the producer after a few rungs.
    """
    ladder = feasibility_ladder((12, 24), base_batch=_LADDER_BASE)
    # Requested sizes are preserved and de-duplicated, ascending.
    assert ladder[:2] == (12, 24)
    # It then doubles (48, 96, ...) so the sweep finds the OOM edge on its own.
    assert ladder[2] == _LADDER_FIRST_DOUBLED
    assert all(ladder[idx] == ladder[idx - 1] * 2 for idx in range(2, len(ladder)))
    assert ladder[-1] <= MAX_SWEEP_BATCH
    assert ladder[-1] * 2 > MAX_SWEEP_BATCH


def test_feasibility_ladder_dedupes_and_defaults_empty_request() -> None:
    """Duplicate/zero sizes collapse and an empty request falls back to the base."""
    assert feasibility_ladder((48, 24, 24, 0), base_batch=_LADDER_BASE)[:2] == (24, 48)
    assert feasibility_ladder((), base_batch=_LADDER_BASE)[0] == _LADDER_BASE


def test_feasibility_ladder_honors_a_custom_cap() -> None:
    """A smaller cap stops the doubling early (the cap is not hardcoded)."""
    ladder = feasibility_ladder((12,), base_batch=_LADDER_BASE, max_batch=48)
    assert ladder == (12, 24, 48)


def test_binary_search_ceiling_pins_largest_feasible_batch() -> None:
    """The bisection returns the largest feasible batch within the granularity."""
    probed: list[int] = []

    def feasible(batch_size: int) -> bool:
        probed.append(batch_size)
        return batch_size <= _CEILING_TRUE_MAX

    ceiling = binary_search_feasible_ceiling(
        feasible,
        low_ok=_CEILING_LOW_OK,
        high_oom=_CEILING_HIGH_OOM,
    )
    # Within granularity of the true 200-batch ceiling, never above it.
    assert ceiling <= _CEILING_TRUE_MAX
    assert _CEILING_TRUE_MAX - ceiling <= CEILING_GRANULARITY
    # A bisection touches O(log range) midpoints, not every candidate batch.
    assert len(probed) <= _CEILING_MAX_PROBES
    assert all(_CEILING_LOW_OK <= size <= _CEILING_HIGH_OOM for size in probed)


def test_binary_search_ceiling_never_probes_or_returns_the_oom_bound() -> None:
    """Feasibility is only ever probed strictly below the known-OOM upper bound."""
    probed: list[int] = []

    def feasible(batch_size: int) -> bool:
        probed.append(batch_size)
        return batch_size <= _CEILING_TIGHT_MAX

    ceiling = binary_search_feasible_ceiling(
        feasible,
        low_ok=96,
        high_oom=_CEILING_OOM_BOUND,
    )
    assert ceiling <= _CEILING_TIGHT_MAX
    # The known-OOM bound is never re-probed (no wasted OOM) and never returned.
    assert all(size < _CEILING_OOM_BOUND for size in probed)
    assert ceiling < _CEILING_OOM_BOUND


def test_headroom_below_margin_flags_only_readings_under_the_margin() -> None:
    """A reading under the byte margin is infeasible; at or above the margin is fine."""
    margin_bytes = int(VRAM_MARGIN_MB * BYTES_PER_MB)
    assert headroom_below_margin(margin_bytes - 1)
    assert not headroom_below_margin(margin_bytes)
    assert not headroom_below_margin(margin_bytes + 1)


def test_headroom_below_margin_honors_a_custom_margin() -> None:
    """The margin is a parameter, not a hardcoded threshold."""
    two_mb = int(2 * BYTES_PER_MB)
    # 1.5 MB of headroom clears a 1 MB margin but not a 2 MB margin.
    one_and_a_half_mb = int(1.5 * BYTES_PER_MB)
    assert not headroom_below_margin(one_and_a_half_mb, margin_mb=1.0)
    assert headroom_below_margin(one_and_a_half_mb, margin_mb=2.0)
    assert headroom_below_margin(two_mb - 1, margin_mb=2.0)


def test_is_oom_error_matches_every_cuda_vram_exhaustion_fragment() -> None:
    """OOM, cuBLAS/cuDNN alloc-failed, and "failed to allocate" all classify as OOM.

    A big batch can exhaust VRAM through the workspace allocator (cuBLAS/cuDNN), which
    does not say "out of memory"; treating those as OOM too keeps the two DDP ranks'
    verdicts symmetric at the VRAM boundary. A genuine non-memory error must NOT match.
    """
    assert is_oom_error(RuntimeError("CUDA out of memory. Tried to allocate ..."))
    assert is_oom_error(RuntimeError("OUT OF MEMORY"))
    assert is_oom_error(RuntimeError("CUBLAS_STATUS_ALLOC_FAILED when calling cublas"))
    assert is_oom_error(RuntimeError("cuDNN error: CUDNN_STATUS_ALLOC_FAILED"))
    assert is_oom_error(RuntimeError("RuntimeError: Failed to allocate workspace"))
    assert not is_oom_error(RuntimeError("device-side assert triggered"))
    assert not is_oom_error(RuntimeError("shape mismatch"))


def test_feasibility_flag_sentinels_are_reduce_summable() -> None:
    """NO_OOM sums to zero only when every rank is feasible; any OOM contributes 1."""
    assert NO_OOM == 0
    assert OOM == 1
    # Two feasible ranks -> reduced sum stays NO_OOM; one infeasible flips it positive.
    assert NO_OOM + NO_OOM == NO_OOM
    assert NO_OOM + OOM > NO_OOM


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="probe_headroom_bytes reads physical VRAM; GPU-only (Kaggle observation)",
)
def test_probe_headroom_bytes_is_a_nonnegative_reading() -> None:
    """On GPU the headroom read is a sane non-negative byte count."""
    headroom = probe_headroom_bytes(torch.device("cuda", 0))
    assert headroom >= 0
