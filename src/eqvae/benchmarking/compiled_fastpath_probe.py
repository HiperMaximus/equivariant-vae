# Copyright 2026 HiperMaximus
"""Synthetic dual-T4 bake-off for the compiled FSQ-style fast path.

`run_compiled_fastpath_probe` measures an eager-DDP-with-sync baseline plus THREE
compiled recipes on synthetic in-memory tensors with no dataset attached, then
picks the fastest recipe by throughput. Every config loads one shared
``reference_state`` into a fresh model, corrupts a synthetic batch with
``InlineStainCorruptor``, and runs the VAE train step built by
``make_fastpath_step_fn``. The probe proves each compiled recipe is compile-stable
(zero post-warmup graph breaks / recompiles), DDP-syncing (a positive cross-rank
parameter-sync check per config plus one shared negative control that must fire),
and finite, and records how fast each recipe is against the eager baseline. It runs
under ``torchrun --standalone --nproc_per_node=2`` on GPU and writes NON-PROMOTABLE
proof/matrix/manifest artifacts (``full_run_eligible`` is false, no dataset sources,
every real-run claim blocked).

The four configs are:

1. ``eager_ddp_sync`` -- fast-path OFF reference (contiguous layout, the eager C++
   reducer, default AdamW, no ``torch.compile``); it still syncs, so it also gets a
   positive sync check.
2. ``python_reducer_whole_step`` -- the expected winner: ``optimize_ddp`` set to the
   Python reducer with compiled autograd and compute/comm overlap reordering, a
   ``channels_last`` fused-AdamW DDP model, and the whole step compiled.
3. ``ddp_optimizer_whole_step`` -- the conservative reference: the C++ reducer with
   Dynamo bucket-boundary splits, whole step compiled.
4. ``ddp_compile_model`` -- the VRAM/warmup fallback: the inner model is compiled and
   then wrapped in DDP while the optimizer step stays eager.

The winner is the compiled recipe with the highest ``samples_sec`` among those that
pass sync + post-warmup stability + zero non-finite losses.

After the single-batch bake-off, an OOM-safe BATCH SWEEP re-measures the eager baseline
plus the two fastest recipes (``ddp_optimizer_whole_step`` and ``ddp_compile_model``;
``python_reducer_whole_step`` is skipped as the slowest) to find each swept recipe's
throughput-optimal batch and its largest feasible batch on the 16 GB T4. A fully timed
doubling ladder draws the throughput curve and brackets the memory ceiling; a binary
search then pins it.

Deadlock-safety is the crux, because the search deliberately probes the memory boundary
where an OOM can land on one rank but not the other. Feasibility is decided on a
SINGLE-GPU (no-DDP) replica: with no DDP wrapper, the OOM-prone forward/backward (and
the inductor compile + cuDNN autotune spike) issues NO collective at all, so an
asymmetric OOM cannot desync anything -- the only cross-rank op is one ``all_reduce`` of
the per-point feasibility flag, matched on both ranks regardless of where the OOM lands.
This holds for ANY recipe, so all three are safe to sweep no matter how their DDP
all_reduce is issued. A batch is feasible only if the probe neither OOMs nor leaves less
than a margin of PHYSICAL free VRAM (read via ``cuda.mem_get_info``, which -- unlike
``max_memory_reserved`` -- already accounts for the CUDA context, cuDNN/Triton modules,
and NCCL buffers). The margin is the extra footprint the DDP timed run adds (buckets,
comm buffers, split graph, fragmentation). Only a batch both ranks agree is feasible
gets a fresh DDP build for the real grad-syncing timed measurement -- built from the
shared reference_state so the replicas start bit-identical (no resync), and guaranteed
by the free-VRAM margin not to OOM, so its ``all_reduce`` stream stays matched. An OOM
or an under-margin batch is recorded as data (``oom=True``), never a failure.

The pure payload builders and the negative-control helper are import-safe and
CPU-testable; the GPU/NCCL core is skipped on CPU.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch
import torch._dynamo as torch_dynamo  # noqa: PLC2701
import torch.distributed as dist
from torch import Tensor, nn
from torch._dynamo import compiled_autograd  # noqa: PLC2701
from torch._dynamo.utils import counters  # noqa: PLC2701
from torch._inductor import config as inductor_config  # noqa: PLC2701
from torch.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel

from eqvae.benchmarking.io import write_csv, write_json
from eqvae.corruption.inline_stain import InlineStainCorruptor
from eqvae.corruption.stain import CONSERVATIVE_DEFAULT_PROFILE, profile_from_name
from eqvae.models.non_equivariant_vae import build_non_equivariant_vae
from eqvae.training.ddp_sync_guard import assert_ddp_parameters_in_sync
from eqvae.training.fastpath_step import make_fastpath_step_fn
from eqvae.training.optim import SpecAdamWConfig, create_adamw_optimizer

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from contextlib import AbstractContextManager

    from eqvae.benchmarking.io import CsvRow, JsonObject
    from eqvae.models.non_equivariant_vae import NonEquivariantVAE
    from eqvae.training.fastpath_step import FastpathStepOutput


COMPILED_FASTPATH_PROBE_KIND = "kaggle_compiled_fastpath_probe"
COMPILED_FASTPATH_PROBE_SOURCE = "kaggle_no_dataset_synthetic_compiled_fastpath"
COMPILED_FASTPATH_PROBE_STATUS_SCOPE = "non_promotable_compiled_fastpath_probe"
COMPILED_FASTPATH_PROBE_SCHEMA_VERSION = "spec0001.compiled_fastpath_probe.v3"
PROBE_STATUS_PASS = "compiled_fastpath_probe_pass"  # noqa: S105
PROBE_STATUS_FAIL = "compiled_fastpath_probe_fail"

EAGER_BASELINE_NAME = "eager_ddp_sync"
RECIPE_PYTHON_REDUCER = "python_reducer_whole_step"
RECIPE_DDP_OPTIMIZER = "ddp_optimizer_whole_step"
RECIPE_DDP_COMPILE_MODEL = "ddp_compile_model"

PROOF_FILENAME = "compiled_fastpath_probe_proof.json"
MATRIX_FILENAME = "compiled_fastpath_probe_matrix.csv"
MANIFEST_FILENAME = "compiled_fastpath_probe_manifest.json"
BENCHMARK_DIRNAME = "benchmark"

BLOCKED_CLAIM_KEYS = (
    "runtime_selection",
    "full_run_readiness",
    "real_data_throughput",
    "convergence",
    "paper_evidence",
    "final_speedup_on_real_data",
)

COMPILED_FASTPATH_PROBE_MATRIX_COLUMNS = (
    "run_name",
    "benchmark_kind",
    "benchmark_source",
    "status_scope",
    "status",
    "full_run_eligible",
    "world_size",
    "nproc_per_node",
    "per_device_batch_size",
    "phase",
    "batch_size",
    "recipe_name",
    "compiled",
    "compile_scope",
    "is_winner",
    "syncs",
    "graph_break_count",
    "recompile_count",
    "stable",
    "step_ms_p50",
    "samples_sec",
    "peak_vram_mb",
    "speedup",
    "nonfinite_loss_count",
    "oom",
    "negative_control_fired",
    "warmup_steps",
    "settle_steps",
    "measured_steps",
)

_COMPILE_SCOPE_NONE = "none"
_COMPILE_SCOPE_STEP = "step"
_COMPILE_SCOPE_MODEL = "model"
_COMPILE_BACKEND = "inductor"

_OPTIMIZE_DDP_DEFAULT = "ddp_optimizer"
_OPTIMIZE_DDP_PYTHON_REDUCER = "python_reducer"

_IMAGE_SIZE = 256
_IMAGE_CHANNELS = 3
_DEFAULT_BATCH_SIZE = 12
_DEFAULT_WARMUP_STEPS = 5
_DEFAULT_SETTLE_STEPS = 5
_DEFAULT_MEASURED_STEPS = 20
_DEFAULT_SYNC_CHECK_STEPS = 3
_DEFAULT_SEED = 20260703
_MAX_GRAD_NORM = 1.0
_SSIM_WEIGHT = 0.1
# Match the selected-runtime runner's conservative AMP GradScaler start scale so
# the eager arm mirrors the runtime the probe is compared against.
_GRAD_SCALER_INIT_SCALE = 16384.0
_RECIPE_BUCKET_CAP_MB = 50
_PRIMARY_RANK = 0
_DESYNC_RANK = 0
_MS_PER_SECOND = 1000.0
_BYTES_PER_MB = 1.0e6
_STABLE_DELTA = 0
_MIN_SYNC_CHECK_STEPS = 1
_SIGNALS_PER_CONFIG = 3
_SWEEP_PHASE_BAKEOFF = "bakeoff"
_SWEEP_PHASE_SWEEP = "sweep"
# torch raises either ``torch.cuda.OutOfMemoryError`` or a plain ``RuntimeError`` whose
# message carries this fragment; the sweep treats both as an out-of-memory point.
_OOM_MESSAGE_FRAGMENT = "out of memory"
_NO_OOM = 0
# Batch-sweep ceiling search. The timed doubling ladder starts at the base and keeps
# doubling (up to the safety cap) until a config is infeasible, bracketing the ceiling;
# a feasibility bisection then pins it to within the granularity. The cap only backstops
# a bug -- these recipes hit the VRAM budget far below it on a 16 GB T4.
_SWEEP_BASE_BATCH = _DEFAULT_BATCH_SIZE
_SWEEP_MAX_BATCH = 512
_SWEEP_CEILING_GRANULARITY = 4
# A batch is "feasible" only if the single-GPU probe leaves at least this much PHYSICAL
# free VRAM (via cuda.mem_get_info, which already accounts for the CUDA context + cuDNN/
# Triton modules + NCCL buffers that max_memory_reserved does NOT). The margin is the
# extra footprint the real DDP timed run adds on top of the probe -- gradient buckets,
# NCCL comm buffers, the DDP-split graph, and allocator fragmentation -- so a batch that
# passes cannot OOM when re-run under DDP (which would risk an asymmetric mid-grad-sync
# OOM -> NCCL hang). Probe reserved is held (allocator does not shrink) when free is
# read, so the reading is the true headroom at the probe's peak.
_SWEEP_VRAM_MARGIN_MB = 1024.0
# The feasibility probe must actually run a step (allocate activations + optimizer state
# + compile scratch) or its free-VRAM reading is meaningless; never let a low
# --warmup-steps skip that.
_SWEEP_PROBE_MIN_WARMUP_STEPS = 2


@dataclass(frozen=True)
class _RecipeSpec:
    """Static configuration for one bake-off config (eager or a compiled recipe)."""

    name: str
    compiled: bool
    compile_scope: str
    channels_last: bool
    fused_optimizer: bool
    optimize_ddp: bool | str
    compiled_autograd: bool
    reorder_compute_comm_overlap: bool
    ddp_static_graph: bool
    ddp_gradient_as_bucket_view: bool
    ddp_broadcast_buffers: bool
    ddp_bucket_cap_mb: int | None


_EAGER_SPEC = _RecipeSpec(
    name=EAGER_BASELINE_NAME,
    compiled=False,
    compile_scope=_COMPILE_SCOPE_NONE,
    channels_last=False,
    fused_optimizer=False,
    optimize_ddp=True,
    compiled_autograd=False,
    reorder_compute_comm_overlap=False,
    ddp_static_graph=False,
    ddp_gradient_as_bucket_view=False,
    ddp_broadcast_buffers=True,
    ddp_bucket_cap_mb=None,
)

_PYTHON_REDUCER_SPEC = _RecipeSpec(
    name=RECIPE_PYTHON_REDUCER,
    compiled=True,
    compile_scope=_COMPILE_SCOPE_STEP,
    channels_last=True,
    fused_optimizer=True,
    optimize_ddp=_OPTIMIZE_DDP_PYTHON_REDUCER,
    compiled_autograd=True,
    reorder_compute_comm_overlap=True,
    ddp_static_graph=False,
    ddp_gradient_as_bucket_view=True,
    ddp_broadcast_buffers=False,
    ddp_bucket_cap_mb=_RECIPE_BUCKET_CAP_MB,
)

_DDP_OPTIMIZER_SPEC = _RecipeSpec(
    name=RECIPE_DDP_OPTIMIZER,
    compiled=True,
    compile_scope=_COMPILE_SCOPE_STEP,
    channels_last=True,
    fused_optimizer=True,
    optimize_ddp=_OPTIMIZE_DDP_DEFAULT,
    compiled_autograd=False,
    reorder_compute_comm_overlap=False,
    ddp_static_graph=False,
    ddp_gradient_as_bucket_view=True,
    ddp_broadcast_buffers=False,
    ddp_bucket_cap_mb=_RECIPE_BUCKET_CAP_MB,
)

_DDP_COMPILE_MODEL_SPEC = _RecipeSpec(
    name=RECIPE_DDP_COMPILE_MODEL,
    compiled=True,
    compile_scope=_COMPILE_SCOPE_MODEL,
    channels_last=True,
    fused_optimizer=True,
    optimize_ddp=_OPTIMIZE_DDP_PYTHON_REDUCER,
    compiled_autograd=True,
    reorder_compute_comm_overlap=True,
    ddp_static_graph=False,
    ddp_gradient_as_bucket_view=True,
    ddp_broadcast_buffers=False,
    ddp_bucket_cap_mb=_RECIPE_BUCKET_CAP_MB,
)

# Order matters: python_reducer's DDP construction mutates process-global dynamo
# state (LEGACY_MOD_INLINELIST, inductor _fuse_ddp_communication) that
# torch._dynamo.reset() does NOT clear, which would disturb DDPOptimizer. Measure
# ddp_optimizer FIRST so its numbers are taken on a clean global state.
_RECIPE_SPECS = (_DDP_OPTIMIZER_SPEC, _PYTHON_REDUCER_SPEC, _DDP_COMPILE_MODEL_SPEC)

# The batch sweep re-measures the eager baseline plus the two fastest recipes (the
# winner ddp_optimizer and the runner-up ddp_compile_model); python_reducer is skipped
# (slowest in v2, close behind ddp_optimizer). Feasibility is probed on a single-GPU
# (no-DDP) replica, so this set is NOT constrained by which recipes let no_sync()
# suppress their all_reduce -- the probe issues no collective regardless of the recipe's
# compile/DDP config, so every recipe is safe to sweep to its VRAM ceiling. Order still
# matters: ddp_optimizer is measured before ddp_compile_model because the latter's
# python-reducer construction mutates process-global dynamo state that reset() leaves.
_SWEEP_SPECS = (_EAGER_SPEC, _DDP_OPTIMIZER_SPEC, _DDP_COMPILE_MODEL_SPEC)


@dataclass(frozen=True)
class CompiledFastpathProbeRequest:
    """Inputs for one dual-T4 compiled fast-path bake-off run."""

    output_dir: Path
    run_name: str = "eqvae_compiled_fastpath_probe"
    per_device_batch_size: int = _DEFAULT_BATCH_SIZE
    batch_sizes: tuple[int, ...] = (_DEFAULT_BATCH_SIZE,)
    warmup_steps: int = _DEFAULT_WARMUP_STEPS
    settle_steps: int = _DEFAULT_SETTLE_STEPS
    measured_steps: int = _DEFAULT_MEASURED_STEPS
    sync_check_steps: int = _DEFAULT_SYNC_CHECK_STEPS
    seed: int = _DEFAULT_SEED


@dataclass(frozen=True)
class CompiledFastpathProbeEnvironment:
    """Resolved distributed environment recorded in the probe artifacts."""

    world_size: int
    nproc_per_node: int
    gpu_names: tuple[str, ...]
    torch_version: str


@dataclass(frozen=True)
class RecipeResult:
    """One config's measured outcome (eager baseline or a compiled recipe)."""

    name: str
    compiled: bool
    compile_scope: str
    syncs: bool
    graph_break_count: int
    recompile_count: int
    step_ms_p50: float
    samples_sec: float
    peak_vram_mb: float
    nonfinite_loss_count: int
    speedup: float

    @property
    def stable(self) -> bool:
        """Return whether the recipe settled with no post-warmup graph churn.

        Returns:
            ``True`` if no graph break or recompile occurred in the settle window.

        """
        return (
            self.graph_break_count == _STABLE_DELTA
            and self.recompile_count == _STABLE_DELTA
        )

    @property
    def passed(self) -> bool:
        """Return whether the compiled recipe is a valid winner candidate.

        Speed is intentionally excluded here: `samples_sec` ranks the passing
        recipes elsewhere, but per-rank timing noise must never flip the
        recipe-correctness verdict. The eager baseline is never a candidate.

        Returns:
            ``True`` if the recipe compiled, kept its ranks in sync, settled
            without graph breaks or recompiles, and produced only finite losses.

        """
        return (
            self.compiled
            and self.syncs
            and self.stable
            and self.nonfinite_loss_count == _STABLE_DELTA
        )


@dataclass(frozen=True)
class CompiledFastpathProbeMeasurement:
    """Measured bake-off outcomes across the eager baseline and compiled recipes."""

    eager: RecipeResult
    recipes: tuple[RecipeResult, ...]
    negative_control_fired: bool
    sync_check_steps: int
    sweep_points: tuple[_SweepPoint, ...] = ()

    @property
    def winner(self) -> RecipeResult | None:
        """Return the fastest recipe that passed sync, stability, and finiteness.

        Returns:
            The highest-throughput passing recipe, or ``None`` if none passed.

        """
        candidates = [recipe for recipe in self.recipes if recipe.passed]
        if not candidates:
            return None
        return max(candidates, key=_recipe_samples_sec)

    @property
    def passed(self) -> bool:
        """Return whether the bake-off earned a pass verdict.

        A run that checked zero sync steps cannot pass: an unverified sync must
        never read as a proof, even if the negative control fired.

        Returns:
            ``True`` if at least one sync step was checked, the negative control
            fired, and at least one recipe passed the sync, stability, and
            finiteness checks (so a winner exists).

        """
        return (
            self.sync_check_steps >= _MIN_SYNC_CHECK_STEPS
            and self.negative_control_fired
            and self.winner is not None
        )


@dataclass(frozen=True)
class CompiledFastpathProbeArtifacts:
    """Paths written by the compiled fast-path probe."""

    proof: Path
    matrix: Path
    manifest: Path


@dataclass(frozen=True)
class _DistributedContext:
    device: torch.device
    rank: int
    local_rank: int
    world_size: int
    nproc_per_node: int


@dataclass(frozen=True)
class _StepContext:
    step_fn: Callable[[Tensor, Tensor, Tensor], FastpathStepOutput]
    x_clean: Tensor
    latent_shape: torch.Size
    beta: Tensor
    optimizer: torch.optim.Optimizer
    scaler: GradScaler
    model: nn.Module
    eps_generator: torch.Generator
    compiled_autograd: bool
    device: torch.device


@dataclass(frozen=True)
class _ConfigMeasurement:
    name: str
    compiled: bool
    compile_scope: str
    syncs: bool
    graph_break_count: int
    recompile_count: int
    step_ms_p50: float
    samples_sec: float
    peak_vram_mb: float
    nonfinite_loss_count: int


@dataclass(frozen=True)
class _SweepPoint:
    """One ``(config, batch_size)`` measurement from the OOM-safe batch sweep.

    An ``oom`` point carries zeroed measurements: both ranks agreed (via a reduced
    per-point OOM flag) that at least one of them ran out of memory, so neither ran
    the point's timing/sync collectives.
    """

    name: str
    batch_size: int
    samples_sec: float
    step_ms_p50: float
    peak_vram_mb: float
    syncs: bool
    stable: bool
    nonfinite: int
    oom: bool


@dataclass(frozen=True)
class _SweepMeasurement:
    """Per-point sweep timing/stability gathered before the cross-rank sync check."""

    step_ms_p50: float
    peak_vram_mb: float
    stable: bool
    nonfinite: int


@dataclass(frozen=True)
class _MatrixContext:
    request: CompiledFastpathProbeRequest
    environment: CompiledFastpathProbeEnvironment
    status: str
    negative_control_fired: bool


class _TwoParameterModule(nn.Module):
    """Throwaway module with exactly two scalar parameters for the desync control."""

    def __init__(self) -> None:
        """Initialize both scalar parameters to zero for an in-sync starting point."""
        super().__init__()
        self.first = nn.Parameter(torch.zeros(()))
        self.second = nn.Parameter(torch.zeros(()))


def graph_break_total() -> int:
    """Return the process-wide TorchDynamo graph-break count.

    Returns:
        Total graph breaks recorded across all break reasons.

    """
    breaks = cast("dict[str, int]", counters["graph_break"])
    return int(sum(breaks.values()))


def unique_graph_count() -> int:
    """Return the process-wide count of distinct compiled graphs.

    Returns:
        The TorchDynamo ``unique_graphs`` stat, which increments on each recompile.

    """
    stats = cast("dict[str, int]", counters["stats"])
    return int(stats.get("unique_graphs", 0))


def run_negative_control_desync(
    *,
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    """Desync a throwaway two-parameter module on rank 0 and re-run the sync guard.

    Builds a zero-initialized two-parameter module identically on every rank, then
    adds a constant to both parameters on rank 0 only, so the ranks now hold
    divergent parameters. `assert_ddp_parameters_in_sync` must then raise on every
    rank: a grad-sync guard that never fires on a genuine desync would give false
    confidence that the compiled fast path keeps ranks in sync. The raise is
    delegated to that guard.
    """
    module = _TwoParameterModule().to(device=device)
    if rank == _DESYNC_RANK:
        with torch.no_grad():
            for parameter in module.parameters():
                parameter.add_(1.0)
    assert_ddp_parameters_in_sync(module, world_size=world_size)


def run_compiled_fastpath_probe(
    request: CompiledFastpathProbeRequest,
) -> CompiledFastpathProbeMeasurement:
    """Run the full compiled fast-path bake-off on the local rank and write artifacts.

    Measures the eager-DDP-with-sync baseline plus three compiled recipes, checks
    each config's cross-rank sync, runs one shared negative control, and then runs an
    OOM-safe batch sweep for a focused config set: a fully timed doubling ladder
    (throughput curve) that auto-extends past ``request.batch_sizes`` until it OOMs,
    then a cheap-probe binary search that pins each compiled recipe's exact ceiling. The
    compile/finiteness counters are reduced across ranks so the verdict and the
    rank-0-written non-promotable proof reflect both T4s; every rank then fails
    closed together if the write failed, the negative control did not fire, any
    config that should sync did not, or any loss was non-finite. Sweep OOM is data,
    not a failure, so it never raises.

    Returns:
        The measured bake-off outcomes, including the chosen winner and sweep points.

    Raises:
        RuntimeError: If rank 0 could not write the artifacts, or if any hard
            correctness invariant (negative control, sync, finiteness) failed; the
            signals are reduced across ranks so every rank raises identically.

    """
    distributed = _init_distributed()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    manual_seed = cast("Callable[[int], torch.Generator]", torch.manual_seed)
    manual_seed(request.seed)
    reference_state = build_non_equivariant_vae().state_dict()
    configs = [
        _measure_config(spec, reference_state, request=request, distributed=distributed)
        for spec in (_EAGER_SPEC, *_RECIPE_SPECS)
    ]
    sweep_points = _run_batch_sweep(
        reference_state,
        request=request,
        distributed=distributed,
    )
    negative_control_fired = _negative_control_fired(
        rank=distributed.rank,
        world_size=distributed.world_size,
        device=distributed.device,
    )
    measurement = _assemble_measurement(
        _reduce_config_signals(configs, device=distributed.device),
        negative_control_fired=negative_control_fired,
        request=request,
        sweep_points=sweep_points,
    )
    environment = CompiledFastpathProbeEnvironment(
        world_size=distributed.world_size,
        nproc_per_node=distributed.nproc_per_node,
        gpu_names=_gpu_names(),
        torch_version=str(torch.__version__),
    )
    write_failed = _write_probe_artifacts_failclosed(
        rank=distributed.rank,
        request=request,
        measurement=measurement,
        environment=environment,
        device=distributed.device,
    )
    barrier = cast("Callable[[], object]", dist.barrier)
    barrier()
    if write_failed:
        message = "rank 0 failed to write compiled fast-path probe artifacts"
        raise RuntimeError(message)
    if not _correctness_held(measurement):
        raise RuntimeError(_failure_message(measurement))
    return measurement


def build_compiled_fastpath_probe_proof(
    *,
    request: CompiledFastpathProbeRequest,
    measurement: CompiledFastpathProbeMeasurement,
    environment: CompiledFastpathProbeEnvironment,
) -> JsonObject:
    """Build the non-promotable proof payload for the compiled fast-path bake-off.

    Returns:
        The proof JSON object.

    """
    return cast(
        "JsonObject",
        {
            **_non_promotable_header(request=request, measurement=measurement),
            "world_size": environment.world_size,
            "nproc_per_node": environment.nproc_per_node,
            "gpu_names": list(environment.gpu_names),
            "torch_version": environment.torch_version,
            "per_device_batch_size": request.per_device_batch_size,
            "warmup_steps": request.warmup_steps,
            "settle_steps": request.settle_steps,
            "measured_steps": request.measured_steps,
            "autocast_dtype": "float16",
            "negative_control_fired": measurement.negative_control_fired,
            "grad_sync": {
                "checked_steps": measurement.sync_check_steps,
                "negative_control_fired": measurement.negative_control_fired,
            },
            "eager_baseline": _eager_baseline_payload(measurement.eager),
            "recipes": [_recipe_payload(recipe) for recipe in measurement.recipes],
            "winner": _winner_payload(measurement.winner),
            "batch_sweep": _batch_sweep_payload(
                batch_sizes=request.batch_sizes,
                sweep_points=measurement.sweep_points,
            ),
        },
    )


def build_compiled_fastpath_probe_matrix_rows(
    *,
    request: CompiledFastpathProbeRequest,
    measurement: CompiledFastpathProbeMeasurement,
    environment: CompiledFastpathProbeEnvironment,
) -> list[CsvRow]:
    """Build the bake-off rows (eager + three recipes) then the batch-sweep rows.

    Returns:
        The four bake-off rows (``phase=bakeoff``) followed by one ``phase=sweep``
        row per measured ``(config, batch_size)`` point.

    """
    winner = measurement.winner
    winner_name = winner.name if winner is not None else ""
    context = _MatrixContext(
        request=request,
        environment=environment,
        status=_status(measurement),
        negative_control_fired=measurement.negative_control_fired,
    )
    rows: list[CsvRow] = [
        _matrix_row(measurement.eager, context=context, is_winner=False),
    ]
    rows.extend(
        _matrix_row(recipe, context=context, is_winner=recipe.name == winner_name)
        for recipe in measurement.recipes
    )
    rows.extend(
        _sweep_matrix_row(point, context=context) for point in measurement.sweep_points
    )
    return rows


def build_compiled_fastpath_probe_manifest(
    *,
    request: CompiledFastpathProbeRequest,
    measurement: CompiledFastpathProbeMeasurement,
    environment: CompiledFastpathProbeEnvironment,
    proof_sha256: str,
    matrix_sha256: str,
) -> JsonObject:
    """Build the non-promotable manifest payload for the compiled fast-path probe.

    Returns:
        The manifest JSON object with artifact hashes.

    """
    return cast(
        "JsonObject",
        {
            **_non_promotable_header(request=request, measurement=measurement),
            "world_size": environment.world_size,
            "gpu_names": list(environment.gpu_names),
            "torch_version": environment.torch_version,
            "artifacts": {
                "proof": PROOF_FILENAME,
                "proof_sha256": proof_sha256,
                "matrix": MATRIX_FILENAME,
                "matrix_sha256": matrix_sha256,
            },
        },
    )


def write_compiled_fastpath_probe_artifacts(
    *,
    request: CompiledFastpathProbeRequest,
    measurement: CompiledFastpathProbeMeasurement,
    environment: CompiledFastpathProbeEnvironment,
) -> CompiledFastpathProbeArtifacts:
    """Write the proof, matrix, and manifest, hashing proof/matrix into the manifest.

    Returns:
        The written artifact paths.

    """
    benchmark_dir = request.output_dir / BENCHMARK_DIRNAME
    proof_path = benchmark_dir / PROOF_FILENAME
    matrix_path = benchmark_dir / MATRIX_FILENAME
    manifest_path = benchmark_dir / MANIFEST_FILENAME
    write_json(
        proof_path,
        build_compiled_fastpath_probe_proof(
            request=request,
            measurement=measurement,
            environment=environment,
        ),
    )
    write_csv(
        matrix_path,
        COMPILED_FASTPATH_PROBE_MATRIX_COLUMNS,
        build_compiled_fastpath_probe_matrix_rows(
            request=request,
            measurement=measurement,
            environment=environment,
        ),
    )
    write_json(
        manifest_path,
        build_compiled_fastpath_probe_manifest(
            request=request,
            measurement=measurement,
            environment=environment,
            proof_sha256=_sha256(proof_path),
            matrix_sha256=_sha256(matrix_path),
        ),
    )
    return CompiledFastpathProbeArtifacts(
        proof=proof_path,
        matrix=matrix_path,
        manifest=manifest_path,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the compiled fast-path bake-off from the command line (one torchrun rank).

    Returns:
        Process exit status.

    """
    parser = argparse.ArgumentParser(description="Compiled fast-path GPU bake-off.")
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--batch-size", type=int, default=_DEFAULT_BATCH_SIZE)
    parser.add_argument("--batch-sizes", type=str, default=None)
    parser.add_argument("--warmup-steps", type=int, default=_DEFAULT_WARMUP_STEPS)
    parser.add_argument("--settle-steps", type=int, default=_DEFAULT_SETTLE_STEPS)
    parser.add_argument("--measured-steps", type=int, default=_DEFAULT_MEASURED_STEPS)
    args = parser.parse_args(argv)
    per_device_batch_size = cast("int", args.batch_size)
    run_compiled_fastpath_probe(
        CompiledFastpathProbeRequest(
            output_dir=cast("Path", args.output_dir),
            per_device_batch_size=per_device_batch_size,
            batch_sizes=_parse_batch_sizes(
                cast("str | None", args.batch_sizes),
                default=per_device_batch_size,
            ),
            warmup_steps=cast("int", args.warmup_steps),
            settle_steps=cast("int", args.settle_steps),
            measured_steps=cast("int", args.measured_steps),
        ),
    )
    return 0


def _parse_batch_sizes(raw: str | None, *, default: int) -> tuple[int, ...]:
    if raw is None:
        return (default,)
    sizes = tuple(int(token) for token in raw.split(",") if token.strip())
    return sizes or (default,)


def _init_distributed() -> _DistributedContext:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    nproc_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    return _DistributedContext(
        device=device,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        nproc_per_node=nproc_per_node,
    )


def _measure_config(
    spec: _RecipeSpec,
    reference_state: dict[str, Tensor],
    *,
    request: CompiledFastpathProbeRequest,
    distributed: _DistributedContext,
) -> _ConfigMeasurement:
    # Free the previous config's DDP model (its reducer hooks form reference cycles,
    # so it survives refcount drop until a collect) before building this one, so this
    # config's peak-VRAM reading reflects only its own footprint, not a lingering peer.
    gc.collect()
    torch.cuda.empty_cache()
    torch_dynamo.reset()
    _apply_dynamo_config(spec)
    context = _build_config_context(
        spec,
        reference_state,
        batch_size=request.per_device_batch_size,
        request=request,
        distributed=distributed,
    )
    _warmup(context, steps=request.warmup_steps)
    # Reset AFTER warmup so peak VRAM reflects the steady-state footprint (model +
    # optimizer state + compiled artifacts + activations), not the one-time inductor
    # compile / cuDNN autotune scratch spike from the first step.
    torch.cuda.reset_peak_memory_stats(distributed.device)
    graph_break_before = graph_break_total()
    unique_before = unique_graph_count()
    settle_nonfinite, sync_ok = _settle_and_sync(
        context,
        settle_steps=request.settle_steps,
        sync_check_steps=min(request.sync_check_steps, request.settle_steps),
        world_size=distributed.world_size,
    )
    step_ms, timing_nonfinite = _time_steps(context, steps=request.measured_steps)
    peak_vram_mb = float(torch.cuda.max_memory_allocated(distributed.device))
    peak_vram_mb /= _BYTES_PER_MB
    step_ms_p50 = statistics.median(step_ms)
    return _ConfigMeasurement(
        name=spec.name,
        compiled=spec.compiled,
        compile_scope=spec.compile_scope,
        syncs=sync_ok,
        graph_break_count=(
            graph_break_total() - graph_break_before if spec.compiled else _STABLE_DELTA
        ),
        recompile_count=(
            unique_graph_count() - unique_before if spec.compiled else _STABLE_DELTA
        ),
        step_ms_p50=step_ms_p50,
        samples_sec=_samples_sec(
            step_ms_p50,
            batch_size=request.per_device_batch_size,
            world_size=distributed.world_size,
        ),
        peak_vram_mb=peak_vram_mb,
        nonfinite_loss_count=settle_nonfinite + timing_nonfinite,
    )


def _apply_dynamo_config(spec: _RecipeSpec) -> None:
    torch_dynamo.config.optimize_ddp = spec.optimize_ddp
    torch_dynamo.config.compiled_autograd = spec.compiled_autograd
    inductor_config.reorder_for_compute_comm_overlap = spec.reorder_compute_comm_overlap


def _build_config_context(  # noqa: PLR0913
    spec: _RecipeSpec,
    reference_state: dict[str, Tensor],
    *,
    batch_size: int,
    request: CompiledFastpathProbeRequest,
    distributed: _DistributedContext,
    wrap_ddp: bool = True,
) -> _StepContext:
    # wrap_ddp=False builds a single-GPU (no-DDP) replica for the batch-sweep
    # feasibility probe: with no DDP wrapper the OOM-prone warmup issues NO collective
    # at all, so an asymmetric OOM there cannot desync the ranks. The timed path builds
    # with wrap_ddp=True to measure the real grad-syncing recipe.
    raw_model = _fresh_model(
        reference_state,
        device=distributed.device,
        channels_last=spec.channels_last,
    )
    optimizer = _build_optimizer(raw_model, fused=spec.fused_optimizer)
    inner_model = _maybe_compile_model(raw_model, spec=spec)
    model = (
        _wrap_ddp(inner_model, spec=spec, local_rank=distributed.local_rank)
        if wrap_ddp
        else inner_model
    )
    corruptor = InlineStainCorruptor(
        profile_from_name(CONSERVATIVE_DEFAULT_PROFILE),
    ).to(device=distributed.device)
    step_fn = make_fastpath_step_fn(
        model,
        corruptor,
        ssim_weight=_SSIM_WEIGHT,
        autocast_dtype=torch.float16,
    )
    if spec.compile_scope == _COMPILE_SCOPE_STEP:
        step_fn = torch.compile(  # pyright: ignore[reportUnknownMemberType]
            step_fn,
            dynamic=False,
            backend=_COMPILE_BACKEND,
        )
    x_clean = _synthetic_clean_batch(
        batch_size=batch_size,
        device=distributed.device,
        channels_last=spec.channels_last,
    )
    # Per-rank epsilon (FU-007): distinct noise per rank makes the positive sync
    # check meaningful -- a recipe whose comm silently drops would let the two
    # ranks' gradients (and thus parameters) diverge, which the guard then flags.
    eps_generator = torch.Generator(device=distributed.device)
    eps_generator.manual_seed(request.seed + distributed.rank)
    return _StepContext(
        step_fn=step_fn,
        x_clean=x_clean,
        latent_shape=_latent_eps_shape(raw_model, x_clean),
        beta=torch.ones((), device=distributed.device, dtype=torch.float32),
        optimizer=optimizer,
        scaler=GradScaler("cuda", init_scale=_GRAD_SCALER_INIT_SCALE),
        model=model,
        eps_generator=eps_generator,
        compiled_autograd=spec.compiled_autograd,
        device=distributed.device,
    )


def _maybe_compile_model(model: NonEquivariantVAE, *, spec: _RecipeSpec) -> nn.Module:
    if spec.compile_scope != _COMPILE_SCOPE_MODEL:
        return model
    return cast(
        "NonEquivariantVAE",
        torch.compile(  # pyright: ignore[reportUnknownMemberType]
            model,
            dynamic=False,
            backend=_COMPILE_BACKEND,
        ),
    )


def _build_optimizer(
    model: NonEquivariantVAE,
    *,
    fused: bool,
) -> torch.optim.Optimizer:
    config = SpecAdamWConfig()
    if not fused:
        optimizer, _ = create_adamw_optimizer(model, config=config)
        return optimizer
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.epsilon,
        weight_decay=config.weight_decay,
        fused=True,
    )


def _wrap_ddp(
    model: nn.Module,
    *,
    spec: _RecipeSpec,
    local_rank: int,
) -> DistributedDataParallel:
    return DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        static_graph=spec.ddp_static_graph,
        gradient_as_bucket_view=spec.ddp_gradient_as_bucket_view,
        broadcast_buffers=spec.ddp_broadcast_buffers,
        find_unused_parameters=False,
        bucket_cap_mb=spec.ddp_bucket_cap_mb,
    )


def _fresh_model(
    reference_state: dict[str, Tensor],
    *,
    device: torch.device,
    channels_last: bool,
) -> NonEquivariantVAE:
    model = build_non_equivariant_vae()
    model.load_state_dict(reference_state)
    if channels_last:
        model.to(  # pyright: ignore[reportCallIssue]
            device=device,
            memory_format=torch.channels_last,
        )
    else:
        model.to(device=device)
    return model


def _synthetic_clean_batch(
    *,
    batch_size: int,
    device: torch.device,
    channels_last: bool,
) -> Tensor:
    field = torch.linspace(
        -1.0,
        1.0,
        steps=_IMAGE_SIZE * _IMAGE_SIZE,
        device=device,
    ).view(_IMAGE_SIZE, _IMAGE_SIZE)
    batch = (
        field
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(
            batch_size,
            _IMAGE_CHANNELS,
            _IMAGE_SIZE,
            _IMAGE_SIZE,
        )
    )
    memory_format = torch.channels_last if channels_last else torch.contiguous_format
    return batch.contiguous(memory_format=memory_format)


def _latent_eps_shape(model: NonEquivariantVAE, x_clean: Tensor) -> torch.Size:
    with torch.no_grad():
        mu, _ = model.encode(x_clean)
    return mu.shape


def _autograd_context(
    *,
    compiled_autograd_enabled: bool,
) -> AbstractContextManager[None]:
    # Engage compiled autograd around the eager backward so the DDP python-reducer
    # all-reduce is traced into a compiled backward graph (comm/compute overlap). The
    # global config flag alone only covers the compiled forward call, so the backward
    # -- which runs after the compiled step returns -- needs this explicit context.
    if not compiled_autograd_enabled:
        return contextlib.nullcontext()
    compiler = cast("Callable[..., object]", torch.compile)
    return cast(
        "AbstractContextManager[None]",
        compiled_autograd._enable(compiler),  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    )


def _run_optimizer_step(context: _StepContext) -> Tensor:
    eps = torch.randn(
        context.latent_shape,
        generator=context.eps_generator,
        device=context.device,
        dtype=torch.float32,
    )
    context.optimizer.zero_grad(set_to_none=True)
    with _autograd_context(compiled_autograd_enabled=context.compiled_autograd):
        output = context.step_fn(context.x_clean, eps, context.beta)
        loss = output.loss
        cast("Callable[[], None]", context.scaler.scale(loss).backward)()
    context.scaler.unscale_(context.optimizer)
    torch.nn.utils.clip_grad_norm_(context.model.parameters(), max_norm=_MAX_GRAD_NORM)
    context.scaler.step(context.optimizer)
    context.scaler.update()
    return loss.detach()


def _warmup(context: _StepContext, *, steps: int) -> None:
    for _ in range(steps):
        _run_optimizer_step(context)


def _time_steps(context: _StepContext, *, steps: int) -> tuple[list[float], int]:
    samples: list[float] = []
    nonfinite = 0
    for _ in range(steps):
        torch.cuda.synchronize(context.device)
        start = time.perf_counter()
        loss = _run_optimizer_step(context)
        torch.cuda.synchronize(context.device)
        samples.append((time.perf_counter() - start) * _MS_PER_SECOND)
        nonfinite += _nonfinite_count(loss)
    return samples, nonfinite


def _settle_and_sync(
    context: _StepContext,
    *,
    settle_steps: int,
    sync_check_steps: int,
    world_size: int,
) -> tuple[int, bool]:
    nonfinite = 0
    sync_ok = True
    for index in range(settle_steps):
        loss = _run_optimizer_step(context)
        nonfinite += _nonfinite_count(loss)
        if index < sync_check_steps:
            # Evaluate the collective unconditionally (never short-circuit it) so
            # every rank runs the same number of all-gathers and cannot deadlock.
            current_in_sync = _in_sync(context.model, world_size=world_size)
            sync_ok = sync_ok and current_in_sync
    return nonfinite, sync_ok


def _in_sync(model: nn.Module, *, world_size: int) -> bool:
    try:
        assert_ddp_parameters_in_sync(model, world_size=world_size)
    except RuntimeError:
        return False
    return True


def _run_batch_sweep(
    reference_state: dict[str, Tensor],
    *,
    request: CompiledFastpathProbeRequest,
    distributed: _DistributedContext,
) -> tuple[_SweepPoint, ...]:
    # Every control-flow decision below branches ONLY on a cross-rank-reduced OOM flag
    # (an all_reduce inside _sweep_measure_symmetric / _sweep_feasible), never on a
    # rank-local number like samples_sec. So both ranks walk the identical sequence of
    # trial batch sizes and issue the identical collective sequence -> deadlock-safe.
    points: list[_SweepPoint] = []
    for spec in _SWEEP_SPECS:
        points.extend(
            _sweep_spec(
                spec,
                reference_state,
                request=request,
                distributed=distributed,
            ),
        )
    return tuple(points)


def _sweep_spec(
    spec: _RecipeSpec,
    reference_state: dict[str, Tensor],
    *,
    request: CompiledFastpathProbeRequest,
    distributed: _DistributedContext,
) -> list[_SweepPoint]:
    # Phase 1 -- fully timed doubling ladder (throughput curve). Ascending so a config
    # that OOMs at batch B skips all larger batches; the OOM point brackets the ceiling.
    points: list[_SweepPoint] = []
    low_ok: int | None = None
    high_oom: int | None = None
    for batch_size in _sweep_ladder_batches(request.batch_sizes):
        point = _sweep_measure_symmetric(
            spec,
            reference_state,
            batch_size=batch_size,
            request=request,
            distributed=distributed,
        )
        points.append(point)
        if point.oom:
            high_oom = batch_size
            break
        low_ok = batch_size
    # Phase 2 -- pin the exact ceiling with a cheap-probe binary search. Compiled
    # recipes only: the eager baseline is a throughput reference we never train, so its
    # OOM ceiling is not decision-relevant and not worth the extra compiles. A missing
    # high_oom means the ladder hit the safety cap without OOMing; low_ok (the cap) is
    # already a timed point, so there is nothing left to bisect.
    if not spec.compiled or low_ok is None or high_oom is None:
        return points
    ceiling_point = _sweep_ceiling_point(
        spec,
        reference_state,
        bracket=(low_ok, high_oom),
        request=request,
        distributed=distributed,
    )
    if ceiling_point is not None:
        points.append(ceiling_point)
    return points


def _sweep_ladder_batches(batch_sizes: tuple[int, ...]) -> tuple[int, ...]:
    # The requested sizes seed the throughput curve; the ladder then keeps doubling
    # past the largest requested size (up to a hard safety cap) so the sweep finds the
    # OOM edge on its own -- no need to enumerate every size by hand.
    requested = sorted({size for size in batch_sizes if size > 0})
    ladder = requested or [_SWEEP_BASE_BATCH]
    nxt = ladder[-1] * 2
    while nxt <= _SWEEP_MAX_BATCH:
        ladder.append(nxt)
        nxt *= 2
    return tuple(ladder)


def _sweep_ceiling_point(
    spec: _RecipeSpec,
    reference_state: dict[str, Tensor],
    *,
    bracket: tuple[int, int],
    request: CompiledFastpathProbeRequest,
    distributed: _DistributedContext,
) -> _SweepPoint | None:
    low_ok, high_oom = bracket

    def feasible(batch_size: int) -> bool:
        return _sweep_feasible(
            spec,
            reference_state,
            batch_size=batch_size,
            request=request,
            distributed=distributed,
        )

    ceiling = _binary_search_ceiling(feasible, low_ok=low_ok, high_oom=high_oom)
    if ceiling <= low_ok:
        # The bisection found nothing larger than the ladder point already measured.
        return None
    # One fully timed point at the true ceiling so its samples/sec + peak VRAM land in
    # the curve and _sweep_max_feasible_batch reports the pinned ceiling.
    return _sweep_measure_symmetric(
        spec,
        reference_state,
        batch_size=ceiling,
        request=request,
        distributed=distributed,
    )


def _binary_search_ceiling(
    feasible: Callable[[int], bool],
    *,
    low_ok: int,
    high_oom: int,
) -> int:
    """Return the largest batch known feasible in ``[low_ok, high_oom)``.

    Invariant on entry: ``feasible(low_ok)`` is True and ``feasible(high_oom)`` is
    False. ``feasible`` MUST return a value both ranks agree on (a cross-rank-reduced
    OOM flag), so both ranks probe the identical midpoint sequence and never diverge
    into mismatched collectives.

    Returns:
        The largest batch size proven feasible.

    """
    while high_oom - low_ok > _SWEEP_CEILING_GRANULARITY:
        mid = (low_ok + high_oom) // 2
        if feasible(mid):
            low_ok = mid
        else:
            high_oom = mid
    return low_ok


def _sweep_feasible(
    spec: _RecipeSpec,
    reference_state: dict[str, Tensor],
    *,
    batch_size: int,
    request: CompiledFastpathProbeRequest,
    distributed: _DistributedContext,
) -> bool:
    # Feasibility probe for the ceiling bisection: the single-GPU (no-DDP) probe below,
    # then a reduced flag so both ranks agree. Same primitive the timed points gate on.
    infeasible = _sweep_feasibility_attempt(
        spec,
        reference_state,
        batch_size=batch_size,
        request=request,
        distributed=distributed,
    )
    reduced = _all_reduce_int(infeasible, device=distributed.device)
    return reduced == _NO_OOM


def _sweep_feasibility_attempt(
    spec: _RecipeSpec,
    reference_state: dict[str, Tensor],
    *,
    batch_size: int,
    request: CompiledFastpathProbeRequest,
    distributed: _DistributedContext,
) -> int:
    # THE deadlock fix. Build a SINGLE-GPU (no-DDP) replica and warm it up. With no DDP
    # wrapper the OOM-prone forward/backward (and the inductor compile + cuDNN autotune
    # spike it drives) issues NO collective at all, so an asymmetric OOM here cannot
    # desync anything -- the only cross-rank op is the caller's reduce of the returned
    # flag. A batch counts as infeasible if it OOMs OR leaves less than the margin of
    # PHYSICAL free VRAM; the margin guarantees the real DDP timed run (which holds
    # more: buckets, NCCL buffers, the split graph) still fits, so it never OOMs.
    device = distributed.device
    gc.collect()
    torch.cuda.empty_cache()
    torch_dynamo.reset()
    _apply_dynamo_config(spec)
    torch.cuda.reset_peak_memory_stats(device)
    probe_warmup_steps = max(_SWEEP_PROBE_MIN_WARMUP_STEPS, request.warmup_steps)
    try:
        context = _build_config_context(
            spec,
            reference_state,
            batch_size=batch_size,
            request=request,
            distributed=distributed,
            wrap_ddp=False,
        )
        _warmup(context, steps=probe_warmup_steps)
        headroom = _sweep_probe_headroom_bytes(device)
    except torch.cuda.OutOfMemoryError:
        gc.collect()
        torch.cuda.empty_cache()
        return 1
    except RuntimeError as error:
        if not _is_oom_error(error):
            raise
        gc.collect()
        torch.cuda.empty_cache()
        return 1
    under_margin = headroom < int(_SWEEP_VRAM_MARGIN_MB * _BYTES_PER_MB)
    # Free the probe replica before the next probe / the timed rebuild so its footprint
    # does not inflate the following point's reading.
    del context
    gc.collect()
    torch.cuda.empty_cache()
    return 1 if under_margin else _NO_OOM


def _sweep_probe_headroom_bytes(device: torch.device) -> int:
    # Physical free VRAM left for the DDP timed build's extra footprint, read at the
    # probe's peak. Take the min of two bounds so it stays safe under either allocator:
    #   - free: exact when the allocator holds its peak reserved segments at read time
    #     (expandable_segments OFF, the default). It already subtracts the CUDA context,
    #     cuDNN / Triton / NCCL resident set that max_memory_reserved does not see.
    #   - (total - reserved): an allocator-independent bound valid even if
    #     expandable_segments returned segments to the OS between the activation peak
    #     and this read (so `free` alone could over-read the steady footprint).
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    return min(free_bytes, total_bytes - peak_reserved)


def _sweep_measure_symmetric(
    spec: _RecipeSpec,
    reference_state: dict[str, Tensor],
    *,
    batch_size: int,
    request: CompiledFastpathProbeRequest,
    distributed: _DistributedContext,
) -> _SweepPoint:
    # Gate on the single-GPU feasibility probe first (no collective except the reduce),
    # so both ranks agree before either builds the collective-issuing DDP model. If the
    # batch is infeasible on either rank, both return an oom point in lockstep.
    infeasible = _sweep_feasibility_attempt(
        spec,
        reference_state,
        batch_size=batch_size,
        request=request,
        distributed=distributed,
    )
    reduced = _all_reduce_int(infeasible, device=distributed.device)
    if reduced > _NO_OOM:
        return _oom_sweep_point(spec.name, batch_size=batch_size)
    # Feasible on both ranks with a VRAM margin -> a fresh DDP build (parameters loaded
    # from the shared reference_state, so the replicas start bit-identical -- no resync
    # needed) can run the real grad-syncing measurement without OOMing, so its gradient
    # all_reduces stay matched across ranks.
    context = _sweep_build_timed_context(
        spec,
        reference_state,
        batch_size=batch_size,
        request=request,
        distributed=distributed,
    )
    measurement = _sweep_measure(spec, context, request=request)
    syncs = _in_sync(context.model, world_size=distributed.world_size)
    return _SweepPoint(
        name=spec.name,
        batch_size=batch_size,
        samples_sec=_samples_sec(
            measurement.step_ms_p50,
            batch_size=batch_size,
            world_size=distributed.world_size,
        ),
        step_ms_p50=measurement.step_ms_p50,
        peak_vram_mb=measurement.peak_vram_mb,
        syncs=syncs,
        stable=measurement.stable,
        nonfinite=measurement.nonfinite,
        oom=False,
    )


def _sweep_build_timed_context(
    spec: _RecipeSpec,
    reference_state: dict[str, Tensor],
    *,
    batch_size: int,
    request: CompiledFastpathProbeRequest,
    distributed: _DistributedContext,
) -> _StepContext:
    # Match _measure_config's cleanup so this point's peak VRAM reflects only its own
    # footprint (the prior probe/point holds reference cycles until collected).
    gc.collect()
    torch.cuda.empty_cache()
    torch_dynamo.reset()
    _apply_dynamo_config(spec)
    return _build_config_context(
        spec,
        reference_state,
        batch_size=batch_size,
        request=request,
        distributed=distributed,
    )


def _sweep_measure(
    spec: _RecipeSpec,
    context: _StepContext,
    *,
    request: CompiledFastpathProbeRequest,
) -> _SweepMeasurement:
    _warmup(context, steps=request.warmup_steps)
    # Reset AFTER warmup so peak VRAM reflects the steady-state footprint, not the
    # one-time inductor compile / cuDNN autotune scratch spike from the first step.
    torch.cuda.reset_peak_memory_stats(context.device)
    graph_break_before = graph_break_total()
    unique_before = unique_graph_count()
    settle_nonfinite = _run_settle_no_sync(context, steps=request.settle_steps)
    step_ms, timing_nonfinite = _time_steps(context, steps=request.measured_steps)
    peak_vram_mb = float(torch.cuda.max_memory_allocated(context.device))
    peak_vram_mb /= _BYTES_PER_MB
    graph_break_count = (
        graph_break_total() - graph_break_before if spec.compiled else _STABLE_DELTA
    )
    recompile_count = (
        unique_graph_count() - unique_before if spec.compiled else _STABLE_DELTA
    )
    return _SweepMeasurement(
        step_ms_p50=statistics.median(step_ms),
        peak_vram_mb=peak_vram_mb,
        stable=(
            graph_break_count == _STABLE_DELTA and recompile_count == _STABLE_DELTA
        ),
        nonfinite=settle_nonfinite + timing_nonfinite,
    )


def _run_settle_no_sync(context: _StepContext, *, steps: int) -> int:
    # Deliberately runs NO cross-rank sync check: the sync-check collective is gated on
    # the reduced OOM flag in _sweep_measure_symmetric so it cannot deadlock a peer.
    nonfinite = 0
    for _ in range(steps):
        loss = _run_optimizer_step(context)
        nonfinite += _nonfinite_count(loss)
    return nonfinite


def _all_reduce_int(value: int, *, device: torch.device) -> int:
    flag = torch.tensor(value, device=device, dtype=torch.int64)
    all_reduce = cast("Callable[[Tensor], object]", dist.all_reduce)
    all_reduce(flag)
    return int(flag.item())


def _is_oom_error(error: RuntimeError) -> bool:
    return _OOM_MESSAGE_FRAGMENT in str(error).lower()


def _oom_sweep_point(name: str, *, batch_size: int) -> _SweepPoint:
    return _SweepPoint(
        name=name,
        batch_size=batch_size,
        samples_sec=0.0,
        step_ms_p50=0.0,
        peak_vram_mb=0.0,
        syncs=False,
        stable=False,
        nonfinite=0,
        oom=True,
    )


def _sweep_throughput_optimal(points: Sequence[_SweepPoint]) -> _SweepPoint | None:
    feasible = [point for point in points if not point.oom]
    if not feasible:
        return None
    return max(feasible, key=_sweep_point_samples_sec)


def _sweep_max_feasible_batch(points: Sequence[_SweepPoint]) -> int | None:
    feasible = [point.batch_size for point in points if not point.oom]
    if not feasible:
        return None
    return max(feasible)


def _sweep_point_samples_sec(point: _SweepPoint) -> float:
    return point.samples_sec


def _negative_control_fired(
    *,
    rank: int,
    world_size: int,
    device: torch.device,
) -> bool:
    try:
        run_negative_control_desync(rank=rank, world_size=world_size, device=device)
    except RuntimeError:
        return True
    return False


def _reduce_config_signals(
    configs: Sequence[_ConfigMeasurement],
    *,
    device: torch.device,
) -> list[_ConfigMeasurement]:
    flat: list[int] = []
    for config in configs:
        flat.extend((
            config.graph_break_count,
            config.recompile_count,
            config.nonfinite_loss_count,
        ))
    signals = torch.tensor(flat, device=device, dtype=torch.int64)
    all_reduce = cast("Callable[[Tensor], object]", dist.all_reduce)
    all_reduce(signals)
    reduced: list[_ConfigMeasurement] = []
    for index, config in enumerate(configs):
        base = index * _SIGNALS_PER_CONFIG
        reduced.append(
            _ConfigMeasurement(
                name=config.name,
                compiled=config.compiled,
                compile_scope=config.compile_scope,
                syncs=config.syncs,
                graph_break_count=int(signals[base].item()),
                recompile_count=int(signals[base + 1].item()),
                step_ms_p50=config.step_ms_p50,
                samples_sec=config.samples_sec,
                peak_vram_mb=config.peak_vram_mb,
                nonfinite_loss_count=int(signals[base + 2].item()),
            ),
        )
    return reduced


def _assemble_measurement(
    configs: Sequence[_ConfigMeasurement],
    *,
    negative_control_fired: bool,
    request: CompiledFastpathProbeRequest,
    sweep_points: tuple[_SweepPoint, ...],
) -> CompiledFastpathProbeMeasurement:
    eager_samples_sec = configs[0].samples_sec
    eager = _to_recipe_result(configs[0], eager_samples_sec=eager_samples_sec)
    recipes = tuple(
        _to_recipe_result(config, eager_samples_sec=eager_samples_sec)
        for config in configs[1:]
    )
    return CompiledFastpathProbeMeasurement(
        eager=eager,
        recipes=recipes,
        negative_control_fired=negative_control_fired,
        sync_check_steps=min(request.sync_check_steps, request.settle_steps),
        sweep_points=sweep_points,
    )


def _to_recipe_result(
    config: _ConfigMeasurement,
    *,
    eager_samples_sec: float,
) -> RecipeResult:
    return RecipeResult(
        name=config.name,
        compiled=config.compiled,
        compile_scope=config.compile_scope,
        syncs=config.syncs,
        graph_break_count=config.graph_break_count,
        recompile_count=config.recompile_count,
        step_ms_p50=config.step_ms_p50,
        samples_sec=config.samples_sec,
        peak_vram_mb=config.peak_vram_mb,
        nonfinite_loss_count=config.nonfinite_loss_count,
        speedup=_speedup(config.samples_sec, eager_samples_sec),
    )


def _correctness_held(measurement: CompiledFastpathProbeMeasurement) -> bool:
    return (
        measurement.sync_check_steps >= _MIN_SYNC_CHECK_STEPS
        and measurement.negative_control_fired
        and measurement.eager.syncs
        and all(recipe.syncs for recipe in measurement.recipes)
        and _nonfinite_total(measurement) == _STABLE_DELTA
    )


def _failure_message(measurement: CompiledFastpathProbeMeasurement) -> str:
    recipe_syncs = {recipe.name: recipe.syncs for recipe in measurement.recipes}
    return (
        "compiled fast-path bake-off failed a hard correctness invariant "
        f"(negative_control_fired={measurement.negative_control_fired}, "
        f"eager_syncs={measurement.eager.syncs}, "
        f"recipe_syncs={recipe_syncs}, "
        f"nonfinite_total={_nonfinite_total(measurement)})"
    )


def _nonfinite_total(measurement: CompiledFastpathProbeMeasurement) -> int:
    return measurement.eager.nonfinite_loss_count + sum(
        recipe.nonfinite_loss_count for recipe in measurement.recipes
    )


def _write_probe_artifacts_failclosed(
    *,
    rank: int,
    request: CompiledFastpathProbeRequest,
    measurement: CompiledFastpathProbeMeasurement,
    environment: CompiledFastpathProbeEnvironment,
    device: torch.device,
) -> bool:
    # Only rank 0 writes, but every rank all-reduces the failure flag so a rank-0
    # write error (e.g. an EDQUOT disk-quota failure on /kaggle/working) fails every
    # rank together instead of leaving the peers hung at the following barrier.
    write_failed = torch.zeros((), device=device, dtype=torch.int64)
    if rank == _PRIMARY_RANK:
        try:
            write_compiled_fastpath_probe_artifacts(
                request=request,
                measurement=measurement,
                environment=environment,
            )
        except OSError:
            write_failed += 1
    all_reduce = cast("Callable[[Tensor], object]", dist.all_reduce)
    all_reduce(write_failed)
    return bool(write_failed.item() > 0)


def _samples_sec(step_ms_p50: float, *, batch_size: int, world_size: int) -> float:
    if step_ms_p50 <= 0.0:
        return 0.0
    return batch_size * world_size / (step_ms_p50 / _MS_PER_SECOND)


def _speedup(samples_sec: float, eager_samples_sec: float) -> float:
    if eager_samples_sec <= 0.0:
        return 0.0
    return samples_sec / eager_samples_sec


def _recipe_samples_sec(recipe: RecipeResult) -> float:
    return recipe.samples_sec


def _nonfinite_count(loss: Tensor) -> int:
    return int((~torch.isfinite(loss)).sum().item())


def _gpu_names() -> tuple[str, ...]:
    return tuple(
        torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
    )


def _eager_baseline_payload(eager: RecipeResult) -> JsonObject:
    return cast(
        "JsonObject",
        {
            "name": eager.name,
            "step_ms_p50": eager.step_ms_p50,
            "samples_sec": eager.samples_sec,
            "peak_vram_mb": eager.peak_vram_mb,
            "syncs": eager.syncs,
        },
    )


def _recipe_payload(recipe: RecipeResult) -> JsonObject:
    return cast(
        "JsonObject",
        {
            "name": recipe.name,
            "compile_scope": recipe.compile_scope,
            "syncs": recipe.syncs,
            "graph_break": recipe.graph_break_count,
            "recompile": recipe.recompile_count,
            "stable": recipe.stable,
            "step_ms_p50": recipe.step_ms_p50,
            "samples_sec": recipe.samples_sec,
            "peak_vram_mb": recipe.peak_vram_mb,
            "speedup": recipe.speedup,
            "nonfinite": recipe.nonfinite_loss_count,
            "passed": recipe.passed,
        },
    )


def _winner_payload(winner: RecipeResult | None) -> JsonObject:
    if winner is None:
        return cast(
            "JsonObject",
            {"found": False, "name": None, "samples_sec": 0.0, "speedup": 0.0},
        )
    return cast(
        "JsonObject",
        {
            "found": True,
            "name": winner.name,
            "samples_sec": winner.samples_sec,
            "speedup": winner.speedup,
        },
    )


def _batch_sweep_payload(
    *,
    batch_sizes: tuple[int, ...],
    sweep_points: tuple[_SweepPoint, ...],
) -> JsonObject:
    grouped = _sweep_points_by_name(sweep_points)
    return cast(
        "JsonObject",
        {
            "requested_batch_sizes": list(batch_sizes),
            # A batch is feasible only if the single-GPU probe leaves at least this much
            # PHYSICAL free VRAM, so max_feasible_batch is the largest batch that fits
            # with this headroom -- the safe batch to train, not the hard OOM edge.
            "vram_margin_mb": _SWEEP_VRAM_MARGIN_MB,
            "points": [_sweep_point_payload(point) for point in sweep_points],
            "recipes": {
                name: _sweep_recipe_summary(points) for name, points in grouped.items()
            },
        },
    )


def _sweep_points_by_name(
    sweep_points: tuple[_SweepPoint, ...],
) -> dict[str, list[_SweepPoint]]:
    grouped: dict[str, list[_SweepPoint]] = {}
    for point in sweep_points:
        grouped.setdefault(point.name, []).append(point)
    return grouped


def _sweep_recipe_summary(points: Sequence[_SweepPoint]) -> JsonObject:
    return cast(
        "JsonObject",
        {
            "throughput_optimal": _throughput_optimal_payload(
                _sweep_throughput_optimal(points),
            ),
            "max_feasible_batch": _sweep_max_feasible_batch(points),
        },
    )


def _throughput_optimal_payload(optimal: _SweepPoint | None) -> JsonObject:
    if optimal is None:
        return cast(
            "JsonObject",
            {
                "found": False,
                "batch_size": None,
                "samples_sec": 0.0,
                "peak_vram_mb": 0.0,
            },
        )
    return cast(
        "JsonObject",
        {
            "found": True,
            "batch_size": optimal.batch_size,
            "samples_sec": optimal.samples_sec,
            "peak_vram_mb": optimal.peak_vram_mb,
        },
    )


def _sweep_point_payload(point: _SweepPoint) -> JsonObject:
    return cast(
        "JsonObject",
        {
            "name": point.name,
            "batch_size": point.batch_size,
            "samples_sec": point.samples_sec,
            "step_ms_p50": point.step_ms_p50,
            "peak_vram_mb": point.peak_vram_mb,
            "syncs": point.syncs,
            "stable": point.stable,
            "nonfinite": point.nonfinite,
            "oom": point.oom,
        },
    )


def _matrix_row(
    result: RecipeResult,
    *,
    context: _MatrixContext,
    is_winner: bool,
) -> CsvRow:
    return {
        "run_name": context.request.run_name,
        "benchmark_kind": COMPILED_FASTPATH_PROBE_KIND,
        "benchmark_source": COMPILED_FASTPATH_PROBE_SOURCE,
        "status_scope": COMPILED_FASTPATH_PROBE_STATUS_SCOPE,
        "status": context.status,
        "full_run_eligible": "false",
        "world_size": str(context.environment.world_size),
        "nproc_per_node": str(context.environment.nproc_per_node),
        "per_device_batch_size": str(context.request.per_device_batch_size),
        "phase": _SWEEP_PHASE_BAKEOFF,
        "batch_size": str(context.request.per_device_batch_size),
        "recipe_name": result.name,
        "compiled": _bool_text(value=result.compiled),
        "compile_scope": result.compile_scope,
        "is_winner": _bool_text(value=is_winner),
        "syncs": _bool_text(value=result.syncs),
        "graph_break_count": str(result.graph_break_count),
        "recompile_count": str(result.recompile_count),
        "stable": _bool_text(value=result.stable),
        "step_ms_p50": f"{result.step_ms_p50:.6f}",
        "samples_sec": f"{result.samples_sec:.6f}",
        "peak_vram_mb": f"{result.peak_vram_mb:.6f}",
        "speedup": f"{result.speedup:.6f}",
        "nonfinite_loss_count": str(result.nonfinite_loss_count),
        "oom": _bool_text(value=False),
        "negative_control_fired": _bool_text(value=context.negative_control_fired),
        "warmup_steps": str(context.request.warmup_steps),
        "settle_steps": str(context.request.settle_steps),
        "measured_steps": str(context.request.measured_steps),
    }


def _sweep_matrix_row(point: _SweepPoint, *, context: _MatrixContext) -> CsvRow:
    return {
        "run_name": context.request.run_name,
        "benchmark_kind": COMPILED_FASTPATH_PROBE_KIND,
        "benchmark_source": COMPILED_FASTPATH_PROBE_SOURCE,
        "status_scope": COMPILED_FASTPATH_PROBE_STATUS_SCOPE,
        "status": context.status,
        "full_run_eligible": "false",
        "world_size": str(context.environment.world_size),
        "nproc_per_node": str(context.environment.nproc_per_node),
        "per_device_batch_size": str(context.request.per_device_batch_size),
        "phase": _SWEEP_PHASE_SWEEP,
        "batch_size": str(point.batch_size),
        "recipe_name": point.name,
        "compiled": _bool_text(value=point.name != EAGER_BASELINE_NAME),
        "compile_scope": "",
        "is_winner": _bool_text(value=False),
        "syncs": _bool_text(value=point.syncs),
        "graph_break_count": "",
        "recompile_count": "",
        "stable": _bool_text(value=point.stable),
        "step_ms_p50": f"{point.step_ms_p50:.6f}",
        "samples_sec": f"{point.samples_sec:.6f}",
        "peak_vram_mb": f"{point.peak_vram_mb:.6f}",
        "speedup": "",
        "nonfinite_loss_count": str(point.nonfinite),
        "oom": _bool_text(value=point.oom),
        "negative_control_fired": _bool_text(value=context.negative_control_fired),
        "warmup_steps": str(context.request.warmup_steps),
        "settle_steps": str(context.request.settle_steps),
        "measured_steps": str(context.request.measured_steps),
    }


def _non_promotable_header(
    *,
    request: CompiledFastpathProbeRequest,
    measurement: CompiledFastpathProbeMeasurement,
) -> dict[str, JsonObject | list[str] | str | bool]:
    return {
        "schema_version": COMPILED_FASTPATH_PROBE_SCHEMA_VERSION,
        "run_name": request.run_name,
        "benchmark_kind": COMPILED_FASTPATH_PROBE_KIND,
        "benchmark_source": COMPILED_FASTPATH_PROBE_SOURCE,
        "status_scope": COMPILED_FASTPATH_PROBE_STATUS_SCOPE,
        "status": _status(measurement),
        "full_run_eligible": False,
        "full_training_launch_ready": False,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
        "blocked_claims": _blocked_claims(),
    }


def _blocked_claims() -> JsonObject:
    return cast("JsonObject", dict.fromkeys(BLOCKED_CLAIM_KEYS, True))


def _status(measurement: CompiledFastpathProbeMeasurement) -> str:
    return PROBE_STATUS_PASS if measurement.passed else PROBE_STATUS_FAIL


def _bool_text(*, value: bool) -> str:
    return "true" if value else "false"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "BLOCKED_CLAIM_KEYS",
    "COMPILED_FASTPATH_PROBE_KIND",
    "COMPILED_FASTPATH_PROBE_MATRIX_COLUMNS",
    "COMPILED_FASTPATH_PROBE_SCHEMA_VERSION",
    "COMPILED_FASTPATH_PROBE_SOURCE",
    "COMPILED_FASTPATH_PROBE_STATUS_SCOPE",
    "EAGER_BASELINE_NAME",
    "MANIFEST_FILENAME",
    "MATRIX_FILENAME",
    "PROBE_STATUS_FAIL",
    "PROBE_STATUS_PASS",
    "PROOF_FILENAME",
    "RECIPE_DDP_COMPILE_MODEL",
    "RECIPE_DDP_OPTIMIZER",
    "RECIPE_PYTHON_REDUCER",
    "CompiledFastpathProbeArtifacts",
    "CompiledFastpathProbeEnvironment",
    "CompiledFastpathProbeMeasurement",
    "CompiledFastpathProbeRequest",
    "RecipeResult",
    "build_compiled_fastpath_probe_manifest",
    "build_compiled_fastpath_probe_matrix_rows",
    "build_compiled_fastpath_probe_proof",
    "graph_break_total",
    "main",
    "run_compiled_fastpath_probe",
    "run_negative_control_desync",
    "unique_graph_count",
    "write_compiled_fastpath_probe_artifacts",
]


if __name__ == "__main__":
    raise SystemExit(main())
