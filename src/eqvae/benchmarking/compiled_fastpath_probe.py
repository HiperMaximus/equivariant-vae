# Copyright 2026 HiperMaximus
"""Synthetic dual-T4 measurement core for the compiled FSQ-style fast path.

`run_compiled_fastpath_probe` exercises the exact fast-path recipe
(`torch.compile(make_fastpath_step_fn(...), dynamic=False)` over a
`channels_last` DDP model with `compiled_autograd=True` / `optimize_ddp=False`)
on synthetic in-memory tensors with no dataset attached. It proves the recipe is
compile-stable (zero post-warmup graph breaks / recompiles), DDP-syncing (a
positive cross-rank parameter-sync check plus a negative control that must fire),
finite, and how fast it is against the current fast-path-OFF eager runtime. It
runs under ``torchrun --standalone --nproc_per_node=2`` on GPU and writes
NON-PROMOTABLE proof/matrix/manifest artifacts (``full_run_eligible`` is false,
no dataset sources, every real-run claim blocked).

The eager baseline is measured FIRST, while ``compiled_autograd`` / ``optimize_ddp``
are still at their defaults and cuDNN benchmarking is off, so it reflects the
current v5 eager runtime exactly (contiguous layout, ``static_graph=False``,
``gradient_as_bucket_view=False``); only then is the fast-path dynamo config
enabled for the compiled phase. The pure payload builders and the negative-control
helper are import-safe and CPU-testable; the GPU/NCCL core is skipped on CPU.
"""

from __future__ import annotations

import argparse
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
from torch._dynamo.utils import counters  # noqa: PLC2701
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

    from eqvae.benchmarking.io import CsvRow, JsonObject
    from eqvae.models.non_equivariant_vae import NonEquivariantVAE
    from eqvae.training.fastpath_step import FastpathStepOutput

COMPILED_FASTPATH_PROBE_KIND = "kaggle_compiled_fastpath_probe"
COMPILED_FASTPATH_PROBE_SOURCE = "kaggle_no_dataset_synthetic_compiled_fastpath"
COMPILED_FASTPATH_PROBE_STATUS_SCOPE = "non_promotable_compiled_fastpath_probe"
COMPILED_FASTPATH_PROBE_SCHEMA_VERSION = "spec0001.compiled_fastpath_probe.v1"
COMPILED_FASTPATH_PROBE_COMPILE_SCOPE = "step"
PROBE_STATUS_PASS = "compiled_fastpath_probe_pass"  # noqa: S105
PROBE_STATUS_FAIL = "compiled_fastpath_probe_fail"

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
    "compile_scope",
    "compile_dynamic",
    "compiled_autograd",
    "optimize_ddp",
    "ddp_static_graph",
    "ddp_gradient_as_bucket_view",
    "memory_format",
    "warmup_steps",
    "settle_steps",
    "graph_break_count",
    "recompile_count",
    "positive_sync_in_sync",
    "negative_control_fired",
    "measured_steps",
    "compiled_step_ms_p50",
    "eager_step_ms_p50",
    "speedup",
    "nonfinite_loss_count",
)

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
_PRIMARY_RANK = 0
_DESYNC_RANK = 0
_MS_PER_SECOND = 1000.0
_STABLE_DELTA = 0
_MIN_SYNC_CHECK_STEPS = 1
_SPEEDUP_FLOOR = 0.97


@dataclass(frozen=True)
class CompiledFastpathProbeRequest:
    """Inputs for one dual-T4 compiled fast-path probe run."""

    output_dir: Path
    run_name: str = "eqvae_compiled_fastpath_probe"
    per_device_batch_size: int = _DEFAULT_BATCH_SIZE
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


@dataclass(frozen=True)
class CompiledFastpathProbeMeasurement:
    """Measured probe outcomes across the settle, sync, and timing phases."""

    graph_break_count: int
    recompile_count: int
    positive_sync_in_sync: bool
    negative_control_fired: bool
    sync_check_steps: int
    compiled_step_ms_p50: float
    eager_step_ms_p50: float
    speedup: float
    nonfinite_loss_count: int

    @property
    def passed(self) -> bool:
        """Return whether every correctness invariant held (speed judged elsewhere).

        Speed is intentionally excluded: `speedup` is recorded for the step-6 gate,
        but per-rank timing noise must never flip the recipe-correctness verdict.
        A run that checked zero sync steps cannot pass: an unverified sync must not
        read as a proof.

        Returns:
            ``True`` if compile settled, both sync checks behaved, no loss was
            non-finite, and at least one positive sync check ran.

        """
        return (
            self.graph_break_count == _STABLE_DELTA
            and self.recompile_count == _STABLE_DELTA
            and self.positive_sync_in_sync
            and self.negative_control_fired
            and self.nonfinite_loss_count == _STABLE_DELTA
            and self.sync_check_steps >= _MIN_SYNC_CHECK_STEPS
        )


@dataclass(frozen=True)
class CompiledFastpathProbeArtifacts:
    """Paths written by the compiled fast-path probe."""

    proof: Path
    matrix: Path
    manifest: Path


@dataclass(frozen=True)
class _StepContext:
    step_fn: Callable[[Tensor, Tensor, Tensor], FastpathStepOutput]
    x_clean: Tensor
    latent_shape: torch.Size
    beta: Tensor
    optimizer: torch.optim.Optimizer
    scaler: GradScaler
    model: nn.Module
    device: torch.device


@dataclass(frozen=True)
class _CompiledPhaseResult:
    step_ms: list[float]
    graph_break_count: int
    recompile_count: int
    positive_sync_in_sync: bool
    nonfinite_loss_count: int


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


def run_compiled_fastpath_probe(  # noqa: PLR0914
    request: CompiledFastpathProbeRequest,
) -> CompiledFastpathProbeMeasurement:
    """Run the full compiled fast-path probe on the local rank and write artifacts.

    Measures the fast-path-OFF eager baseline first (clean dynamo state), then the
    compiled fast path, the cross-rank sync checks, and the finiteness of every
    loss. The compile/finiteness counters are reduced across ranks so the verdict
    and the rank-0-written non-promotable proof reflect both T4s; every rank then
    fails closed together if the write failed or any correctness invariant did not
    hold.

    Returns:
        The measured probe outcomes.

    Raises:
        RuntimeError: If rank 0 could not write the artifacts, or if any
            correctness invariant (settle, sync, finiteness) failed; the verdict
            is reduced across ranks so every rank raises identically.

    """
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    nproc_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch_dynamo.reset()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")

    manual_seed = cast("Callable[[int], torch.Generator]", torch.manual_seed)
    manual_seed(request.seed)
    reference_state = build_non_equivariant_vae().state_dict()

    eager_samples, eager_nonfinite = _measure_eager_phase(
        reference_state,
        request=request,
        device=device,
        local_rank=local_rank,
    )

    _configure_fastpath_dynamo()
    torch.backends.cudnn.benchmark = True
    compiled = _measure_compiled_phase(
        reference_state,
        request=request,
        device=device,
        local_rank=local_rank,
        world_size=world_size,
    )

    negative_control_fired = _negative_control_fired(
        rank=rank,
        world_size=world_size,
        device=device,
    )

    # Reduce the rank-local compile/finiteness counters across ranks so the written
    # proof and the pass/fail verdict reflect BOTH T4s, not just rank 0, and every
    # rank decides identically (the sync/negative-control signals are already
    # cross-rank via the guard's all-gather).
    graph_break_count, recompile_count, nonfinite_loss_count = (
        _reduce_correctness_signals(
            graph_break_count=compiled.graph_break_count,
            recompile_count=compiled.recompile_count,
            nonfinite_loss_count=eager_nonfinite + compiled.nonfinite_loss_count,
            device=device,
        )
    )
    measurement = CompiledFastpathProbeMeasurement(
        graph_break_count=graph_break_count,
        recompile_count=recompile_count,
        positive_sync_in_sync=compiled.positive_sync_in_sync,
        negative_control_fired=negative_control_fired,
        sync_check_steps=min(request.sync_check_steps, request.settle_steps),
        compiled_step_ms_p50=statistics.median(compiled.step_ms),
        eager_step_ms_p50=statistics.median(eager_samples),
        speedup=_speedup(eager_samples, compiled.step_ms),
        nonfinite_loss_count=nonfinite_loss_count,
    )
    write_failed = _write_probe_artifacts_failclosed(
        rank=rank,
        request=request,
        measurement=measurement,
        environment=CompiledFastpathProbeEnvironment(
            world_size=world_size,
            nproc_per_node=nproc_per_node,
            gpu_names=_gpu_names(),
        ),
        device=device,
    )
    barrier = cast("Callable[[], object]", dist.barrier)
    barrier()
    if write_failed:
        message = "rank 0 failed to write compiled fast-path probe artifacts"
        raise RuntimeError(message)
    if not measurement.passed:
        message = (
            "compiled fast-path probe failed a correctness invariant "
            f"(graph_break={measurement.graph_break_count}, "
            f"recompile={measurement.recompile_count}, "
            f"positive_sync={measurement.positive_sync_in_sync}, "
            f"negative_control_fired={measurement.negative_control_fired}, "
            f"nonfinite={measurement.nonfinite_loss_count})"
        )
        raise RuntimeError(message)
    return measurement


def build_compiled_fastpath_probe_proof(
    *,
    request: CompiledFastpathProbeRequest,
    measurement: CompiledFastpathProbeMeasurement,
    environment: CompiledFastpathProbeEnvironment,
) -> JsonObject:
    """Build the non-promotable proof payload for the compiled fast-path probe.

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
            "per_device_batch_size": request.per_device_batch_size,
            "compile": {
                "enabled": True,
                "scope": COMPILED_FASTPATH_PROBE_COMPILE_SCOPE,
                "dynamic": False,
                "compiled_autograd": True,
                "optimize_ddp": False,
            },
            "ddp": {
                "static_graph": True,
                "gradient_as_bucket_view": True,
                "find_unused_parameters": False,
                "memory_format": "channels_last",
            },
            "settle": {
                "warmup_steps": request.warmup_steps,
                "settle_steps": request.settle_steps,
                "graph_break_count": measurement.graph_break_count,
                "recompile_count": measurement.recompile_count,
            },
            "grad_sync": {
                "checked_steps": measurement.sync_check_steps,
                "positive_in_sync": measurement.positive_sync_in_sync,
                "negative_control_fired": measurement.negative_control_fired,
            },
            "throughput": {
                "measured_steps": request.measured_steps,
                "compiled_step_ms_p50": measurement.compiled_step_ms_p50,
                "eager_step_ms_p50": measurement.eager_step_ms_p50,
                "speedup": measurement.speedup,
                "speedup_floor": _SPEEDUP_FLOOR,
                "not_materially_slower": measurement.speedup >= _SPEEDUP_FLOOR,
            },
            "numerics": {
                "nonfinite_loss_count": measurement.nonfinite_loss_count,
                "autocast_dtype": "float16",
            },
        },
    )


def build_compiled_fastpath_probe_matrix_rows(
    *,
    request: CompiledFastpathProbeRequest,
    measurement: CompiledFastpathProbeMeasurement,
    environment: CompiledFastpathProbeEnvironment,
) -> list[CsvRow]:
    """Build the single-row matrix for the compiled fast-path probe.

    Returns:
        A one-element list with the probe summary row.

    """
    rows: list[CsvRow] = [
        {
            "run_name": request.run_name,
            "benchmark_kind": COMPILED_FASTPATH_PROBE_KIND,
            "benchmark_source": COMPILED_FASTPATH_PROBE_SOURCE,
            "status_scope": COMPILED_FASTPATH_PROBE_STATUS_SCOPE,
            "status": _status(measurement),
            "full_run_eligible": "false",
            "world_size": str(environment.world_size),
            "nproc_per_node": str(environment.nproc_per_node),
            "per_device_batch_size": str(request.per_device_batch_size),
            "compile_scope": COMPILED_FASTPATH_PROBE_COMPILE_SCOPE,
            "compile_dynamic": "false",
            "compiled_autograd": "true",
            "optimize_ddp": "false",
            "ddp_static_graph": "true",
            "ddp_gradient_as_bucket_view": "true",
            "memory_format": "channels_last",
            "warmup_steps": str(request.warmup_steps),
            "settle_steps": str(request.settle_steps),
            "graph_break_count": str(measurement.graph_break_count),
            "recompile_count": str(measurement.recompile_count),
            "positive_sync_in_sync": _bool_text(
                value=measurement.positive_sync_in_sync,
            ),
            "negative_control_fired": _bool_text(
                value=measurement.negative_control_fired,
            ),
            "measured_steps": str(request.measured_steps),
            "compiled_step_ms_p50": f"{measurement.compiled_step_ms_p50:.6f}",
            "eager_step_ms_p50": f"{measurement.eager_step_ms_p50:.6f}",
            "speedup": f"{measurement.speedup:.6f}",
            "nonfinite_loss_count": str(measurement.nonfinite_loss_count),
        },
    ]
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
    """Run the compiled fast-path probe from the command line (one torchrun rank).

    Returns:
        Process exit status.

    """
    parser = argparse.ArgumentParser(description="Compiled fast-path GPU probe.")
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--batch-size", type=int, default=_DEFAULT_BATCH_SIZE)
    parser.add_argument("--warmup-steps", type=int, default=_DEFAULT_WARMUP_STEPS)
    parser.add_argument("--settle-steps", type=int, default=_DEFAULT_SETTLE_STEPS)
    parser.add_argument("--measured-steps", type=int, default=_DEFAULT_MEASURED_STEPS)
    args = parser.parse_args(argv)
    run_compiled_fastpath_probe(
        CompiledFastpathProbeRequest(
            output_dir=cast("Path", args.output_dir),
            per_device_batch_size=cast("int", args.batch_size),
            warmup_steps=cast("int", args.warmup_steps),
            settle_steps=cast("int", args.settle_steps),
            measured_steps=cast("int", args.measured_steps),
        ),
    )
    return 0


def _measure_eager_phase(
    reference_state: dict[str, Tensor],
    *,
    request: CompiledFastpathProbeRequest,
    device: torch.device,
    local_rank: int,
) -> tuple[list[float], int]:
    context = _build_step_context(
        reference_state,
        request=request,
        device=device,
        local_rank=local_rank,
        compiled=False,
    )
    _warmup(context, steps=request.warmup_steps)
    return _time_steps(context, steps=request.measured_steps)


def _measure_compiled_phase(
    reference_state: dict[str, Tensor],
    *,
    request: CompiledFastpathProbeRequest,
    device: torch.device,
    local_rank: int,
    world_size: int,
) -> _CompiledPhaseResult:
    context = _build_step_context(
        reference_state,
        request=request,
        device=device,
        local_rank=local_rank,
        compiled=True,
    )
    _warmup(context, steps=request.warmup_steps)
    graph_break_before = graph_break_total()
    unique_before = unique_graph_count()
    settle_nonfinite, sync_ok = _settle_and_sync(
        context,
        settle_steps=request.settle_steps,
        sync_check_steps=min(request.sync_check_steps, request.settle_steps),
        world_size=world_size,
    )
    step_ms, timing_nonfinite = _time_steps(context, steps=request.measured_steps)
    return _CompiledPhaseResult(
        step_ms=step_ms,
        graph_break_count=graph_break_total() - graph_break_before,
        recompile_count=unique_graph_count() - unique_before,
        positive_sync_in_sync=sync_ok,
        nonfinite_loss_count=settle_nonfinite + timing_nonfinite,
    )


def _build_step_context(
    reference_state: dict[str, Tensor],
    *,
    request: CompiledFastpathProbeRequest,
    device: torch.device,
    local_rank: int,
    compiled: bool,
) -> _StepContext:
    raw_model = _fresh_model(reference_state, device=device, channels_last=compiled)
    optimizer, _ = create_adamw_optimizer(raw_model, config=SpecAdamWConfig())
    model = DistributedDataParallel(
        raw_model,
        device_ids=[local_rank],
        output_device=local_rank,
        static_graph=compiled,
        gradient_as_bucket_view=compiled,
        find_unused_parameters=False,
    )
    corruptor = InlineStainCorruptor(
        profile_from_name(CONSERVATIVE_DEFAULT_PROFILE),
    ).to(device=device)
    step_fn = make_fastpath_step_fn(
        model,
        corruptor,
        ssim_weight=_SSIM_WEIGHT,
        autocast_dtype=torch.float16,
    )
    if compiled:
        step_fn = torch.compile(  # pyright: ignore[reportUnknownMemberType]
            step_fn,
            dynamic=False,
        )
    x_clean = _synthetic_clean_batch(
        batch_size=request.per_device_batch_size,
        device=device,
        channels_last=compiled,
    )
    return _StepContext(
        step_fn=step_fn,
        x_clean=x_clean,
        latent_shape=_latent_eps_shape(raw_model, x_clean),
        beta=torch.ones((), device=device, dtype=torch.float32),
        optimizer=optimizer,
        scaler=GradScaler("cuda", init_scale=_GRAD_SCALER_INIT_SCALE),
        model=model,
        device=device,
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


def _run_optimizer_step(context: _StepContext) -> Tensor:
    eps = torch.randn(context.latent_shape, device=context.device, dtype=torch.float32)
    context.optimizer.zero_grad(set_to_none=True)
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


def _configure_fastpath_dynamo() -> None:
    torch_dynamo.config.compiled_autograd = True
    torch_dynamo.config.optimize_ddp = False


def _reduce_correctness_signals(
    *,
    graph_break_count: int,
    recompile_count: int,
    nonfinite_loss_count: int,
    device: torch.device,
) -> tuple[int, int, int]:
    signals = torch.tensor(
        [graph_break_count, recompile_count, nonfinite_loss_count],
        device=device,
        dtype=torch.int64,
    )
    all_reduce = cast("Callable[[Tensor], object]", dist.all_reduce)
    all_reduce(signals)
    return (
        int(signals[0].item()),
        int(signals[1].item()),
        int(signals[2].item()),
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


def _speedup(
    eager_samples: Sequence[float],
    compiled_samples: Sequence[float],
) -> float:
    compiled_p50 = statistics.median(compiled_samples)
    if compiled_p50 <= 0.0:
        return 0.0
    return statistics.median(eager_samples) / compiled_p50


def _nonfinite_count(loss: Tensor) -> int:
    return int((~torch.isfinite(loss)).sum().item())


def _gpu_names() -> tuple[str, ...]:
    return tuple(
        torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
    )


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
    "COMPILED_FASTPATH_PROBE_COMPILE_SCOPE",
    "COMPILED_FASTPATH_PROBE_KIND",
    "COMPILED_FASTPATH_PROBE_MATRIX_COLUMNS",
    "COMPILED_FASTPATH_PROBE_SCHEMA_VERSION",
    "COMPILED_FASTPATH_PROBE_SOURCE",
    "COMPILED_FASTPATH_PROBE_STATUS_SCOPE",
    "MANIFEST_FILENAME",
    "MATRIX_FILENAME",
    "PROBE_STATUS_FAIL",
    "PROBE_STATUS_PASS",
    "PROOF_FILENAME",
    "CompiledFastpathProbeArtifacts",
    "CompiledFastpathProbeEnvironment",
    "CompiledFastpathProbeMeasurement",
    "CompiledFastpathProbeRequest",
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
