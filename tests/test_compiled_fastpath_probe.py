# Copyright 2026 HiperMaximus
"""CPU-only tests for the compiled fast-path bake-off (fast-path port, step 5b).

The NCCL/DDP/CUDA measurement core (`run_compiled_fastpath_probe`) only runs
under `torchrun` on GPU and is exercised on Kaggle, not here. These tests cover
the import-safe, CPU-runnable public surface: the dynamo-counter helpers, the
negative-control desync guard, the per-recipe pass logic, the winner selection,
the non-promotable payload/artifact builders (one row per config), and the eager
step configuration the probe uses.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, cast

import pytest
import torch
import torch._dynamo as torch_dynamo  # noqa: PLC2701
from torch._dynamo.utils import counters  # noqa: PLC2701
from torch._inductor import config as inductor_config  # noqa: PLC2701

from eqvae.benchmarking.compiled_fastpath_probe import (
    _DDP_OPTIMIZER_SPEC,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _PYTHON_REDUCER_SPEC,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    BLOCKED_CLAIM_KEYS,
    COMPILED_FASTPATH_PROBE_STATUS_SCOPE,
    EAGER_BASELINE_NAME,
    PROBE_STATUS_FAIL,
    PROBE_STATUS_PASS,
    RECIPE_DDP_COMPILE_MODEL,
    RECIPE_DDP_OPTIMIZER,
    RECIPE_PYTHON_REDUCER,
    CompiledFastpathProbeEnvironment,
    CompiledFastpathProbeMeasurement,
    CompiledFastpathProbeRequest,
    RecipeResult,
    _apply_dynamo_config,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _binary_search_ceiling,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _build_optimizer,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _sweep_ladder_batches,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _sweep_max_feasible_batch,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _sweep_throughput_optimal,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _SweepPoint,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _wrap_ddp,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    build_compiled_fastpath_probe_matrix_rows,
    build_compiled_fastpath_probe_proof,
    graph_break_total,
    run_negative_control_desync,
    unique_graph_count,
    write_compiled_fastpath_probe_artifacts,
)
from eqvae.corruption.inline_stain import InlineStainCorruptor
from eqvae.corruption.stain import CONSERVATIVE_DEFAULT_PROFILE, profile_from_name
from eqvae.models.non_equivariant_vae import build_non_equivariant_vae
from eqvae.training import ddp_sync_guard, fastpath_recipe
from eqvae.training.fastpath_step import make_fastpath_step_fn

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

_WORLD_SIZE = 2
_IN_SYNC_FINGERPRINT: tuple[float, float] = (0.0, 0.0)
_STEP_BATCH = 2
_STEP_SIZE = 64
_STEP_SSIM_WEIGHT = 0.1
_DEFAULT_SYNC_CHECKS = 3
_EAGER_SAMPLES_SEC = 100.0
_SLOW_SAMPLES_SEC = 150.0
_FAST_SAMPLES_SEC = 240.0
_MATRIX_ROW_COUNT = 4
_SWEEP_BATCH_SIZES = (12, 24, 48, 96)
_SWEEP_OPTIMAL_BATCH = 24
_SWEEP_MAX_FEASIBLE_BATCH = 48
_EAGER_MAX_FEASIBLE_BATCH = 12
# Binary-search ceiling fixtures. _CEILING_GRANULARITY mirrors the source's
# _SWEEP_CEILING_GRANULARITY; the search must stop within that of the true ceiling.
_CEILING_LOW_OK = 48
_CEILING_HIGH_OOM = 384
_CEILING_TRUE_MAX = 200
_CEILING_GRANULARITY = 4
_CEILING_MAX_PROBES = 10
_CEILING_TIGHT_MAX = 150
_CEILING_OOM_BOUND = 192
# Ladder fixtures: base, first doubled rung past the requested seeds, and the cap.
_LADDER_BASE = 12
_LADDER_FIRST_DOUBLED = 48
_LADDER_CAP = 512


def _request(
    output_dir: Path,
    *,
    batch_sizes: tuple[int, ...] | None = None,
) -> CompiledFastpathProbeRequest:
    if batch_sizes is None:
        return CompiledFastpathProbeRequest(output_dir=output_dir)
    return CompiledFastpathProbeRequest(output_dir=output_dir, batch_sizes=batch_sizes)


def _sweep_point(  # noqa: PLR0913
    name: str,
    batch_size: int,
    *,
    samples_sec: float = 0.0,
    peak_vram_mb: float = 0.0,
    syncs: bool = True,
    stable: bool = True,
    nonfinite: int = 0,
    oom: bool = False,
) -> _SweepPoint:
    return _SweepPoint(
        name=name,
        batch_size=batch_size,
        samples_sec=samples_sec,
        step_ms_p50=0.0,
        peak_vram_mb=peak_vram_mb,
        syncs=syncs,
        stable=stable,
        nonfinite=nonfinite,
        oom=oom,
    )


def _sweep_points() -> tuple[_SweepPoint, ...]:
    # ddp_optimizer: fastest at batch 24, still feasible (slower) at 48, OOM at 96.
    # eager and ddp_compile_model each OOM at their second batch.
    return (
        _sweep_point(EAGER_BASELINE_NAME, 12, samples_sec=90.0, peak_vram_mb=6000.0),
        _sweep_point(EAGER_BASELINE_NAME, 24, oom=True),
        _sweep_point(RECIPE_DDP_OPTIMIZER, 12, samples_sec=130.0, peak_vram_mb=2500.0),
        _sweep_point(RECIPE_DDP_OPTIMIZER, 24, samples_sec=210.0, peak_vram_mb=4200.0),
        _sweep_point(RECIPE_DDP_OPTIMIZER, 48, samples_sec=205.0, peak_vram_mb=7800.0),
        _sweep_point(RECIPE_DDP_OPTIMIZER, 96, oom=True),
        _sweep_point(
            RECIPE_DDP_COMPILE_MODEL,
            12,
            samples_sec=120.0,
            peak_vram_mb=2600.0,
        ),
        _sweep_point(RECIPE_DDP_COMPILE_MODEL, 24, oom=True),
    )


def _environment() -> CompiledFastpathProbeEnvironment:
    return CompiledFastpathProbeEnvironment(
        world_size=_WORLD_SIZE,
        nproc_per_node=_WORLD_SIZE,
        gpu_names=("Tesla T4", "Tesla T4"),
        torch_version="2.12.0+cu124",
    )


def _eager_result(
    *,
    syncs: bool = True,
    samples_sec: float = _EAGER_SAMPLES_SEC,
    nonfinite_loss_count: int = 0,
) -> RecipeResult:
    return RecipeResult(
        name=EAGER_BASELINE_NAME,
        compiled=False,
        compile_scope="none",
        syncs=syncs,
        graph_break_count=0,
        recompile_count=0,
        step_ms_p50=12.0,
        samples_sec=samples_sec,
        peak_vram_mb=1000.0,
        nonfinite_loss_count=nonfinite_loss_count,
        speedup=1.0,
    )


def _recipe_result(  # noqa: PLR0913
    name: str,
    *,
    compile_scope: str = "step",
    syncs: bool = True,
    graph_break_count: int = 0,
    recompile_count: int = 0,
    samples_sec: float = _FAST_SAMPLES_SEC,
    nonfinite_loss_count: int = 0,
) -> RecipeResult:
    return RecipeResult(
        name=name,
        compiled=True,
        compile_scope=compile_scope,
        syncs=syncs,
        graph_break_count=graph_break_count,
        recompile_count=recompile_count,
        step_ms_p50=8.0,
        samples_sec=samples_sec,
        peak_vram_mb=1200.0,
        nonfinite_loss_count=nonfinite_loss_count,
        speedup=samples_sec / _EAGER_SAMPLES_SEC,
    )


def _measurement(
    *,
    recipes: tuple[RecipeResult, ...] | None = None,
    negative_control_fired: bool = True,
    eager: RecipeResult | None = None,
    sweep_points: tuple[_SweepPoint, ...] = (),
) -> CompiledFastpathProbeMeasurement:
    default_recipes = (
        _recipe_result(RECIPE_PYTHON_REDUCER, samples_sec=_FAST_SAMPLES_SEC),
        _recipe_result(RECIPE_DDP_OPTIMIZER, samples_sec=_SLOW_SAMPLES_SEC),
        _recipe_result(RECIPE_DDP_COMPILE_MODEL, samples_sec=_SLOW_SAMPLES_SEC),
    )
    return CompiledFastpathProbeMeasurement(
        eager=eager if eager is not None else _eager_result(),
        recipes=recipes if recipes is not None else default_recipes,
        negative_control_fired=negative_control_fired,
        sync_check_steps=_DEFAULT_SYNC_CHECKS,
        sweep_points=sweep_points,
    )


def _clean_fn(value: torch.Tensor) -> torch.Tensor:
    return value + 1.0


def _data_dependent_fn(value: torch.Tensor) -> torch.Tensor:
    result = value + 1.0
    if bool((result.sum() > 0).item()):
        result *= 2.0
    return result


def _load_json(path: Path) -> dict[str, object]:
    return cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))


def test_counters_report_clean_compile_and_graph_break() -> None:
    """A clean compile adds a graph with zero breaks; a data-dependent one breaks."""
    torch_dynamo.reset()
    counters.clear()
    clean = torch.compile(  # pyright: ignore[reportUnknownMemberType]
        _clean_fn,
        backend="eager",
    )
    clean(torch.zeros(3))
    assert graph_break_total() == 0
    assert unique_graph_count() >= 1

    torch_dynamo.reset()
    counters.clear()
    broken = torch.compile(  # pyright: ignore[reportUnknownMemberType]
        _data_dependent_fn,
        backend="eager",
    )
    broken(torch.ones(3))
    assert graph_break_total() > 0


def test_negative_control_raises_on_hand_desynced_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rank 0's desync makes the guard see divergence versus a pristine rank 1."""

    def gather(gathered: list[object], obj: object) -> None:
        gathered[0] = obj
        gathered[1] = _IN_SYNC_FINGERPRINT

    monkeypatch.setattr(ddp_sync_guard.dist, "all_gather_object", gather)
    with pytest.raises(RuntimeError, match="divergent parameters"):
        run_negative_control_desync(
            rank=0,
            world_size=_WORLD_SIZE,
            device=torch.device("cpu"),
        )


def test_negative_control_stays_silent_when_rank_not_desynced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only rank 0 desyncs, so a non-zero rank leaves the pair in sync (no raise)."""

    def gather(gathered: list[object], obj: object) -> None:
        gathered[0] = obj
        gathered[1] = _IN_SYNC_FINGERPRINT

    monkeypatch.setattr(ddp_sync_guard.dist, "all_gather_object", gather)
    run_negative_control_desync(
        rank=_WORLD_SIZE - 1,
        world_size=_WORLD_SIZE,
        device=torch.device("cpu"),
    )


def test_recipe_passes_only_when_synced_stable_and_finite() -> None:
    """A recipe is a winner candidate only if it synced, settled, and stayed finite."""
    assert _recipe_result(RECIPE_PYTHON_REDUCER).passed
    assert not _recipe_result(RECIPE_PYTHON_REDUCER, syncs=False).passed
    assert not _recipe_result(RECIPE_PYTHON_REDUCER, graph_break_count=1).passed
    assert not _recipe_result(RECIPE_PYTHON_REDUCER, recompile_count=1).passed
    assert not _recipe_result(RECIPE_PYTHON_REDUCER, nonfinite_loss_count=1).passed
    assert not _recipe_result(RECIPE_PYTHON_REDUCER, graph_break_count=1).stable
    # The eager baseline never counts as a passing recipe candidate.
    assert not _eager_result().passed


def test_winner_is_fastest_passing_recipe() -> None:
    """The winner is the highest-throughput recipe among those that pass."""
    measurement = _measurement()
    winner = measurement.winner
    assert winner is not None
    assert winner.name == RECIPE_PYTHON_REDUCER
    assert measurement.passed


def test_winner_skips_faster_but_unstable_recipe() -> None:
    """A faster recipe that broke stability is disqualified from winning."""
    measurement = _measurement(
        recipes=(
            _recipe_result(
                RECIPE_PYTHON_REDUCER,
                samples_sec=_FAST_SAMPLES_SEC,
                graph_break_count=1,
            ),
            _recipe_result(RECIPE_DDP_OPTIMIZER, samples_sec=_SLOW_SAMPLES_SEC),
        ),
    )
    winner = measurement.winner
    assert winner is not None
    assert winner.name == RECIPE_DDP_OPTIMIZER


def test_measurement_fails_when_no_recipe_passes() -> None:
    """With every recipe desynced, there is no winner and the verdict is a fail."""
    measurement = _measurement(
        recipes=(
            _recipe_result(RECIPE_PYTHON_REDUCER, syncs=False),
            _recipe_result(RECIPE_DDP_OPTIMIZER, syncs=False),
        ),
    )
    assert measurement.winner is None
    assert not measurement.passed


def test_measurement_fails_when_negative_control_silent() -> None:
    """A negative control that never fired flips the pass verdict to fail."""
    measurement = _measurement(negative_control_fired=False)
    assert measurement.winner is not None
    assert not measurement.passed


def test_proof_payload_marks_non_promotable(tmp_path: Path) -> None:
    """The proof blocks promotion: no eligibility, no sources, every claim blocked."""
    measurement = _measurement()
    proof = build_compiled_fastpath_probe_proof(
        request=_request(tmp_path),
        measurement=measurement,
        environment=_environment(),
    )
    assert proof["full_run_eligible"] is False
    assert proof["full_training_launch_ready"] is False
    assert proof["dataset_sources"] == []
    assert proof["competition_sources"] == []
    assert proof["kernel_sources"] == []
    assert proof["model_sources"] == []
    assert proof["status_scope"] == COMPILED_FASTPATH_PROBE_STATUS_SCOPE
    assert proof["status"] == PROBE_STATUS_PASS
    assert proof["negative_control_fired"] is True
    assert proof["torch_version"] == "2.12.0+cu124"
    assert proof["world_size"] == _WORLD_SIZE
    assert proof["per_device_batch_size"] == _request(tmp_path).per_device_batch_size
    blocked = cast("dict[str, object]", proof["blocked_claims"])
    assert set(blocked) == set(BLOCKED_CLAIM_KEYS)
    assert all(blocked.values())


def test_proof_records_recipes_eager_and_winner(tmp_path: Path) -> None:
    """The proof carries every recipe row, the eager baseline, and the winner."""
    proof = build_compiled_fastpath_probe_proof(
        request=_request(tmp_path),
        measurement=_measurement(),
        environment=_environment(),
    )
    recipes = cast("list[dict[str, object]]", proof["recipes"])
    assert len(recipes) == _MATRIX_ROW_COUNT - 1
    assert {row["name"] for row in recipes} == {
        RECIPE_PYTHON_REDUCER,
        RECIPE_DDP_OPTIMIZER,
        RECIPE_DDP_COMPILE_MODEL,
    }
    eager = cast("dict[str, object]", proof["eager_baseline"])
    assert eager["name"] == EAGER_BASELINE_NAME
    assert eager["syncs"] is True
    winner = cast("dict[str, object]", proof["winner"])
    assert winner["found"] is True
    assert winner["name"] == RECIPE_PYTHON_REDUCER


def test_proof_status_reflects_failed_measurement(tmp_path: Path) -> None:
    """A silent negative control marks the proof status as failed."""
    proof = build_compiled_fastpath_probe_proof(
        request=_request(tmp_path),
        measurement=_measurement(negative_control_fired=False),
        environment=_environment(),
    )
    assert proof["status"] == PROBE_STATUS_FAIL
    winner = cast("dict[str, object]", proof["winner"])
    assert winner["found"] is True


def test_matrix_has_one_row_per_config_and_flags_winner(tmp_path: Path) -> None:
    """The matrix echoes eager plus three recipes, marking exactly one winner."""
    rows = build_compiled_fastpath_probe_matrix_rows(
        request=_request(tmp_path),
        measurement=_measurement(),
        environment=_environment(),
    )
    assert len(rows) == _MATRIX_ROW_COUNT
    assert rows[0]["recipe_name"] == EAGER_BASELINE_NAME
    assert rows[0]["compiled"] == "false"
    winners = [row for row in rows if row["is_winner"] == "true"]
    assert len(winners) == 1
    assert winners[0]["recipe_name"] == RECIPE_PYTHON_REDUCER
    for row in rows:
        assert row["status_scope"] == COMPILED_FASTPATH_PROBE_STATUS_SCOPE
        assert row["full_run_eligible"] == "false"
        assert row["status"] == PROBE_STATUS_PASS
        assert row["negative_control_fired"] == "true"


def test_sweep_throughput_optimal_picks_highest_feasible_samples_sec() -> None:
    """The optimal point is the highest-throughput non-OOM batch, not the largest."""
    points = [
        _sweep_point(RECIPE_DDP_OPTIMIZER, 12, samples_sec=100.0),
        _sweep_point(RECIPE_DDP_OPTIMIZER, 24, samples_sec=180.0),
        _sweep_point(RECIPE_DDP_OPTIMIZER, 48, samples_sec=150.0),
        _sweep_point(RECIPE_DDP_OPTIMIZER, 96, oom=True),
    ]
    optimal = _sweep_throughput_optimal(points)
    assert optimal is not None
    assert optimal.batch_size == _SWEEP_OPTIMAL_BATCH
    # The largest non-OOM batch (48) is feasible but slower than the optimal (24).
    assert _sweep_max_feasible_batch(points) == _SWEEP_MAX_FEASIBLE_BATCH


def test_sweep_ignores_oom_points_when_selecting() -> None:
    """An all-OOM sweep has neither a throughput optimum nor a feasible batch."""
    points = [
        _sweep_point(RECIPE_DDP_OPTIMIZER, 48, oom=True),
        _sweep_point(RECIPE_DDP_OPTIMIZER, 96, oom=True),
    ]
    assert _sweep_throughput_optimal(points) is None
    assert _sweep_max_feasible_batch(points) is None


def test_binary_search_ceiling_pins_largest_feasible_batch() -> None:
    """The bisection returns the largest feasible batch within the granularity."""
    probed: list[int] = []

    def feasible(batch_size: int) -> bool:
        probed.append(batch_size)
        return batch_size <= _CEILING_TRUE_MAX

    ceiling = _binary_search_ceiling(feasible, low_ok=48, high_oom=384)
    # Within granularity (4) of the true 200-batch ceiling, never above it.
    assert ceiling <= _CEILING_TRUE_MAX
    assert _CEILING_TRUE_MAX - ceiling <= _CEILING_GRANULARITY
    # A bisection touches O(log range) midpoints, not every candidate batch.
    assert len(probed) <= _CEILING_MAX_PROBES
    assert all(_CEILING_LOW_OK <= size <= _CEILING_HIGH_OOM for size in probed)


def test_binary_search_ceiling_never_probes_or_returns_the_oom_bound() -> None:
    """Feasibility is only ever probed strictly below the known-OOM upper bound."""
    probed: list[int] = []

    def feasible(batch_size: int) -> bool:
        probed.append(batch_size)
        return batch_size <= _CEILING_TIGHT_MAX

    ceiling = _binary_search_ceiling(feasible, low_ok=96, high_oom=192)
    assert ceiling <= _CEILING_TIGHT_MAX
    # The known-OOM bound is never re-probed (no wasted OOM) and never returned.
    assert all(size < _CEILING_OOM_BOUND for size in probed)
    assert ceiling < _CEILING_OOM_BOUND


def test_sweep_ladder_auto_extends_past_requested_until_the_cap() -> None:
    """The ladder keeps doubling past the largest requested size up to the cap."""
    ladder = _sweep_ladder_batches((12, 24))
    # Requested sizes are preserved and de-duplicated, ascending.
    assert ladder[:2] == (12, 24)
    # It then doubles (48, 96, ...) so the sweep finds the OOM edge on its own.
    assert ladder[2] == _LADDER_FIRST_DOUBLED
    assert all(ladder[idx] == ladder[idx - 1] * 2 for idx in range(2, len(ladder)))
    assert max(ladder) <= _LADDER_CAP


def test_sweep_ladder_dedupes_and_defaults_empty_request() -> None:
    """Duplicate/zero sizes collapse and an empty request falls back to the base."""
    assert _sweep_ladder_batches((48, 24, 24, 0))[:2] == (24, 48)
    assert _sweep_ladder_batches(())[0] == _LADDER_BASE


def test_proof_batch_sweep_reports_optimal_and_feasible(tmp_path: Path) -> None:
    """The proof's batch_sweep section carries every point and per-recipe summaries."""
    points = _sweep_points()
    proof = build_compiled_fastpath_probe_proof(
        request=_request(tmp_path, batch_sizes=_SWEEP_BATCH_SIZES),
        measurement=_measurement(sweep_points=points),
        environment=_environment(),
    )
    sweep = cast("dict[str, object]", proof["batch_sweep"])
    assert sweep["requested_batch_sizes"] == list(_SWEEP_BATCH_SIZES)
    rows = cast("list[dict[str, object]]", sweep["points"])
    assert len(rows) == len(points)
    assert any(row["oom"] is True for row in rows)
    recipes = cast("dict[str, dict[str, object]]", sweep["recipes"])
    ddp = recipes[RECIPE_DDP_OPTIMIZER]
    optimal = cast("dict[str, object]", ddp["throughput_optimal"])
    assert optimal["found"] is True
    assert optimal["batch_size"] == _SWEEP_OPTIMAL_BATCH
    assert ddp["max_feasible_batch"] == _SWEEP_MAX_FEASIBLE_BATCH
    eager = recipes[EAGER_BASELINE_NAME]
    assert eager["max_feasible_batch"] == _EAGER_MAX_FEASIBLE_BATCH


def test_proof_batch_sweep_empty_when_no_points(tmp_path: Path) -> None:
    """A single-batch run with no sweep points yields an empty batch_sweep section."""
    proof = build_compiled_fastpath_probe_proof(
        request=_request(tmp_path),
        measurement=_measurement(),
        environment=_environment(),
    )
    sweep = cast("dict[str, object]", proof["batch_sweep"])
    assert sweep["points"] == []
    assert sweep["recipes"] == {}


def test_matrix_appends_sweep_rows_with_phase_and_batch(tmp_path: Path) -> None:
    """Sweep rows extend the four bake-off rows, tagged phase=sweep with a batch."""
    points = _sweep_points()
    rows = build_compiled_fastpath_probe_matrix_rows(
        request=_request(tmp_path, batch_sizes=_SWEEP_BATCH_SIZES),
        measurement=_measurement(sweep_points=points),
        environment=_environment(),
    )
    bakeoff_rows = [row for row in rows if row["phase"] == "bakeoff"]
    sweep_rows = [row for row in rows if row["phase"] == "sweep"]
    assert len(bakeoff_rows) == _MATRIX_ROW_COUNT
    assert len(sweep_rows) == len(points)
    assert all(row["batch_size"] for row in sweep_rows)
    assert any(row["oom"] == "true" for row in sweep_rows)
    assert all(row["oom"] == "false" for row in bakeoff_rows)


def test_write_artifacts_emits_three_non_promotable_files(tmp_path: Path) -> None:
    """Writing produces proof/matrix/manifest, and the manifest hashes match."""
    artifacts = write_compiled_fastpath_probe_artifacts(
        request=_request(tmp_path),
        measurement=_measurement(),
        environment=_environment(),
    )
    assert artifacts.proof.exists()
    assert artifacts.matrix.exists()
    assert artifacts.manifest.exists()
    proof = _load_json(artifacts.proof)
    manifest = _load_json(artifacts.manifest)
    assert proof["full_run_eligible"] is False
    assert manifest["full_run_eligible"] is False
    manifest_artifacts = cast("dict[str, object]", manifest["artifacts"])
    assert (
        manifest_artifacts["proof_sha256"]
        == hashlib.sha256(
            artifacts.proof.read_bytes(),
        ).hexdigest()
    )
    assert (
        manifest_artifacts["matrix_sha256"]
        == hashlib.sha256(
            artifacts.matrix.read_bytes(),
        ).hexdigest()
    )
    matrix_text = artifacts.matrix.read_text(encoding="utf-8")
    assert PROBE_STATUS_PASS in matrix_text
    assert RECIPE_PYTHON_REDUCER in matrix_text
    assert EAGER_BASELINE_NAME in matrix_text


def test_probe_step_configuration_backprops_eagerly() -> None:
    """The probe's default corruptor + ssim weight give a finite backpropable loss."""
    model = build_non_equivariant_vae()
    corruptor = InlineStainCorruptor(profile_from_name(CONSERVATIVE_DEFAULT_PROFILE))
    step_fn = make_fastpath_step_fn(
        model,
        corruptor,
        ssim_weight=_STEP_SSIM_WEIGHT,
        autocast_dtype=torch.bfloat16,
    )
    generator = torch.Generator().manual_seed(0)
    x_clean = (
        torch.rand((_STEP_BATCH, 3, _STEP_SIZE, _STEP_SIZE), generator=generator) * 2.0
    ) - 1.0
    with torch.no_grad():
        mu_shape = model.forward(x_clean).mu.shape
    eps = torch.randn(mu_shape, generator=generator)

    output = step_fn(x_clean, eps, torch.tensor(1.0))

    assert bool(torch.isfinite(output.loss).item())
    assert output.loss.requires_grad
    cast("Callable[[], None]", output.loss.backward)()


def test_probe_step_stays_finite_on_degenerate_batch() -> None:
    """The probe's degenerate linspace input (all images identical) stays finite."""
    model = build_non_equivariant_vae()
    corruptor = InlineStainCorruptor(profile_from_name(CONSERVATIVE_DEFAULT_PROFILE))
    step_fn = make_fastpath_step_fn(
        model,
        corruptor,
        ssim_weight=_STEP_SSIM_WEIGHT,
        autocast_dtype=torch.bfloat16,
    )
    field = torch.linspace(-1.0, 1.0, steps=_STEP_SIZE * _STEP_SIZE).view(
        _STEP_SIZE,
        _STEP_SIZE,
    )
    x_clean = (
        field
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(_STEP_BATCH, 3, _STEP_SIZE, _STEP_SIZE)
        .contiguous()
    )
    with torch.no_grad():
        mu_shape = model.forward(x_clean).mu.shape
    eps = torch.zeros(mu_shape)

    output = step_fn(x_clean, eps, torch.tensor(1.0))

    assert bool(torch.isfinite(output.loss).item())


def test_probe_build_optimizer_routes_through_the_grouped_builder() -> None:
    """The probe's fused optimizer is grouped, matching the runner's eager path.

    A regression to a flat ``torch.optim.AdamW(model.parameters(), fused=True)``
    would collapse to a single parameter group (and raise on this CPU model).
    """
    model = build_non_equivariant_vae()

    optimizer = _build_optimizer(model, fused=True)

    assert {cast("str", group["name"]) for group in optimizer.param_groups} == {
        "decay",
        "no_decay",
        "gate_no_decay",
    }
    assert all(
        cast("object", group["fused"]) is None for group in optimizer.param_groups
    )


def test_probe_wrap_ddp_forwards_the_spec_ddp_knobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_wrap_ddp` maps the recipe spec's DDP knobs and find_unused=False to DDP."""
    captured: dict[str, object] = {}

    def fake_ddp(model: object, **kwargs: object) -> object:
        captured["model"] = model
        captured["kwargs"] = kwargs
        return "wrapped"

    monkeypatch.setattr(fastpath_recipe, "DistributedDataParallel", fake_ddp)
    model = build_non_equivariant_vae()

    result = _wrap_ddp(model, spec=_DDP_OPTIMIZER_SPEC, local_rank=0)

    assert result == "wrapped"
    assert captured["model"] is model
    assert captured["kwargs"] == {
        "device_ids": [0],
        "output_device": 0,
        "static_graph": _DDP_OPTIMIZER_SPEC.ddp_static_graph,
        "gradient_as_bucket_view": _DDP_OPTIMIZER_SPEC.ddp_gradient_as_bucket_view,
        "broadcast_buffers": _DDP_OPTIMIZER_SPEC.ddp_broadcast_buffers,
        "find_unused_parameters": False,
        "bucket_cap_mb": _DDP_OPTIMIZER_SPEC.ddp_bucket_cap_mb,
    }


def test_probe_apply_dynamo_config_forwards_the_spec_knobs() -> None:
    """`_apply_dynamo_config` writes the recipe spec's dynamo/inductor knobs."""
    original_optimize_ddp = cast("object", torch_dynamo.config.optimize_ddp)
    original_compiled_autograd = cast("object", torch_dynamo.config.compiled_autograd)
    original_reorder = cast(
        "object",
        inductor_config.reorder_for_compute_comm_overlap,
    )
    try:
        _apply_dynamo_config(_PYTHON_REDUCER_SPEC)
        assert (
            cast("object", torch_dynamo.config.optimize_ddp)
            == _PYTHON_REDUCER_SPEC.optimize_ddp
        )
        assert (
            cast("object", torch_dynamo.config.compiled_autograd)
            == _PYTHON_REDUCER_SPEC.compiled_autograd
        )
        assert (
            cast("object", inductor_config.reorder_for_compute_comm_overlap)
            == _PYTHON_REDUCER_SPEC.reorder_compute_comm_overlap
        )
    finally:
        torch_dynamo.config.optimize_ddp = original_optimize_ddp
        torch_dynamo.config.compiled_autograd = original_compiled_autograd
        inductor_config.reorder_for_compute_comm_overlap = original_reorder
