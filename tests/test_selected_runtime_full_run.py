# Copyright 2026 HiperMaximus
# pyright: reportPrivateUsage=false
"""Focused tests for Spec 0009 selected-runtime full-run readiness."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, NoReturn, cast

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

import numpy as np
import pytest
import torch

from eqvae.artifacts.fixed25_equivariance import (
    EQUIVARIANCE_25_COLUMNS,
    REQUIRED_EQUIVARIANCE_METRICS,
)
from eqvae.benchmarking import selected_runtime_gate
from eqvae.benchmarking.runtime_schema import GATE_HEALTH_COLUMNS
from eqvae.benchmarking.selected_runtime_gate import verify_selected_runtime_full_output
from eqvae.checkpointing import CheckpointMetadata, LoadedCheckpoint
from eqvae.cli.selected_runtime_train import main as selected_runtime_train_main
from eqvae.config import resolve_json_config
from eqvae.losses.vae import VaeLossComponents, beta_for_step
from eqvae.models.non_equivariant_vae import build_non_equivariant_vae
from eqvae.training import selected_runtime_runner
from eqvae.training.selected_runtime import (
    SelectedRuntimePlan,
    parse_selected_runtime_plan,
)
from eqvae.training.selected_runtime_runner import SelectedRuntimeTrainRequest

_FULL_CONFIG = Path("configs/spec0001/non_eq_vae_selected_runtime_full.json")
_RUNTIME_CONFIG = Path(
    "runs/kaggle/runtime_selection_v5/benchmark/selected_runtime.json",
)
_FULL_TARGET_UPDATES = 125000
_FULL_EPOCHS = 10
_FULL_UPDATES_PER_EPOCH = 12500
_FULL_HALF_EPOCH_INTERVAL = 6250
_FULL_VALIDATION_BATCHES_PER_VIEW = 20
_LOCAL_DRY_RUN_STEPS = 2
_PRIOR_BEST_VALIDATION_METRIC = 0.25
_FLOAT_TOLERANCE = 1e-12
_EXPECTED_ROW_ID = (
    "dual_t4_ddp__bs12__amp_conservative__compile_none__indexed_masked__"
    "policy_amp_fp16_conservative"
)


class _FullOutputContract(NamedTuple):
    """Tiny verifier contract used to test strict full-output checks cheaply."""

    target_updates: int
    epochs: int
    updates_per_epoch: int
    half_interval: int
    validation_batches: int
    keep_count: int
    world_size: int


def test_full_config_derives_exact_spec0009_schedule(tmp_path: Path) -> None:
    """Full mode resolves the v5 runtime into 125000 target updates."""
    plan = parse_selected_runtime_plan(_RUNTIME_CONFIG)
    settings = selected_runtime_runner._settings(  # noqa: SLF001
        request=SelectedRuntimeTrainRequest(
            config_path=_FULL_CONFIG,
            runtime_config=_RUNTIME_CONFIG,
            output_dir=tmp_path,
            run_name="spec0009_test",
            data="synthetic",
            max_train_steps=_LOCAL_DRY_RUN_STEPS,
            save_every_steps=1,
            dry_run=True,
        ),
        resolved=resolve_json_config(_FULL_CONFIG),
        plan=plan,
    )

    assert plan.optimizer_updates_per_epoch == _FULL_UPDATES_PER_EPOCH
    assert settings.run_mode == "kaggle_selected_runtime_full_train"
    assert settings.max_train_steps == _LOCAL_DRY_RUN_STEPS
    assert settings.target_train_steps == _FULL_TARGET_UPDATES
    assert settings.requested_epochs == _FULL_EPOCHS
    assert settings.optimizer_updates_per_epoch == _FULL_UPDATES_PER_EPOCH
    assert settings.half_epoch_interval_steps == _FULL_HALF_EPOCH_INTERVAL
    assert settings.validation_batches_per_view == _FULL_VALIDATION_BATCHES_PER_VIEW
    assert settings.validation_views == ("clean", "deterministic_denoising")
    assert settings.train_reparameterization == "stochastic_seeded"
    assert not selected_runtime_runner._should_run_scheduled_validation(  # noqa: SLF001
        settings,
        _LOCAL_DRY_RUN_STEPS,
    )
    assert selected_runtime_runner._should_run_scheduled_validation(  # noqa: SLF001
        settings,
        _FULL_HALF_EPOCH_INTERVAL,
    )


def test_full_run_beta_warmup_is_pinned_to_one_epoch(tmp_path: Path) -> None:
    """FU-003: the full-run guard resolves beta warmup to exactly one epoch.

    It resolves against target_train_steps, so a shortened --dry-run
    max_train_steps does not lower the resolved warmup below one epoch.
    """
    settings = _full_settings(tmp_path=tmp_path, max_train_steps=2, save_every=1)

    assert (
        selected_runtime_runner._resolved_beta_warmup_steps(settings)  # noqa: SLF001
        == _FULL_UPDATES_PER_EPOCH
    )
    # Does not raise for the real, pinned schedule (uses target, not max=2).
    selected_runtime_runner._validate_full_run_settings(  # noqa: SLF001
        settings,
        dry_run=True,
    )


def test_full_run_rejects_beta_warmup_not_one_epoch(tmp_path: Path) -> None:
    """FU-003: a warmup fraction that desyncs from one epoch fails closed."""
    settings = _full_settings(tmp_path=tmp_path, max_train_steps=2, save_every=1)
    desynced = replace(settings, beta_warmup_fraction=0.2)

    with pytest.raises(ValueError, match="beta warmup must span exactly one epoch"):
        selected_runtime_runner._validate_full_run_settings(  # noqa: SLF001
            desynced,
            dry_run=True,
        )


def test_full_boundary_logging_waits_at_barrier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Full-run boundaries leave log breadcrumbs and synchronize ranks."""
    settings = _full_settings(tmp_path=tmp_path, max_train_steps=2, save_every=1)
    distributed = _local_distributed_context()
    barrier_ranks: list[int] = []

    def fake_barrier(observed: selected_runtime_runner._DistributedContext) -> None:
        barrier_ranks.append(observed.rank)

    monkeypatch.setattr(selected_runtime_runner, "_barrier", fake_barrier)

    selected_runtime_runner._log_full_boundary_start(  # noqa: SLF001
        settings=settings,
        distributed=distributed,
        optimizer_step=_FULL_HALF_EPOCH_INTERVAL,
    )
    selected_runtime_runner._synchronize_full_boundary_completion(  # noqa: SLF001
        settings=settings,
        distributed=distributed,
        optimizer_step=_FULL_HALF_EPOCH_INTERVAL,
    )
    debug_settings = replace(settings, run_mode="kaggle_selected_runtime_debug_train")
    selected_runtime_runner._synchronize_full_boundary_completion(  # noqa: SLF001
        settings=debug_settings,
        distributed=distributed,
        optimizer_step=_FULL_HALF_EPOCH_INTERVAL,
    )

    output = capsys.readouterr().out
    assert "selected-runtime full boundary start" in output
    assert "validation_views=clean,deterministic_denoising" in output
    assert "selected-runtime full boundary complete" in output
    assert "selected-runtime full boundary barrier resolved" in output
    assert barrier_ranks == [0]


def test_full_config_refuses_missing_max_train_steps(tmp_path: Path) -> None:
    """Full mode fails closed instead of falling back to one optimizer step."""
    payload = cast(
        "dict[str, object]",
        json.loads(_FULL_CONFIG.read_text(encoding="utf-8")),
    )
    payload["source_config"] = str(
        Path("configs/spec0001/non_eq_vae_model_base.json").resolve(),
    )
    training = cast("dict[str, object]", payload["training"])
    training.pop("max_train_steps")
    config_path = tmp_path / "missing_max_train_steps.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="must declare max_train_steps"):
        selected_runtime_runner._settings(  # noqa: SLF001
            request=SelectedRuntimeTrainRequest(
                config_path=config_path,
                runtime_config=_RUNTIME_CONFIG,
                output_dir=tmp_path,
                run_name="spec0009_test",
                data="synthetic",
                dry_run=True,
            ),
            resolved=resolve_json_config(config_path),
            plan=parse_selected_runtime_plan(_RUNTIME_CONFIG),
        )


def test_full_train_eps_uses_seeded_stochastic_generator(tmp_path: Path) -> None:
    """Full-run training epsilon is nonzero and generated from the train generator."""
    settings = selected_runtime_runner._settings(  # noqa: SLF001
        request=SelectedRuntimeTrainRequest(
            config_path=_FULL_CONFIG,
            runtime_config=_RUNTIME_CONFIG,
            output_dir=tmp_path,
            run_name="spec0009_test",
            data="synthetic",
            max_train_steps=1,
            save_every_steps=1,
            dry_run=True,
        ),
        resolved=resolve_json_config(_FULL_CONFIG),
        plan=parse_selected_runtime_plan(_RUNTIME_CONFIG),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(settings.data_seed)

    eps, proof = selected_runtime_runner._train_eps(  # noqa: SLF001
        batch_size=_LOCAL_DRY_RUN_STEPS,
        settings=settings,
        train_generator=generator,
        device=torch.device("cpu"),
    )

    assert eps.shape == (2, 16, 32, 32)
    assert proof.eps_policy == "stochastic_seeded_train_generator"
    assert proof.eps_seed_source == "train_data_torch_generator"
    assert 0.0 <= float(proof.eps_zero_fraction) < 1.0
    assert float(proof.eps_abs_mean) > 0.0


def test_eps_generator_seed_is_per_rank_and_preserves_single_rank() -> None:
    """FU-007/FU-012: rank offset diverges eps; rank 0 fresh keeps data_seed."""
    data_seed = 4242
    ranks = (0, 1)
    seed = selected_runtime_runner._eps_generator_seed  # noqa: SLF001

    # Rank 0 fresh reduces to data_seed, so world_size==1 values are unchanged.
    assert seed(data_seed=data_seed, rank=0) == data_seed
    fresh = {seed(data_seed=data_seed, rank=rank) for rank in ranks}
    assert len(fresh) == len(ranks)
    # A resume folds start_step so ranks stay distinct AND the post-resume stream
    # does not repeat the fresh stream (FU-012).
    resumed = {
        seed(data_seed=data_seed, rank=rank, start_step=_FULL_HALF_EPOCH_INTERVAL)
        for rank in ranks
    }
    assert len(resumed) == len(ranks)
    assert resumed.isdisjoint(fresh)


def test_train_eps_diverges_across_ranks_but_rank0_matches_legacy(
    tmp_path: Path,
) -> None:
    """FU-007: DDP ranks draw independent eps; rank 0 matches the pre-fix stream."""
    settings = _full_settings(tmp_path=tmp_path, max_train_steps=1, save_every=1)

    def draw(seed_value: int) -> tuple[torch.Tensor, float]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed_value)
        eps, proof = selected_runtime_runner._train_eps(  # noqa: SLF001
            batch_size=12,
            settings=settings,
            train_generator=generator,
            device=torch.device("cpu"),
        )
        return eps, float(proof.eps_abs_mean)

    seed = selected_runtime_runner._eps_generator_seed  # noqa: SLF001
    rank0_eps, rank0_mean = draw(seed(data_seed=settings.data_seed, rank=0))
    rank1_eps, rank1_mean = draw(seed(data_seed=settings.data_seed, rank=1))
    legacy_eps, _ = draw(settings.data_seed)

    # Rank 0 fresh eps is bit-identical to the pre-fix single-rank stream.
    assert torch.equal(rank0_eps, legacy_eps)
    # Ranks draw independent eps (never a shared z) with distinct abs-mean.
    assert not torch.equal(rank0_eps, rank1_eps)
    assert rank0_mean != rank1_mean


def test_per_rank_eps_divergent_flags_collapsed_eps() -> None:
    """FU-007 gate-health: identical per-rank eps_abs_mean is flagged as collapse."""
    divergent = [
        _eps_metric_row(rank=0, step=1, eps_abs_mean="0.80"),
        _eps_metric_row(rank=1, step=1, eps_abs_mean="0.81"),
    ]
    collapsed = [
        _eps_metric_row(rank=0, step=1, eps_abs_mean="0.80"),
        _eps_metric_row(rank=1, step=1, eps_abs_mean="0.80"),
    ]
    single = [_eps_metric_row(rank=0, step=1, eps_abs_mean="0.80")]

    assert selected_runtime_runner._per_rank_eps_divergent(divergent)  # noqa: SLF001
    # The FU-007 bug (shared eps stream) records bit-identical means -> flagged.
    assert not selected_runtime_runner._per_rank_eps_divergent(collapsed)  # noqa: SLF001
    # A single-process run has nothing to diverge, so it passes trivially.
    assert selected_runtime_runner._per_rank_eps_divergent(single)  # noqa: SLF001


def test_resume_reapplies_per_rank_eps_offset_under_ddp(tmp_path: Path) -> None:
    """FU-012: a resumed DDP run re-diverges eps; single-rank resume is a no-op."""
    settings = _full_settings(tmp_path=tmp_path, max_train_steps=1, save_every=1)
    loaded = _loaded_checkpoint_stub()

    def restored_generator() -> torch.Generator:
        # Every rank restores rank-0's saved generator state on resume (FU-012).
        generator = torch.Generator(device="cpu")
        generator.manual_seed(settings.data_seed)
        return generator

    def draw(generator: torch.Generator) -> torch.Tensor:
        eps, _ = selected_runtime_runner._train_eps(  # noqa: SLF001
            batch_size=12,
            settings=settings,
            train_generator=generator,
            device=torch.device("cpu"),
        )
        return eps

    def resume(*, rank: int, should_use_ddp: bool) -> torch.Tensor:
        generator = restored_generator()
        distributed = (
            _ddp_distributed_context(rank=rank)
            if should_use_ddp
            else _local_distributed_context()
        )
        selected_runtime_runner._reapply_per_rank_eps_offset_on_resume(  # noqa: SLF001
            train_generator=generator,
            settings=settings,
            distributed=distributed,
            loaded_checkpoint=loaded,
            start_step=_FULL_HALF_EPOCH_INTERVAL,
        )
        return draw(generator)

    restored_stream = draw(restored_generator())
    ddp_rank0 = resume(rank=0, should_use_ddp=True)
    ddp_rank1 = resume(rank=1, should_use_ddp=True)
    single_rank = resume(rank=0, should_use_ddp=False)

    # Post-resume DDP ranks draw distinct eps (no collapse to rank-0's stream)...
    assert not torch.equal(ddp_rank0, ddp_rank1)
    # ...and the re-based stream does not repeat the restored stream.
    assert not torch.equal(ddp_rank0, restored_stream)
    # Single-rank resume keeps the exact restored continuous stream (unchanged).
    assert torch.equal(single_rank, restored_stream)


def test_boundary_selection_metric_uses_denoising_view_not_clean() -> None:
    """FU-008: best is selected on the denoising view, not the easier clean view."""
    rows = (
        _selection_boundary_row(view="clean", l1="0.10", sample_count="240"),
        _selection_boundary_row(
            view="deterministic_denoising",
            l1="0.50",
            sample_count="240",
        ),
    )

    metric = selected_runtime_runner._boundary_selection_metric(  # noqa: SLF001
        boundary_rows=rows,
        distributed=_local_distributed_context(),
    )

    # min-over-views (the FU-008 bug) would return the easier clean 0.10.
    assert metric is not None
    assert abs(metric - 0.50) < _FLOAT_TOLERANCE


def test_boundary_selection_metric_is_cross_rank_sample_weighted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FU-008: selection reduces sum(l1*n)/sum(n) for denoising across ranks."""
    rank1_denoising_weighted = 0.6 * 120
    rank1_sample_count = 120

    def fake_is_initialized() -> bool:
        return True

    def fake_all_gather_object(gathered: list[object], obj: object) -> None:
        gathered[0] = obj
        gathered[1] = (rank1_denoising_weighted, rank1_sample_count)

    monkeypatch.setattr(
        selected_runtime_runner.dist,
        "is_initialized",
        fake_is_initialized,
    )
    monkeypatch.setattr(
        selected_runtime_runner.dist,
        "all_gather_object",
        fake_all_gather_object,
    )

    rank0_rows = (
        _selection_boundary_row(view="clean", l1="0.10", sample_count="240", rank=0),
        _selection_boundary_row(
            view="deterministic_denoising",
            l1="0.40",
            sample_count="240",
            rank=0,
        ),
    )
    metric = selected_runtime_runner._boundary_selection_metric(  # noqa: SLF001
        boundary_rows=rank0_rows,
        distributed=_ddp_distributed_context(rank=0),
    )

    expected = (0.40 * 240 + rank1_denoising_weighted) / (240 + rank1_sample_count)
    assert metric is not None
    assert abs(metric - expected) < _FLOAT_TOLERANCE
    # Mutation guards: rank-0-only (0.40) and average-of-averages (0.50) are wrong.
    assert abs(metric - 0.40) > _FLOAT_TOLERANCE
    assert abs(metric - 0.50) > _FLOAT_TOLERANCE
    # World_size independence: one rank over all 360 samples selects the same value.
    single_rank_metric = selected_runtime_runner._boundary_selection_metric(  # noqa: SLF001
        boundary_rows=(
            _selection_boundary_row(
                view="deterministic_denoising",
                l1=f"{expected:.12f}",
                sample_count="360",
            ),
        ),
        distributed=_local_distributed_context(),
    )
    assert single_rank_metric is not None
    assert abs(single_rank_metric - expected) < _FLOAT_TOLERANCE


def test_synchronized_amp_step_skipped_agrees_or_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AMP-skip is gathered every step so ranks cannot desync at a boundary (FU-020)."""

    def fake_is_initialized() -> bool:
        return True

    monkeypatch.setattr(
        selected_runtime_runner.dist,
        "is_initialized",
        fake_is_initialized,
    )

    # A single-process run skips the collective entirely (no behavior change).
    assert not selected_runtime_runner._synchronized_amp_step_skipped(  # noqa: SLF001
        local_amp_step_skipped=False,
        distributed=_local_distributed_context(),
    )

    def agree(gathered: list[object], obj: object) -> None:
        gathered[0] = obj
        gathered[1] = obj

    monkeypatch.setattr(selected_runtime_runner.dist, "all_gather_object", agree)
    # When ranks agree, the local decision is returned unchanged.
    assert selected_runtime_runner._synchronized_amp_step_skipped(  # noqa: SLF001
        local_amp_step_skipped=True,
        distributed=_ddp_distributed_context(rank=0),
    )

    def disagree(gathered: list[object], obj: object) -> None:
        gathered[0] = obj
        gathered[1] = not bool(obj)

    monkeypatch.setattr(selected_runtime_runner.dist, "all_gather_object", disagree)
    # Divergent skip decisions fail fast (all ranks raise) instead of deadlocking
    # at the next boundary collective.
    with pytest.raises(RuntimeError, match="disagree on the AMP step-skip decision"):
        selected_runtime_runner._synchronized_amp_step_skipped(  # noqa: SLF001
            local_amp_step_skipped=False,
            distributed=_ddp_distributed_context(rank=0),
        )


def test_run_train_steps_selects_best_on_denoising_view_end_to_end(  # noqa: PLR0914
    tmp_path: Path,
) -> None:
    """FU-008 end-to-end: best_model.pt is saved from the denoising-view metric."""
    output_dir = tmp_path / "best_selection"
    request = SelectedRuntimeTrainRequest(
        config_path=_FULL_CONFIG,
        runtime_config=_RUNTIME_CONFIG,
        output_dir=output_dir,
        run_name="spec0009_best_selection",
        data="synthetic",
        max_train_steps=2,
        save_every_steps=1,
        dry_run=True,
    )
    plan = parse_selected_runtime_plan(_RUNTIME_CONFIG)
    resolved = resolve_json_config(_FULL_CONFIG)
    # Force validation at every step so a 2-step local run exercises best-selection.
    settings = replace(
        selected_runtime_runner._settings(  # noqa: SLF001
            request=request,
            resolved=resolved,
            plan=plan,
        ),
        half_epoch_interval_steps=1,
        validation_batches_per_view=1,
    )
    local = _local_distributed_context()
    data_surface = selected_runtime_runner._prepare_data_surface(  # noqa: SLF001
        request=request,
        settings=settings,
        plan=plan,
        distributed=local,
    )
    try:
        (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        model = selected_runtime_runner.build_non_equivariant_vae(
            norm_groups=settings.norm_groups,
        )
        optimizer, _ = selected_runtime_runner.create_adamw_optimizer(
            model,
            config=settings.optimizer_config,
        )
        amp = selected_runtime_runner._amp_execution(  # noqa: SLF001
            plan=plan,
            distributed=local,
            dry_run=True,
        )
        scaler = selected_runtime_runner.GradScaler(
            "cuda",
            init_scale=amp.grad_scaler_init_scale,
            enabled=amp.grad_scaler_enabled,
        )
        train_generator = torch.Generator(device="cpu")
        train_generator.manual_seed(settings.data_seed)
        train_loop = selected_runtime_runner._run_train_steps(  # noqa: SLF001
            request=request,
            resolved=resolved,
            settings=settings,
            plan=plan,
            model=model,
            checkpoint_model=model,
            optimizer=optimizer,
            scaler=scaler,
            amp=amp,
            data_surface=data_surface,
            distributed=local,
            numpy_generator=np.random.default_rng(settings.global_seed),
            train_generator=train_generator,
            runtime_identity=selected_runtime_runner._runtime_identity(plan),  # noqa: SLF001
            start_step=0,
            initial_best_validation_metric=None,
            resume_history=selected_runtime_runner._ResumeArtifactHistory(  # noqa: SLF001
                metric_rows=(),
                validation_rows=(),
                interval_checkpoints=(),
                best_checkpoint=None,
                best_validation_metric=None,
            ),
            write_checkpoints=True,
            interval_flush=None,
            fixed25=None,
        )
    finally:
        selected_runtime_runner._close_data_surface(data_surface)  # noqa: SLF001

    # best_model.pt was saved from a validation boundary (not the train-L1 fallback)...
    assert train_loop.best_validation_checkpoint is not None
    assert (output_dir / "checkpoints" / "best_model.pt").exists()
    assert train_loop.best_validation_metric is not None
    denoising_l1 = [
        float(row["l1_loss"])
        for row in train_loop.validation_rows
        if row["view"] == "deterministic_denoising"
    ]
    clean_l1 = [
        float(row["l1_loss"])
        for row in train_loop.validation_rows
        if row["view"] == "clean"
    ]
    assert denoising_l1
    # ...and selected on the DENOISING view: the best metric is the min denoising-view
    # L1, NOT the easier clean view (the FU-008 bug) or a train-loss value.
    assert abs(train_loop.best_validation_metric - min(denoising_l1)) < _FLOAT_TOLERANCE
    assert abs(min(clean_l1) - min(denoising_l1)) > _FLOAT_TOLERANCE


class _ValidationScaffold(NamedTuple):
    """Minimal setup for a direct ``_validation_view_row`` call (FU-017)."""

    settings: selected_runtime_runner._RunnerSettings
    plan: SelectedRuntimePlan
    amp: selected_runtime_runner._AmpExecution
    data_surface: selected_runtime_runner._DataSurface
    model: torch.nn.Module


def _open_validation_scaffold(tmp_path: Path) -> _ValidationScaffold:
    """Build the model and CPU data surface used by the FU-017 validation tests.

    Returns:
        The validation scaffold; the caller must close its data surface.

    """
    request = SelectedRuntimeTrainRequest(
        config_path=_FULL_CONFIG,
        runtime_config=_RUNTIME_CONFIG,
        output_dir=tmp_path / "validation",
        run_name="spec0009_validation_repro",
        data="synthetic",
        max_train_steps=2,
        save_every_steps=1,
        dry_run=True,
    )
    plan = parse_selected_runtime_plan(_RUNTIME_CONFIG)
    settings = replace(
        selected_runtime_runner._settings(  # noqa: SLF001
            request=request,
            resolved=resolve_json_config(_FULL_CONFIG),
            plan=plan,
        ),
        half_epoch_interval_steps=1,
        validation_batches_per_view=1,
    )
    local = _local_distributed_context()
    data_surface = selected_runtime_runner._prepare_data_surface(  # noqa: SLF001
        request=request,
        settings=settings,
        plan=plan,
        distributed=local,
    )
    scaffold_built = False
    try:
        model = build_non_equivariant_vae(norm_groups=settings.norm_groups)
        model.eval()
        amp = selected_runtime_runner._amp_execution(  # noqa: SLF001
            plan=plan,
            distributed=local,
            dry_run=True,
        )
        scaffold = _ValidationScaffold(
            settings=settings,
            plan=plan,
            amp=amp,
            data_surface=data_surface,
            model=model,
        )
        scaffold_built = True
    finally:
        if not scaffold_built:
            selected_runtime_runner._close_data_surface(data_surface)  # noqa: SLF001
    return scaffold


def _validation_row(
    scaffold: _ValidationScaffold,
    *,
    view: str,
    optimizer_step: int,
) -> Mapping[str, str]:
    return selected_runtime_runner._validation_view_row(  # noqa: SLF001
        model=scaffold.model,
        settings=scaffold.settings,
        plan=scaffold.plan,
        amp=scaffold.amp,
        data_surface=scaffold.data_surface,
        optimizer_step=optimizer_step,
        view=view,
        rank=0,
        device=torch.device("cpu"),
    )


def test_clean_validation_view_consumes_no_corruption_rng(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FU-017: the clean validation view never enters the corruption machinery."""
    scaffold = _open_validation_scaffold(tmp_path)

    def _forbid_corruption(*_args: object, **_kwargs: object) -> NoReturn:
        message = "corrupt_normalized_batch was invoked"
        raise RuntimeError(message)

    monkeypatch.setattr(
        selected_runtime_runner,
        "corrupt_normalized_batch",
        _forbid_corruption,
    )
    try:
        # The clean view is a pure passthrough: the corruption stub must not fire.
        clean_row = _validation_row(scaffold, view="clean", optimizer_step=1)
        assert clean_row["view"] == "clean"
        # The same stub proves the denoising view DOES invoke corruption.
        with pytest.raises(RuntimeError, match="corrupt_normalized_batch was invoked"):
            _validation_row(scaffold, view="deterministic_denoising", optimizer_step=1)
    finally:
        selected_runtime_runner._close_data_surface(scaffold.data_surface)  # noqa: SLF001


def test_deterministic_denoising_validation_row_is_reproducible(
    tmp_path: Path,
) -> None:
    """FU-017: the deterministic_denoising validation row is byte-reproducible.

    Corruption is a pure function of (seed, split, semantic key, step, view), eps is
    zero, and the model is unchanged, so two runs over the same shuffle-false loader
    at the same step must produce byte-identical rows. A non-vacuity control asserts
    the corrupted input actually moves the row (clean vs denoising metrics differ),
    so the byte-equality cannot pass silently if the model output becomes
    input-independent.
    """
    scaffold = _open_validation_scaffold(tmp_path)
    try:
        first = _validation_row(
            scaffold,
            view="deterministic_denoising",
            optimizer_step=1,
        )
        second = _validation_row(
            scaffold,
            view="deterministic_denoising",
            optimizer_step=1,
        )
        clean = _validation_row(scaffold, view="clean", optimizer_step=1)
    finally:
        selected_runtime_runner._close_data_surface(scaffold.data_surface)  # noqa: SLF001

    assert first == second
    # Non-vacuity control: at the same step the ONLY difference between the clean and
    # denoising rows is the corrupted input, so their loss metrics must differ.
    metric_keys = (
        "loss",
        "recon_loss",
        "l1_loss",
        "ssim_loss",
        "ssim_metric",
        "kl_loss",
    )
    assert {key: first[key] for key in metric_keys} != {
        key: clean[key] for key in metric_keys
    }


def test_validation_beta_uses_training_denominator_under_dry_run(
    tmp_path: Path,
) -> None:
    """FU-022: validation beta shares the training denominator (max_train_steps).

    Under --dry-run max_train_steps << target_train_steps, so the two denominators
    diverge; this pins the validation row's beta to the max-based (training) value
    rather than the target-based value.
    """
    scaffold = _open_validation_scaffold(tmp_path)  # dry-run: max_train_steps == 2
    try:
        row = _validation_row(scaffold, view="clean", optimizer_step=2)
    finally:
        selected_runtime_runner._close_data_surface(scaffold.data_surface)  # noqa: SLF001

    settings = scaffold.settings
    max_based = beta_for_step(
        optimizer_step_index=1,
        max_optimizer_steps=settings.max_train_steps,
        target_beta=settings.beta_target,
        warmup_fraction=settings.beta_warmup_fraction,
    )
    target_based = beta_for_step(
        optimizer_step_index=1,
        max_optimizer_steps=settings.target_train_steps,
        target_beta=settings.beta_target,
        warmup_fraction=settings.beta_warmup_fraction,
    )
    assert math.isclose(float(row["beta"]), max_based)
    # The denominators genuinely diverge under --dry-run, so the fix is observable.
    assert not math.isclose(max_based, target_based)


def test_fresh_full_run_flushes_metrics_at_first_boundary(  # noqa: PLR0914
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FU-039: a fresh full run persists metrics at the FIRST half-epoch boundary.

    The first paper-promotable full run restarts from scratch (``start_step == 0``,
    no resume). This drives ``_run_train_steps`` fresh with a real interval-flush
    context and proves the interval flush writes train/validation metric rows at the
    first boundary (update 1), mid-loop, before any final teardown — so a Kaggle
    cancellation after the first boundary still leaves a recoverable partial curve
    (the exact v1 data-loss class this restart depends on not recurring).
    """
    output_dir = tmp_path / "fresh_first_boundary"
    request = SelectedRuntimeTrainRequest(
        config_path=_FULL_CONFIG,
        runtime_config=_RUNTIME_CONFIG,
        output_dir=output_dir,
        run_name="spec0009_fresh_first_boundary",
        data="synthetic",
        max_train_steps=2,
        save_every_steps=1,
        dry_run=True,
    )
    plan = parse_selected_runtime_plan(_RUNTIME_CONFIG)
    resolved = resolve_json_config(_FULL_CONFIG)
    # Force a boundary at every step so a 2-step local run exercises two boundaries.
    settings = replace(
        selected_runtime_runner._settings(  # noqa: SLF001
            request=request,
            resolved=resolved,
            plan=plan,
        ),
        half_epoch_interval_steps=1,
        validation_batches_per_view=1,
    )
    distributed = _local_distributed_context()
    data_surface = selected_runtime_runner._prepare_data_surface(  # noqa: SLF001
        request=request,
        settings=settings,
        plan=plan,
        distributed=distributed,
    )

    # Spy the interval flush: the run never calls the final-teardown writer here, so
    # the ONLY writer of the metric CSVs is this mid-loop flush. Recording each call
    # proves the FIRST flush happened at boundary 1 (not only at the final boundary),
    # which on-disk state alone cannot show because each flush rewrites the whole CSV.
    flush_calls: list[dict[str, object]] = []
    original_flush = selected_runtime_runner._write_interval_artifact_flush  # noqa: SLF001

    def _spy_flush(
        *,
        context: selected_runtime_runner._IntervalFlushContext,
        model: torch.nn.Module,
        local_state: selected_runtime_runner._IntervalFlushState,
    ) -> None:
        original_flush(context=context, model=model, local_state=local_state)
        flush_calls.append(
            {
                "current_step": local_state.current_step,
                "train_csv_exists": context.artifacts.train_steps.exists(),
                "max_update": max(
                    (
                        int(row["successful_optimizer_update_count"])
                        for row in local_state.metric_rows
                    ),
                    default=0,
                ),
            },
        )

    monkeypatch.setattr(
        selected_runtime_runner,
        "_write_interval_artifact_flush",
        _spy_flush,
    )

    artifacts = selected_runtime_runner._artifact_paths(output_dir)  # noqa: SLF001
    launch_command = selected_runtime_runner.build_selected_runtime_torchrun_command(
        config_path=_FULL_CONFIG,
        runtime_config=_RUNTIME_CONFIG,
        data="synthetic",
        output_dir=output_dir,
        run_name=request.run_name,
        max_train_steps=2,
        save_every_steps=1,
        dry_run=True,
    )
    interval_flush = selected_runtime_runner._IntervalFlushContext(  # noqa: SLF001
        artifacts=artifacts,
        request=request,
        settings=settings,
        plan=plan,
        runtime_identity=selected_runtime_runner._runtime_identity(plan),  # noqa: SLF001
        launch_command=launch_command,
        ddp_proof=selected_runtime_runner.build_ddp_rank_device_proof(
            plan=plan,
            probe=distributed.probe,
            launch_command=launch_command,
            dry_run=True,
        ),
        amp=selected_runtime_runner._amp_execution(  # noqa: SLF001
            plan=plan,
            distributed=distributed,
            dry_run=True,
        ),
        data_surface=data_surface,
        distributed=distributed,
    )
    try:
        (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        model = selected_runtime_runner.build_non_equivariant_vae(
            norm_groups=settings.norm_groups,
        )
        optimizer, _ = selected_runtime_runner.create_adamw_optimizer(
            model,
            config=settings.optimizer_config,
        )
        amp = selected_runtime_runner._amp_execution(  # noqa: SLF001
            plan=plan,
            distributed=distributed,
            dry_run=True,
        )
        scaler = selected_runtime_runner.GradScaler(
            "cuda",
            init_scale=amp.grad_scaler_init_scale,
            enabled=amp.grad_scaler_enabled,
        )
        train_generator = torch.Generator(device="cpu")
        train_generator.manual_seed(settings.data_seed)
        selected_runtime_runner._run_train_steps(  # noqa: SLF001
            request=request,
            resolved=resolved,
            settings=settings,
            plan=plan,
            model=model,
            checkpoint_model=model,
            optimizer=optimizer,
            scaler=scaler,
            amp=amp,
            data_surface=data_surface,
            distributed=distributed,
            numpy_generator=np.random.default_rng(settings.global_seed),
            train_generator=train_generator,
            runtime_identity=selected_runtime_runner._runtime_identity(plan),  # noqa: SLF001
            start_step=0,
            initial_best_validation_metric=None,
            resume_history=selected_runtime_runner._ResumeArtifactHistory(  # noqa: SLF001
                metric_rows=(),
                validation_rows=(),
                interval_checkpoints=(),
                best_checkpoint=None,
                best_validation_metric=None,
            ),
            write_checkpoints=True,
            interval_flush=interval_flush,
            fixed25=None,
        )
    finally:
        selected_runtime_runner._close_data_surface(data_surface)  # noqa: SLF001

    # The final-teardown writer (_write_final_artifacts) was never called, so any
    # metric CSV on disk was produced by the mid-loop interval flush of a fresh run.
    assert artifacts.train_steps.exists()
    assert artifacts.validation_metrics.exists()
    with artifacts.train_steps.open(encoding="utf-8", newline="") as handle:
        train_rows = list(csv.DictReader(handle))
    train_updates = {row["successful_optimizer_update_count"] for row in train_rows}
    # FU-018: the runner writes finite decoder-head saturation telemetry per row.
    assert train_rows
    for column in (
        "recon_output_rms",
        "x_hat_min",
        "x_hat_max",
        "frac_x_hat_lt_minus1",
        "frac_x_hat_gt_1",
    ):
        assert all(math.isfinite(float(row[column])) for row in train_rows)
    # The zero-init head is exactly 0.0 at update 1, but a trained update produces
    # nonzero output. Requiring at least one nonzero row fails if the
    # _run_train_step -> result -> _metric_row wiring regresses to the 0.0 defaults.
    assert any(float(row["recon_output_rms"]) > 0.0 for row in train_rows)
    assert any(float(row["x_hat_max"]) > 0.0 for row in train_rows)
    with artifacts.validation_metrics.open(encoding="utf-8", newline="") as handle:
        validation_steps = {row["optimizer_step"] for row in csv.DictReader(handle)}
    # Both boundaries are persisted; update 1 (the first half-epoch boundary) included.
    assert {"1", "2"} <= train_updates
    assert "1" in validation_steps

    # The spy proves the FIRST flush ran at boundary 1 and had already written the
    # train CSV then, so the first-boundary rows survive a cancel before boundary 2.
    assert flush_calls
    assert flush_calls[0]["current_step"] == 1
    assert flush_calls[0]["train_csv_exists"] is True
    assert flush_calls[0]["max_update"] == 1


def test_full_loaded_resume_proof_records_restore_attempts(tmp_path: Path) -> None:
    """The resumed full-run proof carries the stricter restore-attempt evidence."""
    plan = parse_selected_runtime_plan(_RUNTIME_CONFIG)
    resolved = resolve_json_config(_FULL_CONFIG)
    request = SelectedRuntimeTrainRequest(
        config_path=_FULL_CONFIG,
        runtime_config=_RUNTIME_CONFIG,
        output_dir=tmp_path / "full_output",
        run_name="spec0009_resume_test",
        data="synthetic",
        max_train_steps=1,
        save_every_steps=1,
        dry_run=True,
    )
    settings = selected_runtime_runner._settings(  # noqa: SLF001
        request=request,
        resolved=resolved,
        plan=plan,
    )
    runtime_identity = selected_runtime_runner._runtime_identity(plan)  # noqa: SLF001
    loaded = LoadedCheckpoint(
        path=request.output_dir / "checkpoints" / "step_006250.pt",
        schema_version="spec0001.checkpoint.v5",
        run_name="spec0009_resume_source",
        config_path=str(_FULL_CONFIG),
        config_sha256=resolved.invoked_config_hash,
        effective_config_sha256=resolved.effective_config_hash,
        runtime_config_sha256=runtime_identity.sha256,
        selected_row_id=runtime_identity.selected_row_id,
        runtime_policy_id=runtime_identity.runtime_policy_id,
        lr_scheduler_state_status="not_applicable_local_debug_no_scheduler",
        beta_progress_state_status=(
            "deterministic_from_successful_optimizer_update_count"
        ),
        amp_scaler_state_status="selected_runtime_amp_scaler_state",
        torch_cuda_rng_state_status="selected_runtime_cuda_rng_state",
        ddp_sampler_progress_state_status="selected_runtime_ddp_sampler_progress",
        optimizer_step=6250,
        successful_optimizer_update_count=6250,
        metric_name="validation_l1_loss",
        metric_value=0.5,
        torch_generator_names=("train_data",),
    )
    distributed = selected_runtime_runner._DistributedContext(  # noqa: SLF001
        device=torch.device("cuda"),
        rank=0,
        local_rank=0,
        world_size=2,
        nproc_per_node=2,
        should_use_ddp=True,
        initialized_here=True,
        probe=selected_runtime_runner.SelectedRuntimeEnvironmentProbe(
            machine_shape="NvidiaTeslaT4",
            accelerator_mode="dual_t4_ddp",
            cuda_device_count=2,
            visible_device_count=2,
            gpu_names=("Tesla T4", "Tesla T4"),
            world_size=2,
            nproc_per_node=2,
            rank=0,
            local_rank=0,
            torchrun_standalone=True,
            rank_assignments=(),
            distributed_initialized=True,
        ),
    )
    amp = selected_runtime_runner._AmpExecution(  # noqa: SLF001
        enabled=True,
        grad_scaler_enabled=True,
        grad_scaler_init_scale=16384.0,
        autocast_dtype="float16",
        requested_autocast_dtype="float16",
        local_amp_status="selected_runtime_cuda_amp_enabled",
    )

    proof = selected_runtime_runner._loaded_checkpoint_resume_proof(  # noqa: SLF001
        loaded=loaded,
        request=request,
        resolved=resolved,
        settings=settings,
        runtime_identity=runtime_identity,
        amp=amp,
        distributed=distributed,
    )

    assert proof["status"] == "local_pass"
    assert proof["grad_scaler_state_restore_attempted"] is True
    assert proof["grad_scaler_state_restored"] is True
    assert proof["cuda_rng_state_restore_attempted"] is True
    assert proof["cuda_rng_state_restored"] is True


def test_full_dry_run_summary_lists_only_interval_checkpoints(
    tmp_path: Path,
) -> None:
    """Full summary keeps final/best separate from retained interval checkpoints."""
    output_dir = tmp_path / "full-dry-run"

    assert (
        selected_runtime_train_main(
            [
                "--config",
                str(_FULL_CONFIG),
                "--runtime-config",
                str(_RUNTIME_CONFIG),
                "--data",
                "synthetic",
                "--output-dir",
                str(output_dir),
                "--run-name",
                "spec0009_full_dry_run_summary",
                "--max-train-steps",
                str(_LOCAL_DRY_RUN_STEPS),
                "--max-val-steps",
                "1",
                "--save-every-steps",
                "1",
                "--dry-run",
            ],
        )
        == 0
    )

    full_summary = cast(
        "dict[str, object]",
        json.loads(
            (output_dir / "benchmark" / "selected_runtime_full_summary.json").read_text(
                encoding="utf-8",
            ),
        ),
    )
    retained_full_names = cast(
        "list[str]",
        full_summary["retained_interval_checkpoints"],
    )
    assert retained_full_names == ["step_000001.pt", "step_000002.pt"]
    assert "final.pt" not in retained_full_names
    assert "best_model.pt" not in retained_full_names
    # FU-007/FU-008: the runner records the per-rank eps and selection-view audit
    # fields; single-rank divergence is trivially satisfied. This dry run is shorter
    # than one half-epoch, so no validation boundary selects a best and the
    # selection fields honestly report the train-L1 fallback (not a false denoising
    # claim), which the strict verifier would reject as non-promotable.
    assert full_summary["per_rank_reparameterization_eps_divergent"] is True
    assert (
        full_summary["best_validation_selection_view"] == "train_l1_no_validation_best"
    )
    assert (
        full_summary["best_validation_selection_reduction"]
        == "train_l1_no_validation_best"
    )
    # FU-003: run metadata records the beta schedule resolved to epoch 1.
    assert math.isclose(cast("float", full_summary["beta_target"]), 1.0)
    assert math.isclose(cast("float", full_summary["beta_warmup_fraction"]), 0.1)
    assert full_summary["beta_warmup_steps"] == _FULL_UPDATES_PER_EPOCH

    training_summary = cast(
        "dict[str, object]",
        json.loads(
            (output_dir / "benchmark" / "training_summary.json").read_text(
                encoding="utf-8",
            ),
        ),
    )
    retained = cast(
        "list[dict[str, object]]",
        training_summary["retained_interval_checkpoints"],
    )
    retained_names = {Path(cast("str", row["path"])).name for row in retained}
    assert retained_names == {"step_000001.pt", "step_000002.pt"}
    # FU-003: the training summary records the same epoch-1 beta schedule.
    assert math.isclose(cast("float", training_summary["beta_target"]), 1.0)
    assert math.isclose(cast("float", training_summary["beta_warmup_fraction"]), 0.1)
    assert training_summary["beta_warmup_steps"] == _FULL_UPDATES_PER_EPOCH


def test_full_resume_history_merges_prefix_metrics_and_checkpoints(
    tmp_path: Path,
) -> None:
    """A resumed full run preserves pre-resume rows and interval checkpoints."""
    settings = _full_settings(tmp_path=tmp_path, max_train_steps=4, save_every=1)
    artifacts = selected_runtime_runner._artifact_paths(tmp_path)  # noqa: SLF001
    artifacts.train_steps.parent.mkdir(parents=True)
    artifacts.training_summary.parent.mkdir(parents=True)
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    contract = _FullOutputContract(
        target_updates=2,
        epochs=1,
        updates_per_epoch=2,
        half_interval=1,
        validation_batches=1,
        keep_count=4,
        world_size=1,
    )
    prior_rows = [
        _train_step_row(contract=contract, step=step, rank=0) for step in (1, 2)
    ]
    _write_csv(artifacts.train_steps, _train_step_columns(), prior_rows)
    for step in (1, 2, 3, 4):
        (checkpoint_dir / f"step_{step:06d}.pt").write_bytes(
            f"checkpoint:{step}".encode(),
        )
    (checkpoint_dir / "best_model.pt").write_bytes(b"best")
    _write_json(
        artifacts.training_summary,
        {
            "best_validation_metric": _PRIOR_BEST_VALIDATION_METRIC,
            "best_checkpoint": {
                "path": "checkpoints/best_model.pt",
                "successful_optimizer_update_count": 2,
            },
        },
    )

    history = selected_runtime_runner._load_resume_artifact_history(  # noqa: SLF001
        artifacts=artifacts,
        settings=settings,
        distributed=_local_distributed_context(),
        start_step=2,
    )
    new_rows = [
        _train_step_row(contract=contract, step=step, rank=0) for step in (3, 4)
    ]
    merged_rows = selected_runtime_runner._merge_resume_csv_rows(  # noqa: SLF001
        prior_rows=history.metric_rows,
        new_rows=new_rows,
    )
    new_checkpoints = tuple(
        CheckpointMetadata(
            path=checkpoint_dir / f"step_{step:06d}.pt",
            sha256=_sha256_file(checkpoint_dir / f"step_{step:06d}.pt"),
            optimizer_step=step,
            successful_optimizer_update_count=step,
        )
        for step in (3, 4)
    )
    retained = selected_runtime_runner._apply_checkpoint_retention(  # noqa: SLF001
        checkpoints=(*history.interval_checkpoints, *new_checkpoints),
        settings=settings,
    )

    assert [int(row["successful_optimizer_update_count"]) for row in merged_rows] == [
        1,
        2,
        3,
        4,
    ]
    assert [checkpoint.path.name for checkpoint in retained] == [
        "step_000001.pt",
        "step_000002.pt",
        "step_000003.pt",
        "step_000004.pt",
    ]
    assert history.best_checkpoint is not None
    assert history.best_checkpoint.path.name == "best_model.pt"
    assert history.best_validation_metric is not None
    assert (
        abs(history.best_validation_metric - _PRIOR_BEST_VALIDATION_METRIC)
        < _FLOAT_TOLERANCE
    )


def test_full_resume_history_requires_prior_train_metrics(tmp_path: Path) -> None:
    """Full-run resume fails closed when prior metric rows are unavailable."""
    settings = _full_settings(tmp_path=tmp_path, max_train_steps=4, save_every=1)
    artifacts = selected_runtime_runner._artifact_paths(tmp_path)  # noqa: SLF001

    with pytest.raises(
        ValueError,
        match=r"requires existing metrics/train_steps\.csv",
    ):
        selected_runtime_runner._load_resume_artifact_history(  # noqa: SLF001
            artifacts=artifacts,
            settings=settings,
            distributed=_local_distributed_context(),
            start_step=1,
        )


def test_full_interval_flush_state_includes_resume_prefix(tmp_path: Path) -> None:
    """Interval state keeps the resume prefix out of the all-gathered rows.

    The prefix is carried separately so a resumed DDP run prepends it once after
    the all-gather instead of gathering it from every rank; merging the split
    back together must still reproduce the full pre-resume-plus-new sequence.
    """
    settings = _full_settings(tmp_path=tmp_path, max_train_steps=6, save_every=1)
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    contract = _FullOutputContract(
        target_updates=6,
        epochs=1,
        updates_per_epoch=6,
        half_interval=1,
        validation_batches=1,
        keep_count=4,
        world_size=1,
    )
    checkpoints: list[CheckpointMetadata] = []
    for step in range(1, 7):
        path = checkpoint_dir / f"step_{step:06d}.pt"
        path.write_bytes(f"checkpoint:{step}".encode())
        checkpoints.append(
            CheckpointMetadata(
                path=path,
                sha256=_sha256_file(path),
                optimizer_step=step,
                successful_optimizer_update_count=step,
            ),
        )
    resume_history = selected_runtime_runner._ResumeArtifactHistory(  # noqa: SLF001
        metric_rows=tuple(
            _train_step_row(contract=contract, step=step, rank=0)
            for step in range(1, 5)
        ),
        validation_rows=tuple(_validation_rows(contract)[:4]),
        interval_checkpoints=tuple(checkpoints[:4]),
        best_checkpoint=None,
        best_validation_metric=None,
    )

    state = selected_runtime_runner._interval_flush_state(  # noqa: SLF001
        settings=settings,
        resume_history=resume_history,
        metric_rows=(
            _train_step_row(contract=contract, step=5, rank=0),
            _train_step_row(contract=contract, step=6, rank=0),
        ),
        validation_rows=tuple(_validation_rows(contract)[4:6]),
        checkpoints=tuple(checkpoints[4:6]),
        best_checkpoint=None,
        best_validation_metric=None,
        last_result=_step_result(step=6),
        current_step=6,
    )

    # The all-gathered fields carry only this rank's new rows...
    assert [row["successful_optimizer_update_count"] for row in state.metric_rows] == [
        "5",
        "6",
    ]
    # ...while the resume prefix is carried separately for a single prepend.
    assert [
        row["successful_optimizer_update_count"] for row in state.resume_metric_rows
    ] == ["1", "2", "3", "4"]
    merged_metric_rows = selected_runtime_runner._merge_resume_csv_rows(  # noqa: SLF001
        prior_rows=state.resume_metric_rows,
        new_rows=state.metric_rows,
    )
    assert [row["successful_optimizer_update_count"] for row in merged_metric_rows] == [
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
    ]
    assert [row["optimizer_step"] for row in state.resume_validation_rows] == [
        row["optimizer_step"] for row in _validation_rows(contract)[:4]
    ]
    assert [checkpoint.path.name for checkpoint in state.checkpoints] == [
        "step_000003.pt",
        "step_000004.pt",
        "step_000005.pt",
        "step_000006.pt",
    ]
    assert not (checkpoint_dir / "step_000001.pt").exists()
    assert not (checkpoint_dir / "step_000002.pt").exists()


def test_full_interval_flush_writes_resume_history_and_partial_artifacts(  # noqa: PLR0914, PLR0915
    tmp_path: Path,
) -> None:
    """Interval checkpoint flushes preserve metrics before final teardown."""
    flush_step = 2
    output_dir = tmp_path / "interval_flush"
    request = SelectedRuntimeTrainRequest(
        config_path=_FULL_CONFIG,
        runtime_config=_RUNTIME_CONFIG,
        output_dir=output_dir,
        run_name="spec0009_interval_flush",
        data="synthetic",
        max_train_steps=2,
        save_every_steps=1,
        dry_run=True,
    )
    plan = parse_selected_runtime_plan(_RUNTIME_CONFIG)
    resolved = resolve_json_config(_FULL_CONFIG)
    settings = selected_runtime_runner._settings(  # noqa: SLF001
        request=request,
        resolved=resolved,
        plan=plan,
    )
    distributed = _local_distributed_context()
    data_surface = selected_runtime_runner._prepare_data_surface(  # noqa: SLF001
        request=request,
        settings=settings,
        plan=plan,
        distributed=distributed,
    )
    try:
        artifacts = selected_runtime_runner._artifact_paths(output_dir)  # noqa: SLF001
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        contract = _FullOutputContract(
            target_updates=2,
            epochs=1,
            updates_per_epoch=2,
            half_interval=1,
            validation_batches=1,
            keep_count=4,
            world_size=1,
        )
        model = selected_runtime_runner.build_non_equivariant_vae(
            norm_groups=settings.norm_groups,
        )
        launch_command = (
            selected_runtime_runner.build_selected_runtime_torchrun_command(
                config_path=_FULL_CONFIG,
                runtime_config=_RUNTIME_CONFIG,
                data="synthetic",
                output_dir=output_dir,
                run_name=request.run_name,
                max_train_steps=2,
                save_every_steps=1,
                dry_run=True,
            )
        )
        context = selected_runtime_runner._IntervalFlushContext(  # noqa: SLF001
            artifacts=artifacts,
            request=request,
            settings=settings,
            plan=plan,
            runtime_identity=selected_runtime_runner._runtime_identity(  # noqa: SLF001
                plan,
            ),
            launch_command=launch_command,
            ddp_proof=selected_runtime_runner.build_ddp_rank_device_proof(
                plan=plan,
                probe=distributed.probe,
                launch_command=launch_command,
                dry_run=True,
            ),
            amp=selected_runtime_runner._amp_execution(  # noqa: SLF001
                plan=plan,
                distributed=distributed,
                dry_run=True,
            ),
            data_surface=data_surface,
            distributed=distributed,
        )
        metric_rows = [
            _train_step_row(contract=contract, step=step, rank=0)
            for step in range(1, flush_step + 1)
        ]
        for row in metric_rows:
            row["checkpoint_path"] = ""
        pre_checkpoint_state = selected_runtime_runner._IntervalFlushState(  # noqa: SLF001
            metric_rows=tuple(metric_rows),
            validation_rows=tuple(_validation_rows(contract)),
            gate_rows=(),
            checkpoints=(),
            best_checkpoint=None,
            best_validation_metric=None,
            last_result=_step_result(step=flush_step),
            current_step=flush_step,
        )

        selected_runtime_runner._write_interval_artifact_flush(  # noqa: SLF001
            context=context,
            model=model,
            local_state=pre_checkpoint_state,
        )

        pre_checkpoint_proof = cast(
            "dict[str, object]",
            json.loads(artifacts.checkpoint_resume_proof.read_text(encoding="utf-8")),
        )
        pre_checkpoint_manifest = cast(
            "dict[str, object]",
            json.loads(artifacts.artifact_manifest.read_text(encoding="utf-8")),
        )
        checkpoint_path = checkpoint_dir / "step_000002.pt"
        checkpoint_path.write_bytes(b"checkpoint:step_000002")
        best_path = checkpoint_dir / "best_model.pt"
        best_path.write_bytes(b"checkpoint:best")
        checkpoint = CheckpointMetadata(
            path=checkpoint_path,
            sha256=_sha256_file(checkpoint_path),
            optimizer_step=2,
            successful_optimizer_update_count=2,
        )
        best_checkpoint = CheckpointMetadata(
            path=best_path,
            sha256=_sha256_file(best_path),
            optimizer_step=2,
            successful_optimizer_update_count=2,
        )
        metric_rows[-1]["checkpoint_path"] = "checkpoints/step_000002.pt"
        post_checkpoint_state = selected_runtime_runner._IntervalFlushState(  # noqa: SLF001
            metric_rows=tuple(metric_rows),
            validation_rows=tuple(_validation_rows(contract)),
            gate_rows=(),
            checkpoints=(checkpoint,),
            best_checkpoint=best_checkpoint,
            best_validation_metric=0.5,
            last_result=_step_result(step=flush_step),
            current_step=flush_step,
        )

        selected_runtime_runner._write_interval_artifact_flush(  # noqa: SLF001
            context=context,
            model=model,
            local_state=post_checkpoint_state,
        )

        train_rows = list(
            csv.DictReader(
                artifacts.train_steps.open(encoding="utf-8", newline=""),
            ),
        )
        summary = cast(
            "dict[str, object]",
            json.loads(artifacts.training_summary.read_text(encoding="utf-8")),
        )
        full_summary = cast(
            "dict[str, object]",
            json.loads(
                artifacts.selected_runtime_full_summary.read_text(encoding="utf-8"),
            ),
        )
        manifest = cast(
            "dict[str, object]",
            json.loads(artifacts.artifact_manifest.read_text(encoding="utf-8")),
        )
        history = selected_runtime_runner._load_resume_artifact_history(  # noqa: SLF001
            artifacts=artifacts,
            settings=settings,
            distributed=distributed,
            start_step=2,
        )
        blockers = verify_selected_runtime_full_output(
            output_dir=output_dir,
            selected_runtime_path=_RUNTIME_CONFIG,
        )

        assert [row["successful_optimizer_update_count"] for row in train_rows] == [
            "1",
            "2",
        ]
        assert summary["partial_artifact_flush"] is True
        assert summary["full_run_eligible"] is False
        assert summary["optimizer_steps_completed"] == flush_step
        assert full_summary["partial_artifact_flush"] is True
        assert full_summary["status"] == "fail"
        assert pre_checkpoint_proof["latest_metric_prefix_step"] == flush_step
        assert pre_checkpoint_proof["latest_checkpoint_step"] == 0
        assert not pre_checkpoint_proof["resume_checkpoint"]
        assert "checkpoint:step_000002.pt" not in cast(
            "dict[str, object]",
            pre_checkpoint_manifest["artifact_hashes"],
        )
        assert manifest["partial_artifact_flush"] is True
        assert "train_steps" in cast("dict[str, object]", manifest["artifact_hashes"])
        proof = cast(
            "dict[str, object]",
            json.loads(artifacts.checkpoint_resume_proof.read_text(encoding="utf-8")),
        )
        assert proof["latest_checkpoint_step"] == flush_step
        assert proof["resume_checkpoint"] == "checkpoints/step_000002.pt"
        assert len(history.metric_rows) == flush_step
        assert history.best_checkpoint is not None
        assert "selected_runtime_full_output_training_summary_not_pass" in blockers
        assert (
            "selected_runtime_full_output_train_steps_schedule_incomplete" in blockers
        )
    finally:
        selected_runtime_runner._close_data_surface(data_surface)  # noqa: SLF001


def test_full_interval_flush_dedups_resume_prefix_under_simulated_ddp(  # noqa: PLR0914, PLR0915
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resumed 2-rank flush prepends the resume prefix once, not world_size times.

    Regression guard: the resume prefix lives on every rank, so gathering it
    inside the interval flush would duplicate it world_size times in the partial
    metrics CSV and later break strict full-output verification.
    """
    output_dir = tmp_path / "ddp_interval_flush"
    request = SelectedRuntimeTrainRequest(
        config_path=_FULL_CONFIG,
        runtime_config=_RUNTIME_CONFIG,
        output_dir=output_dir,
        run_name="spec0009_ddp_interval_flush",
        data="synthetic",
        max_train_steps=4,
        save_every_steps=2,
        dry_run=True,
    )
    plan = parse_selected_runtime_plan(_RUNTIME_CONFIG)
    resolved = resolve_json_config(_FULL_CONFIG)
    settings = selected_runtime_runner._settings(  # noqa: SLF001
        request=request,
        resolved=resolved,
        plan=plan,
    )
    local = _local_distributed_context()
    data_surface = selected_runtime_runner._prepare_data_surface(  # noqa: SLF001
        request=request,
        settings=settings,
        plan=plan,
        distributed=local,
    )
    try:
        artifacts = selected_runtime_runner._artifact_paths(output_dir)  # noqa: SLF001
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        contract = _FullOutputContract(
            target_updates=4,
            epochs=1,
            updates_per_epoch=4,
            half_interval=2,
            validation_batches=1,
            keep_count=4,
            world_size=2,
        )
        model = selected_runtime_runner.build_non_equivariant_vae(
            norm_groups=settings.norm_groups,
        )
        launch_command = (
            selected_runtime_runner.build_selected_runtime_torchrun_command(
                config_path=_FULL_CONFIG,
                runtime_config=_RUNTIME_CONFIG,
                data="synthetic",
                output_dir=output_dir,
                run_name=request.run_name,
                max_train_steps=4,
                save_every_steps=2,
                dry_run=True,
            )
        )
        ddp = replace(
            local,
            rank=0,
            local_rank=0,
            world_size=2,
            nproc_per_node=2,
            should_use_ddp=True,
            probe=replace(
                local.probe,
                world_size=2,
                nproc_per_node=2,
                distributed_initialized=True,
            ),
        )
        context = selected_runtime_runner._IntervalFlushContext(  # noqa: SLF001
            artifacts=artifacts,
            request=request,
            settings=settings,
            plan=plan,
            runtime_identity=selected_runtime_runner._runtime_identity(plan),  # noqa: SLF001
            launch_command=launch_command,
            ddp_proof=selected_runtime_runner.build_ddp_rank_device_proof(
                plan=plan,
                probe=ddp.probe,
                launch_command=launch_command,
                dry_run=True,
            ),
            amp=selected_runtime_runner._amp_execution(  # noqa: SLF001
                plan=plan,
                distributed=ddp,
                dry_run=True,
            ),
            data_surface=data_surface,
            distributed=ddp,
        )
        # The resume prefix already holds BOTH ranks' rows for steps 1 and 2.
        resume_metric_rows = tuple(
            _train_step_row(contract=contract, step=step, rank=rank)
            for step in (1, 2)
            for rank in (0, 1)
        )
        resume_validation_rows = tuple(_validation_rows(contract)[:2])
        # Fixed-25 rows are canonical global rank-0 rows: the resume prefix holds
        # one copy per boundary, not one per rank (Spec 0010).
        resume_equivariance_rows = (
            _equivariance_row(step=1, degrees=90),
            _equivariance_row(step=2, degrees=90),
        )
        resume_checkpoint_path = checkpoint_dir / "step_000002.pt"
        resume_checkpoint_path.write_bytes(b"checkpoint:step_000002")
        resume_history = selected_runtime_runner._ResumeArtifactHistory(  # noqa: SLF001
            metric_rows=resume_metric_rows,
            validation_rows=resume_validation_rows,
            interval_checkpoints=(
                CheckpointMetadata(
                    path=resume_checkpoint_path,
                    sha256=_sha256_file(resume_checkpoint_path),
                    optimizer_step=2,
                    successful_optimizer_update_count=2,
                ),
            ),
            best_checkpoint=None,
            best_validation_metric=None,
            equivariance_rows=resume_equivariance_rows,
        )
        local_new_metric = (_train_step_row(contract=contract, step=4, rank=0),)
        local_new_validation = tuple(
            row for row in _validation_rows(contract) if row["optimizer_step"] == "4"
        )
        local_new_equivariance = (_equivariance_row(step=4, degrees=90),)

        def fake_is_initialized() -> bool:
            return True

        def peer_rank1_rows(obj: object) -> tuple[dict[str, str], ...]:
            peer: list[dict[str, str]] = []
            for row in cast("Sequence[dict[str, str]]", obj):
                clone = dict(row)
                if "rank" in clone:
                    clone["rank"] = "1"
                event_id = clone.get("event_id", "")
                if "rank0" in event_id:
                    clone["event_id"] = event_id.replace("rank0", "rank1")
                peer.append(clone)
            return tuple(peer)

        def fake_all_gather_object(gathered: list[object], obj: object) -> None:
            # Model two ranks: this rank's rows plus a distinct rank-1 copy.
            gathered[0] = obj
            gathered[1] = peer_rank1_rows(obj)

        def fake_broadcast_object_list(payload: list[object], src: int) -> None:
            _ = (payload, src)

        monkeypatch.setattr(
            selected_runtime_runner.dist,
            "is_initialized",
            fake_is_initialized,
        )
        monkeypatch.setattr(
            selected_runtime_runner.dist,
            "all_gather_object",
            fake_all_gather_object,
        )
        monkeypatch.setattr(
            selected_runtime_runner.dist,
            "broadcast_object_list",
            fake_broadcast_object_list,
        )

        selected_runtime_runner._write_interval_artifact_flush(  # noqa: SLF001
            context=context,
            model=model,
            local_state=selected_runtime_runner._interval_flush_state(  # noqa: SLF001
                settings=settings,
                resume_history=resume_history,
                metric_rows=local_new_metric,
                validation_rows=local_new_validation,
                checkpoints=(),
                best_checkpoint=None,
                best_validation_metric=None,
                last_result=_step_result(step=4),
                current_step=4,
                equivariance_rows=local_new_equivariance,
            ),
        )

        train_rows = list(
            csv.DictReader(
                artifacts.train_steps.open(encoding="utf-8", newline=""),
            ),
        )
        validation_rows = list(
            csv.DictReader(
                artifacts.validation_metrics.open(encoding="utf-8", newline=""),
            ),
        )
        equivariance_rows = list(
            csv.DictReader(
                artifacts.equivariance_25.open(encoding="utf-8", newline=""),
            ),
        )
    finally:
        selected_runtime_runner._close_data_surface(data_surface)  # noqa: SLF001

    resume_prefix_last_step = 2
    prefix_pairs = sorted(
        (row["successful_optimizer_update_count"], row["rank"])
        for row in train_rows
        if int(row["successful_optimizer_update_count"]) <= resume_prefix_last_step
    )
    new_pairs = sorted(
        (row["successful_optimizer_update_count"], row["rank"])
        for row in train_rows
        if int(row["successful_optimizer_update_count"]) > resume_prefix_last_step
    )
    # The 4-row two-rank resume prefix appears exactly once (not world_size times)...
    assert prefix_pairs == [("1", "0"), ("1", "1"), ("2", "0"), ("2", "1")]
    # ...and both ranks' distinct new boundary rows survive the gather.
    assert new_pairs == [("4", "0"), ("4", "1")]
    new_validation = sorted(
        (row["optimizer_step"], row["rank"], row["view"])
        for row in validation_rows
        if row["optimizer_step"] == "4"
    )
    assert new_validation == [
        ("4", "0", "clean"),
        ("4", "0", "deterministic_denoising"),
        ("4", "1", "clean"),
        ("4", "1", "deterministic_denoising"),
    ]
    validation_prefix = [row for row in validation_rows if row["optimizer_step"] == "2"]
    assert len(validation_prefix) == len(resume_validation_rows)
    # The canonical fixed-25 rows are merged, never gathered: the resume prefix
    # appears once per boundary (not world_size times) and the new row survives.
    assert [row["optimizer_step"] for row in equivariance_rows] == ["1", "2", "4"]


def test_full_output_verifier_accepts_strict_artifact_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The full-run verifier accepts the expected long-run artifact shape."""
    contract = _small_full_output_contract(monkeypatch)
    output_dir = tmp_path / "full_output"
    _write_full_output_fixture(output_dir=output_dir, contract=contract)

    blockers = verify_selected_runtime_full_output(
        output_dir=output_dir,
        selected_runtime_path=_RUNTIME_CONFIG,
    )

    assert blockers == ()


def test_full_output_verifier_rejects_collapsed_eps_and_wrong_selection_view(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FU-007/FU-008: the gate rejects collapsed eps and clean-view selection."""
    contract = _small_full_output_contract(monkeypatch)
    output_dir = tmp_path / "full_output"
    _write_full_output_fixture(output_dir=output_dir, contract=contract)
    summary_path = output_dir / "benchmark" / "selected_runtime_full_summary.json"
    summary = cast(
        "dict[str, object]",
        json.loads(summary_path.read_text(encoding="utf-8")),
    )
    summary["per_rank_reparameterization_eps_divergent"] = False
    summary["best_validation_selection_view"] = "clean"
    summary["best_validation_selection_reduction"] = "rank0_local_min_over_views"
    _write_json(summary_path, summary)
    _refresh_manifest_hash(
        output_dir=output_dir,
        artifact_name="selected_runtime_full_summary",
        artifact_path=summary_path,
    )

    blockers = verify_selected_runtime_full_output(
        output_dir=output_dir,
        selected_runtime_path=_RUNTIME_CONFIG,
    )

    assert "selected_runtime_full_output_per_rank_eps_not_divergent" in blockers
    assert (
        "selected_runtime_full_output_best_validation_selection_view_mismatch"
        in blockers
    )
    assert (
        "selected_runtime_full_output_best_validation_selection_reduction_mismatch"
        in blockers
    )


def test_full_output_verifier_rejects_missing_validation_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validation rows are required at every half-epoch boundary and view."""
    contract = _small_full_output_contract(monkeypatch)
    output_dir = tmp_path / "full_output"
    _write_full_output_fixture(output_dir=output_dir, contract=contract)
    validation_path = output_dir / "metrics" / "validation_metrics.csv"
    rows = list(csv.DictReader(validation_path.open(encoding="utf-8", newline="")))
    rows = [
        row
        for row in rows
        if not (
            row["optimizer_step"] == str(contract.half_interval)
            and row["view"] == "clean"
        )
    ]
    _write_csv(validation_path, tuple(rows[0]), rows)

    blockers = verify_selected_runtime_full_output(
        output_dir=output_dir,
        selected_runtime_path=_RUNTIME_CONFIG,
    )

    assert "selected_runtime_full_output_validation_schedule_incomplete" in blockers


def test_full_output_verifier_rejects_incomplete_train_step_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The verifier rejects one-row train evidence that only reaches the max step."""
    contract = _small_full_output_contract(monkeypatch)
    output_dir = tmp_path / "full_output"
    _write_full_output_fixture(output_dir=output_dir, contract=contract)
    train_steps_path = output_dir / "metrics" / "train_steps.csv"
    _write_csv(
        train_steps_path,
        _train_step_columns(),
        [_train_step_row(contract=contract, step=contract.target_updates, rank=0)],
    )
    _refresh_manifest_hash(
        output_dir=output_dir,
        artifact_name="train_steps",
        artifact_path=train_steps_path,
    )

    blockers = verify_selected_runtime_full_output(
        output_dir=output_dir,
        selected_runtime_path=_RUNTIME_CONFIG,
    )

    assert "selected_runtime_full_output_train_steps_row_count_mismatch" in blockers
    assert "selected_runtime_full_output_train_steps_schedule_incomplete" in blockers


@pytest.mark.parametrize(
    "dropped_columns",
    [
        ("recon_loss",),
        ("kl_loss",),
        ("recon_loss", "kl_loss"),
    ],
)
def test_full_output_verifier_rejects_missing_loss_columns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dropped_columns: tuple[str, ...],
) -> None:
    """FU-002: the gate rejects train_steps.csv missing recon_loss or kl_loss.

    Each loss column is required independently, so dropping recon_loss alone,
    kl_loss alone, or both must each trip the missing-column blocker. The rows
    otherwise provide full valid per-rank schedule coverage, so the only train-step
    blocker is the missing-column contract violation, not incidental coverage.
    """
    contract = _small_full_output_contract(monkeypatch)
    output_dir = tmp_path / "full_output"
    _write_full_output_fixture(output_dir=output_dir, contract=contract)
    train_steps_path = output_dir / "metrics" / "train_steps.csv"
    dropped = set(dropped_columns)
    reduced_columns = tuple(
        column for column in _train_step_columns() if column not in dropped
    )
    reduced_rows = [
        {key: value for key, value in row.items() if key not in dropped}
        for row in _train_step_rows(contract)
    ]
    _write_csv(train_steps_path, reduced_columns, reduced_rows)
    _refresh_manifest_hash(
        output_dir=output_dir,
        artifact_name="train_steps",
        artifact_path=train_steps_path,
    )

    blockers = verify_selected_runtime_full_output(
        output_dir=output_dir,
        selected_runtime_path=_RUNTIME_CONFIG,
    )

    assert "selected_runtime_full_output_train_steps_missing_columns" in blockers
    assert "selected_runtime_full_output_train_steps_row_count_mismatch" not in blockers
    assert (
        "selected_runtime_full_output_train_steps_schedule_incomplete" not in blockers
    )


def test_reconstruction_output_stats_flag_out_of_range_decoder_output() -> None:
    """FU-018: the decoder telemetry reports range plus directional saturation.

    The exact boundary values -1.0 and 1.0 are present but must NOT be counted as
    saturated, locking the strict open-interval ``< -1`` / ``> 1`` convention.
    """
    reconstruction = torch.tensor(
        [[[[-2.0, -1.0], [1.0, 2.0]]]],
        dtype=torch.float32,
    )

    stats = selected_runtime_runner._reconstruction_output_stats(reconstruction)  # noqa: SLF001

    assert math.isclose(stats.x_hat_min, -2.0)
    assert math.isclose(stats.x_hat_max, 2.0)
    # Only -2 is < -1 and only 2 is > 1; the boundary values -1 and 1 are excluded.
    assert math.isclose(stats.frac_x_hat_lt_minus1, 0.25)
    assert math.isclose(stats.frac_x_hat_gt_1, 0.25)
    assert math.isclose(stats.recon_output_rms, math.sqrt(10.0 / 4.0))


def test_saturated_decoder_head_records_out_of_range_telemetry() -> None:
    """FU-018: a deliberately saturated output head is flagged by the telemetry."""
    image_size = 64
    model = build_non_equivariant_vae()
    bias = model.output_head.bias
    assert bias is not None
    with torch.no_grad():
        _ = bias.fill_(50.0)
    clean_batch = torch.zeros((1, 3, image_size, image_size), dtype=torch.float32)
    eps = torch.zeros((1, 16, image_size // 8, image_size // 8), dtype=torch.float32)

    output = model.forward(clean_batch, eps=eps)
    stats = selected_runtime_runner._reconstruction_output_stats(  # noqa: SLF001
        output.reconstruction,
    )

    # Zero-init head weight + bias 50 => the raw output is 50.0 at every pixel.
    assert math.isclose(stats.frac_x_hat_gt_1, 1.0)
    assert math.isclose(stats.frac_x_hat_lt_minus1, 0.0)
    assert math.isclose(stats.x_hat_min, 50.0)
    assert math.isclose(stats.x_hat_max, 50.0)
    assert math.isclose(stats.recon_output_rms, 50.0)


@pytest.mark.parametrize(
    "dropped_column",
    [
        "recon_output_rms",
        "x_hat_min",
        "x_hat_max",
        "frac_x_hat_lt_minus1",
        "frac_x_hat_gt_1",
    ],
)
def test_full_output_verifier_rejects_missing_decoder_telemetry_columns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dropped_column: str,
) -> None:
    """FU-018: each decoder-telemetry column is independently required by the gate."""
    contract = _small_full_output_contract(monkeypatch)
    output_dir = tmp_path / "full_output"
    _write_full_output_fixture(output_dir=output_dir, contract=contract)
    train_steps_path = output_dir / "metrics" / "train_steps.csv"
    reduced_columns = tuple(
        column for column in _train_step_columns() if column != dropped_column
    )
    reduced_rows = [
        {key: value for key, value in row.items() if key != dropped_column}
        for row in _train_step_rows(contract)
    ]
    _write_csv(train_steps_path, reduced_columns, reduced_rows)
    _refresh_manifest_hash(
        output_dir=output_dir,
        artifact_name="train_steps",
        artifact_path=train_steps_path,
    )

    blockers = verify_selected_runtime_full_output(
        output_dir=output_dir,
        selected_runtime_path=_RUNTIME_CONFIG,
    )

    assert "selected_runtime_full_output_train_steps_missing_columns" in blockers
    assert "selected_runtime_full_output_train_steps_row_count_mismatch" not in blockers
    assert (
        "selected_runtime_full_output_train_steps_schedule_incomplete" not in blockers
    )


def test_full_output_verifier_rejects_missing_fixed25_originals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fixed-25 originals archive is required (Spec 0010)."""
    contract = _small_full_output_contract(monkeypatch)
    output_dir = tmp_path / "full_output"
    _write_full_output_fixture(output_dir=output_dir, contract=contract)
    (output_dir / "artifacts" / "fixed25" / "originals.pt").unlink()

    blockers = verify_selected_runtime_full_output(
        output_dir=output_dir,
        selected_runtime_path=_RUNTIME_CONFIG,
    )

    assert "selected_runtime_full_output_fixed25_originals_missing" in blockers


def test_full_output_verifier_rejects_missing_equivariance_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The equivariance metrics CSV is required (Spec 0010)."""
    contract = _small_full_output_contract(monkeypatch)
    output_dir = tmp_path / "full_output"
    _write_full_output_fixture(output_dir=output_dir, contract=contract)
    (output_dir / "metrics" / "equivariance_25.csv").unlink()

    blockers = verify_selected_runtime_full_output(
        output_dir=output_dir,
        selected_runtime_path=_RUNTIME_CONFIG,
    )

    assert "selected_runtime_full_output_fixed25_equivariance_csv_missing" in blockers


def test_full_output_verifier_rejects_fixed25_rotation_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fixed-25 manifest must record the locked rot90 convention."""
    contract = _small_full_output_contract(monkeypatch)
    output_dir = tmp_path / "full_output"
    _write_full_output_fixture(output_dir=output_dir, contract=contract)
    manifest_path = output_dir / "artifacts" / "fixed25" / "manifest.json"
    manifest = cast(
        "dict[str, dict[str, object]]",
        json.loads(manifest_path.read_text(encoding="utf-8")),
    )
    manifest["rotation"]["method"] = "bilinear"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _refresh_manifest_hash(
        output_dir=output_dir,
        artifact_name="fixed25_manifest",
        artifact_path=manifest_path,
    )

    blockers = verify_selected_runtime_full_output(
        output_dir=output_dir,
        selected_runtime_path=_RUNTIME_CONFIG,
    )

    assert "selected_runtime_full_output_fixed25_manifest_rotation_mismatch" in blockers


def test_full_output_verifier_retires_reconstruction_sample_requirement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The full-run verifier no longer requires the retired single-patch dump."""
    contract = _small_full_output_contract(monkeypatch)
    output_dir = tmp_path / "full_output"
    _write_full_output_fixture(output_dir=output_dir, contract=contract)
    assert not (output_dir / "artifacts" / "reconstruction_samples.pt").exists()

    blockers = verify_selected_runtime_full_output(
        output_dir=output_dir,
        selected_runtime_path=_RUNTIME_CONFIG,
    )

    assert blockers == ()
    assert not any("reconstruction" in blocker for blocker in blockers)


def test_full_output_verifier_rejects_missing_fixed25_error_maps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full-frame error maps are a required per-boundary artifact (Spec 0010)."""
    contract = _small_full_output_contract(monkeypatch)
    output_dir = tmp_path / "full_output"
    _write_full_output_fixture(output_dir=output_dir, contract=contract)
    fixed25_dir = output_dir / "artifacts" / "fixed25"
    boundary = fixed25_dir / f"boundary_{contract.target_updates:06d}"
    (boundary / "error_maps_angle_180.pt").unlink()

    blockers = verify_selected_runtime_full_output(
        output_dir=output_dir,
        selected_runtime_path=_RUNTIME_CONFIG,
    )

    assert "selected_runtime_full_output_fixed25_boundary_incomplete" in blockers


def test_full_output_verifier_rejects_non_promotable_fixed25(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A synthetic / non-promotable fixed-25 manifest fails the promotable gate."""
    contract = _small_full_output_contract(monkeypatch)
    output_dir = tmp_path / "full_output"
    _write_full_output_fixture(output_dir=output_dir, contract=contract)
    manifest_path = output_dir / "artifacts" / "fixed25" / "manifest.json"
    manifest = cast(
        "dict[str, object]",
        json.loads(manifest_path.read_text(encoding="utf-8")),
    )
    manifest["data_source"] = "synthetic"
    manifest["promotable"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _refresh_manifest_hash(
        output_dir=output_dir,
        artifact_name="fixed25_manifest",
        artifact_path=manifest_path,
    )

    blockers = verify_selected_runtime_full_output(
        output_dir=output_dir,
        selected_runtime_path=_RUNTIME_CONFIG,
    )

    assert "selected_runtime_full_output_fixed25_manifest_non_promotable" in blockers


def test_full_resume_equivariance_prefix_requires_complete_boundaries(
    tmp_path: Path,
) -> None:
    """A truncated pre-resume equivariance CSV fails closed like validation does."""
    settings = _full_settings(tmp_path=tmp_path, max_train_steps=2, save_every=1)
    step = settings.half_epoch_interval_steps
    complete = tuple(
        _equivariance_row(step=step, degrees=degrees, metric=metric)
        for metric in REQUIRED_EQUIVARIANCE_METRICS
        for degrees in (90, 180, 270)
    )

    # A complete prefix and an empty (protocol-inactive) prefix both pass.
    selected_runtime_runner._validate_full_resume_equivariance_prefix(  # noqa: SLF001
        rows=complete,
        settings=settings,
        start_step=step,
    )
    selected_runtime_runner._validate_full_resume_equivariance_prefix(  # noqa: SLF001
        rows=(),
        settings=settings,
        start_step=step,
    )

    with pytest.raises(ValueError, match="equivariance"):
        selected_runtime_runner._validate_full_resume_equivariance_prefix(  # noqa: SLF001
            rows=complete[1:],
            settings=settings,
            start_step=step,
        )


def _small_full_output_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> _FullOutputContract:
    contract = _FullOutputContract(
        target_updates=8,
        epochs=2,
        updates_per_epoch=4,
        half_interval=2,
        validation_batches=2,
        keep_count=4,
        world_size=2,
    )
    monkeypatch.setattr(
        selected_runtime_gate,
        "REMOTE_FULL_TARGET_UPDATES",
        contract.target_updates,
    )
    monkeypatch.setattr(selected_runtime_gate, "REMOTE_FULL_EPOCHS", contract.epochs)
    monkeypatch.setattr(
        selected_runtime_gate,
        "REMOTE_FULL_UPDATES_PER_EPOCH",
        contract.updates_per_epoch,
    )
    monkeypatch.setattr(
        selected_runtime_gate,
        "REMOTE_FULL_HALF_EPOCH_INTERVAL",
        contract.half_interval,
    )
    monkeypatch.setattr(
        selected_runtime_gate,
        "REMOTE_FULL_VALIDATION_BATCHES_PER_VIEW",
        contract.validation_batches,
    )
    monkeypatch.setattr(
        selected_runtime_gate,
        "REMOTE_FULL_INTERVAL_CHECKPOINT_KEEP_COUNT",
        contract.keep_count,
    )
    return contract


def _full_settings(
    *,
    tmp_path: Path,
    max_train_steps: int,
    save_every: int,
) -> selected_runtime_runner._RunnerSettings:
    return selected_runtime_runner._settings(  # noqa: SLF001
        request=SelectedRuntimeTrainRequest(
            config_path=_FULL_CONFIG,
            runtime_config=_RUNTIME_CONFIG,
            output_dir=tmp_path,
            run_name="spec0009_resume_history_test",
            data="synthetic",
            max_train_steps=max_train_steps,
            save_every_steps=save_every,
            dry_run=True,
        ),
        resolved=resolve_json_config(_FULL_CONFIG),
        plan=parse_selected_runtime_plan(_RUNTIME_CONFIG),
    )


def _local_distributed_context() -> selected_runtime_runner._DistributedContext:
    return selected_runtime_runner._DistributedContext(  # noqa: SLF001
        device=torch.device("cpu"),
        rank=0,
        local_rank=0,
        world_size=1,
        nproc_per_node=1,
        should_use_ddp=False,
        initialized_here=False,
        probe=selected_runtime_runner.SelectedRuntimeEnvironmentProbe(
            machine_shape="local_cpu",
            accelerator_mode="local_cpu",
            cuda_device_count=0,
            visible_device_count=0,
            gpu_names=(),
            world_size=1,
            nproc_per_node=1,
            rank=0,
            local_rank=0,
            torchrun_standalone=False,
            rank_assignments=(),
            distributed_initialized=False,
        ),
    )


def _ddp_distributed_context(
    *,
    rank: int,
) -> selected_runtime_runner._DistributedContext:
    local = _local_distributed_context()
    return replace(
        local,
        rank=rank,
        local_rank=rank,
        world_size=2,
        nproc_per_node=2,
        should_use_ddp=True,
        probe=replace(
            local.probe,
            world_size=2,
            nproc_per_node=2,
            rank=rank,
            local_rank=rank,
            distributed_initialized=True,
        ),
    )


def _loaded_checkpoint_stub() -> LoadedCheckpoint:
    return LoadedCheckpoint(
        path=Path("checkpoints/step_006250.pt"),
        schema_version="spec0001.checkpoint.v5",
        run_name="spec0009_resume_stub",
        config_path=str(_FULL_CONFIG),
        config_sha256="",
        effective_config_sha256="",
        runtime_config_sha256="",
        selected_row_id=_EXPECTED_ROW_ID,
        runtime_policy_id="amp_fp16_conservative",
        lr_scheduler_state_status="not_applicable_local_debug_no_scheduler",
        beta_progress_state_status=(
            "deterministic_from_successful_optimizer_update_count"
        ),
        amp_scaler_state_status="selected_runtime_amp_scaler_state",
        torch_cuda_rng_state_status="selected_runtime_cuda_rng_state",
        ddp_sampler_progress_state_status="selected_runtime_ddp_sampler_progress",
        optimizer_step=_FULL_HALF_EPOCH_INTERVAL,
        successful_optimizer_update_count=_FULL_HALF_EPOCH_INTERVAL,
        metric_name="validation_l1_loss",
        metric_value=0.5,
        torch_generator_names=("train_data",),
    )


def _eps_metric_row(
    *,
    rank: int,
    step: int,
    eps_abs_mean: str,
    amp_step_skipped: str = "0",
) -> dict[str, str]:
    return {
        "rank": str(rank),
        "successful_optimizer_update_count": str(step),
        "eps_abs_mean": eps_abs_mean,
        "amp_step_skipped": amp_step_skipped,
    }


def _selection_boundary_row(
    *,
    view: str,
    l1: str,
    sample_count: str,
    rank: int = 0,
) -> dict[str, str]:
    return {
        "event_id": f"rank{rank}_validation_{view}_000001",
        "rank": str(rank),
        "optimizer_step": "1",
        "view": view,
        "l1_loss": l1,
        "sample_count": sample_count,
    }


def _write_full_output_fixture(
    *,
    output_dir: Path,
    contract: _FullOutputContract,
) -> None:
    benchmark = output_dir / "benchmark"
    metrics = output_dir / "metrics"
    checkpoints = output_dir / "checkpoints"
    artifacts = output_dir / "artifacts"
    for directory in (benchmark, metrics, checkpoints, artifacts):
        directory.mkdir(parents=True, exist_ok=True)
    runtime_sha256 = _sha256_file(_RUNTIME_CONFIG)
    interval_checkpoint_names = _interval_checkpoint_names(contract)
    checkpoint_names = (*interval_checkpoint_names, "final.pt", "best_model.pt")
    for name in checkpoint_names:
        (checkpoints / name).write_bytes(f"checkpoint:{name}".encode())
    fixed25_hashes = _write_fixed25_fixture(output_dir=output_dir, contract=contract)

    _write_json(
        benchmark / "training_summary.json",
        {
            "status": "local_pass",
            "run_mode": "kaggle_selected_runtime_full_train",
            "runtime_config": {"sha256": runtime_sha256},
            "target_optimizer_updates": contract.target_updates,
            "optimizer_steps_completed": contract.target_updates,
            "requested_epochs": contract.epochs,
            "optimizer_updates_per_epoch": contract.updates_per_epoch,
            "half_epoch_interval_steps": contract.half_interval,
            "validation_batches_per_view": contract.validation_batches,
            "validation_views": ["clean", "deterministic_denoising"],
            "train_reparameterization": "stochastic_seeded",
            "amp_step_skipped_count": 0,
            "nonfinite_count": 0,
            "checkpoint_retention": "best_final_latest_four_interval",
            "resume_supported": True,
            "retained_interval_checkpoint_count": contract.keep_count,
            "retained_interval_checkpoints": [
                {"path": f"checkpoints/{name}"} for name in interval_checkpoint_names
            ],
            "amp_execution": {"grad_scaler_init_scale": 16384.0},
            "final_checkpoint": {"path": "checkpoints/final.pt"},
            "best_checkpoint": {"path": "checkpoints/best_model.pt"},
        },
    )
    _write_json(
        benchmark / "selected_runtime_full_summary.json",
        {
            "status": "local_pass",
            "selected_runtime_full_run_contract_ready": True,
            "target_optimizer_updates": contract.target_updates,
            "stochastic_train_eps_proven": True,
            "per_rank_reparameterization_eps_divergent": True,
            "best_validation_selection_view": "deterministic_denoising",
            "best_validation_selection_reduction": "cross_rank_sample_weighted_l1",
        },
    )
    _write_json(
        benchmark / "selected_runtime_plan_applied.json",
        {
            "status": "local_pass",
            "plan_applied": True,
            "expected": {"runner_amp_extension": {"grad_scaler_init_scale": 16384.0}},
            "observed": {"runner_amp_extension": {"grad_scaler_init_scale": 16384.0}},
        },
    )
    _write_json(
        benchmark / "checkpoint_resume_proof.json",
        {
            "status": "local_pass",
            "grad_scaler_state_restore_attempted": True,
            "grad_scaler_state_restored": True,
            "cuda_rng_state_restore_attempted": True,
            "cuda_rng_state_restored": True,
            "sampler_progress_restored": True,
            "optimizer_scheduler_progress_restored": True,
            "beta_progress_restored": True,
        },
    )
    _write_json(benchmark / "gate_health_summary.json", {"status": "local_pass"})
    _write_json(
        benchmark / "local_selected_runtime_readiness.json",
        {"status": "local_pass"},
    )

    _write_csv(
        metrics / "train_steps.csv",
        _train_step_columns(),
        _train_step_rows(contract),
    )
    _write_csv(
        metrics / "validation_metrics.csv",
        _validation_columns(),
        _validation_rows(contract),
    )
    _write_csv(
        metrics / "gate_health.csv",
        GATE_HEALTH_COLUMNS,
        [_gate_health_row(contract)],
    )

    artifact_hashes = {
        "training_summary": _sha256_file(benchmark / "training_summary.json"),
        "selected_runtime_full_summary": _sha256_file(
            benchmark / "selected_runtime_full_summary.json",
        ),
        "selected_runtime_plan_applied": _sha256_file(
            benchmark / "selected_runtime_plan_applied.json",
        ),
        "checkpoint_resume_proof": _sha256_file(
            benchmark / "checkpoint_resume_proof.json",
        ),
        "gate_health_summary": _sha256_file(benchmark / "gate_health_summary.json"),
        "local_selected_runtime_readiness": _sha256_file(
            benchmark / "local_selected_runtime_readiness.json",
        ),
        "train_steps": _sha256_file(metrics / "train_steps.csv"),
        "validation_metrics": _sha256_file(metrics / "validation_metrics.csv"),
        "gate_health": _sha256_file(metrics / "gate_health.csv"),
        "checkpoint:final.pt": _sha256_file(checkpoints / "final.pt"),
        "checkpoint:best_model.pt": _sha256_file(checkpoints / "best_model.pt"),
        **fixed25_hashes,
    }
    artifact_hashes.update(
        {
            f"checkpoint:{name}": _sha256_file(checkpoints / name)
            for name in interval_checkpoint_names
        },
    )
    _write_json(
        benchmark / "artifact_manifest.json",
        {
            "status": "local_pass",
            "full_run_eligible": True,
            "reconstruction_sample_nonblank": True,
            "artifact_hashes": artifact_hashes,
        },
    )


def _write_fixed25_fixture(
    *,
    output_dir: Path,
    contract: _FullOutputContract,
) -> dict[str, str]:
    fixed25_dir = output_dir / "artifacts" / "fixed25"
    boundary_steps = list(
        range(
            contract.half_interval,
            contract.target_updates + 1,
            contract.half_interval,
        ),
    )
    last_step = boundary_steps[-1]
    boundary_dir = fixed25_dir / f"boundary_{last_step:06d}"
    (boundary_dir / "grids").mkdir(parents=True, exist_ok=True)
    (fixed25_dir / "originals.pt").write_bytes(b"originals")
    for name in (
        "reconstruction_progress.pt",
        "latent_mu.pt",
        "latent_pca_eqvae_style.png",
        "latent_first3.png",
    ):
        (boundary_dir / name).write_bytes(name.encode())
    (boundary_dir / "grids" / "rotated_input_vs_latent_grid.png").write_bytes(b"grid")
    for degrees in (90, 180, 270):
        (boundary_dir / f"rotated_angle_{degrees}.pt").write_bytes(b"rotated")
        (boundary_dir / f"error_maps_angle_{degrees}.pt").write_bytes(b"errors")
    _write_json(
        fixed25_dir / "manifest.json",
        {
            "schema": "spec0010.fixed25_equivariance.manifest.v1",
            "rotation": {"method": "rot90", "dims": [2, 3], "k_values": [0, 1, 2, 3]},
            "data_source": "real",
            "promotable": True,
            "boundary_optimizer_steps": boundary_steps,
        },
    )
    equivariance_rows = [
        _equivariance_row(step=step, degrees=degrees, metric=metric)
        for step in boundary_steps
        for metric in REQUIRED_EQUIVARIANCE_METRICS
        for degrees in (90, 180, 270)
    ]
    equivariance_path = output_dir / "metrics" / "equivariance_25.csv"
    _write_csv(equivariance_path, EQUIVARIANCE_25_COLUMNS, equivariance_rows)
    return {
        "equivariance_25": _sha256_file(equivariance_path),
        "fixed25_originals": _sha256_file(fixed25_dir / "originals.pt"),
        "fixed25_manifest": _sha256_file(fixed25_dir / "manifest.json"),
    }


def _equivariance_row(
    *,
    step: int,
    degrees: int,
    metric: str = "equivariance_error_25_patches",
) -> dict[str, str]:
    return {
        "optimizer_step": str(step),
        "angle_degrees": str(degrees),
        "metric_name": metric,
        "value": "0.1",
        "mean": "0.1",
        "std": "0.01",
        "n": "25",
        "data_source": "real",
        "promotable": "true",
    }


def _train_step_columns() -> tuple[str, ...]:
    return (
        "event_id",
        "rank",
        "optimizer_step_index",
        "optimizer_step",
        "successful_optimizer_update_count",
        "split",
        "loss",
        "recon_loss",
        "l1_loss",
        "ssim_loss",
        "ssim_metric",
        "kl_loss",
        "beta",
        "grad_norm",
        "param_update_norm",
        "recon_output_rms",
        "x_hat_min",
        "x_hat_max",
        "frac_x_hat_lt_minus1",
        "frac_x_hat_gt_1",
        "nonfinite_count",
        "batch_size",
        "precision_policy",
        "amp_enabled",
        "autocast_dtype",
        "grad_scaler_enabled",
        "fp32_loss",
        "torch_compile_enabled",
        "compile_scope",
        "corruption_strategy",
        "train_reparameterization",
        "eps_policy",
        "eps_seed_source",
        "eps_zero_fraction",
        "eps_abs_mean",
        "amp_step_skipped",
        "checkpoint_path",
    )


def _train_step_rows(contract: _FullOutputContract) -> list[dict[str, str]]:
    return [
        _train_step_row(contract=contract, step=step, rank=rank)
        for step in range(1, contract.target_updates + 1)
        for rank in range(contract.world_size)
    ]


def _train_step_row(
    *,
    contract: _FullOutputContract,
    step: int,
    rank: int,
) -> dict[str, str]:
    return {
        "event_id": f"rank{rank}_train_step_{step:06d}",
        "rank": str(rank),
        "optimizer_step_index": str(step - 1),
        "optimizer_step": str(step),
        "successful_optimizer_update_count": str(step),
        "split": "train",
        "loss": "1.0",
        "recon_loss": "1.0",
        "l1_loss": "0.5",
        "ssim_loss": "0.5",
        "ssim_metric": "0.5",
        "kl_loss": "0.01",
        "beta": "1.0",
        "grad_norm": "1.0",
        "param_update_norm": "0.1",
        "recon_output_rms": "0.8",
        "x_hat_min": "-0.9",
        "x_hat_max": "0.95",
        "frac_x_hat_lt_minus1": "0.0",
        "frac_x_hat_gt_1": "0.0",
        "nonfinite_count": "0",
        "batch_size": "12",
        "precision_policy": "amp_conservative",
        "amp_enabled": "true",
        "autocast_dtype": "float16",
        "grad_scaler_enabled": "true",
        "fp32_loss": "true",
        "torch_compile_enabled": "false",
        "compile_scope": "none",
        "corruption_strategy": "indexed_masked",
        "train_reparameterization": "stochastic_seeded",
        "eps_policy": "stochastic_seeded_train_generator",
        "eps_seed_source": "train_data_torch_generator",
        "eps_zero_fraction": "0.0",
        "eps_abs_mean": "0.8",
        "amp_step_skipped": "0",
        "checkpoint_path": f"checkpoints/step_{step:06d}.pt"
        if step % contract.half_interval == 0
        else "",
    }


def _validation_columns() -> tuple[str, ...]:
    return (
        "event_id",
        "rank",
        "optimizer_step",
        "validation_boundary",
        "split",
        "view",
        "batch_count",
        "sample_count",
        "loss",
        "recon_loss",
        "l1_loss",
        "ssim_loss",
        "ssim_metric",
        "kl_loss",
        "beta",
        "deterministic_eps_used",
        "corruption_strategy",
    )


def _validation_rows(contract: _FullOutputContract) -> list[dict[str, str]]:
    return [
        {
            "event_id": f"rank0_validation_{view}_{step:06d}",
            "rank": "0",
            "optimizer_step": str(step),
            "validation_boundary": "half_epoch",
            "split": "validation",
            "view": view,
            "batch_count": str(contract.validation_batches),
            "sample_count": "240",
            "loss": "1.0",
            "recon_loss": "1.0",
            "l1_loss": "0.5",
            "ssim_loss": "0.5",
            "ssim_metric": "0.5",
            "kl_loss": "0.01",
            "beta": "1.0",
            "deterministic_eps_used": "true",
            "corruption_strategy": "indexed_masked",
        }
        for step in range(
            contract.half_interval,
            contract.target_updates + 1,
            contract.half_interval,
        )
        for view in ("clean", "deterministic_denoising")
    ]


def _gate_health_row(contract: _FullOutputContract) -> dict[str, str]:
    row: dict[str, str] = dict.fromkeys(GATE_HEALTH_COLUMNS, "0.1")
    row.update(
        {
            "run_name": "non_eq_vae_spec0001_selected_runtime_full",
            "benchmark_kind": "kaggle_selected_runtime_real_ubc_runner",
            "benchmark_source": "local_selected_runtime_train_runner_rank0",
            "full_run_eligible": "true",
            "row_id": _EXPECTED_ROW_ID,
            "candidate_row_id": _EXPECTED_ROW_ID,
            "runtime_policy_id": "amp_fp16_conservative",
            "optimizer_step": str(contract.target_updates),
            "gate_health_status": "pass",
        },
    )
    return row


def _step_result(step: int) -> selected_runtime_runner._SelectedRuntimeStepResult:
    scalar = torch.tensor(1.0)
    losses = VaeLossComponents(
        loss=scalar,
        recon_loss=scalar,
        l1_loss=torch.tensor(0.5),
        ssim_loss=torch.tensor(0.5),
        ssim_metric=torch.tensor(0.5),
        kl_loss=torch.tensor(0.01),
        beta=1.0,
    )
    return selected_runtime_runner._SelectedRuntimeStepResult(  # noqa: SLF001
        optimizer_step_index=step - 1,
        successful_optimizer_update_count=step,
        losses=losses,
        grad_norm=1.0,
        param_update_norm=0.1,
        nonfinite_count=0,
        batch_size=12,
        amp_step_skipped=False,
        zero_grad_set_to_none=True,
        train_reparameterization="stochastic_seeded",
        eps_policy="stochastic_seeded_train_generator",
        eps_seed_source="train_data_torch_generator",
        eps_zero_fraction=0.0,
        eps_abs_mean=0.8,
    )


def _interval_checkpoint_names(contract: _FullOutputContract) -> tuple[str, ...]:
    steps = tuple(
        range(
            contract.half_interval,
            contract.target_updates + 1,
            contract.half_interval,
        ),
    )
    return tuple(f"step_{step:06d}.pt" for step in steps[-contract.keep_count :])


def _refresh_manifest_hash(
    *,
    output_dir: Path,
    artifact_name: str,
    artifact_path: Path,
) -> None:
    manifest_path = output_dir / "benchmark" / "artifact_manifest.json"
    manifest = cast(
        "dict[str, object]",
        json.loads(manifest_path.read_text(encoding="utf-8")),
    )
    hashes = cast("dict[str, str]", manifest["artifact_hashes"])
    hashes[artifact_name] = _sha256_file(artifact_path)
    _write_json(manifest_path, manifest)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_csv(
    path: Path,
    columns: Sequence[str],
    rows: Sequence[dict[str, str]],
) -> None:
    fieldnames = tuple(columns)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer: csv.DictWriter[str] = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
