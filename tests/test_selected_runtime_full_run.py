# Copyright 2026 HiperMaximus
# pyright: reportPrivateUsage=false
"""Focused tests for Spec 0009 selected-runtime full-run readiness."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, cast

if TYPE_CHECKING:
    from collections.abc import Sequence

import pytest
import torch

from eqvae.benchmarking import selected_runtime_gate
from eqvae.benchmarking.runtime_schema import GATE_HEALTH_COLUMNS
from eqvae.benchmarking.selected_runtime_gate import verify_selected_runtime_full_output
from eqvae.checkpointing import CheckpointMetadata, LoadedCheckpoint
from eqvae.cli.selected_runtime_train import main as selected_runtime_train_main
from eqvae.config import resolve_json_config
from eqvae.losses.vae import VaeLossComponents
from eqvae.training import selected_runtime_runner
from eqvae.training.selected_runtime import parse_selected_runtime_plan
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
        )
        local_new_metric = (_train_step_row(contract=contract, step=4, rank=0),)
        local_new_validation = tuple(
            row for row in _validation_rows(contract) if row["optimizer_step"] == "4"
        )

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
    (artifacts / "reconstruction_samples.pt").write_bytes(b"reconstruction")

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
        "reconstruction_samples": _sha256_file(artifacts / "reconstruction_samples.pt"),
        "checkpoint:final.pt": _sha256_file(checkpoints / "final.pt"),
        "checkpoint:best_model.pt": _sha256_file(checkpoints / "best_model.pt"),
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
