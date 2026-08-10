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
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from torch.utils.data import DataLoader, Sampler

    from eqvae.benchmarking.io import CsvRow, JsonObject, JsonValue
    from eqvae.corruption.stain import StainCorruptionProfile
    from eqvae.data.training_batches import PatchTrainingBatch, PatchTrainingDataset
    from eqvae.training.fastpath_step import FastpathStepOutput

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
from eqvae.config import ResolvedConfig, resolve_json_config
from eqvae.corruption.inline_stain import InlineStainCorruptor
from eqvae.data.roots import REAL_TRAIN_PATCH_COUNT
from eqvae.losses.vae import VaeLossComponents, beta_for_step
from eqvae.models.latent import LATENT_CHANNELS
from eqvae.models.non_equivariant_vae import (
    NonEquivariantVAE,
    build_non_equivariant_vae,
)
from eqvae.training import selected_runtime_runner
from eqvae.training.fastpath_recipe import FastpathDynamoKnobs
from eqvae.training.fastpath_step import make_fastpath_step_fn
from eqvae.training.selected_runtime import (
    COMPILED_FASTPATH_CORRUPTION_STRATEGY,
    EAGER_INLINE_STAIN_CORRUPTION_STRATEGY,
    EXPECTED_AMP_APPLICATION_STATUS,
    EXPECTED_AMP_OFF_APPLICATION_STATUS,
    EXPECTED_RUNTIME_POLICY_ID,
    EXPECTED_RUNTIME_PROOF_WRITE_POLICY,
    EXPECTED_SELECTED_ROW_ID,
    SelectedRuntimePlan,
    _mixed_precision_errors,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _plan_from_payload,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _runtime_policy_errors,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _runtime_proof_efficiency_errors,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _runtime_proof_write_decision_errors,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _torch_compile_errors,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    composed_selected_runtime_identity,
    parse_selected_runtime_plan,
    selected_runtime_plan_errors,
)
from eqvae.training.selected_runtime_runner import SelectedRuntimeTrainRequest

_FULL_CONFIG = Path("configs/spec0001/non_eq_vae_selected_runtime_full.json")
_SPEC0011_RUNTIME_CONFIG = Path("configs/spec0001/non_eq_vae_selected_runtime.json")
_LR_RANGE_CONFIG = Path(
    "configs/spec0001/non_eq_vae_selected_runtime_lr_range.json",
)
_TINY_OVERFIT_CONFIG = Path(
    "configs/spec0001/non_eq_vae_kaggle_tiny_overfit.json",
)
_SPEC0011_PER_DEVICE_BATCH = 25
_SPEC0011_GLOBAL_BATCH = 50
_SPEC0011_UPDATES_PER_EPOCH = 6000
_LR_RANGE_UPDATES = 192
_RUNTIME_CONFIG = Path(
    "runs/kaggle/runtime_selection_v5/benchmark/selected_runtime.json",
)
_FULL_TARGET_UPDATES = 125000
_SPEC0011_FULL_TARGET_UPDATES = 60_000
_FULL_EPOCHS = 10
_FULL_UPDATES_PER_EPOCH = 12500
_FULL_HALF_EPOCH_INTERVAL = 6250
# Mirrors the runner constant: 0 = full validation sweep every half-epoch (S17f).
_FULL_VALIDATION_BATCHES_PER_VIEW = 0
# A representative larger global batch (96 = 4x the reference 24). floor(300000/96)
# is 3125 -- odd, so it also exercises the floor half-epoch. The de-pinned validator
# must accept this schedule purely from the relationships, never the reference
# literals; target and half are spelled as the relationships they must satisfy.
_LARGER_BATCH_UPDATES_PER_EPOCH = 3125
_LARGER_BATCH_TARGET_UPDATES = _FULL_EPOCHS * _LARGER_BATCH_UPDATES_PER_EPOCH
_LARGER_BATCH_HALF_INTERVAL = _LARGER_BATCH_UPDATES_PER_EPOCH // 2
# The remote gate derives its schedule from floor(REAL_TRAIN_PATCH_COUNT / global_batch)
# and REMOTE_FULL_EPOCHS; at the reference batch 24 this is the real 12500/125000/6250.
_GATE_REFERENCE_GLOBAL_BATCH = 24
# floor(300000 / 200000) == 1 -> half 0 -> the fail-closed guard trips.
_GATE_DEGENERATE_GLOBAL_BATCH = 200_000
_GATE_NEGATIVE_GLOBAL_BATCH = -1
_GATE_INVALID_SCHEDULE_SENTINEL = -1
_LOCAL_DRY_RUN_STEPS = 2
_TINY_CPU_IMAGE_SIZE = 16
_TINY_CPU_BATCH_SIZE = 2
_PRIOR_BEST_VALIDATION_METRIC = 0.25
_FLOAT_TOLERANCE = 1e-12
_LR_REFERENCE_GLOBAL_BATCH = 24
_LR_QUADRUPLE_GLOBAL_BATCH = 96
_LR_REFERENCE_LEARNING_RATE = 5.0e-4
_LR_QUADRUPLE_LEARNING_RATE = 1.0e-3
_FULL_LR_REFERENCE_LEARNING_RATE = 0.000692820323027551
_FULL_LR_QUADRUPLE_LEARNING_RATE = 2.0 * _FULL_LR_REFERENCE_LEARNING_RATE
_S15_RECIPE_BUCKET_CAP_MB = 50
# Measured winner DDP bucket-cap (probe _DDP_OPTIMIZER_SPEC).
_RECIPE_BUCKET_CAP_MB = 50
# The eager-recipe optimize_ddp sentinel (unset dynamo config).
_EAGER_OPTIMIZE_DDP = ""
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


def test_validation_batches_full_sweep_vs_capped() -> None:
    """cap<=0 sweeps the whole loader once; a positive cap yields exactly cap batches.

    Full validation (Spec 0011 S17f) requires the uncapped path to iterate the entire
    validation loader so best-checkpoint selection sees the whole set, while the capped
    debug path yields a fixed count, cycling a short loader. A mutation that capped the
    full sweep, dropped the cap, or failed to cycle a short loader is caught.
    """
    loader = cast("DataLoader[PatchTrainingBatch]", ["a", "b", "c"])

    full = list(selected_runtime_runner._validation_batches(loader, 0))  # noqa: SLF001
    capped = list(selected_runtime_runner._validation_batches(loader, 2))  # noqa: SLF001
    cycled = list(selected_runtime_runner._validation_batches(loader, 5))  # noqa: SLF001

    assert full == ["a", "b", "c"]
    assert capped == ["a", "b"]
    assert cycled == ["a", "b", "c", "a", "b"]


def test_full_config_derives_exact_spec0009_schedule(tmp_path: Path) -> None:
    """Full mode resolves the v5 runtime into 125000 target updates."""
    plan = parse_selected_runtime_plan(_RUNTIME_CONFIG)
    settings = selected_runtime_runner._settings(  # noqa: SLF001
        request=SelectedRuntimeTrainRequest(
            config_path=_FULL_CONFIG,
            runtime_config=_SPEC0011_RUNTIME_CONFIG,
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


def test_full_kernel_rejects_beta_target_drift(tmp_path: Path) -> None:
    """The launch wrapper pins the user-selected beta-0.01 comparison policy.

    Beta 0.1 was explicitly rejected because it lost image information; a config edit
    must fail before the 12-hour run rather than silently reopen that decision.
    """
    from kaggle.kernels.selected_runtime_full import run_template  # noqa: PLC0415

    payload = cast(
        "dict[str, object]",
        json.loads(_FULL_CONFIG.read_text(encoding="utf-8")),
    )
    objective = cast("dict[str, object]", payload["objective"])
    beta = cast("dict[str, object]", objective["beta"])
    beta["target"] = 0.1
    config_path = tmp_path / "beta_drift.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match=r"locked to 0\.01"):
        run_template._validate_full_config(config_path)  # noqa: SLF001


def test_full_run_rejects_beta_warmup_not_one_epoch(tmp_path: Path) -> None:
    """FU-003: a warmup fraction that desyncs from one epoch fails closed."""
    settings = _full_settings(tmp_path=tmp_path, max_train_steps=2, save_every=1)
    desynced = replace(settings, beta_warmup_fraction=0.2)

    with pytest.raises(ValueError, match="beta warmup must span exactly one epoch"):
        selected_runtime_runner._validate_full_run_settings(  # noqa: SLF001
            desynced,
            dry_run=True,
        )


def test_full_run_accepts_derived_schedule_at_larger_batch(tmp_path: Path) -> None:
    """The de-pinned validator accepts any self-consistent goal-derived schedule.

    A larger global batch yields a smaller (here odd) updates_per_epoch; the
    validator must pass purely from target == epochs * updates_per_epoch and
    half == updates_per_epoch // 2, never from the reference-batch literals. Running
    with dry_run=False also exercises the save_every == half-epoch relationship.
    """
    settings = _full_settings(tmp_path=tmp_path, max_train_steps=2, save_every=1)
    larger_batch = replace(
        settings,
        optimizer_updates_per_epoch=_LARGER_BATCH_UPDATES_PER_EPOCH,
        target_train_steps=_LARGER_BATCH_TARGET_UPDATES,
        half_epoch_interval_steps=_LARGER_BATCH_HALF_INTERVAL,
        save_every_steps=_LARGER_BATCH_HALF_INTERVAL,
    )

    selected_runtime_runner._validate_full_run_settings(  # noqa: SLF001
        larger_batch,
        dry_run=False,
    )


def test_full_run_rejects_target_not_epochs_times_updates(tmp_path: Path) -> None:
    """A target_train_steps that is not epochs * updates_per_epoch fails closed."""
    settings = _full_settings(tmp_path=tmp_path, max_train_steps=2, save_every=1)
    drifted = replace(settings, target_train_steps=_FULL_TARGET_UPDATES + 1)

    with pytest.raises(
        ValueError,
        match=r"target_train_steps must equal requested_epochs",
    ):
        selected_runtime_runner._validate_full_run_settings(  # noqa: SLF001
            drifted,
            dry_run=True,
        )


def test_full_run_rejects_half_not_floor_of_updates(tmp_path: Path) -> None:
    """A half_epoch_interval_steps that is not updates_per_epoch // 2 fails closed."""
    settings = _full_settings(tmp_path=tmp_path, max_train_steps=2, save_every=1)
    drifted = replace(
        settings,
        half_epoch_interval_steps=_FULL_HALF_EPOCH_INTERVAL + 1,
    )

    with pytest.raises(
        ValueError,
        match=r"half_epoch_interval_steps must equal",
    ):
        selected_runtime_runner._validate_full_run_settings(  # noqa: SLF001
            drifted,
            dry_run=True,
        )


def test_gate_expected_schedule_derives_reference_batch_24() -> None:
    """The gate re-derives the real 12500/125000/6250 schedule from floor(P / batch).

    This anchors the gate to the dataset independently of the summary (MF2): at the
    reference global batch 24 the derived schedule equals the former frozen literals,
    so a real full run is validated identically.
    """
    schedule = selected_runtime_gate._remote_full_expected_schedule(  # noqa: SLF001
        _GATE_REFERENCE_GLOBAL_BATCH,
    )

    assert schedule.valid is True
    assert schedule.updates_per_epoch == _FULL_UPDATES_PER_EPOCH
    assert schedule.target_updates == _FULL_TARGET_UPDATES
    assert schedule.half_epoch_interval == _FULL_HALF_EPOCH_INTERVAL


def test_gate_expected_schedule_fails_closed_on_degenerate_batch() -> None:
    """A non-positive or too-large global batch yields an invalid, fail-closed schedule.

    A global batch that floors updates_per_epoch below 2 leaves no positive half-epoch
    boundary, so the gate marks the schedule invalid (sentinel sizes the summary can
    never match) rather than emitting a degenerate range().
    """
    for global_batch in (
        0,
        _GATE_NEGATIVE_GLOBAL_BATCH,
        _GATE_DEGENERATE_GLOBAL_BATCH,
    ):
        schedule = selected_runtime_gate._remote_full_expected_schedule(  # noqa: SLF001
            global_batch,
        )

        assert schedule.valid is False
        assert schedule.updates_per_epoch == _GATE_INVALID_SCHEDULE_SENTINEL
        assert schedule.target_updates == _GATE_INVALID_SCHEDULE_SENTINEL
        assert schedule.half_epoch_interval == _GATE_INVALID_SCHEDULE_SENTINEL


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


def _full_config_payload_with_training_edit(
    tmp_path: Path,
    *,
    filename: str,
    edit: Callable[[dict[str, object]], object],
) -> Path:
    """Write a full config copy with an in-place edit applied to its training block.

    Returns:
        Path to the written config copy.

    """
    payload = cast(
        "dict[str, object]",
        json.loads(_FULL_CONFIG.read_text(encoding="utf-8")),
    )
    payload["source_config"] = str(
        Path("configs/spec0001/non_eq_vae_model_base.json").resolve(),
    )
    edit(cast("dict[str, object]", payload["training"]))
    config_path = tmp_path / filename
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return config_path


def test_full_config_refuses_missing_epochs(tmp_path: Path) -> None:
    """Full mode fails closed when the schedule's goal input (epochs) is missing.

    The schedule is derived (epochs * steps_per_epoch), so a missing max_train_steps
    is now fine -- but a missing/zero epochs count leaves the formula undefined and
    must fail closed instead of silently running one optimizer step.
    """
    config_path = _full_config_payload_with_training_edit(
        tmp_path,
        filename="missing_epochs.json",
        edit=lambda training: training.pop("epochs"),
    )

    with pytest.raises(ValueError, match=r"must declare positive training\.epochs"):
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


def test_full_config_rejects_max_train_steps_that_contradicts_derived(
    tmp_path: Path,
) -> None:
    """A config MAY still pin max_train_steps, but it must match the derived target.

    This locks the drift guard: a stale hand-edited literal that disagrees with
    epochs * steps_per_epoch fails closed rather than overriding the real schedule.
    """
    config_path = _full_config_payload_with_training_edit(
        tmp_path,
        filename="stale_max_train_steps.json",
        edit=lambda training: training.__setitem__(
            "max_train_steps",
            _FULL_TARGET_UPDATES + 1,
        ),
    )

    with pytest.raises(
        ValueError,
        match=r"must equal epochs \* optimizer_updates_per_epoch",
    ):
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


def test_full_train_eps_uses_checkpointed_rank_generator(tmp_path: Path) -> None:
    """Paid epsilon advances the rank-local stream saved at durable boundaries.

    Resume correctness depends on the executed reparameterization stream being the one
    checkpointed and rebased per rank; drawing from global RNG would make that evidence
    decorative and could collapse both ranks onto the same latent samples.
    """
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
    before_state = generator.get_state().clone()

    eps, proof = selected_runtime_runner._train_eps(  # noqa: SLF001
        batch_size=_LOCAL_DRY_RUN_STEPS,
        latent_channels=LATENT_CHANNELS,
        settings=settings,
        train_generator=generator,
        device=torch.device("cpu"),
    )

    assert eps.shape == (2, 16, 32, 32)
    assert proof.eps_policy == "stochastic_rank_generator"
    assert proof.eps_seed_source == "checkpointed_rank_rebased_generator"
    assert not torch.equal(generator.get_state(), before_state)
    assert 0.0 <= float(proof.eps_zero_fraction) < 1.0
    assert float(proof.eps_abs_mean) > 0.0


def test_checkpoint_proof_generator_preserves_source_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CUDA checkpoint state is never restored into a CPU generator engine."""
    requested_devices: list[torch.device] = []
    sentinel = object()

    class _CudaGeneratorShape:
        device = torch.device("cuda:1")

    def fake_generator(*, device: torch.device) -> object:
        requested_devices.append(device)
        return sentinel

    monkeypatch.setattr(selected_runtime_runner.torch, "Generator", fake_generator)
    result = selected_runtime_runner._generator_on_same_device(  # noqa: SLF001
        cast("torch.Generator", _CudaGeneratorShape()),
    )

    assert result is sentinel
    assert requested_devices == [torch.device("cuda:1")]


def test_optimizer_config_scales_lr_sqrt_with_global_batch() -> None:
    """The optimizer lr equals the reference at batch 24 and sqrt-scales with batch."""
    effective = resolve_json_config(_FULL_CONFIG).effective_config
    at_reference = selected_runtime_runner._optimizer_config(  # noqa: SLF001
        effective,
        global_batch_size=_LR_REFERENCE_GLOBAL_BATCH,
    )
    at_quadruple = selected_runtime_runner._optimizer_config(  # noqa: SLF001
        effective,
        global_batch_size=_LR_QUADRUPLE_GLOBAL_BATCH,
    )
    assert math.isclose(at_reference.learning_rate, _FULL_LR_REFERENCE_LEARNING_RATE)
    assert math.isclose(at_quadruple.learning_rate, _FULL_LR_QUADRUPLE_LEARNING_RATE)


def test_optimizer_config_uses_flat_lr_without_batch_scaling() -> None:
    """A config without batch_lr_scaling keeps a flat lr at any global batch."""
    effective = resolve_json_config(
        Path("configs/spec0001/non_eq_vae_debug_cpu.json"),
    ).effective_config
    flat = selected_runtime_runner._optimizer_config(  # noqa: SLF001
        effective,
        global_batch_size=_LR_QUADRUPLE_GLOBAL_BATCH,
    )
    assert math.isclose(flat.learning_rate, _LR_REFERENCE_LEARNING_RATE)


def test_optimizer_config_threads_fused_flag_default_off() -> None:
    """`_optimizer_config` defaults fused off and threads the requested flag (S15)."""
    effective = resolve_json_config(_FULL_CONFIG).effective_config
    default = selected_runtime_runner._optimizer_config(  # noqa: SLF001
        effective,
        global_batch_size=_LR_REFERENCE_GLOBAL_BATCH,
    )
    fused = selected_runtime_runner._optimizer_config(  # noqa: SLF001
        effective,
        global_batch_size=_LR_REFERENCE_GLOBAL_BATCH,
        fused=True,
    )
    assert default.fused is False
    assert fused.fused is True


def test_settings_threads_plan_fused_optimizer_flag(tmp_path: Path) -> None:
    """`_settings` threads `plan.fused_optimizer` into the optimizer config (S15).

    The eager-v5 plan keeps fused off (behavior-preserving); a plan that requests
    fused flows through to `SpecAdamWConfig.fused`.
    """
    base_plan = parse_selected_runtime_plan(_RUNTIME_CONFIG)
    resolved = resolve_json_config(_FULL_CONFIG)
    request = SelectedRuntimeTrainRequest(
        config_path=_FULL_CONFIG,
        runtime_config=_RUNTIME_CONFIG,
        output_dir=tmp_path,
        run_name="s15_fused_threading",
        data="synthetic",
        max_train_steps=2,
        save_every_steps=1,
        dry_run=True,
    )
    eager = selected_runtime_runner._settings(  # noqa: SLF001
        request=request,
        resolved=resolved,
        plan=base_plan,
    )
    fused = selected_runtime_runner._settings(  # noqa: SLF001
        request=request,
        resolved=resolved,
        plan=replace(base_plan, fused_optimizer=True),
    )
    assert eager.optimizer_config.fused is False
    assert fused.optimizer_config.fused is True


def test_model_requires_buffer_broadcast_is_structural() -> None:
    """The broadcast rule keys on running-stat buffers, not a hardcoded flag (S15)."""
    requires = selected_runtime_runner.model_requires_buffer_broadcast
    non_eq = build_non_equivariant_vae(norm_groups=8)
    running_stats = torch.nn.BatchNorm2d(4)
    stateless_norm = torch.nn.BatchNorm2d(4, track_running_stats=False)
    assert requires(non_eq) is False
    assert requires(running_stats) is True
    assert requires(stateless_norm) is False


def _spy_on_wrap_ddp(
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict[str, object]]:
    """Replace `wrap_fastpath_ddp` with a spy and expose its captured kwargs.

    Returns:
        The list each `wrap_fastpath_ddp` call appends its keyword arguments to.

    """
    calls: list[dict[str, object]] = []

    def spy(model: torch.nn.Module, **kwargs: object) -> torch.nn.Module:
        calls.append(kwargs)
        return model

    monkeypatch.setattr(selected_runtime_runner, "wrap_fastpath_ddp", spy)
    return calls


def test_maybe_wrap_ddp_single_process_returns_model_unwrapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single-process runs return the raw model; the DDP wrap is never built (S15)."""
    calls = _spy_on_wrap_ddp(monkeypatch)
    model = build_non_equivariant_vae(norm_groups=8)
    plan = parse_selected_runtime_plan(_RUNTIME_CONFIG)
    wrapped = selected_runtime_runner._maybe_wrap_ddp(  # noqa: SLF001
        model=model,
        distributed=_local_distributed_context(),
        plan=plan,
    )
    assert wrapped is model
    assert calls == []


def test_maybe_wrap_ddp_forwards_eager_recipe_behavior_preserving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the eager-v5 plan the DDP wrap gets torch-default knob values (S15)."""
    calls = _spy_on_wrap_ddp(monkeypatch)
    model = build_non_equivariant_vae(norm_groups=8)
    plan = parse_selected_runtime_plan(_RUNTIME_CONFIG)
    selected_runtime_runner._maybe_wrap_ddp(  # noqa: SLF001
        model=model,
        distributed=_ddp_distributed_context(rank=0),
        plan=plan,
    )
    captured = calls[-1]
    assert captured["local_rank"] == 0
    assert captured["static_graph"] == plan.ddp_static_graph
    assert captured["gradient_as_bucket_view"] == plan.ddp_gradient_as_bucket_view
    assert captured["broadcast_buffers"] is True
    assert captured["find_unused_parameters"] is False
    assert captured["bucket_cap_mb"] is None


def test_maybe_wrap_ddp_consumes_distinguishing_recipe_knobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A measured recipe's DDP knobs flow through to the wrap unchanged (S15).

    The model has only constant buffers, so the structural rule contributes nothing
    and the plan's `ddp_broadcast_buffers=False` passes straight through.
    """
    calls = _spy_on_wrap_ddp(monkeypatch)
    hook_calls: list[tuple[torch.nn.Module, str]] = []

    def spy_hook(model: torch.nn.Module, hook_name: str) -> None:
        hook_calls.append((model, hook_name))

    monkeypatch.setattr(
        selected_runtime_runner,
        "register_fastpath_communication_hook",
        spy_hook,
    )
    model = build_non_equivariant_vae(norm_groups=8)
    plan = replace(
        parse_selected_runtime_plan(_RUNTIME_CONFIG),
        ddp_static_graph=False,
        ddp_gradient_as_bucket_view=True,
        ddp_broadcast_buffers=False,
        ddp_find_unused_parameters=True,
        ddp_bucket_cap_mb=_S15_RECIPE_BUCKET_CAP_MB,
        ddp_forward_sync_buffers=False,
        communication_hook="fp16_compress_hook",
    )
    selected_runtime_runner._maybe_wrap_ddp(  # noqa: SLF001
        model=model,
        distributed=_ddp_distributed_context(rank=1),
        plan=plan,
    )
    captured = calls[-1]
    assert captured["local_rank"] == 1
    assert captured["gradient_as_bucket_view"] is True
    assert captured["broadcast_buffers"] is False
    assert captured["find_unused_parameters"] is True
    assert captured["bucket_cap_mb"] == _S15_RECIPE_BUCKET_CAP_MB
    assert captured["forward_sync_buffers"] is False
    assert hook_calls == [(model, "fp16_compress_hook")]


def test_maybe_wrap_ddp_structural_rule_forces_broadcast_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model that needs broadcasting overrides a plan that would disable it (S15)."""
    calls = _spy_on_wrap_ddp(monkeypatch)

    def always_broadcast(_model: torch.nn.Module) -> bool:
        return True

    monkeypatch.setattr(
        selected_runtime_runner,
        "model_requires_buffer_broadcast",
        always_broadcast,
    )
    model = build_non_equivariant_vae(norm_groups=8)
    plan = replace(
        parse_selected_runtime_plan(_RUNTIME_CONFIG),
        ddp_broadcast_buffers=False,
    )
    selected_runtime_runner._maybe_wrap_ddp(  # noqa: SLF001
        model=model,
        distributed=_ddp_distributed_context(rank=0),
        plan=plan,
    )
    assert calls[-1]["broadcast_buffers"] is True


def test_optimizer_lr_scaling_provenance_records_relationship() -> None:
    """The lr-scaling provenance records the reference -> effective lr relationship."""
    effective = resolve_json_config(_FULL_CONFIG).effective_config
    provenance = selected_runtime_runner._optimizer_lr_scaling(  # noqa: SLF001
        effective,
        global_batch_size=_LR_QUADRUPLE_GLOBAL_BATCH,
    )
    assert cast("bool", provenance["scaling_applied"])
    assert cast("str", provenance["rule"]) == "sqrt"
    assert cast("int", provenance["global_batch_size"]) == _LR_QUADRUPLE_GLOBAL_BATCH
    assert math.isclose(
        cast("float", provenance["reference_learning_rate"]),
        _FULL_LR_REFERENCE_LEARNING_RATE,
    )
    assert math.isclose(
        cast("float", provenance["effective_learning_rate"]),
        _FULL_LR_QUADRUPLE_LEARNING_RATE,
    )


def test_optimizer_lr_scaling_provenance_flat_without_batch_scaling() -> None:
    """Provenance is flat (effective == reference) when no scaling block is set."""
    effective = resolve_json_config(
        Path("configs/spec0001/non_eq_vae_debug_cpu.json"),
    ).effective_config
    provenance = selected_runtime_runner._optimizer_lr_scaling(  # noqa: SLF001
        effective,
        global_batch_size=_LR_QUADRUPLE_GLOBAL_BATCH,
    )
    assert not cast("bool", provenance["scaling_applied"])
    assert math.isclose(
        cast("float", provenance["reference_learning_rate"]),
        _LR_REFERENCE_LEARNING_RATE,
    )
    assert math.isclose(
        cast("float", provenance["effective_learning_rate"]),
        _LR_REFERENCE_LEARNING_RATE,
    )


def test_gate_lr_blockers_accept_producer_provenance() -> None:
    """The gate accepts the exact lr provenance the runner records at a scaled batch.

    Producer and gate share the scaling primitive, so the runner's recorded effective
    lr (the quadruple lr at global batch 96) re-derives cleanly in the gate.
    """
    effective = resolve_json_config(_FULL_CONFIG).effective_config
    provenance = selected_runtime_runner._optimizer_lr_scaling(  # noqa: SLF001
        effective,
        global_batch_size=_LR_QUADRUPLE_GLOBAL_BATCH,
    )
    training_summary: JsonObject = {"optimizer_lr_scaling": provenance}

    blockers = selected_runtime_gate._remote_full_lr_blockers(  # noqa: SLF001
        training_summary,
        global_batch_size=_LR_QUADRUPLE_GLOBAL_BATCH,
    )

    assert blockers == ()


def test_gate_lr_blockers_reject_effective_lr_off_relationship() -> None:
    """An effective lr that violates the scaling relationship fails closed."""
    effective = resolve_json_config(_FULL_CONFIG).effective_config
    provenance = dict(
        selected_runtime_runner._optimizer_lr_scaling(  # noqa: SLF001
            effective,
            global_batch_size=_LR_QUADRUPLE_GLOBAL_BATCH,
        ),
    )
    # The correct effective lr at global batch 96 is the quadruple lr; the reference
    # lr violates the sqrt relationship.
    provenance["effective_learning_rate"] = _LR_REFERENCE_LEARNING_RATE
    training_summary: JsonObject = {"optimizer_lr_scaling": provenance}

    blockers = selected_runtime_gate._remote_full_lr_blockers(  # noqa: SLF001
        training_summary,
        global_batch_size=_LR_QUADRUPLE_GLOBAL_BATCH,
    )

    assert "selected_runtime_full_output_lr_scaling_relationship_mismatch" in blockers


def test_gate_lr_blockers_reject_missing_provenance() -> None:
    """A summary with no lr-scaling provenance fails closed."""
    training_summary: JsonObject = {}

    blockers = selected_runtime_gate._remote_full_lr_blockers(  # noqa: SLF001
        training_summary,
        global_batch_size=_LR_REFERENCE_GLOBAL_BATCH,
    )

    assert blockers == ("selected_runtime_full_output_lr_scaling_missing",)


def test_gate_lr_blockers_reject_absent_learning_rate_fields() -> None:
    """A provenance block that drops the learning-rate fields fails closed.

    _float_value maps a missing field to 0.0 and scaled(0.0) == 0.0, so without the
    positive-finite guard a truncated or tampered summary would pass the relationship
    check; the gate must reject it.
    """
    effective = resolve_json_config(_FULL_CONFIG).effective_config
    provenance = dict(
        selected_runtime_runner._optimizer_lr_scaling(  # noqa: SLF001
            effective,
            global_batch_size=_LR_REFERENCE_GLOBAL_BATCH,
        ),
    )
    del provenance["reference_learning_rate"]
    del provenance["effective_learning_rate"]
    training_summary: JsonObject = {"optimizer_lr_scaling": provenance}

    blockers = selected_runtime_gate._remote_full_lr_blockers(  # noqa: SLF001
        training_summary,
        global_batch_size=_LR_REFERENCE_GLOBAL_BATCH,
    )

    assert "selected_runtime_full_output_lr_scaling_learning_rate_invalid" in blockers


def _cross_consistency_training_summary() -> JsonObject:
    return {
        "target_optimizer_updates": _FULL_TARGET_UPDATES,
        "optimizer_steps_completed": _FULL_TARGET_UPDATES,
        "requested_epochs": _FULL_EPOCHS,
        "optimizer_updates_per_epoch": _FULL_UPDATES_PER_EPOCH,
        "half_epoch_interval_steps": _FULL_HALF_EPOCH_INTERVAL,
        "validation_batches_per_view": _FULL_VALIDATION_BATCHES_PER_VIEW,
        "validation_views": ["clean", "deterministic_denoising"],
    }


def test_gate_cross_consistency_accepts_matching_schedule() -> None:
    """A full summary that agrees with the training summary schedule passes."""
    training_summary = _cross_consistency_training_summary()
    full_summary = dict(training_summary)

    blockers = selected_runtime_gate._remote_full_cross_consistency_blockers(  # noqa: SLF001
        training_summary=training_summary,
        full_summary=full_summary,
    )

    assert blockers == ()


def test_gate_cross_consistency_rejects_full_summary_schedule_drift() -> None:
    """A full summary that drifts on any schedule field fails closed.

    The gate anchors only the full summary's target to the derived schedule, so a full
    summary that disagrees with the training summary on updates_per_epoch (or half /
    epochs / cadence) would otherwise go unverified.
    """
    training_summary = _cross_consistency_training_summary()
    full_summary = dict(training_summary)
    full_summary["optimizer_updates_per_epoch"] = _FULL_UPDATES_PER_EPOCH + 1

    blockers = selected_runtime_gate._remote_full_cross_consistency_blockers(  # noqa: SLF001
        training_summary=training_summary,
        full_summary=full_summary,
    )

    assert blockers == ("selected_runtime_full_output_full_summary_schedule_mismatch",)


# --- Spec 0011 S7: plan parser relationship + DDPOptimizer safety checks -------------

_LAUNCH_SCHEDULE_ERROR_NAMES = frozenset(
    {
        "selected_runtime_top_level_wrong_per_device_batch",
        "selected_runtime_top_level_wrong_global_batch",
        "selected_runtime_top_level_wrong_optimizer_updates_per_epoch",
    },
)
_DDP_OPTIMIZER_CONFLICT_ERROR_NAMES = frozenset(
    {
        "selected_runtime_ddp_optimizer_compiled_autograd_conflict",
        "selected_runtime_ddp_optimizer_static_graph_conflict",
        "selected_runtime_ddp_optimizer_find_unused_parameters_conflict",
    },
)


def _committed_runtime_payload() -> JsonObject:
    return cast(
        "JsonObject",
        json.loads(_RUNTIME_CONFIG.read_text(encoding="utf-8")),
    )


def _plan_block(payload: JsonObject, key: str) -> JsonObject:
    block = payload[key]
    assert isinstance(block, dict)
    return cast("JsonObject", block)


def test_plan_parser_accepts_reference_batch_schedule() -> None:
    """The committed batch-24 plan parses with no launch-schedule error (S7 de-pin).

    Called without a path so the linked runtime-proof cross-check is skipped; the whole
    committed plan is already error-free, so this locks in the behavior-preserving
    property that de-pinning 12/24/12500 into relationships did not change the reference
    plan's acceptance.
    """
    errors = selected_runtime_plan_errors(_committed_runtime_payload())

    assert errors == ()


def test_spec0011_checked_in_winner_plan_parses_from_measured_source() -> None:
    """The new consumer plan must resolve exactly to the measured bs25 winner.

    Batch and schedule values are measured/derived evidence, not convenience literals:
    the source-winner hash and parser cross-check protect the translation.
    """
    plan = parse_selected_runtime_plan(_SPEC0011_RUNTIME_CONFIG)

    assert plan.per_device_batch_size == _SPEC0011_PER_DEVICE_BATCH
    assert plan.global_batch_size == _SPEC0011_GLOBAL_BATCH
    assert plan.optimizer_updates_per_epoch == _SPEC0011_UPDATES_PER_EPOCH
    assert plan.runtime_policy_id == ("compile_step_python_reducer_fp16_channels_last")
    assert plan.compile_scope == "step"
    assert plan.fused_optimizer is True


def test_spec0011_winner_plan_rejects_source_hash_drift() -> None:
    """A plan detached from its measured winner cannot become training truth."""
    payload = cast(
        "JsonObject",
        json.loads(_SPEC0011_RUNTIME_CONFIG.read_text(encoding="utf-8")),
    )
    source = _plan_block(payload, "source_winner")
    source["sha256"] = "0" * 64

    errors = selected_runtime_plan_errors(
        payload,
        selected_runtime_path=_SPEC0011_RUNTIME_CONFIG,
    )

    assert "selected_runtime_source_winner_sha256_mismatch" in errors


def test_lr_range_config_resolves_exact_bounded_sweep(tmp_path: Path) -> None:
    """The pre-training LR experiment is one bounded 192-update real-data sweep."""
    plan = parse_selected_runtime_plan(_SPEC0011_RUNTIME_CONFIG)
    settings = selected_runtime_runner._settings(  # noqa: SLF001
        request=SelectedRuntimeTrainRequest(
            config_path=_LR_RANGE_CONFIG,
            runtime_config=_SPEC0011_RUNTIME_CONFIG,
            output_dir=tmp_path,
            run_name="spec0011_lr_range_test",
            data="synthetic",
            dry_run=True,
        ),
        resolved=resolve_json_config(_LR_RANGE_CONFIG),
        plan=plan,
    )

    sweep = settings.learning_rate_range
    assert settings.run_mode == "kaggle_learning_rate_range"
    assert settings.target_train_steps == _LR_RANGE_UPDATES
    assert settings.max_train_steps == _LR_RANGE_UPDATES
    assert sweep is not None
    assert sweep.start == pytest.approx(2e-5)
    assert sweep.end == pytest.approx(3e-3)


def test_lr_range_preserves_parameter_group_multipliers(tmp_path: Path) -> None:
    """Sweeping the base LR must not erase the gate group's half-rate contract."""
    plan = parse_selected_runtime_plan(_SPEC0011_RUNTIME_CONFIG)
    settings = selected_runtime_runner._settings(  # noqa: SLF001
        request=SelectedRuntimeTrainRequest(
            config_path=_LR_RANGE_CONFIG,
            runtime_config=_SPEC0011_RUNTIME_CONFIG,
            output_dir=tmp_path,
            run_name="spec0011_lr_group_test",
            data="synthetic",
            dry_run=True,
        ),
        resolved=resolve_json_config(_LR_RANGE_CONFIG),
        plan=plan,
    )
    ordinary = torch.nn.Parameter(torch.ones(()))
    gate = torch.nn.Parameter(torch.ones(()))
    base_lr = settings.optimizer_config.learning_rate
    optimizer = torch.optim.AdamW(
        [
            {"params": [ordinary], "lr": base_lr},
            {"params": [gate], "lr": base_lr * 0.5},
        ],
    )

    first = selected_runtime_runner._apply_learning_rate_for_step(  # noqa: SLF001
        optimizer=optimizer,
        settings=settings,
        optimizer_step_index=0,
    )
    final = selected_runtime_runner._apply_learning_rate_for_step(  # noqa: SLF001
        optimizer=optimizer,
        settings=settings,
        optimizer_step_index=_LR_RANGE_UPDATES - 1,
    )

    assert first == pytest.approx(2e-5)
    assert final == pytest.approx(3e-3)
    assert cast("float", optimizer.param_groups[0]["lr"]) == pytest.approx(3e-3)
    assert cast("float", optimizer.param_groups[1]["lr"]) == pytest.approx(1.5e-3)


def test_tiny_overfit_uses_ten_update_warmup_then_constant(tmp_path: Path) -> None:
    """The learnability test reaches the measured constant LR on update ten."""
    plan = parse_selected_runtime_plan(_SPEC0011_RUNTIME_CONFIG)
    settings = selected_runtime_runner._settings(  # noqa: SLF001
        request=SelectedRuntimeTrainRequest(
            config_path=_TINY_OVERFIT_CONFIG,
            runtime_config=_SPEC0011_RUNTIME_CONFIG,
            output_dir=tmp_path,
            run_name="spec0011_tiny_lr_schedule_test",
            data="synthetic",
            dry_run=True,
        ),
        resolved=resolve_json_config(_TINY_OVERFIT_CONFIG),
        plan=plan,
    )
    peak = settings.optimizer_config.learning_rate
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(()))], lr=peak)

    first = selected_runtime_runner._apply_learning_rate_for_step(  # noqa: SLF001
        optimizer=optimizer,
        settings=settings,
        optimizer_step_index=0,
    )
    tenth = selected_runtime_runner._apply_learning_rate_for_step(  # noqa: SLF001
        optimizer=optimizer,
        settings=settings,
        optimizer_step_index=9,
    )
    final = selected_runtime_runner._apply_learning_rate_for_step(  # noqa: SLF001
        optimizer=optimizer,
        settings=settings,
        optimizer_step_index=127,
    )

    assert peak == pytest.approx(0.0007216878364870322)
    assert first == pytest.approx(peak * 0.1)
    assert tenth == pytest.approx(peak)
    assert final == pytest.approx(peak)


def test_full_run_uses_warmup_then_cosine_to_floor(tmp_path: Path) -> None:
    """The 60k-update run warms to 1e-3 and cosines to 1e-5 without restarts."""
    plan = parse_selected_runtime_plan(_SPEC0011_RUNTIME_CONFIG)
    settings = selected_runtime_runner._settings(  # noqa: SLF001
        request=SelectedRuntimeTrainRequest(
            config_path=_FULL_CONFIG,
            runtime_config=_SPEC0011_RUNTIME_CONFIG,
            output_dir=tmp_path,
            run_name="spec0011_full_lr_schedule_test",
            data="synthetic",
            dry_run=True,
        ),
        resolved=resolve_json_config(_FULL_CONFIG),
        plan=plan,
    )
    peak = settings.optimizer_config.learning_rate
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(()))], lr=peak)

    observed = [
        selected_runtime_runner._apply_learning_rate_for_step(  # noqa: SLF001
            optimizer=optimizer,
            settings=settings,
            optimizer_step_index=index,
        )
        for index in (0, 599, 600, 59_999)
    ]

    assert settings.target_train_steps == _SPEC0011_FULL_TARGET_UPDATES
    assert peak == pytest.approx(1e-3)
    assert observed == pytest.approx([1e-4, 1e-3, 1e-3, 1e-5])


def test_lr_range_summary_requires_complete_two_rank_learning_curve(
    tmp_path: Path,
) -> None:
    """A complete finite two-rank curve yields one bounded evidence-backed LR."""
    plan = parse_selected_runtime_plan(_SPEC0011_RUNTIME_CONFIG)
    settings = selected_runtime_runner._settings(  # noqa: SLF001
        request=SelectedRuntimeTrainRequest(
            config_path=_LR_RANGE_CONFIG,
            runtime_config=_SPEC0011_RUNTIME_CONFIG,
            output_dir=tmp_path,
            run_name="spec0011_lr_summary_test",
            data="synthetic",
            dry_run=True,
        ),
        resolved=resolve_json_config(_LR_RANGE_CONFIG),
        plan=plan,
    )
    sweep = settings.learning_rate_range
    assert sweep is not None
    rows: list[CsvRow] = []
    for step_index in range(_LR_RANGE_UPDATES):
        fraction = step_index / (_LR_RANGE_UPDATES - 1)
        learning_rate = sweep.start * math.exp(
            math.log(sweep.end / sweep.start) * fraction,
        )
        loss = 1.0 - 0.5 * fraction
        for rank in range(plan.world_size):
            rows.append(  # noqa: PERF401 - clearer fixture rows
                {
                    "rank": str(rank),
                    "successful_optimizer_update_count": str(step_index + 1),
                    "learning_rate": str(learning_rate),
                    "loss": str(loss),
                    "l1_loss": str(loss * 0.9),
                    "recon_loss": str(loss * 0.95),
                    "amp_step_skipped": "0",
                    "nonfinite_count": "0",
                },
            )

    summary = selected_runtime_runner._learning_rate_range_summary(  # noqa: SLF001
        settings=settings,
        plan=plan,
        runtime_identity=selected_runtime_runner._runtime_identity(plan),  # noqa: SLF001
        metric_rows=rows,
        gate_health_summary={"status": "local_pass"},
        plan_applied={"status": "local_pass"},
    )

    assert summary["status"] == "local_pass"
    assert summary["range_completed"] is True
    assert summary["loss_decreased"] is True
    assert summary["observed_ranks"] == [0, 1]
    recommended = cast("float", summary["recommended_learning_rate"])
    assert sweep.start <= recommended <= sweep.end


def test_plan_parser_defaults_recipe_knobs_to_eager() -> None:
    """The committed v5 plan omits every S11 recipe knob, so all default to eager.

    This is the S11 behavior-preservation guarantee: adding the optional recipe fields
    must not change how the pre-S11 fallback plan parses. The defaults reproduce the
    eager v5 recipe (no DDPOptimizer, no compiled autograd, DDP-library defaults for
    broadcast/find-unused/bucket-cap, fused off).
    """
    plan = parse_selected_runtime_plan(_RUNTIME_CONFIG)

    assert plan.compile_backend == "eager"
    assert plan.compile_dynamic is False
    assert plan.optimize_ddp == _EAGER_OPTIMIZE_DDP
    assert plan.compiled_autograd is False
    assert plan.reorder_compute_comm_overlap is False
    assert plan.ddp_broadcast_buffers is True
    assert plan.ddp_find_unused_parameters is False
    assert plan.ddp_bucket_cap_mb is None
    assert plan.fused_optimizer is False
    # v5 recorded false but the runner historically applied foreach=True. Without the
    # new applied marker, parsing preserves that effective legacy behavior.
    assert plan.gradient_clip_foreach is True


def test_plan_parser_reads_recipe_knobs_from_carrier_homes() -> None:
    """Each S11 recipe knob is parsed from its frozen carrier home.

    Dynamo/inductor knobs live in ``torch_compile``; DDP/optimizer knobs live in
    ``runtime_policy`` beside the existing ``ddp_*`` fields. ``_plan_from_payload`` is
    the pure builder (no launch validation / linked-proof cross-check), so a
    hand-mutated payload proves the home-to-field mapping without a valid proof bundle.
    """
    # Every knob is set to a value distinct from its eager default so a dropped or
    # wrong-home read is caught (mutation-proof). _plan_from_payload does no validation,
    # so this deliberately artificial combination only exercises the reader; the S7
    # DDPOptimizer guard is covered by the separate test_plan_parser_rejects_* tests.
    payload = _committed_runtime_payload()
    torch_compile = _plan_block(payload, "torch_compile")
    torch_compile["backend"] = "inductor"
    torch_compile["dynamic"] = True
    torch_compile["optimize_ddp"] = "ddp_optimizer"
    torch_compile["compiled_autograd"] = True
    torch_compile["reorder_compute_comm_overlap"] = True
    runtime_policy = _plan_block(payload, "runtime_policy")
    runtime_policy["ddp_broadcast_buffers"] = False
    runtime_policy["ddp_find_unused_parameters"] = True
    runtime_policy["ddp_bucket_cap_mb"] = 50
    runtime_policy["fused_optimizer"] = True
    runtime_policy["gradient_clip_foreach"] = False
    runtime_policy["gradient_clip_foreach_applied"] = True
    runtime_policy["tf32_enabled"] = False
    runtime_policy["matmul_precision"] = "highest"

    plan = _plan_from_payload(path=_RUNTIME_CONFIG, payload=payload)

    assert plan.compile_backend == "inductor"
    assert plan.compile_dynamic is True
    assert plan.optimize_ddp == "ddp_optimizer"
    assert plan.compiled_autograd is True
    assert plan.reorder_compute_comm_overlap is True
    assert plan.ddp_broadcast_buffers is False
    assert plan.ddp_find_unused_parameters is True
    assert plan.ddp_bucket_cap_mb == _RECIPE_BUCKET_CAP_MB
    assert plan.fused_optimizer is True
    assert plan.gradient_clip_foreach is False
    assert plan.tf32_enabled is False
    assert plan.matmul_precision == "highest"


def test_plan_parser_accepts_relationship_derived_larger_batch() -> None:
    """A re-measured non-24 batch is accepted purely from the derived relationships.

    global 96 = per-device 48 * world 2, and floor(300000 / 96) == 3125 (odd, so it also
    exercises the floor derivation). The snapshot is left as the committed winner-row
    description because re-pointing it is Phase 4 (S17); this isolates the launch
    relationship, which must no longer emit any batch/schedule error for gb 96.
    """
    payload = _committed_runtime_payload()
    payload["per_device_batch_size"] = 48
    payload["global_batch_size"] = _LR_QUADRUPLE_GLOBAL_BATCH
    payload["optimizer_updates_per_epoch"] = _LARGER_BATCH_UPDATES_PER_EPOCH

    errors = selected_runtime_plan_errors(payload)

    assert _LAUNCH_SCHEDULE_ERROR_NAMES.isdisjoint(errors)


def test_plan_parser_rejects_recorded_updates_off_derivation() -> None:
    """A recorded optimizer_updates_per_epoch != floor(P / G) fails the parser.

    This is the plan-recorded schedule cross-check: the plan's own number is validated
    against the single-sourced derivation, not merely trusted. The global batch still
    equals per-device * world, so only the updates relationship trips.
    """
    payload = _committed_runtime_payload()
    payload["optimizer_updates_per_epoch"] = _FULL_UPDATES_PER_EPOCH + 1

    errors = selected_runtime_plan_errors(payload)

    assert "selected_runtime_top_level_wrong_optimizer_updates_per_epoch" in errors
    assert "selected_runtime_top_level_wrong_global_batch" not in errors


def test_plan_parser_rejects_global_batch_off_product() -> None:
    """A global batch that is not per-device * world_size fails the parser."""
    payload = _committed_runtime_payload()
    payload["global_batch_size"] = _GATE_REFERENCE_GLOBAL_BATCH + 1

    errors = selected_runtime_plan_errors(payload)

    assert "selected_runtime_top_level_wrong_global_batch" in errors


def test_plan_parser_rejects_non_positive_per_device_batch() -> None:
    """A non-positive per-device batch fails closed on the preserved error id."""
    payload = _committed_runtime_payload()
    payload["per_device_batch_size"] = 0

    errors = selected_runtime_plan_errors(payload)

    assert "selected_runtime_top_level_wrong_per_device_batch" in errors


def test_plan_parser_accepts_ddp_optimizer_with_safe_flags() -> None:
    """The currently measured DDPOptimizer recipe passes its safety policy.

    _DDP_OPTIMIZER_SPEC pairs optimize_ddp="ddp_optimizer" with
    compiled_autograd=False, static_graph=False, and find_unused_parameters=False.
    This is a measured cross-check, not the only valid pairing; forcing any guarded
    flag on makes the parser reject it.
    """
    payload = _committed_runtime_payload()
    runtime_policy = _plan_block(payload, "runtime_policy")
    runtime_policy["optimize_ddp"] = "ddp_optimizer"
    runtime_policy["compiled_autograd"] = False
    runtime_policy["ddp_static_graph"] = False
    runtime_policy["ddp_find_unused_parameters"] = False

    errors = selected_runtime_plan_errors(payload)

    assert errors == ()


def test_plan_parser_allows_ddp_optimizer_with_compiled_autograd() -> None:
    """DDPOptimizer plus compiled autograd remains a measurable latest-Torch axis.

    PyTorch 2.13 documents compiled-autograd constraints for ``python_reducer`` and
    ``no_optimization`` but does not prohibit this pairing. The current measured row
    keeps it disabled; the parser must not turn that observation into a permanent ban.
    """
    payload = _committed_runtime_payload()
    runtime_policy = _plan_block(payload, "runtime_policy")
    runtime_policy["optimize_ddp"] = "ddp_optimizer"
    runtime_policy["compiled_autograd"] = True

    errors = selected_runtime_plan_errors(payload)

    assert errors == ()


def test_plan_parser_allows_ddp_optimizer_compiled_autograd_in_torch_compile() -> None:
    """The measurable pairing is accepted from its canonical torch_compile carrier.

    compiled_autograd is a Dynamo setting, so a generated plan may carry it with
    optimize_ddp under torch_compile. This is a measurable compatibility option, not a
    promised winner; dropping the carrier read or inventing a conflict changes the
    expected clean result.
    """
    payload = _committed_runtime_payload()
    torch_compile = _plan_block(payload, "torch_compile")
    torch_compile["optimize_ddp"] = "ddp_optimizer"
    torch_compile["compiled_autograd"] = True

    errors = selected_runtime_plan_errors(payload)

    assert errors == ()


@pytest.mark.parametrize(
    ("optimize_ddp", "compiled_autograd", "expected_error"),
    [
        (
            "python_reducer",
            False,
            "selected_runtime_python_reducer_requires_compiled_autograd",
        ),
        (
            "python_reducer_without_compiled_forward",
            False,
            "selected_runtime_python_reducer_requires_compiled_autograd",
        ),
        (
            "no_optimization",
            True,
            "selected_runtime_no_optimization_compiled_autograd_conflict",
        ),
    ],
)
def test_plan_parser_rejects_torch_mode_compiled_autograd_conflicts(
    optimize_ddp: str,
    compiled_autograd: bool,  # noqa: FBT001
    expected_error: str,
) -> None:
    """Selected plans reject the two documented mode/autograd incompatibilities.

    This is compatibility policy needed before a paid run. Removing either carrier
    validation makes its parametrized stable error identifier disappear.
    """
    payload = _committed_runtime_payload()
    torch_compile = _plan_block(payload, "torch_compile")
    torch_compile["optimize_ddp"] = optimize_ddp
    torch_compile["compiled_autograd"] = compiled_autograd

    assert expected_error in selected_runtime_plan_errors(payload)


def test_plan_parser_accepts_reducer_without_compiled_forward_when_detected() -> None:
    """A measured current-Torch backward-only reducer recipe can be promoted.

    This is a feature-detected experimental axis, not a frozen winner. Rejecting the
    token despite compiled autograd would make the generator able to measure a row that
    the runner can never consume.
    """
    payload = _committed_runtime_payload()
    torch_compile = _plan_block(payload, "torch_compile")
    torch_compile["optimize_ddp"] = "python_reducer_without_compiled_forward"
    torch_compile["compiled_autograd"] = True

    assert selected_runtime_plan_errors(payload) == ()


def test_plan_parser_rejects_ddp_optimizer_with_static_graph() -> None:
    """DDPOptimizer + static_graph is the loud dynamo #93672 conflict, so it fails."""
    payload = _committed_runtime_payload()
    runtime_policy = _plan_block(payload, "runtime_policy")
    runtime_policy["optimize_ddp"] = "ddp_optimizer"
    runtime_policy["ddp_static_graph"] = True

    errors = selected_runtime_plan_errors(payload)

    assert "selected_runtime_ddp_optimizer_static_graph_conflict" in errors


def test_plan_parser_rejects_ddp_optimizer_with_find_unused_parameters() -> None:
    """DDPOptimizer + find_unused_parameters is incompatible with the bucket split."""
    payload = _committed_runtime_payload()
    runtime_policy = _plan_block(payload, "runtime_policy")
    runtime_policy["optimize_ddp"] = "ddp_optimizer"
    runtime_policy["ddp_find_unused_parameters"] = True

    errors = selected_runtime_plan_errors(payload)

    assert errors == (
        "selected_runtime_runtime_policy_find_unused_mismatch",
        "selected_runtime_ddp_optimizer_find_unused_parameters_conflict",
    )


def test_plan_parser_ddp_safety_is_noop_without_ddp_optimizer() -> None:
    """The safety guard is gated on optimize_ddp, not a blanket flag ban.

    A plan with static_graph on but no optimize_ddp must not raise any DDPOptimizer
    conflict (the flags are only unsafe when paired with DDPOptimizer); the existing
    runtime policy pin still rejects the flipped flag, proving the guard added nothing
    spurious.
    """
    payload = _committed_runtime_payload()
    runtime_policy = _plan_block(payload, "runtime_policy")
    runtime_policy["ddp_static_graph"] = True

    errors = selected_runtime_plan_errors(payload)

    assert _DDP_OPTIMIZER_CONFLICT_ERROR_NAMES.isdisjoint(errors)
    assert "selected_runtime_runtime_policy_ddp_static_graph_mismatch" in errors


@pytest.mark.parametrize("global_value", [0, -1, "24"])
def test_plan_parser_fails_closed_on_malformed_global_batch(
    global_value: int | str,
) -> None:
    """A malformed global batch fails closed on the preserved id without ever raising.

    The recorded global batch is the divisor in floor(P / G), so a zero, negative, or
    string value must yield wrong_global_batch (not a ZeroDivisionError or TypeError).
    Zero in particular proves the derivation never divides by a bad divisor.
    """
    payload = _committed_runtime_payload()
    payload["global_batch_size"] = global_value

    errors = selected_runtime_plan_errors(payload)

    assert "selected_runtime_top_level_wrong_global_batch" in errors


def test_plan_parser_fails_closed_on_bool_global_batch() -> None:
    """A bool global batch is rejected, not coerced (True is an int subclass)."""
    payload = _committed_runtime_payload()
    payload["global_batch_size"] = True

    errors = selected_runtime_plan_errors(payload)

    assert "selected_runtime_top_level_wrong_global_batch" in errors


def test_plan_parser_fails_closed_on_missing_global_batch() -> None:
    """A plan missing global_batch_size fails closed rather than raising KeyError."""
    payload = _committed_runtime_payload()
    del payload["global_batch_size"]

    errors = selected_runtime_plan_errors(payload)

    assert "selected_runtime_top_level_wrong_global_batch" in errors


def test_plan_parser_accepts_non_dividing_batch_with_floored_updates() -> None:
    """A non-dividing global batch is accepted with floor(P / G) updates, ceil rejected.

    300000 / 7000 == 42.857..., so the recorded updates must be the floored 42 (from the
    single-sourced training_steps_per_epoch); the ceil value 43 must be rejected. This
    pins the floor derivation at the parser itself, not only in the schedule helper, on
    a batch where floor and ceil actually differ.
    """
    payload = _committed_runtime_payload()
    payload["per_device_batch_size"] = 3500
    payload["global_batch_size"] = 7000
    payload["optimizer_updates_per_epoch"] = 42

    assert _LAUNCH_SCHEDULE_ERROR_NAMES.isdisjoint(
        selected_runtime_plan_errors(payload),
    )

    payload["optimizer_updates_per_epoch"] = 43

    assert (
        "selected_runtime_top_level_wrong_optimizer_updates_per_epoch"
        in selected_runtime_plan_errors(payload)
    )


# --- Spec 0011 S17a: recipe value validators accept the compiled winner profile ------

# Every recipe-coherence error id the three de-pinned validators emit for a well-formed
# (dict) recipe payload -- a compiled winner payload must trip NONE of these. The two
# non-dict sentinels (missing_torch_compile / missing_runtime_policy) are unreachable
# from the coherent integration payload and so are not listed. Identity/snapshot pins
# are a separate re-point (Spec 0011 S17b / Kaggle row_id mint), not in this set.
_RECIPE_COHERENCE_ERROR_NAMES = frozenset(
    {
        "selected_runtime_mixed_precision_missing_fp32_loss",
        "selected_runtime_mixed_precision_not_enabled",
        "selected_runtime_mixed_precision_wrong_dtype",
        "selected_runtime_mixed_precision_missing_scaler",
        "selected_runtime_mixed_precision_amp_off_not_disabled",
        "selected_runtime_mixed_precision_amp_off_scaler_enabled",
        "selected_runtime_mixed_precision_amp_off_autocast_dtype",
        "selected_runtime_mixed_precision_wrong_policy",
        "selected_runtime_torch_compile_enabled_mismatch",
        "selected_runtime_torch_compile_scope_mismatch",
        "selected_runtime_torch_compile_dynamic_mismatch",
        "selected_runtime_torch_compile_backend_mismatch",
        "selected_runtime_runtime_policy_memory_format_mismatch",
        "selected_runtime_runtime_policy_ddp_static_graph_mismatch",
        "selected_runtime_runtime_policy_zero_grad_set_to_none_mismatch",
    },
)
# The compiled winner's composed row_id (bs48 amp-off compile-step). S17b accepts it
# structurally when the plan's own fields recompose to it (see the S17b section below).
_COMPILED_WINNER_ROW_ID = (
    "dual_t4_ddp__bs48__amp_off_fp32__compile_step__indexed_masked__"
    "policy_compile_step_ddp_optimizer_fp32_channels_last"
)


def _amp_conservative_block() -> JsonObject:
    return {
        "enabled": True,
        "policy": "amp_conservative",
        "autocast_dtype": "float16",
        "fp32_loss": True,
        "grad_scaler_enabled": True,
    }


def _compiled_winner_mixed_precision() -> JsonObject:
    return {
        "enabled": False,
        "policy": "amp_off_fp32",
        "autocast_dtype": "",
        "fp32_loss": True,
        "grad_scaler_enabled": False,
    }


def _compiled_winner_torch_compile() -> JsonObject:
    # The dynamo knobs (optimize_ddp / compiled_autograd / reorder) live in the
    # torch_compile carrier block, matching the real emitter's block placement.
    return {
        "enabled": True,
        "scope": "step",
        "dynamic": False,
        "backend": "inductor",
        "optimize_ddp": "ddp_optimizer",
        "compiled_autograd": False,
        "reorder_compute_comm_overlap": False,
    }


def _compiled_winner_runtime_policy() -> JsonObject:
    # The DDP-wrap / optimizer knobs live in runtime_policy beside the existing ddp_*.
    return {
        "memory_format": "channels_last",
        "ddp_static_graph": False,
        "ddp_gradient_as_bucket_view": True,
        "zero_grad_set_to_none": True,
        "ddp_broadcast_buffers": False,
        "ddp_find_unused_parameters": False,
        "ddp_bucket_cap_mb": _RECIPE_BUCKET_CAP_MB,
        "fused_optimizer": True,
    }


def _compiled_winner_payload() -> JsonObject:
    """Return the committed plan with the three recipe blocks set to the winner recipe.

    Only the recipe blocks (mixed_precision / torch_compile / runtime_policy) are
    swapped to the amp-off compiled winner; identity and snapshot stay at the eager
    committed values, so this isolates the S17a recipe-coherence surface the way the S7
    larger-batch test isolates the schedule relationship.

    Returns:
        The committed plan payload carrying the compiled winner recipe blocks.

    """
    payload = _committed_runtime_payload()
    payload["mixed_precision"] = _compiled_winner_mixed_precision()
    payload["torch_compile"] = _compiled_winner_torch_compile()
    payload["runtime_policy"] = _compiled_winner_runtime_policy()
    return payload


def test_full_parser_rejects_compiled_recipe_with_inconsistent_identity() -> None:
    """A compiled recipe whose row_id disagrees with its own fields is rejected.

    S17b makes identity structural. ``_compiled_winner_payload`` swaps only the recipe
    blocks (its batch and policy id stay eager), so labeling it with the fully-shaped
    bs48 winner row_id is self-inconsistent and correctly rejected -- while the recipe
    validators still pass, proving recipe acceptance and identity enforcement are
    independent.
    """
    payload = _compiled_winner_payload()
    payload["selected_row_id"] = _COMPILED_WINNER_ROW_ID

    errors = selected_runtime_plan_errors(payload)

    assert _RECIPE_COHERENCE_ERROR_NAMES.isdisjoint(errors)
    assert "selected_runtime_selected_row_id_not_self_consistent" in errors


# --- Spec 0011 S17b: structural identity + snapshot cross-consistency ----------------

_WINNER_POLICY_ID = "compile_step_ddp_optimizer_fp32_channels_last"
_WINNER_PER_DEVICE_BATCH = 48
_WINNER_GLOBAL_BATCH = 96


def _consistent_compiled_winner_payload() -> JsonObject:
    """Return a fully self-consistent compiled winner plan (Spec 0011 S17b).

    Extends ``_compiled_winner_payload`` (which swaps only the recipe blocks) so every
    identity, batch, and snapshot field agrees: the bs48 amp-off-fp32 compile-step
    winner id, its per-device/global batch and derived schedule, its policy id, and a
    snapshot rebuilt to the winner's string cells. The whole plan therefore parses with
    no error, proving the parser accepts a re-measured compiled winner end to end.

    Returns:
        A compiled winner plan whose identity, batch, and snapshot are self-consistent.

    """
    payload = _compiled_winner_payload()
    payload["selected_row_id"] = _COMPILED_WINNER_ROW_ID
    payload["runtime_policy_id"] = _WINNER_POLICY_ID
    payload["per_device_batch_size"] = _WINNER_PER_DEVICE_BATCH
    payload["global_batch_size"] = _WINNER_GLOBAL_BATCH
    payload["optimizer_updates_per_epoch"] = (
        REAL_TRAIN_PATCH_COUNT // _WINNER_GLOBAL_BATCH
    )
    snapshot = _plan_block(payload, "selected_row_snapshot")
    snapshot.update(
        {
            "row_id": _COMPILED_WINNER_ROW_ID,
            "runtime_policy_id": _WINNER_POLICY_ID,
            "precision_policy": "amp_off_fp32",
            "per_device_batch_size": str(_WINNER_PER_DEVICE_BATCH),
            "global_batch_size": str(_WINNER_GLOBAL_BATCH),
            "grad_scaler_enabled": "false",
            "autocast_dtype": "",
        },
    )
    return payload


def test_full_parser_accepts_fully_consistent_compiled_winner() -> None:
    """The parser accepts a self-consistent re-measured compiled winner.

    S17b de-pins identity and the snapshot batch/precision cells to cross-consistency
    with the plan's own fields, so a fully-shaped bs48 amp-off compile-step winner
    (recipe + identity + batch + snapshot all agreeing) parses with zero errors -- the
    acceptance the Kaggle row_id mint needs, proven locally without a mint.
    """
    errors = selected_runtime_plan_errors(_consistent_compiled_winner_payload())

    assert errors == ()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("per_device_batch_size", 12),
        ("runtime_policy_id", "amp_fp16_conservative"),
        ("selected_row_id", EXPECTED_SELECTED_ROW_ID),
    ],
)
def test_structural_identity_rejects_inconsistent_row_id(
    field: str,
    value: JsonValue,
) -> None:
    """Flipping any composing field away from the recorded row_id is rejected."""
    payload = _consistent_compiled_winner_payload()
    payload[field] = value

    assert (
        "selected_runtime_selected_row_id_not_self_consistent"
        in selected_runtime_plan_errors(payload)
    )


def test_structural_identity_rejects_empty_policy_id() -> None:
    """An empty runtime_policy_id is rejected as a missing free identifier."""
    payload = _consistent_compiled_winner_payload()
    payload["runtime_policy_id"] = ""

    assert "selected_runtime_runtime_policy_id_missing" in selected_runtime_plan_errors(
        payload,
    )


@pytest.mark.parametrize(
    ("cell", "value", "expected_error"),
    [
        (
            "per_device_batch_size",
            "12",
            "selected_runtime_snapshot_wrong_per_device_batch",
        ),
        ("global_batch_size", "24", "selected_runtime_snapshot_wrong_global_batch"),
        (
            "precision_policy",
            "amp_conservative",
            "selected_runtime_snapshot_wrong_precision_policy",
        ),
        ("grad_scaler_enabled", "true", "selected_runtime_snapshot_missing_scaler"),
        (
            "autocast_dtype",
            "float16",
            "selected_runtime_snapshot_wrong_autocast_dtype",
        ),
        (
            "runtime_policy_id",
            "amp_fp16_conservative",
            "selected_runtime_snapshot_policy_mismatch",
        ),
        ("row_id", EXPECTED_SELECTED_ROW_ID, "selected_runtime_snapshot_row_mismatch"),
    ],
)
def test_snapshot_cross_consistency_rejects_cell_drift(
    cell: str,
    value: str,
    expected_error: str,
) -> None:
    """A snapshot cell that disagrees with the plan's own field is rejected.

    Each case drifts one snapshot cell away from the winner plan's own field, proving
    the de-pinned cells are genuinely cross-checked (not merely dropped).
    """
    payload = _consistent_compiled_winner_payload()
    snapshot = _plan_block(payload, "selected_row_snapshot")
    snapshot[cell] = value

    assert expected_error in selected_runtime_plan_errors(payload)


@pytest.mark.parametrize(
    ("cell", "value", "expected_error"),
    [
        (
            "torch_version",
            "2.14.0+cu140",
            "selected_runtime_snapshot_wrong_torch_version",
        ),
        (
            "torch_cuda_version",
            "14.0",
            "selected_runtime_snapshot_wrong_cuda_version",
        ),
    ],
)
def test_spec0011_snapshot_rejects_runtime_stack_drift(
    cell: str,
    value: str,
    expected_error: str,
) -> None:
    """The paid run cannot replace the measured Torch/CUDA stack with a new one."""
    payload = cast(
        "JsonObject",
        json.loads(_SPEC0011_RUNTIME_CONFIG.read_text(encoding="utf-8")),
    )
    snapshot = _plan_block(payload, "selected_row_snapshot")
    snapshot[cell] = value

    assert expected_error in selected_runtime_plan_errors(
        payload,
        selected_runtime_path=_SPEC0011_RUNTIME_CONFIG,
    )


@pytest.mark.parametrize(
    ("cell", "expected_error"),
    [
        ("torch_version", "selected_runtime_snapshot_wrong_torch_version"),
        ("torch_cuda_version", "selected_runtime_snapshot_wrong_cuda_version"),
    ],
)
def test_spec0011_snapshot_requires_runtime_stack(
    cell: str,
    expected_error: str,
) -> None:
    """The paid run cannot lose either measured Torch/CUDA stack anchor."""
    payload = cast(
        "JsonObject",
        json.loads(_SPEC0011_RUNTIME_CONFIG.read_text(encoding="utf-8")),
    )
    snapshot = _plan_block(payload, "selected_row_snapshot")
    del snapshot[cell]

    assert expected_error in selected_runtime_plan_errors(
        payload,
        selected_runtime_path=_SPEC0011_RUNTIME_CONFIG,
    )


@pytest.mark.parametrize(
    ("cell", "value", "expected_error"),
    [
        (
            "accelerator_mode",
            "single_visible_t4",
            "selected_runtime_snapshot_not_dual_t4_ddp",
        ),
        (
            "machine_shape",
            "NvidiaA100",
            "selected_runtime_snapshot_wrong_machine_shape",
        ),
        ("status", "fail", "selected_runtime_snapshot_status_not_pass"),
        ("nproc_per_node", "1", "selected_runtime_snapshot_wrong_nproc_per_node"),
        ("world_size", "1", "selected_runtime_snapshot_wrong_world_size"),
    ],
)
def test_snapshot_still_pins_hardware_anchors(
    cell: str,
    value: str,
    expected_error: str,
) -> None:
    """The hardware/status/corruption anchors stay pinned after the S17b de-pin."""
    payload = _consistent_compiled_winner_payload()
    snapshot = _plan_block(payload, "selected_row_snapshot")
    snapshot[cell] = value

    assert expected_error in selected_runtime_plan_errors(payload)


def test_runtime_proof_write_decision_identity_is_structural() -> None:
    """The proof write-decision selected_row_id is checked against the plan's own id."""
    decision: JsonObject = {
        "allowed": True,
        "policy": EXPECTED_RUNTIME_PROOF_WRITE_POLICY,
        "selected_row_id": _COMPILED_WINNER_ROW_ID,
        "stain_corruptor_qa_status": "pass",
        "blockers": [],
        "linked_pass_row_failures": [],
        "stain_corruptor_qa_missing_candidate_row_ids": [],
    }
    mismatch = "selected_runtime_runtime_proof_write_decision_selected_row_id_mismatch"

    assert (
        _runtime_proof_write_decision_errors(
            decision,
            expected_row_id=_COMPILED_WINNER_ROW_ID,
        )
        == ()
    )
    assert mismatch in _runtime_proof_write_decision_errors(
        decision,
        expected_row_id=EXPECTED_SELECTED_ROW_ID,
    )
    # Fail closed when the plan's own id could not be composed.
    assert mismatch in _runtime_proof_write_decision_errors(
        decision,
        expected_row_id=None,
    )


def test_runtime_proof_efficiency_identity_is_structural() -> None:
    """The proof efficiency block's row_id and policy id match the plan's own values."""
    efficiency: JsonObject = {
        "status": "pass",
        "material_speedup_over_baseline": True,
        "selected_row_id": _COMPILED_WINNER_ROW_ID,
        "selected_runtime_policy_id": _WINNER_POLICY_ID,
    }
    row_mismatch = "selected_runtime_runtime_proof_efficiency_selected_row_id_mismatch"
    policy_mismatch = (
        "selected_runtime_runtime_proof_efficiency_selected_runtime_policy_id_mismatch"
    )

    assert (
        _runtime_proof_efficiency_errors(
            efficiency,
            expected_row_id=_COMPILED_WINNER_ROW_ID,
            expected_policy_id=_WINNER_POLICY_ID,
        )
        == ()
    )
    assert row_mismatch in _runtime_proof_efficiency_errors(
        efficiency,
        expected_row_id=EXPECTED_SELECTED_ROW_ID,
        expected_policy_id=_WINNER_POLICY_ID,
    )
    assert policy_mismatch in _runtime_proof_efficiency_errors(
        efficiency,
        expected_row_id=_COMPILED_WINNER_ROW_ID,
        expected_policy_id="amp_fp16_conservative",
    )
    # Fail closed when either plan-side value could not be composed.
    fail_closed = _runtime_proof_efficiency_errors(
        efficiency,
        expected_row_id=None,
        expected_policy_id=None,
    )
    assert row_mismatch in fail_closed
    assert policy_mismatch in fail_closed


def test_composed_identity_of_committed_plan_equals_frozen_constants() -> None:
    """The committed v5 plan recomposes to exactly the two frozen identity constants.

    Two identity vocabularies coexist: the constants, still published for callers and
    tests, and the identity every cross-check now derives from the plan itself. This
    pins them together on the committed plan, so the constants cannot quietly come to
    name something the plan no longer says.
    """
    assert composed_selected_runtime_identity(_committed_runtime_payload()) == (
        EXPECTED_SELECTED_ROW_ID,
        EXPECTED_RUNTIME_POLICY_ID,
    )


def test_composed_identity_of_compiled_winner_is_its_own_identity() -> None:
    """A re-measured compiled winner composes to its own id, not the v5 literal."""
    identity = composed_selected_runtime_identity(_consistent_compiled_winner_payload())

    assert identity == (_COMPILED_WINNER_ROW_ID, _WINNER_POLICY_ID)


def test_composed_identity_ignores_the_recorded_row_id_field() -> None:
    """Identity is composed from the plan's fields, never read off its recorded id.

    A tampered ``selected_row_id`` must not be able to tell the gate which rows to
    accept, so composition ignores the recorded field entirely.
    """
    payload = _consistent_compiled_winner_payload()
    payload["selected_row_id"] = "attacker_supplied_row_id"

    composed, _policy = composed_selected_runtime_identity(payload)

    assert composed == _COMPILED_WINNER_ROW_ID


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("per_device_batch_size", "48"),
        ("per_device_batch_size", None),
        # A bool is an int in Python: unguarded, this composes "bsTrue", not None.
        ("per_device_batch_size", True),
        ("mixed_precision", {}),
        ("mixed_precision", {"policy": 48}),
        ("torch_compile", "compiled"),
        ("torch_compile", {"scope": None}),
        ("corruption", "indexed_masked"),
        ("corruption", {"strategy": ["indexed_masked"]}),
        ("accelerator_mode", None),
        ("runtime_policy_id", 48),
    ],
)
def test_composed_identity_fails_closed_on_unusable_plan(
    field: str,
    value: JsonValue,
) -> None:
    """A plan missing or mistyping any composing field composes to None (fail closed).

    The gate turns a None into an identity blocker, so a plan it cannot understand is
    rejected rather than matching whatever the CSV happens to record.
    """
    payload = _consistent_compiled_winner_payload()
    payload[field] = value

    composed, _policy = composed_selected_runtime_identity(payload)

    assert composed is None


@pytest.mark.parametrize("value", [None, "", 48, ["policy"]])
def test_composed_identity_policy_fails_closed_on_unusable_label(
    value: JsonValue,
) -> None:
    """A missing, empty, or non-string runtime_policy_id composes to None."""
    payload = _consistent_compiled_winner_payload()
    payload["runtime_policy_id"] = value

    _composed, policy = composed_selected_runtime_identity(payload)

    assert policy is None


def test_composed_identity_policy_half_catches_what_the_row_id_cannot() -> None:
    """An empty policy still composes a row_id, so the policy half is what rejects it.

    The row_id suppresses the policy suffix for an empty label, making it identical to a
    legitimately suffixless one -- the row_id comparison alone cannot tell them apart,
    which is why identity carries a separately-checked policy half.
    """
    payload = _consistent_compiled_winner_payload()
    payload["runtime_policy_id"] = ""

    composed, policy = composed_selected_runtime_identity(payload)

    assert composed == _COMPILED_WINNER_ROW_ID.removesuffix(
        f"__policy_{_WINNER_POLICY_ID}",
    )
    assert policy is None


def test_mixed_precision_validator_accepts_both_profiles() -> None:
    """amp_conservative, amp_scalar_gate_relaxed, and amp_off_fp32 all parse clean."""
    conservative = _amp_conservative_block()
    relaxed = {**conservative, "policy": "amp_scalar_gate_relaxed"}

    assert _mixed_precision_errors(conservative) == ()
    assert _mixed_precision_errors(relaxed) == ()
    assert _mixed_precision_errors(_compiled_winner_mixed_precision()) == ()


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        ({"policy": "amp_bogus"}, "selected_runtime_mixed_precision_wrong_policy"),
        ({"fp32_loss": False}, "selected_runtime_mixed_precision_missing_fp32_loss"),
        (
            {"grad_scaler_enabled": False},
            "selected_runtime_mixed_precision_missing_scaler",
        ),
        ({"enabled": False}, "selected_runtime_mixed_precision_not_enabled"),
        (
            {"autocast_dtype": "bfloat16"},
            "selected_runtime_mixed_precision_wrong_dtype",
        ),
    ],
)
def test_mixed_precision_validator_fails_closed_on_amp_profile(
    mutation: JsonObject,
    expected_error: str,
) -> None:
    """Each broken field of the AMP profile fails closed on its stable error id."""
    block = _amp_conservative_block()
    block.update(mutation)

    assert expected_error in _mixed_precision_errors(block)


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        ({"enabled": True}, "selected_runtime_mixed_precision_amp_off_not_disabled"),
        (
            {"grad_scaler_enabled": True},
            "selected_runtime_mixed_precision_amp_off_scaler_enabled",
        ),
        (
            {"autocast_dtype": "float16"},
            "selected_runtime_mixed_precision_amp_off_autocast_dtype",
        ),
        ({"fp32_loss": False}, "selected_runtime_mixed_precision_missing_fp32_loss"),
    ],
)
def test_mixed_precision_validator_fails_closed_on_amp_off_profile(
    mutation: JsonObject,
    expected_error: str,
) -> None:
    """amp_off_fp32 with AMP re-enabled, a scaler on, or no fp32 island fails closed."""
    block = _compiled_winner_mixed_precision()
    block.update(mutation)

    assert expected_error in _mixed_precision_errors(block)


def test_torch_compile_validator_accepts_eager_and_compiled_profiles() -> None:
    """Both stable compile scopes accept the complete measured DDP recipe.

    Scope flexibility is deliberate policy: selection may promote either
    ``model_forward`` or ``step`` without dropping the required DDP mode. Deriving the
    second profile from the measured winner catches a validator mutation that pins
    acceptance to ``step`` while avoiding an incomplete self-attested fixture.
    """
    eager = {"enabled": False, "scope": "none", "dynamic": False, "backend": "eager"}
    model_forward = _compiled_winner_torch_compile()
    model_forward["scope"] = "model_forward"

    assert _torch_compile_errors(eager) == ()
    assert _torch_compile_errors(_compiled_winner_torch_compile()) == ()
    assert _torch_compile_errors(model_forward) == ()


@pytest.mark.parametrize(
    ("block", "expected_error"),
    [
        (
            {"enabled": True, "scope": "none", "dynamic": False, "backend": "inductor"},
            "selected_runtime_torch_compile_scope_mismatch",
        ),
        (
            {"enabled": True, "scope": "step", "dynamic": False, "backend": "eager"},
            "selected_runtime_torch_compile_backend_mismatch",
        ),
        (
            {
                "enabled": False,
                "scope": "none",
                "dynamic": False,
                "backend": "inductor",
            },
            "selected_runtime_torch_compile_backend_mismatch",
        ),
        (
            {"enabled": False, "scope": "step", "dynamic": False, "backend": "eager"},
            "selected_runtime_torch_compile_scope_mismatch",
        ),
        (
            {
                "enabled": True,
                "scope": "model_loss",
                "dynamic": False,
                "backend": "inductor",
            },
            "selected_runtime_torch_compile_scope_mismatch",
        ),
        (
            {"enabled": True, "scope": "step", "dynamic": True, "backend": "inductor"},
            "selected_runtime_torch_compile_dynamic_mismatch",
        ),
        (
            {
                "enabled": "yes",
                "scope": "step",
                "dynamic": False,
                "backend": "inductor",
            },
            "selected_runtime_torch_compile_enabled_mismatch",
        ),
    ],
)
def test_torch_compile_validator_fails_closed(
    block: JsonObject,
    expected_error: str,
) -> None:
    """A plan may launch only a coherent measured compile recipe.

    These deliberate policy failures prevent the runner from silently changing compile
    scope, backend, or shape policy after selection. Accepting any mutation would make
    the paid run execute a recipe that its benchmark never proved.
    """
    assert expected_error in _torch_compile_errors(block)


def test_torch_compile_validator_rejects_non_dict() -> None:
    """A missing torch_compile block fails closed on the preserved id."""
    assert _torch_compile_errors(None) == ("selected_runtime_missing_torch_compile",)


def test_runtime_policy_validator_accepts_both_memory_formats() -> None:
    """Contiguous eager and channels_last compiled runtime policies parse clean."""
    eager = {
        "memory_format": "contiguous",
        "ddp_static_graph": False,
        "ddp_gradient_as_bucket_view": False,
        "zero_grad_set_to_none": True,
    }

    assert _runtime_policy_errors(eager) == ()
    assert _runtime_policy_errors(_compiled_winner_runtime_policy()) == ()


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        (
            {"memory_format": "nhwc_bogus"},
            "selected_runtime_runtime_policy_memory_format_mismatch",
        ),
        (
            {"ddp_static_graph": True},
            "selected_runtime_runtime_policy_ddp_static_graph_mismatch",
        ),
        (
            {"zero_grad_set_to_none": False},
            "selected_runtime_runtime_policy_zero_grad_set_to_none_mismatch",
        ),
    ],
)
def test_runtime_policy_validator_fails_closed(
    mutation: JsonObject,
    expected_error: str,
) -> None:
    """A garbage memory_format, static_graph on, or zero_grad off fails closed."""
    block = _compiled_winner_runtime_policy()
    block.update(mutation)

    assert expected_error in _runtime_policy_errors(block)


def test_runtime_policy_validator_rejects_non_dict() -> None:
    """A missing runtime_policy block fails closed on the preserved id."""
    assert _runtime_policy_errors(None) == ("selected_runtime_missing_runtime_policy",)


def test_eps_generator_seed_separates_ranks_and_resume_segments() -> None:
    """DDP ranks and resumed segments must not share latent-noise streams.

    Rank independence is a correctness requirement because shared epsilon collapses the
    effective DDP batch. Exact seed values and compatibility with an older stream
    are not requirements; the derived uniqueness sets catch removal of either offset.
    """
    data_seed = 4242
    ranks = (0, 1)
    seed = selected_runtime_runner._eps_generator_seed  # noqa: SLF001

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


def test_resume_batch_offset_skips_only_first_loader_traversal() -> None:
    """Resume skips index batches without rereading old patch payloads."""
    sampler = torch.utils.data.SequentialSampler(range(10))
    batches = selected_runtime_runner._FirstEpochBatchOffsetSampler(  # noqa: SLF001
        sampler=sampler,
        batch_size=2,
        drop_last=True,
        completed_batches=7,
    )

    assert list(batches) == [[4, 5], [6, 7], [8, 9]]
    assert list(batches) == [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]


def test_loader_wires_resume_offset_into_batch_sampler() -> None:
    """The real loader constructor uses the index-only first-epoch offset."""
    dataset = cast(
        "PatchTrainingDataset",
        torch.utils.data.TensorDataset(torch.arange(10)),
    )
    loader = selected_runtime_runner._loader(  # noqa: SLF001
        dataset=dataset,
        batch_size=2,
        plan=parse_selected_runtime_plan(_RUNTIME_CONFIG),
        distributed=_local_distributed_context(),
        full_batch_repeated=False,
        first_epoch_batch_offset=7,
    )

    batch_sampler = cast("Sampler[list[int]]", loader.batch_sampler)
    assert isinstance(
        batch_sampler,
        selected_runtime_runner._FirstEpochBatchOffsetSampler,  # noqa: SLF001
    )
    assert list(batch_sampler) == [[4, 5], [6, 7], [8, 9]]


def test_full_resume_rejects_nonboundary_checkpoint(tmp_path: Path) -> None:
    """Only a completed 3,000-step boundary may seed a fresh Kaggle session.

    Metrics and fixed-25 artifacts commit at the same boundary cadence; accepting an
    arbitrary valid checkpoint would make later absolute-step concatenation ambiguous.
    """
    settings = _full_settings(
        tmp_path=tmp_path,
        max_train_steps=_FULL_TARGET_UPDATES,
        save_every=_FULL_HALF_EPOCH_INTERVAL,
    )
    request = SelectedRuntimeTrainRequest(
        config_path=_FULL_CONFIG,
        runtime_config=_RUNTIME_CONFIG,
        output_dir=tmp_path,
        run_name="spec0011_boundary_resume",
        data="synthetic",
        resume=tmp_path / "step_003001.pt",
        dry_run=True,
    )

    with pytest.raises(ValueError, match="completed evaluation boundary"):
        selected_runtime_runner._validate_full_resume_boundary(  # noqa: SLF001
            request=request,
            settings=settings,
            start_step=_FULL_HALF_EPOCH_INTERVAL + 1,
        )

    selected_runtime_runner._validate_full_resume_boundary(  # noqa: SLF001
        request=request,
        settings=settings,
        start_step=_FULL_HALF_EPOCH_INTERVAL,
    )


def test_train_eps_diverges_across_ranks(
    tmp_path: Path,
) -> None:
    """Different DDP ranks must draw different latent samples for one batch.

    This protects effective-batch diversity, not reproducibility: only inequality is
    derived, and no historical tensor or exact RNG draw is pinned. Reusing one generator
    seed across ranks makes both assertions fail.
    """
    settings = _full_settings(tmp_path=tmp_path, max_train_steps=1, save_every=1)

    def draw(seed_value: int) -> tuple[torch.Tensor, float]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed_value)
        eps, proof = selected_runtime_runner._train_eps(  # noqa: SLF001
            batch_size=12,
            latent_channels=LATENT_CHANNELS,
            settings=settings,
            train_generator=generator,
            device=torch.device("cpu"),
        )
        return eps, float(proof.eps_abs_mean)

    seed = selected_runtime_runner._eps_generator_seed  # noqa: SLF001
    rank0_eps, rank0_mean = draw(seed(data_seed=settings.data_seed, rank=0))
    rank1_eps, rank1_mean = draw(seed(data_seed=settings.data_seed, rank=1))

    assert not torch.equal(rank0_eps, rank1_eps)
    assert rank0_mean != rank1_mean


def test_per_rank_eps_divergent_flags_collapsed_eps() -> None:
    """Gate health must distinguish independent DDP noise from a shared stream.

    Equality across ranks is the derived collapse signal; a single-rank run has no
    comparison and therefore passes. Ignoring rank identity or accepting equal means
    would let two ranks train on the same latent sample without detection.
    """
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


def test_paid_train_eps_uses_the_rebased_resume_stream(tmp_path: Path) -> None:
    """A new rank/segment seed changes the executed post-resume epsilon stream.

    Every worker loads rank 0's checkpoint, so failing to rebase the actual generator
    would make resumed DDP ranks replay identical latent noise.
    """
    settings = _full_settings(tmp_path=tmp_path, max_train_steps=1, save_every=1)
    generator = torch.Generator(device="cpu")

    def draw(generator: torch.Generator) -> torch.Tensor:
        eps, _ = selected_runtime_runner._train_eps(  # noqa: SLF001
            batch_size=12,
            latent_channels=LATENT_CHANNELS,
            settings=settings,
            train_generator=generator,
            device=torch.device("cpu"),
        )
        return eps

    seed = selected_runtime_runner._eps_generator_seed  # noqa: SLF001
    generator.manual_seed(seed(data_seed=settings.data_seed, rank=0))
    first = draw(generator)
    generator.manual_seed(
        seed(
            data_seed=settings.data_seed,
            rank=1,
            start_step=_FULL_HALF_EPOCH_INTERVAL,
        ),
    )
    second = draw(generator)
    assert not torch.equal(first, second)


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


def test_assert_ddp_parameters_in_sync_passes_or_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cross-rank parameter divergence fails fast instead of training two models."""
    model = torch.nn.Linear(2, 2)

    def fake_is_initialized() -> bool:
        return True

    monkeypatch.setattr(
        selected_runtime_runner.dist,
        "is_initialized",
        fake_is_initialized,
    )

    # A single-process run skips the collective entirely (no behavior change).
    selected_runtime_runner._assert_ddp_parameters_in_sync(  # noqa: SLF001
        model=model,
        distributed=_local_distributed_context(),
    )

    def agree(gathered: list[object], obj: object) -> None:
        gathered[0] = obj
        gathered[1] = obj

    monkeypatch.setattr(selected_runtime_runner.dist, "all_gather_object", agree)
    # Identical fingerprints across ranks pass without raising.
    selected_runtime_runner._assert_ddp_parameters_in_sync(  # noqa: SLF001
        model=model,
        distributed=_ddp_distributed_context(rank=0),
    )

    def disagree(gathered: list[object], obj: object) -> None:
        gathered[0] = obj
        gathered[1] = (1.0e30, 2.0e30)

    monkeypatch.setattr(selected_runtime_runner.dist, "all_gather_object", disagree)
    # A divergent fingerprint (grads not synced) raises on every rank.
    with pytest.raises(RuntimeError, match="divergent parameters"):
        selected_runtime_runner._assert_ddp_parameters_in_sync(  # noqa: SLF001
            model=model,
            distributed=_ddp_distributed_context(rank=0),
        )


def test_assert_ddp_parameters_in_sync_treats_identical_nan_as_synced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bit-identical NaN params across ranks are in sync, not a spurious desync."""
    model = torch.nn.Linear(2, 2)

    def fake_is_initialized() -> bool:
        return True

    monkeypatch.setattr(
        selected_runtime_runner.dist,
        "is_initialized",
        fake_is_initialized,
    )

    def gather_distinct_nan(gathered: list[object], obj: object) -> None:
        # Real all_gather_object deserializes each rank's tuple into a DISTINCT
        # object, so the two NaN fingerprints are not the same object and nan != nan.
        del obj
        gathered[0] = (float("nan"), float("nan"))
        gathered[1] = (float("nan"), float("nan"))

    monkeypatch.setattr(
        selected_runtime_runner.dist,
        "all_gather_object",
        gather_distinct_nan,
    )
    # NaN is caught by nonfinite_count / GradScaler, not misread as a DDP desync.
    selected_runtime_runner._assert_ddp_parameters_in_sync(  # noqa: SLF001
        model=model,
        distributed=_ddp_distributed_context(rank=0),
    )

    def gather_nan_vs_finite(gathered: list[object], obj: object) -> None:
        del obj
        gathered[0] = (float("nan"), float("nan"))
        gathered[1] = (0.0, 0.0)

    monkeypatch.setattr(
        selected_runtime_runner.dist,
        "all_gather_object",
        gather_nan_vs_finite,
    )
    # One rank NaN and another finite is a genuine desync (one rank diverged).
    with pytest.raises(RuntimeError, match="divergent parameters"):
        selected_runtime_runner._assert_ddp_parameters_in_sync(  # noqa: SLF001
            model=model,
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
        batch_size=_TINY_CPU_BATCH_SIZE,
        image_size=_TINY_CPU_IMAGE_SIZE,
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
        model = build_non_equivariant_vae(
            norm_groups=settings.norm_groups,
        )
        optimizer = selected_runtime_runner.build_fastpath_optimizer(
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
        corruption_generator = torch.Generator(device="cpu")
        corruption_generator.manual_seed(settings.corruption_seed)
        train_loop = selected_runtime_runner._run_train_steps(  # noqa: SLF001
            request=request,
            resolved=resolved,
            settings=settings,
            plan=plan,
            model=model,
            checkpoint_model=model,
            latent_channels=LATENT_CHANNELS,
            optimizer=optimizer,
            scaler=scaler,
            amp=amp,
            data_surface=data_surface,
            distributed=local,
            numpy_generator=np.random.default_rng(settings.global_seed),
            train_generator=train_generator,
            corruption_generator=corruption_generator,
            eager_corruptor=InlineStainCorruptor(settings.corruption_profile),
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


# Odd schedule (Spec 0011 S9 / MF3): a non-dividing global batch makes
# updates_per_epoch odd, so target = epochs * updates_per_epoch is OFF the
# half-epoch grid. Here epochs=1, updates_per_epoch=5 -> half=2, target=5, and 5 is
# not a multiple of 2. The half-grid is {2, 4}; the terminal 5 must be force-included
# as a boundary on the producer side by the shared generator.
_ODD_TARGET_STEPS = 5
_ODD_HALF_EPOCH_INTERVAL = 2
_ODD_SAVE_EVERY = 2
_ODD_BOUNDARY_STEPS = frozenset({2, 4, 5})
# floor(300000 / 60000) == 5 == updates_per_epoch: an honest odd schedule for the gate.
_ODD_GLOBAL_BATCH = 60_000


def test_run_train_steps_checkpoints_and_validates_off_grid_terminal(  # noqa: PLR0914
    tmp_path: Path,
) -> None:
    """Spec 0011 S9: the off-grid terminal is a genuine boundary on the producer side.

    With half=2 and target=5 the terminal (5) is off the {2, 4} half-grid, so the old
    modulo producers dropped it: step 5 was never validated, never checkpointed, and
    never best-selection-eligible. Routing the producers through the shared boundary
    generator makes 5 a real boundary, while the interior boundaries {2, 4} are kept.
    """
    output_dir = tmp_path / "odd_terminal"
    request = SelectedRuntimeTrainRequest(
        config_path=_FULL_CONFIG,
        runtime_config=_RUNTIME_CONFIG,
        output_dir=output_dir,
        run_name="spec0011_odd_terminal",
        data="synthetic",
        max_train_steps=_ODD_TARGET_STEPS,
        save_every_steps=_ODD_SAVE_EVERY,
        dry_run=True,
    )
    plan = parse_selected_runtime_plan(_RUNTIME_CONFIG)
    resolved = resolve_json_config(_FULL_CONFIG)
    # Drive the full odd schedule (max == target == 5, half == save_every == 2)
    # single-process on CPU. _run_train_steps does not re-run the full-run validator,
    # so replace() installs the odd schedule directly (an odd global batch would fail
    # dual-T4 collectives locally; world_size=1 keeps this CPU-only).
    settings = replace(
        selected_runtime_runner._settings(  # noqa: SLF001
            request=request,
            resolved=resolved,
            plan=plan,
        ),
        batch_size=_TINY_CPU_BATCH_SIZE,
        image_size=_TINY_CPU_IMAGE_SIZE,
        max_train_steps=_ODD_TARGET_STEPS,
        target_train_steps=_ODD_TARGET_STEPS,
        half_epoch_interval_steps=_ODD_HALF_EPOCH_INTERVAL,
        save_every_steps=_ODD_SAVE_EVERY,
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
        model = build_non_equivariant_vae(
            norm_groups=settings.norm_groups,
        )
        optimizer = selected_runtime_runner.build_fastpath_optimizer(
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
        corruption_generator = torch.Generator(device="cpu")
        corruption_generator.manual_seed(settings.corruption_seed)
        train_loop = selected_runtime_runner._run_train_steps(  # noqa: SLF001
            request=request,
            resolved=resolved,
            settings=settings,
            plan=plan,
            model=model,
            checkpoint_model=model,
            latent_channels=LATENT_CHANNELS,
            optimizer=optimizer,
            scaler=scaler,
            amp=amp,
            data_surface=data_surface,
            distributed=local,
            numpy_generator=np.random.default_rng(settings.global_seed),
            train_generator=train_generator,
            corruption_generator=corruption_generator,
            eager_corruptor=InlineStainCorruptor(settings.corruption_profile),
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

    # Every boundary is validated, INCLUDING the off-grid terminal 5 (pre-S9 the modulo
    # producer stopped at 4); the interior boundaries {2, 4} are preserved.
    validated_steps = {int(row["optimizer_step"]) for row in train_loop.validation_rows}
    assert validated_steps == _ODD_BOUNDARY_STEPS
    terminal_views = {
        row["view"]
        for row in train_loop.validation_rows
        if int(row["optimizer_step"]) == _ODD_TARGET_STEPS
    }
    assert terminal_views == {"clean", "deterministic_denoising"}
    # The terminal is a genuine interval checkpoint (distinct from final.pt).
    assert (output_dir / "checkpoints" / f"step_{_ODD_TARGET_STEPS:06d}.pt").exists()
    # ...and best-selection-eligible (the terminal can now win best_model.pt).
    assert train_loop.best_validation_checkpoint is not None
    assert (output_dir / "checkpoints" / "best_model.pt").exists()


def test_off_grid_runner_consumers_demand_the_terminal_boundary(tmp_path: Path) -> None:
    """Spec 0011 S9 lockstep: the runner CONSUMERS also demand the off-grid terminal.

    The producer test above proves half the MF3 invariant; this proves the other half.
    A producer-only S9 change (reverting a consumer to the old open-coded ``range``)
    would drop the terminal at an odd batch while the producers still emit it. With
    half=2/target=5 the completeness checker (_validation_schedule_complete) and the
    resume-prefix validator (_validate_full_resume_validation_prefix) must both require
    step 5, which is off the {2, 4} grid (on an on-grid batch both pass either way).
    """
    request = SelectedRuntimeTrainRequest(
        config_path=_FULL_CONFIG,
        runtime_config=_RUNTIME_CONFIG,
        output_dir=tmp_path / "odd_consumers",
        run_name="spec0011_odd_consumers",
        data="synthetic",
        max_train_steps=_ODD_TARGET_STEPS,
        save_every_steps=_ODD_SAVE_EVERY,
        dry_run=True,
    )
    plan = parse_selected_runtime_plan(_RUNTIME_CONFIG)
    settings = replace(
        selected_runtime_runner._settings(  # noqa: SLF001
            request=request,
            resolved=resolve_json_config(_FULL_CONFIG),
            plan=plan,
        ),
        target_train_steps=_ODD_TARGET_STEPS,
        half_epoch_interval_steps=_ODD_HALF_EPOCH_INTERVAL,
    )
    complete_rows: list[dict[str, str]] = [
        {"optimizer_step": str(step), "view": view}
        for step in sorted(_ODD_BOUNDARY_STEPS)
        for view in settings.validation_views
    ]
    without_terminal = [
        row for row in complete_rows if int(row["optimizer_step"]) != _ODD_TARGET_STEPS
    ]

    # Completeness consumer: the off-grid terminal is required, so dropping its rows
    # flips the schedule from complete to incomplete.
    assert selected_runtime_runner._validation_schedule_complete(  # noqa: SLF001
        settings,
        complete_rows,
    )
    assert not selected_runtime_runner._validation_schedule_complete(  # noqa: SLF001
        settings,
        without_terminal,
    )
    # Resume-prefix consumer at start_step == target: the terminal rows must be present.
    selected_runtime_runner._validate_full_resume_validation_prefix(  # noqa: SLF001
        rows=complete_rows,
        settings=settings,
        start_step=_ODD_TARGET_STEPS,
    )
    with pytest.raises(ValueError, match="missing validation rows"):
        selected_runtime_runner._validate_full_resume_validation_prefix(  # noqa: SLF001
            rows=without_terminal,
            settings=settings,
            start_step=_ODD_TARGET_STEPS,
        )


def test_off_grid_gate_expects_the_terminal_interval_checkpoint() -> None:
    """Spec 0011 S9 lockstep: the gate expects the off-grid terminal checkpoint name."""
    schedule = selected_runtime_gate._RemoteFullSchedule(  # noqa: SLF001
        global_batch_size=_ODD_GLOBAL_BATCH,
        updates_per_epoch=_ODD_TARGET_STEPS,
        target_updates=_ODD_TARGET_STEPS,
        half_epoch_interval=_ODD_HALF_EPOCH_INTERVAL,
        valid=True,
    )
    names = selected_runtime_gate._full_expected_interval_checkpoint_names(  # noqa: SLF001
        schedule,
    )
    assert names == ("step_000002.pt", "step_000004.pt", "step_000005.pt")
    assert f"step_{_ODD_TARGET_STEPS:06d}.pt" in names


def test_off_grid_gate_flags_a_missing_terminal_validation_row(tmp_path: Path) -> None:
    """Spec 0011 S9 lockstep: the gate flags a missing terminal validation row."""
    columns = (
        "optimizer_step",
        "view",
        "batch_count",
        "l1_loss",
        "deterministic_eps_used",
        "corruption_strategy",
    )

    def _write_validation_csv(path: Path, steps: list[int]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for step in steps:
                for view in selected_runtime_gate.REMOTE_FULL_VALIDATION_VIEWS:
                    writer.writerow(
                        {
                            "optimizer_step": str(step),
                            "view": view,
                            "batch_count": "1",
                            "l1_loss": "0.1",
                            "deterministic_eps_used": "true",
                            "corruption_strategy": "indexed_masked",
                        },
                    )

    all_steps = sorted(_ODD_BOUNDARY_STEPS)
    complete = tmp_path / "validation_complete.csv"
    _write_validation_csv(complete, all_steps)
    missing_terminal = tmp_path / "validation_missing_terminal.csv"
    _write_validation_csv(
        missing_terminal,
        [step for step in all_steps if step != _ODD_TARGET_STEPS],
    )
    incomplete = "selected_runtime_full_output_validation_schedule_incomplete"

    # Every boundary present (incl. the off-grid terminal) -> no schedule blocker.
    assert incomplete not in selected_runtime_gate._remote_full_validation_blockers(  # noqa: SLF001
        complete,
        half_epoch_interval=_ODD_HALF_EPOCH_INTERVAL,
        target_updates=_ODD_TARGET_STEPS,
    )
    # Terminal row dropped -> the gate demands it and fails closed.
    assert incomplete in selected_runtime_gate._remote_full_validation_blockers(  # noqa: SLF001
        missing_terminal,
        half_epoch_interval=_ODD_HALF_EPOCH_INTERVAL,
        target_updates=_ODD_TARGET_STEPS,
    )


class _ValidationScaffold(NamedTuple):
    """Minimal setup for a direct ``_validation_view_row`` call (FU-017)."""

    settings: selected_runtime_runner._RunnerSettings
    plan: SelectedRuntimePlan
    amp: selected_runtime_runner._AmpExecution
    data_surface: selected_runtime_runner._DataSurface
    model: torch.nn.Module


def _open_validation_scaffold(
    tmp_path: Path,
    *,
    validation_batches_per_view: int = 1,
) -> _ValidationScaffold:
    """Build the model and CPU data surface used by the FU-017 validation tests.

    ``validation_batches_per_view`` sizes the synthetic validation set (``>= 1`` batch);
    the caller may override the cap on the returned scaffold's settings to exercise the
    full-sweep (``0``) versus capped paths.

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
        batch_size=_TINY_CPU_BATCH_SIZE,
        image_size=_TINY_CPU_IMAGE_SIZE,
        half_epoch_interval_steps=1,
        validation_batches_per_view=validation_batches_per_view,
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
    validation_batches_per_view: int | None = None,
    accumulator: torch.Tensor | None = None,
) -> Mapping[str, str]:
    settings = scaffold.settings
    if validation_batches_per_view is not None:
        settings = replace(
            settings,
            validation_batches_per_view=validation_batches_per_view,
        )
    if accumulator is None:
        # Fresh, self-zeroed per call; pass a shared buffer to exercise the production
        # cross-view reuse path (_run_scheduled_validation allocates one for all views).
        accumulator = torch.zeros(
            (2, len(selected_runtime_runner._VALIDATION_LOSS_METRIC_NAMES)),  # noqa: SLF001
            dtype=torch.float64,
        )
    return selected_runtime_runner._validation_view_row(  # noqa: SLF001
        model=scaffold.model,
        latent_channels=LATENT_CHANNELS,
        settings=settings,
        plan=scaffold.plan,
        amp=scaffold.amp,
        data_surface=scaffold.data_surface,
        optimizer_step=optimizer_step,
        view=view,
        rank=0,
        device=torch.device("cpu"),
        validation_accumulator=accumulator,
    )


def test_validation_view_full_sweep_covers_the_whole_loader(tmp_path: Path) -> None:
    """cap=0 sweeps the entire validation loader through the runner (Spec 0011 S17f).

    Full validation drives ``_validation_view_row`` over every batch of the loader,
    not a capped leading slice, so the emitted ``batch_count`` equals the loader
    length. Built with a multi-batch synthetic set so full (cap=0) and capped (cap=1)
    yield different counts -- exercising the cap=0 path end-to-end on CPU, not only the
    isolated ``_validation_batches`` helper.
    """
    scaffold = _open_validation_scaffold(tmp_path, validation_batches_per_view=3)
    try:
        loader_batches = len(scaffold.data_surface.validation_loader)
        full = _validation_row(
            scaffold,
            view="deterministic_denoising",
            optimizer_step=1,
            validation_batches_per_view=0,
        )
        capped = _validation_row(
            scaffold,
            view="deterministic_denoising",
            optimizer_step=1,
            validation_batches_per_view=1,
        )
    finally:
        selected_runtime_runner._close_data_surface(scaffold.data_surface)  # noqa: SLF001

    assert loader_batches > 1
    assert int(full["batch_count"]) == loader_batches
    assert int(capped["batch_count"]) == 1
    # Commit V (Spec 0011 S17f): every loss metric now emits an additive population-std
    # column, finite and nonnegative across a real multi-batch sweep.
    for name in selected_runtime_runner._VALIDATION_LOSS_METRIC_NAMES:  # noqa: SLF001
        std_text = full[f"{name}_std"]
        assert math.isfinite(float(std_text))
        assert float(std_text) >= 0.0


def test_clean_validation_view_consumes_no_corruption_rng(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FU-017: the clean validation view never enters the corruption machinery."""
    scaffold = _open_validation_scaffold(tmp_path)

    def _forbid_corruption(*_args: object, **_kwargs: object) -> NoReturn:
        message = "inline corruptor was invoked"
        raise RuntimeError(message)

    monkeypatch.setattr(
        selected_runtime_runner.InlineStainCorruptor,
        "forward",
        _forbid_corruption,
    )
    try:
        # The clean view is a pure passthrough: the corruptor must not fire.
        clean_row = _validation_row(scaffold, view="clean", optimizer_step=1)
        assert clean_row["view"] == "clean"
        # The same stub proves the denoising view DOES invoke corruption.
        with pytest.raises(RuntimeError, match="inline corruptor was invoked"):
            _validation_row(scaffold, view="deterministic_denoising", optimizer_step=1)
    finally:
        selected_runtime_runner._close_data_surface(scaffold.data_surface)  # noqa: SLF001


def test_denoising_validation_metrics_repeat_within_drift_tolerance(
    tmp_path: Path,
) -> None:
    """Repeated denoising evaluation must remain comparable for an unchanged model.

    The fixed ``1e-5``/``1e-7`` tolerance is deliberate speed-first policy: it accepts
    small numerical drift but catches a free-running validation RNG. The clean-view
    contrast prevents a vacuous pass if corruption disappears. This does not claim to
    prove a checkpoint-selection margin.
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

    metric_keys = (
        "loss",
        "recon_loss",
        "l1_loss",
        "ssim_loss",
        "ssim_metric",
        "kl_loss",
    )
    for key in metric_keys:
        assert float(first[key]) == pytest.approx(
            float(second[key]),
            rel=1.0e-5,
            abs=1.0e-7,
        )
    # Non-vacuity control: at the same step the ONLY difference between the clean and
    # denoising rows is the corrupted input, so their loss metrics must differ.
    assert {key: first[key] for key in metric_keys} != {
        key: clean[key] for key in metric_keys
    }


@pytest.mark.parametrize(
    ("total", "total_sq", "count", "expected"),
    [
        (0.0, 0.0, 0, 0.0),
        (4.0, 16.0, 1, 0.0),  # a single batch has zero spread
        (12.0, 56.0, 3, math.sqrt(8.0 / 3.0)),  # {2, 4, 6} -> pstdev
        (2.0, 1.9999999999, 2, 0.0),  # fp round-off clamps to 0, never NaN
    ],
)
def test_population_std_matches_the_fsq_population_convention(
    total: float,
    total_sq: float,
    count: int,
    expected: float,
) -> None:
    """S17f Commit V: the std helper is the FSQ population convention, clamped >= 0.

    Divides by N (not N-1), so a single batch is exactly 0.0 and fp cancellation in
    ``total_sq / count - mean**2`` clamps to 0 instead of producing a NaN std.
    """
    result = selected_runtime_runner._population_std(  # noqa: SLF001
        total=total,
        total_sq=total_sq,
        count=count,
    )
    assert math.isfinite(result)
    assert math.isclose(result, expected, abs_tol=1e-12)


def test_validation_view_row_aggregates_means_and_population_std(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S17f Commit V: the row is the per-batch-mean average and its population std.

    Stubbing ``compute_vae_loss`` with known per-batch values (distinct per metric and
    varying per batch) pins the on-device aggregation directly: the emitted mean must be
    the average of the batch means (value-preserving vs the old ``_mean_loss_scalars``)
    and each ``*_std`` the population std of those same per-batch values. Distinct
    per-metric means prove the six accumulator columns are not cross-wired.
    """
    metric_names = selected_runtime_runner._VALIDATION_LOSS_METRIC_NAMES  # noqa: SLF001
    recorded: dict[str, list[float]] = {name: [] for name in metric_names}

    def _stub_losses(
        _output: object,
        _target: object,
        *,
        beta: float,
        ssim_weight: float,  # noqa: ARG001
    ) -> VaeLossComponents:
        batch_index = len(recorded["loss"])
        values = {
            name: float(batch_index + offset)
            for offset, name in enumerate(metric_names, start=1)
        }
        for name, value in values.items():
            recorded[name].append(value)
        return VaeLossComponents(
            loss=torch.tensor(values["loss"], dtype=torch.float32),
            recon_loss=torch.tensor(values["recon_loss"], dtype=torch.float32),
            l1_loss=torch.tensor(values["l1_loss"], dtype=torch.float32),
            ssim_loss=torch.tensor(values["ssim_loss"], dtype=torch.float32),
            ssim_metric=torch.tensor(values["ssim_metric"], dtype=torch.float32),
            kl_loss=torch.tensor(values["kl_loss"], dtype=torch.float32),
            beta=beta,
        )

    monkeypatch.setattr(selected_runtime_runner, "compute_vae_loss", _stub_losses)
    scaffold = _open_validation_scaffold(tmp_path, validation_batches_per_view=3)
    try:
        row = _validation_row(
            scaffold,
            view="clean",
            optimizer_step=1,
            validation_batches_per_view=0,
        )
    finally:
        selected_runtime_runner._close_data_surface(scaffold.data_surface)  # noqa: SLF001

    batch_count = len(recorded["loss"])
    assert batch_count > 1  # a multi-batch sweep, so the std is non-vacuous
    assert int(row["batch_count"]) == batch_count
    for name in metric_names:
        values = recorded[name]
        expected_mean = sum(values) / len(values)
        expected_std = math.sqrt(
            sum((value - expected_mean) ** 2 for value in values) / len(values),
        )
        assert math.isclose(float(row[name]), expected_mean, rel_tol=1e-9)
        assert math.isclose(float(row[f"{name}_std"]), expected_std, rel_tol=1e-9)
    # Distinct per-metric means confirm the six columns accumulate independently.
    assert len({row[name] for name in metric_names}) == len(metric_names)


def test_shared_validation_accumulator_is_reset_between_views(tmp_path: Path) -> None:
    """S17f Commit V: the reused accumulator is zeroed per view before aggregating.

    Production ``_run_scheduled_validation`` allocates ONE accumulator and passes it
    to every view in order, so the per-view ``zero_()`` reset in
    ``_validation_view_row`` is load-bearing: without it the second
    (``deterministic_denoising``) view accumulates on top of the first view's sums while
    its ``batch_count`` covers only its own batches, inflating its means/std and
    corrupting best-checkpoint selection (which reads that view). Driving a SHARED
    accumulator across both views must reproduce the second view computed on a FRESH
    buffer byte-for-byte; deleting the reset breaks this equality.
    """
    scaffold = _open_validation_scaffold(tmp_path, validation_batches_per_view=3)
    try:
        shared = torch.zeros(
            (2, len(selected_runtime_runner._VALIDATION_LOSS_METRIC_NAMES)),  # noqa: SLF001
            dtype=torch.float64,
        )
        # Load the shared buffer with the first view, then reuse it for the second view
        # exactly as _run_scheduled_validation does (clean, then denoising).
        first = _validation_row(
            scaffold,
            view="clean",
            optimizer_step=1,
            validation_batches_per_view=0,
            accumulator=shared,
        )
        shared_second = _validation_row(
            scaffold,
            view="deterministic_denoising",
            optimizer_step=1,
            validation_batches_per_view=0,
            accumulator=shared,
        )
        reference_second = _validation_row(
            scaffold,
            view="deterministic_denoising",
            optimizer_step=1,
            validation_batches_per_view=0,
        )
    finally:
        selected_runtime_runner._close_data_surface(scaffold.data_surface)  # noqa: SLF001

    # Non-vacuity: the first view genuinely loaded nonzero sums into the shared buffer,
    # so a missing reset would actually contaminate the second view.
    assert any(
        abs(float(first[name])) > 0.0
        for name in selected_runtime_runner._VALIDATION_LOSS_METRIC_NAMES  # noqa: SLF001
    )
    assert dict(shared_second) == dict(reference_second)


def test_single_batch_validation_view_reports_zero_std(tmp_path: Path) -> None:
    """S17f Commit V: a single-batch view has exactly zero population std.

    With one batch the population variance is ``x**2 - x**2 == 0`` for every metric, so
    all ``*_std`` columns format to ``"0"`` (an N-1 divisor would divide by zero).
    """
    scaffold = _open_validation_scaffold(tmp_path)
    try:
        row = _validation_row(
            scaffold,
            view="deterministic_denoising",
            optimizer_step=1,
            validation_batches_per_view=1,
        )
    finally:
        selected_runtime_runner._close_data_surface(scaffold.data_surface)  # noqa: SLF001

    assert int(row["batch_count"]) == 1
    for name in selected_runtime_runner._VALIDATION_LOSS_METRIC_NAMES:  # noqa: SLF001
        assert row[f"{name}_std"] == "0"


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
        batch_size=_TINY_CPU_BATCH_SIZE,
        image_size=_TINY_CPU_IMAGE_SIZE,
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
        broadcast_buffers=plan.ddp_broadcast_buffers,
    )
    try:
        (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        model = build_non_equivariant_vae(
            norm_groups=settings.norm_groups,
        )
        optimizer = selected_runtime_runner.build_fastpath_optimizer(
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
        corruption_generator = torch.Generator(device="cpu")
        corruption_generator.manual_seed(settings.corruption_seed)
        selected_runtime_runner._run_train_steps(  # noqa: SLF001
            request=request,
            resolved=resolved,
            settings=settings,
            plan=plan,
            model=model,
            checkpoint_model=model,
            latent_channels=LATENT_CHANNELS,
            optimizer=optimizer,
            scaler=scaler,
            amp=amp,
            data_surface=data_surface,
            distributed=distributed,
            numpy_generator=np.random.default_rng(settings.global_seed),
            train_generator=train_generator,
            corruption_generator=corruption_generator,
            eager_corruptor=InlineStainCorruptor(settings.corruption_profile),
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
    """Active AMP/CUDA restores report both attempts and successful restoration.

    Paid-run resume gates consume booleans, so exact positive polarity is expected;
    changing either active restore result to false must fail this contract.
    """
    proof = _loaded_resume_proof(
        tmp_path,
        amp_enabled=True,
        cuda_enabled=True,
        amp_status="selected_runtime_amp_scaler_state",
        cuda_status="selected_runtime_cuda_rng_state",
    )

    assert proof["status"] == "local_pass"
    assert proof["grad_scaler_state_restore_attempted"] is True
    assert proof["grad_scaler_state_restored"] is True
    assert proof["cuda_rng_state_restore_attempted"] is True
    assert proof["cuda_rng_state_restored"] is True


def test_full_loaded_resume_proof_does_not_restore_inactive_state(
    tmp_path: Path,
) -> None:
    """AMP-off CPU resume reports inactive state as neither attempted nor restored.

    “Not applicable” status matches are valid but are not restorations; exact false
    booleans are expected, and conflating status compatibility with work must fail.
    """
    proof = _loaded_resume_proof(
        tmp_path,
        amp_enabled=False,
        cuda_enabled=False,
        amp_status="not_applicable_local_cpu_amp_disabled",
        cuda_status="not_applicable_local_cpu",
    )

    assert proof["status"] == "local_pass"
    assert proof["grad_scaler_state_restore_attempted"] is False
    assert proof["grad_scaler_state_restored"] is False
    assert proof["cuda_rng_state_restore_attempted"] is False
    assert proof["cuda_rng_state_restored"] is False


def test_full_loaded_resume_proof_rejects_mismatched_active_state(
    tmp_path: Path,
) -> None:
    """Active AMP/CUDA restore attempts with mismatched statuses fail honestly.

    A resume cannot be promoted after attempted state restoration disagrees with the
    runtime; fail status and exact false restoration results catch literal-true lies.
    """
    proof = _loaded_resume_proof(
        tmp_path,
        amp_enabled=True,
        cuda_enabled=True,
        amp_status="not_applicable_local_cpu_amp_disabled",
        cuda_status="not_applicable_local_cpu",
    )

    assert proof["status"] == "fail"
    assert proof["grad_scaler_state_restore_attempted"] is True
    assert proof["grad_scaler_state_restored"] is False
    assert proof["cuda_rng_state_restore_attempted"] is True
    assert proof["cuda_rng_state_restored"] is False


def test_full_dry_run_summary_lists_only_interval_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full summary keeps final/best separate using a tiny control-path model.

    The contract is artifact/checkpoint classification, not VAE compute. A one-parameter
    model and synthetic step result exercise the real CLI/writers while keeping this
    laptop test sub-second; real model CUDA behavior belongs to bounded Kaggle tests.
    """
    output_dir = tmp_path / "full-dry-run"

    class TinyModel(torch.nn.Module):
        latent_channels = 1

        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def build_tiny_model(*_args: object, **_kwargs: object) -> TinyModel:
        return TinyModel()

    def run_tiny_step(**kwargs: object) -> object:
        return _step_result(cast("int", kwargs["successful_optimizer_update_count"]))

    def skip_reconstruction_sample(**_kwargs: object) -> bool:
        return False

    monkeypatch.setattr(selected_runtime_runner, "build_model", build_tiny_model)
    monkeypatch.setattr(selected_runtime_runner, "_run_train_step", run_tiny_step)
    monkeypatch.setattr(
        selected_runtime_runner,
        "_write_reconstruction_sample",
        skip_reconstruction_sample,
    )

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
    assert math.isclose(cast("float", full_summary["beta_target"]), 0.01)
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
    assert math.isclose(cast("float", training_summary["beta_target"]), 0.01)
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


def test_full_resume_history_allows_checkpoint_only_session(tmp_path: Path) -> None:
    """A fresh Kaggle session keeps its own rows; earlier rows remain local."""
    settings = _full_settings(tmp_path=tmp_path, max_train_steps=4, save_every=1)
    artifacts = selected_runtime_runner._artifact_paths(tmp_path)  # noqa: SLF001

    history = selected_runtime_runner._load_resume_artifact_history(  # noqa: SLF001
        artifacts=artifacts,
        settings=settings,
        distributed=_local_distributed_context(),
        start_step=1,
    )

    assert history == selected_runtime_runner._empty_resume_artifact_history()  # noqa: SLF001


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
        model = build_non_equivariant_vae(
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
            broadcast_buffers=plan.ddp_broadcast_buffers,
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
        pre_checkpoint_rows = selected_runtime_runner._read_resume_csv_prefix(  # noqa: SLF001
            path=artifacts.train_steps,
            step_key="successful_optimizer_update_count",
            start_step=cast("int", pre_checkpoint_proof["latest_checkpoint_step"]),
            required=True,
            artifact_name="metrics/train_steps.csv",
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
        assert pre_checkpoint_rows == ()
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
        model = build_non_equivariant_vae(
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
            broadcast_buffers=plan.ddp_broadcast_buffers,
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


def test_full_output_verifier_checks_gate_health_identity_against_the_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The full verifier's gate-health identity comes from the plan, not a fixed row.

    The fixture's rows carry the committed plan's identity, so a plan naming a different
    policy must reject them. This is the full-run twin of the debug-path check: both
    verifiers derive the expectation the same way, and an unpinned copy would drift.
    """
    contract = _small_full_output_contract(monkeypatch)
    output_dir = tmp_path / "full_output"
    _write_full_output_fixture(output_dir=output_dir, contract=contract)
    payload = _committed_runtime_payload()
    payload["runtime_policy_id"] = _WINNER_POLICY_ID
    payload["selected_row_id"] = _EXPECTED_ROW_ID.replace(
        "policy_amp_fp16_conservative",
        f"policy_{_WINNER_POLICY_ID}",
    )
    replanned_path = tmp_path / "replanned_selected_runtime.json"
    _write_json(replanned_path, payload)

    blockers = verify_selected_runtime_full_output(
        output_dir=output_dir,
        selected_runtime_path=replanned_path,
    )

    assert "selected_runtime_output_gate_health_row_id_mismatch" in blockers
    assert "selected_runtime_output_gate_health_candidate_mismatch" in blockers
    assert "selected_runtime_output_gate_health_policy_mismatch" in blockers


def test_full_output_verifier_rejects_a_plan_the_parser_rejects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The full verifier keeps the hardware anchors the de-pinned identity relies on.

    The expected identity is derived from the plan, so a plan that self-declares a
    different accelerator must be rejected here exactly as on the debug path -- the
    anchors live in the parser, which this path therefore has to run.
    """
    contract = _small_full_output_contract(monkeypatch)
    output_dir = tmp_path / "full_output"
    _write_full_output_fixture(output_dir=output_dir, contract=contract)
    payload = _committed_runtime_payload()
    payload["accelerator_mode"] = "single_t4"
    tampered_path = tmp_path / "single_t4_selected_runtime.json"
    _write_json(tampered_path, payload)

    blockers = verify_selected_runtime_full_output(
        output_dir=output_dir,
        selected_runtime_path=tampered_path,
    )

    assert "selected_runtime_top_level_not_dual_t4_ddp" in blockers


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

    # Commit T (S17f): the stats are 0-dim device tensors now (buffered, never read per
    # step); materialize to compare.
    assert math.isclose(stats.x_hat_min.item(), -2.0)
    assert math.isclose(stats.x_hat_max.item(), 2.0)
    # Only -2 is < -1 and only 2 is > 1; the boundary values -1 and 1 are excluded.
    assert math.isclose(stats.frac_x_hat_lt_minus1.item(), 0.25)
    assert math.isclose(stats.frac_x_hat_gt_1.item(), 0.25)
    assert math.isclose(stats.recon_output_rms.item(), math.sqrt(10.0 / 4.0))


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
    assert math.isclose(stats.frac_x_hat_gt_1.item(), 1.0)
    assert math.isclose(stats.frac_x_hat_lt_minus1.item(), 0.0)
    assert math.isclose(stats.x_hat_min.item(), 50.0)
    assert math.isclose(stats.x_hat_max.item(), 50.0)
    assert math.isclose(stats.recon_output_rms.item(), 50.0)


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
    # The gate re-derives updates/target/half from
    # floor(REAL_TRAIN_PATCH_COUNT / global_batch) and reads world_size from the
    # committed runtime config (global_batch 24, world_size 2). Shrink the schedule to
    # the contract by patching the patch count so floor(P / global_batch) equals the
    # contract's updates_per_epoch; epochs stays the policy anchor the gate reads, and
    # target/half then derive to the contract's 8/2.
    runtime_payload = cast(
        "dict[str, object]",
        json.loads(_RUNTIME_CONFIG.read_text(encoding="utf-8")),
    )
    runtime_global_batch = cast("int", runtime_payload["global_batch_size"])
    monkeypatch.setattr(
        selected_runtime_gate,
        "REAL_TRAIN_PATCH_COUNT",
        contract.updates_per_epoch * runtime_global_batch,
    )
    monkeypatch.setattr(selected_runtime_gate, "REMOTE_FULL_EPOCHS", contract.epochs)
    # No REMOTE_FULL_VALIDATION_BATCHES_PER_VIEW patch: it is now the full-sweep
    # sentinel 0 (the summary reports 0), and the per-row batch_count is checked as
    # non-empty, not against this constant. contract.validation_batches sets the rows'
    # actual count.
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
            torch_version="2.12.0+cpu",
            cuda_version=None,
        ),
    )


def test_metric_row_records_the_step_actually_taken() -> None:
    """Train-row compile/corruption labels follow the step run, not the plan's claim."""
    plan = replace(
        parse_selected_runtime_plan(_RUNTIME_CONFIG),
        torch_compile_enabled=True,
        compile_scope="step",
    )
    amp = selected_runtime_runner._amp_execution(  # noqa: SLF001
        plan=plan,
        distributed=_local_distributed_context(),
        dry_run=True,
    )
    pending = _pending_row_from(_step_result(step=1))
    metrics = _zero_train_metrics()
    # A plan that claims compile but whose run took the eager step records eager labels
    # and the reproducible corruptor -- catching a build-gate regression.
    eager_row = selected_runtime_runner._metric_row(  # noqa: SLF001
        pending=pending,
        metrics=metrics,
        rank=0,
        plan=plan,
        amp=amp,
        checkpoint_path="",
        corruption_strategy=plan.corruption_strategy,
        compiled_step_active=False,
    )
    assert eager_row["torch_compile_enabled"] == "false"
    assert eager_row["compile_scope"] == "none"
    assert eager_row["corruption_strategy"] == plan.corruption_strategy
    compiled_row = selected_runtime_runner._metric_row(  # noqa: SLF001
        pending=pending,
        metrics=metrics,
        rank=0,
        plan=plan,
        amp=amp,
        checkpoint_path="",
        corruption_strategy=COMPILED_FASTPATH_CORRUPTION_STRATEGY,
        compiled_step_active=True,
    )
    assert compiled_row["torch_compile_enabled"] == "true"
    assert compiled_row["compile_scope"] == "step"
    assert compiled_row["corruption_strategy"] == COMPILED_FASTPATH_CORRUPTION_STRATEGY


def _distinct_step_result(
    *,
    step: int,
    offset: float,
) -> selected_runtime_runner._SelectedRuntimeStepResult:
    # Every metric a distinct value (across columns AND steps), so a swapped column, a
    # misaligned buffer index, or a wrong _TRAIN_STEP_METRIC_NAMES order is caught.
    losses = VaeLossComponents(
        loss=torch.tensor(offset + 0.1),
        recon_loss=torch.tensor(offset + 0.2),
        l1_loss=torch.tensor(offset + 0.3),
        ssim_loss=torch.tensor(offset + 0.4),
        ssim_metric=torch.tensor(offset + 0.5),
        kl_loss=torch.tensor(offset + 0.6),
        beta=offset + 0.7,
    )
    return selected_runtime_runner._SelectedRuntimeStepResult(  # noqa: SLF001
        optimizer_step_index=step - 1,
        successful_optimizer_update_count=step,
        losses=losses,
        grad_norm=torch.tensor(offset + 1.1, dtype=torch.float64),
        param_update_norm=torch.tensor(offset + 1.2, dtype=torch.float64),
        nonfinite_count=torch.tensor(step, dtype=torch.int64),
        recon_output_rms=torch.tensor(offset + 1.3),
        x_hat_min=torch.tensor(offset + 1.4),
        x_hat_max=torch.tensor(offset + 1.5),
        frac_x_hat_lt_minus1=torch.tensor(offset + 0.05),
        frac_x_hat_gt_1=torch.tensor(offset + 0.06),
        batch_size=12,
        amp_step_skipped=False,
        zero_grad_set_to_none=True,
        train_reparameterization="stochastic_seeded",
        eps_policy="stochastic_rank_generator",
        eps_seed_source="checkpointed_rank_rebased_generator",
        eps_zero_fraction=torch.tensor(offset + 0.01),
        eps_abs_mean=torch.tensor(offset + 0.02),
    )


def _direct_train_metrics(
    result: selected_runtime_runner._SelectedRuntimeStepResult,
) -> dict[str, float]:
    # Materialize each device-scalar metric directly at the buffer's fp32 storage dtype
    # (one .item() each) -- the value the deferred fp32 buffer path must reproduce. The
    # fp64 grad/param norms exercise the fp64->fp32 round the buffer applies.
    def _fp32(tensor: torch.Tensor) -> float:
        return float(tensor.to(torch.float32).item())

    losses = result.losses
    return {
        "loss": _fp32(losses.loss),
        "recon_loss": _fp32(losses.recon_loss),
        "l1_loss": _fp32(losses.l1_loss),
        "ssim_loss": _fp32(losses.ssim_loss),
        "ssim_metric": _fp32(losses.ssim_metric),
        "kl_loss": _fp32(losses.kl_loss),
        "grad_norm": _fp32(result.grad_norm),
        "param_update_norm": _fp32(result.param_update_norm),
        "recon_output_rms": _fp32(result.recon_output_rms),
        "x_hat_min": _fp32(result.x_hat_min),
        "x_hat_max": _fp32(result.x_hat_max),
        "frac_x_hat_lt_minus1": _fp32(result.frac_x_hat_lt_minus1),
        "frac_x_hat_gt_1": _fp32(result.frac_x_hat_gt_1),
        "nonfinite_count": _fp32(result.nonfinite_count),
        "eps_zero_fraction": _fp32(result.eps_zero_fraction),
        "eps_abs_mean": _fp32(result.eps_abs_mean),
    }


def _train_row_context() -> selected_runtime_runner._TrainRowContext:
    plan = parse_selected_runtime_plan(_RUNTIME_CONFIG)
    amp = selected_runtime_runner._amp_execution(  # noqa: SLF001
        plan=plan,
        distributed=_local_distributed_context(),
        dry_run=True,
    )
    return selected_runtime_runner._TrainRowContext(  # noqa: SLF001
        rank=0,
        plan=plan,
        amp=amp,
        corruption_strategy=EAGER_INLINE_STAIN_CORRUPTION_STRATEGY,
        compiled_step_active=False,
    )


def _expected_deferred_row(
    result: selected_runtime_runner._SelectedRuntimeStepResult,
    context: selected_runtime_runner._TrainRowContext,
) -> CsvRow:
    return selected_runtime_runner._metric_row(  # noqa: SLF001
        pending=selected_runtime_runner._pending_train_row(result),  # noqa: SLF001
        metrics=_direct_train_metrics(result),
        rank=context.rank,
        plan=context.plan,
        amp=context.amp,
        checkpoint_path="",
        corruption_strategy=context.corruption_strategy,
        compiled_step_active=context.compiled_step_active,
    )


def test_train_step_metric_buffer_materializes_rows_value_preserving_in_order() -> None:
    """Buffered per-step metrics flush into rows matching fp32 eager materialization.

    A single bulk ``.tolist()`` replaces ~14 per-step device->host syncs; this locks
    column alignment, per-step ordering, and that the fp32 buffer reproduces each
    metric's fp32 value exactly. The oracle materializes at fp32 because that is the
    buffer's storage dtype: the fp32-origin columns round-trip bit-exact, while the
    three fp64-reduced norms round to fp32 by design (fp-tolerant telemetry, rule 30).
    """
    context = _train_row_context()
    results = [_distinct_step_result(step=i, offset=10.0 * i) for i in (1, 2, 3)]
    buffer = selected_runtime_runner._TrainStepMetricBuffer(  # noqa: SLF001
        capacity=len(results),
        device=torch.device("cpu"),
        context=context,
    )

    rows: list[CsvRow] = []
    # An empty flush is a no-op, not an error.
    buffer.flush_into(rows)
    assert rows == []
    for result in results:
        buffer.record(result, rows)
    # Nothing materializes until the boundary flush (deferred host read).
    assert rows == []
    buffer.flush_into(rows)

    assert [row["optimizer_step"] for row in rows] == ["1", "2", "3"]
    assert rows == [_expected_deferred_row(result, context) for result in results]


def test_train_step_metric_buffer_auto_flushes_when_window_exceeds_capacity() -> None:
    """A window longer than capacity (AMP skips overshoot) still keeps every row.

    ``record`` drains the full buffer before overwriting it, so correctness never
    depends on the exact capacity; all rows survive, in order.
    """
    context = _train_row_context()
    results = [_distinct_step_result(step=i, offset=10.0 * i) for i in (1, 2, 3, 4, 5)]
    buffer = selected_runtime_runner._TrainStepMetricBuffer(  # noqa: SLF001
        capacity=2,
        device=torch.device("cpu"),
        context=context,
    )

    rows: list[CsvRow] = []
    for result in results:
        buffer.record(result, rows)
    # Capacity 2 forces auto-flushes of every full buffer before the write that would
    # overflow it; only the final partial window (one row) awaits the tail flush.
    assert len(rows) == len(results) - 1
    buffer.flush_into(rows)

    assert [row["optimizer_step"] for row in rows] == ["1", "2", "3", "4", "5"]
    assert rows == [_expected_deferred_row(result, context) for result in results]


def test_train_step_metric_buffer_never_retains_the_autograd_graph() -> None:
    """The persistent buffer must never pin a step's autograd graph.

    The real per-step ``loss`` is a backward target, and the buffer outlives the whole
    half-epoch window -- so writing a grad-attached scalar into it would keep every
    step's graph alive until the flush. ``_write_step_metrics`` detaches to prevent
    that; dropping the detach makes the buffer itself grad-tracking, which this catches.
    """
    context = _train_row_context()
    # A genuinely grad-attached loss, as a real backward target is.
    graph_loss = (torch.tensor([2.0], requires_grad=True) * 3.0).sum()
    assert graph_loss.requires_grad
    result = _distinct_step_result(step=1, offset=10.0)
    result = replace(result, losses=replace(result.losses, loss=graph_loss))
    buffer = selected_runtime_runner._TrainStepMetricBuffer(  # noqa: SLF001
        capacity=4,
        device=torch.device("cpu"),
        context=context,
    )

    rows: list[CsvRow] = []
    buffer.record(result, rows)

    stored = buffer._buffer  # noqa: SLF001
    assert stored.requires_grad is False
    assert stored.grad_fn is None
    # The value still lands, so the detach cannot be "fixed" by skipping the write.
    buffer.flush_into(rows)
    assert rows == [_expected_deferred_row(result, context)]


def test_amp_execution_records_amp_off_status_on_cuda() -> None:
    """A real CUDA amp-off run records that it executed, not the local-CPU sentinel."""
    amp_plan = parse_selected_runtime_plan(_RUNTIME_CONFIG)
    cuda = replace(_local_distributed_context(), device=torch.device("cuda"))
    on_cuda_amp = selected_runtime_runner._amp_execution(  # noqa: SLF001
        plan=amp_plan,
        distributed=cuda,
        dry_run=False,
    )
    assert on_cuda_amp.local_amp_status == EXPECTED_AMP_APPLICATION_STATUS
    assert on_cuda_amp.autocast_dtype == amp_plan.autocast_dtype

    amp_off_plan = replace(
        amp_plan,
        precision_policy="amp_off_fp32",
        amp_enabled=False,
        grad_scaler_enabled=False,
        autocast_dtype="float32",
    )
    on_cuda_off = selected_runtime_runner._amp_execution(  # noqa: SLF001
        plan=amp_off_plan,
        distributed=cuda,
        dry_run=False,
    )
    assert on_cuda_off.enabled is False
    assert on_cuda_off.local_amp_status == EXPECTED_AMP_OFF_APPLICATION_STATUS
    assert on_cuda_off.autocast_dtype == "float32"

    on_cpu_off = selected_runtime_runner._amp_execution(  # noqa: SLF001
        plan=amp_off_plan,
        distributed=_local_distributed_context(),
        dry_run=False,
    )
    assert on_cpu_off.local_amp_status == "not_executed_local_cpu"
    assert on_cpu_off.autocast_dtype == "not_executed_local_cpu"


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


def _loaded_resume_proof(
    tmp_path: Path,
    *,
    amp_enabled: bool,
    cuda_enabled: bool,
    amp_status: str,
    cuda_status: str,
) -> JsonObject:
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
    loaded = replace(
        _loaded_checkpoint_stub(),
        path=request.output_dir / "checkpoints" / "step_006250.pt",
        config_sha256=resolved.invoked_config_hash,
        effective_config_sha256=resolved.effective_config_hash,
        runtime_config_sha256=runtime_identity.sha256,
        selected_row_id=runtime_identity.selected_row_id,
        runtime_policy_id=runtime_identity.runtime_policy_id,
        amp_scaler_state_status=amp_status,
        torch_cuda_rng_state_status=cuda_status,
        ddp_sampler_progress_state_status=(
            "selected_runtime_ddp_sampler_progress"
            if cuda_enabled
            else "not_applicable_local_single_process"
        ),
    )
    distributed = (
        replace(_ddp_distributed_context(rank=0), device=torch.device("cuda", 0))
        if cuda_enabled
        else _local_distributed_context()
    )
    amp = selected_runtime_runner._AmpExecution(  # noqa: SLF001
        enabled=amp_enabled,
        grad_scaler_enabled=amp_enabled,
        grad_scaler_init_scale=16384.0,
        autocast_dtype="float16" if amp_enabled else "not_executed_local_cpu",
        requested_autocast_dtype="float16",
        local_amp_status=(
            "selected_runtime_cuda_amp_enabled"
            if amp_enabled
            else "not_executed_local_cpu"
        ),
    )
    return selected_runtime_runner._loaded_checkpoint_resume_proof(  # noqa: SLF001
        loaded=loaded,
        request=request,
        resolved=resolved,
        settings=settings,
        runtime_identity=runtime_identity,
        amp=amp,
        distributed=distributed,
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
            "optimizer_lr_scaling": {
                "scaling_applied": True,
                "rule": "sqrt",
                "reference_learning_rate": _LR_REFERENCE_LEARNING_RATE,
                "reference_global_batch_size": _LR_REFERENCE_GLOBAL_BATCH,
                "global_batch_size": _LR_REFERENCE_GLOBAL_BATCH,
                "batch_ratio_exponent": 0.5,
                "effective_learning_rate": _LR_REFERENCE_LEARNING_RATE,
            },
            "validation_batches_per_view": 0,  # 0 = full validation sweep (S17f)
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
            "optimizer_steps_completed": contract.target_updates,
            "requested_epochs": contract.epochs,
            "optimizer_updates_per_epoch": contract.updates_per_epoch,
            "half_epoch_interval_steps": contract.half_interval,
            "validation_batches_per_view": 0,  # 0 = full validation sweep (S17f)
            "validation_views": ["clean", "deterministic_denoising"],
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
        "eps_policy": "stochastic_rank_generator",
        "eps_seed_source": "checkpointed_rank_rebased_generator",
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
        # Additive per-view std columns (Spec 0011 S17f Metrics Commit V): the fixture
        # mirrors the real emitter so the end-to-end gate exercises tolerating them.
        "loss_std",
        "recon_loss_std",
        "l1_loss_std",
        "ssim_loss_std",
        "ssim_metric_std",
        "kl_loss_std",
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
            "loss_std": "0.1",
            "recon_loss_std": "0.1",
            "l1_loss_std": "0.05",
            "ssim_loss_std": "0.05",
            "ssim_metric_std": "0.05",
            "kl_loss_std": "0.001",
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
        # Commit T (S17f): the device-scalar metrics are 0-dim tensors now.
        grad_norm=torch.tensor(1.0),
        param_update_norm=torch.tensor(0.1),
        nonfinite_count=torch.tensor(0),
        recon_output_rms=torch.tensor(0.0),
        x_hat_min=torch.tensor(0.0),
        x_hat_max=torch.tensor(0.0),
        frac_x_hat_lt_minus1=torch.tensor(0.0),
        frac_x_hat_gt_1=torch.tensor(0.0),
        batch_size=12,
        amp_step_skipped=False,
        zero_grad_set_to_none=True,
        train_reparameterization="stochastic_seeded",
        eps_policy="stochastic_rank_generator",
        eps_seed_source="checkpointed_rank_rebased_generator",
        eps_zero_fraction=torch.tensor(0.0),
        eps_abs_mean=torch.tensor(0.8),
    )


def _pending_row_from(
    result: selected_runtime_runner._SelectedRuntimeStepResult,
) -> selected_runtime_runner._PendingTrainRow:
    return selected_runtime_runner._pending_train_row(result)  # noqa: SLF001


def _zero_train_metrics() -> dict[str, float]:
    return dict.fromkeys(selected_runtime_runner._TRAIN_STEP_METRIC_NAMES, 0.0)  # noqa: SLF001


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


# --- Spec 0011 S16: compiled whole-step path + drop_last flip ------------------------

_S16_SMALL_SHARD_PER_RANK = 11  # ceil(21 / 2), the padded drop_last=False yield
_S16_AMPLE_SHARD_PER_RANK = 24  # 49 // 2, the floored drop_last=True yield
_S16_COMPILED_UPDATE_COUNT = 2  # optimizer_step_index 1 -> update count 2
_S16_MIN_STOCHASTIC_EPS_ABS_MEAN = 0.1  # stochastic |eps| mean is ~0.8


class _RunnerContext(NamedTuple):
    """The runner pieces a compiled-step test drives, built on synthetic CPU data."""

    request: SelectedRuntimeTrainRequest
    resolved: ResolvedConfig
    settings: selected_runtime_runner._RunnerSettings
    plan: SelectedRuntimePlan
    distributed: selected_runtime_runner._DistributedContext
    data_surface: selected_runtime_runner._DataSurface
    model: NonEquivariantVAE
    optimizer: torch.optim.Optimizer
    amp: selected_runtime_runner._AmpExecution
    scaler: selected_runtime_runner.GradScaler
    train_generator: torch.Generator


def _runner_context(output_dir: Path, *, max_train_steps: int = 2) -> _RunnerContext:
    request = SelectedRuntimeTrainRequest(
        config_path=_FULL_CONFIG,
        runtime_config=_RUNTIME_CONFIG,
        output_dir=output_dir,
        run_name="spec0011_s16_compiled",
        data="synthetic",
        max_train_steps=max_train_steps,
        save_every_steps=1,
        dry_run=True,
    )
    plan = parse_selected_runtime_plan(_RUNTIME_CONFIG)
    resolved = resolve_json_config(_FULL_CONFIG)
    settings = replace(
        selected_runtime_runner._settings(  # noqa: SLF001
            request=request,
            resolved=resolved,
            plan=plan,
        ),
        batch_size=_TINY_CPU_BATCH_SIZE,
        image_size=_TINY_CPU_IMAGE_SIZE,
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
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    model = build_non_equivariant_vae(norm_groups=settings.norm_groups)
    optimizer = selected_runtime_runner.build_fastpath_optimizer(
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
    return _RunnerContext(
        request=request,
        resolved=resolved,
        settings=settings,
        plan=plan,
        distributed=distributed,
        data_surface=data_surface,
        model=model,
        optimizer=optimizer,
        amp=amp,
        scaler=scaler,
        train_generator=train_generator,
    )


def test_safe_drop_last_disables_only_when_a_shard_lacks_a_full_batch() -> None:
    """Spec 0011 S16 keeps drop_last=True unless it would empty a (per-rank) loader.

    The flip gives the compiled step a static batch shape; the guard prevents it from
    silently emptying a loader (which would hang ``_cycle_batches`` forever) --
    including the compounded DDP case where the per-rank shard, floor(dataset /
    world_size), is itself smaller than one batch.
    """
    local = _local_distributed_context()
    ddp = _ddp_distributed_context(rank=0)
    safe_drop_last = selected_runtime_runner._safe_drop_last  # noqa: SLF001

    assert safe_drop_last(
        dataset_size=32,
        batch_size=12,
        distributed=local,
        full_batch_repeated=False,
    )
    assert not safe_drop_last(
        dataset_size=8,
        batch_size=12,
        distributed=local,
        full_batch_repeated=False,
    )
    # DDP shards to floor(dataset / 2): 32 -> 16 >= 12 (keep), 20 -> 10 < 12 (disable).
    assert safe_drop_last(
        dataset_size=32,
        batch_size=12,
        distributed=ddp,
        full_batch_repeated=False,
    )
    assert not safe_drop_last(
        dataset_size=20,
        batch_size=12,
        distributed=ddp,
        full_batch_repeated=False,
    )
    # The fixed-selector sampler pads to whole batches, so dropping never empties it.
    assert safe_drop_last(
        dataset_size=8,
        batch_size=12,
        distributed=local,
        full_batch_repeated=True,
    )


def test_train_eps_sizes_from_realized_batch(tmp_path: Path) -> None:
    """Reparameterization eps is sized from the passed batch (Spec 0011 S16).

    Preserves the realized-batch coverage the old partial-batch integration test gave,
    directly and independent of drop_last: a smaller batch yields a smaller eps tensor.
    """
    context = _runner_context(tmp_path / "eps")
    for batch_size in (context.settings.batch_size, 8):
        eps, _proof = selected_runtime_runner._train_eps(  # noqa: SLF001
            batch_size=batch_size,
            latent_channels=LATENT_CHANNELS,
            settings=context.settings,
            train_generator=context.train_generator,
            device=context.distributed.device,
        )
        assert eps.shape[0] == batch_size
        assert eps.shape[1] == LATENT_CHANNELS
        assert eps.shape[2] == context.settings.image_size // 8


def test_train_sampler_plan_telemetry_tracks_realized_drop_last(tmp_path: Path) -> None:
    """The emitted sampler policy + epoch-sample count reflect the realized drop_last.

    Spec 0011 S16: on a DDP shard smaller than one per-rank batch, ``_safe_drop_last``
    falls back to ``drop_last=False`` (padding), so the proof must report
    ``..._drop_last_false`` and the padded ceil count -- never the hardcoded
    ``..._true`` / floor. The odd shard sizes make floor and ceil differ.
    """
    context = _runner_context(tmp_path / "sampler")
    ddp = _ddp_distributed_context(rank=0)

    # Small shard: floor(21/2)=10 < batch 12 -> drop_last=False (padded to ceil).
    small = selected_runtime_runner._train_sampler_plan(  # noqa: SLF001
        settings=context.settings,
        fixed_train_patch_count=21,
        dataset_size=21,
        batch_size=12,
        distributed=ddp,
    )
    assert small.policy == "distributed_sampler_shuffle_false_drop_last_false"
    assert small.effective_per_rank_epoch_samples == _S16_SMALL_SHARD_PER_RANK

    # Ample shard: floor(49/2)=24 >= batch 12 -> drop_last=True (floor).
    ample = selected_runtime_runner._train_sampler_plan(  # noqa: SLF001
        settings=context.settings,
        fixed_train_patch_count=49,
        dataset_size=49,
        batch_size=12,
        distributed=ddp,
    )
    assert ample.policy == "distributed_sampler_shuffle_false_drop_last_true"
    assert ample.effective_per_rank_epoch_samples == _S16_AMPLE_SHARD_PER_RANK


def test_maybe_build_compiled_step_returns_none_off_the_step_scope(
    tmp_path: Path,
) -> None:
    """Compile stays off unless the plan sets both the enable flag and step scope."""
    context = _runner_context(tmp_path / "off")
    build = selected_runtime_runner._maybe_build_compiled_step  # noqa: SLF001

    # The committed eager v5 plan: torch_compile disabled, scope "none".
    assert (
        build(
            model=context.model,
            plan=context.plan,
            settings=context.settings,
            amp=context.amp,
            device=context.distributed.device,
        )
        is None
    )
    # Enabled but a non-step scope stays eager.
    assert (
        build(
            model=context.model,
            plan=replace(
                context.plan,
                torch_compile_enabled=True,
                compile_scope="model_forward",
            ),
            settings=context.settings,
            amp=context.amp,
            device=context.distributed.device,
        )
        is None
    )
    # Step scope but the enable flag off stays eager (the both-flags gate).
    assert (
        build(
            model=context.model,
            plan=replace(
                context.plan,
                torch_compile_enabled=False,
                compile_scope="step",
            ),
            settings=context.settings,
            amp=context.amp,
            device=context.distributed.device,
        )
        is None
    )


def test_maybe_build_compiled_step_applies_recipe_and_compiles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The step-scope build sets the dynamo recipe, inline corruption, and compiles.

    Mutation-proof via spies: the dynamo config carries the plan's knobs, the
    corruptor is the inline (blake2b-free) one built from the configured profile,
    make_fastpath_step_fn gets the recipe knobs (esp. autocast_enabled=amp.enabled),
    and torch.compile is invoked with dynamic=False and the plan's backend.
    """
    context = _runner_context(tmp_path / "build")
    compiled_plan = replace(
        context.plan,
        torch_compile_enabled=True,
        compile_scope="step",
        compile_backend="eager",
        optimize_ddp="ddp_optimizer",
        compiled_autograd=False,
        reorder_compute_comm_overlap=True,
    )
    corruptor_profiles: list[StainCorruptionProfile] = []
    real_inline = selected_runtime_runner.InlineStainCorruptor

    def spy_inline(profile: StainCorruptionProfile) -> InlineStainCorruptor:
        corruptor_profiles.append(profile)
        return real_inline(profile)

    monkeypatch.setattr(selected_runtime_runner, "InlineStainCorruptor", spy_inline)
    make_calls: list[dict[str, object]] = []
    real_make = selected_runtime_runner.make_fastpath_step_fn

    def spy_make(  # noqa: PLR0913
        model: torch.nn.Module,
        corruptor: torch.nn.Module,
        *,
        ssim_weight: float,
        autocast_dtype: torch.dtype,
        autocast_enabled: bool,
        autocast_cache_enabled: bool,
    ) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], FastpathStepOutput]:
        make_calls.append(
            {
                "model": model,
                "ssim_weight": ssim_weight,
                "autocast_dtype": autocast_dtype,
                "autocast_enabled": autocast_enabled,
                "autocast_cache_enabled": autocast_cache_enabled,
            },
        )
        return real_make(
            model,
            corruptor,
            ssim_weight=ssim_weight,
            autocast_dtype=autocast_dtype,
            autocast_enabled=autocast_enabled,
            autocast_cache_enabled=autocast_cache_enabled,
        )

    monkeypatch.setattr(selected_runtime_runner, "make_fastpath_step_fn", spy_make)
    compile_calls: list[dict[str, object]] = []

    def fake_compile(step_fn: object, **kwargs: object) -> object:
        compile_calls.append(kwargs)
        return step_fn

    monkeypatch.setattr(selected_runtime_runner.torch, "compile", fake_compile)

    step_fn = selected_runtime_runner._maybe_build_compiled_step(  # noqa: SLF001
        model=context.model,
        plan=compiled_plan,
        settings=context.settings,
        amp=context.amp,
        device=context.distributed.device,
    )

    assert step_fn is not None
    # The dynamo config is deliberately NOT applied here -- and cannot be: the runner no
    # longer imports apply_fastpath_dynamo_config at all. wrap_fastpath_ddp owns it and
    # applies it immediately before constructing DDP, which latches optimize_ddp.
    assert not hasattr(selected_runtime_runner, "apply_fastpath_dynamo_config")
    assert corruptor_profiles == [context.settings.corruption_profile]
    assert compile_calls == [
        {
            "dynamic": False,
            "backend": "eager",
            "mode": None,
            "options": None,
        },
    ]
    # The step_fn is built with the exact recipe knobs, above all
    # autocast_enabled=amp.enabled (the eager-parity / CPU-testability knob).
    assert len(make_calls) == 1
    assert make_calls[0]["model"] is context.model
    assert make_calls[0]["ssim_weight"] == context.settings.ssim_weight
    assert make_calls[0]["autocast_dtype"] == selected_runtime_runner._autocast_dtype(  # noqa: SLF001
        compiled_plan.autocast_dtype,
    )
    assert make_calls[0]["autocast_enabled"] is context.amp.enabled
    assert make_calls[0]["autocast_cache_enabled"] is True


def test_maybe_compile_model_forward_applies_complete_recipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A legal model-forward plan consumes mode, cudagraph, and option axes."""
    model = build_non_equivariant_vae(norm_groups=8)
    plan = replace(
        parse_selected_runtime_plan(_RUNTIME_CONFIG),
        torch_compile_enabled=True,
        compile_scope="model_forward",
        compile_backend="inductor",
        compile_mode="artifact-mode",
        cudagraphs="disabled",
        inductor_options_json='{"artifact.flag":true}',
    )

    def fake_resolve(**_kwargs: object) -> tuple[None, dict[str, object]]:
        return None, {"artifact.flag": True, "triton.cudagraphs": False}

    monkeypatch.setattr(
        selected_runtime_runner,
        "resolve_fastpath_compile_invocation",
        fake_resolve,
    )
    calls: list[tuple[object, dict[str, object]]] = []
    compiled = torch.nn.Identity()

    def fake_compile(target: object, **kwargs: object) -> object:
        calls.append((target, kwargs))
        return compiled

    monkeypatch.setattr(selected_runtime_runner.torch, "compile", fake_compile)

    result = selected_runtime_runner._maybe_compile_model_forward(  # noqa: SLF001
        model=model,
        plan=plan,
    )

    assert result is compiled
    assert calls == [
        (
            model,
            {
                "dynamic": False,
                "backend": "inductor",
                "mode": None,
                "options": {
                    "artifact.flag": True,
                    "triton.cudagraphs": False,
                },
            },
        ),
    ]


def test_nccl_environment_requires_strings_and_records_effective_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NCCL overrides are validated, applied, and observed canonically."""
    plan = replace(
        parse_selected_runtime_plan(_RUNTIME_CONFIG),
        nccl_environment_json='{"NCCL_ALGO":"Ring","NCCL_NCHANNELS":"2"}',
    )
    monkeypatch.delenv("NCCL_ALGO", raising=False)
    monkeypatch.delenv("NCCL_NCHANNELS", raising=False)

    selected_runtime_runner._apply_selected_nccl_environment(plan)  # noqa: SLF001

    assert (
        selected_runtime_runner._effective_selected_nccl_environment(  # noqa: SLF001
            plan,
        )
        == '{"NCCL_ALGO":"Ring","NCCL_NCHANNELS":"2"}'
    )
    invalid = _compiled_winner_runtime_policy()
    invalid["nccl_environment_json"] = '{"NCCL_NCHANNELS":2}'
    assert (
        "selected_runtime_runtime_policy_nccl_environment_mismatch"
        in _runtime_policy_errors(invalid)
    )


def test_fastpath_dynamo_knobs_carry_the_plan_recipe_or_none() -> None:
    """The plan's dynamo knobs reach the DDP wrapper, which owns the apply ordering.

    ``wrap_fastpath_ddp`` applies these immediately before constructing DDP (which
    latches ``optimize_ddp``), so the runner's job is only to hand over the right
    knobs -- or ``None`` on the eager plan, where nothing compiles. The ORDER itself is
    pinned structurally, in
    ``test_fastpath_recipe.py`` ->
    ``test_wrap_fastpath_ddp_applies_dynamo_config_before_constructing_ddp``.
    """
    eager_plan = parse_selected_runtime_plan(_RUNTIME_CONFIG)

    assert selected_runtime_runner._fastpath_dynamo_knobs(eager_plan) is None  # noqa: SLF001

    compiled_plan = replace(
        eager_plan,
        torch_compile_enabled=True,
        compile_scope="step",
        optimize_ddp="python_reducer",
        compiled_autograd=True,
        reorder_compute_comm_overlap=True,
    )

    knobs = selected_runtime_runner._fastpath_dynamo_knobs(compiled_plan)  # noqa: SLF001

    assert knobs == FastpathDynamoKnobs(
        optimize_ddp="python_reducer",
        compiled_autograd=True,
        reorder_compute_comm_overlap=True,
    )


def test_run_compiled_train_step_populates_telemetry(tmp_path: Path) -> None:
    """The compiled step maps every telemetry field, not just a finite loss.

    Driven with an UNCOMPILED ``make_fastpath_step_fn`` closure (the spec's CPU
    "compile off" path). Mutation-proof for the hand-written result construction: a
    column swap (e.g. ``kl_loss=output.recon_loss``) breaks the loss identities below.
    The identities hold exactly within the single forward (independent of the random
    corruption values), and run at ``optimizer_step_index=1`` so beta=1.0 makes the KL
    term load-bearing.
    """
    context = _runner_context(tmp_path / "step")
    batch = cast("PatchTrainingBatch", next(iter(context.data_surface.train_loader)))
    corruptor = InlineStainCorruptor(context.settings.corruption_profile)
    step_fn = make_fastpath_step_fn(
        context.model,
        corruptor,
        ssim_weight=context.settings.ssim_weight,
        autocast_dtype=torch.float32,
        autocast_enabled=context.amp.enabled,
    )
    beta_value = beta_for_step(
        optimizer_step_index=1,
        max_optimizer_steps=context.settings.max_train_steps,
        target_beta=context.settings.beta_target,
        warmup_fraction=context.settings.beta_warmup_fraction,
    )

    result = selected_runtime_runner._run_compiled_train_step(  # noqa: SLF001
        compiled_step_fn=step_fn,
        model=context.model,
        latent_channels=LATENT_CHANNELS,
        optimizer=context.optimizer,
        scaler=context.scaler,
        settings=context.settings,
        plan=context.plan,
        amp=context.amp,
        batch=batch,
        optimizer_step_index=1,
        successful_optimizer_update_count=2,
        train_generator=context.train_generator,
        device=context.distributed.device,
    )

    assert result.batch_size == context.settings.batch_size
    # Commit T (S17f): the device-scalar metrics are 0-dim tensors now.
    assert math.isfinite(result.grad_norm.item())
    assert math.isfinite(result.param_update_norm.item())
    assert int(result.nonfinite_count.item()) == 0
    # No GradScaler on the CPU dry run, so the optimizer step is never skipped.
    assert result.amp_step_skipped is False
    assert result.successful_optimizer_update_count == _S16_COMPILED_UPDATE_COUNT

    # Loss-component mapping: the identities that define the composite VAE loss. A
    # swapped/dropped component (kl<->recon, ssim_loss<->ssim_metric, etc.) breaks one.
    total = float(result.losses.loss.detach())
    recon = float(result.losses.recon_loss.detach())
    l1 = float(result.losses.l1_loss.detach())
    ssim_loss = float(result.losses.ssim_loss.detach())
    ssim_metric = float(result.losses.ssim_metric.detach())
    kl = float(result.losses.kl_loss.detach())
    assert math.isclose(result.losses.beta, beta_value)
    assert math.isclose(
        recon,
        l1 + context.settings.ssim_weight * ssim_loss,
        rel_tol=1e-4,
        abs_tol=1e-6,
    )
    assert math.isclose(ssim_loss, 1.0 - ssim_metric, rel_tol=1e-4, abs_tol=1e-6)
    assert math.isclose(
        total,
        recon + result.losses.beta * kl,
        rel_tol=1e-4,
        abs_tol=1e-6,
    )
    # Eps-proof mapping: the full config reparameterizes stochastically.
    assert result.eps_policy == "stochastic_rank_generator"
    assert result.eps_abs_mean > _S16_MIN_STOCHASTIC_EPS_ABS_MEAN


@pytest.mark.parametrize("step_kind", ["eager", "compiled"])
@pytest.mark.parametrize("foreach", [True, False])
def test_runner_step_paths_apply_gradient_clip_foreach(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    step_kind: str,
    *,
    foreach: bool,
) -> None:
    """Both runner step bodies pass the parsed foreach value to gradient clipping.

    ``True`` is the legacy v5 effective behavior; marker-backed future plans may carry
    ``False``. Each real tiny CPU step must reach the clip call with the distinguishing
    value, so a hardcoded or dropped argument fails one of the four cases.
    """
    context = _runner_context(tmp_path / f"foreach-{step_kind}-{foreach}")
    plan = replace(context.plan, gradient_clip_foreach=foreach)
    batch = cast("PatchTrainingBatch", next(iter(context.data_surface.train_loader)))
    observed: list[bool | None] = []
    original_clip = selected_runtime_runner.nn.utils.clip_grad_norm_

    def spy_clip(
        parameters: Iterable[torch.Tensor],
        max_norm: float,
        *,
        foreach: bool | None = None,
    ) -> torch.Tensor:
        observed.append(foreach)
        return original_clip(
            parameters,
            max_norm=max_norm,
            foreach=foreach,
        )

    monkeypatch.setattr(
        selected_runtime_runner.nn.utils,
        "clip_grad_norm_",
        spy_clip,
    )
    try:
        if step_kind == "eager":
            selected_runtime_runner._run_train_step(  # noqa: SLF001
                model=context.model,
                latent_channels=LATENT_CHANNELS,
                optimizer=context.optimizer,
                scaler=context.scaler,
                settings=context.settings,
                plan=plan,
                amp=context.amp,
                batch=batch,
                optimizer_step_index=0,
                successful_optimizer_update_count=1,
                train_generator=context.train_generator,
                corruption_generator=torch.Generator(device="cpu"),
                eager_corruptor=InlineStainCorruptor(
                    context.settings.corruption_profile,
                ),
                device=context.distributed.device,
            )
        else:
            step_fn = make_fastpath_step_fn(
                context.model,
                InlineStainCorruptor(context.settings.corruption_profile),
                ssim_weight=context.settings.ssim_weight,
                autocast_dtype=torch.float32,
                autocast_enabled=context.amp.enabled,
            )
            selected_runtime_runner._run_compiled_train_step(  # noqa: SLF001
                compiled_step_fn=step_fn,
                model=context.model,
                latent_channels=LATENT_CHANNELS,
                optimizer=context.optimizer,
                scaler=context.scaler,
                settings=context.settings,
                plan=plan,
                amp=context.amp,
                batch=batch,
                optimizer_step_index=0,
                successful_optimizer_update_count=1,
                train_generator=context.train_generator,
                device=context.distributed.device,
            )
    finally:
        selected_runtime_runner._close_data_surface(context.data_surface)  # noqa: SLF001

    assert observed == [foreach]


def test_eager_runner_transfers_one_uint8_batch_before_device_corruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The eager control matches compiled transport instead of shipping two fp32 views.

    This is the FSQ-floor B4 invariant: an eager/compiled throughput comparison is
    dishonest if eager normalizes and corrupts on CPU, then transfers clean and corrupt
    fp32 tensors. The spy distinguishes the intended single uint8 transfer from that
    retired two-transfer path while a real tiny step proves the downstream math runs.
    """
    context = _runner_context(tmp_path / "eager-uint8-h2d")
    batch = cast("PatchTrainingBatch", next(iter(context.data_surface.train_loader)))
    transferred_dtypes: list[torch.dtype] = []
    original_to_device = selected_runtime_runner._to_device  # noqa: SLF001

    def spy_to_device(
        tensor: torch.Tensor,
        *,
        device: torch.device,
        plan: SelectedRuntimePlan,
    ) -> torch.Tensor:
        transferred_dtypes.append(tensor.dtype)
        return original_to_device(tensor, device=device, plan=plan)

    monkeypatch.setattr(selected_runtime_runner, "_to_device", spy_to_device)
    try:
        result = selected_runtime_runner._run_train_step(  # noqa: SLF001
            model=context.model,
            latent_channels=LATENT_CHANNELS,
            optimizer=context.optimizer,
            scaler=context.scaler,
            settings=context.settings,
            plan=context.plan,
            amp=context.amp,
            batch=batch,
            optimizer_step_index=0,
            successful_optimizer_update_count=1,
            train_generator=context.train_generator,
            corruption_generator=torch.Generator(device="cpu"),
            eager_corruptor=InlineStainCorruptor(
                context.settings.corruption_profile,
            ),
            device=context.distributed.device,
        )
    finally:
        selected_runtime_runner._close_data_surface(context.data_surface)  # noqa: SLF001

    assert transferred_dtypes == [torch.uint8]
    assert result.batch_size == batch.images_uint8.shape[0]


def test_run_train_steps_takes_the_compiled_branch_when_a_step_fn_is_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With a compiled step_fn the loop drives the compiled step, never the eager one.

    Guards against a vacuous pass: the eager ``_run_train_step`` is forbidden, so a
    completed run can only have driven the compiled branch every step.
    """
    context = _runner_context(tmp_path / "loop")
    compiled_plan = replace(
        context.plan,
        torch_compile_enabled=True,
        compile_scope="step",
        compile_backend="eager",
    )
    corruptor = InlineStainCorruptor(context.settings.corruption_profile)
    step_fn = make_fastpath_step_fn(
        context.model,
        corruptor,
        ssim_weight=context.settings.ssim_weight,
        autocast_dtype=torch.float32,
        autocast_enabled=context.amp.enabled,
    )
    # Forbid the eager step: taking the eager branch raises, so a completed run with
    # valid telemetry proves the compiled branch drove every step (the loop calls one
    # branch per step, and _run_compiled_train_step runs unpatched).

    def forbid_eager(**_kwargs: object) -> object:
        message = "the eager step must not run when a compiled step_fn is present"
        raise AssertionError(message)

    monkeypatch.setattr(selected_runtime_runner, "_run_train_step", forbid_eager)

    try:
        train_loop = selected_runtime_runner._run_train_steps(  # noqa: SLF001
            request=context.request,
            resolved=context.resolved,
            settings=context.settings,
            plan=compiled_plan,
            model=context.model,
            checkpoint_model=context.model,
            compiled_step_fn=step_fn,
            latent_channels=LATENT_CHANNELS,
            optimizer=context.optimizer,
            scaler=context.scaler,
            amp=context.amp,
            data_surface=context.data_surface,
            distributed=context.distributed,
            numpy_generator=np.random.default_rng(context.settings.global_seed),
            train_generator=context.train_generator,
            corruption_generator=torch.Generator(device="cpu"),
            eager_corruptor=InlineStainCorruptor(context.settings.corruption_profile),
            runtime_identity=selected_runtime_runner._runtime_identity(  # noqa: SLF001
                compiled_plan,
            ),
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
        selected_runtime_runner._close_data_surface(context.data_surface)  # noqa: SLF001

    assert train_loop.last_result.batch_size == context.settings.batch_size
    assert math.isfinite(train_loop.last_result.grad_norm.item())
    # The compiled branch labels every train row with the inline corruptor and the
    # compiled scope it actually ran, not the plan's declared blake2b strategy.
    assert train_loop.metric_rows
    assert all(
        row["corruption_strategy"] == COMPILED_FASTPATH_CORRUPTION_STRATEGY
        for row in train_loop.metric_rows
    )
    assert all(row["torch_compile_enabled"] == "true" for row in train_loop.metric_rows)
    assert all(row["compile_scope"] == "step" for row in train_loop.metric_rows)
