# Copyright 2026 HiperMaximus
# ruff: noqa: B010, DOC201, E501, NPY002, PLR2004, PT011, RUF069, S311
"""Mutation-sensitive CPU checks for the locked Spec 0036 training core."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from eqvae.data.latent_shards import (
    LATENT_RECORD_BYTES,
    LATENT_SHARD_HEADER_SIZE,
    LATENT_VALUES,
)
from eqvae.data.supervised_latents import (
    CatalogPart,
    LogicalPointer,
    PhysicalRead,
    SupervisedLatentStore,
)
from eqvae.training import mil_training as mil_training_module
from eqvae.training.mil_training import (
    BOOTSTRAP_SEED,
    CALIBRATION_WSI_IDS,
    EARLY_STOPPING_START_UPDATE,
    MAX_SCALE_BACKOFFS,
    PATIENCE_CHECKS,
    PEAK_LEARNING_RATE,
    TOTAL_UPDATES,
    WARMUP_START_RATIO,
    WARMUP_UPDATES,
    BestSelectionState,
    BranchNumericalError,
    StepScaler,
    ValidationMetrics,
    branch_checkpoint_payload,
    calibrate_grad_scaler,
    capture_rng_state,
    class_weight_for_label,
    committed_update_learning_rate,
    compute_validation_metrics,
    diagnosis_stratified_paired_bootstrap,
    is_validation_boundary,
    load_branch_checkpoint,
    make_default_grad_scaler,
    restore_rng_state,
    save_branch_checkpoint,
    training_epoch_order,
    update_best_selection,
    validate_branch_checkpoint_payload,
    validate_default_grad_scaler,
    weighted_amp_step,
)


class _TrackingGradScaler:
    """Record lifecycle calls while retaining real CPU GradScaler semantics."""

    step_calls: int
    update_calls: int
    scaled_values: list[float]

    def __init__(self, *, init_scale: float = 16.0, growth_interval: int = 2) -> None:
        """Create an enabled CPU scaler suitable for deterministic unit checks."""
        self.inner = torch.amp.GradScaler(
            "cpu",
            init_scale=init_scale,
            growth_interval=growth_interval,
        )
        self.step_calls = 0
        self.update_calls = 0
        self.scaled_values = []

    def scale(self, output: Tensor) -> Tensor:
        """Delegate scaling while preserving a narrow protocol type."""
        self.scaled_values.append(float(output.detach().item()))
        return self.inner.scale(output)

    def step(self, optimizer: torch.optim.Optimizer) -> object | None:
        """Count every required commit-or-skip scaler step."""
        self.step_calls += 1
        return self.inner.step(optimizer)

    def update(self, new_scale: float | Tensor | None = None) -> None:
        """Count every required scaler lifecycle update."""
        self.update_calls += 1
        self.inner.update(new_scale)

    def get_scale(self) -> float:
        """Expose the delegated current scale."""
        return float(self.inner.get_scale())

    def state_dict(self) -> dict[str, object]:
        """Expose a copy-compatible scaler checkpoint."""
        return cast("dict[str, object]", self.inner.state_dict())

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        """Restore a previously captured scaler checkpoint."""
        self.inner.load_state_dict(cast("dict[str, Any]", state_dict))


class _LazyOptimizerStateGradScaler(_TrackingGradScaler):
    """Mimic fused AdamW's permitted first-step state initialization."""

    def step(self, optimizer: torch.optim.Optimizer) -> object | None:
        """Create harmless optimizer bookkeeping before the first skipped step."""
        if self.step_calls == 0:
            parameter = cast("nn.Parameter", optimizer.param_groups[0]["params"][0])
            optimizer.state[parameter]["lazy_amp_bookkeeping"] = torch.zeros(())
        return super().step(optimizer)


def _execute_untrusted_checkpoint(path: str) -> None:
    Path(path).write_text("executed", encoding="utf-8")


class _UntrustedCheckpoint:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self) -> tuple[object, tuple[str]]:
        return _execute_untrusted_checkpoint, (str(self.marker),)


def _metric(*, macro_f1: float, mean_ce: float) -> ValidationMetrics:
    truths = [0] * 5 + [1] * 6 + [2] * 8 + [3] * 2 + [4] * 2
    return replace(
        compute_validation_metrics(
            truths,
            truths,
            [mean_ce] * len(truths),
        ),
        macro_f1=macro_f1,
    )


def _contract_hashes() -> dict[str, str]:
    return {
        "config": "config-hash",
        "input": "input-hash",
        "source": "source-hash",
        "runtime": "runtime-hash",
    }


def _checkpoint_payload(
    *,
    branch_name: str = "normal_vae",
    committed_update: int = 53,
) -> dict[str, object]:
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cpu")
    loss = cast("Tensor", model(torch.ones(1, 2))).sum()
    scaled_loss = scaler.scale(loss)
    scaled_loss.backward()  # pyright: ignore[reportUnknownMemberType]
    scaler.step(optimizer)
    scaler.update()
    metrics = _metric(macro_f1=0.4, mean_ce=1.2)
    epoch, cursor = divmod(committed_update, 106)
    train_history = tuple(
        {"committed_update": update} for update in range(1, committed_update + 1)
    )
    selection = BestSelectionState()
    validation_rows: list[dict[str, object]] = []
    for boundary in range(53, committed_update + 1, 53):
        update = update_best_selection(selection, metrics, boundary=boundary)
        selection = update.state
        validation_rows.append({
            "boundary": boundary,
            "epoch": (boundary - 1) // 106,
            "within_epoch_cursor": (boundary - 1) % 106 + 1,
            **metrics.to_dict(),
            "improved": update.improved,
            "non_improving_checks": selection.non_improving_checks,
        })
    return branch_checkpoint_payload(
        branch_name=branch_name,
        model=model,
        optimizer=optimizer,
        scaler=cast("StepScaler", scaler),
        committed_update=committed_update,
        epoch=epoch,
        within_epoch_cursor=cursor,
        current_order=training_epoch_order(epoch),
        train_history=train_history,
        validation_history=validation_rows,
        best_selection=selection,
        contract_hashes=_contract_hashes(),
        access_transcript=({"wsi_id": 1},),
    )


def test_committed_update_lr_uses_exact_locked_indexing() -> None:
    """The schedule is committed-update driven, so retries cannot consume warmup or decay."""
    assert committed_update_learning_rate(1) == pytest.approx(
        PEAK_LEARNING_RATE * WARMUP_START_RATIO,
    )
    assert committed_update_learning_rate(WARMUP_UPDATES) == PEAK_LEARNING_RATE
    first_decay_update = WARMUP_UPDATES + 1
    expected_first_decay = PEAK_LEARNING_RATE * (
        0.01
        + 0.99
        * (
            1.0
            + math.cos(
                math.pi
                * (first_decay_update - WARMUP_UPDATES)
                / (TOTAL_UPDATES - WARMUP_UPDATES),
            )
        )
        / 2.0
    )
    assert committed_update_learning_rate(first_decay_update) == expected_first_decay
    assert committed_update_learning_rate(TOTAL_UPDATES) == pytest.approx(
        PEAK_LEARNING_RATE * 0.01,
    )
    with pytest.raises(ValueError):
        committed_update_learning_rate(0)
    with pytest.raises(ValueError):
        committed_update_learning_rate(TOTAL_UPDATES + 1)


def test_validation_boundaries_are_only_each_half_epoch() -> None:
    """Selection and stopping may run only after 53 or 106 committed WSIs per epoch."""
    assert is_validation_boundary(53)
    assert is_validation_boundary(106)
    assert is_validation_boundary(159)
    assert is_validation_boundary(TOTAL_UPDATES)
    assert not is_validation_boundary(0)
    assert not is_validation_boundary(52)
    assert not is_validation_boundary(54)
    assert not is_validation_boundary(TOTAL_UPDATES + 53)


def test_validation_metrics_pin_confusion_orientation_and_zero_division() -> None:
    """Absent predictions must score zero without hiding fixed-class support or confusion."""
    metrics = compute_validation_metrics(
        [0, 1, 2, 3, 4],
        [0, 0, 0, 0, 0],
        [0.5, 1.0, 1.5, 2.0, 2.5],
    )
    assert metrics.wsi_count == 5
    assert metrics.mean_ce == 1.5
    assert metrics.accuracy == 0.2
    assert metrics.balanced_accuracy == 0.2
    assert metrics.macro_f1 == pytest.approx(1.0 / 15.0)
    assert metrics.per_class[0].precision == 0.2
    assert metrics.per_class[0].recall == 1.0
    assert metrics.per_class[0].f1 == pytest.approx(1.0 / 3.0)
    assert all(
        item.precision == item.recall == item.f1 == 0.0
        for item in metrics.per_class[1:]
    )
    assert metrics.confusion_matrix == (
        (1, 0, 0, 0, 0),
        (1, 0, 0, 0, 0),
        (1, 0, 0, 0, 0),
        (1, 0, 0, 0, 0),
        (1, 0, 0, 0, 0),
    )
    assert class_weight_for_label(4) == 2.65


def test_selector_ties_and_patience_remain_branch_local() -> None:
    """Independent branches tie-break normally but count misses only after epoch 50."""
    baseline = _metric(macro_f1=0.4, mean_ce=1.0)
    normal = update_best_selection(
        BestSelectionState(),
        baseline,
        boundary=53,
    ).state
    so2 = update_best_selection(
        BestSelectionState(),
        _metric(macro_f1=0.3, mean_ce=0.8),
        boundary=53,
    ).state
    better_ce = update_best_selection(
        normal,
        _metric(macro_f1=0.4, mean_ce=0.9),
        boundary=106,
    )
    assert better_ce.improved
    normal = better_ce.state
    assert normal.non_improving_checks == 0
    assert so2.best_metrics is not normal.best_metrics

    pre_patience_boundaries = [
        update
        for update in range(159, TOTAL_UPDATES + 1)
        if is_validation_boundary(update) and update <= EARLY_STOPPING_START_UPDATE
    ]
    for boundary in pre_patience_boundaries:
        result = update_best_selection(normal, normal.best_metrics, boundary=boundary)  # type: ignore[arg-type]
        assert not result.improved
        normal = result.state
        assert normal.non_improving_checks == 0
        assert not normal.stopped
    assert pre_patience_boundaries[-1] == 5_300

    patience_boundaries = [
        update
        for update in range(EARLY_STOPPING_START_UPDATE + 1, TOTAL_UPDATES + 1)
        if is_validation_boundary(update)
    ]
    assert patience_boundaries[0] == 5_353
    assert patience_boundaries[PATIENCE_CHECKS - 1] == 7_420
    for index, boundary in enumerate(patience_boundaries[:PATIENCE_CHECKS], start=1):
        result = update_best_selection(normal, normal.best_metrics, boundary=boundary)  # type: ignore[arg-type]
        assert not result.improved
        normal = result.state
        assert normal.non_improving_checks == index
        assert normal.stopped is (index == PATIENCE_CHECKS)
    assert so2.non_improving_checks == 0


def test_diagnosis_stratified_paired_bootstrap_is_seeded_and_directional() -> None:
    """Paired resampling must be reproducible and report normal-minus-SO(2)."""
    truths = tuple([0] * 5 + [1] * 6 + [2] * 8 + [3] * 2 + [4] * 2)
    normal = tuple(
        (truth if index % 4 else (truth + 1) % 5) for index, truth in enumerate(truths)
    )
    so2 = tuple(
        (truth if index % 3 else (truth + 2) % 5) for index, truth in enumerate(truths)
    )
    first = diagnosis_stratified_paired_bootstrap(
        truths,
        normal,
        so2,
        replicates=256,
        seed=BOOTSTRAP_SEED,
    )
    second = diagnosis_stratified_paired_bootstrap(
        truths,
        normal,
        so2,
        replicates=256,
        seed=BOOTSTRAP_SEED,
    )
    changed_seed = diagnosis_stratified_paired_bootstrap(
        truths,
        normal,
        so2,
        replicates=256,
        seed=BOOTSTRAP_SEED + 1,
    )
    observed = (
        compute_validation_metrics(truths, normal, [0.0] * len(truths)).accuracy
        - compute_validation_metrics(
            truths,
            so2,
            [0.0] * len(truths),
        ).accuracy
    )
    assert first == second
    assert first["accuracy"].difference == observed
    assert first["accuracy"] != changed_seed["accuracy"]


def test_default_grad_scaler_factory_and_validation_work_on_cpu() -> None:
    """Pinned ordinary defaults must be inspectable without requiring a CUDA device."""
    disabled = make_default_grad_scaler(enabled=False)
    validate_default_grad_scaler(disabled, expected_enabled=False)
    validate_default_grad_scaler(torch.amp.GradScaler("cpu"))
    with pytest.raises(ValueError, match="defaults differ"):
        validate_default_grad_scaler(torch.amp.GradScaler("cpu", init_scale=32.0))
    with pytest.raises(ValueError, match="enabled state"):
        validate_default_grad_scaler(disabled)


def test_weighted_amp_step_scales_the_fp32_weighted_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finite CPU AMP must commit one class-weighted gradient at the requested LR."""
    model = nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(2.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=99.0)
    scaler = _TrackingGradScaler(init_scale=16.0, growth_interval=2_000)

    def unexpected_failure_diagnostics(_model: nn.Module) -> list[dict[str, object]]:
        raise AssertionError

    monkeypatch.setattr(
        mil_training_module,
        "_named_nonfinite_gradients",
        unexpected_failure_diagnostics,
    )
    result = weighted_amp_step(
        model,
        optimizer,
        scaler,
        lambda: cast("Tensor", model(torch.ones(1, 1))).square().sum(),
        class_weight=2.0,
        learning_rate=0.1,
    )
    assert model.weight.item() == pytest.approx(1.2)
    assert result.unweighted_loss == 4.0
    assert result.weighted_loss == 8.0
    assert result.attempts == 1
    assert result.scale_backoffs == 0
    assert scaler.step_calls == scaler.update_calls == 1
    assert scaler.scaled_values == [8.0]


def test_raw_overflow_retries_same_closure_without_advancing_progress() -> None:
    """An overflow must record a skipped scaler step then retry the same loaded WSI."""
    model = nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(2.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = _LazyOptimizerStateGradScaler(init_scale=16.0, growth_interval=2_000)
    attempts = 0
    loaded_wsi = object()
    observed_wsi: list[object] = []
    progress = {"cursor": 7, "committed_update": 19, "patience": 3}

    def overflow_once(gradient: Tensor) -> Tensor:
        return torch.full_like(gradient, float("inf")) if attempts == 1 else gradient

    model.weight.register_hook(  # pyright: ignore[reportUnknownMemberType]
        overflow_once,
    )

    def loss() -> Tensor:
        nonlocal attempts
        attempts += 1
        observed_wsi.append(loaded_wsi)
        return cast("Tensor", model(torch.ones(1, 1))).square().sum()

    result = weighted_amp_step(
        model,
        optimizer,
        scaler,
        loss,
        class_weight=1.0,
        learning_rate=0.1,
        max_scale_backoffs=MAX_SCALE_BACKOFFS,
    )
    assert observed_wsi == [loaded_wsi, loaded_wsi]
    assert progress == {"cursor": 7, "committed_update": 19, "patience": 3}
    assert result.attempts == 2
    assert result.scale_backoffs == 1
    assert result.initial_scale == 16.0
    assert result.final_scale == 8.0
    assert scaler.step_calls == scaler.update_calls == 2
    assert model.weight.item() == pytest.approx(1.6)
    assert "lazy_amp_bookkeeping" in optimizer.state[model.weight]


def test_raw_overflow_stops_after_exactly_three_skipped_updates() -> None:
    """The retry cap is a deliberate safety policy that permits only three backoffs."""
    model = nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(2.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = _TrackingGradScaler(init_scale=16.0, growth_interval=2_000)

    def always_overflow(gradient: Tensor) -> Tensor:
        return torch.full_like(gradient, float("-inf"))

    model.weight.register_hook(  # pyright: ignore[reportUnknownMemberType]
        always_overflow,
    )

    with pytest.raises(BranchNumericalError) as caught:
        weighted_amp_step(
            model,
            optimizer,
            scaler,
            lambda: cast("Tensor", model(torch.ones(1, 1))).square().sum(),
            class_weight=1.0,
            learning_rate=0.1,
        )
    assert caught.value.details["scale_backoffs"] == 3
    assert scaler.step_calls == scaler.update_calls == 3
    assert model.weight.item() == 2.0
    assert not optimizer.state


def test_calibration_restores_every_case_and_retains_only_scaler_state() -> None:
    """Stress WSIs may tune scaler history but must not leak model, optimizer, grad, or RNG state."""
    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)  # pyright: ignore[reportUnknownMemberType]
    model = nn.Linear(2, 1)
    initial_model = cast("dict[str, Tensor]", copy.deepcopy(model.state_dict()))
    scaler = _TrackingGradScaler(init_scale=8.0, growth_interval=2)
    optimizers: list[torch.optim.Optimizer] = []
    observations: list[tuple[int, float, float, float]] = []

    def optimizer_factory() -> torch.optim.Optimizer:
        current = cast("dict[str, Tensor]", model.state_dict())
        assert all(
            torch.equal(value, initial_model[name]) for name, value in current.items()
        )
        assert all(parameter.grad is None for parameter in model.parameters())
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        optimizers.append(optimizer)
        return optimizer

    def loss_for_wsi(wsi_id: int) -> Tensor:
        observations.append(
            (wsi_id, random.random(), float(np.random.random()), torch.rand(()).item()),
        )
        return cast("Tensor", model(torch.ones(1, 2))).sum()

    result = calibrate_grad_scaler(
        model,
        scaler,
        optimizer_factory,
        loss_for_wsi,
        class_weights_by_wsi=dict.fromkeys(CALIBRATION_WSI_IDS, 1.0),
    )
    assert result.scaler is scaler
    assert tuple(case.wsi_id for case in result.cases) == CALIBRATION_WSI_IDS
    assert len({id(optimizer) for optimizer in optimizers}) == len(CALIBRATION_WSI_IDS)
    assert len({row[1:] for row in observations}) == 1
    assert [case.scaler_state["scale"] for case in result.cases] == [
        8.0,
        16.0,
        16.0,
        32.0,
        32.0,
    ]
    assert random.random() == observations[0][1]
    assert float(np.random.random()) == observations[0][2]
    assert torch.rand(()).item() == observations[0][3]
    restored = cast("dict[str, Tensor]", model.state_dict())
    assert all(
        torch.equal(value, initial_model[name]) for name, value in restored.items()
    )
    assert all(parameter.grad is None for parameter in model.parameters())


def test_rng_capture_restore_round_trips_all_cpu_streams() -> None:
    """Resume and calibration depend on replaying Python, NumPy, and Torch draws exactly."""
    state = capture_rng_state()
    expected = (random.random(), float(np.random.random()), torch.rand(3))
    restore_rng_state(state)
    actual = (random.random(), float(np.random.random()), torch.rand(3))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    torch.testing.assert_close(actual[2], expected[2], rtol=0.0, atol=0.0)


def test_atomic_checkpoint_slots_round_trip_and_replace_latest(tmp_path: Path) -> None:
    """Latest, best, and final pointers must expose only complete hash-manifested objects."""
    first = _checkpoint_payload()
    for slot in ("latest", "best", "final"):
        saved = save_branch_checkpoint(tmp_path, slot=slot, payload=first)
        loaded = load_branch_checkpoint(
            tmp_path,
            slot=slot,
            expected_branch_name="normal_vae",
            expected_contract_hashes=_contract_hashes(),
        )
        assert loaded.checkpoint_sha256 == saved.checkpoint_sha256
        assert loaded.payload["committed_update"] == 53
        assert (tmp_path / slot).is_file()

    second = _checkpoint_payload(committed_update=106)
    replaced = save_branch_checkpoint(tmp_path, slot="latest", payload=second)
    assert replaced.payload["committed_update"] == 106
    assert not list(tmp_path.glob(".*.tmp"))
    assert not list((tmp_path / ".objects").glob(".checkpoint-*"))


def test_checkpoint_rejects_corruption_foreign_branch_hash_and_order(  # noqa: PLR0914, PLR0915
    tmp_path: Path,
) -> None:
    """Resume must fail closed for tampering, cross-branch state, stale contracts, or order drift."""
    payload = _checkpoint_payload()
    save_branch_checkpoint(tmp_path, slot="latest", payload=payload)
    with pytest.raises(ValueError, match="Cross-branch"):
        load_branch_checkpoint(
            tmp_path,
            slot="latest",
            expected_branch_name="so2_vae",
        )
    with pytest.raises(ValueError, match="contract hashes differ"):
        load_branch_checkpoint(
            tmp_path,
            slot="latest",
            expected_contract_hashes={**_contract_hashes(), "runtime": "stale"},
        )

    changed_order = dict(payload)
    changed_order["current_order"] = tuple(reversed(training_epoch_order(0)))
    with pytest.raises(ValueError, match="order identity differs"):
        validate_branch_checkpoint_payload(changed_order)

    wrong_progress = dict(payload)
    wrong_progress["within_epoch_cursor"] = 0
    with pytest.raises(ValueError, match="epoch/cursor"):
        validate_branch_checkpoint_payload(wrong_progress)

    wrong_epoch_order = dict(payload)
    wrong_epoch_order["current_order"] = training_epoch_order(1)
    wrong_epoch_order["order_sha256"] = hashlib.sha256(
        (
            json.dumps(
                {"order": list(training_epoch_order(1))},
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode(),
    ).hexdigest()
    with pytest.raises(ValueError, match="saved epoch"):
        validate_branch_checkpoint_payload(wrong_epoch_order)

    missing_train_row = dict(payload)
    missing_train_row["train_history"] = cast(
        "tuple[object, ...]",
        payload["train_history"],
    )[1:]
    with pytest.raises(ValueError, match="training history"):
        validate_branch_checkpoint_payload(missing_train_row)

    missing_validation = dict(payload)
    missing_validation["validation_history"] = ()
    with pytest.raises(ValueError, match="validation history"):
        validate_branch_checkpoint_payload(missing_validation)

    wrong_selector_row = copy.deepcopy(payload)
    validation_rows = list(
        cast(
            "tuple[dict[str, object], ...]",
            wrong_selector_row["validation_history"],
        ),
    )
    validation_rows[0]["improved"] = False
    wrong_selector_row["validation_history"] = tuple(validation_rows)
    with pytest.raises(ValueError, match="selector history"):
        validate_branch_checkpoint_payload(wrong_selector_row)

    coercible_selector = copy.deepcopy(payload)
    selector = cast("dict[str, object]", coercible_selector["best_selection"])
    selector["stopped"] = "false"
    with pytest.raises(TypeError, match="stopped flag"):
        validate_branch_checkpoint_payload(coercible_selector)

    coercible_metric = copy.deepcopy(payload)
    selector = cast("dict[str, object]", coercible_metric["best_selection"])
    best_metrics = cast("dict[str, object]", selector["best_metrics"])
    best_metrics["macro_f1"] = "0.4"
    with pytest.raises(TypeError, match="macro_f1"):
        validate_branch_checkpoint_payload(coercible_metric)

    coercible_rng = copy.deepcopy(payload)
    rng = cast("dict[str, object]", coercible_rng["rng_state"])
    numpy_rng = cast("dict[str, object]", rng["numpy"])
    numpy_rng["position"] = "7"
    with pytest.raises(ValueError, match="NumPy RNG values"):
        validate_branch_checkpoint_payload(coercible_rng)

    pointer = cast(
        "dict[str, object]",
        json.loads((tmp_path / "latest").read_text(encoding="utf-8")),
    )
    manifest = tmp_path / cast("str", pointer["manifest"])
    checkpoint = manifest.parent / "checkpoint.pt"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="file hash differs"):
        load_branch_checkpoint(tmp_path, slot="latest")


def test_checkpoint_rejects_traversal_and_untrusted_pickle(tmp_path: Path) -> None:
    """Resume paths stay confined and weights-only loading never executes pickle."""
    traversal_root = tmp_path / "traversal"
    save_branch_checkpoint(
        traversal_root,
        slot="latest",
        payload=_checkpoint_payload(),
    )
    pointer_path = traversal_root / "latest"
    pointer = cast(
        "dict[str, object]",
        json.loads(pointer_path.read_text(encoding="utf-8")),
    )
    pointer["manifest"] = "../foreign/manifest.json"
    pointer_path.write_text(json.dumps(pointer), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes"):
        load_branch_checkpoint(traversal_root, slot="latest")

    pickle_root = tmp_path / "pickle"
    save_branch_checkpoint(
        pickle_root,
        slot="latest",
        payload=_checkpoint_payload(),
    )
    pointer_path = pickle_root / "latest"
    pointer = cast(
        "dict[str, object]",
        json.loads(pointer_path.read_text(encoding="utf-8")),
    )
    manifest_path = pickle_root / cast("str", pointer["manifest"])
    manifest = cast(
        "dict[str, Any]",
        json.loads(manifest_path.read_text(encoding="utf-8")),
    )
    checkpoint_path = manifest_path.parent / "checkpoint.pt"
    marker = tmp_path / "pickle_executed"
    torch.save(_UntrustedCheckpoint(marker), checkpoint_path)
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    manifest["files"]["checkpoint.pt"] = checkpoint_sha256
    manifest_path.write_text(
        json.dumps(manifest, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pointer["manifest_sha256"] = hashlib.sha256(
        manifest_path.read_bytes(),
    ).hexdigest()
    pointer_path.write_text(json.dumps(pointer), encoding="utf-8")
    with pytest.raises(Exception, match="Weights only load failed"):
        load_branch_checkpoint(pickle_root, slot="latest")
    assert not marker.exists()


def test_bulk_reader_coalesces_contiguous_runs_and_restores_logical_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bulk mmap reads must preserve arbitrary duplicates/gaps while reducing conversion calls."""
    row_count = 6
    buffers = {
        part: bytearray(LATENT_SHARD_HEADER_SIZE)
        + bytearray(
            np.stack(
                [
                    np.full(LATENT_VALUES, part * 100 + index, dtype="<f4")
                    for index in range(row_count)
                ],
            ).tobytes(),
        )
        for part in (1, 2)
    }
    store = object.__new__(SupervisedLatentStore)
    parts = {
        part: CatalogPart(
            part=part,
            model_name="normal_vae",
            kaggle_source=f"owner/source-{part}/1",
            binary_name=f"part-{part}.bin",
            row_count=row_count,
            binary_bytes=LATENT_SHARD_HEADER_SIZE + row_count * LATENT_RECORD_BYTES,
            binary_sha256="0" * 64,
            sidecar_name=f"part-{part}.json",
            sidecar_bytes=1,
            sidecar_sha256="0" * 64,
        )
        for part in buffers
    }
    setattr(store, "_parts", parts)
    setattr(store, "_handles", {})
    setattr(store, "_source_roots", {})

    def mapping(part: int) -> bytearray:
        return buffers[part]

    setattr(store, "_mapping", mapping)
    pointers = (
        LogicalPointer(part=2, file_index=3),
        LogicalPointer(part=1, file_index=1),
        LogicalPointer(part=1, file_index=2),
        LogicalPointer(part=1, file_index=2),
        LogicalPointer(part=2, file_index=0),
        LogicalPointer(part=2, file_index=1),
        LogicalPointer(part=1, file_index=5),
    )
    original_frombuffer = torch.frombuffer
    calls: list[tuple[int, int]] = []

    def counting_frombuffer(
        buffer: bytearray,
        *,
        dtype: torch.dtype,
        count: int,
        offset: int,
    ) -> Tensor:
        calls.append((count, offset))
        return original_frombuffer(buffer, dtype=dtype, count=count, offset=offset)

    monkeypatch.setattr(torch, "frombuffer", counting_frombuffer)
    latents, reads = store.read_rows(pointers)
    expected_values = tuple(
        pointer.part * 100 + pointer.file_index for pointer in pointers
    )
    assert (
        tuple(float(latents[index, 0, 0, 0]) for index in range(len(pointers)))
        == expected_values
    )
    assert reads == tuple(
        sorted(
            (
                PhysicalRead(index, pointer.part, pointer.file_index)
                for index, pointer in enumerate(pointers)
            ),
            key=lambda read: (read.part, read.file_index),
        ),
    )
    assert len(calls) == 5
    assert any(count == 2 * LATENT_VALUES for count, _ in calls)
    assert len(calls) < len(pointers)
