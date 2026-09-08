# Copyright 2026 HiperMaximus
# pyright: reportAny=false, reportIndexIssue=false, reportPrivateUsage=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false, reportUnknownMemberType=false, reportUnknownVariableType=false
# ruff: noqa: PLR2004, SLF001
"""Focused contracts for the disposable Spec 0035 weighted-AMP probe."""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from collections import Counter
from copy import deepcopy
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
import torch
from scripts import build_largest_class_weighted_amp_probe as builder

if TYPE_CHECKING:
    from pathlib import Path
    from types import ModuleType


def _module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("spec0035_runtime", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_builder_selects_exact_largest_training_wsi_per_class() -> None:
    """The stress set must be the unique largest train-only bag in each class."""
    assets, provenance, selected = builder._derive_assets(builder.ROOT)
    assert provenance == {
        "development_contract": builder.DEV_CONTRACT_SHA256,
        "train_bags": builder.TRAIN_BAGS_SHA256,
        "train_instances": builder.TRAIN_INSTANCES_SHA256,
        "physical_catalog": builder.CATALOG_SHA256,
    }
    assert sum(cast("int", row["patch_count"]) for row in selected) == 127_847
    assert {
        row["diagnosis_label"]: (
            row["diagnosis_index"],
            row["class_count"],
            row["class_weight"],
            row["wsi_id"],
            row["patch_count"],
        )
        for row in selected
    } == {
        "CC": (0, 23, 106 / 115, 51_346, 29_150),
        "EC": (1, 24, 106 / 120, 45_630, 32_595),
        "HGSC": (2, 40, 106 / 200, 35_239, 18_981),
        "LGSC": (3, 11, 106 / 55, 57_162, 24_031),
        "MC": (4, 8, 106 / 40, 65_094, 23_090),
    }
    assert set(assets) == {
        "probe/bags.csv",
        "probe/pointers.csv",
        "probe/physical_parts.csv",
    }


def test_builder_emits_hash_bound_account_portable_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """New resources follow the actor while upstream owners remain unchanged."""
    for actor in ("maximshtefan", "professor-account"):
        output = tmp_path / actor
        monkeypatch.setattr(builder, "DEFAULT_ROOT", output)
        contract = builder.build(actor=actor)
        assert builder.validate(expected_actor=actor) == contract
        dataset_reference = f"{actor}/{builder.DATASET_SLUG}"
        assert contract["dataset_reference"] == dataset_reference
        assert contract["dataset_actor"] == actor
        assert contract["kernel_sources"] == list(builder.KERNEL_SOURCES)
        assert len(cast("list[object]", contract["selected"])) == 5

        metadata = json.loads(
            (output / "kernel/kernel-metadata.json").read_text(encoding="utf-8"),
        )
        assert metadata["dataset_sources"] == [dataset_reference]
        assert metadata["kernel_sources"] == list(builder.KERNEL_SOURCES)
        runtime = _module(output / "kernel/run.py")
        assert (
            builder._sha256(
                output / "bundle" / builder.CONTRACT_NAME,
            )
            == runtime.INPUT_CONTRACT_SHA256
        )
        assert dataset_reference == runtime.INPUT_DATASET_REFERENCE
        assert runtime.KERNEL_SOURCES == builder.KERNEL_SOURCES


def test_post_unscale_class_weighting_is_exact_for_batch_one() -> None:
    """Post-unscale FP32 weighting must preserve the intended CE gradient."""
    torch.manual_seed(35)
    direct = torch.nn.Linear(7, 5)
    deferred = torch.nn.Linear(7, 5)
    deferred.load_state_dict(direct.state_dict())
    inputs = torch.randn(1, 7)
    target = torch.tensor([4])
    weight = 2.65

    (torch.nn.functional.cross_entropy(direct(inputs), target) * weight).backward()
    torch.nn.functional.cross_entropy(deferred(inputs), target).backward()
    for parameter in deferred.parameters():
        assert parameter.grad is not None
        parameter.grad.mul_(weight)

    for direct_parameter, deferred_parameter in zip(
        direct.parameters(),
        deferred.parameters(),
        strict=True,
    ):
        assert direct_parameter.grad is not None
        assert deferred_parameter.grad is not None
        torch.testing.assert_close(
            deferred_parameter.grad,
            direct_parameter.grad,
            rtol=1e-6,
            atol=1e-7,
        )


def test_weighted_mean_cross_entropy_cancels_weight_for_one_sample() -> None:
    """The standard weighted-mean CE API must not silently erase batch-1 weights."""
    logits = torch.tensor([[0.4, -0.1, 1.2, 0.3, -0.7]])
    target = torch.tensor([4])
    weights = torch.tensor([0.9, 0.8, 0.5, 1.9, 2.65])
    ordinary = torch.nn.functional.cross_entropy(logits, target)
    weighted_mean = torch.nn.functional.cross_entropy(
        logits,
        target,
        weight=weights,
        reduction="mean",
    )
    torch.testing.assert_close(weighted_mean, ordinary)


def test_runtime_locks_standard_scaler_and_same_wsi_retry() -> None:
    """The rendered recipe must keep scaling, unscale and retry in safe order."""
    source = builder.TEMPLATE_PATH.read_text(encoding="utf-8")
    required = (
        'torch.amp.GradScaler("cuda")',
        "scaler.get_scale() != INITIAL_SCALE",
        "scaler.get_growth_factor() != GROWTH_FACTOR",
        "scaler.get_backoff_factor() != BACKOFF_FACTOR",
        "scaler.get_growth_interval() != GROWTH_INTERVAL",
        "scaler.scale(loss).backward()",
        "scaler.unscale_(optimizer)",
        'parameter.grad.mul_(case["class_weight"])',
        "scaler.step(optimizer)",
        "scaler.update()",
        "optimizer.zero_grad(set_to_none=True)",
        "retries >= MAX_OVERFLOW_BACKOFFS",
        "capture_rng_state(torch)",
        "restore_rng_state(torch, base_rng_state)",
        "calibrated_scaler_before_training",
        "training_optimizer = make_optimizer(torch, model)",
        "calibration_unique_graphs == unique_graphs",
        'result["minimum_successful_probe_scale"]',
    )
    assert all(fragment in source for fragment in required)
    assert source.count("scaler = make_scaler(torch)") == 1
    assert source.count("numerical = make_numerical(torch, model)") == 1
    assert "CrossEntropyLoss" not in source
    assert "scheduler" not in source
    assert "test.csv" not in source


def test_runtime_retries_overflow_without_committing_first_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One true overflow must back off and retry the identical update once."""
    runtime = _module(builder.TEMPLATE_PATH)
    model = torch.nn.Linear(3, 5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cpu", init_scale=16, growth_interval=1000)
    calls = {"count": 0}

    def overflow_once(gradient: torch.Tensor) -> torch.Tensor:
        calls["count"] += 1
        if calls["count"] == 1:
            return torch.full_like(gradient, float("inf"))
        return gradient

    model.weight.register_hook(overflow_once)
    inputs = torch.randn(1, 3)
    target = torch.tensor([4])

    def numerical(
        _bag: object,
        _graph: object,
        _target: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = model(inputs)
        return torch.nn.functional.cross_entropy(logits, target), logits.squeeze(0)

    case = {
        "bag": None,
        "graph": None,
        "target": target,
        "class_weight": 2.65,
        "device": "cpu",
        "wsi_id": 65_094,
        "diagnosis_label": "MC",
    }
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    attempts = runtime.run_weighted_update(
        torch,
        model,
        optimizer,
        scaler,
        numerical,
        case,
        "normal_vae",
    )
    assert [attempt["committed"] for attempt in attempts] == [False, True]
    assert [attempt["scale_before"] for attempt in attempts] == [16.0, 8.0]
    assert attempts[0]["retryable_overflow"] is True
    assert attempts[1]["post_step_state"]["all_finite"] is True
    assert runtime.optimizer_step_value(optimizer) == 1


def test_runtime_stops_after_three_overflow_backoffs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bounded retry policy must fail after exactly three skipped updates."""
    runtime = _module(builder.TEMPLATE_PATH)
    model = torch.nn.Linear(3, 5)
    initial = deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cpu", init_scale=16, growth_interval=1000)
    model.weight.register_hook(
        lambda gradient: torch.full_like(gradient, float("inf")),
    )
    inputs = torch.randn(1, 3)
    target = torch.tensor([4])

    def numerical(
        _bag: object,
        _graph: object,
        _target: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = model(inputs)
        return torch.nn.functional.cross_entropy(logits, target), logits.squeeze(0)

    case = {
        "bag": None,
        "graph": None,
        "target": target,
        "class_weight": 2.65,
        "device": "cpu",
        "wsi_id": 65_094,
        "diagnosis_label": "MC",
    }
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    with pytest.raises(runtime.WeightedUpdateError) as caught:
        runtime.run_weighted_update(
            torch,
            model,
            optimizer,
            scaler,
            numerical,
            case,
            "normal_vae",
        )
    assert len(caught.value.attempts) == 3
    assert [row["scale_before"] for row in caught.value.attempts] == [16.0, 8.0, 4.0]
    assert all(not row["committed"] for row in caught.value.attempts)
    assert not optimizer.state
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, initial[name])


def test_runtime_treats_missing_gradient_as_terminal() -> None:
    """A disconnected parameter must fail, not masquerade as overflow."""
    runtime = _module(builder.TEMPLATE_PATH)
    model = torch.nn.ParameterDict({
        "used": torch.nn.Parameter(torch.ones(2)),
        "unused": torch.nn.Parameter(torch.ones(2)),
    })
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cpu", init_scale=16, growth_interval=1000)

    def numerical(
        _bag: object,
        _graph: object,
        _target: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return model["used"].sum(), torch.zeros(5)

    case = {
        "bag": None,
        "graph": None,
        "target": None,
        "class_weight": 1.0,
        "device": "cpu",
        "wsi_id": 1,
        "diagnosis_label": "CC",
    }
    with pytest.raises(runtime.WeightedUpdateError) as caught:
        runtime.run_weighted_update(
            torch,
            model,
            optimizer,
            scaler,
            numerical,
            case,
            "normal_vae",
        )
    assert len(caught.value.attempts) == 1
    assert caught.value.attempts[0]["terminal_error"] == (
        "missing_or_wrong_dtype_gradient"
    )
    assert optimizer.state == {}


def test_runtime_rejects_zero_data_gradient_before_weight_decay_step() -> None:
    """AdamW decay must not disguise complete loss-gradient underflow as learning."""
    runtime = _module(builder.TEMPLATE_PATH)
    model = torch.nn.ParameterDict({"used": torch.nn.Parameter(torch.ones(2))})
    initial = deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    scaler = torch.amp.GradScaler("cpu", init_scale=16, growth_interval=1000)

    def numerical(
        _bag: object,
        _graph: object,
        _target: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return model["used"].sum() * 0 + 1, torch.zeros(5)

    case = {
        "bag": None,
        "graph": None,
        "target": None,
        "class_weight": 2.65,
        "device": "cpu",
        "wsi_id": 65_094,
        "diagnosis_label": "MC",
    }
    with pytest.raises(runtime.WeightedUpdateError) as caught:
        runtime.run_weighted_update(
            torch,
            model,
            optimizer,
            scaler,
            numerical,
            case,
            "normal_vae",
        )
    assert caught.value.attempts[-1]["terminal_error"] == (
        "zero_or_inexact_weighted_gradient"
    )
    assert optimizer.state == {}
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, initial[name])


def _acceptance_fixture(
    runtime: ModuleType,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    identities = {
        45_630: ("EC", 1, 32_595, 106 / 120),
        51_346: ("CC", 0, 29_150, 106 / 115),
        57_162: ("LGSC", 3, 24_031, 106 / 55),
        65_094: ("MC", 4, 23_090, 106 / 40),
        35_239: ("HGSC", 2, 18_981, 106 / 200),
    }
    scaler_hashes = [f"scaler-{index}" for index in range(11)]

    def row(
        wsi_id: int,
        phase: str,
        ordinal: int,
        scaler_index: int,
    ) -> dict[str, object]:
        label, diagnosis_index, patch_count, weight = identities[wsi_id]
        evidence: dict[str, object] = {
            "phase": phase,
            "scaler_object_id": 35,
            "scaler_before": {"sha256": scaler_hashes[scaler_index]},
            "scaler_after": {"sha256": scaler_hashes[scaler_index + 1]},
        }
        if phase == "calibration":
            evidence.update({
                "disposable_optimizer_ordinal": ordinal,
                "model_matches_base_before": True,
                "rng_matches_base_before": True,
                "optimizer_state_entries_before": 0,
                "optimizer_step_before": 0,
                "optimizer_step_after": 1,
            })
        else:
            evidence.update({
                "training_step_ordinal": ordinal,
                "optimizer_step_before": ordinal - 1,
                "optimizer_step_after": ordinal,
            })
        return {
            "wsi_id": wsi_id,
            "diagnosis_label": label,
            "diagnosis_index": diagnosis_index,
            "patch_count": patch_count,
            "class_weight": weight,
            "graph_sha256": f"graph-{wsi_id}",
            "edges": patch_count * 10,
            "phase_evidence": evidence,
            "attempts": [
                {
                    "committed": True,
                    "raw_gradients": {
                        "all_finite": True,
                        "nonzero_element_count": 1,
                        "l2_norm": 2.0,
                    },
                    "weighted_gradients": {
                        "all_finite": True,
                        "nonzero_element_count": 1,
                        "l2_norm": 2.0 * weight,
                    },
                    "post_step_state": {"all_finite": True},
                },
            ],
        }

    calibration = [
        row(wsi_id, "calibration", ordinal, ordinal - 1)
        for ordinal, wsi_id in enumerate(runtime.CALIBRATION_ORDER, start=1)
    ]
    training_simulation = [
        row(wsi_id, "training_simulation", ordinal, ordinal + 4)
        for ordinal, wsi_id in enumerate(runtime.TRAINING_SIMULATION_ORDER, start=1)
    ]
    boundary: dict[str, object] = {
        "initial_scaler": {"sha256": scaler_hashes[0]},
        "base_model_sha256": "model-base",
        "model_sha256_after_restore": "model-base",
        "model_matches_base_after_calibration": True,
        "model_gradients_cleared_after_calibration": True,
        "base_rng_sha256": "rng-base",
        "rng_sha256_after_restore": "rng-base",
        "rng_matches_base_after_calibration": True,
        "training_optimizer_state_entries_before": 0,
        "training_optimizer_step_before": 0,
        "calibrated_scaler_before_training": {"sha256": scaler_hashes[5]},
        "scaler_after_training_optimizer_creation": {"sha256": scaler_hashes[5]},
        "scaler_state_preserved_across_boundary": True,
        "scaler_object_id": 35,
        "single_scaler_reused": True,
        "final_scaler": {"sha256": scaler_hashes[10]},
    }
    return calibration, training_simulation, boundary


def _branch_accepts(
    runtime: ModuleType,
    calibration: list[dict[str, object]],
    training_simulation: list[dict[str, object]],
    boundary: dict[str, object] | None,
    *,
    graph_counts: tuple[int, int | None] = (2, 2),
) -> bool:
    unique_graphs, calibration_unique_graphs = graph_counts
    return bool(
        runtime.branch_accepts(
            calibration,
            training_simulation,
            None,
            5,
            runtime.INITIAL_SCALE,
            unique_graphs,
            0,
            1,
            runtime.MIN_RESERVED_HEADROOM,
            boundary,
            calibration_unique_graphs,
        ),
    )


def test_acceptance_requires_strict_phase_order_and_compiled_reuse() -> None:
    """The gate must reject legacy phases, reordered cases and boundary recompiles."""
    runtime = _module(builder.TEMPLATE_PATH)
    calibration, training_simulation, boundary = _acceptance_fixture(runtime)
    assert _branch_accepts(runtime, calibration, training_simulation, boundary)

    reordered = deepcopy(calibration)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    assert not _branch_accepts(runtime, reordered, training_simulation, boundary)
    assert not _branch_accepts(runtime, calibration, training_simulation, None)
    assert not _branch_accepts(
        runtime,
        calibration,
        training_simulation,
        boundary,
        graph_counts=(3, 2),
    )
    assert not runtime.paired_transcripts_match([
        {
            "isolated": deepcopy(calibration),
            "persistent": deepcopy(training_simulation),
        },
        {
            "isolated": deepcopy(calibration),
            "persistent": deepcopy(training_simulation),
        },
    ])
    branches = [
        {
            "calibration": deepcopy(calibration),
            "training_simulation": deepcopy(training_simulation),
        },
        {
            "calibration": deepcopy(calibration),
            "training_simulation": deepcopy(training_simulation),
        },
    ]
    assert runtime.paired_transcripts_match(branches)
    branches[1]["training_simulation"][0]["graph_sha256"] = "different"
    assert not runtime.paired_transcripts_match(branches)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("model_matches_base_after_calibration", False),
        ("model_sha256_after_restore", "wrong-model"),
        ("model_gradients_cleared_after_calibration", False),
        ("rng_matches_base_after_calibration", False),
        ("rng_sha256_after_restore", "wrong-rng"),
        ("training_optimizer_state_entries_before", 1),
        ("training_optimizer_step_before", 1),
        ("scaler_state_preserved_across_boundary", False),
        (
            "scaler_after_training_optimizer_creation",
            {"sha256": "fresh-scaler"},
        ),
        ("single_scaler_reused", False),
    ],
)
def test_acceptance_rejects_corrupt_calibration_boundary(
    field: str,
    bad_value: object,
) -> None:
    """Every isolation field is a fail-closed guard against warm-up state leakage."""
    runtime = _module(builder.TEMPLATE_PATH)
    calibration, training_simulation, boundary = _acceptance_fixture(runtime)
    boundary[field] = bad_value
    assert not _branch_accepts(runtime, calibration, training_simulation, boundary)


def test_acceptance_rejects_broken_scaler_state_chain() -> None:
    """A fresh same-scale scaler must not masquerade as the calibrated scaler."""
    runtime = _module(builder.TEMPLATE_PATH)
    calibration, training_simulation, boundary = _acceptance_fixture(runtime)
    evidence = cast(
        "dict[str, object]",
        training_simulation[0]["phase_evidence"],
    )
    evidence["scaler_before"] = {"sha256": "fresh-same-scale-scaler"}
    assert not _branch_accepts(runtime, calibration, training_simulation, boundary)


@pytest.mark.parametrize(
    ("phase", "index", "field", "bad_value"),
    [
        ("calibration", 0, "phase", "training_simulation"),
        ("calibration", 1, "disposable_optimizer_ordinal", 1),
        ("training_simulation", 0, "phase", "calibration"),
        ("training_simulation", 1, "training_step_ordinal", 1),
        ("training_simulation", 2, "optimizer_step_before", 0),
        ("training_simulation", 3, "optimizer_step_after", 3),
    ],
)
def test_acceptance_rejects_phase_and_optimizer_chain_mutations(
    phase: str,
    index: int,
    field: str,
    bad_value: object,
) -> None:
    """Phase labels, ordinals and derived optimizer steps must fail closed."""
    runtime = _module(builder.TEMPLATE_PATH)
    calibration, training_simulation, boundary = _acceptance_fixture(runtime)
    rows = calibration if phase == "calibration" else training_simulation
    evidence = cast("dict[str, object]", rows[index]["phase_evidence"])
    evidence[field] = bad_value
    assert not _branch_accepts(runtime, calibration, training_simulation, boundary)


def test_model_and_rng_restore_form_a_pristine_training_boundary() -> None:
    """Disposable work must leave model gradients and every used RNG unchanged."""
    runtime = _module(builder.TEMPLATE_PATH)
    model = torch.nn.Linear(3, 2)
    base_state = deepcopy(model.state_dict())
    base_model_sha256 = runtime.model_state_sha256(model)
    original_rng = runtime.capture_rng_state(torch)
    try:
        base_rng = runtime.capture_rng_state(torch)
        with torch.no_grad():
            model.weight.add_(1)
        model.weight.grad = torch.ones_like(model.weight)
        random.random()  # noqa: S311 - deliberately advances captured training state
        np.random.random()  # noqa: NPY002 - runtime intentionally captures legacy state
        torch.rand(4)

        runtime.restore_model_state(torch, model, base_state, base_model_sha256)
        runtime.restore_rng_state(torch, base_rng)
        restored_rng = runtime.capture_rng_state(torch)
        assert runtime.model_matches_state(torch, model, base_state)
        assert runtime.model_state_sha256(model) == base_model_sha256
        assert all(parameter.grad is None for parameter in model.parameters())
        assert runtime.rng_state_matches(torch, restored_rng, base_rng)
        assert runtime.rng_state_sha256(restored_rng) == runtime.rng_state_sha256(
            base_rng,
        )
    finally:
        runtime.restore_rng_state(torch, original_rng)


def test_one_scaler_crosses_fresh_optimizer_calibration_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scaler backoff and growth history must survive discarded optimizer state."""
    runtime = _module(builder.TEMPLATE_PATH)
    model = torch.nn.Linear(3, 5)
    base_state = deepcopy(model.state_dict())
    base_model_sha256 = runtime.model_state_sha256(model)
    scaler = torch.amp.GradScaler("cpu", init_scale=16, growth_interval=1000)
    scaler_object_id = id(scaler)
    optimizers: list[torch.optim.Optimizer] = []
    inputs = torch.randn(1, 3)
    target = torch.tensor([4])
    hook_calls = {"count": 0}

    def overflow_once(gradient: torch.Tensor) -> torch.Tensor:
        hook_calls["count"] += 1
        if hook_calls["count"] == 1:
            return torch.full_like(gradient, float("inf"))
        return gradient

    model.weight.register_hook(overflow_once)

    def numerical(
        _bag: object,
        _graph: object,
        _target: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = model(inputs)
        return torch.nn.functional.cross_entropy(logits, target), logits.squeeze(0)

    def update(optimizer: torch.optim.Optimizer, wsi_id: int) -> list[object]:
        case = {
            "bag": None,
            "graph": None,
            "target": target,
            "class_weight": 2.65,
            "device": "cpu",
            "wsi_id": wsi_id,
            "diagnosis_label": "MC",
        }
        return cast(
            "list[object]",
            runtime.run_weighted_update(
                torch,
                model,
                optimizer,
                scaler,
                numerical,
                case,
                "normal_vae",
            ),
        )

    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    for ordinal in range(5):
        runtime.restore_model_state(torch, model, base_state, base_model_sha256)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizers.append(optimizer)
        assert not optimizer.state
        assert runtime.optimizer_step_value(optimizer) == 0
        attempts = update(optimizer, ordinal)
        assert (
            sum(bool(cast("dict[str, object]", row)["committed"]) for row in attempts)
            == 1
        )
        assert runtime.optimizer_step_value(optimizer) == 1

    calibrated_state = deepcopy(scaler.state_dict())
    assert calibrated_state["scale"] == pytest.approx(8.0)
    assert calibrated_state["_growth_tracker"] == 5
    runtime.restore_model_state(torch, model, base_state, base_model_sha256)
    training_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    assert not training_optimizer.state
    assert runtime.optimizer_step_value(training_optimizer) == 0
    assert id(scaler) == scaler_object_id
    assert scaler.state_dict() == calibrated_state

    update(training_optimizer, 99)
    assert runtime.optimizer_step_value(training_optimizer) == 1
    assert scaler.state_dict()["scale"] == calibrated_state["scale"]
    assert scaler.state_dict()["_growth_tracker"] == 6
    assert len({id(optimizer) for optimizer in [*optimizers, training_optimizer]}) == 6


def test_selected_pointer_parts_and_counts_are_complete() -> None:
    """Every selected pointer must resolve through the exact required producers."""
    _, _, selected = builder._derive_assets(builder.ROOT)
    observed: Counter[int] = Counter()
    for row in selected:
        part_counts = cast("dict[str, int]", row["part_counts"])
        for part, count in part_counts.items():
            observed[int(part)] += int(count)
    assert set(observed) == builder.REQUIRED_PARTS
    assert sum(observed.values()) == 127_847
