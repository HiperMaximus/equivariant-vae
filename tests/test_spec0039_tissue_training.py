# Copyright 2026 HiperMaximus
# pyright: reportAny=false, reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownVariableType=false
# ruff: noqa: I001, PT018, SLF001
"""Focused contract tests for the Spec 0039 tissue training campaign."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from scripts import build_tissue_training as builder
from eqvae.data.supervised_latents import TISSUE_HEADER
from eqvae.training.supervised_pairing import full_training_learning_rate

if TYPE_CHECKING:
    from types import ModuleType


def _module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("spec0039_runtime", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_all_five_train_and_selection_validation_geometries_are_exact() -> None:
    """Validation never exceeds its corresponding balanced train budget."""
    assets, config, _logical = builder._derive_assets(builder.ROOT)
    assert builder.BUDGETS == {
        250: {"total_rows": 750, "batch_size": 125, "steps_per_epoch": 6},
        500: {"total_rows": 1_500, "batch_size": 125, "steps_per_epoch": 12},
        1_000: {"total_rows": 3_000, "batch_size": 125, "steps_per_epoch": 24},
        2_500: {"total_rows": 7_500, "batch_size": 125, "steps_per_epoch": 60},
        5_671: {"total_rows": 17_013, "batch_size": 159, "steps_per_epoch": 107},
    }
    assert builder.VALIDATION_PER_CLASS == {
        250: 250,
        500: 500,
        1_000: 1_000,
        2_500: 1_250,
        5_671: 1_250,
    }
    assert config["minimum_completed_epochs"] == builder.MINIMUM_COMPLETED_EPOCHS
    for budget, geometry in builder.BUDGETS.items():
        selected = config["budgets"][str(budget)]
        assert geometry["total_rows"] == (
            geometry["batch_size"] * geometry["steps_per_epoch"]
        )
        assert selected["validation_rows"] == 3 * selected["validation_per_class"]
        assert selected["validation_rows"] <= selected["total_rows"]
        train_rows = builder._read_csv_bytes(
            assets[f"tissue/tissue_train_{budget:04d}_per_class.csv"],
            TISSUE_HEADER,
        )
        validation_rows = builder._read_csv_bytes(
            assets[
                "tissue/tissue_validation_"
                f"{selected['validation_per_class']:04d}_per_class.csv"
            ],
            TISSUE_HEADER,
        )
        assert not {builder._identity(row) for row in train_rows} & {
            builder._identity(row) for row in validation_rows
        }
        assert not {row["wsi_id"] for row in train_rows} & {
            row["wsi_id"] for row in validation_rows
        }
    assert "tissue/tissue_validation.csv" not in assets
    assert not any("test" in name for name in assets)


def test_validation_subsets_are_nested_balanced_and_cover_each_positive_wsi() -> None:
    """Derived validation is WSI-stratified, nested, and never natural-size."""
    assets, _config, _logical = builder._derive_assets(builder.ROOT)
    prior = {label: set() for label in builder.LABELS}
    for per_class in (250, 500, 1_000, 1_250):
        rows = builder._read_csv_bytes(
            assets[f"tissue/tissue_validation_{per_class:04d}_per_class.csv"],
            TISSUE_HEADER,
        )
        assert len(rows) == 3 * per_class
        for label in builder.LABELS:
            selected = [row for row in rows if row["tissue_label"] == label]
            identities = {builder._identity(row) for row in selected}
            assert len(selected) == per_class
            assert prior[label] <= identities
            assert (
                len({row["wsi_id"] for row in selected})
                == {
                    "tumor": 23,
                    "stroma": 21,
                    "necrosis": 5,
                }[label]
            )
            prior[label] = identities


def test_schedule_reaches_the_locked_peak_and_floor() -> None:
    """All campaign geometries use the fixed 30-epoch warmup/cosine schedule."""
    _assets, config, _logical = builder._derive_assets(builder.ROOT)
    for geometry in config["budgets"].values():
        steps = geometry["steps_per_epoch"]
        warmup = geometry["warmup_updates"]
        total = geometry["total_updates"]
        assert full_training_learning_rate(
            peak=builder.PEAK_LR,
            successful_update=warmup,
            steps_per_epoch=steps,
        ) == pytest.approx(builder.PEAK_LR)
        assert full_training_learning_rate(
            peak=builder.PEAK_LR,
            successful_update=total,
            steps_per_epoch=steps,
        ) == pytest.approx(builder.PEAK_LR * 0.01)


def test_builder_is_actor_portable_and_excludes_full_validation_and_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mounted learning surface contains only selected validation files."""
    output = tmp_path / "package"
    monkeypatch.setattr(builder, "DEFAULT_ROOT", output)
    contract = builder.build(actor="professor-account")
    assert builder.validate(expected_actor="professor-account") == contract
    bundle = output / "bundle"
    non_source = {
        path.relative_to(bundle).as_posix()
        for path in bundle.rglob("*")
        if path.is_file() and not path.relative_to(bundle).as_posix().startswith("src/")
    }
    assert "tissue/tissue_validation.csv" not in non_source
    assert not any("test" in name for name in non_source)
    assert {name for name in non_source if "validation_" in name} == {
        "tissue/tissue_validation_0250_per_class.csv",
        "tissue/tissue_validation_0500_per_class.csv",
        "tissue/tissue_validation_1000_per_class.csv",
        "tissue/tissue_validation_1250_per_class.csv",
    }
    metadata = json.loads((output / "kernel/kernel-metadata.json").read_text())
    assert metadata["dataset_sources"] == [contract["dataset_reference"]]
    assert metadata["kernel_sources"] == list(builder.KERNEL_SOURCES)
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    runtime = _module(output / "kernel/run.py")
    assert runtime.SPEC0039_TISSUE_TRAINING_READY is True
    assert (
        builder._sha256(bundle / builder.CONTRACT_NAME) == runtime.INPUT_CONTRACT_SHA256
    )
    monkeypatch.setattr(
        builder,
        "_validate_source_snapshot",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("must use sealed snapshot"),
        ),
    )
    assert (
        builder.validate(
            expected_actor="professor-account",
            sealed_source_snapshot=True,
        )
        == contract
    )


def test_retry_builder_reuses_only_a_verified_frozen_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A launcher retry must preserve an input contract rather than rebuild data."""
    original = tmp_path / "original"
    monkeypatch.setattr(builder, "DEFAULT_ROOT", original)
    contract = builder.build(actor="professor-account")
    frozen_input = original / "bundle"
    (frozen_input / builder.METADATA_NAME).unlink()
    retry_root = tmp_path / "retry-v2"

    assert (
        builder.build_retry(
            actor="professor-account",
            input_bundle=frozen_input,
            output_root=retry_root,
        )
        == contract
    )
    assert (
        builder.validate_retry(
            output_root=retry_root,
            expected_actor="professor-account",
        )
        == contract
    )
    assert builder._sha256(
        frozen_input / builder.CONTRACT_NAME,
    ) == builder._sha256(
        retry_root / "bundle" / builder.CONTRACT_NAME,
    )
    runtime = _module(retry_root / "kernel/run.py")
    assert (
        builder._sha256(frozen_input / builder.CONTRACT_NAME)
        == runtime.INPUT_CONTRACT_SHA256
    )


def test_template_locks_selected_runtime_and_matrix_only_decay() -> None:
    """The actual launcher, not prose, contains selected runtime choices."""
    source = (builder.ROOT / builder.TEMPLATE_PATH).read_text(encoding="utf-8")
    assert 'mode="max-autotune"' in source
    assert "fullgraph=True" in source
    assert "dynamic=False" in source
    assert 'torch.amp.GradScaler("cuda")' in source
    assert "parameter.ndim >= 2" in source
    assert "matrix_weight_decay" in source
    assert "compiled_autograd = True" in source
    assert "ThreadPoolExecutor" not in source
    assert "subprocess.Popen(command, env=environment)" in source
    assert 'environment["CUDA_VISIBLE_DEVICES"] = str(device)' in source
    assert '"--worker"' in source
    assert "A Spec 0039 child must see exactly its assigned T4" in source
    assert "torch.cuda.device_count() != 1" in source
    assert "TORCHINDUCTOR_CACHE_DIR" in source
    assert "TRITON_CACHE_DIR" in source
    assert "outer_stop" not in source
    assert "DEFAULT_MINIMUM_COMPLETED_EPOCHS = 10" in source
    assert "return epoch >= minimum_completed_epochs(config) and step == end" in source
    assert 'if branches[reference_branch]["patience_exhausted"]:' in source
    assert ".pin_memory().to(device=device, non_blocking=True)" in source
    assert "access_transcript" in source
    assert "configuration_sha256" in source
    assert "def train_branch_step(" in source
    assert 'branch["scaler"].step(optimizer)' in source
    assert 'branch["scaler"].update()' in source
    assert "prepare_branch_step" not in source
    assert '"found_inf_per_device"' not in source
    assert "tissue/tissue_validation.csv" in source  # explicit mounted-input rejection


def test_output_command_is_bound_to_the_spec0039_kernel_and_receipted() -> None:
    """A different campaign's receipt cannot download as this campaign."""
    source = (builder.ROOT / "scripts/kaggle_kernel.sh").read_text(encoding="utf-8")
    start = source.index("output_tissue_training()")
    stop = source.index("\nbuild_mil_training()", start)
    output_command = source[start:stop]
    assert "eqvae-tissue-label-efficiency-training" in output_command
    assert "record_kaggle_download" in output_command


def test_retry_push_has_its_own_one_use_guard() -> None:
    """The repaired retry cannot consume or overwrite the v1 authority."""
    source = (builder.ROOT / "scripts/kaggle_kernel.sh").read_text(encoding="utf-8")
    assert "KAGGLE_TISSUE_TRAINING_RETRY_V2_CONFIRMED" in source
    assert "tissue_training_retry_launch_claim" in source
    assert '"spec0039.kernel_launch_retry_v2_claim.v1"' in source
    assert "build-tissue-training-retry-v2" in source
    assert "KAGGLE_TISSUE_TRAINING_RETRY_V3_CONFIRMED" in source
    assert '"spec0039.kernel_launch_retry_v3_claim.v1"' in source
    assert "build-tissue-training-retry-v3" in source
    assert "minimum_completed_epochs" in source


def test_runtime_epoch_batches_are_static_and_rotate_b125_remainders() -> None:
    """The actual runner consumes every training row with the agreed class mix."""
    runtime = _module(builder.ROOT / builder.TEMPLATE_PATH)
    for per_class, batch_size in ((250, 125), (500, 125), (5_671, 159)):
        instances = tuple(
            SimpleNamespace(tissue_label=label)
            for label in builder.LABELS
            for _ in range(per_class)
        )
        dataset = SimpleNamespace(instances=instances)
        batches = runtime.epoch_batches(
            dataset,
            batch_size=batch_size,
            epoch=1,
            seed=builder.INITIALIZATION_SEED,
        )
        assert len(batches) == 3 * per_class // batch_size
        assert all(len(batch) == batch_size for batch in batches)
        labels = [
            dataset.instances[index].tissue_label
            for batch in batches
            for index in batch
        ]
        assert Counter(labels) == Counter(dict.fromkeys(builder.LABELS, per_class))
        if batch_size == builder.BUDGETS[250]["batch_size"]:
            compositions = [
                Counter(dataset.instances[index].tissue_label for index in batch)
                for batch in batches[:3]
            ]
            assert compositions == [
                Counter({"tumor": 42, "stroma": 42, "necrosis": 41}),
                Counter({"tumor": 42, "stroma": 41, "necrosis": 42}),
                Counter({"tumor": 41, "stroma": 42, "necrosis": 42}),
            ]


def test_runtime_does_not_arm_patience_before_ten_full_epochs() -> None:
    """The epoch-10 midpoint cannot terminate a branch early."""
    runtime = _module(builder.ROOT / builder.TEMPLATE_PATH)
    legacy_config = {"maximum_epochs": 30}
    assert (
        runtime.minimum_completed_epochs(legacy_config)
        == runtime.DEFAULT_MINIMUM_COMPLETED_EPOCHS
    )
    assert not runtime.patience_is_armed(
        legacy_config,
        epoch=9,
        step=6,
        end=6,
    )
    assert not runtime.patience_is_armed(
        legacy_config,
        epoch=10,
        step=3,
        end=6,
    )
    assert runtime.patience_is_armed(
        legacy_config,
        epoch=10,
        step=6,
        end=6,
    )


def test_runtime_aggregate_allows_independent_terminal_boundaries(
    tmp_path: Path,
) -> None:
    """Different independent stopping times may share only an order prefix."""
    runtime = _module(builder.ROOT / builder.TEMPLATE_PATH)
    runtime.WORK_ROOT = tmp_path
    config = {"maximum_epochs": 30, "budgets": {"250": {}}}
    normal_updates, so2_updates = 60, 120
    shared = {
        "budget_per_class": 250,
        "train_rows": 750,
        "validation_rows": 750,
        "batch_size": 125,
        "steps_per_epoch": 6,
        "minimum_completed_epochs": 10,
        "initial_state_sha256": "a" * 64,
    }
    rows = {
        "normal_vae": {
            **shared,
            "scheduled_updates": normal_updates,
            "stopped_early": True,
            "order_hashes": [{"epoch": 1, "sha256": "b" * 64}],
            "branches": {
                "normal_vae": {
                    "best": {"macro_f1": 0.5},
                    "calibration": [],
                    "patience_exhausted": True,
                },
            },
        },
        "so2_vae": {
            **shared,
            "scheduled_updates": so2_updates,
            "stopped_early": False,
            "order_hashes": [
                {"epoch": 1, "sha256": "b" * 64},
                {"epoch": 2, "sha256": "c" * 64},
            ],
            "branches": {
                "so2_vae": {
                    "best": {"macro_f1": 0.6},
                    "calibration": [],
                    "patience_exhausted": False,
                },
            },
        },
    }
    for branch, row in rows.items():
        runtime.atomic_json(
            tmp_path / "branches" / branch / "branch_summary.json",
            {"branch": branch, "status": "complete", "budgets": [row]},
        )
    result = runtime._aggregate({"normal_vae": 0, "so2_vae": 0}, config)
    assert result["status"] == "complete"
    merged = result["budgets"][0]["branches"]
    assert merged["normal_vae"]["scheduled_updates"] == normal_updates
    assert merged["so2_vae"]["scheduled_updates"] == so2_updates


def test_runtime_launches_exactly_one_isolated_process_per_representation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The parent sends only the intended worker arguments to each child."""
    runtime = _module(builder.ROOT / builder.TEMPLATE_PATH)
    runtime.WORK_ROOT = tmp_path
    normal, so2 = Mock(), Mock()
    normal.wait.return_value = 0
    so2.wait.return_value = 0
    popen = Mock(side_effect=[normal, so2])
    monkeypatch.setattr(runtime.subprocess, "Popen", popen)
    assert runtime._launch_children() == {"normal_vae": 0, "so2_vae": 0}
    assert popen.call_count == len(runtime.BRANCHES)
    observed = {}
    for call in popen.call_args_list:
        command = call.args[0]
        branch = command[3]
        observed[branch] = {"command": command, "environment": call.kwargs["env"]}
    assert set(observed) == {"normal_vae", "so2_vae"}
    for branch, device in (("normal_vae", "0"), ("so2_vae", "1")):
        command = observed[branch]["command"]
        environment = observed[branch]["environment"]
        assert command == [
            runtime.sys.executable,
            str(Path(runtime.__file__).resolve()),
            "--worker",
            branch,
            "--branch-root",
            str(tmp_path / "branches" / branch),
        ]
        assert environment["CUDA_VISIBLE_DEVICES"] == device
        assert branch in environment["TORCHINDUCTOR_CACHE_DIR"]
        assert branch in environment["TRITON_CACHE_DIR"]
