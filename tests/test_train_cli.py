# Copyright 2026 HiperMaximus
"""Tests for the spec 0001 short train/debug proof CLI."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
import torch

from eqvae.checkpointing import load_training_checkpoint, save_training_checkpoint
from eqvae.cli.train import main as train_main

if TYPE_CHECKING:
    from collections.abc import Callable

SHORT_TRAIN_STEPS = 2
RESUME_TARGET_STEPS = 3
FIXED_TINY_PATCH_COUNT = 32


def test_train_cli_writes_selected_runtime_debug_artifacts(tmp_path: Path) -> None:
    """Selected-runtime debug consumes v5 but stays local/non-promotable."""
    config_path = _debug_config(tmp_path, selected_runtime_required=True)
    runtime_config = _runtime_config(tmp_path)
    output_dir = tmp_path / "debug"

    exit_code = train_main(
        [
            "--config",
            str(config_path),
            "--runtime-config",
            str(runtime_config),
            "--data",
            "synthetic",
            "--output-dir",
            str(output_dir),
            "--run-name",
            "spec0001_selected_runtime_debug_local",
            "--max-train-steps",
            str(SHORT_TRAIN_STEPS),
            "--max-val-steps",
            "1",
            "--save-every-steps",
            "1",
        ],
    )

    assert exit_code == 0
    summary = _load_json(output_dir / "benchmark" / "training_summary.json")
    debug_summary = _load_json(
        output_dir / "benchmark" / "selected_runtime_debug_summary.json",
    )
    manifest = _load_json(output_dir / "benchmark" / "artifact_manifest.json")
    metric_rows = _load_csv(output_dir / "metrics" / "train_metrics.csv")

    assert summary["status"] == "local_pass"
    assert summary["full_run_eligible"] is False
    assert summary["optimizer_steps_completed"] == SHORT_TRAIN_STEPS
    runtime = _object(summary, "runtime_config")
    assert runtime["consumed"] is True
    assert runtime["runtime_policy_id"] == "amp_fp16_conservative"
    assert debug_summary["real_kaggle_debug_status"] == (
        "pending_permission_gated_remote_run"
    )
    assert debug_summary["checkpoint_written"] is True
    assert manifest["reconstruction_sample_nonblank"] is True
    assert len(metric_rows) == SHORT_TRAIN_STEPS
    assert (output_dir / "checkpoints" / "step_000001.pt").exists()
    assert (output_dir / "checkpoints" / "step_000002.pt").exists()
    assert (output_dir / "checkpoints" / "final.pt").exists()
    assert (output_dir / "checkpoints" / "best_model.pt").exists()


def test_train_cli_resume_writes_resume_proof(tmp_path: Path) -> None:
    """Resume proof records restored state and continues step counters."""
    config_path = _debug_config(tmp_path, selected_runtime_required=False)
    first_output = tmp_path / "first"
    resumed_output = tmp_path / "resumed"

    assert (
        train_main(
            [
                "--config",
                str(config_path),
                "--data",
                "synthetic",
                "--output-dir",
                str(first_output),
                "--run-name",
                "spec0001_resume_source",
                "--max-train-steps",
                str(SHORT_TRAIN_STEPS),
                "--save-every-steps",
                "1",
            ],
        )
        == 0
    )
    assert (
        train_main(
            [
                "--config",
                str(config_path),
                "--data",
                "synthetic",
                "--resume",
                str(first_output / "checkpoints" / "step_000001.pt"),
                "--output-dir",
                str(resumed_output),
                "--run-name",
                "spec0001_resume_target",
                "--max-train-steps",
                str(RESUME_TARGET_STEPS),
                "--save-every-steps",
                "1",
            ],
        )
        == 0
    )

    proof = _load_json(resumed_output / "benchmark" / "checkpoint_resume_proof.json")
    summary = _load_json(resumed_output / "benchmark" / "training_summary.json")
    rows = _load_csv(resumed_output / "metrics" / "train_metrics.csv")
    assert proof["status"] == "local_pass"
    assert proof["loaded_successful_optimizer_update_count"] == 1
    assert proof["additional_optimizer_steps"] == SHORT_TRAIN_STEPS
    assert proof["model_state_restored"] is True
    assert proof["optimizer_state_restored"] is True
    assert proof["python_rng_state_restored"] is True
    assert proof["numpy_generator_state_restored"] is True
    assert proof["torch_cpu_rng_state_restored"] is True
    assert proof["torch_generator_states_restored"] is True
    assert proof["torch_generator_names_restored"] == ["train_data"]
    assert proof["torch_cuda_rng_state_status"] == "not_applicable_local_cpu"
    assert (
        proof["lr_scheduler_state_status"] == "not_applicable_local_debug_no_scheduler"
    )
    assert proof["amp_scaler_state_status"] == "not_applicable_local_cpu_amp_disabled"
    assert proof["schedule_resumed_from_successful_optimizer_update_count"] is True
    assert proof["runtime_config_sha256_match"] is True
    assert proof["selected_row_id_match"] is True
    assert proof["runtime_policy_id_match"] is True
    assert summary["optimizer_steps_completed"] == SHORT_TRAIN_STEPS
    assert [row["optimizer_step"] for row in rows] == ["2", "3"]
    checkpoint_payload = _load_checkpoint(
        first_output / "checkpoints" / "step_000001.pt",
    )
    assert checkpoint_payload["schema_version"] == "spec0001.checkpoint.v4"
    assert not checkpoint_payload["runtime_config_sha256"]
    assert not checkpoint_payload["selected_row_id"]
    assert not checkpoint_payload["runtime_policy_id"]
    torch_generator_states = _object(checkpoint_payload, "torch_generator_states")
    assert set(torch_generator_states) == {"train_data"}
    assert isinstance(torch_generator_states["train_data"], torch.Tensor)


def test_train_cli_resume_matches_uninterrupted_state(tmp_path: Path) -> None:
    """Resumed model and optimizer state match the uninterrupted reference."""
    config_path = _debug_config(tmp_path, selected_runtime_required=False)
    full_output = tmp_path / "full"
    resumed_output = tmp_path / "resumed"

    assert (
        train_main(
            [
                "--config",
                str(config_path),
                "--data",
                "synthetic",
                "--output-dir",
                str(full_output),
                "--run-name",
                "spec0001_resume_full_reference",
                "--max-train-steps",
                str(RESUME_TARGET_STEPS),
                "--save-every-steps",
                "1",
            ],
        )
        == 0
    )
    assert (
        train_main(
            [
                "--config",
                str(config_path),
                "--data",
                "synthetic",
                "--resume",
                str(full_output / "checkpoints" / "step_000001.pt"),
                "--output-dir",
                str(resumed_output),
                "--run-name",
                "spec0001_resume_replay",
                "--max-train-steps",
                str(RESUME_TARGET_STEPS),
                "--save-every-steps",
                "1",
            ],
        )
        == 0
    )

    full_payload = _load_checkpoint(full_output / "checkpoints" / "final.pt")
    resumed_payload = _load_checkpoint(resumed_output / "checkpoints" / "final.pt")
    _assert_state_equal(
        full_payload["model_state_dict"],
        resumed_payload["model_state_dict"],
    )
    _assert_state_equal(
        full_payload["optimizer_state_dict"],
        resumed_payload["optimizer_state_dict"],
    )


def test_train_cli_resume_rejects_mismatched_config(tmp_path: Path) -> None:
    """Resume fails closed when the effective config hash changes."""
    config_path = _debug_config(tmp_path, selected_runtime_required=False)
    changed_config_path = _debug_config(
        tmp_path,
        selected_runtime_required=False,
        max_val_steps=2,
    )
    first_output = tmp_path / "first"

    assert (
        train_main(
            [
                "--config",
                str(config_path),
                "--data",
                "synthetic",
                "--output-dir",
                str(first_output),
                "--run-name",
                "spec0001_resume_source",
                "--max-train-steps",
                str(SHORT_TRAIN_STEPS),
                "--save-every-steps",
                "1",
            ],
        )
        == 0
    )

    random.seed(777)
    _set_torch_seed(777)
    expected_python_random = random.random()  # noqa: S311
    expected_torch_random = torch.rand(4)
    random.seed(777)
    _set_torch_seed(777)
    with pytest.raises(ValueError, match="effective_config_sha256 differs"):
        train_main(
            [
                "--config",
                str(changed_config_path),
                "--data",
                "synthetic",
                "--resume",
                str(first_output / "checkpoints" / "step_000001.pt"),
                "--output-dir",
                str(tmp_path / "mismatch"),
                "--run-name",
                "spec0001_resume_mismatch",
                "--max-train-steps",
                str(RESUME_TARGET_STEPS),
                "--save-every-steps",
                "1",
            ],
        )
    assert random.random() == expected_python_random  # noqa: S311
    torch.testing.assert_close(torch.rand(4), expected_torch_random)


def test_train_cli_resume_rejects_runtime_config_mismatch(tmp_path: Path) -> None:
    """Resume checkpoints are bound to the selected runtime artifact."""
    config_path = _debug_config(tmp_path, selected_runtime_required=True)
    runtime_config = _runtime_config(tmp_path)
    mismatched_runtime_config = tmp_path / "selected_runtime_mismatch.json"
    mismatched_payload = _load_json(runtime_config)
    mismatched_payload["selected_row_id"] = "different_pass_runtime_row"
    mismatched_payload["runtime_policy_id"] = "amp_fp16_conservative_different"
    selected_snapshot = _object(mismatched_payload, "selected_row_snapshot")
    selected_snapshot["row_id"] = "different_pass_runtime_row"
    selected_snapshot["runtime_policy_id"] = "amp_fp16_conservative_different"
    _write_json(mismatched_runtime_config, mismatched_payload)
    first_output = tmp_path / "first"

    assert (
        train_main(
            [
                "--config",
                str(config_path),
                "--runtime-config",
                str(runtime_config),
                "--data",
                "synthetic",
                "--output-dir",
                str(first_output),
                "--run-name",
                "spec0001_resume_runtime_source",
                "--max-train-steps",
                str(SHORT_TRAIN_STEPS),
                "--save-every-steps",
                "1",
            ],
        )
        == 0
    )

    with pytest.raises(ValueError, match="runtime_config_sha256 differs"):
        train_main(
            [
                "--config",
                str(config_path),
                "--runtime-config",
                str(mismatched_runtime_config),
                "--data",
                "synthetic",
                "--resume",
                str(first_output / "checkpoints" / "step_000001.pt"),
                "--output-dir",
                str(tmp_path / "runtime-mismatch"),
                "--run-name",
                "spec0001_resume_runtime_mismatch",
                "--max-train-steps",
                str(RESUME_TARGET_STEPS),
                "--save-every-steps",
                "1",
            ],
        )


def test_checkpoint_restores_all_local_rng_streams(tmp_path: Path) -> None:
    """Checkpoint load restores Python, NumPy Generator, and Torch RNG streams."""
    checkpoint_path = tmp_path / "checkpoint.pt"
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    numpy_generator = np.random.default_rng(123)
    torch_generator = torch.Generator(device="cpu")
    torch_generator.manual_seed(123)
    random.seed(123)
    _set_torch_seed(123)

    save_training_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        numpy_generator=numpy_generator,
        torch_generators={"train_data": torch_generator},
        run_name="rng_restore_source",
        config_path=tmp_path / "config.json",
        config_sha256="config",
        effective_config_sha256="effective",
        runtime_config_sha256="runtime",
        selected_row_id="row",
        runtime_policy_id="policy",
        optimizer_step=1,
        successful_optimizer_update_count=1,
        metric_name="loss",
        metric_value=1.0,
    )
    expected_python = random.random()  # noqa: S311
    expected_numpy = int(numpy_generator.integers(0, 1_000_000))
    expected_torch_global = torch.rand(3)
    expected_torch_named = torch.rand(3, generator=torch_generator)

    loaded_model = torch.nn.Linear(2, 1)
    loaded_optimizer = torch.optim.SGD(loaded_model.parameters(), lr=0.1)
    loaded_numpy_generator = np.random.default_rng(999)
    loaded_torch_generator = torch.Generator(device="cpu")
    loaded_torch_generator.manual_seed(999)
    random.seed(999)
    _set_torch_seed(999)

    loaded = load_training_checkpoint(
        path=checkpoint_path,
        model=loaded_model,
        optimizer=loaded_optimizer,
        numpy_generator=loaded_numpy_generator,
        torch_generators={"train_data": loaded_torch_generator},
        expected_effective_config_sha256="effective",
        expected_runtime_config_sha256="runtime",
        expected_selected_row_id="row",
        expected_runtime_policy_id="policy",
    )

    assert loaded.torch_generator_names == ("train_data",)
    assert random.random() == expected_python  # noqa: S311
    assert int(loaded_numpy_generator.integers(0, 1_000_000)) == expected_numpy
    torch.testing.assert_close(torch.rand(3), expected_torch_global)
    torch.testing.assert_close(
        torch.rand(3, generator=loaded_torch_generator),
        expected_torch_named,
    )


def test_train_cli_tiny_overfit_summary_is_local_and_fail_closed(
    tmp_path: Path,
) -> None:
    """Tiny-overfit summary can be schema-tested without real Kaggle proof."""
    config_path = _tiny_config(tmp_path)
    runtime_config = _runtime_config(tmp_path)
    output_dir = tmp_path / "tiny"

    assert (
        train_main(
            [
                "--config",
                str(config_path),
                "--runtime-config",
                str(runtime_config),
                "--data",
                "synthetic",
                "--fixed-train-patches",
                "configs/spec0001/fixed_32_train_overfit_patches.json",
                "--output-dir",
                str(output_dir),
                "--run-name",
                "spec0001_tiny_overfit_local",
                "--max-train-steps",
                str(SHORT_TRAIN_STEPS),
                "--save-every-steps",
                "1",
            ],
        )
        == 0
    )

    tiny = _load_json(output_dir / "benchmark" / "tiny_overfit_summary.json")
    manifest = _load_json(output_dir / "benchmark" / "artifact_manifest.json")
    assert tiny["status"] == "local_pass"
    assert tiny["full_run_eligible"] is False
    assert tiny["runtime_config_sha256"]
    assert tiny["patch_count"] == FIXED_TINY_PATCH_COUNT
    assert tiny["optimizer_steps"] == SHORT_TRAIN_STEPS
    assert tiny["real_tiny_overfit_status"] == "pending_permission_gated_remote_run"
    assert tiny["gate_health_status"] == "local_not_measured"
    artifact_hashes = _object(manifest, "artifact_hashes")
    assert "tiny_overfit_summary" in artifact_hashes


def test_train_cli_requires_runtime_config_when_config_requires_it(
    tmp_path: Path,
) -> None:
    """Selected-runtime-required configs fail closed without runtime config."""
    config_path = _debug_config(tmp_path, selected_runtime_required=True)

    with pytest.raises(ValueError, match="requires --runtime-config"):
        train_main(
            [
                "--config",
                str(config_path),
                "--data",
                "synthetic",
                "--output-dir",
                str(tmp_path / "missing-runtime"),
                "--run-name",
                "spec0001_missing_runtime",
                "--max-train-steps",
                "1",
            ],
        )


def test_checked_in_selected_runtime_debug_config_requires_runtime(
    tmp_path: Path,
) -> None:
    """The checked-in debug config fails closed without runtime selection."""
    with pytest.raises(ValueError, match="requires --runtime-config"):
        train_main(
            [
                "--config",
                "configs/spec0001/non_eq_vae_selected_runtime_debug.json",
                "--data",
                "synthetic",
                "--output-dir",
                str(tmp_path / "missing-runtime"),
                "--run-name",
                "spec0001_checked_in_missing_runtime",
                "--max-train-steps",
                "1",
            ],
        )


def test_train_cli_rejects_thin_runtime_config(tmp_path: Path) -> None:
    """A pass-looking runtime JSON still needs selected-runtime proof fields."""
    config_path = _debug_config(tmp_path, selected_runtime_required=True)
    runtime_config = tmp_path / "thin_runtime.json"
    _write_json(
        runtime_config,
        {
            "status": "pass",
            "full_run_eligible": True,
            "selected_row_id": "row",
            "runtime_policy_id": "policy",
            "launch_blockers": [],
        },
    )

    with pytest.raises(ValueError, match="benchmark_kind"):
        train_main(
            [
                "--config",
                str(config_path),
                "--runtime-config",
                str(runtime_config),
                "--data",
                "synthetic",
                "--output-dir",
                str(tmp_path / "thin"),
                "--run-name",
                "spec0001_thin_runtime",
                "--max-train-steps",
                "1",
            ],
        )


def test_train_cli_rejects_invalid_numeric_settings(tmp_path: Path) -> None:
    """Zero overrides fail instead of silently falling back to defaults."""
    config_path = _debug_config(tmp_path, selected_runtime_required=False)

    with pytest.raises(ValueError, match="save_every_steps must be positive"):
        train_main(
            [
                "--config",
                str(config_path),
                "--data",
                "synthetic",
                "--output-dir",
                str(tmp_path / "invalid"),
                "--run-name",
                "spec0001_invalid_steps",
                "--max-train-steps",
                "1",
                "--save-every-steps",
                "0",
            ],
        )


def test_train_cli_preserves_explicit_zero_seeds(tmp_path: Path) -> None:
    """Seed 0 is a real configured seed, not a missing-value fallback."""
    config_path = _debug_config(
        tmp_path,
        selected_runtime_required=False,
        global_seed=0,
        data_seed=0,
    )
    output_dir = tmp_path / "zero-seed"

    assert (
        train_main(
            [
                "--config",
                str(config_path),
                "--data",
                "synthetic",
                "--output-dir",
                str(output_dir),
                "--run-name",
                "spec0001_zero_seed",
                "--max-train-steps",
                "1",
            ],
        )
        == 0
    )

    summary = _load_json(output_dir / "benchmark" / "training_summary.json")
    seeds = _object(summary, "seeds")
    assert seeds["global_seed"] == 0
    assert seeds["data_seed"] == 0


def _debug_config(
    tmp_path: Path,
    *,
    selected_runtime_required: bool,
    max_val_steps: int = 1,
    global_seed: int = 20260610,
    data_seed: int = 20260611,
) -> Path:
    config_path = tmp_path / f"debug_config_val_{max_val_steps}.json"
    payload: dict[str, object] = {
        "source_config": str(
            Path("configs/spec0001/non_eq_vae_debug_cpu.json").resolve(),
        ),
        "run": {
            "name": "spec0001_train_cli_debug",
            "mode": "local_synthetic_debug",
        },
        "data": {
            "kind": "synthetic",
            "image_size": 64,
            "train_samples": 4,
            "validation_samples": 2,
        },
        "seeds": {
            "global_seed": global_seed,
            "data_seed": data_seed,
        },
        "runtime": {
            "batch_size": 1,
            "max_train_steps": 2,
            "max_val_steps": max_val_steps,
            "selected_runtime_required": selected_runtime_required,
        },
        "training": {
            "max_train_steps": 2,
            "max_val_steps": max_val_steps,
            "save_every_steps": 1,
        },
    }
    _write_json(config_path, payload)
    return config_path


def _runtime_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "selected_runtime.json"
    selected_row_id = (
        "dual_t4_ddp__bs12__amp_conservative__compile_none__indexed_masked__"
        "policy_amp_fp16_conservative"
    )
    payload: dict[str, object] = {
        "status": "pass",
        "benchmark_kind": "kaggle_runtime_selection",
        "benchmark_source": "kaggle_runtime_benchmark",
        "full_run_eligible": True,
        "full_training_launch_ready": False,
        "selected_row_id": selected_row_id,
        "runtime_policy_id": "amp_fp16_conservative",
        "launch_blockers": [
            "missing_selected_runtime_debug_proof",
            "missing_checkpoint_resume_proof",
            "missing_tiny_overfit_proof",
        ],
        "safety": {
            "dataloader_status": "pass",
            "numerical_check_status": "pass",
            "corruption_check_status": "pass",
            "gate_health_status": "pass",
        },
        "selected_row_snapshot": {
            "row_id": selected_row_id,
            "runtime_policy_id": "amp_fp16_conservative",
            "status": "pass",
        },
    }
    _write_json(config_path, payload)
    return config_path


def _tiny_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "tiny_config.json"
    payload: dict[str, object] = {
        "source_config": str(
            Path("configs/spec0001/non_eq_vae_kaggle_tiny_overfit.json").resolve(),
        ),
        "data": {
            "kind": "synthetic",
            "image_size": 64,
            "fixed_train_patches": (
                "configs/spec0001/fixed_32_train_overfit_patches.json"
            ),
        },
        "runtime": {
            "batch_size": 1,
            "selected_runtime_required": True,
        },
        "training": {
            "max_train_steps": 2,
            "max_val_steps": 1,
            "save_every_steps": 1,
        },
    }
    _write_json(config_path, payload)
    return config_path


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )


def _load_json(path: Path) -> dict[str, object]:
    payload = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(payload, dict):
        raise TypeError(path)
    return cast("dict[str, object]", payload)


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def _object(payload: dict[str, object], key: str) -> dict[str, object]:
    value = payload[key]
    if not isinstance(value, dict):
        raise TypeError(key)
    return cast("dict[str, object]", value)


def _load_checkpoint(path: Path) -> dict[str, object]:
    payload = cast(
        "object",
        torch.load(path, map_location="cpu", weights_only=False),
    )
    if not isinstance(payload, dict):
        raise TypeError(path)
    return cast("dict[str, object]", payload)


def _assert_state_equal(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        torch.testing.assert_close(left, right, atol=0.0, rtol=0.0)
        return
    if isinstance(left, dict) and isinstance(right, dict):
        left_dict = cast("dict[object, object]", left)
        right_dict = cast("dict[object, object]", right)
        assert set(left_dict) == set(right_dict)
        for key, value in left_dict.items():
            _assert_state_equal(value, right_dict[key])
        return
    if isinstance(left, list) and isinstance(right, list):
        left_list = cast("list[object]", left)
        right_list = cast("list[object]", right)
        assert len(left_list) == len(right_list)
        for left_item, right_item in zip(left_list, right_list, strict=True):
            _assert_state_equal(left_item, right_item)
        return
    if isinstance(left, tuple) and isinstance(right, tuple):
        left_tuple = cast("tuple[object, ...]", left)
        right_tuple = cast("tuple[object, ...]", right)
        assert len(left_tuple) == len(right_tuple)
        for left_item, right_item in zip(left_tuple, right_tuple, strict=True):
            _assert_state_equal(left_item, right_item)
        return
    assert left == right


def _set_torch_seed(seed: int) -> None:
    manual_seed = cast("Callable[[int], torch.Generator]", torch.manual_seed)
    manual_seed(seed)
