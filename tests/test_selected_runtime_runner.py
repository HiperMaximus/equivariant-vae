# Copyright 2026 HiperMaximus
"""Tests for the spec 0007 selected-runtime train runner."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import cast

from eqvae.cli.selected_runtime_train import main as selected_runtime_train_main
from eqvae.data.fixed_selectors import (
    FIXED_32_TRAIN_OVERFIT_COUNT,
    FIXED_32_TRAIN_OVERFIT_KIND,
    FixedSelectorGenerationContext,
    generate_fixed_selector_document,
    write_fixed_selector_document,
)
from eqvae.data.patch_shards import PatchShardSpec
from eqvae.data.roots import (
    TRAIN_BIN_NAME,
    TRAIN_CSV_NAME,
    VALIDATION_BIN_NAME,
    VALIDATION_CSV_NAME,
)
from eqvae.data.synthetic import SyntheticPatchSpec, write_synthetic_patch_shard
from eqvae.training.selected_runtime import parse_selected_runtime_plan
from eqvae.training.selected_runtime_runner import (
    SELECTED_RUNTIME_AMP_GRAD_SCALER_INIT_SCALE,
    RankDeviceAssignment,
    SelectedRuntimeEnvironmentProbe,
    build_selected_runtime_torchrun_command,
    fixed_selector_full_batch_indices,
    validate_selected_runtime_environment,
    validate_selected_runtime_torchrun_command,
)

SHORT_TRAIN_STEPS = 2
PARTIAL_BATCH_TRAIN_STEPS = 3
TINY_FULL_BATCH_TRAIN_STEPS = 4
SELECTED_RUNTIME_BATCH_SIZE = 12
DDP_WORLD_SIZE = 2
DDP_TINY_EPOCH_BATCHES_PER_RANK = 2
DDP_TINY_PER_RANK_EPOCH_SAMPLES = (
    SELECTED_RUNTIME_BATCH_SIZE * DDP_TINY_EPOCH_BATCHES_PER_RANK
)
DDP_TINY_GLOBAL_EPOCH_SAMPLES = DDP_TINY_PER_RANK_EPOCH_SAMPLES * DDP_WORLD_SIZE
SINGLE_TINY_EPOCH_BATCHES = 3
SINGLE_TINY_FULL_BATCH_EPOCH_SAMPLES = (
    SELECTED_RUNTIME_BATCH_SIZE * SINGLE_TINY_EPOCH_BATCHES
)
IMAGE_SIZE = 64


def test_selected_runtime_runner_dry_run_writes_required_artifacts(  # noqa: PLR0914, PLR0915
    tmp_path: Path,
) -> None:
    """Local dry-run exercises the real runner without promoting readiness."""
    config_path = _runner_config(tmp_path)
    runtime_config = _runtime_config(tmp_path)
    output_dir = tmp_path / "runner-dryrun"

    assert (
        selected_runtime_train_main(
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
                "spec0007_local_runner_dryrun",
                "--max-train-steps",
                str(SHORT_TRAIN_STEPS),
                "--max-val-steps",
                "1",
                "--save-every-steps",
                "1",
                "--dry-run",
            ],
        )
        == 0
    )

    benchmark_dir = output_dir / "benchmark"
    metrics_dir = output_dir / "metrics"
    summary = _load_json(benchmark_dir / "training_summary.json")
    debug_summary = _load_json(benchmark_dir / "selected_runtime_debug_summary.json")
    plan_applied = _load_json(benchmark_dir / "selected_runtime_plan_applied.json")
    resume_proof = _load_json(benchmark_dir / "checkpoint_resume_proof.json")
    gate_health_summary = _load_json(benchmark_dir / "gate_health_summary.json")
    readiness = _load_json(
        benchmark_dir / "local_selected_runtime_readiness.json",
    )
    manifest = _load_json(benchmark_dir / "artifact_manifest.json")
    train_rows = _load_csv(metrics_dir / "train_steps.csv")
    gate_rows = _load_csv(metrics_dir / "gate_health.csv")

    assert not (benchmark_dir / "selected_runtime.json").exists()
    assert summary["status"] == "local_pass"
    assert summary["status_scope"] == "local_selected_runtime_runner"
    assert summary["full_run_eligible"] is False
    assert summary["synthetic_generated"] is True
    assert summary["optimizer_steps_completed"] == SHORT_TRAIN_STEPS
    assert "torchrun --standalone --nproc_per_node=2" in cast(
        "str",
        summary["selected_runtime_launch_command"],
    )
    runtime = _object(summary, "runtime_config")
    assert runtime["consumed"] is True
    assert runtime["per_device_batch_size"] == SELECTED_RUNTIME_BATCH_SIZE
    assert runtime["precision_policy"] == "amp_conservative"
    assert runtime["corruption_strategy"] == "indexed_masked"
    amp_execution = _object(summary, "amp_execution")
    assert (
        amp_execution["grad_scaler_init_scale"]
        == SELECTED_RUNTIME_AMP_GRAD_SCALER_INIT_SCALE
    )
    retained_intervals = cast(
        "list[dict[str, object]]",
        summary["retained_interval_checkpoints"],
    )
    retained_interval_names = {
        Path(cast("str", checkpoint["path"])).name for checkpoint in retained_intervals
    }
    assert retained_interval_names == {"step_000001.pt", "step_000002.pt"}
    assert Path(cast("str", _object(summary, "final_checkpoint")["path"])).name == (
        "final.pt"
    )
    assert Path(cast("str", _object(summary, "best_checkpoint")["path"])).name == (
        "best_model.pt"
    )

    assert debug_summary["real_train_runner_implemented"] is True
    assert debug_summary["remote_pass_ready"] is False
    assert debug_summary["fixed_32_selector_real"] is False
    assert debug_summary["uses_resolve_patch_data_paths"] is True
    assert debug_summary["uses_patch_training_dataset"] is True
    assert debug_summary["uses_collate_patch_training_samples"] is True
    assert debug_summary["uses_normalize_uint8_batch"] is True
    assert "synthetic_dry_run_non_promotable" in _string_list(
        debug_summary["launch_blockers_remaining"],
    )

    assert plan_applied["status"] == "fail"
    assert plan_applied["full_run_eligible"] is False
    assert plan_applied["plan_applied"] is False
    mismatches = _string_list(plan_applied["mismatches"])
    assert any("accelerator_mode" in mismatch for mismatch in mismatches)
    assert any("amp_enabled" in mismatch for mismatch in mismatches)
    assert any("local_ddp_status" in mismatch for mismatch in mismatches)

    assert resume_proof["status"] == "local_pass"
    assert resume_proof["loaded_schema_version"] == "spec0001.checkpoint.v5"
    assert resume_proof["runtime_config_sha256_match"] is True
    assert resume_proof["selected_row_id_match"] is True
    assert resume_proof["runtime_policy_id_match"] is True
    assert resume_proof["model_state_restored"] is True
    assert resume_proof["optimizer_state_restored"] is True

    assert gate_health_summary["status"] == "local_pass"
    assert gate_health_summary["rows_written"] == len(gate_rows)
    assert len(gate_rows) > 0
    assert {row["gate_health_status"] for row in gate_rows} == {"pass"}
    assert {row["requested_autocast_dtype"] for row in gate_rows} == {"float16"}
    assert "module" in gate_rows[0]

    assert readiness["status"] == "fail"
    assert readiness["full_run_eligible"] is False
    assert readiness["real_train_runner_implemented"] is True
    assert readiness["remote_pass_ready"] is False
    assert readiness["fixed_32_selector_real"] is False
    blockers = _string_list(readiness["launch_blockers_remaining"])
    assert "dry_run_synthetic_data_non_promotable" in blockers
    assert "fixed_32_selector_real_false_until_spec0008" in blockers

    assert len(train_rows) == SHORT_TRAIN_STEPS
    assert {row["batch_size"] for row in train_rows} == {"12"}
    assert {row["precision_policy"] for row in train_rows} == {"amp_conservative"}
    assert {row["amp_enabled"] for row in train_rows} == {"false"}
    assert {row["grad_scaler_enabled"] for row in train_rows} == {"false"}
    assert {row["corruption_strategy"] for row in train_rows} == {"indexed_masked"}
    assert manifest["status"] == "local_pass"
    artifact_hashes = _object(manifest, "artifact_hashes")
    assert "selected_runtime_plan_applied" in artifact_hashes
    assert "checkpoint_resume_proof" in artifact_hashes
    assert "gate_health" in artifact_hashes
    assert manifest["reconstruction_sample_nonblank"] is True


def test_selected_runtime_runner_supports_ubc_pre_shuffled_root(
    tmp_path: Path,
) -> None:
    """The real data mode resolves pre-shuffled shard filenames."""
    config_path = _runner_config(tmp_path)
    runtime_config = _runtime_config(tmp_path)
    data_root = _synthetic_ubc_root(tmp_path / "ubc-root")
    output_dir = tmp_path / "runner-ubc-root"

    assert (
        selected_runtime_train_main(
            [
                "--config",
                str(config_path),
                "--runtime-config",
                str(runtime_config),
                "--data",
                "ubc-pre-shuffled",
                "--data-root",
                str(data_root),
                "--output-dir",
                str(output_dir),
                "--run-name",
                "spec0007_local_runner_ubc_root",
                "--max-train-steps",
                "1",
                "--max-val-steps",
                "1",
                "--save-every-steps",
                "1",
                "--dry-run",
            ],
        )
        == 0
    )

    summary = _load_json(output_dir / "benchmark" / "training_summary.json")
    debug_summary = _load_json(
        output_dir / "benchmark" / "selected_runtime_debug_summary.json",
    )
    assert summary["data"] == "ubc-pre-shuffled"
    assert summary["synthetic_generated"] is False
    assert summary["data_root"] == str(data_root)
    assert "synthetic_dry_run_non_promotable" not in _string_list(
        debug_summary["launch_blockers_remaining"],
    )


def test_selected_runtime_runner_consumes_fixed_train_selector(
    tmp_path: Path,
) -> None:
    """A provided fixed-32 selector restricts the selected-runtime train loader."""
    config_path = _runner_config(tmp_path)
    runtime_config = _runtime_config(tmp_path)
    data_root = _synthetic_ubc_root(tmp_path / "ubc-root")
    selector_path = _fixed_selector_for_root(tmp_path, data_root=data_root)
    output_dir = tmp_path / "runner-fixed-selector"

    assert (
        selected_runtime_train_main(
            [
                "--config",
                str(config_path),
                "--runtime-config",
                str(runtime_config),
                "--data",
                "ubc-pre-shuffled",
                "--data-root",
                str(data_root),
                "--fixed-train-patches",
                str(selector_path),
                "--output-dir",
                str(output_dir),
                "--run-name",
                "spec0008_fixed_selector_runner",
                "--max-train-steps",
                "1",
                "--max-val-steps",
                "1",
                "--save-every-steps",
                "1",
                "--dry-run",
            ],
        )
        == 0
    )

    summary = _load_json(output_dir / "benchmark" / "training_summary.json")
    debug_summary = _load_json(
        output_dir / "benchmark" / "selected_runtime_debug_summary.json",
    )
    assert summary["fixed_train_patches"] == str(selector_path)
    assert summary["fixed_train_patch_count"] == FIXED_32_TRAIN_OVERFIT_COUNT
    assert debug_summary["fixed_train_patches"] == str(selector_path)
    assert debug_summary["fixed_train_patch_count"] == FIXED_32_TRAIN_OVERFIT_COUNT


def test_selected_runtime_runner_sizes_eps_from_partial_batch(
    tmp_path: Path,
) -> None:
    """Fixed-32 debug proof handles the final 8-sample batch under bs12."""
    config_path = _runner_config(tmp_path)
    runtime_config = _runtime_config(tmp_path)
    data_root = _synthetic_ubc_root(tmp_path / "ubc-root")
    selector_path = _fixed_selector_for_root(tmp_path, data_root=data_root)
    output_dir = tmp_path / "runner-fixed-selector-partial"

    assert (
        selected_runtime_train_main(
            [
                "--config",
                str(config_path),
                "--runtime-config",
                str(runtime_config),
                "--data",
                "ubc-pre-shuffled",
                "--data-root",
                str(data_root),
                "--fixed-train-patches",
                str(selector_path),
                "--output-dir",
                str(output_dir),
                "--run-name",
                "spec0008_fixed_selector_partial_batch",
                "--max-train-steps",
                str(PARTIAL_BATCH_TRAIN_STEPS),
                "--max-val-steps",
                "1",
                "--save-every-steps",
                "1",
                "--dry-run",
            ],
        )
        == 0
    )

    summary = _load_json(output_dir / "benchmark" / "training_summary.json")
    train_rows = _load_csv(output_dir / "metrics" / "train_steps.csv")

    assert summary["optimizer_steps_completed"] == PARTIAL_BATCH_TRAIN_STEPS
    assert summary["fixed_train_patch_count"] == FIXED_32_TRAIN_OVERFIT_COUNT
    assert [row["batch_size"] for row in train_rows] == ["12", "12", "8"]


def test_fixed_selector_full_batch_indices_pad_ddp_tiny_batches() -> None:
    """Tiny-overfit DDP sampler repeats selector rows into full rank batches."""
    rank0 = fixed_selector_full_batch_indices(
        dataset_size=FIXED_32_TRAIN_OVERFIT_COUNT,
        batch_size=SELECTED_RUNTIME_BATCH_SIZE,
        world_size=DDP_WORLD_SIZE,
        rank=0,
    )
    rank1 = fixed_selector_full_batch_indices(
        dataset_size=FIXED_32_TRAIN_OVERFIT_COUNT,
        batch_size=SELECTED_RUNTIME_BATCH_SIZE,
        world_size=DDP_WORLD_SIZE,
        rank=1,
    )

    assert len(rank0) == DDP_TINY_PER_RANK_EPOCH_SAMPLES
    assert len(rank1) == DDP_TINY_PER_RANK_EPOCH_SAMPLES
    assert len(rank0) % SELECTED_RUNTIME_BATCH_SIZE == 0
    assert len(rank1) % SELECTED_RUNTIME_BATCH_SIZE == 0
    assert set(rank0) | set(rank1) == set(range(FIXED_32_TRAIN_OVERFIT_COUNT))
    assert len(rank0) + len(rank1) == DDP_TINY_GLOBAL_EPOCH_SAMPLES


def test_selected_runtime_tiny_fixed_selector_uses_full_batches(
    tmp_path: Path,
) -> None:
    """Tiny-overfit mode repeats fixed-32 rows instead of emitting tail batches."""
    config_path = _tiny_runner_config(tmp_path)
    runtime_config = _runtime_config(tmp_path)
    data_root = _synthetic_ubc_root(tmp_path / "ubc-root")
    selector_path = _fixed_selector_for_root(tmp_path, data_root=data_root)
    output_dir = tmp_path / "runner-tiny-fixed-selector"

    assert (
        selected_runtime_train_main(
            [
                "--config",
                str(config_path),
                "--runtime-config",
                str(runtime_config),
                "--data",
                "ubc-pre-shuffled",
                "--data-root",
                str(data_root),
                "--fixed-train-patches",
                str(selector_path),
                "--output-dir",
                str(output_dir),
                "--run-name",
                "spec0008_tiny_full_batch_selector",
                "--max-train-steps",
                str(TINY_FULL_BATCH_TRAIN_STEPS),
                "--max-val-steps",
                "1",
                "--save-every-steps",
                str(TINY_FULL_BATCH_TRAIN_STEPS),
                "--dry-run",
            ],
        )
        == 0
    )

    summary = _load_json(output_dir / "benchmark" / "training_summary.json")
    tiny = _load_json(output_dir / "benchmark" / "tiny_overfit_summary.json")
    train_rows = _load_csv(output_dir / "metrics" / "train_steps.csv")

    assert summary["fixed_train_patch_count"] == FIXED_32_TRAIN_OVERFIT_COUNT
    assert summary["fixed_train_repeated_to_full_batch"] is True
    assert summary["train_sampler_policy"] == "fixed32_tiny_full_batch_repeated"
    assert (
        summary["train_effective_global_epoch_samples"]
        == SINGLE_TINY_FULL_BATCH_EPOCH_SAMPLES
    )
    assert (
        summary["train_effective_per_rank_epoch_samples"]
        == SINGLE_TINY_FULL_BATCH_EPOCH_SAMPLES
    )
    assert tiny["patch_count"] == FIXED_32_TRAIN_OVERFIT_COUNT
    assert tiny["fixed_train_repeated_to_full_batch"] is True
    assert tiny["grad_scaler_init_scale"] == SELECTED_RUNTIME_AMP_GRAD_SCALER_INIT_SCALE
    assert tiny["amp_step_skipped_count"] == 0
    assert tiny["nonfinite_count"] == 0
    assert tiny["observed_batch_sizes"] == [SELECTED_RUNTIME_BATCH_SIZE]
    assert [row["batch_size"] for row in train_rows] == ["12", "12", "12", "12"]


def test_selected_runtime_torchrun_command_validation(tmp_path: Path) -> None:
    """The launcher builder is tokenized and rejects misleading commands."""
    plan = parse_selected_runtime_plan(_runtime_config(tmp_path))
    command = build_selected_runtime_torchrun_command(
        config_path=tmp_path / "config.json",
        runtime_config=plan.path,
        data="ubc-pre-shuffled",
        data_root="auto",
        output_dir=tmp_path / "out",
        run_name="spec0007_launch",
        max_train_steps=2,
        max_val_steps=1,
    )

    assert validate_selected_runtime_torchrun_command(command.tokens, plan=plan) == ()
    assert command.tokens[:3] == (
        "torchrun",
        "--standalone",
        "--nproc_per_node=2",
    )
    without_standalone = tuple(
        part for part in command.tokens if part != "--standalone"
    )
    assert "selected_runtime_runner_launch_missing_standalone" in (
        validate_selected_runtime_torchrun_command(without_standalone, plan=plan)
    )
    wrong_nproc = tuple(
        "--nproc_per_node=1" if part == "--nproc_per_node=2" else part
        for part in command.tokens
    )
    assert "selected_runtime_runner_launch_wrong_nproc" in (
        validate_selected_runtime_torchrun_command(wrong_nproc, plan=plan)
    )
    duplicate_nproc = (*command.tokens, "--nproc_per_node", "1")
    assert "selected_runtime_runner_launch_duplicate_nproc" in (
        validate_selected_runtime_torchrun_command(duplicate_nproc, plan=plan)
    )
    wrong_module = tuple(
        "eqvae.cli.train" if part == "eqvae.cli.selected_runtime_train" else part
        for part in command.tokens
    )
    assert "selected_runtime_runner_launch_wrong_module" in (
        validate_selected_runtime_torchrun_command(wrong_module, plan=plan)
    )


def test_selected_runtime_environment_validation(tmp_path: Path) -> None:
    """Rank/device proof accepts exact dual T4 facts and rejects common lies."""
    plan = parse_selected_runtime_plan(_runtime_config(tmp_path))
    passing = SelectedRuntimeEnvironmentProbe(
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
        rank_assignments=(
            RankDeviceAssignment(
                rank=0,
                local_rank=0,
                device=0,
                current_device=0,
                world_size=2,
                device_name="Tesla T4",
            ),
            RankDeviceAssignment(
                rank=1,
                local_rank=1,
                device=1,
                current_device=1,
                world_size=2,
                device_name="Tesla T4",
            ),
        ),
        distributed_initialized=True,
    )

    assert validate_selected_runtime_environment(passing, plan=plan) == ()
    single_t4 = replace(
        passing,
        cuda_device_count=1,
        visible_device_count=1,
        gpu_names=("Tesla T4",),
        world_size=1,
        nproc_per_node=1,
        rank_assignments=(),
    )
    single_errors = validate_selected_runtime_environment(single_t4, plan=plan)
    assert "selected_runtime_runner_single_visible_t4" in single_errors
    assert "selected_runtime_runner_world_size_mismatch" in single_errors
    wrong_accelerator = replace(
        passing,
        machine_shape="NvidiaTeslaP100",
        accelerator_mode="single_gpu",
    )
    wrong_errors = validate_selected_runtime_environment(
        wrong_accelerator,
        plan=plan,
    )
    assert "selected_runtime_runner_wrong_accelerator" in wrong_errors
    assert "selected_runtime_runner_wrong_accelerator_mode" in wrong_errors
    bad_rank = replace(
        passing,
        rank_assignments=(
            RankDeviceAssignment(
                rank=0,
                local_rank=0,
                device=1,
                current_device=1,
                world_size=2,
                device_name="Tesla T4",
            ),
        ),
    )
    assert "selected_runtime_runner_rank_device_mismatch" in (
        validate_selected_runtime_environment(bad_rank, plan=plan)
    )
    not_initialized = replace(passing, distributed_initialized=False)
    assert "selected_runtime_runner_distributed_not_initialized" in (
        validate_selected_runtime_environment(not_initialized, plan=plan)
    )


def _runner_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "runner_config.json"
    payload: dict[str, object] = {
        "source_config": str(
            Path("configs/spec0001/non_eq_vae_debug_cpu.json").resolve(),
        ),
        "run": {
            "name": "spec0007_selected_runtime_runner",
            "mode": "selected_runtime_real_runner",
        },
        "data": {
            "kind": "synthetic",
            "image_size": IMAGE_SIZE,
        },
        "seeds": {
            "global_seed": 20260610,
            "data_seed": 20260611,
            "corruption_seed": 20260612,
        },
        "training": {
            "max_train_steps": SHORT_TRAIN_STEPS,
            "max_val_steps": 1,
            "save_every_steps": 1,
        },
    }
    _write_json(config_path, payload)
    return config_path


def _tiny_runner_config(tmp_path: Path) -> Path:
    config_path = _runner_config(tmp_path)
    payload = _load_json(config_path)
    run = _object(payload, "run")
    run["name"] = "spec0008_selected_runtime_tiny_runner"
    run["mode"] = "kaggle_tiny_overfit"
    tiny_path = tmp_path / "tiny_runner_config.json"
    _write_json(tiny_path, payload)
    return tiny_path


def _runtime_config(tmp_path: Path) -> Path:
    runtime_dir = tmp_path / "runtime_bundle" / "benchmark"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    config_path = runtime_dir / "selected_runtime.json"
    source_dir = Path("runs/kaggle/runtime_selection_v5/benchmark")
    _write_json(config_path, _load_json(source_dir / "selected_runtime.json"))
    (runtime_dir / "runtime_proof.json").write_text(
        (source_dir / "runtime_proof.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return config_path


def _synthetic_ubc_root(root: Path) -> Path:
    write_synthetic_patch_shard(
        bin_path=root / TRAIN_BIN_NAME,
        csv_path=root / TRAIN_CSV_NAME,
        spec=SyntheticPatchSpec(
            count=32,
            image_size=IMAGE_SIZE,
            channels=3,
            seed=20260621,
        ),
        include_idx=False,
    )
    write_synthetic_patch_shard(
        bin_path=root / VALIDATION_BIN_NAME,
        csv_path=root / VALIDATION_CSV_NAME,
        spec=SyntheticPatchSpec(
            count=32,
            image_size=IMAGE_SIZE,
            channels=3,
            seed=20260622,
        ),
        include_idx=True,
    )
    return root


def _fixed_selector_for_root(tmp_path: Path, *, data_root: Path) -> Path:
    selector_path = tmp_path / "fixed_32_train_overfit_patches.json"
    holdout_path = tmp_path / "masked_holdout.csv"
    holdout_path.write_text(
        "image_id,label,is_updated_image_id\nnot_present,HGSC,false\n",
        encoding="utf-8",
    )
    document = generate_fixed_selector_document(
        selector_kind=FIXED_32_TRAIN_OVERFIT_KIND,
        shard_spec=PatchShardSpec(
            bin_path=data_root / TRAIN_BIN_NAME,
            csv_path=data_root / TRAIN_CSV_NAME,
            image_size=IMAGE_SIZE,
            validate_crc=True,
        ),
        source_split="train",
        context=FixedSelectorGenerationContext(
            data_root=data_root,
            masked_holdout_wsi_ids=frozenset({"not_present"}),
        ),
    )
    write_fixed_selector_document(
        path=selector_path,
        document=replace(document, masked_holdout_exclusion=str(holdout_path)),
    )
    return selector_path


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


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        message = "expected list"
        raise TypeError(message)
    items = cast("list[object]", value)
    return [item for item in items if isinstance(item, str)]
