# Copyright 2026 HiperMaximus
"""Tests for the capped real-data runtime pretest scaffold and guard."""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess  # noqa: S404
import sys
from pathlib import Path
from typing import cast

import pytest

from eqvae.benchmarking.real_data_runtime_pretest import (
    RealDataRuntimePretestRequest,
    write_real_data_runtime_pretest,
)
from eqvae.data.synthetic import SyntheticPatchSpec, write_synthetic_patch_shard

_TINY_IMAGE_SIZE = 8
_TINY_CHANNELS = 3
_TINY_TRAIN_PATCHES = 16
_TINY_VALIDATION_PATCHES = 14
_EXPECTED_FILE_HASH_COUNT = 4
_CANONICAL_REAL_TRAIN_PATCHES = 300_000
_CANONICAL_REAL_VALIDATION_PATCHES = 30_000
_CANONICAL_CAP_TRAIN_PATCHES = 8_192
_CANONICAL_CAP_VALIDATION_PATCHES = 2_048
_CANONICAL_WINDOW_PATCHES = 2_048
_CANONICAL_VALIDATION_WINDOW_PATCHES = 1_024


def test_real_data_runtime_pretest_local_wrong_accelerator_artifacts(
    tmp_path: Path,
) -> None:
    """Local CPU runs write non-promotable artifacts and no selected runtime."""
    repo_root = Path(__file__).resolve().parents[1]
    recommendations_path = write_real_data_runtime_pretest(
        RealDataRuntimePretestRequest(
            config_path=(
                repo_root
                / "configs"
                / "spec0001"
                / "non_eq_vae_kaggle_runtime_benchmark.json"
            ),
            output_dir=tmp_path,
        ),
    )

    benchmark_dir = tmp_path / "benchmark"
    assert recommendations_path == (
        benchmark_dir / "real_data_runtime_pretest_recommendations.json"
    )
    assert not (benchmark_dir / "selected_runtime.json").exists()
    runtime_proof = _load_json(benchmark_dir / "runtime_proof.json")
    recommendations = _load_json(recommendations_path)
    assert runtime_proof["full_run_eligible"] is False
    assert runtime_proof["selection_ready"] is False
    assert runtime_proof["eligible_pass_row_count"] == 0
    assert "real-data identity" in cast("str", runtime_proof["evidence_gate"])
    manifest = _load_json(benchmark_dir / "real_data_runtime_pretest_manifest.json")
    assert manifest["real_data_identity_proof_status"] == "skipped_unsupported"
    assert manifest["validation_windows_exercised"] is False
    assert manifest["timed_rows_eligible"] is False
    wrong_accelerator_count = cast("int", runtime_proof["wrong_accelerator_row_count"])
    assert wrong_accelerator_count > 0
    assert recommendations["writes_selected_runtime"] is False
    assert recommendations["status"] == "pretest_skipped"


def test_real_data_runtime_pretest_writes_identity_crc_and_clean_validation_proof(
    tmp_path: Path,
) -> None:
    """A tiny UBC-format root exercises the real-data proof lane locally."""
    repo_root = Path(__file__).resolve().parents[1]
    data_root = _write_tiny_patch_root(tmp_path)
    config_path = _write_tiny_runtime_pretest_config(
        tmp_path=tmp_path,
        repo_root=repo_root,
        data_root=data_root,
    )

    write_real_data_runtime_pretest(
        RealDataRuntimePretestRequest(
            config_path=config_path,
            output_dir=tmp_path / "run",
        ),
    )

    benchmark_dir = tmp_path / "run" / "benchmark"
    manifest = _load_json(benchmark_dir / "real_data_runtime_pretest_manifest.json")
    runtime_proof = _load_json(benchmark_dir / "runtime_proof.json")
    dataloader_rows = _load_csv(benchmark_dir / "dataloader_matrix.csv")
    assert manifest["real_data_identity_proof_status"] == "local_pass"
    assert manifest["row_count_proof_status"] == "pass"
    assert manifest["crc_validation_status"] == "pass"
    assert manifest["train_windows_exercised"] is True
    assert manifest["validation_windows_exercised"] is True
    assert (
        len(cast("list[object]", manifest["file_hashes"])) == _EXPECTED_FILE_HASH_COUNT
    )

    clean_proof = cast(
        "dict[str, object]",
        manifest["clean_validation_dataloader_proof"],
    )
    assert clean_proof["status"] == "pass"
    assert clean_proof["dataset_class"] == "PatchTrainingDataset"
    assert clean_proof["collate_fn"] == "collate_patch_training_samples"
    assert clean_proof["normalizer"] == "normalize_uint8_batch"
    assert clean_proof["corruption_called"] is False
    assert clean_proof["proof_scope"] == "validation_loader_clean_input_only"
    assert clean_proof["corruption_rng_instrumented"] is False
    assert clean_proof["clean_validation_rng_status"] == (
        "not_exercised_in_this_loader_lane"
    )
    assert clean_proof["clean_validation_rng_consumed"] is None
    assert clean_proof["sample_count"] == _TINY_VALIDATION_PATCHES
    assert clean_proof["partial_batch_observed"] is True

    proof = cast("dict[str, object]", manifest["real_data_proof"])
    splits = cast("dict[str, object]", proof["splits"])
    train = cast("dict[str, object]", splits["train"])
    validation = cast("dict[str, object]", splits["validation"])
    assert train["csv_row_count"] == _TINY_TRAIN_PATCHES
    assert validation["csv_row_count"] == _TINY_VALIDATION_PATCHES
    assert cast("dict[str, object]", train["windows"])["selected_patch_count"] == (
        _TINY_TRAIN_PATCHES
    )
    assert (
        cast("dict[str, object]", validation["windows"])["selected_patch_count"]
        == _TINY_VALIDATION_PATCHES
    )
    assert proof["status"] == "local_pass"
    assert cast("dict[str, object]", proof["window_contract"])["status"] == (
        "local_pass"
    )
    assert runtime_proof["real_data_identity_proof_status"] == "local_pass"
    assert runtime_proof["clean_validation_dataloader_status"] == "pass"
    assert "real-data identity" in cast("str", runtime_proof["evidence_gate"])

    validation_rows = [row for row in dataloader_rows if row["split"] == "validation"]
    assert len(validation_rows) == 1
    assert validation_rows[0]["status"] == "ineligible"
    assert (
        validation_rows[0]["failure_kind"]
        == "clean_validation_path_pass_throughput_grid_pending"
    )
    assert validation_rows[0]["rank_sample_count"] == str(_TINY_VALIDATION_PATCHES)
    assert not (benchmark_dir / "selected_runtime.json").exists()


def test_real_data_runtime_pretest_rejects_prefix_only_real_window_contract(
    tmp_path: Path,
) -> None:
    """Canonical real-data configs must keep the locked spread windows."""
    repo_root = Path(__file__).resolve().parents[1]
    data_root = _write_tiny_patch_root(tmp_path)
    config_path = _write_tiny_runtime_pretest_config(
        tmp_path=tmp_path,
        repo_root=repo_root,
        data_root=data_root,
    )
    config = _load_json(config_path)
    data = cast("dict[str, object]", config["data"])
    data["real_train_patch_count"] = _CANONICAL_REAL_TRAIN_PATCHES
    data["real_validation_patch_count"] = _CANONICAL_REAL_VALIDATION_PATCHES
    benchmark_cap = cast("dict[str, object]", data["benchmark_cap"])
    benchmark_cap["train_patch_count"] = _CANONICAL_CAP_TRAIN_PATCHES
    benchmark_cap["validation_patch_count"] = _CANONICAL_CAP_VALIDATION_PATCHES
    benchmark_cap["train_windows"] = [
        {
            "name": "train_prefix_a",
            "start_row": 0,
            "patch_count": _CANONICAL_WINDOW_PATCHES,
        },
        {
            "name": "train_prefix_b",
            "start_row": _CANONICAL_WINDOW_PATCHES,
            "patch_count": _CANONICAL_WINDOW_PATCHES,
        },
        {
            "name": "train_prefix_c",
            "start_row": 2 * _CANONICAL_WINDOW_PATCHES,
            "patch_count": _CANONICAL_WINDOW_PATCHES,
        },
        {
            "name": "train_prefix_d",
            "start_row": 3 * _CANONICAL_WINDOW_PATCHES,
            "patch_count": _CANONICAL_WINDOW_PATCHES,
        },
    ]
    benchmark_cap["validation_windows"] = [
        {
            "name": "validation_prefix_a",
            "start_row": 0,
            "patch_count": _CANONICAL_VALIDATION_WINDOW_PATCHES,
        },
        {
            "name": "validation_prefix_b",
            "start_row": _CANONICAL_VALIDATION_WINDOW_PATCHES,
            "patch_count": _CANONICAL_VALIDATION_WINDOW_PATCHES,
        },
    ]
    config_path.write_text(
        f"{json.dumps(config, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )

    write_real_data_runtime_pretest(
        RealDataRuntimePretestRequest(
            config_path=config_path,
            output_dir=tmp_path / "run_prefix",
        ),
    )

    manifest = _load_json(
        tmp_path
        / "run_prefix"
        / "benchmark"
        / "real_data_runtime_pretest_manifest.json",
    )
    proof = cast("dict[str, object]", manifest["real_data_proof"])
    window_contract = cast("dict[str, object]", proof["window_contract"])
    assert manifest["real_data_identity_proof_status"] == "fail"
    assert window_contract["status"] == "fail"
    assert window_contract["train_windows_match_locked_real_contract"] is False
    assert window_contract["validation_windows_match_locked_real_contract"] is False


def test_real_data_runtime_pretest_rejects_stale_selected_runtime(
    tmp_path: Path,
) -> None:
    """The direct package writer refuses stale selected-runtime artifacts."""
    repo_root = Path(__file__).resolve().parents[1]
    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()
    (benchmark_dir / "selected_runtime.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="selected_runtime"):
        write_real_data_runtime_pretest(
            RealDataRuntimePretestRequest(
                config_path=(
                    repo_root
                    / "configs"
                    / "spec0001"
                    / "non_eq_vae_kaggle_runtime_benchmark.json"
                ),
                output_dir=tmp_path,
            ),
        )


def test_real_data_pretest_push_guard_requires_dataset_confirmation(
    tmp_path: Path,
) -> None:
    """Real-data pretest pushes require explicit dataset attachment approval."""
    repo_root = Path(__file__).resolve().parents[1]
    fake_bin = _fake_bin(tmp_path=tmp_path, repo_root=repo_root)
    kernel_dir = _generated_kernel_dir(
        tmp_path=tmp_path,
        repo_root=repo_root,
        fake_bin=fake_bin,
    )

    completed = subprocess.run(  # noqa: S603
        (
            _required_executable("bash"),
            str(repo_root / "scripts" / "kaggle_kernel.sh"),
            "push",
            str(kernel_dir),
        ),
        cwd=repo_root,
        env=_guard_environment(fake_bin=fake_bin, full_dataset_confirmed=False),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "KAGGLE_FULL_DATASET_CONFIRMED=1" in completed.stderr


def test_real_data_pretest_push_guard_rejects_wrong_dataset_sources(
    tmp_path: Path,
) -> None:
    """The guard rejects missing or drifted real-data source attachments."""
    repo_root = Path(__file__).resolve().parents[1]
    fake_bin = _fake_bin(tmp_path=tmp_path, repo_root=repo_root)
    kernel_dir = _generated_kernel_dir(
        tmp_path=tmp_path,
        repo_root=repo_root,
        fake_bin=fake_bin,
    )
    metadata_path = kernel_dir / "kernel-metadata.json"
    metadata = _load_json(metadata_path)
    metadata["dataset_sources"] = []
    metadata_path.write_text(
        f"{json.dumps(metadata, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )

    completed = subprocess.run(  # noqa: S603
        (
            _required_executable("bash"),
            str(repo_root / "scripts" / "kaggle_kernel.sh"),
            "push",
            str(kernel_dir),
        ),
        cwd=repo_root,
        env=_guard_environment(fake_bin=fake_bin),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "dataset_sources must be exactly" in completed.stderr


def test_real_data_pretest_push_guard_accepts_generated_kernel(
    tmp_path: Path,
) -> None:
    """The positive guard path reaches fake Kaggle without network access."""
    repo_root = Path(__file__).resolve().parents[1]
    fake_bin = _fake_bin(tmp_path=tmp_path, repo_root=repo_root)
    kernel_dir = _generated_kernel_dir(
        tmp_path=tmp_path,
        repo_root=repo_root,
        fake_bin=fake_bin,
    )

    completed = subprocess.run(  # noqa: S603
        (
            _required_executable("bash"),
            str(repo_root / "scripts" / "kaggle_kernel.sh"),
            "push",
            str(kernel_dir),
        ),
        cwd=repo_root,
        env=_guard_environment(fake_bin=fake_bin),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "fake kaggle kernels push" in completed.stdout


def _generated_kernel_dir(
    *,
    tmp_path: Path,
    repo_root: Path,
    fake_bin: Path,
) -> Path:
    kernel_source = repo_root / "kaggle" / "kernels" / "real_data_runtime_pretest"
    kernel_dir = tmp_path / "real_data_runtime_pretest_kernel"
    kernel_dir.mkdir()
    shutil.copy2(kernel_source / "kernel-metadata.json", kernel_dir)
    subprocess.run(  # noqa: S603
        (
            sys.executable,
            str(repo_root / "scripts" / "build_kaggle_embedded_kernel.py"),
            "--repo-root",
            str(repo_root),
            "--kernel-dir",
            str(kernel_dir),
            "--template",
            str(kernel_source / "run_template.py"),
            "--ready-marker",
            "KAGGLE_REAL_DATA_RUNTIME_PRETEST_READY = True",
        ),
        cwd=repo_root,
        env=_guard_environment(
            fake_bin=fake_bin,
            push_confirmed=False,
            full_dataset_confirmed=False,
        ),
        check=True,
    )
    return kernel_dir


def _fake_bin(*, tmp_path: Path, repo_root: Path) -> Path:
    fake_bin = tmp_path / "fake_bin"
    fake_bin.mkdir(exist_ok=True)
    commit = subprocess.run(  # noqa: S603
        (_required_executable("git"), "rev-parse", "HEAD"),
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (fake_bin / "git").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'if [[ "$1" == "rev-parse" && "${2:-}" == "HEAD" ]]; then\n'
        f"  printf '%s\\n' '{commit}'\n"
        'elif [[ "$1" == "status" && "${2:-}" == "--short" ]]; then\n'
        "  exit 0\n"
        "else\n"
        '  command git "$@"\n'
        "fi\n",
        encoding="utf-8",
    )
    (fake_bin / "kaggle").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\nprintf 'fake kaggle %s\\n' \"$*\"\n",
        encoding="utf-8",
    )
    (fake_bin / "git").chmod(0o755)
    (fake_bin / "kaggle").chmod(0o755)
    return fake_bin


def _required_executable(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        message = f"missing executable: {name}"
        raise RuntimeError(message)
    return path


def _guard_environment(
    *,
    fake_bin: Path,
    push_confirmed: bool = True,
    full_dataset_confirmed: bool = True,
) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PATH"] = f"{fake_bin}{os.pathsep}{environment['PATH']}"
    if push_confirmed:
        environment["KAGGLE_PUSH_CONFIRMED"] = "1"
    else:
        environment.pop("KAGGLE_PUSH_CONFIRMED", None)
    if full_dataset_confirmed:
        environment["KAGGLE_FULL_DATASET_CONFIRMED"] = "1"
    else:
        environment.pop("KAGGLE_FULL_DATASET_CONFIRMED", None)
    return environment


def _load_json(path: Path) -> dict[str, object]:
    return cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def _write_tiny_patch_root(tmp_path: Path) -> Path:
    dataset_root = tmp_path / "patches-pre-shuffled-ubc-ocean" / "dataset"
    write_synthetic_patch_shard(
        bin_path=dataset_root / "ubc_train_shuffled.bin",
        csv_path=dataset_root / "ubc_train_shuffled.csv",
        spec=SyntheticPatchSpec(
            count=_TINY_TRAIN_PATCHES,
            image_size=_TINY_IMAGE_SIZE,
            channels=_TINY_CHANNELS,
            seed=20260619,
        ),
        include_idx=False,
    )
    write_synthetic_patch_shard(
        bin_path=dataset_root / "ubc_ocean_valid.bin",
        csv_path=dataset_root / "ubc_ocean_valid.csv",
        spec=SyntheticPatchSpec(
            count=_TINY_VALIDATION_PATCHES,
            image_size=_TINY_IMAGE_SIZE,
            channels=_TINY_CHANNELS,
            seed=20260620,
        ),
        include_idx=True,
    )
    validation_csv = dataset_root / "ubc_ocean_valid.csv"
    validation_csv.write_text(
        validation_csv.read_text(encoding="utf-8").replace(
            "synthetic_wsi_",
            "validation_synthetic_wsi_",
        ),
        encoding="utf-8",
    )
    return dataset_root.parent


def _write_tiny_runtime_pretest_config(
    *,
    tmp_path: Path,
    repo_root: Path,
    data_root: Path,
) -> Path:
    source = repo_root / "configs" / "spec0001" / "non_eq_vae_model_base.json"
    config = _load_json(
        repo_root / "configs" / "spec0001" / "non_eq_vae_kaggle_runtime_benchmark.json",
    )
    config["source_config"] = str(source)
    data = cast("dict[str, object]", config["data"])
    data["data_root"] = str(data_root)
    data["image_size"] = _TINY_IMAGE_SIZE
    data["channels"] = _TINY_CHANNELS
    data["real_train_patch_count"] = _TINY_TRAIN_PATCHES
    data["real_validation_patch_count"] = _TINY_VALIDATION_PATCHES
    data["benchmark_cap"] = {
        "enabled": True,
        "train_patch_count": _TINY_TRAIN_PATCHES,
        "validation_patch_count": _TINY_VALIDATION_PATCHES,
        "window_policy": "fixed_hashed_spread_windows",
        "train_windows": [
            {"name": "train_head", "start_row": 0, "patch_count": 8},
            {"name": "train_tail", "start_row": 8, "patch_count": 8},
        ],
        "validation_windows": [
            {"name": "validation_head", "start_row": 0, "patch_count": 8},
            {"name": "validation_tail", "start_row": 8, "patch_count": 6},
        ],
        "full_epoch_allowed": False,
        "purpose": "tiny_local_real_data_proof_test",
    }
    config_path = tmp_path / "tiny_real_data_runtime_pretest.json"
    config_path.write_text(
        f"{json.dumps(config, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    return config_path
