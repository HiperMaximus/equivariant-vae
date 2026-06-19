# Copyright 2026 HiperMaximus
"""Tests for the capped real-data runtime pretest scaffold and guard."""

from __future__ import annotations

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
    assert manifest["real_data_identity_proof_status"] == "pending"
    assert manifest["validation_windows_exercised"] is False
    assert manifest["timed_rows_eligible"] is False
    wrong_accelerator_count = cast("int", runtime_proof["wrong_accelerator_row_count"])
    assert wrong_accelerator_count > 0
    assert recommendations["writes_selected_runtime"] is False
    assert recommendations["status"] == "pretest_skipped"


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
