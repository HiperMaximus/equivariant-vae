# Copyright 2026 HiperMaximus
"""Tests for generated single-file Kaggle smoke kernels."""

from __future__ import annotations

import json
import os
import shutil
import subprocess  # noqa: S404
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from eqvae.benchmarking.kaggle_smoke import (
    SETUP_DATA_KIND,
    SETUP_SMOKE_KIND,
    SETUP_SMOKE_SOURCE,
)

_EXPECTED_SETUP_APPLIED_COUNT = 2


@dataclass(frozen=True)
class UploadSimulation:
    """Paths for a simulated Kaggle single-file upload."""

    upload_dir: Path
    output_dir: Path


def test_embedded_setup_kernel_survives_single_file_upload_simulation(
    tmp_path: Path,
) -> None:
    """The setup smoke works when only metadata plus `run.py` are present."""
    repo_root = Path(__file__).resolve().parents[1]
    simulation = _build_upload_simulation(
        tmp_path=tmp_path,
        repo_root=repo_root,
        kernel_name="setup_smoke",
        ready_marker="KAGGLE_SETUP_SMOKE_READY = True",
    )

    subprocess.run(  # noqa: S603
        (sys.executable, str(simulation.upload_dir / "run.py")),
        cwd=simulation.upload_dir,
        check=True,
        env=_run_environment(simulation.output_dir),
    )

    artifact_path = simulation.output_dir / "benchmark" / "kaggle_setup_smoke.json"
    payload = _load_json(artifact_path)
    data = cast("dict[str, object]", payload["data"])
    runtime = cast("dict[str, object]", payload["runtime"])
    train = cast("dict[str, object]", payload["train"])
    manifest = cast("dict[str, object]", payload["payload_manifest"])
    assert payload["status"] == "smoke_pass"
    assert payload["status_scope"] == "non_promotable_setup_smoke"
    assert payload["benchmark_kind"] == SETUP_SMOKE_KIND
    assert payload["benchmark_source"] == SETUP_SMOKE_SOURCE
    assert payload["full_run_eligible"] is False
    assert data["kind"] == SETUP_DATA_KIND
    assert not data["dataset_slug"]
    assert data["origin"] == "synthetic_or_ephemeral_path"
    assert runtime["requires_cuda_t4"] is False
    assert train["total_applied_count"] == _EXPECTED_SETUP_APPLIED_COUNT
    assert manifest["schema_version"] == "spec0001.kaggle_payload_manifest.v1"


def test_embedded_real_data_kernel_survives_single_file_upload_simulation(
    tmp_path: Path,
) -> None:
    """The real-data smoke can import from only metadata plus `run.py`."""
    repo_root = Path(__file__).resolve().parents[1]
    simulation = _build_upload_simulation(
        tmp_path=tmp_path,
        repo_root=repo_root,
        kernel_name="non_eq_vae_debug",
        ready_marker="KAGGLE_SMOKE_READY = True",
    )
    environment = _run_environment(simulation.output_dir)
    environment["EQVAE_LOCAL_UPLOAD_SIMULATION_ONLY"] = "1"

    subprocess.run(  # noqa: S603
        (sys.executable, str(simulation.upload_dir / "run.py")),
        cwd=simulation.upload_dir,
        check=True,
        env=environment,
    )

    artifact_path = simulation.output_dir / "benchmark" / "kaggle_import_smoke.json"
    payload = _load_json(artifact_path)
    manifest = cast("dict[str, object]", payload["payload_manifest"])
    assert payload["status"] == "import_smoke_pass"
    assert payload["status_scope"] == "non_promotable_local_upload_simulation"
    assert payload["benchmark_kind"] == "real_data_kaggle_debug_smoke_import_only"
    assert payload["config_exists"] is True
    assert manifest["schema_version"] == "spec0001.kaggle_payload_manifest.v1"


def _build_upload_simulation(
    *,
    tmp_path: Path,
    repo_root: Path,
    kernel_name: str,
    ready_marker: str,
) -> UploadSimulation:
    source_kernel = repo_root / "kaggle" / "kernels" / kernel_name
    build_script = repo_root / "scripts" / "build_kaggle_embedded_kernel.py"
    generated_kernel = tmp_path / f"generated_{kernel_name}"
    upload_dir = tmp_path / f"upload_{kernel_name}"
    output_dir = tmp_path / f"output_{kernel_name}"
    generated_kernel.mkdir()
    upload_dir.mkdir()
    shutil.copy2(source_kernel / "kernel-metadata.json", generated_kernel)

    subprocess.run(  # noqa: S603
        (
            sys.executable,
            str(build_script),
            "--repo-root",
            str(repo_root),
            "--kernel-dir",
            str(generated_kernel),
            "--template",
            str(source_kernel / "run_template.py"),
            "--ready-marker",
            ready_marker,
            "--allow-dirty",
        ),
        cwd=repo_root,
        check=True,
    )

    shutil.copy2(generated_kernel / "kernel-metadata.json", upload_dir)
    shutil.copy2(generated_kernel / "run.py", upload_dir)
    assert not (upload_dir / "payload").exists()
    assert not (upload_dir / "run_template.py").exists()
    return UploadSimulation(upload_dir=upload_dir, output_dir=output_dir)


def _run_environment(output_dir: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment["EQVAE_OUTPUT_DIR"] = str(output_dir)
    environment.pop("EQVAE_DATA_ROOT", None)
    environment.pop("PYTHONPATH", None)
    return environment


def _load_json(path: Path) -> dict[str, object]:
    return cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
