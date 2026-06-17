# Copyright 2026 HiperMaximus
"""Tests for the generated single-file Kaggle setup-smoke kernel."""

from __future__ import annotations

import json
import os
import shutil
import subprocess  # noqa: S404
import sys
from pathlib import Path
from typing import cast

from eqvae.benchmarking.kaggle_smoke import (
    SETUP_DATA_KIND,
    SETUP_SMOKE_KIND,
    SETUP_SMOKE_SOURCE,
)

_EXPECTED_SETUP_APPLIED_COUNT = 2


def test_embedded_setup_kernel_survives_single_file_upload_simulation(
    tmp_path: Path,
) -> None:
    """The setup smoke works when only metadata plus `run.py` are present."""
    repo_root = Path(__file__).resolve().parents[1]
    source_kernel = repo_root / "kaggle" / "kernels" / "setup_smoke"
    build_script = repo_root / "scripts" / "build_kaggle_embedded_kernel.py"
    generated_kernel = tmp_path / "generated_kernel"
    upload_dir = tmp_path / "upload"
    output_dir = tmp_path / "output"
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
            "--allow-dirty",
        ),
        cwd=repo_root,
        check=True,
    )

    shutil.copy2(generated_kernel / "kernel-metadata.json", upload_dir)
    shutil.copy2(generated_kernel / "run.py", upload_dir)
    assert not (upload_dir / "payload").exists()
    assert not (upload_dir / "run_template.py").exists()

    environment = os.environ.copy()
    environment["EQVAE_OUTPUT_DIR"] = str(output_dir)
    environment.pop("EQVAE_DATA_ROOT", None)
    environment.pop("PYTHONPATH", None)
    subprocess.run(  # noqa: S603
        (sys.executable, str(upload_dir / "run.py")),
        cwd=upload_dir,
        check=True,
        env=environment,
    )

    artifact_path = output_dir / "benchmark" / "kaggle_setup_smoke.json"
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


def _load_json(path: Path) -> dict[str, object]:
    return cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
