# Copyright 2026 HiperMaximus
"""Tests for generated single-file Kaggle smoke kernels."""

from __future__ import annotations

import base64
import io
import json
import os
import re
import shutil
import subprocess  # noqa: S404
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from eqvae.benchmarking.kaggle_smoke import (
    SETUP_DATA_KIND,
    SETUP_SMOKE_KIND,
    SETUP_SMOKE_SOURCE,
)
from eqvae.benchmarking.real_data_runtime_pretest import (
    EXPECTED_DATASET_SLUG,
    REAL_DATA_PRETEST_SOURCE,
)
from eqvae.benchmarking.synthetic_timing import (
    MANIFEST_FILENAME,
    MATRIX_FILENAME,
    RECOMMENDATIONS_FILENAME,
    RUNTIME_PROOF_FILENAME,
    SYNTHETIC_TIMING_KIND,
    SYNTHETIC_TIMING_SCOPE,
    SYNTHETIC_TIMING_SOURCE,
)

_EXPECTED_SETUP_APPLIED_COUNT = 2
_EXPECTED_SYNTHETIC_TIMING_BLOCKED_CLAIMS = {
    "final_batch_size",
    "final_precision_policy",
    "final_corruption_strategy",
    "final_dataloader_settings",
    "final_single_vs_dual_t4",
    "real_data_loader_throughput",
    "convergence",
    "paper_evidence",
    "full_run_readiness",
}
_EXPECTED_REAL_DATA_PRETEST_BLOCKED_CLAIMS = {
    "final_runtime_selection",
    "final_batch_size",
    "final_precision_policy",
    "final_corruption_strategy",
    "final_dataloader_settings",
    "single_vs_dual_t4_final_choice",
    "convergence",
    "paper_evidence",
    "full_run_readiness",
}
_RUNTIME_SELECTION_V8_PAYLOAD_FILES = {
    "runs/kaggle/real_data_runtime_pretest_v8/benchmark/runtime_proof.json",
    "runs/kaggle/real_data_runtime_pretest_v8/benchmark/runtime_matrix.csv",
    "runs/kaggle/real_data_runtime_pretest_v8/benchmark/dataloader_matrix.csv",
    "runs/kaggle/real_data_runtime_pretest_v8/benchmark/numerical_checks.csv",
    "runs/kaggle/real_data_runtime_pretest_v8/benchmark/corruption_checks.csv",
    "runs/kaggle/real_data_runtime_pretest_v8/benchmark/gate_health_summary.json",
    "runs/kaggle/real_data_runtime_pretest_v8/metrics/gate_health.csv",
}
_EMBEDDED_PAYLOAD_B64_PATTERN = re.compile(
    r'EMBEDDED_PAYLOAD_B64 = """\n(?P<payload>.*?)\n"""',
    flags=re.DOTALL,
)
_MASKED_HOLDOUT_CSV_PAYLOAD_PATH = "docs/data/ubc_ocean_masked_holdout_ids.csv"


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


def test_embedded_synthetic_timing_kernel_survives_single_file_upload_simulation(
    tmp_path: Path,
) -> None:
    """Synthetic timing works when only metadata plus `run.py` are present."""
    repo_root = Path(__file__).resolve().parents[1]
    simulation = _build_upload_simulation(
        tmp_path=tmp_path,
        repo_root=repo_root,
        kernel_name="synthetic_timing",
        ready_marker="KAGGLE_SYNTHETIC_TIMING_READY = True",
    )
    environment = _run_environment(simulation.output_dir)
    environment["EQVAE_SYNTHETIC_TIMING_TINY_PROFILE"] = "1"

    subprocess.run(  # noqa: S603
        (sys.executable, str(simulation.upload_dir / "run.py")),
        cwd=simulation.upload_dir,
        check=True,
        env=environment,
    )

    benchmark_dir = simulation.output_dir / "benchmark"
    assert {path.name for path in benchmark_dir.iterdir()} == {
        MANIFEST_FILENAME,
        RUNTIME_PROOF_FILENAME,
        MATRIX_FILENAME,
        RECOMMENDATIONS_FILENAME,
    }
    assert not (benchmark_dir / "selected_runtime.json").exists()
    manifest = _load_json(benchmark_dir / MANIFEST_FILENAME)
    runtime_proof = _load_json(benchmark_dir / RUNTIME_PROOF_FILENAME)
    recommendations = _load_json(benchmark_dir / RECOMMENDATIONS_FILENAME)
    for payload in (manifest, runtime_proof, recommendations):
        blocked_claims = cast("dict[str, bool]", payload["blocked_claims"])
        assert payload["benchmark_kind"] == SYNTHETIC_TIMING_KIND
        assert payload["benchmark_source"] == SYNTHETIC_TIMING_SOURCE
        assert payload["status_scope"] == SYNTHETIC_TIMING_SCOPE
        assert payload["full_run_eligible"] is False
        assert set(blocked_claims) == _EXPECTED_SYNTHETIC_TIMING_BLOCKED_CLAIMS
        assert all(blocked_claims.values())
    profile = cast("dict[str, object]", manifest["profile"])
    data = cast("dict[str, object]", manifest["data"])
    timing_plan = cast("dict[str, object]", manifest["timing_plan"])
    assert profile["name"] == "synthetic_binary_tiny_upload_simulation_v1"
    assert data["generation_excluded_from_timing"] is True
    assert data["local_upload_simulation"] is True
    assert timing_plan["timing_phase"] == "local_upload_simulation"
    assert timing_plan["explicit_row_specs"] is False
    assert timing_plan["batch_sizes"] == [2]


def test_embedded_real_data_runtime_pretest_kernel_import_simulation(
    tmp_path: Path,
) -> None:
    """Real-data pretest imports from generated run.py without data access."""
    repo_root = Path(__file__).resolve().parents[1]
    simulation = _build_upload_simulation(
        tmp_path=tmp_path,
        repo_root=repo_root,
        kernel_name="real_data_runtime_pretest",
        ready_marker="KAGGLE_REAL_DATA_RUNTIME_PRETEST_READY = True",
    )
    environment = _run_environment(simulation.output_dir)
    environment["EQVAE_REAL_DATA_RUNTIME_PRETEST_IMPORT_ONLY"] = "1"

    subprocess.run(  # noqa: S603
        (sys.executable, str(simulation.upload_dir / "run.py")),
        cwd=simulation.upload_dir,
        check=True,
        env=environment,
    )

    benchmark_dir = simulation.output_dir / "benchmark"
    assert {path.name for path in benchmark_dir.iterdir()} == {
        "real_data_runtime_pretest_import.json",
    }
    assert not (benchmark_dir / "selected_runtime.json").exists()
    payload = _load_json(benchmark_dir / "real_data_runtime_pretest_import.json")
    blocked_claims = cast("dict[str, bool]", payload["blocked_claims"])
    manifest = cast("dict[str, object]", payload["payload_manifest"])
    entries = cast("dict[str, object]", manifest["entries"])
    assert payload["status"] == "import_smoke_pass"
    assert payload["status_scope"] == "non_promotable_local_upload_simulation"
    assert payload["benchmark_kind"] == "real_data_runtime_pretest_import_only"
    assert payload["benchmark_source"] == REAL_DATA_PRETEST_SOURCE
    assert payload["full_run_eligible"] is False
    assert payload["writes_selected_runtime"] is False
    assert payload["expected_dataset_slug"] == EXPECTED_DATASET_SLUG
    assert set(blocked_claims) == _EXPECTED_REAL_DATA_PRETEST_BLOCKED_CLAIMS
    assert all(blocked_claims.values())
    assert manifest["schema_version"] == "spec0001.kaggle_payload_manifest.v1"
    assert _MASKED_HOLDOUT_CSV_PAYLOAD_PATH in entries
    assert _MASKED_HOLDOUT_CSV_PAYLOAD_PATH in _embedded_payload_names(
        simulation.upload_dir / "run.py",
    )


def test_embedded_real_data_runtime_pretest_kernel_full_local_simulation(
    tmp_path: Path,
) -> None:
    """Generated real-data pretest launcher accepts its full artifact set."""
    repo_root = Path(__file__).resolve().parents[1]
    simulation = _build_upload_simulation(
        tmp_path=tmp_path,
        repo_root=repo_root,
        kernel_name="real_data_runtime_pretest",
        ready_marker="KAGGLE_REAL_DATA_RUNTIME_PRETEST_READY = True",
    )

    subprocess.run(  # noqa: S603
        (sys.executable, str(simulation.upload_dir / "run.py")),
        cwd=simulation.upload_dir,
        check=True,
        env=_run_environment(simulation.output_dir),
    )

    benchmark_dir = simulation.output_dir / "benchmark"
    assert {path.name for path in benchmark_dir.iterdir()} == {
        "real_data_runtime_pretest_manifest.json",
        "runtime_proof.json",
        "runtime_matrix.csv",
        "dataloader_matrix.csv",
        "numerical_checks.csv",
        "corruption_checks.csv",
        "gate_health_summary.json",
        "real_data_runtime_pretest_recommendations.json",
        "phase_timings.json",
    }
    assert not (benchmark_dir / "selected_runtime.json").exists()
    runtime_proof = _load_json(benchmark_dir / "runtime_proof.json")
    manifest = _load_json(benchmark_dir / "real_data_runtime_pretest_manifest.json")
    phase_timings = _load_json(benchmark_dir / "phase_timings.json")
    assert runtime_proof["phase_timings"] == phase_timings
    assert manifest["phase_timings"] == phase_timings


def test_embedded_runtime_selection_kernel_import_simulation(
    tmp_path: Path,
) -> None:
    """Runtime-selection kernel imports and carries v8 provenance payload files."""
    repo_root = Path(__file__).resolve().parents[1]
    simulation = _build_upload_simulation(
        tmp_path=tmp_path,
        repo_root=repo_root,
        kernel_name="runtime_selection",
        ready_marker="KAGGLE_RUNTIME_SELECTION_READY = True",
    )
    environment = _run_environment(simulation.output_dir)
    environment["EQVAE_RUNTIME_SELECTION_IMPORT_ONLY"] = "1"

    subprocess.run(  # noqa: S603
        (sys.executable, str(simulation.upload_dir / "run.py")),
        cwd=simulation.upload_dir,
        check=True,
        env=environment,
    )

    benchmark_dir = simulation.output_dir / "benchmark"
    assert {path.name for path in benchmark_dir.iterdir()} == {
        "runtime_selection_import.json",
    }
    assert not (benchmark_dir / "selected_runtime.json").exists()
    payload = _load_json(benchmark_dir / "runtime_selection_import.json")
    manifest = cast("dict[str, object]", payload["payload_manifest"])
    entries = cast("dict[str, object]", manifest["entries"])
    assert payload["status"] == "import_smoke_pass"
    assert payload["status_scope"] == "non_promotable_local_upload_simulation"
    assert payload["benchmark_kind"] == "runtime_selection_import_only"
    assert payload["benchmark_source"] == "kaggle_runtime_benchmark"
    assert payload["full_run_eligible"] is False
    assert payload["writes_selected_runtime"] is False
    assert payload["selection_slice"] == "v8_shortlist_eager_amp_then_dual_gate"
    assert manifest["schema_version"] == "spec0001.kaggle_payload_manifest.v1"
    assert _RUNTIME_SELECTION_V8_PAYLOAD_FILES.issubset(entries)
    assert _RUNTIME_SELECTION_V8_PAYLOAD_FILES.issubset(
        _embedded_payload_names(simulation.upload_dir / "run.py"),
    )
    template_text = (
        repo_root / "kaggle" / "kernels" / "runtime_selection" / "run_template.py"
    ).read_text(encoding="utf-8")
    assert '"model_inventory.csv"' in template_text


def test_embedded_runtime_selection_kernel_full_local_fail_closed_simulation(
    tmp_path: Path,
) -> None:
    """Generated runtime-selection launcher validates fail-closed artifacts."""
    repo_root = Path(__file__).resolve().parents[1]
    simulation = _build_upload_simulation(
        tmp_path=tmp_path,
        repo_root=repo_root,
        kernel_name="runtime_selection",
        ready_marker="KAGGLE_RUNTIME_SELECTION_READY = True",
    )
    environment = _run_environment(simulation.output_dir)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["EQVAE_RUNTIME_SELECTION_DATA_ROOT"] = str(
        tmp_path / "missing_data_root",
    )

    subprocess.run(  # noqa: S603
        (sys.executable, str(simulation.upload_dir / "run.py")),
        cwd=simulation.upload_dir,
        check=True,
        env=environment,
    )

    benchmark_dir = simulation.output_dir / "benchmark"
    metrics_dir = simulation.output_dir / "metrics"
    assert {path.name for path in benchmark_dir.iterdir()} == {
        "model_count.json",
        "model_inventory.csv",
        "runtime_proof.json",
        "runtime_matrix.csv",
        "dataloader_matrix.csv",
        "numerical_checks.csv",
        "corruption_checks.csv",
        "gate_health_summary.json",
        "stain_corruptor_qa.json",
    }
    assert {path.name for path in metrics_dir.iterdir()} == {"gate_health.csv"}
    assert not (benchmark_dir / "selected_runtime.json").exists()
    runtime_proof = _load_json(benchmark_dir / "runtime_proof.json")
    runtime_environment = cast(
        "dict[str, object]",
        runtime_proof["runtime_environment"],
    )
    assert runtime_proof["status"] == "fail"
    assert runtime_proof["selection_ready"] is False
    assert runtime_proof["selected_runtime_written"] is False
    assert runtime_environment["failure_kind"] == (
        "runtime_selection_evidence_collection_failed"
    )


def test_embedded_kernel_verify_rejects_stale_template(tmp_path: Path) -> None:
    """Generated run.py must prove freshness against the launcher template."""
    repo_root = Path(__file__).resolve().parents[1]
    source_kernel = repo_root / "kaggle" / "kernels" / "setup_smoke"
    build_script = repo_root / "scripts" / "build_kaggle_embedded_kernel.py"
    generated_kernel = tmp_path / "generated_setup_smoke"
    generated_kernel.mkdir()
    shutil.copy2(source_kernel / "kernel-metadata.json", generated_kernel)
    template_copy = tmp_path / "run_template.py"
    shutil.copy2(source_kernel / "run_template.py", template_copy)

    subprocess.run(  # noqa: S603
        (
            sys.executable,
            str(build_script),
            "--repo-root",
            str(repo_root),
            "--kernel-dir",
            str(generated_kernel),
            "--template",
            str(template_copy),
            "--ready-marker",
            "KAGGLE_SETUP_SMOKE_READY = True",
            "--allow-dirty",
        ),
        cwd=repo_root,
        check=True,
    )

    template_copy.write_text(
        f"{template_copy.read_text(encoding='utf-8')}\n# stale-template-test\n",
        encoding="utf-8",
    )
    completed = subprocess.run(  # noqa: S603
        (
            sys.executable,
            str(build_script),
            "--repo-root",
            str(repo_root),
            "--kernel-dir",
            str(generated_kernel),
            "--template",
            str(template_copy),
            "--ready-marker",
            "KAGGLE_SETUP_SMOKE_READY = True",
            "--verify-only",
            "--allow-dirty",
        ),
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "payload template does not match current run_template.py" in completed.stderr


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


def _embedded_payload_names(run_path: Path) -> set[str]:
    match = _EMBEDDED_PAYLOAD_B64_PATTERN.search(run_path.read_text(encoding="utf-8"))
    if match is None:
        message = "missing embedded payload"
        raise AssertionError(message)
    zip_bytes = base64.b64decode(match.group("payload").encode("ascii"))
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        return set(archive.namelist())
