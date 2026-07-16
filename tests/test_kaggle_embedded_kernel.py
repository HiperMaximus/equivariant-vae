# Copyright 2026 HiperMaximus
"""Tests for generated single-file Kaggle smoke kernels."""

from __future__ import annotations

import base64
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess  # noqa: S404
import sys
import sysconfig
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    import pytest

from eqvae.benchmarking.fixed32_selector_readiness import fixed32_selector_status
from eqvae.benchmarking.kaggle_smoke import (
    SETUP_DATA_KIND,
    SETUP_SMOKE_KIND,
    SETUP_SMOKE_SOURCE,
)
from eqvae.benchmarking.real_data_runtime_pretest import (
    EXPECTED_DATASET_SLUG,
    REAL_DATA_PRETEST_SOURCE,
)
from eqvae.benchmarking.schedule import training_steps_per_epoch
from eqvae.benchmarking.synthetic_timing import (
    MANIFEST_FILENAME,
    MATRIX_FILENAME,
    RECOMMENDATIONS_FILENAME,
    RUNTIME_PROOF_FILENAME,
    SYNTHETIC_TIMING_KIND,
    SYNTHETIC_TIMING_SCOPE,
    SYNTHETIC_TIMING_SOURCE,
)
from eqvae.data.fixed_selectors import (
    FIXED_32_TRAIN_OVERFIT_KIND,
    FixedSelectorGenerationContext,
    generate_fixed_selector_document,
    write_fixed_selector_document,
)
from eqvae.data.patch_shards import PatchShardSpec
from eqvae.data.roots import (
    REAL_TRAIN_PATCH_COUNT,
    TRAIN_BIN_NAME,
    TRAIN_CSV_NAME,
    VALIDATION_BIN_NAME,
    VALIDATION_CSV_NAME,
)
from eqvae.data.synthetic import SyntheticPatchSpec, write_synthetic_patch_shard

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
_RUNTIME_SELECTION_BASELINE_PAYLOAD_FILES = {
    "runs/kaggle/runtime_selection_v5/benchmark/selected_runtime.json",
    "runs/kaggle/runtime_selection_v5/benchmark/runtime_proof.json",
}
_EMBEDDED_PAYLOAD_B64_PATTERN = re.compile(
    r'EMBEDDED_PAYLOAD_B64 = """\n(?P<payload>.*?)\n"""',
    flags=re.DOTALL,
)
_MASKED_HOLDOUT_CSV_PAYLOAD_PATH = "docs/data/ubc_ocean_masked_holdout_ids.csv"
_EXPECTED_SELECTED_RUNTIME_DEBUG_RUNNER_CALLS = 2
_FULL_EPOCHS = 10
_FULL_TARGET_UPDATES = 125000
_FULL_HALF_EPOCH_INTERVAL = 6250
_FULL_CONFIG_PAYLOAD_PATH = "configs/spec0001/non_eq_vae_selected_runtime_full.json"
_SELECTED_RUNTIME_PAYLOAD_PATH = (
    "runs/kaggle/runtime_selection_v5/benchmark/selected_runtime.json"
)
_FULL_PUSH_GUARD_HEREDOC_PATTERN = re.compile(
    r"<<'PYFULLPAYLOAD'\n(?P<body>.*?)\nPYFULLPAYLOAD",
    flags=re.DOTALL,
)
_FULL_TARGET_UPDATES_TOKEN = f"FULL_TARGET_UPDATES = {_FULL_TARGET_UPDATES}"
_FULL_UPDATES_PER_EPOCH = 12500
_BUILD_SCRIPT_MODULE = "build_kaggle_embedded_kernel"
_FULL_RUN_TEMPLATE_MODULE = "selected_runtime_full_run_template"
# A NON-dividing batch (64 does not divide REAL_TRAIN_PATCH_COUNT=300000): floor
# 300000//64 = 4687 differs from ceil 4688, so the derive test genuinely guards floor.
_NON_REFERENCE_GLOBAL_BATCH = 64
_NON_REFERENCE_PER_DEVICE_BATCH = 32
_NON_REFERENCE_UPDATES = 4687
_NON_REFERENCE_TARGET_UPDATES = 46870
_NON_REFERENCE_HALF_EPOCH_INTERVAL = 2343
_OFF_PRODUCT_PER_DEVICE_BATCH = 12


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
        _kernel_argv(simulation),
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
    assert data["origin"] == _expected_data_origin(tmp_path)
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
        _kernel_argv(simulation),
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
        _kernel_argv(simulation),
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


def test_embedded_selected_runtime_compile_probe_kernel_embeds_probe_payload(
    tmp_path: Path,
) -> None:
    """The compile-probe kernel builds no-attach and embeds the probe module.

    The probe itself needs dual-T4 NCCL, so it is not executed here; this proves the
    single-file kernel builds, carries the compiled fast-path probe source, launches
    it under torchrun, and advertises only the non-promotable artifacts.
    """
    repo_root = Path(__file__).resolve().parents[1]
    simulation = _build_upload_simulation(
        tmp_path=tmp_path,
        repo_root=repo_root,
        kernel_name="selected_runtime_compile_probe",
        ready_marker="KAGGLE_SELECTED_RUNTIME_COMPILE_PROBE_READY = True",
    )

    names = _embedded_payload_names(simulation.upload_dir / "run.py")
    assert "src/eqvae/benchmarking/compiled_fastpath_probe.py" in names
    metadata = _load_json(simulation.upload_dir / "kernel-metadata.json")
    assert metadata["id"] == "maximusshtefan/eqvae-selected-runtime-compile-probe"
    for source_field in (
        "dataset_sources",
        "competition_sources",
        "kernel_sources",
        "model_sources",
    ):
        assert metadata[source_field] == []

    run_text = (simulation.upload_dir / "run.py").read_text(encoding="utf-8")
    for required_text in (
        "compiled_fastpath_probe_proof.json",
        "compiled_fastpath_probe_matrix.csv",
        "compiled_fastpath_probe_manifest.json",
        "non_promotable_compiled_fastpath_probe",
        "kaggle_compiled_fastpath_probe",
        "torch.distributed.run",
        "--nproc_per_node=2",
        "eqvae.benchmarking.compiled_fastpath_probe",
    ):
        assert required_text in run_text


def test_compile_probe_push_guard_greps_with_end_of_options_separator() -> None:
    """The compile-probe guard needs `grep -q --` for its dash-prefixed marker.

    Its `required_text` list includes `--nproc_per_node=2`; without the `--`
    end-of-options separator, grep parses that as an unknown option and the guard
    fails every push even though the text is present.
    """
    repo_root = Path(__file__).resolve().parents[1]
    script = (repo_root / "scripts" / "kaggle_kernel.sh").read_text(encoding="utf-8")
    header = "guard_selected_runtime_compile_probe_push_ready()"
    start = script.index(header)
    guard_body = script[start : script.index("\nguard_", start + len(header))]

    assert '"--nproc_per_node=2"' in guard_body
    assert 'grep -q -- "$required_text"' in guard_body
    assert 'grep -q "$required_text"' not in guard_body


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
        _kernel_argv(simulation),
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
        _kernel_argv(simulation),
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
        _kernel_argv(simulation),
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
    assert _RUNTIME_SELECTION_BASELINE_PAYLOAD_FILES.issubset(entries)
    assert _RUNTIME_SELECTION_V8_PAYLOAD_FILES.issubset(
        _embedded_payload_names(simulation.upload_dir / "run.py"),
    )
    assert _RUNTIME_SELECTION_BASELINE_PAYLOAD_FILES.issubset(
        _embedded_payload_names(simulation.upload_dir / "run.py"),
    )
    assert payload["baseline_selected_runtime_exists"] is True
    template_text = (
        repo_root / "kaggle" / "kernels" / "runtime_selection" / "run_template.py"
    ).read_text(encoding="utf-8")
    assert '"model_inventory.csv"' in template_text


def test_embedded_fixed25_selector_kernel_import_simulation(
    tmp_path: Path,
) -> None:
    """Fixed-25 selector kernel imports and carries the selector configs + CLIs."""
    repo_root = Path(__file__).resolve().parents[1]
    simulation = _build_upload_simulation(
        tmp_path=tmp_path,
        repo_root=repo_root,
        kernel_name="fixed25_selector",
        ready_marker="KAGGLE_FIXED25_SELECTOR_READY = True",
    )
    environment = _run_environment(simulation.output_dir)
    environment["EQVAE_FIXED25_SELECTOR_IMPORT_ONLY"] = "1"

    subprocess.run(  # noqa: S603
        _kernel_argv(simulation),
        cwd=simulation.upload_dir,
        check=True,
        env=environment,
    )

    benchmark_dir = simulation.output_dir / "benchmark"
    assert {path.name for path in benchmark_dir.iterdir()} == {
        "fixed25_selector_import.json",
    }
    payload = _load_json(benchmark_dir / "fixed25_selector_import.json")
    manifest = cast("dict[str, object]", payload["payload_manifest"])
    assert payload["status"] == "import_smoke_pass"
    assert payload["status_scope"] == "non_promotable_local_upload_simulation"
    assert payload["full_run_eligible"] is False
    assert payload["writes_selector"] is True
    assert payload["writes_originals"] is True
    assert payload["selector_kind"] == "fixed_25_validation"
    assert payload["config_exists"] is True
    assert manifest["schema_version"] == "spec0001.kaggle_payload_manifest.v1"
    names = _embedded_payload_names(simulation.upload_dir / "run.py")
    assert "configs/spec0001/non_eq_vae_selected_runtime_full.json" in names
    assert "configs/spec0001/fixed_25_validation_patches.json" in names
    assert "src/eqvae/cli/select_fixed_patches.py" in names
    assert "src/eqvae/cli/fixed25_originals.py" in names


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
        _kernel_argv(simulation),
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


def test_embedded_selected_runtime_debug_kernel_import_simulation(
    tmp_path: Path,
) -> None:
    """Selected-runtime debug gate imports and carries the v5 runtime payload."""
    repo_root = Path(__file__).resolve().parents[1]
    simulation = _build_upload_simulation(
        tmp_path=tmp_path,
        repo_root=repo_root,
        kernel_name="selected_runtime_debug",
        ready_marker="KAGGLE_SELECTED_RUNTIME_DEBUG_READY = True",
    )
    environment = _run_environment(simulation.output_dir)
    environment["EQVAE_SELECTED_RUNTIME_DEBUG_IMPORT_ONLY"] = "1"

    subprocess.run(  # noqa: S603
        _kernel_argv(simulation),
        cwd=simulation.upload_dir,
        check=True,
        env=environment,
    )

    benchmark_dir = simulation.output_dir / "benchmark"
    assert {path.name for path in benchmark_dir.iterdir()} == {
        "selected_runtime_debug_import.json",
    }
    assert not (benchmark_dir / "selected_runtime.json").exists()
    payload = _load_json(benchmark_dir / "selected_runtime_debug_import.json")
    manifest = cast("dict[str, object]", payload["payload_manifest"])
    entries = cast("dict[str, object]", manifest["entries"])
    assert payload["status"] == "import_smoke_pass"
    assert payload["status_scope"] == "non_promotable_local_upload_simulation"
    assert payload["benchmark_kind"] == "selected_runtime_debug_import_only"
    assert payload["benchmark_source"] == "kaggle_selected_runtime_debug_kernel"
    assert payload["full_run_eligible"] is False
    assert payload["writes_selected_runtime"] is False
    assert _RUNTIME_SELECTION_BASELINE_PAYLOAD_FILES.issubset(entries)
    assert _RUNTIME_SELECTION_BASELINE_PAYLOAD_FILES.issubset(
        _embedded_payload_names(simulation.upload_dir / "run.py"),
    )
    assert payload["selected_runtime_exists"] is True


def test_embedded_selected_runtime_full_kernel_import_simulation(
    tmp_path: Path,
) -> None:
    """Selected-runtime full launcher imports and targets the full config only."""
    repo_root = Path(__file__).resolve().parents[1]
    simulation = _build_upload_simulation(
        tmp_path=tmp_path,
        repo_root=repo_root,
        kernel_name="selected_runtime_full",
        ready_marker="KAGGLE_SELECTED_RUNTIME_FULL_READY = True",
    )
    environment = _run_environment(simulation.output_dir)
    environment["EQVAE_SELECTED_RUNTIME_FULL_IMPORT_ONLY"] = "1"
    environment["EQVAE_SELECTED_RUNTIME_FULL_OUTPUT_DIR"] = str(simulation.output_dir)
    resume_checkpoint = simulation.output_dir / "checkpoints" / "step_006250.pt"
    environment["EQVAE_SELECTED_RUNTIME_FULL_RESUME"] = str(resume_checkpoint)

    subprocess.run(  # noqa: S603
        _kernel_argv(simulation),
        cwd=simulation.upload_dir,
        check=True,
        env=environment,
    )

    benchmark_dir = simulation.output_dir / "benchmark"
    assert {path.name for path in benchmark_dir.iterdir()} == {
        "selected_runtime_full_import.json",
    }
    assert not (benchmark_dir / "selected_runtime.json").exists()
    payload = _load_json(benchmark_dir / "selected_runtime_full_import.json")
    assert payload["status"] == "import_only_pass"
    assert payload["kernel_id"] == "maximusshtefan/eqvae-selected-runtime-full"
    assert payload["ready_marker"] is True
    assert payload["selected_runtime_full_run_contract_ready"] == (
        "selected_runtime_full_run_contract_ready"
    )
    assert payload["target_optimizer_updates"] == _FULL_TARGET_UPDATES
    assert payload["half_epoch_interval_steps"] == _FULL_HALF_EPOCH_INTERVAL
    command = str(payload["torchrun_command"])
    assert "non_eq_vae_selected_runtime_full.json" in command
    assert "eqvae.cli.selected_runtime_train" in command
    assert "--resume" in command
    assert str(resume_checkpoint.resolve()) in command
    assert "--max-train-steps" not in command
    assert "selected_runtime_debug" not in command
    assert _RUNTIME_SELECTION_BASELINE_PAYLOAD_FILES.issubset(
        _embedded_payload_names(simulation.upload_dir / "run.py"),
    )


def test_selected_runtime_full_push_rejects_preflight_dirty_bypass_env(
    tmp_path: Path,
) -> None:
    """The local dirty bypass cannot be exported into the real push guard."""
    repo_root = Path(__file__).resolve().parents[1]
    simulation = _build_upload_simulation(
        tmp_path=tmp_path,
        repo_root=repo_root,
        kernel_name="selected_runtime_full",
        ready_marker="KAGGLE_SELECTED_RUNTIME_FULL_READY = True",
    )
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_kaggle = fake_bin / "kaggle"
    fake_kaggle.write_text(
        '#!/usr/bin/env bash\necho "fake kaggle $*"\n',
        encoding="utf-8",
    )
    fake_kaggle.chmod(0o755)
    environment = os.environ.copy()
    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
    environment["KAGGLE_PUSH_CONFIRMED"] = "1"
    environment["KAGGLE_FULL_DATASET_CONFIRMED"] = "1"
    environment["EQVAE_SELECTED_RUNTIME_FULL_LOCAL_PREFLIGHT_ALLOW_DIRTY"] = "1"
    bash_path = shutil.which("bash")
    assert bash_path is not None

    completed = subprocess.run(  # noqa: S603
        (
            bash_path,
            str(repo_root / "scripts" / "kaggle_kernel.sh"),
            "push",
            str(simulation.upload_dir),
        ),
        cwd=repo_root,
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )

    assert completed.returncode != 0
    assert "only valid" in completed.stderr
    assert "fake kaggle" not in completed.stdout


def test_full_push_guard_accepts_goal_derived_schedule(tmp_path: Path) -> None:
    """The de-pinned full push guard passes on a freshly built kernel (Spec 0011 S8).

    B1 removed the frozen schedule keys from the full config, so the guard's old
    literal ``expected_training`` checks failed closed on every push. This proves the
    goal-derived guard accepts the current config/plan at the reference batch 24.
    """
    repo_root = Path(__file__).resolve().parents[1]
    simulation = _build_upload_simulation(
        tmp_path=tmp_path,
        repo_root=repo_root,
        kernel_name="selected_runtime_full",
        ready_marker="KAGGLE_SELECTED_RUNTIME_FULL_READY = True",
    )
    guard_py = _extract_full_push_guard_python(repo_root=repo_root, tmp_path=tmp_path)
    result = _run_full_push_guard(
        guard_py=guard_py,
        run_py=simulation.upload_dir / "run.py",
        repo_root=repo_root,
    )
    assert result.returncode == 0, result.stderr


def test_full_push_guard_rejects_refrozen_schedule_key(tmp_path: Path) -> None:
    """Re-freezing a runner-derived schedule key fails the guard closed."""
    repo_root = Path(__file__).resolve().parents[1]
    simulation = _build_upload_simulation(
        tmp_path=tmp_path,
        repo_root=repo_root,
        kernel_name="selected_runtime_full",
        ready_marker="KAGGLE_SELECTED_RUNTIME_FULL_READY = True",
    )
    run_py = simulation.upload_dir / "run.py"
    run_py.write_text(
        _rewrite_embedded_payload(
            run_py.read_text(encoding="utf-8"),
            _refreeze_optimizer_updates,
        ),
        encoding="utf-8",
    )
    guard_py = _extract_full_push_guard_python(repo_root=repo_root, tmp_path=tmp_path)
    result = _run_full_push_guard(
        guard_py=guard_py,
        run_py=run_py,
        repo_root=repo_root,
    )
    assert result.returncode != 0
    assert "must not re-freeze" in result.stderr


def test_full_push_guard_rejects_off_derivation_updates(tmp_path: Path) -> None:
    """A plan recording updates != floor(P / global_batch) fails the guard closed."""
    repo_root = Path(__file__).resolve().parents[1]
    simulation = _build_upload_simulation(
        tmp_path=tmp_path,
        repo_root=repo_root,
        kernel_name="selected_runtime_full",
        ready_marker="KAGGLE_SELECTED_RUNTIME_FULL_READY = True",
    )
    run_py = simulation.upload_dir / "run.py"
    run_py.write_text(
        _rewrite_embedded_payload(
            run_py.read_text(encoding="utf-8"),
            _set_off_derivation_updates,
        ),
        encoding="utf-8",
    )
    guard_py = _extract_full_push_guard_python(repo_root=repo_root, tmp_path=tmp_path)
    result = _run_full_push_guard(
        guard_py=guard_py,
        run_py=run_py,
        repo_root=repo_root,
    )
    assert result.returncode != 0
    assert "optimizer_updates_per_epoch must be" in result.stderr


def test_full_push_guard_rejects_non_integer_epochs(tmp_path: Path) -> None:
    """A float ``training.epochs`` (10.0) fails closed, not nulling the derivation.

    Regression for an S8 fail-open: a JSON float epochs passed the ``!= 10`` anchor
    pin yet made the derived FULL_TARGET_UPDATES/FULL_HALF_EPOCH_INTERVAL token check
    silently skip, so a drifted run.py token could slip through with exit 0.
    """
    repo_root = Path(__file__).resolve().parents[1]
    simulation = _build_upload_simulation(
        tmp_path=tmp_path,
        repo_root=repo_root,
        kernel_name="selected_runtime_full",
        ready_marker="KAGGLE_SELECTED_RUNTIME_FULL_READY = True",
    )
    run_py = simulation.upload_dir / "run.py"
    tampered = _rewrite_embedded_payload(
        run_py.read_text(encoding="utf-8"),
        _set_float_epochs,
    ).replace(_FULL_TARGET_UPDATES_TOKEN, "FULL_TARGET_UPDATES = 999999", 1)
    run_py.write_text(tampered, encoding="utf-8")
    guard_py = _extract_full_push_guard_python(repo_root=repo_root, tmp_path=tmp_path)
    result = _run_full_push_guard(
        guard_py=guard_py,
        run_py=run_py,
        repo_root=repo_root,
    )
    assert result.returncode != 0
    assert "training.epochs must be an integer" in result.stderr


def test_full_push_guard_rejects_off_derivation_run_py_token(tmp_path: Path) -> None:
    """A run.py whose FULL_TARGET_UPDATES != epochs * derived updates fails closed."""
    repo_root = Path(__file__).resolve().parents[1]
    simulation = _build_upload_simulation(
        tmp_path=tmp_path,
        repo_root=repo_root,
        kernel_name="selected_runtime_full",
        ready_marker="KAGGLE_SELECTED_RUNTIME_FULL_READY = True",
    )
    run_py = simulation.upload_dir / "run.py"
    run_text = run_py.read_text(encoding="utf-8")
    mutated = run_text.replace(
        _FULL_TARGET_UPDATES_TOKEN,
        "FULL_TARGET_UPDATES = 999999",
        1,
    )
    assert mutated != run_text
    run_py.write_text(mutated, encoding="utf-8")
    guard_py = _extract_full_push_guard_python(repo_root=repo_root, tmp_path=tmp_path)
    result = _run_full_push_guard(
        guard_py=guard_py,
        run_py=run_py,
        repo_root=repo_root,
    )
    assert result.returncode != 0
    assert "missing required text" in result.stderr


def test_build_derives_reference_full_schedule() -> None:
    """The builder derives the batch-24 schedule from the plan via the single source."""
    repo_root = Path(__file__).resolve().parents[1]
    build_module = _load_script_module(
        _BUILD_SCRIPT_MODULE,
        repo_root / "scripts" / "build_kaggle_embedded_kernel.py",
    )
    derive = cast(
        "Callable[[Path], tuple[int, int, int]]",
        build_module.__dict__["_derive_full_schedule"],
    )
    assert derive(repo_root) == (
        _FULL_UPDATES_PER_EPOCH,
        _FULL_TARGET_UPDATES,
        _FULL_HALF_EPOCH_INTERVAL,
    )


def test_build_derives_non_reference_full_schedule(tmp_path: Path) -> None:
    """_derive_full_schedule floors a non-24 plan end-to-end from the single source."""
    repo_root = Path(__file__).resolve().parents[1]
    build_module = _load_script_module(
        _BUILD_SCRIPT_MODULE,
        repo_root / "scripts" / "build_kaggle_embedded_kernel.py",
    )
    derive = cast(
        "Callable[[Path], tuple[int, int, int]]",
        build_module.__dict__["_derive_full_schedule"],
    )
    fake_repo = tmp_path / "repo"
    (fake_repo / "src").mkdir(parents=True)
    (fake_repo / "src" / "eqvae").symlink_to(repo_root / "src" / "eqvae")
    plan_path = (
        fake_repo
        / "runs"
        / "kaggle"
        / "runtime_selection_v5"
        / "benchmark"
        / "selected_runtime.json"
    )
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text(
        json.dumps({"global_batch_size": _NON_REFERENCE_GLOBAL_BATCH}),
        encoding="utf-8",
    )
    config_path = (
        fake_repo / "configs" / "spec0001" / "non_eq_vae_selected_runtime_full.json"
    )
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps({"training": {"epochs": _FULL_EPOCHS}}),
        encoding="utf-8",
    )
    assert derive(fake_repo) == (
        _NON_REFERENCE_UPDATES,
        _NON_REFERENCE_TARGET_UPDATES,
        _NON_REFERENCE_HALF_EPOCH_INTERVAL,
    )


def test_eqvae_never_imports_the_leaked_top_level_nn_package() -> None:
    """`src/eqvae` must never import `nn`: it resolves locally but is absent on Kaggle.

    The editable install's .pth puts the whole `<repo>/src` on sys.path, so
    `import nn` works in the venv -- but the payload ships only `src/eqvae`, and
    `src/nn` is excluded from ruff AND basedpyright, so nothing else would catch
    such an import. It would pass every local check, then raise
    ModuleNotFoundError on Kaggle after the GPU slot was committed. (`src/nn` is
    also dead: nothing imports it, and it needs pytorch-msssim, which commit
    ff54009 dropped from the dependencies.)
    """
    repo_root = Path(__file__).resolve().parents[1]
    pattern = re.compile(r"^\s*(?:import\s+nn\b|from\s+nn[\s.])")
    offenders = [
        f"{path.relative_to(repo_root)}:{number}"
        for path in sorted((repo_root / "src" / "eqvae").rglob("*.py"))
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        )
        if pattern.match(line)
    ]
    assert not offenders, (
        f"src/eqvae imports the top-level `nn` package, which the Kaggle payload "
        f"does not ship: {offenders}"
    )


def test_build_substitutes_non_reference_full_schedule() -> None:
    """A non-24 batch rewrites the schedule constants; a missing one fails closed."""
    repo_root = Path(__file__).resolve().parents[1]
    build_module = _load_script_module(
        _BUILD_SCRIPT_MODULE,
        repo_root / "scripts" / "build_kaggle_embedded_kernel.py",
    )
    apply_substitution = cast(
        "Callable[..., str]",
        build_module.__dict__["_apply_full_schedule_substitution"],
    )
    original = (
        f"FULL_TARGET_UPDATES = {_FULL_TARGET_UPDATES}\n"
        f"FULL_HALF_EPOCH_INTERVAL = {_FULL_HALF_EPOCH_INTERVAL}\n"
    )
    rewritten = apply_substitution(
        original,
        target=_NON_REFERENCE_TARGET_UPDATES,
        half=_NON_REFERENCE_HALF_EPOCH_INTERVAL,
    )
    assert f"FULL_TARGET_UPDATES = {_NON_REFERENCE_TARGET_UPDATES}" in rewritten
    assert (
        f"FULL_HALF_EPOCH_INTERVAL = {_NON_REFERENCE_HALF_EPOCH_INTERVAL}" in rewritten
    )
    raised = False
    try:
        apply_substitution("no schedule constants here", target=1, half=1)
    except RuntimeError:
        raised = True
    assert raised


def test_full_run_template_validator_accepts_non_reference_batch(
    tmp_path: Path,
) -> None:
    """The de-pinned run.py validator accepts a measured non-24 plan by relationship."""
    repo_root = Path(__file__).resolve().parents[1]
    run_template = _load_script_module(
        _FULL_RUN_TEMPLATE_MODULE,
        repo_root / "kaggle" / "kernels" / "selected_runtime_full" / "run_template.py",
    )
    validate = cast(
        "Callable[[Path], None]",
        run_template.__dict__["_validate_baseline_selected_runtime"],
    )
    plan_path = _write_baseline_plan(
        tmp_path / "plan.json",
        run_template,
        per_device_batch_size=_NON_REFERENCE_PER_DEVICE_BATCH,
        global_batch_size=_NON_REFERENCE_GLOBAL_BATCH,
        optimizer_updates_per_epoch=_NON_REFERENCE_UPDATES,
    )
    validate(plan_path)


def test_full_run_template_validator_rejects_off_relationship(
    tmp_path: Path,
) -> None:
    """A non-product batch or off-derivation updates fails the validator closed."""
    repo_root = Path(__file__).resolve().parents[1]
    run_template = _load_script_module(
        _FULL_RUN_TEMPLATE_MODULE,
        repo_root / "kaggle" / "kernels" / "selected_runtime_full" / "run_template.py",
    )
    validate = cast(
        "Callable[[Path], None]",
        run_template.__dict__["_validate_baseline_selected_runtime"],
    )
    off_product = _write_baseline_plan(
        tmp_path / "off_product.json",
        run_template,
        per_device_batch_size=_OFF_PRODUCT_PER_DEVICE_BATCH,
        global_batch_size=_NON_REFERENCE_GLOBAL_BATCH,
        optimizer_updates_per_epoch=_NON_REFERENCE_UPDATES,
    )
    off_updates = _write_baseline_plan(
        tmp_path / "off_updates.json",
        run_template,
        per_device_batch_size=_NON_REFERENCE_PER_DEVICE_BATCH,
        global_batch_size=_NON_REFERENCE_GLOBAL_BATCH,
        optimizer_updates_per_epoch=_NON_REFERENCE_UPDATES + 1,
    )
    for bad_plan, expected in (
        (off_product, "must equal per_device_batch_size * world_size"),
        (off_updates, "optimizer_updates_per_epoch mismatch"),
    ):
        raised = ""
        try:
            validate(bad_plan)
        except RuntimeError as error:
            raised = str(error)
        assert expected in raised


def test_embedded_selected_runtime_debug_kernel_full_local_fail_closed_simulation(
    tmp_path: Path,
) -> None:
    """Generated selected-runtime debug launcher validates fail-closed artifacts."""
    repo_root = Path(__file__).resolve().parents[1]
    simulation = _build_upload_simulation(
        tmp_path=tmp_path,
        repo_root=repo_root,
        kernel_name="selected_runtime_debug",
        ready_marker="KAGGLE_SELECTED_RUNTIME_DEBUG_READY = True",
    )

    subprocess.run(  # noqa: S603
        _kernel_argv(simulation),
        cwd=simulation.upload_dir,
        check=True,
        env=_run_environment(simulation.output_dir),
    )

    benchmark_dir = simulation.output_dir / "benchmark"
    metrics_dir = simulation.output_dir / "metrics"
    assert {path.name for path in benchmark_dir.iterdir()} == {
        "artifact_manifest.json",
        "checkpoint_resume_proof.json",
        "fixed32_selector_readiness.json",
        "gate_health_summary.json",
        "local_selected_runtime_readiness.json",
        "selected_runtime_plan_applied.json",
        "selected_runtime_debug_summary.json",
        "selected_runtime_gate_summary.json",
        "tiny_overfit_summary.json",
        "training_summary.json",
    }
    assert {path.name for path in metrics_dir.iterdir()} == {
        "gate_health.csv",
        "train_metrics.csv",
    }
    assert not (benchmark_dir / "selected_runtime.json").exists()
    summary = _load_json(benchmark_dir / "selected_runtime_gate_summary.json")
    blockers = cast("list[str]", summary["launch_blockers_remaining"])
    component_status = cast("dict[str, object]", summary["component_status"])
    assert summary["status"] == "fail"
    assert summary["benchmark_kind"] == "kaggle_selected_runtime_debug_resume_tiny_gate"
    assert summary["benchmark_source"] == "kaggle_selected_runtime_debug_kernel"
    assert summary["full_run_eligible"] is False
    assert component_status["selected_runtime_transport"] == "pass"
    selector_readiness = _load_json(benchmark_dir / "fixed32_selector_readiness.json")
    assert selector_readiness["status"] == "fail"
    assert selector_readiness["selector_generation_mode"] == "remote_generate"
    assert "fixed_32_selector_placeholder" in blockers


def test_selected_runtime_debug_remote_selector_requires_canonical_status(
    tmp_path: Path,
) -> None:
    """A zero-exit selector CLI is not enough without canonical-real status."""
    from kaggle.kernels.selected_runtime_debug import run_template  # noqa: PLC0415

    payload_dir = tmp_path / "payload"
    output_dir = tmp_path / "output"
    selector_path = output_dir / "benchmark" / "fixed_32_train_overfit_patches.json"
    calls: list[tuple[str, ...]] = []
    payload_dir.mkdir(parents=True)
    original_cwd = Path.cwd()

    def fake_select_fixed_patches(args: object) -> int:
        values = _arg_tuple(args)
        calls.append(values)
        output_value = _option_value(values, "--output")
        output_path = Path(output_value)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("{}\n", encoding="utf-8")
        return 0

    def fake_status(path: Path, *, data_root: str | None) -> dict[str, object]:
        assert Path.cwd() == payload_dir
        assert path == selector_path
        assert data_root == "auto"
        return {
            "status": "fail",
            "canonical_real_ubc": False,
            "failure_kind": "fixed_32_selector_not_canonical_real_ubc",
        }

    result = run_template._generate_remote_fixed32_selector(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        select_fixed_patches_main=fake_select_fixed_patches,
        fixed32_selector_status=fake_status,
        payload_dir=payload_dir,
        output_dir=output_dir,
        data_root="auto",
        selector_path=selector_path,
    )

    artifact = _load_json(output_dir / "benchmark" / "fixed32_selector_readiness.json")
    assert result["status"] == "fail"
    assert result["fixed_32_selector_real"] is False
    assert artifact["status"] == "fail"
    assert artifact["remote_selector_generation_ready"] is False
    assert artifact["failure_kind"] == "fixed_32_selector_not_canonical_real_ubc"
    assert calls
    assert _option_value(calls[0], "--kind") == "fixed_32_train_overfit"
    assert "--validate-crc" in calls[0]
    assert Path.cwd() == original_cwd


def test_selected_runtime_debug_selector_status_resolves_payload_holdout(
    tmp_path: Path,
) -> None:
    """Selector validation uses the embedded payload for relative holdout paths."""
    from kaggle.kernels.selected_runtime_debug import run_template  # noqa: PLC0415

    payload_dir = tmp_path / "payload"
    holdout_csv = payload_dir / _MASKED_HOLDOUT_CSV_PAYLOAD_PATH
    holdout_csv.parent.mkdir(parents=True)
    holdout_csv.write_text("image_id\nnot_in_synthetic\n", encoding="utf-8")

    data_root = tmp_path / "data"
    dataset_dir = data_root / "dataset"
    write_synthetic_patch_shard(
        bin_path=dataset_dir / TRAIN_BIN_NAME,
        csv_path=dataset_dir / TRAIN_CSV_NAME,
        spec=SyntheticPatchSpec(count=40, image_size=8, channels=3),
        include_idx=False,
    )
    write_synthetic_patch_shard(
        bin_path=dataset_dir / VALIDATION_BIN_NAME,
        csv_path=dataset_dir / VALIDATION_CSV_NAME,
        spec=SyntheticPatchSpec(count=25, image_size=8, channels=3, seed=20260613),
        include_idx=True,
    )
    train_spec = PatchShardSpec(
        bin_path=dataset_dir / TRAIN_BIN_NAME,
        csv_path=dataset_dir / TRAIN_CSV_NAME,
        image_size=8,
        channels=3,
        validate_crc=True,
    )
    document = generate_fixed_selector_document(
        selector_kind=FIXED_32_TRAIN_OVERFIT_KIND,
        shard_spec=train_spec,
        source_split="train",
        context=FixedSelectorGenerationContext(
            dataset_slug=EXPECTED_DATASET_SLUG,
            data_root=dataset_dir,
            masked_holdout_wsi_ids=frozenset({"not_in_synthetic"}),
        ),
    )
    selector_path = tmp_path / "working" / "benchmark" / "fixed_32.json"
    write_fixed_selector_document(path=selector_path, document=document)

    original_cwd = Path.cwd()
    fake_working = tmp_path / "working"

    def real_status(path: Path, *, data_root: str | None) -> dict[str, object]:
        return dict(fixed32_selector_status(path, data_root=data_root))

    try:
        os.chdir(fake_working)
        direct_status = fixed32_selector_status(
            selector_path,
            data_root=str(data_root),
        )
        wrapped_status = run_template._fixed32_selector_status_from_payload_cwd(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
            fixed32_selector_status=real_status,
            payload_dir=payload_dir,
            selector_path=selector_path,
            data_root=str(data_root),
        )
    finally:
        os.chdir(original_cwd)

    direct_failure = direct_status["failure_kind"]
    direct_has_holdout = any(
        (parent / _MASKED_HOLDOUT_CSV_PAYLOAD_PATH).exists()
        for parent in selector_path.resolve().parents
    )
    if direct_has_holdout:
        assert direct_failure == "fixed_32_selector_not_canonical_real_ubc"
    else:
        assert direct_failure == "fixed_32_selector_masked_holdout_unavailable"
    assert wrapped_status["failure_kind"] == "fixed_32_selector_not_canonical_real_ubc"
    validation_errors = cast("list[object]", wrapped_status["validation_errors"])
    assert "fixed_32_selector_masked_holdout_unavailable" not in validation_errors


def test_selected_runtime_debug_real_runner_uses_resume_sequence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The remote-pass branch launches 4 updates, then resumes to update 8."""
    from kaggle.kernels.selected_runtime_debug import run_template  # noqa: PLC0415

    calls: list[tuple[Path, tuple[str, ...]]] = []
    payload_dir = tmp_path / "payload"
    payload_src = payload_dir / "src"
    output_dir = tmp_path / "output"
    selected_runtime_path = tmp_path / "selected_runtime.json"
    fixed_train_patches = tmp_path / "fixed_32_train_overfit_patches.json"

    def fake_selected_runtime_train(*, payload_src: Path, args: object) -> int:
        values = _arg_tuple(args)
        calls.append((payload_src, values))
        phase_output = Path(_option_value(values, "--output-dir"))
        if len(calls) == 1:
            checkpoint = phase_output / "checkpoints" / "step_000004.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(b"checkpoint")
        return 0

    monkeypatch.setattr(
        run_template,
        "_run_selected_runtime_train_torchrun",
        fake_selected_runtime_train,
    )

    exit_code = run_template._run_real_selected_runtime_debug(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        payload_src=payload_src,
        payload_dir=payload_dir,
        output_dir=output_dir,
        selected_runtime_path=selected_runtime_path,
        data_root="auto",
        fixed_train_patches=fixed_train_patches,
    )

    assert exit_code == 0
    assert len(calls) == _EXPECTED_SELECTED_RUNTIME_DEBUG_RUNNER_CALLS
    assert {call_payload_src for call_payload_src, _ in calls} == {payload_src}
    phase1 = calls[0][1]
    phase2 = calls[1][1]
    assert "--resume" not in phase1
    assert _option_value(phase1, "--output-dir") == str(
        output_dir / "resume_probe_phase1",
    )
    assert _option_value(phase1, "--max-train-steps") == "4"
    assert _option_value(phase1, "--save-every-steps") == "4"
    assert _option_value(phase1, "--data") == "ubc-pre-shuffled"
    assert _option_value(phase1, "--fixed-train-patches") == str(fixed_train_patches)
    assert _option_value(phase2, "--output-dir") == str(output_dir)
    assert _option_value(phase2, "--resume") == str(
        output_dir / "resume_probe_phase1" / "checkpoints" / "step_000004.pt",
    )
    assert _option_value(phase2, "--max-train-steps") == "8"
    assert _option_value(phase2, "--save-every-steps") == "4"
    assert _option_value(phase2, "--data") == "ubc-pre-shuffled"
    assert _option_value(phase2, "--fixed-train-patches") == str(fixed_train_patches)


def test_selected_runtime_debug_tiny_runner_uses_generated_selector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tiny phase is capped and consumes the generated fixed-32 selector."""
    from kaggle.kernels.selected_runtime_debug import run_template  # noqa: PLC0415

    calls: list[tuple[Path, tuple[str, ...]]] = []
    payload_dir = tmp_path / "payload"
    payload_src = payload_dir / "src"
    output_dir = tmp_path / "output"
    selected_runtime_path = tmp_path / "selected_runtime.json"
    fixed_train_patches = output_dir / "benchmark" / "fixed_32_train_overfit.json"

    def fake_selected_runtime_train(*, payload_src: Path, args: object) -> int:
        values = _arg_tuple(args)
        calls.append((payload_src, values))
        tiny_output = Path(_option_value(values, "--output-dir"))
        summary = tiny_output / "benchmark" / "tiny_overfit_summary.json"
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(
            json.dumps(
                {
                    "status": "local_pass",
                    "patch_count": 32,
                    "optimizer_steps": 128,
                    "successful_metric_row_count": 256,
                    "amp_step_skipped_count": 0,
                    "nonfinite_count": 0,
                    "grad_scaler_init_scale": run_template.AMP_GRAD_SCALER_INIT_SCALE,
                    "train_sampler_policy": "fixed32_tiny_full_batch_repeated",
                    "train_effective_global_epoch_samples": 48,
                    "train_effective_per_rank_epoch_samples": 24,
                    "fixed_train_repeated_to_full_batch": True,
                    "observed_batch_sizes": [12],
                    "l1_improvement_fraction": 0.02,
                    "recon_loss_improvement_fraction": 0.02,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(
        run_template,
        "_run_selected_runtime_train_torchrun",
        fake_selected_runtime_train,
    )

    exit_code = run_template._run_real_selected_runtime_tiny_overfit(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        payload_src=payload_src,
        payload_dir=payload_dir,
        output_dir=output_dir,
        selected_runtime_path=selected_runtime_path,
        data_root="auto",
        fixed_train_patches=fixed_train_patches,
    )

    assert exit_code == 0
    assert len(calls) == 1
    assert calls[0][0] == payload_src
    call = calls[0][1]
    assert _option_value(call, "--config") == str(
        payload_dir / "configs/spec0001/non_eq_vae_kaggle_tiny_overfit.json",
    )
    assert _option_value(call, "--data") == "ubc-pre-shuffled"
    assert _option_value(call, "--fixed-train-patches") == str(fixed_train_patches)
    assert _option_value(call, "--max-train-steps") == "128"
    assert _option_value(call, "--save-every-steps") == "64"
    copied = _load_json(output_dir / "benchmark" / "tiny_overfit_summary.json")
    assert copied["status"] == "local_pass"
    assert copied["grad_scaler_init_scale"] == run_template.AMP_GRAD_SCALER_INIT_SCALE
    assert "source_summary_sha256" in copied


def test_selected_runtime_debug_runner_launches_torch_distributed() -> None:
    """The remote runner command uses torch distributed with two local ranks."""
    from kaggle.kernels.selected_runtime_debug import run_template  # noqa: PLC0415

    command = run_template._selected_runtime_train_torchrun_command(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        ("--config", "debug.json"),
    )

    assert command[:7] == (
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        "-m",
        "eqvae.cli.selected_runtime_train",
    )
    assert command[7:] == ("--config", "debug.json")


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
    """Return a subprocess env where `eqvae` resolves ONLY from the unzipped payload.

    Popping PYTHONPATH used to be enough. It no longer is: `eqvae` is now
    editable-installed into the venv, and the resulting `.pth` puts `<repo>/src`
    on sys.path for every venv process regardless of PYTHONPATH. That silently
    defeated this simulation -- a payload MISSING a module would still import it
    from the venv and pass here, then die with ModuleNotFoundError on Kaggle
    after the GPU slot was committed. (Modules PRESENT in the payload were never
    at risk: run_template inserts the payload at sys.path[0].)

    So the interpreter runs with `-S` (see `_kernel_argv`), which skips site.py
    and hence all .pth processing, and site-packages is re-added explicitly here.
    Net effect: stdlib + installed third-party (torch, numpy) stay importable
    while `eqvae` does not -- exactly the Kaggle contract this simulation proves.

    LIMIT -- the guarantee is PARENT-ONLY. `-S` is a command-line flag, not an env
    var, so it does not survive the `subprocess.run([sys.executable, ...])` calls
    the payload itself makes (`run_template.py` spawns `torch.distributed.run` and
    the output gate). Those children re-run site.py and get `<repo>/src` back, so a
    lazily-imported leaked top-level name (e.g. `nn`) on a torchrun-only code path
    would still pass here. `test_eqvae_never_imports_the_leaked_top_level_nn_package`
    is the load-bearing guard for that case precisely because it greps rather than
    imports. Do not "fix" this by forcing -S into the payload's own subprocess
    calls: that is Kaggle production code, and Kaggle has no .pth to defend against.

    Returns:
        Subprocess environment with payload-only `eqvae` resolution.

    """
    environment = os.environ.copy()
    environment["EQVAE_OUTPUT_DIR"] = str(output_dir)
    environment.pop("EQVAE_DATA_ROOT", None)
    environment["PYTHONPATH"] = sysconfig.get_paths()["purelib"]
    return environment


def _kernel_argv(simulation: UploadSimulation) -> tuple[str, ...]:
    """Return the argv running the uploaded `run.py` under payload-only isolation.

    `-S` is what makes `_run_environment`'s isolation real; keep the two together.

    Returns:
        Argv for the single-file kernel subprocess.

    """
    return (sys.executable, "-S", str(simulation.upload_dir / "run.py"))


def _expected_data_origin(path: Path) -> str:
    if path.parts[:2] == ("/", "tmp"):
        return "synthetic_or_ephemeral_path"
    return "local_or_explicit_path"


def _load_json(path: Path) -> dict[str, object]:
    return cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))


def _arg_tuple(args: object) -> tuple[str, ...]:
    if not isinstance(args, tuple):
        message = "expected tuple arguments"
        raise TypeError(message)
    items = cast("tuple[object, ...]", args)
    if not all(isinstance(item, str) for item in items):
        message = "expected string arguments"
        raise TypeError(message)
    return cast("tuple[str, ...]", items)


def _option_value(args: tuple[str, ...], option: str) -> str:
    try:
        index = args.index(option)
    except ValueError as error:
        message = f"missing option: {option}"
        raise AssertionError(message) from error
    try:
        return args[index + 1]
    except IndexError as error:
        message = f"missing value for option: {option}"
        raise AssertionError(message) from error


def _embedded_payload_names(run_path: Path) -> set[str]:
    match = _EMBEDDED_PAYLOAD_B64_PATTERN.search(run_path.read_text(encoding="utf-8"))
    if match is None:
        message = "missing embedded payload"
        raise AssertionError(message)
    zip_bytes = base64.b64decode(match.group("payload").encode("ascii"))
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        return set(archive.namelist())


def _extract_full_push_guard_python(*, repo_root: Path, tmp_path: Path) -> Path:
    # Run the real PYFULLPAYLOAD guard body verbatim so the test cannot drift from the
    # shipped shell validator (Spec 0011 S8 de-pinned it to goal-derived relationships).
    script = (repo_root / "scripts" / "kaggle_kernel.sh").read_text(encoding="utf-8")
    match = _FULL_PUSH_GUARD_HEREDOC_PATTERN.search(script)
    if match is None:
        message = "missing PYFULLPAYLOAD guard heredoc in kaggle_kernel.sh"
        raise AssertionError(message)
    guard_py = tmp_path / "full_push_guard.py"
    guard_py.write_text(match.group("body"), encoding="utf-8")
    return guard_py


def _run_full_push_guard(
    *,
    guard_py: Path,
    run_py: Path,
    repo_root: Path,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = "src"
    return subprocess.run(  # noqa: S603
        (sys.executable, str(guard_py), str(run_py)),
        cwd=repo_root,
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )


def _rewrite_embedded_payload(
    run_text: str,
    mutate: Callable[[dict[str, bytes]], None],
) -> str:
    match = _EMBEDDED_PAYLOAD_B64_PATTERN.search(run_text)
    if match is None:
        message = "missing embedded payload"
        raise AssertionError(message)
    zip_bytes = base64.b64decode(match.group("payload").encode("ascii"))
    members: dict[str, bytes] = {}
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        for name in archive.namelist():
            members[name] = archive.read(name)
    mutate(members)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return (
        run_text[: match.start("payload")] + encoded + run_text[match.end("payload") :]
    )


def _refreeze_optimizer_updates(members: dict[str, bytes]) -> None:
    config = cast("dict[str, object]", json.loads(members[_FULL_CONFIG_PAYLOAD_PATH]))
    training = cast("dict[str, object]", config["training"])
    training["optimizer_updates_per_epoch"] = _FULL_TARGET_UPDATES
    members[_FULL_CONFIG_PAYLOAD_PATH] = json.dumps(config).encode("utf-8")


def _set_float_epochs(members: dict[str, bytes]) -> None:
    config = cast("dict[str, object]", json.loads(members[_FULL_CONFIG_PAYLOAD_PATH]))
    training = cast("dict[str, object]", config["training"])
    training["epochs"] = float(_FULL_EPOCHS)
    members[_FULL_CONFIG_PAYLOAD_PATH] = json.dumps(config).encode("utf-8")


def _load_script_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        message = f"cannot load module {name}"
        raise AssertionError(message)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_baseline_plan(
    path: Path,
    run_template: ModuleType,
    *,
    per_device_batch_size: int,
    global_batch_size: int,
    optimizer_updates_per_epoch: int,
) -> Path:
    plan = {
        "status": "pass",
        "selected_row_id": cast(
            "str",
            run_template.__dict__["EXPECTED_SELECTED_ROW_ID"],
        ),
        "runtime_policy_id": cast(
            "str",
            run_template.__dict__["EXPECTED_RUNTIME_POLICY_ID"],
        ),
        "world_size": 2,
        "nproc_per_node": 2,
        "per_device_batch_size": per_device_batch_size,
        "global_batch_size": global_batch_size,
        "optimizer_updates_per_epoch": optimizer_updates_per_epoch,
        "full_run_eligible": True,
        "mixed_precision": {"policy": "amp_conservative"},
    }
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


def _set_off_derivation_updates(members: dict[str, bytes]) -> None:
    plan = cast(
        "dict[str, object]",
        json.loads(members[_SELECTED_RUNTIME_PAYLOAD_PATH]),
    )
    global_batch = plan["global_batch_size"]
    if not isinstance(global_batch, int):
        message = "selected runtime global_batch_size must be an integer"
        raise TypeError(message)
    plan["optimizer_updates_per_epoch"] = (
        training_steps_per_epoch(
            real_train_patch_count=REAL_TRAIN_PATCH_COUNT,
            global_batch_size=global_batch,
        )
        + 1
    )
    members[_SELECTED_RUNTIME_PAYLOAD_PATH] = json.dumps(plan).encode("utf-8")
