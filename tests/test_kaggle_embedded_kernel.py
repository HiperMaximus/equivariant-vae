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
_FULL_TARGET_UPDATES = 60000
_FULL_HALF_EPOCH_INTERVAL = 3000
_FULL_CONFIG_PAYLOAD_PATH = "configs/spec0001/non_eq_vae_selected_runtime_full.json"
_SELECTED_RUNTIME_PAYLOAD_PATH = "configs/spec0001/non_eq_vae_selected_runtime.json"
_FULL_PUSH_GUARD_HEREDOC_PATTERN = re.compile(
    r"<<'PYFULLPAYLOAD'\n(?P<body>.*?)\nPYFULLPAYLOAD",
    flags=re.DOTALL,
)
_DEBUG_PUSH_GUARD_HEREDOC_PATTERN = re.compile(
    r"<<'PYDEBUGPAYLOAD'\n(?P<body>.*?)\nPYDEBUGPAYLOAD",
    flags=re.DOTALL,
)
_FULL_TARGET_UPDATES_TOKEN = f"FULL_TARGET_UPDATES = {_FULL_TARGET_UPDATES}"
_FULL_UPDATES_PER_EPOCH = 6000
_BUILD_SCRIPT_MODULE = "build_kaggle_embedded_kernel"
# A NON-dividing batch (64 does not divide REAL_TRAIN_PATCH_COUNT=300000): floor
# 300000//64 = 4687 differs from ceil 4688, so the derive test genuinely guards floor.
_NON_REFERENCE_GLOBAL_BATCH = 64
_NON_REFERENCE_UPDATES = 4687
_NON_REFERENCE_TARGET_UPDATES = 46870
_NON_REFERENCE_HALF_EPOCH_INTERVAL = 2343


@dataclass(frozen=True)
class UploadSimulation:
    """Paths for a simulated Kaggle single-file upload."""

    upload_dir: Path
    output_dir: Path


def test_embedded_setup_kernel_survives_single_file_upload_simulation(
    tmp_path: Path,
) -> None:
    """The embedded setup smoke proves bounded corruption after single-file upload.

    Upload survival and nonzero changed inputs are deliberate requirements; corruption
    totals are derived from bounded per-step counts rather than a frozen RNG draw. This
    catches payload/import breakage or false aggregation without retaining blake2b.
    """
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
    limits = cast("dict[str, object]", payload["limits"])
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
    applied_counts = cast("list[int]", train["applied_counts"])
    steps_completed = cast("int", train["steps_completed"])
    batch_size = cast("int", limits["batch_size"])
    total_applied_count = cast("int", train["total_applied_count"])
    assert len(applied_counts) == steps_completed
    assert all(0 <= count <= batch_size for count in applied_counts)
    assert total_applied_count == sum(applied_counts)
    assert 0 < total_applied_count <= steps_completed * batch_size
    assert max(cast("list[float]", train["input_target_delta_maxes"])) > 0.0
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
    """Import-only fixed-25 execution records payload and runtime-stack evidence.

    The resolved Torch/CUDA artifact is measured producer evidence required for stack
    parity, not a self-attested flag. These assertions catch removing its writer or
    returning from the import-only path before the upgraded runtime is recorded.
    """
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
        "fixed25_selector_runtime_environment.json",
    }
    runtime_environment = _load_json(
        benchmark_dir / "fixed25_selector_runtime_environment.json",
    )
    assert runtime_environment["status"] == "pass"
    assert (
        runtime_environment["benchmark_kind"] == "fixed25_selector_runtime_environment"
    )
    assert isinstance(runtime_environment["torch_version"], str)
    assert runtime_environment["torch_version"]
    assert isinstance(runtime_environment["cuda_available"], bool)
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
    """A dataset-free local launch must fail without fabricating a winner.

    This executes the generated single-file launcher rather than inspecting source text.
    A missing data root must still write diagnostic artifacts and omit
    ``selected_runtime.json``; returning success-ready evidence catches a fail-open
    kernel.
    """
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
    resume_checkpoint = simulation.output_dir / "checkpoints" / "step_003000.pt"
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


def test_embedded_selected_runtime_lr_range_kernel_is_self_contained(
    tmp_path: Path,
) -> None:
    """The LR kernel ships the bounded runner/plan without retired search machinery."""
    repo_root = Path(__file__).resolve().parents[1]
    simulation = _build_upload_simulation(
        tmp_path=tmp_path,
        repo_root=repo_root,
        kernel_name="selected_runtime_lr_range",
        ready_marker="KAGGLE_SELECTED_RUNTIME_LR_RANGE_READY = True",
    )
    run_path = simulation.upload_dir / "run.py"
    run_text = run_path.read_text(encoding="utf-8")
    names = _embedded_payload_names(run_path)

    assert "--nproc_per_node=2" in run_text
    assert "EXPECTED_UPDATES = 192" in run_text
    assert "EXPECTED_START_LR = 2e-5" in run_text
    assert "EXPECTED_END_LR = 3e-3" in run_text
    assert "configs/spec0001/non_eq_vae_selected_runtime.json" in names
    assert "configs/spec0001/non_eq_vae_runtime_winner.json" in names
    assert "configs/spec0001/non_eq_vae_selected_runtime_lr_range.json" in names
    assert "src/eqvae/training/selected_runtime_runner.py" in names
    assert not any("runtime_recipe_bakeoff" in name for name in names)
    assert not any(name in names for name in _RUNTIME_SELECTION_BASELINE_PAYLOAD_FILES)


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
    result = _run_push_guard(
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
    result = _run_push_guard(
        guard_py=guard_py,
        run_py=run_py,
        repo_root=repo_root,
    )
    assert result.returncode != 0
    assert "must not re-freeze" in result.stderr


def test_full_push_guard_rejects_beta_target_drift(tmp_path: Path) -> None:
    """The shell push guard independently pins the accepted beta-0.01 policy."""
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
            _set_rejected_beta_target,
        ),
        encoding="utf-8",
    )
    guard_py = _extract_full_push_guard_python(repo_root=repo_root, tmp_path=tmp_path)
    result = _run_push_guard(
        guard_py=guard_py,
        run_py=run_py,
        repo_root=repo_root,
    )
    assert result.returncode != 0
    assert "objective.beta.target must be locked to 0.01" in result.stderr


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
    result = _run_push_guard(
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
    result = _run_push_guard(
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
    result = _run_push_guard(
        guard_py=guard_py,
        run_py=run_py,
        repo_root=repo_root,
    )
    assert result.returncode != 0
    assert "missing required text" in result.stderr


# --- Spec 0011 S17b-3 sub-step 2: the debug push guard delegates to the parser.
#
# The debug PYDEBUGPAYLOAD heredoc now runs on the venv interpreter and validates the
# embedded selected_runtime.json through selected_runtime_plan_errors instead of
# mirroring the eager identity/recipe/snapshot literals. These tests extract and run the
# real guard body verbatim (no drift) against a freshly built debug kernel whose
# embedded plan is rewritten in place.


def _build_debug_kernel(tmp_path: Path, repo_root: Path) -> UploadSimulation:
    return _build_upload_simulation(
        tmp_path=tmp_path,
        repo_root=repo_root,
        kernel_name="selected_runtime_debug",
        ready_marker="KAGGLE_SELECTED_RUNTIME_DEBUG_READY = True",
    )


def test_debug_push_guard_accepts_committed_plan(tmp_path: Path) -> None:
    """The de-pinned debug push guard passes on a freshly built kernel (S17b-3).

    Behavior-preserving: the committed eager v5 plan still validates once the guard
    delegates to selected_runtime_plan_errors instead of mirroring the eager literals.
    """
    repo_root = Path(__file__).resolve().parents[1]
    simulation = _build_debug_kernel(tmp_path, repo_root)
    guard_py = _extract_debug_push_guard_python(repo_root=repo_root, tmp_path=tmp_path)
    result = _run_push_guard(
        guard_py=guard_py,
        run_py=simulation.upload_dir / "run.py",
        repo_root=repo_root,
    )
    assert result.returncode == 0, result.stderr


def test_debug_push_guard_accepts_compiled_winner(tmp_path: Path) -> None:
    """A re-measured bs47 amp-off compile-step plan now passes the debug push guard.

    This is the S17b-3 goal for the debug surface: the shell mirror no longer rejects a
    compiled winner the runtime search can emit (odd batch 47 proves no divisibility).
    """
    repo_root = Path(__file__).resolve().parents[1]
    simulation = _build_debug_kernel(tmp_path, repo_root)
    run_py = simulation.upload_dir / "run.py"
    run_py.write_text(
        _rewrite_embedded_payload(
            run_py.read_text(encoding="utf-8"),
            _install_compiled_winner_plan,
        ),
        encoding="utf-8",
    )
    guard_py = _extract_debug_push_guard_python(repo_root=repo_root, tmp_path=tmp_path)
    result = _run_push_guard(
        guard_py=guard_py,
        run_py=run_py,
        repo_root=repo_root,
    )
    assert result.returncode == 0, result.stderr


def test_debug_push_guard_rejects_self_inconsistent_plan(tmp_path: Path) -> None:
    """A compiled recipe left on the eager identity fails closed with the parser id.

    Proves the guard delegates to the parser's structural identity rather than mirroring
    the recipe literals: an amp-off precision block on the eager v5 row_id makes the
    composed identity disagree with the recorded one.
    """
    repo_root = Path(__file__).resolve().parents[1]
    simulation = _build_debug_kernel(tmp_path, repo_root)
    run_py = simulation.upload_dir / "run.py"
    run_py.write_text(
        _rewrite_embedded_payload(
            run_py.read_text(encoding="utf-8"),
            _install_amp_off_on_eager_identity,
        ),
        encoding="utf-8",
    )
    guard_py = _extract_debug_push_guard_python(repo_root=repo_root, tmp_path=tmp_path)
    result = _run_push_guard(
        guard_py=guard_py,
        run_py=run_py,
        repo_root=repo_root,
    )
    assert result.returncode != 0
    assert "selected_runtime_selected_row_id_not_self_consistent" in result.stderr


def test_debug_push_guard_keeps_hardware_anchor(tmp_path: Path) -> None:
    """De-pinning identity/recipe must not drop the dual-T4 hardware anchor.

    A compiled winner self-declaring single_visible_t4 is rejected by the parser's
    _launch_errors anchor -- the anchor the eager identity literal carried only
    incidentally (Spec 0011 S17b-2 lesson). The mutation also trips identity
    self-consistency, so the assertion targets the anchor-specific id (only
    _launch_errors emits it) to stay load-bearing on the anchor itself.
    """
    repo_root = Path(__file__).resolve().parents[1]
    simulation = _build_debug_kernel(tmp_path, repo_root)
    run_py = simulation.upload_dir / "run.py"
    run_py.write_text(
        _rewrite_embedded_payload(
            run_py.read_text(encoding="utf-8"),
            _install_wrong_accelerator_winner,
        ),
        encoding="utf-8",
    )
    guard_py = _extract_debug_push_guard_python(repo_root=repo_root, tmp_path=tmp_path)
    result = _run_push_guard(
        guard_py=guard_py,
        run_py=run_py,
        repo_root=repo_root,
    )
    assert result.returncode != 0
    assert "selected_runtime_top_level_not_dual_t4_ddp" in result.stderr


def test_selected_runtime_push_guards_run_on_the_venv_interpreter() -> None:
    """Both plan-delegating push guards must open on the venv interpreter, not python3.

    Each heredoc body imports eqvae (selected_runtime_plan_errors); bare python3 has no
    torch or eqvae, so reverting the opener to `python3` breaks the real push with a
    ModuleNotFoundError. The body-only extraction guards can't see the opener line, so
    assert the venv prefix here directly (S17b-3 / S8).
    """
    repo_root = Path(__file__).resolve().parents[1]
    script = (repo_root / "scripts" / "kaggle_kernel.sh").read_text(encoding="utf-8")
    for marker in ("PYDEBUGPAYLOAD", "PYFULLPAYLOAD"):
        opener = f"""PYTHONPATH=src "$python_bin" - "$run_file" <<'{marker}'"""
        assert opener in script, (
            f"{marker} push guard must open on the venv interpreter, not bare python3"
        )


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
    plan_path = fake_repo / "configs" / "spec0001" / "non_eq_vae_selected_runtime.json"
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text(
        json.dumps({"global_batch_size": _NON_REFERENCE_GLOBAL_BATCH}),
        encoding="utf-8",
    )
    config_path = (
        fake_repo / "configs" / "spec0001" / "non_eq_vae_selected_runtime_full.json"
    )
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps({"training": {"epochs": _FULL_EPOCHS}}),
        encoding="utf-8",
    )
    assert derive(fake_repo) == (
        _NON_REFERENCE_UPDATES,
        _NON_REFERENCE_TARGET_UPDATES,
        _NON_REFERENCE_HALF_EPOCH_INTERVAL,
    )


def test_src_contains_only_the_shipped_eqvae_package() -> None:
    """`src/` must hold nothing but `eqvae`, or local imports diverge from Kaggle.

    The editable install emits a naive .pth containing `<repo>/src`, so EVERY
    directory under src/ becomes importable in the venv -- while the Kaggle payload
    ships only `src/eqvae` (build_kaggle_embedded_kernel.py `_payload_files`). Any
    sibling would therefore import locally and raise ModuleNotFoundError on Kaggle,
    after the GPU slot was committed. `src/nn` was exactly that until it moved to
    `reference/nn`.

    Keeping src/ single-child makes the divergence structurally impossible instead
    of relying on someone noticing. This asserts the invariant the .pth depends on;
    reference-only code belongs in `reference/` (spec 0002).
    """
    repo_root = Path(__file__).resolve().parents[1]
    children = sorted(
        path.name
        for path in (repo_root / "src").iterdir()
        if not path.name.startswith((".", "__"))
    )
    assert children == ["eqvae"], (
        f"src/ must contain only the package the Kaggle payload ships, but holds "
        f"{children}. The editable .pth exposes all of src/, so a sibling imports "
        f"locally and dies on Kaggle. Put reference-only code in reference/."
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


# --- Spec 0011 S17b-3: run_template validators delegate to the single-source parser
#
# Both selected-runtime kernels' _validate_baseline_selected_runtime now call
# eqvae.training.selected_runtime.selected_runtime_plan_errors instead of mirroring its
# identity/recipe/batch pins. The parser's own coherence matrix is exhaustively tested
# in tests/test_selected_runtime_full_run.py; these tests prove the kernel WIRING: the
# pre-check accepts what the parser accepts (committed eager plan + a re-measured
# compiled winner), raises with the parser's error id on rejection, and still enforces
# the hardware anchor the de-pinned identity literal used to carry incidentally.

# A compiled winner a re-measured dual-T4 search could emit (amp-off, whole-step
# compile), deliberately using an ODD per-device batch of 47 (global 94) that does
# NOT divide the 300000 training patches: this proves the kernel accepts whatever
# batch the search picks, dropping the partial last batch
# (optimizer_updates_per_epoch = floor(300000 / 94) = 3191; drop_last=True, S16).
# The recipe blocks mirror _consistent_compiled_winner_payload in
# tests/test_selected_runtime_full_run.py (the parser's own S17b acceptance fixture).
_WINNER_ROW_ID = (
    "dual_t4_ddp__bs47__amp_off_fp32__compile_step__indexed_masked__"
    "policy_compile_step_ddp_optimizer_fp32_channels_last"
)
_WINNER_POLICY_ID = "compile_step_ddp_optimizer_fp32_channels_last"
_WINNER_PER_DEVICE_BATCH = 47
_WINNER_GLOBAL_BATCH = 94
_WINNER_BUCKET_CAP_MB = 50
_SELECTED_RUNTIME_TEMPLATE_KERNELS = (
    "selected_runtime_full",
    "selected_runtime_debug",
)
# The amp-off FP32 precision block the compiled winner declares. Shared so the winner
# shaper and the self-inconsistent mutator cannot drift apart.
_AMP_OFF_FP32_MIXED_PRECISION: dict[str, object] = {
    "enabled": False,
    "policy": "amp_off_fp32",
    "autocast_dtype": "",
    "fp32_loss": True,
    "grad_scaler_enabled": False,
}


def _committed_plan_payload(repo_root: Path) -> dict[str, object]:
    return cast(
        "dict[str, object]",
        json.loads(
            (repo_root / _SELECTED_RUNTIME_PAYLOAD_PATH).read_text(encoding="utf-8"),
        ),
    )


def _shape_compiled_winner(payload: dict[str, object]) -> dict[str, object]:
    """Re-shape a committed plan into a self-consistent compiled winner in place.

    Returns:
        The same payload, mutated into a bs47 amp-off whole-step-compile winner.

    """
    payload["selected_row_id"] = _WINNER_ROW_ID
    payload["runtime_policy_id"] = _WINNER_POLICY_ID
    payload["per_device_batch_size"] = _WINNER_PER_DEVICE_BATCH
    payload["global_batch_size"] = _WINNER_GLOBAL_BATCH
    payload["optimizer_updates_per_epoch"] = (
        REAL_TRAIN_PATCH_COUNT // _WINNER_GLOBAL_BATCH
    )
    payload["mixed_precision"] = dict(_AMP_OFF_FP32_MIXED_PRECISION)
    payload["torch_compile"] = {
        "enabled": True,
        "scope": "step",
        "dynamic": False,
        "backend": "inductor",
        "optimize_ddp": "ddp_optimizer",
        "compiled_autograd": False,
        "reorder_compute_comm_overlap": False,
    }
    payload["runtime_policy"] = {
        "memory_format": "channels_last",
        "ddp_static_graph": False,
        "ddp_gradient_as_bucket_view": True,
        "zero_grad_set_to_none": True,
        "ddp_broadcast_buffers": False,
        "ddp_find_unused_parameters": False,
        "ddp_bucket_cap_mb": _WINNER_BUCKET_CAP_MB,
        "fused_optimizer": True,
    }
    snapshot = cast("dict[str, object]", payload["selected_row_snapshot"])
    snapshot.update(
        {
            "row_id": _WINNER_ROW_ID,
            "runtime_policy_id": _WINNER_POLICY_ID,
            "precision_policy": "amp_off_fp32",
            "per_device_batch_size": str(_WINNER_PER_DEVICE_BATCH),
            "global_batch_size": str(_WINNER_GLOBAL_BATCH),
            "grad_scaler_enabled": "false",
            "autocast_dtype": "",
        },
    )
    return payload


def _compiled_winner_plan_payload(repo_root: Path) -> dict[str, object]:
    """Return a self-consistent re-measured compiled winner (bs47 amp-off compile-step).

    Returns:
        The committed plan re-shaped into a fully self-consistent compiled winner.

    """
    return _shape_compiled_winner(_committed_plan_payload(repo_root))


def _load_baseline_validator(
    repo_root: Path,
    kernel_name: str,
) -> Callable[[Path], None]:
    run_template = _load_script_module(
        f"{kernel_name}_run_template",
        repo_root / "kaggle" / "kernels" / kernel_name / "run_template.py",
    )
    return cast(
        "Callable[[Path], None]",
        run_template.__dict__["_validate_baseline_selected_runtime"],
    )


def _validator_error(validate: Callable[[Path], None], plan_path: Path) -> str:
    try:
        validate(plan_path)
    except RuntimeError as error:
        return str(error)
    return ""


def test_selected_runtime_templates_accept_committed_and_compiled_plans(
    tmp_path: Path,
) -> None:
    """Both kernels' pre-check accepts the committed eager plan AND a compiled winner.

    Proves the delegation goal of Spec 0011 S17b-3: a re-measured bs47 amp-off
    compile-step plan is no longer rejected by the kernel-side mirrors.
    """
    repo_root = Path(__file__).resolve().parents[1]
    for kernel_name in _SELECTED_RUNTIME_TEMPLATE_KERNELS:
        validate = _load_baseline_validator(repo_root, kernel_name)
        for label, payload in (
            ("committed", _committed_plan_payload(repo_root)),
            ("compiled", _compiled_winner_plan_payload(repo_root)),
        ):
            plan_path = tmp_path / f"{kernel_name}_{label}.json"
            plan_path.write_text(json.dumps(payload), encoding="utf-8")
            validate(plan_path)


def test_selected_runtime_templates_propagate_parser_rejection(
    tmp_path: Path,
) -> None:
    """A plan the parser rejects makes the pre-check raise with the parser's error id.

    A compiled recipe left on the eager identity is self-inconsistent under the S17b
    structural identity check -- proving the validators delegate rather than mirror (the
    old hand-copies raised a different, recipe-literal message).
    """
    repo_root = Path(__file__).resolve().parents[1]
    payload = _committed_plan_payload(repo_root)
    payload["mixed_precision"] = dict(_AMP_OFF_FP32_MIXED_PRECISION)
    for kernel_name in _SELECTED_RUNTIME_TEMPLATE_KERNELS:
        validate = _load_baseline_validator(repo_root, kernel_name)
        plan_path = tmp_path / f"{kernel_name}_inconsistent.json"
        plan_path.write_text(json.dumps(payload), encoding="utf-8")
        raised = _validator_error(validate, plan_path)
        assert "selected_runtime_selected_row_id_not_self_consistent" in raised


def test_selected_runtime_templates_keep_hardware_anchor(
    tmp_path: Path,
) -> None:
    """De-pinning identity/recipe must not drop the hardware anchor.

    A compiled winner self-declaring a non-dual-T4 accelerator is still rejected by the
    parser's _launch_errors anchor -- the anchor the identity literal used to enforce
    only incidentally (Spec 0011 S17b-2 lesson).
    """
    repo_root = Path(__file__).resolve().parents[1]
    payload = _compiled_winner_plan_payload(repo_root)
    payload["accelerator_mode"] = "single_visible_t4"
    for kernel_name in _SELECTED_RUNTIME_TEMPLATE_KERNELS:
        validate = _load_baseline_validator(repo_root, kernel_name)
        plan_path = tmp_path / f"{kernel_name}_wrong_accel.json"
        plan_path.write_text(json.dumps(payload), encoding="utf-8")
        raised = _validator_error(validate, plan_path)
        assert "selected_runtime_top_level_not_dual_t4_ddp" in raised


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
    assert "--fixed-train-patches" not in phase1
    assert _option_value(phase2, "--output-dir") == str(output_dir)
    assert _option_value(phase2, "--resume") == str(
        output_dir / "resume_probe_phase1" / "checkpoints" / "step_000004.pt",
    )
    assert _option_value(phase2, "--max-train-steps") == "8"
    assert _option_value(phase2, "--save-every-steps") == "4"
    assert _option_value(phase2, "--data") == "ubc-pre-shuffled"
    assert "--fixed-train-patches" not in phase2


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
                    "train_effective_global_epoch_samples": 50,
                    "train_effective_per_rank_epoch_samples": 25,
                    "fixed_train_repeated_to_full_batch": True,
                    "observed_batch_sizes": [25],
                    "observed_ranks": [0, 1],
                    "complete_two_rank_update_coverage": True,
                    "l1_improvement_fraction": 0.08,
                    "recon_loss_improvement_fraction": 0.08,
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
    assert "generated run.py wrapper does not match current run_template.py" in (
        completed.stderr
    )


def test_embedded_kernel_verify_rejects_tampered_wrapper(tmp_path: Path) -> None:
    """Ignored run.py code outside the payload must match the tracked template.

    The clean-tree manifest covers embedded files, but Kaggle executes the surrounding
    wrapper first. A local edit that bypasses the mandatory Torch upgrade must therefore
    fail verification even when every payload hash remains valid.
    """
    repo_root = Path(__file__).resolve().parents[1]
    source_kernel = repo_root / "kaggle" / "kernels" / "setup_smoke"
    build_script = repo_root / "scripts" / "build_kaggle_embedded_kernel.py"
    generated_kernel = tmp_path / "generated_setup_smoke"
    generated_kernel.mkdir()
    shutil.copy2(source_kernel / "kernel-metadata.json", generated_kernel)

    base_command = (
        sys.executable,
        str(build_script),
        "--repo-root",
        str(repo_root),
        "--kernel-dir",
        str(generated_kernel),
        "--template",
        str(source_kernel / "run_template.py"),
        "--ready-marker",
        "KAGGLE_SETUP_SMOKE_READY = True",
        "--allow-dirty",
    )
    subprocess.run(base_command, cwd=repo_root, check=True)  # noqa: S603
    run_path = generated_kernel / "run.py"
    run_text = run_path.read_text(encoding="utf-8")
    run_path.write_text(
        run_text.replace(
            "    _ensure_latest_torch(cpu_only=True)",
            "    if False:\n        _ensure_latest_torch(cpu_only=True)",
            1,
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(  # noqa: S603
        (*base_command, "--verify-only"),
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "wrapper does not match current run_template.py" in completed.stderr


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


def _extract_debug_push_guard_python(*, repo_root: Path, tmp_path: Path) -> Path:
    # Run the real PYDEBUGPAYLOAD guard body verbatim so the test cannot drift from the
    # shipped shell validator (S17b-3 de-pinned it to the single-source parser).
    script = (repo_root / "scripts" / "kaggle_kernel.sh").read_text(encoding="utf-8")
    match = _DEBUG_PUSH_GUARD_HEREDOC_PATTERN.search(script)
    if match is None:
        message = "missing PYDEBUGPAYLOAD guard heredoc in kaggle_kernel.sh"
        raise AssertionError(message)
    guard_py = tmp_path / "debug_push_guard.py"
    guard_py.write_text(match.group("body"), encoding="utf-8")
    return guard_py


def _run_push_guard(
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


def _set_rejected_beta_target(members: dict[str, bytes]) -> None:
    config = cast("dict[str, object]", json.loads(members[_FULL_CONFIG_PAYLOAD_PATH]))
    objective = cast("dict[str, object]", config["objective"])
    beta = cast("dict[str, object]", objective["beta"])
    beta["target"] = 0.1
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


def _embedded_plan(members: dict[str, bytes]) -> dict[str, object]:
    return cast(
        "dict[str, object]",
        json.loads(members[_SELECTED_RUNTIME_PAYLOAD_PATH]),
    )


def _install_compiled_winner_plan(members: dict[str, bytes]) -> None:
    members[_SELECTED_RUNTIME_PAYLOAD_PATH] = json.dumps(
        _shape_compiled_winner(_embedded_plan(members)),
    ).encode("utf-8")


def _install_amp_off_on_eager_identity(members: dict[str, bytes]) -> None:
    plan = _embedded_plan(members)
    plan["mixed_precision"] = dict(_AMP_OFF_FP32_MIXED_PRECISION)
    members[_SELECTED_RUNTIME_PAYLOAD_PATH] = json.dumps(plan).encode("utf-8")


def _install_wrong_accelerator_winner(members: dict[str, bytes]) -> None:
    plan = _shape_compiled_winner(_embedded_plan(members))
    plan["accelerator_mode"] = "single_visible_t4"
    members[_SELECTED_RUNTIME_PAYLOAD_PATH] = json.dumps(plan).encode("utf-8")
