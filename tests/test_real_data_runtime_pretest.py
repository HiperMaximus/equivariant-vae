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
from typing import TYPE_CHECKING, cast

import pytest
import torch

from eqvae.benchmarking import real_data_runtime_pretest as pretest
from eqvae.benchmarking.real_data_runtime_pretest import (
    RealDataRuntimePretestRequest,
    write_real_data_runtime_pretest,
)
from eqvae.config import resolve_json_config

if TYPE_CHECKING:
    from eqvae.benchmarking.io import CsvRow, JsonObject
from eqvae.data.synthetic import SyntheticPatchSpec, write_synthetic_patch_shard

_TINY_IMAGE_SIZE = 16
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
_TEST_GATE_QUANTILE_CAP = 4
_FLOAT_TOLERANCE = 1.0e-6


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
    phase_timings = _load_json(benchmark_dir / "phase_timings.json")
    _assert_phase_timings(
        phase_timings,
        required_names={
            "config_resolution",
            "real_data_identity_and_clean_path_proof",
            "stage1_runtime_rows",
            "linked_evidence_payload",
            "write_artifacts",
        },
    )
    assert (
        cast("dict[str, object]", manifest["phase_timings"])["schema_version"]
        == "eqvae.phase_timings.v1"
    )
    assert (
        cast("dict[str, object]", runtime_proof["phase_timings"])["schema_version"]
        == "eqvae.phase_timings.v1"
    )
    assert "phase_timings.json" in cast("list[str]", manifest["artifact_allowlist"])
    assert manifest["real_data_identity_proof_status"] == "skipped_unsupported"
    assert manifest["validation_windows_exercised"] is False
    assert manifest["timed_rows_eligible"] is False
    real_data_proof = cast("dict[str, object]", manifest["real_data_proof"])
    assert real_data_proof["failure_kind"] == "data_root_unavailable"
    assert not real_data_proof["resolved_data_root"]
    diagnostics = cast("dict[str, object]", real_data_proof["data_root_diagnostics"])
    assert diagnostics["requested_data_root"] == "auto"
    assert diagnostics["kaggle_input_exists"] is False
    assert diagnostics["candidate_count"]
    assert diagnostics["accepted_candidates"]
    assert diagnostics["complete_unaccepted_candidate_count"] == 0
    assert "env_value" not in diagnostics
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
    numerical_rows = _load_csv(benchmark_dir / "numerical_checks.csv")
    corruption_rows = _load_csv(benchmark_dir / "corruption_checks.csv")
    gate_rows = _load_csv(tmp_path / "run" / "metrics" / "gate_health.csv")
    _assert_tiny_manifest_linked_evidence(manifest)
    _assert_tiny_runtime_proof_linked_evidence(runtime_proof)
    _assert_tiny_linked_csv_rows(
        dataloader_rows=dataloader_rows,
        numerical_rows=numerical_rows,
        corruption_rows=corruption_rows,
        gate_rows=gate_rows,
    )
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


def test_real_data_pretest_validate_allows_current_worktree_payload() -> None:
    """Local validate accepts a payload freshly built from the current worktree."""
    repo_root = Path(__file__).resolve().parents[1]
    kernel_dir = "kaggle/kernels/real_data_runtime_pretest"

    build = subprocess.run(  # noqa: S603
        (
            _required_executable("bash"),
            str(repo_root / "scripts" / "kaggle_kernel.sh"),
            "build",
            kernel_dir,
        ),
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    validate = subprocess.run(  # noqa: S603
        (
            _required_executable("bash"),
            str(repo_root / "scripts" / "kaggle_kernel.sh"),
            "validate",
            kernel_dir,
        ),
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert build.returncode == 0, build.stderr
    assert validate.returncode == 0, validate.stderr
    assert "matches current worktree" in validate.stdout


def test_kaggle_pull_guard_requires_remote_confirmation(tmp_path: Path) -> None:
    """Pull is a remote read and refuses even pull-specific approval alone."""
    repo_root = Path(__file__).resolve().parents[1]
    fake_bin = _fake_bin(tmp_path=tmp_path, repo_root=repo_root)

    completed = subprocess.run(  # noqa: S603
        (
            _required_executable("bash"),
            str(repo_root / "scripts" / "kaggle_kernel.sh"),
            "pull",
            "maximusshtefan/eqvae-real-data-runtime-pretest",
            str(tmp_path / "pulled_kernel"),
        ),
        cwd=repo_root,
        env=_guard_environment(
            fake_bin=fake_bin,
            push_confirmed=False,
            full_dataset_confirmed=False,
            pull_confirmed=True,
            remote_confirmed=False,
        ),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "KAGGLE_REMOTE_CONFIRMED=1" in completed.stderr
    assert "fake kaggle" not in completed.stdout


def test_train_step_target_rows_prioritize_eager_before_compiled() -> None:
    """Candidate evidence spends coverage on eager smaller batches first."""
    rows = [
        _train_step_target_row(
            row_id="compiled_bs4",
            batch_size=4,
            compile_scope="model_forward",
        ),
        _train_step_target_row(
            row_id="eager_bs12",
            batch_size=12,
            compile_scope="none",
        ),
        _train_step_target_row(
            row_id="eager_bs4",
            batch_size=4,
            compile_scope="none",
        ),
        _train_step_target_row(
            row_id="compiled_bs8",
            batch_size=8,
            compile_scope="model_forward",
        ),
        _train_step_target_row(
            row_id="eager_bs8",
            batch_size=8,
            compile_scope="none",
        ),
    ]

    ordered = pretest._unique_train_step_target_rows(rows)  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001

    assert [row["row_id"] for row in ordered] == [
        "eager_bs4",
        "eager_bs8",
        "eager_bs12",
        "compiled_bs4",
        "compiled_bs8",
    ]


def test_gate_quantiles_use_exact_small_tensor_path() -> None:
    """Small gate tensors keep exact torch.quantile telemetry."""
    tensor = torch.tensor([0.0, 1.0, 2.0, 4.0], dtype=torch.float32)

    observed = pretest._tensor_quantile(tensor, 0.50)  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
    expected = float(torch.quantile(tensor.flatten(), 0.50).item())

    assert abs(observed - expected) <= _FLOAT_TOLERANCE


def test_gate_quantiles_sample_large_tensor_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Large gate quantiles are deterministic and bounded without huge tensors."""
    monkeypatch.setattr(
        pretest,
        "MAX_GATE_QUANTILE_ELEMENTS",
        _TEST_GATE_QUANTILE_CAP,
    )
    tensor = torch.arange(10, dtype=torch.float32)

    sampled = pretest._gate_quantile_values(tensor)  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
    first = pretest._tensor_quantile(tensor, 0.50)  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
    second = pretest._tensor_quantile(tensor, 0.50)  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001

    assert sampled.numel() == _TEST_GATE_QUANTILE_CAP
    assert torch.equal(sampled, torch.tensor([0.0, 3.0, 6.0, 9.0]))
    assert abs(first - float(torch.quantile(sampled, 0.50).item())) <= _FLOAT_TOLERANCE
    assert second == first


def test_gate_health_lane_pass_does_not_cover_missing_candidate_rows() -> None:
    """A lane-level pass cannot make uncovered runtime rows gate-health pass."""
    covered = _train_step_target_row(
        row_id="single_visible_t4__bs4__amp_off_fp32__compile_none__branchless_all",
        batch_size=4,
        compile_scope="none",
    )
    uncovered = _train_step_target_row(
        row_id="single_visible_t4__bs8__amp_off_fp32__compile_none__branchless_all",
        batch_size=8,
        compile_scope="none",
    )
    rows = [covered, uncovered]
    linked_evidence = cast(
        "JsonObject",
        {
            "ddp_launch": {"status": "pass"},
            "compile_settle": {"status": "skipped_unsupported"},
            "dataloader_throughput": {
                "status": "pass",
                "rows": [
                    _passing_dataloader_row(row=row, split=split)
                    for row in rows
                    for split in ("train", "validation")
                ],
            },
            "paired_numerical": {
                "status": "pass",
                "rows": [{"row_id": row["row_id"], "status": "pass"} for row in rows],
            },
            "corruption_equivalence": {
                "status": "pass",
                "rows": [{"row_id": row["row_id"], "status": "pass"} for row in rows],
            },
            "gate_health": {
                "status": "pass",
                "rows": [],
                "row_statuses": [
                    {
                        "row_id": covered["row_id"],
                        "accelerator_mode": covered["accelerator_mode"],
                        "world_size": int(covered["world_size"]),
                        "per_device_batch_size": int(covered["per_device_batch_size"]),
                        "precision_policy": covered["precision_policy"],
                        "compile_scope": covered["compile_scope"],
                        "status": "pass",
                    },
                ],
            },
        },
    )
    data_proof = cast(
        "JsonObject",
        {
            "identity_status": "pass",
            "row_count_status": "pass",
            "crc_validation_status": "pass",
            "window_status": "pass",
            "clean_validation_dataloader_status": "pass",
        },
    )

    updated = pretest._rows_with_linked_evidence(  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
        rows=rows,
        data_proof=data_proof,
        linked_evidence=linked_evidence,
    )
    by_id = {row["row_id"]: row for row in updated}

    assert by_id[covered["row_id"]]["status"] == "pass"
    assert by_id[covered["row_id"]]["gate_health_status"] == "pass"
    assert by_id[uncovered["row_id"]]["status"] == "ineligible"
    assert by_id[uncovered["row_id"]]["gate_health_status"] == "skipped_unsupported"
    assert by_id[uncovered["row_id"]]["failure_kind"] == (
        "gate_health_evidence_not_row_pass"
    )


def test_train_step_evidence_failure_preserves_candidate_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All-failed candidate evidence returns proof diagnostics instead of raising."""
    repo_root = Path(__file__).resolve().parents[1]
    data_root = _write_tiny_patch_root(tmp_path)
    config_path = _write_tiny_runtime_pretest_config(
        tmp_path=tmp_path,
        repo_root=repo_root,
        data_root=data_root,
    )
    settings = pretest._settings(  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
        resolve_json_config(config_path),
        data_root_override=None,
    )
    rows = [
        _train_step_target_row(
            row_id="single_visible_t4__bs4__amp_off_fp32__compile_none__branchless_all",
            batch_size=4,
            compile_scope="none",
        ),
        _train_step_target_row(
            row_id="single_visible_t4__bs4__amp_off_fp32__compile_none__indexed_masked",
            batch_size=4,
            compile_scope="none",
            corruption_strategy="indexed_masked",
        ),
    ]

    def fail_fixed_batch(*_args: object, **kwargs: object) -> object:
        target_row = cast("dict[str, str]", kwargs["target_row"])
        raise pretest._CandidateTrainStepEvidenceError(  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
            strategy_attempt="indexed_masked",
            target_corruption_strategy=target_row["corruption_strategy"],
            cause=RuntimeError("synthetic candidate boom"),
        )

    monkeypatch.setattr(
        pretest,
        "_paired_fixed_batch_train_step_evidence",
        fail_fixed_batch,
    )

    evidence = pretest._paired_train_step_evidence(  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
        settings=settings,
        data_proof={"identity_status": "local_pass"},
        rows=rows,
    )
    numerical = pretest._paired_numerical_proof(  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
        settings=settings,
        data_proof={"identity_status": "local_pass"},
        rows=rows,
        train_step_evidence=evidence,
    )
    corruption = pretest._corruption_equivalence_proof(  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
        settings=settings,
        data_proof={"identity_status": "local_pass"},
        rows=rows,
        train_step_evidence=evidence,
    )

    assert evidence["status"] == "fail"
    assert evidence["candidate_evidence_count"] == 0
    assert evidence["failed_candidate_evidence_count"] == 1
    failed = cast("list[dict[str, object]]", evidence["failed_candidate_evidence"])
    assert failed[0]["strategy_attempt"] == "indexed_masked"
    assert failed[0]["target_corruption_strategy"] == "branchless_all"
    assert failed[0]["failure_message_excerpt"] == "synthetic candidate boom"
    assert set(cast("list[str]", failed[0]["affected_row_ids"])) == {
        "single_visible_t4__bs4__amp_off_fp32__compile_none__branchless_all",
        "single_visible_t4__bs4__amp_off_fp32__compile_none__indexed_masked",
    }
    assert numerical["status"] == "fail"
    assert numerical["failed_candidate_evidence_count"] == 1
    numerical_failed = cast(
        "list[dict[str, object]]",
        numerical["failed_candidate_evidence"],
    )
    assert numerical_failed[0]["failure_message_excerpt"] == "synthetic candidate boom"
    assert corruption["status"] == "fail"
    assert corruption["failed_candidate_evidence_count"] == 1
    corruption_failed = cast(
        "list[dict[str, object]]",
        corruption["failed_candidate_evidence"],
    )
    assert corruption_failed[0]["failure_message_excerpt"] == "synthetic candidate boom"


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
    pull_confirmed: bool = False,
    remote_confirmed: bool = False,
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
    if pull_confirmed:
        environment["KAGGLE_PULL_CONFIRMED"] = "1"
    else:
        environment.pop("KAGGLE_PULL_CONFIRMED", None)
    if remote_confirmed:
        environment["KAGGLE_REMOTE_CONFIRMED"] = "1"
    else:
        environment.pop("KAGGLE_REMOTE_CONFIRMED", None)
    return environment


def _train_step_target_row(
    *,
    row_id: str,
    batch_size: int,
    compile_scope: str,
    corruption_strategy: str = "branchless_all",
) -> CsvRow:
    return {
        "row_id": row_id,
        "accelerator_mode": "single_visible_t4",
        "world_size": "1",
        "per_device_batch_size": str(batch_size),
        "precision_policy": "amp_off_fp32",
        "compile_scope": compile_scope,
        "corruption_strategy": corruption_strategy,
        "status": "ineligible",
    }


def _passing_dataloader_row(*, row: CsvRow, split: str) -> CsvRow:
    return {
        "accelerator_mode": row["accelerator_mode"],
        "world_size": row["world_size"],
        "batch_size": row["per_device_batch_size"],
        "split": split,
        "status": "pass",
        "data_wait_fraction_p95": "0.010000",
        "loader_samples_sec": "100.000000",
        "trainer_samples_sec": "10.000000",
    }


def _load_json(path: Path) -> dict[str, object]:
    return cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def _assert_tiny_manifest_linked_evidence(manifest: dict[str, object]) -> None:
    assert manifest["real_data_identity_proof_status"] == "local_pass"
    assert manifest["row_count_proof_status"] == "pass"
    assert manifest["crc_validation_status"] == "pass"
    assert manifest["train_windows_exercised"] is True
    assert manifest["validation_windows_exercised"] is True
    assert manifest["linked_evidence_status"] == "skipped_unsupported"
    assert _object_status(manifest, "compile_settle_proof") == "skipped_unsupported"
    assert (
        _object_field(manifest, "compile_settle_proof", "contract_status")
        == "local_pass"
    )
    assert _object_status(manifest, "ddp_launch_proof") == "skipped_unsupported"
    assert _object_field(manifest, "ddp_launch_proof", "contract_status") == (
        "local_pass"
    )
    assert _object_status(manifest, "dataloader_throughput_proof") == "local_pass"
    assert _object_status(manifest, "paired_numerical_proof") == "local_pass"
    assert (
        _object_field(manifest, "paired_numerical_proof", "candidate_row_specific")
        is False
    )
    assert _object_field(manifest, "paired_numerical_proof", "candidate_evidence_count")
    assert (
        _object_field(
            manifest,
            "paired_numerical_proof",
            "failed_candidate_evidence_count",
        )
        == 0
    )
    assert _object_status(manifest, "corruption_equivalence_proof") == "local_pass"
    assert _object_field(
        manifest,
        "corruption_equivalence_proof",
        "candidate_evidence_count",
    )
    assert (
        _object_field(
            manifest,
            "corruption_equivalence_proof",
            "failed_candidate_evidence_count",
        )
        == 0
    )
    assert (
        _object_field(
            manifest,
            "corruption_equivalence_proof",
            "clean_validation_rng_status",
        )
        == "not_exercised_training_batch_only"
    )
    assert _object_status(manifest, "gate_health_proof") == "local_pass"
    assert (
        len(cast("list[object]", manifest["file_hashes"])) == _EXPECTED_FILE_HASH_COUNT
    )
    _assert_tiny_clean_validation_proof(
        cast("dict[str, object]", manifest["clean_validation_dataloader_proof"]),
    )
    _assert_tiny_real_data_proof(
        cast("dict[str, object]", manifest["real_data_proof"]),
    )


def _assert_tiny_clean_validation_proof(clean_proof: dict[str, object]) -> None:
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


def _assert_tiny_real_data_proof(proof: dict[str, object]) -> None:
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


def _assert_tiny_runtime_proof_linked_evidence(
    runtime_proof: dict[str, object],
) -> None:
    assert runtime_proof["real_data_identity_proof_status"] == "local_pass"
    assert runtime_proof["clean_validation_dataloader_status"] == "pass"
    assert runtime_proof["linked_evidence_status"] == "skipped_unsupported"
    compile_policy = cast("dict[str, object]", runtime_proof["compile_settle_policy"])
    assert compile_policy["implemented_in_this_runner"] is True
    assert compile_policy["implemented_compile_scopes"] == ["model_forward"]
    assert compile_policy["contract_proof_available"] is True
    assert compile_policy["status"] == "skipped_unsupported"
    assert runtime_proof["paired_numerical_status"] == "local_pass"
    assert runtime_proof["corruption_equivalence_status"] == "local_pass"
    assert cast("int", runtime_proof["paired_numerical_candidate_evidence_count"]) >= 1
    assert runtime_proof["paired_numerical_failed_candidate_evidence_count"] == 0
    assert (
        cast("int", runtime_proof["corruption_equivalence_candidate_evidence_count"])
        >= 1
    )
    assert runtime_proof["corruption_equivalence_failed_candidate_evidence_count"] == 0
    assert runtime_proof["gate_health_status"] == "local_pass"
    assert runtime_proof["ddp_launch_status"] == "skipped_unsupported"
    assert "real-data identity" in cast("str", runtime_proof["evidence_gate"])


def _assert_phase_timings(
    payload: dict[str, object],
    *,
    required_names: set[str],
) -> None:
    assert payload["schema_version"] == "eqvae.phase_timings.v1"
    assert payload["recorded_phase_count"] == len(
        cast("list[object]", payload["phases"]),
    )
    assert cast("float", payload["total_elapsed_sec"]) >= 0.0
    phases = [
        cast("dict[str, object]", item)
        for item in cast("list[object]", payload["phases"])
    ]
    names = {cast("str", phase["name"]) for phase in phases}
    assert required_names.issubset(names)
    for phase in phases:
        assert phase["status"] in {"pass", "fail"}
        assert cast("float", phase["elapsed_sec"]) >= 0.0
        assert phase["started_at_utc"]
        assert phase["finished_at_utc"]


def _assert_tiny_linked_csv_rows(
    *,
    dataloader_rows: list[dict[str, str]],
    numerical_rows: list[dict[str, str]],
    corruption_rows: list[dict[str, str]],
    gate_rows: list[dict[str, str]],
) -> None:
    validation_rows = [row for row in dataloader_rows if row["split"] == "validation"]
    train_rows = [row for row in dataloader_rows if row["split"] == "train"]
    assert len(validation_rows) == 1
    assert len(train_rows) == 1
    assert validation_rows[0]["status"] == "local_pass"
    assert train_rows[0]["status"] == "local_pass"
    validation_measured = int(validation_rows[0]["rank_sample_count"])
    assert 0 < validation_measured <= _TINY_VALIDATION_PATCHES
    assert numerical_rows
    assert all(row["status"] == "skipped_unsupported" for row in numerical_rows)
    assert all(
        row["failure_kind"] == "compile_or_ddp_numerical_pending"
        for row in numerical_rows
    )
    assert corruption_rows
    assert all(row["status"] == "skipped_unsupported" for row in corruption_rows)
    assert all(not row["clean_validation_rng_advanced"] for row in corruption_rows)
    assert gate_rows
    assert all(row["gate_health_status"] == "local_pass" for row in gate_rows)


def _object_status(payload: dict[str, object], key: str) -> object:
    return cast("dict[str, object]", payload[key])["status"]


def _object_field(payload: dict[str, object], key: str, field: str) -> object:
    return cast("dict[str, object]", payload[key])[field]


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
