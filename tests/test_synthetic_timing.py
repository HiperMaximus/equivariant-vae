# Copyright 2026 HiperMaximus
"""Tests for the no-dataset synthetic timing pretest."""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess  # noqa: S404
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

from torch.utils.data import DataLoader

from eqvae.benchmarking.synthetic_timing import (
    MANIFEST_FILENAME,
    MATRIX_FILENAME,
    RECOMMENDATIONS_FILENAME,
    RUNTIME_PROOF_FILENAME,
    SYNTHETIC_TIMING_KIND,
    SYNTHETIC_TIMING_PHASE_REPEAT_SHORTLIST,
    SYNTHETIC_TIMING_SCOPE,
    SYNTHETIC_TIMING_SOURCE,
    SYNTHETIC_TIMING_STATUS_PARTIAL,
    SyntheticTimingProfile,
    SyntheticTimingRequest,
    build_synthetic_timing_recommendations_payload,
    build_synthetic_timing_runtime_proof_payload,
    compact_synthetic_timing_profile,
    default_synthetic_timing_profile,
    repeat_shortlist_row_specs,
    write_synthetic_timing_pretest,
)
from eqvae.data.dataloaders import (
    PatchTensorDataset,
    PatchTensorDatasetSpec,
    normalize_uint8_batch,
)
from eqvae.data.patch_shards import PatchShard, PatchShardSpec
from eqvae.data.roots import resolve_patch_data_paths
from eqvae.data.training_batches import (
    PatchTrainingBatch,
    PatchTrainingDataset,
    PatchTrainingDatasetSpec,
    collate_patch_training_samples,
)

if TYPE_CHECKING:
    from eqvae.benchmarking.io import CsvRow

_TINY_PROFILE = SyntheticTimingProfile(
    name="synthetic_binary_tiny_test_v1",
    train_patches=8,
    validation_patches=8,
    image_size=16,
    channels=3,
    seed=20260617,
    write_chunk_patches=4,
)
_EXPECTED_BLOCKED_CLAIM_KEYS = {
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
_TWO_GIB_PROFILE_NAME = "synthetic_binary_2gib_histology_like_v1"
_COMPACT_PROFILE_NAME = "synthetic_binary_0p81gb_histology_like_v1"
_TWO_GIB_SPLIT_PATCHES = 5_456
_TWO_GIB_TOTAL_PATCHES = 10_912
_TWO_GIB_PAYLOAD_BYTES = 2_145_386_496
_COMPACT_SPLIT_PATCHES = 2_048
_COMPACT_TOTAL_PATCHES = 4_096
_COMPACT_PAYLOAD_BYTES = 805_306_368
_PATCH_PAYLOAD_BYTES = 196_608
_GLOBAL_BATCH_128_ELIGIBILITY_PATCHES = 128 * 30
_REPEAT_SHORTLIST_WARMUP_STEPS = 5
_REPEAT_SHORTLIST_MEASURED_STEPS = 25
_EXPECTED_REPEAT_SHORTLIST_ROWS = (
    ("dual_t4_ddp", 8),
    ("single_visible_t4", 32),
    ("single_visible_t4", 12),
    ("single_visible_t4", 4),
)
_EXPECTED_REPEAT_SHORTLIST_ROW_IDS = [
    f"{mode}__bs{batch_size}__amp_off_fp32__compile_off__branchless_all"
    for mode, batch_size in _EXPECTED_REPEAT_SHORTLIST_ROWS
]
_DDP_RANK_COUNT = 2


def test_default_synthetic_timing_profile_is_two_gib_scale() -> None:
    """Default remote profile covers dual global batch 128 without wrapping."""
    profile = default_synthetic_timing_profile()

    assert profile.name == _TWO_GIB_PROFILE_NAME
    assert profile.train_patches == _TWO_GIB_SPLIT_PATCHES
    assert profile.validation_patches == _TWO_GIB_SPLIT_PATCHES
    assert profile.total_patches == _TWO_GIB_TOTAL_PATCHES
    assert profile.patch_payload_bytes == _PATCH_PAYLOAD_BYTES
    assert profile.total_payload_bytes == _TWO_GIB_PAYLOAD_BYTES
    assert profile.train_patches >= _GLOBAL_BATCH_128_ELIGIBILITY_PATCHES


def test_compact_synthetic_timing_profile_preserves_remote_v1_contract() -> None:
    """Historical remote-v1 profile stays named and non-default."""
    profile = compact_synthetic_timing_profile()

    assert profile.name == _COMPACT_PROFILE_NAME
    assert profile.train_patches == _COMPACT_SPLIT_PATCHES
    assert profile.validation_patches == _COMPACT_SPLIT_PATCHES
    assert profile.total_patches == _COMPACT_TOTAL_PATCHES
    assert profile.patch_payload_bytes == _PATCH_PAYLOAD_BYTES
    assert profile.total_payload_bytes == _COMPACT_PAYLOAD_BYTES
    assert profile.train_patches < _GLOBAL_BATCH_128_ELIGIBILITY_PATCHES


def test_repeat_shortlist_row_specs_pin_v3_candidate_order() -> None:
    """Repeat-shortlist rows stay tied to the reviewed v3 broad-screen result."""
    specs = repeat_shortlist_row_specs()

    assert (
        tuple((spec.accelerator_mode, spec.per_device_batch_size) for spec in specs)
        == _EXPECTED_REPEAT_SHORTLIST_ROWS
    )


def test_synthetic_timing_writes_only_non_promotable_artifacts(
    tmp_path: Path,
) -> None:
    """Tiny synthetic timing output proves data parity without selected runtime."""
    artifacts = write_synthetic_timing_pretest(
        SyntheticTimingRequest(
            output_dir=tmp_path,
            run_name="synthetic_timing_test",
            profile=_TINY_PROFILE,
            local_upload_simulation=True,
            batch_sizes=(2,),
            warmup_steps=1,
            measured_steps=1,
            kernel_metadata={"id": "maximusshtefan/eqvae-synthetic-timing"},
        ),
    )

    benchmark_dir = tmp_path / "benchmark"
    assert artifacts.manifest == benchmark_dir / MANIFEST_FILENAME
    assert artifacts.runtime_proof == benchmark_dir / RUNTIME_PROOF_FILENAME
    assert artifacts.matrix == benchmark_dir / MATRIX_FILENAME
    assert artifacts.recommendations == benchmark_dir / RECOMMENDATIONS_FILENAME
    assert {path.name for path in benchmark_dir.iterdir()} == {
        MANIFEST_FILENAME,
        RUNTIME_PROOF_FILENAME,
        MATRIX_FILENAME,
        RECOMMENDATIONS_FILENAME,
    }

    manifest = _load_json(artifacts.manifest)
    runtime_proof = _load_json(artifacts.runtime_proof)
    recommendations = _load_json(artifacts.recommendations)
    assert manifest["status"] == "skipped_unsupported"
    assert runtime_proof["status"] == "skipped_unsupported"
    assert recommendations["status"] == "skipped_unsupported"
    _assert_non_promotable_payload(manifest)
    _assert_non_promotable_payload(runtime_proof)
    _assert_non_promotable_payload(recommendations)
    assert manifest["dataset_sources"] == []
    assert manifest["competition_sources"] == []
    assert manifest["kernel_sources"] == []
    assert manifest["model_sources"] == []

    _assert_named_profiles(manifest)

    data = cast("dict[str, object]", manifest["data"])
    assert data["generation_excluded_from_timing"] is True
    assert data["crc_validated"] is True
    assert data["relative_filenames"] == [
        "dataset/ubc_train_shuffled.bin",
        "dataset/ubc_train_shuffled.csv",
        "dataset/ubc_ocean_valid.bin",
        "dataset/ubc_ocean_valid.csv",
    ]
    assert not (benchmark_dir / "selected_runtime.json").exists()

    train = cast(
        "dict[str, object]",
        cast("dict[str, object]", manifest["splits"])["train"],
    )
    validation = cast(
        "dict[str, object]",
        cast("dict[str, object]", manifest["splits"])["validation"],
    )
    assert train["csv_has_idx"] is False
    assert validation["csv_has_idx"] is True
    assert train["semantic_key_uniqueness_pass"] is True
    assert validation["semantic_key_uniqueness_pass"] is True

    loader_proof = cast("dict[str, object]", manifest["loader_proof"])
    assert loader_proof["tensor_dataset_class"] == "PatchTensorDataset"
    assert loader_proof["training_dataset_class"] == "PatchTrainingDataset"
    assert loader_proof["normalizer"] == "normalize_uint8_batch"
    post_normalization = cast("dict[str, object]", loader_proof["post_normalization"])
    assert post_normalization["dtype"] == "torch.float32"
    assert post_normalization["range_pass"] is True

    rows = _load_csv(artifacts.matrix)
    assert {row["accelerator_mode"] for row in rows} == {
        "single_visible_t4",
        "dual_t4_ddp",
    }
    assert {row["status"] for row in rows} == {"wrong_accelerator"}
    assert {row["full_run_eligible"] for row in rows} == {"false"}
    assert {row["steps_per_epoch"] for row in rows} == {"150000", "75000"}
    assert {row["effective_samples_per_epoch"] for row in rows} == {"300000"}
    summary = cast("dict[str, object]", manifest["timing_row_summary"])
    assert summary["statuses"] == ["wrong_accelerator"]
    assert summary["pass_row_count"] == 0


def test_synthetic_timing_explicit_repeat_shortlist_rows_are_proven(
    tmp_path: Path,
) -> None:
    """Explicit shortlist mode writes only the reviewed repeat rows."""
    artifacts = write_synthetic_timing_pretest(
        SyntheticTimingRequest(
            output_dir=tmp_path,
            run_name="synthetic_timing_repeat_shortlist_test",
            profile=_TINY_PROFILE,
            local_upload_simulation=True,
            batch_sizes=(),
            row_specs=repeat_shortlist_row_specs(),
            warmup_steps=1,
            measured_steps=1,
            timing_phase=SYNTHETIC_TIMING_PHASE_REPEAT_SHORTLIST,
        ),
    )

    manifest = _load_json(artifacts.manifest)
    runtime_proof = _load_json(artifacts.runtime_proof)
    recommendations = _load_json(artifacts.recommendations)
    rows = _load_csv(artifacts.matrix)
    assert [row["row_id"] for row in rows] == _EXPECTED_REPEAT_SHORTLIST_ROW_IDS
    assert [row["row_order"] for row in rows] == ["0", "1", "2", "3"]
    assert {row["status"] for row in rows} == {"wrong_accelerator"}
    for payload in (manifest, runtime_proof, recommendations):
        timing_plan = cast("dict[str, object]", payload["timing_plan"])
        assert timing_plan["timing_phase"] == SYNTHETIC_TIMING_PHASE_REPEAT_SHORTLIST
        assert timing_plan["explicit_row_specs"] is True
        assert timing_plan["batch_sizes"] == []
        assert timing_plan["row_count"] == len(_EXPECTED_REPEAT_SHORTLIST_ROWS)
        planned_rows = cast("list[dict[str, object]]", timing_plan["rows"])
        assert [row["row_id"] for row in planned_rows] == (
            _EXPECTED_REPEAT_SHORTLIST_ROW_IDS
        )


def test_synthetic_timing_generated_root_uses_active_loader_paths(
    tmp_path: Path,
) -> None:
    """Generated shards resolve and load through the active UBC APIs."""
    write_synthetic_timing_pretest(
        SyntheticTimingRequest(
            output_dir=tmp_path,
            profile=_TINY_PROFILE,
            local_upload_simulation=True,
            batch_sizes=(2,),
            warmup_steps=1,
            measured_steps=1,
        ),
    )

    data_root = tmp_path / "synthetic_timing_data"
    paths = resolve_patch_data_paths(data_root)
    train_shard = PatchShard(
        PatchShardSpec(
            bin_path=paths.train.bin_path,
            csv_path=paths.train.csv_path,
            image_size=_TINY_PROFILE.image_size,
            channels=3,
            validate_crc=True,
        ),
    )
    validation_shard = PatchShard(
        PatchShardSpec(
            bin_path=paths.validation.bin_path,
            csv_path=paths.validation.csv_path,
            image_size=_TINY_PROFILE.image_size,
            channels=3,
            validate_crc=True,
        ),
    )
    assert train_shard.crc_validated is True
    assert validation_shard.crc_validated is True

    with paths.train.csv_path.open(encoding="utf-8", newline="") as csv_file:
        assert "idx" not in (csv.DictReader(csv_file).fieldnames or ())
    with paths.validation.csv_path.open(encoding="utf-8", newline="") as csv_file:
        assert "idx" in (csv.DictReader(csv_file).fieldnames or ())

    tensor_dataset = PatchTensorDataset(
        PatchTensorDatasetSpec(
            bin_path=paths.train.bin_path,
            csv_path=paths.train.csv_path,
            split="train",
            image_size=_TINY_PROFILE.image_size,
            channels=3,
        ),
    )
    training_dataset = PatchTrainingDataset(
        PatchTrainingDatasetSpec(
            bin_path=paths.train.bin_path,
            csv_path=paths.train.csv_path,
            split="train",
            image_size=_TINY_PROFILE.image_size,
            channels=3,
        ),
    )
    try:
        assert tensor_dataset[0].shape == (3, 16, 16)
        loader = DataLoader(
            training_dataset,
            batch_size=2,
            num_workers=0,
            collate_fn=collate_patch_training_samples,
        )
        batch = cast("PatchTrainingBatch", next(iter(loader)))
        normalized = normalize_uint8_batch(batch.images_uint8)
        assert normalized.dtype.is_floating_point
        assert float(normalized.min()) >= -1.0
        assert float(normalized.max()) <= 1.0
        assert batch.sample_ids[0].startswith("train:00000000:")
    finally:
        tensor_dataset.close()
        training_dataset.close()


def test_synthetic_timing_marks_large_dual_rows_probe_only(tmp_path: Path) -> None:
    """Dual global batch 128 cannot be ranked with a 2,048-patch split."""
    profile = SyntheticTimingProfile(
        name="synthetic_binary_probe_test_v1",
        train_patches=2048,
        validation_patches=2048,
        image_size=1,
        channels=3,
        seed=20260617,
        write_chunk_patches=128,
    )
    artifacts = write_synthetic_timing_pretest(
        SyntheticTimingRequest(
            output_dir=tmp_path,
            profile=profile,
            local_upload_simulation=True,
            batch_sizes=(64,),
            warmup_steps=1,
            measured_steps=1,
        ),
    )

    rows = _load_csv(artifacts.matrix)
    single = _row_by_mode(rows, "single_visible_t4")
    dual = _row_by_mode(rows, "dual_t4_ddp")
    assert single["global_batch_size"] == "64"
    assert single["non_wrapping_eligible"] == "true"
    assert single["fit_probe_only"] == "false"
    assert dual["global_batch_size"] == "128"
    assert dual["non_wrapping_eligible"] == "false"
    assert dual["fit_probe_only"] == "true"
    assert dual["sample_reuse_count"] == str((128 * 30) - 2048)
    assert single["effective_samples_per_epoch"] == "300000"
    assert dual["effective_samples_per_epoch"] == "300000"


def test_synthetic_timing_recommendations_order_ranked_rows(tmp_path: Path) -> None:
    """Recommendations order feasible passes before probes and failures."""
    rows = cast(
        "list[CsvRow]",
        [
            _recommendation_row(
                row_id="slow",
                status="pass",
                fit_probe_only=False,
                metrics=("18.0", "20.0", "0.8"),
            ),
            _recommendation_row(
                row_id="wrong",
                status="wrong_accelerator",
                fit_probe_only=False,
                metrics=("", "", ""),
            ),
            _recommendation_row(
                row_id="probe",
                status="pass",
                fit_probe_only=True,
                metrics=("8.0", "10.0", "0.9"),
            ),
            _recommendation_row(
                row_id="fast",
                status="pass",
                fit_probe_only=False,
                metrics=("12.0", "15.0", "0.5"),
            ),
        ],
    )
    payload = build_synthetic_timing_recommendations_payload(
        request=SyntheticTimingRequest(output_dir=tmp_path),
        profile=_TINY_PROFILE,
        rows=rows,
    )

    assert (
        payload["estimated_epoch_minutes_scope"]
        == "loader_collate_normalize_h2d_only_projected_to_real_train_patch_count"
    )
    measured_components = cast("dict[str, bool]", payload["measured_components"])
    assert measured_components["loader_collate_normalize_h2d"] is True
    assert measured_components["model_forward_backward"] is False
    assert measured_components["corruption"] is False
    assert measured_components["precision_policy"] is False
    repeat_policy = cast("dict[str, object]", payload["repeat_shortlist_policy"])
    assert repeat_policy["required_before_operational_shortlist"] is True
    assert repeat_policy["warmup_steps"] == _REPEAT_SHORTLIST_WARMUP_STEPS
    assert repeat_policy["measured_steps"] == _REPEAT_SHORTLIST_MEASURED_STEPS
    recommendations = cast("list[dict[str, object]]", payload["recommendations"])
    assert [row["row_id"] for row in recommendations] == [
        "fast",
        "slow",
        "probe",
        "wrong",
    ]
    assert [row["recommendation_rank"] for row in recommendations] == [1, 2, 3, 4]
    assert recommendations[0]["repeat_required_before_operational_shortlist"] is True
    assert recommendations[2]["repeat_required_before_operational_shortlist"] is False


def test_synthetic_timing_recommendations_clear_repeat_gate_after_repeat_phase(
    tmp_path: Path,
) -> None:
    """Repeat-shortlist recommendations do not keep asking for the same repeat."""
    rows = cast(
        "list[CsvRow]",
        [
            _recommendation_row(
                row_id="repeated",
                status="pass",
                fit_probe_only=False,
                metrics=("12.0", "15.0", "0.8"),
            ),
        ],
    )
    payload = build_synthetic_timing_recommendations_payload(
        request=SyntheticTimingRequest(
            output_dir=tmp_path,
            timing_phase=SYNTHETIC_TIMING_PHASE_REPEAT_SHORTLIST,
        ),
        profile=_TINY_PROFILE,
        rows=rows,
    )

    repeat_policy = cast("dict[str, object]", payload["repeat_shortlist_policy"])
    assert repeat_policy["required_before_operational_shortlist"] is False
    assert repeat_policy["completed"] is True
    recommendations = cast("list[dict[str, object]]", payload["recommendations"])
    assert recommendations[0]["recommendation"] == "carry_to_real_benchmark"
    assert recommendations[0]["repeat_required_before_operational_shortlist"] is False


def test_synthetic_timing_recommendations_mark_mixed_rows_partial(
    tmp_path: Path,
) -> None:
    """Top-level status does not hide failed rows behind any passing row."""
    rows = cast(
        "list[CsvRow]",
        [
            _recommendation_row(
                row_id="pass",
                status="pass",
                fit_probe_only=False,
                metrics=("12.0", "15.0", "0.8"),
            ),
            _recommendation_row(
                row_id="wrong",
                status="wrong_accelerator",
                fit_probe_only=False,
                metrics=("", "", ""),
            ),
        ],
    )
    payload = build_synthetic_timing_recommendations_payload(
        request=SyntheticTimingRequest(output_dir=tmp_path),
        profile=_TINY_PROFILE,
        rows=rows,
    )

    assert payload["status"] == SYNTHETIC_TIMING_STATUS_PARTIAL


def test_synthetic_timing_runtime_proof_preserves_ddp_rank_assignments(
    tmp_path: Path,
) -> None:
    """Runtime proof keeps rank/device assignment evidence for dual DDP rows."""
    rows = cast(
        "list[CsvRow]",
        [
            _runtime_proof_row(
                row_id="single_visible_t4__bs8__amp_off_fp32__compile_off",
                row_order="0",
                accelerator_mode="single_visible_t4",
                gpu_names=json.dumps(["Tesla T4"]),
                world_size="1",
                nproc_per_node="1",
                cuda_visible_devices="0",
            ),
            _runtime_proof_row(
                row_id="dual_t4_ddp__bs8__amp_off_fp32__compile_off",
                row_order="1",
                accelerator_mode="dual_t4_ddp",
                gpu_names=json.dumps(["Tesla T4", "Tesla T4"]),
                world_size="2",
                nproc_per_node="2",
                cuda_visible_devices="0,1",
                ddp_torchrun_returncode="0",
                ddp_rank_count="2",
                ddp_rank_order=json.dumps([0, 1]),
                ddp_rank_assignments_json=json.dumps([
                    {
                        "rank": 0,
                        "local_rank": 0,
                        "world_size": 2,
                        "device_name": "Tesla T4",
                        "samples": 96,
                        "measured_batches": 12,
                    },
                    {
                        "rank": 1,
                        "local_rank": 1,
                        "world_size": 2,
                        "device_name": "Tesla T4",
                        "samples": 96,
                        "measured_batches": 12,
                    },
                ]),
            ),
        ],
    )
    payload = build_synthetic_timing_runtime_proof_payload(
        request=SyntheticTimingRequest(output_dir=tmp_path),
        rows=rows,
    )

    dual_summary = cast("dict[str, object]", payload["dual_t4_ddp"])
    assert dual_summary["row_ids_in_matrix_order"] == [
        "dual_t4_ddp__bs8__amp_off_fp32__compile_off",
    ]
    child_returncodes = cast(
        "list[dict[str, object]]",
        dual_summary["child_returncodes"],
    )
    assert child_returncodes == [
        {
            "row_id": "dual_t4_ddp__bs8__amp_off_fp32__compile_off",
            "row_order": 1,
            "status": "pass",
            "child_returncode": "0",
        },
    ]
    rank_rows = cast("list[dict[str, object]]", dual_summary["rank_assignment_rows"])
    assert len(rank_rows) == 1
    rank_row = rank_rows[0]
    assert rank_row["torchrun_returncode"] == 0
    assert rank_row["rank_count"] == _DDP_RANK_COUNT
    assert rank_row["rank_order"] == [0, 1]
    rank_assignments = cast("list[dict[str, object]]", rank_row["rank_assignments"])
    assert [item["local_rank"] for item in rank_assignments] == [0, 1]
    assert [item["device_name"] for item in rank_assignments] == [
        "Tesla T4",
        "Tesla T4",
    ]


def test_synthetic_timing_push_guard_rejects_source_attachments(
    tmp_path: Path,
) -> None:
    """The shell push guard rejects no-dataset contract drift before Kaggle."""
    repo_root = Path(__file__).resolve().parents[1]
    kernel_dir = _generated_kernel_dir(tmp_path=tmp_path, repo_root=repo_root)
    metadata_path = kernel_dir / "kernel-metadata.json"
    metadata = _load_json(metadata_path)
    metadata["dataset_sources"] = ["maximusshtefan/patches-pre-shuffled-ubc-ocean"]
    metadata_path.write_text(
        f"{json.dumps(metadata, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    fake_bin = _fake_bin(tmp_path=tmp_path, repo_root=repo_root)

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
    assert "dataset_sources must be an empty list" in completed.stderr


def test_synthetic_timing_push_guard_accepts_generated_no_dataset_kernel(
    tmp_path: Path,
) -> None:
    """The positive guard path reaches fake Kaggle without network access."""
    repo_root = Path(__file__).resolve().parents[1]
    kernel_dir = _generated_kernel_dir(tmp_path=tmp_path, repo_root=repo_root)
    fake_bin = _fake_bin(tmp_path=tmp_path, repo_root=repo_root)

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


def _generated_kernel_dir(*, tmp_path: Path, repo_root: Path) -> Path:
    kernel_source = repo_root / "kaggle" / "kernels" / "synthetic_timing"
    kernel_dir = tmp_path / "synthetic_timing_kernel"
    kernel_dir.mkdir()
    shutil.copy2(kernel_source / "kernel-metadata.json", kernel_dir)
    fake_bin = _fake_bin(tmp_path=tmp_path, repo_root=repo_root)
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
            "KAGGLE_SYNTHETIC_TIMING_READY = True",
        ),
        cwd=repo_root,
        env=_guard_environment(fake_bin=fake_bin, push_confirmed=False),
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
) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PATH"] = f"{fake_bin}{os.pathsep}{environment['PATH']}"
    environment["KAGGLE_DISABLE_FRESH_OAUTH"] = "1"
    if push_confirmed:
        environment["KAGGLE_PUSH_CONFIRMED"] = "1"
    environment.pop("KAGGLE_FULL_DATASET_CONFIRMED", None)
    return environment


def _recommendation_row(
    *,
    row_id: str,
    status: str,
    fit_probe_only: bool,
    metrics: tuple[str, str, str],
) -> dict[str, str]:
    estimated_epoch_minutes, steady_step_ms_p95, vram_headroom_fraction = metrics
    return {
        "row_id": row_id,
        "accelerator_mode": "single_visible_t4",
        "per_device_batch_size": "4",
        "global_batch_size": "4",
        "non_wrapping_eligible": "false" if fit_probe_only else "true",
        "fit_probe_only": "true" if fit_probe_only else "false",
        "status": status,
        "estimated_epoch_minutes": estimated_epoch_minutes,
        "steady_step_ms_p95": steady_step_ms_p95,
        "vram_headroom_fraction": vram_headroom_fraction,
    }


def _runtime_proof_row(**overrides: str) -> dict[str, str]:
    row = {
        "row_id": "row",
        "row_order": "",
        "accelerator_mode": "single_visible_t4",
        "status": "pass",
        "gpu_names": "[]",
        "world_size": "1",
        "nproc_per_node": "1",
        "cuda_visible_devices": "0",
        "child_returncode": "0",
        "ddp_torchrun_returncode": "",
        "ddp_rank_count": "",
        "ddp_rank_order": "",
        "ddp_rank_assignments_json": "",
    }
    row.update(overrides)
    return row


def _assert_non_promotable_payload(payload: dict[str, object]) -> None:
    assert payload["benchmark_kind"] == SYNTHETIC_TIMING_KIND
    assert payload["benchmark_source"] == SYNTHETIC_TIMING_SOURCE
    assert payload["status_scope"] == SYNTHETIC_TIMING_SCOPE
    assert payload["full_run_eligible"] is False
    blocked_claims = cast("dict[str, bool]", payload["blocked_claims"])
    assert set(blocked_claims) == _EXPECTED_BLOCKED_CLAIM_KEYS
    assert all(blocked_claims.values())


def _assert_named_profiles(manifest: dict[str, object]) -> None:
    profile_payload = cast("dict[str, object]", manifest["profile"])
    named_profiles = cast("dict[str, object]", profile_payload["named_profiles"])
    current_default = cast("dict[str, object]", named_profiles["current_default"])
    historical_remote_v1 = cast(
        "dict[str, object]",
        named_profiles["historical_remote_v1"],
    )
    assert current_default["name"] == _TWO_GIB_PROFILE_NAME
    assert current_default["payload_bytes"] == _TWO_GIB_PAYLOAD_BYTES
    assert historical_remote_v1["name"] == _COMPACT_PROFILE_NAME
    assert historical_remote_v1["payload_bytes"] == _COMPACT_PAYLOAD_BYTES


def _row_by_mode(rows: list[dict[str, str]], accelerator_mode: str) -> dict[str, str]:
    for row in rows:
        if row["accelerator_mode"] == accelerator_mode:
            return row
    raise AssertionError(accelerator_mode)


def _load_json(path: Path) -> dict[str, object]:
    return cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))
