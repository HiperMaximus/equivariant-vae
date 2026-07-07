# Copyright 2026 HiperMaximus
"""No-dataset Kaggle synthetic timing pretest for spec 0001."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess  # noqa: S404
import sys
import tempfile
import time
import traceback
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from eqvae.benchmarking.io import CsvRow, JsonObject, JsonValue, write_csv, write_json
from eqvae.benchmarking.schedule import training_steps_per_epoch
from eqvae.data.dataloaders import (
    PatchTensorDataset,
    PatchTensorDatasetSpec,
    normalize_uint8_batch,
)
from eqvae.data.patch_shards import (
    PATCH_SHARD_HEADER_SIZE,
    PatchRecord,
    PatchShard,
    PatchShardHeader,
    PatchShardSpec,
    compute_patch_payload_crc,
    load_patch_records,
    make_patch_shard_header,
)
from eqvae.data.roots import (
    REAL_TRAIN_PATCH_COUNT,
    PatchDataPaths,
    PatchSplit,
    resolve_patch_data_paths,
)
from eqvae.data.training_batches import (
    PatchTrainingBatch,
    PatchTrainingDataset,
    PatchTrainingDatasetSpec,
    collate_patch_training_samples,
    semantic_key_for_record,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence
    from typing import Protocol

    class _CudaDeviceProperties(Protocol):
        """Typed subset of PyTorch CUDA device properties."""

        total_memory: int


SYNTHETIC_TIMING_SCHEMA_VERSION = "spec0001.kaggle_synthetic_timing.v1"
SYNTHETIC_TIMING_KIND = "kaggle_synthetic_timing_pretest"
SYNTHETIC_TIMING_SOURCE = "kaggle_no_dataset_generated_ubc_shards"
SYNTHETIC_TIMING_SCOPE = "non_promotable_synthetic_timing"
SYNTHETIC_TIMING_STATUS_PASS = "synthetic_timing_pass"  # noqa: S105
SYNTHETIC_TIMING_STATUS_PARTIAL = "synthetic_timing_partial"
SYNTHETIC_TIMING_STATUS_SKIPPED = "skipped_unsupported"
SYNTHETIC_TIMING_DATA_ORIGIN = "/kaggle/working_generated_synthetic"
DEFAULT_PROFILE_NAME = "synthetic_binary_2gib_histology_like_v1"
COMPACT_PROFILE_NAME = "synthetic_binary_0p81gb_histology_like_v1"
TEST_PROFILE_NAME = "synthetic_binary_tiny_upload_simulation_v1"
PIXEL_PROFILE_NAME = "histology_like_rgb_v1"
DEFAULT_TOTAL_PATCHES = 10_912
DEFAULT_SPLIT_PATCHES = 5_456
COMPACT_TOTAL_PATCHES = 4_096
COMPACT_SPLIT_PATCHES = 2_048
DEFAULT_IMAGE_SIZE = 256
DEFAULT_CHANNELS = 3
DEFAULT_SEED = 20260617
DEFAULT_WRITE_CHUNK_PATCHES = 8
TEST_SPLIT_PATCHES = 8
TEST_IMAGE_SIZE = 32
NON_WRAPPING_ELIGIBILITY_STEPS = 30
DEFAULT_WARMUP_STEPS = 3
DEFAULT_MEASURED_STEPS = 12
DEFAULT_BATCH_SIZES = (4, 8, 12, 16, 24, 32, 48, 64)
TEST_BATCH_SIZES = (2,)
REPEAT_SHORTLIST_WARMUP_STEPS = 5
REPEAT_SHORTLIST_MEASURED_STEPS = 25
SYNTHETIC_TIMING_PHASE_BROAD_SCREEN = "broad_screening"
SYNTHETIC_TIMING_PHASE_REPEAT_SHORTLIST = "repeat_shortlist"
SINGLE_T4_DEVICE_COUNT = 1
DUAL_T4_DEVICE_COUNT = 2
MATRIX_FILENAME = "synthetic_timing_matrix.csv"
MANIFEST_FILENAME = "synthetic_timing_manifest.json"
RUNTIME_PROOF_FILENAME = "synthetic_timing_runtime_proof.json"
RECOMMENDATIONS_FILENAME = "synthetic_timing_recommendations.json"
BENCHMARK_DIRNAME = "benchmark"
TRAIN_BIN_RELATIVE = "dataset/ubc_train_shuffled.bin"
TRAIN_CSV_RELATIVE = "dataset/ubc_train_shuffled.csv"
VALIDATION_BIN_RELATIVE = "dataset/ubc_ocean_valid.bin"
VALIDATION_CSV_RELATIVE = "dataset/ubc_ocean_valid.csv"
SYNTHETIC_TIMING_MATRIX_COLUMNS = (
    "run_name",
    "benchmark_kind",
    "benchmark_source",
    "status_scope",
    "full_run_eligible",
    "row_id",
    "row_order",
    "timing_scope",
    "accelerator_mode",
    "machine_shape",
    "visible_device_count",
    "cuda_device_count",
    "gpu_names",
    "ddp_backend",
    "world_size",
    "nproc_per_node",
    "child_returncode",
    "ddp_torchrun_returncode",
    "ddp_rank_count",
    "ddp_rank_order",
    "ddp_rank_assignments_json",
    "per_device_batch_size",
    "global_batch_size",
    "split_patch_count",
    "non_wrapping_eligibility_steps",
    "non_wrapping_eligible",
    "fit_probe_only",
    "sample_reuse_count",
    "warmup_steps",
    "measured_steps",
    "batches_measured",
    "compile_startup_sec",
    "steady_step_ms_p50",
    "steady_step_ms_p95",
    "samples_sec",
    "loader_samples_sec",
    "h2d_ms_p50",
    "h2d_ms_p95",
    "max_vram_allocated_mb",
    "max_vram_reserved_mb",
    "vram_headroom_fraction",
    "real_train_patch_count",
    "drop_last",
    "steps_per_epoch",
    "effective_samples_per_epoch",
    "remainder_samples",
    "estimated_epoch_minutes",
    "generation_excluded_from_timing",
    "precision_policy",
    "amp_enabled",
    "torch_compile_enabled",
    "corruption_strategy",
    "status",
    "failure_kind",
    "failure_message_hash",
    "launch_command_hash",
    "cuda_visible_devices",
)
BLOCKED_CLAIM_KEYS = (
    "final_batch_size",
    "final_precision_policy",
    "final_corruption_strategy",
    "final_dataloader_settings",
    "final_single_vs_dual_t4",
    "real_data_loader_throughput",
    "convergence",
    "paper_evidence",
    "full_run_readiness",
)


@dataclass(frozen=True)
class SyntheticTimingProfile:
    """Generated UBC-format timing profile."""

    name: str
    train_patches: int
    validation_patches: int
    image_size: int
    channels: int
    seed: int
    write_chunk_patches: int

    @property
    def total_patches(self) -> int:
        """Return total train plus validation patches."""
        return self.train_patches + self.validation_patches

    @property
    def patch_payload_bytes(self) -> int:
        """Return payload bytes per patch."""
        return self.channels * self.image_size * self.image_size

    @property
    def total_payload_bytes(self) -> int:
        """Return total payload bytes before CSV and artifacts."""
        return self.total_patches * self.patch_payload_bytes


@dataclass(frozen=True)
class SyntheticTimingRequest:
    """Inputs for writing synthetic timing artifacts."""

    output_dir: Path
    run_name: str = "eqvae_synthetic_timing"
    profile: SyntheticTimingProfile | None = None
    local_upload_simulation: bool = False
    batch_sizes: tuple[int, ...] = DEFAULT_BATCH_SIZES
    row_specs: tuple[SyntheticTimingRowSpec, ...] | None = None
    warmup_steps: int = DEFAULT_WARMUP_STEPS
    measured_steps: int = DEFAULT_MEASURED_STEPS
    timing_phase: str = SYNTHETIC_TIMING_PHASE_BROAD_SCREEN
    payload_manifest: JsonObject | None = None
    kernel_metadata: JsonObject | None = None


@dataclass(frozen=True)
class SyntheticTimingRowSpec:
    """One explicit timing row to run."""

    accelerator_mode: str
    per_device_batch_size: int


@dataclass(frozen=True)
class SyntheticTimingArtifacts:
    """Paths written by the synthetic timing pretest."""

    manifest: Path
    runtime_proof: Path
    matrix: Path
    recommendations: Path


@dataclass(frozen=True)
class SplitWriteResult:
    """Generation and integrity proof for one split."""

    split: PatchSplit
    bin_path: Path
    csv_path: Path
    patch_count: int
    payload_bytes: int
    bin_size: int
    csv_size: int
    bin_sha256: str
    csv_sha256: str
    header_sha256: str
    crc32: int
    write_seconds: float


@dataclass(frozen=True)
class ChildRowConfig:
    """Serializable timing-row child-process configuration."""

    output_dir: Path
    data_root: Path
    row_id: str
    run_name: str
    accelerator_mode: str
    per_device_batch_size: int
    world_size: int
    nproc_per_node: int
    warmup_steps: int
    measured_steps: int
    split_patch_count: int
    non_wrapping_eligibility_steps: int
    real_train_patch_count: int
    image_size: int
    channels: int
    cuda_visible_devices: str
    precision_policy: str = "amp_off_fp32"
    torch_compile_enabled: bool = False
    corruption_strategy: str = "branchless_all"


@dataclass(frozen=True)
class ChildProcessArgs:
    """Typed child-process CLI arguments."""

    child_row: str | None
    ddp_rank_row: str | None


def default_synthetic_timing_profile() -> SyntheticTimingProfile:
    """Return the locked default roughly 2 GiB synthetic profile.

    Returns:
        Default synthetic timing profile.

    """
    return SyntheticTimingProfile(
        name=DEFAULT_PROFILE_NAME,
        train_patches=DEFAULT_SPLIT_PATCHES,
        validation_patches=DEFAULT_SPLIT_PATCHES,
        image_size=DEFAULT_IMAGE_SIZE,
        channels=DEFAULT_CHANNELS,
        seed=DEFAULT_SEED,
        write_chunk_patches=DEFAULT_WRITE_CHUNK_PATCHES,
    )


def compact_synthetic_timing_profile() -> SyntheticTimingProfile:
    """Return the historical 0.81 GB profile used by remote v1.

    Returns:
        Historical compact synthetic timing profile.

    """
    return SyntheticTimingProfile(
        name=COMPACT_PROFILE_NAME,
        train_patches=COMPACT_SPLIT_PATCHES,
        validation_patches=COMPACT_SPLIT_PATCHES,
        image_size=DEFAULT_IMAGE_SIZE,
        channels=DEFAULT_CHANNELS,
        seed=DEFAULT_SEED,
        write_chunk_patches=DEFAULT_WRITE_CHUNK_PATCHES,
    )


def repeat_shortlist_row_specs() -> tuple[SyntheticTimingRowSpec, ...]:
    """Return the explicit 5/25 repeat-shortlist candidate rows.

    Returns:
        Candidate rows chosen from the v3 broad-screening synthetic timing pass.

    """
    return (
        SyntheticTimingRowSpec(
            accelerator_mode="dual_t4_ddp",
            per_device_batch_size=8,
        ),
        SyntheticTimingRowSpec(
            accelerator_mode="single_visible_t4",
            per_device_batch_size=32,
        ),
        SyntheticTimingRowSpec(
            accelerator_mode="single_visible_t4",
            per_device_batch_size=12,
        ),
        SyntheticTimingRowSpec(
            accelerator_mode="single_visible_t4",
            per_device_batch_size=4,
        ),
    )


def _request_row_specs(
    request: SyntheticTimingRequest,
) -> tuple[SyntheticTimingRowSpec, ...]:
    if request.row_specs is not None:
        return request.row_specs
    return tuple(
        SyntheticTimingRowSpec(
            accelerator_mode=accelerator_mode,
            per_device_batch_size=per_device_batch_size,
        )
        for accelerator_mode in ("single_visible_t4", "dual_t4_ddp")
        for per_device_batch_size in request.batch_sizes
    )


def tiny_upload_simulation_profile() -> SyntheticTimingProfile:
    """Return a tiny profile for local single-file upload simulation tests.

    Returns:
        Tiny synthetic timing profile.

    """
    return SyntheticTimingProfile(
        name=TEST_PROFILE_NAME,
        train_patches=TEST_SPLIT_PATCHES,
        validation_patches=TEST_SPLIT_PATCHES,
        image_size=TEST_IMAGE_SIZE,
        channels=DEFAULT_CHANNELS,
        seed=DEFAULT_SEED,
        write_chunk_patches=4,
    )


def write_synthetic_timing_pretest(  # noqa: PLR0914
    request: SyntheticTimingRequest,
) -> SyntheticTimingArtifacts:
    """Generate synthetic shards, attempt timing rows, and write artifacts.

    Returns:
        Paths for the four synthetic timing artifacts.

    """
    profile = request.profile or default_synthetic_timing_profile()
    _validate_request(request=request, profile=profile)
    output_dir = request.output_dir.resolve()
    benchmark_dir = output_dir / BENCHMARK_DIRNAME
    data_root = output_dir / "synthetic_timing_data"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    free_before = _disk_free_bytes(output_dir)
    start_ns = time.perf_counter_ns()
    split_results = _write_synthetic_shards(data_root=data_root, profile=profile)
    generation_seconds = _elapsed_seconds(start_ns)
    free_after = _disk_free_bytes(output_dir)
    paths = resolve_patch_data_paths(data_root)
    matrix_rows = _run_timing_rows(
        request=request,
        profile=profile,
        data_root=data_root,
        output_dir=output_dir,
    )
    manifest_payload = _manifest_payload(
        request=request,
        profile=profile,
        data_root=data_root,
        paths=paths,
        split_results=split_results,
        generation_seconds=generation_seconds,
        free_disk_before=free_before,
        free_disk_after=free_after,
        rows=matrix_rows,
    )
    runtime_proof_payload = build_synthetic_timing_runtime_proof_payload(
        request=request,
        rows=matrix_rows,
    )
    recommendations_payload = build_synthetic_timing_recommendations_payload(
        request=request,
        profile=profile,
        rows=matrix_rows,
    )

    manifest_path = benchmark_dir / MANIFEST_FILENAME
    runtime_proof_path = benchmark_dir / RUNTIME_PROOF_FILENAME
    matrix_path = benchmark_dir / MATRIX_FILENAME
    recommendations_path = benchmark_dir / RECOMMENDATIONS_FILENAME
    write_json(manifest_path, manifest_payload)
    write_json(runtime_proof_path, runtime_proof_payload)
    write_csv(matrix_path, SYNTHETIC_TIMING_MATRIX_COLUMNS, matrix_rows)
    write_json(recommendations_path, recommendations_payload)
    return SyntheticTimingArtifacts(
        manifest=manifest_path,
        runtime_proof=runtime_proof_path,
        matrix=matrix_path,
        recommendations=recommendations_path,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run a timing child or DDP rank child process.

    Returns:
        Process exit status.

    Raises:
        ValueError: If no child mode is requested.

    """
    args = _parse_args(argv)
    if args.child_row is not None:
        row = _run_child_row(_decode_child_config(args.child_row))
        _write_stdout_json(row)
        return 0
    if args.ddp_rank_row is not None:
        _run_ddp_rank_row(_decode_child_config(args.ddp_rank_row))
        return 0
    msg = "synthetic_timing module is intended for child process execution"
    raise ValueError(msg)


def _validate_request(
    *,
    request: SyntheticTimingRequest,
    profile: SyntheticTimingProfile,
) -> None:
    if not request.output_dir.name:
        msg = "output_dir must be a concrete path"
        raise ValueError(msg)
    if profile.channels != DEFAULT_CHANNELS:
        msg = "synthetic timing profile must use RGB channels"
        raise ValueError(msg)
    if profile.train_patches <= 0 or profile.validation_patches <= 0:
        msg = "synthetic timing profile split patch counts must be positive"
        raise ValueError(msg)
    if profile.image_size <= 0:
        msg = "synthetic timing image_size must be positive"
        raise ValueError(msg)
    if not request.batch_sizes and request.row_specs is None:
        msg = "synthetic timing batch_sizes must not be empty without row_specs"
        raise ValueError(msg)
    if request.row_specs is not None and not request.row_specs:
        msg = "synthetic timing row_specs must not be empty when provided"
        raise ValueError(msg)
    for row_spec in request.row_specs or ():
        _row_runtime(row_spec.accelerator_mode)
        if row_spec.per_device_batch_size <= 0:
            msg = "synthetic timing row_specs batch sizes must be positive"
            raise ValueError(msg)


def _write_synthetic_shards(
    *,
    data_root: Path,
    profile: SyntheticTimingProfile,
) -> tuple[SplitWriteResult, SplitWriteResult]:
    if data_root.exists():
        shutil.rmtree(data_root)
    dataset_dir = data_root / "dataset"
    train = _write_split_shard(
        split="train",
        bin_path=dataset_dir / "ubc_train_shuffled.bin",
        csv_path=dataset_dir / "ubc_train_shuffled.csv",
        count=profile.train_patches,
        profile=profile,
        include_idx=False,
        seed_offset=0,
    )
    validation = _write_split_shard(
        split="validation",
        bin_path=dataset_dir / "ubc_ocean_valid.bin",
        csv_path=dataset_dir / "ubc_ocean_valid.csv",
        count=profile.validation_patches,
        profile=profile,
        include_idx=True,
        seed_offset=1_000_000,
    )
    return train, validation


def _write_split_shard(  # noqa: PLR0913
    *,
    split: PatchSplit,
    bin_path: Path,
    csv_path: Path,
    count: int,
    profile: SyntheticTimingProfile,
    include_idx: bool,
    seed_offset: int,
) -> SplitWriteResult:
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    start_ns = time.perf_counter_ns()
    crc32 = 0
    payload_bytes = 0
    with bin_path.open("wb") as binary_file:
        binary_file.write(bytes(PATCH_SHARD_HEADER_SIZE))
        for start_index in range(0, count, profile.write_chunk_patches):
            chunk_count = min(profile.write_chunk_patches, count - start_index)
            chunk = _make_histology_like_chunk(
                count=chunk_count,
                start_index=start_index,
                profile=profile,
                seed_offset=seed_offset,
            )
            payload = _tensor_bytes(chunk)
            binary_file.write(payload)
            crc32 = zlib.crc32(payload, crc32)
            payload_bytes += len(payload)
        crc32 &= 0xFFFFFFFF
        binary_file.seek(0)
        binary_file.write(
            make_patch_shard_header(
                header=PatchShardHeader(
                    crc32=crc32,
                    patch_count=count,
                    channels=profile.channels,
                    height=profile.image_size,
                    width=profile.image_size,
                    version=1,
                    layout=b"CHW",
                ),
            ),
        )
    _write_split_csv(csv_path=csv_path, count=count, include_idx=include_idx)
    with bin_path.open("rb") as binary_file:
        header_bytes = binary_file.read(PATCH_SHARD_HEADER_SIZE)
    header_sha = hashlib.sha256(header_bytes).hexdigest()
    return SplitWriteResult(
        split=split,
        bin_path=bin_path,
        csv_path=csv_path,
        patch_count=count,
        payload_bytes=payload_bytes,
        bin_size=bin_path.stat().st_size,
        csv_size=csv_path.stat().st_size,
        bin_sha256=_digest_file(bin_path),
        csv_sha256=_digest_file(csv_path),
        header_sha256=header_sha,
        crc32=crc32,
        write_seconds=_elapsed_seconds(start_ns),
    )


def _make_histology_like_chunk(  # noqa: PLR0914
    *,
    count: int,
    start_index: int,
    profile: SyntheticTimingProfile,
    seed_offset: int,
) -> Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(profile.seed + seed_offset + start_index)
    size = profile.image_size
    yy = torch.linspace(0.0, 1.0, size).view(1, 1, size, 1)
    xx = torch.linspace(0.0, 1.0, size).view(1, 1, 1, size)
    phase = torch.rand((count, 1, 1, 1), generator=generator) * (2.0 * math.pi)
    freq_x = torch.randint(2, 7, (count, 1, 1, 1), generator=generator).float()
    freq_y = torch.randint(2, 7, (count, 1, 1, 1), generator=generator).float()
    tissue = 0.5 + 0.5 * torch.sin((freq_x * xx + freq_y * yy) * math.pi + phase)
    texture = torch.rand((count, 1, size, size), generator=generator)
    hematoxylin = torch.clamp(0.65 * tissue + 0.35 * texture, 0.0, 1.0)
    eosin_noise = torch.rand((count, 1, size, size), generator=generator)
    eosin = torch.clamp(0.55 * (1.0 - tissue) + 0.45 * eosin_noise, 0.0, 1.0)
    background = torch.tensor([242.0, 226.0, 235.0]).view(1, 3, 1, 1)
    h_stain = torch.tensor([88.0, 96.0, 52.0]).view(1, 3, 1, 1)
    e_stain = torch.tensor([28.0, 78.0, 18.0]).view(1, 3, 1, 1)
    rgb = background - hematoxylin * h_stain + eosin * e_stain
    rgb += torch.rand((count, 3, size, size), generator=generator) * 7.0
    return torch.clamp(rgb, 0.0, 255.0).to(dtype=torch.uint8).contiguous()


def _tensor_bytes(tensor: Tensor) -> bytes:
    payload = bytearray(tensor.numel())
    view = torch.frombuffer(payload, dtype=torch.uint8)
    view.copy_(tensor.contiguous().view(-1))
    return bytes(payload)


def _write_split_csv(*, csv_path: Path, count: int, include_idx: bool) -> None:
    fieldnames = ["wsi_id", "label", "x", "y"]
    if include_idx:
        fieldnames.insert(0, "idx")
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for index in range(count):
            record = _synthetic_record(index)
            row = {
                "wsi_id": record.wsi_id,
                "label": str(record.label),
                "x": str(record.x),
                "y": str(record.y),
            }
            if include_idx:
                row["idx"] = str(record.file_index)
            writer.writerow(row)


def _synthetic_record(index: int) -> PatchRecord:
    return PatchRecord(
        file_index=index,
        row_index=index,
        wsi_id=f"synthetic_timing_wsi_{index // 16:05d}",
        label=index % 5,
        x=(index * 37) % 100_000,
        y=(index * 53) % 100_000,
    )


def _manifest_payload(  # noqa: PLR0913
    *,
    request: SyntheticTimingRequest,
    profile: SyntheticTimingProfile,
    data_root: Path,
    paths: PatchDataPaths,
    split_results: tuple[SplitWriteResult, SplitWriteResult],
    generation_seconds: float,
    free_disk_before: int,
    free_disk_after: int,
    rows: Sequence[CsvRow],
) -> JsonObject:
    train_proof = _split_proof(
        paths=paths,
        split="train",
        result=split_results[0],
        profile=profile,
    )
    validation_proof = _split_proof(
        paths=paths,
        split="validation",
        result=split_results[1],
        profile=profile,
    )
    loader_proof = _loader_proof(paths=paths, profile=profile)
    status = _timing_status(rows)
    return cast(
        "JsonObject",
        {
            "schema_version": SYNTHETIC_TIMING_SCHEMA_VERSION,
            "status": status,
            "status_scope": SYNTHETIC_TIMING_SCOPE,
            "benchmark_kind": SYNTHETIC_TIMING_KIND,
            "benchmark_source": SYNTHETIC_TIMING_SOURCE,
            "full_run_eligible": False,
            "blocked_claims": _blocked_claims(),
            "profile": {
                "name": profile.name,
                "pixel_profile": PIXEL_PROFILE_NAME,
                "seed": profile.seed,
                "total_patches": profile.total_patches,
                "train_patches": profile.train_patches,
                "validation_patches": profile.validation_patches,
                "channels": profile.channels,
                "image_size": profile.image_size,
                "payload_bytes": profile.total_payload_bytes,
                "patch_payload_bytes": profile.patch_payload_bytes,
                "data_generator_version": PIXEL_PROFILE_NAME,
                "named_profiles": {
                    "current_default": _profile_summary(
                        default_synthetic_timing_profile(),
                    ),
                    "historical_remote_v1": _profile_summary(
                        compact_synthetic_timing_profile(),
                    ),
                },
            },
            "data": {
                "origin": SYNTHETIC_TIMING_DATA_ORIGIN,
                "root": str(data_root),
                "resolved_root": str(paths.root),
                "train_bin": str(paths.train.bin_path),
                "train_csv": str(paths.train.csv_path),
                "validation_bin": str(paths.validation.bin_path),
                "validation_csv": str(paths.validation.csv_path),
                "relative_filenames": [
                    TRAIN_BIN_RELATIVE,
                    TRAIN_CSV_RELATIVE,
                    VALIDATION_BIN_RELATIVE,
                    VALIDATION_CSV_RELATIVE,
                ],
                "crc_validated": True,
                "generation_excluded_from_timing": True,
                "generation_seconds": generation_seconds,
                "write_throughput_mib_s": _mib_per_second(
                    bytes_count=profile.total_payload_bytes,
                    seconds=generation_seconds,
                ),
                "free_disk_before_bytes": free_disk_before,
                "free_disk_after_bytes": free_disk_after,
                "cache_state": "post_generation_hot_cache_biased",
                "local_upload_simulation": request.local_upload_simulation,
            },
            "kaggle_metadata": request.kernel_metadata,
            "timing_plan": _timing_plan_payload(request=request),
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
            "payload_manifest": request.payload_manifest,
            "splits": {
                "train": train_proof,
                "validation": validation_proof,
            },
            "loader_proof": loader_proof,
            "timing_row_summary": _timing_row_summary(rows),
        },
    )


def _split_proof(
    *,
    paths: PatchDataPaths,
    split: PatchSplit,
    result: SplitWriteResult,
    profile: SyntheticTimingProfile,
) -> JsonObject:
    split_paths = paths.for_split(split)
    shard = PatchShard(
        PatchShardSpec(
            bin_path=split_paths.bin_path,
            csv_path=split_paths.csv_path,
            image_size=profile.image_size,
            channels=profile.channels,
            validate_crc=True,
        ),
    )
    records = load_patch_records(split_paths.csv_path)
    semantic_keys = {semantic_key_for_record(record, split=split) for record in records}
    representative_records = (records[0], records[-1])
    return cast(
        "JsonObject",
        {
            "split": split,
            "bin_path": str(result.bin_path),
            "csv_path": str(result.csv_path),
            "patch_count": result.patch_count,
            "row_count": len(records),
            "payload_bytes": result.payload_bytes,
            "bin_file_size": result.bin_size,
            "csv_file_size": result.csv_size,
            "expected_bin_file_size": (
                PATCH_SHARD_HEADER_SIZE
                + result.patch_count * profile.patch_payload_bytes
            ),
            "bin_sha256": result.bin_sha256,
            "csv_sha256": result.csv_sha256,
            "header_sha256": result.header_sha256,
            "crc32": result.crc32,
            "computed_crc32": compute_patch_payload_crc(
                bin_path=split_paths.bin_path,
                header_size=PATCH_SHARD_HEADER_SIZE,
            ),
            "crc_validated": shard.crc_validated,
            "parsed_header": {
                "crc32": shard.header.crc32,
                "patch_count": shard.header.patch_count,
                "channels": shard.header.channels,
                "height": shard.header.height,
                "width": shard.header.width,
                "version": shard.header.version,
                "layout": shard.header.layout.decode("ascii"),
            },
            "csv_has_idx": split == "validation",
            "semantic_key_unique_count": len(semantic_keys),
            "semantic_key_uniqueness_pass": len(semantic_keys) == len(records),
            "representative_records": [
                _record_proof(record=record, split=split)
                for record in representative_records
            ],
            "write_seconds": result.write_seconds,
        },
    )


def _record_proof(*, record: PatchRecord, split: PatchSplit) -> JsonObject:
    sample_id = record.sample_id(split)
    semantic_key = semantic_key_for_record(record, split=split)
    return {
        "row_index": record.row_index,
        "file_index": record.file_index,
        "sample_id": sample_id,
        "sample_id_sha256": _hash_text(sample_id),
        "semantic_sample_key_sha256": _hash_text(semantic_key),
        "wsi_id": record.wsi_id,
        "label": record.label,
        "x": record.x,
        "y": record.y,
    }


def _profile_summary(profile: SyntheticTimingProfile) -> JsonObject:
    return {
        "name": profile.name,
        "total_patches": profile.total_patches,
        "train_patches": profile.train_patches,
        "validation_patches": profile.validation_patches,
        "channels": profile.channels,
        "image_size": profile.image_size,
        "payload_bytes": profile.total_payload_bytes,
        "patch_payload_bytes": profile.patch_payload_bytes,
    }


def _timing_plan_payload(*, request: SyntheticTimingRequest) -> JsonObject:
    row_specs = _request_row_specs(request)
    rows: list[JsonObject] = []
    for row_order, row_spec in enumerate(row_specs):
        world_size, cuda_mask = _row_runtime(row_spec.accelerator_mode)
        rows.append({
            "row_order": row_order,
            "row_id": _row_id(
                accelerator_mode=row_spec.accelerator_mode,
                per_device_batch_size=row_spec.per_device_batch_size,
            ),
            "accelerator_mode": row_spec.accelerator_mode,
            "per_device_batch_size": row_spec.per_device_batch_size,
            "world_size": world_size,
            "global_batch_size": row_spec.per_device_batch_size * world_size,
            "cuda_visible_devices": cuda_mask,
        })
    return cast(
        "JsonObject",
        {
            "timing_phase": request.timing_phase,
            "run_name": request.run_name,
            "warmup_steps": request.warmup_steps,
            "measured_steps": request.measured_steps,
            "explicit_row_specs": request.row_specs is not None,
            "batch_sizes": list(request.batch_sizes),
            "row_count": len(row_specs),
            "rows": rows,
        },
    )


def _loader_proof(
    *,
    paths: PatchDataPaths,
    profile: SyntheticTimingProfile,
) -> JsonObject:
    train_paths = paths.train
    tensor_dataset = PatchTensorDataset(
        PatchTensorDatasetSpec(
            bin_path=train_paths.bin_path,
            csv_path=train_paths.csv_path,
            split=train_paths.split,
            image_size=profile.image_size,
            channels=profile.channels,
            validate_crc=False,
        ),
    )
    training_dataset = PatchTrainingDataset(
        PatchTrainingDatasetSpec(
            bin_path=train_paths.bin_path,
            csv_path=train_paths.csv_path,
            split=train_paths.split,
            image_size=profile.image_size,
            channels=profile.channels,
            validate_crc=False,
        ),
    )
    try:
        tensor_sample = tensor_dataset[0]
        loader = DataLoader(
            training_dataset,
            batch_size=min(2, len(training_dataset)),
            shuffle=False,
            num_workers=0,
            collate_fn=collate_patch_training_samples,
        )
        batch = cast("PatchTrainingBatch", next(iter(loader)))
        normalized = normalize_uint8_batch(batch.images_uint8)
        return cast(
            "JsonObject",
            {
                "resolve_patch_data_paths_used": True,
                "tensor_dataset_class": "PatchTensorDataset",
                "training_dataset_class": "PatchTrainingDataset",
                "training_loader": (
                    "DataLoader(PatchTrainingDataset, "
                    "collate_fn=collate_patch_training_samples)"
                ),
                "normalizer": "normalize_uint8_batch",
                "tensor_sample": {
                    "dtype": str(tensor_sample.dtype),
                    "shape": list(tensor_sample.shape),
                    "file_index": tensor_dataset.file_index_for_row(0),
                },
                "collate_pre_normalization": {
                    "dtype": str(batch.images_uint8.dtype),
                    "shape": list(batch.images_uint8.shape),
                    "min": int(batch.images_uint8.min().item()),
                    "max": int(batch.images_uint8.max().item()),
                    "sample_ids": list(batch.sample_ids),
                    "sample_id_hashes": [
                        _hash_text(value) for value in batch.sample_ids
                    ],
                    "row_indices": list(batch.row_indices),
                    "file_indices": list(batch.file_indices),
                },
                "post_normalization": {
                    "dtype": str(normalized.dtype),
                    "shape": list(normalized.shape),
                    "min": float(normalized.min().item()),
                    "max": float(normalized.max().item()),
                    "range_pass": bool(
                        normalized.min().item() >= -1.0
                        and normalized.max().item() <= 1.0,
                    ),
                },
            },
        )
    finally:
        tensor_dataset.close()
        training_dataset.close()


def _run_timing_rows(
    *,
    request: SyntheticTimingRequest,
    profile: SyntheticTimingProfile,
    data_root: Path,
    output_dir: Path,
) -> list[CsvRow]:
    rows: list[CsvRow] = []
    for row_spec in _request_row_specs(request):
        world_size, cuda_mask = _row_runtime(row_spec.accelerator_mode)
        row_id = _row_id(
            accelerator_mode=row_spec.accelerator_mode,
            per_device_batch_size=row_spec.per_device_batch_size,
        )
        config = ChildRowConfig(
            output_dir=output_dir,
            data_root=data_root,
            row_id=row_id,
            run_name=request.run_name,
            accelerator_mode=row_spec.accelerator_mode,
            per_device_batch_size=row_spec.per_device_batch_size,
            world_size=world_size,
            nproc_per_node=world_size,
            warmup_steps=request.warmup_steps,
            measured_steps=request.measured_steps,
            split_patch_count=profile.train_patches,
            non_wrapping_eligibility_steps=NON_WRAPPING_ELIGIBILITY_STEPS,
            real_train_patch_count=REAL_TRAIN_PATCH_COUNT,
            image_size=profile.image_size,
            channels=profile.channels,
            cuda_visible_devices=cuda_mask,
        )
        row = dict(_run_row_child(config))
        row["row_order"] = str(len(rows))
        rows.append(row)
    return rows


def _run_row_child(config: ChildRowConfig) -> CsvRow:
    encoded_config = _encode_child_config(config)
    command = [
        sys.executable,
        "-m",
        "eqvae.benchmarking.synthetic_timing",
        "--child-row",
        encoded_config,
    ]
    environment = _child_environment(cuda_visible_devices=config.cuda_visible_devices)
    launch_hash = _hash_text(" ".join(command))
    try:
        completed = subprocess.run(  # noqa: S603
            command,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            cwd=str(config.output_dir),
            timeout=900,
        )
    except subprocess.TimeoutExpired as error:
        row = dict(
            _failure_row(
                config=config,
                status="runtime_error",
                failure_kind="child_timeout",
                failure_message=_exception_message(error),
                launch_command_hash=launch_hash,
            ),
        )
        row["child_returncode"] = "timeout"
        return row
    if completed.returncode != 0:
        row = dict(
            _failure_row(
                config=config,
                status="runtime_error",
                failure_kind="child_process_error",
                failure_message=completed.stderr[-1000:],
                launch_command_hash=launch_hash,
            ),
        )
        row["child_returncode"] = str(completed.returncode)
        return row
    try:
        row = cast("CsvRow", json.loads(completed.stdout))
    except json.JSONDecodeError:
        row = dict(
            _failure_row(
                config=config,
                status="runtime_error",
                failure_kind="child_output_decode_error",
                failure_message=completed.stdout[-1000:],
                launch_command_hash=launch_hash,
            ),
        )
        row["child_returncode"] = str(completed.returncode)
        return row
    mutable = dict(row)
    mutable["launch_command_hash"] = launch_hash
    mutable["child_returncode"] = str(completed.returncode)
    return mutable


def _run_child_row(config: ChildRowConfig) -> CsvRow:
    accelerator = _accelerator_observation()
    accelerator_failure = _accelerator_failure(config=config, accelerator=accelerator)
    if accelerator_failure is not None:
        return _failure_row(
            config=config,
            status=accelerator_failure[0],
            failure_kind=accelerator_failure[1],
            failure_message=accelerator_failure[2],
            accelerator=accelerator,
        )
    if config.accelerator_mode == "dual_t4_ddp":
        return _run_dual_row_with_torchrun(config=config, accelerator=accelerator)
    measurement = _measure_training_loader(config=config, rank=None, world_size=1)
    return _success_row(config=config, measurement=measurement, accelerator=accelerator)


def _run_dual_row_with_torchrun(
    *,
    config: ChildRowConfig,
    accelerator: JsonObject,
) -> CsvRow:
    with tempfile.TemporaryDirectory(
        prefix=f"eqvae_synthetic_timing_{config.row_id}_",
    ) as rank_temp_dir:
        rank_dir = Path(rank_temp_dir)
        encoded = _encode_child_config(config)
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            "-m",
            "eqvae.benchmarking.synthetic_timing",
            "--ddp-rank-row",
            encoded,
        ]
        environment = _child_environment(
            cuda_visible_devices=config.cuda_visible_devices,
        )
        environment["EQVAE_SYNTHETIC_TIMING_RANK_DIR"] = str(rank_dir)
        completed = subprocess.run(  # noqa: S603
            command,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            cwd=str(config.output_dir),
            timeout=900,
        )
        if completed.returncode != 0:
            return _failure_row(
                config=config,
                status="ddp_fail",
                failure_kind="torchrun_failed",
                failure_message=completed.stderr[-1000:],
                accelerator=accelerator,
                ddp_torchrun_returncode=str(completed.returncode),
            )
        rank_payloads = _load_rank_measurements(rank_dir=rank_dir, world_size=2)
        measurement = _aggregate_rank_measurements(
            rank_payloads=rank_payloads,
            torchrun_returncode=completed.returncode,
        )
    return _success_row(config=config, measurement=measurement, accelerator=accelerator)


def _run_ddp_rank_row(config: ChildRowConfig) -> None:
    import torch.distributed as dist  # noqa: PLC0415

    rank_dir = Path(os.environ["EQVAE_SYNTHETIC_TIMING_RANK_DIR"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    try:
        world_size = int(dist.get_world_size())
        measurement = _measure_training_loader(
            config=config,
            rank=rank,
            world_size=world_size,
            device_index=local_rank,
        )
        payload = {
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "measurement": measurement,
            "device_name": torch.cuda.get_device_name(local_rank),
        }
        write_json(rank_dir / f"rank_{rank}.json", cast("JsonObject", payload))
        barrier = cast("Callable[[], object]", dist.barrier)
        barrier()
    finally:
        dist.destroy_process_group()


def _measure_training_loader(
    *,
    config: ChildRowConfig,
    rank: int | None,
    world_size: int,
    device_index: int = 0,
) -> JsonObject:
    paths = resolve_patch_data_paths(config.data_root)
    train_paths = paths.train
    dataset = PatchTrainingDataset(
        PatchTrainingDatasetSpec(
            bin_path=train_paths.bin_path,
            csv_path=train_paths.csv_path,
            split=train_paths.split,
            image_size=config.image_size,
            channels=config.channels,
            validate_crc=False,
        ),
    )
    if rank is None:
        loader_dataset = dataset
    else:
        loader_dataset = Subset(dataset, range(rank, len(dataset), world_size))
    loader = cast(
        "DataLoader[PatchTrainingBatch]",
        DataLoader(
            loader_dataset,
            batch_size=config.per_device_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_patch_training_samples,
        ),
    )
    iterator = iter(loader)
    device = torch.device("cuda", device_index)
    step_ms: list[float] = []
    h2d_ms: list[float] = []
    samples = 0
    try:
        for _ in range(config.warmup_steps):
            _next_normalized_batch(iterator=iterator, device=device)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        for _ in range(config.measured_steps):
            start_ns = time.perf_counter_ns()
            h2d_start_ns = time.perf_counter_ns()
            batch = _next_normalized_batch(iterator=iterator, device=device)
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            h2d_ms.append(_elapsed_ms(h2d_start_ns))
            step_ms.append(_elapsed_ms(start_ns))
            samples += int(batch.shape[0])
    finally:
        del iterator
        del loader
        dataset.close()
    return cast(
        "JsonObject",
        {
            "step_ms": step_ms,
            "h2d_ms": h2d_ms,
            "samples": samples,
            "max_vram_allocated_mb": _cuda_allocated_mb(device),
            "max_vram_reserved_mb": _cuda_reserved_mb(device),
            "vram_headroom_fraction": _cuda_headroom_fraction(device),
        },
    )


def _next_normalized_batch(
    *,
    iterator: Iterator[PatchTrainingBatch],
    device: torch.device,
) -> Tensor:
    batch = next(iterator)
    normalized = normalize_uint8_batch(batch.images_uint8)
    return normalized.to(device=device, non_blocking=True)


def _success_row(
    *,
    config: ChildRowConfig,
    measurement: JsonObject,
    accelerator: JsonObject,
) -> CsvRow:
    step_ms = _float_list(measurement, "step_ms")
    h2d_ms = _float_list(measurement, "h2d_ms")
    global_batch_size = config.per_device_batch_size * config.world_size
    steady_p50 = _percentile(step_ms, 0.50)
    samples_sec = (
        0.0 if steady_p50 <= 0.0 else global_batch_size / (steady_p50 / 1000.0)
    )
    row = dict(_base_row(config=config, accelerator=accelerator))
    row.update({
        "batches_measured": str(len(step_ms)),
        "steady_step_ms_p50": _format_float(steady_p50),
        "steady_step_ms_p95": _format_float(_percentile(step_ms, 0.95)),
        "samples_sec": _format_float(samples_sec),
        "loader_samples_sec": _format_float(samples_sec),
        "h2d_ms_p50": _format_float(_percentile(h2d_ms, 0.50)),
        "h2d_ms_p95": _format_float(_percentile(h2d_ms, 0.95)),
        "max_vram_allocated_mb": _format_json_float(
            measurement,
            "max_vram_allocated_mb",
        ),
        "max_vram_reserved_mb": _format_json_float(
            measurement,
            "max_vram_reserved_mb",
        ),
        "vram_headroom_fraction": _format_json_float(
            measurement,
            "vram_headroom_fraction",
        ),
        "estimated_epoch_minutes": _format_float(
            _estimated_epoch_minutes(
                real_train_patch_count=config.real_train_patch_count,
                global_batch_size=global_batch_size,
                steady_step_ms_p50=steady_p50,
            ),
        ),
        "status": "pass",
        "failure_kind": "",
        "failure_message_hash": "",
    })
    if "ddp_rank_assignments" in measurement:
        row.update({
            "ddp_torchrun_returncode": str(
                _json_int(measurement, "ddp_torchrun_returncode"),
            ),
            "ddp_rank_count": str(_json_int(measurement, "ddp_rank_count")),
            "ddp_rank_order": json.dumps(
                _json_int_list(measurement, "ddp_rank_order"),
            ),
            "ddp_rank_assignments_json": json.dumps(
                _json_object_list(measurement, "ddp_rank_assignments"),
                sort_keys=True,
            ),
        })
    return row


def _failure_row(  # noqa: PLR0913
    *,
    config: ChildRowConfig,
    status: str,
    failure_kind: str,
    failure_message: str,
    accelerator: JsonObject | None = None,
    launch_command_hash: str = "",
    ddp_torchrun_returncode: str = "",
) -> CsvRow:
    observed_accelerator = accelerator or {
        "visible_device_count": 0,
        "cuda_device_count": 0,
        "gpu_names": [],
    }
    row = dict(_base_row(config=config, accelerator=observed_accelerator))
    row.update({
        "batches_measured": "0",
        "steady_step_ms_p50": "",
        "steady_step_ms_p95": "",
        "samples_sec": "",
        "loader_samples_sec": "",
        "h2d_ms_p50": "",
        "h2d_ms_p95": "",
        "max_vram_allocated_mb": "",
        "max_vram_reserved_mb": "",
        "vram_headroom_fraction": "",
        "estimated_epoch_minutes": "",
        "status": status,
        "failure_kind": failure_kind,
        "failure_message_hash": _hash_text(failure_message),
        "launch_command_hash": launch_command_hash,
        "ddp_torchrun_returncode": ddp_torchrun_returncode,
    })
    return row


def _base_row(*, config: ChildRowConfig, accelerator: JsonObject) -> CsvRow:
    global_batch_size = config.per_device_batch_size * config.world_size
    non_wrapping_eligible = (
        global_batch_size * config.non_wrapping_eligibility_steps
        <= config.split_patch_count
    )
    sample_reuse_count = max(
        0,
        global_batch_size * config.non_wrapping_eligibility_steps
        - config.split_patch_count,
    )
    steps_per_epoch = training_steps_per_epoch(
        real_train_patch_count=config.real_train_patch_count,
        global_batch_size=global_batch_size,
    )
    remainder_samples = config.real_train_patch_count % global_batch_size
    return {
        "run_name": config.run_name,
        "benchmark_kind": SYNTHETIC_TIMING_KIND,
        "benchmark_source": SYNTHETIC_TIMING_SOURCE,
        "status_scope": SYNTHETIC_TIMING_SCOPE,
        "full_run_eligible": "false",
        "row_id": config.row_id,
        "row_order": "",
        "timing_scope": "loader_collate_normalize_h2d_only",
        "accelerator_mode": config.accelerator_mode,
        "machine_shape": "NvidiaTeslaT4",
        "visible_device_count": str(_json_int(accelerator, "visible_device_count")),
        "cuda_device_count": str(_json_int(accelerator, "cuda_device_count")),
        "gpu_names": json.dumps(_json_str_list(accelerator, "gpu_names")),
        "ddp_backend": "nccl" if config.accelerator_mode == "dual_t4_ddp" else "",
        "world_size": str(config.world_size),
        "nproc_per_node": str(config.nproc_per_node),
        "child_returncode": "",
        "ddp_torchrun_returncode": "",
        "ddp_rank_count": "",
        "ddp_rank_order": "",
        "ddp_rank_assignments_json": "",
        "per_device_batch_size": str(config.per_device_batch_size),
        "global_batch_size": str(global_batch_size),
        "split_patch_count": str(config.split_patch_count),
        "non_wrapping_eligibility_steps": str(
            config.non_wrapping_eligibility_steps,
        ),
        "non_wrapping_eligible": _format_bool(value=non_wrapping_eligible),
        "fit_probe_only": _format_bool(value=not non_wrapping_eligible),
        "sample_reuse_count": str(sample_reuse_count),
        "warmup_steps": str(config.warmup_steps),
        "measured_steps": str(config.measured_steps),
        "compile_startup_sec": "0.000000",
        "real_train_patch_count": str(config.real_train_patch_count),
        "drop_last": "false",
        "steps_per_epoch": str(steps_per_epoch),
        "effective_samples_per_epoch": str(config.real_train_patch_count),
        "remainder_samples": str(remainder_samples),
        "generation_excluded_from_timing": "true",
        "precision_policy": config.precision_policy,
        "amp_enabled": "false",
        "torch_compile_enabled": _format_bool(value=config.torch_compile_enabled),
        "corruption_strategy": config.corruption_strategy,
        "launch_command_hash": "",
        "cuda_visible_devices": config.cuda_visible_devices,
    }


def _accelerator_observation() -> JsonObject:
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    names = [torch.cuda.get_device_name(index) for index in range(device_count)]
    return cast(
        "JsonObject",
        {
            "visible_device_count": device_count,
            "cuda_device_count": device_count,
            "gpu_names": names,
            "cuda_available": cuda_available,
        },
    )


def _accelerator_failure(
    *,
    config: ChildRowConfig,
    accelerator: JsonObject,
) -> tuple[str, str, str] | None:
    device_count = _json_int(accelerator, "cuda_device_count")
    gpu_names = _json_str_list(accelerator, "gpu_names")
    if config.accelerator_mode == "single_visible_t4":
        if device_count != SINGLE_T4_DEVICE_COUNT or not _all_t4(gpu_names):
            return (
                "wrong_accelerator",
                "wrong_accelerator",
                f"expected one visible T4, got count={device_count}, names={gpu_names}",
            )
        return None
    if device_count != DUAL_T4_DEVICE_COUNT or not _all_t4(gpu_names):
        return (
            "wrong_accelerator",
            "wrong_accelerator",
            (
                "expected two visible T4 devices, got "
                f"count={device_count}, names={gpu_names}"
            ),
        )
    return None


def build_synthetic_timing_runtime_proof_payload(
    *,
    request: SyntheticTimingRequest,
    rows: Sequence[CsvRow],
) -> JsonObject:
    """Build non-promotable runtime proof from synthetic timing matrix rows.

    Returns:
        JSON payload for `synthetic_timing_runtime_proof.json`.

    """
    single_rows = [
        row for row in rows if row["accelerator_mode"] == "single_visible_t4"
    ]
    dual_rows = [row for row in rows if row["accelerator_mode"] == "dual_t4_ddp"]
    return cast(
        "JsonObject",
        {
            "schema_version": SYNTHETIC_TIMING_SCHEMA_VERSION,
            "status": _timing_status(rows),
            "status_scope": SYNTHETIC_TIMING_SCOPE,
            "benchmark_kind": SYNTHETIC_TIMING_KIND,
            "benchmark_source": SYNTHETIC_TIMING_SOURCE,
            "full_run_eligible": False,
            "blocked_claims": _blocked_claims(),
            "kernel_metadata": request.kernel_metadata,
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
            "machine_shape": "NvidiaTeslaT4",
            "timing_plan": _timing_plan_payload(request=request),
            "accelerator_modes_checked": ["single_visible_t4", "dual_t4_ddp"],
            "single_visible_t4": _mode_summary(single_rows),
            "dual_t4_ddp": _mode_summary(dual_rows),
            "row_count": len(rows),
            "wrong_accelerator_row_count": sum(
                1 for row in rows if row["status"] == "wrong_accelerator"
            ),
            "skipped_unsupported_row_count": sum(
                1 for row in rows if row["status"] == "skipped_unsupported"
            ),
        },
    )


def build_synthetic_timing_recommendations_payload(
    *,
    request: SyntheticTimingRequest,
    profile: SyntheticTimingProfile,
    rows: Sequence[CsvRow],
) -> JsonObject:
    """Build ordered non-promotable recommendations from timing matrix rows.

    Returns:
        JSON payload for `synthetic_timing_recommendations.json`.

    """
    ordered_rows = sorted(rows, key=_recommendation_sort_key)
    repeat_required = request.timing_phase != SYNTHETIC_TIMING_PHASE_REPEAT_SHORTLIST
    recommendations = [
        _recommendation_for_row(
            row=row,
            rank=index + 1,
            repeat_required=repeat_required,
        )
        for index, row in enumerate(ordered_rows)
    ]
    return cast(
        "JsonObject",
        {
            "schema_version": SYNTHETIC_TIMING_SCHEMA_VERSION,
            "status": _timing_status(rows),
            "status_scope": SYNTHETIC_TIMING_SCOPE,
            "benchmark_kind": SYNTHETIC_TIMING_KIND,
            "benchmark_source": SYNTHETIC_TIMING_SOURCE,
            "full_run_eligible": False,
            "blocked_claims": _blocked_claims(),
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
            "profile_name": profile.name,
            "run_name": request.run_name,
            "timing_phase": request.timing_phase,
            "timing_plan": _timing_plan_payload(request=request),
            "recommendations": recommendations,
            "estimated_epoch_minutes_scope": (
                "loader_collate_normalize_h2d_only_projected_to_real_train_patch_count"
            ),
            "measured_components": {
                "loader_collate_normalize_h2d": True,
                "model_forward_backward": False,
                "optimizer_step": False,
                "corruption": False,
                "precision_policy": False,
                "torch_compile": False,
            },
            "ordering": (
                "Rows are ordered by promotability, lower loader/H2D-projected "
                "real epoch minutes, lower p95 step time, higher VRAM headroom, "
                "then row_id."
            ),
            "selection_policy": (
                "Synthetic timing may only carry, prune structural failures, "
                "mark fit probes, or request real-data confirmation; it never "
                "selects a runtime."
            ),
            "repeat_shortlist_policy": {
                "required_before_operational_shortlist": repeat_required,
                "completed": not repeat_required,
                "timing_phase": request.timing_phase,
                "warmup_steps": REPEAT_SHORTLIST_WARMUP_STEPS,
                "measured_steps": REPEAT_SHORTLIST_MEASURED_STEPS,
                "repeats": 1,
                "selection_requires_user_decision": True,
            },
            "interpretation_warning": (
                "Projected real epoch minutes are derived only from "
                "loader/collate/normalization/H2D timing. They do not measure "
                "model forward/backward, optimizer, corruption, AMP, compile, "
                "DDP gradient synchronization, convergence, or paper evidence."
            ),
        },
    )


def _recommendation_for_row(
    *,
    row: CsvRow,
    rank: int,
    repeat_required: bool,
) -> JsonObject:
    status = row["status"]
    fit_probe_only = row["fit_probe_only"] == "true"
    if status in {
        "wrong_accelerator",
        "oom",
        "ddp_fail",
        "compile_fail",
        "nonfinite_fail",
        "amp_skipped_fail",
    }:
        recommendation = "prune_obvious_failure"
    elif fit_probe_only:
        recommendation = "fit_probe_only"
    elif status == "pass":
        recommendation = "carry_to_real_benchmark"
    else:
        recommendation = "needs_real_data_confirmation"
    return {
        "row_id": row["row_id"],
        "recommendation_rank": rank,
        "accelerator_mode": row["accelerator_mode"],
        "per_device_batch_size": int(row["per_device_batch_size"]),
        "global_batch_size": int(row["global_batch_size"]),
        "non_wrapping_eligible": row["non_wrapping_eligible"] == "true",
        "fit_probe_only": fit_probe_only,
        "status": status,
        "recommendation": recommendation,
        "repeat_required_before_operational_shortlist": (
            repeat_required and recommendation == "carry_to_real_benchmark"
        ),
        "estimated_epoch_minutes": row["estimated_epoch_minutes"],
        "steady_step_ms_p95": row["steady_step_ms_p95"],
        "vram_headroom_fraction": row["vram_headroom_fraction"],
    }


def _recommendation_sort_key(row: CsvRow) -> tuple[float, float, float, float, str]:
    status = row["status"]
    fit_probe_only = row["fit_probe_only"] == "true"
    if status == "pass" and not fit_probe_only:
        group = 0.0
    elif status == "pass":
        group = 1.0
    elif status in {"wrong_accelerator", "oom", "ddp_fail"}:
        group = 2.0
    else:
        group = 3.0
    return (
        group,
        _csv_float_or_inf(row, "estimated_epoch_minutes"),
        _csv_float_or_inf(row, "steady_step_ms_p95"),
        -_csv_float_or_inf(row, "vram_headroom_fraction", missing=-math.inf),
        row["row_id"],
    )


def _timing_status(rows: Sequence[CsvRow]) -> str:
    if rows and all(row["status"] == "pass" for row in rows):
        return SYNTHETIC_TIMING_STATUS_PASS
    if any(row["status"] == "pass" for row in rows):
        return SYNTHETIC_TIMING_STATUS_PARTIAL
    return SYNTHETIC_TIMING_STATUS_SKIPPED


def _timing_row_summary(rows: Sequence[CsvRow]) -> JsonObject:
    return cast(
        "JsonObject",
        {
            "row_count": len(rows),
            "statuses": sorted({row["status"] for row in rows}),
            "pass_row_count": sum(1 for row in rows if row["status"] == "pass"),
            "non_wrapping_eligible_row_count": sum(
                1 for row in rows if row["non_wrapping_eligible"] == "true"
            ),
            "fit_probe_only_row_count": sum(
                1 for row in rows if row["fit_probe_only"] == "true"
            ),
            "max_sample_reuse_count": max(
                (int(row["sample_reuse_count"]) for row in rows),
                default=0,
            ),
            "accelerator_modes_checked": sorted(
                {row["accelerator_mode"] for row in rows},
            ),
        },
    )


def _mode_summary(rows: Sequence[CsvRow]) -> JsonObject:
    return cast(
        "JsonObject",
        {
            "attempted": bool(rows),
            "row_count": len(rows),
            "statuses": sorted({row["status"] for row in rows}),
            "row_ids_in_matrix_order": [row["row_id"] for row in rows],
            "child_returncodes": [
                {
                    "row_id": row["row_id"],
                    "row_order": _csv_int_or_none(row, "row_order"),
                    "status": row["status"],
                    "child_returncode": row["child_returncode"],
                }
                for row in rows
            ],
            "gpu_names": rows[0]["gpu_names"] if rows else "[]",
            "world_size": int(rows[0]["world_size"]) if rows else 0,
            "nproc_per_node": int(rows[0]["nproc_per_node"]) if rows else 0,
            "cuda_visible_devices": rows[0]["cuda_visible_devices"] if rows else "",
            "rank_assignment_rows": _rank_assignment_rows(rows),
        },
    )


def _load_rank_measurements(
    *,
    rank_dir: Path,
    world_size: int,
) -> tuple[JsonObject, ...]:
    payloads: list[JsonObject] = []
    for rank in range(world_size):
        path = rank_dir / f"rank_{rank}.json"
        payloads.append(
            cast("JsonObject", json.loads(path.read_text(encoding="utf-8"))),
        )
    return tuple(payloads)


def _aggregate_rank_measurements(
    *,
    rank_payloads: Sequence[JsonObject],
    torchrun_returncode: int,
) -> JsonObject:
    ordered_payloads = sorted(
        rank_payloads,
        key=lambda payload: _json_int(payload, "rank"),
    )
    measurements = [
        _required_object(payload, "measurement") for payload in ordered_payloads
    ]
    step_lists = [_float_list(measurement, "step_ms") for measurement in measurements]
    h2d_lists = [_float_list(measurement, "h2d_ms") for measurement in measurements]
    step_count = min(len(values) for values in step_lists)
    step_ms = [
        max(values[index] for values in step_lists) for index in range(step_count)
    ]
    h2d_ms = [max(values[index] for values in h2d_lists) for index in range(step_count)]
    return cast(
        "JsonObject",
        {
            "step_ms": step_ms,
            "h2d_ms": h2d_ms,
            "samples": sum(
                _json_int(measurement, "samples") for measurement in measurements
            ),
            "max_vram_allocated_mb": max(
                _json_float(measurement, "max_vram_allocated_mb")
                for measurement in measurements
            ),
            "max_vram_reserved_mb": max(
                _json_float(measurement, "max_vram_reserved_mb")
                for measurement in measurements
            ),
            "vram_headroom_fraction": min(
                _json_float(measurement, "vram_headroom_fraction")
                for measurement in measurements
            ),
            "ddp_torchrun_returncode": torchrun_returncode,
            "ddp_rank_count": len(ordered_payloads),
            "ddp_rank_order": [
                _json_int(payload, "rank") for payload in ordered_payloads
            ],
            "ddp_rank_assignments": [
                _rank_assignment(payload) for payload in ordered_payloads
            ],
        },
    )


def _rank_assignment(payload: JsonObject) -> JsonObject:
    measurement = _required_object(payload, "measurement")
    return {
        "rank": _json_int(payload, "rank"),
        "local_rank": _json_int(payload, "local_rank"),
        "world_size": _json_int(payload, "world_size"),
        "device_name": _required_str(payload, "device_name"),
        "samples": _json_int(measurement, "samples"),
        "measured_batches": len(_float_list(measurement, "step_ms")),
    }


def _rank_assignment_rows(rows: Sequence[CsvRow]) -> list[JsonObject]:
    proofs: list[JsonObject] = []
    for row in rows:
        assignments_json = row["ddp_rank_assignments_json"]
        if not assignments_json:
            continue
        proofs.append({
            "row_id": row["row_id"],
            "row_order": _csv_int_or_none(row, "row_order"),
            "status": row["status"],
            "torchrun_returncode": _csv_int_or_none(
                row,
                "ddp_torchrun_returncode",
            ),
            "rank_count": _csv_int_or_none(row, "ddp_rank_count"),
            "rank_order": _json_load_list(row["ddp_rank_order"]),
            "rank_assignments": _json_load_list(assignments_json),
        })
    return proofs


def _parse_args(argv: Sequence[str] | None) -> ChildProcessArgs:
    parser = argparse.ArgumentParser(description="Synthetic timing child helper.")
    parser.add_argument("--child-row")
    parser.add_argument("--ddp-rank-row")
    namespace = parser.parse_args(argv)
    return ChildProcessArgs(
        child_row=_optional_arg_str(namespace, "child_row"),
        ddp_rank_row=_optional_arg_str(namespace, "ddp_rank_row"),
    )


def _optional_arg_str(namespace: argparse.Namespace, key: str) -> str | None:
    value = cast("object", getattr(namespace, key))
    if value is None or isinstance(value, str):
        return value
    msg = f"Expected optional string argument {key!r}"
    raise TypeError(msg)


def _encode_child_config(config: ChildRowConfig) -> str:
    payload = {
        "output_dir": str(config.output_dir),
        "data_root": str(config.data_root),
        "row_id": config.row_id,
        "run_name": config.run_name,
        "accelerator_mode": config.accelerator_mode,
        "per_device_batch_size": config.per_device_batch_size,
        "world_size": config.world_size,
        "nproc_per_node": config.nproc_per_node,
        "warmup_steps": config.warmup_steps,
        "measured_steps": config.measured_steps,
        "split_patch_count": config.split_patch_count,
        "non_wrapping_eligibility_steps": config.non_wrapping_eligibility_steps,
        "real_train_patch_count": config.real_train_patch_count,
        "image_size": config.image_size,
        "channels": config.channels,
        "cuda_visible_devices": config.cuda_visible_devices,
        "precision_policy": config.precision_policy,
        "torch_compile_enabled": config.torch_compile_enabled,
        "corruption_strategy": config.corruption_strategy,
    }
    return base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")


def _decode_child_config(encoded: str) -> ChildRowConfig:
    payload = cast(
        "Mapping[str, object]",
        json.loads(base64.urlsafe_b64decode(encoded.encode("ascii"))),
    )
    return ChildRowConfig(
        output_dir=Path(_required_str(payload, "output_dir")),
        data_root=Path(_required_str(payload, "data_root")),
        row_id=_required_str(payload, "row_id"),
        run_name=_required_str(payload, "run_name"),
        accelerator_mode=_required_str(payload, "accelerator_mode"),
        per_device_batch_size=_required_int(payload, "per_device_batch_size"),
        world_size=_required_int(payload, "world_size"),
        nproc_per_node=_required_int(payload, "nproc_per_node"),
        warmup_steps=_required_int(payload, "warmup_steps"),
        measured_steps=_required_int(payload, "measured_steps"),
        split_patch_count=_required_int(payload, "split_patch_count"),
        non_wrapping_eligibility_steps=_required_int(
            payload,
            "non_wrapping_eligibility_steps",
        ),
        real_train_patch_count=_required_int(payload, "real_train_patch_count"),
        image_size=_required_int(payload, "image_size"),
        channels=_required_int(payload, "channels"),
        cuda_visible_devices=_required_str(payload, "cuda_visible_devices"),
        precision_policy=_required_str(payload, "precision_policy"),
        torch_compile_enabled=_required_bool(payload, "torch_compile_enabled"),
        corruption_strategy=_required_str(payload, "corruption_strategy"),
    )


def _row_id(*, accelerator_mode: str, per_device_batch_size: int) -> str:
    return (
        f"{accelerator_mode}__bs{per_device_batch_size}"
        "__amp_off_fp32__compile_off__branchless_all"
    )


def _row_runtime(accelerator_mode: str) -> tuple[int, str]:
    if accelerator_mode == "single_visible_t4":
        return SINGLE_T4_DEVICE_COUNT, "0"
    if accelerator_mode == "dual_t4_ddp":
        return DUAL_T4_DEVICE_COUNT, "0,1"
    msg = f"unsupported synthetic timing accelerator_mode: {accelerator_mode}"
    raise ValueError(msg)


def _blocked_claims() -> JsonObject:
    return dict.fromkeys(BLOCKED_CLAIM_KEYS, True)


def _all_t4(gpu_names: Sequence[str]) -> bool:
    return bool(gpu_names) and all("T4" in name for name in gpu_names)


def _child_environment(*, cuda_visible_devices: str) -> dict[str, str]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    inherited_pythonpath = environment.get("PYTHONPATH")
    path_entries = [entry for entry in sys.path if entry]
    if inherited_pythonpath:
        path_entries.append(inherited_pythonpath)
    environment["PYTHONPATH"] = os.pathsep.join(path_entries)
    return environment


def _estimated_epoch_minutes(
    *,
    real_train_patch_count: int,
    global_batch_size: int,
    steady_step_ms_p50: float,
) -> float:
    steps_per_epoch = training_steps_per_epoch(
        real_train_patch_count=real_train_patch_count,
        global_batch_size=global_batch_size,
    )
    return steps_per_epoch * steady_step_ms_p50 / 60_000.0


def _disk_free_bytes(path: Path) -> int:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free


def _digest_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _mib_per_second(*, bytes_count: int, seconds: float) -> float:
    if seconds <= 0.0:
        return 0.0
    return bytes_count / (1024.0 * 1024.0) / seconds


def _csv_float_or_inf(
    row: CsvRow,
    key: str,
    *,
    missing: float = math.inf,
) -> float:
    value = row[key]
    if not value:
        return missing
    return float(value)


def _csv_int_or_none(row: CsvRow, key: str) -> int | None:
    value = row[key]
    if not value:
        return None
    return int(value)


def _json_load_list(value: str) -> list[JsonValue]:
    loaded = cast("object", json.loads(value))
    if isinstance(loaded, list):
        return cast("list[JsonValue]", loaded)
    msg = "Expected JSON list encoded in CSV field"
    raise TypeError(msg)


def _elapsed_seconds(start_ns: int) -> float:
    return (time.perf_counter_ns() - start_ns) / 1_000_000_000.0


def _elapsed_ms(start_ns: int) -> float:
    return (time.perf_counter_ns() - start_ns) / 1_000_000.0


def _exception_message(error: BaseException) -> str:
    return "".join(traceback.format_exception_only(type(error), error)).strip()


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        msg = "Cannot compute percentile of an empty sequence"
        raise ValueError(msg)
    sorted_values = sorted(values)
    index = round((len(sorted_values) - 1) * quantile)
    return sorted_values[index]


def _cuda_allocated_mb(device: torch.device) -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)


def _cuda_reserved_mb(device: torch.device) -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_reserved(device) / (1024.0 * 1024.0)


def _cuda_headroom_fraction(device: torch.device) -> float:
    if not torch.cuda.is_available():
        return 0.0
    get_properties = cast(
        "Callable[[torch.device], _CudaDeviceProperties]",
        torch.cuda.get_device_properties,
    )
    properties = get_properties(device)
    total = float(properties.total_memory)
    reserved = float(torch.cuda.max_memory_reserved(device))
    if total <= 0.0:
        return 0.0
    return max(0.0, (total - reserved) / total)


def _format_float(value: float) -> str:
    return f"{value:.6f}"


def _format_json_float(payload: JsonObject, key: str) -> str:
    return _format_float(_json_float(payload, key))


def _format_bool(*, value: bool) -> str:
    return "true" if value else "false"


def _float_list(payload: JsonObject, key: str) -> list[float]:
    value = payload.get(key)
    if not isinstance(value, list):
        msg = f"Expected list field {key!r}"
        raise TypeError(msg)
    result: list[float] = []
    for item in value:
        if isinstance(item, int | float) and not isinstance(item, bool):
            result.append(float(item))
        else:
            msg = f"Expected numeric item in list field {key!r}"
            raise TypeError(msg)
    return result


def _json_float(payload: JsonObject, key: str) -> float:
    value = payload.get(key)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    msg = f"Expected numeric field {key!r}"
    raise TypeError(msg)


def _json_int(payload: JsonObject, key: str) -> int:
    value = payload.get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    msg = f"Expected integer field {key!r}"
    raise TypeError(msg)


def _json_str_list(payload: JsonObject, key: str) -> list[str]:
    value = payload.get(key)
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return [cast("str", item) for item in value]
    msg = f"Expected string-list field {key!r}"
    raise TypeError(msg)


def _json_int_list(payload: JsonObject, key: str) -> list[int]:
    value = payload.get(key)
    if isinstance(value, list) and all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        return [cast("int", item) for item in value]
    msg = f"Expected integer-list field {key!r}"
    raise TypeError(msg)


def _json_object_list(payload: JsonObject, key: str) -> list[JsonObject]:
    value = payload.get(key)
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        return [cast("JsonObject", item) for item in value]
    msg = f"Expected object-list field {key!r}"
    raise TypeError(msg)


def _required_object(payload: JsonObject, key: str) -> JsonObject:
    value = payload.get(key)
    if isinstance(value, dict):
        return cast("JsonObject", value)
    msg = f"Expected object field {key!r}"
    raise TypeError(msg)


def _required_str(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if isinstance(value, str):
        return value
    msg = f"Expected string field {key!r}"
    raise TypeError(msg)


def _required_int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    msg = f"Expected integer field {key!r}"
    raise TypeError(msg)


def _required_bool(payload: Mapping[str, object], key: str) -> bool:
    value = payload.get(key)
    if isinstance(value, bool):
        return value
    msg = f"Expected boolean field {key!r}"
    raise TypeError(msg)


def _write_stdout_json(payload: CsvRow) -> None:
    sys.stdout.write(json.dumps(dict(payload), sort_keys=True))


__all__ = [
    "BLOCKED_CLAIM_KEYS",
    "COMPACT_PROFILE_NAME",
    "DEFAULT_PROFILE_NAME",
    "MANIFEST_FILENAME",
    "MATRIX_FILENAME",
    "RECOMMENDATIONS_FILENAME",
    "REPEAT_SHORTLIST_MEASURED_STEPS",
    "REPEAT_SHORTLIST_WARMUP_STEPS",
    "RUNTIME_PROOF_FILENAME",
    "SYNTHETIC_TIMING_KIND",
    "SYNTHETIC_TIMING_MATRIX_COLUMNS",
    "SYNTHETIC_TIMING_PHASE_BROAD_SCREEN",
    "SYNTHETIC_TIMING_PHASE_REPEAT_SHORTLIST",
    "SYNTHETIC_TIMING_SCOPE",
    "SYNTHETIC_TIMING_SOURCE",
    "SYNTHETIC_TIMING_STATUS_PARTIAL",
    "SYNTHETIC_TIMING_STATUS_PASS",
    "SyntheticTimingArtifacts",
    "SyntheticTimingProfile",
    "SyntheticTimingRequest",
    "SyntheticTimingRowSpec",
    "build_synthetic_timing_recommendations_payload",
    "build_synthetic_timing_runtime_proof_payload",
    "compact_synthetic_timing_profile",
    "default_synthetic_timing_profile",
    "repeat_shortlist_row_specs",
    "tiny_upload_simulation_profile",
    "write_synthetic_timing_pretest",
]


if __name__ == "__main__":
    raise SystemExit(main())
