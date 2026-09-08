# Copyright 2026 HiperMaximus
"""Fixed-record FP32 posterior-mean shards with exact resumability."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import mmap
import os
import struct
import sys
import warnings
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Final, Literal, Self, cast

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

type ModelName = Literal["normal_vae", "so2_vae"]
type MmapAdvice = Literal["sequential", "random", "none"]
type TaskName = Literal["cancer", "tissue"]
type SplitName = Literal["train", "validation", "test"]

LATENT_SHARD_MAGIC: Final = b"EQV_LATN"
LATENT_SHARD_HEADER_FORMAT: Final = "<8sIQiiii4s3s21x"
LATENT_SHARD_HEADER_SIZE: Final = 64
LATENT_SHARD_VERSION: Final = 1
LATENT_SHARD_DTYPE: Final = b"F32L"
LATENT_SHARD_LAYOUT: Final = b"CHW"
LATENT_CHANNELS: Final = 16
LATENT_HEIGHT: Final = 32
LATENT_WIDTH: Final = 32
LATENT_VALUES: Final = LATENT_CHANNELS * LATENT_HEIGHT * LATENT_WIDTH
LATENT_RECORD_BYTES: Final = LATENT_VALUES * 4
HASH_CHUNK_BYTES: Final = 8 * 1024 * 1024
SHA256_HEX_LENGTH: Final = 64
TENSOR_BATCH_NDIM: Final = 4
UNION_HEADER: Final = (
    "atlas_row_index",
    "wsi_id",
    "diagnosis_label",
    "diagnosis_index",
    "x",
    "y",
    "split",
    "cancer_ae_selected",
    "tissue_selected",
    "tissue_label",
)
COORDINATE_ONLY_HEADER: Final = ("atlas_row_index", "wsi_id", "x", "y")
CANCER_TASK_HEADER: Final = (
    "atlas_row_index",
    "wsi_id",
    "diagnosis_label",
    "diagnosis_index",
    "x",
    "y",
    "split",
)
TISSUE_TASK_HEADER: Final = (
    *CANCER_TASK_HEADER,
    "tissue_label",
    "annotated_fraction",
    "dominant_fraction",
    "purity",
    "tumor_fraction",
    "stroma_fraction",
    "necrosis_fraction",
)
LATENT_LOCATION_HEADER: Final = (
    "run_number",
    "file_index",
    "atlas_row_index",
    "wsi_id",
    "x",
    "y",
    "split",
    "diagnosis_label",
    "diagnosis_index",
    "tissue_label",
)
TASK_NAMES: Final[tuple[TaskName, ...]] = ("cancer", "tissue")
SPLIT_NAMES: Final[tuple[SplitName, ...]] = ("train", "validation", "test")
EXPECTED_TASK_MANIFEST_SHA256: Final = {
    "cancer_train": (
        "03e8e0d7aefb8d30d049c8521c44ebc88ca3709ea26a65765b772df2b3af65fb"
    ),
    "cancer_validation": (
        "3d96c739320ab1facbdf2bb93b18fc8d2c951a7f30a3d61f5ce3587e4c402ce5"
    ),
    "cancer_test": ("1d0e4059f469d350ff3960cc10208221548f6afdfc1788e40e6d5da7829806cc"),
    "tissue_train": (
        "fe81ea26956a2d0c15625992c8604a23194325ce4668595b71102110bfcbcca9"
    ),
    "tissue_validation": (
        "092671a60ea271e28db83b5783e07259588d3df826863f273d2967b8031321f2"
    ),
    "tissue_test": ("5f02c70025e4e1f4b8839460b23bd5575f1dd57ac4a0225e662669cb9b8b310c"),
}
EXPECTED_TASK_VIEW_ROWS: Final = {
    "cancer_train": 314_755,
    "cancer_validation": 68_045,
    "cancer_test": 67_138,
    "tissue_train": 145_215,
    "tissue_validation": 31_339,
    "tissue_test": 31_572,
}
EXPECTED_WORK_MANIFEST_SHA256: Final = {
    1: "76cc5f9b86b75b9e46250c80e7b5c98d2f0451a12b38665ae45b055767b7a456",
    2: "11d21482c9d3e083bc5138973c6b09f6b659e7d753853c0290382320e78e6200",
    3: "9414f638abc24963e821de8bbedccaf294a14a7ea768a01687d5b105e634402b",
    4: "e7c5d8d08996e3bac440b5779547b2bbeb4443e7481dcb74998a87aa3054697f",
    5: "5485c44ababeaba9a4e2cc75a1a6b2927d89f10ed83a121dfcb81a68cfc23d6a",
}
EXPECTED_CHECKPOINT_SHA256: Final[dict[ModelName, str]] = {
    "normal_vae": "f733304e9178e468546113642bdf01e11348570b340c366cf148973083cb9075",
    "so2_vae": "041e0cd7483cb8642bb72eb1b63c3a36774bf9cadd0b659c9d1db6a813c8f4c7",
}
EXPECTED_UNION_SHA256: Final = (
    "f92558fa7aced13debc839c733e2c96d03c1a3df4b2a0b194b60558c69c04012"
)
EXPECTED_UNION_ROWS: Final = 599_398
EXPECTED_CANCER_ROWS: Final = 449_938
EXPECTED_TISSUE_ROWS: Final = 208_126
EXPECTED_SHARED_ROWS: Final = 58_666

if struct.calcsize(LATENT_SHARD_HEADER_FORMAT) != LATENT_SHARD_HEADER_SIZE:
    message = "Latent-shard header must stay exactly 64 bytes"
    raise RuntimeError(message)


@dataclass(frozen=True)
class LatentRowIdentity:
    """Storage identity carried from one exact union-manifest row."""

    atlas_row_index: int
    wsi_id: int
    x: int
    y: int


@dataclass(frozen=True)
class LatentShardHeader:
    """Parsed fixed-size latent header."""

    payload_crc32: int
    tensor_count: int
    channels: int
    height: int
    width: int
    version: int
    dtype: bytes
    layout: bytes


@dataclass(frozen=True)
class WorkManifestRow:
    """One validated row from a Spec 0019 work manifest."""

    identity: LatentRowIdentity
    split: SplitName
    diagnosis_label: str
    diagnosis_index: int
    tissue_label: str
    cancer_selected: bool
    tissue_selected: bool


@dataclass(frozen=True)
class WorkManifest:
    """Hash-bound work manifest with exact WSI row boundaries."""

    path: Path
    sha256: str
    run_number: int
    rows: tuple[WorkManifestRow, ...]
    wsi_ranges: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class LatentTaskLocation:
    """One canonical logical-view row resolving into a physical shard."""

    run_number: int
    file_index: int
    identity: LatentRowIdentity
    split: SplitName
    diagnosis_label: str
    diagnosis_index: int
    tissue_label: str


@dataclass(frozen=True)
class LatentArtifact:
    """Validated completed artifact and its exact source-manifest binding."""

    bin_path: Path
    sidecar_path: Path
    manifest: WorkManifest
    header: LatentShardHeader
    sidecar: dict[str, object]
    file_sha256: str | None


def make_latent_shard_header(*, tensor_count: int, payload_crc32: int) -> bytes:
    """Pack the one supported 64-byte latent header.

    Returns:
        Exact little-endian header bytes.

    Raises:
        ValueError: If the tensor count is negative.

    """
    if tensor_count < 0:
        message = f"Negative tensor count {tensor_count}"
        raise ValueError(message)
    return struct.pack(
        LATENT_SHARD_HEADER_FORMAT,
        LATENT_SHARD_MAGIC,
        payload_crc32 & 0xFFFFFFFF,
        tensor_count,
        LATENT_CHANNELS,
        LATENT_HEIGHT,
        LATENT_WIDTH,
        LATENT_SHARD_VERSION,
        LATENT_SHARD_DTYPE,
        LATENT_SHARD_LAYOUT,
    )


def parse_latent_shard_header(header_bytes: bytes) -> LatentShardHeader:
    """Parse and reject every header outside the single supported contract.

    Returns:
        Validated header fields.

    Raises:
        ValueError: If bytes do not match the fixed latent contract.

    """
    if len(header_bytes) != LATENT_SHARD_HEADER_SIZE:
        message = f"Expected {LATENT_SHARD_HEADER_SIZE}-byte latent header"
        raise ValueError(message)
    unpacked = cast(
        "tuple[bytes, int, int, int, int, int, int, bytes, bytes]",
        struct.unpack(LATENT_SHARD_HEADER_FORMAT, header_bytes),
    )
    magic, crc, count, channels, height, width, version, dtype, layout = unpacked
    expected = (
        LATENT_SHARD_MAGIC,
        LATENT_CHANNELS,
        LATENT_HEIGHT,
        LATENT_WIDTH,
        LATENT_SHARD_VERSION,
        LATENT_SHARD_DTYPE,
        LATENT_SHARD_LAYOUT,
    )
    observed = (magic, channels, height, width, version, dtype, layout)
    if observed != expected:
        message = f"Unsupported latent header contract: {observed!r}"
        raise ValueError(message)
    return LatentShardHeader(
        payload_crc32=crc,
        tensor_count=count,
        channels=channels,
        height=height,
        width=width,
        version=version,
        dtype=dtype,
        layout=layout,
    )


class LatentShardWriter:
    """Append exact manifest rows and commit only complete WSI boundaries."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        bin_path: Path,
        manifest_path: Path,
        run_number: int,
        model_name: ModelName,
        checkpoint_sha256: str,
        expected_checkpoint_sha256: str,
        expected_manifest_sha256: str,
        expected_union_sha256: str,
        coordinate_only_manifest: bool = False,
    ) -> None:
        """Bind trusted provenance and recover or create one shard."""
        _validate_provenance(
            model_name,
            checkpoint_sha256,
            expected_checkpoint_sha256,
            expected_manifest_sha256,
            expected_union_sha256,
        )
        self.bin_path = bin_path
        self.sidecar_path = bin_path.with_suffix(".json")
        self.partial_path = Path(f"{bin_path}.partial")
        self.state_path = bin_path.with_suffix(".resume.json")
        self.manifest = load_work_manifest(
            manifest_path,
            run_number=run_number,
            expected_sha256=expected_manifest_sha256,
            require_run_basename=not coordinate_only_manifest,
        )
        self.model_name: ModelName = model_name
        self.checkpoint_sha256 = checkpoint_sha256
        self.expected_checkpoint_sha256 = expected_checkpoint_sha256
        self.expected_union_sha256 = expected_union_sha256
        self.coordinate_only_manifest = coordinate_only_manifest
        self._handle: BinaryIO | None = None
        self._written_rows = 0
        self._committed_rows = 0
        self._completed_wsi_ids: list[int] = []
        self._crc32 = 0
        self._sha256 = hashlib.sha256()
        self._complete = False
        self._complete_file_sha256: str | None = None
        bin_path.parent.mkdir(parents=True, exist_ok=True)
        self._recover_or_create()

    @property
    def complete(self) -> bool:
        """Whether the validated completion sidecar has been published."""
        return self._complete

    @property
    def next_row_start(self) -> int:
        """The exact row index expected by the next append.

        Returns:
            Number of payload rows currently written in this process.

        """
        return self._written_rows

    @property
    def committed_rows(self) -> int:
        """The durable row prefix ending at a WSI boundary."""
        return self._committed_rows

    @property
    def completed_wsi_ids(self) -> tuple[int, ...]:
        """The durable, manifest-ordered WSI prefix."""
        return tuple(self._completed_wsi_ids)

    def append_batch(
        self,
        *,
        row_start: int,
        identities: Sequence[LatentRowIdentity],
        tensors: Tensor,
    ) -> None:
        """Append one contiguous batch wholly contained in the active WSI.

        Raises:
            RuntimeError: If the shard is already complete.
            ValueError: If tensors or identities violate the next manifest slice.

        """
        if self._complete:
            message = "Cannot append to a completed latent shard"
            raise RuntimeError(message)
        if row_start != self._written_rows:
            message = f"Expected row_start {self._written_rows}, got {row_start}"
            raise ValueError(message)
        batch_count = len(identities)
        if (
            batch_count < 1
            or tensors.ndim != TENSOR_BATCH_NDIM
            or tensors.shape[0] != batch_count
        ):
            message = "Identity and tensor batch lengths must be equal and nonzero"
            raise ValueError(message)
        end = row_start + batch_count
        if end > len(self.manifest.rows):
            message = "Latent batch overruns the source manifest"
            raise ValueError(message)
        expected = tuple(row.identity for row in self.manifest.rows[row_start:end])
        if tuple(identities) != expected:
            message = "Latent batch identities do not match the next manifest slice"
            raise ValueError(message)
        wsi_id = expected[0].wsi_id
        if any(identity.wsi_id != wsi_id for identity in expected):
            message = "One append batch cannot cross a WSI boundary"
            raise ValueError(message)
        wsi_start, wsi_end = self._range_for_row(row_start)
        if row_start < wsi_start or end > wsi_end:
            message = "Latent append does not stay inside the active WSI"
            raise ValueError(message)
        payload = _tensor_payload(tensors)
        handle = self._require_handle()
        handle.seek(LATENT_SHARD_HEADER_SIZE + row_start * LATENT_RECORD_BYTES)
        handle.write(payload)
        self._crc32 = zlib.crc32(payload, self._crc32) & 0xFFFFFFFF
        self._sha256.update(payload)
        self._written_rows = end
        if end == wsi_end:
            handle.flush()
            os.fsync(handle.fileno())
            self._committed_rows = end
            self._completed_wsi_ids.append(wsi_id)
            self._write_state()

    def finalize(self) -> LatentArtifact:
        """Publish the binary and JSON only after every manifest WSI is committed.

        Returns:
            The fully validated completed artifact and its full-file SHA-256.

        Raises:
            RuntimeError: If any manifest WSI remains incomplete.

        """
        if self._complete:
            return validate_latent_artifact(
                bin_path=self.bin_path,
                manifest_path=self.manifest.path,
                run_number=self.manifest.run_number,
                model_name=self.model_name,
                checkpoint_sha256=self.checkpoint_sha256,
                expected_checkpoint_sha256=self.expected_checkpoint_sha256,
                expected_manifest_sha256=self.manifest.sha256,
                expected_union_sha256=self.expected_union_sha256,
                validate_payload=True,
                coordinate_only_manifest=self.coordinate_only_manifest,
            )
        if self._written_rows != len(self.manifest.rows):
            message = (
                f"Cannot finalize {self._written_rows}/{len(self.manifest.rows)} rows"
            )
            raise RuntimeError(message)
        if self._committed_rows != self._written_rows:
            message = "Cannot finalize an incomplete active WSI"
            raise RuntimeError(message)
        self._publish_from_partial()
        artifact = validate_latent_artifact(
            bin_path=self.bin_path,
            manifest_path=self.manifest.path,
            run_number=self.manifest.run_number,
            model_name=self.model_name,
            checkpoint_sha256=self.checkpoint_sha256,
            expected_checkpoint_sha256=self.expected_checkpoint_sha256,
            expected_manifest_sha256=self.manifest.sha256,
            expected_union_sha256=self.expected_union_sha256,
            validate_payload=False,
            coordinate_only_manifest=self.coordinate_only_manifest,
        )
        if self._complete_file_sha256 is None:
            message = "Completed latent shard lacks its streamed file identity"
            raise RuntimeError(message)
        return LatentArtifact(
            artifact.bin_path,
            artifact.sidecar_path,
            artifact.manifest,
            artifact.header,
            artifact.sidecar,
            self._complete_file_sha256,
        )

    def close(self) -> None:
        """Close the private partial-file handle without claiming completion."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def rollback_uncommitted_wsi(self) -> None:
        """Fsync, truncate, and rescan at the durable WSI boundary.

        Raises:
            ValueError: If provenance or committed bytes fail validation.

        """
        if self._complete:
            validate_latent_artifact(
                bin_path=self.bin_path,
                manifest_path=self.manifest.path,
                run_number=self.manifest.run_number,
                model_name=self.model_name,
                checkpoint_sha256=self.checkpoint_sha256,
                expected_checkpoint_sha256=self.expected_checkpoint_sha256,
                expected_manifest_sha256=self.manifest.sha256,
                expected_union_sha256=self.expected_union_sha256,
                validate_payload=True,
                coordinate_only_manifest=self.coordinate_only_manifest,
            )
            _fsync_directory(self.bin_path.parent)
            return
        if self._handle is not None:
            self._handle.flush()
            os.fsync(self._handle.fileno())
        self.close()
        # Validate the state and committed bytes before altering the file.
        self._restore_state(require_all_rows=False)
        committed_boundary = (
            LATENT_SHARD_HEADER_SIZE + self._committed_rows * LATENT_RECORD_BYTES
        )
        with self.partial_path.open("r+b") as handle:
            handle.truncate(committed_boundary)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(self.partial_path.parent)
        if self.partial_path.stat().st_size != committed_boundary:
            message = "Rolled-back latent binary has the wrong committed size"
            raise ValueError(message)
        # Rescan the now-exact prefix before it can be named by an incomplete marker.
        self._restore_state(require_all_rows=False)
        self._handle = self.partial_path.open("r+b")
        self._handle.seek(0, os.SEEK_END)

    def __enter__(self) -> Self:
        """Return this writer for scoped use.

        Returns:
            The active writer.

        """
        return self

    def __exit__(self, *_args: object) -> None:
        """Close resources; callers explicitly finalize successful output."""
        self.close()

    def _recover_or_create(self) -> None:
        partial = self.partial_path.exists()
        final = self.bin_path.exists()
        state = self.state_path.exists()
        sidecar = self.sidecar_path.exists()
        if partial and final:
            message = "Simultaneous partial and final latent binaries"
            raise ValueError(message)
        if final and sidecar:
            validate_latent_artifact(
                bin_path=self.bin_path,
                manifest_path=self.manifest.path,
                run_number=self.manifest.run_number,
                model_name=self.model_name,
                checkpoint_sha256=self.checkpoint_sha256,
                expected_checkpoint_sha256=self.expected_checkpoint_sha256,
                expected_manifest_sha256=self.manifest.sha256,
                expected_union_sha256=self.expected_union_sha256,
                validate_payload=True,
                coordinate_only_manifest=self.coordinate_only_manifest,
            )
            if state:
                # A stale state is private, but it still claims provenance. Validate it
                # against the completed binary before deleting it so a mismatched state
                # cannot be silently treated as a harmless cleanup artifact.
                self._restore_state(require_all_rows=True)
                self._validate_final_against_state()
                self.state_path.unlink()
                _fsync_directory(self.state_path.parent)
            self._complete = True
            self._written_rows = len(self.manifest.rows)
            self._committed_rows = self._written_rows
            self._completed_wsi_ids = [
                wsi_id for wsi_id, _start, _end in self.manifest.wsi_ranges
            ]
            return
        if sidecar and not final:
            message = "Latent completion JSON exists without its final binary"
            raise ValueError(message)
        if final:
            if not state:
                message = "Orphan final latent binary has no JSON or resume state"
                raise ValueError(message)
            self._restore_state(require_all_rows=True)
            self._validate_final_against_state()
            self._publish_sidecar()
            self.state_path.unlink()
            _fsync_directory(self.state_path.parent)
            self._complete = True
            return
        if partial != state:
            message = "Partial latent binary and resume state must exist together"
            raise ValueError(message)
        if partial:
            self._restore_state(require_all_rows=False)
            if self._committed_rows == len(self.manifest.rows):
                self._handle = self.partial_path.open("r+b")
                self._publish_from_partial()
                return
            self._handle = self.partial_path.open("r+b")
            self._handle.truncate(
                LATENT_SHARD_HEADER_SIZE + self._committed_rows * LATENT_RECORD_BYTES,
            )
            self._handle.seek(0, os.SEEK_END)
            self._written_rows = self._committed_rows
            return
        self._handle = self.partial_path.open("w+b")
        self._handle.write(make_latent_shard_header(tensor_count=0, payload_crc32=0))
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._write_state()

    def _restore_state(  # noqa: C901, PLR0912, PLR0914, PLR0915
        self,
        *,
        require_all_rows: bool,
    ) -> None:
        state = _read_json(self.state_path)
        expected_state_keys = {
            "schema_version",
            "status",
            "model_name",
            "checkpoint_sha256",
            "pinned_union_sha256",
            "source_manifest",
            "tensor",
            "committed_rows",
            "committed_bytes",
            "completed_wsi_ids",
            "prefix_crc32",
            "prefix_sha256",
        }
        if set(state) != expected_state_keys:
            message = "Resume state fields do not match the canonical schema"
            raise ValueError(message)
        expected_static = self._state_payload(include_progress=False)
        for key, expected in expected_static.items():
            if state.get(key) != expected:
                message = f"Resume state provenance mismatch for {key}"
                raise ValueError(message)
        committed_rows = _json_int(state, "committed_rows")
        committed_bytes = _json_int(state, "committed_bytes")
        completed = _json_int_list(state, "completed_wsi_ids")
        expected_ids, derived_rows = self._completed_prefix(committed_rows)
        if completed != expected_ids or derived_rows != committed_rows:
            message = "Resume completed-WSI prefix is inconsistent"
            raise ValueError(message)
        expected_boundary = (
            LATENT_SHARD_HEADER_SIZE + committed_rows * LATENT_RECORD_BYTES
        )
        if committed_bytes != expected_boundary:
            message = "Resume committed byte boundary is inconsistent"
            raise ValueError(message)
        if require_all_rows and committed_rows != len(self.manifest.rows):
            message = "Renamed final binary does not have all rows committed"
            raise ValueError(message)
        if (
            not require_all_rows
            and self.partial_path.stat().st_size < expected_boundary
        ):
            message = "Partial latent binary is shorter than its committed boundary"
            raise ValueError(message)
        scan_path = self.bin_path if require_all_rows else self.partial_path
        with scan_path.open("rb") as handle:
            header = parse_latent_shard_header(handle.read(LATENT_SHARD_HEADER_SIZE))
        all_rows = committed_rows == len(self.manifest.rows)
        provisional = header.tensor_count == 0 and header.payload_crc32 == 0
        final = header.tensor_count == len(
            self.manifest.rows,
        ) and header.payload_crc32 == _json_int(state, "prefix_crc32")
        if require_all_rows and not final:
            message = "Renamed latent binary does not contain its final header"
            raise ValueError(message)
        if not require_all_rows and not provisional and not (all_rows and final):
            message = "Partial latent binary has an invalid provisional/final header"
            raise ValueError(message)
        crc, digest, _bytes = _scan_payload(scan_path, stop=expected_boundary)
        if crc != _json_int(state, "prefix_crc32"):
            message = "Resume prefix CRC32 mismatch"
            raise ValueError(message)
        if digest != _json_str(state, "prefix_sha256"):
            message = "Resume prefix SHA-256 mismatch"
            raise ValueError(message)
        self._crc32 = crc
        self._sha256 = hashlib.sha256()
        with scan_path.open("rb") as handle:
            handle.seek(LATENT_SHARD_HEADER_SIZE)
            remaining = committed_rows * LATENT_RECORD_BYTES
            while remaining:
                chunk = handle.read(min(HASH_CHUNK_BYTES, remaining))
                if not chunk:
                    message = "Committed latent prefix ended during SHA-256 rescan"
                    raise ValueError(message)
                self._sha256.update(chunk)
                remaining -= len(chunk)
        self._completed_wsi_ids = completed
        self._committed_rows = committed_rows
        self._written_rows = committed_rows

    def _validate_final_against_state(self) -> None:
        with self.bin_path.open("rb") as handle:
            header = parse_latent_shard_header(handle.read(LATENT_SHARD_HEADER_SIZE))
        if (
            header.tensor_count != len(self.manifest.rows)
            or header.payload_crc32 != self._crc32
            or self.bin_path.stat().st_size
            != LATENT_SHARD_HEADER_SIZE + len(self.manifest.rows) * LATENT_RECORD_BYTES
        ):
            message = "Recovered final latent header/size disagrees with resume state"
            raise ValueError(message)

    def _publish_from_partial(self) -> None:
        handle = self._require_handle()
        handle.seek(0)
        handle.write(
            make_latent_shard_header(
                tensor_count=len(self.manifest.rows),
                payload_crc32=self._crc32,
            ),
        )
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        self._handle = None
        self.partial_path.replace(self.bin_path)
        _fsync_directory(self.bin_path.parent)
        self._publish_sidecar()
        self.state_path.unlink(missing_ok=True)
        _fsync_directory(self.state_path.parent)
        self._complete = True

    def _publish_sidecar(self) -> None:
        payload = self._sidecar_payload()
        temporary = self.sidecar_path.with_suffix(".json.tmp")
        _write_json(temporary, payload)
        temporary.replace(self.sidecar_path)
        _fsync_directory(self.sidecar_path.parent)

    def _sidecar_payload(self) -> dict[str, object]:
        first = self.manifest.rows[0].identity if self.manifest.rows else None
        last = self.manifest.rows[-1].identity if self.manifest.rows else None
        payload_crc, payload_sha, payload_bytes, file_sha256 = _scan_payload_and_file(
            self.bin_path,
        )
        if payload_crc != self._crc32:
            message = "Final latent payload CRC differs from the streamed writer state"
            raise ValueError(message)
        self._complete_file_sha256 = file_sha256
        return {
            "schema_version": "spec0020.latent_shard.v1",
            "status": "complete",
            "model_name": self.model_name,
            "checkpoint_sha256": self.checkpoint_sha256,
            "pinned_union_sha256": self.expected_union_sha256,
            "source_manifest": {
                "logical_basename": self.manifest.path.name,
                "sha256": self.manifest.sha256,
                "run_number": self.manifest.run_number,
                "row_count": len(self.manifest.rows),
                "first_identity": _identity_json(first),
                "last_identity": _identity_json(last),
            },
            "tensor": {
                "dtype": "float32_le",
                "shape": [LATENT_CHANNELS, LATENT_HEIGHT, LATENT_WIDTH],
                "layout": "CHW",
                "record_bytes": LATENT_RECORD_BYTES,
                "count": len(self.manifest.rows),
            },
            "payload_crc32": payload_crc,
            "payload_sha256": payload_sha,
            "payload_bytes": payload_bytes,
            "file_size": self.bin_path.stat().st_size,
            "completed_wsi_ids": self._completed_wsi_ids,
            "completed_wsi_count": len(self._completed_wsi_ids),
        }

    def _write_state(self) -> None:
        temporary = self.state_path.with_suffix(".json.tmp")
        _write_json(temporary, self._state_payload(include_progress=True))
        temporary.replace(self.state_path)
        _fsync_directory(self.state_path.parent)

    def _state_payload(self, *, include_progress: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": "spec0020.latent_resume.v1",
            "status": "in_progress",
            "model_name": self.model_name,
            "checkpoint_sha256": self.checkpoint_sha256,
            "pinned_union_sha256": self.expected_union_sha256,
            "source_manifest": {
                "logical_basename": self.manifest.path.name,
                "sha256": self.manifest.sha256,
                "run_number": self.manifest.run_number,
                "row_count": len(self.manifest.rows),
            },
            "tensor": {
                "dtype": "float32_le",
                "shape": [LATENT_CHANNELS, LATENT_HEIGHT, LATENT_WIDTH],
                "layout": "CHW",
                "record_bytes": LATENT_RECORD_BYTES,
            },
        }
        if include_progress:
            payload.update({
                "committed_rows": self._committed_rows,
                "committed_bytes": (
                    LATENT_SHARD_HEADER_SIZE
                    + self._committed_rows * LATENT_RECORD_BYTES
                ),
                "completed_wsi_ids": self._completed_wsi_ids,
                "prefix_crc32": self._crc32,
                "prefix_sha256": self._sha256.hexdigest(),
            })
        return payload

    def _completed_prefix(self, committed_rows: int) -> tuple[list[int], int]:
        completed: list[int] = []
        derived = 0
        for wsi_id, start, end in self.manifest.wsi_ranges:
            if end <= committed_rows:
                completed.append(wsi_id)
                derived = end
            elif start < committed_rows:
                message = "Committed row boundary cuts through a WSI"
                raise ValueError(message)
            else:
                break
        return completed, derived

    def _range_for_row(self, row_index: int) -> tuple[int, int]:
        for _wsi_id, start, end in self.manifest.wsi_ranges:
            if start <= row_index < end:
                return start, end
        message = f"Row {row_index} lies outside the manifest"
        raise IndexError(message)

    def _require_handle(self) -> BinaryIO:
        if self._handle is None:
            message = "Latent writer has no open partial binary"
            raise RuntimeError(message)
        return self._handle


class LatentTensorDataset(Dataset[Tensor]):
    """Expose fixed FP32 records as zero-copy read-only mmap tensor views."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        bin_path: Path,
        manifest_path: Path,
        run_number: int,
        model_name: ModelName,
        checkpoint_sha256: str,
        expected_checkpoint_sha256: str,
        expected_manifest_sha256: str,
        expected_union_sha256: str,
        advice: MmapAdvice = "none",
        validate_payload: bool = False,
    ) -> None:
        """Validate provenance before retaining only worker-local mmap state.

        Raises:
            RuntimeError: If the host cannot expose little-endian zero-copy views.
            ValueError: If advice, structure, or provenance is invalid.

        """
        if sys.byteorder != "little":
            message = "Zero-copy F32L mmap requires a little-endian host"
            raise RuntimeError(message)
        if advice not in {"sequential", "random", "none"}:
            message = f"Unsupported mmap advice {advice!r}"
            raise ValueError(message)
        # Validation can raise after Python has allocated this object. Initialize
        # cleanup-owned attributes first so __del__ remains safe on that path.
        self._file: BinaryIO | None = None
        self._mmap: mmap.mmap | None = None
        artifact = validate_latent_artifact(
            bin_path=bin_path,
            manifest_path=manifest_path,
            run_number=run_number,
            model_name=model_name,
            checkpoint_sha256=checkpoint_sha256,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            expected_manifest_sha256=expected_manifest_sha256,
            expected_union_sha256=expected_union_sha256,
            validate_payload=validate_payload,
        )
        self.bin_path = bin_path
        self.records = tuple(row.identity for row in artifact.manifest.rows)
        self.advice: MmapAdvice = advice

    def __len__(self) -> int:
        """Return the manifest-bound tensor count.

        Returns:
            Number of fixed latent records.

        """
        return len(self.records)

    def __getitem__(self, index: int) -> Tensor:
        """Return one zero-copy CHW FP32 tensor view.

        Returns:
            Read-only mmap-backed tensor in canonical CHW shape.

        Raises:
            IndexError: If the row index is outside this shard.

        """
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            message = f"Latent index {index} outside length {len(self)}"
            raise IndexError(message)
        mapping = self._ensure_mmap()
        offset = LATENT_SHARD_HEADER_SIZE + index * LATENT_RECORD_BYTES
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The given buffer is not writable",
                category=UserWarning,
            )
            values = torch.frombuffer(
                mapping,
                dtype=torch.float32,
                count=LATENT_VALUES,
                offset=offset,
            )
        return values.reshape(LATENT_CHANNELS, LATENT_HEIGHT, LATENT_WIDTH)

    def close(self) -> None:
        """Close process-local resources after all tensor views are released."""
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._file is not None:
            self._file.close()
            self._file = None

    def __getstate__(self) -> dict[str, object]:
        """Drop worker-local file and mmap handles during pickling.

        Returns:
            Pickle-safe dataset state.

        """
        state = dict(self.__dict__)
        state["_file"] = None
        state["_mmap"] = None
        return state

    def __del__(self) -> None:
        """Best-effort cleanup during interpreter shutdown."""
        try:
            self.close()
        except BufferError:
            return

    def _ensure_mmap(self) -> mmap.mmap:
        if self._mmap is None:
            self._file = self.bin_path.open("rb")
            self._mmap = mmap.mmap(
                self._file.fileno(),
                length=0,
                access=mmap.ACCESS_READ,
            )
            _advise(self._mmap, self.advice)
        return self._mmap


class LatentTaskView(Dataset[tuple[Tensor, LatentTaskLocation]]):
    """Resolve one canonical task/split location CSV over five model shards."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        model_name: ModelName,
        shard_paths: Mapping[int, Path],
        work_manifest_paths: Mapping[int, Path],
        location_path: Path,
        checkpoint_sha256: str,
        expected_checkpoint_sha256: str,
        expected_work_manifest_hashes: Mapping[int, str],
        expected_union_sha256: str,
        advice: MmapAdvice = "random",
        validate_payload: bool = False,
    ) -> None:
        """Validate all physical stores and every logical location identity.

        Raises:
            ValueError: If the run maps or location identities are invalid.

        """
        expected_runs = set(range(1, 6))
        if (
            set(shard_paths) != expected_runs
            or set(work_manifest_paths) != expected_runs
            or set(expected_work_manifest_hashes) != expected_runs
        ):
            message = "Latent task view requires exactly runs 1 through 5"
            raise ValueError(message)
        self._datasets = {
            run: LatentTensorDataset(
                bin_path=shard_paths[run],
                manifest_path=work_manifest_paths[run],
                run_number=run,
                model_name=model_name,
                checkpoint_sha256=checkpoint_sha256,
                expected_checkpoint_sha256=expected_checkpoint_sha256,
                expected_manifest_sha256=expected_work_manifest_hashes[run],
                expected_union_sha256=expected_union_sha256,
                advice=advice,
                validate_payload=validate_payload,
            )
            for run in range(1, 6)
        }
        try:
            self.locations = _load_latent_task_locations(
                location_path,
                datasets=self._datasets,
                work_manifest_paths=work_manifest_paths,
                expected_work_manifest_hashes=expected_work_manifest_hashes,
            )
        except Exception:
            self.close()
            raise

    def __len__(self) -> int:
        """Return the exact number of logical task rows.

        Returns:
            Number of rows in this logical view.

        """
        return len(self.locations)

    def __getitem__(self, index: int) -> tuple[Tensor, LatentTaskLocation]:
        """Return one zero-copy tensor and its unchanged task metadata.

        Returns:
            Tensor view and exact location metadata for ``index``.

        Raises:
            IndexError: If ``index`` lies outside the logical view.

        """
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            message = f"Latent task index {index} outside length {len(self)}"
            raise IndexError(message)
        location = self.locations[index]
        return self._datasets[location.run_number][location.file_index], location

    def close(self) -> None:
        """Close all five physical shard readers."""
        for dataset in self._datasets.values():
            dataset.close()

    def __del__(self) -> None:
        """Best-effort cleanup during interpreter shutdown."""
        try:
            self.close()
        except (AttributeError, BufferError):
            return


def _load_latent_task_locations(  # noqa: PLR0914
    path: Path,
    *,
    datasets: Mapping[int, LatentTensorDataset],
    work_manifest_paths: Mapping[int, Path],
    expected_work_manifest_hashes: Mapping[int, str],
) -> tuple[LatentTaskLocation, ...]:
    suffix = "_locations.csv"
    if not path.name.endswith(suffix):
        message = f"Unexpected latent location basename {path.name!r}"
        raise ValueError(message)
    task_split = path.name.removesuffix(suffix)
    if task_split not in EXPECTED_TASK_MANIFEST_SHA256:
        message = f"Unexpected latent task/split {task_split!r}"
        raise ValueError(message)
    task_name, expected_split = task_split.split("_", maxsplit=1)
    manifests = {
        run: load_work_manifest(
            work_manifest_paths[run],
            run_number=run,
            expected_sha256=expected_work_manifest_hashes[run],
        )
        for run in range(1, 6)
    }
    locations: list[LatentTaskLocation] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != LATENT_LOCATION_HEADER:
            message = f"Unexpected latent location header in {path}"
            raise ValueError(message)
        for row_index, raw in enumerate(reader):
            row = cast("Mapping[str, str | None]", raw)
            run_number = _csv_int(row, "run_number", row_index)
            file_index = _csv_int(row, "file_index", row_index)
            if run_number not in datasets or file_index >= len(datasets[run_number]):
                message = (
                    f"Latent location lies outside a physical shard at row {row_index}"
                )
                raise ValueError(message)
            identity = LatentRowIdentity(
                atlas_row_index=_csv_int(row, "atlas_row_index", row_index),
                wsi_id=_csv_int(row, "wsi_id", row_index),
                x=_csv_int(row, "x", row_index),
                y=_csv_int(row, "y", row_index),
            )
            work_row = manifests[run_number].rows[file_index]
            split_value = _csv_str(row, "split", row_index)
            if split_value != expected_split:
                message = f"Invalid latent location split at row {row_index}"
                raise ValueError(message)
            location = LatentTaskLocation(
                run_number=run_number,
                file_index=file_index,
                identity=identity,
                split=cast("SplitName", split_value),
                diagnosis_label=_csv_str(row, "diagnosis_label", row_index),
                diagnosis_index=_csv_int(row, "diagnosis_index", row_index),
                tissue_label=_csv_str(row, "tissue_label", row_index),
            )
            manifest_identity = (
                work_row.identity,
                work_row.split,
                work_row.diagnosis_label,
                work_row.diagnosis_index,
            )
            location_identity = (
                identity,
                location.split,
                location.diagnosis_label,
                location.diagnosis_index,
            )
            expected_tissue_label = (
                work_row.tissue_label if task_name == "tissue" else ""
            )
            tissue_disagrees = location.tissue_label != expected_tissue_label
            if location_identity != manifest_identity or tissue_disagrees:
                message = f"Latent location metadata disagrees at row {row_index}"
                raise ValueError(message)
            locations.append(location)
    return tuple(locations)


def validate_latent_artifact(  # noqa: PLR0913
    *,
    bin_path: Path,
    manifest_path: Path,
    run_number: int,
    model_name: ModelName,
    checkpoint_sha256: str,
    expected_checkpoint_sha256: str,
    expected_manifest_sha256: str,
    expected_union_sha256: str,
    validate_payload: bool = True,
    coordinate_only_manifest: bool = False,
) -> LatentArtifact:
    """Validate one completed binary, sidecar, and manifest binding.

    Returns:
        Validated artifact description.

    Raises:
        ValueError: If structure, checksums, or provenance disagree.

    """
    _validate_provenance(
        model_name,
        checkpoint_sha256,
        expected_checkpoint_sha256,
        expected_manifest_sha256,
        expected_union_sha256,
    )
    manifest = load_work_manifest(
        manifest_path,
        run_number=run_number,
        expected_sha256=expected_manifest_sha256,
        require_run_basename=not coordinate_only_manifest,
    )
    sidecar_path = bin_path.with_suffix(".json")
    if not sidecar_path.is_file():
        message = f"Missing latent completion sidecar {sidecar_path}"
        raise ValueError(message)
    with bin_path.open("rb") as handle:
        header = parse_latent_shard_header(handle.read(LATENT_SHARD_HEADER_SIZE))
    expected_size = LATENT_SHARD_HEADER_SIZE + len(manifest.rows) * LATENT_RECORD_BYTES
    if (
        header.tensor_count != len(manifest.rows)
        or bin_path.stat().st_size != expected_size
    ):
        message = "Latent header count or exact file size disagrees with manifest"
        raise ValueError(message)
    sidecar = _read_json(sidecar_path)
    _validate_sidecar(
        sidecar,
        bin_path=bin_path,
        manifest=manifest,
        model_name=model_name,
        checkpoint_sha256=checkpoint_sha256,
        expected_union_sha256=expected_union_sha256,
        header=header,
    )
    file_sha256: str | None = None
    if validate_payload:
        crc, digest, payload_bytes, file_sha256 = _scan_payload_and_file(bin_path)
        if (
            crc != header.payload_crc32
            or crc != _json_int(sidecar, "payload_crc32")
            or digest != _json_str(sidecar, "payload_sha256")
            or payload_bytes != _json_int(sidecar, "payload_bytes")
        ):
            message = "Latent payload checksum disagrees with header or sidecar"
            raise ValueError(message)
    return LatentArtifact(
        bin_path,
        sidecar_path,
        manifest,
        header,
        sidecar,
        file_sha256,
    )


def validate_latent_store_pair(  # noqa: C901, PLR0912, PLR0913, PLR0914, PLR0915
    *,
    union_manifest_path: Path,
    expected_union_sha256: str,
    work_manifest_paths: Mapping[int, Path],
    expected_work_manifest_hashes: Mapping[int, str],
    task_manifest_paths: Mapping[str, Path],
    expected_task_manifest_hashes: Mapping[str, str],
    normal_shards: Mapping[int, Path],
    so2_shards: Mapping[int, Path],
    normal_checkpoint_sha256: str,
    expected_normal_checkpoint_sha256: str,
    so2_checkpoint_sha256: str,
    expected_so2_checkpoint_sha256: str,
    validate_payload: bool = True,
    global_audit_path: Path | None = None,
) -> dict[str, object]:
    """Prove two five-shard stores expose twelve split-specific logical views.

    Returns:
        Counts and compact hashes of the validated logical task locations.

    Raises:
        ValueError: If runs, provenance, row order, or task counts disagree.

    """
    expected_runs = set(range(1, 6))
    if expected_union_sha256 != EXPECTED_UNION_SHA256:
        message = "Store-pair validator requires the canonical union SHA-256"
        raise ValueError(message)
    if dict(expected_work_manifest_hashes) != EXPECTED_WORK_MANIFEST_SHA256:
        message = "Store-pair validator requires canonical work-manifest hashes"
        raise ValueError(message)
    if dict(expected_task_manifest_hashes) != EXPECTED_TASK_MANIFEST_SHA256:
        message = "Store-pair validator requires canonical task-manifest hashes"
        raise ValueError(message)
    expected_task_keys = set(EXPECTED_TASK_MANIFEST_SHA256)
    if set(task_manifest_paths) != expected_task_keys:
        message = "Task manifests must contain exactly the six canonical task/splits"
        raise ValueError(message)
    for name, mapping in (
        ("work manifests", work_manifest_paths),
        ("manifest hashes", expected_work_manifest_hashes),
        ("normal shards", normal_shards),
        ("SO(2) shards", so2_shards),
    ):
        if set(mapping) != expected_runs:
            message = f"{name} must contain exactly runs 1 through 5"
            raise ValueError(message)
    union_hash = _sha256_file(union_manifest_path)
    if union_hash != expected_union_sha256:
        message = "Pinned union manifest SHA-256 mismatch"
        raise ValueError(message)
    union = load_work_manifest(
        union_manifest_path,
        run_number=0,
        expected_sha256=expected_union_sha256,
        require_run_basename=False,
    )
    works: dict[int, WorkManifest] = {}
    artifact_counts: dict[str, list[int]] = {"normal_vae": [], "so2_vae": []}
    artifact_file_sha256: dict[str, dict[str, str]] = {
        "normal_vae": {},
        "so2_vae": {},
    }
    union_offset = 0
    for run in range(1, 6):
        work_path = work_manifest_paths[run]
        work_hash = expected_work_manifest_hashes[run]
        work = load_work_manifest(
            work_path,
            run_number=run,
            expected_sha256=work_hash,
        )
        works[run] = work
        next_offset = union_offset + len(work.rows)
        if work.rows != union.rows[union_offset:next_offset]:
            message = (
                "Work-shard concatenation does not equal the pinned union manifest"
            )
            raise ValueError(message)
        union_offset = next_offset
        for model_name, paths, observed_checkpoint, expected_checkpoint in (
            (
                "normal_vae",
                normal_shards,
                normal_checkpoint_sha256,
                expected_normal_checkpoint_sha256,
            ),
            (
                "so2_vae",
                so2_shards,
                so2_checkpoint_sha256,
                expected_so2_checkpoint_sha256,
            ),
        ):
            artifact = validate_latent_artifact(
                bin_path=paths[run],
                manifest_path=work_path,
                run_number=run,
                model_name=cast("ModelName", model_name),
                checkpoint_sha256=observed_checkpoint,
                expected_checkpoint_sha256=expected_checkpoint,
                expected_manifest_sha256=work_hash,
                expected_union_sha256=expected_union_sha256,
                validate_payload=validate_payload,
            )
            artifact_counts[model_name].append(artifact.header.tensor_count)
            if artifact.file_sha256 is not None:
                artifact_file_sha256[model_name][str(run)] = artifact.file_sha256
    if union_offset != len(union.rows):
        message = "Work-shard concatenation does not cover the pinned union manifest"
        raise ValueError(message)
    task_counts = {"cancer": 0, "tissue": 0, "shared": 0}
    task_digests = {name: hashlib.sha256() for name in task_counts}
    locations: dict[LatentRowIdentity, tuple[int, int, WorkManifestRow]] = {}
    global_index = 0
    for run in range(1, 6):
        work = works[run]
        for file_index, row in enumerate(work.rows):
            location_bytes = struct.pack("<II", run, file_index)
            if row.identity in locations:
                message = "Union manifest contains a duplicate exact row identity"
                raise ValueError(message)
            locations[row.identity] = (run, file_index, row)
            if row.cancer_selected:
                task_counts["cancer"] += 1
                task_digests["cancer"].update(location_bytes)
            if row.tissue_selected:
                task_counts["tissue"] += 1
                task_digests["tissue"].update(location_bytes)
            if row.cancer_selected and row.tissue_selected:
                task_counts["shared"] += 1
                task_digests["shared"].update(location_bytes)
            if row != union.rows[global_index]:
                message = "Task-view row order differs from the pinned union"
                raise ValueError(message)
            global_index += 1
    observed = (
        len(union.rows),
        task_counts["cancer"],
        task_counts["tissue"],
        task_counts["shared"],
    )
    expected = (
        EXPECTED_UNION_ROWS,
        EXPECTED_CANCER_ROWS,
        EXPECTED_TISSUE_ROWS,
        EXPECTED_SHARED_ROWS,
    )
    if observed != expected:
        message = f"Latent task-view counts disagree: {observed} != {expected}"
        raise ValueError(message)
    (
        task_manifest_hashes,
        split_counts,
        split_location_hashes,
        location_file_hashes,
    ) = _validate_task_manifests(
        task_manifest_paths=task_manifest_paths,
        expected_task_manifest_hashes=expected_task_manifest_hashes,
        locations=locations,
    )
    logical_view_counts: dict[str, int] = {}
    logical_view_location_sha256: dict[str, str] = {}
    for model_name in ("normal_vae", "so2_vae"):
        for task_name in TASK_NAMES:
            for split_name in SPLIT_NAMES:
                task_split = f"{task_name}_{split_name}"
                view_name = f"{model_name}/{task_name}/{split_name}"
                logical_view_counts[view_name] = split_counts[task_split]
                logical_view_location_sha256[view_name] = split_location_hashes[
                    task_split
                ]
    result: dict[str, object] = {
        "status": "pass",
        "union_sha256": union_hash,
        "row_count": len(union.rows),
        "artifact_counts": artifact_counts,
        "artifact_file_sha256": artifact_file_sha256,
        "task_manifest_sha256": task_manifest_hashes,
        "task_counts_per_model": task_counts,
        "task_location_sha256": {
            name: digest.hexdigest() for name, digest in task_digests.items()
        },
        "logical_view_counts": logical_view_counts,
        "logical_view_location_sha256": logical_view_location_sha256,
        "location_file_sha256": location_file_hashes,
    }
    if global_audit_path is not None:
        _publish_global_audit(global_audit_path, result)
    return result


def load_work_manifest(  # noqa: C901, PLR0912, PLR0914, PLR0915
    path: Path,
    *,
    run_number: int,
    expected_sha256: str,
    require_run_basename: bool = True,
) -> WorkManifest:
    """Load and fully validate one hash-bound coordinate work manifest.

    Returns:
        The parsed rows and exact WSI boundaries.

    Raises:
        ValueError: If the hash, basename, header, or row ordering differs.

    """
    observed_hash = _sha256_file(path)
    if observed_hash != expected_sha256:
        message = f"Source manifest SHA-256 mismatch: {observed_hash}"
        raise ValueError(message)
    if require_run_basename and path.name != f"run_{run_number:02d}_of_05.csv":
        message = f"Unexpected work-manifest basename {path.name!r}"
        raise ValueError(message)
    rows: list[WorkManifestRow] = []
    ranges: list[tuple[int, int, int]] = []
    previous: tuple[int, int, int] | None = None
    previous_atlas_index: int | None = None
    active_wsi: int | None = None
    active_start = 0
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = tuple(reader.fieldnames or ())
        coordinate_only = header == COORDINATE_ONLY_HEADER
        if header != UNION_HEADER and not (
            coordinate_only and not require_run_basename
        ):
            message = f"Unexpected union manifest header in {path}"
            raise ValueError(message)
        for file_index, raw in enumerate(reader):
            row = cast("Mapping[str, str | None]", raw)
            identity = LatentRowIdentity(
                atlas_row_index=_csv_int(row, "atlas_row_index", file_index),
                wsi_id=_csv_int(row, "wsi_id", file_index),
                x=_csv_int(row, "x", file_index),
                y=_csv_int(row, "y", file_index),
            )
            key = (identity.wsi_id, identity.y, identity.x)
            if previous is not None and key <= previous:
                message = f"Manifest order/identity failure at file_index {file_index}"
                raise ValueError(message)
            if (
                previous_atlas_index is not None
                and identity.atlas_row_index <= previous_atlas_index
            ):
                message = f"Atlas-row identity order failure at file_index {file_index}"
                raise ValueError(message)
            previous = key
            previous_atlas_index = identity.atlas_row_index
            if coordinate_only:
                split: SplitName = "train"
                diagnosis_label = ""
                diagnosis_index = 0
                tissue_label = ""
                cancer = True
                tissue = False
            else:
                split_value = _csv_str(row, "split", file_index)
                if split_value not in SPLIT_NAMES:
                    message = f"Invalid split on union row {file_index}"
                    raise ValueError(message)
                split = split_value
                diagnosis_label = _csv_str(row, "diagnosis_label", file_index)
                diagnosis_index = _csv_int(row, "diagnosis_index", file_index)
                tissue_label = _csv_str(row, "tissue_label", file_index)
                cancer = _csv_bool(row, "cancer_ae_selected", file_index)
                tissue = _csv_bool(row, "tissue_selected", file_index)
                if not cancer and not tissue:
                    message = f"Union row {file_index} belongs to no task"
                    raise ValueError(message)
            if active_wsi is None:
                active_wsi = identity.wsi_id
            elif identity.wsi_id != active_wsi:
                ranges.append((active_wsi, active_start, file_index))
                active_wsi = identity.wsi_id
                active_start = file_index
            rows.append(
                WorkManifestRow(
                    identity=identity,
                    split=split,
                    diagnosis_label=diagnosis_label,
                    diagnosis_index=diagnosis_index,
                    tissue_label=tissue_label,
                    cancer_selected=cancer,
                    tissue_selected=tissue,
                ),
            )
    if active_wsi is not None:
        ranges.append((active_wsi, active_start, len(rows)))
    return WorkManifest(path, observed_hash, run_number, tuple(rows), tuple(ranges))


def _validate_task_manifests(  # noqa: C901, PLR0912, PLR0914, PLR0915
    *,
    task_manifest_paths: Mapping[str, Path],
    expected_task_manifest_hashes: Mapping[str, str],
    locations: Mapping[LatentRowIdentity, tuple[int, int, WorkManifestRow]],
) -> tuple[dict[str, str], dict[str, int], dict[str, str], dict[str, str]]:
    observed_hashes: dict[str, str] = {}
    counts: dict[str, int] = {}
    location_hashes: dict[str, str] = {}
    location_file_hashes: dict[str, str] = {}
    for task_name in TASK_NAMES:
        selected_identities = {
            identity
            for identity, (_run, _file_index, work_row) in locations.items()
            if (
                work_row.cancer_selected
                if task_name == "cancer"
                else work_row.tissue_selected
            )
        }
        observed_task_identities: set[LatentRowIdentity] = set()
        for split_name in SPLIT_NAMES:
            task_split = f"{task_name}_{split_name}"
            path = task_manifest_paths[task_split]
            if path.name != f"{task_split}.csv":
                message = f"Unexpected task-manifest basename {path.name!r}"
                raise ValueError(message)
            observed_hash = _sha256_file(path)
            expected_hash = expected_task_manifest_hashes[task_split]
            if observed_hash != expected_hash:
                message = f"Task manifest SHA-256 mismatch for {task_split}"
                raise ValueError(message)
            observed_hashes[task_split] = observed_hash
            expected_header = (
                CANCER_TASK_HEADER if task_name == "cancer" else TISSUE_TASK_HEADER
            )
            digest = hashlib.sha256()
            csv_digest = hashlib.sha256()
            csv_digest.update(_location_csv_line(LATENT_LOCATION_HEADER))
            split_identities: set[LatentRowIdentity] = set()
            previous_key: tuple[int, int, int] | None = None
            previous_atlas_index: int | None = None
            with path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if tuple(reader.fieldnames or ()) != expected_header:
                    message = f"Unexpected task manifest header in {path}"
                    raise ValueError(message)
                for row_index, raw in enumerate(reader):
                    row = cast("Mapping[str, str | None]", raw)
                    identity = LatentRowIdentity(
                        atlas_row_index=_csv_int(
                            row,
                            "atlas_row_index",
                            row_index,
                        ),
                        wsi_id=_csv_int(row, "wsi_id", row_index),
                        x=_csv_int(row, "x", row_index),
                        y=_csv_int(row, "y", row_index),
                    )
                    order_key = (identity.wsi_id, identity.y, identity.x)
                    if previous_key is not None and order_key <= previous_key:
                        message = (
                            f"Task manifest order/identity failure in {task_split} "
                            f"at row {row_index}"
                        )
                        raise ValueError(message)
                    if (
                        previous_atlas_index is not None
                        and identity.atlas_row_index <= previous_atlas_index
                    ):
                        message = (
                            f"Task atlas-row order failure in {task_split} "
                            f"at row {row_index}"
                        )
                        raise ValueError(message)
                    previous_key = order_key
                    previous_atlas_index = identity.atlas_row_index
                    if identity in observed_task_identities:
                        message = f"Duplicate task identity across {task_name} splits"
                        raise ValueError(message)
                    location = locations.get(identity)
                    if location is None:
                        message = f"Task identity is absent from union: {identity}"
                        raise ValueError(message)
                    run_number, file_index, work_row = location
                    if _csv_str(row, "split", row_index) != split_name:
                        message = f"Task row has the wrong split in {task_split}"
                        raise ValueError(message)
                    if work_row.split != split_name:
                        message = f"Task split disagrees with union in {task_split}"
                        raise ValueError(message)
                    if (
                        _csv_str(row, "diagnosis_label", row_index)
                        != work_row.diagnosis_label
                        or _csv_int(row, "diagnosis_index", row_index)
                        != work_row.diagnosis_index
                    ):
                        message = (
                            f"Task diagnosis metadata disagrees with union in "
                            f"{task_split}"
                        )
                        raise ValueError(message)
                    selected = (
                        work_row.cancer_selected
                        if task_name == "cancer"
                        else work_row.tissue_selected
                    )
                    if not selected:
                        message = f"Task row is not selected by union in {task_split}"
                        raise ValueError(message)
                    if (
                        task_name == "tissue"
                        and _csv_str(row, "tissue_label", row_index)
                        != work_row.tissue_label
                    ):
                        message = (
                            f"Task tissue metadata disagrees with union in {task_split}"
                        )
                        raise ValueError(message)
                    split_identities.add(identity)
                    observed_task_identities.add(identity)
                    digest.update(struct.pack("<II", run_number, file_index))
                    csv_digest.update(
                        _location_csv_line(
                            (
                                run_number,
                                file_index,
                                identity.atlas_row_index,
                                identity.wsi_id,
                                identity.x,
                                identity.y,
                                work_row.split,
                                work_row.diagnosis_label,
                                work_row.diagnosis_index,
                                work_row.tissue_label if task_name == "tissue" else "",
                            ),
                        ),
                    )
            expected_count = EXPECTED_TASK_VIEW_ROWS[task_split]
            if len(split_identities) != expected_count:
                message = (
                    f"Task view count disagrees for {task_split}: "
                    f"{len(split_identities)} != {expected_count}"
                )
                raise ValueError(message)
            expected_split_identities = {
                identity
                for identity in selected_identities
                if locations[identity][2].split == split_name
            }
            if split_identities != expected_split_identities:
                message = f"Task manifest is not the exact union view for {task_split}"
                raise ValueError(message)
            counts[task_split] = len(split_identities)
            location_hashes[task_split] = digest.hexdigest()
            location_file_hashes[task_split] = csv_digest.hexdigest()
        if observed_task_identities != selected_identities:
            message = f"Task manifests do not exhaust the union {task_name} selection"
            raise ValueError(message)
    return observed_hashes, counts, location_hashes, location_file_hashes


def _location_csv_line(values: Sequence[object]) -> bytes:
    buffer = io.StringIO(newline="")
    csv.writer(buffer, lineterminator="\n").writerow(values)
    return buffer.getvalue().encode()


def _publish_global_audit(path: Path, result: Mapping[str, object]) -> None:
    payload: dict[str, object] = {
        "schema_version": "spec0021.latent_store_global_audit.v1",
        "status": "complete",
        "validation": dict(result),
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        message = f"Refusing stale global-audit temporary {temporary}"
        raise FileExistsError(message)
    _write_json(temporary, payload)
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != temporary.read_bytes():
                message = f"Refusing to overwrite global audit {path}"
                raise FileExistsError(message) from None
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
        _fsync_directory(path.parent)


def _validate_provenance(
    model_name: ModelName,
    checkpoint_sha256: str,
    expected_checkpoint_sha256: str,
    expected_manifest_sha256: str,
    expected_union_sha256: str,
) -> None:
    if model_name not in EXPECTED_CHECKPOINT_SHA256:
        message = f"Unsupported latent model {model_name!r}"
        raise ValueError(message)
    if expected_checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256[model_name]:
        message = f"Untrusted expected checkpoint for {model_name}"
        raise ValueError(message)
    if checkpoint_sha256 != expected_checkpoint_sha256:
        message = f"Observed checkpoint SHA-256 mismatch for {model_name}"
        raise ValueError(message)
    for name, value in (
        ("manifest", expected_manifest_sha256),
        ("union", expected_union_sha256),
    ):
        if len(value) != SHA256_HEX_LENGTH or any(
            char not in "0123456789abcdef" for char in value
        ):
            message = f"Invalid expected {name} SHA-256"
            raise ValueError(message)


def _validate_sidecar(  # noqa: PLR0913
    sidecar: Mapping[str, object],
    *,
    bin_path: Path,
    manifest: WorkManifest,
    model_name: ModelName,
    checkpoint_sha256: str,
    expected_union_sha256: str,
    header: LatentShardHeader,
) -> None:
    expected_keys = {
        "schema_version",
        "status",
        "model_name",
        "checkpoint_sha256",
        "pinned_union_sha256",
        "source_manifest",
        "tensor",
        "payload_crc32",
        "payload_sha256",
        "payload_bytes",
        "file_size",
        "completed_wsi_ids",
        "completed_wsi_count",
    }
    if set(sidecar) != expected_keys:
        message = "Latent sidecar fields do not match the canonical schema"
        raise ValueError(message)
    first = manifest.rows[0].identity if manifest.rows else None
    last = manifest.rows[-1].identity if manifest.rows else None
    expected_top = {
        "schema_version": "spec0020.latent_shard.v1",
        "status": "complete",
        "model_name": model_name,
        "checkpoint_sha256": checkpoint_sha256,
        "pinned_union_sha256": expected_union_sha256,
        "file_size": bin_path.stat().st_size,
        "payload_crc32": header.payload_crc32,
    }
    for key, expected in expected_top.items():
        if sidecar.get(key) != expected:
            message = f"Latent sidecar mismatch for {key}"
            raise ValueError(message)
    source = _json_mapping(sidecar, "source_manifest")
    expected_source = {
        "logical_basename": manifest.path.name,
        "sha256": manifest.sha256,
        "run_number": manifest.run_number,
        "row_count": len(manifest.rows),
        "first_identity": _identity_json(first),
        "last_identity": _identity_json(last),
    }
    if dict(source) != expected_source:
        message = "Latent sidecar source-manifest binding disagrees"
        raise ValueError(message)
    tensor = _json_mapping(sidecar, "tensor")
    expected_tensor = {
        "dtype": "float32_le",
        "shape": [LATENT_CHANNELS, LATENT_HEIGHT, LATENT_WIDTH],
        "layout": "CHW",
        "record_bytes": LATENT_RECORD_BYTES,
        "count": len(manifest.rows),
    }
    if dict(tensor) != expected_tensor:
        message = "Latent sidecar tensor contract disagrees"
        raise ValueError(message)
    expected_wsi_ids = [wsi_id for wsi_id, _start, _end in manifest.wsi_ranges]
    if _json_int_list(sidecar, "completed_wsi_ids") != expected_wsi_ids:
        message = "Latent sidecar completed WSI order disagrees"
        raise ValueError(message)
    if _json_int(sidecar, "completed_wsi_count") != len(expected_wsi_ids):
        message = "Latent sidecar completed WSI count disagrees"
        raise ValueError(message)
    if _json_int(sidecar, "payload_bytes") != len(manifest.rows) * LATENT_RECORD_BYTES:
        message = "Latent sidecar payload byte count disagrees"
        raise ValueError(message)
    payload_sha = _json_str(sidecar, "payload_sha256")
    if len(payload_sha) != SHA256_HEX_LENGTH or any(
        char not in "0123456789abcdef" for char in payload_sha
    ):
        message = "Latent sidecar payload SHA-256 is invalid"
        raise ValueError(message)


def _tensor_payload(tensors: Tensor) -> bytes:
    if tensors.device.type != "cpu":
        message = f"Expected CPU latent tensor, got {tensors.device}"
        raise ValueError(message)
    if tensors.dtype != torch.float32:
        message = f"Expected FP32 latent tensor, got {tensors.dtype}"
        raise TypeError(message)
    if tuple(tensors.shape[1:]) != (LATENT_CHANNELS, LATENT_HEIGHT, LATENT_WIDTH):
        message = f"Unexpected latent tensor shape {tuple(tensors.shape)}"
        raise ValueError(message)
    if not bool(torch.isfinite(tensors).all()):
        message = "Latent tensor contains nonfinite values"
        raise ValueError(message)
    contiguous = tensors.contiguous()
    array = np.asarray(contiguous.numpy(), dtype=np.dtype("<f4"), order="C")
    return array.tobytes(order="C")


def _scan_payload(path: Path, *, stop: int | None = None) -> tuple[int, str, int]:
    crc = 0
    digest = hashlib.sha256()
    payload_bytes = 0
    with path.open("rb") as handle:
        handle.seek(LATENT_SHARD_HEADER_SIZE)
        remaining = None if stop is None else stop - LATENT_SHARD_HEADER_SIZE
        while remaining is None or remaining > 0:
            size = (
                HASH_CHUNK_BYTES
                if remaining is None
                else min(HASH_CHUNK_BYTES, remaining)
            )
            chunk = handle.read(size)
            if not chunk:
                break
            crc = zlib.crc32(chunk, crc)
            digest.update(chunk)
            payload_bytes += len(chunk)
            if remaining is not None:
                remaining -= len(chunk)
        if remaining not in {None, 0}:
            message = "Payload ended before the requested committed boundary"
            raise ValueError(message)
    return crc & 0xFFFFFFFF, digest.hexdigest(), payload_bytes


def _scan_payload_and_file(path: Path) -> tuple[int, str, int, str]:
    """Hash a complete shard and its payload in one sequential file pass.

    Returns:
        Payload CRC32, payload SHA-256, payload bytes, and full-file SHA-256.

    Raises:
        ValueError: If the shard ends before its fixed header.

    """
    crc = 0
    payload_digest = hashlib.sha256()
    file_digest = hashlib.sha256()
    payload_bytes = 0
    with path.open("rb") as handle:
        header = handle.read(LATENT_SHARD_HEADER_SIZE)
        if len(header) != LATENT_SHARD_HEADER_SIZE:
            message = "Latent shard ended before its fixed header"
            raise ValueError(message)
        file_digest.update(header)
        while chunk := handle.read(HASH_CHUNK_BYTES):
            file_digest.update(chunk)
            payload_digest.update(chunk)
            crc = zlib.crc32(chunk, crc)
            payload_bytes += len(chunk)
    return (
        crc & 0xFFFFFFFF,
        payload_digest.hexdigest(),
        payload_bytes,
        file_digest.hexdigest(),
    )


def _identity_json(identity: LatentRowIdentity | None) -> dict[str, int] | None:
    if identity is None:
        return None
    return {
        "atlas_row_index": identity.atlas_row_index,
        "wsi_id": identity.wsi_id,
        "x": identity.x,
        "y": identity.y,
    }


def _advise(mapping: mmap.mmap, advice: MmapAdvice) -> None:
    if advice == "none":
        return
    constant_name = "MADV_SEQUENTIAL" if advice == "sequential" else "MADV_RANDOM"
    try:
        mapping.madvise(cast("int", getattr(mmap, constant_name)))
    except (AttributeError, OSError, ValueError):
        return


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    """Persist same-directory renames that define commit boundaries."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = cast("object", json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError) as error:
        message = f"Invalid latent JSON {path}"
        raise ValueError(message) from error
    if not isinstance(value, dict):
        message = f"Latent JSON {path} must contain an object"
        raise TypeError(message)
    return cast("dict[str, object]", value)


def _json_mapping(value: Mapping[str, object], key: str) -> Mapping[str, object]:
    result = value.get(key)
    if not isinstance(result, dict):
        message = f"Latent JSON field {key!r} must be an object"
        raise TypeError(message)
    return cast("Mapping[str, object]", result)


def _json_int(value: Mapping[str, object], key: str) -> int:
    result = value.get(key)
    if not isinstance(result, int) or isinstance(result, bool):
        message = f"Latent JSON field {key!r} must be an integer"
        raise TypeError(message)
    return result


def _json_str(value: Mapping[str, object], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str):
        message = f"Latent JSON field {key!r} must be a string"
        raise TypeError(message)
    return result


def _json_int_list(value: Mapping[str, object], key: str) -> list[int]:
    result = value.get(key)
    if not isinstance(result, list):
        message = f"Latent JSON field {key!r} must be an integer list"
        raise TypeError(message)
    items = cast("list[object]", result)
    if any(not isinstance(item, int) or isinstance(item, bool) for item in items):
        message = f"Latent JSON field {key!r} must be an integer list"
        raise TypeError(message)
    return cast("list[int]", items)


def _csv_int(row: Mapping[str, str | None], key: str, row_index: int) -> int:
    value = row.get(key)
    try:
        parsed = int(cast("str", value))
    except (TypeError, ValueError) as error:
        message = f"Invalid {key} on union row {row_index}"
        raise ValueError(message) from error
    if parsed < 0:
        message = f"Negative {key} on union row {row_index}"
        raise ValueError(message)
    return parsed


def _csv_str(row: Mapping[str, str | None], key: str, row_index: int) -> str:
    value = row.get(key)
    if value is None:
        message = f"Missing {key} on manifest row {row_index}"
        raise ValueError(message)
    return value


def _csv_bool(row: Mapping[str, str | None], key: str, row_index: int) -> bool:
    value = row.get(key)
    if value not in {"true", "false"}:
        message = f"Invalid {key} on union row {row_index}"
        raise ValueError(message)
    return value == "true"


__all__ = [
    "EXPECTED_CHECKPOINT_SHA256",
    "EXPECTED_TASK_MANIFEST_SHA256",
    "EXPECTED_TASK_VIEW_ROWS",
    "EXPECTED_UNION_SHA256",
    "EXPECTED_WORK_MANIFEST_SHA256",
    "LATENT_LOCATION_HEADER",
    "LATENT_RECORD_BYTES",
    "LATENT_SHARD_DTYPE",
    "LATENT_SHARD_HEADER_FORMAT",
    "LATENT_SHARD_HEADER_SIZE",
    "LATENT_SHARD_LAYOUT",
    "LATENT_SHARD_MAGIC",
    "LATENT_SHARD_VERSION",
    "LatentArtifact",
    "LatentRowIdentity",
    "LatentShardHeader",
    "LatentShardWriter",
    "LatentTaskLocation",
    "LatentTaskView",
    "LatentTensorDataset",
    "WorkManifest",
    "WorkManifestRow",
    "load_work_manifest",
    "make_latent_shard_header",
    "parse_latent_shard_header",
    "validate_latent_artifact",
    "validate_latent_store_pair",
]
