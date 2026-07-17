# Copyright 2026 HiperMaximus
"""UBC-OCEAN binary patch-shard loading for spec 0001."""

from __future__ import annotations

import csv
import struct
import zlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

PATCH_SHARD_MAGIC = b"UBC_DATA"
PATCH_SHARD_HEADER_FORMAT = "<8sIQiiii3s25x"
PATCH_SHARD_HEADER_SIZE = 64
PATCH_SHARD_LAYOUT = b"CHW"
PATCH_SHARD_VERSION = 1
PATCH_LABELS = frozenset({0, 1, 2, 3, 4})
PATCH_CSV_COLUMNS = frozenset({"wsi_id", "label", "x", "y"})
CRC_CHUNK_BYTES = 8 * 1024 * 1024

if struct.calcsize(PATCH_SHARD_HEADER_FORMAT) != PATCH_SHARD_HEADER_SIZE:
    message = "Patch-shard header struct must stay exactly 64 bytes"
    raise RuntimeError(message)


@dataclass(frozen=True)
class PatchRecord:
    """One row of patch metadata and its binary patch index."""

    file_index: int
    row_index: int
    wsi_id: str
    label: int
    x: int
    y: int

    @property
    def index(self) -> int:
        """Backward-compatible alias for the canonical binary file index.

        Returns:
            Binary file index for this patch.

        """
        return self.file_index

    def sample_id(self, split: str | None = None) -> str:
        """Return the stable sample identifier used in metric rows.

        Returns:
            Stable sample identifier.

        """
        split_name = "unknown" if split is None else split
        return (
            f"{split_name}:{self.file_index:08d}:{self.wsi_id}:"
            f"{self.label}:{self.x}:{self.y}"
        )


@dataclass(frozen=True)
class PatchShardHeader:
    """Parsed 64-byte UBC patch-shard header."""

    crc32: int
    patch_count: int
    channels: int
    height: int
    width: int
    version: int
    layout: bytes


@dataclass(frozen=True)
class PatchShardSpec:
    """Filesystem and tensor-shape contract for one patch shard."""

    bin_path: Path
    csv_path: Path
    image_size: int = 256
    channels: int = 3
    header_size: int = PATCH_SHARD_HEADER_SIZE
    magic: bytes = PATCH_SHARD_MAGIC
    validate_crc: bool = False


class PatchShard:
    """A UBC binary patch shard with CSV metadata."""

    def __init__(self, spec: PatchShardSpec) -> None:
        """Validate the shard and keep records for indexed reads."""
        self._spec = spec
        self._validate_shape()
        self._records = tuple(load_patch_records(spec.csv_path))
        self._patch_bytes = spec.channels * spec.image_size * spec.image_size
        self._header: PatchShardHeader | None = None
        self._crc_validated = False
        self._validate_records()
        self._validate_binary()

    @property
    def header(self) -> PatchShardHeader:
        """The parsed and validated shard header.

        Returns:
            Parsed header.

        Raises:
            RuntimeError: If validation has not populated the header.

        """
        if self._header is None:
            message = "Patch shard header has not been validated"
            raise RuntimeError(message)
        return self._header

    @property
    def crc_validated(self) -> bool:
        """Whether payload CRC32 was checked during construction.

        Returns:
            True if payload CRC32 was checked.

        """
        return self._crc_validated

    @property
    def records(self) -> tuple[PatchRecord, ...]:
        """Immutable patch records in CSV order.

        Returns:
            Patch records in metadata order.

        """
        return self._records

    def __len__(self) -> int:
        """Return the number of patch records.

        Returns:
            Number of patch records.

        """
        return len(self._records)

    def read_uint8(self, index: int) -> torch.Tensor:
        """Read one patch as a CHW uint8 tensor.

        Args:
            index: Binary patch index from `0` to `len(self) - 1`.

        Returns:
            Tensor with shape `(channels, image_size, image_size)`.

        Raises:
            EOFError: If the patch payload is truncated.
            IndexError: If the requested patch index is outside the shard.

        """
        if index < 0 or index >= len(self):
            message = f"Patch index {index} outside shard length {len(self)}"
            raise IndexError(message)

        offset = self._spec.header_size + index * self._patch_bytes
        buffer = bytearray(self._patch_bytes)
        with self._spec.bin_path.open("rb") as binary_file:
            binary_file.seek(offset)
            read_count = binary_file.readinto(buffer)
        if read_count != self._patch_bytes:
            message = (
                f"Expected {self._patch_bytes} bytes for patch {index}, "
                f"read {read_count}"
            )
            raise EOFError(message)

        return (
            torch
            .frombuffer(buffer, dtype=torch.uint8)
            .reshape(self._spec.channels, self._spec.image_size, self._spec.image_size)
            .clone()
        )

    def read_normalized(self, index: int) -> torch.Tensor:
        """Read one patch normalized to `[-1, 1]` in FP32.

        Returns:
            FP32 normalized patch tensor.

        """
        return self.read_uint8(index).to(dtype=torch.float32).div(127.5).sub(1.0)

    def _validate_shape(self) -> None:
        if self._spec.channels <= 0:
            message = f"channels must be positive, got {self._spec.channels}"
            raise ValueError(message)
        if self._spec.image_size <= 0:
            message = f"image_size must be positive, got {self._spec.image_size}"
            raise ValueError(message)
        if self._spec.header_size < len(self._spec.magic):
            message = "header_size must be at least the magic length"
            raise ValueError(message)

    def _validate_records(self) -> None:
        expected_indices = tuple(range(len(self._records)))
        observed_indices = tuple(sorted(record.file_index for record in self._records))
        if observed_indices != expected_indices:
            message = (
                "Patch CSV idx values must be unique and contiguous from 0 to "
                f"{len(self._records) - 1}"
            )
            raise ValueError(message)

    def _validate_binary(self) -> None:
        with self._spec.bin_path.open("rb") as binary_file:
            header_bytes = binary_file.read(self._spec.header_size)
        header = parse_patch_shard_header(
            header_bytes=header_bytes,
            expected_magic=self._spec.magic,
        )
        self._validate_header_fields(header)
        self._header = header

        expected_size = self._spec.header_size + len(self._records) * self._patch_bytes
        actual_size = self._spec.bin_path.stat().st_size
        if actual_size != expected_size:
            message = (
                f"Patch shard size mismatch for {self._spec.bin_path}: "
                f"expected {expected_size} bytes, found {actual_size}"
            )
            raise ValueError(message)

        if self._spec.validate_crc:
            observed_crc = compute_patch_payload_crc(
                bin_path=self._spec.bin_path,
                header_size=self._spec.header_size,
            )
            if observed_crc != header.crc32:
                message = (
                    f"Patch shard CRC32 mismatch for {self._spec.bin_path}: "
                    f"expected {header.crc32}, observed {observed_crc}"
                )
                raise ValueError(message)
            self._crc_validated = True

    def _validate_header_fields(self, header: PatchShardHeader) -> None:
        if header.version != PATCH_SHARD_VERSION:
            message = f"Unsupported patch-shard version {header.version}"
            raise ValueError(message)
        if header.layout != PATCH_SHARD_LAYOUT:
            message = f"Unsupported patch-shard layout {header.layout!r}"
            raise ValueError(message)
        if header.channels != self._spec.channels:
            message = (
                f"Header channels {header.channels} do not match "
                f"expected {self._spec.channels}"
            )
            raise ValueError(message)
        if (
            header.height != self._spec.image_size
            or header.width != self._spec.image_size
        ):
            message = (
                f"Header image size {header.height}x{header.width} does not match "
                f"expected {self._spec.image_size}x{self._spec.image_size}"
            )
            raise ValueError(message)
        if header.patch_count != len(self._records):
            message = (
                f"Header patch count {header.patch_count} does not match "
                f"CSV row count {len(self._records)}"
            )
            raise ValueError(message)


def load_patch_records(csv_path: Path) -> list[PatchRecord]:
    """Load patch metadata rows by column name.

    The `idx` column is optional. If it is absent, CSV row order is used as the
    binary patch index. If it is present, every row must define a nonblank
    integer `idx`.

    Args:
        csv_path: Patch metadata CSV.

    Returns:
        Patch records in CSV order.

    """
    with csv_path.open(encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        _validate_csv_columns(csv_path=csv_path, fieldnames=reader.fieldnames)
        idx_column_present = "idx" in (reader.fieldnames or ())
        records: list[PatchRecord] = []
        for row_number, raw_row in enumerate(reader):
            row = cast("Mapping[str, str | None]", raw_row)
            records.append(
                _parse_record(
                    row=row,
                    row_number=row_number,
                    idx_column_present=idx_column_present,
                ),
            )
    return records


def _validate_csv_columns(csv_path: Path, fieldnames: Sequence[str] | None) -> None:
    if fieldnames is None:
        message = f"Patch metadata CSV has no header: {csv_path}"
        raise ValueError(message)
    missing = sorted(PATCH_CSV_COLUMNS.difference(fieldnames))
    if missing:
        message = f"Patch metadata CSV {csv_path} is missing columns: {missing}"
        raise ValueError(message)


def _parse_record(
    *,
    row: Mapping[str, str | None],
    row_number: int,
    idx_column_present: bool,
) -> PatchRecord:
    idx_value = row.get("idx")
    if idx_value is None and not idx_column_present:
        file_index = row_number
    else:
        file_index = _parse_int(idx_value, column="idx", row_number=row_number)
    label = _parse_int(_required(row, "label", row_number), "label", row_number)
    if label not in PATCH_LABELS:
        message = f"Invalid label {label} on row {row_number}; expected 0..4"
        raise ValueError(message)
    return PatchRecord(
        file_index=file_index,
        row_index=row_number,
        wsi_id=_required(row, "wsi_id", row_number),
        label=label,
        x=_parse_int(_required(row, "x", row_number), "x", row_number),
        y=_parse_int(_required(row, "y", row_number), "y", row_number),
    )


def _required(row: Mapping[str, str | None], column: str, row_number: int) -> str:
    value = row.get(column)
    if not value:
        message = f"Missing {column!r} value on patch CSV row {row_number}"
        raise ValueError(message)
    return value


def _parse_int(value: str | None, column: str, row_number: int) -> int:
    if not value:
        message = f"Missing {column!r} value on patch CSV row {row_number}"
        raise ValueError(message)
    try:
        parsed = int(value)
    except ValueError as error:
        message = f"Invalid integer {value!r} for {column!r} on row {row_number}"
        raise ValueError(message) from error
    if parsed < 0:
        message = f"Negative integer {parsed} for {column!r} on row {row_number}"
        raise ValueError(message)
    return parsed


def parse_patch_shard_header(
    *,
    header_bytes: bytes,
    expected_magic: bytes = PATCH_SHARD_MAGIC,
) -> PatchShardHeader:
    """Parse and validate the fixed-size UBC patch-shard header.

    Returns:
        Parsed patch-shard header.

    Raises:
        ValueError: If the byte length or magic value is invalid.

    """
    if len(header_bytes) != PATCH_SHARD_HEADER_SIZE:
        message = f"Expected {PATCH_SHARD_HEADER_SIZE}-byte header"
        raise ValueError(message)
    unpacked = cast(
        "tuple[bytes, int, int, int, int, int, int, bytes]",
        struct.unpack(PATCH_SHARD_HEADER_FORMAT, header_bytes),
    )
    magic, crc32, patch_count, channels, height, width, version, layout = unpacked
    if magic != expected_magic:
        message = f"Missing {expected_magic!r} magic in patch-shard header"
        raise ValueError(message)
    return PatchShardHeader(
        crc32=crc32,
        patch_count=patch_count,
        channels=channels,
        height=height,
        width=width,
        version=version,
        layout=layout,
    )


def make_patch_shard_header(
    *,
    header: PatchShardHeader,
    magic: bytes = PATCH_SHARD_MAGIC,
) -> bytes:
    """Pack a 64-byte UBC patch-shard header.

    Returns:
        Packed 64-byte header.

    """
    return struct.pack(
        PATCH_SHARD_HEADER_FORMAT,
        magic,
        header.crc32 & 0xFFFFFFFF,
        header.patch_count,
        header.channels,
        header.height,
        header.width,
        header.version,
        header.layout,
    )


def compute_patch_payload_crc(*, bin_path: Path, header_size: int) -> int:
    """Compute CRC32 over patch payload bytes.

    Returns:
        Unsigned CRC32 checksum.

    """
    checksum = 0
    with bin_path.open("rb") as binary_file:
        binary_file.seek(header_size)
        while True:
            chunk = binary_file.read(CRC_CHUNK_BYTES)
            if not chunk:
                break
            checksum = zlib.crc32(chunk, checksum)
    return checksum & 0xFFFFFFFF


__all__ = [
    "PATCH_CSV_COLUMNS",
    "PATCH_LABELS",
    "PATCH_SHARD_HEADER_FORMAT",
    "PATCH_SHARD_HEADER_SIZE",
    "PATCH_SHARD_LAYOUT",
    "PATCH_SHARD_MAGIC",
    "PATCH_SHARD_VERSION",
    "PatchRecord",
    "PatchShard",
    "PatchShardHeader",
    "PatchShardSpec",
    "compute_patch_payload_crc",
    "load_patch_records",
    "make_patch_shard_header",
    "parse_patch_shard_header",
]
