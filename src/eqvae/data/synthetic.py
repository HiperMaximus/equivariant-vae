# Copyright 2026 HiperMaximus
"""Tiny deterministic synthetic patch metadata and shard helpers."""

from __future__ import annotations

import csv
import zlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from eqvae.data.patch_shards import (
    PatchRecord,
    PatchShardHeader,
    make_patch_shard_header,
)

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class SyntheticPatchSpec:
    """Small synthetic patch-set descriptor for local smoke commands."""

    count: int = 32
    image_size: int = 256
    channels: int = 3
    seed: int = 20260612


def synthetic_patch_ids(spec: SyntheticPatchSpec) -> list[str]:
    """Return deterministic synthetic sample identifiers.

    Returns:
        Synthetic sample IDs in stable order.

    """
    return [f"synthetic_{index:05d}" for index in range(spec.count)]


def make_synthetic_patches(spec: SyntheticPatchSpec) -> torch.Tensor:
    """Create deterministic CHW uint8 patches for local tests.

    Args:
        spec: Synthetic patch-set descriptor.

    Returns:
        Tensor with shape `(count, channels, image_size, image_size)`.

    """
    _validate_spec(spec)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(spec.seed)
    return torch.randint(
        low=0,
        high=256,
        size=(spec.count, spec.channels, spec.image_size, spec.image_size),
        generator=generator,
        dtype=torch.uint8,
    )


def synthetic_patch_records(spec: SyntheticPatchSpec) -> list[PatchRecord]:
    """Create deterministic patch metadata matching synthetic patches.

    Returns:
        Patch metadata records.

    """
    _validate_spec(spec)
    return [
        PatchRecord(
            file_index=index,
            row_index=index,
            wsi_id=f"synthetic_wsi_{index // 5:04d}",
            label=index % 5,
            x=(index * 17) % 10_000,
            y=(index * 31) % 10_000,
        )
        for index in range(spec.count)
    ]


def write_synthetic_patch_shard(
    *,
    bin_path: Path,
    csv_path: Path,
    spec: SyntheticPatchSpec,
    include_idx: bool = True,
) -> torch.Tensor:
    """Write a deterministic tiny UBC-format patch shard.

    Args:
        bin_path: Output binary path.
        csv_path: Output metadata CSV path.
        spec: Synthetic patch-set descriptor.
        include_idx: Whether to write the optional `idx` column.

    Returns:
        The generated uint8 patches.

    """
    patches = make_synthetic_patches(spec)
    records = synthetic_patch_records(spec)
    payload_buffer = bytearray(patches.numel())
    payload_tensor = torch.frombuffer(payload_buffer, dtype=torch.uint8)
    payload_tensor.copy_(patches.contiguous().view(-1))
    payload = bytes(payload_buffer)
    header = make_patch_shard_header(
        header=PatchShardHeader(
            crc32=zlib.crc32(payload) & 0xFFFFFFFF,
            patch_count=spec.count,
            channels=spec.channels,
            height=spec.image_size,
            width=spec.image_size,
            version=1,
            layout=b"CHW",
        ),
    )

    bin_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with bin_path.open("wb") as binary_file:
        binary_file.write(header)
        binary_file.write(payload)

    fieldnames = ["wsi_id", "label", "x", "y"]
    if include_idx:
        fieldnames.insert(0, "idx")
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for record in records:
            row = {
                "wsi_id": record.wsi_id,
                "label": str(record.label),
                "x": str(record.x),
                "y": str(record.y),
            }
            if include_idx:
                row["idx"] = str(record.file_index)
            writer.writerow(row)

    return patches


def _validate_spec(spec: SyntheticPatchSpec) -> None:
    if spec.count <= 0:
        message = f"count must be positive, got {spec.count}"
        raise ValueError(message)
    if spec.channels <= 0:
        message = f"channels must be positive, got {spec.channels}"
        raise ValueError(message)
    if spec.image_size <= 0:
        message = f"image_size must be positive, got {spec.image_size}"
        raise ValueError(message)


__all__ = [
    "SyntheticPatchSpec",
    "make_synthetic_patches",
    "synthetic_patch_ids",
    "synthetic_patch_records",
    "write_synthetic_patch_shard",
]
