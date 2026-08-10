# Copyright 2026 HiperMaximus
"""Tests for spec 0001 patch-shard loading."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from eqvae.data.patch_shards import (
    PATCH_SHARD_HEADER_SIZE,
    PATCH_SHARD_LAYOUT,
    PATCH_SHARD_VERSION,
    PatchShard,
    PatchShardSpec,
    compute_patch_payload_crc,
    load_patch_records,
)
from eqvae.data.synthetic import SyntheticPatchSpec, write_synthetic_patch_shard

if TYPE_CHECKING:
    from pathlib import Path

PATCH_COUNT = 4
PATCH_SIZE = 16
CHANNELS = 3
FIRST_PATCH_INDEX = 0
SECOND_PATCH_INDEX = 1
MIN_NORMALIZED_VALUE = -1.0
MAX_NORMALIZED_VALUE = 1.0


def test_synthetic_patch_shard_uses_real_header_and_crc(tmp_path: Path) -> None:
    """Synthetic shards follow the same binary/CSV contract as UBC shards."""
    bin_path = tmp_path / "synthetic.bin"
    csv_path = tmp_path / "synthetic.csv"
    spec = SyntheticPatchSpec(
        count=PATCH_COUNT,
        image_size=PATCH_SIZE,
        channels=CHANNELS,
        seed=7,
    )

    patches = write_synthetic_patch_shard(
        bin_path=bin_path,
        csv_path=csv_path,
        spec=spec,
    )
    shard = PatchShard(
        PatchShardSpec(
            bin_path=bin_path,
            csv_path=csv_path,
            image_size=PATCH_SIZE,
            channels=CHANNELS,
            validate_crc=True,
        ),
    )

    assert len(shard) == PATCH_COUNT
    assert shard.crc_validated is True
    assert shard.header.patch_count == PATCH_COUNT
    assert shard.header.channels == CHANNELS
    assert shard.header.height == PATCH_SIZE
    assert shard.header.width == PATCH_SIZE
    assert shard.header.version == PATCH_SHARD_VERSION
    assert shard.header.layout == PATCH_SHARD_LAYOUT
    assert shard.header.crc32 == compute_patch_payload_crc(
        bin_path=bin_path,
        header_size=PATCH_SHARD_HEADER_SIZE,
    )
    assert torch.equal(shard.read_uint8(FIRST_PATCH_INDEX), patches[FIRST_PATCH_INDEX])
    normalized = shard.read_normalized(SECOND_PATCH_INDEX)
    assert normalized.dtype == torch.float32
    assert float(normalized.min()) >= MIN_NORMALIZED_VALUE
    assert float(normalized.max()) <= MAX_NORMALIZED_VALUE
    assert (
        shard
        .records[SECOND_PATCH_INDEX]
        .sample_id("train")
        .startswith(
            "train:00000001:synthetic_wsi_0000:",
        )
    )


def test_patch_records_use_row_order_when_idx_is_absent(tmp_path: Path) -> None:
    """The train CSV variant without `idx` maps row order to file offsets."""
    bin_path = tmp_path / "synthetic.bin"
    csv_path = tmp_path / "synthetic.csv"
    write_synthetic_patch_shard(
        bin_path=bin_path,
        csv_path=csv_path,
        spec=SyntheticPatchSpec(count=PATCH_COUNT, image_size=PATCH_SIZE),
        include_idx=False,
    )

    records = load_patch_records(csv_path)
    shard = PatchShard(
        PatchShardSpec(bin_path=bin_path, csv_path=csv_path, image_size=PATCH_SIZE),
    )

    assert [record.file_index for record in records] == list(range(PATCH_COUNT))
    assert [record.row_index for record in records] == list(range(PATCH_COUNT))
    assert [record.file_index for record in shard.records] == list(range(PATCH_COUNT))


def test_patch_records_reject_blank_idx_when_idx_column_exists(tmp_path: Path) -> None:
    """A present `idx` column must define every binary file index."""
    bin_path = tmp_path / "synthetic.bin"
    csv_path = tmp_path / "synthetic.csv"
    write_synthetic_patch_shard(
        bin_path=bin_path,
        csv_path=csv_path,
        spec=SyntheticPatchSpec(count=PATCH_COUNT, image_size=PATCH_SIZE),
    )
    text = csv_path.read_text(encoding="utf-8")
    csv_path.write_text(text.replace("\n1,", "\n,", 1), encoding="utf-8")

    with pytest.raises(ValueError, match="Missing 'idx'"):
        load_patch_records(csv_path)


def test_patch_shard_rejects_crc_mismatch(tmp_path: Path) -> None:
    """A shard with changed payload bytes must fail before training reads it.

    CRC is the deliberate integrity guard linking CSV metadata to the binary payload;
    flipping one byte catches a loader that skips or miscomputes that verification.
    """
    bin_path = tmp_path / "synthetic.bin"
    csv_path = tmp_path / "synthetic.csv"
    write_synthetic_patch_shard(
        bin_path=bin_path,
        csv_path=csv_path,
        spec=SyntheticPatchSpec(count=PATCH_COUNT, image_size=PATCH_SIZE),
    )
    with bin_path.open("r+b") as binary_file:
        binary_file.seek(-1, 2)
        byte = binary_file.read(1)
        binary_file.seek(-1, 2)
        binary_file.write(bytes([byte[0] ^ 1]))

    with pytest.raises(ValueError, match="CRC32 mismatch"):
        PatchShard(
            PatchShardSpec(
                bin_path=bin_path,
                csv_path=csv_path,
                image_size=PATCH_SIZE,
                validate_crc=True,
            ),
        )


def test_patch_shard_rejects_non_contiguous_idx(tmp_path: Path) -> None:
    """Optional `idx` must be a contiguous binary-offset index."""
    bin_path = tmp_path / "synthetic.bin"
    csv_path = tmp_path / "synthetic.csv"
    write_synthetic_patch_shard(
        bin_path=bin_path,
        csv_path=csv_path,
        spec=SyntheticPatchSpec(count=PATCH_COUNT, image_size=PATCH_SIZE),
    )
    text = csv_path.read_text(encoding="utf-8")
    csv_path.write_text(text.replace("\n1,", "\n3,", 1), encoding="utf-8")

    with pytest.raises(ValueError, match="unique and contiguous"):
        PatchShard(
            PatchShardSpec(
                bin_path=bin_path,
                csv_path=csv_path,
                image_size=PATCH_SIZE,
            ),
        )
