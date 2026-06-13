# Copyright 2026 HiperMaximus
"""Tests for the fast tensor-only patch dataloader rail."""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from eqvae.data.dataloaders import (
    PatchTensorDataset,
    PatchTensorDatasetSpec,
    normalize_uint8_batch,
)
from eqvae.data.synthetic import SyntheticPatchSpec, write_synthetic_patch_shard

if TYPE_CHECKING:
    from pathlib import Path

PATCH_COUNT = 6
PATCH_SIZE = 8
CHANNELS = 3
BATCH_SIZE = 2
FIRST_ROW = 0
FIRST_FILE_INDEX_AFTER_REORDER = 5
FIRST_FILE_INDEX = 0


def test_tensor_dataset_distinguishes_row_index_from_file_index(
    tmp_path: Path,
) -> None:
    """Validation CSV row order can differ from binary patch offsets."""
    bin_path = tmp_path / "validation.bin"
    csv_path = tmp_path / "validation.csv"
    patches = write_synthetic_patch_shard(
        bin_path=bin_path,
        csv_path=csv_path,
        spec=SyntheticPatchSpec(count=PATCH_COUNT, image_size=PATCH_SIZE),
        include_idx=True,
    )
    _reverse_csv_data_rows(csv_path)
    dataset = PatchTensorDataset(
        PatchTensorDatasetSpec(
            bin_path=bin_path,
            csv_path=csv_path,
            split="validation",
            image_size=PATCH_SIZE,
            channels=CHANNELS,
        ),
    )

    assert dataset.file_index_for_row(FIRST_ROW) == FIRST_FILE_INDEX_AFTER_REORDER
    assert torch.equal(dataset[FIRST_ROW], patches[FIRST_FILE_INDEX_AFTER_REORDER])
    assert torch.equal(dataset.read_by_file_index(FIRST_FILE_INDEX), patches[0])

    dataset.close()


def test_tensor_dataset_batches_uint8_and_normalizes(tmp_path: Path) -> None:
    """The hot path returns tensor batches without selector metadata."""
    bin_path = tmp_path / "train.bin"
    csv_path = tmp_path / "train.csv"
    write_synthetic_patch_shard(
        bin_path=bin_path,
        csv_path=csv_path,
        spec=SyntheticPatchSpec(count=PATCH_COUNT, image_size=PATCH_SIZE),
        include_idx=False,
    )
    dataset = PatchTensorDataset(
        PatchTensorDatasetSpec(
            bin_path=bin_path,
            csv_path=csv_path,
            split="train",
            image_size=PATCH_SIZE,
            channels=CHANNELS,
        ),
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0)

    batch = cast("Tensor", next(iter(loader)))
    normalized = normalize_uint8_batch(batch)

    assert batch.dtype == torch.uint8
    assert tuple(batch.shape) == (BATCH_SIZE, CHANNELS, PATCH_SIZE, PATCH_SIZE)
    assert normalized.dtype == torch.float32
    assert float(normalized.min()) >= -1.0
    assert float(normalized.max()) <= 1.0

    dataset.close()


def test_tensor_dataset_worker_pickle_drops_open_handles(tmp_path: Path) -> None:
    """Worker pickling starts from a clean mmap/file-handle state."""
    bin_path = tmp_path / "train.bin"
    csv_path = tmp_path / "train.csv"
    write_synthetic_patch_shard(
        bin_path=bin_path,
        csv_path=csv_path,
        spec=SyntheticPatchSpec(count=PATCH_COUNT, image_size=PATCH_SIZE),
        include_idx=False,
    )
    dataset = PatchTensorDataset(
        PatchTensorDatasetSpec(
            bin_path=bin_path,
            csv_path=csv_path,
            split="train",
            image_size=PATCH_SIZE,
            channels=CHANNELS,
        ),
    )
    _ = dataset[FIRST_ROW]

    state = dataset.__getstate__()

    assert state["_file"] is None
    assert state["_mmap"] is None

    dataset.close()


def _reverse_csv_data_rows(csv_path: Path) -> None:
    with csv_path.open(encoding="utf-8", newline="") as csv_file:
        rows = list(csv.reader(csv_file))
    header = rows[0]
    data_rows = list(reversed(rows[1:]))
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(data_rows)
