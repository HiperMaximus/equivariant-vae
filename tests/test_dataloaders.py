# Copyright 2026 HiperMaximus
"""Tests for the fast tensor-only patch dataloader rail."""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data._utils.pin_memory import (  # noqa: PLC2701
    pin_memory as torch_pin_memory,  # pyright: ignore[reportUnknownVariableType]
)

from eqvae.data.dataloaders import (
    PatchTensorDataset,
    PatchTensorDatasetSpec,
    normalize_uint8_batch,
)
from eqvae.data.synthetic import SyntheticPatchSpec, write_synthetic_patch_shard
from eqvae.data.training_batches import PatchTrainingBatch

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

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


def test_training_batch_implements_the_dataloader_pin_memory_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``DataLoader(pin_memory=True)`` must really pin our batch, not silently skip it.

    ``torch.utils.data._utils.pin_memory.pin_memory`` dispatches on ``isinstance
    Tensor`` -> ``hasattr(data, "pin_memory")`` -> Mapping -> tuple/namedtuple ->
    Sequence. ``PatchTrainingBatch`` is a plain dataclass, so it matched NONE of those
    and the helper returned it UNCHANGED: ``pin_memory=True`` was a silent no-op and
    every ``non_blocking=True`` H2D silently degraded to a synchronous copy from
    pageable memory. The batch therefore implements the ``pin_memory()`` hook, the
    branch torch provides for custom batch types.

    Pinning is stubbed because it requires an accelerator and the repo's tests are
    CPU-only (AGENTS.md rule 24); what is under test is the DISPATCH, not torch's
    page-locking.
    """
    pinned: list[str] = []

    def fake_pin(self: Tensor) -> Tensor:
        pinned.append("called")
        return self

    monkeypatch.setattr(torch.Tensor, "pin_memory", fake_pin)
    batch = PatchTrainingBatch(
        images_uint8=torch.zeros((2, 3, 4, 4), dtype=torch.uint8),
        split="train",
        file_indices=(0, 0),
        row_indices=(0, 1),
        wsi_ids=("a", "a"),
        labels=(0, 0),
        xs=(0, 0),
        ys=(0, 0),
        semantic_sample_keys=("k0", "k1"),
        sample_ids=("s0", "s1"),
    )

    result = cast("PatchTrainingBatch", torch_pin_memory(batch))

    # The image tensor was pinned exactly once, and the batch was genuinely rebuilt --
    # a fall-through would return the SAME object with nothing pinned.
    assert pinned == ["called"]
    assert result is not batch
    # Host-only provenance is shared, never pinned or rebuilt per batch.
    assert result.sample_ids is batch.sample_ids
    assert result.split == batch.split
