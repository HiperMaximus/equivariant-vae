# Copyright 2026 HiperMaximus
"""Metadata-carrying training batches for corruption and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.utils.data import Dataset

from eqvae.corruption.stain import semantic_sample_key
from eqvae.data.dataloaders import PatchTensorDataset, PatchTensorDatasetSpec
from eqvae.data.roots import PatchSplit, normalize_patch_split

if TYPE_CHECKING:
    from pathlib import Path

    from eqvae.data.patch_shards import PatchRecord


@dataclass(frozen=True)
class PatchTrainingDatasetSpec:
    """Dataset spec for training/evaluation batches that need metadata."""

    bin_path: Path
    csv_path: Path
    split: PatchSplit
    image_size: int = 256
    channels: int = 3
    validate_crc: bool = False
    madvise_sequential: bool = True


@dataclass(frozen=True)
class PatchTrainingSample:
    """One image plus the metadata required by deterministic corruption."""

    image_uint8: Tensor
    split: PatchSplit
    file_index: int
    row_index: int
    wsi_id: str
    label: int
    x: int
    y: int
    semantic_sample_key: str
    sample_id: str


@dataclass(frozen=True)
class PatchTrainingBatch:
    """Collated image batch plus per-sample provenance."""

    images_uint8: Tensor
    split: PatchSplit
    file_indices: tuple[int, ...]
    row_indices: tuple[int, ...]
    wsi_ids: tuple[str, ...]
    labels: tuple[int, ...]
    xs: tuple[int, ...]
    ys: tuple[int, ...]
    semantic_sample_keys: tuple[str, ...]
    sample_ids: tuple[str, ...]


class PatchTrainingDataset(Dataset[PatchTrainingSample]):
    """Patch dataset for trainer/evaluator code that needs metadata.

    `PatchTensorDataset` remains the tensor-only throughput benchmark rail. This
    wrapper composes it for real training/evaluation paths where corruption RNG,
    metric rows, and visual QA need stable sample identities.
    """

    def __init__(self, spec: PatchTrainingDatasetSpec) -> None:
        """Create the metadata-carrying dataset."""
        split: PatchSplit = normalize_patch_split(spec.split)
        self._split: PatchSplit = split
        self._tensor_dataset = PatchTensorDataset(
            PatchTensorDatasetSpec(
                bin_path=spec.bin_path,
                csv_path=spec.csv_path,
                split=split,
                image_size=spec.image_size,
                channels=spec.channels,
                validate_crc=spec.validate_crc,
                madvise_sequential=spec.madvise_sequential,
            ),
        )

    @property
    def split(self) -> PatchSplit:
        """Return the canonical split name.

        Returns:
            Canonical split name.

        """
        return self._split

    @property
    def records(self) -> tuple[PatchRecord, ...]:
        """Return immutable patch records in CSV row order.

        Returns:
            Patch records.

        """
        return self._tensor_dataset.records

    def __len__(self) -> int:
        """Return the dataset length.

        Returns:
            Number of patch records.

        """
        return len(self._tensor_dataset)

    def __getitem__(self, index: int) -> PatchTrainingSample:
        """Return one uint8 image plus stable corruption metadata.

        Args:
            index: Dataset row index.

        Returns:
            Training sample with image and metadata.

        """
        record = self.records[index]
        return PatchTrainingSample(
            image_uint8=self._tensor_dataset[index],
            split=self._split,
            file_index=record.file_index,
            row_index=record.row_index,
            wsi_id=record.wsi_id,
            label=record.label,
            x=record.x,
            y=record.y,
            semantic_sample_key=semantic_key_for_record(record, split=self._split),
            sample_id=record.sample_id(self._split),
        )

    def close(self) -> None:
        """Close worker-local file handles."""
        self._tensor_dataset.close()


def semantic_key_for_record(record: PatchRecord, *, split: PatchSplit) -> str:
    """Return the corruption semantic key for one patch record.

    Returns:
        `{split}:{wsi_id}:{label}:{x}:{y}` semantic key.

    """
    return semantic_sample_key(
        split=split,
        wsi_id=record.wsi_id,
        label=record.label,
        x=record.x,
        y=record.y,
    )


def collate_patch_training_samples(
    samples: list[PatchTrainingSample],
) -> PatchTrainingBatch:
    """Collate metadata-carrying samples into one training batch.

    Returns:
        Batch with stacked CHW uint8 images and tuple metadata.

    Raises:
        ValueError: If the batch is empty or mixes splits.

    """
    if not samples:
        message = "Cannot collate an empty patch-training batch"
        raise ValueError(message)
    split = samples[0].split
    if any(sample.split != split for sample in samples):
        message = "Patch-training batches must not mix splits"
        raise ValueError(message)
    return PatchTrainingBatch(
        images_uint8=torch.stack(
            [sample.image_uint8 for sample in samples],
            dim=0,
        ),
        split=split,
        file_indices=tuple(sample.file_index for sample in samples),
        row_indices=tuple(sample.row_index for sample in samples),
        wsi_ids=tuple(sample.wsi_id for sample in samples),
        labels=tuple(sample.label for sample in samples),
        xs=tuple(sample.x for sample in samples),
        ys=tuple(sample.y for sample in samples),
        semantic_sample_keys=tuple(sample.semantic_sample_key for sample in samples),
        sample_ids=tuple(sample.sample_id for sample in samples),
    )


__all__ = [
    "PatchTrainingBatch",
    "PatchTrainingDataset",
    "PatchTrainingDatasetSpec",
    "PatchTrainingSample",
    "collate_patch_training_samples",
    "semantic_key_for_record",
]
