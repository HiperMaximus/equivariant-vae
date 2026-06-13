# Copyright 2026 HiperMaximus
"""Fast mmap-backed tensor datasets for UBC patch shards."""

from __future__ import annotations

import mmap
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, BinaryIO

import torch
from torch import Tensor
from torch.utils.data import Dataset

from eqvae.data.patch_shards import (
    PATCH_SHARD_HEADER_SIZE,
    PatchRecord,
    PatchShard,
    PatchShardSpec,
)
from eqvae.data.roots import PatchSplit, normalize_patch_split

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class PatchTensorDatasetSpec:
    """Fast tensor-only dataset configuration."""

    bin_path: Path
    csv_path: Path
    split: PatchSplit
    image_size: int = 256
    channels: int = 3
    validate_crc: bool = False
    madvise_sequential: bool = True


class PatchTensorDataset(Dataset[Tensor]):
    """Read UBC patch tensors through a worker-local read-only mmap.

    This is the hot-path training/benchmark dataset. It intentionally returns
    only CHW `uint8` tensors so metadata handling does not pollute loader
    throughput measurements.
    """

    def __init__(self, spec: PatchTensorDatasetSpec) -> None:
        """Validate the shard and keep only offset metadata for fast reads."""
        split = normalize_patch_split(spec.split)
        self._spec = PatchTensorDatasetSpec(
            bin_path=spec.bin_path,
            csv_path=spec.csv_path,
            split=split,
            image_size=spec.image_size,
            channels=spec.channels,
            validate_crc=spec.validate_crc,
            madvise_sequential=spec.madvise_sequential,
        )
        shard = PatchShard(
            PatchShardSpec(
                bin_path=spec.bin_path,
                csv_path=spec.csv_path,
                image_size=spec.image_size,
                channels=spec.channels,
                validate_crc=spec.validate_crc,
            ),
        )
        self._records = shard.records
        self._header_size = PATCH_SHARD_HEADER_SIZE
        self._patch_bytes = spec.channels * spec.image_size * spec.image_size
        self._file: BinaryIO | None = None
        self._mmap: mmap.mmap | None = None

    @property
    def split(self) -> PatchSplit:
        """Return the canonical split name.

        Returns:
            Canonical split name.

        """
        return self._spec.split

    @property
    def records(self) -> tuple[PatchRecord, ...]:
        """Return immutable CSV records for non-hot-path audit code.

        Returns:
            Patch records in CSV order.

        """
        return self._records

    def __len__(self) -> int:
        """Return the number of rows in the dataset.

        Returns:
            Number of rows.

        """
        return len(self._records)

    def __getitem__(self, index: int) -> Tensor:
        """Return one CHW `uint8` patch tensor from the mmap.

        Args:
            index: Dataset row index.

        Returns:
            CHW `uint8` tensor backed by the read-only mmap.

        """
        return self._read_file_index(self.file_index_for_row(index))

    def file_index_for_row(self, index: int) -> int:
        """Return the binary file index associated with one dataset row.

        Args:
            index: Dataset row index.

        Returns:
            Binary patch offset index.

        Raises:
            IndexError: If the row index is outside the dataset.

        """
        if index < 0 or index >= len(self):
            message = f"Dataset index {index} outside length {len(self)}"
            raise IndexError(message)
        return self._records[index].file_index

    def read_by_file_index(self, file_index: int) -> Tensor:
        """Read one patch by canonical binary file index.

        Selector validation/replay code can use this when it already has a
        validated `file_index`; normal dataset indexing should use row indices.

        Args:
            file_index: Binary patch offset index.

        Returns:
            CHW `uint8` tensor backed by the read-only mmap.

        Raises:
            IndexError: If the file index is outside the shard.

        """
        if file_index < 0 or file_index >= len(self):
            message = f"Patch file_index {file_index} outside length {len(self)}"
            raise IndexError(message)
        return self._read_file_index(file_index)

    def _read_file_index(self, file_index: int) -> Tensor:
        mapping = self._ensure_mmap()
        offset = self._header_size + file_index * self._patch_bytes
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The given buffer is not writable",
                category=UserWarning,
            )
            patch = torch.frombuffer(
                mapping,
                dtype=torch.uint8,
                count=self._patch_bytes,
                offset=offset,
            )
        return patch.reshape(
            self._spec.channels,
            self._spec.image_size,
            self._spec.image_size,
        )

    def close(self) -> None:
        """Close the mmap and file handle if they are open."""
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._file is not None:
            self._file.close()
            self._file = None

    def __getstate__(self) -> dict[str, object]:
        """Drop process-local file handles during worker pickling.

        Returns:
            Picklable dataset state.

        """
        state = dict(self.__dict__)
        state["_file"] = None
        state["_mmap"] = None
        return state

    def __del__(self) -> None:
        """Release OS resources during normal interpreter cleanup."""
        self.close()

    def _ensure_mmap(self) -> mmap.mmap:
        if self._mmap is None:
            self._file = self._spec.bin_path.open("rb")
            self._mmap = mmap.mmap(
                self._file.fileno(),
                length=0,
                access=mmap.ACCESS_READ,
            )
            if self._spec.madvise_sequential:
                _madvise_sequential(self._mmap)
        return self._mmap


def normalize_uint8_batch(batch: Tensor) -> Tensor:
    """Normalize a `uint8` batch to FP32 `[-1, 1]`.

    Returns:
        FP32 normalized batch.

    Raises:
        TypeError: If the batch is not `uint8`.

    """
    if batch.dtype != torch.uint8:
        message = f"Expected uint8 batch, got {batch.dtype}"
        raise TypeError(message)
    return batch.to(dtype=torch.float32).div(127.5).sub(1.0)


def _madvise_sequential(mapping: mmap.mmap) -> None:
    try:
        mapping.madvise(mmap.MADV_SEQUENTIAL)
    except (AttributeError, OSError, ValueError):
        return


__all__ = [
    "PatchTensorDataset",
    "PatchTensorDatasetSpec",
    "normalize_uint8_batch",
]
