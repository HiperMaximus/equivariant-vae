# Copyright 2026 HiperMaximus
# ruff: noqa: DOC201, DOC501, EM101, EM102, PLR0916, TRY003, TRY301
"""Compact catalog-backed latent reads for the one Spec 0023 experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import mmap
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, BinaryIO, Final, Literal, Self, cast

import torch
from torch import Tensor

from eqvae.data.latent_shards import (
    LATENT_CHANNELS,
    LATENT_HEIGHT,
    LATENT_RECORD_BYTES,
    LATENT_SHARD_HEADER_SIZE,
    LATENT_VALUES,
    LATENT_WIDTH,
    ModelName,
    parse_latent_shard_header,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

CATALOG_HEADER: Final = (
    "part",
    "model_name",
    "kaggle_source",
    "binary_name",
    "row_count",
    "binary_bytes",
    "binary_sha256",
    "sidecar_name",
    "sidecar_bytes",
    "sidecar_sha256",
)
WSI_INSTANCE_HEADER: Final = (
    "instance_row",
    "atlas_row_index",
    "wsi_id",
    "x",
    "y",
    "diagnosis_label",
    "diagnosis_index",
    "split",
    "part",
    "file_index",
)
WSI_BAG_HEADER: Final = (
    "bag_row",
    "wsi_id",
    "diagnosis_label",
    "diagnosis_index",
    "split",
    "instance_start",
    "instance_count",
)
TISSUE_HEADER: Final = (
    "dataset_row",
    "atlas_row_index",
    "wsi_id",
    "x",
    "y",
    "tissue_label",
    "split",
    "selection_rank",
    "part",
    "file_index",
)
TISSUE_INDICES: Final = {"tumor": 0, "stroma": 1, "necrosis": 2}
SHA256_HEX_LENGTH: Final = 64
type Split = Literal["train", "validation", "test"]
type LearningSplit = Literal["train", "validation"]


@dataclass(frozen=True)
class CatalogPart:
    """One model-specific remote physical part."""

    part: int
    model_name: ModelName
    kaggle_source: str
    binary_name: str
    row_count: int
    binary_bytes: int
    binary_sha256: str
    sidecar_name: str
    sidecar_bytes: int
    sidecar_sha256: str


@dataclass(frozen=True)
class LogicalPointer:
    """One logical row's representation-independent physical pointer."""

    part: int
    file_index: int


@dataclass(frozen=True)
class PhysicalRead:
    """One actual read in monotonically grouped physical order."""

    logical_index: int
    part: int
    file_index: int


@dataclass(frozen=True)
class WSIInstance:
    """One frozen WSI-bag instance and its physical pointer."""

    instance_row: int
    atlas_row_index: int
    wsi_id: int
    x: int
    y: int
    diagnosis_label: str
    diagnosis_index: int
    split: Split
    pointer: LogicalPointer


@dataclass(frozen=True)
class WSIBag:
    """One WSI-labelled contiguous logical instance range."""

    bag_row: int
    wsi_id: int
    diagnosis_label: str
    diagnosis_index: int
    split: Split
    instance_start: int
    instance_count: int


@dataclass(frozen=True)
class LoadedWSIBag:
    """One complete WSI bag restored to logical coordinate order."""

    bag: WSIBag
    instances: tuple[WSIInstance, ...]
    latents: Tensor
    physical_reads: tuple[PhysicalRead, ...]


@dataclass(frozen=True)
class TissueInstance:
    """One frozen high-purity tissue example and its physical pointer."""

    dataset_row: int
    atlas_row_index: int
    wsi_id: int
    x: int
    y: int
    tissue_label: str
    tissue_index: int
    split: Split
    selection_rank: int | None
    pointer: LogicalPointer


@dataclass(frozen=True)
class LoadedTissueBatch:
    """One paired-order tissue batch restored after grouped physical reads."""

    instances: tuple[TissueInstance, ...]
    latents: Tensor
    labels: Tensor
    physical_reads: tuple[PhysicalRead, ...]


class SupervisedLatentStore:
    """Read catalog-selected fixed records from read-only mounted binaries."""

    def __init__(
        self,
        *,
        catalog_path: Path,
        model_name: ModelName,
        source_roots: Mapping[str, Path],
    ) -> None:
        """Load the compact catalog without opening any binary payload."""
        catalog = _load_catalog(catalog_path)
        self._parts = {
            part: record
            for (part, record_model), record in catalog.items()
            if record_model == model_name
        }
        if not self._parts:
            raise ValueError(f"Catalog contains no rows for {model_name}")
        missing_sources = {
            record.kaggle_source
            for record in self._parts.values()
            if record.kaggle_source not in source_roots
        }
        if missing_sources:
            raise ValueError(f"Missing mounted sources: {sorted(missing_sources)}")
        self._source_roots = dict(source_roots)
        for record in self._parts.values():
            self._validate_mounted_part(record)
        self._handles: dict[int, tuple[BinaryIO, mmap.mmap]] = {}

    def _validate_mounted_part(self, record: CatalogPart) -> None:
        root = self._source_roots[record.kaggle_source]
        binary_path = root / record.binary_name
        sidecar_path = root / record.sidecar_name
        if binary_path.stat().st_size != record.binary_bytes:
            raise ValueError(f"Mounted binary size differs for part {record.part}")
        sidecar_bytes = sidecar_path.read_bytes()
        if (
            len(sidecar_bytes) != record.sidecar_bytes
            or hashlib.sha256(sidecar_bytes).hexdigest() != record.sidecar_sha256
        ):
            raise ValueError(f"Mounted sidecar identity differs for part {record.part}")
        sidecar = cast("object", json.loads(sidecar_bytes))
        if not isinstance(sidecar, dict):
            raise TypeError("Mounted latent sidecar must be an object")
        metadata = cast("dict[str, object]", sidecar)
        tensor = metadata.get("tensor")
        if not isinstance(tensor, dict):
            raise TypeError("Mounted latent sidecar tensor metadata is missing")
        tensor_metadata = cast("dict[str, object]", tensor)
        with binary_path.open("rb") as handle:
            header = parse_latent_shard_header(handle.read(LATENT_SHARD_HEADER_SIZE))
        if (
            metadata.get("schema_version") != "spec0020.latent_shard.v1"
            or metadata.get("status") != "complete"
            or metadata.get("model_name") != record.model_name
            or metadata.get("file_size") != record.binary_bytes
            or metadata.get("payload_bytes")
            != record.binary_bytes - LATENT_SHARD_HEADER_SIZE
            or metadata.get("payload_crc32") != header.payload_crc32
            or tensor_metadata.get("count") != record.row_count
            or tensor_metadata.get("dtype") != "float32_le"
            or tensor_metadata.get("layout") != "CHW"
            or tensor_metadata.get("shape")
            != [LATENT_CHANNELS, LATENT_HEIGHT, LATENT_WIDTH]
            or header.tensor_count != record.row_count
        ):
            raise ValueError(f"Mounted sidecar contract differs for part {record.part}")

    def read_rows(
        self,
        pointers: Sequence[LogicalPointer],
    ) -> tuple[Tensor, tuple[PhysicalRead, ...]]:
        """Read grouped physical rows and restore their supplied logical order."""
        if not pointers:
            raise ValueError("A supervised latent batch may not be empty")
        reads = tuple(
            sorted(
                (
                    PhysicalRead(logical_index, pointer.part, pointer.file_index)
                    for logical_index, pointer in enumerate(pointers)
                ),
                key=lambda read: (read.part, read.file_index),
            ),
        )
        for read in reads:
            record = self._parts.get(read.part)
            if record is None:
                raise ValueError(f"Part {read.part} is absent for this model")
            if read.file_index not in range(record.row_count):
                raise IndexError(
                    f"Index {read.file_index} outside part {read.part} row count",
                )
        result = torch.empty(
            (len(pointers), LATENT_CHANNELS, LATENT_HEIGHT, LATENT_WIDTH),
            dtype=torch.float32,
        )
        run_start = 0
        while run_start < len(reads):
            first = reads[run_start]
            run_stop = run_start + 1
            while run_stop < len(reads):
                previous = reads[run_stop - 1]
                candidate = reads[run_stop]
                if (
                    candidate.part != previous.part
                    or candidate.file_index != previous.file_index + 1
                ):
                    break
                run_stop += 1

            run_length = run_stop - run_start
            mapping = self._mapping(first.part)
            offset = LATENT_SHARD_HEADER_SIZE + first.file_index * LATENT_RECORD_BYTES
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="The given buffer is not writable",
                    category=UserWarning,
                )
                values = torch.frombuffer(
                    mapping,
                    dtype=torch.float32,
                    count=run_length * LATENT_VALUES,
                    offset=offset,
                )
            logical_indices = torch.tensor(
                [read.logical_index for read in reads[run_start:run_stop]],
                dtype=torch.long,
            )
            result.index_copy_(
                0,
                logical_indices,
                values.reshape(
                    run_length,
                    LATENT_CHANNELS,
                    LATENT_HEIGHT,
                    LATENT_WIDTH,
                ),
            )
            run_start = run_stop
        return result, reads

    def close(self) -> None:
        """Close the small set of lazily opened process-local mappings."""
        for handle, mapping in self._handles.values():
            mapping.close()
            handle.close()
        self._handles.clear()

    def __enter__(self) -> Self:
        """Return the active mounted-store reader."""
        return self

    def __exit__(self, *_args: object) -> None:
        """Close the mounted-store reader."""
        self.close()

    def __del__(self) -> None:
        """Best-effort cleanup during interpreter shutdown."""
        try:
            self.close()
        except (AttributeError, BufferError):
            return

    def _mapping(self, part: int) -> mmap.mmap:
        opened = self._handles.get(part)
        if opened is not None:
            return opened[1]
        record = self._parts[part]
        path = self._source_roots[record.kaggle_source] / record.binary_name
        if path.stat().st_size != record.binary_bytes:
            raise ValueError(f"Mounted binary size differs for part {part}")
        handle = path.open("rb")
        try:
            header = parse_latent_shard_header(handle.read(LATENT_SHARD_HEADER_SIZE))
            if header.tensor_count != record.row_count:
                raise ValueError(f"Mounted binary row count differs for part {part}")
            mapping = mmap.mmap(handle.fileno(), length=0, access=mmap.ACCESS_READ)
        except BaseException:
            handle.close()
            raise
        self._handles[part] = (handle, mapping)
        return mapping


class WSIBagDataset:
    """Resolve complete frozen WSI bags through one model-specific store."""

    def __init__(
        self,
        *,
        instance_path: Path,
        bag_path: Path,
        store: SupervisedLatentStore,
    ) -> None:
        """Validate and retain one split's compact logical WSI files."""
        self.instances = _load_wsi_instances(instance_path)
        self.bags = _load_wsi_bags(bag_path, self.instances)
        self.store = store

    def __len__(self) -> int:
        """Return the number of WSI statistical examples."""
        return len(self.bags)

    def __getitem__(self, index: int) -> LoadedWSIBag:
        """Load every fixed instance in one WSI bag."""
        bag = self.bags[index]
        stop = bag.instance_start + bag.instance_count
        instances = self.instances[bag.instance_start : stop]
        latents, reads = self.store.read_rows(
            tuple(instance.pointer for instance in instances),
        )
        return LoadedWSIBag(bag, instances, latents, reads)


class TissueDataset:
    """Resolve one frozen tissue CSV through one model-specific store."""

    def __init__(
        self,
        *,
        path: Path,
        split: LearningSplit,
        store: SupervisedLatentStore,
    ) -> None:
        """Validate and retain one split's model-independent tissue rows."""
        self.instances = _load_tissue_instances(path, expected_split=split)
        self.store = store

    def __len__(self) -> int:
        """Return the number of patch-level statistical examples."""
        return len(self.instances)

    def read_batch(self, indices: Sequence[int]) -> LoadedTissueBatch:
        """Load one supplied logical batch without changing its paired order."""
        if not indices or len(set(indices)) != len(indices):
            raise ValueError("A tissue batch must contain unique logical rows")
        if any(index not in range(len(self.instances)) for index in indices):
            raise IndexError("Tissue batch index is outside the frozen CSV")
        try:
            instances = tuple(self.instances[index] for index in indices)
        except IndexError as error:
            raise IndexError("Tissue batch index is outside the frozen CSV") from error
        latents, reads = self.store.read_rows(
            tuple(instance.pointer for instance in instances),
        )
        labels = torch.tensor(
            [instance.tissue_index for instance in instances],
            dtype=torch.long,
        )
        return LoadedTissueBatch(instances, latents, labels, reads)


def _load_catalog(path: Path) -> dict[tuple[int, ModelName], CatalogPart]:
    records: dict[tuple[int, ModelName], CatalogPart] = {}
    paired: dict[int, dict[ModelName, CatalogPart]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CATALOG_HEADER:
            raise ValueError("Unexpected physical-parts catalog header")
        for raw in reader:
            model_value = raw["model_name"]
            if model_value not in {"normal_vae", "so2_vae"}:
                raise ValueError("Unexpected catalog model name")
            model = cast("ModelName", model_value)
            record = CatalogPart(
                part=_csv_int(raw, "part"),
                model_name=model,
                kaggle_source=raw["kaggle_source"],
                binary_name=raw["binary_name"],
                row_count=_csv_int(raw, "row_count"),
                binary_bytes=_csv_int(raw, "binary_bytes"),
                binary_sha256=raw["binary_sha256"],
                sidecar_name=raw["sidecar_name"],
                sidecar_bytes=_csv_int(raw, "sidecar_bytes"),
                sidecar_sha256=raw["sidecar_sha256"],
            )
            key = (record.part, record.model_name)
            if record.part < 1 or record.row_count < 1 or key in records:
                raise ValueError("Invalid or duplicate physical catalog row")
            expected_bytes = (
                LATENT_SHARD_HEADER_SIZE + record.row_count * LATENT_RECORD_BYTES
            )
            if record.binary_bytes != expected_bytes:
                raise ValueError("Catalog binary geometry differs")
            if (
                not _sha256_string(record.binary_sha256)
                or not record.sidecar_name
                or record.sidecar_bytes < 1
                or not _sha256_string(record.sidecar_sha256)
            ):
                raise ValueError("Catalog artifact identity differs")
            records[key] = record
            paired.setdefault(record.part, {})[model] = record
    if not records or any(
        set(models) != {"normal_vae", "so2_vae"}
        or models["normal_vae"].row_count != models["so2_vae"].row_count
        for models in paired.values()
    ):
        raise ValueError("Every catalog part must contain one aligned model pair")
    return records


def _sha256_string(value: str) -> bool:
    return len(value) == SHA256_HEX_LENGTH and all(
        character in "0123456789abcdef" for character in value
    )


def _load_wsi_instances(path: Path) -> tuple[WSIInstance, ...]:
    rows: list[WSIInstance] = []
    previous: tuple[int, int, int] | None = None
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != WSI_INSTANCE_HEADER:
            raise ValueError("Unexpected WSI instance header")
        for row_index, raw in enumerate(reader):
            split = _split(raw["split"])
            row = WSIInstance(
                instance_row=_csv_int(raw, "instance_row"),
                atlas_row_index=_csv_int(raw, "atlas_row_index"),
                wsi_id=_csv_int(raw, "wsi_id"),
                x=_csv_int(raw, "x"),
                y=_csv_int(raw, "y"),
                diagnosis_label=raw["diagnosis_label"],
                diagnosis_index=_csv_int(raw, "diagnosis_index"),
                split=split,
                pointer=LogicalPointer(
                    _csv_int(raw, "part"),
                    _csv_int(raw, "file_index"),
                ),
            )
            order = (row.wsi_id, row.y, row.x)
            if row.instance_row != row_index or (
                previous is not None and order <= previous
            ):
                raise ValueError("WSI instances are not in canonical logical order")
            previous = order
            rows.append(row)
    return tuple(rows)


def _load_wsi_bags(
    path: Path,
    instances: Sequence[WSIInstance],
) -> tuple[WSIBag, ...]:
    bags: list[WSIBag] = []
    seen_wsi_ids: set[int] = set()
    cursor = 0
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != WSI_BAG_HEADER:
            raise ValueError("Unexpected WSI bag header")
        for bag_index, raw in enumerate(reader):
            bag = WSIBag(
                bag_row=_csv_int(raw, "bag_row"),
                wsi_id=_csv_int(raw, "wsi_id"),
                diagnosis_label=raw["diagnosis_label"],
                diagnosis_index=_csv_int(raw, "diagnosis_index"),
                split=_split(raw["split"]),
                instance_start=_csv_int(raw, "instance_start"),
                instance_count=_csv_int(raw, "instance_count"),
            )
            stop = cursor + bag.instance_count
            selected = instances[cursor:stop]
            if (
                bag.bag_row != bag_index
                or bag.wsi_id in seen_wsi_ids
                or bag.instance_start != cursor
                or bag.instance_count < 1
                or len(selected) != bag.instance_count
                or any(
                    (row.wsi_id, row.diagnosis_label, row.diagnosis_index, row.split)
                    != (bag.wsi_id, bag.diagnosis_label, bag.diagnosis_index, bag.split)
                    for row in selected
                )
            ):
                raise ValueError("WSI bag range or metadata differs")
            bags.append(bag)
            seen_wsi_ids.add(bag.wsi_id)
            cursor = stop
    if cursor != len(instances):
        raise ValueError("WSI bags do not consume every instance")
    return tuple(bags)


def _load_tissue_instances(
    path: Path,
    *,
    expected_split: LearningSplit,
) -> tuple[TissueInstance, ...]:
    rows: list[TissueInstance] = []
    previous: tuple[int, int, int] | None = None
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != TISSUE_HEADER:
            raise ValueError("Unexpected tissue dataset header")
        for row_index, raw in enumerate(reader):
            label = raw["tissue_label"]
            if label not in TISSUE_INDICES:
                raise ValueError("Unexpected tissue label")
            split = _split(raw["split"])
            if split != expected_split:
                raise ValueError("Tissue row differs from the requested learning split")
            rank_value = raw["selection_rank"]
            row = TissueInstance(
                dataset_row=_csv_int(raw, "dataset_row"),
                atlas_row_index=_csv_int(raw, "atlas_row_index"),
                wsi_id=_csv_int(raw, "wsi_id"),
                x=_csv_int(raw, "x"),
                y=_csv_int(raw, "y"),
                tissue_label=label,
                tissue_index=TISSUE_INDICES[label],
                split=split,
                selection_rank=(
                    None if not rank_value else _csv_int(raw, "selection_rank")
                ),
                pointer=LogicalPointer(
                    _csv_int(raw, "part"),
                    _csv_int(raw, "file_index"),
                ),
            )
            order = (row.wsi_id, row.y, row.x)
            if row.dataset_row != row_index or (
                previous is not None and order <= previous
            ):
                raise ValueError("Tissue rows are not in canonical logical order")
            previous = order
            rows.append(row)
    return tuple(rows)


def _csv_int(raw: Mapping[str, str], key: str) -> int:
    try:
        value = int(raw[key])
    except (KeyError, ValueError) as error:
        raise ValueError(f"Invalid CSV integer {key}") from error
    if value < 0:
        raise ValueError(f"CSV integer {key} may not be negative")
    return value


def _split(value: str) -> Split:
    if value not in {"train", "validation", "test"}:
        raise ValueError(f"Unexpected split {value!r}")
    return cast("Split", value)


__all__ = [
    "TISSUE_HEADER",
    "CatalogPart",
    "LoadedTissueBatch",
    "LoadedWSIBag",
    "LogicalPointer",
    "PhysicalRead",
    "SupervisedLatentStore",
    "TissueDataset",
    "TissueInstance",
    "WSIBag",
    "WSIBagDataset",
    "WSIInstance",
]
