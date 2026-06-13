# Copyright 2026 HiperMaximus
"""Deterministic fixed-patch selector generation and validation."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from eqvae.data.patch_shards import (
    PATCH_LABELS,
    PATCH_SHARD_HEADER_SIZE,
    PatchRecord,
    PatchShard,
    PatchShardHeader,
    PatchShardSpec,
)
from eqvae.data.roots import PatchSplit, normalize_patch_split

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

type SelectorKind = Literal["fixed_25_validation", "fixed_32_train_overfit"]
type SelectorStatus = Literal["pass", "requires_real_data_generation"]
type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]

FIXED_SELECTOR_SCHEMA_VERSION = "spec0001.fixed_selector.v1"
FIXED_SELECTOR_PLACEHOLDER_STATUS: SelectorStatus = "requires_real_data_generation"
FIXED_SELECTOR_READY_STATUS: SelectorStatus = "pass"
FIXED_25_VALIDATION_KIND: SelectorKind = "fixed_25_validation"
FIXED_32_TRAIN_OVERFIT_KIND: SelectorKind = "fixed_32_train_overfit"
FIXED_25_VALIDATION_SEED = "20260610"
FIXED_32_TRAIN_OVERFIT_SEED = "20260611:tiny-overfit"
FIXED_25_VALIDATION_COUNT = 25
FIXED_25_VALIDATION_PER_LABEL = 5
FIXED_32_TRAIN_OVERFIT_COUNT = 32
DEFAULT_DATASET_SLUG = "maximusshtefan/patches-pre-shuffled-ubc-ocean"


@dataclass(frozen=True)
class FixedSelectorGenerationContext:
    """Optional source context for selector generation."""

    dataset_slug: str = DEFAULT_DATASET_SLUG
    data_root: Path | None = None
    masked_holdout_wsi_ids: frozenset[str] = frozenset()


@dataclass(frozen=True)
class FixedSelectorSource:
    """Source CSV/bin provenance for a selector document."""

    dataset_slug: str
    data_root: Path | None
    source_split: PatchSplit
    csv_path: Path
    csv_sha256: str
    bin_path: Path
    bin_file_size: int
    header_sha256: str
    header: PatchShardHeader
    row_count: int
    patch_count: int
    idx_policy: str
    crc_checked: bool

    def as_json(self) -> JsonObject:
        """Return JSON-ready source provenance.

        Returns:
            JSON object.

        """
        return {
            "dataset_slug": self.dataset_slug,
            "data_root": None if self.data_root is None else str(self.data_root),
            "source_split": self.source_split,
            "csv_path": str(self.csv_path),
            "csv_sha256": self.csv_sha256,
            "bin_path": str(self.bin_path),
            "bin_file_size": self.bin_file_size,
            "header_sha256": self.header_sha256,
            "header": {
                "crc32": self.header.crc32,
                "patch_count": self.header.patch_count,
                "channels": self.header.channels,
                "height": self.header.height,
                "width": self.header.width,
                "version": self.header.version,
                "layout": self.header.layout.decode("ascii"),
            },
            "row_count": self.row_count,
            "patch_count": self.patch_count,
            "idx_policy": self.idx_policy,
            "crc_checked": self.crc_checked,
        }


@dataclass(frozen=True)
class FixedSelectorRecord:
    """One fixed selector row with expected resolved metadata."""

    rank: int
    source_split: PatchSplit
    file_index: int
    row_index: int
    sample_id: str
    wsi_id: str
    label: int
    x: int
    y: int
    selection_key_sha256: str
    patch_sha256: str

    def as_json(self) -> JsonObject:
        """Return JSON-ready selector row.

        Returns:
            JSON object.

        """
        return {
            "rank": self.rank,
            "source_split": self.source_split,
            "file_index": self.file_index,
            "row_index": self.row_index,
            "sample_id": self.sample_id,
            "wsi_id": self.wsi_id,
            "label": self.label,
            "x": self.x,
            "y": self.y,
            "selection_key_sha256": self.selection_key_sha256,
            "patch_sha256": self.patch_sha256,
        }


@dataclass(frozen=True)
class FixedSelectorDocument:
    """A complete fixed selector artifact."""

    selector_kind: SelectorKind
    status: SelectorStatus
    source_split: PatchSplit
    expected_count: int
    selector_seed: str
    source: FixedSelectorSource
    selectors: tuple[FixedSelectorRecord, ...]
    expected_per_label: int | None = None
    masked_holdout_exclusion: str | None = None

    def as_json(self) -> JsonObject:
        """Return a stable JSON payload.

        Returns:
            JSON object.

        """
        payload: JsonObject = {
            "schema_version": FIXED_SELECTOR_SCHEMA_VERSION,
            "status": self.status,
            "selector_kind": self.selector_kind,
            "source_split": self.source_split,
            "expected_count": self.expected_count,
            "selector_seed": self.selector_seed,
            "source": self.source.as_json(),
            "selectors": [selector.as_json() for selector in self.selectors],
        }
        if self.expected_per_label is not None:
            payload["expected_per_label"] = self.expected_per_label
        if self.masked_holdout_exclusion is not None:
            payload["masked_holdout_exclusion"] = self.masked_holdout_exclusion
        return payload


def generate_fixed_selector_document(
    *,
    selector_kind: SelectorKind,
    shard_spec: PatchShardSpec,
    source_split: str,
    context: FixedSelectorGenerationContext | None = None,
) -> FixedSelectorDocument:
    """Generate a fixed selector document from a real or synthetic shard.

    Returns:
        Fixed selector document with `status = "pass"`.

    Raises:
        ValueError: If the selector kind/split is invalid or fixed-32 generation
            lacks masked-holdout IDs.

    """
    split = normalize_patch_split(source_split)
    _validate_kind_split(selector_kind=selector_kind, source_split=split)
    generation_context = context or FixedSelectorGenerationContext()
    if (
        selector_kind == FIXED_32_TRAIN_OVERFIT_KIND
        and not generation_context.masked_holdout_wsi_ids
    ):
        message = "fixed_32_train_overfit generation requires masked holdout WSI IDs"
        raise ValueError(message)
    shard = PatchShard(shard_spec)
    records = shard.records
    _ensure_unique_semantic_identity(records)
    selected = _select_records(
        selector_kind=selector_kind,
        records=records,
        masked_holdout_wsi_ids=generation_context.masked_holdout_wsi_ids,
    )
    seed = selector_seed_for_kind(selector_kind)
    source = build_selector_source(
        shard=shard,
        shard_spec=shard_spec,
        source_split=split,
        dataset_slug=generation_context.dataset_slug,
        data_root=generation_context.data_root,
    )
    selectors = tuple(
        _selector_record(
            rank=rank,
            record=record,
            split=split,
            seed=seed,
            shard_spec=shard_spec,
        )
        for rank, record in enumerate(selected)
    )
    return FixedSelectorDocument(
        selector_kind=selector_kind,
        status=FIXED_SELECTOR_READY_STATUS,
        source_split=split,
        expected_count=expected_count_for_kind(selector_kind),
        expected_per_label=(
            FIXED_25_VALIDATION_PER_LABEL
            if selector_kind == FIXED_25_VALIDATION_KIND
            else None
        ),
        selector_seed=seed,
        source=source,
        selectors=selectors,
        masked_holdout_exclusion=(
            "docs/data/ubc_ocean_masked_holdout_ids.csv"
            if selector_kind == FIXED_32_TRAIN_OVERFIT_KIND
            else None
        ),
    )


def build_selector_source(
    *,
    shard: PatchShard,
    shard_spec: PatchShardSpec,
    source_split: PatchSplit,
    dataset_slug: str,
    data_root: Path | None = None,
) -> FixedSelectorSource:
    """Build top-level selector source provenance.

    Returns:
        Source provenance.

    """
    header_bytes = _read_header_bytes(shard_spec.bin_path)
    return FixedSelectorSource(
        dataset_slug=dataset_slug,
        data_root=data_root,
        source_split=source_split,
        csv_path=shard_spec.csv_path,
        csv_sha256=_sha256_file(shard_spec.csv_path),
        bin_path=shard_spec.bin_path,
        bin_file_size=shard_spec.bin_path.stat().st_size,
        header_sha256=hashlib.sha256(header_bytes).hexdigest(),
        header=shard.header,
        row_count=len(shard.records),
        patch_count=shard.header.patch_count,
        idx_policy=_idx_policy(shard_spec.csv_path),
        crc_checked=shard.crc_validated,
    )


def write_fixed_selector_document(
    *,
    path: Path,
    document: FixedSelectorDocument,
    allow_tracked_config_overwrite: bool = False,
) -> None:
    """Write a fixed selector JSON document.

    Args:
        path: Output JSON path.
        document: Selector document.
        allow_tracked_config_overwrite: Whether tracked spec config placeholders
            may be overwritten.

    Raises:
        PermissionError: If the output is a tracked canonical config path and
            explicit overwrite permission was not supplied.

    """
    if _is_canonical_selector_config(path):
        if not allow_tracked_config_overwrite:
            message = (
                "Refusing to overwrite canonical selector config without "
                "--allow-tracked-config-overwrite"
            )
            raise PermissionError(message)
        if not document.source.crc_checked:
            message = (
                "Refusing to overwrite canonical selector config without a "
                "CRC-validated source; rerun with --validate-crc"
            )
            raise PermissionError(message)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"{json.dumps(document.as_json(), indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )


def load_fixed_selector_document(path: Path) -> FixedSelectorDocument:
    """Load a selector document from JSON.

    Returns:
        Parsed selector document.

    Raises:
        ValueError: If the selector is a placeholder or has an unsupported
            schema.

    """
    payload = _read_json_object(path)
    status = _required_str(payload, "status")
    if status != FIXED_SELECTOR_READY_STATUS:
        message = f"Fixed selector {path} is not ready: status={status!r}"
        raise ValueError(message)
    schema_version = _required_str(payload, "schema_version")
    if schema_version != FIXED_SELECTOR_SCHEMA_VERSION:
        message = (
            f"Unsupported fixed selector schema {schema_version!r}; expected "
            f"{FIXED_SELECTOR_SCHEMA_VERSION!r}"
        )
        raise ValueError(message)
    selector_kind = _selector_kind(_required_str(payload, "selector_kind"))
    source_split = normalize_patch_split(_required_str(payload, "source_split"))
    source = _parse_source(_required_object(payload, "source"))
    selectors = tuple(
        _parse_selector(cast("Mapping[str, object]", row))
        for row in _required_list(payload, "selectors")
    )
    return FixedSelectorDocument(
        selector_kind=selector_kind,
        status=FIXED_SELECTOR_READY_STATUS,
        source_split=source_split,
        expected_count=_required_int(payload, "expected_count"),
        expected_per_label=_optional_int(payload, "expected_per_label"),
        selector_seed=_required_str(payload, "selector_seed"),
        source=source,
        selectors=selectors,
        masked_holdout_exclusion=_optional_str(payload, "masked_holdout_exclusion"),
    )


def validate_fixed_selector_document(
    *,
    document: FixedSelectorDocument,
    shard_spec: PatchShardSpec,
    expected_kind: SelectorKind | None = None,
    masked_holdout_wsi_ids: frozenset[str] = frozenset(),
) -> tuple[FixedSelectorRecord, ...]:
    """Validate a selector document against the current CSV/bin pair.

    Stored selector fields are expected values. This function recomputes them
    from the current shard and fails on drift.

    Returns:
        Validated selector records.

    Raises:
        ValueError: If the selector kind, source provenance, canonical selected
            rows, row count, seed, or per-row fields do not match.

    """
    if expected_kind is not None and document.selector_kind != expected_kind:
        message = (
            f"Expected selector kind {expected_kind}, got {document.selector_kind}"
        )
        raise ValueError(message)
    _validate_kind_split(
        selector_kind=document.selector_kind,
        source_split=document.source_split,
    )
    shard = PatchShard(shard_spec)
    _ensure_unique_semantic_identity(shard.records)
    current_source = build_selector_source(
        shard=shard,
        shard_spec=shard_spec,
        source_split=document.source_split,
        dataset_slug=document.source.dataset_slug,
        data_root=document.source.data_root,
    )
    _validate_source(document.source, current_source)
    _validate_expected_counts(document)
    _validate_selector_policy(
        document=document,
        records=shard.records,
        masked_holdout_wsi_ids=masked_holdout_wsi_ids,
    )
    records_by_file_index = {record.file_index: record for record in shard.records}
    for rank, selector in enumerate(document.selectors):
        _validate_selector_record(
            selector=selector,
            rank=rank,
            document=document,
            records_by_file_index=records_by_file_index,
            shard_spec=shard_spec,
        )
    return document.selectors


def selector_seed_for_kind(selector_kind: SelectorKind) -> str:
    """Return the locked selector seed for one selector kind.

    Returns:
        Selector seed string.

    """
    if selector_kind == FIXED_25_VALIDATION_KIND:
        return FIXED_25_VALIDATION_SEED
    return FIXED_32_TRAIN_OVERFIT_SEED


def expected_count_for_kind(selector_kind: SelectorKind) -> int:
    """Return the expected selector row count for one selector kind.

    Returns:
        Expected count.

    """
    if selector_kind == FIXED_25_VALIDATION_KIND:
        return FIXED_25_VALIDATION_COUNT
    return FIXED_32_TRAIN_OVERFIT_COUNT


def selection_key_sha256(*, seed: str, record: PatchRecord) -> str:
    """Compute the locked deterministic selector key.

    Returns:
        Hex SHA-256 selection key.

    """
    payload = f"{seed}:{record.wsi_id}:{record.label}:{record.x}:{record.y}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def infer_selector_kind(path: Path, explicit_kind: str | None = None) -> SelectorKind:
    """Infer selector kind from an explicit value or output filename.

    Returns:
        Selector kind.

    Raises:
        ValueError: If the explicit value or filename cannot identify a known
            selector kind.

    """
    if explicit_kind is not None:
        return _selector_kind(explicit_kind)
    filename = path.name
    if "fixed_25_validation" in filename:
        return FIXED_25_VALIDATION_KIND
    if "fixed_32_train" in filename:
        return FIXED_32_TRAIN_OVERFIT_KIND
    message = (
        "Could not infer selector kind from output path; pass --kind "
        "fixed_25_validation or fixed_32_train_overfit"
    )
    raise ValueError(message)


def _select_records(
    *,
    selector_kind: SelectorKind,
    records: Sequence[PatchRecord],
    masked_holdout_wsi_ids: frozenset[str],
) -> tuple[PatchRecord, ...]:
    if selector_kind == FIXED_25_VALIDATION_KIND:
        return _select_fixed_25_validation(records)
    return _select_fixed_32_train_overfit(
        records=records,
        masked_holdout_wsi_ids=masked_holdout_wsi_ids,
    )


def _select_fixed_25_validation(
    records: Sequence[PatchRecord],
) -> tuple[PatchRecord, ...]:
    seed = selector_seed_for_kind(FIXED_25_VALIDATION_KIND)
    selected: list[PatchRecord] = []
    for label in sorted(PATCH_LABELS):
        label_records = [record for record in records if record.label == label]
        if len(label_records) < FIXED_25_VALIDATION_PER_LABEL:
            message = (
                f"Label {label} has {len(label_records)} validation records; "
                f"need {FIXED_25_VALIDATION_PER_LABEL}"
            )
            raise ValueError(message)
        selected.extend(
            sorted(
                label_records,
                key=lambda record: (
                    selection_key_sha256(seed=seed, record=record),
                    record.wsi_id,
                    record.x,
                    record.y,
                ),
            )[:FIXED_25_VALIDATION_PER_LABEL],
        )
    return tuple(selected)


def _select_fixed_32_train_overfit(
    *,
    records: Sequence[PatchRecord],
    masked_holdout_wsi_ids: frozenset[str],
) -> tuple[PatchRecord, ...]:
    seed = selector_seed_for_kind(FIXED_32_TRAIN_OVERFIT_KIND)
    candidates = [
        record for record in records if record.wsi_id not in masked_holdout_wsi_ids
    ]
    if len(candidates) < FIXED_32_TRAIN_OVERFIT_COUNT:
        message = (
            f"Only {len(candidates)} train records remain after holdout exclusion; "
            f"need {FIXED_32_TRAIN_OVERFIT_COUNT}"
        )
        raise ValueError(message)
    return tuple(
        sorted(
            candidates,
            key=lambda record: (
                selection_key_sha256(seed=seed, record=record),
                record.wsi_id,
                record.label,
                record.x,
                record.y,
            ),
        )[:FIXED_32_TRAIN_OVERFIT_COUNT],
    )


def _selector_record(
    *,
    rank: int,
    record: PatchRecord,
    split: PatchSplit,
    seed: str,
    shard_spec: PatchShardSpec,
) -> FixedSelectorRecord:
    return FixedSelectorRecord(
        rank=rank,
        source_split=split,
        file_index=record.file_index,
        row_index=record.row_index,
        sample_id=record.sample_id(split),
        wsi_id=record.wsi_id,
        label=record.label,
        x=record.x,
        y=record.y,
        selection_key_sha256=selection_key_sha256(seed=seed, record=record),
        patch_sha256=_patch_sha256(
            bin_path=shard_spec.bin_path,
            file_index=record.file_index,
            header_size=shard_spec.header_size,
            patch_bytes=shard_spec.channels
            * shard_spec.image_size
            * shard_spec.image_size,
        ),
    )


def _validate_selector_record(
    *,
    selector: FixedSelectorRecord,
    rank: int,
    document: FixedSelectorDocument,
    records_by_file_index: Mapping[int, PatchRecord],
    shard_spec: PatchShardSpec,
) -> None:
    if selector.rank != rank:
        message = f"Selector rank mismatch: expected {rank}, got {selector.rank}"
        raise ValueError(message)
    if selector.source_split != document.source_split:
        message = (
            f"Selector split mismatch for rank {rank}: "
            f"{selector.source_split} != {document.source_split}"
        )
        raise ValueError(message)
    record = records_by_file_index.get(selector.file_index)
    if record is None:
        message = f"Selector file_index {selector.file_index} is absent from CSV"
        raise ValueError(message)
    expected = _selector_record(
        rank=rank,
        record=record,
        split=document.source_split,
        seed=document.selector_seed,
        shard_spec=shard_spec,
    )
    if selector != expected:
        message = f"Selector drift at rank {rank}: stored row does not match CSV/bin"
        raise ValueError(message)


def _validate_source(
    stored: FixedSelectorSource,
    current: FixedSelectorSource,
) -> None:
    checks = {
        "source_split": (stored.source_split, current.source_split),
        "csv_sha256": (stored.csv_sha256, current.csv_sha256),
        "bin_file_size": (stored.bin_file_size, current.bin_file_size),
        "header_sha256": (stored.header_sha256, current.header_sha256),
        "row_count": (stored.row_count, current.row_count),
        "patch_count": (stored.patch_count, current.patch_count),
        "idx_policy": (stored.idx_policy, current.idx_policy),
        "crc_checked": (stored.crc_checked, current.crc_checked),
    }
    for name, (stored_value, current_value) in checks.items():
        if stored_value != current_value:
            message = (
                f"Selector source {name} mismatch: "
                f"stored {stored_value!r}, current {current_value!r}"
            )
            raise ValueError(message)
    if stored.header != current.header:
        message = "Selector source header fields do not match current shard"
        raise ValueError(message)


def _validate_expected_counts(document: FixedSelectorDocument) -> None:
    if len(document.selectors) != document.expected_count:
        message = (
            f"Selector count mismatch: expected {document.expected_count}, "
            f"got {len(document.selectors)}"
        )
        raise ValueError(message)
    expected_count = expected_count_for_kind(document.selector_kind)
    if document.expected_count != expected_count:
        message = (
            f"Selector expected_count {document.expected_count} does not match "
            f"{document.selector_kind} policy {expected_count}"
        )
        raise ValueError(message)
    expected_seed = selector_seed_for_kind(document.selector_kind)
    if document.selector_seed != expected_seed:
        message = (
            f"Selector seed {document.selector_seed!r} does not match "
            f"{document.selector_kind} policy {expected_seed!r}"
        )
        raise ValueError(message)
    if document.selector_kind == FIXED_25_VALIDATION_KIND:
        label_counts = Counter(selector.label for selector in document.selectors)
        expected_counts = dict.fromkeys(PATCH_LABELS, FIXED_25_VALIDATION_PER_LABEL)
        if dict(sorted(label_counts.items())) != expected_counts:
            message = f"Fixed-25 selector label counts mismatch: {label_counts}"
            raise ValueError(message)


def _validate_selector_policy(
    *,
    document: FixedSelectorDocument,
    records: Sequence[PatchRecord],
    masked_holdout_wsi_ids: frozenset[str],
) -> None:
    if (
        document.selector_kind == FIXED_32_TRAIN_OVERFIT_KIND
        and not masked_holdout_wsi_ids
    ):
        message = "fixed_32_train_overfit validation requires masked holdout WSI IDs"
        raise ValueError(message)
    expected_file_indices = tuple(
        record.file_index
        for record in _select_records(
            selector_kind=document.selector_kind,
            records=records,
            masked_holdout_wsi_ids=masked_holdout_wsi_ids,
        )
    )
    observed_file_indices = tuple(
        selector.file_index for selector in document.selectors
    )
    if observed_file_indices != expected_file_indices:
        message = "Selector policy mismatch: stored rows are not the canonical set"
        raise ValueError(message)


def _ensure_unique_semantic_identity(records: Sequence[PatchRecord]) -> None:
    identities = [
        (record.wsi_id, record.label, record.x, record.y) for record in records
    ]
    duplicates = [
        identity for identity, count in Counter(identities).items() if count > 1
    ]
    if duplicates:
        message = f"Patch CSV contains duplicate semantic identities: {duplicates[:3]}"
        raise ValueError(message)


def _validate_kind_split(
    *,
    selector_kind: SelectorKind,
    source_split: PatchSplit,
) -> None:
    if selector_kind == FIXED_25_VALIDATION_KIND and source_split != "validation":
        message = "fixed_25_validation selectors require validation split"
        raise ValueError(message)
    if selector_kind == FIXED_32_TRAIN_OVERFIT_KIND and source_split != "train":
        message = "fixed_32_train_overfit selectors require train split"
        raise ValueError(message)


def _idx_policy(csv_path: Path) -> str:
    with csv_path.open(encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or ()
    return "idx_column" if "idx" in fieldnames else "row_order"


def _read_header_bytes(bin_path: Path) -> bytes:
    with bin_path.open("rb") as binary_file:
        header_bytes = binary_file.read(PATCH_SHARD_HEADER_SIZE)
    if len(header_bytes) != PATCH_SHARD_HEADER_SIZE:
        message = (
            f"Could not read {PATCH_SHARD_HEADER_SIZE}-byte header from {bin_path}"
        )
        raise EOFError(message)
    return header_bytes


def _patch_sha256(
    *,
    bin_path: Path,
    file_index: int,
    header_size: int,
    patch_bytes: int,
) -> str:
    with bin_path.open("rb") as binary_file:
        binary_file.seek(header_size + file_index * patch_bytes)
        payload = binary_file.read(patch_bytes)
    if len(payload) != patch_bytes:
        message = f"Could not read complete patch payload for index {file_index}"
        raise EOFError(message)
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _selector_kind(value: str) -> SelectorKind:
    if value in {FIXED_25_VALIDATION_KIND, FIXED_32_TRAIN_OVERFIT_KIND}:
        return cast("SelectorKind", value)
    message = f"Unknown selector kind {value!r}"
    raise ValueError(message)


def _parse_source(payload: Mapping[str, object]) -> FixedSelectorSource:
    header_payload = _required_object(payload, "header")
    header = PatchShardHeader(
        crc32=_required_int(header_payload, "crc32"),
        patch_count=_required_int(header_payload, "patch_count"),
        channels=_required_int(header_payload, "channels"),
        height=_required_int(header_payload, "height"),
        width=_required_int(header_payload, "width"),
        version=_required_int(header_payload, "version"),
        layout=_required_str(header_payload, "layout").encode("ascii"),
    )
    return FixedSelectorSource(
        dataset_slug=_required_str(payload, "dataset_slug"),
        data_root=_optional_path(payload, "data_root"),
        source_split=normalize_patch_split(_required_str(payload, "source_split")),
        csv_path=Path(_required_str(payload, "csv_path")),
        csv_sha256=_required_str(payload, "csv_sha256"),
        bin_path=Path(_required_str(payload, "bin_path")),
        bin_file_size=_required_int(payload, "bin_file_size"),
        header_sha256=_required_str(payload, "header_sha256"),
        header=header,
        row_count=_required_int(payload, "row_count"),
        patch_count=_required_int(payload, "patch_count"),
        idx_policy=_required_str(payload, "idx_policy"),
        crc_checked=_required_bool(payload, "crc_checked"),
    )


def _parse_selector(payload: Mapping[str, object]) -> FixedSelectorRecord:
    return FixedSelectorRecord(
        rank=_required_int(payload, "rank"),
        source_split=normalize_patch_split(_required_str(payload, "source_split")),
        file_index=_required_int(payload, "file_index"),
        row_index=_required_int(payload, "row_index"),
        sample_id=_required_str(payload, "sample_id"),
        wsi_id=_required_str(payload, "wsi_id"),
        label=_required_int(payload, "label"),
        x=_required_int(payload, "x"),
        y=_required_int(payload, "y"),
        selection_key_sha256=_required_str(payload, "selection_key_sha256"),
        patch_sha256=_required_str(payload, "patch_sha256"),
    )


def _read_json_object(path: Path) -> Mapping[str, object]:
    payload = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(payload, dict):
        message = f"Expected JSON object in {path}"
        raise TypeError(message)
    return cast("Mapping[str, object]", payload)


def _required_object(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, dict):
        message = f"Expected object field {key!r}"
        raise TypeError(message)
    return cast("Mapping[str, object]", value)


def _required_list(payload: Mapping[str, object], key: str) -> Sequence[object]:
    value = payload.get(key)
    if not isinstance(value, list):
        message = f"Expected list field {key!r}"
        raise TypeError(message)
    return tuple(cast("list[object]", value))


def _required_str(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        message = f"Expected string field {key!r}"
        raise TypeError(message)
    return value


def _optional_str(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        message = f"Expected string field {key!r}"
        raise TypeError(message)
    return value


def _optional_path(payload: Mapping[str, object], key: str) -> Path | None:
    value = _optional_str(payload, key)
    if value is None:
        return None
    return Path(value)


def _required_int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        message = f"Expected integer field {key!r}"
        raise TypeError(message)
    return value


def _optional_int(payload: Mapping[str, object], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        message = f"Expected integer field {key!r}"
        raise TypeError(message)
    return value


def _required_bool(payload: Mapping[str, object], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        message = f"Expected boolean field {key!r}"
        raise TypeError(message)
    return value


def _is_canonical_selector_config(path: Path) -> bool:
    normalized = path.as_posix()
    return normalized.endswith(
        (
            "configs/spec0001/fixed_25_validation_patches.json",
            "configs/spec0001/fixed_32_train_overfit_patches.json",
        ),
    )


__all__ = [
    "DEFAULT_DATASET_SLUG",
    "FIXED_25_VALIDATION_COUNT",
    "FIXED_25_VALIDATION_KIND",
    "FIXED_25_VALIDATION_PER_LABEL",
    "FIXED_25_VALIDATION_SEED",
    "FIXED_32_TRAIN_OVERFIT_COUNT",
    "FIXED_32_TRAIN_OVERFIT_KIND",
    "FIXED_32_TRAIN_OVERFIT_SEED",
    "FIXED_SELECTOR_PLACEHOLDER_STATUS",
    "FIXED_SELECTOR_READY_STATUS",
    "FIXED_SELECTOR_SCHEMA_VERSION",
    "FixedSelectorDocument",
    "FixedSelectorGenerationContext",
    "FixedSelectorRecord",
    "FixedSelectorSource",
    "SelectorKind",
    "SelectorStatus",
    "expected_count_for_kind",
    "generate_fixed_selector_document",
    "infer_selector_kind",
    "load_fixed_selector_document",
    "selection_key_sha256",
    "selector_seed_for_kind",
    "validate_fixed_selector_document",
    "write_fixed_selector_document",
]
