# Copyright 2026 HiperMaximus
"""Validate the frozen masked-WSI split and materialize two task manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, cast

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

type SplitName = Literal["train", "validation", "test"]
type TissueName = Literal["tumor", "stroma", "necrosis"]
type RawRow = dict[str, str | None]

ATLAS_SHA256: Final = "9258a98f9512e1a1e04e09ea8dbca28463009fd3276345daed5b2d8119692557"
COHORT_SHA256: Final = (
    "9fa03421694f77d3964097da2388beca45b31ce3e50da0b170c13781b86dea81"
)
SPLIT_SHA256: Final = "216f69f64ed7a3e5636173d6cfe83297632113bc68310e4f6285d7ffc22cd43c"
EXPECTED_CANCER_MANIFEST_SHA256: Final = (
    "710c3f8166f577f5ae60bec94a544dabed9619342a46b82374a2e739162d648c"
)
EXPECTED_TISSUE_MANIFEST_SHA256: Final = (
    "7e2f61c492167129911e73515c4d82ef9291fd89b920dc191be922a44f30ac92"
)
ATLAS_PATH: Final = Path(
    "runs/kaggle/ubc_ocean_test_atlas_v3/dataset/ubc_ocean_test_atlas.csv",
)
COHORT_PATH: Final = Path("docs/data/ubc_ocean_masked_holdout_ids.csv")
SPLIT_PATH: Final = Path("docs/data/ubc_ocean_eval_wsi_split.csv")
AUDIT_PATH: Final = Path("docs/data/ubc_ocean_eval_split_audit.json")
OUTPUT_DIR: Final = Path("runs/local/ubc_ocean_eval_manifests")
GENERATOR_REPO_PATH: Final = Path(
    "src/eqvae/cli/generate_ubc_eval_manifests.py",
)

SPLITS: Final[tuple[SplitName, ...]] = ("train", "validation", "test")
TISSUES: Final[tuple[TissueName, ...]] = ("tumor", "stroma", "necrosis")
LABEL_INDEX: Final = {"CC": 0, "EC": 1, "HGSC": 2, "LGSC": 3, "MC": 4}
INDEX_LABEL: Final = {value: key for key, value in LABEL_INDEX.items()}
OTSU_SOURCES: Final = frozenset({"otsu", "mask+otsu"})
ATLAS_SOURCES: Final = frozenset({"otsu", "mask", "mask+otsu"})
MIN_ANNOTATED_FRACTION: Final = 0.10
MIN_TISSUE_PURITY: Final = 0.95
MIN_HOLDOUT_NECROSIS_PATCHES_PER_WSI: Final = 100
MIN_HOLDOUT_STROMA_WSI_COUNT: Final = 20
MIN_TRAIN_NECROSIS_PATCHES: Final = 5_000
EXPECTED_COHORT_WSI_COUNT: Final = 152
EXPECTED_ATLAS_ROWS: Final = 1_822_340
EXPECTED_CANCER_ROWS: Final = 1_750_221
EXPECTED_TISSUE_ROWS: Final = 666_807
EXPECTED_QUOTAS: Final = {
    "train": {"CC": 23, "EC": 24, "HGSC": 40, "LGSC": 11, "MC": 8},
    "validation": {"CC": 5, "EC": 6, "HGSC": 8, "LGSC": 2, "MC": 2},
    "test": {"CC": 5, "EC": 6, "HGSC": 8, "LGSC": 2, "MC": 2},
}
EXPECTED_UPDATED_COUNTS: Final = {"train": 12, "validation": 3, "test": 3}
EXPECTED_NECROSIS_WSI_COUNTS: Final = {
    "train": 17,
    "validation": 5,
    "test": 5,
}
EXPECTED_HEADERS: Final = (
    "wsi_id",
    "label",
    "x",
    "y",
    "selection_source",
    "mask_status",
    "mask_label",
    "annotated_fraction",
    "tumor_fraction",
    "stroma_fraction",
    "necrosis_fraction",
)
CANCER_HEADER: Final = (
    "atlas_row_index",
    "wsi_id",
    "diagnosis_label",
    "diagnosis_index",
    "x",
    "y",
    "split",
)
TISSUE_HEADER: Final = (
    "atlas_row_index",
    "wsi_id",
    "diagnosis_label",
    "diagnosis_index",
    "x",
    "y",
    "split",
    "tissue_label",
    "annotated_fraction",
    "dominant_fraction",
    "purity",
    "tumor_fraction",
    "stroma_fraction",
    "necrosis_fraction",
)


@dataclass(frozen=True)
class CohortEntry:
    """Canonical diagnosis and technical-update metadata for one WSI."""

    wsi_id: int
    diagnosis_label: str
    diagnosis_index: int
    is_updated_image_id: bool


@dataclass(frozen=True)
class SplitEntry:
    """Frozen split membership and duplicated metadata for one WSI."""

    cohort: CohortEntry
    split: SplitName


@dataclass(frozen=True)
class AtlasRow:
    """Validated atlas row with a derived optional pure-tissue label."""

    row_index: int
    wsi_id: int
    diagnosis_index: int
    x: int
    y: int
    selection_source: str
    annotated_text: str
    fractions_text: tuple[str, str, str]
    annotated_fraction: float
    fractions: tuple[float, float, float]
    tissue_label: TissueName | None
    dominant_fraction: float
    purity: float


@dataclass
class WsiStats:
    """Patch availability aggregated before the frozen split is accepted."""

    cancer_count: int = 0
    tissue_counts: Counter[TissueName] = field(default_factory=Counter)


@dataclass(frozen=True)
class AtlasSummary:
    """Validated cohort-wide atlas totals and per-WSI task availability."""

    atlas_row_count: int
    cancer_count: int
    tissue_count: int
    stats: dict[int, WsiStats]


def sha256_file(path: Path) -> str:
    """Hash a file in bounded chunks so the 116 MiB atlas is never copied.

    Returns:
        Lowercase SHA-256 hexadecimal digest.

    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_pinned_hash(path: Path, expected_sha256: str, artifact_name: str) -> str:
    """Reject structurally valid bytes that would change sealed membership.

    Returns:
        The verified lowercase SHA-256 hexadecimal digest.

    Raises:
        ValueError: If the file bytes do not match the sealed artifact.

    """
    observed = sha256_file(path)
    if observed != expected_sha256:
        message = f"{artifact_name} SHA-256 mismatch: {observed}"
        raise ValueError(message)
    return observed


def load_cohort(path: Path) -> dict[int, CohortEntry]:
    """Load the exact 152-WSI cohort and reject ambiguous metadata.

    Returns:
        Cohort entries keyed by WSI ID in canonical order.

    Raises:
        ValueError: If the cohort schema, values, count, or order is invalid.

    """
    expected = ("image_id", "label", "is_updated_image_id")
    rows: dict[int, CohortEntry] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != expected:
            message = f"Unexpected cohort header: {reader.fieldnames!r}"
            raise ValueError(message)
        for raw in reader:
            row = cast("RawRow", raw)
            wsi_id = _int_value(row, "image_id")
            label = _required(row, "label")
            if label not in LABEL_INDEX:
                message = f"Unknown diagnosis label {label!r} for WSI {wsi_id}"
                raise ValueError(message)
            updated_text = _required(row, "is_updated_image_id")
            if updated_text not in {"true", "false"}:
                message = (
                    f"Invalid updated-image flag {updated_text!r} for WSI {wsi_id}"
                )
                raise ValueError(message)
            if wsi_id in rows:
                message = f"Duplicate cohort WSI {wsi_id}"
                raise ValueError(message)
            rows[wsi_id] = CohortEntry(
                wsi_id=wsi_id,
                diagnosis_label=label,
                diagnosis_index=LABEL_INDEX[label],
                is_updated_image_id=updated_text == "true",
            )
    if len(rows) != EXPECTED_COHORT_WSI_COUNT:
        message = (
            f"Expected {EXPECTED_COHORT_WSI_COUNT} cohort WSIs, observed {len(rows)}"
        )
        raise ValueError(message)
    if list(rows) != sorted(rows):
        message = "Cohort rows must be ordered by ascending image_id"
        raise ValueError(message)
    return rows


def load_split(  # noqa: C901
    path: Path,
    cohort: Mapping[int, CohortEntry],
) -> dict[int, SplitEntry]:
    """Load the frozen assignment and prove that it exactly covers the cohort.

    Returns:
        Frozen split entries keyed by WSI ID.

    Raises:
        ValueError: If the split does not exactly and consistently cover the cohort.

    """
    expected = (
        "wsi_id",
        "diagnosis_label",
        "diagnosis_index",
        "is_updated_image_id",
        "split",
    )
    rows: dict[int, SplitEntry] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != expected:
            message = f"Unexpected split header: {reader.fieldnames!r}"
            raise ValueError(message)
        for raw in reader:
            row = cast("RawRow", raw)
            wsi_id = _int_value(row, "wsi_id")
            if wsi_id in rows:
                message = f"Duplicate split WSI {wsi_id}"
                raise ValueError(message)
            cohort_entry = cohort.get(wsi_id)
            if cohort_entry is None:
                message = f"Split contains non-cohort WSI {wsi_id}"
                raise ValueError(message)
            split_text = _required(row, "split")
            if split_text not in SPLITS:
                message = f"Invalid split {split_text!r} for WSI {wsi_id}"
                raise ValueError(message)
            if _required(row, "diagnosis_label") != cohort_entry.diagnosis_label:
                message = f"Split diagnosis label disagrees for WSI {wsi_id}"
                raise ValueError(message)
            if _int_value(row, "diagnosis_index") != cohort_entry.diagnosis_index:
                message = f"Split diagnosis index disagrees for WSI {wsi_id}"
                raise ValueError(message)
            expected_updated = "true" if cohort_entry.is_updated_image_id else "false"
            if _required(row, "is_updated_image_id") != expected_updated:
                message = f"Split updated-image flag disagrees for WSI {wsi_id}"
                raise ValueError(message)
            rows[wsi_id] = SplitEntry(
                cohort=cohort_entry,
                split=split_text,
            )
    if set(rows) != set(cohort):
        missing = sorted(set(cohort) - set(rows))
        extra = sorted(set(rows) - set(cohort))
        message = f"Split/cohort mismatch: missing={missing}, extra={extra}"
        raise ValueError(message)
    if list(rows) != sorted(rows):
        message = "Split rows must be ordered by ascending wsi_id"
        raise ValueError(message)
    return rows


def iter_atlas_rows(  # noqa: PLR0914
    path: Path,
    cohort: Mapping[int, CohortEntry],
) -> Iterator[AtlasRow]:
    """Yield validated atlas rows while preserving the authoritative order.

    Yields:
        Parsed rows in exact atlas order.

    Raises:
        ValueError: If schema, cohort, order, labels, or fractions are invalid.

    """
    previous_key: tuple[int, int, int] | None = None
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != EXPECTED_HEADERS:
            message = f"Unexpected atlas header: {reader.fieldnames!r}"
            raise ValueError(message)
        for row_index, raw in enumerate(reader):
            row = cast("RawRow", raw)
            wsi_id = _int_value(row, "wsi_id")
            x = _int_value(row, "x")
            y = _int_value(row, "y")
            if x < 0 or y < 0:
                message = f"Negative atlas coordinate at row {row_index}"
                raise ValueError(message)
            key = (wsi_id, y, x)
            if previous_key is not None and key <= previous_key:
                message = f"Atlas order/uniqueness failure at row {row_index}: {key}"
                raise ValueError(message)
            previous_key = key
            cohort_entry = cohort.get(wsi_id)
            if cohort_entry is None:
                message = f"Atlas row {row_index} contains non-cohort WSI {wsi_id}"
                raise ValueError(message)
            diagnosis_index = _int_value(row, "label")
            if diagnosis_index != cohort_entry.diagnosis_index:
                message = f"Atlas diagnosis disagrees for WSI {wsi_id}"
                raise ValueError(message)
            source = _required(row, "selection_source")
            if source not in ATLAS_SOURCES:
                message = f"Invalid selection source {source!r} at row {row_index}"
                raise ValueError(message)
            annotated_text = _required(row, "annotated_fraction")
            fractions_text = (
                _required(row, "tumor_fraction"),
                _required(row, "stroma_fraction"),
                _required(row, "necrosis_fraction"),
            )
            annotated = _fraction(annotated_text, "annotated_fraction", row_index)
            fractions = cast(
                "tuple[float, float, float]",
                tuple(
                    _fraction(text, f"{tissue}_fraction", row_index)
                    for tissue, text in zip(TISSUES, fractions_text, strict=True)
                ),
            )
            if any(value > annotated + 1e-12 for value in fractions):
                message = f"Class fraction exceeds annotation at row {row_index}"
                raise ValueError(message)
            if sum(fractions) + 1e-12 < annotated:
                message = f"Class fractions miss annotated coverage at row {row_index}"
                raise ValueError(message)
            tissue_label, dominant, purity = _pure_tissue(annotated, fractions)
            yield AtlasRow(
                row_index=row_index,
                wsi_id=wsi_id,
                diagnosis_index=diagnosis_index,
                x=x,
                y=y,
                selection_source=source,
                annotated_text=annotated_text,
                fractions_text=fractions_text,
                annotated_fraction=annotated,
                fractions=fractions,
                tissue_label=tissue_label,
                dominant_fraction=dominant,
                purity=purity,
            )


def summarize_atlas(
    path: Path,
    cohort: Mapping[int, CohortEntry],
) -> AtlasSummary:
    """Aggregate task availability without retaining 1.8 million atlas rows.

    Returns:
        Cohort-wide and per-WSI task availability.

    Raises:
        ValueError: If any cohort WSI lacks Otsu-qualified patches.

    """
    stats = {wsi_id: WsiStats() for wsi_id in cohort}
    atlas_count = 0
    cancer_count = 0
    tissue_count = 0
    for row in iter_atlas_rows(path, cohort):
        atlas_count += 1
        wsi_stats = stats[row.wsi_id]
        if row.selection_source in OTSU_SOURCES:
            cancer_count += 1
            wsi_stats.cancer_count += 1
        if row.tissue_label is not None:
            tissue_count += 1
            wsi_stats.tissue_counts[row.tissue_label] += 1
    missing = [wsi_id for wsi_id, value in stats.items() if value.cancer_count == 0]
    if missing:
        message = f"Otsu task has no patches for WSIs {missing}"
        raise ValueError(message)
    return AtlasSummary(
        atlas_row_count=atlas_count,
        cancer_count=cancer_count,
        tissue_count=tissue_count,
        stats=stats,
    )


def validate_split(  # noqa: C901
    split_entries: Mapping[int, SplitEntry],
    summary: AtlasSummary,
    *,
    enforce_real_counts: bool,
) -> dict[str, object]:
    """Reject a split that looks large but lacks independent rare-class WSIs.

    Returns:
        Per-split audit payload after every constraint passes.

    Raises:
        ValueError: If any frozen split or real-atlas acceptance constraint fails.

    """
    audit: dict[str, object] = {}
    for split_name in SPLITS:
        ids = [
            wsi_id
            for wsi_id, entry in split_entries.items()
            if entry.split == split_name
        ]
        diagnosis_counts = Counter(
            split_entries[wsi_id].cohort.diagnosis_label for wsi_id in ids
        )
        if dict(diagnosis_counts) != EXPECTED_QUOTAS[split_name]:
            message = (
                f"{split_name} diagnosis quotas disagree: "
                f"{dict(diagnosis_counts)} != {EXPECTED_QUOTAS[split_name]}"
            )
            raise ValueError(message)
        updated_count = sum(
            split_entries[wsi_id].cohort.is_updated_image_id for wsi_id in ids
        )
        if updated_count != EXPECTED_UPDATED_COUNTS[split_name]:
            message = f"{split_name} updated-image count is {updated_count}"
            raise ValueError(message)
        tissue_patch_counts: dict[TissueName, int] = {
            tissue: sum(summary.stats[wsi_id].tissue_counts[tissue] for wsi_id in ids)
            for tissue in TISSUES
        }
        tissue_positive_counts: dict[TissueName, int] = {
            tissue: sum(
                summary.stats[wsi_id].tissue_counts[tissue] > 0 for wsi_id in ids
            )
            for tissue in TISSUES
        }
        if tissue_positive_counts["tumor"] != len(ids):
            message = f"{split_name} contains a WSI without pure tumor patches"
            raise ValueError(message)
        if (
            split_name != "train"
            and tissue_positive_counts["stroma"] < MIN_HOLDOUT_STROMA_WSI_COUNT
        ):
            message = (
                f"{split_name} has fewer than "
                f"{MIN_HOLDOUT_STROMA_WSI_COUNT} stroma-positive WSIs"
            )
            raise ValueError(message)
        if (
            tissue_positive_counts["necrosis"]
            != EXPECTED_NECROSIS_WSI_COUNTS[split_name]
        ):
            message = f"{split_name} necrosis-positive WSI count disagrees"
            raise ValueError(message)
        if split_name == "train":
            if tissue_patch_counts["necrosis"] < MIN_TRAIN_NECROSIS_PATCHES:
                message = (
                    f"Train has fewer than {MIN_TRAIN_NECROSIS_PATCHES} "
                    "necrosis patches"
                )
                raise ValueError(message)
        else:
            _validate_holdout_tissue(split_name, ids, split_entries, summary)
        audit[split_name] = _split_audit(
            ids,
            split_entries,
            summary,
            diagnosis_counts,
            updated_count,
            tissue_patch_counts,
            tissue_positive_counts,
        )
    _validate_mc_necrosis(split_entries, summary)
    if enforce_real_counts:
        observed = (
            summary.atlas_row_count,
            summary.cancer_count,
            summary.tissue_count,
        )
        expected = (EXPECTED_ATLAS_ROWS, EXPECTED_CANCER_ROWS, EXPECTED_TISSUE_ROWS)
        if observed != expected:
            message = f"Real atlas row counts disagree: {observed} != {expected}"
            raise ValueError(message)
    return audit


def materialize_manifests(  # noqa: PLR0913
    *,
    atlas_path: Path,
    cohort_path: Path,
    split_path: Path,
    output_dir: Path,
    audit_path: Path,
    expected_atlas_sha256: str | None = ATLAS_SHA256,
    enforce_real_counts: bool = True,
) -> dict[str, object]:
    """Validate frozen inputs, write both manifests, and publish the audit last.

    Returns:
        The exact audit payload written as the completion marker.

    Raises:
        ValueError: If inputs, split constraints, or output row counts disagree.

    """
    atlas_hash = sha256_file(atlas_path)
    if expected_atlas_sha256 is not None:
        validate_pinned_hash(atlas_path, expected_atlas_sha256, "Atlas")
    validate_pinned_hash(cohort_path, COHORT_SHA256, "Cohort")
    validate_pinned_hash(split_path, SPLIT_SHA256, "Canonical split")
    cohort = load_cohort(cohort_path)
    split_entries = load_split(split_path, cohort)
    summary = summarize_atlas(atlas_path, cohort)
    split_audit = validate_split(
        split_entries,
        summary,
        enforce_real_counts=enforce_real_counts,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    cancer_path = output_dir / "cancer_ae_patch_manifest.csv"
    tissue_path = output_dir / "tissue_patch_manifest.csv"
    cancer_tmp = cancer_path.with_suffix(".csv.tmp")
    tissue_tmp = tissue_path.with_suffix(".csv.tmp")
    audit_tmp = audit_path.with_suffix(".json.tmp")
    audit_path.unlink(missing_ok=True)
    for temporary in (cancer_tmp, tissue_tmp, audit_tmp):
        temporary.unlink(missing_ok=True)
    try:
        written_cancer, written_tissue = write_manifests(
            atlas_path=atlas_path,
            cohort=cohort,
            split_entries=split_entries,
            cancer_path=cancer_tmp,
            tissue_path=tissue_tmp,
        )
        if written_cancer != summary.cancer_count:
            message = f"Cancer manifest row count changed: {written_cancer}"
            raise ValueError(message)
        if written_tissue != summary.tissue_count:
            message = f"Tissue manifest row count changed: {written_tissue}"
            raise ValueError(message)
        cancer_tmp.replace(cancer_path)
        tissue_tmp.replace(tissue_path)
        audit = _audit_payload(
            atlas_path=atlas_path,
            atlas_hash=atlas_hash,
            cohort_path=cohort_path,
            split_path=split_path,
            cancer_path=cancer_path,
            tissue_path=tissue_path,
            summary=summary,
            split_audit=split_audit,
        )
        _write_json(audit_tmp, audit)
        audit_tmp.replace(audit_path)
        return audit
    finally:
        for temporary in (cancer_tmp, tissue_tmp, audit_tmp):
            temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Spec 0018 local manifest materializer.

    Returns:
        Zero after validated artifacts and the audit completion marker are written.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", type=Path, default=ATLAS_PATH)
    parser.add_argument("--cohort", type=Path, default=COHORT_PATH)
    parser.add_argument("--split", type=Path, default=SPLIT_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--audit-output", type=Path, default=AUDIT_PATH)
    arguments = parser.parse_args(argv)
    audit = materialize_manifests(
        atlas_path=cast("Path", arguments.atlas),
        cohort_path=cast("Path", arguments.cohort),
        split_path=cast("Path", arguments.split),
        output_dir=cast("Path", arguments.output_dir),
        audit_path=cast("Path", arguments.audit_output),
    )
    sys.stdout.write(f"{json.dumps(audit['row_counts'], sort_keys=True)}\n")
    return 0


def _pure_tissue(
    annotated: float,
    fractions: tuple[float, float, float],
) -> tuple[TissueName | None, float, float]:
    if annotated < MIN_ANNOTATED_FRACTION:
        return None, 0.0, 0.0
    dominant = max(fractions)
    winners = [index for index, value in enumerate(fractions) if value == dominant]
    purity = dominant / annotated
    if len(winners) != 1 or purity < MIN_TISSUE_PURITY:
        return None, dominant, purity
    return TISSUES[winners[0]], dominant, purity


def _validate_holdout_tissue(
    split_name: SplitName,
    ids: Sequence[int],
    split_entries: Mapping[int, SplitEntry],
    summary: AtlasSummary,
) -> None:
    patch_counts = {
        tissue: sum(summary.stats[wsi_id].tissue_counts[tissue] for wsi_id in ids)
        for tissue in TISSUES
    }
    floors = {"tumor": 50_000, "stroma": 5_000, "necrosis": 1_000}
    for tissue, floor in floors.items():
        if patch_counts[tissue] < floor:
            message = f"{split_name} has fewer than {floor} {tissue} patches"
            raise ValueError(message)
    necrosis_ids = [
        wsi_id for wsi_id in ids if summary.stats[wsi_id].tissue_counts["necrosis"]
    ]
    low_ids = [
        wsi_id
        for wsi_id in necrosis_ids
        if summary.stats[wsi_id].tissue_counts["necrosis"]
        < MIN_HOLDOUT_NECROSIS_PATCHES_PER_WSI
    ]
    if low_ids:
        message = f"{split_name} has low-volume necrosis WSIs {low_ids}"
        raise ValueError(message)
    diagnoses = {
        split_entries[wsi_id].cohort.diagnosis_label for wsi_id in necrosis_ids
    }
    required = {"CC", "EC", "HGSC"}
    if not required.issubset(diagnoses):
        message = f"{split_name} necrosis diagnoses lack {sorted(required - diagnoses)}"
        raise ValueError(message)


def _validate_mc_necrosis(
    split_entries: Mapping[int, SplitEntry],
    summary: AtlasSummary,
) -> None:
    counts = Counter(
        entry.split
        for wsi_id, entry in split_entries.items()
        if entry.cohort.diagnosis_label == "MC"
        and summary.stats[wsi_id].tissue_counts["necrosis"] > 0
    )
    if counts != Counter({"train": 1, "test": 1}):
        message = f"MC-necrosis allocation disagrees: {dict(counts)}"
        raise ValueError(message)


def _split_audit(  # noqa: PLR0913, PLR0917
    ids: Sequence[int],
    split_entries: Mapping[int, SplitEntry],
    summary: AtlasSummary,
    diagnosis_counts: Counter[str],
    updated_count: int,
    tissue_patch_counts: Mapping[TissueName, int],
    tissue_positive_counts: Mapping[TissueName, int],
) -> dict[str, object]:
    diagnosis_tissue_patches: dict[str, dict[str, int]] = {}
    diagnosis_tissue_positive: dict[str, dict[str, int]] = {}
    for diagnosis in LABEL_INDEX:
        diagnosis_ids = [
            wsi_id
            for wsi_id in ids
            if split_entries[wsi_id].cohort.diagnosis_label == diagnosis
        ]
        diagnosis_tissue_patches[diagnosis] = {
            tissue: sum(
                summary.stats[wsi_id].tissue_counts[tissue] for wsi_id in diagnosis_ids
            )
            for tissue in TISSUES
        }
        diagnosis_tissue_positive[diagnosis] = {
            tissue: sum(
                summary.stats[wsi_id].tissue_counts[tissue] > 0
                for wsi_id in diagnosis_ids
            )
            for tissue in TISSUES
        }
    return {
        "wsi_count": len(ids),
        "diagnosis_wsi_counts": dict(sorted(diagnosis_counts.items())),
        "updated_wsi_count": updated_count,
        "cancer_ae_patch_count": sum(
            summary.stats[wsi_id].cancer_count for wsi_id in ids
        ),
        "tissue_patch_counts": dict(tissue_patch_counts),
        "tissue_positive_wsi_counts": dict(tissue_positive_counts),
        "necrosis_patch_counts_by_positive_wsi": {
            str(wsi_id): summary.stats[wsi_id].tissue_counts["necrosis"]
            for wsi_id in ids
            if summary.stats[wsi_id].tissue_counts["necrosis"] > 0
        },
        "diagnosis_tissue_patch_counts": diagnosis_tissue_patches,
        "diagnosis_tissue_positive_wsi_counts": diagnosis_tissue_positive,
    }


def write_manifests(
    *,
    atlas_path: Path,
    cohort: Mapping[int, CohortEntry],
    split_entries: Mapping[int, SplitEntry],
    cancer_path: Path,
    tissue_path: Path,
) -> tuple[int, int]:
    """Write both logical task views in exact atlas order.

    Returns:
        Cancer/AE and tissue data-row counts.

    """
    cancer_count = 0
    tissue_count = 0
    with (
        cancer_path.open("w", encoding="utf-8", newline="") as cancer_handle,
        tissue_path.open("w", encoding="utf-8", newline="") as tissue_handle,
    ):
        cancer_writer = csv.writer(cancer_handle, lineterminator="\n")
        tissue_writer = csv.writer(tissue_handle, lineterminator="\n")
        cancer_writer.writerow(CANCER_HEADER)
        tissue_writer.writerow(TISSUE_HEADER)
        for row in iter_atlas_rows(atlas_path, cohort):
            entry = split_entries[row.wsi_id]
            common = (
                row.row_index,
                row.wsi_id,
                entry.cohort.diagnosis_label,
                entry.cohort.diagnosis_index,
                row.x,
                row.y,
                entry.split,
            )
            if row.selection_source in OTSU_SOURCES:
                cancer_writer.writerow(common)
                cancer_count += 1
            if row.tissue_label is not None:
                tissue_writer.writerow(
                    (
                        *common,
                        row.tissue_label,
                        row.annotated_text,
                        row.fractions_text[TISSUES.index(row.tissue_label)],
                        format(row.purity, ".17g"),
                        *row.fractions_text,
                    ),
                )
                tissue_count += 1
        cancer_handle.flush()
        tissue_handle.flush()
        os.fsync(cancer_handle.fileno())
        os.fsync(tissue_handle.fileno())
    return cancer_count, tissue_count


def _audit_payload(  # noqa: PLR0913
    *,
    atlas_path: Path,
    atlas_hash: str,
    cohort_path: Path,
    split_path: Path,
    cancer_path: Path,
    tissue_path: Path,
    summary: AtlasSummary,
    split_audit: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": "spec0018.ubc_eval_manifests.v1",
        "status": "pass",
        "selection_disclosure": (
            "WSI assignment is mask-stratified; cancer/AE coordinates and exported "
            "features are the complete Otsu-derived set and contain no mask metadata."
        ),
        "primary_vae_population": "sealed test WSIs only",
        "thresholds": {
            "minimum_annotated_fraction": MIN_ANNOTATED_FRACTION,
            "minimum_tissue_purity": MIN_TISSUE_PURITY,
        },
        "locked_constraints": {
            "diagnosis_wsi_quotas": EXPECTED_QUOTAS,
            "updated_wsi_counts": EXPECTED_UPDATED_COUNTS,
            "necrosis_positive_wsi_counts": EXPECTED_NECROSIS_WSI_COUNTS,
            "minimum_holdout_stroma_positive_wsis": MIN_HOLDOUT_STROMA_WSI_COUNT,
            "minimum_holdout_necrosis_patches_per_positive_wsi": (
                MIN_HOLDOUT_NECROSIS_PATCHES_PER_WSI
            ),
            "minimum_train_necrosis_patches": MIN_TRAIN_NECROSIS_PATCHES,
            "minimum_holdout_tissue_patches": {
                "tumor": 50_000,
                "stroma": 5_000,
                "necrosis": 1_000,
            },
            "required_holdout_necrosis_diagnoses": ["CC", "EC", "HGSC"],
            "mc_necrosis_wsi_allocation": {"train": 1, "test": 1},
            "cancer_selection_sources": sorted(OTSU_SOURCES),
            "expected_real_row_counts": {
                "atlas": EXPECTED_ATLAS_ROWS,
                "cancer_ae": EXPECTED_CANCER_ROWS,
                "tissue": EXPECTED_TISSUE_ROWS,
            },
        },
        "inputs": {
            "atlas": {
                "path": str(atlas_path),
                "sha256": atlas_hash,
                "kaggle_kernel": "maximusshtefan/eqvae-ubc-ocean-test-atlas",
                "kaggle_kernel_version": 3,
            },
            "cohort": {"path": str(cohort_path), "sha256": sha256_file(cohort_path)},
            "split": {"path": str(split_path), "sha256": sha256_file(split_path)},
            "generator": {
                "path": str(GENERATOR_REPO_PATH),
                "sha256": sha256_file(Path(__file__)),
            },
        },
        "outputs": {
            "cancer_ae_manifest": {
                "path": str(cancer_path),
                "sha256": sha256_file(cancer_path),
            },
            "tissue_manifest": {
                "path": str(tissue_path),
                "sha256": sha256_file(tissue_path),
            },
        },
        "row_counts": {
            "atlas": summary.atlas_row_count,
            "cancer_ae": summary.cancer_count,
            "tissue": summary.tissue_count,
        },
        "splits": dict(split_audit),
        "acceptance": {
            "canonical_split_valid": True,
            "canonical_cohort_hash_valid": True,
            "canonical_split_hash_valid": True,
            "diagnosis_quotas_valid": True,
            "updated_image_quotas_valid": True,
            "tissue_positive_wsi_constraints_valid": True,
            "holdout_necrosis_per_wsi_minimum_valid": True,
            "holdout_tissue_patch_floors_valid": True,
            "holdout_necrosis_diagnosis_coverage_valid": True,
            "mc_necrosis_allocation_valid": True,
            "cancer_coordinates_are_complete_otsu_subset": True,
            "mask_only_excluded_from_cancer": True,
            "tissue_coverage_and_purity_valid": True,
            "shared_wsi_split": True,
            "atlas_order_preserved": True,
            "no_model_outputs_read": True,
        },
    }


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _fraction(text: str, name: str, row_index: int) -> float:
    try:
        value = float(text)
    except ValueError as error:
        message = f"Invalid {name} at atlas row {row_index}: {text!r}"
        raise ValueError(message) from error
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        message = f"Out-of-range {name} at atlas row {row_index}: {value}"
        raise ValueError(message)
    return value


def _int_value(row: Mapping[str, str | None], name: str) -> int:
    text = _required(row, name)
    try:
        return int(text)
    except ValueError as error:
        message = f"Invalid integer {name}: {text!r}"
        raise ValueError(message) from error


def _required(row: Mapping[str, str | None], name: str) -> str:
    value = row.get(name)
    if not value:
        message = f"Missing required CSV value {name!r}"
        raise ValueError(message)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
