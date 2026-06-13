# Copyright 2026 HiperMaximus
"""Split leakage and count validation helpers for spec 0001."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from eqvae.data.patch_shards import PatchRecord

type SplitStatus = Literal["synthetic_pass", "pass", "warn", "fail"]
type SplitValidationMode = Literal["synthetic", "real"]


@dataclass(frozen=True)
class SplitSummary:
    """Patch, WSI, and label counts for one split."""

    split: str
    patch_count: int
    wsi_count: int
    label_counts: dict[int, int]


@dataclass(frozen=True)
class SplitValidationResult:
    """Validation result for train/validation split separation."""

    status: SplitStatus
    train: SplitSummary
    validation: SplitSummary
    overlap_wsi_ids: tuple[str, ...]
    masked_holdout_overlap_wsi_ids: tuple[str, ...]
    warnings: tuple[str, ...]
    failures: tuple[str, ...]


@dataclass(frozen=True)
class SplitValidationOptions:
    """Optional count and provenance policy for split validation."""

    expected_train_wsi_count: int | None = None
    expected_validation_wsi_count: int | None = None
    expected_train_patch_count: int | None = None
    expected_validation_patch_count: int | None = None
    masked_holdout_wsi_ids: frozenset[str] = frozenset()
    mode: SplitValidationMode = "real"
    non_tma_provenance_checked: bool = False


def summarize_records(split: str, records: Sequence[PatchRecord]) -> SplitSummary:
    """Summarize one split's patch metadata.

    Returns:
        Split summary.

    """
    label_counter = Counter(record.label for record in records)
    return SplitSummary(
        split=split,
        patch_count=len(records),
        wsi_count=len({record.wsi_id for record in records}),
        label_counts=dict(sorted(label_counter.items())),
    )


def validate_train_validation_splits(
    *,
    train_records: Sequence[PatchRecord],
    validation_records: Sequence[PatchRecord],
    options: SplitValidationOptions | None = None,
) -> SplitValidationResult:
    """Validate WSI separation and optional expected split counts.

    Args:
        train_records: Train patch records.
        validation_records: Validation patch records.
        options: Optional count, holdout, and provenance policy.

    Returns:
        Split validation result with concrete failure messages.

    """
    validation_options = options or SplitValidationOptions()
    train_summary = summarize_records("train", train_records)
    validation_summary = summarize_records("validation", validation_records)

    train_wsi_ids = {record.wsi_id for record in train_records}
    validation_wsi_ids = {record.wsi_id for record in validation_records}
    holdout_wsi_ids = set(validation_options.masked_holdout_wsi_ids)
    overlap = tuple(sorted(train_wsi_ids.intersection(validation_wsi_ids)))
    masked_overlap = tuple(
        sorted(train_wsi_ids.union(validation_wsi_ids).intersection(holdout_wsi_ids)),
    )

    failures: list[str] = []
    warnings: list[str] = []
    if overlap:
        failures.append(f"train/validation WSI overlap: {list(overlap)}")
    if masked_overlap:
        failures.append(f"masked-holdout WSI overlap: {list(masked_overlap)}")
    if validation_options.mode == "real":
        _append_real_evidence_warnings(warnings, options=validation_options)
    _append_expected_failure(
        failures,
        name="train patch count",
        expected=validation_options.expected_train_patch_count,
        observed=train_summary.patch_count,
    )
    _append_expected_failure(
        failures,
        name="validation patch count",
        expected=validation_options.expected_validation_patch_count,
        observed=validation_summary.patch_count,
    )
    _append_expected_failure(
        failures,
        name="train WSI count",
        expected=validation_options.expected_train_wsi_count,
        observed=train_summary.wsi_count,
    )
    _append_expected_failure(
        failures,
        name="validation WSI count",
        expected=validation_options.expected_validation_wsi_count,
        observed=validation_summary.wsi_count,
    )

    status = _split_status(
        mode=validation_options.mode,
        has_failures=bool(failures),
        has_warnings=bool(warnings),
    )
    return SplitValidationResult(
        status=status,
        train=train_summary,
        validation=validation_summary,
        overlap_wsi_ids=overlap,
        masked_holdout_overlap_wsi_ids=masked_overlap,
        warnings=tuple(warnings),
        failures=tuple(failures),
    )


def load_masked_holdout_wsi_ids(
    csv_path: Path,
    *,
    column_name: str = "image_id",
) -> frozenset[str]:
    """Load masked-holdout image IDs from a CSV by column name.

    Returns:
        Masked-holdout IDs as strings.

    Raises:
        ValueError: If the requested column is missing.

    """
    with csv_path.open(encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None or column_name not in reader.fieldnames:
            message = f"Masked holdout CSV {csv_path} is missing {column_name!r}"
            raise ValueError(message)
        values = [
            _required(cast("dict[str, str | None]", row), column_name) for row in reader
        ]
    return frozenset(values)


def _append_expected_failure(
    failures: list[str],
    *,
    name: str,
    expected: int | None,
    observed: int,
) -> None:
    if expected is not None and observed != expected:
        failures.append(f"{name}: expected {expected}, observed {observed}")


def _append_real_evidence_warnings(
    warnings: list[str],
    *,
    options: SplitValidationOptions,
) -> None:
    if options.expected_train_patch_count is None:
        warnings.append("real split lacks expected train patch count")
    if options.expected_validation_patch_count is None:
        warnings.append("real split lacks expected validation patch count")
    if options.expected_train_wsi_count is None:
        warnings.append("real split lacks expected train WSI count")
    if options.expected_validation_wsi_count is None:
        warnings.append("real split lacks expected validation WSI count")
    if not options.masked_holdout_wsi_ids:
        warnings.append("real split lacks masked-holdout ID list")
    if not options.non_tma_provenance_checked:
        warnings.append("real split lacks non-TMA provenance check")


def _split_status(
    *,
    mode: SplitValidationMode,
    has_failures: bool,
    has_warnings: bool,
) -> SplitStatus:
    if has_failures:
        return "fail"
    if mode == "synthetic":
        return "synthetic_pass"
    if has_warnings:
        return "warn"
    return "pass"


def _required(row: dict[str, str | None], column_name: str) -> str:
    value = row.get(column_name)
    if not value:
        message = f"Missing {column_name!r} value in masked holdout CSV"
        raise ValueError(message)
    return value


__all__ = [
    "SplitStatus",
    "SplitSummary",
    "SplitValidationMode",
    "SplitValidationOptions",
    "SplitValidationResult",
    "load_masked_holdout_wsi_ids",
    "summarize_records",
    "validate_train_validation_splits",
]
