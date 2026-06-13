# Copyright 2026 HiperMaximus
"""Tests for spec 0001 split validation helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from eqvae.data.patch_shards import PatchRecord
from eqvae.data.splits import (
    SplitValidationOptions,
    load_masked_holdout_wsi_ids,
    summarize_records,
    validate_train_validation_splits,
)

if TYPE_CHECKING:
    from pathlib import Path

TRAIN_PATCH_COUNT = 3
VALIDATION_PATCH_COUNT = 2
TRAIN_WSI_COUNT = 2
VALIDATION_WSI_COUNT = 2
LEAKAGE_FAILURE_COUNT = 2


def test_synthetic_split_validation_uses_distinct_status() -> None:
    """Synthetic fixtures can pass without pretending to be real UBC evidence."""
    train_records = _records("train_wsi", TRAIN_PATCH_COUNT)
    validation_records = _records("validation_wsi", VALIDATION_PATCH_COUNT)

    result = validate_train_validation_splits(
        train_records=train_records,
        validation_records=validation_records,
        options=SplitValidationOptions(
            expected_train_patch_count=TRAIN_PATCH_COUNT,
            expected_validation_patch_count=VALIDATION_PATCH_COUNT,
            expected_train_wsi_count=TRAIN_WSI_COUNT,
            expected_validation_wsi_count=VALIDATION_WSI_COUNT,
            mode="synthetic",
        ),
    )

    assert result.status == "synthetic_pass"
    assert result.failures == ()
    assert result.warnings == ()
    assert result.train.patch_count == TRAIN_PATCH_COUNT
    assert result.validation.patch_count == VALIDATION_PATCH_COUNT


def test_real_split_validation_warns_without_non_tma_provenance() -> None:
    """Real split checks cannot claim pass without non-TMA provenance."""
    result = validate_train_validation_splits(
        train_records=_records("train_wsi", TRAIN_PATCH_COUNT),
        validation_records=_records("validation_wsi", VALIDATION_PATCH_COUNT),
        options=SplitValidationOptions(mode="real"),
    )

    assert result.status == "warn"
    assert result.failures == ()
    assert "real split lacks non-TMA provenance check" in result.warnings
    assert "real split lacks expected train patch count" in result.warnings
    assert "real split lacks masked-holdout ID list" in result.warnings


def test_real_split_validation_passes_with_non_tma_provenance() -> None:
    """Real split checks pass only when all real-data evidence is explicit."""
    result = validate_train_validation_splits(
        train_records=_records("train_wsi", TRAIN_PATCH_COUNT),
        validation_records=_records("validation_wsi", VALIDATION_PATCH_COUNT),
        options=SplitValidationOptions(
            expected_train_patch_count=TRAIN_PATCH_COUNT,
            expected_validation_patch_count=VALIDATION_PATCH_COUNT,
            expected_train_wsi_count=TRAIN_WSI_COUNT,
            expected_validation_wsi_count=VALIDATION_WSI_COUNT,
            masked_holdout_wsi_ids=frozenset({"holdout_wsi_0000"}),
            mode="real",
            non_tma_provenance_checked=True,
        ),
    )

    assert result.status == "pass"
    assert result.failures == ()
    assert result.warnings == ()


def test_split_validation_detects_overlap_and_masked_holdout() -> None:
    """WSI leakage and masked-holdout leakage force failure."""
    train_records = _records("shared", TRAIN_PATCH_COUNT)
    validation_records = _records("shared", VALIDATION_PATCH_COUNT)

    result = validate_train_validation_splits(
        train_records=train_records,
        validation_records=validation_records,
        options=SplitValidationOptions(
            masked_holdout_wsi_ids=frozenset({"shared_0000"}),
            mode="synthetic",
        ),
    )

    assert result.status == "fail"
    assert result.overlap_wsi_ids == ("shared_0000", "shared_0001")
    assert result.masked_holdout_overlap_wsi_ids == ("shared_0000",)
    assert len(result.failures) == LEAKAGE_FAILURE_COUNT


def test_summarize_records_reports_label_counts() -> None:
    """Split summaries include patch, WSI, and numeric label counts."""
    summary = summarize_records("train", _records("train_wsi", TRAIN_PATCH_COUNT))

    assert summary.patch_count == TRAIN_PATCH_COUNT
    assert summary.wsi_count == TRAIN_WSI_COUNT
    assert summary.label_counts == {0: 1, 1: 1, 2: 1}


def test_load_masked_holdout_wsi_ids_by_column_name(tmp_path: Path) -> None:
    """The committed holdout CSV convention loads image IDs as strings."""
    csv_path = tmp_path / "masked.csv"
    csv_path.write_text(
        "image_id,label,is_updated_image_id\n101,HGSC,false\n202,CC,true\n",
        encoding="utf-8",
    )

    assert load_masked_holdout_wsi_ids(csv_path) == frozenset({"101", "202"})


def _records(prefix: str, count: int) -> list[PatchRecord]:
    return [
        PatchRecord(
            file_index=index,
            row_index=index,
            wsi_id=f"{prefix}_{index % 2:04d}",
            label=index % 5,
            x=index * 10,
            y=index * 20,
        )
        for index in range(count)
    ]
