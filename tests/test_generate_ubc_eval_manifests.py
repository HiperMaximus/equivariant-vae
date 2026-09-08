# Copyright 2026 HiperMaximus
"""Tests for the frozen shared-WSI evaluation manifest contract."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import replace
from typing import TYPE_CHECKING, cast

import pytest

import eqvae.cli.generate_ubc_eval_manifests as manifest_cli
from eqvae.cli.generate_ubc_eval_manifests import (
    ATLAS_PATH,
    CANCER_HEADER,
    COHORT_PATH,
    COHORT_SHA256,
    EXPECTED_CANCER_MANIFEST_SHA256,
    EXPECTED_TISSUE_MANIFEST_SHA256,
    SPLIT_PATH,
    SPLIT_SHA256,
    TISSUE_HEADER,
    AtlasSummary,
    CohortEntry,
    SplitEntry,
    TissueName,
    WsiStats,
    iter_atlas_rows,
    load_cohort,
    load_split,
    materialize_manifests,
    sha256_file,
    validate_pinned_hash,
    validate_split,
    write_manifests,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

EXPECTED_COHORT_COUNT = 152
EXPECTED_SMALL_TASK_ROWS = 2
EXPECTED_HOLDOUT_NECROSIS_WSI_COUNT = 5
MIN_HOLDOUT_NECROSIS_PATCHES = 100
AUDIT_FAILURE_MESSAGE = "simulated audit publication failure"

TRAIN_NECROSIS_IDS = {
    1925,
    1952,
    9697,
    14424,
    15188,
    16986,
    21432,
    22489,
    24759,
    35239,
    36678,
    37655,
    38048,
    43390,
    45185,
    45630,
    56947,
}
VALIDATION_NECROSIS_IDS = {22155, 39255, 39728, 59760, 63941}
TEST_NECROSIS_IDS = {10246, 19255, 22221, 38019, 40888}


def test_canonical_split_exactly_covers_the_committed_cohort() -> None:
    """The sealed test IDs must remain a complete metadata-consistent partition."""
    cohort = load_cohort(COHORT_PATH)
    split = load_split(SPLIT_PATH, cohort)

    assert len(cohort) == EXPECTED_COHORT_COUNT
    assert set(split) == set(cohort)
    assert Counter(entry.split for entry in split.values()) == {
        "train": 106,
        "validation": 23,
        "test": 23,
    }


def test_exact_cohort_and_split_bytes_are_sealed() -> None:
    """Quota-preserving ID swaps must not silently replace the frozen test cohort."""
    assert sha256_file(COHORT_PATH) == COHORT_SHA256
    assert sha256_file(SPLIT_PATH) == SPLIT_SHA256


def test_same_stratum_split_swap_fails_the_pinned_hash(tmp_path: Path) -> None:
    """Structural balance cannot authorize changing which HGSC WSI is held out."""
    changed_split = tmp_path / "changed_split.csv"
    text = SPLIT_PATH.read_text(encoding="utf-8")
    text = text.replace("1020,HGSC,2,false,train", "1020,HGSC,2,false,test")
    text = text.replace("1252,HGSC,2,false,test", "1252,HGSC,2,false,train")
    changed_split.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="Canonical split SHA-256 mismatch"):
        validate_pinned_hash(changed_split, SPLIT_SHA256, "Canonical split")


def test_split_validator_counts_independent_necrosis_wsis() -> None:
    """Large patch totals cannot substitute for the locked rare-WSI coverage gates."""
    cohort = load_cohort(COHORT_PATH)
    split = load_split(SPLIT_PATH, cohort)
    summary = _passing_summary(split)

    audit = validate_split(split, summary, enforce_real_counts=False)

    validation = cast("dict[str, object]", audit["validation"])
    assert validation["tissue_positive_wsi_counts"] == {
        "tumor": 23,
        "stroma": 23,
        "necrosis": 5,
    }


def test_split_validator_rejects_a_low_volume_holdout_necrosis_wsi() -> None:
    """A painted edge speck must not satisfy the five-independent-WSI holdout claim."""
    cohort = load_cohort(COHORT_PATH)
    split = load_split(SPLIT_PATH, cohort)
    summary = _passing_summary(split)
    summary.stats[22155].tissue_counts["necrosis"] = 99

    with pytest.raises(ValueError, match="low-volume necrosis"):
        validate_split(split, summary, enforce_real_counts=False)


def test_split_validator_rejects_changed_updated_image_count() -> None:
    """Diagnosis balance alone cannot replace the locked technical-image quota."""
    cohort = load_cohort(COHORT_PATH)
    split = load_split(SPLIT_PATH, cohort)
    summary = _passing_summary(split)
    changed = dict(split)
    wsi_id = next(
        candidate
        for candidate, entry in changed.items()
        if entry.split == "validation" and entry.cohort.is_updated_image_id
    )
    changed[wsi_id] = SplitEntry(
        replace(changed[wsi_id].cohort, is_updated_image_id=False),
        "validation",
    )

    with pytest.raises(ValueError, match="updated-image count"):
        validate_split(changed, summary, enforce_real_counts=False)


def test_split_validator_rejects_stroma_wsi_and_patch_floor_failures() -> None:
    """Many patches from a few slides cannot satisfy both holdout stroma gates."""
    cohort = load_cohort(COHORT_PATH)
    split = load_split(SPLIT_PATH, cohort)
    validation_ids = [
        wsi_id for wsi_id, entry in split.items() if entry.split == "validation"
    ]

    low_wsi_summary = _passing_summary(split)
    for wsi_id in validation_ids[:4]:
        low_wsi_summary.stats[wsi_id].tissue_counts["stroma"] = 0
    with pytest.raises(ValueError, match="stroma-positive WSIs"):
        validate_split(split, low_wsi_summary, enforce_real_counts=False)

    low_patch_summary = _passing_summary(split)
    for wsi_id in validation_ids:
        low_patch_summary.stats[wsi_id].tissue_counts["stroma"] = 1
    with pytest.raises(ValueError, match="fewer than 5000 stroma patches"):
        validate_split(split, low_patch_summary, enforce_real_counts=False)


def test_task_manifests_share_split_but_not_mask_selection(tmp_path: Path) -> None:
    """Mask-only patches belong to tissue, never the primary cancer/AE set."""
    atlas_path = tmp_path / "atlas.csv"
    cancer_path = tmp_path / "cancer.csv"
    tissue_path = tmp_path / "tissue.csv"
    _write_small_atlas(atlas_path)
    cohort = {
        1: CohortEntry(
            wsi_id=1,
            diagnosis_label="CC",
            diagnosis_index=0,
            is_updated_image_id=False,
        ),
    }
    split = {1: SplitEntry(cohort[1], "test")}

    cancer_count, tissue_count = write_manifests(
        atlas_path=atlas_path,
        cohort=cohort,
        split_entries=split,
        cancer_path=cancer_path,
        tissue_path=tissue_path,
    )

    cancer_rows = list(csv.DictReader(cancer_path.open(encoding="utf-8", newline="")))
    tissue_rows = list(csv.DictReader(tissue_path.open(encoding="utf-8", newline="")))
    assert cancer_count == EXPECTED_SMALL_TASK_ROWS
    assert tissue_count == EXPECTED_SMALL_TASK_ROWS
    assert tuple(cancer_rows[0]) == CANCER_HEADER
    assert tuple(tissue_rows[0]) == TISSUE_HEADER
    assert [row["atlas_row_index"] for row in cancer_rows] == ["0", "2"]
    assert [row["atlas_row_index"] for row in tissue_rows] == ["1", "2"]
    assert {row["split"] for row in cancer_rows + tissue_rows} == {"test"}
    assert "annotated_fraction" not in cancer_rows[0]


def test_atlas_iterator_rejects_duplicate_or_reordered_coordinates(
    tmp_path: Path,
) -> None:
    """Manifest order must remain a one-to-one view of the authoritative atlas."""
    atlas_path = tmp_path / "atlas.csv"
    _write_small_atlas(atlas_path)
    text = atlas_path.read_text(encoding="utf-8")
    atlas_path.write_text(text.replace("256,0,mask", "0,0,mask"), encoding="utf-8")
    cohort = {
        1: CohortEntry(
            wsi_id=1,
            diagnosis_label="CC",
            diagnosis_index=0,
            is_updated_image_id=False,
        ),
    }

    with pytest.raises(ValueError, match="order/uniqueness"):
        list(iter_atlas_rows(atlas_path, cohort))


def test_loaders_reject_unknown_cohort_label_and_invalid_atlas_fraction(
    tmp_path: Path,
) -> None:
    """Canonical labels and authoritative mask fractions fail closed."""
    changed_cohort = tmp_path / "cohort.csv"
    cohort_text = COHORT_PATH.read_text(encoding="utf-8")
    changed_cohort.write_text(
        cohort_text.replace(",LGSC,", ",UNKNOWN,", 1),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unknown diagnosis label"):
        load_cohort(changed_cohort)

    atlas_path = tmp_path / "atlas.csv"
    _write_small_atlas(atlas_path)
    atlas_text = atlas_path.read_text(encoding="utf-8")
    atlas_path.write_text(
        atlas_text.replace(",0.5,0.5,0,0\n", ",0.5,0.6,0,0\n", 1),
        encoding="utf-8",
    )
    cohort = {
        1: CohortEntry(
            wsi_id=1,
            diagnosis_label="CC",
            diagnosis_index=0,
            is_updated_image_id=False,
        ),
    }
    with pytest.raises(ValueError, match="Class fraction exceeds annotation"):
        list(iter_atlas_rows(atlas_path, cohort))


def test_small_materialization_is_deterministic_and_publishes_audit_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The audit is a truthful completion marker written after stable manifests."""
    atlas_path = tmp_path / "atlas.csv"
    atlas_path.write_text("fixture-atlas\n", encoding="utf-8")
    output_dir = tmp_path / "manifests"
    audit_path = tmp_path / "audit.json"
    _install_small_materialization_stubs(monkeypatch)

    observed_order: list[str] = []

    def recording_write_json(path: Path, payload: Mapping[str, object]) -> None:
        cancer_path = output_dir / "cancer_ae_patch_manifest.csv"
        tissue_path = output_dir / "tissue_patch_manifest.csv"
        assert cancer_path.is_file()
        assert tissue_path.is_file()
        assert not audit_path.exists()
        outputs = cast("dict[str, dict[str, str]]", payload["outputs"])
        assert outputs["cancer_ae_manifest"]["sha256"] == sha256_file(cancer_path)
        assert outputs["tissue_manifest"]["sha256"] == sha256_file(tissue_path)
        observed_order.append("audit_after_manifests")
        path.write_text(
            f"{json.dumps(payload, indent=2, sort_keys=True)}\n",
            encoding="utf-8",
            newline="\n",
        )

    monkeypatch.setattr(manifest_cli, "_write_json", recording_write_json)
    first = materialize_manifests(
        atlas_path=atlas_path,
        cohort_path=COHORT_PATH,
        split_path=SPLIT_PATH,
        output_dir=output_dir,
        audit_path=audit_path,
        expected_atlas_sha256=None,
        enforce_real_counts=False,
    )
    first_bytes = (
        (output_dir / "cancer_ae_patch_manifest.csv").read_bytes(),
        (output_dir / "tissue_patch_manifest.csv").read_bytes(),
        audit_path.read_bytes(),
    )
    second = materialize_manifests(
        atlas_path=atlas_path,
        cohort_path=COHORT_PATH,
        split_path=SPLIT_PATH,
        output_dir=output_dir,
        audit_path=audit_path,
        expected_atlas_sha256=None,
        enforce_real_counts=False,
    )
    second_bytes = (
        (output_dir / "cancer_ae_patch_manifest.csv").read_bytes(),
        (output_dir / "tissue_patch_manifest.csv").read_bytes(),
        audit_path.read_bytes(),
    )

    assert observed_order == ["audit_after_manifests", "audit_after_manifests"]
    assert first == second == json.loads(audit_path.read_text(encoding="utf-8"))
    assert first_bytes == second_bytes


def test_audit_write_failure_leaves_no_completion_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A publication failure may leave manifests, but must never claim completion."""
    atlas_path = tmp_path / "atlas.csv"
    atlas_path.write_text("fixture-atlas\n", encoding="utf-8")
    output_dir = tmp_path / "manifests"
    audit_path = tmp_path / "audit.json"
    _install_small_materialization_stubs(monkeypatch)

    def fail_audit_write(path: Path, payload: Mapping[str, object]) -> None:
        del path, payload
        assert (output_dir / "cancer_ae_patch_manifest.csv").is_file()
        assert (output_dir / "tissue_patch_manifest.csv").is_file()
        assert not audit_path.exists()
        raise OSError(AUDIT_FAILURE_MESSAGE)

    monkeypatch.setattr(manifest_cli, "_write_json", fail_audit_write)
    with pytest.raises(OSError, match="simulated audit publication failure"):
        materialize_manifests(
            atlas_path=atlas_path,
            cohort_path=COHORT_PATH,
            split_path=SPLIT_PATH,
            output_dir=output_dir,
            audit_path=audit_path,
            expected_atlas_sha256=None,
            enforce_real_counts=False,
        )

    assert not audit_path.exists()
    assert not audit_path.with_suffix(".json.tmp").exists()


@pytest.mark.skipif(
    not ATLAS_PATH.is_file(),
    reason="verified real atlas is local-only",
)
def test_real_materialization_matches_both_sealed_manifest_hashes(
    tmp_path: Path,
) -> None:
    """The complete atlas must reproduce both views, not merely pass unit fixtures."""
    output_dir = tmp_path / "manifests"
    audit_path = tmp_path / "audit.json"

    audit = materialize_manifests(
        atlas_path=ATLAS_PATH,
        cohort_path=COHORT_PATH,
        split_path=SPLIT_PATH,
        output_dir=output_dir,
        audit_path=audit_path,
    )

    assert audit["row_counts"] == {
        "atlas": 1_822_340,
        "cancer_ae": 1_750_221,
        "tissue": 666_807,
    }
    assert (
        sha256_file(output_dir / "cancer_ae_patch_manifest.csv")
        == EXPECTED_CANCER_MANIFEST_SHA256
    )
    assert (
        sha256_file(output_dir / "tissue_patch_manifest.csv")
        == EXPECTED_TISSUE_MANIFEST_SHA256
    )
    stored_audit = cast(
        "dict[str, object]",
        json.loads(audit_path.read_text(encoding="utf-8")),
    )
    assert stored_audit == audit
    outputs = cast("dict[str, dict[str, str]]", audit["outputs"])
    assert outputs["cancer_ae_manifest"]["sha256"] == EXPECTED_CANCER_MANIFEST_SHA256
    assert outputs["tissue_manifest"]["sha256"] == EXPECTED_TISSUE_MANIFEST_SHA256
    constraints = cast("dict[str, object]", audit["locked_constraints"])
    assert (
        constraints["minimum_holdout_necrosis_patches_per_positive_wsi"]
        == MIN_HOLDOUT_NECROSIS_PATCHES
    )
    splits = cast("dict[str, dict[str, object]]", audit["splits"])
    for split_name in ("validation", "test"):
        counts = cast(
            "dict[str, int]",
            splits[split_name]["necrosis_patch_counts_by_positive_wsi"],
        )
        assert len(counts) == EXPECTED_HOLDOUT_NECROSIS_WSI_COUNT
        assert min(counts.values()) >= MIN_HOLDOUT_NECROSIS_PATCHES
    acceptance = cast("dict[str, bool]", audit["acceptance"])
    assert acceptance
    assert all(acceptance.values())


def _passing_summary(split: dict[int, SplitEntry]) -> AtlasSummary:
    """Build cheap WSI statistics that exercise every locked split constraint.

    Returns:
        Synthetic per-WSI counts satisfying every constraint.

    """
    stats: dict[int, WsiStats] = {}
    all_necrosis = TRAIN_NECROSIS_IDS | VALIDATION_NECROSIS_IDS | TEST_NECROSIS_IDS
    for wsi_id in split:
        tissue_counts: Counter[TissueName] = Counter(
            {"tumor": 3_000, "stroma": 300},
        )
        if wsi_id in all_necrosis:
            tissue_counts["necrosis"] = 300
        stats[wsi_id] = WsiStats(
            cancer_count=1_000,
            tissue_counts=Counter(tissue_counts),
        )
    return AtlasSummary(
        atlas_row_count=0,
        cancer_count=0,
        tissue_count=0,
        stats=stats,
    )


def _write_small_atlas(path: Path) -> None:
    """Write three rows that distinguish Otsu, mask-only, and their intersection."""
    path.write_text(
        "wsi_id,label,x,y,selection_source,mask_status,mask_label,"
        "annotated_fraction,tumor_fraction,stroma_fraction,necrosis_fraction\n"
        "1,0,0,0,otsu,unannotated,unknown,0,0,0,0\n"
        "1,0,256,0,mask,annotated,tumor,0.5,0.5,0,0\n"
        "1,0,0,256,mask+otsu,annotated,stroma,0.5,0,0.5,0\n"
        "1,0,256,256,mask,annotated,tumor,1,0.94,0.06,0\n",
        encoding="utf-8",
    )


def _install_small_materialization_stubs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replace the expensive atlas pass while retaining real sealed metadata."""

    def fake_summarize(
        path: Path,
        cohort: Mapping[int, CohortEntry],
    ) -> AtlasSummary:
        del path, cohort
        return AtlasSummary(
            atlas_row_count=3,
            cancer_count=1,
            tissue_count=1,
            stats={},
        )

    def fake_validate(
        split_entries: Mapping[int, SplitEntry],
        summary: AtlasSummary,
        *,
        enforce_real_counts: bool,
    ) -> dict[str, object]:
        del split_entries, summary, enforce_real_counts
        return {"train": {}, "validation": {}, "test": {}}

    def fake_write(
        *,
        atlas_path: Path,
        cohort: Mapping[int, CohortEntry],
        split_entries: Mapping[int, SplitEntry],
        cancer_path: Path,
        tissue_path: Path,
    ) -> tuple[int, int]:
        del atlas_path, cohort, split_entries
        cancer_path.write_text("cancer\n", encoding="utf-8", newline="")
        tissue_path.write_text("tissue\n", encoding="utf-8", newline="")
        return 1, 1

    monkeypatch.setattr(manifest_cli, "summarize_atlas", fake_summarize)
    monkeypatch.setattr(manifest_cli, "validate_split", fake_validate)
    monkeypatch.setattr(manifest_cli, "write_manifests", fake_write)
