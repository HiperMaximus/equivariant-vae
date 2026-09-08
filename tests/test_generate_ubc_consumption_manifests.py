# Copyright 2026 HiperMaximus
# ruff: noqa: PLR0914, PLR0915, PLR2004
"""Tests for Spec 0019 task consumption manifests."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import pytest

import eqvae.cli.generate_ubc_consumption_manifests as consumption_cli
from eqvae.cli.generate_ubc_consumption_manifests import (
    ASSIGNMENT_HEADER,
    CANCER_HEADER,
    CANCER_INPUT,
    EXPECTED_CANCER,
    EXPECTED_OVERLAP,
    EXPECTED_TISSUE,
    EXPECTED_TISSUE_CLASS,
    EXPECTED_UNION,
    PART_COUNT,
    TISSUE_HEADER,
    TISSUE_INPUT,
    UNION_HEADER,
    Coverage,
    materialize_consumption_manifests,
    midpoint_indices,
    sha256_file,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

EXPECTED_UNION_HASH = "f92558fa7aced13debc839c733e2c96d03c1a3df4b2a0b194b60558c69c04012"
EXPECTED_ASSIGNMENT_HASH = (
    "a5377662911d147a14e1a17714c01b0ed5c6c1f2231429342f1373a1001edeff"
)
SIMULATED_SHARD_FAILURE = "simulated shard publication failure"


def test_midpoint_indices_keep_all_small_groups_and_center_capped_buckets() -> None:
    """Selection is exact, deterministic, unique, and never a raster prefix."""
    assert midpoint_indices(0, 3) == ()
    assert midpoint_indices(3, 3) == (0, 1, 2)
    assert midpoint_indices(10, 4) == (1, 3, 6, 8)
    selected = midpoint_indices(3_001, 3_000)
    assert len(selected) == len(set(selected)) == 3_000
    assert selected[0] == 0
    assert selected[-1] == 3_000


@pytest.mark.parametrize(("size", "cap"), [(-1, 3), (2, 0)])
def test_midpoint_indices_reject_invalid_contract(size: int, cap: int) -> None:
    """Invalid selector inputs fail rather than silently changing population."""
    with pytest.raises(ValueError, match="Invalid midpoint selector"):
        midpoint_indices(size, cap)


def test_empty_coverage_is_a_valid_all_groups_below_cap_audit() -> None:
    """Small fixtures need no invented worst capped group."""
    assert Coverage().audit() == {
        "capped_group_count": 0,
        "macro_mean": None,
        "minimum": None,
        "minimum_numerator": None,
        "minimum_denominator": None,
        "worst_group": None,
    }


@pytest.mark.parametrize("mutation", ["reordered", "duplicate_atlas"])
def test_candidate_manifests_reject_reordered_or_duplicate_identity(
    tmp_path: Path,
    mutation: str,
) -> None:
    """A plausible row count cannot hide broken authoritative identity/order."""
    fixture = _write_small_inputs(tmp_path)
    cancer_rows = _read_csv(fixture.cancer)
    if mutation == "reordered":
        cancer_rows[1], cancer_rows[2] = cancer_rows[2], cancer_rows[1]
        expected = "Manifest order/duplicate failure"
    else:
        cancer_rows[1]["atlas_row_index"] = cancer_rows[0]["atlas_row_index"]
        expected = "Duplicate atlas row"
    _write_dict_rows(fixture.cancer, CANCER_HEADER, cancer_rows)

    with pytest.raises(ValueError, match=expected):
        _materialize_small(fixture, tmp_path / "output")


def test_candidate_split_metadata_cannot_leak_across_wsi_partition(
    tmp_path: Path,
) -> None:
    """Patch rows must inherit the frozen WSI split instead of self-declaring it."""
    fixture = _write_small_inputs(tmp_path)
    rows = _read_csv(fixture.cancer)
    rows[0]["split"] = "test"
    _write_dict_rows(fixture.cancer, CANCER_HEADER, rows)

    with pytest.raises(ValueError, match="Split metadata mismatch for WSI 1"):
        _materialize_small(fixture, tmp_path / "output")


def test_interleaved_tissue_classes_restore_scan_order_without_guiding_cancer(
    tmp_path: Path,
) -> None:
    """Tissue grouping may reorder internally, but cannot alter cancer membership."""
    fixture = _write_small_inputs(tmp_path)
    first_output = tmp_path / "first"
    _materialize_small(fixture, first_output)
    cancer_before = {
        split: (first_output / f"cancer_{split}.csv").read_bytes()
        for split in ("train", "validation", "test")
    }

    tissue_rows = _read_csv(fixture.tissue)
    tissue_rows.insert(
        4,
        _tissue_row(
            atlas_row_index=15,
            wsi_id=1,
            coordinate=(512, 512),
            split="train",
            tissue="stroma",
        ),
    )
    _write_dict_rows(fixture.tissue, TISSUE_HEADER, tissue_rows)
    second_output = tmp_path / "second"
    _materialize_small(fixture, second_output)

    assert {
        split: (second_output / f"cancer_{split}.csv").read_bytes()
        for split in ("train", "validation", "test")
    } == cancer_before
    tissue_train = _read_csv(second_output / "tissue_train.csv")
    keys = [(int(row["wsi_id"]), int(row["y"]), int(row["x"])) for row in tissue_train]
    assert keys == sorted(keys)
    assert [row["tissue_label"] for row in tissue_train if row["wsi_id"] == "1"] == [
        "tumor",
        "stroma",
        "tumor",
        "necrosis",
        "stroma",
    ]


def test_small_materialization_is_byte_identical_when_repeated_in_place(
    tmp_path: Path,
) -> None:
    """A resume or rerun must not silently change any consumption artifact bytes."""
    fixture = _write_small_inputs(tmp_path)
    output_dir = tmp_path / "output"
    assignment = tmp_path / "assignment.csv"
    audit = tmp_path / "audit.json"

    _materialize_small(
        fixture,
        output_dir,
        assignment_path=assignment,
        audit_path=audit,
    )
    first = _artifact_bytes(output_dir, assignment, audit)
    _materialize_small(
        fixture,
        output_dir,
        assignment_path=assignment,
        audit_path=audit,
    )

    assert _artifact_bytes(output_dir, assignment, audit) == first


def test_shard_failure_cannot_leave_an_audit_completion_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The audit must prove every shard exists, so publication happens strictly last."""
    fixture = _write_small_inputs(tmp_path)
    output_dir = tmp_path / "output"
    assignment = tmp_path / "assignment.csv"
    audit = tmp_path / "audit.json"
    audit.write_text("stale completion marker\n", encoding="utf-8")

    def fail_shards(*_args: object, **_kwargs: object) -> list[Path]:
        raise OSError(SIMULATED_SHARD_FAILURE)

    monkeypatch.setattr(consumption_cli, "_write_shards", fail_shards)
    with pytest.raises(OSError, match=SIMULATED_SHARD_FAILURE):
        _materialize_small(
            fixture,
            output_dir,
            assignment_path=assignment,
            audit_path=audit,
        )

    assert not audit.exists()
    assert not audit.with_suffix(".json.tmp").exists()


@pytest.mark.skipif(
    not CANCER_INPUT.is_file() or not TISSUE_INPUT.is_file(),
    reason="verified Spec 0018 full manifests are local-only",
)
def test_real_consumption_materialization_matches_sealed_contract(
    tmp_path: Path,
) -> None:
    """The full candidates reproduce all task, union, ordering, and shard facts."""
    output_dir = tmp_path / "consumption"
    assignment_path = tmp_path / "assignment.csv"
    audit_path = tmp_path / "audit.json"

    audit = materialize_consumption_manifests(
        output_dir=output_dir,
        assignment_path=assignment_path,
        audit_path=audit_path,
    )

    counts = cast("dict[str, object]", audit["counts"])
    assert counts["cancer_by_split"] == EXPECTED_CANCER
    assert counts["tissue_by_split"] == EXPECTED_TISSUE
    assert counts["union_by_split"] == EXPECTED_UNION
    assert counts["overlap"] == EXPECTED_OVERLAP
    assert counts["tissue_by_split_diagnosis_tissue"]
    assert audit["launch_ready"] is False
    assert sha256_file(output_dir / "union_patch_manifest.csv") == EXPECTED_UNION_HASH
    assert sha256_file(assignment_path) == EXPECTED_ASSIGNMENT_HASH

    coverage = cast("dict[str, dict[str, object]]", audit["coverage"])
    assert coverage["cancer"] == {
        "capped_group_count": 146,
        "macro_mean": 0.986048521387179,
        "minimum": 0.9333333333333333,
        "minimum_numerator": 70,
        "minimum_denominator": 75,
        "worst_group": "train:37190",
    }
    assert coverage["tissue"] == {
        "capped_group_count": 143,
        "macro_mean": 0.9781013992306213,
        "minimum": 0.9180327868852459,
        "minimum_numerator": 56,
        "minimum_denominator": 61,
        "worst_group": "train:45185:tumor",
    }

    with assignment_path.open(encoding="utf-8", newline="") as handle:
        assignment_rows = list(csv.DictReader(handle))
    assert tuple(assignment_rows[0]) == ASSIGNMENT_HEADER
    assert len(assignment_rows) == 152
    assert {int(cast("str", row["run_number"])) for row in assignment_rows} == set(
        range(1, PART_COUNT + 1),
    )
    assert len({cast("str", row["wsi_id"]) for row in assignment_rows}) == 152

    union_rows = 0
    previous: tuple[int, int, int] | None = None
    with (output_dir / "union_patch_manifest.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == UNION_HEADER
        for row in reader:
            key = (
                int(cast("str", row["wsi_id"])),
                int(cast("str", row["y"])),
                int(cast("str", row["x"])),
            )
            assert previous is None or previous < key
            assert (
                row["cancer_ae_selected"] == "true" or row["tissue_selected"] == "true"
            )
            if row["cancer_ae_selected"] == "false":
                assert row["tissue_selected"] == "true"
                assert row["tissue_label"] in {"tumor", "stroma", "necrosis"}
            previous = key
            union_rows += 1
    assert union_rows == sum(EXPECTED_UNION.values())

    shard_rows = 0
    shard_wsis: set[str] = set()
    with (output_dir / "union_patch_manifest.csv").open(
        encoding="utf-8",
        newline="",
    ) as union_handle:
        union_iterator = iter(csv.DictReader(union_handle))
        for run in range(1, PART_COUNT + 1):
            shard_path = output_dir / "work_shards" / f"run_{run:02d}_of_05.csv"
            current_wsis: set[str] = set()
            with shard_path.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    assert row == next(union_iterator)
                    current_wsis.add(cast("str", row["wsi_id"]))
                    shard_rows += 1
            assert shard_wsis.isdisjoint(current_wsis)
            shard_wsis.update(current_wsis)
        with pytest.raises(StopIteration):
            next(union_iterator)
    assert shard_rows == union_rows
    assert len(shard_wsis) == 152
    stored = cast(
        "dict[str, object]",
        json.loads(audit_path.read_text(encoding="utf-8")),
    )
    assert stored == audit


def test_expected_tissue_class_totals_match_split_totals() -> None:
    """The locked class table and task totals cannot drift independently."""
    assert {
        split: sum(EXPECTED_TISSUE_CLASS[split].values()) for split in EXPECTED_TISSUE
    } == EXPECTED_TISSUE


@dataclass(frozen=True)
class FixturePaths:
    """Paths belonging to one isolated five-WSI consumption fixture."""

    cancer: Path
    tissue: Path
    split: Path


def _write_small_inputs(root: Path) -> FixturePaths:
    """Write the smallest fixture that can still exercise all five work shards.

    Returns:
        Paths to the three canonical fixture inputs.

    """
    cancer = root / "cancer_input.csv"
    tissue = root / "tissue_input.csv"
    split = root / "split.csv"
    splits = {1: "train", 2: "train", 3: "validation", 4: "test", 5: "test"}
    _write_dict_rows(
        cancer,
        CANCER_HEADER,
        [
            _cancer_row(
                atlas_row_index=wsi_id * 10,
                wsi_id=wsi_id,
                x=0,
                y=0,
                split=splits[wsi_id],
            )
            for wsi_id in range(1, 6)
        ],
    )
    tissue_rows = [
        _tissue_row(11, 1, (0, 256), "train", "tumor"),
        _tissue_row(12, 1, (256, 256), "train", "stroma"),
        _tissue_row(13, 1, (0, 512), "train", "tumor"),
        _tissue_row(14, 1, (256, 512), "train", "necrosis"),
        *[
            _tissue_row(
                wsi_id * 10,
                wsi_id,
                (0, 0),
                splits[wsi_id],
                "tumor",
            )
            for wsi_id in range(2, 6)
        ],
    ]
    _write_dict_rows(tissue, TISSUE_HEADER, tissue_rows)
    with split.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow((
            "wsi_id",
            "diagnosis_label",
            "diagnosis_index",
            "is_updated_image_id",
            "split",
        ))
        for wsi_id in range(1, 153):
            writer.writerow((wsi_id, "CC", 0, "false", splits.get(wsi_id, "train")))
    return FixturePaths(cancer=cancer, tissue=tissue, split=split)


def _cancer_row(
    atlas_row_index: int,
    wsi_id: int,
    x: int,
    y: int,
    split: str,
) -> dict[str, str]:
    """Build a mask-independent candidate row with canonical diagnosis metadata.

    Returns:
        One complete cancer-manifest row.

    """
    return {
        "atlas_row_index": str(atlas_row_index),
        "wsi_id": str(wsi_id),
        "diagnosis_label": "CC",
        "diagnosis_index": "0",
        "x": str(x),
        "y": str(y),
        "split": split,
    }


def _tissue_row(
    atlas_row_index: int,
    wsi_id: int,
    coordinate: tuple[int, int],
    split: str,
    tissue: str,
) -> dict[str, str]:
    """Build a pure annotated row so grouping, not thresholding, is under test.

    Returns:
        One complete tissue-manifest row.

    """
    x, y = coordinate
    row = _cancer_row(atlas_row_index, wsi_id, x, y, split)
    fractions = {
        "tumor": ("1", "0", "0"),
        "stroma": ("0", "1", "0"),
        "necrosis": ("0", "0", "1"),
    }[tissue]
    row.update({
        "tissue_label": tissue,
        "annotated_fraction": "1",
        "dominant_fraction": "1",
        "purity": "1",
        "tumor_fraction": fractions[0],
        "stroma_fraction": fractions[1],
        "necrosis_fraction": fractions[2],
    })
    return row


def _write_dict_rows(
    path: Path,
    header: tuple[str, ...],
    rows: Sequence[Mapping[str, str]],
) -> None:
    """Write exact LF CSV bytes so byte-determinism assertions are meaningful."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Load a tiny fixture or output without obscuring its named fields.

    Returns:
        Every data row as a string-valued mapping.

    """
    with path.open(encoding="utf-8", newline="") as handle:
        return [
            {key: cast("str", value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def _materialize_small(
    fixture: FixturePaths,
    output_dir: Path,
    *,
    assignment_path: Path | None = None,
    audit_path: Path | None = None,
) -> dict[str, object]:
    """Run real selection logic while disabling only full-cohort count seals.

    Returns:
        The completion audit produced by the real materializer.

    """
    assignment = (
        assignment_path or output_dir.parent / f"{output_dir.name}_assignment.csv"
    )
    audit = audit_path or output_dir.parent / f"{output_dir.name}_audit.json"
    return materialize_consumption_manifests(
        cancer_input=fixture.cancer,
        tissue_input=fixture.tissue,
        split_path=fixture.split,
        output_dir=output_dir,
        assignment_path=assignment,
        audit_path=audit,
        expected_cancer_hash=None,
        expected_tissue_hash=None,
        expected_split_hash=None,
        enforce_real_counts=False,
    )


def _artifact_bytes(
    output_dir: Path,
    assignment_path: Path,
    audit_path: Path,
) -> dict[str, bytes]:
    """Capture every published byte so determinism includes shards and completion.

    Returns:
        Published artifacts keyed by stable fixture-relative names.

    """
    result = {
        str(path.relative_to(output_dir)): path.read_bytes()
        for path in sorted(output_dir.rglob("*"))
        if path.is_file()
    }
    result["../assignment.csv"] = assignment_path.read_bytes()
    result["../audit.json"] = audit_path.read_bytes()
    return result
