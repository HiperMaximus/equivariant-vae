# Copyright 2026 HiperMaximus
# ruff: noqa: D103, PLC2701, PLR2004, TC003
# pyright: reportPrivateUsage=false
"""Focused tests for the compact supervised logical-manifest builder."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from eqvae.cli.build_ubc_cancer_topup import BAG_HEADER, INSTANCE_HEADER
from eqvae.cli.build_ubc_supervised_manifests import (
    CATALOG_HEADER,
    TISSUE_HEADER,
    WSI_INSTANCE_HEADER,
    TissueRow,
    _catalog_rows,
    _write_tissue_files,
    _write_wsi_files,
)


def _artifact(model: str, part: int, rows: int) -> dict[str, object]:
    return {
        "binary": {
            "name": f"{model}_{part}.bin",
            "bytes": 64 + rows * 65_536,
            "sha256": f"{part:x}" * 64,
        },
        "sidecar": {
            "name": f"{model}_{part}.json",
            "bytes": 100 + part,
            "sha256": f"{part + 5:x}" * 64,
        },
    }


def _catalog_inputs() -> tuple[dict[str, object], dict[str, object]]:
    base: dict[str, object] = {
        "artifacts": {
            model: {
                str(part): _artifact(model, part, 20 + part) for part in range(1, 6)
            }
            for model in ("normal_vae", "so2_vae")
        },
    }
    pair: dict[str, object] = {
        "row_count": 7,
        "artifacts": {
            model: {
                "bin_name": f"{model}_topup.bin",
                "bin_bytes": 64 + 7 * 65_536,
                "bin_sha256": "a" * 64,
                "sidecar_name": f"{model}_topup.json",
                "sidecar_bytes": 111,
                "sidecar_sha256": "b" * 64,
            }
            for model in ("normal_vae", "so2_vae")
        },
    }
    return base, pair


def _write_csv(
    path: Path,
    header: tuple[str, ...],
    rows: list[tuple[object, ...]],
) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def test_catalog_has_six_paired_parts_and_ordinary_part_11() -> None:
    base, pair = _catalog_inputs()
    rows, counts = _catalog_rows(base, pair)

    assert len(rows) == 12
    assert set(counts) == {1, 2, 3, 4, 5, 11}
    assert {(row[0], row[1]) for row in rows} == {
        (part, model)
        for part in (1, 2, 3, 4, 5, 11)
        for model in ("normal_vae", "so2_vae")
    }
    by_part_model = {(row[0], row[1]): row for row in rows}
    for part in counts:
        assert by_part_model[part, "normal_vae"][4] == by_part_model[part, "so2_vae"][4]
    assert all(len(row) == len(CATALOG_HEADER) for row in rows)


def test_wsi_rows_convert_exact_sealed_pointers_to_parts(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    for split in ("train", "validation", "test"):
        instances: list[tuple[object, ...]] = []
        bags: list[tuple[object, ...]] = []
        if split == "train":
            instances = [
                (0, 10, 100, 2, 1, "HGSC", 2, split, "base", 1, 4),
                (1, 11, 100, 3, 1, "HGSC", 2, split, "topup", 1, 0),
            ]
            bags = [(0, 100, "HGSC", 2, split, 0, 2)]
        _write_csv(
            source / f"wsi_cancer_{split}_instances.csv",
            INSTANCE_HEADER,
            instances,
        )
        _write_csv(source / f"wsi_cancer_{split}_bags.csv", BAG_HEADER, bags)

    records, counts, splits = _write_wsi_files(
        source,
        output,
        {1: 10, 11: 2},
        base_pointers={(10, 100, 2, 1): (1, 4)},
        topup_pointers={(11, 100, 3, 1): 0},
    )

    with (output / "wsi_cancer_train_instances.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert tuple(rows[0]) == WSI_INSTANCE_HEADER
    assert [(row["part"], row["file_index"]) for row in rows] == [
        ("1", "4"),
        ("11", "0"),
    ]
    assert counts == {"train": 2, "validation": 0, "test": 0}
    assert splits == {100: "train"}
    assert len(records) == 6


def test_wsi_builder_rejects_merely_in_range_but_wrong_base_pointer(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    for split in ("train", "validation", "test"):
        instances: list[tuple[object, ...]] = []
        bags: list[tuple[object, ...]] = []
        if split == "train":
            instances = [(0, 10, 100, 2, 1, "HGSC", 2, split, "base", 1, 4)]
            bags = [(0, 100, "HGSC", 2, split, 0, 1)]
        _write_csv(
            source / f"wsi_cancer_{split}_instances.csv",
            INSTANCE_HEADER,
            instances,
        )
        _write_csv(source / f"wsi_cancer_{split}_bags.csv", BAG_HEADER, bags)

    with pytest.raises(ValueError, match="base pointer differs"):
        _write_wsi_files(
            source,
            output,
            {1: 10},
            base_pointers={(10, 100, 2, 1): (1, 5)},
            topup_pointers={},
        )


def test_tissue_training_files_are_nested_and_share_physical_pointers(
    tmp_path: Path,
) -> None:
    rows: dict[str, list[TissueRow]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    atlas = 0
    for tissue_index, tissue in enumerate(("tumor", "stroma", "necrosis")):
        for offset in range(3):
            rows["train"].append(
                TissueRow(
                    atlas_row_index=atlas,
                    wsi_id=100 + tissue_index,
                    x=offset,
                    y=tissue_index,
                    tissue_label=tissue,
                    split="train",
                    part=1 + tissue_index,
                    file_index=offset,
                ),
            )
            atlas += 1
        for split in ("validation", "test"):
            rows[split].append(
                TissueRow(
                    atlas_row_index=atlas,
                    wsi_id=200 + tissue_index + (100 if split == "test" else 0),
                    x=0,
                    y=0,
                    tissue_label=tissue,
                    split=split,
                    part=1,
                    file_index=atlas,
                ),
            )
            atlas += 1
    output = tmp_path / "tissue"
    output.mkdir()

    records, _summary = _write_tissue_files(rows, output, tissue_sizes=(1, 2))

    with (output / "tissue_train_0001_per_class.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        small = list(csv.DictReader(handle))
    with (output / "tissue_train_0002_per_class.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        large = list(csv.DictReader(handle))
    small_ids = {row["atlas_row_index"] for row in small}
    large_ids = {row["atlas_row_index"] for row in large}
    assert small_ids < large_ids
    assert tuple(small[0]) == TISSUE_HEADER
    assert {int(row["part"]) for row in large} <= {1, 2, 3, 4, 5}
    assert len(records) == 4
