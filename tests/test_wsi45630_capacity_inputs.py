# Copyright 2026 HiperMaximus
# ruff: noqa: TC003
# pyright: reportPrivateUsage=false
"""Focused invariants for the full-coverage WSI45630 capacity input package."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from scripts import build_wsi45630_capacity as builder

from eqvae.data.latent_shards import UNION_HEADER


def test_physical_file_indices_survive_wsi_filtering(tmp_path: Path) -> None:
    """Keep original file offsets after filtering so bytes map to their real records.

    This DERIVED pointer invariant catches the subtle regression where filtering first
    renumbers WSI45630 rows and silently reads the wrong payload vectors.
    """
    manifest = tmp_path / "part4.csv"
    _write_csv(
        manifest,
        UNION_HEADER,
        [
            (0, 9, "CC", 0, 0, 0, "train", "true", "false", ""),
            (1, builder.WSI_ID, "EC", 1, 0, 0, "train", "true", "false", ""),
            (2, builder.WSI_ID, "EC", 1, 256, 0, "train", "true", "false", ""),
            (3, 10, "CC", 0, 0, 0, "train", "true", "false", ""),
        ],
    )

    rows = builder._physical_rows(manifest, expected_header=UNION_HEADER)  # noqa: SLF001

    assert rows == {
        (1, builder.WSI_ID, 0, 0): 1,
        (2, builder.WSI_ID, 256, 0): 2,
    }


def test_otsu_sources_must_be_disjoint_and_cover_the_candidate_set() -> None:
    """Bind every Otsu candidate once, preventing gaps or duplicate full-bag patches.

    This DERIVED coverage invariant protects the requested 32,595-patch memory
    measurement: any omission or overlap changes the capacity premise.
    """
    candidates = {
        (0, builder.WSI_ID, 0, 0): None,
        (1, builder.WSI_ID, 256, 0): None,
        (2, builder.WSI_ID, 0, 256): None,
    }
    pointers, ordered = builder._bind_pointers(  # noqa: SLF001
        candidates,
        {
            12: {(2, builder.WSI_ID, 0, 256): 0},
            4: {(0, builder.WSI_ID, 0, 0): 9},
            11: {(1, builder.WSI_ID, 256, 0): 4},
        },
    )

    assert ordered == [
        (0, builder.WSI_ID, 0, 0),
        (1, builder.WSI_ID, 256, 0),
        (2, builder.WSI_ID, 0, 256),
    ]
    assert pointers[2, builder.WSI_ID, 0, 256] == (12, 0)
    with pytest.raises(ValueError, match="exactly cover"):
        builder._bind_pointers(candidates, {4: {(0, builder.WSI_ID, 0, 0): 9}})  # noqa: SLF001
    with pytest.raises(ValueError, match="overlap"):
        builder._bind_pointers(  # noqa: SLF001
            candidates,
            {4: {(0, builder.WSI_ID, 0, 0): 9}, 11: {(0, builder.WSI_ID, 0, 0): 4}},
        )


def _write_csv(
    path: Path,
    header: tuple[str, ...],
    rows: list[tuple[object, ...]],
) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
