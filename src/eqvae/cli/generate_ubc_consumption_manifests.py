# Copyright 2026 HiperMaximus
# ruff: noqa: DOC201, DOC501, EM101, EM102, PLR0914, PLR2004, PT018, S101, T201, TRY003
# pyright: reportAny=false, reportArgumentType=false, reportMissingTypeArgument=false
# pyright: reportReturnType=false, reportUnknownMemberType=false
# pyright: reportUnknownParameterType=false, reportUnknownVariableType=false
# pyright: reportUnnecessaryCast=false
"""Select final UBC evaluation patches and make whole-WSI work shards."""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

from eqvae.cli.generate_ubc_eval_manifests import (
    CANCER_HEADER,
    SPLIT_PATH,
    SPLIT_SHA256,
    TISSUE_HEADER,
    sha256_file,
    validate_pinned_hash,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

CANCER_INPUT: Final = Path(
    "runs/local/ubc_ocean_eval_manifests/cancer_ae_patch_manifest.csv",
)
TISSUE_INPUT: Final = Path(
    "runs/local/ubc_ocean_eval_manifests/tissue_patch_manifest.csv",
)
OUTPUT_DIR: Final = Path("runs/local/ubc_ocean_eval_consumption")
ASSIGNMENT_PATH: Final = Path("docs/data/ubc_ocean_eval_work_assignment.csv")
AUDIT_PATH: Final = Path("docs/data/ubc_ocean_eval_consumption_audit.json")
GENERATOR_PATH: Final = Path(
    "src/eqvae/cli/generate_ubc_consumption_manifests.py",
)
CANCER_INPUT_SHA256: Final = (
    "710c3f8166f577f5ae60bec94a544dabed9619342a46b82374a2e739162d648c"
)
TISSUE_INPUT_SHA256: Final = (
    "7e2f61c492167129911e73515c4d82ef9291fd89b920dc191be922a44f30ac92"
)
CANCER_CAP: Final = 3_000
TISSUE_CAP: Final = 1_000
PART_COUNT: Final = 5
PATCH_BYTES: Final = 3 * 256 * 256
SPLITS: Final = ("train", "validation", "test")
TISSUES: Final = ("tumor", "stroma", "necrosis")
UNION_HEADER: Final = (
    "atlas_row_index",
    "wsi_id",
    "diagnosis_label",
    "diagnosis_index",
    "x",
    "y",
    "split",
    "cancer_ae_selected",
    "tissue_selected",
    "tissue_label",
)
ASSIGNMENT_HEADER: Final = (
    "run_number",
    "wsi_id",
    "split",
    "diagnosis_label",
    "diagnosis_index",
    "cancer_ae_patch_count",
    "tissue_patch_count",
    "union_patch_count",
    "min_x",
    "max_x",
    "min_y",
    "max_y",
    "projected_raw_bytes",
)
EXPECTED_CANCER = {"train": 314_755, "validation": 68_045, "test": 67_138}
EXPECTED_TISSUE = {"train": 145_215, "validation": 31_339, "test": 31_572}
EXPECTED_TISSUE_CLASS = {
    "train": {"tumor": 102_188, "stroma": 37_356, "necrosis": 5_671},
    "validation": {"tumor": 22_255, "stroma": 7_809, "necrosis": 1_275},
    "test": {"tumor": 21_796, "stroma": 8_500, "necrosis": 1_276},
}
EXPECTED_UNION = {"train": 418_685, "validation": 90_311, "test": 90_402}
EXPECTED_OVERLAP: Final = 58_666


@dataclass(frozen=True)
class SelectedRow:
    """One validated selected manifest row."""

    values: dict[str, str]
    key: tuple[int, int, int]
    atlas_index: int


@dataclass
class WsiWork:
    """Counts and selected coordinate bounds for one WSI."""

    split: str
    diagnosis_label: str
    diagnosis_index: int
    cancer: int = 0
    tissue: int = 0
    union: int = 0
    min_x: int = sys.maxsize
    max_x: int = -1
    min_y: int = sys.maxsize
    max_y: int = -1


@dataclass
class Coverage:
    """Per-capped-group occupied-cell coverage evidence."""

    values: list[tuple[float, str, int, int]] = field(default_factory=list)

    def add(
        self,
        candidates: Sequence[SelectedRow],
        selected: Sequence[SelectedRow],
        name: str,
    ) -> None:
        """Record exact 10x10 candidate-cell retention for one capped group."""
        xs = [row.key[2] for row in candidates]
        ys = [row.key[1] for row in candidates]
        xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
        dx, dy = max(1, xmax - xmin + 1), max(1, ymax - ymin + 1)

        def cell(row: SelectedRow) -> tuple[int, int]:
            x, y = row.key[2], row.key[1]
            return (
                min(9, ((x - xmin) * 10) // dx),
                min(9, ((y - ymin) * 10) // dy),
            )

        occupied = {cell(row) for row in candidates}
        retained = {cell(row) for row in selected}
        self.values.append((
            len(retained) / len(occupied),
            name,
            len(retained),
            len(occupied),
        ))

    def audit(self) -> dict[str, object]:
        """Return macro and worst-group coverage summaries."""
        if not self.values:
            return {
                "capped_group_count": 0,
                "macro_mean": None,
                "minimum": None,
                "minimum_numerator": None,
                "minimum_denominator": None,
                "worst_group": None,
            }
        worst = min(self.values)
        return {
            "capped_group_count": len(self.values),
            "macro_mean": sum(value for value, _name, _n, _d in self.values)
            / len(self.values),
            "minimum": worst[0],
            "minimum_numerator": worst[2],
            "minimum_denominator": worst[3],
            "worst_group": worst[1],
        }


def midpoint_indices(size: int, cap: int) -> tuple[int, ...]:
    """Return centered systematic ranks, or every rank below the cap."""
    if size < 0 or cap < 1:
        message = f"Invalid midpoint selector size={size}, cap={cap}"
        raise ValueError(message)
    if size <= cap:
        return tuple(range(size))
    return tuple(((2 * index + 1) * size) // (2 * cap) for index in range(cap))


def materialize_consumption_manifests(  # noqa: PLR0913
    *,
    cancer_input: Path = CANCER_INPUT,
    tissue_input: Path = TISSUE_INPUT,
    split_path: Path = SPLIT_PATH,
    output_dir: Path = OUTPUT_DIR,
    assignment_path: Path = ASSIGNMENT_PATH,
    audit_path: Path = AUDIT_PATH,
    expected_cancer_hash: str | None = CANCER_INPUT_SHA256,
    expected_tissue_hash: str | None = TISSUE_INPUT_SHA256,
    expected_split_hash: str | None = SPLIT_SHA256,
    enforce_real_counts: bool = True,
) -> dict[str, object]:
    """Create task sets, their physical union, five shards, and audit last."""
    input_hashes = {
        "cancer": sha256_file(cancer_input),
        "tissue": sha256_file(tissue_input),
        "split": sha256_file(split_path),
    }
    for path, expected, name in (
        (cancer_input, expected_cancer_hash, "Cancer manifest"),
        (tissue_input, expected_tissue_hash, "Tissue manifest"),
        (split_path, expected_split_hash, "Canonical split"),
    ):
        if expected is not None:
            validate_pinned_hash(path, expected, name)
    split_metadata = _load_split_metadata(split_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    assignment_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.unlink(missing_ok=True)
    temporary_root = output_dir / ".building"
    if temporary_root.exists():
        shutil.rmtree(temporary_root)
    temporary_root.mkdir()
    try:
        cancer_counts, cancer_cross, cancer_coverage = _select_cancer(
            cancer_input,
            temporary_root,
            split_metadata,
        )
        tissue_counts, tissue_cross, tissue_coverage = _select_tissue(
            tissue_input,
            temporary_root,
            split_metadata,
        )
        _publish_task_files(temporary_root, output_dir)
        union_path = output_dir / "union_patch_manifest.csv"
        work, union_counts, overlap = _write_union(output_dir, union_path)
        plans = _plan_parts(work, PART_COUNT)
        assignment_tmp = assignment_path.with_suffix(".csv.tmp")
        _write_assignment(assignment_tmp, work, plans)
        assignment_tmp.replace(assignment_path)
        shard_paths = _write_shards(output_dir, union_path, plans)
        if enforce_real_counts:
            _validate_real_counts(
                cancer_counts,
                tissue_counts,
                tissue_cross,
                union_counts,
                overlap,
            )
        output_paths = [
            *(output_dir / f"cancer_{split}.csv" for split in SPLITS),
            *(output_dir / f"tissue_{split}.csv" for split in SPLITS),
            union_path,
            assignment_path,
            *shard_paths,
        ]
        audit = {
            "schema_version": "spec0019.ubc_consumption.v1",
            "status": "pass",
            "launch_ready": False,
            "launch_blockers": [
                "publish immutable private Kaggle inputs",
                "complete the guarded dual-T4 inference kernel and resource audit",
                "pass worst-case WSI runtime/output pilot",
                "generate and globally validate both five-shard latent stores",
            ],
            "selector": {
                "formula": "((2*j + 1) * N) // (2*C)",
                "cancer_cap_per_wsi": CANCER_CAP,
                "tissue_cap_per_wsi_class": TISSUE_CAP,
                "order": "wsi_id,y,x",
            },
            "inputs": input_hashes,
            "counts": {
                "cancer_by_split": cancer_counts,
                "cancer_by_split_diagnosis": cancer_cross,
                "tissue_by_split": tissue_counts,
                "tissue_by_split_diagnosis_tissue": tissue_cross,
                "union_by_split": union_counts,
                "overlap": overlap,
                "union_total": sum(union_counts.values()),
            },
            "coverage": {
                "cancer": cancer_coverage.audit(),
                "tissue": tissue_coverage.audit(),
            },
            "work_shards": _plan_audit(work, plans),
            "projected_raw_bytes": sum(value.union for value in work.values())
            * PATCH_BYTES,
            "outputs": {str(path): sha256_file(path) for path in output_paths},
            "generator": {
                "path": str(GENERATOR_PATH),
                "sha256": sha256_file(Path(__file__)),
            },
            "acceptance": {
                "exact_caps_and_counts": enforce_real_counts,
                "independent_task_selection": True,
                "wsi_id_y_x_order": True,
                "whole_wsi_shards": True,
                "audit_published_last": True,
                "raw_patch_output_forbidden": True,
            },
        }
        audit_tmp = audit_path.with_suffix(".json.tmp")
        _write_json(audit_tmp, audit)
        audit_tmp.replace(audit_path)
        return audit
    finally:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)


def _load_split_metadata(path: Path) -> dict[int, tuple[str, str, int]]:
    rows: dict[int, tuple[str, str, int]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            wsi_id = int(cast("str", row["wsi_id"]))
            rows[wsi_id] = (
                cast("str", row["split"]),
                cast("str", row["diagnosis_label"]),
                int(cast("str", row["diagnosis_index"])),
            )
    if len(rows) != 152:
        message = f"Expected 152 split WSIs, found {len(rows)}"
        raise ValueError(message)
    return rows


def _iter_groups(
    path: Path,
    header: tuple[str, ...],
) -> Iterator[tuple[int, list[SelectedRow]]]:
    previous: tuple[int, int, int] | None = None
    current_id: int | None = None
    group: list[SelectedRow] = []
    seen_atlas: set[int] = set()
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != header:
            message = f"Unexpected header in {path}: {reader.fieldnames!r}"
            raise ValueError(message)
        for raw in reader:
            values = {key: cast("str", value) for key, value in raw.items()}
            wsi_id, y, x = int(values["wsi_id"]), int(values["y"]), int(values["x"])
            key = (wsi_id, y, x)
            atlas_index = int(values["atlas_row_index"])
            if previous is not None and key <= previous:
                raise ValueError(f"Manifest order/duplicate failure at {key}")
            if atlas_index in seen_atlas:
                raise ValueError(f"Duplicate atlas row {atlas_index}")
            seen_atlas.add(atlas_index)
            if current_id is not None and wsi_id != current_id:
                yield current_id, group
                group = []
            current_id = wsi_id
            group.append(SelectedRow(values, key, atlas_index))
            previous = key
    if current_id is not None:
        yield current_id, group


def _validate_row(
    row: SelectedRow,
    metadata: Mapping[int, tuple[str, str, int]],
) -> None:
    expected = metadata.get(row.key[0])
    observed = (
        row.values["split"],
        row.values["diagnosis_label"],
        int(row.values["diagnosis_index"]),
    )
    if expected != observed:
        raise ValueError(f"Split metadata mismatch for WSI {row.key[0]}")


def _selected(rows: Sequence[SelectedRow], cap: int) -> list[SelectedRow]:
    return [rows[index] for index in midpoint_indices(len(rows), cap)]


def _open_task_writers(
    root: Path,
    task: str,
    header: tuple[str, ...],
) -> tuple[dict[str, object], dict[str, csv.DictWriter]]:
    handles: dict[str, object] = {}
    writers: dict[str, csv.DictWriter] = {}
    for split in SPLITS:
        handle = (root / f"{task}_{split}.csv").open("w", encoding="utf-8", newline="")
        writer = csv.DictWriter(handle, fieldnames=header, lineterminator="\n")
        writer.writeheader()
        handles[split], writers[split] = handle, writer
    return handles, writers


def _close_handles(handles: Mapping[str, object]) -> None:
    for raw_handle in handles.values():
        handle = cast("object", raw_handle)
        handle.flush()  # type: ignore[attr-defined]
        os.fsync(handle.fileno())  # type: ignore[attr-defined]
        handle.close()  # type: ignore[attr-defined]


def _select_cancer(
    path: Path,
    root: Path,
    metadata: Mapping[int, tuple[str, str, int]],
) -> tuple[dict[str, int], dict[str, dict[str, int]], Coverage]:
    handles, writers = _open_task_writers(root, "cancer", CANCER_HEADER)
    counts: Counter[str] = Counter()
    cross: dict[str, Counter[str]] = defaultdict(Counter)
    coverage = Coverage()
    try:
        for wsi_id, rows in _iter_groups(path, CANCER_HEADER):
            for row in rows:
                _validate_row(row, metadata)
            chosen = _selected(rows, CANCER_CAP)
            if len(rows) > CANCER_CAP:
                coverage.add(rows, chosen, f"{rows[0].values['split']}:{wsi_id}")
            for row in chosen:
                split = row.values["split"]
                writers[split].writerow(row.values)
                counts[split] += 1
                cross[split][row.values["diagnosis_label"]] += 1
    finally:
        _close_handles(handles)
    return dict(counts), {key: dict(value) for key, value in cross.items()}, coverage


def _select_tissue(
    path: Path,
    root: Path,
    metadata: Mapping[int, tuple[str, str, int]],
) -> tuple[dict[str, int], dict[str, dict[str, dict[str, int]]], Coverage]:
    handles, writers = _open_task_writers(root, "tissue", TISSUE_HEADER)
    counts: Counter[str] = Counter()
    cross: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter),
    )
    coverage = Coverage()
    try:
        for wsi_id, rows in _iter_groups(path, TISSUE_HEADER):
            by_tissue: dict[str, list[SelectedRow]] = defaultdict(list)
            for row in rows:
                _validate_row(row, metadata)
                tissue = row.values["tissue_label"]
                if tissue not in TISSUES:
                    raise ValueError(f"Invalid tissue label {tissue!r}")
                by_tissue[tissue].append(row)
            chosen: list[SelectedRow] = []
            for tissue in TISSUES:
                candidates = by_tissue[tissue]
                selected = _selected(candidates, TISSUE_CAP)
                if len(candidates) > TISSUE_CAP:
                    coverage.add(
                        candidates,
                        selected,
                        f"{rows[0].values['split']}:{wsi_id}:{tissue}",
                    )
                chosen.extend(selected)
            for row in sorted(chosen, key=lambda value: value.key):
                split = row.values["split"]
                writers[split].writerow(row.values)
                counts[split] += 1
                cross[split][row.values["diagnosis_label"]][
                    row.values["tissue_label"]
                ] += 1
    finally:
        _close_handles(handles)
    return (
        dict(counts),
        {s: {d: dict(c) for d, c in values.items()} for s, values in cross.items()},
        coverage,
    )


def _publish_task_files(source: Path, destination: Path) -> None:
    for task in ("cancer", "tissue"):
        for split in SPLITS:
            (source / f"{task}_{split}.csv").replace(
                destination / f"{task}_{split}.csv",
            )


def _merged_task_rows(
    output_dir: Path,
    task: str,
    header: tuple[str, ...],
) -> Iterator[SelectedRow]:
    iterators = [
        _iter_single_file(output_dir / f"{task}_{split}.csv", header)
        for split in SPLITS
    ]
    yield from heapq.merge(*iterators, key=lambda row: row.key)


def _iter_single_file(path: Path, header: tuple[str, ...]) -> Iterator[SelectedRow]:
    for _wsi_id, rows in _iter_groups(path, header):
        yield from rows


def _write_union(
    output_dir: Path,
    path: Path,
) -> tuple[dict[int, WsiWork], dict[str, int], int]:
    cancer = iter(_merged_task_rows(output_dir, "cancer", CANCER_HEADER))
    tissue = iter(_merged_task_rows(output_dir, "tissue", TISSUE_HEADER))
    left, right = next(cancer, None), next(tissue, None)
    work: dict[int, WsiWork] = {}
    counts: Counter[str] = Counter()
    overlap = 0
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=UNION_HEADER, lineterminator="\n")
        writer.writeheader()
        while left is not None or right is not None:
            if right is None or (left is not None and left.key < right.key):
                row, cancer_flag, tissue_flag, tissue_label = left, True, False, ""
                left = next(cancer, None)
            elif left is None or right.key < left.key:
                row, cancer_flag, tissue_flag = right, False, True
                tissue_label = cast("SelectedRow", right).values["tissue_label"]
                right = next(tissue, None)
            else:
                assert left is not None and right is not None
                if left.atlas_index != right.atlas_index:
                    raise ValueError("Task coordinate identity disagrees")
                row, cancer_flag, tissue_flag = left, True, True
                tissue_label = right.values["tissue_label"]
                left, right = next(cancer, None), next(tissue, None)
                overlap += 1
            assert row is not None
            values = row.values
            writer.writerow({
                "atlas_row_index": row.atlas_index,
                "wsi_id": row.key[0],
                "diagnosis_label": values["diagnosis_label"],
                "diagnosis_index": values["diagnosis_index"],
                "x": row.key[2],
                "y": row.key[1],
                "split": values["split"],
                "cancer_ae_selected": str(cancer_flag).lower(),
                "tissue_selected": str(tissue_flag).lower(),
                "tissue_label": tissue_label,
            })
            wsi_id, y, x = row.key
            item = work.setdefault(
                wsi_id,
                WsiWork(
                    values["split"],
                    values["diagnosis_label"],
                    int(values["diagnosis_index"]),
                ),
            )
            item.cancer += int(cancer_flag)
            item.tissue += int(tissue_flag)
            item.union += 1
            item.min_x = min(item.min_x, x)
            item.max_x = max(item.max_x, x)
            item.min_y = min(item.min_y, y)
            item.max_y = max(item.max_y, y)
            counts[values["split"]] += 1
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)
    return work, dict(counts), overlap


def _plan_parts(work: Mapping[int, WsiWork], part_count: int) -> list[tuple[int, ...]]:
    items = [(wsi_id, work[wsi_id].union) for wsi_id in sorted(work)]
    plans: list[tuple[int, ...]] = []
    next_wsi, remaining = 0, sum(count for _wsi, count in items)
    for number in range(1, part_count + 1):
        remaining_parts = part_count - number + 1
        target = remaining / remaining_parts
        max_end = len(items) - (remaining_parts - 1)
        selected: list[int] = []
        selected_count = 0
        while next_wsi < max_end:
            wsi_id, count = items[next_wsi]
            if selected and abs(selected_count - target) <= abs(
                selected_count + count - target,
            ):
                break
            selected.append(wsi_id)
            selected_count += count
            next_wsi += 1
        if not selected:
            wsi_id, count = items[next_wsi]
            selected = [wsi_id]
            selected_count = count
            next_wsi += 1
        plans.append(tuple(selected))
        remaining -= selected_count
    return plans


def _write_assignment(
    path: Path,
    work: Mapping[int, WsiWork],
    plans: Sequence[tuple[int, ...]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(ASSIGNMENT_HEADER)
        for run_number, ids in enumerate(plans, start=1):
            for wsi_id in ids:
                value = work[wsi_id]
                writer.writerow((
                    run_number,
                    wsi_id,
                    value.split,
                    value.diagnosis_label,
                    value.diagnosis_index,
                    value.cancer,
                    value.tissue,
                    value.union,
                    value.min_x,
                    value.max_x,
                    value.min_y,
                    value.max_y,
                    value.union * PATCH_BYTES,
                ))
        handle.flush()
        os.fsync(handle.fileno())


def _write_shards(
    output_dir: Path,
    union_path: Path,
    plans: Sequence[tuple[int, ...]],
) -> list[Path]:
    shard_dir = output_dir / "work_shards"
    shard_dir.mkdir(exist_ok=True)
    run_by_wsi = {wsi: run for run, ids in enumerate(plans, start=1) for wsi in ids}
    paths = [shard_dir / f"run_{run:02d}_of_05.csv" for run in range(1, 6)]
    handles = [
        path.with_suffix(".csv.tmp").open("w", encoding="utf-8", newline="")
        for path in paths
    ]
    writers = [
        csv.DictWriter(handle, fieldnames=UNION_HEADER, lineterminator="\n")
        for handle in handles
    ]
    for writer in writers:
        writer.writeheader()
    try:
        with union_path.open(encoding="utf-8", newline="") as source:
            for row in csv.DictReader(source):
                writers[run_by_wsi[int(cast("str", row["wsi_id"]))] - 1].writerow(row)
        for handle in handles:
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        for path in paths:
            path.with_suffix(".csv.tmp").replace(path)
    finally:
        for handle in handles:
            if not handle.closed:
                handle.close()
    return paths


def _plan_audit(
    work: Mapping[int, WsiWork],
    plans: Sequence[tuple[int, ...]],
) -> list[dict[str, object]]:
    result = []
    for number, ids in enumerate(plans, start=1):
        values = [work[wsi] for wsi in ids]
        split_counts: Counter[str] = Counter()
        for value in values:
            split_counts[value.split] += value.union
        result.append({
            "run_number": number,
            "first_wsi_id": ids[0],
            "last_wsi_id": ids[-1],
            "wsi_ids": list(ids),
            "wsi_count": len(ids),
            "split_patch_counts": dict(split_counts),
            "union_patch_count": sum(v.union for v in values),
            "cancer_patch_count": sum(v.cancer for v in values),
            "tissue_patch_count": sum(v.tissue for v in values),
            "min_x": min(v.min_x for v in values),
            "max_x": max(v.max_x for v in values),
            "min_y": min(v.min_y for v in values),
            "max_y": max(v.max_y for v in values),
            "projected_raw_bytes": sum(v.union for v in values) * PATCH_BYTES,
        })
    return result


def _validate_real_counts(
    cancer: Mapping[str, int],
    tissue: Mapping[str, int],
    tissue_cross: Mapping[str, Mapping[str, Mapping[str, int]]],
    union: Mapping[str, int],
    overlap: int,
) -> None:
    if dict(cancer) != EXPECTED_CANCER:
        raise ValueError(f"Cancer counts disagree: {cancer}")
    if dict(tissue) != EXPECTED_TISSUE:
        raise ValueError(f"Tissue counts disagree: {tissue}")
    observed_class = {
        split: {
            name: sum(
                by_diagnosis.get(name, 0)
                for by_diagnosis in tissue_cross.get(split, {}).values()
            )
            for name in TISSUES
        }
        for split in SPLITS
    }
    if observed_class != EXPECTED_TISSUE_CLASS:
        raise ValueError(f"Tissue class counts disagree: {observed_class}")
    if dict(union) != EXPECTED_UNION or overlap != EXPECTED_OVERLAP:
        raise ValueError(f"Union counts disagree: {union}, overlap={overlap}")


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Spec 0019 materializer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cancer-input", type=Path, default=CANCER_INPUT)
    parser.add_argument("--tissue-input", type=Path, default=TISSUE_INPUT)
    parser.add_argument("--split", type=Path, default=SPLIT_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--assignment-output", type=Path, default=ASSIGNMENT_PATH)
    parser.add_argument("--audit-output", type=Path, default=AUDIT_PATH)
    args = parser.parse_args(argv)
    audit = materialize_consumption_manifests(
        cancer_input=args.cancer_input,
        tissue_input=args.tissue_input,
        split_path=args.split,
        output_dir=args.output_dir,
        assignment_path=args.assignment_output,
        audit_path=args.audit_output,
    )
    print(json.dumps(audit["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
