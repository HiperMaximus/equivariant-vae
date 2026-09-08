# Copyright 2026 HiperMaximus
# ruff: noqa: C901, DOC201, DOC501, EM101, EM102, PLR0912, PLR0913, PLR0914, PLR0915, PLR0916, PLW0717, T201, TRY003, TRY300, TRY301
"""Build the compact unified physical catalog and supervised logical datasets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from operator import itemgetter
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import numpy as np

from eqvae.data.latent_shards import (
    EXPECTED_WORK_MANIFEST_SHA256,
    load_work_manifest,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

MODELS: Final = ("normal_vae", "so2_vae")
SPLITS: Final = ("train", "validation", "test")
TISSUES: Final = ("tumor", "stroma", "necrosis")
TISSUE_SIZES: Final = (250, 500, 1_000, 2_500, 5_671)
SELECTION_SEED: Final = 20_260_827
TOPUP_PART: Final = 11
LATENT_HEADER_BYTES: Final = 64
LATENT_RECORD_BYTES: Final = 65_536
SHA256_HEX_LENGTH: Final = 64
BASE_AUDIT: Final = Path(
    "runs/kaggle/ubc_ocean_latent_store_finalizer/dataset/"
    "spec0021_latent_store_global_audit.json",
)
BASE_VIEWS: Final = BASE_AUDIT.parent / "views"
BASE_WORK_ROOT: Final = Path(
    "runs/local/ubc_ocean_eval_consumption/work_shards",
)
CANCER_PLAN: Final = Path("runs/local/ubc_ocean_cancer_topup")
TOPUP_METADATA: Final = Path(
    "runs/kaggle/ubc_ocean_cancer_topup_metadata/dataset",
)
SPLIT_PATH: Final = Path("docs/data/ubc_ocean_eval_wsi_split.csv")
OUTPUT_ROOT: Final = Path("runs/local/ubc_ocean_supervised_manifests")
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


@dataclass(frozen=True)
class TissueRow:
    """One model-independent tissue embedding pointer."""

    atlas_row_index: int
    wsi_id: int
    x: int
    y: int
    tissue_label: str
    split: str
    part: int
    file_index: int

    @property
    def identity(self) -> tuple[int, int, int, int]:
        """The canonical physical row identity."""
        return (self.atlas_row_index, self.wsi_id, self.x, self.y)

    @property
    def order_key(self) -> tuple[int, int, int]:
        """The canonical logical ordering key."""
        return (self.wsi_id, self.y, self.x)


def build_supervised_manifests(
    *,
    base_audit_path: Path = BASE_AUDIT,
    base_views_root: Path = BASE_VIEWS,
    base_work_root: Path = BASE_WORK_ROOT,
    cancer_plan_root: Path = CANCER_PLAN,
    topup_metadata_root: Path = TOPUP_METADATA,
    split_path: Path = SPLIT_PATH,
    output_root: Path = OUTPUT_ROOT,
    tissue_sizes: Sequence[int] = TISSUE_SIZES,
    enforce_real_counts: bool = True,
) -> dict[str, object]:
    """Materialize all compact model-independent supervised dataset files."""
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite {output_root}")
    base_audit = _read_object(base_audit_path)
    logical_audit_path = cancer_plan_root / "spec0022_logical_dataset_audit.json"
    logical_audit = _read_object(logical_audit_path)
    pair_path = topup_metadata_root / "spec0022_cancer_topup_pair_audit.json"
    pair = _read_object(pair_path)
    if (
        base_audit.get("schema_version") != "spec0021.latent_store_global_audit.v1"
        or base_audit.get("status") != "complete"
        or logical_audit.get("schema_version")
        != "spec0022.cancer_logical_dataset_audit.v1"
        or logical_audit.get("status") != "complete"
        or pair.get("schema_version") != "spec0022.cancer_topup_pair_audit.v1"
        or pair.get("status") != "complete"
        or cast("Mapping[str, object]", logical_audit["topup_pair_audit"])["sha256"]
        != _sha256(pair_path)
    ):
        raise ValueError("Completed compact Spec 0021/0022 audits are required")
    split_by_wsi = _load_split(split_path)
    staging = output_root.with_name(f".{output_root.name}.building")
    if staging.exists():
        raise FileExistsError(f"Stale staging directory exists: {staging}")
    (staging / "wsi").mkdir(parents=True)
    (staging / "tissue").mkdir()
    try:
        catalog_rows, part_counts = _catalog_rows(base_audit, pair)
        catalog_path = staging / "physical_parts.csv"
        _write_csv(catalog_path, CATALOG_HEADER, catalog_rows)
        base_pointers = _load_base_pointers(base_work_root)
        topup_pointers = _load_topup_pointers(
            cancer_plan_root / "inference_bundle/cancer_topup_manifest.csv",
        )
        logical_root = cancer_plan_root / "logical"
        _validate_logical_sources(
            logical_root,
            _mapping(logical_audit, "validated_logical_files"),
        )
        wsi_records, wsi_counts, wsi_splits = _write_wsi_files(
            logical_root,
            staging / "wsi",
            part_counts,
            base_pointers=base_pointers,
            topup_pointers=topup_pointers,
        )
        location_records = _mapping(base_audit, "location_files")
        tissue_rows = {
            split: _load_tissue_locations(
                base_views_root / f"tissue_{split}_locations.csv",
                split=split,
                part_counts=part_counts,
                expected_record=_mapping(location_records, f"tissue_{split}"),
            )
            for split in SPLITS
        }
        tissue_records, tissue_summary = _write_tissue_files(
            tissue_rows,
            staging / "tissue",
            tissue_sizes=tissue_sizes,
        )
        _validate_split_membership(wsi_splits, tissue_rows, split_by_wsi)
        if enforce_real_counts:
            expected_wsi = {"train": 308_359, "validation": 66_706, "test": 66_194}
            expected_tissue = {"validation": 31_339, "test": 31_572}
            if wsi_counts != expected_wsi or any(
                len(tissue_rows[split]) != count
                for split, count in expected_tissue.items()
            ):
                raise ValueError("Unified supervised dataset counts differ")
        inputs = {
            "base_global_audit_sha256": _sha256(base_audit_path),
            "spec0022_logical_audit_sha256": _sha256(logical_audit_path),
            "topup_pair_audit_sha256": _sha256(pair_path),
            "split_sha256": _sha256(split_path),
            **{
                f"work_part_{part:02d}_sha256": _sha256(
                    base_work_root / f"run_{part:02d}_of_05.csv",
                )
                for part in range(1, 6)
            },
            **{
                f"tissue_{split}_locations_sha256": _sha256(
                    base_views_root / f"tissue_{split}_locations.csv",
                )
                for split in SPLITS
            },
        }
        audit: dict[str, object] = {
            "schema_version": "spec0023.supervised_manifest_audit.v1",
            "status": "complete",
            "selection_seed": SELECTION_SEED,
            "inputs": inputs,
            "physical_catalog": _file_record(catalog_path, len(catalog_rows)),
            "wsi_files": wsi_records,
            "wsi_instance_counts": wsi_counts,
            "tissue_files": tissue_records,
            "tissue_summary": tissue_summary,
            "acceptance": {
                "all_latent_binaries_remain_remote": True,
                "model_independent_logical_files": True,
                "part_11_uses_ordinary_catalog_resolution": True,
                "wsi_splits_disjoint": True,
                "tissue_subsets_nested": True,
            },
        }
        audit_path = staging / "spec0023_supervised_manifest_audit.json"
        _write_json(audit_path, audit)
        _fsync_directory(staging)
        staging.replace(output_root)
        _fsync_directory(output_root.parent)
        return audit
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _catalog_rows(
    base_audit: Mapping[str, object],
    pair: Mapping[str, object],
) -> tuple[list[tuple[object, ...]], dict[int, int]]:
    artifacts = _mapping(base_audit, "artifacts")
    rows: list[tuple[object, ...]] = []
    part_counts: dict[int, int] = {}
    for part in range(1, 6):
        counts: set[int] = set()
        for model in MODELS:
            model_parts = _mapping(artifacts, model)
            record = _mapping(model_parts, str(part))
            binary = _mapping(record, "binary")
            sidecar = _mapping(record, "sidecar")
            binary_bytes = _integer(binary, "bytes")
            row_count = _latent_row_count(binary_bytes)
            counts.add(row_count)
            rows.append((
                part,
                model,
                f"maximusshtefan/eqvae-ubc-ocean-latent-run-{part:02d}",
                _string(binary, "name"),
                row_count,
                binary_bytes,
                _sha_field(binary, "sha256"),
                _string(sidecar, "name"),
                _integer(sidecar, "bytes"),
                _sha_field(sidecar, "sha256"),
            ))
        if len(counts) != 1:
            raise ValueError(f"Base part {part} model row counts differ")
        part_counts[part] = counts.pop()
    topup_artifacts = _mapping(pair, "artifacts")
    topup_count = _integer(pair, "row_count")
    for model in MODELS:
        record = _mapping(topup_artifacts, model)
        binary_bytes = _integer(record, "bin_bytes")
        if _latent_row_count(binary_bytes) != topup_count:
            raise ValueError("Part 11 binary size/row count differs")
        rows.append((
            TOPUP_PART,
            model,
            "maximusshtefan/eqvae-ubc-ocean-cancer-latent-top-up",
            _string(record, "bin_name"),
            topup_count,
            binary_bytes,
            _sha_field(record, "bin_sha256"),
            _string(record, "sidecar_name"),
            _integer(record, "sidecar_bytes"),
            _sha_field(record, "sidecar_sha256"),
        ))
    part_counts[TOPUP_PART] = topup_count
    return rows, part_counts


def _write_wsi_files(
    source_root: Path,
    output_root: Path,
    part_counts: Mapping[int, int],
    *,
    base_pointers: Mapping[tuple[int, int, int, int], tuple[int, int]],
    topup_pointers: Mapping[tuple[int, int, int, int], int],
) -> tuple[dict[str, dict[str, object]], dict[str, int], dict[int, str]]:
    records: dict[str, dict[str, object]] = {}
    counts: dict[str, int] = {}
    wsi_splits: dict[int, str] = {}
    all_identities: set[tuple[int, int, int, int]] = set()
    for split in SPLITS:
        source_instances = source_root / f"wsi_cancer_{split}_instances.csv"
        source_bags = source_root / f"wsi_cancer_{split}_bags.csv"
        destination_instances = output_root / source_instances.name
        destination_bags = output_root / source_bags.name
        summaries: list[tuple[int, str, int, int, int]] = []
        active: tuple[int, str, int, int] | None = None
        previous: tuple[int, int, int] | None = None
        count = 0
        with (
            source_instances.open(encoding="utf-8", newline="") as source,
            (
                destination_instances.open("x", encoding="utf-8", newline="")
            ) as destination,
        ):
            reader = csv.DictReader(source)
            writer = csv.writer(destination, lineterminator="\n")
            writer.writerow(WSI_INSTANCE_HEADER)
            for index, raw in enumerate(reader):
                if raw.get("split") != split or _csv_int(raw, "instance_row") != index:
                    raise ValueError(f"Invalid WSI instance row in {source_instances}")
                identity = (
                    _csv_int(raw, "atlas_row_index"),
                    _csv_int(raw, "wsi_id"),
                    _csv_int(raw, "x"),
                    _csv_int(raw, "y"),
                )
                key = (identity[1], identity[3], identity[2])
                if previous is not None and key <= previous:
                    raise ValueError(
                        f"WSI instance order differs in {source_instances}",
                    )
                if identity in all_identities:
                    raise ValueError("Duplicate WSI logical identity")
                previous = key
                all_identities.add(identity)
                source_name = raw.get("store_source")
                shard = _csv_int(raw, "shard_number")
                file_index = _csv_int(raw, "file_index")
                if source_name == "base":
                    part = shard
                    if base_pointers.get(identity) != (part, file_index):
                        raise ValueError(
                            "WSI base pointer differs from the sealed view",
                        )
                elif source_name == "topup" and shard == 1:
                    part = TOPUP_PART
                    if topup_pointers.get(identity) != file_index:
                        raise ValueError("WSI top-up pointer differs from its manifest")
                else:
                    raise ValueError("Unknown WSI physical source")
                if part not in part_counts or file_index not in range(
                    part_counts[part],
                ):
                    raise ValueError("WSI physical pointer is outside its part")
                label = raw["diagnosis_label"]
                label_index = _csv_int(raw, "diagnosis_index")
                writer.writerow((
                    index,
                    *identity,
                    label,
                    label_index,
                    split,
                    part,
                    file_index,
                ))
                if active is None:
                    active = (identity[1], label, label_index, index)
                elif identity[1] != active[0]:
                    summaries.append((*active, index - active[3]))
                    active = (identity[1], label, label_index, index)
                elif (label, label_index) != active[1:3]:
                    raise ValueError("Within-WSI diagnosis differs")
                count = index + 1
            if active is not None:
                summaries.append((*active, count - active[3]))
        bags = _read_csv(source_bags)
        if len(bags) != len(summaries):
            raise ValueError("WSI bag/instance count differs")
        with destination_bags.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(WSI_BAG_HEADER)
            cursor = 0
            for bag_row, (raw, summary) in enumerate(zip(bags, summaries, strict=True)):
                wsi_id, label, label_index, start, instance_count = summary
                if (
                    _csv_int(raw, "bag_row") != bag_row
                    or _csv_int(raw, "wsi_id") != wsi_id
                    or raw["diagnosis_label"] != label
                    or _csv_int(raw, "diagnosis_index") != label_index
                    or raw["split"] != split
                    or _csv_int(raw, "instance_start") != cursor
                    or start != cursor
                    or _csv_int(raw, "instance_count") != instance_count
                ):
                    raise ValueError("WSI bag range differs")
                previous_split = wsi_splits.setdefault(wsi_id, split)
                if previous_split != split:
                    raise ValueError("WSI crosses logical splits")
                writer.writerow((
                    bag_row,
                    wsi_id,
                    label,
                    label_index,
                    split,
                    cursor,
                    instance_count,
                ))
                cursor += instance_count
            if cursor != count:
                raise ValueError("WSI bags do not consume every instance")
        counts[split] = count
        records[destination_instances.name] = _file_record(
            destination_instances,
            count,
        )
        records[destination_bags.name] = _file_record(
            destination_bags,
            len(bags),
        )
    return records, counts, wsi_splits


def _load_base_pointers(
    work_root: Path,
) -> dict[tuple[int, int, int, int], tuple[int, int]]:
    result: dict[tuple[int, int, int, int], tuple[int, int]] = {}
    for part in range(1, 6):
        path = work_root / f"run_{part:02d}_of_05.csv"
        manifest = load_work_manifest(
            path,
            run_number=part,
            expected_sha256=EXPECTED_WORK_MANIFEST_SHA256[part],
        )
        for file_index, row in enumerate(manifest.rows):
            identity = (
                row.identity.atlas_row_index,
                row.identity.wsi_id,
                row.identity.x,
                row.identity.y,
            )
            pointer = (part, file_index)
            if identity in result:
                raise ValueError("Duplicate identity in sealed base work manifests")
            result[identity] = pointer
    return result


def _load_topup_pointers(path: Path) -> dict[tuple[int, int, int, int], int]:
    result: dict[tuple[int, int, int, int], int] = {}
    for file_index, raw in enumerate(_read_csv(path)):
        identity = (
            _csv_int(raw, "atlas_row_index"),
            _csv_int(raw, "wsi_id"),
            _csv_int(raw, "x"),
            _csv_int(raw, "y"),
        )
        if identity in result:
            raise ValueError("Duplicate identity in top-up inference manifest")
        result[identity] = file_index
    return result


def _validate_logical_sources(
    logical_root: Path,
    records: Mapping[str, object],
) -> None:
    expected_names = {
        f"wsi_cancer_{split}_{kind}.csv"
        for split in SPLITS
        for kind in ("instances", "bags")
    }
    if set(records) != expected_names:
        raise ValueError("Sealed Spec 0022 logical file set differs")
    for name in sorted(expected_names):
        _validate_recorded_file(logical_root / name, _mapping(records, name))


def _load_tissue_locations(
    path: Path,
    *,
    split: str,
    part_counts: Mapping[int, int],
    expected_record: Mapping[str, object] | None = None,
) -> list[TissueRow]:
    if expected_record is not None:
        _validate_recorded_file(path, expected_record)
    rows: list[TissueRow] = []
    seen: set[tuple[int, int, int, int]] = set()
    previous: tuple[int, int, int] | None = None
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            part = _csv_int(raw, "run_number")
            file_index = _csv_int(raw, "file_index")
            row = TissueRow(
                atlas_row_index=_csv_int(raw, "atlas_row_index"),
                wsi_id=_csv_int(raw, "wsi_id"),
                x=_csv_int(raw, "x"),
                y=_csv_int(raw, "y"),
                tissue_label=raw["tissue_label"],
                split=raw["split"],
                part=part,
                file_index=file_index,
            )
            if (
                row.split != split
                or row.tissue_label not in TISSUES
                or part not in part_counts
                or file_index not in range(part_counts[part])
                or row.identity in seen
                or (previous is not None and row.order_key <= previous)
            ):
                raise ValueError(f"Invalid tissue location row in {path}")
            seen.add(row.identity)
            previous = row.order_key
            rows.append(row)
    return rows


def _write_tissue_files(
    rows_by_split: Mapping[str, Sequence[TissueRow]],
    output_root: Path,
    *,
    tissue_sizes: Sequence[int],
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    summary: dict[str, object] = {}
    priorities = _tissue_priorities(rows_by_split["train"])
    prior_sets: dict[str, set[tuple[int, int, int, int]]] = {
        tissue: set() for tissue in TISSUES
    }
    for size in tissue_sizes:
        selected: list[tuple[TissueRow, int]] = []
        class_counts: dict[str, int] = {}
        wsi_counts: dict[str, int] = {}
        for tissue in TISSUES:
            priority = priorities[tissue]
            if len(priority) < size:
                raise ValueError(f"Tissue {tissue} lacks {size} training rows")
            prefix = priority[:size]
            identities = {row.identity for row, _rank in prefix}
            if not prior_sets[tissue] <= identities:
                raise ValueError("Tissue subsets are not nested")
            prior_sets[tissue] = identities
            selected.extend(prefix)
            class_counts[tissue] = len(prefix)
            wsi_counts[tissue] = len({row.wsi_id for row, _rank in prefix})
        selected.sort(key=lambda item: item[0].order_key)
        name = f"tissue_train_{size:04d}_per_class.csv"
        path = output_root / name
        _write_tissue_csv(path, selected)
        records[name] = _file_record(path, len(selected))
        summary[name] = {
            "class_counts": class_counts,
            "wsi_counts_by_class": wsi_counts,
        }
    for split in ("validation", "test"):
        name = f"tissue_{split}.csv"
        path = output_root / name
        selected = [(row, -1) for row in rows_by_split[split]]
        _write_tissue_csv(path, selected)
        records[name] = _file_record(path, len(selected))
        summary[name] = {
            "class_counts": dict(Counter(row.tissue_label for row, _ in selected)),
            "wsi_counts_by_class": {
                tissue: len({
                    row.wsi_id for row, _rank in selected if row.tissue_label == tissue
                })
                for tissue in TISSUES
            },
        }
    return records, summary


def _tissue_priorities(
    rows: Sequence[TissueRow],
) -> dict[str, list[tuple[TissueRow, int]]]:
    grouped: dict[tuple[str, int], list[TissueRow]] = defaultdict(list)
    for row in rows:
        grouped[row.tissue_label, row.wsi_id].append(row)
    result: dict[str, list[tuple[TissueRow, int]]] = {}
    for tissue_index, tissue in enumerate(TISSUES):
        ranked: list[tuple[int, Fraction, int, int, TissueRow]] = []
        for (group_tissue, wsi_id), group in sorted(grouped.items()):
            if group_tissue != tissue:
                continue
            ordered = sorted(group, key=lambda row: row.order_key)
            generator = np.random.Generator(
                np.random.PCG64(
                    np.random.SeedSequence([
                        SELECTION_SEED,
                        tissue_index,
                        wsi_id,
                    ]),
                ),
            )
            positions = cast(
                "list[int]",
                generator.permutation(len(ordered)).tolist(),
            )
            for within_rank, position in enumerate(positions):
                row = ordered[position]
                ranked.append((
                    0 if within_rank == 0 else 1,
                    Fraction(within_rank, len(ordered)),
                    wsi_id,
                    within_rank,
                    row,
                ))
        ranked.sort(key=itemgetter(slice(4)))
        result[tissue] = [
            (row, selection_rank) for selection_rank, (*_key, row) in enumerate(ranked)
        ]
    return result


def _write_tissue_csv(path: Path, rows: Sequence[tuple[TissueRow, int]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(TISSUE_HEADER)
        for dataset_row, (row, rank) in enumerate(rows):
            writer.writerow((
                dataset_row,
                row.atlas_row_index,
                row.wsi_id,
                row.x,
                row.y,
                row.tissue_label,
                row.split,
                "" if rank < 0 else rank,
                row.part,
                row.file_index,
            ))


def _validate_split_membership(
    wsi_splits: Mapping[int, str],
    tissue_rows: Mapping[str, Sequence[TissueRow]],
    split_by_wsi: Mapping[int, str],
) -> None:
    observed = dict(wsi_splits)
    for split, rows in tissue_rows.items():
        for row in rows:
            if split_by_wsi.get(row.wsi_id) != split:
                raise ValueError("Tissue WSI disagrees with the frozen split")
            prior = observed.setdefault(row.wsi_id, split)
            if prior != split:
                raise ValueError("A WSI crosses supervised splits")
    if any(split_by_wsi.get(wsi_id) != split for wsi_id, split in observed.items()):
        raise ValueError("WSI logical data disagrees with the frozen split")


def _load_split(path: Path) -> dict[int, str]:
    result: dict[int, str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            wsi_id = _csv_int(raw, "wsi_id")
            split = raw["split"]
            if split not in SPLITS or wsi_id in result:
                raise ValueError("Frozen WSI split is invalid")
            result[wsi_id] = split
    return result


def _latent_row_count(binary_bytes: int) -> int:
    payload = binary_bytes - LATENT_HEADER_BYTES
    if payload < 0 or payload % LATENT_RECORD_BYTES:
        raise ValueError("Latent binary bytes do not end on a record boundary")
    return payload // LATENT_RECORD_BYTES


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(
    path: Path,
    header: Sequence[str],
    rows: Iterable[Sequence[object]],
) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _file_record(path: Path, row_count: int) -> dict[str, object]:
    return {
        "path": path.name,
        "row_count": row_count,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _validate_recorded_file(
    path: Path,
    record: Mapping[str, object],
) -> None:
    rows = sum(1 for _ in path.open(encoding="utf-8")) - 1
    if (
        _integer(record, "row_count") != rows
        or _integer(record, "bytes") != path.stat().st_size
        or _sha_field(record, "sha256") != _sha256(path)
    ):
        raise ValueError(f"Compact file differs from its sealed audit: {path}")


def _mapping(raw: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"{key} must be an object")
    return cast("Mapping[str, object]", value)


def _integer(raw: Mapping[str, object], key: str) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer")
    return value


def _string(raw: Mapping[str, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise TypeError(f"{key} must be a nonempty string")
    return value


def _sha_field(raw: Mapping[str, object], key: str) -> str:
    value = _string(raw, key)
    if len(value) != SHA256_HEX_LENGTH or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{key} must be a SHA-256")
    return value


def _csv_int(raw: Mapping[str, str], key: str) -> int:
    try:
        return int(raw[key])
    except (KeyError, ValueError) as error:
        raise ValueError(f"Invalid CSV integer {key}") from error


def _read_object(path: Path) -> dict[str, object]:
    value = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return cast("dict[str, object]", value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(f"{json.dumps(payload, indent=2, sort_keys=True)}\n")
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main(argv: Sequence[str] | None = None) -> int:
    """Build the real compact supervised manifests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args(argv)
    audit = build_supervised_manifests(output_root=cast("Path", args.output_root))
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
