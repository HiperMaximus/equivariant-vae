#!/usr/bin/env python3
"""Build the frozen Spec 0054 cohort, folds, and A0 sentinel contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

CLASS_ORDER = ("CC", "EC", "HGSC", "LGSC", "MC")
CLASS_INDEX = {name: index for index, name in enumerate(CLASS_ORDER)}
FOLD_COUNT = 5
FOLD_SEED = 5401
PROBE_FOLD = 0
PATCH_RECORD_BYTES = 3 * 256 * 256
PATCH_HEADER_BYTES = 64


@dataclass(frozen=True)
class WSI:
    wsi_id: int
    diagnosis_index: int
    source_role: str
    coordinates: tuple[tuple[int, int], ...]

    @property
    def patch_count(self) -> int:
        return len(self.coordinates)


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def stable_key(wsi_id: int) -> str:
    return hashlib.sha256(f"{FOLD_SEED}:{wsi_id}".encode()).hexdigest()


def load_split(path: Path, role: str) -> list[WSI]:
    grouped: dict[int, list[tuple[int, int]]] = defaultdict(list)
    labels: dict[int, int] = {}
    seen: set[tuple[int, int, int]] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            wsi_id = int(row["wsi_id"])
            label = int(row["label"])
            coordinate = (int(row["x"]), int(row["y"]))
            identity = (wsi_id, *coordinate)
            if identity in seen:
                raise ValueError(f"Duplicate patch identity: {identity}")
            seen.add(identity)
            if wsi_id in labels and labels[wsi_id] != label:
                raise ValueError(f"Inconsistent diagnosis for WSI {wsi_id}")
            labels[wsi_id] = label
            grouped[wsi_id].append(coordinate)
    return [
        WSI(wsi_id, labels[wsi_id], role, tuple(coordinates))
        for wsi_id, coordinates in grouped.items()
    ]


def source_identities(path: Path) -> set[tuple[int, int, int, int]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            (int(row["wsi_id"]), int(row["label"]), int(row["x"]), int(row["y"]))
            for row in csv.DictReader(handle)
        }


def validate_atlas(train_csv: Path, validation_csv: Path, atlas_csv: Path) -> None:
    observed: dict[str, set[tuple[int, int, int, int]]] = {
        "train": set(),
        "valid": set(),
    }
    with atlas_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            observed[row["split"]].add(
                (
                    int(row["image_id"]),
                    CLASS_INDEX[row["label"]],
                    int(row["x"]),
                    int(row["y"]),
                )
            )
    if observed["train"] != source_identities(train_csv):
        raise ValueError("Train patch identities differ from the canonical atlas")
    if observed["valid"] != source_identities(validation_csv):
        raise ValueError("Validation patch identities differ from the canonical atlas")


def graph_summary(coordinates: tuple[tuple[int, int], ...]) -> dict[str, float | int]:
    lattice = {(x // 256, y // 256) for x, y in coordinates}
    if len(lattice) != len(coordinates) or any(x % 256 or y % 256 for x, y in coordinates):
        raise ValueError("Coordinates must be unique on the 256-pixel lattice")
    degrees = [
        sum((gx + dx, gy + dy) in lattice for dx in range(-2, 3) for dy in range(-2, 3))
        for gx, gy in lattice
    ]
    xs = [item[0] for item in lattice]
    ys = [item[1] for item in lattice]
    occupied_area = (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1)
    return {
        "patch_count": len(lattice),
        "log_patch_count": math.log(len(lattice)),
        "gx_min": min(xs),
        "gx_max": max(xs),
        "gy_min": min(ys),
        "gy_max": max(ys),
        "bounding_lattice_area": occupied_area,
        "occupancy_fraction": len(lattice) / occupied_area,
        "degree_min": min(degrees),
        "degree_mean": statistics.fmean(degrees),
        "degree_max": max(degrees),
    }


def assign_folds(wsis: list[WSI]) -> dict[int, int]:
    assignments: dict[int, int] = {}
    fold_sizes = [0] * FOLD_COUNT
    fold_patches = [0] * FOLD_COUNT
    for diagnosis_index in range(len(CLASS_ORDER)):
        rows = [row for row in wsis if row.diagnosis_index == diagnosis_index]
        rows.sort(
            key=lambda row: (
                row.source_role != "vae_validation",
                -row.patch_count,
                stable_key(row.wsi_id),
            )
        )
        class_counts = [0] * FOLD_COUNT
        role_counts = [Counter() for _ in range(FOLD_COUNT)]
        for row in rows:
            fold = min(
                range(FOLD_COUNT),
                key=lambda index: (
                    class_counts[index],
                    role_counts[index][row.source_role],
                    fold_sizes[index],
                    fold_patches[index],
                    hashlib.sha256(
                        f"{FOLD_SEED}:{row.wsi_id}:{index}".encode()
                    ).hexdigest(),
                ),
            )
            assignments[row.wsi_id] = fold
            class_counts[fold] += 1
            role_counts[fold][row.source_role] += 1
            fold_sizes[fold] += 1
            fold_patches[fold] += row.patch_count
        if max(class_counts) - min(class_counts) > 1:
            raise RuntimeError("Primary diagnosis balance failed")
    return assignments


def choose_sentinels(wsis: list[WSI]) -> dict[int, int]:
    sentinels = {}
    for diagnosis_index in range(len(CLASS_ORDER)):
        candidates = [
            row
            for row in wsis
            if row.diagnosis_index == diagnosis_index
            and row.source_role == "vae_validation"
        ]
        median = statistics.median(row.patch_count for row in candidates)
        chosen = min(
            candidates,
            key=lambda row: (abs(row.patch_count - median), row.patch_count, row.wsi_id),
        )
        sentinels[diagnosis_index] = chosen.wsi_id
    return sentinels


def median_wsi(rows: list[WSI]) -> int:
    median = statistics.median(row.patch_count for row in rows)
    return min(
        rows,
        key=lambda row: (abs(row.patch_count - median), row.patch_count, row.wsi_id),
    ).wsi_id


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--atlas-csv", type=Path, required=True)
    parser.add_argument("--historical-holdout-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    wsis = load_split(args.train_csv, "vae_train") + load_split(
        args.validation_csv, "vae_validation"
    )
    validate_atlas(args.train_csv, args.validation_csv, args.atlas_csv)
    if len(wsis) != 361 or len({row.wsi_id for row in wsis}) != 361:
        raise RuntimeError("Expected 361 distinct VAE-development WSI")
    assignments = assign_folds(wsis)
    sentinels = choose_sentinels(wsis)
    ordered_by_size = sorted(wsis, key=lambda row: (row.patch_count, row.wsi_id))
    cost_panel = {
        "median": ordered_by_size[len(ordered_by_size) // 2],
        "p99": ordered_by_size[math.ceil(0.99 * len(ordered_by_size)) - 1],
        "maximum": ordered_by_size[-1],
    }
    fold_sentinels = {
        str(fold): {
            "train": median_wsi([row for row in wsis if assignments[row.wsi_id] != fold]),
            "holdout": median_wsi([row for row in wsis if assignments[row.wsi_id] == fold]),
        }
        for fold in range(FOLD_COUNT)
    }
    with args.historical_holdout_csv.open(newline="", encoding="utf-8") as handle:
        historical_ids = {int(row["image_id"]) for row in csv.DictReader(handle)}
    overlap = historical_ids & {row.wsi_id for row in wsis}
    if overlap:
        raise ValueError(f"VAE-development cohort overlaps historical 152: {overlap}")
    rows = []
    for wsi in sorted(wsis, key=lambda item: item.wsi_id):
        rows.append(
            {
                "wsi_id": wsi.wsi_id,
                "diagnosis": CLASS_ORDER[wsi.diagnosis_index],
                "diagnosis_index": wsi.diagnosis_index,
                "vae_source_role": wsi.source_role,
                "fold": assignments[wsi.wsi_id],
                "is_a0_sentinel": int(sentinels[wsi.diagnosis_index] == wsi.wsi_id),
                **graph_summary(wsi.coordinates),
            }
        )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=tuple(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    fold_class_counts = {
        str(fold): {
            CLASS_ORDER[index]: sum(
                row["fold"] == fold and row["diagnosis_index"] == index for row in rows
            )
            for index in range(len(CLASS_ORDER))
        }
        for fold in range(FOLD_COUNT)
    }
    payload = {
        "schema_version": "spec0054.a0_contract.v1",
        "scope": "361_vae_development_wsi_no_historical_152",
        "fold_algorithm": "diagnosis_primary_greedy_role_then_patch_balance",
        "fold_count": FOLD_COUNT,
        "fold_seed": FOLD_SEED,
        "probe_fold": PROBE_FOLD,
        "class_order": list(CLASS_ORDER),
        "class_counts": {
            CLASS_ORDER[index]: sum(row.diagnosis_index == index for row in wsis)
            for index in range(len(CLASS_ORDER))
        },
        "fold_class_counts": fold_class_counts,
        "source_role_counts": dict(sorted(Counter(row.source_role for row in wsis).items())),
        "sentinels": {
            CLASS_ORDER[index]: {
                "wsi_id": wsi_id,
                "patch_count": next(row.patch_count for row in wsis if row.wsi_id == wsi_id),
                "source_role": "vae_validation",
            }
            for index, wsi_id in sentinels.items()
        },
        "cost_panel": {
            name: {
                "wsi_id": row.wsi_id,
                "patch_count": row.patch_count,
                "diagnosis": CLASS_ORDER[row.diagnosis_index],
                "source_role": row.source_role,
            }
            for name, row in cost_panel.items()
        },
        "fold_sentinels": fold_sentinels,
        "sources": {
            "patch_dataset": "maximusshtefan/patches-pre-shuffled-ubc-ocean/1",
            "atlas_dataset": "maximusshtefan/train-val-atlas-ubc-ocean/1",
            "frozen_weights": "maximusshtefan/eqvae-frozen-vae-weights-v1/1",
            "train_csv_sha256": sha256(args.train_csv),
            "validation_csv_sha256": sha256(args.validation_csv),
            "atlas_csv_sha256": sha256(args.atlas_csv),
            "normal_state_file_sha256": "30064fa414f21deeb7ec5f312467ad4e8a3a93f6e7199e7be77f6ed3b28887c7",
            "so2_state_file_sha256": "06802ceb6ba4fb0f46d2b88f751a33a355600db84b412806b563a6d757cb3c12",
        },
        "patch_sources": {
            "vae_train": {
                "csv_name": "ubc_train_shuffled.csv",
                "binary_name": "ubc_train_shuffled.bin",
                "rows": 300000,
                "binary_bytes": PATCH_HEADER_BYTES + 300000 * PATCH_RECORD_BYTES,
                "binary_crc32": 1289496176,
            },
            "vae_validation": {
                "csv_name": "ubc_ocean_valid.csv",
                "binary_name": "ubc_ocean_valid.bin",
                "rows": 30000,
                "binary_bytes": PATCH_HEADER_BYTES + 30000 * PATCH_RECORD_BYTES,
                "binary_crc32": 3532641891,
            },
            "header_bytes": PATCH_HEADER_BYTES,
            "record_bytes": PATCH_RECORD_BYTES,
            "physical_index": "zero_based_csv_data_row_aligned_to_binary_record",
            "logical_order": "wsi_id_y_x",
        },
        "historical_152_overlap_count": 0,
        "historical_holdout_csv_sha256": sha256(args.historical_holdout_csv),
        "cohort_csv": args.output_csv.name,
        "cohort_csv_sha256": sha256(args.output_csv),
        "patient_or_case_group": "unavailable_in_canonical_source",
        "acquisition_site": "unavailable_in_canonical_source",
        "image_update_indicator": "unavailable_in_canonical_source",
    }
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
