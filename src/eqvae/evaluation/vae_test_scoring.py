# Copyright 2026 HiperMaximus
# pyright: reportAny=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
# ruff: noqa: C420, DOC201, DOC501, EM101, EM102, PLR0913, PLR0916, S311, TRY003, TRY004
"""Patch-level reconstruction scoring with WSI-cluster uncertainty."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

from eqvae.evaluation.vae_test import (
    REDACTED_LOCATION_HEADER,
    REMOTE_METRIC_HEADER,
    canonical_json_sha256,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

DIAGNOSES: Final = ("CC", "EC", "HGSC", "LGSC", "MC")
DIAGNOSIS_TO_INDEX: Final = {"HGSC": 0, "LGSC": 1, "EC": 2, "CC": 3, "MC": 4}
EXPECTED_DIAGNOSIS_WSI_COUNTS: Final = {
    "CC": 5,
    "EC": 6,
    "HGSC": 8,
    "LGSC": 2,
    "MC": 2,
}
EXPECTED_DIAGNOSIS_PATCH_COUNTS: Final = {
    "CC": 15_000,
    "EC": 16_138,
    "HGSC": 24_000,
    "LGSC": 6_000,
    "MC": 6_000,
}
BRANCHES: Final = ("normal", "so2")
METRICS: Final = ("mae_norm", "mse_norm", "psnr_img", "ssim_img")
PRIMARY_METRIC: Final = "mae_norm"
BOOTSTRAP_REPLICATES: Final = 10_000
BOOTSTRAP_SEED: Final = 4501
TEST_VECTOR_PATH: Final = Path("docs/data/spec0045_vae_test_scorer_vector.json")


@dataclass(frozen=True)
class OracleLabel:
    """Diagnosis attached locally only after remote evidence is frozen."""

    diagnosis_label: str
    diagnosis_index: int
    wsi_id: int
    x: int
    y: int
    split: str


def load_remote_metric_rows(path: Path) -> list[dict[str, object]]:
    """Load the exact diagnosis-free paired remote metric table."""
    rows: list[dict[str, object]] = []
    with gzip.open(path, mode="rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != REMOTE_METRIC_HEADER:
            raise ValueError("Remote metric CSV header differs")
        rows.extend(cast("dict[str, object]", row) for row in reader)
    return rows


def load_oracle_labels(
    path: Path,
    *,
    expected_sha256: str,
) -> dict[int, OracleLabel]:
    """Authenticate and load the local-only diagnosis and identity oracle."""
    if _sha256_file(path) != expected_sha256:
        raise ValueError("VAE test diagnosis oracle SHA-256 differs")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected = (
            "atlas_row_index",
            "wsi_id",
            "diagnosis_label",
            "diagnosis_index",
            "x",
            "y",
            "split",
        )
        if tuple(reader.fieldnames or ()) != expected:
            raise ValueError("VAE test diagnosis oracle header differs")
        labels = {
            _integer(row["atlas_row_index"], "atlas_row_index"): OracleLabel(
                diagnosis_label=row["diagnosis_label"],
                diagnosis_index=_integer(
                    row["diagnosis_index"],
                    "diagnosis_index",
                ),
                wsi_id=_integer(row["wsi_id"], "wsi_id"),
                x=_integer(row["x"], "x"),
                y=_integer(row["y"], "y"),
                split=row["split"],
            )
            for row in reader
        }
    if len(labels) != sum(EXPECTED_DIAGNOSIS_PATCH_COUNTS.values()):
        raise ValueError("VAE test diagnosis oracle row count differs")
    return labels


def score_vae_test_metrics(
    *,
    remote_rows: Sequence[Mapping[str, object]],
    labels_by_atlas_row: Mapping[int, OracleLabel],
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    """Score overall paired patch metrics plus diagnosis diagnostics."""
    return _score_vae_test_metrics(
        remote_rows=remote_rows,
        labels_by_atlas_row=labels_by_atlas_row,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
        expected_patch_counts=EXPECTED_DIAGNOSIS_PATCH_COUNTS,
        expected_wsi_counts=EXPECTED_DIAGNOSIS_WSI_COUNTS,
    )


def _score_vae_test_metrics(
    *,
    remote_rows: Sequence[Mapping[str, object]],
    labels_by_atlas_row: Mapping[int, OracleLabel],
    bootstrap_replicates: int,
    bootstrap_seed: int,
    expected_patch_counts: Mapping[str, int],
    expected_wsi_counts: Mapping[str, int],
) -> dict[str, object]:
    """Apply the frozen estimator while allowing a compact scorer-vector support."""
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    parsed = _validate_and_join(remote_rows, labels_by_atlas_row)
    per_wsi = _per_wsi_means(parsed)
    _validate_support(
        parsed,
        per_wsi,
        expected_patch_counts=expected_patch_counts,
        expected_wsi_counts=expected_wsi_counts,
    )
    branch_summaries: dict[str, dict[str, object]] = {
        branch: _branch_summary(parsed, per_wsi, branch=branch) for branch in BRANCHES
    }
    bootstrap_rows, intervals = _paired_cluster_bootstrap(
        per_wsi,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    normal_pooled = cast(
        "Mapping[str, float | None]",
        branch_summaries["normal"]["pooled_patch_mean"],
    )
    so2_pooled = cast(
        "Mapping[str, float | None]",
        branch_summaries["so2"]["pooled_patch_mean"],
    )
    point_differences = {
        metric: _optional_difference(
            normal_pooled[metric],
            so2_pooled[metric],
        )
        for metric in METRICS
    }
    return {
        "schema_version": "spec0045.scored_result.v1",
        "primary_endpoint": {
            "metric": PRIMARY_METRIC,
            "aggregation": "pooled_patch_mean",
            "difference": "normal_minus_so2",
            "lower_is_better": True,
            "normal_value": normal_pooled[PRIMARY_METRIC],
            "so2_value": so2_pooled[PRIMARY_METRIC],
            "point_difference": point_differences[PRIMARY_METRIC],
            "interval": intervals[PRIMARY_METRIC],
        },
        "branches": branch_summaries,
        "paired_normal_minus_so2": {
            metric: {
                "point_difference": point_differences[metric],
                "interval": intervals[metric],
                "confirmatory": metric == PRIMARY_METRIC,
                "direction": (
                    "negative_favors_normal"
                    if metric in {"mae_norm", "mse_norm"}
                    else "positive_favors_normal"
                ),
            }
            for metric in METRICS
        },
        "per_wsi": [_json_safe_metric_row(row) for row in per_wsi],
        "bootstrap_rows": bootstrap_rows,
        "bootstrap": {
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "confidence": 0.95,
            "sampling_unit": "WSI",
            "stratified_by": None,
            "cluster_count": len(per_wsi),
            "estimand": "pooled_patch_mean_weighted_by_patch_count",
            "same_draws_for_both_models_and_all_metrics": True,
            "covers_training_seed_uncertainty": False,
            "covers_patch_selector_uncertainty": False,
        },
        "limitations": [
            "Diagnosis is exploratory; LGSC and MC each contain only two WSIs.",
            "Intervals cover WSI resampling, not training seeds or patch selectors.",
            "Patch-level population SD is descriptive dispersion, not uncertainty.",
            "MSE, PSNR and SSIM are prespecified secondary endpoints.",
        ],
    }


def verify_test_vector(path: Path | None = None) -> None:
    """Recompute the frozen estimator vector and require its exact result hash."""
    vector_path = path or (_repo_root(Path(__file__).resolve()) / TEST_VECTOR_PATH)
    vector = _read_object(vector_path)
    if (
        vector.get("schema_version") != "spec0045.vae_test_scorer_vector.v1"
        or vector.get("replicates") != BOOTSTRAP_REPLICATES
        or vector.get("seed") != BOOTSTRAP_SEED
        or vector.get("direction") != "normal_minus_so2"
        or vector.get("quantile_method") != "linear"
    ):
        raise ValueError("Spec 0045 scorer vector contract differs")
    rows, labels, patch_counts = _build_vector_fixture(
        cast("Mapping[str, object]", vector["fixture"]),
    )
    result = _score_vae_test_metrics(
        remote_rows=rows,
        labels_by_atlas_row=labels,
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
        bootstrap_seed=BOOTSTRAP_SEED,
        expected_patch_counts=patch_counts,
        expected_wsi_counts=EXPECTED_DIAGNOSIS_WSI_COUNTS,
    )
    if canonical_json_sha256(result) != vector.get("expected_result_sha256"):
        raise ValueError("Spec 0045 scorer vector result differs")


def _build_vector_fixture(
    fixture: Mapping[str, object],
) -> tuple[list[dict[str, object]], dict[int, OracleLabel], dict[str, int]]:
    """Create a small unequal-cluster fixture spanning the real five strata."""
    patch_pattern = cast("Sequence[object]", fixture.get("patch_count_pattern"))
    if tuple(patch_pattern) != (1, 2, 3, 2):
        raise ValueError("Spec 0045 scorer fixture pattern differs")
    rows: list[dict[str, object]] = []
    labels: dict[int, OracleLabel] = {}
    patch_counts = {diagnosis: 0 for diagnosis in DIAGNOSES}
    atlas = 0
    wsi_id = 1000
    for diagnosis_index, diagnosis in enumerate(DIAGNOSES):
        for local_wsi in range(EXPECTED_DIAGNOSIS_WSI_COUNTS[diagnosis]):
            wsi_id += 1
            count = cast("int", patch_pattern[local_wsi % len(patch_pattern)])
            for patch_index in range(count):
                baseline = (
                    0.08
                    + 0.01 * diagnosis_index
                    + 0.002 * local_wsi
                    + 0.0001 * patch_index
                )
                row: dict[str, object] = {
                    "run_number": diagnosis_index + 1,
                    "file_index": atlas,
                    "atlas_row_index": atlas,
                    "wsi_id": wsi_id,
                    "x": patch_index * 256,
                    "y": 0,
                    "split": "test",
                    "normal_mae_norm": baseline,
                    "normal_mse_norm": baseline**2,
                    "normal_psnr_img": 31.0 - baseline,
                    "normal_ssim_img": 0.91 - baseline / 10.0,
                    "so2_mae_norm": baseline + 0.004 - 0.001 * diagnosis_index,
                    "so2_mse_norm": (baseline + 0.006) ** 2,
                    "so2_psnr_img": 30.6 - baseline,
                    "so2_ssim_img": 0.905 - baseline / 10.0,
                }
                rows.append(row)
                labels[atlas] = OracleLabel(
                    diagnosis,
                    DIAGNOSIS_TO_INDEX[diagnosis],
                    wsi_id,
                    patch_index * 256,
                    0,
                    "test",
                )
                patch_counts[diagnosis] += 1
                atlas += 1
    return rows, labels, patch_counts


def _validate_and_join(
    remote_rows: Sequence[Mapping[str, object]],
    labels: Mapping[int, OracleLabel],
) -> list[dict[str, object]]:
    if len(remote_rows) != len(labels):
        raise ValueError("Remote metric and oracle row counts differ")
    joined: list[dict[str, object]] = []
    seen_atlas: set[int] = set()
    previous: tuple[int, int, int] | None = None
    for index, row in enumerate(remote_rows):
        if set(row) != set(REMOTE_METRIC_HEADER):
            raise ValueError(f"Remote metric schema differs at row {index}")
        identity = {
            name: _integer(row[name], name) for name in REDACTED_LOCATION_HEADER[:-1]
        }
        if row["split"] != "test":
            raise ValueError("Remote metric row is not test split")
        atlas = identity["atlas_row_index"]
        if atlas in seen_atlas or atlas not in labels:
            raise ValueError("Remote metric atlas identity differs")
        seen_atlas.add(atlas)
        key = (identity["wsi_id"], identity["y"], identity["x"])
        if previous is not None and key <= previous:
            raise ValueError("Remote metric row order differs")
        previous = key
        label = labels[atlas]
        if (
            label.diagnosis_label not in DIAGNOSES
            or label.diagnosis_index != DIAGNOSIS_TO_INDEX[label.diagnosis_label]
            or label.wsi_id != identity["wsi_id"]
            or label.x != identity["x"]
            or label.y != identity["y"]
            or label.split != row["split"]
        ):
            raise ValueError("Remote metric location differs from the local oracle")
        values = {
            f"{branch}_{metric}": _metric_float(
                row[f"{branch}_{metric}"],
                allow_inf=metric == "psnr_img",
            )
            for branch in BRANCHES
            for metric in METRICS
        }
        joined.append(
            {
                **identity,
                "split": "test",
                "diagnosis_label": label.diagnosis_label,
                "diagnosis_index": label.diagnosis_index,
                **values,
            },
        )
    if seen_atlas != set(labels):
        raise ValueError("Remote metrics do not cover the exact oracle")
    return joined


def _per_wsi_means(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[cast("int", row["wsi_id"])].append(row)
    output: list[dict[str, object]] = []
    for wsi_id in sorted(grouped):
        group = grouped[wsi_id]
        diagnoses = {cast("str", row["diagnosis_label"]) for row in group}
        if len(diagnoses) != 1:
            raise ValueError(f"WSI {wsi_id} has multiple diagnoses")
        output.append(
            {
                "wsi_id": wsi_id,
                "diagnosis_label": diagnoses.pop(),
                "patch_count": len(group),
                **{
                    f"{branch}_{metric}": _mean(
                        [cast("float", row[f"{branch}_{metric}"]) for row in group],
                    )
                    for branch in BRANCHES
                    for metric in METRICS
                },
            },
        )
    return output


def _validate_support(
    patch_rows: Sequence[Mapping[str, object]],
    per_wsi: Sequence[Mapping[str, object]],
    *,
    expected_patch_counts: Mapping[str, int],
    expected_wsi_counts: Mapping[str, int],
) -> None:
    patch_counts = {diagnosis: 0 for diagnosis in DIAGNOSES}
    wsi_counts = {diagnosis: 0 for diagnosis in DIAGNOSES}
    for row in patch_rows:
        diagnosis = cast("str", row["diagnosis_label"])
        if diagnosis not in patch_counts:
            raise ValueError(f"Unexpected diagnosis {diagnosis}")
        patch_counts[diagnosis] += 1
    for row in per_wsi:
        diagnosis = cast("str", row["diagnosis_label"])
        wsi_counts[diagnosis] += 1
    if patch_counts != dict(expected_patch_counts) or wsi_counts != dict(
        expected_wsi_counts,
    ):
        raise ValueError("Diagnosis patch/WSI support differs")


def _branch_summary(
    patch_rows: Sequence[Mapping[str, object]],
    per_wsi: Sequence[Mapping[str, object]],
    *,
    branch: str,
) -> dict[str, object]:
    per_diagnosis: dict[str, object] = {}
    diagnosis_means: dict[str, dict[str, float | None]] = {}
    for diagnosis in DIAGNOSES:
        wsi_rows = [row for row in per_wsi if row["diagnosis_label"] == diagnosis]
        diagnosis_patch_rows = [
            row for row in patch_rows if row["diagnosis_label"] == diagnosis
        ]
        diagnosis_patch_summary = {
            metric: _distribution([
                cast("float", row[f"{branch}_{metric}"]) for row in diagnosis_patch_rows
            ])
            for metric in METRICS
        }
        diagnosis_means[diagnosis] = {
            metric: _finite_aggregate(
                [cast("float", row[f"{branch}_{metric}"]) for row in wsi_rows],
            )
            for metric in METRICS
        }
        per_diagnosis[diagnosis] = {
            "wsi_count": len(wsi_rows),
            "patch_count": len(diagnosis_patch_rows),
            "pooled_patch_mean": {
                metric: cast("float | None", diagnosis_patch_summary[metric]["mean"])
                for metric in METRICS
            },
            "patch_distribution": diagnosis_patch_summary,
            "wsi_macro": diagnosis_means[diagnosis],
        }
    patch_summary = {
        metric: _distribution(
            [cast("float", row[f"{branch}_{metric}"]) for row in patch_rows],
        )
        for metric in METRICS
    }
    wsi_macro = {
        metric: _finite_aggregate(
            [cast("float", row[f"{branch}_{metric}"]) for row in per_wsi],
        )
        for metric in METRICS
    }
    return {
        "equal_wsi_macro": wsi_macro,
        "pooled_patch_mean": {
            metric: cast("float | None", patch_summary[metric]["mean"])
            for metric in METRICS
        },
        "patch_distribution": patch_summary,
        "per_diagnosis": per_diagnosis,
    }


def _paired_cluster_bootstrap(
    per_wsi: Sequence[Mapping[str, object]],
    *,
    replicates: int,
    seed: int,
) -> tuple[list[dict[str, float | int]], dict[str, object]]:
    rng = random.Random(seed)
    draws: dict[str, list[float]] = {metric: [] for metric in METRICS}
    rows: list[dict[str, float | int]] = []
    psnr_has_inf = any(
        not math.isfinite(cast("float", row[f"{branch}_psnr_img"]))
        for row in per_wsi
        for branch in BRANCHES
    )
    for replicate in range(replicates):
        sampled = rng.choices(per_wsi, k=len(per_wsi))
        record: dict[str, float | int] = {"replicate": replicate}
        for metric in METRICS:
            if metric == "psnr_img" and psnr_has_inf:
                continue
            total_patches = sum(cast("int", row["patch_count"]) for row in sampled)
            difference = (
                math.fsum(
                    cast("int", row["patch_count"])
                    * (
                        cast("float", row[f"normal_{metric}"])
                        - cast("float", row[f"so2_{metric}"])
                    )
                    for row in sampled
                )
                / total_patches
            )
            record[f"normal_minus_so2_{metric}"] = difference
            draws[metric].append(difference)
        rows.append(record)
    intervals: dict[str, object] = {}
    for metric, values in draws.items():
        if metric == "psnr_img" and psnr_has_inf:
            intervals[metric] = {
                "status": "omitted_positive_infinity",
                "low": None,
                "high": None,
            }
        else:
            intervals[metric] = {
                "status": "exploratory" if metric != PRIMARY_METRIC else "primary",
                "low": _quantile(values, 0.025),
                "high": _quantile(values, 0.975),
            }
    return rows, intervals


def _distribution(values: Sequence[float]) -> dict[str, int | float | None]:
    if not values:
        raise ValueError("Cannot summarize an empty distribution")
    inf_count = sum(math.isinf(value) for value in values)
    if any(math.isnan(value) for value in values):
        raise ValueError("Metric distribution contains NaN")
    finite = [value for value in values if math.isfinite(value)]
    if inf_count:
        return {
            "n": len(values),
            "inf_count": inf_count,
            "mean": None,
            "population_sd": None,
            "finite_mean": _mean(finite) if finite else None,
            "finite_population_sd": statistics.pstdev(finite) if finite else None,
            "median": None,
            "q1": None,
            "q3": None,
            "min": min(finite) if finite else None,
            "max": max(finite) if finite else None,
        }
    ordered = sorted(values)
    return {
        "n": len(values),
        "inf_count": 0,
        "mean": _mean(values),
        "population_sd": statistics.pstdev(values),
        "finite_mean": None,
        "finite_population_sd": None,
        "median": _quantile(ordered, 0.5),
        "q1": _quantile(ordered, 0.25),
        "q3": _quantile(ordered, 0.75),
        "min": ordered[0],
        "max": ordered[-1],
    }


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values or not 0.0 <= probability <= 1.0:
        raise ValueError("Quantile input is invalid")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty sequence")
    return math.fsum(values) / len(values)


def _finite_aggregate(values: Sequence[float | None]) -> float | None:
    """Average only when the complete estimand is finite and therefore defined."""
    if not values:
        raise ValueError("Cannot average an empty sequence")
    if any(value is None or not math.isfinite(value) for value in values):
        return None
    return _mean([cast("float", value) for value in values])


def _optional_difference(left: float | None, right: float | None) -> float | None:
    """Keep undefined infinity-bearing contrasts JSON-safe instead of capping them."""
    if left is None or right is None:
        return None
    return left - right


def _json_safe_metric_row(row: Mapping[str, object]) -> dict[str, object]:
    """Replace positive-infinite WSI PSNR with an explicit lossless marker."""
    output = dict(row)
    for branch in BRANCHES:
        key = f"{branch}_psnr_img"
        value = cast("float", output[key])
        output[f"{key}_positive_infinity"] = math.isinf(value)
        if math.isinf(value):
            output[key] = None
    return output


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        parsed = int(cast("str | int", value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer") from error
    if parsed < 0 or str(parsed) != str(value):
        raise ValueError(f"{name} must be canonical and nonnegative")
    return parsed


def _metric_float(value: object, *, allow_inf: bool) -> float:
    try:
        parsed = float(cast("str | float", value))
    except (TypeError, ValueError) as error:
        raise ValueError("Remote metric must be numeric") from error
    if math.isnan(parsed) or (math.isinf(parsed) and (not allow_inf or parsed < 0)):
        raise ValueError("Remote metric violates the finite-value contract")
    return parsed


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return cast("dict[str, object]", value)


def _repo_root(path: Path) -> Path:
    for parent in path.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise ValueError("Could not locate repository root")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "BRANCHES",
    "DIAGNOSES",
    "DIAGNOSIS_TO_INDEX",
    "EXPECTED_DIAGNOSIS_PATCH_COUNTS",
    "EXPECTED_DIAGNOSIS_WSI_COUNTS",
    "METRICS",
    "PRIMARY_METRIC",
    "TEST_VECTOR_PATH",
    "OracleLabel",
    "load_oracle_labels",
    "load_remote_metric_rows",
    "score_vae_test_metrics",
    "verify_test_vector",
]
