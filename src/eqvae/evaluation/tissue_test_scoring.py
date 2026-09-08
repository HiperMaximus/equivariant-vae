# Copyright 2026 HiperMaximus
# pyright: reportAny=false, reportArgumentType=false, reportUnnecessaryComparison=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
# ruff: noqa: C901, DOC201, DOC501, EM101, EM102, PLR0912, PLR0913, PLR0914, PLR0915, PLR0916, PLR2004, PLW0717, TRY003
"""Frozen post-retrieval scorer for the label-blind Spec 0043 tissue test."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

CLASS_ORDER: Final = ("tumor", "stroma", "necrosis")
BRANCHES: Final = ("normal_vae", "so2_vae")
BUDGETS: Final = (250, 500, 1_000, 2_500, 5_671)
BOOTSTRAP_REPLICATES: Final = 10_000
BOOTSTRAP_SEED: Final = 3901
TEST_PATCH_COUNT: Final = 31_572
TEST_WSI_COUNT: Final = 23
EXPECTED_PATCH_SUPPORT: Final = {"tumor": 21_796, "stroma": 8_500, "necrosis": 1_276}
EXPECTED_WSI_SUPPORT: Final = {"tumor": 23, "stroma": 21, "necrosis": 5}
EXPECTED_STRATA: Final = {
    (0,): 2,
    (0, 1): 16,
    (0, 1, 2): 5,
}
LABEL_ORACLE_SHA256: Final = (
    "ff4183da11bdee4065a061410ef7ec13f45f1fe9791cdbe7139c664721b44082"
)
TEST_VECTOR_PATH: Final = Path("docs/data/spec0043_tissue_test_scorer_vector.json")
ROW_IDENTITY_SCHEMA: Final = b"eqvae_spec0043_tissue_row_identity_v1"
IDENTITY_FIELDS: Final = (
    "dataset_row",
    "atlas_row_index",
    "wsi_id",
    "x",
    "y",
    "part",
    "file_index",
)
PREDICTION_FIELDS: Final = frozenset({
    *IDENTITY_FIELDS,
    "logit_tumor",
    "logit_stroma",
    "logit_necrosis",
    "prediction",
})
PREDICTION_HEADER: Final = (
    *IDENTITY_FIELDS,
    "logit_tumor",
    "logit_stroma",
    "logit_necrosis",
    "prediction",
)
ORACLE_FIELDS: Final = frozenset({
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
})
ORACLE_HEADER: Final = (
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
SCALAR_METRICS: Final = (
    "macro_f1",
    "balanced_accuracy",
    "accuracy",
    "mean_cross_entropy",
)

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int64]
type LogicalIdentity = tuple[int, int, int, int, int, int, int]


def score_tissue_test_predictions(
    *,
    predictions: Mapping[str, Mapping[int, Sequence[Mapping[str, object]]]],
    oracle_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Score the complete frozen test once under the locked Spec 0043 contract."""
    return _score_tissue_test_predictions(
        predictions=predictions,
        oracle_rows=oracle_rows,
        replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
        expected_patch_count=TEST_PATCH_COUNT,
        expected_patch_support=EXPECTED_PATCH_SUPPORT,
        expected_wsi_support=EXPECTED_WSI_SUPPORT,
        expected_strata=EXPECTED_STRATA,
    )


def load_prediction_csv_gz(path: Path) -> list[dict[str, object]]:
    """Load one exact label-free remote prediction table."""
    rows: list[dict[str, object]] = []
    with gzip.open(path, mode="rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != PREDICTION_HEADER:
            raise ValueError("Remote prediction CSV header differs")
        rows.extend(
            {
                **{field: _parse_int(raw[field], field) for field in IDENTITY_FIELDS},
                **{
                    field: _parse_float(raw[field], field)
                    for field in (
                        "logit_tumor",
                        "logit_stroma",
                        "logit_necrosis",
                    )
                },
                "prediction": _parse_int(raw["prediction"], "prediction"),
            }
            for raw in reader
        )
    return _validate_prediction_rows(rows, expected_count=TEST_PATCH_COUNT)


def load_label_oracle(path: Path) -> list[dict[str, object]]:
    """Open and authenticate the frozen local-only tissue label oracle."""
    if _sha256(path) != LABEL_ORACLE_SHA256:
        raise ValueError("Tissue test label oracle differs")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != ORACLE_HEADER:
            raise ValueError("Tissue test label oracle header differs")
        rows = [
            {
                **{field: _parse_int(row[field], field) for field in IDENTITY_FIELDS},
                "tissue_label": row["tissue_label"],
                "split": row["split"],
                "selection_rank": row["selection_rank"],
            }
            for row in reader
        ]
    return _validate_oracle_rows(
        rows,
        expected_patch_count=TEST_PATCH_COUNT,
        expected_patch_support=EXPECTED_PATCH_SUPPORT,
        expected_wsi_support=EXPECTED_WSI_SUPPORT,
        expected_strata=EXPECTED_STRATA,
    )[0]


def verify_test_vector(path: Path | None = None) -> None:
    """Recompute the frozen compact scorer vector and require byte-pinned output."""
    vector_path = path or (_repo_root(Path(__file__).resolve()) / TEST_VECTOR_PATH)
    vector = _read_object(vector_path)
    if (
        vector.get("schema_version") != "spec0043.tissue_test_scorer_vector.v1"
        or vector.get("replicates") != BOOTSTRAP_REPLICATES
        or vector.get("seed") != BOOTSTRAP_SEED
        or vector.get("budgets") != list(BUDGETS)
        or vector.get("direction") != "normal_minus_so2"
        or vector.get("quantile_method") != "linear"
    ):
        raise ValueError("Spec 0043 scorer vector contract differs")
    fixture = cast("Mapping[str, object]", vector.get("fixture"))
    predictions, oracle_rows = _build_vector_fixture(fixture)
    result = _score_tissue_test_predictions(
        predictions=predictions,
        oracle_rows=oracle_rows,
        replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
        expected_patch_count=len(oracle_rows),
        expected_patch_support=None,
        expected_wsi_support={"tumor": 23, "stroma": 21, "necrosis": 5},
        expected_strata=EXPECTED_STRATA,
    )
    if _json_sha256(result) != vector.get("expected_result_sha256"):
        raise ValueError("Spec 0043 scorer vector result differs")


def write_exclusive_json(path: Path, value: object) -> None:
    """Atomically create, but never replace, a receipt or pre-score claim."""
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(_canonical_json(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def score_retrieved_tissue_test_output(
    *,
    remote_output_root: Path,
    launch_receipt_path: Path,
    launch_claim_path: Path,
    label_oracle_path: Path,
    output_root: Path,
    expected_contract: Mapping[str, object],
    expected_input_contract_sha256: str,
    expected_input_receipt_sha256: str,
    expected_input_receipt_bytes: int,
    expected_kernel_sha256: str,
    expected_metadata_sha256: str,
    expected_kernel_bytes: int,
    expected_metadata_bytes: int,
) -> dict[str, object]:
    """Authenticate one retrieved run, seal a claim, then open labels and score."""
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite {output_root}")
    staging = output_root.with_name(f".{output_root.name}.scoring")
    if staging.exists():
        raise FileExistsError(f"Stale scoring directory exists: {staging}")
    scorer_path = Path(__file__).resolve()
    repository = _repo_root(scorer_path)
    vector_path = repository / TEST_VECTOR_PATH
    scorer_sha256 = _sha256(scorer_path)
    vector_sha256 = _sha256(vector_path)
    if (
        expected_contract.get("schema_version") != "spec0043.label_blind_input.v1"
        or expected_contract.get("scorer_sha256") != scorer_sha256
        or expected_contract.get("test_vector_sha256") != vector_sha256
        or expected_contract.get("budgets_per_class") != list(BUDGETS)
        or set(cast("Mapping[str, object]", expected_contract.get("branches")))
        != set(BRANCHES)
    ):
        raise ValueError("Spec 0043 expected scoring contract differs")
    verify_test_vector(vector_path)
    authenticated = _authenticate_retrieved_output(
        remote_output_root=remote_output_root,
        launch_receipt_path=launch_receipt_path,
        launch_claim_path=launch_claim_path,
        expected_contract=expected_contract,
        expected_input_contract_sha256=expected_input_contract_sha256,
        expected_input_receipt_sha256=expected_input_receipt_sha256,
        expected_input_receipt_bytes=expected_input_receipt_bytes,
        expected_kernel_sha256=expected_kernel_sha256,
        expected_metadata_sha256=expected_metadata_sha256,
        expected_kernel_bytes=expected_kernel_bytes,
        expected_metadata_bytes=expected_metadata_bytes,
    )
    prediction_files = cast(
        "dict[str, dict[int, Path]]",
        authenticated["prediction_files"],
    )
    prediction_sha256 = {
        branch: {
            str(budget): _sha256(prediction_files[branch][budget]) for budget in BUDGETS
        }
        for branch in BRANCHES
    }
    pre_score = {
        "schema_version": "spec0043.pre_score_contract.v1",
        "remote_launch_receipt_sha256": _sha256(launch_receipt_path),
        "exclusive_launch_claim_sha256": _sha256(launch_claim_path),
        "remote_output_receipt_sha256": _sha256(
            remote_output_root / "kaggle_output_receipt.json",
        ),
        "remote_run_contract_sha256": _sha256(
            cast("Path", authenticated["payload_root"]) / "run_contract.json",
        ),
        "remote_overall_status_sha256": _sha256(
            cast("Path", authenticated["payload_root"]) / "overall_status.json",
        ),
        "remote_prediction_sha256": prediction_sha256,
        "input_contract_sha256": expected_input_contract_sha256,
        "input_dataset_receipt": {
            "bytes": expected_input_receipt_bytes,
            "sha256": expected_input_receipt_sha256,
        },
        "label_oracle_sha256": LABEL_ORACLE_SHA256,
        "scorer_sha256": scorer_sha256,
        "test_vector_sha256": vector_sha256,
        "spec_sha256": expected_contract["spec_sha256"],
        "policy": "no_remote_retry_after_this_contract",
    }
    external_claim = output_root.with_name(
        f"{output_root.name}.pre_score_claim.json",
    )
    write_exclusive_json(external_claim, pre_score)

    staging.mkdir(parents=True)
    try:
        _write_json(staging / "pre_score_contract.json", pre_score)
        predictions = {
            branch: {
                budget: load_prediction_csv_gz(prediction_files[branch][budget])
                for budget in BUDGETS
            }
            for branch in BRANCHES
        }
        # This is the first operation that opens the local-only label oracle. The
        # exclusive prediction/scorer claim above therefore already exists.
        oracle = load_label_oracle(label_oracle_path)
        scored = score_tissue_test_predictions(
            predictions=predictions,
            oracle_rows=oracle,
        )
        _write_json(staging / "scored_result.json", scored)
        _write_json(
            staging / "paired_bootstrap.json",
            scored["paired_normal_minus_so2"],
        )
        _write_json(
            staging / "label_efficiency_table.json",
            _label_efficiency_table(scored),
        )
        for branch in BRANCHES:
            for budget in BUDGETS:
                _write_joined_predictions(
                    staging
                    / "branches"
                    / branch
                    / f"budget_{budget:04d}_per_class"
                    / "scored_predictions.csv.gz",
                    rows=predictions[branch][budget],
                    oracle_rows=oracle,
                )
        artifacts = {
            path.relative_to(staging).as_posix(): _file_record(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        completion = {
            "schema_version": "spec0043.scoring_completion.v1",
            "status": "complete",
            "primary_metric": "macro_f1",
            "direction": "normal_minus_so2",
            "budgets_per_class": list(BUDGETS),
            "artifacts": artifacts,
            "branch_results": {
                branch: cast("Mapping[str, object]", scored["branches"])[branch]
                for branch in BRANCHES
            },
            "paired_normal_minus_so2": scored["paired_normal_minus_so2"],
            "bootstrap": scored["bootstrap"],
            "limitations": scored["limitations"],
        }
        # Completion is deliberately the final file created inside staging.
        _write_json(staging / "overall_status.json", completion)
        staging.replace(output_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return completion


def _score_tissue_test_predictions(
    *,
    predictions: Mapping[str, Mapping[int, Sequence[Mapping[str, object]]]],
    oracle_rows: Sequence[Mapping[str, object]],
    replicates: int,
    seed: int,
    expected_patch_count: int,
    expected_patch_support: Mapping[str, int] | None,
    expected_wsi_support: Mapping[str, int],
    expected_strata: Mapping[tuple[int, ...], int],
) -> dict[str, object]:
    if replicates < 1:
        raise ValueError("Bootstrap replicate count must be positive")
    if set(predictions) != set(BRANCHES):
        raise ValueError("Prediction branch set differs")
    oracle, truths, wsi_ids, stratum_indices = _validate_oracle_rows(
        oracle_rows,
        expected_patch_count=expected_patch_count,
        expected_patch_support=expected_patch_support,
        expected_wsi_support=expected_wsi_support,
        expected_strata=expected_strata,
    )
    oracle_identities = [_identity(row) for row in oracle]

    normalized: dict[str, dict[int, list[dict[str, object]]]] = {}
    for branch in BRANCHES:
        branch_tables = predictions[branch]
        if set(branch_tables) != set(BUDGETS):
            raise ValueError(f"Prediction budget set differs for {branch}")
        normalized[branch] = {}
        for budget in BUDGETS:
            rows = _validate_prediction_rows(
                branch_tables[budget],
                expected_count=expected_patch_count,
            )
            if [_identity(row) for row in rows] != oracle_identities:
                message = (
                    "Prediction and oracle logical identity/order differ: "
                    f"{branch}/{budget}"
                )
                raise ValueError(
                    message,
                )
            normalized[branch][budget] = rows

    reference = normalized[BRANCHES[0]][BUDGETS[0]]
    reference_identities = [_identity(row) for row in reference]
    for branch in BRANCHES:
        for budget in BUDGETS:
            if [
                _identity(row) for row in normalized[branch][budget]
            ] != reference_identities:
                raise ValueError("Cross-branch/budget logical identity differs")

    multiplicities = _bootstrap_multiplicities(
        wsi_count=len(wsi_ids),
        stratum_indices=stratum_indices,
        replicates=replicates,
        seed=seed,
    )
    wsi_lookup = {wsi_id: index for index, wsi_id in enumerate(wsi_ids)}
    wsi_indices = np.asarray(
        [wsi_lookup[int(row["wsi_id"])] for row in oracle],
        dtype=np.int64,
    )
    truths_array = np.asarray(truths, dtype=np.int64)

    branch_output: dict[str, object] = {}
    bootstrap_metrics: dict[str, dict[int, dict[str, FloatArray]]] = {}
    point_metrics: dict[str, dict[int, dict[str, object]]] = {}
    for branch in BRANCHES:
        budget_output: dict[str, object] = {}
        bootstrap_metrics[branch] = {}
        point_metrics[branch] = {}
        for budget in BUDGETS:
            rows = normalized[branch][budget]
            logits = _logits(rows)
            predicted = logits.argmax(axis=1).astype(np.int64, copy=False)
            remote_predicted = np.asarray(
                [int(row["prediction"]) for row in rows],
                dtype=np.int64,
            )
            if not np.array_equal(predicted, remote_predicted):
                raise ValueError(f"Remote argmax differs: {branch}/{budget}")
            losses = _cross_entropies(logits, truths_array)
            metrics = _metrics(
                truths_array,
                predicted,
                losses,
                wsi_indices,
                wsi_ids,
            )
            samples = _bootstrap_metric_arrays(
                truths=truths_array,
                predictions=predicted,
                losses=losses,
                wsi_indices=wsi_indices,
                multiplicities=multiplicities,
                wsi_count=len(wsi_ids),
            )
            point_metrics[branch][budget] = metrics
            bootstrap_metrics[branch][budget] = samples
            budget_output[str(budget)] = {
                "metrics": metrics,
                "percentile_95": _metric_intervals(samples),
            }
        branch_macro = np.column_stack([
            bootstrap_metrics[branch][budget]["macro_f1"] for budget in BUDGETS
        ])
        point_macro = np.asarray(
            [float(point_metrics[branch][budget]["macro_f1"]) for budget in BUDGETS],
            dtype=np.float64,
        )
        branch_aulc_samples = _aulc(branch_macro)
        branch_aulc = float(_aulc(point_macro[np.newaxis, :])[0])
        branch_output[branch] = {
            "budgets": budget_output,
            "aulc": {
                "estimate": branch_aulc,
                "percentile_95": _interval(branch_aulc_samples),
            },
        }

    paired_budgets: dict[str, object] = {}
    delta_bootstrap_columns: list[FloatArray] = []
    delta_points: list[float] = []
    for budget in BUDGETS:
        normal_samples = bootstrap_metrics["normal_vae"][budget]
        so2_samples = bootstrap_metrics["so2_vae"][budget]
        sample_differences = {
            name: normal_samples[name] - so2_samples[name] for name in normal_samples
        }
        estimate_differences = _metric_differences(
            point_metrics["normal_vae"][budget],
            point_metrics["so2_vae"][budget],
        )
        delta_bootstrap_columns.append(sample_differences["macro_f1"])
        delta_points.append(float(estimate_differences["macro_f1"]))
        paired_budgets[str(budget)] = {
            "estimates": _nest_flat_metrics(estimate_differences),
            "pointwise_percentile_95": _metric_intervals(sample_differences),
        }

    normal_macro = np.column_stack([
        bootstrap_metrics["normal_vae"][budget]["macro_f1"] for budget in BUDGETS
    ])
    so2_macro = np.column_stack([
        bootstrap_metrics["so2_vae"][budget]["macro_f1"] for budget in BUDGETS
    ])
    normal_point_macro = np.asarray(
        [float(point_metrics["normal_vae"][budget]["macro_f1"]) for budget in BUDGETS],
        dtype=np.float64,
    )
    so2_point_macro = np.asarray(
        [float(point_metrics["so2_vae"][budget]["macro_f1"]) for budget in BUDGETS],
        dtype=np.float64,
    )
    delta_aulc_samples = _aulc(normal_macro) - _aulc(so2_macro)
    delta_aulc = float(
        _aulc(normal_point_macro[np.newaxis, :])[0]
        - _aulc(so2_point_macro[np.newaxis, :])[0],
    )
    delta_bootstrap = np.column_stack([*delta_bootstrap_columns, delta_aulc_samples])
    delta_estimates = np.asarray([*delta_points, delta_aulc], dtype=np.float64)
    centered_errors = delta_bootstrap - delta_estimates[np.newaxis, :]
    critical_value = float(
        np.quantile(
            np.max(np.abs(centered_errors), axis=1),
            0.95,
            method="linear",
        ),
    )
    contrast_names = [*(f"macro_f1_budget_{budget}" for budget in BUDGETS), "aulc"]
    simultaneous = {
        name: {
            "estimate": float(estimate),
            "lower": float(estimate - critical_value),
            "upper": float(estimate + critical_value),
        }
        for name, estimate in zip(contrast_names, delta_estimates, strict=True)
    }

    return {
        "schema_version": "spec0043.tissue_test_scored_result.v1",
        "class_order": list(CLASS_ORDER),
        "budgets": list(BUDGETS),
        "direction": "normal_minus_so2",
        "patch_count": len(oracle),
        "wsi_count": len(wsi_ids),
        "branches": branch_output,
        "paired_normal_minus_so2": {
            "budgets": paired_budgets,
            "aulc": {
                "estimate": delta_aulc,
                "pointwise_percentile_95": _interval(delta_aulc_samples),
            },
            "six_contrast_simultaneous_95": {
                "method": "max_abs_centered_bootstrap_error",
                "quantile_method": "linear",
                "critical_value": critical_value,
                "contrasts": simultaneous,
            },
        },
        "bootstrap": {
            "replicates": replicates,
            "seed": seed,
            "confidence": 0.95,
            "sampling_unit": "WSI",
            "stratified_by": "true_tissue_support",
            "stratum_counts": {
                "tumor_only": expected_strata[0,],
                "tumor_stroma": expected_strata[0, 1],
                "tumor_stroma_necrosis": expected_strata[0, 1, 2],
            },
            "shared_multiplicities_all_branches_and_budgets": True,
            "conditional_on_observed_tissue_support_strata": True,
            "covers_training_seed_uncertainty": False,
        },
        "aulc": {
            "axis": "log10_labels_per_class",
            "normalization": "divide_by_log10_budget_range",
            "integration": "adjacent_trapezoids_without_smoothing_or_envelope",
        },
        "limitations": [
            (
                "Intervals cover WSI sampling conditional on the observed support "
                "strata, not training-seed uncertainty."
            ),
            (
                "All five budgets have one optimization trajectory and share nested "
                "training data and test patches."
            ),
            "Necrosis evidence comes from five WSIs.",
        ],
    }


def _authenticate_retrieved_output(
    *,
    remote_output_root: Path,
    launch_receipt_path: Path,
    launch_claim_path: Path,
    expected_contract: Mapping[str, object],
    expected_input_contract_sha256: str,
    expected_input_receipt_sha256: str,
    expected_input_receipt_bytes: int,
    expected_kernel_sha256: str,
    expected_metadata_sha256: str,
    expected_kernel_bytes: int,
    expected_metadata_bytes: int,
) -> dict[str, object]:
    dataset_reference = _required_string(expected_contract, "dataset_reference")
    actor = _required_string(expected_contract, "dataset_actor")
    kernel_id = _required_string(expected_contract, "kernel_id")
    kernel_sources = cast("Sequence[object]", expected_contract.get("kernel_sources"))
    expected_sources = {
        "competition_sources": [],
        "dataset_sources": [dataset_reference],
        "kernel_sources": list(kernel_sources),
        "model_sources": [],
    }
    expected_kernel_files = {
        "kernel-metadata.json": {
            "bytes": expected_metadata_bytes,
            "sha256": expected_metadata_sha256,
        },
        "run.py": {
            "bytes": expected_kernel_bytes,
            "sha256": expected_kernel_sha256,
        },
    }
    expected_claim = {
        "schema_version": "spec0043.exclusive_launch_claim.v1",
        "status": "claimed_before_remote_push",
        "authorization": "ok let's do the patch tissue test evaluation",
        "authorization_date": "2026-09-06",
        "dataset_reference": dataset_reference,
        "kernel_id": kernel_id,
        "input_contract_sha256": expected_input_contract_sha256,
        "input_dataset_receipt": {
            "bytes": expected_input_receipt_bytes,
            "sha256": expected_input_receipt_sha256,
        },
        "kernel_files": expected_kernel_files,
        "scientific_retries_authorized": 0,
    }
    if _read_object(launch_claim_path) != expected_claim:
        raise ValueError("Spec 0043 exclusive launch claim differs")

    launch = _read_object(launch_receipt_path)
    version = launch.get("accepted_version")
    reference = launch.get("kernel_reference")
    if (
        set(launch)
        != {
            "schema_version",
            "actor",
            "original_kernel_id",
            "requested_kernel_id",
            "kernel_id",
            "accepted_version",
            "kernel_reference",
            "source_locators",
            "source_metadata_sha256",
            "upload_metadata_sha256",
            "source_files",
            "upload_files",
        }
        or launch.get("schema_version") != "eqvae.kaggle_kernel_launch.v1"
        or launch.get("actor") != actor
        or launch.get("original_kernel_id") != kernel_id
        or launch.get("requested_kernel_id") != kernel_id
        or launch.get("kernel_id") != kernel_id
        or isinstance(version, bool)
        or not isinstance(version, int)
        or version < 1
        or reference != f"{kernel_id}/{version}"
        or launch.get("source_locators") != expected_sources
        or launch.get("source_metadata_sha256") != expected_metadata_sha256
        or launch.get("upload_metadata_sha256") != expected_metadata_sha256
        or launch.get("source_files") != expected_kernel_files
        or launch.get("upload_files") != expected_kernel_files
    ):
        raise ValueError("Spec 0043 launch receipt differs")

    receipt_path = remote_output_root / "kaggle_output_receipt.json"
    receipt = _read_object(receipt_path)
    declared = cast("Mapping[str, object]", receipt.get("files"))
    observed = {
        path.relative_to(remote_output_root).as_posix(): _file_record(path)
        for path in sorted(remote_output_root.rglob("*"))
        if path.is_file() and path != receipt_path
    }
    if (
        receipt.get("schema_version") != "eqvae.kaggle_download.v1"
        or receipt.get("resource_kind") != "kernel"
        or receipt.get("resource_owner") != actor
        or receipt.get("resource_slug") != kernel_id.split("/", maxsplit=1)[1]
        or receipt.get("resource_version") != version
        or receipt.get("resource_reference") != reference
        or dict(declared) != observed
        or any(path.is_symlink() for path in remote_output_root.rglob("*"))
    ):
        raise ValueError("Spec 0043 downloaded output receipt/bytes differ")

    payload_candidates = [
        candidate
        for candidate in (
            remote_output_root,
            remote_output_root / "tissue_test_evaluation",
        )
        if (candidate / "overall_status.json").is_file()
    ]
    if len(payload_candidates) != 1:
        raise ValueError("Spec 0043 output payload root is ambiguous")
    payload = payload_candidates[0]
    overall = _read_object(payload / "overall_status.json")
    run_contract = _read_object(payload / "run_contract.json")
    runtime = _read_object(payload / "runtime.json")
    expected_branches = cast("Mapping[str, object]", expected_contract["branches"])
    expected_test = cast("Mapping[str, object]", expected_contract["test"])
    if (
        expected_test.get("row_count") != TEST_PATCH_COUNT
        or expected_test.get("wsi_count") != TEST_WSI_COUNT
        or expected_test.get("label_fields_present") is not False
        or expected_test.get("class_order_local_only") is not False
        or any(
            set(cast("Mapping[str, object]", expected_branches[branch]))
            != {str(value) for value in BUDGETS}
            for branch in BRANCHES
        )
    ):
        raise ValueError("Spec 0043 expected test/model contract differs")
    if (
        set(run_contract)
        != {
            "schema_version",
            "input_contract_sha256",
            "input_dataset_reference",
            "spec_sha256",
            "scorer_sha256",
            "test_vector_sha256",
            "branches",
            "test",
            "runtime",
            "optimizer_updates",
        }
        or run_contract.get("schema_version") != "spec0043.remote_run_contract.v1"
        or run_contract.get("input_contract_sha256") != expected_input_contract_sha256
        or run_contract.get("input_dataset_reference") != dataset_reference
        or run_contract.get("spec_sha256") != expected_contract["spec_sha256"]
        or run_contract.get("scorer_sha256") != expected_contract["scorer_sha256"]
        or run_contract.get("test_vector_sha256")
        != expected_contract["test_vector_sha256"]
        or run_contract.get("branches") != dict(expected_branches)
        or run_contract.get("test") != dict(expected_test)
        or run_contract.get("runtime") != runtime
        or run_contract.get("optimizer_updates") != 0
    ):
        raise ValueError("Spec 0043 remote run contract differs")
    _validate_remote_runtime(runtime)

    observed_payload_artifacts = {
        path.relative_to(payload).as_posix(): _file_record(path)
        for path in sorted(payload.rglob("*"))
        if path.is_file() and path.name != "overall_status.json"
    }
    if (
        set(overall)
        != {
            "schema_version",
            "status",
            "branches",
            "budgets_per_class",
            "optimizer_updates",
            "artifacts",
        }
        or overall.get("schema_version") != "spec0043.remote_status.v1"
        or overall.get("status") != "complete"
        or overall.get("branches") != list(BRANCHES)
        or overall.get("budgets_per_class") != list(BUDGETS)
        or overall.get("optimizer_updates") != 0
        or overall.get("artifacts") != observed_payload_artifacts
    ):
        raise ValueError("Spec 0043 remote completion differs")

    prediction_files: dict[str, dict[int, Path]] = {}
    identity_sha256: set[str] = set()
    for branch in BRANCHES:
        worker_root = payload / "branches" / branch
        status = _read_object(worker_root / "worker_status.json")
        prediction_records = cast(
            "Mapping[str, Mapping[str, object]]",
            status.get("predictions"),
        )
        if (
            set(status)
            != {
                "schema_version",
                "status",
                "branch",
                "optimizer_updates",
                "batch_size",
                "final_batch_size",
                "row_identity_sha256",
                "access_by_part",
                "load_seconds",
                "transfer_seconds",
                "predictions",
            }
            or status.get("schema_version") != "spec0043.worker_status.v1"
            or status.get("status") != "complete"
            or status.get("branch") != branch
            or status.get("optimizer_updates") != 0
            or status.get("batch_size") != 159
            or status.get("final_batch_size") != TEST_PATCH_COUNT % 159
            or status.get("row_identity_sha256") != expected_test["row_identity_sha256"]
            or set(prediction_records) != {str(value) for value in BUDGETS}
            or not _finite_nonnegative(status.get("load_seconds"))
            or not _finite_nonnegative(status.get("transfer_seconds"))
        ):
            raise ValueError(f"Spec 0043 worker status differs: {branch}")
        prediction_files[branch] = {}
        reference_access: dict[str, int] | None = None
        for budget in BUDGETS:
            expected_relative = f"budget_{budget:04d}_per_class/predictions.csv.gz"
            path = worker_root / expected_relative
            rows, evidence = _prediction_file_evidence(path)
            if (
                _contract_row_identity_sha256(rows)
                != expected_test["row_identity_sha256"]
            ):
                raise ValueError("Spec 0043 contract row identity differs")
            record = prediction_records[str(budget)]
            if (
                set(record)
                != {
                    "path",
                    "file",
                    "row_count",
                    "identity_sha256",
                    "content_sha256",
                    "checkpoint",
                    "compute_seconds",
                }
                or record.get("path") != expected_relative
                or record.get("file") != _file_record(path)
                or any(record.get(key) != value for key, value in evidence.items())
                or record.get("checkpoint")
                != cast("Mapping[str, object]", expected_branches[branch])[str(budget)]
                or not _finite_nonnegative(record.get("compute_seconds"))
            ):
                raise ValueError(
                    f"Spec 0043 prediction evidence differs: {branch}/{budget}",
                )
            identity_sha256.add(str(evidence["identity_sha256"]))
            access = dict(
                Counter(str(int(row["part"])) for row in rows),
            )
            if reference_access is None:
                reference_access = access
            elif access != reference_access:
                raise ValueError("Spec 0043 within-worker physical access differs")
            prediction_files[branch][budget] = path
        if status.get("access_by_part") != reference_access:
            raise ValueError(f"Spec 0043 worker access counts differ: {branch}")
    if len(identity_sha256) != 1:
        raise ValueError("Spec 0043 prediction logical identities differ")
    return {"payload_root": payload, "prediction_files": prediction_files}


def _validate_remote_runtime(runtime: Mapping[str, object]) -> None:
    torch_version = runtime.get("torch")
    if (
        set(runtime)
        != {"torch", "cuda", "cudnn", "devices", "capabilities", "driver_versions"}
        or not isinstance(torch_version, str)
        or torch_version.split("+", maxsplit=1)[0] != "2.14.0"
        or runtime.get("cuda") != "13.0"
        or not isinstance(runtime.get("cudnn"), int)
        or not _valid_string_pair(runtime.get("devices"), required_substring="T4")
        or not _valid_capability_pair(runtime.get("capabilities"))
        or not _valid_string_pair(runtime.get("driver_versions"))
    ):
        raise ValueError("Spec 0043 remote runtime differs")


def _valid_string_pair(
    value: object,
    *,
    required_substring: str | None = None,
) -> bool:
    if not isinstance(value, list) or len(value) != 2:
        return False
    strings = cast("list[object]", value)
    return all(
        isinstance(item, str)
        and bool(item)
        and (required_substring is None or required_substring in item)
        for item in strings
    )


def _valid_capability_pair(value: object) -> bool:
    if not isinstance(value, list) or len(value) != 2:
        return False
    capabilities = cast("list[object]", value)
    for item in capabilities:
        if not isinstance(item, list) or len(item) != 2:
            return False
        parts = cast("list[object]", item)
        if not all(
            isinstance(part, int) and not isinstance(part, bool) for part in parts
        ):
            return False
    return True


def _prediction_file_evidence(
    path: Path,
) -> tuple[list[dict[str, object]], dict[str, int | str]]:
    raw_identities: list[LogicalIdentity] = []
    content = hashlib.sha256(b"eqvae_spec0043_prediction_rows_v1")
    with gzip.open(path, mode="rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != PREDICTION_HEADER:
            raise ValueError("Spec 0043 prediction header differs")
        for raw in reader:
            prediction = _parse_int(raw["prediction"], "prediction")
            identity = cast(
                "LogicalIdentity",
                tuple(_parse_int(raw[name], name) for name in IDENTITY_FIELDS),
            )
            raw_identities.append(identity)
            content.update(_canonical_json({**raw, "prediction": prediction}))
            content.update(b"\n")
    rows = load_prediction_csv_gz(path)
    if raw_identities != [_identity(row) for row in rows]:
        raise ValueError("Spec 0043 prediction evidence identity differs")
    return rows, {
        "row_count": len(rows),
        "identity_sha256": _json_sha256(raw_identities),
        "content_sha256": content.hexdigest(),
    }


def _contract_row_identity_sha256(
    rows: Sequence[Mapping[str, object]],
) -> str:
    digest = hashlib.sha256(ROW_IDENTITY_SCHEMA)
    field_order = (
        "dataset_row",
        "atlas_row_index",
        "wsi_id",
        "x",
        "y",
        "split",
        "part",
        "file_index",
    )
    for row in rows:
        logical_row = {**dict(row), "split": "test"}
        for name in field_order:
            digest.update(str(logical_row[name]).encode())
            digest.update(b"\0")
        digest.update(b"\n")
    return digest.hexdigest()


def _finite_nonnegative(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _label_efficiency_table(scored: Mapping[str, object]) -> dict[str, object]:
    branches = cast("Mapping[str, Mapping[str, object]]", scored["branches"])
    normal_branch = branches["normal_vae"]
    so2_branch = branches["so2_vae"]
    normal_budgets = cast("Mapping[str, object]", normal_branch["budgets"])
    so2_budgets = cast("Mapping[str, object]", so2_branch["budgets"])
    paired = cast(
        "Mapping[str, object]",
        scored["paired_normal_minus_so2"],
    )
    paired_budgets = cast("Mapping[str, object]", paired["budgets"])
    simultaneous = cast(
        "Mapping[str, object]",
        paired["six_contrast_simultaneous_95"],
    )
    contrasts = cast("Mapping[str, object]", simultaneous["contrasts"])
    rows = []
    for budget in BUDGETS:
        key = str(budget)
        rows.append({
            "labels_per_class": budget,
            "normal_vae": normal_budgets[key],
            "so2_vae": so2_budgets[key],
            "normal_minus_so2": paired_budgets[key],
            "primary_simultaneous_95": contrasts[f"macro_f1_budget_{budget}"],
        })
    return {
        "schema_version": "spec0043.label_efficiency_table.v1",
        "direction": "normal_minus_so2",
        "rows": rows,
        "aulc": {
            "normal_vae": normal_branch["aulc"],
            "so2_vae": so2_branch["aulc"],
            "normal_minus_so2": paired["aulc"],
            "primary_simultaneous_95": contrasts["aulc"],
        },
    }


def _write_joined_predictions(
    path: Path,
    *,
    rows: Sequence[Mapping[str, object]],
    oracle_rows: Sequence[Mapping[str, object]],
) -> None:
    if len(rows) != len(oracle_rows):
        raise ValueError("Joined prediction/oracle row count differs")
    logits = _logits(rows)
    truths = np.asarray(
        [CLASS_ORDER.index(str(row["tissue_label"])) for row in oracle_rows],
        dtype=np.int64,
    )
    losses = _cross_entropies(logits, truths)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (*PREDICTION_HEADER, "tissue_label", "truth", "cross_entropy")
    with gzip.open(
        path,
        mode="wt",
        newline="",
        encoding="utf-8",
        compresslevel=6,
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for row, oracle, truth, loss in zip(
            rows,
            oracle_rows,
            truths.tolist(),
            losses.tolist(),
            strict=True,
        ):
            if _identity(row) != _identity(oracle):
                raise ValueError("Joined prediction/oracle logical identity differs")
            writer.writerow({
                **dict(row),
                "tissue_label": oracle["tissue_label"],
                "truth": truth,
                "cross_entropy": loss,
            })


def _validate_prediction_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    expected_count: int,
) -> list[dict[str, object]]:
    if len(rows) != expected_count:
        raise ValueError(
            f"Prediction row count differs: {len(rows)} != {expected_count}",
        )
    normalized: list[dict[str, object]] = []
    identities: set[LogicalIdentity] = set()
    for offset, row in enumerate(rows):
        if set(row) != PREDICTION_FIELDS:
            raise ValueError(
                "Remote prediction row schema is not the label-free allow-list",
            )
        identity_values = {
            field: _require_int(row[field], field) for field in IDENTITY_FIELDS
        }
        if identity_values["dataset_row"] != offset:
            raise ValueError("Remote prediction canonical row order differs")
        if (
            identity_values["wsi_id"] <= 0
            or identity_values["x"] < 0
            or identity_values["y"] < 0
            or identity_values["part"] <= 0
            or identity_values["file_index"] < 0
        ):
            raise ValueError("Remote prediction logical identity is invalid")
        logits = [
            _require_float(row[field], field)
            for field in ("logit_tumor", "logit_stroma", "logit_necrosis")
        ]
        prediction = _require_int(row["prediction"], "prediction")
        if not 0 <= prediction < len(CLASS_ORDER):
            raise ValueError("Remote prediction class index differs")
        if prediction != int(np.argmax(np.asarray(logits, dtype=np.float64))):
            raise ValueError("Remote prediction does not equal canonical argmax")
        normalized_row: dict[str, object] = {
            **identity_values,
            **dict(
                zip(
                    ("logit_tumor", "logit_stroma", "logit_necrosis"),
                    logits,
                    strict=True,
                ),
            ),
            "prediction": prediction,
        }
        identity = _identity(normalized_row)
        if identity in identities:
            raise ValueError("Remote prediction logical identity is duplicated")
        identities.add(identity)
        normalized.append(normalized_row)
    return normalized


def _validate_oracle_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    expected_patch_count: int,
    expected_patch_support: Mapping[str, int] | None,
    expected_wsi_support: Mapping[str, int],
    expected_strata: Mapping[tuple[int, ...], int],
) -> tuple[
    list[dict[str, object]],
    list[int],
    tuple[int, ...],
    tuple[tuple[int, ...], ...],
]:
    if len(rows) != expected_patch_count:
        raise ValueError("Tissue oracle patch count differs")
    normalized: list[dict[str, object]] = []
    truths: list[int] = []
    identities: set[LogicalIdentity] = set()
    supports_by_wsi: dict[int, set[int]] = {}
    patch_support = dict.fromkeys(CLASS_ORDER, 0)
    for offset, row in enumerate(rows):
        if set(row) != ORACLE_FIELDS:
            raise ValueError("Tissue oracle row schema differs")
        identity_values = {
            field: _require_int(row[field], field) for field in IDENTITY_FIELDS
        }
        if identity_values["dataset_row"] != offset or row["split"] != "test":
            raise ValueError("Tissue oracle split/order differs")
        if row["selection_rank"] not in {None, ""}:
            raise ValueError("Tissue test selection rank must be empty")
        tissue = row["tissue_label"]
        if not isinstance(tissue, str) or tissue not in CLASS_ORDER:
            raise ValueError("Tissue oracle class differs")
        truth = CLASS_ORDER.index(tissue)
        normalized_row: dict[str, object] = {
            **identity_values,
            "tissue_label": tissue,
            "split": "test",
            "selection_rank": "",
        }
        identity = _identity(normalized_row)
        if identity in identities:
            raise ValueError("Tissue oracle logical identity is duplicated")
        identities.add(identity)
        normalized.append(normalized_row)
        truths.append(truth)
        patch_support[tissue] += 1
        supports_by_wsi.setdefault(identity_values["wsi_id"], set()).add(truth)
    wsi_ids = tuple(supports_by_wsi)
    if len(wsi_ids) != TEST_WSI_COUNT or tuple(sorted(wsi_ids)) != wsi_ids:
        raise ValueError("Tissue oracle WSI identity/count/order differs")
    observed_wsi_support = {
        tissue: sum(index in support for support in supports_by_wsi.values())
        for index, tissue in enumerate(CLASS_ORDER)
    }
    if observed_wsi_support != dict(expected_wsi_support):
        raise ValueError("Tissue oracle per-class WSI support differs")
    if expected_patch_support is not None and patch_support != dict(
        expected_patch_support,
    ):
        raise ValueError("Tissue oracle patch support differs")
    strata: dict[tuple[int, ...], list[int]] = {}
    for index, wsi_id in enumerate(wsi_ids):
        key = tuple(sorted(supports_by_wsi[wsi_id]))
        strata.setdefault(key, []).append(index)
    if {key: len(value) for key, value in strata.items()} != dict(expected_strata):
        raise ValueError("Tissue-support WSI strata differ")
    return (
        normalized,
        truths,
        wsi_ids,
        tuple(tuple(strata[key]) for key in expected_strata),
    )


def _bootstrap_multiplicities(
    *,
    wsi_count: int,
    stratum_indices: Sequence[Sequence[int]],
    replicates: int,
    seed: int,
) -> IntArray:
    rng = np.random.default_rng(seed)
    multiplicities = np.zeros((replicates, wsi_count), dtype=np.int64)
    replicate_indices = np.arange(replicates)
    for stratum in stratum_indices:
        indices = np.asarray(stratum, dtype=np.int64)
        sampled = rng.choice(indices, size=(replicates, len(indices)), replace=True)
        for column in range(sampled.shape[1]):
            np.add.at(
                multiplicities,
                (replicate_indices, sampled[:, column]),
                1,
            )
    if not bool((multiplicities.sum(axis=1) == wsi_count).all()):
        raise RuntimeError("Bootstrap WSI multiplicities differ")
    return multiplicities


def _bootstrap_metric_arrays(
    *,
    truths: IntArray,
    predictions: IntArray,
    losses: FloatArray,
    wsi_indices: IntArray,
    multiplicities: IntArray,
    wsi_count: int,
) -> dict[str, FloatArray]:
    wsi_confusion = np.zeros((wsi_count, 3, 3), dtype=np.int64)
    np.add.at(wsi_confusion, (wsi_indices, truths, predictions), 1)
    wsi_loss = np.bincount(
        wsi_indices,
        weights=losses,
        minlength=wsi_count,
    ).astype(np.float64, copy=False)
    confusion = np.einsum("rw,wab->rab", multiplicities, wsi_confusion)
    loss_sums = multiplicities @ wsi_loss
    return _metric_arrays_from_confusion(confusion, loss_sums)


def _metric_arrays_from_confusion(
    confusion: IntArray,
    loss_sums: FloatArray,
) -> dict[str, FloatArray]:
    true_positive = np.diagonal(confusion, axis1=1, axis2=2).astype(np.float64)
    support = confusion.sum(axis=2).astype(np.float64)
    predicted = confusion.sum(axis=1).astype(np.float64)
    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros_like(true_positive),
        where=predicted != 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros_like(true_positive),
        where=support != 0,
    )
    denominator = precision + recall
    f1 = np.divide(
        2.0 * precision * recall,
        denominator,
        out=np.zeros_like(true_positive),
        where=denominator != 0,
    )
    total = confusion.sum(axis=(1, 2)).astype(np.float64)
    result: dict[str, FloatArray] = {
        "macro_f1": f1.mean(axis=1),
        "balanced_accuracy": recall.mean(axis=1),
        "accuracy": true_positive.sum(axis=1) / total,
        "mean_cross_entropy": loss_sums / total,
    }
    for index, tissue in enumerate(CLASS_ORDER):
        result[f"{tissue}.precision"] = precision[:, index]
        result[f"{tissue}.recall"] = recall[:, index]
        result[f"{tissue}.f1"] = f1[:, index]
    return result


def _metrics(
    truths: IntArray,
    predictions: IntArray,
    losses: FloatArray,
    wsi_indices: IntArray,
    wsi_ids: Sequence[int],
) -> dict[str, object]:
    confusion = np.zeros((3, 3), dtype=np.int64)
    np.add.at(confusion, (truths, predictions), 1)
    arrays = _metric_arrays_from_confusion(
        confusion[np.newaxis, :, :],
        np.asarray([losses.sum()], dtype=np.float64),
    )
    per_class = []
    for index, tissue in enumerate(CLASS_ORDER):
        per_class.append({
            "tissue": tissue,
            "precision": float(arrays[f"{tissue}.precision"][0]),
            "recall": float(arrays[f"{tissue}.recall"][0]),
            "f1": float(arrays[f"{tissue}.f1"][0]),
            "support": int(confusion[index].sum()),
        })
    return {
        **{name: float(arrays[name][0]) for name in SCALAR_METRICS},
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "n_patch": len(truths),
        "n_wsi": len(wsi_ids),
        "wsi_count_per_class": _wsi_class_counts(truths, wsi_indices, wsi_ids),
    }


def _wsi_class_counts(
    truths: IntArray,
    wsi_indices: IntArray,
    wsi_ids: Sequence[int],
) -> dict[str, int]:
    if len(wsi_ids) != TEST_WSI_COUNT:
        raise ValueError("WSI support count differs")
    return {
        tissue: len(set(wsi_indices[truths == index].tolist()))
        for index, tissue in enumerate(CLASS_ORDER)
    }


def _metric_differences(
    normal: Mapping[str, object],
    so2: Mapping[str, object],
) -> dict[str, float]:
    result = {name: float(normal[name]) - float(so2[name]) for name in SCALAR_METRICS}
    normal_per_class = cast("Sequence[Mapping[str, object]]", normal["per_class"])
    so2_per_class = cast("Sequence[Mapping[str, object]]", so2["per_class"])
    for normal_row, so2_row, tissue in zip(
        normal_per_class,
        so2_per_class,
        CLASS_ORDER,
        strict=True,
    ):
        if normal_row["tissue"] != tissue or so2_row["tissue"] != tissue:
            raise RuntimeError("Per-class metric order differs")
        for metric in ("precision", "recall", "f1"):
            result[f"{tissue}.{metric}"] = float(normal_row[metric]) - float(
                so2_row[metric],
            )
    return result


def _metric_intervals(samples: Mapping[str, FloatArray]) -> dict[str, object]:
    return _nest_flat_metrics({
        name: _interval(values) for name, values in samples.items()
    })


def _nest_flat_metrics(values: Mapping[str, object]) -> dict[str, object]:
    result: dict[str, object] = {
        name: values[name] for name in SCALAR_METRICS if name in values
    }
    result["per_class"] = {
        tissue: {
            metric: values[f"{tissue}.{metric}"]
            for metric in ("precision", "recall", "f1")
        }
        for tissue in CLASS_ORDER
    }
    return result


def _interval(values: FloatArray) -> dict[str, float]:
    lower, upper = np.quantile(values, (0.025, 0.975), method="linear")
    return {"lower": float(lower), "upper": float(upper)}


def _aulc(curves: FloatArray) -> FloatArray:
    x = np.log10(np.asarray(BUDGETS, dtype=np.float64))
    return np.trapezoid(curves, x=x, axis=1) / (x[-1] - x[0])


def _logits(rows: Sequence[Mapping[str, object]]) -> FloatArray:
    return np.asarray(
        [
            [row["logit_tumor"], row["logit_stroma"], row["logit_necrosis"]]
            for row in rows
        ],
        dtype=np.float64,
    )


def _cross_entropies(logits: FloatArray, truths: IntArray) -> FloatArray:
    maximum = logits.max(axis=1)
    return (
        np.log(np.exp(logits - maximum[:, np.newaxis]).sum(axis=1))
        + maximum
        - logits[np.arange(len(truths)), truths]
    )


def _identity(row: Mapping[str, object]) -> LogicalIdentity:
    return cast("LogicalIdentity", tuple(int(row[field]) for field in IDENTITY_FIELDS))


def _build_vector_fixture(
    fixture: Mapping[str, object],
) -> tuple[dict[str, dict[int, list[dict[str, object]]]], list[dict[str, object]]]:
    supports = cast("Sequence[Sequence[object]]", fixture.get("wsi_truths"))
    normal_moduli = cast("Sequence[object]", fixture.get("normal_error_moduli"))
    so2_moduli = cast("Sequence[object]", fixture.get("so2_error_moduli"))
    if (
        len(supports) != TEST_WSI_COUNT
        or len(normal_moduli) != 5
        or len(so2_moduli) != 5
    ):
        raise ValueError("Spec 0043 scorer vector fixture shape differs")
    oracle_rows: list[dict[str, object]] = []
    for wsi_offset, tissue_indices in enumerate(supports, start=1):
        for tissue_index_value in tissue_indices:
            tissue_index = int(tissue_index_value)
            dataset_row = len(oracle_rows)
            oracle_rows.append({
                "dataset_row": dataset_row,
                "atlas_row_index": 10_000 + dataset_row,
                "wsi_id": wsi_offset,
                "x": dataset_row * 256,
                "y": wsi_offset * 256,
                "tissue_label": CLASS_ORDER[tissue_index],
                "split": "test",
                "selection_rank": "",
                "part": 1,
                "file_index": dataset_row,
            })
    predictions: dict[str, dict[int, list[dict[str, object]]]] = {
        branch: {} for branch in BRANCHES
    }
    for branch, moduli in zip(BRANCHES, (normal_moduli, so2_moduli), strict=True):
        for budget_offset, (budget, modulus_value) in enumerate(
            zip(BUDGETS, moduli, strict=True),
        ):
            modulus = int(modulus_value)
            if modulus < 2:
                raise ValueError("Spec 0043 vector error modulus differs")
            rows: list[dict[str, object]] = []
            for oracle in oracle_rows:
                truth = CLASS_ORDER.index(str(oracle["tissue_label"]))
                row_index = int(oracle["dataset_row"])
                error = (
                    row_index + 2 * budget_offset + (branch == "so2_vae")
                ) % modulus == 0
                prediction = (truth + 1) % 3 if error else truth
                logits = [-1.0, -1.0, -1.0]
                logits[prediction] = 2.0
                rows.append({
                    **{field: oracle[field] for field in IDENTITY_FIELDS},
                    "logit_tumor": logits[0],
                    "logit_stroma": logits[1],
                    "logit_necrosis": logits[2],
                    "prediction": prediction,
                })
            predictions[branch][budget] = rows
    return predictions, oracle_rows


def _require_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    return value


def _require_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _parse_int(value: str, field: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{field} must be an integer") from error
    if str(parsed) != value:
        raise ValueError(f"{field} integer encoding differs")
    return parsed


def _parse_float(value: str, field: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{field} must be numeric") from error
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    return parsed


def _repo_root(path: Path) -> Path:
    for parent in path.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise ValueError("Cannot resolve repository root for scorer")


def _required_string(value: Mapping[str, object], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise TypeError(f"{key} must be a non-empty string")
    return result


def _read_object(path: Path) -> dict[str, object]:
    value = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return cast("dict[str, object]", value)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(_canonical_json(value) + b"\n")
    temporary.replace(path)
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _file_record(path: Path) -> dict[str, int | str]:
    return {"bytes": path.stat().st_size, "sha256": _sha256(path)}


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "BRANCHES",
    "BUDGETS",
    "CLASS_ORDER",
    "load_label_oracle",
    "load_prediction_csv_gz",
    "score_retrieved_tissue_test_output",
    "score_tissue_test_predictions",
    "verify_test_vector",
    "write_exclusive_json",
]
