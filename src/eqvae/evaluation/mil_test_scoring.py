# Copyright 2026 HiperMaximus
# pyright: reportArgumentType=false, reportIndexIssue=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false
# ruff: noqa: C901, DOC201, DOC501, EM101, EM102, PLR0913, PLR0914, PLR0916, PLR2004, PLW0717, TRY003
"""Post-retrieval scorer for the label-blind Spec 0041 MIL inference."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
from dataclasses import asdict
from itertools import starmap
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

from eqvae.models.local_global_mil import CLASS_ORDER
from eqvae.training.mil_training import (
    compute_validation_metrics,
    diagnosis_stratified_paired_bootstrap,
    predictions_from_logits,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

LABEL_ORACLE_SHA256: Final = (
    "216f69f64ed7a3e5636173d6cfe83297632113bc68310e4f6285d7ffc22cd43c"
)
TEST_VECTOR_PATH: Final = Path("docs/data/spec0041_mil_test_scorer_vector.json")
TEST_WSI_COUNT: Final = 23
EXPECTED_SUPPORT: Final = {"CC": 5, "EC": 6, "HGSC": 8, "LGSC": 2, "MC": 2}
BRANCHES: Final = ("normal_vae", "so2_vae")
FORBIDDEN_REMOTE_FIELDS: Final = frozenset({
    "truth",
    "diagnosis",
    "diagnosis_label",
    "diagnosis_index",
    "target",
    "cross_entropy",
    "loss",
})


def score_mil_test_predictions(
    *,
    normal_rows: Sequence[Mapping[str, object]],
    so2_rows: Sequence[Mapping[str, object]],
    labels: Mapping[int, tuple[str, int]],
    replicates: int = 10_000,
    seed: int = 3601,
) -> dict[str, object]:
    """Score two already-frozen aligned prediction tables."""
    normal = _validate_prediction_rows(normal_rows)
    so2 = _validate_prediction_rows(so2_rows)
    normal_ids = [int(row["wsi_id"]) for row in normal]
    so2_ids = [int(row["wsi_id"]) for row in so2]
    if normal_ids != so2_ids or set(normal_ids) != set(labels):
        raise ValueError("Prediction and label WSI identities differ")
    stable_fields = ("wsi_id", "bag_size", "graph_identity", "graph_degree")
    if any(
        any(left[field] != right[field] for field in stable_fields)
        for left, right in zip(normal, so2, strict=True)
    ):
        raise ValueError("Paired prediction geometry differs")

    truths = [labels[wsi_id][1] for wsi_id in normal_ids]
    branch_outputs: dict[str, object] = {}
    losses_by_branch: dict[str, list[float]] = {}
    predictions_by_branch: dict[str, tuple[int, ...]] = {}
    for branch, rows in zip(BRANCHES, (normal, so2), strict=True):
        logits = [cast("list[float]", row["logits"]) for row in rows]
        predictions = predictions_from_logits(logits)
        if predictions != tuple(int(row["prediction"]) for row in rows):
            raise ValueError(f"Remote argmax differs for {branch}")
        losses = list(starmap(_cross_entropy, zip(logits, truths, strict=True)))
        metrics = compute_validation_metrics(truths, predictions, losses)
        scored_rows = [
            {
                **dict(row),
                "truth": truth,
                "diagnosis": labels[int(row["wsi_id"])][0],
                "cross_entropy": loss,
            }
            for row, truth, loss in zip(rows, truths, losses, strict=True)
        ]
        predictions_by_branch[branch] = predictions
        losses_by_branch[branch] = losses
        branch_outputs[branch] = {
            "schema_version": "spec0041.scored_branch.v1",
            "branch": branch,
            "metrics": metrics.to_dict(),
            "rows": scored_rows,
            "rows_sha256": _json_sha256(scored_rows),
        }
    paired = diagnosis_stratified_paired_bootstrap(
        truths,
        predictions_by_branch["normal_vae"],
        predictions_by_branch["so2_vae"],
        normal_cross_entropies=losses_by_branch["normal_vae"],
        so2_cross_entropies=losses_by_branch["so2_vae"],
        replicates=replicates,
        seed=seed,
    )
    return {
        "schema_version": "spec0041.scored_result.v1",
        "class_order": list(CLASS_ORDER),
        "wsi_count": len(truths),
        "branches": branch_outputs,
        "paired_normal_minus_so2": {
            key: asdict(value) for key, value in paired.items()
        },
        "bootstrap": {
            "replicates": replicates,
            "seed": seed,
            "confidence": 0.95,
            "sampling_unit": "WSI",
            "stratified_by": "diagnosis",
            "covers_training_seed_uncertainty": False,
        },
        "limitations": [
            "One fixed 23-WSI test split and one classifier initialization.",
            "LGSC and MC each have support 2.",
            "Bootstrap intervals cover WSI-sampling, not training-seed uncertainty.",
        ],
    }


def score_retrieved_mil_test_output(
    *,
    remote_output_root: Path,
    launch_receipt_path: Path,
    launch_claim_path: Path,
    label_oracle_path: Path,
    output_root: Path,
    expected_scorer_sha256: str,
    expected_test_vector_sha256: str,
    expected_input_contract_sha256: str,
    expected_dataset_reference: str,
    expected_spec_sha256: str,
    expected_branches: Mapping[str, object],
    expected_test: Mapping[str, object],
    expected_kernel_sources: Sequence[str],
    expected_kernel_sha256: str,
    expected_metadata_sha256: str,
    expected_kernel_bytes: int,
    expected_metadata_bytes: int,
    normalization_amendment_path: Path,
    expected_normalization_amendment_sha256: str,
) -> dict[str, object]:
    """Authenticate label-blind output, freeze a pre-score claim, then score once."""
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite {output_root}")
    scorer_path = Path(__file__).resolve()
    amendment = _validate_normalization_amendment(
        path=normalization_amendment_path,
        expected_sha256=expected_normalization_amendment_sha256,
        current_scorer_sha256=_sha256(scorer_path),
        remote_scorer_sha256=expected_scorer_sha256,
        launch_receipt_sha256=_sha256(launch_receipt_path),
    )
    vector_path = _repo_root(scorer_path) / TEST_VECTOR_PATH
    if _sha256(vector_path) != expected_test_vector_sha256:
        raise ValueError("Scorer test vector differs from the pre-launch hash")
    _verify_test_vector(vector_path)
    remote = _authenticate_remote_output(
        remote_output_root=remote_output_root,
        launch_receipt_path=launch_receipt_path,
        launch_claim_path=launch_claim_path,
        expected_scorer_sha256=expected_scorer_sha256,
        expected_test_vector_sha256=expected_test_vector_sha256,
        expected_input_contract_sha256=expected_input_contract_sha256,
        expected_dataset_reference=expected_dataset_reference,
        expected_spec_sha256=expected_spec_sha256,
        expected_branches=expected_branches,
        expected_test=expected_test,
        expected_kernel_sources=expected_kernel_sources,
        expected_kernel_sha256=expected_kernel_sha256,
        expected_metadata_sha256=expected_metadata_sha256,
        expected_kernel_bytes=expected_kernel_bytes,
        expected_metadata_bytes=expected_metadata_bytes,
        expected_accepted_kernel_reference=str(amendment["accepted_kernel_reference"]),
    )

    staging = output_root.with_name(f".{output_root.name}.scoring")
    if staging.exists():
        raise FileExistsError(f"Stale scoring directory exists: {staging}")
    claim_path = output_root.with_name(f"{output_root.name}.pre_score_claim.json")
    pre_score = {
        "schema_version": "spec0041.pre_score_contract.v1",
        "remote_launch_receipt_sha256": _sha256(launch_receipt_path),
        "exclusive_launch_claim_sha256": _sha256(launch_claim_path),
        "remote_output_receipt_sha256": _sha256(
            remote_output_root / "kaggle_output_receipt.json",
        ),
        "remote_overall_status_sha256": _sha256(remote["overall_path"]),
        "remote_prediction_sha256": remote["prediction_sha256"],
        "label_oracle_sha256": LABEL_ORACLE_SHA256,
        "scorer_sha256": expected_scorer_sha256,
        "amended_local_scorer_sha256": _sha256(scorer_path),
        "normalization_amendment_sha256": expected_normalization_amendment_sha256,
        "test_vector_sha256": expected_test_vector_sha256,
        "policy": "no_remote_retry_after_this_contract",
    }
    _write_exclusive_json(claim_path, pre_score)
    staging.mkdir(parents=True)
    try:
        _write_json(staging / "pre_score_contract.json", pre_score)
        # This is the first operation that parses the sealed diagnosis oracle.
        # The immutable prediction/evaluator claim above therefore already exists.
        labels = _load_labels(label_oracle_path)
        scored = score_mil_test_predictions(
            normal_rows=remote["rows"]["normal_vae"],
            so2_rows=remote["rows"]["so2_vae"],
            labels=labels,
        )
        for branch in BRANCHES:
            _write_json(
                staging / branch / "scored_predictions_and_metrics.json",
                cast("dict[str, object]", scored["branches"])[branch],
            )
        _write_json(
            staging / "paired_bootstrap.json",
            scored["paired_normal_minus_so2"],
        )
        artifact_files = {
            path.relative_to(staging).as_posix(): _file_record(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        completion = {
            "schema_version": "spec0041.scoring_completion.v1",
            "status": "complete",
            "primary_metric": "macro_f1",
            "direction": "normal_minus_so2",
            "artifacts": artifact_files,
            "result_summary": {
                branch: cast(
                    "dict[str, object]",
                    cast("dict[str, object]", scored["branches"])[branch],
                )["metrics"]
                for branch in BRANCHES
            },
            "paired_normal_minus_so2": scored["paired_normal_minus_so2"],
            "bootstrap": scored["bootstrap"],
            "limitations": scored["limitations"],
        }
        _write_json(staging / "overall_status.json", completion)
        staging.replace(output_root)
    except BaseException:
        _remove_tree(staging)
        raise
    return completion


def _validate_prediction_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    if len(rows) != TEST_WSI_COUNT:
        raise ValueError("Each branch must contain exactly 23 prediction rows")
    normalized: list[dict[str, object]] = []
    previous = -1
    for row in rows:
        if FORBIDDEN_REMOTE_FIELDS & set(row):
            raise ValueError("Remote prediction row contains a label-dependent field")
        if set(row) != {
            "wsi_id",
            "prediction",
            "logits",
            "bag_size",
            "graph_identity",
            "graph_degree",
            "access",
        }:
            raise ValueError("Remote prediction row schema differs")
        wsi_id = int(row["wsi_id"])
        logits = [float(value) for value in cast("Sequence[object]", row["logits"])]
        if (
            wsi_id <= previous
            or len(logits) != len(CLASS_ORDER)
            or not all(math.isfinite(value) for value in logits)
        ):
            raise ValueError("Remote prediction row identity or logits differ")
        if int(row["bag_size"]) < 1 or not isinstance(row["graph_identity"], str):
            raise ValueError("Remote bag or graph identity differs")
        normalized.append({**dict(row), "wsi_id": wsi_id, "logits": logits})
        previous = wsi_id
    return normalized


def _load_labels(path: Path) -> dict[int, tuple[str, int]]:
    if _sha256(path) != LABEL_ORACLE_SHA256:
        raise ValueError("Test label oracle differs")
    labels: dict[int, tuple[str, int]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != (
            "wsi_id",
            "diagnosis_label",
            "diagnosis_index",
            "is_updated_image_id",
            "split",
        ):
            raise ValueError("Test label oracle header differs")
        for row in reader:
            if row["split"] != "test":
                continue
            diagnosis = row["diagnosis_label"]
            index = int(row["diagnosis_index"])
            wsi_id = int(row["wsi_id"])
            if diagnosis not in CLASS_ORDER or CLASS_ORDER[index] != diagnosis:
                raise ValueError("Test label mapping differs")
            labels[wsi_id] = (diagnosis, index)
    support = dict.fromkeys(CLASS_ORDER, 0)
    for diagnosis, _ in labels.values():
        support[diagnosis] += 1
    if len(labels) != TEST_WSI_COUNT or support != EXPECTED_SUPPORT:
        raise ValueError("Test label support differs")
    return labels


def _authenticate_remote_output(
    *,
    remote_output_root: Path,
    launch_receipt_path: Path,
    launch_claim_path: Path,
    expected_scorer_sha256: str,
    expected_test_vector_sha256: str,
    expected_input_contract_sha256: str,
    expected_dataset_reference: str,
    expected_spec_sha256: str,
    expected_branches: Mapping[str, object],
    expected_test: Mapping[str, object],
    expected_kernel_sources: Sequence[str],
    expected_kernel_sha256: str,
    expected_metadata_sha256: str,
    expected_kernel_bytes: int,
    expected_metadata_bytes: int,
    expected_accepted_kernel_reference: str,
) -> dict[str, object]:
    launch = _read_object(launch_receipt_path)
    claim = _read_object(launch_claim_path)
    reference = launch.get("kernel_reference")
    expected_actor = expected_dataset_reference.split("/", maxsplit=1)[0]
    expected_kernel_id = f"{expected_actor}/eqvae-local-global-mil-test-evaluation"
    accepted_parts = expected_accepted_kernel_reference.split("/")
    if len(accepted_parts) != 3 or not accepted_parts[2].isdigit():
        raise ValueError("Spec 0041 amended accepted reference differs")
    expected_accepted_kernel_id = "/".join(accepted_parts[:2])
    expected_accepted_version = int(accepted_parts[2])
    expected_sources = {
        "competition_sources": [],
        "dataset_sources": [expected_dataset_reference],
        "kernel_sources": list(expected_kernel_sources),
        "model_sources": [],
    }
    expected_files = {
        "kernel-metadata.json": {
            "bytes": expected_metadata_bytes,
            "sha256": expected_metadata_sha256,
        },
        "run.py": {
            "bytes": expected_kernel_bytes,
            "sha256": expected_kernel_sha256,
        },
    }
    accepted_version = launch.get("accepted_version")
    if (
        claim
        != {
            "schema_version": "spec0041.exclusive_launch_claim.v1",
            "authorization": "one_private_label_blind_test_inference_launch",
            "input_contract_sha256": expected_input_contract_sha256,
            "kernel_sha256": expected_kernel_sha256,
            "metadata_sha256": expected_metadata_sha256,
        }
        or launch.get("schema_version") != "eqvae.kaggle_kernel_launch.v1"
        or not isinstance(reference, str)
        or isinstance(accepted_version, bool)
        or not isinstance(accepted_version, int)
        or accepted_version < 1
        or launch.get("actor") != expected_actor
        or launch.get("original_kernel_id") != expected_kernel_id
        or launch.get("kernel_id") != expected_accepted_kernel_id
        or launch.get("requested_kernel_id") != expected_kernel_id
        or launch.get("source_locators") != expected_sources
        or launch.get("source_metadata_sha256") != expected_metadata_sha256
        or launch.get("upload_metadata_sha256") != expected_metadata_sha256
        or launch.get("source_files") != expected_files
        or launch.get("upload_files") != expected_files
        or accepted_version != expected_accepted_version
        or reference != expected_accepted_kernel_reference
    ):
        raise ValueError("Spec 0041 launch receipt differs")
    receipt_path = remote_output_root / "kaggle_output_receipt.json"
    receipt = _read_object(receipt_path)
    if (
        receipt.get("schema_version") != "eqvae.kaggle_download.v1"
        or receipt.get("resource_reference") != reference
    ):
        raise ValueError("Spec 0041 output receipt differs")
    declared = cast("dict[str, object]", receipt.get("files"))
    observed = {
        path.relative_to(remote_output_root).as_posix(): _file_record(path)
        for path in sorted(remote_output_root.rglob("*"))
        if path.is_file() and path != receipt_path
    }
    if declared != observed:
        raise ValueError("Spec 0041 downloaded output bytes differ")
    payload_roots = [
        candidate
        for candidate in (
            remote_output_root,
            remote_output_root / "spec0041_mil_test_inference",
        )
        if (candidate / "overall_status.json").is_file()
    ]
    if len(payload_roots) != 1:
        raise ValueError("Spec 0041 output payload root is ambiguous")
    payload = payload_roots[0]
    overall_path = payload / "overall_status.json"
    overall = _read_object(overall_path)
    run_contract = _read_object(payload / "run_contract.json")
    if (
        overall.get("status") != "complete"
        or run_contract.get("schema_version") != "spec0041.remote_run_contract.v1"
        or run_contract.get("input_contract_sha256") != expected_input_contract_sha256
        or run_contract.get("input_dataset_reference") != expected_dataset_reference
        or run_contract.get("spec_sha256") != expected_spec_sha256
        or run_contract.get("scorer_sha256") != expected_scorer_sha256
        or run_contract.get("test_vector_sha256") != expected_test_vector_sha256
        or run_contract.get("branches") != dict(expected_branches)
        or run_contract.get("test") != dict(expected_test)
        or run_contract.get("optimizer_updates") != 0
    ):
        raise ValueError("Spec 0041 remote completion contract differs")
    rows: dict[str, list[dict[str, object]]] = {}
    prediction_sha256: dict[str, str] = {}
    for branch in BRANCHES:
        path = payload / branch / "predictions.json"
        artifact = _read_object(path)
        branch_rows = cast("list[dict[str, object]]", artifact.get("rows"))
        if (
            artifact.get("branch") != branch
            or artifact.get(
                "rows_sha256",
            )
            != _json_sha256(branch_rows)
            or artifact.get("checkpoint")
            != cast("Mapping[str, object]", expected_branches[branch])["source"]
            or artifact.get("optimizer_updates") != 0
        ):
            raise ValueError(f"Spec 0041 remote prediction identity differs: {branch}")
        rows[branch] = branch_rows
        prediction_sha256[branch] = _sha256(path)
    expected_graphs = cast(
        "Mapping[str, object]",
        expected_test["graph_identities"],
    )
    expected_sizes = cast("Mapping[str, object]", expected_test["bag_sizes"])
    for branch_rows in rows.values():
        observed_ids = {str(row["wsi_id"]) for row in branch_rows}
        if observed_ids != set(expected_graphs) or any(
            row["graph_identity"] != expected_graphs[str(row["wsi_id"])]
            or row["bag_size"] != expected_sizes[str(row["wsi_id"])]
            for row in branch_rows
        ):
            raise ValueError("Spec 0041 remote test geometry differs")
    return {
        "overall_path": overall_path,
        "rows": rows,
        "prediction_sha256": prediction_sha256,
    }


def _validate_normalization_amendment(
    *,
    path: Path,
    expected_sha256: str,
    current_scorer_sha256: str,
    remote_scorer_sha256: str,
    launch_receipt_sha256: str,
) -> dict[str, object]:
    if _sha256(path) != expected_sha256:
        raise ValueError("Spec 0041 normalization amendment bytes differ")
    amendment = _read_object(path)
    if (
        amendment.get("schema_version")
        != "spec0042.spec0041_slug_normalization_amendment.v1"
        or amendment.get("scope") != "administrative_slug_normalization_only"
        or amendment.get("original_scorer_sha256") != remote_scorer_sha256
        or amendment.get("amended_scorer_sha256") != current_scorer_sha256
        or amendment.get("launch_receipt_sha256") != launch_receipt_sha256
        or amendment.get("requested_kernel_id")
        != "maximshtefan/eqvae-local-global-mil-test-evaluation"
        or amendment.get("accepted_kernel_reference")
        != "maximshtefan/eqvae-label-blind-mil-test-evaluation/1"
        or amendment.get("scientific_changes") != []
        or amendment.get("outputs_retrieved_before_amendment") is not False
    ):
        raise ValueError("Spec 0041 normalization amendment contract differs")
    return amendment


def _verify_test_vector(path: Path) -> None:
    vector = _read_object(path)
    labels = {
        int(key): (str(value[0]), int(value[1]))
        for key, value in cast(
            "dict[str, list[object]]",
            vector["labels"],
        ).items()
    }
    wsi_ids = sorted(labels)

    def rows(predictions: Sequence[object]) -> list[dict[str, object]]:
        built = []
        for wsi_id, prediction_value in zip(wsi_ids, predictions, strict=True):
            prediction = int(prediction_value)
            logits = [-2.0] * len(CLASS_ORDER)
            logits[prediction] = 2.0
            built.append({
                "wsi_id": wsi_id,
                "prediction": prediction,
                "logits": logits,
                "bag_size": wsi_id,
                "graph_identity": hashlib.sha256(str(wsi_id).encode()).hexdigest(),
                "graph_degree": {"minimum": 1, "maximum": 1, "mean": 1.0},
                "access": {"parts": {"1": wsi_id}},
            })
        return built

    result = score_mil_test_predictions(
        normal_rows=rows(cast("list[object]", vector["normal_predictions"])),
        so2_rows=rows(cast("list[object]", vector["so2_predictions"])),
        labels=labels,
        replicates=int(vector["replicates"]),
        seed=int(vector["seed"]),
    )
    if _json_sha256(result) != vector.get("expected_result_sha256"):
        raise ValueError("Spec 0041 scorer test vector result differs")


def _cross_entropy(logits: Sequence[float], truth: int) -> float:
    maximum = max(logits)
    return (
        math.log(sum(math.exp(value - maximum) for value in logits))
        + maximum
        - logits[truth]
    )


def _repo_root(path: Path) -> Path:
    for parent in path.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise ValueError("Cannot resolve repository root for scorer")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(_canonical_json(value) + b"\n")
    temporary.replace(path)
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _write_exclusive_json(path: Path, value: object) -> None:
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


def _read_object(path: Path) -> dict[str, object]:
    value = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return cast("dict[str, object]", value)


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


def _remove_tree(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


__all__ = ["score_mil_test_predictions", "score_retrieved_mil_test_output"]
