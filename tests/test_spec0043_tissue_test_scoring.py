# Copyright 2026 HiperMaximus
# pyright: reportAny=false, reportArgumentType=false, reportPrivateUsage=false, reportUnnecessaryCast=false
# ruff: noqa: D103, PLC2701, PLR0914, PLR0915, PLR2004
"""Focused tests for the frozen Spec 0043 tissue test scorer."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest

import eqvae.evaluation.tissue_test_scoring as scorer
from eqvae.evaluation.tissue_test_scoring import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    BUDGETS,
    EXPECTED_STRATA,
    _aulc,
    _bootstrap_multiplicities,
    _build_vector_fixture,
    _contract_row_identity_sha256,
    _file_record,
    _json_sha256,
    _prediction_file_evidence,
    _score_tissue_test_predictions,
    _sha256,
    _validate_prediction_rows,
    verify_test_vector,
    write_exclusive_json,
)


def _fixture() -> tuple[
    dict[str, dict[int, list[dict[str, object]]]],
    list[dict[str, object]],
]:
    vector = cast(
        "dict[str, object]",
        json.loads(
            Path("docs/data/spec0043_tissue_test_scorer_vector.json").read_text(
                encoding="utf-8",
            ),
        ),
    )
    return _build_vector_fixture(
        cast("dict[str, object]", vector["fixture"]),
    )


def _score(*, replicates: int = 250) -> dict[str, object]:
    predictions, oracle = _fixture()
    return _score_tissue_test_predictions(
        predictions=predictions,
        oracle_rows=oracle,
        replicates=replicates,
        seed=BOOTSTRAP_SEED,
        expected_patch_count=len(oracle),
        expected_patch_support=None,
        expected_wsi_support={"tumor": 23, "stroma": 21, "necrosis": 5},
        expected_strata=EXPECTED_STRATA,
    )


def test_frozen_scorer_vector_reproduces_exact_result() -> None:
    path = Path("docs/data/spec0043_tissue_test_scorer_vector.json")
    vector = cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
    verify_test_vector(path)
    predictions, oracle = _fixture()
    result = _score_tissue_test_predictions(
        predictions=predictions,
        oracle_rows=oracle,
        replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
        expected_patch_count=len(oracle),
        expected_patch_support=None,
        expected_wsi_support={"tumor": 23, "stroma": 21, "necrosis": 5},
        expected_strata=EXPECTED_STRATA,
    )
    assert _json_sha256(result) == vector["expected_result_sha256"]


def test_reports_pooled_metrics_all_budgets_and_normal_minus_so2() -> None:
    result = _score()
    branches = cast("dict[str, dict[str, object]]", result["branches"])
    paired = cast(
        "dict[str, object]",
        result["paired_normal_minus_so2"],
    )
    assert set(branches) == {"normal_vae", "so2_vae"}
    for branch in branches.values():
        budgets = cast("dict[str, dict[str, object]]", branch["budgets"])
        assert set(budgets) == {str(value) for value in BUDGETS}
        for budget in budgets.values():
            metrics = cast("dict[str, object]", budget["metrics"])
            assert metrics["n_patch"] == 49
            assert metrics["n_wsi"] == 23
            assert metrics["wsi_count_per_class"] == {
                "tumor": 23,
                "stroma": 21,
                "necrosis": 5,
            }
            assert len(cast("list[object]", metrics["confusion_matrix"])) == 3
    paired_budgets = cast("dict[str, dict[str, object]]", paired["budgets"])
    normal_500 = cast(
        "dict[str, object]",
        cast(
            "dict[str, dict[str, object]]",
            cast("dict[str, object]", branches["normal_vae"])["budgets"],
        )["500"]["metrics"],
    )
    so2_500 = cast(
        "dict[str, object]",
        cast(
            "dict[str, dict[str, object]]",
            cast("dict[str, object]", branches["so2_vae"])["budgets"],
        )["500"]["metrics"],
    )
    estimates = cast("dict[str, object]", paired_budgets["500"]["estimates"])
    assert float(estimates["macro_f1"]) == pytest.approx(
        float(normal_500["macro_f1"]) - float(so2_500["macro_f1"]),
    )


def test_stratified_bootstrap_preserves_2_16_5_counts_and_is_deterministic() -> None:
    strata = (tuple(range(2)), tuple(range(2, 18)), tuple(range(18, 23)))
    first = _bootstrap_multiplicities(
        wsi_count=23,
        stratum_indices=strata,
        replicates=100,
        seed=BOOTSTRAP_SEED,
    )
    second = _bootstrap_multiplicities(
        wsi_count=23,
        stratum_indices=strata,
        replicates=100,
        seed=BOOTSTRAP_SEED,
    )
    assert np.array_equal(first, second)
    assert np.all(first[:, :2].sum(axis=1) == 2)
    assert np.all(first[:, 2:18].sum(axis=1) == 16)
    assert np.all(first[:, 18:].sum(axis=1) == 5)
    assert np.all(first.sum(axis=1) == 23)


def test_aulc_is_normalized_trapezoid_on_log10_budget_axis() -> None:
    curve = np.asarray([[0.1, 0.2, 0.4, 0.7, 0.9]], dtype=np.float64)
    x = np.log10(np.asarray(BUDGETS, dtype=np.float64))
    expected = np.trapezoid(curve[0], x=x) / (x[-1] - x[0])
    assert _aulc(curve)[0] == pytest.approx(expected)
    assert _aulc(np.ones((1, 5), dtype=np.float64))[0] == pytest.approx(1.0)


def test_simultaneous_family_contains_five_budgets_and_aulc() -> None:
    paired = cast("dict[str, object]", _score()["paired_normal_minus_so2"])
    simultaneous = cast(
        "dict[str, object]",
        paired["six_contrast_simultaneous_95"],
    )
    contrasts = cast("dict[str, dict[str, float]]", simultaneous["contrasts"])
    assert simultaneous["quantile_method"] == "linear"
    assert set(contrasts) == {
        "macro_f1_budget_250",
        "macro_f1_budget_500",
        "macro_f1_budget_1000",
        "macro_f1_budget_2500",
        "macro_f1_budget_5671",
        "aulc",
    }
    critical = float(cast("float", simultaneous["critical_value"]))
    for interval in contrasts.values():
        assert interval["lower"] == pytest.approx(interval["estimate"] - critical)
        assert interval["upper"] == pytest.approx(interval["estimate"] + critical)


def test_rejects_labels_nonfinite_logits_argmax_drift_and_order_drift() -> None:
    predictions, _ = _fixture()
    rows = predictions["normal_vae"][250]

    labelled = [dict(row) for row in rows]
    labelled[0]["truth"] = 0
    with pytest.raises(ValueError, match="label-free allow-list"):
        _validate_prediction_rows(labelled, expected_count=len(labelled))

    nonfinite = [dict(row) for row in rows]
    nonfinite[0]["logit_tumor"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        _validate_prediction_rows(nonfinite, expected_count=len(nonfinite))

    wrong_argmax = [dict(row) for row in rows]
    wrong_argmax[0]["prediction"] = (int(wrong_argmax[0]["prediction"]) + 1) % 3
    with pytest.raises(ValueError, match="argmax"):
        _validate_prediction_rows(wrong_argmax, expected_count=len(wrong_argmax))

    reordered = [dict(row) for row in rows]
    reordered[0], reordered[1] = reordered[1], reordered[0]
    with pytest.raises(ValueError, match="canonical row order"):
        _validate_prediction_rows(reordered, expected_count=len(reordered))


def test_rejects_cross_budget_logical_identity_drift() -> None:
    predictions, oracle = _fixture()
    predictions["so2_vae"][5671][0]["x"] = 999_999
    with pytest.raises(ValueError, match="logical identity/order"):
        _score_tissue_test_predictions(
            predictions=predictions,
            oracle_rows=oracle,
            replicates=10,
            seed=BOOTSTRAP_SEED,
            expected_patch_count=len(oracle),
            expected_patch_support=None,
            expected_wsi_support={"tumor": 23, "stroma": 21, "necrosis": 5},
            expected_strata=EXPECTED_STRATA,
        )


def test_pre_score_claim_write_is_exclusive(tmp_path: Path) -> None:
    claim = tmp_path / "pre_score_claim.json"
    write_exclusive_json(claim, {"frozen": True})
    with pytest.raises(FileExistsError):
        write_exclusive_json(claim, {"frozen": False})
    assert json.loads(claim.read_text(encoding="utf-8")) == {"frozen": True}


def test_receipt_bound_retrieval_claims_before_join_and_completes_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictions, oracle = _fixture()
    patch_count = len(oracle)
    monkeypatch.setattr(scorer, "TEST_PATCH_COUNT", patch_count)
    monkeypatch.setattr(
        scorer,
        "EXPECTED_PATCH_SUPPORT",
        {"tumor": 23, "stroma": 21, "necrosis": 5},
    )
    remote_root = tmp_path / "remote"
    payload = remote_root / "tissue_test_evaluation"
    payload.mkdir(parents=True)
    row_identity = _contract_row_identity_sha256(predictions["normal_vae"][250])
    branches = {
        branch: {
            str(budget): {
                "branch": branch,
                "budget_per_class": budget,
                "file_sha256": f"{branch}-{budget}",
            }
            for budget in BUDGETS
        }
        for branch in ("normal_vae", "so2_vae")
    }
    test_contract = {
        "row_count": patch_count,
        "wsi_count": 23,
        "class_order_local_only": False,
        "locations": {"bytes": 1, "sha256": "locations"},
        "row_identity_sha256": row_identity,
        "label_fields_present": False,
    }
    dataset_reference = "test-actor/eqvae-tissue-test-inputs-v1"
    kernel_id = "test-actor/eqvae-label-blind-tissue-test-evaluation"
    source_names = [f"source-owner/source-{index}" for index in range(6)]
    expected_contract: dict[str, object] = {
        "schema_version": "spec0043.label_blind_input.v1",
        "dataset_reference": dataset_reference,
        "dataset_actor": "test-actor",
        "kernel_id": kernel_id,
        "spec_sha256": "s" * 64,
        "scorer_sha256": _sha256(Path(scorer.__file__)),
        "test_vector_sha256": _sha256(
            Path("docs/data/spec0043_tissue_test_scorer_vector.json"),
        ),
        "kernel_sources": source_names,
        "budgets_per_class": list(BUDGETS),
        "branches": branches,
        "test": test_contract,
    }
    for branch in ("normal_vae", "so2_vae"):
        worker = payload / "branches" / branch
        records: dict[str, object] = {}
        for budget in BUDGETS:
            relative = f"budget_{budget:04d}_per_class/predictions.csv.gz"
            path = worker / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=scorer.PREDICTION_HEADER)
                writer.writeheader()
                writer.writerows(predictions[branch][budget])
            _, evidence = _prediction_file_evidence(path)
            records[str(budget)] = {
                "path": relative,
                "file": _file_record(path),
                **evidence,
                "checkpoint": branches[branch][str(budget)],
                "compute_seconds": 1.0,
            }
        _write_test_json(
            worker / "worker_status.json",
            {
                "schema_version": "spec0043.worker_status.v1",
                "status": "complete",
                "branch": branch,
                "optimizer_updates": 0,
                "batch_size": 159,
                "final_batch_size": patch_count % 159,
                "row_identity_sha256": row_identity,
                "access_by_part": {"1": patch_count},
                "load_seconds": 1.0,
                "transfer_seconds": 1.0,
                "predictions": records,
            },
        )
    runtime = {
        "torch": "2.14.0+cu130",
        "cuda": "13.0",
        "cudnn": 9,
        "devices": ["Tesla T4", "Tesla T4"],
        "capabilities": [[7, 5], [7, 5]],
        "driver_versions": ["580", "580"],
    }
    _write_test_json(payload / "runtime.json", runtime)
    input_sha256 = "i" * 64
    _write_test_json(
        payload / "run_contract.json",
        {
            "schema_version": "spec0043.remote_run_contract.v1",
            "input_contract_sha256": input_sha256,
            "input_dataset_reference": dataset_reference,
            "spec_sha256": expected_contract["spec_sha256"],
            "scorer_sha256": expected_contract["scorer_sha256"],
            "test_vector_sha256": expected_contract["test_vector_sha256"],
            "branches": branches,
            "test": test_contract,
            "runtime": runtime,
            "optimizer_updates": 0,
        },
    )
    artifacts = {
        path.relative_to(payload).as_posix(): _file_record(path)
        for path in sorted(payload.rglob("*"))
        if path.is_file()
    }
    _write_test_json(
        payload / "overall_status.json",
        {
            "schema_version": "spec0043.remote_status.v1",
            "status": "complete",
            "branches": ["normal_vae", "so2_vae"],
            "budgets_per_class": list(BUDGETS),
            "optimizer_updates": 0,
            "artifacts": artifacts,
        },
    )
    metadata_sha256 = "m" * 64
    kernel_sha256 = "k" * 64
    metadata_bytes = 123
    kernel_bytes = 456
    input_receipt_sha256 = "r" * 64
    input_receipt_bytes = 789
    kernel_files = {
        "kernel-metadata.json": {
            "bytes": metadata_bytes,
            "sha256": metadata_sha256,
        },
        "run.py": {"bytes": kernel_bytes, "sha256": kernel_sha256},
    }
    claim_path = tmp_path / "launch_claim.json"
    _write_test_json(
        claim_path,
        {
            "schema_version": "spec0043.exclusive_launch_claim.v1",
            "status": "claimed_before_remote_push",
            "authorization": "ok let's do the patch tissue test evaluation",
            "authorization_date": "2026-09-06",
            "dataset_reference": dataset_reference,
            "kernel_id": kernel_id,
            "input_contract_sha256": input_sha256,
            "input_dataset_receipt": {
                "bytes": input_receipt_bytes,
                "sha256": input_receipt_sha256,
            },
            "kernel_files": kernel_files,
            "scientific_retries_authorized": 0,
        },
    )
    launch_path = tmp_path / "launch.json"
    version = 1
    reference = f"{kernel_id}/{version}"
    _write_test_json(
        launch_path,
        {
            "schema_version": "eqvae.kaggle_kernel_launch.v1",
            "actor": "test-actor",
            "original_kernel_id": kernel_id,
            "requested_kernel_id": kernel_id,
            "kernel_id": kernel_id,
            "accepted_version": version,
            "kernel_reference": reference,
            "source_locators": {
                "competition_sources": [],
                "dataset_sources": [dataset_reference],
                "kernel_sources": source_names,
                "model_sources": [],
            },
            "source_metadata_sha256": metadata_sha256,
            "upload_metadata_sha256": metadata_sha256,
            "source_files": kernel_files,
            "upload_files": kernel_files,
        },
    )
    receipt_files = {
        path.relative_to(remote_root).as_posix(): _file_record(path)
        for path in sorted(remote_root.rglob("*"))
        if path.is_file()
    }
    _write_test_json(
        remote_root / "kaggle_output_receipt.json",
        {
            "schema_version": "eqvae.kaggle_download.v1",
            "resource_kind": "kernel",
            "resource_owner": "test-actor",
            "resource_slug": "eqvae-label-blind-tissue-test-evaluation",
            "resource_version": version,
            "resource_reference": reference,
            "files": receipt_files,
        },
    )
    oracle_path = tmp_path / "tissue_test.csv"
    with oracle_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=scorer.ORACLE_HEADER)
        writer.writeheader()
        writer.writerows(oracle)
    monkeypatch.setattr(
        scorer,
        "LABEL_ORACLE_SHA256",
        hashlib.sha256(oracle_path.read_bytes()).hexdigest(),
    )
    output = tmp_path / "scored"
    completion = scorer.score_retrieved_tissue_test_output(
        remote_output_root=remote_root,
        launch_receipt_path=launch_path,
        launch_claim_path=claim_path,
        label_oracle_path=oracle_path,
        output_root=output,
        expected_contract=expected_contract,
        expected_input_contract_sha256=input_sha256,
        expected_input_receipt_sha256=input_receipt_sha256,
        expected_input_receipt_bytes=input_receipt_bytes,
        expected_kernel_sha256=kernel_sha256,
        expected_metadata_sha256=metadata_sha256,
        expected_kernel_bytes=kernel_bytes,
        expected_metadata_bytes=metadata_bytes,
    )
    assert completion["status"] == "complete"
    assert output.is_dir()
    assert (output / "overall_status.json").is_file()
    assert (tmp_path / "scored.pre_score_claim.json").is_file()
    assert len(list((output / "branches").rglob("scored_predictions.csv.gz"))) == 10

    tampered_launch = cast("dict[str, object]", json.loads(launch_path.read_text()))
    tampered_files = cast(
        "dict[str, dict[str, object]]",
        tampered_launch["source_files"],
    )
    tampered_files["run.py"]["sha256"] = "x" * 64
    _write_test_json(launch_path, tampered_launch)
    with pytest.raises(ValueError, match="launch receipt differs"):
        scorer.score_retrieved_tissue_test_output(
            remote_output_root=remote_root,
            launch_receipt_path=launch_path,
            launch_claim_path=claim_path,
            label_oracle_path=oracle_path,
            output_root=tmp_path / "tampered-scored",
            expected_contract=expected_contract,
            expected_input_contract_sha256=input_sha256,
            expected_input_receipt_sha256=input_receipt_sha256,
            expected_input_receipt_bytes=input_receipt_bytes,
            expected_kernel_sha256=kernel_sha256,
            expected_metadata_sha256=metadata_sha256,
            expected_kernel_bytes=kernel_bytes,
            expected_metadata_bytes=metadata_bytes,
        )


def _write_test_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
