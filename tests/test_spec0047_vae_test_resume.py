# Copyright 2026 HiperMaximus
# pyright: reportPrivateUsage=false
# ruff: noqa: D103, PLC2701
"""Focused tests for the administrative Spec 0047 scoring resume."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import scripts.build_vae_test_evaluation as builder

from eqvae.evaluation.vae_test import canonical_json_sha256, sha256_file
from eqvae.evaluation.vae_test_index_adapter import (
    FROZEN_ORACLE_DIAGNOSIS_TO_INDEX,
    _adapt_oracle_labels,
)
from eqvae.evaluation.vae_test_reporting import (
    _validate_normalization_amendment,
    write_exclusive_json,
)
from eqvae.evaluation.vae_test_resume import (
    FROZEN_SCORER_SHA256,
    OLD_DIAGNOSIS_TO_INDEX,
    _consume_resume_authority,
    _validate_frozen_scorer,
    _validate_resume_inputs,
    validate_diagnosis_index_amendment,
)
from eqvae.evaluation.vae_test_scoring import (
    BOOTSTRAP_SEED,
    EXPECTED_DIAGNOSIS_WSI_COUNTS,
    TEST_VECTOR_PATH,
    OracleLabel,
    _build_vector_fixture,
    _score_vae_test_metrics,
)

AMENDMENT = Path(
    "docs/data/spec0047_spec0045_diagnosis_index_resume_amendment.json",
)
PRE_SCORE_CLAIM = Path(
    "runs/local/vae_test_reconstruction_scored_v1.pre_score_claim.json",
)
REMOTE_OUTPUT = Path("runs/kaggle/vae_test_reconstruction_v1")
LAUNCH_RECEIPT = Path(
    "runs/local/kaggle_launches/maximshtefan/"
    "eqvae-frozen-vae-full-test-reconstruction/v0001.json",
)


def _read(path: Path) -> dict[str, object]:
    return cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))


def test_adapter_preserves_frozen_scorer_result_vector() -> None:
    vector_path = Path(TEST_VECTOR_PATH)
    vector = _read(vector_path)
    rows, frozen_labels, patch_counts = _build_vector_fixture(
        cast("dict[str, object]", vector["fixture"]),
    )
    oracle_labels = {
        atlas: replace(
            label,
            diagnosis_index=FROZEN_ORACLE_DIAGNOSIS_TO_INDEX[label.diagnosis_label],
        )
        for atlas, label in frozen_labels.items()
    }
    adapted = _adapt_oracle_labels(oracle_labels)
    assert all(
        label.diagnosis_index == OLD_DIAGNOSIS_TO_INDEX[label.diagnosis_label]
        for label in adapted.values()
    )
    result = _score_vae_test_metrics(
        remote_rows=rows,
        labels_by_atlas_row=adapted,
        bootstrap_replicates=10_000,
        bootstrap_seed=BOOTSTRAP_SEED,
        expected_patch_counts=patch_counts,
        expected_wsi_counts=EXPECTED_DIAGNOSIS_WSI_COUNTS,
    )
    assert canonical_json_sha256(result) == vector["expected_result_sha256"]
    assert sha256_file(Path("src/eqvae/evaluation/vae_test_scoring.py")) == (
        "55a0249661d5f972e2be42724d5afac797997f26c75f57d14988da52b01936c1"
    )


def test_adapter_rejects_nonfrozen_oracle_mapping() -> None:
    label = OracleLabel("CC", 3, 1, 0, 0, "test")
    with pytest.raises(ValueError, match="frozen oracle diagnosis mapping"):
        _adapt_oracle_labels({1: label})


def test_live_frozen_scorer_hash_is_enforced(tmp_path: Path) -> None:
    scorer = tmp_path / "scorer.py"
    scorer.write_bytes(Path("src/eqvae/evaluation/vae_test_scoring.py").read_bytes())
    assert _validate_frozen_scorer(scorer) == FROZEN_SCORER_SHA256
    scorer.write_bytes(scorer.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="frozen scorer bytes differ"):
        _validate_frozen_scorer(scorer)


def test_amendment_binds_adapter_resume_and_prior_claim() -> None:
    amendment = _read(AMENDMENT)
    result = validate_diagnosis_index_amendment(
        path=AMENDMENT,
        expected_sha256=builder.DIAGNOSIS_INDEX_AMENDMENT_SHA256,
        current_index_adapter_sha256=sha256_file(
            Path("src/eqvae/evaluation/vae_test_index_adapter.py"),
        ),
        resume_module_sha256=sha256_file(
            Path("src/eqvae/evaluation/vae_test_resume.py"),
        ),
        prior_pre_score_claim_sha256=sha256_file(PRE_SCORE_CLAIM),
    )
    assert result["scientific_changes"] == []
    assert sha256_file(AMENDMENT) == builder.DIAGNOSIS_INDEX_AMENDMENT_SHA256
    assert amendment["frozen_scorer_sha256"] == sha256_file(
        Path("src/eqvae/evaluation/vae_test_scoring.py"),
    )
    with pytest.raises(ValueError, match="amendment contract"):
        validate_diagnosis_index_amendment(
            path=AMENDMENT,
            expected_sha256=builder.DIAGNOSIS_INDEX_AMENDMENT_SHA256,
            current_index_adapter_sha256="0" * 64,
            resume_module_sha256=cast("str", amendment["resume_module_sha256"]),
            prior_pre_score_claim_sha256=cast(
                "str",
                amendment["prior_pre_score_claim_sha256"],
            ),
        )


def test_resume_authority_revalidates_exact_existing_evidence(tmp_path: Path) -> None:
    input_contract_path = (
        Path("runs/local/vae_test_evaluation_input") / "spec0045_vae_test_input.json"
    )
    contract = _read(input_contract_path)
    normalization = _validate_normalization_amendment(
        path=builder.NORMALIZATION_AMENDMENT_PATH,
        expected_sha256=builder.NORMALIZATION_AMENDMENT_SHA256,
        current_reporter_sha256=sha256_file(
            Path("src/eqvae/evaluation/vae_test_reporting.py"),
        ),
        remote_reporter_sha256=cast("str", contract["reporter_sha256"]),
        launch_receipt_sha256=sha256_file(LAUNCH_RECEIPT),
    )
    diagnosis = validate_diagnosis_index_amendment(
        path=AMENDMENT,
        expected_sha256=builder.DIAGNOSIS_INDEX_AMENDMENT_SHA256,
        current_index_adapter_sha256=sha256_file(
            Path("src/eqvae/evaluation/vae_test_index_adapter.py"),
        ),
        resume_module_sha256=sha256_file(
            Path("src/eqvae/evaluation/vae_test_resume.py"),
        ),
        prior_pre_score_claim_sha256=sha256_file(PRE_SCORE_CLAIM),
    )
    prior = _validate_resume_inputs(
        remote_output_root=REMOTE_OUTPUT,
        launch_receipt_path=LAUNCH_RECEIPT,
        input_receipt_path=builder.INPUT_RECEIPT_PATH,
        input_contract_path=input_contract_path,
        kernel_root=builder.KERNEL_ROOT,
        oracle_path=builder.ORACLE_PATH,
        prior_pre_score_claim_path=PRE_SCORE_CLAIM,
        normalization_amendment=normalization,
        diagnosis_amendment=diagnosis,
        vector_path=Path(TEST_VECTOR_PATH),
    )
    assert prior["remote_metric_sha256"] == diagnosis["remote_metric_sha256"]

    mutated = tmp_path / "pre_score_claim.json"
    payload = _read(PRE_SCORE_CLAIM)
    payload["remote_metric_sha256"] = "0" * 64
    mutated.write_text(
        json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="prior scoring authority"):
        _validate_resume_inputs(
            remote_output_root=REMOTE_OUTPUT,
            launch_receipt_path=LAUNCH_RECEIPT,
            input_receipt_path=builder.INPUT_RECEIPT_PATH,
            input_contract_path=input_contract_path,
            kernel_root=builder.KERNEL_ROOT,
            oracle_path=builder.ORACLE_PATH,
            prior_pre_score_claim_path=mutated,
            normalization_amendment=normalization,
            diagnosis_amendment=diagnosis,
            vector_path=Path(TEST_VECTOR_PATH),
        )


def test_shell_resume_route_is_exact_and_has_no_path_override() -> None:
    source = Path("scripts/kaggle_kernel.sh").read_text(encoding="utf-8")
    assert "resume-score-vae-test)" in source
    assert '[[ "$#" -eq 1 ]]' in source
    assert "resume-score-vae-test accepts no path overrides" in source
    assert "--remote-output-root runs/kaggle/vae_test_reconstruction_v1" in source
    assert "--output-root runs/local/vae_test_reconstruction_scored_v1" in source


def test_resume_claim_is_exclusive_and_precedes_any_metric_reopen(
    tmp_path: Path,
) -> None:
    claim = tmp_path / "resume_claim.json"
    payload = {"schema_version": "test"}

    def fail_after_claim() -> dict[str, object]:
        raise RuntimeError

    with pytest.raises(RuntimeError):
        _consume_resume_authority(
            resume_claim_path=claim,
            resume_claim=payload,
            post_claim_validation=fail_after_claim,
        )
    assert _read(claim) == payload
    with pytest.raises(FileExistsError):
        write_exclusive_json(claim, {"schema_version": "replacement"})

    source = Path("src/eqvae/evaluation/vae_test_resume.py").read_text(
        encoding="utf-8",
    )
    consume_call = source.index("prior_claim = _consume_resume_authority(")
    metric_reopen = source.index("remote_rows = load_remote_metric_rows(metric_path)")
    oracle_reopen = source.index("labels = load_oracle_labels(")
    assert consume_call < metric_reopen
    assert consume_call < oracle_reopen
