# Copyright 2026 HiperMaximus
# pyright: reportAny=false, reportArgumentType=false, reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownVariableType=false
# ruff: noqa: D103, PLR2004, PLC2701
"""Focused tests for the frozen Spec 0041 post-retrieval scorer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import pytest

from eqvae.evaluation.mil_test_scoring import (
    _validate_normalization_amendment,
    _write_exclusive_json,
    score_mil_test_predictions,
)


def _fixture() -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[int, tuple[str, int]],
]:
    vector = cast(
        "dict[str, object]",
        json.loads(
            Path("docs/data/spec0041_mil_test_scorer_vector.json").read_text(
                encoding="utf-8",
            ),
        ),
    )
    labels = {
        int(key): (str(value[0]), int(value[1]))
        for key, value in cast("dict[str, list[object]]", vector["labels"]).items()
    }

    def rows(predictions: list[object]) -> list[dict[str, object]]:
        result = []
        for wsi_id, prediction_value in zip(
            sorted(labels),
            predictions,
            strict=True,
        ):
            prediction = int(cast("int", prediction_value))
            logits = [-2.0] * 5
            logits[prediction] = 2.0
            result.append(
                {
                    "wsi_id": wsi_id,
                    "prediction": prediction,
                    "logits": logits,
                    "bag_size": wsi_id,
                    "graph_identity": f"graph-{wsi_id}",
                    "graph_degree": {"minimum": 1, "maximum": 1, "mean": 1.0},
                    "access": {"parts": {"1": wsi_id}},
                },
            )
        return result

    return (
        rows(cast("list[object]", vector["normal_predictions"])),
        rows(cast("list[object]", vector["so2_predictions"])),
        labels,
    )


def test_scores_both_branches_and_paired_primary_direction() -> None:
    normal, so2, labels = _fixture()
    result = score_mil_test_predictions(
        normal_rows=normal,
        so2_rows=so2,
        labels=labels,
        replicates=100,
        seed=4101,
    )
    branches = cast("dict[str, dict[str, object]]", result["branches"])
    paired = cast("dict[str, object]", result["paired_normal_minus_so2"])
    assert set(branches) == {"normal_vae", "so2_vae"}
    assert result["wsi_count"] == 23
    assert "macro_f1" in paired


def test_rejects_remote_truth_and_pair_misalignment() -> None:
    normal, so2, labels = _fixture()
    normal[0]["truth"] = 0
    with pytest.raises(ValueError, match="label-dependent"):
        score_mil_test_predictions(normal_rows=normal, so2_rows=so2, labels=labels)

    normal, so2, labels = _fixture()
    so2[0]["graph_identity"] = "different"
    with pytest.raises(ValueError, match="geometry"):
        score_mil_test_predictions(normal_rows=normal, so2_rows=so2, labels=labels)


def test_pre_score_claim_is_exclusive(tmp_path: Path) -> None:
    claim = tmp_path / "claim.json"
    _write_exclusive_json(claim, {"frozen": True})
    with pytest.raises(FileExistsError):
        _write_exclusive_json(claim, {"frozen": False})
    assert json.loads(claim.read_text(encoding="utf-8")) == {"frozen": True}


def test_normalization_amendment_binds_exact_receipt_and_scorers() -> None:
    path = Path("docs/data/spec0042_spec0041_slug_normalization_amendment.json")
    amendment = json.loads(path.read_text(encoding="utf-8"))
    result = _validate_normalization_amendment(
        path=path,
        expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        current_scorer_sha256=amendment["amended_scorer_sha256"],
        remote_scorer_sha256=amendment["original_scorer_sha256"],
        launch_receipt_sha256=amendment["launch_receipt_sha256"],
    )
    assert result["scientific_changes"] == []
    with pytest.raises(ValueError, match="amendment contract"):
        _validate_normalization_amendment(
            path=path,
            expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            current_scorer_sha256="0" * 64,
            remote_scorer_sha256=amendment["original_scorer_sha256"],
            launch_receipt_sha256=amendment["launch_receipt_sha256"],
        )
