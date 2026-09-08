# Copyright 2026 HiperMaximus
# pyright: reportAny=false, reportPrivateUsage=false
# ruff: noqa: D103, SLF001
"""Focused package and inference-only guards for Spec 0043."""

from __future__ import annotations

import ast
import csv
import json
from pathlib import Path
from typing import cast

import pytest
import scripts.build_tissue_test_evaluation as builder
import torch

ROOT = Path("runs/local/tissue_test_evaluation")
BUDGETS = (250, 500, 1000, 2500, 5671)
BRANCHES = ("normal_vae", "so2_vae")
TEST_ROW_COUNT = 31_572
STATE_TENSOR_COUNT = 14


def test_built_bundle_is_label_blind_and_model_state_only() -> None:
    contract = json.loads(
        (ROOT / "bundle/tissue_test_inference_input.json").read_text(
            encoding="utf-8",
        ),
    )
    files = set(contract["files"])
    non_source = {name for name in files if not name.startswith("src/")}
    assert non_source == {
        "test/physical_parts.csv",
        "test/tissue_test_locations.csv",
    } | {
        f"weights/{branch}/budget_{budget:04d}_per_class.pt"
        for branch in BRANCHES
        for budget in BUDGETS
    }
    with (ROOT / "bundle/test/tissue_test_locations.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        rows = list(reader)
    assert fields == (
        "dataset_row",
        "atlas_row_index",
        "wsi_id",
        "x",
        "y",
        "split",
        "part",
        "file_index",
    )
    assert len(rows) == TEST_ROW_COUNT
    assert {row["split"] for row in rows} == {"test"}
    assert not set(fields) & {
        "tissue_label",
        "selection_rank",
        "truth",
        "target",
        "diagnosis",
    }
    for branch in BRANCHES:
        for budget in BUDGETS:
            state = cast(
                "object",
                torch.load(
                    ROOT / f"bundle/weights/{branch}/budget_{budget:04d}_per_class.pt",
                    map_location="cpu",
                    weights_only=True,
                ),
            )
            assert isinstance(state, dict)
            state_dict = cast("dict[str, object]", state)
            assert len(state_dict) == STATE_TENSOR_COUNT
            assert "manifest" not in state_dict
            assert "selection" not in state_dict


def test_kernel_has_no_training_scoring_or_update_call_path() -> None:
    path = ROOT / "kernel/run.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    assert not any(name.startswith("eqvae.training") for name in imports)
    assert not any(name.startswith("eqvae.evaluation") for name in imports)
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not calls & {"backward", "step", "zero_grad"}
    assert "cross_entropy" not in source
    assert '_sha256(binary) != row["binary_sha256"]' in source
    assert "non_source != allowed" in source
    assert "store.read_rows" in source
    assert "torch.inference_mode" in source


def test_shell_route_requires_verified_input_and_one_use_claim() -> None:
    source = Path("scripts/kaggle_kernel.sh").read_text(encoding="utf-8")
    assert 'tissue_test_launch_claim="$tissue_test_root/launch_claim.json"' in source
    assert '[[ -f "$tissue_test_input_receipt" ]]' in source
    assert '[[ ! -e "$tissue_test_launch_claim" ]]' in source
    assert 'KAGGLE_TISSUE_TEST_ROUTE_ACTIVE=1 "$0" push' in source
    assert 'validate-claimed-launch --actor "$tissue_test_actor"' in source
    assert "EQVAE_KAGGLE_LAUNCH_RECEIPT_ROOT:-runs/local/kaggle_launches" in source
    assert '"spec0043.exclusive_launch_claim.v1"' in Path(
        "scripts/build_tissue_test_evaluation.py",
    ).read_text(encoding="utf-8")
    assert "os.O_EXCL" in Path("scripts/build_tissue_test_evaluation.py").read_text(
        encoding="utf-8",
    )


def test_verified_input_receipt_is_exact_and_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "package"
    bundle = output / "bundle"
    bundle.mkdir(parents=True)
    (bundle / builder.CONTRACT_NAME).write_text("{}\n", encoding="utf-8")
    (bundle / "payload.bin").write_bytes(b"sealed")
    contract = {"dataset_reference": "test-actor/eqvae-tissue-test-inputs-v1"}
    receipt = {
        "schema_version": "spec0043.input_dataset_receipt.v1",
        "dataset_reference": contract["dataset_reference"],
        "dataset_version": 1,
        "visibility": "private",
        "status": "verified",
        "input_contract_sha256": builder._sha256(bundle / builder.CONTRACT_NAME),
        "remote_files": builder._artifact_records(
            bundle,
            exclude={builder.METADATA_NAME},
        ),
    }
    (output / builder.INPUT_RECEIPT_NAME).write_text(
        json.dumps(receipt),
        encoding="utf-8",
    )
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    monkeypatch.setattr(builder, "OUTPUT_ROOT", Path("package"))
    assert builder._validate_input_receipt(contract)["sha256"]
    receipt["dataset_version"] = 2
    (output / builder.INPUT_RECEIPT_NAME).write_text(
        json.dumps(receipt),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="receipt differs"):
        builder._validate_input_receipt(contract)
