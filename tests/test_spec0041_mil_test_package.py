# Copyright 2026 HiperMaximus
# pyright: reportAny=false
# ruff: noqa: D103
"""Focused package and inference-only guards for Spec 0041."""

from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

ROOT = Path("runs/local/ubc_ocean_mil_test_evaluation")


def test_built_bundle_is_label_blind_and_model_state_only() -> None:
    contract = json.loads(
        (ROOT / "bundle/mil_test_inference_input.json").read_text(encoding="utf-8"),
    )
    files = set(contract["files"])
    non_source = {name for name in files if not name.startswith("src/")}
    assert non_source == {
        "test/physical_parts.csv",
        "test/wsi_cancer_test_bags.csv",
        "test/wsi_cancer_test_instances.csv",
        "weights/normal_vae.pt",
        "weights/so2_vae.pt",
    }
    for name in ("wsi_cancer_test_bags.csv", "wsi_cancer_test_instances.csv"):
        with (ROOT / "bundle/test" / name).open(newline="", encoding="utf-8") as handle:
            fields = set(csv.DictReader(handle).fieldnames or ())
        assert not fields & {
            "diagnosis",
            "diagnosis_label",
            "diagnosis_index",
            "truth",
            "target",
        }


def test_kernel_has_no_training_or_scoring_call_path() -> None:
    path = ROOT / "kernel/run.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
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
    source = path.read_text(encoding="utf-8")
    assert '_sha256(binary) != row["binary_sha256"]' in source
    assert "non_source != allowed" in source


def test_shell_route_consumes_an_exclusive_launch_claim() -> None:
    source = Path("scripts/kaggle_kernel.sh").read_text(encoding="utf-8")
    assert 'mil_test_launch_claim="$mil_test_root/launch_claim.json"' in source
    assert '"spec0041.exclusive_launch_claim.v1"' in source
    assert "os.O_EXCL" in source
    assert (
        'mil_test_accepted_reference="maximshtefan/'
        'eqvae-label-blind-mil-test-evaluation/1"' in source
    )
    assert 'receipt_sha256" == "$mil_test_launch_receipt_sha256' in source
