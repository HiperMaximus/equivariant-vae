# Copyright 2026 HiperMaximus
# ruff: noqa: COM812, DOC501, E501, EM101, PLC0415, PLR2004, TRY003
# pyright: reportAny=false, reportArgumentType=false, reportMissingParameterType=false, reportOperatorIssue=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownVariableType=false
"""Regression gates for the bounded Spec 0058 JVP epsilon calibration."""

from __future__ import annotations

import hashlib
import json
import re
import runpy
import subprocess  # noqa: S404
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from types import FunctionType

    import pytest


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _template() -> dict[str, object]:
    return runpy.run_path(
        _root() / "kaggle/kernels/jvp_epsilon_grid_calibration/run_template.py"
    )


def _contract() -> dict[str, object]:
    return json.loads(
        (
            _root() / "docs/data/spec0058_jvp_epsilon_grid_calibration_contract.json"
        ).read_text(encoding="utf-8")
    )


def test_calibration_samples_every_locked_epsilon_and_selects_smallest(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A linear decoder makes every central-difference epsilon exact on all directions."""
    import torch

    namespace = _template()

    class LinearDecode:
        @staticmethod
        def encode(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return values, torch.zeros_like(values)

        @staticmethod
        def decode(values: torch.Tensor) -> torch.Tensor:
            return values * 1.5

    events: list[tuple[str, dict[str, object]]] = []
    calibrate_branch = cast("FunctionType", namespace["_calibrate_branch"])
    calibrate_branch.__globals__["ACTIVE_STAGE"] = "branch"
    calibrate_branch.__globals__["_emit_progress"] = lambda event, **fields: (
        events.append((event, fields))
    )
    summary, rows = calibrate_branch(
        LinearDecode(),
        torch.ones((1, 2, 2, 2), dtype=torch.float32),
        label="branch_a",
        torch=torch,
    )
    assert set(summary) == {"0.002", "0.004", "0.008", "0.016"}
    assert {epsilon: len(values) for epsilon, values in rows.items()} == {
        0.002: 8,
        0.004: 8,
        0.008: 8,
        0.016: 8,
    }
    assert len(events) == 32
    assert {
        fields["epsilon"]
        for event, fields in events
        if event == "jvp_direction_measured"
    } == {0.002, 0.004, 0.008, 0.016}
    selection = namespace["_select_epsilon"](summary)  # type: ignore[index,operator]
    assert selection["status"] == "selected"
    assert format(float(selection["public"]["epsilon"]), ".3f") == "0.002"  # type: ignore[index]
    assert not capsys.readouterr().out


def test_progress_schema_rejects_unlocked_epsilon_and_identity(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The verbose JSONL telemetry remains bounded and blind despite per-direction rows."""
    namespace = _template()
    namespace["_set_stage"]("branch")  # type: ignore[index,operator]
    for fields in (
        {
            "branch": "branch_a",
            "direction_index": 0,
            "epsilon": 0.001,
            "residual": 0.0,
        },
        {
            "branch": "normal_vae",
            "direction_index": 0,
            "epsilon": 0.004,
            "residual": 0.0,
        },
    ):
        try:
            namespace["_emit_progress"](  # type: ignore[index,operator]
                "jvp_direction_measured", **fields
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("progress helper accepted an invalid calibration row")
    assert capsys.readouterr().out


def test_output_is_redacted_and_records_no_eligible_selection(tmp_path: Path) -> None:
    """A no-selection outcome is an artifact, not a failure or model comparison."""
    namespace = _template()
    output = tmp_path / "jvp_epsilon_grid_calibration_v1"
    namespace["_write_output"].__globals__["OUTPUT_ROOT"] = output  # type: ignore[index,union-attr]
    summary = {"aggregate": 0.01, "worst": 0.02, "p50": 0.01, "p95": 0.02}
    namespace["_write_output"](  # type: ignore[index,operator]
        contract=_contract(),
        branches={"branch_a": {"0.002": summary}, "branch_b": {"0.002": summary}},
        global_summary={"0.002": summary},
        selection={"status": "no_eligible_epsilon", "public": {}},
    )
    payload = json.loads((output / "calibration.json").read_text(encoding="utf-8"))
    assert payload["selection"] == {"status": "no_eligible_epsilon"}
    assert {path.name for path in output.iterdir()} == {"calibration.json"}
    public_text = (output / "calibration.json").read_text(encoding="utf-8").lower()
    for forbidden in ("normal_vae", "so2_vae", "payload_manifest", "comparison"):
        assert forbidden not in public_text
    assert re.search(r"\b[a-f0-9]{64}\b", public_text) is None


def test_contract_builder_shell_guard_and_generated_kernel_are_exact() -> None:
    """The uploadable calibration is pinned to Spec 0058 and cannot become a full gate."""
    root = _root()
    contract_hash = hashlib.sha256(
        (
            root / "docs/data/spec0058_jvp_epsilon_grid_calibration_contract.json"
        ).read_bytes()
    ).hexdigest()
    spec_hash = hashlib.sha256(
        (root / "docs/specs/0058-jvp-epsilon-grid-calibration.md").read_bytes()
    ).hexdigest()
    template = (
        root / "kaggle/kernels/jvp_epsilon_grid_calibration/run_template.py"
    ).read_text(encoding="utf-8")
    builder = (root / "scripts/build_kaggle_embedded_kernel.py").read_text(
        encoding="utf-8"
    )
    shell = (root / "scripts/kaggle_kernel.sh").read_text(encoding="utf-8")
    for source in (template, shell):
        assert contract_hash in source
        assert spec_hash in source
    assert "docs/data/spec0058_jvp_epsilon_grid_calibration_contract.json" in builder
    for marker in (
        "KAGGLE_JVP_EPSILON_GRID_CALIBRATION_READY = True",
        "EPSILON_GRID = (0.002, 0.004, 0.008, 0.016)",
        "SELECTION_MAX = 0.005",
        "torch.autograd.functional.jvp",
        "frozen_bundle_ready",
    ):
        assert marker in template
    for forbidden in ("torch.compile(", "functional.vjp", "torch.optim", ".backward("):
        assert forbidden not in template
    assert "spec0058_jvp_epsilon_grid_calibration_authorized" in shell
    assert "jvp_epsilon_calibration_push=1" in shell
    remote_preflight = (
        'preflight_functional_geometry_slug "$jvp_epsilon_calibration_kernel_id"'
    )
    preflight_start = shell.index("preflight_functional_geometry_slug()")
    preflight_body = shell[preflight_start : shell.index("\n}\n", preflight_start)]
    assert "require_remote_confirmed" in preflight_body
    assert remote_preflight in shell
    assert shell.index(
        "claim_jvp_epsilon_calibration_push", shell.index(remote_preflight)
    ) > shell.index(remote_preflight)
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(root / "scripts/build_kaggle_embedded_kernel.py"),
            "--kernel-dir",
            str(root / "kaggle/kernels/jvp_epsilon_grid_calibration"),
            "--ready-marker",
            "KAGGLE_JVP_EPSILON_GRID_CALIBRATION_READY = True",
            "--allow-dirty",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    run_source = (
        root / "kaggle/kernels/jvp_epsilon_grid_calibration/run.py"
    ).read_text(encoding="utf-8")
    assert contract_hash in run_source
    assert spec_hash in run_source
    assert "KAGGLE_JVP_EPSILON_GRID_CALIBRATION_READY = True" in run_source
