# Copyright 2026 HiperMaximus
# ruff: noqa: COM812, DOC501, E501, EM101, PLC0415, PLR2004, TRY003
# pyright: reportAny=false, reportArgumentType=false, reportMissingParameterType=false, reportOperatorIssue=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownVariableType=false
"""Regression gates for the bounded, result-blind Spec 0057 preflight."""

from __future__ import annotations

import hashlib
import json
import math
import runpy
import subprocess  # noqa: S404
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    import pytest


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _template() -> dict[str, object]:
    return runpy.run_path(
        _root() / "kaggle/kernels/functional_geometry_preflight/run_template.py"
    )


def _contract() -> dict[str, object]:
    return json.loads(
        (_root() / "docs/data/spec0057_jvp_epsilon_ladder_contract.json").read_text(
            encoding="utf-8",
        ),
    )


def _event_epsilon(event: dict[str, object]) -> dict[str, object]:
    epsilon = event["epsilon"]
    assert isinstance(epsilon, dict)
    return cast("dict[str, object]", epsilon)


def test_analytic_fixtures_cover_full_d4_f1_and_numerical_contract() -> None:
    """The result-blind fixture gate exercises every declared analytic family."""
    import numpy as np
    import torch

    result = _template()["_run_analytic_fixtures"](  # type: ignore[index,operator]
        _contract(),
        np=np,
        torch=torch,
    )
    assert result["passed"]
    assert result["d4"]["table_entries"] == 64
    assert result["d4"]["f1_vector_exact"]
    assert result["d4"]["f1_pseudovector_exact"]
    assert result["metric"]["checks"] >= 14


def test_public_parent_output_is_redacted_and_uses_jsonl_ledger(tmp_path: Path) -> None:
    """No public parent file may recover a branch identity or payload manifest."""
    namespace = _template()
    output = tmp_path / "preflight_jvp_ladder_parent_v1"
    namespace["_write_parent_output"].__globals__["OUTPUT_ROOT"] = output  # type: ignore[index,union-attr]
    namespace["_write_parent_output"](  # type: ignore[index,operator]
        contract=_contract(),
        fixture={"passed": True},
        branches={"branch_a": {"seconds": 1.0}, "branch_b": {"seconds": 2.0}},
        runtime={"precision": "fp32_forward_autodiff_cpu_fp64_detached_accumulation"},
    )
    assert (output / "work_units.jsonl").is_file()
    assert not (output / "run_contract.json").exists()
    public_text = "\n".join(
        path.read_text(encoding="utf-8") for path in output.glob("*.json")
    ).lower()
    for forbidden in (
        "normal_vae",
        "so2_vae",
        "payload_manifest",
        "branch_permutation",
    ):
        assert forbidden not in public_text
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert "manifest.json" not in manifest["files"]


def test_blind_guard_rejects_identity_in_every_public_document() -> None:
    """A nested identity token is not permitted merely because rows are anonymous."""
    namespace = _template()
    namespace["_assert_blind_document"]({"branch_a": {"seconds": 1.0}})  # type: ignore[index,operator]
    try:
        namespace["_assert_blind_document"]({"nested": {"normal_vae": 1}})  # type: ignore[index,operator]
    except RuntimeError:
        pass
    else:
        raise AssertionError("blindness guard accepted a model identity")


def test_progress_jsonl_has_closed_schema_sequence_and_rejects_extra_fields(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The only print path serializes allow-listed, result-blind JSONL events."""
    namespace = _template()
    namespace["_set_stage"]("payload")  # type: ignore[index,operator]
    namespace["_emit_progress"]("payload_ready")  # type: ignore[index,operator]
    rows = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.strip()
    ]
    assert [row["schema"] for row in rows] == ["spec0053.preflight_progress.v1"] * 2
    assert [row["event"] for row in rows] == ["stage_started", "payload_ready"]
    assert [row["sequence"] for row in rows] == [1, 2]
    for prohibited in (
        {"path": "/kaggle/input/private"},
        {"hash": "abc"},
        {"model_identity": "hidden"},
    ):
        try:
            namespace["_emit_progress"](  # type: ignore[index,operator]
                "payload_ready",
                **prohibited,
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("progress helper accepted a non-contract field")


def test_frozen_bundle_event_is_real_jsonl_and_nested_values_are_closed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Bundle lookup is observable without accepting paths or hashes as values."""
    namespace = _template()
    namespace["_set_stage"]("frozen_state")  # type: ignore[index,operator]
    namespace["_emit_progress"]("frozen_bundle_ready")  # type: ignore[index,operator]
    rows = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.strip()
    ]
    bundle_event = rows[-1]
    assert bundle_event["event"] == "frozen_bundle_ready"
    assert bundle_event["stage"] == "frozen_state"
    assert set(bundle_event) == {
        "elapsed_seconds",
        "event",
        "schema",
        "sequence",
        "stage",
    }

    namespace["_set_stage"]("branch")  # type: ignore[index,operator]
    residuals = {
        "jvp_fd": 0.0,
        "vjp_adjoint": 0.0,
        "hvp_fd": 0.0,
        "repeat": 0.0,
    }
    limits = dict(residuals)

    def reject(event: str, **fields: object) -> None:
        try:
            namespace["_emit_progress"](event, **fields)  # type: ignore[index,operator]
        except RuntimeError:
            return
        raise AssertionError("progress helper accepted a contaminated nested value")

    reject(
        "eager_direction_measured",
        branch="branch_a",
        direction_index=0,
        epsilon={"jvp": "/kaggle/input/private", "hvp": 0.002},
        residuals=residuals,
        limits=limits,
        failed_checks=[],
    )
    reject(
        "eager_direction_measured",
        branch="branch_a",
        direction_index=0,
        diagnostic_only=True,
        epsilon={"jvp": 0.001, "hvp": 0.002},
        residuals=residuals,
        limits={},
        failed_checks=[],
    )
    reject(
        "eager_direction_measured",
        branch="branch_a",
        direction_index=0,
        epsilon={"jvp": 0.004, "hvp": 0.002},
        residuals=residuals,
        limits=limits,
        failed_checks=[],
    )
    reject(
        "workload_complete",
        branch="branch_a",
        workload="c4",
        profile={
            "workload": "c4",
            "seconds": "sha256:secret",
            "peak_allocated_bytes": 0,
            "peak_reserved_bytes": 0,
            "free_fraction_before": 1.0,
            "free_fraction_after": 1.0,
        },
    )


def test_template_requires_fp64_reductions_real_microbatches_and_fullgraph_compile() -> (
    None
):
    """Static invariants prevent a cheaper, non-equivalent numerical preflight."""
    source = (
        _root() / "kaggle/kernels/functional_geometry_preflight/run_template.py"
    ).read_text(encoding="utf-8")
    for marker in (
        "torch.complex128 if left.is_complex() or right.is_complex()",
        'to(device="cpu", dtype=dtype)',
        'to(device="cpu", dtype=torch.float64)',
        "torch.func.jvp(model.decode, (primals,), (directions,))",
        "direction_count=count",
        "5053 + direction_count",
        "compiled_jvp = torch.compile(",
        "compiled_hvp = torch.compile(",
        "fullgraph=True,",
        '"differentiable_path"',
        '"padded_continuous"',
        '"eager_direction_measured"',
        '"spec0053.preflight_progress.v1"',
        "allow_nan=False",
        "flush=True",
        'limits["tangent_count"]',
        "_rademacher_direction",
        '"jvp_diagnostic_epsilons"',
        '"diagnostic_hvp_epsilon"',
        '"jvp_epsilon_0_004"',
        '"jvp_epsilon_0_008"',
        '"compilation_started"',
        '"compilation_complete"',
        "_PROGRESS_EVENT_FIELDSETS",
        "_validate_progress_event",
    ):
        assert marker in source
    assert "torch.optim" not in source
    assert ".backward(" not in source
    assert "traceback.print_exc" not in source
    assert "str(error)" not in source


def test_eight_rademacher_projections_are_distinct_rms_one_and_reported() -> None:
    """All fixed projections and their diagnostic companions run on CPU."""
    import torch

    namespace = _template()
    contract = _contract()
    numerics = dict(contract["numerics"])  # type: ignore[index,arg-type]
    projection_shape = torch.empty((1, 16, 32, 32), dtype=torch.float32)
    directions = [
        namespace["_rademacher_direction"](  # type: ignore[index,operator]
            projection_shape,
            seed=int(numerics["tangent_seed"]) + index,
            torch=torch,
        )
        for index in range(int(numerics["tangent_count"]))
    ]
    assert len({bytes(direction.cpu().numpy()) for direction in directions}) == 8
    assert all(
        torch.equal(direction.square().mean(), torch.tensor(1.0))
        for direction in directions
    )
    z = torch.ones((1, 2, 2, 2), dtype=torch.float32, requires_grad=True)

    class LinearDecode:
        @staticmethod
        def decode(values: torch.Tensor) -> torch.Tensor:
            return values * 1.5

    events: list[tuple[str, dict[str, object]]] = []
    namespace["_eager_autodiff_set"].__globals__["_emit_progress"] = (  # type: ignore[index,union-attr]
        lambda event, **fields: events.append((event, fields))
    )
    result = namespace["_eager_autodiff_set"](  # type: ignore[index,operator]
        LinearDecode(),
        z,
        numerics,
        label="branch_a",
        torch=torch,
    )
    assert result["failed_checks"] == []
    eager_events = [
        fields for event, fields in events if event == "eager_direction_measured"
    ]
    assert len(eager_events) == 24
    assert sum(bool(event.get("diagnostic_only")) for event in eager_events) == 16
    assert {
        format(float(_event_epsilon(event)["jvp"]), ".3f")
        for event in eager_events
        if event.get("diagnostic_only")
    } == {"0.004", "0.008"}
    assert (
        sum(
            bool(
                event.get("diagnostic_only")
                and math.isclose(
                    float(_event_epsilon(event)["jvp"]),
                    0.004,
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
            )
            for event in eager_events
        )
        == 8
    )
    assert (
        sum(
            bool(
                event.get("diagnostic_only")
                and math.isclose(
                    float(_event_epsilon(event)["jvp"]),
                    0.008,
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
            )
            for event in eager_events
        )
        == 8
    )
    assert {
        tuple(
            sorted(
                (name, format(float(value), ".3f"))
                for name, value in _event_epsilon(event).items()
            )
        )
        for event in eager_events
    } == {
        (("hvp", "0.002"), ("jvp", "0.001")),
        (("hvp", "0.002"), ("jvp", "0.004")),
        (("hvp", "0.002"), ("jvp", "0.008")),
    }
    assert set(result["diagnostic_summary"]) == {  # type: ignore[arg-type,index]
        "jvp_epsilon_0_004",
        "jvp_epsilon_0_008",
    }


def test_cpu_fp64_residuals_and_worst_projection_gate_are_scale_aware() -> None:
    """Detached reductions use stable denominators and never use a mean-only gate."""
    import torch

    namespace = _template()
    residual, numerator, denominator = namespace["_symmetric_l2"](  # type: ignore[index,operator]
        torch.tensor([4.0], dtype=torch.float32),
        torch.tensor([2.0], dtype=torch.float32),
        torch=torch,
    )
    assert (residual, numerator, denominator) == (0.5, 2.0, 4.0)
    adjoint, numerator, denominator = namespace["_adjoint_error"](  # type: ignore[index,operator]
        torch.tensor([2.0], dtype=torch.float32),
        torch.tensor([3.0], dtype=torch.float32),
        torch.tensor([4.0], dtype=torch.float32),
        torch.tensor([1.0], dtype=torch.float32),
        torch=torch,
    )
    assert (adjoint, numerator, denominator) == (1 / 3, 2.0, 6.0)

    rows = [(0.001, 0.001, 1.0)] * 7 + [(0.02, 0.02, 1.0)]
    summary = namespace["_summary_from_details"](  # type: ignore[index,operator]
        {"jvp_fd": rows}
    )["jvp_fd"]
    assert abs(summary["p50"] - 0.001) < 1e-15
    assert abs(summary["worst"] - 0.02) < 1e-15
    assert summary["aggregate"] < 0.01
    assert summary["worst"] > 0.01


def test_receipt_bound_child_builds_and_authenticates_exact_parent(
    tmp_path: Path,
) -> None:
    """The child accepts only a canonical v1 receipt and exact parent manifest."""
    namespace = _template()
    parent = tmp_path / "download" / "preflight_jvp_ladder_parent_v1"
    namespace["_write_parent_output"].__globals__["OUTPUT_ROOT"] = parent  # type: ignore[index,union-attr]
    namespace["_write_parent_output"](  # type: ignore[index,operator]
        contract=_contract(),
        fixture={"passed": True},
        branches={"branch_a": {}, "branch_b": {}},
        runtime={},
    )
    receipt = tmp_path / "parent-receipt.json"
    parent_id = "maximshtefan/eqvae-functional-geometry-preflight-04a08ab5"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": "eqvae.kaggle_kernel_launch.v1",
                "actor": "maximshtefan",
                "original_kernel_id": parent_id,
                "requested_kernel_id": parent_id,
                "kernel_id": parent_id,
                "accepted_version": 1,
                "kernel_reference": f"{parent_id}/1",
            },
        ),
        encoding="utf-8",
    )
    output_receipt = parent.parent / "kaggle_output_receipt.json"
    output_receipt.write_text(
        json.dumps(
            {
                "schema_version": "eqvae.kaggle_download.v1",
                "resource_kind": "kernel",
                "resource_owner": "maximshtefan",
                "resource_slug": "eqvae-functional-geometry-preflight-04a08ab5",
                "resource_version": 1,
                "resource_reference": f"{parent_id}/1",
                "files": {
                    f"preflight_jvp_ladder_parent_v1/{path.name}": {
                        "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                    for path in parent.iterdir()
                },
            },
        ),
        encoding="utf-8",
    )
    child = tmp_path / "child"
    builder = _root() / "scripts/build_functional_geometry_preflight_continuation.py"
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(builder),
            "--launch-receipt",
            str(receipt),
            "--parent-output",
            str(parent.parent),
            "--parent-output-receipt",
            str(output_receipt),
            "--destination",
            str(child),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    child_source = (child / "run.py").read_text(encoding="utf-8")
    assert '"parent_version": 1' in child_source
    assert '"parent_manifest_sha256"' in child_source
    assert '"parent_output_receipt_sha256"' in child_source
    assert "KAGGLE_FUNCTIONAL_GEOMETRY_JVP_LADDER_RESUME_READY = True" in child_source
    assert "preflight_jvp_ladder_resume_v1" in child_source
    assert "torch" not in child_source.lower()
    (parent / "runtime.json").write_text('{"tampered": true}\n', encoding="utf-8")
    tampered = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(builder),
            "--launch-receipt",
            str(receipt),
            "--parent-output",
            str(parent.parent),
            "--parent-output-receipt",
            str(output_receipt),
            "--destination",
            str(tmp_path / "must-not-exist"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert tampered.returncode != 0
    assert "receipt hash differs" in tampered.stderr


def test_contract_hashes_and_shell_guard_fix_the_exact_v1_parent() -> None:
    """The one-shot parent cannot be silently canonicalized to another slug/version."""
    root = _root()
    spec_hash = hashlib.sha256(
        (root / "docs/specs/0057-jvp-epsilon-ladder-preflight.md").read_bytes(),
    ).hexdigest()
    contract_hash = hashlib.sha256(
        (root / "docs/data/spec0057_jvp_epsilon_ladder_contract.json").read_bytes(),
    ).hexdigest()
    template = (
        root / "kaggle/kernels/functional_geometry_preflight/run_template.py"
    ).read_text(encoding="utf-8")
    shell = (root / "scripts/kaggle_kernel.sh").read_text(encoding="utf-8")
    for source in (template, shell):
        assert spec_hash in source
        assert contract_hash in source
    assert "preflight_functional_geometry_slug" in shell
    assert "eqvae-functional-geometry-preflight-04a08ab5/1" in shell
    builder = (
        root / "scripts/build_functional_geometry_preflight_continuation.py"
    ).read_text(encoding="utf-8")
    assert "eqvae-functional-geometry-preflight-04a08ab5-resume" in builder
    assert "KAGGLE_FUNCTIONAL_GEOMETRY_JVP_LADDER_CONFIRMED" in shell


def test_frozen_bundle_hash_is_canonical_and_preload_event_is_closed() -> None:
    """The retry cannot regress to the rejected nibble or hide bundle lookup."""
    root = _root()
    canonical = hashlib.sha256(
        (
            root / "runs/local/vae_test_evaluation_input/spec0045_vae_test_input.json"
        ).read_bytes(),
    ).hexdigest()
    contract = _contract()
    inputs = contract["inputs"]
    assert isinstance(inputs, dict)
    source = (
        root / "kaggle/kernels/functional_geometry_preflight/run_template.py"
    ).read_text(encoding="utf-8")
    assert (
        canonical == "b5a32ebffd0d88a88d6f21b64ba5c9a23016f05d7a2db0546e442f12a0acecc1"
    )
    assert inputs["weight_bundle_contract_sha256"] == canonical
    numerics = contract["numerics"]
    assert isinstance(numerics, dict)
    assert math.isclose(float(numerics["jvp_epsilon"]), 0.001, rel_tol=0.0, abs_tol=0.0)
    assert [
        format(float(value), ".3f") for value in numerics["jvp_diagnostic_epsilons"]
    ] == [  # type: ignore[index]
        "0.004",
        "0.008",
    ]
    assert math.isclose(
        float(numerics["diagnostic_hvp_epsilon"]), 0.002, rel_tol=0.0, abs_tol=0.0
    )
    assert "b5d32" not in source
    assert '"frozen_bundle_ready": (frozenset(),),' in source
    assert '_emit_progress("frozen_bundle_ready")' in source
    assert source.index('_emit_progress("frozen_bundle_ready")') < source.index(
        "_load_models("
    )


def test_generated_kernel_retains_the_canonical_bundle_hash() -> None:
    """The uploadable single-file artifact carries the frozen-bundle binding."""
    root = _root()
    builder = root / "scripts/build_kaggle_embedded_kernel.py"
    kernel = root / "kaggle/kernels/functional_geometry_preflight"
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(builder),
            "--kernel-dir",
            str(kernel),
            "--ready-marker",
            "KAGGLE_FUNCTIONAL_GEOMETRY_JVP_LADDER_READY = True",
            "--allow-dirty",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    source = (kernel / "run.py").read_text(encoding="utf-8")
    assert "WEIGHT_BUNDLE_CONTRACT_SHA256" in source
    assert "b5a32ebffd0d88a88d6f21b64ba5c9a23016f05d7a2db0546e442f12a0acecc1" in source
    assert '"jvp_diagnostic_epsilons"' in source
    assert '"jvp_epsilon_0_008"' in source


def test_parent_metadata_is_private_t4_and_slug_aligned() -> None:
    """The parent has fixed hardware, no internet, and a non-ambiguous identity."""
    metadata = json.loads(
        (
            _root()
            / "kaggle/kernels/functional_geometry_preflight/kernel-metadata.json"
        ).read_text(encoding="utf-8"),
    )
    assert metadata["title"] == "eqvae-functional-geometry-preflight-04a08ab5"
    assert metadata["is_private"] == "true"
    assert metadata["enable_internet"] == "false"
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["kernel_sources"] == []
