# Copyright 2026 HiperMaximus
"""Focused local contracts for the one-shot Spec 0015 readiness probe."""
# pyright: reportPrivateUsage=false
# ruff: noqa: PLC2701, PLR2004, RUF069

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from eqvae.benchmarking.so2_runtime_readiness import (
    GATE_ROW_COUNT,
    _named_update_proof,
    _optimizer_config,
    _snapshot_named,
    _verdict,
    optimizer_policy_proof,
    parse_probe_config,
    validate_selected_plan,
)
from eqvae.models.registry import MODEL_KIND_SO2_FIXED, build_model
from eqvae.training.selected_runtime import parse_selected_runtime_plan
from eqvae.training.selected_runtime_runner import (
    _AmpExecution,
    _build_selected_model,
    _gate_health_rows,
)

_CONFIG = Path("configs/spec0015/so2_vae_selected_runtime_readiness.json")
_RUNTIME = Path("configs/spec0001/non_eq_vae_selected_runtime.json")


def test_readiness_config_is_one_generated_batch_one_coordinate() -> None:
    """The probe config cannot grow a dataset or batch/runtime search surface."""
    config = parse_probe_config(_CONFIG)
    assert config.warmup_updates == 3
    assert config.settled_updates == 3
    assert config.runtime_path == _RUNTIME


def test_readiness_rejects_selected_runtime_drift() -> None:
    """Any runtime-policy substitution fails before CUDA execution."""
    plan = parse_selected_runtime_plan(_RUNTIME)
    validate_selected_plan(plan)
    with pytest.raises(ValueError, match="locked Spec 0015 bundle"):
        validate_selected_plan(replace(plan, compiled_autograd=False))
    with pytest.raises(ValueError, match="locked Spec 0015 bundle"):
        validate_selected_plan(replace(plan, ddp_bucket_cap_mb=25))
    with pytest.raises(ValueError, match="locked Spec 0015 bundle"):
        validate_selected_plan(replace(plan, artifact_sha256="0" * 64))


def test_so2_optimizer_policy_covers_coefficients_and_all_gate_families() -> None:
    """SO2 coefficients decay while every radial-gate tensor uses 0.5x/no-decay."""
    plan = parse_selected_runtime_plan(_RUNTIME)
    model = build_model(MODEL_KIND_SO2_FIXED)
    proof = optimizer_policy_proof(model, _optimizer_config(plan))
    assert proof["status"] == "pass"
    assert proof["coefficient_parameter_count"] == 1_172_304
    assert proof["gate_parameter_count"] == 4_096
    assert proof["gate_weight_decay"] == 0.0
    assert cast("float", proof["gate_learning_rate"]) == (
        cast("float", proof["base_learning_rate"]) * 0.5
    )


def test_selected_runner_builds_the_explicit_fixed_kind() -> None:
    """The shared runner selects the fixed model without architecture knobs."""
    settings = SimpleNamespace(model_kind=MODEL_KIND_SO2_FIXED, norm_groups=8)
    model = _build_selected_model(settings)  # pyright: ignore[reportArgumentType]
    assert type(model).__name__ == "SO2VAE"


def test_shared_runner_does_not_certify_structural_so2_gate_placeholders() -> None:
    """Only the activation-capturing readiness path may certify SO2 gate rows."""
    plan = parse_selected_runtime_plan(_RUNTIME)
    model = build_model(MODEL_KIND_SO2_FIXED)
    amp = _AmpExecution(
        enabled=True,
        grad_scaler_enabled=True,
        grad_scaler_init_scale=16_384.0,
        autocast_dtype="float16",
        requested_autocast_dtype="float16",
        local_amp_status="test",
    )
    probe = SimpleNamespace(
        accelerator_mode="dual_t4_ddp",
        machine_shape="NvidiaTeslaT4",
    )
    assert (
        _gate_health_rows(
            run_name="test",
            plan=plan,
            probe=probe,  # pyright: ignore[reportArgumentType]
            amp=amp,
            model=model,
            optimizer_step=2,
        )
        == ()
    )


def test_named_update_proof_requires_gradient_and_parameter_motion() -> None:
    """Named update evidence fails closed unless both signals are positive."""
    model = build_model(MODEL_KIND_SO2_FIXED)
    before = _snapshot_named(model)
    parameter = dict(model.named_parameters())["output_head.coeff00"]
    parameter.grad = torch.ones_like(parameter)
    with torch.no_grad():
        parameter.add_(1.0)
    proof = _named_update_proof(model, before, labels=("output_head",))
    assert cast("dict[str, object]", proof["output_head"])["status"] == "pass"
    parameter.grad = None
    with pytest.raises(RuntimeError, match="gradient-driven updates missing"):
        _named_update_proof(model, before, labels=("output_head",))
    parameter.grad = torch.ones_like(parameter)
    stationary = _snapshot_named(model)
    with pytest.raises(RuntimeError, match="gradient-driven updates missing"):
        _named_update_proof(model, stationary, labels=("output_head",))


def test_readiness_verdict_rejects_graph_or_gate_evidence_failure() -> None:
    """The producer verdict rejects either compiled or per-family evidence drift."""
    compiled = {
        "post_settle_graph_break_count": 0,
        "post_settle_recompile_count": 0,
        "amp_step_skipped_count": 0,
        "finite_losses": True,
        "finite_parameters": True,
    }
    result = {"compiled_execution": compiled}
    rows = tuple({"gate_health_status": "pass"} for _ in range(GATE_ROW_COUNT))
    assert _verdict(result, rows) == (True, [])  # pyright: ignore[reportArgumentType]
    compiled["post_settle_graph_break_count"] = 1
    passed, failures = _verdict(result, rows)  # pyright: ignore[reportArgumentType]
    assert not passed
    assert "graph_breaks" in failures
    compiled["post_settle_graph_break_count"] = 0
    failed_rows = list(rows)
    failed_rows[0] = {"gate_health_status": "fail"}
    passed, failures = _verdict(  # pyright: ignore[reportArgumentType]
        result,  # pyright: ignore[reportArgumentType]
        tuple(failed_rows),
    )
    assert not passed
    assert "finite_gate_rows" in failures
    passed, failures = _verdict(  # pyright: ignore[reportArgumentType]
        result,  # pyright: ignore[reportArgumentType]
        rows[:-1],
    )
    assert not passed
    assert "gate_rows" in failures
