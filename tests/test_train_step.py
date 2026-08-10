# Copyright 2026 HiperMaximus
"""Tests for the spec 0001 local model/loss train-step contract."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import cast

import pytest
import torch

from eqvae.cli.benchmark_runtime import main as benchmark_runtime_main
from eqvae.losses.vae import beta_for_step
from eqvae.models.latent import LATENT_CHANNELS
from eqvae.models.non_equivariant_vae import (
    DEFAULT_LOGVAR_CLAMP_MAX,
    DEFAULT_LOGVAR_CLAMP_MIN,
    VaeForwardOutput,
    build_non_equivariant_vae,
)
from eqvae.training.optim import create_adamw_optimizer
from eqvae.training.step import TrainStepRequest, run_train_step

TEST_IMAGE_SIZE = 64
TEST_BATCH_SIZE = 1
TEST_MAX_OPTIMIZER_STEPS = 8


def test_forward_exposes_explicit_eps_and_clamped_logvar_contract() -> None:
    """Forward uses the clamped posterior in the derived sampling identity.

    This derived relationship protects numerical safety without freezing random model
    outputs: both saturated and interior log-variance values are forced, and nonzero
    epsilon makes the scale term observable. It catches bypassing the clamp, changing
    ``exp(0.5 * logvar)``, or reporting a clamp count unrelated to changed elements.
    """
    model = build_non_equivariant_vae()
    logvar_bias = model.logvar_head.bias
    assert logvar_bias is not None
    with torch.no_grad():
        model.logvar_head.weight.zero_()
        logvar_bias.zero_()
        logvar_bias[0] = DEFAULT_LOGVAR_CLAMP_MIN - 2.0
        logvar_bias[1] = DEFAULT_LOGVAR_CLAMP_MAX + 2.0
    clean_batch = _clean_batch(batch_size=TEST_BATCH_SIZE, image_size=TEST_IMAGE_SIZE)
    eps_shape = (
        TEST_BATCH_SIZE,
        LATENT_CHANNELS,
        TEST_IMAGE_SIZE // 8,
        TEST_IMAGE_SIZE // 8,
    )
    eps = torch.linspace(-1.0, 1.0, steps=math.prod(eps_shape)).reshape(eps_shape)

    output: VaeForwardOutput = model.forward(clean_batch, eps=eps)

    expected_logvar = output.logvar.clamp(
        min=DEFAULT_LOGVAR_CLAMP_MIN,
        max=DEFAULT_LOGVAR_CLAMP_MAX,
    )
    expected_z = output.mu + (torch.exp(0.5 * expected_logvar) * eps)
    expected_clamp_count = torch.count_nonzero(output.logvar != expected_logvar)
    assert output.reconstruction.shape == clean_batch.shape
    assert output.mu.shape == eps.shape
    assert output.logvar.shape == eps.shape
    assert output.logvar_clamped.shape == eps.shape
    assert torch.equal(output.eps, eps)
    assert torch.equal(output.logvar_clamped, expected_logvar)
    assert 0 < expected_clamp_count < output.logvar.numel()
    assert torch.equal(output.logvar_clamp_count, expected_clamp_count)
    assert torch.allclose(output.z, expected_z)
    assert math.isclose(_max_abs(output.reconstruction), 0.0, abs_tol=0.0)


def test_identity_clean_train_step_produces_finite_update() -> None:
    """A local synthetic identity-clean update has finite loss and gradients."""
    model = build_non_equivariant_vae()
    optimizer, summary = create_adamw_optimizer(model)
    clean_batch = _clean_batch(batch_size=TEST_BATCH_SIZE, image_size=TEST_IMAGE_SIZE)
    eps = torch.zeros(
        (TEST_BATCH_SIZE, LATENT_CHANNELS, TEST_IMAGE_SIZE // 8, TEST_IMAGE_SIZE // 8),
    )

    result = run_train_step(
        TrainStepRequest(
            model=model,
            optimizer=optimizer,
            clean_batch=clean_batch,
            eps=eps,
            beta=beta_for_step(
                optimizer_step_index=0,
                max_optimizer_steps=TEST_MAX_OPTIMIZER_STEPS,
            ),
            ssim_weight=0.1,
            optimizer_step_index=0,
            gradient_clip_global_norm=1.0,
        ),
    )

    assert summary.all_trainable_parameters_covered_once is True
    assert summary.gate_parameters_in_gate_no_decay_group is True
    assert math.isclose(result.losses.beta, 0.0, abs_tol=0.0)
    assert math.isfinite(result.losses.loss.item())
    assert result.grad_norm > 0.0
    assert result.param_update_norm > 0.0
    assert result.nonfinite_count == 0
    assert result.trainable_parameter_tensor_count > 0
    assert result.nonzero_grad_parameter_tensor_count > 0
    assert result.nonzero_update_parameter_tensor_count > 0
    assert result.optimizer_step_index == 0
    assert result.successful_optimizer_update_count == 1


def test_local_model_loss_train_step_artifact_is_non_promotable(
    tmp_path: Path,
) -> None:
    """The benchmark artifact records local evidence and cannot unlock Kaggle."""
    config_path = _tiny_debug_config(tmp_path)

    exit_code = benchmark_runtime_main(
        [
            "--config",
            str(config_path),
            "--data",
            "synthetic",
            "--device",
            "cpu",
            "--output-dir",
            str(tmp_path),
            "--run-name",
            "spec0001_cpu_model_loss_train_step_test",
            "--max-benchmark-rows",
            "1",
            "--warmup-steps",
            "1",
            "--measured-steps",
            "1",
            "--model-loss-train-step",
        ],
    )

    assert exit_code == 0
    artifact_path = tmp_path / "benchmark" / "model_loss_train_step.json"
    payload = _load_json(artifact_path)
    forward_contract = _object(payload, "forward_contract")
    zero_head = _object(payload, "zero_head")
    optimizer = _object(payload, "optimizer")
    backward_update = _object(payload, "backward_update")
    loss = _object(payload, "loss")

    assert payload["status"] == "local_pass"
    assert payload["benchmark_kind"] == "local_synthetic_model_loss_train_step"
    assert payload["benchmark_source"] == "local_cpu_synthetic_train_step"
    assert payload["corruption_strategy"] == "identity_clean_no_corruption"
    assert payload["full_run_eligible"] is False
    assert payload["model_count_status"] == "pass"
    assert payload["matches_spec_target"] is True
    assert payload["model_count_sha256"] == _sha256(
        tmp_path / "benchmark" / "model_count.json",
    )
    assert not (tmp_path / "benchmark" / "selected_runtime.json").exists()
    assert forward_contract["explicit_eps_used"] is True
    assert zero_head["status"] == "pass"
    assert math.isclose(
        _float(zero_head, "initial_reconstruction_max_abs"),
        0.0,
        abs_tol=0.0,
    )
    assert loss["all_finite"] is True
    assert optimizer["all_trainable_parameters_covered_once"] is True
    assert optimizer["gate_parameters_in_gate_no_decay_group"] is True
    assert _float(backward_update, "grad_norm") > 0.0
    assert _float(backward_update, "param_update_norm") > 0.0
    assert _float(backward_update, "nonzero_grad_parameter_tensor_count") > 0.0
    assert _float(backward_update, "nonzero_update_parameter_tensor_count") > 0.0
    assert backward_update["first_step_update_scope"] == (
        "zero_head_final_rgb_head_smoke"
    )
    assert backward_update["nonfinite_count"] == 0
    assert backward_update["optimizer_step_index"] == 0
    assert backward_update["successful_optimizer_update_count"] == 1


def test_local_model_loss_train_step_rejects_promotable_config(
    tmp_path: Path,
) -> None:
    """Local model/loss evidence fails closed if the config claims promotion."""
    config_path = _tiny_debug_config(
        tmp_path,
        model_loss_overrides={"full_run_eligible": True},
    )

    with pytest.raises(ValueError, match="must not be full-run eligible"):
        benchmark_runtime_main(
            [
                "--config",
                str(config_path),
                "--data",
                "synthetic",
                "--device",
                "cpu",
                "--output-dir",
                str(tmp_path),
                "--run-name",
                "spec0001_cpu_model_loss_train_step_test",
                "--model-loss-train-step",
            ],
        )

    assert not (tmp_path / "benchmark" / "model_count.json").exists()
    assert not (tmp_path / "benchmark" / "model_loss_train_step.json").exists()


def test_local_model_loss_train_step_rejects_logvar_clamp_drift(
    tmp_path: Path,
) -> None:
    """Config clamp values must stay tied to the implementation constants."""
    config_path = _tiny_debug_config(
        tmp_path,
        objective_overrides={"logvar_clamp": [-9.0, 4.0]},
    )

    with pytest.raises(ValueError, match=r"objective\.logvar_clamp"):
        benchmark_runtime_main(
            [
                "--config",
                str(config_path),
                "--data",
                "synthetic",
                "--device",
                "cpu",
                "--output-dir",
                str(tmp_path),
                "--run-name",
                "spec0001_cpu_model_loss_train_step_test",
                "--model-loss-train-step",
            ],
        )

    assert not (tmp_path / "benchmark" / "model_count.json").exists()
    assert not (tmp_path / "benchmark" / "model_loss_train_step.json").exists()


def _tiny_debug_config(
    tmp_path: Path,
    *,
    model_loss_overrides: dict[str, object] | None = None,
    objective_overrides: dict[str, object] | None = None,
) -> Path:
    config_path = tmp_path / "tiny_model_loss_config.json"
    model_loss_config: dict[str, object] = {
        "benchmark_kind": "local_synthetic_model_loss_train_step",
        "benchmark_source": "local_cpu_synthetic_train_step",
        "full_run_eligible": False,
        "corruption_strategy": "identity_clean_no_corruption",
        "batch_size": TEST_BATCH_SIZE,
        "max_optimizer_steps": TEST_MAX_OPTIMIZER_STEPS,
        "required_precision_policy": "amp_off_fp32",
        "optional_compile_smoke": False,
        "optional_float16_smoke": False,
    }
    if model_loss_overrides is not None:
        model_loss_config.update(model_loss_overrides)

    objective_config: dict[str, object] = {}
    if objective_overrides is not None:
        objective_config.update(objective_overrides)

    payload = {
        "source_config": str(
            Path("configs/spec0001/non_eq_vae_debug_cpu.json").resolve(),
        ),
        "run": {
            "name": "spec0001_cpu_model_loss_train_step_test",
            "mode": "local_synthetic_debug",
        },
        "data": {
            "image_size": TEST_IMAGE_SIZE,
            "train_samples": 2,
            "validation_samples": 2,
        },
        "runtime": {
            "batch_size": TEST_BATCH_SIZE,
            "max_train_steps": TEST_MAX_OPTIMIZER_STEPS,
            "max_val_steps": 1,
        },
        "objective": objective_config,
        "model_loss_train_step": model_loss_config,
    }
    config_path.write_text(
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    return config_path


def _clean_batch(*, batch_size: int, image_size: int) -> torch.Tensor:
    values = torch.linspace(
        -1.0,
        1.0,
        steps=batch_size * 3 * image_size * image_size,
        dtype=torch.float32,
    )
    return values.reshape(batch_size, 3, image_size, image_size)


def _load_json(path: Path) -> dict[str, object]:
    payload = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(payload, dict):
        raise TypeError(path)
    return cast("dict[str, object]", payload)


def _object(payload: dict[str, object], key: str) -> dict[str, object]:
    value = payload[key]
    if not isinstance(value, dict):
        raise TypeError(key)
    return cast("dict[str, object]", value)


def _float(payload: dict[str, object], key: str) -> float:
    value = payload[key]
    if isinstance(value, bool):
        raise TypeError(key)
    if isinstance(value, int | float):
        return float(value)
    raise TypeError(key)


def _max_abs(tensor: torch.Tensor) -> float:
    return float(tensor.detach().to(dtype=torch.float32).abs().max().item())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
