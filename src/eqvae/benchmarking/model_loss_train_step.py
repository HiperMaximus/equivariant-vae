# Copyright 2026 HiperMaximus
"""Local model/loss train-step evidence writer for spec 0001."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

import torch

from eqvae.benchmarking.io import JsonObject, JsonValue, write_json
from eqvae.benchmarking.model_count import write_model_count
from eqvae.config import resolve_json_config
from eqvae.losses.vae import beta_for_step, beta_warmup_steps
from eqvae.metrics.reconstruction import reconstruction_metric_summaries
from eqvae.models.non_equivariant_vae import (
    DEFAULT_GROUPNORM_GROUPS,
    DEFAULT_LOGVAR_CLAMP_MAX,
    DEFAULT_LOGVAR_CLAMP_MIN,
    LATENT_CHANNELS,
)
from eqvae.models.registry import MODEL_KIND_NON_EQ_TRANSLATABLE, build_model
from eqvae.training.optim import SpecAdamWConfig, create_adamw_optimizer
from eqvae.training.step import TrainStepRequest, TrainStepResult, run_train_step

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from eqvae.config import ResolvedConfig
    from eqvae.models.non_equivariant_vae import NonEquivariantVAE, VaeForwardOutput
    from eqvae.training.optim import OptimizerGroupSummary

_ADAM_BETA_COUNT = 2
_LOCAL_TRAIN_STEP_KIND = "local_synthetic_model_loss_train_step"
_LOCAL_TRAIN_STEP_SOURCE = "local_cpu_synthetic_train_step"
_IDENTITY_CORRUPTION_STRATEGY = "identity_clean_no_corruption"
_REQUIRED_PRECISION_POLICY = "amp_off_fp32"
_OPTIONAL_SMOKE_BATCH_SIZE = 1
_OPTIONAL_SMOKE_IMAGE_SIZE = 64
_LOGVAR_CLAMP_COUNT = 2


class _CompiledVae(Protocol):
    """Typed call protocol for a locally compiled VAE."""

    def __call__(
        self,
        inputs: torch.Tensor,
        *,
        eps: torch.Tensor | None = None,
    ) -> VaeForwardOutput:
        """Run the compiled VAE."""
        ...


@dataclass(frozen=True)
class LocalModelLossTrainStepRequest:
    """Inputs for local model/loss train-step evidence."""

    config_path: Path
    output_dir: Path
    run_name: str


@dataclass(frozen=True)
class _ArtifactPaths:
    model_count: Path
    train_step: Path


@dataclass(frozen=True)
class _TrainStepSettings:
    batch_size: int
    image_size: int
    max_optimizer_steps: int
    ssim_weight: float
    beta: float
    warmup_fraction: float
    optimizer_config: SpecAdamWConfig
    train_config: JsonObject


@dataclass(frozen=True)
class _ZeroHeadProof:
    weight_zero: bool
    bias_zero: bool


@dataclass(frozen=True)
class _SmokeInputs:
    clean_batch: torch.Tensor
    eps: torch.Tensor


@dataclass(frozen=True)
class _SmokeResults:
    compile: JsonObject
    float16: JsonObject


@dataclass(frozen=True)
class _PayloadContext:
    request: LocalModelLossTrainStepRequest
    resolved_config: ResolvedConfig
    paths: _ArtifactPaths
    settings: _TrainStepSettings
    clean_batch: torch.Tensor
    eps: torch.Tensor
    model: NonEquivariantVAE
    zero_head_proof: _ZeroHeadProof
    smoke_results: _SmokeResults
    result: TrainStepResult
    optimizer_summary: OptimizerGroupSummary


def write_local_model_loss_train_step(
    request: LocalModelLossTrainStepRequest,
) -> Path:
    """Write non-promotable local model/loss train-step evidence.

    Returns:
        Path to `benchmark/model_loss_train_step.json`.

    """
    paths = _artifact_paths(request)
    resolved_config = resolve_json_config(request.config_path)
    effective_config = resolved_config.effective_config
    settings = _train_step_settings(effective_config)
    model_count_payload = write_model_count(
        config_path=request.config_path,
        output_path=paths.model_count,
    )
    _require_model_count_pass(
        model_count_payload=model_count_payload,
        effective_config_hash=resolved_config.effective_config_hash,
        architecture_id=_required_string(
            _required_object(effective_config, "model"),
            "architecture_id",
        ),
        topology_version=_required_string(
            _required_object(effective_config, "model"),
            "topology_version",
        ),
    )

    model = build_model(
        MODEL_KIND_NON_EQ_TRANSLATABLE,
        model_config={"norm_groups": _norm_groups(effective_config)},
    )
    optimizer, optimizer_summary = create_adamw_optimizer(
        model,
        config=settings.optimizer_config,
    )
    clean_batch = _synthetic_clean_batch(
        batch_size=settings.batch_size,
        image_size=settings.image_size,
        seed=_data_seed(effective_config),
    )
    eps = _zero_latent_eps(settings)
    smoke_inputs = _smoke_inputs(seed=_data_seed(effective_config))
    zero_head_proof = _zero_head_proof(model)
    smoke_results = _smoke_results(
        model=model,
        inputs=smoke_inputs,
        settings=settings,
    )
    result = run_train_step(
        TrainStepRequest(
            model=model,
            optimizer=optimizer,
            clean_batch=clean_batch,
            eps=eps,
            beta=settings.beta,
            ssim_weight=settings.ssim_weight,
            optimizer_step_index=0,
            gradient_clip_global_norm=(
                settings.optimizer_config.gradient_clip_global_norm
            ),
        ),
    )
    payload = _artifact_payload(
        _PayloadContext(
            request=request,
            resolved_config=resolved_config,
            paths=paths,
            settings=settings,
            clean_batch=clean_batch,
            eps=eps,
            model=model,
            zero_head_proof=zero_head_proof,
            smoke_results=smoke_results,
            result=result,
            optimizer_summary=optimizer_summary,
        ),
    )
    write_json(paths.train_step, payload)
    return paths.train_step


def _artifact_paths(request: LocalModelLossTrainStepRequest) -> _ArtifactPaths:
    benchmark_dir = request.output_dir / "benchmark"
    return _ArtifactPaths(
        model_count=benchmark_dir / "model_count.json",
        train_step=benchmark_dir / "model_loss_train_step.json",
    )


def _train_step_settings(effective_config: JsonObject) -> _TrainStepSettings:
    train_config = _required_object(effective_config, "model_loss_train_step")
    _validate_train_step_rail(train_config)
    data_config = _required_object(effective_config, "data")
    objective_config = _required_object(effective_config, "objective")
    _validate_logvar_clamp(objective_config)
    beta_config = _required_object(objective_config, "beta")
    max_optimizer_steps = _required_int(train_config, "max_optimizer_steps")
    warmup_fraction = _required_float(beta_config, "step_limited_warmup_fraction")
    return _TrainStepSettings(
        batch_size=_required_int(train_config, "batch_size"),
        image_size=_required_int(data_config, "image_size"),
        max_optimizer_steps=max_optimizer_steps,
        ssim_weight=_required_float(objective_config, "ssim_weight"),
        beta=beta_for_step(
            optimizer_step_index=0,
            max_optimizer_steps=max_optimizer_steps,
            target_beta=_required_float(beta_config, "target"),
            warmup_fraction=warmup_fraction,
        ),
        warmup_fraction=warmup_fraction,
        optimizer_config=_optimizer_config(effective_config),
        train_config=train_config,
    )


def _zero_latent_eps(settings: _TrainStepSettings) -> torch.Tensor:
    return torch.zeros(
        (
            settings.batch_size,
            LATENT_CHANNELS,
            settings.image_size // 8,
            settings.image_size // 8,
        ),
        dtype=torch.float32,
    )


def _smoke_inputs(*, seed: int) -> _SmokeInputs:
    return _SmokeInputs(
        clean_batch=_synthetic_clean_batch(
            batch_size=_OPTIONAL_SMOKE_BATCH_SIZE,
            image_size=_OPTIONAL_SMOKE_IMAGE_SIZE,
            seed=seed,
        ),
        eps=torch.zeros(
            (
                _OPTIONAL_SMOKE_BATCH_SIZE,
                LATENT_CHANNELS,
                _OPTIONAL_SMOKE_IMAGE_SIZE // 8,
                _OPTIONAL_SMOKE_IMAGE_SIZE // 8,
            ),
            dtype=torch.float32,
        ),
    )


def _smoke_results(
    *,
    model: NonEquivariantVAE,
    inputs: _SmokeInputs,
    settings: _TrainStepSettings,
) -> _SmokeResults:
    return _SmokeResults(
        compile=_compile_smoke_payload(
            model=model,
            clean_batch=inputs.clean_batch,
            eps=inputs.eps,
            enabled=_optional_bool(settings.train_config, "optional_compile_smoke"),
        ),
        float16=_float16_smoke_payload(
            model=model,
            clean_batch=inputs.clean_batch,
            eps=inputs.eps,
            enabled=_optional_bool(settings.train_config, "optional_float16_smoke"),
        ),
    )


def _artifact_payload(context: _PayloadContext) -> JsonObject:
    effective_config = context.resolved_config.effective_config
    model_config = _required_object(effective_config, "model")
    result_passed = _result_passed(context.result)
    payload: JsonObject = {
        "status": "local_pass" if result_passed else "fail",
        "benchmark_kind": _LOCAL_TRAIN_STEP_KIND,
        "benchmark_source": _LOCAL_TRAIN_STEP_SOURCE,
        "full_run_eligible": False,
        "run_name": context.request.run_name,
        "config_path": str(context.request.config_path),
        "config_sha256": context.resolved_config.invoked_config_hash,
        "effective_config_sha256": context.resolved_config.effective_config_hash,
        "architecture_id": _required_string(model_config, "architecture_id"),
        "topology_version": _required_string(model_config, "topology_version"),
        "model_count_path": "benchmark/model_count.json",
        "model_count_sha256": _sha256_file(context.paths.model_count),
        "model_count_status": "pass",
        "matches_spec_target": True,
        "accelerator_mode": "local_cpu",
        "machine_shape": "local_cpu",
        "device": "cpu",
        "precision_policy": "amp_off_fp32",
        "amp_enabled": False,
        "torch_compile": context.smoke_results.compile,
        "float16_smoke": context.smoke_results.float16,
        "corruption_strategy": _required_string(
            context.settings.train_config,
            "corruption_strategy",
        ),
        "batch_size": context.settings.batch_size,
        "input_shape": list(context.clean_batch.shape),
        "latent_shape": list(context.result.forward.z.shape),
        "forward_contract": {
            "explicit_eps_used": bool(
                torch.equal(context.result.forward.eps, context.eps),
            ),
            "returned_reconstruction": True,
            "returned_mu": True,
            "returned_logvar_raw": True,
            "returned_logvar_clamped": True,
            "returned_z": True,
            "returned_eps": True,
            "returned_logvar_clamp_count": True,
        },
        "zero_head": _zero_head_payload(context),
        "loss": {
            **context.result.losses.detached_scalars(),
            "all_finite": _losses_are_finite(
                context.result.losses.detached_scalars(),
            ),
        },
        "posterior": _posterior_payload(context.result),
        "optimizer": {
            "name": "AdamW",
            "parameter_group_count": context.optimizer_summary.parameter_group_count,
            "all_trainable_parameters_covered_once": (
                context.optimizer_summary.all_trainable_parameters_covered_once
            ),
            "gate_parameters_in_gate_no_decay_group": (
                context.optimizer_summary.gate_parameters_in_gate_no_decay_group
            ),
            "base_lr": context.settings.optimizer_config.learning_rate,
            "weight_decay": context.settings.optimizer_config.weight_decay,
            "gate_lr_multiplier": context.settings.optimizer_config.gate_lr_multiplier,
        },
        "backward_update": {
            "grad_norm": context.result.grad_norm,
            "param_update_norm": context.result.param_update_norm,
            "nonfinite_count": context.result.nonfinite_count,
            "trainable_parameter_tensor_count": (
                context.result.trainable_parameter_tensor_count
            ),
            "nonzero_grad_parameter_tensor_count": (
                context.result.nonzero_grad_parameter_tensor_count
            ),
            "nonzero_update_parameter_tensor_count": (
                context.result.nonzero_update_parameter_tensor_count
            ),
            "first_step_update_scope": "zero_head_final_rgb_head_smoke",
            "optimizer_step_index": context.result.optimizer_step_index,
            "successful_optimizer_update_count": (
                context.result.successful_optimizer_update_count
            ),
            "beta_warmup_steps": beta_warmup_steps(
                context.settings.max_optimizer_steps,
                warmup_fraction=context.settings.warmup_fraction,
            ),
        },
        "metrics": _metric_payload(
            result=context.result,
            clean_batch=context.clean_batch,
        ),
        "failure_kind": "" if result_passed else "local_train_step_fail",
        "failure_message_hash": "",
    }
    return payload


def _zero_head_payload(context: _PayloadContext) -> JsonObject:
    return {
        "weight_zero": context.zero_head_proof.weight_zero,
        "bias_zero": context.zero_head_proof.bias_zero,
        "initial_reconstruction_max_abs": _max_abs(
            context.result.forward.reconstruction,
        ),
        "status": (
            "pass"
            if context.zero_head_proof.weight_zero and context.zero_head_proof.bias_zero
            else "fail"
        ),
    }


def _zero_head_proof(model: NonEquivariantVAE) -> _ZeroHeadProof:
    output_head = model.output_head
    return _ZeroHeadProof(
        weight_zero=bool(
            torch.equal(
                output_head.weight,
                torch.zeros_like(output_head.weight),
            ),
        ),
        bias_zero=bool(
            output_head.bias is not None
            and torch.equal(output_head.bias, torch.zeros_like(output_head.bias)),
        ),
    )


def _require_model_count_pass(
    *,
    model_count_payload: JsonObject,
    effective_config_hash: str,
    architecture_id: str,
    topology_version: str,
) -> None:
    if model_count_payload.get("status") != "pass":
        message = "model_count proof did not pass"
        raise RuntimeError(message)
    if model_count_payload.get("matches_spec_target") is not True:
        message = "model_count proof does not match the spec target"
        raise RuntimeError(message)
    if model_count_payload.get("effective_config_hash") != effective_config_hash:
        message = "model_count effective config hash differs from train-step config"
        raise RuntimeError(message)
    if model_count_payload.get("architecture_id") != architecture_id:
        message = "model_count architecture_id differs from train-step config"
        raise RuntimeError(message)
    if model_count_payload.get("topology_version") != topology_version:
        message = "model_count topology_version differs from train-step config"
        raise RuntimeError(message)


def _synthetic_clean_batch(
    *,
    batch_size: int,
    image_size: int,
    seed: int,
) -> torch.Tensor:
    if image_size % 8 != 0:
        message = f"image_size must be divisible by 8, got {image_size}"
        raise ValueError(message)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return (
        torch.rand(
            (batch_size, 3, image_size, image_size),
            generator=generator,
            dtype=torch.float32,
        )
        * 2.0
    ) - 1.0


def _metric_payload(
    *,
    result: TrainStepResult,
    clean_batch: torch.Tensor,
) -> JsonObject:
    with torch.no_grad():
        summaries = reconstruction_metric_summaries(
            result.forward.reconstruction.detach(),
            clean_batch.detach(),
        )
    payload: JsonObject = {}
    for metric_name, summary in summaries.items():
        mean = summary.mean if summary.mean is not None else summary.finite_mean
        payload[metric_name] = mean
    return payload


def _posterior_payload(result: TrainStepResult) -> JsonObject:
    raw = result.forward.logvar.to(dtype=torch.float32)
    clamped = result.forward.logvar_clamped.to(dtype=torch.float32)
    clamp_count = int(result.forward.logvar_clamp_count.item())
    return {
        "mu_mean": _mean(result.forward.mu),
        "mu_std": _std(result.forward.mu),
        "logvar_raw_mean": _mean(raw),
        "logvar_raw_std": _std(raw),
        "logvar_clamped_mean": _mean(clamped),
        "logvar_clamped_std": _std(clamped),
        "logvar_clamp_count": clamp_count,
        "logvar_clamp_fraction": clamp_count / raw.numel(),
    }


def _optimizer_config(config: JsonObject) -> SpecAdamWConfig:
    optimizer_config = _required_object(config, "optimizer")
    betas = _required_list(optimizer_config, "betas")
    if len(betas) != _ADAM_BETA_COUNT:
        message = "optimizer.betas must contain exactly two values"
        raise ValueError(message)
    gate_group = _required_object(
        _required_object(optimizer_config, "parameter_groups"),
        "gate_no_decay",
    )
    return SpecAdamWConfig(
        learning_rate=_required_float(optimizer_config, "learning_rate"),
        beta1=_float_value(betas[0], key="optimizer.betas[0]"),
        beta2=_float_value(betas[1], key="optimizer.betas[1]"),
        epsilon=_required_float(optimizer_config, "epsilon"),
        weight_decay=_required_float(optimizer_config, "weight_decay"),
        gradient_clip_global_norm=_required_float(
            optimizer_config,
            "gradient_clip_global_norm",
        ),
        gate_lr_multiplier=_required_float(gate_group, "lr_multiplier"),
    )


def _validate_logvar_clamp(objective_config: JsonObject) -> None:
    clamp_values = _required_list(objective_config, "logvar_clamp")
    if len(clamp_values) != _LOGVAR_CLAMP_COUNT:
        message = "objective.logvar_clamp must contain exactly two numbers"
        raise ValueError(message)
    minimum = _float_value(clamp_values[0], key="objective.logvar_clamp[0]")
    maximum = _float_value(clamp_values[1], key="objective.logvar_clamp[1]")
    if not math.isclose(
        minimum,
        DEFAULT_LOGVAR_CLAMP_MIN,
        abs_tol=0.0,
    ) or not math.isclose(maximum, DEFAULT_LOGVAR_CLAMP_MAX, abs_tol=0.0):
        message = (
            "objective.logvar_clamp must match the implementation constants "
            f"[{DEFAULT_LOGVAR_CLAMP_MIN}, {DEFAULT_LOGVAR_CLAMP_MAX}]"
        )
        raise ValueError(message)


def _norm_groups(config: JsonObject) -> int:
    model_config = _required_object(config, "model")
    normalization = _required_object(model_config, "normalization")
    groups = normalization.get("num_groups")
    if groups is None:
        return DEFAULT_GROUPNORM_GROUPS
    return _int_value(groups, key="model.normalization.num_groups")


def _data_seed(config: JsonObject) -> int:
    seeds = _required_object(config, "seeds")
    return _required_int(seeds, "data_seed")


def _result_passed(result: TrainStepResult) -> bool:
    scalar_values = result.losses.detached_scalars()
    return (
        _losses_are_finite(scalar_values)
        and result.grad_norm > 0.0
        and result.param_update_norm > 0.0
        and result.nonzero_grad_parameter_tensor_count > 0
        and result.nonzero_update_parameter_tensor_count > 0
        and result.nonfinite_count == 0
    )


def _losses_are_finite(values: dict[str, float]) -> bool:
    return all(math.isfinite(value) for value in values.values())


def _compile_smoke_payload(
    *,
    model: NonEquivariantVAE,
    clean_batch: torch.Tensor,
    eps: torch.Tensor,
    enabled: bool,
) -> JsonObject:
    if not enabled:
        return _optional_smoke_payload(enabled=False, failure_kind="")
    try:
        compiled = _compile_eager(model)
        with torch.no_grad():
            output = compiled(clean_batch, eps=eps)
        _validate_smoke_output(output=output, clean_batch=clean_batch, eps=eps)
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        return _optional_smoke_payload(
            enabled=True,
            failure_kind=_smoke_failure_kind("local_cpu_compile_smoke", exc),
        )
    return _smoke_pass_payload(enabled=True)


def _float16_smoke_payload(
    *,
    model: NonEquivariantVAE,
    clean_batch: torch.Tensor,
    eps: torch.Tensor,
    enabled: bool,
) -> JsonObject:
    if not enabled:
        return _optional_smoke_payload(enabled=False, failure_kind="")
    try:
        with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.float16):
            output: VaeForwardOutput = model.forward(clean_batch, eps=eps)
        _validate_smoke_output(output=output, clean_batch=clean_batch, eps=eps)
    except (RuntimeError, TypeError, ValueError) as exc:
        return _optional_smoke_payload(
            enabled=True,
            failure_kind=_smoke_failure_kind("local_cpu_float16_smoke", exc),
        )
    return _smoke_pass_payload(enabled=True)


def _compile_eager(model: NonEquivariantVAE) -> _CompiledVae:
    compile_fn = cast("Callable[..., _CompiledVae]", torch.compile)
    return compile_fn(model, backend="eager")


def _validate_smoke_output(
    *,
    output: VaeForwardOutput,
    clean_batch: torch.Tensor,
    eps: torch.Tensor,
) -> None:
    if output.reconstruction.shape != clean_batch.shape:
        message = "Smoke reconstruction shape does not match input batch"
        raise ValueError(message)
    if not bool(torch.equal(output.eps, eps)):
        message = "Smoke forward did not preserve explicit eps"
        raise ValueError(message)
    if not bool(torch.isfinite(output.reconstruction).all().item()):
        message = "Smoke reconstruction contains non-finite values"
        raise ValueError(message)


def _smoke_pass_payload(*, enabled: bool) -> JsonObject:
    return {"enabled": enabled, "status": "local_pass", "failure_kind": ""}


def _smoke_failure_kind(prefix: str, error: Exception) -> str:
    return f"{prefix}_{type(error).__name__}"


def _optional_smoke_payload(*, enabled: bool, failure_kind: str) -> JsonObject:
    if not enabled:
        return {
            "enabled": False,
            "status": "skipped_unsupported",
            "failure_kind": "optional_smoke_disabled",
        }
    return {
        "enabled": True,
        "status": "skipped_unsupported",
        "failure_kind": failure_kind,
    }


def _required_object(payload: JsonObject, key: str) -> JsonObject:
    value = payload.get(key)
    if isinstance(value, dict):
        return cast("JsonObject", value)
    message = f"Expected object at `{key}`"
    raise TypeError(message)


def _required_list(payload: JsonObject, key: str) -> list[JsonValue]:
    value = payload.get(key)
    if isinstance(value, list):
        return value
    message = f"Expected list at `{key}`"
    raise TypeError(message)


def _required_string(payload: JsonObject, key: str) -> str:
    value = payload.get(key)
    if isinstance(value, str):
        return value
    message = f"Expected string at `{key}`"
    raise TypeError(message)


def _required_int(payload: JsonObject, key: str) -> int:
    value = payload.get(key)
    return _int_value(value, key=key)


def _required_float(payload: JsonObject, key: str) -> float:
    value = payload.get(key)
    return _float_value(value, key=key)


def _required_bool(payload: JsonObject, key: str) -> bool:
    value = payload.get(key)
    if isinstance(value, bool):
        return value
    message = f"Expected boolean at `{key}`"
    raise TypeError(message)


def _optional_bool(payload: JsonObject, key: str) -> bool:
    value = payload.get(key)
    if isinstance(value, bool):
        return value
    return False


def _int_value(value: object, *, key: str) -> int:
    if type(value) is int:
        return value
    message = f"Expected integer at `{key}`"
    raise TypeError(message)


def _float_value(value: object, *, key: str) -> float:
    if type(value) in {float, int}:
        return float(cast("float | int", value))
    message = f"Expected number at `{key}`"
    raise TypeError(message)


def _mean(tensor: torch.Tensor) -> float:
    return float(tensor.detach().to(dtype=torch.float32).mean().item())


def _std(tensor: torch.Tensor) -> float:
    return float(tensor.detach().to(dtype=torch.float32).std(unbiased=False).item())


def _max_abs(tensor: torch.Tensor) -> float:
    return float(tensor.detach().to(dtype=torch.float32).abs().max().item())


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_train_step_rail(train_config: JsonObject) -> None:
    if _required_string(train_config, "benchmark_kind") != _LOCAL_TRAIN_STEP_KIND:
        message = (
            f"`model_loss_train_step.benchmark_kind` must be {_LOCAL_TRAIN_STEP_KIND}"
        )
        raise ValueError(message)
    if _required_string(train_config, "benchmark_source") != _LOCAL_TRAIN_STEP_SOURCE:
        message = (
            "`model_loss_train_step.benchmark_source` must be "
            f"{_LOCAL_TRAIN_STEP_SOURCE}"
        )
        raise ValueError(message)
    if _required_bool(train_config, "full_run_eligible"):
        message = "Local model/loss train-step must not be full-run eligible"
        raise ValueError(message)
    if (
        _required_string(train_config, "corruption_strategy")
        != _IDENTITY_CORRUPTION_STRATEGY
    ):
        message = (
            "`model_loss_train_step.corruption_strategy` must be "
            f"{_IDENTITY_CORRUPTION_STRATEGY}"
        )
        raise ValueError(message)
    if (
        _required_string(train_config, "required_precision_policy")
        != _REQUIRED_PRECISION_POLICY
    ):
        message = (
            "`model_loss_train_step.required_precision_policy` must be "
            f"{_REQUIRED_PRECISION_POLICY}"
        )
        raise ValueError(message)


__all__ = [
    "LocalModelLossTrainStepRequest",
    "write_local_model_loss_train_step",
]
