# Copyright 2026 HiperMaximus
"""One-shot generated-data dual-T4 readiness proof for the fixed SO2 VAE."""
# pyright: reportAny=false, reportArgumentType=false, reportAssignmentType=false
# pyright: reportCallIssue=false, reportPrivateUsage=false, reportReturnType=false
# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# ruff: noqa: DOC201, DOC501, E501, EM101, EM102, PLR0913, PLR0914, PLR2004, RUF069, TRY003

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import torch
import torch._dynamo as torch_dynamo  # noqa: PLC2701
import torch.distributed as dist
from torch import nn
from torch._dynamo.utils import counters  # noqa: PLC2701
from torch._inductor import config as inductor_config  # noqa: PLC2701
from torch.amp import GradScaler

from eqvae.benchmarking.io import write_csv, write_json
from eqvae.benchmarking.runtime_schema import GATE_HEALTH_COLUMNS
from eqvae.benchmarking.torch_runtime import torch_runtime_versions
from eqvae.config import JsonObject, resolve_json_config
from eqvae.corruption.inline_stain import InlineStainCorruptor
from eqvae.corruption.stain import profile_from_config
from eqvae.data.dataloaders import normalize_uint8_batch
from eqvae.models.registry import (
    MODEL_KIND_SO2_FIXED,
    assert_fixed_so2_model,
    build_model,
)
from eqvae.models.so2_architecture_probe import (
    _RADIUS_EPS,
    FixedF01FieldNorm,
    FixedF01RadialGate,
)
from eqvae.training.fastpath_precision import (
    EXPECTED_RUNNER_AMP_GRAD_SCALER_INIT_SCALE,
    run_fastpath_optimizer_step_with_metrics,
)
from eqvae.training.fastpath_recipe import (
    FastpathDynamoKnobs,
    apply_cudnn_flags,
    build_fastpath_optimizer,
    compiled_autograd_context,
    model_requires_buffer_broadcast,
    resolve_fastpath_compile_invocation,
    wrap_fastpath_ddp,
)
from eqvae.training.fastpath_step import FastpathStepOutput, make_fastpath_step_fn
from eqvae.training.optim import (
    BatchLrScaling,
    SpecAdamWConfig,
    build_adamw_parameter_groups,
    scaled_learning_rate,
)
from eqvae.training.selected_runtime import (
    SelectedRuntimePlan,
    parse_selected_runtime_plan,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.nn.parallel import DistributedDataParallel

    from eqvae.benchmarking.io import CsvRow
    from eqvae.models.so2_vae import SO2VAE

SCHEMA_VERSION: Final = "spec0015.so2_selected_runtime_readiness.v1"
PROBE_KIND: Final = "fixed_so2_selected_runtime_readiness"
ARTIFACT_FILENAME: Final = "spec0015_so2_runtime_readiness.json"
GATE_FILENAME: Final = "spec0015_so2_gate_health.csv"
FIELD_NORM_COUNT: Final = 40
RADIAL_GATE_COUNT: Final = 34
DDP_REFERENCE_LIMIT: Final = 1e-6
GATE_LOW_THRESHOLD: Final = 0.01
GATE_HIGH_THRESHOLD: Final = 0.99
REQUIRED_WORLD_SIZE: Final = 2
REQUIRED_GPU_NAME: Final = "Tesla T4"
PER_DEVICE_BATCH: Final = 1
IMAGE_SIZE: Final = 256
LATENT_SIZE: Final = 32
WARMUP_UPDATES: Final = 3
SETTLED_UPDATES: Final = 3
GATE_ROW_COUNT: Final = 68
PRIMARY_RANK: Final = 0
BYTES_PER_MIB: Final = 1024.0**2
EXPECTED_RUNTIME_POLICY_ID: Final = "compile_step_python_reducer_fp16_channels_last"
EXPECTED_RUNTIME_ARTIFACT_SHA256: Final = (
    "e9e998fd161f0955959c64aed7cd7ddbdfcb55a271b9ce05805903c97c93efb8"
)
EXPECTED_SELECTED_ROW_ID: Final = (
    "dual_t4_ddp__bs25__amp_conservative__compile_step__indexed_masked__"
    "policy_compile_step_python_reducer_fp16_channels_last"
)
_NAMED_UPDATE_PARAMETERS: Final = {
    "output_head": "output_head.coeff00",
    "decoder": "decoder_blocks.7.main_conv2.coeff00",
    "posterior": "mu_head.coeff00",
    "encoder": "encoder_blocks.7.main_conv2.coeff00",
    "stem": "stem_conv.coeff00",
    "f0_gate": "stem_gate.f0_a",
    "f1_gate": "stem_gate.f1_a",
}


@dataclass(frozen=True)
class _Distributed:
    device: torch.device
    rank: int
    local_rank: int
    world_size: int


@dataclass(frozen=True)
class _ProbeConfig:
    run_name: str
    warmup_updates: int
    settled_updates: int
    shared_contract_path: Path
    runtime_path: Path


def _read_object(path: Path) -> JsonObject:
    payload = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(payload, dict):
        message = f"expected JSON object: {path}"
        raise TypeError(message)
    return cast("JsonObject", payload)


def _required_object(payload: JsonObject, key: str) -> JsonObject:
    value = payload.get(key)
    if not isinstance(value, dict):
        message = f"expected object at {key!r}"
        raise TypeError(message)
    return cast("JsonObject", value)


def _required_str(payload: JsonObject, key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        message = f"expected nonempty string at {key!r}"
        raise TypeError(message)
    return value


def _required_int(payload: JsonObject, key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        message = f"expected integer at {key!r}"
        raise TypeError(message)
    return value


def parse_probe_config(path: Path) -> _ProbeConfig:
    """Parse and fail closed on the one readiness coordinate.

    Returns:
        Validated local configuration.

    """
    payload = _read_object(path)
    model = _required_object(payload, "model")
    data = _required_object(payload, "data")
    readiness = _required_object(payload, "readiness")
    run = _required_object(payload, "run")
    expected = {
        "schema": payload.get("schema_version") == "spec0015.so2_readiness.v1",
        "model_kind": model.get("kind") == MODEL_KIND_SO2_FIXED,
        "architecture": model.get("architecture_id") == "spec0014_fixed_f01_so2_vae",
        "data_kind": data.get("kind") == "generated_device_resident",
        "no_sources": data.get("dataset_sources") == [],
        "world_size": readiness.get("world_size") == REQUIRED_WORLD_SIZE,
        "batch": readiness.get("per_device_batch_size") == PER_DEVICE_BATCH,
        "image_size": data.get("image_size") == IMAGE_SIZE,
        "training_forbidden": readiness.get("full_training_authorized") is False,
    }
    failures = sorted(name for name, passed in expected.items() if not passed)
    if failures:
        message = f"Spec 0015 readiness config drift: {failures}"
        raise ValueError(message)
    return _ProbeConfig(
        run_name=_required_str(run, "name"),
        warmup_updates=_required_int(readiness, "warmup_updates"),
        settled_updates=_required_int(readiness, "settled_updates"),
        shared_contract_path=Path(_required_str(payload, "shared_training_contract")),
        runtime_path=Path(_required_str(readiness, "runtime_config")),
    )


def validate_selected_plan(plan: SelectedRuntimePlan) -> None:
    """Reject any runtime drift instead of silently adapting the probe."""
    facts = {
        "artifact_sha256": plan.artifact_sha256 == EXPECTED_RUNTIME_ARTIFACT_SHA256,
        "selected_row": plan.selected_row_id == EXPECTED_SELECTED_ROW_ID,
        "runtime_policy": plan.runtime_policy_id == EXPECTED_RUNTIME_POLICY_ID,
        "machine": plan.accelerator_mode == "dual_t4_ddp"
        and plan.machine_shape == "NvidiaTeslaT4",
        "launch": plan.world_size == REQUIRED_WORLD_SIZE
        and plan.nproc_per_node == REQUIRED_WORLD_SIZE
        and plan.torchrun_standalone,
        "selected_batch": plan.per_device_batch_size == 25
        and plan.global_batch_size == 50
        and plan.gradient_accumulation_steps == 1,
        "compiled_step": plan.torch_compile_enabled
        and plan.compile_scope == "step"
        and plan.compile_backend == "inductor"
        and not plan.compile_dynamic
        and plan.compile_mode == "default"
        and plan.cudagraphs == "mode_default"
        and plan.inductor_options_json == "{}",
        "amp_fp16": plan.amp_enabled
        and plan.autocast_dtype == "float16"
        and plan.autocast_cache_enabled
        and plan.fp32_loss
        and plan.grad_scaler_enabled,
        "ddp": plan.optimize_ddp == "python_reducer"
        and plan.compiled_autograd
        and plan.ddp_gradient_as_bucket_view
        and not plan.ddp_static_graph
        and not plan.ddp_broadcast_buffers
        and not plan.ddp_find_unused_parameters
        and plan.ddp_bucket_cap_mb == 50
        and plan.ddp_forward_sync_buffers is None
        and plan.communication_hook == "none"
        and plan.nccl_environment_json == "{}",
        "optimizer": plan.fused_optimizer
        and plan.gradient_clip_foreach
        and plan.zero_grad_set_to_none,
        "memory": plan.memory_format == "channels_last"
        and plan.tf32_enabled
        and plan.matmul_precision == "high",
        "loader": plan.dataloader_num_workers == 0
        and plan.dataloader_prefetch_factor is None
        and not plan.dataloader_pin_memory
        and not plan.dataloader_persistent_workers
        and plan.dataloader_non_blocking_h2d,
        "runtime_snapshot": plan.torch_version == "2.13.0+cu130"
        and plan.cuda_version == "13.0",
    }
    failures = sorted(name for name, passed in facts.items() if not passed)
    if failures:
        message = f"selected runtime is not the locked Spec 0015 bundle: {failures}"
        raise ValueError(message)


def _init_distributed() -> _Distributed:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
    if (
        world_size != REQUIRED_WORLD_SIZE
        or local_world_size != REQUIRED_WORLD_SIZE
        or torch.cuda.device_count() != REQUIRED_WORLD_SIZE
        or rank not in range(REQUIRED_WORLD_SIZE)
        or local_rank != rank
    ):
        message = "Spec 0015 readiness requires exactly two rank-matched CUDA devices"
        raise RuntimeError(message)
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://", device_id=device)
    return _Distributed(
        device=device,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
    )


def _device_assignments(distributed: _Distributed) -> list[JsonObject]:
    local: JsonObject = {
        "rank": distributed.rank,
        "local_rank": distributed.local_rank,
        "current_device": torch.cuda.current_device(),
        "device_name": torch.cuda.get_device_name(distributed.device),
    }
    gathered: list[object] = [None] * distributed.world_size
    cast("Callable[..., None]", dist.all_gather_object)(gathered, local)
    assignments = [cast("JsonObject", value) for value in gathered]
    expected = {(0, 0, 0), (1, 1, 1)}
    observed = {
        (item["rank"], item["local_rank"], item["current_device"])
        for item in assignments
    }
    if observed != expected or any(
        item["device_name"] != REQUIRED_GPU_NAME for item in assignments
    ):
        message = f"invalid dual-T4 rank assignment: {assignments}"
        raise RuntimeError(message)
    return assignments


def _buffer_sync_proof(model: SO2VAE) -> JsonObject:
    local_maximum = 0.0
    worst_name = ""
    for name, buffer in model.named_buffers():
        rank_zero = buffer.detach().clone()
        cast("Callable[..., object]", dist.broadcast)(rank_zero, src=0)
        difference = float((buffer.detach() - rank_zero).abs().max())
        if difference > local_maximum:
            local_maximum = difference
            worst_name = name
    gathered: list[object] = [None] * REQUIRED_WORLD_SIZE
    cast("Callable[..., None]", dist.all_gather_object)(
        gathered,
        (local_maximum, worst_name),
    )
    maximum, worst = max(cast("list[tuple[float, str]]", gathered))
    if maximum != 0.0:
        message = f"pre-compile buffers diverged at {worst}: {maximum}"
        raise RuntimeError(message)
    return {
        "status": "pass",
        "checked_before_ddp_and_compile": True,
        "buffer_count": len(list(model.buffers())),
        "max_abs_difference": maximum,
        "worst_buffer": worst,
    }


def _apply_runtime(plan: SelectedRuntimePlan) -> JsonObject:
    apply_cudnn_flags(
        benchmark=True,
        deterministic=False,
    )
    torch.backends.cuda.matmul.allow_tf32 = plan.tf32_enabled
    torch.backends.cudnn.allow_tf32 = plan.tf32_enabled
    torch.use_deterministic_algorithms(mode=False)
    torch.set_float32_matmul_precision(plan.matmul_precision)
    return {
        "runtime_policy_id": plan.runtime_policy_id,
        "source_selected_row_id": plan.selected_row_id,
        "readiness_per_device_batch_size": PER_DEVICE_BATCH,
        "memory_format": plan.memory_format,
        "autocast_dtype": plan.autocast_dtype,
        "fp32_loss": plan.fp32_loss,
        "grad_scaler_enabled": plan.grad_scaler_enabled,
        "compile_scope": plan.compile_scope,
        "compile_backend": plan.compile_backend,
        "compile_dynamic": plan.compile_dynamic,
        "optimize_ddp": plan.optimize_ddp,
        "compiled_autograd": plan.compiled_autograd,
        "reorder_compute_comm_overlap": plan.reorder_compute_comm_overlap,
        "fused_optimizer": plan.fused_optimizer,
        "gradient_clip_foreach": plan.gradient_clip_foreach,
        "zero_grad_set_to_none": plan.zero_grad_set_to_none,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "matmul_precision": torch.get_float32_matmul_precision(),
    }


def _optimizer_config(plan: SelectedRuntimePlan) -> SpecAdamWConfig:
    learning_rate = scaled_learning_rate(
        reference_lr=5.0e-4,
        scaling=BatchLrScaling(reference_global_batch_size=24, rule="sqrt"),
        global_batch_size=PER_DEVICE_BATCH * REQUIRED_WORLD_SIZE,
    )
    return SpecAdamWConfig(learning_rate=learning_rate, fused=plan.fused_optimizer)


def optimizer_policy_proof(model: SO2VAE, config: SpecAdamWConfig) -> JsonObject:
    """Return the exact SO2 semantic AdamW coverage proof."""
    groups, summary = build_adamw_parameter_groups(model, config=config)
    by_name = {group["name"]: group for group in groups}
    named = dict(model.named_parameters())
    coefficient_ids = {
        id(parameter)
        for name, parameter in named.items()
        if name.rsplit(".", 1)[-1].startswith("coeff")
    }
    gate_ids = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, FixedF01RadialGate)
        for parameter in module.parameters(recurse=False)
    }
    decay_ids = {id(parameter) for parameter in by_name["decay"]["params"]}
    gate_group_ids = {id(parameter) for parameter in by_name["gate_no_decay"]["params"]}
    passed = (
        summary.all_trainable_parameters_covered_once
        and coefficient_ids.issubset(decay_ids)
        and gate_ids == gate_group_ids
        and by_name["decay"]["weight_decay"] == config.weight_decay
        and by_name["gate_no_decay"]["weight_decay"] == 0.0
        and by_name["gate_no_decay"]["lr"]
        == config.learning_rate * config.gate_lr_multiplier
    )
    if not passed:
        raise RuntimeError("SO2 optimizer parameter policy drift")
    return {
        "status": "pass",
        "all_parameters_covered_once": summary.all_trainable_parameters_covered_once,
        "coefficient_parameter_count": sum(
            named_parameter.numel()
            for name, named_parameter in named.items()
            if name.rsplit(".", 1)[-1].startswith("coeff")
        ),
        "gate_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if id(parameter) in gate_ids
        ),
        "coefficient_weight_decay": config.weight_decay,
        "gate_weight_decay": 0.0,
        "base_learning_rate": config.learning_rate,
        "gate_learning_rate": config.learning_rate * config.gate_lr_multiplier,
        "fused_requested": config.fused,
    }


def _master_dtype_proof(model: SO2VAE) -> JsonObject:
    parameter_dtypes = sorted({
        str(parameter.dtype) for parameter in model.parameters()
    })
    buffer_dtypes = sorted({str(buffer.dtype) for buffer in model.buffers()})
    norm_count = sum(
        isinstance(module, FixedF01FieldNorm) for module in model.modules()
    )
    gate_count = sum(
        isinstance(module, FixedF01RadialGate) for module in model.modules()
    )
    passed = (
        parameter_dtypes == ["torch.float32"]
        and buffer_dtypes == ["torch.float32"]
        and norm_count == FIELD_NORM_COUNT
        and gate_count == RADIAL_GATE_COUNT
    )
    if not passed:
        raise RuntimeError("SO2 FP32 master-state or norm/gate topology drift")
    return {
        "status": "pass",
        "parameter_dtypes": parameter_dtypes,
        "buffer_dtypes": buffer_dtypes,
        "field_norm_count": norm_count,
        "radial_gate_count": gate_count,
        "norm_and_radial_math_dtype": "float32",
    }


def _wrap_model(
    model: SO2VAE,
    *,
    plan: SelectedRuntimePlan,
    distributed: _Distributed,
) -> DistributedDataParallel:
    if model_requires_buffer_broadcast(model):
        raise RuntimeError("fixed SO2 model unexpectedly requires buffer broadcast")
    return wrap_fastpath_ddp(
        model,
        local_rank=distributed.local_rank,
        static_graph=plan.ddp_static_graph,
        gradient_as_bucket_view=plan.ddp_gradient_as_bucket_view,
        broadcast_buffers=plan.ddp_broadcast_buffers,
        find_unused_parameters=plan.ddp_find_unused_parameters,
        bucket_cap_mb=plan.ddp_bucket_cap_mb,
        dynamo=FastpathDynamoKnobs(
            optimize_ddp=plan.optimize_ddp,
            compiled_autograd=plan.compiled_autograd,
            reorder_compute_comm_overlap=plan.reorder_compute_comm_overlap,
        ),
        forward_sync_buffers=plan.ddp_forward_sync_buffers,
    )


def _ddp_runtime_readback(
    ddp: DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    plan: SelectedRuntimePlan,
) -> JsonObject:
    readback = {
        "python_reducer": getattr(ddp, "_use_python_reducer", None),
        "static_graph": getattr(ddp, "static_graph", None),
        "gradient_as_bucket_view": getattr(ddp, "gradient_as_bucket_view", None),
        "broadcast_buffers": getattr(ddp, "broadcast_buffers", None),
        "find_unused_parameters": getattr(ddp, "find_unused_parameters", None),
        "bucket_bytes_cap": getattr(ddp, "bucket_bytes_cap", None),
        "optimize_ddp": torch_dynamo.config.optimize_ddp,
        "compiled_autograd": torch_dynamo.config.compiled_autograd,
        "reorder_compute_comm_overlap": (
            inductor_config.reorder_for_compute_comm_overlap
        ),
        "optimizer_fused": optimizer.defaults.get("fused"),
    }
    expected = {
        "python_reducer": True,
        "static_graph": plan.ddp_static_graph,
        "gradient_as_bucket_view": plan.ddp_gradient_as_bucket_view,
        "broadcast_buffers": plan.ddp_broadcast_buffers,
        "find_unused_parameters": plan.ddp_find_unused_parameters,
        "bucket_bytes_cap": cast("int", plan.ddp_bucket_cap_mb) * 1024 * 1024,
        "optimize_ddp": plan.optimize_ddp,
        "compiled_autograd": plan.compiled_autograd,
        "reorder_compute_comm_overlap": plan.reorder_compute_comm_overlap,
        "optimizer_fused": plan.fused_optimizer,
    }
    failures = sorted(key for key, value in expected.items() if readback[key] != value)
    if failures:
        message = f"selected DDP/compiler/optimizer readback drift: {failures}"
        raise RuntimeError(message)
    return {"status": "pass", "requested": expected, "effective": readback}


def _generated_inputs(distributed: _Distributed) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=distributed.device)
    generator.manual_seed(15015 + distributed.rank)
    inputs = torch.randint(
        0,
        256,
        (PER_DEVICE_BATCH, 3, IMAGE_SIZE, IMAGE_SIZE),
        dtype=torch.uint8,
        device=distributed.device,
        generator=generator,
    ).contiguous(memory_format=torch.channels_last)
    eps = torch.randn(
        (PER_DEVICE_BATCH, 16, LATENT_SIZE, LATENT_SIZE),
        dtype=torch.float32,
        device=distributed.device,
        generator=generator,
    )
    return inputs, eps


def _gradient_mean_proof(
    ddp: DistributedDataParallel,
    step_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], FastpathStepOutput],
    inputs: torch.Tensor,
    eps: torch.Tensor,
    distributed: _Distributed,
) -> JsonObject:
    parameter = cast("SO2VAE", ddp.module).output_head.bias
    beta = torch.zeros((), device=distributed.device, dtype=torch.float32)
    rng_state = torch.cuda.get_rng_state(distributed.device)
    ddp.zero_grad(set_to_none=True)
    with ddp.no_sync():
        local_output = step_fn(inputs, eps, beta)
        local_output.loss.backward()
    local_gradient = cast("torch.Tensor", parameter.grad).detach().clone()
    gathered = [torch.empty_like(local_gradient) for _ in range(distributed.world_size)]
    cast("Callable[..., object]", dist.all_gather)(gathered, local_gradient)
    expected = torch.stack(gathered).mean(dim=0)
    local_gradients_differ = not torch.equal(gathered[0], gathered[1])

    ddp.zero_grad(set_to_none=True)
    torch.cuda.set_rng_state(rng_state, distributed.device)
    reduced_output = step_fn(inputs, eps, beta)
    reduced_output.loss.backward()
    reduced = cast("torch.Tensor", parameter.grad).detach()
    maximum_error = float((reduced - expected).abs().max())
    passed = local_gradients_differ and maximum_error <= DDP_REFERENCE_LIMIT
    ddp.zero_grad(set_to_none=True)
    if not passed:
        raise RuntimeError("full-model DDP gradient-mean proof failed")
    return {
        "status": "pass",
        "parameter": "output_head.bias",
        "local_pre_reduction_gradients_differ": local_gradients_differ,
        "reduced_gradient_max_abs_error": maximum_error,
    }


def _graph_break_total() -> int:
    return int(sum(cast("dict[str, int]", counters["graph_break"]).values()))


def _unique_graph_total() -> int:
    return int(cast("dict[str, int]", counters["stats"]).get("unique_graphs", 0))


def _snapshot_named(model: SO2VAE) -> dict[str, torch.Tensor]:
    named = dict(model.named_parameters())
    return {
        label: named[name].detach().clone()
        for label, name in _NAMED_UPDATE_PARAMETERS.items()
    }


def _named_update_proof(
    model: SO2VAE,
    before: dict[str, torch.Tensor],
    *,
    labels: tuple[str, ...],
) -> JsonObject:
    named = dict(model.named_parameters())
    proof: JsonObject = {}
    failures: list[str] = []
    for label in labels:
        parameter = named[_NAMED_UPDATE_PARAMETERS[label]]
        gradient = parameter.grad
        update_norm = float((parameter.detach() - before[label]).float().norm())
        gradient_norm = (
            0.0 if gradient is None else float(gradient.detach().float().norm())
        )
        passed = (
            update_norm > 0.0
            and gradient_norm > 0.0
            and math.isfinite(update_norm + gradient_norm)
        )
        proof[label] = {
            "parameter": _NAMED_UPDATE_PARAMETERS[label],
            "update_norm": update_norm,
            "gradient_norm": gradient_norm,
            "status": "pass" if passed else "fail",
        }
        if not passed:
            failures.append(label)
    if failures:
        message = f"named gradient-driven updates missing: {failures}"
        raise RuntimeError(message)
    return proof


def _parameter_sync_proof(model: SO2VAE, distributed: _Distributed) -> JsonObject:
    local_maximum = 0.0
    worst_name = ""
    for name, parameter in model.named_parameters():
        rank_zero = parameter.detach().clone()
        cast("Callable[..., object]", dist.broadcast)(rank_zero, src=0)
        difference = float((parameter.detach() - rank_zero).abs().max())
        if difference > local_maximum:
            local_maximum = difference
            worst_name = name
    gathered: list[object] = [None] * distributed.world_size
    cast("Callable[..., None]", dist.all_gather_object)(
        gathered,
        (local_maximum, worst_name),
    )
    maximum, worst = max(cast("list[tuple[float, str]]", gathered))
    if maximum != 0.0:
        raise RuntimeError(f"DDP parameters diverged at {worst}: {maximum}")
    return {
        "status": "pass",
        "max_abs_difference": maximum,
        "worst_parameter": worst,
    }


def _float(value: torch.Tensor) -> float:
    return float(value.detach().float().item())


def _stats(values: torch.Tensor) -> tuple[float, float, float, float]:
    flat = values.detach().float().flatten()
    return (
        _float(flat.min()),
        _float(flat.max()),
        _float(flat.mean()),
        _float(flat.std(unbiased=False)),
    )


def _gate_capture_rows(
    model: SO2VAE,
    *,
    inputs: torch.Tensor,
    eps: torch.Tensor,
    initial: dict[str, torch.Tensor],
    config: _ProbeConfig,
    plan: SelectedRuntimePlan,
) -> tuple[CsvRow, ...]:
    rows_by_module: dict[str, list[CsvRow]] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def hook_for(name: str, gate_module: FixedF01RadialGate) -> Callable[..., None]:
        def hook(
            _module: nn.Module,
            arguments: tuple[torch.Tensor, ...],
            raw_output: torch.Tensor,
        ) -> None:
            gate_input = arguments[0]
            values = gate_input.float()
            output = raw_output.float()
            layout = gate_module.layout
            scalar_input = values[:, : layout.n0]
            scalar_output = output[:, : layout.n0]
            vector_input = values[:, layout.f1_offset :].view(
                values.shape[0],
                layout.n1,
                2,
                values.shape[2],
                values.shape[3],
            )
            vector_output = output[:, layout.f1_offset :].view_as(vector_input)
            radius = torch.sqrt(vector_input.square().sum(dim=2) + _RADIUS_EPS)
            scalar_gate = torch.sigmoid(
                gate_module.f0_a.view(1, -1, 1, 1) * scalar_input
                + gate_module.f0_b.view(1, -1, 1, 1),
            ).to(dtype=gate_input.dtype)
            vector_gate = torch.sigmoid(
                gate_module.f1_a.view(1, -1, 1, 1) * radius
                + gate_module.f1_b.view(1, -1, 1, 1),
            ).to(dtype=gate_input.dtype)
            families = (
                (
                    "f0_scalar",
                    gate_module.f0_a,
                    gate_module.f0_b,
                    scalar_input,
                    scalar_output,
                    scalar_gate,
                ),
                (
                    "f1_radial",
                    gate_module.f1_a,
                    gate_module.f1_b,
                    vector_input,
                    vector_output,
                    vector_gate,
                ),
            )
            module_rows: list[CsvRow] = []
            for family, a, b, family_input, family_output, gate in families:
                a_min, a_max, a_mean, a_std = _stats(a)
                b_min, b_max, b_mean, b_std = _stats(b)
                gate_values = gate.detach().float()
                flattened = gate_values.transpose(0, 1).flatten(1)
                low_by_channel = (flattened < GATE_LOW_THRESHOLD).float().mean(dim=1)
                high_by_channel = (flattened > GATE_HIGH_THRESHOLD).float().mean(dim=1)
                input_rms = _float(family_input.detach().float().square().mean().sqrt())
                output_rms = _float(
                    family_output.detach().float().square().mean().sqrt(),
                )
                a_grad = a.grad
                b_grad = b.grad
                initial_a = initial[f"{name}.{family}.a"]
                initial_b = initial[f"{name}.{family}.b"]
                a_update_ratio = _float(
                    (a.detach() - initial_a).float().norm(),
                ) / max(_float(initial_a.float().norm()), 1e-12)
                b_update_ratio = _float(
                    (b.detach() - initial_b).float().norm(),
                ) / max(_float(initial_b.float().norm()), 1e-12)
                a_grad_norm = (
                    0.0 if a_grad is None else _float(a_grad.detach().float().norm())
                )
                b_grad_norm = (
                    0.0 if b_grad is None else _float(b_grad.detach().float().norm())
                )
                finite_tensors = [a, b, gate_values, family_input, family_output]
                healthy = (
                    a_grad is not None
                    and b_grad is not None
                    and all(
                        bool(torch.isfinite(value).all()) for value in finite_tensors
                    )
                    and math.isfinite(
                        a_grad_norm + b_grad_norm + a_update_ratio + b_update_ratio,
                    )
                    and a_grad_norm > 0.0
                    and b_grad_norm > 0.0
                    and a_update_ratio > 0.0
                    and b_update_ratio > 0.0
                )
                module_rows.append({
                    "run_name": config.run_name,
                    "benchmark_kind": PROBE_KIND,
                    "benchmark_source": "generated_device_resident_rank0",
                    "full_run_eligible": "false",
                    "accelerator_mode": plan.accelerator_mode,
                    "machine_shape": plan.machine_shape,
                    "row_id": "spec0015_so2_bs1_readiness",
                    "candidate_row_id": plan.selected_row_id,
                    "runtime_policy_id": plan.runtime_policy_id,
                    "optimizer_step": str(
                        config.warmup_updates + config.settled_updates,
                    ),
                    "module": f"{name}:{family}",
                    "gate_kind": family,
                    "num_channels": str(a.numel()),
                    "num_elements": str(gate_values.numel()),
                    "a_min": f"{a_min:.9g}",
                    "a_max": f"{a_max:.9g}",
                    "a_mean": f"{a_mean:.9g}",
                    "a_std": f"{a_std:.9g}",
                    "b_min": f"{b_min:.9g}",
                    "b_max": f"{b_max:.9g}",
                    "b_mean": f"{b_mean:.9g}",
                    "b_std": f"{b_std:.9g}",
                    "max_abs_a": f"{_float(a.detach().abs().max()):.9g}",
                    "max_abs_b": f"{_float(b.detach().abs().max()):.9g}",
                    "gate_mean": f"{_float(gate_values.mean()):.9g}",
                    "gate_std": f"{_float(gate_values.std(unbiased=False)):.9g}",
                    "gate_p01": f"{_float(torch.quantile(gate_values, 0.01)):.9g}",
                    "gate_p50": f"{_float(torch.quantile(gate_values, 0.50)):.9g}",
                    "gate_p99": f"{_float(torch.quantile(gate_values, 0.99)):.9g}",
                    "frac_gate_lt_0_01": f"{_float((gate_values < 0.01).float().mean()):.9g}",
                    "frac_gate_gt_0_99": f"{_float((gate_values > 0.99).float().mean()):.9g}",
                    "worst_channel_frac_gate_lt_0_01": f"{_float(low_by_channel.max()):.9g}",
                    "worst_channel_frac_gate_gt_0_99": f"{_float(high_by_channel.max()):.9g}",
                    "dead_channel_count": str(
                        int(torch.count_nonzero(low_by_channel == 1.0)),
                    ),
                    "input_rms": f"{input_rms:.9g}",
                    "output_rms": f"{output_rms:.9g}",
                    "output_input_rms_ratio": f"{output_rms / max(input_rms, 1e-12):.9g}",
                    "a_grad_norm": f"{a_grad_norm:.9g}",
                    "b_grad_norm": f"{b_grad_norm:.9g}",
                    "a_update_to_param_norm": f"{a_update_ratio:.9g}",
                    "b_update_to_param_norm": f"{b_update_ratio:.9g}",
                    "gate_force_fp32": "true",
                    "input_dtype": str(gate_input.dtype).removeprefix("torch."),
                    "gate_math_dtype": "float32",
                    "gate_tensor_dtype": str(gate.dtype).removeprefix("torch."),
                    "output_dtype": str(raw_output.dtype).removeprefix("torch."),
                    "requested_autocast_dtype": plan.autocast_dtype,
                    "precision_proof_status": "pass",
                    "gate_health_status": "pass" if healthy else "fail",
                })
            rows_by_module[name] = module_rows

        return hook

    for name, module in model.named_modules():
        if isinstance(module, FixedF01RadialGate):
            handles.append(module.register_forward_hook(hook_for(name, module)))
    clean = normalize_uint8_batch(inputs)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        model(clean, eps=eps)
    for handle in handles:
        handle.remove()
    rows = tuple(row for name in sorted(rows_by_module) for row in rows_by_module[name])
    if len(rows) != GATE_ROW_COUNT or {row["gate_kind"] for row in rows} != {
        "f0_scalar",
        "f1_radial",
    }:
        raise RuntimeError(
            f"expected exactly {GATE_ROW_COUNT} F0/F1 rows, got {len(rows)}",
        )
    return rows


def _initial_gate_snapshots(model: SO2VAE) -> dict[str, torch.Tensor]:
    snapshots: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if not isinstance(module, FixedF01RadialGate):
            continue
        for family, a, b in (
            ("f0_scalar", module.f0_a, module.f0_b),
            ("f1_radial", module.f1_a, module.f1_b),
        ):
            snapshots[f"{name}.{family}.a"] = a.detach().clone()
            snapshots[f"{name}.{family}.b"] = b.detach().clone()
    return snapshots


def _verdict(
    result: JsonObject,
    gate_rows: tuple[CsvRow, ...],
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    compiled = _required_object(result, "compiled_execution")
    checks = {
        "graph_breaks": compiled.get("post_settle_graph_break_count") == 0,
        "recompiles": compiled.get("post_settle_recompile_count") == 0,
        "amp_skips": compiled.get("amp_step_skipped_count") == 0,
        "finite_losses": compiled.get("finite_losses") is True,
        "finite_parameters": compiled.get("finite_parameters") is True,
        "gate_rows": len(gate_rows) == GATE_ROW_COUNT,
        "finite_gate_rows": all(
            row["gate_health_status"] == "pass" for row in gate_rows
        ),
    }
    failures.extend(name for name, passed in checks.items() if not passed)
    return not failures, failures


def run(config_path: Path, output_dir: Path) -> JsonObject:  # noqa: C901, PLR0915
    """Execute the one fixed readiness coordinate and write rank-zero artifacts.

    Returns:
        Compact result payload on every rank.

    """
    config = parse_probe_config(config_path)
    if (
        config.warmup_updates != WARMUP_UPDATES
        or config.settled_updates != SETTLED_UPDATES
    ):
        raise ValueError("Spec 0015 warmup/settled update counts drifted")
    plan = parse_selected_runtime_plan(config.runtime_path)
    validate_selected_plan(plan)
    if (
        torch.__version__ != plan.torch_version
        or torch.version.cuda != plan.cuda_version
    ):
        message = (
            "installed Torch/CUDA differs from the selected runtime snapshot: "
            f"{torch.__version__}/{torch.version.cuda}"
        )
        raise RuntimeError(message)
    distributed = _init_distributed()
    assignments = _device_assignments(distributed)
    runtime = _apply_runtime(plan)

    torch.manual_seed(15015)
    torch.cuda.manual_seed(15015)
    model = assert_fixed_so2_model(build_model(MODEL_KIND_SO2_FIXED))
    model.to(device=distributed.device, memory_format=torch.channels_last)
    master_dtype = _master_dtype_proof(model)
    buffer_sync = _buffer_sync_proof(model)
    optimizer_config = _optimizer_config(plan)
    optimizer_policy = optimizer_policy_proof(model, optimizer_config)
    initial_gate_state = _initial_gate_snapshots(model)
    initial_named_state = _snapshot_named(model)
    ddp = _wrap_model(model, plan=plan, distributed=distributed)
    optimizer = build_fastpath_optimizer(model, config=optimizer_config)
    ddp_readback = _ddp_runtime_readback(ddp, optimizer, plan)

    shared = resolve_json_config(config.shared_contract_path).effective_config
    corruption = InlineStainCorruptor(
        profile_from_config(_required_object(shared, "corruption")),
    ).to(distributed.device)
    eager_step = make_fastpath_step_fn(
        ddp,
        corruption,
        ssim_weight=0.1,
        autocast_dtype=torch.float16,
        autocast_enabled=True,
        autocast_cache_enabled=plan.autocast_cache_enabled,
    )
    inputs, eps = _generated_inputs(distributed)
    gradient_mean = _gradient_mean_proof(ddp, eager_step, inputs, eps, distributed)

    torch_dynamo.reset()
    counters.clear()
    compile_mode, compile_options = resolve_fastpath_compile_invocation(
        compile_mode=plan.compile_mode,
        cudagraphs=plan.cudagraphs,
        inductor_options_json=plan.inductor_options_json,
    )
    compiled_step = cast(
        "Callable[[torch.Tensor, torch.Tensor, torch.Tensor], FastpathStepOutput]",
        torch.compile(
            eager_step,
            backend=plan.compile_backend,
            dynamic=plan.compile_dynamic,
            mode=compile_mode,
            options=compile_options,
        ),
    )
    scaler = GradScaler(
        "cuda",
        init_scale=EXPECTED_RUNNER_AMP_GRAD_SCALER_INIT_SCALE,
        enabled=True,
    )

    def update(beta_value: float) -> tuple[FastpathStepOutput, bool]:
        optimizer.zero_grad(set_to_none=plan.zero_grad_set_to_none)
        beta = torch.tensor(beta_value, device=distributed.device, dtype=torch.float32)
        output = compiled_step(inputs, eps, beta)
        step_result = run_fastpath_optimizer_step_with_metrics(
            loss=output.loss,
            optimizer=optimizer,
            parameters=model.parameters(),
            scaler=scaler,
            grad_scaler_enabled=True,
            gradient_clip_global_norm=optimizer_config.gradient_clip_global_norm,
            gradient_clip_foreach=plan.gradient_clip_foreach,
            backward_context=compiled_autograd_context(enabled=plan.compiled_autograd),
            observe_skip=True,
        )
        return output, step_result.step_skipped

    torch.cuda.synchronize(distributed.device)
    compile_started = time.perf_counter()
    first_output, first_skip = update(0.0)
    torch.cuda.synchronize(distributed.device)
    compile_startup_seconds = time.perf_counter() - compile_started
    if first_skip:
        raise RuntimeError("GradScaler skipped the first readiness update")
    first_proof = _named_update_proof(
        model,
        initial_named_state,
        labels=("output_head",),
    )
    first_upstream_gradient_norms = {
        label: 0.0
        if dict(model.named_parameters())[_NAMED_UPDATE_PARAMETERS[label]].grad is None
        else float(
            cast(
                "torch.Tensor",
                dict(model.named_parameters())[_NAMED_UPDATE_PARAMETERS[label]].grad,
            )
            .detach()
            .float()
            .norm(),
        )
        for label in ("decoder", "posterior", "encoder", "stem", "f0_gate", "f1_gate")
    }
    if any(value != 0.0 for value in first_upstream_gradient_norms.values()):
        message = "zero RGB head did not block first-step upstream gradients"
        raise RuntimeError(message)
    after_first = _snapshot_named(model)
    second_output, second_skip = update(0.01)
    if second_skip:
        raise RuntimeError("GradScaler skipped the second readiness update")
    subsequent_proof = _named_update_proof(
        model,
        after_first,
        labels=("decoder", "posterior", "encoder", "stem", "f0_gate", "f1_gate"),
    )
    warmup_skips = 0
    for _ in range(config.warmup_updates - 2):
        _output, did_skip = update(0.01)
        warmup_skips += int(did_skip)

    graph_break_before = _graph_break_total()
    unique_graph_before = _unique_graph_total()
    torch.cuda.reset_peak_memory_stats(distributed.device)
    timings_ms: list[float] = []
    settled_skips = 0
    last_output = second_output
    for _ in range(config.settled_updates):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        last_output, did_skip = update(0.01)
        end.record()
        end.synchronize()
        timings_ms.append(float(start.elapsed_time(end)))
        settled_skips += int(did_skip)
    torch.cuda.synchronize(distributed.device)
    graph_breaks = _graph_break_total() - graph_break_before
    recompiles = _unique_graph_total() - unique_graph_before
    allocated_mib = torch.cuda.max_memory_allocated(distributed.device) / BYTES_PER_MIB
    reserved_mib = torch.cuda.max_memory_reserved(distributed.device) / BYTES_PER_MIB
    total_mib = (
        torch.cuda.get_device_properties(distributed.device).total_memory
        / BYTES_PER_MIB
    )
    parameter_sync = _parameter_sync_proof(model, distributed)
    finite_parameters = all(
        bool(torch.isfinite(parameter).all()) for parameter in model.parameters()
    )
    finite_losses = bool(
        torch.isfinite(first_output.loss) and torch.isfinite(last_output.loss),
    )

    gate_rows: tuple[CsvRow, ...] = ()
    if distributed.rank == PRIMARY_RANK:
        gate_rows = _gate_capture_rows(
            model,
            inputs=inputs,
            eps=eps,
            initial=initial_gate_state,
            config=config,
            plan=plan,
        )
    rank_metrics: JsonObject = {
        "rank": distributed.rank,
        "compile_startup_seconds": compile_startup_seconds,
        "settled_step_ms": timings_ms,
        "peak_allocated_mib": allocated_mib,
        "peak_reserved_mib": reserved_mib,
        "total_device_memory_mib": total_mib,
        "reserved_headroom_fraction": (total_mib - reserved_mib) / total_mib,
        "amp_step_skipped_count": warmup_skips + settled_skips,
        "post_settle_graph_break_count": graph_breaks,
        "post_settle_recompile_count": recompiles,
        "finite_losses": finite_losses,
        "finite_parameters": finite_parameters,
    }
    gathered_metrics: list[object] = [None] * distributed.world_size
    cast("Callable[..., None]", dist.all_gather_object)(gathered_metrics, rank_metrics)
    rank_results = [cast("JsonObject", value) for value in gathered_metrics]
    max_allocated = max(
        float(cast("float", value["peak_allocated_mib"])) for value in rank_results
    )
    max_reserved = max(
        float(cast("float", value["peak_reserved_mib"])) for value in rank_results
    )
    min_headroom = min(
        float(cast("float", value["reserved_headroom_fraction"]))
        for value in rank_results
    )
    all_timings = [
        float(item)
        for value in rank_results
        for item in cast("list[object]", value["settled_step_ms"])
    ]
    result: JsonObject = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_kind": PROBE_KIND,
        "status": "pending_verdict",
        "full_run_eligible": False,
        "full_training_authorized": False,
        "model_kind": MODEL_KIND_SO2_FIXED,
        "model_identity": {
            "concrete_class": type(model).__name__,
            "learned_parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "latent_channels": model.latent_channels,
            "learned_convolution_count": 43,
            "radial_gate_count": RADIAL_GATE_COUNT,
        },
        "world_size": distributed.world_size,
        "per_device_batch_size": PER_DEVICE_BATCH,
        "input_shape": [PER_DEVICE_BATCH, 3, IMAGE_SIZE, IMAGE_SIZE],
        "data_source": "generated_device_resident",
        "dataset_sources": [],
        "rank_device_assignments": assignments,
        **torch_runtime_versions(),
        "runtime_requested_and_effective": runtime,
        "master_dtype_proof": master_dtype,
        "pre_compile_buffer_sync": buffer_sync,
        "ddp_runtime_readback": ddp_readback,
        "optimizer_policy": optimizer_policy,
        "gradient_mean_reference": gradient_mean,
        "update_sequence": {
            "first_zero_head_update": first_proof,
            "first_upstream_gradient_norms": first_upstream_gradient_norms,
            "subsequent_named_updates": subsequent_proof,
        },
        "compiled_execution": {
            "warmup_updates": config.warmup_updates,
            "settled_updates": config.settled_updates,
            "compile_startup_seconds_rank_max": max(
                float(cast("float", value["compile_startup_seconds"]))
                for value in rank_results
            ),
            "diagnostic_settled_step_ms_p50": statistics.median(all_timings),
            "diagnostic_settled_step_ms_rank_samples": [
                value["settled_step_ms"] for value in rank_results
            ],
            "amp_step_skipped_count": sum(
                int(cast("int", value["amp_step_skipped_count"]))
                for value in rank_results
            ),
            "post_settle_graph_break_count": max(
                int(cast("int", value["post_settle_graph_break_count"]))
                for value in rank_results
            ),
            "post_settle_recompile_count": max(
                int(cast("int", value["post_settle_recompile_count"]))
                for value in rank_results
            ),
            "finite_losses": all(
                bool(value["finite_losses"]) for value in rank_results
            ),
            "finite_parameters": all(
                bool(value["finite_parameters"]) for value in rank_results
            ),
            "peak_allocated_mib_rank_max": max_allocated,
            "peak_reserved_mib_rank_max": max_reserved,
            "reserved_vram_headroom_fraction_rank_min": min_headroom,
            "diagnostic_only_no_training_projection": True,
        },
        "parameter_sync": parameter_sync,
        "gate_health": {
            "expected_rows": GATE_ROW_COUNT,
            "rows_written": len(gate_rows)
            if distributed.rank == PRIMARY_RANK
            else GATE_ROW_COUNT,
            "families": ["f0_scalar", "f1_radial"],
        },
        "rank_metrics": rank_results,
    }
    passed, failures = _verdict(
        result,
        gate_rows
        if distributed.rank == PRIMARY_RANK
        else cast(
            "tuple[CsvRow, ...]",
            ({"gate_health_status": "pass"},) * GATE_ROW_COUNT,
        ),
    )
    result["status"] = "pass" if passed else "fail"
    result["acceptance_failures"] = failures
    if distributed.rank == PRIMARY_RANK:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(output_dir / GATE_FILENAME, GATE_HEALTH_COLUMNS, gate_rows)
        write_json(output_dir / ARTIFACT_FILENAME, result)
    cast("Callable[[], object]", dist.barrier)()
    if not passed:
        raise RuntimeError(f"Spec 0015 readiness failed: {failures}")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run(args.config, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARTIFACT_FILENAME",
    "GATE_FILENAME",
    "GATE_ROW_COUNT",
    "optimizer_policy_proof",
    "parse_probe_config",
    "run",
    "validate_selected_plan",
]
