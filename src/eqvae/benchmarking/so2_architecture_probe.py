# Copyright 2026 HiperMaximus
"""One-use dual-T4 mechanics probe for the locked Spec 0013 modules.

This is deliberately a singular benchmark, not a runtime tuner. It executes the
selected Spec 0011 bundle once on generated fixed-shape tensors and writes one
compact result. The padded-bmm/direct mechanics are singular and fixed.
"""
# pyright: reportPrivateUsage=false, reportUnusedFunction=false

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
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
from torch.amp.grad_scaler import GradScaler

from eqvae.benchmarking.io import write_json
from eqvae.benchmarking.torch_runtime import torch_runtime_versions
from eqvae.models.non_equivariant_vae import DecoderUpResBlock, EncoderResBlock
from eqvae.models.so2_architecture_probe import (
    A_LAYOUT,
    B_LAYOUT,
    D_LAYOUT,
    SO2DecoderTransitionBA,
    SO2EncoderTransitionAB,
    SO2IdentityResidualBlockA,
    SO2LargestDDConv,
)
from eqvae.training.fastpath_recipe import (
    FastpathDynamoKnobs,
    apply_fastpath_dynamo_config,
    compiled_autograd_context,
    wrap_fastpath_ddp,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from eqvae.benchmarking.io import JsonObject

SCHEMA_VERSION: Final = "spec0013.so2_dual_t4_final.v1"
ARTIFACT_FILENAME: Final = "spec0013_so2_dual_t4_probe.json"
PROBE_KIND: Final = "locked_so2_architecture_mechanics_final"
RUNTIME_BUNDLE_ID: Final = "compile_step_python_reducer_fp16_channels_last"
PER_DEVICE_BATCH: Final = 4
SETTLED_UPDATES: Final = 32
WARMUP_UPDATES: Final = 20
TIMED_WINDOW_UPDATES: Final = 50
TIMED_WINDOW_COUNT: Final = 2
IMAGE_SIZE: Final = 256
LATENT_SIZE: Final = 32
GRAD_SCALER_INIT_SCALE: Final = 16384.0
MAX_GRAD_NORM: Final = 1.0
BUCKET_CAP_MB: Final = 50
MILLISECONDS_PER_SECOND: Final = 1000.0
BYTES_PER_MIB: Final = 1024.0**2
OUTPUT_RELATIVE_LIMIT: Final = 5e-3
GRADIENT_RELATIVE_LIMIT: Final = 2e-2
COMPILED_EAGER_RATIO_LIMIT: Final = 1.10
NORMAL_RATIO_LIMIT: Final = 5.0
PEAK_RESERVED_MIB_LIMIT: Final = 14.5 * 1024.0
PEAK_ALLOCATED_MIB_LIMIT: Final = 13.5 * 1024.0
DDP_REFERENCE_LIMIT: Final = 1e-6
REQUIRED_WORLD_SIZE: Final = 2
REQUIRED_GPU_NAME: Final = "Tesla T4"
_PRIMARY_RANK: Final = 0
_SELECTED_RUNTIME_PATH: Final = Path(
    "configs/spec0001/non_eq_vae_selected_runtime.json",
)
_SELECTED_RUNTIME_SHA256: Final = (
    "e9e998fd161f0955959c64aed7cd7ddbdfcb55a271b9ce05805903c97c93efb8"
)


@dataclass(frozen=True)
class _Distributed:
    device: torch.device
    rank: int
    local_rank: int
    world_size: int
    nproc_per_node: int


@dataclass(frozen=True)
class _BlockCase:
    name: str
    equivariant: nn.Module
    normal: nn.Module
    shape: tuple[int, int, int, int]


@dataclass(frozen=True)
class _PreparedStep:
    name: str
    step: Callable[[], tuple[torch.Tensor, torch.Tensor]]
    scaler: GradScaler


class _MechanicsSuite(nn.Module):
    """Four independent fixed probe paths joined only for one DDP update."""

    def __init__(self) -> None:
        """Build exactly the four Spec 0013 benchmark signatures."""
        super().__init__()
        self.identity = SO2IdentityResidualBlockA()
        self.encoder = SO2EncoderTransitionAB()
        self.decoder = SO2DecoderTransitionBA()
        self.largest = SO2LargestDDConv()

    def forward(
        self,
        identity_input: torch.Tensor,
        encoder_input: torch.Tensor,
        decoder_input: torch.Tensor,
        largest_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the four independent paths in one DDP-visible forward.

        Returns:
            Outputs in identity, encoder, decoder, largest order.

        """
        return (
            cast("torch.Tensor", self.identity(identity_input)),
            cast("torch.Tensor", self.encoder(encoder_input)),
            cast("torch.Tensor", self.decoder(decoder_input)),
            cast("torch.Tensor", self.largest(largest_input)),
        )


def _init_distributed() -> _Distributed:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    nproc_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
    launch_values = (world_size, nproc_per_node, torch.cuda.device_count())
    ranks_valid = rank in range(REQUIRED_WORLD_SIZE) and local_rank in range(
        REQUIRED_WORLD_SIZE,
    )
    if (
        launch_values != (REQUIRED_WORLD_SIZE,) * 3
        or not ranks_valid
        or rank != local_rank
    ):
        message = "Spec 0013 probe requires exactly two visible GPUs and world_size=2"
        raise RuntimeError(message)
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://", device_id=device)
    return _Distributed(device, rank, local_rank, world_size, nproc_per_node)


def _device_assignments(distributed: _Distributed) -> list[dict[str, object]]:
    local: dict[str, object] = {
        "rank": distributed.rank,
        "local_rank": distributed.local_rank,
        "current_device": torch.cuda.current_device(),
        "device_name": torch.cuda.get_device_name(distributed.device),
    }
    gathered: list[object] = [None] * distributed.world_size
    cast("Callable[..., None]", dist.all_gather_object)(gathered, local)
    assignments = [cast("dict[str, object]", item) for item in gathered]
    if {
        (item["rank"], item["local_rank"], item["current_device"])
        for item in assignments
    } != {(0, 0, 0), (1, 1, 1)}:
        message = f"rank-to-device assignment is not bijective: {assignments}"
        raise RuntimeError(message)
    if [item["device_name"] for item in assignments] != [REQUIRED_GPU_NAME] * 2:
        message = f"Spec 0013 requires two {REQUIRED_GPU_NAME} GPUs: {assignments}"
        raise RuntimeError(message)
    return assignments


def _apply_selected_runtime() -> dict[str, object]:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.use_deterministic_algorithms(mode=False)
    torch.set_float32_matmul_precision("high")
    apply_fastpath_dynamo_config(
        optimize_ddp="python_reducer",
        compiled_autograd=True,
        reorder_compute_comm_overlap=True,
    )
    requested: dict[str, object] = {
        "runtime_policy_id": RUNTIME_BUNDLE_ID,
        "memory_format": "channels_last",
        "autocast_dtype": "float16",
        "autocast_cache_enabled": True,
        "fp32_loss": True,
        "grad_scaler_enabled": True,
        "compile_scope": "step",
        "compile_backend": "inductor",
        "compile_dynamic": False,
        "compile_mode": "default",
        "cudagraphs": "mode_default",
        "inductor_options_json": "{}",
        "optimize_ddp": "python_reducer",
        "compiled_autograd": True,
        "reorder_compute_comm_overlap": True,
        "fused_adamw": True,
        "gradient_clip_foreach": True,
        "zero_grad_set_to_none": True,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "ddp_static_graph": False,
        "ddp_gradient_as_bucket_view": True,
        "ddp_broadcast_buffers": False,
        "ddp_find_unused_parameters": False,
        "ddp_bucket_cap_mb": BUCKET_CAP_MB,
        "communication_hook": "none",
    }
    selected = cast(
        "dict[str, object]",
        json.loads(_SELECTED_RUNTIME_PATH.read_text(encoding="utf-8")),
    )
    selected_mixed = cast("dict[str, object]", selected["mixed_precision"])
    selected_compile = cast("dict[str, object]", selected["torch_compile"])
    selected_policy = cast("dict[str, object]", selected["runtime_policy"])
    expected = {
        "runtime_policy_id": selected["runtime_policy_id"],
        "memory_format": selected_policy["memory_format"],
        "autocast_dtype": selected_mixed["autocast_dtype"],
        "autocast_cache_enabled": selected_mixed["autocast_cache_enabled"],
        "fp32_loss": selected_mixed["fp32_loss"],
        "grad_scaler_enabled": selected_mixed["grad_scaler_enabled"],
        "compile_scope": selected_compile["scope"],
        "compile_backend": selected_compile["backend"],
        "compile_dynamic": selected_compile["dynamic"],
        "compile_mode": selected_compile["mode"],
        "cudagraphs": selected_compile["cudagraphs"],
        "inductor_options_json": selected_compile["inductor_options_json"],
        "optimize_ddp": selected_compile["optimize_ddp"],
        "compiled_autograd": selected_compile["compiled_autograd"],
        "reorder_compute_comm_overlap": selected_compile[
            "reorder_compute_comm_overlap"
        ],
        "fused_adamw": selected_policy["fused_optimizer"],
        "gradient_clip_foreach": selected_policy["gradient_clip_foreach"],
        "zero_grad_set_to_none": selected_policy["zero_grad_set_to_none"],
        "cudnn_benchmark": selected_policy["cudnn_benchmark"],
        "cudnn_deterministic": selected_policy["cudnn_deterministic"],
        "tf32_matmul": selected_policy["tf32_enabled"],
        "tf32_cudnn": selected_policy["tf32_enabled"],
        "matmul_precision": selected_policy["matmul_precision"],
        "ddp_static_graph": selected_policy["ddp_static_graph"],
        "ddp_gradient_as_bucket_view": selected_policy["ddp_gradient_as_bucket_view"],
        "ddp_broadcast_buffers": selected_policy["ddp_broadcast_buffers"],
        "ddp_find_unused_parameters": selected_policy["ddp_find_unused_parameters"],
        "ddp_bucket_cap_mb": selected_policy["ddp_bucket_cap_mb"],
        "communication_hook": selected_policy["communication_hook"],
    }
    if requested != expected:
        mismatches = sorted(
            key for key in requested if requested[key] != expected.get(key)
        )
        message = f"Spec 0011 runtime transfer drift: {mismatches}"
        raise RuntimeError(message)
    selected_bytes = _SELECTED_RUNTIME_PATH.read_bytes()
    selected_hash = hashlib.sha256(selected_bytes).hexdigest()
    if selected_hash != _SELECTED_RUNTIME_SHA256:
        message = "selected Spec 0011 runtime source hash drift"
        raise RuntimeError(message)
    return {
        "source_path": str(_SELECTED_RUNTIME_PATH),
        "source_sha256": selected_hash,
        "requested": requested,
        "effective": {
            **requested,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "matmul_precision": torch.get_float32_matmul_precision(),
            "optimize_ddp": torch_dynamo.config.optimize_ddp,
            "compiled_autograd": torch_dynamo.config.compiled_autograd,
            "reorder_compute_comm_overlap": (
                inductor_config.reorder_for_compute_comm_overlap
            ),
        },
    }


def _to_device(module: nn.Module, device: torch.device) -> nn.Module:
    return cast(
        "nn.Module",
        module.to(  # pyright: ignore[reportCallIssue]
            device=device,
            memory_format=torch.channels_last,
        ),
    )


def _manual_seed(seed: int) -> None:
    cast("Callable[[int], torch.Generator]", torch.manual_seed)(seed)


def _inputs(
    distributed: _Distributed,
    *,
    batch: int = PER_DEVICE_BATCH,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device=distributed.device)
    generator.manual_seed(20260813 + distributed.rank)
    shapes = (
        (batch, A_LAYOUT.channels, IMAGE_SIZE, IMAGE_SIZE),
        (batch, A_LAYOUT.channels, IMAGE_SIZE, IMAGE_SIZE),
        (batch, B_LAYOUT.channels, IMAGE_SIZE // 2, IMAGE_SIZE // 2),
        (batch, D_LAYOUT.channels, LATENT_SIZE, LATENT_SIZE),
    )
    return tuple(
        torch.randn(
            shape,
            generator=generator,
            device=distributed.device,
            dtype=torch.float32,
        ).contiguous(memory_format=torch.channels_last)
        for shape in shapes
    )


def _loss(outputs: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.stack([output.float().square().mean() for output in outputs]).sum()


def _buffer_fingerprint(module: nn.Module) -> tuple[tuple[object, ...], ...]:
    rows: list[tuple[object, ...]] = []
    for name, buffer in module.named_buffers():
        payload = buffer.detach().cpu().contiguous().numpy().tobytes()
        rows.append(
            (
                name,
                tuple(buffer.shape),
                str(buffer.dtype),
                buffer.device.type,
                hashlib.sha256(payload).hexdigest(),
            ),
        )
    return tuple(rows)


def _check_buffers_across_ranks(
    module: nn.Module,
    distributed: _Distributed,
) -> dict[str, object]:
    local = _buffer_fingerprint(module)
    gathered: list[object] = [None] * distributed.world_size
    cast("Callable[..., None]", dist.all_gather_object)(gathered, local)
    identical = all(item == local for item in gathered)
    if not identical:
        message = "persistent buffers differ across ranks before compilation"
        raise RuntimeError(message)
    return {"count": len(local), "identical_across_ranks": identical}


def _wrap_selected_ddp(
    module: nn.Module,
    distributed: _Distributed,
) -> nn.Module:
    wrapped = wrap_fastpath_ddp(
        module,
        local_rank=distributed.local_rank,
        static_graph=False,
        gradient_as_bucket_view=True,
        broadcast_buffers=False,
        find_unused_parameters=False,
        bucket_cap_mb=BUCKET_CAP_MB,
        dynamo=FastpathDynamoKnobs(
            optimize_ddp="python_reducer",
            compiled_autograd=True,
            reorder_compute_comm_overlap=True,
        ),
    )
    use_python_reducer = cast("object", getattr(wrapped, "_use_python_reducer", None))
    readback = {
        "python_reducer": use_python_reducer,
        "static_graph": getattr(wrapped, "static_graph", None),
        "gradient_as_bucket_view": getattr(
            wrapped,
            "gradient_as_bucket_view",
            None,
        ),
        "broadcast_buffers": getattr(wrapped, "broadcast_buffers", None),
        "find_unused_parameters": getattr(
            wrapped,
            "find_unused_parameters",
            None,
        ),
        "bucket_bytes_cap": getattr(wrapped, "bucket_bytes_cap", None),
        "optimize_ddp": torch_dynamo.config.optimize_ddp,
        "compiled_autograd": torch_dynamo.config.compiled_autograd,
        "reorder_compute_comm_overlap": (
            inductor_config.reorder_for_compute_comm_overlap
        ),
    }
    expected = {
        "python_reducer": True,
        "static_graph": False,
        "gradient_as_bucket_view": True,
        "broadcast_buffers": False,
        "find_unused_parameters": False,
        "bucket_bytes_cap": BUCKET_CAP_MB * 1024 * 1024,
        "optimize_ddp": "python_reducer",
        "compiled_autograd": True,
        "reorder_compute_comm_overlap": True,
    }
    if readback != expected:
        mismatches = sorted(key for key in expected if readback[key] != expected[key])
        message = f"selected DDP/compiler readback drift: {mismatches}"
        raise RuntimeError(message)
    return wrapped


def _gradient_mean_check(distributed: _Distributed) -> dict[str, object]:
    _manual_seed(130013)
    ddp_raw = cast(
        "SO2IdentityResidualBlockA",
        _to_device(SO2IdentityResidualBlockA(), distributed.device),
    )
    reference = cast(
        "SO2IdentityResidualBlockA",
        _to_device(copy.deepcopy(ddp_raw), distributed.device),
    )
    ddp = _wrap_selected_ddp(ddp_raw, distributed)
    ddp_optimizer = torch.optim.AdamW(ddp_raw.parameters(), lr=1e-4, fused=True)
    reference_optimizer = torch.optim.AdamW(
        reference.parameters(),
        lr=1e-4,
        fused=True,
    )
    inputs = _inputs(distributed, batch=1)[0][:, :, :16, :16]
    ddp_optimizer.zero_grad(set_to_none=True)
    reference_optimizer.zero_grad(set_to_none=True)
    reference_loss = cast("torch.Tensor", reference(inputs)).float().square().mean()
    reference_loss.backward()  # pyright: ignore[reportUnknownMemberType]
    local_gradient = cast("torch.Tensor", reference.main_conv1.coeff00.grad).detach()
    gathered = [torch.empty_like(local_gradient) for _ in range(distributed.world_size)]
    cast("Callable[..., object]", dist.all_gather)(gathered, local_gradient)
    local_gradients_differ = not torch.equal(gathered[0], gathered[1])
    for parameter in reference.parameters():
        gradient = parameter.grad
        if gradient is not None:
            cast("Callable[..., object]", dist.all_reduce)(
                gradient,
                op=dist.ReduceOp.SUM,
            )
            gradient.div_(distributed.world_size)
    ddp_loss = cast("torch.Tensor", ddp(inputs)).float().square().mean()
    ddp_loss.backward()  # pyright: ignore[reportUnknownMemberType]
    gradient_error = _maximum_parameter_difference(ddp_raw, reference, gradients=True)
    ddp_optimizer.step()  # pyright: ignore[reportUnknownMemberType]
    reference_optimizer.step()  # pyright: ignore[reportUnknownMemberType]
    update_error = _maximum_parameter_difference(ddp_raw, reference, gradients=False)
    passed = (
        local_gradients_differ
        and gradient_error <= DDP_REFERENCE_LIMIT
        and update_error <= DDP_REFERENCE_LIMIT
    )
    if not passed:
        message = "DDP gradient mean/update reference check failed"
        raise RuntimeError(message)
    del ddp, ddp_raw, reference, ddp_optimizer, reference_optimizer
    return {
        "local_pre_reduction_gradients_differ": local_gradients_differ,
        "reduced_gradient_max_abs_error": gradient_error,
        "final_update_max_abs_error": update_error,
        "pass": passed,
    }


def _maximum_parameter_difference(
    left: nn.Module,
    right: nn.Module,
    *,
    gradients: bool,
) -> float:
    maximum = 0.0
    for left_parameter, right_parameter in zip(
        left.parameters(),
        right.parameters(),
        strict=True,
    ):
        left_value = left_parameter.grad if gradients else left_parameter
        right_value = right_parameter.grad if gradients else right_parameter
        if left_value is None or right_value is None:
            continue
        difference = (left_value - right_value).detach().abs().max()
        maximum = max(maximum, float(difference))
    return maximum


def _graph_break_total() -> int:
    return int(sum(cast("dict[str, int]", counters["graph_break"]).values()))


def _unique_graph_total() -> int:
    return int(cast("dict[str, int]", counters["stats"]).get("unique_graphs", 0))


def _compiled_ddp_updates(  # noqa: PLR0914, PLR0915
    distributed: _Distributed,
) -> tuple[dict[str, object], _MechanicsSuite]:
    torch_dynamo.reset()
    counters.clear()
    _manual_seed(230013)
    raw = cast("_MechanicsSuite", _to_device(_MechanicsSuite(), distributed.device))
    buffer_result = _check_buffers_across_ranks(raw, distributed)
    ddp = _wrap_selected_ddp(raw, distributed)
    optimizer = torch.optim.AdamW(ddp.parameters(), lr=1e-4, fused=True)
    scaler = GradScaler("cuda", init_scale=GRAD_SCALER_INIT_SCALE)
    inputs = _inputs(distributed)

    def forward_loss(*arguments: torch.Tensor) -> torch.Tensor:
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            cache_enabled=True,
        ):
            outputs = cast("tuple[torch.Tensor, ...]", ddp(*arguments))
        return _loss(outputs)

    compiled = cast(
        "Callable[..., torch.Tensor]",
        torch.compile(  # pyright: ignore[reportUnknownMemberType]
            forward_loss,
            backend="inductor",
            dynamic=False,
        ),
    )

    def update() -> tuple[torch.Tensor, bool]:
        optimizer.zero_grad(set_to_none=True)
        with compiled_autograd_context(enabled=True):
            loss = compiled(*inputs)
            scaler.scale(loss).backward()  # pyright: ignore[reportUnknownMemberType]
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            ddp.parameters(),
            max_norm=MAX_GRAD_NORM,
            foreach=True,
        )
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        return loss.detach(), scaler.get_scale() < previous_scale

    initial_graph_before = _graph_break_total()
    for _ in range(WARMUP_UPDATES):
        update()
    initial_graph_breaks = _graph_break_total() - initial_graph_before
    graph_before = _graph_break_total()
    unique_before = _unique_graph_total()
    torch.cuda.reset_peak_memory_stats(distributed.device)
    losses: list[float] = []
    skipped = 0
    for _ in range(SETTLED_UPDATES):
        loss, did_skip = update()
        losses.append(float(loss))
        skipped += int(did_skip)
    torch.cuda.synchronize(distributed.device)
    graph_breaks = _graph_break_total() - graph_before
    recompiles = _unique_graph_total() - unique_before
    allocated = (
        float(torch.cuda.max_memory_allocated(distributed.device)) / BYTES_PER_MIB
    )
    reserved = float(torch.cuda.max_memory_reserved(distributed.device)) / BYTES_PER_MIB
    finite = all(
        bool(torch.isfinite(parameter).all().item()) for parameter in raw.parameters()
    )
    nonfinite_losses = sum(not math.isfinite(loss) for loss in losses)
    local_worst_parameter = ""
    local_parameter_difference = 0.0
    for name, parameter in raw.named_parameters():
        rank_zero = parameter.detach().clone()
        cast("Callable[..., object]", dist.broadcast)(rank_zero, src=0)
        difference = float((parameter.detach() - rank_zero).abs().max())
        if difference > local_parameter_difference:
            local_parameter_difference = difference
            local_worst_parameter = name
    gathered_parameter_differences: list[object] = [None] * distributed.world_size
    cast("Callable[..., None]", dist.all_gather_object)(
        gathered_parameter_differences,
        (local_parameter_difference, local_worst_parameter),
    )
    parameter_difference, worst_parameter = max(
        cast("list[tuple[float, str]]", gathered_parameter_differences),
    )
    rank_max = torch.tensor(
        [
            allocated,
            reserved,
            float(skipped),
            float(graph_breaks),
            float(recompiles),
            float(nonfinite_losses),
            float(not finite),
            float(initial_graph_breaks),
        ],
        device=distributed.device,
    )
    cast("Callable[..., object]", dist.all_reduce)(rank_max, op=dist.ReduceOp.MAX)
    rank_values = [float(rank_max[index].item()) for index in range(8)]
    (
        allocated,
        reserved,
        skipped_float,
        graph_float,
        recompile_float,
        nonfinite_float,
        nonfinite_parameters_float,
        initial_graph_breaks_float,
    ) = rank_values
    result: dict[str, object] = {
        "warmup_updates": WARMUP_UPDATES,
        "settled_updates": SETTLED_UPDATES,
        "nonfinite_loss_count": int(nonfinite_float),
        "finite_parameters": not bool(nonfinite_parameters_float),
        "amp_skip_count": int(skipped_float),
        "post_settle_graph_break_count": int(graph_float),
        "post_settle_recompile_count": int(recompile_float),
        "initial_graph_break_count": int(initial_graph_breaks_float),
        "cross_rank_parameter_max_abs_difference": parameter_difference,
        "cross_rank_worst_parameter_name": worst_parameter,
        "peak_allocated_mib": allocated,
        "peak_reserved_mib": reserved,
        "buffers": buffer_result,
    }
    return result, raw


def _relative_rms(observed: torch.Tensor, reference: torch.Tensor) -> float:
    numerator = (observed.double() - reference.double()).square().mean().sqrt()
    denominator = reference.double().square().mean().sqrt().clamp_min(1e-12)
    return float((numerator / denominator).detach())


def _block_cases() -> tuple[_BlockCase, ...]:
    return (
        _BlockCase(
            "identity_A",
            SO2IdentityResidualBlockA(),
            EncoderResBlock(
                in_channels=A_LAYOUT.channels,
                out_channels=A_LAYOUT.channels,
                downsample=False,
                norm_groups=8,
            ),
            (PER_DEVICE_BATCH, A_LAYOUT.channels, IMAGE_SIZE, IMAGE_SIZE),
        ),
        _BlockCase(
            "encoder_A_to_B",
            SO2EncoderTransitionAB(),
            EncoderResBlock(
                in_channels=A_LAYOUT.channels,
                out_channels=B_LAYOUT.channels,
                downsample=True,
                norm_groups=8,
            ),
            (PER_DEVICE_BATCH, A_LAYOUT.channels, IMAGE_SIZE, IMAGE_SIZE),
        ),
        _BlockCase(
            "decoder_B_to_A",
            SO2DecoderTransitionBA(),
            DecoderUpResBlock(
                in_channels=B_LAYOUT.channels,
                out_channels=A_LAYOUT.channels,
                upsample=True,
                norm_groups=8,
            ),
            (PER_DEVICE_BATCH, B_LAYOUT.channels, IMAGE_SIZE // 2, IMAGE_SIZE // 2),
        ),
        _BlockCase(
            "largest_D_to_D",
            SO2LargestDDConv(),
            nn.Conv2d(
                D_LAYOUT.channels,
                D_LAYOUT.channels,
                kernel_size=5,
                padding=2,
                bias=False,
            ),
            (PER_DEVICE_BATCH, D_LAYOUT.channels, LATENT_SIZE, LATENT_SIZE),
        ),
    )


def _coefficient_gradient_comparison(
    eager: nn.Module,
    compiled: nn.Module,
) -> tuple[float, str, float, float, list[str], int]:
    eager_coefficients = {
        name: parameter
        for name, parameter in eager.named_parameters()
        if name.rsplit(".", 1)[-1].startswith("coeff")
    }
    compiled_coefficients = {
        name: parameter
        for name, parameter in compiled.named_parameters()
        if name.rsplit(".", 1)[-1].startswith("coeff")
    }
    missing = sorted(set(eager_coefficients) ^ set(compiled_coefficients))
    worst_error = 0.0
    worst_name = ""
    worst_reference_rms = 0.0
    worst_difference_rms = 0.0
    nonfinite = 0
    for name in sorted(set(eager_coefficients) & set(compiled_coefficients)):
        eager_gradient = eager_coefficients[name].grad
        compiled_gradient = compiled_coefficients[name].grad
        if eager_gradient is None or compiled_gradient is None:
            missing.append(name)
            continue
        nonfinite += int(not bool(torch.isfinite(eager_gradient).all().item()))
        nonfinite += int(not bool(torch.isfinite(compiled_gradient).all().item()))
        reference_rms = float(eager_gradient.double().square().mean().sqrt())
        difference_rms = float(
            (compiled_gradient.double() - eager_gradient.double())
            .square()
            .mean()
            .sqrt(),
        )
        relative_error = difference_rms / max(reference_rms, 1e-12)
        if relative_error > worst_error:
            worst_error = relative_error
            worst_name = name
            worst_reference_rms = reference_rms
            worst_difference_rms = difference_rms
    return (
        worst_error,
        worst_name,
        worst_reference_rms,
        worst_difference_rms,
        sorted(set(missing)),
        nonfinite,
    )


def _selected_accuracy_backward(loss: torch.Tensor, module: nn.Module) -> None:
    scaler = GradScaler("cuda", init_scale=GRAD_SCALER_INIT_SCALE)
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-4, fused=True)
    with compiled_autograd_context(enabled=True):
        scaler.scale(loss).backward()  # pyright: ignore[reportUnknownMemberType]
    scaler.unscale_(optimizer)


def _accuracy_case(  # noqa: PLR0914
    case: _BlockCase,
    device: torch.device,
) -> dict[str, object]:
    _manual_seed(330013)
    eager = _to_device(copy.deepcopy(case.equivariant), device)
    compiled_source = _to_device(copy.deepcopy(eager), device)
    inputs = torch.randn(case.shape, device=device).contiguous(
        memory_format=torch.channels_last,
    )
    eager_inputs = inputs.detach().clone().requires_grad_()
    compiled_inputs = inputs.detach().clone().requires_grad_()
    eager_output = cast("torch.Tensor", eager(eager_inputs))
    eager_loss = eager_output.square().mean()
    eager_loss.backward()  # pyright: ignore[reportUnknownMemberType]
    compiled = cast(
        "Callable[[torch.Tensor], torch.Tensor]",
        torch.compile(  # pyright: ignore[reportUnknownMemberType]
            compiled_source,
            backend="inductor",
            fullgraph=True,
            dynamic=False,
        ),
    )
    with torch.autocast("cuda", dtype=torch.float16, cache_enabled=True):
        compiled_output = compiled(compiled_inputs)
    compiled_loss = compiled_output.float().square().mean()
    _selected_accuracy_backward(compiled_loss, compiled_source)
    (
        gradient_error,
        worst_gradient_name,
        worst_gradient_reference_rms,
        worst_gradient_difference_rms,
        missing_gradients,
        nonfinite_gradients,
    ) = _coefficient_gradient_comparison(eager, compiled_source)
    return {
        "output_relative_rms": _relative_rms(compiled_output.float(), eager_output),
        "max_coefficient_gradient_relative_rms": gradient_error,
        "worst_coefficient_gradient_name": worst_gradient_name,
        "worst_coefficient_gradient_reference_rms": worst_gradient_reference_rms,
        "worst_coefficient_gradient_difference_rms": worst_gradient_difference_rms,
        "missing_coefficient_gradients": missing_gradients,
        "nonfinite_coefficient_gradient_count": nonfinite_gradients,
    }


def _timed_cuda_call(
    function: Callable[..., torch.Tensor],
    device: torch.device,
    *arguments: torch.Tensor,
) -> float:
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    function(*arguments)
    torch.cuda.synchronize(device)
    return (time.perf_counter() - start) * MILLISECONDS_PER_SECOND


def _window_summary(samples: Sequence[float]) -> dict[str, object]:
    return {
        "samples_ms": list(samples),
        "median_ms": statistics.median(samples),
        "coefficient_variation": statistics.pstdev(samples) / statistics.mean(samples),
    }


def _prepare_timed_step(
    name: str,
    module: nn.Module,
    inputs: torch.Tensor,
    distributed: _Distributed,
    *,
    compile_module: bool,
) -> _PreparedStep:
    ddp = _wrap_selected_ddp(module, distributed)
    optimizer = torch.optim.AdamW(ddp.parameters(), lr=1e-4, fused=True)
    scaler = GradScaler("cuda", init_scale=GRAD_SCALER_INIT_SCALE)

    def forward_loss() -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.float16, cache_enabled=True):
            output = cast("torch.Tensor", ddp(inputs))
        return output.float().square().mean()

    callable_loss: Callable[[], torch.Tensor] = (
        torch.compile(  # pyright: ignore[reportUnknownMemberType]
            forward_loss,
            backend="inductor",
            dynamic=False,
        )
        if compile_module
        else forward_loss
    )

    def step() -> tuple[torch.Tensor, torch.Tensor]:
        optimizer.zero_grad(set_to_none=True)
        with compiled_autograd_context(enabled=compile_module):
            loss = callable_loss()
            scaler.scale(loss).backward()  # pyright: ignore[reportUnknownMemberType]
        scaler.unscale_(optimizer)
        finite_gradients = torch.stack(
            tuple(
                torch.isfinite(parameter.grad).all()
                for parameter in module.parameters()
                if parameter.grad is not None
            ),
        ).all()
        torch.nn.utils.clip_grad_norm_(
            ddp.parameters(),
            max_norm=MAX_GRAD_NORM,
            foreach=True,
        )
        scaler.step(optimizer)
        scaler.update()
        return loss.detach(), finite_gradients

    return _PreparedStep(name, step, scaler)


def _timed_step(
    prepared: _PreparedStep,
    device: torch.device,
) -> tuple[float, bool, bool, bool]:
    previous_scale = prepared.scaler.get_scale()
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    loss, finite_gradients = prepared.step()
    torch.cuda.synchronize(device)
    elapsed = (time.perf_counter() - start) * MILLISECONDS_PER_SECOND
    did_skip = prepared.scaler.get_scale() < previous_scale
    finite_loss = bool(torch.isfinite(loss).item())
    finite_gradient = bool(finite_gradients.item())
    return elapsed, did_skip, finite_loss, finite_gradient


def _measure_case_steps(
    case: _BlockCase,
    distributed: _Distributed,
) -> JsonObject:
    generator = torch.Generator(device=distributed.device)
    generator.manual_seed(430013 + distributed.rank)
    inputs = torch.randn(
        case.shape,
        generator=generator,
        device=distributed.device,
    ).contiguous(memory_format=torch.channels_last)
    paths = (
        _prepare_timed_step(
            "equivariant_eager",
            _to_device(copy.deepcopy(case.equivariant), distributed.device),
            inputs,
            distributed,
            compile_module=False,
        ),
        _prepare_timed_step(
            "equivariant_compiled",
            _to_device(copy.deepcopy(case.equivariant), distributed.device),
            inputs,
            distributed,
            compile_module=True,
        ),
        _prepare_timed_step(
            "normal_compiled",
            _to_device(copy.deepcopy(case.normal), distributed.device),
            inputs,
            distributed,
            compile_module=True,
        ),
    )
    for _ in range(WARMUP_UPDATES):
        for path in paths:
            path.step()
    rows: dict[str, dict[str, object]] = {
        path.name: {
            "windows": [],
            "amp_skip_count": 0,
            "nonfinite_loss_count": 0,
            "nonfinite_gradient_count": 0,
        }
        for path in paths
    }
    for order in (paths, tuple(reversed(paths))):
        samples: dict[str, list[float]] = {path.name: [] for path in paths}
        for _ in range(TIMED_WINDOW_UPDATES):
            for path in order:
                elapsed, skipped, finite_loss, finite_gradient = _timed_step(
                    path,
                    distributed.device,
                )
                samples[path.name].append(elapsed)
                row = rows[path.name]
                row["amp_skip_count"] = int(
                    cast("int", row["amp_skip_count"]),
                ) + int(skipped)
                row["nonfinite_loss_count"] = int(
                    cast("int", row["nonfinite_loss_count"]),
                ) + int(not finite_loss)
                row["nonfinite_gradient_count"] = int(
                    cast("int", row["nonfinite_gradient_count"]),
                ) + int(not finite_gradient)
        for path in paths:
            cast("list[object]", rows[path.name]["windows"]).append(
                _window_summary(samples[path.name]),
            )
    for path in paths:
        window_rows = cast("list[dict[str, object]]", rows[path.name]["windows"])
        pooled = cast("list[float]", window_rows[0]["samples_ms"]) + cast(
            "list[float]",
            window_rows[1]["samples_ms"],
        )
        rows[path.name]["pooled"] = _window_summary(pooled)
    eager_median = float(
        cast(
            "float",
            cast("dict[str, object]", rows["equivariant_eager"]["pooled"])["median_ms"],
        ),
    )
    compiled_median = float(
        cast(
            "float",
            cast("dict[str, object]", rows["equivariant_compiled"]["pooled"])[
                "median_ms"
            ],
        ),
    )
    normal_median = float(
        cast(
            "float",
            cast("dict[str, object]", rows["normal_compiled"]["pooled"])["median_ms"],
        ),
    )
    return cast(
        "JsonObject",
        {
            "name": case.name,
            "shape": list(case.shape),
            **_accuracy_case(case, distributed.device),
            "paths": rows,
            "compiled_over_eager": compiled_median / eager_median,
            "equivariant_over_normal": compiled_median / normal_median,
        },
    )


def _measure_blocks(distributed: _Distributed) -> list[JsonObject]:
    rows: list[JsonObject] = []
    for case in _block_cases():
        rows.append(_measure_case_steps(case, distributed))
        torch_dynamo.reset()
        gc.collect()
        torch.cuda.empty_cache()
    return rows


def _assembly_diagnostic(
    source: SO2LargestDDConv,
    distributed: _Distributed,
) -> JsonObject:
    inputs = _inputs(distributed)[-1]
    module = _to_device(copy.deepcopy(source), distributed.device)

    def expand() -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.float16, cache_enabled=True):
            return cast("SO2LargestDDConv", module).expanded_kernel()

    def forward(values: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.float16, cache_enabled=True):
            return cast("torch.Tensor", module(values))

    compiled_expand = torch.compile(  # pyright: ignore[reportUnknownMemberType]
        expand,
        backend="inductor",
        fullgraph=True,
        dynamic=False,
    )
    compiled_forward = cast(
        "Callable[[torch.Tensor], torch.Tensor]",
        torch.compile(  # pyright: ignore[reportUnknownMemberType]
            forward,
            backend="inductor",
            fullgraph=True,
            dynamic=False,
        ),
    )
    for _ in range(WARMUP_UPDATES):
        compiled_expand()
        compiled_forward(inputs)
    windows: list[JsonObject] = []
    expansion_pooled: list[float] = []
    complete_pooled: list[float] = []
    for reverse in (False, True):
        expansion: list[float] = []
        complete: list[float] = []
        for _ in range(TIMED_WINDOW_UPDATES):
            calls = (
                (
                    (compiled_forward, (inputs,), complete),
                    (compiled_expand, (), expansion),
                )
                if reverse
                else (
                    (compiled_expand, (), expansion),
                    (compiled_forward, (inputs,), complete),
                )
            )
            for function, arguments, samples in calls:
                samples.append(
                    _timed_cuda_call(function, distributed.device, *arguments),
                )
        expansion_pooled.extend(expansion)
        complete_pooled.extend(complete)
        windows.append(
            cast(
                "JsonObject",
                {
                    "expansion": _window_summary(expansion),
                    "complete": _window_summary(complete),
                    "assembly_fraction": (
                        statistics.median(expansion) / statistics.median(complete)
                    ),
                },
            ),
        )
    return cast(
        "JsonObject",
        {
            "selection_gate": False,
            "windows": windows,
            "pooled_expansion": _window_summary(expansion_pooled),
            "pooled_complete": _window_summary(complete_pooled),
            "assembly_fraction": (
                statistics.median(expansion_pooled) / statistics.median(complete_pooled)
            ),
        },
    )


def _finite_timing_summary(summary: JsonObject) -> bool:
    samples = cast("list[object]", summary.get("samples_ms", []))
    values = (
        summary.get("median_ms"),
        summary.get("coefficient_variation"),
        *samples,
    )
    return all(
        isinstance(value, int | float) and math.isfinite(float(value))
        for value in values
    )


def _verdict(  # noqa: C901, PLR0914
    updates: dict[str, object],
    rank_results: Sequence[JsonObject],
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    expected_blocks = {
        "identity_A",
        "encoder_A_to_B",
        "decoder_B_to_A",
        "largest_D_to_D",
    }
    expected_paths = {
        "equivariant_eager",
        "equivariant_compiled",
        "normal_compiled",
    }
    if [cast("int", row.get("rank")) for row in rank_results] != [0, 1]:
        failures.append("rank_measurement_set")
    scalar_checks = (
        (int(cast("int", updates["amp_skip_count"])) == 0, "amp_skips"),
        (
            int(cast("int", updates["nonfinite_loss_count"])) == 0,
            "nonfinite_losses",
        ),
        (bool(cast("bool", updates["finite_parameters"])), "nonfinite_parameters"),
        (
            int(cast("int", updates["post_settle_graph_break_count"])) == 0,
            "graph_breaks",
        ),
        (
            int(cast("int", updates["initial_graph_break_count"])) == 0,
            "initial_graph_breaks",
        ),
        (int(cast("int", updates["post_settle_recompile_count"])) == 0, "recompiles"),
        (
            float(cast("float", updates["peak_allocated_mib"]))
            < PEAK_ALLOCATED_MIB_LIMIT,
            "allocated_vram",
        ),
        (
            float(cast("float", updates["peak_reserved_mib"]))
            < PEAK_RESERVED_MIB_LIMIT,
            "reserved_vram",
        ),
        (
            float(cast("float", updates["cross_rank_parameter_max_abs_difference"]))
            <= DDP_REFERENCE_LIMIT,
            "cross_rank_parameters",
        ),
    )
    failures.extend(name for passed, name in scalar_checks if not passed)
    for rank_result in rank_results:
        rank = cast("int", rank_result["rank"])
        blocks = cast("list[JsonObject]", rank_result.get("blocks", []))
        if (
            len(blocks) != len(expected_blocks)
            or {cast("str", row.get("name")) for row in blocks} != expected_blocks
        ):
            failures.append(f"rank{rank}:block_set")
        assembly = cast("JsonObject", rank_result.get("assembly_diagnostic", {}))
        assembly_windows = cast("list[JsonObject]", assembly.get("windows", []))
        assembly_summaries = [
            cast("JsonObject", window.get(path, {}))
            for window in assembly_windows
            for path in ("expansion", "complete")
        ] + [
            cast("JsonObject", assembly.get(path, {}))
            for path in ("pooled_expansion", "pooled_complete")
        ]
        assembly_schema_valid = (
            assembly.get("selection_gate") is False
            and len(assembly_windows) == TIMED_WINDOW_COUNT
            and all(
                len(
                    cast(
                        "list[object]",
                        cast("JsonObject", window.get(path, {})).get(
                            "samples_ms",
                            [],
                        ),
                    ),
                )
                == TIMED_WINDOW_UPDATES
                for window in assembly_windows
                for path in ("expansion", "complete")
            )
            and all(
                len(
                    cast(
                        "list[object]",
                        cast("JsonObject", assembly.get(path, {})).get(
                            "samples_ms",
                            [],
                        ),
                    ),
                )
                == 2 * TIMED_WINDOW_UPDATES
                for path in ("pooled_expansion", "pooled_complete")
            )
            and all(_finite_timing_summary(summary) for summary in assembly_summaries)
        )
        if not assembly_schema_valid:
            failures.append(f"rank{rank}:assembly_diagnostic_schema")
        for row in blocks:
            name = cast("str", row["name"])
            checks = (
                (
                    float(cast("float", row["output_relative_rms"]))
                    <= OUTPUT_RELATIVE_LIMIT,
                    "output",
                ),
                (
                    float(
                        cast("float", row["max_coefficient_gradient_relative_rms"]),
                    )
                    <= GRADIENT_RELATIVE_LIMIT,
                    "gradient",
                ),
                (
                    cast("list[object]", row["missing_coefficient_gradients"]) == [],
                    "missing_gradient",
                ),
                (
                    int(cast("int", row["nonfinite_coefficient_gradient_count"])) == 0,
                    "nonfinite_gradient",
                ),
                (
                    float(cast("float", row["compiled_over_eager"]))
                    <= COMPILED_EAGER_RATIO_LIMIT,
                    "compiled_eager",
                ),
                (
                    float(cast("float", row["equivariant_over_normal"]))
                    <= NORMAL_RATIO_LIMIT,
                    "normal_ratio",
                ),
            )
            failures.extend(
                f"rank{rank}:{name}:{metric}" for passed, metric in checks if not passed
            )
            paths = cast("dict[str, object]", row.get("paths", {}))
            if set(paths) != expected_paths:
                failures.append(f"rank{rank}:{name}:path_set")
            for path_name, raw_path in paths.items():
                path = cast("JsonObject", raw_path)
                invalid_steps = sum(
                    int(cast("int", path[key]))
                    for key in (
                        "amp_skip_count",
                        "nonfinite_loss_count",
                        "nonfinite_gradient_count",
                    )
                )
                if invalid_steps:
                    failures.append(f"rank{rank}:{name}:{path_name}:invalid_step")
                summaries = [
                    *cast("list[JsonObject]", path.get("windows", [])),
                    cast("JsonObject", path.get("pooled", {})),
                ]
                if len(summaries) != TIMED_WINDOW_COUNT + 1 or any(
                    len(cast("list[object]", summary.get("samples_ms", [])))
                    != expected_count
                    for summary, expected_count in zip(
                        summaries,
                        (
                            TIMED_WINDOW_UPDATES,
                            TIMED_WINDOW_UPDATES,
                            2 * TIMED_WINDOW_UPDATES,
                        ),
                        strict=True,
                    )
                ):
                    failures.append(f"rank{rank}:{name}:{path_name}:sample_schema")
                if any(not _finite_timing_summary(summary) for summary in summaries):
                    failures.append(f"rank{rank}:{name}:{path_name}:timing_schema")
    return not failures, failures


def _gather_rank_results(
    blocks: list[JsonObject],
    assembly_diagnostic: JsonObject,
    distributed: _Distributed,
) -> list[JsonObject]:
    local = cast(
        "JsonObject",
        {
            "rank": distributed.rank,
            "blocks": blocks,
            "assembly_diagnostic": assembly_diagnostic,
        },
    )
    gathered: list[object] = [None] * distributed.world_size
    cast("Callable[..., None]", dist.all_gather_object)(gathered, local)
    return [cast("JsonObject", item) for item in gathered]


def run(output_dir: Path) -> JsonObject:
    """Execute the one fixed transfer check and write its rank-zero result.

    Returns:
        Compact result payload on every rank.

    Raises:
        RuntimeError: If hardware or any locked acceptance check fails.

    """
    distributed = _init_distributed()
    device_assignments = _device_assignments(distributed)
    requested_effective_runtime = _apply_selected_runtime()
    gradient_mean = _gradient_mean_check(distributed)
    updates, trained_suite = _compiled_ddp_updates(distributed)
    largest = copy.deepcopy(trained_suite.largest)
    del trained_suite
    gc.collect()
    torch.cuda.empty_cache()
    blocks = _measure_blocks(distributed)
    assembly_diagnostic = _assembly_diagnostic(largest, distributed)
    rank_results = _gather_rank_results(
        blocks,
        assembly_diagnostic,
        distributed,
    )
    passed, failures = _verdict(updates, rank_results)
    result = cast(
        "JsonObject",
        {
            "schema_version": SCHEMA_VERSION,
            "benchmark_kind": PROBE_KIND,
            "status": "pass" if passed else "fail",
            "architecture_locked": True,
            "selected_mechanics": "padded_bmm_direct",
            "full_vae_assembled": False,
            "follow_up_probe_permitted": False,
            "world_size": distributed.world_size,
            "nproc_per_node": distributed.nproc_per_node,
            "gpu_names": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
            "rank_device_assignments": device_assignments,
            **torch_runtime_versions(),
            "per_device_batch_size": PER_DEVICE_BATCH,
            "fixed_shapes": {
                "identity_A": [
                    PER_DEVICE_BATCH,
                    A_LAYOUT.channels,
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                ],
                "encoder_A_to_B": [
                    PER_DEVICE_BATCH,
                    A_LAYOUT.channels,
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                ],
                "decoder_B_to_A": [
                    PER_DEVICE_BATCH,
                    B_LAYOUT.channels,
                    IMAGE_SIZE // 2,
                    IMAGE_SIZE // 2,
                ],
                "largest_D_to_D": [
                    PER_DEVICE_BATCH,
                    D_LAYOUT.channels,
                    LATENT_SIZE,
                    LATENT_SIZE,
                ],
            },
            "runtime_requested_and_effective": requested_effective_runtime,
            "gradient_mean_reference": gradient_mean,
            "compiled_ddp_updates": updates,
            "rank_measurements": rank_results,
            "acceptance_failures": failures,
        },
    )
    if distributed.rank == _PRIMARY_RANK:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / ARTIFACT_FILENAME, result)
    cast("Callable[[], object]", dist.barrier)()
    if not passed:
        message = f"Spec 0013 dual-T4 acceptance failed: {failures}"
        raise RuntimeError(message)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """Run the CLI entrypoint.

    Returns:
        Process exit status.

    """
    args = _parse_args()
    run(cast("Path", args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARTIFACT_FILENAME",
    "PER_DEVICE_BATCH",
    "RUNTIME_BUNDLE_ID",
    "SCHEMA_VERSION",
    "SETTLED_UPDATES",
    "main",
    "run",
]
