# Copyright 2026 HiperMaximus
"""One-use dual-T4 mechanics probe for the locked Spec 0013 modules.

This is deliberately a singular benchmark, not a runtime tuner. It executes the
selected Spec 0011 bundle once on generated fixed-shape tensors and writes one
compact result. Any follow-up arm must first be justified by this run and added
explicitly to Spec 0013.
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
from itertools import starmap
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import torch
import torch._dynamo as torch_dynamo  # noqa: PLC2701
import torch.distributed as dist
from torch import nn
from torch._dynamo.utils import counters  # noqa: PLC2701
from torch._inductor import config as inductor_config  # noqa: PLC2701
from torch.amp.grad_scaler import GradScaler
from torch.nn import functional

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
    _expand_pair,
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

SCHEMA_VERSION: Final = "spec0013.so2_dual_t4_follow_up.v1"
ARTIFACT_FILENAME: Final = "spec0013_so2_dual_t4_probe.json"
PROBE_KIND: Final = "locked_so2_architecture_mechanics_follow_up"
RUNTIME_BUNDLE_ID: Final = "compile_step_python_reducer_fp16_channels_last"
PER_DEVICE_BATCH: Final = 4
SETTLED_UPDATES: Final = 32
WARMUP_UPDATES: Final = 5
TIMED_UPDATES: Final = 20
FOLLOW_UP_WARMUPS: Final = 20
FOLLOW_UP_WINDOW_UPDATES: Final = 50
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
ASSEMBLY_FRACTION_LIMIT: Final = 0.10
NORMAL_RATIO_LIMIT: Final = 5.0
TIMING_CV_LIMIT: Final = 0.10
PEAK_RESERVED_MIB_LIMIT: Final = 14.5 * 1024.0
PEAK_ALLOCATED_MIB_LIMIT: Final = 13.5 * 1024.0
DDP_REFERENCE_LIMIT: Final = 1e-6
FP32_CANDIDATE_KERNEL_LIMIT: Final = 1e-6
REQUIRED_WORLD_SIZE: Final = 2
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
    topology_weight: int


@dataclass(frozen=True)
class _TimingResult:
    median_ms: float
    coefficient_variation: float
    samples_ms: tuple[float, ...]
    amp_skip_count: int
    nonfinite_loss_count: int
    nonfinite_gradient_count: int


@dataclass(frozen=True)
class _CompiledArm:
    name: str
    module: nn.Module
    expand: Callable[[], torch.Tensor]
    forward: Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class _PreparedStep:
    name: str
    step: Callable[[], tuple[torch.Tensor, torch.Tensor]]
    scaler: GradScaler


class _DDirectAssemblyConv(nn.Module):
    """Measured D-to-D candidate: four mm calls and one final slice buffer."""

    def __init__(self, source: SO2LargestDDConv) -> None:
        super().__init__()
        self.conv = copy.deepcopy(source.conv)

    def expanded_kernel(self) -> torch.Tensor:
        """Expand four fixed pairs into one freshly allocated dense kernel.

        Returns:
            Dense D-to-D kernel.

        """
        conv = self.conv
        kernel00 = _expand_pair(
            conv.coeff00,
            conv.basis00,
            output_copies=D_LAYOUT.n0,
            input_copies=D_LAYOUT.n0,
            output_dimension=1,
            input_dimension=1,
            kernel_size=7,
        )
        kernel10 = _expand_pair(
            conv.coeff10,
            conv.basis10,
            output_copies=D_LAYOUT.n1,
            input_copies=D_LAYOUT.n0,
            output_dimension=2,
            input_dimension=1,
            kernel_size=7,
        )
        kernel01 = _expand_pair(
            conv.coeff01,
            conv.basis01,
            output_copies=D_LAYOUT.n0,
            input_copies=D_LAYOUT.n1,
            output_dimension=1,
            input_dimension=2,
            kernel_size=7,
        )
        kernel11 = _expand_pair(
            conv.coeff11,
            conv.basis11,
            output_copies=D_LAYOUT.n1,
            input_copies=D_LAYOUT.n1,
            output_dimension=2,
            input_dimension=2,
            kernel_size=7,
        )
        kernel = kernel00.new_empty(
            (D_LAYOUT.channels, D_LAYOUT.channels, 7, 7),
        )
        kernel[: D_LAYOUT.n0, : D_LAYOUT.n0] = kernel00
        kernel[: D_LAYOUT.n0, D_LAYOUT.n0 :] = kernel01
        kernel[D_LAYOUT.n0 :, : D_LAYOUT.n0] = kernel10
        kernel[D_LAYOUT.n0 :, D_LAYOUT.n0 :] = kernel11
        return kernel

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Execute exactly one dense convolution with the direct assembly.

        Returns:
            D-layout output.

        """
        return functional.conv2d(inputs, self.expanded_kernel(), padding=3)


def _reshape_padded_pair(
    flat: torch.Tensor,
    *,
    output_dimension: int,
    input_dimension: int,
) -> torch.Tensor:
    selected = flat[:, : output_dimension * input_dimension * 7 * 7]
    return (
        selected
        .view(
            D_LAYOUT.n0,
            D_LAYOUT.n0,
            output_dimension,
            input_dimension,
            7,
            7,
        )
        .permute(0, 2, 1, 3, 4, 5)
        .reshape(
            D_LAYOUT.n0 * output_dimension,
            D_LAYOUT.n0 * input_dimension,
            7,
            7,
        )
    )


class _DPaddedBmmAssemblyConv(nn.Module):
    """Measured D-to-D candidate: one padded bmm and direct final assembly."""

    packed_bases: torch.Tensor

    def __init__(self, source: SO2LargestDDConv) -> None:
        super().__init__()
        self.conv = copy.deepcopy(source.conv)
        packed = self.conv.basis00.new_zeros((4, 14, 196))
        packed[0, :4, :49] = self.conv.basis00
        packed[1, :6, :98] = self.conv.basis10
        packed[2, :6, :98] = self.conv.basis01
        packed[3, :14, :196] = self.conv.basis11
        self.register_buffer("packed_bases", packed.contiguous(), persistent=True)

    def expanded_kernel(self) -> torch.Tensor:
        """Expand all four fixed pairs in one padded batched contraction.

        Returns:
            Dense D-to-D kernel.

        """
        conv = self.conv
        coefficients = torch.stack(
            (
                functional.pad(conv.coeff00, (0, 10)),
                functional.pad(conv.coeff10, (0, 8)),
                functional.pad(conv.coeff01, (0, 8)),
                conv.coeff11,
            ),
        )
        expanded = torch.bmm(coefficients, self.packed_bases)
        kernel00 = _reshape_padded_pair(
            expanded[0],
            output_dimension=1,
            input_dimension=1,
        )
        kernel10 = _reshape_padded_pair(
            expanded[1],
            output_dimension=2,
            input_dimension=1,
        )
        kernel01 = _reshape_padded_pair(
            expanded[2],
            output_dimension=1,
            input_dimension=2,
        )
        kernel11 = _reshape_padded_pair(
            expanded[3],
            output_dimension=2,
            input_dimension=2,
        )
        kernel = kernel00.new_empty(
            (D_LAYOUT.channels, D_LAYOUT.channels, 7, 7),
        )
        kernel[: D_LAYOUT.n0, : D_LAYOUT.n0] = kernel00
        kernel[: D_LAYOUT.n0, D_LAYOUT.n0 :] = kernel01
        kernel[D_LAYOUT.n0 :, : D_LAYOUT.n0] = kernel10
        kernel[D_LAYOUT.n0 :, D_LAYOUT.n0 :] = kernel11
        return kernel

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Execute exactly one dense convolution with the padded contraction.

        Returns:
            D-layout output.

        """
        return functional.conv2d(inputs, self.expanded_kernel(), padding=3)


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
    if (
        world_size != REQUIRED_WORLD_SIZE
        or nproc_per_node != REQUIRED_WORLD_SIZE
        or torch.cuda.device_count() != REQUIRED_WORLD_SIZE
    ):
        message = "Spec 0013 probe requires exactly two visible GPUs and world_size=2"
        raise RuntimeError(message)
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    return _Distributed(device, rank, local_rank, world_size, nproc_per_node)


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


def _compiled_ddp_updates(  # noqa: PLR0914
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
    rank_max = torch.tensor(
        [
            allocated,
            reserved,
            float(skipped),
            float(graph_breaks),
            float(recompiles),
            float(nonfinite_losses),
            float(not finite),
        ],
        device=distributed.device,
    )
    cast("Callable[..., object]", dist.all_reduce)(rank_max, op=dist.ReduceOp.MAX)
    rank_values = [float(rank_max[index].item()) for index in range(7)]
    (
        allocated,
        reserved,
        skipped_float,
        graph_float,
        recompile_float,
        nonfinite_float,
        nonfinite_parameters_float,
    ) = rank_values
    result: dict[str, object] = {
        "warmup_updates": WARMUP_UPDATES,
        "settled_updates": SETTLED_UPDATES,
        "nonfinite_loss_count": int(nonfinite_float),
        "finite_parameters": not bool(nonfinite_parameters_float),
        "amp_skip_count": int(skipped_float),
        "post_settle_graph_break_count": int(graph_float),
        "post_settle_recompile_count": int(recompile_float),
        "initial_graph_break_count": initial_graph_breaks,
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
            7,
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
            2,
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
            2,
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
            7,
        ),
    )


def _time_block(
    module: nn.Module,
    inputs: torch.Tensor,
    distributed: _Distributed,
    *,
    compile_module: bool,
) -> _TimingResult:
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-4, fused=True)
    scaler = GradScaler("cuda", init_scale=GRAD_SCALER_INIT_SCALE)
    ddp = _wrap_selected_ddp(module, distributed)

    def forward_loss(values: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.float16, cache_enabled=True):
            output = cast("torch.Tensor", ddp(values))
        return output.float().square().mean()

    callable_step = (
        cast(
            "Callable[[torch.Tensor], torch.Tensor]",
            torch.compile(  # pyright: ignore[reportUnknownMemberType]
                forward_loss,
                backend="inductor",
                dynamic=False,
            ),
        )
        if compile_module
        else forward_loss
    )

    def step() -> tuple[torch.Tensor, torch.Tensor]:
        optimizer.zero_grad(set_to_none=True)
        with compiled_autograd_context(enabled=compile_module):
            loss = callable_step(inputs)
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
            module.parameters(),
            max_norm=MAX_GRAD_NORM,
            foreach=True,
        )
        scaler.step(optimizer)
        scaler.update()
        return loss.detach(), finite_gradients

    for _ in range(WARMUP_UPDATES):
        step()
    samples: list[float] = []
    skips = 0
    nonfinite_losses = 0
    nonfinite_gradients = 0
    for _ in range(TIMED_UPDATES):
        previous_scale = scaler.get_scale()
        torch.cuda.synchronize(inputs.device)
        start = time.perf_counter()
        loss, finite_gradients = step()
        torch.cuda.synchronize(inputs.device)
        samples.append((time.perf_counter() - start) * MILLISECONDS_PER_SECOND)
        skips += int(scaler.get_scale() < previous_scale)
        nonfinite_losses += int(not bool(torch.isfinite(loss).item()))
        nonfinite_gradients += int(not bool(finite_gradients.item()))
    median = statistics.median(samples)
    coefficient_variation = statistics.pstdev(samples) / statistics.mean(samples)
    return _TimingResult(
        median,
        coefficient_variation,
        tuple(samples),
        skips,
        nonfinite_losses,
        nonfinite_gradients,
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


def _measure_blocks(
    distributed: _Distributed,
) -> tuple[list[JsonObject], float]:
    rows: list[JsonObject] = []
    weighted_equivariant = 0.0
    weighted_normal = 0.0
    for case in _block_cases():
        generator = torch.Generator(device=distributed.device)
        generator.manual_seed(430013 + distributed.rank)
        inputs = torch.randn(
            case.shape,
            generator=generator,
            device=distributed.device,
        ).contiguous(memory_format=torch.channels_last)
        accuracy = _accuracy_case(case, distributed.device)
        eager_module = _to_device(copy.deepcopy(case.equivariant), distributed.device)
        compiled_module = _to_device(
            copy.deepcopy(case.equivariant),
            distributed.device,
        )
        normal_module = _to_device(case.normal, distributed.device)
        eager = _time_block(
            eager_module,
            inputs,
            distributed,
            compile_module=False,
        )
        compiled = _time_block(
            compiled_module,
            inputs,
            distributed,
            compile_module=True,
        )
        normal = _time_block(
            normal_module,
            inputs,
            distributed,
            compile_module=True,
        )
        ratio = compiled.median_ms / normal.median_ms
        weighted_equivariant += case.topology_weight * compiled.median_ms
        weighted_normal += case.topology_weight * normal.median_ms
        rows.append(
            cast(
                "JsonObject",
                {
                    "name": case.name,
                    "shape": list(case.shape),
                    "topology_weight": case.topology_weight,
                    **accuracy,
                    "eager_fp16_step_ms_p50": eager.median_ms,
                    "compiled_fp16_step_ms_p50": compiled.median_ms,
                    "normal_compiled_step_ms_p50": normal.median_ms,
                    "compiled_over_eager": compiled.median_ms / eager.median_ms,
                    "equivariant_over_normal": ratio,
                    "eager_timing_cv": eager.coefficient_variation,
                    "compiled_timing_cv": compiled.coefficient_variation,
                    "normal_timing_cv": normal.coefficient_variation,
                    "eager_amp_skip_count": eager.amp_skip_count,
                    "compiled_amp_skip_count": compiled.amp_skip_count,
                    "normal_amp_skip_count": normal.amp_skip_count,
                    "eager_nonfinite_loss_count": eager.nonfinite_loss_count,
                    "compiled_nonfinite_loss_count": compiled.nonfinite_loss_count,
                    "normal_nonfinite_loss_count": normal.nonfinite_loss_count,
                    "eager_nonfinite_gradient_count": eager.nonfinite_gradient_count,
                    "compiled_nonfinite_gradient_count": (
                        compiled.nonfinite_gradient_count
                    ),
                    "normal_nonfinite_gradient_count": normal.nonfinite_gradient_count,
                },
            ),
        )
        del eager_module, compiled_module, normal_module, inputs
        torch_dynamo.reset()
        gc.collect()
        torch.cuda.empty_cache()
    return rows, weighted_equivariant / weighted_normal


def _gather_rank_measurements(
    rows: list[JsonObject],
    weighted_ratio: float,
    assembly_fraction: float,
    distributed: _Distributed,
) -> tuple[list[JsonObject], float, float]:
    local = cast(
        "JsonObject",
        {
            "rank": distributed.rank,
            "blocks": rows,
            "topology_weighted_equivariant_over_normal": weighted_ratio,
            "largest_D_to_D_assembly_fraction": assembly_fraction,
        },
    )
    gathered: list[object] = [None] * distributed.world_size
    cast("Callable[..., None]", dist.all_gather_object)(gathered, local)
    rank_results = [cast("JsonObject", item) for item in gathered]
    worst_weighted = max(
        float(cast("float", item["topology_weighted_equivariant_over_normal"]))
        for item in rank_results
    )
    worst_assembly = max(
        float(cast("float", item["largest_D_to_D_assembly_fraction"]))
        for item in rank_results
    )
    return rank_results, worst_weighted, worst_assembly


def _assembly_fraction(
    model: _MechanicsSuite,
    distributed: _Distributed,
) -> float:
    largest = model.largest
    inputs = _inputs(distributed)[-1]

    def expand() -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.float16, cache_enabled=True):
            return largest.expanded_kernel()

    def forward(values: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.float16, cache_enabled=True):
            return cast("torch.Tensor", largest(values))

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
    assembly: list[float] = []
    complete: list[float] = []
    for _ in range(TIMED_UPDATES):
        torch.cuda.synchronize(distributed.device)
        start = time.perf_counter()
        compiled_expand()
        torch.cuda.synchronize(distributed.device)
        assembly.append(time.perf_counter() - start)
        start = time.perf_counter()
        compiled_forward(inputs)
        torch.cuda.synchronize(distributed.device)
        complete.append(time.perf_counter() - start)
    return statistics.median(assembly) / statistics.median(complete)


def _compile_follow_up_arm(
    name: str,
    module: nn.Module,
) -> _CompiledArm:
    def expand() -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.float16, cache_enabled=True):
            expanded_kernel = cast(
                "Callable[[], torch.Tensor]",
                module.expanded_kernel,  # pyright: ignore[reportAttributeAccessIssue]
            )
            return expanded_kernel()

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
    return _CompiledArm(name, module, compiled_expand, compiled_forward)


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


def _prepare_follow_up_step(
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


def _timed_follow_up_step(
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


def _follow_up_step_controls(
    source: SO2LargestDDConv,
    distributed: _Distributed,
) -> JsonObject:
    inputs = _inputs(distributed)[-1]
    paths = (
        _prepare_follow_up_step(
            "equivariant_eager",
            _to_device(copy.deepcopy(source), distributed.device),
            inputs,
            distributed,
            compile_module=False,
        ),
        _prepare_follow_up_step(
            "equivariant_compiled",
            _to_device(copy.deepcopy(source), distributed.device),
            inputs,
            distributed,
            compile_module=True,
        ),
        _prepare_follow_up_step(
            "normal_compiled",
            _to_device(
                nn.Conv2d(
                    D_LAYOUT.channels,
                    D_LAYOUT.channels,
                    kernel_size=5,
                    padding=2,
                    bias=False,
                ),
                distributed.device,
            ),
            inputs,
            distributed,
            compile_module=True,
        ),
    )
    for _ in range(FOLLOW_UP_WARMUPS):
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
        for _ in range(FOLLOW_UP_WINDOW_UPDATES):
            for path in order:
                elapsed, skipped, finite_loss, finite_gradient = _timed_follow_up_step(
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
            "paths": rows,
            "compiled_over_eager": compiled_median / eager_median,
            "equivariant_over_normal": compiled_median / normal_median,
        },
    )


def _follow_up_arm_accuracy(  # noqa: PLR0914
    reference: SO2LargestDDConv,
    candidate: nn.Module,
    inputs: torch.Tensor,
) -> dict[str, object]:
    eager = copy.deepcopy(reference)
    compiled_source = copy.deepcopy(candidate)
    eager_inputs = inputs.detach().clone().requires_grad_()
    compiled_inputs = inputs.detach().clone().requires_grad_()
    eager_output = cast("torch.Tensor", eager(eager_inputs))
    eager_output.square().mean().backward()  # pyright: ignore[reportUnknownMemberType]
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
        worst_name,
        reference_rms,
        difference_rms,
        missing,
        nonfinite,
    ) = _coefficient_gradient_comparison(eager, compiled_source)
    eager_kernel = eager.expanded_kernel()
    candidate_kernel = cast("torch.Tensor", candidate.expanded_kernel())  # type: ignore[attr-defined]
    return {
        "fp32_kernel_exact": torch.equal(candidate_kernel, eager_kernel),
        "fp32_kernel_relative_rms": _relative_rms(candidate_kernel, eager_kernel),
        "output_relative_rms": _relative_rms(compiled_output.float(), eager_output),
        "max_coefficient_gradient_relative_rms": gradient_error,
        "worst_coefficient_gradient_name": worst_name,
        "worst_coefficient_gradient_reference_rms": reference_rms,
        "worst_coefficient_gradient_difference_rms": difference_rms,
        "missing_coefficient_gradients": missing,
        "nonfinite_coefficient_gradient_count": nonfinite,
    }


def _candidate_runtime_evidence(
    name: str,
    module: nn.Module,
    inputs: torch.Tensor,
    distributed: _Distributed,
) -> JsonObject:
    buffers = _check_buffers_across_ranks(module, distributed)
    torch_dynamo.reset()
    counters.clear()
    arm = _compile_follow_up_arm(name, module)
    for _ in range(FOLLOW_UP_WARMUPS):
        arm.expand()
        arm.forward(inputs)
    initial_graph_breaks = _graph_break_total()
    graph_before = _graph_break_total()
    unique_before = _unique_graph_total()
    torch.cuda.reset_peak_memory_stats(distributed.device)
    for _ in range(FOLLOW_UP_WINDOW_UPDATES):
        arm.expand()
        arm.forward(inputs)
    torch.cuda.synchronize(distributed.device)
    return cast(
        "JsonObject",
        {
            "buffers": buffers,
            "initial_graph_break_count": initial_graph_breaks,
            "post_settle_graph_break_count": _graph_break_total() - graph_before,
            "post_settle_recompile_count": _unique_graph_total() - unique_before,
            "peak_allocated_mib": (
                float(torch.cuda.max_memory_allocated(distributed.device))
                / BYTES_PER_MIB
            ),
            "peak_reserved_mib": (
                float(torch.cuda.max_memory_reserved(distributed.device))
                / BYTES_PER_MIB
            ),
        },
    )


def _follow_up_modules(
    source: SO2LargestDDConv,
    device: torch.device,
) -> tuple[tuple[str, nn.Module], ...]:
    return (
        ("four_mm_three_cat", _to_device(copy.deepcopy(source), device)),
        ("four_mm_direct", _to_device(_DDirectAssemblyConv(source), device)),
        ("padded_bmm_direct", _to_device(_DPaddedBmmAssemblyConv(source), device)),
    )


def _follow_up_assembly_arms(
    source: SO2LargestDDConv,
    distributed: _Distributed,
) -> list[JsonObject]:
    inputs = _inputs(distributed)[-1]
    runtime_evidence: dict[str, JsonObject] = {}
    for name, module in _follow_up_modules(source, distributed.device):
        runtime_evidence[name] = _candidate_runtime_evidence(
            name,
            module,
            inputs,
            distributed,
        )
        del module
        torch_dynamo.reset()
        gc.collect()
        torch.cuda.empty_cache()
    modules = _follow_up_modules(source, distributed.device)
    source_copy = cast("SO2LargestDDConv", modules[0][1])
    accuracy = {
        name: _follow_up_arm_accuracy(source_copy, module, inputs)
        for name, module in modules
    }
    arms = tuple(starmap(_compile_follow_up_arm, modules))
    for arm in arms:
        for _ in range(FOLLOW_UP_WARMUPS):
            arm.expand()
            arm.forward(inputs)
    windows: dict[str, dict[str, list[list[float]]]] = {
        arm.name: {"expansion": [], "complete": []} for arm in arms
    }
    for order in (arms, tuple(reversed(arms))):
        expansion: dict[str, list[float]] = {arm.name: [] for arm in arms}
        complete: dict[str, list[float]] = {arm.name: [] for arm in arms}
        for _ in range(FOLLOW_UP_WINDOW_UPDATES):
            for arm in order:
                expansion[arm.name].append(
                    _timed_cuda_call(arm.expand, distributed.device),
                )
                complete[arm.name].append(
                    _timed_cuda_call(arm.forward, distributed.device, inputs),
                )
        for arm in arms:
            windows[arm.name]["expansion"].append(expansion[arm.name])
            windows[arm.name]["complete"].append(complete[arm.name])
    rows: list[JsonObject] = []
    for arm in arms:
        expansion_windows = windows[arm.name]["expansion"]
        complete_windows = windows[arm.name]["complete"]
        expansion_pooled = expansion_windows[0] + expansion_windows[1]
        complete_pooled = complete_windows[0] + complete_windows[1]
        window_summaries = [
            {
                "expansion": _window_summary(expansion_window),
                "complete": _window_summary(complete_window),
                "assembly_fraction": (
                    statistics.median(expansion_window)
                    / statistics.median(complete_window)
                ),
            }
            for expansion_window, complete_window in zip(
                expansion_windows,
                complete_windows,
                strict=True,
            )
        ]
        rows.append(
            cast(
                "JsonObject",
                {
                    "name": arm.name,
                    "runtime": runtime_evidence[arm.name],
                    "accuracy": accuracy[arm.name],
                    "windows": window_summaries,
                    "pooled_expansion": _window_summary(expansion_pooled),
                    "pooled_complete": _window_summary(complete_pooled),
                    "assembly_fraction": (
                        statistics.median(expansion_pooled)
                        / statistics.median(complete_pooled)
                    ),
                },
            ),
        )
    return rows


def _verdict(
    updates: dict[str, object],
    rank_results: Sequence[JsonObject],
    *,
    assembly_fraction: float,
    weighted_ratio: float,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
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
        (assembly_fraction <= ASSEMBLY_FRACTION_LIMIT, "assembly_fraction"),
        (weighted_ratio <= NORMAL_RATIO_LIMIT, "weighted_normal_ratio"),
    )
    failures.extend(name for passed, name in scalar_checks if not passed)
    for rank_result in rank_results:
        rank = cast("int", rank_result["rank"])
        for row in cast("list[JsonObject]", rank_result["blocks"]):
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
                (
                    float(cast("float", row["compiled_timing_cv"])) <= TIMING_CV_LIMIT,
                    "timing_cv",
                ),
                (
                    float(cast("float", row["eager_timing_cv"])) <= TIMING_CV_LIMIT,
                    "eager_timing_cv",
                ),
                (
                    float(cast("float", row["normal_timing_cv"])) <= TIMING_CV_LIMIT,
                    "normal_timing_cv",
                ),
                (
                    sum(
                        int(cast("int", row[key]))
                        for key in (
                            "eager_amp_skip_count",
                            "compiled_amp_skip_count",
                            "normal_amp_skip_count",
                        )
                    )
                    == 0,
                    "timed_amp_skip",
                ),
                (
                    sum(
                        int(cast("int", row[key]))
                        for key in (
                            "eager_nonfinite_loss_count",
                            "compiled_nonfinite_loss_count",
                            "normal_nonfinite_loss_count",
                            "eager_nonfinite_gradient_count",
                            "compiled_nonfinite_gradient_count",
                            "normal_nonfinite_gradient_count",
                        )
                    )
                    == 0,
                    "timed_nonfinite",
                ),
            )
            failures.extend(
                f"rank{rank}:{name}:{metric}" for passed, metric in checks if not passed
            )
    return not failures, failures


def _follow_up_verdict(  # noqa: C901, PLR0912, PLR0914, PLR0915
    updates: dict[str, object],
    rank_results: Sequence[JsonObject],
) -> tuple[str | None, list[str], dict[str, list[str]]]:
    failures: list[str] = []
    update_checks = (
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
            int(cast("int", updates["post_settle_recompile_count"])) == 0,
            "recompiles",
        ),
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
    )
    failures.extend(name for passed, name in update_checks if not passed)
    passing_arms: list[tuple[float, str]] = []
    rejected_arms: dict[str, list[str]] = {}
    arm_names = ("four_mm_three_cat", "four_mm_direct", "padded_bmm_direct")
    for rank_result in rank_results:
        rank = int(cast("int", rank_result["rank"]))
        controls = cast("JsonObject", rank_result["corrected_step_controls"])
        if (
            float(cast("float", controls["compiled_over_eager"]))
            > COMPILED_EAGER_RATIO_LIMIT
        ):
            failures.append(f"rank{rank}:largest_D_to_D:compiled_eager")
        if (
            float(cast("float", controls["equivariant_over_normal"]))
            > NORMAL_RATIO_LIMIT
        ):
            failures.append(f"rank{rank}:largest_D_to_D:normal_ratio")
        for path_name, raw_path in cast(
            "dict[str, object]",
            controls["paths"],
        ).items():
            path = cast("JsonObject", raw_path)
            if sum(
                int(cast("int", path[key]))
                for key in (
                    "amp_skip_count",
                    "nonfinite_loss_count",
                    "nonfinite_gradient_count",
                )
            ):
                failures.append(f"rank{rank}:largest_D_to_D:{path_name}:invalid_step")
            summaries = [
                *cast("list[JsonObject]", path["windows"]),
                cast("JsonObject", path["pooled"]),
            ]
            for summary_index, summary in enumerate(summaries):
                if (
                    float(cast("float", summary["coefficient_variation"]))
                    > TIMING_CV_LIMIT
                ):
                    failures.append(
                        f"rank{rank}:largest_D_to_D:{path_name}:cv{summary_index}",
                    )
        for accuracy in cast("list[JsonObject]", rank_result["corrected_accuracy"]):
            name = cast("str", accuracy["name"])
            if (
                float(cast("float", accuracy["output_relative_rms"]))
                > OUTPUT_RELATIVE_LIMIT
            ):
                failures.append(f"rank{rank}:{name}:output")
            if (
                float(
                    cast(
                        "float",
                        accuracy["max_coefficient_gradient_relative_rms"],
                    ),
                )
                > GRADIENT_RELATIVE_LIMIT
            ):
                failures.append(f"rank{rank}:{name}:gradient")
            if cast("list[object]", accuracy["missing_coefficient_gradients"]):
                failures.append(f"rank{rank}:{name}:missing_gradient")
            if (
                int(
                    cast(
                        "int",
                        accuracy["nonfinite_coefficient_gradient_count"],
                    ),
                )
                != 0
            ):
                failures.append(f"rank{rank}:{name}:nonfinite_gradient")
    for arm_name in arm_names:
        arm_failures: list[str] = []
        worst_median = 0.0
        for rank_result in rank_results:
            rank = int(cast("int", rank_result["rank"]))
            arm = next(
                row
                for row in cast("list[JsonObject]", rank_result["arms"])
                if row["name"] == arm_name
            )
            accuracy = cast("JsonObject", arm["accuracy"])
            runtime = cast("JsonObject", arm["runtime"])
            checks = (
                (
                    int(cast("int", runtime["initial_graph_break_count"])) == 0,
                    "initial_graph_breaks",
                ),
                (
                    int(cast("int", runtime["post_settle_graph_break_count"])) == 0,
                    "graph_breaks",
                ),
                (
                    int(cast("int", runtime["post_settle_recompile_count"])) == 0,
                    "recompiles",
                ),
                (
                    float(cast("float", runtime["peak_allocated_mib"]))
                    < PEAK_ALLOCATED_MIB_LIMIT,
                    "allocated_vram",
                ),
                (
                    float(cast("float", runtime["peak_reserved_mib"]))
                    < PEAK_RESERVED_MIB_LIMIT,
                    "reserved_vram",
                ),
                (
                    float(cast("float", accuracy["fp32_kernel_relative_rms"]))
                    <= FP32_CANDIDATE_KERNEL_LIMIT,
                    "fp32_kernel",
                ),
                (
                    float(cast("float", accuracy["output_relative_rms"]))
                    <= OUTPUT_RELATIVE_LIMIT,
                    "output",
                ),
                (
                    float(
                        cast(
                            "float",
                            accuracy["max_coefficient_gradient_relative_rms"],
                        ),
                    )
                    <= GRADIENT_RELATIVE_LIMIT,
                    "gradient",
                ),
                (
                    cast("list[object]", accuracy["missing_coefficient_gradients"])
                    == [],
                    "missing_gradient",
                ),
                (
                    int(
                        cast(
                            "int",
                            accuracy["nonfinite_coefficient_gradient_count"],
                        ),
                    )
                    == 0,
                    "nonfinite_gradient",
                ),
                (
                    float(cast("float", arm["assembly_fraction"]))
                    <= ASSEMBLY_FRACTION_LIMIT,
                    "assembly_fraction",
                ),
            )
            arm_failures.extend(
                f"rank{rank}:{arm_name}:{metric}"
                for passed, metric in checks
                if not passed
            )
            for window_index, window in enumerate(
                cast("list[JsonObject]", arm["windows"]),
            ):
                if (
                    float(cast("float", window["assembly_fraction"]))
                    > ASSEMBLY_FRACTION_LIMIT
                ):
                    arm_failures.append(
                        f"rank{rank}:{arm_name}:window{window_index}:assembly_fraction",
                    )
                for path in ("expansion", "complete"):
                    summary = cast("JsonObject", window[path])
                    if (
                        float(cast("float", summary["coefficient_variation"]))
                        > TIMING_CV_LIMIT
                    ):
                        arm_failures.append(
                            f"rank{rank}:{arm_name}:window{window_index}:{path}_cv",
                        )
            for path in ("pooled_expansion", "pooled_complete"):
                summary = cast("JsonObject", arm[path])
                if (
                    float(cast("float", summary["coefficient_variation"]))
                    > TIMING_CV_LIMIT
                ):
                    arm_failures.append(f"rank{rank}:{arm_name}:{path}_cv")
            pooled_complete = cast("JsonObject", arm["pooled_complete"])
            worst_median = max(
                worst_median,
                float(cast("float", pooled_complete["median_ms"])),
            )
        if not arm_failures:
            passing_arms.append((worst_median, arm_name))
        else:
            rejected_arms[arm_name] = arm_failures
    provisional_selection = min(passing_arms)[1] if passing_arms else None
    if provisional_selection is None:
        failures.extend(
            failure
            for arm_failures in rejected_arms.values()
            for failure in arm_failures
        )
        failures.append("no_follow_up_arm_passed")
    selected = provisional_selection if not failures else None
    return selected, failures, rejected_arms


def _gather_follow_up(
    corrected_accuracy: list[JsonObject],
    corrected_step_controls: JsonObject,
    arms: list[JsonObject],
    distributed: _Distributed,
) -> list[JsonObject]:
    local = cast(
        "JsonObject",
        {
            "rank": distributed.rank,
            "corrected_accuracy": corrected_accuracy,
            "corrected_step_controls": corrected_step_controls,
            "arms": arms,
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
    requested_effective_runtime = _apply_selected_runtime()
    gradient_mean = _gradient_mean_check(distributed)
    updates, trained_suite = _compiled_ddp_updates(distributed)
    largest = copy.deepcopy(trained_suite.largest)
    del trained_suite
    gc.collect()
    torch.cuda.empty_cache()
    corrected_accuracy = [
        cast(
            "JsonObject",
            {"name": case.name, **_accuracy_case(case, distributed.device)},
        )
        for case in _block_cases()
    ]
    corrected_step_controls = _follow_up_step_controls(
        largest,
        distributed,
    )
    arms = _follow_up_assembly_arms(
        largest,
        distributed,
    )
    rank_results = _gather_follow_up(
        corrected_accuracy,
        corrected_step_controls,
        arms,
        distributed,
    )
    selected_arm, failures, rejected_arms = _follow_up_verdict(updates, rank_results)
    passed = selected_arm is not None and not failures
    result = cast(
        "JsonObject",
        {
            "schema_version": SCHEMA_VERSION,
            "benchmark_kind": PROBE_KIND,
            "status": "pass" if passed else "fail",
            "architecture_locked": True,
            "full_vae_assembled": False,
            "follow_up_probe_permitted": False,
            "world_size": distributed.world_size,
            "nproc_per_node": distributed.nproc_per_node,
            "gpu_names": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
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
            "selected_arm": selected_arm,
            "rejected_arms": rejected_arms,
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
