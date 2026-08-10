# Copyright 2026 HiperMaximus
"""Shared compiled fast-path recipe application (Spec 0011 S10).

Single source for the three recipe components that must be applied identically by
the efficiency probe that *measures* a compiled fast-path recipe
(``benchmarking.compiled_fastpath_probe``) and the training runner that later
*consumes* it (``training.selected_runtime_runner``, wired in Spec 0011 S15):

1. the grouped, optionally fused AdamW optimizer,
2. the ``DistributedDataParallel`` wrap, and
3. the process-global dynamo/inductor configuration.

Both callers routing through these helpers is what makes the measured recipe and
the consumed recipe bit-identical -- a measured throughput/VRAM number only
transfers to the real run if the run builds the optimizer, wraps DDP, and sets
the dynamo knobs exactly as the probe did. The helpers take plain scalar knobs
rather than a probe- or runner-specific spec object so either caller can drive
them from its own recipe carrier without depending on the other's types.
"""

from __future__ import annotations

import contextlib
import importlib
import json
from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
import torch._dynamo as torch_dynamo  # noqa: PLC2701
from torch._dynamo import compiled_autograd  # noqa: PLC2701
from torch._inductor import config as inductor_config  # noqa: PLC2701
from torch.nn.parallel import DistributedDataParallel

from eqvae.training.optim import create_adamw_optimizer

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import nn

    from eqvae.training.optim import SpecAdamWConfig


def build_fastpath_optimizer(
    model: nn.Module,
    *,
    config: SpecAdamWConfig | None = None,
) -> torch.optim.AdamW:
    """Build the grouped, optionally fused AdamW for the fast-path recipe.

    Routes through :func:`eqvae.training.optim.create_adamw_optimizer` so the
    fused kernel (CUDA-gated inside that helper) is applied to the *same*
    spec-0001 semantic parameter groups (decay / no-decay / gate-no-decay) as the
    eager path -- never a flat ungrouped ``model.parameters()`` set that would
    weight-decay norms and biases and drop the gate learning-rate multiplier. The
    coverage summary is not part of the recipe seam, so it is discarded here.

    Args:
        model: The model whose parameters are optimized.
        config: AdamW configuration; ``None`` uses the spec-0001 defaults.

    Returns:
        The grouped AdamW optimizer.

    """
    optimizer, _ = create_adamw_optimizer(model, config=config)
    return optimizer


def resolve_fastpath_compile_invocation(
    *,
    compile_mode: str,
    cudagraphs: str,
    inductor_options_json: str,
    mode_options: Mapping[str, object] | None = None,
) -> tuple[str | None, dict[str, object] | None]:
    """Resolve one legal and artifact-replayable torch.compile mode/options pair.

    Returns:
        Mutually exclusive ``mode`` and ``options`` keyword values.

    Raises:
        TypeError: If JSON/options or the named installed preset are malformed.

    """
    try:
        decoded = cast("object", json.loads(inductor_options_json))
    except json.JSONDecodeError as error:
        message = "Inductor option bundle is malformed"
        raise TypeError(message) from error
    if not isinstance(decoded, dict):
        message = "Inductor option bundle must be a JSON object"
        raise TypeError(message)
    custom_options = cast("dict[str, object]", decoded)
    if cudagraphs == "mode_default" and not custom_options:
        return (None if compile_mode == "default" else compile_mode), None
    if mode_options is None:
        inductor = importlib.import_module("torch._inductor")
        list_modes = getattr(inductor, "list_mode_options", None)
        raw_mode_object: object = (
            cast("Callable[[], object]", list_modes)() if callable(list_modes) else {}
        )
    else:
        raw_mode_object = mode_options
    if not isinstance(raw_mode_object, Mapping):
        message = "installed compile modes did not resolve to a mapping"
        raise TypeError(message)
    raw_modes = cast("Mapping[object, object]", raw_mode_object)
    preset = raw_modes.get(compile_mode, {})
    if not isinstance(preset, Mapping):
        message = f"installed compile mode has no option mapping: {compile_mode}"
        raise TypeError(message)
    options = {
        str(key): value
        for key, value in cast("Mapping[object, object]", preset).items()
    }
    options.update(custom_options)
    if cudagraphs != "mode_default":
        options["triton.cudagraphs"] = cudagraphs == "enabled"
    return None, options


def register_fastpath_communication_hook(model: nn.Module, hook_name: str) -> None:
    """Register one selected installed DDP compression hook.

    Raises:
        TypeError: If the selected hook is unavailable in the executing runtime.

    """
    if hook_name == "none":
        return
    hooks = importlib.import_module(
        "torch.distributed.algorithms.ddp_comm_hooks.default_hooks",
    )
    hook = getattr(hooks, hook_name, None)
    register = getattr(model, "register_comm_hook", None)
    if not callable(hook) or not callable(register):
        message = f"communication hook is unavailable: {hook_name}"
        raise TypeError(message)
    register(state=None, hook=hook)


def apply_fastpath_dynamo_config(
    *,
    optimize_ddp: bool | str,
    compiled_autograd: bool,
    reorder_compute_comm_overlap: bool,
) -> None:
    """Apply the process-global dynamo/inductor knobs for one recipe.

    These settings are process-global, so the caller is responsible for ordering
    (and, for the probe, ``torch._dynamo.reset()``) between recipes.

    Args:
        optimize_ddp: ``torch._dynamo.config.optimize_ddp`` (bool or mode string).
        compiled_autograd: ``torch._dynamo.config.compiled_autograd``.
        reorder_compute_comm_overlap:
            ``torch._inductor.config.reorder_for_compute_comm_overlap``.

    """
    torch_dynamo.config.optimize_ddp = optimize_ddp
    torch_dynamo.config.compiled_autograd = compiled_autograd
    inductor_config.reorder_for_compute_comm_overlap = reorder_compute_comm_overlap


def apply_cudnn_flags(*, benchmark: bool, deterministic: bool) -> None:
    """Set the process-global cuDNN autotuning/determinism backend flags.

    ``cudnn.benchmark=True`` lets cuDNN autotune the fastest convolution algorithm
    for the (fixed, ``drop_last``-guaranteed) input shapes, and
    ``cudnn.deterministic=False`` permits the fastest non-deterministic kernels --
    the speed-first default (Spec 0011 S17f), matching the FSQ reference and the
    compiled probe. Exact reproducibility is an explicit non-goal, so these are
    fixed speed-first flags rather than a searched axis. Setting them is a plain
    attribute write that is harmless on a CPU-only build (the ``torch.backends.cudnn``
    module exists whether or not a GPU is present), so the caller device-gates the
    invocation, not this helper.

    Shared by the runner that *consumes* the runtime
    (``selected_runtime_runner``) and the single-GPU pre-screen that *measures* it
    (``real_data_runtime_pretest``) so both drive the same backend flags from one
    source; the dual-T4 executor keeps its richer ``_apply_backend_policy`` (which
    also captures/restores state and handles TF32/matmul precision) and only shares
    the speed-first default value.

    Args:
        benchmark: ``torch.backends.cudnn.benchmark``.
        deterministic: ``torch.backends.cudnn.deterministic``.

    """
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.deterministic = deterministic


def compiled_autograd_context(*, enabled: bool) -> AbstractContextManager[None]:
    """Return the eager-backward context that engages compiled autograd for one recipe.

    Compiled autograd traces the eager backward (so the DDP python-reducer all_reduce
    folds into a compiled backward graph for compute/comm overlap). The process-global
    ``torch._dynamo.config.compiled_autograd`` flag only covers the compiled forward
    call; the backward runs after the compiled step returns, so it needs this explicit
    context. ``enabled=False`` returns a no-op so a recipe that does not use compiled
    autograd (the measured ``ddp_optimizer`` winner pairs it with
    ``compiled_autograd=False``) pays nothing.

    Args:
        enabled: Whether to engage compiled autograd around the backward.

    Returns:
        A context manager to wrap the eager backward.

    """
    if not enabled:
        return contextlib.nullcontext()
    compiler = cast("Callable[..., object]", torch.compile)
    return cast(
        "AbstractContextManager[None]",
        compiled_autograd._enable(compiler),  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    )


@dataclass(frozen=True)
class FastpathDynamoKnobs:
    """Dynamo knobs that MUST be applied before ``DistributedDataParallel`` exists."""

    optimize_ddp: bool | str
    compiled_autograd: bool
    reorder_compute_comm_overlap: bool


def wrap_fastpath_ddp(  # noqa: PLR0913
    model: nn.Module,
    *,
    local_rank: int,
    static_graph: bool,
    gradient_as_bucket_view: bool,
    broadcast_buffers: bool,
    find_unused_parameters: bool,
    bucket_cap_mb: int | None,
    dynamo: FastpathDynamoKnobs | None,
    forward_sync_buffers: bool | None = None,
) -> DistributedDataParallel:
    """Wrap a model in ``DistributedDataParallel`` with the recipe's collective knobs.

    ``dynamo`` is applied HERE, immediately before construction, and is a REQUIRED
    argument so no caller can forget it. ``DistributedDataParallel.__init__`` latches
    the dynamo ``optimize_ddp`` mode at construction time (it sets
    ``_use_python_reducer`` from ``get_optimize_ddp_mode()`` there), so setting it after
    silently leaves DDP on its C++ reducer: compiled autograd has no Python reducer to
    trace, there is zero comm/compute overlap, and NOTHING raises. Owning the order in
    this one function makes that ordering structural instead of a convention each call
    site could quietly break (Spec 0011 S17f).

    Args:
        model: The (optionally compiled) model to wrap.
        local_rank: The device index for ``device_ids``/``output_device``.
        static_graph: DDP ``static_graph``.
        gradient_as_bucket_view: DDP ``gradient_as_bucket_view``.
        broadcast_buffers: DDP ``broadcast_buffers``.
        find_unused_parameters: DDP ``find_unused_parameters``.
        bucket_cap_mb: DDP ``bucket_cap_mb`` (``None`` uses the DDP default).
        dynamo: Knobs to apply before construction, or ``None`` for a run that compiles
            nothing (the eager path), where there is no dynamo state to establish.
        forward_sync_buffers: Current DDP forward-buffer synchronization control. When
            set, the deprecated ``broadcast_buffers`` argument is left unset.

    Returns:
        The DDP-wrapped model.

    """
    if dynamo is not None:
        apply_fastpath_dynamo_config(
            optimize_ddp=dynamo.optimize_ddp,
            compiled_autograd=dynamo.compiled_autograd,
            reorder_compute_comm_overlap=dynamo.reorder_compute_comm_overlap,
        )
    kwargs: dict[str, object] = {
        "device_ids": [local_rank],
        "output_device": local_rank,
        "static_graph": static_graph,
        "gradient_as_bucket_view": gradient_as_bucket_view,
        "find_unused_parameters": find_unused_parameters,
        "bucket_cap_mb": bucket_cap_mb,
    }
    if forward_sync_buffers is None:
        kwargs["broadcast_buffers"] = broadcast_buffers
    else:
        kwargs["forward_sync_buffers"] = forward_sync_buffers
    return DistributedDataParallel(model, **kwargs)  # pyright: ignore[reportArgumentType]


# Leaf names of the persistent buffers DDP would have to broadcast from rank 0 every
# forward to keep replicas identical: the running statistics that every standard
# PyTorch normalization (BatchNorm*/SyncBatchNorm/InstanceNorm* with
# track_running_stats) registers and mutates per batch, so they diverge across ranks
# unless broadcast.
_RANK_DIVERGENT_BUFFER_LEAVES = frozenset(
    {"running_mean", "running_var", "num_batches_tracked"},
)


def model_requires_buffer_broadcast(model: nn.Module) -> bool:
    """Return whether DDP must broadcast this model's buffers from rank 0.

    Shared by the runner that *consumes* the recipe
    (``training.selected_runtime_runner``) and the efficiency generator that
    *measures* it (``benchmarking.runtime_selection_executor``), so the DDP wrap is
    driven by the same ``broadcast_buffers`` value in both places.

    Disabling ``broadcast_buffers`` is only safe when every persistent buffer is a
    non-trainable, rank-identical constant that no forward pass mutates, so it can
    never diverge across ranks. The standard modules that violate this are the
    running-statistics normalizations, whose ``running_mean``/``running_var``/
    ``num_batches_tracked`` buffers accumulate per-rank batch statistics and would
    silently desync the replicas with ``broadcast_buffers=False``. This is a
    structural, model-agnostic rule rather than a flag hardcoded for one model: the
    non-equivariant baseline (GroupNorm plus the constant binomial downsample
    kernels) registers no such buffer and needs no broadcast, while a future model
    that introduces a running-stat buffer flips the result to ``True`` instead of
    training divergent replicas. Detection is by the standard torch running-stat
    buffer names, so a model that maintains rank-divergent state under a different
    buffer name must request ``ddp_broadcast_buffers=True`` in its plan (or extend
    this rule). ``named_buffers`` omits ``None``-valued buffers, so a normalization
    with ``track_running_stats=False`` correctly contributes nothing.

    Returns:
        ``True`` if any persistent buffer holds mutable running statistics; ``False``
        when every buffer is a rank-identical constant.

    """
    return any(
        name.rsplit(".", 1)[-1] in _RANK_DIVERGENT_BUFFER_LEAVES
        for name, _ in model.named_buffers()
    )


__all__ = [
    "FastpathDynamoKnobs",
    "apply_cudnn_flags",
    "apply_fastpath_dynamo_config",
    "build_fastpath_optimizer",
    "compiled_autograd_context",
    "model_requires_buffer_broadcast",
    "register_fastpath_communication_hook",
    "resolve_fastpath_compile_invocation",
    "wrap_fastpath_ddp",
]
