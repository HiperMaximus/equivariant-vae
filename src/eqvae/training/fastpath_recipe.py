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
from contextlib import AbstractContextManager
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


def wrap_fastpath_ddp(  # noqa: PLR0913
    model: nn.Module,
    *,
    local_rank: int,
    static_graph: bool,
    gradient_as_bucket_view: bool,
    broadcast_buffers: bool,
    find_unused_parameters: bool,
    bucket_cap_mb: int | None,
) -> DistributedDataParallel:
    """Wrap a model in ``DistributedDataParallel`` with the recipe's collective knobs.

    Args:
        model: The (optionally compiled) model to wrap.
        local_rank: The device index for ``device_ids``/``output_device``.
        static_graph: DDP ``static_graph``.
        gradient_as_bucket_view: DDP ``gradient_as_bucket_view``.
        broadcast_buffers: DDP ``broadcast_buffers``.
        find_unused_parameters: DDP ``find_unused_parameters``.
        bucket_cap_mb: DDP ``bucket_cap_mb`` (``None`` uses the DDP default).

    Returns:
        The DDP-wrapped model.

    """
    return DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        static_graph=static_graph,
        gradient_as_bucket_view=gradient_as_bucket_view,
        broadcast_buffers=broadcast_buffers,
        find_unused_parameters=find_unused_parameters,
        bucket_cap_mb=bucket_cap_mb,
    )


__all__ = [
    "apply_fastpath_dynamo_config",
    "build_fastpath_optimizer",
    "compiled_autograd_context",
    "wrap_fastpath_ddp",
]
