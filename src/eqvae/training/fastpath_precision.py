# Copyright 2026 HiperMaximus
"""Shared AMP mechanics for measured and executed compiled fast paths."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from torch import nn
from torch.amp import GradScaler

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

FastpathUpdateProbe = tuple[nn.Parameter, torch.Tensor] | None
FASTPATH_METRIC_COUNT = 16


@dataclass(frozen=True)
class FastpathOptimizerStepResult:
    """Hot-loop optimizer outcome retained without host materialization."""

    step_skipped: bool
    grad_norm: torch.Tensor
    nonfinite_count: torch.Tensor


EXPECTED_RUNNER_AMP_GRAD_SCALER_INIT_SCALE = 16384.0


def fastpath_autocast_dtype(name: str, *, amp_enabled: bool) -> torch.dtype:
    """Resolve one runtime row's autocast dtype without version fallbacks.

    Args:
        name: Serialized runtime dtype name.
        amp_enabled: Whether the row enables autocast.

    Returns:
        The torch dtype consumed by the compiled-step closure.

    Raises:
        ValueError: If an AMP row names an unsupported dtype.

    """
    if not amp_enabled:
        return torch.float32
    if name in {"", "float16", "fp16"}:
        return torch.float16
    if name in {"bfloat16", "bf16"}:
        return torch.bfloat16
    message = f"Unsupported AMP autocast dtype: {name!r}"
    raise ValueError(message)


def build_fastpath_grad_scaler(*, enabled: bool) -> GradScaler:
    """Build the canonical scaler shared by probe, measurement, and runner.

    Returns:
        A CUDA GradScaler using the selected-runtime contract's initial scale.

    """
    return GradScaler(
        "cuda",
        init_scale=EXPECTED_RUNNER_AMP_GRAD_SCALER_INIT_SCALE,
        enabled=enabled,
    )


def clone_fastpath_update_probe(model: nn.Module) -> FastpathUpdateProbe:
    """Clone one smallest trainable tensor as a low-cost update sentinel.

    Returns:
        The sampled parameter and its pre-update value, or ``None``.

    """
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable:
        return None
    parameter = min(trainable, key=lambda value: value.numel())
    return parameter, parameter.detach().clone()


def fastpath_update_probe_norm(probe: FastpathUpdateProbe) -> torch.Tensor:
    """Return the sampled trainable tensor's on-device update norm.

    Returns:
        A scalar tensor that remains on the parameter device.

    """
    if probe is None:
        return torch.zeros((), dtype=torch.float64)
    parameter, before = probe
    delta = (parameter.detach().float() - before.float()).double()
    return delta.square().sum().sqrt()


def transfer_fastpath_uint8(
    tensor: torch.Tensor,
    *,
    device: torch.device,
    memory_format: str,
    non_blocking: bool,
) -> torch.Tensor:
    """Execute the runner/probe's single fused-layout uint8 H2D operation.

    Returns:
        The device-resident uint8 tensor in the selected layout.

    """
    target_format = (
        torch.channels_last
        if memory_format == "channels_last"
        else torch.preserve_format
    )
    return tensor.to(
        device=device,
        non_blocking=non_blocking and device.type == "cuda",
        memory_format=target_format,
    )


def fastpath_eps_metrics(eps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the two on-device epsilon telemetry reductions paid by training.

    Returns:
        Zero fraction and absolute mean, retained on device.

    """
    values = eps.detach().float()
    zero_fraction = torch.count_nonzero(values == 0).float() / values.numel()
    return zero_fraction, values.abs().mean()


def write_fastpath_metric_row(
    row: torch.Tensor,
    scalars: tuple[torch.Tensor, ...],
) -> None:
    """Write one complete runner/probe telemetry row without host materialization."""
    for column, scalar in enumerate(scalars):
        row[column] = scalar.detach()


def run_fastpath_optimizer_step(  # noqa: PLR0913
    *,
    loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    parameters: Iterable[nn.Parameter],
    scaler: GradScaler,
    grad_scaler_enabled: bool,
    gradient_clip_global_norm: float,
    gradient_clip_foreach: bool,
    backward_context: AbstractContextManager[None] | None = None,
    observe_skip: bool = True,
) -> bool:
    """Backpropagate, clip, and update with runner-identical AMP ordering.

    Args:
        loss: Scalar loss whose graph is backpropagated.
        optimizer: Optimizer updated by the step.
        parameters: Parameters included in global-norm clipping.
        scaler: The row's persistent GradScaler.
        grad_scaler_enabled: Whether scaling is active for this row.
        gradient_clip_global_norm: Maximum global norm; non-positive disables clipping.
        gradient_clip_foreach: Whether the foreach clip implementation is requested.
        backward_context: Optional compiled-autograd context around backward only.
        observe_skip: Whether to materialize the scaler state on the host. Disable
            inside timed/hot loops and compare scale once around the whole block.

    Returns:
        Whether GradScaler skipped the optimizer update after detecting non-finite
        gradients.

    """
    return run_fastpath_optimizer_step_with_metrics(
        loss=loss,
        optimizer=optimizer,
        parameters=parameters,
        scaler=scaler,
        grad_scaler_enabled=grad_scaler_enabled,
        gradient_clip_global_norm=gradient_clip_global_norm,
        gradient_clip_foreach=gradient_clip_foreach,
        backward_context=backward_context,
        observe_skip=observe_skip,
    ).step_skipped


def run_fastpath_optimizer_step_with_metrics(  # noqa: PLR0913
    *,
    loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    parameters: Iterable[nn.Parameter],
    scaler: GradScaler,
    grad_scaler_enabled: bool,
    gradient_clip_global_norm: float,
    gradient_clip_foreach: bool,
    backward_context: AbstractContextManager[None] | None = None,
    observe_skip: bool = True,
) -> FastpathOptimizerStepResult:
    """Run the shared optimizer body and reuse clipping's on-device norm telemetry.

    Returns:
        Skip, gradient-norm, and finite-state evidence without a hot-loop host read.

    """
    parameter_list = list(parameters)
    with backward_context or nullcontext():
        if grad_scaler_enabled:
            old_scale = float(scaler.get_scale()) if observe_skip else 0.0
            scaled_backward = cast("Callable[[], None]", scaler.scale(loss).backward)
            scaled_backward()
        else:
            old_scale = 1.0
            backward = cast("Callable[[], None]", loss.backward)
            backward()
    if grad_scaler_enabled:
        scaler.unscale_(optimizer)
    if gradient_clip_global_norm > 0.0:
        observed_norm = torch.nn.utils.clip_grad_norm_(
            parameter_list,
            gradient_clip_global_norm,
            foreach=gradient_clip_foreach,
        )
        grad_norm = observed_norm
    else:
        gradients = [
            parameter.grad.detach().float()
            for parameter in parameter_list
            if parameter.grad is not None
        ]
        grad_norm = (
            torch
            .stack([gradient.square().sum() for gradient in gradients])
            .double()
            .sum()
            .sqrt()
            if gradients
            else torch.zeros((), dtype=torch.float64, device=loss.device)
        )
    nonfinite_count = (~torch.isfinite(grad_norm)).to(dtype=torch.int64)
    if grad_scaler_enabled:
        scaler.step(optimizer)
        scaler.update()
        skipped = observe_skip and float(scaler.get_scale()) < old_scale
    else:
        optimizer.step()
        skipped = False
    return FastpathOptimizerStepResult(
        step_skipped=skipped,
        grad_norm=grad_norm,
        nonfinite_count=nonfinite_count,
    )


__all__ = [
    "EXPECTED_RUNNER_AMP_GRAD_SCALER_INIT_SCALE",
    "FASTPATH_METRIC_COUNT",
    "FastpathOptimizerStepResult",
    "build_fastpath_grad_scaler",
    "clone_fastpath_update_probe",
    "fastpath_autocast_dtype",
    "fastpath_eps_metrics",
    "fastpath_update_probe_norm",
    "run_fastpath_optimizer_step",
    "run_fastpath_optimizer_step_with_metrics",
    "transfer_fastpath_uint8",
    "write_fastpath_metric_row",
]
