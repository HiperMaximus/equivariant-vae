# Copyright 2026 HiperMaximus
"""Single local train-step helper for the spec 0001 model/loss slice."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from eqvae.losses.vae import compute_vae_loss

if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch import nn

    from eqvae.losses.vae import VaeLossComponents
    from eqvae.models.non_equivariant_vae import NonEquivariantVAE, VaeForwardOutput


@dataclass(frozen=True)
class TrainStepRequest:
    """Inputs for one identity-clean local optimizer update."""

    model: NonEquivariantVAE
    optimizer: torch.optim.Optimizer
    clean_batch: torch.Tensor
    eps: torch.Tensor
    beta: float
    ssim_weight: float
    optimizer_step_index: int
    gradient_clip_global_norm: float


@dataclass(frozen=True)
class TrainStepResult:
    """Telemetry from one successful optimizer update."""

    forward: VaeForwardOutput
    losses: VaeLossComponents
    grad_norm: float
    param_update_norm: float
    nonfinite_count: int
    trainable_parameter_tensor_count: int
    nonzero_grad_parameter_tensor_count: int
    nonzero_update_parameter_tensor_count: int
    optimizer_step_index: int
    successful_optimizer_update_count: int


def run_train_step(request: TrainStepRequest) -> TrainStepResult:
    """Run one identity-clean local training step.

    Returns:
        Train-step telemetry from the successful update.

    Raises:
        ValueError: If step indexing or gradient clipping settings are invalid.

    """
    if request.optimizer_step_index < 0:
        message = (
            "optimizer_step_index must be nonnegative, got "
            f"{request.optimizer_step_index}"
        )
        raise ValueError(message)
    if request.gradient_clip_global_norm <= 0.0:
        message = (
            "gradient_clip_global_norm must be positive, got "
            f"{request.gradient_clip_global_norm}"
        )
        raise ValueError(message)

    request.optimizer.zero_grad(set_to_none=True)
    output: VaeForwardOutput = request.model.forward(
        request.clean_batch,
        eps=request.eps,
    )
    losses = compute_vae_loss(
        output,
        request.clean_batch,
        beta=request.beta,
        ssim_weight=request.ssim_weight,
    )
    torch.autograd.backward(losses.loss)
    nonfinite_count = _nonfinite_parameter_count(request.model.parameters())
    grad_norm = _global_grad_norm(request.model.parameters())
    nonzero_grad_count = _nonzero_grad_parameter_tensor_count(
        request.model.parameters(),
    )
    torch.nn.utils.clip_grad_norm_(
        request.model.parameters(),
        request.gradient_clip_global_norm,
    )
    before_update = _clone_trainable_parameters(request.model.parameters())
    request.optimizer.step()
    update_norm = _parameter_update_norm(
        before=before_update,
        after=_trainable_parameters(request.model.parameters()),
    )
    nonzero_update_count = _nonzero_update_parameter_tensor_count(
        before=before_update,
        after=_trainable_parameters(request.model.parameters()),
    )
    return TrainStepResult(
        forward=output,
        losses=losses,
        grad_norm=grad_norm,
        param_update_norm=update_norm,
        nonfinite_count=nonfinite_count,
        trainable_parameter_tensor_count=len(before_update),
        nonzero_grad_parameter_tensor_count=nonzero_grad_count,
        nonzero_update_parameter_tensor_count=nonzero_update_count,
        optimizer_step_index=request.optimizer_step_index,
        successful_optimizer_update_count=request.optimizer_step_index + 1,
    )


def _global_grad_norm(parameters: Iterable[nn.Parameter]) -> float:
    squared_norm = 0.0
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        gradient_f32 = gradient.detach().to(dtype=torch.float32)
        squared_norm += float(gradient_f32.square().sum().item())
    return math.sqrt(squared_norm)


def _nonfinite_parameter_count(parameters: Iterable[nn.Parameter]) -> int:
    count = 0
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is not None:
            count += int((~torch.isfinite(gradient)).sum().item())
        count += int((~torch.isfinite(parameter.detach())).sum().item())
    return count


def _nonzero_grad_parameter_tensor_count(parameters: Iterable[nn.Parameter]) -> int:
    count = 0
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is not None and bool(torch.count_nonzero(gradient).item()):
            count += 1
    return count


def _trainable_parameters(parameters: Iterable[nn.Parameter]) -> list[nn.Parameter]:
    return [parameter for parameter in parameters if bool(parameter.requires_grad)]


def _clone_trainable_parameters(
    parameters: Iterable[nn.Parameter],
) -> list[torch.Tensor]:
    return [
        parameter.detach().clone()
        for parameter in parameters
        if bool(parameter.requires_grad)
    ]


def _parameter_update_norm(
    *,
    before: list[torch.Tensor],
    after: list[nn.Parameter],
) -> float:
    if len(before) != len(after):
        message = "Before/after parameter lists have different lengths"
        raise ValueError(message)
    squared_norm = 0.0
    for before_tensor, after_parameter in zip(before, after, strict=True):
        delta = after_parameter.detach().to(dtype=torch.float32) - before_tensor.to(
            dtype=torch.float32,
        )
        squared_norm += float(delta.square().sum().item())
    return math.sqrt(squared_norm)


def _nonzero_update_parameter_tensor_count(
    *,
    before: list[torch.Tensor],
    after: list[nn.Parameter],
) -> int:
    if len(before) != len(after):
        message = "Before/after parameter lists have different lengths"
        raise ValueError(message)
    count = 0
    for before_tensor, after_parameter in zip(before, after, strict=True):
        delta = after_parameter.detach() - before_tensor
        if bool(torch.count_nonzero(delta).item()):
            count += 1
    return count


__all__ = ["TrainStepRequest", "TrainStepResult", "run_train_step"]
