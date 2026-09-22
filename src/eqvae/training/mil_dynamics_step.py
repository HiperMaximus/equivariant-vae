# Copyright 2026 HiperMaximus
"""One instrumented AMP/AdamW attempt preserving the historical retry rule."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

import torch
from torch import Tensor, nn

from eqvae.training.mil_dynamics import AdamWTelemetry

if TYPE_CHECKING:
    from collections.abc import Callable


class StepScaler(Protocol):
    """Public GradScaler surface used by the instrumented attempt."""

    def scale(self, outputs: Tensor) -> Tensor: ...
    def unscale_(self, optimizer: torch.optim.Optimizer) -> None: ...
    def step(self, optimizer: torch.optim.Optimizer) -> object: ...
    def update(self, new_scale: float | Tensor | None = None) -> None: ...
    def get_scale(self) -> float: ...


@dataclass(frozen=True)
class InstrumentedStepResult:
    """One committed or overflow-skipped optimizer attempt."""

    committed: bool
    initial_scale: float
    final_scale: float
    logits: tuple[float, ...]
    unweighted_loss: float
    weighted_loss: float
    optimizer: dict[str, float]


def instrumented_adamw_attempt(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: StepScaler,
    telemetry: AdamWTelemetry,
    next_committed_update: int,
    forward_loss: Callable[[], tuple[Tensor, Tensor, Tensor]],
) -> InstrumentedStepResult:
    """Run one attempt; an AMP backoff records telemetry but advances no state.

    ``forward_loss`` returns logits, ordinary CE and the already class-weighted
    scalar objective. Gradients are unscaled exactly once before telemetry. The
    probe uses the standard GradScaler contract with ``backoff_factor < 1``;
    therefore a reduced scale identifies an overflow-skipped optimizer step.
    """
    optimizer.zero_grad(set_to_none=True)
    logits, unweighted_loss, weighted_loss = forward_loss()
    if logits.ndim != 1 or unweighted_loss.ndim != 0 or weighted_loss.ndim != 0:
        raise ValueError("Instrumented MIL closure shapes differ")
    if not torch.isfinite(unweighted_loss) or not torch.isfinite(weighted_loss):
        raise FloatingPointError("Instrumented MIL loss is nonfinite")
    initial_scale = float(scaler.get_scale())
    scaler.scale(weighted_loss).backward()
    scaler.unscale_(optimizer)
    telemetry.begin_step(model, optimizer)
    scaler.step(optimizer)
    scaler.update()
    final_scale = float(scaler.get_scale())
    committed = final_scale >= initial_scale
    optimizer_record = telemetry.finish_step(
        model,
        next_committed_update=next_committed_update,
        committed=committed,
    )
    optimizer.zero_grad(set_to_none=True)
    logits_values = tuple(
        float(value) for value in logits.detach().float().cpu().tolist()
    )
    unweighted = float(unweighted_loss.detach().float().cpu().item())
    weighted = float(weighted_loss.detach().float().cpu().item())
    if not all(
        math.isfinite(value) for value in (*logits_values, unweighted, weighted)
    ):
        raise FloatingPointError("Instrumented MIL outputs are nonfinite")
    return InstrumentedStepResult(
        committed=committed,
        initial_scale=initial_scale,
        final_scale=final_scale,
        logits=logits_values,
        unweighted_loss=unweighted,
        weighted_loss=weighted,
        optimizer=optimizer_record,
    )


def classification_example_record(
    *,
    logits: Tensor,
    target: int,
    unweighted_loss: Tensor,
    weighted_loss: Tensor,
) -> dict[str, float | int | bool]:
    """Return compact per-WSI prediction dynamics without retaining tensors."""
    if logits.ndim != 1 or not 0 <= target < logits.numel():
        raise ValueError("Classification example identity differs")
    probabilities = torch.softmax(logits.detach().float(), dim=0)
    prediction = int(probabilities.argmax().item())
    incorrect = logits.detach().float().clone()
    incorrect[target] = -torch.inf
    margin = logits[target].detach().float() - incorrect.max()
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    record: dict[str, float | int | bool] = {
        "target": target,
        "prediction": prediction,
        "correct": prediction == target,
        "unweighted_loss": float(unweighted_loss.detach().float().item()),
        "weighted_loss": float(weighted_loss.detach().float().item()),
        "true_probability": float(probabilities[target].item()),
        "maximum_probability": float(probabilities.max().item()),
        "predictive_entropy": float(entropy.item()),
        "true_class_margin": float(margin.item()),
        "logit_l2": float(logits.detach().float().norm().item()),
        "logit_max_abs": float(logits.detach().float().abs().max().item()),
        "logit_range": float(
            (logits.detach().float().max() - logits.detach().float().min()).item()
        ),
    }
    for index, value in enumerate(logits.detach().float().cpu().tolist()):
        record[f"logit_{index}"] = float(value)
    for index, value in enumerate(probabilities.cpu().tolist()):
        record[f"probability_{index}"] = float(value)
    return record


__all__ = [
    "InstrumentedStepResult",
    "StepScaler",
    "classification_example_record",
    "instrumented_adamw_attempt",
]
