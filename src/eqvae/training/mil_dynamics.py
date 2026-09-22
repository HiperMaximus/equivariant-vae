# Copyright 2026 HiperMaximus
"""Non-invasive learning-dynamics telemetry for the local-global MIL model.

The hot-path helpers reduce tensors on device and transfer one compact vector.
Detailed activation/activation-gradient probes are deliberately eager and are
intended for a frozen diagnostic copy of a checkpoint, never the compiled
training model.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Protocol, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from eqvae.models.local_global_mil import (
    LocalGlobalMILClassifier,
    LocalTransformerBlock,
    PackedLinear,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

COMPACT_STAT_NAMES: Final = (
    "count",
    "finite_fraction",
    "nonfinite_fraction",
    "mean",
    "mean_abs",
    "std",
    "rms",
    "l2",
    "max_abs",
    "minimum",
    "maximum",
    "zero_fraction",
    "near_zero_fraction",
)
DEFAULT_QUANTILES: Final = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
DEFAULT_CLIP_THRESHOLDS: Final = (0.5, 1.0, 2.0, 5.0, 10.0)
EPSILON: Final = 1e-12


class Stateful(Protocol):
    """Minimal stateful object used by checkpoint integration."""

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state_dict: dict[str, Any]) -> None: ...


class ExampleDynamicsTracker:
    """Resumable WSI-level learning/forgetting trajectories at fixed boundaries."""

    def __init__(self, wsi_ids: Sequence[int]) -> None:
        identities = tuple(int(value) for value in wsi_ids)
        if not identities or len(set(identities)) != len(identities):
            raise ValueError("Example dynamics require unique WSI identities")
        self.wsi_ids = identities
        self._index = {wsi_id: index for index, wsi_id in enumerate(identities)}
        count = len(identities)
        self.observations = torch.zeros(count, dtype=torch.int64)
        self.previous_correct = torch.zeros(count, dtype=torch.bool)
        self.has_previous = torch.zeros(count, dtype=torch.bool)
        self.learning_events = torch.zeros(count, dtype=torch.int64)
        self.forgetting_events = torch.zeros(count, dtype=torch.int64)
        self.first_learned_boundary = torch.full((count,), -1, dtype=torch.int64)
        self.last_forgotten_boundary = torch.full((count,), -1, dtype=torch.int64)
        self.margin_sum = torch.zeros(count, dtype=torch.float64)
        self.true_probability_sum = torch.zeros(count, dtype=torch.float64)

    def update(
        self,
        *,
        wsi_id: int,
        boundary: int,
        correct: bool,
        margin: float,
        true_probability: float,
    ) -> None:
        """Commit one fixed-boundary observation for a known WSI."""
        if wsi_id not in self._index:
            raise ValueError("Unknown WSI identity in dynamics tracker")
        if boundary < 0 or not math.isfinite(margin) or not 0 <= true_probability <= 1:
            raise ValueError("Example dynamics observation is invalid")
        index = self._index[wsi_id]
        had_previous = bool(self.has_previous[index])
        previous = bool(self.previous_correct[index])
        if correct and (not had_previous or not previous):
            self.learning_events[index] += 1
            if self.first_learned_boundary[index] < 0:
                self.first_learned_boundary[index] = boundary
        if had_previous and previous and not correct:
            self.forgetting_events[index] += 1
            self.last_forgotten_boundary[index] = boundary
        self.observations[index] += 1
        self.previous_correct[index] = correct
        self.has_previous[index] = True
        self.margin_sum[index] += margin
        self.true_probability_sum[index] += true_probability

    def record(self, wsi_id: int) -> dict[str, int | float | bool]:
        """Return the compact cumulative dynamics state for one WSI."""
        if wsi_id not in self._index:
            raise ValueError("Unknown WSI identity in dynamics tracker")
        index = self._index[wsi_id]
        observations = int(self.observations[index])
        return {
            "wsi_id": wsi_id,
            "observations": observations,
            "currently_correct": bool(self.previous_correct[index]),
            "learning_events": int(self.learning_events[index]),
            "forgetting_events": int(self.forgetting_events[index]),
            "first_learned_boundary": int(self.first_learned_boundary[index]),
            "last_forgotten_boundary": int(self.last_forgotten_boundary[index]),
            "mean_margin": float(self.margin_sum[index] / max(observations, 1)),
            "mean_true_probability": float(
                self.true_probability_sum[index] / max(observations, 1)
            ),
        }

    def state_dict(self) -> dict[str, Any]:
        """Return the exact compact state required for checkpoint continuation."""
        return {
            "schema_version": "spec0054.example_dynamics.v1",
            "wsi_ids": self.wsi_ids,
            "observations": self.observations.clone(),
            "previous_correct": self.previous_correct.clone(),
            "has_previous": self.has_previous.clone(),
            "learning_events": self.learning_events.clone(),
            "forgetting_events": self.forgetting_events.clone(),
            "first_learned_boundary": self.first_learned_boundary.clone(),
            "last_forgotten_boundary": self.last_forgotten_boundary.clone(),
            "margin_sum": self.margin_sum.clone(),
            "true_probability_sum": self.true_probability_sum.clone(),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restore only a state with identical WSI ordering and tensor geometry."""
        expected = set(self.state_dict())
        if set(state_dict) != expected:
            raise ValueError("Example dynamics state fields differ")
        if state_dict["schema_version"] != "spec0054.example_dynamics.v1":
            raise ValueError("Example dynamics schema differs")
        if tuple(state_dict["wsi_ids"]) != self.wsi_ids:
            raise ValueError("Example dynamics WSI identities differ")
        for name in expected - {"schema_version", "wsi_ids"}:
            value = state_dict[name]
            current = getattr(self, name)
            if not isinstance(value, Tensor) or value.shape != current.shape:
                raise ValueError(f"Example dynamics tensor differs: {name}")
            current.copy_(value.to(dtype=current.dtype, device=current.device))


class T0ForwardWrapper(nn.Module):
    """Compile-friendly forward returning logits and six compact device rows."""

    capture_names: tuple[str, ...] = (
        "patches_x0",
        "patches_x1",
        "patches_x2",
        "global_tokens_t1",
        "cls",
        "bag_embedding",
    )
    statistic_names: tuple[str, ...] = (
        "count",
        "sum",
        "sum_squares",
        "max_abs",
        "near_zero_count",
        "nonfinite_count",
    )

    def __init__(
        self, model: LocalGlobalMILClassifier, *, near_zero: float = 1e-8
    ) -> None:
        super().__init__()
        if near_zero < 0:
            raise ValueError("Near-zero threshold must be nonnegative")
        self.model = model
        self.near_zero = near_zero

    def forward(self, latents: Tensor, graph: Any) -> tuple[Tensor, Tensor]:
        """Return the ordinary logits and a fixed `[6,6]` reduction tensor."""
        logits, representations = self.model.forward_with_representations(
            latents, graph
        )
        rows: list[Tensor] = []
        for representation in representations:
            values = representation.detach().float().reshape(-1)
            finite = torch.isfinite(values)
            safe = torch.where(finite, values, torch.zeros_like(values))
            rows.append(
                torch.stack(
                    (
                        torch.scalar_tensor(
                            values.numel(), device=values.device, dtype=torch.float32
                        ),
                        safe.sum(),
                        safe.square().sum(),
                        safe.abs().amax(),
                        ((safe.abs() <= self.near_zero) & finite).sum(
                            dtype=torch.float32
                        ),
                        (~finite).sum(dtype=torch.float32),
                    )
                )
            )
        return logits, torch.stack(rows)


def t0_forward_records(summary: Tensor) -> dict[str, dict[str, float]]:
    """Materialize one fixed T0 wrapper tensor with a single host transfer."""
    expected = (
        len(T0ForwardWrapper.capture_names),
        len(T0ForwardWrapper.statistic_names),
    )
    if tuple(summary.shape) != expected:
        raise ValueError("T0 forward summary shape differs")
    values = cast("list[list[float]]", summary.detach().double().cpu().tolist())
    result: dict[str, dict[str, float]] = {}
    for capture_name, row in zip(T0ForwardWrapper.capture_names, values, strict=True):
        raw = dict(zip(T0ForwardWrapper.statistic_names, row, strict=True))
        count = max(raw["count"], 1.0)
        finite_count = max(count - raw["nonfinite_count"], 1.0)
        mean = raw["sum"] / finite_count
        second_moment = raw["sum_squares"] / finite_count
        result[capture_name] = {
            **raw,
            "finite_fraction": (count - raw["nonfinite_count"]) / count,
            "mean": mean,
            "std": math.sqrt(max(second_moment - mean * mean, 0.0)),
            "rms": math.sqrt(max(second_moment, 0.0)),
            "near_zero_fraction": raw["near_zero_count"] / finite_count,
        }
    return result


@dataclass(frozen=True)
class DeviceSummary:
    """One ordered vector of reductions that stays on device until materialized."""

    names: tuple[str, ...]
    values: Tensor

    def record(self, *, prefix: str = "") -> dict[str, float]:
        """Transfer the compact vector once and return JSON-safe scalars."""
        values = cast("list[float]", self.values.detach().double().cpu().tolist())
        return {
            f"{prefix}{name}": float(value)
            for name, value in zip(self.names, values, strict=True)
        }


def compact_tensor_summary(
    tensor: Tensor,
    *,
    near_zero: float = 1e-8,
) -> DeviceSummary:
    """Reduce a tensor without dynamic finite-value indexing or per-stat syncs."""
    if near_zero < 0:
        raise ValueError("Near-zero threshold must be nonnegative")
    values = tensor.detach().float().reshape(-1)
    if values.numel() == 0:
        raise ValueError("Cannot summarize an empty tensor")
    finite = torch.isfinite(values)
    safe = torch.where(finite, values, torch.zeros_like(values))
    count = torch.scalar_tensor(
        values.numel(), device=values.device, dtype=torch.float32
    )
    finite_count = finite.sum(dtype=torch.float32)
    denominator = finite_count.clamp_min(1.0)
    mean = safe.sum() / denominator
    mean_abs = safe.abs().sum() / denominator
    second_moment = safe.square().sum() / denominator
    std = (second_moment - mean.square()).clamp_min(0.0).sqrt()
    rms = second_moment.sqrt()
    l2 = safe.square().sum().sqrt()
    max_abs = safe.abs().amax()
    positive_inf = torch.full_like(values, torch.inf)
    negative_inf = torch.full_like(values, -torch.inf)
    minimum = torch.where(finite, values, positive_inf).amin()
    maximum = torch.where(finite, values, negative_inf).amax()
    has_finite = finite_count > 0
    minimum = torch.where(has_finite, minimum, torch.zeros_like(minimum))
    maximum = torch.where(has_finite, maximum, torch.zeros_like(maximum))
    finite_fraction = finite_count / count
    zero_fraction = ((safe == 0) & finite).sum(dtype=torch.float32) / denominator
    near_zero_fraction = ((safe.abs() <= near_zero) & finite).sum(
        dtype=torch.float32
    ) / denominator
    reductions = torch.stack(
        (
            count,
            finite_fraction,
            1.0 - finite_fraction,
            mean,
            mean_abs,
            std,
            rms,
            l2,
            max_abs,
            minimum,
            maximum,
            zero_fraction,
            near_zero_fraction,
        )
    )
    return DeviceSummary(COMPACT_STAT_NAMES, reductions)


def detailed_tensor_summary(
    tensor: Tensor,
    *,
    near_zero: float = 1e-8,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    maximum_quantile_values: int = 131_072,
) -> DeviceSummary:
    """Add deterministic sampled quantiles for a T1/T2 tensor observation."""
    if maximum_quantile_values < 1:
        raise ValueError("Quantile sample budget must be positive")
    if any(not 0.0 <= value <= 1.0 for value in quantiles):
        raise ValueError("Quantiles must lie in [0,1]")
    compact = compact_tensor_summary(tensor, near_zero=near_zero)
    flattened = tensor.detach().float().reshape(-1)
    stride = max(1, math.ceil(flattened.numel() / maximum_quantile_values))
    sample = flattened[::stride]
    sample = torch.where(torch.isfinite(sample), sample, torch.nan)
    q = torch.tensor(tuple(quantiles), device=sample.device, dtype=torch.float32)
    quantile_values = torch.nanquantile(sample, q)
    quantile_values = torch.nan_to_num(quantile_values)
    names = compact.names + tuple(f"q{int(value * 100):02d}" for value in quantiles)
    return DeviceSummary(names, torch.cat((compact.values, quantile_values)))


def semantic_parameter_group(parameter_name: str) -> str:
    """Map a parameter to the six stable Spec 0026 semantic subsystems."""
    if parameter_name.startswith("patch_encoder."):
        return "patch_encoder"
    if parameter_name.startswith("local_blocks.0."):
        return "local_block_0"
    if parameter_name.startswith("local_blocks.1."):
        return "local_block_1"
    if parameter_name == "global_tokens" or parameter_name.startswith(
        "global_summary."
    ):
        return "global_summary"
    if parameter_name.startswith("cls_block."):
        return "cls_block"
    if parameter_name.startswith(("final_norm.", "classifier.")):
        return "classifier_head"
    raise ValueError(f"Unmapped MIL parameter: {parameter_name}")


@dataclass
class _GroupTotals:
    parameter_sq: Tensor
    gradient_sq: Tensor
    update_sq: Tensor
    data_update_sq: Tensor
    decay_update_sq: Tensor
    gradient_update_dot: Tensor
    weight_gradient_dot: Tensor
    weight_update_dot: Tensor
    gradient_max: Tensor
    gradient_nonfinite: Tensor
    gradient_elements: Tensor


def _zero_group(device: torch.device) -> _GroupTotals:
    zero = torch.zeros((), device=device, dtype=torch.float64)
    return _GroupTotals(*(zero.clone() for _ in range(11)))


class AdamWTelemetry:
    """Measure unscaled gradients and the actual committed AdamW parameter delta.

    ``begin_step`` is called after the single AMP ``unscale_`` and before
    ``scaler.step``. ``finish_step`` is called after ``scaler.update`` with the
    already known commit/overflow result. Buffers remain on device; one compact
    stacked vector is copied to the host per optimizer attempt.
    """

    def __init__(self) -> None:
        self.previous_gradients: dict[str, Tensor] = {}
        self.previous_updates: dict[str, Tensor] = {}
        self.last_committed_update = 0
        self._parameters_before: dict[str, Tensor] = {}
        self._gradients: dict[str, Tensor] = {}
        self._options: dict[str, tuple[float, float]] = {}

    def begin_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Capture reusable on-device buffers after gradients have been unscaled."""
        if self._parameters_before:
            raise RuntimeError("A telemetry optimizer attempt is already pending")
        named = dict(model.named_parameters())
        parameter_to_name = {id(parameter): name for name, parameter in named.items()}
        for group in optimizer.param_groups:
            learning_rate = float(group["lr"])
            weight_decay = float(group["weight_decay"])
            for parameter in cast("list[nn.Parameter]", group["params"]):
                if parameter.grad is None:
                    continue
                name = parameter_to_name.get(id(parameter))
                if name is None:
                    raise ValueError("Optimizer contains a parameter outside the model")
                self._parameters_before[name] = parameter.detach().float().clone()
                self._gradients[name] = parameter.grad.detach().float().clone()
                self._options[name] = (learning_rate, weight_decay)
        if not self._parameters_before:
            raise ValueError("Optimizer attempt has no gradients")

    def finish_step(
        self,
        model: nn.Module,
        *,
        next_committed_update: int,
        committed: bool,
    ) -> dict[str, float]:
        """Reduce one attempted step and advance temporal state only on commit."""
        if not self._parameters_before:
            raise RuntimeError("No telemetry optimizer attempt is pending")
        if next_committed_update != self.last_committed_update + 1:
            raise ValueError("Telemetry committed-update index is not contiguous")
        named = dict(model.named_parameters())
        first_name = next(iter(self._parameters_before))
        device = self._parameters_before[first_name].device
        group_names = (
            "global",
            "patch_encoder",
            "local_block_0",
            "local_block_1",
            "global_summary",
            "cls_block",
            "classifier_head",
        )
        groups = {name: _zero_group(device) for name in group_names}
        current_updates: dict[str, Tensor] = {}
        previous_gradient_dot = torch.zeros((), device=device, dtype=torch.float64)
        previous_gradient_sq = torch.zeros_like(previous_gradient_dot)
        current_gradient_sq = torch.zeros_like(previous_gradient_dot)
        previous_update_dot = torch.zeros_like(previous_gradient_dot)
        previous_update_sq = torch.zeros_like(previous_gradient_dot)
        current_update_sq = torch.zeros_like(previous_gradient_dot)
        update_sign_flips = torch.zeros_like(previous_gradient_dot)
        update_sign_elements = torch.zeros_like(previous_gradient_dot)

        for name, parameter_before in self._parameters_before.items():
            if name not in named or named[name].shape != parameter_before.shape:
                raise ValueError(f"Telemetry parameter identity differs: {name}")
            gradient = self._gradients[name]
            if committed:
                update = named[name].detach().float() - parameter_before
                learning_rate, weight_decay = self._options[name]
                decay_update = -learning_rate * weight_decay * parameter_before
                data_update = update - decay_update
            else:
                update = torch.zeros_like(parameter_before)
                decay_update = torch.zeros_like(parameter_before)
                data_update = torch.zeros_like(parameter_before)
            semantic = semantic_parameter_group(name)
            for group_name in ("global", semantic):
                totals = groups[group_name]
                totals.parameter_sq += parameter_before.double().square().sum()
                totals.gradient_sq += gradient.double().square().sum()
                totals.update_sq += update.double().square().sum()
                totals.data_update_sq += data_update.double().square().sum()
                totals.decay_update_sq += decay_update.double().square().sum()
                totals.gradient_update_dot += (
                    gradient.double() * update.double()
                ).sum()
                totals.weight_gradient_dot += (
                    parameter_before.double() * gradient.double()
                ).sum()
                totals.weight_update_dot += (
                    parameter_before.double() * update.double()
                ).sum()
                totals.gradient_max = torch.maximum(
                    totals.gradient_max,
                    torch.nan_to_num(
                        gradient.abs().amax().double(),
                        nan=torch.inf,
                        posinf=torch.inf,
                        neginf=torch.inf,
                    ),
                )
                totals.gradient_nonfinite += (~torch.isfinite(gradient)).sum().double()
                totals.gradient_elements += torch.scalar_tensor(
                    gradient.numel(), device=device, dtype=torch.float64
                )
            if committed:
                current_updates[name] = update.clone()
                if name in self.previous_gradients:
                    previous = self.previous_gradients[name].to(device)
                    previous_gradient_dot += (
                        previous.double() * gradient.double()
                    ).sum()
                    previous_gradient_sq += previous.double().square().sum()
                    current_gradient_sq += gradient.double().square().sum()
                if name in self.previous_updates:
                    previous = self.previous_updates[name].to(device)
                    previous_update_dot += (previous.double() * update.double()).sum()
                    previous_update_sq += previous.double().square().sum()
                    current_update_sq += update.double().square().sum()
                    nonzero = (previous != 0) & (update != 0)
                    update_sign_flips += (
                        ((previous.sign() != update.sign()) & nonzero).sum().double()
                    )
                    update_sign_elements += nonzero.sum().double()

        output_names: list[str] = ["attempt.committed"]
        output_values: list[Tensor] = [
            torch.scalar_tensor(float(committed), device=device, dtype=torch.float64)
        ]

        def add(name: str, value: Tensor) -> None:
            output_names.append(name)
            output_values.append(value.double())

        for group_name in group_names:
            totals = groups[group_name]
            parameter_norm = totals.parameter_sq.sqrt()
            gradient_norm = totals.gradient_sq.sqrt()
            update_norm = totals.update_sq.sqrt()
            add(f"{group_name}.parameter_l2", parameter_norm)
            add(f"{group_name}.gradient_l2", gradient_norm)
            add(f"{group_name}.gradient_max_abs", totals.gradient_max)
            add(
                f"{group_name}.gradient_nonfinite_fraction",
                totals.gradient_nonfinite / totals.gradient_elements.clamp_min(1.0),
            )
            add(f"{group_name}.update_l2", update_norm)
            add(f"{group_name}.data_update_l2", totals.data_update_sq.sqrt())
            add(f"{group_name}.decay_update_l2", totals.decay_update_sq.sqrt())
            add(
                f"{group_name}.gradient_to_weight",
                gradient_norm / parameter_norm.clamp_min(EPSILON),
            )
            add(
                f"{group_name}.update_to_weight",
                update_norm / parameter_norm.clamp_min(EPSILON),
            )
            add(
                f"{group_name}.gradient_update_cosine",
                totals.gradient_update_dot
                / (gradient_norm * update_norm).clamp_min(EPSILON),
            )
            add(
                f"{group_name}.weight_gradient_cosine",
                totals.weight_gradient_dot
                / (parameter_norm * gradient_norm).clamp_min(EPSILON),
            )
            add(
                f"{group_name}.weight_update_cosine",
                totals.weight_update_dot
                / (parameter_norm * update_norm).clamp_min(EPSILON),
            )
            add(
                f"{group_name}.gradient_energy_fraction",
                totals.gradient_sq / groups["global"].gradient_sq.clamp_min(EPSILON),
            )
            add(
                f"{group_name}.update_energy_fraction",
                totals.update_sq / groups["global"].update_sq.clamp_min(EPSILON),
            )
        add(
            "global.gradient_previous_cosine",
            previous_gradient_dot
            / (previous_gradient_sq * current_gradient_sq).sqrt().clamp_min(EPSILON),
        )
        add(
            "global.update_previous_cosine",
            previous_update_dot
            / (previous_update_sq * current_update_sq).sqrt().clamp_min(EPSILON),
        )
        add(
            "global.update_sign_flip_fraction",
            update_sign_flips / update_sign_elements.clamp_min(1.0),
        )
        global_gradient_norm = groups["global"].gradient_sq.sqrt()
        for threshold in DEFAULT_CLIP_THRESHOLDS:
            add(
                f"clip_coefficient_{threshold:g}",
                torch.minimum(
                    torch.ones_like(global_gradient_norm),
                    torch.scalar_tensor(
                        threshold,
                        device=device,
                        dtype=torch.float64,
                    )
                    / (global_gradient_norm + 1e-6),
                ),
            )
        materialized = cast(
            "list[float]",
            torch.stack(output_values).detach().cpu().tolist(),
        )
        record = {
            name: float(value)
            for name, value in zip(output_names, materialized, strict=True)
        }
        if committed:
            self.previous_gradients = {
                name: value.clone() for name, value in self._gradients.items()
            }
            self.previous_updates = current_updates
            self.last_committed_update = next_committed_update
        self._parameters_before.clear()
        self._gradients.clear()
        self._options.clear()
        return record

    def state_dict(self) -> dict[str, Any]:
        """Return exact temporal state; checkpoints require an optimizer boundary."""
        if self._parameters_before:
            raise RuntimeError(
                "Cannot checkpoint telemetry during an optimizer attempt"
            )
        return {
            "schema_version": "spec0054.adamw_telemetry.v2",
            "last_committed_update": self.last_committed_update,
            "previous_gradients": {
                name: value.detach().cpu().clone()
                for name, value in self.previous_gradients.items()
            },
            "previous_updates": {
                name: value.detach().cpu().clone()
                for name, value in self.previous_updates.items()
            },
        }

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        *,
        model: nn.Module | None = None,
    ) -> None:
        """Restore temporal vectors and validate them against the live model."""
        if self._parameters_before:
            raise RuntimeError("Cannot restore telemetry during an optimizer attempt")
        if set(state_dict) != {
            "schema_version",
            "last_committed_update",
            "previous_gradients",
            "previous_updates",
        }:
            raise ValueError("Telemetry state fields differ")
        if state_dict["schema_version"] != "spec0054.adamw_telemetry.v2":
            raise ValueError("Telemetry state schema differs")
        update = state_dict["last_committed_update"]
        gradients = state_dict["previous_gradients"]
        updates = state_dict["previous_updates"]
        if not isinstance(update, int) or update < 0:
            raise ValueError("Telemetry committed update is invalid")
        if not isinstance(gradients, dict) or not isinstance(updates, dict):
            raise TypeError("Telemetry temporal vectors must be mappings")
        if set(gradients) != set(updates):
            raise ValueError("Gradient and update telemetry names differ")
        named = dict(model.named_parameters()) if model is not None else {}
        restored_gradients: dict[str, Tensor] = {}
        restored_updates: dict[str, Tensor] = {}
        for name, gradient_value in gradients.items():
            update_value = updates[name]
            if not isinstance(gradient_value, Tensor) or not isinstance(
                update_value, Tensor
            ):
                raise TypeError("Telemetry temporal values must be tensors")
            if gradient_value.shape != update_value.shape:
                raise ValueError("Gradient and update telemetry shapes differ")
            device = torch.device("cpu")
            if model is not None:
                if name not in named or named[name].shape != gradient_value.shape:
                    raise ValueError(f"Telemetry parameter identity differs: {name}")
                device = named[name].device
            restored_gradients[name] = gradient_value.detach().to(device=device).clone()
            restored_updates[name] = update_value.detach().to(device=device).clone()
        self.previous_gradients = restored_gradients
        self.previous_updates = restored_updates
        self.last_committed_update = update


def semantic_capture_modules(model: LocalGlobalMILClassifier) -> dict[str, nn.Module]:
    """Return stable named activation capture points for the accepted classifier."""
    modules: dict[str, nn.Module] = {}
    for index, layer in enumerate(model.patch_encoder.layers):
        modules[f"patch_encoder.layers.{index}"] = layer
    for block_index, block_value in enumerate(model.local_blocks):
        block = cast("LocalTransformerBlock", block_value)
        prefix = f"local_blocks.{block_index}"
        modules[f"{prefix}.attention_norm"] = block.attention_norm
        modules[f"{prefix}.attention.qkv"] = block.attention.qkv
        modules[f"{prefix}.attention.output"] = block.attention.output
        modules[f"{prefix}.attention"] = block.attention
        modules[f"{prefix}.ffn_norm"] = block.ffn_norm
        modules[f"{prefix}.ffn.input"] = block.ffn.input
        modules[f"{prefix}.ffn.output"] = block.ffn.output
        modules[prefix] = block
    modules.update(
        {
            "global_summary.query_norm": model.global_summary.query_norm,
            "global_summary.patch_norm": model.global_summary.patch_norm,
            "global_summary.attention.query": model.global_summary.attention.query,
            "global_summary.attention.kv": model.global_summary.attention.kv,
            "global_summary.attention.output": model.global_summary.attention.output,
            "global_summary.attention": model.global_summary.attention,
            "global_summary.ffn_norm": model.global_summary.ffn_norm,
            "global_summary.ffn.input": model.global_summary.ffn.input,
            "global_summary.ffn.output": model.global_summary.ffn.output,
            "global_summary": model.global_summary,
            "cls_block.attention_norm": model.cls_block.attention_norm,
            "cls_block.query": model.cls_block.query,
            "cls_block.kv": model.cls_block.kv,
            "cls_block.output": model.cls_block.output,
            "cls_block.ffn_norm": model.cls_block.ffn_norm,
            "cls_block.ffn.input": model.cls_block.ffn.input,
            "cls_block.ffn.output": model.cls_block.ffn.output,
            "cls_block": model.cls_block,
            "final_norm": model.final_norm,
            # The historical forward deliberately calls functional.linear in FP32,
            # so the nn.Linear module itself is not invoked and cannot fire a hook.
            "classifier_logits": model,
        }
    )
    return modules


class EagerLayerProbe:
    """Capture detailed layer activations and gradients on a diagnostic model."""

    def __init__(
        self,
        modules: Mapping[str, nn.Module],
        *,
        near_zero: float = 1e-8,
    ) -> None:
        self.modules = dict(modules)
        self.near_zero = near_zero
        self._handles: list[Any] = []
        self._records: dict[str, dict[str, float]] = {}
        self._input_gradient_norms: dict[str, float] = {}
        self._output_gradient_norms: dict[str, float] = {}

    def __enter__(self) -> EagerLayerProbe:
        if self._handles:
            raise RuntimeError("Layer probe is already active")
        for name, module in self.modules.items():
            self._handles.append(module.register_forward_hook(self._hook(name)))
        return self

    def __exit__(self, *_args: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _hook(self, name: str) -> Callable[[nn.Module, tuple[Any, ...], Any], None]:
        def capture(_module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            if not isinstance(output, Tensor):
                raise TypeError(f"Captured module output is not a tensor: {name}")
            input_tensor = next(
                (item for item in inputs if isinstance(item, Tensor)), None
            )
            self._records[name] = detailed_tensor_summary(
                output,
                near_zero=self.near_zero,
            ).record(prefix="forward.")
            if isinstance(_module, PackedLinear):
                parts = _module.split(output)
                labels = _logical_projection_labels(name, len(parts))
                for label, part in zip(labels, parts, strict=True):
                    self._records[name].update(
                        detailed_tensor_summary(
                            part,
                            near_zero=self.near_zero,
                        ).record(prefix=f"forward_{label}.")
                    )
            channel_axis = 1 if output.ndim == 4 else output.ndim - 1
            if output.ndim >= 2:
                self._records[name].update(
                    _channel_distribution_record(
                        output,
                        channel_axis=channel_axis,
                        near_zero=self.near_zero,
                        prefix="forward.channel.",
                    )
                )
            if input_tensor is not None and input_tensor.requires_grad:
                input_axis = 1 if input_tensor.ndim == 4 else input_tensor.ndim - 1
                input_tensor.register_hook(
                    self._gradient_hook(name, "backward_input", input_axis)
                )
            if output.requires_grad:
                output.register_hook(
                    self._gradient_hook(name, "backward_output", channel_axis)
                )

        return capture

    def _gradient_hook(
        self,
        name: str,
        prefix: str,
        channel_axis: int,
    ) -> Callable[[Tensor], None]:
        def capture(gradient: Tensor) -> None:
            self._records[name].update(
                detailed_tensor_summary(
                    gradient,
                    near_zero=self.near_zero,
                ).record(prefix=f"{prefix}.")
            )
            if gradient.ndim >= 2:
                self._records[name].update(
                    _channel_distribution_record(
                        gradient,
                        channel_axis=channel_axis,
                        near_zero=self.near_zero,
                        prefix=f"{prefix}.channel.",
                    )
                )
            norm = float(gradient.detach().float().norm().item())
            if prefix == "backward_input":
                self._input_gradient_norms[name] = norm
            else:
                self._output_gradient_norms[name] = norm

        return capture

    def records(self) -> dict[str, dict[str, float]]:
        """Return summaries accumulated by forward and transient gradient hooks."""
        result = copy.deepcopy(self._records)
        for name, input_norm in self._input_gradient_norms.items():
            if name in self._output_gradient_norms:
                result[name]["backward.amplification"] = input_norm / max(
                    self._output_gradient_norms[name], EPSILON
                )
        return result


def _channel_distribution_record(
    tensor: Tensor,
    *,
    channel_axis: int,
    near_zero: float,
    prefix: str,
) -> dict[str, float]:
    values = (
        tensor.detach()
        .float()
        .movedim(channel_axis, -1)
        .reshape(-1, tensor.shape[channel_axis])
    )
    finite = torch.isfinite(values)
    safe = torch.where(finite, values, torch.zeros_like(values))
    counts = finite.sum(dim=0).float().clamp_min(1.0)
    means = safe.sum(dim=0) / counts
    second = safe.square().sum(dim=0) / counts
    standard_deviations = (second - means.square()).clamp_min(0).sqrt()
    rms = second.sqrt()
    record_tensor = torch.stack(
        (
            standard_deviations.mean(),
            standard_deviations.amin(),
            standard_deviations.amax(),
            rms.mean(),
            rms.std(unbiased=False) / rms.mean().clamp_min(EPSILON),
            (standard_deviations <= near_zero).float().mean(),
        )
    )
    names = (
        "std_mean",
        "std_min",
        "std_max",
        "rms_mean",
        "rms_cv",
        "near_dead_fraction",
    )
    values_host = cast("list[float]", record_tensor.cpu().tolist())
    return {
        f"{prefix}{name}": float(value)
        for name, value in zip(names, values_host, strict=True)
    }


def _logical_projection_labels(name: str, count: int) -> tuple[str, ...]:
    if name.endswith("attention.qkv") and count == 3:
        return ("query", "key", "value")
    if name.endswith("ffn.input") and count == 2:
        return ("gate", "value")
    if name.endswith(("attention.kv", "cls_block.kv")) and count == 2:
        return ("key", "value")
    return tuple(f"logical_{index}" for index in range(count))


def run_eager_layer_probe(
    model: LocalGlobalMILClassifier,
    loss_closure: Callable[[LocalGlobalMILClassifier], Tensor],
) -> dict[str, dict[str, float]]:
    """Run T1 on a disposable checkpoint copy without touching live training state."""
    diagnostic_model = copy.deepcopy(model)
    diagnostic_model.train(model.training)
    diagnostic_model.zero_grad(set_to_none=True)
    with EagerLayerProbe(semantic_capture_modules(diagnostic_model)) as probe:
        loss = loss_closure(diagnostic_model)
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise ValueError("Diagnostic closure must return one finite scalar loss")
        loss.backward()
        records = probe.records()
    diagnostic_model.zero_grad(set_to_none=True)
    return records


def parameter_optimizer_records(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    last_updates: Mapping[str, Tensor] | None = None,
) -> dict[str, dict[str, float]]:
    """Collect sampled T1 parameter, gradient, Adam-state and update statistics."""
    parameter_options: dict[int, tuple[float, float, float, float, float]] = {}
    for group in optimizer.param_groups:
        beta1, beta2 = cast("tuple[float, float]", group["betas"])
        learning_rate = float(group["lr"])
        epsilon = float(group["eps"])
        weight_decay = float(group["weight_decay"])
        for parameter in cast("list[nn.Parameter]", group["params"]):
            parameter_options[id(parameter)] = (
                learning_rate,
                weight_decay,
                beta1,
                beta2,
                epsilon,
            )
    records: dict[str, dict[str, float]] = {}
    for name, parameter in model.named_parameters():
        if id(parameter) not in parameter_options:
            raise ValueError(f"Model parameter is absent from optimizer: {name}")
        learning_rate, weight_decay, beta1, beta2, epsilon = parameter_options[
            id(parameter)
        ]
        weight = parameter.detach().float()
        record = detailed_tensor_summary(weight).record(prefix="weight.")
        record.update(
            {
                "optimizer.learning_rate": learning_rate,
                "optimizer.weight_decay": weight_decay,
            }
        )
        gradient = (
            parameter.grad.detach().float() if parameter.grad is not None else None
        )
        if gradient is not None:
            record.update(detailed_tensor_summary(gradient).record(prefix="gradient."))
            weight_norm = weight.norm()
            gradient_norm = gradient.norm()
            dot = (weight * gradient).sum()
            radial_defined = bool(weight_norm > EPSILON)
            radial_energy = (
                dot.square() / weight_norm.square()
                if radial_defined
                else torch.full_like(dot, torch.nan)
            )
            record.update(
                {
                    "gradient_to_weight": float(
                        (gradient_norm / weight_norm.clamp_min(EPSILON)).item()
                    ),
                    "weight_gradient_cosine": float(
                        (dot / (weight_norm * gradient_norm).clamp_min(EPSILON)).item()
                    ),
                    "radial_defined": float(radial_defined),
                    "radial_gradient_energy_fraction": float(
                        (
                            radial_energy / gradient_norm.square().clamp_min(EPSILON)
                        ).item()
                    ),
                    "perpendicular_gradient_ratio": float(
                        (
                            (gradient_norm.square() - radial_energy).clamp_min(0).sqrt()
                            / gradient_norm.clamp_min(EPSILON)
                        ).item()
                    ),
                }
            )
        if weight.ndim >= 2:
            rows = weight.reshape(weight.shape[0], -1).norm(dim=1)
            record["weight.row_norm_cv"] = float(
                (rows.std(unbiased=False) / rows.mean().clamp_min(EPSILON)).item()
            )
        state = optimizer.state.get(parameter, {})
        exp_avg = state.get("exp_avg")
        exp_avg_sq = state.get("exp_avg_sq")
        step_value = state.get("step", 0)
        step = (
            int(step_value.item())
            if isinstance(step_value, Tensor)
            else int(step_value)
        )
        record["optimizer.step"] = float(step)
        if isinstance(exp_avg, Tensor) and isinstance(exp_avg_sq, Tensor) and step > 0:
            first = exp_avg.detach().float()
            second = exp_avg_sq.detach().float()
            record.update(detailed_tensor_summary(first).record(prefix="adam_m."))
            record.update(detailed_tensor_summary(second).record(prefix="adam_v."))
            bias1 = 1.0 - beta1**step
            bias2 = 1.0 - beta2**step
            denominator = (second / bias2).sqrt()
            effective = learning_rate * (first / bias1) / (denominator + epsilon)
            record.update(
                detailed_tensor_summary(effective).record(prefix="adam_effective_step.")
            )
            record["adam_epsilon_dominated_fraction"] = float(
                (denominator <= epsilon).float().mean().item()
            )
            if gradient is not None:
                record["gradient_adam_m_cosine"] = float(
                    functional.cosine_similarity(
                        gradient.reshape(-1), first.reshape(-1), dim=0
                    ).item()
                )
        if last_updates is not None and name in last_updates:
            update = last_updates[name].detach().to(weight.device).float()
            if update.shape != weight.shape:
                raise ValueError(f"Last-update shape differs: {name}")
            record.update(detailed_tensor_summary(update).record(prefix="last_update."))
            record["last_update_to_weight"] = float(
                (update.norm() / weight.norm().clamp_min(EPSILON)).item()
            )
            if gradient is not None:
                record["gradient_last_update_cosine"] = float(
                    functional.cosine_similarity(
                        gradient.reshape(-1), update.reshape(-1), dim=0
                    ).item()
                )
        records[name] = record
    return records


def representation_spectrum_summary(
    representation: Tensor,
    *,
    maximum_rows: int = 4_096,
) -> dict[str, float]:
    """Compute deterministic light-T2 rank/collapse summaries for `[tokens,dim]`."""
    if representation.ndim != 2 or min(representation.shape) < 1:
        raise ValueError("Representation spectrum requires a nonempty matrix")
    if maximum_rows < 2:
        raise ValueError("Spectrum row budget must be at least two")
    matrix = representation.detach().float()
    if matrix.shape[0] > maximum_rows:
        indices = (
            torch.linspace(
                0,
                matrix.shape[0] - 1,
                steps=maximum_rows,
                device=matrix.device,
            )
            .round()
            .long()
        )
        matrix = matrix[indices]
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    energy = singular_values.square()
    total_energy = energy.sum()
    spectrum_defined = bool(total_energy > EPSILON)
    probabilities = energy / total_energy.clamp_min(EPSILON)
    entropy = -(probabilities * probabilities.clamp_min(EPSILON).log()).sum()
    effective_rank = (
        entropy.exp() if spectrum_defined else torch.full_like(entropy, torch.nan)
    )
    stable_rank = (
        total_energy / energy.amax()
        if spectrum_defined
        else torch.full_like(total_energy, torch.nan)
    )
    normalized = torch.nn.functional.normalize(matrix, dim=1)
    row_norms = matrix.norm(dim=1)
    pairwise_defined = bool(matrix.shape[0] > 1 and torch.all(row_norms > EPSILON))
    mean_pairwise_cosine = (
        (normalized.sum(dim=0).square().sum() - normalized.shape[0])
        / (normalized.shape[0] * (normalized.shape[0] - 1))
        if pairwise_defined
        else torch.full((), torch.nan, device=matrix.device)
    )
    top_energy = (
        torch.cumsum(energy, dim=0) / total_energy
        if spectrum_defined
        else torch.full_like(energy, torch.nan)
    )
    record = {
        "rows": float(matrix.shape[0]),
        "columns": float(matrix.shape[1]),
        "spectrum_defined": float(spectrum_defined),
        "effective_rank": float(effective_rank.item()),
        "stable_rank": float(stable_rank.item()),
        "singular_entropy": float(entropy.item()),
        "largest_singular_value": float(singular_values[0].item()),
        "pairwise_cosine_defined": float(pairwise_defined),
        "mean_pairwise_cosine": float(mean_pairwise_cosine.item()),
    }
    for k in (1, 4, 8, 16, 32):
        if k <= top_energy.numel():
            record[f"top_{k}_energy_fraction"] = float(top_energy[k - 1].item())
    return record


def attention_distribution_summary(weights: Tensor) -> dict[str, float]:
    """Summarize the final axis of softmax or nonnegative sigmoid attention."""
    if weights.ndim < 1 or weights.shape[-1] < 1:
        raise ValueError("Attention weights require a nonempty instance axis")
    values = weights.detach().float()
    if not torch.isfinite(values).all() or (values < 0).any():
        raise ValueError("Attention weights must be finite and nonnegative")
    raw_mass = values.sum(dim=-1)
    valid = raw_mass > EPSILON
    probability = values / raw_mass[..., None].clamp_min(EPSILON)
    entropy = -(probability * probability.clamp_min(EPSILON).log()).sum(dim=-1)
    effective_count = entropy.exp()
    sorted_probability = probability.sort(dim=-1, descending=True).values
    nan = torch.full((), torch.nan, device=values.device)

    def valid_mean(metric: Tensor) -> Tensor:
        return metric[valid].mean() if bool(valid.any()) else nan

    record = {
        "raw_mass_mean": float(raw_mass.mean().item()),
        "distribution_defined_fraction": float(valid.float().mean().item()),
        "entropy_mean": float(valid_mean(entropy).item()),
        "normalized_entropy_mean": float(
            valid_mean(
                entropy / math.log(values.shape[-1])
                if values.shape[-1] > 1
                else entropy * 0
            ).item()
        ),
        "effective_instance_count_mean": float(valid_mean(effective_count).item()),
        "maximum_mass_mean": float(valid_mean(sorted_probability[..., 0]).item()),
    }
    for k in (5, 10, 50, 100):
        if k <= values.shape[-1]:
            record[f"top_{k}_mass_mean"] = float(
                valid_mean(sorted_probability[..., :k].sum(dim=-1)).item()
            )
    return record


__all__ = [
    "AdamWTelemetry",
    "DeviceSummary",
    "EagerLayerProbe",
    "ExampleDynamicsTracker",
    "T0ForwardWrapper",
    "attention_distribution_summary",
    "compact_tensor_summary",
    "detailed_tensor_summary",
    "parameter_optimizer_records",
    "representation_spectrum_summary",
    "run_eager_layer_probe",
    "semantic_capture_modules",
    "semantic_parameter_group",
    "t0_forward_records",
]
