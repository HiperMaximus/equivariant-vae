# Copyright 2026 HiperMaximus
"""Semantic AdamW parameter groups for spec 0001."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict, cast

import torch
from torch import nn

from eqvae.models.activations import GatedScalarActivation

if TYPE_CHECKING:
    from collections.abc import Iterable


class OptimizerParamGroup(TypedDict):
    """Typed AdamW parameter group with stable spec metadata."""

    name: str
    params: list[nn.Parameter]
    lr: float
    weight_decay: float


@dataclass(frozen=True)
class SpecAdamWConfig:
    """AdamW defaults locked by spec 0001."""

    learning_rate: float = 5.0e-4
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1.0e-8
    weight_decay: float = 1.0e-5
    gradient_clip_global_norm: float = 1.0
    gate_lr_multiplier: float = 0.5


_BATCH_LR_SCALING_EXPONENTS: dict[str, float] = {"sqrt": 0.5, "linear": 1.0}


@dataclass(frozen=True)
class BatchLrScaling:
    """Batch-size to learning-rate scaling policy (Spec 0011).

    Encodes the re-tunable rule that maps a global batch size to a learning-rate
    multiplier relative to a reference batch, so each (model x hardware) run scales
    its reference lr to the selected batch by formula rather than a hardcoded
    number. ``rule="sqrt"`` applies the square-root rule
    (multiplier = ``sqrt(global / reference)``); ``rule="linear"`` applies linear
    scaling (multiplier = ``global / reference``).
    """

    reference_global_batch_size: int
    rule: str = "sqrt"

    def __post_init__(self) -> None:
        """Validate the reference batch and rule at construction.

        Raises:
            ValueError: If the reference batch is not positive, or the rule is not
                a known scaling rule.

        """
        if self.reference_global_batch_size <= 0:
            message = (
                "reference_global_batch_size must be positive, got "
                f"{self.reference_global_batch_size}"
            )
            raise ValueError(message)
        if self.rule not in _BATCH_LR_SCALING_EXPONENTS:
            known = sorted(_BATCH_LR_SCALING_EXPONENTS)
            message = (
                f"unknown batch-lr scaling rule {self.rule!r}; known rules: {known}"
            )
            raise ValueError(message)

    @property
    def exponent(self) -> float:
        """Return the batch-ratio exponent for this rule.

        Returns:
            The exponent applied to ``global / reference`` (0.5 for ``sqrt``,
            1.0 for ``linear``).

        """
        return _BATCH_LR_SCALING_EXPONENTS[self.rule]


def scaled_learning_rate(
    *,
    reference_lr: float,
    scaling: BatchLrScaling,
    global_batch_size: int,
) -> float:
    """Scale a reference learning rate to a global batch size (Spec 0011).

    At the reference batch the multiplier is exactly 1.0, so a run at the
    reference batch is behavior-preserving.

    Returns:
        ``reference_lr`` scaled by ``(global / reference) ** exponent``.

    Raises:
        ValueError: If ``global_batch_size`` is not positive.

    """
    if global_batch_size <= 0:
        message = f"global_batch_size must be positive, got {global_batch_size}"
        raise ValueError(message)
    batch_ratio = global_batch_size / scaling.reference_global_batch_size
    return reference_lr * math.pow(batch_ratio, scaling.exponent)


@dataclass(frozen=True)
class OptimizerGroupSummary:
    """Coverage proof for semantic optimizer groups."""

    parameter_group_count: int
    trainable_parameter_count: int
    grouped_parameter_count: int
    all_trainable_parameters_covered_once: bool
    gate_parameters_in_gate_no_decay_group: bool


@dataclass(frozen=True)
class _GroupedParameters:
    groups: dict[str, list[nn.Parameter]]
    duplicate_parameter_ids: set[int]


def build_adamw_parameter_groups(
    model: nn.Module,
    *,
    config: SpecAdamWConfig | None = None,
) -> tuple[list[OptimizerParamGroup], OptimizerGroupSummary]:
    """Build semantic AdamW parameter groups.

    Returns:
        Parameter groups plus coverage summary.

    """
    resolved_config = config or SpecAdamWConfig()
    named_modules = cast("Iterable[tuple[str, nn.Module]]", model.named_modules())
    module_by_name: dict[str, nn.Module] = dict(named_modules)
    grouped = _grouped_parameters(model=model, module_by_name=module_by_name)
    groups = grouped.groups
    gate_parameter_ids = _gate_parameter_ids(model)

    trainable_parameter_ids = {
        id(parameter)
        for parameter in model.parameters()
        if bool(parameter.requires_grad)
    }
    grouped_parameter_ids = {
        id(parameter) for parameters in groups.values() for parameter in parameters
    }
    all_covered = (
        trainable_parameter_ids == grouped_parameter_ids
        and len(grouped.duplicate_parameter_ids) == 0
    )
    gate_group_ids = {id(parameter) for parameter in groups["gate_no_decay"]}
    gate_parameters_grouped = gate_parameter_ids.issubset(gate_group_ids)
    param_groups: list[OptimizerParamGroup] = [
        {
            "name": "decay",
            "params": groups["decay"],
            "lr": resolved_config.learning_rate,
            "weight_decay": resolved_config.weight_decay,
        },
        {
            "name": "no_decay",
            "params": groups["no_decay"],
            "lr": resolved_config.learning_rate,
            "weight_decay": 0.0,
        },
        {
            "name": "gate_no_decay",
            "params": groups["gate_no_decay"],
            "lr": resolved_config.learning_rate * resolved_config.gate_lr_multiplier,
            "weight_decay": 0.0,
        },
    ]
    return param_groups, OptimizerGroupSummary(
        parameter_group_count=len(param_groups),
        trainable_parameter_count=len(trainable_parameter_ids),
        grouped_parameter_count=len(grouped_parameter_ids),
        all_trainable_parameters_covered_once=all_covered,
        gate_parameters_in_gate_no_decay_group=gate_parameters_grouped,
    )


def _grouped_parameters(
    *,
    model: nn.Module,
    module_by_name: dict[str, nn.Module],
) -> _GroupedParameters:
    groups: dict[str, list[nn.Parameter]] = {
        "decay": [],
        "no_decay": [],
        "gate_no_decay": [],
    }
    seen_parameter_ids: set[int] = set()
    duplicate_parameter_ids: set[int] = set()
    named_parameters = cast(
        "Iterable[tuple[str, nn.Parameter]]",
        model.named_parameters(remove_duplicate=False),
    )
    for parameter_name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        parameter_id = id(parameter)
        if parameter_id in seen_parameter_ids:
            duplicate_parameter_ids.add(parameter_id)
        seen_parameter_ids.add(parameter_id)
        group_name = _group_name_for_parameter(
            parameter_name=parameter_name,
            module_by_name=module_by_name,
        )
        groups[group_name].append(parameter)
    return _GroupedParameters(
        groups=groups,
        duplicate_parameter_ids=duplicate_parameter_ids,
    )


def create_adamw_optimizer(
    model: nn.Module,
    *,
    config: SpecAdamWConfig | None = None,
) -> tuple[torch.optim.AdamW, OptimizerGroupSummary]:
    """Create AdamW with the semantic spec 0001 parameter groups.

    Returns:
        Optimizer and coverage summary.

    """
    resolved_config = config or SpecAdamWConfig()
    parameter_groups, summary = build_adamw_parameter_groups(
        model,
        config=resolved_config,
    )
    optimizer = torch.optim.AdamW(
        cast("list[dict[str, object]]", parameter_groups),
        lr=resolved_config.learning_rate,
        betas=(resolved_config.beta1, resolved_config.beta2),
        eps=resolved_config.epsilon,
        weight_decay=resolved_config.weight_decay,
    )
    return optimizer, summary


def _group_name_for_parameter(
    *,
    parameter_name: str,
    module_by_name: dict[str, nn.Module],
) -> str:
    module_name, _, leaf_name = parameter_name.rpartition(".")
    module = module_by_name.get(module_name)
    if isinstance(module, GatedScalarActivation):
        return "gate_no_decay"
    if isinstance(module, nn.Conv2d) and leaf_name == "weight":
        return "decay"
    return "no_decay"


def _gate_parameter_ids(model: nn.Module) -> set[int]:
    parameter_ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, GatedScalarActivation):
            parameter_ids.add(id(module.a))
            parameter_ids.add(id(module.b))
    return parameter_ids


__all__ = [
    "BatchLrScaling",
    "OptimizerGroupSummary",
    "OptimizerParamGroup",
    "SpecAdamWConfig",
    "build_adamw_parameter_groups",
    "create_adamw_optimizer",
    "scaled_learning_rate",
]
