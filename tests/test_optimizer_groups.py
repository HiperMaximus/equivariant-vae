# Copyright 2026 HiperMaximus
"""Tests for spec 0001 semantic optimizer groups."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from torch import nn

from eqvae.models.activations import GatedScalarActivation
from eqvae.models.non_equivariant_vae import build_non_equivariant_vae
from eqvae.training.optim import build_adamw_parameter_groups

if TYPE_CHECKING:
    from collections.abc import Iterable

EXPECTED_OPTIMIZER_GROUPS = 3


def test_semantic_adamw_groups_cover_every_trainable_parameter_once() -> None:
    """The model's trainable parameters are neither dropped nor duplicated."""
    model = build_non_equivariant_vae()

    groups, summary = build_adamw_parameter_groups(model)

    assert summary.parameter_group_count == EXPECTED_OPTIMIZER_GROUPS
    assert summary.all_trainable_parameters_covered_once is True
    assert summary.gate_parameters_in_gate_no_decay_group is True
    assert {group["name"] for group in groups} == {
        "decay",
        "gate_no_decay",
        "no_decay",
    }


def test_conv_weights_decay_and_gate_parameters_use_gate_group() -> None:
    """Conv kernels decay; learned gate `a,b` parameters do not decay."""
    model = build_non_equivariant_vae()
    groups, _summary = build_adamw_parameter_groups(model)
    group_ids = {
        group["name"]: {id(parameter) for parameter in group["params"]}
        for group in groups
    }
    module_iter = cast("Iterable[tuple[str, nn.Module]]", model.named_modules())
    named_modules: dict[str, nn.Module] = dict(module_iter)
    named_parameters = dict(model.named_parameters())

    assert id(named_parameters["stem_conv.weight"]) in group_ids["decay"]
    assert id(named_parameters["output_head.bias"]) in group_ids["no_decay"]
    for module_name, module in named_modules.items():
        if isinstance(module, GatedScalarActivation):
            assert (
                id(named_parameters[f"{module_name}.a"]) in group_ids["gate_no_decay"]
            )
            assert (
                id(named_parameters[f"{module_name}.b"]) in group_ids["gate_no_decay"]
            )
    assert all(
        not isinstance(module, nn.BatchNorm2d) for module in named_modules.values()
    )


def test_semantic_adamw_groups_reject_duplicate_parameter_references() -> None:
    """A tied parameter cannot be reported as covered exactly once."""
    model = nn.Module()
    shared = nn.Parameter(torch.ones(1))
    model.register_parameter("left", shared)
    model.register_parameter("right", shared)

    _groups, summary = build_adamw_parameter_groups(model)

    assert summary.all_trainable_parameters_covered_once is False
