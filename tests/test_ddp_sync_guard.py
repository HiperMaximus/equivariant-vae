# Copyright 2026 HiperMaximus
"""Tests for the shared DDP parameter-sync guard."""

from __future__ import annotations

import math

import pytest
import torch

from eqvae.training import ddp_sync_guard
from eqvae.training.ddp_sync_guard import (
    assert_ddp_parameters_in_sync,
    parameter_fingerprint,
)

_DDP_WORLD_SIZE = 2


def test_parameter_fingerprint_is_the_two_moment_sum() -> None:
    """The fingerprint is the deterministic (sum, sum-of-squares) over all params."""
    model = torch.nn.Linear(2, 2)

    fingerprint = parameter_fingerprint(model)

    assert fingerprint == parameter_fingerprint(model)
    expected_sum = sum(
        float(parameter.detach().double().sum().item())
        for parameter in model.parameters()
    )
    assert math.isclose(fingerprint[0], expected_sum)


def test_assert_ddp_parameters_in_sync_passes_or_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Identical fingerprints pass; a divergent one raises on every rank."""
    model = torch.nn.Linear(2, 2)

    def agree(gathered: list[object], obj: object) -> None:
        gathered[0] = obj
        gathered[1] = obj

    monkeypatch.setattr(ddp_sync_guard.dist, "all_gather_object", agree)
    assert_ddp_parameters_in_sync(model, world_size=_DDP_WORLD_SIZE)

    def disagree(gathered: list[object], obj: object) -> None:
        gathered[0] = obj
        gathered[1] = (1.0e30, 2.0e30)

    monkeypatch.setattr(ddp_sync_guard.dist, "all_gather_object", disagree)
    with pytest.raises(RuntimeError, match="divergent parameters"):
        assert_ddp_parameters_in_sync(model, world_size=_DDP_WORLD_SIZE)


def test_assert_ddp_parameters_in_sync_treats_identical_nan_as_synced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bit-identical NaN params across ranks are in sync, not a spurious desync."""
    model = torch.nn.Linear(2, 2)

    def gather_distinct_nan(gathered: list[object], obj: object) -> None:
        del obj
        gathered[0] = (float("nan"), float("nan"))
        gathered[1] = (float("nan"), float("nan"))

    monkeypatch.setattr(ddp_sync_guard.dist, "all_gather_object", gather_distinct_nan)
    assert_ddp_parameters_in_sync(model, world_size=_DDP_WORLD_SIZE)

    def gather_nan_vs_finite(gathered: list[object], obj: object) -> None:
        del obj
        gathered[0] = (float("nan"), float("nan"))
        gathered[1] = (0.0, 0.0)

    monkeypatch.setattr(ddp_sync_guard.dist, "all_gather_object", gather_nan_vs_finite)
    with pytest.raises(RuntimeError, match="divergent parameters"):
        assert_ddp_parameters_in_sync(model, world_size=_DDP_WORLD_SIZE)
