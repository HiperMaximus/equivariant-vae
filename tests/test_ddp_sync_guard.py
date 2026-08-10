# Copyright 2026 HiperMaximus
"""Tests for the shared DDP parameter-sync guard."""

from __future__ import annotations

import math
from typing import cast

import pytest
import torch

from eqvae.training import ddp_sync_guard
from eqvae.training.ddp_sync_guard import (
    assert_ddp_parameters_exactly_in_sync,
    assert_ddp_parameters_in_sync,
    parameter_fingerprint,
    parameter_sha256,
)

_DDP_WORLD_SIZE = 2
_SHA256_HEX_LENGTH = 64


def test_parameter_fingerprint_is_the_two_moment_sum() -> None:
    """Both fingerprint moments are derived from every parameter value.

    The derived two-moment relationship makes equal sums insufficient to hide rank
    drift; it catches replacing the square-sum producer with a constant or a duplicate
    of the first moment without pinning random initialization values.
    """
    model = torch.nn.Linear(2, 2)

    fingerprint = parameter_fingerprint(model)

    assert fingerprint == parameter_fingerprint(model)
    expected_sum = sum(
        float(parameter.detach().double().sum().item())
        for parameter in model.parameters()
    )
    expected_square_sum = sum(
        float(parameter.detach().double().square().sum().item())
        for parameter in model.parameters()
    )
    assert math.isclose(fingerprint[0], expected_sum)
    assert math.isclose(fingerprint[1], expected_square_sum)


def test_assert_ddp_parameters_in_sync_passes_or_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second-moment-only rank divergence triggers the deliberate fail-fast guard.

    Equal first moments can still describe different models, so the policy must compare
    both fingerprint components on every rank. This catches a comparison that silently
    ignores the square-sum while avoiding frozen parameter values.
    """
    model = torch.nn.Linear(2, 2)

    def agree(gathered: list[object], obj: object) -> None:
        gathered[0] = obj
        gathered[1] = obj

    monkeypatch.setattr(ddp_sync_guard.dist, "all_gather_object", agree)
    assert_ddp_parameters_in_sync(model, world_size=_DDP_WORLD_SIZE)

    def disagree(gathered: list[object], obj: object) -> None:
        fingerprint = cast("tuple[float, float]", obj)
        gathered[0] = fingerprint
        gathered[1] = (fingerprint[0], fingerprint[1] + 1.0)

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


def test_exact_parameter_digest_distinguishes_moment_collision() -> None:
    """The untimed digest distinguishes models with identical sum and sum-square."""
    left = torch.nn.Linear(2, 1, bias=False)
    right = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        left.weight.copy_(torch.tensor([[1.0, -1.0]]))
        right.weight.copy_(torch.tensor([[-1.0, 1.0]]))

    assert parameter_fingerprint(left) == parameter_fingerprint(right)
    assert parameter_sha256(left) != parameter_sha256(right)


def test_exact_parameter_digest_accepts_scalar_parameter() -> None:
    """A zero-dimensional learnable gate remains hashable by the exact proof."""

    class ScalarModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate = torch.nn.Parameter(torch.tensor(1.0))

    digest = parameter_sha256(ScalarModel())

    assert len(digest) == _SHA256_HEX_LENGTH


def test_exact_parameter_sync_fails_closed_on_missing_or_distinct_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Untimed proof rejects both malformed and byte-distinct gathered rank values."""
    model = torch.nn.Linear(2, 2)

    def missing(gathered: list[object], obj: object) -> None:
        gathered[0] = obj

    monkeypatch.setattr(ddp_sync_guard.dist, "all_gather_object", missing)
    with pytest.raises(RuntimeError, match="divergent exact parameter bytes"):
        assert_ddp_parameters_exactly_in_sync(model, world_size=_DDP_WORLD_SIZE)

    def distinct(gathered: list[object], obj: object) -> None:
        gathered[0] = obj
        gathered[1] = "0" * 64

    monkeypatch.setattr(ddp_sync_guard.dist, "all_gather_object", distinct)
    with pytest.raises(RuntimeError, match="divergent exact parameter bytes"):
        assert_ddp_parameters_exactly_in_sync(model, world_size=_DDP_WORLD_SIZE)
