# Copyright 2026 HiperMaximus
"""Tests for shared compiled-fastpath AMP mechanics."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import Mock

import torch

from eqvae.training import fastpath_precision
from eqvae.training.selected_runtime import (
    EXPECTED_RUNNER_AMP_GRAD_SCALER_INIT_SCALE,
)

if TYPE_CHECKING:
    import pytest


def test_build_fastpath_grad_scaler_uses_runner_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The probe scaler uses the runner's device, initial scale, and enabled flag.

    The fp16 pretest, VRAM screen, and dual-T4 measurement must begin with the exact
    scaler contract consumed by the selected-runtime runner. The constructor spy makes
    a changed device, default scale, or dropped enablement observable without CUDA.
    """
    calls: list[tuple[str, float, bool]] = []
    sentinel = object()

    def fake_scaler(
        device: str,
        *,
        init_scale: float,
        enabled: bool,
    ) -> object:
        calls.append((device, init_scale, enabled))
        return sentinel

    monkeypatch.setattr(fastpath_precision, "GradScaler", fake_scaler)

    enabled_scaler = fastpath_precision.build_fastpath_grad_scaler(enabled=True)
    disabled_scaler = fastpath_precision.build_fastpath_grad_scaler(enabled=False)

    assert enabled_scaler is sentinel
    assert disabled_scaler is sentinel
    assert calls == [
        ("cuda", EXPECTED_RUNNER_AMP_GRAD_SCALER_INIT_SCALE, True),
        ("cuda", EXPECTED_RUNNER_AMP_GRAD_SCALER_INIT_SCALE, False),
    ]


def test_run_fastpath_optimizer_step_preserves_amp_order_and_reports_skip(  # noqa: C901
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AMP follows scale/backward, unscale, clip, step, update and reports a skip.

    The order is the behavioral contract shared with the selected-runtime runner: clip
    must see unscaled gradients and a scale backoff means the optimizer update was
    skipped. Reordering or bypassing any operation changes the event trace; returning a
    constant skip flag fails the final assertion.
    """
    events: list[str] = []
    clip_calls: list[tuple[object, float, bool]] = []
    parameters = (torch.nn.Parameter(torch.tensor(1.0)),)

    class _Loss:
        def __init__(self, event_log: list[str]) -> None:
            self.event_log = event_log

        def backward(self) -> None:
            self.event_log.append("backward")

    class _ScaledLoss:
        def __init__(self, event_log: list[str]) -> None:
            self.event_log = event_log

        def backward(self) -> None:
            self.event_log.append("scaled_backward")

    class _Scaler:
        def __init__(self, event_log: list[str]) -> None:
            self.event_log = event_log
            self.scale_value = 16.0

        def get_scale(self) -> float:
            return self.scale_value

        def scale(self, loss: object) -> _ScaledLoss:
            del loss
            self.event_log.append("scale")
            return _ScaledLoss(self.event_log)

        def unscale_(self, optimizer: object) -> None:
            del optimizer
            self.event_log.append("unscale")

        def step(self, optimizer: object) -> None:
            del optimizer
            self.event_log.append("scaler_step")

        def update(self) -> None:
            self.event_log.append("update")
            self.scale_value = 8.0

    class _Optimizer:
        def __init__(self, event_log: list[str]) -> None:
            self.event_log = event_log

        def step(self) -> None:
            self.event_log.append("optimizer_step")

    def fake_clip(
        parameters: object,
        max_norm: float,
        *,
        foreach: bool,
    ) -> torch.Tensor:
        clip_calls.append((parameters, max_norm, foreach))
        events.append("clip")
        return torch.tensor(1.0)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", fake_clip)

    skipped = fastpath_precision.run_fastpath_optimizer_step(
        loss=cast("torch.Tensor", _Loss(events)),
        optimizer=cast("torch.optim.Optimizer", _Optimizer(events)),
        parameters=parameters,
        scaler=cast("torch.amp.GradScaler", _Scaler(events)),
        grad_scaler_enabled=True,
        gradient_clip_global_norm=1.0,
        gradient_clip_foreach=True,
    )

    assert skipped is True
    assert events == [
        "scale",
        "scaled_backward",
        "unscale",
        "clip",
        "scaler_step",
        "update",
    ]
    assert clip_calls == [(list(parameters), 1.0, True)]


def test_run_fastpath_optimizer_step_keeps_fp32_path_unscaled(  # noqa: C901
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fp32 path performs ordinary backward/clip/step and cannot report AMP skip.

    This guards the shared helper's non-AMP branch: accidentally scaling fp32 or using
    ``scaler.step`` would enter a method that deliberately fails, while removing the
    direct optimizer update changes the trace.
    """
    events: list[str] = []
    clip_calls: list[tuple[object, float, bool]] = []
    parameters = (torch.nn.Parameter(torch.tensor(1.0)),)

    class _Loss:
        def __init__(self, event_log: list[str]) -> None:
            self.event_log = event_log

        def backward(self) -> None:
            self.event_log.append("backward")

    class _Scaler:
        def __init__(self, event_log: list[str]) -> None:
            self.event_log = event_log

        def get_scale(self) -> float:
            self.event_log.append("unexpected_get_scale")
            return 1.0

        def scale(self, loss: object) -> object:
            del loss
            self.event_log.append("unexpected_scale")
            return object()

        def unscale_(self, optimizer: object) -> None:
            del optimizer
            self.event_log.append("unexpected_unscale")

        def step(self, optimizer: object) -> None:
            del optimizer
            self.event_log.append("unexpected_scaler_step")

        def update(self) -> None:
            self.event_log.append("unexpected_update")

    class _Optimizer:
        def __init__(self, event_log: list[str]) -> None:
            self.event_log = event_log

        def step(self) -> None:
            self.event_log.append("optimizer_step")

    def fake_clip(
        parameters: object,
        max_norm: float,
        *,
        foreach: bool,
    ) -> torch.Tensor:
        clip_calls.append((parameters, max_norm, foreach))
        events.append("clip")
        return torch.tensor(1.0)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", fake_clip)

    skipped = fastpath_precision.run_fastpath_optimizer_step(
        loss=cast("torch.Tensor", _Loss(events)),
        optimizer=cast("torch.optim.Optimizer", _Optimizer(events)),
        parameters=parameters,
        scaler=cast("torch.amp.GradScaler", _Scaler(events)),
        grad_scaler_enabled=False,
        gradient_clip_global_norm=1.0,
        gradient_clip_foreach=True,
    )

    assert skipped is False
    assert events == ["backward", "clip", "optimizer_step"]
    assert clip_calls == [(list(parameters), 1.0, True)]


def test_timed_amp_step_does_not_materialize_scale_on_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Timed AMP updates can defer skip observation to one block-boundary read."""
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    loss = parameter.square()

    scaler = Mock()
    scaler.get_scale.side_effect = AssertionError  # pyright: ignore[reportAny]
    scaler.scale.return_value = loss  # pyright: ignore[reportAny]
    monkeypatch.setattr(
        torch.nn.utils,
        "clip_grad_norm_",
        Mock(return_value=torch.tensor(1.0)),
    )
    skipped = fastpath_precision.run_fastpath_optimizer_step(
        loss=loss,
        optimizer=cast("torch.optim.Optimizer", object()),
        parameters=(parameter,),
        scaler=cast("torch.amp.GradScaler", scaler),
        grad_scaler_enabled=True,
        gradient_clip_global_norm=1.0,
        gradient_clip_foreach=True,
        observe_skip=False,
    )
    assert skipped is False


def test_optimizer_metrics_and_sampled_update_probe_share_real_step() -> None:
    """The common optimizer seam retains finite norm and honest update evidence."""
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    optimizer = torch.optim.SGD((parameter,), lr=0.1)
    scaler = fastpath_precision.build_fastpath_grad_scaler(enabled=False)
    update_probe = fastpath_precision.clone_fastpath_update_probe(
        torch.nn.ParameterList((parameter,)),
    )

    result = fastpath_precision.run_fastpath_optimizer_step_with_metrics(
        loss=parameter.square().sum(),
        optimizer=optimizer,
        parameters=(parameter,),
        scaler=scaler,
        grad_scaler_enabled=False,
        gradient_clip_global_norm=10.0,
        gradient_clip_foreach=True,
        observe_skip=False,
    )
    update_norm = fastpath_precision.fastpath_update_probe_norm(update_probe)

    assert result.step_skipped is False
    assert torch.allclose(result.grad_norm, torch.sqrt(torch.tensor(20.0)))
    assert result.nonfinite_count.item() == 0
    assert update_norm.item() > 0.0


def test_shared_transport_eps_and_metric_writes_match_hot_loop_work() -> None:
    """Probe/runner utilities perform fused layout and every telemetry write."""
    batch = torch.zeros((2, 3, 4, 4), dtype=torch.uint8)
    transferred = fastpath_precision.transfer_fastpath_uint8(
        batch,
        device=torch.device("cpu"),
        memory_format="channels_last",
        non_blocking=True,
    )
    eps = torch.tensor([0.0, -1.0, 3.0, 0.0])
    zero_fraction, abs_mean = fastpath_precision.fastpath_eps_metrics(eps)
    row = torch.empty((fastpath_precision.FASTPATH_METRIC_COUNT,))
    scalars = tuple(
        torch.tensor(float(index))
        for index in range(fastpath_precision.FASTPATH_METRIC_COUNT)
    )
    fastpath_precision.write_fastpath_metric_row(row, scalars)

    assert transferred.is_contiguous(memory_format=torch.channels_last)
    assert torch.allclose(zero_fraction, torch.tensor(0.5))
    assert torch.allclose(abs_mean, torch.tensor(1.0))
    assert torch.equal(row, torch.arange(16, dtype=torch.float32))
