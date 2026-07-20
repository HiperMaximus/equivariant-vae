# Copyright 2026 HiperMaximus
"""Tests for the shared compiled fast-path recipe helpers (Spec 0011 S10)."""

from __future__ import annotations

import contextlib
import math
from typing import TYPE_CHECKING, cast

import torch
import torch._dynamo as torch_dynamo  # noqa: PLC2701
from torch._inductor import config as inductor_config  # noqa: PLC2701

from eqvae.models.non_equivariant_vae import build_non_equivariant_vae
from eqvae.training import fastpath_recipe
from eqvae.training.fastpath_recipe import (
    apply_cudnn_flags,
    apply_fastpath_dynamo_config,
    build_fastpath_optimizer,
    compiled_autograd_context,
    wrap_fastpath_ddp,
)
from eqvae.training.optim import SpecAdamWConfig

if TYPE_CHECKING:
    import pytest

EXPECTED_OPTIMIZER_GROUPS = 3


def test_build_fastpath_optimizer_uses_the_grouped_semantic_groups() -> None:
    """The fast-path optimizer is grouped, never a flat ungrouped parameter set.

    A flat ``model.parameters()`` optimizer would weight-decay every parameter and
    apply one learning rate; the grouped builder decays only conv weights, zeroes
    decay for the no-decay and gate parameters, and halves the gate learning rate.
    """
    model = build_non_equivariant_vae()

    optimizer = build_fastpath_optimizer(model, config=SpecAdamWConfig(fused=True))

    by_name = {cast("str", group["name"]): group for group in optimizer.param_groups}
    assert set(by_name) == {"decay", "no_decay", "gate_no_decay"}
    assert len(optimizer.param_groups) == EXPECTED_OPTIMIZER_GROUPS
    config = SpecAdamWConfig()
    assert math.isclose(
        cast("float", by_name["decay"]["weight_decay"]),
        config.weight_decay,
    )
    assert math.isclose(
        cast("float", by_name["no_decay"]["weight_decay"]),
        0.0,
        abs_tol=0.0,
    )
    assert math.isclose(
        cast("float", by_name["gate_no_decay"]["weight_decay"]),
        0.0,
        abs_tol=0.0,
    )
    assert math.isclose(
        cast("float", by_name["gate_no_decay"]["lr"]),
        config.learning_rate * config.gate_lr_multiplier,
    )
    # Fused is CUDA-gated inside create_adamw_optimizer, so a CPU model keeps fused
    # as None (never an unconditional fused kernel that would raise on CPU).
    assert all(
        cast("object", group["fused"]) is None for group in optimizer.param_groups
    )


def test_apply_cudnn_flags_sets_the_process_global_backend_flags() -> None:
    """The cuDNN benchmark/determinism flags are written to their global backend module.

    Both arguments are honored independently -- both directions are asserted -- so a
    helper that ignored ``deterministic`` or hardcoded a value is caught
    (mutation-proof).
    """
    original_benchmark = torch.backends.cudnn.benchmark
    original_deterministic = torch.backends.cudnn.deterministic
    try:
        apply_cudnn_flags(benchmark=True, deterministic=False)
        assert torch.backends.cudnn.benchmark is True
        assert torch.backends.cudnn.deterministic is False
        apply_cudnn_flags(benchmark=False, deterministic=True)
        assert torch.backends.cudnn.benchmark is False
        assert torch.backends.cudnn.deterministic is True
    finally:
        torch.backends.cudnn.benchmark = original_benchmark
        torch.backends.cudnn.deterministic = original_deterministic


def test_apply_fastpath_dynamo_config_sets_the_process_global_knobs() -> None:
    """The dynamo/inductor knobs are written to their process-global config objects."""
    original_optimize_ddp = cast("object", torch_dynamo.config.optimize_ddp)
    original_compiled_autograd = cast("object", torch_dynamo.config.compiled_autograd)
    original_reorder = cast(
        "object",
        inductor_config.reorder_for_compute_comm_overlap,
    )
    try:
        apply_fastpath_dynamo_config(
            optimize_ddp="python_reducer",
            compiled_autograd=True,
            reorder_compute_comm_overlap=True,
        )
        assert cast("object", torch_dynamo.config.optimize_ddp) == "python_reducer"
        assert cast("object", torch_dynamo.config.compiled_autograd) is True
        assert cast("object", inductor_config.reorder_for_compute_comm_overlap) is True
    finally:
        torch_dynamo.config.optimize_ddp = original_optimize_ddp
        torch_dynamo.config.compiled_autograd = original_compiled_autograd
        inductor_config.reorder_for_compute_comm_overlap = original_reorder


def test_wrap_fastpath_ddp_forwards_every_recipe_knob_to_ddp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every recipe knob reaches ``DistributedDataParallel`` unchanged.

    The measured recipe only transfers to the real run if the DDP construction is
    identical, so the exact keyword set is the bit-identity contract.
    """
    captured: dict[str, object] = {}

    def fake_ddp(model: object, **kwargs: object) -> object:
        captured["model"] = model
        captured["kwargs"] = kwargs
        return "wrapped"

    monkeypatch.setattr(fastpath_recipe, "DistributedDataParallel", fake_ddp)
    model = build_non_equivariant_vae()

    result = wrap_fastpath_ddp(
        model,
        local_rank=1,
        static_graph=True,
        gradient_as_bucket_view=True,
        broadcast_buffers=False,
        find_unused_parameters=False,
        bucket_cap_mb=25,
    )

    assert result == "wrapped"
    assert captured["model"] is model
    assert captured["kwargs"] == {
        "device_ids": [1],
        "output_device": 1,
        "static_graph": True,
        "gradient_as_bucket_view": True,
        "broadcast_buffers": False,
        "find_unused_parameters": False,
        "bucket_cap_mb": 25,
    }


def test_compiled_autograd_context_is_a_noop_when_disabled() -> None:
    """A recipe without compiled autograd pays nothing (Spec 0011 S16).

    The ``ddp_optimizer`` winner pairs with ``compiled_autograd=False``, so the runner's
    eager backward must run under a plain no-op context.
    """
    ctx = compiled_autograd_context(enabled=False)

    assert isinstance(ctx, contextlib.nullcontext)
    with ctx:
        pass


def test_compiled_autograd_context_engages_when_enabled() -> None:
    """When enabled the context is a real (non-no-op) compiled-autograd scope."""
    ctx = compiled_autograd_context(enabled=True)

    assert not isinstance(ctx, contextlib.nullcontext)
    with ctx:
        pass
