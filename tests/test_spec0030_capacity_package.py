# Copyright 2026 HiperMaximus
# pyright: reportAny=false, reportIndexIssue=false, reportPrivateUsage=false
"""Acceptance tests for the portable Spec 0030 capacity package."""

from __future__ import annotations

import importlib.util
import inspect
import json
import shutil
import sys
import textwrap
from copy import deepcopy
from itertools import starmap
from typing import TYPE_CHECKING, cast

import pytest
import torch
from scripts import build_wsi45630_local_global_capacity as builder
from torch import nn

from eqvae.kaggle_resources import create_portable_kernel_snapshot

if TYPE_CHECKING:
    from pathlib import Path
    from types import ModuleType


def _module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("spec0030_runtime_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def test_builder_emits_exact_tracked_account_portable_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invariant: a clone receives exact model bytes and mixed-owner sources."""
    output = tmp_path / "package"
    monkeypatch.setattr(builder, "DEFAULT_ROOT", output)

    contract = builder.build()
    kernel = output / "kernel"
    assert {path.name for path in kernel.iterdir()} == builder.PACKAGE_FILES
    rendered = (kernel / "run.py").read_text(encoding="utf-8")
    embedded_block = rendered.split("# BEGIN EMBEDDED SPEC0026 MODEL\n", 1)[1].split(
        "# END EMBEDDED SPEC0026 MODEL\n",
        1,
    )[0]
    assert textwrap.dedent(embedded_block.removeprefix("if False:\n")) == (
        builder._embedded_model_source(builder.MODEL_PATH)  # noqa: SLF001
    )
    assert contract["model"]["parameter_count"] == builder.EXPECTED_PARAMETER_COUNT
    assert contract["input_dataset"] == {
        "reference": builder.INPUT_DATASET_REFERENCE,
        "version": 1,
        "contract_sha256": builder.INPUT_CONTRACT_SHA256,
        "pointer_sha256": builder.POINTER_SHA256,
    }

    portable = tmp_path / "portable"
    snapshot = create_portable_kernel_snapshot(kernel, portable, actor="professor")
    metadata = json.loads((portable / "kernel-metadata.json").read_text())
    assert metadata["id"] == "professor/eqvae-wsi45630-local-global-mil-capacity"
    assert metadata["dataset_sources"] == [builder.INPUT_DATASET_REFERENCE]
    assert metadata["kernel_sources"] == list(builder.KERNEL_SOURCES)
    assert snapshot.source_locators["kernel_sources"] == list(builder.KERNEL_SOURCES)

    unexpected = kernel / "unexpected"
    unexpected.mkdir()
    with pytest.raises(ValueError, match="allow-list"):
        builder.validate()
    unexpected.rmdir()
    (kernel / "run.py").unlink()
    (kernel / "run.py").symlink_to(builder.ROOT / builder.TEMPLATE_PATH)
    with pytest.raises(ValueError, match="allow-list"):
        builder.validate()


def test_runtime_authenticates_package_and_gates_both_steps_atomically() -> None:
    """Invariant: neither paired optimizer advances before both gradients pass."""
    package = builder.ROOT / builder.DEFAULT_ROOT / "kernel"
    runtime = _module(package / "run.py")
    contract = runtime.resolve_package()
    assert contract["execution"] == {
        "model_devices": {"normal_vae": 0, "so2_vae": 1},
        "direct_complete_bag_only": True,
        "checkpointing": False,
        "fallback": None,
        "paired_step_atomicity": "both_finite_before_either_step",
    }
    source = inspect.getsource(runtime.run_probe)
    assert source.index("finite_checks =") < source.index(
        "commit_paired_optimizer_steps",
    )
    assert source.count('for phase in ("warmup", "measured")') == 1
    assert "activation_checkpoint" not in source
    assert 'result["active_stage"] = {}' in source
    assert 'result["live_stage_memory"] = stage_memory' in source
    assert "logits_records" not in source
    assert "torch" not in runtime.__dict__
    assert "make_paired_local_global_mil_models" not in runtime.__dict__


def test_paired_optimizer_commit_rolls_back_first_branch_on_second_failure() -> None:
    """Invariant: a second-branch commit failure restores both branch states."""
    runtime = _module(builder.ROOT / builder.DEFAULT_ROOT / "kernel/run.py")
    models = [nn.Linear(3, 2), nn.Linear(3, 2)]
    optimizers = [torch.optim.AdamW(model.parameters(), lr=0.1) for model in models]
    for model, optimizer in zip(models, optimizers, strict=True):
        runtime.materialize_optimizer_state(model, optimizer)
        model(torch.ones(1, 3)).sum().backward()
    before_models = [deepcopy(model.state_dict()) for model in models]
    before_steps = [
        [state["step"].clone() for state in optimizer.state.values()]
        for optimizer in optimizers
    ]

    class FakeScaler:
        def __init__(self, *, fail: bool) -> None:
            self.fail = fail

        def step(self, optimizer: torch.optim.Optimizer) -> None:
            if self.fail:
                message = "second commit failed"
                raise RuntimeError(message)
            optimizer.step()

    with pytest.raises(RuntimeError, match="second commit failed"):
        runtime.commit_paired_optimizer_steps(
            models,
            optimizers,
            [FakeScaler(fail=False), FakeScaler(fail=True)],
        )

    for model, expected in zip(models, before_models, strict=True):
        for name, value in model.state_dict().items():
            assert torch.equal(value, expected[name])
    for optimizer, expected_steps in zip(optimizers, before_steps, strict=True):
        observed = [state["step"] for state in optimizer.state.values()]
        assert all(
            starmap(torch.equal, zip(observed, expected_steps, strict=True)),
        )


def test_input_authority_requires_exact_contract_and_file_receipt(
    tmp_path: Path,
) -> None:
    """Invariant: every verified input file remains bound to the pinned contract."""
    receipt = tmp_path / builder.INPUT_RECEIPT_PATH
    contract = tmp_path / builder.INPUT_CONTRACT_PATH
    receipt.parent.mkdir(parents=True)
    contract.parent.mkdir(parents=True)
    shutil.copy2(builder.ROOT / builder.INPUT_RECEIPT_PATH, receipt)
    shutil.copy2(builder.ROOT / builder.INPUT_CONTRACT_PATH, contract)
    builder._validate_input_authority(tmp_path)  # noqa: SLF001

    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["files"]["probe/physical_parts.csv"]["sha256"] = "0" * 64
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="file provenance"):
        builder._validate_input_authority(tmp_path)  # noqa: SLF001


def test_runtime_gradient_gate_requires_every_finite_parameter() -> None:
    """Invariant: missing or nonfinite gradients block the atomic paired step."""
    runtime = _module(builder.ROOT / builder.DEFAULT_ROOT / "kernel/run.py")
    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 1))
    passed, count, invalid = runtime.gradients_are_finite(model)
    assert not passed
    assert count == 0
    assert invalid == "0.weight"

    model(torch.ones(2, 3)).sum().backward()
    passed, count, invalid = runtime.gradients_are_finite(model)
    assert passed
    assert count == sum(1 for _ in model.parameters())
    assert invalid is None

    cast("torch.Tensor", model[1].bias.grad)[0] = torch.nan
    passed, _, invalid = runtime.gradients_are_finite(model)
    assert not passed
    assert invalid == "1.bias"
