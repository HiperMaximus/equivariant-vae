# Copyright 2026 HiperMaximus
# pyright: reportAny=false, reportIndexIssue=false, reportPrivateUsage=false
# ruff: noqa: PLR2004
"""Focused contracts for the disposable Spec 0034 Kaggle package."""

from __future__ import annotations

import importlib.util
import json
import sys
import textwrap
from typing import TYPE_CHECKING

import pytest
import torch
from scripts import build_wsi45630_full_compile_probe as builder

from eqvae.kaggle_resources import create_portable_kernel_snapshot

if TYPE_CHECKING:
    from pathlib import Path
    from types import ModuleType


def _module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("spec0034_runtime", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_builder_emits_exact_account_portable_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The package embeds exact model/candidate bytes and preserves source owners."""
    output = tmp_path / "package"
    monkeypatch.setattr(builder, "DEFAULT_ROOT", output)
    contract = builder.build()
    kernel = output / "kernel"
    assert {path.name for path in kernel.iterdir()} == builder.PACKAGE_FILES
    runtime = _module(kernel / "run.py")
    resolved, model, candidate = runtime.resolve_package()
    assert resolved == contract
    assert model == builder._embedded_model_source(builder.MODEL_PATH)  # noqa: SLF001
    assert "LowPrecisionGroupNorm" not in model
    assert "LowPrecisionLayerNorm" not in model
    assert candidate == builder._embedded_candidate_source(  # noqa: SLF001
        builder.CANDIDATE_PATH,
    )
    assert "WholeBagFixed25Attention" in candidate
    assert "fixed25_inductor_attention" in candidate
    runtime.activate_sources(model, candidate)
    network = runtime.LocalGlobalMILClassifier()
    before = network.state_dict()
    runtime.use_whole_bag_fixed25_attention(network)
    assert all(
        type(block.attention).__name__ == "WholeBagFixed25Attention"
        for block in network.local_blocks
    )
    assert before.keys() == network.state_dict().keys()

    portable = tmp_path / "portable"
    snapshot = create_portable_kernel_snapshot(kernel, portable, actor="professor")
    metadata = json.loads((portable / "kernel-metadata.json").read_text())
    assert metadata["id"] == (
        "professor/eqvae-wsi45630-full-compiled-fixed25-mil-probe"
    )
    assert metadata["dataset_sources"] == [builder.INPUT_DATASET_REFERENCE]
    assert metadata["kernel_sources"] == list(builder.KERNEL_SOURCES)
    assert snapshot.source_locators["kernel_sources"] == list(builder.KERNEL_SOURCES)


def test_contract_locks_full_compilation_and_independent_branches() -> None:
    """The execution recipe contains the requested dynamic full-network gate."""
    contract = builder._contract(builder.ROOT)  # noqa: SLF001
    assert contract["authorization"] == "spec0034_pinned_torch_retry_v7_authorized"
    assert contract["scope"] == (
        "capacity_optimization_only_not_learning_or_evaluation"
    )
    assert contract["compile"] == {
        "backend": "inductor",
        "mode": "max-autotune-no-cudagraphs",
        "fullgraph": True,
        "recompile_limit": 3,
        "dynamic_axes": "N_only_permissive_cached_specialization",
        "optimizer": "grad_scaler_and_native_fused_adamw_eager",
    }
    assert contract["runtime_dependency"] == {
        "torch": "2.14.0",
        "cuda_wheel": "cu130",
        "index_url": "https://download.pytorch.org/whl/cu130",
        "install_scope": "torch_only_no_domain_libraries",
    }
    assert contract["correctness"] == {
        "criterion": "compiler_training_effect_lte_max_amp_or_repeat_effect",
        "steps": 5,
        "precision_control": "same_fixed25_eager_fp32",
        "repeat_control": "same_fixed25_eager_amp",
        "decision_units": "per_parameter_optimizer_state_and_behavior_per_step",
        "amp_skip_gate": "eager_replay_compiled_histories_must_match",
    }
    assert contract["execution"] == {
        "branch_devices": {"normal_vae": 0, "so2_vae": 1},
        "compile_warmups": 2,
        "dynamic_reuse_bag_size": 64,
        "measured_steps": 5,
        "measured_committed_steps_required": 5,
        "branch_failure_policy": "record_and_continue_other_branch",
        "benchmark_workload": "single_wsi_repeated_not_epoch_requeue",
        "checkpointing": False,
        "cudagraphs": False,
        "fallback": None,
    }
    assert contract["optimizer"]["fused"] is True
    assert contract["optimizer"]["capturable"] is True


def test_runtime_proves_compile_memory_and_numerical_contracts() -> None:
    """The generated source measures rather than merely naming each invariant."""
    rendered = builder._render(  # noqa: SLF001
        builder.ROOT,
        contract=builder._contract(builder.ROOT),  # noqa: SLF001
    ).decode()
    required = (
        "full_model_correctness",
        "use_whole_bag_fixed25_attention(model)",
        '"fullgraph": fullgraph',
        'kwargs["recompile_limit"] = 3',
        'PINNED_TORCH_VERSION = "2.14.0"',
        'PINNED_TORCH_CUDA = "13.0"',
        'PINNED_TORCH_INDEX = "https://download.pytorch.org/whl/cu130"',
        "install_pinned_torch()",
        "torch._dynamo.maybe_mark_dynamic",
        "specialization_count not in {0, 1}",
        "torch.optim.AdamW(",
        "fused=True",
        "capturable=True",
        "torch.amp.GradScaler(",
        "scaler.scale(loss).backward()",
        "scaler.unscale_(optimizer)",
        "scaler.step(optimizer)",
        "scaler.update()",
        '"policy": "skip_update_and_continue"',
        "accumulate_training_effect",
        "correctness_training_step",
        "correctness_evaluation",
        "compiler_training_effect_lte_max_amp_or_repeat_effect",
        "torch.cuda.reset_peak_memory_stats",
        "conservative_reserved_headroom_bytes",
        "WARMUP_STEPS = 2",
        "MEASURED_STEPS = 5",
        'for index, name in enumerate(("normal_vae", "so2_vae"))',
        '"measured_committed_steps"] == MEASURED_STEPS',
        "except Exception as error:",
    )
    assert all(value in rendered for value in required)
    assert "torch._dynamo.mark_dynamic" not in rendered
    assert "GRADIENT_RELATIVE_L2" not in rendered
    assert "OUTPUT_TOLERANCE" not in rendered
    assert "LOSS_SCALE" not in rendered
    assert "torch._foreach_mul_" not in rendered
    assert '"--upgrade"' not in rendered
    assert '"torchvision"' not in rendered
    assert '"torchaudio"' not in rendered
    assert "strict = compile_callable(torch, update" not in rendered
    assert 'numerical_unique_graphs"] in {1, 2}' in rendered
    assert "cross_entropy" in rendered
    assert "activation_checkpoint" not in rendered
    assert "sealed" not in rendered.lower()
    assert "train_test" not in rendered


def test_full_bag_failures_are_isolated_and_skipped_measurements_reject() -> None:
    """One branch still runs after the other fails; timing requires real updates."""
    rendered = builder.TEMPLATE_PATH.read_text(encoding="utf-8")
    start = rendered.index("def run_probe(")
    end = rendered.index("\ndef install_pinned_torch(", start)
    probe = rendered[start:end]
    assert (
        probe.index("try:")
        < probe.index("run_branch(")
        < probe.index(
            "except Exception as error:",
        )
    )
    assert '"measured_committed_steps"] == MEASURED_STEPS' in probe


def test_training_effect_uses_measured_precision_and_repeat_controls() -> None:
    """Compiler impact is bounded by observed AMP or replay impact."""
    runtime = _module(builder.TEMPLATE_PATH)
    passing = {}
    runtime.accumulate_training_effect(
        torch,
        passing,
        "row",
        torch.tensor([0.0]),
        torch.tensor([2.0]),
        torch.tensor([2.5]),
        torch.tensor([1.5]),
    )
    assert runtime.finalize_training_effects(passing)["row"]["pass"] is True

    failing = {}
    runtime.accumulate_training_effect(
        torch,
        failing,
        "row",
        torch.tensor([0.0]),
        torch.tensor([2.0]),
        torch.tensor([2.5]),
        torch.tensor([-3.0]),
    )
    assert runtime.finalize_training_effects(failing)["row"]["pass"] is False

    exact = {}
    runtime.accumulate_training_effect(
        torch,
        exact,
        "row",
        torch.tensor([1.0]),
        torch.tensor([1.0]),
        torch.tensor([1.0]),
        torch.tensor([1.0]),
    )
    exact_row = runtime.finalize_training_effects(exact)["row"]
    assert not exact_row["accepted_training_effect"]
    assert not exact_row["compiler_effect"]
    assert exact_row["pass"] is True


@pytest.mark.parametrize(
    ("mutation", "compiled"),
    [
        ("equal_radius_opposite_direction", torch.tensor([0.9, 1.1])),
        ("wrong_scale", torch.tensor([4.0, 4.0])),
        ("skipped_update", torch.tensor([0.0, 0.0])),
        ("detached_path", torch.tensor([3.0, -3.0])),
    ],
)
def test_training_effect_gate_rejects_material_mutations(
    mutation: str,
    compiled: torch.Tensor,
) -> None:
    """Representative corrupt training states exceed the measured AMP effect."""
    del mutation
    runtime = _module(builder.TEMPLATE_PATH)
    totals = {}
    runtime.accumulate_training_effect(
        torch,
        totals,
        "mutated_parameter",
        torch.tensor([1.0, 1.0]),
        torch.tensor([1.1, 0.9]),
        torch.tensor([1.1, 0.9]),
        compiled,
    )
    assert not runtime.finalize_training_effects(totals)["mutated_parameter"]["pass"]


def test_training_effect_gate_rejects_nonfinite_control_state() -> None:
    """A NaN in any measured trajectory cannot evade the gate through max()."""
    runtime = _module(builder.TEMPLATE_PATH)
    totals = {}
    runtime.accumulate_training_effect(
        torch,
        totals,
        "state",
        torch.tensor([1.0]),
        torch.tensor([1.1]),
        torch.tensor([float("nan")]),
        torch.tensor([1.1]),
    )
    row = runtime.finalize_training_effects(totals)["state"]
    assert row["finite"] is False
    assert row["pass"] is False


def test_grad_scaler_uses_supported_dynamic_policy() -> None:
    """The probe uses GradScaler rather than manually scaling gradients."""
    runtime = _module(builder.TEMPLATE_PATH)
    assert pytest.approx(32768.0) == runtime.GRAD_SCALER_INIT_SCALE
    assert runtime.GRAD_SCALER_GROWTH_INTERVAL == 1_000_000
    disabled = runtime.make_grad_scaler(torch, enabled=False)
    assert disabled.is_enabled() is False
    assert pytest.approx(1.0) == disabled.get_scale()


def test_runtime_install_is_exact_torch_only_official_cuda_wheel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The probe cannot resolve a floating stack or unrelated domain packages."""
    runtime = _module(builder.TEMPLATE_PATH)
    calls: list[list[str]] = []
    monkeypatch.setattr(runtime.subprocess, "check_call", calls.append)
    runtime.install_pinned_torch()
    assert calls == [
        [
            runtime.sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "torch==2.14.0",
            "--index-url",
            "https://download.pytorch.org/whl/cu130",
        ],
    ]


def test_measured_step_uses_the_recommended_grad_scaler_order() -> None:
    """The timed path lets GradScaler unscale/check/skip the fused optimizer."""
    rendered = builder.TEMPLATE_PATH.read_text(encoding="utf-8")
    start = rendered.index("def timed_training_step(")
    end = rendered.index("\ndef summarize_steps(", start)
    step = rendered[start:end]
    assert step.index("scaler.scale(loss).backward()") < step.index(
        "scaler.step(optimizer)",
    )
    assert step.index("scaler.step(optimizer)") < step.index("scaler.update()")
    assert "scaler.unscale_(optimizer)" not in step


def test_optimizer_state_validator_rejects_nonfinite_moments() -> None:
    """The full-bag acceptance path requires complete finite optimizer state."""
    runtime = _module(builder.TEMPLATE_PATH)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()  # pyright: ignore[reportUnknownMemberType]
    runtime.validate_optimizer_state(torch, model, optimizer, 1)
    parameter = next(model.parameters())
    optimizer.state[parameter]["exp_avg"].fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="nonfinite"):
        runtime.validate_optimizer_state(torch, model, optimizer, 1)


def test_training_effect_groups_isolate_structural_attention_parameters() -> None:
    """Small topology parameters cannot hide inside a whole-model norm."""
    runtime = _module(builder.TEMPLATE_PATH)
    assert runtime.semantic_parameter_group("patch_encoder.layers.0.weight") == (
        "patch_encoder"
    )
    assert (
        runtime.semantic_parameter_group(
            "local_blocks.0.attention.null_key",
        )
        == "local_0_structure"
    )
    assert (
        runtime.semantic_parameter_group(
            "local_blocks.1.attention.relative_bias",
        )
        == "local_1_structure"
    )
    assert runtime.semantic_parameter_group("local_blocks.1.ffn.output.weight") == (
        "local_1_remaining"
    )
    assert runtime.semantic_parameter_group("global_tokens") == "global_summary"
    assert runtime.semantic_parameter_group("classifier.weight") == "cls_and_head"


def test_candidate_embedding_removes_only_runtime_model_import() -> None:
    """Candidate embedding stays byte-derived rather than becoming a hand fork."""
    source = builder.CANDIDATE_PATH.read_text(encoding="utf-8")
    embedded = builder._embedded_candidate_source(builder.CANDIDATE_PATH)  # noqa: SLF001
    assert source.count("from eqvae.models.local_global_mil import (") == 2
    assert embedded.count("from eqvae.models.local_global_mil import (") == 1
    assert "if TYPE_CHECKING:" in embedded
    assert textwrap.dedent(embedded).count("def fixed25_inductor_attention") == 1


def test_builder_rejects_extra_package_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The immutable package allow-list rejects accidental upload content."""
    output = tmp_path / "package"
    monkeypatch.setattr(builder, "DEFAULT_ROOT", output)
    builder.build()
    unexpected = output / "kernel/unexpected.txt"
    unexpected.write_text("unexpected", encoding="utf-8")
    with pytest.raises(ValueError, match="allow-list"):
        builder.validate()


def test_metadata_is_private_t4_and_owner_qualified() -> None:
    """The actor is portable while every attached source remains canonical."""
    metadata = builder._kernel_metadata()  # noqa: SLF001
    assert metadata["is_private"] == "true"
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["dataset_sources"] == [builder.INPUT_DATASET_REFERENCE]
    assert metadata["kernel_sources"] == list(builder.KERNEL_SOURCES)


def test_generic_launcher_has_exact_one_launch_guard() -> None:
    """Spec 0034 cannot fall through to an unrelated guard or launch twice."""
    launcher = builder.ROOT.joinpath("scripts/kaggle_kernel.sh").read_text(
        encoding="utf-8",
    )
    assert "KAGGLE_FULL_COMPILE_PROBE_CONFIRMED" in launcher
    assert "spec0034_pinned_torch_retry_v7_authorized" in launcher
    assert "Spec 0034 version-2 launch receipt is required" in launcher
    assert "Spec 0034 version-3 launch receipt is required" in launcher
    assert "Spec 0034 version-4 launch receipt is required" in launcher
    assert "Spec 0034 version-5 launch receipt is required" in launcher
    assert "Spec 0034 version-6 launch receipt is required" in launcher
    assert "Spec 0034 version-7 retry authority was already consumed" in launcher
    assert "build_wsi45630_full_compile_probe.py validate" in launcher
