# Copyright 2026 HiperMaximus
"""Focused packaging guards for the Spec 0013 dual-T4 mechanics probe."""

from __future__ import annotations

import base64
import copy
import hashlib
import inspect
import io
import json
import re
import subprocess  # noqa: S404
import sys
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch

from eqvae.benchmarking import so2_architecture_probe
from eqvae.models.so2_architecture_probe import SO2LargestDDConv

# pyright: reportAny=false, reportPrivateUsage=false, reportUnknownMemberType=false

if TYPE_CHECKING:
    from collections.abc import Callable

    import pytest

    from eqvae.benchmarking.io import JsonObject


def test_follow_up_assembly_arms_preserve_kernel_output_and_gradients() -> None:
    """Keep both measured assembly candidates mathematically identical locally."""
    cast("Callable[[int], torch.Generator]", torch.manual_seed)(130013)
    source = SO2LargestDDConv()
    inputs = torch.randn(1, 144, 9, 9)
    for candidate in (
        so2_architecture_probe._DDirectAssemblyConv(source),  # noqa: SLF001
        so2_architecture_probe._DPaddedBmmAssemblyConv(source),  # noqa: SLF001
    ):
        reference = copy.deepcopy(source)
        candidate_inputs = inputs.detach().clone().requires_grad_()
        reference_inputs = inputs.detach().clone().requires_grad_()
        candidate_output = cast("torch.Tensor", candidate(candidate_inputs))
        reference_output = cast("torch.Tensor", reference(reference_inputs))
        assert torch.allclose(candidate_output, reference_output, rtol=1e-6, atol=1e-7)
        assert torch.allclose(
            candidate.expanded_kernel(),
            reference.expanded_kernel(),
            rtol=1e-6,
            atol=1e-7,
        )
        candidate_output.square().mean().backward()
        reference_output.square().mean().backward()
        candidate_parameters = dict(candidate.named_parameters())
        reference_parameters = dict(reference.named_parameters())
        assert candidate_parameters.keys() == reference_parameters.keys()
        for name, parameter in candidate_parameters.items():
            assert parameter.grad is not None
            assert reference_parameters[name].grad is not None
            candidate_gradient = parameter.grad
            reference_gradient = cast(
                "torch.Tensor",
                reference_parameters[name].grad,
            )
            assert torch.allclose(
                candidate_gradient,
                reference_gradient,
                rtol=2e-6,
                atol=1e-8,
            )
        assert torch.allclose(
            cast("torch.Tensor", candidate_inputs.grad),
            cast("torch.Tensor", reference_inputs.grad),
            rtol=2e-6,
            atol=1e-8,
        )
        compiled = cast(
            "Callable[[torch.Tensor], torch.Tensor]",
            torch.compile(  # pyright: ignore[reportUnknownMemberType]
                candidate,
                backend="eager",
                fullgraph=True,
                dynamic=False,
            ),
        )
        assert torch.allclose(compiled(inputs), reference_output.detach())


def test_so2_cpu_evidence_is_bound_to_the_current_probe_sources() -> None:
    """Fail when compact local evidence is stale relative to reviewed source."""
    repository = Path(__file__).resolve().parents[1]
    evidence = cast(
        "dict[str, object]",
        json.loads(
            (repository / "docs/data/spec0013_so2_cpu_probe.json").read_text(
                encoding="utf-8",
            ),
        ),
    )
    source_hashes = cast("dict[str, str]", evidence["source_files_sha256"])
    observed = {
        relative: hashlib.sha256((repository / relative).read_bytes()).hexdigest()
        for relative in source_hashes
    }
    assert observed == source_hashes


def test_so2_probe_build_embeds_fixed_runner_without_data(tmp_path: Path) -> None:
    """Keep the remote probe self-contained, fixed, and free of dataset setup."""
    repository = Path(__file__).resolve().parents[1]
    kernel_dir = repository / "kaggle/kernels/so2_architecture_probe"
    output = tmp_path / "run.py"
    subprocess.run(  # noqa: S603
        (
            sys.executable,
            str(repository / "scripts/build_kaggle_embedded_kernel.py"),
            "--repo-root",
            str(repository),
            "--kernel-dir",
            str(kernel_dir),
            "--output-run",
            str(output),
            "--ready-marker",
            "KAGGLE_SO2_ARCHITECTURE_PROBE_READY = True",
            "--allow-dirty",
        ),
        cwd=repository,
        check=True,
    )
    metadata = cast(
        "dict[str, object]",
        json.loads((kernel_dir / "kernel-metadata.json").read_text(encoding="utf-8")),
    )
    assert metadata["id"] == "maximusshtefan/eqvae-so2-architecture-probe"
    for field in (
        "dataset_sources",
        "competition_sources",
        "kernel_sources",
        "model_sources",
    ):
        assert metadata[field] == []

    run_text = output.read_text(encoding="utf-8")
    match = re.search(
        r'EMBEDDED_PAYLOAD_B64 = """\n(?P<payload>.*?)\n"""',
        run_text,
        flags=re.DOTALL,
    )
    assert match is not None
    with zipfile.ZipFile(
        io.BytesIO(base64.b64decode(match.group("payload"))),
    ) as archive:
        names = set(archive.namelist())
        runner = archive.read(
            "src/eqvae/benchmarking/so2_architecture_probe.py",
        ).decode("utf-8")
    assert "src/eqvae/models/so2_architecture_probe.py" in names
    assert "PER_DEVICE_BATCH: Final = 4" in runner
    assert "SETTLED_UPDATES: Final = 32" in runner
    assert "FOLLOW_UP_WARMUPS: Final = 20" in runner
    assert "FOLLOW_UP_WINDOW_UPDATES: Final = 50" in runner
    assert 'SCHEMA_VERSION: Final = "spec0013.so2_dual_t4_follow_up.v1"' in runner
    assert "class _DDirectAssemblyConv" in runner
    assert "class _DPaddedBmmAssemblyConv" in runner
    assert "compile_step_python_reducer_fp16_channels_last" in runner
    assert "full_vae_assembled" in runner


def test_so2_probe_has_specific_local_and_remote_guards() -> None:
    """Prevent the new kernel from falling through to legacy generic push checks."""
    repository = Path(__file__).resolve().parents[1]
    script = (repository / "scripts/kaggle_kernel.sh").read_text(encoding="utf-8")
    assert "preflight-so2-architecture-probe" in script
    assert "guard_so2_architecture_probe_push_ready" in script
    assert '"KAGGLE_SO2_ARCHITECTURE_PROBE_READY = True"' in script
    assert '"${KAGGLE_PUSH_CONFIRMED:-}" != "1"' in script
    assert 'local mode="${3:-push}"' in script
    assert '"local_preflight"' in script
    assert "no remote write performed" in script


def test_so2_probe_reads_and_matches_the_selected_runtime_plan() -> None:
    """Reject a benchmark that merely self-labels a hard-coded runtime bundle."""
    runtime = so2_architecture_probe._apply_selected_runtime()  # noqa: SLF001
    requested = cast("dict[str, object]", runtime["requested"])
    effective = cast("dict[str, object]", runtime["effective"])
    assert requested == effective
    assert requested["runtime_policy_id"] == (
        "compile_step_python_reducer_fp16_channels_last"
    )
    assert cast("str", runtime["source_sha256"]) == (
        "e9e998fd161f0955959c64aed7cd7ddbdfcb55a271b9ce05805903c97c93efb8"
    )


def test_remote_accuracy_paths_keep_selected_scaled_backward_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep GradScaler, compiled autograd, and unscale in both CUDA diagnostics."""
    events: list[str] = []
    optimizer = object()

    class FakeScaledLoss:
        def backward(self) -> None:  # noqa: PLR6301
            events.append("backward")

    class FakeScaler:
        def __init__(self, device: str, *, init_scale: float) -> None:
            assert device == "cuda"
            assert init_scale == so2_architecture_probe.GRAD_SCALER_INIT_SCALE

        def scale(self, loss: torch.Tensor) -> FakeScaledLoss:  # noqa: PLR6301
            del loss
            events.append("scale")
            return FakeScaledLoss()

        def unscale_(self, observed_optimizer: object) -> None:  # noqa: PLR6301
            assert observed_optimizer is optimizer
            events.append("unscale")

    @contextmanager
    def fake_compiled_autograd_context(*, enabled: bool):  # noqa: ANN202
        assert enabled
        events.append("enter_compiled_autograd")
        try:
            yield
        finally:
            events.append("exit_compiled_autograd")

    def fake_adamw(*args: object, **kwargs: object) -> object:
        del args, kwargs
        return optimizer

    monkeypatch.setattr(so2_architecture_probe, "GradScaler", FakeScaler)
    monkeypatch.setattr(
        so2_architecture_probe,
        "compiled_autograd_context",
        fake_compiled_autograd_context,
    )
    monkeypatch.setattr(so2_architecture_probe.torch.optim, "AdamW", fake_adamw)
    so2_architecture_probe._selected_accuracy_backward(  # noqa: SLF001
        torch.tensor(1.0),
        torch.nn.Linear(1, 1),
    )
    assert events == [
        "enter_compiled_autograd",
        "scale",
        "backward",
        "exit_compiled_autograd",
        "unscale",
    ]
    for function in (
        so2_architecture_probe._accuracy_case,  # noqa: SLF001
        so2_architecture_probe._follow_up_arm_accuracy,  # noqa: SLF001
    ):
        source = inspect.getsource(function)
        assert "_selected_accuracy_backward(compiled_loss, compiled_source)" in source


def test_so2_probe_verdict_rejects_noisy_or_skipped_timing_rows() -> None:
    """Make control CV, AMP skip, and nonfinite timing evidence load-bearing."""
    updates: dict[str, object] = {
        "amp_skip_count": 0,
        "nonfinite_loss_count": 0,
        "finite_parameters": True,
        "post_settle_graph_break_count": 0,
        "post_settle_recompile_count": 0,
        "peak_allocated_mib": 1.0,
        "peak_reserved_mib": 1.0,
    }
    row: dict[str, object] = {
        "name": "identity_A",
        "output_relative_rms": 0.0,
        "max_coefficient_gradient_relative_rms": 0.0,
        "missing_coefficient_gradients": [],
        "nonfinite_coefficient_gradient_count": 0,
        "compiled_over_eager": 1.0,
        "equivariant_over_normal": 1.0,
        "compiled_timing_cv": 0.01,
        "eager_timing_cv": 0.01,
        "normal_timing_cv": 0.01,
        "eager_amp_skip_count": 0,
        "compiled_amp_skip_count": 0,
        "normal_amp_skip_count": 0,
        "eager_nonfinite_loss_count": 0,
        "compiled_nonfinite_loss_count": 0,
        "normal_nonfinite_loss_count": 0,
        "eager_nonfinite_gradient_count": 0,
        "compiled_nonfinite_gradient_count": 0,
        "normal_nonfinite_gradient_count": 0,
    }
    rank_results = cast(
        "list[JsonObject]",
        [{"rank": 0, "blocks": [cast("JsonObject", row)]}],
    )
    passed, failures = so2_architecture_probe._verdict(  # noqa: SLF001
        updates,
        rank_results,
        assembly_fraction=0.01,
        weighted_ratio=1.0,
    )
    assert passed
    assert failures == []
    row["eager_timing_cv"] = 0.11
    row["compiled_amp_skip_count"] = 1
    passed, failures = so2_architecture_probe._verdict(  # noqa: SLF001
        updates,
        rank_results,
        assembly_fraction=0.01,
        weighted_ratio=1.0,
    )
    assert not passed
    assert "rank0:identity_A:eager_timing_cv" in failures
    assert "rank0:identity_A:timed_amp_skip" in failures


def test_follow_up_selects_a_passing_arm_without_accepting_rejected_arms() -> None:
    """Permit selection among fixed arms while preserving every rejection reason."""
    updates: dict[str, object] = {
        "amp_skip_count": 0,
        "nonfinite_loss_count": 0,
        "finite_parameters": True,
        "post_settle_graph_break_count": 0,
        "post_settle_recompile_count": 0,
        "peak_allocated_mib": 1.0,
        "peak_reserved_mib": 1.0,
    }

    def arm(name: str, assembly_fraction: float, median_ms: float) -> JsonObject:
        summary = cast(
            "JsonObject",
            {
                "samples_ms": [median_ms] * 4,
                "median_ms": median_ms,
                "coefficient_variation": 0.0,
            },
        )
        return cast(
            "JsonObject",
            {
                "name": name,
                "runtime": {
                    "initial_graph_break_count": int(name == "four_mm_three_cat"),
                    "post_settle_graph_break_count": 0,
                    "post_settle_recompile_count": 0,
                    "peak_allocated_mib": 1.0,
                    "peak_reserved_mib": 1.0,
                },
                "accuracy": {
                    "fp32_kernel_relative_rms": 0.0,
                    "output_relative_rms": 0.0,
                    "max_coefficient_gradient_relative_rms": 0.0,
                    "missing_coefficient_gradients": [],
                    "nonfinite_coefficient_gradient_count": 0,
                },
                "windows": [
                    {
                        "expansion": summary,
                        "complete": summary,
                        "assembly_fraction": assembly_fraction,
                    },
                    {
                        "expansion": summary,
                        "complete": summary,
                        "assembly_fraction": assembly_fraction,
                    },
                ],
                "pooled_expansion": summary,
                "pooled_complete": summary,
                "assembly_fraction": assembly_fraction,
            },
        )

    rank_results = cast(
        "list[JsonObject]",
        [
            {
                "rank": 0,
                "corrected_step_controls": {
                    "compiled_over_eager": 1.0,
                    "equivariant_over_normal": 1.0,
                    "paths": {
                        name: {
                            "windows": [
                                {
                                    "samples_ms": [1.0],
                                    "median_ms": 1.0,
                                    "coefficient_variation": 0.0,
                                },
                                {
                                    "samples_ms": [1.0],
                                    "median_ms": 1.0,
                                    "coefficient_variation": 0.0,
                                },
                            ],
                            "pooled": {
                                "samples_ms": [1.0, 1.0],
                                "median_ms": 1.0,
                                "coefficient_variation": 0.0,
                            },
                            "amp_skip_count": 0,
                            "nonfinite_loss_count": 0,
                            "nonfinite_gradient_count": 0,
                        }
                        for name in (
                            "equivariant_eager",
                            "equivariant_compiled",
                            "normal_compiled",
                        )
                    },
                },
                "corrected_accuracy": [
                    {
                        "name": "decoder_B_to_A",
                        "output_relative_rms": 0.0,
                        "max_coefficient_gradient_relative_rms": 0.0,
                        "missing_coefficient_gradients": [],
                        "nonfinite_coefficient_gradient_count": 0,
                    },
                ],
                "arms": [
                    arm("four_mm_three_cat", 0.4, 5.0),
                    arm("four_mm_direct", 0.08, 4.0),
                    arm("padded_bmm_direct", 0.2, 3.0),
                ],
            },
        ],
    )
    selected, failures, rejected = so2_architecture_probe._follow_up_verdict(  # noqa: SLF001
        updates,
        rank_results,
    )
    assert selected == "four_mm_direct"
    assert failures == []
    assert set(rejected) == {"four_mm_three_cat", "padded_bmm_direct"}

    corrected_accuracy = cast(
        "list[JsonObject]",
        rank_results[0]["corrected_accuracy"],
    )
    corrected_accuracy[0]["max_coefficient_gradient_relative_rms"] = 0.03
    selected, failures, rejected = so2_architecture_probe._follow_up_verdict(  # noqa: SLF001
        updates,
        rank_results,
    )
    assert selected is None
    assert failures == ["rank0:decoder_B_to_A:gradient"]
    assert set(rejected) == {"four_mm_three_cat", "padded_bmm_direct"}
