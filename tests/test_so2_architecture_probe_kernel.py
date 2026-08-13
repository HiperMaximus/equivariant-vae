# Copyright 2026 HiperMaximus
"""Focused packaging guards for the Spec 0013 dual-T4 mechanics probe."""

from __future__ import annotations

import base64
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
from eqvae.models.so2_architecture_probe import _F01ToF01Conv  # noqa: PLC2701

# pyright: reportAny=false, reportPrivateUsage=false, reportUnknownMemberType=false

if TYPE_CHECKING:
    import pytest

    from eqvae.benchmarking.io import JsonObject


def test_selected_runtime_has_one_static_padded_bmm_direct_path() -> None:
    """Prevent rejected contractions or concatenation assembly from returning."""
    source = inspect.getsource(_F01ToF01Conv.expanded_kernel)
    assert source.count("torch.bmm") == 1
    assert "torch.mm" not in source
    assert "_expand_pair" not in source
    assert "new_empty" in source
    assert "torch.cat" not in source
    assert "for " not in source


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
    assert "WARMUP_UPDATES: Final = 20" in runner
    assert "TIMED_WINDOW_UPDATES: Final = 50" in runner
    assert 'SCHEMA_VERSION: Final = "spec0013.so2_dual_t4_final.v1"' in runner
    assert "four_mm_three_cat" not in runner
    assert "four_mm_direct" not in runner
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
    source = inspect.getsource(so2_architecture_probe._accuracy_case)  # noqa: SLF001
    assert "_selected_accuracy_backward(compiled_loss, compiled_source)" in source


def test_so2_probe_verdict_rejects_noisy_or_skipped_timing_rows() -> None:
    """Make control CV, AMP skip, and nonfinite timing evidence load-bearing."""
    updates: dict[str, object] = {
        "amp_skip_count": 0,
        "nonfinite_loss_count": 0,
        "finite_parameters": True,
        "post_settle_graph_break_count": 0,
        "initial_graph_break_count": 0,
        "post_settle_recompile_count": 0,
        "cross_rank_parameter_max_abs_difference": 0.0,
        "peak_allocated_mib": 1.0,
        "peak_reserved_mib": 1.0,
    }

    def summary(count: int) -> dict[str, object]:
        return {
            "samples_ms": [1.0] * count,
            "coefficient_variation": 0.01,
        }

    def block(name: str) -> JsonObject:
        return cast(
            "JsonObject",
            {
                "name": name,
                "output_relative_rms": 0.0,
                "max_coefficient_gradient_relative_rms": 0.0,
                "missing_coefficient_gradients": [],
                "nonfinite_coefficient_gradient_count": 0,
                "compiled_over_eager": 1.0,
                "equivariant_over_normal": 1.0,
                "paths": {
                    path_name: {
                        "windows": [summary(50), summary(50)],
                        "pooled": summary(100),
                        "amp_skip_count": 0,
                        "nonfinite_loss_count": 0,
                        "nonfinite_gradient_count": 0,
                    }
                    for path_name in (
                        "equivariant_eager",
                        "equivariant_compiled",
                        "normal_compiled",
                    )
                },
            },
        )

    def rank_result(rank: int) -> JsonObject:
        return cast(
            "JsonObject",
            {
                "rank": rank,
                "blocks": [
                    block(name)
                    for name in (
                        "identity_A",
                        "encoder_A_to_B",
                        "decoder_B_to_A",
                        "largest_D_to_D",
                    )
                ],
                "assembly_diagnostic": {
                    "selection_gate": False,
                    "windows": [
                        {
                            "expansion": summary(50),
                            "complete": summary(50),
                        },
                        {
                            "expansion": summary(50),
                            "complete": summary(50),
                        },
                    ],
                    "pooled_expansion": summary(100),
                    "pooled_complete": summary(100),
                },
            },
        )

    rank_results = [rank_result(0), rank_result(1)]
    passed, failures = so2_architecture_probe._verdict(  # noqa: SLF001
        updates,
        rank_results,
    )
    assert passed
    assert failures == []
    rank_zero_blocks = cast("list[JsonObject]", rank_results[0]["blocks"])
    paths = cast("dict[str, JsonObject]", rank_zero_blocks[0]["paths"])
    eager_windows = cast("list[JsonObject]", paths["equivariant_eager"]["windows"])
    eager_windows[0]["coefficient_variation"] = 0.11
    paths["equivariant_compiled"]["amp_skip_count"] = 1
    passed, failures = so2_architecture_probe._verdict(  # noqa: SLF001
        updates,
        rank_results,
    )
    assert not passed
    assert "rank0:identity_A:equivariant_eager:cv0" in failures
    assert "rank0:identity_A:equivariant_compiled:invalid_step" in failures


def test_final_verdict_requires_complete_measurement_schema() -> None:
    """Reject missing blocks and truncated windows instead of silently passing."""
    updates: dict[str, object] = {
        "amp_skip_count": 0,
        "nonfinite_loss_count": 0,
        "finite_parameters": True,
        "initial_graph_break_count": 0,
        "post_settle_graph_break_count": 0,
        "post_settle_recompile_count": 0,
        "cross_rank_parameter_max_abs_difference": 0.0,
        "peak_allocated_mib": 1.0,
        "peak_reserved_mib": 1.0,
    }
    passed, failures = so2_architecture_probe._verdict(  # noqa: SLF001
        updates,
        [cast("JsonObject", {"rank": 0, "blocks": []})],
    )
    assert not passed
    assert "rank_measurement_set" in failures
    assert "rank0:block_set" in failures
    assert "rank0:assembly_diagnostic_schema" in failures


def test_final_verdict_requires_compiled_ddp_parameter_agreement() -> None:
    """Reject a selected compiled path whose optimizer updates diverge by rank."""
    updates: dict[str, object] = {
        "amp_skip_count": 0,
        "nonfinite_loss_count": 0,
        "finite_parameters": True,
        "initial_graph_break_count": 0,
        "post_settle_graph_break_count": 0,
        "post_settle_recompile_count": 0,
        "cross_rank_parameter_max_abs_difference": 2e-6,
        "peak_allocated_mib": 1.0,
        "peak_reserved_mib": 1.0,
    }
    passed, failures = so2_architecture_probe._verdict(  # noqa: SLF001
        updates,
        [],
    )
    assert not passed
    assert "cross_rank_parameters" in failures
