# Copyright 2026 HiperMaximus
"""Focused packaging guards for the Spec 0013 dual-T4 mechanics probe."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import subprocess  # noqa: S404
import sys
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, cast

from eqvae.benchmarking import so2_architecture_probe

# pyright: reportPrivateUsage=false

if TYPE_CHECKING:
    from eqvae.benchmarking.io import JsonObject

_SHA256_HEX_LENGTH = 64


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
    assert len(cast("str", runtime["source_sha256"])) == _SHA256_HEX_LENGTH


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
