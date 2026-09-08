# Copyright 2026 HiperMaximus
# pyright: reportAny=false, reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownVariableType=false
# ruff: noqa: C420, I001, PLR2004, PT018, SIM300, SLF001, TC002
"""Focused package and train-only contracts for Spec 0037."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from typing import TYPE_CHECKING

import pytest

from scripts import build_tissue_fastpath_probe as builder

if TYPE_CHECKING:
    from pathlib import Path
    from types import ModuleType


def _module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("spec0037_runtime", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_geometry_has_no_tail_for_each_locked_budget() -> None:
    """Each drop-last static shape must consume its full balanced train pool."""
    assert builder.BATCH_GEOMETRY == {
        250: {"total_rows": 750, "batch_size": 125, "steps_per_epoch": 6},
        500: {"total_rows": 1_500, "batch_size": 125, "steps_per_epoch": 12},
        1_000: {"total_rows": 3_000, "batch_size": 125, "steps_per_epoch": 24},
        2_500: {"total_rows": 7_500, "batch_size": 125, "steps_per_epoch": 60},
        5_671: {"total_rows": 17_013, "batch_size": 159, "steps_per_epoch": 107},
    }
    assert all(
        geometry["total_rows"] == geometry["batch_size"] * geometry["steps_per_epoch"]
        for geometry in builder.BATCH_GEOMETRY.values()
    )


def test_probe_batches_are_balanced_and_rotate_the_125_remainder() -> None:
    """Three 125 batches give each tissue class exactly 125 probe examples."""
    _assets, _manifest, probe = builder._derive_assets(builder.ROOT)
    rows = builder._read_csv_bytes(
        (builder.ROOT / builder.MANIFEST_ROOT / builder.TRAIN_MANIFEST).read_bytes(),
        builder.TISSUE_HEADER,
    )
    for batch_size in (125, 159):
        batches = probe["batch_sizes"][str(batch_size)]
        assert len(batches) == 3
        assert all(
            len(batch) == batch_size and len(set(batch)) == batch_size
            for batch in batches
        )
    counts_125 = [
        Counter(rows[index]["tissue_label"] for index in batch)
        for batch in probe["batch_sizes"]["125"]
    ]
    assert [dict(count) for count in counts_125] == [
        {"tumor": 42, "stroma": 42, "necrosis": 41},
        {"tumor": 42, "stroma": 41, "necrosis": 42},
        {"tumor": 41, "stroma": 42, "necrosis": 42},
    ]
    assert sum(counts_125, Counter()) == Counter({
        label: 125 for label in builder.LABELS
    })
    assert all(
        Counter(rows[index]["tissue_label"] for index in batch)
        == Counter({label: 53 for label in builder.LABELS})
        for batch in probe["batch_sizes"]["159"]
    )


def test_builder_is_actor_portable_and_stages_no_eval_logical_asset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only source code may contain generic test words; logical data is train-only."""
    output = tmp_path / "package"
    monkeypatch.setattr(builder, "DEFAULT_ROOT", output)
    contract = builder.build(actor="professor-account")
    assert builder.validate(expected_actor="professor-account") == contract
    assert (
        contract["dataset_reference"]
        == "professor-account/eqvae-tissue-fastpath-probe-inputs"
    )
    bundle = output / "bundle"
    non_source = [
        path.relative_to(bundle).as_posix()
        for path in bundle.rglob("*")
        if path.is_file() and not path.relative_to(bundle).as_posix().startswith("src/")
    ]
    assert not any("validation" in name or "test" in name for name in non_source)
    metadata = json.loads((output / "kernel/kernel-metadata.json").read_text())
    assert metadata["dataset_sources"] == [contract["dataset_reference"]]
    assert metadata["kernel_sources"] == list(builder.KERNEL_SOURCES)
    monkeypatch.setattr(
        builder,
        "_validate_source_snapshot",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("must use sealed snapshot"),
        ),
    )
    monkeypatch.setattr(
        builder,
        "_render",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("must use sealed launcher"),
        ),
    )
    monkeypatch.setattr(
        builder,
        "_derive_assets",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("must not inspect live assets after sealing"),
        ),
    )
    assert (
        builder.validate(
            expected_actor="professor-account",
            sealed_source_snapshot=True,
        )
        == contract
    )
    runtime = _module(output / "kernel/run.py")
    assert runtime.INPUT_DATASET_REFERENCE == contract["dataset_reference"]
    assert runtime.INPUT_CONTRACT_SHA256 == builder._sha256(
        bundle / builder.CONTRACT_NAME,
    )


def test_runtime_keeps_default_scaler_and_native_optimizer_recovery() -> None:
    """The compiled closure may not replace normal GradScaler semantics."""
    source = builder.TEMPLATE_PATH.read_text(encoding="utf-8")
    required = (
        'torch.amp.GradScaler("cuda")',
        "fused=True",
        "capturable=False",
        "scaler.scale(loss).backward()",
        "scaler.unscale_(optimizer)",
        "scaler.step(optimizer)",
        "scaler.update()",
        "MAX_OVERFLOW_BACKOFFS = 3",
        '"fullgraph": True',
        "max-autotune-no-cudagraphs",
        "compiled_autograd",
        "torch.use_deterministic_algorithms(False)",
    )
    assert all(fragment in source for fragment in required)
    assert "init_scale=" not in source
    assert "validation.csv" not in source and "test.csv" not in source


def test_kaggle_guard_claims_the_single_remote_launch_before_push() -> None:
    """A failed or accepted first push must not silently spend a second version."""
    launcher = (builder.ROOT / "scripts/kaggle_kernel.sh").read_text(
        encoding="utf-8",
    )
    assert "tissue_fastpath_probe_launch_claim=" in launcher
    assert "Spec 0037's one private kernel-launch authority is consumed" in launcher
    assert "spec0037.kernel_launch_claim.v1" in launcher
    assert launcher.index("KAGGLE_FULL_DATASET_CONFIRMED:-}") < launcher.index(
        "PYSPEC0037CLAIM",
    )
    assert launcher.index("PYSPEC0037CLAIM") < launcher.index(
        'kaggle_api kernels push -p "$upload_kernel_dir"',
    )
