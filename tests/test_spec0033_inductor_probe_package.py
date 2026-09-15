# Copyright 2026 HiperMaximus
# ruff: noqa: PLR2004
"""Package contract tests for the private Spec 0033 T4 probe."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

ROOT = Path("kaggle/kernels/wsi45630_inductor_attention_probe")


def _runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("spec0033_probe", ROOT / "run.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_metadata_is_private_portable_t4_probe() -> None:
    """Metadata fixes scope and shared owner-qualified source, not actor identity."""
    metadata = cast(
        "dict[str, object]",
        json.loads((ROOT / "kernel-metadata.json").read_text(encoding="utf-8")),
    )
    assert metadata["id"] == "maximusshtefan/eqvae-wsi45630-inductor-attention-probe"
    assert metadata["is_private"] == "true"
    assert metadata["enable_gpu"] == "true"
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["dataset_sources"] == [
        "maximusshtefan/eqvae-wsi45630-capacity-inputs",
    ]
    assert metadata["competition_sources"] == []
    assert metadata["kernel_sources"] == []
    assert metadata["model_sources"] == []


def test_runner_builds_physical_graph_and_edge_view() -> None:
    """Sequence-near but coordinate-far patches never become neighbors."""
    runner = _runner()
    build_graph = cast(
        "Callable[[tuple[tuple[int, int], ...]], dict[str, list[object]]]",
        runner.build_graph,
    )
    graph = build_graph(((0, 0), (256, 0), (4096, 0), (256, 256)))
    indices = cast("list[list[int]]", graph["indices"])
    valid = cast("list[list[bool]]", graph["valid"])
    assert 2 not in [
        key for key, keep in zip(indices[1], valid[1], strict=True) if keep
    ]
    assert graph["lengths"] == [3, 3, 1, 3]
    assert len(graph["edge_key"]) == sum(cast("list[int]", graph["lengths"]))


def test_runner_contains_exact_candidates_and_no_learning() -> None:
    """Probe source contains the locked comparison without a training campaign."""
    source = (ROOT / "run.py").read_text(encoding="utf-8")
    assert "SPEC0033_INDUCTOR_ATTENTION_PROBE_READY = True" in source
    assert "SDPBackend.EFFICIENT_ATTENTION" in source
    assert "torch.segment_reduce" in source
    assert 'fullgraph": True' in source
    assert "max-autotune-no-cudagraphs" in source
    assert "max_autotune_pointwise" in source
    assert "torch._dynamo.mark_dynamic" in source
    assert "optimizer" not in source.lower()
    assert "cross_entropy" not in source


def test_launcher_has_no_spec0033_permission_branch() -> None:
    """The generic launcher has no Spec 0033-specific permission logic."""
    launcher = Path("scripts/kaggle_kernel.sh").read_text(encoding="utf-8")
    assert "inductor_attention_probe" not in launcher
    assert "CONFIRMED" not in launcher
