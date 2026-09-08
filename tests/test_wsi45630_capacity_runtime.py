# Copyright 2026 HiperMaximus
"""Focused evidence for the full-WSI probe's mount and model reuse."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    import pytest


def _module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("capacity_test_module", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_wsi_capacity_resolves_equal_filenames_by_sidecar_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep old and new top-ups distinct despite identical filenames.

    This DELIBERATE guard prevents reading part-11 bytes for part-12 pointers.
    """
    module = _module(
        Path("kaggle/kernels/wsi45630_capacity/run_template.py"),
    )
    rows: list[dict[str, str]] = []
    expected: dict[str, Path] = {}
    for part in (11, 12):
        root = tmp_path / f"producer{part}/dataset"
        root.mkdir(parents=True)
        source = f"owner/producer{part}"
        expected[source] = root
        for model in ("normal", "so2"):
            name = f"{model}_mu_cancer_topup.json"
            payload = f"{part}-{model}".encode()
            (root / name).write_bytes(payload)
            rows.append({
                "kaggle_source": source,
                "sidecar_name": name,
                "sidecar_bytes": str(len(payload)),
                "sidecar_sha256": hashlib.sha256(payload).hexdigest(),
            })
    catalog = tmp_path / "catalog.csv"
    with catalog.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    monkeypatch.setattr(module, "INPUT_ROOT", tmp_path)
    resolve = cast("Callable[[Path], dict[str, Path]]", module.resolve_sources)
    assert resolve(catalog) == expected


def test_full_wsi_capacity_reused_model_keeps_all_parameters_in_backward() -> None:
    """Keep the proven full CNN/transformer graph instead of detached features.

    This DERIVED small-shape proof checks every parameter has finite gradients;
    full-size GPU capacity is measured only on Kaggle.
    """
    module = _module(
        Path("kaggle/kernels/ubc_ocean_mil_transformer_capacity/run.py"),
    )
    factory = cast("Callable[[], nn.Module]", module.make_model)
    with torch.random.fork_rng(devices=[]):  # pyright: ignore[reportUnknownMemberType]
        model = factory()
        expected_parameters = 438_693
        assert sum(p.numel() for p in model.parameters()) == expected_parameters
        for name, parameter in model.named_parameters():
            if name.endswith("bias"):
                assert torch.count_nonzero(parameter).item() == 0
        logits = cast("Tensor", model(torch.randn(3, 16, 32, 32)))
        assert logits.shape == (1, 5)
        logits.square().sum().backward()  # pyright: ignore[reportUnknownMemberType]
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
            for parameter in model.parameters()
        )
