# Copyright 2026 HiperMaximus
"""Tests for the shared Kaggle latest-Torch execution policy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

_KERNELS_ROOT = Path("kaggle/kernels")
_REQUIRED_PACKAGES = ("torch", "torchvision", "torchaudio")


def test_every_kaggle_kernel_upgrades_torch_before_running() -> None:
    """Every run template upgrades the same Torch stack before project import."""
    for template_path in sorted(_KERNELS_ROOT.glob("*/run_template.py")):
        source = template_path.read_text(encoding="utf-8")
        call_index = source.find("_ensure_latest_torch(")
        definition_index = source.find("def _ensure_latest_torch(")
        assert 0 <= call_index < definition_index, template_path
        before_upgrade = source[:call_index]
        assert "\nimport torch" not in before_upgrade, template_path
        assert "\nimport eqvae" not in before_upgrade, template_path
        helper_source = source[definition_index:]
        assert '"--upgrade"' in helper_source, template_path
        for package in _REQUIRED_PACKAGES:
            assert f'"{package}"' in helper_source, template_path
        metadata = cast(
            "dict[str, object]",
            json.loads(
                template_path.with_name("kernel-metadata.json").read_text(
                    encoding="utf-8",
                ),
            ),
        )
        assert metadata["enable_internet"] == "true"
