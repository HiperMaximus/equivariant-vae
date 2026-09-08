# Copyright 2026 HiperMaximus
"""Tests for the selected Kaggle Torch bootstrap policy."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import cast

_KERNELS_ROOT = Path("kaggle/kernels")
_REQUIRED_PACKAGES = ("torch", "torchvision", "torchaudio")
_DYNAMIC_METADATA_TEMPLATES = {
    "full_foreground_completion",
    "ubc_ocean_cancer_topup",
    "ubc_ocean_latent_finalizer",
    "ubc_ocean_latent_inference",
    "ubc_ocean_mil_capacity_probe",
    "ubc_ocean_supervised_calibration",
    "wsi45630_completion",
    "wsi45630_capacity",
    "wsi45630_local_global_capacity",
}
_CPU_ONLY_NO_BOOTSTRAP = {"ubc_ocean_latent_finalizer"}
_PINNED_TORCH_TEMPLATES = {
    "largest_class_weighted_amp_probe",
    "ubc_ocean_mil_training",
    "wsi45630_full_compile_probe",
}
_DIRECT_KERNEL_SOURCES = {
    _KERNELS_ROOT / "wsi45630_local_attention_probe/run.py",
}


def test_every_kaggle_kernel_upgrades_torch_before_running() -> None:
    """Every run template installs its selected Torch before project import."""
    sources = {*_KERNELS_ROOT.glob("*/run_template.py"), *_DIRECT_KERNEL_SOURCES}
    for template_path in sorted(sources):
        source = template_path.read_text(encoding="utf-8")
        if template_path.parent.name in _CPU_ONLY_NO_BOOTSTRAP:
            assert "_ensure_latest_torch(" not in source, template_path
            assert '"pip"' not in source, template_path
            continue
        if template_path.parent.name in _PINNED_TORCH_TEMPLATES:
            call_indices = [
                match.start()
                for match in re.finditer(r"install_pinned_torch\(", source)
                if source[max(0, match.start() - 4) : match.start()] != "def "
            ]
            definition_index = source.find("def install_pinned_torch(")
            assert call_indices, template_path
            assert definition_index >= 0, template_path
            call_index = min(call_indices)
            before_install = source[:call_index]
            assert "\nimport torch" not in before_install, template_path
            assert "\nimport eqvae" not in before_install, template_path
            helper_source = source[definition_index:call_index]
            assert 'f"torch=={PINNED_TORCH_VERSION}"' in helper_source, template_path
            assert '"--no-cache-dir"' in helper_source, template_path
            assert '"torchvision"' not in helper_source, template_path
            assert '"torchaudio"' not in helper_source, template_path
            metadata_path = template_path.with_name("kernel-metadata.json")
            if metadata_path.is_file():
                metadata = cast(
                    "dict[str, object]",
                    json.loads(metadata_path.read_text(encoding="utf-8")),
                )
                assert metadata["enable_internet"] == "true"
            continue
        call_indices = [
            match.start()
            for match in re.finditer(r"_ensure_latest_torch\(", source)
            if source[max(0, match.start() - 4) : match.start()] != "def "
        ]
        definition_index = source.find("def _ensure_latest_torch(")
        assert call_indices, template_path
        assert definition_index >= 0, template_path
        call_index = min(call_indices)
        before_upgrade = source[:call_index]
        assert "\nimport torch" not in before_upgrade, template_path
        assert "\nimport eqvae" not in before_upgrade, template_path
        helper_source = source[definition_index:]
        assert '"--upgrade"' in helper_source, template_path
        for package in _REQUIRED_PACKAGES:
            assert f'"{package}"' in helper_source, template_path
        metadata_path = template_path.with_name("kernel-metadata.json")
        if not metadata_path.is_file():
            assert template_path.parent.name in _DYNAMIC_METADATA_TEMPLATES
            continue
        metadata = cast(
            "dict[str, object]",
            json.loads(
                metadata_path.read_text(encoding="utf-8"),
            ),
        )
        assert metadata["enable_internet"] == "true"
