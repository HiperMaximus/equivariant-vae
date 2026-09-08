# Copyright 2026 HiperMaximus
# ruff: noqa: PLR2004
"""Focused guard for the one-off largest-WSI Kaggle capacity probe."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from eqvae.cli import build_ubc_mil_capacity_probe as probe_builder
from eqvae.data.latent_shards import LATENT_RECORD_BYTES, LATENT_SHARD_HEADER_SIZE
from eqvae.data.supervised_latents import (
    CATALOG_HEADER,
    WSI_BAG_HEADER,
    WSI_INSTANCE_HEADER,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def test_probe_builds_only_the_unchunked_largest_training_bag(tmp_path: Path) -> None:
    """Pin the probe to a full train WSI so test data or chunking cannot enter."""
    manifests = tmp_path / "manifests"
    wsi_root = manifests / "wsi"
    wsi_root.mkdir(parents=True)
    catalog_rows = [
        (
            part,
            model_name,
            source,
            f"{model_name}_part_{part}.bin",
            20,
            LATENT_SHARD_HEADER_SIZE + 20 * LATENT_RECORD_BYTES,
            "a" * 64,
            f"{model_name}_part_{part}.json",
            100,
            "b" * 64,
        )
        for part, source in ((4, "source-four"), (11, "source-eleven"))
        for model_name in ("normal_vae", "so2_vae")
    ]
    _write_csv(manifests / "physical_parts.csv", CATALOG_HEADER, catalog_rows)

    instances = [
        (0, 100, 10, 0, 0, "CC", 0, "train", 4, 1),
        (1, 101, 10, 1, 0, "CC", 0, "train", 11, 2),
        (2, 102, 10, 2, 0, "CC", 0, "train", 11, 3),
        (3, 103, 20, 0, 0, "EC", 1, "train", 4, 4),
        (4, 104, 30, 0, 0, "HGSC", 2, "train", 4, 5),
        (5, 105, 40, 0, 0, "LGSC", 3, "train", 4, 6),
        (6, 106, 50, 0, 0, "MC", 4, "train", 4, 7),
    ]
    bags = [
        (0, 10, "CC", 0, "train", 0, 3),
        (1, 20, "EC", 1, "train", 3, 1),
        (2, 30, "HGSC", 2, "train", 4, 1),
        (3, 40, "LGSC", 3, "train", 5, 1),
        (4, 50, "MC", 4, "train", 6, 1),
    ]
    _write_csv(
        wsi_root / "wsi_cancer_train_instances.csv",
        WSI_INSTANCE_HEADER,
        instances,
    )
    _write_csv(wsi_root / "wsi_cancer_train_bags.csv", WSI_BAG_HEADER, bags)

    repo_root = Path(__file__).resolve().parents[1]
    output = tmp_path / "kernel"
    config = probe_builder.build_probe(
        repo_root=repo_root,
        manifest_root=manifests,
        output_root=output,
        enforce_real=False,
    )
    metadata = cast(
        "dict[str, object]",
        json.loads((output / "kernel-metadata.json").read_text()),
    )

    assert {path.name for path in output.iterdir()} == probe_builder.UPLOAD_FILES
    assert config["wsi_id"] == 10
    assert config["instance_count"] == 3
    assert config["part_counts"] == {"4": 1, "11": 2}
    assert config["checkpoint_chunk_size"] is None
    assert config["warmup_steps"] == 1
    assert config["measured_steps"] == 1
    assert config["grad_scaler_init_scale"] == 32_768
    assert config["grad_scaler_growth_interval"] == 1_000_000
    assert config["required_deadline_reserve_seconds"] == 3600
    assert config["projected_output_bytes"] == 100_000
    assert config["test_release_status"] == "not_authorized_not_mounted"
    assert metadata["kernel_sources"] == ["source-four", "source-eleven"]
    assert metadata["dataset_sources"] == []
    assert metadata["competition_sources"] == []
    wrapper = output / "run.py"
    assert "KAGGLE_UBC_OCEAN_MIL_CAPACITY_PROBE_READY = True" in wrapper.read_text()
    compile(wrapper.read_bytes(), str(wrapper), "exec")

    with wrapper.open("a", encoding="utf-8") as handle:
        handle.write("# mutation\n")
    with pytest.raises(ValueError, match=r"run\.py differs"):
        probe_builder.validate_probe(
            repo_root=repo_root,
            manifest_root=manifests,
            output_root=output,
            enforce_real=False,
        )


def _write_csv(
    path: Path,
    header: Sequence[str],
    rows: Sequence[Sequence[object]],
) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
