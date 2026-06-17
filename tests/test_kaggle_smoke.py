# Copyright 2026 HiperMaximus
"""Tests for the capped Kaggle smoke plumbing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
from torch.utils.data import DataLoader

from eqvae.benchmarking.kaggle_smoke import (
    EXPECTED_REAL_DATASET_SLUG,
    SETUP_CORRUPTION_VIEW,
    SETUP_DATA_KIND,
    SETUP_SMOKE_KIND,
    SETUP_SMOKE_SOURCE,
    KaggleSmokeRequest,
    write_kaggle_smoke,
)
from eqvae.data.synthetic import SyntheticPatchSpec, write_synthetic_patch_shard
from eqvae.data.training_batches import (
    PatchTrainingBatch,
    PatchTrainingDataset,
    PatchTrainingDatasetSpec,
    collate_patch_training_samples,
)

_IMAGE_SIZE = 64
_TINY_SHARD_COUNT = 4
_EXPECTED_SMOKE_STEPS = 3
_EXPECTED_APPLIED_COUNTS = [0, 0, 1]
_EXPECTED_SETUP_APPLIED_COUNT = 2


@dataclass(frozen=True)
class SmokeConfigOptions:
    """Options for synthetic smoke config fixtures."""

    max_train_steps: int = 3
    benchmark_kind: str = "local_synthetic_kaggle_smoke"
    benchmark_source: str = "local_cpu_synthetic_kaggle_smoke"
    data_kind: str = "synthetic-ubc-local-smoke"
    dataset_slug: str = ""
    validate_crc: bool = False
    corruption_view: str = "train_corrupted_local_smoke"


DEFAULT_SMOKE_CONFIG_OPTIONS = SmokeConfigOptions()


def test_patch_training_dataset_collates_semantic_metadata(tmp_path: Path) -> None:
    """Training batches carry semantic keys without changing tensor loader rail."""
    data_root = _write_tiny_shards(tmp_path)
    dataset = PatchTrainingDataset(
        PatchTrainingDatasetSpec(
            bin_path=data_root / "dataset" / "ubc_train_shuffled.bin",
            csv_path=data_root / "dataset" / "ubc_train_shuffled.csv",
            split="train",
            image_size=_IMAGE_SIZE,
            channels=3,
        ),
    )
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_patch_training_samples,
    )

    batch = cast("PatchTrainingBatch", next(iter(loader)))

    assert batch.images_uint8.shape == (2, 3, _IMAGE_SIZE, _IMAGE_SIZE)
    assert batch.split == "train"
    assert batch.file_indices == (0, 1)
    assert batch.row_indices == (0, 1)
    assert batch.semantic_sample_keys[0] == "train:synthetic_wsi_0000:0:0:0"
    assert batch.sample_ids[0] == "train:00000000:synthetic_wsi_0000:0:0:0"


def test_kaggle_smoke_writes_non_promotable_artifact(tmp_path: Path) -> None:
    """A tiny synthetic UBC shard can exercise the capped smoke path locally."""
    data_root = _write_tiny_shards(tmp_path)
    config_path = _write_smoke_config(tmp_path, data_root=data_root)

    output_path = write_kaggle_smoke(
        KaggleSmokeRequest(
            config_path=config_path,
            output_dir=tmp_path / "run",
        ),
    )

    payload = _load_json(output_path)
    data = payload["data"]
    train = payload["train"]
    validation = payload["validation"]
    assert isinstance(data, dict)
    assert isinstance(train, dict)
    assert isinstance(validation, dict)
    assert output_path == tmp_path / "run" / "benchmark" / "kaggle_smoke.json"
    assert payload["status"] == "smoke_pass"
    assert payload["status_scope"] == "non_promotable_debug"
    assert payload["full_run_eligible"] is False
    assert data["data_integrity_status"] == "not_checked"
    assert data["train_record_count"] == _TINY_SHARD_COUNT
    assert train["steps_completed"] == _EXPECTED_SMOKE_STEPS
    assert train["applied_counts"] == _EXPECTED_APPLIED_COUNTS
    assert train["total_applied_count"] == 1
    assert max(cast("list[float]", train["input_target_delta_maxes"])) > 0.0
    assert all(count > 0 for count in cast("list[int]", train["nonzero_update_counts"]))
    assert validation["batches_completed"] == 1
    assert validation["clean_validation_rng_advanced"] is False
    assert validation["finite_outputs"] == [True]


def test_setup_smoke_writes_distinct_non_promotable_artifact(
    tmp_path: Path,
) -> None:
    """Synthetic setup smoke cannot be confused with real-data smoke evidence."""
    data_root = _write_tiny_shards(tmp_path)
    config_path = _write_smoke_config(
        tmp_path,
        data_root=data_root,
        options=SmokeConfigOptions(
            benchmark_kind=SETUP_SMOKE_KIND,
            benchmark_source=SETUP_SMOKE_SOURCE,
            data_kind=SETUP_DATA_KIND,
            validate_crc=True,
            corruption_view=SETUP_CORRUPTION_VIEW,
        ),
    )

    output_path = write_kaggle_smoke(
        KaggleSmokeRequest(
            config_path=config_path,
            output_dir=tmp_path / "run",
        ),
    )

    payload = _load_json(output_path)
    data = cast("dict[str, object]", payload["data"])
    runtime = cast("dict[str, object]", payload["runtime"])
    train = cast("dict[str, object]", payload["train"])
    assert output_path == tmp_path / "run" / "benchmark" / "kaggle_setup_smoke.json"
    assert payload["status"] == "smoke_pass"
    assert payload["status_scope"] == "non_promotable_setup_smoke"
    assert payload["benchmark_kind"] == SETUP_SMOKE_KIND
    assert payload["benchmark_source"] == SETUP_SMOKE_SOURCE
    assert payload["full_run_eligible"] is False
    assert data["kind"] == SETUP_DATA_KIND
    assert not data["dataset_slug"]
    assert data["origin"] == "synthetic_or_ephemeral_path"
    assert data["data_integrity_status"] == "crc_checked"
    assert runtime["requires_cuda_t4"] is False
    assert train["total_applied_count"] == _EXPECTED_SETUP_APPLIED_COUNT


def test_kaggle_smoke_rejects_uncapped_config(tmp_path: Path) -> None:
    """The capped smoke cannot be converted into a longer run by config drift."""
    data_root = _write_tiny_shards(tmp_path)
    config_path = _write_smoke_config(
        tmp_path,
        data_root=data_root,
        options=SmokeConfigOptions(max_train_steps=4),
    )

    with pytest.raises(ValueError, match="max_train_steps"):
        write_kaggle_smoke(
            KaggleSmokeRequest(
                config_path=config_path,
                output_dir=tmp_path / "run",
            ),
        )


def test_setup_smoke_rejects_real_dataset_slug(tmp_path: Path) -> None:
    """The synthetic setup path must never attach or claim the real dataset."""
    data_root = _write_tiny_shards(tmp_path)
    config_path = _write_smoke_config(
        tmp_path,
        data_root=data_root,
        options=SmokeConfigOptions(
            benchmark_kind=SETUP_SMOKE_KIND,
            benchmark_source=SETUP_SMOKE_SOURCE,
            data_kind=SETUP_DATA_KIND,
            dataset_slug=EXPECTED_REAL_DATASET_SLUG,
            validate_crc=True,
            corruption_view=SETUP_CORRUPTION_VIEW,
        ),
    )

    with pytest.raises(ValueError, match="dataset slug"):
        write_kaggle_smoke(
            KaggleSmokeRequest(
                config_path=config_path,
                output_dir=tmp_path / "run",
            ),
        )


def test_real_data_kind_cannot_use_setup_smoke_source(tmp_path: Path) -> None:
    """Real-data contracts cannot bypass T4 checks with setup source strings."""
    data_root = _write_tiny_shards(tmp_path)
    config_path = _write_smoke_config(
        tmp_path,
        data_root=data_root,
        options=SmokeConfigOptions(
            benchmark_kind=SETUP_SMOKE_KIND,
            benchmark_source=SETUP_SMOKE_SOURCE,
            data_kind="ubc-pre-shuffled",
            dataset_slug=EXPECTED_REAL_DATASET_SLUG,
            validate_crc=True,
            corruption_view=SETUP_CORRUPTION_VIEW,
        ),
    )

    with pytest.raises(ValueError, match=r"Setup smoke data\.kind"):
        write_kaggle_smoke(
            KaggleSmokeRequest(
                config_path=config_path,
                output_dir=tmp_path / "run",
            ),
        )


def _write_tiny_shards(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "dataset"
    write_synthetic_patch_shard(
        bin_path=dataset_dir / "ubc_train_shuffled.bin",
        csv_path=dataset_dir / "ubc_train_shuffled.csv",
        spec=SyntheticPatchSpec(count=4, image_size=_IMAGE_SIZE, channels=3, seed=11),
        include_idx=False,
    )
    write_synthetic_patch_shard(
        bin_path=dataset_dir / "ubc_ocean_valid.bin",
        csv_path=dataset_dir / "ubc_ocean_valid.csv",
        spec=SyntheticPatchSpec(count=4, image_size=_IMAGE_SIZE, channels=3, seed=12),
        include_idx=True,
    )
    return tmp_path


def _write_smoke_config(
    tmp_path: Path,
    *,
    data_root: Path,
    options: SmokeConfigOptions = DEFAULT_SMOKE_CONFIG_OPTIONS,
) -> Path:
    config_path = tmp_path / "kaggle_smoke_config.json"
    payload = {
        "source_config": str(
            Path("configs/spec0001/non_eq_vae_debug_cpu.json").resolve(),
        ),
        "data": {
            "kind": options.data_kind,
            "dataset_slug": options.dataset_slug,
            "data_root": str(data_root),
            "image_size": _IMAGE_SIZE,
            "channels": 3,
        },
        "kaggle_smoke": {
            "benchmark_kind": options.benchmark_kind,
            "benchmark_source": options.benchmark_source,
            "full_run_eligible": False,
            "batch_size": 1,
            "max_train_steps": options.max_train_steps,
            "max_validation_batches": 1,
            "num_workers": 0,
            "validate_crc": options.validate_crc,
            "corruption_view": options.corruption_view,
        },
    }
    config_path.write_text(
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    return config_path


def _load_json(path: Path) -> dict[str, object]:
    return cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
