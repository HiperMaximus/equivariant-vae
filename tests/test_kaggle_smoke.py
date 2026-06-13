# Copyright 2026 HiperMaximus
"""Tests for the capped Kaggle smoke plumbing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from torch.utils.data import DataLoader

from eqvae.benchmarking.kaggle_smoke import (
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
    train = payload["train"]
    validation = payload["validation"]
    assert isinstance(train, dict)
    assert isinstance(validation, dict)
    assert output_path == tmp_path / "run" / "benchmark" / "kaggle_smoke.json"
    assert payload["status"] == "smoke_pass"
    assert payload["full_run_eligible"] is False
    assert train["steps_completed"] == 1
    assert validation["batches_completed"] == 1
    assert validation["clean_validation_rng_advanced"] is False
    assert validation["finite_outputs"] == [True]


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


def _write_smoke_config(tmp_path: Path, *, data_root: Path) -> Path:
    config_path = tmp_path / "kaggle_smoke_config.json"
    payload = {
        "source_config": str(
            Path("configs/spec0001/non_eq_vae_debug_cpu.json").resolve(),
        ),
        "data": {
            "kind": "ubc-pre-shuffled",
            "data_root": str(data_root),
            "image_size": _IMAGE_SIZE,
            "channels": 3,
        },
        "kaggle_smoke": {
            "benchmark_kind": "local_synthetic_kaggle_smoke",
            "benchmark_source": "local_cpu_synthetic_kaggle_smoke",
            "full_run_eligible": False,
            "batch_size": 1,
            "max_train_steps": 1,
            "max_validation_batches": 1,
            "num_workers": 0,
            "validate_crc": False,
            "corruption_view": "train_corrupted_local_smoke",
        },
    }
    config_path.write_text(
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    return config_path


def _load_json(path: Path) -> dict[str, object]:
    return cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
