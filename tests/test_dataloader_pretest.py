# Copyright 2026 HiperMaximus
"""Tests for measured local dataloader pre-test artifacts."""

from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING

from eqvae.benchmarking import dataloader_pretest
from eqvae.benchmarking.dataloader_pretest import (
    LocalDataloaderPretestRequest,
    write_local_dataloader_pretest,
)
from eqvae.benchmarking.runtime_schema import DATALOADER_MATRIX_COLUMNS

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_local_dataloader_pretest_writes_non_promotable_rows(
    tmp_path: Path,
) -> None:
    """Measured local loader rows use local_pass without runtime eligibility."""
    config_path = _write_tiny_pretest_config(tmp_path)

    output_path = write_local_dataloader_pretest(
        LocalDataloaderPretestRequest(
            config_path=config_path,
            output_dir=tmp_path,
            run_name="local_pretest",
        ),
    )

    rows = _load_csv(output_path)
    assert output_path == tmp_path / "benchmark" / "dataloader_matrix.csv"
    assert (tmp_path / "data" / "local_synthetic_pretest" / "train.bin").exists()
    assert tuple(rows[0]) == DATALOADER_MATRIX_COLUMNS
    assert {row["split"] for row in rows} == {"train", "validation"}
    assert {row["benchmark_kind"] for row in rows} == {"local_synthetic_pretest"}
    assert {row["benchmark_source"] for row in rows} == {
        "local_cpu_synthetic_pretest",
    }
    assert {row["status"] for row in rows} == {"local_pass"}
    assert {row["full_run_eligible"] for row in rows} == {"false"}
    assert {row["accelerator_mode"] for row in rows} == {"local_cpu"}
    assert {row["machine_shape"] for row in rows} == {"local_cpu"}
    assert {row["prefetch_factor"] for row in rows} == {""}
    assert {row["pin_memory"] for row in rows} == {"false"}
    assert {row["persistent_workers"] for row in rows} == {"false"}
    assert {row["non_blocking_h2d"] for row in rows} == {"false"}
    assert {row["h2d_ms_p50"] for row in rows} == {""}
    assert {row["trainer_samples_sec"] for row in rows} == {""}
    assert {row["data_wait_fraction_p95"] for row in rows} == {""}
    assert {row["batches_measured"] for row in rows} == {"2"}
    assert {row["rank_sample_count"] for row in rows} == {"4"}
    assert all(float(row["batch_fetch_ms_p50"]) >= 0.0 for row in rows)
    assert all(float(row["loader_samples_sec"]) > 0.0 for row in rows)


def test_local_dataloader_pretest_records_worker_transport_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Worker candidates become explicit failure rows when transport is blocked."""
    config_path = _write_worker_pretest_config(tmp_path)

    def transport_unavailable() -> bool:
        return False

    monkeypatch.setattr(
        dataloader_pretest,
        "_worker_transport_available",
        transport_unavailable,
    )

    output_path = write_local_dataloader_pretest(
        LocalDataloaderPretestRequest(
            config_path=config_path,
            output_dir=tmp_path,
            run_name="local_pretest_worker_failure",
        ),
    )

    rows = _load_csv(output_path)
    assert {row["split"] for row in rows} == {"train", "validation"}
    assert {row["num_workers"] for row in rows} == {"1"}
    assert {row["status"] for row in rows} == {"fail"}
    assert {row["failure_kind"] for row in rows} == {
        "local_worker_transport_unavailable",
    }
    assert {row["full_run_eligible"] for row in rows} == {"false"}
    assert {row["batches_measured"] for row in rows} == {"0"}
    assert {row["loader_samples_sec"] for row in rows} == {""}


def _write_tiny_pretest_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "tiny_pretest.json"
    config_path.write_text(
        f"{json.dumps(_tiny_pretest_payload(), indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    return config_path


def _write_worker_pretest_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "worker_pretest.json"
    payload = _tiny_pretest_payload()
    dataloader_config = payload["dataloader_pretest"]
    if not isinstance(dataloader_config, dict):
        raise TypeError
    dataloader_config["candidates"] = [
        {
            "num_workers": 1,
            "prefetch_factor": 2,
            "pin_memory": False,
            "persistent_workers": False,
            "non_blocking_h2d": False,
        },
    ]
    config_path.write_text(
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    return config_path


def _tiny_pretest_payload() -> dict[str, object]:
    return {
        "schema_version": "spec0001.v0",
        "status": "local_pretest_test",
        "seeds": {"data_seed": 1234},
        "data": {
            "kind": "synthetic",
            "image_size": 8,
            "channels": 3,
            "train_samples": 6,
            "validation_samples": 6,
        },
        "dataloader_pretest": {
            "benchmark_kind": "local_synthetic_pretest",
            "benchmark_source": "local_cpu_synthetic_pretest",
            "full_run_eligible": False,
            "hot_path": "mmap_tensor_only_v1",
            "selector_provenance": "fixed_selector_json_v1",
            "batch_size": 2,
            "warmup_batches": 1,
            "measured_batches": 2,
            "candidates": [
                {
                    "num_workers": 0,
                    "prefetch_factor": None,
                    "pin_memory": False,
                    "persistent_workers": False,
                    "non_blocking_h2d": False,
                },
            ],
        },
    }


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))
