# Copyright 2026 HiperMaximus
"""Tests for the first spec 0001 benchmark-unblock scaffold."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch
from torch import nn

from eqvae.benchmarking.model_count import (
    MODEL_INVENTORY_COLUMNS,
    build_model_count_payload,
    write_model_count,
)
from eqvae.benchmarking.runtime_schema import (
    DATALOADER_MATRIX_COLUMNS,
    GATE_HEALTH_COLUMNS,
    NUMERICAL_CHECK_COLUMNS,
    RUNTIME_MATRIX_COLUMNS,
    SyntheticBenchmarkRequest,
    write_synthetic_benchmark_artifacts,
)
from eqvae.config import resolve_json_config
from eqvae.models.activations import GatedScalarActivation
from eqvae.models.non_equivariant_vae import build_non_equivariant_vae
from eqvae.models.resampling import FieldwiseBilinearUpsample2x

if TYPE_CHECKING:
    import pytest

EXPECTED_GATE_ROWS = 1
EXPECTED_MODEL_INVENTORY_ROWS = 129
EXPECTED_RUNTIME_ROWS = 2
SPEC_TOTAL_LEARNED_PARAMETERS = 3_958_435


def test_model_count_payload_matches_spec_target(tmp_path: Path) -> None:
    """Model-count smoke output uses the locked spec 0001 target."""
    output = tmp_path / "benchmark" / "model_count.json"
    payload = write_model_count(
        config_path=Path("configs/spec0001/non_eq_vae_debug_cpu.json"),
        output_path=output,
    )

    written = _load_json(output)
    inventory = _load_csv(output.with_name("model_inventory.csv"))

    assert payload["status"] == "pass"
    assert written["status"] == "pass"
    assert written["benchmark_kind"] == "implementation_model_count"
    assert written["benchmark_source"] == "instantiated_model"
    assert written["architecture_id"] == "spec0001_non_eq_vae_translatable"
    assert written["topology_version"] == "spec0001.count.v1"
    assert (
        written["model_config_hash_source"]
        == "canonical_json_sorted_compact_effective_config"
    )
    assert written["model_config_hash"] == written["effective_config_hash"]
    assert written["full_run_eligible"] is True
    assert written["module_inventory_path"] == "benchmark/model_inventory.csv"
    assert written["matches_spec_target"] is True
    assert written["total_learned_parameters"] == SPEC_TOTAL_LEARNED_PARAMETERS
    implementation = written["implementation"]
    assert isinstance(implementation, dict)
    assert implementation["inventory_matches_expected"] is True
    assert implementation["forward_order_verified"] is True
    assert tuple(inventory[0]) == MODEL_INVENTORY_COLUMNS
    assert len(inventory) == EXPECTED_MODEL_INVENTORY_ROWS
    assert {row["count_category"] for row in inventory} == {
        "fixed_resampling",
        "groupnorm_affine",
        "learned_convolution",
        "learned_gate",
    }


def test_model_count_resolves_layered_runtime_config() -> None:
    """Kaggle-thin configs inherit the model contract from `source_config`."""
    config_path = Path("configs/spec0001/non_eq_vae_kaggle_runtime_benchmark.json")
    source_path = Path("configs/spec0001/non_eq_vae_model_base.json")
    payload, _inventory = build_model_count_payload(config_path=config_path)

    assert payload["status"] == "pass"
    assert payload["config_resolution"] == "source_config_deep_merge_v1"
    assert payload["model_config_hash"] == payload["effective_config_hash"]
    assert payload["invoked_config_hash"] != payload["effective_config_hash"]
    assert payload["invoked_config_hash"] == _sha256_file(config_path)
    source_config_chain = payload["source_config_chain"]
    assert isinstance(source_config_chain, list)
    assert len(source_config_chain) == 1
    source_config = source_config_chain[0]
    assert isinstance(source_config, dict)
    assert source_config["path"] == str(source_path)
    assert source_config["sha256"] == _sha256_file(source_path)


def test_kaggle_runtime_config_does_not_inherit_local_pretest_fields() -> None:
    """Kaggle benchmark config resolves from model-only base, not CPU debug."""
    resolved = resolve_json_config(
        Path("configs/spec0001/non_eq_vae_kaggle_runtime_benchmark.json"),
    )
    effective = resolved.effective_config

    assert effective["run"] == {
        "name": "non_eq_vae_spec0001_runtime_benchmark",
        "mode": "kaggle_runtime_benchmark",
    }
    assert "dataloader_pretest" not in effective
    assert "benchmark" not in effective
    runtime = effective["runtime_matrix"]
    assert isinstance(runtime, dict)
    assert runtime["machine_shape"] == "NvidiaTeslaT4"
    data = effective["data"]
    assert isinstance(data, dict)
    assert data["kind"] == "ubc-pre-shuffled"


def test_model_count_resolves_source_config_without_repo_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Absolute invoked configs resolve repo-root-style sources from any cwd."""
    config_path = Path("configs/spec0001/non_eq_vae_kaggle_runtime_benchmark.json")
    absolute_config_path = config_path.resolve()
    monkeypatch.chdir(tmp_path)

    payload, _inventory = build_model_count_payload(config_path=absolute_config_path)

    source_config_chain = payload["source_config_chain"]
    assert payload["status"] == "pass"
    assert isinstance(source_config_chain, list)
    assert len(source_config_chain) == 1
    source_config = source_config_chain[0]
    assert isinstance(source_config, dict)
    source_path = source_config["path"]
    assert isinstance(source_path, str)
    assert source_path.endswith(
        "/configs/spec0001/non_eq_vae_model_base.json",
    )


def test_model_count_rejects_uninventoried_banned_leaf_module() -> None:
    """Extra parameter-free modules cannot hide outside the inventory."""
    model = build_non_equivariant_vae()
    model.add_module("extra_nearest_upsample", nn.Upsample(scale_factor=2.0))

    payload, _inventory = build_model_count_payload(
        config_path=Path("configs/spec0001/non_eq_vae_debug_cpu.json"),
        model=model,
    )

    implementation = payload["implementation"]
    assert isinstance(implementation, dict)
    assert payload["status"] == "fail"
    assert payload["full_run_eligible"] is False
    assert implementation["banned_operations_checked"] is False


def test_model_count_rejects_extra_countable_leaf_module() -> None:
    """Allowed module types still fail if they are absent from the inventory."""
    model = build_non_equivariant_vae()
    model.add_module("extra_bilinear_upsample", FieldwiseBilinearUpsample2x(3))

    payload, _inventory = build_model_count_payload(
        config_path=Path("configs/spec0001/non_eq_vae_debug_cpu.json"),
        model=model,
    )

    implementation = payload["implementation"]
    assert isinstance(implementation, dict)
    assert payload["status"] == "fail"
    assert payload["inventory_mismatch_count"] == 1
    assert implementation["inventory_matches_expected"] is False
    assert implementation["banned_operations_checked"] is True


def test_gated_scalar_activation_fp16_input_is_finite() -> None:
    """FP16 inputs are allowed while scalar gate sigmoid math remains FP32."""
    activation = GatedScalarActivation(channels=2)
    inputs = torch.linspace(-2.0, 2.0, steps=16, dtype=torch.float16).reshape(
        1,
        2,
        2,
        4,
    )

    outputs = activation.forward(inputs)

    assert outputs.dtype == torch.float16
    assert torch.isfinite(outputs).all()


def test_synthetic_benchmark_schema_outputs(tmp_path: Path) -> None:
    """Synthetic benchmark smoke writes every local schema artifact."""
    artifacts = write_synthetic_benchmark_artifacts(
        SyntheticBenchmarkRequest(
            config_path=Path("configs/spec0001/non_eq_vae_debug_cpu.json"),
            output_dir=tmp_path,
            run_name="spec0001_cpu_runtime_benchmark",
            max_benchmark_rows=EXPECTED_RUNTIME_ROWS,
            warmup_steps=1,
            measured_steps=2,
        ),
    )

    runtime_rows = _load_csv(artifacts.runtime_matrix)
    selected_runtime = _load_json(artifacts.selected_runtime)
    dataloader_rows = _load_csv(artifacts.dataloader_matrix)
    numerical_rows = _load_csv(artifacts.numerical_checks)
    gate_rows = _load_csv(artifacts.gate_health)
    gate_summary = _load_json(artifacts.gate_health_summary)
    model_count = _load_json(artifacts.model_count)

    assert model_count["status"] == "pass"
    assert model_count["benchmark_kind"] == "implementation_model_count"
    assert len(runtime_rows) == EXPECTED_RUNTIME_ROWS
    assert tuple(runtime_rows[0]) == RUNTIME_MATRIX_COLUMNS
    assert {row["benchmark_kind"] for row in runtime_rows} == {
        "local_synthetic_schema",
    }
    assert {row["full_run_eligible"] for row in runtime_rows} == {"false"}
    assert {row["gate_health_status"] for row in runtime_rows} == {"schema_pass"}
    assert {row["numerical_check_status"] for row in runtime_rows} == {
        "schema_pass",
    }
    assert selected_runtime["status"] == "schema_pass"
    assert selected_runtime["benchmark_kind"] == "local_synthetic_schema"
    assert selected_runtime["benchmark_source"] == "local_synthetic_schema_smoke"
    assert selected_runtime["full_run_eligible"] is False
    selected_dataloader = selected_runtime["dataloader"]
    assert isinstance(selected_dataloader, dict)
    assert selected_dataloader["prefetch_factor"] is None
    assert selected_dataloader["non_blocking_h2d"] is False
    assert tuple(dataloader_rows[0]) == DATALOADER_MATRIX_COLUMNS
    assert {row["split"] for row in dataloader_rows} == {"train", "validation"}
    assert {row["benchmark_kind"] for row in dataloader_rows} == {
        "local_synthetic_schema",
    }
    assert {row["full_run_eligible"] for row in dataloader_rows} == {"false"}
    assert {row["machine_shape"] for row in dataloader_rows} == {"local_cpu"}
    assert {row["non_blocking_h2d"] for row in dataloader_rows} == {"false"}
    assert {row["h2d_ms_p50"] for row in dataloader_rows} == {""}
    assert len(numerical_rows) == EXPECTED_RUNTIME_ROWS
    assert tuple(numerical_rows[0]) == NUMERICAL_CHECK_COLUMNS
    assert {row["full_run_eligible"] for row in numerical_rows} == {"false"}
    assert {row["gate_health_status"] for row in numerical_rows} == {"schema_pass"}
    assert len(gate_rows) == EXPECTED_GATE_ROWS
    assert tuple(gate_rows[0]) == GATE_HEALTH_COLUMNS
    assert {row["full_run_eligible"] for row in gate_rows} == {"false"}
    assert {row["gate_health_status"] for row in gate_rows} == {"schema_pass"}
    assert gate_summary["benchmark_source"] == "local_synthetic_schema_smoke"
    assert gate_summary["overall_status"] == "schema_pass"


def _load_json(path: Path) -> dict[str, object]:
    payload = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(payload, dict):
        message = f"Expected JSON object in {path}"
        raise TypeError(message)
    return cast("dict[str, object]", payload)


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
