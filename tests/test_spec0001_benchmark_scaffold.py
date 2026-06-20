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
    CORRUPTION_CHECK_COLUMNS,
    DATALOADER_MATRIX_COLUMNS,
    GATE_HEALTH_COLUMNS,
    NUMERICAL_CHECK_COLUMNS,
    RUNTIME_MATRIX_COLUMNS,
    BenchmarkArtifactPaths,
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
EXPECTED_REAL_PRETEST_MEASURED_STEPS = 25
EXPECTED_REAL_PRETEST_TRAIN_PATCHES = 8192
EXPECTED_REAL_PRETEST_VALIDATION_PATCHES = 2048
EXPECTED_REAL_PRETEST_WARMUP_STEPS = 5
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
    assert "compile_options" not in runtime
    assert runtime["compile_scopes"] == [
        "none",
        "model_forward",
        "model_loss",
        "train_step_no_optimizer",
    ]
    assert runtime["candidate_per_device_batch_sizes"] == [4, 8, 12, 32]
    assert runtime["warmup_steps"] == EXPECTED_REAL_PRETEST_WARMUP_STEPS
    assert runtime["measured_steps"] == EXPECTED_REAL_PRETEST_MEASURED_STEPS
    compile_settle_policy = runtime["compile_settle_policy"]
    assert isinstance(compile_settle_policy, dict)
    assert (
        compile_settle_policy["compile_settle_steps"]
        == EXPECTED_REAL_PRETEST_WARMUP_STEPS
    )
    assert compile_settle_policy["excluded_from_timing"] is True
    assert compile_settle_policy["counter_source"] == (
        "torch._dynamo.utils.counters_with_reset_per_row"
    )
    assert compile_settle_policy["post_settle_required_zero_fields"] == [
        "graph_break_count",
        "recompile_count",
    ]
    must_exercise = compile_settle_policy["must_exercise"]
    assert isinstance(must_exercise, list)
    assert "mask_cardinality_all" in must_exercise
    data = effective["data"]
    assert isinstance(data, dict)
    assert data["kind"] == "ubc-pre-shuffled"
    benchmark_cap = data["benchmark_cap"]
    assert isinstance(benchmark_cap, dict)
    assert benchmark_cap["enabled"] is True
    assert benchmark_cap["train_patch_count"] == EXPECTED_REAL_PRETEST_TRAIN_PATCHES
    assert (
        benchmark_cap["validation_patch_count"]
        == EXPECTED_REAL_PRETEST_VALIDATION_PATCHES
    )
    assert benchmark_cap["window_policy"] == "fixed_hashed_spread_windows"
    runtime_pretest = effective["runtime_pretest"]
    assert isinstance(runtime_pretest, dict)
    assert runtime_pretest["benchmark_kind"] == "real_data_runtime_pretest"
    assert runtime_pretest["full_run_eligible"] is False
    assert runtime_pretest["writes_selected_runtime"] is False
    blocked_claims = runtime_pretest["blocked_claims"]
    assert isinstance(blocked_claims, dict)
    assert all(blocked_claims.values())


def test_runtime_config_stage_order_and_shortlist_provenance() -> None:
    """Runtime benchmark config keeps FP32-first staging and v4 provenance."""
    effective = resolve_json_config(
        Path("configs/spec0001/non_eq_vae_kaggle_runtime_benchmark.json"),
    ).effective_config
    runtime = effective["runtime_matrix"]
    assert isinstance(runtime, dict)

    stages = runtime["stages"]
    assert isinstance(stages, list)
    assert [stage["name"] for stage in stages if isinstance(stage, dict)] == [
        "fp32_compile_corruption_screen",
        "amp_followup_on_stable_fp32_candidates",
    ]
    fp32_stage = stages[0]
    amp_stage = stages[1]
    assert isinstance(fp32_stage, dict)
    assert isinstance(amp_stage, dict)
    assert fp32_stage["precision_policies"] == ["amp_off_fp32"]
    assert fp32_stage["compile_scopes"] == [
        "none",
        "model_forward",
        "model_loss",
        "train_step_no_optimizer",
    ]
    assert amp_stage["candidate_source"] == "stable_fp32_compile_corruption_candidates"

    candidates = runtime["seeded_candidates"]
    assert isinstance(candidates, list)
    assert any(
        isinstance(candidate, dict)
        and candidate["candidate_role"] == "sentinel_non_shortlisted_baseline"
        for candidate in candidates
    )
    parent_rows = [
        candidate["synthetic_v4_row_id"]
        for candidate in candidates
        if isinstance(candidate, dict)
        and candidate["candidate_role"] == "synthetic_v4_seed"
    ]
    assert parent_rows == [
        "dual_t4_ddp__bs8__amp_off_fp32__compile_off__branchless_all",
        "single_visible_t4__bs4__amp_off_fp32__compile_off__branchless_all",
        "single_visible_t4__bs32__amp_off_fp32__compile_off__branchless_all",
        "single_visible_t4__bs12__amp_off_fp32__compile_off__branchless_all",
    ]


def test_runtime_config_v8_carry_forward_is_shortlist_only() -> None:
    """The v8 pretest can seed the next slice but cannot select a runtime."""
    effective = resolve_json_config(
        Path("configs/spec0001/non_eq_vae_kaggle_runtime_benchmark.json"),
    ).effective_config
    runtime = effective["runtime_matrix"]
    assert isinstance(runtime, dict)

    v8_carry_forward = runtime["v8_carry_forward"]
    assert isinstance(v8_carry_forward, dict)
    assert v8_carry_forward["status"] == "pretest_incomplete"
    assert v8_carry_forward["status_scope"] == (
        "non_promotable_real_data_runtime_pretest"
    )
    assert v8_carry_forward["full_run_eligible"] is False
    assert v8_carry_forward["writes_selected_runtime"] is False
    assert v8_carry_forward["used_for"] == "candidate_shortlist_only"
    assert v8_carry_forward["artifact_hashes_required_before_use"] is True
    assert (
        v8_carry_forward["rows_may_not_satisfy_selected_runtime_linked_proof"] is True
    )
    assert v8_carry_forward["pretest_passing_eager_single_visible_rows"] == [
        "single_visible_t4__bs4__amp_off_fp32__compile_none__branchless_all",
        "single_visible_t4__bs4__amp_off_fp32__compile_none__indexed_masked",
        "single_visible_t4__bs8__amp_off_fp32__compile_none__branchless_all",
        "single_visible_t4__bs8__amp_off_fp32__compile_none__indexed_masked",
        "single_visible_t4__bs12__amp_off_fp32__compile_none__branchless_all",
        "single_visible_t4__bs12__amp_off_fp32__compile_none__indexed_masked",
    ]

    selection_slice = runtime["selection_benchmark_slice"]
    assert isinstance(selection_slice, dict)
    assert selection_slice["status"] == "planned_after_v8"
    assert selection_slice["v8_artifacts_are_promotable"] is False
    assert selection_slice["v8_artifact_hashes_required_in_runtime_proof"] is True
    assert selection_slice["selected_runtime_write_policy"] == (
        "write_only_after_this_benchmark_full_linked_proof_passes"
    )
    assert selection_slice["compiled_rows_policy"] == (
        "diagnostic_only_until_full_compile_settle_coverage_passes"
    )
    next_stages = selection_slice["stages"]
    assert isinstance(next_stages, list)
    assert [stage["name"] for stage in next_stages if isinstance(stage, dict)] == [
        "v8_shortlist_fp32_eager_confirmation",
        "amp_followup_on_confirmed_eager_fp32",
        "dual_t4_train_step_gate",
        "write_selected_runtime_after_all_gates",
    ]
    first_next_stage = next_stages[0]
    assert isinstance(first_next_stage, dict)
    assert first_next_stage["per_device_batch_sizes"] == [8, 12]
    assert first_next_stage["fallback_per_device_batch_sizes"] == [4]
    assert first_next_stage["compile_scopes"] == ["none"]
    assert first_next_stage["selected_runtime_write_allowed"] is False
    dual_gate = next_stages[2]
    assert isinstance(dual_gate, dict)
    assert dual_gate["accelerator_modes"] == ["dual_t4_ddp"]
    assert dual_gate["required_before_selected_runtime"] is True
    final_stage = next_stages[3]
    assert isinstance(final_stage, dict)
    assert final_stage["selected_runtime_write_allowed"] is True
    assert final_stage["requires_hash_links"] is True
    assert final_stage["required_inputs"] == [
        "runtime_proof:pass",
        "runtime_matrix:selected_row_pass",
        "dataloader_matrix:pass",
        "numerical_checks:pass",
        "corruption_checks:pass",
        "gate_health_summary:pass",
        "model_count:pass",
    ]
    promotion_blockers = selection_slice["promotion_blockers"]
    assert isinstance(promotion_blockers, list)
    assert "missing_real_dual_t4_train_step_timing" in promotion_blockers


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


def test_synthetic_benchmark_schema_core_outputs(tmp_path: Path) -> None:
    """Synthetic benchmark smoke writes core local schema artifacts."""
    artifacts = _write_schema_artifacts(tmp_path)
    runtime_rows = _load_csv(artifacts.runtime_matrix)
    selected_runtime = _load_json(artifacts.selected_runtime)
    dataloader_rows = _load_csv(artifacts.dataloader_matrix)
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
    assert {row["compile_scope"] for row in runtime_rows} == {
        "none",
        "model_forward",
    }
    assert {row["compile_settle_steps"] for row in runtime_rows} == {"0"}
    assert {row["graph_break_count"] for row in runtime_rows} == {"0"}
    assert {row["recompile_count"] for row in runtime_rows} == {"0"}
    assert selected_runtime["status"] == "schema_pass"
    assert selected_runtime["benchmark_kind"] == "local_synthetic_schema"
    assert selected_runtime["benchmark_source"] == "local_synthetic_schema_smoke"
    assert selected_runtime["full_run_eligible"] is False
    selected_dataloader = selected_runtime["dataloader"]
    assert isinstance(selected_dataloader, dict)
    assert selected_dataloader["prefetch_factor"] is None
    assert selected_dataloader["non_blocking_h2d"] is False
    selected_compile = selected_runtime["torch_compile"]
    assert isinstance(selected_compile, dict)
    assert selected_compile["scope"] == "none"
    selected_safety = selected_runtime["safety"]
    assert isinstance(selected_safety, dict)
    assert selected_safety["corruption_check_status"] == "schema_pass"
    selected_snapshot = selected_runtime["selected_row_snapshot"]
    assert isinstance(selected_snapshot, dict)
    assert selected_snapshot["compile_scope"] == "none"
    assert selected_snapshot["post_settle_graph_break_count"] == 0
    assert selected_snapshot["post_settle_recompile_count"] == 0
    assert tuple(dataloader_rows[0]) == DATALOADER_MATRIX_COLUMNS
    assert {row["split"] for row in dataloader_rows} == {"train", "validation"}
    assert {row["benchmark_kind"] for row in dataloader_rows} == {
        "local_synthetic_schema",
    }
    assert {row["full_run_eligible"] for row in dataloader_rows} == {"false"}
    assert {row["machine_shape"] for row in dataloader_rows} == {"local_cpu"}
    assert {row["non_blocking_h2d"] for row in dataloader_rows} == {"false"}
    assert {row["h2d_ms_p50"] for row in dataloader_rows} == {""}


def test_synthetic_benchmark_schema_dependency_outputs(tmp_path: Path) -> None:
    """Synthetic benchmark smoke writes dependency artifact schemas."""
    artifacts = _write_schema_artifacts(tmp_path)
    numerical_rows = _load_csv(artifacts.numerical_checks)
    corruption_rows = _load_csv(artifacts.corruption_checks)
    gate_rows = _load_csv(artifacts.gate_health)
    gate_summary = _load_json(artifacts.gate_health_summary)
    runtime_proof = _load_json(artifacts.runtime_proof)

    assert runtime_proof["status"] == "schema_pass"
    assert runtime_proof["full_run_eligible"] is False
    assert len(numerical_rows) == EXPECTED_RUNTIME_ROWS
    assert tuple(numerical_rows[0]) == NUMERICAL_CHECK_COLUMNS
    assert {row["full_run_eligible"] for row in numerical_rows} == {"false"}
    assert {row["gate_health_status"] for row in numerical_rows} == {"schema_pass"}
    assert {row["compile_scope"] for row in numerical_rows} == {
        "none",
        "model_forward",
    }
    assert len(corruption_rows) == EXPECTED_RUNTIME_ROWS
    assert tuple(corruption_rows[0]) == CORRUPTION_CHECK_COLUMNS
    assert {row["full_run_eligible"] for row in corruption_rows} == {"false"}
    assert {row["status"] for row in corruption_rows} == {"schema_pass"}
    assert len(gate_rows) == EXPECTED_GATE_ROWS
    assert tuple(gate_rows[0]) == GATE_HEALTH_COLUMNS
    assert {row["full_run_eligible"] for row in gate_rows} == {"false"}
    assert {row["gate_health_status"] for row in gate_rows} == {"schema_pass"}
    assert gate_summary["benchmark_source"] == "local_synthetic_schema_smoke"
    assert gate_summary["overall_status"] == "schema_pass"


def _write_schema_artifacts(tmp_path: Path) -> BenchmarkArtifactPaths:
    return write_synthetic_benchmark_artifacts(
        SyntheticBenchmarkRequest(
            config_path=Path("configs/spec0001/non_eq_vae_debug_cpu.json"),
            output_dir=tmp_path,
            run_name="spec0001_cpu_runtime_benchmark",
            max_benchmark_rows=EXPECTED_RUNTIME_ROWS,
            warmup_steps=1,
            measured_steps=2,
        ),
    )


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
