# Copyright 2026 HiperMaximus
"""Tests for the selected-runtime debug/resume/tiny gate contract."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from eqvae.benchmarking import selected_runtime_gate
from eqvae.benchmarking.selected_runtime_gate import (
    EXPECTED_DATASET_SLUG,
    EXPECTED_RUNTIME_POLICY_ID,
    EXPECTED_SELECTED_ROW_ID,
    SelectedRuntimeGateRequest,
    write_selected_runtime_gate,
)
from eqvae.cli.selected_runtime_gate import main as selected_runtime_gate_main
from eqvae.data.fixed_selectors import (
    FIXED_32_TRAIN_OVERFIT_COUNT,
    FIXED_32_TRAIN_OVERFIT_KIND,
    FIXED_32_TRAIN_OVERFIT_SEED,
    FIXED_SELECTOR_READY_STATUS,
    FIXED_SELECTOR_SCHEMA_VERSION,
    FixedSelectorGenerationContext,
    generate_fixed_selector_document,
    write_fixed_selector_document,
)
from eqvae.data.patch_shards import PatchShardSpec
from eqvae.data.roots import (
    TRAIN_BIN_NAME,
    TRAIN_CSV_NAME,
    VALIDATION_BIN_NAME,
    VALIDATION_CSV_NAME,
)
from eqvae.data.synthetic import SyntheticPatchSpec, write_synthetic_patch_shard
from eqvae.training.selected_runtime import (
    EXPECTED_AMP_APPLICATION_STATUS,
    EXPECTED_DDP_APPLICATION_STATUS,
    SelectedRuntimeApplicationObservation,
    build_plan_applied_proof,
    parse_selected_runtime_plan,
)

if TYPE_CHECKING:
    from collections.abc import Callable

PATCH_SIZE = 8
TRAIN_PATCH_COUNT = 40
VALIDATION_PATCH_COUNT = 25
FIXED32_SELECTOR_COUNT = 32
REMOTE_TINY_EFFECTIVE_GLOBAL_EPOCH_SAMPLES = 48
REMOTE_TINY_EFFECTIVE_PER_RANK_EPOCH_SAMPLES = 24
SELECTED_RUNTIME_BATCH_SIZE = 12
MASKED_HOLDOUT_WSI = "synthetic_wsi_0000"
# A re-measured plan's identity: the policy a Kaggle re-mint would select, and the
# row_id its own fields then compose. Used to prove the gate compares gate-health rows
# against the loaded plan rather than the frozen eager-v5 literal (Spec 0011 S17b-2).
_REPLANNED_POLICY_ID = "compile_step_ddp_optimizer_fp32_channels_last"
_REPLANNED_ROW_ID = (
    f"dual_t4_ddp__bs{SELECTED_RUNTIME_BATCH_SIZE}__amp_conservative__compile_none__"
    f"indexed_masked__policy_{_REPLANNED_POLICY_ID}"
)
_GATE_HEALTH_ROW_BLOCKER = "selected_runtime_output_gate_health_row_id_mismatch"
_GATE_HEALTH_CANDIDATE_BLOCKER = (
    "selected_runtime_output_gate_health_candidate_mismatch"
)
_GATE_HEALTH_POLICY_BLOCKER = "selected_runtime_output_gate_health_policy_mismatch"


def test_selected_runtime_gate_writes_fail_closed_contract(tmp_path: Path) -> None:
    """The real gate surface writes explicit fail-closed artifacts locally."""
    output_dir = tmp_path / "selected_runtime_gate"

    write_selected_runtime_gate(
        SelectedRuntimeGateRequest(
            debug_config_path=Path(
                "configs/spec0001/non_eq_vae_selected_runtime_debug.json",
            ),
            tiny_config_path=Path(
                "configs/spec0001/non_eq_vae_kaggle_tiny_overfit.json",
            ),
            selected_runtime_path=Path(
                "runs/kaggle/runtime_selection_v5/benchmark/selected_runtime.json",
            ),
            output_dir=output_dir,
            run_name="spec0001_selected_runtime_gate_test",
            data_root="auto",
        ),
    )

    benchmark_dir = output_dir / "benchmark"
    metrics_dir = output_dir / "metrics"
    assert {path.name for path in benchmark_dir.iterdir()} == {
        "artifact_manifest.json",
        "checkpoint_resume_proof.json",
        "gate_health_summary.json",
        "local_selected_runtime_readiness.json",
        "selected_runtime_plan_applied.json",
        "selected_runtime_debug_summary.json",
        "selected_runtime_gate_summary.json",
        "tiny_overfit_summary.json",
        "training_summary.json",
    }
    assert {path.name for path in metrics_dir.iterdir()} == {
        "gate_health.csv",
        "train_metrics.csv",
    }
    assert not (benchmark_dir / "selected_runtime.json").exists()

    summary = _load_json(benchmark_dir / "selected_runtime_gate_summary.json")
    selected_runtime = _object(summary, "selected_runtime")
    component_status = _object(summary, "component_status")
    blockers = _string_list(summary["launch_blockers_remaining"])
    assert summary["status"] == "fail"
    assert summary["full_run_eligible"] is False
    assert selected_runtime["selected_row_id"] == EXPECTED_SELECTED_ROW_ID
    assert selected_runtime["runtime_policy_id"] == EXPECTED_RUNTIME_POLICY_ID
    assert selected_runtime["validation_errors"] == []
    assert component_status["selected_runtime_transport"] == "pass"
    assert component_status["selected_runtime_plan_applied"] == "fail"
    assert component_status["local_readiness"] == "fail"
    assert component_status["real_ubc_debug"] == "fail"
    training = _load_json(benchmark_dir / "training_summary.json")
    assert training["failure_kind"] == "selected_runtime_debug_remote_proof_pending"
    assert "fixed_32_selector_placeholder" in blockers

    tiny = _load_json(benchmark_dir / "tiny_overfit_summary.json")
    assert tiny["status"] == "fail"
    assert tiny["patch_count"] == 0
    assert tiny["failure_kind"] == "fixed_32_selector_placeholder"

    manifest = _load_json(benchmark_dir / "artifact_manifest.json")
    assert manifest["status"] == "fail"
    assert manifest["contract_written"] is True
    assert manifest["full_run_eligible"] is False
    assert manifest["missing_artifacts"] == []
    assert "selected_runtime_gate_summary" in _object(manifest, "artifact_hashes")
    assert "selected_runtime_plan_applied" in _object(manifest, "artifact_hashes")
    assert "local_selected_runtime_readiness" in _object(manifest, "artifact_hashes")

    plan_applied = _load_json(benchmark_dir / "selected_runtime_plan_applied.json")
    readiness = _load_json(benchmark_dir / "local_selected_runtime_readiness.json")
    assert plan_applied["status"] == "fail"
    assert plan_applied["full_run_eligible"] is False
    assert plan_applied["plan_applied"] is False
    assert readiness["status"] == "fail"
    assert readiness["full_run_eligible"] is False
    assert readiness["remote_pass_ready"] is False
    assert readiness["real_train_runner_implemented"] is True
    assert readiness["fixed_32_selector_real"] is False


def test_selected_runtime_gate_marks_fabricated_runtime_failed(
    tmp_path: Path,
) -> None:
    """A pass-looking but wrong selected runtime is recorded as invalid."""
    runtime_path = tmp_path / "selected_runtime.json"
    payload = _load_json(
        Path("runs/kaggle/runtime_selection_v5/benchmark/selected_runtime.json"),
    )
    payload["selected_row_id"] = "fake_pass_row"
    runtime_path.write_text(
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "fabricated_runtime_gate"

    write_selected_runtime_gate(
        SelectedRuntimeGateRequest(
            debug_config_path=Path(
                "configs/spec0001/non_eq_vae_selected_runtime_debug.json",
            ),
            tiny_config_path=Path(
                "configs/spec0001/non_eq_vae_kaggle_tiny_overfit.json",
            ),
            selected_runtime_path=runtime_path,
            output_dir=output_dir,
            run_name="spec0001_selected_runtime_gate_fake_runtime",
        ),
    )

    summary = _load_json(
        output_dir / "benchmark" / "selected_runtime_gate_summary.json",
    )
    selected_runtime = _object(summary, "selected_runtime")
    component_status = _object(summary, "component_status")
    assert component_status["selected_runtime_transport"] == "fail"
    # Spec 0011 S17b: the identity is structural -- a fabricated row_id no longer
    # matches the id recomposed from the plan's own (still-v5) fields.
    assert "selected_runtime_selected_row_id_not_self_consistent" in _string_list(
        selected_runtime["validation_errors"],
    )
    assert "selected_runtime_transport_validation_failed" in _string_list(
        summary["launch_blockers_remaining"],
    )


def test_selected_runtime_gate_rejects_corrupted_launch_fields(
    tmp_path: Path,
) -> None:
    """Top-level launch settings must match the v5 selected runtime."""
    runtime_path = tmp_path / "selected_runtime.json"
    payload = _load_json(
        Path("runs/kaggle/runtime_selection_v5/benchmark/selected_runtime.json"),
    )
    payload["global_batch_size"] = 12
    mixed_precision = _object(payload, "mixed_precision")
    mixed_precision["grad_scaler_enabled"] = False
    runtime_path.write_text(
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )

    write_selected_runtime_gate(
        SelectedRuntimeGateRequest(
            debug_config_path=Path(
                "configs/spec0001/non_eq_vae_selected_runtime_debug.json",
            ),
            tiny_config_path=Path(
                "configs/spec0001/non_eq_vae_kaggle_tiny_overfit.json",
            ),
            selected_runtime_path=runtime_path,
            output_dir=tmp_path / "corrupted_launch_fields",
            run_name="spec0001_selected_runtime_gate_corrupted_launch",
        ),
    )

    summary = _load_json(
        tmp_path
        / "corrupted_launch_fields"
        / "benchmark"
        / "selected_runtime_gate_summary.json",
    )
    selected_runtime = _object(summary, "selected_runtime")
    assert "selected_runtime_top_level_wrong_global_batch" in _string_list(
        selected_runtime["validation_errors"],
    )
    assert "selected_runtime_mixed_precision_missing_scaler" in _string_list(
        selected_runtime["validation_errors"],
    )


def test_selected_runtime_plan_parser_rejects_corrupted_launch_fields(
    tmp_path: Path,
) -> None:
    """Train and gate validation share the same strict v5 parser."""
    runtime_path = tmp_path / "selected_runtime.json"
    payload = _load_json(
        Path("runs/kaggle/runtime_selection_v5/benchmark/selected_runtime.json"),
    )
    payload["global_batch_size"] = 12
    mixed_precision = _object(payload, "mixed_precision")
    mixed_precision["grad_scaler_enabled"] = False
    runtime_path.write_text(
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="selected_runtime_top_level_wrong_global_batch",
    ):
        parse_selected_runtime_plan(runtime_path)


def test_selected_runtime_plan_parser_rejects_non_standalone_runtime_proof(
    tmp_path: Path,
) -> None:
    """The v5 plan parser follows linked proof and requires standalone torchrun."""
    runtime_dir = tmp_path / "runtime" / "benchmark"
    runtime_dir.mkdir(parents=True)
    source_dir = Path("runs/kaggle/runtime_selection_v5/benchmark")
    runtime_path = runtime_dir / "selected_runtime.json"
    proof_path = runtime_dir / "runtime_proof.json"
    payload = _load_json(source_dir / "selected_runtime.json")
    proof = _load_json(source_dir / "runtime_proof.json")
    dual_gate = _object(proof, "dual_t4_train_step_gate")
    dual_gate["child_process_launch_command"] = (
        "torchrun --nproc_per_node=2 -m eqvae.benchmarking.runtime_selection_executor"
    )
    runtime_environment = _object(proof, "runtime_environment")
    runtime_environment["child_process_launch_command"] = (
        "torchrun --nproc_per_node=2 -m eqvae.benchmarking.runtime_selection_executor"
    )
    proof_path.write_text(
        f"{json.dumps(proof, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    artifacts = _object(payload, "artifacts")
    artifacts["runtime_proof_sha256"] = _sha256(proof_path)
    runtime_path.write_text(
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="selected_runtime_runtime_proof_not_standalone_nproc2",
    ):
        parse_selected_runtime_plan(runtime_path)


def test_selected_runtime_plan_parser_rejects_misleading_torchrun_substrings(
    tmp_path: Path,
) -> None:
    """The linked proof command must be a real torchrun invocation."""
    runtime_path = _runtime_bundle_with_proof_mutation(
        tmp_path,
        lambda proof: _replace_runtime_proof_commands(
            proof,
            "notorchrun --standalone --nproc_per_node=20 -m fake",
        ),
    )

    with pytest.raises(
        ValueError,
        match="selected_runtime_runtime_proof_not_standalone_nproc2",
    ):
        parse_selected_runtime_plan(runtime_path)


@pytest.mark.parametrize(
    "proof_key",
    [
        "dual_t4_train_step_gate",
        "runtime_environment",
    ],
)
def test_selected_runtime_plan_parser_requires_both_runtime_proof_commands(
    tmp_path: Path,
    proof_key: str,
) -> None:
    """Both linked proof blocks must prove the standalone dual-rank launch."""

    def _mutate(proof: dict[str, object]) -> None:
        block = _object(proof, proof_key)
        block["child_process_launch_command"] = (
            "python -m eqvae.benchmarking.runtime_selection_executor"
        )

    runtime_path = _runtime_bundle_with_proof_mutation(tmp_path, _mutate)

    with pytest.raises(
        ValueError,
        match="selected_runtime_runtime_proof_not_standalone_nproc2",
    ):
        parse_selected_runtime_plan(runtime_path)


def test_selected_runtime_plan_parser_rejects_duplicate_torchrun_nproc(
    tmp_path: Path,
) -> None:
    """A valid-looking launch with conflicting nproc options fails closed."""
    runtime_path = _runtime_bundle_with_proof_mutation(
        tmp_path,
        lambda proof: _replace_runtime_proof_commands(
            proof,
            (
                "torchrun --standalone --nproc_per_node=2 "
                "--nproc_per_node 1 -m eqvae.benchmarking.runtime_selection_executor"
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match="selected_runtime_runtime_proof_not_standalone_nproc2",
    ):
        parse_selected_runtime_plan(runtime_path)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            "write_decision_allowed",
            "selected_runtime_runtime_proof_write_decision_allowed_mismatch",
        ),
        (
            "dual_gate_rank_assignment",
            "selected_runtime_runtime_proof_dual_gate_rank_assignment_mismatch",
        ),
        (
            "runtime_environment_returncode",
            "selected_runtime_runtime_proof_environment_child_process_returncode_mismatch",
        ),
    ],
)
def test_selected_runtime_plan_parser_rejects_linked_runtime_proof_drift(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    """Linked proof status/write/rank/return-code drift rejects the plan."""

    def _mutate(proof: dict[str, object]) -> None:
        if mutation == "write_decision_allowed":
            decision = _object(proof, "selected_runtime_write_decision")
            decision["allowed"] = False
        elif mutation == "dual_gate_rank_assignment":
            dual_gate = _object(proof, "dual_t4_train_step_gate")
            dual_gate["rank_assignments"] = [
                {
                    "rank": 0,
                    "local_rank": 0,
                    "device": 0,
                    "current_device": 0,
                    "world_size": 2,
                },
                {
                    "rank": 1,
                    "local_rank": 1,
                    "device": 1,
                    "current_device": 0,
                    "world_size": 2,
                },
            ]
        elif mutation == "runtime_environment_returncode":
            environment = _object(proof, "runtime_environment")
            environment["child_process_returncode"] = 1
        else:
            raise AssertionError(mutation)

    runtime_path = _runtime_bundle_with_proof_mutation(tmp_path, _mutate)

    with pytest.raises(ValueError, match=match):
        parse_selected_runtime_plan(runtime_path)


def test_selected_runtime_plan_parser_rejects_failed_runtime_proof(
    tmp_path: Path,
) -> None:
    """A matching proof hash is not enough when proof status fields fail."""

    def _mutate(proof: dict[str, object]) -> None:
        proof["status"] = "fail"
        proof["selection_ready"] = False
        proof["selected_runtime_written"] = False
        dual_gate = _object(proof, "dual_t4_train_step_gate")
        dual_gate["status"] = "fail"

    runtime_path = _runtime_bundle_with_proof_mutation(tmp_path, _mutate)

    with pytest.raises(
        ValueError,
        match="selected_runtime_runtime_proof_status_mismatch",
    ):
        parse_selected_runtime_plan(runtime_path)


def test_selected_runtime_plan_applied_proof_rejects_recorded_not_applied() -> None:
    """A v5 runtime record is not enough unless train settings apply it."""
    plan = parse_selected_runtime_plan(
        Path("runs/kaggle/runtime_selection_v5/benchmark/selected_runtime.json"),
    )
    proof = build_plan_applied_proof(
        plan=plan,
        observed=SelectedRuntimeApplicationObservation(
            selected_row_id=EXPECTED_SELECTED_ROW_ID,
            runtime_policy_id=EXPECTED_RUNTIME_POLICY_ID,
            accelerator_mode="local_cpu",
            machine_shape="local_cpu",
            world_size=1,
            nproc_per_node=1,
            torchrun_standalone=False,
            batch_size=1,
            global_batch_size=1,
            optimizer_updates_per_epoch=plan.optimizer_updates_per_epoch,
            amp_enabled=False,
            grad_scaler_enabled=False,
            fp32_loss=False,
            autocast_dtype="float32",
            torch_compile_enabled=False,
            compile_scope="none",
            dataloader_num_workers=0,
            dataloader_prefetch_factor=None,
            dataloader_pin_memory=False,
            dataloader_persistent_workers=False,
            dataloader_non_blocking_h2d=True,
            corruption_strategy="identity_clean_no_corruption",
            memory_format="contiguous",
            ddp_static_graph=False,
            ddp_gradient_as_bucket_view=False,
            zero_grad_set_to_none=True,
            local_ddp_status="not_executed",
            local_amp_status="not_executed",
        ),
    )

    assert proof["status"] == "fail"
    assert proof["full_run_eligible"] is False
    assert proof["plan_applied"] is False
    mismatches = _string_list(proof["mismatches"])
    assert any("per_device_batch_size" in mismatch for mismatch in mismatches)
    assert any("amp_enabled" in mismatch for mismatch in mismatches)
    assert any("corruption_strategy" in mismatch for mismatch in mismatches)


def test_selected_runtime_plan_applied_proof_rejects_unexecuted_ddp_amp() -> None:
    """Matching scalar fields still fail if DDP/AMP did not actually execute."""
    plan = parse_selected_runtime_plan(
        Path("runs/kaggle/runtime_selection_v5/benchmark/selected_runtime.json"),
    )
    proof = build_plan_applied_proof(
        plan=plan,
        observed=SelectedRuntimeApplicationObservation(
            selected_row_id=plan.selected_row_id,
            runtime_policy_id=plan.runtime_policy_id,
            accelerator_mode=plan.accelerator_mode,
            machine_shape=plan.machine_shape,
            world_size=plan.world_size,
            nproc_per_node=plan.nproc_per_node,
            torchrun_standalone=plan.torchrun_standalone,
            batch_size=plan.per_device_batch_size,
            global_batch_size=plan.global_batch_size,
            optimizer_updates_per_epoch=plan.optimizer_updates_per_epoch,
            amp_enabled=plan.amp_enabled,
            grad_scaler_enabled=plan.grad_scaler_enabled,
            fp32_loss=plan.fp32_loss,
            autocast_dtype=plan.autocast_dtype,
            torch_compile_enabled=plan.torch_compile_enabled,
            compile_scope=plan.compile_scope,
            dataloader_num_workers=plan.dataloader_num_workers,
            dataloader_prefetch_factor=plan.dataloader_prefetch_factor,
            dataloader_pin_memory=plan.dataloader_pin_memory,
            dataloader_persistent_workers=plan.dataloader_persistent_workers,
            dataloader_non_blocking_h2d=plan.dataloader_non_blocking_h2d,
            corruption_strategy=plan.corruption_strategy,
            memory_format=plan.memory_format,
            ddp_static_graph=plan.ddp_static_graph,
            ddp_gradient_as_bucket_view=plan.ddp_gradient_as_bucket_view,
            zero_grad_set_to_none=plan.zero_grad_set_to_none,
            local_ddp_status="not_executed",
            local_amp_status="not_executed",
        ),
    )

    assert proof["status"] == "fail"
    assert proof["plan_applied"] is False
    mismatches = _string_list(proof["mismatches"])
    assert any("local_ddp_status" in mismatch for mismatch in mismatches)
    assert any("local_amp_status" in mismatch for mismatch in mismatches)


def test_selected_runtime_plan_applied_proof_rejects_optimizer_updates_drift() -> None:
    """Changing optimizer_updates_per_epoch alone fails plan application."""
    plan = parse_selected_runtime_plan(
        Path("runs/kaggle/runtime_selection_v5/benchmark/selected_runtime.json"),
    )
    drifted_updates = plan.optimizer_updates_per_epoch + 1
    observed = SelectedRuntimeApplicationObservation(
        selected_row_id=plan.selected_row_id,
        runtime_policy_id=plan.runtime_policy_id,
        accelerator_mode=plan.accelerator_mode,
        machine_shape=plan.machine_shape,
        world_size=plan.world_size,
        nproc_per_node=plan.nproc_per_node,
        torchrun_standalone=plan.torchrun_standalone,
        batch_size=plan.per_device_batch_size,
        global_batch_size=plan.global_batch_size,
        optimizer_updates_per_epoch=drifted_updates,
        amp_enabled=plan.amp_enabled,
        grad_scaler_enabled=plan.grad_scaler_enabled,
        fp32_loss=plan.fp32_loss,
        autocast_dtype=plan.autocast_dtype,
        torch_compile_enabled=plan.torch_compile_enabled,
        compile_scope=plan.compile_scope,
        dataloader_num_workers=plan.dataloader_num_workers,
        dataloader_prefetch_factor=plan.dataloader_prefetch_factor,
        dataloader_pin_memory=plan.dataloader_pin_memory,
        dataloader_persistent_workers=plan.dataloader_persistent_workers,
        dataloader_non_blocking_h2d=plan.dataloader_non_blocking_h2d,
        corruption_strategy=plan.corruption_strategy,
        memory_format=plan.memory_format,
        ddp_static_graph=plan.ddp_static_graph,
        ddp_gradient_as_bucket_view=plan.ddp_gradient_as_bucket_view,
        zero_grad_set_to_none=plan.zero_grad_set_to_none,
        local_ddp_status=EXPECTED_DDP_APPLICATION_STATUS,
        local_amp_status=EXPECTED_AMP_APPLICATION_STATUS,
    )
    proof = build_plan_applied_proof(plan=plan, observed=observed)

    assert proof["status"] == "fail"
    assert proof["plan_applied"] is False
    expected_mismatch = (
        f"optimizer_updates_per_epoch: expected "
        f"{plan.optimizer_updates_per_epoch!r}, observed {drifted_updates!r}"
    )
    assert _string_list(proof["mismatches"]) == [expected_mismatch]
    assert observed.as_json()["optimizer_updates_per_epoch"] == drifted_updates


def test_selected_runtime_gate_rejects_fabricated_fixed32_selector(
    tmp_path: Path,
) -> None:
    """A fake 32-row pass JSON is not enough to unlock tiny-overfit."""
    selector_path = tmp_path / "fake_fixed_32_train_overfit_patches.json"
    selector_path.write_text(
        f"{json.dumps(_fake_fixed32_selector_payload(), indent=2)}\n",
        encoding="utf-8",
    )

    write_selected_runtime_gate(
        SelectedRuntimeGateRequest(
            debug_config_path=Path(
                "configs/spec0001/non_eq_vae_selected_runtime_debug.json",
            ),
            tiny_config_path=Path(
                "configs/spec0001/non_eq_vae_kaggle_tiny_overfit.json",
            ),
            selected_runtime_path=Path(
                "runs/kaggle/runtime_selection_v5/benchmark/selected_runtime.json",
            ),
            fixed_train_patches=selector_path,
            output_dir=tmp_path / "fake_selector",
            run_name="spec0001_selected_runtime_gate_fake_selector",
        ),
    )

    summary = _load_json(
        tmp_path / "fake_selector" / "benchmark" / "selected_runtime_gate_summary.json",
    )
    fixed_selector = _object(summary, "fixed_train_patches")
    assert fixed_selector["status"] == "fail"
    assert fixed_selector["selector_count"] == FIXED32_SELECTOR_COUNT
    assert fixed_selector["failure_kind"] == "fixed_32_selector_schema_invalid"
    assert "fixed_32_selector_schema_invalid" in _string_list(
        summary["launch_blockers_remaining"],
    )


def test_selected_runtime_gate_rejects_synthetic_fixed32_selector_replay(
    tmp_path: Path,
) -> None:
    """Local schema replay is not enough for the real UBC selector gate."""
    data_root = tmp_path / "ubc"
    selector_path = tmp_path / "fixed_32_train_overfit_patches.json"
    holdout_path = tmp_path / "masked_holdout.csv"
    _write_complete_data_root(data_root)
    holdout_path.write_text(
        f"image_id,label,is_updated_image_id\n{MASKED_HOLDOUT_WSI},HGSC,false\n",
        encoding="utf-8",
    )
    document = generate_fixed_selector_document(
        selector_kind=FIXED_32_TRAIN_OVERFIT_KIND,
        shard_spec=PatchShardSpec(
            bin_path=data_root / TRAIN_BIN_NAME,
            csv_path=data_root / TRAIN_CSV_NAME,
            image_size=PATCH_SIZE,
            validate_crc=True,
        ),
        source_split="train",
        context=FixedSelectorGenerationContext(
            dataset_slug=EXPECTED_DATASET_SLUG,
            data_root=data_root,
            masked_holdout_wsi_ids=frozenset({MASKED_HOLDOUT_WSI}),
        ),
    )
    write_fixed_selector_document(
        path=selector_path,
        document=replace(document, masked_holdout_exclusion=str(holdout_path)),
    )

    write_selected_runtime_gate(
        SelectedRuntimeGateRequest(
            debug_config_path=Path(
                "configs/spec0001/non_eq_vae_selected_runtime_debug.json",
            ),
            tiny_config_path=Path(
                "configs/spec0001/non_eq_vae_kaggle_tiny_overfit.json",
            ),
            selected_runtime_path=Path(
                "runs/kaggle/runtime_selection_v5/benchmark/selected_runtime.json",
            ),
            fixed_train_patches=selector_path,
            output_dir=tmp_path / "synthetic_selector",
            run_name="spec0001_selected_runtime_gate_synthetic_selector",
            data_root=str(data_root),
        ),
    )

    benchmark_dir = tmp_path / "synthetic_selector" / "benchmark"
    summary = _load_json(benchmark_dir / "selected_runtime_gate_summary.json")
    fixed_selector = _object(summary, "fixed_train_patches")
    blockers = _string_list(summary["launch_blockers_remaining"])
    tiny = _load_json(benchmark_dir / "tiny_overfit_summary.json")
    assert fixed_selector["status"] == "fail"
    assert fixed_selector["selector_count"] == FIXED32_SELECTOR_COUNT
    assert fixed_selector["failure_kind"] == "fixed_32_selector_not_canonical_real_ubc"
    assert fixed_selector["canonical_real_ubc"] is False
    assert "fixed_32_selector_placeholder" not in blockers
    assert "fixed_32_selector_validation_failed" not in blockers
    assert "fixed_32_selector_not_canonical_real_ubc" in blockers
    assert "selected_runtime_debug_remote_proof_pending" not in blockers
    assert tiny["patch_count"] == FIXED32_SELECTOR_COUNT
    assert tiny["failure_kind"] == "fixed_32_selector_not_canonical_real_ubc"


def test_selected_runtime_push_readiness_cli_fails_closed() -> None:
    """The structured push verifier blocks the current contract-only state."""
    exit_code = selected_runtime_gate_main(
        [
            "--verify-push-ready",
            "--debug-config",
            "configs/spec0001/non_eq_vae_selected_runtime_debug.json",
            "--tiny-config",
            "configs/spec0001/non_eq_vae_kaggle_tiny_overfit.json",
            "--runtime-config",
            "runs/kaggle/runtime_selection_v5/benchmark/selected_runtime.json",
            "--fixed-train-patches",
            "configs/spec0001/fixed_32_train_overfit_patches.json",
        ],
    )

    assert exit_code == 1


def test_selected_runtime_push_readiness_excludes_remote_proof_blockers(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Push readiness blocks on capabilities, not proof artifacts it would run."""
    selected_runtime_gate_main(
        [
            "--verify-push-ready",
            "--debug-config",
            "configs/spec0001/non_eq_vae_selected_runtime_debug.json",
            "--tiny-config",
            "configs/spec0001/non_eq_vae_kaggle_tiny_overfit.json",
            "--runtime-config",
            "runs/kaggle/runtime_selection_v5/benchmark/selected_runtime.json",
            "--fixed-train-patches",
            "configs/spec0001/fixed_32_train_overfit_patches.json",
        ],
    )

    stderr = capsys.readouterr().err
    assert "fixed_32_selector_placeholder" in stderr
    assert "selected_runtime_runtime_plan_not_applied_to_training" in stderr
    assert "missing_real_tiny_overfit_proof" not in stderr
    assert "missing_real_checkpoint_resume_proof" not in stderr


def test_selected_runtime_push_readiness_remote_generate_passes() -> None:
    """Spec 0008 pre-push mode passes without a local canonical selector."""
    exit_code = selected_runtime_gate_main(
        [
            "--verify-push-ready",
            "--selector-generation-mode",
            "remote_generate",
            "--debug-config",
            "configs/spec0001/non_eq_vae_selected_runtime_debug.json",
            "--tiny-config",
            "configs/spec0001/non_eq_vae_kaggle_tiny_overfit.json",
            "--runtime-config",
            "runs/kaggle/runtime_selection_v5/benchmark/selected_runtime.json",
            "--fixed-train-patches",
            "configs/spec0001/fixed_32_train_overfit_patches.json",
        ],
    )

    assert exit_code == 0


def test_selected_runtime_verify_output_accepts_complete_artifact_contract(
    tmp_path: Path,
) -> None:
    """The Spec 0008 post-download verifier accepts the strict artifact shape."""
    output_dir, runtime_path = _write_complete_remote_output_fixture(tmp_path)

    exit_code = selected_runtime_gate_main(
        [
            "--verify-output",
            "--runtime-config",
            str(runtime_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert exit_code == 0


def test_selected_runtime_verify_output_rejects_selector_hash_mismatch(
    tmp_path: Path,
) -> None:
    """Downloaded readiness must hash-link to the downloaded fixed-32 selector."""
    output_dir, runtime_path = _write_complete_remote_output_fixture(tmp_path)
    selector_path = output_dir / "benchmark" / "fixed_32_train_overfit_patches.json"
    selector_payload = _load_json(selector_path)
    selector_payload["expected_count"] = 31
    _write_json(selector_path, selector_payload)

    blockers = selected_runtime_gate.verify_selected_runtime_debug_output(
        output_dir=output_dir,
        selected_runtime_path=runtime_path,
    )

    assert "selected_runtime_output_fixed32_selector_sha_mismatch" in blockers
    assert "selected_runtime_output_fixed32_selector_metadata_mismatch" in blockers


def test_selected_runtime_verify_output_rejects_manifest_hash_mismatch(
    tmp_path: Path,
) -> None:
    """Artifact manifest hashes are replayed against downloaded files."""
    output_dir, runtime_path = _write_complete_remote_output_fixture(tmp_path)
    train_steps = output_dir / "metrics" / "train_steps.csv"
    train_steps.write_text(
        f"{train_steps.read_text(encoding='utf-8')}# tampered\n",
        encoding="utf-8",
    )

    blockers = selected_runtime_gate.verify_selected_runtime_debug_output(
        output_dir=output_dir,
        selected_runtime_path=runtime_path,
    )

    assert "selected_runtime_output_manifest_hash_mismatch_metrics_train_steps" in (
        blockers
    )


def test_selected_runtime_verify_output_rejects_thin_gate_health_csv(
    tmp_path: Path,
) -> None:
    """Gate-health CSV content is checked, not only its summary artifact."""
    output_dir, runtime_path = _write_complete_remote_output_fixture(tmp_path)
    (output_dir / "metrics" / "gate_health.csv").write_text(
        "gate_health_status\npass\n",
        encoding="utf-8",
    )

    blockers = selected_runtime_gate.verify_selected_runtime_debug_output(
        output_dir=output_dir,
        selected_runtime_path=runtime_path,
    )

    assert "selected_runtime_output_gate_health_missing_columns" in blockers
    assert "selected_runtime_output_manifest_hash_mismatch_metrics_gate_health" in (
        blockers
    )


def test_gate_health_identity_accepts_rows_matching_the_expected_identity(
    tmp_path: Path,
) -> None:
    """Gate-health rows matching the expected identity raise no blocker.

    The no-false-positive control for the cases below: without it, a check that blocked
    unconditionally would satisfy every one of them.
    """
    assert (
        _gate_health_blockers(
            tmp_path,
            expected_row_id=EXPECTED_SELECTED_ROW_ID,
            expected_policy_id=EXPECTED_RUNTIME_POLICY_ID,
        )
        == ()
    )


@pytest.mark.parametrize(
    ("expected_row_id", "expected_policy_id", "expected_blockers"),
    [
        pytest.param(
            _REPLANNED_ROW_ID,
            EXPECTED_RUNTIME_POLICY_ID,
            (_GATE_HEALTH_ROW_BLOCKER, _GATE_HEALTH_CANDIDATE_BLOCKER),
            id="row_id_of_a_different_plan",
        ),
        pytest.param(
            EXPECTED_SELECTED_ROW_ID,
            _REPLANNED_POLICY_ID,
            (_GATE_HEALTH_POLICY_BLOCKER,),
            id="policy_id_of_a_different_plan",
        ),
        pytest.param(
            None,
            EXPECTED_RUNTIME_POLICY_ID,
            (_GATE_HEALTH_ROW_BLOCKER, _GATE_HEALTH_CANDIDATE_BLOCKER),
            id="uncomposable_row_id_fails_closed",
        ),
        pytest.param(
            EXPECTED_SELECTED_ROW_ID,
            None,
            (_GATE_HEALTH_POLICY_BLOCKER,),
            id="unusable_policy_id_fails_closed",
        ),
    ],
)
def test_gate_health_identity_is_checked_against_the_caller_supplied_identity(
    tmp_path: Path,
    expected_row_id: str | None,
    expected_policy_id: str | None,
    expected_blockers: tuple[str, ...],
) -> None:
    """Identity blockers track the caller's identity, never a fixed row.

    The rows on disk always carry the committed v5 identity, so an expectation drawn
    from a different plan must reject them, and an expectation the plan could not
    compose (None) must fail closed rather than match whatever the rows happen to say.
    """
    blockers = _gate_health_blockers(
        tmp_path,
        expected_row_id=expected_row_id,
        expected_policy_id=expected_policy_id,
    )

    assert set(blockers) == set(expected_blockers)


def test_selected_runtime_verify_output_accepts_a_replanned_identity(
    tmp_path: Path,
) -> None:
    """A re-minted plan verifies clean on its own identity (Spec 0011 S17b).

    A plan carrying a different policy -- and therefore a different composed row_id --
    passes the downloaded-output verifier once it parses and its gate-health rows carry
    that identity. Re-measuring the runtime on Kaggle is therefore a data change, not a
    source change.
    """
    output_dir, _runtime_path = _write_complete_remote_output_fixture(tmp_path)
    replanned_path = _replan_remote_output_fixture(tmp_path, output_dir)

    blockers = selected_runtime_gate.verify_selected_runtime_debug_output(
        output_dir=output_dir,
        selected_runtime_path=replanned_path,
    )

    assert blockers == ()


def test_selected_runtime_verify_output_rejects_stale_identity_after_replan(
    tmp_path: Path,
) -> None:
    """Rows still carrying the old plan's identity are rejected after a re-plan.

    Which identity is expected is plan-derived; that it is checked at all is not
    negotiable. A bundle whose gate-health rows were produced under a previous plan must
    not verify against a re-minted one.
    """
    output_dir, _runtime_path = _write_complete_remote_output_fixture(tmp_path)
    replanned_path = _replan_remote_output_fixture(
        tmp_path,
        output_dir,
        gate_health_identity=(EXPECTED_SELECTED_ROW_ID, EXPECTED_RUNTIME_POLICY_ID),
    )

    blockers = selected_runtime_gate.verify_selected_runtime_debug_output(
        output_dir=output_dir,
        selected_runtime_path=replanned_path,
    )

    assert _GATE_HEALTH_ROW_BLOCKER in blockers
    assert _GATE_HEALTH_CANDIDATE_BLOCKER in blockers
    assert _GATE_HEALTH_POLICY_BLOCKER in blockers


def test_selected_runtime_verify_output_rejects_a_plan_the_parser_rejects(
    tmp_path: Path,
) -> None:
    """A plan that self-declares different hardware is rejected on the remote path.

    The identity the verifier expects is derived from the plan, so the plan must first
    satisfy the parser's hardware/topology anchors -- otherwise a bundle could
    re-declare its own accelerator and have every gate-health row agree with the
    re-declaration. The anchors live in the parser, so the remote path has to run it.
    """
    output_dir, _runtime_path = _write_complete_remote_output_fixture(tmp_path)
    replanned_path = _write_replanned_runtime_tree(tmp_path)
    payload = _load_json(replanned_path)
    payload["accelerator_mode"] = "single_t4"
    payload["world_size"] = 1
    row_id = _REPLANNED_ROW_ID.replace("dual_t4_ddp__", "single_t4__", 1)
    payload["selected_row_id"] = row_id
    _write_json(replanned_path, payload)
    _replan_gate_health_rows(output_dir, row_id, _REPLANNED_POLICY_ID)
    _relink_remote_output_plan(output_dir, replanned_path)

    blockers = selected_runtime_gate.verify_selected_runtime_debug_output(
        output_dir=output_dir,
        selected_runtime_path=replanned_path,
    )

    assert "selected_runtime_top_level_not_dual_t4_ddp" in blockers
    assert "selected_runtime_top_level_wrong_world_size" in blockers


def test_selected_runtime_verify_output_requires_tiny_sampler_evidence(
    tmp_path: Path,
) -> None:
    """Downloaded tiny proof must show the fixed-32 full-batch sampler ran."""
    output_dir, runtime_path = _write_complete_remote_output_fixture(tmp_path)
    tiny_path = output_dir / "benchmark" / "tiny_overfit_summary.json"
    tiny = _load_json(tiny_path)
    tiny["train_sampler_policy"] = "distributed_sampler_shuffle_false_drop_last_false"
    tiny["fixed_train_repeated_to_full_batch"] = False
    tiny["observed_batch_sizes"] = [4, SELECTED_RUNTIME_BATCH_SIZE]
    _write_json(tiny_path, tiny)

    blockers = selected_runtime_gate.verify_selected_runtime_debug_output(
        output_dir=output_dir,
        selected_runtime_path=runtime_path,
    )

    assert "selected_runtime_output_tiny_sampler_policy_mismatch" in blockers
    assert "selected_runtime_output_tiny_not_repeated_to_full_batch" in blockers
    assert "selected_runtime_output_tiny_batch_sizes_not_full" in blockers
    assert (
        "selected_runtime_output_manifest_hash_mismatch_benchmark_tiny_overfit_summary_json"
        in (blockers)
    )


def test_selected_runtime_verify_output_requires_grad_scaler_init_scale(
    tmp_path: Path,
) -> None:
    """Downloaded proof must show the selected-runtime AMP scaler startup scale."""
    output_dir, runtime_path = _write_complete_remote_output_fixture(tmp_path)
    training_path = output_dir / "benchmark" / "training_summary.json"
    training = _load_json(training_path)
    amp_execution = _object(training, "amp_execution")
    del amp_execution["grad_scaler_init_scale"]
    _write_json(training_path, training)
    tiny_path = output_dir / "benchmark" / "tiny_overfit_summary.json"
    tiny = _load_json(tiny_path)
    del tiny["grad_scaler_init_scale"]
    _write_json(tiny_path, tiny)

    blockers = selected_runtime_gate.verify_selected_runtime_debug_output(
        output_dir=output_dir,
        selected_runtime_path=runtime_path,
    )

    assert "selected_runtime_output_grad_scaler_init_scale_mismatch" in blockers
    assert "selected_runtime_output_tiny_grad_scaler_init_scale_mismatch" in blockers


def test_selected_runtime_verify_output_checks_tiny_metric_rows(
    tmp_path: Path,
) -> None:
    """A passing tiny summary cannot hide skipped/nonfinite tiny metric rows."""
    output_dir, runtime_path = _write_complete_remote_output_fixture(tmp_path)
    tiny_steps = output_dir / "tiny_overfit_phase" / "metrics" / "train_steps.csv"
    rows = _tiny_train_step_rows()
    rows[3]["amp_step_skipped"] = "1"
    rows[3]["grad_norm"] = "inf"
    rows[3]["nonfinite_count"] = "125"
    _write_csv_rows(
        tiny_steps,
        selected_runtime_gate.REMOTE_DEBUG_REQUIRED_TRAIN_STEP_COLUMNS,
        rows,
    )

    blockers = selected_runtime_gate.verify_selected_runtime_debug_output(
        output_dir=output_dir,
        selected_runtime_path=runtime_path,
    )

    assert "selected_runtime_output_tiny_train_steps_amp_skip" in blockers
    assert "selected_runtime_output_tiny_train_steps_nonfinite" in blockers
    assert "selected_runtime_output_tiny_train_steps_grad_norm_nonfinite" in blockers
    assert (
        "selected_runtime_output_manifest_hash_mismatch_metrics_tiny_overfit_train_steps"
        in blockers
    )


def test_selected_runtime_push_readiness_depends_on_structured_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Config/capability booleans cannot bypass structured readiness artifacts."""
    debug_config = _config_with_ready_flags(
        tmp_path,
        Path("configs/spec0001/non_eq_vae_selected_runtime_debug.json"),
        gate_key="selected_runtime_debug",
    )
    tiny_config = _config_with_ready_flags(
        tmp_path,
        Path("configs/spec0001/non_eq_vae_kaggle_tiny_overfit.json"),
        gate_key="selected_runtime_debug_gate",
    )

    monkeypatch.setattr(
        selected_runtime_gate,
        "REAL_UBC_SELECTED_RUNTIME_TRAIN_RUNNER_IMPLEMENTED",
        True,
    )
    monkeypatch.setattr(
        selected_runtime_gate,
        "SELECTED_RUNTIME_DEBUG_WRAPPER_WIRED_TO_REAL_RUNNER",
        True,
    )
    monkeypatch.setattr(
        selected_runtime_gate,
        "SELECTED_RUNTIME_PLAN_APPLIED_TO_TRAINING",
        True,
    )
    monkeypatch.setattr(
        selected_runtime_gate,
        "fixed32_selector_status",
        _passing_selector_status,
    )

    blockers = selected_runtime_gate.verify_selected_runtime_debug_push_ready(
        debug_config_path=debug_config,
        tiny_config_path=tiny_config,
        selected_runtime_path=Path(
            "runs/kaggle/runtime_selection_v5/benchmark/selected_runtime.json",
        ),
        fixed_train_patches=Path(
            "configs/spec0001/fixed_32_train_overfit_patches.json",
        ),
    )

    assert "local_selected_runtime_readiness_status_not_pass" in blockers
    assert any(
        blocker.startswith(
            "local_selected_runtime_readiness_component_selected_runtime_plan_applied_",
        )
        for blocker in blockers
    )
    assert "selected_runtime_debug_wrapper_not_wired_to_real_runner" not in blockers


def _fake_fixed32_selector_payload() -> dict[str, object]:
    return {
        "schema_version": "spec0001.fixed_selector.v1",
        "status": "pass",
        "selector_kind": "fixed_32_train_overfit",
        "source_split": "train",
        "dataset_slug": EXPECTED_DATASET_SLUG,
        "expected_count": FIXED32_SELECTOR_COUNT,
        "selector_seed": "20260611:tiny-overfit",
        "selectors": [{"rank": rank} for rank in range(FIXED32_SELECTOR_COUNT)],
    }


def _runtime_bundle_with_proof_mutation(
    tmp_path: Path,
    mutate: Callable[[dict[str, object]], None],
) -> Path:
    runtime_dir = tmp_path / "runtime" / "benchmark"
    runtime_dir.mkdir(parents=True)
    source_dir = Path("runs/kaggle/runtime_selection_v5/benchmark")
    runtime_path = runtime_dir / "selected_runtime.json"
    proof_path = runtime_dir / "runtime_proof.json"
    payload = _load_json(source_dir / "selected_runtime.json")
    proof = _load_json(source_dir / "runtime_proof.json")
    mutate(proof)
    proof_path.write_text(
        f"{json.dumps(proof, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    artifacts = _object(payload, "artifacts")
    artifacts["runtime_proof_sha256"] = _sha256(proof_path)
    runtime_path.write_text(
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    return runtime_path


def _replace_runtime_proof_commands(
    proof: dict[str, object],
    command: str,
) -> None:
    dual_gate = _object(proof, "dual_t4_train_step_gate")
    dual_gate["child_process_launch_command"] = command
    runtime_environment = _object(proof, "runtime_environment")
    runtime_environment["child_process_launch_command"] = command


def _config_with_ready_flags(
    tmp_path: Path,
    source_path: Path,
    *,
    gate_key: str,
) -> Path:
    payload = _load_json(source_path)
    source_config = payload.get("source_config")
    if isinstance(source_config, str):
        payload["source_config"] = str(Path(source_config).resolve())
    gate = _object(payload, gate_key)
    gate["remote_pass_ready"] = True
    gate["real_train_runner_implemented"] = True
    gate["fixed_32_selector_real"] = True
    output_path = tmp_path / source_path.name
    output_path.write_text(
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    return output_path


def _passing_selector_status(
    path: Path,
    *,
    data_root: str | None,
) -> dict[str, object]:
    del data_root
    return {
        "path": str(path),
        "sha256": "synthetic-test-sha256",
        "status": "pass",
        "selector_count": FIXED32_SELECTOR_COUNT,
        "expected_count": FIXED32_SELECTOR_COUNT,
        "failure_kind": "",
        "validation_errors": [],
        "validation_detail": "",
        "canonical_real_ubc": True,
    }


def _write_complete_remote_output_fixture(tmp_path: Path) -> tuple[Path, Path]:
    output_dir = tmp_path / "downloaded-output"
    benchmark_dir = output_dir / "benchmark"
    metrics_dir = output_dir / "metrics"
    artifacts_dir = output_dir / "artifacts"
    tiny_metrics_dir = output_dir / "tiny_overfit_phase" / "metrics"
    benchmark_dir.mkdir(parents=True)
    metrics_dir.mkdir(parents=True)
    artifacts_dir.mkdir(parents=True)
    tiny_metrics_dir.mkdir(parents=True)
    runtime_path = Path(
        "runs/kaggle/runtime_selection_v5/benchmark/selected_runtime.json",
    )
    runtime_sha256 = _sha256(runtime_path)
    _write_json(
        benchmark_dir / "training_summary.json",
        {
            "status": "local_pass",
            "optimizer_steps_completed": 8,
            "amp_step_skipped_count": 0,
            "nonfinite_count": 0,
            "amp_execution": {
                "grad_scaler_init_scale": (
                    selected_runtime_gate.REMOTE_AMP_GRAD_SCALER_INIT_SCALE
                ),
            },
            "runtime_config": {"sha256": runtime_sha256},
        },
    )
    _write_json(
        benchmark_dir / "selected_runtime_debug_summary.json",
        {"remote_pass_ready": False},
    )
    _write_json(
        benchmark_dir / "local_selected_runtime_readiness.json",
        {"status": "local_pass"},
    )
    _write_json(
        benchmark_dir / "selected_runtime_plan_applied.json",
        {
            "status": "local_pass",
            "plan_applied": True,
            "expected": {
                "runner_amp_extension": {
                    "grad_scaler_init_scale": (
                        selected_runtime_gate.REMOTE_AMP_GRAD_SCALER_INIT_SCALE
                    ),
                },
            },
            "observed": {
                "runner_amp_extension": {
                    "grad_scaler_init_scale": (
                        selected_runtime_gate.REMOTE_AMP_GRAD_SCALER_INIT_SCALE
                    ),
                },
            },
        },
    )
    _write_json(
        benchmark_dir / "checkpoint_resume_proof.json",
        {
            "status": "local_pass",
            "loaded_successful_optimizer_update_count": 4,
            "additional_optimizer_steps": 4,
        },
    )
    _write_json(benchmark_dir / "gate_health_summary.json", {"status": "local_pass"})
    selector_path = benchmark_dir / "fixed_32_train_overfit_patches.json"
    _write_json(selector_path, _canonical_fixed32_selector_payload())
    selector_sha256 = _sha256(selector_path)
    _write_json(
        benchmark_dir / "fixed32_selector_readiness.json",
        {
            "status": "pass",
            "fixed_32_selector_real": True,
            "selector_status": {
                "status": "pass",
                "canonical_real_ubc": True,
                "selector_count": FIXED_32_TRAIN_OVERFIT_COUNT,
                "sha256": selector_sha256,
            },
        },
    )
    _write_json(
        benchmark_dir / "tiny_overfit_summary.json",
        {
            "status": "local_pass",
            "patch_count": 32,
            "optimizer_steps": 128,
            "successful_metric_row_count": 256,
            "amp_step_skipped_count": 0,
            "nonfinite_count": 0,
            "grad_scaler_init_scale": (
                selected_runtime_gate.REMOTE_AMP_GRAD_SCALER_INIT_SCALE
            ),
            "train_sampler_policy": "fixed32_tiny_full_batch_repeated",
            "train_effective_global_epoch_samples": (
                REMOTE_TINY_EFFECTIVE_GLOBAL_EPOCH_SAMPLES
            ),
            "train_effective_per_rank_epoch_samples": (
                REMOTE_TINY_EFFECTIVE_PER_RANK_EPOCH_SAMPLES
            ),
            "fixed_train_repeated_to_full_batch": True,
            "observed_batch_sizes": [SELECTED_RUNTIME_BATCH_SIZE],
            "l1_improvement_fraction": 0.02,
            "recon_loss_improvement_fraction": 0.02,
        },
    )
    _write_json(
        benchmark_dir / "selected_runtime_gate_summary.json",
        {"status": "local_pass"},
    )
    _write_csv_rows(
        metrics_dir / "gate_health.csv",
        selected_runtime_gate.GATE_HEALTH_COLUMNS,
        [_gate_health_row()],
    )
    _write_csv_rows(
        metrics_dir / "train_steps.csv",
        selected_runtime_gate.REMOTE_DEBUG_REQUIRED_TRAIN_STEP_COLUMNS,
        _train_step_rows(),
    )
    _write_csv_rows(
        tiny_metrics_dir / "train_steps.csv",
        selected_runtime_gate.REMOTE_DEBUG_REQUIRED_TRAIN_STEP_COLUMNS,
        _tiny_train_step_rows(),
    )
    (artifacts_dir / "reconstruction_samples.pt").write_bytes(b"nonblank")
    _write_json(
        benchmark_dir / "artifact_manifest.json",
        {
            "status": "local_pass",
            "reconstruction_sample_nonblank": True,
            "artifact_hashes": _remote_manifest_hashes(output_dir),
        },
    )
    return output_dir, runtime_path


def _canonical_fixed32_selector_payload() -> dict[str, object]:
    requirements = selected_runtime_gate.canonical_real_ubc_requirements()
    selectors = [
        {
            "rank": rank,
            "source_split": "train",
            "file_index": rank,
            "row_index": rank,
            "sample_id": f"train_{rank:06d}",
            "wsi_id": f"real_wsi_{rank:04d}",
            "label": rank % 5,
            "x": rank,
            "y": rank * 2,
            "selection_key_sha256": hashlib.sha256(
                f"selection-{rank}".encode(),
            ).hexdigest(),
            "patch_sha256": hashlib.sha256(
                f"patch-{rank}".encode(),
            ).hexdigest(),
        }
        for rank in range(FIXED_32_TRAIN_OVERFIT_COUNT)
    ]
    return {
        "schema_version": FIXED_SELECTOR_SCHEMA_VERSION,
        "status": FIXED_SELECTOR_READY_STATUS,
        "selector_kind": FIXED_32_TRAIN_OVERFIT_KIND,
        "source_split": "train",
        "expected_count": FIXED_32_TRAIN_OVERFIT_COUNT,
        "selector_seed": FIXED_32_TRAIN_OVERFIT_SEED,
        "masked_holdout_exclusion": "docs/data/ubc_ocean_masked_holdout_ids.csv",
        "source": {
            "dataset_slug": requirements["dataset_slug"],
            "data_root": "/kaggle/input/patches-pre-shuffled-ubc-ocean",
            "source_split": "train",
            "csv_path": f"/kaggle/input/dataset/{requirements['train_csv_filename']}",
            "csv_sha256": requirements["train_csv_sha256"],
            "bin_path": f"/kaggle/input/dataset/{requirements['train_bin_filename']}",
            "bin_file_size": requirements["train_bin_file_size"],
            "header_sha256": hashlib.sha256(b"header").hexdigest(),
            "header": {
                "crc32": requirements["train_header_crc32"],
                "patch_count": requirements["patch_count"],
                "channels": requirements["channels"],
                "height": requirements["height"],
                "width": requirements["width"],
                "version": 1,
                "layout": requirements["layout"],
            },
            "row_count": requirements["row_count"],
            "patch_count": requirements["patch_count"],
            "idx_policy": requirements["idx_policy"],
            "crc_checked": requirements["crc_checked"],
        },
        "selectors": selectors,
    }


def _write_replanned_runtime_tree(tmp_path: Path) -> Path:
    """Re-mint the committed plan tree under a new runtime policy, kept parse-clean.

    A real Kaggle re-mint rewrites the plan, the snapshot cells that echo its identity,
    and the runtime-proof artifact the plan hash-links to. Copying the whole committed
    tree keeps those sibling links resolvable, so the result is a plan the parser
    accepts on its own merits -- the only honest way to prove the gate accepts a
    re-minted identity rather than accepting an unparseable plan.

    Returns:
        The path of the re-minted plan inside the copied tree.

    """
    tree = tmp_path / "replanned_runtime_selection"
    shutil.copytree(Path("runs/kaggle/runtime_selection_v5"), tree)
    plan_path = tree / "benchmark" / "selected_runtime.json"
    payload = _load_json(plan_path)
    payload["runtime_policy_id"] = _REPLANNED_POLICY_ID
    payload["selected_row_id"] = _REPLANNED_ROW_ID
    snapshot = cast("dict[str, object]", payload["selected_row_snapshot"])
    snapshot["runtime_policy_id"] = _REPLANNED_POLICY_ID
    snapshot["row_id"] = _REPLANNED_ROW_ID

    artifacts = cast("dict[str, object]", payload["artifacts"])
    proof_path = tree / cast("str", artifacts["runtime_proof"])
    proof = _load_json(proof_path)
    write_decision = cast("dict[str, object]", proof["selected_runtime_write_decision"])
    write_decision["selected_row_id"] = _REPLANNED_ROW_ID
    efficiency = cast("dict[str, object]", proof["efficiency_followup"])
    efficiency["selected_row_id"] = _REPLANNED_ROW_ID
    efficiency["selected_runtime_policy_id"] = _REPLANNED_POLICY_ID
    _write_json(proof_path, proof)
    artifacts["runtime_proof_sha256"] = _sha256(proof_path)
    _write_json(plan_path, payload)
    return plan_path


def _replan_remote_output_fixture(
    tmp_path: Path,
    output_dir: Path,
    *,
    gate_health_identity: tuple[str, str] = (_REPLANNED_ROW_ID, _REPLANNED_POLICY_ID),
) -> Path:
    """Repoint a downloaded-output fixture at a re-minted plan carrying a new identity.

    Rewrites exactly what a real Kaggle re-mint would change and nothing else: the plan
    tree (see ``_write_replanned_runtime_tree``), the gate-health rows that must carry
    the re-minted identity, the training summary's hash link to the plan, and the
    artifact manifest the gate replays. ``gate_health_identity`` defaults to the
    re-minted identity; pass the old one to model a stale bundle.

    Returns:
        The path of the re-minted plan.

    """
    replanned_path = _write_replanned_runtime_tree(tmp_path)
    row_id, policy_id = gate_health_identity
    _replan_gate_health_rows(output_dir, row_id, policy_id)
    _relink_remote_output_plan(output_dir, replanned_path)
    return replanned_path


def _replan_gate_health_rows(output_dir: Path, row_id: str, policy_id: str) -> None:
    """Rewrite the downloaded gate-health rows to carry the given identity."""
    row = _gate_health_row()
    row.update(
        {
            "row_id": row_id,
            "candidate_row_id": row_id,
            "runtime_policy_id": policy_id,
        },
    )
    _write_csv_rows(
        output_dir / "metrics" / "gate_health.csv",
        selected_runtime_gate.GATE_HEALTH_COLUMNS,
        [row],
    )


def _relink_remote_output_plan(output_dir: Path, plan_path: Path) -> None:
    """Re-point the summary's plan hash link and replay the manifest hashes.

    Keeps the bundle's own integrity chain intact around a re-minted plan, so a test
    reaches the identity check instead of tripping the hash-link blockers first.
    """
    summary_path = output_dir / "benchmark" / "training_summary.json"
    summary = _load_json(summary_path)
    summary["runtime_config"] = {"sha256": _sha256(plan_path)}
    _write_json(summary_path, summary)

    manifest_path = output_dir / "benchmark" / "artifact_manifest.json"
    manifest = _load_json(manifest_path)
    manifest["artifact_hashes"] = _remote_manifest_hashes(output_dir)
    _write_json(manifest_path, manifest)


def _gate_health_blockers(
    tmp_path: Path,
    *,
    expected_row_id: str | None,
    expected_policy_id: str | None,
) -> tuple[str, ...]:
    """Run the gate-health blocker check over one committed-v5-identity row.

    Returns:
        The blockers raised for rows carrying the committed v5 identity.

    """
    path = tmp_path / "gate_health.csv"
    _write_csv_rows(
        path,
        selected_runtime_gate.GATE_HEALTH_COLUMNS,
        [_gate_health_row()],
    )
    return selected_runtime_gate._remote_output_gate_health_blockers(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        path,
        expected_row_id=expected_row_id,
        expected_policy_id=expected_policy_id,
    )


def _gate_health_row() -> dict[str, str]:
    row = dict.fromkeys(selected_runtime_gate.GATE_HEALTH_COLUMNS, "")
    row.update(
        {
            "run_name": "remote_debug",
            "benchmark_kind": "kaggle_selected_runtime_real_ubc_runner",
            "benchmark_source": "local_selected_runtime_train_runner_rank0",
            "full_run_eligible": "false",
            "accelerator_mode": "dual_t4_ddp",
            "machine_shape": "NvidiaTeslaT4",
            "row_id": EXPECTED_SELECTED_ROW_ID,
            "candidate_row_id": EXPECTED_SELECTED_ROW_ID,
            "runtime_policy_id": EXPECTED_RUNTIME_POLICY_ID,
            "optimizer_step": "8",
            "module": "rank0:encoder_gate",
            "gate_kind": "gated_scalar_activation",
            "num_channels": "16",
            "num_elements": "16",
            "gate_force_fp32": "true",
            "input_dtype": "torch.float16",
            "gate_math_dtype": "torch.float32",
            "gate_tensor_dtype": "torch.float32",
            "output_dtype": "torch.float16",
            "requested_autocast_dtype": "float16",
            "precision_proof_status": "pass",
            "gate_health_status": "pass",
        },
    )
    for column in selected_runtime_gate.REMOTE_GATE_HEALTH_FINITE_COLUMNS:
        row[column] = "0.1"
    for column in selected_runtime_gate.REMOTE_GATE_HEALTH_SATURATION_COLUMNS:
        row[column] = "0.0"
    return row


def _train_step_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for step, batch_size in zip((5, 6, 7, 8), (12, 12, 8, 12), strict=True):
        row: dict[str, str] = dict.fromkeys(
            selected_runtime_gate.REMOTE_DEBUG_REQUIRED_TRAIN_STEP_COLUMNS,
            "",
        )
        row.update(
            {
                "event_id": f"rank0_train_step_{step:06d}",
                "rank": "0",
                "optimizer_step_index": str(step - 1),
                "optimizer_step": str(step),
                "successful_optimizer_update_count": str(step),
                "split": "train",
                "loss": "1.0",
                "recon_loss": "1.0",
                "l1_loss": "0.5",
                "ssim_loss": "0.5",
                "ssim_metric": "0.5",
                "kl_loss": "0.1",
                "beta": "0.1",
                "grad_norm": "1.0",
                "param_update_norm": "0.01",
                "nonfinite_count": "0",
                "batch_size": str(batch_size),
                "precision_policy": "amp_conservative",
                "amp_enabled": "true",
                "autocast_dtype": "float16",
                "grad_scaler_enabled": "true",
                "fp32_loss": "true",
                "torch_compile_enabled": "false",
                "compile_scope": "none",
                "corruption_strategy": "indexed_masked",
                "amp_step_skipped": "0",
                "checkpoint_path": "",
            },
        )
        rows.append(row)
    return rows


def _tiny_train_step_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for rank in range(2):
        for step in range(1, selected_runtime_gate.REMOTE_TINY_MAX_STEP + 1):
            row: dict[str, str] = dict.fromkeys(
                selected_runtime_gate.REMOTE_DEBUG_REQUIRED_TRAIN_STEP_COLUMNS,
                "",
            )
            row.update(
                {
                    "event_id": f"rank{rank}_train_step_{step:06d}",
                    "rank": str(rank),
                    "optimizer_step_index": str(step - 1),
                    "optimizer_step": str(step),
                    "successful_optimizer_update_count": str(step),
                    "split": "train",
                    "loss": "1.0",
                    "recon_loss": "1.0",
                    "l1_loss": "0.5",
                    "ssim_loss": "0.5",
                    "ssim_metric": "0.5",
                    "kl_loss": "0.1",
                    "beta": "0.1",
                    "grad_norm": "1.0",
                    "param_update_norm": "0.01",
                    "nonfinite_count": "0",
                    "batch_size": str(SELECTED_RUNTIME_BATCH_SIZE),
                    "precision_policy": "amp_conservative",
                    "amp_enabled": "true",
                    "autocast_dtype": "float16",
                    "grad_scaler_enabled": "true",
                    "fp32_loss": "true",
                    "torch_compile_enabled": "false",
                    "compile_scope": "none",
                    "corruption_strategy": "indexed_masked",
                    "amp_step_skipped": "0",
                    "checkpoint_path": "",
                },
            )
            rows.append(row)
    return rows


def _remote_manifest_hashes(output_dir: Path) -> dict[str, str]:
    names = {
        "benchmark:checkpoint_resume_proof.json": output_dir
        / "benchmark"
        / "checkpoint_resume_proof.json",
        "benchmark:fixed32_selector_readiness.json": output_dir
        / "benchmark"
        / "fixed32_selector_readiness.json",
        "benchmark:fixed_32_train_overfit_patches.json": output_dir
        / "benchmark"
        / "fixed_32_train_overfit_patches.json",
        "benchmark:gate_health_summary.json": output_dir
        / "benchmark"
        / "gate_health_summary.json",
        "benchmark:local_selected_runtime_readiness.json": output_dir
        / "benchmark"
        / "local_selected_runtime_readiness.json",
        "benchmark:selected_runtime_debug_summary.json": output_dir
        / "benchmark"
        / "selected_runtime_debug_summary.json",
        "benchmark:selected_runtime_gate_summary.json": output_dir
        / "benchmark"
        / "selected_runtime_gate_summary.json",
        "benchmark:selected_runtime_plan_applied.json": output_dir
        / "benchmark"
        / "selected_runtime_plan_applied.json",
        "benchmark:tiny_overfit_summary.json": output_dir
        / "benchmark"
        / "tiny_overfit_summary.json",
        "benchmark:training_summary.json": output_dir
        / "benchmark"
        / "training_summary.json",
        "metrics:gate_health": output_dir / "metrics" / "gate_health.csv",
        "metrics:train_steps": output_dir / "metrics" / "train_steps.csv",
        "metrics:tiny_overfit_train_steps": output_dir
        / "tiny_overfit_phase"
        / "metrics"
        / "train_steps.csv",
        "artifact:reconstruction_samples": output_dir
        / "artifacts"
        / "reconstruction_samples.pt",
    }
    return {name: _sha256(path) for name, path in sorted(names.items())}


def _write_csv_rows(
    path: Path,
    columns: tuple[str, ...],
    rows: list[dict[str, str]],
) -> None:
    payload = [",".join(columns)]
    payload.extend(",".join(row[column] for column in columns) for row in rows)
    path.write_text(f"{'\n'.join(payload)}\n", encoding="utf-8")


def _write_complete_data_root(root: Path) -> None:
    write_synthetic_patch_shard(
        bin_path=root / TRAIN_BIN_NAME,
        csv_path=root / TRAIN_CSV_NAME,
        spec=SyntheticPatchSpec(count=TRAIN_PATCH_COUNT, image_size=PATCH_SIZE),
        include_idx=False,
    )
    write_synthetic_patch_shard(
        bin_path=root / VALIDATION_BIN_NAME,
        csv_path=root / VALIDATION_CSV_NAME,
        spec=SyntheticPatchSpec(count=VALIDATION_PATCH_COUNT, image_size=PATCH_SIZE),
        include_idx=True,
    )


def _load_json(path: Path) -> dict[str, object]:
    payload = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(payload, dict):
        raise TypeError(path)
    return cast("dict[str, object]", payload)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _object(payload: dict[str, object], key: str) -> dict[str, object]:
    value = payload[key]
    if not isinstance(value, dict):
        raise TypeError(key)
    return cast("dict[str, object]", value)


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(value)
    items = cast("list[object]", value)
    if not all(isinstance(item, str) for item in items):
        message = "expected list of strings"
        raise TypeError(message)
    return cast("list[str]", items)
