# Copyright 2026 HiperMaximus
"""Tests for the selected-runtime debug/resume/tiny gate contract."""

from __future__ import annotations

import hashlib
import json
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
    FIXED_32_TRAIN_OVERFIT_KIND,
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
MASKED_HOLDOUT_WSI = "synthetic_wsi_0000"


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
    assert "real_ubc_selected_runtime_train_runner_not_implemented" in blockers
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
    assert readiness["real_train_runner_implemented"] is False
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
    assert "selected_runtime_row_not_v5_fallback" in _string_list(
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
    assert "real_ubc_selected_runtime_train_runner_not_implemented" in blockers
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
    assert "real_ubc_selected_runtime_train_runner_not_implemented" in stderr
    assert "fixed_32_selector_placeholder" in stderr
    assert "missing_real_tiny_overfit_proof" not in stderr
    assert "missing_real_checkpoint_resume_proof" not in stderr


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
        "SELECTED_RUNTIME_PLAN_APPLIED_TO_TRAINING",
        True,
    )
    monkeypatch.setattr(
        selected_runtime_gate,
        "_selector_status",
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
    assert "real_ubc_selected_runtime_train_runner_not_implemented" not in blockers


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
