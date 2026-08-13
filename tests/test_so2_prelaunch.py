# Copyright 2026 HiperMaximus
"""Focused contracts for the one-off Spec 0016 SO2 prelaunch path."""
# pyright: reportPrivateUsage=false, reportArgumentType=false
# ruff: noqa: PLC2701, PLR2004

from __future__ import annotations

import csv
import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import torch

if TYPE_CHECKING:
    from collections.abc import Iterable

from eqvae.benchmarking.so2_prelaunch import (
    _cost_link_errors,
    _gate_csv_errors,
    execution_identity,
    validate_prelaunch_artifacts,
    validate_prelaunch_verdict,
)
from eqvae.config import resolve_json_config
from eqvae.data.training_batches import PatchTrainingBatch
from eqvae.models.registry import MODEL_KIND_SO2_FIXED, build_model
from eqvae.models.so2_architecture_probe import FixedF01RadialGate
from eqvae.training.selected_runtime import parse_selected_runtime_plan
from eqvae.training.selected_runtime_runner import (
    _AmpExecution,
    _gate_health_rows,
    _initial_so2_gate_snapshots,
    _RuntimeIdentity,
    _tiny_overfit_summary,
)

_RUNTIME = Path("configs/spec0001/non_eq_vae_selected_runtime.json")
_NORMAL_BASE = Path("configs/spec0001/non_eq_vae_model_base.json")
_SO2_BASE = Path("configs/spec0016/so2_training_base.json")
_DEBUG = Path("configs/spec0016/so2_selected_runtime_debug.json")
_TINY = Path("configs/spec0016/so2_kaggle_tiny_overfit.json")
_FULL = Path("configs/spec0016/so2_selected_runtime_full.json")


def test_so2_configs_keep_the_normal_training_coordinate_but_replace_the_model() -> (
    None
):
    """Parity means one fixed SO2 model on the proven normal data/runtime contract."""
    for path in (_DEBUG, _TINY, _FULL):
        effective = resolve_json_config(path).effective_config
        model = cast("dict[str, object]", effective["model"])
        data = cast("dict[str, object]", effective["data"])
        objective = cast("dict[str, object]", effective["objective"])
        beta = cast("dict[str, object]", objective["beta"])
        expected = cast("dict[str, object]", model["expected_model_count"])
        assert model["kind"] == MODEL_KIND_SO2_FIXED
        assert expected["total_learned_parameters"] == 1_180_035
        assert expected["gate_family_row_count"] == 68
        assert data["dataset_slug"] == "maximusshtefan/patches-pre-shuffled-ubc-ocean"
        assert math.isclose(cast("float", beta["target"]), 0.01)
    plan = parse_selected_runtime_plan(_RUNTIME)
    assert plan.per_device_batch_size == 25
    assert plan.global_batch_size == 50


def test_so2_base_matches_normal_training_contract_except_fixed_model_and_beta() -> (
    None
):
    """The first attempt changes architecture and the already-selected beta only."""
    normal = resolve_json_config(_NORMAL_BASE).effective_config
    so2 = resolve_json_config(_SO2_BASE).effective_config
    assert so2["seeds"] == normal["seeds"]
    assert so2["optimizer"] == normal["optimizer"]
    assert so2["corruption"] == normal["corruption"]
    assert so2["fixed25_equivariance"] == normal["fixed25_equivariance"]
    expected_objective = deepcopy(normal["objective"])
    cast("dict[str, object]", cast("dict[str, object]", expected_objective)["beta"])[
        "target"
    ] = 0.01
    assert so2["objective"] == expected_objective


def test_actual_so2_gate_capture_requires_all_68_gradients_and_updates() -> None:
    """Every real F0/F1 family must prove activation, gradient, and parameter motion."""
    model = build_model(MODEL_KIND_SO2_FIXED)
    initial = _initial_so2_gate_snapshots(model)
    gates = [
        module for module in model.modules() if isinstance(module, FixedF01RadialGate)
    ]
    gate_parameters = [
        parameter
        for gate in gates
        for parameter in (gate.f0_a, gate.f0_b, gate.f1_a, gate.f1_b)
    ]
    with torch.no_grad():
        for parameter in gate_parameters:
            parameter.add_(0.01)
            parameter.grad = torch.ones_like(parameter)
    batch = PatchTrainingBatch(
        images_uint8=torch.randint(0, 256, (1, 3, 32, 32), dtype=torch.uint8),
        split="validation",
        file_indices=(0,),
        row_indices=(0,),
        wsi_ids=("wsi",),
        labels=(0,),
        xs=(0,),
        ys=(0,),
        semantic_sample_keys=("key",),
        sample_ids=("sample",),
    )
    amp = _AmpExecution(
        enabled=False,
        grad_scaler_enabled=False,
        grad_scaler_init_scale=16_384.0,
        autocast_dtype="float16",
        requested_autocast_dtype="float16",
        local_amp_status="local_cpu",
    )
    rows = _gate_health_rows(
        run_name="spec0016_test",
        plan=parse_selected_runtime_plan(_RUNTIME),
        probe=SimpleNamespace(
            accelerator_mode="local_cpu",
            machine_shape="local_cpu",
        ),
        amp=amp,
        model=model,
        optimizer_step=128,
        rank=0,
        settings=SimpleNamespace(image_size=32),
        data_surface=SimpleNamespace(validation_loader=(batch,)),
        device=torch.device("cpu"),
        initial=initial,
    )
    assert len(rows) == 68
    assert len({row["module"] for row in rows}) == 68
    assert {row["gate_kind"] for row in rows} == {"f0_scalar", "f1_radial"}
    assert all(row["gate_health_status"] == "pass" for row in rows)
    assert all(float(row["input_rms"]) > 0.0 for row in rows)
    assert all(row["gate_math_dtype"] == "float32" for row in rows)

    gates[0].f0_a.grad = None
    failed = _gate_health_rows(
        run_name="spec0016_test",
        plan=parse_selected_runtime_plan(_RUNTIME),
        probe=SimpleNamespace(
            accelerator_mode="local_cpu",
            machine_shape="local_cpu",
        ),
        amp=amp,
        model=model,
        optimizer_step=128,
        rank=0,
        settings=SimpleNamespace(image_size=32),
        data_surface=SimpleNamespace(validation_loader=(batch,)),
        device=torch.device("cpu"),
        initial=initial,
    )
    assert sum(row["gate_health_status"] == "fail" for row in failed) == 1

    gates[0].f0_a.grad = torch.ones_like(gates[0].f0_a)

    def perturb_output(
        _module: torch.nn.Module,
        _arguments: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> torch.Tensor:
        return output + 1.0

    perturbation = gates[0].register_forward_hook(
        perturb_output,
    )
    mismatched = _gate_health_rows(
        run_name="spec0016_test",
        plan=parse_selected_runtime_plan(_RUNTIME),
        probe=SimpleNamespace(
            accelerator_mode="local_cpu",
            machine_shape="local_cpu",
        ),
        amp=amp,
        model=model,
        optimizer_step=128,
        rank=0,
        settings=SimpleNamespace(image_size=32),
        data_surface=SimpleNamespace(validation_loader=(batch,)),
        device=torch.device("cpu"),
        initial=initial,
    )
    perturbation.remove()
    broken = [row for row in mismatched if row["module"].startswith("stem_gate:")]
    assert len(broken) == 2
    assert {row["precision_proof_status"] for row in broken} == {"fail"}
    assert {row["gate_health_status"] for row in broken} == {"fail"}

    with torch.no_grad():
        gates[0].f1_a.copy_(initial["stem_gate.f1_a"])
    gates[0].f1_a.grad = torch.zeros_like(gates[0].f1_a)
    zeroed = _gate_health_rows(
        run_name="spec0016_test",
        plan=parse_selected_runtime_plan(_RUNTIME),
        probe=SimpleNamespace(accelerator_mode="local_cpu", machine_shape="local_cpu"),
        amp=amp,
        model=model,
        optimizer_step=128,
        rank=0,
        settings=SimpleNamespace(image_size=32),
        data_surface=SimpleNamespace(validation_loader=(batch,)),
        device=torch.device("cpu"),
        initial=initial,
    )
    f1 = next(row for row in zeroed if row["module"] == "stem_gate:f1_radial")
    assert f1["gate_health_status"] == "fail"


def test_so2_tiny_verdict_requires_two_stable_settled_rank_windows() -> None:
    """Learning alone cannot authorize a full run without measured two-rank cost."""
    plan = parse_selected_runtime_plan(_RUNTIME)
    rows = tuple(
        {
            "successful_optimizer_update_count": str(step),
            "rank": str(rank),
            "l1_loss": str(1.0 - 0.5 * step / 128),
            "recon_loss": str(1.0 - 0.4 * step / 128),
            "nonfinite_count": "0",
            "amp_step_skipped": "0",
            "batch_size": "25",
        }
        for step in range(1, 129)
        for rank in range(2)
    )
    performance = tuple(
        {
            "rank": rank,
            "settled_update_count": 20,
            "mean_step_ms": 1000.0 + rank,
            "post_settle_graph_break_count": 0,
            "post_settle_recompile_count": 0,
        }
        for rank in range(2)
    )
    kwargs = {
        "runtime_identity": _RuntimeIdentity(
            path=_RUNTIME,
            sha256=plan.artifact_sha256,
            selected_row_id=plan.selected_row_id,
            runtime_policy_id=plan.runtime_policy_id,
        ),
        "plan": plan,
        "corruption_strategy": "compiled_fastpath_inline_stain",
        "data_surface": SimpleNamespace(
            fixed_train_patch_count=32,
            fixed_train_patches=Path("fixed32.json"),
            fixed_train_patches_sha256="a" * 64,
            train_sampler_policy="fixed32_tiny_full_batch_repeated",
            train_effective_global_epoch_samples=50,
            train_effective_per_rank_epoch_samples=25,
        ),
        "metric_rows": rows,
        "gate_health_summary": {"status": "local_pass"},
        "performance": performance,
        "require_performance": True,
    }
    passed = _tiny_overfit_summary(**kwargs)
    assert passed["status"] == "local_pass"
    assert math.isclose(
        cast("float", passed["projected_epoch_seconds"]),
        plan.optimizer_updates_per_epoch * 1.001,
    )

    bad_performance = list(performance)
    bad_performance[1] = {
        **bad_performance[1],
        "post_settle_recompile_count": 1,
    }
    failed = _tiny_overfit_summary(
        **{**kwargs, "performance": tuple(bad_performance)},
    )
    assert failed["status"] == "fail"


def test_full_authorization_rejects_stale_or_unstable_prelaunch(
    tmp_path: Path,
) -> None:
    """A prior green label cannot authorize changed code or hidden rank instability."""
    repository = Path(__file__).resolve().parents[1]
    rank_metrics = [
        {
            "rank": rank,
            "settled_update_count": 20,
            "mean_step_ms": 1000.0 + rank,
            "mean_data_wait_ms": 2.0,
            "data_wait_fraction": 2.0 / (1000.0 + rank),
            "settled_step_ms": [1000.0 + rank] * 20,
            "data_wait_ms": [2.0] * 20,
            "peak_allocated_mib": 8000.0,
            "peak_reserved_mib": 9000.0,
            "total_device_memory_mib": 15000.0,
            "reserved_headroom_fraction": 0.4,
            "post_settle_graph_break_count": 0,
            "post_settle_recompile_count": 0,
        }
        for rank in range(2)
    ]
    verdict: dict[str, object] = {
        "schema_version": "spec0016.so2_prelaunch_verdict.v1",
        "status": "pass",
        "full_run_eligible": False,
        "source_git_commit": "a" * 40,
        "source_git_dirty": False,
        "checks": {
            "debug": True,
            "resume": True,
            "tiny": True,
            "gates": True,
            "performance": True,
        },
        "selected_runtime": {"per_device_batch_size": 25, "global_batch_size": 50},
        "execution_identity_sha256": execution_identity(repository),
        "projected_epoch_seconds": 6006.0,
        "slower_rank_mean_step_ms": 1001.0,
        "settled_real_loader_rank_metrics": rank_metrics,
    }
    path = tmp_path / "verdict.json"
    path.write_text(f"{json.dumps(verdict)}\n", encoding="utf-8")
    assert (
        validate_prelaunch_verdict(
            path,
            repo_root=repository,
            expected_source_commit="a" * 40,
        )
        == ()
    )

    rank_metrics[1]["post_settle_recompile_count"] = 1
    path.write_text(f"{json.dumps(verdict)}\n", encoding="utf-8")
    assert "prelaunch_rank_metric_invalid" in validate_prelaunch_verdict(
        path,
        repo_root=repository,
        expected_source_commit="a" * 40,
    )
    rank_metrics[1]["post_settle_recompile_count"] = 0

    cast("dict[str, str]", verdict["execution_identity_sha256"])["src/eqvae"] = "0" * 64
    path.write_text(f"{json.dumps(verdict)}\n", encoding="utf-8")
    assert "prelaunch_execution_identity_stale" in validate_prelaunch_verdict(
        path,
        repo_root=repository,
        expected_source_commit="a" * 40,
    )


def test_full_artifact_validator_rejects_standalone_green_label(tmp_path: Path) -> None:
    """A compact pass without its hashed phase evidence cannot authorize full."""
    repository = Path(__file__).resolve().parents[1]
    benchmark = tmp_path / "benchmark"
    benchmark.mkdir()
    verdict: dict[str, object] = {
        "schema_version": "spec0016.so2_prelaunch_verdict.v1",
        "status": "pass",
        "full_run_eligible": False,
        "source_git_commit": "a" * 40,
        "source_git_dirty": False,
        "checks": {
            "debug": True,
            "resume": True,
            "tiny": True,
            "gates": True,
            "performance": True,
        },
        "selected_runtime": {"per_device_batch_size": 25, "global_batch_size": 50},
        "execution_identity_sha256": execution_identity(repository),
        "projected_epoch_seconds": 6000.0,
        "slower_rank_mean_step_ms": 1000.0,
        "settled_real_loader_rank_metrics": [],
    }
    (benchmark / "so2_prelaunch_verdict.json").write_text(
        f"{json.dumps(verdict)}\n",
        encoding="utf-8",
    )
    blockers = validate_prelaunch_artifacts(tmp_path, repo_root=repository)
    assert "prelaunch_phase_manifest_hashes_missing" in blockers
    assert "prelaunch_debug_resume_manifest_hash_mismatch" in blockers
    assert "prelaunch_tiny_overfit_manifest_hash_mismatch" in blockers


def test_downloaded_gate_csv_requires_exact_unique_68_family_identities(
    tmp_path: Path,
) -> None:
    """Renamed, duplicated, mispaired, or dropped gate-family rows fail closed."""
    model = build_model(MODEL_KIND_SO2_FIXED)
    named_modules = cast(
        "Iterable[tuple[str, torch.nn.Module]]",
        model.named_modules(),
    )
    identities = [
        (name, family)
        for name, module in named_modules
        if isinstance(module, FixedF01RadialGate)
        for family in ("f0_scalar", "f1_radial")
    ]
    rows = [
        {
            "module": f"{name}:{family}",
            "gate_kind": family,
            "gate_health_status": "pass",
            "precision_proof_status": "pass",
            "input_dtype": "float16",
            "output_dtype": "float16",
            "gate_tensor_dtype": "float16",
            "gate_math_dtype": "float32",
            "input_rms": "1",
            "output_rms": "1",
            "a_grad_norm": "1",
            "b_grad_norm": "1",
            "a_update_to_param_norm": "1",
            "b_update_to_param_norm": "1",
        }
        for name, family in identities
    ]
    path = tmp_path / "gate_health.csv"

    def write_rows(values: list[dict[str, str]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(values)

    write_rows(rows)
    assert _gate_csv_errors(path) == []
    for mutated in (
        rows[:-1],
        [*rows[:-1], rows[0]],
        [{**rows[0], "module": "invented:f0_scalar"}, *rows[1:]],
        [{**rows[0], "gate_kind": "f1_radial"}, *rows[1:]],
        [{**rows[0], "a_update_to_param_norm": "nan"}, *rows[1:]],
    ):
        write_rows(mutated)
        assert _gate_csv_errors(path) == ["prelaunch_gate_csv_invalid"]


def test_compact_cost_must_equal_the_hashed_tiny_summary() -> None:
    """The user cannot be shown cost numbers detached from measured evidence."""
    metrics = [{"rank": 0, "mean_step_ms": 1000.0}]
    verdict: dict[str, object] = {
        "settled_real_loader_rank_metrics": metrics,
        "slower_rank_mean_step_ms": 1000.0,
        "projected_epoch_seconds": 6000.0,
    }
    tiny = deepcopy(verdict)
    assert _cost_link_errors(verdict, tiny) == []
    mutations: tuple[tuple[str, object], ...] = (
        ("settled_real_loader_rank_metrics", []),
        ("slower_rank_mean_step_ms", 1.0),
        ("projected_epoch_seconds", 6.0),
    )
    for field, replacement in mutations:
        mutated = deepcopy(tiny)
        mutated[field] = replacement
        assert _cost_link_errors(verdict, mutated) == [
            "prelaunch_cost_evidence_mismatch",
        ]


def test_complete_artifact_root_passes_then_rejects_checkpoint_lineage(
    tmp_path: Path,
) -> None:
    """The complete positive proof must break if phase-1 checkpoint bytes change."""
    repository = Path(__file__).resolve().parents[1]
    root = _write_valid_artifact_root(tmp_path, repository)
    assert (
        validate_prelaunch_artifacts(
            root,
            repo_root=repository,
            expected_source_commit="a" * 40,
        )
        == ()
    )
    checkpoint = root / "debug_phase1/checkpoints/step_000004.pt"
    checkpoint.write_bytes(b"changed checkpoint")
    blockers = validate_prelaunch_artifacts(
        root,
        repo_root=repository,
        expected_source_commit="a" * 40,
    )
    assert any(
        "artifact_hash_mismatch:checkpoint:step_000004.pt" in item for item in blockers
    )
    assert "prelaunch_debug_resume_proof_invalid" in blockers


def _write_valid_artifact_root(root: Path, repository: Path) -> Path:  # noqa: PLR0914
    metrics = [
        {
            "rank": rank,
            "settled_update_count": 20,
            "settled_step_ms": [1000.0 + rank] * 20,
            "data_wait_ms": [2.0] * 20,
            "mean_step_ms": 1000.0 + rank,
            "mean_data_wait_ms": 2.0,
            "data_wait_fraction": 2.0 / (1000.0 + rank),
            "peak_allocated_mib": 8000.0,
            "peak_reserved_mib": 9000.0,
            "total_device_memory_mib": 15000.0,
            "reserved_headroom_fraction": 0.4,
            "post_settle_graph_break_count": 0,
            "post_settle_recompile_count": 0,
        }
        for rank in range(2)
    ]
    model = build_model(MODEL_KIND_SO2_FIXED)
    named_modules = cast(
        "Iterable[tuple[str, torch.nn.Module]]",
        model.named_modules(),
    )
    gate_rows = [
        {
            "module": f"{name}:{family}",
            "gate_kind": family,
            "gate_health_status": "pass",
            "precision_proof_status": "pass",
            "input_dtype": "float16",
            "output_dtype": "float16",
            "gate_tensor_dtype": "float16",
            "gate_math_dtype": "float32",
            "input_rms": "1",
            "output_rms": "1",
            "a_grad_norm": "1",
            "b_grad_norm": "1",
            "a_update_to_param_norm": "1",
            "b_update_to_param_norm": "1",
        }
        for name, module in named_modules
        if isinstance(module, FixedF01RadialGate)
        for family in ("f0_scalar", "f1_radial")
    ]
    expected_amp = {
        "enabled": True,
        "grad_scaler_enabled": True,
        "grad_scaler_init_scale": 16384.0,
        "autocast_dtype": "float16",
        "requested_autocast_dtype": "float16",
        "local_amp_status": "executed_amp_fp16_conservative",
        "fp32_objective_island": True,
    }
    checkpoint = root / "debug_phase1/checkpoints/step_000004.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"phase one checkpoint")
    for phase_name in ("debug_phase1", "debug_resume", "tiny_overfit"):
        phase = root / phase_name
        benchmark = phase / "benchmark"
        metrics_dir = phase / "metrics"
        benchmark.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        documents: dict[str, dict[str, object]] = {
            "training_summary": {
                "status": "local_pass",
                "ddp_rank_device_proof": {"status": "local_pass"},
                "amp_execution": expected_amp,
                "amp_step_skipped_count": 0,
                "nonfinite_count": 0,
            },
            "selected_runtime_plan_applied": {"status": "local_pass"},
            "checkpoint_resume_proof": {"status": "local_pass"},
            "gate_health_summary": {"status": "local_pass", "rows_written": 68},
            "local_selected_runtime_readiness": {"status": "local_pass"},
            "selected_runtime_debug_summary": {"status": "local_pass"},
        }
        if phase_name == "debug_resume":
            documents["checkpoint_resume_proof"].update(
                {
                    "loaded_successful_optimizer_update_count": 4,
                    "additional_optimizer_steps": 4,
                    "final_optimizer_step": 8,
                    "resume_sequence": "loaded_checkpoint_before_training_continued",
                    "resume_checkpoint_sha256": _file_sha256(checkpoint),
                    "config_sha256_match": True,
                    "runtime_config_sha256_match": True,
                    "selected_row_id_match": True,
                    "runtime_policy_id_match": True,
                    "model_state_restored": True,
                    "optimizer_state_restored": True,
                    "grad_scaler_state_restored": True,
                    "sampler_progress_restored": True,
                },
            )
        if phase_name == "tiny_overfit":
            documents["tiny_overfit_summary"] = {
                "status": "local_pass",
                "settled_real_loader_rank_metrics": metrics,
                "slower_rank_mean_step_ms": 1001.0,
                "projected_epoch_seconds": 6006.0,
            }
        artifact_hashes: dict[str, str] = {}
        for name, payload in documents.items():
            path = benchmark / f"{name}.json"
            path.write_text(
                f"{json.dumps(payload, sort_keys=True)}\n",
                encoding="utf-8",
            )
            artifact_hashes[name] = _file_sha256(path)
        train_steps = metrics_dir / "train_steps.csv"
        train_steps.write_text("status\npass\n", encoding="utf-8")
        artifact_hashes["train_steps"] = _file_sha256(train_steps)
        gate_csv = metrics_dir / "gate_health.csv"
        with gate_csv.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(gate_rows[0]))
            writer.writeheader()
            writer.writerows(gate_rows)
        artifact_hashes["gate_health"] = _file_sha256(gate_csv)
        if phase_name == "debug_phase1":
            artifact_hashes["checkpoint:step_000004.pt"] = _file_sha256(checkpoint)
        manifest: dict[str, object] = {
            "status": "local_pass",
            "missing_artifacts": [],
            "artifact_hashes": artifact_hashes,
        }
        (benchmark / "artifact_manifest.json").write_text(
            f"{json.dumps(manifest, sort_keys=True)}\n",
            encoding="utf-8",
        )
    phase_hashes = {
        name: _file_sha256(root / name / "benchmark/artifact_manifest.json")
        for name in ("debug_phase1", "debug_resume", "tiny_overfit")
    }
    verdict: dict[str, object] = {
        "schema_version": "spec0016.so2_prelaunch_verdict.v1",
        "status": "pass",
        "full_run_eligible": False,
        "source_git_commit": "a" * 40,
        "source_git_dirty": False,
        "checks": dict.fromkeys(
            ("debug", "resume", "tiny", "gates", "performance"),
            True,
        ),
        "selected_runtime": {"per_device_batch_size": 25, "global_batch_size": 50},
        "execution_identity_sha256": execution_identity(repository),
        "phase_artifact_manifest_sha256": phase_hashes,
        "settled_real_loader_rank_metrics": metrics,
        "slower_rank_mean_step_ms": 1001.0,
        "projected_epoch_seconds": 6006.0,
    }
    (root / "benchmark").mkdir(exist_ok=True)
    (root / "benchmark/so2_prelaunch_verdict.json").write_text(
        f"{json.dumps(verdict, sort_keys=True)}\n",
        encoding="utf-8",
    )
    return root


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
