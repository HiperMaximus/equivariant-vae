# Copyright 2026 HiperMaximus
"""Tiny synthetic runtime-benchmark schema writer for spec 0001."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from eqvae.benchmarking.io import CsvRow, JsonObject, write_csv, write_json
from eqvae.benchmarking.model_count import write_model_count

if TYPE_CHECKING:
    from pathlib import Path

DEFAULT_GRADIENT_ACCUMULATION_STEPS: Final = 1
DEFAULT_LOCAL_BATCH_SIZE: Final = 2
LOCAL_SYNTHETIC_SAMPLE_RATE: Final = 128.0
LOCAL_SYNTHETIC_STEP_MS: Final = 15.625

RUNTIME_MATRIX_COLUMNS: Final[tuple[str, ...]] = (
    "run_name",
    "benchmark_kind",
    "benchmark_source",
    "full_run_eligible",
    "row_id",
    "accelerator_mode",
    "machine_shape",
    "visible_device_count",
    "cuda_device_count",
    "gpu_names",
    "ddp_backend",
    "world_size",
    "nproc_per_node",
    "precision_policy",
    "amp_enabled",
    "torch_compile_enabled",
    "compile_scope",
    "runtime_policy_id",
    "memory_format",
    "autocast_dtype",
    "fp32_loss",
    "grad_scaler_enabled",
    "cudnn_benchmark",
    "cudnn_deterministic",
    "deterministic_algorithms",
    "tf32_enabled",
    "matmul_precision",
    "ddp_static_graph",
    "ddp_gradient_as_bucket_view",
    "optimizer_implementation",
    "zero_grad_set_to_none",
    "gradient_clip_foreach",
    "compile_dynamic",
    "corruption_strategy",
    "per_device_batch_size",
    "global_batch_size",
    "gradient_accumulation_steps",
    "warmup_steps",
    "measured_steps",
    "repeats",
    "compile_startup_sec",
    "compile_settle_steps",
    "steady_step_ms_p50",
    "steady_step_ms_p95",
    "samples_sec",
    "trainer_samples_sec",
    "max_vram_allocated_mb",
    "max_vram_reserved_mb",
    "vram_headroom_fraction",
    "amp_step_skipped_count",
    "gate_health_status",
    "gate_health_warning_count",
    "numerical_check_status",
    "data_wait_fraction_p95",
    "graph_break_count",
    "recompile_count",
    "oom",
    "status",
    "failure_kind",
    "failure_message_hash",
)

DATALOADER_MATRIX_COLUMNS: Final[tuple[str, ...]] = (
    "run_name",
    "benchmark_kind",
    "benchmark_source",
    "full_run_eligible",
    "accelerator_mode",
    "machine_shape",
    "world_size",
    "runtime_policy_id",
    "memory_format",
    "rank",
    "split",
    "num_workers",
    "prefetch_factor",
    "pin_memory",
    "persistent_workers",
    "non_blocking_h2d",
    "batch_size",
    "batches_measured",
    "batch_fetch_ms_p50",
    "batch_fetch_ms_p95",
    "h2d_ms_p50",
    "h2d_ms_p95",
    "loader_samples_sec",
    "trainer_samples_sec",
    "data_wait_fraction_p50",
    "data_wait_fraction_p95",
    "rank_sample_count",
    "dropped_sample_count",
    "status",
    "failure_kind",
)

NUMERICAL_CHECK_COLUMNS: Final[tuple[str, ...]] = (
    "run_name",
    "benchmark_kind",
    "benchmark_source",
    "full_run_eligible",
    "accelerator_mode",
    "machine_shape",
    "row_id",
    "reference_row_id",
    "candidate_row_id",
    "runtime_policy_id",
    "batch_index",
    "precision_policy",
    "torch_compile_enabled",
    "compile_scope",
    "corruption_strategy",
    "total_loss_abs_delta",
    "total_loss_rel_delta",
    "recon_loss_abs_delta",
    "recon_loss_rel_delta",
    "l1_loss_abs_delta",
    "l1_loss_rel_delta",
    "ssim_loss_abs_delta",
    "ssim_loss_rel_delta",
    "kl_loss_abs_delta",
    "kl_loss_rel_delta",
    "grad_norm_abs_delta",
    "grad_norm_rel_delta",
    "param_update_norm_abs_delta",
    "param_update_norm_rel_delta",
    "x_hat_min_abs_delta",
    "x_hat_max_abs_delta",
    "mu_mean_abs_delta",
    "mu_std_abs_delta",
    "logvar_mean_abs_delta",
    "logvar_std_abs_delta",
    "logvar_clamp_count_delta",
    "gate_health_status",
    "nonfinite_count",
    "amp_step_skipped",
    "status",
    "failure_kind",
)

CORRUPTION_CHECK_COLUMNS: Final[tuple[str, ...]] = (
    "run_name",
    "benchmark_kind",
    "benchmark_source",
    "full_run_eligible",
    "accelerator_mode",
    "machine_shape",
    "row_id",
    "reference_row_id",
    "candidate_row_id",
    "runtime_policy_id",
    "batch_index",
    "corruption_version",
    "profile_name",
    "corruption_strategy",
    "corruption_view",
    "corruption_step",
    "split",
    "semantic_sample_key_hash",
    "binary_sample_id_hash",
    "rank",
    "world_size",
    "applied_mask_hash",
    "stain_param_hash",
    "noise_std_hash",
    "noise_field_hash",
    "clean_sample_unchanged_count",
    "clean_validation_rng_advanced",
    "status",
    "failure_kind",
)

GATE_HEALTH_COLUMNS: Final[tuple[str, ...]] = (
    "run_name",
    "benchmark_kind",
    "benchmark_source",
    "full_run_eligible",
    "accelerator_mode",
    "machine_shape",
    "row_id",
    "candidate_row_id",
    "runtime_policy_id",
    "optimizer_step",
    "module",
    "gate_kind",
    "num_channels",
    "num_elements",
    "a_min",
    "a_max",
    "a_mean",
    "a_std",
    "b_min",
    "b_max",
    "b_mean",
    "b_std",
    "max_abs_a",
    "max_abs_b",
    "gate_mean",
    "gate_std",
    "gate_p01",
    "gate_p50",
    "gate_p99",
    "frac_gate_lt_0_01",
    "frac_gate_gt_0_99",
    "worst_channel_frac_gate_lt_0_01",
    "worst_channel_frac_gate_gt_0_99",
    "dead_channel_count",
    "input_rms",
    "output_rms",
    "output_input_rms_ratio",
    "a_grad_norm",
    "b_grad_norm",
    "a_update_to_param_norm",
    "b_update_to_param_norm",
    "gate_health_status",
)


@dataclass(frozen=True)
class BenchmarkArtifactPaths:
    """Paths written by the local synthetic benchmark schema smoke."""

    model_count: Path
    runtime_proof: Path
    runtime_matrix: Path
    selected_runtime: Path
    dataloader_matrix: Path
    numerical_checks: Path
    corruption_checks: Path
    gate_health: Path
    gate_health_summary: Path


@dataclass(frozen=True)
class SyntheticBenchmarkRequest:
    """Inputs for the local synthetic benchmark schema smoke."""

    config_path: Path
    output_dir: Path
    run_name: str
    max_benchmark_rows: int
    warmup_steps: int
    measured_steps: int


def write_synthetic_benchmark_artifacts(
    request: SyntheticBenchmarkRequest,
) -> BenchmarkArtifactPaths:
    """Write tiny local benchmark artifacts with spec-compatible schemas.

    Returns:
        Paths for every artifact written by the schema smoke.

    """
    _require_positive("max_benchmark_rows", request.max_benchmark_rows)
    _require_positive("warmup_steps", request.warmup_steps)
    _require_positive("measured_steps", request.measured_steps)

    benchmark_dir = request.output_dir / "benchmark"
    metrics_dir = request.output_dir / "metrics"
    model_count_path = benchmark_dir / "model_count.json"
    runtime_proof_path = benchmark_dir / "runtime_proof.json"
    runtime_matrix_path = benchmark_dir / "runtime_matrix.csv"
    selected_runtime_path = benchmark_dir / "selected_runtime.json"
    dataloader_matrix_path = benchmark_dir / "dataloader_matrix.csv"
    numerical_checks_path = benchmark_dir / "numerical_checks.csv"
    corruption_checks_path = benchmark_dir / "corruption_checks.csv"
    gate_health_path = metrics_dir / "gate_health.csv"
    gate_health_summary_path = benchmark_dir / "gate_health_summary.json"

    runtime_rows = _runtime_rows(
        run_name=request.run_name,
        max_benchmark_rows=request.max_benchmark_rows,
        warmup_steps=request.warmup_steps,
        measured_steps=request.measured_steps,
    )
    write_model_count(config_path=request.config_path, output_path=model_count_path)
    write_json(runtime_proof_path, _runtime_proof_payload(run_name=request.run_name))
    write_csv(runtime_matrix_path, RUNTIME_MATRIX_COLUMNS, runtime_rows)
    write_json(selected_runtime_path, _selected_runtime_payload(runtime_rows[0]))
    write_csv(
        dataloader_matrix_path,
        DATALOADER_MATRIX_COLUMNS,
        _dataloader_rows(run_name=request.run_name),
    )
    write_csv(
        numerical_checks_path,
        NUMERICAL_CHECK_COLUMNS,
        _numerical_check_rows(run_name=request.run_name, runtime_rows=runtime_rows),
    )
    write_csv(
        corruption_checks_path,
        CORRUPTION_CHECK_COLUMNS,
        _corruption_check_rows(run_name=request.run_name, runtime_rows=runtime_rows),
    )
    write_csv(
        gate_health_path,
        GATE_HEALTH_COLUMNS,
        _gate_health_rows(request.run_name),
    )
    write_json(gate_health_summary_path, _gate_health_summary_payload())

    return BenchmarkArtifactPaths(
        model_count=model_count_path,
        runtime_proof=runtime_proof_path,
        runtime_matrix=runtime_matrix_path,
        selected_runtime=selected_runtime_path,
        dataloader_matrix=dataloader_matrix_path,
        numerical_checks=numerical_checks_path,
        corruption_checks=corruption_checks_path,
        gate_health=gate_health_path,
        gate_health_summary=gate_health_summary_path,
    )


def _runtime_rows(
    *,
    run_name: str,
    max_benchmark_rows: int,
    warmup_steps: int,
    measured_steps: int,
) -> list[CsvRow]:
    candidates = (
        ("amp_off_fp32", "false", "false", "none", "branchless_all"),
        ("amp_off_fp32", "false", "true", "model_forward", "branchless_all"),
        ("amp_off_fp32", "false", "false", "none", "indexed_masked"),
    )
    rows: list[CsvRow] = []
    for row_index, candidate in enumerate(candidates[:max_benchmark_rows]):
        (
            precision_policy,
            amp_enabled,
            compile_enabled,
            compile_scope,
            corruption_strategy,
        ) = candidate
        rows.append(
            {
                "run_name": run_name,
                "benchmark_kind": "local_synthetic_schema",
                "benchmark_source": "local_synthetic_schema_smoke",
                "full_run_eligible": "false",
                "row_id": f"local_cpu_schema_{row_index:03d}",
                "accelerator_mode": "local_cpu",
                "machine_shape": "local_cpu",
                "visible_device_count": "0",
                "cuda_device_count": "0",
                "gpu_names": "[]",
                "ddp_backend": "none",
                "world_size": "1",
                "nproc_per_node": "1",
                "precision_policy": precision_policy,
                "amp_enabled": amp_enabled,
                "torch_compile_enabled": compile_enabled,
                "compile_scope": compile_scope,
                "runtime_policy_id": "local_schema_policy",
                "memory_format": "contiguous",
                "autocast_dtype": "",
                "fp32_loss": "true",
                "grad_scaler_enabled": "false",
                "cudnn_benchmark": "false",
                "cudnn_deterministic": "false",
                "deterministic_algorithms": "false",
                "tf32_enabled": "false",
                "matmul_precision": "highest",
                "ddp_static_graph": "false",
                "ddp_gradient_as_bucket_view": "false",
                "optimizer_implementation": "adamw_default",
                "zero_grad_set_to_none": "true",
                "gradient_clip_foreach": "false",
                "compile_dynamic": "false",
                "corruption_strategy": corruption_strategy,
                "per_device_batch_size": str(DEFAULT_LOCAL_BATCH_SIZE),
                "global_batch_size": str(DEFAULT_LOCAL_BATCH_SIZE),
                "gradient_accumulation_steps": str(
                    DEFAULT_GRADIENT_ACCUMULATION_STEPS,
                ),
                "warmup_steps": str(warmup_steps),
                "measured_steps": str(measured_steps),
                "repeats": "1",
                "compile_startup_sec": "0.0",
                "compile_settle_steps": "0",
                "steady_step_ms_p50": str(LOCAL_SYNTHETIC_STEP_MS),
                "steady_step_ms_p95": str(LOCAL_SYNTHETIC_STEP_MS),
                "samples_sec": str(LOCAL_SYNTHETIC_SAMPLE_RATE),
                "trainer_samples_sec": str(LOCAL_SYNTHETIC_SAMPLE_RATE),
                "max_vram_allocated_mb": "0.0",
                "max_vram_reserved_mb": "0.0",
                "vram_headroom_fraction": "1.0",
                "amp_step_skipped_count": "0",
                "gate_health_status": "schema_pass",
                "gate_health_warning_count": "0",
                "numerical_check_status": "schema_pass",
                "data_wait_fraction_p95": "0.0",
                "graph_break_count": "0",
                "recompile_count": "0",
                "oom": "false",
                "status": "schema_pass",
                "failure_kind": "",
                "failure_message_hash": "",
            },
        )
    return rows


def _selected_runtime_payload(selected_row: CsvRow) -> JsonObject:
    return {
        "status": "schema_pass",
        "benchmark_kind": "local_synthetic_schema",
        "benchmark_source": "local_synthetic_schema_smoke",
        "full_run_eligible": False,
        "notes": (
            "Local CPU synthetic schema smoke only; run the permission-gated "
            "Kaggle runtime benchmark before selecting a full-run runtime."
        ),
        "selected_row_id": selected_row["row_id"],
        "accelerator_mode": selected_row["accelerator_mode"],
        "machine_shape": selected_row["machine_shape"],
        "world_size": 1,
        "nproc_per_node": 1,
        "gpu_names": [],
        "per_device_batch_size": DEFAULT_LOCAL_BATCH_SIZE,
        "global_batch_size": DEFAULT_LOCAL_BATCH_SIZE,
        "gradient_accumulation_steps": DEFAULT_GRADIENT_ACCUMULATION_STEPS,
        "optimizer_updates_per_epoch": 0,
        "lr_warmup_steps": 0,
        "beta_warmup_steps": 0,
        "mixed_precision": {"enabled": False, "policy": "amp_off_fp32"},
        "torch_compile": {
            "enabled": selected_row["torch_compile_enabled"] == "true",
            "backend": "eager",
            "scope": selected_row["compile_scope"],
        },
        "corruption": {"strategy": selected_row["corruption_strategy"]},
        "dataloader": {
            "num_workers": 0,
            "prefetch_factor": None,
            "pin_memory": False,
            "persistent_workers": False,
            "non_blocking_h2d": False,
        },
        "throughput": {
            "samples_sec": LOCAL_SYNTHETIC_SAMPLE_RATE,
            "steady_step_ms_p50": LOCAL_SYNTHETIC_STEP_MS,
            "compile_startup_sec": 0.0,
            "compile_settle_steps": 0,
            "estimated_10_epoch_wall_time_sec": 0.0,
        },
        "safety": {
            "numerical_check_status": "schema_pass",
            "corruption_check_status": "schema_pass",
            "gate_health_status": "schema_pass",
            "dataloader_status": "schema_pass",
            "amp_step_skipped_count": 0,
        },
        "artifacts": {
            "runtime_matrix": "benchmark/runtime_matrix.csv",
            "model_count": "benchmark/model_count.json",
            "runtime_proof": "benchmark/runtime_proof.json",
            "dataloader_matrix": "benchmark/dataloader_matrix.csv",
            "numerical_checks": "benchmark/numerical_checks.csv",
            "corruption_checks": "benchmark/corruption_checks.csv",
            "gate_health_summary": "benchmark/gate_health_summary.json",
        },
        "selected_row_snapshot": {
            **dict(selected_row),
            "compile_settle_protocol_sha256": "",
            "post_settle_graph_break_count": 0,
            "post_settle_recompile_count": 0,
        },
    }


def _runtime_proof_payload(*, run_name: str) -> JsonObject:
    return {
        "status": "schema_pass",
        "benchmark_kind": "local_synthetic_schema",
        "benchmark_source": "local_synthetic_schema_smoke",
        "full_run_eligible": False,
        "run_name": run_name,
        "accelerator_mode": "local_cpu",
        "machine_shape": "local_cpu",
        "visible_device_count": 0,
        "cuda_device_count": 0,
        "gpu_names": [],
        "ddp_backend": "none",
        "world_size": 1,
        "nproc_per_node": 1,
        "dataset_slug": "",
        "launcher_command_sha256": "",
        "kaggle_cli_version": "",
        "compile_settle_policy": {
            "compile_settle_steps": 0,
            "exercised_paths": [],
            "counter_source": "not_applicable_local_schema_smoke",
        },
        "notes": "Local CPU schema proof only; not Kaggle runtime evidence.",
    }


def _dataloader_rows(*, run_name: str) -> list[CsvRow]:
    return [
        _dataloader_row(run_name=run_name, split="train"),
        _dataloader_row(run_name=run_name, split="validation"),
    ]


def _dataloader_row(*, run_name: str, split: str) -> CsvRow:
    return {
        "run_name": run_name,
        "benchmark_kind": "local_synthetic_schema",
        "benchmark_source": "local_synthetic_schema_smoke",
        "full_run_eligible": "false",
        "accelerator_mode": "local_cpu",
        "machine_shape": "local_cpu",
        "world_size": "1",
        "rank": "0",
        "split": split,
        "num_workers": "0",
        "prefetch_factor": "",
        "pin_memory": "false",
        "persistent_workers": "false",
        "non_blocking_h2d": "false",
        "batch_size": str(DEFAULT_LOCAL_BATCH_SIZE),
        "batches_measured": "2",
        "batch_fetch_ms_p50": "0.1",
        "batch_fetch_ms_p95": "0.1",
        "h2d_ms_p50": "",
        "h2d_ms_p95": "",
        "loader_samples_sec": "256.0",
        "trainer_samples_sec": str(LOCAL_SYNTHETIC_SAMPLE_RATE),
        "data_wait_fraction_p50": "0.0",
        "data_wait_fraction_p95": "0.0",
        "rank_sample_count": "4",
        "dropped_sample_count": "0",
        "status": "schema_pass",
        "failure_kind": "",
    }


def _numerical_check_rows(*, run_name: str, runtime_rows: list[CsvRow]) -> list[CsvRow]:
    return [
        {
            "run_name": run_name,
            "benchmark_kind": "local_synthetic_schema",
            "benchmark_source": "local_synthetic_schema_smoke",
            "full_run_eligible": "false",
            "accelerator_mode": "local_cpu",
            "machine_shape": "local_cpu",
            "row_id": row["row_id"],
            "reference_row_id": "local_cpu_schema_reference",
            "candidate_row_id": row["row_id"],
            "batch_index": "0",
            "precision_policy": row["precision_policy"],
            "torch_compile_enabled": row["torch_compile_enabled"],
            "compile_scope": row["compile_scope"],
            "corruption_strategy": row["corruption_strategy"],
            "total_loss_abs_delta": "0.0",
            "total_loss_rel_delta": "0.0",
            "recon_loss_abs_delta": "0.0",
            "recon_loss_rel_delta": "0.0",
            "l1_loss_abs_delta": "0.0",
            "l1_loss_rel_delta": "0.0",
            "ssim_loss_abs_delta": "0.0",
            "ssim_loss_rel_delta": "0.0",
            "kl_loss_abs_delta": "0.0",
            "kl_loss_rel_delta": "0.0",
            "grad_norm_abs_delta": "0.0",
            "grad_norm_rel_delta": "0.0",
            "param_update_norm_abs_delta": "0.0",
            "param_update_norm_rel_delta": "0.0",
            "x_hat_min_abs_delta": "0.0",
            "x_hat_max_abs_delta": "0.0",
            "mu_mean_abs_delta": "0.0",
            "mu_std_abs_delta": "0.0",
            "logvar_mean_abs_delta": "0.0",
            "logvar_std_abs_delta": "0.0",
            "logvar_clamp_count_delta": "0",
            "gate_health_status": "schema_pass",
            "nonfinite_count": "0",
            "amp_step_skipped": "0",
            "status": "schema_pass",
            "failure_kind": "",
        }
        for row in runtime_rows
    ]


def _corruption_check_rows(
    *,
    run_name: str,
    runtime_rows: list[CsvRow],
) -> list[CsvRow]:
    return [
        {
            "run_name": run_name,
            "benchmark_kind": "local_synthetic_schema",
            "benchmark_source": "local_synthetic_schema_smoke",
            "full_run_eligible": "false",
            "accelerator_mode": "local_cpu",
            "machine_shape": "local_cpu",
            "row_id": row["row_id"],
            "reference_row_id": "local_cpu_schema_reference",
            "candidate_row_id": row["row_id"],
            "batch_index": "0",
            "corruption_version": "spec0001.hed_corruptor.v1",
            "profile_name": "conservative_default",
            "corruption_strategy": row["corruption_strategy"],
            "corruption_view": "schema_smoke",
            "corruption_step": "0",
            "split": "train",
            "semantic_sample_key_hash": "schema",
            "binary_sample_id_hash": "schema",
            "rank": "0",
            "world_size": "1",
            "applied_mask_hash": "schema",
            "stain_param_hash": "schema",
            "noise_std_hash": "schema",
            "noise_field_hash": "schema",
            "clean_sample_unchanged_count": "0",
            "clean_validation_rng_advanced": "false",
            "status": "schema_pass",
            "failure_kind": "",
        }
        for row in runtime_rows
    ]


def _gate_health_rows(run_name: str) -> list[CsvRow]:
    return [
        {
            "run_name": run_name,
            "benchmark_kind": "local_synthetic_schema",
            "benchmark_source": "local_synthetic_schema_smoke",
            "full_run_eligible": "false",
            "accelerator_mode": "local_cpu",
            "machine_shape": "local_cpu",
            "row_id": "schema_gate_health__schema_runtime_row_0",
            "candidate_row_id": "schema_runtime_row_0",
            "optimizer_step": "0",
            "module": "schema_smoke/gated_scalar",
            "gate_kind": "scalar",
            "num_channels": "1",
            "num_elements": "1",
            "a_min": "1.0",
            "a_max": "1.0",
            "a_mean": "1.0",
            "a_std": "0.0",
            "b_min": "0.0",
            "b_max": "0.0",
            "b_mean": "0.0",
            "b_std": "0.0",
            "max_abs_a": "1.0",
            "max_abs_b": "0.0",
            "gate_mean": "0.5",
            "gate_std": "0.0",
            "gate_p01": "0.5",
            "gate_p50": "0.5",
            "gate_p99": "0.5",
            "frac_gate_lt_0_01": "0.0",
            "frac_gate_gt_0_99": "0.0",
            "worst_channel_frac_gate_lt_0_01": "0.0",
            "worst_channel_frac_gate_gt_0_99": "0.0",
            "dead_channel_count": "0",
            "input_rms": "1.0",
            "output_rms": "0.5",
            "output_input_rms_ratio": "0.5",
            "a_grad_norm": "0.0",
            "b_grad_norm": "0.0",
            "a_update_to_param_norm": "0.0",
            "b_update_to_param_norm": "0.0",
            "gate_health_status": "schema_pass",
        },
    ]


def _gate_health_summary_payload() -> JsonObject:
    return {
        "status": "schema_pass",
        "benchmark_kind": "local_synthetic_schema",
        "benchmark_source": "local_synthetic_schema_smoke",
        "overall_status": "schema_pass",
        "full_run_eligible": False,
        "logged_intervals": 1,
        "module_count": 1,
        "nonfinite_count": 0,
        "max_abs_a": 1.0,
        "max_abs_b": 0.0,
        "worst_frac_gate_lt_0_01": 0.0,
        "worst_frac_gate_gt_0_99": 0.0,
        "dead_channel_count": 0,
        "zero_gradient_interval_count": 0,
        "worst_output_input_rms_ratio": 0.5,
        "failing_modules": [],
        "warning_modules": [],
        "notes": "Local synthetic schema smoke only.",
    }


def _require_positive(name: str, value: int) -> None:
    if value > 0:
        return
    message = f"{name} must be positive"
    raise ValueError(message)
