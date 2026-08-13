# Copyright 2026 HiperMaximus
# ruff: noqa: C901, EM101, PERF401, PLR0912, PLR0914, PLR0915, PLR0916, PLR2004, RUF069, TRY003
"""Generated launcher for the fixed Spec 0015 dual-T4 readiness proof."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import math
import os
import shutil
import statistics
import subprocess  # noqa: S404
import sys
import zipfile
from pathlib import Path
from typing import cast

KAGGLE_SO2_RUNTIME_READINESS_READY = True
PROBE_MODULE = "eqvae.benchmarking.so2_runtime_readiness"
CONFIG_PATH = Path("configs/spec0015/so2_vae_selected_runtime_readiness.json")
ARTIFACT_FILENAME = "spec0015_so2_runtime_readiness.json"
GATE_FILENAME = "spec0015_so2_gate_health.csv"
SCHEMA_VERSION = "spec0015.so2_selected_runtime_readiness.v1"
PROBE_KIND = "fixed_so2_selected_runtime_readiness"
RUNTIME_POLICY_ID = "compile_step_python_reducer_fp16_channels_last"
REQUIRED_GPU_NAME = "Tesla T4"
GATE_ROW_COUNT = 68
RADIAL_GATE_COUNT = 34
SETTLED_UPDATES = 3
EXPECTED_GATE_MODULES = {
    "stem_gate",
    "latent_projection_gate",
    *(
        f"encoder_blocks.{index}.{kind}_gate"
        for index in range(8)
        for kind in ("main", "output")
    ),
    *(
        f"decoder_blocks.{index}.{kind}_gate"
        for index in range(8)
        for kind in ("main", "output")
    ),
}
PROBE_TIMEOUT_SECONDS = 10800
DEFAULT_OUTPUT_DIR = Path("/kaggle/working")
EMBEDDED_PAYLOAD_B64 = """
$embedded_payload_b64
"""
EMBEDDED_PAYLOAD_ZIP_SHA256 = "$embedded_payload_zip_sha256"
EMBEDDED_PAYLOAD_MANIFEST_SHA256 = "$embedded_payload_manifest_sha256"


def main() -> int:
    """Launch and validate the single private no-dataset readiness coordinate.

    Returns:
        Process exit status.

    """
    _require_python_version()
    _ensure_latest_torch()
    output_dir = _output_dir()
    payload_dir = _extract_payload(_payload_extract_dir())
    _launch(payload_dir=payload_dir, output_dir=output_dir)
    _validate_artifacts(output_dir=output_dir, payload_dir=payload_dir)
    return 0


def _launch(*, payload_dir: Path, output_dir: Path) -> None:
    benchmark_dir = output_dir / "benchmark"
    command = (
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        "-m",
        PROBE_MODULE,
        "--config",
        str(CONFIG_PATH),
        "--output-dir",
        str(benchmark_dir),
    )
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(payload_dir / "src"), *([existing] if existing else [])],
    )
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    environment["TORCH_LOGS"] = "graph_breaks,recompiles"
    subprocess.run(  # noqa: S603
        command,
        check=True,
        cwd=payload_dir,
        env=environment,
        timeout=PROBE_TIMEOUT_SECONDS,
    )


def _output_dir() -> Path:
    configured = os.environ.get("EQVAE_OUTPUT_DIR")
    output_dir = Path(configured) if configured else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir.resolve()


def _payload_extract_dir() -> Path:
    if Path("/kaggle/temp").exists():
        return Path("/kaggle/temp/eqvae_so2_runtime_readiness_payload")
    return Path("/tmp/eqvae_so2_runtime_readiness_payload")  # noqa: S108


def _require_python_version() -> None:
    if sys.version_info < (3, 12):  # noqa: UP036
        raise RuntimeError("Spec 0015 readiness requires Python >= 3.12")


def _ensure_latest_torch() -> None:
    if os.environ.get("EQVAE_SKIP_TORCH_UPGRADE") == "1":
        return
    if not Path("/kaggle/working").exists():
        return
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "torch",
            "torchvision",
            "torchaudio",
        ],
    )


def _extract_payload(destination: Path) -> Path:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    zip_bytes = base64.b64decode(EMBEDDED_PAYLOAD_B64.encode("ascii"))
    if hashlib.sha256(zip_bytes).hexdigest() != EMBEDDED_PAYLOAD_ZIP_SHA256:
        raise RuntimeError("embedded payload zip SHA-256 mismatch")
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        for member in archive.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                message = f"unsafe embedded payload path: {member.filename}"
                raise RuntimeError(message)
        archive.extractall(destination)
    manifest = destination / "payload_manifest.json"
    if (
        hashlib.sha256(manifest.read_bytes()).hexdigest()
        != EMBEDDED_PAYLOAD_MANIFEST_SHA256
    ):
        raise RuntimeError("embedded payload manifest SHA-256 mismatch")
    return destination


def _finite_number(value: object, *, positive: bool = False) -> bool:
    if isinstance(value, bool):
        return False
    try:
        parsed = float(cast("str | int | float", value))
    except (TypeError, ValueError):
        return False
    return math.isfinite(parsed) and (not positive or parsed > 0.0)


def _proof_body_errors(payload: dict[str, object]) -> list[str]:
    errors: list[str] = []
    master = payload.get("master_dtype_proof")
    if master != {
        "status": "pass",
        "parameter_dtypes": ["torch.float32"],
        "buffer_dtypes": ["torch.float32"],
        "field_norm_count": 40,
        "radial_gate_count": RADIAL_GATE_COUNT,
        "norm_and_radial_math_dtype": "float32",
    }:
        errors.append("master dtype proof body drifted")
    buffer_sync = payload.get("pre_compile_buffer_sync")
    if not isinstance(buffer_sync, dict) or any(
        (
            buffer_sync.get("status") != "pass",
            buffer_sync.get("checked_before_ddp_and_compile") is not True,
            buffer_sync.get("buffer_count") != 54,
            buffer_sync.get("max_abs_difference") != 0.0,
        ),
    ):
        errors.append("pre-compile buffer-sync proof body must be exact")
    ddp = payload.get("ddp_runtime_readback")
    expected_ddp = {
        "python_reducer": True,
        "static_graph": False,
        "gradient_as_bucket_view": True,
        "broadcast_buffers": False,
        "find_unused_parameters": False,
        "bucket_bytes_cap": 50 * 1024 * 1024,
        "optimize_ddp": "python_reducer",
        "compiled_autograd": True,
        "reorder_compute_comm_overlap": True,
        "optimizer_fused": True,
    }
    if (
        not isinstance(ddp, dict)
        or ddp.get("status") != "pass"
        or ddp.get("requested") != expected_ddp
        or ddp.get("effective") != expected_ddp
    ):
        errors.append("DDP/compiler/optimizer readback body must be exact")
    optimizer = payload.get("optimizer_policy")
    if not isinstance(optimizer, dict):
        errors.append("optimizer policy proof body is missing")
    else:
        base_lr = optimizer.get("base_learning_rate")
        gate_lr = optimizer.get("gate_learning_rate")
        if (
            optimizer.get("status") != "pass"
            or optimizer.get("all_parameters_covered_once") is not True
            or optimizer.get("coefficient_parameter_count") != 1_172_304
            or optimizer.get("gate_parameter_count") != 4_096
            or optimizer.get("coefficient_weight_decay") != 1e-5
            or optimizer.get("gate_weight_decay") != 0.0
            or optimizer.get("fused_requested") is not True
            or not _finite_number(base_lr, positive=True)
            or not _finite_number(gate_lr, positive=True)
            or not math.isclose(
                float(cast("int | float", gate_lr)),
                0.5 * float(cast("int | float", base_lr)),
                rel_tol=1e-12,
            )
        ):
            errors.append("optimizer policy proof body must be exact")
    gradient = payload.get("gradient_mean_reference")
    if (
        not isinstance(gradient, dict)
        or gradient.get("status") != "pass"
        or gradient.get("parameter") != "output_head.bias"
        or gradient.get("local_pre_reduction_gradients_differ") is not True
        or not _finite_number(gradient.get("reduced_gradient_max_abs_error"))
        or float(cast("int | float", gradient["reduced_gradient_max_abs_error"])) > 1e-6
    ):
        errors.append("DDP gradient-mean proof body must be valid")
    parameter_sync = payload.get("parameter_sync")
    if (
        not isinstance(parameter_sync, dict)
        or parameter_sync.get("status") != "pass"
        or parameter_sync.get("max_abs_difference") != 0.0
    ):
        errors.append("parameter-sync proof body must be exact")
    return errors


def _rank_metric_errors(
    payload: dict[str, object],
    compiled: dict[str, object],
) -> list[str]:
    raw_metrics = payload.get("rank_metrics")
    if not isinstance(raw_metrics, list) or len(raw_metrics) != 2:
        return ["rank_metrics must contain exactly two rank records"]
    metrics = [item for item in raw_metrics if isinstance(item, dict)]
    if len(metrics) != 2 or {item.get("rank") for item in metrics} != {0, 1}:
        return ["rank_metrics must identify ranks 0 and 1 exactly"]
    errors: list[str] = []
    for item in metrics:
        if (
            item.get("amp_step_skipped_count") != 0
            or item.get("post_settle_graph_break_count") != 0
            or item.get("post_settle_recompile_count") != 0
            or item.get("finite_losses") is not True
            or item.get("finite_parameters") is not True
            or not _finite_number(item.get("compile_startup_seconds"))
            or not _finite_number(item.get("peak_allocated_mib"), positive=True)
            or not _finite_number(item.get("peak_reserved_mib"), positive=True)
            or not _finite_number(item.get("total_device_memory_mib"), positive=True)
            or not _finite_number(item.get("reserved_headroom_fraction"), positive=True)
        ):
            errors.append("each rank metric must contain finite passing evidence")
        timings = item.get("settled_step_ms")
        if (
            not isinstance(timings, list)
            or len(timings) != SETTLED_UPDATES
            or any(not _finite_number(value, positive=True) for value in timings)
        ):
            errors.append("each rank must contain three finite settled timings")
    if errors:
        return errors
    expected_aggregates = {
        "amp_step_skipped_count": 0,
        "post_settle_graph_break_count": 0,
        "post_settle_recompile_count": 0,
        "finite_losses": True,
        "finite_parameters": True,
        "peak_allocated_mib_rank_max": max(
            float(cast("int | float", item["peak_allocated_mib"])) for item in metrics
        ),
        "peak_reserved_mib_rank_max": max(
            float(cast("int | float", item["peak_reserved_mib"])) for item in metrics
        ),
        "reserved_vram_headroom_fraction_rank_min": min(
            float(cast("int | float", item["reserved_headroom_fraction"]))
            for item in metrics
        ),
        "compile_startup_seconds_rank_max": max(
            float(cast("int | float", item["compile_startup_seconds"]))
            for item in metrics
        ),
        "diagnostic_settled_step_ms_p50": statistics.median(
            float(cast("int | float", timing))
            for item in metrics
            for timing in cast("list[object]", item["settled_step_ms"])
        ),
        "diagnostic_settled_step_ms_rank_samples": [
            item["settled_step_ms"] for item in metrics
        ],
    }
    if any(compiled.get(key) != value for key, value in expected_aggregates.items()):
        errors.append("compiled aggregates must exactly match both rank records")
    return errors


def _validate_artifacts(*, output_dir: Path, payload_dir: Path) -> None:
    benchmark_dir = output_dir / "benchmark"
    observed = {path.name for path in benchmark_dir.iterdir()}
    expected_files = {ARTIFACT_FILENAME, GATE_FILENAME}
    if observed != expected_files:
        message = f"unexpected Spec 0015 artifacts: {sorted(observed)}"
        raise RuntimeError(message)
    payload = cast(
        "dict[str, object]",
        json.loads((benchmark_dir / ARTIFACT_FILENAME).read_text(encoding="utf-8")),
    )
    errors: list[str] = []
    expected = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_kind": PROBE_KIND,
        "status": "pass",
        "full_run_eligible": False,
        "full_training_authorized": False,
        "model_kind": "so2_vae_fixed",
        "world_size": 2,
        "per_device_batch_size": 1,
        "data_source": "generated_device_resident",
        "dataset_sources": [],
        "acceptance_failures": [],
    }
    errors.extend(
        f"{key} must be {value!r}"
        for key, value in expected.items()
        if payload.get(key) != value
    )
    if payload.get("model_identity") != {
        "concrete_class": "SO2VAE",
        "learned_parameter_count": 1_180_035,
        "latent_channels": 16,
        "learned_convolution_count": 43,
        "radial_gate_count": 34,
    }:
        errors.append("model identity/count proof must remain exact")
    assignments = payload.get("rank_device_assignments")
    if not isinstance(assignments, list) or {
        (item.get("rank"), item.get("local_rank"), item.get("current_device"))
        for item in assignments
        if isinstance(item, dict)
    } != {(0, 0, 0), (1, 1, 1)}:
        errors.append("rank-device assignments must be the two-device bijection")
    elif [item.get("device_name") for item in assignments] != [REQUIRED_GPU_NAME] * 2:
        errors.append("both assigned devices must be Tesla T4")
    runtime = payload.get("runtime_requested_and_effective")
    expected_runtime = {
        "runtime_policy_id": RUNTIME_POLICY_ID,
        "memory_format": "channels_last",
        "autocast_dtype": "float16",
        "fp32_loss": True,
        "grad_scaler_enabled": True,
        "compile_scope": "step",
        "compile_backend": "inductor",
        "compile_dynamic": False,
        "optimize_ddp": "python_reducer",
        "compiled_autograd": True,
        "reorder_compute_comm_overlap": True,
        "fused_optimizer": True,
        "gradient_clip_foreach": True,
        "zero_grad_set_to_none": True,
        "cudnn_benchmark": True,
        "cudnn_deterministic": False,
        "tf32_matmul": True,
        "tf32_cudnn": True,
        "matmul_precision": "high",
    }
    if not isinstance(runtime, dict) or any(
        runtime.get(key) != value for key, value in expected_runtime.items()
    ):
        errors.append("runtime must be the locked selected bundle")
    compiled = payload.get("compiled_execution")
    if not isinstance(compiled, dict):
        errors.append("compiled_execution must be an object")
    else:
        for key in (
            "amp_step_skipped_count",
            "post_settle_graph_break_count",
            "post_settle_recompile_count",
        ):
            if compiled.get(key) != 0:
                errors.append(f"compiled_execution.{key} must be zero")
        for key in ("finite_losses", "finite_parameters"):
            if compiled.get(key) is not True:
                errors.append(f"compiled_execution.{key} must be true")
        for key in (
            "compile_startup_seconds_rank_max",
            "diagnostic_settled_step_ms_p50",
            "peak_allocated_mib_rank_max",
            "peak_reserved_mib_rank_max",
            "reserved_vram_headroom_fraction_rank_min",
        ):
            value = compiled.get(key)
            if not _finite_number(value) or float(cast("int | float", value)) < 0:
                errors.append(f"compiled_execution.{key} must be nonnegative")
    for key in (
        "master_dtype_proof",
        "pre_compile_buffer_sync",
        "ddp_runtime_readback",
        "optimizer_policy",
        "gradient_mean_reference",
        "parameter_sync",
    ):
        proof = payload.get(key)
        if not isinstance(proof, dict) or proof.get("status") != "pass":
            errors.append(f"{key} must pass")
    errors.extend(_proof_body_errors(payload))
    if isinstance(compiled, dict):
        errors.extend(_rank_metric_errors(payload, compiled))
    update_sequence = payload.get("update_sequence")
    if not isinstance(update_sequence, dict):
        errors.append("update_sequence must be an object")
    else:
        first_update = cast(
            "dict[str, object]",
            update_sequence.get("first_zero_head_update", {}),
        )
        if set(first_update) != {"output_head"}:
            errors.append("first update must prove the zero output head")
        elif not _valid_update_proof(first_update.get("output_head")):
            errors.append("output-head update proof must be finite and pass")
        upstream = update_sequence.get("first_upstream_gradient_norms")
        required = {"decoder", "posterior", "encoder", "stem", "f0_gate", "f1_gate"}
        if (
            not isinstance(upstream, dict)
            or set(upstream) != required
            or any(
                not _finite_number(value) or bool(float(cast("int | float", value)))
                for value in upstream.values()
            )
        ):
            errors.append("first-step upstream gradients must be exact finite zeros")
        subsequent = update_sequence.get("subsequent_named_updates")
        if not isinstance(subsequent, dict) or set(subsequent) != required:
            errors.append("subsequent update proof is incomplete")
        elif any(not _valid_update_proof(value) for value in subsequent.values()):
            errors.append("every subsequent named update must be finite and pass")
    with (benchmark_dir / GATE_FILENAME).open(encoding="utf-8", newline="") as file_obj:
        rows = list(csv.DictReader(file_obj))
    if len(rows) != GATE_ROW_COUNT:
        errors.append("gate CSV must contain exactly 68 rows")
    gate_health = payload.get("gate_health")
    if gate_health != {
        "expected_rows": GATE_ROW_COUNT,
        "rows_written": GATE_ROW_COUNT,
        "families": ["f0_scalar", "f1_radial"],
    }:
        errors.append("gate-health summary must describe the exact 68-row evidence")
    if {row.get("gate_kind") for row in rows} != {"f0_scalar", "f1_radial"}:
        errors.append("gate CSV must cover F0 and F1 separately")
    expected_identities = {
        (f"{module}:{family}", family)
        for module in EXPECTED_GATE_MODULES
        for family in ("f0_scalar", "f1_radial")
    }
    identities = {(row.get("module"), row.get("gate_kind")) for row in rows}
    if (
        len(EXPECTED_GATE_MODULES) != RADIAL_GATE_COUNT
        or identities != expected_identities
    ):
        errors.append("gate CSV must contain the exact 34 modules times two families")
    if any(row.get("gate_health_status") != "pass" for row in rows):
        errors.append("every gate-health row must pass")
    finite_fields = (
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
        "input_rms",
        "output_rms",
        "output_input_rms_ratio",
    )
    positive_fields = (
        "a_grad_norm",
        "b_grad_norm",
        "a_update_to_param_norm",
        "b_update_to_param_norm",
    )
    if any(
        any(not _finite_number(row.get(field)) for field in finite_fields)
        or any(
            not _finite_number(row.get(field), positive=True)
            for field in positive_fields
        )
        or row.get("gate_force_fp32") != "true"
        or row.get("input_dtype") != "float16"
        or row.get("gate_math_dtype") != "float32"
        or row.get("gate_tensor_dtype") != "float16"
        or row.get("output_dtype") != "float16"
        or row.get("requested_autocast_dtype") != "float16"
        or row.get("precision_proof_status") != "pass"
        for row in rows
    ):
        errors.append("every gate row must contain finite actual FP16/FP32 evidence")
    if not (payload_dir / CONFIG_PATH).is_file():
        errors.append("embedded payload is missing the fixed readiness config")
    if errors:
        raise RuntimeError("; ".join(errors))
    if Path("/kaggle/working").exists() and output_dir != DEFAULT_OUTPUT_DIR:
        raise RuntimeError("Kaggle artifact must remain under /kaggle/working")


def _valid_update_proof(value: object) -> bool:
    return (
        isinstance(value, dict)
        and value.get("status") == "pass"
        and isinstance(value.get("parameter"), str)
        and bool(value["parameter"])
        and _finite_number(value.get("gradient_norm"), positive=True)
        and _finite_number(value.get("update_norm"), positive=True)
    )


if __name__ == "__main__":
    raise SystemExit(main())
