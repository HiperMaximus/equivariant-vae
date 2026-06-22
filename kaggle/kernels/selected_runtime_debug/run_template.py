# Copyright 2026 HiperMaximus
"""Generated single-file Kaggle selected-runtime debug gate template."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import cast

KAGGLE_SELECTED_RUNTIME_DEBUG_READY = True
SELECTED_RUNTIME_DEBUG_GATE_CONTRACT_READY = (
    "selected_runtime_debug_gate_contract_ready"
)
GATE_KIND = "kaggle_selected_runtime_debug_resume_tiny_gate"
GATE_SOURCE = "kaggle_selected_runtime_debug_kernel"
EXPECTED_DATASET_SLUG = "maximusshtefan/patches-pre-shuffled-ubc-ocean"
EXPECTED_SELECTED_ROW_ID = (
    "dual_t4_ddp__bs12__amp_conservative__compile_none__indexed_masked__"
    "policy_amp_fp16_conservative"
)
EXPECTED_RUNTIME_POLICY_ID = "amp_fp16_conservative"
KERNEL_METADATA = {
    "id": "maximusshtefan/eqvae-selected-runtime-debug",
    "title": "eqvae selected runtime debug",
    "code_file": "run.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_internet": "false",
    "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": [EXPECTED_DATASET_SLUG],
    "competition_sources": [],
    "kernel_sources": [],
    "model_sources": [],
}
DEFAULT_KAGGLE_OUTPUT_DIR = Path("/kaggle/working")
LOCAL_FALLBACK_OUTPUT_DIR = Path("runs/kaggle/selected_runtime_debug_local")
BASELINE_SELECTED_RUNTIME = Path(
    "runs/kaggle/runtime_selection_v5/benchmark/selected_runtime.json",
)
DEBUG_CONFIG = Path("configs/spec0001/non_eq_vae_selected_runtime_debug.json")
TINY_CONFIG = Path("configs/spec0001/non_eq_vae_kaggle_tiny_overfit.json")
FIXED_TINY_SELECTOR = Path("configs/spec0001/fixed_32_train_overfit_patches.json")
IMPORT_ARTIFACT = "selected_runtime_debug_import.json"
ALLOWED_BENCHMARK_ARTIFACTS = {
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
ALLOWED_METRIC_ARTIFACTS = {
    "gate_health.csv",
    "train_metrics.csv",
}
EMBEDDED_PAYLOAD_B64 = """
$embedded_payload_b64
"""
EMBEDDED_PAYLOAD_ZIP_SHA256 = "$embedded_payload_zip_sha256"
EMBEDDED_PAYLOAD_MANIFEST_SHA256 = "$embedded_payload_manifest_sha256"


def main() -> int:
    """Run the selected-runtime debug gate from an embedded payload.

    Returns:
        Process exit status.

    """
    output_dir = _output_dir()
    return _run_selected_runtime_debug(output_dir)


def _run_selected_runtime_debug(output_dir: Path) -> int:
    _require_python_version()
    payload_dir = _extract_payload(_payload_extract_dir(output_dir))
    payload_src = payload_dir / "src"
    sys.path.insert(0, str(payload_src))
    _prepare_environment(payload_src=payload_src)
    single_visible_t4()
    dual_t4_ddp()
    torchrun_nproc_per_node_2()
    wrong_accelerator()

    import eqvae  # noqa: PLC0415
    from eqvae.cli.selected_runtime_gate import (  # noqa: PLC0415
        main as selected_runtime_gate_main,
    )

    _assert_import_origin(
        module_file=Path(cast("str", eqvae.__file__)),
        payload_src=payload_src,
    )
    manifest_path = payload_dir / "payload_manifest.json"
    payload_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected_runtime_path = payload_dir / BASELINE_SELECTED_RUNTIME
    _validate_baseline_selected_runtime(selected_runtime_path)
    if os.environ.get("EQVAE_SELECTED_RUNTIME_DEBUG_IMPORT_ONLY") == "1":
        _write_import_only_artifact(
            output_dir=output_dir,
            payload_manifest=payload_manifest,
            selected_runtime_path=selected_runtime_path,
        )
        _validate_import_only_artifacts(output_dir=output_dir)
        return 0

    gate_args = [
        "--debug-config",
        str(payload_dir / DEBUG_CONFIG),
        "--tiny-config",
        str(payload_dir / TINY_CONFIG),
        "--runtime-config",
        str(selected_runtime_path),
        "--fixed-train-patches",
        str(payload_dir / FIXED_TINY_SELECTOR),
        "--output-dir",
        str(output_dir),
        "--run-name",
        "non_eq_vae_spec0001_selected_runtime_gate",
    ]
    data_root_override = os.environ.get("EQVAE_SELECTED_RUNTIME_DEBUG_DATA_ROOT")
    if data_root_override:
        gate_args.extend(("--data-root", data_root_override))
    exit_code = selected_runtime_gate_main(tuple(gate_args))
    _validate_gate_artifacts(output_dir=output_dir)
    return exit_code


def _output_dir() -> Path:
    configured = os.environ.get("EQVAE_OUTPUT_DIR")
    if configured:
        output_dir = Path(configured)
    else:
        output_dir = (
            DEFAULT_KAGGLE_OUTPUT_DIR
            if DEFAULT_KAGGLE_OUTPUT_DIR.exists()
            else LOCAL_FALLBACK_OUTPUT_DIR
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir.resolve()


def _payload_extract_dir(output_dir: Path) -> Path:
    if Path("/kaggle/temp").exists():
        return Path("/kaggle/temp/eqvae_selected_runtime_debug_payload")
    if Path("/tmp").exists():  # noqa: S108
        return Path("/tmp/eqvae_selected_runtime_debug_payload")  # noqa: S108
    return output_dir / "embedded_payload"


def _require_python_version() -> None:
    if sys.version_info < (3, 12):  # noqa: UP036
        message = (
            "eqvae selected-runtime debug gate requires Python >= 3.12 because "
            "active source uses Python 3.12 type-alias syntax"
        )
        raise RuntimeError(message)


def _extract_payload(destination: Path) -> Path:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    zip_bytes = base64.b64decode(EMBEDDED_PAYLOAD_B64.encode("ascii"))
    actual_zip_hash = hashlib.sha256(zip_bytes).hexdigest()
    if actual_zip_hash != EMBEDDED_PAYLOAD_ZIP_SHA256:
        message = "embedded payload zip SHA-256 mismatch"
        raise RuntimeError(message)

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        for member in archive.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                message = f"unsafe embedded payload path: {member.filename}"
                raise RuntimeError(message)
        archive.extractall(destination)

    manifest_path = destination / "payload_manifest.json"
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if manifest_hash != EMBEDDED_PAYLOAD_MANIFEST_SHA256:
        message = "embedded payload manifest SHA-256 mismatch"
        raise RuntimeError(message)
    return destination


def _prepare_environment(*, payload_src: Path) -> None:
    os.environ.pop("EQVAE_DATA_ROOT", None)
    existing = os.environ.get("PYTHONPATH")
    entries = [str(payload_src)]
    if existing:
        entries.append(existing)
    os.environ["PYTHONPATH"] = os.pathsep.join(entries)


def single_visible_t4() -> None:
    """Record the single visible T4 selected-runtime validation hook."""
    return


def dual_t4_ddp() -> None:
    """Record the dual_t4_ddp selected-runtime debug/tiny launch hook."""
    return


def torchrun_nproc_per_node_2() -> None:
    """Record the torchrun --standalone --nproc_per_node=2 launch hook."""
    return


def wrong_accelerator() -> None:
    """Record that wrong accelerator states must fail closed."""
    return


def _assert_import_origin(*, module_file: Path, payload_src: Path) -> None:
    resolved_module = module_file.resolve()
    resolved_payload_src = payload_src.resolve()
    if resolved_payload_src not in resolved_module.parents:
        message = f"eqvae imported from {resolved_module}, not {resolved_payload_src}"
        raise RuntimeError(message)


def _validate_baseline_selected_runtime(path: Path) -> None:
    if not path.exists():
        message = f"missing embedded selected runtime: {path}"
        raise RuntimeError(message)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        message = "embedded selected runtime must be a JSON object"
        raise TypeError(message)
    snapshot = payload.get("selected_row_snapshot")
    if not isinstance(snapshot, dict):
        message = "embedded selected runtime missing selected_row_snapshot"
        raise TypeError(message)
    errors = [
        *_baseline_top_level_errors(payload),
        *_baseline_launch_errors(payload),
        *_baseline_snapshot_errors(cast("dict[str, object]", snapshot)),
    ]
    if errors:
        raise RuntimeError("; ".join(errors))


def _baseline_top_level_errors(payload: dict[str, object]) -> list[str]:
    required = {
        "status": "pass",
        "benchmark_kind": "kaggle_runtime_selection",
        "benchmark_source": "kaggle_runtime_benchmark",
        "selected_row_id": EXPECTED_SELECTED_ROW_ID,
        "runtime_policy_id": EXPECTED_RUNTIME_POLICY_ID,
        "accelerator_mode": "dual_t4_ddp",
        "machine_shape": "NvidiaTeslaT4",
    }
    errors = [
        f"selected_runtime.{key} must be {expected!r}"
        for key, expected in required.items()
        if payload.get(key) != expected
    ]
    if payload.get("full_run_eligible") is not True:
        errors.append("selected runtime must be full_run_eligible")
    if payload.get("full_training_launch_ready") is not False:
        errors.append("selected runtime must not be full-training-launch-ready")
    return errors


def _baseline_launch_errors(payload: dict[str, object]) -> list[str]:
    expected = {
        "world_size": 2,
        "nproc_per_node": 2,
        "per_device_batch_size": 12,
        "global_batch_size": 24,
        "gradient_accumulation_steps": 1,
    }
    errors = [
        f"selected_runtime.{key} must be {expected!r}"
        for key, expected in expected.items()
        if payload.get(key) != expected
    ]
    mixed_precision = payload.get("mixed_precision")
    if not isinstance(mixed_precision, dict):
        errors.append("selected_runtime.mixed_precision must be an object")
    else:
        errors.extend(_baseline_mixed_precision_errors(mixed_precision))
    dataloader = payload.get("dataloader")
    if not isinstance(dataloader, dict):
        errors.append("selected_runtime.dataloader must be an object")
    else:
        errors.extend(_baseline_dataloader_errors(dataloader))
    corruption = payload.get("corruption")
    if not isinstance(corruption, dict):
        errors.append("selected_runtime.corruption must be an object")
    elif corruption.get("strategy") != "indexed_masked":
        errors.append("selected_runtime.corruption.strategy must be 'indexed_masked'")
    return errors


def _baseline_mixed_precision_errors(payload: dict[str, object]) -> list[str]:
    expected = {
        "enabled": True,
        "policy": "amp_conservative",
        "autocast_dtype": "float16",
        "fp32_loss": True,
        "grad_scaler_enabled": True,
    }
    return [
        f"selected_runtime.mixed_precision.{key} must be {expected!r}"
        for key, expected in expected.items()
        if payload.get(key) != expected
    ]


def _baseline_dataloader_errors(payload: dict[str, object]) -> list[str]:
    expected = {
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
        "non_blocking_h2d": True,
    }
    errors = [
        f"selected_runtime.dataloader.{key} must be {expected!r}"
        for key, expected in expected.items()
        if payload.get(key) != expected
    ]
    if payload.get("prefetch_factor") is not None:
        errors.append("selected_runtime.dataloader.prefetch_factor must be null")
    return errors


def _baseline_snapshot_errors(snapshot: dict[str, object]) -> list[str]:
    snapshot_required = {
        "row_id": EXPECTED_SELECTED_ROW_ID,
        "runtime_policy_id": EXPECTED_RUNTIME_POLICY_ID,
        "status": "pass",
        "accelerator_mode": "dual_t4_ddp",
        "machine_shape": "NvidiaTeslaT4",
        "precision_policy": "amp_conservative",
        "corruption_strategy": "indexed_masked",
        "grad_scaler_enabled": "true",
        "autocast_dtype": "float16",
        "world_size": "2",
        "nproc_per_node": "2",
        "per_device_batch_size": "12",
        "global_batch_size": "24",
    }
    return [
        f"selected_runtime.selected_row_snapshot.{key} must be {expected!r}"
        for key, expected in snapshot_required.items()
        if snapshot.get(key) != expected
    ]


def _write_import_only_artifact(
    *,
    output_dir: Path,
    payload_manifest: dict[str, object],
    selected_runtime_path: Path,
) -> Path:
    benchmark_dir = output_dir / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "import_smoke_pass",
        "status_scope": "non_promotable_local_upload_simulation",
        "benchmark_kind": "selected_runtime_debug_import_only",
        "benchmark_source": GATE_SOURCE,
        "full_run_eligible": False,
        "writes_selected_runtime": False,
        "selected_runtime_path": str(selected_runtime_path),
        "selected_runtime_exists": selected_runtime_path.exists(),
        "payload_manifest": payload_manifest,
    }
    output_path = benchmark_dir / IMPORT_ARTIFACT
    output_path.write_text(
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    return output_path


def _validate_import_only_artifacts(*, output_dir: Path) -> None:
    benchmark_dir = output_dir / "benchmark"
    observed = {path.name for path in benchmark_dir.iterdir()}
    if observed != {IMPORT_ARTIFACT}:
        message = f"unexpected import-only artifacts: {sorted(observed)}"
        raise RuntimeError(message)
    if (benchmark_dir / "selected_runtime.json").exists():
        message = "selected-runtime debug import simulation wrote selected_runtime"
        raise RuntimeError(message)


def _validate_gate_artifacts(*, output_dir: Path) -> None:
    benchmark_dir = output_dir / "benchmark"
    metrics_dir = output_dir / "metrics"
    observed_benchmark = {path.name for path in benchmark_dir.iterdir()}
    unexpected = observed_benchmark - ALLOWED_BENCHMARK_ARTIFACTS
    missing = ALLOWED_BENCHMARK_ARTIFACTS - observed_benchmark
    if unexpected or missing:
        message = (
            "unexpected selected-runtime debug benchmark artifacts: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
        raise RuntimeError(message)
    observed_metrics = (
        {path.name for path in metrics_dir.iterdir()} if metrics_dir.exists() else set()
    )
    if observed_metrics != ALLOWED_METRIC_ARTIFACTS:
        message = (
            "unexpected selected-runtime debug metric artifacts: "
            f"{sorted(observed_metrics)}"
        )
        raise RuntimeError(message)
    if (benchmark_dir / "selected_runtime.json").exists():
        message = (
            "selected-runtime debug gate must consume, not write, selected_runtime"
        )
        raise RuntimeError(message)
    gate_summary = _validate_json_artifact(
        benchmark_dir / "selected_runtime_gate_summary.json",
    )
    if gate_summary.get("benchmark_kind") != GATE_KIND:
        message = "selected-runtime gate summary has wrong benchmark_kind"
        raise RuntimeError(message)
    if gate_summary.get("benchmark_source") != GATE_SOURCE:
        message = "selected-runtime gate summary has wrong benchmark_source"
        raise RuntimeError(message)
    if gate_summary.get("full_run_eligible") is not False:
        message = "selected-runtime gate must remain non-promotable until pass"
        raise RuntimeError(message)


def _validate_json_artifact(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        message = f"{path.name} must contain a JSON object"
        raise TypeError(message)
    return cast("dict[str, object]", payload)


if __name__ == "__main__":
    raise SystemExit(main())
