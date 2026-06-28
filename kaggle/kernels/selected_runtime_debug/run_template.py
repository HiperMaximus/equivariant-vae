# Copyright 2026 HiperMaximus
"""Generated single-file Kaggle selected-runtime debug gate template."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
import shutil
import subprocess  # noqa: S404
import sys
import traceback
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

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
GENERATED_FIXED_TINY_SELECTOR = Path("benchmark/fixed_32_train_overfit_patches.json")
FIXED32_SELECTOR_READINESS_ARTIFACT = "fixed32_selector_readiness.json"
DEBUG_RESUME_STEP = 4
DEBUG_FINAL_STEP = 8
TINY_MAX_STEP = 128
TINY_SAVE_EVERY_STEP = 64
FIXED_TINY_SELECTOR_COUNT = 32
TINY_MIN_IMPROVEMENT_FRACTION = 0.01
IMPORT_ARTIFACT = "selected_runtime_debug_import.json"
ALLOWED_BENCHMARK_ARTIFACTS = {
    "artifact_manifest.json",
    "checkpoint_resume_proof.json",
    "fixed32_selector_readiness.json",
    "gate_health_summary.json",
    "local_selected_runtime_readiness.json",
    "selected_runtime_plan_applied.json",
    "selected_runtime_debug_summary.json",
    "selected_runtime_gate_summary.json",
    "tiny_overfit_summary.json",
    "training_summary.json",
}
OPTIONAL_BENCHMARK_ARTIFACTS = {
    "fixed_32_train_overfit_patches.json",
}
REAL_REQUIRED_BENCHMARK_ARTIFACTS = {
    "artifact_manifest.json",
    "checkpoint_resume_proof.json",
    "fixed32_selector_readiness.json",
    "fixed_32_train_overfit_patches.json",
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
    from eqvae.benchmarking.fixed32_selector_readiness import (  # noqa: PLC0415
        fixed32_selector_status,
    )
    from eqvae.cli.select_fixed_patches import (  # noqa: PLC0415
        main as select_fixed_patches_main,
    )
    from eqvae.cli.selected_runtime_gate import (  # noqa: PLC0415
        main as selected_runtime_gate_main,
    )
    from eqvae.cli.selected_runtime_train import (  # noqa: PLC0415
        main as selected_runtime_train_main,
    )

    _assert_import_origin(
        module_file=Path(cast("str", eqvae.__file__)),
        payload_src=payload_src,
    )
    manifest_path = payload_dir / "payload_manifest.json"
    payload_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected_runtime_path = payload_dir / BASELINE_SELECTED_RUNTIME
    _validate_baseline_selected_runtime(selected_runtime_path)
    _ensure_selected_runtime_train_entrypoint(selected_runtime_train_main)
    if os.environ.get("EQVAE_SELECTED_RUNTIME_DEBUG_IMPORT_ONLY") == "1":
        _write_import_only_artifact(
            output_dir=output_dir,
            payload_manifest=payload_manifest,
            selected_runtime_path=selected_runtime_path,
        )
        _validate_import_only_artifacts(output_dir=output_dir)
        return 0

    data_root_override = os.environ.get("EQVAE_SELECTED_RUNTIME_DEBUG_DATA_ROOT")
    data_root_value = data_root_override or "auto"
    generated_selector_path = output_dir / GENERATED_FIXED_TINY_SELECTOR
    selector_generation = _generate_remote_fixed32_selector(
        select_fixed_patches_main=select_fixed_patches_main,
        fixed32_selector_status=fixed32_selector_status,
        payload_dir=payload_dir,
        output_dir=output_dir,
        data_root=data_root_value,
        selector_path=generated_selector_path,
    )
    if selector_generation.get("status") == "pass":
        exit_code = _run_real_selected_runtime_debug(
            payload_src=payload_src,
            payload_dir=payload_dir,
            output_dir=output_dir,
            selected_runtime_path=selected_runtime_path,
            data_root=data_root_value,
            fixed_train_patches=generated_selector_path,
        )
        if exit_code != 0:
            return exit_code
        exit_code = _run_real_selected_runtime_tiny_overfit(
            payload_src=payload_src,
            payload_dir=payload_dir,
            output_dir=output_dir,
            selected_runtime_path=selected_runtime_path,
            data_root=data_root_value,
            fixed_train_patches=generated_selector_path,
        )
        if exit_code != 0:
            return exit_code
        _write_real_gate_summary(
            output_dir=output_dir,
            selector_generation=selector_generation,
            selected_runtime_path=selected_runtime_path,
        )
        _validate_real_runner_artifacts(output_dir=output_dir)
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
        "--selector-generation-mode",
        "remote_generate",
        "--output-dir",
        str(output_dir),
        "--run-name",
        "non_eq_vae_spec0001_selected_runtime_gate",
    ]
    if data_root_override:
        gate_args.extend(("--data-root", data_root_override))
    exit_code = selected_runtime_gate_main(tuple(gate_args))
    _validate_gate_artifacts(output_dir=output_dir)
    return exit_code


def _ensure_selected_runtime_train_entrypoint(
    selected_runtime_train_main: Callable[[Sequence[str] | None], int],
) -> None:
    if not callable(selected_runtime_train_main):
        message = "selected_runtime_train entrypoint is not callable"
        raise TypeError(message)


def _generate_remote_fixed32_selector(  # noqa: PLR0913
    *,
    select_fixed_patches_main: Callable[[Sequence[str] | None], int],
    fixed32_selector_status: Callable[..., dict[str, object]],
    payload_dir: Path,
    output_dir: Path,
    data_root: str,
    selector_path: Path,
) -> dict[str, object]:
    benchmark_dir = output_dir / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = benchmark_dir / FIXED32_SELECTOR_READINESS_ARTIFACT
    args = (
        "--config",
        str(payload_dir / TINY_CONFIG),
        "--kind",
        "fixed_32_train_overfit",
        "--data-root",
        data_root,
        "--masked-holdout-csv",
        str(payload_dir / "docs/data/ubc_ocean_masked_holdout_ids.csv"),
        "--output",
        str(selector_path),
        "--validate-crc",
    )
    try:
        exit_code = select_fixed_patches_main(args)
    except Exception as error:  # noqa: BLE001
        payload = {
            "schema_version": "spec0008.fixed32_selector_remote_generation.v1",
            "status": "fail",
            "selector_generation_mode": "remote_generate",
            "remote_selector_generation_ready": False,
            "fixed_32_selector_real": False,
            "selector_path": str(selector_path),
            "data_root": data_root,
            "failure_kind": "fixed32_remote_selector_generation_failed",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback_excerpt": traceback.format_exc(limit=4),
        }
    else:
        selector_status = (
            _fixed32_selector_status_from_payload_cwd(
                fixed32_selector_status=fixed32_selector_status,
                payload_dir=payload_dir,
                selector_path=selector_path,
                data_root=data_root,
            )
            if exit_code == 0
            else {
                "status": "fail",
                "canonical_real_ubc": False,
                "failure_kind": "fixed32_selector_cli_failed",
            }
        )
        selector_real = (
            selector_status.get("status") == "pass"
            and selector_status.get("canonical_real_ubc") is True
        )
        payload = {
            "schema_version": "spec0008.fixed32_selector_remote_generation.v1",
            "status": "pass" if selector_real else "fail",
            "selector_generation_mode": "remote_generate",
            "remote_selector_generation_ready": selector_real,
            "fixed_32_selector_real": selector_real,
            "selector_path": str(selector_path),
            "data_root": data_root,
            "selector_status": selector_status,
            "failure_kind": ""
            if selector_real
            else _selector_generation_failure_kind(selector_status),
        }
    artifact_path.write_text(
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    return payload


def _fixed32_selector_status_from_payload_cwd(
    *,
    fixed32_selector_status: Callable[..., dict[str, object]],
    payload_dir: Path,
    selector_path: Path,
    data_root: str,
) -> dict[str, object]:
    original_cwd = Path.cwd()
    try:
        os.chdir(payload_dir)
        return fixed32_selector_status(selector_path, data_root=data_root)
    finally:
        os.chdir(original_cwd)


def _selector_generation_failure_kind(selector_status: dict[str, object]) -> str:
    failure_kind = selector_status.get("failure_kind")
    if isinstance(failure_kind, str) and failure_kind:
        return failure_kind
    return "fixed32_remote_selector_not_canonical_real"


def _run_real_selected_runtime_debug(  # noqa: PLR0913
    *,
    payload_src: Path,
    payload_dir: Path,
    output_dir: Path,
    selected_runtime_path: Path,
    data_root: str,
    fixed_train_patches: Path,
) -> int:
    phase1_dir = output_dir / "resume_probe_phase1"
    phase1_args = [
        "--config",
        str(payload_dir / DEBUG_CONFIG),
        "--runtime-config",
        str(selected_runtime_path),
        "--data",
        "ubc-pre-shuffled",
        "--data-root",
        data_root,
        "--fixed-train-patches",
        str(fixed_train_patches),
        "--output-dir",
        str(phase1_dir),
        "--run-name",
        "non_eq_vae_spec0001_selected_runtime_debug_resume_probe",
        "--max-train-steps",
        str(DEBUG_RESUME_STEP),
        "--max-val-steps",
        "1",
        "--save-every-steps",
        str(DEBUG_RESUME_STEP),
    ]
    phase1_exit_code = _run_selected_runtime_train_torchrun(
        payload_src=payload_src,
        args=tuple(phase1_args),
    )
    if phase1_exit_code != 0:
        return phase1_exit_code
    resume_checkpoint = phase1_dir / "checkpoints" / f"step_{DEBUG_RESUME_STEP:06d}.pt"
    if not resume_checkpoint.exists():
        message = f"selected-runtime resume checkpoint missing: {resume_checkpoint}"
        raise RuntimeError(message)
    runner_args = [
        "--config",
        str(payload_dir / DEBUG_CONFIG),
        "--runtime-config",
        str(selected_runtime_path),
        "--data",
        "ubc-pre-shuffled",
        "--data-root",
        data_root,
        "--fixed-train-patches",
        str(fixed_train_patches),
        "--output-dir",
        str(output_dir),
        "--run-name",
        "non_eq_vae_spec0001_selected_runtime_debug",
        "--resume",
        str(resume_checkpoint),
        "--max-train-steps",
        str(DEBUG_FINAL_STEP),
        "--max-val-steps",
        "1",
        "--save-every-steps",
        str(DEBUG_RESUME_STEP),
    ]
    return _run_selected_runtime_train_torchrun(
        payload_src=payload_src,
        args=tuple(runner_args),
    )


def _run_real_selected_runtime_tiny_overfit(  # noqa: PLR0913
    *,
    payload_src: Path,
    payload_dir: Path,
    output_dir: Path,
    selected_runtime_path: Path,
    data_root: str,
    fixed_train_patches: Path,
) -> int:
    tiny_output_dir = output_dir / "tiny_overfit_phase"
    args = [
        "--config",
        str(payload_dir / TINY_CONFIG),
        "--runtime-config",
        str(selected_runtime_path),
        "--data",
        "ubc-pre-shuffled",
        "--data-root",
        data_root,
        "--fixed-train-patches",
        str(fixed_train_patches),
        "--output-dir",
        str(tiny_output_dir),
        "--run-name",
        "non_eq_vae_spec0001_selected_runtime_tiny_overfit",
        "--max-train-steps",
        str(TINY_MAX_STEP),
        "--max-val-steps",
        "1",
        "--save-every-steps",
        str(TINY_SAVE_EVERY_STEP),
    ]
    exit_code = _run_selected_runtime_train_torchrun(
        payload_src=payload_src,
        args=tuple(args),
    )
    if exit_code != 0:
        return exit_code
    source = tiny_output_dir / "benchmark" / "tiny_overfit_summary.json"
    if not source.exists():
        message = f"selected-runtime tiny-overfit summary missing: {source}"
        raise RuntimeError(message)
    payload = _validate_json_artifact(source)
    payload["source_output_dir"] = str(tiny_output_dir)
    payload["source_summary_sha256"] = _sha256_file(source)
    target = output_dir / "benchmark" / "tiny_overfit_summary.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    return 0


def _run_selected_runtime_train_torchrun(
    *,
    payload_src: Path,
    args: Sequence[str],
) -> int:
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(payload_src)
        if not existing_pythonpath
        else f"{payload_src}{os.pathsep}{existing_pythonpath}"
    )
    completed = subprocess.run(  # noqa: S603
        _selected_runtime_train_torchrun_command(args),
        env=environment,
        check=False,
    )
    return int(completed.returncode)


def _selected_runtime_train_torchrun_command(args: Sequence[str]) -> tuple[str, ...]:
    return (
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        "-m",
        "eqvae.cli.selected_runtime_train",
        *tuple(args),
    )


def _write_real_gate_summary(
    *,
    output_dir: Path,
    selector_generation: dict[str, object],
    selected_runtime_path: Path,
) -> Path:
    benchmark_dir = output_dir / "benchmark"
    training_summary = _validate_json_artifact(benchmark_dir / "training_summary.json")
    plan_applied = _validate_json_artifact(
        benchmark_dir / "selected_runtime_plan_applied.json",
    )
    resume_proof = _validate_json_artifact(
        benchmark_dir / "checkpoint_resume_proof.json",
    )
    gate_health_summary = _validate_json_artifact(
        benchmark_dir / "gate_health_summary.json",
    )
    artifact_manifest = _validate_json_artifact(
        benchmark_dir / "artifact_manifest.json",
    )
    tiny_summary = _validate_json_artifact(benchmark_dir / "tiny_overfit_summary.json")
    component_status = {
        "selector_generation": selector_generation.get("status"),
        "real_ubc_debug": training_summary.get("status"),
        "selected_runtime_plan_applied": plan_applied.get("status"),
        "checkpoint_resume": resume_proof.get("status"),
        "gate_health": gate_health_summary.get("status"),
        "artifact_manifest": artifact_manifest.get("status"),
        "tiny_overfit": tiny_summary.get("status"),
    }
    launch_blockers = [
        name
        for name, status in sorted(component_status.items())
        if status not in {"pass", "local_pass"}
    ]
    payload = {
        "schema_version": "spec0008.selected_runtime_debug_real_gate.v1",
        "status": "local_pass" if not launch_blockers else "fail",
        "status_scope": "permission_gated_remote_debug_tiny_proof",
        "benchmark_kind": GATE_KIND,
        "benchmark_source": GATE_SOURCE,
        "full_run_eligible": False,
        "selected_runtime_path": str(selected_runtime_path),
        "component_status": component_status,
        "launch_blockers_remaining": launch_blockers,
        "selector_generation": selector_generation,
    }
    path = benchmark_dir / "selected_runtime_gate_summary.json"
    path.write_text(
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    _write_real_artifact_manifest(output_dir=output_dir)
    return path


def _write_real_artifact_manifest(*, output_dir: Path) -> Path:
    benchmark_dir = output_dir / "benchmark"
    metrics_dir = output_dir / "metrics"
    reconstruction_samples = output_dir / "artifacts" / "reconstruction_samples.pt"
    artifacts = {
        f"benchmark:{name}": benchmark_dir / name
        for name in sorted(REAL_REQUIRED_BENCHMARK_ARTIFACTS)
        if name != "artifact_manifest.json"
    }
    artifacts["metrics:gate_health"] = metrics_dir / "gate_health.csv"
    artifacts["metrics:train_steps"] = metrics_dir / "train_steps.csv"
    artifacts["artifact:reconstruction_samples"] = reconstruction_samples
    missing = [name for name, path in sorted(artifacts.items()) if not path.exists()]
    payload = {
        "schema_version": "spec0008.selected_runtime_debug_artifact_manifest.v1",
        "status": "local_pass" if not missing else "fail",
        "status_scope": "permission_gated_remote_debug_tiny_proof",
        "full_run_eligible": False,
        "artifact_hashes": {
            name: _sha256_file(path)
            for name, path in sorted(artifacts.items())
            if path.exists()
        },
        "missing_artifacts": missing,
        "metric_row_count": _csv_row_count(metrics_dir / "train_steps.csv"),
        "reconstruction_sample_nonblank": reconstruction_samples.exists(),
    }
    path = benchmark_dir / "artifact_manifest.json"
    path.write_text(
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    return path


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
    unexpected = (
        observed_benchmark - ALLOWED_BENCHMARK_ARTIFACTS - OPTIONAL_BENCHMARK_ARTIFACTS
    )
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


def _validate_real_runner_artifacts(  # noqa: C901, PLR0912, PLR0914, PLR0915
    *,
    output_dir: Path,
) -> None:
    benchmark_dir = output_dir / "benchmark"
    metrics_dir = output_dir / "metrics"
    required_benchmark = REAL_REQUIRED_BENCHMARK_ARTIFACTS
    observed_benchmark = {path.name for path in benchmark_dir.iterdir()}
    unexpected = observed_benchmark - required_benchmark
    missing = required_benchmark - observed_benchmark
    if unexpected or missing:
        message = (
            "unexpected selected-runtime real-runner benchmark artifacts: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
        raise RuntimeError(message)
    observed_metrics = (
        {path.name for path in metrics_dir.iterdir()} if metrics_dir.exists() else set()
    )
    if observed_metrics != {"gate_health.csv", "train_steps.csv"}:
        message = (
            "unexpected selected-runtime real-runner metric artifacts: "
            f"{sorted(observed_metrics)}"
        )
        raise RuntimeError(message)
    if (benchmark_dir / "selected_runtime.json").exists():
        message = "selected-runtime debug must consume, not write, selected_runtime"
        raise RuntimeError(message)
    training_summary = _validate_json_artifact(benchmark_dir / "training_summary.json")
    debug_summary = _validate_json_artifact(
        benchmark_dir / "selected_runtime_debug_summary.json",
    )
    resume_proof = _validate_json_artifact(
        benchmark_dir / "checkpoint_resume_proof.json",
    )
    selector_readiness = _validate_json_artifact(
        benchmark_dir / FIXED32_SELECTOR_READINESS_ARTIFACT,
    )
    plan_applied = _validate_json_artifact(
        benchmark_dir / "selected_runtime_plan_applied.json",
    )
    gate_health_summary = _validate_json_artifact(
        benchmark_dir / "gate_health_summary.json",
    )
    artifact_manifest = _validate_json_artifact(
        benchmark_dir / "artifact_manifest.json",
    )
    tiny_summary = _validate_json_artifact(
        benchmark_dir / "tiny_overfit_summary.json",
    )
    gate_summary = _validate_json_artifact(
        benchmark_dir / "selected_runtime_gate_summary.json",
    )
    if training_summary.get("optimizer_steps_completed") != DEBUG_FINAL_STEP:
        message = "selected-runtime debug must complete exactly 8 optimizer updates"
        raise RuntimeError(message)
    if training_summary.get("amp_step_skipped_count") != 0:
        message = "selected-runtime debug must have zero AMP skipped updates"
        raise RuntimeError(message)
    if training_summary.get("nonfinite_count") != 0:
        message = "selected-runtime debug must have zero nonfinite rows"
        raise RuntimeError(message)
    if debug_summary.get("remote_pass_ready") is not False:
        message = "local artifact must not claim remote_pass_ready in wrapper"
        raise RuntimeError(message)
    if plan_applied.get("status") != "local_pass":
        message = "selected-runtime plan-applied proof did not pass"
        raise RuntimeError(message)
    if plan_applied.get("plan_applied") is not True:
        message = "selected-runtime plan-applied proof must be true"
        raise RuntimeError(message)
    if gate_health_summary.get("status") != "local_pass":
        message = "selected-runtime gate-health summary did not pass"
        raise RuntimeError(message)
    if artifact_manifest.get("status") != "local_pass":
        message = "selected-runtime artifact manifest did not pass"
        raise RuntimeError(message)
    if artifact_manifest.get("reconstruction_sample_nonblank") is not True:
        message = "selected-runtime reconstruction sample must be nonblank"
        raise RuntimeError(message)
    if (
        resume_proof.get("loaded_successful_optimizer_update_count")
        != DEBUG_RESUME_STEP
    ):
        message = "selected-runtime debug must resume from update 4"
        raise RuntimeError(message)
    if resume_proof.get("additional_optimizer_steps") != DEBUG_RESUME_STEP:
        message = "selected-runtime debug must continue four updates after resume"
        raise RuntimeError(message)
    if selector_readiness.get("fixed_32_selector_real") is not True:
        message = "real-runner branch requires canonical fixed-32 selector proof"
        raise RuntimeError(message)
    if tiny_summary.get("status") != "local_pass":
        message = "selected-runtime tiny-overfit summary did not pass"
        raise RuntimeError(message)
    if tiny_summary.get("patch_count") != FIXED_TINY_SELECTOR_COUNT:
        message = "selected-runtime tiny-overfit must use the fixed 32 selector"
        raise RuntimeError(message)
    if tiny_summary.get("optimizer_steps") != TINY_MAX_STEP:
        message = "selected-runtime tiny-overfit must run exactly the tiny cap"
        raise RuntimeError(message)
    if tiny_summary.get("l1_improvement_fraction", 0.0) < TINY_MIN_IMPROVEMENT_FRACTION:
        message = "selected-runtime tiny-overfit L1 improvement is below threshold"
        raise RuntimeError(message)
    if (
        tiny_summary.get("recon_loss_improvement_fraction", 0.0)
        < TINY_MIN_IMPROVEMENT_FRACTION
    ):
        message = "selected-runtime tiny-overfit recon improvement is below threshold"
        raise RuntimeError(message)
    if gate_summary.get("status") != "local_pass":
        message = "selected-runtime gate summary did not pass"
        raise RuntimeError(message)
    _validate_train_steps_csv(metrics_dir / "train_steps.csv")


def _validate_json_artifact(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        message = f"{path.name} must contain a JSON object"
        raise TypeError(message)
    return cast("dict[str, object]", payload)


def _validate_train_steps_csv(path: Path) -> None:
    with path.open(encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        message = "selected-runtime train_steps.csv must contain rows"
        raise RuntimeError(message)
    successful_steps = {
        int(row["successful_optimizer_update_count"])
        for row in rows
        if row.get("amp_step_skipped") == "0"
    }
    if max(successful_steps, default=0) != DEBUG_FINAL_STEP:
        message = "selected-runtime train_steps.csv did not reach update 8"
        raise RuntimeError(message)
    if min(successful_steps, default=DEBUG_FINAL_STEP) <= DEBUG_RESUME_STEP:
        message = "selected-runtime train_steps.csv must contain resumed updates only"
        raise RuntimeError(message)
    if any(row.get("amp_step_skipped") != "0" for row in rows):
        message = "selected-runtime train_steps.csv contains AMP skipped rows"
        raise RuntimeError(message)
    if any(int(row.get("nonfinite_count", "0")) != 0 for row in rows):
        message = "selected-runtime train_steps.csv contains nonfinite rows"
        raise RuntimeError(message)


def _csv_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8", newline="") as csv_file:
        return sum(1 for _ in csv.DictReader(csv_file))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
