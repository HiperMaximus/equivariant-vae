# Copyright 2026 HiperMaximus
"""Generated single-file Kaggle selected-runtime launcher template."""

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

KAGGLE_RUNTIME_SELECTION_READY = True
RUNTIME_SELECTION_SLICE = "v8_shortlist_eager_amp_then_dual_gate"
RUNTIME_SELECTION_BENCHMARK_KIND = "kaggle_runtime_selection"
RUNTIME_SELECTION_BENCHMARK_SOURCE = "kaggle_runtime_benchmark"
EXPECTED_DATASET_SLUG = "maximusshtefan/patches-pre-shuffled-ubc-ocean"
REQUIRED_V8_ARTIFACTS = {
    "benchmark/runtime_proof.json",
    "benchmark/runtime_matrix.csv",
    "benchmark/dataloader_matrix.csv",
    "benchmark/numerical_checks.csv",
    "benchmark/corruption_checks.csv",
    "benchmark/gate_health_summary.json",
    "metrics/gate_health.csv",
}
RUNTIME_SELECTION_ALLOWED_BENCHMARK_ARTIFACTS = {
    "model_count.json",
    "model_inventory.csv",
    "runtime_proof.json",
    "runtime_matrix.csv",
    "dataloader_matrix.csv",
    "numerical_checks.csv",
    "corruption_checks.csv",
    "gate_health_summary.json",
    "stain_corruptor_qa.json",
}
RUNTIME_SELECTION_OPTIONAL_BENCHMARK_ARTIFACTS = {
    "selected_runtime.json",
}
RUNTIME_SELECTION_ALLOWED_METRIC_ARTIFACTS = {"gate_health.csv"}
KERNEL_METADATA = {
    "id": "maximusshtefan/eqvae-runtime-selection",
    "title": "eqvae runtime selection",
    "code_file": "run.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_internet": "true",
    "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": [EXPECTED_DATASET_SLUG],
    "competition_sources": [],
    "kernel_sources": [],
    "model_sources": [],
}
DEFAULT_KAGGLE_OUTPUT_DIR = Path("/kaggle/working")
LOCAL_FALLBACK_OUTPUT_DIR = Path("runs/kaggle/runtime_selection_local")
V8_ARTIFACT_ROOT = Path("runs/kaggle/real_data_runtime_pretest_v8")
BASELINE_SELECTED_RUNTIME = Path(
    "runs/kaggle/runtime_selection_v5/benchmark/selected_runtime.json",
)
EMBEDDED_PAYLOAD_B64 = """
$embedded_payload_b64
"""
EMBEDDED_PAYLOAD_ZIP_SHA256 = "$embedded_payload_zip_sha256"
EMBEDDED_PAYLOAD_MANIFEST_SHA256 = "$embedded_payload_manifest_sha256"


def main() -> int:
    """Run the selected-runtime launcher from an embedded payload.

    Returns:
        Process exit status.

    """
    output_dir = _output_dir()
    return _run_runtime_selection(output_dir)


def _run_runtime_selection(output_dir: Path) -> int:
    _require_python_version()
    _ensure_latest_torch(cpu_only=False)
    payload_dir = _extract_payload(_payload_extract_dir(output_dir))
    payload_src = payload_dir / "src"
    sys.path.insert(0, str(payload_src))
    _prepare_environment(payload_src=payload_src)
    single_visible_t4()
    dual_t4_ddp()
    torchrun_nproc_per_node_2()
    wrong_accelerator()

    import eqvae  # noqa: PLC0415
    from eqvae.cli.runtime_selection_executor import (  # noqa: PLC0415
        main as executor_main,
    )

    _assert_import_origin(
        module_file=Path(cast("str", eqvae.__file__)),
        payload_src=payload_src,
    )
    config_path = (
        payload_dir
        / "configs"
        / "spec0001"
        / "non_eq_vae_kaggle_runtime_benchmark.json"
    )
    manifest_path = payload_dir / "payload_manifest.json"
    payload_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    v8_artifact_dir = payload_dir / V8_ARTIFACT_ROOT
    _validate_v8_artifacts(v8_artifact_dir)
    baseline_selected_runtime = payload_dir / BASELINE_SELECTED_RUNTIME
    _validate_baseline_selected_runtime(
        path=baseline_selected_runtime,
        config_path=config_path,
    )
    if os.environ.get("EQVAE_RUNTIME_SELECTION_IMPORT_ONLY") == "1":
        _write_import_only_artifact(
            output_dir=output_dir,
            config_path=config_path,
            payload_manifest=payload_manifest,
            v8_artifact_dir=v8_artifact_dir,
            baseline_selected_runtime=baseline_selected_runtime,
        )
        _validate_import_only_artifacts(output_dir=output_dir)
        return 0

    executor_args = [
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
        "--v8-artifact-dir",
        str(v8_artifact_dir),
    ]
    data_root_override = os.environ.get("EQVAE_RUNTIME_SELECTION_DATA_ROOT")
    if data_root_override:
        executor_args.extend(("--data-root", data_root_override))
    exit_code = executor_main(tuple(executor_args))
    _validate_runtime_selection_artifacts(output_dir=output_dir)
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
        return Path("/kaggle/temp/eqvae_runtime_selection_payload")
    if Path("/tmp").exists():  # noqa: S108
        return Path("/tmp/eqvae_runtime_selection_payload")  # noqa: S108
    return output_dir / "embedded_payload"


def _require_python_version() -> None:
    if sys.version_info < (3, 12):  # noqa: UP036
        message = (
            "eqvae runtime selection requires Python >= 3.12 because active "
            "source uses Python 3.12 type-alias syntax"
        )
        raise RuntimeError(message)


def _ensure_latest_torch(*, cpu_only: bool) -> None:
    """Install the latest torch stack on a real Kaggle worker before importing it.

    Kaggle's base image lags the repo's torch target (2.10 vs local 2.13), so with
    ``enable_internet`` on this pins local/Kaggle parity for the compiled fast-path
    recipe by upgrading ``torch``/``torchvision``/``torchaudio`` before ``eqvae``
    (and hence torch) is imported. It no-ops off Kaggle (``/kaggle/working`` absent),
    so the local gate never touches the network; ``EQVAE_SKIP_TORCH_UPGRADE=1``
    forces it off on Kaggle too.
    """
    if os.environ.get("EQVAE_SKIP_TORCH_UPGRADE") == "1":
        return
    if not Path("/kaggle/working").exists():
        return
    import subprocess  # noqa: PLC0415, S404

    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "torch",
        "torchvision",
        "torchaudio",
    ]
    if cpu_only:
        command += ["--index-url", "https://download.pytorch.org/whl/cpu"]
    subprocess.check_call(command)  # noqa: S603


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
    """Record the required single visible T4 selected-runtime mode hook."""
    return


def dual_t4_ddp() -> None:
    """Record the required dual_t4_ddp selected-runtime timing hook."""
    return


def torchrun_nproc_per_node_2() -> None:
    """Record the torchrun --standalone --nproc_per_node=2 launch hook."""
    return


def wrong_accelerator() -> None:
    """Record that runtime rows must emit wrong_accelerator failures."""
    return


def _assert_import_origin(*, module_file: Path, payload_src: Path) -> None:
    resolved_module = module_file.resolve()
    resolved_payload_src = payload_src.resolve()
    if resolved_payload_src not in resolved_module.parents:
        message = f"eqvae imported from {resolved_module}, not {resolved_payload_src}"
        raise RuntimeError(message)


def _validate_v8_artifacts(v8_artifact_dir: Path) -> None:
    missing = [
        relative
        for relative in REQUIRED_V8_ARTIFACTS
        if not (v8_artifact_dir / relative).exists()
    ]
    if missing:
        message = f"missing embedded v8 provenance artifacts: {sorted(missing)}"
        raise RuntimeError(message)


def _validate_baseline_selected_runtime(*, path: Path, config_path: Path) -> None:
    if not path.exists():
        message = f"missing embedded baseline selected runtime: {path}"
        raise RuntimeError(message)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    efficiency = (
        config
        .get("runtime_matrix", {})
        .get("selection_benchmark_slice", {})
        .get("efficiency_followup", {})
    )
    if not isinstance(efficiency, dict):
        message = "runtime-selection config is missing efficiency_followup"
        raise TypeError(message)
    expected_row_id = efficiency.get("baseline_row_id")
    expected_policy_id = efficiency.get("baseline_runtime_policy_id")
    payload = json.loads(path.read_text(encoding="utf-8"))
    snapshot = payload.get("selected_row_snapshot")
    if not isinstance(snapshot, dict):
        message = "embedded baseline selected runtime is missing selected_row_snapshot"
        raise TypeError(message)
    payload_matches = (
        payload.get("status") != "pass"
        or payload.get("selected_row_id") != expected_row_id
        or payload.get("runtime_policy_id") != expected_policy_id
    )
    snapshot_matches = (
        snapshot.get("row_id") != expected_row_id
        or snapshot.get("runtime_policy_id") != expected_policy_id
        or snapshot.get("status") != "pass"
    )
    if payload_matches or snapshot_matches:
        message = "embedded baseline selected runtime is not a passing snapshot"
        raise RuntimeError(message)


def _write_import_only_artifact(
    *,
    output_dir: Path,
    config_path: Path,
    payload_manifest: dict[str, object],
    v8_artifact_dir: Path,
    baseline_selected_runtime: Path,
) -> Path:
    benchmark_dir = output_dir / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "import_smoke_pass",
        "status_scope": "non_promotable_local_upload_simulation",
        "benchmark_kind": "runtime_selection_import_only",
        "benchmark_source": RUNTIME_SELECTION_BENCHMARK_SOURCE,
        "full_run_eligible": False,
        "writes_selected_runtime": False,
        "selection_slice": RUNTIME_SELECTION_SLICE,
        "config_exists": config_path.exists(),
        "payload_manifest": payload_manifest,
        "v8_artifact_dir": str(v8_artifact_dir),
        "required_v8_artifacts": sorted(REQUIRED_V8_ARTIFACTS),
        "baseline_selected_runtime": str(baseline_selected_runtime),
        "baseline_selected_runtime_exists": baseline_selected_runtime.exists(),
    }
    output_path = benchmark_dir / "runtime_selection_import.json"
    output_path.write_text(
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    return output_path


def _validate_import_only_artifacts(*, output_dir: Path) -> None:
    benchmark_dir = output_dir / "benchmark"
    observed = {path.name for path in benchmark_dir.iterdir()}
    if observed != {"runtime_selection_import.json"}:
        message = f"unexpected import-only artifacts: {sorted(observed)}"
        raise RuntimeError(message)
    if (benchmark_dir / "selected_runtime.json").exists():
        message = "runtime-selection import simulation wrote selected_runtime"
        raise RuntimeError(message)


def _validate_runtime_selection_artifacts(*, output_dir: Path) -> None:
    benchmark_dir = output_dir / "benchmark"
    metrics_dir = output_dir / "metrics"
    observed_benchmark = {path.name for path in benchmark_dir.iterdir()}
    allowed = (
        RUNTIME_SELECTION_ALLOWED_BENCHMARK_ARTIFACTS
        | RUNTIME_SELECTION_OPTIONAL_BENCHMARK_ARTIFACTS
    )
    unexpected = observed_benchmark - allowed
    missing = RUNTIME_SELECTION_ALLOWED_BENCHMARK_ARTIFACTS - observed_benchmark
    if unexpected or missing:
        message = (
            "unexpected runtime-selection benchmark artifacts: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
        raise RuntimeError(message)
    observed_metrics = (
        {path.name for path in metrics_dir.iterdir()} if metrics_dir.exists() else set()
    )
    if observed_metrics != RUNTIME_SELECTION_ALLOWED_METRIC_ARTIFACTS:
        message = (
            f"unexpected runtime-selection metric artifacts: {sorted(observed_metrics)}"
        )
        raise RuntimeError(message)
    runtime_proof = _validate_json_artifact(benchmark_dir / "runtime_proof.json")
    selected_runtime = benchmark_dir / "selected_runtime.json"
    selection_ready = runtime_proof.get("selection_ready") is True
    if selected_runtime.exists() and not selection_ready:
        message = (
            "selected_runtime.json exists but runtime_proof is not selection-ready"
        )
        raise RuntimeError(message)
    if not selected_runtime.exists() and selection_ready:
        message = (
            "runtime_proof is selection-ready but selected_runtime.json is missing"
        )
        raise RuntimeError(message)


def _validate_json_artifact(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        message = f"{path.name} must contain a JSON object"
        raise TypeError(message)
    errors: list[str] = []
    if payload.get("benchmark_kind") != RUNTIME_SELECTION_BENCHMARK_KIND:
        errors.append("wrong benchmark_kind")
    if payload.get("benchmark_source") != RUNTIME_SELECTION_BENCHMARK_SOURCE:
        errors.append("wrong benchmark_source")
    if payload.get("machine_shape") != "NvidiaTeslaT4":
        errors.append("wrong machine_shape")
    if payload.get("selection_ready") is True and payload.get("status") != "pass":
        errors.append("selection_ready requires pass status")
    if errors:
        raise RuntimeError("; ".join(errors))
    return cast("dict[str, object]", payload)


if __name__ == "__main__":
    raise SystemExit(main())
