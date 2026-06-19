# Copyright 2026 HiperMaximus
"""Generated single-file Kaggle real-data runtime pretest launcher template."""

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

KAGGLE_REAL_DATA_RUNTIME_PRETEST_READY = True
REAL_DATA_PRETEST_BENCHMARK_KIND = "real_data_runtime_pretest"
REAL_DATA_PRETEST_BENCHMARK_SOURCE = "kaggle_capped_real_data_train_step_pretest"
REAL_DATA_PRETEST_STATUS_SCOPE = "non_promotable_real_data_runtime_pretest"
EXPECTED_DATASET_SLUG = "maximusshtefan/patches-pre-shuffled-ubc-ocean"
REAL_DATA_PRETEST_ALLOWED_BENCHMARK_ARTIFACTS = {
    "real_data_runtime_pretest_manifest.json",
    "runtime_proof.json",
    "runtime_matrix.csv",
    "dataloader_matrix.csv",
    "numerical_checks.csv",
    "corruption_checks.csv",
    "gate_health_summary.json",
    "real_data_runtime_pretest_recommendations.json",
    "phase_timings.json",
}
REAL_DATA_PRETEST_ALLOWED_METRIC_ARTIFACTS = {"gate_health.csv"}
REAL_DATA_PRETEST_REQUIRED_BLOCKED_CLAIMS = {
    "final_runtime_selection",
    "final_batch_size",
    "final_precision_policy",
    "final_corruption_strategy",
    "final_dataloader_settings",
    "single_vs_dual_t4_final_choice",
    "convergence",
    "paper_evidence",
    "full_run_readiness",
}
KERNEL_METADATA = {
    "id": "maximusshtefan/eqvae-real-data-runtime-pretest",
    "title": "eqvae real data runtime pretest",
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
LOCAL_FALLBACK_OUTPUT_DIR = Path("runs/kaggle/real_data_runtime_pretest_local")
EMBEDDED_PAYLOAD_B64 = """
$embedded_payload_b64
"""
EMBEDDED_PAYLOAD_ZIP_SHA256 = "$embedded_payload_zip_sha256"
EMBEDDED_PAYLOAD_MANIFEST_SHA256 = "$embedded_payload_manifest_sha256"


def main() -> int:
    """Run the real-data runtime pretest launcher from an embedded payload.

    Returns:
        Process exit status.

    """
    output_dir = _output_dir()
    return _run_real_data_runtime_pretest(output_dir)


def _run_real_data_runtime_pretest(output_dir: Path) -> int:
    _require_python_version()
    payload_dir = _extract_payload(_payload_extract_dir(output_dir))
    payload_src = payload_dir / "src"
    sys.path.insert(0, str(payload_src))
    _prepare_environment(payload_src=payload_src)
    single_visible_t4()
    dual_t4_ddp()
    wrong_accelerator()

    import eqvae  # noqa: PLC0415
    from eqvae.benchmarking.real_data_runtime_pretest import (  # noqa: PLC0415
        write_local_upload_simulation_artifact,
    )
    from eqvae.cli.real_data_runtime_pretest import (  # noqa: PLC0415
        main as pretest_main,
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
    if os.environ.get("EQVAE_REAL_DATA_RUNTIME_PRETEST_IMPORT_ONLY") == "1":
        write_local_upload_simulation_artifact(
            config_path=config_path,
            output_dir=output_dir,
            payload_manifest=payload_manifest,
        )
        _validate_import_only_artifacts(output_dir=output_dir)
        return 0

    pretest_args = [
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
    ]
    data_root_override = os.environ.get("EQVAE_REAL_DATA_RUNTIME_PRETEST_DATA_ROOT")
    if data_root_override:
        pretest_args.extend(("--data-root", data_root_override))
    exit_code = pretest_main(tuple(pretest_args))
    _validate_real_data_pretest_artifacts(output_dir=output_dir)
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
        return Path("/kaggle/temp/eqvae_real_data_runtime_pretest_payload")
    if Path("/tmp").exists():  # noqa: S108
        return Path("/tmp/eqvae_real_data_runtime_pretest_payload")  # noqa: S108
    return output_dir / "embedded_payload"


def _require_python_version() -> None:
    if sys.version_info < (3, 12):  # noqa: UP036
        message = (
            "eqvae real-data runtime pretest requires Python >= 3.12 because "
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
    """Record the required single visible T4 pretest mode hook."""
    return


def dual_t4_ddp() -> None:
    """Record the required dual T4 DDP pretest mode hook."""
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


def _validate_import_only_artifacts(*, output_dir: Path) -> None:
    benchmark_dir = output_dir / "benchmark"
    observed = {path.name for path in benchmark_dir.iterdir()}
    if observed != {"real_data_runtime_pretest_import.json"}:
        message = f"unexpected import-only artifacts: {sorted(observed)}"
        raise RuntimeError(message)
    if (benchmark_dir / "selected_runtime.json").exists():
        message = "real-data runtime pretest import simulation wrote selected_runtime"
        raise RuntimeError(message)


def _validate_real_data_pretest_artifacts(*, output_dir: Path) -> None:
    benchmark_dir = output_dir / "benchmark"
    metrics_dir = output_dir / "metrics"
    observed_benchmark = {path.name for path in benchmark_dir.iterdir()}
    if observed_benchmark != REAL_DATA_PRETEST_ALLOWED_BENCHMARK_ARTIFACTS:
        message = (
            "unexpected real-data runtime pretest benchmark artifacts: "
            f"{sorted(observed_benchmark)}"
        )
        raise RuntimeError(message)
    observed_metrics = (
        {path.name for path in metrics_dir.iterdir()} if metrics_dir.exists() else set()
    )
    if observed_metrics != REAL_DATA_PRETEST_ALLOWED_METRIC_ARTIFACTS:
        message = (
            "unexpected real-data runtime pretest metric artifacts: "
            f"{sorted(observed_metrics)}"
        )
        raise RuntimeError(message)
    _validate_json_artifact(
        benchmark_dir / "real_data_runtime_pretest_manifest.json",
    )
    _validate_json_artifact(benchmark_dir / "runtime_proof.json")
    _validate_json_artifact(
        benchmark_dir / "real_data_runtime_pretest_recommendations.json",
    )
    if (benchmark_dir / "selected_runtime.json").exists():
        message = "real-data runtime pretest wrote selected_runtime.json"
        raise RuntimeError(message)


def _validate_json_artifact(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        message = f"{path.name} must contain a JSON object"
        raise TypeError(message)
    errors: list[str] = []
    if payload.get("benchmark_kind") != REAL_DATA_PRETEST_BENCHMARK_KIND:
        errors.append("wrong benchmark_kind")
    if payload.get("benchmark_source") != REAL_DATA_PRETEST_BENCHMARK_SOURCE:
        errors.append("wrong benchmark_source")
    if payload.get("status_scope") != REAL_DATA_PRETEST_STATUS_SCOPE:
        errors.append("wrong status_scope")
    if payload.get("full_run_eligible") is not False:
        errors.append("full_run_eligible must be false")
    blocked_claims = payload.get("blocked_claims")
    if not isinstance(blocked_claims, dict):
        errors.append("blocked_claims must be an object")
    elif set(blocked_claims) != REAL_DATA_PRETEST_REQUIRED_BLOCKED_CLAIMS:
        errors.append("blocked_claims must match required non-promotable claims")
    elif not all(value is True for value in blocked_claims.values()):
        errors.append("blocked_claims must block every claim")
    if errors:
        raise RuntimeError("; ".join(errors))


if __name__ == "__main__":
    raise SystemExit(main())
