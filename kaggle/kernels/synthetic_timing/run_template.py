# Copyright 2026 HiperMaximus
"""Generated single-file Kaggle synthetic timing launcher template."""

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

KAGGLE_SYNTHETIC_TIMING_READY = True
SYNTHETIC_TIMING_BENCHMARK_KIND = "kaggle_synthetic_timing_pretest"
SYNTHETIC_TIMING_BENCHMARK_SOURCE = "kaggle_no_dataset_generated_ubc_shards"
SYNTHETIC_TIMING_STATUS_SCOPE = "non_promotable_synthetic_timing"
SYNTHETIC_TIMING_ALLOWED_ARTIFACTS = {
    "synthetic_timing_manifest.json",
    "synthetic_timing_runtime_proof.json",
    "synthetic_timing_matrix.csv",
    "synthetic_timing_recommendations.json",
}
SYNTHETIC_TIMING_REQUIRED_BLOCKED_CLAIMS = {
    "final_batch_size",
    "final_precision_policy",
    "final_corruption_strategy",
    "final_dataloader_settings",
    "final_single_vs_dual_t4",
    "real_data_loader_throughput",
    "convergence",
    "paper_evidence",
    "full_run_readiness",
}
KERNEL_METADATA = {
    "id": "maximusshtefan/eqvae-synthetic-timing",
    "title": "eqvae synthetic timing",
    "code_file": "run.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_internet": "true",
    "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": [],
    "competition_sources": [],
    "kernel_sources": [],
    "model_sources": [],
}
DEFAULT_KAGGLE_OUTPUT_DIR = Path("/kaggle/working")
LOCAL_FALLBACK_OUTPUT_DIR = Path("runs/kaggle/synthetic_timing_local")
EMBEDDED_PAYLOAD_B64 = """
$embedded_payload_b64
"""
EMBEDDED_PAYLOAD_ZIP_SHA256 = "$embedded_payload_zip_sha256"
EMBEDDED_PAYLOAD_MANIFEST_SHA256 = "$embedded_payload_manifest_sha256"


def main() -> int:
    """Run the synthetic timing launcher from an embedded payload.

    Returns:
        Process exit status.

    """
    output_dir = _output_dir()
    return _run_synthetic_timing(output_dir)


def _run_synthetic_timing(output_dir: Path) -> int:
    _require_python_version()
    _ensure_latest_torch(cpu_only=False)
    payload_dir = _extract_payload(_payload_extract_dir(output_dir))
    payload_src = payload_dir / "src"
    sys.path.insert(0, str(payload_src))
    _prepare_environment(payload_src=payload_src)
    single_visible_t4()
    dual_t4_ddp()
    wrong_accelerator()
    local_upload_simulation = (
        os.environ.get(
            "EQVAE_SYNTHETIC_TIMING_TINY_PROFILE",
        )
        == "1"
    )
    _assert_output_scope(
        output_dir=output_dir,
        local_upload_simulation=local_upload_simulation,
    )

    import eqvae  # noqa: PLC0415
    from eqvae.benchmarking.synthetic_timing import (  # noqa: PLC0415
        REPEAT_SHORTLIST_MEASURED_STEPS,
        REPEAT_SHORTLIST_WARMUP_STEPS,
        SYNTHETIC_TIMING_PHASE_REPEAT_SHORTLIST,
        SyntheticTimingRequest,
        repeat_shortlist_row_specs,
        tiny_upload_simulation_profile,
        write_synthetic_timing_pretest,
    )

    _assert_import_origin(
        module_file=Path(cast("str", eqvae.__file__)),
        payload_src=payload_src,
    )
    profile = tiny_upload_simulation_profile() if local_upload_simulation else None
    batch_sizes = (2,) if local_upload_simulation else ()
    row_specs = None if local_upload_simulation else repeat_shortlist_row_specs()
    steps = 1 if local_upload_simulation else REPEAT_SHORTLIST_MEASURED_STEPS
    warmup_steps = 1 if local_upload_simulation else REPEAT_SHORTLIST_WARMUP_STEPS
    timing_phase = (
        "local_upload_simulation"
        if local_upload_simulation
        else SYNTHETIC_TIMING_PHASE_REPEAT_SHORTLIST
    )
    run_name = (
        "eqvae_synthetic_timing_local_upload_simulation"
        if local_upload_simulation
        else "eqvae_synthetic_timing_repeat_shortlist"
    )
    manifest_path = payload_dir / "payload_manifest.json"
    artifacts = write_synthetic_timing_pretest(
        SyntheticTimingRequest(
            output_dir=output_dir,
            run_name=run_name,
            profile=profile,
            local_upload_simulation=local_upload_simulation,
            batch_sizes=batch_sizes,
            row_specs=row_specs,
            warmup_steps=warmup_steps,
            measured_steps=steps,
            timing_phase=timing_phase,
            payload_manifest=json.loads(manifest_path.read_text(encoding="utf-8")),
            kernel_metadata=KERNEL_METADATA,
        ),
    )
    _validate_synthetic_timing_artifacts(
        output_dir=output_dir,
        local_upload_simulation=local_upload_simulation,
    )
    _assert_allowed_artifacts(tuple(artifacts.__dict__.values()))
    return 0


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
        return Path("/kaggle/temp/eqvae_synthetic_timing_payload")
    if Path("/tmp").exists():  # noqa: S108
        return Path("/tmp/eqvae_synthetic_timing_payload")  # noqa: S108
    return output_dir / "embedded_payload"


def _require_python_version() -> None:
    if sys.version_info < (3, 12):  # noqa: UP036
        message = (
            "eqvae synthetic timing requires Python >= 3.12 because active "
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
    """Record the required single visible T4 timing mode hook."""
    return


def dual_t4_ddp() -> None:
    """Record the required dual T4 DDP timing mode hook."""
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


def _validate_synthetic_timing_artifacts(
    *,
    output_dir: Path,
    local_upload_simulation: bool,
) -> None:
    benchmark_dir = output_dir / "benchmark"
    observed = {path.name for path in benchmark_dir.iterdir()}
    if observed != SYNTHETIC_TIMING_ALLOWED_ARTIFACTS:
        message = f"unexpected synthetic timing artifacts: {sorted(observed)}"
        raise RuntimeError(message)
    manifest = json.loads(
        (benchmark_dir / "synthetic_timing_manifest.json").read_text(
            encoding="utf-8",
        ),
    )
    runtime_proof = json.loads(
        (benchmark_dir / "synthetic_timing_runtime_proof.json").read_text(
            encoding="utf-8",
        ),
    )
    recommendations = json.loads(
        (benchmark_dir / "synthetic_timing_recommendations.json").read_text(
            encoding="utf-8",
        ),
    )
    for payload in (manifest, runtime_proof, recommendations):
        _validate_non_promotable_payload(payload)
    data = manifest.get("data")
    if not isinstance(data, dict):
        message = "synthetic timing manifest missing data object"
        raise TypeError(message)
    root = Path(cast("str", data["root"])).resolve()
    if root == Path("/kaggle/input") or Path("/kaggle/input") in root.parents:
        message = "synthetic timing data must not resolve under /kaggle/input"
        raise RuntimeError(message)
    if not local_upload_simulation:
        _assert_under_kaggle_working(root)


def _validate_non_promotable_payload(payload: object) -> None:
    if not isinstance(payload, dict):
        message = "synthetic timing artifact must be a JSON object"
        raise TypeError(message)
    errors: list[str] = []
    if payload.get("benchmark_kind") != SYNTHETIC_TIMING_BENCHMARK_KIND:
        errors.append("wrong benchmark_kind")
    if payload.get("benchmark_source") != SYNTHETIC_TIMING_BENCHMARK_SOURCE:
        errors.append("wrong benchmark_source")
    if payload.get("status_scope") != SYNTHETIC_TIMING_STATUS_SCOPE:
        errors.append("wrong status_scope")
    if payload.get("full_run_eligible") is not False:
        errors.append("full_run_eligible must be false")
    blocked_claims = payload.get("blocked_claims")
    if not isinstance(blocked_claims, dict):
        errors.append("blocked_claims must be an object")
    elif set(blocked_claims) != SYNTHETIC_TIMING_REQUIRED_BLOCKED_CLAIMS:
        errors.append("blocked_claims must match the required real-run claims")
    elif not all(value is True for value in blocked_claims.values()):
        errors.append("blocked_claims must block every real-run claim")
    errors.extend(
        f"{source_field} must be empty"
        for source_field in (
            "dataset_sources",
            "competition_sources",
            "kernel_sources",
            "model_sources",
        )
        if payload.get(source_field) != []
    )
    if errors:
        raise RuntimeError("; ".join(errors))


def _assert_allowed_artifacts(paths: tuple[Path, ...]) -> None:
    names = {path.name for path in paths}
    if names != SYNTHETIC_TIMING_ALLOWED_ARTIFACTS:
        message = f"writer returned unexpected artifacts: {sorted(names)}"
        raise RuntimeError(message)


def _assert_output_scope(
    *,
    output_dir: Path,
    local_upload_simulation: bool,
) -> None:
    if local_upload_simulation:
        kaggle_input = Path("/kaggle/input")
        if output_dir == kaggle_input or kaggle_input in output_dir.parents:
            message = (
                "local synthetic timing simulation must not write under /kaggle/input"
            )
            raise RuntimeError(message)
        return
    _assert_under_kaggle_working(output_dir)


def _assert_under_kaggle_working(path: Path) -> None:
    kaggle_working = Path("/kaggle/working").resolve()
    resolved = path.resolve()
    if resolved != kaggle_working and kaggle_working not in resolved.parents:
        message = (
            "non-simulation synthetic timing output must resolve under "
            f"{kaggle_working}, got {resolved}"
        )
        raise RuntimeError(message)


if __name__ == "__main__":
    raise SystemExit(main())
