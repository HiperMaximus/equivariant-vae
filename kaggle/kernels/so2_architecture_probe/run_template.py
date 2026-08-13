# Copyright 2026 HiperMaximus
# ruff: noqa: EM101, TRY003
"""Generated launcher for the locked Spec 0013 dual-T4 mechanics follow-up."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import shutil
import subprocess  # noqa: S404
import sys
import zipfile
from pathlib import Path
from typing import cast

KAGGLE_SO2_ARCHITECTURE_PROBE_READY = True
PROBE_MODULE = "eqvae.benchmarking.so2_architecture_probe"
ARTIFACT_FILENAME = "spec0013_so2_dual_t4_probe.json"
SCHEMA_VERSION = "spec0013.so2_dual_t4_follow_up.v1"
PROBE_KIND = "locked_so2_architecture_mechanics_follow_up"
RUNTIME_BUNDLE_ID = "compile_step_python_reducer_fp16_channels_last"
SELECTED_RUNTIME_SHA256 = (
    "e9e998fd161f0955959c64aed7cd7ddbdfcb55a271b9ce05805903c97c93efb8"
)
SELECTED_RUNTIME_PATH = Path("configs/spec0001/non_eq_vae_selected_runtime.json")
SETTLED_UPDATES = 32
PROBE_TIMEOUT_SECONDS = 10800
DEFAULT_OUTPUT_DIR = Path("/kaggle/working")
EMBEDDED_PAYLOAD_B64 = """
$embedded_payload_b64
"""
EMBEDDED_PAYLOAD_ZIP_SHA256 = "$embedded_payload_zip_sha256"
EMBEDDED_PAYLOAD_MANIFEST_SHA256 = "$embedded_payload_manifest_sha256"


def main() -> int:
    """Launch the single fixed no-dataset dual-T4 transfer check.

    Returns:
        Process exit status.

    """
    _require_python_version()
    _ensure_latest_torch(cpu_only=False)
    output_dir = _output_dir()
    payload_dir = _extract_payload(_payload_extract_dir())
    payload_src = payload_dir / "src"
    _launch(payload_src=payload_src, output_dir=output_dir)
    _validate_artifact(output_dir=output_dir, payload_dir=payload_dir)
    return 0


def _launch(*, payload_src: Path, output_dir: Path) -> None:
    benchmark_dir = output_dir / "benchmark"
    command = (
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        "-m",
        PROBE_MODULE,
        "--output-dir",
        str(benchmark_dir),
    )
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(payload_src), *([existing] if existing else [])],
    )
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    environment["TORCH_LOGS"] = "graph_breaks,recompiles"
    subprocess.run(  # noqa: S603
        command,
        check=True,
        cwd=payload_src.parent,
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
        return Path("/kaggle/temp/eqvae_so2_architecture_probe_payload")
    return Path("/tmp/eqvae_so2_architecture_probe_payload")  # noqa: S108


def _require_python_version() -> None:
    if sys.version_info < (3, 12):  # noqa: UP036
        raise RuntimeError("Spec 0013 probe requires Python >= 3.12")


def _ensure_latest_torch(*, cpu_only: bool) -> None:
    """Install the latest Torch stack before importing the embedded package."""
    if os.environ.get("EQVAE_SKIP_TORCH_UPGRADE") == "1":
        return
    if not Path("/kaggle/working").exists():
        return
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
    if hashlib.sha256(zip_bytes).hexdigest() != EMBEDDED_PAYLOAD_ZIP_SHA256:
        raise RuntimeError("embedded payload zip SHA-256 mismatch")
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        for member in archive.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                message = f"unsafe embedded payload path: {member.filename}"
                raise RuntimeError(message)
        archive.extractall(destination)
    manifest_path = destination / "payload_manifest.json"
    if (
        hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        != EMBEDDED_PAYLOAD_MANIFEST_SHA256
    ):
        raise RuntimeError("embedded payload manifest SHA-256 mismatch")
    return destination


def _validate_artifact(  # noqa: C901, PLR0912
    *,
    output_dir: Path,
    payload_dir: Path,
) -> None:
    benchmark_dir = output_dir / "benchmark"
    observed = {path.name for path in benchmark_dir.iterdir()}
    if observed != {ARTIFACT_FILENAME}:
        message = f"unexpected Spec 0013 artifacts: {sorted(observed)}"
        raise RuntimeError(message)
    payload = cast(
        "dict[str, object]",
        json.loads((benchmark_dir / ARTIFACT_FILENAME).read_text(encoding="utf-8")),
    )
    expected = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_kind": PROBE_KIND,
        "status": "pass",
        "architecture_locked": True,
        "full_vae_assembled": False,
        "world_size": 2,
        "nproc_per_node": 2,
        "per_device_batch_size": 4,
    }
    errors = [
        f"{key} must be {value!r}"
        for key, value in expected.items()
        if payload.get(key) != value
    ]
    runtime = payload.get("runtime_requested_and_effective")
    if not isinstance(runtime, dict):
        errors.append("runtime_requested_and_effective must be an object")
    else:
        requested = runtime.get("requested")
        effective = runtime.get("effective")
        if (
            not isinstance(requested, dict)
            or requested.get("runtime_policy_id") != RUNTIME_BUNDLE_ID
        ):
            errors.append("runtime bundle must remain the selected Spec 0011 winner")
        if requested != effective:
            errors.append("effective runtime readback must match every requested field")
        selected_path = payload_dir / SELECTED_RUNTIME_PATH
        selected_hash = hashlib.sha256(selected_path.read_bytes()).hexdigest()
        if selected_hash != SELECTED_RUNTIME_SHA256:
            errors.append("bundled selected runtime does not match reviewed hash")
        if runtime.get("source_sha256") != selected_hash:
            errors.append("selected runtime source hash mismatch")
    updates = payload.get("compiled_ddp_updates")
    if (
        not isinstance(updates, dict)
        or updates.get("settled_updates") != SETTLED_UPDATES
    ):
        errors.append("compiled DDP probe must complete 32 settled updates")
    if payload.get("acceptance_failures") != []:
        errors.append("acceptance_failures must be empty")
    if payload.get("selected_arm") not in {
        "four_mm_three_cat",
        "four_mm_direct",
        "padded_bmm_direct",
    }:
        errors.append("follow-up must select one predeclared passing arm")
    if payload.get("follow_up_probe_permitted") is not False:
        errors.append("follow-up runner must not authorize another remote arm")
    if errors:
        raise RuntimeError("; ".join(errors))
    if Path("/kaggle/working").exists() and output_dir != DEFAULT_OUTPUT_DIR:
        raise RuntimeError("Kaggle artifact must remain under /kaggle/working")


if __name__ == "__main__":
    raise SystemExit(main())
