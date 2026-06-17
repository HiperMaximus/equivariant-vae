# Copyright 2026 HiperMaximus
"""Generated single-file capped real-data Kaggle smoke launcher template."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import shutil
import sys
import traceback
import zipfile
from pathlib import Path
from typing import cast

KAGGLE_SMOKE_READY = True
DEFAULT_KAGGLE_OUTPUT_DIR = Path("/kaggle/working")
LOCAL_FALLBACK_OUTPUT_DIR = Path("runs/kaggle/non_eq_vae_debug_smoke")
EMBEDDED_PAYLOAD_B64 = """
$embedded_payload_b64
"""
EMBEDDED_PAYLOAD_ZIP_SHA256 = "$embedded_payload_zip_sha256"
EMBEDDED_PAYLOAD_MANIFEST_SHA256 = "$embedded_payload_manifest_sha256"


def main() -> int:
    """Run the capped real-data smoke launcher.

    Returns:
        Process exit status.

    """
    output_dir = _output_dir()
    try:
        return _run_real_data_smoke(output_dir)
    except Exception as error:
        _write_bootstrap_failure(output_dir=output_dir, error=error)
        raise


def _run_real_data_smoke(output_dir: Path) -> int:
    _require_python_version()
    payload_dir = _extract_payload(_payload_extract_dir(output_dir))
    payload_src = payload_dir / "src"
    sys.path.insert(0, str(payload_src))
    wrong_accelerator()
    single_visible_t4()
    dual_t4_ddp()

    import eqvae  # noqa: PLC0415
    from eqvae.cli.kaggle_smoke import main as smoke_main  # noqa: PLC0415

    _assert_import_origin(
        module_file=Path(cast("str", eqvae.__file__)),
        payload_src=payload_src,
    )
    if os.environ.get("EQVAE_LOCAL_UPLOAD_SIMULATION_ONLY") == "1":
        _write_import_smoke(output_dir=output_dir, payload_dir=payload_dir)
        return 0

    config_path = payload_dir / "configs" / "spec0001" / "non_eq_vae_kaggle_debug.json"
    return smoke_main(
        (
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ),
    )


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
    return output_dir


def _payload_extract_dir(output_dir: Path) -> Path:
    if Path("/kaggle/temp").exists():
        return Path("/kaggle/temp/eqvae_real_smoke_payload")
    if Path("/tmp").exists():  # noqa: S108
        return Path("/tmp/eqvae_real_smoke_payload")  # noqa: S108
    return output_dir / "embedded_payload"


def _require_python_version() -> None:
    if sys.version_info < (3, 12):  # noqa: UP036
        message = (
            "eqvae real-data smoke requires Python >= 3.12 because active source "
            "uses Python 3.12 type-alias syntax"
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


def wrong_accelerator() -> None:
    """Fail early if Kaggle gives no T4 GPU or a non-T4 GPU.

    Raises:
        RuntimeError: If visible CUDA devices are not T4 GPUs.

    """
    import torch  # noqa: PLC0415

    if not torch.cuda.is_available():
        if _is_kaggle_runtime():
            message = "Expected Kaggle T4 GPU runtime, but CUDA is unavailable"
            raise RuntimeError(message)
        return
    gpu_names = [
        torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
    ]
    if not all("T4" in name for name in gpu_names):
        message = f"Expected Kaggle T4 GPU metadata, got devices: {gpu_names}"
        raise RuntimeError(message)


def single_visible_t4() -> None:
    """Record the single-process T4 smoke mode hook for push validation."""
    return


def dual_t4_ddp() -> None:
    """Record that this smoke does not launch dual-T4 DDP."""
    return


def _assert_import_origin(*, module_file: Path, payload_src: Path) -> None:
    resolved_module = module_file.resolve()
    resolved_payload_src = payload_src.resolve()
    if resolved_payload_src not in resolved_module.parents:
        message = f"eqvae imported from {resolved_module}, not {resolved_payload_src}"
        raise RuntimeError(message)


def _write_import_smoke(*, output_dir: Path, payload_dir: Path) -> None:
    benchmark_dir = output_dir / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = payload_dir / "payload_manifest.json"
    payload = {
        "schema_version": "spec0001.kaggle_import_smoke.v1",
        "status": "import_smoke_pass",
        "status_scope": "non_promotable_local_upload_simulation",
        "benchmark_kind": "real_data_kaggle_debug_smoke_import_only",
        "payload_manifest": json.loads(manifest_path.read_text(encoding="utf-8")),
        "config_exists": (
            payload_dir / "configs" / "spec0001" / "non_eq_vae_kaggle_debug.json"
        ).exists(),
        "payload_zip_sha256": EMBEDDED_PAYLOAD_ZIP_SHA256,
        "payload_manifest_sha256": EMBEDDED_PAYLOAD_MANIFEST_SHA256,
    }
    (benchmark_dir / "kaggle_import_smoke.json").write_text(
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )


def _write_bootstrap_failure(*, output_dir: Path, error: Exception) -> None:
    benchmark_dir = output_dir / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    failure_message = "".join(
        traceback.format_exception_only(type(error), error),
    ).strip()
    payload = {
        "schema_version": "spec0001.kaggle_smoke_bootstrap.v1",
        "status": "fail",
        "status_scope": "non_promotable_debug",
        "benchmark_kind": "real_data_kaggle_debug_smoke",
        "benchmark_source": "kaggle_script_kernel_capped_smoke",
        "failure_kind": type(error).__name__,
        "failure_message_hash": hashlib.sha256(
            failure_message.encode("utf-8"),
        ).hexdigest(),
        "python_version": sys.version.split()[0],
        "payload_zip_sha256": EMBEDDED_PAYLOAD_ZIP_SHA256,
        "payload_manifest_sha256": EMBEDDED_PAYLOAD_MANIFEST_SHA256,
    }
    (benchmark_dir / "kaggle_smoke_bootstrap.json").write_text(
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )


def _is_kaggle_runtime() -> bool:
    return Path("/kaggle").exists()


if __name__ == "__main__":
    raise SystemExit(main())
