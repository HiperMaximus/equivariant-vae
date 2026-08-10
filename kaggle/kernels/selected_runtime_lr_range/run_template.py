# Copyright 2026 HiperMaximus
"""Generated single-file Kaggle learning-rate range template."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import math
import os
import shutil
import subprocess  # noqa: S404
import sys
import traceback
import zipfile
from pathlib import Path

KAGGLE_SELECTED_RUNTIME_LR_RANGE_READY = True
EXPECTED_DATASET_SLUG = "maximusshtefan/patches-pre-shuffled-ubc-ocean"
KERNEL_METADATA = {
    "id": "maximusshtefan/eqvae-selected-runtime-lr-range",
    "title": "eqvae selected runtime lr range",
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
LOCAL_FALLBACK_OUTPUT_DIR = Path("runs/kaggle/selected_runtime_lr_range_local")
RUNTIME_CONFIG = Path("configs/spec0001/non_eq_vae_selected_runtime.json")
LR_RANGE_CONFIG = Path(
    "configs/spec0001/non_eq_vae_selected_runtime_lr_range.json",
)
EXPECTED_UPDATES = 192
EXPECTED_WORLD_SIZE = 2
EXPECTED_START_LR = 2e-5
EXPECTED_END_LR = 3e-3
EMBEDDED_PAYLOAD_B64 = """
$embedded_payload_b64
"""
EMBEDDED_PAYLOAD_ZIP_SHA256 = "$embedded_payload_zip_sha256"
EMBEDDED_PAYLOAD_MANIFEST_SHA256 = "$embedded_payload_manifest_sha256"


def main() -> int:
    """Run one bounded real-data LR sweep on the measured dual-T4 recipe.

    Returns:
        Process exit status.

    """
    output_dir = _output_dir()
    try:
        return _run(output_dir)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 1


def _run(output_dir: Path) -> int:
    _require_python_version()
    _ensure_latest_torch()
    payload_dir = _extract_payload(_payload_extract_dir(output_dir))
    payload_src = payload_dir / "src"
    _prepare_environment(payload_src)
    _validate_dual_t4()
    completed = subprocess.run(  # noqa: S603
        (
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            "-m",
            "eqvae.cli.selected_runtime_train",
            "--config",
            str(payload_dir / LR_RANGE_CONFIG),
            "--runtime-config",
            str(payload_dir / RUNTIME_CONFIG),
            "--data",
            "ubc-pre-shuffled",
            "--data-root",
            "auto",
            "--output-dir",
            str(output_dir),
            "--run-name",
            "non_eq_vae_spec0001_selected_runtime_lr_range",
        ),
        env=os.environ.copy(),
        check=False,
    )
    if completed.returncode != 0:
        return int(completed.returncode)
    _validate_outputs(output_dir)
    return 0


def _output_dir() -> Path:
    configured = os.environ.get("EQVAE_OUTPUT_DIR")
    output_dir = (
        Path(configured)
        if configured
        else (
            DEFAULT_KAGGLE_OUTPUT_DIR
            if DEFAULT_KAGGLE_OUTPUT_DIR.exists()
            else LOCAL_FALLBACK_OUTPUT_DIR
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir.resolve()


def _payload_extract_dir(output_dir: Path) -> Path:
    if Path("/kaggle/temp").exists():
        return Path("/kaggle/temp/eqvae_selected_runtime_lr_range_payload")
    if Path("/tmp").exists():  # noqa: S108
        return Path("/tmp/eqvae_selected_runtime_lr_range_payload")  # noqa: S108
    return output_dir / "embedded_payload"


def _require_python_version() -> None:
    if sys.version_info < (3, 12):  # noqa: UP036
        message = "eqvae learning-rate range requires Python >= 3.12"
        raise RuntimeError(message)


def _ensure_latest_torch() -> None:
    if os.environ.get("EQVAE_SKIP_TORCH_UPGRADE") == "1":
        return
    if not Path("/kaggle/working").exists():
        return
    subprocess.check_call(
        (
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "torch",
            "torchvision",
            "torchaudio",
        ),
    )


def _extract_payload(destination: Path) -> Path:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    zip_bytes = base64.b64decode(EMBEDDED_PAYLOAD_B64.encode("ascii"))
    if hashlib.sha256(zip_bytes).hexdigest() != EMBEDDED_PAYLOAD_ZIP_SHA256:
        message = "embedded payload zip SHA-256 mismatch"
        raise RuntimeError(message)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        for member in archive.infolist():
            path = Path(member.filename)
            if path.is_absolute() or ".." in path.parts:
                message = f"unsafe embedded payload path: {member.filename}"
                raise RuntimeError(message)
        archive.extractall(destination)
    manifest = destination / "payload_manifest.json"
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != (
        EMBEDDED_PAYLOAD_MANIFEST_SHA256
    ):
        message = "embedded payload manifest SHA-256 mismatch"
        raise RuntimeError(message)
    return destination


def _prepare_environment(payload_src: Path) -> None:
    os.environ.pop("EQVAE_DATA_ROOT", None)
    existing = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(payload_src), *([existing] if existing else [])],
    )


def _validate_dual_t4() -> None:
    import torch  # noqa: PLC0415

    names = [
        torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
    ]
    if len(names) != EXPECTED_WORLD_SIZE or any(
        "T4" not in name.upper() for name in names
    ):
        message = (
            f"learning-rate range requires exactly two visible T4 GPUs, got {names}"
        )
        raise RuntimeError(message)


def _validate_outputs(output_dir: Path) -> None:
    summary_path = output_dir / "benchmark" / "learning_rate_range_summary.json"
    rows_path = output_dir / "metrics" / "train_steps.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict) or summary.get("status") != "local_pass":
        message = "learning-rate range summary did not pass"
        raise RuntimeError(message)
    if summary.get("successful_updates") != EXPECTED_UPDATES:
        message = "learning-rate range did not complete 192 updates"
        raise RuntimeError(message)
    recommended = float(summary.get("recommended_learning_rate", 0.0))
    if not EXPECTED_START_LR <= recommended <= EXPECTED_END_LR:
        message = "recommended learning rate is outside the configured range"
        raise RuntimeError(message)
    with rows_path.open(encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    successful = [row for row in rows if row.get("amp_step_skipped") == "0"]
    if len(successful) != EXPECTED_UPDATES * 2:
        message = "learning-rate range train-step row coverage mismatch"
        raise RuntimeError(message)
    if sorted({row.get("rank") for row in successful}) != ["0", "1"]:
        message = "learning-rate range rank coverage mismatch"
        raise RuntimeError(message)
    learning_rates = [float(row["learning_rate"]) for row in successful]
    if not math.isclose(min(learning_rates), EXPECTED_START_LR, rel_tol=1e-6):
        message = "learning-rate range start endpoint mismatch"
        raise RuntimeError(message)
    if not math.isclose(max(learning_rates), EXPECTED_END_LR, rel_tol=1e-6):
        message = "learning-rate range end endpoint mismatch"
        raise RuntimeError(message)


if __name__ == "__main__":
    raise SystemExit(main())
