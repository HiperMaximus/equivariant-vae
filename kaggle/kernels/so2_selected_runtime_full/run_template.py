# Copyright 2026 HiperMaximus
# ruff: noqa: EM101, PLW0717, TRY003
"""Single-file SO2 selected-runtime full continuation kernel."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import subprocess  # noqa: S404
import sys
import traceback
import zipfile
from pathlib import Path

KAGGLE_SO2_SELECTED_RUNTIME_FULL_READY = True
EXPECTED_DATASET_SLUG = "maximusshtefan/patches-pre-shuffled-ubc-ocean"
RESUME_DATASET_SLUG = "maximusshtefan/eqvae-so2-session1-step9000"
RESUME_CHECKPOINT = Path(
    "/kaggle/input/eqvae-so2-session1-step9000/step_009000.pt",
)
RESUME_CHECKPOINT_SHA256 = (
    "1f53fe16aecf6382bf450cd0ac2be5db9fe2bbe6405dfcaa2c196cb40bca8e7d"
)
KERNEL_METADATA = {
    "id": "maximusshtefan/eqvae-so2-selected-runtime-full",
    "title": "eqvae so2 selected runtime full",
    "code_file": "run.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_internet": "true",
    "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": [EXPECTED_DATASET_SLUG, RESUME_DATASET_SLUG],
    "competition_sources": [],
    "kernel_sources": [],
    "model_sources": [],
}
OUTPUT = Path("/kaggle/working")
RUNTIME = Path("configs/spec0001/non_eq_vae_selected_runtime.json")
FULL = Path("configs/spec0016/so2_selected_runtime_full.json")
EMBEDDED_PAYLOAD_B64 = """
$embedded_payload_b64
"""
EMBEDDED_PAYLOAD_ZIP_SHA256 = "$embedded_payload_zip_sha256"
EMBEDDED_PAYLOAD_MANIFEST_SHA256 = "$embedded_payload_manifest_sha256"


def main() -> int:
    """Resume the SO2 full run from session 1's exact update-9000 commit.

    Returns:
        Process exit status.

    """
    output = Path(os.environ.get("EQVAE_SO2_FULL_OUTPUT_DIR", OUTPUT)).resolve()
    try:
        _ensure_latest_torch()
        payload = _extract(output / "embedded_payload")
        if os.environ.get("EQVAE_SO2_FULL_IMPORT_ONLY") == "1":
            _write_json(
                output / "benchmark/so2_full_import.json",
                {
                    "status": "pass",
                    "fresh_start": False,
                    "resume_checkpoint": str(RESUME_CHECKPOINT),
                    "resume_checkpoint_sha256": RESUME_CHECKPOINT_SHA256,
                },
            )
            return 0
        resume_checkpoint = _resume_checkpoint_path()
        _validate_resume_checkpoint(resume_checkpoint)
        source = payload / "src"
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            part for part in (str(source), environment.get("PYTHONPATH", "")) if part
        )
        environment["PYTHONUNBUFFERED"] = "1"
        data_root = os.environ.get("EQVAE_SO2_FULL_DATA_ROOT", "auto")
        command = (
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            "-m",
            "eqvae.cli.selected_runtime_train",
            "--config",
            str(payload / FULL),
            "--runtime-config",
            str(payload / RUNTIME),
            "--data",
            "ubc-pre-shuffled",
            "--data-root",
            data_root,
            "--output-dir",
            str(output),
            "--run-name",
            "so2_spec0016_selected_runtime_full",
            "--resume",
            str(resume_checkpoint),
        )
        completed = subprocess.run(  # noqa: S603
            command,
            cwd=payload,
            env=environment,
            check=False,
        )
        return int(completed.returncode)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 1


def _resume_checkpoint_path() -> Path:
    override = os.environ.get("EQVAE_SO2_FULL_RESUME")
    return Path(override).resolve() if override else RESUME_CHECKPOINT


def _validate_resume_checkpoint(path: Path) -> None:
    if not path.is_file():
        message = f"SO2 resume checkpoint missing: {path}"
        raise RuntimeError(message)
    observed = _sha256(path)
    if observed != RESUME_CHECKPOINT_SHA256:
        message = (
            "SO2 resume checkpoint SHA-256 mismatch: "
            f"expected {RESUME_CHECKPOINT_SHA256}, observed {observed}"
        )
        raise RuntimeError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ensure_latest_torch() -> None:
    if (
        not Path("/kaggle/working").exists()
        or os.environ.get("EQVAE_SKIP_TORCH_UPGRADE") == "1"
    ):
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


def _extract(destination: Path) -> Path:
    data = base64.b64decode(EMBEDDED_PAYLOAD_B64)
    if hashlib.sha256(data).hexdigest() != EMBEDDED_PAYLOAD_ZIP_SHA256:
        raise RuntimeError("embedded payload hash mismatch")
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        if any(
            Path(name).is_absolute() or ".." in Path(name).parts
            for name in archive.namelist()
        ):
            raise RuntimeError("unsafe embedded payload path")
        archive.extractall(destination)
    manifest = destination / "payload_manifest.json"
    if (
        hashlib.sha256(manifest.read_bytes()).hexdigest()
        != EMBEDDED_PAYLOAD_MANIFEST_SHA256
    ):
        raise RuntimeError("payload manifest hash mismatch")
    full = json.loads((destination / FULL).read_text(encoding="utf-8"))
    contract = full.get("selected_runtime_full") if isinstance(full, dict) else None
    if (
        not isinstance(contract, dict)
        or contract.get("fresh_start_required") is not True
    ):
        raise RuntimeError("SO2 full experiment config does not pin a fresh lineage")
    return destination


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
