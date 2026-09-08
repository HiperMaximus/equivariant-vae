# Copyright 2026 HiperMaximus
# ruff: noqa: BLE001, D103, EM101, EM102, PLC0415, PLW0717, S404, S603, S607, TRY003, TRY301
"""Generated diagnosis-free wrapper for Spec 0045 VAE test reconstruction."""

from __future__ import annotations

import base64
import hashlib
import io
import os
import shutil
import subprocess
import sys
import traceback
import zipfile
from pathlib import Path

# fmt: off
SPEC0045_VAE_TEST_RECONSTRUCTION_READY = True
EMBEDDED_PAYLOAD_B64 = """
$embedded_payload_b64
"""
EMBEDDED_PAYLOAD_ZIP_SHA256 = "$embedded_payload_zip_sha256"
EMBEDDED_PAYLOAD_MANIFEST_SHA256 = "$embedded_payload_manifest_sha256"
INPUT_CONTRACT_SHA256 = "$input_contract_sha256"
INPUT_DATASET_REFERENCE = "$input_dataset_reference"
# fmt: on
WORKING_ROOT = Path("/kaggle/working")
PRIVATE_ROOT = WORKING_ROOT / ".spec0045_private"
SCRATCH_ROOT = WORKING_ROOT / ".spec0045_scratch"
PYVIPS_VERSION = "3.1.0"
TORCH_VERSION = "2.14.0"
TORCH_INDEX = "https://download.pytorch.org/whl/cu130"


def main() -> int:
    try:
        _ensure_pinned_torch()
        _install_pyvips()
        PRIVATE_ROOT.mkdir(parents=False, exist_ok=False)
        payload_root = _extract_payload(PRIVATE_ROOT / "payload")
        payload_src = payload_root / "src"
        sys.path.insert(0, str(payload_src))
        os.environ["PYTHONPATH"] = os.pathsep.join(
            part
            for part in (str(payload_src), os.environ.get("PYTHONPATH", ""))
            if part
        )
        import eqvae
        from eqvae.evaluation.vae_test_runtime import main as evaluation_main

        if payload_src.resolve() not in Path(str(eqvae.__file__)).resolve().parents:
            raise RuntimeError("eqvae imported outside the embedded payload")
        return evaluation_main(
            [
                "--expected-input-contract-sha256",
                INPUT_CONTRACT_SHA256,
                "--expected-dataset-reference",
                INPUT_DATASET_REFERENCE,
            ],
        )
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        shutil.rmtree(PRIVATE_ROOT, ignore_errors=True)
        shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)


def _ensure_pinned_torch() -> None:
    if not WORKING_ROOT.exists() or os.environ.get("EQVAE_SKIP_REMOTE_SETUP") == "1":
        return
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--no-cache-dir",
            "--index-url",
            TORCH_INDEX,
            f"torch=={TORCH_VERSION}",
        ],
    )
    subprocess.check_call(
        [
            sys.executable,
            "-c",
            (
                "import torch; "
                "assert torch.__version__ == '2.14.0+cu130'; "
                "assert torch.version.cuda == '13.0'"
            ),
        ],
    )


def _install_pyvips() -> None:
    if not WORKING_ROOT.exists() or os.environ.get("EQVAE_SKIP_REMOTE_SETUP") == "1":
        return
    subprocess.check_call(["apt-get", "update", "-qq"], stdout=subprocess.DEVNULL)
    subprocess.check_call(
        ["apt-get", "install", "-y", "--no-install-recommends", "libvips"],
    )
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", f"pyvips=={PYVIPS_VERSION}"],
    )


def _extract_payload(destination: Path) -> Path:
    payload = base64.b64decode(EMBEDDED_PAYLOAD_B64.encode("ascii"))
    if hashlib.sha256(payload).hexdigest() != EMBEDDED_PAYLOAD_ZIP_SHA256:
        raise RuntimeError("embedded payload zip hash mismatch")
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name in archive.namelist():
            candidate = Path(name)
            if candidate.is_absolute() or ".." in candidate.parts:
                raise RuntimeError(f"unsafe embedded payload path: {name}")
        archive.extractall(destination)
    manifest = destination / "payload_manifest.json"
    if (
        hashlib.sha256(manifest.read_bytes()).hexdigest()
        != EMBEDDED_PAYLOAD_MANIFEST_SHA256
    ):
        raise RuntimeError("embedded payload manifest hash mismatch")
    return destination


if __name__ == "__main__":
    raise SystemExit(main())
