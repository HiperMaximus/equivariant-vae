# Copyright 2026 HiperMaximus
# ruff: noqa: BLE001, D103, EM101, EM102, PERF401, PLC0415, PLW0717, S404, S603, S607, TRY003, TRY300
"""Generated one-off wrapper for the Spec 0022 paired cancer top-up."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path
from typing import cast

KAGGLE_UBC_OCEAN_CANCER_TOPUP_READY = True
EMBEDDED_PAYLOAD_B64 = """
$embedded_payload_b64
"""
EMBEDDED_PAYLOAD_SHA256 = "$embedded_payload_sha256"
EMBEDDED_CONFIG_B64 = "$embedded_config_b64"
EMBEDDED_CONFIG_SHA256 = "$embedded_config_sha256"
WORKING_ROOT = Path("/kaggle/working")
INPUT_ROOT = Path("/kaggle/input")
PRIVATE_ROOT = WORKING_ROOT / ".spec0022_private"
SCRATCH_ROOT = WORKING_ROOT / ".spec0022_topup_work"
OUTPUT_ROOT = WORKING_ROOT / "dataset"
PYVIPS_VERSION = "3.1.0"
OUTPUT_ALLOWLIST = frozenset({
    "normal_vae_mu_cancer_topup.bin",
    "normal_vae_mu_cancer_topup.json",
    "so2_vae_mu_cancer_topup.bin",
    "so2_vae_mu_cancer_topup.json",
    "spec0022_cancer_topup_pair_audit.json",
})


def main() -> int:
    try:
        os.environ["EQVAE_SESSION_START_MONOTONIC"] = str(time.monotonic())
        _ensure_latest_torch()
        _install_dependencies()
        PRIVATE_ROOT.mkdir(parents=False, exist_ok=False)
        source_root = _extract_payload(PRIVATE_ROOT / "payload")
        sys.path.insert(0, str(source_root / "src"))
        config = _embedded_config()
        mounted_input = _resolve_input(config)
        wsi_dir = _resolve_wsi_dir()
        os.environ["EQVAE_CANCER_TOPUP_CONFIRMED"] = "1"
        from eqvae.cli.generate_ubc_cancer_topup import main as topup_main

        return_code = topup_main([
            "--input-root",
            str(mounted_input),
            "--wsi-dir",
            str(wsi_dir),
            "--output-root",
            str(OUTPUT_ROOT),
            "--scratch-root",
            str(SCRATCH_ROOT),
        ])
        if return_code != 0:
            return return_code
        shutil.rmtree(PRIVATE_ROOT, ignore_errors=True)
        shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)
        _assert_publishable_output()
        return 0
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        os.environ.pop("EQVAE_CANCER_TOPUP_CONFIRMED", None)
        os.environ.pop("EQVAE_SESSION_START_MONOTONIC", None)
        shutil.rmtree(PRIVATE_ROOT, ignore_errors=True)
        shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)


def _ensure_latest_torch() -> None:
    if not WORKING_ROOT.exists() or os.environ.get("EQVAE_SKIP_REMOTE_SETUP") == "1":
        return
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "torch",
        "torchvision",
        "torchaudio",
    ])


def _install_dependencies() -> None:
    if not WORKING_ROOT.exists() or os.environ.get("EQVAE_SKIP_REMOTE_SETUP") == "1":
        return
    subprocess.check_call(["apt-get", "update", "-qq"], stdout=subprocess.DEVNULL)
    subprocess.check_call([
        "apt-get",
        "install",
        "-y",
        "--no-install-recommends",
        "libvips",
    ])
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        f"pyvips=={PYVIPS_VERSION}",
    ])


def _extract_payload(destination: Path) -> Path:
    payload = base64.b64decode(EMBEDDED_PAYLOAD_B64.encode("ascii"))
    if hashlib.sha256(payload).hexdigest() != EMBEDDED_PAYLOAD_SHA256:
        raise RuntimeError("Spec 0022 embedded source hash mismatch")
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name in archive.namelist():
            path = Path(name)
            if path.is_absolute() or ".." in path.parts:
                raise RuntimeError(f"Unsafe embedded path: {name}")
        archive.extractall(destination)
    return destination


def _embedded_config() -> dict[str, object]:
    encoded = base64.b64decode(EMBEDDED_CONFIG_B64.encode("ascii"))
    if hashlib.sha256(encoded).hexdigest() != EMBEDDED_CONFIG_SHA256:
        raise RuntimeError("Spec 0022 embedded config hash mismatch")
    value = cast("object", json.loads(encoded))
    if not isinstance(value, dict):
        raise TypeError("Spec 0022 embedded config must be an object")
    return cast("dict[str, object]", value)


def _resolve_input(config: dict[str, object]) -> Path:
    expected_hash = config.get("input_contract_sha256")
    matches: list[Path] = []
    for contract in INPUT_ROOT.rglob("spec0022_topup_inference_contract.json"):
        if hashlib.sha256(contract.read_bytes()).hexdigest() == expected_hash:
            matches.append(contract.parent)
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one exact Spec 0022 input mount, found {len(matches)}",
        )
    return matches[0]


def _resolve_wsi_dir() -> Path:
    candidates = (
        INPUT_ROOT / "UBC-OCEAN/train_images",
        INPUT_ROOT / "competitions/UBC-OCEAN/train_images",
    )
    matches = [path for path in candidates if path.is_dir()]
    if len(matches) != 1:
        raise RuntimeError("Expected exactly one official UBC-OCEAN train_images mount")
    return matches[0]


def _assert_publishable_output() -> None:
    observed_root = {path.name for path in WORKING_ROOT.iterdir()}
    if (
        observed_root != {"dataset"}
        or not OUTPUT_ROOT.is_dir()
        or OUTPUT_ROOT.is_symlink()
    ):
        raise RuntimeError("Kaggle working output must contain only dataset/")
    observed = {path.name for path in OUTPUT_ROOT.iterdir() if path.is_file()}
    if observed != OUTPUT_ALLOWLIST or any(
        not path.is_file() for path in OUTPUT_ROOT.iterdir()
    ):
        raise RuntimeError("Spec 0022 output allow-list mismatch")


if __name__ == "__main__":
    raise SystemExit(main())
