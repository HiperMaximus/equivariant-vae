# Copyright 2026 HiperMaximus
# ruff: noqa: D103, EM101, EM102, PLC0415, PLW0717, S603, S607, TRY003, TRY300, TRY301
"""Generated single-file wrapper for Spec 0021 latent inference."""

from __future__ import annotations

import base64
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
from typing import cast

KAGGLE_UBC_OCEAN_LATENT_INFERENCE_READY = True
EMBEDDED_PAYLOAD_B64 = """
$embedded_payload_b64
"""
EMBEDDED_PAYLOAD_ZIP_SHA256 = "$embedded_payload_zip_sha256"
EMBEDDED_PAYLOAD_MANIFEST_SHA256 = "$embedded_payload_manifest_sha256"
EMBEDDED_INFERENCE_CONFIG_B64 = "$embedded_inference_config_b64"
EMBEDDED_INFERENCE_CONFIG_SHA256 = "$embedded_inference_config_sha256"
EMBEDDED_PILOT_AUTHORITY_B64 = "$embedded_pilot_authority_b64"
EMBEDDED_PILOT_AUTHORITY_SHA256 = "$embedded_pilot_authority_sha256"
WORKING_ROOT = Path("/kaggle/working")
OUTPUT_ROOT = WORKING_ROOT / "dataset"
SCRATCH_ROOT = WORKING_ROOT / ".spec0021_scratch"
PRIVATE_ROOT = WORKING_ROOT / ".spec0021_private"
PILOT_AUTHORITY_ENV = "EQVAE_SPEC0021_PILOT_AUTHORITY_PATH"
PRODUCTION_CONFIRMATION_ENV = "EQVAE_LATENT_PRODUCTION_CONFIRMED"
PILOT_OUTPUT_ALLOWLIST = frozenset(
    {
        "spec0021_pilot_matrix.csv",
        "spec0021_pilot_authority.json",
    },
)
PYVIPS_VERSION = "3.1.0"


def main() -> int:
    try:
        _ensure_latest_torch()
        _install_dependencies()
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(SCRATCH_ROOT / "inductor")
        os.environ["TRITON_CACHE_DIR"] = str(SCRATCH_ROOT / "triton")
        PRIVATE_ROOT.mkdir(parents=False, exist_ok=False)
        payload_root = _extract_payload(PRIVATE_ROOT / "payload")
        payload_src = payload_root / "src"
        sys.path.insert(0, str(payload_src))
        os.environ["PYTHONPATH"] = _pythonpath(
            payload_src,
            os.environ.get("PYTHONPATH", ""),
        )
        config_path = PRIVATE_ROOT / "spec0021_inference_config.json"
        pilot_authority_path = PRIVATE_ROOT / "spec0021_pilot_authority.json"
        _write_embedded_config(config_path)
        _write_optional_embedded(
            pilot_authority_path,
            EMBEDDED_PILOT_AUTHORITY_B64,
            EMBEDDED_PILOT_AUTHORITY_SHA256,
        )
        import eqvae
        from eqvae.cli.generate_ubc_latent_stores import (
            main as worker_main,
        )

        _assert_import_origin(Path(cast("str", eqvae.__file__)), payload_src)
        config = cast(
            "dict[str, object]",
            json.loads(config_path.read_text(encoding="utf-8")),
        )
        mode = config.get("mode")
        if mode not in {"pilot", "production"}:
            raise ValueError("embedded inference mode is invalid")
        _configure_worker_environment(
            cast("str", mode),
            pilot_authority_path=pilot_authority_path,
        )
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
        return_code = worker_main(
            [
                cast("str", mode),
                "--config",
                str(config_path),
                "--payload-root",
                str(payload_root),
                "--output-root",
                str(OUTPUT_ROOT),
                "--scratch-root",
                str(SCRATCH_ROOT),
            ],
        )
        if return_code != 0:
            return return_code
        _cleanup_runtime_roots()
        _assert_publishable_working_tree(cast("str", mode))
        return 0
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 1
    finally:
        _cleanup_runtime_roots()
        _clear_worker_environment()


def _configure_worker_environment(
    mode: str,
    *,
    pilot_authority_path: Path,
) -> None:
    os.environ[PILOT_AUTHORITY_ENV] = str(pilot_authority_path)
    if mode == "production":
        os.environ[PRODUCTION_CONFIRMATION_ENV] = "1"
    else:
        os.environ.pop(PRODUCTION_CONFIRMATION_ENV, None)


def _clear_worker_environment() -> None:
    for name in (
        PILOT_AUTHORITY_ENV,
        PRODUCTION_CONFIRMATION_ENV,
    ):
        os.environ.pop(name, None)


def _cleanup_runtime_roots() -> None:
    shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)
    shutil.rmtree(PRIVATE_ROOT, ignore_errors=True)


def _assert_publishable_working_tree(mode: str) -> None:
    observed = {path.name for path in WORKING_ROOT.iterdir()}
    if (
        observed != {OUTPUT_ROOT.name}
        or not OUTPUT_ROOT.is_dir()
        or OUTPUT_ROOT.is_symlink()
    ):
        raise RuntimeError("Kaggle working output must contain only dataset/")
    if mode != "pilot":
        return
    pilot_files = {
        path.name
        for path in OUTPUT_ROOT.iterdir()
        if path.is_file() and not path.is_symlink()
    }
    if pilot_files != PILOT_OUTPUT_ALLOWLIST or any(
        not path.is_file() or path.is_symlink() for path in OUTPUT_ROOT.iterdir()
    ):
        raise RuntimeError("pilot dataset output allow-list mismatch")


def _ensure_latest_torch() -> None:
    if not WORKING_ROOT.exists() or os.environ.get("EQVAE_SKIP_REMOTE_SETUP") == "1":
        return
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "torch",
            "torchvision",
            "torchaudio",
        ],
    )


def _install_dependencies() -> None:
    if not WORKING_ROOT.exists() or os.environ.get("EQVAE_SKIP_REMOTE_SETUP") == "1":
        return
    subprocess.check_call(
        ["apt-get", "update", "-qq"],
        stdout=subprocess.DEVNULL,
    )
    subprocess.check_call(
        ["apt-get", "install", "-y", "--no-install-recommends", "libvips"],
    )
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", f"pyvips=={PYVIPS_VERSION}"],
    )


def _extract_payload(destination: Path) -> Path:
    zip_bytes = base64.b64decode(EMBEDDED_PAYLOAD_B64.encode("ascii"))
    if hashlib.sha256(zip_bytes).hexdigest() != EMBEDDED_PAYLOAD_ZIP_SHA256:
        raise RuntimeError("embedded payload zip hash mismatch")
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        for name in archive.namelist():
            path = Path(name)
            if path.is_absolute() or ".." in path.parts:
                raise RuntimeError(f"unsafe embedded payload path: {name}")
        archive.extractall(destination)
    manifest_bytes = (destination / "payload_manifest.json").read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != EMBEDDED_PAYLOAD_MANIFEST_SHA256:
        raise RuntimeError("embedded payload manifest hash mismatch")
    return destination


def _write_embedded_config(path: Path) -> None:
    payload = base64.b64decode(EMBEDDED_INFERENCE_CONFIG_B64.encode("ascii"))
    if hashlib.sha256(payload).hexdigest() != EMBEDDED_INFERENCE_CONFIG_SHA256:
        raise RuntimeError("embedded inference config hash mismatch")
    path.write_bytes(payload)


def _write_optional_embedded(
    path: Path,
    payload_b64: str,
    expected_sha256: str,
) -> None:
    payload = base64.b64decode(payload_b64.encode("ascii"))
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise RuntimeError(f"embedded authority hash mismatch: {path.name}")
    if payload:
        path.write_bytes(payload)


def _pythonpath(payload_src: Path, existing: str) -> str:
    return os.pathsep.join(part for part in (str(payload_src), existing) if part)


def _assert_import_origin(module_file: Path, payload_src: Path) -> None:
    if payload_src.resolve() not in module_file.resolve().parents:
        raise RuntimeError(f"eqvae imported outside payload: {module_file}")


if __name__ == "__main__":
    raise SystemExit(main())
