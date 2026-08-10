# Copyright 2026 HiperMaximus
# ruff: noqa: EM101, EM102, TRY003
"""Generated single-file Kaggle fixed-25 selector generator template.

Runs on Kaggle where the UBC dataset is mounted (CPU-only, no GPU): it generates
the canonical fixed-25 VALIDATION selector from the real validation shard and
archives the 25 selected patch images (``originals.pt`` plus a montage). Both go to
the kernel output so the exact selected patches can be reviewed before the tracked
selector config is committed. It is generation / inspection tooling only; it never
trains, needs a GPU, or touches the model.
"""

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

KAGGLE_FIXED25_SELECTOR_READY = True
FIXED25_SELECTOR_CONFIG = "configs/spec0001/non_eq_vae_selected_runtime_full.json"
FIXED25_SELECTOR_OUTPUT = "fixed_25_validation_patches.json"
FIXED25_ORIGINALS_PT = "artifacts/fixed25/originals.pt"
FIXED25_ORIGINALS_PNG = "artifacts/fixed25/originals.png"
IMPORT_ARTIFACT = "fixed25_selector_import.json"
RUNTIME_ENVIRONMENT_ARTIFACT = "fixed25_selector_runtime_environment.json"
EXPECTED_SELECTOR_COUNT = 25
DEFAULT_KAGGLE_OUTPUT_DIR = Path("/kaggle/working")
LOCAL_FALLBACK_OUTPUT_DIR = Path("runs/kaggle/fixed25_selector_local")
KERNEL_METADATA = {
    "id": "maximusshtefan/eqvae-fixed25-selector",
    "title": "eqvae fixed25 selector",
    "code_file": "run.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": "true",
    "enable_gpu": "false",
    "enable_internet": "true",
    "dataset_sources": ["maximusshtefan/patches-pre-shuffled-ubc-ocean"],
    "competition_sources": [],
    "kernel_sources": [],
    "model_sources": [],
}
EMBEDDED_PAYLOAD_B64 = """
$embedded_payload_b64
"""
EMBEDDED_PAYLOAD_ZIP_SHA256 = "$embedded_payload_zip_sha256"
EMBEDDED_PAYLOAD_MANIFEST_SHA256 = "$embedded_payload_manifest_sha256"


def main() -> int:
    """Generate the fixed-25 selector and archive the selected patches.

    Returns:
        Process exit status.

    """
    output_dir = _output_dir()
    try:
        return _generate(output_dir)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 1


def _generate(output_dir: Path) -> int:
    _ensure_latest_torch(cpu_only=True)
    _write_runtime_environment(output_dir)
    payload_dir = _extract_payload(output_dir / "embedded_payload")
    payload_src = payload_dir / "src"
    sys.path.insert(0, str(payload_src))
    _prepare_environment(payload_src=payload_src)

    import eqvae  # noqa: PLC0415

    _assert_import_origin(
        module_file=Path(cast("str", eqvae.__file__)),
        payload_src=payload_src,
    )
    manifest = json.loads(
        (payload_dir / "payload_manifest.json").read_text(encoding="utf-8"),
    )
    config_path = payload_dir / FIXED25_SELECTOR_CONFIG
    if not config_path.is_file():
        raise RuntimeError(f"missing embedded config: {config_path}")
    if os.environ.get("EQVAE_FIXED25_SELECTOR_IMPORT_ONLY") == "1":
        _write_import_artifact(
            output_dir=output_dir,
            payload_manifest=manifest,
            config_path=config_path,
        )
        return 0
    selector_output = output_dir / FIXED25_SELECTOR_OUTPUT
    select_code = _run_select_fixed_patches(
        payload_src=payload_src,
        payload_dir=payload_dir,
        config_path=config_path,
        selector_output=selector_output,
    )
    if select_code != 0:
        return select_code
    originals_code = _run_fixed25_originals(
        payload_src=payload_src,
        payload_dir=payload_dir,
        config_path=config_path,
        selector_output=selector_output,
        output_dir=output_dir,
    )
    if originals_code != 0:
        return originals_code
    _validate_outputs(output_dir=output_dir, selector_output=selector_output)
    return 0


def _write_runtime_environment(output_dir: Path) -> None:
    """Record the resolved stack after the mandatory upgrade and before payload use."""
    import torch  # noqa: PLC0415

    benchmark_dir = output_dir / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "status": "pass",
        "benchmark_kind": "fixed25_selector_runtime_environment",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    (benchmark_dir / RUNTIME_ENVIRONMENT_ARTIFACT).write_text(
        f"{json.dumps(artifact, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )


def _run_select_fixed_patches(
    *,
    payload_src: Path,
    payload_dir: Path,
    config_path: Path,
    selector_output: Path,
) -> int:
    # Generate the canonical fixed-25 VALIDATION selector on the mounted shard.
    # --kind fixed_25_validation forces the validation split (belt-and-suspenders),
    # and --validate-crc records crc_checked=True so the selector loads in the
    # CRC-validating full run. Output is /kaggle/working (not the tracked config),
    # so no --allow-tracked-config-overwrite is needed.
    command = [
        sys.executable,
        "-m",
        "eqvae.cli.select_fixed_patches",
        "--config",
        str(config_path),
        "--kind",
        "fixed_25_validation",
        "--data-root",
        "auto",
        "--output",
        str(selector_output),
        "--validate-crc",
    ]
    return _run(command=command, payload_src=payload_src, payload_dir=payload_dir)


def _run_fixed25_originals(
    *,
    payload_src: Path,
    payload_dir: Path,
    config_path: Path,
    selector_output: Path,
    output_dir: Path,
) -> int:
    # Archive the 25 selected images (originals.pt plus a montage) from the freshly
    # generated selector; no model or checkpoint is required.
    command = [
        sys.executable,
        "-m",
        "eqvae.cli.fixed25_originals",
        "--config",
        str(config_path),
        "--data-root",
        "auto",
        "--selector",
        str(selector_output),
        "--output-dir",
        str(output_dir),
    ]
    return _run(command=command, payload_src=payload_src, payload_dir=payload_dir)


def _run(*, command: list[str], payload_src: Path, payload_dir: Path) -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = _pythonpath(payload_src, env.get("PYTHONPATH", ""))
    result = subprocess.run(command, cwd=payload_dir, env=env, check=False)  # noqa: S603
    return int(result.returncode)


def _validate_outputs(*, output_dir: Path, selector_output: Path) -> None:
    if not selector_output.is_file():
        raise RuntimeError(f"selector not written: {selector_output}")
    document = json.loads(selector_output.read_text(encoding="utf-8"))
    if document.get("status") != "pass":
        raise RuntimeError("generated selector status is not 'pass'")
    selectors = document.get("selectors")
    if not isinstance(selectors, list) or len(selectors) != EXPECTED_SELECTOR_COUNT:
        raise RuntimeError("generated selector must contain exactly 25 rows")
    for artifact in (
        output_dir / FIXED25_ORIGINALS_PT,
        output_dir / FIXED25_ORIGINALS_PNG,
    ):
        if not artifact.is_file():
            raise RuntimeError(f"missing selected-patch artifact: {artifact}")


def _write_import_artifact(
    *,
    output_dir: Path,
    payload_manifest: dict[str, object],
    config_path: Path,
) -> None:
    benchmark_dir = output_dir / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "status": "import_smoke_pass",
        "status_scope": "non_promotable_local_upload_simulation",
        "benchmark_kind": "fixed25_selector_import_only",
        "benchmark_source": "kaggle_fixed25_selector",
        "kernel_id": KERNEL_METADATA["id"],
        "full_run_eligible": False,
        "writes_selector": True,
        "writes_originals": True,
        "selector_kind": "fixed_25_validation",
        "selector_output": FIXED25_SELECTOR_OUTPUT,
        "originals_pt": FIXED25_ORIGINALS_PT,
        "originals_png": FIXED25_ORIGINALS_PNG,
        "config_path": str(config_path),
        "config_exists": config_path.is_file(),
        "payload_manifest": payload_manifest,
        "payload_zip_sha256": EMBEDDED_PAYLOAD_ZIP_SHA256,
        "payload_manifest_sha256": EMBEDDED_PAYLOAD_MANIFEST_SHA256,
    }
    (benchmark_dir / IMPORT_ARTIFACT).write_text(
        f"{json.dumps(artifact, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
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
    if hashlib.sha256(zip_bytes).hexdigest() != EMBEDDED_PAYLOAD_ZIP_SHA256:
        raise RuntimeError("embedded payload zip SHA-256 mismatch")
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        for member in archive.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise RuntimeError(f"unsafe embedded payload path: {member.filename}")
        archive.extractall(destination)
    manifest_path = destination / "payload_manifest.json"
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if manifest_hash != EMBEDDED_PAYLOAD_MANIFEST_SHA256:
        raise RuntimeError("embedded payload manifest SHA-256 mismatch")
    return destination


def _prepare_environment(*, payload_src: Path) -> None:
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.pop("EQVAE_DATA_ROOT", None)
    os.environ["PYTHONPATH"] = _pythonpath(
        payload_src,
        os.environ.get("PYTHONPATH", ""),
    )


def _pythonpath(payload_src: Path, existing: str) -> str:
    parts = [str(payload_src)]
    if existing:
        parts.append(existing)
    return os.pathsep.join(parts)


def _assert_import_origin(*, module_file: Path, payload_src: Path) -> None:
    resolved_module = module_file.resolve()
    resolved_payload_src = payload_src.resolve()
    if resolved_payload_src not in resolved_module.parents:
        raise RuntimeError(
            f"eqvae imported from {resolved_module}, not {resolved_payload_src}",
        )


if __name__ == "__main__":
    raise SystemExit(main())
