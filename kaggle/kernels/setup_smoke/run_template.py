# Copyright 2026 HiperMaximus
"""Generated single-file Kaggle setup-smoke launcher template."""

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

KAGGLE_SETUP_SMOKE_READY = True
SETUP_BENCHMARK_KIND = "synthetic_kaggle_setup_smoke"
SETUP_BENCHMARK_SOURCE = "kaggle_script_kernel_synthetic_setup_smoke"
SETUP_CORRUPTION_VIEW = "train_corrupted_kaggle_setup_smoke"
DEFAULT_KAGGLE_OUTPUT_DIR = Path("/kaggle/working")
LOCAL_FALLBACK_OUTPUT_DIR = Path("runs/kaggle/setup_smoke_local")
EMBEDDED_PAYLOAD_B64 = """
$embedded_payload_b64
"""
EMBEDDED_PAYLOAD_ZIP_SHA256 = "$embedded_payload_zip_sha256"
EMBEDDED_PAYLOAD_MANIFEST_SHA256 = "$embedded_payload_manifest_sha256"


def main() -> int:
    """Run the setup smoke from an embedded payload.

    Returns:
        Process exit status.

    """
    output_dir = _output_dir()
    try:
        return _run_setup_smoke(output_dir)
    except Exception as error:
        _write_bootstrap_failure(output_dir=output_dir, error=error)
        raise


def _run_setup_smoke(output_dir: Path) -> int:
    _require_python_version()
    _ensure_latest_torch(cpu_only=True)
    payload_dir = _extract_payload(output_dir / "embedded_payload")
    payload_src = payload_dir / "src"
    sys.path.insert(0, str(payload_src))
    _clear_setup_environment()

    import eqvae  # noqa: PLC0415
    from eqvae.cli.kaggle_smoke import main as smoke_main  # noqa: PLC0415

    _assert_import_origin(
        module_file=Path(cast("str", eqvae.__file__)),
        payload_src=payload_src,
    )
    data_root = output_dir / "setup_smoke_data"
    _write_synthetic_shards(data_root=data_root)
    config_path = _write_setup_config(payload_dir=payload_dir, data_root=data_root)
    result = smoke_main(
        (
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--data-root",
            str(data_root),
        ),
    )
    _validate_setup_artifact(output_dir)
    return result


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


def _require_python_version() -> None:
    if sys.version_info < (3, 12):  # noqa: UP036
        message = (
            "eqvae setup smoke requires Python >= 3.12 because active source "
            "uses Python 3.12 type-alias syntax"
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


def _clear_setup_environment() -> None:
    os.environ.pop("EQVAE_DATA_ROOT", None)


def _assert_import_origin(*, module_file: Path, payload_src: Path) -> None:
    resolved_module = module_file.resolve()
    resolved_payload_src = payload_src.resolve()
    if resolved_payload_src not in resolved_module.parents:
        message = f"eqvae imported from {resolved_module}, not {resolved_payload_src}"
        raise RuntimeError(message)


def _write_synthetic_shards(*, data_root: Path) -> None:
    from eqvae.data.synthetic import (  # noqa: PLC0415
        SyntheticPatchSpec,
        write_synthetic_patch_shard,
    )

    dataset_dir = data_root / "dataset"
    write_synthetic_patch_shard(
        bin_path=dataset_dir / "ubc_train_shuffled.bin",
        csv_path=dataset_dir / "ubc_train_shuffled.csv",
        spec=SyntheticPatchSpec(count=4, image_size=64, channels=3, seed=1301),
        include_idx=False,
    )
    write_synthetic_patch_shard(
        bin_path=dataset_dir / "ubc_ocean_valid.bin",
        csv_path=dataset_dir / "ubc_ocean_valid.csv",
        spec=SyntheticPatchSpec(count=4, image_size=64, channels=3, seed=1302),
        include_idx=True,
    )


def _write_setup_config(*, payload_dir: Path, data_root: Path) -> Path:
    config_path = (
        payload_dir
        / "configs"
        / "spec0001"
        / "non_eq_vae_kaggle_setup_smoke.generated.json"
    )
    payload = {
        "schema_version": "spec0001.v0",
        "status": "kaggle_setup_smoke_ready",
        "run": {
            "name": "eqvae_kaggle_setup_smoke",
            "mode": "synthetic_kaggle_setup_smoke",
        },
        "source_config": str(
            payload_dir / "configs" / "spec0001" / "non_eq_vae_debug_cpu.json",
        ),
        "data": {
            "kind": "synthetic-ubc-setup-smoke",
            "dataset_slug": "",
            "data_root": str(data_root),
            "image_size": 64,
            "channels": 3,
            "normalization": "uint8_to_minus1_plus1",
        },
        "kaggle_smoke": {
            "benchmark_kind": SETUP_BENCHMARK_KIND,
            "benchmark_source": SETUP_BENCHMARK_SOURCE,
            "full_run_eligible": False,
            "batch_size": 1,
            "max_train_steps": 3,
            "max_validation_batches": 1,
            "num_workers": 0,
            "validate_crc": True,
            "corruption_view": SETUP_CORRUPTION_VIEW,
        },
    }
    config_path.write_text(
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    return config_path


def _validate_setup_artifact(output_dir: Path) -> None:
    artifact_path = output_dir / "benchmark" / "kaggle_setup_smoke.json"
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if payload.get("status") != "smoke_pass":
        errors.append("setup smoke artifact did not pass")
    if payload.get("status_scope") != "non_promotable_setup_smoke":
        errors.append("setup smoke artifact has wrong status_scope")
    if payload.get("benchmark_kind") != SETUP_BENCHMARK_KIND:
        errors.append("setup smoke artifact has wrong benchmark_kind")
    if payload.get("benchmark_source") != SETUP_BENCHMARK_SOURCE:
        errors.append("setup smoke artifact has wrong benchmark_source")

    data = payload.get("data")
    if not isinstance(data, dict) or data.get("origin") == "kaggle_input_mount":
        errors.append("setup smoke used a Kaggle input mount")

    runtime = payload.get("runtime")
    if not isinstance(runtime, dict) or runtime.get("requires_cuda_t4") is not False:
        errors.append("setup smoke should not require CUDA T4")

    train = payload.get("train")
    if (
        not isinstance(train, dict)
        or not isinstance(train.get("total_applied_count"), int)
        or train["total_applied_count"] <= 0
    ):
        errors.append("setup smoke did not apply a deterministic corruption")

    if errors:
        message = "; ".join(errors)
        raise RuntimeError(message)


def _write_bootstrap_failure(*, output_dir: Path, error: Exception) -> None:
    benchmark_dir = output_dir / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    failure_message = "".join(
        traceback.format_exception_only(type(error), error),
    ).strip()
    payload = {
        "schema_version": "spec0001.kaggle_setup_smoke_bootstrap.v1",
        "status": "fail",
        "status_scope": "non_promotable_setup_smoke",
        "benchmark_kind": SETUP_BENCHMARK_KIND,
        "benchmark_source": SETUP_BENCHMARK_SOURCE,
        "failure_kind": type(error).__name__,
        "failure_message_hash": hashlib.sha256(
            failure_message.encode("utf-8"),
        ).hexdigest(),
        "python_version": sys.version.split()[0],
        "payload_zip_sha256": EMBEDDED_PAYLOAD_ZIP_SHA256,
        "payload_manifest_sha256": EMBEDDED_PAYLOAD_MANIFEST_SHA256,
    }
    (benchmark_dir / "kaggle_setup_smoke_bootstrap.json").write_text(
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
