# Copyright 2026 HiperMaximus
# ruff: noqa: EM101, TRY003
"""Generated launcher for the final locked Spec 0013 dual-T4 mechanics probe."""

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
SCHEMA_VERSION = "spec0013.so2_dual_t4_final.v1"
PROBE_KIND = "locked_so2_architecture_mechanics_final"
RUNTIME_BUNDLE_ID = "compile_step_python_reducer_fp16_channels_last"
SELECTED_RUNTIME_SHA256 = (
    "e9e998fd161f0955959c64aed7cd7ddbdfcb55a271b9ce05805903c97c93efb8"
)
SELECTED_RUNTIME_PATH = Path("configs/spec0001/non_eq_vae_selected_runtime.json")
SETTLED_UPDATES = 32
TIMED_WINDOW_UPDATES = 50
TIMED_WINDOW_COUNT = 2
REQUIRED_GPU_NAME = "Tesla T4"
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


def _validate_artifact(  # noqa: C901, PLR0912, PLR0914, PLR0915
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
        "selected_mechanics": "padded_bmm_direct",
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
    assignments = payload.get("rank_device_assignments")
    if not isinstance(assignments, list) or {
        (item.get("rank"), item.get("local_rank"), item.get("current_device"))
        for item in assignments
        if isinstance(item, dict)
    } != {(0, 0, 0), (1, 1, 1)}:
        errors.append("rank-to-device assignments must be the two-device bijection")
    elif [item.get("device_name") for item in assignments] != [REQUIRED_GPU_NAME] * 2:
        errors.append("rank-to-device assignments must use two Tesla T4 GPUs")
    if payload.get("gpu_names") != [REQUIRED_GPU_NAME] * 2:
        errors.append("gpu_names must identify exactly two Tesla T4 GPUs")
    rank_measurements = payload.get("rank_measurements")
    expected_blocks = {
        "identity_A",
        "encoder_A_to_B",
        "decoder_B_to_A",
        "largest_D_to_D",
    }
    expected_paths = {
        "equivariant_eager",
        "equivariant_compiled",
        "normal_compiled",
    }
    if not isinstance(rank_measurements, list) or [
        row.get("rank") for row in rank_measurements if isinstance(row, dict)
    ] != [0, 1]:
        errors.append("rank_measurements must contain ranks 0 and 1")
    else:
        for rank_row in rank_measurements:
            if not isinstance(rank_row, dict):
                errors.append("every rank measurement must be an object")
                continue
            blocks = rank_row.get("blocks")
            if (
                not isinstance(blocks, list)
                or len(blocks) != len(expected_blocks)
                or {block.get("name") for block in blocks if isinstance(block, dict)}
                != expected_blocks
            ):
                errors.append("every rank must contain the exact four block results")
                continue
            for block in blocks:
                paths = block.get("paths") if isinstance(block, dict) else None
                if not isinstance(paths, dict) or set(paths) != expected_paths:
                    errors.append(
                        "every block must contain the exact three timing paths",
                    )
                    continue
                for path in paths.values():
                    if not isinstance(path, dict):
                        errors.append("every timing path must be an object")
                        continue
                    windows = path.get("windows")
                    pooled = path.get("pooled")
                    if (
                        not isinstance(windows, list)
                        or len(windows) != TIMED_WINDOW_COUNT
                        or any(
                            not isinstance(window, dict)
                            or len(window.get("samples_ms", [])) != TIMED_WINDOW_UPDATES
                            for window in windows
                        )
                        or not isinstance(pooled, dict)
                        or len(pooled.get("samples_ms", [])) != 2 * TIMED_WINDOW_UPDATES
                    ):
                        errors.append("every timing path must contain two full windows")
            assembly = rank_row.get("assembly_diagnostic")
            assembly_windows = (
                assembly.get("windows") if isinstance(assembly, dict) else None
            )
            assembly_header_valid = (
                isinstance(assembly, dict)
                and assembly.get("selection_gate") is False
                and isinstance(assembly_windows, list)
                and len(assembly_windows) == TIMED_WINDOW_COUNT
            )
            assembly_windows_valid = isinstance(assembly_windows, list) and all(
                isinstance(window, dict)
                and all(
                    isinstance(window.get(name), dict)
                    and len(window[name].get("samples_ms", [])) == TIMED_WINDOW_UPDATES
                    for name in ("expansion", "complete")
                )
                for window in assembly_windows
            )
            assembly_pooled_valid = isinstance(assembly, dict) and all(
                isinstance(assembly.get(name), dict)
                and len(assembly[name].get("samples_ms", []))
                == 2 * TIMED_WINDOW_UPDATES
                for name in ("pooled_expansion", "pooled_complete")
            )
            if (
                not assembly_header_valid
                or not assembly_windows_valid
                or not assembly_pooled_valid
            ):
                errors.append("assembly diagnostic must contain two full windows")
    if payload.get("acceptance_failures") != []:
        errors.append("acceptance_failures must be empty")
    if payload.get("follow_up_probe_permitted") is not False:
        errors.append("final runner must not authorize another remote arm")
    if errors:
        raise RuntimeError("; ".join(errors))
    if Path("/kaggle/working").exists() and output_dir != DEFAULT_OUTPUT_DIR:
        raise RuntimeError("Kaggle artifact must remain under /kaggle/working")


if __name__ == "__main__":
    raise SystemExit(main())
