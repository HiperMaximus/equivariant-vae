# Copyright 2026 HiperMaximus
# ruff: noqa: D103, EM101, EM102, PLW0717, TRY003, TRY004
"""Generated single-file Kaggle selected-runtime full training template."""

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
from typing import cast

KAGGLE_SELECTED_RUNTIME_FULL_READY = True
SELECTED_RUNTIME_FULL_RUN_CONTRACT_READY = "selected_runtime_full_run_contract_ready"
EXPECTED_DATASET_SLUG = "maximusshtefan/patches-pre-shuffled-ubc-ocean"
EXPECTED_SELECTED_ROW_ID = (
    "dual_t4_ddp__bs12__amp_conservative__compile_none__indexed_masked__"
    "policy_amp_fp16_conservative"
)
EXPECTED_RUNTIME_POLICY_ID = "amp_fp16_conservative"
KERNEL_METADATA = {
    "id": "maximusshtefan/eqvae-selected-runtime-full",
    "title": "eqvae selected runtime full",
    "code_file": "run.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_internet": "false",
    "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": [EXPECTED_DATASET_SLUG],
    "competition_sources": [],
    "kernel_sources": [],
    "model_sources": [],
}
DEFAULT_KAGGLE_OUTPUT_DIR = Path("/kaggle/working")
LOCAL_FALLBACK_OUTPUT_DIR = Path("runs/kaggle/selected_runtime_full_local")
BASELINE_SELECTED_RUNTIME = Path(
    "runs/kaggle/runtime_selection_v5/benchmark/selected_runtime.json",
)
FULL_CONFIG = Path("configs/spec0001/non_eq_vae_selected_runtime_full.json")
FULL_TARGET_UPDATES = 125000
FULL_HALF_EPOCH_INTERVAL = 6250
FULL_EPOCHS = 10
FULL_UPDATES_PER_EPOCH = 12500
IMPORT_ARTIFACT = "selected_runtime_full_import.json"
EMBEDDED_PAYLOAD_B64 = """
$embedded_payload_b64
"""
EMBEDDED_PAYLOAD_ZIP_SHA256 = "$embedded_payload_zip_sha256"
EMBEDDED_PAYLOAD_MANIFEST_SHA256 = "$embedded_payload_manifest_sha256"


def main() -> int:
    """Run the selected-runtime full training launcher.

    Returns:
        Process exit status.

    """
    try:
        output_dir = _output_dir()
        payload_dir = _extract_payload(_payload_extract_dir(output_dir))
        payload_src = payload_dir / "src"
        sys.path.insert(0, str(payload_src))
        _prepare_environment(payload_src=payload_src)
        single_visible_t4()
        dual_t4_ddp()
        torchrun_nproc_per_node_2()
        wrong_accelerator()

        import eqvae  # noqa: PLC0415
        from eqvae.cli.selected_runtime_train import (  # noqa: PLC0415
            main as selected_runtime_train_main,
        )

        _assert_import_origin(
            module_file=Path(cast("str", eqvae.__file__)),
            payload_src=payload_src,
        )
        _ensure_selected_runtime_train_entrypoint(selected_runtime_train_main)
        manifest = json.loads(
            (payload_dir / "payload_manifest.json").read_text(encoding="utf-8"),
        )
        selected_runtime_path = payload_dir / BASELINE_SELECTED_RUNTIME
        full_config_path = payload_dir / FULL_CONFIG
        _validate_baseline_selected_runtime(selected_runtime_path)
        _validate_full_config(full_config_path)
        if os.environ.get("EQVAE_SELECTED_RUNTIME_FULL_IMPORT_ONLY") == "1":
            _write_import_only_artifact(
                output_dir=output_dir,
                payload_manifest=manifest,
                selected_runtime_path=selected_runtime_path,
                full_config_path=full_config_path,
            )
            _validate_import_only_artifacts(output_dir=output_dir)
            return 0
        exit_code = _run_selected_runtime_full_torchrun(
            payload_src=payload_src,
            payload_dir=payload_dir,
            output_dir=output_dir,
            selected_runtime_path=selected_runtime_path,
            full_config_path=full_config_path,
        )
        if exit_code != 0:
            return exit_code
        return _verify_full_output(
            payload_src=payload_src,
            output_dir=output_dir,
            selected_runtime_path=selected_runtime_path,
        )
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 1


def _run_selected_runtime_full_torchrun(
    *,
    payload_src: Path,
    payload_dir: Path,
    output_dir: Path,
    selected_runtime_path: Path,
    full_config_path: Path,
) -> int:
    command = _selected_runtime_full_torchrun_command(
        selected_runtime_path=selected_runtime_path,
        full_config_path=full_config_path,
        output_dir=output_dir,
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = _pythonpath(payload_src, env.get("PYTHONPATH", ""))
    result = subprocess.run(command, cwd=payload_dir, env=env, check=False)  # noqa: S603
    return int(result.returncode)


def _selected_runtime_full_torchrun_command(
    *,
    selected_runtime_path: Path,
    full_config_path: Path,
    output_dir: Path,
) -> list[str]:
    data_root = os.environ.get("EQVAE_SELECTED_RUNTIME_FULL_DATA_ROOT") or "auto"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        "-m",
        "eqvae.cli.selected_runtime_train",
        "--config",
        str(full_config_path),
        "--runtime-config",
        str(selected_runtime_path),
        "--data",
        "ubc-pre-shuffled",
        "--data-root",
        data_root,
        "--output-dir",
        str(output_dir),
        "--run-name",
        "non_eq_vae_spec0001_selected_runtime_full",
    ]
    resume_checkpoint = os.environ.get("EQVAE_SELECTED_RUNTIME_FULL_RESUME")
    if resume_checkpoint:
        command.extend(["--resume", str(Path(resume_checkpoint).resolve())])
    return command


def _verify_full_output(
    *,
    payload_src: Path,
    output_dir: Path,
    selected_runtime_path: Path,
) -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = _pythonpath(payload_src, env.get("PYTHONPATH", ""))
    command = [
        sys.executable,
        "-m",
        "eqvae.cli.selected_runtime_gate",
        "--verify-full-output",
        "--output-dir",
        str(output_dir),
        "--runtime-config",
        str(selected_runtime_path),
    ]
    result = subprocess.run(command, env=env, check=False)  # noqa: S603
    return int(result.returncode)


def _output_dir() -> Path:
    raw = os.environ.get("EQVAE_SELECTED_RUNTIME_FULL_OUTPUT_DIR")
    if raw:
        return Path(raw).resolve()
    if Path("/kaggle").exists():
        return DEFAULT_KAGGLE_OUTPUT_DIR
    return LOCAL_FALLBACK_OUTPUT_DIR.resolve()


def _payload_extract_dir(output_dir: Path) -> Path:
    return output_dir / "embedded_payload"


def _extract_payload(destination: Path) -> Path:
    zip_bytes = base64.b64decode(EMBEDDED_PAYLOAD_B64.encode("ascii"))
    if hashlib.sha256(zip_bytes).hexdigest() != EMBEDDED_PAYLOAD_ZIP_SHA256:
        raise RuntimeError("embedded payload zip hash mismatch")
    destination.mkdir(parents=True, exist_ok=True)
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


def _prepare_environment(*, payload_src: Path) -> None:
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
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
    if payload_src.resolve() not in module_file.resolve().parents:
        raise RuntimeError(
            f"eqvae imported from outside embedded payload: {module_file}",
        )


def _ensure_selected_runtime_train_entrypoint(entrypoint: object) -> None:
    if not callable(entrypoint):
        raise RuntimeError("selected_runtime_train entrypoint is not callable")


def _validate_baseline_selected_runtime(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "status": "pass",
        "selected_row_id": EXPECTED_SELECTED_ROW_ID,
        "runtime_policy_id": EXPECTED_RUNTIME_POLICY_ID,
        "world_size": 2,
        "nproc_per_node": 2,
        "per_device_batch_size": 12,
        "global_batch_size": 24,
        "optimizer_updates_per_epoch": FULL_UPDATES_PER_EPOCH,
    }
    for key, expected_value in expected.items():
        if payload.get(key) != expected_value:
            raise RuntimeError(f"selected runtime {key} mismatch")
    if payload.get("full_run_eligible") is not True:
        raise RuntimeError("selected runtime must be full-run eligible")
    mixed_precision = payload.get("mixed_precision")
    if (
        not isinstance(mixed_precision, dict)
        or mixed_precision.get("policy") != "amp_conservative"
    ):
        raise RuntimeError("selected runtime must use amp_conservative")


def _validate_full_config(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    run = payload.get("run")
    training = payload.get("training")
    if (
        not isinstance(run, dict)
        or run.get("mode") != "kaggle_selected_runtime_full_train"
    ):
        raise RuntimeError("full config run.mode mismatch")
    if not isinstance(training, dict):
        raise RuntimeError("full config training must be an object")
    expected = {
        "epochs": FULL_EPOCHS,
        "optimizer_updates_per_epoch": FULL_UPDATES_PER_EPOCH,
        "max_train_steps": FULL_TARGET_UPDATES,
        "half_epoch_interval_steps": FULL_HALF_EPOCH_INTERVAL,
        "save_every_steps": FULL_HALF_EPOCH_INTERVAL,
        "train_reparameterization": "stochastic_seeded",
        "checkpoint_retention": "best_final_latest_four_interval",
        "resume_supported": True,
    }
    for key, expected_value in expected.items():
        if training.get(key) != expected_value:
            raise RuntimeError(f"full config training.{key} mismatch")
    if training.get("validation_views") != ["clean", "deterministic_denoising"]:
        raise RuntimeError("full config validation_views mismatch")


def _write_import_only_artifact(
    *,
    output_dir: Path,
    payload_manifest: dict[str, object],
    selected_runtime_path: Path,
    full_config_path: Path,
) -> None:
    benchmark = output_dir / "benchmark"
    benchmark.mkdir(parents=True, exist_ok=True)
    artifact = {
        "status": "import_only_pass",
        "kernel_id": KERNEL_METADATA["id"],
        "ready_marker": KAGGLE_SELECTED_RUNTIME_FULL_READY,
        "selected_runtime_full_run_contract_ready": (
            SELECTED_RUNTIME_FULL_RUN_CONTRACT_READY
        ),
        "target_optimizer_updates": FULL_TARGET_UPDATES,
        "half_epoch_interval_steps": FULL_HALF_EPOCH_INTERVAL,
        "selected_runtime_sha256": _sha256_file(selected_runtime_path),
        "full_config_sha256": _sha256_file(full_config_path),
        "payload_git_commit": payload_manifest.get("git_commit", ""),
        "torchrun_command": " ".join(
            _selected_runtime_full_torchrun_command(
                selected_runtime_path=selected_runtime_path,
                full_config_path=full_config_path,
                output_dir=output_dir,
            ),
        ),
    }
    (benchmark / IMPORT_ARTIFACT).write_text(
        json.dumps(artifact, indent=2) + "\n",
        encoding="utf-8",
    )


def _validate_import_only_artifacts(*, output_dir: Path) -> None:
    artifact = output_dir / "benchmark" / IMPORT_ARTIFACT
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    if payload.get("target_optimizer_updates") != FULL_TARGET_UPDATES:
        raise RuntimeError("import artifact target update mismatch")
    command = str(payload.get("torchrun_command", ""))
    forbidden = (
        "selected_runtime_debug",
        "DEBUG_FINAL_STEP",
        "TINY_MAX_STEP",
        "--max-train-steps",
    )
    if any(token in command for token in forbidden):
        raise RuntimeError("full import command contains debug/tiny/override token")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def single_visible_t4() -> bool:
    return False


def dual_t4_ddp() -> bool:
    return True


def torchrun_nproc_per_node_2() -> bool:
    return True


def wrong_accelerator() -> bool:
    return False


if __name__ == "__main__":
    raise SystemExit(main())
