# Copyright 2026 HiperMaximus
# ruff: noqa: DOC501, EM101, EM102, PLR0913, PLR2004, PLW0717, TRY003, TRY301
"""Single-file private SO2 debug/resume/fixed32 prelaunch kernel."""

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

KAGGLE_SO2_PRELAUNCH_READY = True
EXPECTED_DATASET_SLUG = "maximusshtefan/patches-pre-shuffled-ubc-ocean"
KERNEL_METADATA = {
    "id": "maximusshtefan/eqvae-so2-prelaunch",
    "title": "eqvae so2 prelaunch",
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
OUTPUT = Path("/kaggle/working")
RUNTIME = Path("configs/spec0001/non_eq_vae_selected_runtime.json")
DEBUG = Path("configs/spec0016/so2_selected_runtime_debug.json")
TINY = Path("configs/spec0016/so2_kaggle_tiny_overfit.json")
FULL = Path("configs/spec0016/so2_selected_runtime_full.json")
SELECTOR = Path("benchmark/fixed_32_train_overfit_patches.json")
EMBEDDED_PAYLOAD_B64 = """
$embedded_payload_b64
"""
EMBEDDED_PAYLOAD_ZIP_SHA256 = "$embedded_payload_zip_sha256"
EMBEDDED_PAYLOAD_MANIFEST_SHA256 = "$embedded_payload_manifest_sha256"


def main() -> int:
    """Run the exact bounded SO2 prelaunch sequence.

    Returns:
        Process exit status.

    """
    output = Path(os.environ.get("EQVAE_SO2_PRELAUNCH_OUTPUT_DIR", OUTPUT)).resolve()
    try:
        _ensure_latest_torch()
        payload = _extract(output / "embedded_payload")
        manifest = _read_json(payload / "payload_manifest.json")
        sys.path.insert(0, str(payload / "src"))
        from eqvae.benchmarking.so2_prelaunch import (  # noqa: PLC0415
            execution_identity,
            validate_prelaunch_artifacts,
        )

        if os.environ.get("EQVAE_SO2_PRELAUNCH_IMPORT_ONLY") == "1":
            _write_json(
                output / "benchmark/so2_prelaunch_import.json",
                {"status": "pass", "payload_manifest": manifest},
            )
            return 0
        environment = _environment(payload / "src")
        data_root = os.environ.get("EQVAE_SO2_PRELAUNCH_DATA_ROOT", "auto")
        selector = output / SELECTOR
        _run(
            payload,
            environment,
            (
                "-m",
                "eqvae.cli.select_fixed_patches",
                "--config",
                str(payload / TINY),
                "--kind",
                "fixed_32_train_overfit",
                "--data-root",
                data_root,
                "--masked-holdout-csv",
                str(payload / "docs/data/ubc_ocean_masked_holdout_ids.csv"),
                "--output",
                str(selector),
                "--validate-crc",
            ),
        )
        phase1 = output / "debug_phase1"
        _train(
            payload,
            environment,
            config=DEBUG,
            output=phase1,
            data_root=data_root,
            run_name="so2_spec0016_debug_phase1",
            max_steps=4,
            save_every=4,
        )
        checkpoint = phase1 / "checkpoints/step_000004.pt"
        if not checkpoint.is_file():
            raise RuntimeError("SO2 debug checkpoint at update 4 is missing")
        debug = output / "debug_resume"
        _train(
            payload,
            environment,
            config=DEBUG,
            output=debug,
            data_root=data_root,
            run_name="so2_spec0016_debug_resume",
            max_steps=8,
            save_every=4,
            resume=checkpoint,
        )
        tiny = output / "tiny_overfit"
        _train(
            payload,
            environment,
            config=TINY,
            output=tiny,
            data_root=data_root,
            run_name="so2_spec0016_tiny_overfit",
            max_steps=128,
            save_every=64,
            selector=selector,
        )
        verdict = _verdict(
            manifest=manifest,
            phase1=phase1,
            debug=debug,
            tiny=tiny,
            identity=execution_identity(payload),
        )
        verdict_path = output / "benchmark/so2_prelaunch_verdict.json"
        _write_json(verdict_path, verdict)
        blockers = validate_prelaunch_artifacts(
            output,
            repo_root=payload,
            expected_source_commit=cast("str", manifest["git_commit"]),
        )
        if blockers:
            raise RuntimeError(f"invalid SO2 prelaunch verdict: {blockers}")
        if verdict["status"] != "pass":
            raise RuntimeError("SO2 prelaunch artifact verdict failed")
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 1
    return 0


def _train(
    payload: Path,
    environment: dict[str, str],
    *,
    config: Path,
    output: Path,
    data_root: str,
    run_name: str,
    max_steps: int,
    save_every: int,
    resume: Path | None = None,
    selector: Path | None = None,
) -> None:
    arguments = [
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        "-m",
        "eqvae.cli.selected_runtime_train",
        "--config",
        str(payload / config),
        "--runtime-config",
        str(payload / RUNTIME),
        "--data",
        "ubc-pre-shuffled",
        "--data-root",
        data_root,
        "--output-dir",
        str(output),
        "--run-name",
        run_name,
        "--max-train-steps",
        str(max_steps),
        "--max-val-steps",
        "1",
        "--save-every-steps",
        str(save_every),
    ]
    if resume is not None:
        arguments.extend(("--resume", str(resume)))
    if selector is not None:
        arguments.extend(("--fixed-train-patches", str(selector)))
    _run(payload, environment, tuple(arguments))


def _run(
    payload: Path,
    environment: dict[str, str],
    arguments: tuple[str, ...],
) -> None:
    completed = subprocess.run(  # noqa: S603
        (sys.executable, *arguments),
        cwd=payload,
        env=environment,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"prelaunch subprocess failed with {completed.returncode}")


def _verdict(
    *,
    manifest: dict[str, object],
    phase1: Path,
    debug: Path,
    tiny: Path,
    identity: dict[str, str],
) -> dict[str, object]:
    debug_summary = _read_json(debug / "benchmark/selected_runtime_debug_summary.json")
    resume = _read_json(debug / "benchmark/checkpoint_resume_proof.json")
    debug_proofs = _runtime_proofs_pass(debug)
    tiny_summary = _read_json(tiny / "benchmark/tiny_overfit_summary.json")
    gate_summary = _read_json(tiny / "benchmark/gate_health_summary.json")
    tiny_proofs = _runtime_proofs_pass(tiny)
    performance = cast(
        "list[object]",
        tiny_summary.get("settled_real_loader_rank_metrics", []),
    )
    checks = {
        "debug": debug_summary.get("status") == "local_pass" and debug_proofs,
        "resume": resume.get("status") == "local_pass",
        "tiny": tiny_summary.get("status") == "local_pass" and tiny_proofs,
        "gates": gate_summary.get("status") == "local_pass"
        and gate_summary.get("rows_written") == 68,
        "performance": tiny_summary.get("settled_real_loader_performance_status")
        == "pass"
        and len(performance) == 2,
    }
    return {
        "schema_version": "spec0016.so2_prelaunch_verdict.v1",
        "status": "pass" if all(checks.values()) else "fail",
        "full_run_eligible": False,
        "full_push_requires_explicit_measured_cost_acceptance": True,
        "checks": checks,
        "source_git_commit": manifest.get("git_commit"),
        "source_git_dirty": manifest.get("git_dirty"),
        "execution_identity_sha256": identity,
        "phase_artifact_manifest_sha256": {
            "debug_phase1": _sha256(phase1 / "benchmark/artifact_manifest.json"),
            "debug_resume": _sha256(debug / "benchmark/artifact_manifest.json"),
            "tiny_overfit": _sha256(tiny / "benchmark/artifact_manifest.json"),
        },
        "selected_runtime": {
            "per_device_batch_size": 25,
            "global_batch_size": 50,
        },
        "slower_rank_mean_step_ms": tiny_summary.get("slower_rank_mean_step_ms"),
        "projected_epoch_seconds": tiny_summary.get("projected_epoch_seconds"),
        "settled_real_loader_rank_metrics": performance,
    }


def _runtime_proofs_pass(run: Path) -> bool:
    summary = _read_json(run / "benchmark/training_summary.json")
    plan = _read_json(run / "benchmark/selected_runtime_plan_applied.json")
    checkpoint = _read_json(run / "benchmark/checkpoint_resume_proof.json")
    gates = _read_json(run / "benchmark/gate_health_summary.json")
    ddp = summary.get("ddp_rank_device_proof")
    amp = summary.get("amp_execution")
    return (
        plan.get("status") == "local_pass"
        and checkpoint.get("status") == "local_pass"
        and gates.get("status") == "local_pass"
        and isinstance(ddp, dict)
        and ddp.get("status") == "local_pass"
        and isinstance(amp, dict)
        and amp.get("enabled") is True
        and amp.get("grad_scaler_enabled") is True
        and amp.get("autocast_dtype") == "float16"
        and amp.get("requested_autocast_dtype") == "float16"
        and amp.get("local_amp_status") == "executed_amp_fp16_conservative"
    )


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
    if (
        _sha256(destination / "payload_manifest.json")
        != EMBEDDED_PAYLOAD_MANIFEST_SHA256
    ):
        raise RuntimeError("payload manifest hash mismatch")
    return destination


def _environment(source: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(source), environment.get("PYTHONPATH", "")) if part
    )
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    value = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return cast("dict[str, object]", value)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
