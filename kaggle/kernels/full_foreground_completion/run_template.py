# Copyright 2026 HiperMaximus
# ruff: noqa: DOC201, DOC501, EM101, INP001, PLC0415, S404, S607, T201, TRY003
"""One fixed whole-WSI completion job using the unchanged paired extractor."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import cast

KAGGLE_FULL_FOREGROUND_COMPLETION_READY = True
CONFIG = json.loads(r"""$config""")
DATASET_REFERENCE = "maximusshtefan/eqvae-full-foreground-inputs"
CONTRACT_NAME = "full_foreground_input.json"
WORKER_CONTRACT_NAME = "spec0022_topup_inference_contract.json"
MANIFEST_NAME = "cancer_topup_manifest.csv"
INPUT_ROOT = Path("/kaggle/input")
WORKING_ROOT = Path("/kaggle/working")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_bundle() -> Path:
    matches = list(INPUT_ROOT.rglob(CONTRACT_NAME))
    if len(matches) != 1 or _sha256(matches[0]) != CONFIG["input_contract_sha256"]:
        raise RuntimeError("Expected the exact shared full-foreground input contract")
    contract = cast("dict[str, object]", json.loads(matches[0].read_text()))
    if contract.get("dataset_reference") != DATASET_REFERENCE:
        raise RuntimeError("Shared input dataset identity differs")
    root = matches[0].parent
    files = cast("dict[str, dict[str, object]]", contract["files"])
    observed = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if observed != {*files, CONTRACT_NAME}:
        raise RuntimeError("Shared input allow-list differs")
    for name, record in files.items():
        path = root / name
        if path.stat().st_size != record["bytes"] or _sha256(path) != record["sha256"]:
            raise RuntimeError("Mounted source/checkpoint/manifest bytes differ")
    run_root = root / f"runs/run_{CONFIG['run_number']:02d}"
    if (
        _sha256(run_root / MANIFEST_NAME) != CONFIG["manifest_sha256"]
        or _sha256(run_root / WORKER_CONTRACT_NAME) != CONFIG["worker_contract_sha256"]
    ):
        raise RuntimeError("Per-run manifest or worker contract differs")
    worker = cast(
        "dict[str, object]",
        json.loads((run_root / WORKER_CONTRACT_NAME).read_text()),
    )
    if (
        worker.get("completion_run_number") != CONFIG["run_number"]
        or worker.get("producer") != CONFIG["producer"]
    ):
        raise RuntimeError("Per-run producer binding differs")
    return root


def _prepare_inference(bundle: Path, destination: Path) -> None:
    destination.mkdir()
    run_root = bundle / f"runs/run_{CONFIG['run_number']:02d}"
    for name in (MANIFEST_NAME, WORKER_CONTRACT_NAME):
        (destination / name).symlink_to(run_root / name)
    for model in ("normal_vae", "so2_vae"):
        name = f"{model}_step_060000.pt"
        (destination / name).symlink_to(bundle / "checkpoints" / name)


def _execute(bundle: Path, wsi_dir: Path) -> None:
    from eqvae.cli.generate_ubc_cancer_topup import run_topup

    inference = WORKING_ROOT / ".full_foreground_inference"
    _prepare_inference(bundle, inference)
    try:
        result = run_topup(
            input_root=inference,
            wsi_dir=wsi_dir,
            output_root=WORKING_ROOT / "dataset",
            scratch_root=WORKING_ROOT / ".full_foreground_work",
        )
        if result["status"] != "complete" or result["row_count"] != CONFIG["row_count"]:
            raise RuntimeError("Whole-WSI completion output is incomplete")
        print(
            json.dumps({
                "producer": CONFIG["producer"],
                "run_number": CONFIG["run_number"],
                "input_contract_sha256": CONFIG["input_contract_sha256"],
                "pair": result,
            }),
            flush=True,
        )
    finally:
        shutil.rmtree(inference)


def _ensure_latest_torch() -> None:
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


def main() -> int:
    """Authenticate inputs, expose only this job's four files, and encode once."""
    os.environ["EQVAE_SESSION_START_MONOTONIC"] = str(time.monotonic())
    try:
        _ensure_latest_torch()
        subprocess.check_call(["apt-get", "update", "-qq"], stdout=subprocess.DEVNULL)
        subprocess.check_call([
            "apt-get",
            "install",
            "-y",
            "--no-install-recommends",
            "libvips",
        ])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyvips==3.1.0"])
        bundle = _resolve_bundle()
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(bundle / "src"))
        import torch

        print(
            json.dumps({
                "torch": str(torch.__version__),
                "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "devices": [
                    torch.cuda.get_device_name(i)
                    for i in range(torch.cuda.device_count())
                ],
                "scope": "frozen_full_foreground_missing_only",
                "config": CONFIG,
            }),
            flush=True,
        )
        roots = [
            root
            for root in (
                INPUT_ROOT / "competitions/UBC-OCEAN/train_images",
                INPUT_ROOT / "UBC-OCEAN/train_images",
            )
            if all((root / f"{wsi}.png").is_file() for wsi in CONFIG["wsi_ids"])
        ]
        if len(roots) != 1:
            raise RuntimeError("Expected one official source root for this job's WSIs")
        os.environ["EQVAE_CANCER_TOPUP_CONFIRMED"] = "1"
        _execute(bundle, roots[0])
        return 0
    finally:
        os.environ.pop("EQVAE_CANCER_TOPUP_CONFIRMED", None)
        os.environ.pop("EQVAE_SESSION_START_MONOTONIC", None)


if __name__ == "__main__":
    raise SystemExit(main())
