# Copyright 2026 HiperMaximus
# ruff: noqa: BLE001, DOC201, DOC501, EM101, INP001, PLC0415, PLR2004, PLW0717, S404, S607, T201, TRY003, TRY300, TRY301
"""One WSI's missing posterior means, using the unchanged paired top-up worker."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import cast

KAGGLE_WSI45630_COMPLETION_READY = True
INPUT_CONTRACT_SHA256 = "$input_contract_sha256"
INPUT_CONTRACT_NAME = "wsi45630_input.json"
DATASET_REFERENCE = "maximusshtefan/eqvae-wsi45630-completion-inputs"
WORKING_ROOT = Path("/kaggle/working")
INPUT_ROOT = Path("/kaggle/input")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_bundle() -> Path:
    matches = list(INPUT_ROOT.rglob(INPUT_CONTRACT_NAME))
    if len(matches) != 1 or _sha256(matches[0]) != INPUT_CONTRACT_SHA256:
        raise RuntimeError("Expected the exact single-WSI input contract")
    contract = cast("dict[str, object]", json.loads(matches[0].read_text()))
    if contract["dataset_reference"] != DATASET_REFERENCE:
        raise RuntimeError("Unexpected input dataset identity")
    root = matches[0].parent
    files = cast("dict[str, dict[str, object]]", contract["files"])
    for name, record in files.items():
        path = root / name
        if path.stat().st_size != record["bytes"] or _sha256(path) != record["sha256"]:
            raise RuntimeError("Mounted source or inference input differs")
    observed = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if observed != {*files, INPUT_CONTRACT_NAME}:
        raise RuntimeError("Unexpected file in the single-WSI input bundle")
    with (root / "inference/cancer_topup_manifest.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 22649 or any(row["wsi_id"] != "45630" for row in rows):
        raise RuntimeError("Inference must contain only 22,649 WSI45630 patches")
    return root


def _execute(bundle: Path, wsi_dir: Path) -> None:
    from eqvae.cli.generate_ubc_cancer_topup import run_topup

    result = run_topup(
        input_root=bundle / "inference",
        wsi_dir=wsi_dir,
        output_root=WORKING_ROOT / "dataset",
        scratch_root=WORKING_ROOT / ".wsi45630_work",
    )
    if result["status"] != "complete" or result["row_count"] != 22649:
        raise RuntimeError("Single-WSI output is incomplete")
    print(json.dumps(result), flush=True)


def main() -> int:
    """Read only 45630.png and publish the new pair in this kernel's output."""
    try:
        os.environ["EQVAE_SESSION_START_MONOTONIC"] = str(time.monotonic())
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
                "input_contract_sha256": INPUT_CONTRACT_SHA256,
                "scope": "WSI45630_missing_latents_only_for_capacity",
            }),
            flush=True,
        )
        candidates = [
            INPUT_ROOT / "competitions/UBC-OCEAN/train_images",
            INPUT_ROOT / "UBC-OCEAN/train_images",
        ]
        roots = [p for p in candidates if (p / "45630.png").is_file()]
        if len(roots) != 1:
            raise RuntimeError("Expected the official WSI45630 source PNG")
        os.environ["EQVAE_CANCER_TOPUP_CONFIRMED"] = "1"
        _execute(bundle, roots[0])
        return 0
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        os.environ.pop("EQVAE_CANCER_TOPUP_CONFIRMED", None)
        os.environ.pop("EQVAE_SESSION_START_MONOTONIC", None)


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


if __name__ == "__main__":
    raise SystemExit(main())
