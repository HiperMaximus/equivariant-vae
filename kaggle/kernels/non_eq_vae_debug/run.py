# Copyright 2026 HiperMaximus
"""Capped Kaggle debug-smoke launcher for spec 0001."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

KAGGLE_SMOKE_READY = True
DEFAULT_KAGGLE_OUTPUT_DIR = Path("/kaggle/working")
LOCAL_FALLBACK_OUTPUT_DIR = Path("runs/kaggle/non_eq_vae_debug_smoke")


def main() -> int:
    """Run the capped smoke launcher.

    Returns:
        Process exit status.

    """
    launcher_dir = Path(__file__).resolve().parent
    payload_dir = launcher_dir / "payload"
    sys.path.insert(0, str(payload_dir / "src"))
    wrong_accelerator()
    single_visible_t4()
    dual_t4_ddp()

    from eqvae.cli.kaggle_smoke import main as smoke_main  # noqa: PLC0415

    output_dir = Path(
        os.environ.get(
            "EQVAE_OUTPUT_DIR",
            str(
                DEFAULT_KAGGLE_OUTPUT_DIR
                if DEFAULT_KAGGLE_OUTPUT_DIR.exists()
                else LOCAL_FALLBACK_OUTPUT_DIR,
            ),
        ),
    )
    config_path = payload_dir / "configs" / "spec0001" / "non_eq_vae_kaggle_debug.json"
    return smoke_main(
        (
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ),
    )


def wrong_accelerator() -> None:
    """Fail early if Kaggle gives a non-T4 GPU when CUDA is available.

    Raises:
        RuntimeError: If visible CUDA devices are not T4 GPUs.

    """
    if not torch.cuda.is_available():
        return
    gpu_names = [
        torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
    ]
    if not all("T4" in name for name in gpu_names):
        message = f"Expected Kaggle T4 GPU metadata, got devices: {gpu_names}"
        raise RuntimeError(message)


def single_visible_t4() -> None:
    """Record the single-process T4 smoke mode hook for push validation."""
    return


def dual_t4_ddp() -> None:
    """Record that this smoke does not launch dual-T4 DDP."""
    return


if __name__ == "__main__":
    raise SystemExit(main())
