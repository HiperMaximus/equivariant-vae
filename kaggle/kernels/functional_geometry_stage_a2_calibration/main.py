# Copyright 2026 HiperMaximus
# ruff: noqa: D103, INP001, PLC0415, S404, S603, S607
"""Thin Kaggle entrypoint; models are mounted and code comes from GitHub."""

from __future__ import annotations

import multiprocessing as mp
import subprocess
import sys
import time
from pathlib import Path

REPOSITORY = "https://github.com/HiperMaximus/equivariant-vae.git"
SOURCE_ROOT = Path("/kaggle/working/equivariant-vae")


def main() -> int:
    started_at = time.perf_counter()
    subprocess.run(
        [
            "git",
            "clone",
            "--depth=1",
            "--filter=blob:none",
            "--sparse",
            REPOSITORY,
            str(SOURCE_ROOT),
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(SOURCE_ROOT),
            "sparse-checkout",
            "set",
            "src",
            "experiments",
            "docs/data",
            "runs/kaggle/fixed25_selector",
        ],
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "-C", str(SOURCE_ROOT), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    sys.path[:0] = [str(SOURCE_ROOT), str(SOURCE_ROOT / "src")]

    from experiments.spec0053_stage_a2_calibration import run

    return run(repo_root=SOURCE_ROOT, source_commit=commit, started_at=started_at)


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
