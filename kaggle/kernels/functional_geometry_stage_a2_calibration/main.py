# Copyright 2026 HiperMaximus
"""Thin Kaggle entrypoint; models are mounted and code comes from GitHub."""

import multiprocessing as mp
import subprocess
import sys
import time
from pathlib import Path

REPOSITORY = "https://github.com/HiperMaximus/equivariant-vae.git"
SOURCE_ROOT = Path("/kaggle/temp/equivariant-vae")


def main() -> int:
    started_at = time.perf_counter()
    subprocess.run(
        [
            "git",
            "clone",
            "--depth=1",
            "--filter=blob:none",
            "--sparse",
            "--branch=main",
            REPOSITORY,
            str(SOURCE_ROOT),
        ],
        check=True,
        timeout=300,
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
            "configs/spec0001",
        ],
        check=True,
        timeout=300,
    )
    commit = subprocess.check_output(
        ["git", "-C", str(SOURCE_ROOT), "rev-parse", "HEAD"],
        text=True,
        timeout=60,
    ).strip()
    sys.path[:0] = [str(SOURCE_ROOT), str(SOURCE_ROOT / "src")]

    from experiments.spec0053_stage_a2_calibration import run

    return run(repo_root=SOURCE_ROOT, source_commit=commit, started_at=started_at)


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
