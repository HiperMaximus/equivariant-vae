# Copyright 2026 HiperMaximus
"""Thin Kaggle entrypoint; models are mounted and code comes from GitHub."""

import multiprocessing as mp
import subprocess
import sys
import time
from pathlib import Path

REPOSITORY = "https://github.com/HiperMaximus/equivariant-vae.git"
SOURCE_COMMIT = "b9879ecb38f5d25cb182ea4868ddce5899ec7241"
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
        timeout=300,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(SOURCE_ROOT),
            "fetch",
            "--depth=1",
            "origin",
            SOURCE_COMMIT,
        ],
        check=True,
        timeout=300,
    )
    subprocess.run(
        ["git", "-C", str(SOURCE_ROOT), "checkout", "--detach", SOURCE_COMMIT],
        check=True,
        timeout=60,
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
    if commit != SOURCE_COMMIT:
        raise RuntimeError(f"source commit mismatch: {commit} != {SOURCE_COMMIT}")
    sys.path[:0] = [str(SOURCE_ROOT), str(SOURCE_ROOT / "src")]

    from experiments.spec0053_stage_a2_calibration import run

    return run(repo_root=SOURCE_ROOT, source_commit=commit, started_at=started_at)


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
