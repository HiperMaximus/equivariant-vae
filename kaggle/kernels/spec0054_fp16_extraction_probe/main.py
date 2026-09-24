# Copyright 2026 HiperMaximus
"""Thin Kaggle entrypoint for the fifth latent shard."""

import subprocess
import sys
from pathlib import Path

REPOSITORY = "https://github.com/HiperMaximus/equivariant-vae.git"
SOURCE_ROOT = Path("/kaggle/temp/equivariant-vae")


def main() -> int:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "pyvips[binary]==3.1.0",
        ],
        check=True,
        timeout=300,
    )
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
    from experiments.spec0054_fp16_latent_extraction import run

    return run(
        repo_root=SOURCE_ROOT,
        source_commit=commit,
        shard_index=5,
    )


if __name__ == "__main__":
    raise SystemExit(main())
