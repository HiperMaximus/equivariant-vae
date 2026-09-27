# Copyright 2026 HiperMaximus
"""Thin Kaggle entrypoint for the resumable Spec 0054 fold-0 run."""

import subprocess
import sys
from pathlib import Path

REPOSITORY = "https://github.com/HiperMaximus/equivariant-vae.git"
SOURCE_COMMIT = "0c9fd5729941a2f817f402b9fc41b820607d801f"
SOURCE_ROOT = Path("/kaggle/temp/equivariant-vae")


def main() -> None:
    SOURCE_ROOT.mkdir(parents=True, exist_ok=False)
    subprocess.run(["git", "init", str(SOURCE_ROOT)], check=True, timeout=60)
    subprocess.run(
        ["git", "-C", str(SOURCE_ROOT), "remote", "add", "origin", REPOSITORY],
        check=True, timeout=60,
    )
    subprocess.run(
        ["git", "-C", str(SOURCE_ROOT), "fetch", "--depth=1", "origin", SOURCE_COMMIT],
        check=True, timeout=300,
    )
    subprocess.run(
        ["git", "-C", str(SOURCE_ROOT), "checkout", "--detach", "FETCH_HEAD"],
        check=True, timeout=60,
    )
    sys.path[:0] = [str(SOURCE_ROOT), str(SOURCE_ROOT / "src")]
    from experiments.spec0054_mil_fold0 import run

    run(
        repo_root=SOURCE_ROOT,
        latent_root=Path("/kaggle/input"),
        output_root=Path("/kaggle/working/spec0054_mil_fold0"),
        resume_root=None,
        effective_batch=1,
    )


if __name__ == "__main__":
    main()
