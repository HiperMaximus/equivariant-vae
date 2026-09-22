# Copyright 2026 HiperMaximus
"""Thin Kaggle entrypoint for the bounded Spec 0054 A0 probe."""

import subprocess
import sys
from pathlib import Path

REPOSITORY = "https://github.com/HiperMaximus/equivariant-vae.git"
SOURCE_COMMIT = "9ae903f3db2c92258f47feaa331f9bc8d2bc403a"
SOURCE_ROOT = Path("/kaggle/temp/equivariant-vae")


def main() -> int:
    SOURCE_ROOT.mkdir(parents=True, exist_ok=False)
    subprocess.run(["git", "init", str(SOURCE_ROOT)], check=True, timeout=60)
    subprocess.run(
        ["git", "-C", str(SOURCE_ROOT), "remote", "add", "origin", REPOSITORY],
        check=True,
        timeout=60,
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
        ["git", "-C", str(SOURCE_ROOT), "checkout", "--detach", "FETCH_HEAD"],
        check=True,
        timeout=60,
    )
    observed = subprocess.check_output(
        ["git", "-C", str(SOURCE_ROOT), "rev-parse", "HEAD"],
        text=True,
        timeout=60,
    ).strip()
    if observed != SOURCE_COMMIT:
        raise RuntimeError("Fetched source commit differs")
    sys.path[:0] = [str(SOURCE_ROOT), str(SOURCE_ROOT / "src")]
    from experiments.spec0054_mil_a0_probe import run

    return run(repo_root=SOURCE_ROOT, source_commit=observed)


if __name__ == "__main__":
    raise SystemExit(main())
