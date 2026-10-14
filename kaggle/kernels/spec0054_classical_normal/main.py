"""Thin CPU entrypoint for one branch of the classical WSI baseline."""

import subprocess
import sys
from pathlib import Path

REPOSITORY = "https://github.com/HiperMaximus/equivariant-vae.git"
SOURCE_COMMIT = "12d7c5a01dc9474314ca5796ed65f3e2fe8d415d"
SOURCE_ROOT = Path("/kaggle/temp/equivariant-vae")
BRANCH = "normal_vae"


def main():
    SOURCE_ROOT.mkdir(parents=True, exist_ok=False)
    subprocess.run(["git", "init", str(SOURCE_ROOT)], check=True, timeout=60)
    subprocess.run(["git", "-C", str(SOURCE_ROOT), "remote", "add", "origin", REPOSITORY], check=True, timeout=60)
    subprocess.run(["git", "-C", str(SOURCE_ROOT), "fetch", "--depth=1", "origin", SOURCE_COMMIT], check=True, timeout=300)
    subprocess.run(["git", "-C", str(SOURCE_ROOT), "checkout", "--detach", "FETCH_HEAD"], check=True, timeout=60)
    sys.path.insert(0, str(SOURCE_ROOT))
    from experiments.spec0054_classical_extract import run
    run(SOURCE_ROOT, Path("/kaggle/input"), Path("/kaggle/working") / BRANCH, BRANCH)


if __name__ == "__main__":
    main()
