"""Thin CPU entrypoint for train-only H2T prototypes and conditional patch means."""

import json
import os
import subprocess
import sys
from pathlib import Path

REPOSITORY = "https://github.com/HiperMaximus/equivariant-vae.git"
SOURCE_COMMIT = "6dd8a49d9136ce86bc4fb1463644f36f3c57ec84"
SOURCE_ROOT = Path("/kaggle/temp/equivariant-vae")
BRANCH = "normal_vae"


def main():
    SOURCE_ROOT.mkdir(parents=True, exist_ok=False)
    subprocess.run(["git", "init", str(SOURCE_ROOT)], check=True, timeout=60)
    subprocess.run(["git", "-C", str(SOURCE_ROOT), "remote", "add", "origin", REPOSITORY], check=True, timeout=60)
    subprocess.run(["git", "-C", str(SOURCE_ROOT), "fetch", "--depth=1", "origin", SOURCE_COMMIT], check=True, timeout=300)
    subprocess.run(["git", "-C", str(SOURCE_ROOT), "checkout", "--detach", "FETCH_HEAD"], check=True, timeout=60)
    config = json.loads((SOURCE_ROOT / "docs/data/spec0054_classical_wsi.json").read_text())
    recipe = config["h2t"]
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = str(recipe["cpu_threads"])
    sys.path.insert(0, str(SOURCE_ROOT))
    from experiments.spec0054_pattern_histograms import run
    run(SOURCE_ROOT, Path(recipe["sources"][BRANCH]["mount"]),
        Path("/kaggle/working") / BRANCH, BRANCH, "h2t")


if __name__ == "__main__":
    main()
