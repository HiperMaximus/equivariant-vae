"""Paired CPU Kaggle run: cached PANTHER statistics and nested linear heads."""

import json
import os
import subprocess
import sys
from pathlib import Path

REPOSITORY = "https://github.com/HiperMaximus/equivariant-vae.git"
SOURCE_COMMIT = "45f64febece116b80b22214f2676fbffc0e1cd71"
SOURCE_ROOT = Path("/kaggle/temp/equivariant-vae")
WORKING_ROOT = Path("/kaggle/working")


def main():
    SOURCE_ROOT.mkdir(parents=True, exist_ok=False)
    subprocess.run(["git", "init", str(SOURCE_ROOT)], check=True, timeout=60)
    subprocess.run(["git", "-C", str(SOURCE_ROOT), "remote", "add", "origin", REPOSITORY], check=True, timeout=60)
    subprocess.run(["git", "-C", str(SOURCE_ROOT), "fetch", "--depth=1", "origin", SOURCE_COMMIT], check=True, timeout=300)
    subprocess.run(["git", "-C", str(SOURCE_ROOT), "checkout", "--detach", "FETCH_HEAD"], check=True, timeout=60)
    config = json.loads((SOURCE_ROOT / "docs/data/spec0054_classical_wsi.json").read_text())
    recipe = config["panther"]
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = str(recipe["cpu_threads"])
    sys.path.insert(0, str(SOURCE_ROOT))
    from threadpoolctl import threadpool_limits
    from experiments.spec0054_pattern_histograms import run
    from experiments.spec0054_classical_compare import run_pattern_comparison
    caches = {branch: Path(recipe["sources"][branch]["mount"]) for branch in config["branches"]}
    with threadpool_limits(limits=recipe["cpu_threads"]):
        for branch in config["branches"]:
            run(SOURCE_ROOT, caches[branch], WORKING_ROOT / branch, branch, "panther")
        run_pattern_comparison(
            SOURCE_ROOT, caches["normal_vae"], caches["so2_vae"],
            WORKING_ROOT / "normal_vae", WORKING_ROOT / "so2_vae",
            WORKING_ROOT / "classification", "panther",
        )


if __name__ == "__main__":
    main()
