"""Thin CPU-only shard-1 source comparison entry."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPOSITORY = "https://github.com/HiperMaximus/equivariant-vae.git"
SOURCE_COMMIT = "88cfffb68b1947f032bcc8b5bbbf1d379f4ed570"
SOURCE_ROOT = Path("/kaggle/temp/equivariant-vae")
SESSION_DEADLINE_UNIX = int(time.time()) + 360 * 60

def main():
    print(json.dumps({"event": "source_fetch_start", "commit": SOURCE_COMMIT}), flush=True)
    SOURCE_ROOT.mkdir(parents=True, exist_ok=False)
    subprocess.run(["git", "init", str(SOURCE_ROOT)], check=True)
    subprocess.run(["git", "-C", str(SOURCE_ROOT), "remote", "add", "origin", REPOSITORY], check=True)
    subprocess.run(["git", "-C", str(SOURCE_ROOT), "fetch", "--depth=1", "origin", SOURCE_COMMIT], check=True)
    subprocess.run(["git", "-C", str(SOURCE_ROOT), "checkout", "--detach", "FETCH_HEAD"], check=True)
    print(json.dumps({"event": "source_fetch_done", "commit": SOURCE_COMMIT}), flush=True)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["PYTHONPATH"] = f"{SOURCE_ROOT}:{SOURCE_ROOT / 'src'}"
    subprocess.run([sys.executable, "-m", "experiments.spec0054_shard1_mount_check",
                    "--benchmark", "--deadline-unix", str(SESSION_DEADLINE_UNIX)], env=environment, check=True)

if __name__ == "__main__":
    main()
