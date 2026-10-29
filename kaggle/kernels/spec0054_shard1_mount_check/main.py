"""Thin CPU-only shard-1 mount check entry."""
import json
import os
import subprocess
import sys
from pathlib import Path

REPOSITORY = "https://github.com/HiperMaximus/equivariant-vae.git"
SOURCE_COMMIT = "a129a5125a2617576af04e6d4fff52ee8b0d3dfc"
SOURCE_ROOT = Path("/kaggle/temp/equivariant-vae")

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
    environment["PYTHONPATH"] = f"{SOURCE_ROOT}:{SOURCE_ROOT / 'src'}"
    subprocess.run([sys.executable, "-m", "experiments.spec0054_shard1_mount_check"], env=environment, check=True)

if __name__ == "__main__":
    main()
