# Copyright 2026 HiperMaximus
"""Brief patch-encoder timing: one frozen VAE's shards per T4 process."""

import json
import os
import subprocess
import sys
from pathlib import Path

REPOSITORY = "https://github.com/HiperMaximus/equivariant-vae.git"
SOURCE_COMMIT = "dc539b6b8e1f1b03682b038f75b8424dc5fbd102"
SOURCE_ROOT = Path("/kaggle/temp/equivariant-vae")
OUTPUT_ROOT = Path("/kaggle/working/spec0054_patch_speed")


def main():
    print(json.dumps({"event": "source_fetch_start", "commit": SOURCE_COMMIT}), flush=True)
    SOURCE_ROOT.mkdir(parents=True)
    subprocess.run(["git", "init", str(SOURCE_ROOT)], check=True)
    subprocess.run(["git", "-C", str(SOURCE_ROOT), "remote", "add", "origin", REPOSITORY], check=True)
    subprocess.run(["git", "-C", str(SOURCE_ROOT), "fetch", "--depth=1", "origin", SOURCE_COMMIT], check=True)
    subprocess.run(["git", "-C", str(SOURCE_ROOT), "checkout", "--detach", "FETCH_HEAD"], check=True)
    OUTPUT_ROOT.mkdir(parents=True)
    (OUTPUT_ROOT / "source.json").write_text(json.dumps({"source_commit": SOURCE_COMMIT}) + "\n")
    workers = []
    for branch, gpu in (("normal_vae", "0"), ("so2_vae", "1")):
        environment = os.environ.copy()
        environment.update({
            "CUDA_VISIBLE_DEVICES": gpu,
            "PYTHONPATH": f"{SOURCE_ROOT}:{SOURCE_ROOT / 'src'}",
            "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            "TORCHINDUCTOR_CACHE_DIR": f"/kaggle/temp/inductor_{branch}",
            "TRITON_CACHE_DIR": f"/kaggle/temp/triton_{branch}",
        })
        command = [
            sys.executable, "-m", "experiments.spec0054_patch_speed_probe",
            "--repo-root", str(SOURCE_ROOT), "--latent-root", "/kaggle/input",
            "--output-root", str(OUTPUT_ROOT), "--branch", branch,
        ]
        print(json.dumps({"event": "worker_launch", "branch": branch, "physical_gpu": gpu}), flush=True)
        workers.append(subprocess.Popen(command, env=environment))
    statuses = [worker.wait() for worker in workers]
    print(json.dumps({"event": "workers_exit", "statuses": statuses}), flush=True)
    for worker, status in zip(workers, statuses, strict=True):
        if status:
            raise subprocess.CalledProcessError(status, worker.args)


if __name__ == "__main__":
    main()
