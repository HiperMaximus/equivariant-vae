# Copyright 2026 HiperMaximus
"""Thin gated-ABMIL entry: one independent frozen-VAE comparison per GPU."""

import json
import os
import subprocess
import sys
from pathlib import Path

REPOSITORY = "https://github.com/HiperMaximus/equivariant-vae.git"
SOURCE_COMMIT = "e634fe5f1390a2fd8ade9ff9245c00e74efe2a65"
SOURCE_ROOT = Path("/kaggle/temp/equivariant-vae")
OUTPUT_ROOT = Path("/kaggle/working/spec0054_abmil_fold0")
RESUME_ROOT = None  # Fresh architecture: old local-global weights are incompatible.
SMOKE = True  # Two complete WSIs per GPU before the full fold.
EFFECTIVE_BATCH = 1


def main() -> None:
    print(json.dumps({"event": "source_fetch_start", "commit": SOURCE_COMMIT}), flush=True)
    SOURCE_ROOT.mkdir(parents=True, exist_ok=False)
    subprocess.run(["git", "init", str(SOURCE_ROOT)], check=True)
    subprocess.run(["git", "-C", str(SOURCE_ROOT), "remote", "add", "origin", REPOSITORY], check=True)
    subprocess.run(["git", "-C", str(SOURCE_ROOT), "fetch", "--depth=1", "origin", SOURCE_COMMIT], check=True)
    subprocess.run(["git", "-C", str(SOURCE_ROOT), "checkout", "--detach", "FETCH_HEAD"], check=True)
    print(json.dumps({"event": "source_fetch_done", "commit": SOURCE_COMMIT}), flush=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    workers = []
    for branch, gpu in (("normal_vae", "0"), ("so2_vae", "1")):
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu
        environment["PYTHONPATH"] = f"{SOURCE_ROOT}:{SOURCE_ROOT / 'src'}"
        environment["PYTHONUNBUFFERED"] = "1"
        environment["OMP_NUM_THREADS"] = "1"
        command = [
            sys.executable, "-m", "experiments.spec0054_mil_fold0",
            "--repo-root", str(SOURCE_ROOT), "--latent-root", "/kaggle/input",
            "--output-root", str(OUTPUT_ROOT), "--branch", branch,
            "--effective-batch", str(EFFECTIVE_BATCH),
        ]
        if SMOKE:
            command.append("--smoke")
        if RESUME_ROOT is not None:
            command.extend(["--resume-root", str(RESUME_ROOT)])
        print(json.dumps({"event": "worker_launch", "branch": branch, "physical_gpu": gpu,
                          "smoke": SMOKE, "effective_batch": EFFECTIVE_BATCH}), flush=True)
        workers.append(subprocess.Popen(command, env=environment))
    statuses = [worker.wait() for worker in workers]
    print(json.dumps({"event": "workers_exit", "statuses": statuses}), flush=True)
    for worker, status in zip(workers, statuses, strict=True):
        if status:
            raise subprocess.CalledProcessError(status, worker.args)


if __name__ == "__main__":
    main()
