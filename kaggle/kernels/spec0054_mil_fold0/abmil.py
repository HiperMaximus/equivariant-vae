# Copyright 2026 HiperMaximus
"""Thin gated-ABMIL entry: one independent frozen-VAE comparison per GPU."""

import json
import os
import subprocess
import sys
from pathlib import Path

REPOSITORY = "https://github.com/HiperMaximus/equivariant-vae.git"
SOURCE_COMMIT = "68a835c79f9d4120cd83a3f72e9db8314a5b0bce"
SOURCE_ROOT = Path("/kaggle/temp/equivariant-vae")
OUTPUT_ROOT = Path("/kaggle/working/spec0054_abmil_fold0")
RESUME_ROOT = None  # Fresh architecture: old local-global weights are incompatible.
SMOKE = True  # Eight updates; restore checkpoint after four. No full fold.
EFFECTIVE_BATCH = 4
READ_PROBE = True  # Ten warm-cache reads per VAE; no training or GPU work.


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
        environment["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
        environment["TORCHINDUCTOR_CACHE_DIR"] = f"/kaggle/temp/inductor_{branch}"
        environment["TRITON_CACHE_DIR"] = f"/kaggle/temp/triton_{branch}"
        command = [
            sys.executable, "-m", "experiments.spec0054_mil_fold0",
            "--repo-root", str(SOURCE_ROOT), "--latent-root", "/kaggle/input",
            "--output-root", str(OUTPUT_ROOT), "--branch", branch,
            "--effective-batch", str(EFFECTIVE_BATCH),
        ]
        if READ_PROBE:
            command.append("--read-probe")
        elif SMOKE:
            command.append("--smoke")
        if RESUME_ROOT is not None:
            command.extend(["--resume-root", str(RESUME_ROOT)])
        print(json.dumps({"event": "worker_launch", "branch": branch, "physical_gpu": gpu,
                          "smoke": SMOKE and not READ_PROBE, "read_probe": READ_PROBE,
                          "effective_batch": EFFECTIVE_BATCH}), flush=True)
        workers.append(subprocess.Popen(command, env=environment))
    statuses = [worker.wait() for worker in workers]
    print(json.dumps({"event": "workers_exit", "statuses": statuses}), flush=True)
    for worker, status in zip(workers, statuses, strict=True):
        if status:
            raise subprocess.CalledProcessError(status, worker.args)


if __name__ == "__main__":
    main()
