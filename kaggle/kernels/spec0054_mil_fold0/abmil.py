# Copyright 2026 HiperMaximus
"""Thin gated-ABMIL entry for paired training or CPU-only read probes."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPOSITORY = "https://github.com/HiperMaximus/equivariant-vae.git"
SOURCE_COMMIT = "69ba6951b44f395481a88a624ae5877bdaee963c"
SOURCE_ROOT = Path("/kaggle/temp/equivariant-vae")
OUTPUT_ROOT = Path("/kaggle/working/spec0054_io_probe")
SMOKE = False
RESUME = False
EFFECTIVE_BATCH = 4
READ_PROBE = True
LR_SEARCH = False
LR_CONFIRMATION = False
SESSION_DEADLINE_UNIX = int(time.time()) + 660 * 60  # Read budget starts when Kaggle executes this entry.


def main() -> None:
    print(json.dumps({"event": "source_fetch_start", "commit": SOURCE_COMMIT}), flush=True)
    SOURCE_ROOT.mkdir(parents=True, exist_ok=False)
    subprocess.run(["git", "init", str(SOURCE_ROOT)], check=True)
    subprocess.run(["git", "-C", str(SOURCE_ROOT), "remote", "add", "origin", REPOSITORY], check=True)
    subprocess.run(["git", "-C", str(SOURCE_ROOT), "fetch", "--depth=1", "origin", SOURCE_COMMIT], check=True)
    subprocess.run(["git", "-C", str(SOURCE_ROOT), "checkout", "--detach", "FETCH_HEAD"], check=True)
    print(json.dumps({"event": "source_fetch_done", "commit": SOURCE_COMMIT}), flush=True)
    resume_root = None
    if RESUME:
        resume_identity = next(Path("/kaggle/input").rglob("normal_vae/identity.json"))
        resume_root = resume_identity.parent.parent.parent if LR_CONFIRMATION else resume_identity.parent.parent
        print(json.dumps({"event": "resume_found", "root": str(resume_root)}), flush=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    workers = []
    for branch, gpu in (("normal_vae", "0"), ("so2_vae", "1")):
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = "" if READ_PROBE else gpu
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
        if LR_CONFIRMATION:
            command.append("--lr-confirmation")
        elif LR_SEARCH:
            command.append("--lr-search")
        elif READ_PROBE:
            command.append("--read-probe")
        elif SMOKE:
            command.append("--smoke")
        if resume_root is not None:
            command.extend(["--resume-root", str(resume_root)])
        if SESSION_DEADLINE_UNIX is not None:
            command.extend(["--session-deadline-unix", str(SESSION_DEADLINE_UNIX)])
        print(json.dumps({"event": "worker_launch", "branch": branch, "physical_gpu": None if READ_PROBE else gpu,
                          "smoke": SMOKE and not READ_PROBE, "read_probe": READ_PROBE,
                          "effective_batch": EFFECTIVE_BATCH, "lr_search": LR_SEARCH,
                          "lr_confirmation": LR_CONFIRMATION,
                          "session_deadline_unix": SESSION_DEADLINE_UNIX}), flush=True)
        workers.append(subprocess.Popen(command, env=environment))
    statuses = [worker.wait() for worker in workers]
    print(json.dumps({"event": "workers_exit", "statuses": statuses}), flush=True)
    for worker, status in zip(workers, statuses, strict=True):
        if status:
            raise subprocess.CalledProcessError(status, worker.args)


if __name__ == "__main__":
    main()
