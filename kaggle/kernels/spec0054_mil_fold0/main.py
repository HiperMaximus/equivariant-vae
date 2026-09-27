# Copyright 2026 HiperMaximus
"""Thin Kaggle entrypoint for the resumable Spec 0054 fold-0 run."""

import os
import subprocess
import sys
from pathlib import Path

REPOSITORY = "https://github.com/HiperMaximus/equivariant-vae.git"
SOURCE_COMMIT = "0c9fd5729941a2f817f402b9fc41b820607d801f"
SOURCE_ROOT = Path("/kaggle/temp/equivariant-vae")


def main() -> None:
    SOURCE_ROOT.mkdir(parents=True, exist_ok=False)
    subprocess.run(["git", "init", str(SOURCE_ROOT)], check=True, timeout=60)
    subprocess.run(
        ["git", "-C", str(SOURCE_ROOT), "remote", "add", "origin", REPOSITORY],
        check=True, timeout=60,
    )
    subprocess.run(
        ["git", "-C", str(SOURCE_ROOT), "fetch", "--depth=1", "origin", SOURCE_COMMIT],
        check=True, timeout=300,
    )
    subprocess.run(
        ["git", "-C", str(SOURCE_ROOT), "checkout", "--detach", "FETCH_HEAD"],
        check=True, timeout=60,
    )
    resume_identity = next(Path("/kaggle/input").rglob("normal_vae/identity.json"))
    resume_root = resume_identity.parent.parent
    worker_code = """
import os
import sys
from pathlib import Path

import torch
import experiments.spec0054_mil_fold0 as fold

print("branch", sys.argv[1], "physical_gpu", os.environ["CUDA_VISIBLE_DEVICES"],
      "visible_gpu_count", torch.cuda.device_count(), flush=True)
fold.BRANCHES = (sys.argv[1],)
fold.run(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]),
         Path(sys.argv[5]), 1)
"""
    workers = []
    for branch, gpu in (("normal_vae", "0"), ("so2_vae", "1")):
        output_root = Path("/kaggle/working") / f"spec0054_mil_fold0_{branch}"
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu
        environment["PYTHONPATH"] = f"{SOURCE_ROOT}:{SOURCE_ROOT / 'src'}"
        environment["PYTHONUNBUFFERED"] = "1"
        workers.append(subprocess.Popen(
            [sys.executable, "-c", worker_code, branch, str(SOURCE_ROOT),
             "/kaggle/input", str(output_root), str(resume_root)],
            env=environment,
        ))
    statuses = [worker.wait() for worker in workers]
    for worker, status in zip(workers, statuses, strict=True):
        if status:
            raise subprocess.CalledProcessError(status, worker.args)


if __name__ == "__main__":
    main()
