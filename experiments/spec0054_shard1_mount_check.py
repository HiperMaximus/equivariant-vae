"""CPU-only availability check of the two frozen shard-1 input locators."""

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from experiments.spec0054_shard_bags import RECORD_BYTES


def main():
    root = Path("/kaggle/input")
    output = Path("/kaggle/working/shard1_mount_check.json")
    repo = Path(__file__).resolve().parents[1]
    expected = json.loads((repo / "docs/data/spec0054_fold0_run.json").read_text())["sources"][0]
    sources = {
        "dataset": "eqvae-spec0054-frozen-latent-shard-1",
        "kernel_output": "eqvae-spec0054-fp16-latent-shard-1",
    }
    result = {
        "source_commit": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
        "gpu_used": False,
        "scope": "Mount, frozen metadata and three 32KiB records twice per branch/source; not a WSI-throughput benchmark.",
        "files": {}, "reads": [], "completed": False,
    }
    for kind, slug in sources.items():
        path = next(p for p in root.rglob("result_shard_01_of_12.json") if slug in p.parts)
        payload = path.read_bytes()
        metadata = json.loads(payload)
        index = path.parent / metadata["files"]["index"]["name"]
        result["files"][kind] = {
            "path": str(path.parent),
            "result_sha256": hashlib.sha256(payload).hexdigest(),
            "frozen_result_equal": hashlib.sha256(payload).hexdigest() == expected["result_sha256"],
            "index_hash_equal": hashlib.sha256(index.read_bytes()).hexdigest() == metadata["files"]["index"]["sha256"],
            "binary_sizes_equal": {},
        }
        for branch in ("normal_vae", "so2_vae"):
            binary = path.parent / metadata["files"][branch]["name"]
            result["files"][kind]["binary_sizes_equal"][branch] = binary.stat().st_size == metadata["files"][branch]["bytes"]
            descriptor = os.open(binary, os.O_RDONLY)
            try:
                for repeat in range(2):
                    for record in (0, metadata["rows_written"] // 2, metadata["rows_written"] - 1):
                        started = time.perf_counter()
                        data = os.pread(descriptor, RECORD_BYTES, record * RECORD_BYTES)
                        row = {"source": kind, "branch": branch, "repeat": repeat, "record": record,
                               "seconds": time.perf_counter() - started, "bytes": len(data),
                               "sha256": hashlib.sha256(data).hexdigest()}
                        result["reads"].append(row)
                        print(json.dumps(row), flush=True)
                        output.write_text(json.dumps(result, indent=2) + "\n")
            finally:
                os.close(descriptor)
    result["record_hashes_equal"] = all(
        len({r["sha256"] for r in result["reads"] if r["branch"] == branch and r["record"] == record}) == 1
        for branch in ("normal_vae", "so2_vae") for record in {r["record"] for r in result["reads"]}
    )
    result["all_read_lengths_equal"] = all(r["bytes"] == RECORD_BYTES for r in result["reads"])
    result["completed"] = True
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"event": "mount_check_done", "files": result["files"],
                      "record_hashes_equal": result["record_hashes_equal"]}), flush=True)


if __name__ == "__main__":
    main()
