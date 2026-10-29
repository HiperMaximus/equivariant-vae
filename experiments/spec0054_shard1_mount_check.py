"""CPU-only availability check and bounded WSI comparison of shard-1 mounts."""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

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


def run_benchmark(repo, root, output, deadline_unix=None):
    """Compare paired source arms with natural cache and production CPU inputs."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch

    from experiments.spec0054_mil_fold0 import (
        BRANCHES, _prefetch_training_inputs, _sample_indices,
    )
    from experiments.spec0054_shard_bags import ShardBags
    from eqvae.training.supervised_pairing import paired_epoch_order

    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    started = time.perf_counter()
    deadline = deadline_unix if deadline_unix is not None else time.time() + 360 * 60
    directory = output.parent
    events = (directory / "shard1_source_benchmark_events.jsonl").open("w", buffering=1)
    config = json.loads((repo / "docs/data/spec0054_fold0_run.json").read_text())
    extraction = json.loads((repo / "docs/data/spec0054_fp16_latent_extraction.json").read_text())
    shard = extraction["shards"][0]
    with (repo / "docs/data/spec0054_cohort_folds.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        cohort = list(reader)
        fields = list(cohort[0])
    panel = [row for row in cohort if shard["first_wsi"] <= int(row["wsi_id"]) <= shard["last_wsi"]]
    cohort_path = directory / "shard1_source_benchmark_cohort.csv"
    with cohort_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(panel)
    all_train = sorted(int(row["wsi_id"]) for row in cohort if int(row["fold"]) != config["fold"])
    train = sorted(int(row["wsi_id"]) for row in panel if int(row["fold"]) != config["fold"])
    roots = {
        "dataset": root / "datasets/maximusshtefan/eqvae-spec0054-frozen-latent-shard-1",
        "kernel_output": root / "notebooks/maximshtefan/eqvae-spec0054-fp16-latent-shard-1",
    }
    bags = {kind: ShardBags(root, {**extraction, "shards": [shard]}, cohort_path,
                            {1: config["sources"][0]["result_sha256"]})
            for kind, root in roots.items()}
    source_order = ("dataset", "kernel_output", "kernel_output", "dataset")
    result = {
        "source_commit": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
        "gpu_used": False, "host_pinning": False, "h2d_measured": False,
        "scope": "All shard1 fold0-training WSIs; full bags then matched 60% sampled C4 inputs at epochs5/6/7. No sealed WSIs.",
        "cache_policy": "Natural cache; no global resets. Sampled reads retain production WILLNEED/DONTNEED. First panel access is not a cold-cache claim.",
        "timing_scope": "Read/prepare timers exclude consumer hashes; window/arm/session wall and process CPU include hashing and logging. Branch windows overlap; process CPU is measured only for whole arms.",
        "deadline_policy": "Checked before scheduling and between consumed reads; blocking reads and draining up to eight pending futures per branch can overrun.",
        "deadline_unix": deadline, "source_order": source_order, "sampled_epochs": [5, 6, 7],
        "seed": config["seed"], "patch_retention": config["patch_retention"],
        "c4_augmentation": config["c4_augmentation"], "training_io": config["training_io"],
        "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
        "logical_cpu_count": os.cpu_count(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "training_wsi_ids": train, "panel_paired_bytes_per_source": sum(bags["dataset"].locations[wsi].count for wsi in train) * RECORD_BYTES * 2,
        "files": {}, "branches": {branch: {"reads": [], "windows": []} for branch in BRANCHES},
        "arms": [], "completed": False,
    }
    for kind, reader in bags.items():
        result_path = next(roots[kind].rglob("result_shard_01_of_12.json"))
        metadata = json.loads(result_path.read_text())
        result["files"][kind] = {
            "root": str(roots[kind]), "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
            "index_sha256": metadata["files"]["index"]["sha256"],
            "binaries": {branch: {"path": str(reader.files[branch][1]),
                                   "bytes": reader.files[branch][1].stat().st_size,
                                   "device": reader.files[branch][1].stat().st_dev,
                                   "inode": reader.files[branch][1].stat().st_ino}
                         for branch in BRANCHES},
        }
    lock = Lock()

    def save():
        # Call only under lock, including every mutation of the shared result.
        output.write_text(json.dumps(result, indent=2) + "\n")

    def emit(event, **row):
        with lock:
            payload = {"event": event, "session_seconds": time.perf_counter() - started, **row}
            events.write(json.dumps(payload) + "\n")
            print(json.dumps(payload), flush=True)

    def memory():
        return {"meminfo_kib": {key: int(value.split()[0]) for key, value in
                                (line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
                                if key in ("MemTotal", "MemAvailable", "Cached")},
                "process_kib": {key: int(value.split()[0]) for key, value in
                                (line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines() if ":" in line)
                                if key in ("VmRSS", "VmHWM")}}

    def run_branch(kind, branch, arm, visit):
        reader = bags[kind]
        done = True
        for phase_index, epoch in enumerate((None, 5, 6, 7)):
            if time.time() >= deadline:
                return False
            phase = "full" if epoch is None else "sampled"
            context = {"source": kind, "branch": branch, "arm": arm, "source_visit": visit,
                       "source_first_arm": visit == 0, "phase": phase, "epoch": epoch,
                       "panel_pass_index": visit * 4 + phase_index}
            metrics = {}

            def log(event, **row):
                if event == "bag_transfer_start":
                    metrics[row["wsi_id"]] = row
                emit(event, **context, **row)

            begin = time.perf_counter()
            emit("window_start", **context, **memory())
            consumed = 0
            hash_seconds = 0.0
            if epoch is None:
                for wsi_id in train:
                    if time.time() >= deadline:
                        break
                    emit("full_read_start", **context, wsi_id=wsi_id)
                    read_start, cpu_start = time.perf_counter(), time.thread_time()
                    array = reader.read(branch, wsi_id)
                    timing = {"read_seconds": time.perf_counter() - read_start,
                              "read_cpu_seconds": time.thread_time() - cpu_start}
                    hash_start = time.perf_counter()
                    digest = hashlib.sha256(memoryview(array).cast("B")).hexdigest()
                    row = {**context, "wsi_id": wsi_id, "raw_sha256": digest,
                           "selected_bytes": array.nbytes, "bag_size": len(array), **timing,
                           "hash_seconds": time.perf_counter() - hash_start}
                    del array
                    hash_seconds += row["hash_seconds"]
                    with lock:
                        result["branches"][branch]["reads"].append(row)
                        save()
                    emit("read_verified", **row)
                    consumed += 1
            else:
                sample_epoch = int(epoch)
                order = [all_train[position] for position in paired_epoch_order(
                    row_count=len(all_train), epoch=epoch, seed=config["seed"])
                    if all_train[position] in reader.locations and all_train[position] in train]

                def requests():
                    for wsi_id in order:
                        if time.time() >= deadline:
                            return
                        yield sample_epoch, wsi_id, _sample_indices(reader.locations[wsi_id].count,
                                                            config["seed"], sample_epoch, wsi_id,
                                                            config["patch_retention"])

                with _prefetch_training_inputs(reader, branch, torch.device("cpu"), config, requests(), log) as inputs:
                    for wsi_id, indices, latents in inputs:
                        hash_start = time.perf_counter()
                        # NHWC is the contiguous physical storage of channels-last tensors.
                        physical = latents.permute(0, 2, 3, 1).numpy()
                        digest = hashlib.sha256(memoryview(physical).cast("B")).hexdigest()
                        row = {**context, **metrics.pop(wsi_id), "prepared_physical_sha256": digest,
                               "prepared_layout": "NHWC contiguous FP16", "bag_size": reader.locations[wsi_id].count,
                               "hash_seconds": time.perf_counter() - hash_start}
                        hash_seconds += row["hash_seconds"]
                        del physical, latents
                        with lock:
                            result["branches"][branch]["reads"].append(row)
                            save()
                        emit("read_verified", **row)
                        consumed += 1
                        if time.time() >= deadline:
                            break
            window = {**context, "wall_seconds_including_hashing": time.perf_counter() - begin,
                      "consumer_hash_seconds": hash_seconds, "consumed_reads": consumed,
                      "planned_reads": len(train), "completed": consumed == len(train)}
            with lock:
                result["branches"][branch]["windows"].append(window)
                save()
            emit("window_done", **window, **memory())
            done = done and window["completed"]
        return done

    visits = {kind: 0 for kind in roots}
    with lock:
        save()
    emit("benchmark_start", **memory(), training_wsi_ids=train, source_order=source_order)
    try:
        for arm, kind in enumerate(source_order):
            if time.time() >= deadline:
                break
            arm_start, cpu_start = time.perf_counter(), time.process_time()
            emit("arm_start", arm=arm, source=kind, source_visit=visits[kind], **memory())
            with ThreadPoolExecutor(max_workers=2) as workers:
                futures = [workers.submit(run_branch, kind, branch, arm, visits[kind]) for branch in BRANCHES]
                completed = [future.result() for future in futures]
            row = {"arm": arm, "source": kind, "source_visit": visits[kind],
                   "wall_seconds_including_hashing": time.perf_counter() - arm_start,
                   "process_cpu_seconds_including_hashing": time.process_time() - cpu_start,
                   "completed": all(completed)}
            visits[kind] += 1
            with lock:
                result["arms"].append(row)
                save()
            emit("arm_done", **row, **memory())
    finally:
        for reader in bags.values():
            reader.close()
        events.close()
    with lock:
        result["completed"] = len(result["arms"]) == len(source_order) and all(row["completed"] for row in result["arms"])
        result["session_seconds_including_hashing"] = time.perf_counter() - started
        result["deadline_reached"] = time.time() >= deadline
        save()
    print(json.dumps({"event": "benchmark_done", "completed": result["completed"], "output": str(output)}), flush=True)
    return result


def benchmark(deadline_unix=None):
    return run_benchmark(Path(__file__).resolve().parents[1], Path("/kaggle/input"),
                         Path("/kaggle/working/shard1_source_benchmark.json"), deadline_unix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--deadline-unix", type=float)
    arguments = parser.parse_args()
    if arguments.benchmark:
        benchmark(arguments.deadline_unix)
    else:
        main()
