"""CPU descriptors from frozen historical Otsu bags; no VAE inference or fitting."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from experiments.spec0054_shard_bags import ShardBags


def pool_patches(latents: np.ndarray, grid: int) -> np.ndarray:
    """Channel-major block averages, accumulated in FP32 before flattening."""
    block = latents.shape[-1] // grid
    return latents.reshape(len(latents), 16, grid, block, grid, block).mean(
        axis=(3, 5), dtype=np.float32,
    ).reshape(len(latents), -1)


def merge_moments(count, mean, m2, values):
    """Parallel Welford merge in FP64; M2 is the sum of squared deviations."""
    values = values.astype(np.float64)
    batch_count = len(values)
    batch_mean = values.mean(axis=0)
    batch_m2 = ((values - batch_mean) ** 2).sum(axis=0)
    delta = batch_mean - mean
    total = count + batch_count
    return total, mean + delta * (batch_count / total), (
        m2 + batch_m2 + delta ** 2 * (count * batch_count / total)
    )


def read_chunks(bags, branch, wsi_id, chunk_size, executor):
    """One producer overlaps the next contiguous read with CPU reductions."""
    count = bags.locations[wsi_id].count
    chunks = iter(range(0, count, chunk_size))
    first = next(chunks)
    pending = executor.submit(
        bags.read, branch, wsi_id, np.arange(first, min(first + chunk_size, count)),
    )
    for first in chunks:
        values = pending.result()
        pending = executor.submit(
            bags.read, branch, wsi_id, np.arange(first, min(first + chunk_size, count)),
        )
        yield values
    yield pending.result()


def run(repo_root: Path, latent_root: Path, output_root: Path, branch: str):
    started = time.monotonic()
    config_path = repo_root / "docs/data/spec0054_classical_wsi.json"
    config = json.loads(config_path.read_text())
    extraction_path = repo_root / "docs/data/spec0054_fp16_latent_extraction.json"
    extraction = json.loads(extraction_path.read_text())
    source_config = json.loads((repo_root / "docs/data/spec0054_fold0_run.json").read_text())
    cohort_path = repo_root / "docs/data/spec0054_cohort_folds.csv"
    bags = ShardBags(latent_root, extraction, cohort_path, {
        row["shard"]: row["result_sha256"] for row in source_config["sources"]
    })
    output_root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, output_root / config_path.name)
    shutil.copy2(cohort_path, output_root / cohort_path.name)
    grids = config["pool_grids"]
    ids = np.array(sorted(bags.locations), dtype=np.int64)
    tables = {grid: np.empty((len(ids), 2 * 16 * grid ** 2), np.float32) for grid in grids}
    cache_grid = config["patch_cache_grid"]
    total_rows = sum(location.count for location in bags.locations.values())
    cache_path = output_root / f"patches_grid{cache_grid}.npy"
    cache = np.lib.format.open_memmap(
        cache_path, mode="w+", dtype=config["patch_cache_dtype"],
        shape=(total_rows, 16 * cache_grid ** 2),
    )
    cursor = 0
    offsets = []
    labels = []
    folds = []
    with ThreadPoolExecutor(max_workers=1) as executor, (
        (output_root / "patch_index.csv").open("w", newline="")
    ) as index_handle:
        writer = csv.writer(index_handle)
        writer.writerow(["patch_row", "wsi_id", "shard", "file_index", "x", "y"])
        try:
            for row, wsi_id in enumerate(ids.tolist()):
                wsi_started = time.monotonic()
                location = bags.locations[wsi_id]
                offsets.append(cursor)
                labels.append(location.diagnosis_index)
                folds.append(location.fold)
                moments = {grid: (0, np.zeros(16 * grid ** 2), np.zeros(16 * grid ** 2))
                           for grid in grids}
                for latents in read_chunks(bags, branch, wsi_id, config["chunk_patches"], executor):
                    for grid in grids:
                        descriptor = pool_patches(latents, grid)
                        moments[grid] = merge_moments(*moments[grid], descriptor)
                        if grid == cache_grid:
                            cache[cursor:cursor + len(latents)] = descriptor
                    cursor += len(latents)
                for grid in grids:
                    count, mean, m2 = moments[grid]
                    tables[grid][row] = np.concatenate((mean, np.sqrt(m2 / count)))
                for within_wsi, (x, y) in enumerate(location.coordinates):
                    writer.writerow([offsets[-1] + within_wsi, wsi_id, location.shard,
                                     location.first_record + within_wsi, x, y])
                print(json.dumps({"event": "wsi_done", "branch": branch, "wsi_id": wsi_id,
                                  "wsi_done": row + 1, "patches_done": cursor,
                                  "seconds": time.monotonic() - wsi_started}), flush=True)
        finally:
            bags.close()
    cache.flush()
    del cache
    np.savez(output_root / "wsi_features.npz", wsi_id=ids,
             diagnosis_index=np.array(labels), fold=np.array(folds),
             patch_offset=np.array(offsets),
             patch_count=np.array([bags.locations[int(wsi)].count for wsi in ids]),
             **{f"grid{grid}": table for grid, table in tables.items()})
    identity = {
        "branch": branch,
        "source_commit": subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True,
        ).strip(),
        "config": config,
        "cohort_sha256": hashlib.sha256(cohort_path.read_bytes()).hexdigest(),
        "sources": source_config["sources"],
        "representation": "frozen_posterior_mu_block_mean_then_wsi_mean_population_std",
        "foreground": "frozen_historical_otsu_population_not_exhaustive_tissue",
        "patch_rows": cursor,
        "wsi_rows": len(ids),
        "patch_cache_bytes": cache_path.stat().st_size,
        "elapsed_seconds": time.monotonic() - started,
        "numpy_version": np.__version__,
    }
    (output_root / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    (output_root / "dataset-metadata.json").write_text(json.dumps({
        "id": config["output_datasets"][branch],
        "title": f"eqvae spec0054 classical {branch}",
        "licenses": [{"name": "CC0-1.0"}],
        "description": "Derived frozen VAE descriptors for 361 development WSI; private research artifact.",
    }, indent=2) + "\n")
    print(json.dumps({"event": "extraction_done", **identity}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--branch", choices=("normal_vae", "so2_vae"), required=True)
    args = parser.parse_args()
    run(args.repo_root, args.latent_root, args.output_root, args.branch)
