# Copyright 2026 HiperMaximus
"""Compare three AMP batch sizes on the same real WSI patches."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.spec0054_fp16_latent_extraction import (
    _encoder_function,
    _iter_patch_batches,
    _load_shard_rows,
    _materialize_so2_encoder,
    _payload,
    _wsi_directory,
)
from experiments.spec0054_mil_a0_probe import _load_frozen_model, _sha256, _unique_input

SAMPLE_PATCHES = 512
WARMUP_PASSES = 10
REPEATS = 5
CASES = (
    ("amp_b32", 32),
    ("amp_b48", 48),
    ("amp_b64", 64),
)
OUTPUT = Path("/kaggle/working/spec0054_fp16_inference_speed_probe.json")


def _run_case(
    images: np.ndarray[Any, np.dtype[np.uint8]],
    models: dict[str, Any],
    *,
    batch_size: int,
    torch: Any,
) -> dict[str, Any]:
    functions = {
        name: _encoder_function(model, torch) for name, model in models.items()
    }
    devices = {"normal_vae": "cuda:0", "so2_vae": "cuda:1"}

    def batches(names: tuple[str, ...]) -> None:
        with (
            torch.inference_mode(),
            torch.autocast("cuda", dtype=torch.float16),
        ):
            for start in range(0, len(images), batch_size):
                source = torch.from_numpy(
                    images[start : start + batch_size]
                ).pin_memory()
                latents = {
                    name: functions[name](source.to(devices[name], non_blocking=True))
                    for name in names
                }
                for name, latent in latents.items():
                    stored = latent.to(device="cpu", dtype=torch.float16).contiguous()
                    _payload(stored)

    compile_started = time.perf_counter()
    with (
        torch.inference_mode(),
        torch.autocast("cuda", dtype=torch.float16),
    ):
        remainder = len(images) % batch_size
        compile_batch_sizes = (batch_size, remainder) if remainder else (batch_size,)
        for size in compile_batch_sizes:
            warm = torch.from_numpy(images[:size]).pin_memory()
            for name in models:
                functions[name](warm.to(devices[name], non_blocking=True))
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)
    compile_seconds = time.perf_counter() - compile_started

    warmup_started = time.perf_counter()
    paired_names = tuple(models)
    for _ in range(WARMUP_PASSES):
        batches(paired_names)
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)
    warmup_seconds = time.perf_counter() - warmup_started

    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.reset_peak_memory_stats(1)
    seconds: list[float] = []
    for _ in range(REPEATS):
        started = time.perf_counter()
        batches(paired_names)
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)
        seconds.append(time.perf_counter() - started)
    per_model_seconds: dict[str, list[float]] = {}
    for index, name in enumerate(models):
        observations: list[float] = []
        for _ in range(2):
            started = time.perf_counter()
            batches((name,))
            torch.cuda.synchronize(index)
            observations.append(time.perf_counter() - started)
        per_model_seconds[name] = observations
    return {
        "batch_size": batch_size,
        "amp_fp16": True,
        "compile_and_initial_batch_seconds": compile_seconds,
        "ten_full_warmup_passes_seconds": warmup_seconds,
        "repeat_seconds": seconds,
        "median_patches_per_second": len(images) / statistics.median(seconds),
        "per_model_repeat_seconds": per_model_seconds,
        "per_model_median_patches_per_second": {
            name: len(images) / statistics.median(values)
            for name, values in per_model_seconds.items()
        },
        "peak_allocated_bytes": {
            name: torch.cuda.max_memory_allocated(index)
            for index, name in enumerate(models)
        },
    }


def run(*, repo_root: Path, source_commit: str) -> int:
    import torch

    contract_path = repo_root / "docs/data/spec0054_fp16_latent_extraction.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    atlas = _unique_input("raw_atlas.csv", contract["atlas"]["sha256"])
    weights = {
        "normal_vae": _unique_input(
            "normal_vae_state.pt", contract["weights"]["normal_vae_sha256"]
        ),
        "so2_vae": _unique_input(
            "so2_vae_state.pt", contract["weights"]["so2_vae_sha256"]
        ),
    }
    rows = _load_shard_rows(atlas, contract["shards"][0])[:SAMPLE_PATCHES]
    read_started = time.perf_counter()
    images = np.concatenate(
        [batch for _, batch, _ in _iter_patch_batches(rows, _wsi_directory())]
    )
    read_seconds = time.perf_counter() - read_started
    models = {
        "normal_vae": _load_frozen_model(
            "normal_vae", weights["normal_vae"], torch.device("cuda:0"), torch
        ),
        "so2_vae": _load_frozen_model(
            "so2_vae", weights["so2_vae"], torch.device("cuda:1"), torch
        ),
    }
    materialized_count = _materialize_so2_encoder(models["so2_vae"])
    observations: dict[str, Any] = {}
    for name, batch_size in CASES:
        torch.compiler.reset()
        timing = _run_case(images, models, batch_size=batch_size, torch=torch)
        observations[name] = timing
        print(json.dumps({"case": name, **timing}, sort_keys=True), flush=True)
    result = {
        "schema_version": "spec0054.fp16_inference_speed_probe.v2",
        "source_commit": source_commit,
        "contract_sha256": _sha256(contract_path),
        "sample_wsi_id": rows[0].wsi_id,
        "sample_atlas_row_indices": [rows[0].atlas_row_index, rows[-1].atlas_row_index],
        "sample_patches": len(images),
        "warmup_full_passes_per_case": WARMUP_PASSES,
        "timed_full_passes_per_case": REPEATS,
        "sample_read_seconds": read_seconds,
        "so2_materialized_encoder_kernels": materialized_count,
        "cases": observations,
        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda},
    }
    OUTPUT.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


__all__ = ["run"]
