# Copyright 2026 HiperMaximus
"""Finish the real-WSI AMP batch sweep after the first four measured cases."""

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
REPEATS = 3
CASES = (
    ("fp32_b8", 8, False),
    ("amp_b32", 32, True),
    ("amp_b64", 64, True),
)
OUTPUT = Path("/kaggle/working/spec0054_fp16_inference_speed_probe.json")


def _run_case(
    images: np.ndarray[Any, np.dtype[np.uint8]],
    models: dict[str, Any],
    *,
    batch_size: int,
    amp: bool,
    torch: Any,
) -> tuple[dict[str, Any], dict[str, np.ndarray[Any, Any]]]:
    functions = {
        name: _encoder_function(model, torch) for name, model in models.items()
    }
    devices = {"normal_vae": "cuda:0", "so2_vae": "cuda:1"}

    def batches(
        collect: bool, names: tuple[str, ...]
    ) -> dict[str, np.ndarray[Any, Any]]:
        outputs: dict[str, list[np.ndarray[Any, Any]]] = {name: [] for name in names}
        with (
            torch.inference_mode(),
            torch.autocast("cuda", dtype=torch.float16, enabled=amp),
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
                    payload = _payload(stored)
                    if collect:
                        outputs[name].append(
                            np.frombuffer(payload, dtype="<f2")
                            .reshape(-1, 16, 32, 32)
                            .copy()
                        )
        return (
            {name: np.concatenate(chunks) for name, chunks in outputs.items()}
            if collect
            else {}
        )

    compile_started = time.perf_counter()
    with (
        torch.inference_mode(),
        torch.autocast("cuda", dtype=torch.float16, enabled=amp),
    ):
        warm = torch.from_numpy(images[:batch_size]).pin_memory()
        for name in models:
            functions[name](warm.to(devices[name], non_blocking=True))
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)
    compile_seconds = time.perf_counter() - compile_started

    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.reset_peak_memory_stats(1)
    seconds: list[float] = []
    stored_outputs: dict[str, np.ndarray[Any, Any]] = {}
    paired_names = tuple(models)
    for repeat in range(REPEATS):
        started = time.perf_counter()
        observed = batches(collect=repeat == 0, names=paired_names)
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)
        seconds.append(time.perf_counter() - started)
        if repeat == 0:
            stored_outputs = observed
    per_model_seconds: dict[str, list[float]] = {}
    for index, name in enumerate(models):
        observations: list[float] = []
        for _ in range(2):
            started = time.perf_counter()
            batches(collect=False, names=(name,))
            torch.cuda.synchronize(index)
            observations.append(time.perf_counter() - started)
        per_model_seconds[name] = observations
    return (
        {
            "batch_size": batch_size,
            "amp_fp16": amp,
            "compile_and_warmup_seconds": compile_seconds,
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
        },
        stored_outputs,
    )


def _difference(
    reference: np.ndarray[Any, Any], candidate: np.ndarray[Any, Any]
) -> dict[str, float | int | None]:
    left = reference.astype(np.float32)
    right = candidate.astype(np.float32)
    finite = np.isfinite(left) & np.isfinite(right)
    delta = (right[finite] - left[finite]).astype(np.float64)
    absolute = np.abs(delta)
    reference_norm = np.linalg.norm(left[finite].astype(np.float64))
    return {
        "elements": int(left.size),
        "nonfinite_count": int(np.count_nonzero(~np.isfinite(right))),
        "changed_fp16_elements": int(np.count_nonzero(reference != candidate)),
        "relative_l2": (
            float(np.linalg.norm(delta) / reference_norm)
            if reference_norm > 0
            else None
        ),
        "rmse": float(np.sqrt(np.mean(delta * delta))) if delta.size else None,
        "p99_absolute": float(np.quantile(absolute, 0.99)) if delta.size else None,
        "max_absolute": float(np.max(absolute)) if delta.size else None,
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
    reference: dict[str, np.ndarray[Any, Any]] = {}
    for name, batch_size, amp in CASES:
        timing, outputs = _run_case(
            images, models, batch_size=batch_size, amp=amp, torch=torch
        )
        if name == "fp32_b8":
            reference = outputs
        timing["difference_from_fp32_b8_stored_fp16"] = {
            model_name: _difference(reference[model_name], outputs[model_name])
            for model_name in models
        }
        observations[name] = timing
        print(json.dumps({"case": name, **timing}, sort_keys=True), flush=True)
    result = {
        "schema_version": "spec0054.fp16_inference_speed_probe.v1",
        "source_commit": source_commit,
        "contract_sha256": _sha256(contract_path),
        "sample_wsi_id": rows[0].wsi_id,
        "sample_atlas_row_indices": [rows[0].atlas_row_index, rows[-1].atlas_row_index],
        "sample_patches": len(images),
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
