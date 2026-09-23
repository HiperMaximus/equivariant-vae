# Copyright 2026 HiperMaximus
"""Extract paired complete-WSI posterior means directly to FP16 shards."""

from __future__ import annotations

import csv
import hashlib
import importlib
import itertools
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from experiments.spec0054_mil_a0_probe import (
    _load_frozen_model,
    _sha256,
    _unique_input,
)

OUTPUT_ROOT = Path("/kaggle/working/spec0054_fp16_latents")
CONTRACT_RELATIVE = Path("docs/data/spec0054_fp16_latent_extraction.json")
COHORT_RELATIVE = Path("docs/data/spec0054_cohort_folds.csv")
BATCH_SIZE = 8
PATCH_SIZE = 256
LATENT_SHAPE = (16, 32, 32)
LATENT_RECORD_BYTES = int(np.prod(LATENT_SHAPE)) * np.dtype("<f2").itemsize


@dataclass(frozen=True)
class AtlasRow:
    atlas_row_index: int
    wsi_id: int
    x: int
    y: int


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_shard_rows(atlas_path: Path, shard: dict[str, Any]) -> list[AtlasRow]:
    rows: list[AtlasRow] = []
    with atlas_path.open(newline="", encoding="utf-8") as handle:
        for atlas_row_index, raw in enumerate(csv.DictReader(handle)):
            wsi_id = int(raw["image_id"])
            if int(shard["first_wsi"]) <= wsi_id <= int(shard["last_wsi"]):
                rows.append(
                    AtlasRow(
                        atlas_row_index=atlas_row_index,
                        wsi_id=wsi_id,
                        x=int(raw["x"]),
                        y=int(raw["y"]),
                    )
                )
    return rows


def _load_cohort(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {int(row["wsi_id"]): row for row in csv.DictReader(handle)}


def _write_index(
    path: Path,
    rows: list[AtlasRow],
    cohort: dict[int, dict[str, str]],
) -> None:
    fields = (
        "file_index",
        "atlas_row_index",
        "wsi_id",
        "x",
        "y",
        "diagnosis",
        "diagnosis_index",
        "vae_source_role",
        "fold",
    )
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for file_index, row in enumerate(rows):
            metadata = cohort[row.wsi_id]
            writer.writerow(
                {
                    "file_index": file_index,
                    "atlas_row_index": row.atlas_row_index,
                    "wsi_id": row.wsi_id,
                    "x": row.x,
                    "y": row.y,
                    "diagnosis": metadata["diagnosis"],
                    "diagnosis_index": metadata["diagnosis_index"],
                    "vae_source_role": metadata["vae_source_role"],
                    "fold": metadata["fold"],
                }
            )


def _wsi_directory() -> Path:
    candidates = (
        Path("/kaggle/input/UBC-OCEAN/train_images"),
        Path("/kaggle/input/competitions/UBC-OCEAN/train_images"),
    )
    matches = [path for path in candidates if path.is_dir()]
    if len(matches) != 1:
        raise RuntimeError("Expected one official UBC-OCEAN train_images mount")
    return matches[0]


def _iter_patch_batches(
    rows: list[AtlasRow],
    wsi_directory: Path,
) -> Any:
    pyvips = importlib.import_module("pyvips")
    pyvips.cache_set_max(0)
    pyvips.cache_set_max_mem(0)
    pyvips.cache_set_max_files(0)
    for wsi_id, grouped in itertools.groupby(rows, key=lambda row: row.wsi_id):
        wsi_rows = list(grouped)
        image = pyvips.Image.new_from_file(
            str(wsi_directory / f"{wsi_id}.png"),
            access="sequential",
            fail=True,
        )
        if int(image.bands) != 3:
            raise RuntimeError(f"WSI {wsi_id} is not RGB")
        region = pyvips.Region.new(image)
        pending: list[np.ndarray[Any, np.dtype[np.uint8]]] = []
        processed = 0
        for y, same_y in itertools.groupby(wsi_rows, key=lambda row: row.y):
            y_rows = list(same_y)
            span_start = y_rows[0].x
            span_width = y_rows[-1].x + PATCH_SIZE - span_start
            blob = region.fetch(span_start, y, span_width, PATCH_SIZE)
            strip = np.frombuffer(blob, dtype=np.uint8).reshape(
                PATCH_SIZE,
                span_width,
                3,
            )
            for row in y_rows:
                offset = row.x - span_start
                crop = strip[:, offset : offset + PATCH_SIZE, :]
                pending.append(np.ascontiguousarray(crop.transpose(2, 0, 1)))
                processed += 1
                if len(pending) == BATCH_SIZE or processed == len(wsi_rows):
                    yield (
                        wsi_id,
                        np.stack(pending),
                        processed == len(wsi_rows),
                    )
                    pending.clear()


def _materialize_so2_encoder(model: Any) -> int:
    roots = (
        model.stem_conv,
        model.encoder_blocks,
        model.mu_head,
        model.logvar_head,
    )
    count = 0
    visited: set[int] = set()
    for root in roots:
        for module in root.modules():
            identity = id(module)
            materialize = getattr(module, "materialize_frozen_kernel", None)
            if identity not in visited and callable(materialize):
                materialize()
                visited.add(identity)
                count += 1
    return count


def _encoder_function(model: Any, torch: Any) -> Any:
    def encode(images_uint8: Any) -> Any:
        normalized = images_uint8.float().div(255).mul(2).sub(1)
        return model.encode(normalized)[0]

    return torch.compile(
        encode,
        backend="inductor",
        fullgraph=True,
        dynamic=False,
        mode="max-autotune-no-cudagraphs",
    )


def _delta(left: Any, right: Any, torch: Any) -> dict[str, float | int]:
    difference = right.detach().float() - left.detach().float()
    return {
        "element_count": difference.numel(),
        "maximum_absolute_difference": float(difference.abs().max().item()),
        "rmse": float(torch.sqrt(torch.mean(difference.double().square())).item()),
        "exact_equal_count": int((difference == 0).sum().item()),
    }


def _padded_tensor(images: np.ndarray[Any, np.dtype[np.uint8]], torch: Any) -> Any:
    valid_count = images.shape[0]
    if valid_count < BATCH_SIZE:
        images = np.concatenate(
            (images, np.repeat(images[-1:], BATCH_SIZE - valid_count, axis=0))
        )
    return torch.from_numpy(images).pin_memory()


def _prepare_encoders(
    models: dict[str, Any],
    sample: np.ndarray[Any, np.dtype[np.uint8]],
    torch: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    devices = {
        "normal_vae": torch.device("cuda:0"),
        "so2_vae": torch.device("cuda:1"),
    }
    source = _padded_tensor(sample, torch)
    normal_input = source.to(devices["normal_vae"], non_blocking=True)
    so2_input = source.to(devices["so2_vae"], non_blocking=True)
    normalize = lambda value: value.float().div(255).mul(2).sub(1)
    with torch.inference_mode():
        normal_eager = models["normal_vae"].encode(normalize(normal_input))[0]
        so2_before = models["so2_vae"].encode(normalize(so2_input))[0]
        materialized_count = _materialize_so2_encoder(models["so2_vae"])
        so2_materialized = models["so2_vae"].encode(normalize(so2_input))[0]
        functions = {
            name: _encoder_function(model, torch) for name, model in models.items()
        }
        normal_compiled = functions["normal_vae"](normal_input)
        so2_compiled = functions["so2_vae"](so2_input)
    torch.cuda.synchronize(devices["normal_vae"])
    torch.cuda.synchronize(devices["so2_vae"])
    probe = {
        "so2_materialized_encoder_kernel_count": materialized_count,
        "normal_eager_vs_compiled": _delta(normal_eager, normal_compiled, torch),
        "so2_eager_vs_materialized": _delta(so2_before, so2_materialized, torch),
        "so2_materialized_vs_compiled": _delta(
            so2_materialized,
            so2_compiled,
            torch,
        ),
    }
    return functions, probe


def _encode_batch(
    functions: dict[str, Any],
    images: np.ndarray[Any, np.dtype[np.uint8]],
    torch: Any,
) -> tuple[Any, Any]:
    valid_count = images.shape[0]
    source = _padded_tensor(images, torch)
    with torch.inference_mode():
        normal = functions["normal_vae"](
            source.to("cuda:0", non_blocking=True)
        )
        so2 = functions["so2_vae"](source.to("cuda:1", non_blocking=True))
        normal_cpu = normal[:valid_count].to(device="cpu", dtype=torch.float16)
        so2_cpu = so2[:valid_count].to(device="cpu", dtype=torch.float16)
    return normal_cpu.contiguous(), so2_cpu.contiguous()


def _payload(tensors: Any) -> bytes:
    return tensors.numpy().astype("<f2", copy=False).tobytes(order="C")


def run(
    *,
    repo_root: Path,
    source_commit: str,
    shard_index: int,
    row_limit: int | None = None,
) -> int:
    import torch

    started = time.perf_counter()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    contract_path = repo_root / CONTRACT_RELATIVE
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    shard = dict(contract["shards"][shard_index - 1])
    atlas = _unique_input("raw_atlas.csv", contract["atlas"]["sha256"])
    state_paths = {
        "normal_vae": _unique_input(
            "normal_vae_state.pt", contract["weights"]["normal_vae_sha256"]
        ),
        "so2_vae": _unique_input(
            "so2_vae_state.pt", contract["weights"]["so2_vae_sha256"]
        ),
    }
    rows = _load_shard_rows(atlas, shard)
    if row_limit is not None:
        rows = rows[:row_limit]
    cohort_path = repo_root / COHORT_RELATIVE
    cohort = _load_cohort(cohort_path)
    suffix = f"shard_{shard_index:02d}_of_{len(contract['shards']):02d}"
    if row_limit is not None:
        suffix += f"_probe_{row_limit}"
    index_path = OUTPUT_ROOT / f"index_{suffix}.csv"
    _write_index(index_path, rows, cohort)

    models = {
        "normal_vae": _load_frozen_model(
            "normal_vae", state_paths["normal_vae"], torch.device("cuda:0"), torch
        ),
        "so2_vae": _load_frozen_model(
            "so2_vae", state_paths["so2_vae"], torch.device("cuda:1"), torch
        ),
    }
    batches = iter(_iter_patch_batches(rows, _wsi_directory()))
    first_wsi, first_images, first_is_final = next(batches)
    functions, numerical_probe = _prepare_encoders(models, first_images, torch)

    normal_partial = OUTPUT_ROOT / f"normal_vae_mu_{suffix}.fp16.bin.partial"
    so2_partial = OUTPUT_ROOT / f"so2_vae_mu_{suffix}.fp16.bin.partial"
    progress_path = OUTPUT_ROOT / f"progress_{suffix}.json"
    digests = {"normal_vae": hashlib.sha256(), "so2_vae": hashlib.sha256()}
    rows_written = 0
    completed_wsi: list[int] = []
    batch_stream = itertools.chain(
        ((first_wsi, first_images, first_is_final),),
        batches,
    )
    with normal_partial.open("xb") as normal_handle, so2_partial.open(
        "xb"
    ) as so2_handle:
        for wsi_id, images, is_final_wsi_batch in batch_stream:
            normal, so2 = _encode_batch(functions, images, torch)
            payloads = {
                "normal_vae": _payload(normal),
                "so2_vae": _payload(so2),
            }
            normal_handle.write(payloads["normal_vae"])
            so2_handle.write(payloads["so2_vae"])
            for name, payload in payloads.items():
                digests[name].update(payload)
            rows_written += images.shape[0]
            if is_final_wsi_batch:
                completed_wsi.append(wsi_id)
                normal_handle.flush()
                so2_handle.flush()
                _write_json(
                    progress_path,
                    {
                        "shard": shard_index,
                        "rows_written": rows_written,
                        "last_completed_wsi": wsi_id,
                        "completed_wsi_count": len(completed_wsi),
                    },
                )

    if rows_written != len(rows):
        raise RuntimeError("The encoded row count does not match the shard index")

    normal_path = normal_partial.with_suffix("")
    so2_path = so2_partial.with_suffix("")
    normal_partial.replace(normal_path)
    so2_partial.replace(so2_path)
    progress_path.unlink()
    elapsed = time.perf_counter() - started
    result = {
        "schema_version": "spec0054.fp16_latent_shard.v1",
        "source_commit": source_commit,
        "contract_sha256": _sha256(contract_path),
        "cohort_sha256": _sha256(cohort_path),
        "shard": shard,
        "row_limit": row_limit,
        "expected_rows": len(rows),
        "rows_written": rows_written,
        "wsi_ids": completed_wsi,
        "dtype": "<f2",
        "record_shape": list(LATENT_SHAPE),
        "record_bytes": LATENT_RECORD_BYTES,
        "files": {
            "index": {
                "name": index_path.name,
                "bytes": index_path.stat().st_size,
                "sha256": _sha256(index_path),
            },
            "normal_vae": {
                "name": normal_path.name,
                "bytes": normal_path.stat().st_size,
                "sha256": digests["normal_vae"].hexdigest(),
            },
            "so2_vae": {
                "name": so2_path.name,
                "bytes": so2_path.stat().st_size,
                "sha256": digests["so2_vae"].hexdigest(),
            },
        },
        "numerical_probe": numerical_probe,
        "elapsed_seconds": elapsed,
        "rows_per_second": rows_written / elapsed,
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
            "peak_allocated_bytes": {
                "normal_vae": torch.cuda.max_memory_allocated(0),
                "so2_vae": torch.cuda.max_memory_allocated(1),
            },
        },
    }
    result_path = OUTPUT_ROOT / f"result_{suffix}.json"
    _write_json(result_path, result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


__all__ = ["run"]
