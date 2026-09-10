# Copyright 2026 HiperMaximus
# ruff: noqa: ANN001, ANN201, ANN202, BLE001, DOC201, DOC501, EM101, INP001, PLC0415, PLR2004, PLW0717, S404, TRY003
"""One real, complete WSI through the already-tested two-block transformer."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import time
import traceback
from collections import Counter
from operator import itemgetter
from pathlib import Path

KAGGLE_WSI45630_CAPACITY_READY = True
INPUT_CONTRACT_SHA256 = "$input_contract_sha256"
INPUT_CONTRACT_NAME = "wsi45630_capacity_input.json"
INPUT_ROOT = Path("/kaggle/input")
OUTPUT_PATH = Path("/kaggle/working/wsi45630_transformer_capacity.json")


def sha256(path):
    """Hash only compact inputs and sidecars, never latent payloads."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_bundle():
    """Authenticate the fixed private input dataset before importing its source."""
    matches = list(INPUT_ROOT.rglob(INPUT_CONTRACT_NAME))
    if len(matches) != 1 or sha256(matches[0]) != INPUT_CONTRACT_SHA256:
        raise RuntimeError("Expected the exact full-WSI capacity input contract")
    root = matches[0].parent
    contract = json.loads(matches[0].read_text())
    if (
        contract["dataset_reference"] != "maximusshtefan/eqvae-wsi45630-capacity-inputs"
        or contract["wsi_id"] != 45630
        or contract["patch_count"] != 32595
        or contract["diagnosis_index"] != 1
        or contract["part_counts"] != {"4": 5136, "11": 4810, "12": 22649}
    ):
        raise RuntimeError("Capacity input scope differs")
    for name, record in contract["files"].items():
        path = root / name
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise RuntimeError("Mounted capacity source or manifest differs")
    observed = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if observed != {*contract["files"], INPUT_CONTRACT_NAME}:
        raise RuntimeError("Unexpected file in capacity input dataset")
    return root, contract


def resolve_sources(catalog):
    """Disambiguate equal top-up filenames by authenticated sidecar identity."""
    roots = {}
    with catalog.open(newline="") as handle:
        for row in csv.DictReader(handle):
            candidates = [
                path.parent
                for path in INPUT_ROOT.rglob(row["sidecar_name"])
                if path.is_file()
                and path.stat().st_size == int(row["sidecar_bytes"])
                and sha256(path) == row["sidecar_sha256"]
            ]
            if len(candidates) != 1:
                raise RuntimeError("Expected one hash-matched producer sidecar")
            source = row["kaggle_source"]
            if source in roots and roots[source] != candidates[0]:
                raise RuntimeError("Paired producer binaries have different roots")
            roots[source] = candidates[0]
    return roots


def load_bags(root, contract, result):
    """Read every paired training pointer once and retain both complete GPU bags."""
    import torch

    from eqvae.data.supervised_latents import LogicalPointer, SupervisedLatentStore

    catalog = root / "probe/physical_parts.csv"
    source_roots = resolve_sources(catalog)
    if set(source_roots) != set(contract["kernel_sources"]):
        raise RuntimeError("Unexpected latent producers")
    with (root / "probe/pointers.csv").open(newline="") as handle:
        rows = [{k: int(v) for k, v in row.items()} for row in csv.DictReader(handle)]
    identity = tuple(
        (r["atlas_row_index"], r["wsi_id"], r["x"], r["y"], r["part"], r["file_index"])
        for r in rows
    )
    if (
        len(identity) != 32595
        or {r[1] for r in identity} != {45630}
        or len({(r[2], r[3]) for r in identity}) != 32595
        or identity != tuple(sorted(identity, key=itemgetter(3, 2)))
    ):
        raise RuntimeError("Expected all unique WSI45630 coordinates in y,x order")
    pointers = tuple(LogicalPointer(r[4], r[5]) for r in identity)
    result["input_reads"] = {}
    bags = []
    shared_reads = None
    started = time.monotonic()
    for device, model in enumerate(("normal_vae", "so2_vae")):
        result["phase"] = f"load_{model}"
        with SupervisedLatentStore(
            catalog_path=catalog,
            model_name=model,
            source_roots=source_roots,
        ) as store:
            latents, reads = store.read_rows(pointers)
        counts = Counter(str(r.part) for r in reads)
        if (
            latents.shape != (32595, 16, 32, 32)
            or dict(counts) != contract["part_counts"]
            or not torch.isfinite(latents).all().item()
            or (shared_reads is not None and shared_reads != reads)
        ):
            raise RuntimeError("Full training-bag identity, alignment or values differ")
        shared_reads = reads
        bags.append(latents.to(f"cuda:{device}"))
        result["input_reads"][model] = {
            "rows": len(identity),
            "part_counts": dict(counts),
            "identity_sha256": hashlib.sha256(
                json.dumps(identity).encode(),
            ).hexdigest(),
            "input_finite": True,
        }
        del latents
    result["load_seconds"] = time.monotonic() - started
    return bags


def run_probe(root, contract, result):  # noqa: C901
    """Measure full CNN/transformer backward and optimizer state on both T4s."""
    import torch
    from torch.nn import functional
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformer_probe import make_model

    from eqvae.training.supervised_pairing import make_paired_models

    result["runtime"] = {
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "devices": [
            {
                "name": torch.cuda.get_device_name(i),
                "total_bytes": torch.cuda.get_device_properties(i).total_memory,
                "capability": torch.cuda.get_device_capability(i),
            }
            for i in range(torch.cuda.device_count())
        ],
    }
    if torch.cuda.device_count() != 2 or any(
        "T4" not in d["name"] for d in result["runtime"]["devices"]
    ):
        raise RuntimeError("Exactly two T4 GPUs are required")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    models = make_paired_models(make_model, seed=1701)
    if not all(
        torch.equal(a, b) and a.data_ptr() != b.data_ptr()
        for a, b in zip(models[0].parameters(), models[1].parameters(), strict=True)
    ):
        raise RuntimeError("Initial paired parameters differ or share storage")
    result["identical_independent_initialization"] = True
    result["parameter_count"] = sum(p.numel() for p in models[0].parameters())
    if result["parameter_count"] != 438693:
        raise RuntimeError("Transformer architecture differs from the fitted control")
    models = [m.to(f"cuda:{i}") for i, m in enumerate(models)]
    bags = load_bags(root, contract, result)
    targets = [torch.tensor([1], device=f"cuda:{i}") for i in range(2)]
    optimizers = [
        torch.optim.AdamW(
            [
                {
                    "params": [p for p in m.parameters() if p.ndim >= 2],
                    "weight_decay": 1e-4,
                },
                {
                    "params": [p for p in m.parameters() if p.ndim < 2],
                    "weight_decay": 0.0,
                },
            ],
            lr=2e-4,
        )
        for m in models
    ]
    scalers = [
        torch.amp.GradScaler("cuda", init_scale=32768, growth_interval=1000000)
        for _ in range(2)
    ]
    result["steps"] = []
    for phase in ("warmup", "measured"):
        result["phase"] = phase
        for i in range(2):
            optimizers[i].zero_grad(set_to_none=True)
            torch.cuda.synchronize(i)
            torch.cuda.reset_peak_memory_stats(i)
        started = time.perf_counter()
        losses = []
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            for i in range(2):
                with torch.cuda.device(i), torch.autocast("cuda", dtype=torch.float16):
                    logits = models[i](bags[i])
                    loss = functional.cross_entropy(logits.float(), targets[i])
                scalers[i].scale(loss).backward()
                scalers[i].unscale_(optimizers[i])
                losses.append(loss.detach())
        finite = [
            bool(torch.isfinite(losses[i]).item())
            and all(
                p.grad is not None and bool(torch.isfinite(p.grad).all().item())
                for p in models[i].parameters()
            )
            for i in range(2)
        ]
        result["finite_gradients"] = finite
        if not all(finite):
            raise FloatingPointError(
                "Nonfinite loss/gradient; neither optimizer stepped",
            )
        for i in range(2):
            scalers[i].step(optimizers[i])
            scalers[i].update()
        for i in range(2):
            torch.cuda.synchronize(i)
        result["steps"].append({
            "phase": phase,
            "paired_wall_seconds": time.perf_counter() - started,
            "losses": [float(loss.item()) for loss in losses],
            "scales": [s.get_scale() for s in scalers],
            "finite_gradients": finite,
            "peak_allocated_bytes": [
                torch.cuda.max_memory_allocated(i) for i in range(2)
            ],
            "peak_reserved_bytes": [
                torch.cuda.max_memory_reserved(i) for i in range(2)
            ],
            "optimizer_state_entries": [len(opt.state) for opt in optimizers],
        })
        if any(s.get_scale() != 32768 for s in scalers) or any(
            len(opt.state) != len(list(model.parameters()))
            for opt, model in zip(optimizers, models, strict=True)
        ):
            raise FloatingPointError("Scaler skip or incomplete optimizer state")
        print(json.dumps(result["steps"][-1]), flush=True)
    result["status"] = "fits_real_full_bag_training_step"


def main():
    """Persist compact capacity evidence even after allocation or numerical failure."""
    result = {
        "status": "failed",
        "scope": "real_full_wsi_capacity_only_not_learning",
        "wsi_id": 45630,
        "patch_count": 32595,
        "sequence_length": 32604,
        "layers": 2,
        "cls_tokens": 1,
        "registers": 8,
        "width": 128,
        "heads": 4,
        "swiglu_inner": 256,
        "checkpoint_chunk_size": None,
        "sdpa_backend": "EFFICIENT_ATTENTION_only",
        "lr": 2e-4,
        "precision": "FP32 inputs/parameters; FP16 autocast; FP32 head/loss",
        "input_contract_sha256": INPUT_CONTRACT_SHA256,
        "model_devices": {"normal_vae": 0, "so2_vae": 1},
    }
    start = time.monotonic()
    try:
        _ensure_latest_torch()
        root, contract = resolve_bundle()
        sys.dont_write_bytecode = True
        sys.path[:0] = [str(root / "src"), str(root)]
        result["kernel_sources"] = contract["kernel_sources"]
        run_probe(root, contract, result)
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        traceback.print_exc()
    finally:
        result["session_seconds"] = time.monotonic() - start
        if "torch" in sys.modules:
            import torch

            result["final_peak_allocated_bytes"] = [
                torch.cuda.max_memory_allocated(i)
                for i in range(torch.cuda.device_count())
            ]
            result["final_peak_reserved_bytes"] = [
                torch.cuda.max_memory_reserved(i)
                for i in range(torch.cuda.device_count())
            ]
        OUTPUT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result), flush=True)
    return 0 if result["status"] == "fits_real_full_bag_training_step" else 1


def _ensure_latest_torch():
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "torch",
        "torchvision",
        "torchaudio",
    ])


if __name__ == "__main__":
    raise SystemExit(main())
