# Copyright 2026 HiperMaximus
"""Disposable five-step AMP timing of three patch encoders on one train WSI."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.spec0054_mil_fold0 import _compiled_closure, _make_optimizer
from experiments.spec0054_shard_bags import ShardBags
from eqvae.models.gated_abmil import GatedABMILClassifier, GatedAttention
from eqvae.models.local_global_mil import LocalGlobalPatchEncoder
from eqvae.training.mil_dynamics import AdamWTelemetry, t0_forward_records
from eqvae.training.mil_telemetry_io import write_telemetry_table

WSI_ID = 26190  # 7,869 patches; closest fold-train WSI to median full-bag size.
SEED = 1701
CASES = (("cnn", 0, 192), ("transformer_2x2_d64", 2, 64),
         ("transformer_4x4_d128", 4, 128))


class SpatialBlock(nn.Module):
    def __init__(self, width: int, *, cls_only: bool) -> None:
        super().__init__()
        self.width = width
        self.cls_only = cls_only
        self.norm_attention = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width) if not cls_only else nn.Identity()
        self.query = nn.Linear(width, width) if cls_only else nn.Identity()
        self.kv = nn.Linear(width, 2 * width) if cls_only else nn.Identity()
        self.output = nn.Linear(width, width)
        self.norm_mlp = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(),
                                 nn.Linear(2 * width, width))

    def forward(self, tokens):
        normalized = self.norm_attention(tokens)
        if self.cls_only:
            q = self.query(normalized[:, :1])
            k, v = self.kv(normalized).chunk(2, dim=-1)
            residual = tokens[:, :1]
        else:
            q, k, v = self.qkv(normalized).chunk(3, dim=-1)
            residual = tokens
        q = q.reshape(q.shape[0], q.shape[1], 4, self.width // 4).transpose(1, 2)
        k = k.reshape(k.shape[0], k.shape[1], 4, self.width // 4).transpose(1, 2)
        v = v.reshape(v.shape[0], v.shape[1], 4, self.width // 4).transpose(1, 2)
        attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0,
                                                 is_causal=False)
        attended = attended.transpose(1, 2).reshape(tokens.shape[0], -1, self.width)
        result = residual + self.output(attended)
        return result + self.mlp(self.norm_mlp(result))


class SpatialEncoder(nn.Module):
    def __init__(self, block: int, width: int) -> None:
        super().__init__()
        self.block = block
        self.grid = 32 // block
        packed_width = 16 * block * block
        self.projection = nn.Identity() if packed_width == width else nn.Linear(packed_width, width)
        self.position = nn.Parameter(torch.empty(1, self.grid * self.grid, width))
        self.cls = nn.Parameter(torch.empty(1, 1, width))
        nn.init.normal_(self.position, std=0.02)
        nn.init.normal_(self.cls, std=0.02)
        self.full_block = SpatialBlock(width, cls_only=False)
        self.cls_block = SpatialBlock(width, cls_only=True)
        self.final_norm = nn.LayerNorm(width)

    def pack(self, maps):
        patches = maps.shape[0]
        pixels = maps.permute(0, 2, 3, 1)
        return pixels.reshape(patches, self.grid, self.block, self.grid, self.block, 16).permute(
            0, 1, 3, 2, 4, 5,
        ).reshape(patches, self.grid * self.grid, 16 * self.block * self.block)

    def forward(self, maps):
        tokens = self.projection(self.pack(maps))
        tokens = tokens + self.position.to(dtype=tokens.dtype)
        cls = self.cls.to(dtype=tokens.dtype).expand(maps.shape[0], -1, -1)
        tokens = torch.cat((cls, tokens), dim=1)
        tokens = self.full_block(tokens)
        return self.final_norm(self.cls_block(tokens))[:, 0]


class ProbeClassifier(nn.Module):
    def __init__(self, block: int, width: int) -> None:
        super().__init__()
        self.patch_encoder = LocalGlobalPatchEncoder() if block == 0 else SpatialEncoder(block, width)
        self.attention = GatedAttention(128)
        self.attention.tanh_projection = nn.Linear(width, 128)
        self.attention.gate_projection = nn.Linear(width, 128)
        self.classifier = nn.Linear(width, 5)
        self.apply(GatedABMILClassifier._initialize)
        nn.init.zeros_(self.attention.score.weight)
        nn.init.zeros_(self.classifier.weight)

    def forward_with_representations(self, latents):
        patches = self.patch_encoder(latents)
        embedding, attention = self.attention(patches)
        with torch.autocast(device_type=latents.device.type, enabled=False):
            logits = self.classifier(embedding.float())
        return logits, (patches, *attention, embedding, logits)


def log(event, **values):
    print(json.dumps({"event": event, **values}), flush=True)


def run_case(name, block, width, array, target_index, class_weight, directory):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = ProbeClassifier(block, width).cuda().train()
    if block == 0:
        model = model.to(memory_format=torch.channels_last)  # pyright: ignore[reportCallIssue]
    transfer_start = time.perf_counter()
    latents = torch.from_numpy(array).to("cuda", memory_format=torch.channels_last)
    torch.cuda.synchronize()
    transfer_seconds = time.perf_counter() - transfer_start
    target = torch.tensor([target_index], device="cuda")
    weight = torch.tensor(class_weight, device="cuda")
    optimizer = _make_optimizer(model, peak_lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024.0)
    telemetry = AdamWTelemetry()
    compiled = _compiled_closure(model)
    committed_updates = 0
    rows = []
    attention_kernels = []

    def step(index):
        nonlocal committed_updates
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        optimizer.zero_grad(set_to_none=True)
        start = time.perf_counter()
        events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        events[0].record()
        initial_scale = scaler.get_scale()
        log("forward_start", case=name, step=index)
        logits, loss, weighted, summary = compiled(latents, target, weight)
        events[1].record()
        log("backward_start", case=name, step=index)
        scaler.scale(weighted).backward()
        events[2].record()
        log("optimizer_start", case=name, step=index)
        scaler.unscale_(optimizer)
        telemetry.begin_step(model, optimizer)
        scaler.step(optimizer)
        scaler.update()
        committed = scaler.get_scale() >= initial_scale
        optimizer_record = telemetry.finish_step(
            model, next_committed_update=committed_updates + 1, committed=committed,
        )
        committed_updates += int(committed)
        events[3].record()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        forward = {
            f"forward.{capture}.{metric}": value
            for capture, values in t0_forward_records(
                summary, capture_names=GatedABMILClassifier.capture_names,
            ).items() for metric, value in values.items()
        }
        row = {
            "step": index, "timed": index >= 2, "committed": committed,
            "wall_seconds": seconds,
            "forward_gpu_ms": events[0].elapsed_time(events[1]),
            "backward_gpu_ms": events[1].elapsed_time(events[2]),
            "optimizer_and_gradient_telemetry_gpu_ms": events[2].elapsed_time(events[3]),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "loss": float(loss.detach()), "amp_scale_before": initial_scale,
            "amp_scale_after": scaler.get_scale(),
            **forward, **optimizer_record,
        }
        rows.append(row)
        log("step_done", case=name, **{key: row[key] for key in (
            "step", "timed", "committed", "wall_seconds", "forward_gpu_ms",
            "backward_gpu_ms", "peak_allocated_bytes", "loss", "amp_scale_after",
        )})
        return logits

    log("case_start", case=name, patch_count=len(array), width=width, packing=block,
        parameters=sum(parameter.numel() for parameter in model.parameters()))
    step(0)  # Cold forward/backward compilation is explicitly excluded.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as profile:
        step(1)  # Profile only the second warmup, never a timed step.
    attention_kernels = sorted({event.key for event in profile.key_averages()
                                if any(word in event.key.lower() for word in
                                       ("attention", "fmha", "flash", "cutlass"))})
    del profile
    for index in range(2, 5):
        step(index)
    write_telemetry_table(directory / f"{name}_t0.npz", rows)
    timed = rows[2:]
    result = {
        "case": name, "packing": block, "patch_vector_width": width,
        "spatial_tokens_with_cls": 0 if block == 0 else (32 // block) ** 2 + 1,
        "heads": 0 if block == 0 else 4,
        "head_width": 0 if block == 0 else width // 4,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "transfer_seconds": transfer_seconds,
        "warmup_seconds": [row["wall_seconds"] for row in rows[:2]],
        "timed_seconds": [row["wall_seconds"] for row in timed],
        "median_step_seconds": statistics.median(row["wall_seconds"] for row in timed),
        "median_forward_gpu_ms": statistics.median(row["forward_gpu_ms"] for row in timed),
        "median_backward_gpu_ms": statistics.median(row["backward_gpu_ms"] for row in timed),
        "peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in timed),
        "peak_reserved_bytes": max(row["peak_reserved_bytes"] for row in timed),
        "committed_updates": committed_updates,
        "attention_kernel_names": attention_kernels,
        "status": "complete",
    }
    log("case_done", **result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--branch", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True
    directory = args.output_root / args.branch
    directory.mkdir(parents=True)
    log("worker_start", branch=args.branch, physical_gpu=os.environ["CUDA_VISIBLE_DEVICES"],
        device_name=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda)
    contract = json.loads((args.repo_root / "docs/data/spec0054_fp16_latent_extraction.json").read_text())
    config = json.loads((args.repo_root / "docs/data/spec0054_fold0_run.json").read_text())
    read_start = time.perf_counter()
    bags = ShardBags(args.latent_root, contract,
                     args.repo_root / "docs/data/spec0054_cohort_folds.csv",
                     {row["shard"]: row["result_sha256"] for row in config["sources"]})
    location = bags.locations[WSI_ID]
    train = [bag for bag in bags.locations.values() if bag.fold != config["fold"]]
    counts = np.bincount([bag.diagnosis_index for bag in train], minlength=5)
    class_weight = len(train) / (5 * counts[location.diagnosis_index])
    indices = np.sort(np.random.default_rng(SEED).choice(location.count,
                                                       int(0.8 * location.count), replace=False))
    log("bag_read_start", wsi_id=WSI_ID, full_patch_count=location.count, fold=location.fold)
    array = bags.read(args.branch, WSI_ID)[indices].copy()
    bags.close()
    report = {
        "branch": args.branch, "physical_gpu": os.environ["CUDA_VISIBLE_DEVICES"],
        "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "cuda": torch.version.cuda, "wsi_id": WSI_ID, "fold": location.fold,
        "full_patch_count": location.count, "sampled_patch_count": len(array),
        "sample_indices_sha256": hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest(),
        "reader_and_sampling_seconds": time.perf_counter() - read_start,
        "warmup_steps": 2, "timed_steps": 3, "amp": "fp16",
        "compile_mode": "max-autotune-no-cudagraphs", "dynamic_axis": "patch_count_only",
        "effective_batch": 1, "aem": False,
        "note": "Common class-weighted CE isolates encoder cost; no quality comparison or fold training.",
        "cases": [],
    }
    for name, block, width in CASES:
        torch.compiler.reset()
        gc.collect()
        torch.cuda.empty_cache()
        try:
            result = run_case(name, block, width, array, location.diagnosis_index,
                              float(class_weight), directory)
        except torch.cuda.OutOfMemoryError as error:
            # A measured hardware limit is an experimental result, not a new fallback recipe.
            result = {"case": name, "status": "cuda_oom", "message": str(error)}
            log("case_cuda_oom", **result)
        report["cases"].append(result)
        (directory / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    log("worker_done", branch=args.branch)


if __name__ == "__main__":
    main()
