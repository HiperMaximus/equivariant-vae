# Copyright 2026 HiperMaximus
# ruff: noqa: ANN001, ANN201, ANN202, ANN204, BLE001, DOC201, DOC501, EM101, INP001, PLC0415, PLR2004, PLR6104, S404, TRY003
"""One-off synthetic full-bag transformer fit check; no datasets or training."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path


def make_model():
    """Keep the exact proposed network self-contained for this one upload."""
    import torch
    from torch import nn
    from torch.nn import functional

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = nn.LayerNorm(128)
            self.qkv = nn.Linear(128, 384)
            self.proj = nn.Linear(128, 128)
            self.norm2 = nn.LayerNorm(128)
            self.gate_up = nn.Linear(128, 512)
            self.down = nn.Linear(256, 128)

        def forward(self, x):
            length = x.shape[1]
            qkv = self.qkv(self.norm1(x)).reshape(1, length, 3, 4, 32)
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            attended = functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
            x = x + self.proj(attended.transpose(1, 2).reshape(1, length, 128))
            gate, up = self.gate_up(self.norm2(x)).chunk(2, dim=-1)
            return x + self.down(functional.silu(gate) * up)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(16, 32, 5, stride=2, padding=2),
                nn.GroupNorm(8, 32),
                nn.GELU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.GroupNorm(8, 64),
                nn.GELU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.GroupNorm(8, 128),
                nn.GELU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(1),
            )
            self.cls = nn.Parameter(torch.empty(1, 1, 128))
            self.registers = nn.Parameter(torch.empty(1, 8, 128))
            self.blocks = nn.Sequential(Block(), Block())
            self.norm = nn.LayerNorm(128)
            self.head = nn.Linear(128, 5)
            for module in self.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        module.weight,
                        mode="fan_out",
                        nonlinearity="relu",
                    )
                elif isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                elif isinstance(module, (nn.GroupNorm, nn.LayerNorm)):
                    nn.init.ones_(module.weight)
                if isinstance(
                    module,
                    (nn.Conv2d, nn.Linear, nn.GroupNorm, nn.LayerNorm),
                ):
                    nn.init.zeros_(module.bias)
            nn.init.normal_(self.cls, std=0.02)
            nn.init.normal_(self.registers, std=0.02)

        def forward(self, patches):
            h = self.encoder(patches).unsqueeze(0)
            x = torch.cat((self.cls.to(h.dtype), h, self.registers.to(h.dtype)), dim=1)
            x = self.norm(self.blocks(x)[:, 0])
            with torch.autocast(patches.device.type, enabled=False):
                return self.head(x.float())

    return Model()


def run_probe(result):
    """Allocate both full replicas and measure two complete optimizer steps."""
    import torch
    from torch.nn import functional
    from torch.nn.attention import SDPBackend, sdpa_kernel

    result["runtime"] = {
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "devices": [
            {
                "name": torch.cuda.get_device_name(i),
                "capability": torch.cuda.get_device_capability(i),
                "total_bytes": torch.cuda.get_device_properties(i).total_memory,
            }
            for i in range(torch.cuda.device_count())
        ],
    }
    if torch.cuda.device_count() != 2 or any(
        "T4" not in device["name"] for device in result["runtime"]["devices"]
    ):
        raise RuntimeError("This probe requires exactly two T4s")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.manual_seed(1701)
    prototype = make_model()
    result["parameter_count"] = sum(p.numel() for p in prototype.parameters())
    models = [copy.deepcopy(prototype).to(f"cuda:{i}") for i in range(2)]
    bag = torch.randn(8149, 16, 32, 32)
    bags = [bag.to(f"cuda:{i}") for i in range(2)]
    targets = [torch.tensor([1], device=f"cuda:{i}") for i in range(2)]
    optimizers = [
        torch.optim.AdamW(
            [
                {
                    "params": [p for p in model.parameters() if p.ndim >= 2],
                    "weight_decay": 1e-4,
                },
                {
                    "params": [p for p in model.parameters() if p.ndim < 2],
                    "weight_decay": 0.0,
                },
            ],
            lr=2e-4,
        )
        for model in models
    ]
    scalers = [
        torch.amp.GradScaler("cuda", init_scale=32768, growth_interval=1000000)
        for _ in models
    ]
    result["steps"] = []
    for phase in ("warmup", "measured"):
        result["phase"] = phase
        for i in range(2):
            optimizers[i].zero_grad(set_to_none=True)
            torch.cuda.synchronize(i)
            torch.cuda.reset_peak_memory_stats(i)
        start = time.perf_counter()
        losses = []
        # Explicit backend: successful forward AND backward prove its execution.
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
        if not all(finite):
            result["finite_gradients"] = finite
            raise FloatingPointError(
                "Nonfinite synthetic loss/gradient; neither optimizer stepped",
            )
        for i in range(2):
            scalers[i].step(optimizers[i])
            scalers[i].update()
        for i in range(2):
            torch.cuda.synchronize(i)
        elapsed = time.perf_counter() - start
        result["steps"].append({
            "phase": phase,
            "paired_wall_seconds": elapsed,
            "losses": [float(loss.item()) for loss in losses],
            "scales": [scaler.get_scale() for scaler in scalers],
            "peak_allocated_bytes": [
                torch.cuda.max_memory_allocated(i) for i in range(2)
            ],
            "peak_reserved_bytes": [
                torch.cuda.max_memory_reserved(i) for i in range(2)
            ],
            "optimizer_state_entries": [len(opt.state) for opt in optimizers],
        })
        if any(scaler.get_scale() != 32768 for scaler in scalers):
            raise FloatingPointError("Unexpected GradScaler skip/backoff")
        print(json.dumps(result["steps"][-1]), flush=True)
    result["status"] = "fits_synthetic_full_training_step"


def main():
    """Write compact evidence even if the GPU experiment fails."""
    result = {
        "status": "failed",
        "scope": "synthetic_capacity_only_not_learning",
        "patch_count": 8149,
        "sequence_length": 8158,
        "layers": 2,
        "cls_tokens": 1,
        "registers": 8,
        "width": 128,
        "heads": 4,
        "head_width": 32,
        "swiglu_inner": 256,
        "sdpa_backend": "EFFICIENT_ATTENTION_only",
        "dataset_sources": [],
        "precision": "FP32 inputs/parameters; FP16 autocast; FP32 head/loss",
        "replicas": "identical synthetic copies on normal/SO2 device slots",
        "checkpoint_chunk_size": None,
        "lr": 2e-4,
    }
    start = time.monotonic()
    try:
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
        run_probe(result)
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        traceback.print_exc()
    finally:
        import torch

        result["session_seconds"] = time.monotonic() - start
        result["final_peak_allocated_bytes"] = [
            torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())
        ]
        result["final_peak_reserved_bytes"] = [
            torch.cuda.max_memory_reserved(i) for i in range(torch.cuda.device_count())
        ]
        Path("/kaggle/working/transformer_capacity.json").write_text(
            json.dumps(result, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result), flush=True)
    return 0 if result["status"] == "fits_synthetic_full_training_step" else 1


if __name__ == "__main__":
    raise SystemExit(main())
