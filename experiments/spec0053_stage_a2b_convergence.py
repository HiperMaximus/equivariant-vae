# Copyright 2026 HiperMaximus
"""Continue the most informative Stage A2 full-latent paths."""

import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

OUTPUT_ROOT = Path("/kaggle/working/functional_geometry_stage_a2b")
CHECKPOINT_ROOT = Path(
    "/kaggle/input/datasets/maximusshtefan/eqvae-stage-a2-complete-checkpoints"
)
MODEL_DEVICE_MAP = {"normal_vae": 0, "so2_vae": 1}
PATH_NAMES = tuple(
    f"rank_{rank}_{name}"
    for rank in (0, 12)
    for name in ("encoded_side_0", "prescribed_side_0", "bridge_1")
)
STEPS = 1024
SEGMENTS = 32
CHUNK = 8
LR = 0.005 * math.sqrt(32 / 16384)
STARTED = time.perf_counter()


def _log(event, **values):
    print(
        json.dumps(
            {
                "elapsed_seconds": round(time.perf_counter() - STARTED, 3),
                "event": event,
                **values,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _edge_energy(model, latents, total_segments, *, torch):
    decoded = model.decode(latents)
    return (
        total_segments * (decoded[1:] - decoded[:-1]).flatten(1).square().mean(1).sum()
    )


def _path_energy(left, interior, right, edge_energy, *, backward, torch):
    total = 0.0
    with torch.set_grad_enabled(backward):
        for start in range(0, SEGMENTS, CHUNK):
            stop = start + CHUNK
            pieces = ([left] if start == 0 else []) + [
                interior[max(start - 1, 0) : min(stop, SEGMENTS - 1)]
            ]
            if stop == SEGMENTS:
                pieces.append(right)
            energy = edge_energy(
                torch.cat(pieces), interior.new_tensor(float(SEGMENTS))
            )
            total += float(energy.detach())
            if backward:
                energy.backward()
    return total


def _decoded_metrics(decoded, *, torch):
    edge_rms = torch.sqrt((decoded[1:] - decoded[:-1]).flatten(1).square().mean(1))
    endpoint = decoded[-1] - decoded[0]
    endpoint_rms = torch.sqrt(endpoint.square().mean())
    direction = endpoint.flatten()
    denominator = direction.square().sum().clamp_min(torch.finfo(direction.dtype).tiny)
    coefficient = ((decoded - decoded[0]).flatten(1) @ direction / denominator).clamp(
        0.0, 1.0
    )
    chord = decoded[0].unsqueeze(0) + coefficient[:, None, None, None] * endpoint
    return {
        "bottleneck_to_endpoint_chord_rms": float(
            torch.sqrt((decoded - chord).flatten(1).square().mean(1)).max()
        ),
        "decoded_energy": float(SEGMENTS * edge_rms.square().sum()),
        "decoded_length": float(edge_rms.sum()),
        "endpoint_decoded_rms": float(endpoint_rms),
        "energy_over_endpoint_rms_squared": float(
            SEGMENTS
            * edge_rms.square().sum()
            / endpoint_rms.square().clamp_min(torch.finfo(endpoint_rms.dtype).tiny)
        ),
    }


def _atomic_save(path, value, *, torch):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _optimize(
    model,
    initial_path,
    output_path,
    *,
    eager_energy,
    compiled_energy,
    decode_knots,
    model_name,
    name,
    torch,
):
    left, right = initial_path[:1], initial_path[-1:]
    interior = initial_path[1:-1].detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([interior], lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=128,
        threshold=1e-4,
        threshold_mode="rel",
        min_lr=LR / 64,
    )
    initial_energy = _path_energy(
        left, interior, right, eager_energy, backward=False, torch=torch
    )
    best_energy, best_path, best_step = initial_energy, initial_path.detach().clone(), 0
    trace = []

    def save(step, **extra):
        _atomic_save(
            output_path,
            {
                "best_energy": best_energy,
                "best_path": best_path.detach().to("cpu"),
                "best_step": best_step,
                "current_path": torch.cat((left, interior.detach(), right)).to("cpu"),
                "initial_energy": initial_energy,
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "step": step,
                "trace": trace,
                **extra,
            },
            torch=torch,
        )

    for step in range(1, STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        energy = _path_energy(
            left, interior, right, compiled_energy, backward=True, torch=torch
        )
        if step == 1 or step % 32 == 0:
            trace.append(
                {
                    "energy": energy,
                    "gradient_l2": float(torch.linalg.vector_norm(interior.grad)),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "step": step,
                }
            )
        if energy < best_energy:
            best_energy = energy
            best_path = torch.cat((left, interior.detach().clone(), right))
            best_step = step - 1
        optimizer.step()
        scheduler.step(energy)
        if step % 128 == 0:
            save(step)
            _log(
                "path_checkpointed",
                energy=best_energy,
                model=model_name,
                path=name,
                step=step,
            )

    final_energy = _path_energy(
        left, interior, right, eager_energy, backward=False, torch=torch
    )
    if final_energy < best_energy:
        best_energy = final_energy
        best_path = torch.cat((left, interior.detach(), right))
        best_step = STEPS
    decoded = decode_knots(model, best_path, torch=torch)
    metrics = _decoded_metrics(decoded, torch=torch)
    save(
        STEPS,
        decoded_raw_fp16=decoded.detach().to("cpu", torch.float16),
        final_energy=final_energy,
        metrics=metrics,
    )
    return {
        "best_energy": best_energy,
        "best_step": best_step,
        "final_energy": final_energy,
        "final_learning_rate": optimizer.param_groups[0]["lr"],
        "initial_energy": initial_energy,
        "metrics": metrics,
        "name": name,
    }


def _worker(model_name, device_index, repo_root_text, source_commit):
    repo_root = Path(repo_root_text)
    sys.path[:0] = [str(repo_root), str(repo_root / "src")]
    import torch

    from experiments.spec0053_stage_a2_calibration import (
        WEIGHT_DATASET_ROOT,
        _decode_knots,
        _load_frozen_model,
        _write_json,
    )

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device)
    model = _load_frozen_model(
        model_name,
        WEIGHT_DATASET_ROOT / f"{model_name}_state.pt",
        device,
        torch=torch,
    )
    first = torch.load(
        CHECKPOINT_ROOT / f"{model_name}__{PATH_NAMES[0]}.pt",
        map_location="cpu",
        weights_only=True,
    )["best_path"].to(device)

    def eager(latents, total_segments):
        return _edge_energy(model, latents, total_segments, torch=torch)

    compiled = torch.compile(eager, dynamic=False, fullgraph=True, mode="default")
    probe = first[: CHUNK + 1].detach().clone().requires_grad_(True)
    compiled(probe, probe.new_tensor(float(SEGMENTS))).backward()
    torch.cuda.synchronize(device)

    checkpoint_output = OUTPUT_ROOT / "checkpoints"
    checkpoint_output.mkdir(parents=True, exist_ok=True)
    results = []
    for name in PATH_NAMES:
        saved = torch.load(
            CHECKPOINT_ROOT / f"{model_name}__{name}.pt",
            map_location="cpu",
            weights_only=True,
        )
        _log("path_started", model=model_name, path=name)
        results.append(
            _optimize(
                model,
                saved["best_path"].to(device),
                checkpoint_output / f"{model_name}__{name}.pt",
                eager_energy=eager,
                compiled_energy=compiled,
                decode_knots=_decode_knots,
                model_name=model_name,
                name=name,
                torch=torch,
            )
        )
        _write_json(
            OUTPUT_ROOT / f"{model_name}.json",
            {
                "additional_steps": STEPS,
                "model": model_name,
                "paths": results,
                "source_commit": source_commit,
            },
        )
        _log("path_complete", model=model_name, path=name)


def run(*, repo_root, source_commit):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("spawn")
    workers = [
        context.Process(
            target=_worker,
            args=(model, device, str(repo_root), source_commit),
        )
        for model, device in MODEL_DEVICE_MAP.items()
    ]
    for process in workers:
        process.start()
    for process in workers:
        process.join()
    return next((process.exitcode for process in workers if process.exitcode), 0)
