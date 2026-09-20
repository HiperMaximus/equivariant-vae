# Copyright 2026 HiperMaximus
"""Continue Stage A2b from its exact optimizer and scheduler states."""

import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

INPUT_ROOT = Path("/kaggle/input/eqvae-stage-a2b-continuation-checkpoints")
OUTPUT_ROOT = Path("/kaggle/working/functional_geometry_stage_a2c")
ADDITIONAL_STEPS = 1024


def _continue_path(
    model,
    saved,
    output_path,
    *,
    eager_energy,
    compiled_energy,
    decode_knots,
    model_name,
    name,
    torch,
):
    from experiments.spec0053_stage_a2b_convergence import (
        _atomic_save,
        _decoded_metrics,
        _log,
        _path_energy,
    )

    current_path = saved["current_path"]
    left, right = current_path[:1], current_path[-1:]
    interior = current_path[1:-1].detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([interior])
    optimizer.load_state_dict(saved["optimizer_state"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    scheduler.load_state_dict(saved["scheduler_state"])
    best_energy = float(saved["best_energy"])
    best_path = saved["best_path"]
    best_step = int(saved["best_step"])
    initial_energy = float(saved["initial_energy"])
    start_step = int(saved["step"])
    trace = list(saved["trace"])

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

    end_step = start_step + ADDITIONAL_STEPS
    for step in range(start_step + 1, end_step + 1):
        optimizer.zero_grad(set_to_none=True)
        energy = _path_energy(
            left, interior, right, compiled_energy, backward=True, torch=torch
        )
        if step % 32 == 0:
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
        best_step = end_step
    decoded = decode_knots(model, best_path, torch=torch)
    metrics = _decoded_metrics(decoded, torch=torch)
    save(
        end_step,
        decoded_raw_fp16=decoded.detach().to("cpu", torch.float16),
        final_energy=final_energy,
        metrics=metrics,
    )
    return {
        "best_energy": best_energy,
        "best_step": best_step,
        "end_step": end_step,
        "final_energy": final_energy,
        "final_learning_rate": optimizer.param_groups[0]["lr"],
        "metrics": metrics,
        "name": name,
        "start_step": start_step,
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
    from experiments.spec0053_stage_a2b_convergence import (
        CHUNK,
        PATH_NAMES,
        SEGMENTS,
        _edge_energy,
        _log,
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
        INPUT_ROOT / f"{model_name}__{PATH_NAMES[0]}.pt",
        map_location=device,
        weights_only=True,
    )

    def eager(latents, total_segments):
        return _edge_energy(model, latents, total_segments, torch=torch)

    compiled = torch.compile(eager, dynamic=False, fullgraph=True, mode="default")
    probe = first["current_path"][: CHUNK + 1].detach().clone().requires_grad_(True)
    compiled(probe, probe.new_tensor(float(SEGMENTS))).backward()
    torch.cuda.synchronize(device)

    checkpoint_output = OUTPUT_ROOT / "checkpoints"
    checkpoint_output.mkdir(parents=True, exist_ok=True)
    results = []
    for name in PATH_NAMES:
        saved = torch.load(
            INPUT_ROOT / f"{model_name}__{name}.pt",
            map_location=device,
            weights_only=True,
        )
        _log("path_started", model=model_name, path=name, step=saved["step"])
        results.append(
            _continue_path(
                model,
                saved,
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
                "additional_steps": ADDITIONAL_STEPS,
                "model": model_name,
                "paths": results,
                "source_commit": source_commit,
            },
        )
        _log("path_complete", model=model_name, path=name)


def run(*, repo_root, source_commit):
    from experiments.spec0053_stage_a2b_convergence import MODEL_DEVICE_MAP

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
