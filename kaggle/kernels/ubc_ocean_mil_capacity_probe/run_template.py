# Copyright 2026 HiperMaximus
# ruff: noqa: ANN001, ANN202, BLE001, DOC201, EM101, EM102, PLC0415, PLR0913, PLR0914, PLR2004, PLW0717, S404, T201, TRY003, TRY300
"""Generated wrapper for the exact Spec 0023 largest-WSI capacity probe."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path
from typing import cast

KAGGLE_UBC_OCEAN_MIL_CAPACITY_PROBE_READY = True
EMBEDDED_PAYLOAD_B64 = """
$embedded_payload_b64
"""
EMBEDDED_PAYLOAD_SHA256 = "$embedded_payload_sha256"
EMBEDDED_CONFIG_B64 = "$embedded_config_b64"
EMBEDDED_CONFIG_SHA256 = "$embedded_config_sha256"
INPUT_ROOT = Path("/kaggle/input")
WORKING_ROOT = Path("/kaggle/working")
PRIVATE_ROOT = WORKING_ROOT / ".spec0023_mil_capacity_probe"
OUTPUT_PATH = WORKING_ROOT / "spec0023_mil_capacity_probe.json"


def main() -> int:
    """Run only the authorized unchunked complete-bag capacity measurement."""
    session_start = time.monotonic()
    try:
        _ensure_latest_torch()
        source_root = _extract_payload(PRIVATE_ROOT)
        sys.path.insert(0, str(source_root / "src"))
        config = _embedded_config()
        result = _run_probe(source_root, config)
        elapsed = time.monotonic() - session_start
        session_limit = cast("int", config["session_limit_seconds"])
        required_reserve = cast(
            "int",
            config["required_deadline_reserve_seconds"],
        )
        result.update({
            "session_elapsed_seconds": elapsed,
            "session_limit_seconds": session_limit,
            "required_deadline_reserve_seconds": required_reserve,
            "remaining_session_seconds": session_limit - elapsed,
            "deadline_reserve_pass": elapsed <= session_limit - required_reserve,
            "saved_output_limit_bytes": config["saved_output_limit_bytes"],
            "projected_output_bytes": config["projected_output_bytes"],
            "output_allowlist": config["output_allowlist"],
            "test_release_status": config["test_release_status"],
            "binary_integrity_basis": config["binary_integrity_basis"],
        })
        shutil.rmtree(PRIVATE_ROOT, ignore_errors=True)
        _write_result(result, config)
        return 0
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        shutil.rmtree(PRIVATE_ROOT, ignore_errors=True)


def _ensure_latest_torch() -> None:
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


def _run_probe(source_root: Path, config: dict[str, object]) -> dict[str, object]:
    import torch

    from eqvae.data.supervised_latents import SupervisedLatentStore, WSIBagDataset
    from eqvae.models.supervised import AttentionMILClassifier
    from eqvae.training.supervised_pairing import make_paired_models

    if config.get("checkpoint_chunk_size") is not None:
        raise RuntimeError("The first capacity probe must be unchunked")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("Spec 0023 capacity probe requires exactly two CUDA devices")
    device_names = [torch.cuda.get_device_name(index) for index in range(2)]
    if any("T4" not in name for name in device_names):
        raise RuntimeError(f"Expected paired T4 devices, found {device_names!r}")

    probe_root = source_root / "probe"
    source_roots = _resolve_source_roots(config)
    loaded_by_model = {}
    coordinate_identity = None
    for model_name in ("normal_vae", "so2_vae"):
        with SupervisedLatentStore(
            catalog_path=probe_root / "physical_parts.csv",
            model_name=model_name,
            source_roots=source_roots,
        ) as store:
            loaded = WSIBagDataset(
                instance_path=probe_root / "wsi_instances.csv",
                bag_path=probe_root / "wsi_bags.csv",
                store=store,
            )[0]
        identity = tuple(
            (
                row.atlas_row_index,
                row.wsi_id,
                row.x,
                row.y,
                row.pointer.part,
                row.pointer.file_index,
            )
            for row in loaded.instances
        )
        if coordinate_identity is None:
            coordinate_identity = identity
        elif identity != coordinate_identity:
            raise RuntimeError("Normal and SO(2) probe coordinates differ")
        loaded_by_model[model_name] = loaded.latents

    if (
        coordinate_identity is None
        or len(coordinate_identity) != config["instance_count"]
    ):
        raise RuntimeError("Largest bag instance count differs after mounted reads")
    normal_model, so2_model = make_paired_models(
        AttentionMILClassifier,
        seed=cast("int", config["initialization_seed"]),
    )
    initial_state_equal = all(
        torch.equal(normal_value, so2_model.state_dict()[name])
        for name, normal_value in normal_model.state_dict().items()
    )
    if not initial_state_equal:
        raise RuntimeError("Paired MIL initial states differ")

    models = {"normal_vae": normal_model, "so2_vae": so2_model}
    devices = {"normal_vae": torch.device("cuda:0"), "so2_vae": torch.device("cuda:1")}
    target_index = cast("int", config["diagnosis_index"])
    weights = cast("dict[str, float]", config["diagnosis_class_weights"])
    try:
        warmup, measured, memory = _execute_gpu_probe(
            models=models,
            cpu_latents=loaded_by_model,
            devices=devices,
            target_index=target_index,
            class_weight=weights[str(target_index)],
            scaler_init_scale=cast("float", config["grad_scaler_init_scale"]),
            scaler_growth_interval=cast(
                "int",
                config["grad_scaler_growth_interval"],
            ),
        )
    except torch.cuda.OutOfMemoryError as error:
        memory = {
            model_name: {
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "device_total_bytes": torch.cuda.get_device_properties(
                    device,
                ).total_memory,
            }
            for model_name, device in devices.items()
        }
        return {
            "schema_version": "spec0023.mil_capacity_probe.v1",
            "status": "capacity_failed",
            "failure_kind": "cuda_out_of_memory",
            "fits": False,
            "selected_checkpoint_chunk_size": "fallback_required",
            "error": str(error),
            "config_sha256": hashlib.sha256(_canonical_json(config)).hexdigest(),
            "wsi_id": config["wsi_id"],
            "split": config["split"],
            "diagnosis_index": target_index,
            "instance_count": config["instance_count"],
            "part_counts": config["part_counts"],
            "mounted_binary_bytes": config["mounted_binary_bytes"],
            "kernel_sources": config["kernel_sources"],
            "device_names": device_names,
            "cuda_device_count": torch.cuda.device_count(),
            "initial_state_equal": initial_state_equal,
            "precision": config["precision"],
            "grad_scaler_init_scale": config["grad_scaler_init_scale"],
            "grad_scaler_growth_interval": config["grad_scaler_growth_interval"],
            "checkpoint_chunk_size": None,
            "memory": memory,
        }
    all_steps = (*warmup.values(), *measured.values())
    finite = all(cast("bool", row["finite_gradients"]) for row in all_steps)
    attention_exact = all(
        cast("int", row["attention_length"]) == config["instance_count"]
        and abs(cast("float", row["attention_sum"]) - 1.0) < 1e-4
        for row in all_steps
    )
    no_skips = all(not cast("bool", row["grad_scaler_skipped"]) for row in all_steps)
    fits = finite and attention_exact and no_skips
    return {
        "schema_version": "spec0023.mil_capacity_probe.v1",
        "status": "complete" if fits else "capacity_failed",
        "failure_kind": None if fits else "nonfinite_or_skipped_step",
        "fits": fits,
        "selected_checkpoint_chunk_size": None if fits else "fallback_required",
        "config_sha256": hashlib.sha256(_canonical_json(config)).hexdigest(),
        "wsi_id": config["wsi_id"],
        "split": config["split"],
        "diagnosis_index": target_index,
        "instance_count": config["instance_count"],
        "part_counts": config["part_counts"],
        "mounted_binary_bytes": config["mounted_binary_bytes"],
        "kernel_sources": config["kernel_sources"],
        "device_names": device_names,
        "cuda_device_count": torch.cuda.device_count(),
        "initial_state_equal": initial_state_equal,
        "precision": config["precision"],
        "grad_scaler_init_scale": config["grad_scaler_init_scale"],
        "grad_scaler_growth_interval": config["grad_scaler_growth_interval"],
        "checkpoint_chunk_size": None,
        "warmup": warmup,
        "measured": measured,
        "memory": memory,
    }


def _execute_gpu_probe(
    *,
    models,
    cpu_latents,
    devices,
    target_index,
    class_weight,
    scaler_init_scale,
    scaler_growth_interval,
):
    import torch

    gpu_latents = {}
    for model_name in ("normal_vae", "so2_vae"):
        device = devices[model_name]
        models[model_name] = models[model_name].to(device)
        gpu_latents[model_name] = cpu_latents.pop(model_name).to(device)
    optimizers = {
        model_name: torch.optim.AdamW(
            models[model_name].parameters(),
            lr=3e-4,
            weight_decay=1e-4,
        )
        for model_name in models
    }
    scalers = {
        model_name: torch.amp.GradScaler(
            "cuda",
            init_scale=scaler_init_scale,
            growth_interval=scaler_growth_interval,
        )
        for model_name in models
    }
    warmup = _paired_step(
        models=models,
        latents=gpu_latents,
        optimizers=optimizers,
        scalers=scalers,
        devices=devices,
        target_index=target_index,
        class_weight=class_weight,
    )
    for device in devices.values():
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    baseline = {
        model_name: {
            "allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
        }
        for model_name, device in devices.items()
    }
    measured = _paired_step(
        models=models,
        latents=gpu_latents,
        optimizers=optimizers,
        scalers=scalers,
        devices=devices,
        target_index=target_index,
        class_weight=class_weight,
    )
    memory = {}
    for model_name, device in devices.items():
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        memory[model_name] = {
            "baseline": baseline[model_name],
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "post_step_free_bytes": free_bytes,
            "device_total_bytes": total_bytes,
        }
    return warmup, measured, memory


def _paired_step(
    *,
    models,
    latents,
    optimizers,
    scalers,
    devices,
    target_index,
    class_weight,
):
    import torch
    from torch.nn import functional

    for optimizer in optimizers.values():
        optimizer.zero_grad(set_to_none=True)
    starts = {name: torch.cuda.Event(enable_timing=True) for name in models}
    ends = {name: torch.cuda.Event(enable_timing=True) for name in models}
    scale_before = {name: scalers[name].get_scale() for name in models}
    outputs = {}
    wall_start = time.perf_counter()
    for model_name in ("normal_vae", "so2_vae"):
        device = devices[model_name]
        with torch.cuda.device(device):
            starts[model_name].record()
            with torch.autocast("cuda", dtype=torch.float16, cache_enabled=True):
                logits, attention = models[model_name](
                    latents[model_name],
                    checkpoint_chunk_size=None,
                )
            target = torch.tensor([target_index], device=device)
            unweighted_loss = functional.cross_entropy(logits.float()[None, :], target)
            loss = unweighted_loss * class_weight
            scalers[model_name].scale(loss).backward()
            outputs[model_name] = (loss, attention)
    for model_name in ("normal_vae", "so2_vae"):
        with torch.cuda.device(devices[model_name]):
            scalers[model_name].step(optimizers[model_name])
            scalers[model_name].update()
            ends[model_name].record()
    for device in devices.values():
        torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - wall_start
    rows = {}
    for model_name in ("normal_vae", "so2_vae"):
        loss, attention = outputs[model_name]
        gradients = [parameter.grad for parameter in models[model_name].parameters()]
        rows[model_name] = {
            "loss": float(loss.detach().item()),
            "attention_length": int(attention.numel()),
            "attention_sum": float(attention.detach().float().sum().item()),
            "finite_gradients": bool(
                gradients
                and all(
                    gradient is not None and torch.isfinite(gradient).all().item()
                    for gradient in gradients
                ),
            ),
            "grad_scaler_skipped": scalers[model_name].get_scale()
            < scale_before[model_name],
            "device_step_milliseconds": float(
                starts[model_name].elapsed_time(ends[model_name]),
            ),
            "paired_wall_seconds": wall_seconds,
        }
    return rows


def _resolve_source_roots(config: dict[str, object]) -> dict[str, Path]:
    records = cast("list[dict[str, object]]", config["source_records"])
    roots = {}
    for record in records:
        source = cast("str", record["kaggle_source"])
        binaries = cast("list[dict[str, object]]", record["binaries"])
        parents = set()
        for binary in binaries:
            name = cast("str", binary["name"])
            matches = [path for path in INPUT_ROOT.rglob(name) if path.is_file()]
            if len(matches) != 1:
                raise RuntimeError(f"Expected one mounted {name}, found {len(matches)}")
            parents.add(matches[0].parent)
        if len(parents) != 1:
            raise RuntimeError(f"Mounted files for {source} do not share one root")
        roots[source] = parents.pop()
    return roots


def _extract_payload(destination: Path) -> Path:
    payload = base64.b64decode(EMBEDDED_PAYLOAD_B64.encode("ascii"))
    if hashlib.sha256(payload).hexdigest() != EMBEDDED_PAYLOAD_SHA256:
        raise RuntimeError("Spec 0023 embedded payload hash mismatch")
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name in archive.namelist():
            path = Path(name)
            if path.is_absolute() or ".." in path.parts:
                raise RuntimeError(f"Unsafe embedded path: {name}")
        archive.extractall(destination)
    return destination


def _embedded_config() -> dict[str, object]:
    encoded = base64.b64decode(EMBEDDED_CONFIG_B64.encode("ascii"))
    if hashlib.sha256(encoded).hexdigest() != EMBEDDED_CONFIG_SHA256:
        raise RuntimeError("Spec 0023 embedded config hash mismatch")
    value = cast("object", json.loads(encoded))
    if not isinstance(value, dict):
        raise TypeError("Spec 0023 embedded config must be an object")
    return cast("dict[str, object]", value)


def _canonical_json(payload: dict[str, object]) -> bytes:
    return f"{json.dumps(payload, sort_keys=True, separators=(',', ':'))}\n".encode()


def _write_result(result: dict[str, object], config: dict[str, object]) -> None:
    result["output_bytes"] = 0
    while True:
        encoded = _canonical_json(result)
        if result["output_bytes"] == len(encoded):
            break
        result["output_bytes"] = len(encoded)
    if len(encoded) > cast("int", config["projected_output_bytes"]) or len(
        encoded,
    ) > cast("int", config["saved_output_limit_bytes"]):
        raise RuntimeError("Capacity-probe JSON exceeds its sealed output budget")
    OUTPUT_PATH.write_bytes(encoded)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
