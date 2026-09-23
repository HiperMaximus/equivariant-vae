# Copyright 2026 HiperMaximus
"""Measure whether FP16 latent storage changes the AMP MIL training step."""

from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from experiments.spec0054_mil_a0_probe import (
    CLASSIFIER_SEED,
    GraphInstance,
    PatchRow,
    _load_frozen_model,
    _load_patch_rows,
    _make_classifier,
    _make_optimizer,
    _read_patch_batch,
    _sha256,
    _unique_input,
)

OUTPUT_ROOT = Path("/kaggle/working/spec0054_fp16_storage_probe")
CONTRACT_RELATIVE = Path("docs/data/spec0054_a0_contract.json")
PROBE_WSI_ID = 52108
ENCODER_BATCH_SIZE = 8


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _tensor_delta(reference: Any, candidate: Any, torch: Any) -> dict[str, Any]:
    left = reference.detach().float().reshape(-1)
    right = candidate.detach().float().reshape(-1)
    difference = right - left
    absolute = difference.abs()
    reference_norm = torch.linalg.vector_norm(left.double())
    difference_norm = torch.linalg.vector_norm(difference.double())
    return {
        "element_count": left.numel(),
        "reference_dtype": str(reference.dtype),
        "candidate_dtype": str(candidate.dtype),
        "exact_equal_count": int((left == right).sum().item()),
        "maximum_absolute_difference": float(absolute.max().item()),
        "mean_absolute_difference": float(absolute.mean().item()),
        "absolute_difference_p50": float(torch.quantile(absolute, 0.50).item()),
        "absolute_difference_p90": float(torch.quantile(absolute, 0.90).item()),
        "absolute_difference_p99": float(torch.quantile(absolute, 0.99).item()),
        "rmse": float(torch.sqrt(torch.mean(difference.double().square())).item()),
        "relative_l2": float((difference_norm / reference_norm.clamp_min(1e-300)).item()),
        "reference_nonfinite_count": int((~torch.isfinite(left)).sum().item()),
        "candidate_nonfinite_count": int((~torch.isfinite(right)).sum().item()),
        "nonzero_to_zero_count": int(((left != 0) & (right == 0)).sum().item()),
        "reference_minimum": float(left.min().item()),
        "reference_maximum": float(left.max().item()),
        "candidate_minimum": float(right.min().item()),
        "candidate_maximum": float(right.max().item()),
    }


def _vector_comparison(
    left_values: dict[str, Any],
    right_values: dict[str, Any],
    torch: Any,
) -> dict[str, Any]:
    left = torch.cat(
        [left_values[name].detach().cpu().float().reshape(-1) for name in left_values]
    )
    right = torch.cat(
        [right_values[name].detach().cpu().float().reshape(-1) for name in left_values]
    )
    difference = right - left
    left64 = left.double()
    right64 = right.double()
    left_norm = torch.linalg.vector_norm(left64)
    right_norm = torch.linalg.vector_norm(right64)
    denominator = left_norm * right_norm
    cosine = torch.dot(left64, right64) / denominator if denominator > 0 else torch.tensor(0.0)
    return {
        "element_count": left.numel(),
        "left_l2": float(left_norm.item()),
        "right_l2": float(right_norm.item()),
        "maximum_absolute_difference": float(difference.abs().max().item()),
        "rmse": float(torch.sqrt(torch.mean(difference.double().square())).item()),
        "cosine": float(cosine.item()),
        "left_nonfinite_count": int((~torch.isfinite(left)).sum().item()),
        "right_nonfinite_count": int((~torch.isfinite(right)).sum().item()),
    }


def _parameter_updates(
    initial: dict[str, Any],
    final: dict[str, Any],
    parameter_names: tuple[str, ...],
) -> dict[str, Any]:
    return {
        name: final[name].detach().cpu().float() - initial[name].detach().cpu().float()
        for name in parameter_names
    }


def _training_step(
    classifier_state: dict[str, Any],
    latent: Any,
    graph: Any,
    target: Any,
    *,
    compiled: bool,
    torch: Any,
) -> dict[str, Any]:
    device = latent.device
    model = _make_classifier(classifier_state, device, torch)
    optimizer = _make_optimizer(model, torch)
    parameter_names = tuple(name for name, _parameter in model.named_parameters())

    def forward(input_latent: Any, input_graph: Any, input_target: Any) -> Any:
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(input_latent, input_graph)
            loss = torch.nn.functional.cross_entropy(
                logits.float().unsqueeze(0), input_target
            )
        return logits, loss

    execute = (
        torch.compile(
            forward,
            backend="inductor",
            fullgraph=True,
            dynamic=None,
            mode="max-autotune-no-cudagraphs",
        )
        if compiled
        else forward
    )
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    logits, loss = execute(latent, graph, target)
    loss.backward()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    gradients = {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    optimizer.step()
    final_state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    updates = _parameter_updates(classifier_state, final_state, parameter_names)
    result = {
        "input_dtype": str(latent.dtype),
        "logits": [float(value) for value in logits.detach().float().cpu()],
        "loss": float(loss.detach().float().cpu()),
        "seconds_including_cold_compile": elapsed,
        "gradients": gradients,
        "updates": updates,
    }
    return result


def _compare_steps(left: dict[str, Any], right: dict[str, Any], torch: Any) -> dict[str, Any]:
    left_logits = torch.tensor(left["logits"])
    right_logits = torch.tensor(right["logits"])
    return {
        "logits": _tensor_delta(left_logits, right_logits, torch),
        "loss_absolute_difference": abs(left["loss"] - right["loss"]),
        "gradients": _vector_comparison(left["gradients"], right["gradients"], torch),
        "adamw_updates": _vector_comparison(left["updates"], right["updates"], torch),
        "left_seconds_including_cold_compile": left["seconds_including_cold_compile"],
        "right_seconds_including_cold_compile": right["seconds_including_cold_compile"],
    }


def _encode_probe_bag(
    rows: list[PatchRow],
    binary_path: Path,
    state_paths: dict[str, Path],
    torch: Any,
) -> dict[str, Any]:
    models = {
        "normal_vae": _load_frozen_model(
            "normal_vae", state_paths["normal_vae"], torch.device("cuda:0"), torch
        ),
        "so2_vae": _load_frozen_model(
            "so2_vae", state_paths["so2_vae"], torch.device("cuda:1"), torch
        ),
    }
    parts: dict[str, list[Any]] = defaultdict(list)
    for start in range(0, len(rows), ENCODER_BATCH_SIZE):
        selected = rows[start : start + ENCODER_BATCH_SIZE]
        images = (
            torch.from_numpy(_read_patch_batch(binary_path, selected))
            .float()
            .div_(255)
            .mul_(2)
            .sub_(1)
        )
        for name, model in models.items():
            device = next(model.parameters()).device
            with torch.inference_mode():
                mean, _logvar = model.encode(images.to(device))
            parts[name].append(mean.detach().cpu())
    output = {name: torch.cat(values) for name, values in parts.items()}
    del models
    torch.cuda.empty_cache()
    return output


def _branch_probe(
    latent_cpu: Any,
    rows: list[PatchRow],
    classifier_state: dict[str, Any],
    device_index: int,
    torch: Any,
) -> dict[str, Any]:
    from eqvae.models.local_global_mil import build_local_attention_graph

    device = torch.device(f"cuda:{device_index}")
    graph = build_local_attention_graph(
        [GraphInstance(row.wsi_id, row.x, row.y) for row in rows],
        expected_instance_count=len(rows),
    ).to(device)
    target = torch.tensor([rows[0].label], device=device)
    latent32 = latent_cpu.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
    stored = latent_cpu.to(dtype=torch.float16).contiguous().numpy()
    storage_path = OUTPUT_ROOT / f"probe_latent_cuda{device_index}.fp16.bin"
    stored.tofile(storage_path)
    reloaded = np.fromfile(storage_path, dtype=np.float16).reshape(stored.shape)
    storage_bytes = storage_path.stat().st_size
    storage_path.unlink()
    latent16 = torch.from_numpy(reloaded).to(
        device=device,
        memory_format=torch.channels_last,
    )

    first_layer_model = _make_classifier(classifier_state, device, torch)
    first_convolution = first_layer_model.patch_encoder.layers[0]
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        first32 = first_convolution(latent32)
        first16 = first_convolution(latent16)
        eager32 = first_layer_model(latent32, graph)
        eager16 = first_layer_model(latent16, graph)
    del first_layer_model
    torch.cuda.empty_cache()

    eager_step32 = _training_step(
        classifier_state, latent32, graph, target, compiled=False, torch=torch
    )
    eager_step16 = _training_step(
        classifier_state, latent16, graph, target, compiled=False, torch=torch
    )
    compiled_step32 = _training_step(
        classifier_state, latent32, graph, target, compiled=True, torch=torch
    )
    compiled_step16 = _training_step(
        classifier_state, latent16, graph, target, compiled=True, torch=torch
    )
    result = {
        "storage_roundtrip": {
            "bytes": storage_bytes,
            "exact": bool(np.array_equal(stored, reloaded)),
        },
        "stored_latent_cast": _tensor_delta(latent32, latent16, torch),
        "first_convolution_under_autocast": _tensor_delta(first32, first16, torch),
        "eager_forward_logits": _tensor_delta(eager32, eager16, torch),
        "eager_training_step": _compare_steps(eager_step32, eager_step16, torch),
        "compiled_training_step": _compare_steps(
            compiled_step32, compiled_step16, torch
        ),
        "fp32_eager_vs_compiled": _compare_steps(
            eager_step32, compiled_step32, torch
        ),
        "fp16_eager_vs_compiled": _compare_steps(
            eager_step16, compiled_step16, torch
        ),
    }
    del latent16, latent32, graph
    torch.cuda.empty_cache()
    return result


def run(*, repo_root: Path, source_commit: str) -> int:
    import torch

    started = time.perf_counter()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    contract_path = repo_root / CONTRACT_RELATIVE
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    sources = contract["sources"]
    train_csv = _unique_input("ubc_train_shuffled.csv", sources["train_csv_sha256"])
    train_binary = _unique_input("ubc_train_shuffled.bin")
    state_paths = {
        "normal_vae": _unique_input(
            "normal_vae_state.pt", sources["normal_state_file_sha256"]
        ),
        "so2_vae": _unique_input(
            "so2_vae_state.pt", sources["so2_state_file_sha256"]
        ),
    }
    train_rows, _audit = _load_patch_rows(train_csv, "vae_train", 300000)
    rows = sorted(
        (row for row in train_rows if row.wsi_id == PROBE_WSI_ID),
        key=lambda row: (row.y, row.x, row.source_row_index),
    )
    latent_bags = _encode_probe_bag(rows, train_binary, state_paths, torch)

    random.seed(CLASSIFIER_SEED)
    np.random.seed(CLASSIFIER_SEED)
    torch.manual_seed(CLASSIFIER_SEED)
    from eqvae.models.local_global_mil import LocalGlobalMILClassifier

    classifier_state = {
        name: value.detach().cpu().clone()
        for name, value in LocalGlobalMILClassifier().state_dict().items()
    }
    branches = {
        name: _branch_probe(
            latent_bags[name], rows, classifier_state, device_index, torch
        )
        for device_index, name in enumerate(("normal_vae", "so2_vae"))
    }
    result = {
        "schema_version": "spec0054.fp16_storage_probe.v1",
        "source_commit": source_commit,
        "contract_sha256": _sha256(contract_path),
        "wsi_id": PROBE_WSI_ID,
        "patch_count": len(rows),
        "diagnosis_index": rows[0].label,
        "vae_compute_dtype": "torch.float32",
        "stored_candidate_dtype": "torch.float16",
        "classifier_input_policy": "direct FP16 under CUDA autocast FP16",
        "gradient_scaler_used": False,
        "branches": branches,
        "elapsed_seconds": time.perf_counter() - started,
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        },
    }
    _write_json(OUTPUT_ROOT / "fp16_storage_probe.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


__all__ = ["run"]
