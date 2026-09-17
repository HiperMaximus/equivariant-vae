# Copyright 2026 HiperMaximus
"""Reusable full-latent path pieces for the forthcoming Stage A2 runner.

The numerical calibration which used this module is closed. The scientific
runner will replace its orchestration in this file; these helpers retain only
the decoder-energy mechanics that its frozen contract reuses.
"""

import json
import math
import os
import time
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

DATASET_ROOT = Path("/kaggle/input/datasets")
WORKING_ROOT = Path("/kaggle/working")
SELECTOR_PATH = Path("configs/spec0001/fixed_25_validation_patches.json")
MODEL_KINDS = {
    "normal_vae": "non_eq_vae_translatable",
    "so2_vae": "so2_vae_fixed",
}
MODEL_DEVICE_MAP = {"normal_vae": 0, "so2_vae": 1}
PATCH_BYTES = 3 * 256 * 256
HEADER_BYTES = 64
_STARTED = time.perf_counter()


def _log(event: str, **values: object) -> None:
    print(
        json.dumps(
            {
                "elapsed_seconds": round(time.perf_counter() - _STARTED, 3),
                "event": event,
                **values,
            },
            allow_nan=False,
            sort_keys=True,
        ),
        flush=True,
    )


def _write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")


def _read_selected_patch_bytes(handle, rows, selected_ranks):
    payloads = []
    for rank in selected_ranks:
        row = rows[rank]
        handle.seek(HEADER_BYTES + int(row["file_index"]) * PATCH_BYTES)
        payloads.append(handle.read(PATCH_BYTES))
    return payloads


def _load_patches(repo_root, selected_ranks, *, np, torch):
    selector = json.loads((repo_root / SELECTOR_PATH).read_text(encoding="utf-8"))
    rows = selector["selectors"]
    with Path(selector["source"]["bin_path"]).open("rb") as handle:
        payloads = _read_selected_patch_bytes(handle, rows, selected_ranks)
    arrays = [
        np.frombuffer(raw, dtype=np.uint8).reshape(3, 256, 256).copy()
        for raw in payloads
    ]
    patches = (
        torch.from_numpy(np.stack(arrays)).to(torch.float32).div(255).mul(2).sub(1)
    )
    return patches, [rows[rank] for rank in selected_ranks]


def _encode(model, images, *, batch_size, torch):
    means = []
    with torch.no_grad():
        for start in range(0, images.shape[0], batch_size):
            mean, _ = model.encode(images[start : start + batch_size])
            means.append(mean.detach())
    return torch.cat(means)


def _load_frozen_model(model_name, state_path, device, *, torch):
    from eqvae.models.registry import build_model

    state = torch.load(state_path, map_location="cpu", weights_only=True)
    model = build_model(MODEL_KINDS[model_name])
    model.load_state_dict(state, strict=True)
    return model.to(device).eval().requires_grad_(False)


def _latent_line(left, right, segments, *, torch):
    times = torch.linspace(
        0.0,
        1.0,
        segments + 1,
        device=left.device,
        dtype=left.dtype,
    ).reshape(-1, *(1 for _ in left.shape[1:]))
    return left + times * (right - left)


def _decoder_edge_energy(model, latents, total_segments, *, torch):
    decoded = model.decode(latents)
    differences = decoded[1:] - decoded[:-1]
    return total_segments * differences.flatten(1).square().mean(dim=1).sum()


def _prepare_decoder_runtime(model, probe_latents, *, torch):
    """Materialize frozen kernels and compile the fixed eight-edge closure."""
    materialize = getattr(model, "materialize_frozen_decoder_kernels", None)
    if materialize is not None:
        materialize()

    def eager(latents, total_segments):
        return _decoder_edge_energy(model, latents, total_segments, torch=torch)

    compiled = torch.compile(eager, dynamic=False, fullgraph=True, mode="default")
    with torch.enable_grad():
        variable = probe_latents.detach().clone().requires_grad_(True)
        compiled(variable, probe_latents.new_tensor(1.0)).backward()
    if probe_latents.device.type == "cuda":
        torch.cuda.synchronize(probe_latents.device)
    return eager, compiled


def _path_from_interior(left, interior, right, *, torch):
    return torch.cat((left, interior, right), dim=0)


def _chunked_path_energy(
    left,
    interior,
    right,
    *,
    edge_energy,
    total_segments,
    chunk_segments,
    backward,
    torch,
):
    """Evaluate one path objective in bounded decoder batches."""
    total = 0.0
    with torch.set_grad_enabled(backward):
        for start in range(0, total_segments, chunk_segments):
            stop = start + chunk_segments
            chunks = []
            if start == 0:
                chunks.append(left)
            interior_start = max(start - 1, 0)
            interior_stop = min(stop, total_segments - 1)
            if interior_start < interior_stop:
                chunks.append(interior[interior_start:interior_stop])
            if stop == total_segments:
                chunks.append(right)
            energy = edge_energy(
                torch.cat(chunks, dim=0),
                interior.new_tensor(float(total_segments)),
            )
            total += float(energy.detach())
            if backward:
                energy.backward()
    return total


def _full_latent_learning_rate(
    base_learning_rate, latent_dimension, reference_dimension
):
    return base_learning_rate * math.sqrt(reference_dimension / latent_dimension)


def _optimize_full_latent_path(
    left,
    right,
    *,
    learning_rate,
    learning_rate_reference_dimension,
    path_segments,
    optimizer_steps,
    chunk_segments,
    eager_edge_energy,
    optimized_edge_energy,
    torch,
):
    """Optimize direct full-latent knots and retain the lowest-energy iterate."""
    line = _latent_line(left, right, path_segments, torch=torch)
    interior = line[1:-1].detach().clone().requires_grad_(True)
    effective_learning_rate = _full_latent_learning_rate(
        learning_rate,
        interior[0].numel(),
        learning_rate_reference_dimension,
    )
    optimizer = torch.optim.Adam([interior], lr=effective_learning_rate)
    best_energy = math.inf
    best_interior = interior.detach().clone()
    best_iteration = 0

    for iteration in range(optimizer_steps):
        optimizer.zero_grad(set_to_none=True)
        energy = _chunked_path_energy(
            left,
            interior,
            right,
            edge_energy=optimized_edge_energy,
            total_segments=path_segments,
            chunk_segments=chunk_segments,
            backward=True,
            torch=torch,
        )
        if energy < best_energy:
            best_energy = energy
            best_interior = interior.detach().clone()
            best_iteration = iteration
        optimizer.step()

    final_energy = _chunked_path_energy(
        left,
        interior,
        right,
        edge_energy=eager_edge_energy,
        total_segments=path_segments,
        chunk_segments=chunk_segments,
        backward=False,
        torch=torch,
    )
    if final_energy < best_energy:
        best_energy = final_energy
        best_interior = interior.detach().clone()
        best_iteration = optimizer_steps
    return {
        "best_energy": best_energy,
        "best_iteration": best_iteration,
        "best_path": _path_from_interior(left, best_interior, right, torch=torch),
        "effective_learning_rate": effective_learning_rate,
        "final_energy": final_energy,
    }
