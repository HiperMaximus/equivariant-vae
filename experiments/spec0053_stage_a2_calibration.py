# Copyright 2026 HiperMaximus
# PyTorch experiment payloads are intentionally dynamic dictionaries.
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false
"""One-shot scientific runner for Spec 0053 Stage A2."""

import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

WORKING_ROOT = Path("/kaggle/working")
OUTPUT_ROOT = WORKING_ROOT / "functional_geometry_stage_a2"
CONTRACT_PATH = Path("docs/data/functional_geometry_stage_a2_contract.json")
SELECTOR_PATH = Path("configs/spec0001/fixed_25_validation_patches.json")
WEIGHT_DATASET_ROOT = Path(
    "/kaggle/input/datasets/maximusshtefan/eqvae-frozen-vae-weights-v1"
)
CHECKPOINT_DATASET_ROOT = Path(
    "/kaggle/input/eqvae-stage-a2-complete-checkpoints"
)
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
    model = model.to(device).eval().requires_grad_(False)
    if model_name == "so2_vae":
        model.materialize_frozen_decoder_kernels()
    return model


def _rms(values, *, torch):
    return torch.sqrt(values.square().mean())


def _decode_knots(model, latents, *, torch):
    decoded = []
    with torch.no_grad():
        for start in range(0, latents.shape[0], 8):
            decoded.append(model.decode(latents[start : start + 8]))
    return torch.cat(decoded)


def _save_path(tensors, name, path, decoded, *, torch):
    tensors["paths"][name] = {
        "decoded_raw_fp16": decoded.detach().to("cpu", torch.float16),
        "latents": path.detach().to("cpu"),
    }


def _load_completed_path(model, *, model_name, name, device, tensors, torch):
    saved = torch.load(
        CHECKPOINT_DATASET_ROOT / f"{model_name}__{name}.pt",
        map_location="cpu",
        weights_only=True,
    )
    path = saved["best_path"].to(device)
    decoded = _decode_knots(model, path, torch=torch)
    _save_path(tensors, name, path, decoded, torch=torch)
    _log("path_loaded", model=model_name, path=name)
    return path, decoded, saved["metrics"]


def _anchor_metrics(encoded, prescribed, decoded_encoded, decoded_prescribed, *, torch):
    rows = []
    for turn in range(4):
        rows.append(
            {
                "encoded_vs_prescribed_decoded_rms": float(
                    _rms(decoded_encoded[turn] - decoded_prescribed[turn], torch=torch)
                ),
                "encoded_vs_rotated_origin_rms": float(
                    _rms(
                        decoded_encoded[turn]
                        - torch.rot90(decoded_encoded[0], turn, (-2, -1)),
                        torch=torch,
                    )
                ),
                "prescribed_vs_rotated_origin_rms": float(
                    _rms(
                        decoded_prescribed[turn]
                        - torch.rot90(
                            decoded_prescribed[0],
                            turn,
                            (-2, -1),
                        ),
                        torch=torch,
                    )
                ),
                "latent_encoded_vs_prescribed_l2": float(
                    torch.linalg.vector_norm((encoded[turn] - prescribed[turn]).flatten())
                ),
                "turn": turn,
            }
        )
    return rows


def _bridge_metrics(decoded, *, torch):
    left = decoded[0]
    right = decoded[-1]
    to_left = torch.sqrt((decoded - left).flatten(1).square().mean(1))
    to_right = torch.sqrt((decoded - right).flatten(1).square().mean(1))
    return {
        "endpoint_decoded_rms": float(_rms(right - left, torch=torch)),
        "maximum_decoded_diameter_from_left_rms": float(to_left.max()),
        "maximum_decoded_diameter_from_right_rms": float(to_right.max()),
        "maximum_decoded_diameter_rms": float(torch.maximum(to_left, to_right).max()),
    }


def _decoded_covariance(paths, *, torch):
    rows = []
    for side in range(4):
        following = (side + 1) % 4
        pointwise = torch.sqrt(
            (
                paths[following]["decoded"]
                - torch.rot90(paths[side]["decoded"], 1, (-2, -1))
            )
            .flatten(1)
            .square()
            .mean(1)
        )
        tangentwise = torch.sqrt(
            (
                (paths[following]["decoded"][1:] - paths[following]["decoded"][:-1])
                - torch.rot90(
                    paths[side]["decoded"][1:] - paths[side]["decoded"][:-1],
                    1,
                    (-2, -1),
                )
            )
            .flatten(1)
            .square()
            .mean(1)
        )
        rows.append(
            {
                "decoded_curve_rms": float(_rms(pointwise, torch=torch)),
                "decoded_knot_rms": [float(value) for value in pointwise],
                "decoded_knot_rms_max": float(pointwise.max()),
                "decoded_tangent_rms": float(_rms(tangentwise, torch=torch)),
                "from_side": side,
                "to_side": following,
            }
        )
    return rows


def _dft_metrics(observed, rollout, *, torch):
    def sectors(values):
        c0 = (values[0] + values[1] + values[2] + values[3]) / 2.0
        c2 = (values[0] - values[1] + values[2] - values[3]) / 2.0
        cosine = (values[0] - values[2]) / math.sqrt(2.0)
        sine = (values[1] - values[3]) / math.sqrt(2.0)
        return {"q0": c0, "q2": c2, "q1_cos": cosine, "q1_sin": sine}

    observed_sectors = sectors(observed)
    rollout_sectors = sectors(rollout)
    observed_centered_energy = sum(
        values.square().sum()
        for name, values in observed_sectors.items()
        if name != "q0"
    )
    rollout_centered_energy = sum(
        values.square().sum()
        for name, values in rollout_sectors.items()
        if name != "q0"
    )
    observed_degenerate = bool(observed_centered_energy == 0)
    rollout_degenerate = bool(rollout_centered_energy == 0)
    return {
        "observed_centered_energy": float(observed_centered_energy),
        "observed_degenerate": observed_degenerate,
        "rollout_centered_energy": float(rollout_centered_energy),
        "rollout_degenerate": rollout_degenerate,
        "sectors": {
            name: {
                "observed_fraction": (
                    None
                    if observed_degenerate
                    else float(value.square().sum() / observed_centered_energy)
                ),
                "prediction_rms": float(
                    _rms(value - rollout_sectors[name], torch=torch)
                ),
                "rollout_fraction": (
                    None
                    if rollout_degenerate
                    else float(
                        rollout_sectors[name].square().sum()
                        / rollout_centered_energy
                    )
                ),
            }
            for name, value in observed_sectors.items()
        },
    }


def _latent_from_coordinates(U0, coordinates):
    return (coordinates.reshape(-1, *([1] * (U0.ndim - 1))) * U0).sum(0)


def _decoded_tangent(model, latent, tangent, *, torch):
    def decode_one(point):
        return model.decode(point.unsqueeze(0)).squeeze(0)

    return torch.func.jvp(decode_one, (latent,), (tangent,))[1]


def _decoded_frame(model, latent, U0, coordinates, *, torch):
    directions = torch.einsum("ij,i...->j...", coordinates, U0)
    responses = []
    for start in range(0, directions.shape[0], 4):
        tangent = directions[start : start + 4]
        primals = latent.unsqueeze(0).repeat(tangent.shape[0], *([1] * latent.ndim))
        _, response = torch.func.jvp(model.decode, (primals,), (tangent,))
        responses.append(response)
    return torch.cat(responses)


def _transport_metrics(model, rollout, transport, U0, *, torch):
    if transport["intrinsic_status"] != "defined":
        return {
            "intrinsic_status": "undefined",
            "orthogonal_holonomy_status": "not_evaluated_free_rollout_not_explicitly_closed",
        }
    coordinates = transport["coordinates"]
    frames = transport["frames"]
    latents = rollout["latents"]
    initial_tangent = _latent_from_coordinates(U0, coordinates[0])
    returned_tangent = _latent_from_coordinates(U0, coordinates[-1])
    initial_decoded = _decoded_tangent(model, latents[0], initial_tangent, torch=torch)
    returned_decoded = _decoded_tangent(
        model,
        latents[-1],
        returned_tangent,
        torch=torch,
    )
    denominator = (
        torch.linalg.vector_norm(initial_decoded.flatten())
        * torch.linalg.vector_norm(returned_decoded.flatten())
    ).clamp_min(torch.finfo(torch.float32).tiny)
    cosine = torch.dot(initial_decoded.flatten(), returned_decoded.flatten()) / denominator
    initial_frame = _decoded_frame(model, latents[0], U0, frames[0], torch=torch)
    returned_frame = _decoded_frame(model, latents[-1], U0, frames[-1], torch=torch)
    initial_columns = initial_frame.flatten(1).T.to(torch.float64)
    returned_columns = returned_frame.flatten(1).T.to(torch.float64)
    return_map = initial_columns.T @ returned_columns / initial_columns.shape[0]
    frame_residual = returned_columns - initial_columns @ return_map
    singular_values = torch.linalg.svdvals(return_map.to(torch.float64))
    path_singular_values = transport["singular_values"]
    path_conditions = path_singular_values[:, 0] / path_singular_values[:, -1]
    return {
        "decoded_tangent_return_rms": float(
            _rms(initial_decoded - returned_decoded, torch=torch)
        ),
        "frame_return_determinant": float(torch.linalg.det(return_map.to(torch.float64))),
        "frame_return_map": [[float(value) for value in row] for row in return_map],
        "frame_return_orthogonality_defect": float(
            torch.linalg.matrix_norm(
                return_map.T @ return_map
                - torch.eye(return_map.shape[0], device=return_map.device, dtype=return_map.dtype)
            )
        ),
        "frame_return_out_of_start_subspace_rms": float(
            _rms(frame_residual, torch=torch)
        ),
        "frame_return_singular_values": [float(value) for value in singular_values],
        "intrinsic_status": "defined",
        "orthogonal_holonomy_status": "not_evaluated_free_rollout_not_explicitly_closed",
        "path_condition_max": float(path_conditions.max()),
        "path_min_singular_value": float(path_singular_values[:, -1].min()),
        "path_singular_values": [
            [float(value) for value in row] for row in path_singular_values
        ],
        "return_tangent_angle_radians": float(torch.acos(cosine.clamp(-1.0, 1.0))),
        "return_tangent_cosine": float(cosine),
        "return_tangent_norm_ratio": float(
            torch.linalg.vector_norm(returned_decoded.flatten())
            / torch.linalg.vector_norm(initial_decoded.flatten()).clamp_min(torch.finfo(torch.float32).tiny)
        ),
    }


def _free_rollout(model, side0, z1, *, torch):
    from eqvae.evaluation.functional_geometry_stage_a2 import (
        construct_path_chart,
        estimate_initial_velocity,
        shooting_consistency_rollout,
        transport_rollout,
    )

    chart = construct_path_chart(model.decode, side0["path"])
    U0 = chart["U0"]
    c0 = estimate_initial_velocity(side0["path"], U0)
    rollouts = shooting_consistency_rollout(
        model.decode,
        side0["path"][0],
        U0,
        c0,
        chart["aggregate_singular_values"][0]
        if chart["aggregate_singular_values"].numel()
        else side0["path"].new_zeros(()),
        target=z1,
    )
    primary = rollouts["primary"]
    refined = rollouts["refined"]
    primary_decoded = _decode_knots(model, primary["latents"], torch=torch)
    refined_decoded = _decode_knots(model, refined["latents"], torch=torch)
    transport = transport_rollout(
        model.decode,
        primary["latents"],
        U0,
        chart["aggregate_singular_values"][0]
        if chart["aggregate_singular_values"].numel()
        else side0["path"].new_zeros(()),
        c0,
    )
    transport_refined = transport_rollout(
        model.decode,
        refined["latents"],
        U0,
        chart["aggregate_singular_values"][0]
        if chart["aggregate_singular_values"].numel()
        else side0["path"].new_zeros(()),
        c0,
    )
    return {
        "c0": c0,
        "chart": chart,
        "primary": primary,
        "primary_decoded": primary_decoded,
        "refined": refined,
        "refined_decoded": refined_decoded,
        "transport": transport,
        "transport_refined": transport_refined,
    }


def _free_metrics(model, free, cycle, *, torch):
    primary = free["primary"]
    refined = free["refined"]
    chart = free["chart"]
    primary_quarters = primary["latents"][:: primary["steps_per_quarter"]]
    refined_quarters = refined["latents"][:: refined["steps_per_quarter"]]
    target_residual = primary["target_residual"]
    refined_target_residual = refined["target_residual"]
    common = {
        "chart_aggregate_singular_values": [
            float(value) for value in chart["aggregate_singular_values"]
        ],
        "chart_condition": float(chart["condition"]),
        "chart_inactive_rank": int(chart["inactive"].shape[0]),
        "chart_knot_metric_variation": [
            float(value) for value in chart["knot_metric_variation"]
        ],
        "chart_knot_rank_valid": [bool(value) for value in chart["knot_rank_valid"]],
        "chart_knot_singular_values": [
            [float(value) for value in row] for row in chart["knot_singular_values"]
        ],
        "chart_path_rank": chart["path_rank"],
        "chart_rank": chart["rank"],
        "chart_status": (
            "defined"
            if chart["rank"] and bool(chart["knot_rank_valid"].all())
            else "undefined"
        ),
        "initial_velocity_l2": float(torch.linalg.vector_norm(free["c0"])),
        "primary_status": primary["intrinsic_status"],
        "refined_status": refined["intrinsic_status"],
        "target_residual_z1": (
            None if target_residual is None else float(target_residual)
        ),
        "target_residual_z1_refined": (
            None
            if refined_target_residual is None
            else float(refined_target_residual)
        ),
        "transport": _transport_metrics(
            model,
            primary,
            free["transport"],
            chart["U0"],
            torch=torch,
        ),
        "transport_refined": _transport_metrics(
            model,
            refined,
            free["transport_refined"],
            chart["U0"],
            torch=torch,
        ),
    }
    result = {
        **common,
        "q_dft": None,
        "shooting_8_vs_16_decoded_rms": None,
        "shooting_8_vs_16_latent_l2": None,
        "transport_8_vs_16_return_frame_fro": None,
        "transport_8_vs_16_return_tangent_l2": None,
        "withheld_anchor_residuals": None,
    }
    if primary_quarters.shape[0] == 5:
        target_metrics = {
            f"z{turn}_latent_l2": float(
                torch.linalg.vector_norm(
                    (primary_quarters[turn] - cycle[turn % 4]).flatten()
                )
            )
            for turn in range(5)
        }
        target_metrics.update(
            {
                f"z{turn}_decoded_rms": float(
                    _rms(
                        free["primary_decoded"][
                            turn * primary["steps_per_quarter"]
                        ]
                        - model.decode(cycle[turn % 4].unsqueeze(0))[0],
                        torch=torch,
                    )
                )
                for turn in range(5)
            }
        )
        result["withheld_anchor_residuals"] = target_metrics
        result["q_dft"] = _dft_metrics(cycle, primary_quarters, torch=torch)
    if primary_quarters.shape[0] == 5 and refined_quarters.shape[0] == 5:
        result["shooting_8_vs_16_latent_l2"] = float(
            torch.linalg.vector_norm((primary_quarters - refined_quarters).flatten())
        )
        result["shooting_8_vs_16_decoded_rms"] = float(
            _rms(
                free["primary_decoded"][:: primary["steps_per_quarter"]]
                - free["refined_decoded"][:: refined["steps_per_quarter"]],
                torch=torch,
            )
        )
    if (
        free["transport"]["intrinsic_status"] == "defined"
        and free["transport_refined"]["intrinsic_status"] == "defined"
    ):
        result["transport_8_vs_16_return_tangent_l2"] = float(
            torch.linalg.vector_norm(
                free["transport"]["return_tangent"]
                - free["transport_refined"]["return_tangent"]
            )
        )
        result["transport_8_vs_16_return_frame_fro"] = float(
            torch.linalg.matrix_norm(
                free["transport"]["return_frame"]
                - free["transport_refined"]["return_frame"]
            )
        )
    return result


def _run_cycle(
    model,
    cycle,
    *,
    cycle_name,
    tensors,
    model_name,
    torch,
):
    paths = []
    path, decoded, metrics = _load_completed_path(
        model,
        name=f"{cycle_name}_side_0",
        model_name=model_name,
        device=cycle.device,
        tensors=tensors,
        torch=torch,
    )
    side0 = {"decoded": decoded, "metrics": metrics, "path": path}
    paths.append(side0)

    # This call receives only side 0 and z1. z2/z3 and all closure scores are
    # first consumed below, after both IVPs have been written.
    free = _free_rollout(model, side0, cycle[1], torch=torch)
    tensors["free"][cycle_name] = {
        "U0": free["chart"]["U0"].detach().to("cpu"),
        "inactive": free["chart"]["inactive"].detach().to("cpu"),
        "path_basis": free["chart"]["path_basis"].detach().to("cpu"),
        "path_singular_values": free["chart"]["path_singular_values"].detach().to("cpu"),
        "aggregate_eigenvalues": free["chart"]["aggregate_eigenvalues"].detach().to("cpu"),
        "aggregate_singular_values": free["chart"]["aggregate_singular_values"].detach().to("cpu"),
        "knot_singular_values": free["chart"]["knot_singular_values"].detach().to("cpu"),
        "knot_metric_variation": free["chart"]["knot_metric_variation"].detach().to("cpu"),
        "knot_rank_valid": free["chart"]["knot_rank_valid"].detach().to("cpu"),
        "numeric_floor": free["chart"]["numeric_floor"].detach().to("cpu"),
        "c0": free["c0"].detach().to("cpu"),
        "primary_latents": free["primary"]["latents"].detach().to("cpu"),
        "primary_decoded_raw_fp16": free["primary_decoded"].detach().to("cpu", torch.float16),
        "refined_latents": free["refined"]["latents"].detach().to("cpu"),
        "refined_decoded_raw_fp16": free["refined_decoded"].detach().to("cpu", torch.float16),
        "transport_coordinates": free["transport"]["coordinates"].detach().to("cpu"),
        "transport_frames": free["transport"]["frames"].detach().to("cpu"),
        "transport_maps": free["transport"]["maps"].detach().to("cpu"),
        "transport_singular_values": free["transport"]["singular_values"].detach().to("cpu"),
        "transport_refined_coordinates": free["transport_refined"]["coordinates"].detach().to("cpu"),
        "transport_refined_frames": free["transport_refined"]["frames"].detach().to("cpu"),
        "transport_refined_maps": free["transport_refined"]["maps"].detach().to("cpu"),
        "transport_refined_singular_values": free["transport_refined"]["singular_values"].detach().to("cpu"),
    }

    for side in range(1, 4):
        path, decoded, metrics = _load_completed_path(
            model,
            name=f"{cycle_name}_side_{side}",
            model_name=model_name,
            device=cycle.device,
            tensors=tensors,
            torch=torch,
        )
        paths.append({"decoded": decoded, "metrics": metrics, "path": path})

    return {
        "ambient_covariance": _decoded_covariance(paths, torch=torch),
        "decoded_cycle_closure_rms": float(
            _rms(paths[-1]["decoded"][-1] - paths[0]["decoded"][0], torch=torch)
        ),
        "free_rollout": _free_metrics(model, free, cycle, torch=torch),
        "sides": [row["metrics"] for row in paths],
    }


def _run_patch(
    model,
    *,
    rank,
    encoded,
    prescribed,
    tensors,
    model_name,
    torch,
):
    with torch.no_grad():
        decoded_encoded = model.decode(encoded)
        decoded_prescribed = model.decode(prescribed)
    tensors["anchors"][str(rank)] = {
        "encoded": encoded.detach().to("cpu"),
        "prescribed": prescribed.detach().to("cpu"),
        "decoded_encoded_raw_fp16": decoded_encoded.detach().to("cpu", torch.float16),
        "decoded_prescribed_raw_fp16": decoded_prescribed.detach().to("cpu", torch.float16),
    }
    result = {
        "anchors": _anchor_metrics(
            encoded,
            prescribed,
            decoded_encoded,
            decoded_prescribed,
            torch=torch,
        ),
        "bridges": [],
        "cycles": {},
        "rank": rank,
    }
    for turn in range(1, 4):
        path, decoded, metrics = _load_completed_path(
            model,
            name=f"rank_{rank}_bridge_{turn}",
            model_name=model_name,
            device=encoded.device,
            tensors=tensors,
            torch=torch,
        )
        result["bridges"].append(
            {"turn": turn, **_bridge_metrics(decoded, torch=torch), **metrics}
        )
    for name, cycle in (("encoded", encoded), ("prescribed", prescribed)):
        result["cycles"][name] = _run_cycle(
            model,
            cycle,
            cycle_name=f"rank_{rank}_{name}",
            tensors=tensors,
            model_name=model_name,
            torch=torch,
        )
    return result


def _worker(
    model_name,
    device_index,
    repo_root_text,
    output_text,
    source_commit,
):
    repo_root = Path(repo_root_text)
    output = Path(output_text)
    sys.path.insert(0, str(repo_root / "src"))
    import numpy as np
    import torch

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"/tmp/eqvae_spec0053_stage_a2_{model_name}_{device_index}"
    os.environ["TRITON_CACHE_DIR"] = f"/tmp/eqvae_spec0053_stage_a2_{model_name}_{device_index}"
    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    contract = json.loads((repo_root / CONTRACT_PATH).read_text(encoding="utf-8"))
    patches, selectors = _load_patches(repo_root, contract["scope"]["pilot_patch_ranks"], np=np, torch=torch)
    model = _load_frozen_model(
        model_name,
        WEIGHT_DATASET_ROOT / f"{model_name}_state.pt",
        device,
        torch=torch,
    )
    selected = patches.to(device)
    encoded = torch.stack(
        [_encode(model, torch.rot90(selected, turn, (-2, -1)), batch_size=4, torch=torch) for turn in range(4)]
    )
    prescribed = torch.stack([torch.rot90(encoded[0], turn, (-2, -1)) for turn in range(4)])
    del patches, selected
    tensors = {"anchors": {}, "free": {}, "paths": {}}
    results = []
    for local_index, rank in enumerate(contract["scope"]["pilot_patch_ranks"]):
        _log("patch_started", model=model_name, rank=rank)
        results.append(
            _run_patch(
                model,
                rank=rank,
                encoded=encoded[:, local_index],
                prescribed=prescribed[:, local_index],
                tensors=tensors,
                model_name=model_name,
                torch=torch,
            )
        )
        _log("patch_complete", model=model_name, rank=rank)
    torch.save(tensors, output / f"{model_name}_tensors.pt")
    worker = {
        "device": str(device),
        "model": model_name,
        "patches": results,
        "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "runtime": {"cuda": torch.version.cuda, "torch": torch.__version__},
        "selectors": [
            {key: row[key] for key in ("label", "rank", "sample_id", "wsi_id")}
            for row in selectors
        ],
        "source_commit": source_commit,
    }
    _write_json(output / f"{model_name}.json", worker)
    _log("worker_complete", model=model_name)


def _paired(normal, so2):
    if isinstance(normal, dict) and isinstance(so2, dict):
        if normal.keys() != so2.keys():
            return {"normal": normal, "so2": so2}
        return {
            key: _paired(normal[key], so2[key])
            for key in normal
        }
    if isinstance(normal, (float, int)) and isinstance(so2, (float, int)):
        normal_value = float(normal)
        so2_value = float(so2)
        return {
            "normal": normal_value,
            "so2": so2_value,
            "so2_minus_normal": so2_value - normal_value,
            "so2_over_normal": None if normal_value == 0 else so2_value / normal_value,
        }
    return {"normal": normal, "so2": so2}


def run(*, repo_root: Path, source_commit: str, started_at: float) -> int:
    del started_at
    output = OUTPUT_ROOT
    output.mkdir(parents=True, exist_ok=True)
    contract = json.loads((repo_root / CONTRACT_PATH).read_text(encoding="utf-8"))
    context = mp.get_context("spawn")
    _log("run_started", model_device_map=MODEL_DEVICE_MAP)
    workers = [
        context.Process(
            target=_worker,
            args=(model_name, device, str(repo_root), str(output), source_commit),
            name=f"stage-a2-{model_name}",
        )
        for model_name, device in MODEL_DEVICE_MAP.items()
    ]
    for process in workers:
        process.start()
    pending = set(workers)
    while pending:
        for process in tuple(pending):
            process.join(timeout=1)
            if process.exitcode is None:
                continue
            pending.remove(process)
            if process.exitcode != 0:
                for other in pending:
                    other.terminate()
                for other in pending:
                    other.join()
                return process.exitcode
    model_results = {
        name: json.loads((output / f"{name}.json").read_text(encoding="utf-8"))
        for name in MODEL_KINDS
    }
    result = {
        "comparison": [
            _paired(normal, so2)
            for normal, so2 in zip(
                model_results["normal_vae"]["patches"],
                model_results["so2_vae"]["patches"],
                strict=True,
            )
        ],
        "contract": contract,
        "models": model_results,
        "schema": "eqvae.functional_geometry.stage_a2.v1",
        "source_commit": source_commit,
        "status": "complete",
    }
    _write_json(output / "stage_a2_result.json", result)
    _log("run_complete", output_root=output.name)
    return 0
