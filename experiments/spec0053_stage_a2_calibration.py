# Copyright 2026 HiperMaximus
"""Numerical calibration experiment for functional-geometry Stage A2."""

import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

DATASET_ROOT = Path("/kaggle/input/datasets")
WORKING_ROOT = Path("/kaggle/working")
OUTPUT_ROOT = WORKING_ROOT / "functional_geometry_stage_a2_calibration"
CONTRACT_PATH = Path("docs/data/functional_geometry_stage_a2_calibration_contract.json")
SELECTOR_PATH = Path("configs/spec0001/fixed_25_validation_patches.json")
MODEL_KINDS = {
    "normal_vae": "non_eq_vae_translatable",
    "so2_vae": "so2_vae_fixed",
}
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


def _memory(device, torch) -> dict[str, int]:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "free_bytes": free_bytes,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "total_bytes": total_bytes,
    }


def _read_selected_patch_bytes(handle, rows, selected_ranks):
    payloads = []
    for rank in selected_ranks:
        row = rows[rank]
        handle.seek(HEADER_BYTES + int(row["file_index"]) * PATCH_BYTES)
        payloads.append(handle.read(PATCH_BYTES))
    return payloads


def _load_patches(repo_root, contract, *, np, torch):
    selector_path = repo_root / SELECTOR_PATH
    selector = json.loads(selector_path.read_text(encoding="utf-8"))
    rows = selector["selectors"]
    selected_ranks = contract["scope"]["calibration_patch_ranks"]
    selected_rows = [rows[rank] for rank in selected_ranks]
    with Path(selector["source"]["bin_path"]).open("rb") as handle:
        payloads = _read_selected_patch_bytes(handle, rows, selected_ranks)
    arrays = [
        np.frombuffer(raw, dtype=np.uint8).reshape(3, 256, 256).copy()
        for raw in payloads
    ]
    patches = (
        torch.from_numpy(np.stack(arrays)).to(torch.float32).div(255).mul(2).sub(1)
    )
    return patches, selected_rows


def _encode(model, images, *, batch_size, torch):
    means = []
    with torch.no_grad():
        for start in range(0, images.shape[0], batch_size):
            mean, _ = model.encode(images[start : start + batch_size])
            means.append(mean.detach())
    return torch.cat(means)


def _path_coordinates(*, segments, endpoint_coordinates, torch):
    times = torch.linspace(
        0.0,
        1.0,
        segments + 1,
        device=endpoint_coordinates.device,
        dtype=endpoint_coordinates.dtype,
    )
    times = times.reshape(-1, *(1 for _ in endpoint_coordinates.shape))
    return times * endpoint_coordinates.unsqueeze(0)


def _decoder_edge_energy(model, latents, total_segments, *, torch):
    decoded = model.decode(latents)
    differences = decoded[1:] - decoded[:-1]
    return total_segments * differences.flatten(1).square().mean(dim=1).sum()


def _prepare_decoder_runtime(model, probe_latents, contract, *, torch):
    """Materialize frozen SO(2) kernels and compile the repeated closure."""
    compile_contract = contract["numerics"]["compilation"]
    segment_scale = probe_latents.new_tensor(1.0)

    materialized_count = 0
    materialize = getattr(model, "materialize_frozen_decoder_kernels", None)
    if materialize is not None:
        materialized_count = materialize()

    def eager(latents, total_segments):
        return _decoder_edge_energy(
            model,
            latents,
            total_segments,
            torch=torch,
        )

    compile_started = time.perf_counter()
    compiled = torch.compile(
        eager,
        dynamic=compile_contract["dynamic"],
        fullgraph=compile_contract["fullgraph"],
        mode=compile_contract["mode"],
    )
    with torch.enable_grad():
        variable = probe_latents.detach().clone().requires_grad_(True)
        compiled(variable, segment_scale).backward()
    torch.cuda.synchronize(probe_latents.device)
    compile_seconds = time.perf_counter() - compile_started
    telemetry = {
        "compile_cold_seconds": compile_seconds,
        "materialized_decoder_kernel_count": materialized_count,
        "selected_backend": "compiled",
    }
    return eager, compiled, telemetry


def _chunked_path_energy(
    left,
    right,
    *,
    endpoint_coordinates,
    interior_coordinates,
    coordinates_to_latents,
    edge_energy,
    total_segments,
    chunk_segments,
    backward,
    torch,
):
    """Evaluate the same edge sum in bounded decoder batches.

    Returns:
        Scalar path energy.

    """
    total = interior_coordinates.new_zeros(())
    for start in range(0, total_segments, chunk_segments):
        stop = start + chunk_segments
        all_coordinates = torch.cat(
            (
                interior_coordinates.new_zeros((1, *interior_coordinates.shape[1:])),
                interior_coordinates,
                endpoint_coordinates.unsqueeze(0),
            ),
            dim=0,
        )
        latents = coordinates_to_latents(all_coordinates[start : stop + 1])
        if start == 0:
            latents = torch.cat((left, latents[1:]), dim=0)
        if stop == total_segments:
            latents = torch.cat((latents[:-1], right), dim=0)
        segment_scale = latents.new_tensor(float(total_segments))
        energy = edge_energy(
            latents,
            segment_scale,
        )
        if backward:
            energy.backward()
        else:
            detached = energy.detach()
            total = total + detached
    if backward:
        return None
    return float(total)


def _optimizer_probe(
    model,
    left,
    right,
    *,
    learning_rate,
    path_segments,
    optimizer_contract,
    eager_edge_energy,
    optimized_edge_energy,
    torch,
):
    segments = path_segments
    secant = right - left
    secant_norm = torch.linalg.vector_norm(secant).detach()
    endpoint_coordinates = secant[0] / secant_norm

    def coordinates_to_latents(coordinates):
        return left + secant_norm * coordinates

    line = _path_coordinates(
        segments=segments,
        endpoint_coordinates=endpoint_coordinates,
        torch=torch,
    )
    interior = line[1:-1].clone().detach().requires_grad_(True)
    coordinate_dimension = endpoint_coordinates.numel()
    effective_learning_rate = learning_rate * math.sqrt(
        optimizer_contract["learning_rate_reference_dimension"] / coordinate_dimension
    )
    optimizer = torch.optim.Adam([interior], lr=effective_learning_rate)
    milestones = set(optimizer_contract["milestones"])
    with torch.no_grad():
        endpoint_decoded = model.decode(torch.cat((left, right), dim=0))
        endpoint_chord_squared = float(
            (endpoint_decoded[1] - endpoint_decoded[0]).square().mean()
        )

    history = []
    maximum_deviation_seen = 0.0
    chunk_segments = optimizer_contract["energy_chunk_segments"]

    def record(iteration, preupdate_gradient_norm):
        with torch.no_grad():
            energy = (
                _chunked_path_energy(
                    left,
                    right,
                    endpoint_coordinates=endpoint_coordinates,
                    interior_coordinates=interior,
                    coordinates_to_latents=coordinates_to_latents,
                    edge_energy=eager_edge_energy,
                    total_segments=segments,
                    chunk_segments=chunk_segments,
                    backward=False,
                    torch=torch,
                )
                or 0.0
            )
            deviations = torch.linalg.vector_norm(
                (interior - line[1:-1]).flatten(1),
                dim=1,
            )
            maximum_deviation = float(deviations.max())
            history.append(
                {
                    "energy": energy,
                    "iteration": iteration,
                    "last_preupdate_gradient_norm": preupdate_gradient_norm,
                    "maximum_line_deviation_fraction": maximum_deviation,
                    "maximum_line_deviation_fraction_seen": maximum_deviation_seen,
                    "normalized_energy": energy / max(endpoint_chord_squared, 1e-12),
                }
            )

    record(0, None)
    started = time.perf_counter()
    for iteration in range(1, optimizer_contract["iterations"] + 1):
        optimizer.zero_grad(set_to_none=True)
        _chunked_path_energy(
            left,
            right,
            endpoint_coordinates=endpoint_coordinates,
            interior_coordinates=interior,
            coordinates_to_latents=coordinates_to_latents,
            edge_energy=optimized_edge_energy,
            total_segments=segments,
            chunk_segments=chunk_segments,
            backward=True,
            torch=torch,
        )
        preupdate_gradient_norm = (
            float(torch.linalg.vector_norm(interior.grad))
            if iteration in milestones
            else None
        )
        optimizer.step()
        with torch.no_grad():
            deviation = torch.linalg.vector_norm(
                (interior - line[1:-1]).flatten(1),
                dim=1,
            )
            maximum_deviation_seen = max(maximum_deviation_seen, float(deviation.max()))
        if iteration in milestones:
            record(iteration, preupdate_gradient_norm)
    if left.device.type == "cuda":
        torch.cuda.synchronize(left.device)
    elapsed = time.perf_counter() - started
    result = {
        "algorithm": optimizer_contract["algorithm"],
        "best_recorded_energy": min(row["energy"] for row in history),
        "coordinate_space": "full",
        "decoder_forward_backward_iterations": optimizer_contract["iterations"],
        "energy_chunk_segments": chunk_segments,
        "elapsed_seconds": elapsed,
        "endpoint_chord_squared_rms": endpoint_chord_squared,
        "effective_learning_rate": effective_learning_rate,
        "history": history,
        "learning_rate": learning_rate,
        "learning_rate_reference_dimension": optimizer_contract[
            "learning_rate_reference_dimension"
        ],
        "path_segments": segments,
        "maximum_line_deviation_fraction_seen": maximum_deviation_seen,
        "status": "complete",
    }
    return result


def _summarize(workers, contract):
    candidates = []
    for worker in workers:
        for workload in worker["workloads"]:
            for candidate in workload["optimizer_probes"]:
                initial = candidate["history"][0]["energy"]
                final = candidate["history"][-1]["energy"]
                best = candidate["best_recorded_energy"]
                candidates.append(
                    {
                        "best_to_initial_energy_ratio": best / max(initial, 1e-30),
                        "final_to_best_energy_ratio": final / max(best, 1e-30),
                        "maximum_line_deviation_fraction_seen": candidate[
                            "maximum_line_deviation_fraction_seen"
                        ],
                        "model": worker["model"],
                        "path_segments": candidate["path_segments"],
                        "rank": workload["rank"],
                        "route": workload["route"],
                    }
                )
    optimizer = contract["numerics"]["optimizer"]
    return {
        "candidates": candidates,
        "numerics": {
            "coordinate_space": "full",
            "optimizer_iterations": optimizer["iterations"],
            "optimizer_learning_rate": optimizer["learning_rate"],
            "path_segments": optimizer["path_segments"],
        },
        "schema": "eqvae.functional_geometry.stage_a2.calibration.summary.v4",
        "status": "complete",
    }


def _run_worker(model_name, device_index, repo_root_text, output_text, contract):
    try:
        compiler_cache = f"/tmp/eqvae_spec0053_compile_{model_name}"
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = compiler_cache
        os.environ["TRITON_CACHE_DIR"] = compiler_cache
        repo_root = Path(repo_root_text)
        output = Path(output_text)
        sys.path.insert(0, str(repo_root / "src"))
        import numpy as np
        import torch

        from eqvae.models.registry import build_model

        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        device = torch.device(f"cuda:{device_index}")
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        peak_allocated_observed = 0
        peak_reserved_observed = 0

        def capture_memory():
            nonlocal peak_allocated_observed, peak_reserved_observed
            memory = _memory(device, torch)
            peak_allocated_observed = max(
                peak_allocated_observed,
                memory["peak_allocated_bytes"],
            )
            peak_reserved_observed = max(
                peak_reserved_observed,
                memory["peak_reserved_bytes"],
            )
            return memory

        _log(
            "worker_started",
            model=model_name,
            device=str(device),
            device_name=torch.cuda.get_device_name(device_index),
        )

        patches, selectors = _load_patches(repo_root, contract, np=np, torch=torch)
        calibration_ranks = contract["scope"]["calibration_patch_ranks"]

        weight_root = next(
            DATASET_ROOT.glob(f"*/{contract['inputs']['weight_dataset_slug']}")
        )
        state_path = weight_root / f"{model_name}_state.pt"
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        model = build_model(MODEL_KINDS[model_name])
        model.load_state_dict(state, strict=True)
        model = model.to(device).eval().requires_grad_(False)
        del state
        selected = patches.to(device)
        base_mu = _encode(model, selected, batch_size=4, torch=torch)
        encoded_rotated_mu = _encode(
            model,
            torch.rot90(selected, 1, (-2, -1)),
            batch_size=4,
            torch=torch,
        )
        del patches, selected
        load_memory = capture_memory()
        _log(
            "inputs_and_model_ready",
            model=model_name,
            calibration_ranks=calibration_ranks,
            **load_memory,
        )

        numerics = contract["numerics"]
        workload_rows = []
        rank_to_index = {rank: index for index, rank in enumerate(calibration_ranks)}

        def endpoints_for(workload):
            local_index = rank_to_index[workload["rank"]]
            left = base_mu[local_index : local_index + 1]
            endpoints = {
                "encoded": encoded_rotated_mu[local_index : local_index + 1],
                "prescribed": torch.rot90(left, 1, (-2, -1)),
            }
            return local_index, left, endpoints[workload["route"]]

        first_workload = contract["scope"]["optimizer_workloads"][0]
        _, probe_left, probe_right = endpoints_for(first_workload)
        probe_times = torch.linspace(
            0.0,
            1.0,
            numerics["optimizer"]["energy_chunk_segments"] + 1,
            device=device,
        ).reshape(-1, 1, 1, 1)
        probe_latents = probe_left + probe_times * (probe_right - probe_left)
        eager_edge_energy, optimized_edge_energy, compile_telemetry = (
            _prepare_decoder_runtime(model, probe_latents, contract, torch=torch)
        )
        runtime_memory = capture_memory()
        _log(
            "decoder_runtime_ready",
            model=model_name,
            **compile_telemetry,
            **runtime_memory,
        )
        del probe_latents, probe_left, probe_right

        for workload_contract in contract["scope"]["optimizer_workloads"]:
            local_index, left, right = endpoints_for(workload_contract)
            rank = workload_contract["rank"]
            route = workload_contract["route"]
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            _log(
                "calibration_workload_started",
                model=model_name,
                rank=rank,
                route=route,
            )
            path_segments = numerics["optimizer"]["path_segments"]
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            candidate = _optimizer_probe(
                model,
                left,
                right,
                learning_rate=numerics["optimizer"]["learning_rate"],
                path_segments=path_segments,
                optimizer_contract=numerics["optimizer"],
                eager_edge_energy=eager_edge_energy,
                optimized_edge_energy=optimized_edge_energy,
                torch=torch,
            )
            candidate["peak_memory"] = capture_memory()
            _log(
                "optimizer_candidate_complete",
                model=model_name,
                rank=rank,
                route=route,
                coordinate_space="full",
                path_segments=path_segments,
                **_memory(device, torch),
            )
            torch.cuda.empty_cache()

            workload_rows.append(
                {
                    "label": selectors[local_index]["label"],
                    "optimizer_probes": [candidate],
                    "rank": rank,
                    "route": route,
                    "sample_id": selectors[local_index]["sample_id"],
                    "status": "complete",
                }
            )
            _log(
                "calibration_workload_complete",
                model=model_name,
                rank=rank,
                route=route,
                optimizer_probe_count=1,
                **_memory(device, torch),
            )
            del candidate, right
            torch.cuda.empty_cache()

        memory = capture_memory()
        memory["peak_allocated_bytes"] = peak_allocated_observed
        memory["peak_reserved_bytes"] = peak_reserved_observed
        worker = {
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device_index),
            "model": model_name,
            "peak_memory": memory,
            "decoder_runtime": compile_telemetry,
            "runtime": {"cuda": torch.version.cuda, "torch": torch.__version__},
            "workloads": workload_rows,
        }
        _write_json(output / f"{model_name}.json", worker)
        _log("worker_complete", model=model_name, **memory)
    except Exception:
        raise


def run(*, repo_root: Path, source_commit: str, started_at: float) -> int:
    processes = []
    try:
        _log("run_started", model_device_map={"normal_vae": 0, "so2_vae": 1})
        contract_path = repo_root / CONTRACT_PATH
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        deadline = started_at + contract["resources"]["wall_time_minutes_max"] * 60
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
        _log("source_ready", source_commit=source_commit)
        context = mp.get_context("spawn")
        for model_name, device_index in contract["scope"]["model_device_map"].items():
            process = context.Process(
                target=_run_worker,
                args=(
                    model_name,
                    device_index,
                    str(repo_root),
                    str(OUTPUT_ROOT),
                    contract,
                ),
                name=f"stage-a2-calibration-{model_name}",
            )
            process.start()
            processes.append(process)
            _log(
                "worker_spawned",
                model=model_name,
                device_index=device_index,
                child_pid=process.pid,
            )
        while any(process.is_alive() for process in processes):
            failed = {
                process.name: process.exitcode
                for process in processes
                if process.exitcode not in (None, 0)
            }
            if failed:
                raise RuntimeError(f"worker processes failed: {failed}")
            if time.perf_counter() >= deadline:
                raise RuntimeError("calibration wall-time ceiling exceeded")
            time.sleep(1)
        for process in processes:
            process.join()
            _log("worker_joined", process_name=process.name, exit_code=process.exitcode)
        failed = {
            process.name: process.exitcode
            for process in processes
            if process.exitcode != 0
        }
        if failed:
            raise RuntimeError(f"worker processes failed: {failed}")

        workers = {
            model: json.loads(
                (OUTPUT_ROOT / f"{model}.json").read_text(encoding="utf-8")
            )
            for model in MODEL_KINDS
        }
        summary = _summarize(tuple(workers.values()), contract)
        _write_json(OUTPUT_ROOT / "calibration_summary.json", summary)
        result = {
            "models": workers,
            "schema": "eqvae.functional_geometry.stage_a2.calibration.result.v4",
            "source_commit": source_commit,
            "status": "complete_calibration_probe",
        }
        _write_json(OUTPUT_ROOT / "run_contract.json", contract)
        runtime = {
            "elapsed_seconds": time.perf_counter() - _STARTED,
            "source_commit": source_commit,
            "worker_runtimes": {
                model: worker["runtime"] for model, worker in workers.items()
            },
        }
        _write_json(OUTPUT_ROOT / "runtime.json", runtime)
        _write_json(OUTPUT_ROOT / "calibration_result.json", result)
        _log(
            "run_complete",
            output_root=OUTPUT_ROOT.name,
            **runtime,
        )
        return 0
    except Exception:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=5)
        for process in processes:
            if process.is_alive():
                process.kill()
        for process in processes:
            process.join(timeout=5)
        raise
