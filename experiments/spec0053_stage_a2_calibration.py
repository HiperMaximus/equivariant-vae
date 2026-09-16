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
OUTPUT_ROOT = WORKING_ROOT / "functional_geometry_stage_a2_calibration_v4"
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
        if row.get("rank") != rank:
            raise RuntimeError("fixed25 ranks differ")
        handle.seek(HEADER_BYTES + int(row["file_index"]) * PATCH_BYTES)
        raw = handle.read(PATCH_BYTES)
        if len(raw) != PATCH_BYTES:
            raise RuntimeError(f"fixed25 patch {rank} is truncated")
        payloads.append(raw)
    return payloads


def _load_patches(repo_root, contract, *, np, torch):
    selector_path = repo_root / SELECTOR_PATH
    selector = json.loads(selector_path.read_text(encoding="utf-8"))
    rows = selector.get("selectors")
    if not isinstance(rows, list) or len(rows) != 25:
        raise RuntimeError("fixed25 inputs differ")
    selected_ranks = contract["scope"]["calibration_patch_ranks"]
    final_ranks = set(contract["scope"]["final_pilot_patch_ranks"])
    forbidden_wsis = set(contract["scope"]["forbidden_calibration_wsis"])
    if final_ranks.intersection(selected_ranks):
        raise RuntimeError("final pilot leaked into numerical calibration")
    selected_rows = []
    for rank in selected_ranks:
        row = rows[rank]
        sample_id_parts = row["sample_id"].split(":")
        if len(sample_id_parts) < 3 or sample_id_parts[2] in forbidden_wsis:
            raise RuntimeError("forbidden WSI leaked into numerical calibration")
        selected_rows.append(row)
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


def _spectrum_payload(spectrum) -> dict[str, object]:
    return {
        "condition_number": spectrum.condition_number,
        "dimension": spectrum.dimension,
        "minimum_to_maximum_ratio": spectrum.minimum_to_maximum_ratio,
        "singular_values_descending": list(spectrum.singular_values_descending),
    }


def _chart_diagnostics(
    model,
    left,
    right,
    *,
    rank,
    route,
    numerics,
    linearize_decoder,
    decoder_visible_chart,
    thin_metric_spectra,
    torch,
):
    chart_contract = numerics["chart"]
    microbatch = numerics["jvp_microbatch"]
    if chart_contract["line_parameters"] != [0.0, 0.5, 1.0]:
        raise RuntimeError("calibration line parameters differ")
    if chart_contract["random_probe_count"] != chart_contract["maximum_dimension"] - 1:
        raise RuntimeError("chart probe count differs from maximum dimension")
    secant = right - left
    secant_norm = torch.linalg.vector_norm(secant.to(torch.float64)).to(torch.float32)
    midpoint = (left + right) / 2
    midpoint_operator = linearize_decoder(model.decode, midpoint)
    route_offset = 0 if route == "encoded" else 1
    basis = decoder_visible_chart(
        midpoint_operator,
        secant,
        maximum_dimension=chart_contract["maximum_dimension"],
        seed=chart_contract["seed"] + 10 * rank + route_offset,
        microbatch=microbatch,
    )
    flat_basis = basis.flatten(1)
    orthonormality_error = float(
        torch.linalg.matrix_norm(
            (flat_basis @ flat_basis.T).to(torch.float64)
            - torch.eye(basis.shape[0], device=basis.device, dtype=torch.float64),
            ord=2,
        )
    )
    secant_projection = torch.einsum("dn,bn->d", flat_basis, secant.flatten(1))
    secant_residual = secant.flatten(1) - secant_projection @ flat_basis
    secant_relative_residual = float(
        torch.linalg.vector_norm(secant_residual.to(torch.float64))
        / secant_norm.to(torch.float64).clamp_min(1e-12)
    )

    metric_rows = []
    for parameter_index, parameter in enumerate(chart_contract["line_parameters"]):
        point = left + float(parameter) * secant
        operator = (
            midpoint_operator
            if parameter_index == 1
            else linearize_decoder(model.decode, point)
        )
        spectra = thin_metric_spectra(
            operator,
            basis,
            dimensions=tuple(chart_contract["dimensions"]),
            microbatch=microbatch,
        )
        metric_rows.append({
            "line_parameter": parameter,
            "spectra": [_spectrum_payload(spectrum) for spectrum in spectra],
        })

    result = {
        "basis": {
            "construction": "secant_plus_midpoint_G_rademacher_then_qr",
            "orthonormality_operator_error": orthonormality_error,
            "secant_relative_projection_residual": secant_relative_residual,
        },
        "metrics": metric_rows,
        "secant_euclidean_norm": float(secant_norm),
    }
    json.dumps(result, allow_nan=False)
    return basis, result


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


def _value_and_gradient(function, latents, total_segments, *, torch):
    variable = latents.detach().clone().requires_grad_(True)
    value = function(variable, total_segments)
    gradient = torch.autograd.grad(value, variable)[0]
    return value.detach(), gradient.detach()


def _relative_tensor_error(left, right, *, torch):
    numerator = torch.linalg.vector_norm((left - right).to(torch.float64))
    denominator = torch.maximum(
        torch.linalg.vector_norm(left.to(torch.float64)),
        torch.linalg.vector_norm(right.to(torch.float64)),
    ).clamp_min(1e-30)
    return float(numerator / denominator)


def _prepare_decoder_runtime(model, probe_latents, contract, *, torch):
    """Materialize frozen SO(2) kernels and compile the repeated scalar closure."""
    compile_contract = contract["numerics"]["compilation"]
    segment_scale = torch.tensor(
        float(compile_contract["probe_path_segments"]),
        device=probe_latents.device,
        dtype=probe_latents.dtype,
    )

    with torch.enable_grad():
        uncached_output = model.decode(probe_latents).detach()
        uncached_value, uncached_gradient = _value_and_gradient(
            lambda latents, scale: _decoder_edge_energy(
                model,
                latents,
                scale,
                torch=torch,
            ),
            probe_latents,
            segment_scale,
            torch=torch,
        )

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

    with torch.enable_grad():
        cached_output = model.decode(probe_latents).detach()
        cached_value, cached_gradient = _value_and_gradient(
            eager,
            probe_latents,
            segment_scale,
            torch=torch,
        )

    cache_output_error = _relative_tensor_error(
        uncached_output,
        cached_output,
        torch=torch,
    )
    cache_energy_error = _relative_tensor_error(
        uncached_value,
        cached_value,
        torch=torch,
    )
    cache_gradient_error = _relative_tensor_error(
        uncached_gradient,
        cached_gradient,
        torch=torch,
    )
    if max(cache_output_error, cache_energy_error, cache_gradient_error) != 0.0:
        raise RuntimeError("materialized decoder differs from coefficient expansion")

    compile_started = time.perf_counter()
    compiled = torch.compile(
        eager,
        dynamic=compile_contract["dynamic"],
        fullgraph=compile_contract["fullgraph"],
        mode=compile_contract["mode"],
    )
    with torch.enable_grad():
        compiled_value, compiled_gradient = _value_and_gradient(
            compiled,
            probe_latents,
            segment_scale,
            torch=torch,
        )
    torch.cuda.synchronize(probe_latents.device)
    compile_seconds = time.perf_counter() - compile_started

    compiled_energy_error = _relative_tensor_error(
        cached_value,
        compiled_value,
        torch=torch,
    )
    compiled_gradient_error = _relative_tensor_error(
        cached_gradient,
        compiled_gradient,
        torch=torch,
    )
    if compiled_energy_error > compile_contract["energy_relative_l2_max"]:
        raise RuntimeError("compiled edge energy differs from eager")
    if compiled_gradient_error > compile_contract["gradient_relative_l2_max"]:
        raise RuntimeError("compiled edge gradient differs from eager")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("frozen decoder accumulated parameter gradients")

    def settled_seconds(function):
        samples = []
        for _ in range(compile_contract["settled_repetitions"]):
            torch.cuda.synchronize(probe_latents.device)
            started = time.perf_counter()
            with torch.enable_grad():
                _value_and_gradient(
                    function,
                    probe_latents,
                    segment_scale,
                    torch=torch,
                )
            torch.cuda.synchronize(probe_latents.device)
            samples.append(time.perf_counter() - started)
        return sorted(samples)[len(samples) // 2]

    eager_seconds = settled_seconds(eager)
    compiled_seconds = settled_seconds(compiled)
    selected = compiled if compiled_seconds < eager_seconds else eager
    telemetry = {
        "cache_energy_relative_l2": cache_energy_error,
        "cache_gradient_relative_l2": cache_gradient_error,
        "cache_output_relative_l2": cache_output_error,
        "compile_cold_seconds": compile_seconds,
        "compiled_energy_relative_l2": compiled_energy_error,
        "compiled_gradient_relative_l2": compiled_gradient_error,
        "compiled_settled_seconds": compiled_seconds,
        "eager_settled_seconds": eager_seconds,
        "materialized_decoder_kernel_count": materialized_count,
        "selected_backend": "compiled" if selected is compiled else "eager",
        "settled_speedup": eager_seconds / compiled_seconds,
    }
    return eager, selected, telemetry


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
    if total_segments % chunk_segments != 0:
        raise ValueError("compiled path blocks must have one fixed shape")
    total = None
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
            total = detached if total is None else total + detached
    if backward:
        return None
    if total is None:
        raise RuntimeError("path contains no edge blocks")
    return float(total)


def _optimizer_probe(
    model,
    left,
    right,
    basis,
    *,
    coordinate_space,
    learning_rate,
    path_segments,
    optimizer_contract,
    affine_chart_latents,
    eager_edge_energy,
    optimized_edge_energy,
    project_line_deviation_,
    torch,
):
    segments = path_segments
    secant = right - left
    secant_norm = torch.linalg.vector_norm(secant).detach()
    if coordinate_space == "full":
        endpoint_coordinates = secant[0] / secant_norm
        projection_error = 0.0

        def coordinates_to_latents(coordinates):
            return left + secant_norm * coordinates

    else:
        dimension = int(coordinate_space)
        chart = basis[:dimension]
        endpoint_coordinates = (
            torch.einsum("dn,bn->d", chart.flatten(1), secant.flatten(1))
            / secant_norm
        )
        projection = torch.einsum("d,dn->n", endpoint_coordinates, chart.flatten(1))
        projection_error = float(
            torch.linalg.vector_norm(projection - secant.flatten(1) / secant_norm)
        )
        if projection_error > 1e-4:
            raise RuntimeError("literal endpoint is outside the affine chart")

        def coordinates_to_latents(coordinates):
            return affine_chart_latents(
                left,
                chart,
                coordinates,
                secant_norm=secant_norm,
            )

    line = _path_coordinates(
        segments=segments,
        endpoint_coordinates=endpoint_coordinates,
        torch=torch,
    )
    interior = line[1:-1].clone().detach().requires_grad_(True)
    coordinate_dimension = endpoint_coordinates.numel()
    effective_learning_rate = learning_rate * math.sqrt(
        optimizer_contract["learning_rate_reference_dimension"]
        / coordinate_dimension
    )
    optimizer = torch.optim.Adam([interior], lr=effective_learning_rate)
    milestones = set(optimizer_contract["milestones"])
    with torch.no_grad():
        endpoint_decoded = model.decode(torch.cat((left, right), dim=0))
        endpoint_chord_squared = float(
            (endpoint_decoded[1] - endpoint_decoded[0]).square().mean()
        )

    history = []
    maximum_allowed = optimizer_contract["trust_deviation_fraction_of_endpoint_secant"]
    maximum_postprojection_deviation = 0.0
    maximum_preprojection_deviation = 0.0
    projection_applied_count = 0
    projection_iteration_count = 0
    chunk_segments = optimizer_contract["energy_chunk_segments"]

    def record(iteration, preupdate_gradient_norm):
        with torch.no_grad():
            energy = _chunked_path_energy(
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
            if energy is None:
                raise RuntimeError("eager energy evaluation returned no value")
            deviations = torch.linalg.vector_norm(
                (interior - line[1:-1]).flatten(1),
                dim=1,
            )
            maximum_deviation = float(deviations.max())
            history.append({
                "energy": energy,
                "iteration": iteration,
                "last_preupdate_gradient_norm": preupdate_gradient_norm,
                "maximum_line_deviation_fraction": maximum_deviation,
                "maximum_postprojection_deviation_seen": (
                    maximum_postprojection_deviation
                ),
                "maximum_preprojection_deviation_seen": (
                    maximum_preprojection_deviation
                ),
                "normalized_energy": energy / max(endpoint_chord_squared, 1e-12),
                "projection_applied_count": projection_applied_count,
                "projection_iteration_count": projection_iteration_count,
            })

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
            preprojection = torch.linalg.vector_norm(
                (interior - line[1:-1]).flatten(1),
                dim=1,
            )
            preprojection_maximum = float(preprojection.max())
            maximum_preprojection_deviation = max(
                maximum_preprojection_deviation,
                preprojection_maximum,
            )
            projected_rows = int((preprojection > maximum_allowed).sum())
            projection_applied_count += projected_rows
            projection_iteration_count += int(projected_rows > 0)
        project_line_deviation_(
            interior,
            line[1:-1],
            maximum_deviation=maximum_allowed,
        )
        with torch.no_grad():
            postprojection_maximum = float(
                torch.linalg.vector_norm(
                    (interior - line[1:-1]).flatten(1),
                    dim=1,
                ).max()
            )
            maximum_postprojection_deviation = max(
                maximum_postprojection_deviation,
                postprojection_maximum,
            )
        if iteration in milestones:
            record(iteration, preupdate_gradient_norm)
    if left.device.type == "cuda":
        torch.cuda.synchronize(left.device)
    elapsed = time.perf_counter() - started
    result = {
        "algorithm": optimizer_contract["algorithm"],
        "best_recorded_energy": min(row["energy"] for row in history),
        "coordinate_space": coordinate_space,
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
        "projection_applied_count": projection_applied_count,
        "projection_iteration_count": projection_iteration_count,
        "projection_error": projection_error,
        "status": "complete",
        "trust_boundary_contact": projection_iteration_count > 0,
    }
    json.dumps(result, allow_nan=False)
    return result


def _select_numerics(workers, contract):
    numerics = contract["numerics"]
    workload_contracts = contract["scope"]["optimizer_workloads"]
    workloads = [
        workload
        for worker in workers
        for workload in worker["workloads"]
        if workload.get("status") == "complete"
    ]
    expected_workloads = len(workers) * len(workload_contracts)
    blockers = []
    if len(workloads) != expected_workloads:
        blockers.append("one_or_more_calibration_workloads_failed")

    ratio_floor = numerics["selection"]["conditioning_minimum_ratio"]
    dimension_minimum_ratios = {}
    for dimension in numerics["chart"]["dimensions"]:
        values = [
            spectrum["minimum_to_maximum_ratio"]
            for workload in workloads
            for metric in workload["chart_diagnostics"]["metrics"]
            for spectrum in metric["spectra"]
            if spectrum["dimension"] == dimension
        ]
        expected = len(workloads) * len(numerics["chart"]["line_parameters"])
        if expected > 0 and len(values) == expected:
            dimension_minimum_ratios[str(dimension)] = min(values)

    expected_candidates = sum(
        len(candidate["coordinate_spaces"])
        for workload in workload_contracts
        for candidate in workload["candidates"]
    ) * len(workers)
    candidates = [
        candidate
        for workload in workloads
        for candidate in workload["optimizer_probes"]
        if candidate.get("status") == "complete"
    ]
    if len(candidates) != expected_candidates:
        blockers.append("one_or_more_optimizer_candidates_failed")

    minimum_energy_decrease = numerics["selection"][
        "optimizer_minimum_relative_energy_decrease"
    ]
    plateau_slack = numerics["selection"]["optimizer_energy_plateau_relative_slack"]
    gap_limit = numerics["selection"]["reduced_to_full_best_energy_excess_max"]
    candidate_rows = []
    for worker in workers:
        for workload in worker["workloads"]:
            complete = {
                (candidate["path_segments"], str(candidate["coordinate_space"])): candidate
                for candidate in workload.get("optimizer_probes", [])
                if candidate.get("status") == "complete"
            }
            for path_segments in {key[0] for key in complete}:
                full = complete.get((path_segments, "full"))
                if full is None:
                    continue
                full_best = full["best_recorded_energy"]
                for (segments, coordinate_space), candidate in complete.items():
                    if segments != path_segments:
                        continue
                    initial = candidate["history"][0]["energy"]
                    final = candidate["history"][-1]["energy"]
                    best = candidate["best_recorded_energy"]
                    candidate_rows.append({
                        "best_to_initial_energy_ratio": best / max(initial, 1e-30),
                        "best_to_full_energy_excess": max(
                            0.0,
                            best / max(full_best, 1e-30) - 1.0,
                        ),
                        "coordinate_space": coordinate_space,
                        "final_to_best_energy_ratio": final / max(best, 1e-30),
                        "model": worker["model"],
                        "path_segments": path_segments,
                        "rank": workload["rank"],
                        "route": workload["route"],
                        "trust_boundary_contact": candidate["trust_boundary_contact"],
                    })

    def optimization_passes(row):
        return (
            row["best_to_initial_energy_ratio"] <= 1.0 - minimum_energy_decrease
            and row["final_to_best_energy_ratio"] <= 1.0 + plateau_slack
            and not row["trust_boundary_contact"]
        )

    def row_key(row):
        return (
            row["model"],
            row["rank"],
            row["route"],
            row["path_segments"],
        )

    full_rows = {
        row_key(row): row
        for row in candidate_rows
        if row["coordinate_space"] == "full"
    }

    def space_passes(space):
        rows = [
            row
            for row in candidate_rows
            if str(row["coordinate_space"]) == str(space)
        ]
        expected = sum(
            str(space) in {str(item) for item in candidate["coordinate_spaces"]}
            for workload in workload_contracts
            for candidate in workload["candidates"]
        ) * len(workers)
        return len(rows) == expected and all(
            optimization_passes(row)
            and (
                space == "full"
                or (
                    row["best_to_full_energy_excess"] <= gap_limit
                    and row_key(row) in full_rows
                    and optimization_passes(full_rows[row_key(row)])
                )
            )
            for row in rows
        )

    d128_conditioned = dimension_minimum_ratios.get("128", -1.0) >= ratio_floor
    if d128_conditioned and space_passes(128):
        selected_space = 128
    elif space_passes("full"):
        selected_space = "full"
    else:
        selected_space = None
        blockers.append("neither_d128_nor_full_latent_met_the_common_budget")

    selected_iterations = None
    if selected_space is not None:
        selected_candidates = [
            candidate
            for workload in workloads
            for candidate in workload["optimizer_probes"]
            if candidate.get("status") == "complete"
            and str(candidate["coordinate_space"]) == str(selected_space)
        ]
        for milestone in numerics["optimizer"]["milestones"][1:]:
            if all(
                next(
                    row["energy"]
                    for row in candidate["history"]
                    if row["iteration"] == milestone
                )
                <= (1.0 + plateau_slack) * candidate["best_recorded_energy"]
                for candidate in selected_candidates
            ):
                selected_iterations = milestone
                break
        if selected_iterations is None:
            blockers.append("selected_space_not_near_best_within_128_steps")

    selected = {
        "coordinate_space": selected_space,
        "conditioning_minimum_ratio": ratio_floor,
        "optimizer_iterations": selected_iterations,
        "optimizer_learning_rate": numerics["optimizer"]["learning_rate"],
        "optimizer_minimum_relative_energy_decrease": minimum_energy_decrease,
        "path_segments_primary": 16,
        "path_segments_refinement": 32,
    }
    return {
        "blockers": blockers,
        "decision_inputs": {
            "candidate_comparisons": candidate_rows,
            "d128_conditioned": d128_conditioned,
            "d32_passes_full_latent_control": space_passes(32),
            "d128_passes_full_latent_control": space_passes(128),
            "dimension_minimum_ratios": dimension_minimum_ratios,
            "full_latent_passes_common_budget": space_passes("full"),
        },
        "schema": "eqvae.functional_geometry.stage_a2.calibration.selection.v3",
        "selected_numerics": selected,
        "status": "selected" if not blockers else "unresolved",
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

        from eqvae.evaluation.functional_geometry_calibration import (
            affine_chart_latents,
            decoder_visible_chart,
            project_line_deviation_,
            thin_metric_spectra,
        )
        from eqvae.evaluation.functional_geometry_rla import linearize_decoder
        from eqvae.models.registry import build_model

        if not torch.cuda.is_available() or torch.cuda.device_count() <= device_index:
            raise RuntimeError(f"CUDA device {device_index} is unavailable")
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

        peak_ceiling = contract["resources"]["peak_reserved_bytes_max_per_device"]

        _log(
            "worker_started",
            model=model_name,
            device=str(device),
            device_name=torch.cuda.get_device_name(device_index),
        )

        patches, selectors = _load_patches(repo_root, contract, np=np, torch=torch)
        calibration_ranks = contract["scope"]["calibration_patch_ranks"]
        if [row["rank"] for row in selectors] != calibration_ranks:
            raise RuntimeError("loaded calibration ranks differ")

        weight_root = DATASET_ROOT / contract["inputs"]["weight_dataset"]
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
        if load_memory["peak_reserved_bytes"] >= peak_ceiling:
            raise RuntimeError("input/model load exceeded peak-memory ceiling")
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
        if runtime_memory["peak_reserved_bytes"] >= peak_ceiling:
            raise RuntimeError("decoder compilation exceeded peak-memory ceiling")
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
            basis = None
            diagnostics = None
            try:
                basis, diagnostics = _chart_diagnostics(
                    model,
                    left,
                    right,
                    rank=rank,
                    route=route,
                    numerics=numerics,
                    linearize_decoder=linearize_decoder,
                    decoder_visible_chart=decoder_visible_chart,
                    thin_metric_spectra=thin_metric_spectra,
                    torch=torch,
                )
                if capture_memory()["peak_reserved_bytes"] >= peak_ceiling:
                    raise RuntimeError("chart exceeded peak-memory ceiling")
            except Exception as error:
                failure_memory = capture_memory()
                workload_rows.append({
                    "exception_message": str(error),
                    "exception_type": type(error).__name__,
                    "label": selectors[local_index]["label"],
                    "peak_memory": failure_memory,
                    "rank": rank,
                    "route": route,
                    "sample_id": selectors[local_index]["sample_id"],
                    "status": "failed",
                })
                _log(
                    "calibration_workload_failed",
                    model=model_name,
                    rank=rank,
                    route=route,
                    exception_type=type(error).__name__,
                    exception_message=str(error),
                    **failure_memory,
                )
                del basis, diagnostics
                torch.cuda.empty_cache()
                continue

            optimizer_rows = []
            for candidate_contract in workload_contract["candidates"]:
                path_segments = candidate_contract["path_segments"]
                for coordinate_space in candidate_contract["coordinate_spaces"]:
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(device)
                    try:
                        candidate = _optimizer_probe(
                            model,
                            left,
                            right,
                            basis,
                            coordinate_space=coordinate_space,
                            learning_rate=numerics["optimizer"]["learning_rate"],
                            path_segments=path_segments,
                            optimizer_contract=numerics["optimizer"],
                            affine_chart_latents=affine_chart_latents,
                            eager_edge_energy=eager_edge_energy,
                            optimized_edge_energy=optimized_edge_energy,
                            project_line_deviation_=project_line_deviation_,
                            torch=torch,
                        )
                    except Exception as error:
                        candidate_memory = capture_memory()
                        candidate = {
                            "coordinate_space": coordinate_space,
                            "exception_message": str(error),
                            "exception_type": type(error).__name__,
                            "learning_rate": numerics["optimizer"]["learning_rate"],
                            "path_segments": path_segments,
                            "peak_memory": candidate_memory,
                            "status": "failed",
                        }
                        event = "optimizer_candidate_failed"
                    else:
                        candidate_memory = capture_memory()
                        candidate["peak_memory"] = candidate_memory
                        if candidate_memory["peak_reserved_bytes"] >= peak_ceiling:
                            candidate["status"] = "resource_ceiling_exceeded"
                            event = "optimizer_candidate_resource_exceeded"
                        else:
                            event = "optimizer_candidate_complete"
                    optimizer_rows.append(candidate)
                    _log(
                        event,
                        model=model_name,
                        rank=rank,
                        route=route,
                        coordinate_space=coordinate_space,
                        path_segments=path_segments,
                        **_memory(device, torch),
                    )
                    torch.cuda.empty_cache()

            workload_rows.append({
                "chart_diagnostics": diagnostics,
                "label": selectors[local_index]["label"],
                "optimizer_probes": optimizer_rows,
                "rank": rank,
                "route": route,
                "sample_id": selectors[local_index]["sample_id"],
                "status": "complete",
            })
            _log(
                "calibration_workload_complete",
                model=model_name,
                rank=rank,
                route=route,
                optimizer_probe_count=len(optimizer_rows),
                **_memory(device, torch),
            )
            del basis, diagnostics, optimizer_rows, right
            torch.cuda.empty_cache()

        memory = capture_memory()
        memory["peak_allocated_bytes"] = peak_allocated_observed
        memory["peak_reserved_bytes"] = peak_reserved_observed
        worker = {
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device_index),
            "model": model_name,
            "peak_memory": memory,
            "peak_memory_ceiling_exceeded_by_any_candidate": (
                peak_reserved_observed >= peak_ceiling
            ),
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
        for process in processes:
            process.join(timeout=max(0.0, deadline - time.perf_counter()))
            if process.is_alive():
                raise RuntimeError("calibration wall-time ceiling exceeded")
            _log("worker_joined", process_name=process.name, exit_code=process.exitcode)
        failed = {
            process.name: process.exitcode
            for process in processes
            if process.exitcode != 0
        }
        if failed:
            raise RuntimeError(f"worker processes failed: {failed}")
        if time.perf_counter() >= deadline:
            raise RuntimeError("calibration wall-time ceiling exceeded")

        workers = {
            model: json.loads(
                (OUTPUT_ROOT / f"{model}.json").read_text(encoding="utf-8")
            )
            for model in MODEL_KINDS
        }
        selection = _select_numerics(tuple(workers.values()), contract)
        if time.perf_counter() >= deadline:
            raise RuntimeError("calibration wall-time ceiling exceeded")
        selection_path = OUTPUT_ROOT / "calibration_selection.json"
        _write_json(selection_path, selection)
        result = {
            "models": workers,
            "schema": "eqvae.functional_geometry.stage_a2.calibration.result.v3",
            "selection_status": selection["status"],
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
        if time.perf_counter() >= deadline:
            raise RuntimeError("calibration wall-time ceiling exceeded")
        result_path = OUTPUT_ROOT / "calibration_result.json"
        _write_json(result_path, result)
        if time.perf_counter() >= deadline:
            result_path.unlink()
            raise RuntimeError("calibration wall-time ceiling exceeded")
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
