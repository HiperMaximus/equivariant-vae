# Copyright 2026 HiperMaximus
# ruff: noqa: ANN001, ANN202, BLE001, C901, COM812, D103, EM101, EM102, FBT003, PLC0415, PLR0912, PLR0913, PLR0914, PLR0915, PLR1702, PLR2004, PLW0717, T201, TRY003, TRY203, TRY300, TRY301
"""Numerical calibration experiment for functional-geometry Stage A2."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

INPUT_ROOT = Path("/kaggle/input")
WORKING_ROOT = Path("/kaggle/working")
OUTPUT_ROOT = WORKING_ROOT / "functional_geometry_stage_a2_calibration_v2"
CONTRACT_PATH = Path("docs/data/functional_geometry_stage_a2_calibration_contract.json")
SELECTOR_PATH = Path("runs/kaggle/fixed25_selector/fixed_25_validation_patches.json")
WEIGHT_ROOT = INPUT_ROOT / "eqvae-vae-test-reconstruction-inputs-v1"
PATCH_PATH = (
    INPUT_ROOT / "patches-pre-shuffled-ubc-ocean" / "dataset" / "ubc_ocean_valid.bin"
)
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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
        if (
            len(raw) != PATCH_BYTES
            or hashlib.sha256(raw).hexdigest() != row["patch_sha256"]
        ):
            raise RuntimeError(f"fixed25 patch {rank} differs")
        payloads.append(raw)
    return payloads


def _load_patches(repo_root, contract, *, np, torch):
    selector_path = repo_root / SELECTOR_PATH
    if _sha256(selector_path) != contract["inputs"]["fixed_selector_sha256"]:
        raise RuntimeError("fixed25 selector differs")
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
    with PATCH_PATH.open("rb") as handle:
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


def _relative_l2(left, right, *, torch) -> float:
    numerator = torch.linalg.vector_norm((left - right).to(torch.float64))
    denominator = torch.maximum(
        torch.linalg.vector_norm(left.to(torch.float64)),
        torch.linalg.vector_norm(right.to(torch.float64)),
    ).clamp_min(1e-12)
    return float(numerator / denominator)


def _spectrum_payload(spectrum) -> dict[str, object]:
    return {
        "condition_number": spectrum.condition_number,
        "dimension": spectrum.dimension,
        "minimum_to_maximum_ratio": spectrum.minimum_to_maximum_ratio,
        "singular_values_descending": list(spectrum.singular_values_descending),
    }


def _linearity_directions(basis, dimension, *, seed, combination_count, torch):
    chart = basis[:dimension]
    generator = torch.Generator(device=basis.device).manual_seed(seed)
    signs = torch.randint(
        0,
        2,
        (combination_count, dimension),
        generator=generator,
        device=basis.device,
        dtype=torch.int8,
    )
    coefficients = (signs.to(basis.dtype) * 2 - 1) / math.sqrt(dimension)
    combinations = coefficients @ chart.flatten(1)
    return torch.cat((
        chart[:1],
        combinations.reshape(combination_count, *chart.shape[1:]),
    ))


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
    midpoint_operator = linearize_decoder(
        model.decode,
        midpoint,
        compile_operators=False,
    )
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
    line_operators = []
    for parameter_index, parameter in enumerate(chart_contract["line_parameters"]):
        point = left + float(parameter) * secant
        operator = (
            midpoint_operator
            if parameter_index == 1
            else linearize_decoder(model.decode, point, compile_operators=False)
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
        line_operators.append((float(parameter), point, operator))

    linearity_rows = []
    linearity = numerics["linearity"]
    for dimension in chart_contract["dimensions"]:
        directions = _linearity_directions(
            basis,
            dimension,
            seed=linearity["seed"] + 100 * rank + 10 * route_offset + dimension,
            combination_count=linearity["rademacher_combination_count"],
            torch=torch,
        )
        for parameter, point, operator in line_operators:
            tangents = torch.cat([
                operator.jvp_batch(directions[start : start + microbatch])
                for start in range(0, directions.shape[0], microbatch)
            ])
            for radius_fraction in linearity["radius_fractions_of_endpoint_secant"]:
                for sign in (-1, 1):
                    scale = sign * float(radius_fraction) * secant_norm
                    with torch.no_grad():
                        output1 = model.decode(point + scale * directions)
                    linear = scale * tangents
                    linearity_rows.extend(
                        {
                            "chart_dimension": dimension,
                            "direction_index": direction_index,
                            "direction_kind": (
                                "secant" if direction_index == 0 else "rademacher"
                            ),
                            "line_parameter": parameter,
                            "radius_fraction": radius_fraction,
                            "relative_l2": _relative_l2(
                                output1[direction_index] - operator.output[0],
                                linear[direction_index],
                                torch=torch,
                            ),
                            "sign": sign,
                        }
                        for direction_index in range(directions.shape[0])
                    )

    fd_contract = numerics["finite_difference"]
    generator = torch.Generator(device=left.device).manual_seed(
        fd_contract["seed"] + 10 * rank + route_offset
    )
    fd_signs = torch.randint(
        0,
        2,
        left.shape,
        generator=generator,
        device=left.device,
        dtype=torch.int8,
    )
    fd_direction = fd_signs.to(torch.float32) * 2 - 1
    fd_tangent = midpoint_operator.jvp_batch(fd_direction)
    fd_rows = []
    for epsilon in fd_contract["epsilon_candidates"]:
        with torch.no_grad():
            finite = (
                model.decode(midpoint + float(epsilon) * fd_direction)
                - model.decode(midpoint - float(epsilon) * fd_direction)
            ) / (2 * float(epsilon))
        fd_rows.append({
            "epsilon": epsilon,
            "relative_l2": _relative_l2(fd_tangent, finite, torch=torch),
        })

    result = {
        "basis": {
            "construction": "secant_plus_midpoint_G_rademacher_then_qr",
            "orthonormality_operator_error": orthonormality_error,
            "secant_relative_projection_residual": secant_relative_residual,
        },
        "finite_difference": fd_rows,
        "linearity": linearity_rows,
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
    return times[:, None] * endpoint_coordinates[None, :]


def _chunked_path_energy(
    model,
    left,
    right,
    *,
    basis,
    interior_coordinates,
    secant_norm,
    total_segments,
    chunk_segments,
    affine_chart_latents,
    decoder_path_edge_energy,
    backward,
    torch,
):
    """Evaluate the same edge sum in bounded decoder batches.

    Returns:
        Scalar path energy.

    """
    endpoint = (right - left).flatten(1) @ basis.flatten(1).T / secant_norm
    total = 0.0
    for start in range(0, total_segments, chunk_segments):
        stop = min(start + chunk_segments, total_segments)
        all_coordinates = torch.cat(
            (
                interior_coordinates.new_zeros((1, interior_coordinates.shape[1])),
                interior_coordinates,
                endpoint,
            ),
            dim=0,
        )
        latents = affine_chart_latents(
            left,
            basis,
            all_coordinates[start : stop + 1],
            secant_norm=secant_norm,
        )
        if start == 0:
            latents = torch.cat((left, latents[1:]), dim=0)
        if stop == total_segments:
            latents = torch.cat((latents[:-1], right), dim=0)
        energy = decoder_path_edge_energy(
            model.decode(latents),
            total_path_segments=total_segments,
        )
        if backward:
            energy.backward()
        total += float(energy.detach())
    return total


def _optimizer_probe(
    model,
    left,
    right,
    basis,
    *,
    dimension,
    learning_rate,
    path_segments,
    optimizer_contract,
    affine_chart_latents,
    decoder_path_edge_energy,
    project_line_deviation_,
    torch,
):
    segments = path_segments
    chart = basis[:dimension]
    secant = right - left
    secant_norm = torch.linalg.vector_norm(secant).detach()
    endpoint_coordinates = (
        torch.einsum("dn,bn->d", chart.flatten(1), secant.flatten(1)) / secant_norm
    )
    projection = torch.einsum("d,dn->n", endpoint_coordinates, chart.flatten(1))
    projection_error = float(
        torch.linalg.vector_norm(projection - secant.flatten(1) / secant_norm)
    )
    if projection_error > 1e-4:
        raise RuntimeError("literal endpoint is outside the affine chart")
    line = _path_coordinates(
        segments=segments,
        endpoint_coordinates=endpoint_coordinates,
        torch=torch,
    )
    interior = line[1:-1].clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([interior], lr=learning_rate)
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
                model,
                left,
                right,
                basis=chart,
                interior_coordinates=interior,
                secant_norm=secant_norm,
                total_segments=segments,
                chunk_segments=chunk_segments,
                affine_chart_latents=affine_chart_latents,
                decoder_path_edge_energy=decoder_path_edge_energy,
                backward=False,
                torch=torch,
            )
            deviations = torch.linalg.vector_norm(interior - line[1:-1], dim=1)
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
    last_gradient_norm = None
    started = time.perf_counter()
    for iteration in range(1, optimizer_contract["iterations"] + 1):
        optimizer.zero_grad(set_to_none=True)
        _chunked_path_energy(
            model,
            left,
            right,
            basis=chart,
            interior_coordinates=interior,
            secant_norm=secant_norm,
            total_segments=segments,
            chunk_segments=chunk_segments,
            affine_chart_latents=affine_chart_latents,
            decoder_path_edge_energy=decoder_path_edge_energy,
            backward=True,
            torch=torch,
        )
        last_gradient_norm = float(torch.linalg.vector_norm(interior.grad))
        optimizer.step()
        with torch.no_grad():
            preprojection = torch.linalg.vector_norm(
                interior - line[1:-1],
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
                torch.linalg.vector_norm(interior - line[1:-1], dim=1).max()
            )
            maximum_postprojection_deviation = max(
                maximum_postprojection_deviation,
                postprojection_maximum,
            )
        if iteration in milestones:
            record(iteration, last_gradient_norm)
    if left.device.type == "cuda":
        torch.cuda.synchronize(left.device)
    elapsed = time.perf_counter() - started
    result = {
        "algorithm": optimizer_contract["algorithm"],
        "chart_dimension": dimension,
        "decoder_forward_backward_iterations": optimizer_contract["iterations"],
        "energy_chunk_segments": chunk_segments,
        "elapsed_seconds": elapsed,
        "endpoint_chord_squared_rms": endpoint_chord_squared,
        "history": history,
        "learning_rate": learning_rate,
        "path_segments": segments,
        "projection_applied_count": projection_applied_count,
        "projection_iteration_count": projection_iteration_count,
        "projection_error": projection_error,
        "status": "complete",
        "trust_boundary_contact": projection_iteration_count > 0,
    }
    json.dumps(result, allow_nan=False)
    return result


def _runtime_probe(
    model,
    left,
    right,
    basis,
    *,
    segment_candidates,
    chunk_segments,
    peak_reserved_bytes_max,
    affine_chart_latents,
    decoder_path_edge_energy,
    torch,
):
    dimension = min(16, basis.shape[0])
    chart = basis[:dimension]
    secant = right - left
    secant_norm = torch.linalg.vector_norm(secant).detach()
    endpoint_coordinates = (
        torch.einsum("dn,bn->d", chart.flatten(1), secant.flatten(1)) / secant_norm
    )
    rows = []
    for segments in segment_candidates:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(left.device)
        line = _path_coordinates(
            segments=segments,
            endpoint_coordinates=endpoint_coordinates,
            torch=torch,
        )
        interior = line[1:-1].clone().detach().requires_grad_(True)
        torch.cuda.synchronize(left.device)
        started = time.perf_counter()
        try:
            energy = _chunked_path_energy(
                model,
                left,
                right,
                basis=chart,
                interior_coordinates=interior,
                secant_norm=secant_norm,
                total_segments=segments,
                chunk_segments=chunk_segments,
                affine_chart_latents=affine_chart_latents,
                decoder_path_edge_energy=decoder_path_edge_energy,
                backward=True,
                torch=torch,
            )
            torch.cuda.synchronize(left.device)
            peak_memory = _memory(left.device, torch)
            rows.append({
                "elapsed_seconds": time.perf_counter() - started,
                "energy": energy,
                "energy_chunk_segments": chunk_segments,
                "path_segments": segments,
                "peak_memory": peak_memory,
                "status": (
                    "complete"
                    if peak_memory["peak_reserved_bytes"] < peak_reserved_bytes_max
                    else "resource_ceiling_exceeded"
                ),
            })
        except torch.cuda.OutOfMemoryError as error:
            rows.append({
                "exception_message": str(error),
                "path_segments": segments,
                "peak_memory": _memory(left.device, torch),
                "status": "cuda_oom",
            })
        finally:
            del interior, line
            torch.cuda.empty_cache()
    return rows


def _select_numerics(workers, contract):
    numerics = contract["numerics"]
    workloads = [
        workload
        for worker in workers
        for workload in worker["workloads"]
        if workload.get("status") == "complete"
    ]
    expected_workloads = (
        len(workers)
        * len(contract["scope"]["calibration_patch_ranks"])
        * len(contract["scope"]["routes"])
    )
    blockers = []
    if len(workloads) != expected_workloads:
        blockers.append("one_or_more_calibration_workloads_failed")

    epsilon_worst = {}
    for epsilon in numerics["finite_difference"]["epsilon_candidates"]:
        values = [
            row["relative_l2"]
            for workload in workloads
            for row in workload["chart_diagnostics"]["finite_difference"]
            if row["epsilon"] == epsilon
        ]
        if workloads and len(values) == len(workloads):
            epsilon_worst[str(epsilon)] = max(values)
    epsilon_limit = numerics["selection"]["finite_difference_worst_relative_l2_max"]
    passing_epsilons = [
        float(key) for key, value in epsilon_worst.items() if value <= epsilon_limit
    ]
    selected_epsilon = (
        min(
            passing_epsilons,
            key=lambda value: (epsilon_worst[str(value)], value),
        )
        if passing_epsilons
        else None
    )
    if selected_epsilon is None:
        blockers.append("no_finite_difference_epsilon_passed")

    ratio_floor = numerics["selection"]["conditioning_minimum_ratio"]
    dimension_minimum_ratios = {}
    passing_dimensions = []
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
            minimum = min(values)
            dimension_minimum_ratios[str(dimension)] = minimum
            if minimum >= ratio_floor:
                passing_dimensions.append(dimension)
    primary_dimension = None
    refinement_dimension = None
    if len(passing_dimensions) >= 2:
        primary_dimension = passing_dimensions[-2]
        refinement_dimension = passing_dimensions[-1]
    else:
        blockers.append("fewer_than_two_conditioned_nested_chart_dimensions")
    conditioning_dimension_pass_by_floor = {
        str(floor): [
            dimension
            for dimension in numerics["chart"]["dimensions"]
            if dimension_minimum_ratios.get(str(dimension), -1.0) >= floor
        ]
        for floor in numerics["conditioning"][
            "relative_minimum_singular_value_candidates"
        ]
    }

    linearity_worst = {}
    selected_radius = None
    linearity_limit = numerics["selection"]["sampled_linearity_relative_l2_max"]
    selected_dimensions = tuple(
        dimension
        for dimension in (primary_dimension, refinement_dimension)
        if dimension is not None
    )
    if selected_dimensions:
        expected_linearity_values = (
            len(workloads)
            * len(selected_dimensions)
            * len(numerics["chart"]["line_parameters"])
            * (1 + numerics["linearity"]["rademacher_combination_count"])
            * 2
        )
        for radius in numerics["linearity"]["radius_fractions_of_endpoint_secant"]:
            values = [
                row["relative_l2"]
                for workload in workloads
                for row in workload["chart_diagnostics"]["linearity"]
                if row["chart_dimension"] in selected_dimensions
                and row["radius_fraction"] == radius
            ]
            if len(values) == expected_linearity_values:
                linearity_worst[str(radius)] = max(values)
        acceptable_radii = [
            float(radius)
            for radius, value in linearity_worst.items()
            if value <= linearity_limit
        ]
        if acceptable_radii:
            selected_radius = max(acceptable_radii)
    if selected_radius is None:
        blockers.append("no_sampled_linearity_radius_passed")
    linearity_radius_pass_by_target = {
        str(target): [
            float(radius)
            for radius, value in linearity_worst.items()
            if value <= target
        ]
        for target in numerics["linearity"]["relative_l2_targets"]
    }

    optimizer_expected = (
        len(workers)
        * len(contract["scope"]["optimization_patch_ranks"])
        * len(contract["scope"]["routes"])
        * len(selected_dimensions)
        * len(numerics["optimizer"]["path_segments"])
    )
    learning_rate_worst_energy_ratio = {}
    optimizer_by_learning_rate = {}
    if selected_dimensions:
        for learning_rate in numerics["optimizer"]["learning_rates"]:
            candidates = [
                candidate
                for workload in workloads
                if workload["rank"] in contract["scope"]["optimization_patch_ranks"]
                for candidate in workload["optimizer_probes"]
                if candidate.get("status") == "complete"
                and candidate["chart_dimension"] in selected_dimensions
                and candidate["learning_rate"] == learning_rate
            ]
            if len(candidates) != optimizer_expected or any(
                candidate["trust_boundary_contact"] for candidate in candidates
            ):
                continue
            ratios = [
                min(row["normalized_energy"] for row in candidate["history"])
                / max(candidate["history"][0]["normalized_energy"], 1e-30)
                for candidate in candidates
            ]
            learning_rate_worst_energy_ratio[str(learning_rate)] = max(ratios)
            optimizer_by_learning_rate[str(learning_rate)] = candidates
    minimum_energy_decrease = numerics["selection"][
        "optimizer_minimum_relative_energy_decrease"
    ]
    passing_learning_rates = [
        float(key)
        for key, value in learning_rate_worst_energy_ratio.items()
        if value <= 1.0 - minimum_energy_decrease
    ]
    selected_learning_rate = (
        min(
            passing_learning_rates,
            key=lambda value: (
                learning_rate_worst_energy_ratio[str(value)],
                value,
            ),
        )
        if passing_learning_rates
        else None
    )
    selected_iterations = None
    if selected_learning_rate is None:
        blockers.append("no_common_improving_optimizer_candidate")
    else:
        candidates = optimizer_by_learning_rate[str(selected_learning_rate)]
        slack = numerics["selection"]["optimizer_energy_plateau_relative_slack"]
        for milestone in numerics["optimizer"]["milestones"][1:]:
            milestone_passes = []
            for candidate in candidates:
                history_by_iteration = {
                    row["iteration"]: row["normalized_energy"]
                    for row in candidate["history"]
                }
                best = min(history_by_iteration.values())
                milestone_passes.append(
                    history_by_iteration[milestone] <= (1.0 + slack) * best
                )
            if all(milestone_passes):
                selected_iterations = milestone
                break
        if selected_iterations is None or selected_iterations == 0:
            blockers.append("optimizer_did_not_improve_within_grid")
        elif selected_iterations == numerics["optimizer"]["milestones"][-1]:
            selected_iterations = None
            blockers.append("optimizer_not_plateaued_before_iteration_ceiling")

    runtime_complete = {}
    for segments in numerics["runtime_path_segments"]:
        rows = [
            row
            for worker in workers
            for row in (worker["runtime_path_probes"] or [])
            if row["path_segments"] == segments
        ]
        runtime_complete[str(segments)] = len(rows) == len(workers) and all(
            row["status"] == "complete" for row in rows
        )
    if not runtime_complete.get("16") or not runtime_complete.get("32"):
        blockers.append("K16_or_K32_runtime_probe_failed")

    selected = {
        "chart_primary_dimension": primary_dimension,
        "chart_refinement_dimension": refinement_dimension,
        "conditioning_minimum_ratio": ratio_floor,
        "finite_difference_epsilon": selected_epsilon,
        "finite_difference_worst_relative_l2_max": epsilon_limit,
        "optimizer_iterations": selected_iterations,
        "optimizer_learning_rate": selected_learning_rate,
        "optimizer_minimum_relative_energy_decrease": minimum_energy_decrease,
        "path_segments_primary": 16 if runtime_complete.get("16") else None,
        "path_segments_refinement": 32 if runtime_complete.get("32") else None,
        "path_segments_sensitivity": 64 if runtime_complete.get("64") else None,
        "sampled_linearity_radius_fraction": selected_radius,
        "sampled_linearity_relative_l2_max": linearity_limit,
    }
    return {
        "blockers": blockers,
        "decision_inputs": {
            "conditioning_dimension_pass_by_floor": (
                conditioning_dimension_pass_by_floor
            ),
            "dimension_minimum_ratios": dimension_minimum_ratios,
            "finite_difference_worst_relative_l2": epsilon_worst,
            "learning_rate_worst_best_to_initial_energy_ratio": (
                learning_rate_worst_energy_ratio
            ),
            "runtime_complete": runtime_complete,
            "sampled_linearity_radius_pass_by_target": (
                linearity_radius_pass_by_target
            ),
            "sampled_linearity_worst_relative_l2": linearity_worst,
        },
        "schema": "eqvae.functional_geometry.stage_a2.calibration.selection.v2",
        "selected_numerics": selected,
        "status": "selected" if not blockers else "unresolved",
    }


def _run_worker(model_name, device_index, repo_root_text, output_text, contract):
    try:
        repo_root = Path(repo_root_text)
        output = Path(output_text)
        sys.path.insert(0, str(repo_root / "src"))
        import numpy as np
        import torch

        from eqvae.evaluation.functional_geometry_calibration import (
            affine_chart_latents,
            decoder_path_edge_energy,
            decoder_visible_chart,
            project_line_deviation_,
            thin_metric_spectra,
        )
        from eqvae.evaluation.functional_geometry_rla import linearize_decoder
        from eqvae.evaluation.vae_test import state_dict_sha256
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

        weight_contract_path = WEIGHT_ROOT / "spec0045_vae_test_input.json"
        if (
            _sha256(weight_contract_path)
            != contract["inputs"]["weight_bundle_contract_sha256"]
        ):
            raise RuntimeError("frozen weight contract differs")
        weight_contract = json.loads(weight_contract_path.read_text(encoding="utf-8"))
        record = weight_contract["weights"][model_name]
        state_path = WEIGHT_ROOT / f"{model_name}_state.pt"
        if _sha256(state_path) != record["state_file_sha256"]:
            raise RuntimeError(f"frozen weights differ for {model_name}")
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        if state_dict_sha256(state) != record["state_dict_sha256"]:
            raise RuntimeError(f"frozen state differs for {model_name}")
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
        runtime_rows = []

        for local_index, rank in enumerate(calibration_ranks):
            left = base_mu[local_index : local_index + 1]
            endpoints = {
                "encoded": encoded_rotated_mu[local_index : local_index + 1],
                "prescribed": torch.rot90(left, 1, (-2, -1)),
            }
            for route in contract["scope"]["routes"]:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                right = endpoints[route]
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
                if rank in contract["scope"]["optimization_patch_ranks"]:
                    optimizer_grid = itertools.product(
                        numerics["optimizer"]["chart_dimensions"],
                        numerics["optimizer"]["path_segments"],
                        numerics["optimizer"]["learning_rates"],
                    )
                    for dimension, path_segments, learning_rate in optimizer_grid:
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats(device)
                        try:
                            candidate = _optimizer_probe(
                                model,
                                left,
                                right,
                                basis,
                                dimension=dimension,
                                learning_rate=learning_rate,
                                path_segments=path_segments,
                                optimizer_contract=numerics["optimizer"],
                                affine_chart_latents=affine_chart_latents,
                                decoder_path_edge_energy=decoder_path_edge_energy,
                                project_line_deviation_=project_line_deviation_,
                                torch=torch,
                            )
                        except Exception as error:
                            candidate_memory = capture_memory()
                            candidate = {
                                "chart_dimension": dimension,
                                "exception_message": str(error),
                                "exception_type": type(error).__name__,
                                "learning_rate": learning_rate,
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
                            chart_dimension=dimension,
                            learning_rate=learning_rate,
                            path_segments=path_segments,
                            **_memory(device, torch),
                        )
                        torch.cuda.empty_cache()
                if (
                    not runtime_rows
                    and rank == calibration_ranks[0]
                    and route == "encoded"
                ):
                    runtime_rows = _runtime_probe(
                        model,
                        left,
                        right,
                        basis,
                        segment_candidates=numerics["runtime_path_segments"],
                        chunk_segments=numerics["optimizer"]["energy_chunk_segments"],
                        peak_reserved_bytes_max=peak_ceiling,
                        affine_chart_latents=affine_chart_latents,
                        decoder_path_edge_energy=decoder_path_edge_energy,
                        torch=torch,
                    )
                    for runtime_row in runtime_rows:
                        runtime_memory = runtime_row.get("peak_memory")
                        if runtime_memory is not None:
                            peak_allocated_observed = max(
                                peak_allocated_observed,
                                runtime_memory["peak_allocated_bytes"],
                            )
                            peak_reserved_observed = max(
                                peak_reserved_observed,
                                runtime_memory["peak_reserved_bytes"],
                            )
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
            "runtime": {"cuda": torch.version.cuda, "torch": torch.__version__},
            "runtime_path_probes": runtime_rows,
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
        selection_sha256 = _sha256(selection_path)
        result = {
            "models": workers,
            "schema": "eqvae.functional_geometry.stage_a2.calibration.result.v2",
            "selection_sha256": selection_sha256,
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
            process.join()
        raise
