# Copyright 2026 HiperMaximus
# ruff: noqa: ANN001, ANN202, B023, BLE001, C901, COM812, D103, EM101, EM102, FBT003, INP001, PERF401, PLC0415, PLR0912, PLR0913, PLR0914, PLR0915, PLR2004, PLW0603, PLW0717, RUF031, S311, TRY003, TRY300, TRY301
"""Exact-C4 and matrix-free SLQ Stage A1 comparison on two T4s."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
import random
import shutil
import sys
import time
import traceback
import zipfile
from functools import partial
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# fmt: off
KAGGLE_FUNCTIONAL_GEOMETRY_STAGE_A1_READY = True
EMBEDDED_PAYLOAD_B64 = """
$embedded_payload_b64
"""
EMBEDDED_PAYLOAD_ZIP_SHA256 = "$embedded_payload_zip_sha256"
EMBEDDED_PAYLOAD_MANIFEST_SHA256 = "$embedded_payload_manifest_sha256"
# fmt: on

INPUT_ROOT = Path("/kaggle/input")
WORKING_ROOT = Path("/kaggle/working")
SOURCE_ROOT = WORKING_ROOT / ".functional_geometry_stage_a1_source"
OUTPUT_ROOT = WORKING_ROOT / "functional_geometry_stage_a1_c4_slq_v1"
CONTRACT_PATH = Path("docs/data/functional_geometry_stage_a1_contract.json")
SELECTOR_PATH = Path("runs/kaggle/fixed25_selector/fixed_25_validation_patches.json")
CONTRACT_SHA256 = "3e903de1b96437889ed11624e7c70733fbd7f181178d0a56d733bbbd18a58746"
MODEL_KINDS = {
    "normal_vae": "non_eq_vae_translatable",
    "so2_vae": "so2_vae_fixed",
}
PATCH_BYTES = 3 * 256 * 256
HEADER_BYTES = 64
_STARTED = time.perf_counter()
_SEQUENCE = 0


def _log(event: str, **values: object) -> None:
    global _SEQUENCE
    print(
        json.dumps(
            {
                "elapsed_seconds": round(time.perf_counter() - _STARTED, 3),
                "event": event,
                "pid": os.getpid(),
                "sequence": _SEQUENCE,
                **values,
            },
            allow_nan=False,
            sort_keys=True,
        ),
        flush=True,
    )
    _SEQUENCE += 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_payload() -> Path:
    payload = base64.b64decode(EMBEDDED_PAYLOAD_B64.encode("ascii"))
    if hashlib.sha256(payload).hexdigest() != EMBEDDED_PAYLOAD_ZIP_SHA256:
        raise RuntimeError("embedded payload differs")
    SOURCE_ROOT.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name in archive.namelist():
            path = Path(name)
            if path.is_absolute() or ".." in path.parts:
                raise RuntimeError("embedded payload path differs")
        archive.extractall(SOURCE_ROOT)
    manifest = SOURCE_ROOT / "payload_manifest.json"
    if _sha256(manifest) != EMBEDDED_PAYLOAD_MANIFEST_SHA256:
        raise RuntimeError("embedded payload manifest differs")
    contract_path = SOURCE_ROOT / CONTRACT_PATH
    if _sha256(contract_path) != CONTRACT_SHA256:
        raise RuntimeError("Stage A1 contract differs")
    return SOURCE_ROOT


def _atomic_json(path: Path, value: object) -> None:
    pending = path.with_suffix(path.suffix + ".tmp")
    with pending.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    pending.replace(path)


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


def _load_patches(payload_root, contract, *, np, torch):
    selector_path = payload_root / SELECTOR_PATH
    if _sha256(selector_path) != contract["inputs"]["fixed_selector_sha256"]:
        raise RuntimeError("fixed25 selector differs")
    selector = json.loads(selector_path.read_text(encoding="utf-8"))
    rows = selector.get("selectors")
    candidates = list(INPUT_ROOT.rglob("ubc_ocean_valid.bin"))
    if not isinstance(rows, list) or len(rows) != 25 or len(candidates) != 1:
        raise RuntimeError("fixed25 inputs differ")
    arrays = []
    with candidates[0].open("rb") as handle:
        for rank, row in enumerate(rows):
            if row.get("rank") != rank:
                raise RuntimeError("fixed25 ranks differ")
            handle.seek(HEADER_BYTES + int(row["file_index"]) * PATCH_BYTES)
            raw = handle.read(PATCH_BYTES)
            if (
                len(raw) != PATCH_BYTES
                or hashlib.sha256(raw).hexdigest() != row["patch_sha256"]
            ):
                raise RuntimeError(f"fixed25 patch {rank} differs")
            arrays.append(
                np.frombuffer(raw, dtype=np.uint8).reshape(3, 256, 256).copy()
            )
    patches = (
        torch.from_numpy(np.stack(arrays)).to(torch.float32).div(255).mul(2).sub(1)
    )
    return patches, rows


def _find_weight_bundle(contract):
    expected = contract["inputs"]["weight_bundle_contract_sha256"]
    candidates = [
        path
        for path in INPUT_ROOT.rglob("spec0045_vae_test_input.json")
        if _sha256(path) == expected
        and (path.parent / "normal_vae_state.pt").is_file()
        and (path.parent / "so2_vae_state.pt").is_file()
    ]
    if len(candidates) != 1:
        raise RuntimeError("frozen weight bundle differs")
    return candidates[0].parent, json.loads(candidates[0].read_text(encoding="utf-8"))


def _encode(model, images, *, batch_size, torch):
    means, logvars = [], []
    with torch.no_grad():
        for start in range(0, images.shape[0], batch_size):
            mean, logvar = model.encode(images[start : start + batch_size])
            means.append(mean.detach())
            logvars.append(logvar.detach())
    return torch.cat(means), torch.cat(logvars)


def _decode(model, latents, *, batch_size, torch):
    outputs = []
    with torch.no_grad():
        for start in range(0, latents.shape[0], batch_size):
            outputs.append(model.decode(latents[start : start + batch_size]).detach())
    return torch.cat(outputs)


def _relative_l2_rows(left, right, *, torch):
    numerator = torch.linalg.vector_norm(
        (left - right).to(torch.float64).flatten(1), dim=1
    )
    denominator = torch.maximum(
        torch.linalg.vector_norm(left.to(torch.float64).flatten(1), dim=1),
        torch.linalg.vector_norm(right.to(torch.float64).flatten(1), dim=1),
    ).clamp_min(1e-12)
    return (numerator / denominator).cpu().tolist()


def _rms_rows(values, *, torch):
    return (
        values.to(torch.float64).flatten(1).square().mean(dim=1).sqrt().cpu().tolist()
    )


def _percentile(sorted_values, probability):
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    return sorted_values[lower] * (upper - position) + sorted_values[upper] * (
        position - lower
    )


def _slq_uncertainty(result, *, dimension, seed, resamples, confidence, np):
    nodes = np.asarray([probe.nodes for probe in result.probes], dtype=np.float64)
    weights = np.asarray([probe.weights for probe in result.probes], dtype=np.float64)
    probe_traces = dimension * (nodes * weights).sum(axis=1)
    rng = np.random.default_rng(seed)
    estimates = {"trace": [], "d90": [], "d95": [], "d99": []}
    flat_nodes = nodes.reshape(-1)
    flat_weights = weights.reshape(-1)
    owners = np.repeat(np.arange(len(result.probes)), nodes.shape[1])
    order = np.argsort(flat_nodes)[::-1]
    flat_nodes, flat_weights, owners = (
        flat_nodes[order],
        flat_weights[order],
        owners[order],
    )
    for start in range(0, resamples, 64):
        count = min(64, resamples - start)
        draws = rng.integers(0, len(result.probes), size=(count, len(result.probes)))
        multiplicity = np.stack([
            np.bincount(row, minlength=len(result.probes)) for row in draws
        ])
        atom_multiplicity = (
            dimension
            * flat_weights[None]
            * multiplicity[:, owners]
            / len(result.probes)
        )
        atom_energy = atom_multiplicity * flat_nodes[None]
        totals = atom_energy.sum(axis=1)
        estimates["trace"].extend(totals.tolist())
        cumulative_energy = np.cumsum(atom_energy, axis=1)
        cumulative_dimension = np.cumsum(atom_multiplicity, axis=1)
        for label, fraction in (("d90", 0.90), ("d95", 0.95), ("d99", 0.99)):
            indices = (cumulative_energy >= fraction * totals[:, None]).argmax(axis=1)
            row_indices = np.arange(count)
            previous_energy = np.where(
                indices > 0,
                cumulative_energy[row_indices, np.maximum(indices - 1, 0)],
                0.0,
            )
            previous_dimension = np.where(
                indices > 0,
                cumulative_dimension[row_indices, np.maximum(indices - 1, 0)],
                0.0,
            )
            required = (fraction * totals - previous_energy) / flat_nodes[indices]
            estimates[label].extend(np.ceil(previous_dimension + required).tolist())
    alpha = (1.0 - confidence) / 2.0
    summaries = {}
    points = {
        "trace": result.energy.total_energy,
        "d90": result.energy.dimensions_for_90_percent,
        "d95": result.energy.dimensions_for_95_percent,
        "d99": result.energy.dimensions_for_99_percent,
    }
    for label, values in estimates.items():
        ordered = sorted(float(value) for value in values)
        summaries[label] = {
            "point": points[label],
            "bootstrap_std": float(np.std(ordered, ddof=1)),
            "ci99": [_percentile(ordered, alpha), _percentile(ordered, 1.0 - alpha)],
        }
    summaries["trace"]["probe_std"] = float(np.std(probe_traces, ddof=1))
    return summaries


def _run_worker(model_name, device_index, payload_root_text, staging_text, contract):
    active = {"phase": "worker_start"}
    failure_path = WORKING_ROOT / f"stage_a1_{model_name}_failure.json"
    try:
        payload_root = Path(payload_root_text)
        staging = Path(staging_text)
        sys.path.insert(0, str(payload_root / "src"))
        import numpy as np
        import torch

        from eqvae.evaluation.functional_geometry_rla import linearize_decoder
        from eqvae.evaluation.functional_geometry_slq import (
            decoder_metric_matvec,
            stochastic_lanczos_quadrature_batched,
        )
        from eqvae.evaluation.vae_test import sha256_file, state_dict_sha256
        from eqvae.models.registry import build_model

        if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
            raise RuntimeError("exactly two CUDA devices are required")
        if any("T4" not in torch.cuda.get_device_name(index) for index in range(2)):
            raise RuntimeError("both CUDA devices must be Tesla T4")
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
        _log(
            "worker_started",
            model=model_name,
            device=str(device),
            device_name=torch.cuda.get_device_name(device_index),
        )

        active = {"phase": "load_inputs"}
        patches, selectors = _load_patches(payload_root, contract, np=np, torch=torch)
        bundle_root, weight_contract = _find_weight_bundle(contract)
        record = weight_contract["weights"][model_name]
        state_path = bundle_root / f"{model_name}_state.pt"
        if (
            state_path.stat().st_size != record["state_file_bytes"]
            or sha256_file(state_path) != record["state_file_sha256"]
        ):
            raise RuntimeError(f"state file differs for {model_name}")
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        if state_dict_sha256(state) != record["state_dict_sha256"]:
            raise RuntimeError(f"state dictionary differs for {model_name}")
        model = build_model(MODEL_KINDS[model_name])
        model.load_state_dict(state, strict=True)
        model = model.to(device).eval().requires_grad_(False)
        del state
        state_hash = state_dict_sha256(model.state_dict())
        patches = patches.to(device)
        _log(
            "inputs_and_model_ready",
            model=model_name,
            patch_count=25,
            **_memory(device, torch),
        )

        active = {"phase": "c4"}
        base_mu, base_logvar = _encode(model, patches, batch_size=4, torch=torch)
        base_decoded = _decode(model, base_mu, batch_size=4, torch=torch)
        c4_rows, heavy = [], {}
        for angle in contract["scope"]["angles_degrees"]:
            turns = angle // 90
            observed_mu, observed_logvar = _encode(
                model, torch.rot90(patches, turns, (-2, -1)), batch_size=4, torch=torch
            )
            expected_mu = torch.rot90(base_mu, turns, (-2, -1))
            expected_logvar = torch.rot90(base_logvar, turns, (-2, -1))
            decoded_action = _decode(model, expected_mu, batch_size=4, torch=torch)
            mu_errors = _relative_l2_rows(observed_mu, expected_mu, torch=torch)
            logvar_errors = _relative_l2_rows(
                observed_logvar, expected_logvar, torch=torch
            )
            decoder_errors = _rms_rows(
                decoded_action - torch.rot90(base_decoded, turns, (-2, -1)), torch=torch
            )
            for rank, selector in enumerate(selectors):
                c4_rows.append({
                    "angle_degrees": angle,
                    "decoder_action_rms": decoder_errors[rank],
                    "encoder_logvar_relative_l2": logvar_errors[rank],
                    "encoder_mu_relative_l2": mu_errors[rank],
                    "label": selector["label"],
                    "rank": rank,
                    "sample_id": selector["sample_id"],
                })
            for rank in contract["scope"]["heavy_patch_ranks"]:
                heavy[(rank, angle)] = observed_mu[rank : rank + 1].cpu()
            _log(
                "c4_angle_complete",
                model=model_name,
                angle_degrees=angle,
                row_count=25,
                **_memory(device, torch),
            )
            del (
                observed_mu,
                observed_logvar,
                expected_mu,
                expected_logvar,
                decoded_action,
            )
        del base_mu, base_logvar, base_decoded, patches
        torch.cuda.empty_cache()

        numerics = contract["numerics"]
        metric_rows = 0
        validations = None
        spectra = []
        for state_index, ((rank, angle), cpu_latent) in enumerate(
            sorted(heavy.items())
        ):
            active = {"phase": "slq", "rank": rank, "angle_degrees": angle}
            latent = cpu_latent.to(device)
            operator = linearize_decoder(model.decode, latent, compile_operators=False)
            if validations is None:
                generator = torch.Generator(device=device).manual_seed(
                    numerics["probe_seed"] + 1
                )
                direction = (
                    torch
                    .randint(0, 2, latent.shape, generator=generator, device=device)
                    .mul(2)
                    .sub(1)
                    .to(torch.float32)
                )
                tangent = operator.jvp_batch(direction)
                epsilon = numerics["jvp_epsilon"]
                with torch.no_grad():
                    finite = (
                        model.decode(latent + epsilon * direction)
                        - model.decode(latent - epsilon * direction)
                    ) / (2 * epsilon)
                denominator = torch.maximum(
                    torch.linalg.vector_norm(tangent.to(torch.float64)),
                    torch.linalg.vector_norm(finite.to(torch.float64)),
                ).clamp_min(1e-12)
                jvp_error = float(
                    torch.linalg.vector_norm((tangent - finite).to(torch.float64))
                    / denominator
                )
                metric = decoder_metric_matvec(
                    operator, direction, microbatch=numerics["probe_batch_size"]
                )
                tangent_energy = float(tangent.to(torch.float64).square().sum())
                metric_energy = float(
                    (direction.to(torch.float64) * metric.to(torch.float64)).sum()
                )
                metric_error = abs(metric_energy - tangent_energy) / max(
                    tangent_energy, 1e-12
                )
                if (
                    jvp_error > numerics["jvp_relative_l2_max"]
                    or metric_error > numerics["metric_identity_relative_error_max"]
                ):
                    raise RuntimeError("JVP or metric identity validation failed")
                validations = {
                    "jvp_relative_l2": jvp_error,
                    "metric_identity_relative_error": metric_error,
                }
                _log(
                    "operator_validation_complete",
                    model=model_name,
                    **validations,
                    **_memory(device, torch),
                )
                del direction, tangent, finite, metric

            chord = torch.rot90(latent, 1, (-2, -1)) - latent
            chord_tangent = operator.jvp_batch(chord)
            chord_norm = float(torch.linalg.vector_norm(chord.to(torch.float64)))
            chord_energy = float(chord_tangent.to(torch.float64).square().sum())
            del chord_tangent

            def progress(probe_start, probe_count, step):
                nonlocal metric_rows
                metric_rows += probe_count
                values = {
                    "angle_degrees": angle,
                    "model": model_name,
                    "probe_first": probe_start + 1,
                    "probe_last": probe_start + probe_count,
                    "rank": rank,
                    "step": step,
                }
                if step == 1 or step % 8 == 0 or step == numerics["lanczos_steps"]:
                    values.update(_memory(device, torch))
                _log("slq_step_complete", **values)

            _log(
                "slq_state_started",
                model=model_name,
                rank=rank,
                angle_degrees=angle,
                probes=numerics["probe_count"],
                steps=numerics["lanczos_steps"],
            )
            metric_matvec = partial(
                decoder_metric_matvec, operator, microbatch=numerics["probe_batch_size"]
            )
            result = stochastic_lanczos_quadrature_batched(
                metric_matvec,
                vector_shape=tuple(latent.shape[1:]),
                probe_count=numerics["probe_count"],
                lanczos_steps=numerics["lanczos_steps"],
                seed=numerics["probe_seed"],
                probe_batch_size=numerics["probe_batch_size"],
                device=device,
                dtype=torch.float32,
                progress_callback=progress,
            )
            if any(
                probe.iterations != numerics["lanczos_steps"] for probe in result.probes
            ):
                raise RuntimeError("Lanczos depth differs")
            uncertainty = _slq_uncertainty(
                result,
                dimension=math.prod(latent.shape[1:]),
                seed=numerics["bootstrap_seed"] + state_index,
                resamples=numerics["bootstrap_resamples"],
                confidence=numerics["confidence_level"],
                np=np,
            )
            spectra.append({
                "angle_degrees": angle,
                "chord": {
                    "decoder_energy": chord_energy,
                    "decoder_energy_per_latent_norm_squared": chord_energy
                    / max(chord_norm * chord_norm, 1e-12),
                    "euclidean_norm": chord_norm,
                },
                "dimensions_90_estimate": result.energy.dimensions_for_90_percent,
                "dimensions_95_estimate": result.energy.dimensions_for_95_percent,
                "dimensions_99_estimate": result.energy.dimensions_for_99_percent,
                "probes": [
                    {
                        "iterations": probe.iterations,
                        "nodes": list(probe.nodes),
                        "weights": list(probe.weights),
                    }
                    for probe in result.probes
                ],
                "rank": rank,
                "trace_estimate": result.energy.total_energy,
                "uncertainty": uncertainty,
            })
            _log(
                "slq_state_complete",
                model=model_name,
                rank=rank,
                angle_degrees=angle,
                trace=result.energy.total_energy,
                d99=result.energy.dimensions_for_99_percent,
                **_memory(device, torch),
            )
            del chord, latent, metric_matvec, operator, result
            torch.cuda.empty_cache()

        expected_rows = contract["resources"]["heavy_metric_vector_rows"] // 2
        memory = _memory(device, torch)
        if metric_rows != expected_rows:
            raise RuntimeError(f"metric-vector row count differs: {metric_rows}")
        if (
            memory["peak_reserved_bytes"]
            >= contract["resources"]["peak_reserved_bytes_max_per_device"]
        ):
            raise RuntimeError("peak reserved CUDA memory exceeds contract")
        if state_dict_sha256(model.state_dict()) != state_hash:
            raise RuntimeError("frozen model state changed")
        worker = {
            "c4_rows": c4_rows,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device_index),
            "metric_vector_rows": metric_rows,
            "model": model_name,
            "peak_memory": memory,
            "runtime": {"cuda": torch.version.cuda, "torch": torch.__version__},
            "spectra": spectra,
            "validations": validations,
        }
        _atomic_json(staging / f"{model_name}.json", worker)
        _log(
            "worker_complete",
            model=model_name,
            metric_vector_rows=metric_rows,
            **memory,
        )
    except Exception as error:
        failure = {
            "active_work_unit": active,
            "exception_message": str(error),
            "exception_type": type(error).__name__,
            "model": model_name,
            "traceback": traceback.format_exc(),
        }
        _log("worker_failed", **failure)
        traceback.print_exc()
        try:
            _atomic_json(failure_path, failure)
        except Exception as artifact_error:
            _log(
                "failure_artifact_write_failed",
                model=model_name,
                exception_message=str(artifact_error),
            )
        raise


def _paired_c4_comparisons(workers, contract):
    normal = workers["normal_vae"]["c4_rows"]
    so2 = workers["so2_vae"]["c4_rows"]
    confidence = contract["numerics"]["confidence_level"]
    alpha = (1.0 - confidence) / 2.0
    results = []
    metrics = (
        "encoder_mu_relative_l2",
        "encoder_logvar_relative_l2",
        "decoder_action_rms",
    )
    for angle in contract["scope"]["angles_degrees"]:
        left = [row for row in normal if row["angle_degrees"] == angle]
        right = [row for row in so2 if row["angle_degrees"] == angle]
        for metric_index, metric in enumerate(metrics):
            differences = [
                a[metric] - b[metric] for a, b in zip(left, right, strict=True)
            ]
            rng = random.Random(
                contract["numerics"]["bootstrap_seed"] + angle + metric_index
            )
            boot = sorted(
                sum(differences[rng.randrange(len(differences))] for _ in differences)
                / len(differences)
                for _ in range(contract["numerics"]["bootstrap_resamples"])
            )
            mean = sum(differences) / len(differences)
            results.append({
                "angle_degrees": angle,
                "bootstrap_std": math.sqrt(
                    sum((value - sum(boot) / len(boot)) ** 2 for value in boot)
                    / (len(boot) - 1)
                ),
                "ci99": [_percentile(boot, alpha), _percentile(boot, 1.0 - alpha)],
                "mean_difference_normal_minus_so2": mean,
                "metric": metric,
            })
    return results


def main() -> int:
    active = "startup"
    staging = OUTPUT_ROOT.with_name(f".{OUTPUT_ROOT.name}.tmp")
    try:
        _log("run_started", model_device_map={"normal_vae": 0, "so2_vae": 1})
        payload_root = _extract_payload()
        contract = json.loads(
            (payload_root / CONTRACT_PATH).read_text(encoding="utf-8")
        )
        staging.mkdir(parents=True, exist_ok=False)
        _log("payload_contract_and_staging_ready", contract_sha256=CONTRACT_SHA256)
        active = "workers"
        context = mp.get_context("spawn")
        processes = []
        for model_name, device_index in contract["scope"]["model_device_map"].items():
            process = context.Process(
                target=_run_worker,
                args=(
                    model_name,
                    device_index,
                    str(payload_root),
                    str(staging),
                    contract,
                ),
                name=f"stage-a1-{model_name}",
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
            process.join()
            _log("worker_joined", process_name=process.name, exit_code=process.exitcode)
        failed = {
            process.name: process.exitcode
            for process in processes
            if process.exitcode != 0
        }
        if failed:
            raise RuntimeError(f"worker processes failed: {failed}")

        active = "merge"
        workers = {
            model: json.loads((staging / f"{model}.json").read_text(encoding="utf-8"))
            for model in MODEL_KINDS
        }
        result = {
            "c4_paired_comparisons": _paired_c4_comparisons(workers, contract),
            "models": workers,
            "schema": "eqvae.functional_geometry.stage_a1.result.v1",
            "scientific_model_comparison": True,
            "status": "complete_stage_a1",
        }
        _atomic_json(staging / "stage_a1_result.json", result)
        _atomic_json(staging / "run_contract.json", contract)
        runtime = {
            "elapsed_seconds": time.perf_counter() - _STARTED,
            "metric_vector_rows": sum(
                worker["metric_vector_rows"] for worker in workers.values()
            ),
            "worker_runtimes": {
                model: worker["runtime"] for model, worker in workers.items()
            },
        }
        _atomic_json(staging / "runtime.json", runtime)
        output_bytes = sum(path.stat().st_size for path in staging.iterdir())
        if output_bytes > contract["resources"]["output_bytes_max"]:
            raise RuntimeError("output exceeds byte ceiling")
        staging.replace(OUTPUT_ROOT)
        _log(
            "run_complete",
            output_root=OUTPUT_ROOT.name,
            output_bytes=output_bytes,
            **runtime,
        )
        return 0
    except Exception as error:
        failure = {
            "active_phase": active,
            "exception_message": str(error),
            "exception_type": type(error).__name__,
            "schema": "eqvae.functional_geometry.stage_a1.failure.v1",
            "traceback": traceback.format_exc(),
        }
        _log("run_failed", **failure)
        traceback.print_exc()
        try:
            _atomic_json(WORKING_ROOT / "stage_a1_failure.json", failure)
        except Exception as artifact_error:
            _log("failure_artifact_write_failed", exception_message=str(artifact_error))
        return 1
    finally:
        shutil.rmtree(SOURCE_ROOT, ignore_errors=True)


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
