# Copyright 2026 HiperMaximus
# ruff: noqa: ANN001, ANN202, ARG001, BLE001, C901, COM812, D103, EM101, EM102, FBT003, INP001, PERF401, PLC0415, PLR0913, PLR0914, PLR0915, PLR2004, RUF001, SLF001, TRY003
"""Generated private T4 wrapper for the locked Spec 0050 validation."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import shutil
import sys
import time
import traceback
import zipfile
from operator import itemgetter
from pathlib import Path

# fmt: off
KAGGLE_CORRECTED_ROTATION_GEOMETRY_READY = True
EMBEDDED_PAYLOAD_B64 = """
$embedded_payload_b64
"""
EMBEDDED_PAYLOAD_ZIP_SHA256 = (
    "$embedded_payload_zip_sha256"
)
EMBEDDED_PAYLOAD_MANIFEST_SHA256 = (
    "$embedded_payload_manifest_sha256"
)
# fmt: on

INPUT_ROOT = Path("/kaggle/input")
WORKING_ROOT = Path("/kaggle/working")
PRIVATE_ROOT = WORKING_ROOT / ".spec0050_payload"
OUTPUT_ROOT = WORKING_ROOT / "corrected_rotation_geometry_v1"
FIGURE_ROOT = OUTPUT_ROOT / "figures"
PATCH_BYTES = 3 * 256 * 256
HEADER_BYTES = 64
CONTRACT_SHA256 = "1dc6975979b120d85680526a3177e30caf93ca2eba7889a4452f2e78ca399756"
SPEC_SHA256 = "fbc58459d214ee32f1148f6b58507820586be1993749811294f14eddbf467b61"
MODEL_KINDS = {
    "normal_vae": "non_eq_vae_translatable",
    "so2_vae": "so2_vae_fixed",
}
FIT_RANKS = (23, 22, 1, 8, 4, 24, 9, 20, 11, 10, 0, 14, 5, 16, 7, 17, 12)
HELDOUT_RANKS = (18, 2, 21, 19, 3, 13, 15, 6)
FIT_ANGLES = tuple(range(0, 360, 5))
HELDOUT_ANGLES = tuple(angle for angle in range(360) if angle % 5)


def main() -> int:
    try:
        return _run()
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        shutil.rmtree(PRIVATE_ROOT, ignore_errors=True)


def _run() -> int:
    started = time.perf_counter()
    payload_root, payload_manifest = _extract_payload(PRIVATE_ROOT)
    sys.path.insert(0, str(payload_root / "src"))
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from eqvae.artifacts.rotation_orbits import (
        centered_disk_mask,
        continuous_rotate,
        disk_mean_f1,
        masked_relative_rms,
        rotate_f1_field,
        rotation_fixture_measurements,
        unpack_final_encoder_f1,
    )
    from eqvae.evaluation.rotation_geometry import (
        canonicalization_statistics,
        cyclic_geometry,
        evaluate_strict_rollout,
        exclude_cardinal_neighborhoods,
        f1_copy_diagnostics,
        fit_angle_probe,
        fit_generator_from_orbits,
        fit_orthogonal_step,
        fit_shared_basis,
        harmonic_summary,
        local_pca,
        masked_orbit_vectors,
        operator_numerics,
        paired_bootstrap_median_difference,
        principal_angles,
        score_angle_probe,
        tangent_summary,
    )
    from eqvae.evaluation.vae_test import sha256_file, state_dict_sha256
    from eqvae.models.registry import build_model

    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA GPU is required")
    if OUTPUT_ROOT.exists():
        raise RuntimeError("output root already exists")
    device = torch.device("cuda:0")
    rotation_fixture_audit = {
        "cpu": [rotation_fixture_measurements(size) for size in (32, 256)],
        "cuda": [
            rotation_fixture_measurements(size, device=device) for size in (32, 256)
        ],
    }
    _validate_rotation_fixture(rotation_fixture_audit)
    contract_path = payload_root / "docs/data/spec0050_rotation_geometry_contract.json"
    spec_path = (
        payload_root / "docs/specs/0050-corrected-rotation-geometry-validation.md"
    )
    _require_hash(contract_path, CONTRACT_SHA256)
    _require_hash(spec_path, SPEC_SHA256)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    selector_path = (
        payload_root / "runs/kaggle/fixed25_selector/fixed_25_validation_patches.json"
    )
    selector = json.loads(selector_path.read_text(encoding="utf-8"))
    _require_hash(selector_path, contract["inputs"]["selector_sha256"])
    patches = _load_fixed25(selector, np=np, torch=torch).to(device)
    bundle_root, weight_contract = _find_weight_bundle(
        contract["inputs"]["weight_bundle_contract_sha256"]
    )
    models = {}
    model_provenance = {}
    for branch, kind in MODEL_KINDS.items():
        record = weight_contract["weights"][branch]
        state_path = bundle_root / f"{branch}_state.pt"
        if state_path.stat().st_size != record["state_file_bytes"]:
            raise RuntimeError(f"state file size differs for {branch}")
        state_file_hash = sha256_file(state_path)
        if state_file_hash != record["state_file_sha256"]:
            raise RuntimeError(f"state file differs for {branch}")
        if (
            state_file_hash
            != contract["inputs"][f"{branch.split('_')[0]}_state_file_sha256"]
        ):
            raise RuntimeError(f"locked state file differs for {branch}")
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        state_hash = state_dict_sha256(state)
        if state_hash != record["state_dict_sha256"]:
            raise RuntimeError(f"state dict differs for {branch}")
        if (
            state_hash
            != contract["inputs"][f"{branch.split('_')[0]}_state_dict_sha256"]
        ):
            raise RuntimeError(f"locked state dict differs for {branch}")
        if (
            record["source_checkpoint_sha256"]
            != contract["inputs"][f"{branch.split('_')[0]}_checkpoint_sha256"]
        ):
            raise RuntimeError(f"locked checkpoint differs for {branch}")
        model = build_model(kind)
        model.load_state_dict(state, strict=True)
        model = model.to(device).eval().requires_grad_(False)
        models[branch] = model
        model_provenance[branch] = {
            "state_file_sha256": state_file_hash,
            "state_dict_sha256": state_hash,
            "source_checkpoint_sha256": record["source_checkpoint_sha256"],
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
        }

    latent_mask_t = centered_disk_mask(32, radius=14.0, device=device)
    input_mask_t = centered_disk_mask(256, radius=112.0, device=device)
    input_summary = _measure_input_paths(
        patches,
        input_mask_t,
        continuous_rotate=continuous_rotate,
        np=np,
        torch=torch,
    )

    branch_states = {}
    branch_results = {}
    f1_result = None
    for branch in ("normal_vae", "so2_vae"):
        collected = _collect_branch(
            models[branch],
            patches,
            latent_mask_t,
            is_so2=branch == "so2_vae",
            continuous_rotate=continuous_rotate,
            rotate_f1_field=rotate_f1_field,
            unpack_final_encoder_f1=unpack_final_encoder_f1,
            disk_mean_f1=disk_mean_f1,
            masked_relative_rms=masked_relative_rms,
            np=np,
            torch=torch,
        )
        vectors = masked_orbit_vectors(collected["mu"], latent_mask_t.cpu().numpy())
        del collected["mu"]
        shared = fit_shared_basis(vectors)
        pca_rows, tangent_rows, pairwise_angles = _local_geometry(
            vectors,
            local_pca=local_pca,
            harmonic_summary=harmonic_summary,
            tangent_summary=tangent_summary,
            principal_angles=principal_angles,
            np=np,
        )
        dimensions = _dimension_curve(
            vectors,
            shared,
            fit_generator_from_orbits=fit_generator_from_orbits,
            fit_orthogonal_step=fit_orthogonal_step,
            fit_shared_basis=fit_shared_basis,
            evaluate_strict_rollout=evaluate_strict_rollout,
            operator_numerics=operator_numerics,
            np=np,
        )
        geometry = {
            str(step): cyclic_geometry(vectors, step_degrees=step) for step in (1, 2, 5)
        }
        branch_results[branch] = {
            "geometry": geometry,
            "cardinal_excluded": {
                str(step): exclude_cardinal_neighborhoods(
                    vectors,
                    step_degrees=step,
                )
                for step in (1, 2, 5)
            },
            "local_pca": pca_rows,
            "tangents": tangent_rows,
            "tangent_subspace_pairwise_principal_angles": pairwise_angles,
            "shared_action_dimensions": dimensions,
            "spatial_action": {
                "dense_residual_per_patch": np.median(
                    collected["spatial_residual"], axis=1
                ).tolist(),
                "dense_residual_median": float(
                    np.median(collected["spatial_residual"])
                ),
                "exact_quarter_residual": collected["exact_quarter"],
                "canonicalization_all25": canonicalization_statistics(
                    collected["spatial_canonical"],
                    vectors,
                ),
            },
        }
        if branch == "so2_vae":
            f1_result = f1_copy_diagnostics(
                collected["f1_pooled"],
                collected["f1_residual"],
            )
        branch_states[branch] = {
            "vectors": vectors,
            "shared": shared,
            "pca_rows": pca_rows,
            "spatial_canonical": collected["spatial_canonical"],
            "f1_pooled": collected.get("f1_pooled"),
            "f1_residual": collected.get("f1_residual"),
        }
        torch.cuda.empty_cache()

    selected_dimension = _select_dimension(branch_results, np=np)
    decisions = _evaluate_decisions(
        branch_results,
        branch_states,
        input_summary,
        selected_dimension,
        f1_result,
        fit_generator_from_orbits=fit_generator_from_orbits,
        canonicalization_statistics=canonicalization_statistics,
        fit_angle_probe=fit_angle_probe,
        score_angle_probe=score_angle_probe,
        np=np,
    )
    bundle_result = _bundle_diagnostics(
        branch_states["so2_vae"],
        branch_results["so2_vae"],
        input_summary,
        selected_dimension,
        f1_result,
        np=np,
    )
    comparisons = _population_comparisons(
        branch_results,
        input_summary,
        paired_bootstrap_median_difference=paired_bootstrap_median_difference,
        np=np,
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=False)
    _render_figures(
        branch_results,
        comparisons,
        decisions,
        f1_result,
        selected_dimension,
        plt=plt,
        np=np,
    )
    arrays_path = OUTPUT_ROOT / "selected_arrays.npz"
    np.savez_compressed(
        arrays_path,
        rank0_normal=np.asarray(branch_results["normal_vae"]["local_pca"][0]["scores"]),
        rank0_so2=np.asarray(branch_results["so2_vae"]["local_pca"][0]["scores"]),
        rank12_normal=np.asarray(
            branch_results["normal_vae"]["local_pca"][12]["scores"]
        ),
        rank12_so2=np.asarray(branch_results["so2_vae"]["local_pca"][12]["scores"]),
        shared_scores_normal=_shared_scores(branch_states["normal_vae"], np=np),
        shared_scores_so2=_shared_scores(branch_states["so2_vae"], np=np),
        population_normal=np.column_stack([
            comparisons[key]["normal"]
            for key in ("local_linearity_ratio", "step_size_cv", "path_length")
        ]),
        population_so2=np.column_stack([
            comparisons[key]["so2"]
            for key in ("local_linearity_ratio", "step_size_cv", "path_length")
        ]),
        population_input=np.column_stack([
            comparisons[key]["input"]
            for key in ("local_linearity_ratio", "step_size_cv", "path_length")
        ]),
        f1_pooled=branch_states["so2_vae"]["f1_pooled"],
        f1_residual=branch_states["so2_vae"]["f1_residual"],
    )
    figure_hashes = {
        str(path.relative_to(OUTPUT_ROOT)): _sha256(path)
        for path in sorted(FIGURE_ROOT.glob("*.png"))
    }
    result = {
        "schema": "spec0050.corrected_rotation_geometry.v1",
        "label": "fixed-validation post-hoc exploratory; not sealed-test evidence",
        "contract": contract,
        "contract_sha256": CONTRACT_SHA256,
        "spec_sha256": SPEC_SHA256,
        "payload": payload_manifest,
        "models": model_provenance,
        "runtime": {
            "seconds": time.perf_counter() - started,
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "matplotlib": mpl.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "fp32": True,
        },
        "input_path": input_summary,
        "rotation_fixture_audit": rotation_fixture_audit,
        "branches": branch_results,
        "f1": f1_result,
        "comparisons": comparisons,
        "selected_dimension": selected_dimension,
        "decisions": decisions,
        "bundle_diagnostics": bundle_result,
        "output_hashes": figure_hashes | {"selected_arrays.npz": _sha256(arrays_path)},
    }
    summary_path = OUTPUT_ROOT / "rotation_geometry_summary.json"
    summary_path.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    manifest = {
        "schema": "spec0050.corrected_rotation_geometry_manifest.v1",
        "summary_sha256": _sha256(summary_path),
        "files": {
            str(path.relative_to(OUTPUT_ROOT)): _sha256(path)
            for path in sorted(OUTPUT_ROOT.rglob("*"))
            if path.is_file()
        },
        "supersedes": {
            "remote": "maximshtefan/eqvae-fixed25-dense-rotation-population/1",
            "local": "runs/local/frozen_vae_rotation_orbits",
            "reason": "mixed opposite rotation conventions at cardinal angles",
            "rank12_reproduction": {
                "input_step_rms_87_to_88": 0.10192135,
                "input_step_rms_88_to_89": 0.10346428,
                "input_step_rms_89_to_90": 0.24039730,
                "input_step_rms_90_to_91": 0.23987600,
                "input_step_rms_91_to_92": 0.10254467,
                "angle_89_999_vs_positive_quarter_rms": 0.23747779,
                "angle_89_999_vs_negative_quarter_rms": 0.00016210,
            },
        },
    }
    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return 0


def _extract_payload(destination: Path):
    payload = base64.b64decode(EMBEDDED_PAYLOAD_B64.encode("ascii"))
    if hashlib.sha256(payload).hexdigest() != EMBEDDED_PAYLOAD_ZIP_SHA256:
        raise RuntimeError("embedded payload zip hash mismatch")
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name in archive.namelist():
            candidate = Path(name)
            if candidate.is_absolute() or ".." in candidate.parts:
                raise RuntimeError(f"unsafe embedded payload path: {name}")
        archive.extractall(destination)
    manifest_path = destination / "payload_manifest.json"
    if _sha256(manifest_path) != EMBEDDED_PAYLOAD_MANIFEST_SHA256:
        raise RuntimeError("embedded payload manifest hash mismatch")
    return destination, json.loads(manifest_path.read_text(encoding="utf-8"))


def _find_weight_bundle(expected_contract_sha256):
    candidates = [
        path
        for path in INPUT_ROOT.rglob("spec0045_vae_test_input.json")
        if _sha256(path) == expected_contract_sha256
        and (path.parent / "normal_vae_state.pt").is_file()
        and (path.parent / "so2_vae_state.pt").is_file()
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one exact frozen weight bundle, found {candidates}"
        )
    contract_path = candidates[0]
    return contract_path.parent, json.loads(contract_path.read_text(encoding="utf-8"))


def _validate_rotation_fixture(audit):
    for device_rows in audit.values():
        for row in device_rows:
            if max(row["centroid_error_normalized"].values()) >= 0.02:
                raise RuntimeError("rotation fixture centroid threshold failed")
            if max(row["orientation_error_degrees"].values()) >= 0.5:
                raise RuntimeError("rotation fixture orientation threshold failed")
            if max(row["cardinal_absolute_error"].values()) > 2e-6:
                raise RuntimeError("rotation fixture cardinal threshold failed")
            sided = [
                value
                for sides in row["cardinal_sided_rms"].values()
                for variants in sides.values()
                for value in variants.values()
            ]
            if max(sided) >= 5e-4:
                raise RuntimeError("rotation fixture sided-limit threshold failed")
            inverse = [
                value
                for variants in row["inverse_normalized_rms"].values()
                for value in variants.values()
            ]
            composition = [
                value
                for variants in row["composition_normalized_rms"].values()
                for value in variants.values()
            ]
            if max(inverse + composition) >= 0.03:
                raise RuntimeError("rotation fixture interpolation threshold failed")
            if row["l_arrow_positive_quarter_max_absolute_error"] > 2e-6:
                raise RuntimeError("rotation fixture L-arrow threshold failed")
            if row["boundary_wedge_border_mutant_normalized_rms"] <= 1e-4:
                raise RuntimeError("rotation fixture padding discriminator failed")
            f1 = row["f1_fixture"]
            if f1 is not None and (
                f1["noncardinal_centroid_error_normalized"] >= 0.02
                or f1["noncardinal_phase_error_degrees"] >= 0.5
                or f1["positive_quarter_component_x_max"] > 2e-6
                or f1["positive_quarter_component_y_max"] < 0.90
            ):
                raise RuntimeError("rotation fixture F1 threshold failed")


def _load_fixed25(selector, *, np, torch):
    candidates = list(INPUT_ROOT.rglob("ubc_ocean_valid.bin"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one validation binary, found {candidates}")
    rows = selector.get("selectors")
    if not isinstance(rows, list) or len(rows) != 25:
        raise RuntimeError("fixed25 selector differs")
    arrays = []
    with candidates[0].open("rb") as handle:
        for expected_rank, row in enumerate(rows):
            if row.get("rank") != expected_rank:
                raise RuntimeError("fixed25 selector ranks are not contiguous")
            handle.seek(HEADER_BYTES + int(row["file_index"]) * PATCH_BYTES)
            raw = handle.read(PATCH_BYTES)
            if (
                len(raw) != PATCH_BYTES
                or hashlib.sha256(raw).hexdigest() != row["patch_sha256"]
            ):
                raise RuntimeError(f"fixed25 patch {expected_rank} differs")
            arrays.append(
                np.frombuffer(raw, dtype=np.uint8).reshape(3, 256, 256).copy()
            )
    return (
        torch
        .from_numpy(np.stack(arrays))
        .to(dtype=torch.float32)
        .div(255)
        .mul(2)
        .sub(1)
    )


def _measure_input_paths(patches, mask, *, continuous_rotate, np, torch):
    variants = {"full": [], "disk": []}
    stabilizers = []
    inverse_rows = []
    composition_rows = []
    with torch.inference_mode():
        for patch in patches:
            orbit = torch.cat(
                [continuous_rotate(patch[None], angle) for angle in range(360)],
                dim=0,
            )
            for name, selected in (
                ("full", orbit.reshape(1, 360, -1)),
                ("disk", orbit[:, :, mask].reshape(1, 360, -1)),
            ):
                variants[name].append(
                    {
                        str(step): _single_row_geometry(selected, step=step, np=np)
                        for step in (1, 2, 5)
                    }
                    | {
                        "cardinal_excluded": {
                            str(step): _single_row_exclusion(
                                selected,
                                step=step,
                                np=np,
                            )
                            for step in (1, 2, 5)
                        }
                    }
                )
            disk_values = orbit[:, :, mask].reshape(360, -1)
            centered = disk_values[0] - disk_values[0].mean()
            scale = centered.square().mean().sqrt().clamp_min(1e-8)
            self_distance = (disk_values - disk_values[0]).square().mean(
                dim=1
            ).sqrt() / scale
            orientation_strength = float(self_distance.max())
            c2_strength = 1.0 - float(self_distance[180]) / (
                0.5 * float(self_distance[90] + self_distance[270]) + 1e-8
            )
            stabilizers.append({
                "self_distance": self_distance.cpu().tolist(),
                "orientation_strength": orientation_strength,
                "c2_strength": c2_strength,
                "near_isotropic": orientation_strength < 0.10,
                "possible_c2": float(self_distance[180]) < 0.25
                and float(self_distance[180])
                < 0.5 * 0.5 * float(self_distance[90] + self_distance[270]),
            })
            inverse = {}
            for angle in (17, 31, 73):
                restored = continuous_rotate(
                    continuous_rotate(patch[None], angle), -angle
                )
                inverse[str(angle)] = {
                    "full": float(
                        (restored - patch[None]).square().mean().sqrt()
                        / patch.square().mean().sqrt().clamp_min(1e-8)
                    ),
                    "disk": float(
                        (restored[:, :, mask] - patch[None, :, mask])
                        .square()
                        .mean()
                        .sqrt()
                        / patch[:, mask].square().mean().sqrt().clamp_min(1e-8)
                    ),
                }
            inverse_rows.append(inverse)
            composition = {}
            for alpha, beta in ((17, -17), (31, 73), (-73, 31)):
                observed = continuous_rotate(
                    continuous_rotate(patch[None], beta), alpha
                )
                expected = continuous_rotate(patch[None], alpha + beta)
                composition[f"{alpha},{beta}"] = {
                    "full": float(
                        (observed - expected).square().mean().sqrt()
                        / expected.square().mean().sqrt().clamp_min(1e-8)
                    ),
                    "disk": float(
                        (observed[:, :, mask] - expected[:, :, mask])
                        .square()
                        .mean()
                        .sqrt()
                        / expected[:, :, mask].square().mean().sqrt().clamp_min(1e-8)
                    ),
                }
            composition_rows.append(composition)
            del orbit
    return {
        "variants": variants,
        "stabilizers": stabilizers,
        "inverse_normalized_rms": inverse_rows,
        "composition_normalized_rms": composition_rows,
    }


def _single_row_geometry(values, *, step, np):
    sampled = values[:, ::step]
    first = np.roll(sampled.cpu().numpy(), -1, axis=1) - sampled.cpu().numpy()
    second = np.roll(first, -1, axis=1) - first
    sampled_np = sampled.cpu().numpy()
    delta_radians = math.radians(step)
    tangent = (np.roll(sampled_np, -1, axis=1) - np.roll(sampled_np, 1, axis=1)) / (
        2.0 * delta_radians
    )
    acceleration = (
        np.roll(sampled_np, -1, axis=1)
        - 2.0 * sampled_np
        + np.roll(sampled_np, 1, axis=1)
    ) / delta_radians**2
    tangent_norm = np.sqrt(np.mean(np.square(tangent), axis=2))
    acceleration_norm = np.sqrt(np.mean(np.square(acceleration), axis=2))
    tangent_dot = np.mean(tangent * acceleration, axis=2)
    curvature = np.sqrt(
        np.maximum(
            acceleration_norm**2 - np.square(tangent_dot / (tangent_norm + 1e-8)),
            0.0,
        )
    ) / (np.square(tangent_norm) + 1e-8)
    tangent_next = np.roll(tangent, -1, axis=1)
    turning = np.arccos(
        np.clip(
            np.sum(tangent * tangent_next, axis=2)
            / (
                np.linalg.norm(tangent, axis=2) * np.linalg.norm(tangent_next, axis=2)
                + 1e-8
            ),
            -1.0,
            1.0,
        )
    )
    step_rms = np.sqrt(np.mean(np.square(first), axis=2))[0]
    return {
        "local_linearity_ratio": float(
            np.sqrt(np.mean(np.square(second)))
            / (np.sqrt(np.mean(np.square(first))) + 1e-8)
        ),
        "step_size_cv": float(np.std(step_rms) / (np.mean(step_rms) + 1e-8)),
        "path_length": float(np.sum(step_rms)),
        "curvature_median": float(np.median(curvature)),
        "curvature_q90": float(np.quantile(curvature, 0.9)),
        "tangent_norm": tangent_norm[0].tolist(),
        "acceleration_norm": acceleration_norm[0].tolist(),
        "turning_angle_radians": turning[0].tolist(),
        "curvature": curvature[0].tolist(),
        "step_rms": step_rms.tolist(),
    }


def _single_row_exclusion(values, *, step, np):
    from eqvae.evaluation.rotation_geometry import exclude_cardinal_neighborhoods

    return {
        key: value[0] if isinstance(value, list) else value
        for key, value in exclude_cardinal_neighborhoods(
            values.cpu().numpy(),
            step_degrees=step,
        ).items()
    }


def _collect_branch(
    model,
    patches,
    latent_mask,
    *,
    is_so2,
    continuous_rotate,
    rotate_f1_field,
    unpack_final_encoder_f1,
    disk_mean_f1,
    masked_relative_rms,
    np,
    torch,
):
    mu_values = np.empty((25, 360, 16, 32, 32), dtype=np.float32)
    canonical = np.empty((25, 360, 16 * int(latent_mask.sum())), dtype=np.float32)
    spatial_residual = np.empty((25, 360), dtype=np.float32)
    f1_pooled = np.empty((25, 360, 48, 2), dtype=np.float32) if is_so2 else None
    f1_residual = np.empty((25, 360, 48), dtype=np.float32) if is_so2 else None
    base_mu = None
    base_f1 = None
    with torch.inference_mode():
        for angle in range(360):
            rotated = continuous_rotate(patches, angle)
            if is_so2:
                hidden = model._encode_features(rotated)
                mu = model.mu_head(hidden)
                f1 = unpack_final_encoder_f1(hidden)
                if angle == 0:
                    base_f1 = f1.detach().clone()
                f1_pooled[:, angle] = disk_mean_f1(f1, mask=latent_mask).cpu().numpy()
                expected_f1 = rotate_f1_field(base_f1, angle)
                weights = latent_mask[None, None, None].to(f1.dtype)
                numerator = ((f1 - expected_f1).square() * weights).sum(dim=(2, 3, 4))
                denominator = (expected_f1.square() * weights).sum(dim=(2, 3, 4))
                f1_residual[:, angle] = (
                    (
                        torch.sqrt(numerator / (2.0 * latent_mask.sum()))
                        / (torch.sqrt(denominator / (2.0 * latent_mask.sum())) + 1e-8)
                    )
                    .cpu()
                    .numpy()
                )
            else:
                mu = model.encode(rotated)[0]
            if angle == 0:
                base_mu = mu.detach().clone()
            expected_mu = continuous_rotate(base_mu, angle)
            spatial_residual[:, angle] = (
                masked_relative_rms(
                    mu,
                    expected_mu,
                    mask=latent_mask,
                )
                .cpu()
                .numpy()
            )
            restored = continuous_rotate(mu, -angle)
            canonical[:, angle] = (
                restored[:, :, latent_mask].reshape(25, -1).cpu().numpy()
            )
            mu_values[:, angle] = mu.cpu().numpy()
    exact_quarter = _exact_quarter_control(
        model,
        patches,
        latent_mask,
        is_so2=is_so2,
        masked_relative_rms=masked_relative_rms,
        np=np,
        torch=torch,
    )
    return {
        "mu": mu_values,
        "spatial_canonical": canonical.astype(np.float64),
        "spatial_residual": spatial_residual,
        "exact_quarter": exact_quarter,
        "f1_pooled": f1_pooled,
        "f1_residual": f1_residual,
    }


def _exact_quarter_control(
    model, patches, mask, *, is_so2, masked_relative_rms, np, torch
):
    rows = []
    f1_rows = []
    with torch.inference_mode():
        base = model.encode(patches)[0]
        base_f1 = None
        if is_so2:
            from eqvae.artifacts.rotation_orbits import unpack_final_encoder_f1

            base_f1 = unpack_final_encoder_f1(model._encode_features(patches))
        for angle in (90, 180, 270):
            observed = model.encode(torch.rot90(patches, angle // 90, (-2, -1)))[0]
            expected = torch.rot90(base, angle // 90, (-2, -1))
            rows.append(
                masked_relative_rms(observed, expected, mask=mask).cpu().numpy()
            )
            if is_so2:
                observed_f1 = unpack_final_encoder_f1(
                    model._encode_features(torch.rot90(patches, angle // 90, (-2, -1)))
                )
                spatial = torch.rot90(base_f1, angle // 90, (-2, -1))
                if angle == 90:
                    expected_f1 = torch.stack(
                        (-spatial[:, :, 1], spatial[:, :, 0]), dim=2
                    )
                elif angle == 180:
                    expected_f1 = -spatial
                else:
                    expected_f1 = torch.stack(
                        (spatial[:, :, 1], -spatial[:, :, 0]), dim=2
                    )
                weights = mask[None, None, None].to(observed_f1.dtype)
                numerator = ((observed_f1 - expected_f1).square() * weights).sum(
                    dim=(2, 3, 4)
                )
                denominator = (expected_f1.square() * weights).sum(dim=(2, 3, 4))
                f1_rows.append(
                    (
                        torch.sqrt(numerator / (2.0 * mask.sum()))
                        / (torch.sqrt(denominator / (2.0 * mask.sum())) + 1e-8)
                    )
                    .cpu()
                    .numpy()
                )
    values = np.stack(rows, axis=1)
    result = {
        "angles": [90, 180, 270],
        "per_patch": values.tolist(),
        "median": np.median(values, axis=0).tolist(),
    }
    if f1_rows:
        f1_values = np.stack(f1_rows, axis=1)
        result["f1_full_field_per_patch_angle_copy"] = f1_values.tolist()
        result["f1_full_field_copy_median"] = np.median(
            f1_values,
            axis=(0, 1),
        ).tolist()
        result["f1_full_field_global_median"] = float(np.median(f1_values))
    return result


def _local_geometry(
    vectors,
    *,
    local_pca,
    harmonic_summary,
    tangent_summary,
    principal_angles,
    np,
):
    pca_rows = []
    tangent_rows = []
    subspaces = []
    for patch in range(25):
        pca = local_pca(vectors[patch], components=6)
        raw_centered = vectors[patch] - vectors[patch].mean(axis=0, keepdims=True)
        raw_total = float(np.mean(np.sum(np.square(raw_centered), axis=1)))
        harmonic = harmonic_summary(pca.scores, raw_total_energy=raw_total)
        pca_rows.append({
            "rank": patch,
            "scores": pca.scores.tolist(),
            "explained_fraction": pca.explained_fraction.tolist(),
            "cumulative_explained": np.cumsum(pca.explained_fraction).tolist(),
            "eigenvalues": pca.eigenvalues.tolist(),
            "harmonics": harmonic,
        })
        tangent = tangent_summary(vectors[patch])
        subspaces.append(np.asarray(tangent.pop("top2_left_vectors"), dtype=np.float64))
        tangent["rank"] = patch
        tangent_rows.append(tangent)
    pairwise = []
    for left in range(25):
        for right in range(left + 1, 25):
            pairwise.append({
                "left": left,
                "right": right,
                "angles_radians": principal_angles(
                    subspaces[left], subspaces[right]
                ).tolist(),
            })
    return pca_rows, tangent_rows, pairwise


def _dimension_curve(
    vectors,
    shared,
    *,
    fit_generator_from_orbits,
    fit_orthogonal_step,
    fit_shared_basis,
    evaluate_strict_rollout,
    operator_numerics,
    np,
):
    rows = []
    fit_rank_array = np.asarray(FIT_RANKS, dtype=np.int64)
    fit_angle_array = np.asarray(FIT_ANGLES, dtype=np.int64)
    fit_values = vectors[fit_rank_array][:, fit_angle_array]
    fit_centers = fit_values.mean(axis=1, keepdims=True)
    lopo_shared = {
        omitted: fit_shared_basis(
            vectors,
            fit_ranks=tuple(rank for rank in FIT_RANKS if rank != omitted),
        )
        for omitted in FIT_RANKS
    }
    heldout_values = vectors[np.asarray(HELDOUT_RANKS, dtype=np.int64)]
    heldout_centers = heldout_values[:, fit_angle_array].mean(axis=1, keepdims=True)
    for dimension in range(1, 7):
        basis = shared.basis[:, :dimension]
        fit_residual = fit_values - fit_centers
        heldout_residual = heldout_values - heldout_centers
        fit_capture = float(
            np.sum(np.square(fit_residual @ basis))
            / (np.sum(np.square(fit_residual)) + 1e-8)
        )
        heldout_capture = float(
            np.sum(np.square(heldout_residual @ basis))
            / (np.sum(np.square(heldout_residual)) + 1e-8)
        )
        strict = fit_generator_from_orbits(
            vectors, shared, dimension=dimension, conditional=False
        )
        conditional = fit_generator_from_orbits(
            vectors, shared, dimension=dimension, conditional=True
        )
        shuffled = fit_generator_from_orbits(
            vectors,
            shared,
            dimension=dimension,
            conditional=False,
            shuffled_seed=284733341,
        )
        procrustes_states = (fit_values - shared.global_center[None, None, :]) @ basis
        procrustes = fit_orthogonal_step(
            procrustes_states.reshape(-1, dimension),
            np.roll(procrustes_states, -1, axis=1).reshape(-1, dimension),
        )
        conditional_states = (fit_values - fit_centers) @ basis
        conditional_procrustes = fit_orthogonal_step(
            conditional_states.reshape(-1, dimension),
            np.roll(conditional_states, -1, axis=1).reshape(-1, dimension),
        )
        zero = type(strict)(
            matrix=np.zeros_like(strict.matrix),
            design_singular_values=np.empty(0),
            design_rank=0,
            design_condition=1.0,
            eigenfrequencies=np.empty(0),
        )
        strict_eval = evaluate_strict_rollout(
            vectors, shared, strict, dimension=dimension
        )
        identity_eval = evaluate_strict_rollout(
            vectors, shared, zero, dimension=dimension
        )
        shuffled_eval = evaluate_strict_rollout(
            vectors, shared, shuffled, dimension=dimension
        )
        conditional_eval = _evaluate_conditional(
            vectors, shared, conditional, dimension, np=np
        )
        local_eval = _evaluate_preceding_anchor(
            vectors, shared, strict, dimension, np=np
        )
        per_patch = _evaluate_patch_specific(vectors, shared, dimension, np=np)
        lopo = _lopo_generator(vectors, lopo_shared, dimension, np=np)
        rows.append({
            "dimension": dimension,
            "fit_variance_capture": fit_capture,
            "heldout_variance_capture": heldout_capture,
            "strict_generator": _generator_dict(strict, np=np),
            "conditional_generator": _generator_dict(conditional, np=np),
            "shuffled_generator": _generator_dict(shuffled, np=np),
            "orthogonal_procrustes_step": procrustes.tolist(),
            "operator_numerics": operator_numerics(
                strict.matrix,
                procrustes,
                step_radians=math.radians(5.0),
            ),
            "conditional_orthogonal_procrustes_step": conditional_procrustes.tolist(),
            "conditional_operator_numerics": operator_numerics(
                conditional.matrix,
                conditional_procrustes,
                step_radians=math.radians(5.0),
            ),
            "invariant_planes": _invariant_plane_diagnostics(
                strict.matrix,
                procrustes_states.reshape(-1, dimension),
                np=np,
            ),
            "strict_global": strict_eval,
            "identity": identity_eval,
            "shuffled": shuffled_eval,
            "coarse_orbit_conditioned": conditional_eval,
            "preceding_anchor": local_eval,
            "patch_specific_overfit": per_patch,
            "leave_one_fit_patch_out": lopo,
        })
    return rows


def _generator_dict(fit, *, np):
    return {
        "matrix": fit.matrix.tolist(),
        "design_singular_values": fit.design_singular_values.tolist(),
        "design_rank": fit.design_rank,
        "design_condition": fit.design_condition,
        "eigenfrequencies": fit.eigenfrequencies.tolist(),
        "skew_error": float(np.max(np.abs(fit.matrix + fit.matrix.T))),
    }


def _invariant_plane_diagnostics(matrix, states, *, np):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    total = float(np.sum(np.square(states)))
    rows = []
    for index in np.flatnonzero(eigenvalues.imag > 1e-8):
        vector = eigenvectors[:, index]
        plane, _ = np.linalg.qr(np.column_stack((vector.real, -vector.imag)))
        if (plane[:, 1] @ matrix @ plane[:, 0]) < 0:
            plane[:, 1] *= -1.0
        coordinates = states @ plane
        rows.append({
            "frequency": float(eigenvalues[index].imag),
            "fit_excited_energy_fraction": float(
                np.sum(np.square(coordinates)) / (total + 1e-8)
            ),
        })
    return rows


def _evaluate_conditional(vectors, shared, fit, dimension, *, np):
    from eqvae.evaluation.rotation_geometry import matrix_exponential

    basis = shared.basis[:, :dimension]
    angles = np.asarray(HELDOUT_ANGLES, dtype=np.int64)
    rows = []
    r2 = []
    horizons = []
    for rank in HELDOUT_RANKS:
        center = vectors[rank, np.asarray(FIT_ANGLES)].mean(axis=0)
        initial = vectors[rank, 0] - center
        projected = basis.T @ initial
        orthogonal = initial - basis @ projected
        prediction = np.stack([
            center
            + basis @ (matrix_exponential(fit.matrix, math.radians(angle)) @ projected)
            + orthogonal
            for angle in angles
        ])
        truth = vectors[rank, angles]
        error = prediction - truth
        centered = truth - truth.mean(axis=0, keepdims=True)
        sse = float(np.sum(np.square(error)))
        sst = float(np.sum(np.square(centered)))
        rows.append(math.sqrt(sse / error.size) / (math.sqrt(sst / truth.size) + 1e-8))
        r2.append(1.0 - sse / (sst + 1e-8))
        horizons.append(
            (
                np.sqrt(np.mean(np.square(error), axis=1))
                / (math.sqrt(sst / truth.size) + 1e-8)
            ).tolist()
        )
    return {
        "nrmse": rows,
        "r2": r2,
        "median_nrmse": float(np.median(rows)),
        "normalized_error_by_angle": horizons,
    }


def _evaluate_preceding_anchor(vectors, shared, fit, dimension, *, np):
    from eqvae.evaluation.rotation_geometry import matrix_exponential

    basis = shared.basis[:, :dimension]
    errors = []
    horizons = []
    for rank in HELDOUT_RANKS:
        center = vectors[rank, np.asarray(FIT_ANGLES)].mean(axis=0)
        patch_errors = []
        for angle in HELDOUT_ANGLES:
            anchor = angle - angle % 5
            initial = vectors[rank, anchor] - center
            projected = basis.T @ initial
            orthogonal = initial - basis @ projected
            prediction = (
                center
                + basis
                @ (
                    matrix_exponential(fit.matrix, math.radians(angle - anchor))
                    @ projected
                )
                + orthogonal
            )
            patch_errors.append(
                float(np.sqrt(np.mean(np.square(prediction - vectors[rank, angle]))))
            )
        scale = float(
            np.sqrt(np.mean(np.square(vectors[rank] - vectors[rank].mean(axis=0))))
        )
        errors.append(float(np.mean(patch_errors)) / (scale + 1e-8))
        horizons.append((np.asarray(patch_errors) / (scale + 1e-8)).tolist())
    return {
        "nrmse": errors,
        "median_nrmse": float(np.median(errors)),
        "normalized_error_by_angle": horizons,
    }


def _evaluate_patch_specific(vectors, shared, dimension, *, np):
    from eqvae.evaluation.rotation_geometry import fit_skew_generator

    rows = []
    for rank in HELDOUT_RANKS:
        patch_center = vectors[rank, np.asarray(FIT_ANGLES)].mean(axis=0)
        states = (vectors[rank, np.asarray(FIT_ANGLES)] - patch_center) @ shared.basis[
            :, :dimension
        ]
        derivatives = (np.roll(states, -1, axis=0) - np.roll(states, 1, axis=0)) / (
            2.0 * math.radians(5.0)
        )
        fit = fit_skew_generator(states, derivatives)
        single = _evaluate_single_rank(
            vectors,
            shared,
            fit,
            dimension,
            rank,
            center=patch_center,
            np=np,
        )
        rows.append(single)
    return {"nrmse": rows, "median_nrmse": float(np.median(rows))}


def _evaluate_single_rank(vectors, shared, fit, dimension, rank, *, center=None, np):
    from eqvae.evaluation.rotation_geometry import matrix_exponential

    basis = shared.basis[:, :dimension]
    if center is None:
        center = shared.global_center
    initial = vectors[rank, 0] - center
    projected = basis.T @ initial
    orthogonal = initial - basis @ projected
    angles = np.asarray(HELDOUT_ANGLES, dtype=np.int64)
    prediction = np.stack([
        center
        + basis @ (matrix_exponential(fit.matrix, math.radians(angle)) @ projected)
        + orthogonal
        for angle in angles
    ])
    truth = vectors[rank, angles]
    sse = float(np.sum(np.square(prediction - truth)))
    sst = float(np.sum(np.square(truth - truth.mean(axis=0, keepdims=True))))
    return math.sqrt(sse / truth.size) / (math.sqrt(sst / truth.size) + 1e-8)


def _lopo_generator(vectors, lopo_shared, dimension, *, np):
    from eqvae.evaluation.rotation_geometry import fit_generator_from_orbits

    errors = []
    frequencies = []
    for omitted in FIT_RANKS:
        ranks = tuple(rank for rank in FIT_RANKS if rank != omitted)
        shared = lopo_shared[omitted]
        fit = fit_generator_from_orbits(
            vectors,
            shared,
            dimension=dimension,
            conditional=False,
            fit_ranks=ranks,
        )
        errors.append(
            _evaluate_single_rank(vectors, shared, fit, dimension, omitted, np=np)
        )
        frequencies.append(fit.eigenfrequencies.tolist())
    return {
        "omitted_ranks": list(FIT_RANKS),
        "nrmse": errors,
        "median_nrmse": float(np.median(errors)),
        "standard_error": float(np.std(errors, ddof=1) / math.sqrt(len(errors))),
        "eigenfrequencies": frequencies,
    }


def _select_dimension(branch_results, *, np):
    rows = []
    for dimension in range(1, 7):
        normal = branch_results["normal_vae"]["shared_action_dimensions"][
            dimension - 1
        ]["leave_one_fit_patch_out"]
        so2 = branch_results["so2_vae"]["shared_action_dimensions"][dimension - 1][
            "leave_one_fit_patch_out"
        ]
        model_medians = np.asarray(
            [normal["median_nrmse"], so2["median_nrmse"]],
            dtype=np.float64,
        )
        rows.append({
            "dimension": dimension,
            "mean": float(np.mean(model_medians)),
            "standard_error": float(
                math.sqrt(normal["standard_error"] ** 2 + so2["standard_error"] ** 2)
                / 2.0
            ),
            "model_median_nrmse": model_medians.tolist(),
        })
    best = min(rows, key=itemgetter("mean"))
    threshold = best["mean"] + best["standard_error"]
    selected = min(row["dimension"] for row in rows if row["mean"] <= threshold)
    return {
        "dimension": selected,
        "fit_only_rows": rows,
        "one_standard_error_threshold": threshold,
    }


def _evaluate_decisions(
    branch_results,
    branch_states,
    input_summary,
    selected_dimension,
    f1_result,
    *,
    fit_generator_from_orbits,
    canonicalization_statistics,
    fit_angle_probe,
    score_angle_probe,
    np,
):
    dimension = selected_dimension["dimension"]
    action = {}
    learned_factorization = {}
    for branch in ("normal_vae", "so2_vae"):
        row = branch_results[branch]["shared_action_dimensions"][dimension - 1]
        strict = row["strict_global"]
        identity = row["identity"]
        shuffled = row["shuffled"]
        improvements = [
            s < i for s, i in zip(strict["nrmse"], identity["nrmse"], strict=True)
        ]
        lopo_stable = _frequency_stability(row, np=np)
        branch_pass = (
            strict["median_nrmse"] <= 0.9 * identity["median_nrmse"]
            and strict["median_nrmse"] <= 0.9 * shuffled["median_nrmse"]
            and strict["median_r2"] > 0.0
            and sum(improvements) >= 6
            and lopo_stable
        )
        action[branch] = {
            "passes": branch_pass,
            "favorable_vs_identity": sum(improvements),
            "frequency_stable": lopo_stable,
        }
        vectors = branch_states[branch]["vectors"]
        shared = branch_states[branch]["shared"]
        conditional = fit_generator_from_orbits(
            vectors,
            shared,
            dimension=dimension,
            conditional=True,
        )
        fit_canonical = _canonicalize_learned(
            vectors[np.asarray(FIT_RANKS)],
            shared.basis[:, :dimension],
            conditional.matrix,
            np=np,
        )
        heldout_raw = vectors[np.asarray(HELDOUT_RANKS)]
        heldout_canonical = _canonicalize_learned(
            heldout_raw,
            shared.basis[:, :dimension],
            conditional.matrix,
            np=np,
        )
        statistics = canonicalization_statistics(heldout_canonical, heldout_raw)
        identity_statistics = canonicalization_statistics(heldout_raw, heldout_raw)
        fit_raw_features = _centered_projected_features(
            vectors[np.asarray(FIT_RANKS)],
            shared.basis[:, :dimension],
            np=np,
        )
        heldout_raw_features = _centered_projected_features(
            heldout_raw,
            shared.basis[:, :dimension],
            np=np,
        )
        fit_can_features = _centered_projected_features(
            fit_canonical,
            shared.basis[:, :dimension],
            np=np,
        )
        heldout_can_features = _centered_projected_features(
            heldout_canonical,
            shared.basis[:, :dimension],
            np=np,
        )
        anchor = np.asarray(FIT_ANGLES, dtype=np.int64)
        withheld = np.asarray(HELDOUT_ANGLES, dtype=np.int64)
        train_angles = np.tile(anchor, len(FIT_RANKS)).tolist()
        test_angles = np.tile(withheld, len(HELDOUT_RANKS)).tolist()
        raw_probe = fit_angle_probe(
            fit_raw_features[:, anchor].reshape(-1, dimension), train_angles
        )
        raw_score = score_angle_probe(
            heldout_raw_features[:, withheld].reshape(-1, dimension),
            test_angles,
            raw_probe,
        )
        canonical_probe = fit_angle_probe(
            fit_can_features[:, anchor].reshape(-1, dimension),
            train_angles,
        )
        canonical_score = score_angle_probe(
            heldout_can_features[:, withheld].reshape(-1, dimension),
            test_angles,
            canonical_probe,
        )
        prereq = (
            branch_pass
            and statistics["identity_w"] >= 1e-6
            and statistics["identity_b"] >= 1e-6
            and raw_score["mean_cosine_alignment"] >= 0.50
        )
        factor_pass = (
            prereq
            and statistics["patch_w_ratio_median"] <= 0.25
            and 0.8 <= statistics["b_ratio"] <= 1.2
            and statistics["retrieval_mean"] >= 0.9
            and statistics["distance_stress_defined"]
            and identity_statistics["distance_stress_defined"]
            and statistics["distance_stress_median"]
            <= 1.1 * identity_statistics["distance_stress_median"]
            and canonical_score["mean_cosine_alignment"]
            <= 0.5 * raw_score["mean_cosine_alignment"]
        )
        content_gate_failed = (
            statistics["patch_w_ratio_median"] > 0.25
            or not 0.8 <= statistics["b_ratio"] <= 1.2
            or statistics["retrieval_mean"] < 0.9
            or not statistics["distance_stress_defined"]
            or not identity_statistics["distance_stress_defined"]
            or (
                statistics["distance_stress_defined"]
                and identity_statistics["distance_stress_defined"]
                and statistics["distance_stress_median"]
                > 1.1 * identity_statistics["distance_stress_median"]
            )
        )
        learned_factorization[branch] = {
            "statistics": statistics,
            "identity_statistics": identity_statistics,
            "raw_angle_probe": raw_score,
            "canonical_angle_probe": canonical_score,
            "nondegenerate_prerequisites": prereq,
            "passes": factor_pass,
            "status": (
                "supported"
                if factor_pass
                else "not_supported"
                if prereq and content_gate_failed
                else "unresolved"
            ),
        }
    normal_nrmse = branch_results["normal_vae"]["shared_action_dimensions"][
        dimension - 1
    ]["strict_global"]["nrmse"]
    so2_nrmse = branch_results["so2_vae"]["shared_action_dimensions"][dimension - 1][
        "strict_global"
    ]["nrmse"]
    so2_specific = (
        action["so2_vae"]["passes"]
        and float(np.median(so2_nrmse)) <= 0.9 * float(np.median(normal_nrmse))
        and sum(s < n for s, n in zip(so2_nrmse, normal_nrmse, strict=True)) >= 6
    )
    known = {}
    for branch in ("normal_vae", "so2_vae"):
        values = branch_results[branch]["spatial_action"]["dense_residual_per_patch"]
        median = float(np.median(values))
        known[branch] = {
            "median": median,
            "verified": median <= 0.25 and sum(value <= 0.25 for value in values) >= 20,
            "poor": median > 0.50,
        }
    return {
        "shared_action": action | {"so2_specific_advantage": so2_specific},
        "learned_factorization": learned_factorization,
        "spatial_action": known,
        "f1_h3_passes": f1_result["clean_copy_count"] >= 24,
    }


def _frequency_stability(row, *, np):
    reference = np.asarray(
        row["strict_generator"]["eigenfrequencies"], dtype=np.float64
    )
    if reference.size == 0:
        return row["dimension"] == 1
    for frequencies in row["leave_one_fit_patch_out"]["eigenfrequencies"]:
        observed = np.asarray(frequencies, dtype=np.float64)
        if observed.size != reference.size:
            return False
        relative = np.abs(observed - reference) / np.maximum(reference, 1e-8)
        if np.max(relative) > 0.20:
            return False
    return True


def _canonicalize_learned(vectors, basis, matrix, *, np):
    from eqvae.evaluation.rotation_geometry import matrix_exponential

    centers = vectors[:, np.asarray(FIT_ANGLES)].mean(axis=1)
    output = np.empty_like(vectors)
    for patch in range(vectors.shape[0]):
        residual = vectors[patch] - centers[patch]
        projected = residual @ basis
        orthogonal = residual - projected @ basis.T
        for angle in range(360):
            inverse = matrix_exponential(matrix, -math.radians(angle))
            output[patch, angle] = (
                centers[patch]
                + projected[angle] @ inverse.T @ basis.T
                + orthogonal[angle]
            )
    return output


def _centered_projected_features(vectors, basis, *, np):
    centers = vectors[:, np.asarray(FIT_ANGLES)].mean(axis=1, keepdims=True)
    return (vectors - centers) @ basis


def _population_comparisons(
    branch_results, input_summary, *, paired_bootstrap_median_difference, np
):
    metrics = (
        "local_linearity_ratio",
        "step_size_cv",
        "path_length",
        "curvature_median",
        "curvature_q90",
    )

    def summarize_metric(step, metric, *, exclude_cardinals):
        collection = "cardinal_excluded" if exclude_cardinals else "geometry"
        input_rows = input_summary["variants"]["disk"]
        if exclude_cardinals:
            input_values = np.asarray([
                row["cardinal_excluded"][str(step)][metric] for row in input_rows
            ])
            input_steps = np.asarray([
                row["cardinal_excluded"][str(step)]["step_rms"] for row in input_rows
            ])
        else:
            input_values = np.asarray([row[str(step)][metric] for row in input_rows])
            input_steps = np.asarray([row[str(step)]["step_rms"] for row in input_rows])
        normal_source = branch_results["normal_vae"][collection][str(step)]
        so2_source = branch_results["so2_vae"][collection][str(step)]
        normal_raw = np.asarray(normal_source[metric])
        so2_raw = np.asarray(so2_source[metric])
        normal_speed_gain = None
        so2_speed_gain = None
        if metric == "step_size_cv":
            normal_steps = np.asarray(normal_source["step_rms"])
            so2_steps = np.asarray(so2_source["step_rms"])
            normal_speed_gain = normal_steps / (input_steps + 1e-8)
            so2_speed_gain = so2_steps / (input_steps + 1e-8)
            normal = np.std(normal_speed_gain, axis=1) / (
                np.mean(normal_speed_gain, axis=1) + 1e-8
            )
            so2 = np.std(so2_speed_gain, axis=1) / (
                np.mean(so2_speed_gain, axis=1) + 1e-8
            )
        else:
            normal = normal_raw / (input_values + 1e-8)
            so2 = so2_raw / (input_values + 1e-8)
        summary = {
            "step_degrees": step,
            "cardinal_neighborhoods_excluded": exclude_cardinals,
            "input_normalization": (
                "CV across pointwise latent/input one-step RMS gains"
                if metric == "step_size_cv"
                else "per-patch latent metric divided by input metric"
            ),
            "normal": normal.tolist(),
            "so2": so2.tolist(),
            "normal_raw": normal_raw.tolist(),
            "so2_raw": so2_raw.tolist(),
            "input": input_values.tolist(),
            "input_median": float(np.median(input_values)),
            "normal_median": float(np.median(normal)),
            "so2_median": float(np.median(so2)),
            "so2_lower_count": int(np.sum(so2 < normal)),
            "paired_bootstrap_so2_minus_normal": paired_bootstrap_median_difference(
                so2, normal
            ),
            "normal_raw_median": float(np.median(normal_raw)),
            "so2_raw_median": float(np.median(so2_raw)),
            "so2_lower_raw_count": int(np.sum(so2_raw < normal_raw)),
            "paired_bootstrap_raw_so2_minus_normal": (
                paired_bootstrap_median_difference(so2_raw, normal_raw)
            ),
        }
        if normal_speed_gain is not None:
            summary["normal_pointwise_speed_gain"] = normal_speed_gain.tolist()
            summary["so2_pointwise_speed_gain"] = so2_speed_gain.tolist()
        return summary

    output = {
        "subsampling": {
            str(step): {
                metric: summarize_metric(
                    step,
                    metric,
                    exclude_cardinals=False,
                )
                for metric in metrics
            }
            for step in (1, 2, 5)
        },
        "cardinal_excluded": {
            str(step): {
                metric: summarize_metric(
                    step,
                    metric,
                    exclude_cardinals=True,
                )
                for metric in metrics
            }
            for step in (1, 2, 5)
        },
    }
    for metric in metrics:
        output[metric] = output["subsampling"]["1"][metric]
    normal_harmonic = np.asarray([
        row["harmonics"]["low_harmonic_fraction_raw"]
        for row in branch_results["normal_vae"]["local_pca"]
    ])
    so2_harmonic = np.asarray([
        row["harmonics"]["low_harmonic_fraction_raw"]
        for row in branch_results["so2_vae"]["local_pca"]
    ])
    normal_effective = np.asarray([
        row["harmonics"]["effective_frequency_count"]
        for row in branch_results["normal_vae"]["local_pca"]
    ])
    so2_effective = np.asarray([
        row["harmonics"]["effective_frequency_count"]
        for row in branch_results["so2_vae"]["local_pca"]
    ])
    output["harmonics"] = {
        "normal_low_fraction": normal_harmonic.tolist(),
        "so2_low_fraction": so2_harmonic.tolist(),
        "so2_higher_count": int(np.sum(so2_harmonic > normal_harmonic)),
        "normal_effective_count": normal_effective.tolist(),
        "so2_effective_count": so2_effective.tolist(),
        "so2_lower_effective_count": int(np.sum(so2_effective < normal_effective)),
    }
    ll = output["local_linearity_ratio"]
    output["h1_passes"] = (
        ll["so2_median"] <= 0.9 * ll["normal_median"] and ll["so2_lower_count"] >= 18
    )
    output["h2_passes"] = (
        float(np.median(so2_harmonic)) >= 1.1 * float(np.median(normal_harmonic))
        and int(np.sum(so2_harmonic > normal_harmonic)) >= 18
        and float(np.median(so2_effective)) <= 0.9 * float(np.median(normal_effective))
        and int(np.sum(so2_effective < normal_effective)) >= 18
    )
    return output


def _bundle_diagnostics(
    state,
    result,
    input_summary,
    selected_dimension,
    f1_result,
    *,
    np,
):
    from eqvae.evaluation.rotation_geometry import matrix_exponential

    dimension = selected_dimension["dimension"]
    row = result["shared_action_dimensions"][dimension - 1]
    generator = np.asarray(row["strict_generator"]["matrix"], dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eig(generator)
    positive = np.flatnonzero(eigenvalues.imag > 1e-8)
    m1 = [index for index in positive if 0.8 <= eigenvalues[index].imag <= 1.2]
    other = [index for index in positive if index not in m1]
    separated = all(
        abs(eigenvalues[left].imag - eigenvalues[right].imag) >= 0.25
        for left in m1
        for right in other
    )
    plane = None
    energy_fraction = 0.0
    heldout_rows = []
    if m1 and separated:
        columns = []
        for index in m1:
            vector = eigenvectors[:, index]
            columns.extend((vector.real, -vector.imag))
        plane, _ = np.linalg.qr(np.column_stack(columns))
        plane = plane[:, : 2 * len(m1)]
        restricted = plane.T @ generator @ plane
        vectors = state["vectors"]
        shared = state["shared"]
        patch_centers = vectors[:, np.asarray(FIT_ANGLES)].mean(axis=1, keepdims=True)
        scores = (vectors - patch_centers) @ shared.basis[:, :dimension]
        coordinates = scores @ plane
        fit_coordinates = coordinates[np.asarray(FIT_RANKS)]
        fit_centered = fit_coordinates - fit_coordinates.mean(axis=1, keepdims=True)
        coefficients = np.fft.rfft(fit_centered, axis=1) / 360.0
        power = np.square(np.abs(coefficients))
        power[:, 1:-1] *= 2.0
        energy_fraction = float(np.sum(power[:, 1]) / (np.sum(power[:, 1:]) + 1e-8))
        if energy_fraction >= 0.50:
            anchors = np.asarray(FIT_ANGLES, dtype=np.int64)
            withheld = np.asarray(HELDOUT_ANGLES, dtype=np.int64)
            for rank in HELDOUT_RANKS:
                anchor_templates = np.stack([
                    matrix_exponential(restricted, -math.radians(int(angle)))
                    @ coordinates[rank, angle]
                    for angle in anchors
                ])
                template = anchor_templates.mean(axis=0)
                predicted = np.stack([
                    matrix_exponential(restricted, math.radians(int(angle))) @ template
                    for angle in withheld
                ])
                observed = coordinates[rank, withheld]
                scale = float(np.sqrt(np.mean(np.square(observed))))
                residual = float(np.sqrt(np.mean(np.square(predicted - observed)))) / (
                    scale + 1e-8
                )
                phase_rmse = None
                phase_origin = None
                if len(m1) == 1:
                    predicted_phase = np.arctan2(predicted[:, 1], predicted[:, 0])
                    observed_phase = np.arctan2(observed[:, 1], observed[:, 0])
                    phase_error = np.arctan2(
                        np.sin(observed_phase - predicted_phase),
                        np.cos(observed_phase - predicted_phase),
                    )
                    phase_rmse = float(np.sqrt(np.mean(np.square(phase_error))))
                    phase_origin = float(math.atan2(template[1], template[0]))
                heldout_rows.append({
                    "rank": rank,
                    "anchor_template_amplitude": float(np.linalg.norm(template)),
                    "withheld_action_nrmse": residual,
                    "withheld_phase_rmse_radians": phase_rmse,
                    "local_phase_origin_radians": phase_origin,
                })
    eligible = bool(len(m1) == 1 and separated and energy_fraction >= 0.50)
    orientation = np.asarray([
        row["orientation_strength"] for row in input_summary["stabilizers"]
    ])[np.asarray(HELDOUT_RANKS)]
    c2 = np.asarray([row["c2_strength"] for row in input_summary["stabilizers"]])[
        np.asarray(HELDOUT_RANKS)
    ]
    heldout_phase_error = np.asarray([
        row["withheld_phase_rmse_radians"]
        for row in heldout_rows
        if row["withheld_phase_rmse_radians"] is not None
    ])
    phase_association = None
    if heldout_phase_error.size == len(HELDOUT_RANKS):
        rho_orientation = _spearman(heldout_phase_error, orientation, np=np)
        rho_c2 = _spearman(heldout_phase_error, c2, np=np)
        phase_association = {
            "spearman_phase_error_vs_orientation": rho_orientation,
            "spearman_phase_error_vs_c2": rho_c2,
        }
        phase_association["failure_concentrated_in_symmetric_patches"] = bool(
            (rho_orientation is not None and rho_orientation <= -0.5)
            or (rho_c2 is not None and rho_c2 >= 0.5)
        )
    all_orientation = np.asarray([
        row["orientation_strength"] for row in input_summary["stabilizers"]
    ])
    all_c2 = np.asarray([row["c2_strength"] for row in input_summary["stabilizers"]])
    f1_phase = np.asarray(f1_result["patch_median_phase_rmse"])
    return {
        "eligible_m1_subspace": eligible,
        "m1_plane_count": len(m1),
        "mode": "isolated" if len(m1) == 1 else "repeated" if m1 else "absent",
        "separated_from_other_frequencies": bool(separated),
        "m1_subspace_energy_fraction": energy_fraction,
        "heldout_scores": heldout_rows,
        "heldout_orientation_strength": orientation.tolist(),
        "heldout_c2_strength": c2.tolist(),
        "heldout_phase_association": phase_association,
        "all25_f1_phase_association": {
            "spearman_phase_error_vs_orientation": _spearman(
                f1_phase,
                all_orientation,
                np=np,
            ),
            "spearman_phase_error_vs_c2": _spearman(f1_phase, all_c2, np=np),
        },
        "interpretation": (
            "eligible isolated m=1 plane; withheld phase scored"
            if eligible
            else "descriptive only; no eligible isolated transferable m=1 plane"
        ),
        "topology_claim_allowed": False,
        "global_product_claim_allowed": False,
    }


def _spearman(left, right, *, np):
    def ranks(values):
        order = np.argsort(values, kind="mergesort")
        output = np.empty(values.size, dtype=np.float64)
        start = 0
        while start < values.size:
            stop = start + 1
            while stop < values.size and values[order[stop]] == values[order[start]]:
                stop += 1
            output[order[start:stop]] = 0.5 * (start + stop - 1)
            start = stop
        return output

    left_rank = ranks(np.asarray(left, dtype=np.float64))
    right_rank = ranks(np.asarray(right, dtype=np.float64))
    if np.std(left_rank) <= 1e-12 or np.std(right_rank) <= 1e-12:
        return None
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _shared_scores(state, *, np):
    vectors = state["vectors"]
    shared = state["shared"]
    centers = vectors[:, np.asarray(FIT_ANGLES)].mean(axis=1, keepdims=True)
    return ((vectors - centers) @ shared.basis).astype(np.float32)


def _render_figures(
    branch_results,
    comparisons,
    decisions,
    f1_result,
    selected_dimension,
    *,
    plt,
    np,
):
    _figure_all25(branch_results, plt=plt, np=np)
    _figure_population(comparisons, plt=plt, np=np)
    _figure_examples(branch_results, plt=plt, np=np)
    _figure_pairwise(branch_results, plt=plt, np=np)
    _figure_harmonics(branch_results, plt=plt, np=np)
    _figure_f1(f1_result, plt=plt, np=np)
    _figure_generators(branch_results, selected_dimension, plt=plt, np=np)
    _figure_factorization(decisions, plt=plt, np=np)


def _figure_all25(results, *, plt, np):
    fig, axes = plt.subplots(5, 10, figsize=(24, 13), constrained_layout=True)
    for rank in range(25):
        row = rank // 5
        pair = rank % 5
        for offset, (branch, color, label) in enumerate((
            ("normal_vae", "#3a86ff", "Normal"),
            ("so2_vae", "#e63946", "SO(2)"),
        )):
            axis = axes[row, pair * 2 + offset]
            scores = np.asarray(results[branch]["local_pca"][rank]["scores"])
            axis.plot(
                scores[:, 0], scores[:, 1], color=color, lw=0.8, alpha=0.85, label=label
            )
            axis.scatter(
                scores[[0, 90, 180, 270], 0],
                scores[[0, 90, 180, 270], 1],
                color=color,
                s=8,
            )
            axis.set_title(f"Rango {rank} · {label}", fontsize=7)
            axis.set_aspect("equal", adjustable="datalim")
            axis.set_xticks([])
            axis.set_yticks([])
    fig.suptitle(
        "Órbitas corregidas 0°–359° · cada panel usa su propio PCA local",
        fontsize=16,
    )
    _save_figure(fig, FIGURE_ROOT / "01-corrected-all25-orbits.png", plt=plt)


def _figure_population(comparisons, *, plt, np):
    fig, axes = plt.subplots(1, 4, figsize=(19, 5), constrained_layout=True)
    for axis, metric, title in zip(
        axes[:3],
        ("local_linearity_ratio", "step_size_cv", "path_length"),
        ("Linealidad / entrada", "CV del paso / entrada", "Longitud / entrada"),
        strict=True,
    ):
        normal = np.asarray(comparisons[metric]["normal"])
        so2 = np.asarray(comparisons[metric]["so2"])
        axis.scatter(normal, so2, s=24, color="#6a4c93")
        bounds = [min(normal.min(), so2.min()), max(normal.max(), so2.max())]
        axis.plot(bounds, bounds, "k--", lw=0.8)
        axis.set(xlabel="Normal", ylabel="SO(2)", title=title)
        axis.set_aspect("equal", adjustable="box")
    input_cv = np.asarray(comparisons["step_size_cv"]["input"])
    axes[3].hist(input_cv, bins=10, color="#2a9d8f")
    axes[3].set(
        xlabel="CV del paso de entrada",
        ylabel="Parches",
        title="Control de interpolación (disco)",
    )
    fig.suptitle("Métricas poblacionales corregidas · 25 pares", fontsize=15)
    _save_figure(fig, FIGURE_ROOT / "02-population-input-controls.png", plt=plt)


def _figure_examples(results, *, plt, np):
    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    for row, rank in enumerate((0, 12)):
        for column, branch in enumerate(("normal_vae", "so2_vae")):
            axis = fig.add_subplot(2, 2, row * 2 + column + 1, projection="3d")
            scores = np.asarray(results[branch]["local_pca"][rank]["scores"])
            colors = np.arange(360)
            axis.scatter(
                scores[:, 0], scores[:, 1], scores[:, 2], c=colors, cmap="twilight", s=5
            )
            axis.set_proj_type("ortho")
            axis.view_init(elev=25, azim=-60)
            axis.set_box_aspect((1, 1, 1))
            axis.set_title(
                f"Rango {rank} · {'Normal' if branch == 'normal_vae' else 'SO(2)'}"
            )
            axis.set_xlabel("PC1")
            axis.set_ylabel("PC2")
            axis.set_zlabel("PC3")
    fig.suptitle("Geometría PCA local corregida", fontsize=15)
    _save_figure(fig, FIGURE_ROOT / "03-local-pca-ranks-00-12.png", plt=plt)


def _figure_pairwise(results, *, plt, np):
    planes = [(left, right) for left in range(6) for right in range(left + 1, 6)]
    fig, axes = plt.subplots(4, 15, figsize=(30, 9), constrained_layout=True)
    for row, (rank, branch) in enumerate((
        (0, "normal_vae"),
        (0, "so2_vae"),
        (12, "normal_vae"),
        (12, "so2_vae"),
    )):
        scores = np.asarray(results[branch]["local_pca"][rank]["scores"])
        for column, (left, right) in enumerate(planes):
            target = axes[row, column]
            target.scatter(
                scores[:, left],
                scores[:, right],
                c=np.arange(360),
                cmap="twilight",
                s=2,
            )
            target.scatter(
                scores[[0, 90, 180, 270], left],
                scores[[0, 90, 180, 270], right],
                color="black",
                s=5,
            )
            target.set_title(f"PC{left + 1}–PC{right + 1}", fontsize=6)
            target.set_aspect("equal", adjustable="datalim")
            target.set_xticks([])
            target.set_yticks([])
    fig.suptitle("Planos PC1–PC6 · rangos 0 y 12", fontsize=15)
    _save_figure(fig, FIGURE_ROOT / "03b-local-pca-pc1-pc6.png", plt=plt)


def _figure_harmonics(results, *, plt, np):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for row, rank in enumerate((0, 12)):
        for column, branch in enumerate(("normal_vae", "so2_vae")):
            scores = np.asarray(results[branch]["local_pca"][rank]["scores"])
            coefficient = np.fft.rfft(scores - scores.mean(axis=0), axis=0) / 360.0
            power = np.square(np.abs(coefficient[:7])).T
            power[:, 1:] *= 2.0
            image = axes[row, column].imshow(power, aspect="auto", cmap="magma")
            axes[row, column].set_title(
                f"Rango {rank} · {'Normal' if branch == 'normal_vae' else 'SO(2)'}"
            )
            axes[row, column].set_xlabel("Armónico m")
            axes[row, column].set_ylabel("PC")
            axes[row, column].set_xticks(range(7))
            axes[row, column].set_yticks(range(6), labels=range(1, 7))
            fig.colorbar(image, ax=axes[row, column], shrink=0.8)
    fig.suptitle("Energía armónica PC1–PC6", fontsize=15)
    _save_figure(fig, FIGURE_ROOT / "04-pc1-pc6-harmonics.png", plt=plt)


def _figure_f1(summary, *, plt, np):
    rows = summary["copies"]
    copies = np.arange(48)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    for axis, key, title in (
        (axes[0, 0], "m1_purity_median", "Pureza |m|=1"),
        (axes[0, 1], "phase_slope_median", "Pendiente de fase"),
        (axes[1, 0], "amplitude_cv_median", "CV de amplitud"),
        (axes[1, 1], "full_field_residual_median", "Residual de campo completo"),
    ):
        axis.bar(copies, [row[key] for row in rows], color="#7b2cbf")
        axis.set_title(title)
        axis.set_xlabel("Copia F1")
    fig.suptitle(
        f"48 copias F1 · limpias {summary['clean_copy_count']}/48", fontsize=15
    )
    _save_figure(fig, FIGURE_ROOT / "05-all48-f1-summary.png", plt=plt)


def _figure_generators(results, selected, *, plt, np):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    dimensions = np.arange(1, 7)
    for branch, color, label in (
        ("normal_vae", "#3a86ff", "Normal"),
        ("so2_vae", "#e63946", "SO(2)"),
    ):
        medians = [
            row["strict_global"]["median_nrmse"]
            for row in results[branch]["shared_action_dimensions"]
        ]
        axes[0, 0].plot(dimensions, medians, marker="o", color=color, label=label)
    axes[0, 0].axvline(
        selected["dimension"],
        color="black",
        ls="--",
        label=f"d*={selected['dimension']}",
    )
    axes[0, 0].set(
        xlabel="Dimensión compartida",
        ylabel="NRMSE mediana, rollout desde 0°",
        xticks=dimensions,
    )
    axes[0, 0].legend()
    axes[0, 0].set_title("Transferencia estricta frente a dimensión")
    selected_index = selected["dimension"] - 1
    withheld = np.asarray(HELDOUT_ANGLES)
    for axis, branch, title in (
        (axes[0, 1], "normal_vae", "Normal"),
        (axes[1, 0], "so2_vae", "SO(2)"),
    ):
        row = results[branch]["shared_action_dimensions"][selected_index]
        for key, label, color in (
            ("strict_global", "estricto: una vista", "#264653"),
            ("coarse_orbit_conditioned", "centro multi-vista", "#e76f51"),
            ("preceding_anchor", "ancla previa", "#2a9d8f"),
        ):
            horizons = np.asarray(row[key]["normalized_error_by_angle"])
            axis.plot(withheld, np.median(horizons, axis=0), label=label, color=color)
        axis.set(
            xlabel="Ángulo retenido (grados)",
            ylabel="Error RMS normalizado mediano",
            title=f"Horizonte de error · {title}",
        )
        axis.legend(fontsize=8)
    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.05,
        0.85,
        "Solo el rollout estricto desde 0°\npuede apoyar una acción compartida.\n"
        "Los otros trazos usan información\nangular del parche retenido.",
        va="top",
        fontsize=12,
    )
    fig.suptitle("Acción compartida · 8 parches retenidos", fontsize=15)
    _save_figure(fig, FIGURE_ROOT / "06-shared-generator-vs-dimension.png", plt=plt)


def _figure_factorization(decisions, *, plt, np):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    labels = ["Normal", "SO(2)"]
    rows = [
        decisions["learned_factorization"][branch]
        for branch in ("normal_vae", "so2_vae")
    ]
    axes[0].bar(
        labels,
        [row["statistics"]["patch_w_ratio_median"] for row in rows],
        color=["#3a86ff", "#e63946"],
    )
    axes[0].axhline(0.25, color="black", ls="--")
    axes[0].set_title("Mediana por parche: W / W identidad")
    axes[1].bar(
        labels,
        [row["statistics"]["b_ratio"] for row in rows],
        color=["#3a86ff", "#e63946"],
    )
    axes[1].axhspan(0.8, 1.2, color="grey", alpha=0.2)
    axes[1].set_title("B / B identidad")
    fig.suptitle("Canonicalización aprendida · evaluación retenida", fontsize=15)
    _save_figure(fig, FIGURE_ROOT / "07-canonicalization-factorization.png", plt=plt)


def _save_figure(fig, path, *, plt):
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def _require_hash(path: Path, expected: str) -> None:
    observed = _sha256(path)
    if observed != expected:
        raise RuntimeError(f"SHA-256 mismatch for {path}: {observed} != {expected}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
