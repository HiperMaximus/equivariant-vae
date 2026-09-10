# Copyright 2026 HiperMaximus
# ruff: noqa: ANN001, ANN202, ARG001, BLE001, C901, COM812, D103, E501, EM101, EM102, FBT003, INP001, PLC0415, PLR0912, PLR0913, PLR0914, PLR0915, PLR1702, PLR2004, TRY003
"""Private T4 wrapper for the locked Spec 0051 decoded-transform audit."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import shutil
import sys
import time
import traceback
import zipfile
from pathlib import Path

# fmt: off
KAGGLE_DECODED_LATENT_TRANSFORM_READY = True
EMBEDDED_PAYLOAD_B64 = """
$embedded_payload_b64
"""
EMBEDDED_PAYLOAD_ZIP_SHA256 = "$embedded_payload_zip_sha256"
EMBEDDED_PAYLOAD_MANIFEST_SHA256 = "$embedded_payload_manifest_sha256"
# fmt: on

INPUT_ROOT = Path("/kaggle/input")
WORKING_ROOT = Path("/kaggle/working")
PRIVATE_ROOT = WORKING_ROOT / ".spec0051_payload"
OUTPUT_ROOT = WORKING_ROOT / "decoded_latent_transform_v1"
FIGURE_ROOT = OUTPUT_ROOT / "figures"
PATCH_BYTES = 3 * 256 * 256
HEADER_BYTES = 64
CONTRACT_SHA256 = "905ef933a51c26bb3f01fcef4bb4985fa98e00f1dbecd814da3df488bd7c1018"
SPEC_SHA256 = "7a472a2de56813544506b4a9eb58e2f1f152fcee8f7d60e65ed2c2793b349679"
MODEL_KINDS = {
    "normal_vae": "non_eq_vae_translatable",
    "so2_vae": "so2_vae_fixed",
}
ANGLES = tuple(range(0, 360, 5))
EXAMPLE_RANKS = (0, 12)
EXAMPLE_ANGLES = (0, 45, 90, 135, 180, 225, 270, 315)
CHUNK_SIZE = 5


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

    from eqvae.artifacts.rotation_orbits import centered_disk_mask, continuous_rotate
    from eqvae.evaluation.decoded_transform import (
        EXACT_D4_NAMES,
        EXACT_D4_NONIDENTITY_NAMES,
        aggregate_rms_ratio,
        exact_spatial_transform,
        gradient_mse_per_image,
        inverse_exact_transform_name,
        masked_mae_per_image,
        masked_mse_per_image,
        out_of_range_per_image,
        paired_bootstrap_median_difference,
    )
    from eqvae.evaluation.vae_test import sha256_file, state_dict_sha256
    from eqvae.metrics.reconstruction import (
        normalized_to_image_domain,
        ssim_per_image,
    )
    from eqvae.models.registry import build_model

    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA GPU is required")
    device_name = torch.cuda.get_device_name(0)
    if "T4" not in device_name:
        raise RuntimeError(f"a Tesla T4 is required, found {device_name}")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if OUTPUT_ROOT.exists():
        raise RuntimeError("output root already exists")
    contract_path = payload_root / "docs/data/spec0051_decoded_transform_contract.json"
    spec_path = payload_root / "docs/specs/0051-decoded-latent-transform-consistency.md"
    _require_hash(contract_path, CONTRACT_SHA256)
    _require_hash(spec_path, SPEC_SHA256)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if tuple(contract["dense_rotations"]["angles_degrees"]) != ANGLES:
        raise RuntimeError("angle contract differs")
    if tuple(contract["examples"]["ranks"]) != EXAMPLE_RANKS:
        raise RuntimeError("example-rank contract differs")
    if tuple(contract["examples"]["angles_degrees"]) != EXAMPLE_ANGLES:
        raise RuntimeError("example-angle contract differs")
    if tuple(contract["exact_d4"]["elements"]) != EXACT_D4_NAMES:
        raise RuntimeError("exact D4 element order differs")
    if tuple(contract["exact_d4"]["metric_elements"]) != EXACT_D4_NONIDENTITY_NAMES:
        raise RuntimeError("exact D4 metric-element order differs")
    selector_path = (
        payload_root / "runs/kaggle/fixed25_selector/fixed_25_validation_patches.json"
    )
    _require_hash(selector_path, contract["inputs"]["selector_sha256"])
    selector = json.loads(selector_path.read_text(encoding="utf-8"))
    if len({str(row["wsi_id"]) for row in selector["selectors"]}) != 16:
        raise RuntimeError("fixed25 WSI-cluster count differs")
    device = torch.device("cuda:0")
    patches = _load_fixed25(selector, np=np, torch=torch).to(device)
    bundle_root, weight_contract = _find_weight_bundle(
        contract["inputs"]["weight_bundle_contract_sha256"]
    )
    models, model_provenance = _load_models(
        bundle_root,
        weight_contract,
        contract,
        build_model=build_model,
        sha256_file=sha256_file,
        state_dict_sha256=state_dict_sha256,
        device=device,
        torch=torch,
    )
    image_mask = centered_disk_mask(256, radius=112.0, device=device)
    gradient_mask = centered_disk_mask(256, radius=110.0, device=device)
    latent_mask = centered_disk_mask(32, radius=14.0, device=device)

    branches = {}
    selected_images = {}
    dense_arrays = {}
    for branch in ("normal_vae", "so2_vae"):
        dense, images, bases = _evaluate_dense(
            models[branch],
            patches,
            image_mask,
            gradient_mask,
            latent_mask,
            continuous_rotate=continuous_rotate,
            masked_mse_per_image=masked_mse_per_image,
            masked_mae_per_image=masked_mae_per_image,
            gradient_mse_per_image=gradient_mse_per_image,
            out_of_range_per_image=out_of_range_per_image,
            normalized_to_image_domain=normalized_to_image_domain,
            ssim_per_image=ssim_per_image,
            np=np,
            torch=torch,
        )
        exact = _evaluate_exact_d4(
            models[branch],
            patches,
            bases,
            image_mask,
            latent_mask,
            exact_names=EXACT_D4_NONIDENTITY_NAMES,
            exact_spatial_transform=exact_spatial_transform,
            inverse_exact_transform_name=inverse_exact_transform_name,
            masked_mse_per_image=masked_mse_per_image,
            normalized_to_image_domain=normalized_to_image_domain,
            ssim_per_image=ssim_per_image,
            np=np,
            torch=torch,
        )
        branches[branch] = _summarize_branch(
            dense,
            exact,
            aggregate_rms_ratio=aggregate_rms_ratio,
            np=np,
        )
        selected_images[branch] = images
        for name, values in dense.items():
            dense_arrays[f"{branch}_{name}"] = values
        torch.cuda.empty_cache()

    comparisons = _comparisons(
        branches,
        cluster_ids=tuple(str(row["wsi_id"]) for row in selector["selectors"]),
        paired_bootstrap_median_difference=paired_bootstrap_median_difference,
        np=np,
    )
    decisions = _decisions(branches, np=np)
    algebra = _exact_algebra_checks(
        patches,
        exact_names=EXACT_D4_NAMES,
        exact_spatial_transform=exact_spatial_transform,
        torch=torch,
    )
    if not all(algebra.values()):
        raise RuntimeError(f"exact D4 algebra failed: {algebra}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=False)
    arrays_path = OUTPUT_ROOT / "metric_arrays.npz"
    np.savez_compressed(arrays_path, angles=np.asarray(ANGLES), **dense_arrays)
    selected_path = OUTPUT_ROOT / "selected_images_uint8.npz"
    np.savez_compressed(
        selected_path,
        **{
            f"{branch}_{key}": value
            for branch, images in selected_images.items()
            for key, value in images.items()
        },
    )
    _render_population(branches, plt=plt, np=np)
    _render_angle_series(branches, plt=plt, np=np)
    for rank in EXAMPLE_RANKS:
        _render_examples(rank, selected_images, kind="action", plt=plt, np=np)
        _render_examples(rank, selected_images, kind="canonical", plt=plt, np=np)
    _render_exact(branches, plt=plt, np=np)
    _render_latent_decoded(branches, plt=plt, np=np)
    figure_hashes = {
        str(path.relative_to(OUTPUT_ROOT)): _sha256(path)
        for path in sorted(FIGURE_ROOT.glob("*.png"))
    }
    result = {
        "schema": "spec0051.decoded_latent_transform.v1",
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
            "device": device_name,
            "deterministic_algorithms": True,
            "tf32": False,
            "fp32": True,
        },
        "exact_d4_algebra": algebra,
        "branches": branches,
        "comparisons": comparisons,
        "decisions": decisions,
        "geometric_interpretation": {
            "architecture_guarantees": "SO2 candidate spatial action on final F0 fields; no reflection guarantee",
            "empirical_scope": "decoded outputs on the fixed validation 25",
            "trained_decoder_behavior_if_h3_passes": "relative decoded suppression of finite residual latent differences; no identified null space or causal attribution",
            "patch_transfer": "same parameter-free action applied to every patch",
            "bundle_limit": "decoded canonicalization alone does not establish a latent section or factorization",
            "speculative": "population generalization, causal attribution, seed robustness, topology and global product",
        },
        "output_hashes": figure_hashes
        | {
            "metric_arrays.npz": _sha256(arrays_path),
            "selected_images_uint8.npz": _sha256(selected_path),
        },
    }
    summary_path = OUTPUT_ROOT / "decoded_latent_transform_summary.json"
    summary_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    manifest = {
        "schema": "spec0051.decoded_latent_transform_manifest.v1",
        "summary_sha256": _sha256(summary_path),
        "files": {
            str(path.relative_to(OUTPUT_ROOT)): _sha256(path)
            for path in sorted(OUTPUT_ROOT.rglob("*"))
            if path.is_file()
        },
    }
    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return 0


def _evaluate_dense(
    model,
    patches,
    image_mask,
    gradient_mask,
    latent_mask,
    *,
    continuous_rotate,
    masked_mse_per_image,
    masked_mae_per_image,
    gradient_mse_per_image,
    out_of_range_per_image,
    normalized_to_image_domain,
    ssim_per_image,
    np,
    torch,
):
    metric_names = (
        "action_mse_disk",
        "action_identity_mse_disk",
        "input_commutation_mse_disk",
        "action_wrong_mse_disk",
        "canonical_mse_disk",
        "canonical_identity_mse_disk",
        "canonical_wrong_mse_disk",
        "oracle_mse_disk",
        "action_mae_disk",
        "canonical_mae_disk",
        "action_mse_full",
        "action_mae_full",
        "canonical_mse_full",
        "canonical_mae_full",
        "action_ssim_crop",
        "action_identity_ssim_crop",
        "canonical_ssim_crop",
        "canonical_identity_ssim_crop",
        "action_out_fraction",
        "action_target_out_fraction",
        "canonical_out_fraction",
        "canonical_target_out_fraction",
        "action_overshoot",
        "action_target_overshoot",
        "canonical_overshoot",
        "canonical_target_overshoot",
        "action_gradient_mse",
        "action_identity_gradient_mse",
        "canonical_gradient_mse",
        "canonical_identity_gradient_mse",
        "latent_canonical_mse_disk",
        "latent_identity_mse_disk",
    )
    dense = {
        name: np.empty((25, len(ANGLES)), dtype=np.float64) for name in metric_names
    }
    selected = {}
    with torch.inference_mode():
        z0 = model.encode(patches)[0]
        y0 = model.decode(z0)
        for angle_index, angle in enumerate(ANGLES):
            for start in range(0, 25, CHUNK_SIZE):
                stop = min(start + CHUNK_SIZE, 25)
                patch = patches[start:stop]
                z_base = z0[start:stop]
                y_base = y0[start:stop]
                transformed_input = continuous_rotate(patch, angle)
                z_transformed = model.encode(transformed_input)[0]
                target_action = continuous_rotate(y_base, angle)
                z_action = continuous_rotate(z_base, angle)
                z_action_wrong = continuous_rotate(z_base, -angle)
                z_canonical = continuous_rotate(z_transformed, -angle)
                z_canonical_wrong = continuous_rotate(z_transformed, angle)
                decoded = model.decode(
                    torch.cat(
                        (
                            z_transformed,
                            z_action,
                            z_action_wrong,
                            z_canonical,
                            z_canonical_wrong,
                        ),
                        dim=0,
                    )
                )
                y_input, y_action, y_action_wrong, y_can, y_can_wrong = decoded.chunk(5)
                y_oracle = continuous_rotate(y_input, -angle)
                crop = (slice(None), slice(None), slice(16, 240), slice(16, 240))
                action_out, action_over = out_of_range_per_image(y_action)
                action_ref_out, action_ref_over = out_of_range_per_image(target_action)
                can_out, can_over = out_of_range_per_image(y_can)
                can_ref_out, can_ref_over = out_of_range_per_image(y_base)
                values = {
                    "action_mse_disk": masked_mse_per_image(
                        y_action, target_action, image_mask
                    ),
                    "action_identity_mse_disk": masked_mse_per_image(
                        y_base, target_action, image_mask
                    ),
                    "input_commutation_mse_disk": masked_mse_per_image(
                        y_input, target_action, image_mask
                    ),
                    "action_wrong_mse_disk": masked_mse_per_image(
                        y_action_wrong, target_action, image_mask
                    ),
                    "canonical_mse_disk": masked_mse_per_image(
                        y_can, y_base, image_mask
                    ),
                    "canonical_identity_mse_disk": masked_mse_per_image(
                        y_input, y_base, image_mask
                    ),
                    "canonical_wrong_mse_disk": masked_mse_per_image(
                        y_can_wrong, y_base, image_mask
                    ),
                    "oracle_mse_disk": masked_mse_per_image(
                        y_oracle, y_base, image_mask
                    ),
                    "action_mae_disk": masked_mae_per_image(
                        y_action, target_action, image_mask
                    ),
                    "canonical_mae_disk": masked_mae_per_image(
                        y_can, y_base, image_mask
                    ),
                    "action_mse_full": (y_action - target_action)
                    .square()
                    .mean(dim=(1, 2, 3)),
                    "action_mae_full": (y_action - target_action)
                    .abs()
                    .mean(dim=(1, 2, 3)),
                    "canonical_mse_full": (y_can - y_base).square().mean(dim=(1, 2, 3)),
                    "canonical_mae_full": (y_can - y_base).abs().mean(dim=(1, 2, 3)),
                    "action_ssim_crop": ssim_per_image(
                        normalized_to_image_domain(y_action[crop]),
                        normalized_to_image_domain(target_action[crop]),
                    ),
                    "action_identity_ssim_crop": ssim_per_image(
                        normalized_to_image_domain(y_base[crop]),
                        normalized_to_image_domain(target_action[crop]),
                    ),
                    "canonical_ssim_crop": ssim_per_image(
                        normalized_to_image_domain(y_can[crop]),
                        normalized_to_image_domain(y_base[crop]),
                    ),
                    "canonical_identity_ssim_crop": ssim_per_image(
                        normalized_to_image_domain(y_input[crop]),
                        normalized_to_image_domain(y_base[crop]),
                    ),
                    "action_out_fraction": action_out,
                    "action_target_out_fraction": action_ref_out,
                    "canonical_out_fraction": can_out,
                    "canonical_target_out_fraction": can_ref_out,
                    "action_overshoot": action_over,
                    "action_target_overshoot": action_ref_over,
                    "canonical_overshoot": can_over,
                    "canonical_target_overshoot": can_ref_over,
                    "action_gradient_mse": gradient_mse_per_image(
                        y_action, target_action, gradient_mask
                    ),
                    "action_identity_gradient_mse": gradient_mse_per_image(
                        y_base, target_action, gradient_mask
                    ),
                    "canonical_gradient_mse": gradient_mse_per_image(
                        y_can, y_base, gradient_mask
                    ),
                    "canonical_identity_gradient_mse": gradient_mse_per_image(
                        y_input, y_base, gradient_mask
                    ),
                    "latent_canonical_mse_disk": masked_mse_per_image(
                        z_canonical, z_base, latent_mask
                    ),
                    "latent_identity_mse_disk": masked_mse_per_image(
                        z_transformed, z_base, latent_mask
                    ),
                }
                for name, tensor in values.items():
                    dense[name][start:stop, angle_index] = tensor.float().cpu().numpy()
                if angle in EXAMPLE_ANGLES:
                    for local, rank in enumerate(range(start, stop)):
                        if rank in EXAMPLE_RANKS:
                            for name, tensor in (
                                ("input", patch),
                                ("base", y_base),
                                ("action_target", target_action),
                                ("action", y_action),
                                ("transformed_input_recon", y_input),
                                ("canonical", y_can),
                                ("oracle", y_oracle),
                            ):
                                selected[f"rank{rank}_angle{angle}_{name}"] = _uint8(
                                    tensor[local]
                                )
    if not all(np.isfinite(values).all() for values in dense.values()):
        raise RuntimeError("dense metric arrays contain non-finite values")
    return dense, selected, {"z0": z0, "y0": y0}


def _evaluate_exact_d4(
    model,
    patches,
    bases,
    image_mask,
    latent_mask,
    *,
    exact_names,
    exact_spatial_transform,
    inverse_exact_transform_name,
    masked_mse_per_image,
    normalized_to_image_domain,
    ssim_per_image,
    np,
    torch,
):
    rows = {}
    z0, y0 = bases["z0"], bases["y0"]
    with torch.inference_mode():
        for name in exact_names:
            inverse_name = inverse_exact_transform_name(name)
            values = {
                key: np.empty(25, dtype=np.float64)
                for key in (
                    "action_num",
                    "action_den",
                    "input_num",
                    "can_num",
                    "can_den",
                    "latent_num",
                    "latent_den",
                    "action_ssim_gain",
                    "canonical_ssim_gain",
                )
            }
            for start in range(0, 25, CHUNK_SIZE):
                stop = min(start + CHUNK_SIZE, 25)
                transformed_input = exact_spatial_transform(patches[start:stop], name)
                zt = model.encode(transformed_input)[0]
                z_base, y_base = z0[start:stop], y0[start:stop]
                target = exact_spatial_transform(y_base, name)
                y_input, y_action, y_can = model.decode(
                    torch.cat(
                        (
                            zt,
                            exact_spatial_transform(z_base, name),
                            exact_spatial_transform(zt, inverse_name),
                        ),
                        dim=0,
                    )
                ).chunk(3)
                crop = (slice(None), slice(None), slice(16, 240), slice(16, 240))
                tensors = {
                    "action_num": masked_mse_per_image(y_action, target, image_mask),
                    "action_den": masked_mse_per_image(y_base, target, image_mask),
                    "input_num": masked_mse_per_image(y_input, target, image_mask),
                    "can_num": masked_mse_per_image(y_can, y_base, image_mask),
                    "can_den": masked_mse_per_image(y_input, y_base, image_mask),
                    "latent_num": masked_mse_per_image(
                        exact_spatial_transform(zt, inverse_name), z_base, latent_mask
                    ),
                    "latent_den": masked_mse_per_image(zt, z_base, latent_mask),
                    "action_ssim_gain": ssim_per_image(
                        normalized_to_image_domain(y_action[crop]),
                        normalized_to_image_domain(target[crop]),
                    )
                    - ssim_per_image(
                        normalized_to_image_domain(y_base[crop]),
                        normalized_to_image_domain(target[crop]),
                    ),
                    "canonical_ssim_gain": ssim_per_image(
                        normalized_to_image_domain(y_can[crop]),
                        normalized_to_image_domain(y_base[crop]),
                    )
                    - ssim_per_image(
                        normalized_to_image_domain(y_input[crop]),
                        normalized_to_image_domain(y_base[crop]),
                    ),
                }
                for key, tensor in tensors.items():
                    values[key][start:stop] = tensor.float().cpu().numpy()
            rows[name] = {
                "action_mse_per_patch": values["action_num"].tolist(),
                "action_identity_mse_per_patch": values["action_den"].tolist(),
                "input_commutation_mse_per_patch": values["input_num"].tolist(),
                "canonical_mse_per_patch": values["can_num"].tolist(),
                "canonical_identity_mse_per_patch": values["can_den"].tolist(),
                "action_ratio_per_patch": (
                    np.sqrt(values["action_num"])
                    / (np.sqrt(values["action_den"]) + 1e-8)
                ).tolist(),
                "input_commutation_ratio_per_patch": (
                    np.sqrt(values["input_num"])
                    / (np.sqrt(values["action_den"]) + 1e-8)
                ).tolist(),
                "canonical_ratio_per_patch": (
                    np.sqrt(values["can_num"]) / (np.sqrt(values["can_den"]) + 1e-8)
                ).tolist(),
                "raw_latent_ratio_per_patch": (
                    np.sqrt(values["latent_num"])
                    / (np.sqrt(values["latent_den"]) + 1e-8)
                ).tolist(),
                "canonical_pose_signal_rms_per_patch": np.sqrt(
                    values["can_den"]
                ).tolist(),
                "action_ssim_gain_per_patch": values["action_ssim_gain"].tolist(),
                "canonical_ssim_gain_per_patch": values["canonical_ssim_gain"].tolist(),
            }
    return rows


def _summarize_branch(dense, exact, *, aggregate_rms_ratio, np):
    nonzero = [index for index, angle in enumerate(ANGLES) if angle != 0]
    action = aggregate_rms_ratio(
        dense["action_mse_disk"],
        dense["action_identity_mse_disk"],
        angle_indices=nonzero,
    )
    input_commutation = aggregate_rms_ratio(
        dense["input_commutation_mse_disk"],
        dense["action_identity_mse_disk"],
        angle_indices=nonzero,
    )
    canonical = aggregate_rms_ratio(
        dense["canonical_mse_disk"],
        dense["canonical_identity_mse_disk"],
        angle_indices=nonzero,
    )
    raw_latent = aggregate_rms_ratio(
        dense["latent_canonical_mse_disk"],
        dense["latent_identity_mse_disk"],
        angle_indices=nonzero,
    )
    pose_signal = np.sqrt(dense["canonical_identity_mse_disk"][:, nonzero].mean(axis=1))
    action_pose_signal = np.sqrt(
        dense["action_identity_mse_disk"][:, nonzero].mean(axis=1)
    )
    action_gradient_ratio = aggregate_rms_ratio(
        dense["action_gradient_mse"],
        dense["action_identity_gradient_mse"],
        angle_indices=nonzero,
    )
    canonical_gradient_ratio = aggregate_rms_ratio(
        dense["canonical_gradient_mse"],
        dense["canonical_identity_gradient_mse"],
        angle_indices=nonzero,
    )
    action_ssim_gain = (
        dense["action_ssim_crop"][:, nonzero]
        - dense["action_identity_ssim_crop"][:, nonzero]
    ).mean(axis=1)
    canonical_ssim_gain = (
        dense["canonical_ssim_crop"][:, nonzero]
        - dense["canonical_identity_ssim_crop"][:, nonzero]
    ).mean(axis=1)
    rows = [
        {
            "rank": rank,
            "action_ratio": float(action[rank]),
            "input_commutation_ratio": float(input_commutation[rank]),
            "canonical_ratio": float(canonical[rank]),
            "action_rms_disk": float(
                np.sqrt(np.mean(dense["action_mse_disk"][rank, nonzero]))
            ),
            "action_mae_disk": float(np.mean(dense["action_mae_disk"][rank, nonzero])),
            "action_rms_full": float(
                np.sqrt(np.mean(dense["action_mse_full"][rank, nonzero]))
            ),
            "action_mae_full": float(np.mean(dense["action_mae_full"][rank, nonzero])),
            "canonical_rms_disk": float(
                np.sqrt(np.mean(dense["canonical_mse_disk"][rank, nonzero]))
            ),
            "canonical_mae_disk": float(
                np.mean(dense["canonical_mae_disk"][rank, nonzero])
            ),
            "canonical_rms_full": float(
                np.sqrt(np.mean(dense["canonical_mse_full"][rank, nonzero]))
            ),
            "canonical_mae_full": float(
                np.mean(dense["canonical_mae_full"][rank, nonzero])
            ),
            "wrong_sign_action_ratio": float(
                np.sqrt(np.mean(dense["action_wrong_mse_disk"][rank, nonzero]))
                / (
                    np.sqrt(np.mean(dense["action_identity_mse_disk"][rank, nonzero]))
                    + 1e-8
                )
            ),
            "wrong_sign_canonical_ratio": float(
                np.sqrt(np.mean(dense["canonical_wrong_mse_disk"][rank, nonzero]))
                / (
                    np.sqrt(
                        np.mean(dense["canonical_identity_mse_disk"][rank, nonzero])
                    )
                    + 1e-8
                )
            ),
            "output_control_canonical_ratio": float(
                np.sqrt(np.mean(dense["oracle_mse_disk"][rank, nonzero]))
                / (
                    np.sqrt(
                        np.mean(dense["canonical_identity_mse_disk"][rank, nonzero])
                    )
                    + 1e-8
                )
            ),
            "raw_latent_canonical_ratio": float(raw_latent[rank]),
            "canonical_pose_signal_rms": float(pose_signal[rank]),
            "action_pose_signal_rms": float(action_pose_signal[rank]),
            "action_nondegenerate": bool(action_pose_signal[rank] >= 0.02),
            "canonical_nondegenerate": bool(pose_signal[rank] >= 0.02),
            "action_ssim_gain": float(action_ssim_gain[rank]),
            "canonical_ssim_gain": float(canonical_ssim_gain[rank]),
            "action_excess_overshoot": float(
                np.mean(
                    dense["action_overshoot"][rank, nonzero]
                    - dense["action_target_overshoot"][rank, nonzero]
                )
            ),
            "action_excess_out_of_range_fraction": float(
                np.mean(
                    dense["action_out_fraction"][rank, nonzero]
                    - dense["action_target_out_fraction"][rank, nonzero]
                )
            ),
            "canonical_excess_overshoot": float(
                np.mean(
                    dense["canonical_overshoot"][rank, nonzero]
                    - dense["canonical_target_overshoot"][rank, nonzero]
                )
            ),
            "canonical_excess_out_of_range_fraction": float(
                np.mean(
                    dense["canonical_out_fraction"][rank, nonzero]
                    - dense["canonical_target_out_fraction"][rank, nonzero]
                )
            ),
            "action_gradient_rms": float(
                np.sqrt(np.mean(dense["action_gradient_mse"][rank, nonzero]))
            ),
            "canonical_gradient_rms": float(
                np.sqrt(np.mean(dense["canonical_gradient_mse"][rank, nonzero]))
            ),
            "action_gradient_error_ratio": float(action_gradient_ratio[rank]),
            "canonical_gradient_error_ratio": float(canonical_gradient_ratio[rank]),
        }
        for rank in range(25)
    ]
    exact_out = {}
    for name, values in exact.items():
        action_signal = np.sqrt(np.asarray(values["action_identity_mse_per_patch"]))
        canonical_signal = np.sqrt(
            np.asarray(values["canonical_identity_mse_per_patch"])
        )
        valid = (action_signal >= 0.02) & (canonical_signal >= 0.02)
        valid_action = np.asarray(values["action_ratio_per_patch"])[valid]
        valid_input = np.asarray(values["input_commutation_ratio_per_patch"])[valid]
        valid_canonical = np.asarray(values["canonical_ratio_per_patch"])[valid]
        exact_out[name] = values | {
            "joint_valid_per_patch": valid.tolist(),
            "joint_valid_count": int(valid.sum()),
            "action_ratio_median_valid": _optional_median(valid_action, np=np),
            "input_commutation_ratio_median_valid": _optional_median(
                valid_input, np=np
            ),
            "canonical_ratio_median_valid": _optional_median(valid_canonical, np=np),
            "action_success_fraction_valid": _optional_fraction(
                valid_action <= 0.75, np=np
            ),
            "input_commutation_success_fraction_valid": _optional_fraction(
                valid_input <= 0.75, np=np
            ),
            "canonical_success_fraction_valid": _optional_fraction(
                valid_canonical <= 0.75, np=np
            ),
        }
    quarter_names = ("rot90", "rot180", "rot270")
    exact_action = np.sqrt(
        np.mean(
            [exact_out[name]["action_mse_per_patch"] for name in quarter_names], axis=0
        )
    ) / (
        np.sqrt(
            np.mean(
                [
                    exact_out[name]["action_identity_mse_per_patch"]
                    for name in quarter_names
                ],
                axis=0,
            )
        )
        + 1e-8
    )
    exact_can = np.sqrt(
        np.mean(
            [exact_out[name]["canonical_mse_per_patch"] for name in quarter_names],
            axis=0,
        )
    ) / (
        np.sqrt(
            np.mean(
                [
                    exact_out[name]["canonical_identity_mse_per_patch"]
                    for name in quarter_names
                ],
                axis=0,
            )
        )
        + 1e-8
    )
    exact_action_signal = np.sqrt(
        np.mean(
            [
                exact_out[name]["action_identity_mse_per_patch"]
                for name in quarter_names
            ],
            axis=0,
        )
    )
    exact_can_signal = np.sqrt(
        np.mean(
            [
                exact_out[name]["canonical_identity_mse_per_patch"]
                for name in quarter_names
            ],
            axis=0,
        )
    )
    exact_joint_valid = (exact_action_signal >= 0.02) & (exact_can_signal >= 0.02)
    ratio_by_angle = {}
    for name, numerator, denominator in (
        ("action", "action_mse_disk", "action_identity_mse_disk"),
        ("input_commutation", "input_commutation_mse_disk", "action_identity_mse_disk"),
        ("canonical", "canonical_mse_disk", "canonical_identity_mse_disk"),
    ):
        values = np.sqrt(dense[numerator][:, 1:]) / (
            np.sqrt(dense[denominator][:, 1:]) + 1e-8
        )
        ratio_by_angle[name] = {
            "angles_degrees": list(ANGLES[1:]),
            "median": np.median(values, axis=0).tolist(),
            "q1": np.quantile(values, 0.25, axis=0).tolist(),
            "q3": np.quantile(values, 0.75, axis=0).tolist(),
        }
    return {
        "per_patch": rows,
        "population": {
            "action_ratio_median": float(np.median(action)),
            "input_commutation_ratio_median": float(np.median(input_commutation)),
            "action_ratio_success_count": int(np.sum(action <= 0.75)),
            "canonical_ratio_median": float(np.median(canonical)),
            "canonical_ratio_success_count": int(np.sum(canonical <= 0.75)),
            "canonical_nondegenerate_count": int(np.sum(pose_signal >= 0.02)),
            "action_nondegenerate_count": int(np.sum(action_pose_signal >= 0.02)),
            "raw_latent_canonical_ratio_median": float(np.median(raw_latent)),
            "action_ssim_gain_positive_count": int(np.sum(action_ssim_gain > 0)),
            "canonical_ssim_gain_positive_count": int(np.sum(canonical_ssim_gain > 0)),
            "exact_quarter_action_ratio_median": float(np.median(exact_action)),
            "exact_quarter_canonical_ratio_median": float(np.median(exact_can)),
        },
        "exact_quarter_aggregate": {
            "action_ratio_per_patch": exact_action.tolist(),
            "canonical_ratio_per_patch": exact_can.tolist(),
            "action_pose_signal_rms_per_patch": exact_action_signal.tolist(),
            "canonical_pose_signal_rms_per_patch": exact_can_signal.tolist(),
            "joint_valid_per_patch": exact_joint_valid.tolist(),
            "joint_valid_count": int(exact_joint_valid.sum()),
        },
        "angle_summary": {
            name: {
                "median": np.median(values, axis=0).tolist(),
                "q1": np.quantile(values, 0.25, axis=0).tolist(),
                "q3": np.quantile(values, 0.75, axis=0).tolist(),
            }
            for name, values in dense.items()
        },
        "ratio_by_angle": ratio_by_angle,
        "robustness": {
            str(step): _subset_summary(
                dense, step, aggregate_rms_ratio=aggregate_rms_ratio, np=np
            )
            for step in (10, 20)
        }
        | {
            "exclude_cardinal_5deg": _cardinal_excluded_summary(
                dense, aggregate_rms_ratio=aggregate_rms_ratio, np=np
            )
        },
        "exact_d4": exact_out,
    }


def _subset_summary(dense, step, *, aggregate_rms_ratio, np):
    indices = [
        index for index, angle in enumerate(ANGLES) if angle and angle % step == 0
    ]
    return {
        "action_ratio_median": float(
            np.median(
                aggregate_rms_ratio(
                    dense["action_mse_disk"],
                    dense["action_identity_mse_disk"],
                    angle_indices=indices,
                )
            )
        ),
        "canonical_ratio_median": float(
            np.median(
                aggregate_rms_ratio(
                    dense["canonical_mse_disk"],
                    dense["canonical_identity_mse_disk"],
                    angle_indices=indices,
                )
            )
        ),
    }


def _cardinal_excluded_summary(dense, *, aggregate_rms_ratio, np):
    def distance(angle):
        return min(
            abs(((angle - cardinal + 180) % 360) - 180)
            for cardinal in (0, 90, 180, 270)
        )

    indices = [index for index, angle in enumerate(ANGLES) if distance(angle) > 5]
    return {
        "angles": [ANGLES[index] for index in indices],
        "action_ratio_median": float(
            np.median(
                aggregate_rms_ratio(
                    dense["action_mse_disk"],
                    dense["action_identity_mse_disk"],
                    angle_indices=indices,
                )
            )
        ),
        "canonical_ratio_median": float(
            np.median(
                aggregate_rms_ratio(
                    dense["canonical_mse_disk"],
                    dense["canonical_identity_mse_disk"],
                    angle_indices=indices,
                )
            )
        ),
    }


def _optional_median(values, *, np):
    return float(np.median(values)) if values.size else None


def _optional_fraction(values, *, np):
    return float(np.mean(values)) if values.size else None


def _comparisons(
    branches,
    *,
    cluster_ids,
    paired_bootstrap_median_difference,
    np,
):
    output = {}
    directions = {
        "action_ratio": "lower",
        "input_commutation_ratio": "lower",
        "canonical_ratio": "lower",
        "action_rms_disk": "lower",
        "action_mae_disk": "lower",
        "action_rms_full": "lower",
        "action_mae_full": "lower",
        "canonical_rms_disk": "lower",
        "canonical_mae_disk": "lower",
        "canonical_rms_full": "lower",
        "canonical_mae_full": "lower",
        "wrong_sign_action_ratio": None,
        "wrong_sign_canonical_ratio": None,
        "output_control_canonical_ratio": None,
        "raw_latent_canonical_ratio": None,
        "action_ssim_gain": "higher",
        "canonical_ssim_gain": "higher",
        "action_excess_out_of_range_fraction": "lower",
        "canonical_excess_out_of_range_fraction": "lower",
        "action_excess_overshoot": "lower",
        "canonical_excess_overshoot": "lower",
        "action_gradient_error_ratio": "lower",
        "canonical_gradient_error_ratio": "lower",
    }
    for key, direction in directions.items():
        normal = np.asarray([row[key] for row in branches["normal_vae"]["per_patch"]])
        so2 = np.asarray([row[key] for row in branches["so2_vae"]["per_patch"]])
        output[key] = {
            "normal": normal.tolist(),
            "so2": so2.tolist(),
            "normal_median": float(np.median(normal)),
            "so2_median": float(np.median(so2)),
            "paired": paired_bootstrap_median_difference(
                normal,
                so2,
                seed=1827149135,
                favorable_direction=direction,
                cluster_ids=cluster_ids,
            ),
        }
        valid_field = {
            "action_ratio": "action_nondegenerate",
            "canonical_ratio": "canonical_nondegenerate",
        }.get(key)
        if valid_field is not None:
            valid = np.asarray(
                [row[valid_field] for row in branches["so2_vae"]["per_patch"]],
                dtype=bool,
            )
            valid_ranks = np.flatnonzero(valid)
            output[key]["supporting_so2_valid_subset"] = {
                "ranks": valid_ranks.tolist(),
                "normal": normal[valid].tolist(),
                "so2": so2[valid].tolist(),
                "paired": (
                    paired_bootstrap_median_difference(
                        normal[valid],
                        so2[valid],
                        seed=1827149135,
                        favorable_direction=direction,
                        cluster_ids=np.asarray(cluster_ids)[valid].tolist(),
                    )
                    if valid_ranks.size
                    else None
                ),
            }
    output["exact_d4"] = {}
    for transform in branches["normal_vae"]["exact_d4"]:
        output["exact_d4"][transform] = {}
        normal_valid = np.asarray(
            branches["normal_vae"]["exact_d4"][transform]["joint_valid_per_patch"],
            dtype=bool,
        )
        so2_valid = np.asarray(
            branches["so2_vae"]["exact_d4"][transform]["joint_valid_per_patch"],
            dtype=bool,
        )
        jointly_valid = normal_valid & so2_valid
        valid_ranks = np.flatnonzero(jointly_valid)
        for endpoint in (
            "input_commutation_ratio_per_patch",
            "action_ratio_per_patch",
            "canonical_ratio_per_patch",
            "raw_latent_ratio_per_patch",
        ):
            normal = branches["normal_vae"]["exact_d4"][transform][endpoint]
            so2 = branches["so2_vae"]["exact_d4"][transform][endpoint]
            output["exact_d4"][transform][endpoint] = {
                "all_25_descriptive": {
                    "normal": normal,
                    "so2": so2,
                    "normal_median": float(np.median(normal)),
                    "so2_median": float(np.median(so2)),
                },
                "supporting_joint_valid_subset": {
                    "ranks": valid_ranks.tolist(),
                    "count": int(valid_ranks.size),
                    "assessable": bool(valid_ranks.size >= 18),
                    "normal": np.asarray(normal)[jointly_valid].tolist(),
                    "so2": np.asarray(so2)[jointly_valid].tolist(),
                    "normal_median": _optional_median(
                        np.asarray(normal)[jointly_valid], np=np
                    ),
                    "so2_median": _optional_median(
                        np.asarray(so2)[jointly_valid], np=np
                    ),
                    "paired": (
                        paired_bootstrap_median_difference(
                            np.asarray(normal)[jointly_valid],
                            np.asarray(so2)[jointly_valid],
                            seed=1827149135,
                            favorable_direction=(
                                None
                                if endpoint == "raw_latent_ratio_per_patch"
                                else "lower"
                            ),
                            cluster_ids=np.asarray(cluster_ids)[jointly_valid].tolist(),
                        )
                        if valid_ranks.size
                        else None
                    ),
                },
            }
    return output


def _decisions(branches, *, np):
    normal = branches["normal_vae"]
    so2 = branches["so2_vae"]
    action_n = np.asarray([row["action_ratio"] for row in normal["per_patch"]])
    action_s = np.asarray([row["action_ratio"] for row in so2["per_patch"]])
    can_n = np.asarray([row["canonical_ratio"] for row in normal["per_patch"]])
    can_s = np.asarray([row["canonical_ratio"] for row in so2["per_patch"]])
    action_valid = np.asarray(
        [row["action_nondegenerate"] for row in so2["per_patch"]], dtype=bool
    )
    can_valid = np.asarray(
        [row["canonical_nondegenerate"] for row in so2["per_patch"]], dtype=bool
    )
    action_count = int(action_valid.sum())
    can_count = int(can_valid.sum())
    action_ssim = np.asarray([row["action_ssim_gain"] for row in so2["per_patch"]])
    can_ssim = np.asarray([row["canonical_ssim_gain"] for row in so2["per_patch"]])
    action_out = np.asarray([
        row["action_excess_out_of_range_fraction"] for row in so2["per_patch"]
    ])
    can_out = np.asarray([
        row["canonical_excess_out_of_range_fraction"] for row in so2["per_patch"]
    ])
    action_overshoot = np.asarray([
        row["action_excess_overshoot"] for row in so2["per_patch"]
    ])
    can_overshoot = np.asarray([
        row["canonical_excess_overshoot"] for row in so2["per_patch"]
    ])
    action_gradient = np.asarray([
        row["action_gradient_error_ratio"] for row in so2["per_patch"]
    ])
    can_gradient = np.asarray([
        row["canonical_gradient_error_ratio"] for row in so2["per_patch"]
    ])
    exact_action = np.asarray(so2["exact_quarter_aggregate"]["action_ratio_per_patch"])
    exact_can = np.asarray(so2["exact_quarter_aggregate"]["canonical_ratio_per_patch"])
    exact_joint_valid = np.asarray(
        so2["exact_quarter_aggregate"]["joint_valid_per_patch"], dtype=bool
    )
    h1_exact_valid = action_valid & exact_joint_valid
    h2_exact_valid = can_valid & exact_joint_valid
    h1_exact_count = int(h1_exact_valid.sum())
    h2_exact_count = int(h2_exact_valid.sum())
    h1_checks = {
        "nondegenerate_at_least_18": action_count >= 18,
        "exact_quarter_joint_nondegenerate_at_least_18": h1_exact_count >= 18,
        "valid_median_at_most_0_50": bool(
            action_count and np.median(action_s[action_valid]) <= 0.5
        ),
        "valid_success_fraction_at_least_0_75": bool(
            action_count and np.mean(action_s[action_valid] <= 0.75) >= 0.75
        ),
        "relative_advantage_at_least_20pct": bool(
            action_count
            and np.median(action_s[action_valid])
            <= 0.8 * np.median(action_n[action_valid])
        ),
        "paired_favorable_fraction_at_least_0_75": bool(
            action_count
            and np.mean(action_s[action_valid] < action_n[action_valid]) >= 0.75
        ),
        "exact_quarter_at_most_0_75": bool(
            h1_exact_count and np.median(exact_action[h1_exact_valid]) <= 0.75
        ),
        "valid_ssim_gain_positive_fraction_at_least_0_75": bool(
            action_count and np.mean(action_ssim[action_valid] > 0) >= 0.75
        ),
        "artifact_out_of_range_pass": bool(
            action_count and np.median(action_out[action_valid]) <= 0.01
        ),
        "artifact_overshoot_pass": bool(
            action_count and np.median(action_overshoot[action_valid]) <= 0.002
        ),
        "artifact_gradient_pass": bool(
            action_count and np.median(action_gradient[action_valid]) <= 1.0
        ),
    }
    h2_checks = {
        "nondegenerate_at_least_18": can_count >= 18,
        "exact_quarter_joint_nondegenerate_at_least_18": h2_exact_count >= 18,
        "valid_median_at_most_0_50": bool(
            can_count and np.median(can_s[can_valid]) <= 0.5
        ),
        "valid_success_fraction_at_least_0_75": bool(
            can_count and np.mean(can_s[can_valid] <= 0.75) >= 0.75
        ),
        "valid_relative_advantage_at_least_20pct": bool(
            can_count
            and np.median(can_s[can_valid]) <= 0.8 * np.median(can_n[can_valid])
        ),
        "valid_paired_favorable_fraction_at_least_0_75": bool(
            can_count and np.mean(can_s[can_valid] < can_n[can_valid]) >= 0.75
        ),
        "exact_quarter_at_most_0_75": bool(
            h2_exact_count and np.median(exact_can[h2_exact_valid]) <= 0.75
        ),
        "valid_ssim_gain_positive_fraction_at_least_0_75": bool(
            can_count and np.mean(can_ssim[can_valid] > 0) >= 0.75
        ),
        "artifact_out_of_range_pass": bool(
            can_count and np.median(can_out[can_valid]) <= 0.01
        ),
        "artifact_overshoot_pass": bool(
            can_count and np.median(can_overshoot[can_valid]) <= 0.002
        ),
        "artifact_gradient_pass": bool(
            can_count and np.median(can_gradient[can_valid]) <= 1.0
        ),
    }
    h1_status = (
        "unresolved_insufficient_pose_signal"
        if action_count < 18
        else (
            "unresolved_insufficient_exact_quarter_pose_signal"
            if h1_exact_count < 18
            else ("supported" if all(h1_checks.values()) else "not_supported")
        )
    )
    h2_status = (
        "unresolved_insufficient_pose_signal"
        if can_count < 18
        else (
            "unresolved_insufficient_exact_quarter_pose_signal"
            if h2_exact_count < 18
            else ("supported" if all(h2_checks.values()) else "not_supported")
        )
    )
    raw_latent = np.asarray([
        row["raw_latent_canonical_ratio"] for row in so2["per_patch"]
    ])
    h3_checks = {
        "decoded_canonicalization_supported": h2_status == "supported",
        "valid_raw_latent_median_above_0_75": bool(
            can_count and np.median(raw_latent[can_valid]) > 0.75
        ),
        "same_patch_joint_fraction_at_least_0_75": bool(
            can_count
            and np.mean((raw_latent[can_valid] > 0.75) & (can_s[can_valid] <= 0.50))
            >= 0.75
        ),
    }
    if h2_status.startswith("unresolved_"):
        h3_status = "unresolved_inherited_insufficient_pose_signal"
    else:
        h3_status = "supported" if all(h3_checks.values()) else "not_supported"
    reflection = {}
    for name in ("flip_h", "flip_v", "flip_diag", "flip_anti_diag"):
        model_results = {}
        for branch in ("normal_vae", "so2_vae"):
            row = branches[branch]["exact_d4"][name]
            assessable = row["joint_valid_count"] >= 18
            checks = {
                "joint_valid_at_least_18": assessable,
                "input_median_at_most_0_50": assessable
                and row["input_commutation_ratio_median_valid"] <= 0.5,
                "input_success_fraction_at_least_0_75": assessable
                and row["input_commutation_success_fraction_valid"] >= 0.75,
                "action_median_at_most_0_50": assessable
                and row["action_ratio_median_valid"] <= 0.5,
                "action_success_fraction_at_least_0_75": assessable
                and row["action_success_fraction_valid"] >= 0.75,
                "canonical_median_at_most_0_50": assessable
                and row["canonical_ratio_median_valid"] <= 0.5,
                "canonical_success_fraction_at_least_0_75": assessable
                and row["canonical_success_fraction_valid"] >= 0.75,
            }
            status = (
                "unresolved_insufficient_pose_signal"
                if not assessable
                else ("supported" if all(checks.values()) else "not_supported")
            )
            model_results[branch] = {"status": status, "checks": checks}
        normal_supported = model_results["normal_vae"]["status"] == "supported"
        so2_supported = model_results["so2_vae"]["status"] == "supported"
        normal_unresolved = model_results["normal_vae"]["status"].startswith(
            "unresolved_"
        )
        so2_unresolved = model_results["so2_vae"]["status"].startswith("unresolved_")
        if normal_supported and so2_supported:
            interpretation = "generic_both_models"
        elif normal_supported and so2_unresolved:
            interpretation = "normal_supported_so2_unresolved"
        elif so2_supported and normal_unresolved:
            interpretation = "so2_supported_normal_unresolved"
        elif normal_unresolved or so2_unresolved:
            interpretation = "unresolved_in_at_least_one_model"
        elif so2_supported:
            interpretation = "so2_checkpoint_only_not_architecture_proof"
        elif normal_supported:
            interpretation = "normal_checkpoint_only"
        else:
            interpretation = "not_supported_either_model"
        reflection[name] = {
            "models": model_results,
            "interpretation": interpretation,
        }
    return {
        "H1_decoded_rotational_action": {
            "status": h1_status,
            "supported": h1_status == "supported",
            "checks": h1_checks,
        },
        "H2_decoded_rotational_canonicalization": {
            "status": h2_status,
            "supported": h2_status == "supported",
            "checks": h2_checks,
        },
        "H3_decoded_residual_suppression": {
            "status": h3_status,
            "supported": h3_status == "supported",
            "checks": h3_checks,
        },
        "H4_reflection_robustness": reflection,
    }


def _render_population(branches, *, plt, np):
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.8), constrained_layout=True)
    for axis, key, title in zip(
        axes,
        ("action_ratio", "canonical_ratio"),
        ("Acción decodificada", "Canonicalización decodificada"),
        strict=True,
    ):
        values = [
            [row[key] for row in branches[branch]["per_patch"]]
            for branch in ("normal_vae", "so2_vae")
        ]
        axis.boxplot(values, tick_labels=["normal", "SO(2)"], showfliers=False)
        for index, branch_values in enumerate(values, start=1):
            jitter = np.linspace(-0.08, 0.08, len(branch_values))
            axis.scatter(index + jitter, branch_values, s=11, alpha=0.7)
        axis.axhline(1.0, color="0.45", linestyle="--", linewidth=1)
        axis.axhline(0.5, color="#2ca02c", linestyle=":", linewidth=1)
        axis.set_title(title)
        axis.set_ylabel("razón RMS (menor es mejor)")
    figure.savefig(FIGURE_ROOT / "01-population-ratios.png", dpi=180)
    plt.close(figure)


def _render_angle_series(branches, *, plt, np):
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for axis, ratio_name, title in (
        (axes[0], "action", "Acción"),
        (axes[1], "canonical", "Canonicalización"),
    ):
        for branch, color in (("normal_vae", "#2f7ed8"), ("so2_vae", "#e67e22")):
            summary = branches[branch]["ratio_by_angle"][ratio_name]
            plotted_angles = summary["angles_degrees"]
            ratio = np.asarray(summary["median"])
            q1 = np.asarray(summary["q1"])
            q3 = np.asarray(summary["q3"])
            label = "normal" if branch == "normal_vae" else "SO(2)"
            axis.plot(plotted_angles, ratio, label=label, color=color)
            axis.fill_between(plotted_angles, q1, q3, color=color, alpha=0.15)
        axis.axhline(1.0, color="0.45", linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("ángulo (grados)")
        axis.set_ylabel("razón RMS mediana")
        axis.legend()
    figure.savefig(FIGURE_ROOT / "02-ratio-by-angle.png", dpi=180)
    plt.close(figure)


def _render_examples(rank, selected_images, *, kind, plt, np):
    columns = ("action_target", "action") if kind == "action" else ("base", "canonical")
    figure, axes = plt.subplots(
        len(EXAMPLE_ANGLES), 6, figsize=(12.5, 15.5), constrained_layout=True
    )
    for row, angle in enumerate(EXAMPLE_ANGLES):
        normal_target = selected_images["normal_vae"][
            f"rank{rank}_angle{angle}_{columns[0]}"
        ]
        normal_value = selected_images["normal_vae"][
            f"rank{rank}_angle{angle}_{columns[1]}"
        ]
        so2_target = selected_images["so2_vae"][f"rank{rank}_angle{angle}_{columns[0]}"]
        so2_value = selected_images["so2_vae"][f"rank{rank}_angle{angle}_{columns[1]}"]
        shown = [
            normal_target,
            normal_value,
            np.abs(normal_value.astype(float) - normal_target).mean(axis=0),
            so2_target,
            so2_value,
            np.abs(so2_value.astype(float) - so2_target).mean(axis=0),
        ]
        for column, value in enumerate(shown):
            axis = axes[row, column]
            if value.ndim == 2:
                axis.imshow(value, cmap="magma", vmin=0, vmax=64)
            else:
                axis.imshow(value.transpose(1, 2, 0))
            axis.set_xticks([])
            axis.set_yticks([])
            if column == 0:
                axis.set_ylabel(f"{angle}°")
    titles = (
        "objetivo normal",
        "normal",
        "|error|",
        "objetivo SO(2)",
        "SO(2)",
        "|error|",
    )
    for axis, title in zip(axes[0], titles, strict=True):
        axis.set_title(title)
    figure.savefig(FIGURE_ROOT / f"03-{kind}-rank{rank}.png", dpi=180)
    plt.close(figure)


def _render_exact(branches, *, plt, np):
    names = tuple(branches["normal_vae"]["exact_d4"])
    x = np.arange(len(names))
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4), constrained_layout=True)
    for axis, key, title in (
        (
            axes[0],
            "input_commutation_ratio_median_valid",
            "Ruta de entrada exacta",
        ),
        (axes[1], "action_ratio_median_valid", "Acción exacta"),
        (axes[2], "canonical_ratio_median_valid", "Canonicalización exacta"),
    ):
        normal = [
            _plot_value(branches["normal_vae"]["exact_d4"][name][key]) for name in names
        ]
        so2 = [
            _plot_value(branches["so2_vae"]["exact_d4"][name][key]) for name in names
        ]
        axis.bar(x - 0.18, normal, 0.36, label="normal")
        axis.bar(x + 0.18, so2, 0.36, label="SO(2)")
        axis.axhline(1, color="0.45", linestyle="--")
        axis.set_xticks(x, names, rotation=35, ha="right")
        axis.set_title(title)
        axis.set_ylabel("razón RMS mediana")
        axis.legend()
    figure.savefig(FIGURE_ROOT / "04-exact-d4.png", dpi=180)
    plt.close(figure)


def _plot_value(value):
    return float("nan") if value is None else value


def _render_latent_decoded(branches, *, plt, np):
    figure, axis = plt.subplots(figsize=(5.5, 4.5), constrained_layout=True)
    for branch, color in (("normal_vae", "#2f7ed8"), ("so2_vae", "#e67e22")):
        x = [row["raw_latent_canonical_ratio"] for row in branches[branch]["per_patch"]]
        y = [row["canonical_ratio"] for row in branches[branch]["per_patch"]]
        axis.scatter(
            x,
            y,
            label="normal" if branch == "normal_vae" else "SO(2)",
            color=color,
            alpha=0.8,
        )
    axis.axvline(0.75, color="0.45", linestyle=":")
    axis.axhline(0.5, color="0.45", linestyle=":")
    axis.set_xlabel("razón de discrepancia en mu")
    axis.set_ylabel("razón tras decodificar")
    axis.legend()
    figure.savefig(FIGURE_ROOT / "05-latent-vs-decoded.png", dpi=180)
    plt.close(figure)


def _load_models(
    bundle_root,
    weight_contract,
    contract,
    *,
    build_model,
    sha256_file,
    state_dict_sha256,
    device,
    torch,
):
    models, provenance = {}, {}
    for branch, kind in MODEL_KINDS.items():
        record = weight_contract["weights"][branch]
        state_path = bundle_root / f"{branch}_state.pt"
        prefix = branch.split("_")[0]
        if state_path.stat().st_size != record["state_file_bytes"]:
            raise RuntimeError(f"state file size differs for {branch}")
        state_file_hash = sha256_file(state_path)
        if (
            state_file_hash != record["state_file_sha256"]
            or state_file_hash != contract["inputs"][f"{prefix}_state_file_sha256"]
        ):
            raise RuntimeError(f"state file differs for {branch}")
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        state_hash = state_dict_sha256(state)
        if (
            state_hash != record["state_dict_sha256"]
            or state_hash != contract["inputs"][f"{prefix}_state_dict_sha256"]
        ):
            raise RuntimeError(f"state dict differs for {branch}")
        if (
            record["source_checkpoint_sha256"]
            != contract["inputs"][f"{prefix}_checkpoint_sha256"]
        ):
            raise RuntimeError(f"checkpoint differs for {branch}")
        model = build_model(kind)
        model.load_state_dict(state, strict=True)
        models[branch] = model.to(device).eval().requires_grad_(False)
        provenance[branch] = {
            "state_file_sha256": state_file_hash,
            "state_dict_sha256": state_hash,
            "source_checkpoint_sha256": record["source_checkpoint_sha256"],
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
        }
    return models, provenance


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
    return candidates[0].parent, json.loads(candidates[0].read_text(encoding="utf-8"))


def _exact_algebra_checks(values, *, exact_names, exact_spatial_transform, torch):
    return {
        "eight_declared_elements": len(exact_names) == 8,
        "identity_is_exact": bool(
            torch.equal(exact_spatial_transform(values, "identity"), values)
        ),
        "input_flip_h_then_v_equals_rot180": bool(
            torch.equal(
                exact_spatial_transform(
                    exact_spatial_transform(values, "flip_h"), "flip_v"
                ),
                exact_spatial_transform(values, "rot180"),
            )
        ),
        "input_flip_v_then_h_equals_rot180": bool(
            torch.equal(
                exact_spatial_transform(
                    exact_spatial_transform(values, "flip_v"), "flip_h"
                ),
                exact_spatial_transform(values, "rot180"),
            )
        ),
        "rot90_after_flip_h_equals_flip_diag": bool(
            torch.equal(
                exact_spatial_transform(
                    exact_spatial_transform(values, "flip_h"), "rot90"
                ),
                exact_spatial_transform(values, "flip_diag"),
            )
        ),
        "rot180_after_flip_h_equals_flip_v": bool(
            torch.equal(
                exact_spatial_transform(
                    exact_spatial_transform(values, "flip_h"), "rot180"
                ),
                exact_spatial_transform(values, "flip_v"),
            )
        ),
        "rot270_after_flip_h_equals_flip_anti_diag": bool(
            torch.equal(
                exact_spatial_transform(
                    exact_spatial_transform(values, "flip_h"), "rot270"
                ),
                exact_spatial_transform(values, "flip_anti_diag"),
            )
        ),
    }


def _rms_ratio(numerator_mse, denominator_mse):
    import torch

    return torch.sqrt(numerator_mse) / (torch.sqrt(denominator_mse) + 1e-8)


def _uint8(values):
    return (
        values
        .detach()
        .float()
        .add(1)
        .div(2)
        .clamp(0, 1)
        .mul(255)
        .round()
        .to(dtype=__import__("torch").uint8)
        .cpu()
        .numpy()
    )


def _extract_payload(destination):
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


def _require_hash(path, expected):
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(f"hash differs for {path}: {actual} != {expected}")


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
