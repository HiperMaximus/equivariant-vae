# Copyright 2026 HiperMaximus
# ruff: noqa: ANN202, BLE001, COM812, D103, E501, EM101, EM102, FBT003, INP001, PLC0415, PLR0913, PLR0914, PLR2004, TRY003
"""Generated GPU wrapper for the exploratory fixed25 dense rotation population."""

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
KAGGLE_FIXED25_ROTATION_POPULATION_READY = True
EMBEDDED_PAYLOAD_B64 = """
$embedded_payload_b64
"""
EMBEDDED_PAYLOAD_ZIP_SHA256 = "$embedded_payload_zip_sha256"
EMBEDDED_PAYLOAD_MANIFEST_SHA256 = "$embedded_payload_manifest_sha256"
# fmt: on

INPUT_ROOT = Path("/kaggle/input")
WORKING_ROOT = Path("/kaggle/working")
PRIVATE_ROOT = WORKING_ROOT / ".rotation_population_payload"
OUTPUT_ROOT = WORKING_ROOT / "fixed25_rotation_population"
PATCH_BYTES = 3 * 256 * 256
HEADER_BYTES = 64
MODEL_KINDS = {
    "normal_vae": "non_eq_vae_translatable",
    "so2_vae": "so2_vae_fixed",
}


def main() -> int:
    try:
        return _run()
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        shutil.rmtree(PRIVATE_ROOT, ignore_errors=True)


def _run() -> int:
    payload_root = _extract_payload(PRIVATE_ROOT)
    sys.path.insert(0, str(payload_root / "src"))
    import numpy as np
    import torch

    from eqvae.artifacts.rotation_orbits import (
        LATENT_DISK_RADIUS,
        centered_disk_mask,
        collect_dense_mu_population,
        render_dense_orbit_population_png,
        summarize_dense_orbit_population,
    )
    from eqvae.evaluation.vae_test import sha256_file, state_dict_sha256
    from eqvae.models.registry import build_model

    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA GPU is required")
    device = torch.device("cuda:0")
    selector = json.loads(
        (
            payload_root
            / "runs/kaggle/fixed25_selector/fixed_25_validation_patches.json"
        ).read_text(encoding="utf-8")
    )
    patches = _load_fixed25(selector, np=np, torch=torch).to(device)
    bundle_root, contract = _find_weight_bundle()
    models = {}
    for branch, kind in MODEL_KINDS.items():
        record = contract["weights"][branch]
        state_path = bundle_root / f"{branch}_state.pt"
        if (
            state_path.stat().st_size != record["state_file_bytes"]
            or sha256_file(state_path) != record["state_file_sha256"]
        ):
            raise RuntimeError(f"state file differs for {branch}")
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        if state_dict_sha256(state) != record["state_dict_sha256"]:
            raise RuntimeError(f"state dict differs for {branch}")
        model = build_model(kind)
        model.load_state_dict(state, strict=True)
        if (
            sum(parameter.numel() for parameter in model.parameters())
            != record["parameter_count"]
        ):
            raise RuntimeError(f"parameter count differs for {branch}")
        models[branch] = model.to(device).eval().requires_grad_(False)

    angles = tuple(range(360))
    mask = centered_disk_mask(32, radius=LATENT_DISK_RADIUS)
    started = time.perf_counter()
    summaries = {}
    for branch in ("normal_vae", "so2_vae"):
        values = collect_dense_mu_population(
            model=models[branch],
            patches=patches,
            angles_degrees=angles,
            batch_size=32,
        )
        summaries[branch] = summarize_dense_orbit_population(
            values,
            angles_degrees=angles,
            mask=mask,
        )
        del values
        torch.cuda.empty_cache()
    elapsed = time.perf_counter() - started
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    render_dense_orbit_population_png(
        path=OUTPUT_ROOT / "07-all25-latent-orbits.png",
        normal=summaries["normal_vae"],
        so2=summaries["so2_vae"],
    )
    _write_result(
        OUTPUT_ROOT / "07-all25-latent-orbits.json",
        summaries=summaries,
        contract=contract,
        selector=selector,
        elapsed=elapsed,
        torch=torch,
    )
    return 0


def _extract_payload(destination: Path) -> Path:
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
    manifest = destination / "payload_manifest.json"
    if (
        hashlib.sha256(manifest.read_bytes()).hexdigest()
        != EMBEDDED_PAYLOAD_MANIFEST_SHA256
    ):
        raise RuntimeError("embedded payload manifest hash mismatch")
    return destination


def _find_weight_bundle() -> tuple[Path, dict[str, object]]:
    candidates = list(INPUT_ROOT.rglob("spec0045_vae_test_input.json"))
    for contract_path in candidates:
        root = contract_path.parent
        if (root / "normal_vae_state.pt").is_file() and (
            root / "so2_vae_state.pt"
        ).is_file():
            return root, json.loads(contract_path.read_text(encoding="utf-8"))
    raise RuntimeError("frozen VAE weight bundle was not found")


def _load_fixed25(selector: dict[str, object], *, np: object, torch: object):
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


def _write_result(
    path: Path,
    *,
    summaries: dict[str, object],
    contract: dict[str, object],
    selector: dict[str, object],
    elapsed: float,
    torch: object,
) -> None:
    import numpy as np

    normal = summaries["normal_vae"]
    so2 = summaries["so2_vae"]

    def payload(summary: object) -> dict[str, object]:
        return {
            "pca_explained_variance": summary.pca_explained_variance.tolist(),
            "local_linearity_ratio": summary.local_linearity_ratio.tolist(),
            "step_size_cv": summary.step_size_cv.tolist(),
            "median": {
                "pca_explained_variance": float(
                    np.median(summary.pca_explained_variance)
                ),
                "local_linearity_ratio": float(
                    np.median(summary.local_linearity_ratio)
                ),
                "step_size_cv": float(np.median(summary.step_size_cv)),
            },
        }

    document = {
        "schema": "spec0038.fixed25_dense_orbit_population.v1",
        "label": "Fixed validation 25; post-hoc continuous-angle exploratory visualization; not sealed-test evaluation.",
        "angles_degrees": list(range(360)),
        "display_endpoint_degrees": 360,
        "batch_size": 32,
        "normal": payload(normal),
        "so2": payload(so2),
        "paired_favorable_counts": {
            "so2_lower_local_linearity_ratio": int(
                np.sum(so2.local_linearity_ratio < normal.local_linearity_ratio)
            ),
            "so2_lower_step_size_cv": int(
                np.sum(so2.step_size_cv < normal.step_size_cv)
            ),
            "so2_higher_pca_explained_variance": int(
                np.sum(so2.pca_explained_variance > normal.pca_explained_variance)
            ),
        },
        "measurement": {
            "pca": "one raw-score PCA fit per model and patch over all 360 angles; display only",
            "local_linearity_ratio": "cyclic RMS second difference divided by cyclic RMS first difference on disk-masked raw mu; lower is locally smoother at 1 degree",
            "step_size_cv": "coefficient of variation of cyclic one-degree disk-masked raw-mu RMS; lower is more uniform",
            "inference": "descriptive fixed-validation evidence only; no bootstrap, model selection, or sealed-test claim",
        },
        "source": {
            "weight_dataset_reference": contract["dataset_reference"],
            "normal_checkpoint_sha256": contract["weights"]["normal_vae"][
                "source_checkpoint_sha256"
            ],
            "so2_checkpoint_sha256": contract["weights"]["so2_vae"][
                "source_checkpoint_sha256"
            ],
            "selector_seed": selector["selector_seed"],
            "patch_sha256": [row["patch_sha256"] for row in selector["selectors"]],
        },
        "runtime": {
            "elapsed_seconds": elapsed,
            "torch_version": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(0),
        },
    }
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    raise SystemExit(main())
