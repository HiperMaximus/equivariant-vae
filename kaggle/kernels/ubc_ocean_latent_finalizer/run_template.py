# Copyright 2026 HiperMaximus
# ruff: noqa: D103, EM101, EM102, PLC0415, PLW0717, TRY003, TRY300, TRY301
"""Generated disk-safe wrapper for the Spec 0021 latent-store finalizer."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import shutil
import sys
import traceback
import zipfile
from pathlib import Path
from typing import cast

KAGGLE_UBC_OCEAN_LATENT_INFERENCE_READY = True
EMBEDDED_PAYLOAD_B64 = """
$embedded_payload_b64
"""
EMBEDDED_PAYLOAD_ZIP_SHA256 = "$embedded_payload_zip_sha256"
EMBEDDED_PAYLOAD_MANIFEST_SHA256 = "$embedded_payload_manifest_sha256"
EMBEDDED_INFERENCE_CONFIG_B64 = "$embedded_inference_config_b64"
EMBEDDED_INFERENCE_CONFIG_SHA256 = "$embedded_inference_config_sha256"
EMBEDDED_PILOT_AUTHORITY_B64 = "$embedded_pilot_authority_b64"
EMBEDDED_PILOT_AUTHORITY_SHA256 = "$embedded_pilot_authority_sha256"
EMBEDDED_PNG_RESOURCE_AUDIT_B64 = "$embedded_png_resource_audit_b64"
EMBEDDED_PNG_RESOURCE_AUDIT_SHA256 = "$embedded_png_resource_audit_sha256"
INPUT_ROOT = Path("/kaggle/input")
WORKING_ROOT = Path("/kaggle/working")
OUTPUT_ROOT = WORKING_ROOT / "dataset"
PAYLOAD_ROOT = WORKING_ROOT / ".spec0021_finalizer_payload"
PAIR_ROOT = WORKING_ROOT / ".spec0021_finalizer_pairs"
CONFIG_ROOT = WORKING_ROOT / ".spec0021_finalizer_configs"
INPUT_BUNDLE_ROOT = WORKING_ROOT / ".spec0021_finalizer_input"
RECEIPT_PATH = WORKING_ROOT / ".spec0021_finalizer_input_receipt.json"
CONFIG_PATH = WORKING_ROOT / ".spec0021_finalizer_config.json"
FINALIZER_SCHEMA = "spec0021.finalizer_config.v1"
INPUT_DATASET_REFERENCE = "maximusshtefan/eqvae-ubc-ocean-latent-inputs"
KERNEL_SOURCES = tuple(
    f"maximusshtefan/eqvae-ubc-ocean-latent-run-{run:02d}" for run in range(1, 6)
)
REFERENCE_PART_COUNT = 2


def main() -> int:
    success = False
    try:
        payload_root = _extract_payload(PAYLOAD_ROOT)
        payload_src = payload_root / "src"
        sys.path.insert(0, str(payload_src))
        os.environ["PYTHONPATH"] = _pythonpath(
            payload_src,
            os.environ.get("PYTHONPATH", ""),
        )
        config = _write_embedded_config(CONFIG_PATH)
        if config.get("spec_sha256") != _sha256(
            payload_root / "docs/specs/0021-dual-model-wsi-latent-inference.md",
        ):
            raise ValueError("embedded finalizer Spec 0021 hash mismatch")
        input_contract = _find_input_contract(INPUT_ROOT, config)
        _validate_config(config, input_contract)
        _write_input_receipt(config, RECEIPT_PATH)
        _write_production_configs(config, CONFIG_ROOT)
        _prepare_pair_root(INPUT_ROOT, PAIR_ROOT)

        import eqvae
        from eqvae.cli.finalize_ubc_latent_stores import main as finalizer_main

        _assert_import_origin(Path(cast("str", eqvae.__file__)), payload_src)
        status = finalizer_main(
            [
                "--input-contract",
                str(input_contract),
                "--config-root",
                str(CONFIG_ROOT),
                "--input-receipt",
                str(RECEIPT_PATH),
                "--pair-root",
                str(PAIR_ROOT),
                "--output-root",
                str(OUTPUT_ROOT),
            ],
        )
        if status != 0:
            raise RuntimeError("Spec 0021 finalizer returned nonzero status")
        _validate_output_allow_list(OUTPUT_ROOT)
        success = True
        return 0
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 1
    finally:
        for path in (PAYLOAD_ROOT, PAIR_ROOT, CONFIG_ROOT, INPUT_BUNDLE_ROOT):
            shutil.rmtree(path, ignore_errors=True)
        for path in (RECEIPT_PATH, CONFIG_PATH):
            path.unlink(missing_ok=True)
        if not success:
            shutil.rmtree(OUTPUT_ROOT, ignore_errors=True)
            shutil.rmtree(
                OUTPUT_ROOT.with_name(f".{OUTPUT_ROOT.name}.staging"),
                ignore_errors=True,
            )


def _extract_payload(destination: Path) -> Path:
    zip_bytes = base64.b64decode(EMBEDDED_PAYLOAD_B64.encode("ascii"))
    if hashlib.sha256(zip_bytes).hexdigest() != EMBEDDED_PAYLOAD_ZIP_SHA256:
        raise RuntimeError("embedded payload zip hash mismatch")
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        for name in archive.namelist():
            path = Path(name)
            if path.is_absolute() or ".." in path.parts:
                raise RuntimeError(f"unsafe embedded payload path: {name}")
        archive.extractall(destination)
    manifest_bytes = (destination / "payload_manifest.json").read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != EMBEDDED_PAYLOAD_MANIFEST_SHA256:
        raise RuntimeError("embedded payload manifest hash mismatch")
    return destination


def _write_embedded_config(path: Path) -> dict[str, object]:
    payload = base64.b64decode(EMBEDDED_INFERENCE_CONFIG_B64.encode("ascii"))
    if hashlib.sha256(payload).hexdigest() != EMBEDDED_INFERENCE_CONFIG_SHA256:
        raise RuntimeError("embedded finalizer config hash mismatch")
    path.write_bytes(payload)
    value = cast("object", json.loads(payload))
    if not isinstance(value, dict):
        raise TypeError("embedded finalizer config must be an object")
    return cast("dict[str, object]", value)


def _find_input_contract(input_root: Path, config: dict[str, object]) -> Path:
    receipt = config.get("input_dataset_receipt")
    if not isinstance(receipt, dict):
        raise TypeError("finalizer config input receipt must be an object")
    owner, slug = _reference_parts(receipt.get("dataset_reference"), "dataset")
    version = receipt.get("dataset_version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError("finalizer input receipt version must be positive")
    nested_root = input_root / "datasets" / owner / slug
    root = _unique_complete_root(
        candidates=(
            input_root / slug,
            input_root / owner / slug,
            nested_root,
            nested_root / "versions" / str(version),
        ),
        required=(Path("spec0021_input_contract.json"), Path("bundle.zip")),
        input_root=input_root,
        label="immutable Spec 0021 input",
        match_any=True,
    )
    direct = root / "spec0021_input_contract.json"
    upload_archive = root / "bundle.zip"
    if direct.is_file() and not upload_archive.exists():
        return direct
    if upload_archive.is_file() and not direct.exists():
        from eqvae.inference.input_bundle import extract_fresh_upload_archive

        validated = extract_fresh_upload_archive(
            upload_archive,
            INPUT_BUNDLE_ROOT,
            expected_dataset_slug=INPUT_DATASET_REFERENCE,
            expected_provenance_sha256=cast("str", config["input_contract_sha256"]),
        )
        return validated.root / "spec0021_input_contract.json"
    raise ValueError("finalizer input must expose exactly contract or bundle.zip")


def _validate_config(config: dict[str, object], input_contract: Path) -> None:
    if (
        config.get("schema_version") != FINALIZER_SCHEMA
        or config.get("mode") != "finalizer"
        or config.get("kernel_sources") != list(KERNEL_SOURCES)
        or config.get("input_contract_sha256") != _sha256(input_contract)
    ):
        raise ValueError("embedded finalizer config binding mismatch")
    receipt = config.get("input_dataset_receipt")
    if not isinstance(receipt, dict):
        raise TypeError("finalizer config input receipt must be an object")
    if receipt.get("dataset_reference") != INPUT_DATASET_REFERENCE:
        raise ValueError("finalizer input receipt dataset mismatch")


def _write_input_receipt(config: dict[str, object], path: Path) -> None:
    receipt = config["input_dataset_receipt"]
    _write_exclusive(path, _canonical_json_bytes(receipt))


def _write_production_configs(config: dict[str, object], root: Path) -> None:
    raw = config.get("production_configs")
    if not isinstance(raw, dict) or set(raw) != {
        f"run_{run:02d}" for run in range(1, 6)
    }:
        raise ValueError("finalizer must embed exactly five production configs")
    root.mkdir(parents=True, exist_ok=False)
    for run in range(1, 6):
        run_name = f"run_{run:02d}"
        record = raw[run_name]
        if not isinstance(record, dict) or set(record) != {"config", "sha256"}:
            raise ValueError(f"embedded {run_name} config record differs")
        encoded = _canonical_json_bytes(record["config"])
        if hashlib.sha256(encoded).hexdigest() != record["sha256"]:
            raise ValueError(f"embedded {run_name} config SHA-256 mismatch")
        run_root = root / run_name
        run_root.mkdir()
        _write_exclusive(run_root / "spec0021_inference_config.json", encoded)


def _prepare_pair_root(input_root: Path, pair_root: Path) -> None:
    pair_root.mkdir(parents=True, exist_ok=False)
    used: set[Path] = set()
    for run, reference in enumerate(KERNEL_SOURCES, start=1):
        owner, slug = _reference_parts(reference, f"run {run:02d} kernel")
        audit_name = f"spec0021_pair_audit_run_{run:02d}_of_05.json"
        source_root = _unique_complete_root(
            candidates=(input_root / "notebooks" / owner / slug / "dataset",),
            required=(Path(audit_name),),
            input_root=input_root,
            label=f"kernel-output mount for run {run:02d}",
        )
        resolved = source_root.resolve()
        if resolved in used:
            raise ValueError("one kernel-output mount cannot satisfy two runs")
        expected_files = {
            f"normal_vae_mu_run_{run:02d}_of_05.bin",
            f"normal_vae_mu_run_{run:02d}_of_05.json",
            f"so2_vae_mu_run_{run:02d}_of_05.bin",
            f"so2_vae_mu_run_{run:02d}_of_05.json",
            audit_name,
        }
        if {path.name for path in source_root.iterdir()} != expected_files:
            raise ValueError(f"run {run:02d} source allow-list differs")
        if any(
            path.is_symlink() or not path.is_file() for path in source_root.iterdir()
        ):
            raise ValueError(f"run {run:02d} source contains non-physical files")
        used.add(resolved)
        (pair_root / f"run_{run:02d}_of_05").symlink_to(
            source_root,
            target_is_directory=True,
        )


def _unique_complete_root(
    *,
    candidates: tuple[Path, ...],
    required: tuple[Path, ...],
    input_root: Path,
    label: str,
    match_any: bool = False,
) -> Path:
    input_resolved = input_root.resolve()
    matches: dict[Path, Path] = {}
    for candidate in candidates:
        if not candidate.is_dir() or candidate.is_symlink():
            continue
        resolved = candidate.resolve()
        if input_resolved not in resolved.parents:
            raise ValueError(f"{label} escapes /kaggle/input")
        checks = tuple((candidate / relative).is_file() for relative in required)
        if any(checks) if match_any else all(checks):
            matches.setdefault(resolved, candidate)
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {label}; found {len(matches)}")
    return next(iter(matches.values()))


def _reference_parts(reference: object, label: str) -> tuple[str, str]:
    if not isinstance(reference, str):
        raise TypeError(f"{label} reference must be text")
    parts = reference.split("/")
    if len(parts) != REFERENCE_PART_COUNT or any(
        not part or part in {".", ".."} for part in parts
    ):
        raise ValueError(f"{label} reference must be owner/slug")
    return parts[0], parts[1]


def _validate_output_allow_list(output_root: Path) -> None:
    expected_views = {
        f"{task}_{split}_locations.csv"
        for task in ("cancer", "tissue")
        for split in ("train", "validation", "test")
    }
    if {path.name for path in output_root.iterdir()} != {
        "views",
        "spec0021_latent_store_global_audit.json",
    }:
        raise ValueError("finalizer output root differs from exact allow-list")
    views = output_root / "views"
    if (
        views.is_symlink()
        or {path.name for path in views.iterdir()} != expected_views
        or any(path.is_symlink() or not path.is_file() for path in views.iterdir())
    ):
        raise ValueError("finalizer location views differ from exact allow-list")
    audit = output_root / "spec0021_latent_store_global_audit.json"
    if audit.is_symlink() or not audit.is_file():
        raise ValueError("finalizer global audit is missing or symlinked")


def _canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _write_exclusive(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _pythonpath(payload_src: Path, existing: str) -> str:
    return os.pathsep.join(part for part in (str(payload_src), existing) if part)


def _assert_import_origin(module_file: Path, payload_src: Path) -> None:
    if payload_src.resolve() not in module_file.resolve().parents:
        raise RuntimeError(f"eqvae imported outside payload: {module_file}")


if __name__ == "__main__":
    raise SystemExit(main())
