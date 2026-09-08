# Copyright 2026 HiperMaximus
# ruff: noqa: DOC201, DOC501, PLR0913
"""Build guarded pilot and production upload directories for Spec 0021."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

from scripts.build_kaggle_embedded_kernel import BuildArgs, build_run_text

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

CONFIG_SCHEMA: Final = "spec0021.inference_config.v2"
FINALIZER_CONFIG_SCHEMA: Final = "spec0021.finalizer_config.v1"
INPUT_RECEIPT_SCHEMA: Final = "spec0021.input_dataset_receipt.v1"
PILOT_AUTHORITY_SCHEMA: Final = "spec0021.pilot_authority.v2"
RESUME_RECEIPT_SCHEMA: Final = "spec0021.resume_dataset_receipt.v1"
READY_MARKER: Final = "KAGGLE_UBC_OCEAN_LATENT_INFERENCE_READY = True"
DEFAULT_OUTPUT_ROOT = Path("runs/local/ubc_ocean_latent_kernels")
DEFAULT_INPUT_CONTRACT = Path(
    "runs/local/ubc_ocean_latent_input_bundle/spec0021_input_contract.json",
)
DEFAULT_INPUT_RECEIPT = Path(
    "runs/local/ubc_ocean_latent_authority/input_dataset_receipt.json",
)
DEFAULT_PILOT_AUTHORITY = Path(
    "runs/kaggle/ubc_ocean_latent_pilot/dataset/spec0021_pilot_authority.json",
)
DEFAULT_PILOT_RECEIPT = Path(
    "runs/local/ubc_ocean_latent_authority/pilot_receipt.json",
)
TEMPLATE_PATH = Path("kaggle/kernels/ubc_ocean_latent_inference/run_template.py")
FINALIZER_TEMPLATE_PATH = Path(
    "kaggle/kernels/ubc_ocean_latent_finalizer/run_template.py",
)
SPEC_PATH = Path("docs/specs/0021-dual-model-wsi-latent-inference.md")
WORK_HASHES: Final = {
    1: "76cc5f9b86b75b9e46250c80e7b5c98d2f0451a12b38665ae45b055767b7a456",
    2: "11d21482c9d3e083bc5138973c6b09f6b659e7d753853c0290382320e78e6200",
    3: "9414f638abc24963e821de8bbedccaf294a14a7ea768a01687d5b105e634402b",
    4: "e7c5d8d08996e3bac440b5779547b2bbeb4443e7481dcb74998a87aa3054697f",
    5: "5485c44ababeaba9a4e2cc75a1a6b2927d89f10ed83a121dfcb81a68cfc23d6a",
}
SHA256_LENGTH: Final = 64
PRODUCTION_KERNEL_SOURCES: Final = tuple(
    f"maximusshtefan/eqvae-ubc-ocean-latent-run-{run:02d}" for run in range(1, 6)
)
PRODUCTION_CONFIG_FIELDS: Final = frozenset(
    {
        "schema_version",
        "mode",
        "run_number",
        "spec_sha256",
        "input_dataset_receipt",
        "input_contract_sha256",
        "work_manifest_sha256",
        "normal_checkpoint_sha256",
        "so2_checkpoint_sha256",
        "pilot_authority_sha256",
        "selected_recipe",
        "expected_binary_output_bytes",
        "saved_output_limit_bytes",
        "resume_dataset_receipt",
    },
)
NORMAL_CHECKPOINT_SHA256: Final = (
    "f733304e9178e468546113642bdf01e11348570b340c366cf148973083cb9075"
)
SO2_CHECKPOINT_SHA256: Final = (
    "041e0cd7483cb8642bb72eb1b63c3a36774bf9cadd0b659c9d1db6a813c8f4c7"
)
WORK_ROW_COUNTS: Final = {
    1: 121_199,
    2: 119_898,
    3: 118_901,
    4: 118_513,
    5: 120_887,
}
LATENT_RECORD_BYTES: Final = 65_536
LATENT_HEADER_BYTES: Final = 64
KAGGLE_SAVED_OUTPUT_LIMIT_BYTES: Final = 20_000_000_000
PUBLISHED_METADATA_RESERVE_BYTES: Final = 10_000_000


def main(argv: Sequence[str] | None = None) -> int:  # noqa: C901, PLR0914
    """Build one staged pilot or all five production kernel directories."""
    args = _parser().parse_args(argv)
    repo_root = Path(cast("str", args.repo_root)).resolve()
    mode = cast("str", args.mode)
    output_root = _resolve(repo_root, cast("str", args.output_root))
    input_contract_path = _resolve(repo_root, cast("str", args.input_contract))
    input_contract = _read_object(input_contract_path)
    if input_contract.get("schema_version") != "spec0021.input_bundle.v1":
        message = "Input contract schema mismatch"
        raise ValueError(message)
    input_receipt_path = _resolve(repo_root, cast("str", args.input_receipt))
    input_receipt = (
        _validate_input_receipt(_read_object(input_receipt_path))
        if input_receipt_path.is_file()
        else None
    )
    if input_receipt is not None and input_receipt.get(
        "input_contract_sha256",
    ) != _sha256(input_contract_path):
        message = "Input receipt does not bind the selected local contract"
        raise ValueError(message)
    if mode == "finalizer":
        if input_receipt is None:
            message = "Finalizer kernel build requires the pinned input receipt"
            raise ValueError(message)
        _build_finalizer(
            repo_root=repo_root,
            kernel_dir=output_root / "finalizer",
            input_contract_path=input_contract_path,
            input_receipt=input_receipt,
            config_root=output_root,
        )
        return 0
    if mode == "pilot":
        _build_one(
            repo_root=repo_root,
            kernel_dir=output_root / "pilot",
            config=_config(
                repo_root=repo_root,
                mode="pilot",
                run_number=None,
                input_contract_path=input_contract_path,
                input_receipt=input_receipt,
                pilot_authority=None,
                pilot_authority_sha256=None,
                resume_receipt=None,
            ),
        )
        return 0
    pilot_path = _resolve(repo_root, cast("str", args.pilot_authority))
    pilot_authority = _validate_pilot_authority(_read_object(pilot_path))
    pilot_authority_sha256 = _sha256(pilot_path)
    pilot_receipt_path = _resolve(repo_root, cast("str", args.pilot_receipt))
    _validate_pilot_receipt(
        _read_object(pilot_receipt_path),
        authority_sha256=pilot_authority_sha256,
        matrix_sha256=cast("str", pilot_authority["matrix_sha256"]),
    )
    if input_receipt is None:
        message = "Production kernel build requires the pinned input receipt"
        raise ValueError(message)
    if pilot_authority.get("input_dataset_receipt_sha256") != _canonical_hash(
        input_receipt,
    ):
        message = "Pilot authority belongs to a different input dataset receipt"
        raise ValueError(message)
    run_numbers = range(1, 6)
    resume_receipt: Mapping[str, object] | None = None
    if mode == "resume":
        run_number_arg = cast("int | None", args.run_number)
        if run_number_arg is None:
            message = "Resume kernel build requires --run-number"
            raise ValueError(message)
        resume_path_arg = cast("str | None", args.resume_receipt)
        if resume_path_arg is None:
            message = "Resume kernel build requires --resume-receipt"
            raise ValueError(message)
        resume_receipt = _validate_resume_receipt(
            _read_object(_resolve(repo_root, resume_path_arg)),
            run_number=run_number_arg,
            input_bundle_sha256=_sha256(input_contract_path),
        )
        run_numbers = (run_number_arg,)
    for run_number in run_numbers:
        _build_one(
            repo_root=repo_root,
            kernel_dir=output_root / f"run_{run_number:02d}",
            config=_config(
                repo_root=repo_root,
                mode="production",
                run_number=run_number,
                input_contract_path=input_contract_path,
                input_receipt=input_receipt,
                pilot_authority=pilot_authority,
                pilot_authority_sha256=pilot_authority_sha256,
                resume_receipt=resume_receipt,
            ),
        )
    return 0


def _config(
    *,
    repo_root: Path,
    mode: str,
    run_number: int | None,
    input_contract_path: Path,
    input_receipt: Mapping[str, object] | None,
    pilot_authority: Mapping[str, object] | None,
    pilot_authority_sha256: str | None,
    resume_receipt: Mapping[str, object] | None,
) -> dict[str, object]:
    selected_recipe = (
        pilot_authority.get("selected_recipe") if pilot_authority is not None else None
    )
    expected_binary_output_bytes = (
        None
        if run_number is None
        else 2
        * (LATENT_HEADER_BYTES + WORK_ROW_COUNTS[run_number] * LATENT_RECORD_BYTES)
    )
    if (
        expected_binary_output_bytes is not None
        and expected_binary_output_bytes + PUBLISHED_METADATA_RESERVE_BYTES
        > KAGGLE_SAVED_OUTPUT_LIMIT_BYTES
    ):
        message = f"Run {run_number:02d} exceeds Kaggle's saved-output cap"
        raise ValueError(message)
    return {
        "schema_version": CONFIG_SCHEMA,
        "mode": mode,
        "run_number": run_number,
        "spec_sha256": _sha256(repo_root / SPEC_PATH),
        "input_dataset_receipt": input_receipt,
        "input_contract_sha256": _sha256(input_contract_path),
        "work_manifest_sha256": (
            None if run_number is None else WORK_HASHES[run_number]
        ),
        "normal_checkpoint_sha256": NORMAL_CHECKPOINT_SHA256,
        "so2_checkpoint_sha256": SO2_CHECKPOINT_SHA256,
        "pilot_authority_sha256": pilot_authority_sha256,
        "selected_recipe": selected_recipe,
        "expected_binary_output_bytes": expected_binary_output_bytes,
        "saved_output_limit_bytes": (
            None if run_number is None else KAGGLE_SAVED_OUTPUT_LIMIT_BYTES
        ),
        "resume_dataset_receipt": resume_receipt,
    }


def _build_one(
    *,
    repo_root: Path,
    kernel_dir: Path,
    config: Mapping[str, object],
) -> None:
    kernel_dir.mkdir(parents=True, exist_ok=True)
    mode = cast("str", config["mode"])
    run_number = cast("int | None", config["run_number"])
    suffix = "pilot" if mode == "pilot" else f"run-{run_number:02d}"
    metadata = {
        "id": f"maximusshtefan/eqvae-ubc-ocean-latent-{suffix}",
        "title": f"eqvae UBC-OCEAN latent {suffix}",
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": _dataset_sources(config),
        "competition_sources": ["UBC-OCEAN"],
        "kernel_sources": [],
        "model_sources": [],
    }
    _write_json(kernel_dir / "kernel-metadata.json", metadata, compact=False)
    _write_json(
        kernel_dir / "spec0021_inference_config.json",
        config,
        compact=True,
    )
    build_args = BuildArgs(
        repo_root=repo_root,
        kernel_dir=kernel_dir,
        template_path=repo_root / TEMPLATE_PATH,
        output_run_path=kernel_dir / "run.py",
        verify_only=False,
        allow_dirty=True,
        ready_marker=READY_MARKER,
    )
    (kernel_dir / "run.py").write_text(build_run_text(build_args), encoding="utf-8")


def _build_finalizer(
    *,
    repo_root: Path,
    kernel_dir: Path,
    input_contract_path: Path,
    input_receipt: Mapping[str, object],
    config_root: Path,
) -> None:
    production_configs: dict[str, object] = {}
    shared_bindings: dict[str, object] | None = None
    spec_sha256 = _sha256(repo_root / SPEC_PATH)
    input_contract_sha256 = _sha256(input_contract_path)
    for run_number in range(1, 6):
        config_path = (
            config_root / f"run_{run_number:02d}" / "spec0021_inference_config.json"
        )
        config = _read_object(config_path)
        observed_shared = _validate_finalizer_production_config(
            config,
            run_number=run_number,
            spec_sha256=spec_sha256,
            input_contract_sha256=input_contract_sha256,
            input_receipt=input_receipt,
        )
        if shared_bindings is None:
            shared_bindings = observed_shared
        elif observed_shared != shared_bindings:
            message = "Finalizer production configs have inconsistent shared bindings"
            raise ValueError(message)
        if config_path.read_bytes() != _canonical_json_bytes(config):
            message = f"Finalizer production config {run_number:02d} is not canonical"
            raise ValueError(message)
        production_configs[f"run_{run_number:02d}"] = {
            "config": config,
            "sha256": _sha256(config_path),
        }
    finalizer_config: dict[str, object] = {
        "schema_version": FINALIZER_CONFIG_SCHEMA,
        "mode": "finalizer",
        "spec_sha256": spec_sha256,
        "input_contract_sha256": input_contract_sha256,
        "input_dataset_receipt": input_receipt,
        "kernel_sources": list(PRODUCTION_KERNEL_SOURCES),
        "production_configs": production_configs,
    }
    kernel_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "id": "maximusshtefan/eqvae-ubc-ocean-latent-finalizer",
        "title": "eqvae UBC-OCEAN latent finalizer",
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "false",
        "enable_internet": "true",
        "dataset_sources": _dataset_sources(
            {
                "input_dataset_receipt": input_receipt,
                "resume_dataset_receipt": None,
            },
        ),
        "competition_sources": [],
        "kernel_sources": list(PRODUCTION_KERNEL_SOURCES),
        "model_sources": [],
    }
    _write_json(kernel_dir / "kernel-metadata.json", metadata, compact=False)
    _write_json(
        kernel_dir / "spec0021_inference_config.json",
        finalizer_config,
        compact=True,
    )
    build_args = BuildArgs(
        repo_root=repo_root,
        kernel_dir=kernel_dir,
        template_path=repo_root / FINALIZER_TEMPLATE_PATH,
        output_run_path=kernel_dir / "run.py",
        verify_only=False,
        allow_dirty=True,
        ready_marker=READY_MARKER,
    )
    (kernel_dir / "run.py").write_text(build_run_text(build_args), encoding="utf-8")


def _validate_finalizer_production_config(
    config: Mapping[str, object],
    *,
    run_number: int,
    spec_sha256: str,
    input_contract_sha256: str,
    input_receipt: Mapping[str, object],
) -> dict[str, object]:
    if frozenset(config) != PRODUCTION_CONFIG_FIELDS:
        message = f"Finalizer production config {run_number:02d} fields differ"
        raise ValueError(message)
    expected = {
        "schema_version": CONFIG_SCHEMA,
        "mode": "production",
        "run_number": run_number,
        "spec_sha256": spec_sha256,
        "input_dataset_receipt": input_receipt,
        "input_contract_sha256": input_contract_sha256,
        "work_manifest_sha256": WORK_HASHES[run_number],
        "normal_checkpoint_sha256": NORMAL_CHECKPOINT_SHA256,
        "so2_checkpoint_sha256": SO2_CHECKPOINT_SHA256,
    }
    if any(config.get(name) != value for name, value in expected.items()):
        message = f"Finalizer production config {run_number:02d} binding mismatch"
        raise ValueError(message)
    raw_resume = config.get("resume_dataset_receipt")
    if raw_resume is not None:
        if not isinstance(raw_resume, dict):
            message = f"Finalizer production config {run_number:02d} resume is invalid"
            raise TypeError(message)
        _validate_resume_receipt(
            cast("dict[str, object]", raw_resume),
            run_number=run_number,
            input_bundle_sha256=input_contract_sha256,
        )
    _require_sha256(
        config.get("pilot_authority_sha256"),
        f"production config {run_number:02d} pilot_authority_sha256",
    )
    recipe = config.get("selected_recipe")
    if not isinstance(recipe, dict) or not recipe:
        message = f"Finalizer production config {run_number:02d} recipe is invalid"
        raise ValueError(message)
    expected_binary_output_bytes = 2 * (
        LATENT_HEADER_BYTES + WORK_ROW_COUNTS[run_number] * LATENT_RECORD_BYTES
    )
    if config.get("expected_binary_output_bytes") != expected_binary_output_bytes:
        message = f"Finalizer production config {run_number:02d} output size is invalid"
        raise ValueError(message)
    if config.get("saved_output_limit_bytes") != KAGGLE_SAVED_OUTPUT_LIMIT_BYTES:
        message = f"Finalizer production config {run_number:02d} cap is invalid"
        raise ValueError(message)
    if (
        expected_binary_output_bytes + PUBLISHED_METADATA_RESERVE_BYTES
        > KAGGLE_SAVED_OUTPUT_LIMIT_BYTES
    ):
        message = f"Finalizer production config {run_number:02d} exceeds output cap"
        raise ValueError(message)
    return {
        name: config[name]
        for name in (
            "input_dataset_receipt",
            "normal_checkpoint_sha256",
            "so2_checkpoint_sha256",
            "pilot_authority_sha256",
            "selected_recipe",
            "saved_output_limit_bytes",
        )
    }


def _dataset_sources(config: Mapping[str, object]) -> list[str]:
    raw_receipt = config.get("input_dataset_receipt")
    if raw_receipt is None:
        return []
    receipt = cast("Mapping[str, object]", raw_receipt)
    reference = receipt.get("dataset_reference")
    if not isinstance(reference, str) or not reference:
        message = "Input receipt has no dataset_reference"
        raise ValueError(message)
    sources = [reference]
    raw_resume = config.get("resume_dataset_receipt")
    if raw_resume is not None:
        resume = cast("Mapping[str, object]", raw_resume)
        resume_reference = resume.get("dataset_reference")
        if not isinstance(resume_reference, str) or not resume_reference:
            message = "Resume receipt has no dataset_reference"
            raise ValueError(message)
        sources.append(resume_reference)
    return sources


def _validate_input_receipt(payload: dict[str, object]) -> dict[str, object]:
    if payload.get("schema_version") != INPUT_RECEIPT_SCHEMA:
        message = "Input dataset receipt schema mismatch"
        raise ValueError(message)
    _dataset_sources({"input_dataset_receipt": payload, "resume_dataset_receipt": None})
    version = payload.get("dataset_version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        message = "Input dataset receipt version must be positive"
        raise ValueError(message)
    return payload


def _validate_pilot_authority(payload: dict[str, object]) -> dict[str, object]:
    expected: dict[str, object] = {
        "schema_version": PILOT_AUTHORITY_SCHEMA,
        "status": "smoke_passed",
        "smoke_passed": True,
        "smoke_wsi_id": 15_188,
        "smoke_patch_count": 16,
        "selected_recipe": {
            "batch_size": 8,
            "d2h": "synchronous",
            "numeric": "FP32",
            "execution": "eager",
        },
    }
    if any(payload.get(name) != value for name, value in expected.items()):
        message = "Pilot authority does not certify the locked 16-patch smoke"
        raise ValueError(message)
    _require_sha256(payload.get("matrix_sha256"), "pilot matrix hash")
    _require_sha256(
        payload.get("input_dataset_receipt_sha256"),
        "pilot input-receipt hash",
    )
    return payload


def _validate_pilot_receipt(
    payload: dict[str, object],
    *,
    authority_sha256: str,
    matrix_sha256: str,
) -> None:
    if (
        payload.get("schema_version") != "spec0021.pilot_receipt.v1"
        or payload.get("status") != "smoke_passed"
        or payload.get("authority_sha256") != authority_sha256
        or payload.get("matrix_sha256") != matrix_sha256
    ):
        message = "Pilot receipt does not bind the validated tiny smoke artifacts"
        raise ValueError(message)


def _validate_resume_receipt(
    payload: dict[str, object],
    *,
    run_number: int,
    input_bundle_sha256: str,
) -> dict[str, object]:
    expected_reference = (
        f"maximusshtefan/eqvae-ubc-ocean-latent-run-{run_number:02d}-resume"
    )
    if (
        payload.get("schema_version") != RESUME_RECEIPT_SCHEMA
        or payload.get("dataset_reference") != expected_reference
        or payload.get("run_number") != run_number
        or payload.get("input_bundle_sha256") != input_bundle_sha256
        or payload.get("work_manifest_sha256") != WORK_HASHES[run_number]
    ):
        message = "Resume dataset receipt identity mismatch"
        raise ValueError(message)
    version = payload.get("dataset_version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        message = "Resume dataset receipt version must be positive"
        raise ValueError(message)
    for name in (
        "provenance_sha256",
        "run_config_sha256",
        "remote_listing_sha256",
    ):
        value = payload.get(name)
        if (
            not isinstance(value, str)
            or len(value) != SHA256_LENGTH
            or any(char not in "0123456789abcdef" for char in value)
        ):
            message = f"Resume dataset receipt {name} is invalid"
            raise ValueError(message)
    remote_files = payload.get("remote_files")
    if not isinstance(remote_files, list) or not remote_files:
        message = "Resume dataset receipt remote_files is invalid"
        raise ValueError(message)
    return payload


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != SHA256_LENGTH
        or any(char not in "0123456789abcdef" for char in value)
    ):
        message = f"{label} must be a lowercase SHA-256"
        raise ValueError(message)
    return value


def _read_object(path: Path) -> dict[str, object]:
    payload = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(payload, dict):
        message = f"Expected JSON object in {path}"
        raise TypeError(message)
    return cast("dict[str, object]", payload)


def _write_json(path: Path, payload: Mapping[str, object], *, compact: bool) -> None:
    separators = (",", ":") if compact else None
    serialized = json.dumps(
        payload,
        indent=None if compact else 2,
        sort_keys=True,
        separators=separators,
    )
    text = f"{serialized}\n"
    path.write_text(text, encoding="utf-8")


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _resolve(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=("pilot", "production-all", "resume", "finalizer"),
    )
    parser.add_argument("--repo-root", default=str(Path.cwd()))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--input-contract", default=str(DEFAULT_INPUT_CONTRACT))
    parser.add_argument("--input-receipt", default=str(DEFAULT_INPUT_RECEIPT))
    parser.add_argument("--pilot-authority", default=str(DEFAULT_PILOT_AUTHORITY))
    parser.add_argument("--pilot-receipt", default=str(DEFAULT_PILOT_RECEIPT))
    parser.add_argument("--run-number", type=int, choices=range(1, 6))
    parser.add_argument("--resume-receipt")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
