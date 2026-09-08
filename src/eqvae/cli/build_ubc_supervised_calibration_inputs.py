# Copyright 2026 HiperMaximus
# ruff: noqa: C901, DOC201, DOC501, EM101, PLR0913, PLW0717, TRY003
# pyright: reportPrivateUsage=false
"""Stage the small private input dataset for one Spec 0023 calibration phase."""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

from eqvae.cli.build_ubc_supervised_calibration import (
    INPUT_DATASET_SLUGS,
    SPEC_PATH,
    _assets_for_mode,
    _sha256,
    _validate_contract_sources,
    _validate_manifest_audit,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

DATASET_SLUGS: Final = INPUT_DATASET_SLUGS
CONTRACT_NAME: Final = "spec0023_supervised_calibration_input_contract.json"
METADATA_NAME: Final = "dataset-metadata.json"
UPLOAD_ARCHIVE_NAME: Final = "bundle.zip"


def build_input_bundle(
    *,
    repo_root: Path,
    manifest_root: Path,
    output_root: Path,
    package_mode: str,
    selection_audit_path: Path | None = None,
    manifest_audit_path: Path | None = None,
) -> dict[str, object]:
    """Stage exact source plus canonical logical CSVs without any latent binary."""
    if package_mode not in DATASET_SLUGS:
        message = f"Unknown calibration input mode: {package_mode}"
        raise ValueError(message)
    if output_root.exists():
        message = f"Refusing to overwrite {output_root}"
        raise FileExistsError(message)
    assets, manifest_hashes, _ = _assets_for_mode(
        manifest_root=manifest_root,
        package_mode=package_mode,
        selection_audit_path=selection_audit_path,
    )
    manifest_audit_path = manifest_audit_path or (
        manifest_root / "spec0023_supervised_manifest_audit.json"
    )
    manifest_audit_sha256 = _validate_manifest_audit(
        manifest_audit_path,
        manifest_hashes,
        package_mode=package_mode,
    )
    source_files = {
        path.relative_to(repo_root).as_posix(): path
        for path in sorted((repo_root / "src/eqvae").rglob("*.py"))
        if "__pycache__" not in path.parts
    }
    staging = output_root.with_name(f".{output_root.name}.staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        for logical_name, source in source_files.items():
            target = staging / logical_name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        for logical_name, payload in assets.items():
            target = staging / logical_name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        files = {
            path.relative_to(staging).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        contract: dict[str, object] = {
            "schema_version": "spec0023.supervised_calibration_input_contract.v1",
            "package_mode": package_mode,
            "dataset_reference": DATASET_SLUGS[package_mode],
            "spec_sha256": _sha256(repo_root / SPEC_PATH),
            "supervised_manifest_audit_sha256": manifest_audit_sha256,
            "logical_manifest_sha256": manifest_hashes,
            "files": files,
        }
        (staging / CONTRACT_NAME).write_bytes(_canonical_json(contract))
        (staging / METADATA_NAME).write_bytes(
            _canonical_json(_dataset_metadata(package_mode)),
        )
        staging.replace(output_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    validate_input_bundle(
        repo_root=repo_root,
        manifest_root=manifest_root,
        bundle_root=output_root,
        package_mode=package_mode,
        selection_audit_path=selection_audit_path,
        manifest_audit_path=manifest_audit_path,
    )
    return contract


def validate_input_bundle(
    *,
    repo_root: Path,
    manifest_root: Path,
    bundle_root: Path,
    package_mode: str,
    selection_audit_path: Path | None = None,
    manifest_audit_path: Path | None = None,
) -> dict[str, object]:
    """Reject source, manifest, split-surface, or dataset-identity drift."""
    contract = _read_object(bundle_root / CONTRACT_NAME)
    if (
        contract.get("schema_version")
        != "spec0023.supervised_calibration_input_contract.v1"
        or contract.get("package_mode") != package_mode
        or contract.get("dataset_reference") != DATASET_SLUGS[package_mode]
        or contract.get("spec_sha256") != _sha256(repo_root / SPEC_PATH)
    ):
        raise ValueError("Calibration input contract identity differs")
    expected_metadata = _dataset_metadata(package_mode)
    if _read_object(bundle_root / METADATA_NAME) != expected_metadata:
        raise ValueError("Calibration input dataset metadata differs")
    files = cast("dict[str, dict[str, object]]", contract.get("files"))
    observed = {
        path.relative_to(bundle_root).as_posix()
        for path in bundle_root.rglob("*")
        if path.is_file()
    }
    if observed != {*files, CONTRACT_NAME, METADATA_NAME}:
        raise ValueError("Calibration input bundle allow-list differs")
    for logical_name, record in files.items():
        path = bundle_root / logical_name
        if path.stat().st_size != record["bytes"] or _sha256(path) != record["sha256"]:
            message = f"Calibration input file differs: {logical_name}"
            raise ValueError(message)
    assets, manifest_hashes, _ = _assets_for_mode(
        manifest_root=manifest_root,
        package_mode=package_mode,
        selection_audit_path=selection_audit_path,
    )
    if contract.get("logical_manifest_sha256") != manifest_hashes:
        raise ValueError("Calibration input canonical manifest hashes differ")
    for logical_name, payload in assets.items():
        if (bundle_root / logical_name).read_bytes() != payload:
            message = f"Calibration input asset differs: {logical_name}"
            raise ValueError(message)
    manifest_audit_path = manifest_audit_path or (
        manifest_root / "spec0023_supervised_manifest_audit.json"
    )
    if contract.get("supervised_manifest_audit_sha256") != _validate_manifest_audit(
        manifest_audit_path,
        manifest_hashes,
        package_mode=package_mode,
    ):
        raise ValueError("Calibration input manifest authority differs")
    _validate_contract_sources(contract, repo_root=repo_root)
    if package_mode == "sweep" and any(
        "validation" in name or "test" in name for name in assets
    ):
        raise ValueError("Sweep input dataset broadened beyond training")
    if package_mode == "horizon" and any(
        "tissue" in name or "test" in name for name in assets
    ):
        raise ValueError("MIL horizon input broadened beyond WSI learning/resume")
    if package_mode in {
        "width128",
        "class_specific",
        "class_specific_scale_fix",
    } and any(
        "tissue" in name or "test" in name or "resume" in name for name in assets
    ):
        raise ValueError("MIL architecture input broadened beyond WSI learning")
    return contract


def stage_upload_envelope(
    *,
    bundle_root: Path,
    destination: Path,
) -> Path:
    """Wrap the validated nested bundle in Kaggle's flat upload envelope."""
    if destination.exists():
        message = f"Refusing to overwrite {destination}"
        raise FileExistsError(message)
    destination.mkdir(parents=True)
    try:
        shutil.copy2(bundle_root / METADATA_NAME, destination / METADATA_NAME)
        with zipfile.ZipFile(
            destination / UPLOAD_ARCHIVE_NAME,
            mode="x",
            compression=zipfile.ZIP_STORED,
            allowZip64=True,
        ) as archive:
            for path in sorted(bundle_root.rglob("*")):
                if path.is_file() and path.name != METADATA_NAME:
                    archive.write(path, path.relative_to(bundle_root).as_posix())
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return destination


def seal_remote_receipt(
    *,
    bundle_root: Path,
    downloaded_root: Path,
    remote_metadata_path: Path,
    dataset_version: int,
    output_path: Path,
    package_mode: str,
) -> dict[str, object]:
    """Byte-verify the private remote dataset before pinning its receipt."""
    if output_path.exists() or dataset_version < 1:
        raise ValueError("Refusing invalid or replacement calibration input receipt")
    contract_path = bundle_root / CONTRACT_NAME
    contract = _read_object(contract_path)
    files = cast("dict[str, dict[str, object]]", contract["files"])
    required = {*files, CONTRACT_NAME}
    downloaded = {
        path.relative_to(downloaded_root).as_posix(): path
        for path in downloaded_root.rglob("*")
        if path.is_file() and path.name != METADATA_NAME
    }
    if set(downloaded) != required:
        raise ValueError("Downloaded calibration input allow-list differs")
    remote_files: list[dict[str, object]] = []
    for logical_name in sorted(required):
        local = bundle_root / logical_name
        remote = downloaded[logical_name]
        if local.stat().st_size != remote.stat().st_size or _sha256(local) != _sha256(
            remote,
        ):
            message = f"Downloaded calibration input differs: {logical_name}"
            raise ValueError(message)
        remote_files.append({
            "logical_name": logical_name,
            "bytes": local.stat().st_size,
            "sha256": _sha256(local),
        })
    metadata = _read_object(remote_metadata_path)
    info = cast("dict[str, object]", metadata.get("info"))
    owner, slug = DATASET_SLUGS[package_mode].split("/", maxsplit=1)
    if (
        info.get("ownerUser") != owner
        or info.get("datasetSlug") != slug
        or info.get("isPrivate") is not True
    ):
        raise ValueError("Remote calibration input identity/privacy differs")
    receipt: dict[str, object] = {
        "schema_version": "spec0023.supervised_calibration_input_receipt.v1",
        "package_mode": package_mode,
        "dataset_reference": DATASET_SLUGS[package_mode],
        "dataset_version": dataset_version,
        "input_contract_sha256": _sha256(contract_path),
        "remote_files": remote_files,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".json.tmp")
    temporary.write_bytes(_canonical_json(receipt))
    temporary.replace(output_path)
    return receipt


def _dataset_metadata(package_mode: str) -> dict[str, object]:
    title_suffix = {
        "sweep": "sweep v2",
        "confirmation": "confirmation",
        "horizon": "MIL horizon",
        "width128": "MIL width128",
        "class_specific": "MIL class attention",
        "class_specific_scale_fix": "MIL class scale",
    }[package_mode]
    return {
        "id": DATASET_SLUGS[package_mode],
        "title": f"EQVAE Spec 0023 calibration {title_suffix}",
        "description": (
            f"Private immutable train/learning inputs for Spec 0023 {package_mode}."
        ),
        "licenses": [{"name": "unknown"}],
    }


def _canonical_json(value: Mapping[str, object]) -> bytes:
    return f"{json.dumps(value, sort_keys=True, separators=(',', ':'))}\n".encode()


def _read_object(path: Path) -> dict[str, object]:
    value = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(value, dict):
        message = f"Expected JSON object in {path}"
        raise TypeError(message)
    return cast("dict[str, object]", value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_mode", choices=tuple(DATASET_SLUGS))
    parser.add_argument("action", choices=("build", "validate", "stage-upload"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=Path("runs/local/ubc_ocean_supervised_manifests"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--selection-audit", type=Path)
    parser.add_argument("--manifest-audit", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build, validate, or stage the exact mode-specific private input."""
    args = _parser().parse_args(argv)
    repo_root = cast("Path", args.repo_root).resolve()
    manifest_root = cast("Path", args.manifest_root)
    package_mode = cast("str", args.package_mode)
    selection_audit_path = cast("Path | None", args.selection_audit)
    manifest_audit_path = cast("Path | None", args.manifest_audit)
    output_root = cast("Path", args.output_root)
    action = cast("str", args.action)
    if action == "build":
        build_input_bundle(
            repo_root=repo_root,
            manifest_root=manifest_root,
            output_root=output_root,
            package_mode=package_mode,
            selection_audit_path=selection_audit_path,
            manifest_audit_path=manifest_audit_path,
        )
    elif action == "validate":
        validate_input_bundle(
            repo_root=repo_root,
            manifest_root=manifest_root,
            bundle_root=output_root,
            package_mode=package_mode,
            selection_audit_path=selection_audit_path,
            manifest_audit_path=manifest_audit_path,
        )
    else:
        destination = cast("Path | None", args.destination)
        if destination is None:
            raise ValueError("stage-upload requires --destination")
        validate_input_bundle(
            repo_root=repo_root,
            manifest_root=manifest_root,
            bundle_root=output_root,
            package_mode=package_mode,
            selection_audit_path=selection_audit_path,
            manifest_audit_path=manifest_audit_path,
        )
        stage_upload_envelope(bundle_root=output_root, destination=destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
