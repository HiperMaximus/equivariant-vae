# Copyright 2026 HiperMaximus
"""Immutable fresh-input and resume bundles for Spec 0021 inference."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, cast

from eqvae.data.latent_shards import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_TASK_MANIFEST_SHA256,
    EXPECTED_UNION_SHA256,
    EXPECTED_WORK_MANIFEST_SHA256,
    LATENT_RECORD_BYTES,
    LATENT_SHARD_HEADER_SIZE,
    LatentShardHeader,
    WorkManifest,
    load_work_manifest,
    parse_latent_shard_header,
    validate_latent_artifact,
)
from eqvae.inference.dual_writer import INCOMPLETE_SCHEMA, WORKER_RESUME_SCHEMA

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]
type ModelName = Literal["normal_vae", "so2_vae"]
type ResumeWindow = Literal[
    "partial_with_state",
    "final_with_state",
    "complete_with_stale_state",
    "complete",
]

FRESH_BUNDLE_SCHEMA: Final = "spec0021.input_bundle.v1"
RESUME_BUNDLE_SCHEMA: Final = "spec0021.resume_bundle.v1"
DATASET_METADATA_FILENAME: Final = "dataset-metadata.json"
FRESH_PROVENANCE_FILENAME: Final = "spec0021_input_contract.json"
RESUME_PROVENANCE_FILENAME: Final = "spec0021_resume_contract.json"
FRESH_DATASET_TITLE: Final = "EQVAE Spec 0021 immutable inference inputs"
RESUME_DATASET_TITLE_PREFIX: Final = "EQVAE Spec 0021 inference resume run"
NORMAL_CHECKPOINT_NAME: Final = "checkpoints/normal_vae_step_060000.pt"
SO2_CHECKPOINT_NAME: Final = "checkpoints/so2_vae_step_060000.pt"
UNION_MANIFEST_NAME: Final = "manifests/union_patch_manifest.csv"
# Frozen creation provenance for the published immutable-input v2 bytes.
# Current execution policy is bound separately by inference config.spec_sha256.
SPECIFICATION_SHA256: Final = {
    "spec0019": "731351e8aef519ae9b2bbcbf3f079fa6f121a2cbdfbbeecb7443f59a3dd1aa2c",
    "spec0020": "6dbf4b1b60538764df8e11fb86be19005e059a6ec3dceed90b74ea959b9292ae",
    "spec0021": "b3d471805b39406aa5e0e661ec5bf451d3d47cdb473b06aaae1f9b9568574317",
}
_SHA256_LENGTH: Final = 64
_HASH_CHUNK_BYTES: Final = 8 * 1024 * 1024
_ZIP_MODE_MASK: Final = 0o170000
_ZIP_SYMLINK_MODE: Final = 0o120000
_DATASET_SLUG = re.compile(r"^[a-z0-9][a-z0-9-]*/[a-z0-9][a-z0-9-]*$")
_MODEL_NAMES: Final[tuple[ModelName, ...]] = ("normal_vae", "so2_vae")
FRESH_UPLOAD_ARCHIVE_FILENAME: Final = "bundle.zip"


@dataclass(frozen=True)
class BundleFile:
    """One logical bundle file bound to exact immutable bytes."""

    logical_name: str
    size: int
    sha256: str


@dataclass(frozen=True)
class ValidatedInputBundle:
    """Validated immutable fresh-input bundle identity."""

    root: Path
    provenance_sha256: str
    dataset_slug: str
    files: tuple[BundleFile, ...]


@dataclass(frozen=True)
class ResumeBundleAuthority:
    """Values embedded before a read-only resume attachment may be consumed."""

    provenance_sha256: str
    dataset_slug: str
    dataset_version: int
    run_number: int
    input_bundle_sha256: str
    run_config_sha256: str
    work_manifest_sha256: str


@dataclass(frozen=True)
class ValidatedResumeBundle:
    """Validated resume attachment and its two Spec 0020 writer windows."""

    root: Path
    authority: ResumeBundleAuthority
    files: tuple[BundleFile, ...]
    writer_windows: Mapping[ModelName, ResumeWindow]


def stage_fresh_upload_archive(
    bundle_root: Path,
    destination: Path,
    *,
    expected_dataset_slug: str,
) -> Path:
    """Wrap the validated directory bundle in Kaggle's flat-file upload format.

    Returns:
        Path to the staged ``bundle.zip``.

    Raises:
        FileExistsError: If the destination or its staging path already exists.
        ValueError: If the source bundle differs from its sealed contract.

    """
    validated = validate_fresh_input_bundle(
        bundle_root,
        expected_dataset_slug=expected_dataset_slug,
    )
    if destination.exists():
        message = f"Fresh upload envelope already exists: {destination}"
        raise FileExistsError(message)
    staging = destination.with_name(f".{destination.name}.staging")
    if staging.exists():
        message = f"Stale fresh upload staging path exists: {staging}"
        raise FileExistsError(message)
    staging.mkdir(parents=True)
    shutil.copy2(
        bundle_root / DATASET_METADATA_FILENAME,
        staging / DATASET_METADATA_FILENAME,
    )
    archive_path = staging / FRESH_UPLOAD_ARCHIVE_FILENAME
    logical_names = sorted(
        [file.logical_name for file in validated.files] + [FRESH_PROVENANCE_FILENAME],
    )
    with zipfile.ZipFile(
        archive_path,
        mode="x",
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
    ) as archive:
        for logical_name in logical_names:
            source = bundle_root / logical_name
            if source.is_symlink() or not source.is_file():
                message = f"Fresh bundle member is not a physical file: {logical_name}"
                raise ValueError(message)
            archive.write(source, arcname=logical_name)
    _fsync_file(archive_path)
    _fsync_directory(staging)
    staging.replace(destination)
    _fsync_directory(destination.parent)
    return destination / FRESH_UPLOAD_ARCHIVE_FILENAME


def extract_fresh_upload_archive(
    archive_path: Path,
    destination: Path,
    *,
    expected_dataset_slug: str,
    expected_provenance_sha256: str,
) -> ValidatedInputBundle:
    """Safely extract one flat Kaggle upload archive and validate every member.

    Returns:
        The extracted and fully validated immutable input bundle.

    Raises:
        FileExistsError: If the private extraction destination already exists.
        ValueError: If archive structure or sealed member bytes differ.

    """
    if destination.exists():
        message = f"Fresh archive destination already exists: {destination}"
        raise FileExistsError(message)
    staging = destination.with_name(f".{destination.name}.staging")
    if staging.exists():
        message = f"Stale fresh archive staging path exists: {staging}"
        raise FileExistsError(message)
    if archive_path.is_symlink() or not archive_path.is_file():
        message = f"Fresh upload archive is not a physical file: {archive_path}"
        raise ValueError(message)
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)) or any(
            _fresh_archive_member_is_unsafe(info) for info in infos
        ):
            message = "Fresh upload archive contains unsafe or duplicate members"
            raise ValueError(message)
        sealed_files = _fresh_archive_sealed_files(
            archive,
            expected_provenance_sha256=expected_provenance_sha256,
        )
        expected_names = {*sealed_files, FRESH_PROVENANCE_FILENAME}
        if set(names) != expected_names:
            message = "Fresh upload archive member allow-list differs"
            raise ValueError(message)
        _extract_fresh_archive_members(archive, infos, sealed_files, staging)
    validated = validate_fresh_input_bundle(
        staging,
        expected_dataset_slug=expected_dataset_slug,
        expected_provenance_sha256=expected_provenance_sha256,
    )
    staging.replace(destination)
    _fsync_directory(destination.parent)
    return ValidatedInputBundle(
        root=destination,
        provenance_sha256=validated.provenance_sha256,
        dataset_slug=validated.dataset_slug,
        files=validated.files,
    )


def _fresh_archive_member_is_unsafe(info: zipfile.ZipInfo) -> bool:
    path = Path(info.filename)
    mode = (info.external_attr >> 16) & _ZIP_MODE_MASK
    return (
        info.is_dir()
        or path.is_absolute()
        or ".." in path.parts
        or mode == _ZIP_SYMLINK_MODE
    )


def _fresh_archive_sealed_files(
    archive: zipfile.ZipFile,
    *,
    expected_provenance_sha256: str,
) -> dict[str, object]:
    try:
        contract_info = archive.getinfo(FRESH_PROVENANCE_FILENAME)
    except KeyError as error:
        message = "Fresh upload archive lacks its input contract"
        raise ValueError(message) from error
    if contract_info.file_size > 1024 * 1024:
        message = "Fresh upload archive contract is unexpectedly large"
        raise ValueError(message)
    contract_bytes = archive.read(contract_info)
    if hashlib.sha256(contract_bytes).hexdigest() != expected_provenance_sha256:
        message = "Fresh upload archive contract SHA-256 differs"
        raise ValueError(message)
    contract = cast("object", json.loads(contract_bytes))
    if not isinstance(contract, dict):
        message = "Fresh upload archive contract schema is invalid"
        raise TypeError(message)
    typed_contract = cast("dict[str, object]", contract)
    sealed_files = typed_contract.get("files")
    if not isinstance(sealed_files, dict):
        message = "Fresh upload archive contract schema is invalid"
        raise TypeError(message)
    return cast("dict[str, object]", sealed_files)


def _extract_fresh_archive_members(
    archive: zipfile.ZipFile,
    infos: Sequence[zipfile.ZipInfo],
    sealed_files: Mapping[str, object],
    staging: Path,
) -> None:
    staging.mkdir(parents=True)
    try:
        for info in infos:
            _validate_fresh_archive_member_size(info, sealed_files)
            target = staging / info.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output, length=_HASH_CHUNK_BYTES)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _validate_fresh_archive_member_size(
    info: zipfile.ZipInfo,
    sealed_files: Mapping[str, object],
) -> None:
    record = sealed_files.get(info.filename)
    if (
        isinstance(record, dict)
        and cast("dict[str, object]", record).get(
            "bytes",
        )
        != info.file_size
    ):
        message = f"Fresh archive member size differs: {info.filename}"
        raise ValueError(message)


def stage_fresh_input_bundle(  # noqa: PLR0913
    destination: Path,
    *,
    work_manifests: Mapping[int, Path],
    union_manifest: Path,
    task_manifests: Mapping[str, Path],
    normal_checkpoint: Path,
    so2_checkpoint: Path,
    dataset_slug: str,
) -> ValidatedInputBundle:
    """Stage the exact five manifests and two frozen checkpoints atomically.

    Returns:
        The fully revalidated staged bundle.

    Raises:
        ValueError: If identities, hashes, files, or dataset metadata differ.

    """
    _validate_dataset_slug(dataset_slug)
    if set(work_manifests) != set(range(1, 6)):
        message = "Fresh input bundle requires exactly work manifests 1 through 5"
        raise ValueError(message)
    sources: dict[str, Path] = {}
    for run_number in range(1, 6):
        path = work_manifests[run_number]
        expected_hash = EXPECTED_WORK_MANIFEST_SHA256[run_number]
        load_work_manifest(
            path,
            run_number=run_number,
            expected_sha256=expected_hash,
        )
        sources[_work_manifest_name(run_number)] = path
    load_work_manifest(
        union_manifest,
        run_number=0,
        expected_sha256=EXPECTED_UNION_SHA256,
        require_run_basename=False,
    )
    sources[UNION_MANIFEST_NAME] = union_manifest
    if set(task_manifests) != set(EXPECTED_TASK_MANIFEST_SHA256):
        message = "Fresh input bundle requires exactly six task manifests"
        raise ValueError(message)
    for task_split, expected_hash in EXPECTED_TASK_MANIFEST_SHA256.items():
        task_path = task_manifests[task_split]
        if task_path.name != f"{task_split}.csv":
            message = f"Unexpected task-manifest basename: {task_path.name}"
            raise ValueError(message)
        _require_file_hash(task_path, expected_hash, task_split)
        sources[_task_manifest_name(task_split)] = task_path
    checkpoint_sources = {
        NORMAL_CHECKPOINT_NAME: (
            normal_checkpoint,
            EXPECTED_CHECKPOINT_SHA256["normal_vae"],
        ),
        SO2_CHECKPOINT_NAME: (
            so2_checkpoint,
            EXPECTED_CHECKPOINT_SHA256["so2_vae"],
        ),
    }
    for logical_name, (path, expected_hash) in checkpoint_sources.items():
        _require_file_hash(path, expected_hash, logical_name)
        sources[logical_name] = path

    def populate(staging: Path) -> None:
        records = list(_copy_sources(sources, staging))
        metadata = _dataset_metadata(
            slug=dataset_slug,
            title=FRESH_DATASET_TITLE,
            description="Private immutable Spec 0021 inference inputs.",
        )
        metadata_bytes = _canonical_json_bytes(metadata)
        _write_file_fsync(staging / DATASET_METADATA_FILENAME, metadata_bytes)
        records.append(
            BundleFile(
                DATASET_METADATA_FILENAME,
                len(metadata_bytes),
                hashlib.sha256(metadata_bytes).hexdigest(),
            ),
        )
        records.sort(key=lambda record: record.logical_name)
        payload: JsonObject = {
            "checkpoint_sha256": {
                str(name): value for name, value in EXPECTED_CHECKPOINT_SHA256.items()
            },
            "dataset_slug": dataset_slug,
            "files": _records_json(records),
            "pinned_union_sha256": EXPECTED_UNION_SHA256,
            "schema_version": FRESH_BUNDLE_SCHEMA,
            "specification_sha256": dict(SPECIFICATION_SHA256),
            "status": "complete",
            "work_manifest_sha256": {
                str(run): value for run, value in EXPECTED_WORK_MANIFEST_SHA256.items()
            },
            "task_manifest_sha256": dict(EXPECTED_TASK_MANIFEST_SHA256),
        }
        provenance = _canonical_json_bytes(payload)
        _write_file_fsync(staging / FRESH_PROVENANCE_FILENAME, provenance)

    _stage_directory(destination, populate)
    return validate_fresh_input_bundle(
        destination,
        expected_dataset_slug=dataset_slug,
    )


def validate_fresh_input_bundle(
    root: Path,
    *,
    expected_dataset_slug: str,
    expected_provenance_sha256: str | None = None,
) -> ValidatedInputBundle:
    """Validate an immutable fresh-input bundle without trusting its manifest.

    Returns:
        Exact validated bundle identity and file records.

    Raises:
        ValueError: If any field, file, hash, size, or canonical byte differs.

    """
    _validate_dataset_slug(expected_dataset_slug)
    expected_names = {
        *(_work_manifest_name(run) for run in range(1, 6)),
        NORMAL_CHECKPOINT_NAME,
        SO2_CHECKPOINT_NAME,
        UNION_MANIFEST_NAME,
        *(_task_manifest_name(name) for name in EXPECTED_TASK_MANIFEST_SHA256),
        FRESH_PROVENANCE_FILENAME,
        DATASET_METADATA_FILENAME,
    }
    _require_exact_files(root, expected_names)
    provenance_path = root / FRESH_PROVENANCE_FILENAME
    provenance, provenance_bytes = _canonical_json(provenance_path)
    provenance_sha256 = hashlib.sha256(provenance_bytes).hexdigest()
    if (
        expected_provenance_sha256 is not None
        and provenance_sha256 != expected_provenance_sha256
    ):
        message = "Fresh bundle provenance SHA-256 mismatch"
        raise ValueError(message)
    expected_static: JsonObject = {
        "checkpoint_sha256": {
            str(name): value for name, value in EXPECTED_CHECKPOINT_SHA256.items()
        },
        "dataset_slug": expected_dataset_slug,
        "pinned_union_sha256": EXPECTED_UNION_SHA256,
        "schema_version": FRESH_BUNDLE_SCHEMA,
        "specification_sha256": dict(SPECIFICATION_SHA256),
        "status": "complete",
        "task_manifest_sha256": dict(EXPECTED_TASK_MANIFEST_SHA256),
        "work_manifest_sha256": {
            str(run): value for run, value in EXPECTED_WORK_MANIFEST_SHA256.items()
        },
    }
    if set(provenance) != {*expected_static, "files"}:
        message = "Fresh bundle provenance fields differ from the fixed schema"
        raise ValueError(message)
    for key, expected in expected_static.items():
        if provenance.get(key) != expected:
            message = f"Fresh bundle provenance mismatch for {key}"
            raise ValueError(message)
    records = _parse_records(provenance.get("files"))
    expected_payload_names = expected_names - {FRESH_PROVENANCE_FILENAME}
    if {record.logical_name for record in records} != expected_payload_names:
        message = "Fresh bundle payload allow-list differs"
        raise ValueError(message)
    _validate_record_files(root, records)
    for run_number in range(1, 6):
        load_work_manifest(
            root / _work_manifest_name(run_number),
            run_number=run_number,
            expected_sha256=EXPECTED_WORK_MANIFEST_SHA256[run_number],
        )
    load_work_manifest(
        root / UNION_MANIFEST_NAME,
        run_number=0,
        expected_sha256=EXPECTED_UNION_SHA256,
        require_run_basename=False,
    )
    for task_split, expected_hash in EXPECTED_TASK_MANIFEST_SHA256.items():
        _require_file_hash(
            root / _task_manifest_name(task_split),
            expected_hash,
            task_split,
        )
    _require_file_hash(
        root / NORMAL_CHECKPOINT_NAME,
        EXPECTED_CHECKPOINT_SHA256["normal_vae"],
        NORMAL_CHECKPOINT_NAME,
    )
    _require_file_hash(
        root / SO2_CHECKPOINT_NAME,
        EXPECTED_CHECKPOINT_SHA256["so2_vae"],
        SO2_CHECKPOINT_NAME,
    )
    expected_metadata = _dataset_metadata(
        slug=expected_dataset_slug,
        title=FRESH_DATASET_TITLE,
        description="Private immutable Spec 0021 inference inputs.",
    )
    _require_canonical_payload(root / DATASET_METADATA_FILENAME, expected_metadata)
    return ValidatedInputBundle(
        root=root,
        provenance_sha256=provenance_sha256,
        dataset_slug=expected_dataset_slug,
        files=records,
    )


def stage_resume_bundle(  # noqa: PLR0913
    destination: Path,
    *,
    artifacts_dir: Path,
    work_manifest_path: Path,
    dataset_slug: str,
    dataset_version: int,
    run_number: int,
    input_bundle_sha256: str,
    run_config_sha256: str,
) -> ResumeBundleAuthority:
    """Stage one strict run-specific resume dataset atomically.

    Returns:
        Authority values to embed before the bundle may be attached.

    Raises:
        ValueError: If artifacts are not an accepted Spec 0020 recovery window.

    """
    _validate_dataset_identity(dataset_slug, dataset_version)
    _validate_run_number(run_number)
    _validate_sha256(input_bundle_sha256, "input bundle SHA-256")
    _validate_sha256(run_config_sha256, "run config SHA-256")
    manifest_hash = EXPECTED_WORK_MANIFEST_SHA256[run_number]
    manifest = load_work_manifest(
        work_manifest_path,
        run_number=run_number,
        expected_sha256=manifest_hash,
    )
    artifact_names = _resume_artifact_names(run_number)
    observed = _observed_file_names(artifacts_dir)
    windows = _detect_resume_windows(observed, run_number)
    required_common = {
        artifact_names["worker_resume"],
        artifact_names["incomplete"],
    }
    allowed = set(required_common)
    for model in _MODEL_NAMES:
        allowed.update(_writer_window_names(model, run_number, windows[model]))
    if observed != allowed:
        message = "Resume artifact directory contains missing or extra files"
        raise ValueError(message)
    binding = ResumeBundleAuthority(
        provenance_sha256="0" * _SHA256_LENGTH,
        dataset_slug=dataset_slug,
        dataset_version=dataset_version,
        run_number=run_number,
        input_bundle_sha256=input_bundle_sha256,
        run_config_sha256=run_config_sha256,
        work_manifest_sha256=manifest_hash,
    )
    _validate_resume_artifacts(
        artifacts_dir,
        binding=binding,
        manifest=manifest,
        windows=windows,
    )

    def populate(staging: Path) -> None:
        sources = {name: artifacts_dir / name for name in sorted(observed)}
        records = list(_copy_sources(sources, staging))
        metadata = _dataset_metadata(
            slug=dataset_slug,
            title=f"{RESUME_DATASET_TITLE_PREFIX} {run_number:02d}",
            description=f"Private Spec 0021 run {run_number:02d} resume input.",
        )
        metadata_bytes = _canonical_json_bytes(metadata)
        _write_file_fsync(staging / DATASET_METADATA_FILENAME, metadata_bytes)
        records.append(
            BundleFile(
                DATASET_METADATA_FILENAME,
                len(metadata_bytes),
                hashlib.sha256(metadata_bytes).hexdigest(),
            ),
        )
        records.sort(key=lambda record: record.logical_name)
        payload = _resume_provenance_payload(
            binding=binding,
            records=records,
            windows=windows,
        )
        encoded = _canonical_json_bytes(payload)
        _write_file_fsync(staging / RESUME_PROVENANCE_FILENAME, encoded)

    _stage_directory(destination, populate)
    provenance_sha256 = _sha256_file(destination / RESUME_PROVENANCE_FILENAME)
    authority = ResumeBundleAuthority(
        provenance_sha256=provenance_sha256,
        dataset_slug=dataset_slug,
        dataset_version=dataset_version,
        run_number=run_number,
        input_bundle_sha256=input_bundle_sha256,
        run_config_sha256=run_config_sha256,
        work_manifest_sha256=manifest_hash,
    )
    validate_resume_bundle(
        destination,
        authority=authority,
        work_manifest_path=work_manifest_path,
    )
    return authority


def validate_resume_bundle(
    root: Path,
    *,
    authority: ResumeBundleAuthority,
    work_manifest_path: Path,
) -> ValidatedResumeBundle:
    """Validate a pinned read-only resume attachment in place.

    Returns:
        Validated file allow-list and writer recovery windows.

    Raises:
        ValueError: If authority, canonical provenance, files, or states differ.

    """
    _validate_authority(authority)
    manifest = load_work_manifest(
        work_manifest_path,
        run_number=authority.run_number,
        expected_sha256=authority.work_manifest_sha256,
    )
    provenance, encoded = _canonical_json(root / RESUME_PROVENANCE_FILENAME)
    if hashlib.sha256(encoded).hexdigest() != authority.provenance_sha256:
        message = "Resume provenance SHA-256 differs from embedded authority"
        raise ValueError(message)
    records = _parse_records(provenance.get("files"))
    windows = _parse_writer_windows(provenance.get("writer_windows"))
    artifact_names = _resume_artifact_names(authority.run_number)
    expected_record_names = {
        artifact_names["worker_resume"],
        artifact_names["incomplete"],
        DATASET_METADATA_FILENAME,
    }
    for model in _MODEL_NAMES:
        expected_record_names.update(
            _writer_window_names(model, authority.run_number, windows[model]),
        )
    if {record.logical_name for record in records} != expected_record_names:
        message = "Resume bundle payload allow-list differs from its writer windows"
        raise ValueError(message)
    expected_payload = _resume_provenance_payload(
        binding=authority,
        records=records,
        windows=windows,
    )
    if provenance != expected_payload:
        message = "Resume provenance fields differ from embedded authority"
        raise ValueError(message)
    expected_names = {
        *(record.logical_name for record in records),
        RESUME_PROVENANCE_FILENAME,
        DATASET_METADATA_FILENAME,
    }
    _require_exact_files(root, expected_names)
    _validate_record_files(root, records)
    expected_metadata = _dataset_metadata(
        slug=authority.dataset_slug,
        title=f"{RESUME_DATASET_TITLE_PREFIX} {authority.run_number:02d}",
        description=f"Private Spec 0021 run {authority.run_number:02d} resume input.",
    )
    _require_canonical_payload(root / DATASET_METADATA_FILENAME, expected_metadata)
    _validate_resume_artifacts(
        root,
        binding=authority,
        manifest=manifest,
        windows=windows,
    )
    return ValidatedResumeBundle(root, authority, records, windows)


def copy_validated_resume_bundle(
    source: Path,
    destination: Path,
    *,
    authority: ResumeBundleAuthority,
    work_manifest_path: Path,
) -> Path:
    """Validate in place, then durably copy only allow-listed worker files.

    Returns:
        The same-directory working-set path ready for writer construction.

    """
    validated = validate_resume_bundle(
        source,
        authority=authority,
        work_manifest_path=work_manifest_path,
    )
    sources = {
        record.logical_name: source / record.logical_name
        for record in validated.files
        if record.logical_name != DATASET_METADATA_FILENAME
    }

    def populate(staging: Path) -> None:
        copied = _copy_sources(sources, staging)
        expected = tuple(
            record
            for record in validated.files
            if record.logical_name != DATASET_METADATA_FILENAME
        )
        if copied != expected:
            message = "Copied resume files differ from validated attachment records"
            raise ValueError(message)

    _stage_directory(destination, populate)
    return destination


def _resume_provenance_payload(
    *,
    binding: ResumeBundleAuthority,
    records: Sequence[BundleFile],
    windows: Mapping[ModelName, ResumeWindow],
) -> JsonObject:
    return {
        "dataset": {
            "slug": binding.dataset_slug,
            "version": binding.dataset_version,
        },
        "files": _records_json(records),
        "input_bundle_sha256": binding.input_bundle_sha256,
        "run_config_sha256": binding.run_config_sha256,
        "run_number": binding.run_number,
        "schema_version": RESUME_BUNDLE_SCHEMA,
        "status": "complete",
        "work_manifest_sha256": binding.work_manifest_sha256,
        "writer_windows": {str(model): window for model, window in windows.items()},
    }


def _validate_resume_artifacts(
    root: Path,
    *,
    binding: ResumeBundleAuthority,
    manifest: WorkManifest,
    windows: Mapping[ModelName, ResumeWindow],
) -> None:
    names = _resume_artifact_names(binding.run_number)
    worker_state = _read_object(root / names["worker_resume"])
    expected_worker_static = {
        "schema_version": WORKER_RESUME_SCHEMA,
        "status": "in_progress",
        "input_bundle_sha256": binding.input_bundle_sha256,
        "run_config_sha256": binding.run_config_sha256,
        "work_manifest_sha256": binding.work_manifest_sha256,
        "run_number": binding.run_number,
    }
    if any(
        worker_state.get(key) != value for key, value in expected_worker_static.items()
    ):
        message = "Worker resume state disagrees with pinned resume authority"
        raise ValueError(message)
    if set(worker_state) != {
        *expected_worker_static,
        "normal_prefix",
        "so2_prefix",
        "completed_wsi_evidence",
        "active_wsi",
    }:
        message = "Worker resume state fields differ from the fixed schema"
        raise ValueError(message)
    normal_prefix = _parse_worker_prefix(worker_state.get("normal_prefix"))
    so2_prefix = _parse_worker_prefix(worker_state.get("so2_prefix"))
    if normal_prefix != so2_prefix:
        message = "Worker journal base prefixes are not converged"
        raise ValueError(message)
    evidence_ids = _parse_completed_evidence(
        worker_state.get("completed_wsi_evidence"),
    )
    if evidence_ids != normal_prefix[1]:
        message = "Worker evidence does not match its converged WSI prefix"
        raise ValueError(message)
    active = _parse_active_wsi(worker_state.get("active_wsi"), manifest)
    observed_prefixes: dict[ModelName, tuple[int, tuple[int, ...]]] = {
        model: _validate_writer_window(
            root,
            model=model,
            run_number=binding.run_number,
            window=windows[model],
            manifest=manifest,
        )
        for model in _MODEL_NAMES
    }
    recovery_kind, catch_up_model = _validate_prefix_relationship(
        normal_prefix,
        active,
        observed_prefixes,
    )
    marker = _read_object(root / names["incomplete"])
    expected_marker = {
        "schema_version": INCOMPLETE_SCHEMA,
        "status": "incomplete",
        "worker_resume": names["worker_resume"],
        "normal_completed_wsi_ids": list(observed_prefixes["normal_vae"][1]),
        "so2_completed_wsi_ids": list(observed_prefixes["so2_vae"][1]),
        "recovery_kind": recovery_kind,
        "catch_up_model": catch_up_model,
    }
    if marker != expected_marker:
        message = "Incomplete marker disagrees with validated writer prefixes"
        raise ValueError(message)


def _validate_writer_window(  # noqa: C901, PLR0912, PLR0914, PLR0915
    root: Path,
    *,
    model: ModelName,
    run_number: int,
    window: ResumeWindow,
    manifest: WorkManifest,
) -> tuple[int, tuple[int, ...]]:
    names = _writer_names(model, run_number)
    if window == "complete":
        _validate_complete_artifact(root, names, model, manifest)
        return len(manifest.rows), tuple(item[0] for item in manifest.wsi_ranges)
    state = _read_object(root / names["state"])
    expected_state = {
        "schema_version": "spec0020.latent_resume.v1",
        "status": "in_progress",
        "model_name": model,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256[model],
        "pinned_union_sha256": EXPECTED_UNION_SHA256,
    }
    if any(state.get(key) != value for key, value in expected_state.items()):
        message = f"{model} resume state provenance mismatch"
        raise ValueError(message)
    if set(state) != {
        *expected_state,
        "source_manifest",
        "tensor",
        "committed_rows",
        "committed_bytes",
        "completed_wsi_ids",
        "prefix_crc32",
        "prefix_sha256",
    }:
        message = f"{model} resume state fields differ from the fixed schema"
        raise ValueError(message)
    expected_tensor = {
        "dtype": "float32_le",
        "shape": [16, 32, 32],
        "layout": "CHW",
        "record_bytes": LATENT_RECORD_BYTES,
    }
    if state.get("tensor") != expected_tensor:
        message = f"{model} resume tensor contract mismatch"
        raise ValueError(message)
    source = state.get("source_manifest")
    if not isinstance(source, dict):
        message = f"{model} resume source manifest must be an object"
        raise TypeError(message)
    if source != {
        "logical_basename": manifest.path.name,
        "sha256": manifest.sha256,
        "run_number": manifest.run_number,
        "row_count": len(manifest.rows),
    }:
        message = f"{model} resume source manifest mismatch"
        raise ValueError(message)
    committed_rows = _plain_int(state.get("committed_rows"), "committed_rows")
    committed_bytes = _plain_int(state.get("committed_bytes"), "committed_bytes")
    expected_boundary = LATENT_SHARD_HEADER_SIZE + committed_rows * LATENT_RECORD_BYTES
    if committed_bytes != expected_boundary:
        message = f"{model} resume committed byte boundary mismatch"
        raise ValueError(message)
    expected_ids = [
        wsi for wsi, _start, end in manifest.wsi_ranges if end <= committed_rows
    ]
    if state.get("completed_wsi_ids") != expected_ids:
        message = f"{model} resume WSI prefix mismatch"
        raise ValueError(message)
    if committed_rows not in {0, *(end for _wsi, _start, end in manifest.wsi_ranges)}:
        message = f"{model} committed rows cut through a WSI"
        raise ValueError(message)
    binary_name = names["partial"] if window == "partial_with_state" else names["final"]
    binary_path = root / binary_name
    if binary_path.stat().st_size != expected_boundary:
        message = f"{model} resume binary size differs from committed boundary"
        raise ValueError(message)
    header, crc, digest = _scan_latent_prefix(binary_path)
    prefix_crc32 = _plain_int(state.get("prefix_crc32"), "prefix_crc32")
    prefix_sha256 = state.get("prefix_sha256")
    if not isinstance(prefix_sha256, str):
        message = f"{model} resume prefix SHA-256 must be a string"
        raise TypeError(message)
    _validate_sha256(prefix_sha256, "prefix SHA-256")
    if prefix_crc32 != crc or prefix_sha256 != digest:
        message = f"{model} resume prefix checksum mismatch"
        raise ValueError(message)
    if window == "partial_with_state" and (
        header.tensor_count != 0 or header.payload_crc32 != 0
    ):
        message = f"{model} partial header must remain provisional"
        raise ValueError(message)
    if window != "partial_with_state" and (
        committed_rows != len(manifest.rows)
        or header.tensor_count != len(manifest.rows)
        or header.payload_crc32 != crc
    ):
        message = f"{model} final-with-state window is not fully committed"
        raise ValueError(message)
    if window == "complete_with_stale_state":
        _validate_complete_artifact(root, names, model, manifest)
    return committed_rows, tuple(expected_ids)


def _validate_complete_artifact(
    root: Path,
    names: Mapping[str, str],
    model: ModelName,
    manifest: WorkManifest,
) -> None:
    validate_latent_artifact(
        bin_path=root / names["final"],
        manifest_path=manifest.path,
        run_number=manifest.run_number,
        model_name=model,
        checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256[model],
        expected_checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256[model],
        expected_manifest_sha256=manifest.sha256,
        expected_union_sha256=EXPECTED_UNION_SHA256,
        validate_payload=True,
    )


def _parse_worker_prefix(value: object) -> tuple[int, tuple[int, ...]]:
    if not isinstance(value, dict):
        message = "Worker prefix fields are invalid"
        raise TypeError(message)
    mapping = cast("dict[str, object]", value)
    if set(mapping) != {
        "committed_rows",
        "completed_wsi_ids",
    }:
        message = "Worker prefix fields are invalid"
        raise ValueError(message)
    rows = _plain_int(mapping.get("committed_rows"), "committed_rows")
    raw_ids = mapping.get("completed_wsi_ids")
    if not isinstance(raw_ids, list):
        message = "completed_wsi_ids must be a list"
        raise TypeError(message)
    ids = tuple(
        _plain_int(item, "completed_wsi_id") for item in cast("list[object]", raw_ids)
    )
    return rows, ids


def _parse_completed_evidence(value: object) -> tuple[int, ...]:
    if not isinstance(value, list):
        message = "completed_wsi_evidence must be a list"
        raise TypeError(message)
    return tuple(_parse_evidence_wsi_id(item) for item in cast("list[object]", value))


def _parse_evidence_wsi_id(value: object) -> int:
    if not isinstance(value, dict):
        message = "WSI evidence fields are invalid"
        raise TypeError(message)
    mapping = cast("dict[str, object]", value)
    if set(mapping) != {
        "wsi_id",
        "png_bytes",
        "png_sha256",
        "transcript_sha256",
    }:
        message = "WSI evidence fields are invalid"
        raise ValueError(message)
    wsi_id = _plain_int(mapping.get("wsi_id"), "wsi_id")
    _plain_int(mapping.get("png_bytes"), "png_bytes")
    for key in ("png_sha256", "transcript_sha256"):
        raw = mapping.get(key)
        if not isinstance(raw, str):
            message = f"{key} must be a string"
            raise TypeError(message)
        _validate_sha256(raw, key)
    return wsi_id


def _parse_active_wsi(
    value: object,
    manifest: WorkManifest,
) -> tuple[int, int, int] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        message = "active_wsi fields are invalid"
        raise TypeError(message)
    mapping = cast("dict[str, object]", value)
    if set(mapping) != {
        "row_start",
        "row_end",
        "evidence",
    }:
        message = "active_wsi fields are invalid"
        raise ValueError(message)
    row_start = _plain_int(mapping.get("row_start"), "row_start")
    row_end = _plain_int(mapping.get("row_end"), "row_end")
    wsi_id = _parse_evidence_wsi_id(mapping.get("evidence"))
    if (wsi_id, row_start, row_end) not in manifest.wsi_ranges:
        message = "Active WSI journal disagrees with manifest boundaries"
        raise ValueError(message)
    return row_start, row_end, wsi_id


def _validate_prefix_relationship(
    base: tuple[int, tuple[int, ...]],
    active: tuple[int, int, int] | None,
    observed: Mapping[ModelName, tuple[int, tuple[int, ...]]],
) -> tuple[str | None, ModelName | None]:
    if active is None:
        if any(prefix != base for prefix in observed.values()):
            message = "Writer prefixes disagree with inactive worker journal"
            raise ValueError(message)
        return None, None
    row_start, row_end, wsi_id = active
    if base[0] != row_start:
        message = "Active WSI does not begin at the converged worker prefix"
        raise ValueError(message)
    after = (row_end, (*base[1], wsi_id))
    normal = observed["normal_vae"]
    so2 = observed["so2_vae"]
    if normal == base and so2 == base:
        return None, None
    if normal == after and so2 == after:
        return "verify_converged", None
    if normal == after and so2 == base:
        return "catch_up", "so2_vae"
    if so2 == after and normal == base:
        return "catch_up", "normal_vae"
    message = "Writer prefixes are outside accepted dual-worker recovery windows"
    raise ValueError(message)


def _scan_latent_prefix(path: Path) -> tuple[LatentShardHeader, int, str]:
    digest = hashlib.sha256()
    crc = 0
    with path.open("rb") as handle:
        header = parse_latent_shard_header(handle.read(LATENT_SHARD_HEADER_SIZE))
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            crc = zlib.crc32(chunk, crc) & 0xFFFFFFFF
            digest.update(chunk)
    return header, crc, digest.hexdigest()


def _detect_resume_windows(
    observed: set[str],
    run_number: int,
) -> dict[ModelName, ResumeWindow]:
    return {
        model: _detect_writer_window(observed, model, run_number)
        for model in _MODEL_NAMES
    }


def _detect_writer_window(
    observed: set[str],
    model: ModelName,
    run_number: int,
) -> ResumeWindow:
    names = _writer_names(model, run_number)
    present = {key for key, name in names.items() if name in observed}
    windows: dict[frozenset[str], ResumeWindow] = {
        frozenset({"partial", "state"}): "partial_with_state",
        frozenset({"final", "state"}): "final_with_state",
        frozenset({"final", "sidecar", "state"}): "complete_with_stale_state",
        frozenset({"final", "sidecar"}): "complete",
    }
    try:
        return windows[frozenset(present)]
    except KeyError:
        message = f"{model} files do not form an accepted Spec 0020 recovery window"
        raise ValueError(message) from None


def _parse_writer_windows(value: object) -> dict[ModelName, ResumeWindow]:
    if not isinstance(value, dict):
        message = "Resume writer_windows fields differ"
        raise TypeError(message)
    mapping = cast("dict[str, object]", value)
    if set(mapping) != set(_MODEL_NAMES):
        message = "Resume writer_windows fields differ"
        raise ValueError(message)
    allowed = {
        "partial_with_state",
        "final_with_state",
        "complete_with_stale_state",
        "complete",
    }
    result: dict[ModelName, ResumeWindow] = {}
    for model in _MODEL_NAMES:
        raw = mapping[model]
        if not isinstance(raw, str) or raw not in allowed:
            message = f"Invalid resume writer window for {model}"
            raise ValueError(message)
        result[model] = cast("ResumeWindow", raw)
    return result


def _writer_window_names(
    model: ModelName,
    run_number: int,
    window: ResumeWindow,
) -> set[str]:
    names = _writer_names(model, run_number)
    keys = {
        "partial_with_state": ("partial", "state"),
        "final_with_state": ("final", "state"),
        "complete_with_stale_state": ("final", "sidecar", "state"),
        "complete": ("final", "sidecar"),
    }[window]
    return {names[key] for key in keys}


def _writer_names(model: ModelName, run_number: int) -> dict[str, str]:
    stem = f"{model}_mu_run_{run_number:02d}_of_05"
    return {
        "partial": f"{stem}.bin.partial",
        "state": f"{stem}.resume.json",
        "final": f"{stem}.bin",
        "sidecar": f"{stem}.json",
    }


def _resume_artifact_names(run_number: int) -> dict[str, str]:
    return {
        "worker_resume": f"spec0021_worker_run_{run_number:02d}.resume.json",
        "incomplete": f"spec0021_incomplete_run_{run_number:02d}.json",
    }


def _copy_sources(
    sources: Mapping[str, Path],
    destination: Path,
) -> tuple[BundleFile, ...]:
    records: list[BundleFile] = []
    for logical_name in sorted(sources):
        _validate_logical_name(logical_name)
        source = sources[logical_name]
        target = destination / logical_name
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as source_handle, target.open("xb") as target_handle:
            digest = hashlib.sha256()
            size = 0
            while chunk := source_handle.read(_HASH_CHUNK_BYTES):
                target_handle.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            target_handle.flush()
            os.fsync(target_handle.fileno())
        records.append(BundleFile(logical_name, size, digest.hexdigest()))
        _fsync_directory(target.parent)
    _fsync_directory(destination)
    return tuple(records)


def _stage_directory(destination: Path, populate: Callable[[Path], None]) -> None:
    if destination.exists():
        message = f"Bundle destination already exists: {destination}"
        raise FileExistsError(message)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent),
    )
    try:
        populate(staging)
        _fsync_directory(staging)
        staging.replace(destination)
        _fsync_directory(destination.parent)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _records_json(records: Sequence[BundleFile]) -> JsonObject:
    return {
        record.logical_name: {"bytes": record.size, "sha256": record.sha256}
        for record in records
    }


def _parse_records(value: object) -> tuple[BundleFile, ...]:
    if not isinstance(value, dict) or not value:
        message = "Bundle files must be a nonempty object"
        raise TypeError(message)
    mapping = cast("dict[str, object]", value)
    records: list[BundleFile] = []
    for logical_name in sorted(mapping):
        _validate_logical_name(logical_name)
        raw = mapping[logical_name]
        if not isinstance(raw, dict):
            message = f"Invalid bundle file record for {logical_name}"
            raise TypeError(message)
        record = cast("dict[str, object]", raw)
        if set(record) != {"bytes", "sha256"}:
            message = f"Invalid bundle file record for {logical_name}"
            raise ValueError(message)
        size = _plain_int(record.get("bytes"), "bytes")
        sha256 = record.get("sha256")
        if not isinstance(sha256, str):
            message = f"Bundle file SHA-256 must be a string: {logical_name}"
            raise TypeError(message)
        _validate_sha256(sha256, f"{logical_name} SHA-256")
        records.append(BundleFile(logical_name, size, sha256))
    return tuple(records)


def _validate_record_files(root: Path, records: Sequence[BundleFile]) -> None:
    for record in records:
        path = root / record.logical_name
        if path.stat().st_size != record.size or _sha256_file(path) != record.sha256:
            message = f"Bundle file bytes differ from allow-list: {record.logical_name}"
            raise ValueError(message)


def _dataset_metadata(*, slug: str, title: str, description: str) -> JsonObject:
    return {
        "description": description,
        "id": slug,
        "licenses": [{"name": "unknown"}],
        "title": title,
    }


def _canonical_json(path: Path) -> tuple[JsonObject, bytes]:
    encoded = path.read_bytes()
    raw = cast("object", json.loads(encoded.decode("utf-8")))
    if not isinstance(raw, dict):
        message = f"Expected JSON object: {path.name}"
        raise TypeError(message)
    payload = cast("JsonObject", raw)
    if encoded != _canonical_json_bytes(payload):
        message = f"JSON is not canonical: {path.name}"
        raise ValueError(message)
    return payload, encoded


def _canonical_json_bytes(payload: JsonObject) -> bytes:
    return f"{json.dumps(payload, indent=2, sort_keys=True)}\n".encode()


def _require_canonical_payload(path: Path, expected: JsonObject) -> None:
    payload, _encoded = _canonical_json(path)
    if payload != expected:
        message = f"Canonical JSON fields differ: {path.name}"
        raise ValueError(message)


def _read_object(path: Path) -> dict[str, object]:
    raw = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(raw, dict):
        message = f"Expected JSON object: {path.name}"
        raise TypeError(message)
    return cast("dict[str, object]", raw)


def _write_file_fsync(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _require_exact_files(root: Path, expected: set[str]) -> None:
    if not root.is_dir():
        message = f"Bundle directory is missing: {root}"
        raise ValueError(message)
    entries = tuple(root.rglob("*"))
    if any(
        path.is_symlink() or (not path.is_file() and not path.is_dir())
        for path in entries
    ):
        message = "Bundle entries must be regular files or directories"
        raise ValueError(message)
    observed = {path.relative_to(root).as_posix() for path in entries if path.is_file()}
    expected_directories = {
        parent.as_posix()
        for name in expected
        for parent in Path(name).parents
        if parent != Path()
    }
    observed_directories = {
        path.relative_to(root).as_posix() for path in entries if path.is_dir()
    }
    if observed != expected or observed_directories != expected_directories:
        message = "Bundle contains missing, extra, nested, or non-file entries"
        raise ValueError(message)


def _observed_file_names(root: Path) -> set[str]:
    if not root.is_dir():
        return set()
    entries = tuple(root.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in entries):
        message = "Bundle entries must all be top-level regular files"
        raise ValueError(message)
    return {path.name for path in entries}


def _require_file_hash(path: Path, expected: str, name: str) -> None:
    if not path.is_file() or _sha256_file(path) != expected:
        message = f"Pinned file SHA-256 mismatch: {name}"
        raise ValueError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _work_manifest_name(run_number: int) -> str:
    return f"manifests/work_shards/run_{run_number:02d}_of_05.csv"


def _task_manifest_name(task_split: str) -> str:
    return f"manifests/task_views/{task_split}.csv"


def _validate_dataset_identity(slug: str, version: int) -> None:
    _validate_dataset_slug(slug)
    if isinstance(version, bool) or version < 1:
        message = "Dataset version must be a positive integer"
        raise ValueError(message)


def _validate_dataset_slug(slug: str) -> None:
    if _DATASET_SLUG.fullmatch(slug) is None:
        message = f"Invalid private Kaggle dataset slug: {slug!r}"
        raise ValueError(message)


def _validate_run_number(run_number: int) -> None:
    if isinstance(run_number, bool) or run_number not in range(1, 6):
        message = "Resume run number must be in 1..5"
        raise ValueError(message)


def _validate_authority(authority: ResumeBundleAuthority) -> None:
    _validate_dataset_identity(authority.dataset_slug, authority.dataset_version)
    _validate_run_number(authority.run_number)
    for name, value in (
        ("resume provenance SHA-256", authority.provenance_sha256),
        ("input bundle SHA-256", authority.input_bundle_sha256),
        ("run config SHA-256", authority.run_config_sha256),
        ("work manifest SHA-256", authority.work_manifest_sha256),
    ):
        _validate_sha256(value, name)
    if (
        authority.work_manifest_sha256
        != EXPECTED_WORK_MANIFEST_SHA256[authority.run_number]
    ):
        message = "Resume authority work manifest is not canonical"
        raise ValueError(message)


def _validate_sha256(value: str, name: str) -> None:
    if len(value) != _SHA256_LENGTH or any(
        char not in "0123456789abcdef" for char in value
    ):
        message = f"Invalid {name}"
        raise ValueError(message)


def _validate_logical_name(value: str) -> None:
    path = Path(value)
    if (
        path.is_absolute()
        or value in {"", ".", ".."}
        or ".." in path.parts
        or path.as_posix() != value
    ):
        message = f"Bundle logical name must be one safe relative path: {value!r}"
        raise ValueError(message)


def _plain_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        message = f"{name} must be a nonnegative integer"
        raise TypeError(message)
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


__all__ = [
    "DATASET_METADATA_FILENAME",
    "FRESH_BUNDLE_SCHEMA",
    "FRESH_PROVENANCE_FILENAME",
    "FRESH_UPLOAD_ARCHIVE_FILENAME",
    "NORMAL_CHECKPOINT_NAME",
    "RESUME_BUNDLE_SCHEMA",
    "RESUME_PROVENANCE_FILENAME",
    "SO2_CHECKPOINT_NAME",
    "SPECIFICATION_SHA256",
    "BundleFile",
    "ResumeBundleAuthority",
    "ValidatedInputBundle",
    "ValidatedResumeBundle",
    "copy_validated_resume_bundle",
    "extract_fresh_upload_archive",
    "stage_fresh_input_bundle",
    "stage_fresh_upload_archive",
    "stage_resume_bundle",
    "validate_fresh_input_bundle",
    "validate_resume_bundle",
]
