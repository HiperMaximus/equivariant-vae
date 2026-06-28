# Copyright 2026 HiperMaximus
"""Spec 0008 fixed-32 selector readiness helpers."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from eqvae.benchmarking.io import JsonObject, write_json
from eqvae.data.fixed_selectors import (
    FIXED_32_TRAIN_OVERFIT_COUNT,
    FIXED_32_TRAIN_OVERFIT_KIND,
    FixedSelectorDocument,
    FixedSelectorGenerationContext,
    generate_fixed_selector_document,
    load_fixed_selector_document,
    validate_fixed_selector_document,
    write_fixed_selector_document,
)
from eqvae.data.patch_shards import PatchShardSpec
from eqvae.data.roots import TRAIN_BIN_NAME, TRAIN_CSV_NAME, resolve_patch_data_paths
from eqvae.data.splits import load_masked_holdout_wsi_ids
from eqvae.data.synthetic import SyntheticPatchSpec, write_synthetic_patch_shard
from eqvae.training.selected_runtime import EXPECTED_DATASET_SLUG

if TYPE_CHECKING:
    from collections.abc import Mapping

FIXED32_READINESS_SCHEMA_VERSION = "spec0008.fixed32_selector_readiness.v1"
FIXED32_SELECTOR_STATUS_SCHEMA_VERSION = "spec0008.fixed32_selector_status.v1"
REMOTE_GENERATE_MODE = "remote_generate"
LOCAL_SELECTOR_MODE = "local_selector"
EXPECTED_REAL_TRAIN_PATCH_COUNT = 300_000
EXPECTED_REAL_TRAIN_CSV_SHA256 = (
    "8fc4959f7de006eed259f818ef2cc4ea03d1f3ec6ba483bf7229c04562f22a52"
)
EXPECTED_REAL_TRAIN_BIN_FILE_SIZE = 58_982_400_064
EXPECTED_REAL_TRAIN_HEADER_CRC32 = 1_289_496_176
EXPECTED_TINY_SELECTOR_COUNT = FIXED_32_TRAIN_OVERFIT_COUNT
OK_STATUS = "pass"
FAIL_STATUS = "fail"
DEFAULT_SYNTHETIC_TRAIN_COUNT = 40
DEFAULT_SYNTHETIC_VALIDATION_COUNT = 25


@dataclass(frozen=True)
class Fixed32RemoteGenerateReadinessRequest:
    """Inputs for the local Spec 0008 remote-generate readiness proof."""

    output_dir: Path
    synthetic_root: Path
    config_path: Path
    masked_holdout_csv: Path
    image_size: int = 256
    channels: int = 3


@dataclass(frozen=True)
class Fixed32RemoteGenerateReadinessResult:
    """Paths written by the fixed-32 readiness preflight."""

    output_dir: Path
    selector_path: Path
    readiness_path: Path


def write_fixed32_remote_generate_readiness(
    request: Fixed32RemoteGenerateReadinessRequest,
) -> Fixed32RemoteGenerateReadinessResult:
    """Generate a synthetic selector and prove only remote generation may pass.

    Returns:
        Paths for the synthetic selector and readiness artifact.

    """
    _prepare_synthetic_root(
        root=request.synthetic_root,
        image_size=request.image_size,
        channels=request.channels,
    )
    output_dir = request.output_dir
    benchmark_dir = output_dir / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    selector_a = request.synthetic_root / "fixed_32_train_overfit_patches.json"
    selector_b = benchmark_dir / "fixed_32_train_overfit_patches_second.json"
    holdout_ids = load_masked_holdout_wsi_ids(request.masked_holdout_csv)
    document_a = _generate_synthetic_fixed32_document(
        synthetic_root=request.synthetic_root,
        masked_holdout_csv=request.masked_holdout_csv,
        masked_holdout_wsi_ids=holdout_ids,
        image_size=request.image_size,
        channels=request.channels,
    )
    document_b = _generate_synthetic_fixed32_document(
        synthetic_root=request.synthetic_root,
        masked_holdout_csv=request.masked_holdout_csv,
        masked_holdout_wsi_ids=holdout_ids,
        image_size=request.image_size,
        channels=request.channels,
    )
    write_fixed_selector_document(path=selector_a, document=document_a)
    write_fixed_selector_document(path=selector_b, document=document_b)
    synthetic_status = fixed32_selector_status(
        selector_a,
        data_root=str(request.synthetic_root),
    )
    deterministic = _sha256_file(selector_a) == _sha256_file(selector_b)
    readiness_path = benchmark_dir / "fixed32_selector_readiness.json"
    readiness = _remote_generate_readiness_payload(
        request=request,
        selector_path=selector_a,
        synthetic_status=synthetic_status,
        deterministic=deterministic,
    )
    write_json(readiness_path, readiness)
    return Fixed32RemoteGenerateReadinessResult(
        output_dir=output_dir,
        selector_path=selector_a,
        readiness_path=readiness_path,
    )


def fixed32_selector_status(path: Path, *, data_root: str | None) -> JsonObject:
    """Return canonical-real readiness for one fixed-32 selector.

    Returns:
        JSON-safe readiness status. Synthetic selectors can be schema-valid and
        still fail this canonical-real check.

    """
    if not path.exists():
        return {
            "schema_version": FIXED32_SELECTOR_STATUS_SCHEMA_VERSION,
            "path": str(path),
            "sha256": "",
            "status": FAIL_STATUS,
            "selector_count": 0,
            "expected_count": EXPECTED_TINY_SELECTOR_COUNT,
            "failure_kind": "fixed_32_selector_missing",
            "validation_errors": ["fixed_32_selector_missing"],
            "canonical_real_ubc": False,
        }
    payload = _load_json(path)
    selectors = payload.get("selectors")
    selector_count = len(selectors) if isinstance(selectors, list) else 0
    errors = list(_raw_selector_errors(payload, selector_count=selector_count))
    validation_detail = ""
    if not errors:
        try:
            document = load_fixed_selector_document(path)
        except (KeyError, TypeError, ValueError) as error:
            errors.append("fixed_32_selector_schema_invalid")
            validation_detail = str(error)
        else:
            document_errors, validation_detail = _selector_document_errors(
                path=path,
                data_root=data_root,
                document=document,
            )
            errors.extend(document_errors)
    return cast(
        "JsonObject",
        {
            "schema_version": FIXED32_SELECTOR_STATUS_SCHEMA_VERSION,
            "path": str(path),
            "sha256": _sha256_file(path),
            "status": OK_STATUS if not errors else FAIL_STATUS,
            "selector_count": selector_count,
            "expected_count": EXPECTED_TINY_SELECTOR_COUNT,
            "failure_kind": "" if not errors else errors[0],
            "validation_errors": errors,
            "validation_detail": validation_detail,
            "canonical_real_ubc": not errors,
            "canonical_requirements": canonical_real_ubc_requirements(),
        },
    )


def canonical_real_ubc_requirements() -> JsonObject:
    """Return the locked canonical train-shard requirements.

    Returns:
        JSON-safe expected fingerprint payload.

    """
    return {
        "dataset_slug": EXPECTED_DATASET_SLUG,
        "source_split": "train",
        "train_csv_filename": TRAIN_CSV_NAME,
        "train_bin_filename": TRAIN_BIN_NAME,
        "train_csv_sha256": EXPECTED_REAL_TRAIN_CSV_SHA256,
        "train_bin_file_size": EXPECTED_REAL_TRAIN_BIN_FILE_SIZE,
        "train_header_crc32": EXPECTED_REAL_TRAIN_HEADER_CRC32,
        "row_count": EXPECTED_REAL_TRAIN_PATCH_COUNT,
        "patch_count": EXPECTED_REAL_TRAIN_PATCH_COUNT,
        "channels": 3,
        "height": 256,
        "width": 256,
        "layout": "CHW",
        "idx_policy": "row_order",
        "crc_checked": True,
    }


def _remote_generate_readiness_payload(
    *,
    request: Fixed32RemoteGenerateReadinessRequest,
    selector_path: Path,
    synthetic_status: JsonObject,
    deterministic: bool,
) -> JsonObject:
    synthetic_rejected = (
        synthetic_status.get("status") == FAIL_STATUS
        and synthetic_status.get("failure_kind")
        == "fixed_32_selector_not_canonical_real_ubc"
    )
    selector_count = synthetic_status.get("selector_count")
    status = (
        OK_STATUS
        if deterministic
        and synthetic_rejected
        and selector_count == EXPECTED_TINY_SELECTOR_COUNT
        else FAIL_STATUS
    )
    return {
        "schema_version": FIXED32_READINESS_SCHEMA_VERSION,
        "status": status,
        "selector_generation_mode": REMOTE_GENERATE_MODE,
        "remote_selector_generation_ready": status == OK_STATUS,
        "fixed_32_selector_real": False,
        "config_path": str(request.config_path),
        "synthetic_root": str(request.synthetic_root),
        "synthetic_selector_path": str(selector_path),
        "synthetic_selector_sha256": _sha256_file(selector_path),
        "synthetic_selector_deterministic": deterministic,
        "synthetic_selector_status": synthetic_status,
        "synthetic_selector_canonical_real_rejected": synthetic_rejected,
        "canonical_requirements": canonical_real_ubc_requirements(),
        "failure_kind": "" if status == OK_STATUS else "fixed32_readiness_failed",
    }


def _prepare_synthetic_root(*, root: Path, image_size: int, channels: int) -> None:
    if root.exists():
        shutil.rmtree(root)
    dataset = root / "dataset"
    write_synthetic_patch_shard(
        bin_path=dataset / TRAIN_BIN_NAME,
        csv_path=dataset / TRAIN_CSV_NAME,
        spec=SyntheticPatchSpec(
            count=DEFAULT_SYNTHETIC_TRAIN_COUNT,
            image_size=image_size,
            channels=channels,
        ),
        include_idx=False,
    )
    write_synthetic_patch_shard(
        bin_path=dataset / "ubc_ocean_valid.bin",
        csv_path=dataset / "ubc_ocean_valid.csv",
        spec=SyntheticPatchSpec(
            count=DEFAULT_SYNTHETIC_VALIDATION_COUNT,
            image_size=image_size,
            channels=channels,
            seed=20260613,
        ),
        include_idx=True,
    )


def _generate_synthetic_fixed32_document(
    *,
    synthetic_root: Path,
    masked_holdout_csv: Path,
    masked_holdout_wsi_ids: frozenset[str],
    image_size: int,
    channels: int,
) -> FixedSelectorDocument:
    paths = resolve_patch_data_paths(synthetic_root)
    train = paths.for_split("train")
    shard_spec = PatchShardSpec(
        bin_path=train.bin_path,
        csv_path=train.csv_path,
        image_size=image_size,
        channels=channels,
        validate_crc=True,
    )
    document = generate_fixed_selector_document(
        selector_kind=FIXED_32_TRAIN_OVERFIT_KIND,
        shard_spec=shard_spec,
        source_split="train",
        context=FixedSelectorGenerationContext(
            dataset_slug=EXPECTED_DATASET_SLUG,
            data_root=paths.root,
            masked_holdout_wsi_ids=masked_holdout_wsi_ids,
        ),
    )
    validate_fixed_selector_document(
        document=document,
        shard_spec=shard_spec,
        expected_kind=FIXED_32_TRAIN_OVERFIT_KIND,
        masked_holdout_wsi_ids=masked_holdout_wsi_ids,
    )
    return FixedSelectorDocument(
        selector_kind=document.selector_kind,
        status=document.status,
        source_split=document.source_split,
        expected_count=document.expected_count,
        selector_seed=document.selector_seed,
        source=document.source,
        selectors=document.selectors,
        expected_per_label=document.expected_per_label,
        masked_holdout_exclusion=str(masked_holdout_csv),
    )


def _raw_selector_errors(
    payload: JsonObject,
    *,
    selector_count: int,
) -> tuple[str, ...]:
    errors: list[str] = []
    if payload.get("status") == "requires_real_data_generation":
        errors.append("fixed_32_selector_placeholder")
    if payload.get("selector_kind") != FIXED_32_TRAIN_OVERFIT_KIND:
        errors.append("fixed_32_selector_wrong_kind")
    if payload.get("source_split") != "train":
        errors.append("fixed_32_selector_not_train_split")
    if _selector_dataset_slug(payload) != EXPECTED_DATASET_SLUG:
        errors.append("fixed_32_selector_wrong_dataset")
    if selector_count != EXPECTED_TINY_SELECTOR_COUNT:
        errors.append("fixed_32_selector_count_not_32")
    return tuple(errors)


def _selector_dataset_slug(payload: JsonObject) -> str:
    dataset_slug = payload.get("dataset_slug")
    if isinstance(dataset_slug, str):
        return dataset_slug
    source = payload.get("source")
    if isinstance(source, dict):
        source_slug = source.get("dataset_slug")
        if isinstance(source_slug, str):
            return source_slug
    return ""


def _selector_document_errors(
    *,
    path: Path,
    data_root: str | None,
    document: FixedSelectorDocument,
) -> tuple[tuple[str, ...], str]:
    detail = ""
    errors = list(_selector_document_basic_errors(document))
    if errors:
        return tuple(errors), detail

    document_data_root = (
        None if document.source.data_root is None else str(document.source.data_root)
    )
    resolved_data_root = data_root or document_data_root or "auto"
    try:
        paths = resolve_patch_data_paths(resolved_data_root)
    except FileNotFoundError as error:
        return ("fixed_32_selector_data_unavailable",), str(error)

    holdout_path = _masked_holdout_path(
        selector_path=path,
        selector_value=document.masked_holdout_exclusion,
    )
    try:
        masked_holdout_wsi_ids = load_masked_holdout_wsi_ids(holdout_path)
    except (OSError, ValueError) as error:
        return ("fixed_32_selector_masked_holdout_unavailable",), str(error)

    train_paths = paths.for_split("train")
    shard_spec = PatchShardSpec(
        bin_path=train_paths.bin_path,
        csv_path=train_paths.csv_path,
        image_size=document.source.header.height,
        channels=document.source.header.channels,
        validate_crc=True,
    )
    try:
        validate_fixed_selector_document(
            document=document,
            shard_spec=shard_spec,
            expected_kind=FIXED_32_TRAIN_OVERFIT_KIND,
            masked_holdout_wsi_ids=masked_holdout_wsi_ids,
        )
    except (EOFError, OSError, TypeError, ValueError) as error:
        return ("fixed_32_selector_validation_failed",), str(error)
    canonical_errors = _canonical_real_ubc_selector_errors(document)
    if canonical_errors:
        return ("fixed_32_selector_not_canonical_real_ubc",), "; ".join(
            canonical_errors,
        )
    return (), detail


def _selector_document_basic_errors(
    document: FixedSelectorDocument,
) -> tuple[str, ...]:
    errors: list[str] = []
    if document.selector_kind != FIXED_32_TRAIN_OVERFIT_KIND:
        errors.append("fixed_32_selector_wrong_kind")
    if document.source_split != "train":
        errors.append("fixed_32_selector_not_train_split")
    if document.source.dataset_slug != EXPECTED_DATASET_SLUG:
        errors.append("fixed_32_selector_wrong_dataset")
    if len(document.selectors) != EXPECTED_TINY_SELECTOR_COUNT:
        errors.append("fixed_32_selector_count_not_32")
    if not document.source.crc_checked:
        errors.append("fixed_32_selector_crc_not_checked")
    return tuple(errors)


def _canonical_real_ubc_selector_errors(
    document: FixedSelectorDocument,
) -> tuple[str, ...]:
    source = document.source
    header = source.header
    checks: tuple[tuple[str, object, object], ...] = (
        ("source.dataset_slug", source.dataset_slug, EXPECTED_DATASET_SLUG),
        ("source.source_split", source.source_split, "train"),
        ("source.csv_path.name", source.csv_path.name, TRAIN_CSV_NAME),
        ("source.csv_sha256", source.csv_sha256, EXPECTED_REAL_TRAIN_CSV_SHA256),
        ("source.bin_path.name", source.bin_path.name, TRAIN_BIN_NAME),
        (
            "source.bin_file_size",
            source.bin_file_size,
            EXPECTED_REAL_TRAIN_BIN_FILE_SIZE,
        ),
        ("source.row_count", source.row_count, EXPECTED_REAL_TRAIN_PATCH_COUNT),
        ("source.patch_count", source.patch_count, EXPECTED_REAL_TRAIN_PATCH_COUNT),
        ("source.idx_policy", source.idx_policy, "row_order"),
        ("source.crc_checked", source.crc_checked, True),
        ("header.crc32", header.crc32, EXPECTED_REAL_TRAIN_HEADER_CRC32),
        ("header.patch_count", header.patch_count, EXPECTED_REAL_TRAIN_PATCH_COUNT),
        ("header.channels", header.channels, 3),
        ("header.height", header.height, 256),
        ("header.width", header.width, 256),
        ("header.version", header.version, 1),
        ("header.layout", header.layout, b"CHW"),
    )
    return tuple(
        f"{name}: expected {expected!r}, got {actual!r}"
        for name, actual, expected in checks
        if actual != expected
    )


def _masked_holdout_path(
    *,
    selector_path: Path,
    selector_value: str | None,
) -> Path:
    if selector_value is None or not selector_value:
        return _resolve_relative_to_ancestors(
            base_path=selector_path,
            relative_path=Path("docs/data/ubc_ocean_masked_holdout_ids.csv"),
        )
    configured = Path(selector_value)
    if configured.is_absolute():
        return configured
    return _resolve_relative_to_ancestors(
        base_path=selector_path,
        relative_path=configured,
    )


def _resolve_relative_to_ancestors(*, base_path: Path, relative_path: Path) -> Path:
    for parent in base_path.resolve().parents:
        candidate = parent / relative_path
        if candidate.exists():
            return candidate
    return Path.cwd() / relative_path


def _load_json(path: Path) -> JsonObject:
    payload = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(payload, dict):
        message = f"Expected JSON object in {path}"
        raise TypeError(message)
    return cast("JsonObject", payload)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def readiness_blockers(readiness: Mapping[str, object]) -> tuple[str, ...]:
    """Return stable blocker names for a fixed-32 remote-generate artifact.

    Returns:
        Stable blocker identifiers.

    """
    blockers: list[str] = []
    if readiness.get("status") != OK_STATUS:
        blockers.append("fixed32_remote_generate_readiness_status_not_pass")
    if readiness.get("selector_generation_mode") != REMOTE_GENERATE_MODE:
        blockers.append("fixed32_selector_generation_mode_not_remote_generate")
    if readiness.get("remote_selector_generation_ready") is not True:
        blockers.append("fixed32_remote_selector_generation_ready_not_true")
    if readiness.get("fixed_32_selector_real") is not False:
        blockers.append("fixed32_remote_generate_must_not_claim_real_selector")
    if readiness.get("synthetic_selector_deterministic") is not True:
        blockers.append("fixed32_synthetic_selector_not_deterministic")
    if readiness.get("synthetic_selector_canonical_real_rejected") is not True:
        blockers.append("fixed32_synthetic_selector_not_rejected")
    return tuple(blockers)


__all__ = [
    "EXPECTED_REAL_TRAIN_BIN_FILE_SIZE",
    "EXPECTED_REAL_TRAIN_CSV_SHA256",
    "EXPECTED_REAL_TRAIN_HEADER_CRC32",
    "EXPECTED_REAL_TRAIN_PATCH_COUNT",
    "EXPECTED_TINY_SELECTOR_COUNT",
    "FAIL_STATUS",
    "FIXED32_READINESS_SCHEMA_VERSION",
    "FIXED32_SELECTOR_STATUS_SCHEMA_VERSION",
    "LOCAL_SELECTOR_MODE",
    "OK_STATUS",
    "REMOTE_GENERATE_MODE",
    "Fixed32RemoteGenerateReadinessRequest",
    "Fixed32RemoteGenerateReadinessResult",
    "canonical_real_ubc_requirements",
    "fixed32_selector_status",
    "readiness_blockers",
    "write_fixed32_remote_generate_readiness",
]
