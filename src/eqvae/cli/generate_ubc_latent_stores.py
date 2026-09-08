# Copyright 2026 HiperMaximus
# ruff: noqa: C901, DOC201, PLR0912, PLR0913, PLR0914, PLR0915, PLR0916, PLR2004, PLW0717, TRY300
"""Run or validate the guarded Spec 0021 embedding-generation workflow."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import statistics
import sys
import time
import zlib
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import torch

from eqvae.data.latent_shards import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_UNION_SHA256,
    EXPECTED_WORK_MANIFEST_SHA256,
    LATENT_RECORD_BYTES,
    LATENT_SHARD_HEADER_SIZE,
    LatentShardWriter,
    WorkManifest,
    load_work_manifest,
)
from eqvae.data.wsi_batches import iter_wsi_patch_interval, read_wsi_patch
from eqvae.inference.checkpoints import load_frozen_checkpoint
from eqvae.inference.dual_writer import DualLatentWriter, WorkerResumeBinding
from eqvae.inference.input_bundle import (
    FRESH_PROVENANCE_FILENAME,
    FRESH_UPLOAD_ARCHIVE_FILENAME,
    NORMAL_CHECKPOINT_NAME,
    RESUME_PROVENANCE_FILENAME,
    SO2_CHECKPOINT_NAME,
    ResumeBundleAuthority,
    copy_validated_resume_bundle,
    extract_fresh_upload_archive,
    stage_resume_bundle,
    validate_fresh_input_bundle,
)
from eqvae.inference.pilot import (
    PILOT_SUMMARY_SCHEMA,
    PILOT_WSI_ID,
    PilotCandidateResult,
    pilot_recipes,
    pilot_sentinel_indices,
    pilot_wsi_range,
    remove_pilot_payload,
    require_exact_dual_t4,
    select_pilot_candidate,
)
from eqvae.inference.worker import (
    DualDeviceExecutor,
    DualEncoderTiming,
    DualEncoderWorker,
    EncoderRecipe,
    FrozenEncoderRunner,
    finalize_pair,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import BinaryIO

    from eqvae.data.wsi_batches import WSIReader

INPUT_DATASET_SLUG: Final = "maximusshtefan/eqvae-ubc-ocean-latent-inputs"
SPEC_PATH: Final = Path("docs/specs/0021-dual-model-wsi-latent-inference.md")
CONFIG_SCHEMA: Final = "spec0021.inference_config.v2"
MATRIX_HEADER: Final = (
    "candidate_index",
    "batch_size",
    "d2h",
    "numeric",
    "execution",
    "status",
    "failure_kind",
    "warmup_passes",
    "measured_passes",
    "patch_count",
    "decode_seconds",
    "h2d_seconds",
    "normalization_seconds",
    "normal_encoder_seconds",
    "so2_encoder_seconds",
    "d2h_seconds",
    "serialization_seconds",
    "fsync_seconds",
    "end_to_end_seconds_p50",
    "patches_per_second_p50",
    "peak_rss_bytes",
    "pinned_bytes",
    "normal_peak_reserved_bytes",
    "so2_peak_reserved_bytes",
    "peak_total_gpu_bytes",
    "output_bytes",
    "finite",
    "identity_match",
    "eligible",
)
PILOT_CANDIDATE_COUNT: Final = 1
MEASURED_PASSES: Final = 1
PILOT_SMOKE_PATCHES: Final = 16
PROC_STATUS_FIELDS: Final = 3
SHA256_HEX_LENGTH: Final = 64
CANONICAL_SPLIT_SHA256: Final = (
    "216f69f64ed7a3e5636173d6cfe83297632113bc68310e4f6285d7ffc22cd43c"
)
KAGGLE_SAVED_OUTPUT_LIMIT_BYTES: Final = 20_000_000_000
PUBLISHED_METADATA_RESERVE_BYTES: Final = 10_000_000
WORK_ROW_COUNTS: Final = {
    1: 121_199,
    2: 119_898,
    3: 118_901,
    4: 118_513,
    5: 120_887,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch pilot, production, or downloaded-pilot validation."""
    arguments = _parser().parse_args(argv)
    command = cast("str", arguments.command)
    if command == "validate-pilot":
        _validate_pilot_download(
            Path(cast("str", arguments.input)),
            Path(cast("str", arguments.output)),
        )
        return 0
    config_path = Path(cast("str", arguments.config)).resolve()
    payload_root = Path(cast("str", arguments.payload_root)).resolve()
    output_root = Path(cast("str", arguments.output_root)).resolve()
    scratch_root = Path(cast("str", arguments.scratch_root)).resolve()
    config = _validate_config(config_path, expected_mode=command)
    _require_hash(payload_root / SPEC_PATH, cast("str", config["spec_sha256"]))
    bundle_root = _resolve_bundle_root(config, scratch_root=scratch_root)
    receipt = cast("Mapping[str, object]", config["input_dataset_receipt"])
    bundle = validate_fresh_input_bundle(
        bundle_root,
        expected_dataset_slug=INPUT_DATASET_SLUG,
        expected_provenance_sha256=cast("str", config["input_contract_sha256"]),
    )
    wsi_dir = _resolve_wsi_dir()
    if command == "pilot":
        _run_pilot(
            bundle_root=bundle.root,
            wsi_dir=wsi_dir,
            output_root=output_root,
            scratch_root=scratch_root,
            input_receipt=receipt,
        )
        return 0
    _run_production(
        config=config,
        config_path=config_path,
        bundle_root=bundle.root,
        wsi_dir=wsi_dir,
        output_root=output_root,
        scratch_root=scratch_root,
        input_receipt=receipt,
    )
    return 0


def _run_production(
    *,
    config: Mapping[str, object],
    config_path: Path,
    bundle_root: Path,
    wsi_dir: Path,
    output_root: Path,
    scratch_root: Path,
    input_receipt: Mapping[str, object],
) -> None:
    if os.environ.get("EQVAE_LATENT_PRODUCTION_CONFIRMED") != "1":
        message = "Production inference confirmation is missing"
        raise RuntimeError(message)
    run_number = cast("int", config["run_number"])
    pilot_path = Path(
        os.environ.get(
            "EQVAE_SPEC0021_PILOT_AUTHORITY_PATH",
            "/kaggle/working/spec0021_pilot_authority.json",
        ),
    )
    _require_hash(pilot_path, cast("str", config["pilot_authority_sha256"]))
    authority = _validate_production_evidence(
        config=config,
        input_receipt=input_receipt,
        pilot_path=pilot_path,
    )
    manifest_path = (
        bundle_root / "manifests" / "work_shards" / f"run_{run_number:02d}_of_05.csv"
    )
    manifest = load_work_manifest(
        manifest_path,
        run_number=run_number,
        expected_sha256=cast("str", config["work_manifest_sha256"]),
    )
    _validate_saved_output_size(config, manifest)
    normal_device, so2_device = require_exact_dual_t4()
    selected = cast("Mapping[str, object]", authority["selected_recipe"])
    recipe = EncoderRecipe(
        batch_size=cast("int", selected["batch_size"]),
        numeric_mode=cast("str", selected["numeric"]),  # type: ignore[arg-type]
        execution=cast("str", selected["execution"]),  # type: ignore[arg-type]
        d2h_mode=cast("str", selected["d2h"]),  # type: ignore[arg-type]
    )
    normal_model = load_frozen_checkpoint(
        bundle_root / NORMAL_CHECKPOINT_NAME,
        model_name="normal_vae",
    )
    so2_model = load_frozen_checkpoint(
        bundle_root / SO2_CHECKPOINT_NAME,
        model_name="so2_vae",
    )
    work_root = scratch_root.with_name(f".spec0021_run_{run_number:02d}_work")
    run_config_sha256 = _sha256(config_path)
    _prepare_work_root(
        work_root=work_root,
        config=config,
        run_number=run_number,
        manifest_path=manifest_path,
        current_run_config_sha256=run_config_sha256,
    )
    dual: DualLatentWriter | None = None
    try:
        normal_writer = _writer(work_root, manifest_path, run_number, "normal_vae")
        so2_writer = _writer(work_root, manifest_path, run_number, "so2_vae")
        binding = WorkerResumeBinding(
            input_bundle_sha256=cast("str", config["input_contract_sha256"]),
            run_config_sha256=run_config_sha256,
            work_manifest_sha256=manifest.sha256,
            run_number=run_number,
        )
        dual = DualLatentWriter(
            normal_writer=normal_writer,
            so2_writer=so2_writer,
            state_path=work_root / f"spec0021_worker_run_{run_number:02d}.resume.json",
            binding=binding,
        )
        worker = DualEncoderWorker(
            normal=FrozenEncoderRunner(
                normal_model,
                device=normal_device,
                recipe=recipe,
            ),
            so2=FrozenEncoderRunner(so2_model, device=so2_device, recipe=recipe),
        )
        worker.run(
            manifest=manifest,
            wsi_dir=wsi_dir,
            writers=dual,
            incomplete_path=work_root
            / f"spec0021_incomplete_run_{run_number:02d}.json",
            expected_png_identities=None,
        )
        if normal_writer.committed_rows != len(manifest.rows):
            _publish_resume_window(
                work_root=work_root,
                output_root=output_root,
                run_number=run_number,
                manifest_path=manifest_path,
                input_bundle_sha256=cast("str", config["input_contract_sha256"]),
                run_config_sha256=run_config_sha256,
            )
            return
        pair_name = f"spec0021_pair_audit_run_{run_number:02d}_of_05.json"
        finalize_pair(
            writers=dual,
            pair_audit_path=work_root / pair_name,
            expected_union_sha256=EXPECTED_UNION_SHA256,
            run_config_sha256=run_config_sha256,
            input_receipt_sha256=_canonical_hash(input_receipt),
        )
        _publish_complete_pair(work_root, output_root, run_number, pair_name)
    except BaseException:
        pair_path = work_root / f"spec0021_pair_audit_run_{run_number:02d}_of_05.json"
        if work_root.exists() and dual is not None and not pair_path.exists():
            marker = work_root / f"spec0021_incomplete_run_{run_number:02d}.json"
            dual.publish_incomplete(marker)
            _publish_resume_window(
                work_root=work_root,
                output_root=output_root,
                run_number=run_number,
                manifest_path=manifest_path,
                input_bundle_sha256=cast("str", config["input_contract_sha256"]),
                run_config_sha256=run_config_sha256,
            )
        raise
    finally:
        shutil.rmtree(work_root, ignore_errors=True)


def _run_pilot(
    *,
    bundle_root: Path,
    wsi_dir: Path,
    output_root: Path,
    scratch_root: Path,
    input_receipt: Mapping[str, object],
) -> None:
    require_exact_dual_t4()
    pilot_manifest = load_work_manifest(
        bundle_root / "manifests/work_shards/run_02_of_05.csv",
        run_number=2,
        expected_sha256=EXPECTED_WORK_MANIFEST_SHA256[2],
    )
    pilot_wsi_range(pilot_manifest)
    scratch_root.mkdir(parents=True, exist_ok=False)
    try:
        candidate_root = scratch_root / "smoke"
        candidate_root.mkdir()
        row, selection = _run_candidate(
            index=0,
            recipe=pilot_recipes()[0],
            manifest=pilot_manifest,
            bundle_root=bundle_root,
            wsi_dir=wsi_dir,
            candidate_root=candidate_root,
        )
    finally:
        shutil.rmtree(scratch_root, ignore_errors=True)
    if scratch_root.exists():
        message = "Pilot scratch cleanup failed"
        raise RuntimeError(message)
    selected = select_pilot_candidate((selection,))
    matrix_path = output_root / "spec0021_pilot_matrix.csv"
    _write_matrix(matrix_path, (row,))
    authority: dict[str, object] = {
        "schema_version": PILOT_SUMMARY_SCHEMA,
        "status": "smoke_passed",
        "smoke_passed": True,
        "smoke_wsi_id": PILOT_WSI_ID,
        "smoke_patch_count": PILOT_SMOKE_PATCHES,
        "input_dataset_receipt_sha256": _canonical_hash(input_receipt),
        "matrix_sha256": _sha256(matrix_path),
        "selected_recipe": _recipe_payload(selected.recipe),
        "runtime": _runtime_fingerprint(),
    }
    _write_json(output_root / "spec0021_pilot_authority.json", authority)


def _run_candidate(
    *,
    index: int,
    recipe: object,
    manifest: WorkManifest,
    bundle_root: Path,
    wsi_dir: Path,
    candidate_root: Path,
) -> tuple[dict[str, object], PilotCandidateResult]:
    typed_recipe = cast("PilotRecipe", recipe)
    encoder_recipe = typed_recipe.encoder_recipe()
    encoder_recipe = EncoderRecipe(
        encoder_recipe.batch_size,
        encoder_recipe.numeric_mode,
        encoder_recipe.execution,
        typed_recipe.d2h_mode,
    )
    base = _candidate_base(index, typed_recipe)
    try:
        devices = require_exact_dual_t4()
        normal_model = load_frozen_checkpoint(
            bundle_root / NORMAL_CHECKPOINT_NAME,
            model_name="normal_vae",
        )
        so2_model = load_frozen_checkpoint(
            bundle_root / SO2_CHECKPOINT_NAME,
            model_name="so2_vae",
        )
        normal = FrozenEncoderRunner(
            normal_model,
            device=devices[0],
            recipe=encoder_recipe,
        )
        so2 = FrozenEncoderRunner(so2_model, device=devices[1], recipe=encoder_recipe)
        executor = DualDeviceExecutor(normal=normal, so2=so2)
        for device in devices:
            torch.cuda.reset_peak_memory_stats(device)
        finite = True
        identity_match = True
        passes = [
            _pilot_pass(
                manifest,
                wsi_dir,
                executor,
                candidate_root,
                measure=True,
            ),
        ]
        median_seconds = statistics.median(
            item["end_to_end_seconds"] for item in passes
        )
        throughput = PILOT_SMOKE_PATCHES / median_seconds
        peak_normal = torch.cuda.max_memory_reserved(devices[0])
        peak_so2 = torch.cuda.max_memory_reserved(devices[1])
        finite = finite and all(cast("bool", item["finite"]) for item in passes)
        identity_match = identity_match and all(
            cast("bool", item["identity_match"]) for item in passes
        )
        eligible = finite and identity_match
        aggregates = {
            name: statistics.median(item[name] for item in passes)
            for name in (
                "decode_seconds",
                "h2d_seconds",
                "normalization_seconds",
                "normal_encoder_seconds",
                "so2_encoder_seconds",
                "d2h_seconds",
                "serialization_seconds",
                "fsync_seconds",
                "output_bytes",
            )
        }
        row: dict[str, object] = {
            **base,
            "status": "pass" if eligible else "smoke_fail",
            "failure_kind": "" if eligible else "finite_or_identity",
            **aggregates,
            "end_to_end_seconds_p50": median_seconds,
            "patches_per_second_p50": throughput,
            "peak_rss_bytes": _peak_rss_bytes(),
            "pinned_bytes": executor.pinned_bytes,
            "normal_peak_reserved_bytes": peak_normal,
            "so2_peak_reserved_bytes": peak_so2,
            "peak_total_gpu_bytes": peak_normal + peak_so2,
            "finite": finite,
            "identity_match": identity_match,
            "eligible": eligible,
            "pilot_validation_seconds": statistics.median(
                cast("float", item["validation_seconds"]) for item in passes
            ),
        }
        selection = PilotCandidateResult(
            recipe=typed_recipe,
            eligible=eligible,
            median_patches_per_second=throughput,
            peak_total_gpu_bytes=peak_normal + peak_so2,
            peak_rss_bytes=cast("int", row["peak_rss_bytes"]),
        )
        return row, selection
    except (RuntimeError, ValueError, torch.OutOfMemoryError) as error:
        torch.cuda.empty_cache()
        return _failed_candidate(
            index,
            typed_recipe,
            failure_kind=type(error).__name__,
            failure=str(error),
        )


def _candidate_base(index: int, recipe: PilotRecipe) -> dict[str, object]:
    return {
        "candidate_index": index,
        "batch_size": recipe.batch_size,
        "d2h": recipe.d2h_mode,
        "numeric": recipe.numeric_mode,
        "execution": recipe.execution,
        "warmup_passes": 0,
        "measured_passes": MEASURED_PASSES,
        "patch_count": PILOT_SMOKE_PATCHES,
    }


def _failed_candidate(
    index: int,
    recipe: PilotRecipe,
    *,
    failure_kind: str,
    failure: str,
) -> tuple[dict[str, object], PilotCandidateResult]:
    base = _candidate_base(index, recipe)
    row = cast(
        "dict[str, object]",
        {
            **dict.fromkeys(MATRIX_HEADER, 0),
            **base,
            "status": "failed",
            "failure_kind": failure_kind,
            "finite": False,
            "identity_match": False,
            "eligible": False,
        },
    )
    return row, PilotCandidateResult(
        recipe=recipe,
        eligible=False,
        median_patches_per_second=0.0,
        peak_total_gpu_bytes=0,
        peak_rss_bytes=0,
        failure=failure,
    )


def _pilot_pass(
    manifest: WorkManifest,
    wsi_dir: Path,
    executor: DualDeviceExecutor,
    scratch_root: Path,
    *,
    measure: bool,
    reader: WSIReader | None = None,
) -> dict[str, float | int]:
    start_row, stop_row = pilot_wsi_range(manifest)
    normal_path = scratch_root / "normal.bin"
    so2_path = scratch_root / "so2.bin"
    normal_path.unlink(missing_ok=True)
    so2_path.unlink(missing_ok=True)
    totals = {
        "decode_seconds": 0.0,
        "h2d_seconds": 0.0,
        "normalization_seconds": 0.0,
        "normal_encoder_seconds": 0.0,
        "so2_encoder_seconds": 0.0,
        "d2h_seconds": 0.0,
        "serialization_seconds": 0.0,
        "fsync_seconds": 0.0,
    }
    next_identity_row = start_row
    identity_match = True
    finite = True
    started = time.perf_counter()
    iterator = iter_wsi_patch_interval(
        manifest=manifest,
        wsi_dir=wsi_dir,
        batch_size=executor.recipe.batch_size,
        start_row=start_row,
        stop_row=stop_row,
        reader=reader,
    )
    sentinel_rows = set(pilot_sentinel_indices(manifest))
    with normal_path.open("ab") as normal_handle, so2_path.open("ab") as so2_handle:
        while True:
            decode_started = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                break
            totals["decode_seconds"] += time.perf_counter() - decode_started
            for offset, image in enumerate(batch.images_uint8):
                manifest_index = batch.row_start + offset
                if manifest_index not in sentinel_rows:
                    continue
                direct = read_wsi_patch(
                    identity=manifest.rows[manifest_index].identity,
                    wsi_dir=wsi_dir,
                    reader=reader,
                )
                if not torch.equal(image, direct):
                    message = (
                        f"Pilot sentinel mismatch at manifest index {manifest_index}"
                    )
                    raise ValueError(message)
            executor.submit(
                row_start=batch.row_start,
                identities=batch.identities,
                images_uint8=batch.images_uint8,
                final_wsi_evidence=batch.final_wsi_evidence,
            )
            if executor.pending_count == executor.slot_count:
                next_identity_row, slot_finite, slot_identity_match = _drain_pilot_slot(
                    executor,
                    normal_handle,
                    so2_handle,
                    totals,
                    manifest=manifest,
                    expected_row=next_identity_row,
                )
                finite = finite and slot_finite
                identity_match = identity_match and slot_identity_match
        while executor.pending_count:
            next_identity_row, slot_finite, slot_identity_match = _drain_pilot_slot(
                executor,
                normal_handle,
                so2_handle,
                totals,
                manifest=manifest,
                expected_row=next_identity_row,
            )
            finite = finite and slot_finite
            identity_match = identity_match and slot_identity_match
        fsync_started = time.perf_counter()
        normal_handle.flush()
        so2_handle.flush()
        os.fsync(normal_handle.fileno())
        os.fsync(so2_handle.fileno())
        totals["fsync_seconds"] += time.perf_counter() - fsync_started
    elapsed = time.perf_counter() - started
    expected_model_bytes = PILOT_SMOKE_PATCHES * 65_536
    if (
        normal_path.stat().st_size != expected_model_bytes
        or so2_path.stat().st_size != expected_model_bytes
    ):
        message = "Tiny pilot payload size does not equal exactly 16 latent rows"
        raise ValueError(message)
    output_bytes = 2 * expected_model_bytes
    validation_seconds = 0.0
    if measure:
        validation_started = time.perf_counter()
        _validate_pilot_payload(normal_path)
        _validate_pilot_payload(so2_path)
        validation_seconds = time.perf_counter() - validation_started
    remove_pilot_payload(normal_path)
    remove_pilot_payload(so2_path)
    return {
        **totals,
        "end_to_end_seconds": elapsed if measure else 0.0,
        "output_bytes": output_bytes,
        "finite": finite,
        "identity_match": identity_match and next_identity_row == stop_row,
        "validation_seconds": validation_seconds,
    }


def _drain_pilot_slot(
    executor: DualDeviceExecutor,
    normal_handle: BinaryIO,
    so2_handle: BinaryIO,
    totals: dict[str, float],
    *,
    manifest: WorkManifest,
    expected_row: int,
) -> tuple[int, bool, bool]:
    encoded = executor.drain_one()
    try:
        expected_identities = tuple(
            row.identity
            for row in manifest.rows[
                encoded.row_start : encoded.row_start + len(encoded.identities)
            ]
        )
        identity_match = (
            encoded.identity_match
            and encoded.row_start == expected_row
            and encoded.identities == expected_identities
        )
        _add_dual_timing(totals, encoded.timing)
        serialization_started = time.perf_counter()
        normal_handle.write(_payload_bytes(encoded.normal_tensors))
        so2_handle.write(_payload_bytes(encoded.so2_tensors))
        totals["serialization_seconds"] += time.perf_counter() - serialization_started
        return (
            encoded.row_start + len(encoded.identities),
            encoded.finite,
            identity_match,
        )
    finally:
        executor.release(encoded)


def _validate_saved_output_size(
    config: Mapping[str, object],
    manifest: WorkManifest,
) -> None:
    expected = 2 * (LATENT_SHARD_HEADER_SIZE + len(manifest.rows) * LATENT_RECORD_BYTES)
    if config.get("expected_binary_output_bytes") != expected:
        message = "Derived production binary size differs from the embedded config"
        raise ValueError(message)
    if expected + PUBLISHED_METADATA_RESERVE_BYTES > cast(
        "int",
        config["saved_output_limit_bytes"],
    ):
        message = "Production artifacts exceed Kaggle's saved-output cap"
        raise ValueError(message)


def _writer(
    root: Path,
    manifest_path: Path,
    run_number: int,
    model_name: str,
) -> LatentShardWriter:
    typed = "normal_vae" if model_name == "normal_vae" else "so2_vae"
    return LatentShardWriter(
        bin_path=root / f"{typed}_mu_run_{run_number:02d}_of_05.bin",
        manifest_path=manifest_path,
        run_number=run_number,
        model_name=typed,
        checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256[typed],
        expected_checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256[typed],
        expected_manifest_sha256=EXPECTED_WORK_MANIFEST_SHA256[run_number],
        expected_union_sha256=EXPECTED_UNION_SHA256,
    )


def _prepare_work_root(
    *,
    work_root: Path,
    config: Mapping[str, object],
    run_number: int,
    manifest_path: Path,
    current_run_config_sha256: str,
) -> None:
    raw_receipt = config.get("resume_dataset_receipt")
    if raw_receipt is None:
        work_root.mkdir(parents=True, exist_ok=False)
        return
    if not isinstance(raw_receipt, dict):
        message = "Resume dataset receipt must be an object"
        raise TypeError(message)
    receipt = cast("Mapping[str, object]", raw_receipt)
    expected_reference = (
        f"maximusshtefan/eqvae-ubc-ocean-latent-run-{run_number:02d}-resume"
    )
    if (
        receipt.get("schema_version") != "spec0021.resume_dataset_receipt.v1"
        or receipt.get("dataset_reference") != expected_reference
        or receipt.get("run_number") != run_number
        or receipt.get("input_bundle_sha256") != config["input_contract_sha256"]
        or receipt.get("work_manifest_sha256") != config["work_manifest_sha256"]
    ):
        message = "Resume dataset receipt disagrees with the production config"
        raise ValueError(message)
    version = _positive_int(receipt.get("dataset_version"), "resume dataset version")
    provenance = _sha256_field(receipt, "provenance_sha256")
    previous_config_sha256 = _sha256_field(receipt, "run_config_sha256")
    source = _resolve_receipt_mount_root(
        receipt,
        required_filenames=(RESUME_PROVENANCE_FILENAME,),
        label="immutable resume bundle",
    )
    authority = ResumeBundleAuthority(
        provenance_sha256=provenance,
        dataset_slug=expected_reference,
        dataset_version=version,
        run_number=run_number,
        input_bundle_sha256=cast("str", config["input_contract_sha256"]),
        run_config_sha256=previous_config_sha256,
        work_manifest_sha256=cast("str", config["work_manifest_sha256"]),
    )
    copy_validated_resume_bundle(
        source,
        work_root,
        authority=authority,
        work_manifest_path=manifest_path,
    )
    _rebind_worker_resume(
        work_root / f"spec0021_worker_run_{run_number:02d}.resume.json",
        expected_previous_sha256=previous_config_sha256,
        current_sha256=current_run_config_sha256,
    )


def _rebind_worker_resume(
    path: Path,
    *,
    expected_previous_sha256: str,
    current_sha256: str,
) -> None:
    payload = _read_object(path)
    if payload.get("run_config_sha256") != expected_previous_sha256:
        message = "Copied worker journal does not match the resume receipt"
        raise ValueError(message)
    payload["run_config_sha256"] = current_sha256
    _replace_json_fsync(path, payload)


def _publish_complete_pair(
    work_root: Path,
    output_root: Path,
    run_number: int,
    pair_name: str,
) -> None:
    names = (
        f"normal_vae_mu_run_{run_number:02d}_of_05.bin",
        f"normal_vae_mu_run_{run_number:02d}_of_05.json",
        f"so2_vae_mu_run_{run_number:02d}_of_05.bin",
        f"so2_vae_mu_run_{run_number:02d}_of_05.json",
        pair_name,
    )
    _publish_link_set(
        source_root=work_root,
        output_root=output_root,
        names=names,
        final_name=pair_name,
    )


def _publish_resume_window(
    *,
    work_root: Path,
    output_root: Path,
    run_number: int,
    manifest_path: Path,
    input_bundle_sha256: str,
    run_config_sha256: str,
) -> None:
    marker_name = f"spec0021_incomplete_run_{run_number:02d}.json"
    marker = work_root / marker_name
    if not marker.is_file():
        message = "A validated resume publication requires its incomplete marker"
        raise ValueError(message)
    validation_root = work_root.with_name(f".{work_root.name}.validated_resume")
    try:
        stage_resume_bundle(
            validation_root,
            artifacts_dir=work_root,
            work_manifest_path=manifest_path,
            dataset_slug=(
                f"maximusshtefan/eqvae-ubc-ocean-latent-run-{run_number:02d}-resume"
            ),
            dataset_version=1,
            run_number=run_number,
            input_bundle_sha256=input_bundle_sha256,
            run_config_sha256=run_config_sha256,
        )
        names = tuple(
            sorted(
                path.name
                for path in validation_root.iterdir()
                if path.is_file()
                and path.name
                not in {"dataset-metadata.json", RESUME_PROVENANCE_FILENAME}
            ),
        )
        if marker_name not in names:
            message = "Validated resume bundle omitted its incomplete marker"
            raise ValueError(message)
        _publish_link_set(
            source_root=validation_root,
            output_root=output_root,
            names=names,
            final_name=marker_name,
        )
    finally:
        shutil.rmtree(validation_root, ignore_errors=True)


def _publish_link_set(
    *,
    source_root: Path,
    output_root: Path,
    names: Sequence[str],
    final_name: str,
) -> None:
    if not output_root.is_dir() or any(output_root.iterdir()):
        message = "Inference output directory must exist and be empty"
        raise ValueError(message)
    if final_name not in names:
        message = "Publication completion marker is missing from its allow-list"
        raise ValueError(message)
    ordered = (*sorted(name for name in names if name != final_name), final_name)
    published: list[Path] = []
    try:
        for name in ordered:
            source = source_root / name
            target = output_root / name
            _require_safe_link(source, target, name)
            os.link(source, target)
            published.append(target)
            if name == final_name or len(published) == len(ordered) - 1:
                _fsync_directory(output_root)
    except BaseException:
        for path in reversed(published):
            path.unlink(missing_ok=True)
        _fsync_directory(output_root)
        raise


def _recipe_payload(recipe: PilotRecipe) -> dict[str, object]:
    return {
        "batch_size": recipe.batch_size,
        "d2h": recipe.d2h_mode,
        "numeric": recipe.numeric_mode,
        "execution": recipe.execution,
    }


def _runtime_fingerprint() -> dict[str, object]:
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "devices": [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
            }
            for index in range(torch.cuda.device_count())
        ],
    }


def _validate_production_evidence(
    *,
    config: Mapping[str, object],
    input_receipt: Mapping[str, object],
    pilot_path: Path,
) -> dict[str, object]:
    authority = _read_object(pilot_path)
    if (
        authority.get("schema_version") != PILOT_SUMMARY_SCHEMA
        or authority.get("status") != "smoke_passed"
        or authority.get("smoke_passed") is not True
        or authority.get("smoke_wsi_id") != PILOT_WSI_ID
        or authority.get("smoke_patch_count") != PILOT_SMOKE_PATCHES
        or authority.get("input_dataset_receipt_sha256")
        != _canonical_hash(input_receipt)
        or authority.get("selected_recipe") != config.get("selected_recipe")
    ):
        message = "Tiny pilot smoke evidence does not match production config"
        raise ValueError(message)
    return authority


def _require_safe_link(source: Path, target: Path, name: str) -> None:
    if not source.is_file() or source.is_symlink() or target.exists():
        message = f"Unsafe publication source or target: {name}"
        raise ValueError(message)


def _replace_json_fsync(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    encoded = f"{json.dumps(payload, sort_keys=True, separators=(',', ':'))}\n".encode()
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sha256_field(payload: Mapping[str, object], name: str) -> str:
    value = payload.get(name)
    if (
        not isinstance(value, str)
        or len(value) != SHA256_HEX_LENGTH
        or any(char not in "0123456789abcdef" for char in value)
    ):
        message = f"{name} is not a SHA-256"
        raise ValueError(message)
    return value


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        message = f"{name} must be a positive integer"
        raise ValueError(message)
    return value


def _validate_pilot_download(input_root: Path, output_path: Path) -> None:
    dataset = (
        input_root / "dataset" if (input_root / "dataset").is_dir() else input_root
    )
    expected_files = {
        "spec0021_pilot_matrix.csv",
        "spec0021_pilot_authority.json",
    }
    if {path.name for path in dataset.iterdir()} != expected_files or any(
        path.is_symlink() or not path.is_file() for path in dataset.iterdir()
    ):
        message = "Downloaded pilot output allow-list differs"
        raise ValueError(message)
    matrix = dataset / "spec0021_pilot_matrix.csv"
    authority_path = dataset / "spec0021_pilot_authority.json"
    authority = _read_object(authority_path)
    if (
        authority.get("schema_version") != PILOT_SUMMARY_SCHEMA
        or authority.get("status") != "smoke_passed"
        or authority.get("smoke_passed") is not True
        or authority.get("smoke_wsi_id") != PILOT_WSI_ID
        or authority.get("smoke_patch_count") != PILOT_SMOKE_PATCHES
        or authority.get("matrix_sha256") != _sha256(matrix)
    ):
        message = "Downloaded tiny pilot did not pass its smoke contract"
        raise ValueError(message)
    _sha256_field(authority, "input_dataset_receipt_sha256")
    with matrix.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if (
            tuple(reader.fieldnames or ()) != MATRIX_HEADER
            or len(rows) != PILOT_CANDIDATE_COUNT
        ):
            message = "Pilot matrix header or row count mismatch"
            raise ValueError(message)
    for index, (row, recipe) in enumerate(zip(rows, pilot_recipes(), strict=True)):
        if (
            row["candidate_index"] != str(index)
            or row["batch_size"] != str(recipe.batch_size)
            or row["d2h"] != recipe.d2h_mode
            or row["numeric"] != recipe.numeric_mode
            or row["execution"] != recipe.execution
            or row["status"] != "pass"
            or row["failure_kind"]
            or row["warmup_passes"] != "0"
            or row["measured_passes"] != "1"
            or row["patch_count"] != str(PILOT_SMOKE_PATCHES)
            or row["output_bytes"] != str(2 * PILOT_SMOKE_PATCHES * 65_536)
            or row["finite"].casefold() != "true"
            or row["identity_match"].casefold() != "true"
            or row["eligible"].casefold() != "true"
        ):
            message = "Pilot matrix does not prove the exact tiny smoke pass"
            raise ValueError(message)
    selected = authority.get("selected_recipe")
    if not isinstance(selected, dict) or not any(
        row["eligible"].casefold() == "true"
        and {
            "batch_size": int(row["batch_size"]),
            "d2h": row["d2h"],
            "numeric": row["numeric"],
            "execution": row["execution"],
        }
        == selected
        for row in rows
    ):
        message = "Pilot selected recipe is not an eligible matrix row"
        raise ValueError(message)
    receipt = {
        "schema_version": "spec0021.pilot_receipt.v1",
        "status": "smoke_passed",
        "authority_sha256": _sha256(authority_path),
        "matrix_sha256": _sha256(matrix),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, receipt)


def _validate_config(path: Path, *, expected_mode: str) -> dict[str, object]:
    config = _read_object(path)
    expected_fields = {
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
    }
    if set(config) != expected_fields or config.get("schema_version") != CONFIG_SCHEMA:
        message = "Inference config schema or fields mismatch"
        raise ValueError(message)
    if config.get("mode") != expected_mode:
        message = "Inference command and embedded config mode differ"
        raise ValueError(message)
    if config.get("input_dataset_receipt") is None:
        message = "Remote inference requires a pinned input dataset receipt"
        raise ValueError(message)
    if (
        config.get("normal_checkpoint_sha256")
        != EXPECTED_CHECKPOINT_SHA256["normal_vae"]
    ):
        message = "Normal checkpoint config hash mismatch"
        raise ValueError(message)
    if config.get("so2_checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256["so2_vae"]:
        message = "SO2 checkpoint config hash mismatch"
        raise ValueError(message)
    receipt = config.get("input_dataset_receipt")
    if not isinstance(receipt, dict):
        message = "Input dataset receipt must be an object"
        raise TypeError(message)
    _sha256_field(config, "input_contract_sha256")
    _sha256_field(config, "spec_sha256")
    mode = cast("str", config["mode"])
    if mode == "pilot":
        for name in (
            "run_number",
            "work_manifest_sha256",
            "pilot_authority_sha256",
            "selected_recipe",
            "expected_binary_output_bytes",
            "saved_output_limit_bytes",
            "resume_dataset_receipt",
        ):
            if config.get(name) is not None:
                message = f"Pilot config field {name} must be null"
                raise ValueError(message)
    else:
        run_number = _positive_int(config.get("run_number"), "production run")
        if run_number not in EXPECTED_WORK_MANIFEST_SHA256:
            message = "Production run must be one of 1 through 5"
            raise ValueError(message)
        if (
            config.get("work_manifest_sha256")
            != EXPECTED_WORK_MANIFEST_SHA256[run_number]
        ):
            message = "Production work manifest hash mismatch"
            raise ValueError(message)
        _sha256_field(config, "pilot_authority_sha256")
        selected = config.get("selected_recipe")
        if not isinstance(selected, dict):
            message = "Production selected recipe must be an object"
            raise TypeError(message)
        selected_mapping = cast("Mapping[str, object]", selected)
        if set(selected_mapping) != {
            "batch_size",
            "d2h",
            "numeric",
            "execution",
        }:
            message = "Production selected recipe fields differ"
            raise ValueError(message)
        if selected_mapping != _recipe_payload(pilot_recipes()[0]):
            message = "Production must use the fixed conservative smoke recipe"
            raise ValueError(message)
        expected_binary_output_bytes = 2 * (
            LATENT_SHARD_HEADER_SIZE + WORK_ROW_COUNTS[run_number] * LATENT_RECORD_BYTES
        )
        if config.get("expected_binary_output_bytes") != expected_binary_output_bytes:
            message = "Production expected binary output bytes mismatch"
            raise ValueError(message)
        if config.get("saved_output_limit_bytes") != KAGGLE_SAVED_OUTPUT_LIMIT_BYTES:
            message = "Production saved-output cap mismatch"
            raise ValueError(message)
        if (
            expected_binary_output_bytes + PUBLISHED_METADATA_RESERVE_BYTES
            > KAGGLE_SAVED_OUTPUT_LIMIT_BYTES
        ):
            message = "Production output does not fit Kaggle's saved-output cap"
            raise ValueError(message)
    return config


def _resolve_bundle_root(
    config: Mapping[str, object],
    *,
    scratch_root: Path,
) -> Path:
    receipt = cast("Mapping[str, object]", config["input_dataset_receipt"])
    mounted = _resolve_receipt_mount_root(
        receipt,
        required_filenames=(
            FRESH_PROVENANCE_FILENAME,
            FRESH_UPLOAD_ARCHIVE_FILENAME,
        ),
        label="immutable input bundle",
    )
    direct_contract = mounted / FRESH_PROVENANCE_FILENAME
    archive = mounted / FRESH_UPLOAD_ARCHIVE_FILENAME
    if direct_contract.is_file() and not archive.exists():
        return mounted
    if archive.is_file() and not direct_contract.exists():
        return extract_fresh_upload_archive(
            archive,
            scratch_root / "input_bundle",
            expected_dataset_slug=INPUT_DATASET_SLUG,
            expected_provenance_sha256=cast(
                "str",
                config["input_contract_sha256"],
            ),
        ).root
    message = "Mounted immutable input must expose exactly contract or bundle.zip"
    raise ValueError(message)


def _resolve_receipt_mount_root(
    receipt: Mapping[str, object],
    *,
    required_filenames: tuple[str, ...],
    label: str,
    input_root: Path = Path("/kaggle/input"),
) -> Path:
    reference = receipt.get("dataset_reference")
    if not isinstance(reference, str):
        message = f"{label} receipt reference must be text"
        raise TypeError(message)
    parts = reference.split("@", maxsplit=1)[0].split("/")
    if len(parts) != 2 or any(not part or part in {".", ".."} for part in parts):
        message = f"{label} receipt reference must be owner/slug"
        raise ValueError(message)
    owner, slug = parts
    version = _positive_int(receipt.get("dataset_version"), f"{label} version")
    nested = input_root / "datasets" / owner / slug
    candidates = (
        input_root / slug,
        input_root / "datasets" / slug,
        input_root / owner / slug,
        nested,
        nested / "versions" / str(version),
    )
    input_resolved = input_root.resolve()
    matches: dict[Path, Path] = {}
    for candidate in candidates:
        if not candidate.is_dir() or candidate.is_symlink():
            continue
        resolved = candidate.resolve()
        if input_resolved not in resolved.parents:
            message = f"{label} mount escapes the Kaggle input root"
            raise ValueError(message)
        required = [candidate / name for name in required_filenames]
        if any(path.is_file() and not path.is_symlink() for path in required):
            matches.setdefault(resolved, candidate)
    if len(matches) != 1:
        message = f"Expected exactly one mounted {label}; found {len(matches)}"
        raise ValueError(message)
    return next(iter(matches.values()))


def _resolve_wsi_dir() -> Path:
    candidates = (
        Path("/kaggle/input/UBC-OCEAN/train_images"),
        Path("/kaggle/input/competitions/UBC-OCEAN/train_images"),
    )
    matches = [path for path in candidates if path.is_dir()]
    if len(matches) != 1:
        message = "Expected exactly one official UBC-OCEAN train_images mount"
        raise ValueError(message)
    return matches[0]


def _add_dual_timing(
    totals: dict[str, float],
    timing: DualEncoderTiming,
) -> None:
    totals["h2d_seconds"] += timing.h2d_seconds
    totals["normalization_seconds"] += timing.normalization_seconds
    totals["normal_encoder_seconds"] += timing.normal_encoder_seconds
    totals["so2_encoder_seconds"] += timing.so2_encoder_seconds
    totals["d2h_seconds"] += timing.d2h_seconds


def _payload_bytes(tensors: torch.Tensor) -> bytes:
    return tensors.contiguous().numpy().astype("<f4", copy=False).tobytes()


def _validate_pilot_payload(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    checksum = 0
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            checksum = zlib.crc32(chunk, checksum)
            digest.update(chunk)
    return checksum & 0xFFFFFFFF, digest.hexdigest()


def _write_matrix(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MATRIX_HEADER, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in MATRIX_HEADER})


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    encoded = f"{json.dumps(payload, indent=2, sort_keys=True)}\n".encode()
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _read_object(path: Path) -> dict[str, object]:
    payload = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(payload, dict):
        message = f"Expected JSON object in {path}"
        raise TypeError(message)
    return cast("dict[str, object]", payload)


def _require_hash(path: Path, expected: str) -> None:
    if _sha256(path) != expected:
        message = f"SHA-256 mismatch for {path.name}"
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _peak_rss_bytes() -> int:
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith("VmHWM:"):
            fields = line.split()
            if len(fields) != PROC_STATUS_FIELDS or fields[2] != "kB":
                break
            return int(fields[1]) * 1024
    message = "Linux /proc/self/status has no canonical VmHWM value"
    raise RuntimeError(message)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("pilot", "production"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--config", required=True)
        subparser.add_argument("--payload-root", required=True)
        subparser.add_argument("--output-root", required=True)
        subparser.add_argument("--scratch-root", required=True)
    validator = subparsers.add_parser("validate-pilot")
    validator.add_argument("--input", required=True)
    validator.add_argument("--output", required=True)
    return parser


if TYPE_CHECKING:
    from eqvae.inference.pilot import PilotRecipe


if __name__ == "__main__":
    raise SystemExit(main())
