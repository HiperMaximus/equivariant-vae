# Copyright 2026 HiperMaximus
# ruff: noqa: DOC201, PLR0913
"""Validate ten latent shards and publish split locations plus global audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

from eqvae.benchmarking.torch_runtime import torch_runtime_versions
from eqvae.data.latent_shards import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_TASK_MANIFEST_SHA256,
    EXPECTED_UNION_SHA256,
    EXPECTED_WORK_MANIFEST_SHA256,
    LATENT_LOCATION_HEADER,
    LatentRowIdentity,
    LatentTaskView,
    WorkManifestRow,
    load_work_manifest,
    validate_latent_store_pair,
)
from eqvae.inference.input_bundle import validate_fresh_input_bundle

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from eqvae.data.latent_shards import ModelName

GLOBAL_AUDIT_SCHEMA: Final = "spec0021.latent_store_global_audit.v1"
INPUT_DATASET_SLUG: Final = "maximusshtefan/eqvae-ubc-ocean-latent-inputs"
LOGICAL_VIEW_COUNT: Final = 12
SHA256_HEX_LENGTH: Final = 64


def main(argv: Sequence[str] | None = None) -> int:  # noqa: PLR0914
    """Run full payload validation and atomically publish the logical views."""
    args = _parser().parse_args(argv)
    contract_path = Path(cast("str", args.input_contract)).resolve()
    config_root = Path(cast("str", args.config_root)).resolve()
    input_receipt_path = Path(cast("str", args.input_receipt)).resolve()
    pair_root = Path(cast("str", args.pair_root)).resolve()
    output_root = Path(cast("str", args.output_root)).resolve()
    bundle = validate_fresh_input_bundle(
        contract_path.parent,
        expected_dataset_slug=INPUT_DATASET_SLUG,
        expected_provenance_sha256=_sha256(contract_path),
    )
    manifests_root = bundle.root / "manifests"
    work_paths = {
        run: manifests_root / "work_shards" / f"run_{run:02d}_of_05.csv"
        for run in range(1, 6)
    }
    task_paths = {
        name: manifests_root / "task_views" / f"{name}.csv"
        for name in EXPECTED_TASK_MANIFEST_SHA256
    }
    normal_shards, so2_shards, pair_audits = _pair_paths(pair_root)
    expected_run_config_hashes, expected_input_receipt_sha256 = (
        _expected_production_bindings(
            config_root=config_root,
            input_receipt_path=input_receipt_path,
            input_contract_sha256=_sha256(contract_path),
        )
    )
    result = validate_latent_store_pair(
        union_manifest_path=manifests_root / "union_patch_manifest.csv",
        expected_union_sha256=EXPECTED_UNION_SHA256,
        work_manifest_paths=work_paths,
        expected_work_manifest_hashes=EXPECTED_WORK_MANIFEST_SHA256,
        task_manifest_paths=task_paths,
        expected_task_manifest_hashes=EXPECTED_TASK_MANIFEST_SHA256,
        normal_shards=normal_shards,
        so2_shards=so2_shards,
        normal_checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256["normal_vae"],
        expected_normal_checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256["normal_vae"],
        so2_checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256["so2_vae"],
        expected_so2_checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256["so2_vae"],
        validate_payload=True,
    )
    artifact_records = _validate_pair_audits(
        work_paths=work_paths,
        pair_audits=pair_audits,
        normal_shards=normal_shards,
        so2_shards=so2_shards,
        expected_run_config_hashes=expected_run_config_hashes,
        expected_input_receipt_sha256=expected_input_receipt_sha256,
        validated_file_hashes=_mapping_field(result, "artifact_file_sha256"),
    )
    _publish_output(
        output_root=output_root,
        work_paths=work_paths,
        task_paths=task_paths,
        normal_shards=normal_shards,
        so2_shards=so2_shards,
        artifact_records=artifact_records,
        validation=result,
        input_contract=contract_path,
    )
    return 0


def _pair_paths(
    pair_root: Path,
) -> tuple[dict[int, Path], dict[int, Path], dict[int, Path]]:
    normal: dict[int, Path] = {}
    so2: dict[int, Path] = {}
    audits: dict[int, Path] = {}
    expected_run_names = {f"run_{run:02d}_of_05" for run in range(1, 6)}
    observed_run_names = {path.name for path in pair_root.iterdir()}
    if observed_run_names != expected_run_names:
        message = "Pair root does not contain exactly runs 01 through 05"
        raise ValueError(message)
    for run in range(1, 6):
        run_root = pair_root / f"run_{run:02d}_of_05"
        nested = run_root / "dataset"
        dataset = nested if nested.is_dir() else run_root
        if dataset == nested and (
            {path.name for path in run_root.iterdir()} != {"dataset"}
            or nested.is_symlink()
        ):
            message = f"Run {run:02d} does not contain exactly physical dataset/"
            raise ValueError(message)
        normal[run] = dataset / f"normal_vae_mu_run_{run:02d}_of_05.bin"
        so2[run] = dataset / f"so2_vae_mu_run_{run:02d}_of_05.bin"
        audits[run] = dataset / f"spec0021_pair_audit_run_{run:02d}_of_05.json"
        expected_files = {
            normal[run].name,
            normal[run].with_suffix(".json").name,
            so2[run].name,
            so2[run].with_suffix(".json").name,
            audits[run].name,
        }
        observed_files = {path.name for path in dataset.iterdir()}
        if observed_files != expected_files or any(
            path.is_symlink() or not path.is_file() for path in dataset.iterdir()
        ):
            message = f"Run {run:02d} complete output allow-list differs"
            raise ValueError(message)
    return normal, so2, audits


def _validate_pair_audits(  # noqa: C901, PLR0912, PLR0914, PLR0915
    *,
    work_paths: Mapping[int, Path],
    pair_audits: Mapping[int, Path],
    normal_shards: Mapping[int, Path],
    so2_shards: Mapping[int, Path],
    expected_run_config_hashes: Mapping[int, str],
    expected_input_receipt_sha256: str,
    validated_file_hashes: Mapping[str, object],
) -> dict[str, object]:
    pair_records: dict[str, object] = {}
    model_records: dict[str, object] = {"normal_vae": {}, "so2_vae": {}}
    expected_keys = {
        "schema_version",
        "status",
        "run_number",
        "row_count",
        "work_manifest_sha256",
        "union_manifest_sha256",
        "run_config_sha256",
        "input_receipt_sha256",
        "completed_wsi_evidence",
        "artifacts",
    }
    if set(expected_run_config_hashes) != set(range(1, 6)):
        message = "Expected run-config hashes must contain exactly runs 1 through 5"
        raise ValueError(message)
    _require_sha256(expected_input_receipt_sha256, "expected input receipt")
    for run in range(1, 6):
        manifest = load_work_manifest(
            work_paths[run],
            run_number=run,
            expected_sha256=EXPECTED_WORK_MANIFEST_SHA256[run],
        )
        audit_path = pair_audits[run]
        audit = _read_object(audit_path)
        canonical = f"{json.dumps(audit, indent=2, sort_keys=True)}\n"
        if audit_path.read_text(encoding="utf-8") != canonical:
            message = f"Pair audit for run {run:02d} is not canonical JSON"
            raise ValueError(message)
        expected_top = {
            "schema_version": "spec0021.latent_pair_audit.v1",
            "status": "complete",
            "run_number": run,
            "row_count": len(manifest.rows),
            "work_manifest_sha256": EXPECTED_WORK_MANIFEST_SHA256[run],
            "union_manifest_sha256": EXPECTED_UNION_SHA256,
        }
        if set(audit) != expected_keys or any(
            audit.get(key) != value for key, value in expected_top.items()
        ):
            message = f"Pair audit for run {run:02d} disagrees with its manifest"
            raise ValueError(message)
        expected_provenance = {
            "run_config_sha256": _require_sha256(
                expected_run_config_hashes[run],
                f"expected run {run:02d} config",
            ),
            "input_receipt_sha256": expected_input_receipt_sha256,
        }
        if any(audit.get(key) != value for key, value in expected_provenance.items()):
            message = f"Pair audit provenance differs for run {run:02d}"
            raise ValueError(message)
        artifacts = _mapping_field(audit, "artifacts")
        if set(artifacts) != {
            "normal_vae",
            "so2_vae",
        }:
            message = f"Pair audit artifacts differ for run {run:02d}"
            raise ValueError(message)
        for model_name, shards in (
            ("normal_vae", normal_shards),
            ("so2_vae", so2_shards),
        ):
            bin_path = shards[run]
            sidecar_path = bin_path.with_suffix(".json")
            model_hashes = _mapping_field(validated_file_hashes, model_name)
            bin_sha256 = _require_sha256(
                model_hashes.get(str(run)),
                f"validated {model_name} run {run:02d} binary",
            )
            identity = {
                "bin_name": bin_path.name,
                "bin_bytes": bin_path.stat().st_size,
                "bin_sha256": bin_sha256,
                "sidecar_name": sidecar_path.name,
                "sidecar_bytes": sidecar_path.stat().st_size,
                "sidecar_sha256": _sha256(sidecar_path),
            }
            if artifacts.get(model_name) != identity:
                message = f"Pair audit {model_name} identity differs for run {run:02d}"
                raise ValueError(message)
            cast("dict[str, object]", model_records[model_name])[str(run)] = {
                "binary": {
                    "name": bin_path.name,
                    "bytes": bin_path.stat().st_size,
                    "sha256": bin_sha256,
                },
                "sidecar": _file_record(sidecar_path),
            }
        raw_evidence = audit.get("completed_wsi_evidence")
        if not isinstance(raw_evidence, list):
            message = f"Pair audit WSI evidence differs for run {run:02d}"
            raise TypeError(message)
        evidence = cast("list[object]", raw_evidence)
        if len(evidence) != len(manifest.wsi_ranges):
            message = f"Pair audit WSI evidence count differs for run {run:02d}"
            raise ValueError(message)
        expected_wsi_ids = [wsi_id for wsi_id, _start, _end in manifest.wsi_ranges]
        for expected_wsi_id, raw in zip(expected_wsi_ids, evidence, strict=True):
            if not isinstance(raw, dict):
                message = f"Pair audit WSI evidence must be an object for run {run:02d}"
                raise TypeError(message)
            record = cast("Mapping[str, object]", raw)
            if set(record) != {
                "wsi_id",
                "png_bytes",
                "png_sha256",
                "transcript_sha256",
            }:
                message = f"Pair audit WSI evidence schema differs for run {run:02d}"
                raise ValueError(message)
            if record.get("wsi_id") != expected_wsi_id:
                message = f"Pair audit WSI order differs for run {run:02d}"
                raise ValueError(message)
            png_bytes = record.get("png_bytes")
            if (
                isinstance(png_bytes, bool)
                or not isinstance(png_bytes, int)
                or png_bytes <= 0
            ):
                message = f"Pair audit PNG size is invalid for run {run:02d}"
                raise ValueError(message)
            _require_sha256(record.get("png_sha256"), "pair audit PNG")
            _require_sha256(record.get("transcript_sha256"), "pair audit transcript")
        pair_records[str(run)] = _file_record(audit_path)
    return {"pair_audits": pair_records, "models": model_records}


def _publish_output(
    *,
    output_root: Path,
    work_paths: Mapping[int, Path],
    task_paths: Mapping[str, Path],
    normal_shards: Mapping[int, Path],
    so2_shards: Mapping[int, Path],
    artifact_records: Mapping[str, object],
    validation: Mapping[str, object],
    input_contract: Path,
) -> None:
    if output_root.exists():
        message = f"Global output already exists: {output_root}"
        raise FileExistsError(message)
    staging = output_root.with_name(f".{output_root.name}.staging")
    if staging.exists():
        message = f"Stale global staging directory exists: {staging}"
        raise FileExistsError(message)
    views = staging / "views"
    views.mkdir(parents=True)
    locations = _location_map(work_paths)
    expected_location_hashes = _mapping_field(validation, "location_file_sha256")
    expected_view_counts = _mapping_field(validation, "logical_view_counts")
    location_artifacts: dict[str, dict[str, object]] = {}
    for name, task_path in sorted(task_paths.items()):
        destination = views / f"{name}_locations.csv"
        count = _write_locations(destination, task_path, locations)
        observed_hash = _sha256(destination)
        if (
            expected_location_hashes.get(name) != observed_hash
            or expected_view_counts.get(f"normal_vae/{name.replace('_', '/')}") != count
            or expected_view_counts.get(f"so2_vae/{name.replace('_', '/')}") != count
        ):
            message = f"Published location identity/count differs for {name}"
            raise ValueError(message)
        location_artifacts[name] = {
            "path": destination.relative_to(staging).as_posix(),
            "row_count": count,
            "bytes": destination.stat().st_size,
            "sha256": observed_hash,
        }
    _fsync_directory(views)
    logical_views = _validate_logical_views(
        views=views,
        work_paths=work_paths,
        normal_shards=normal_shards,
        so2_shards=so2_shards,
        location_artifacts=location_artifacts,
        artifact_records=artifact_records,
    )
    pair_records = _mapping_field(artifact_records, "pair_audits")
    model_records = _mapping_field(artifact_records, "models")
    audit: dict[str, object] = {
        "schema_version": GLOBAL_AUDIT_SCHEMA,
        "status": "complete",
        "runtime_environment": {
            "python_version": platform.python_version(),
            **torch_runtime_versions(),
        },
        "input_contract_sha256": _sha256(input_contract),
        "pair_audits": dict(pair_records),
        "artifacts": dict(model_records),
        "location_files": location_artifacts,
        "logical_views": logical_views,
        "validation": dict(validation),
    }
    _write_json_fsync(
        staging / "spec0021_latent_store_global_audit.json",
        audit,
    )
    _fsync_directory(staging)
    staging.replace(output_root)
    _fsync_directory(output_root.parent)


def _expected_production_bindings(
    *,
    config_root: Path,
    input_receipt_path: Path,
    input_contract_sha256: str,
) -> tuple[dict[int, str], str]:
    receipt = _read_object(input_receipt_path)
    receipt_sha256 = _canonical_hash(receipt)
    config_hashes: dict[int, str] = {}
    for run in range(1, 6):
        config_path = config_root / f"run_{run:02d}" / "spec0021_inference_config.json"
        config = _read_object(config_path)
        embedded_receipt = config.get("input_dataset_receipt")
        if not isinstance(embedded_receipt, dict):
            message = f"Production config {run:02d} lacks its input receipt"
            raise TypeError(message)
        if (
            config.get("schema_version") != "spec0021.inference_config.v2"
            or config.get("mode") != "production"
            or config.get("run_number") != run
            or config.get("input_contract_sha256") != input_contract_sha256
            or _canonical_hash(cast("Mapping[str, object]", embedded_receipt))
            != receipt_sha256
        ):
            message = f"Production config binding differs for run {run:02d}"
            raise ValueError(message)
        config_hashes[run] = _sha256(config_path)
    return config_hashes, receipt_sha256


def _location_map(
    work_paths: Mapping[int, Path],
) -> dict[LatentRowIdentity, tuple[int, int, WorkManifestRow]]:
    locations: dict[LatentRowIdentity, tuple[int, int, WorkManifestRow]] = {}
    for run, path in sorted(work_paths.items()):
        manifest = load_work_manifest(
            path,
            run_number=run,
            expected_sha256=EXPECTED_WORK_MANIFEST_SHA256[run],
        )
        for index, row in enumerate(manifest.rows):
            if row.identity in locations:
                message = "Duplicate union identity while writing locations"
                raise ValueError(message)
            locations[row.identity] = (run, index, row)
    return locations


def _write_locations(
    destination: Path,
    task_path: Path,
    locations: Mapping[LatentRowIdentity, tuple[int, int, WorkManifestRow]],
) -> int:
    temporary = destination.with_suffix(".csv.tmp")
    count = 0
    with (
        task_path.open(encoding="utf-8", newline="") as source,
        temporary.open(
            "x",
            encoding="utf-8",
            newline="",
        ) as output,
    ):
        reader = csv.DictReader(source)
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(LATENT_LOCATION_HEADER)
        for raw in reader:
            identity = LatentRowIdentity(
                int(raw["atlas_row_index"]),
                int(raw["wsi_id"]),
                int(raw["x"]),
                int(raw["y"]),
            )
            try:
                run_number, file_index, work_row = locations[identity]
            except KeyError as error:
                message = f"Task row has no physical latent location: {identity}"
                raise ValueError(message) from error
            tissue_label = raw.get("tissue_label") or ""
            writer.writerow(
                (
                    run_number,
                    file_index,
                    identity.atlas_row_index,
                    identity.wsi_id,
                    identity.x,
                    identity.y,
                    work_row.split,
                    work_row.diagnosis_label,
                    work_row.diagnosis_index,
                    tissue_label,
                ),
            )
            count += 1
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(destination)
    return count


def _validate_logical_views(
    *,
    views: Path,
    work_paths: Mapping[int, Path],
    normal_shards: Mapping[int, Path],
    so2_shards: Mapping[int, Path],
    location_artifacts: Mapping[str, Mapping[str, object]],
    artifact_records: Mapping[str, object],
) -> dict[str, object]:
    model_records = _mapping_field(artifact_records, "models")
    descriptors: dict[str, object] = {}
    for raw_model_name, shards in (
        ("normal_vae", normal_shards),
        ("so2_vae", so2_shards),
    ):
        model_name = cast("ModelName", raw_model_name)
        checkpoint = EXPECTED_CHECKPOINT_SHA256[model_name]
        raw_run_records = _mapping_field(model_records, model_name)
        shard_hashes: dict[str, object] = {}
        for run in range(1, 6):
            run_record = _mapping_field(raw_run_records, str(run))
            binary = _mapping_field(run_record, "binary")
            shard_hashes[str(run)] = binary["sha256"]
        for task_split, location_record in sorted(location_artifacts.items()):
            location_path = views / f"{task_split}_locations.csv"
            view = LatentTaskView(
                model_name=model_name,
                shard_paths=shards,
                work_manifest_paths=work_paths,
                location_path=location_path,
                checkpoint_sha256=checkpoint,
                expected_checkpoint_sha256=checkpoint,
                expected_work_manifest_hashes=EXPECTED_WORK_MANIFEST_SHA256,
                expected_union_sha256=EXPECTED_UNION_SHA256,
            )
            try:
                if len(view) != location_record["row_count"]:
                    message = (
                        f"Logical view count differs for {model_name}/{task_split}"
                    )
                    raise ValueError(message)
            finally:
                view.close()
            view_name = f"{model_name}/{task_split.replace('_', '/')}"
            descriptors[view_name] = {
                "model_name": model_name,
                "task_split": task_split,
                "row_count": location_record["row_count"],
                "location_sha256": location_record["sha256"],
                "shard_sha256": shard_hashes,
            }
    if len(descriptors) != LOGICAL_VIEW_COUNT:
        message = "Global closure requires exactly twelve logical views"
        raise ValueError(message)
    return descriptors


def _mapping_field(value: Mapping[str, object], key: str) -> Mapping[str, object]:
    result = value.get(key)
    if not isinstance(result, dict):
        message = f"Finalizer field {key!r} must be an object"
        raise TypeError(message)
    return cast("Mapping[str, object]", result)


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        message = f"{label} must be a lowercase SHA-256"
        raise ValueError(message)
    return value


def _file_record(path: Path) -> dict[str, object]:
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _read_object(path: Path) -> dict[str, object]:
    payload = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(payload, dict):
        message = f"Expected JSON object in {path}"
        raise TypeError(message)
    return cast("dict[str, object]", payload)


def _write_json_fsync(path: Path, payload: Mapping[str, object]) -> None:
    encoded = f"{json.dumps(payload, indent=2, sort_keys=True)}\n".encode()
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-contract", required=True)
    parser.add_argument("--config-root", required=True)
    parser.add_argument("--input-receipt", required=True)
    parser.add_argument("--pair-root", required=True)
    parser.add_argument("--output-root", required=True)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
