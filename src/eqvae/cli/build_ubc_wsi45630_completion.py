# Copyright 2026 HiperMaximus
# ruff: noqa: DOC201, DOC501, EM101, EM102, FURB118, PLR0914, PLR0916, PLR2004, PLW0717, T201, TC003, TRY003
"""Build the fixed one-WSI 45630 completion input package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import shutil
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, cast

from eqvae.cli.build_ubc_supervised_calibration_inputs import stage_upload_envelope
from eqvae.cli.generate_ubc_cancer_topup import (
    _validate_input_tree,  # pyright: ignore[reportPrivateUsage]
)
from eqvae.data.latent_shards import EXPECTED_CHECKPOINT_SHA256, EXPECTED_UNION_SHA256

WSI_ID: Final = 45_630
MISSING_ROWS: Final = 22_649
BASE_REUSE_ROWS: Final = 5_136
TOPUP_REUSE_ROWS: Final = 4_810
TARGET_ROWS: Final = 32_595
DATASET_REFERENCE: Final = "maximusshtefan/eqvae-wsi45630-completion-inputs"
SPEC_PATH: Final = Path("docs/specs/0022-additive-cancer-coverage-topup.md")
CANDIDATE_PATH: Final = Path(
    "runs/local/ubc_ocean_eval_manifests/cancer_ae_patch_manifest.csv",
)
UNION_PATH: Final = Path(
    "runs/local/ubc_ocean_eval_consumption/union_patch_manifest.csv",
)
OLD_ROOT: Final = Path("runs/local/ubc_ocean_cancer_topup")
OLD_MANIFEST_PATH: Final = OLD_ROOT / "inference_bundle/cancer_topup_manifest.csv"
OLD_CONTRACT_PATH: Final = (
    OLD_ROOT / "inference_bundle/spec0022_topup_inference_contract.json"
)
LOGICAL_AUDIT_PATH: Final = OLD_ROOT / "spec0022_logical_dataset_audit.json"
PAIR_AUDIT_PATH: Final = Path(
    "runs/kaggle/ubc_ocean_cancer_topup_metadata/dataset/"
    "spec0022_cancer_topup_pair_audit.json",
)
NORMAL_CHECKPOINT_PATH: Final = OLD_ROOT / "inference_bundle/normal_vae_step_060000.pt"
SO2_CHECKPOINT_PATH: Final = OLD_ROOT / "inference_bundle/so2_vae_step_060000.pt"
OUTPUT_ROOT: Final = Path("runs/local/wsi45630_completion")

CANDIDATE_SHA256: Final = (
    "710c3f8166f577f5ae60bec94a544dabed9619342a46b82374a2e739162d648c"
)
OLD_MANIFEST_SHA256: Final = (
    "ed9b2d7a7c8d51cf37710867cb5512677dfca97e8245d560324ec6d8461d6308"
)
OLD_CONTRACT_SHA256: Final = (
    "17ddedd331a3c592ca0a43693e1209ea16ffb9953985420fcf6a84c9117f90d2"
)
LOGICAL_AUDIT_SHA256: Final = (
    "3655629db4af520a37a0e0c7992813c091ef7d2efc16328f9ad869c9e07cde52"
)
PAIR_AUDIT_SHA256: Final = (
    "17c0b577d9e30bf94a48724a9d93e9b3be16e12be806cfc31e7b08d89e02201d"
)
INFERENCE_FILES: Final = frozenset({
    "cancer_topup_manifest.csv",
    "normal_vae_step_060000.pt",
    "so2_vae_step_060000.pt",
    "spec0022_topup_inference_contract.json",
})
INPUT_CONTRACT_NAME: Final = "wsi45630_input.json"
METADATA_NAME: Final = "dataset-metadata.json"
MANIFEST_HEADER: Final = ("atlas_row_index", "wsi_id", "x", "y")
LATENT_HEADER_BYTES: Final = 64
LATENT_RECORD_BYTES: Final = 65_536

Identity = tuple[int, int, int, int]


def build(*, repo_root: Path, output_root: Path) -> dict[str, object]:
    """Stage one immutable missing-coordinate package without contacting Kaggle."""
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite {output_root}")
    sources = _sources(repo_root)
    source_hashes = {
        name: _require_hash(path, expected) for name, path, expected in sources
    }
    candidates = _load_candidates_train_only(repo_root / CANDIDATE_PATH)
    union = _load_filtered(repo_root / UNION_PATH, None)
    old = _load_filtered(repo_root / OLD_MANIFEST_PATH, MANIFEST_HEADER)
    _validate_prior_authorities(repo_root)
    missing, base_reuse, topup_reuse = derive_missing(
        candidates=candidates,
        union=union,
        old=old,
    )
    if (len(candidates), len(missing), base_reuse, topup_reuse) != (
        TARGET_ROWS,
        MISSING_ROWS,
        BASE_REUSE_ROWS,
        TOPUP_REUSE_ROWS,
    ):
        raise ValueError("WSI 45630 completion counts differ from the sealed extension")
    staging = output_root.with_name(f".{output_root.name}.building")
    if staging.exists():
        raise FileExistsError(f"Stale completion staging directory exists: {staging}")
    bundle = staging / "bundle"
    try:
        inference = bundle / "inference"
        inference.mkdir(parents=True)
        _write_manifest(inference / "cancer_topup_manifest.csv", missing)
        shutil.copy2(
            repo_root / NORMAL_CHECKPOINT_PATH,
            inference / "normal_vae_step_060000.pt",
        )
        shutil.copy2(
            repo_root / SO2_CHECKPOINT_PATH,
            inference / "so2_vae_step_060000.pt",
        )
        worker_contract = _completion_worker_contract(
            old_contract=_read_object(repo_root / OLD_CONTRACT_PATH),
            manifest=inference / "cancer_topup_manifest.csv",
        )
        _write_canonical_json(
            inference / "spec0022_topup_inference_contract.json",
            worker_contract,
        )
        _copy_source(repo_root, bundle / "src")
        files = _artifact_records(bundle, exclude={INPUT_CONTRACT_NAME, METADATA_NAME})
        contract: dict[str, object] = {
            "schema_version": "spec0022.wsi45630_completion_input.v1",
            "dataset_reference": DATASET_REFERENCE,
            "spec_sha256": _sha256(repo_root / SPEC_PATH),
            "wsi_id": WSI_ID,
            "target_rows": TARGET_ROWS,
            "missing_rows": MISSING_ROWS,
            "files": files,
        }
        _write_canonical_json(bundle / INPUT_CONTRACT_NAME, contract)
        _write_canonical_json(bundle / METADATA_NAME, _dataset_metadata())
        audit = {
            "schema_version": "spec0022.wsi45630_completion_audit.v1",
            "status": "complete",
            "inputs": source_hashes,
            "counts": {
                "target_rows": TARGET_ROWS,
                "base_reuse_rows": base_reuse,
                "topup_reuse_rows": topup_reuse,
                "missing_rows": len(missing),
            },
            "input_contract_sha256": _sha256(bundle / INPUT_CONTRACT_NAME),
            "worker_contract_sha256": _sha256(
                inference / "spec0022_topup_inference_contract.json",
            ),
            "bundle_files": files,
        }
        _write_canonical_json(staging / "audit.json", audit)
        staging.replace(output_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    upload = stage_upload_envelope(
        bundle_root=output_root / "bundle",
        destination=output_root / "upload",
    )
    audit = _read_object(output_root / "audit.json")
    audit["upload"] = _artifact_records(upload, exclude=set())
    _write_canonical_json(output_root / "audit.json", audit)
    return validate(repo_root=repo_root, output_root=output_root)


def validate(*, repo_root: Path, output_root: Path) -> dict[str, object]:
    """Reject changed package bytes, source snapshots, or bounded completion claims."""
    bundle = output_root / "bundle"
    contract = _read_object(bundle / INPUT_CONTRACT_NAME)
    if (
        contract.get("schema_version") != "spec0022.wsi45630_completion_input.v1"
        or contract.get("dataset_reference") != DATASET_REFERENCE
        or contract.get("spec_sha256") != _sha256(repo_root / SPEC_PATH)
        or contract.get("wsi_id") != WSI_ID
        or contract.get("target_rows") != TARGET_ROWS
        or contract.get("missing_rows") != MISSING_ROWS
        or _read_object(bundle / METADATA_NAME) != _dataset_metadata()
    ):
        raise ValueError("WSI 45630 input contract identity differs")
    files = _records(contract.get("files"))
    observed = _artifact_records(bundle, exclude={INPUT_CONTRACT_NAME, METADATA_NAME})
    if files != observed:
        raise ValueError("WSI 45630 input bundle files differ")
    source_files = _source_files(repo_root)
    expected_source_names = {f"src/{name}" for name in source_files}
    if {name for name in files if name.startswith("src/")} != expected_source_names:
        raise ValueError("WSI 45630 source snapshot allow-list differs")
    for name, source in source_files.items():
        if (bundle / f"src/{name}").read_bytes() != source.read_bytes():
            raise ValueError(f"WSI 45630 source snapshot differs: {name}")
    inference_names = {
        name.removeprefix("inference/")
        for name in files
        if name.startswith("inference/")
    }
    if inference_names != set(INFERENCE_FILES):
        raise ValueError("WSI 45630 inference allow-list differs")
    _validate_input_tree(
        bundle / "inference",
        bundle / "inference/spec0022_topup_inference_contract.json",
    )
    worker = _read_object(bundle / "inference/spec0022_topup_inference_contract.json")
    supplement = cast("dict[str, object]", worker["supplement_manifest"])
    deadline = cast("dict[str, object]", worker["deadline_projection"])
    if (
        supplement.get("path") != "cancer_topup_manifest.csv"
        or supplement.get("row_count") != MISSING_ROWS
        or supplement.get("sha256")
        != _sha256(bundle / "inference/cancer_topup_manifest.csv")
        or worker.get("expected_binary_output_bytes")
        != 2 * (LATENT_HEADER_BYTES + MISSING_ROWS * LATENT_RECORD_BYTES)
        or deadline.get("worst_observed_seconds_per_wsi") != 1_960
        or deadline.get("projected_inference_seconds") != 1_960
        or deadline.get("projected_total_seconds") != 5_560
    ):
        raise ValueError("WSI 45630 worker contract differs")
    audit = _read_object(output_root / "audit.json")
    if (
        audit.get("status") != "complete"
        or audit.get("bundle_files") != files
        or audit.get("input_contract_sha256") != _sha256(bundle / INPUT_CONTRACT_NAME)
        or audit.get("upload")
        != _artifact_records(output_root / "upload", exclude=set())
    ):
        raise ValueError("WSI 45630 audit differs")
    _validate_upload(output_root / "upload", bundle)
    return contract


def derive_missing(
    *,
    candidates: Mapping[Identity, object],
    union: Mapping[Identity, object],
    old: Mapping[Identity, object],
) -> tuple[tuple[Identity, ...], int, int]:
    """Subtract all already-stored identities after enforcing coordinate identity."""
    candidate_coords = _coordinate_index(candidates, "candidate")
    union_coords = _coordinate_index(union, "union")
    old_coords = _coordinate_index(old, "old top-up")
    for name, rows, _coords in (
        ("union", union, union_coords),
        ("old top-up", old, old_coords),
    ):
        for identity in rows:
            candidate_identity = candidate_coords.get(identity[1:])
            if candidate_identity is not None and candidate_identity != identity:
                raise ValueError(f"{name} identity drift for coordinate {identity[1:]}")
    candidate_ids = set(candidates)
    union_ids = candidate_ids & set(union)
    old_ids = candidate_ids & set(old)
    if union_ids & old_ids:
        raise ValueError("Base union and old top-up overlap for WSI 45630")
    missing = tuple(
        sorted(candidate_ids - union_ids - old_ids, key=lambda row: (row[3], row[2])),
    )
    return missing, len(union_ids), len(old_ids)


def _sources(repo_root: Path) -> tuple[tuple[str, Path, str], ...]:
    return (
        ("candidate_manifest", repo_root / CANDIDATE_PATH, CANDIDATE_SHA256),
        ("base_union", repo_root / UNION_PATH, EXPECTED_UNION_SHA256),
        (
            "completed_part11_manifest",
            repo_root / OLD_MANIFEST_PATH,
            OLD_MANIFEST_SHA256,
        ),
        (
            "completed_input_contract",
            repo_root / OLD_CONTRACT_PATH,
            OLD_CONTRACT_SHA256,
        ),
        (
            "completed_logical_audit",
            repo_root / LOGICAL_AUDIT_PATH,
            LOGICAL_AUDIT_SHA256,
        ),
        ("completed_pair_audit", repo_root / PAIR_AUDIT_PATH, PAIR_AUDIT_SHA256),
        (
            "normal_checkpoint",
            repo_root / NORMAL_CHECKPOINT_PATH,
            EXPECTED_CHECKPOINT_SHA256["normal_vae"],
        ),
        (
            "so2_checkpoint",
            repo_root / SO2_CHECKPOINT_PATH,
            EXPECTED_CHECKPOINT_SHA256["so2_vae"],
        ),
    )


def _validate_prior_authorities(repo_root: Path) -> None:
    logical = _read_object(repo_root / LOGICAL_AUDIT_PATH)
    pair = _read_object(repo_root / PAIR_AUDIT_PATH)
    if (
        logical.get("status") != "complete"
        or pair.get("status") != "complete"
        or pair.get("row_count") != 74_033
        or pair.get("supplement_manifest_sha256") != OLD_MANIFEST_SHA256
        or pair.get("input_contract_sha256") != OLD_CONTRACT_SHA256
    ):
        raise ValueError("Completed Spec 0022 authorities differ")


def _load_filtered(
    path: Path,
    expected_header: tuple[str, ...] | None,
) -> dict[Identity, object]:
    """Filter textual numeric WSI IDs before CSV parsing any candidate rows."""
    with path.open(encoding="utf-8", newline="") as handle:
        header = handle.readline().rstrip("\r\n")
        fields = tuple(header.split(","))
        if expected_header is not None and fields != expected_header:
            raise ValueError(f"Unexpected header in {path}")
        if len(fields) < 4 or fields[0] != "atlas_row_index" or fields[1] != "wsi_id":
            raise ValueError(f"Missing coordinate columns in {path}")
        kept = [header]
        for raw in handle:
            fields = raw.rstrip("\r\n").split(",")
            if len(fields) > 1 and fields[1].isdigit() and int(fields[1]) == WSI_ID:
                kept.append(raw.rstrip("\r\n"))
    reader = csv.DictReader(io.StringIO("\n".join(kept)))
    result: dict[Identity, object] = {}
    coordinates: set[tuple[int, int, int]] = set()
    previous: tuple[int, int] | None = None
    for raw in reader:
        identity = (
            int(raw["atlas_row_index"]),
            int(raw["wsi_id"]),
            int(raw["x"]),
            int(raw["y"]),
        )
        coordinate = identity[1:]
        if identity[1] != WSI_ID or identity in result or coordinate in coordinates:
            raise ValueError(f"Duplicate or foreign WSI identity in {path}")
        order = (identity[3], identity[2])
        if previous is not None and order <= previous:
            raise ValueError(f"WSI 45630 rows are not ordered y,x in {path}")
        previous = order
        result[identity] = None
        coordinates.add(coordinate)
    return result


def _load_candidates_train_only(path: Path) -> dict[Identity, object]:
    rows = _load_filtered(path, None)
    with path.open(encoding="utf-8", newline="") as handle:
        header = tuple(handle.readline().rstrip("\r\n").split(","))
        if header != (
            "atlas_row_index",
            "wsi_id",
            "diagnosis_label",
            "diagnosis_index",
            "x",
            "y",
            "split",
        ):
            raise ValueError("Unexpected candidate header")
        for raw in handle:
            fields = raw.rstrip("\r\n").split(",")
            if (
                len(fields) > 6
                and fields[1].isdigit()
                and int(fields[1]) == WSI_ID
                and fields[6] != "train"
            ):
                raise ValueError("WSI 45630 candidate must be train")
    return rows


def _coordinate_index(
    rows: Mapping[Identity, object],
    name: str,
) -> dict[tuple[int, int, int], Identity]:
    result: dict[tuple[int, int, int], Identity] = {}
    for identity in rows:
        coordinate = identity[1:]
        if coordinate in result:
            raise ValueError(f"Duplicate {name} coordinate {coordinate}")
        result[coordinate] = identity
    return result


def _completion_worker_contract(
    *,
    old_contract: dict[str, object],
    manifest: Path,
) -> dict[str, object]:
    contract = dict(old_contract)
    manifest_hash = _sha256(manifest)
    contract["supplement_manifest"] = {
        "path": "cancer_topup_manifest.csv",
        "sha256": manifest_hash,
        "row_count": MISSING_ROWS,
        "header": list(MANIFEST_HEADER),
    }
    contract["expected_binary_output_bytes"] = 2 * (
        LATENT_HEADER_BYTES + MISSING_ROWS * LATENT_RECORD_BYTES
    )
    deadline = dict(cast("dict[str, object]", contract["deadline_projection"]))
    source_runs = cast("list[dict[str, object]]", deadline["source_runs"])
    row_projection = math.ceil(
        max(
            MISSING_ROWS
            * cast("float", run["elapsed_seconds"])
            / cast("int", run["row_count"])
            for run in source_runs
        ),
    )
    wsi_projection = math.ceil(
        max(
            cast("float", run["elapsed_seconds"]) / cast("int", run["wsi_count"])
            for run in source_runs
        ),
    )
    projected_inference = max(row_projection, wsi_projection)
    if projected_inference != 1_960:
        raise ValueError("Sealed timing evidence no longer projects 1,960 seconds")
    deadline.update({
        "supplement_rows": MISSING_ROWS,
        "supplement_wsi_count": 1,
        "row_projection_seconds": row_projection,
        "wsi_projection_seconds": wsi_projection,
        "worst_observed_seconds_per_wsi": projected_inference,
        "projected_inference_seconds": projected_inference,
        "projected_total_seconds": projected_inference + 3_600,
        "fits": True,
    })
    contract["deadline_projection"] = deadline
    return contract


def _copy_source(repo_root: Path, destination: Path) -> None:
    for name, source in _source_files(repo_root).items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _source_files(repo_root: Path) -> dict[str, Path]:
    return {
        path.relative_to(repo_root / "src").as_posix(): path
        for path in sorted((repo_root / "src/eqvae").rglob("*.py"))
        if "__pycache__" not in path.parts
    }


def _write_manifest(path: Path, rows: Sequence[Identity]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(MANIFEST_HEADER)
        writer.writerows(rows)


def _artifact_records(root: Path, *, exclude: set[str]) -> dict[str, dict[str, object]]:
    return {
        path.relative_to(root).as_posix(): {
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.relative_to(root).as_posix() not in exclude
    }


def _dataset_metadata() -> dict[str, object]:
    return {
        "id": DATASET_REFERENCE,
        "title": "eqvae wsi45630 completion inputs",
        "licenses": [{"name": "other"}],
    }


def _validate_upload(upload: Path, bundle: Path) -> None:
    if {path.name for path in upload.iterdir()} != {"bundle.zip", METADATA_NAME}:
        raise ValueError("WSI 45630 upload envelope differs")
    if _read_object(upload / METADATA_NAME) != _dataset_metadata():
        raise ValueError("WSI 45630 upload metadata differs")
    expected = {
        path.relative_to(bundle).as_posix()
        for path in bundle.rglob("*")
        if path.is_file() and path.name != METADATA_NAME
    }
    with zipfile.ZipFile(upload / "bundle.zip") as archive:
        if set(archive.namelist()) != expected:
            raise ValueError("WSI 45630 upload archive allow-list differs")
        for name in expected:
            if archive.read(name) != (bundle / name).read_bytes():
                raise ValueError(f"WSI 45630 upload archive differs: {name}")


def _records(value: object) -> dict[str, dict[str, object]]:
    if not isinstance(value, dict):
        raise TypeError("WSI 45630 input files must be records")
    records = cast("dict[str, object]", value)
    if not all(isinstance(record, dict) for record in records.values()):
        raise TypeError("WSI 45630 input files must be records")
    return cast("dict[str, dict[str, object]]", value)


def _read_object(path: Path) -> dict[str, object]:
    value = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain an object")
    return cast("dict[str, object]", value)


def _require_hash(path: Path, expected: str) -> str:
    observed = _sha256(path)
    if observed != expected:
        raise ValueError(f"Frozen input hash differs: {path}")
    return observed


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_canonical_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Build or validate the fixed local WSI 45630 completion package."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("build", "validate"),
        nargs="?",
        default="build",
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args(argv)
    repo_root = cast("Path", args.repo_root).resolve()
    output_root = cast("Path", args.output_root)
    action = cast("str", args.action)
    result = (
        validate(repo_root=repo_root, output_root=output_root)
        if action == "validate"
        else build(repo_root=repo_root, output_root=output_root)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
