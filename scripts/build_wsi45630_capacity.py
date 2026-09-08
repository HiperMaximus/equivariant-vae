# Copyright 2026 HiperMaximus
# ruff: noqa: DOC201, DOC501, EM101, EM102, PLC2701, PLR0916, PLR2004, PLW0717, T201, TC003, TRY003
"""Build the private full-coverage WSI45630 transformer-capacity input package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import zipfile
from collections.abc import Mapping, Sequence
from operator import itemgetter
from pathlib import Path
from string import Template
from typing import Final, cast

from eqvae.cli.build_ubc_supervised_calibration_inputs import stage_upload_envelope
from eqvae.cli.build_ubc_wsi45630_completion import (
    CANDIDATE_PATH,
    CANDIDATE_SHA256,
    _load_candidates_train_only,  # pyright: ignore[reportPrivateUsage]
)
from eqvae.data.latent_shards import EXPECTED_WORK_MANIFEST_SHA256, UNION_HEADER
from eqvae.data.supervised_latents import (
    CATALOG_HEADER,
)

ROOT = Path.cwd()
DEFAULT_ROOT: Final = Path("runs/local/wsi45630_capacity")
WSI_ID: Final = 45_630
PATCH_COUNT: Final = 32_595
DIAGNOSIS_INDEX: Final = 1
PART_COUNTS: Final = {"4": 5_136, "11": 4_810, "12": 22_649}
DATASET_REFERENCE: Final = "maximusshtefan/eqvae-wsi45630-capacity-inputs"
KERNEL_ID: Final = "maximusshtefan/eqvae-wsi45630-transformer-capacity"
KERNEL_SOURCES: Final = (
    "maximusshtefan/eqvae-ubc-ocean-latent-run-04",
    "maximusshtefan/eqvae-ubc-ocean-cancer-latent-top-up",
    "maximusshtefan/eqvae-wsi45630-completion",
)
SPEC_PATH: Final = Path("docs/specs/0023-matched-supervised-latent-evaluation.md")
TEMPLATE_PATH: Final = Path("kaggle/kernels/wsi45630_capacity/run_template.py")
CATALOG_PATH: Final = Path(
    "runs/local/ubc_ocean_supervised_manifests/physical_parts.csv",
)
PART4_PATH: Final = Path(
    "runs/local/ubc_ocean_eval_consumption/work_shards/run_04_of_05.csv",
)
PART11_PATH: Final = Path(
    "runs/local/ubc_ocean_cancer_topup/inference_bundle/cancer_topup_manifest.csv",
)
COMPLETION_PATH: Final = Path(
    "runs/local/wsi45630_completion/bundle/inference/cancer_topup_manifest.csv",
)
COMPLETION_AUDIT_PATH: Final = Path(
    "runs/kaggle/wsi45630_completion_v1/dataset/spec0022_cancer_topup_pair_audit.json",
)
MANIFEST_AUDIT_PATH: Final = Path(
    "runs/local/ubc_ocean_supervised_manifests/spec0023_supervised_manifest_audit.json",
)
CONTRACT_NAME: Final = "wsi45630_capacity_input.json"
METADATA_NAME: Final = "dataset-metadata.json"
COMPLETION_MANIFEST_SHA256: Final = (
    "df4d5281c585bd39826a6d528eba2b8b1ddd15444e173c0c59cf3b6750202adc"
)
COMPLETION_AUDIT_SHA256: Final = (
    "267a8b8128e1a637352223ee3e5939edd9c47b29e7e8617fdbedcb1aae1ea20b"
)
TRANSFORMER_PROBE_PATH: Final = Path(
    "kaggle/kernels/ubc_ocean_mil_transformer_capacity/run.py",
)
TRANSFORMER_PROBE_SHA256: Final = (
    "f7cc5c0eea786efd030c4f80a87ab29b706d1a3195df7e0b3509d2d7e0f4e67b"
)
MANIFEST_AUDIT_SHA256: Final = (
    "805d0dc94b38b0a12a67779df30a17a6b531b99f9db6279f8b73f433d9d6de9e"
)

Identity = tuple[int, int, int, int]
POINTER_HEADER: Final = ("atlas_row_index", "wsi_id", "x", "y", "part", "file_index")


def build() -> dict[str, object]:
    """Stage one immutable capacity-only package below the repository root."""
    output = ROOT / DEFAULT_ROOT
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    assets, provenance = _derive_assets(ROOT)
    staging = output.with_name(f".{output.name}.building")
    if staging.exists():
        raise FileExistsError(f"Stale staging directory exists: {staging}")
    bundle = staging / "bundle"
    try:
        for name, payload in assets.items():
            path = bundle / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        _copy_source(ROOT, bundle / "src")
        files = _artifact_records(bundle, exclude={CONTRACT_NAME, METADATA_NAME})
        contract: dict[str, object] = {
            "schema_version": "spec0023.wsi45630_capacity_input.v1",
            "dataset_reference": DATASET_REFERENCE,
            "spec_sha256": _sha256(ROOT / SPEC_PATH),
            "wsi_id": WSI_ID,
            "patch_count": PATCH_COUNT,
            "diagnosis_index": DIAGNOSIS_INDEX,
            "part_counts": PART_COUNTS,
            "kernel_sources": list(KERNEL_SOURCES),
            "source_provenance": provenance,
            "files": files,
        }
        _write_json(bundle / CONTRACT_NAME, contract)
        _write_json(bundle / METADATA_NAME, _dataset_metadata())
        kernel = staging / "kernel"
        kernel.mkdir()
        _write_json(kernel / "kernel-metadata.json", _kernel_metadata())
        (kernel / "run.py").write_bytes(_render(ROOT, _sha256(bundle / CONTRACT_NAME)))
        staging.replace(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    stage_upload_envelope(bundle_root=output / "bundle", destination=output / "upload")
    return validate()


def validate() -> dict[str, object]:
    """Reject hash, topology, metadata, source, or rendered-launcher drift."""
    output = ROOT / DEFAULT_ROOT
    bundle = output / "bundle"
    contract = _read_object(bundle / CONTRACT_NAME)
    if (
        contract.get("schema_version") != "spec0023.wsi45630_capacity_input.v1"
        or contract.get("dataset_reference") != DATASET_REFERENCE
        or contract.get("spec_sha256") != _sha256(ROOT / SPEC_PATH)
        or contract.get("wsi_id") != WSI_ID
        or contract.get("patch_count") != PATCH_COUNT
        or contract.get("diagnosis_index") != DIAGNOSIS_INDEX
        or contract.get("part_counts") != PART_COUNTS
        or contract.get("kernel_sources") != list(KERNEL_SOURCES)
        or _read_object(bundle / METADATA_NAME) != _dataset_metadata()
    ):
        raise ValueError("WSI45630 capacity contract identity differs")
    files = _records(contract.get("files"))
    if files != _artifact_records(bundle, exclude={CONTRACT_NAME, METADATA_NAME}):
        raise ValueError("WSI45630 capacity bundle bytes differ")
    _validate_source_snapshot(bundle, files)
    assets, provenance = _derive_assets(ROOT)
    if contract.get("source_provenance") != provenance:
        raise ValueError("WSI45630 capacity authorities differ")
    for name, payload in assets.items():
        if (bundle / name).read_bytes() != payload:
            raise ValueError(f"WSI45630 capacity probe asset differs: {name}")
    kernel = output / "kernel"
    if {path.name for path in kernel.iterdir()} != {"kernel-metadata.json", "run.py"}:
        raise ValueError("WSI45630 capacity kernel allow-list differs")
    if _read_object(kernel / "kernel-metadata.json") != _kernel_metadata():
        raise ValueError("WSI45630 capacity kernel metadata differs")
    if (kernel / "run.py").read_bytes() != _render(
        ROOT,
        _sha256(bundle / CONTRACT_NAME),
    ):
        raise ValueError("WSI45630 capacity rendered launcher differs")
    _validate_upload(output / "upload", bundle)
    return contract


def _derive_assets(root: Path) -> tuple[dict[str, bytes], dict[str, str]]:
    """Derive only the frozen train WSI pointer topology from sealed text inputs."""
    provenance = _authenticate_authorities(root)
    candidates = _load_candidates_train_only(root / CANDIDATE_PATH)
    raw_parts = {
        4: _physical_rows(root / PART4_PATH, expected_header=UNION_HEADER),
        11: _physical_rows(
            root / PART11_PATH,
            expected_header=("atlas_row_index", "wsi_id", "x", "y"),
        ),
        12: _physical_rows(
            root / COMPLETION_PATH,
            expected_header=("atlas_row_index", "wsi_id", "x", "y"),
        ),
    }
    parts = {
        part: {
            identity: file_index
            for identity, file_index in rows.items()
            if identity in candidates
        }
        for part, rows in raw_parts.items()
    }
    if {part: len(rows) for part, rows in parts.items()} != {
        int(k): v for k, v in PART_COUNTS.items()
    }:
        raise ValueError("WSI45630 part counts differ from the locked capacity bag")
    pointers, ordered = _bind_pointers(candidates, parts)
    if len(ordered) != PATCH_COUNT:
        raise ValueError("WSI45630 capacity bag size differs")
    pointers_csv: list[dict[str, str]] = []
    for identity in ordered:
        part, file_index = pointers[identity]
        pointers_csv.append({
            "atlas_row_index": str(identity[0]),
            "wsi_id": str(WSI_ID),
            "x": str(identity[2]),
            "y": str(identity[3]),
            "part": str(part),
            "file_index": str(file_index),
        })
    catalog = _probe_catalog(root)
    assets = {
        "probe/physical_parts.csv": _csv_bytes(CATALOG_HEADER, catalog),
        "probe/pointers.csv": _csv_bytes(POINTER_HEADER, pointers_csv),
        "transformer_probe.py": (root / TRANSFORMER_PROBE_PATH).read_bytes(),
    }
    return assets, provenance


def _bind_pointers(
    candidates: Mapping[Identity, object],
    parts: Mapping[int, Mapping[Identity, int]],
) -> tuple[dict[Identity, tuple[int, int]], list[Identity]]:
    """Bind all disjoint physical sources to one complete y,x logical bag."""
    pointers: dict[Identity, tuple[int, int]] = {}
    for part, rows in parts.items():
        for identity, file_index in rows.items():
            if identity in pointers:
                raise ValueError("WSI45630 source parts overlap")
            pointers[identity] = (part, file_index)
    if set(pointers) != set(candidates):
        raise ValueError("WSI45630 sources do not exactly cover frozen candidates")
    return pointers, sorted(pointers, key=itemgetter(3, 2))


def _authenticate_authorities(root: Path) -> dict[str, str]:
    """Fail closed on the sealed inputs that bind all three physical sources."""
    sources = {
        "candidate_manifest": (root / CANDIDATE_PATH, CANDIDATE_SHA256),
        "part4_work_manifest": (root / PART4_PATH, EXPECTED_WORK_MANIFEST_SHA256[4]),
        "part11_manifest": (
            root / PART11_PATH,
            "ed9b2d7a7c8d51cf37710867cb5512677dfca97e8245d560324ec6d8461d6308",
        ),
        "completion_manifest": (root / COMPLETION_PATH, COMPLETION_MANIFEST_SHA256),
        "completion_pair_audit": (
            root / COMPLETION_AUDIT_PATH,
            COMPLETION_AUDIT_SHA256,
        ),
        "transformer_probe": (
            root / TRANSFORMER_PROBE_PATH,
            TRANSFORMER_PROBE_SHA256,
        ),
        "supervised_manifest_audit": (
            root / MANIFEST_AUDIT_PATH,
            MANIFEST_AUDIT_SHA256,
        ),
    }
    observed = {
        name: _require_hash(path, expected)
        for name, (path, expected) in sources.items()
    }
    audit = _read_object(root / COMPLETION_AUDIT_PATH)
    artifacts = cast("dict[str, object]", audit.get("artifacts"))
    if (
        audit.get("status") != "complete"
        or audit.get("row_count") != PART_COUNTS["12"]
        or audit.get("supplement_manifest_sha256") != COMPLETION_MANIFEST_SHA256
        or not isinstance(artifacts.get("normal_vae"), dict)
        or not isinstance(artifacts.get("so2_vae"), dict)
    ):
        raise ValueError("WSI45630 completion audit differs")
    manifest_audit = _read_object(root / MANIFEST_AUDIT_PATH)
    physical_catalog = cast("dict[str, object]", manifest_audit.get("physical_catalog"))
    if (
        manifest_audit.get("status") != "complete"
        or physical_catalog.get("path") != "physical_parts.csv"
        or physical_catalog.get("sha256") != _sha256(root / CATALOG_PATH)
        or physical_catalog.get("row_count") != 12
    ):
        raise ValueError("Supervised physical catalog authority differs")
    return observed


def _physical_rows(
    path: Path,
    *,
    expected_header: Sequence[str],
) -> dict[Identity, int]:
    """Enumerate every physical record before parsing the target WSI's rows."""
    with path.open(encoding="utf-8", newline="") as handle:
        header = handle.readline().rstrip("\r\n")
        if tuple(header.split(",")) != tuple(expected_header):
            raise ValueError(f"Unexpected header in {path}")
        kept: list[tuple[int, str]] = []
        for file_index, raw in enumerate(handle):
            fields = raw.rstrip("\r\n").split(",")
            if len(fields) > 1 and fields[1].isdigit() and int(fields[1]) == WSI_ID:
                kept.append((file_index, raw.rstrip("\r\n")))
    rows: dict[Identity, int] = {}
    for file_index, raw in kept:
        row = next(csv.DictReader(io.StringIO(f"{header}\n{raw}\n")))
        identity = _identity(row)
        if identity[1] != WSI_ID or identity in rows:
            raise ValueError(f"Duplicate or foreign WSI45630 source row in {path}")
        rows[identity] = file_index
    return rows


def _probe_catalog(root: Path) -> list[dict[str, str]]:
    """Keep just the three authenticated physical source pairs for this probe."""
    catalog = _read_csv(root / CATALOG_PATH, CATALOG_HEADER)
    selected = [row for row in catalog if row["part"] in {"4", "11"}]
    audit = _read_object(root / COMPLETION_AUDIT_PATH)
    artifacts = cast("dict[str, dict[str, object]]", audit["artifacts"])
    for model_name in ("normal_vae", "so2_vae"):
        artifact = artifacts[model_name]
        selected.append({
            "part": "12",
            "model_name": model_name,
            "kaggle_source": KERNEL_SOURCES[2],
            "binary_name": cast("str", artifact["bin_name"]),
            "row_count": str(PART_COUNTS["12"]),
            "binary_bytes": str(artifact["bin_bytes"]),
            "binary_sha256": cast("str", artifact["bin_sha256"]),
            "sidecar_name": cast("str", artifact["sidecar_name"]),
            "sidecar_bytes": str(artifact["sidecar_bytes"]),
            "sidecar_sha256": cast("str", artifact["sidecar_sha256"]),
        })
    by_part_model = {(row["part"], row["model_name"]): row for row in selected}
    if set(by_part_model) != {
        (part, model)
        for part in ("4", "11", "12")
        for model in ("normal_vae", "so2_vae")
    }:
        raise ValueError("Capacity catalog model topology differs")
    for model_name in ("normal_vae", "so2_vae"):
        part11 = by_part_model["11", model_name]
        part12 = by_part_model["12", model_name]
        if (
            part11["binary_name"] != part12["binary_name"]
            or part11["sidecar_sha256"] == part12["sidecar_sha256"]
        ):
            raise ValueError("Part11/12 sidecar collision is unauthenticated")
    ordered = sorted(selected, key=lambda row: (int(row["part"]), row["model_name"]))
    if tuple(dict.fromkeys(row["kaggle_source"] for row in ordered)) != KERNEL_SOURCES:
        raise ValueError("Capacity catalog sources differ")
    return ordered


def _copy_source(root: Path, destination: Path) -> None:
    for source in sorted((root / "src/eqvae").rglob("*.py")):
        if "__pycache__" not in source.parts:
            target = destination / source.relative_to(root / "src")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def _validate_source_snapshot(bundle: Path, files: Mapping[str, object]) -> None:
    expected = {
        f"src/{path.relative_to(ROOT / 'src').as_posix()}": path
        for path in sorted((ROOT / "src/eqvae").rglob("*.py"))
        if "__pycache__" not in path.parts
    }
    if {name for name in files if name.startswith("src/")} != set(expected):
        raise ValueError("WSI45630 capacity source allow-list differs")
    for name, source in expected.items():
        if (bundle / name).read_bytes() != source.read_bytes():
            raise ValueError(f"WSI45630 capacity source differs: {name}")


def _render(root: Path, contract_hash: str) -> bytes:
    template = (root / TEMPLATE_PATH).read_text(encoding="utf-8")
    if template.count("$input_contract_sha256") != 1:
        raise ValueError(
            "WSI45630 capacity template must have one contract placeholder",
        )
    rendered = (
        Template(template).substitute(input_contract_sha256=contract_hash).encode()
    )
    if len(rendered) >= 1_000_000:
        raise ValueError("WSI45630 capacity launcher exceeds Kaggle's upload limit")
    compile(rendered, str(root / TEMPLATE_PATH), "exec")
    return rendered


def _dataset_metadata() -> dict[str, object]:
    return {
        "id": DATASET_REFERENCE,
        "title": "eqvae wsi45630 capacity inputs",
        "licenses": [{"name": "other"}],
    }


def _kernel_metadata() -> dict[str, object]:
    return {
        "id": KERNEL_ID,
        "title": "eqvae wsi45630 transformer capacity",
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [DATASET_REFERENCE],
        "competition_sources": [],
        "kernel_sources": list(KERNEL_SOURCES),
        "model_sources": [],
    }


def _validate_upload(upload: Path, bundle: Path) -> None:
    if {path.name for path in upload.iterdir()} != {"bundle.zip", METADATA_NAME}:
        raise ValueError("WSI45630 capacity upload envelope differs")
    if _read_object(upload / METADATA_NAME) != _dataset_metadata():
        raise ValueError("WSI45630 capacity upload metadata differs")
    with zipfile.ZipFile(upload / "bundle.zip") as archive:
        expected = {
            path.relative_to(bundle).as_posix()
            for path in bundle.rglob("*")
            if path.is_file() and path.name != METADATA_NAME
        }
        if set(archive.namelist()) != expected:
            raise ValueError("WSI45630 capacity upload files differ")


def _read_csv(path: Path, header: Sequence[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(header):
            raise ValueError(f"Unexpected header in {path}")
        return [dict(row) for row in reader]


def _identity(row: Mapping[str, str]) -> Identity:
    return (
        int(row["atlas_row_index"]),
        int(row["wsi_id"]),
        int(row["x"]),
        int(row["y"]),
    )


def _csv_bytes(header: Sequence[str], rows: Sequence[Mapping[str, str]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=header, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode()


def _artifact_records(root: Path, *, exclude: set[str]) -> dict[str, dict[str, object]]:
    return {
        path.relative_to(root).as_posix(): {
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.relative_to(root).as_posix() not in exclude
    }


def _records(value: object) -> dict[str, dict[str, object]]:
    if not isinstance(value, dict):
        raise TypeError("Capacity input files must be records")
    records = cast("dict[str, object]", value)
    if not all(isinstance(record, dict) for record in records.values()):
        raise TypeError("Capacity input files must be records")
    return cast("dict[str, dict[str, object]]", value)


def _read_object(path: Path) -> dict[str, object]:
    value = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(value, dict):
        raise TypeError(f"Expected object in {path}")
    return cast("dict[str, object]", value)


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hash(path: Path, expected: str) -> str:
    observed = _sha256(path)
    if observed != expected:
        raise ValueError(f"Frozen authority hash differs: {path}")
    return observed


def main(argv: Sequence[str] | None = None) -> int:
    """Build or validate the fixed local WSI45630 capacity input package."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("build", "validate"),
        nargs="?",
        default="build",
    )
    args = parser.parse_args(argv)
    action = cast("str", args.action)
    result = validate() if action == "validate" else build()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
