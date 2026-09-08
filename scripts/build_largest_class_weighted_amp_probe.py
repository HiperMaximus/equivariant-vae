# Copyright 2026 HiperMaximus
# ruff: noqa: C901, DOC201, DOC501, EM101, EM102, PLR0912, PLR0914, PLR0915, PLR0916, PLR2004, T201, TC003, TRY003
"""Build the exact Spec 0035 input dataset and Kaggle probe package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from string import Template
from typing import Final, cast

from eqvae.cli.build_ubc_supervised_calibration_inputs import stage_upload_envelope
from eqvae.data.supervised_latents import CATALOG_HEADER

ROOT: Final = Path.cwd()
DEFAULT_ROOT: Final = Path("runs/local/largest_class_weighted_amp_probe")
SPEC_PATH: Final = Path("docs/specs/0035-largest-class-weighted-amp-probe.md")
DEV_ROOT: Final = Path(
    "runs/local/ubc_ocean_full_foreground_manifests/development",
)
DEV_CONTRACT: Final = DEV_ROOT / "dataset.json"
DEV_CONTRACT_SHA256: Final = (
    "6aa25a3f56db62b903d3451e154c47dc0039233f88179c7bdab03b7a48319fe0"
)
TRAIN_BAGS_SHA256: Final = (
    "c8b320054a17a68b1e4ba21dfcfd00354d4d27e002793b1bb0bc91ee1ebe209a"
)
TRAIN_INSTANCES_SHA256: Final = (
    "dcc2d2343ce0b86588fffdb4564796178c7193ff4b11eea74dc3bc6d23be992e"
)
CATALOG_SHA256: Final = (
    "9303120aa99ab105eafdb14868bd8e1a1f785b643f149f103bb60beeabf8d92e"
)
DATASET_SLUG: Final = "eqvae-largest-class-weighted-amp-inputs"
KERNEL_ID: Final = "maximusshtefan/eqvae-largest-class-weighted-amp-probe"
CONTRACT_NAME: Final = "largest_class_weighted_amp_input.json"
METADATA_NAME: Final = "dataset-metadata.json"
TEMPLATE_PATH: Final = Path(
    "kaggle/kernels/largest_class_weighted_amp_probe/run_template.py",
)
MODEL_PATH: Final = Path("src/eqvae/models/local_global_mil.py")
CANDIDATE_PATH: Final = Path("src/eqvae/models/local_attention_candidates.py")
PARAMETER_COUNT: Final = 1_513_055
TOTAL_TRAIN_WSI: Final = 106
SELECTED: Final = {
    "CC": {
        "index": 0,
        "count": 23,
        "weight": 106 / (5 * 23),
        "wsi": 51_346,
        "patches": 29_150,
    },
    "EC": {
        "index": 1,
        "count": 24,
        "weight": 106 / (5 * 24),
        "wsi": 45_630,
        "patches": 32_595,
    },
    "HGSC": {
        "index": 2,
        "count": 40,
        "weight": 106 / (5 * 40),
        "wsi": 35_239,
        "patches": 18_981,
    },
    "LGSC": {
        "index": 3,
        "count": 11,
        "weight": 106 / (5 * 11),
        "wsi": 57_162,
        "patches": 24_031,
    },
    "MC": {
        "index": 4,
        "count": 8,
        "weight": 106 / (5 * 8),
        "wsi": 65_094,
        "patches": 23_090,
    },
}
REQUIRED_PARTS: Final = {3, 4, 5, 11, 12, 17, 19, 20}
KERNEL_SOURCES: Final = (
    "maximusshtefan/eqvae-ubc-ocean-latent-run-03",
    "maximusshtefan/eqvae-ubc-ocean-latent-run-04",
    "maximusshtefan/eqvae-ubc-ocean-latent-run-05",
    "maximusshtefan/eqvae-ubc-ocean-cancer-latent-top-up",
    "maximusshtefan/eqvae-wsi45630-completion",
    "maximusshtefan/eqvae-full-foreground-05",
    "maximusshtefan/eqvae-full-foreground-07",
    "maximusshtefan/eqvae-full-foreground-08",
)
POINTER_HEADER: Final = (
    "wsi_id",
    "diagnosis_label",
    "diagnosis_index",
    "atlas_row_index",
    "x",
    "y",
    "part",
    "file_index",
)
BAG_HEADER: Final = (
    "wsi_id",
    "diagnosis_label",
    "diagnosis_index",
    "instance_count",
    "class_count",
    "class_weight",
)


def build(*, actor: str) -> dict[str, object]:
    """Create one immutable local dataset/kernel package."""
    dataset_reference = _dataset_reference(actor)
    output = ROOT / DEFAULT_ROOT
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    assets, provenance, selected_contract = _derive_assets(ROOT)
    staging = output.with_name(f".{output.name}.building")
    if staging.exists():
        raise FileExistsError(f"Stale staging directory exists: {staging}")
    try:
        _build_staging(
            staging,
            assets,
            provenance,
            selected_contract,
            dataset_reference,
        )
        staging.replace(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    stage_upload_envelope(bundle_root=output / "bundle", destination=output / "upload")
    return validate(expected_actor=actor)


def _build_staging(
    staging: Path,
    assets: Mapping[str, bytes],
    provenance: Mapping[str, str],
    selected_contract: Sequence[Mapping[str, object]],
    dataset_reference: str,
) -> None:
    bundle = staging / "bundle"
    for name, payload in assets.items():
        path = bundle / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    _copy_source(ROOT, bundle / "src")
    files = _artifact_records(bundle, exclude={CONTRACT_NAME, METADATA_NAME})
    contract: dict[str, object] = {
        "schema_version": "spec0035.largest_class_weighted_amp_input.v1",
        "dataset_reference": dataset_reference,
        "dataset_actor": dataset_reference.split("/", maxsplit=1)[0],
        "spec_sha256": _sha256(ROOT / SPEC_PATH),
        "scope": "train_only_numerical_probe_not_learning",
        "train_wsi_count": TOTAL_TRAIN_WSI,
        "selected": list(selected_contract),
        "required_parts": sorted(REQUIRED_PARTS),
        "kernel_sources": list(KERNEL_SOURCES),
        "source_provenance": dict(provenance),
        "model_sha256": _sha256(ROOT / MODEL_PATH),
        "candidate_sha256": _sha256(ROOT / CANDIDATE_PATH),
        "parameter_count": PARAMETER_COUNT,
        "files": files,
    }
    _write_json(bundle / CONTRACT_NAME, contract)
    _write_json(bundle / METADATA_NAME, _dataset_metadata(dataset_reference))
    kernel = staging / "kernel"
    kernel.mkdir()
    _write_json(kernel / "kernel-metadata.json", _kernel_metadata(dataset_reference))
    (kernel / "run.py").write_bytes(
        _render(ROOT, _sha256(bundle / CONTRACT_NAME), dataset_reference),
    )


def validate(*, expected_actor: str | None = None) -> dict[str, object]:
    """Reject source, selected-bag, metadata, package or rendered-code drift."""
    output = ROOT / DEFAULT_ROOT
    bundle = output / "bundle"
    contract = _read_object(bundle / CONTRACT_NAME)
    dataset_reference = cast("str", contract.get("dataset_reference"))
    dataset_actor = dataset_reference.split("/", maxsplit=1)[0]
    assets, provenance, selected_contract = _derive_assets(ROOT)
    if (
        contract.get("schema_version") != "spec0035.largest_class_weighted_amp_input.v1"
        or dataset_reference != _dataset_reference(dataset_actor)
        or contract.get("dataset_actor") != dataset_actor
        or (expected_actor is not None and dataset_actor != expected_actor)
        or contract.get("spec_sha256") != _sha256(ROOT / SPEC_PATH)
        or contract.get("scope") != "train_only_numerical_probe_not_learning"
        or contract.get("train_wsi_count") != TOTAL_TRAIN_WSI
        or contract.get("selected") != selected_contract
        or contract.get("required_parts") != sorted(REQUIRED_PARTS)
        or contract.get("kernel_sources") != list(KERNEL_SOURCES)
        or contract.get("source_provenance") != provenance
        or contract.get("model_sha256") != _sha256(ROOT / MODEL_PATH)
        or contract.get("candidate_sha256") != _sha256(ROOT / CANDIDATE_PATH)
        or contract.get("parameter_count") != PARAMETER_COUNT
        or _read_object(bundle / METADATA_NAME) != _dataset_metadata(dataset_reference)
    ):
        raise ValueError("Spec 0035 input contract identity differs")
    files = _records(contract.get("files"))
    if files != _artifact_records(bundle, exclude={CONTRACT_NAME, METADATA_NAME}):
        raise ValueError("Spec 0035 input bundle bytes differ")
    for name, payload in assets.items():
        if (bundle / name).read_bytes() != payload:
            raise ValueError(f"Spec 0035 derived asset differs: {name}")
    _validate_source_snapshot(bundle, files)
    _validate_upload(output / "upload", bundle, dataset_reference)
    kernel = output / "kernel"
    if {path.name for path in kernel.iterdir()} != {"kernel-metadata.json", "run.py"}:
        raise ValueError("Spec 0035 kernel allow-list differs")
    if _read_object(kernel / "kernel-metadata.json") != _kernel_metadata(
        dataset_reference,
    ):
        raise ValueError("Spec 0035 kernel metadata differs")
    if (kernel / "run.py").read_bytes() != _render(
        ROOT,
        _sha256(bundle / CONTRACT_NAME),
        dataset_reference,
    ):
        raise ValueError("Spec 0035 rendered launcher differs")
    return contract


def _derive_assets(
    root: Path,
) -> tuple[dict[str, bytes], dict[str, str], list[dict[str, object]]]:
    """Select the unique largest complete training WSI in each diagnosis."""
    dev = root / DEV_ROOT
    provenance = {
        "development_contract": _require_hash(root / DEV_CONTRACT, DEV_CONTRACT_SHA256),
        "train_bags": _require_hash(
            dev / "wsi_cancer_train_bags.csv",
            TRAIN_BAGS_SHA256,
        ),
        "train_instances": _require_hash(
            dev / "wsi_cancer_train_instances.csv",
            TRAIN_INSTANCES_SHA256,
        ),
        "physical_catalog": _require_hash(dev / "physical_parts.csv", CATALOG_SHA256),
    }
    bags = _read_csv(dev / "wsi_cancer_train_bags.csv")
    if len(bags) != TOTAL_TRAIN_WSI:
        raise ValueError("Training WSI count differs")
    counts = Counter(row["diagnosis_label"] for row in bags)
    largest: dict[str, dict[str, str]] = {}
    for row in bags:
        label = row["diagnosis_label"]
        if label not in largest or int(row["instance_count"]) > int(
            largest[label]["instance_count"],
        ):
            largest[label] = row
    selected_ids = {cast("int", value["wsi"]) for value in SELECTED.values()}
    selected_rows = {int(row["wsi_id"]): row for row in largest.values()}
    if set(selected_rows) != selected_ids:
        raise ValueError("Largest-per-class WSI identities differ")
    selected_contract: list[dict[str, object]] = []
    bag_assets: list[dict[str, object]] = []
    for label, expected in sorted(
        SELECTED.items(),
        key=lambda item: cast("int", item[1]["index"]),
    ):
        row = selected_rows[cast("int", expected["wsi"])]
        observed_weight = TOTAL_TRAIN_WSI / (5 * counts[label])
        if (
            int(row["diagnosis_index"]) != expected["index"]
            or int(row["instance_count"]) != expected["patches"]
            or counts[label] != expected["count"]
            or observed_weight != expected["weight"]
        ):
            raise ValueError(f"Largest-per-class contract differs for {label}")
        record: dict[str, object] = {
            "diagnosis_label": label,
            "diagnosis_index": expected["index"],
            "class_count": expected["count"],
            "class_weight": expected["weight"],
            "wsi_id": expected["wsi"],
            "patch_count": expected["patches"],
        }
        selected_contract.append(record)
        bag_assets.append({
            "wsi_id": expected["wsi"],
            "diagnosis_label": label,
            "diagnosis_index": expected["index"],
            "instance_count": expected["patches"],
            "class_count": expected["count"],
            "class_weight": repr(expected["weight"]),
        })
    pointer_assets: list[dict[str, str]] = []
    part_counts: Counter[int] = Counter()
    per_wsi: Counter[int] = Counter()
    per_wsi_parts: dict[int, Counter[int]] = {
        cast("int", value["wsi"]): Counter() for value in SELECTED.values()
    }
    coordinates: set[tuple[int, int, int]] = set()
    for row in _read_csv(dev / "wsi_cancer_train_instances.csv"):
        wsi_id = int(row["wsi_id"])
        if wsi_id not in selected_ids:
            continue
        coordinate = (wsi_id, int(row["x"]), int(row["y"]))
        if coordinate in coordinates:
            raise ValueError("Duplicate selected WSI coordinate")
        coordinates.add(coordinate)
        part = int(row["part"])
        part_counts[part] += 1
        per_wsi[wsi_id] += 1
        per_wsi_parts[wsi_id][part] += 1
        pointer_assets.append({key: row[key] for key in POINTER_HEADER})
    if set(part_counts) != REQUIRED_PARTS:
        raise ValueError("Selected physical-part set differs")
    if any(
        per_wsi[cast("int", value["wsi"])] != value["patches"]
        for value in SELECTED.values()
    ):
        raise ValueError("Selected WSI pointer counts differ")
    for record in selected_contract:
        record["part_counts"] = {
            str(part): count
            for part, count in sorted(
                per_wsi_parts[cast("int", record["wsi_id"])].items(),
            )
        }
    pointer_assets.sort(
        key=lambda row: (int(row["wsi_id"]), int(row["y"]), int(row["x"])),
    )
    catalog = [
        row
        for row in _read_csv(dev / "physical_parts.csv")
        if int(row["part"]) in REQUIRED_PARTS
    ]
    if len(catalog) != 2 * len(REQUIRED_PARTS):
        raise ValueError("Selected physical catalog cardinality differs")
    sources = tuple(dict.fromkeys(row["kaggle_source"] for row in catalog))
    if sources != KERNEL_SOURCES:
        raise ValueError("Selected producer locator order differs")
    assets = {
        "probe/physical_parts.csv": _csv_bytes(CATALOG_HEADER, catalog),
        "probe/bags.csv": _csv_bytes(BAG_HEADER, bag_assets),
        "probe/pointers.csv": _csv_bytes(POINTER_HEADER, pointer_assets),
    }
    return assets, provenance, selected_contract


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
        raise ValueError("Spec 0035 source allow-list differs")
    for name, source in expected.items():
        if (bundle / name).read_bytes() != source.read_bytes():
            raise ValueError(f"Spec 0035 source differs: {name}")


def _render(root: Path, contract_hash: str, dataset_reference: str) -> bytes:
    template = (root / TEMPLATE_PATH).read_text(encoding="utf-8")
    placeholders = {
        "input_contract_sha256": contract_hash,
        "input_dataset_reference": dataset_reference,
    }
    if any(template.count(f"${name}") != 1 for name in placeholders):
        raise ValueError("Spec 0035 template placeholder count differs")
    rendered = Template(template).substitute(placeholders).encode()
    if len(rendered) >= 1_000_000:
        raise ValueError("Spec 0035 launcher exceeds Kaggle's 1 MB limit")
    compile(rendered, str(root / TEMPLATE_PATH), "exec")
    return rendered


def _dataset_reference(actor: str) -> str:
    if not actor or "/" in actor or actor.strip() != actor:
        raise ValueError("Kaggle actor must be one nonempty owner component")
    return f"{actor}/{DATASET_SLUG}"


def _dataset_metadata(dataset_reference: str) -> dict[str, object]:
    return {
        "id": dataset_reference,
        "title": "eqvae largest class weighted amp inputs",
        "licenses": [{"name": "other"}],
    }


def _kernel_metadata(dataset_reference: str) -> dict[str, object]:
    return {
        "id": KERNEL_ID,
        "title": "eqvae largest class weighted amp probe",
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [dataset_reference],
        "competition_sources": [],
        "kernel_sources": list(KERNEL_SOURCES),
        "model_sources": [],
    }


def _validate_upload(upload: Path, bundle: Path, dataset_reference: str) -> None:
    if {path.name for path in upload.iterdir()} != {"bundle.zip", METADATA_NAME}:
        raise ValueError("Spec 0035 upload envelope differs")
    if _read_object(upload / METADATA_NAME) != _dataset_metadata(dataset_reference):
        raise ValueError("Spec 0035 upload metadata differs")
    with zipfile.ZipFile(upload / "bundle.zip") as archive:
        expected = {
            path.relative_to(bundle).as_posix()
            for path in bundle.rglob("*")
            if path.is_file() and path.name != METADATA_NAME
        }
        if set(archive.namelist()) != expected:
            raise ValueError("Spec 0035 upload archive allow-list differs")
        for name in expected:
            if archive.read(name) != (bundle / name).read_bytes():
                raise ValueError(f"Spec 0035 upload member differs: {name}")


def _artifact_records(root: Path, *, exclude: set[str]) -> dict[str, object]:
    return {
        path.relative_to(root).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in exclude
    }


def _records(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("Artifact records must be an object")
    return cast("dict[str, object]", value)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _csv_bytes(header: Sequence[str], rows: Sequence[Mapping[str, object]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=header, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_object(path: Path) -> dict[str, object]:
    value = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return cast("dict[str, object]", value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hash(path: Path, expected: str) -> str:
    observed = _sha256(path)
    if observed != expected:
        raise ValueError(f"Authority hash differs for {path}: {observed}")
    return observed


def main(argv: Sequence[str] | None = None) -> int:
    """Build or validate the local Spec 0035 package."""
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("build", "validate"))
    parser.add_argument("--actor")
    args = parser.parse_args(argv)
    action = cast("str", args.action)
    actor = cast("str | None", args.actor)
    if action == "build" and actor is None:
        parser.error("build requires --actor with the authenticated Kaggle username")
    contract = (
        build(actor=cast("str", actor))
        if action == "build"
        else validate(
            expected_actor=actor,
        )
    )
    print(json.dumps(contract, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
