# Copyright 2026 HiperMaximus
# ruff: noqa: ARG001, C420, DOC201, DOC501, EM101, EM102, PLR2004, S311, T201, TC003, TRY003
"""Build the immutable train-only input and launcher for Spec 0037."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import shutil
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from string import Template
from typing import Final, cast

from eqvae.cli.build_ubc_supervised_calibration_inputs import stage_upload_envelope
from eqvae.data.supervised_latents import CATALOG_HEADER, TISSUE_HEADER

ROOT: Final = Path.cwd()
DEFAULT_ROOT: Final = Path("runs/local/tissue_fastpath_calibration_probe")
SPEC_PATH: Final = Path("docs/specs/0037-tissue-fastpath-calibration-probe.md")
MANIFEST_ROOT: Final = Path("runs/local/ubc_ocean_supervised_manifests")
DATASET_SLUG: Final = "eqvae-tissue-fastpath-probe-inputs"
KERNEL_ID: Final = "maximusshtefan/eqvae-tissue-fastpath-probe"
CONTRACT_NAME: Final = "tissue_fastpath_probe_input.json"
METADATA_NAME: Final = "dataset-metadata.json"
TEMPLATE_PATH: Final = Path("kaggle/kernels/tissue_fastpath_probe/run_template.py")
TRAIN_MANIFEST: Final = Path("tissue/tissue_train_5671_per_class.csv")
CATALOG: Final = Path("physical_parts.csv")
TRAIN_AUDIT_NAME: Final = "train_manifest_audit.json"
PROBE_BATCHES_NAME: Final = "probe_batches.json"
SOURCE_ROOT: Final = Path("src/eqvae")
LABELS: Final = ("tumor", "stroma", "necrosis")
PER_CLASS_ROWS: Final = 5_671
TOTAL_ROWS: Final = 17_013
INITIALIZATION_SEED: Final = 3_407
BATCH_GEOMETRY: Final = {
    250: {"total_rows": 750, "batch_size": 125, "steps_per_epoch": 6},
    500: {"total_rows": 1_500, "batch_size": 125, "steps_per_epoch": 12},
    1_000: {"total_rows": 3_000, "batch_size": 125, "steps_per_epoch": 24},
    2_500: {"total_rows": 7_500, "batch_size": 125, "steps_per_epoch": 60},
    5_671: {"total_rows": 17_013, "batch_size": 159, "steps_per_epoch": 107},
}
KERNEL_SOURCES: Final = (
    "maximusshtefan/eqvae-ubc-ocean-latent-run-01",
    "maximusshtefan/eqvae-ubc-ocean-latent-run-02",
    "maximusshtefan/eqvae-ubc-ocean-latent-run-03",
    "maximusshtefan/eqvae-ubc-ocean-latent-run-04",
    "maximusshtefan/eqvae-ubc-ocean-latent-run-05",
    "maximusshtefan/eqvae-ubc-ocean-cancer-latent-top-up",
)


def build(*, actor: str) -> dict[str, object]:
    """Create one non-overwritable actor-portable Spec 0037 package."""
    output = ROOT / DEFAULT_ROOT
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    dataset_reference = _dataset_reference(actor)
    assets, manifest, probe_batches = _derive_assets(ROOT)
    staging = output.with_name(f".{output.name}.building")
    if staging.exists():
        raise FileExistsError(f"Stale staging directory exists: {staging}")
    try:
        _build_staging(staging, dataset_reference, assets, manifest, probe_batches)
        staging.replace(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    stage_upload_envelope(bundle_root=output / "bundle", destination=output / "upload")
    return validate(expected_actor=actor)


def _build_staging(
    staging: Path,
    dataset_reference: str,
    assets: Mapping[str, bytes],
    manifest: Mapping[str, object],
    probe_batches: Mapping[str, object],
) -> None:
    bundle = staging / "bundle"
    for name, payload in assets.items():
        target = bundle / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    _copy_source(ROOT, bundle / "src")
    files = _artifact_records(bundle, exclude={CONTRACT_NAME, METADATA_NAME})
    contract: dict[str, object] = {
        "schema_version": "spec0037.tissue_fastpath_probe_input.v1",
        "dataset_reference": dataset_reference,
        "dataset_actor": dataset_reference.split("/", maxsplit=1)[0],
        "visibility": "private",
        "scope": "tissue_train_only_runtime_amp_probe_not_learning",
        "spec_sha256": _sha256(ROOT / SPEC_PATH),
        "model": {
            "source": "src/eqvae/models/supervised.py",
            "sha256": _sha256(ROOT / "src/eqvae/models/supervised.py"),
            "initialization_seed": INITIALIZATION_SEED,
            "classes": list(LABELS),
        },
        "batch_geometry": {str(key): value for key, value in BATCH_GEOMETRY.items()},
        "kernel_sources": list(KERNEL_SOURCES),
        "physical_sources": manifest["physical_sources"],
        "train_manifest": manifest,
        "probe_batches_sha256": _sha256(bundle / PROBE_BATCHES_NAME),
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


def validate(
    *,
    expected_actor: str | None = None,
    sealed_source_snapshot: bool = False,
) -> dict[str, object]:
    """Reject any logical, source-snapshot, packaging or rendering drift."""
    output = ROOT / DEFAULT_ROOT
    bundle = output / "bundle"
    contract = _read_object(bundle / CONTRACT_NAME)
    dataset_reference = cast("str", contract.get("dataset_reference"))
    actor = dataset_reference.split("/", maxsplit=1)[0]
    if sealed_source_snapshot:
        identity_matches = _sealed_contract_identity_matches(
            contract,
            actor=actor,
            expected_actor=expected_actor,
            bundle=bundle,
        )
    else:
        assets, manifest, probe_batches = _derive_assets(ROOT)
        identity_matches = _live_contract_identity_matches(
            contract,
            actor=actor,
            expected_actor=expected_actor,
            bundle=bundle,
            manifest=manifest,
        )
    if not identity_matches:
        raise ValueError("Spec 0037 input contract identity differs")
    files = _records(contract.get("files"))
    if files != _artifact_records(bundle, exclude={CONTRACT_NAME, METADATA_NAME}):
        raise ValueError("Spec 0037 input bundle bytes differ")
    if sealed_source_snapshot:
        _validate_sealed_source_snapshot(contract, files)
    else:
        for name, payload in assets.items():
            if (bundle / name).read_bytes() != payload:
                raise ValueError(f"Spec 0037 derived asset differs: {name}")
        if _read_object(bundle / PROBE_BATCHES_NAME) != probe_batches:
            raise ValueError("Spec 0037 fixed probe batches differ")
        _validate_source_snapshot(bundle, files)
    _validate_upload(output / "upload", bundle, dataset_reference)
    _validate_kernel(
        output / "kernel",
        bundle=bundle,
        dataset_reference=dataset_reference,
        sealed_source_snapshot=sealed_source_snapshot,
    )
    return contract


def _live_contract_identity_matches(
    contract: Mapping[str, object],
    *,
    actor: str,
    expected_actor: str | None,
    bundle: Path,
    manifest: Mapping[str, object],
) -> bool:
    expected_geometry = {str(key): value for key, value in BATCH_GEOMETRY.items()}
    return (
        contract.get("schema_version") == "spec0037.tissue_fastpath_probe_input.v1"
        and contract.get("dataset_reference") == _dataset_reference(actor)
        and contract.get("dataset_actor") == actor
        and (expected_actor is None or actor == expected_actor)
        and contract.get("visibility") == "private"
        and contract.get("scope") == "tissue_train_only_runtime_amp_probe_not_learning"
        and contract.get("spec_sha256") == _sha256(ROOT / SPEC_PATH)
        and contract.get("model")
        == {
            "source": "src/eqvae/models/supervised.py",
            "sha256": _sha256(ROOT / "src/eqvae/models/supervised.py"),
            "initialization_seed": INITIALIZATION_SEED,
            "classes": list(LABELS),
        }
        and contract.get("batch_geometry") == expected_geometry
        and contract.get("kernel_sources") == list(KERNEL_SOURCES)
        and contract.get("physical_sources") == manifest["physical_sources"]
        and contract.get("train_manifest") == manifest
        and contract.get("probe_batches_sha256") == _sha256(bundle / PROBE_BATCHES_NAME)
        and _read_object(bundle / METADATA_NAME)
        == _dataset_metadata(
            cast("str", contract["dataset_reference"]),
        )
    )


def _sealed_contract_identity_matches(
    contract: Mapping[str, object],
    *,
    actor: str,
    expected_actor: str | None,
    bundle: Path,
) -> bool:
    return (
        contract.get("schema_version") == "spec0037.tissue_fastpath_probe_input.v1"
        and contract.get("dataset_reference") == _dataset_reference(actor)
        and contract.get("dataset_actor") == actor
        and (expected_actor is None or actor == expected_actor)
        and contract.get("visibility") == "private"
        and contract.get("scope") == "tissue_train_only_runtime_amp_probe_not_learning"
        and contract.get("batch_geometry")
        == {str(key): value for key, value in BATCH_GEOMETRY.items()}
        and contract.get("kernel_sources") == list(KERNEL_SOURCES)
        and _read_object(bundle / METADATA_NAME)
        == _dataset_metadata(
            cast("str", contract["dataset_reference"]),
        )
    )


def _validate_kernel(
    kernel: Path,
    *,
    bundle: Path,
    dataset_reference: str,
    sealed_source_snapshot: bool,
) -> None:
    if {path.name for path in kernel.iterdir()} != {"kernel-metadata.json", "run.py"}:
        raise ValueError("Spec 0037 kernel allow-list differs")
    if _read_object(kernel / "kernel-metadata.json") != _kernel_metadata(
        dataset_reference,
    ):
        raise ValueError("Spec 0037 kernel metadata differs")
    launcher = (kernel / "run.py").read_bytes()
    if sealed_source_snapshot:
        _validate_sealed_launcher(
            launcher,
            contract_sha256=_sha256(bundle / CONTRACT_NAME),
            dataset_reference=dataset_reference,
        )
    elif launcher != _render(
        ROOT,
        _sha256(bundle / CONTRACT_NAME),
        dataset_reference,
    ):
        raise ValueError("Spec 0037 rendered launcher differs")


def _derive_assets(
    root: Path,
) -> tuple[dict[str, bytes], dict[str, object], dict[str, object]]:
    """Derive a train-only manifest audit and fixed balanced stress batches."""
    manifest_root = root / MANIFEST_ROOT
    catalog = manifest_root / CATALOG
    train = manifest_root / TRAIN_MANIFEST
    catalog_bytes = catalog.read_bytes()
    train_bytes = train.read_bytes()
    catalog_rows = _read_csv_bytes(catalog_bytes, CATALOG_HEADER)
    train_rows = _read_csv_bytes(train_bytes, TISSUE_HEADER)
    if any(
        "validation" in name or "test" in name
        for name in (CATALOG.as_posix(), TRAIN_MANIFEST.as_posix())
    ):
        raise ValueError("Spec 0037 input names broaden beyond tissue training")
    if len(train_rows) != TOTAL_ROWS or {row["split"] for row in train_rows} != {
        "train",
    }:
        raise ValueError("Spec 0037 requires exactly the 5,671-per-class train pool")
    labels = Counter(row["tissue_label"] for row in train_rows)
    if labels != Counter({label: PER_CLASS_ROWS for label in LABELS}):
        raise ValueError("Spec 0037 tissue class balance differs")
    if [int(row["dataset_row"]) for row in train_rows] != list(range(TOTAL_ROWS)):
        raise ValueError("Spec 0037 tissue manifest rows are not contiguous")
    physical_sources = _physical_sources(catalog_rows)
    manifest: dict[str, object] = {
        "schema_version": "spec0037.tissue_train_manifest_audit.v1",
        "train_rows": TOTAL_ROWS,
        "per_class_rows": {label: PER_CLASS_ROWS for label in LABELS},
        "labels": list(LABELS),
        "catalog_sha256": _sha256(catalog),
        "train_csv_sha256": _sha256(train),
        "physical_sources": physical_sources,
    }
    probe_batches = _probe_batches(train_rows)
    assets = {
        CATALOG.as_posix(): catalog_bytes,
        TRAIN_MANIFEST.as_posix(): train_bytes,
        TRAIN_AUDIT_NAME: _json_bytes(manifest),
        PROBE_BATCHES_NAME: _json_bytes(probe_batches),
    }
    return assets, manifest, probe_batches


def _physical_sources(rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    if len(rows) != 12:
        raise ValueError("Spec 0037 physical catalog must contain six paired parts")
    sources = tuple(dict.fromkeys(row["kaggle_source"] for row in rows))
    if sources != KERNEL_SOURCES:
        raise ValueError("Spec 0037 latent producer locators differ")
    if {row["model_name"] for row in rows} != {"normal_vae", "so2_vae"}:
        raise ValueError("Spec 0037 catalog representations differ")
    return [dict(row) for row in rows]


def _probe_batches(rows: Sequence[Mapping[str, str]]) -> dict[str, object]:
    """Select three deterministic balanced stress batches for each static shape."""
    by_label = {
        label: [index for index, row in enumerate(rows) if row["tissue_label"] == label]
        for label in LABELS
    }
    if {len(indices) for indices in by_label.values()} != {PER_CLASS_ROWS}:
        raise ValueError("Spec 0037 probe source balance differs")
    selected: dict[str, list[int]] = {}
    for label_index, label in enumerate(LABELS):
        indices = by_label[label].copy()
        random.Random(INITIALIZATION_SEED + 100 * (label_index + 1)).shuffle(indices)
        selected[label] = indices[:159]
    batches_159 = [
        _mixed_batch(
            {
                label: selected[label][53 * ordinal : 53 * (ordinal + 1)]
                for label in LABELS
            },
            ordinal,
        )
        for ordinal in range(3)
    ]
    sizes_125 = (
        {"tumor": 42, "stroma": 42, "necrosis": 41},
        {"tumor": 42, "stroma": 41, "necrosis": 42},
        {"tumor": 41, "stroma": 42, "necrosis": 42},
    )
    cursors = {label: 0 for label in LABELS}
    batches_125: list[list[int]] = []
    for ordinal, composition in enumerate(sizes_125):
        groups: dict[str, list[int]] = {}
        for label in LABELS:
            start = cursors[label]
            stop = start + composition[label]
            groups[label] = selected[label][start:stop]
            cursors[label] = stop
        batches_125.append(_mixed_batch(groups, ordinal))
    if cursors != {label: 125 for label in LABELS}:
        raise ValueError("Spec 0037 125-batch class rotation differs")
    return {
        "schema_version": "spec0037.tissue_probe_batches.v1",
        "batch_sizes": {
            "125": batches_125,
            "159": batches_159,
        },
        "class_counts": {
            "125": [
                {
                    label: len([
                        index for index in batch if rows[index]["tissue_label"] == label
                    ])
                    for label in LABELS
                }
                for batch in batches_125
            ],
            "159": [{label: 53 for label in LABELS} for _ in range(3)],
        },
    }


def _mixed_batch(groups: Mapping[str, Sequence[int]], ordinal: int) -> list[int]:
    mixed = [index for label in LABELS for index in groups[label]]
    random.Random(INITIALIZATION_SEED + 10_000 + ordinal).shuffle(mixed)
    return mixed


def _copy_source(root: Path, destination: Path) -> None:
    for source in sorted((root / SOURCE_ROOT).rglob("*.py")):
        if "__pycache__" not in source.parts:
            target = destination / source.relative_to(root / "src")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def _validate_source_snapshot(bundle: Path, files: Mapping[str, object]) -> None:
    expected = {
        f"src/{path.relative_to(ROOT / 'src').as_posix()}": path
        for path in sorted((ROOT / SOURCE_ROOT).rglob("*.py"))
        if "__pycache__" not in path.parts
    }
    if {name for name in files if name.startswith("src/")} != set(expected):
        raise ValueError("Spec 0037 source allow-list differs")
    for name, source in expected.items():
        if (bundle / name).read_bytes() != source.read_bytes():
            raise ValueError(f"Spec 0037 source differs: {name}")


def _validate_sealed_source_snapshot(
    contract: Mapping[str, object],
    files: Mapping[str, object],
) -> None:
    source_names = {name for name in files if name.startswith("src/")}
    if not source_names or any(
        not name.startswith("src/eqvae/") or not name.endswith(".py")
        for name in source_names
    ):
        raise ValueError("Spec 0037 sealed source allow-list differs")
    model = contract.get("model")
    model_record = files.get("src/eqvae/models/supervised.py")
    if not isinstance(model, dict) or not isinstance(model_record, dict):
        raise TypeError("Spec 0037 sealed model binding type differs")
    expected_model = {
        "source": "src/eqvae/models/supervised.py",
        "initialization_seed": INITIALIZATION_SEED,
        "classes": list(LABELS),
    }
    if any(model.get(key) != value for key, value in expected_model.items()) or (
        model.get("sha256") != model_record.get("sha256")
    ):
        raise ValueError("Spec 0037 sealed model binding differs")


def _validate_sealed_launcher(
    launcher: bytes,
    *,
    contract_sha256: str,
    dataset_reference: str,
) -> None:
    source = launcher.decode("utf-8")
    compile(source, str(TEMPLATE_PATH), "exec")
    required = (
        "SPEC0037_TISSUE_FASTPATH_PROBE_READY = True",
        f'INPUT_CONTRACT_SHA256 = "{contract_sha256}"',
        f'INPUT_DATASET_REFERENCE = "{dataset_reference}"',
    )
    if not all(fragment in source for fragment in required):
        raise ValueError("Spec 0037 sealed launcher binding differs")


def _render(root: Path, contract_hash: str, dataset_reference: str) -> bytes:
    template = (root / TEMPLATE_PATH).read_text(encoding="utf-8")
    placeholders = {
        "input_contract_sha256": contract_hash,
        "input_dataset_reference": dataset_reference,
    }
    if any(template.count(f"${name}") != 1 for name in placeholders):
        raise ValueError("Spec 0037 template placeholder count differs")
    rendered = Template(template).substitute(placeholders).encode()
    if len(rendered) >= 1_000_000:
        raise ValueError("Spec 0037 launcher exceeds Kaggle's 1 MB limit")
    compile(rendered, str(root / TEMPLATE_PATH), "exec")
    return rendered


def _dataset_reference(actor: str) -> str:
    if not actor or "/" in actor or actor.strip() != actor:
        raise ValueError("Kaggle actor must be one nonempty owner component")
    return f"{actor}/{DATASET_SLUG}"


def _dataset_metadata(dataset_reference: str) -> dict[str, object]:
    return {
        "id": dataset_reference,
        "title": "eqvae tissue fastpath probe inputs",
        "licenses": [{"name": "other"}],
    }


def _kernel_metadata(dataset_reference: str) -> dict[str, object]:
    return {
        "id": KERNEL_ID,
        "title": "eqvae tissue fastpath calibration probe",
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
        raise ValueError("Spec 0037 upload envelope differs")
    if _read_object(upload / METADATA_NAME) != _dataset_metadata(dataset_reference):
        raise ValueError("Spec 0037 upload metadata differs")
    with zipfile.ZipFile(upload / "bundle.zip") as archive:
        expected = {
            path.relative_to(bundle).as_posix()
            for path in bundle.rglob("*")
            if path.is_file() and path.name != METADATA_NAME
        }
        if set(archive.namelist()) != expected:
            raise ValueError("Spec 0037 upload archive allow-list differs")
        for name in expected:
            if archive.read(name) != (bundle / name).read_bytes():
                raise ValueError(f"Spec 0037 upload member differs: {name}")


def _artifact_records(root: Path, *, exclude: set[str]) -> dict[str, object]:
    return {
        path.relative_to(root).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in exclude
    }


def _read_csv_bytes(payload: bytes, header: Sequence[str]) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8")))
    if tuple(reader.fieldnames or ()) != tuple(header):
        raise ValueError("Spec 0037 CSV header differs")
    return list(reader)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_json_bytes(value))


def _read_object(path: Path) -> dict[str, object]:
    value = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return cast("dict[str, object]", value)


def _records(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("Artifact records must be an object")
    return cast("dict[str, object]", value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    """Build or validate the actor-portable Spec 0037 package."""
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("build", "validate"))
    parser.add_argument("--actor")
    parser.add_argument("--sealed-source-snapshot", action="store_true")
    args = parser.parse_args(argv)
    actor = cast("str | None", args.actor)
    if args.action == "build" and actor is None:
        parser.error("build requires --actor with the authenticated Kaggle username")
    contract = (
        build(actor=cast("str", actor))
        if args.action == "build"
        else validate(
            expected_actor=actor,
            sealed_source_snapshot=args.sealed_source_snapshot,
        )
    )
    print(json.dumps(contract, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
