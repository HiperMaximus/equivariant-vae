# Copyright 2026 HiperMaximus
# ruff: noqa: ARG001, C401, C420, COM812, D103, DOC201, DOC501, EM101, EM102, FURB118, PLR0914, PLR2004, T201, TRY003, TRY004
"""Build the sealed development-only input and launcher for Spec 0039."""

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
from fractions import Fraction
from pathlib import Path
from string import Template
from typing import Final, cast

import numpy as np
import torch

from eqvae.cli.build_ubc_supervised_calibration_inputs import stage_upload_envelope
from eqvae.data.supervised_latents import CATALOG_HEADER, TISSUE_HEADER
from eqvae.models.supervised import TissueClassifier
from eqvae.training.supervised_pairing import make_paired_models

ROOT: Final = Path.cwd()
DEFAULT_ROOT: Final = Path("runs/local/tissue_label_efficiency_training")
SPEC_PATH: Final = Path("docs/specs/0039-tissue-label-efficiency-training.md")
MANIFEST_ROOT: Final = Path("runs/local/ubc_ocean_supervised_manifests")
DATASET_SLUG: Final = "eqvae-tissue-label-efficiency-training-inputs"
KERNEL_ID: Final = "maximusshtefan/eqvae-tissue-label-efficiency-training"
CONTRACT_NAME: Final = "tissue_training_input.json"
METADATA_NAME: Final = "dataset-metadata.json"
TEMPLATE_PATH: Final = Path("kaggle/kernels/tissue_training/run_template.py")
CATALOG: Final = Path("physical_parts.csv")
MANIFEST_AUDIT: Final = Path("spec0023_supervised_manifest_audit.json")
SOURCE_ROOT: Final = Path("src/eqvae")
CONFIG_NAME: Final = "tissue_training_config.json"
INITIAL_STATE_NAME: Final = "tissue_initial_state.pt"
LABELS: Final = ("tumor", "stroma", "necrosis")
INITIALIZATION_SEED: Final = 3_407
PEAK_LR: Final = 1.5e-3
WEIGHT_DECAY: Final = 5e-3
MAXIMUM_EPOCHS: Final = 30
PATIENCE_CHECKS: Final = 10
MINIMUM_COMPLETED_EPOCHS: Final = 10
VALIDATION_BATCH_SIZE: Final = 159
VALIDATION_SELECTION_SEED: Final = 20_260_904
KERNEL_SOURCES: Final = (
    "maximusshtefan/eqvae-ubc-ocean-latent-run-01",
    "maximusshtefan/eqvae-ubc-ocean-latent-run-02",
    "maximusshtefan/eqvae-ubc-ocean-latent-run-03",
    "maximusshtefan/eqvae-ubc-ocean-latent-run-04",
    "maximusshtefan/eqvae-ubc-ocean-latent-run-05",
    "maximusshtefan/eqvae-ubc-ocean-cancer-latent-top-up",
)
BUDGETS: Final = {
    250: {"total_rows": 750, "batch_size": 125, "steps_per_epoch": 6},
    500: {"total_rows": 1_500, "batch_size": 125, "steps_per_epoch": 12},
    1_000: {"total_rows": 3_000, "batch_size": 125, "steps_per_epoch": 24},
    2_500: {"total_rows": 7_500, "batch_size": 125, "steps_per_epoch": 60},
    5_671: {"total_rows": 17_013, "batch_size": 159, "steps_per_epoch": 107},
}
VALIDATION_PER_CLASS: Final = {
    250: 250,
    500: 500,
    1_000: 1_000,
    2_500: 1_250,
    5_671: 1_250,
}


def build(*, actor: str) -> dict[str, object]:
    """Create one non-overwritable, actor-portable development campaign."""
    output = ROOT / DEFAULT_ROOT
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    dataset_reference = _dataset_reference(actor)
    assets, configuration, logical = _derive_assets(ROOT)
    staging = output.with_name(f".{output.name}.building")
    if staging.exists():
        raise FileExistsError(f"Stale staging directory exists: {staging}")
    try:
        _build_staging(staging, dataset_reference, assets, configuration, logical)
        staging.replace(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    stage_upload_envelope(bundle_root=output / "bundle", destination=output / "upload")
    return validate(expected_actor=actor)


def build_retry(
    *,
    actor: str,
    input_bundle: Path,
    output_root: Path,
) -> dict[str, object]:
    """Build a kernel-only retry against one already verified immutable input."""
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite {output_root}")
    contract = _validate_retry_input(input_bundle, actor=actor)
    staging = output_root.with_name(f".{output_root.name}.building")
    if staging.exists():
        raise FileExistsError(f"Stale staging directory exists: {staging}")
    try:
        _build_retry_staging(staging, input_bundle=input_bundle, contract=contract)
        staging.replace(output_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return validate_retry(output_root=output_root, expected_actor=actor)


def _build_retry_staging(
    staging: Path, *, input_bundle: Path, contract: Mapping[str, object]
) -> None:
    """Copy a frozen input and render only the repaired execution envelope."""
    shutil.copytree(input_bundle, staging / "bundle")
    kernel = staging / "kernel"
    kernel.mkdir()
    reference = cast("str", contract["dataset_reference"])
    _write_json(kernel / "kernel-metadata.json", _kernel_metadata(reference))
    (kernel / "run.py").write_bytes(
        _render(ROOT, _sha256(input_bundle / CONTRACT_NAME), reference),
    )


def validate_retry(*, output_root: Path, expected_actor: str) -> dict[str, object]:
    """Validate a retry package without comparing its frozen source to HEAD."""
    bundle = output_root / "bundle"
    contract = _validate_retry_input(bundle, actor=expected_actor)
    _validate_kernel(
        output_root / "kernel",
        bundle=bundle,
        dataset_reference=cast("str", contract["dataset_reference"]),
        sealed_source_snapshot=True,
    )
    return contract


def _validate_retry_input(input_bundle: Path, *, actor: str) -> dict[str, object]:
    """Authenticate a frozen development input as an immutable retry base."""
    contract_path = input_bundle / CONTRACT_NAME
    contract = _read_object(contract_path)
    reference = _dataset_reference(actor)
    files = contract.get("files")
    expected_identity = {
        "schema_version": "spec0039.tissue_training_input.v1",
        "dataset_reference": reference,
        "dataset_actor": actor,
        "visibility": "private",
        "scope": "tissue_development_training_no_sealed_test",
        "kernel_sources": list(KERNEL_SOURCES),
    }
    if any(
        contract.get(key) != value for key, value in expected_identity.items()
    ) or not isinstance(files, dict):
        raise ValueError("Spec 0039 retry input identity differs")
    observed = {
        path.relative_to(input_bundle).as_posix()
        for path in input_bundle.rglob("*")
        if path.is_file()
    }
    if observed != {*files, CONTRACT_NAME}:
        raise ValueError("Spec 0039 retry input allow-list differs")
    for name, record in files.items():
        if not isinstance(record, dict):
            raise ValueError("Spec 0039 retry file record is malformed")
        path = input_bundle / name
        if path.stat().st_size != record.get("bytes") or _sha256(path) != record.get(
            "sha256",
        ):
            raise ValueError(f"Spec 0039 retry input differs: {name}")
    config_path = input_bundle / CONFIG_NAME
    if _sha256(config_path) != contract.get("configuration_sha256"):
        raise ValueError("Spec 0039 retry configuration binding differs")
    configuration = _read_object(config_path)
    _validate_retry_configuration(configuration)
    model = contract.get("model")
    initial_state = input_bundle / INITIAL_STATE_NAME
    if (
        not isinstance(model, dict)
        or model.get("source") != "src/eqvae/models/supervised.py"
        or model.get("initialization_seed") != INITIALIZATION_SEED
        or model.get("classes") != list(LABELS)
        or model.get("initial_state_file_sha256") != _sha256(initial_state)
    ):
        raise ValueError("Spec 0039 retry model binding differs")
    return contract


def _validate_retry_configuration(configuration: Mapping[str, object]) -> None:
    """Keep the retry on the selected learning protocol, not current source bytes."""
    _assets, selected, _logical = _derive_assets(ROOT)
    for key in (
        "initialization_seed",
        "initial_state_sha256",
        "budgets",
        "maximum_epochs",
        "patience_checks",
        "validation_batch_size",
        "validation_selection",
        "optimizer",
        "schedule",
    ):
        if configuration.get(key) != selected.get(key):
            raise ValueError(f"Spec 0039 retry configuration differs: {key}")
    minimum_epochs = configuration.get("minimum_completed_epochs")
    if minimum_epochs not in {None, MINIMUM_COMPLETED_EPOCHS}:
        raise ValueError("Spec 0039 retry minimum-epoch policy differs")
    runtime = configuration.get("runtime")
    selected_runtime = selected.get("runtime")
    if not isinstance(runtime, Mapping) or not isinstance(selected_runtime, Mapping):
        raise ValueError("Spec 0039 retry runtime configuration is malformed")
    ignored = {"branch_dispatch", "branch_execution"}
    if {key: value for key, value in runtime.items() if key not in ignored} != {
        key: value for key, value in selected_runtime.items() if key not in ignored
    }:
        raise ValueError("Spec 0039 retry runtime configuration differs")


def _build_staging(
    staging: Path,
    dataset_reference: str,
    assets: Mapping[str, bytes],
    configuration: Mapping[str, object],
    logical: Mapping[str, object],
) -> None:
    bundle = staging / "bundle"
    for name, payload in assets.items():
        target = bundle / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    _copy_source(ROOT, bundle / "src")
    files = _artifact_records(bundle, exclude={CONTRACT_NAME, METADATA_NAME})
    contract: dict[str, object] = {
        "schema_version": "spec0039.tissue_training_input.v1",
        "dataset_reference": dataset_reference,
        "dataset_actor": dataset_reference.split("/", maxsplit=1)[0],
        "visibility": "private",
        "scope": "tissue_development_training_no_sealed_test",
        "spec_sha256": _sha256(ROOT / SPEC_PATH),
        "model": {
            "source": "src/eqvae/models/supervised.py",
            "sha256": _sha256(ROOT / "src/eqvae/models/supervised.py"),
            "initialization_seed": INITIALIZATION_SEED,
            "classes": list(LABELS),
            "initial_state_file_sha256": _sha256(bundle / INITIAL_STATE_NAME),
        },
        "kernel_sources": list(KERNEL_SOURCES),
        "physical_sources": logical["physical_sources"],
        "logical_assets": logical["logical_assets"],
        "configuration_sha256": _sha256(bundle / CONFIG_NAME),
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
    *, expected_actor: str | None = None, sealed_source_snapshot: bool = False
) -> dict[str, object]:
    """Reject data, source, launch or sealed-test surface drift."""
    output = ROOT / DEFAULT_ROOT
    bundle = output / "bundle"
    contract = _read_object(bundle / CONTRACT_NAME)
    dataset_reference = cast("str", contract.get("dataset_reference"))
    actor = dataset_reference.split("/", maxsplit=1)[0]
    if not _contract_identity_matches(
        contract,
        actor=actor,
        expected_actor=expected_actor,
        bundle=bundle,
        sealed_source_snapshot=sealed_source_snapshot,
    ):
        raise ValueError("Spec 0039 input contract identity differs")
    files = _records(contract.get("files"))
    if files != _artifact_records(bundle, exclude={CONTRACT_NAME, METADATA_NAME}):
        raise ValueError("Spec 0039 input bundle bytes differ")
    if any(
        ("test" in name or "sealed" in name) and not name.startswith("src/")
        for name in files
    ):
        raise ValueError("Spec 0039 development input exposes a sealed-test surface")
    if sealed_source_snapshot:
        _validate_sealed_source_snapshot(contract, files)
    else:
        assets, configuration, logical = _derive_assets(ROOT)
        for name, payload in assets.items():
            if (bundle / name).read_bytes() != payload:
                raise ValueError(f"Spec 0039 derived asset differs: {name}")
        if _read_object(bundle / CONFIG_NAME) != configuration:
            raise ValueError("Spec 0039 training configuration differs")
        if contract.get("physical_sources") != logical["physical_sources"] or (
            contract.get("logical_assets") != logical["logical_assets"]
        ):
            raise ValueError("Spec 0039 logical data identity differs")
        _validate_source_snapshot(bundle, files)
    _validate_upload(output / "upload", bundle, dataset_reference)
    _validate_kernel(
        output / "kernel",
        bundle=bundle,
        dataset_reference=dataset_reference,
        sealed_source_snapshot=sealed_source_snapshot,
    )
    return contract


def _contract_identity_matches(
    contract: Mapping[str, object],
    *,
    actor: str,
    expected_actor: str | None,
    bundle: Path,
    sealed_source_snapshot: bool,
) -> bool:
    model = contract.get("model")
    common = (
        contract.get("schema_version") == "spec0039.tissue_training_input.v1"
        and contract.get("dataset_reference") == _dataset_reference(actor)
        and contract.get("dataset_actor") == actor
        and (expected_actor is None or actor == expected_actor)
        and contract.get("visibility") == "private"
        and contract.get("scope") == "tissue_development_training_no_sealed_test"
        and contract.get("kernel_sources") == list(KERNEL_SOURCES)
        and contract.get("configuration_sha256") == _sha256(bundle / CONFIG_NAME)
        and _read_object(bundle / METADATA_NAME)
        == _dataset_metadata(_dataset_reference(actor))
        and isinstance(model, dict)
        and model.get("source") == "src/eqvae/models/supervised.py"
        and model.get("initialization_seed") == INITIALIZATION_SEED
        and model.get("classes") == list(LABELS)
        and model.get("initial_state_file_sha256")
        == _sha256(bundle / INITIAL_STATE_NAME)
    )
    return common and (
        sealed_source_snapshot
        or (
            contract.get("spec_sha256") == _sha256(ROOT / SPEC_PATH)
            and model.get("sha256") == _sha256(ROOT / "src/eqvae/models/supervised.py")
        )
    )


def _derive_assets(
    root: Path,
) -> tuple[dict[str, bytes], dict[str, object], dict[str, object]]:
    """Stage all and only fixed development logical members."""
    manifest_root = root / MANIFEST_ROOT
    catalog_bytes = (manifest_root / CATALOG).read_bytes()
    catalog_rows = _read_csv_bytes(catalog_bytes, CATALOG_HEADER)
    physical_sources = _physical_sources(catalog_rows)
    names = [f"tissue/tissue_train_{budget:04d}_per_class.csv" for budget in BUDGETS]
    if any("test" in name for name in names):
        raise ValueError("Spec 0039 may not stage a tissue test asset")
    logical_assets: dict[str, object] = {}
    assets: dict[str, bytes] = {CATALOG.as_posix(): catalog_bytes}
    train_identity_sets: list[set[tuple[str, ...]]] = []
    for budget, name in zip(BUDGETS, names, strict=True):
        payload = (manifest_root / name).read_bytes()
        rows = _read_csv_bytes(payload, TISSUE_HEADER)
        _validate_train_rows(rows, budget)
        identities = {_identity(row) for row in rows}
        if train_identity_sets and not train_identity_sets[-1] <= identities:
            raise ValueError("Spec 0039 tissue budgets are not nested")
        train_identity_sets.append(identities)
        assets[name] = payload
        logical_assets[name] = _record_bytes(payload, len(rows))
    validation_name = "tissue/tissue_validation.csv"
    validation_bytes = (manifest_root / validation_name).read_bytes()
    validation_rows = _read_csv_bytes(validation_bytes, TISSUE_HEADER)
    _validate_validation_rows(validation_rows)
    validation_subsets = _validation_subsets(validation_rows)
    for per_class, rows in validation_subsets.items():
        name = f"tissue/tissue_validation_{per_class:04d}_per_class.csv"
        payload = _tissue_csv_bytes(rows)
        assets[name] = payload
        logical_assets[name] = _record_bytes(payload, len(rows))
    for budget, train_identities in zip(BUDGETS, train_identity_sets, strict=True):
        validation_selection = validation_subsets[VALIDATION_PER_CLASS[budget]]
        validation_identities = {_identity(row) for row in validation_selection}
        if train_identities & validation_identities:
            raise ValueError("Spec 0039 train/validation logical identities overlap")
        train_wsis = {
            row["wsi_id"]
            for row in _read_csv_bytes(
                assets[f"tissue/tissue_train_{budget:04d}_per_class.csv"],
                TISSUE_HEADER,
            )
        }
        validation_wsis = {row["wsi_id"] for row in validation_selection}
        if train_wsis & validation_wsis:
            raise ValueError("Spec 0039 train/validation WSI identities overlap")
    audit_bytes = (manifest_root / MANIFEST_AUDIT).read_bytes()
    audit = cast("dict[str, object]", json.loads(audit_bytes))
    _validate_audit(audit, logical_assets, validation_bytes)
    assets[MANIFEST_AUDIT.as_posix()] = audit_bytes
    initial_state = _initial_state_bytes()
    assets[INITIAL_STATE_NAME] = initial_state
    configuration = _training_configuration(
        logical_assets=logical_assets,
        initial_state_sha256=_state_sha256(initial_state),
    )
    assets[CONFIG_NAME] = _json_bytes(configuration)
    return (
        assets,
        configuration,
        {
            "physical_sources": physical_sources,
            "logical_assets": logical_assets,
        },
    )


def _validate_train_rows(rows: Sequence[Mapping[str, str]], per_class: int) -> None:
    expected = 3 * per_class
    if len(rows) != expected or {row["split"] for row in rows} != {"train"}:
        raise ValueError("Spec 0039 training rows have an unexpected split/size")
    if Counter(row["tissue_label"] for row in rows) != Counter({
        label: per_class for label in LABELS
    }):
        raise ValueError("Spec 0039 training class balance differs")
    if [int(row["dataset_row"]) for row in rows] != list(range(expected)):
        raise ValueError("Spec 0039 train dataset rows are not contiguous")


def _validate_validation_rows(rows: Sequence[Mapping[str, str]]) -> None:
    if len(rows) != 31_339 or {row["split"] for row in rows} != {"validation"}:
        raise ValueError("Spec 0039 requires the full natural validation split")
    if [int(row["dataset_row"]) for row in rows] != list(range(len(rows))):
        raise ValueError("Spec 0039 validation dataset rows are not contiguous")
    if set(row["tissue_label"] for row in rows) != set(LABELS):
        raise ValueError("Spec 0039 validation labels differ")


def _validate_audit(
    audit: Mapping[str, object],
    logical_assets: Mapping[str, object],
    validation_bytes: bytes,
) -> None:
    tissue_files = audit.get("tissue_files")
    if not isinstance(tissue_files, dict):
        raise ValueError("Spec 0039 supervised manifest audit is malformed")
    for name, record in logical_assets.items():
        if not Path(name).name.startswith("tissue_train_"):
            continue
        audit_name = Path(name).name
        audit_record = tissue_files.get(audit_name)
        if (
            not isinstance(audit_record, dict)
            or audit_record.get("sha256") != record["sha256"]
        ):
            raise ValueError(f"Spec 0039 audit differs for {audit_name}")
    validation = tissue_files.get("tissue_validation.csv")
    if not isinstance(validation, dict) or validation.get("sha256") != _sha256_bytes(
        validation_bytes,
    ):
        raise ValueError("Spec 0039 validation-source audit differs")


def _validation_subsets(
    rows: Sequence[Mapping[str, str]],
) -> dict[int, list[dict[str, str]]]:
    """Create nested WSI-stratified held-out validation prefixes."""
    priority = _validation_priority(rows)
    result: dict[int, list[dict[str, str]]] = {}
    for per_class in sorted(set(VALIDATION_PER_CLASS.values())):
        selected: list[tuple[Mapping[str, str], int]] = []
        for label in LABELS:
            rows_for_label = priority[label]
            if len(rows_for_label) < per_class:
                raise ValueError("Spec 0039 validation class lacks the selected prefix")
            selected.extend(rows_for_label[:per_class])
        selected.sort(key=lambda item: _order_key(item[0]))
        result[per_class] = [
            {
                **row,
                "dataset_row": str(index),
                "selection_rank": str(rank),
            }
            for index, (row, rank) in enumerate(selected)
        ]
    return result


def _validation_priority(
    rows: Sequence[Mapping[str, str]],
) -> dict[str, list[tuple[Mapping[str, str], int]]]:
    grouped: dict[tuple[str, int], list[Mapping[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["tissue_label"], int(row["wsi_id"])), []).append(row)
    priority: dict[str, list[tuple[Mapping[str, str], int]]] = {}
    for label_index, label in enumerate(LABELS):
        ranked: list[tuple[int, Fraction, int, int, Mapping[str, str]]] = []
        for (group_label, wsi_id), group in sorted(grouped.items()):
            if group_label != label:
                continue
            ordered = sorted(group, key=_order_key)
            generator = np.random.Generator(
                np.random.PCG64(
                    np.random.SeedSequence([
                        VALIDATION_SELECTION_SEED,
                        label_index,
                        wsi_id,
                    ]),
                ),
            )
            for within_rank, position in enumerate(generator.permutation(len(ordered))):
                ranked.append((
                    0 if within_rank == 0 else 1,
                    Fraction(within_rank, len(ordered)),
                    wsi_id,
                    within_rank,
                    ordered[int(position)],
                ))
        ordered_ranked = sorted(ranked, key=lambda item: item[:4])
        priority[label] = [
            (row, rank) for rank, (*_ignored, row) in enumerate(ordered_ranked)
        ]
    if set(priority) != set(LABELS):
        raise ValueError("Spec 0039 validation priority labels differ")
    return priority


def _physical_sources(rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    if (
        len(rows) != 12
        or tuple(dict.fromkeys(row["kaggle_source"] for row in rows)) != KERNEL_SOURCES
    ):
        raise ValueError("Spec 0039 physical source locators differ")
    if {row["model_name"] for row in rows} != {"normal_vae", "so2_vae"}:
        raise ValueError("Spec 0039 catalog representations differ")
    return [dict(row) for row in rows]


def _initial_state_bytes() -> bytes:
    first, second = make_paired_models(TissueClassifier, seed=INITIALIZATION_SEED)
    if first.state_dict().keys() != second.state_dict().keys() or any(
        not torch.equal(first.state_dict()[name], second.state_dict()[name])
        for name in first.state_dict()
    ):
        raise ValueError("Spec 0039 paired initialization differs")
    destination = io.BytesIO()
    torch.save(
        {
            "schema_version": "spec0039.tissue_initial_state.v1",
            "seed": INITIALIZATION_SEED,
            "state_dict": first.state_dict(),
        },
        destination,
    )
    return destination.getvalue()


def _training_configuration(
    *, logical_assets: Mapping[str, object], initial_state_sha256: str
) -> dict[str, object]:
    schedules = {
        str(budget): {
            **geometry,
            "validation_per_class": VALIDATION_PER_CLASS[budget],
            "validation_rows": 3 * VALIDATION_PER_CLASS[budget],
            "warmup_updates": (geometry["steps_per_epoch"] + 9) // 10,
            "total_updates": geometry["steps_per_epoch"] * MAXIMUM_EPOCHS,
        }
        for budget, geometry in BUDGETS.items()
    }
    return {
        "schema_version": "spec0039.tissue_training_config.v1",
        "labels": list(LABELS),
        "initialization_seed": INITIALIZATION_SEED,
        "initial_state_sha256": initial_state_sha256,
        "budgets": schedules,
        "maximum_epochs": MAXIMUM_EPOCHS,
        "patience_checks": PATIENCE_CHECKS,
        "minimum_completed_epochs": MINIMUM_COMPLETED_EPOCHS,
        "validation_batch_size": VALIDATION_BATCH_SIZE,
        "validation_selection": {
            "seed": VALIDATION_SELECTION_SEED,
            "source": "held_out_tissue_validation",
            "nested_per_class": sorted(set(VALIDATION_PER_CLASS.values())),
        },
        "optimizer": {
            "name": "AdamW",
            "peak_lr": PEAK_LR,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "fused": True,
            "matrix_weight_decay": WEIGHT_DECAY,
            "vector_weight_decay": 0.0,
        },
        "schedule": {"warmup_fraction": 0.1, "minimum_ratio": 0.01},
        "runtime": {
            "torch": "2.14.0+cu130",
            "branch_dispatch": "subprocess_per_gpu",
            "branch_execution": "independent",
            "compile_mode": "max-autotune",
            "compiled_autograd": True,
            "fullgraph": True,
            "dynamic": False,
            "amp_dtype": "float16",
            "channels_last": True,
            "cudnn_benchmark": True,
            "deterministic": False,
            "tf32": False,
            "scaler": "torch.amp.GradScaler(cuda)_default",
            "calibration_batches": 3,
            "max_scaler_backoffs": 3,
        },
        "logical_assets": logical_assets,
    }


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
        raise ValueError("Spec 0039 source allow-list differs")
    for name, source in expected.items():
        if (bundle / name).read_bytes() != source.read_bytes():
            raise ValueError(f"Spec 0039 source differs: {name}")


def _validate_sealed_source_snapshot(
    contract: Mapping[str, object], files: Mapping[str, object]
) -> None:
    source_names = {name for name in files if name.startswith("src/")}
    model = contract.get("model")
    model_record = files.get("src/eqvae/models/supervised.py")
    if (
        not source_names
        or any(
            not name.startswith("src/eqvae/") or not name.endswith(".py")
            for name in source_names
        )
        or not isinstance(model, dict)
        or not isinstance(model_record, dict)
        or model.get("sha256") != model_record.get("sha256")
    ):
        raise ValueError("Spec 0039 sealed source/model binding differs")


def _validate_kernel(
    kernel: Path, *, bundle: Path, dataset_reference: str, sealed_source_snapshot: bool
) -> None:
    if {path.name for path in kernel.iterdir()} != {"kernel-metadata.json", "run.py"}:
        raise ValueError("Spec 0039 kernel allow-list differs")
    if _read_object(kernel / "kernel-metadata.json") != _kernel_metadata(
        dataset_reference
    ):
        raise ValueError("Spec 0039 kernel metadata differs")
    launcher = (kernel / "run.py").read_bytes()
    if sealed_source_snapshot:
        source = launcher.decode("utf-8")
        compile(source, str(TEMPLATE_PATH), "exec")
        required = (
            "SPEC0039_TISSUE_TRAINING_READY = True",
            f'INPUT_CONTRACT_SHA256 = "{_sha256(bundle / CONTRACT_NAME)}"',
            f'INPUT_DATASET_REFERENCE = "{dataset_reference}"',
        )
        if not all(part in source for part in required):
            raise ValueError("Spec 0039 sealed launcher binding differs")
    elif launcher != _render(ROOT, _sha256(bundle / CONTRACT_NAME), dataset_reference):
        raise ValueError("Spec 0039 rendered launcher differs")


def _render(root: Path, contract_hash: str, dataset_reference: str) -> bytes:
    template = (root / TEMPLATE_PATH).read_text(encoding="utf-8")
    placeholders = {
        "input_contract_sha256": contract_hash,
        "input_dataset_reference": dataset_reference,
    }
    if any(template.count(f"${name}") != 1 for name in placeholders):
        raise ValueError("Spec 0039 template placeholder count differs")
    rendered = Template(template).substitute(placeholders).encode("utf-8")
    if len(rendered) >= 1_000_000:
        raise ValueError("Spec 0039 launcher exceeds Kaggle's 1 MB limit")
    compile(rendered, str(root / TEMPLATE_PATH), "exec")
    return rendered


def _dataset_reference(actor: str) -> str:
    if not actor or "/" in actor or actor.strip() != actor:
        raise ValueError("Kaggle actor must be one nonempty owner component")
    return f"{actor}/{DATASET_SLUG}"


def _dataset_metadata(dataset_reference: str) -> dict[str, object]:
    return {
        "id": dataset_reference,
        "title": "eqvae tissue label-efficiency training inputs",
        "licenses": [{"name": "other"}],
    }


def _kernel_metadata(dataset_reference: str) -> dict[str, object]:
    return {
        "id": KERNEL_ID,
        "title": "eqvae tissue label-efficiency training",
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
        raise ValueError("Spec 0039 upload envelope differs")
    if _read_object(upload / METADATA_NAME) != _dataset_metadata(dataset_reference):
        raise ValueError("Spec 0039 upload metadata differs")
    with zipfile.ZipFile(upload / "bundle.zip") as archive:
        expected = {
            path.relative_to(bundle).as_posix()
            for path in bundle.rglob("*")
            if path.is_file() and path.name != METADATA_NAME
        }
        if set(archive.namelist()) != expected:
            raise ValueError("Spec 0039 upload archive allow-list differs")
        for name in expected:
            if archive.read(name) != (bundle / name).read_bytes():
                raise ValueError(f"Spec 0039 upload member differs: {name}")


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
        raise ValueError("Spec 0039 CSV header differs")
    return list(reader)


def _identity(row: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(
        row[name]
        for name in ("atlas_row_index", "wsi_id", "x", "y", "part", "file_index")
    )


def _order_key(row: Mapping[str, str]) -> tuple[int, int, int]:
    return int(row["wsi_id"]), int(row["y"]), int(row["x"])


def _tissue_csv_bytes(rows: Sequence[Mapping[str, str]]) -> bytes:
    destination = io.StringIO(newline="")
    writer = csv.DictWriter(destination, fieldnames=TISSUE_HEADER, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return destination.getvalue().encode("utf-8")


def _record_bytes(payload: bytes, rows: int) -> dict[str, object]:
    return {"bytes": len(payload), "sha256": _sha256_bytes(payload), "row_count": rows}


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


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _state_sha256(payload: bytes) -> str:
    loaded = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    state = loaded.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError("Spec 0039 serialized initial state is malformed")
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("build", "validate", "build-retry", "validate-retry")
    )
    parser.add_argument("--actor")
    parser.add_argument("--sealed-source-snapshot", action="store_true")
    parser.add_argument("--input-bundle")
    parser.add_argument("--output-root")
    args = parser.parse_args(argv)
    actor = cast("str | None", args.actor)
    if args.action in {"build", "build-retry", "validate-retry"} and actor is None:
        parser.error(
            f"{args.action} requires --actor with the authenticated Kaggle username"
        )
    if args.action in {"build-retry", "validate-retry"} and args.output_root is None:
        parser.error(f"{args.action} requires --output-root")
    if args.action == "build-retry" and args.input_bundle is None:
        parser.error("build-retry requires --input-bundle")
    if args.action == "build":
        contract = build(actor=cast("str", actor))
    elif args.action == "validate":
        contract = validate(
            expected_actor=actor, sealed_source_snapshot=args.sealed_source_snapshot
        )
    elif args.action == "build-retry":
        contract = build_retry(
            actor=cast("str", actor),
            input_bundle=Path(cast("str", args.input_bundle)),
            output_root=Path(cast("str", args.output_root)),
        )
    else:
        contract = validate_retry(
            output_root=Path(cast("str", args.output_root)),
            expected_actor=cast("str", actor),
        )
    print(json.dumps(contract, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
