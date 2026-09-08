# Copyright 2026 HiperMaximus
# ruff: noqa: C901, DOC201, DOC501, E501, EM101, EM102, PLC2701, PLR0912, PLR0913, PLR0914, PLR0915, PLR0916, PLR2004, PLW0717, T201, TC003, TRY003, TRY301
"""Build the immutable Spec 0036 development-training Kaggle package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import struct
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from string import Template
from typing import Final, cast

import numpy as np
import torch
from torch import Tensor

from eqvae.cli.build_ubc_supervised_calibration_inputs import stage_upload_envelope
from eqvae.data.supervised_latents import (
    CATALOG_HEADER,
    WSI_BAG_HEADER,
    WSI_INSTANCE_HEADER,
)
from eqvae.kaggle_resources import KaggleResourceRef
from eqvae.models.local_attention_candidates import (
    use_whole_bag_fixed25_attention,
)
from eqvae.models.local_global_mil import (
    CLASS_ORDER,
    EXPECTED_PARAMETER_COUNT,
    LOCAL_MAX_DEGREE,
    LOCAL_RADIUS,
    NEIGHBOR_PADDING_INDEX,
    PATCH_SIZE_PIXELS,
    RADIAL_CODEBOOK,
    RADIAL_PADDING_CODE,
    LocalGlobalMILClassifier,
    _graph_identity_sha256,  # pyright: ignore[reportPrivateUsage]
)
from eqvae.training.mil_training import (
    EARLY_STOPPING_START_EPOCH,
    EARLY_STOPPING_START_UPDATE,
    MAXIMUM_EPOCHS,
    PATIENCE_CHECKS,
    PATIENCE_EPOCHS,
    TOTAL_UPDATES,
    WARMUP_EPOCHS,
    WARMUP_START_LEARNING_RATE,
    WARMUP_START_RATIO,
    WARMUP_UPDATES,
    WEIGHT_DECAY,
    LoadedBranchCheckpoint,
    load_branch_checkpoint,
)

ROOT: Final = Path.cwd()
DEFAULT_ROOT: Final = Path("runs/local/ubc_ocean_mil_training_v3")
SPEC_PATH: Final = Path("docs/specs/0036-local-global-mil-training.md")
DEV_ROOT: Final = Path(
    "runs/local/ubc_ocean_full_foreground_manifests/development",
)
DEV_CONTRACT_SHA256: Final = (
    "6aa25a3f56db62b903d3451e154c47dc0039233f88179c7bdab03b7a48319fe0"
)
DEVELOPMENT_FILE_SHA256: Final = {
    "dataset.json": DEV_CONTRACT_SHA256,
    "physical_parts.csv": (
        "9303120aa99ab105eafdb14868bd8e1a1f785b643f149f103bb60beeabf8d92e"
    ),
    "wsi_cancer_train_bags.csv": (
        "c8b320054a17a68b1e4ba21dfcfd00354d4d27e002793b1bb0bc91ee1ebe209a"
    ),
    "wsi_cancer_train_instances.csv": (
        "dcc2d2343ce0b86588fffdb4564796178c7193ff4b11eea74dc3bc6d23be992e"
    ),
    "wsi_cancer_validation_bags.csv": (
        "120308e96614ae5fd76f1839dcc6253fee20dde3041715cd44da47f59bb45a30"
    ),
    "wsi_cancer_validation_instances.csv": (
        "cbb3a3ae30634090668b741800a67fde88804d6e329f42be44f752f8519124c0"
    ),
}
DEVELOPMENT_FILES: Final = tuple(DEVELOPMENT_FILE_SHA256)
DATASET_SLUG: Final = "eqvae-local-global-mil-training-inputs-v3"
RESUME_CONTRACT_NAME: Final = "mil_training_resume.json"
KERNEL_ID: Final = "maximusshtefan/eqvae-local-global-mil-training"
CONTRACT_NAME: Final = "mil_training_input.json"
METADATA_NAME: Final = "dataset-metadata.json"
INITIAL_STATE_NAME: Final = "initial_state.pt"
TEMPLATE_PATH: Final = Path(
    "kaggle/kernels/ubc_ocean_mil_training/run_template.py",
)
MODEL_PATH: Final = Path("src/eqvae/models/local_global_mil.py")
CANDIDATE_PATH: Final = Path("src/eqvae/models/local_attention_candidates.py")
INITIALIZATION_SEED: Final = 1701
TRAIN_WSI_COUNT: Final = 106
VALIDATION_WSI_COUNT: Final = 23
KERNEL_SOURCES: Final = (
    "maximusshtefan/eqvae-ubc-ocean-latent-run-01",
    "maximusshtefan/eqvae-ubc-ocean-latent-run-02",
    "maximusshtefan/eqvae-ubc-ocean-latent-run-03",
    "maximusshtefan/eqvae-ubc-ocean-latent-run-04",
    "maximusshtefan/eqvae-ubc-ocean-latent-run-05",
    "maximusshtefan/eqvae-ubc-ocean-cancer-latent-top-up",
    "maximusshtefan/eqvae-wsi45630-completion",
    "maximusshtefan/eqvae-full-foreground-01",
    "maximusshtefan/eqvae-full-foreground-02",
    "maximusshtefan/eqvae-full-foreground-03",
    "maximusshtefan/eqvae-full-foreground-04",
    "maximusshtefan/eqvae-full-foreground-05",
    "maximusshtefan/eqvae-full-foreground-06",
    "maximusshtefan/eqvae-full-foreground-07",
    "maximusshtefan/eqvae-full-foreground-08",
)
CLASS_COUNTS: Final = {"CC": 23, "EC": 24, "HGSC": 40, "LGSC": 11, "MC": 8}
CLASS_WEIGHTS: Final = {
    label: TRAIN_WSI_COUNT / (len(CLASS_COUNTS) * count)
    for label, count in CLASS_COUNTS.items()
}
_STATE_HASH_SCHEMA: Final = b"eqvae_spec0036_initial_state_v1"
_HASH_CHUNK_BYTES: Final = 1024 * 1024
SOURCE_EXCLUDES: Final = frozenset({
    Path("eqvae/artifacts/rotation_orbits.py"),
    Path("eqvae/cli/render_frozen_vae_rotation_orbits.py"),
})


def build(*, actor: str) -> dict[str, object]:
    """Create one new local package without mutating an existing artifact."""
    dataset_reference = _dataset_reference(actor)
    output = ROOT / DEFAULT_ROOT
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    staging = output.with_name(f".{output.name}.building")
    if staging.exists():
        raise FileExistsError(f"Stale staging directory exists: {staging}")

    development, physical_sources, graph_identities = _derive_development(ROOT)
    initial_state, initial_state_identity = _derive_initial_state()
    training_config = _training_config()
    try:
        _build_staging(
            staging=staging,
            development=development,
            physical_sources=physical_sources,
            graph_identities=graph_identities,
            initial_state=initial_state,
            initial_state_identity=initial_state_identity,
            training_config=training_config,
            dataset_reference=dataset_reference,
        )
        stage_upload_envelope(
            bundle_root=staging / "bundle",
            destination=staging / "upload",
        )
        staging.replace(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return validate(expected_actor=actor)


def _build_staging(
    *,
    staging: Path,
    development: Mapping[str, bytes],
    physical_sources: Sequence[Mapping[str, str]],
    graph_identities: Mapping[str, Mapping[str, str]],
    initial_state: bytes,
    initial_state_identity: str,
    training_config: Mapping[str, object],
    dataset_reference: str,
) -> None:
    bundle = staging / "bundle"
    development_root = bundle / "development"
    development_root.mkdir(parents=True)
    for name, payload in development.items():
        (development_root / name).write_bytes(payload)
    (bundle / INITIAL_STATE_NAME).write_bytes(initial_state)
    _copy_source(ROOT, bundle / "src")

    files = _artifact_records(bundle, exclude={CONTRACT_NAME, METADATA_NAME})
    contract: dict[str, object] = {
        "schema_version": "spec0036.mil_training_input.v1",
        "scope": "development_training_only_no_sealed_test",
        "visibility": "private",
        "dataset_reference": dataset_reference,
        "dataset_actor": dataset_reference.split("/", maxsplit=1)[0],
        "spec_sha256": _sha256(ROOT / SPEC_PATH),
        "development_contract_sha256": DEV_CONTRACT_SHA256,
        "development_files": {
            name: _file_record(ROOT / DEV_ROOT / name) for name in DEVELOPMENT_FILES
        },
        "physical_sources": list(physical_sources),
        "kernel_sources": list(KERNEL_SOURCES),
        "graph_identities": {
            split: dict(identities) for split, identities in graph_identities.items()
        },
        "training_config": dict(training_config),
        "training_config_sha256": _json_sha256(training_config),
        "model": {
            "source": MODEL_PATH.as_posix(),
            "sha256": _sha256(ROOT / MODEL_PATH),
            "parameter_count": EXPECTED_PARAMETER_COUNT,
        },
        "candidate": {
            "source": CANDIDATE_PATH.as_posix(),
            "sha256": _sha256(ROOT / CANDIDATE_PATH),
            "backend": "whole_bag_fixed25_inductor",
        },
        "initial_state": {
            "path": INITIAL_STATE_NAME,
            "seed": INITIALIZATION_SEED,
            "file_sha256": _bytes_sha256(initial_state),
            "state_sha256": initial_state_identity,
            "parameter_count": EXPECTED_PARAMETER_COUNT,
        },
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
    """Reject any input, identity, byte, metadata, or allow-list drift."""
    output = ROOT / DEFAULT_ROOT
    output_entries = list(output.iterdir())
    if {path.name for path in output_entries} != {"bundle", "kernel", "upload"} or any(
        not path.is_dir() or path.is_symlink() for path in output_entries
    ):
        raise ValueError("Spec 0036 output allow-list differs")
    bundle = output / "bundle"
    contract = _read_object(bundle / CONTRACT_NAME)
    dataset_reference = contract.get("dataset_reference")
    if not isinstance(dataset_reference, str):
        raise TypeError("Spec 0036 dataset reference must be a string")
    dataset_actor = KaggleResourceRef.parse(
        dataset_reference,
        allow_version=False,
    ).owner
    development, physical_sources, graph_identities = _derive_development(ROOT)
    initial_state, initial_state_identity = _derive_initial_state()
    training_config = _training_config()
    expected_development = {
        name: _file_record(ROOT / DEV_ROOT / name) for name in DEVELOPMENT_FILES
    }
    if (
        contract.get("schema_version") != "spec0036.mil_training_input.v1"
        or contract.get("scope") != "development_training_only_no_sealed_test"
        or contract.get("visibility") != "private"
        or dataset_reference != _dataset_reference(dataset_actor)
        or contract.get("dataset_actor") != dataset_actor
        or (expected_actor is not None and dataset_actor != expected_actor)
        or contract.get("spec_sha256") != _sha256(ROOT / SPEC_PATH)
        or contract.get("development_contract_sha256") != DEV_CONTRACT_SHA256
        or contract.get("development_files") != expected_development
        or contract.get("physical_sources") != physical_sources
        or contract.get("kernel_sources") != list(KERNEL_SOURCES)
        or contract.get("graph_identities") != graph_identities
        or contract.get("training_config") != training_config
        or contract.get("training_config_sha256") != _json_sha256(training_config)
        or contract.get("model")
        != {
            "source": MODEL_PATH.as_posix(),
            "sha256": _sha256(ROOT / MODEL_PATH),
            "parameter_count": EXPECTED_PARAMETER_COUNT,
        }
        or contract.get("candidate")
        != {
            "source": CANDIDATE_PATH.as_posix(),
            "sha256": _sha256(ROOT / CANDIDATE_PATH),
            "backend": "whole_bag_fixed25_inductor",
        }
        or contract.get("initial_state")
        != {
            "path": INITIAL_STATE_NAME,
            "seed": INITIALIZATION_SEED,
            "file_sha256": _bytes_sha256(initial_state),
            "state_sha256": initial_state_identity,
            "parameter_count": EXPECTED_PARAMETER_COUNT,
        }
        or _read_object(bundle / METADATA_NAME) != _dataset_metadata(dataset_reference)
    ):
        raise ValueError("Spec 0036 input contract identity differs")

    files = _records(contract.get("files"))
    if files != _artifact_records(bundle, exclude={CONTRACT_NAME, METADATA_NAME}):
        raise ValueError("Spec 0036 input bundle bytes differ")
    _validate_bundle_tree(bundle, files)
    _validate_development_stage(bundle / "development", development)
    if (bundle / INITIAL_STATE_NAME).read_bytes() != initial_state:
        raise ValueError("Spec 0036 initial model state differs")
    _validate_source_snapshot(bundle, files)
    _validate_upload(output / "upload", bundle, dataset_reference)
    _validate_kernel(output / "kernel", dataset_reference, bundle)
    return contract


def _derive_development(
    root: Path,
) -> tuple[
    dict[str, bytes],
    list[dict[str, str]],
    dict[str, dict[str, str]],
]:
    development_root = root / DEV_ROOT
    development: dict[str, bytes] = {}
    for name, expected_sha256 in DEVELOPMENT_FILE_SHA256.items():
        path = development_root / name
        if _require_hash(path, expected_sha256) != expected_sha256:
            raise AssertionError("unreachable development hash mismatch")
        development[name] = path.read_bytes()
    if {path.name for path in development_root.iterdir()} != set(DEVELOPMENT_FILES):
        raise ValueError("Spec 0025 development source allow-list differs")
    dataset_contract = _read_object(development_root / "dataset.json")
    if (
        dataset_contract.get("schema_version")
        != "spec0025.full_foreground_development.v1"
        or dataset_contract.get("test_release") != "not_authorized"
    ):
        raise ValueError("Spec 0025 development contract scope differs")
    physical_sources = _physical_sources(development_root / "physical_parts.csv")
    graph_identities = {
        split: _split_graph_identities(development_root, split=split)
        for split in ("train", "validation")
    }
    if (
        len(graph_identities["train"]) != TRAIN_WSI_COUNT
        or len(
            graph_identities["validation"],
        )
        != VALIDATION_WSI_COUNT
    ):
        raise ValueError("Spec 0036 graph-identity split counts differ")
    return development, physical_sources, graph_identities


def _physical_sources(path: Path) -> list[dict[str, str]]:
    rows = _read_csv(path, expected_header=CATALOG_HEADER)
    if len(rows) != 30:
        raise ValueError("Spec 0036 physical catalog must contain 30 rows")
    part_models: Counter[int] = Counter()
    for row in rows:
        part_models[int(row["part"])] += 1
        KaggleResourceRef.parse(row["kaggle_source"], allow_version=False)
        for field in ("binary_sha256", "sidecar_sha256"):
            _validate_sha256(row[field], field=field)
    if set(part_models.values()) != {2} or set(part_models) != {
        1,
        2,
        3,
        4,
        5,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
    }:
        raise ValueError("Spec 0036 physical part/model pairing differs")
    sources = tuple(dict.fromkeys(row["kaggle_source"] for row in rows))
    if sources != KERNEL_SOURCES:
        raise ValueError("Spec 0036 physical producer order differs")
    return rows


def _split_graph_identities(root: Path, *, split: str) -> dict[str, str]:
    bags_path = root / f"wsi_cancer_{split}_bags.csv"
    instances_path = root / f"wsi_cancer_{split}_instances.csv"
    bags = _read_csv(bags_path, expected_header=WSI_BAG_HEADER)
    identities: dict[str, str] = {}
    expected_instance_row = 0
    with instances_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != WSI_INSTANCE_HEADER:
            raise ValueError(f"Spec 0036 {split} instance header differs")
        for expected_bag_row, bag in enumerate(bags):
            if (
                int(bag["bag_row"]) != expected_bag_row
                or bag["split"] != split
                or int(bag["instance_start"]) != expected_instance_row
            ):
                raise ValueError(f"Spec 0036 {split} bag ordering differs")
            coordinates: list[tuple[int, int]] = []
            instance_count = int(bag["instance_count"])
            for _ in range(instance_count):
                row = next(reader, None)
                if row is None:
                    raise ValueError(f"Spec 0036 {split} instances ended early")
                if (
                    int(row["instance_row"]) != expected_instance_row
                    or row["split"] != split
                    or row["wsi_id"] != bag["wsi_id"]
                    or row["diagnosis_label"] != bag["diagnosis_label"]
                    or row["diagnosis_index"] != bag["diagnosis_index"]
                ):
                    raise ValueError(f"Spec 0036 {split} instance identity differs")
                coordinates.append((int(row["x"]), int(row["y"])))
                expected_instance_row += 1
            graph_identity = _graph_identity_from_coordinates(
                wsi_id=int(bag["wsi_id"]),
                coordinates=coordinates,
            )
            key = bag["wsi_id"]
            if key in identities:
                raise ValueError(f"Spec 0036 {split} repeats WSI {key}")
            identities[key] = graph_identity
        if next(reader, None) is not None:
            raise ValueError(f"Spec 0036 {split} instances contain extra rows")
    return identities


def _graph_identity_from_coordinates(
    *,
    wsi_id: int,
    coordinates: Sequence[tuple[int, int]],
) -> str:
    if not coordinates:
        raise ValueError("Spec 0036 graph may not be empty")
    pixels = np.asarray(coordinates, dtype=np.int64)
    if bool(np.remainder(pixels, PATCH_SIZE_PIXELS).any()):
        raise ValueError("Spec 0036 graph coordinates are off the patch lattice")
    lattice = pixels // PATCH_SIZE_PIXELS
    lattice_coordinates = tuple(
        (int(pair[0]), int(pair[1])) for pair in lattice.tolist()
    )
    if len(set(lattice_coordinates)) != len(lattice_coordinates):
        raise ValueError("Spec 0036 graph coordinates are not unique")
    node_count = len(lattice_coordinates)
    lookup = {coordinate: index for index, coordinate in enumerate(lattice_coordinates)}
    neighbor_index = np.full(
        (node_count, LOCAL_MAX_DEGREE),
        NEIGHBOR_PADDING_INDEX,
        dtype=np.int64,
    )
    neighbor_valid = np.zeros((node_count, LOCAL_MAX_DEGREE), dtype=np.bool_)
    radial_code = np.full(
        (node_count, LOCAL_MAX_DEGREE),
        RADIAL_PADDING_CODE,
        dtype=np.uint8,
    )
    slots = np.zeros(node_count, dtype=np.int64)
    radius_to_code = {radius: code for code, radius in enumerate(RADIAL_CODEBOOK)}
    for delta_x in range(-LOCAL_RADIUS, LOCAL_RADIUS + 1):
        for delta_y in range(-LOCAL_RADIUS, LOCAL_RADIUS + 1):
            keys = np.fromiter(
                (
                    lookup.get((grid_x + delta_x, grid_y + delta_y), -1)
                    for grid_x, grid_y in lattice_coordinates
                ),
                dtype=np.int64,
                count=node_count,
            )
            rows = np.flatnonzero(keys >= 0)
            row_slots = slots[rows]
            neighbor_index[rows, row_slots] = keys[rows]
            neighbor_valid[rows, row_slots] = True
            squared_radius = delta_x * delta_x + delta_y * delta_y
            radial_code[rows, row_slots] = radius_to_code[squared_radius]
            slots[rows] += 1
    return _graph_identity_sha256(
        wsi_id=wsi_id,
        coordinates=lattice_coordinates,
        neighbor_index=torch.from_numpy(neighbor_index),  # pyright: ignore[reportUnknownMemberType]
        neighbor_valid=torch.from_numpy(neighbor_valid),  # pyright: ignore[reportUnknownMemberType]
        radial_code=torch.from_numpy(radial_code),  # pyright: ignore[reportUnknownMemberType]
    )


def _derive_initial_state() -> tuple[bytes, str]:
    torch.manual_seed(INITIALIZATION_SEED)  # pyright: ignore[reportUnknownMemberType]
    model = LocalGlobalMILClassifier()
    before = _state_dict_sha256(model.state_dict())
    use_whole_bag_fixed25_attention(model)
    after = _state_dict_sha256(model.state_dict())
    if before != after:
        raise ValueError("Whole-bag attention installation changed initial state")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise ValueError("Spec 0036 initial model parameter count differs")
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getvalue(), after


def _state_dict_sha256(state: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256(_STATE_HASH_SCHEMA)
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        metadata = json.dumps(
            {"dtype": str(tensor.dtype), "name": name, "shape": list(tensor.shape)},
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        payload = tensor.numpy().tobytes(order="C")
        digest.update(struct.pack("<Q", len(metadata)))
        digest.update(metadata)
        digest.update(struct.pack("<Q", len(payload)))
        digest.update(payload)
    return digest.hexdigest()


def _training_config() -> dict[str, object]:
    return {
        "seed": INITIALIZATION_SEED,
        "bootstrap_seed": 3601,
        "bootstrap_replicates": 10_000,
        "confidence": 0.95,
        "peak_lr": 2e-4,
        "minimum_lr_ratio": 0.01,
        "maximum_epochs": MAXIMUM_EPOCHS,
        "steps_per_epoch": TRAIN_WSI_COUNT,
        "total_updates": TOTAL_UPDATES,
        "warmup_epochs": WARMUP_EPOCHS,
        "warmup_updates": WARMUP_UPDATES,
        "warmup_start_ratio": WARMUP_START_RATIO,
        "warmup_start_lr": WARMUP_START_LEARNING_RATE,
        "validation_boundaries": [53, 106],
        "early_stopping_start_epoch": EARLY_STOPPING_START_EPOCH,
        "early_stopping_start_update": EARLY_STOPPING_START_UPDATE,
        "patience_epochs": PATIENCE_EPOCHS,
        "patience_checks": PATIENCE_CHECKS,
        "max_overflow_backoffs": 3,
        "class_order": list(CLASS_ORDER),
        "class_counts": dict(CLASS_COUNTS),
        "class_weights": dict(CLASS_WEIGHTS),
        "optimizer": {
            "name": "AdamW",
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "matrix_weight_decay": WEIGHT_DECAY,
            "other_weight_decay": 0.0,
            "no_decay_roles": [
                "bias",
                "normalization",
                "relative_attention_bias_or_offset",
                "global_cls_reg_tokens",
                "local_null_keys",
            ],
            "fused": True,
            "capturable": False,
        },
        "grad_scaler": {
            "api": "torch.amp.GradScaler",
            "initial_scale": 65_536.0,
            "growth_factor": 2.0,
            "backoff_factor": 0.5,
            "growth_interval": 2_000,
        },
        "torch_version": "2.14.0",
        "cuda_version": "13.0",
        "cuda_wheel": "cu130",
        "torch_index_url": "https://download.pytorch.org/whl/cu130",
        "compile": {
            "backend": "inductor",
            "fullgraph": True,
            "mode": "max-autotune-no-cudagraphs",
            "recompile_limit": 3,
            "graph_counts": "checkpoint_and_final_telemetry",
            "dynamic_axes": "N_only_permissive_cached_specialization",
        },
        "execution": {
            "batch_size": 1,
            "complete_bags": True,
            "branches": {"normal_vae": 0, "so2_vae": 1},
            "processes": 2,
            "processes_per_branch": 1,
            "ddp": False,
            "cross_branch_gating": False,
        },
    }


def _copy_source(root: Path, destination: Path) -> None:
    sources = _runtime_source_paths(root)
    if not sources:
        raise FileNotFoundError("No eqvae Python sources found")
    for source in sources:
        if "__pycache__" in source.parts:
            continue
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"Source snapshot refuses non-regular file: {source}")
        target = destination / source.relative_to(root / "src")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _validate_source_snapshot(bundle: Path, files: Mapping[str, object]) -> None:
    expected = {
        f"src/{path.relative_to(ROOT / 'src').as_posix()}": path
        for path in _runtime_source_paths(ROOT)
    }
    if {name for name in files if name.startswith("src/")} != set(expected):
        raise ValueError("Spec 0036 source snapshot allow-list differs")
    for name, source in expected.items():
        if source.is_symlink() or (bundle / name).read_bytes() != source.read_bytes():
            raise ValueError(f"Spec 0036 source snapshot differs: {name}")


def _runtime_source_paths(root: Path) -> list[Path]:
    sources = [
        path
        for path in sorted((root / "src/eqvae").rglob("*.py"))
        if "__pycache__" not in path.parts
        and path.relative_to(root / "src") not in SOURCE_EXCLUDES
    ]
    if not sources:
        raise FileNotFoundError("No Spec 0036 runtime Python sources found")
    return sources


def _validate_development_stage(
    development_root: Path,
    expected: Mapping[str, bytes],
) -> None:
    entries = list(development_root.iterdir())
    if {path.name for path in entries} != set(DEVELOPMENT_FILES) or any(
        not path.is_file() or path.is_symlink() for path in entries
    ):
        raise ValueError("Spec 0036 staged development allow-list differs")
    if any("test" in path.name or "sealed" in path.name for path in entries):
        raise ValueError("Spec 0036 staged a sealed-test logical file")
    for name, payload in expected.items():
        if (development_root / name).read_bytes() != payload:
            raise ValueError(f"Spec 0036 staged development bytes differ: {name}")


def _validate_bundle_tree(bundle: Path, files: Mapping[str, object]) -> None:
    expected_files = {*files, CONTRACT_NAME, METADATA_NAME}
    expected_directories: set[str] = set()
    for name in expected_files:
        parent = Path(name).parent
        while parent != Path():
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    for path in bundle.rglob("*"):
        relative = path.relative_to(bundle).as_posix()
        if path.is_symlink():
            raise ValueError(f"Spec 0036 bundle refuses symbolic links: {relative}")
        if path.is_file():
            observed_files.add(relative)
        elif path.is_dir():
            observed_directories.add(relative)
        else:
            raise ValueError(f"Spec 0036 bundle refuses special file: {relative}")
    if observed_files != expected_files or observed_directories != expected_directories:
        raise ValueError("Spec 0036 bundle tree allow-list differs")


def _validate_kernel(kernel: Path, dataset_reference: str, bundle: Path) -> None:
    entries = list(kernel.iterdir())
    if {path.name for path in entries} != {"kernel-metadata.json", "run.py"} or any(
        not path.is_file() or path.is_symlink() for path in entries
    ):
        raise ValueError("Spec 0036 kernel allow-list differs")
    if _read_object(kernel / "kernel-metadata.json") != _kernel_metadata(
        dataset_reference,
    ):
        raise ValueError("Spec 0036 kernel metadata differs")
    expected = _render(ROOT, _sha256(bundle / CONTRACT_NAME), dataset_reference)
    run_path = kernel / "run.py"
    if run_path.read_bytes() != expected:
        raise ValueError("Spec 0036 rendered launcher differs")
    if run_path.stat().st_size >= 1_000_000:
        raise ValueError("Spec 0036 launcher exceeds Kaggle's 1 MB limit")


def _render(
    root: Path,
    contract_hash: str,
    dataset_reference: str,
    *,
    resume_dataset_reference: str = "",
    resume_contract_sha256: str = "",
) -> bytes:
    template = (root / TEMPLATE_PATH).read_text(encoding="utf-8")
    placeholders = {
        "input_contract_sha256": contract_hash,
        "input_dataset_reference": dataset_reference,
        "resume_dataset_reference": resume_dataset_reference,
        "resume_contract_sha256": resume_contract_sha256,
    }
    if any(template.count(f"${name}") != 1 for name in placeholders):
        raise ValueError("Spec 0036 template placeholder count differs")
    rendered = Template(template).substitute(placeholders).encode()
    if len(rendered) >= 1_000_000:
        raise ValueError("Spec 0036 launcher exceeds Kaggle's 1 MB limit")
    compile(rendered, str(root / TEMPLATE_PATH), "exec")
    return rendered


def _dataset_reference(actor: str) -> str:
    return KaggleResourceRef(owner=actor, slug=DATASET_SLUG).canonical_id


def _dataset_metadata(dataset_reference: str) -> dict[str, object]:
    return {
        "id": dataset_reference,
        "title": "eqvae local global mil training inputs",
        "licenses": [{"name": "other"}],
    }


def _kernel_metadata(
    dataset_reference: str,
    resume_dataset_reference: str | None = None,
) -> dict[str, object]:
    return {
        "id": KERNEL_ID,
        "title": "eqvae local global mil training",
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [dataset_reference]
        + ([resume_dataset_reference] if resume_dataset_reference else []),
        "competition_sources": [],
        "kernel_sources": list(KERNEL_SOURCES),
        "model_sources": [],
    }


def _validate_upload(upload: Path, bundle: Path, dataset_reference: str) -> None:
    entries = list(upload.iterdir())
    if {path.name for path in entries} != {"bundle.zip", METADATA_NAME} or any(
        not path.is_file() or path.is_symlink() for path in entries
    ):
        raise ValueError("Spec 0036 upload envelope differs")
    if _read_object(upload / METADATA_NAME) != _dataset_metadata(dataset_reference):
        raise ValueError("Spec 0036 upload metadata differs")
    with zipfile.ZipFile(upload / "bundle.zip") as archive:
        expected = {
            path.relative_to(bundle).as_posix()
            for path in bundle.rglob("*")
            if path.is_file() and path.name != METADATA_NAME
        }
        if set(archive.namelist()) != expected:
            raise ValueError("Spec 0036 upload archive allow-list differs")
        for name in expected:
            info = archive.getinfo(name)
            if info.compress_type != zipfile.ZIP_STORED:
                raise ValueError(f"Spec 0036 upload member is compressed: {name}")
            if archive.read(name) != (bundle / name).read_bytes():
                raise ValueError(f"Spec 0036 upload member differs: {name}")


def _artifact_records(root: Path, *, exclude: set[str]) -> dict[str, object]:
    return {
        path.relative_to(root).as_posix(): _file_record(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in exclude
    }


def _file_record(path: Path) -> dict[str, int | str]:
    return {"bytes": path.stat().st_size, "sha256": _sha256(path)}


def _records(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("Artifact records must be an object")
    return cast("dict[str, object]", value)


def _read_csv(path: Path, *, expected_header: Sequence[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(expected_header):
            raise ValueError(f"CSV header differs: {path}")
        return list(reader)


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


def _json_sha256(value: object) -> str:
    return _bytes_sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode(),
    )


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hash(path: Path, expected: str) -> str:
    observed = _sha256(path)
    if observed != expected:
        raise ValueError(f"Authority hash differs for {path}: {observed}")
    return observed


def _validate_prior_output_authority(
    *,
    output_root: Path,
    launch_receipt_path: Path,
) -> tuple[
    dict[str, object],
    dict[str, object],
    str,
    dict[str, str],
    dict[str, object],
    Path,
]:
    if output_root.is_symlink() or launch_receipt_path.is_symlink():
        raise ValueError("Prior Spec 0036 authority paths may not be symlinks")
    launch = _read_object(launch_receipt_path)
    kernel_reference = launch.get("kernel_reference")
    kernel_id = launch.get("kernel_id")
    launch_actor = launch.get("actor")
    accepted_version = launch.get("accepted_version")
    if (
        launch.get("schema_version") != "eqvae.kaggle_kernel_launch.v1"
        or not isinstance(kernel_reference, str)
        or not isinstance(kernel_id, str)
        or not isinstance(launch_actor, str)
        or isinstance(accepted_version, bool)
        or not isinstance(accepted_version, int)
        or accepted_version < 1
        or kernel_reference != f"{kernel_id}/{accepted_version}"
        or kernel_id != f"{launch_actor}/eqvae-local-global-mil-training"
    ):
        raise ValueError("Prior Spec 0036 launch receipt identity differs")
    output_root = output_root.resolve()
    payload_candidates = (
        output_root,
        output_root / "spec0036_mil_training",
    )
    payload_roots = [
        candidate
        for candidate in payload_candidates
        if (candidate / "run_contract.json").is_file()
    ]
    if len(payload_roots) != 1 or payload_roots[0].is_symlink():
        raise ValueError("Prior Spec 0036 output payload root is ambiguous")
    payload_root = payload_roots[0]
    run_contract = _read_object(payload_root / "run_contract.json")
    if run_contract.get(
        "schema_version",
    ) != "spec0036.run_contract.v1" or run_contract.get("kernel_sources") != list(
        KERNEL_SOURCES,
    ):
        raise ValueError("Prior Spec 0036 run contract differs")
    hashes = _records(run_contract.get("contract_hashes"))
    if not hashes or any(not isinstance(value, str) for value in hashes.values()):
        raise TypeError("Prior Spec 0036 contract hashes must be text")
    input_reference = run_contract.get("input_dataset_reference")
    if not isinstance(input_reference, str):
        raise TypeError("Prior input dataset reference must be text")
    locators = launch.get("source_locators")
    if not isinstance(locators, dict):
        raise TypeError("Prior Spec 0036 launch source locators are missing")
    locator_records = cast("dict[str, object]", locators)
    datasets = locator_records.get("dataset_sources")
    dataset_records = (
        cast("list[object]", datasets) if isinstance(datasets, list) else []
    )
    if (
        not isinstance(datasets, list)
        or not 1 <= len(dataset_records) <= 2
        or dataset_records[0] != input_reference
        or locator_records.get("kernel_sources") != list(KERNEL_SOURCES)
        or locator_records.get("competition_sources") != []
        or locator_records.get("model_sources") != []
    ):
        raise ValueError("Prior Spec 0036 launch sources differ")

    output_receipt_path = output_root / "kaggle_output_receipt.json"
    output_receipt = _read_object(output_receipt_path)
    if (
        output_receipt.get("schema_version") != "eqvae.kaggle_download.v1"
        or output_receipt.get("resource_kind") != "kernel"
        or output_receipt.get("resource_reference") != kernel_reference
        or output_receipt.get("resource_owner") != launch_actor
        or output_receipt.get("resource_slug") != "eqvae-local-global-mil-training"
        or output_receipt.get("resource_version") != accepted_version
    ):
        raise ValueError("Prior Spec 0036 output receipt identity differs")
    receipt_files = _records(output_receipt.get("files"))
    observed_files: dict[str, object] = {}
    for path in sorted(output_root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Prior Spec 0036 output may not contain symlinks")
        if path.is_file() and path != output_receipt_path:
            observed_files[path.relative_to(output_root).as_posix()] = _file_record(
                path,
            )
    if receipt_files != observed_files:
        raise ValueError("Prior Spec 0036 output receipt does not bind every byte")
    provenance = {
        "kernel_reference": kernel_reference,
        "launch_receipt_sha256": _sha256(launch_receipt_path),
        "output_receipt_sha256": _sha256(output_receipt_path),
    }
    return (
        run_contract,
        hashes,
        input_reference,
        provenance,
        receipt_files,
        payload_root,
    )


def build_resume(
    *,
    actor: str,
    output_root: Path,
    launch_receipt_path: Path,
) -> dict[str, object]:
    """Stage authenticated resumable branches plus any carried terminal peer."""
    output_root = output_root.resolve()
    _, hashes, input_reference, provenance, prior_files, payload_root = (
        _validate_prior_output_authority(
            output_root=output_root,
            launch_receipt_path=launch_receipt_path,
        )
    )
    loaded: dict[str, LoadedBranchCheckpoint] = {}
    carried: dict[str, Path] = {}
    for branch in ("normal_vae", "so2_vae"):
        source = payload_root / branch / "checkpoints"
        if any(path.is_symlink() for path in source.rglob("*")):
            raise ValueError("Resume checkpoint tree may not contain symlinks")
        if not (source / "latest").is_file():
            summary_path = payload_root / branch / "final_summary.json"
            summary = _read_object(summary_path)
            if summary.get("branch") != branch or summary.get("status") != "failed":
                raise ValueError(
                    f"Omitted Spec 0036 branch lacks terminal failure evidence: {branch}",
                )
            carried[branch] = summary_path
            continue
        latest = load_branch_checkpoint(
            source,
            slot="latest",
            expected_branch_name=branch,
            expected_contract_hashes=cast("Mapping[str, str]", hashes),
        )
        best = load_branch_checkpoint(
            source,
            slot="best",
            expected_branch_name=branch,
            expected_contract_hashes=cast("Mapping[str, str]", hashes),
        )
        best_state_value = latest.payload["best_selection"]
        if not isinstance(best_state_value, dict):
            raise TypeError("Prior Spec 0036 best selection must be an object")
        best_state = cast("dict[str, object]", best_state_value)
        if best.payload["committed_update"] != best_state.get("best_boundary"):
            raise ValueError(f"Prior Spec 0036 best checkpoint differs: {branch}")
        stopped = best_state.get("stopped") is True
        if latest.payload["committed_update"] == TOTAL_UPDATES or stopped:
            final = load_branch_checkpoint(
                source,
                slot="final",
                expected_branch_name=branch,
                expected_contract_hashes=cast("Mapping[str, str]", hashes),
            )
            if final.payload["committed_update"] != latest.payload["committed_update"]:
                raise ValueError(f"Prior Spec 0036 final checkpoint differs: {branch}")
        loaded[branch] = latest
    if not loaded:
        raise ValueError("Spec 0036 resume requires at least one resumable branch")
    resume_identity = _json_sha256({
        "actor": actor,
        "branches": {
            branch: checkpoint.manifest_sha256 for branch, checkpoint in loaded.items()
        },
        "carried": {
            branch: _sha256(summary_path) for branch, summary_path in carried.items()
        },
        "provenance": provenance,
    })[:16]
    resume_slug = f"eqvae-local-global-mil-resume-{resume_identity}"
    destination = ROOT / "runs/local/ubc_ocean_mil_training_resume" / resume_identity
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    staging = destination.with_name(f".{destination.name}.building")
    try:
        bundle = staging / "bundle"
        branches: dict[str, str] = {}
        for branch in loaded:
            source = payload_root / branch / "checkpoints"
            relative = f"branches/{branch}/checkpoints"
            shutil.copytree(source, bundle / relative)
            for copied in (bundle / relative).rglob("*"):
                if copied.is_file():
                    source_name = (
                        (source / copied.relative_to(bundle / relative))
                        .relative_to(output_root)
                        .as_posix()
                    )
                    if prior_files.get(source_name) != _file_record(copied):
                        raise ValueError(
                            f"Prior output receipt does not bind copied byte: {source_name}",
                        )
            branches[branch] = relative
        carried_branches: dict[str, str] = {}
        for branch, summary_path in carried.items():
            relative = f"carried/{branch}/final_summary.json"
            destination_summary = bundle / relative
            destination_summary.parent.mkdir(parents=True)
            shutil.copy2(summary_path, destination_summary)
            source_name = summary_path.relative_to(output_root).as_posix()
            if prior_files.get(source_name) != _file_record(destination_summary):
                raise ValueError(
                    f"Prior output receipt does not bind carried byte: {source_name}",
                )
            carried_branches[branch] = relative
        files = _artifact_records(bundle, exclude=set())
        reference = f"{actor}/{resume_slug}"
        contract = {
            "schema_version": "spec0036.mil_training_resume.v1",
            "dataset_reference": reference,
            "input_dataset_reference": input_reference,
            "input_contract_sha256": hashes["input_contract_sha256"],
            "contract_hashes": hashes,
            "kernel_sources": list(KERNEL_SOURCES),
            "prior_run": provenance,
            "branches": branches,
            "carried_branches": carried_branches,
            "files": files,
        }
        _write_json(bundle / RESUME_CONTRACT_NAME, contract)
        _write_json(bundle / METADATA_NAME, _dataset_metadata(reference))
        stage_upload_envelope(bundle_root=bundle, destination=staging / "upload")
        kernel = staging / "kernel"
        kernel.mkdir()
        _write_json(
            kernel / "kernel-metadata.json",
            _kernel_metadata(input_reference, reference),
        )
        (kernel / "run.py").write_bytes(
            _render(
                ROOT,
                cast("str", hashes["input_contract_sha256"]),
                input_reference,
                resume_dataset_reference=reference,
                resume_contract_sha256=_sha256(bundle / RESUME_CONTRACT_NAME),
            ),
        )
        staging.replace(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return validate_resume(resume_root=destination, expected_actor=actor)


def validate_resume(
    *,
    resume_root: Path,
    expected_actor: str | None = None,
) -> dict[str, object]:
    """Validate the immutable local two-branch resume package."""
    root = resume_root
    bundle = root / "bundle"
    contract = _read_object(bundle / RESUME_CONTRACT_NAME)
    reference = contract.get("dataset_reference")
    if not isinstance(reference, str):
        raise TypeError("Resume dataset reference must be text")
    actor = KaggleResourceRef.parse(reference, allow_version=False).owner
    input_reference = contract.get("input_dataset_reference")
    hashes = _records(contract.get("contract_hashes"))
    prior_run = _records(contract.get("prior_run"))
    branches = _records(contract.get("branches"))
    carried_branches = _records(contract.get("carried_branches"))
    branch_names = {"normal_vae", "so2_vae"}
    if isinstance(input_reference, str):
        KaggleResourceRef.parse(input_reference, allow_version=False)
    if (
        contract.get("schema_version") != "spec0036.mil_training_resume.v1"
        or not reference.startswith(f"{actor}/eqvae-local-global-mil-resume-")
        or (expected_actor is not None and actor != expected_actor)
        or contract.get("kernel_sources") != list(KERNEL_SOURCES)
        or not isinstance(input_reference, str)
        or hashes.get("input_contract_sha256") != contract.get("input_contract_sha256")
        or set(prior_run)
        != {
            "kernel_reference",
            "launch_receipt_sha256",
            "output_receipt_sha256",
        }
        or not branches
        or set(branches) | set(carried_branches) != branch_names
        or set(branches) & set(carried_branches)
        or branches != {branch: f"branches/{branch}/checkpoints" for branch in branches}
        or carried_branches
        != {
            branch: f"carried/{branch}/final_summary.json"
            for branch in carried_branches
        }
    ):
        raise ValueError("Spec 0036 resume identity differs")
    KaggleResourceRef.parse(cast("str", prior_run["kernel_reference"]))
    for field in ("launch_receipt_sha256", "output_receipt_sha256"):
        value = prior_run[field]
        if not isinstance(value, str):
            raise TypeError(f"Resume {field} must be text")
        _validate_sha256(value, field=field)
    files = _records(contract.get("files"))
    if files != _artifact_records(
        bundle,
        exclude={RESUME_CONTRACT_NAME, METADATA_NAME},
    ):
        raise ValueError("Spec 0036 resume bytes differ")
    _validate_resume_bundle_tree(bundle, files)
    for branch, relative in branches.items():
        if not isinstance(relative, str):
            raise TypeError("Spec 0036 resume branch path must be text")
        checkpoint_root = bundle / relative
        latest = load_branch_checkpoint(
            checkpoint_root,
            slot="latest",
            expected_branch_name=branch,
            expected_contract_hashes=cast("Mapping[str, str]", hashes),
        )
        best = load_branch_checkpoint(
            checkpoint_root,
            slot="best",
            expected_branch_name=branch,
            expected_contract_hashes=cast("Mapping[str, str]", hashes),
        )
        best_state_value = latest.payload["best_selection"]
        if not isinstance(best_state_value, dict):
            raise TypeError("Spec 0036 resume best selection must be an object")
        best_state = cast("dict[str, object]", best_state_value)
        if best.payload["committed_update"] != best_state.get("best_boundary"):
            raise ValueError(f"Spec 0036 resume best checkpoint differs: {branch}")
        if (
            latest.payload["committed_update"] == TOTAL_UPDATES
            or best_state.get("stopped") is True
        ):
            final = load_branch_checkpoint(
                checkpoint_root,
                slot="final",
                expected_branch_name=branch,
                expected_contract_hashes=cast("Mapping[str, str]", hashes),
            )
            if final.payload["committed_update"] != latest.payload["committed_update"]:
                raise ValueError(f"Spec 0036 resume final checkpoint differs: {branch}")
    for branch, relative in carried_branches.items():
        if not isinstance(relative, str):
            raise TypeError("Spec 0036 carried branch path must be text")
        summary = _read_object(bundle / relative)
        if summary.get("branch") != branch or summary.get("status") != "failed":
            raise ValueError(f"Spec 0036 carried branch evidence differs: {branch}")
    _validate_upload(root / "upload", bundle, reference)
    kernel_metadata = _read_object(root / "kernel/kernel-metadata.json")
    if kernel_metadata != _kernel_metadata(input_reference, reference):
        raise ValueError("Resume kernel does not bind exactly its resume dataset")
    run_source = (root / "kernel/run.py").read_text(encoding="utf-8")
    if (
        reference not in run_source
        or _sha256(bundle / RESUME_CONTRACT_NAME) not in run_source
    ):
        raise ValueError("Resume kernel lacks exact external binding")
    return contract


def _validate_resume_bundle_tree(bundle: Path, files: Mapping[str, object]) -> None:
    expected_files = {*files, RESUME_CONTRACT_NAME, METADATA_NAME}
    observed_files: set[str] = set()
    for path in bundle.rglob("*"):
        relative = path.relative_to(bundle).as_posix()
        if path.is_symlink():
            raise ValueError(f"Spec 0036 resume refuses symbolic links: {relative}")
        if path.is_file():
            observed_files.add(relative)
        elif not path.is_dir():
            raise ValueError(f"Spec 0036 resume refuses special file: {relative}")
    if observed_files != expected_files:
        raise ValueError("Spec 0036 resume bundle allow-list differs")


def _validate_sha256(value: str, *, field: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{field} must be lowercase SHA-256")


def main(argv: Sequence[str] | None = None) -> int:
    """Build or validate the local Spec 0036 package."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=("build", "validate", "build-resume", "validate-resume"),
    )
    parser.add_argument("--actor")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--launch-receipt", type=Path)
    args = parser.parse_args(argv)
    action = cast("str", args.action)
    actor = cast("str | None", args.actor)
    output_root = cast("Path | None", args.output_root)
    launch_receipt = cast("Path | None", args.launch_receipt)
    if action in {"build", "build-resume"} and actor is None:
        parser.error(f"{action} requires --actor")
    if action == "build-resume" and output_root is None:
        parser.error("build-resume requires --output-root")
    if action == "build-resume" and launch_receipt is None:
        parser.error("build-resume requires --launch-receipt")
    if action == "build":
        contract = build(actor=cast("str", actor))
    elif action == "validate":
        contract = validate(expected_actor=actor)
    elif action == "build-resume":
        contract = build_resume(
            actor=cast("str", actor),
            output_root=cast("Path", output_root),
            launch_receipt_path=cast("Path", launch_receipt),
        )
    else:
        if output_root is None:
            parser.error("validate-resume requires --output-root")
        contract = validate_resume(
            resume_root=output_root,
            expected_actor=actor,
        )
    print(json.dumps(contract, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
