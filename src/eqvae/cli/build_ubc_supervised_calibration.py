# Copyright 2026 HiperMaximus
# ruff: noqa: C901, DOC201, DOC501, EM101, EM102, PLR0913, PLR0914, PLR0915, PLR0916, TRY003
"""Build the two deliberately small Spec 0023 calibration Kaggle packages."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import math
import shutil
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING, Final, cast

import torch

from eqvae.data.supervised_latents import (
    CATALOG_HEADER,
    TISSUE_HEADER,
    WSI_BAG_HEADER,
    WSI_INSTANCE_HEADER,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

TEMPLATE_PATH: Final = Path(
    "kaggle/kernels/ubc_ocean_supervised_calibration/run_template.py",
)
SPEC_PATH: Final = Path("docs/specs/0023-matched-supervised-latent-evaluation.md")
DEFAULT_MANIFEST_ROOT: Final = Path("runs/local/ubc_ocean_supervised_manifests")
DEFAULT_MANIFEST_AUDIT: Final = (
    DEFAULT_MANIFEST_ROOT / "spec0023_supervised_manifest_audit.json"
)
DEFAULT_OUTPUT_ROOT: Final = Path("runs/local/ubc_ocean_supervised_calibration")
DEFAULT_INPUT_ROOT: Final = Path(
    "runs/local/ubc_ocean_supervised_calibration_inputs",
)
DEFAULT_INPUT_AUTHORITY_ROOT: Final = Path(
    "runs/local/ubc_ocean_supervised_calibration_authority",
)
PACKAGE_MODES: Final = (
    "sweep",
    "confirmation",
    "horizon",
    "width128",
    "class_specific",
    "class_specific_scale_fix",
)
INPUT_DATASET_SLUGS: Final = {
    "sweep": "maximusshtefan/eqvae-ubc-ocean-supcal-sweep-v2-inputs",
    "confirmation": ("maximusshtefan/eqvae-ubc-ocean-supcal-confirmation-inputs"),
    "horizon": "maximusshtefan/eqvae-ubc-ocean-mil-horizon-inputs",
    "width128": "maximusshtefan/eqvae-ubc-ocean-mil-width128-inputs",
    "class_specific": "maximusshtefan/eqvae-ubc-ocean-mil-class-attn-inputs",
    "class_specific_scale_fix": (
        "maximusshtefan/eqvae-ubc-ocean-mil-class-scale-inputs"
    ),
}
INPUT_RECEIPT_FILENAMES: Final = {
    "sweep": "sweep_v2_input_dataset_receipt.json",
    "confirmation": "confirmation_input_dataset_receipt.json",
    "horizon": "mil_horizon_input_dataset_receipt.json",
    "width128": "mil_width128_input_dataset_receipt.json",
    "class_specific": "mil_class_attn_input_dataset_receipt.json",
    "class_specific_scale_fix": "mil_class_scale_input_dataset_receipt.json",
}
INPUT_CONTRACT_NAME: Final = "spec0023_supervised_calibration_input_contract.json"
UPLOAD_FILES: Final = frozenset({
    "kernel-metadata.json",
    "run.py",
    "spec0023_supervised_calibration_config.json",
})
TASKS: Final = ("mil", "tissue")
REPRESENTATIONS: Final = ("normal_vae", "so2_vae")
SHA256_HEX_LENGTH: Final = 64
SWEEP_MAXIMUM_LR: Final = 3e-3
KAGGLE_SCRIPT_LIMIT_BYTES: Final = 1_000_000
MINIMUM_SELECTABLE_SWEEP_UPDATES: Final = 10
V1_SWEEP_AUDIT_SHA256: Final = (
    "654c8c244698717c1a628e3b395c9b414f9e1945cafc1fdd893ce1457d6246ef"
)
V1_MIL_INITIALIZATION_SHA256: Final = (
    "ae60948164a694c65c6b758e556d7199ccdf325fd5b35ea4fb6c3f7a3b815a1d"
)
V1_MIL_ORDER_SHA256: Final = (
    "8267113c14c486039933a1fb15a3ae35fb2e608ae6abd5ac9e758332a69940cb"
)
RECOVERY_LEGACY_UPDATES: Final = 11
RECOVERY_ATTEMPTS: Final = 12
RECOVERY_WSI_ID: Final = 65094
RECOVERY_DIAGNOSIS_INDEX: Final = 4
RECOVERY_INSTANCE_COUNT: Final = 5773
RECOVERY_SCALER: Final = 32_768.0
HORIZON_PEAK: Final = 2e-4
HORIZON_START_EPOCH: Final = 2
HORIZON_TARGET_EPOCH: Final = 5
HORIZON_STEPS_PER_EPOCH: Final = 106
HORIZON_RESUME_UPDATES: Final = HORIZON_START_EPOCH * HORIZON_STEPS_PER_EPOCH
CLASS_SPECIFIC_ATTENTION_DIM: Final = 128
CLASS_SPECIFIC_WARMUP_UPDATES: Final = 11
CLASS_SPECIFIC_UPDATES: Final = HORIZON_TARGET_EPOCH * HORIZON_STEPS_PER_EPOCH
CLASS_SPECIFIC_VALIDATION_CHECKS: Final = 2 * HORIZON_TARGET_EPOCH
CLASS_SPECIFIC_ATTENTION_MAPS: Final = 5
CLASS_SPECIFIC_FAILURE_SUCCESSFUL_UPDATES: Final = 465
CLASS_SPECIFIC_FAILED_ATTEMPT: Final = 466
CLASS_SPECIFIC_FAILURE_VALIDATION_CHECKS: Final = 8
SUPERVISED_WEIGHT_DECAY: Final = 1e-4
GRAD_SCALER_INIT_SCALE: Final = 32_768
GRAD_SCALER_GROWTH_INTERVAL: Final = 1_000_000
HORIZON_RESUME_ROOT: Final = Path(
    "runs/kaggle/ubc_ocean_supervised_calibration_confirmation_v1/"
    "mil/initial/epoch_2.0",
)
HORIZON_CONFIRMATION_AUDIT: Final = Path(
    "runs/kaggle/ubc_ocean_supervised_calibration_confirmation_v1/"
    "spec0023_supervised_calibration_audit.json",
)
HORIZON_CONFIRMATION_CONFIG: Final = Path(
    "runs/local/ubc_ocean_supervised_calibration/confirmation_v1/"
    "spec0023_supervised_calibration_config.json",
)
HORIZON_AUTHORITY_SHA256: Final = {
    "resume/manifest.json": (
        "825ea820ede5ff530b5d7d04d1c7f68948e63cc167a262b56f7146dacc678f75"
    ),
    "resume/progress.json": (
        "d2425f0f3c2f5fc1add68aa2a867686e7fbd7a44d4260300cf51264f89420072"
    ),
    "resume/paired_checkpoint.pt": (
        "d9488f01e466987fe0942b1ad29e70b58d56325efd1082a4cd946714318a21de"
    ),
    "resume/confirmation_v1_audit.json": (
        "ec7a3554e4864b328dd86485c30eef6d5a726fc014acf89012bca07aea529571"
    ),
    "resume/confirmation_v1_config.json": (
        "c03a31aa817c95fc90cc232dda5f97736c46a492e4cf7b5f6ec3ee2eab6cfa42"
    ),
}
WIDTH128_BASELINE_AUDIT_SHA256: Final = (
    "2edeb5d20471ebc559e3bb7f6f7cde4aeaa06723b98bdcfc28a7dca4039525fb"
)
CLASS_SPECIFIC_BASELINE_AUDIT_SHA256: Final = (
    "3e0933d87e334ba977dfaaca45bab99b70984cd88a1fa0f0097f9259bb77de23"
)
CLASS_SPECIFIC_BASELINE_CONFIG_SHA256: Final = (
    "b2de025ca3b647a2c9feecb640115bd60ea9624c76c4587de631ae166b691dde"
)
WIDTH128_INITIALIZATION_SHA256: Final = (
    "cfef73d525bf5e58501ca7aa9c93ed65a9a069f08ebf2fcefea63687b04dbcbf"
)
CLASS_SPECIFIC_INITIALIZATION_SHA256: Final = (
    "d6f916aafd6b544d62c2f481ff5f619e6485156dac08461b96546be8b5717c22"
)
CLASS_SPECIFIC_BASELINE_AUDIT: Final = Path(
    "runs/kaggle/ubc_ocean_mil_width128_v1/spec0023_supervised_calibration_audit.json",
)
CLASS_SPECIFIC_FAILURE_AUDIT_SHA256: Final = (
    "07fd04096c151fbd0be2b2b997e7f3c62dad9cdc84a0b421d6eec4a175ad9efc"
)
CLASS_SPECIFIC_FAILURE_CONFIG_SHA256: Final = (
    "09fcb3374bfb6c29b99ace77a55b327a6b523117c6d8ef7be085445cb0893d10"
)
CLASS_SPECIFIC_FAILURE_AUDIT: Final = Path(
    "runs/kaggle/ubc_ocean_mil_class_attention_v1/"
    "spec0023_supervised_calibration_audit.json",
)


def build_calibration_package(  # noqa: PLR0912
    *,
    repo_root: Path,
    manifest_root: Path,
    output_root: Path,
    package_mode: str,
    selection_audit_path: Path | None = None,
    sweep_audit_path: Path | None = None,
    sweep_config_path: Path | None = None,
    manifest_audit_path: Path | None = None,
    input_bundle_root: Path | None = None,
    input_receipt_path: Path | None = None,
    horizon_resume_root: Path = HORIZON_RESUME_ROOT,
    horizon_audit_path: Path = HORIZON_CONFIRMATION_AUDIT,
    horizon_config_path: Path = HORIZON_CONFIRMATION_CONFIG,
    class_specific_baseline_audit_path: Path = CLASS_SPECIFIC_BASELINE_AUDIT,
    class_specific_failure_audit_path: Path = CLASS_SPECIFIC_FAILURE_AUDIT,
    _skip_validation: bool = False,
) -> dict[str, object]:
    """Build one local-only upload directory for the locked calibration phase."""
    if package_mode not in PACKAGE_MODES:
        raise ValueError(f"Unknown Spec 0023 calibration package mode: {package_mode}")
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite {output_root}")

    canonical_assets, manifest_hashes, catalog_rows = _assets_for_mode(
        manifest_root=manifest_root,
        package_mode=package_mode,
        selection_audit_path=selection_audit_path,
        horizon_resume_root=horizon_resume_root,
        horizon_audit_path=horizon_audit_path,
        horizon_config_path=horizon_config_path,
    )
    manifest_audit_path = manifest_audit_path or (
        manifest_root / "spec0023_supervised_manifest_audit.json"
    )
    manifest_audit_sha256 = _validate_manifest_audit(
        manifest_audit_path,
        manifest_hashes,
        package_mode=package_mode,
    )
    source_records = _source_records(catalog_rows)
    class_specific_baseline = None
    if package_mode in {"class_specific", "class_specific_scale_fix"}:
        class_specific_baseline = _validate_width128_baseline_audit(
            class_specific_baseline_audit_path,
            manifest_hashes=manifest_hashes,
        )
    class_specific_failure = None
    if package_mode == "class_specific_scale_fix":
        if class_specific_baseline is None:
            raise RuntimeError("Class-specific predecessor validation was skipped")
        class_specific_failure = _validate_class_specific_failure_audit(
            class_specific_failure_audit_path,
            manifest_hashes=manifest_hashes,
            baseline=class_specific_baseline,
        )
    input_bundle_root = input_bundle_root or DEFAULT_INPUT_ROOT / package_mode
    input_receipt_path = input_receipt_path or (
        DEFAULT_INPUT_AUTHORITY_ROOT / INPUT_RECEIPT_FILENAMES[package_mode]
    )
    spec_sha256 = _sha256(repo_root / SPEC_PATH)
    input_receipt = _validate_input_receipt(
        repo_root=repo_root,
        bundle_root=input_bundle_root,
        receipt_path=input_receipt_path,
        package_mode=package_mode,
        expected_spec_sha256=spec_sha256,
        expected_manifest_audit_sha256=manifest_audit_sha256,
        require_current_source_tree=False,
    )
    input_contract = _read_object(input_bundle_root / INPUT_CONTRACT_NAME)
    config: dict[str, object] = {
        "schema_version": "spec0023.supervised_calibration_config.v1",
        "package_mode": package_mode,
        "spec_sha256": spec_sha256,
        "logical_manifest_sha256": manifest_hashes,
        "supervised_manifest_audit_sha256": manifest_audit_sha256,
        "input_source_tree_sha256": _source_tree_sha256(input_contract),
        "input_dataset_receipt": input_receipt,
        "kernel_sources": [record["kaggle_source"] for record in source_records],
        "source_records": source_records,
        "model_devices": {"normal_vae": 0, "so2_vae": 1},
        "optimizer": {
            "name": "AdamW",
            "matrix_weight_decay": 1e-4,
            "vector_weight_decay": 0.0,
            "grouping": "parameter_ndim_ge_2",
        },
        "precision": "FP16-autocast-with-GradScaler",
        "grad_scaler_init_scale": 32_768,
        "grad_scaler_growth_interval": 1_000_000,
        "checkpoint_chunk_size": None,
        "numerical_recovery": {
            "v1_sweep_audit_sha256": V1_SWEEP_AUDIT_SHA256,
            "legacy_initialization_sha256": V1_MIL_INITIALIZATION_SHA256,
            "legacy_order_sha256": V1_MIL_ORDER_SHA256,
            "legacy_failure_attempt": 12,
            "legacy_failure_wsi_id": 65094,
            "legacy_class_weight": 2.65,
            "replacement": "post_grad_scaler_unscale",
            "replacement_replay_updates": 12,
            "replay_state_reused_by_sweep": False,
        },
        "sweep": {
            "start_learning_rate": 1e-5,
            "stop_learning_rate": SWEEP_MAXIMUM_LR,
            "ewma_alpha": 0.1,
            "divergence_multiple": 4.0,
            "mil_updates": 106,
            "tissue_updates": 132,
        },
        "tasks": {
            "mil": {
                "initialization_seed": 1701,
                "steps_per_epoch": HORIZON_STEPS_PER_EPOCH,
                "class_weight_application": "post_grad_scaler_unscale",
                "train_instances": "wsi/wsi_cancer_train_instances.csv",
                "train_bags": "wsi/wsi_cancer_train_bags.csv",
            },
            "tissue": {
                "initialization_seed": 3407,
                "steps_per_epoch": 132,
                "batch_size": 128,
                "drop_last": True,
                "train_instances": "tissue/tissue_train_5671_per_class.csv",
            },
        },
        "output_allowlist": ["spec0023_supervised_calibration_audit.json"],
    }
    if package_mode == "confirmation":
        if sweep_audit_path is None or sweep_config_path is None:
            raise ValueError("Confirmation requires the downloaded sweep audit/config")
        selection = _read_selection_audit(canonical_assets["selection_audit.json"])
        sweep_contract = _read_object(
            DEFAULT_INPUT_ROOT / "sweep" / INPUT_CONTRACT_NAME,
        )
        sweep_spec_sha256 = _contract_spec_sha256(sweep_contract)
        sweep_input_receipt = _validate_input_receipt(
            repo_root=repo_root,
            bundle_root=DEFAULT_INPUT_ROOT / "sweep",
            receipt_path=(
                DEFAULT_INPUT_AUTHORITY_ROOT / INPUT_RECEIPT_FILENAMES["sweep"]
            ),
            package_mode="sweep",
            expected_spec_sha256=sweep_spec_sha256,
            expected_manifest_audit_sha256=manifest_audit_sha256,
            require_current_source_tree=False,
        )
        sweep_evidence = _validate_sweep_audit(
            sweep_audit_path,
            expected_sha256=cast("str", selection["sweep_audit_sha256"]),
            expected_manifest_hashes=manifest_hashes,
            sweep_config_path=sweep_config_path,
            expected_input_receipt=sweep_input_receipt,
            expected_provenance={
                "spec_sha256": sweep_spec_sha256,
                "supervised_manifest_audit_sha256": manifest_audit_sha256,
                "input_source_tree_sha256": _source_tree_sha256(sweep_contract),
                "source_records": source_records,
            },
        )
        _validate_selection_bounds(selection, sweep_evidence)
        config["selection_audit"] = {
            "path": "selection_audit.json",
            "sha256": manifest_hashes["selection_audit.json"],
            **selection,
            "verified_sweep": sweep_evidence,
        }
        config["tasks"] = cast("dict[str, object]", config["tasks"])
        mil = cast("dict[str, object]", config["tasks"]["mil"])
        tissue = cast("dict[str, object]", config["tasks"]["tissue"])
        mil.update({
            "validation_instances": "wsi/wsi_cancer_validation_instances.csv",
            "validation_bags": "wsi/wsi_cancer_validation_bags.csv",
        })
        tissue["validation_instances"] = "tissue/tissue_validation.csv"
        config["confirmation"] = {
            "epochs": 2,
            "validation_every": "half_epoch",
            "active_tasks": ["mil"],
            "frozen_tasks": ["tissue"],
        }
        config["output_allowlist"] = [
            "mil",
            "spec0023_supervised_calibration_audit.json",
        ]
    elif package_mode == "horizon":
        config.pop("numerical_recovery")
        config.pop("sweep")
        config["tasks"] = {
            "mil": {
                "initialization_seed": 1701,
                "steps_per_epoch": 106,
                "class_weight_application": "post_grad_scaler_unscale",
                "train_instances": "wsi/wsi_cancer_train_instances.csv",
                "train_bags": "wsi/wsi_cancer_train_bags.csv",
                "validation_instances": ("wsi/wsi_cancer_validation_instances.csv"),
                "validation_bags": "wsi/wsi_cancer_validation_bags.csv",
            },
        }
        config["horizon"] = {
            "source": "confirmation_v1_epoch_2.0",
            "resume_boundary": "resume",
            "resume_authority_sha256": HORIZON_AUTHORITY_SHA256,
            "peak": HORIZON_PEAK,
            "start_epoch": HORIZON_START_EPOCH,
            "target_epoch": HORIZON_TARGET_EPOCH,
            "continuation_updates": (
                (HORIZON_TARGET_EPOCH - HORIZON_START_EPOCH) * HORIZON_STEPS_PER_EPOCH
            ),
            "validation_every": "half_epoch",
            "warmup_repeated": False,
            "active_tasks": ["mil"],
            "frozen_tasks": ["tissue"],
        }
        config["output_allowlist"] = [
            "mil",
            "spec0023_supervised_calibration_audit.json",
        ]
    elif package_mode == "width128":
        config.pop("numerical_recovery")
        config.pop("sweep")
        config["tasks"] = {
            "mil": {
                "initialization_seed": 1701,
                "steps_per_epoch": HORIZON_STEPS_PER_EPOCH,
                "class_weight_application": "post_grad_scaler_unscale",
                "attention_dim": 128,
                "train_instances": "wsi/wsi_cancer_train_instances.csv",
                "train_bags": "wsi/wsi_cancer_train_bags.csv",
                "validation_instances": ("wsi/wsi_cancer_validation_instances.csv"),
                "validation_bags": "wsi/wsi_cancer_validation_bags.csv",
            },
        }
        config["architecture_diagnostic"] = {
            "kind": "gated_attention_scorer_width",
            "baseline_attention_dim": 64,
            "attention_dim": 128,
            "fresh_initialization": True,
            "peak": HORIZON_PEAK,
            "epochs": HORIZON_TARGET_EPOCH,
            "warmup_updates": 11,
            "validation_every": "half_epoch",
            "baseline_audit_sha256": WIDTH128_BASELINE_AUDIT_SHA256,
            "active_tasks": ["mil"],
            "frozen_tasks": ["tissue"],
        }
        config["output_allowlist"] = [
            "mil",
            "spec0023_supervised_calibration_audit.json",
        ]
    elif package_mode == "class_specific":
        if class_specific_baseline is None:
            raise RuntimeError("Class-specific predecessor validation was skipped")
        config.pop("numerical_recovery")
        config.pop("sweep")
        config["tasks"] = {
            "mil": {
                "initialization_seed": 1701,
                "steps_per_epoch": HORIZON_STEPS_PER_EPOCH,
                "class_weight_application": "post_grad_scaler_unscale",
                "attention_dim": 128,
                "class_specific_attention": True,
                "expected_initialization_sha256": (
                    CLASS_SPECIFIC_INITIALIZATION_SHA256
                ),
                "train_instances": "wsi/wsi_cancer_train_instances.csv",
                "train_bags": "wsi/wsi_cancer_train_bags.csv",
                "validation_instances": ("wsi/wsi_cancer_validation_instances.csv"),
                "validation_bags": "wsi/wsi_cancer_validation_bags.csv",
            },
        }
        config["architecture_diagnostic"] = {
            "kind": "class_specific_gated_attention",
            "attention_dim": 128,
            "attention_maps": CLASS_SPECIFIC_ATTENTION_MAPS,
            "fresh_initialization": True,
            "function_preserving_initialization": True,
            "peak": HORIZON_PEAK,
            "epochs": HORIZON_TARGET_EPOCH,
            "warmup_updates": 11,
            "validation_every": "half_epoch",
            "baseline": class_specific_baseline,
            "active_tasks": ["mil"],
            "frozen_tasks": ["tissue"],
        }
        config["output_allowlist"] = [
            "mil",
            "spec0023_supervised_calibration_audit.json",
        ]
    elif package_mode == "class_specific_scale_fix":
        if class_specific_baseline is None or class_specific_failure is None:
            raise RuntimeError("Class-specific correction authority was skipped")
        config.pop("numerical_recovery")
        config.pop("sweep")
        config["tasks"] = {
            "mil": {
                "initialization_seed": 1701,
                "steps_per_epoch": HORIZON_STEPS_PER_EPOCH,
                "class_weight_application": "post_grad_scaler_unscale",
                "attention_dim": 128,
                "class_specific_attention": True,
                "expected_initialization_sha256": (
                    CLASS_SPECIFIC_INITIALIZATION_SHA256
                ),
                "max_paired_scale_backoffs": 1,
                "train_instances": "wsi/wsi_cancer_train_instances.csv",
                "train_bags": "wsi/wsi_cancer_train_bags.csv",
                "validation_instances": ("wsi/wsi_cancer_validation_instances.csv"),
                "validation_bags": "wsi/wsi_cancer_validation_bags.csv",
            },
        }
        config["architecture_diagnostic"] = {
            "kind": "class_specific_gated_attention",
            "attention_dim": 128,
            "attention_maps": CLASS_SPECIFIC_ATTENTION_MAPS,
            "fresh_initialization": True,
            "function_preserving_initialization": True,
            "peak": HORIZON_PEAK,
            "epochs": HORIZON_TARGET_EPOCH,
            "warmup_updates": CLASS_SPECIFIC_WARMUP_UPDATES,
            "validation_every": "half_epoch",
            "baseline": class_specific_baseline,
            "correction": "single_synchronized_paired_scale_backoff",
            "active_tasks": ["mil"],
            "frozen_tasks": ["tissue"],
        }
        config["paired_scale_correction"] = {
            "failure_audit": class_specific_failure,
            "maximum_backoffs": 1,
            "initial_scale": GRAD_SCALER_INIT_SCALE,
            "backoff_factor": 0.5,
            "retry_same_loaded_bag": True,
            "retry_same_learning_rate": True,
            "advance_successful_cursor_on_discard": False,
            "raw_gradient_check": "after_unscale_before_class_weight",
            "weighted_gradient_check": "terminal_after_fp32_class_weight",
        }
        config["output_allowlist"] = [
            "mil",
            "spec0023_supervised_calibration_audit.json",
        ]

    config_bytes = _canonical_json(config)
    wrapper = _render_wrapper(repo_root, config_bytes)
    if len(wrapper) >= KAGGLE_SCRIPT_LIMIT_BYTES:
        raise ValueError("Calibration wrapper exceeds Kaggle's script limit")
    output_root.mkdir(parents=True)
    try:
        (output_root / "spec0023_supervised_calibration_config.json").write_bytes(
            config_bytes,
        )
        (output_root / "kernel-metadata.json").write_text(
            json.dumps(_metadata(package_mode, source_records), indent=2) + "\n",
            encoding="utf-8",
        )
        (output_root / "run.py").write_bytes(wrapper)
        if not _skip_validation:
            validate_calibration_package(
                repo_root=repo_root,
                manifest_root=manifest_root,
                output_root=output_root,
                package_mode=package_mode,
                selection_audit_path=selection_audit_path,
                sweep_audit_path=sweep_audit_path,
                sweep_config_path=sweep_config_path,
                manifest_audit_path=manifest_audit_path,
                input_bundle_root=input_bundle_root,
                input_receipt_path=input_receipt_path,
                horizon_resume_root=horizon_resume_root,
                horizon_audit_path=horizon_audit_path,
                horizon_config_path=horizon_config_path,
                class_specific_baseline_audit_path=(class_specific_baseline_audit_path),
                class_specific_failure_audit_path=(class_specific_failure_audit_path),
            )
    except BaseException:
        shutil.rmtree(output_root, ignore_errors=True)
        raise
    return config


def validate_calibration_package(
    *,
    repo_root: Path,
    manifest_root: Path,
    output_root: Path,
    package_mode: str,
    selection_audit_path: Path | None = None,
    sweep_audit_path: Path | None = None,
    sweep_config_path: Path | None = None,
    manifest_audit_path: Path | None = None,
    input_bundle_root: Path | None = None,
    input_receipt_path: Path | None = None,
    horizon_resume_root: Path = HORIZON_RESUME_ROOT,
    horizon_audit_path: Path = HORIZON_CONFIRMATION_AUDIT,
    horizon_config_path: Path = HORIZON_CONFIRMATION_CONFIG,
    class_specific_baseline_audit_path: Path = CLASS_SPECIFIC_BASELINE_AUDIT,
    class_specific_failure_audit_path: Path = CLASS_SPECIFIC_FAILURE_AUDIT,
) -> None:
    """Reject any drift or broadened input surface in one calibration package."""
    observed = {
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("*")
        if path.is_file()
    }
    if observed != set(UPLOAD_FILES):
        raise ValueError("Calibration upload allow-list differs")
    expected_root = output_root.parent / f".{output_root.name}.validation"
    if expected_root.exists():
        shutil.rmtree(expected_root)
    try:
        expected = build_calibration_package(
            repo_root=repo_root,
            manifest_root=manifest_root,
            output_root=expected_root,
            package_mode=package_mode,
            selection_audit_path=selection_audit_path,
            sweep_audit_path=sweep_audit_path,
            sweep_config_path=sweep_config_path,
            manifest_audit_path=manifest_audit_path,
            input_bundle_root=input_bundle_root,
            input_receipt_path=input_receipt_path,
            horizon_resume_root=horizon_resume_root,
            horizon_audit_path=horizon_audit_path,
            horizon_config_path=horizon_config_path,
            class_specific_baseline_audit_path=(class_specific_baseline_audit_path),
            class_specific_failure_audit_path=(class_specific_failure_audit_path),
            _skip_validation=True,
        )
        actual = _read_object(
            output_root / "spec0023_supervised_calibration_config.json",
        )
        if actual != expected:
            raise ValueError("Calibration config differs from its locked inputs")
        for name in UPLOAD_FILES:
            if (output_root / name).read_bytes() != (expected_root / name).read_bytes():
                raise ValueError(f"Calibration package file differs: {name}")
    finally:
        shutil.rmtree(expected_root, ignore_errors=True)


def _assets_for_mode(
    *,
    manifest_root: Path,
    package_mode: str,
    selection_audit_path: Path | None,
    horizon_resume_root: Path = HORIZON_RESUME_ROOT,
    horizon_audit_path: Path = HORIZON_CONFIRMATION_AUDIT,
    horizon_config_path: Path = HORIZON_CONFIRMATION_CONFIG,
) -> tuple[dict[str, bytes], dict[str, str], list[dict[str, str]]]:
    paths = {
        "physical_parts.csv": manifest_root / "physical_parts.csv",
        "wsi/wsi_cancer_train_instances.csv": (
            manifest_root / "wsi/wsi_cancer_train_instances.csv"
        ),
        "wsi/wsi_cancer_train_bags.csv": manifest_root
        / "wsi/wsi_cancer_train_bags.csv",
    }
    if package_mode in {"sweep", "confirmation"}:
        paths["tissue/tissue_train_5671_per_class.csv"] = (
            manifest_root / "tissue/tissue_train_5671_per_class.csv"
        )
    if package_mode in {
        "confirmation",
        "horizon",
        "width128",
        "class_specific",
        "class_specific_scale_fix",
    }:
        paths.update({
            "wsi/wsi_cancer_validation_instances.csv": (
                manifest_root / "wsi/wsi_cancer_validation_instances.csv"
            ),
            "wsi/wsi_cancer_validation_bags.csv": (
                manifest_root / "wsi/wsi_cancer_validation_bags.csv"
            ),
        })
    if package_mode == "confirmation":
        if selection_audit_path is None:
            raise ValueError("Confirmation requires the human selection audit")
        paths.update({
            "tissue/tissue_validation.csv": manifest_root
            / "tissue/tissue_validation.csv",
            "selection_audit.json": selection_audit_path,
        })
    elif package_mode == "horizon":
        paths.update({
            "resume/manifest.json": horizon_resume_root / "manifest.json",
            "resume/progress.json": horizon_resume_root / "progress.json",
            "resume/paired_checkpoint.pt": (
                horizon_resume_root / "paired_checkpoint.pt"
            ),
            "resume/confirmation_v1_audit.json": horizon_audit_path,
            "resume/confirmation_v1_config.json": horizon_config_path,
        })
    assets = {name: path.read_bytes() for name, path in paths.items()}
    catalog_rows = _read_csv_bytes(assets["physical_parts.csv"], CATALOG_HEADER)
    _validate_assets(assets, package_mode)
    return (
        assets,
        {name: hashlib.sha256(payload).hexdigest() for name, payload in assets.items()},
        catalog_rows,
    )


def _validate_assets(assets: Mapping[str, bytes], package_mode: str) -> None:
    common = {
        "physical_parts.csv",
        "wsi/wsi_cancer_train_instances.csv",
        "wsi/wsi_cancer_train_bags.csv",
    }
    expected_by_mode = {
        "sweep": {
            *common,
            "tissue/tissue_train_5671_per_class.csv",
        },
        "confirmation": {
            *common,
            "tissue/tissue_train_5671_per_class.csv",
            "wsi/wsi_cancer_validation_instances.csv",
            "wsi/wsi_cancer_validation_bags.csv",
            "tissue/tissue_validation.csv",
            "selection_audit.json",
        },
        "horizon": {
            *common,
            "wsi/wsi_cancer_validation_instances.csv",
            "wsi/wsi_cancer_validation_bags.csv",
            *HORIZON_AUTHORITY_SHA256,
        },
        "width128": {
            *common,
            "wsi/wsi_cancer_validation_instances.csv",
            "wsi/wsi_cancer_validation_bags.csv",
        },
        "class_specific": {
            *common,
            "wsi/wsi_cancer_validation_instances.csv",
            "wsi/wsi_cancer_validation_bags.csv",
        },
        "class_specific_scale_fix": {
            *common,
            "wsi/wsi_cancer_validation_instances.csv",
            "wsi/wsi_cancer_validation_bags.csv",
        },
    }
    if (
        package_mode not in expected_by_mode
        or set(assets) != expected_by_mode[package_mode]
    ):
        raise ValueError("Calibration assets differ from the locked mode surface")
    allowed = {
        *common,
        "tissue/tissue_train_5671_per_class.csv",
        "wsi/wsi_cancer_validation_instances.csv",
        "wsi/wsi_cancer_validation_bags.csv",
        "tissue/tissue_validation.csv",
        "selection_audit.json",
        *HORIZON_AUTHORITY_SHA256,
    }
    if set(assets) - allowed:
        raise ValueError("Calibration assets broadened beyond the locked manifests")
    _read_csv_bytes(assets["physical_parts.csv"], CATALOG_HEADER)
    train_instances = _read_csv_bytes(
        assets["wsi/wsi_cancer_train_instances.csv"],
        WSI_INSTANCE_HEADER,
    )
    train_bags = _read_csv_bytes(
        assets["wsi/wsi_cancer_train_bags.csv"],
        WSI_BAG_HEADER,
    )
    _validate_wsi_split(train_instances, train_bags, split="train", expected_bags=106)
    tissue_rows: list[dict[str, str]] = []
    if package_mode in {"sweep", "confirmation"}:
        tissue_rows = _read_csv_bytes(
            assets["tissue/tissue_train_5671_per_class.csv"],
            TISSUE_HEADER,
        )
        if len(tissue_rows) != 3 * 5671 or {row["split"] for row in tissue_rows} != {
            "train",
        }:
            raise ValueError(
                "The calibration tissue training CSV is not the locked 5671 pool",
            )
    if package_mode in {
        "confirmation",
        "horizon",
        "width128",
        "class_specific",
        "class_specific_scale_fix",
    }:
        validation_instances = _read_csv_bytes(
            assets["wsi/wsi_cancer_validation_instances.csv"],
            WSI_INSTANCE_HEADER,
        )
        validation_bags = _read_csv_bytes(
            assets["wsi/wsi_cancer_validation_bags.csv"],
            WSI_BAG_HEADER,
        )
        _validate_wsi_split(
            validation_instances,
            validation_bags,
            split="validation",
            expected_bags=23,
        )
        _validate_disjoint_rows(
            train_instances,
            validation_instances,
            label="WSI",
        )
    if package_mode == "confirmation":
        validation_tissue = _read_csv_bytes(
            assets["tissue/tissue_validation.csv"],
            TISSUE_HEADER,
        )
        if not validation_tissue or {row["split"] for row in validation_tissue} != {
            "validation",
        }:
            raise ValueError(
                "Calibration tissue validation rows must all be validation",
            )
        _validate_disjoint_rows(
            tissue_rows,
            validation_tissue,
            label="tissue",
        )
        _read_selection_audit(assets["selection_audit.json"])
    elif package_mode == "horizon":
        _validate_horizon_resume_assets(assets)
    elif "selection_audit.json" in assets:
        raise ValueError("Sweep package may not embed a peak-selection audit")


def _validate_horizon_resume_assets(assets: Mapping[str, bytes]) -> None:
    """Authenticate the one allowed confirmation-v1 epoch-2 resume boundary."""
    observed = {
        name: hashlib.sha256(assets[name]).hexdigest()
        for name in HORIZON_AUTHORITY_SHA256
    }
    if observed != HORIZON_AUTHORITY_SHA256:
        raise ValueError("MIL horizon confirmation-v1 authority differs")
    manifest_value = cast(
        "object",
        json.loads(assets["resume/manifest.json"]),
    )
    progress_value = cast(
        "object",
        json.loads(assets["resume/progress.json"]),
    )
    if not isinstance(manifest_value, dict) or not isinstance(progress_value, dict):
        raise TypeError("MIL horizon boundary metadata must be JSON objects")
    manifest = cast("dict[str, object]", manifest_value)
    progress = cast("dict[str, object]", progress_value)
    checkpoint_hash = HORIZON_AUTHORITY_SHA256["resume/paired_checkpoint.pt"]
    progress_hash = HORIZON_AUTHORITY_SHA256["resume/progress.json"]
    if manifest != {
        "paired_checkpoint.pt": checkpoint_hash,
        "progress.json": progress_hash,
    } or progress != {
        "checkpoint": "paired_checkpoint.pt",
        "checkpoint_sha256": checkpoint_hash,
        "epoch_fraction": 2.0,
        "schema_version": "spec0023.paired_boundary.v1",
    }:
        raise ValueError("MIL horizon boundary manifest/progress differs")
    checkpoint = cast(
        "dict[str, object]",
        torch.load(
            io.BytesIO(assets["resume/paired_checkpoint.pt"]),
            map_location="cpu",
            weights_only=False,
        ),
    )
    if (
        checkpoint.get("schema_version") != "spec0023.paired_supervised_checkpoint.v1"
        or checkpoint.get("task") != "mil_confirmation"
        or not _same_float(checkpoint.get("epoch_fraction"), HORIZON_START_EPOCH)
        or checkpoint.get("completed_epoch") != HORIZON_START_EPOCH
        or checkpoint.get("within_epoch_cursor") != HORIZON_STEPS_PER_EPOCH
        or checkpoint.get("successful_pair_count") != HORIZON_RESUME_UPDATES
        or checkpoint.get("schedule")
        != {
            "kind": "warmup_then_hold",
            "peak": HORIZON_PEAK,
            "steps_per_epoch": HORIZON_STEPS_PER_EPOCH,
        }
    ):
        raise ValueError("MIL horizon checkpoint identity differs")
    branches = cast("dict[str, dict[str, object]]", checkpoint.get("branches"))
    if set(branches) != set(REPRESENTATIONS):
        raise ValueError("MIL horizon checkpoint branch set differs")
    for branch in branches.values():
        optimizer = cast("dict[str, object]", branch.get("optimizer"))
        groups = cast("list[dict[str, object]]", optimizer.get("param_groups"))
        scaler = cast("dict[str, object]", branch.get("grad_scaler"))
        if (
            len(groups) != 1
            or not _same_float(groups[0].get("lr"), HORIZON_PEAK)
            or not _same_float(
                groups[0].get("weight_decay"),
                SUPERVISED_WEIGHT_DECAY,
            )
            or not _same_float(scaler.get("scale"), GRAD_SCALER_INIT_SCALE)
            or scaler.get("growth_interval") != GRAD_SCALER_GROWTH_INTERVAL
        ):
            raise ValueError("MIL horizon optimizer/scaler state differs")
    original_config = cast(
        "dict[str, object]",
        json.loads(assets["resume/confirmation_v1_config.json"]),
    )
    original_audit = cast(
        "dict[str, object]",
        json.loads(assets["resume/confirmation_v1_audit.json"]),
    )
    original_hashes = cast(
        "dict[str, str]",
        original_config.get("logical_manifest_sha256"),
    )
    learning_names = {
        "physical_parts.csv",
        "wsi/wsi_cancer_train_instances.csv",
        "wsi/wsi_cancer_train_bags.csv",
        "wsi/wsi_cancer_validation_instances.csv",
        "wsi/wsi_cancer_validation_bags.csv",
    }
    if any(
        original_hashes.get(name) != hashlib.sha256(assets[name]).hexdigest()
        for name in learning_names
    ):
        raise ValueError("MIL horizon manifests differ from confirmation-v1")
    mil_result = cast(
        "dict[str, object]",
        cast("dict[str, object]", original_audit.get("task_results")).get("mil"),
    )
    attempts = cast("list[dict[str, object]]", mil_result.get("attempts"))
    if (
        original_audit.get("config_sha256")
        != HORIZON_AUTHORITY_SHA256["resume/confirmation_v1_config.json"]
        or original_config.get("checkpoint_chunk_size") is not None
        or original_config.get("optimizer")
        != {"name": "AdamW", "weight_decay": SUPERVISED_WEIGHT_DECAY}
        or original_config.get("grad_scaler_init_scale") != GRAD_SCALER_INIT_SCALE
        or original_config.get("grad_scaler_growth_interval")
        != GRAD_SCALER_GROWTH_INTERVAL
        or cast("dict[str, object]", original_audit.get("shared_peaks")).get("mil")
        != HORIZON_PEAK
        or len(attempts) != 1
        or attempts[0].get("peak") != HORIZON_PEAK
        or any(
            len(history) != HORIZON_RESUME_UPDATES
            for history in cast(
                "dict[str, list[object]]",
                attempts[0].get("histories"),
            ).values()
        )
    ):
        raise ValueError("MIL horizon confirmation-v1 audit/config differs")


def _same_float(value: object, expected: float) -> bool:
    """Compare a serialized numeric checkpoint field to its exact authority."""
    return isinstance(value, int | float) and math.isclose(
        float(value),
        float(expected),
        rel_tol=0.0,
        abs_tol=0.0,
    )


def _validate_wsi_split(
    instances: Sequence[Mapping[str, str]],
    bags: Sequence[Mapping[str, str]],
    *,
    split: str,
    expected_bags: int,
) -> None:
    if not instances or len(bags) != expected_bags:
        raise ValueError("Calibration WSI split has an unexpected bag count")
    if any(row["split"] != split for row in (*instances, *bags)):
        raise ValueError("Calibration WSI manifest split differs from its package mode")
    cursor = 0
    seen_wsi_ids: set[str] = set()
    for bag_row, bag in enumerate(bags):
        count = int(bag["instance_count"])
        stop = cursor + count
        selected = instances[cursor:stop]
        if (
            int(bag["bag_row"]) != bag_row
            or bag["wsi_id"] in seen_wsi_ids
            or int(bag["instance_start"]) != cursor
            or count < 1
            or len(selected) != count
            or any(
                row["wsi_id"] != bag["wsi_id"]
                or row["diagnosis_label"] != bag["diagnosis_label"]
                or row["diagnosis_index"] != bag["diagnosis_index"]
                for row in selected
            )
        ):
            raise ValueError("Calibration WSI bag range differs from its instances")
        cursor = stop
        seen_wsi_ids.add(bag["wsi_id"])
    if cursor != len(instances):
        raise ValueError("Calibration WSI bag ranges do not cover every instance")


def _validate_disjoint_rows(
    train_rows: Sequence[Mapping[str, str]],
    validation_rows: Sequence[Mapping[str, str]],
    *,
    label: str,
) -> None:
    train_wsi = {row["wsi_id"] for row in train_rows}
    validation_wsi = {row["wsi_id"] for row in validation_rows}
    train_pointers = {(row["part"], row["file_index"]) for row in train_rows}
    validation_pointers = {(row["part"], row["file_index"]) for row in validation_rows}
    if train_wsi & validation_wsi or train_pointers & validation_pointers:
        raise ValueError(f"Calibration {label} train/validation rows overlap")


def _validate_width128_baseline_audit(
    path: Path,
    *,
    manifest_hashes: Mapping[str, str],
    expected_sha256: str = CLASS_SPECIFIC_BASELINE_AUDIT_SHA256,
) -> dict[str, object]:
    """Bind the class-specific stage to the exact completed width-128 evidence."""
    if _sha256(path) != expected_sha256:
        raise ValueError("Class-specific width-128 predecessor audit hash differs")
    audit = _read_object(path)
    task = cast("dict[str, object]", audit.get("task_result"))
    histories = cast("dict[str, list[object]]", task.get("histories"))
    validation = cast("dict[str, list[object]]", task.get("validation_history"))
    observations = cast(
        "dict[str, dict[str, object]]",
        audit.get("learning_observations"),
    )
    if (
        audit.get("schema_version") != "spec0023.mil_width128_audit.v1"
        or audit.get("package_mode") != "width128"
        or audit.get("status") != "complete"
        or audit.get("scientific_result") != "insufficient_learning"
        or audit.get("config_sha256") != CLASS_SPECIFIC_BASELINE_CONFIG_SHA256
        or audit.get("peak") != HORIZON_PEAK
        or audit.get("epochs") != HORIZON_TARGET_EPOCH
        or audit.get("attention_dim") != CLASS_SPECIFIC_ATTENTION_DIM
        or audit.get("logical_access") != "train_and_validation_only"
        or audit.get("frozen_tasks") != ["tissue"]
        or audit.get("checkpoint_chunk_size") is not None
        or task.get("status") != "insufficient_learning"
        or task.get("initialization_sha256") != WIDTH128_INITIALIZATION_SHA256
        or task.get("steps_per_epoch") != HORIZON_STEPS_PER_EPOCH
        or task.get("warmup_updates") != CLASS_SPECIFIC_WARMUP_UPDATES
        or task.get("zero_scaler_skips") is not True
        or task.get("logical_manifest_sha256") != dict(manifest_hashes)
        or set(histories) != set(REPRESENTATIONS)
        or any(
            len(histories[name]) != CLASS_SPECIFIC_UPDATES for name in REPRESENTATIONS
        )
        or set(validation) != set(REPRESENTATIONS)
        or any(
            len(validation[name]) != CLASS_SPECIFIC_VALIDATION_CHECKS
            for name in REPRESENTATIONS
        )
        or set(observations) != set(REPRESENTATIONS)
    ):
        raise ValueError("Class-specific width-128 predecessor audit differs")
    best_macro_f1 = {
        name: cast("dict[str, object]", observations[name]["best_validation"])[
            "macro_f1"
        ]
        for name in REPRESENTATIONS
    }
    expected_best = {
        "normal_vae": 0.2439628482972136,
        "so2_vae": 0.24643962848297213,
    }
    if best_macro_f1 != expected_best:
        raise ValueError("Class-specific width-128 reference metrics differ")
    return {
        "audit_sha256": expected_sha256,
        "config_sha256": CLASS_SPECIFIC_BASELINE_CONFIG_SHA256,
        "width128_initialization_sha256": WIDTH128_INITIALIZATION_SHA256,
        "class_specific_initialization_sha256": (CLASS_SPECIFIC_INITIALIZATION_SHA256),
        "best_macro_f1": best_macro_f1,
    }


def _validate_class_specific_failure_audit(
    path: Path,
    *,
    manifest_hashes: Mapping[str, str],
    baseline: Mapping[str, object],
) -> dict[str, object]:
    """Authenticate the exact one-sided overflow that permits one paired retry."""
    if _sha256(path) != CLASS_SPECIFIC_FAILURE_AUDIT_SHA256:
        raise ValueError("Class-specific failure audit hash differs")
    audit = _read_object(path)
    task = cast("dict[str, object]", audit.get("task_result"))
    diagnostics = cast("dict[str, object]", task.get("failure_diagnostics"))
    sample = cast("dict[str, object]", diagnostics.get("input"))
    logical_sample = cast("dict[str, object]", sample.get("logical_sample"))
    branches = cast(
        "dict[str, dict[str, object]]",
        diagnostics.get("branches"),
    )
    forward = cast("dict[str, dict[str, object]]", diagnostics.get("forward"))
    first_affected = cast("dict[str, object]", diagnostics.get("first_affected"))
    histories = cast("dict[str, list[object]]", task.get("histories"))
    validation = cast("dict[str, list[object]]", task.get("validation_history"))
    expected_baseline = {
        "audit_sha256": baseline["audit_sha256"],
        "config_sha256": baseline["config_sha256"],
        "width128_initialization_sha256": (baseline["width128_initialization_sha256"]),
        "class_specific_initialization_sha256": (
            baseline["class_specific_initialization_sha256"]
        ),
        "best_macro_f1": baseline["best_macro_f1"],
    }
    if (
        audit.get("schema_version") != "spec0023.mil_class_specific_audit.v1"
        or audit.get("package_mode") != "class_specific"
        or audit.get("status") != "failed"
        or audit.get("scientific_result") != "numerical_failure"
        or audit.get("config_sha256") != CLASS_SPECIFIC_FAILURE_CONFIG_SHA256
        or audit.get("peak") != HORIZON_PEAK
        or audit.get("epochs") != HORIZON_TARGET_EPOCH
        or audit.get("attention_dim") != CLASS_SPECIFIC_ATTENTION_DIM
        or audit.get("attention_maps") != CLASS_SPECIFIC_ATTENTION_MAPS
        or audit.get("logical_access") != "train_and_validation_only"
        or audit.get("frozen_tasks") != ["tissue"]
        or audit.get("checkpoint_chunk_size") is not None
        or audit.get("baseline") != expected_baseline
        or task.get("attempt") != "class_specific"
        or task.get("task") != "mil"
        or task.get("status") != "numerical_failure"
        or task.get("successful_updates") != CLASS_SPECIFIC_FAILURE_SUCCESSFUL_UPDATES
        or task.get("failed_attempt") != CLASS_SPECIFIC_FAILED_ATTEMPT
        or task.get("attention_dim") != CLASS_SPECIFIC_ATTENTION_DIM
        or task.get("class_specific_attention") is not True
        or task.get("logical_manifest_sha256") != dict(manifest_hashes)
        or task.get("half_peak_fallback_eligible") is not False
        or not _same_float(task.get("learning_rate"), HORIZON_PEAK)
        or not _same_float(task.get("peak"), HORIZON_PEAK)
        or set(histories) != set(REPRESENTATIONS)
        or any(
            len(histories[name]) != CLASS_SPECIFIC_FAILURE_SUCCESSFUL_UPDATES
            for name in REPRESENTATIONS
        )
        or set(validation) != set(REPRESENTATIONS)
        or any(
            len(validation[name]) != CLASS_SPECIFIC_FAILURE_VALIDATION_CHECKS
            for name in REPRESENTATIONS
        )
        or diagnostics.get("kind") != "gradient"
        or diagnostics.get("weight_application") != "post_grad_scaler_unscale"
        or first_affected
        != {
            "branch": "normal_vae",
            "dtype": "torch.float32",
            "max_finite_abs": None,
            "missing": False,
            "name": "patch_encoder.layers.0.weight",
            "nonfinite_count": 12_800,
        }
        or logical_sample
        != {
            "checkpoint_chunk_size": None,
            "class_weight": 0.9217391304347826,
            "diagnosis_index": 0,
            "diagnosis_label": "CC",
            "instance_count": 3869,
            "wsi_id": 55287,
        }
        or set(branches) != set(REPRESENTATIONS)
        or any(
            not _same_float(branches[name].get("scaler"), GRAD_SCALER_INIT_SCALE)
            for name in REPRESENTATIONS
        )
        or set(forward) != set(REPRESENTATIONS)
        or not all(_branch_forward_is_finite(forward[name]) for name in REPRESENTATIONS)
        or not all(_tensor_summary_is_finite(sample[name]) for name in REPRESENTATIONS)
    ):
        raise ValueError("Class-specific failure evidence differs")
    normal_parameters = cast(
        "list[dict[str, object]]",
        branches["normal_vae"].get("parameters"),
    )
    so2_parameters = cast(
        "list[dict[str, object]]",
        branches["so2_vae"].get("parameters"),
    )
    normal_nonfinite = {
        cast("str", row.get("name"))
        for row in normal_parameters
        if cast("int", row.get("nonfinite_count", 0)) > 0
    }
    if (
        not any(name.startswith("patch_encoder.") for name in normal_nonfinite)
        or not any(name.startswith("attention_") for name in normal_nonfinite)
        or any(
            cast("int", row.get("nonfinite_count", 0)) > 0
            for row in normal_parameters
            if cast("str", row.get("name")).startswith("head.")
        )
        or any(cast("int", row.get("nonfinite_count", 0)) > 0 for row in so2_parameters)
    ):
        raise ValueError("Class-specific branch-local overflow signature differs")
    return {
        "audit_sha256": CLASS_SPECIFIC_FAILURE_AUDIT_SHA256,
        "config_sha256": CLASS_SPECIFIC_FAILURE_CONFIG_SHA256,
        "predecessor_audit_sha256": CLASS_SPECIFIC_BASELINE_AUDIT_SHA256,
        "initialization_sha256": CLASS_SPECIFIC_INITIALIZATION_SHA256,
        "successful_updates": CLASS_SPECIFIC_FAILURE_SUCCESSFUL_UPDATES,
        "failed_attempt": CLASS_SPECIFIC_FAILED_ATTEMPT,
        "scale": GRAD_SCALER_INIT_SCALE,
        "failed_branch": "normal_vae",
        "first_affected_parameter": "patch_encoder.layers.0.weight",
        "failing_wsi_id": 55287,
        "instance_count": 3869,
        "complete_bag": True,
        "inputs_and_forwards_finite": True,
        "logical_access": "train_and_validation_only",
    }


def _tensor_summary_is_finite(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    summary = cast("dict[str, object]", value)
    return (
        _finite_number(summary.get("finite_max_abs"))
        and summary.get("nan_count") == 0
        and summary.get("negative_inf_count") == 0
        and summary.get("positive_inf_count") == 0
    )


def _branch_forward_is_finite(value: Mapping[str, object]) -> bool:
    return (
        _tensor_summary_is_finite(value.get("attention"))
        and _tensor_summary_is_finite(value.get("logits"))
        and _tensor_summary_is_finite(value.get("unweighted_cross_entropy"))
        and _finite_number(value.get("weighted_loss"))
    )


def _validate_manifest_audit(
    path: Path,
    manifest_hashes: Mapping[str, str],
    *,
    package_mode: str,
) -> str:
    audit = _read_object(path)
    if (
        audit.get("schema_version") != "spec0023.supervised_manifest_audit.v1"
        or audit.get("status") != "complete"
    ):
        raise ValueError("Canonical supervised manifest audit is not complete")
    physical = cast("dict[str, object]", audit["physical_catalog"])
    wsi_files = cast("dict[str, dict[str, object]]", audit["wsi_files"])
    tissue_files = cast("dict[str, dict[str, object]]", audit["tissue_files"])
    expected: dict[str, str] = {
        "physical_parts.csv": cast("str", physical["sha256"]),
        "wsi/wsi_cancer_train_instances.csv": cast(
            "str",
            wsi_files["wsi_cancer_train_instances.csv"]["sha256"],
        ),
        "wsi/wsi_cancer_train_bags.csv": cast(
            "str",
            wsi_files["wsi_cancer_train_bags.csv"]["sha256"],
        ),
    }
    if package_mode in {"sweep", "confirmation"}:
        expected["tissue/tissue_train_5671_per_class.csv"] = cast(
            "str",
            tissue_files["tissue_train_5671_per_class.csv"]["sha256"],
        )
    if package_mode in {
        "confirmation",
        "horizon",
        "width128",
        "class_specific",
        "class_specific_scale_fix",
    }:
        expected.update({
            "wsi/wsi_cancer_validation_instances.csv": cast(
                "str",
                wsi_files["wsi_cancer_validation_instances.csv"]["sha256"],
            ),
            "wsi/wsi_cancer_validation_bags.csv": cast(
                "str",
                wsi_files["wsi_cancer_validation_bags.csv"]["sha256"],
            ),
        })
    if package_mode == "confirmation":
        expected["tissue/tissue_validation.csv"] = cast(
            "str",
            tissue_files["tissue_validation.csv"]["sha256"],
        )
    observed = {name: manifest_hashes.get(name) for name in expected}
    if observed != expected:
        raise ValueError("Calibration inputs differ from the canonical manifest audit")
    return _sha256(path)


def _validate_sweep_audit(
    path: Path,
    *,
    expected_sha256: str,
    expected_manifest_hashes: Mapping[str, str] | None = None,
    sweep_config_path: Path,
    expected_input_receipt: Mapping[str, object] | None = None,
    expected_provenance: Mapping[str, object],
) -> dict[str, object]:
    if _sha256(path) != expected_sha256:
        raise ValueError("Downloaded sweep audit differs from the selection audit")
    audit = _read_object(path)
    if (
        audit.get("schema_version") != "spec0023.supervised_calibration_audit.v1"
        or audit.get("package_mode") != "sweep"
        or audit.get("status") != "selection_ready"
        or audit.get("selection_status")
        != "human_selection_required_before_confirmation"
    ):
        raise ValueError("Selection evidence is not a selection-ready Spec 0023 sweep")
    results = cast("dict[str, dict[str, object]]", audit.get("task_results"))
    config_sha256 = audit.get("config_sha256")
    if (
        set(results) != set(TASKS)
        or not isinstance(config_sha256, str)
        or len(config_sha256) != SHA256_HEX_LENGTH
    ):
        raise ValueError("Sweep evidence does not contain both paired tasks")
    if _sha256(sweep_config_path) != config_sha256:
        raise ValueError("Sweep configuration differs from its audit")
    sweep_config = _read_object(sweep_config_path)
    _validate_sweep_config(
        sweep_config,
        expected_manifest_hashes,
        expected_input_receipt=expected_input_receipt,
        expected_provenance=expected_provenance,
    )
    _validate_numerical_recovery(audit.get("numerical_recovery"))
    maximum_shared_lrs: dict[str, float] = {}
    for task, expected_steps in (("mil", 106), ("tissue", 132)):
        row = results[task]
        if (
            row.get("task") != task
            or row.get("steps_per_epoch") != expected_steps
            or set(cast("dict[str, object]", row.get("curves")))
            != {"normal_vae", "so2_vae"}
        ):
            raise ValueError("Sweep evidence task contract differs")
        maximum_shared_lrs[task] = _validate_sweep_task_result(
            row,
            expected_steps=expected_steps,
        )
        if expected_manifest_hashes is not None:
            task_names = ["physical_parts.csv"]
            if task == "mil":
                task_names.extend([
                    "wsi/wsi_cancer_train_instances.csv",
                    "wsi/wsi_cancer_train_bags.csv",
                ])
            else:
                task_names.append("tissue/tissue_train_5671_per_class.csv")
            expected_task_hashes = {
                name: expected_manifest_hashes[name] for name in task_names
            }
            if row.get("logical_manifest_sha256") != expected_task_hashes:
                raise ValueError("Sweep evidence logical manifests differ")
    return {
        "sha256": expected_sha256,
        "config_sha256": config_sha256,
        "task_terminal_status": {
            task: results[task].get("terminal_status") for task in TASKS
        },
        "maximum_shared_lrs": maximum_shared_lrs,
    }


def _validate_numerical_recovery(value: object) -> None:  # noqa: PLR0912
    """Require the exact v1 reproduction and isolated corrected 12-step replay."""
    if not isinstance(value, dict):
        raise TypeError("Sweep evidence lacks the required numerical recovery")
    recovery = cast("dict[str, object]", value)
    expected = {
        "attempt": RECOVERY_ATTEMPTS,
        "wsi_id": RECOVERY_WSI_ID,
        "diagnosis_label": "MC",
        "instance_count": RECOVERY_INSTANCE_COUNT,
        "class_weight": 2.65,
        "grad_scaler_scale": RECOVERY_SCALER,
    }
    if (
        recovery.get("status") != "pass"
        or recovery.get("expected_failure") != expected
        or recovery.get("v1_contract")
        != {
            "v1_sweep_audit_sha256": V1_SWEEP_AUDIT_SHA256,
            "legacy_initialization_sha256": V1_MIL_INITIALIZATION_SHA256,
            "legacy_order_sha256": V1_MIL_ORDER_SHA256,
            "legacy_failure_attempt": RECOVERY_ATTEMPTS,
            "legacy_failure_wsi_id": RECOVERY_WSI_ID,
            "legacy_class_weight": 2.65,
            "replacement": "post_grad_scaler_unscale",
            "replacement_replay_updates": RECOVERY_ATTEMPTS,
            "replay_state_reused_by_sweep": False,
        }
        or recovery.get("replay_state_reused_by_sweep") is not False
        or recovery.get("checkpoint_chunk_size") is not None
    ):
        raise ValueError("Sweep numerical recovery contract differs")
    legacy_row = _object_dict(recovery.get("legacy_scaled_loss_replay"))
    corrected_row = _object_dict(recovery.get("post_unscale_replay"))
    if legacy_row is None or corrected_row is None:
        raise TypeError("Sweep numerical recovery replays are missing")
    for field in ("initialization_sha256", "order_sha256"):
        identity = legacy_row.get(field)
        if (
            not isinstance(identity, str)
            or len(identity) != SHA256_HEX_LENGTH
            or corrected_row.get(field) != identity
        ):
            raise ValueError("Sweep numerical recovery replay identity differs")
    if (
        legacy_row.get("initialization_sha256") != V1_MIL_INITIALIZATION_SHA256
        or legacy_row.get("order_sha256") != V1_MIL_ORDER_SHA256
    ):
        raise ValueError("Sweep numerical recovery does not match sealed v1")
    legacy_committed = _object_list(legacy_row.get("committed"))
    corrected_committed = _object_list(corrected_row.get("committed"))
    failure = _object_dict(legacy_row.get("failure"))
    if (
        legacy_row.get("weight_application") != "scaled_loss"
        or legacy_row.get("successful_updates") != RECOVERY_LEGACY_UPDATES
        or legacy_committed is None
        or len(legacy_committed) != RECOVERY_LEGACY_UPDATES
        or failure is None
        or corrected_row.get("weight_application") != "post_grad_scaler_unscale"
        or corrected_row.get("successful_updates") != RECOVERY_ATTEMPTS
        or corrected_committed is None
        or len(corrected_committed) != RECOVERY_ATTEMPTS
        or corrected_row.get("failure") is not None
    ):
        raise ValueError("Sweep numerical recovery replay boundary differs")
    legacy_rows = legacy_committed
    corrected_rows = corrected_committed
    rates = tuple(
        1e-5 * math.exp(math.log(SWEEP_MAXIMUM_LR / 1e-5) * index / (106 - 1))
        for index in range(RECOVERY_ATTEMPTS)
    )
    for rows, count in (
        (legacy_rows, RECOVERY_LEGACY_UPDATES),
        (corrected_rows, RECOVERY_ATTEMPTS),
    ):
        for index, raw_row in enumerate(rows, start=1):
            if not isinstance(raw_row, dict):
                raise TypeError("Sweep numerical recovery commit must be an object")
            row = cast("dict[str, object]", raw_row)
            raw_losses = row.get("losses")
            losses = (
                cast("dict[str, object]", raw_losses)
                if isinstance(raw_losses, dict)
                else None
            )
            if (
                row.get("attempt") != index
                or not _same_finite_number(row.get("learning_rate"), rates[index - 1])
                or losses is None
                or set(losses) != set(REPRESENTATIONS)
                or not all(_finite_number(loss) for loss in losses.values())
            ):
                raise ValueError("Sweep numerical recovery committed prefix differs")
        if len(rows) != count:
            raise ValueError("Sweep numerical recovery prefix length differs")
    failure_row = failure
    if (
        failure_row.get("attempt") != RECOVERY_ATTEMPTS
        or not _same_finite_number(
            failure_row.get("learning_rate"),
            rates[RECOVERY_ATTEMPTS - 1],
        )
        or failure_row.get("wsi_id") != RECOVERY_WSI_ID
        or failure_row.get("diagnosis_label") != "MC"
        or failure_row.get("diagnosis_index") != RECOVERY_DIAGNOSIS_INDEX
        or failure_row.get("instance_count") != RECOVERY_INSTANCE_COUNT
    ):
        raise ValueError("Sweep numerical recovery failure identity differs")
    raw_diagnostics = _object_dict(failure_row.get("diagnostics"))
    if raw_diagnostics is None:
        raise TypeError("Sweep numerical recovery lacks gradient failure evidence")
    diagnostics = raw_diagnostics
    if diagnostics.get("kind") != "gradient":
        raise ValueError("Sweep numerical recovery lacks gradient failure evidence")
    diagnostic_row = diagnostics
    first_affected = diagnostic_row.get("first_affected")
    raw_inputs = _object_dict(diagnostic_row.get("input"))
    raw_branches = _object_dict(diagnostic_row.get("branches"))
    if (
        raw_inputs is None
        or raw_branches is None
        or not isinstance(first_affected, dict)
    ):
        raise TypeError("Sweep numerical recovery diagnostics are incomplete")
    inputs = raw_inputs
    branches = raw_branches
    sample = inputs.get("logical_sample")
    if not isinstance(sample, dict) or sample != {
        "wsi_id": RECOVERY_WSI_ID,
        "diagnosis_label": "MC",
        "diagnosis_index": RECOVERY_DIAGNOSIS_INDEX,
        "instance_count": RECOVERY_INSTANCE_COUNT,
        "class_weight": 2.65,
        "checkpoint_chunk_size": None,
    }:
        raise ValueError("Sweep numerical recovery input identity differs")
    if set(branches) != set(REPRESENTATIONS):
        raise ValueError("Sweep numerical recovery branch diagnostics differ")
    has_nonfinite = False
    listed_affected: list[dict[str, object]] = []
    for name in REPRESENTATIONS:
        raw_branch = _object_dict(branches[name])
        if raw_branch is None:
            raise TypeError("Sweep numerical recovery scaler evidence differs")
        branch = raw_branch
        if not _same_finite_number(branch.get("scaler"), RECOVERY_SCALER):
            raise ValueError("Sweep numerical recovery scaler evidence differs")
        raw_parameters = _object_list(branch.get("parameters"))
        if raw_parameters is None or not raw_parameters:
            raise ValueError("Sweep numerical recovery parameter evidence is missing")
        parameters = raw_parameters
        for parameter in parameters:
            typed_parameter = _object_dict(parameter)
            if typed_parameter is None:
                raise TypeError("Sweep recovery parameter evidence must be an object")
            count = typed_parameter.get("nonfinite_count")
            if isinstance(count, int) and not isinstance(count, bool) and count > 0:
                has_nonfinite = True
                listed_affected.append({"branch": name, **typed_parameter})
    if not has_nonfinite:
        raise ValueError(
            "Sweep numerical recovery did not reproduce a nonfinite gradient",
        )
    if first_affected not in listed_affected:
        raise ValueError("Sweep numerical recovery first affected parameter differs")


def _validate_sweep_task_result(  # noqa: PLR0912
    row: Mapping[str, object],
    *,
    expected_steps: int,
) -> float:
    """Return the last selectable aligned LR after validating one task curve."""
    raw_curves = row.get("curves")
    if not isinstance(raw_curves, dict):
        raise TypeError("Sweep evidence lacks paired curves")
    curves = cast("dict[str, object]", raw_curves)
    normal = curves.get("normal_vae")
    so2 = curves.get("so2_vae")
    learning_rates = row.get("learning_rates")
    attempts = row.get("attempts")
    successful = row.get("successful_updates")
    if not all(
        isinstance(value, list) for value in (normal, so2, learning_rates, attempts)
    ):
        raise TypeError("Sweep evidence curve arrays must be lists")
    normal_points = cast("list[object]", normal)
    so2_points = cast("list[object]", so2)
    rate_values = cast("list[object]", learning_rates)
    attempt_rows = cast("list[object]", attempts)
    if (
        isinstance(successful, bool)
        or not isinstance(successful, int)
        or successful < MINIMUM_SELECTABLE_SWEEP_UPDATES
        or len(normal_points) != successful
        or len(so2_points) != successful
        or len(rate_values) != successful
    ):
        raise ValueError("Sweep evidence lacks a usable aligned curve prefix")
    expected_rates = tuple(
        1e-5
        * math.exp(math.log(SWEEP_MAXIMUM_LR / 1e-5) * index / (expected_steps - 1))
        for index in range(expected_steps)
    )
    ewma_history: dict[str, list[float]] = {name: [] for name in REPRESENTATIONS}
    first_divergence: int | None = None
    for index, observed_lr in enumerate(rate_values, start=1):
        if not _same_finite_number(observed_lr, expected_rates[index - 1]):
            raise ValueError("Sweep evidence learning-rate prefix differs")
        for branch, curve in zip(
            REPRESENTATIONS,
            (normal_points, so2_points),
            strict=True,
        ):
            raw_point = curve[index - 1]
            if not isinstance(raw_point, dict):
                raise TypeError("Sweep evidence curve point must be an object")
            point = cast("dict[str, object]", raw_point)
            loss = point.get("loss")
            ewma = point.get("ewma")
            if (
                point.get("update") != index
                or not _finite_number(loss)
                or not _finite_number(ewma)
            ):
                raise ValueError("Sweep evidence curve values differ")
            numeric_loss = float(cast("float | int", loss))
            history = ewma_history[branch]
            expected_ewma = (
                numeric_loss if not history else 0.1 * numeric_loss + 0.9 * history[-1]
            )
            if not _same_finite_number(ewma, expected_ewma):
                raise ValueError("Sweep evidence EWMA recurrence differs")
            numeric_ewma = float(cast("float | int", ewma))
            if (
                index >= MINIMUM_SELECTABLE_SWEEP_UPDATES
                and numeric_ewma > 4.0 * min(history)
                and first_divergence is None
            ):
                first_divergence = index
            history.append(numeric_ewma)
        raw_attempt = attempt_rows[index - 1] if index <= len(attempt_rows) else None
        if not isinstance(raw_attempt, dict):
            raise TypeError("Sweep evidence attempt must be an object")
        attempt = cast("dict[str, object]", raw_attempt)
        if (
            attempt.get("attempt") != index
            or attempt.get("status") != "committed"
            or not _same_finite_number(
                attempt.get("learning_rate"),
                expected_rates[index - 1],
            )
        ):
            raise ValueError("Sweep evidence committed attempts differ")

    terminal = row.get("terminal_status")
    if terminal == "complete":
        if (
            successful != expected_steps
            or len(attempt_rows) != expected_steps
            or first_divergence is not None
        ):
            raise ValueError("Complete sweep evidence has an incomplete epoch")
        return expected_rates[-1]
    if terminal == "diverged":
        if (
            successful > expected_steps
            or len(attempt_rows) != successful
            or first_divergence != successful
        ):
            raise ValueError("Diverged sweep evidence has inconsistent attempts")
        return expected_rates[successful - 2]
    if isinstance(terminal, str) and terminal.startswith("numerical_failure:"):
        if (
            successful >= expected_steps
            or len(attempt_rows) != successful + 1
            or first_divergence is not None
        ):
            raise ValueError("Failed sweep evidence has inconsistent attempts")
        raw_failure = attempt_rows[-1]
        if not isinstance(raw_failure, dict):
            raise TypeError("Sweep failure attempt must be an object")
        failure = cast("dict[str, object]", raw_failure)
        if (
            failure.get("attempt") != successful + 1
            or failure.get("status") != "numerical_failure"
            or not _same_finite_number(
                failure.get("learning_rate"),
                expected_rates[successful],
            )
        ):
            raise ValueError("Sweep numerical-failure boundary differs")
        return expected_rates[successful - 1]
    raise ValueError("Sweep terminal status is not selectable")


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (float, int))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _same_finite_number(value: object, expected: float) -> bool:
    return _finite_number(value) and math.isclose(
        float(cast("float | int", value)),
        expected,
        rel_tol=1e-12,
        abs_tol=0.0,
    )


def _validate_sweep_config(
    config: Mapping[str, object],
    expected_manifest_hashes: Mapping[str, str] | None,
    *,
    expected_input_receipt: Mapping[str, object] | None,
    expected_provenance: Mapping[str, object],
) -> None:
    locked = {
        "schema_version": "spec0023.supervised_calibration_config.v1",
        "package_mode": "sweep",
        "model_devices": {"normal_vae": 0, "so2_vae": 1},
        "optimizer": {"name": "AdamW", "weight_decay": 1e-4},
        "precision": "FP16-autocast-with-GradScaler",
        "grad_scaler_init_scale": 32_768,
        "grad_scaler_growth_interval": 1_000_000,
        "checkpoint_chunk_size": None,
        "numerical_recovery": {
            "v1_sweep_audit_sha256": V1_SWEEP_AUDIT_SHA256,
            "legacy_initialization_sha256": V1_MIL_INITIALIZATION_SHA256,
            "legacy_order_sha256": V1_MIL_ORDER_SHA256,
            "legacy_failure_attempt": 12,
            "legacy_failure_wsi_id": 65094,
            "legacy_class_weight": 2.65,
            "replacement": "post_grad_scaler_unscale",
            "replacement_replay_updates": 12,
            "replay_state_reused_by_sweep": False,
        },
        "sweep": {
            "start_learning_rate": 1e-5,
            "stop_learning_rate": SWEEP_MAXIMUM_LR,
            "ewma_alpha": 0.1,
            "divergence_multiple": 4.0,
            "mil_updates": 106,
            "tissue_updates": 132,
        },
        "tasks": {
            "mil": {
                "initialization_seed": 1701,
                "steps_per_epoch": 106,
                "class_weight_application": "post_grad_scaler_unscale",
                "train_instances": "wsi/wsi_cancer_train_instances.csv",
                "train_bags": "wsi/wsi_cancer_train_bags.csv",
            },
            "tissue": {
                "initialization_seed": 3407,
                "steps_per_epoch": 132,
                "batch_size": 128,
                "drop_last": True,
                "train_instances": "tissue/tissue_train_5671_per_class.csv",
            },
        },
    }
    if any(config.get(key) != value for key, value in locked.items()):
        raise ValueError("Sweep configuration differs from the locked calibration")
    receipt = _object_dict(config.get("input_dataset_receipt"))
    logical_hashes = _object_dict(config.get("logical_manifest_sha256"))
    expected_logical_names = {
        "physical_parts.csv",
        "wsi/wsi_cancer_train_instances.csv",
        "wsi/wsi_cancer_train_bags.csv",
        "tissue/tissue_train_5671_per_class.csv",
    }
    required_hashes = (
        "spec_sha256",
        "supervised_manifest_audit_sha256",
        "input_source_tree_sha256",
    )
    if (
        receipt is None
        or receipt.get("schema_version")
        != "spec0023.supervised_calibration_input_receipt.v1"
        or receipt.get("package_mode") != "sweep"
        or receipt.get("dataset_reference") != INPUT_DATASET_SLUGS["sweep"]
        or isinstance(receipt.get("dataset_version"), bool)
        or not isinstance(receipt.get("dataset_version"), int)
        or cast("int", receipt["dataset_version"]) < 1
        or not _sha256_string(receipt.get("input_contract_sha256"))
        or not _valid_remote_files(receipt.get("remote_files"))
        or logical_hashes is None
        or set(logical_hashes) != expected_logical_names
        or any(not _sha256_string(value) for value in logical_hashes.values())
        or any(not _sha256_string(config.get(name)) for name in required_hashes)
        or config.get("output_allowlist")
        != ["spec0023_supervised_calibration_audit.json"]
    ):
        raise ValueError("Sweep configuration v2 input provenance differs")
    if expected_input_receipt is not None and receipt != expected_input_receipt:
        raise ValueError("Sweep configuration differs from the verified v2 receipt")
    if any(
        config.get(name) != expected_provenance.get(name) for name in required_hashes
    ):
        raise ValueError("Sweep configuration source/spec provenance differs")
    expected_source_records = expected_provenance.get("source_records")
    expected_kernel_sources = (
        [
            record.get("kaggle_source")
            for record in cast("list[dict[str, object]]", expected_source_records)
        ]
        if isinstance(expected_source_records, list)
        else None
    )
    if (
        config.get("source_records") != expected_source_records
        or config.get("kernel_sources") != expected_kernel_sources
        or not _valid_source_records(
            config.get("kernel_sources"),
            config.get("source_records"),
        )
    ):
        raise ValueError("Sweep configuration latent-source provenance differs")
    if expected_manifest_hashes is not None:
        expected = {
            name: expected_manifest_hashes[name]
            for name in (
                "physical_parts.csv",
                "wsi/wsi_cancer_train_instances.csv",
                "wsi/wsi_cancer_train_bags.csv",
                "tissue/tissue_train_5671_per_class.csv",
            )
        }
        if config.get("logical_manifest_sha256") != expected:
            raise ValueError("Sweep configuration logical manifests differ")


def _sha256_string(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == SHA256_HEX_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _object_dict(value: object) -> dict[str, object] | None:
    return cast("dict[str, object]", value) if isinstance(value, dict) else None


def _object_list(value: object) -> list[object] | None:
    return cast("list[object]", value) if isinstance(value, list) else None


def _valid_remote_files(value: object) -> bool:
    rows = _object_list(value)
    if not rows:
        return False
    contract_seen = False
    for raw_row in rows:
        row = _object_dict(raw_row)
        if row is None:
            return False
        size = row.get("bytes")
        if (
            not isinstance(row.get("logical_name"), str)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 1
            or not _sha256_string(row.get("sha256"))
        ):
            return False
        contract_seen = contract_seen or row["logical_name"] == INPUT_CONTRACT_NAME
    return contract_seen


def _valid_source_records(kernel_value: object, records_value: object) -> bool:
    kernel_sources = _object_list(kernel_value)
    records = _object_list(records_value)
    if not kernel_sources or not records:
        return False
    observed_sources: list[object] = []
    for raw_record in records:
        record = _object_dict(raw_record)
        if record is None:
            return False
        binaries = _object_list(record.get("binaries"))
        if not isinstance(record.get("kaggle_source"), str) or not binaries:
            return False
        observed_sources.append(record["kaggle_source"])
    return kernel_sources == observed_sources


def _validate_selection_bounds(
    selection: Mapping[str, object],
    sweep_evidence: Mapping[str, object],
) -> None:
    maximum_lrs = cast("dict[str, float]", sweep_evidence["maximum_shared_lrs"])
    peaks = cast("dict[str, float]", selection["shared_peaks"])
    if any(peaks[task] > maximum_lrs[task] for task in TASKS):
        raise ValueError(
            "A selected shared peak exceeds its aligned stable sweep prefix",
        )


def _read_selection_audit(payload: bytes) -> dict[str, object]:
    value = cast("object", json.loads(payload))
    required = {"sweep_audit_sha256", "ewma", *TASKS}
    if not isinstance(value, dict):
        raise TypeError("Selection audit must be an object")
    audit = cast("dict[str, object]", value)
    if set(audit) != required:
        raise ValueError(
            "Selection audit must bind sweep evidence and exactly the two task peaks",
        )
    sweep_hash = audit["sweep_audit_sha256"]
    if (
        not isinstance(sweep_hash, str)
        or len(sweep_hash) != SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in sweep_hash)
    ):
        raise ValueError("Selection audit must bind a lowercase sweep-audit SHA-256")
    raw_ewma = audit["ewma"]
    if not isinstance(raw_ewma, dict):
        raise TypeError("Selection audit EWMA rule must be an object")
    ewma = cast("dict[str, object]", raw_ewma)
    if ewma != {
        "alpha": 0.1,
        "divergence_multiple": 4.0,
    }:
        raise ValueError("Selection audit must retain the locked EWMA rule")
    peaks: dict[str, float] = {}
    for task in TASKS:
        raw_row = audit[task]
        if not isinstance(raw_row, dict):
            raise TypeError("Each selection-audit task must be an object")
        row = cast("dict[str, object]", raw_row)
        if set(row) != {"shared_peak", "rationale"}:
            raise ValueError(
                "Each selection-audit task needs one shared peak and rationale",
            )
        peak = row["shared_peak"]
        if (
            not isinstance(peak, (float, int))
            or isinstance(peak, bool)
            or not 0.0 < peak <= SWEEP_MAXIMUM_LR
        ):
            raise ValueError("A shared calibration peak must be within the swept range")
        if not isinstance(row["rationale"], str) or not row["rationale"].strip():
            raise ValueError("The human selection audit requires a nonempty rationale")
        peaks[task] = float(peak)
    return {
        "sweep_audit_sha256": sweep_hash,
        "ewma": {"alpha": 0.1, "divergence_multiple": 4.0},
        "shared_peaks": peaks,
        "fallback_policy": "only_after_a_recorded_paired_numerical_failure",
    }


def _source_records(
    catalog_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in catalog_rows:
        grouped.setdefault(row["kaggle_source"], []).append({
            "name": row["binary_name"],
            "bytes": int(row["binary_bytes"]),
            "sha256": row["binary_sha256"],
        })
    return [
        {
            "kaggle_source": source,
            "binaries": sorted(records, key=lambda record: cast("str", record["name"])),
        }
        for source, records in grouped.items()
    ]


def _source_tree_sha256(contract: Mapping[str, object]) -> str:
    files = cast("dict[str, dict[str, object]]", contract.get("files"))
    source_records = {
        name: record
        for name, record in files.items()
        if name.startswith("src/eqvae/") and name.endswith(".py")
    }
    return hashlib.sha256(_canonical_json(source_records)).hexdigest()


def _contract_spec_sha256(contract: Mapping[str, object]) -> str:
    value = contract.get("spec_sha256")
    if not _sha256_string(value):
        raise ValueError("Calibration input contract lacks a valid spec SHA-256")
    return cast("str", value)


def _validate_contract_sources(
    contract: Mapping[str, object],
    *,
    repo_root: Path,
) -> str:
    files = cast("dict[str, dict[str, object]]", contract.get("files"))
    contracted = {
        name: record
        for name, record in files.items()
        if name.startswith("src/eqvae/") and name.endswith(".py")
    }
    current = {
        path.relative_to(repo_root).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted((repo_root / "src/eqvae").rglob("*.py"))
        if "__pycache__" not in path.parts
    }
    if contracted != current:
        raise ValueError("Calibration input source tree differs from the current repo")
    return _source_tree_sha256(contract)


def _validate_input_receipt(
    *,
    repo_root: Path,
    bundle_root: Path,
    receipt_path: Path,
    package_mode: str,
    expected_spec_sha256: str,
    expected_manifest_audit_sha256: str,
    require_current_source_tree: bool = True,
) -> dict[str, object]:
    contract_path = bundle_root / INPUT_CONTRACT_NAME
    contract = _read_object(contract_path)
    receipt = _read_object(receipt_path)
    dataset_reference = INPUT_DATASET_SLUGS[package_mode]
    version = receipt.get("dataset_version")
    if (
        contract.get("schema_version")
        != "spec0023.supervised_calibration_input_contract.v1"
        or contract.get("package_mode") != package_mode
        or contract.get("dataset_reference") != dataset_reference
        or contract.get("spec_sha256") != expected_spec_sha256
        or contract.get("supervised_manifest_audit_sha256")
        != expected_manifest_audit_sha256
        or receipt.get("schema_version")
        != "spec0023.supervised_calibration_input_receipt.v1"
        or receipt.get("package_mode") != package_mode
        or receipt.get("dataset_reference") != dataset_reference
        or isinstance(version, bool)
        or not isinstance(version, int)
        or version < 1
        or receipt.get("input_contract_sha256") != _sha256(contract_path)
    ):
        raise ValueError("Calibration input receipt or contract identity differs")
    files = cast("dict[str, dict[str, object]]", contract["files"])
    expected = {
        name: (cast("int", record["bytes"]), cast("str", record["sha256"]))
        for name, record in files.items()
    }
    expected[INPUT_CONTRACT_NAME] = (
        contract_path.stat().st_size,
        _sha256(contract_path),
    )
    remote_files = cast("list[dict[str, object]]", receipt.get("remote_files"))
    observed = {
        cast("str", record["logical_name"]): (
            cast("int", record["bytes"]),
            cast("str", record["sha256"]),
        )
        for record in remote_files
    }
    if observed != expected:
        raise ValueError("Calibration input receipt file authority differs")
    if require_current_source_tree:
        _validate_contract_sources(contract, repo_root=repo_root)
    return receipt


def _metadata(
    package_mode: str,
    source_records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if package_mode == "confirmation":
        kernel_slug = "eqvae-ubc-ocean-supcal-confirmation"
        title = "eqvae UBC-OCEAN supcal confirmation"
    elif package_mode == "horizon":
        kernel_slug = "eqvae-ubc-ocean-mil-horizon"
        title = "eqvae UBC-OCEAN MIL horizon"
    elif package_mode == "width128":
        kernel_slug = "eqvae-ubc-ocean-mil-width128"
        title = "eqvae UBC-OCEAN MIL width128"
    elif package_mode == "class_specific":
        kernel_slug = "eqvae-ubc-ocean-mil-class-attn"
        title = "eqvae UBC-OCEAN MIL class attention"
    elif package_mode == "class_specific_scale_fix":
        kernel_slug = "eqvae-ubc-ocean-mil-class-scale"
        title = "eqvae UBC-OCEAN MIL class scale"
    else:
        kernel_slug = "eqvae-ubc-ocean-supervised-calibration-sweep"
        title = "eqvae UBC-OCEAN supervised calibration sweep"
    return {
        "id": f"maximusshtefan/{kernel_slug}",
        "title": title,
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [INPUT_DATASET_SLUGS[package_mode]],
        "competition_sources": [],
        "kernel_sources": [record["kaggle_source"] for record in source_records],
        "model_sources": [],
    }


def _render_wrapper(repo_root: Path, config: bytes) -> bytes:
    substitutions = {
        "embedded_config_b64": base64.b64encode(config).decode(),
        "embedded_config_sha256": hashlib.sha256(config).hexdigest(),
    }
    template = Template((repo_root / TEMPLATE_PATH).read_text(encoding="utf-8"))
    return template.substitute(substitutions).encode()


def _read_csv_bytes(payload: bytes, header: Sequence[str]) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(payload.decode(), newline=""))
    if tuple(reader.fieldnames or ()) != tuple(header):
        raise ValueError("Unexpected calibration logical CSV header")
    return [dict(row) for row in reader]


def _read_object(path: Path) -> dict[str, object]:
    value = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return cast("dict[str, object]", value)


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    return f"{json.dumps(payload, sort_keys=True, separators=(',', ':'))}\n".encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_mode", choices=PACKAGE_MODES)
    parser.add_argument(
        "action",
        choices=("build", "validate"),
        nargs="?",
        default="build",
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--selection-audit", type=Path)
    parser.add_argument("--sweep-audit", type=Path)
    parser.add_argument("--sweep-config", type=Path)
    parser.add_argument("--manifest-audit", type=Path)
    parser.add_argument("--input-bundle-root", type=Path)
    parser.add_argument("--input-receipt", type=Path)
    parser.add_argument(
        "--class-specific-baseline-audit",
        type=Path,
        default=CLASS_SPECIFIC_BASELINE_AUDIT,
    )
    parser.add_argument(
        "--class-specific-failure-audit",
        type=Path,
        default=CLASS_SPECIFIC_FAILURE_AUDIT,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build or validate the selected compact local calibration package."""
    args = _parser().parse_args(argv)
    mode = cast("str", args.package_mode)
    output_root = cast("Path | None", args.output_root) or DEFAULT_OUTPUT_ROOT / mode
    repo_root = cast("Path", args.repo_root).resolve()
    manifest_root = cast("Path", args.manifest_root)
    selection_audit_path = cast("Path | None", args.selection_audit)
    sweep_audit_path = cast("Path | None", args.sweep_audit)
    sweep_config_path = cast("Path | None", args.sweep_config)
    manifest_audit_path = cast("Path | None", args.manifest_audit)
    input_bundle_root = cast("Path | None", args.input_bundle_root)
    input_receipt_path = cast("Path | None", args.input_receipt)
    class_specific_baseline_audit_path = cast(
        "Path",
        args.class_specific_baseline_audit,
    )
    class_specific_failure_audit_path = cast(
        "Path",
        args.class_specific_failure_audit,
    )
    if cast("str", args.action) == "build":
        build_calibration_package(
            repo_root=repo_root,
            manifest_root=manifest_root,
            output_root=output_root,
            package_mode=mode,
            selection_audit_path=selection_audit_path,
            sweep_audit_path=sweep_audit_path,
            sweep_config_path=sweep_config_path,
            manifest_audit_path=manifest_audit_path,
            input_bundle_root=input_bundle_root,
            input_receipt_path=input_receipt_path,
            class_specific_baseline_audit_path=(class_specific_baseline_audit_path),
            class_specific_failure_audit_path=(class_specific_failure_audit_path),
        )
    else:
        validate_calibration_package(
            repo_root=repo_root,
            manifest_root=manifest_root,
            output_root=output_root,
            package_mode=mode,
            selection_audit_path=selection_audit_path,
            sweep_audit_path=sweep_audit_path,
            sweep_config_path=sweep_config_path,
            manifest_audit_path=manifest_audit_path,
            input_bundle_root=input_bundle_root,
            input_receipt_path=input_receipt_path,
            class_specific_baseline_audit_path=(class_specific_baseline_audit_path),
            class_specific_failure_audit_path=(class_specific_failure_audit_path),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
