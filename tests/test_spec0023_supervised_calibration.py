# Copyright 2026 HiperMaximus
# ruff: noqa: PLC2701, PLR2004
# pyright: reportAny=false
# pyright: reportPrivateUsage=false
"""Focused checks for the bounded paired Spec 0023 calibration mechanics."""

from __future__ import annotations

import ast
import copy
import csv
import hashlib
import io
import json
import shutil
import zipfile
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
import torch
from torch import nn

from eqvae.cli.build_ubc_supervised_calibration import (
    CLASS_SPECIFIC_BASELINE_CONFIG_SHA256,
    CLASS_SPECIFIC_INITIALIZATION_SHA256,
    HORIZON_AUTHORITY_SHA256,
    HORIZON_PEAK,
    HORIZON_START_EPOCH,
    HORIZON_TARGET_EPOCH,
    INPUT_RECEIPT_FILENAMES,
    KAGGLE_SCRIPT_LIMIT_BYTES,
    V1_MIL_INITIALIZATION_SHA256,
    V1_MIL_ORDER_SHA256,
    V1_SWEEP_AUDIT_SHA256,
    WIDTH128_BASELINE_AUDIT_SHA256,
    WIDTH128_INITIALIZATION_SHA256,
    _contract_spec_sha256,
    _metadata,
    _read_selection_audit,
    _validate_assets,
    _validate_disjoint_rows,
    _validate_input_receipt,
    _validate_manifest_audit,
    _validate_selection_bounds,
    _validate_sweep_audit,
    _validate_width128_baseline_audit,
    build_calibration_package,
)
from eqvae.cli.build_ubc_supervised_calibration_inputs import (
    CONTRACT_NAME,
    DATASET_SLUGS,
    METADATA_NAME,
    _dataset_metadata,
    build_input_bundle,
    seal_remote_receipt,
    stage_upload_envelope,
    validate_input_bundle,
)
from eqvae.data.supervised_latents import (
    CATALOG_HEADER,
    TISSUE_HEADER,
    WSI_BAG_HEADER,
    WSI_INSTANCE_HEADER,
)
from eqvae.models.supervised import AttentionMILClassifier, TissueClassifier
from eqvae.training.supervised_calibration import (
    BRANCH_NAMES,
    BranchState,
    PairedNumericalError,
    commit_paired_boundary,
    load_paired_boundary,
    make_branch_state,
    paired_atomic_step,
    paired_checkpoint_payload,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
from eqvae.training.supervised_pairing import (
    confirmation_learning_rate,
    exponential_learning_rates,
    full_training_learning_rate,
    half_epoch_boundaries,
    warmup_update_count,
)


def _cpu_branches() -> dict[str, BranchState]:
    reference = nn.Linear(2, 1)
    nn.init.ones_(reference.weight)
    nn.init.zeros_(reference.bias)
    return {
        name: make_branch_state(copy.deepcopy(reference), device=torch.device("cpu"))
        for name in BRANCH_NAMES
    }


def _finite_loss_with_nonfinite_gradient(value: torch.Tensor) -> torch.Tensor:
    """Produce a finite scalar whose inactive autograd path proves gradient checks run.

    Returns:
        A scalar with finite forward value and nonfinite backward value.

    """
    return torch.where(
        torch.ones_like(value, dtype=torch.bool),
        value,
        value / 0.0,
    )


def _loss_that_overflows_only_at_large_scale(value: torch.Tensor) -> torch.Tensor:
    """Model the observed finite-forward overflow so a halved scale repairs it.

    Returns:
        The unchanged finite value with a deliberately scale-sensitive backward hook.

    """

    def scaled_gradient(gradient: torch.Tensor) -> torch.Tensor:
        if float(gradient.item()) > 20_000.0:
            return torch.full_like(gradient, float("inf"))
        return gradient

    value.register_hook(scaled_gradient)  # pyright: ignore[reportUnknownMemberType]
    return value


def _csv_payload(
    header: Sequence[str],
    rows: Sequence[Sequence[object]],
) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return output.getvalue().encode()


def _package_assets(*, confirmation: bool) -> dict[str, bytes]:
    catalog_rows = [
        (
            1,
            model,
            "source",
            f"{model}.bin",
            20_000,
            1_310_720_064,
            "a" * 64,
            f"{model}.json",
            10,
            "b" * 64,
        )
        for model in ("normal_vae", "so2_vae")
    ]
    train_instances = [
        (index, index, 1000 + index, 0, 0, "HGSC", 0, "train", 1, index)
        for index in range(106)
    ]
    train_bags = [
        (index, 1000 + index, "HGSC", 0, "train", index, 1) for index in range(106)
    ]
    tissues = ("tumor", "stroma", "necrosis")
    tissue_train = [
        (
            row,
            row,
            10_000 + row,
            0,
            0,
            tissues[row // 5671],
            "train",
            row % 5671,
            1,
            row,
        )
        for row in range(3 * 5671)
    ]
    assets = {
        "physical_parts.csv": _csv_payload(CATALOG_HEADER, catalog_rows),
        "wsi/wsi_cancer_train_instances.csv": _csv_payload(
            WSI_INSTANCE_HEADER,
            train_instances,
        ),
        "wsi/wsi_cancer_train_bags.csv": _csv_payload(WSI_BAG_HEADER, train_bags),
        "tissue/tissue_train_5671_per_class.csv": _csv_payload(
            TISSUE_HEADER,
            tissue_train,
        ),
    }
    if confirmation:
        validation_instances = [
            (
                index,
                20_000 + index,
                30_000 + index,
                0,
                0,
                "HGSC",
                0,
                "validation",
                1,
                1000 + index,
            )
            for index in range(23)
        ]
        validation_bags = [
            (index, 30_000 + index, "HGSC", 0, "validation", index, 1)
            for index in range(23)
        ]
        tissue_validation = [
            (
                index,
                40_000 + index,
                50_000 + index,
                0,
                0,
                tissue,
                "validation",
                "",
                1,
                18_000 + index,
            )
            for index, tissue in enumerate(tissues)
        ]
        selection = {
            "sweep_audit_sha256": "c" * 64,
            "ewma": {"alpha": 0.1, "divergence_multiple": 4.0},
            "mil": {"shared_peak": 3e-4, "rationale": "Both curves descend."},
            "tissue": {"shared_peak": 5e-4, "rationale": "Both curves descend."},
        }
        assets.update({
            "wsi/wsi_cancer_validation_instances.csv": _csv_payload(
                WSI_INSTANCE_HEADER,
                validation_instances,
            ),
            "wsi/wsi_cancer_validation_bags.csv": _csv_payload(
                WSI_BAG_HEADER,
                validation_bags,
            ),
            "tissue/tissue_validation.csv": _csv_payload(
                TISSUE_HEADER,
                tissue_validation,
            ),
            "selection_audit.json": json.dumps(selection).encode(),
        })
    return assets


def _sweep_task_result(
    task: str,
    steps: int,
    *,
    successful: int | None = None,
    terminal: str = "complete",
) -> dict[str, object]:
    successful = steps if successful is None else successful
    rates = exponential_learning_rates(start=1e-5, stop=3e-3, update_count=steps)
    prefix = rates[:successful]
    losses = [1.0 / index for index in range(1, successful + 1)]
    if terminal == "diverged":
        losses[-1] = 1_000_000.0
    curve: list[dict[str, object]] = []
    ewma: float | None = None
    for index, loss in enumerate(losses, start=1):
        ewma = loss if ewma is None else 0.1 * loss + 0.9 * ewma
        curve.append({"update": index, "loss": loss, "ewma": ewma})
    curves = {name: copy.deepcopy(curve) for name in BRANCH_NAMES}
    attempts = [
        {
            "attempt": index,
            "learning_rate": rate,
            "status": "committed",
        }
        for index, rate in enumerate(prefix, start=1)
    ]
    if terminal.startswith("numerical_failure:"):
        attempts.append({
            "attempt": successful + 1,
            "learning_rate": rates[successful],
            "status": "numerical_failure",
        })
    return {
        "task": task,
        "steps_per_epoch": steps,
        "curves": curves,
        "learning_rates": list(prefix),
        "attempts": attempts,
        "successful_updates": successful,
        "terminal_status": terminal,
    }


def _numerical_recovery_result() -> dict[str, object]:
    rates = exponential_learning_rates(start=1e-5, stop=3e-3, update_count=106)

    def committed(count: int) -> list[dict[str, object]]:
        return [
            {
                "attempt": index,
                "learning_rate": rates[index - 1],
                "losses": dict.fromkeys(BRANCH_NAMES, 1.0),
            }
            for index in range(1, count + 1)
        ]

    parameter = {
        "name": "head.weight",
        "dtype": "torch.float32",
        "missing": False,
        "nonfinite_count": 1,
        "max_finite_abs": 2.0,
    }
    diagnostics = {
        "kind": "gradient",
        "first_affected": {"branch": "normal_vae", **copy.deepcopy(parameter)},
        "branches": {
            name: {
                "scaler": 32768.0,
                "loss_weight": 1.0,
                "parameters": [copy.deepcopy(parameter)],
            }
            for name in BRANCH_NAMES
        },
        "input": {
            "logical_sample": {
                "wsi_id": 65094,
                "diagnosis_label": "MC",
                "diagnosis_index": 4,
                "instance_count": 5773,
                "class_weight": 2.65,
                "checkpoint_chunk_size": None,
            },
        },
    }
    identity = V1_MIL_INITIALIZATION_SHA256
    order = V1_MIL_ORDER_SHA256
    return {
        "status": "pass",
        "expected_failure": {
            "attempt": 12,
            "wsi_id": 65094,
            "diagnosis_label": "MC",
            "instance_count": 5773,
            "class_weight": 2.65,
            "grad_scaler_scale": 32768.0,
        },
        "v1_contract": {
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
        "legacy_scaled_loss_replay": {
            "weight_application": "scaled_loss",
            "initialization_sha256": identity,
            "order_sha256": order,
            "successful_updates": 11,
            "committed": committed(11),
            "failure": {
                "attempt": 12,
                "learning_rate": rates[11],
                "wsi_id": 65094,
                "diagnosis_label": "MC",
                "diagnosis_index": 4,
                "instance_count": 5773,
                "error": "missing/nonfinite gradient",
                "diagnostics": diagnostics,
            },
        },
        "post_unscale_replay": {
            "weight_application": "post_grad_scaler_unscale",
            "initialization_sha256": identity,
            "order_sha256": order,
            "successful_updates": 12,
            "committed": committed(12),
            "failure": None,
        },
        "replay_state_reused_by_sweep": False,
        "checkpoint_chunk_size": None,
    }


def _expected_sweep_provenance() -> dict[str, object]:
    return {
        "spec_sha256": "c" * 64,
        "supervised_manifest_audit_sha256": "d" * 64,
        "input_source_tree_sha256": "e" * 64,
        "source_records": [
            {
                "kaggle_source": "source",
                "binaries": [{"name": "normal.bin", "bytes": 10, "sha256": "f" * 64}],
            },
        ],
    }


def _sweep_config_provenance() -> dict[str, object]:
    receipt = {
        "schema_version": "spec0023.supervised_calibration_input_receipt.v1",
        "package_mode": "sweep",
        "dataset_reference": DATASET_SLUGS["sweep"],
        "dataset_version": 1,
        "input_contract_sha256": "a" * 64,
        "remote_files": [
            {
                "logical_name": CONTRACT_NAME,
                "bytes": 10,
                "sha256": "b" * 64,
            },
        ],
    }
    expected = _expected_sweep_provenance()
    return {
        **expected,
        "input_dataset_receipt": receipt,
        "logical_manifest_sha256": {
            "physical_parts.csv": "1" * 64,
            "wsi/wsi_cancer_train_instances.csv": "2" * 64,
            "wsi/wsi_cancer_train_bags.csv": "3" * 64,
            "tissue/tissue_train_5671_per_class.csv": "4" * 64,
        },
        "kernel_sources": ["source"],
        "output_allowlist": ["spec0023_supervised_calibration_audit.json"],
    }


def test_exact_sweeps_and_agreed_warmups() -> None:
    """Pin exact endpoints and first-positive 10%-of-epoch confirmation LRs."""
    for steps in (106, 132):
        rates = exponential_learning_rates(start=1e-5, stop=3e-3, update_count=steps)
        assert len(rates) == steps
        assert rates[0] == pytest.approx(1e-5)
        assert rates[-1] == pytest.approx(3e-3)
        assert all(left < right for left, right in pairwise(rates))

    assert warmup_update_count(106) == 11
    assert warmup_update_count(132) == 14
    assert confirmation_learning_rate(
        peak=1e-3,
        successful_update=1,
        steps_per_epoch=106,
    ) == pytest.approx(1e-3 / 11)
    assert confirmation_learning_rate(
        peak=1e-3,
        successful_update=11,
        steps_per_epoch=106,
    ) == pytest.approx(1e-3)


def test_package_assets_exclude_test_and_bind_one_shared_peak_per_task() -> None:
    """Fail closed on leakage and representation-specific peak selection."""
    sweep = _package_assets(confirmation=False)
    _validate_assets(sweep, "sweep")
    assert not any("test" in name for name in sweep)

    confirmation = _package_assets(confirmation=True)
    _validate_assets(confirmation, "confirmation")
    assert not any("test" in name for name in confirmation)
    selection = _read_selection_audit(confirmation["selection_audit.json"])
    assert selection["shared_peaks"] == {"mil": 3e-4, "tissue": 5e-4}

    invalid = json.loads(confirmation["selection_audit.json"])
    invalid["mil"] = {
        "normal_peak": 3e-4,
        "so2_peak": 2e-4,
        "rationale": "invalid",
    }
    with pytest.raises(ValueError, match="one shared peak"):
        _read_selection_audit(json.dumps(invalid).encode())

    train_rows = [{"wsi_id": "1", "part": "1", "file_index": "4"}]
    validation_rows = [{"wsi_id": "2", "part": "1", "file_index": "4"}]
    with pytest.raises(ValueError, match="overlap"):
        _validate_disjoint_rows(train_rows, validation_rows, label="fixture")


def test_mil_horizon_is_only_the_exact_epoch_two_to_five_continuation() -> None:
    """Pin the one changed value without reopening calibration or tissue training."""
    assert (HORIZON_START_EPOCH, HORIZON_TARGET_EPOCH, HORIZON_PEAK) == (2, 5, 2e-4)
    assert set(HORIZON_AUTHORITY_SHA256) == {
        "resume/manifest.json",
        "resume/progress.json",
        "resume/paired_checkpoint.pt",
        "resume/confirmation_v1_audit.json",
        "resume/confirmation_v1_config.json",
    }
    assert DATASET_SLUGS["horizon"].endswith("mil-horizon-inputs")
    template = Path(
        "kaggle/kernels/ubc_ocean_supervised_calibration/run_template.py",
    ).read_text(encoding="utf-8")
    tree = ast.parse(template)
    wrapper_authority = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "HORIZON_AUTHORITY_SHA256"
            for target in node.targets
        )
    )
    assert wrapper_authority == HORIZON_AUTHORITY_SHA256
    horizon = template[
        template.index("def _run_mil_horizon(") : template.index(
            "def _validate_horizon_authority(",
        )
    ]
    assert "for epoch in range(2, 5):" in horizon
    assert '"continuation_updates": 318' in horizon
    assert '"warmup_repeated": False' in horizon
    assert '"frozen_tasks": ["tissue"]' in horizon
    assert '"checkpoint_chunk_size": None' in horizon
    assert "confirmation_learning_rate" not in horizon
    assert '"status": "complete"' in horizon


def test_width128_is_one_fresh_five_epoch_architecture_change() -> None:
    """Pin the narrow scorer-width diagnostic without opening tissue or test data."""
    assert WIDTH128_BASELINE_AUDIT_SHA256 == (
        "2edeb5d20471ebc559e3bb7f6f7cde4aeaa06723b98bdcfc28a7dca4039525fb"
    )
    template = Path(
        "kaggle/kernels/ubc_ocean_supervised_calibration/run_template.py",
    ).read_text(encoding="utf-8")
    width = template[
        template.index("def _run_mil_width128(") : template.index(
            "def _width128_observations(",
        )
    ]
    assert '"attention_dim": 128' in width
    assert '"fresh_initialization": True' in width
    assert "peak=2e-4" in width
    assert "epochs=5" in width
    assert '"frozen_tasks": ["tissue"]' in width
    assert '"checkpoint_chunk_size": None' in width
    assert '"status": "failed" if numerical_failure else "complete"' in width
    assert 'attempt.get("attention_dim") != 128' in width
    assert "resume" not in width
    context = template[
        template.index("def _task_context(") : template.index(
            "def _task_order(",
        )
    ]
    assert "make_attention_mil_width_variant" in context
    assert "observed_shapes != expected_shapes" in context


def test_class_specific_is_one_fresh_five_epoch_architecture_change() -> None:
    """Pin five full-bag maps without reopening tissue, test, LR, or width search."""
    template = Path(
        "kaggle/kernels/ubc_ocean_supervised_calibration/run_template.py",
    ).read_text(encoding="utf-8")
    stage = template[
        template.index("def _run_mil_class_specific(") : template.index(
            "def _class_specific_observations(",
        )
    ]
    assert '"attention_dim": 128' in stage
    assert '"attention_maps": 5' in stage
    assert '"function_preserving_initialization": True' in stage
    assert "peak=2e-4" in stage
    assert "epochs=5" in stage
    assert '"frozen_tasks": ["tissue"]' in stage
    assert '"checkpoint_chunk_size": None' in stage
    assert 'attempt.get("class_specific_attention") is not True' in stage
    assert "resume" not in stage
    context = template[
        template.index("def _task_context(") : template.index(
            "def _task_order(",
        )
    ]
    assert "make_class_specific_attention_mil_variant" in context
    assert "attention_maps = 5 if class_specific_attention else 1" in context


def test_class_specific_scale_fix_is_one_paired_amp_correction() -> None:
    """Permit one paired scaler backoff without changing the experiment surface."""
    assert DATASET_SLUGS["class_specific_scale_fix"] == (
        "maximusshtefan/eqvae-ubc-ocean-mil-class-scale-inputs"
    )
    assert INPUT_RECEIPT_FILENAMES["class_specific_scale_fix"] == (
        "mil_class_scale_input_dataset_receipt.json"
    )
    metadata = _metadata(
        "class_specific_scale_fix",
        [{"kaggle_source": "owner/source"}],
    )
    assert metadata["id"] == "maximusshtefan/eqvae-ubc-ocean-mil-class-scale"
    assert metadata["dataset_sources"] == [DATASET_SLUGS["class_specific_scale_fix"]]

    template = Path(
        "kaggle/kernels/ubc_ocean_supervised_calibration/run_template.py",
    ).read_text(encoding="utf-8")
    stage = template[
        template.index("def _run_mil_class_specific(") : template.index(
            "def _class_specific_observations(",
        )
    ]
    assert 'config.get("package_mode") == "class_specific_scale_fix"' in stage
    assert '"maximum_backoffs": 1' in stage
    assert '"initial_scale": 32768' in stage
    assert '"backoff_factor": 0.5' in stage
    assert '"retry_same_loaded_bag": True' in stage
    assert '"retry_same_learning_rate": True' in stage
    assert '"advance_successful_cursor_on_discard": False' in stage
    assert '"raw_gradient_check": "after_unscale_before_class_weight"' in stage
    assert '"weighted_gradient_check": "terminal_after_fp32_class_weight"' in stage
    assert (
        'phase="class_specific_scale_fix" if scale_fix else "class_specific"' in stage
    )
    assert '"frozen_tasks": ["tissue"]' in stage
    assert '"checkpoint_chunk_size": None' in stage
    assert "resume" not in stage

    confirmation = template[
        template.index("def _run_confirmation_task(") : template.index(
            "def _patience_state(",
        )
    ]
    assert 'scale_backoff_budget = 1 if phase == "class_specific_scale_fix" else 0' in (
        confirmation
    )
    assert "scale_backoff_budget - len(scale_backoff_events)" in confirmation
    assert '"scale_backoff_budget": scale_backoff_budget' in confirmation
    assert '"scale_backoffs_used": len(scale_backoff_events)' in confirmation
    assert '"scale_backoffs_remaining": (' in confirmation
    assert '"scale_backoff_events": scale_backoff_events' in confirmation


def test_class_specific_predecessor_audit_is_fail_closed(tmp_path: Path) -> None:
    """Require the exact completed width-128 evidence before changing aggregation."""
    manifests: dict[str, str] = {
        "physical_parts.csv": "a" * 64,
        "wsi/wsi_cancer_train_instances.csv": "b" * 64,
        "wsi/wsi_cancer_train_bags.csv": "c" * 64,
        "wsi/wsi_cancer_validation_instances.csv": "d" * 64,
        "wsi/wsi_cancer_validation_bags.csv": "e" * 64,
    }
    audit: dict[str, object] = {
        "schema_version": "spec0023.mil_width128_audit.v1",
        "package_mode": "width128",
        "status": "complete",
        "scientific_result": "insufficient_learning",
        "config_sha256": CLASS_SPECIFIC_BASELINE_CONFIG_SHA256,
        "peak": 2e-4,
        "epochs": 5,
        "attention_dim": 128,
        "logical_access": "train_and_validation_only",
        "frozen_tasks": ["tissue"],
        "checkpoint_chunk_size": None,
        "task_result": {
            "status": "insufficient_learning",
            "initialization_sha256": WIDTH128_INITIALIZATION_SHA256,
            "steps_per_epoch": 106,
            "warmup_updates": 11,
            "zero_scaler_skips": True,
            "logical_manifest_sha256": manifests,
            "histories": {name: [{}] * 530 for name in BRANCH_NAMES},
            "validation_history": {name: [{}] * 10 for name in BRANCH_NAMES},
        },
        "learning_observations": {
            "normal_vae": {"best_validation": {"macro_f1": 0.2439628482972136}},
            "so2_vae": {"best_validation": {"macro_f1": 0.24643962848297213}},
        },
    }
    path = tmp_path / "width128.json"
    path.write_text(json.dumps(audit), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    result = _validate_width128_baseline_audit(
        path,
        manifest_hashes=manifests,
        expected_sha256=digest,
    )
    assert result["audit_sha256"] == digest
    assert result["class_specific_initialization_sha256"] == (
        CLASS_SPECIFIC_INITIALIZATION_SHA256
    )
    task_result = cast("dict[str, object]", audit["task_result"])
    task_result["initialization_sha256"] = "0" * 64
    path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError, match="predecessor audit differs"):
        _validate_width128_baseline_audit(
            path,
            manifest_hashes=manifests,
            expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )


def test_historical_sweep_uses_its_sealed_spec_preimage() -> None:
    """Keep later live-spec status edits distinct from immutable sweep evidence."""
    sealed = "a" * 64
    assert _contract_spec_sha256({"spec_sha256": sealed}) == sealed
    for invalid in (None, "A" * 64, "0" * 63):
        with pytest.raises(ValueError, match="valid spec SHA-256"):
            _contract_spec_sha256({"spec_sha256": invalid})


def test_confirmation_requires_the_actual_sweep_audit(  # noqa: PLR0914, PLR0915
    tmp_path: Path,
) -> None:
    """Reject an invented digest even when it is syntactically valid SHA-256."""
    config: dict[str, object] = {
        "schema_version": "spec0023.supervised_calibration_config.v1",
        "package_mode": "sweep",
        **_sweep_config_provenance(),
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
            "stop_learning_rate": 3e-3,
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
    config_path = tmp_path / "sweep_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    sweep: dict[str, object] = {
        "schema_version": "spec0023.supervised_calibration_audit.v1",
        "package_mode": "sweep",
        "status": "selection_ready",
        "selection_status": "human_selection_required_before_confirmation",
        "numerical_recovery": _numerical_recovery_result(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "task_results": {
            task: _sweep_task_result(task, steps)
            for task, steps in (("mil", 106), ("tissue", 132))
        },
    }
    path = tmp_path / "sweep.json"
    path.write_text(json.dumps(sweep), encoding="utf-8")
    with pytest.raises(ValueError, match="differs"):
        _validate_sweep_audit(
            path,
            expected_sha256="c" * 64,
            sweep_config_path=config_path,
            expected_provenance=_expected_sweep_provenance(),
        )
    audit_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    _validate_sweep_audit(
        path,
        expected_sha256=audit_sha256,
        sweep_config_path=config_path,
        expected_provenance=_expected_sweep_provenance(),
    )
    invalid_recoveries: list[dict[str, object]] = []
    missing = copy.deepcopy(sweep)
    missing.pop("numerical_recovery")
    invalid_recoveries.append(missing)
    failed = copy.deepcopy(sweep)
    cast("dict[str, object]", failed["numerical_recovery"])["status"] = "failed"
    invalid_recoveries.append(failed)
    mismatched = copy.deepcopy(sweep)
    corrected = cast(
        "dict[str, object]",
        cast("dict[str, object]", mismatched["numerical_recovery"])[
            "post_unscale_replay"
        ],
    )
    corrected["order_sha256"] = "f" * 64
    invalid_recoveries.append(mismatched)
    for invalid_recovery in invalid_recoveries:
        path.write_text(json.dumps(invalid_recovery), encoding="utf-8")
        invalid_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        with pytest.raises((TypeError, ValueError), match="numerical recovery"):
            _validate_sweep_audit(
                path,
                expected_sha256=invalid_sha256,
                sweep_config_path=config_path,
                expected_provenance=_expected_sweep_provenance(),
            )
    path.write_text(json.dumps(sweep), encoding="utf-8")
    invalid_provenance_configs: list[dict[str, object]] = []
    missing_receipt = copy.deepcopy(config)
    missing_receipt.pop("input_dataset_receipt")
    invalid_provenance_configs.append(missing_receipt)
    stale_receipt = copy.deepcopy(config)
    cast("dict[str, object]", stale_receipt["input_dataset_receipt"])[
        "dataset_reference"
    ] = "maximusshtefan/eqvae-ubc-ocean-supcal-sweep-inputs"
    invalid_provenance_configs.append(stale_receipt)
    for hash_name in (
        "spec_sha256",
        "supervised_manifest_audit_sha256",
        "input_source_tree_sha256",
    ):
        stale_hash = copy.deepcopy(config)
        stale_hash[hash_name] = "0" * 64
        invalid_provenance_configs.append(stale_hash)
    for binary_field, replacement in (
        ("name", "wrong.bin"),
        ("bytes", 11),
        ("sha256", "0" * 64),
    ):
        stale_binary = copy.deepcopy(config)
        records = cast("list[dict[str, object]]", stale_binary["source_records"])
        binaries = cast("list[dict[str, object]]", records[0]["binaries"])
        binaries[0][binary_field] = replacement
        invalid_provenance_configs.append(stale_binary)
    stale_source = copy.deepcopy(config)
    stale_source_records = cast(
        "list[dict[str, object]]",
        stale_source["source_records"],
    )
    stale_source_records[0]["kaggle_source"] = "wrong/source"
    cast("list[object]", stale_source["kernel_sources"])[0] = "wrong/source"
    invalid_provenance_configs.append(stale_source)
    for invalid_config in invalid_provenance_configs:
        config_path.write_text(json.dumps(invalid_config), encoding="utf-8")
        invalid_audit = copy.deepcopy(sweep)
        invalid_audit["config_sha256"] = hashlib.sha256(
            config_path.read_bytes(),
        ).hexdigest()
        path.write_text(json.dumps(invalid_audit), encoding="utf-8")
        invalid_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        with pytest.raises(ValueError, match="provenance"):
            _validate_sweep_audit(
                path,
                expected_sha256=invalid_sha256,
                sweep_config_path=config_path,
                expected_provenance=_expected_sweep_provenance(),
            )
    config_path.write_text(json.dumps(config), encoding="utf-8")
    path.write_text(json.dumps(sweep), encoding="utf-8")
    config["grad_scaler_init_scale"] = 65_536
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="configuration differs"):
        _validate_sweep_audit(
            path,
            expected_sha256=audit_sha256,
            sweep_config_path=config_path,
            expected_provenance=_expected_sweep_provenance(),
        )


def test_sweep_selection_requires_aligned_prefix_and_respects_failure_bound(
    tmp_path: Path,
) -> None:
    """Block confirmation when curves are unusable or a peak crosses failure."""
    config = {
        "schema_version": "spec0023.supervised_calibration_config.v1",
        "package_mode": "sweep",
        **_sweep_config_provenance(),
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
            "stop_learning_rate": 3e-3,
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
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    def write_audit(mil: dict[str, object]) -> tuple[Path, str]:
        audit = {
            "schema_version": "spec0023.supervised_calibration_audit.v1",
            "package_mode": "sweep",
            "status": "selection_ready",
            "selection_status": "human_selection_required_before_confirmation",
            "numerical_recovery": _numerical_recovery_result(),
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "task_results": {
                "mil": mil,
                "tissue": _sweep_task_result("tissue", 132),
            },
        }
        path = tmp_path / "audit.json"
        path.write_text(json.dumps(audit), encoding="utf-8")
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    for successful in (0, 9):
        path, digest = write_audit(
            _sweep_task_result(
                "mil",
                106,
                successful=successful,
                terminal="numerical_failure:nonfinite",
            ),
        )
        with pytest.raises(ValueError, match="usable aligned curve"):
            _validate_sweep_audit(
                path,
                expected_sha256=digest,
                sweep_config_path=config_path,
                expected_provenance=_expected_sweep_provenance(),
            )

    invalid_ewma = _sweep_task_result("mil", 106)
    raw_curves = invalid_ewma["curves"]
    assert isinstance(raw_curves, dict)
    raw_normal = cast("object", raw_curves["normal_vae"])
    assert isinstance(raw_normal, list)
    raw_point = cast("object", raw_normal[4])
    assert isinstance(raw_point, dict)
    raw_point["ewma"] = 123.0
    path, digest = write_audit(invalid_ewma)
    with pytest.raises(ValueError, match="EWMA recurrence"):
        _validate_sweep_audit(
            path,
            expected_sha256=digest,
            sweep_config_path=config_path,
            expected_provenance=_expected_sweep_provenance(),
        )

    false_divergence = _sweep_task_result("mil", 106, successful=20)
    false_divergence["terminal_status"] = "diverged"
    path, digest = write_audit(false_divergence)
    with pytest.raises(ValueError, match="Diverged sweep"):
        _validate_sweep_audit(
            path,
            expected_sha256=digest,
            sweep_config_path=config_path,
            expected_provenance=_expected_sweep_provenance(),
        )

    path, digest = write_audit(
        _sweep_task_result("mil", 106, successful=20, terminal="diverged"),
    )
    evidence = _validate_sweep_audit(
        path,
        expected_sha256=digest,
        sweep_config_path=config_path,
        expected_provenance=_expected_sweep_provenance(),
    )
    rates = exponential_learning_rates(start=1e-5, stop=3e-3, update_count=106)
    maximum_lrs = evidence["maximum_shared_lrs"]
    assert isinstance(maximum_lrs, dict)
    assert maximum_lrs["mil"] == pytest.approx(rates[18])
    with pytest.raises(ValueError, match="exceeds"):
        _validate_selection_bounds(
            {"shared_peaks": {"mil": rates[19], "tissue": 1e-4}},
            evidence,
        )


def test_package_inputs_must_match_the_canonical_manifest_audit(
    tmp_path: Path,
) -> None:
    """Reject relabelled or substituted learning CSV bytes before packaging."""
    assets = _package_assets(confirmation=False)
    hashes = {
        name: hashlib.sha256(payload).hexdigest() for name, payload in assets.items()
    }
    audit = {
        "schema_version": "spec0023.supervised_manifest_audit.v1",
        "status": "complete",
        "physical_catalog": {"sha256": hashes["physical_parts.csv"]},
        "wsi_files": {
            "wsi_cancer_train_instances.csv": {
                "sha256": hashes["wsi/wsi_cancer_train_instances.csv"],
            },
            "wsi_cancer_train_bags.csv": {
                "sha256": hashes["wsi/wsi_cancer_train_bags.csv"],
            },
        },
        "tissue_files": {
            "tissue_train_5671_per_class.csv": {
                "sha256": hashes["tissue/tissue_train_5671_per_class.csv"],
            },
        },
    }
    path = tmp_path / "manifest_audit.json"
    path.write_text(json.dumps(audit), encoding="utf-8")
    _validate_manifest_audit(path, hashes, package_mode="sweep")
    hashes["wsi/wsi_cancer_train_bags.csv"] = "0" * 64
    with pytest.raises(ValueError, match="canonical"):
        _validate_manifest_audit(path, hashes, package_mode="sweep")
    assert confirmation_learning_rate(
        peak=1e-3,
        successful_update=12,
        steps_per_epoch=106,
    ) == pytest.approx(1e-3)


def test_private_input_dataset_pins_full_train_bytes_and_small_wrapper(  # noqa: PLR0914, PLR0915
    tmp_path: Path,
) -> None:
    """Keep canonical manifests out of run.py while sealing their remote bytes."""
    repo_root = Path(__file__).parents[1]
    manifest_root = tmp_path / "manifests"
    assets = _package_assets(confirmation=False)
    for name, payload in assets.items():
        path = manifest_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    hashes = {
        name: hashlib.sha256(payload).hexdigest() for name, payload in assets.items()
    }
    audit = {
        "schema_version": "spec0023.supervised_manifest_audit.v1",
        "status": "complete",
        "physical_catalog": {"sha256": hashes["physical_parts.csv"]},
        "wsi_files": {
            "wsi_cancer_train_instances.csv": {
                "sha256": hashes["wsi/wsi_cancer_train_instances.csv"],
            },
            "wsi_cancer_train_bags.csv": {
                "sha256": hashes["wsi/wsi_cancer_train_bags.csv"],
            },
        },
        "tissue_files": {
            "tissue_train_5671_per_class.csv": {
                "sha256": hashes["tissue/tissue_train_5671_per_class.csv"],
            },
        },
    }
    (manifest_root / "spec0023_supervised_manifest_audit.json").write_text(
        json.dumps(audit),
        encoding="utf-8",
    )
    bundle = tmp_path / "bundle"
    contract = build_input_bundle(
        repo_root=repo_root,
        manifest_root=manifest_root,
        output_root=bundle,
        package_mode="sweep",
    )
    validate_input_bundle(
        repo_root=repo_root,
        manifest_root=manifest_root,
        bundle_root=bundle,
        package_mode="sweep",
    )
    assert contract["logical_manifest_sha256"] == hashes
    assert not any("validation" in name or "test" in name for name in assets)

    envelope = stage_upload_envelope(
        bundle_root=bundle,
        destination=tmp_path / "envelope",
    )
    with zipfile.ZipFile(envelope / "bundle.zip") as archive:
        names = set(archive.namelist())
    assert CONTRACT_NAME in names
    assert METADATA_NAME not in names
    assert "wsi/wsi_cancer_train_instances.csv" in names

    downloaded = tmp_path / "downloaded"
    for path in bundle.rglob("*"):
        if path.is_file() and path.name != METADATA_NAME:
            target = downloaded / path.relative_to(bundle)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    remote_metadata = tmp_path / "remote_metadata.json"
    remote_metadata.write_text(
        json.dumps({
            "info": {
                "ownerUser": "maximusshtefan",
                "datasetSlug": DATASET_SLUGS["sweep"].split("/", 1)[1],
                "isPrivate": True,
            },
        }),
        encoding="utf-8",
    )
    receipt_path = tmp_path / "receipt.json"
    receipt = seal_remote_receipt(
        bundle_root=bundle,
        downloaded_root=downloaded,
        remote_metadata_path=remote_metadata,
        dataset_version=1,
        output_path=receipt_path,
        package_mode="sweep",
    )
    manifest_audit_path = manifest_root / "spec0023_supervised_manifest_audit.json"
    assert (
        _validate_input_receipt(
            repo_root=repo_root,
            bundle_root=bundle,
            receipt_path=receipt_path,
            package_mode="sweep",
            expected_spec_sha256=hashlib.sha256(
                (
                    repo_root
                    / "docs/specs/0023-matched-supervised-latent-evaluation.md"
                ).read_bytes(),
            ).hexdigest(),
            expected_manifest_audit_sha256=hashlib.sha256(
                manifest_audit_path.read_bytes(),
            ).hexdigest(),
        )
        == receipt
    )

    changed_repo = tmp_path / "changed_repo"
    shutil.copytree(repo_root / "src/eqvae", changed_repo / "src/eqvae")
    changed_spec = (
        changed_repo / "docs/specs/0023-matched-supervised-latent-evaluation.md"
    )
    changed_spec.parent.mkdir(parents=True)
    shutil.copy2(
        repo_root / "docs/specs/0023-matched-supervised-latent-evaluation.md",
        changed_spec,
    )
    changed_source = changed_repo / "src/eqvae/__init__.py"
    changed_source.write_text(
        f"{changed_source.read_text(encoding='utf-8')}\n# stale-input test\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="source tree differs"):
        _validate_input_receipt(
            repo_root=changed_repo,
            bundle_root=bundle,
            receipt_path=receipt_path,
            package_mode="sweep",
            expected_spec_sha256=hashlib.sha256(changed_spec.read_bytes()).hexdigest(),
            expected_manifest_audit_sha256=hashlib.sha256(
                manifest_audit_path.read_bytes(),
            ).hexdigest(),
        )
    assert (
        _validate_input_receipt(
            repo_root=changed_repo,
            bundle_root=bundle,
            receipt_path=receipt_path,
            package_mode="sweep",
            expected_spec_sha256=hashlib.sha256(changed_spec.read_bytes()).hexdigest(),
            expected_manifest_audit_sha256=hashlib.sha256(
                manifest_audit_path.read_bytes(),
            ).hexdigest(),
            require_current_source_tree=False,
        )
        == receipt
    )

    package = tmp_path / "package"
    build_calibration_package(
        repo_root=repo_root,
        manifest_root=manifest_root,
        output_root=package,
        package_mode="sweep",
        input_bundle_root=bundle,
        input_receipt_path=receipt_path,
    )
    assert (package / "run.py").stat().st_size < KAGGLE_SCRIPT_LIMIT_BYTES
    generated = (package / "run.py").read_text(encoding="utf-8")
    assert "_run_mil_numerical_recovery" in generated
    assert "_mil_recovery_passes" in generated
    assert "has_nonfinite_gradient" in generated
    assert 'result["runtime_stack"] = _runtime_stack()' in generated
    for runtime_field in ("torch", "torch_cuda", "cuda_device_names"):
        assert f'"{runtime_field}"' in generated
    assert 'weight_application="post_grad_scaler_unscale"' in generated
    assert "checkpoint_chunk_size=None" in generated
    config = json.loads(
        (package / "spec0023_supervised_calibration_config.json").read_text(),
    )
    assert config["numerical_recovery"] == {
        "v1_sweep_audit_sha256": V1_SWEEP_AUDIT_SHA256,
        "legacy_initialization_sha256": V1_MIL_INITIALIZATION_SHA256,
        "legacy_order_sha256": V1_MIL_ORDER_SHA256,
        "legacy_failure_attempt": 12,
        "legacy_failure_wsi_id": 65094,
        "legacy_class_weight": 2.65,
        "replacement": "post_grad_scaler_unscale",
        "replacement_replay_updates": 12,
        "replay_state_reused_by_sweep": False,
    }
    assert config["tasks"]["mil"]["class_weight_application"] == (
        "post_grad_scaler_unscale"
    )
    metadata = json.loads((package / "kernel-metadata.json").read_text())
    assert metadata["dataset_sources"] == [DATASET_SLUGS["sweep"]]


def test_shell_routes_both_input_modes_and_guards_confirmation() -> None:
    """Keep all dataset actions mode-aware so confirmation cannot omit its audit."""
    repo_root = Path(__file__).parents[1]
    script = repo_root / "scripts/kaggle_kernel.sh"
    source = script.read_text(encoding="utf-8")
    for action in ("build", "publish", "verify"):
        assert f"{action}-supervised-calibration-input)" in source
        assert f'{action}_supervised_calibration_input "${{2:-}}" "${{3:-}}"' in source
        assert f"{action}-supervised-calibration-sweep-input)" not in source
    assert source.count("confirmation input requires its selection audit") == 2
    assert INPUT_RECEIPT_FILENAMES["sweep"] in source
    assert INPUT_RECEIPT_FILENAMES["class_specific"] in source
    assert "sweep_input_dataset_receipt.json" not in source


def test_kaggle_dataset_id_and_title_fit_remote_limits() -> None:
    """Keep both private input modes within Kaggle's enforced 6-50 character fields."""
    for mode, reference in DATASET_SLUGS.items():
        slug = reference.split("/", maxsplit=1)[1]
        title = _dataset_metadata(mode)["title"]
        assert isinstance(title, str)
        assert 6 <= len(slug) <= 50
        assert 6 <= len(title) <= 50


def test_kaggle_kernel_id_and_title_fit_remote_limits() -> None:
    """Keep both compact calibration kernels within Kaggle's field limits."""
    for mode in (
        "sweep",
        "confirmation",
        "horizon",
        "width128",
        "class_specific",
        "class_specific_scale_fix",
    ):
        metadata = _metadata(mode, [{"kaggle_source": "owner/source"}])
        slug = cast("str", metadata["id"]).split("/", maxsplit=1)[1]
        title = cast("str", metadata["title"])
        assert 6 <= len(slug) <= 50
        assert 6 <= len(title) <= 50


def test_revised_confirmation_runs_only_mil_and_keeps_tissue_frozen() -> None:
    """Keep the authorized retry to one LR-only MIL change without rerunning tissue."""
    repo_root = Path(__file__).parents[1]
    builder = (
        repo_root / "src/eqvae/cli/build_ubc_supervised_calibration.py"
    ).read_text(encoding="utf-8")
    runner = (
        repo_root / "kaggle/kernels/ubc_ocean_supervised_calibration/run_template.py"
    ).read_text(encoding="utf-8")
    assert '"active_tasks": ["mil"]' in builder
    assert '"frozen_tasks": ["tissue"]' in builder
    assert '"mil",\n            "spec0023_supervised_calibration_audit.json"' in builder
    assert "for task in active_tasks:" in runner
    assert 'active_tasks != ["mil"] or frozen_tasks != ["tissue"]' in runner


def test_full_schedule_uses_each_configuration_epoch_and_exact_endpoint() -> None:
    """Reuse one peak while deriving warmup/cosine geometry from each subset."""
    for steps in (5, 11, 23, 58, 106, 132):
        warmup = warmup_update_count(steps)
        assert full_training_learning_rate(
            peak=2e-3,
            successful_update=1,
            steps_per_epoch=steps,
        ) == pytest.approx(2e-3 / warmup)
        assert full_training_learning_rate(
            peak=2e-3,
            successful_update=warmup,
            steps_per_epoch=steps,
        ) == pytest.approx(2e-3)
        assert full_training_learning_rate(
            peak=2e-3,
            successful_update=30 * steps,
            steps_per_epoch=steps,
        ) == pytest.approx(2e-5)
    assert half_epoch_boundaries(106) == (53, 106)
    assert half_epoch_boundaries(23) == (11, 23)


def test_nonfinite_branch_commits_neither_parameter_set() -> None:
    """Prove a one-sided failure cannot advance the other representation."""
    branches = _cpu_branches()
    before = {
        name: copy.deepcopy(branches[name].model.state_dict()) for name in BRANCH_NAMES
    }
    x = torch.tensor([[1.0, 2.0]])
    with pytest.raises(PairedNumericalError, match="nonfinite"):
        paired_atomic_step(
            branches,
            {
                "normal_vae": lambda: branches["normal_vae"].model(x).sum(),
                "so2_vae": lambda: (
                    branches["so2_vae"].model(x).sum() * torch.tensor(float("nan"))
                ),
            },
            learning_rate=1e-3,
        )
    for name in BRANCH_NAMES:
        observed = branches[name].model.state_dict()
        for key, value in before[name].items():
            torch.testing.assert_close(observed[key], value)


def test_paired_scale_backoff_retries_same_losses_and_commits_once() -> None:
    """A raw AMP overflow halves both scales without splitting paired progress."""
    branches = _cpu_branches()
    for branch in branches.values():
        branch.scaler = torch.amp.GradScaler(
            "cpu",
            init_scale=32_768.0,
            growth_interval=1_000_000,
        )
    before = {
        name: copy.deepcopy(branches[name].model.state_dict()) for name in BRANCH_NAMES
    }
    calls: dict[str, int] = dict.fromkeys(BRANCH_NAMES, 0)
    x = torch.tensor([[1.0, 2.0]])

    def loss_for(name: str) -> torch.Tensor:
        calls[name] += 1
        loss = branches[name].model(x).sum()
        return (
            _loss_that_overflows_only_at_large_scale(loss)
            if name == "normal_vae"
            else loss
        )

    result = paired_atomic_step(
        branches,
        {name: lambda name=name: loss_for(name) for name in BRANCH_NAMES},
        learning_rate=1e-3,
        max_scale_backoffs=1,
    )

    assert calls == dict.fromkeys(BRANCH_NAMES, 2)
    assert result.scale_backoffs == 1
    assert result.scales_before == dict.fromkeys(BRANCH_NAMES, 32_768.0)
    assert result.scales_after == dict.fromkeys(BRANCH_NAMES, 16_384.0)
    for name in BRANCH_NAMES:
        observed = branches[name].model.state_dict()
        assert any(
            not torch.equal(observed[key], value) for key, value in before[name].items()
        )
        for state in branches[name].optimizer.state.values():
            assert int(cast("torch.Tensor", state["step"]).item()) == 1
    for normal, so2 in zip(
        branches["normal_vae"].model.parameters(),
        branches["so2_vae"].model.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(normal, so2)


def test_class_weight_overflow_is_not_retried_as_amp_overflow() -> None:
    """Loss-scale backoff cannot repair FP32 overflow introduced after unscale."""
    branches = _cpu_branches()
    calls: dict[str, int] = dict.fromkeys(BRANCH_NAMES, 0)
    x = torch.tensor([[1.0, 2.0]])

    def loss_for(name: str) -> torch.Tensor:
        calls[name] += 1
        return branches[name].model(x).sum()

    with pytest.raises(PairedNumericalError) as caught:
        paired_atomic_step(
            branches,
            {name: lambda name=name: loss_for(name) for name in BRANCH_NAMES},
            learning_rate=1e-3,
            loss_weights=dict.fromkeys(BRANCH_NAMES, 3e38),
            max_scale_backoffs=1,
        )
    assert caught.value.details["kind"] == "weighted_gradient"
    assert calls == dict.fromkeys(BRANCH_NAMES, 1)


def test_supervised_adamw_decays_only_matrix_like_parameters() -> None:
    """Apply the VAE matrix-only decay rule to both supervised architectures."""
    for model in (AttentionMILClassifier(), TissueClassifier()):
        branch = make_branch_state(model, device=torch.device("cpu"))
        groups = {group["name"]: group for group in branch.optimizer.param_groups}
        assert set(groups) == {"decay", "no_decay"}
        assert groups["decay"]["weight_decay"] == pytest.approx(1e-4)
        assert groups["no_decay"]["weight_decay"] == pytest.approx(0.0)
        decay_ids = {id(parameter) for parameter in groups["decay"]["params"]}
        no_decay_ids = {id(parameter) for parameter in groups["no_decay"]["params"]}
        for parameter in model.parameters():
            expected = decay_ids if parameter.ndim >= 2 else no_decay_ids
            assert id(parameter) in expected
            assert id(parameter) not in (
                no_decay_ids if expected is decay_ids else decay_ids
            )


def test_weighted_gradients_match_the_weighted_objective_after_amp_unscale() -> None:
    """Keep weighted MIL optimization exact without amplifying FP16 backward signals.

    The gradient multiplier must be mathematically identical to multiplying the
    scalar objective, or class-frequency weighting would silently change the
    supervised comparison while attempting to fix AMP stability.
    """
    branches = _cpu_branches()
    references = {name: copy.deepcopy(branches[name].model) for name in BRANCH_NAMES}
    reference_optimizers = {
        name: torch.optim.AdamW(
            references[name].parameters(),
            lr=7e-4,
            weight_decay=1e-4,
        )
        for name in BRANCH_NAMES
    }
    loss_weights = {"normal_vae": 2.65, "so2_vae": 0.75}
    x = torch.tensor([[1.0, 2.0]])
    targets = {"normal_vae": torch.tensor([[0.5]]), "so2_vae": torch.tensor([[0.5]])}

    result = paired_atomic_step(
        branches,
        {
            name: lambda name=name: nn.functional.mse_loss(
                branches[name].model(x),
                targets[name],
            )
            for name in BRANCH_NAMES
        },
        learning_rate=7e-4,
        loss_weights=loss_weights,
    )

    for name in BRANCH_NAMES:
        expected_loss = nn.functional.mse_loss(references[name](x), targets[name])
        (expected_loss * loss_weights[name]).backward()  # pyright: ignore[reportUnknownMemberType]
        for observed, expected in zip(
            branches[name].model.parameters(),
            references[name].parameters(),
            strict=True,
        ):
            torch.testing.assert_close(observed.grad, expected.grad)
        reference_optimizers[name].step()  # pyright: ignore[reportUnknownMemberType]
        for observed, expected in zip(
            branches[name].model.parameters(),
            references[name].parameters(),
            strict=True,
        ):
            torch.testing.assert_close(observed, expected)
        assert result.losses[name] == pytest.approx(
            float((expected_loss * loss_weights[name]).item()),
        )


def test_gradient_failure_records_named_diagnostics_and_commits_neither_branch() -> (
    None
):
    """Expose the exact failing branch and parameter before a paired step can split.

    A finite loss can still create a nonfinite backward path under AMP; retaining
    branch, parameter, dtype, finite-range, and scaler evidence makes that one-off
    Kaggle failure diagnosable without permitting either representation to advance.
    """
    branches = _cpu_branches()
    before = {
        name: copy.deepcopy(branches[name].model.state_dict()) for name in BRANCH_NAMES
    }
    x = torch.tensor([[1.0, 2.0]])
    with pytest.raises(PairedNumericalError, match="missing/nonfinite") as caught:
        paired_atomic_step(
            branches,
            {
                "normal_vae": lambda: branches["normal_vae"].model(x).sum(),
                "so2_vae": lambda: _finite_loss_with_nonfinite_gradient(
                    branches["so2_vae"].model(x).sum(),
                ),
            },
            learning_rate=1e-3,
        )

    details = caught.value.details
    assert details["kind"] == "gradient"
    assert details["first_affected"] == {
        "branch": "so2_vae",
        "name": "weight",
        "dtype": "torch.float32",
        "missing": False,
        "nonfinite_count": 2,
        "max_finite_abs": None,
    }
    details_branches = cast("dict[str, object]", details["branches"])
    for name in BRANCH_NAMES:
        branch_details = cast("dict[str, object]", details_branches[name])
        assert float(cast("float", branch_details["scaler"])) > 0.0
        parameters = cast("list[dict[str, object]]", branch_details["parameters"])
        assert {parameter["name"] for parameter in parameters} == {"weight", "bias"}
        assert all(parameter["dtype"] == "torch.float32" for parameter in parameters)
        assert all(parameter["missing"] is False for parameter in parameters)
        if name == "so2_vae":
            assert all(
                int(cast("int", parameter["nonfinite_count"])) > 0
                for parameter in parameters
            )
            assert all(parameter["max_finite_abs"] is None for parameter in parameters)
        else:
            assert all(parameter["nonfinite_count"] == 0 for parameter in parameters)
            assert all(
                float(cast("float", parameter["max_finite_abs"])) > 0.0
                for parameter in parameters
            )

    for name in BRANCH_NAMES:
        observed = branches[name].model.state_dict()
        for key, value in before[name].items():
            torch.testing.assert_close(observed[key], value)


def test_finite_step_uses_one_lr_and_boundary_is_fully_paired(tmp_path: Path) -> None:
    """Commit a matched step and one hash-manifested resumable epoch fraction."""
    branches = _cpu_branches()
    x = torch.tensor([[1.0, 2.0]])
    targets = {"normal_vae": torch.tensor([[0.5]]), "so2_vae": torch.tensor([[0.5]])}
    result = paired_atomic_step(
        branches,
        {
            name: lambda name=name: nn.functional.mse_loss(
                branches[name].model(x),
                targets[name],
            )
            for name in BRANCH_NAMES
        },
        learning_rate=7e-4,
    )
    assert result.learning_rate == pytest.approx(7e-4)
    assert result.losses["normal_vae"] == pytest.approx(result.losses["so2_vae"])
    for name in BRANCH_NAMES:
        assert branches[name].optimizer.param_groups[0]["lr"] == pytest.approx(7e-4)

    payload = paired_checkpoint_payload(
        branches=branches,
        task="tissue_confirmation",
        epoch_fraction=0.5,
        completed_epoch=0,
        within_epoch_cursor=66,
        successful_pair_count=66,
        schedule={"kind": "warmup_hold", "peak": 7e-4, "steps_per_epoch": 132},
        order_identity={"epoch": 0, "sha256": "a" * 64, "cursor": 66},
        validation_history={name: [] for name in BRANCH_NAMES},
        best_metrics=dict.fromkeys(BRANCH_NAMES),
        patience_state=dict.fromkeys(BRANCH_NAMES, 0),
        access_transcript={"row_count": 66, "prefix_sha256": "b" * 64},
        campaign_progress={"phase": "confirmation", "task_index": 1},
        training_history={name: [{"update": 66}] for name in BRANCH_NAMES},
        ewma_state=dict.fromkeys(BRANCH_NAMES, 0.75),
    )
    boundary = commit_paired_boundary(tmp_path, epoch_fraction=0.5, payload=payload)
    progress = json.loads((boundary / "progress.json").read_text())
    manifest = json.loads((boundary / "manifest.json").read_text())
    restored = torch.load(
        boundary / "paired_checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert set(restored["branches"]) == set(BRANCH_NAMES)
    assert restored["successful_pair_count"] == 66
    assert restored["schedule"]["peak"] == pytest.approx(7e-4)
    assert restored["training_history"]["normal_vae"] == [{"update": 66}]
    assert restored["ewma_state"] == dict.fromkeys(BRANCH_NAMES, 0.75)
    assert set(restored["rng"]) == {"python", "numpy", "torch_cpu", "torch_cuda"}
    assert progress["epoch_fraction"] == pytest.approx(0.5)
    assert set(manifest) == {"paired_checkpoint.pt", "progress.json"}
    assert not (tmp_path / ".epoch_0.5.tmp").exists()

    resumed = _cpu_branches()
    loaded, paired_cursor, within_epoch = load_paired_boundary(boundary, resumed)
    assert loaded["task"] == "tissue_confirmation"
    assert (paired_cursor, within_epoch) == (66, 66)
    for name in BRANCH_NAMES:
        for key, expected in branches[name].model.state_dict().items():
            torch.testing.assert_close(resumed[name].model.state_dict()[key], expected)
        assert resumed[name].optimizer.param_groups[0]["lr"] == pytest.approx(7e-4)

    for active in (branches, resumed):
        paired_atomic_step(
            active,
            {
                name: lambda name=name, active=active: nn.functional.mse_loss(
                    active[name].model(x),
                    targets[name],
                )
                for name in BRANCH_NAMES
            },
            learning_rate=7e-4,
        )
    for name in BRANCH_NAMES:
        for key, expected in branches[name].model.state_dict().items():
            torch.testing.assert_close(resumed[name].model.state_dict()[key], expected)
