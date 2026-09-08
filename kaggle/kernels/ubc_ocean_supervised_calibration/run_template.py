# Copyright 2026 HiperMaximus
# ruff: noqa: ANN001, ANN202, BLE001, B905, C408, C420, C901, COM812, DOC201, DOC501, E731, EM101, EM102, PLC0415, PLR0912, PLR0913, PLR0914, PLR0915, PLR0916, PLR2004, PLW0717, RUF069, RUF100, S404, T201, TRY003, TRY300, TRY301
"""Generated wrapper for the compact paired Spec 0023 calibration package."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import cast

KAGGLE_UBC_OCEAN_SUPERVISED_CALIBRATION_READY = True
EMBEDDED_CONFIG_B64 = "$embedded_config_b64"
EMBEDDED_CONFIG_SHA256 = "$embedded_config_sha256"
INPUT_ROOT = Path("/kaggle/input")
WORKING_ROOT = Path("/kaggle/working")
OUTPUT_PATH = WORKING_ROOT / "spec0023_supervised_calibration_audit.json"
INPUT_CONTRACT_NAME = "spec0023_supervised_calibration_input_contract.json"
HORIZON_AUTHORITY_SHA256 = {
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


def main() -> int:
    """Run precisely one local-built sweep or human-sealed confirmation package."""
    try:
        _ensure_latest_torch()
        config = _embedded_config()
        source_root = _resolve_calibration_input(config)
        sys.path.insert(0, str(source_root / "src"))
        _validate_hardware(config)
        source_roots = _resolve_source_roots(config)
        package_mode = cast("str", config["package_mode"])
        if package_mode == "sweep":
            result = _run_sweeps(source_root, config, source_roots)
        elif package_mode == "confirmation":
            result = _run_confirmations(source_root, config, source_roots)
        elif package_mode == "horizon":
            result = _run_mil_horizon(source_root, config, source_roots)
        elif package_mode == "width128":
            result = _run_mil_width128(source_root, config, source_roots)
        elif package_mode in {"class_specific", "class_specific_scale_fix"}:
            result = _run_mil_class_specific(source_root, config, source_roots)
        else:
            raise RuntimeError("Unknown embedded Spec 0023 calibration package mode")
        result["runtime_stack"] = _runtime_stack()
        _write_audit(result, config)
        return 1 if result.get("status") == "failed" else 0
    except Exception:
        traceback.print_exc()
        return 1


def _ensure_latest_torch() -> None:
    if not WORKING_ROOT.exists() or os.environ.get("EQVAE_SKIP_REMOTE_SETUP") == "1":
        return
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "torch",
        "torchvision",
        "torchaudio",
    ])


def _validate_hardware(config: dict[str, object]) -> None:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("Spec 0023 calibration requires exactly two CUDA devices")
    devices = cast("dict[str, int]", config["model_devices"])
    if devices != {"normal_vae": 0, "so2_vae": 1}:
        raise RuntimeError("Spec 0023 calibration device assignment differs")
    names = [torch.cuda.get_device_name(index) for index in range(2)]
    if any("T4" not in name for name in names):
        raise RuntimeError(f"Spec 0023 calibration expects two T4s, found {names!r}")


def _runtime_stack() -> dict[str, object]:
    import torch
    import torchaudio
    import torchvision

    return {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "torchaudio": torchaudio.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_names": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
        "cuda_capabilities": [
            list(torch.cuda.get_device_capability(index))
            for index in range(torch.cuda.device_count())
        ],
    }


def _run_sweeps(
    source_root: Path, config: dict[str, object], source_roots: dict[str, Path]
) -> dict[str, object]:
    if "selection_audit" in config:
        raise RuntimeError("Sweep package must not include a selected peak")
    recovery = _run_mil_numerical_recovery(
        source_root=source_root,
        config=config,
        source_roots=source_roots,
    )
    if recovery["status"] != "pass":
        return {
            "schema_version": "spec0023.supervised_calibration_audit.v1",
            "package_mode": "sweep",
            "status": "failed",
            "numerical_recovery": recovery,
            "task_results": {},
            "selection_status": "selection_blocked_numerical_recovery",
            "logical_access": "train_only",
        }
    mil = _run_sweep_task(
        source_root=source_root,
        config=config,
        source_roots=source_roots,
        task="mil",
    )
    tissue = _run_sweep_task(
        source_root=source_root,
        config=config,
        source_roots=source_roots,
        task="tissue",
    )
    selection_ready = all(
        cast("int", result["successful_updates"]) >= 10 for result in (mil, tissue)
    )
    return {
        "schema_version": "spec0023.supervised_calibration_audit.v1",
        "package_mode": "sweep",
        "status": "selection_ready" if selection_ready else "failed",
        "numerical_recovery": recovery,
        "task_results": {"mil": mil, "tissue": tissue},
        "selection_status": (
            "human_selection_required_before_confirmation"
            if selection_ready
            else "selection_blocked_insufficient_aligned_prefix"
        ),
        "logical_access": "train_only",
    }


def _run_mil_numerical_recovery(*, source_root, config, source_roots):
    """Reproduce the v1 MC overflow, then prove the one fixed execution path."""
    legacy = _run_mil_replay(
        source_root=source_root,
        config=config,
        source_roots=source_roots,
        weight_application="scaled_loss",
    )
    corrected = _run_mil_replay(
        source_root=source_root,
        config=config,
        source_roots=source_roots,
        weight_application="post_grad_scaler_unscale",
    )
    v1_contract = cast("dict[str, object]", config["numerical_recovery"])
    passed = _mil_recovery_passes(legacy, corrected, v1_contract)
    return {
        "status": "pass" if passed else "failed",
        "expected_failure": {
            "attempt": 12,
            "wsi_id": 65094,
            "diagnosis_label": "MC",
            "instance_count": 5773,
            "class_weight": 2.65,
            "grad_scaler_scale": 32768.0,
        },
        "v1_contract": v1_contract,
        "legacy_scaled_loss_replay": legacy,
        "post_unscale_replay": corrected,
        "replay_state_reused_by_sweep": False,
        "checkpoint_chunk_size": None,
    }


def _mil_recovery_passes(legacy, corrected, v1_contract):
    failure = legacy.get("failure")
    if not isinstance(failure, dict):
        return False
    diagnostics = failure.get("diagnostics")
    if not isinstance(diagnostics, dict) or diagnostics.get("kind") != "gradient":
        return False
    inputs = diagnostics.get("input")
    if not isinstance(inputs, dict):
        return False
    sample = inputs.get("logical_sample")
    branches = diagnostics.get("branches")
    first_affected = diagnostics.get("first_affected")
    if (
        not isinstance(sample, dict)
        or not isinstance(branches, dict)
        or not isinstance(first_affected, dict)
    ):
        return False
    branch_rows = [branches.get(name) for name in ("normal_vae", "so2_vae")]
    if not all(isinstance(row, dict) for row in branch_rows):
        return False
    has_nonfinite_gradient = any(
        any(
            isinstance(parameter, dict)
            and isinstance(parameter.get("nonfinite_count"), int)
            and parameter["nonfinite_count"] > 0
            for parameter in row.get("parameters", [])
        )
        for row in branch_rows
    )
    return (
        legacy.get("weight_application") == "scaled_loss"
        and legacy.get("successful_updates") == 11
        and len(legacy.get("committed", [])) == 11
        and failure.get("attempt") == 12
        and failure.get("wsi_id") == 65094
        and failure.get("diagnosis_label") == "MC"
        and failure.get("instance_count") == 5773
        and sample.get("wsi_id") == 65094
        and sample.get("diagnosis_label") == "MC"
        and sample.get("instance_count") == 5773
        and _same_number(sample.get("class_weight"), 2.65)
        and sample.get("checkpoint_chunk_size") is None
        and all(_same_number(row.get("scaler"), 32768.0) for row in branch_rows)
        and has_nonfinite_gradient
        and first_affected.get("branch") in {"normal_vae", "so2_vae"}
        and isinstance(first_affected.get("name"), str)
        and isinstance(first_affected.get("nonfinite_count"), int)
        and first_affected["nonfinite_count"] > 0
        and corrected.get("weight_application") == "post_grad_scaler_unscale"
        and corrected.get("successful_updates") == 12
        and len(corrected.get("committed", [])) == 12
        and corrected.get("failure") is None
        and legacy.get("initialization_sha256")
        == corrected.get("initialization_sha256")
        and legacy.get("order_sha256") == corrected.get("order_sha256")
        and legacy.get("initialization_sha256")
        == v1_contract.get("legacy_initialization_sha256")
        and legacy.get("order_sha256") == v1_contract.get("legacy_order_sha256")
        and v1_contract.get("v1_sweep_audit_sha256")
        == "654c8c244698717c1a628e3b395c9b414f9e1945cafc1fdd893ce1457d6246ef"
    )


def _same_number(value, expected):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isclose(float(value), expected, rel_tol=1e-12, abs_tol=0.0)
    )


def _run_mil_replay(*, source_root, config, source_roots, weight_application):
    import torch

    from eqvae.training.supervised_calibration import PairedNumericalError
    from eqvae.training.supervised_pairing import exponential_learning_rates

    context = _task_context(
        source_root, config, source_roots, "mil", include_validation=False
    )
    try:
        rates = exponential_learning_rates(
            start=cast("float", config["sweep"]["start_learning_rate"]),
            stop=cast("float", config["sweep"]["stop_learning_rate"]),
            update_count=cast("int", context["steps_per_epoch"]),
        )
        order = _task_order(context, epoch=0)
        committed = []
        failure = None
        for attempt in range(1, 13):
            row = _training_row(context, order, attempt - 1)
            try:
                observation = _paired_train_step(
                    context["branches"],
                    row,
                    rates[attempt - 1],
                    "mil",
                    context["class_weights"],
                    weight_application=weight_application,
                )
            except PairedNumericalError as error:
                loaded = row["normal_vae"]
                failure = {
                    "attempt": attempt,
                    "learning_rate": rates[attempt - 1],
                    "wsi_id": loaded.bag.wsi_id,
                    "diagnosis_label": loaded.bag.diagnosis_label,
                    "diagnosis_index": loaded.bag.diagnosis_index,
                    "instance_count": loaded.bag.instance_count,
                    "error": str(error),
                    "diagnostics": error.details,
                }
                break
            committed.append({
                "attempt": attempt,
                "learning_rate": rates[attempt - 1],
                "losses": observation["losses"],
            })
        return {
            "weight_application": weight_application,
            "initialization_sha256": context["initialization_sha256"],
            "order_sha256": _sha256_rows(order),
            "successful_updates": len(committed),
            "committed": committed,
            "failure": failure,
        }
    finally:
        _close_context(context)
        torch.cuda.empty_cache()


def _run_confirmations(
    source_root: Path, config: dict[str, object], source_roots: dict[str, Path]
) -> dict[str, object]:
    selection = cast("dict[str, object]", config.get("selection_audit"))
    peaks = cast("dict[str, float]", selection.get("shared_peaks"))
    if set(peaks) != {"mil", "tissue"} or any(value <= 0.0 for value in peaks.values()):
        raise RuntimeError(
            "Confirmation requires exactly one positive shared peak per task"
        )
    confirmation = cast("dict[str, object]", config.get("confirmation"))
    active_tasks = cast("list[str]", confirmation.get("active_tasks"))
    frozen_tasks = cast("list[str]", confirmation.get("frozen_tasks"))
    if active_tasks != ["mil"] or frozen_tasks != ["tissue"]:
        raise RuntimeError("Revised confirmation must run MIL and freeze tissue")
    results = {}
    for task in active_tasks:
        initial = _run_confirmation_task(
            source_root=source_root,
            config=config,
            source_roots=source_roots,
            task=task,
            peak=peaks[task],
            attempt="initial",
        )
        attempts = [initial]
        effective_peak = peaks[task]
        if initial["status"] == "numerical_failure":
            effective_peak = peaks[task] / 2.0
            attempts.append(
                _run_confirmation_task(
                    source_root=source_root,
                    config=config,
                    source_roots=source_roots,
                    task=task,
                    peak=effective_peak,
                    attempt="half_peak_fallback",
                )
            )
        results[task] = {
            "status": attempts[-1]["status"],
            "selected_peak": peaks[task],
            "effective_peak": effective_peak,
            "attempts": attempts,
        }
    passed = all(result["status"] == "pass" for result in results.values())
    return {
        "schema_version": "spec0023.supervised_calibration_audit.v1",
        "package_mode": "confirmation",
        "selection_audit_sha256": selection["sha256"],
        "shared_peaks": peaks,
        "task_results": results,
        "active_tasks": active_tasks,
        "frozen_tasks": frozen_tasks,
        "status": "pass" if passed else "failed",
        "logical_access": "train_and_validation_only",
    }


def _run_mil_horizon(
    source_root: Path, config: dict[str, object], source_roots: dict[str, Path]
) -> dict[str, object]:
    import torch

    from eqvae.training.supervised_calibration import (
        commit_paired_boundary,
        load_paired_boundary,
        paired_checkpoint_payload,
        update_ewma,
    )
    from eqvae.training.supervised_pairing import half_epoch_boundaries

    horizon = cast("dict[str, object]", config.get("horizon"))
    expected_horizon = {
        "source": "confirmation_v1_epoch_2.0",
        "resume_boundary": "resume",
        "resume_authority_sha256": HORIZON_AUTHORITY_SHA256,
        "peak": 2e-4,
        "start_epoch": 2,
        "target_epoch": 5,
        "continuation_updates": 318,
        "validation_every": "half_epoch",
        "warmup_repeated": False,
        "active_tasks": ["mil"],
        "frozen_tasks": ["tissue"],
    }
    if horizon != expected_horizon:
        raise RuntimeError("MIL horizon configuration differs")
    expected_assets = {
        "physical_parts.csv",
        "wsi/wsi_cancer_train_instances.csv",
        "wsi/wsi_cancer_train_bags.csv",
        "wsi/wsi_cancer_validation_instances.csv",
        "wsi/wsi_cancer_validation_bags.csv",
        "resume/manifest.json",
        "resume/progress.json",
        "resume/paired_checkpoint.pt",
        "resume/confirmation_v1_audit.json",
        "resume/confirmation_v1_config.json",
    }
    if set(cast("dict[str, str]", config["logical_manifest_sha256"])) != (
        expected_assets
    ) or set(cast("dict[str, object]", config["tasks"])) != {"mil"}:
        raise RuntimeError("MIL horizon input surface differs")
    context = _task_context(
        source_root, config, source_roots, "mil", include_validation=True
    )
    task_root = WORKING_ROOT / "mil" / "horizon"
    task_root.parent.mkdir(exist_ok=True)
    task_root.mkdir(exist_ok=False)
    try:
        prior_attempt = _validate_horizon_authority(source_root, config, context)
        branches = context["branches"]
        payload, cursor, within_epoch = load_paired_boundary(
            source_root / "resume", branches
        )
        if (
            payload.get("task") != "mil_confirmation"
            or payload.get("epoch_fraction") != 2.0
            or payload.get("completed_epoch") != 2
            or (cursor, within_epoch) != (212, 106)
            or payload.get("schedule")
            != {"kind": "warmup_then_hold", "peak": 2e-4, "steps_per_epoch": 106}
        ):
            raise RuntimeError("MIL horizon restored checkpoint identity differs")
        for branch in branches.values():
            groups = branch.optimizer.param_groups
            if (
                len(groups) != 1
                or groups[0]["lr"] != 2e-4
                or groups[0]["weight_decay"] != 1e-4
                or branch.scaler.get_scale() != 32768.0
                or branch.scaler.get_growth_interval() != 1000000
            ):
                raise RuntimeError("MIL horizon restored optimizer/scaler differs")
        validation_history = cast(
            "dict[str, list[dict[str, object]]]", payload["validation_history"]
        )
        if validation_history != prior_attempt["validation_history"]:
            raise RuntimeError("MIL horizon checkpoint validation history differs")
        histories = cast(
            "dict[str, list[dict[str, object]]]", prior_attempt["histories"]
        )
        if any(len(history) != 212 for history in histories.values()):
            raise RuntimeError("MIL horizon pre-resume training history differs")
        ewma = {name: float(history[-1]["ewma"]) for name, history in histories.items()}
        access_prefix, accessed_rows = _reconstruct_mil_access_prefix(
            context, completed_epochs=2
        )
        saved_access = cast("dict[str, object]", payload["access_transcript"])
        if (
            access_prefix.hexdigest()
            != saved_access.get("physical_pointer_prefix_sha256")
            or accessed_rows != saved_access.get("physical_pointer_rows")
            or access_prefix.hexdigest()
            != prior_attempt["physical_pointer_prefix_sha256"]
            or accessed_rows != prior_attempt["physical_pointer_rows"]
        ):
            raise RuntimeError("MIL horizon reconstructed access prefix differs")
        steps = cast("int", context["steps_per_epoch"])
        midpoint, endpoint = half_epoch_boundaries(steps)
        boundaries = []
        for epoch in range(2, 5):
            epoch_order = _task_order(context, epoch=epoch)
            for new_within_epoch in range(1, steps + 1):
                row = _training_row(context, epoch_order, new_within_epoch - 1)
                accessed_rows += _extend_access_prefix(access_prefix, row)
                observation = _paired_train_step(
                    branches,
                    row,
                    2e-4,
                    "mil",
                    context.get("class_weights"),
                )
                cursor += 1
                for name, loss in observation["losses"].items():
                    ewma[name] = update_ewma(ewma[name], loss)
                    histories[name].append({
                        "update": cursor,
                        "loss": loss,
                        "ewma": ewma[name],
                    })
                if new_within_epoch in {midpoint, endpoint}:
                    metrics = _validate_context(context, "mil")
                    for name in branches:
                        validation_history[name].append(metrics[name])
                    fraction = epoch + (0.5 if new_within_epoch == midpoint else 1.0)
                    checkpoint_payload = paired_checkpoint_payload(
                        branches=branches,
                        task="mil_horizon",
                        epoch_fraction=fraction,
                        completed_epoch=(
                            epoch if new_within_epoch == midpoint else epoch + 1
                        ),
                        within_epoch_cursor=new_within_epoch,
                        successful_pair_count=cursor,
                        schedule={
                            "kind": "warmup_then_hold",
                            "peak": 2e-4,
                            "steps_per_epoch": steps,
                            "warmup_completed_epoch": 1,
                            "target_epoch": 5,
                        },
                        order_identity={
                            "seed": context["seed"],
                            "epoch": epoch,
                            "sha256": _sha256_rows(epoch_order),
                            "cursor": new_within_epoch,
                        },
                        validation_history=validation_history,
                        training_history=histories,
                        ewma_state=ewma,
                        best_metrics=_best_validation(validation_history),
                        patience_state=_patience_state(validation_history),
                        access_transcript={
                            "logical_access": "train_and_validation_only",
                            "logical_manifest_sha256": context[
                                "logical_manifest_sha256"
                            ],
                            "current_order_sha256": _sha256_rows(epoch_order),
                            "within_epoch_cursor": new_within_epoch,
                            "physical_pointer_prefix_sha256": (
                                access_prefix.hexdigest()
                            ),
                            "physical_pointer_rows": accessed_rows,
                            "validation_pointer_sha256": context[
                                "validation_pointer_sha256"
                            ],
                            "validation_pointer_rows": context[
                                "validation_pointer_rows"
                            ],
                            "validation_checks": len(validation_history["normal_vae"]),
                            "resume_source": "confirmation_v1_epoch_2.0",
                        },
                        campaign_progress={
                            "phase": "mil_horizon",
                            "task": "mil",
                            "peak": 2e-4,
                            "source_epoch_fraction": 2.0,
                            "target_epoch_fraction": 5.0,
                        },
                    )
                    boundary = commit_paired_boundary(
                        task_root,
                        epoch_fraction=fraction,
                        payload=checkpoint_payload,
                    )
                    _, restored_cursor, restored_within_epoch = load_paired_boundary(
                        boundary, branches
                    )
                    if (restored_cursor, restored_within_epoch) != (
                        cursor,
                        new_within_epoch,
                    ):
                        raise RuntimeError("MIL horizon boundary cursor differs")
                    boundaries.append({
                        "epoch_fraction": fraction,
                        "manifest_sha256": _sha256_file(boundary / "manifest.json"),
                        "checkpoint_sha256": _sha256_file(
                            boundary / "paired_checkpoint.pt"
                        ),
                        "progress_sha256": _sha256_file(boundary / "progress.json"),
                    })
        if cursor != 530 or len(boundaries) != 6:
            raise RuntimeError("MIL horizon continuation length differs")
        observations = _horizon_learning_observations(histories, validation_history)
        return {
            "schema_version": "spec0023.mil_horizon_audit.v1",
            "package_mode": "horizon",
            "status": "complete",
            "scientific_result": (
                "learning_observed"
                if all(
                    row["epoch5_ewma_below_epoch2"] and row["epoch5_mean_below_epoch2"]
                    for row in observations.values()
                )
                else "insufficient_learning"
            ),
            "peak": 2e-4,
            "warmup_repeated": False,
            "source_epoch_fraction": 2.0,
            "target_epoch_fraction": 5.0,
            "successful_updates_before_resume": 212,
            "successful_continuation_updates": 318,
            "successful_updates_total": cursor,
            "histories": histories,
            "validation_history": validation_history,
            "learning_observations": observations,
            "boundaries": boundaries,
            "resume_authority_sha256": horizon["resume_authority_sha256"],
            "logical_access": "train_and_validation_only",
            "frozen_tasks": ["tissue"],
            "checkpoint_chunk_size": None,
            "physical_pointer_prefix_sha256": access_prefix.hexdigest(),
            "physical_pointer_rows": accessed_rows,
            "zero_scaler_skips": True,
        }
    finally:
        _close_context(context)
        torch.cuda.empty_cache()


def _run_mil_width128(source_root, config, source_roots):
    diagnostic = cast("dict[str, object]", config.get("architecture_diagnostic"))
    expected = {
        "kind": "gated_attention_scorer_width",
        "baseline_attention_dim": 64,
        "attention_dim": 128,
        "fresh_initialization": True,
        "peak": 2e-4,
        "epochs": 5,
        "warmup_updates": 11,
        "validation_every": "half_epoch",
        "baseline_audit_sha256": (
            "2edeb5d20471ebc559e3bb7f6f7cde4aeaa06723b98bdcfc28a7dca4039525fb"
        ),
        "active_tasks": ["mil"],
        "frozen_tasks": ["tissue"],
    }
    expected_assets = {
        "physical_parts.csv",
        "wsi/wsi_cancer_train_instances.csv",
        "wsi/wsi_cancer_train_bags.csv",
        "wsi/wsi_cancer_validation_instances.csv",
        "wsi/wsi_cancer_validation_bags.csv",
    }
    task = cast("dict[str, object]", config["tasks"])["mil"]
    if (
        diagnostic != expected
        or set(cast("dict[str, str]", config["logical_manifest_sha256"]))
        != expected_assets
        or set(cast("dict[str, object]", config["tasks"])) != {"mil"}
        or cast("dict[str, object]", task).get("attention_dim") != 128
    ):
        raise RuntimeError("MIL width128 configuration or input surface differs")
    attempt = _run_confirmation_task(
        source_root=source_root,
        config=config,
        source_roots=source_roots,
        task="mil",
        peak=2e-4,
        attempt="width128",
        epochs=5,
        phase="width128",
    )
    if attempt.get("attention_dim") != 128:
        raise RuntimeError("MIL width128 runtime architecture differs")
    numerical_failure = attempt["status"] == "numerical_failure"
    observations = None if numerical_failure else _width128_observations(attempt)
    return {
        "schema_version": "spec0023.mil_width128_audit.v1",
        "package_mode": "width128",
        "status": "failed" if numerical_failure else "complete",
        "scientific_result": attempt["status"],
        "peak": 2e-4,
        "epochs": 5,
        "attention_dim": 128,
        "baseline_attention_dim": 64,
        "baseline_audit_sha256": diagnostic["baseline_audit_sha256"],
        "task_result": attempt,
        "learning_observations": observations,
        "logical_access": "train_and_validation_only",
        "frozen_tasks": ["tissue"],
        "checkpoint_chunk_size": None,
    }


def _width128_observations(attempt):
    baseline_best = {"normal_vae": 0.245, "so2_vae": 0.24643962848297213}
    result = {}
    for name in ("normal_vae", "so2_vae"):
        history = attempt["histories"][name]
        validation = attempt["validation_history"][name]
        best = _best_validation({name: validation})[name]
        result[name] = {
            "epoch_mean_weighted_losses": [
                sum(row["loss"] for row in history[start : start + 106]) / 106
                for start in range(0, 530, 106)
            ],
            "epoch_final_ewmas": [
                history[index - 1]["ewma"] for index in range(106, 531, 106)
            ],
            "best_validation": best,
            "width64_best_macro_f1": baseline_best[name],
            "best_macro_f1_above_width64": best["macro_f1"] > baseline_best[name],
        }
    return result


def _run_mil_class_specific(source_root, config, source_roots):
    scale_fix = config.get("package_mode") == "class_specific_scale_fix"
    diagnostic = cast("dict[str, object]", config.get("architecture_diagnostic"))
    expected = {
        "kind": "class_specific_gated_attention",
        "attention_dim": 128,
        "attention_maps": 5,
        "fresh_initialization": True,
        "function_preserving_initialization": True,
        "peak": 2e-4,
        "epochs": 5,
        "warmup_updates": 11,
        "validation_every": "half_epoch",
        "baseline": {
            "audit_sha256": (
                "3e0933d87e334ba977dfaaca45bab99b70984cd88a1fa0f0097f9259bb77de23"
            ),
            "config_sha256": (
                "b2de025ca3b647a2c9feecb640115bd60ea9624c76c4587de631ae166b691dde"
            ),
            "width128_initialization_sha256": (
                "cfef73d525bf5e58501ca7aa9c93ed65a9a069f08ebf2fcefea63687b04dbcbf"
            ),
            "class_specific_initialization_sha256": (
                "d6f916aafd6b544d62c2f481ff5f619e6485156dac08461b96546be8b5717c22"
            ),
            "best_macro_f1": {
                "normal_vae": 0.2439628482972136,
                "so2_vae": 0.24643962848297213,
            },
        },
        "active_tasks": ["mil"],
        "frozen_tasks": ["tissue"],
    }
    if scale_fix:
        expected["correction"] = "single_synchronized_paired_scale_backoff"
    expected_scale_correction = (
        {
            "failure_audit": {
                "audit_sha256": (
                    "07fd04096c151fbd0be2b2b997e7f3c62dad9cdc84a0b421d6eec4a175ad9efc"
                ),
                "config_sha256": (
                    "09fcb3374bfb6c29b99ace77a55b327a6b523117c6d8ef7be085445cb0893d10"
                ),
                "predecessor_audit_sha256": (
                    "3e0933d87e334ba977dfaaca45bab99b70984cd88a1fa0f0097f9259bb77de23"
                ),
                "initialization_sha256": (
                    "d6f916aafd6b544d62c2f481ff5f619e6485156dac08461b96546be8b5717c22"
                ),
                "successful_updates": 465,
                "failed_attempt": 466,
                "scale": 32768,
                "failed_branch": "normal_vae",
                "first_affected_parameter": "patch_encoder.layers.0.weight",
                "failing_wsi_id": 55287,
                "instance_count": 3869,
                "complete_bag": True,
                "inputs_and_forwards_finite": True,
                "logical_access": "train_and_validation_only",
            },
            "maximum_backoffs": 1,
            "initial_scale": 32768,
            "backoff_factor": 0.5,
            "retry_same_loaded_bag": True,
            "retry_same_learning_rate": True,
            "advance_successful_cursor_on_discard": False,
            "raw_gradient_check": "after_unscale_before_class_weight",
            "weighted_gradient_check": "terminal_after_fp32_class_weight",
        }
        if scale_fix
        else None
    )
    expected_assets = {
        "physical_parts.csv",
        "wsi/wsi_cancer_train_instances.csv",
        "wsi/wsi_cancer_train_bags.csv",
        "wsi/wsi_cancer_validation_instances.csv",
        "wsi/wsi_cancer_validation_bags.csv",
    }
    task = cast("dict[str, object]", config["tasks"])["mil"]
    if (
        diagnostic != expected
        or set(cast("dict[str, str]", config["logical_manifest_sha256"]))
        != expected_assets
        or set(cast("dict[str, object]", config["tasks"])) != {"mil"}
        or cast("dict[str, object]", task).get("attention_dim") != 128
        or cast("dict[str, object]", task).get("class_specific_attention") is not True
        or cast("dict[str, object]", task).get("expected_initialization_sha256")
        != diagnostic["baseline"]["class_specific_initialization_sha256"]
        or cast("dict[str, object]", task).get("max_paired_scale_backoffs")
        != (1 if scale_fix else None)
        or config.get("optimizer")
        != {
            "name": "AdamW",
            "matrix_weight_decay": 1e-4,
            "vector_weight_decay": 0.0,
            "grouping": "parameter_ndim_ge_2",
        }
        or config.get("checkpoint_chunk_size") is not None
        or config.get("grad_scaler_init_scale") != 32768
        or config.get("grad_scaler_growth_interval") != 1000000
        or config.get("paired_scale_correction") != expected_scale_correction
    ):
        raise RuntimeError("MIL class-specific configuration or input differs")
    attempt = _run_confirmation_task(
        source_root=source_root,
        config=config,
        source_roots=source_roots,
        task="mil",
        peak=2e-4,
        attempt="class_specific",
        epochs=5,
        phase="class_specific_scale_fix" if scale_fix else "class_specific",
    )
    if (
        attempt.get("attention_dim") != 128
        or attempt.get("class_specific_attention") is not True
    ):
        raise RuntimeError("MIL class-specific runtime architecture differs")
    scale_backoff_events = cast(
        "list[dict[str, object]]",
        attempt.get("scale_backoff_events"),
    )
    if scale_fix:
        if (
            attempt.get("scale_backoff_budget") != 1
            or not isinstance(scale_backoff_events, list)
            or attempt.get("scale_backoffs_used") != len(scale_backoff_events)
            or attempt.get("scale_backoffs_remaining") != 1 - len(scale_backoff_events)
            or len(scale_backoff_events) > 1
        ):
            raise RuntimeError("MIL paired scale-backoff accounting differs")
        if scale_backoff_events:
            event = scale_backoff_events[0]
            if (
                not isinstance(event.get("update"), int)
                or not 1 <= event["update"] <= 530
                or event.get("scales_before")
                != {"normal_vae": 32768.0, "so2_vae": 32768.0}
                or event.get("scales_after")
                != {"normal_vae": 16384.0, "so2_vae": 16384.0}
            ):
                raise RuntimeError("MIL paired scale-backoff event differs")
    elif (
        attempt.get("scale_backoff_budget") != 0
        or attempt.get("scale_backoffs_used") != 0
        or attempt.get("scale_backoffs_remaining") != 0
        or scale_backoff_events != []
    ):
        raise RuntimeError("Legacy class-specific run used a scale correction")
    numerical_failure = attempt["status"] == "numerical_failure"
    observations = None if numerical_failure else _class_specific_observations(attempt)
    return {
        "schema_version": (
            "spec0023.mil_class_specific_scale_fix_audit.v1"
            if scale_fix
            else "spec0023.mil_class_specific_audit.v1"
        ),
        "package_mode": cast("str", config["package_mode"]),
        "status": "failed" if numerical_failure else "complete",
        "scientific_result": attempt["status"],
        "peak": 2e-4,
        "epochs": 5,
        "attention_dim": 128,
        "attention_maps": 5,
        "baseline": diagnostic["baseline"],
        "paired_scale_correction": config.get("paired_scale_correction"),
        "task_result": attempt,
        "learning_observations": observations,
        "logical_access": "train_and_validation_only",
        "frozen_tasks": ["tissue"],
        "checkpoint_chunk_size": None,
    }


def _class_specific_observations(attempt):
    baseline_best = {
        "normal_vae": 0.2439628482972136,
        "so2_vae": 0.24643962848297213,
    }
    result = {}
    for name in ("normal_vae", "so2_vae"):
        history = attempt["histories"][name]
        validation = attempt["validation_history"][name]
        best = _best_validation({name: validation})[name]
        result[name] = {
            "epoch_mean_weighted_losses": [
                sum(row["loss"] for row in history[start : start + 106]) / 106
                for start in range(0, 530, 106)
            ],
            "epoch_final_ewmas": [
                history[index - 1]["ewma"] for index in range(106, 531, 106)
            ],
            "best_validation": best,
            "width128_best_macro_f1": baseline_best[name],
            "best_macro_f1_above_width128": (best["macro_f1"] > baseline_best[name]),
        }
    return result


def _validate_horizon_authority(source_root, config, context):
    horizon = cast("dict[str, object]", config["horizon"])
    authority = cast("dict[str, str]", horizon["resume_authority_sha256"])
    if authority != HORIZON_AUTHORITY_SHA256:
        raise RuntimeError("MIL horizon embedded resume authority differs")
    if any(
        _sha256_file(source_root / name) != digest for name, digest in authority.items()
    ):
        raise RuntimeError("MIL horizon mounted resume authority differs")
    original_config = json.loads(
        (source_root / "resume/confirmation_v1_config.json").read_text()
    )
    original_audit = json.loads(
        (source_root / "resume/confirmation_v1_audit.json").read_text()
    )
    if (
        original_audit.get("config_sha256")
        != authority["resume/confirmation_v1_config.json"]
        or original_config.get("checkpoint_chunk_size") is not None
        or original_config.get("optimizer") != {"name": "AdamW", "weight_decay": 1e-4}
        or original_config.get("grad_scaler_init_scale") != 32768
        or original_config.get("grad_scaler_growth_interval") != 1000000
        or original_audit.get("shared_peaks", {}).get("mil") != 2e-4
    ):
        raise RuntimeError("MIL horizon confirmation-v1 config/audit differs")
    attempt = original_audit["task_results"]["mil"]["attempts"]
    if len(attempt) != 1 or attempt[0].get("peak") != 2e-4:
        raise RuntimeError("MIL horizon confirmation-v1 attempt differs")
    prior = attempt[0]
    if prior.get("logical_manifest_sha256") != context["logical_manifest_sha256"]:
        raise RuntimeError("MIL horizon logical manifests differ from resume")
    return prior


def _reconstruct_mil_access_prefix(context, *, completed_epochs):
    digest = hashlib.sha256()
    rows = 0
    datasets = context["train"]
    for epoch in range(completed_epochs):
        for index in _task_order(context, epoch=epoch):
            identities = []
            for name in ("normal_vae", "so2_vae"):
                dataset = datasets[name]
                bag = dataset.bags[index]
                instances = dataset.instances[
                    bag.instance_start : bag.instance_start + bag.instance_count
                ]
                identities.append(
                    tuple(
                        (
                            instance.atlas_row_index,
                            instance.wsi_id,
                            instance.x,
                            instance.y,
                            instance.pointer.part,
                            instance.pointer.file_index,
                        )
                        for instance in instances
                    )
                )
            if identities[0] != identities[1]:
                raise RuntimeError("MIL horizon paired pointer identity differs")
            for identity in identities[0]:
                digest.update(json.dumps(identity, separators=(",", ":")).encode())
                digest.update(b"\n")
            rows += len(identities[0])
    return digest, rows


def _horizon_learning_observations(histories, validation_history):
    result = {}
    for name in ("normal_vae", "so2_vae"):
        history = histories[name]
        epoch2 = history[106:212]
        epoch5 = history[424:530]
        prior_validation = validation_history[name][:4]
        post_validation = validation_history[name][4:]
        prior_best = _best_validation({name: prior_validation})[name]
        post_best = _best_validation({name: post_validation})[name]
        result[name] = {
            "epoch2_final_ewma": history[211]["ewma"],
            "epoch5_final_ewma": history[529]["ewma"],
            "epoch5_ewma_below_epoch2": (history[529]["ewma"] < history[211]["ewma"]),
            "epoch2_mean_weighted_loss": sum(row["loss"] for row in epoch2) / 106,
            "epoch5_mean_weighted_loss": sum(row["loss"] for row in epoch5) / 106,
            "epoch5_mean_below_epoch2": (
                sum(row["loss"] for row in epoch5) < sum(row["loss"] for row in epoch2)
            ),
            "pre_resume_best_validation": prior_best,
            "post_resume_best_validation": post_best,
            "post_resume_validation_improved": (
                (post_best["macro_f1"], -post_best["loss"])
                > (prior_best["macro_f1"], -prior_best["loss"])
            ),
        }
    return result


def _run_sweep_task(*, source_root, config, source_roots, task):
    import torch

    from eqvae.training.supervised_calibration import (
        PairedNumericalError,
        update_ewma,
    )
    from eqvae.training.supervised_pairing import exponential_learning_rates

    context = _task_context(
        source_root, config, source_roots, task, include_validation=False
    )
    try:
        branches = context["branches"]
        rates = exponential_learning_rates(
            start=cast("float", config["sweep"]["start_learning_rate"]),
            stop=cast("float", config["sweep"]["stop_learning_rate"]),
            update_count=cast("int", context["steps_per_epoch"]),
        )
        history = {name: [] for name in branches}
        ewma = dict.fromkeys(branches)
        attempts = []
        access_prefix = hashlib.sha256()
        accessed_rows = 0
        terminal = "complete"
        order = _task_order(context, epoch=0)
        for index, (row, learning_rate) in enumerate(
            (
                (_training_row(context, order, update_index), rate)
                for update_index, rate in enumerate(rates)
            ),
            start=1,
        ):
            accessed_rows += _extend_access_prefix(access_prefix, row)
            try:
                observation = _paired_train_step(
                    branches,
                    row,
                    learning_rate,
                    task,
                    context.get("class_weights"),
                )
            except PairedNumericalError as error:
                terminal = f"numerical_failure:{error}"
                attempts.append({
                    "attempt": index,
                    "learning_rate": learning_rate,
                    "status": "numerical_failure",
                    "diagnostics": error.details,
                })
                break
            attempts.append({
                "attempt": index,
                "learning_rate": learning_rate,
                "status": "committed",
            })
            for name, loss in observation["losses"].items():
                smoothed = update_ewma(ewma[name], loss)
                ewma[name] = smoothed
                history[name].append({"update": index, "loss": loss, "ewma": smoothed})
            if any(
                len(history[name]) >= 10
                and history[name][-1]["ewma"]
                > 4.0 * min(entry["ewma"] for entry in history[name][:-1])
                for name in branches
            ):
                terminal = "diverged"
                break
        return {
            "initialization_sha256": context["initialization_sha256"],
            "task": task,
            "steps_per_epoch": context["steps_per_epoch"],
            "learning_rates": rates[: len(history["normal_vae"])],
            "curves": history,
            "terminal_status": terminal,
            "successful_updates": len(history["normal_vae"]),
            "attempts": attempts,
            "order_sha256": _sha256_rows(order),
            "logical_manifest_sha256": context["logical_manifest_sha256"],
            "physical_pointer_prefix_sha256": access_prefix.hexdigest(),
            "physical_pointer_rows": accessed_rows,
            "checkpoint_chunk_size": None,
        }
    finally:
        _close_context(context)
        torch.cuda.empty_cache()


def _run_confirmation_task(
    *,
    source_root,
    config,
    source_roots,
    task,
    peak,
    attempt,
    epochs=2,
    phase="confirmation",
):
    import torch

    from eqvae.training.supervised_calibration import (
        PairedNumericalError,
        commit_paired_boundary,
        load_paired_boundary,
        paired_checkpoint_payload,
        update_ewma,
    )
    from eqvae.training.supervised_pairing import (
        confirmation_learning_rate,
        half_epoch_boundaries,
    )

    context = _task_context(
        source_root, config, source_roots, task, include_validation=True
    )
    task_root = WORKING_ROOT / task / attempt
    task_root.parent.mkdir(exist_ok=True)
    task_root.mkdir(exist_ok=False)
    try:
        branches = context["branches"]
        steps = cast("int", context["steps_per_epoch"])
        midpoint, endpoint = half_epoch_boundaries(steps)
        histories = {name: [] for name in branches}
        validation_history = {name: [] for name in branches}
        ewma = dict.fromkeys(branches)
        access_prefix = hashlib.sha256()
        accessed_rows = 0
        cursor = 0
        scale_backoff_budget = 1 if phase == "class_specific_scale_fix" else 0
        scale_backoff_events = []
        for epoch in range(epochs):
            epoch_order = _task_order(context, epoch=epoch)
            for within_epoch in range(1, steps + 1):
                row = _training_row(context, epoch_order, within_epoch - 1)
                accessed_rows += _extend_access_prefix(access_prefix, row)
                cursor += 1
                learning_rate = confirmation_learning_rate(
                    peak=peak,
                    successful_update=cursor,
                    steps_per_epoch=steps,
                )
                try:
                    observation = _paired_train_step(
                        branches,
                        row,
                        learning_rate,
                        task,
                        context.get("class_weights"),
                        max_scale_backoffs=(
                            scale_backoff_budget - len(scale_backoff_events)
                        ),
                    )
                except PairedNumericalError as error:
                    if error.details.get("scale_backoffs_used"):
                        scale_backoff_events.append({
                            "update": cursor,
                            "scales_before": error.details["scales_initial"],
                            "scales_after": error.details["scales_for_attempt"],
                        })
                    return {
                        "task": task,
                        "attempt": attempt,
                        "status": "numerical_failure",
                        "failure": str(error),
                        "failure_diagnostics": error.details,
                        "peak": peak,
                        "attention_dim": context["attention_dim"],
                        "class_specific_attention": context["class_specific_attention"],
                        "failed_attempt": cursor,
                        "successful_updates": cursor - 1,
                        "learning_rate": learning_rate,
                        "initialization_sha256": context["initialization_sha256"],
                        "half_peak_fallback_eligible": attempt == "initial",
                        "insufficient_learning_fallback_eligible": False,
                        "histories": histories,
                        "validation_history": validation_history,
                        "scale_backoff_budget": scale_backoff_budget,
                        "scale_backoffs_used": len(scale_backoff_events),
                        "scale_backoffs_remaining": (
                            scale_backoff_budget - len(scale_backoff_events)
                        ),
                        "scale_backoff_events": scale_backoff_events,
                        "logical_manifest_sha256": context["logical_manifest_sha256"],
                        "physical_pointer_prefix_sha256": access_prefix.hexdigest(),
                        "physical_pointer_rows": accessed_rows,
                    }
                if observation["scale_backoffs"]:
                    scale_backoff_events.append({
                        "update": cursor,
                        "scales_before": observation["scales_before"],
                        "scales_after": observation["scales_after"],
                    })
                for name, loss in observation["losses"].items():
                    smoothed = update_ewma(ewma[name], loss)
                    ewma[name] = smoothed
                    histories[name].append({
                        "update": cursor,
                        "loss": loss,
                        "ewma": smoothed,
                    })
                if within_epoch in {midpoint, endpoint}:
                    metrics = _validate_context(context, task)
                    for name in branches:
                        validation_history[name].append(metrics[name])
                    fraction = epoch + (0.5 if within_epoch == midpoint else 1.0)
                    schedule = {
                        "kind": "warmup_then_hold",
                        "peak": peak,
                        "steps_per_epoch": steps,
                    }
                    if epochs != 2:
                        schedule["target_epoch"] = epochs
                    payload = paired_checkpoint_payload(
                        branches=branches,
                        task=f"{task}_{phase}",
                        epoch_fraction=fraction,
                        completed_epoch=epoch
                        if within_epoch == midpoint
                        else epoch + 1,
                        within_epoch_cursor=within_epoch,
                        successful_pair_count=cursor,
                        schedule=schedule,
                        order_identity={
                            "seed": context["seed"],
                            "epoch": epoch,
                            "sha256": _sha256_rows(epoch_order),
                            "cursor": within_epoch,
                        },
                        validation_history=validation_history,
                        training_history=histories,
                        ewma_state=ewma,
                        best_metrics=_best_validation(validation_history),
                        patience_state=_patience_state(validation_history),
                        access_transcript={
                            "logical_access": "train_and_validation_only",
                            "logical_manifest_sha256": context[
                                "logical_manifest_sha256"
                            ],
                            "current_order_sha256": _sha256_rows(epoch_order),
                            "within_epoch_cursor": within_epoch,
                            "physical_pointer_prefix_sha256": access_prefix.hexdigest(),
                            "physical_pointer_rows": accessed_rows,
                            "validation_pointer_sha256": context[
                                "validation_pointer_sha256"
                            ],
                            "validation_pointer_rows": context[
                                "validation_pointer_rows"
                            ],
                            "validation_checks": len(validation_history["normal_vae"]),
                        },
                        campaign_progress={
                            "phase": phase,
                            "task": task,
                            "peak": peak,
                            "attempt": attempt,
                            "scale_backoff_budget": scale_backoff_budget,
                            "scale_backoffs_used": len(scale_backoff_events),
                            "scale_backoffs_remaining": (
                                scale_backoff_budget - len(scale_backoff_events)
                            ),
                            "scale_backoff_events": scale_backoff_events,
                        },
                    )
                    boundary = commit_paired_boundary(
                        task_root, epoch_fraction=fraction, payload=payload
                    )
                    _, restored_cursor, restored_within_epoch = load_paired_boundary(
                        boundary,
                        branches,
                    )
                    if (restored_cursor, restored_within_epoch) != (
                        cursor,
                        within_epoch,
                    ):
                        raise RuntimeError(
                            "Paired boundary continuation cursor differs"
                        )
        warmup = (steps + 9) // 10
        passed = all(
            histories[name][-1]["ewma"] < histories[name][warmup - 1]["ewma"]
            for name in branches
        )
        return {
            "task": task,
            "attempt": attempt,
            "status": "pass" if passed else "insufficient_learning",
            "peak": peak,
            "steps_per_epoch": steps,
            "epochs": epochs,
            "warmup_updates": warmup,
            "initialization_sha256": context["initialization_sha256"],
            "attention_dim": context["attention_dim"],
            "class_specific_attention": context["class_specific_attention"],
            "histories": histories,
            "validation_history": validation_history,
            "learning_check_pass": passed,
            "half_peak_fallback_eligible": False,
            "insufficient_learning_fallback_eligible": False,
            "zero_scaler_skips": not scale_backoff_events,
            "scale_backoff_budget": scale_backoff_budget,
            "scale_backoffs_used": len(scale_backoff_events),
            "scale_backoffs_remaining": (
                scale_backoff_budget - len(scale_backoff_events)
            ),
            "scale_backoff_events": scale_backoff_events,
            "logical_access": "train_and_validation_only",
            "logical_manifest_sha256": context["logical_manifest_sha256"],
            "physical_pointer_prefix_sha256": access_prefix.hexdigest(),
            "physical_pointer_rows": accessed_rows,
            "validation_pointer_sha256": context["validation_pointer_sha256"],
            "validation_pointer_rows": context["validation_pointer_rows"],
        }
    finally:
        _close_context(context)
        torch.cuda.empty_cache()


def _task_context(source_root, config, source_roots, task, *, include_validation):
    import torch

    from eqvae.data.supervised_latents import (
        SupervisedLatentStore,
        TissueDataset,
        WSIBagDataset,
    )
    from eqvae.models.supervised import (
        TissueClassifier,
        make_attention_mil_width_variant,
        make_class_specific_attention_mil_variant,
    )
    from eqvae.training.supervised_calibration import make_branch_state
    from eqvae.training.supervised_pairing import make_paired_models

    assets = source_root
    task_config = cast("dict[str, object]", config["tasks"])[task]
    if task == "mil" and task_config.get("class_weight_application") != (
        "post_grad_scaler_unscale"
    ):
        raise RuntimeError("MIL class-weight application differs from Spec 0023")
    if task == "mil":
        attention_dim = int(task_config.get("attention_dim", 64))
        class_specific_attention = bool(
            task_config.get("class_specific_attention", False)
        )
        if class_specific_attention:
            factory = make_class_specific_attention_mil_variant
        else:
            factory = lambda: make_attention_mil_width_variant(
                attention_dim=attention_dim
            )
    else:
        factory = TissueClassifier
    normal, so2 = make_paired_models(
        factory, seed=cast("int", task_config["initialization_seed"])
    )
    if any(
        not torch.equal(value, so2.state_dict()[key])
        for key, value in normal.state_dict().items()
    ):
        raise RuntimeError("Paired classifier initialization differs")
    branches = {
        "normal_vae": make_branch_state(normal, device=torch.device("cuda:0")),
        "so2_vae": make_branch_state(so2, device=torch.device("cuda:1")),
    }
    initialization_sha256 = _initialization_sha256(branches)
    expected_initialization = task_config.get("expected_initialization_sha256")
    if (
        expected_initialization is not None
        and initialization_sha256 != expected_initialization
    ):
        raise RuntimeError("Paired classifier initialization identity differs")
    if task == "mil":
        attention_maps = 5 if class_specific_attention else 1
        expected_shapes = {
            "attention_v": (attention_dim, 128),
            "attention_u": (attention_dim, 128),
            "attention_w": (attention_maps, attention_dim),
            "head": (5, 128),
        }
        for branch in branches.values():
            observed_shapes = {
                name: tuple(getattr(branch.model, name).weight.shape)
                for name in expected_shapes
            }
            if observed_shapes != expected_shapes:
                raise RuntimeError("MIL scorer width differs from configuration")
    stores = {
        name: SupervisedLatentStore(
            catalog_path=assets / "physical_parts.csv",
            model_name=cast("str", name),
            source_roots=source_roots,
        )
        for name in branches
    }
    if task == "mil":
        train = {
            name: WSIBagDataset(
                instance_path=assets / cast("str", task_config["train_instances"]),
                bag_path=assets / cast("str", task_config["train_bags"]),
                store=stores[name],
            )
            for name in branches
        }
        validation = None
        if include_validation:
            validation = {
                name: WSIBagDataset(
                    instance_path=assets
                    / cast("str", task_config["validation_instances"]),
                    bag_path=assets / cast("str", task_config["validation_bags"]),
                    store=stores[name],
                )
                for name in branches
            }
        access_rows = sum(len(item.instances) for item in train["normal_vae"])
        counts = {}
        for bag in train["normal_vae"].bags:
            counts[bag.diagnosis_index] = counts.get(bag.diagnosis_index, 0) + 1
        class_weights = {
            label: len(train["normal_vae"]) / (5 * count)
            for label, count in counts.items()
        }
    else:
        train = {
            name: TissueDataset(
                path=assets / cast("str", task_config["train_instances"]),
                split="train",
                store=stores[name],
            )
            for name in branches
        }
        validation = None
        if include_validation:
            validation = {
                name: TissueDataset(
                    path=assets / cast("str", task_config["validation_instances"]),
                    split="validation",
                    store=stores[name],
                )
                for name in branches
            }
        access_rows = len(train["normal_vae"])
        class_weights = None
    validation_pointer_sha256 = None
    validation_pointer_rows = 0
    if validation is not None:
        validation_pointer_sha256, validation_pointer_rows = (
            _paired_dataset_pointer_identity(validation)
        )
    return {
        "task": task,
        "seed": task_config["initialization_seed"],
        "branches": branches,
        "stores": stores,
        "train": train,
        "validation": validation,
        "steps_per_epoch": 106 if task == "mil" else 132,
        "access_rows": access_rows,
        "class_weights": class_weights,
        "initialization_sha256": initialization_sha256,
        "attention_dim": attention_dim if task == "mil" else None,
        "class_specific_attention": (
            class_specific_attention if task == "mil" else None
        ),
        "logical_manifest_sha256": _task_manifest_hashes(
            config,
            task_config,
            include_validation=include_validation,
        ),
        "validation_pointer_sha256": validation_pointer_sha256,
        "validation_pointer_rows": validation_pointer_rows,
    }


def _task_order(context, *, epoch):
    from eqvae.training.supervised_pairing import (
        nested_tissue_epoch_order,
        paired_epoch_order,
    )

    train = context["train"]
    if context["task"] == "mil":
        return paired_epoch_order(
            row_count=len(train["normal_vae"]), epoch=epoch, seed=1701
        )
    full_rows = tuple(range(len(train["normal_vae"])))
    return nested_tissue_epoch_order(full_rows, full_rows, epoch=epoch)


def _training_row(context, order, update_index):
    train = context["train"]
    if context["task"] == "mil":
        return _paired_wsi_row(train, order[update_index])
    return _paired_tissue_row(train, order, update_index * 128)


def _task_manifest_hashes(config, task_config, *, include_validation):
    hashes = cast("dict[str, str]", config["logical_manifest_sha256"])
    names = ["physical_parts.csv", cast("str", task_config["train_instances"])]
    if "train_bags" in task_config:
        names.append(cast("str", task_config["train_bags"]))
    if include_validation:
        names.append(cast("str", task_config["validation_instances"]))
        if "validation_bags" in task_config:
            names.append(cast("str", task_config["validation_bags"]))
    return {name: hashes[name] for name in names}


def _paired_wsi_row(train, index):
    return {name: dataset[index] for name, dataset in train.items()}


def _paired_tissue_row(train, order, offset):
    indices = order[offset : offset + 128]
    return {name: dataset.read_batch(indices) for name, dataset in train.items()}


def _extend_access_prefix(digest, row):
    normal = _logical_pointer_rows(row["normal_vae"])
    so2 = _logical_pointer_rows(row["so2_vae"])
    if normal != so2:
        raise RuntimeError("Paired physical pointer identity differs")
    for identity in normal:
        digest.update(json.dumps(identity, separators=(",", ":")).encode())
        digest.update(b"\n")
    return len(normal)


def _logical_pointer_rows(loaded):
    instances = loaded.instances
    return tuple(
        (
            instance.atlas_row_index,
            instance.wsi_id,
            instance.x,
            instance.y,
            instance.pointer.part,
            instance.pointer.file_index,
        )
        for instance in instances
    )


def _paired_dataset_pointer_identity(datasets):
    identities = []
    for name in ("normal_vae", "so2_vae"):
        dataset = datasets[name]
        rows = tuple(
            (
                instance.atlas_row_index,
                instance.wsi_id,
                instance.x,
                instance.y,
                instance.pointer.part,
                instance.pointer.file_index,
            )
            for instance in dataset.instances
        )
        identities.append(rows)
    if identities[0] != identities[1]:
        raise RuntimeError("Paired validation pointer identity differs")
    digest = hashlib.sha256()
    for identity in identities[0]:
        digest.update(json.dumps(identity, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest(), len(identities[0])


def _best_validation(histories):
    return {
        name: {
            **history[best_index],
            "check_index": best_index,
        }
        for name, history in histories.items()
        for best_index in [
            max(
                range(len(history)),
                key=lambda index: (
                    history[index]["macro_f1"],
                    -history[index]["loss"],
                    -index,
                ),
            )
        ]
    }


def _patience_state(histories):
    best = _best_validation(histories)
    return {
        name: len(history) - 1 - best[name]["check_index"]
        for name, history in histories.items()
    }


def _paired_train_step(
    branches,
    row,
    learning_rate,
    task,
    class_weights,
    *,
    weight_application="post_grad_scaler_unscale",
    max_scale_backoffs=0,
):
    import torch
    from torch.nn import functional

    from eqvae.training.supervised_calibration import (
        PairedNumericalError,
        paired_atomic_step,
    )

    if weight_application not in {"scaled_loss", "post_grad_scaler_unscale"}:
        raise ValueError("Unknown MIL class-weight application")

    forward_diagnostics = {}
    weights = (
        {
            name: cast("dict[int, float]", class_weights)[loaded.bag.diagnosis_index]
            for name, loaded in row.items()
        }
        if task == "mil"
        else {}
    )

    def loss_for(name):
        branch = branches[name]
        loaded = row[name]
        with torch.autocast("cuda", dtype=torch.float16, cache_enabled=True):
            if task == "mil":
                logits, attention = branch.model(
                    loaded.latents.to(branch.device), checkpoint_chunk_size=None
                )
                target = torch.tensor(
                    [loaded.bag.diagnosis_index], device=branch.device
                )
                loss = functional.cross_entropy(logits.float()[None, :], target)
                weight = weights[name]
                forward_diagnostics[name] = {
                    "logits": _tensor_numerical_summary(logits),
                    "attention": _tensor_numerical_summary(attention),
                    "unweighted_cross_entropy": _tensor_numerical_summary(loss),
                    "class_weight": weight,
                    "weighted_loss": float((loss.detach() * weight).item()),
                }
                return loss * weight if weight_application == "scaled_loss" else loss
            logits = branch.model(loaded.latents.to(branch.device))
            labels = loaded.labels.to(branch.device)
            return functional.cross_entropy(logits.float(), labels)

    try:
        return paired_atomic_step(
            branches,
            {name: (lambda name=name: loss_for(name)) for name in branches},
            learning_rate=learning_rate,
            loss_weights=(
                None
                if task != "mil" or weight_application == "scaled_loss"
                else weights
            ),
            max_scale_backoffs=max_scale_backoffs,
        ).__dict__
    except PairedNumericalError as error:
        error.details.update({
            "weight_application": weight_application,
            "forward": forward_diagnostics,
            "input": _paired_input_numerical_summary(row, task, class_weights),
        })
        raise


def _paired_input_numerical_summary(row, task, class_weights):
    result = {
        name: _tensor_numerical_summary(loaded.latents) for name, loaded in row.items()
    }
    if task == "mil":
        bag = row["normal_vae"].bag
        result["logical_sample"] = {
            "wsi_id": bag.wsi_id,
            "diagnosis_label": bag.diagnosis_label,
            "diagnosis_index": bag.diagnosis_index,
            "instance_count": bag.instance_count,
            "class_weight": cast("dict[int, float]", class_weights)[
                bag.diagnosis_index
            ],
            "checkpoint_chunk_size": None,
        }
    return result


def _tensor_numerical_summary(tensor):
    import torch

    detached = tensor.detach()
    finite = torch.isfinite(detached)
    finite_values = detached[finite]
    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "nan_count": int(torch.isnan(detached).sum().item()),
        "positive_inf_count": int(torch.isposinf(detached).sum().item()),
        "negative_inf_count": int(torch.isneginf(detached).sum().item()),
        "finite_max_abs": (
            float(finite_values.abs().max().item()) if finite_values.numel() else None
        ),
    }


def _validate_context(context, task):
    import torch
    from torch.nn import functional

    results = {}
    for name, branch in context["branches"].items():
        branch.model.eval()
        labels = []
        predictions = []
        total_loss = 0.0
        with torch.no_grad():
            dataset = context["validation"][name]
            if task == "mil":
                for index in range(len(dataset)):
                    loaded = dataset[index]
                    with torch.autocast(
                        "cuda", dtype=torch.float16, cache_enabled=True
                    ):
                        logits, _ = branch.model(
                            loaded.latents.to(branch.device), checkpoint_chunk_size=None
                        )
                    target = loaded.bag.diagnosis_index
                    total_loss += float(
                        functional.cross_entropy(
                            logits.float()[None, :],
                            torch.tensor([target], device=branch.device),
                        ).item()
                    )
                    labels.append(target)
                    predictions.append(int(logits.argmax().item()))
            else:
                for offset in range(0, len(dataset), 128):
                    loaded = dataset.read_batch(
                        tuple(range(offset, min(offset + 128, len(dataset))))
                    )
                    with torch.autocast(
                        "cuda", dtype=torch.float16, cache_enabled=True
                    ):
                        logits = branch.model(loaded.latents.to(branch.device))
                    targets = loaded.labels.to(branch.device)
                    total_loss += float(
                        functional.cross_entropy(
                            logits.float(), targets, reduction="sum"
                        ).item()
                    )
                    labels.extend(loaded.labels.tolist())
                    predictions.extend(logits.argmax(dim=1).cpu().tolist())
        branch.model.train()
        results[name] = {
            "loss": total_loss / len(labels),
            "macro_f1": _macro_f1(labels, predictions),
        }
    return results


def _macro_f1(labels, predictions):
    classes = sorted(set(labels))
    scores = []
    for label in classes:
        true_positive = sum(
            actual == label and predicted == label
            for actual, predicted in zip(labels, predictions)
        )
        false_positive = sum(
            actual != label and predicted == label
            for actual, predicted in zip(labels, predictions)
        )
        false_negative = sum(
            actual == label and predicted != label
            for actual, predicted in zip(labels, predictions)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return sum(scores) / len(scores)


def _close_context(context):
    for store in context["stores"].values():
        store.close()


def _initialization_sha256(branches):
    digest = hashlib.sha256()
    for name in ("normal_vae", "so2_vae"):
        for key, value in branches[name].model.state_dict().items():
            digest.update(key.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _sha256_rows(rows):
    return hashlib.sha256("\n".join(str(row) for row in rows).encode()).hexdigest()


def _resolve_source_roots(config: dict[str, object]) -> dict[str, Path]:
    roots = {}
    for record in cast("list[dict[str, object]]", config["source_records"]):
        source = cast("str", record["kaggle_source"])
        parents = set()
        for binary in cast("list[dict[str, object]]", record["binaries"]):
            name = cast("str", binary["name"])
            matches = [path for path in INPUT_ROOT.rglob(name) if path.is_file()]
            if len(matches) != 1:
                raise RuntimeError(
                    f"Expected exactly one mounted {name}, found {len(matches)}"
                )
            if matches[0].stat().st_size != cast("int", binary["bytes"]):
                raise RuntimeError(f"Mounted byte count differs for {name}")
            parents.add(matches[0].parent)
        if len(parents) != 1:
            raise RuntimeError(f"Mounted binaries for {source} do not share one root")
        roots[source] = parents.pop()
    return roots


def _resolve_calibration_input(config: dict[str, object]) -> Path:
    matches = [path for path in INPUT_ROOT.rglob(INPUT_CONTRACT_NAME) if path.is_file()]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one mounted calibration contract, found {len(matches)}"
        )
    contract_path = matches[0]
    contract_bytes = contract_path.read_bytes()
    receipt = cast("dict[str, object]", config["input_dataset_receipt"])
    if hashlib.sha256(contract_bytes).hexdigest() != receipt["input_contract_sha256"]:
        raise RuntimeError("Mounted calibration contract differs from its receipt")
    contract = cast("object", json.loads(contract_bytes))
    if not isinstance(contract, dict):
        raise TypeError("Mounted calibration input contract must be an object")
    typed = cast("dict[str, object]", contract)
    mode = config["package_mode"]
    expected_identity = {
        "schema_version": "spec0023.supervised_calibration_input_contract.v1",
        "package_mode": mode,
        "dataset_reference": receipt["dataset_reference"],
        "spec_sha256": config["spec_sha256"],
        "supervised_manifest_audit_sha256": config["supervised_manifest_audit_sha256"],
        "logical_manifest_sha256": config["logical_manifest_sha256"],
    }
    if any(typed.get(key) != value for key, value in expected_identity.items()) or (
        _source_tree_sha256(typed) != config["input_source_tree_sha256"]
    ):
        raise RuntimeError("Mounted calibration input contract identity differs")
    files = cast("dict[str, dict[str, object]]", typed["files"])
    root = contract_path.parent
    for logical_name, record in files.items():
        relative = Path(logical_name)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"Unsafe calibration input path: {logical_name}")
        path = root / relative
        if (
            not path.is_file()
            or path.stat().st_size != record["bytes"]
            or _sha256_file(path) != record["sha256"]
        ):
            raise RuntimeError(f"Mounted calibration input differs: {logical_name}")
    if mode == "sweep" and set(
        cast("dict[str, str]", typed["logical_manifest_sha256"])
    ) != {
        "physical_parts.csv",
        "wsi/wsi_cancer_train_instances.csv",
        "wsi/wsi_cancer_train_bags.csv",
        "tissue/tissue_train_5671_per_class.csv",
    }:
        raise RuntimeError("Sweep input surface differs from train-only manifests")
    return root


def _source_tree_sha256(contract: dict[str, object]) -> str:
    files = cast("dict[str, dict[str, object]]", contract["files"])
    source_records = {
        name: record
        for name, record in files.items()
        if name.startswith("src/eqvae/") and name.endswith(".py")
    }
    return hashlib.sha256(_canonical_json(source_records)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _embedded_config() -> dict[str, object]:
    payload = base64.b64decode(EMBEDDED_CONFIG_B64.encode("ascii"))
    if hashlib.sha256(payload).hexdigest() != EMBEDDED_CONFIG_SHA256:
        raise RuntimeError("Spec 0023 calibration config hash mismatch")
    value = cast("object", json.loads(payload))
    if not isinstance(value, dict):
        raise TypeError("Spec 0023 calibration config must be an object")
    return cast("dict[str, object]", value)


def _write_audit(result: dict[str, object], config: dict[str, object]) -> None:
    result["config_sha256"] = hashlib.sha256(_canonical_json(config)).hexdigest()
    result["output_allowlist"] = config["output_allowlist"]
    OUTPUT_PATH.write_bytes(_canonical_json(result))
    print(json.dumps(result, indent=2, sort_keys=True))


def _canonical_json(payload: dict[str, object]) -> bytes:
    return f"{json.dumps(payload, sort_keys=True, separators=(',', ':'))}\n".encode()


if __name__ == "__main__":
    raise SystemExit(main())
