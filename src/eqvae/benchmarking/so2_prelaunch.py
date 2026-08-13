# Copyright 2026 HiperMaximus
"""Strict local identity/verdict checks for the one-off Spec 0016 prelaunch."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import subprocess  # noqa: S404
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch import nn

    from eqvae.benchmarking.io import JsonObject

IDENTITY_ROOTS = (
    Path("src/eqvae"),
    Path("configs/spec0001"),
    Path("configs/spec0016"),
)
IDENTITY_FILES = (
    Path("docs/data/ubc_ocean_masked_holdout_ids.csv"),
    Path("kaggle/kernels/so2_prelaunch/kernel-metadata.json"),
    Path("kaggle/kernels/so2_prelaunch/run_template.py"),
    Path("kaggle/kernels/so2_selected_runtime_full/kernel-metadata.json"),
    Path("kaggle/kernels/so2_selected_runtime_full/run_template.py"),
    Path("pyproject.toml"),
    Path("uv.lock"),
)
_WORLD_SIZE = 2
_SETTLED_UPDATES = 20
_SHA1_HEX_LENGTH = 40
_GIT = shutil.which("git") or "/usr/bin/git"


def execution_identity(repo_root: Path) -> dict[str, str]:
    """Hash every execution-bearing prelaunch/full input.

    Returns:
        Relative-path to SHA-256 mapping.

    """
    identity = {str(path): _tree_sha256(repo_root / path) for path in IDENTITY_ROOTS}
    identity.update({str(path): _sha256(repo_root / path) for path in IDENTITY_FILES})
    return identity


def validate_prelaunch_verdict(  # noqa: C901, PLR0912, PLR0914, PLR0915
    path: Path,
    *,
    repo_root: Path,
    expected_source_commit: str | None = None,
) -> tuple[str, ...]:
    """Reject failed, incomplete, unstable, or stale prelaunch evidence.

    Returns:
        Concrete blocker strings; empty means the artifact is current and valid.

    """
    payload = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(payload, dict):
        return ("prelaunch_verdict_not_object",)
    verdict = cast("JsonObject", payload)
    blockers: list[str] = []
    if verdict.get("schema_version") != "spec0016.so2_prelaunch_verdict.v1":
        blockers.append("prelaunch_schema_mismatch")
    if verdict.get("status") != "pass":
        blockers.append("prelaunch_status_not_pass")
    if verdict.get("full_run_eligible") is not False:
        blockers.append("prelaunch_must_not_claim_full_run_eligible")
    source_commit = verdict.get("source_git_commit")
    expected_commit = expected_source_commit or _git_commit(repo_root)
    if (
        not isinstance(source_commit, str)
        or len(source_commit) != _SHA1_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        blockers.append("prelaunch_source_commit_invalid")
    elif source_commit != expected_commit:
        blockers.append("prelaunch_source_commit_stale")
    if verdict.get("source_git_dirty") is not False:
        blockers.append("prelaunch_source_must_be_clean")
    checks = verdict.get("checks")
    if not isinstance(checks, dict) or checks != {
        "debug": True,
        "resume": True,
        "tiny": True,
        "gates": True,
        "performance": True,
    }:
        blockers.append("prelaunch_nested_checks_invalid")
    selected = verdict.get("selected_runtime")
    if not isinstance(selected, dict) or selected != {
        "per_device_batch_size": 25,
        "global_batch_size": 50,
    }:
        blockers.append("prelaunch_batch_coordinate_mismatch")
    metrics = verdict.get("settled_real_loader_rank_metrics")
    means: list[float] = []
    if not isinstance(metrics, list) or len(metrics) != _WORLD_SIZE:
        blockers.append("prelaunch_rank_metrics_missing")
    else:
        ranks = [row.get("rank") for row in metrics if isinstance(row, dict)]
        if len(ranks) != _WORLD_SIZE or sorted(
            rank
            for rank in ranks
            if isinstance(rank, int) and not isinstance(rank, bool)
        ) != [0, 1]:
            blockers.append("prelaunch_rank_metrics_identity_mismatch")
        for row in metrics:
            if not isinstance(row, dict):
                blockers.append("prelaunch_rank_metric_not_object")
                continue
            metric = cast("dict[str, object]", row)
            step_samples = metric.get("settled_step_ms")
            wait_samples = metric.get("data_wait_ms")
            numeric = (
                metric.get("mean_step_ms"),
                metric.get("mean_data_wait_ms"),
                metric.get("data_wait_fraction"),
                metric.get("peak_allocated_mib"),
                metric.get("peak_reserved_mib"),
                metric.get("total_device_memory_mib"),
                metric.get("reserved_headroom_fraction"),
            )
            numeric_valid = all(_finite_number(value) for value in numeric)
            step_values = _numeric_samples(step_samples, positive=True)
            wait_values = _numeric_samples(wait_samples, positive=False)
            if not numeric_valid or step_values is None or wait_values is None:
                blockers.append("prelaunch_rank_metric_invalid")
                continue
            mean_step = float(cast("int | float", metric["mean_step_ms"]))
            mean_wait = float(cast("int | float", metric["mean_data_wait_ms"]))
            wait_fraction = float(
                cast("int | float", metric["data_wait_fraction"]),
            )
            allocated = float(cast("int | float", metric["peak_allocated_mib"]))
            reserved = float(cast("int | float", metric["peak_reserved_mib"]))
            total = float(cast("int | float", metric["total_device_memory_mib"]))
            headroom = float(
                cast("int | float", metric["reserved_headroom_fraction"]),
            )
            if (
                metric.get("settled_update_count") != _SETTLED_UPDATES  # noqa: PLR0916
                or metric.get("post_settle_graph_break_count") != 0
                or metric.get("post_settle_recompile_count") != 0
                or mean_step <= 0.0
                or mean_wait < 0.0
                or not math.isclose(
                    mean_step,
                    sum(map(float, step_values)) / _SETTLED_UPDATES,
                )
                or not math.isclose(
                    mean_wait,
                    sum(map(float, wait_values)) / _SETTLED_UPDATES,
                )
                or not math.isclose(wait_fraction, mean_wait / mean_step)
                or allocated < 0.0
                or reserved < allocated
                or total <= 0.0
                or reserved > total
                or not math.isclose(headroom, (total - reserved) / total)
            ):
                blockers.append("prelaunch_rank_metric_invalid")
                continue
            means.append(mean_step)
    identity = verdict.get("execution_identity_sha256")
    if identity != execution_identity(repo_root):
        blockers.append("prelaunch_execution_identity_stale")
    projected = verdict.get("projected_epoch_seconds")
    slower = verdict.get("slower_rank_mean_step_ms")
    if (
        not isinstance(projected, int | float)
        or isinstance(projected, bool)
        or not math.isfinite(float(projected))
        or float(projected) <= 0.0
    ):
        blockers.append("prelaunch_projected_epoch_invalid")
    elif (
        not isinstance(slower, int | float)  # noqa: PLR0916
        or isinstance(slower, bool)
        or not math.isfinite(float(slower))
        or len(means) != _WORLD_SIZE
        or not math.isclose(float(slower), max(means))
        or not math.isclose(float(projected), max(means) * 6.0)
    ):
        blockers.append("prelaunch_projected_epoch_inconsistent")
    return tuple(blockers)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    for item in sorted(
        candidate for candidate in path.rglob("*") if candidate.is_file()
    ):
        if "__pycache__" in item.parts or item.suffix in {".pyc", ".pyo"}:
            continue
        hasher.update(item.relative_to(path).as_posix().encode())
        hasher.update(b"\0")
        hasher.update(_sha256(item).encode("ascii"))
        hasher.update(b"\0")
    return hasher.hexdigest()


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _numeric_samples(value: object, *, positive: bool) -> tuple[float, ...] | None:
    if not isinstance(value, list):
        return None
    items = cast("list[object]", value)
    if len(items) != _SETTLED_UPDATES or not all(
        _finite_number(item)
        and (
            float(cast("int | float", item)) > 0.0
            if positive
            else float(cast("int | float", item)) >= 0.0
        )
        for item in items
    ):
        return None
    return tuple(float(cast("int | float", item)) for item in items)


def _git_commit(repo_root: Path) -> str:
    return subprocess.run(  # noqa: S603
        (_GIT, "rev-parse", "HEAD"),
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate_prelaunch_artifacts(
    artifact_root: Path,
    *,
    repo_root: Path,
    expected_source_commit: str | None = None,
) -> tuple[str, ...]:
    """Validate the compact verdict and every load-bearing downloaded artifact.

    Returns:
        Concrete blocker strings; empty means the proof tree is consistent and valid.

    """
    verdict_path = artifact_root / "benchmark/so2_prelaunch_verdict.json"
    blockers = list(
        validate_prelaunch_verdict(
            verdict_path,
            repo_root=repo_root,
            expected_source_commit=expected_source_commit,
        ),
    )
    verdict = _json_object(verdict_path)
    raw_phase_hashes = verdict.get("phase_artifact_manifest_sha256")
    phase_hashes = (
        cast("dict[str, object]", raw_phase_hashes)
        if isinstance(raw_phase_hashes, dict)
        else {}
    )
    if not phase_hashes:
        blockers.append("prelaunch_phase_manifest_hashes_missing")
    phases = {
        "debug_phase1": artifact_root / "debug_phase1",
        "debug_resume": artifact_root / "debug_resume",
        "tiny_overfit": artifact_root / "tiny_overfit",
    }
    for name, phase in phases.items():
        manifest_path = phase / "benchmark/artifact_manifest.json"
        if not manifest_path.is_file() or phase_hashes.get(name) != _sha256(
            manifest_path,
        ):
            blockers.append(f"prelaunch_{name}_manifest_hash_mismatch")
            continue
        blockers.extend(_phase_errors(phase, tiny=name == "tiny_overfit"))
    blockers.extend(
        _debug_resume_errors(
            phases["debug_resume"],
            source_checkpoint=phases["debug_phase1"] / "checkpoints/step_000004.pt",
        ),
    )
    blockers.extend(
        _gate_csv_errors(phases["tiny_overfit"] / "metrics/gate_health.csv"),
    )
    blockers.extend(
        _cost_link_errors(
            verdict,
            _json_object(
                phases["tiny_overfit"] / "benchmark/tiny_overfit_summary.json",
            ),
        ),
    )
    return tuple(blockers)


def _cost_link_errors(
    verdict: dict[str, object],
    tiny_summary: dict[str, object],
) -> list[str]:
    linked_fields = (
        "settled_real_loader_rank_metrics",
        "slower_rank_mean_step_ms",
        "projected_epoch_seconds",
    )
    if all(verdict.get(field) == tiny_summary.get(field) for field in linked_fields):
        return []
    return ["prelaunch_cost_evidence_mismatch"]


def _phase_errors(phase: Path, *, tiny: bool) -> list[str]:  # noqa: C901
    manifest = _json_object(phase / "benchmark/artifact_manifest.json")
    errors: list[str] = []
    if (
        manifest.get("status") != "local_pass"
        or manifest.get("missing_artifacts") != []
    ):
        errors.append("prelaunch_phase_manifest_invalid")
    raw_hashes = manifest.get("artifact_hashes")
    if not isinstance(raw_hashes, dict):
        return [*errors, "prelaunch_phase_artifact_hashes_missing"]
    hashes = cast("dict[str, object]", raw_hashes)
    required = {
        "training_summary",
        "selected_runtime_plan_applied",
        "checkpoint_resume_proof",
        "gate_health_summary",
        "local_selected_runtime_readiness",
        "train_steps",
        "gate_health",
        "selected_runtime_debug_summary",
    }
    if tiny:
        required.add("tiny_overfit_summary")
    if not required.issubset(hashes):
        errors.append("prelaunch_phase_required_artifact_hash_missing")
    for name, expected in hashes.items():
        if not isinstance(expected, str):
            errors.append("prelaunch_phase_artifact_hash_invalid")
            continue
        path = _manifest_artifact_path(phase, name)
        if not path.is_file() or _sha256(path) != expected:
            errors.append(f"prelaunch_phase_artifact_hash_mismatch:{name}")
    nested_paths = (
        "benchmark/training_summary.json",
        "benchmark/selected_runtime_plan_applied.json",
        "benchmark/checkpoint_resume_proof.json",
        "benchmark/gate_health_summary.json",
    )
    errors.extend(
        f"prelaunch_nested_status_invalid:{relative}"
        for relative in nested_paths
        if _json_object(phase / relative).get("status") != "local_pass"
    )
    summary = _json_object(phase / "benchmark/training_summary.json")
    raw_ddp = summary.get("ddp_rank_device_proof")
    raw_amp = summary.get("amp_execution")
    ddp = cast("dict[str, object]", raw_ddp) if isinstance(raw_ddp, dict) else {}
    if ddp.get("status") != "local_pass":
        errors.append("prelaunch_ddp_proof_invalid")
    expected_amp = {
        "enabled": True,
        "grad_scaler_enabled": True,
        "grad_scaler_init_scale": 16384.0,
        "autocast_dtype": "float16",
        "requested_autocast_dtype": "float16",
        "local_amp_status": "executed_amp_fp16_conservative",
        "fp32_objective_island": True,
    }
    if not isinstance(raw_amp, dict) or raw_amp != expected_amp:
        errors.append("prelaunch_amp_proof_invalid")
    if (
        summary.get("amp_step_skipped_count") != 0
        or summary.get("nonfinite_count") != 0
    ):
        errors.append("prelaunch_training_finiteness_invalid")
    return errors


def _debug_resume_errors(phase: Path, *, source_checkpoint: Path) -> list[str]:
    proof = _json_object(phase / "benchmark/checkpoint_resume_proof.json")
    expected = {
        "status": "local_pass",
        "loaded_successful_optimizer_update_count": 4,
        "additional_optimizer_steps": 4,
        "final_optimizer_step": 8,
        "resume_sequence": "loaded_checkpoint_before_training_continued",
        "resume_checkpoint_sha256": (
            _sha256(source_checkpoint) if source_checkpoint.is_file() else ""
        ),
        "config_sha256_match": True,
        "runtime_config_sha256_match": True,
        "selected_row_id_match": True,
        "runtime_policy_id_match": True,
        "model_state_restored": True,
        "optimizer_state_restored": True,
        "grad_scaler_state_restored": True,
        "sampler_progress_restored": True,
    }
    if all(proof.get(key) == value for key, value in expected.items()):
        return []
    return ["prelaunch_debug_resume_proof_invalid"]


def _gate_csv_errors(path: Path) -> list[str]:
    if not path.is_file():
        return ["prelaunch_gate_csv_missing"]
    from eqvae.models.registry import MODEL_KIND_SO2_FIXED, build_model  # noqa: PLC0415
    from eqvae.models.so2_architecture_probe import FixedF01RadialGate  # noqa: PLC0415

    named_modules = cast(
        "Iterable[tuple[str, nn.Module]]",
        build_model(MODEL_KIND_SO2_FIXED).named_modules(),
    )
    expected: set[tuple[str, str]] = {
        (name, family)
        for name, module in named_modules
        if isinstance(module, FixedF01RadialGate)
        for family in ("f0_scalar", "f1_radial")
    }
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    observed: list[tuple[str, str]] = []
    valid = len(rows) == len(expected)
    positive_fields = (
        "input_rms",
        "output_rms",
        "a_grad_norm",
        "b_grad_norm",
        "a_update_to_param_norm",
        "b_update_to_param_norm",
    )
    for row in rows:
        family = row.get("gate_kind", "")
        module_field = row.get("module", "")
        suffix = f":{family}"
        module = module_field.removesuffix(suffix)
        observed.append((module, family))
        valid = valid and module_field == f"{module}:{family}"
        valid = valid and row.get("gate_health_status") == "pass"
        valid = valid and row.get("precision_proof_status") == "pass"
        valid = valid and row.get("input_dtype") == "float16"
        valid = valid and row.get("output_dtype") == "float16"
        valid = valid and row.get("gate_tensor_dtype") == "float16"
        valid = valid and row.get("gate_math_dtype") == "float32"
        valid = valid and all(
            _positive_finite_text(row.get(field)) for field in positive_fields
        )
    valid = valid and len(set(observed)) == len(expected) and set(observed) == expected
    return [] if valid else ["prelaunch_gate_csv_invalid"]


def _manifest_artifact_path(phase: Path, name: str) -> Path:
    if name.startswith("checkpoint:"):
        return phase / "checkpoints" / name.removeprefix("checkpoint:")
    mapping = {
        "training_summary": "benchmark/training_summary.json",
        "selected_runtime_plan_applied": "benchmark/selected_runtime_plan_applied.json",
        "checkpoint_resume_proof": "benchmark/checkpoint_resume_proof.json",
        "gate_health_summary": "benchmark/gate_health_summary.json",
        "local_selected_runtime_readiness": (
            "benchmark/local_selected_runtime_readiness.json"
        ),
        "selected_runtime_debug_summary": (
            "benchmark/selected_runtime_debug_summary.json"
        ),
        "tiny_overfit_summary": "benchmark/tiny_overfit_summary.json",
        "train_steps": "metrics/train_steps.csv",
        "gate_health": "metrics/gate_health.csv",
        "reconstruction_samples": "artifacts/reconstruction_samples.pt",
    }
    return phase / mapping.get(name, f"__unknown__/{name}")


def _json_object(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    value = cast("object", json.loads(path.read_text(encoding="utf-8")))
    return cast("dict[str, object]", value) if isinstance(value, dict) else {}


def _positive_finite_text(value: object) -> bool:
    try:
        number = float(cast("str", value))
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0.0


__all__ = [
    "IDENTITY_FILES",
    "IDENTITY_ROOTS",
    "execution_identity",
    "validate_prelaunch_artifacts",
    "validate_prelaunch_verdict",
]
