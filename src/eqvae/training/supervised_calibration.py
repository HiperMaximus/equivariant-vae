# Copyright 2026 HiperMaximus
# ruff: noqa: DOC201, DOC501, EM101, EM102, PLR0913, TRY003
"""Direct paired update and boundary mechanics for the one Spec 0023 calibration."""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.amp import GradScaler

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

BRANCH_NAMES: Final = ("normal_vae", "so2_vae")
SCALER_INIT_SCALE: Final = 32_768.0
SCALER_GROWTH_INTERVAL: Final = 1_000_000
WEIGHT_DECAY: Final = 1e-4
MATRIX_PARAMETER_NDIM: Final = 2
DIVERGENCE_MINIMUM_UPDATES: Final = 10


class PairedNumericalError(RuntimeError):
    """Stop a paired run before either branch commits an invalid update."""

    def __init__(self, message: str, *, details: Mapping[str, object]) -> None:
        """Retain the compact machine-readable evidence needed to stop safely."""
        super().__init__(message)
        self.details = dict(details)


@dataclass
class BranchState:
    """The small mutable state owned by one representation branch."""

    model: nn.Module
    optimizer: torch.optim.Optimizer
    scaler: GradScaler
    device: torch.device


@dataclass(frozen=True)
class PairedStepResult:
    """The aligned observation emitted after one committed paired update."""

    learning_rate: float
    losses: dict[str, float]
    scales_before: dict[str, float]
    scales_after: dict[str, float]
    scale_backoffs: int


def make_branch_state(model: nn.Module, *, device: torch.device) -> BranchState:
    """Move one fresh classifier to its device and apply the locked optimizer/AMP."""
    moved = model.to(device)
    decay = [
        parameter
        for parameter in moved.parameters()
        if parameter.requires_grad and parameter.ndim >= MATRIX_PARAMETER_NDIM
    ]
    no_decay = [
        parameter
        for parameter in moved.parameters()
        if parameter.requires_grad and parameter.ndim < MATRIX_PARAMETER_NDIM
    ]
    return BranchState(
        model=moved,
        optimizer=torch.optim.AdamW(
            [
                {
                    "name": "decay",
                    "params": decay,
                    "weight_decay": WEIGHT_DECAY,
                },
                {
                    "name": "no_decay",
                    "params": no_decay,
                    "weight_decay": 0.0,
                },
            ],
            lr=1e-5,
        ),
        scaler=GradScaler(
            "cuda",
            init_scale=SCALER_INIT_SCALE,
            growth_interval=SCALER_GROWTH_INTERVAL,
            enabled=device.type == "cuda",
        ),
        device=device,
    )


def paired_atomic_step(  # noqa: C901, PLR0912
    branches: Mapping[str, BranchState],
    losses: Mapping[str, Callable[[], Tensor]],
    *,
    learning_rate: float,
    loss_weights: Mapping[str, float] | None = None,
    max_scale_backoffs: int = 0,
) -> PairedStepResult:
    """Commit both finite branch updates or commit neither branch update.

    A MIL class weight is applied only after AMP unscales each FP32 parameter
    gradient.  This preserves the weighted optimization objective without
    multiplying the FP16 backward signal before unscaling.
    """
    _validate_branches(branches, losses)
    if not learning_rate > 0.0:
        raise ValueError("A paired update requires a positive learning rate")
    if max_scale_backoffs < 0:
        raise ValueError("max_scale_backoffs may not be negative")
    resolved_loss_weights = _resolve_loss_weights(loss_weights)
    initial_scales = {
        name: float(branches[name].scaler.get_scale()) for name in BRANCH_NAMES
    }
    if len(set(initial_scales.values())) != 1:
        raise ValueError("Paired branches require one shared GradScaler scale")
    scale_backoffs = 0
    while True:
        for name in BRANCH_NAMES:
            branch = branches[name]
            branch.optimizer.zero_grad(set_to_none=True)
            for group in branch.optimizer.param_groups:
                group["lr"] = learning_rate

        scales_for_attempt = {
            name: float(branches[name].scaler.get_scale()) for name in BRANCH_NAMES
        }
        loss_tensors: dict[str, Tensor] = {}
        for name in BRANCH_NAMES:
            loss = losses[name]()
            if loss.ndim != 0 or not bool(torch.isfinite(loss.detach()).item()):
                _discard_gradients(branches)
                raise PairedNumericalError(
                    f"{name} produced a nonfinite scalar loss",
                    details={
                        "kind": "scalar_loss",
                        "branch": name,
                        "loss_dtype": str(loss.dtype),
                        "scaler": scales_for_attempt[name],
                    },
                )
            branches[name].scaler.scale(loss).backward()  # pyright: ignore[reportUnknownMemberType]
            loss_tensors[name] = loss

        raw_gradients_finite = True
        for name in BRANCH_NAMES:
            branch = branches[name]
            branch.scaler.unscale_(branch.optimizer)
            raw_gradients_finite = raw_gradients_finite and _check_gradients(branch)
        if not raw_gradients_finite:
            branch_diagnostics = {
                name: _gradient_diagnostics(
                    branches[name],
                    loss_weight=resolved_loss_weights[name],
                    scaler=scales_for_attempt[name],
                )
                for name in BRANCH_NAMES
            }
            details = {
                "kind": "gradient",
                "branches": branch_diagnostics,
                "first_affected": _first_affected_gradient(branch_diagnostics),
                "scale_backoffs_used": scale_backoffs,
                "scales_initial": initial_scales,
                "scales_for_attempt": scales_for_attempt,
            }
            _discard_gradients(branches)
            if scale_backoffs >= max_scale_backoffs:
                raise PairedNumericalError(
                    "A paired branch produced a missing/nonfinite gradient",
                    details=details,
                )
            shared_scale = 0.5 * min(scales_for_attempt.values())
            for name in BRANCH_NAMES:
                branches[name].scaler.update(new_scale=shared_scale)
            if any(
                not np.isclose(
                    float(branches[name].scaler.get_scale()),
                    shared_scale,
                )
                for name in BRANCH_NAMES
            ):
                raise PairedNumericalError(
                    "Paired GradScalers did not accept the shared backoff",
                    details={
                        "kind": "scale_backoff",
                        "requested_scale": shared_scale,
                    },
                )
            scale_backoffs += 1
            continue

        for name in BRANCH_NAMES:
            _weight_gradients(
                branches[name],
                loss_weight=resolved_loss_weights[name],
            )
        if not all(_check_gradients(branches[name]) for name in BRANCH_NAMES):
            branch_diagnostics = {
                name: _gradient_diagnostics(
                    branches[name],
                    loss_weight=resolved_loss_weights[name],
                    scaler=scales_for_attempt[name],
                )
                for name in BRANCH_NAMES
            }
            details = {
                "kind": "weighted_gradient",
                "branches": branch_diagnostics,
                "first_affected": _first_affected_gradient(branch_diagnostics),
                "scale_backoffs_used": scale_backoffs,
                "scales_initial": initial_scales,
                "scales_for_attempt": scales_for_attempt,
            }
            _discard_gradients(branches)
            raise PairedNumericalError(
                "A paired class weight produced a missing/nonfinite gradient",
                details=details,
            )

        # Gradients and both scalers' recorded found-inf state are already unscaled and
        # synchronously checked together, so neither subsequent scaler step can discover
        # a one-sided skip after the other branch has committed.
        for name in BRANCH_NAMES:
            branches[name].scaler.step(branches[name].optimizer)
        for name in BRANCH_NAMES:
            branches[name].scaler.update()

        scales_after = {
            name: float(branches[name].scaler.get_scale()) for name in BRANCH_NAMES
        }
        if any(scales_after[name] < scales_for_attempt[name] for name in BRANCH_NAMES):
            raise PairedNumericalError(
                "GradScaler decreased after a prechecked paired step",
                details={
                    "kind": "post_step_scaler",
                    "scales_before": scales_for_attempt,
                    "scales_after": scales_after,
                },
            )
        return PairedStepResult(
            learning_rate=learning_rate,
            losses={
                name: float(
                    (loss_tensors[name].detach() * resolved_loss_weights[name]).item(),
                )
                for name in BRANCH_NAMES
            },
            scales_before=initial_scales,
            scales_after=scales_after,
            scale_backoffs=scale_backoffs,
        )


def update_ewma(previous: float | None, value: float, *, alpha: float = 0.1) -> float:
    """Apply the fixed loss smoothing used only to inspect the two sweep curves."""
    if not 0.0 < alpha <= 1.0 or not np.isfinite(value):
        raise ValueError("EWMA inputs must be finite with alpha in (0,1]")
    return value if previous is None else alpha * value + (1.0 - alpha) * previous


def ewma_has_diverged(history: list[float]) -> bool:
    """Apply the predeclared four-times-earlier-minimum rule after ten updates."""
    return len(history) >= DIVERGENCE_MINIMUM_UPDATES and history[-1] > 4.0 * min(
        history[:-1],
    )


def logical_identity_sha256(rows: list[tuple[object, ...]]) -> str:
    """Bind a supplied logical order without emitting a very large transcript row."""
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(row, separators=(",", ":"), ensure_ascii=True).encode("utf-8"),
        )
        digest.update(b"\n")
    return digest.hexdigest()


def paired_checkpoint_payload(
    *,
    branches: Mapping[str, BranchState],
    task: str,
    epoch_fraction: float,
    completed_epoch: int,
    within_epoch_cursor: int,
    successful_pair_count: int,
    schedule: Mapping[str, object],
    order_identity: Mapping[str, object],
    validation_history: Mapping[str, object],
    best_metrics: Mapping[str, object],
    patience_state: Mapping[str, object],
    access_transcript: Mapping[str, object],
    campaign_progress: Mapping[str, object],
    training_history: Mapping[str, object] | None = None,
    ewma_state: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build the full paired boundary payload required for exact continuation."""
    _validate_branch_names(branches)
    return {
        "schema_version": "spec0023.paired_supervised_checkpoint.v1",
        "task": task,
        "epoch_fraction": epoch_fraction,
        "completed_epoch": completed_epoch,
        "within_epoch_cursor": within_epoch_cursor,
        "successful_pair_count": successful_pair_count,
        "schedule": dict(schedule),
        "order_identity": dict(order_identity),
        "validation_history": dict(validation_history),
        "training_history": dict(training_history or {}),
        "ewma_state": dict(ewma_state or {}),
        "best_metrics": dict(best_metrics),
        "patience_state": dict(patience_state),
        "access_transcript": dict(access_transcript),
        "campaign_progress": dict(campaign_progress),
        "branches": {
            name: {
                "model": branches[name].model.state_dict(),
                "optimizer": branches[name].optimizer.state_dict(),
                "grad_scaler": branches[name].scaler.state_dict(),
                "device": str(branches[name].device),
            }
            for name in BRANCH_NAMES
        },
        "rng": _rng_state(),
    }


def commit_paired_boundary(
    output_root: Path,
    *,
    epoch_fraction: float,
    payload: Mapping[str, object],
) -> Path:
    """Expose one paired epoch-fraction directory only after every file is durable."""
    name = f"epoch_{epoch_fraction:.1f}"
    destination = output_root / name
    temporary = output_root / f".{name}.tmp"
    if destination.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to replace paired boundary {name}")
    output_root.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        _write_boundary_files(temporary, epoch_fraction=epoch_fraction, payload=payload)
        temporary.replace(destination)
    except BaseException:
        for path in temporary.glob("*"):
            path.unlink()
        temporary.rmdir()
        raise
    return destination


def restore_paired_checkpoint(
    branches: Mapping[str, BranchState],
    payload: Mapping[str, object],
) -> tuple[int, int]:
    """Restore both branches and the single paired cursor from one committed bundle."""
    _validate_branch_names(branches)
    if payload.get("schema_version") != "spec0023.paired_supervised_checkpoint.v1":
        raise ValueError("Unexpected paired supervised checkpoint schema")
    saved = cast("Mapping[str, Mapping[str, object]]", payload.get("branches"))
    _validate_branch_names(saved)
    for name in BRANCH_NAMES:
        branch = branches[name]
        branch.model.load_state_dict(cast("dict[str, Tensor]", saved[name]["model"]))
        branch.optimizer.load_state_dict(
            cast("dict[str, object]", saved[name]["optimizer"]),
        )
        branch.scaler.load_state_dict(
            cast("dict[str, object]", saved[name]["grad_scaler"]),
        )
    _restore_rng_state(cast("Mapping[str, object]", payload["rng"]))
    return int(cast("int", payload["successful_pair_count"])), int(
        cast("int", payload["within_epoch_cursor"]),
    )


def load_paired_boundary(
    boundary: Path,
    branches: Mapping[str, BranchState],
) -> tuple[dict[str, object], int, int]:
    """Verify a committed boundary bundle before restoring either branch."""
    checkpoint = boundary / "paired_checkpoint.pt"
    progress_path = boundary / "progress.json"
    manifest_path = boundary / "manifest.json"
    manifest = cast("object", json.loads(manifest_path.read_text(encoding="utf-8")))
    if not isinstance(manifest, dict):
        raise TypeError("Paired boundary manifest must be an object")
    manifest_object = cast("dict[str, object]", manifest)
    if set(manifest_object) != {
        "paired_checkpoint.pt",
        "progress.json",
    }:
        raise ValueError("Paired boundary manifest is incomplete")
    expected = cast("dict[str, str]", manifest_object)
    if expected != {
        "paired_checkpoint.pt": _sha256(checkpoint),
        "progress.json": _sha256(progress_path),
    }:
        raise ValueError("Paired boundary file hash differs")
    progress_value = cast(
        "object",
        json.loads(progress_path.read_text(encoding="utf-8")),
    )
    if not isinstance(progress_value, dict):
        raise TypeError("Paired boundary progress must be an object")
    progress = cast("dict[str, object]", progress_value)
    if (
        progress.get("schema_version") != "spec0023.paired_boundary.v1"
        or progress.get("checkpoint") != checkpoint.name
        or progress.get("checkpoint_sha256") != expected[checkpoint.name]
    ):
        raise ValueError("Paired boundary progress differs from its manifest")
    payload = cast(
        "dict[str, object]",
        torch.load(checkpoint, map_location="cpu", weights_only=False),
    )
    if payload.get("epoch_fraction") != progress.get("epoch_fraction"):
        raise ValueError("Paired boundary epoch fraction differs")
    successful, within_epoch = restore_paired_checkpoint(branches, payload)
    return payload, successful, within_epoch


def _write_boundary_files(
    temporary: Path,
    *,
    epoch_fraction: float,
    payload: Mapping[str, object],
) -> None:
    checkpoint = temporary / "paired_checkpoint.pt"
    torch.save(dict(payload), checkpoint)
    checkpoint_sha256 = _sha256(checkpoint)
    progress = {
        "schema_version": "spec0023.paired_boundary.v1",
        "epoch_fraction": epoch_fraction,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": checkpoint_sha256,
    }
    progress_path = temporary / "progress.json"
    progress_path.write_bytes(_canonical_json(progress))
    manifest = {
        "paired_checkpoint.pt": checkpoint_sha256,
        "progress.json": _sha256(progress_path),
    }
    (temporary / "manifest.json").write_bytes(_canonical_json(manifest))
    for path in temporary.iterdir():
        with path.open("rb") as handle:
            os.fsync(handle.fileno())


def _validate_branches(
    branches: Mapping[str, BranchState],
    losses: Mapping[str, Callable[[], Tensor]],
) -> None:
    _validate_branch_names(branches)
    _validate_branch_names(losses)


def _resolve_loss_weights(
    loss_weights: Mapping[str, float] | None,
) -> dict[str, float]:
    if loss_weights is None:
        return dict.fromkeys(BRANCH_NAMES, 1.0)
    _validate_branch_names(loss_weights)
    resolved = {name: float(loss_weights[name]) for name in BRANCH_NAMES}
    if any(not np.isfinite(weight) or weight <= 0.0 for weight in resolved.values()):
        raise ValueError("Paired loss weights must be finite and positive")
    return resolved


def _check_gradients(branch: BranchState) -> bool:
    finite_checks: list[Tensor] = []
    fp32_gradients = True
    for _, parameter in branch.model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            fp32_gradients = False
            continue
        if gradient.dtype != torch.float32:
            fp32_gradients = False
        finite_checks.append(torch.isfinite(gradient).all())
    return (
        bool(finite_checks)
        and fp32_gradients
        and bool(
            torch.stack(finite_checks).all().item(),
        )
    )


def _weight_gradients(branch: BranchState, *, loss_weight: float) -> None:
    for parameter in branch.model.parameters():
        gradient = parameter.grad
        if gradient is not None:
            gradient.mul_(loss_weight)


def _gradient_diagnostics(
    branch: BranchState,
    *,
    loss_weight: float,
    scaler: float,
) -> dict[str, object]:
    parameters: list[dict[str, object]] = []
    for parameter_name, parameter in branch.model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            parameters.append(
                {
                    "name": parameter_name,
                    "dtype": None,
                    "missing": True,
                    "nonfinite_count": None,
                    "max_finite_abs": None,
                },
            )
            continue
        finite_mask = torch.isfinite(gradient)
        nonfinite_count = int((~finite_mask).sum().item())
        finite_values = gradient[finite_mask]
        parameters.append(
            {
                "name": parameter_name,
                "dtype": str(gradient.dtype),
                "missing": False,
                "nonfinite_count": nonfinite_count,
                "max_finite_abs": (
                    float(finite_values.abs().max().item())
                    if finite_values.numel()
                    else None
                ),
            },
        )
    return {
        "scaler": scaler,
        "loss_weight": loss_weight,
        "parameters": parameters,
    }


def _first_affected_gradient(
    branches: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    for branch_name in BRANCH_NAMES:
        parameters = cast(
            "list[dict[str, object]]",
            branches[branch_name]["parameters"],
        )
        for parameter in parameters:
            if parameter["missing"] is True or cast(
                "int | None",
                parameter["nonfinite_count"],
            ) not in {None, 0}:
                return {"branch": branch_name, **parameter}
    raise RuntimeError("Gradient diagnostics did not locate the failed parameter")


def _validate_branch_names(values: Mapping[str, Any]) -> None:
    if tuple(values) != BRANCH_NAMES:
        raise ValueError(f"Paired mappings must be ordered exactly as {BRANCH_NAMES!r}")


def _discard_gradients(branches: Mapping[str, BranchState]) -> None:
    for branch in branches.values():
        branch.optimizer.zero_grad(set_to_none=True)


def _rng_state() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),  # noqa: NPY002
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }


def _restore_rng_state(state: Mapping[str, object]) -> None:
    if set(state) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise ValueError("Paired checkpoint RNG state is incomplete")
    random.setstate(cast("tuple[Any, ...]", state["python"]))
    np.random.set_state(cast("tuple[Any, ...]", state["numpy"]))  # noqa: NPY002
    torch.set_rng_state(cast("Tensor", state["torch_cpu"]))
    cuda_states = cast("list[Tensor]", state["torch_cuda"])
    if cuda_states:
        if not torch.cuda.is_available():
            raise ValueError("CUDA checkpoint RNG cannot be restored without CUDA")
        torch.cuda.set_rng_state_all(cuda_states)


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    return f"{json.dumps(payload, sort_keys=True, separators=(',', ':'))}\n".encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "BRANCH_NAMES",
    "BranchState",
    "PairedNumericalError",
    "PairedStepResult",
    "commit_paired_boundary",
    "ewma_has_diverged",
    "load_paired_boundary",
    "logical_identity_sha256",
    "make_branch_state",
    "paired_atomic_step",
    "paired_checkpoint_payload",
    "restore_paired_checkpoint",
    "update_ewma",
]
