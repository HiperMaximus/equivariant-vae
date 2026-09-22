# Copyright 2026 HiperMaximus
"""One atomic ``latest.pt`` checkpoint for the MIL dynamics probe."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

import numpy as np
import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from eqvae.training.mil_dynamics import AdamWTelemetry, Stateful

CHECKPOINT_FORMAT: Final = "spec0054.mil_probe_checkpoint.v1"
SESSION_LIMIT_SECONDS: Final = 12 * 60 * 60
SESSION_SAVE_MARGIN_SECONDS: Final = 15 * 60


@dataclass(frozen=True)
class RestoredDynamicsProgress:
    """Exact position from which the WSI loop continues."""

    committed_update: int
    exposure_count: int
    epoch: int
    within_epoch_cursor: int
    current_order: tuple[int, ...]
    effective_batch_size: int


def capture_rng_state() -> dict[str, Any]:
    """Capture process RNGs needed to reproduce the next update."""
    numpy_state = cast(
        "tuple[str, np.ndarray[Any, Any], int, int, float]",
        np.random.get_state(legacy=True),
    )
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": torch.from_numpy(numpy_state[1].copy()),
            "position": int(numpy_state[2]),
            "has_gaussian": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.random.get_rng_state().clone(),
        "torch_cuda": tuple(state.clone() for state in torch.cuda.get_rng_state_all())
        if torch.cuda.is_available()
        else (),
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore a state emitted by :func:`capture_rng_state`."""
    numpy_state = cast("Mapping[str, Any]", state["numpy"])
    numpy_values = cast("Tensor", numpy_state["state"])
    random.setstate(cast("tuple[Any, ...]", state["python"]))
    np.random.set_state(
        (
            cast("str", numpy_state["bit_generator"]),
            numpy_values.cpu().numpy().astype(np.uint32, copy=False),
            cast("int", numpy_state["position"]),
            cast("int", numpy_state["has_gaussian"]),
            cast("float", numpy_state["cached_gaussian"]),
        )
    )
    torch.random.set_rng_state(cast("Tensor", state["torch_cpu"]).cpu())
    cuda_states = cast("Sequence[Tensor]", state["torch_cuda"])
    if cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError("Cannot restore CUDA RNG state without CUDA")
        torch.cuda.set_rng_state_all(list(cuda_states))


def build_dynamics_checkpoint(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Stateful,
    scheduler: Stateful | None,
    telemetry: AdamWTelemetry,
    dynamics: Stateful,
    committed_update: int,
    exposure_count: int,
    epoch: int,
    within_epoch_cursor: int,
    current_order: Sequence[int],
    effective_batch_size: int,
) -> dict[str, Any]:
    """Capture the state at a committed optimizer boundary."""
    payload: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "model": copy_state_to_cpu(model.state_dict()),
        "optimizer": copy_state_to_cpu(optimizer.state_dict()),
        "scaler": copy_state_to_cpu(scaler.state_dict()),
        "scheduler": (
            copy_state_to_cpu(scheduler.state_dict()) if scheduler is not None else None
        ),
        "telemetry": telemetry.state_dict(),
        "dynamics": copy_state_to_cpu(dynamics.state_dict()),
        "committed_update": committed_update,
        "exposure_count": exposure_count,
        "epoch": epoch,
        "within_epoch_cursor": within_epoch_cursor,
        "current_order": tuple(int(value) for value in current_order),
        "effective_batch_size": effective_batch_size,
        "rng": capture_rng_state(),
    }
    validate_dynamics_checkpoint(payload)
    return payload


def validate_dynamics_checkpoint(payload: Mapping[str, Any]) -> None:
    """Check only invariants required for an exact continuation."""
    required = {
        "format",
        "model",
        "optimizer",
        "scaler",
        "scheduler",
        "telemetry",
        "dynamics",
        "committed_update",
        "exposure_count",
        "epoch",
        "within_epoch_cursor",
        "current_order",
        "effective_batch_size",
        "rng",
    }
    if set(payload) != required or payload["format"] != CHECKPOINT_FORMAT:
        raise ValueError("MIL probe checkpoint format differs")
    for name in ("committed_update", "exposure_count", "epoch", "within_epoch_cursor"):
        value = payload[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"Invalid checkpoint field: {name}")
    batch = payload["effective_batch_size"]
    if not isinstance(batch, int) or isinstance(batch, bool) or batch < 1:
        raise ValueError("Invalid effective batch size")
    order = payload["current_order"]
    if not isinstance(order, (tuple, list)) or not order:
        raise ValueError("Checkpoint order must be nonempty")
    if len(set(order)) != len(order) or set(order) != set(range(len(order))):
        raise ValueError("Checkpoint order is not a complete permutation")
    if payload["within_epoch_cursor"] > len(order):
        raise ValueError("Checkpoint cursor exceeds the current order")
    if payload["exposure_count"] < payload["committed_update"]:
        raise ValueError("Exposures cannot be fewer than committed updates")
    telemetry = payload["telemetry"]
    if (
        not isinstance(telemetry, dict)
        or telemetry.get("last_committed_update") != payload["committed_update"]
    ):
        raise ValueError("Telemetry state differs from checkpoint progress")


def save_dynamics_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace ``latest.pt`` or ``final.pt``."""
    validate_dynamics_checkpoint(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_dynamics_checkpoint(path: Path) -> dict[str, Any]:
    """Load one local checkpoint; the run directory supplies its identity."""
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise TypeError("MIL probe checkpoint must contain a mapping")
    payload = cast("dict[str, Any]", value)
    validate_dynamics_checkpoint(payload)
    return payload


def restore_dynamics_checkpoint(
    payload: Mapping[str, Any],
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Stateful,
    scheduler: Stateful | None,
    telemetry: AdamWTelemetry,
    dynamics: Stateful,
) -> RestoredDynamicsProgress:
    """Restore all state needed for the next WSI/update."""
    validate_dynamics_checkpoint(payload)
    model.load_state_dict(cast("dict[str, Tensor]", payload["model"]))
    optimizer.load_state_dict(cast("dict[str, Any]", payload["optimizer"]))
    scaler.load_state_dict(cast("dict[str, Any]", payload["scaler"]))
    scheduler_state = payload["scheduler"]
    if (scheduler is None) != (scheduler_state is None):
        raise ValueError("Checkpoint scheduler presence differs")
    if scheduler is not None:
        scheduler.load_state_dict(cast("dict[str, Any]", scheduler_state))
    telemetry.load_state_dict(cast("dict[str, Any]", payload["telemetry"]), model=model)
    dynamics.load_state_dict(cast("dict[str, Any]", payload["dynamics"]))
    restore_rng_state(cast("dict[str, Any]", payload["rng"]))
    return RestoredDynamicsProgress(
        committed_update=cast("int", payload["committed_update"]),
        exposure_count=cast("int", payload["exposure_count"]),
        epoch=cast("int", payload["epoch"]),
        within_epoch_cursor=cast("int", payload["within_epoch_cursor"]),
        current_order=tuple(cast("Sequence[int]", payload["current_order"])),
        effective_batch_size=cast("int", payload["effective_batch_size"]),
    )


def should_pause_for_session(
    *,
    session_started_unix: float,
    now_unix: float,
    session_limit_seconds: float = SESSION_LIMIT_SECONDS,
    save_margin_seconds: float = SESSION_SAVE_MARGIN_SECONDS,
) -> bool:
    """Leave a fixed margin for flushing tables and writing ``latest.pt``."""
    return (
        now_unix >= session_started_unix + session_limit_seconds - save_margin_seconds
    )


def copy_state_to_cpu(value: Any) -> Any:
    """Copy a Torch state tree to CPU for a portable checkpoint."""
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: copy_state_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [copy_state_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(copy_state_to_cpu(item) for item in value)
    return value


__all__ = [
    "CHECKPOINT_FORMAT",
    "RestoredDynamicsProgress",
    "build_dynamics_checkpoint",
    "capture_rng_state",
    "load_dynamics_checkpoint",
    "restore_dynamics_checkpoint",
    "restore_rng_state",
    "save_dynamics_checkpoint",
    "should_pause_for_session",
]
