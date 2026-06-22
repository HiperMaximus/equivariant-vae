# Copyright 2026 HiperMaximus
"""Checkpoint helpers for the spec 0001 debug training loop."""

from __future__ import annotations

import hashlib
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeGuard, cast

import torch

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from numpy.random import Generator
    from torch import nn


_SCHEMA_VERSION = "spec0001.checkpoint.v4"
_PYTHON_RANDOM_STATE_LEN = 3
type _PythonRandomState = tuple[int, tuple[int, ...], float | None]


@dataclass(frozen=True)
class CheckpointMetadata:
    """Metadata returned after writing a checkpoint."""

    path: Path
    sha256: str
    optimizer_step: int
    successful_optimizer_update_count: int


@dataclass(frozen=True)
class CheckpointResumeMetadata:
    """Resume metadata read without mutating runtime state."""

    path: Path
    schema_version: str
    run_name: str
    config_path: str
    config_sha256: str
    effective_config_sha256: str
    runtime_config_sha256: str
    selected_row_id: str
    runtime_policy_id: str
    optimizer_step: int
    successful_optimizer_update_count: int
    metric_name: str
    metric_value: float


@dataclass(frozen=True)
class LoadedCheckpoint(CheckpointResumeMetadata):
    """Metadata restored from a checkpoint."""

    torch_generator_names: tuple[str, ...]


def save_training_checkpoint(  # noqa: PLR0913
    *,
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    numpy_generator: Generator,
    run_name: str,
    config_path: Path,
    config_sha256: str,
    effective_config_sha256: str,
    optimizer_step: int,
    successful_optimizer_update_count: int,
    metric_name: str,
    metric_value: float,
    runtime_config_sha256: str = "",
    selected_row_id: str = "",
    runtime_policy_id: str = "",
    torch_generators: Mapping[str, torch.Generator] | None = None,
) -> CheckpointMetadata:
    """Save model, optimizer, and local RNG state for debug/resume proof.

    Returns:
        Checkpoint metadata with path, hash, and step counters.

    Raises:
        ValueError: If step counters are invalid.

    """
    if optimizer_step < 0:
        message = f"optimizer_step must be nonnegative, got {optimizer_step}"
        raise ValueError(message)
    if successful_optimizer_update_count < 0:
        message = (
            "successful_optimizer_update_count must be nonnegative, got "
            f"{successful_optimizer_update_count}"
        )
        raise ValueError(message)
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "run_name": run_name,
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "effective_config_sha256": effective_config_sha256,
        "runtime_config_sha256": runtime_config_sha256,
        "selected_row_id": selected_row_id,
        "runtime_policy_id": runtime_policy_id,
        "optimizer_step": optimizer_step,
        "successful_optimizer_update_count": successful_optimizer_update_count,
        "metric_name": metric_name,
        "metric_value": float(metric_value),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "python_rng_state": random.getstate(),
        "numpy_generator_state": _numpy_generator_state_payload(numpy_generator),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_generator_states": _torch_generator_states_payload(torch_generators),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return CheckpointMetadata(
        path=path,
        sha256=_sha256_file(path),
        optimizer_step=optimizer_step,
        successful_optimizer_update_count=successful_optimizer_update_count,
    )


def load_training_checkpoint(  # noqa: PLR0913
    *,
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    numpy_generator: Generator,
    torch_generators: Mapping[str, torch.Generator] | None = None,
    expected_effective_config_sha256: str | None = None,
    expected_runtime_config_sha256: str | None = None,
    expected_selected_row_id: str | None = None,
    expected_runtime_policy_id: str | None = None,
) -> LoadedCheckpoint:
    """Restore model, optimizer, and local RNG state from a debug checkpoint.

    Returns:
        Loaded checkpoint metadata after state restoration.

    Raises:
        TypeError: If the checkpoint payload has an invalid shape.

    """
    payload_object = cast(
        "object",
        torch.load(path, map_location="cpu", weights_only=False),
    )
    if not isinstance(payload_object, dict):
        message = f"Checkpoint payload must be an object: {path}"
        raise TypeError(message)
    payload = cast("dict[str, object]", payload_object)
    metadata = _metadata_from_payload(path=path, payload=payload)
    validate_checkpoint_resume_metadata(
        metadata,
        expected_effective_config_sha256=expected_effective_config_sha256,
        expected_runtime_config_sha256=expected_runtime_config_sha256,
        expected_selected_row_id=expected_selected_row_id,
        expected_runtime_policy_id=expected_runtime_policy_id,
    )
    model_state = payload.get("model_state_dict")
    optimizer_state = payload.get("optimizer_state_dict")
    python_rng_state = payload.get("python_rng_state")
    numpy_generator_state = payload.get("numpy_generator_state")
    rng_state = payload.get("torch_cpu_rng_state")
    torch_generator_states = payload.get("torch_generator_states")
    if not isinstance(model_state, dict):
        message = "checkpoint model_state_dict must be an object"
        raise TypeError(message)
    if not isinstance(optimizer_state, dict):
        message = "checkpoint optimizer_state_dict must be an object"
        raise TypeError(message)
    if not isinstance(python_rng_state, tuple):
        message = "checkpoint python_rng_state must be a tuple"
        raise TypeError(message)
    python_rng_state_tuple = cast("tuple[object, ...]", python_rng_state)
    if not _is_python_random_state(python_rng_state_tuple):
        message = "checkpoint python_rng_state has invalid shape"
        raise TypeError(message)
    if not isinstance(numpy_generator_state, dict):
        message = "checkpoint numpy_generator_state must be an object"
        raise TypeError(message)
    if not isinstance(rng_state, torch.Tensor):
        message = "checkpoint torch_cpu_rng_state must be a tensor"
        raise TypeError(message)
    if not isinstance(torch_generator_states, dict):
        message = "checkpoint torch_generator_states must be an object"
        raise TypeError(message)

    model.load_state_dict(cast("dict[str, object]", model_state))
    optimizer.load_state_dict(cast("dict[str, object]", optimizer_state))
    random.setstate(python_rng_state_tuple)
    _restore_numpy_generator_state(
        numpy_generator,
        cast("dict[str, object]", numpy_generator_state),
    )
    torch.set_rng_state(rng_state.to(dtype=torch.uint8, device="cpu"))
    torch_generator_names = _restore_torch_generator_states(
        torch_generators=torch_generators,
        payload=cast("dict[object, object]", torch_generator_states),
    )
    return LoadedCheckpoint(
        path=metadata.path,
        schema_version=metadata.schema_version,
        run_name=metadata.run_name,
        config_path=metadata.config_path,
        config_sha256=metadata.config_sha256,
        effective_config_sha256=metadata.effective_config_sha256,
        runtime_config_sha256=metadata.runtime_config_sha256,
        selected_row_id=metadata.selected_row_id,
        runtime_policy_id=metadata.runtime_policy_id,
        optimizer_step=metadata.optimizer_step,
        successful_optimizer_update_count=(metadata.successful_optimizer_update_count),
        metric_name=metadata.metric_name,
        metric_value=metadata.metric_value,
        torch_generator_names=torch_generator_names,
    )


def read_training_checkpoint_metadata(*, path: Path) -> CheckpointResumeMetadata:
    """Read checkpoint metadata without restoring model, optimizer, or RNG state.

    Returns:
        Checkpoint resume metadata.

    Raises:
        TypeError: If the checkpoint payload has an invalid shape.

    """
    payload_object = cast(
        "object",
        torch.load(path, map_location="cpu", weights_only=False),
    )
    if not isinstance(payload_object, dict):
        message = f"Checkpoint payload must be an object: {path}"
        raise TypeError(message)
    return _metadata_from_payload(
        path=path,
        payload=cast("dict[str, object]", payload_object),
    )


def validate_checkpoint_resume_metadata(
    metadata: CheckpointResumeMetadata,
    *,
    expected_effective_config_sha256: str | None = None,
    expected_runtime_config_sha256: str | None = None,
    expected_selected_row_id: str | None = None,
    expected_runtime_policy_id: str | None = None,
) -> None:
    """Fail before state restore when checkpoint identity does not match."""
    _validate_expected_str(
        actual=metadata.effective_config_sha256,
        expected=expected_effective_config_sha256,
        field_name="effective_config_sha256",
    )
    _validate_expected_str(
        actual=metadata.runtime_config_sha256,
        expected=expected_runtime_config_sha256,
        field_name="runtime_config_sha256",
    )
    _validate_expected_str(
        actual=metadata.selected_row_id,
        expected=expected_selected_row_id,
        field_name="selected_row_id",
    )
    _validate_expected_str(
        actual=metadata.runtime_policy_id,
        expected=expected_runtime_policy_id,
        field_name="runtime_policy_id",
    )


def _metadata_from_payload(
    *,
    path: Path,
    payload: dict[str, object],
) -> CheckpointResumeMetadata:
    schema_version = _required_str(payload, "schema_version")
    if schema_version != _SCHEMA_VERSION:
        message = f"Unsupported checkpoint schema {schema_version!r}"
        raise ValueError(message)
    return CheckpointResumeMetadata(
        path=path,
        schema_version=schema_version,
        run_name=_required_str(payload, "run_name"),
        config_path=_required_str(payload, "config_path"),
        config_sha256=_required_str(payload, "config_sha256"),
        effective_config_sha256=_required_str(payload, "effective_config_sha256"),
        runtime_config_sha256=_required_str(payload, "runtime_config_sha256"),
        selected_row_id=_required_str(payload, "selected_row_id"),
        runtime_policy_id=_required_str(payload, "runtime_policy_id"),
        optimizer_step=_required_int(payload, "optimizer_step"),
        successful_optimizer_update_count=_required_int(
            payload,
            "successful_optimizer_update_count",
        ),
        metric_name=_required_str(payload, "metric_name"),
        metric_value=_required_float(payload, "metric_value"),
    )


def _validate_expected_str(
    *,
    actual: str,
    expected: str | None,
    field_name: str,
) -> None:
    if expected is None or actual == expected:
        return
    message = f"resume checkpoint {field_name} differs from current run"
    raise ValueError(message)


def _required_str(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if isinstance(value, str):
        return value
    message = f"Expected string checkpoint field {key}"
    raise TypeError(message)


def _required_int(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool):
        message = f"Expected integer checkpoint field {key}"
        raise TypeError(message)
    if isinstance(value, int):
        return value
    message = f"Expected integer checkpoint field {key}"
    raise TypeError(message)


def _required_float(payload: dict[str, object], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool):
        message = f"Expected numeric checkpoint field {key}"
        raise TypeError(message)
    if isinstance(value, int | float):
        return float(value)
    message = f"Expected numeric checkpoint field {key}"
    raise TypeError(message)


def _numpy_generator_state_payload(numpy_generator: Generator) -> dict[str, object]:
    state = cast("object", deepcopy(numpy_generator.bit_generator.state))
    if not isinstance(state, dict):
        message = "NumPy generator state must be an object"
        raise TypeError(message)
    return cast("dict[str, object]", state)


def _restore_numpy_generator_state(
    numpy_generator: Generator,
    payload: dict[str, object],
) -> None:
    numpy_generator.bit_generator.state = payload


def _torch_generator_states_payload(
    torch_generators: Mapping[str, torch.Generator] | None,
) -> dict[str, torch.Tensor]:
    if torch_generators is None:
        return {}
    states: dict[str, torch.Tensor] = {}
    for name, generator in sorted(torch_generators.items()):
        if not name:
            message = "Torch generator checkpoint names must be nonempty"
            raise ValueError(message)
        states[name] = generator.get_state().detach().cpu().clone()
    return states


def _restore_torch_generator_states(
    *,
    torch_generators: Mapping[str, torch.Generator] | None,
    payload: dict[object, object],
) -> tuple[str, ...]:
    generators = {} if torch_generators is None else dict(torch_generators)
    payload_names = {key for key in payload if isinstance(key, str)}
    if len(payload_names) != len(payload):
        message = "checkpoint torch_generator_states keys must be strings"
        raise TypeError(message)
    expected_names = set(generators)
    if payload_names != expected_names:
        message = (
            "checkpoint torch_generator_states keys do not match requested "
            f"generators: {sorted(payload_names)} != {sorted(expected_names)}"
        )
        raise ValueError(message)
    for name, generator in sorted(generators.items()):
        state = payload[name]
        if not isinstance(state, torch.Tensor):
            message = f"checkpoint torch_generator_states.{name} must be a tensor"
            raise TypeError(message)
        generator.set_state(state.to(dtype=torch.uint8, device="cpu"))
    return tuple(sorted(generators))


def _is_python_random_state(value: object) -> TypeGuard[_PythonRandomState]:
    if not isinstance(value, tuple):
        return False
    items = cast("tuple[object, ...]", value)
    if len(items) != _PYTHON_RANDOM_STATE_LEN:
        return False
    version, keys, gaussian = items
    if not isinstance(keys, tuple):
        return False
    key_items = cast("tuple[object, ...]", keys)
    return (
        isinstance(version, int)
        and all(isinstance(item, int) for item in key_items)
        and (gaussian is None or isinstance(gaussian, float))
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "CheckpointMetadata",
    "CheckpointResumeMetadata",
    "LoadedCheckpoint",
    "load_training_checkpoint",
    "read_training_checkpoint_metadata",
    "save_training_checkpoint",
    "validate_checkpoint_resume_metadata",
]
