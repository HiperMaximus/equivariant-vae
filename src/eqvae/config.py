# Copyright 2026 HiperMaximus
"""Configuration loading helpers for spec 0001 JSON files."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from pathlib import Path

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]


@dataclass(frozen=True)
class ConfigHash:
    """Raw-byte hash provenance for one config file."""

    path: Path
    sha256: str

    def as_json(self) -> JsonObject:
        """Return a JSON-ready provenance object.

        Returns:
            JSON object with path and SHA-256 fields.

        """
        return {
            "path": str(self.path),
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class ResolvedConfig:
    """A config after recursively applying any `source_config` overlays."""

    invoked_path: Path
    effective_config: JsonObject
    invoked_config_hash: str
    effective_config_hash: str
    source_config_chain: tuple[ConfigHash, ...]


def resolve_json_config(path: Path) -> ResolvedConfig:
    """Load a JSON config and recursively merge `source_config` overlays.

    Later configs override earlier source configs. The `source_config` key is
    provenance, so it is not copied into the effective config payload.

    Returns:
        Resolved effective config plus invoked/source hash provenance.

    """
    return _resolve_json_config(path=path, stack=())


def canonical_json_hash(payload: JsonObject) -> str:
    """Return the canonical SHA-256 used by benchmark artifacts.

    Returns:
        Hex SHA-256 of sorted compact JSON.

    """
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _resolve_json_config(path: Path, stack: tuple[Path, ...]) -> ResolvedConfig:
    path_identity = _path_identity(path)
    if path_identity in stack:
        cycle = " -> ".join(str(item) for item in (*stack, path_identity))
        message = f"Config source_config cycle detected: {cycle}"
        raise ValueError(message)

    raw_payload = _read_json_object(path)
    invoked_hash = _sha256_file(path)
    source_value = raw_payload.get("source_config")
    if source_value is None:
        return ResolvedConfig(
            invoked_path=path,
            effective_config=raw_payload,
            invoked_config_hash=invoked_hash,
            effective_config_hash=canonical_json_hash(raw_payload),
            source_config_chain=(),
        )
    if not isinstance(source_value, str):
        message = f"Expected string `source_config` in {path}"
        raise TypeError(message)

    source_path = _resolve_source_path(source_value, current_path=path)
    source_config = _resolve_json_config(
        path=source_path,
        stack=(*stack, path_identity),
    )
    overlay = {
        key: value for key, value in raw_payload.items() if key != "source_config"
    }
    effective = _deep_merge(source_config.effective_config, overlay)
    return ResolvedConfig(
        invoked_path=path,
        effective_config=effective,
        invoked_config_hash=invoked_hash,
        effective_config_hash=canonical_json_hash(effective),
        source_config_chain=(
            *source_config.source_config_chain,
            ConfigHash(
                path=source_config.invoked_path,
                sha256=source_config.invoked_config_hash,
            ),
        ),
    )


def _read_json_object(path: Path) -> JsonObject:
    payload = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(payload, dict):
        message = f"Expected JSON object in {path}"
        raise TypeError(message)
    return cast("JsonObject", payload)


def _resolve_source_path(source: str, *, current_path: Path) -> Path:
    source_path = type(current_path)(source)
    if source_path.is_absolute():
        return source_path
    for parent in (current_path.parent, *current_path.parent.parents):
        candidate = parent / source_path
        if candidate.exists():
            return candidate
    return current_path.parent / source_path


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _path_identity(path: Path) -> Path:
    if path.exists():
        return path.resolve()
    return path.absolute()


def _deep_merge(base: JsonObject, overlay: JsonObject) -> JsonObject:
    merged: JsonObject = dict(base)
    for key, value in overlay.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(
                cast("JsonObject", base_value),
                cast("JsonObject", value),
            )
        else:
            merged[key] = value
    return merged


__all__ = [
    "ConfigHash",
    "ResolvedConfig",
    "canonical_json_hash",
    "resolve_json_config",
]
