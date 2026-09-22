# Copyright 2026 HiperMaximus
"""Small atomic NPZ tables for the one-off MIL dynamics probe."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, TypeAlias, cast

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

TELEMETRY_FORMAT: Final = "spec0054.telemetry_table.v1"
TelemetryScalar: TypeAlias = int | float | bool | str


def write_telemetry_table(
    path: Path,
    rows: Sequence[Mapping[str, TelemetryScalar]],
) -> None:
    """Atomically replace one small cumulative telemetry table.

    The probe keeps one table per stream (for example ``t0.npz``). Rewriting a
    compressed table at a checkpoint is simpler than managing a shard registry.
    If a crash leaves telemetry ahead of ``latest.pt``,
    :func:`load_telemetry_table` can truncate it to the restored update.
    """
    if not rows:
        raise ValueError("Telemetry table requires at least one row")
    columns = tuple(sorted(rows[0]))
    if "update" not in columns or any(tuple(sorted(row)) != columns for row in rows):
        raise ValueError("Telemetry rows need one fixed schema including update")
    arrays = {
        column: _column_array([row[column] for row in rows]) for column in columns
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}-",
        suffix=".npz",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        save = cast("Any", np.savez_compressed)
        save(
            temporary,
            format=np.asarray(TELEMETRY_FORMAT),
            columns=np.asarray(columns),
            **arrays,
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_telemetry_table(
    path: Path,
    *,
    through_update: int | None = None,
) -> dict[str, np.ndarray]:
    """Load a table, optionally discarding observations past a restored update."""
    if through_update is not None and through_update < 0:
        raise ValueError("Telemetry update boundary must be nonnegative")
    with np.load(path, allow_pickle=False) as archive:
        if str(archive["format"].item()) != TELEMETRY_FORMAT:
            raise ValueError("Telemetry table format differs")
        columns = tuple(str(value) for value in archive["columns"].tolist())
        if "update" not in columns or set(archive.files) != {
            "format",
            "columns",
            *columns,
        }:
            raise ValueError("Telemetry table columns differ")
        arrays = {column: archive[column].copy() for column in columns}
    if any(array.ndim != 1 or array.dtype.hasobject for array in arrays.values()):
        raise ValueError("Telemetry columns must be safe one-dimensional arrays")
    lengths = {len(array) for array in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("Telemetry column lengths differ")
    if through_update is not None:
        mask = arrays["update"] <= through_update
        arrays = {column: values[mask] for column, values in arrays.items()}
    return arrays


def _column_array(values: Sequence[TelemetryScalar]) -> np.ndarray:
    if all(isinstance(value, bool) for value in values):
        return np.asarray(values, dtype=np.bool_)
    if all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        return np.asarray(values, dtype=np.int64)
    if all(isinstance(value, str) for value in values):
        return np.asarray(values, dtype=np.str_)
    if any(isinstance(value, str) for value in values):
        raise TypeError("Telemetry columns may not mix text and numbers")
    if not all(isinstance(value, (int, float, bool)) for value in values):
        raise TypeError("Telemetry values must be scalar")
    return np.asarray(values, dtype=np.float64)


__all__ = [
    "TELEMETRY_FORMAT",
    "TelemetryScalar",
    "load_telemetry_table",
    "write_telemetry_table",
]
