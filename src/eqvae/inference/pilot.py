# Copyright 2026 HiperMaximus
# ruff: noqa: D102, DOC201, DOC501, PLR2004, TC003
"""Pure tiny-smoke contract logic for Spec 0021."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

import torch
from torch import Tensor

from eqvae.inference.worker import EncoderRecipe, ExecutionMode, NumericMode

if TYPE_CHECKING:
    from collections.abc import Sequence

    from eqvae.data.latent_shards import WorkManifest

type D2HMode = Literal["synchronous", "bounded-pinned-double-buffer"]

PILOT_WSI_ID: Final = 15_188
PILOT_RUN_NUMBER: Final = 2
PILOT_ROW_COUNT: Final = 16
PILOT_WSI_ROW_COUNT: Final = 5_668
PILOT_MATRIX_SCHEMA: Final = "spec0021.pilot_matrix.v2"
PILOT_SUMMARY_SCHEMA: Final = "spec0021.pilot_authority.v2"


@dataclass(frozen=True)
class PilotRecipe:
    """The single fixed recipe used by the tiny pilot smoke check."""

    batch_size: int
    d2h_mode: D2HMode
    numeric_mode: NumericMode
    execution: ExecutionMode

    def encoder_recipe(self) -> EncoderRecipe:
        return EncoderRecipe(
            batch_size=self.batch_size,
            numeric_mode=self.numeric_mode,
            execution=self.execution,
        )


@dataclass(frozen=True)
class PilotCandidateResult:
    """Outcome fields for the single fixed smoke recipe."""

    recipe: PilotRecipe
    eligible: bool
    median_patches_per_second: float
    peak_total_gpu_bytes: int
    peak_rss_bytes: int
    failure: str | None = None


def pilot_recipes() -> tuple[PilotRecipe, ...]:
    """Return the one conservative recipe used by the smoke check."""
    return (
        PilotRecipe(
            batch_size=8,
            d2h_mode="synchronous",
            numeric_mode="FP32",
            execution="eager",
        ),
    )


def select_pilot_candidate(
    rows: Sequence[PilotCandidateResult],
) -> PilotCandidateResult:
    """Return the fixed recipe result if and only if it is eligible."""
    if tuple(row.recipe for row in rows) != pilot_recipes():
        message = "Pilot results must contain the single fixed smoke recipe"
        raise ValueError(message)
    selected = rows[0]
    if not selected.eligible:
        message = "Pilot smoke recipe is not eligible"
        raise ValueError(message)
    return selected


def pilot_wsi_range(manifest: WorkManifest) -> tuple[int, int]:
    """Return the first 16 rows of frozen train-only WSI 15188."""
    if manifest.run_number != PILOT_RUN_NUMBER:
        message = "Pilot requires work manifest run 02"
        raise ValueError(message)
    matches = [
        (start, end)
        for wsi_id, start, end in manifest.wsi_ranges
        if wsi_id == PILOT_WSI_ID
    ]
    if len(matches) != 1 or matches[0][1] - matches[0][0] != PILOT_WSI_ROW_COUNT:
        message = "Pilot WSI 15188 must contain exactly 5668 selected rows"
        raise ValueError(message)
    start, _wsi_end = matches[0]
    end = start + PILOT_ROW_COUNT
    rows = manifest.rows[start:end]
    if len(rows) != PILOT_ROW_COUNT or any(row.split != "train" for row in rows):
        message = "Pilot rows must all belong to the frozen train split"
        raise ValueError(message)
    return start, end


def pilot_sentinel_indices(manifest: WorkManifest) -> tuple[int, int, int]:
    """Use first, middle, and last indices of the 16-row smoke slice."""
    start, end = pilot_wsi_range(manifest)
    return start, start + PILOT_ROW_COUNT // 2, end - 1


def require_exact_dual_t4() -> tuple[torch.device, torch.device]:
    """Fail unless exactly two visible NVIDIA T4 devices are assigned."""
    if torch.cuda.device_count() != 2:
        message = "Spec 0021 requires exactly two visible CUDA devices"
        raise RuntimeError(message)
    names = tuple(torch.cuda.get_device_name(index) for index in range(2))
    if any("T4" not in name.upper() for name in names):
        message = f"Spec 0021 requires two NVIDIA T4 devices, got {names!r}"
        raise RuntimeError(message)
    return torch.device("cuda:0"), torch.device("cuda:1")


def write_pilot_payload(path: Path, tensors: Tensor, *, append: bool = False) -> int:
    """Write exact Spec 0020 payload bytes to a scratch-only sink."""
    if tensors.dtype != torch.float32 or tensors.device.type != "cpu":
        message = "Pilot serialization requires CPU FP32 tensors"
        raise TypeError(message)
    if tensors.ndim != 4 or tuple(tensors.shape[1:]) != (16, 32, 32):
        message = "Pilot tensors must have shape (B,16,32,32)"
        raise ValueError(message)
    if not torch.isfinite(tensors).all():
        message = "Pilot tensors contain nonfinite values"
        raise ValueError(message)
    payload = tensors.contiguous().numpy().astype("<f4", copy=False).tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab" if append else "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return len(payload)


def remove_pilot_payload(path: Path) -> None:
    """Remove the ephemeral sink and fsync its directory before publication."""
    path.unlink(missing_ok=True)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


__all__ = [
    "PILOT_MATRIX_SCHEMA",
    "PILOT_ROW_COUNT",
    "PILOT_RUN_NUMBER",
    "PILOT_SUMMARY_SCHEMA",
    "PILOT_WSI_ID",
    "PilotCandidateResult",
    "PilotRecipe",
    "pilot_recipes",
    "pilot_sentinel_indices",
    "pilot_wsi_range",
    "remove_pilot_payload",
    "require_exact_dual_t4",
    "select_pilot_candidate",
    "write_pilot_payload",
]
