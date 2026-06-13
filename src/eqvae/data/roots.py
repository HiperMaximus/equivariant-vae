# Copyright 2026 HiperMaximus
"""Deterministic data-root resolution for UBC pre-shuffled patch shards."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

type PatchSplit = Literal["train", "validation"]

DATA_ROOT_ENV_VAR = "EQVAE_DATA_ROOT"
TRAIN_BIN_NAME = "ubc_train_shuffled.bin"
TRAIN_CSV_NAME = "ubc_train_shuffled.csv"
VALIDATION_BIN_NAME = "ubc_ocean_valid.bin"
VALIDATION_CSV_NAME = "ubc_ocean_valid.csv"

KNOWN_AUTO_DATA_ROOTS = (
    Path("/kaggle/input/datasets/maximusshtefan/patches-pre-shuffled-ubc-ocean"),
    Path("/kaggle/input/patches-pre-shuffled-ubc-ocean"),
)


@dataclass(frozen=True)
class PatchSplitPaths:
    """Resolved binary and CSV paths for one split."""

    split: PatchSplit
    bin_path: Path
    csv_path: Path


@dataclass(frozen=True)
class PatchDataPaths:
    """Resolved binary and CSV paths for all train/validation shards."""

    root: Path
    train: PatchSplitPaths
    validation: PatchSplitPaths

    def for_split(self, split: PatchSplit) -> PatchSplitPaths:
        """Return paths for one canonical split.

        Returns:
            Split-specific paths.

        """
        if split == "train":
            return self.train
        return self.validation


def normalize_patch_split(value: str) -> PatchSplit:
    """Normalize historical split aliases to canonical spec 0001 names.

    Returns:
        Canonical split name.

    Raises:
        ValueError: If the split is unknown.

    """
    normalized = value.strip().lower()
    if normalized == "train":
        return "train"
    if normalized in {"validation", "valid", "val"}:
        return "validation"
    message = f"Unknown patch split {value!r}; expected train or validation"
    raise ValueError(message)


def resolve_patch_data_paths(data_root: str | Path) -> PatchDataPaths:
    """Resolve UBC pre-shuffled train/validation shard paths.

    Args:
        data_root: Explicit root path, or `"auto"` for deterministic local/Kaggle
            lookup.

    Returns:
        Resolved train and validation paths.

    Raises:
        FileNotFoundError: If no complete shard set is found.

    """
    root_value = str(data_root)
    if root_value == "auto":
        for candidate in _auto_root_candidates():
            paths = _paths_from_root(candidate)
            if _paths_exist(paths):
                return paths
        searched = ", ".join(str(path) for path in _auto_root_candidates())
        message = (
            "Could not resolve data_root='auto'. Set "
            f"{DATA_ROOT_ENV_VAR} or pass --data-root explicitly. Searched: {searched}"
        )
        raise FileNotFoundError(message)

    explicit_paths = _paths_from_root(Path(root_value))
    if not _paths_exist(explicit_paths):
        missing = _missing_paths(explicit_paths)
        message = f"Missing UBC patch shard files under {root_value}: {missing}"
        raise FileNotFoundError(message)
    return explicit_paths


def _auto_root_candidates() -> tuple[Path, ...]:
    env_value = os.environ.get(DATA_ROOT_ENV_VAR)
    env_candidates = (
        () if env_value is None or not env_value.strip() else (Path(env_value),)
    )
    repo_root = Path(__file__).resolve().parents[3]
    local_candidates = (
        repo_root / "data" / "patches-pre-shuffled-ubc-ocean",
        repo_root / "dataset",
    )
    return (*env_candidates, *KNOWN_AUTO_DATA_ROOTS, *local_candidates)


def _paths_from_root(root: Path) -> PatchDataPaths:
    dataset_root = root if _contains_split_files(root) else root / "dataset"
    return PatchDataPaths(
        root=dataset_root,
        train=PatchSplitPaths(
            split="train",
            bin_path=dataset_root / TRAIN_BIN_NAME,
            csv_path=dataset_root / TRAIN_CSV_NAME,
        ),
        validation=PatchSplitPaths(
            split="validation",
            bin_path=dataset_root / VALIDATION_BIN_NAME,
            csv_path=dataset_root / VALIDATION_CSV_NAME,
        ),
    )


def _contains_split_files(root: Path) -> bool:
    return (root / TRAIN_BIN_NAME).exists() or (root / VALIDATION_BIN_NAME).exists()


def _paths_exist(paths: PatchDataPaths) -> bool:
    return not _missing_paths(paths)


def _missing_paths(paths: PatchDataPaths) -> tuple[str, ...]:
    candidates = (
        paths.train.bin_path,
        paths.train.csv_path,
        paths.validation.bin_path,
        paths.validation.csv_path,
    )
    return tuple(str(path) for path in candidates if not path.exists())


__all__ = [
    "DATA_ROOT_ENV_VAR",
    "KNOWN_AUTO_DATA_ROOTS",
    "PatchDataPaths",
    "PatchSplit",
    "PatchSplitPaths",
    "normalize_patch_split",
    "resolve_patch_data_paths",
]
