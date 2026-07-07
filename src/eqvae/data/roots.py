# Copyright 2026 HiperMaximus
"""Deterministic data-root resolution for UBC pre-shuffled patch shards."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

type PatchSplit = Literal["train", "validation"]
type DataRootDiagnosticValue = (
    str
    | int
    | bool
    | Sequence[DataRootDiagnosticValue]
    | Mapping[str, DataRootDiagnosticValue]
    | None
)
type DataRootDiagnostics = dict[str, DataRootDiagnosticValue]

DATA_ROOT_ENV_VAR = "EQVAE_DATA_ROOT"
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_DATASET_OWNER = "maximusshtefan"
KAGGLE_DATASET_NAME = "patches-pre-shuffled-ubc-ocean"
TRAIN_BIN_NAME = "ubc_train_shuffled.bin"
TRAIN_CSV_NAME = "ubc_train_shuffled.csv"
VALIDATION_BIN_NAME = "ubc_ocean_valid.bin"
VALIDATION_CSV_NAME = "ubc_ocean_valid.csv"
# The real UBC pre-shuffled training split holds exactly this many patches. It is the
# single canonical source for P in the goal-derived schedule floor(P / global_batch)
# (Spec 0011): the benchmark generators and the remote gate's floor(P / G) anchor all
# import this one constant instead of re-declaring the number.
REAL_TRAIN_PATCH_COUNT = 300_000
REQUIRED_PATCH_FILENAMES = (
    TRAIN_BIN_NAME,
    TRAIN_CSV_NAME,
    VALIDATION_BIN_NAME,
    VALIDATION_CSV_NAME,
)
KAGGLE_INPUT_SCAN_MAX_DEPTH = 7
KAGGLE_INPUT_SCAN_MAX_ENTRIES = 256

KNOWN_AUTO_DATA_ROOTS = (
    KAGGLE_INPUT_ROOT / "datasets" / KAGGLE_DATASET_OWNER / KAGGLE_DATASET_NAME,
    KAGGLE_INPUT_ROOT / KAGGLE_DATASET_NAME,
    KAGGLE_INPUT_ROOT / KAGGLE_DATASET_OWNER / KAGGLE_DATASET_NAME,
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


def data_root_resolution_diagnostics(data_root: str | Path) -> DataRootDiagnostics:
    """Return JSON-safe diagnostics for patch data-root resolution.

    Returns:
        Diagnostic paths and missing-file information. The payload intentionally
        inspects only paths and required shard filenames, not secrets.

    """
    requested = str(data_root)
    candidates = _auto_root_candidates() if requested == "auto" else (Path(requested),)
    complete_kaggle_candidates = _complete_kaggle_input_candidates()
    complete_unaccepted_candidates = tuple(
        candidate
        for candidate in complete_kaggle_candidates
        if not _is_expected_kaggle_candidate(candidate)
    )
    return {
        "requested_data_root": requested,
        "env_var": DATA_ROOT_ENV_VAR,
        "env_value_present": _env_value_present(),
        "kaggle_input_root": str(KAGGLE_INPUT_ROOT),
        "kaggle_input_exists": KAGGLE_INPUT_ROOT.exists(),
        "kaggle_input_scan_truncated": _kaggle_input_scan_truncated(),
        "required_filenames": list(REQUIRED_PATCH_FILENAMES),
        "candidate_count": len(candidates),
        "accepted_candidate_count": len(candidates),
        "candidates": [_candidate_diagnostics(candidate) for candidate in candidates],
        "accepted_candidates": [
            _candidate_diagnostics(candidate) for candidate in candidates
        ],
        "complete_unaccepted_candidate_count": len(complete_unaccepted_candidates),
        "complete_unaccepted_candidates": [
            _candidate_diagnostics(candidate)
            for candidate in complete_unaccepted_candidates
        ],
        "kaggle_input_snapshot": _kaggle_input_snapshot(),
    }


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
    return _dedupe_paths(
        (
            *env_candidates,
            *KNOWN_AUTO_DATA_ROOTS,
            *_accepted_kaggle_input_candidates(),
            *local_candidates,
        ),
    )


def _dedupe_paths(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    seen: set[str] = set()
    deduped: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return tuple(deduped)


def _accepted_kaggle_input_candidates() -> tuple[Path, ...]:
    return tuple(
        candidate
        for candidate in _complete_kaggle_input_candidates()
        if _is_expected_kaggle_candidate(candidate)
    )


def _complete_kaggle_input_candidates() -> tuple[Path, ...]:
    if not KAGGLE_INPUT_ROOT.exists():
        return ()
    candidates: list[Path] = []
    for directory in _walk_dirs_bounded(
        root=KAGGLE_INPUT_ROOT,
        max_depth=KAGGLE_INPUT_SCAN_MAX_DEPTH,
        max_entries=KAGGLE_INPUT_SCAN_MAX_ENTRIES,
    ):
        paths = _paths_from_root(directory)
        if _paths_exist(paths):
            candidates.append(directory)
    return tuple(candidates)


def _is_expected_kaggle_candidate(candidate: Path) -> bool:
    try:
        relative_parts = candidate.relative_to(KAGGLE_INPUT_ROOT).parts
    except ValueError:
        return False
    mount_parts = (
        relative_parts[:-1]
        if relative_parts and relative_parts[-1] == "dataset"
        else relative_parts
    )
    return _relative_parts_match_expected_kaggle_mount(mount_parts)


def _relative_parts_match_expected_kaggle_mount(parts: tuple[str, ...]) -> bool:
    direct_mount = (KAGGLE_DATASET_NAME,)
    owner_mount = (KAGGLE_DATASET_OWNER, KAGGLE_DATASET_NAME)
    dataset_mount = ("datasets", KAGGLE_DATASET_OWNER, KAGGLE_DATASET_NAME)
    versioned_prefix = (*dataset_mount, "versions")
    return parts in {direct_mount, owner_mount, dataset_mount} or (
        len(parts) == len(versioned_prefix) + 1
        and parts[: len(versioned_prefix)] == versioned_prefix
        and bool(parts[-1])
    )


def _walk_dirs_bounded(
    *,
    root: Path,
    max_depth: int,
    max_entries: int,
) -> tuple[Path, ...]:
    directories, _ = _walk_dirs_bounded_with_truncation(
        root=root,
        max_depth=max_depth,
        max_entries=max_entries,
    )
    return directories


def _walk_dirs_bounded_with_truncation(
    *,
    root: Path,
    max_depth: int,
    max_entries: int,
) -> tuple[tuple[Path, ...], bool]:
    directories: list[Path] = []
    queue: list[tuple[Path, int]] = [(root, 0)]
    visited = 0
    while queue and visited < max_entries:
        current, depth = queue.pop(0)
        visited += 1
        if current != root:
            directories.append(current)
        if depth >= max_depth:
            continue
        try:
            children = sorted(
                (child for child in current.iterdir() if child.is_dir()),
                key=lambda child: child.name,
            )
        except OSError:
            continue
        queue.extend((child, depth + 1) for child in children)
    return tuple(directories), bool(queue)


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


def _candidate_diagnostics(candidate: Path) -> DataRootDiagnostics:
    paths = _paths_from_root(candidate)
    return {
        "candidate_root": str(candidate),
        "candidate_is_expected_kaggle_mount": _is_expected_kaggle_candidate(candidate),
        "candidate_exists": candidate.exists(),
        "resolved_dataset_root": str(paths.root),
        "resolved_dataset_root_exists": paths.root.exists(),
        "missing_paths": list(_missing_paths(paths)),
        "complete": _paths_exist(paths),
    }


def _kaggle_input_snapshot() -> list[DataRootDiagnostics]:
    if not KAGGLE_INPUT_ROOT.exists():
        return []
    snapshot: list[DataRootDiagnostics] = []
    queue: list[tuple[Path, int]] = [(KAGGLE_INPUT_ROOT, 0)]
    visited = 0
    while queue and visited < KAGGLE_INPUT_SCAN_MAX_ENTRIES:
        current, depth = queue.pop(0)
        visited += 1
        if current != KAGGLE_INPUT_ROOT:
            snapshot.append(_path_snapshot_item(current))
        if depth >= KAGGLE_INPUT_SCAN_MAX_DEPTH:
            continue
        try:
            children = sorted(current.iterdir(), key=lambda child: child.name)
        except OSError:
            continue
        queue.extend(
            (child, depth + 1)
            for child in children
            if child.is_dir() or child.name in REQUIRED_PATCH_FILENAMES
        )
    return snapshot


def _kaggle_input_scan_truncated() -> bool:
    if not KAGGLE_INPUT_ROOT.exists():
        return False
    _, truncated = _walk_dirs_bounded_with_truncation(
        root=KAGGLE_INPUT_ROOT,
        max_depth=KAGGLE_INPUT_SCAN_MAX_DEPTH,
        max_entries=KAGGLE_INPUT_SCAN_MAX_ENTRIES,
    )
    return truncated


def _path_snapshot_item(path: Path) -> DataRootDiagnostics:
    item: DataRootDiagnostics = {
        "path": str(path),
        "relative_path": _relative_to_kaggle_input(path),
        "kind": "dir" if path.is_dir() else "file",
    }
    if path.is_file() and path.name in REQUIRED_PATCH_FILENAMES:
        with_size = _file_size_or_none(path)
        item["size_bytes"] = with_size
    return item


def _relative_to_kaggle_input(path: Path) -> str:
    try:
        return str(path.relative_to(KAGGLE_INPUT_ROOT))
    except ValueError:
        return str(path)


def _file_size_or_none(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


def _env_value_present() -> bool:
    value = os.environ.get(DATA_ROOT_ENV_VAR)
    return value is not None and bool(value.strip())


__all__ = [
    "DATA_ROOT_ENV_VAR",
    "KAGGLE_DATASET_NAME",
    "KAGGLE_DATASET_OWNER",
    "KAGGLE_INPUT_ROOT",
    "KNOWN_AUTO_DATA_ROOTS",
    "REAL_TRAIN_PATCH_COUNT",
    "DataRootDiagnosticValue",
    "DataRootDiagnostics",
    "PatchDataPaths",
    "PatchSplit",
    "PatchSplitPaths",
    "data_root_resolution_diagnostics",
    "normalize_patch_split",
    "resolve_patch_data_paths",
]
