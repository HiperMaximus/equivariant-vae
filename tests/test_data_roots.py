# Copyright 2026 HiperMaximus
"""Tests for deterministic UBC patch data-root resolution."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from eqvae.data import roots
from eqvae.data.roots import (
    DATA_ROOT_ENV_VAR,
    TRAIN_BIN_NAME,
    TRAIN_CSV_NAME,
    VALIDATION_BIN_NAME,
    VALIDATION_CSV_NAME,
    resolve_patch_data_paths,
)
from eqvae.data.synthetic import SyntheticPatchSpec, write_synthetic_patch_shard

if TYPE_CHECKING:
    from pathlib import Path

PATCH_COUNT = 6
PATCH_SIZE = 8


def test_explicit_data_root_accepts_dataset_subdirectory(tmp_path: Path) -> None:
    """Explicit roots may point at the Kaggle-style parent directory."""
    root = tmp_path / "patches-pre-shuffled-ubc-ocean"
    _write_complete_root(root)

    paths = resolve_patch_data_paths(root)

    assert paths.root == root / "dataset"
    assert paths.train.bin_path.name == TRAIN_BIN_NAME
    assert paths.validation.csv_path.name == VALIDATION_CSV_NAME


def test_auto_data_root_uses_env_without_cwd_dependence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`auto` checks the environment path before repo-local candidates."""
    root = tmp_path / "env-data-root"
    other_cwd = tmp_path / "other-cwd"
    other_cwd.mkdir()
    _write_complete_root(root)
    monkeypatch.setenv(DATA_ROOT_ENV_VAR, str(root))
    monkeypatch.chdir(other_cwd)

    paths = resolve_patch_data_paths("auto")

    assert paths.root == root / "dataset"
    assert paths.train.csv_path.exists()
    assert paths.validation.bin_path.exists()


def test_auto_data_root_ignores_blank_env_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A blank env var must not resolve to the current working directory."""
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    _write_complete_root(cwd)
    monkeypatch.setenv(DATA_ROOT_ENV_VAR, " ")
    monkeypatch.chdir(cwd)

    with pytest.raises(FileNotFoundError, match=DATA_ROOT_ENV_VAR):
        resolve_patch_data_paths("auto")


@pytest.mark.parametrize(
    "relative_parts",
    [
        (roots.KAGGLE_DATASET_NAME,),
        (roots.KAGGLE_DATASET_OWNER, roots.KAGGLE_DATASET_NAME),
        ("datasets", roots.KAGGLE_DATASET_OWNER, roots.KAGGLE_DATASET_NAME),
        (
            "datasets",
            roots.KAGGLE_DATASET_OWNER,
            roots.KAGGLE_DATASET_NAME,
            "versions",
            "1",
        ),
    ],
)
def test_auto_data_root_scans_expected_kaggle_mount_variants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_parts: tuple[str, ...],
) -> None:
    """Kaggle source mounts may vary, but only expected slug paths resolve."""
    kaggle_input = tmp_path / "kaggle" / "input"
    mounted_root = kaggle_input.joinpath(*relative_parts)
    _write_complete_root(mounted_root)
    monkeypatch.setattr(roots, "KAGGLE_INPUT_ROOT", kaggle_input)
    monkeypatch.setattr(roots, "KNOWN_AUTO_DATA_ROOTS", ())
    monkeypatch.delenv(DATA_ROOT_ENV_VAR, raising=False)

    paths = roots.resolve_patch_data_paths("auto")

    assert paths.root == mounted_root / "dataset"


def test_auto_data_root_refuses_unrelated_complete_kaggle_shards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Complete shard filenames alone are not enough for auto resolution."""
    kaggle_input = tmp_path / "kaggle" / "input"
    unrelated_root = kaggle_input / "unrelated-dataset"
    _write_complete_root(unrelated_root)
    monkeypatch.setattr(roots, "KAGGLE_INPUT_ROOT", kaggle_input)
    monkeypatch.setattr(roots, "KNOWN_AUTO_DATA_ROOTS", ())
    monkeypatch.delenv(DATA_ROOT_ENV_VAR, raising=False)

    with pytest.raises(FileNotFoundError, match=DATA_ROOT_ENV_VAR):
        roots.resolve_patch_data_paths("auto")

    diagnostics = roots.data_root_resolution_diagnostics("auto")
    complete_unaccepted = cast(
        "list[dict[str, object]]",
        diagnostics["complete_unaccepted_candidates"],
    )
    assert any(
        "unrelated-dataset" in cast("str", candidate["candidate_root"])
        for candidate in complete_unaccepted
    )


def test_data_root_diagnostics_report_kaggle_input_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Diagnostics expose safe mount paths and missing files for logs/artifacts."""
    kaggle_input = tmp_path / "kaggle" / "input"
    mounted_root = kaggle_input / "patches-pre-shuffled-ubc-ocean"
    _write_complete_root(mounted_root)
    monkeypatch.setattr(roots, "KAGGLE_INPUT_ROOT", kaggle_input)
    monkeypatch.setattr(roots, "KNOWN_AUTO_DATA_ROOTS", ())
    monkeypatch.delenv(DATA_ROOT_ENV_VAR, raising=False)

    diagnostics = roots.data_root_resolution_diagnostics("auto")

    assert diagnostics["requested_data_root"] == "auto"
    assert diagnostics["kaggle_input_exists"] is True
    assert diagnostics["kaggle_input_scan_truncated"] is False
    assert "env_value" not in diagnostics
    candidates = cast("list[dict[str, object]]", diagnostics["candidates"])
    assert any(candidate["complete"] is True for candidate in candidates)
    accepted_candidates = cast(
        "list[dict[str, object]]",
        diagnostics["accepted_candidates"],
    )
    assert any(
        candidate["candidate_is_expected_kaggle_mount"] is True
        for candidate in accepted_candidates
    )
    assert diagnostics["complete_unaccepted_candidate_count"] == 0
    snapshot = cast("list[dict[str, object]]", diagnostics["kaggle_input_snapshot"])
    assert any(
        item["relative_path"] == "patches-pre-shuffled-ubc-ocean" for item in snapshot
    )


def test_missing_data_root_reports_all_required_files(tmp_path: Path) -> None:
    """Missing roots fail before a loader can use an incomplete shard."""
    with pytest.raises(FileNotFoundError, match=TRAIN_BIN_NAME):
        resolve_patch_data_paths(tmp_path / "missing")


def _write_complete_root(root: Path) -> None:
    dataset = root / "dataset"
    spec = SyntheticPatchSpec(count=PATCH_COUNT, image_size=PATCH_SIZE)
    write_synthetic_patch_shard(
        bin_path=dataset / TRAIN_BIN_NAME,
        csv_path=dataset / TRAIN_CSV_NAME,
        spec=spec,
        include_idx=False,
    )
    write_synthetic_patch_shard(
        bin_path=dataset / VALIDATION_BIN_NAME,
        csv_path=dataset / VALIDATION_CSV_NAME,
        spec=spec,
        include_idx=True,
    )
