# Copyright 2026 HiperMaximus
"""Tests for Spec 0008 fixed-32 selector readiness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

from eqvae.benchmarking.fixed32_selector_readiness import (
    EXPECTED_TINY_SELECTOR_COUNT,
    Fixed32RemoteGenerateReadinessRequest,
    fixed32_selector_status,
    write_fixed32_remote_generate_readiness,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def test_fixed32_remote_generate_readiness_rejects_synthetic_selector(
    tmp_path: Path,
) -> None:
    """Local synthetic generation passes only as non-canonical readiness proof."""
    result = write_fixed32_remote_generate_readiness(
        Fixed32RemoteGenerateReadinessRequest(
            output_dir=tmp_path / "readiness",
            synthetic_root=tmp_path / "synthetic-root",
            config_path=Path("configs/spec0001/non_eq_vae_kaggle_tiny_overfit.json"),
            masked_holdout_csv=Path("docs/data/ubc_ocean_masked_holdout_ids.csv"),
            image_size=8,
            channels=3,
        ),
    )

    readiness = _load_json(result.readiness_path)
    synthetic_status = _object(readiness, "synthetic_selector_status")
    assert readiness["status"] == "pass"
    assert readiness["selector_generation_mode"] == "remote_generate"
    assert readiness["remote_selector_generation_ready"] is True
    assert readiness["fixed_32_selector_real"] is False
    assert readiness["synthetic_selector_deterministic"] is True
    assert readiness["synthetic_selector_canonical_real_rejected"] is True
    assert synthetic_status["selector_count"] == EXPECTED_TINY_SELECTOR_COUNT
    assert synthetic_status["failure_kind"] == (
        "fixed_32_selector_not_canonical_real_ubc"
    )


def test_fixed32_selector_status_rejects_mutated_selector_payloads(
    tmp_path: Path,
) -> None:
    """Placeholder, wrong-count, wrong-dataset, and no-CRC selectors fail."""
    result = write_fixed32_remote_generate_readiness(
        Fixed32RemoteGenerateReadinessRequest(
            output_dir=tmp_path / "readiness",
            synthetic_root=tmp_path / "synthetic-root",
            config_path=Path("configs/spec0001/non_eq_vae_kaggle_tiny_overfit.json"),
            masked_holdout_csv=Path("docs/data/ubc_ocean_masked_holdout_ids.csv"),
            image_size=8,
            channels=3,
        ),
    )
    payload = _load_json(result.selector_path)

    placeholder_status = fixed32_selector_status(
        Path("configs/spec0001/fixed_32_train_overfit_patches.json"),
        data_root=None,
    )
    assert placeholder_status["failure_kind"] == "fixed_32_selector_placeholder"

    wrong_count = _write_mutated_selector(
        tmp_path / "wrong_count.json",
        payload,
        _drop_one_selector,
    )
    wrong_dataset = _write_mutated_selector(
        tmp_path / "wrong_dataset.json",
        payload,
        lambda candidate: _object(candidate, "source").__setitem__(
            "dataset_slug",
            "wrong/dataset",
        ),
    )
    no_crc = _write_mutated_selector(
        tmp_path / "no_crc.json",
        payload,
        _set_no_crc,
    )

    assert (
        fixed32_selector_status(
            wrong_count,
            data_root=str(tmp_path / "synthetic-root"),
        )["failure_kind"]
        == "fixed_32_selector_count_not_32"
    )
    assert (
        fixed32_selector_status(
            wrong_dataset,
            data_root=str(tmp_path / "synthetic-root"),
        )["failure_kind"]
        == "fixed_32_selector_wrong_dataset"
    )
    assert (
        fixed32_selector_status(
            no_crc,
            data_root=str(tmp_path / "synthetic-root"),
        )["failure_kind"]
        == "fixed_32_selector_crc_not_checked"
    )


def _write_mutated_selector(
    path: Path,
    payload: dict[str, object],
    mutate: Callable[[dict[str, object]], None],
) -> Path:
    candidate = cast("object", json.loads(json.dumps(payload)))
    if not isinstance(candidate, dict):
        raise TypeError(path)
    typed_candidate = cast("dict[str, object]", candidate)
    mutate(typed_candidate)
    path.write_text(
        f"{json.dumps(typed_candidate, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    return path


def _drop_one_selector(candidate: dict[str, object]) -> None:
    selectors = candidate.get("selectors")
    if not isinstance(selectors, list):
        message = "selectors"
        raise TypeError(message)
    selectors.pop()


def _set_no_crc(candidate: dict[str, object]) -> None:
    _object(candidate, "source")["crc_checked"] = False


def _load_json(path: Path) -> dict[str, object]:
    payload = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(payload, dict):
        raise TypeError(path)
    return cast("dict[str, object]", payload)


def _object(payload: dict[str, object], key: str) -> dict[str, object]:
    value = payload[key]
    if not isinstance(value, dict):
        raise TypeError(key)
    return cast("dict[str, object]", value)
