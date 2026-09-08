# Copyright 2026 HiperMaximus
# pyright: reportPrivateUsage=false
"""Focused invariants for the bounded WSI 45630 completion package."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable

import pytest

from eqvae.cli import build_ubc_wsi45630_completion as builder


def test_missing_subtraction_keeps_stored_but_unselected_coordinates_out() -> None:
    """Exclude stored patches even when absent from the current MIL bag.

    This DERIVED subtraction prevents re-encoding usable existing rows.
    """
    candidates = _rows((0, 0), (1, 256), (2, 512), (3, 768))
    union = _rows((0, 0), (2, 512))
    old = _rows((1, 256))

    missing, base_reuse, topup_reuse = builder.derive_missing(
        candidates=candidates,
        union=union,
        old=old,
    )

    assert missing == ((3, builder.WSI_ID, 768, 0),)
    assert (base_reuse, topup_reuse) == (2, 1)


def test_identity_drift_at_an_intersecting_coordinate_is_rejected() -> None:
    """Reject atlas identity drift at a reused coordinate.

    This DELIBERATE guard prevents aligning bytes to the wrong atlas patch.
    """
    candidates = _rows((0, 0), (1, 256))
    union = _rows((99, 0))

    with pytest.raises(ValueError, match="identity drift"):
        builder.derive_missing(candidates=candidates, union=union, old={})


def test_wrong_split_candidate_rows_are_rejected_before_derivation(
    tmp_path: Path,
) -> None:
    """Reject non-training candidates before extracting coordinates.

    This DELIBERATE guard keeps the label-free worker training-only.
    """
    path = tmp_path / "candidates.csv"
    path.write_text(
        "atlas_row_index,wsi_id,diagnosis_label,diagnosis_index,x,y,split\n"
        f"0,{builder.WSI_ID},HGSC,2,0,0,validation\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="train"):
        builder._load_candidates_train_only(path)  # noqa: SLF001


def test_duplicate_coordinate_is_rejected_even_with_distinct_atlas_rows(
    tmp_path: Path,
) -> None:
    """Enforce coordinate uniqueness independently of atlas-row uniqueness.

    This DELIBERATE guard prevents encoding one physical patch twice.
    """
    path = tmp_path / "candidates.csv"
    path.write_text(
        "atlas_row_index,wsi_id,diagnosis_label,diagnosis_index,x,y,split\n"
        f"0,{builder.WSI_ID},HGSC,2,0,0,train\n"
        f"1,{builder.WSI_ID},HGSC,2,0,0,train\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate"):
        builder._load_candidates_train_only(path)  # noqa: SLF001


def test_runtime_template_calls_the_unchanged_worker_with_only_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass only the legacy four-file inference root to the reused worker.

    This DELIBERATE isolation keeps source files outside the worker's inputs.
    """
    template_path = (
        Path(__file__).parents[1] / "kaggle/kernels/wsi45630_completion/run_template.py"
    )
    spec = importlib.util.spec_from_file_location(
        "wsi45630_template",
        template_path,
    )
    assert spec is not None
    assert spec.loader is not None
    template = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(template)
    calls: dict[str, Path] = {}

    def fake_run_topup(**kwargs: Path) -> dict[str, object]:
        calls.update(kwargs)
        return {"status": "complete", "row_count": builder.MISSING_ROWS}

    monkeypatch.setattr(
        "eqvae.cli.generate_ubc_cancer_topup.run_topup",
        fake_run_topup,
    )
    monkeypatch.setattr(template, "WORKING_ROOT", tmp_path / "working")
    bundle = tmp_path / "bundle"
    wsi_dir = tmp_path / "train_images"

    execute = cast("Callable[[Path, Path], None]", template._execute)  # noqa: SLF001
    execute(bundle, wsi_dir)

    assert calls == {
        "input_root": bundle / "inference",
        "wsi_dir": wsi_dir,
        "output_root": tmp_path / "working/dataset",
        "scratch_root": tmp_path / "working/.wsi45630_work",
    }


def _rows(*values: tuple[int, int]) -> dict[builder.Identity, object]:
    return {(atlas, builder.WSI_ID, x, 0): None for atlas, x in values}
