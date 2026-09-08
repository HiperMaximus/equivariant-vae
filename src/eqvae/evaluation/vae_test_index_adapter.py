# Copyright 2026 HiperMaximus
# pyright: reportAny=false, reportExplicitAny=false
# ruff: noqa: DOC201, EM101, TRY003
"""Administrative diagnosis-index adapter for the Spec 0045 frozen scorer."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Final

from eqvae.evaluation.vae_test_scoring import (
    DIAGNOSIS_TO_INDEX,
    OracleLabel,
    score_vae_test_metrics,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

FROZEN_ORACLE_DIAGNOSIS_TO_INDEX: Final = {
    "CC": 0,
    "EC": 1,
    "HGSC": 2,
    "LGSC": 3,
    "MC": 4,
}


def score_with_frozen_oracle_indices(
    *,
    remote_rows: Sequence[Mapping[str, object]],
    labels_by_atlas_row: Mapping[int, OracleLabel],
) -> dict[str, object]:
    """Validate oracle numbering, then satisfy only the frozen redundant invariant."""
    scorer_labels = _adapt_oracle_labels(labels_by_atlas_row)
    return score_vae_test_metrics(
        remote_rows=remote_rows,
        labels_by_atlas_row=scorer_labels,
    )


def _adapt_oracle_labels(
    labels_by_atlas_row: Mapping[int, OracleLabel],
) -> dict[int, OracleLabel]:
    scorer_labels: dict[int, OracleLabel] = {}
    for atlas_row_index, label in labels_by_atlas_row.items():
        expected_oracle_index = FROZEN_ORACLE_DIAGNOSIS_TO_INDEX.get(
            label.diagnosis_label,
        )
        if (
            expected_oracle_index is None
            or label.diagnosis_index != expected_oracle_index
        ):
            raise ValueError("Spec 0047 frozen oracle diagnosis mapping differs")
        scorer_labels[atlas_row_index] = replace(
            label,
            diagnosis_index=DIAGNOSIS_TO_INDEX[label.diagnosis_label],
        )
    return scorer_labels


__all__ = ["FROZEN_ORACLE_DIAGNOSIS_TO_INDEX", "score_with_frozen_oracle_indices"]
