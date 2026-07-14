# Copyright 2026 HiperMaximus
"""Unit tests for the canonical selected-runtime row_id composer (Spec 0011 S17b)."""

from __future__ import annotations

import pytest

from eqvae.benchmarking.row_id import (
    DEFAULT_RUNTIME_POLICY_ID,
    compose_row_id_base,
    compose_selected_row_id,
)
from eqvae.training.selected_runtime import EXPECTED_SELECTED_ROW_ID


def test_compose_row_id_base_shape() -> None:
    """The base id encodes accelerator, batch, precision, scope, and corruption."""
    assert (
        compose_row_id_base(
            accelerator_mode="dual_t4_ddp",
            batch_size=12,
            precision_policy="amp_conservative",
            compile_scope="none",
            corruption_strategy="indexed_masked",
        )
        == "dual_t4_ddp__bs12__amp_conservative__compile_none__indexed_masked"
    )


def test_compose_selected_row_id_appends_distinguishing_policy_suffix() -> None:
    """A distinguishing runtime policy id is appended as a ``__policy_`` suffix."""
    assert compose_selected_row_id(
        accelerator_mode="dual_t4_ddp",
        batch_size=48,
        precision_policy="amp_off_fp32",
        compile_scope="step",
        corruption_strategy="indexed_masked",
        runtime_policy_id="compile_step_ddp_optimizer_fp32_channels_last",
    ) == (
        "dual_t4_ddp__bs48__amp_off_fp32__compile_step__indexed_masked__"
        "policy_compile_step_ddp_optimizer_fp32_channels_last"
    )


@pytest.mark.parametrize("policy_id", ["", DEFAULT_RUNTIME_POLICY_ID])
def test_compose_selected_row_id_suppresses_suffixless_policy(policy_id: str) -> None:
    """The empty id and the default eager policy carry no suffix (bare base)."""
    base = compose_row_id_base(
        accelerator_mode="dual_t4_ddp",
        batch_size=8,
        precision_policy="amp_off_fp32",
        compile_scope="none",
        corruption_strategy="branchless_all",
    )
    assert (
        compose_selected_row_id(
            accelerator_mode="dual_t4_ddp",
            batch_size=8,
            precision_policy="amp_off_fp32",
            compile_scope="none",
            corruption_strategy="branchless_all",
            runtime_policy_id=policy_id,
        )
        == base
    )


def test_compose_selected_row_id_reproduces_committed_v5_id() -> None:
    """The composer reproduces the committed v5 fallback id byte for byte.

    This pins the leaf-vs-parser contract: the id the parser structurally validates
    against (``EXPECTED_SELECTED_ROW_ID``) is exactly what the composer emits from the
    v5 fields, so a drift in either side is caught here.
    """
    assert (
        compose_selected_row_id(
            accelerator_mode="dual_t4_ddp",
            batch_size=12,
            precision_policy="amp_conservative",
            compile_scope="none",
            corruption_strategy="indexed_masked",
            runtime_policy_id="amp_fp16_conservative",
        )
        == EXPECTED_SELECTED_ROW_ID
    )
