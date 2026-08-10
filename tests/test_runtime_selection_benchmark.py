# Copyright 2026 HiperMaximus
"""Tests for the selected-runtime benchmark proof path."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
import torch

from eqvae.benchmarking import real_data_runtime_pretest as pretest
from eqvae.benchmarking import runtime_selection_executor
from eqvae.benchmarking.io import JsonObject, write_csv, write_json
from eqvae.benchmarking.real_data_runtime_pretest import RowSpec
from eqvae.benchmarking.real_data_runtime_pretest import (
    _base_row as _pretest_base_row,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
)
from eqvae.benchmarking.real_data_runtime_pretest import (
    _settings as _pretest_settings,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
)
from eqvae.benchmarking.real_data_runtime_pretest import (
    _stage1_row_specs as _pretest_stage1_row_specs,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
)
from eqvae.benchmarking.runtime_schema import (
    CORRUPTION_CHECK_COLUMNS,
    DATALOADER_MATRIX_COLUMNS,
    EAGER_RECIPE_KNOB_COLUMNS,
    GATE_HEALTH_COLUMNS,
    NUMERICAL_CHECK_COLUMNS,
    RUNTIME_MATRIX_COLUMNS,
)
from eqvae.benchmarking.runtime_schema import (
    _runtime_rows as _synthetic_runtime_rows,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
)
from eqvae.benchmarking.runtime_selection import (
    COMPILE_MODEL_FORWARD,
    COMPILE_STEP,
    INELIGIBLE_STATUS,
    PASS_STATUS,
    RuntimeSelectionBenchmarkRequest,
    RuntimeSelectionEvidence,
    _bool_from_csv,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _compile_settle_protocol_id,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _compiled_row_stable,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _EfficiencyPolicySelection,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _enforce_compiled_rows_diagnostic_only,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _optional_int_from_csv,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _reference_row_id,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _runtime_row_candidate_pass,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _selected_runtime_payload,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _selection_candidate_scope_matches,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _SelectionSettings,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    load_runtime_selection_evidence,
    write_runtime_selection_benchmark,
)
from eqvae.benchmarking.runtime_selection import (
    _runtime_row as _src_runtime_row,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
)
from eqvae.benchmarking.runtime_selection_executor import (
    _COMPILED_EXECUTION_PROOF_FAILURE_KIND,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _DDP_RUNTIME_OOM_FAILURE_KIND,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _STEP_COMPILE_BACKEND,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _VRAM_FEASIBILITY_AMP_SKIPS_FAILURE_KIND,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _VRAM_INFEASIBLE_FAILURE_KIND,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    RuntimeSelectionExecutionRequest,
    _amp_phase_accounting,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _build_compiled_ddp_step,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _compile_ddp_model_if_requested,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _ddp_rank_failure_payload,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _DdpLaunchResult,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _DdpRowConfig,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _decode_ddp_config,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _dual_corruption_rows,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _dual_dataloader_rows,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _dual_numerical_rows,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _dual_row_from_rank_payloads,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _dual_row_specs,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _efficiency_row_enumerable,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _encode_ddp_config,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _failure_row,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _FeasibilityProbeAmpStepsSkippedError,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _parameter_update_parity,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _probe_headroom_after_successful_optimizer_update,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _row_proof_reference_batch_size,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _run_compiled_ddp_execution_proof,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _run_dual_row,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _run_until_successful_amp_proof,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _runtime_policies,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _selection_stage_settings,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _successful_feasibility_optimizer_updates,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _vram_infeasible_rank_payload,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
)
from eqvae.benchmarking.runtime_selection_executor import (
    _base_selection_row as _executor_base_selection_row,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
)
from eqvae.benchmarking.runtime_selection_executor import (
    _row_spec as _executor_row_spec,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
)
from eqvae.benchmarking.runtime_selection_executor import (
    _RuntimePolicy as _ExecutorRuntimePolicy,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
)
from eqvae.config import resolve_json_config
from eqvae.data.roots import REAL_TRAIN_PATCH_COUNT
from eqvae.models.activations import GatedScalarActivation
from eqvae.models.non_equivariant_vae import build_non_equivariant_vae
from eqvae.training.fastpath_step import FastpathStepOutput

CONFIG_PATH = Path("configs/spec0001/non_eq_vae_kaggle_runtime_benchmark.json")
RUN_NAME = "runtime_selection_test"
CORRUPTION_STRATEGIES = ("branchless_all", "indexed_masked")
EXPECTED_DUAL_RUNTIME_ROWS = 6
EXPECTED_DUAL_WORLD_SIZE = 2
# Spec 0011 S14c: the compiled bigger-batch winner candidate and its DDP bucket cap,
# plus the efficiency slice's policy count (kept amp follow-up + the compiled winner).
_WINNER_BATCH_SIZE = 48
_WINNER_BUCKET_CAP_MB = 50
_PROOF_REFERENCE_BATCH_SIZE = 12
_FUTURE_NON_EFFICIENCY_BATCH_SIZE = 24
_ORIGINAL_COMPILED_POLICY_IDS = {
    "compile_step_ddp_optimizer_fp32_channels_last",
    "compile_step_ddp_optimizer_fp16_channels_last",
}
_DIAGNOSTIC_FSQ_POLICY_ID = "diagnostic_fsq_compiled_autograd_no_optimization_fp16"
_EFFICIENCY_POLICY_COUNT = 3
# Measured winner DDP bucket-cap (probe _DDP_OPTIMIZER_SPEC).
_RECIPE_BUCKET_CAP_MB = 50
# The eager-recipe optimize_ddp sentinel (unset dynamo config).
_EAGER_OPTIMIZE_DDP = ""

if TYPE_CHECKING:
    from collections.abc import Mapping


def _payload_settings() -> _SelectionSettings:
    """Minimal selection settings for a direct payload-emitter unit test.

    The payload emitter reads only ``real_train_patch_count`` and
    ``effective_config_hash`` from settings; the rest are inert placeholders.

    Returns:
        A ``_SelectionSettings`` sufficient to build one selected-runtime payload.

    """
    return _SelectionSettings(
        run_name=RUN_NAME,
        effective_config_hash="unit-config-hash",
        real_train_patch_count=REAL_TRAIN_PATCH_COUNT,
        warmup_steps=5,
        measured_steps=25,
        repeats=3,
        v8_artifact_dir=Path("unused-v8"),
        fp32_batch_sizes=(4, 8, 12),
        fallback_batch_sizes=(12,),
        dual_batch_sizes=(12,),
        corruption_strategies=("indexed_masked",),
        baseline_selected_runtime_path=None,
        baseline_selected_row_id="",
        baseline_runtime_policy_id="",
        minimum_material_speedup_fraction=0.05,
        efficiency_accelerator_modes=("dual_t4_ddp",),
        efficiency_batch_sizes=(12,),
        efficiency_proof_reference_batch_size=_PROOF_REFERENCE_BATCH_SIZE,
        efficiency_corruption_strategies=("indexed_masked",),
        efficiency_policies=(),
    )


def _eager_selected_row() -> dict[str, str]:
    """Return an eager dual-T4 winner row carrying the eager recipe columns.

    Post-S13 the shared ``_runtime_row`` spreads ``EAGER_RECIPE_KNOB_COLUMNS``, so this
    row carries all seven knob columns at their eager values. The absent-column ``.get``
    fallback (pre-S13 CSVs) is covered by
    ``test_selected_runtime_payload_reads_eager_recipe_from_legacy_row``.

    Returns:
        A runtime-matrix row for an eager amp-conservative dual-T4 candidate.

    """
    return _runtime_row(
        accelerator_mode="dual_t4_ddp",
        per_device_batch_size=12,
        precision_policy="amp_conservative",
        compile_scope="none",
        corruption_strategy="indexed_masked",
        world_size=EXPECTED_DUAL_WORLD_SIZE,
        samples_sec=300.0,
        runtime_policy_id="amp_fp16_conservative",
    )


def test_non_efficiency_row_keeps_same_batch_reference() -> None:
    """The cross-batch proof rule applies only to configured efficiency policies.

    The proof batch is an evidence control for the efficiency slice, not a global
    rewrite of every future row. A non-efficiency bs48 row is expected to keep bs48 in
    its reference identity, so applying ``min(candidate, proof)`` unconditionally fails.
    """
    row = _runtime_row(
        accelerator_mode="single_visible_t4",
        per_device_batch_size=48,
        precision_policy="amp_off_fp32",
        compile_scope="none",
        corruption_strategy="indexed_masked",
        world_size=1,
        samples_sec=1.0,
    )

    reference = _reference_row_id(settings=_payload_settings(), runtime_row=row)

    assert "__bs48__" in reference


def test_writer_requires_explicit_efficiency_proof_batch(tmp_path: Path) -> None:
    """Writer settings reject an omitted efficiency proof-reference batch.

    Cross-batch linked evidence must never inherit an implicit batch; an exact parser
    error is expected, so restoring an optional/default value must fail.
    """
    config_path = _write_config_with_efficiency_policies(
        tmp_path=tmp_path,
        policies=None,
    )
    payload = _load_json(config_path)
    runtime = cast("dict[str, object]", payload["runtime_matrix"])
    selection = cast("dict[str, object]", runtime["selection_benchmark_slice"])
    efficiency = cast("dict[str, object]", selection["efficiency_followup"])
    efficiency.pop("proof_reference_per_device_batch_size")
    write_json(config_path, cast("JsonObject", payload))

    with pytest.raises(
        TypeError,
        match="Expected integer field: proof_reference_per_device_batch_size",
    ):
        write_runtime_selection_benchmark(
            RuntimeSelectionBenchmarkRequest(
                config_path=config_path,
                output_dir=tmp_path / "output",
            ),
        )


def test_writer_validates_efficiency_proof_batch_against_fp32_gate(
    tmp_path: Path,
) -> None:
    """Writer settings require the proof batch to be present in the fp32 gate.

    Batch 48 is a timed efficiency candidate but not a dual-gate reference; the shared
    validator's exact rejection is expected, so bypassing validation must fail.
    """
    config_path = _write_config_with_efficiency_policies(
        tmp_path=tmp_path,
        policies=None,
    )
    payload = _load_json(config_path)
    runtime = cast("dict[str, object]", payload["runtime_matrix"])
    selection = cast("dict[str, object]", runtime["selection_benchmark_slice"])
    efficiency = cast("dict[str, object]", selection["efficiency_followup"])
    efficiency["proof_reference_per_device_batch_size"] = 48
    write_json(config_path, cast("JsonObject", payload))

    with pytest.raises(
        ValueError,
        match="must be measured by the dual_t4_train_step_gate",
    ):
        write_runtime_selection_benchmark(
            RuntimeSelectionBenchmarkRequest(
                config_path=config_path,
                output_dir=tmp_path / "output",
            ),
        )


def test_selected_runtime_payload_emits_recipe_knobs_at_eager_defaults() -> None:
    """A winner row carrying eager recipe columns yields the eager recipe (S11/S13).

    Post-S13 every produced row carries the recipe-knob columns at their eager values
    (``EAGER_RECIPE_KNOB_COLUMNS``), so ``_selected_runtime_payload`` reproduces the v5
    recipe -- no DDPOptimizer, no compiled autograd, DDP-library defaults for
    broadcast/find-unused/bucket-cap, fused off. The absent-column ``.get`` fallback for
    pre-S13 CSVs is covered by
    ``test_selected_runtime_payload_reads_eager_recipe_from_legacy_row``.
    """
    selected_row = _eager_selected_row()
    dataloader_rows = tuple(_dataloader_rows((selected_row,)))

    payload = _selected_runtime_payload(
        settings=_payload_settings(),
        selected_row=selected_row,
        dataloader_rows=dataloader_rows,
        artifact_hashes={},
    )

    torch_compile = cast("dict[str, object]", payload["torch_compile"])
    runtime_policy = cast("dict[str, object]", payload["runtime_policy"])
    assert torch_compile["optimize_ddp"] == _EAGER_OPTIMIZE_DDP
    assert torch_compile["compiled_autograd"] is False
    assert torch_compile["reorder_compute_comm_overlap"] is False
    assert runtime_policy["ddp_broadcast_buffers"] is True
    assert runtime_policy["ddp_find_unused_parameters"] is False
    assert runtime_policy["ddp_bucket_cap_mb"] is None
    assert runtime_policy["fused_optimizer"] is False
    assert runtime_policy["gradient_clip_foreach"] is True
    assert runtime_policy["gradient_clip_foreach_applied"] is True


def test_selected_runtime_payload_sources_recipe_knobs_from_measured_row() -> None:
    """The generator sources each recipe knob from its measured winner-row column (S11).

    Proves the ``.get(col, ...)`` reads the real column when present (the forward path
    S13/S14 populate), routing each knob into its frozen carrier block: dynamo knobs to
    ``torch_compile``, DDP/optimizer knobs to ``runtime_policy``.
    """
    # Every knob column is set to a value distinct from its eager default so a dropped
    # or wrong-block read is caught (mutation-proof). The emitter does no validation, so
    # this deliberately artificial combination only exercises the per-column sourcing.
    selected_row = {
        **_eager_selected_row(),
        "optimize_ddp": "ddp_optimizer",
        "compiled_autograd": "true",
        "reorder_compute_comm_overlap": "true",
        "ddp_broadcast_buffers": "false",
        "ddp_find_unused_parameters": "true",
        "ddp_bucket_cap_mb": str(_RECIPE_BUCKET_CAP_MB),
        "fused_optimizer": "true",
        "gradient_clip_foreach": "true",
    }
    dataloader_rows = tuple(_dataloader_rows((selected_row,)))

    payload = _selected_runtime_payload(
        settings=_payload_settings(),
        selected_row=selected_row,
        dataloader_rows=dataloader_rows,
        artifact_hashes={},
    )

    torch_compile = cast("dict[str, object]", payload["torch_compile"])
    runtime_policy = cast("dict[str, object]", payload["runtime_policy"])
    assert torch_compile["optimize_ddp"] == "ddp_optimizer"
    assert torch_compile["compiled_autograd"] is True
    assert torch_compile["reorder_compute_comm_overlap"] is True
    assert runtime_policy["ddp_broadcast_buffers"] is False
    assert runtime_policy["ddp_find_unused_parameters"] is True
    assert runtime_policy["ddp_bucket_cap_mb"] == _RECIPE_BUCKET_CAP_MB
    assert runtime_policy["fused_optimizer"] is True
    assert runtime_policy["gradient_clip_foreach"] is True


def test_compiled_selected_snapshot_hashes_its_real_settle_protocol() -> None:
    """A compiled winner hashes its derived settle protocol, not the eager policy.

    The snapshot is evidence consumed after measurement, so this derived relationship
    must change with the row's compile protocol. Reusing the eager constant would make
    both the exact-hash and unequal-to-eager assertions fail.
    """
    selected_row = _fp16_compiled_bs48_row()
    payload = _selected_runtime_payload(
        settings=_payload_settings(),
        selected_row=selected_row,
        dataloader_rows=tuple(_dataloader_rows((selected_row,))),
        artifact_hashes={},
    )
    snapshot = cast("dict[str, object]", payload["selected_row_snapshot"])
    compiled_protocol = _compile_settle_protocol_id(selected_row)
    eager_hash = hashlib.sha256(
        b"runtime_selection_compile_none_eager_no_settle_v1",
    ).hexdigest()

    assert (
        snapshot["compile_settle_protocol_sha256"]
        == hashlib.sha256(
            compiled_protocol.encode(),
        ).hexdigest()
    )
    assert snapshot["compile_settle_protocol_sha256"] != eager_hash


def test_selected_runtime_payload_reads_eager_recipe_from_legacy_row() -> None:
    """A pre-S13 winner row (no recipe columns) still yields the eager recipe (S11).

    Proves the additive contract survives old artifacts: a row missing every recipe
    column (e.g. a committed pre-S13 ``runtime_matrix.csv``) resolves each knob through
    the ``.get(col, eager_default)`` fallback, so the emitted plan matches the eager v5
    recipe. This is the absent-key branch the post-S13 producers no longer exercise.
    """
    legacy_row = {
        key: value
        for key, value in _eager_selected_row().items()
        if key not in EAGER_RECIPE_KNOB_COLUMNS
    }
    assert not (EAGER_RECIPE_KNOB_COLUMNS.keys() & legacy_row.keys())
    dataloader_rows = tuple(_dataloader_rows((legacy_row,)))

    payload = _selected_runtime_payload(
        settings=_payload_settings(),
        selected_row=legacy_row,
        dataloader_rows=dataloader_rows,
        artifact_hashes={},
    )

    torch_compile = cast("dict[str, object]", payload["torch_compile"])
    runtime_policy = cast("dict[str, object]", payload["runtime_policy"])
    assert torch_compile["optimize_ddp"] == _EAGER_OPTIMIZE_DDP
    assert torch_compile["compiled_autograd"] is False
    assert runtime_policy["ddp_broadcast_buffers"] is True
    assert runtime_policy["ddp_bucket_cap_mb"] is None
    assert runtime_policy["fused_optimizer"] is False


def _real_producer_runtime_rows() -> dict[str, dict[str, str]]:
    """One runtime-matrix row from each of the four real column producers.

    Returns:
        A mapping from producer name to a freshly built runtime-matrix row.

    """
    pretest_settings = _pretest_settings(
        resolve_json_config(CONFIG_PATH),
        data_root_override=None,
    )
    pretest_row_spec = _pretest_stage1_row_specs(pretest_settings)[0]
    return {
        "runtime_schema._runtime_rows": dict(
            _synthetic_runtime_rows(
                run_name=RUN_NAME,
                max_benchmark_rows=1,
                warmup_steps=5,
                measured_steps=25,
            )[0],
        ),
        "real_data_runtime_pretest._base_row": dict(
            _pretest_base_row(settings=pretest_settings, row_spec=pretest_row_spec),
        ),
        "runtime_selection_executor._base_selection_row": dict(
            _executor_base_selection_row(
                settings=pretest_settings,
                row_spec=pretest_row_spec,
            ),
        ),
        "runtime_selection._runtime_row": dict(
            _src_runtime_row(
                settings=_payload_settings(),
                row_id="dual_t4_ddp__bs12__amp_off_fp32__compile_none__branchless_all",
                accelerator_mode="dual_t4_ddp",
                per_device_batch_size=12,
                precision_policy="amp_off_fp32",
                compile_scope="none",
                corruption_strategy="branchless_all",
                world_size=EXPECTED_DUAL_WORLD_SIZE,
                nproc_per_node=EXPECTED_DUAL_WORLD_SIZE,
                status="pass",
                samples_sec=300.0,
                steady_step_ms_p50=25.0,
                steady_step_ms_p95=30.0,
            ),
        ),
    }


def test_real_producers_emit_parseable_recipe_knobs_through_csv_roundtrip(
    tmp_path: Path,
) -> None:
    """Real RUNTIME_MATRIX_COLUMNS producers survive a CSV round-trip parse (S13).

    The DictWriter ``restval=''`` trap bites the SOURCE producers, not only the in-test
    helper: a produced ``runtime_matrix.csv`` is later reloaded (e.g. Kaggle replay via
    ``load_runtime_selection_evidence`` -> ``_selected_runtime_payload`` ->
    ``_bool_from_csv``). Had any producer omitted the recipe columns, the reloaded bool
    cells would be ``''`` and ``_bool_from_csv('')`` would raise. Driving all four real
    producers through the round-trip makes a reverted src spread fail HERE, not just on
    Kaggle.
    """
    produced = _real_producer_runtime_rows()
    reloaded_rows: dict[str, dict[str, str]] = {}
    for name, row in produced.items():
        matrix_path = tmp_path / f"{name.replace('.', '__')}.csv"
        write_csv(matrix_path, RUNTIME_MATRIX_COLUMNS, (row,))
        with matrix_path.open(encoding="utf-8", newline="") as handle:
            reloaded = next(iter(csv.DictReader(handle)))
        reloaded_rows[name] = reloaded
        for column, eager_value in EAGER_RECIPE_KNOB_COLUMNS.items():
            assert reloaded[column] == eager_value, (name, column)
        # The exact operation that raises on an omitted bool column ('' -> ValueError).
        assert _bool_from_csv(reloaded["compiled_autograd"]) is False
        assert _bool_from_csv(reloaded["ddp_broadcast_buffers"]) is True
        assert _bool_from_csv(reloaded["fused_optimizer"]) is False
        assert _optional_int_from_csv(reloaded["ddp_bucket_cap_mb"]) is None

    # End-to-end: the reloaded pass winner still builds the eager plan without crashing.
    winner = reloaded_rows["runtime_selection._runtime_row"]
    payload = _selected_runtime_payload(
        settings=_payload_settings(),
        selected_row=winner,
        dataloader_rows=tuple(_dataloader_rows((winner,))),
        artifact_hashes={},
    )
    runtime_policy = cast("dict[str, object]", payload["runtime_policy"])
    assert runtime_policy["fused_optimizer"] is False
    assert runtime_policy["ddp_broadcast_buffers"] is True


def test_runtime_policy_enables_cudnn_benchmark_by_default() -> None:
    """A dual-T4 policy autotunes cuDNN by default (Spec 0011 S17f).

    The benchmark config declares no cuDNN axis, so both the dataclass field default and
    the config-parse default must yield ``benchmark=True`` -- ``_apply_backend_policy``
    then measures each row under the same cuDNN autotuning the paper-promotable run
    uses, not the slower ``benchmark=False`` torch default. ``deterministic`` stays
    ``False``
    (speed-first). Both defaults are asserted so a revert of either is caught.
    """
    direct = _ExecutorRuntimePolicy(
        runtime_policy_id="fp32_eager_default",
        precision_policy="amp_off_fp32",
        compile_scope="none",
    )
    (parsed,) = _runtime_policies([
        {
            "runtime_policy_id": "fp32_eager_default",
            "precision_policy": "amp_off_fp32",
            "compile_scope": "none",
        },
    ])

    assert direct.cudnn_benchmark is True
    assert direct.cudnn_deterministic is False
    assert parsed.cudnn_benchmark is True
    assert parsed.cudnn_deterministic is False


def _measured_recipe_policy() -> _ExecutorRuntimePolicy:
    """Build a compiled-recipe policy whose knobs differ from their eager defaults.

    The distinct-from-eager combination is deliberately artificial (real winners keep
    ``compiled_autograd`` off): it only proves each knob is threaded/emitted from its
    own field, so a dropped or mis-keyed knob is caught (mutation-proof). The
    ``compile_scope`` is ``model_forward`` (a stable knob-carrying scope); the recipe
    knobs are scope-independent, and S14b's whole-step admission is covered separately
    by ``test_runtime_policies_admits_whole_step_compile_scope``.

    Returns:
        A ``_RuntimePolicy`` carrying all seven recipe knobs at non-eager values.

    """
    return _ExecutorRuntimePolicy(
        runtime_policy_id="compiled_whole_step_ddp_optimizer",
        precision_policy="amp_off_fp32",
        compile_scope=COMPILE_MODEL_FORWARD,
        optimize_ddp="ddp_optimizer",
        compiled_autograd=True,
        reorder_compute_comm_overlap=True,
        ddp_broadcast_buffers=False,
        ddp_find_unused_parameters=True,
        ddp_bucket_cap_mb=_RECIPE_BUCKET_CAP_MB,
        fused_optimizer=True,
    )


def test_runtime_policies_parses_measured_recipe_knobs() -> None:
    """``_runtime_policies`` reads each recipe knob from the policy config (S14a).

    The efficiency search declares a compiled winner policy carrying the seven recipe
    knobs; every value differs from its eager default so a dropped or mis-keyed parse
    is caught (mutation-proof).
    """
    (policy,) = _runtime_policies([
        {
            "runtime_policy_id": "compiled_whole_step_ddp_optimizer",
            "precision_policy": "amp_off_fp32",
            "compile_scope": COMPILE_MODEL_FORWARD,
            "optimize_ddp": "ddp_optimizer",
            "compiled_autograd": True,
            "reorder_compute_comm_overlap": True,
            "ddp_broadcast_buffers": False,
            "ddp_find_unused_parameters": True,
            "ddp_bucket_cap_mb": _RECIPE_BUCKET_CAP_MB,
            "fused_optimizer": True,
        },
    ])

    assert policy.optimize_ddp == "ddp_optimizer"
    assert policy.compiled_autograd is True
    assert policy.reorder_compute_comm_overlap is True
    assert policy.ddp_broadcast_buffers is False
    assert policy.ddp_find_unused_parameters is True
    assert policy.ddp_bucket_cap_mb == _RECIPE_BUCKET_CAP_MB
    assert policy.fused_optimizer is True


def test_runtime_policies_defaults_recipe_knobs_to_eager_when_absent() -> None:
    """A policy that declares no recipe knobs parses to the eager-v5 defaults (S14a).

    Guards behaviour-preservation: today's efficiency policy carries none of the seven
    knobs, so it must default to the eager recipe (empty ``optimize_ddp``, fused off,
    DDP library defaults) -- byte-identical to the pre-S14a policy.
    """
    (policy,) = _runtime_policies([
        {
            "runtime_policy_id": "amp_fp16_scalar_gate_relaxed",
            "precision_policy": "amp_scalar_gate_relaxed",
            "compile_scope": "none",
        },
    ])

    assert policy.optimize_ddp == _EAGER_OPTIMIZE_DDP
    assert policy.compiled_autograd is False
    assert policy.reorder_compute_comm_overlap is False
    assert policy.ddp_broadcast_buffers is True
    assert policy.ddp_find_unused_parameters is False
    assert policy.ddp_bucket_cap_mb is None
    assert policy.fused_optimizer is False


def test_executor_base_selection_row_emits_measured_recipe_knobs() -> None:
    """A compiled policy threads its knobs onto the RowSpec and the row emits them.

    Closes the producer gap the S11 payload test left open (S14a): S11 proved the plan
    emitter *reads* the knob columns, S13 proved every producer emits *eager* values;
    here the executor selection row emits the *measured* winner values (not the eager
    defaults), so ``_selected_runtime_payload`` reads the real recipe off a compiled
    winner row. Every knob differs from its eager default -> mutation-proof.
    """
    settings = _pretest_settings(
        resolve_json_config(CONFIG_PATH),
        data_root_override=None,
    )
    row_spec = _executor_row_spec(
        settings=settings,
        accelerator_mode="dual_t4_ddp",
        batch_size=48,
        corruption_strategy="indexed_masked",
        candidate_role="selected_runtime_efficiency_followup",
        policy=_measured_recipe_policy(),
    )
    assert row_spec.optimize_ddp == "ddp_optimizer"
    assert row_spec.ddp_bucket_cap_mb == _RECIPE_BUCKET_CAP_MB
    assert row_spec.fused_optimizer is True

    row = _executor_base_selection_row(settings=settings, row_spec=row_spec)

    assert row["optimize_ddp"] == "ddp_optimizer"
    assert row["compiled_autograd"] == "true"
    assert row["reorder_compute_comm_overlap"] == "true"
    assert row["ddp_broadcast_buffers"] == "false"
    assert row["ddp_find_unused_parameters"] == "true"
    assert row["ddp_bucket_cap_mb"] == str(_RECIPE_BUCKET_CAP_MB)
    assert row["fused_optimizer"] == "true"


def test_encode_ddp_config_round_trips_measured_recipe_knobs(tmp_path: Path) -> None:
    """The recipe knobs survive the RowSpec base64 round-trip to the child (S14a).

    ``_run_dual_row`` ships the RowSpec to the ``torchrun`` child as base64 JSON via
    ``_row_spec_payload`` (S14a routes ``_encode_ddp_config`` through it); had a knob
    been omitted from ``_row_spec_payload`` or ``_row_spec_from_payload`` the child
    would rebuild an eager RowSpec and silently measure the wrong recipe. Every knob
    differs from its eager default -> a dropped serialization field is caught here.
    """
    settings = _pretest_settings(
        resolve_json_config(CONFIG_PATH),
        data_root_override=None,
    )
    row_spec = _executor_row_spec(
        settings=settings,
        accelerator_mode="dual_t4_ddp",
        batch_size=48,
        corruption_strategy="indexed_masked",
        candidate_role="selected_runtime_efficiency_followup",
        policy=_measured_recipe_policy(),
    )
    config = _DdpRowConfig(
        config_path=CONFIG_PATH,
        output_dir=tmp_path,
        data_root="/unit/data/root",
        row_spec=row_spec,
        proof_reference_per_device_batch_size=12,
    )

    decoded = _decode_ddp_config(_encode_ddp_config(config))

    assert decoded.row_spec.optimize_ddp == "ddp_optimizer"
    assert decoded.row_spec.compiled_autograd is True
    assert decoded.row_spec.reorder_compute_comm_overlap is True
    assert decoded.row_spec.ddp_broadcast_buffers is False
    assert decoded.row_spec.ddp_find_unused_parameters is True
    assert decoded.row_spec.ddp_bucket_cap_mb == _RECIPE_BUCKET_CAP_MB
    assert decoded.row_spec.fused_optimizer is True


def test_pretest_base_row_emits_measured_recipe_knobs() -> None:
    """The pretest ``_base_row`` emits a RowSpec's measured recipe knobs (S14a).

    Both RowSpec-owning producers share ``_recipe_knob_columns``, so the pretest row --
    like the executor selection row -- must reflect a compiled RowSpec's knobs, not the
    eager constant. A revert to the constant spread fails here. Every knob differs from
    its eager default -> mutation-proof.
    """
    settings = _pretest_settings(
        resolve_json_config(CONFIG_PATH),
        data_root_override=None,
    )
    row_spec = _executor_row_spec(
        settings=settings,
        accelerator_mode="dual_t4_ddp",
        batch_size=48,
        corruption_strategy="indexed_masked",
        candidate_role="selected_runtime_efficiency_followup",
        policy=_measured_recipe_policy(),
    )

    row = _pretest_base_row(settings=settings, row_spec=row_spec)

    assert row["optimize_ddp"] == "ddp_optimizer"
    assert row["compiled_autograd"] == "true"
    assert row["reorder_compute_comm_overlap"] == "true"
    assert row["ddp_broadcast_buffers"] == "false"
    assert row["ddp_find_unused_parameters"] == "true"
    assert row["ddp_bucket_cap_mb"] == str(_RECIPE_BUCKET_CAP_MB)
    assert row["fused_optimizer"] == "true"


@pytest.mark.parametrize("compile_scope", [COMPILE_MODEL_FORWARD, COMPILE_STEP])
def test_settle_proven_compiled_row_is_selectable(compile_scope: str) -> None:
    """A coherent settle-proven compiled row is selectable at either compiled scope.

    This is a derived eligibility relationship, not a measured winner: Python reducer
    requires compiled autograd plus settle >=5 and zero breaks/recompiles. Removing a
    scope or accepting a blank/incoherent mode makes one of the exact assertions fail.
    """
    row = _runtime_row(
        accelerator_mode="dual_t4_ddp",
        per_device_batch_size=12,
        precision_policy="amp_conservative",
        compile_scope=compile_scope,
        corruption_strategy="indexed_masked",
        world_size=EXPECTED_DUAL_WORLD_SIZE,
        samples_sec=400.0,
        runtime_policy_id=f"compile_{compile_scope}_fp32_channels_last",
    )
    row["optimize_ddp"] = "python_reducer"
    row["compiled_autograd"] = "true"

    assert _compiled_row_stable(row) is True
    assert _enforce_compiled_rows_diagnostic_only((row,))[0]["status"] == PASS_STATUS


def test_stable_overlap_partition_is_selectable_with_reported_graph_breaks() -> None:
    """Stable overlap structure remains selectable with positive break telemetry.

    Graph-break count is a MEASURED diagnostic, not a correctness target: excluding a
    stable positive count would pre-select a graph aesthetic instead of the fastest
    end-to-end dual-T4 recipe. Post-settle recompilation remains the failure boundary.
    """
    row = _runtime_row(
        accelerator_mode="dual_t4_ddp",
        per_device_batch_size=12,
        precision_policy="amp_conservative",
        compile_scope=COMPILE_STEP,
        corruption_strategy="indexed_masked",
        world_size=EXPECTED_DUAL_WORLD_SIZE,
        samples_sec=400.0,
        runtime_policy_id="compile_step_fp32_channels_last",
    )
    row["optimize_ddp"] = "ddp_optimizer"
    row["graph_break_count"] = "2"

    assert _compiled_row_stable(row) is True
    assert _enforce_compiled_rows_diagnostic_only((row,))[0]["status"] == PASS_STATUS


@pytest.mark.parametrize(
    ("optimize_ddp", "graph_break_count", "recompile_count"),
    [
        ("", "0", "0"),
        ("unknown", "0", "0"),
        ("ddp_optimizer", "", "0"),
        ("ddp_optimizer", "-1", "0"),
        ("no_optimization", "0", "1"),
    ],
)
def test_compiled_stability_rejects_invalid_mode_telemetry(
    optimize_ddp: str,
    graph_break_count: str,
    recompile_count: str,
) -> None:
    """Compiled stability rejects unknown modes or unavailable/unstable telemetry.

    Exact false outcomes distinguish missing mode, missing/negative graph telemetry,
    and post-settle recompiles. Positive stable break counts are intentionally absent:
    they are diagnostic and remain throughput-eligible.
    """
    row = _runtime_row(
        accelerator_mode="dual_t4_ddp",
        per_device_batch_size=12,
        precision_policy="amp_conservative",
        compile_scope=COMPILE_STEP,
        corruption_strategy="indexed_masked",
        world_size=EXPECTED_DUAL_WORLD_SIZE,
        samples_sec=400.0,
        runtime_policy_id="compile_step_fp16_channels_last",
    )
    row["optimize_ddp"] = optimize_ddp
    row["graph_break_count"] = graph_break_count
    row["recompile_count"] = recompile_count

    assert _compiled_row_stable(row) is False
    normalized = _enforce_compiled_rows_diagnostic_only((row,))[0]
    assert normalized["status"] == INELIGIBLE_STATUS


def test_runtime_policies_admits_whole_step_compile_scope() -> None:
    """``_runtime_policies`` accepts a whole-step efficiency policy (S14b executor).

    S12/S13 opened the *selection* side to ``compile_scope == "step"``; S14b opens the
    *executor* so the efficiency search can time the compiled whole-step recipe.
    Dropping ``COMPILE_STEP`` from the accepted set makes this raise
    ``Unsupported compile_scope``.
    """
    (policy,) = _runtime_policies([
        {
            "runtime_policy_id": "compiled_whole_step_ddp_optimizer",
            "precision_policy": "amp_off_fp32",
            "compile_scope": COMPILE_STEP,
            "optimize_ddp": "ddp_optimizer",
        },
    ])

    assert policy.compile_scope == COMPILE_STEP
    assert policy.optimize_ddp == "ddp_optimizer"


def test_runtime_policies_rejects_unknown_compile_scope() -> None:
    """An unrecognized compile scope still fails closed after S14b opened ``step``.

    Guards the widened accepted set: only none/model_forward/step are executable, so a
    scope the executor cannot run (e.g. ``train_step_no_optimizer``) must raise rather
    than silently fall through to an eager measurement mislabeled as that scope.
    """
    with pytest.raises(ValueError, match="Unsupported compile_scope"):
        _runtime_policies([
            {
                "runtime_policy_id": "compiled_train_step_no_optimizer",
                "precision_policy": "amp_off_fp32",
                "compile_scope": "train_step_no_optimizer",
            },
        ])


@pytest.mark.parametrize(
    ("optimize_ddp", "compiled_autograd", "match"),
    [
        ("python_reducer", False, "requires compiled_autograd=true"),
        ("no_optimization", True, "requires compiled_autograd=false"),
    ],
)
def test_runtime_policies_reject_mode_compiled_autograd_conflicts(
    optimize_ddp: str,
    compiled_autograd: bool,  # noqa: FBT001
    match: str,
) -> None:
    """The executor rejects the two mode/autograd pairings PyTorch forbids.

    These are compatibility-policy guards: admitting either pair can disable DDP
    synchronization or fail at runtime. Removing either validation branch makes its
    parametrized error case stop raising.
    """
    with pytest.raises(ValueError, match=match):
        _runtime_policies([
            {
                "runtime_policy_id": "invalid_mode_pair",
                "precision_policy": "amp_off_fp32",
                "compile_scope": COMPILE_STEP,
                "optimize_ddp": optimize_ddp,
                "compiled_autograd": compiled_autograd,
            },
        ])


def test_runtime_policies_keep_ddp_optimizer_compiled_autograd_measurable() -> None:
    """DDPOptimizer plus compiled autograd remains a measurable, not proven, option.

    Current PyTorch does not define this pair as a permanent incompatibility, so the
    parser must preserve the configured measured candidate. Adding an inferred ban or
    silently forcing the flag off breaks the exact parsed-value assertion.
    """
    (policy,) = _runtime_policies([
        {
            "runtime_policy_id": "ddp_optimizer_compiled_autograd_probe",
            "precision_policy": "amp_off_fp32",
            "compile_scope": COMPILE_STEP,
            "optimize_ddp": "ddp_optimizer",
            "compiled_autograd": True,
        },
    ])

    assert policy.compiled_autograd is True


def test_runtime_policies_admit_feature_detected_reducer_without_forward() -> None:
    """The installed runtime's backward-only reducer mode remains measurable.

    PyTorch 2.13 exposes this experimental mode in its optimize-DDP registry. Omitting
    it would silently shrink the dual-T4 overlap search before execution can decide
    whether it is useful for this model.
    """
    (policy,) = _runtime_policies([
        {
            "runtime_policy_id": "python_reducer_without_forward_fp16",
            "precision_policy": "amp_conservative",
            "compile_scope": COMPILE_STEP,
            "optimize_ddp": "python_reducer_without_compiled_forward",
            "compiled_autograd": True,
        },
    ])

    assert policy.optimize_ddp == "python_reducer_without_compiled_forward"
    assert policy.compiled_autograd is True


def test_runtime_policies_admit_unknown_installed_mode_only_as_diagnostic() -> None:
    """A newly discovered torch mode can be observed but never silently promoted."""
    (policy,) = _runtime_policies([
        {
            "runtime_policy_id": "future_overlap_probe",
            "precision_policy": "amp_conservative",
            "compile_scope": COMPILE_STEP,
            "optimize_ddp": "future_overlap_mode",
            "compiled_autograd": True,
            "diagnostic_only": True,
        },
    ])

    assert policy.optimize_ddp == "future_overlap_mode"
    assert policy.diagnostic_only is True

    with pytest.raises(ValueError, match="Compiled runtime policies must name"):
        _runtime_policies([
            {
                "runtime_policy_id": "future_overlap_unproven",
                "precision_policy": "amp_conservative",
                "compile_scope": COMPILE_STEP,
                "optimize_ddp": "future_overlap_mode",
                "compiled_autograd": True,
            },
        ])


def test_diagnostic_fsq_row_is_never_in_selection_scope() -> None:
    """A successful diagnostic observation cannot become the runtime winner."""
    settings = replace(
        _payload_settings(),
        efficiency_policies=(
            _EfficiencyPolicySelection(
                runtime_policy_id=_DIAGNOSTIC_FSQ_POLICY_ID,
                precision_policy="amp_conservative",
                compile_scope=COMPILE_STEP,
                diagnostic_only=True,
            ),
        ),
    )
    row = _runtime_row(
        accelerator_mode="dual_t4_ddp",
        per_device_batch_size=12,
        precision_policy="amp_conservative",
        compile_scope=COMPILE_STEP,
        corruption_strategy="indexed_masked",
        world_size=EXPECTED_DUAL_WORLD_SIZE,
        samples_sec=10_000.0,
        runtime_policy_id=_DIAGNOSTIC_FSQ_POLICY_ID,
    )

    assert not _selection_candidate_scope_matches(settings=settings, row=row)


def test_compile_ddp_model_if_requested_leaves_whole_step_uncompiled() -> None:
    """Whole-step rows pass the model through uncompiled; model-forward rows compile it.

    The whole-step recipe compiles the train-*step* closure
    (``_build_compiled_ddp_step``), not the model object, so
    ``_compile_ddp_model_if_requested`` must return a step row's model untouched --
    compiling here would wrap the model a second time. The model-forward case proves the
    pass-through is scope-specific rather than an unconditional no-op: a mutation that
    also routed ``COMPILE_STEP`` into ``torch.compile`` would return the sentinel
    instead of the model.
    """
    model = object()
    compiled_sentinel = object()

    class _StubTorch:
        @staticmethod
        def compile(module: object, *, dynamic: bool) -> object:
            del module, dynamic
            return compiled_sentinel

    settings = _pretest_settings(
        resolve_json_config(CONFIG_PATH),
        data_root_override=None,
    )

    def _row_spec_for(compile_scope: str) -> RowSpec:
        return _executor_row_spec(
            settings=settings,
            accelerator_mode="dual_t4_ddp",
            batch_size=12,
            corruption_strategy="indexed_masked",
            candidate_role="selected_runtime_efficiency_followup",
            policy=_ExecutorRuntimePolicy(
                runtime_policy_id=f"compile_{compile_scope}_fp32_channels_last",
                precision_policy="amp_off_fp32",
                compile_scope=compile_scope,
            ),
        )

    step_result = _compile_ddp_model_if_requested(
        torch_module=_StubTorch(),
        model=model,
        row_spec=_row_spec_for(COMPILE_STEP),
    )
    forward_result = _compile_ddp_model_if_requested(
        torch_module=_StubTorch(),
        model=model,
        row_spec=_row_spec_for(COMPILE_MODEL_FORWARD),
    )

    assert step_result is model
    assert forward_result is compiled_sentinel


def test_step_compile_backend_matches_selected_runtime_payload() -> None:
    """The executor's compiled-step backend equals the plan the generator emits (S14b).

    ``_build_compiled_ddp_step`` compiles with ``_STEP_COMPILE_BACKEND``, and the
    generator derives the consumed plan's backend from the winner row's compile scope
    (``_selected_runtime_payload``: any compiled scope -> ``"inductor"``). The measured
    recipe only transfers to the real run if these agree, so a mutation of either
    constant is caught here.
    """
    step_row = _runtime_row(
        accelerator_mode="dual_t4_ddp",
        per_device_batch_size=12,
        precision_policy="amp_off_fp32",
        compile_scope=COMPILE_STEP,
        corruption_strategy="indexed_masked",
        world_size=EXPECTED_DUAL_WORLD_SIZE,
        samples_sec=400.0,
        runtime_policy_id="compile_step_fp32_channels_last",
    )
    dataloader_rows = tuple(_dataloader_rows((step_row,)))

    payload = _selected_runtime_payload(
        settings=_payload_settings(),
        selected_row=step_row,
        dataloader_rows=dataloader_rows,
        artifact_hashes={},
    )

    torch_compile = cast("dict[str, object]", payload["torch_compile"])
    assert torch_compile["scope"] == COMPILE_STEP
    assert torch_compile["backend"] == _STEP_COMPILE_BACKEND
    assert _STEP_COMPILE_BACKEND == "inductor"


def test_build_compiled_ddp_step_wires_amp_precision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The compiled-step builder passes AMP dtype/enabled state into the real closure.

    The fp16 compiled candidate must measure the same autocast recipe the runner uses;
    exact sentinels and float16 wiring are expected, so rejecting AMP or hardcoding
    fp32 must fail without requiring CUDA construction.
    """
    captured: dict[str, object] = {}
    ddp_model = object()
    optimizer = object()
    step_fn = object()
    compiled_step_fn = object()

    def fake_wrap_fastpath_ddp(model: object, **kwargs: object) -> object:
        captured["wrapped_model"] = model
        captured["ddp_kwargs"] = kwargs
        return ddp_model

    def fake_build_fastpath_optimizer(model: object, *, config: object) -> object:
        captured["optimizer_model"] = model
        captured["optimizer_config"] = config
        return optimizer

    def fake_model_requires_buffer_broadcast(_model: object) -> bool:
        return False

    class FakeCorruptor:
        def __init__(self, profile: object) -> None:
            captured["profile"] = profile

        def to(self, *, device: object) -> FakeCorruptor:
            captured["corruptor_device"] = device
            return self

    def fake_make_fastpath_step_fn(
        model: object,
        corruptor: object,
        *,
        ssim_weight: float,
        autocast_dtype: torch.dtype,
        autocast_enabled: bool,
    ) -> object:
        captured["step_model"] = model
        captured["corruptor"] = corruptor
        captured["ssim_weight"] = ssim_weight
        captured["autocast_dtype"] = autocast_dtype
        captured["autocast_enabled"] = autocast_enabled
        return step_fn

    class FakeTorch:
        @staticmethod
        def compile(module: object, *, dynamic: bool, backend: str) -> object:
            captured["compiled_module"] = module
            captured["compile_dynamic"] = dynamic
            captured["compile_backend"] = backend
            return compiled_step_fn

    monkeypatch.setattr(
        "eqvae.training.fastpath_recipe.wrap_fastpath_ddp",
        fake_wrap_fastpath_ddp,
    )
    monkeypatch.setattr(
        "eqvae.training.fastpath_recipe.model_requires_buffer_broadcast",
        fake_model_requires_buffer_broadcast,
    )
    monkeypatch.setattr(
        "eqvae.training.fastpath_recipe.build_fastpath_optimizer",
        fake_build_fastpath_optimizer,
    )
    monkeypatch.setattr(
        "eqvae.corruption.inline_stain.InlineStainCorruptor",
        FakeCorruptor,
    )
    monkeypatch.setattr(
        "eqvae.training.fastpath_step.make_fastpath_step_fn",
        fake_make_fastpath_step_fn,
    )
    settings = _pretest_settings(
        resolve_json_config(CONFIG_PATH),
        data_root_override=None,
    )
    amp_step_spec = _executor_row_spec(
        settings=settings,
        accelerator_mode="dual_t4_ddp",
        batch_size=12,
        corruption_strategy="indexed_masked",
        candidate_role="selected_runtime_efficiency_followup",
        policy=_ExecutorRuntimePolicy(
            runtime_policy_id="compile_step_amp_conservative",
            precision_policy="amp_conservative",
            compile_scope=COMPILE_STEP,
            autocast_dtype="float16",
            grad_scaler_enabled=True,
        ),
    )
    raw_model = object()
    device = object()
    profile = object()

    built = _build_compiled_ddp_step(
        raw_model=raw_model,
        local_rank=0,
        device=device,
        profile=profile,
        settings=settings,
        row_spec=amp_step_spec,
        torch_module=FakeTorch(),
    )

    assert built == (ddp_model, optimizer, step_fn, compiled_step_fn)
    assert captured["wrapped_model"] is raw_model
    assert captured["optimizer_model"] is raw_model
    assert captured["step_model"] is ddp_model
    assert captured["autocast_dtype"] == torch.float16
    assert captured["autocast_enabled"] is True
    assert captured["compiled_module"] is step_fn
    assert captured["compile_dynamic"] is False
    assert captured["compile_backend"] == _STEP_COMPILE_BACKEND


def test_build_compiled_ddp_step_rejects_static_graph() -> None:
    """The compiled-step probe fails closed on ``ddp_static_graph=True`` (S14b).

    A step row runs its eager numerical-proof backward before compiled settle on the
    same DDP module; ``static_graph=True`` would lock that eager backward structure and
    then encounter the differently structured compiled backward. The measured winner
    keeps ``static_graph=False``, so this guards a silently divergent (or crashing)
    measurement. The guard runs before any DDP/CUDA construction, so the check is
    CPU-safe.
    """
    settings = _pretest_settings(
        resolve_json_config(CONFIG_PATH),
        data_root_override=None,
    )
    static_graph_step_spec = _executor_row_spec(
        settings=settings,
        accelerator_mode="dual_t4_ddp",
        batch_size=12,
        corruption_strategy="indexed_masked",
        candidate_role="selected_runtime_efficiency_followup",
        policy=_ExecutorRuntimePolicy(
            runtime_policy_id="compile_step_static_graph",
            precision_policy="amp_off_fp32",
            compile_scope=COMPILE_STEP,
            ddp_static_graph=True,
        ),
    )

    with pytest.raises(ValueError, match="ddp_static_graph must be False"):
        _build_compiled_ddp_step(
            raw_model=object(),
            local_rank=0,
            device=object(),
            profile=object(),
            settings=settings,
            row_spec=static_graph_step_spec,
            torch_module=object(),
        )


def test_compiled_execution_proof_checks_health_and_rank_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The untimed proof checks a real compiled update without eager bit parity."""
    calls: list[str] = []
    sync_world_sizes: list[int] = []
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    initial_weight = model.weight.detach().clone()

    class Batch:
        images_uint8 = torch.zeros((1, 3, 8, 8), dtype=torch.uint8)

    def step_output(x_uint8: torch.Tensor) -> FastpathStepOutput:
        base = model.weight.sum()
        value = base + x_uint8.float().mean()
        detached = value.detach()
        return FastpathStepOutput(
            loss=value,
            recon_loss=detached,
            l1_loss=detached,
            ssim_loss=detached,
            ssim_metric=detached,
            kl_loss=detached,
            reconstruction=x_uint8.float(),
            logvar_clamp_count=torch.tensor(0),
            recon_output_rms=detached,
            x_hat_min=detached,
            x_hat_max=detached,
            frac_x_hat_lt_minus1=detached,
            frac_x_hat_gt_1=detached,
        )

    def eager_step(
        x_uint8: torch.Tensor,
        eps: torch.Tensor,
        beta: torch.Tensor,
    ) -> FastpathStepOutput:
        del x_uint8, eps, beta
        message = "eager step must not run"
        raise AssertionError(message)

    def compiled_step(
        x_uint8: torch.Tensor,
        eps: torch.Tensor,
        beta: torch.Tensor,
    ) -> FastpathStepOutput:
        del eps, beta
        calls.append("compiled")
        return step_output(x_uint8)

    def fake_sync_guard(observed_model: object, *, world_size: int) -> None:
        assert observed_model is model
        sync_world_sizes.append(world_size)

    def fake_beta_for_step(
        *,
        optimizer_step_index: int,
        max_optimizer_steps: int,
        target_beta: float,
        warmup_fraction: float,
    ) -> float:
        del optimizer_step_index, max_optimizer_steps, target_beta, warmup_fraction
        return 0.5

    monkeypatch.setattr(
        "eqvae.training.ddp_sync_guard.assert_ddp_parameters_exactly_in_sync",
        fake_sync_guard,
    )
    settings = _pretest_settings(
        resolve_json_config(CONFIG_PATH),
        data_root_override=None,
    )

    def run_proof() -> JsonObject:
        return _run_compiled_ddp_execution_proof(
            iterator=iter([Batch()]),
            eager_step_fn=eager_step,
            compiled_step_fn=compiled_step,
            optimizer=optimizer,
            scaler=object(),
            model=model,
            raw_model=model,
            device=torch.device("cpu"),
            settings=settings,
            step_index=0,
            row_spec=_dual_step_row_spec(),
            latent_channels=1,
            beta_for_step_fn=fake_beta_for_step,
            torch_module=torch,
            dist_module=_FakeDistModule(),
        )

    proof = run_proof()

    assert proof["status"] == PASS_STATUS
    assert proof["outputs_finite"] is True
    assert proof["parameter_update_finite_nonzero"] is True
    assert calls == ["compiled"]
    assert sync_world_sizes == [EXPECTED_DUAL_WORLD_SIZE]
    assert not torch.equal(model.weight.detach(), initial_weight)


def test_compiled_update_parity_compares_deltas_not_absolute_weights() -> None:
    """A 10% wrong small update fails even when absolute post-weights look close.

    With an O(1) weight and O(1e-3) optimizer update, the looser output tolerance
    accepts the two post-weights. Delta-relative comparison must reject them, catching
    a wrong-but-rank-synchronized compiled gradient.
    """
    initial = torch.tensor([0.5])
    eager_updated = torch.tensor([0.499])
    compiled_updated = torch.tensor([0.4989])

    assert torch.allclose(
        eager_updated,
        compiled_updated,
        rtol=pretest.NUMERICAL_REL_THRESHOLD,
        atol=pretest.NUMERICAL_ABS_THRESHOLD,
    )
    close, max_abs_delta = _parameter_update_parity(
        initial_parameters=(initial,),
        eager_updated_parameters=(eager_updated,),
        compiled_updated_parameters=(compiled_updated,),
        torch_module=torch,
    )

    assert close is False
    assert max_abs_delta > 0.0


class _FakeCudaModule:
    @staticmethod
    def current_device() -> int:
        return 0

    @staticmethod
    def get_device_name(index: int) -> str:
        del index
        return "Tesla T4"


class _FakeTorchModule:
    cuda = _FakeCudaModule()


class _FakeDistModule:
    @staticmethod
    def get_world_size() -> int:
        return EXPECTED_DUAL_WORLD_SIZE


def _dual_accelerator() -> JsonObject:
    return cast(
        "JsonObject",
        {
            "visible_device_count": EXPECTED_DUAL_WORLD_SIZE,
            "cuda_device_count": EXPECTED_DUAL_WORLD_SIZE,
            "gpu_names": ["Tesla T4", "Tesla T4"],
        },
    )


def _dual_step_row_spec() -> RowSpec:
    settings = _pretest_settings(
        resolve_json_config(CONFIG_PATH),
        data_root_override=None,
    )
    return _executor_row_spec(
        settings=settings,
        accelerator_mode="dual_t4_ddp",
        batch_size=48,
        corruption_strategy="indexed_masked",
        candidate_role="selected_runtime_efficiency_followup",
        policy=_ExecutorRuntimePolicy(
            runtime_policy_id="compile_step_fp32_channels_last",
            precision_policy="amp_off_fp32",
            compile_scope=COMPILE_STEP,
        ),
    )


def test_vram_headroom_requires_a_successful_optimizer_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated AMP skips fail before a headroom read can certify feasibility.

    A skipped GradScaler step does not allocate fused-optimizer state, so zero
    successful updates is a specific non-OOM benchmark failure. The headroom spy must
    remain untouched; deleting the guard or moving it after the read fails the test.
    """
    headroom_reads = 0

    def fake_probe_headroom(device: object) -> int:
        nonlocal headroom_reads
        del device
        headroom_reads += 1
        return 123

    monkeypatch.setattr(
        runtime_selection_executor,
        "probe_headroom_bytes",
        fake_probe_headroom,
    )

    with pytest.raises(
        _FeasibilityProbeAmpStepsSkippedError,
        match=_VRAM_FEASIBILITY_AMP_SKIPS_FAILURE_KIND,
    ):
        _probe_headroom_after_successful_optimizer_update(
            successful_optimizer_updates=0,
            device=object(),
        )

    assert headroom_reads == 0


def test_amp_proof_retries_calibration_skips_without_emitting_them() -> None:
    """Initial scale backoffs retry the same proof until one successful update.

    Two skipped attempts followed by success must return only the successful payload
    while preserving calibration diagnostics. Counting the skipped payload as linked
    selection evidence or failing immediately breaks the exact result.
    """
    skip_sequence = (True, True, False)
    attempts = iter(skip_sequence)

    def run_attempt() -> JsonObject:
        return cast("JsonObject", {"amp_step_skipped": next(attempts)})

    proof, attempt_count, skipped_count = _run_until_successful_amp_proof(
        run_attempt=run_attempt,
        fixed_batch_index=0,
    )

    assert proof["amp_step_skipped"] is False
    assert attempt_count == len(skip_sequence)
    assert skipped_count == sum(skip_sequence)


def test_amp_accounting_requires_timing_success_and_gates_measured_only() -> None:
    """Proof success cannot certify the distinct timing scaler or pollute selection.

    Five successful proof updates coexist with an all-skipped timing calibration and
    one measured skip. The derived timing success must remain zero and selection must
    see exactly the measured skip, never proof/timing calibration counts.
    """
    accounting = _amp_phase_accounting(
        proof_calibration_step_count=7,
        proof_calibration_skipped_count=2,
        timing_calibration_step_count=3,
        timing_calibration_skipped_count=3,
        measured_amp_step_skipped_count=1,
    )

    assert accounting.timing_successful_optimizer_update_count == 0
    assert accounting.selection_amp_step_skipped_count == 1
    with pytest.raises(RuntimeError, match="no successful optimizer update"):
        pretest._require_successful_amp_calibration_update(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
            grad_scaler_enabled=True,
            calibration_step_count=accounting.timing_calibration_step_count,
            successful_optimizer_update_count=(
                accounting.timing_successful_optimizer_update_count
            ),
        )


def test_feasibility_probe_counts_only_non_skipped_optimizer_updates() -> None:
    """The bounded probe derives success from each GradScaler skip result.

    A mixed skip sequence must report exactly one successful optimizer update. This
    catches discarding the helper result, inverting skip polarity, or assuming the
    bounded probe always allocates optimizer state immediately.
    """
    successful_updates = _successful_feasibility_optimizer_updates(
        amp_step_skips=(True, False),
    )

    assert successful_updates == 1


def test_vram_headroom_reads_after_a_successful_optimizer_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One successful update is sufficient to make the optimizer-state probe honest.

    The returned measured value and one exact spy call prove the positive branch still
    reads physical headroom after the allocation guard; making the guard unconditional
    or substituting a constant fails this derived behavior.
    """
    devices: list[object] = []
    device = object()
    expected_headroom = 456

    def fake_probe_headroom(observed_device: object) -> int:
        devices.append(observed_device)
        return expected_headroom

    monkeypatch.setattr(
        runtime_selection_executor,
        "probe_headroom_bytes",
        fake_probe_headroom,
    )

    headroom = _probe_headroom_after_successful_optimizer_update(
        successful_optimizer_updates=1,
        device=device,
    )

    assert headroom == expected_headroom
    assert devices == [device]


@pytest.mark.parametrize(
    ("error", "expected_kind", "expected_oom"),
    [
        (
            RuntimeError("CUDA out of memory while allocating tensor"),
            _DDP_RUNTIME_OOM_FAILURE_KIND,
            True,
        ),
        (
            _FeasibilityProbeAmpStepsSkippedError(
                _VRAM_FEASIBILITY_AMP_SKIPS_FAILURE_KIND,
            ),
            _VRAM_FEASIBILITY_AMP_SKIPS_FAILURE_KIND,
            False,
        ),
    ],
)
def test_ddp_rank_failure_payload_preserves_specific_failure_semantics(
    error: BaseException,
    expected_kind: str,
    *,
    expected_oom: bool,
) -> None:
    """Child payloads distinguish runtime OOM from bounded AMP-probe skips.

    Both errors terminate torchrun, but only actual allocation failure is ``oom=true``;
    exact kinds let the parent recover the cause from one surviving rank payload.
    Collapsing either branch to ``ddp_rank_RuntimeError`` fails its parameter case.
    """
    payload = _ddp_rank_failure_payload(
        rank=0,
        local_rank=0,
        row_id=_dual_step_row_spec().row_id,
        error=error,
        torch_module=_FakeTorchModule(),
        dist_module=_FakeDistModule(),
    )

    assert payload["failure_kind"] == expected_kind
    assert payload["oom"] is expected_oom


def test_nonzero_torchrun_preserves_available_rank_oom_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The parent recovers clean OOM evidence even when torchrun exits nonzero.

    A single rank writes the classified payload before torchrun kills its peer. The
    resulting row must retain that exact kind and ``oom=true`` rather than degrade to
    generic ``torchrun_failed``; requiring both rank files would also fail this case.
    """
    settings = _pretest_settings(
        resolve_json_config(CONFIG_PATH),
        data_root_override=None,
    )

    def fake_accelerator_failure(*, row_spec: object, accelerator: object) -> None:
        del row_spec, accelerator

    class _FailedProcess:
        returncode = 1
        stderr = "torchrun child failed"
        stdout = ""

    def fake_run(  # noqa: PLR0913
        command: object,
        *,
        cwd: object,
        env: Mapping[str, str],
        capture_output: bool,
        text: bool,
        check: bool,
        timeout: int,
    ) -> _FailedProcess:
        del command, cwd, capture_output, text, check, timeout
        rank_dir = Path(env["EQVAE_RUNTIME_SELECTION_RANK_DIR"])
        write_json(
            rank_dir / "rank_0.json",
            _ddp_rank_failure_payload(
                rank=0,
                local_rank=0,
                row_id=_dual_step_row_spec().row_id,
                error=RuntimeError("CUDA out of memory in compiled backward"),
                torch_module=_FakeTorchModule(),
                dist_module=_FakeDistModule(),
            ),
        )
        return _FailedProcess()

    monkeypatch.setattr(
        runtime_selection_executor.pretest,
        "_accelerator_observation",
        _dual_accelerator,
    )
    monkeypatch.setattr(
        runtime_selection_executor.pretest,
        "_accelerator_failure",
        fake_accelerator_failure,
    )
    monkeypatch.setattr(runtime_selection_executor.subprocess, "run", fake_run)

    result = _run_dual_row(
        request=RuntimeSelectionExecutionRequest(
            config_path=CONFIG_PATH,
            output_dir=tmp_path,
        ),
        settings=settings,
        row_spec=_dual_step_row_spec(),
        proof_reference_per_device_batch_size=_PROOF_REFERENCE_BATCH_SIZE,
    )

    assert result.returncode == 1
    assert result.failure_kind == _DDP_RUNTIME_OOM_FAILURE_KIND
    assert result.row["failure_kind"] == _DDP_RUNTIME_OOM_FAILURE_KIND
    assert result.row["oom"] == "true"
    assert result.rank_payloads == ()


def test_vram_infeasible_rank_payload_is_a_clean_oom_verdict() -> None:
    """The screen's infeasible payload is a non-pass, oom=true, parseable rank record.

    Both ranks write this then exit 0, so ``_run_dual_row`` parses it instead of
    discarding a non-zero child; it must carry ``status != pass`` (never selectable),
    ``oom=True`` (the clean verdict), and the fields the row parser reads
    (``rank``/``failure_kind``). CPU-safe via fake torch/dist modules.
    """
    payload = _vram_infeasible_rank_payload(
        rank=1,
        local_rank=1,
        row_id="dual_t4_ddp__bs48__amp_off_fp32__compile_step",
        torch_module=_FakeTorchModule(),
        dist_module=_FakeDistModule(),
    )

    assert payload["status"] != PASS_STATUS
    assert payload["oom"] is True
    assert payload["failure_kind"] == _VRAM_INFEASIBLE_FAILURE_KIND
    assert payload["rank"] == 1
    assert payload["world_size"] == EXPECTED_DUAL_WORLD_SIZE


def test_dual_row_from_infeasible_payloads_stamps_oom_true() -> None:
    """An infeasible-screen dual row reads oom=true, not an anonymous failure (S14c).

    Feeds the exact payloads ``_vram_infeasible_rank_payload`` writes through
    ``_dual_row_from_rank_payloads`` (the producer->consumer path a returncode-0 child
    exercises). The row must be non-pass with the oom cell set, so the selector skips it
    as a clean "does not fit" verdict. Dropping the ``oom=oom`` propagation makes the
    ``oom == 'true'`` assertion fail.
    """
    settings = _pretest_settings(
        resolve_json_config(CONFIG_PATH),
        data_root_override=None,
    )
    payloads = [
        _vram_infeasible_rank_payload(
            rank=rank,
            local_rank=rank,
            row_id="dual_t4_ddp__bs48__amp_off_fp32__compile_step",
            torch_module=_FakeTorchModule(),
            dist_module=_FakeDistModule(),
        )
        for rank in range(EXPECTED_DUAL_WORLD_SIZE)
    ]

    row = _dual_row_from_rank_payloads(
        settings=settings,
        row_spec=_dual_step_row_spec(),
        accelerator=_dual_accelerator(),
        rank_payloads=payloads,
    )

    assert row["status"] != PASS_STATUS
    assert row["oom"] == "true"
    assert row["failure_kind"] == _VRAM_INFEASIBLE_FAILURE_KIND


def test_dual_row_from_non_oom_failure_keeps_oom_false() -> None:
    """A non-oom rank failure leaves oom=false, so the flag is not a false positive.

    A generic crash payload (no ``oom`` key) must NOT stamp the oom cell, or every dual
    failure would masquerade as an infeasible batch. Guards the ``any(bool(...))``
    detection against over-broad matching.
    """
    settings = _pretest_settings(
        resolve_json_config(CONFIG_PATH),
        data_root_override=None,
    )
    payloads = [
        cast(
            "JsonObject",
            {
                "status": "fail",
                "rank": rank,
                "failure_kind": "ddp_rank_RuntimeError",
            },
        )
        for rank in range(EXPECTED_DUAL_WORLD_SIZE)
    ]

    row = _dual_row_from_rank_payloads(
        settings=settings,
        row_spec=_dual_step_row_spec(),
        accelerator=_dual_accelerator(),
        rank_payloads=payloads,
    )

    assert row["status"] != PASS_STATUS
    assert row["oom"] == "false"
    assert row["failure_kind"] == "ddp_rank_RuntimeError"


@pytest.mark.parametrize("unavailable_rank", [0, 1])
def test_compiled_dual_row_fails_when_one_dynamo_counter_is_unavailable(
    unavailable_rank: int,
) -> None:
    """A compiled row without Dynamo telemetry cannot claim measured zero instability.

    This fail-closed policy prevents an unobservable compile from becoming a Kaggle
    winner. Treating missing counters as derived zeros makes the row pass instead of
    emitting the exact unavailable-source failure.
    """
    settings = _pretest_settings(
        resolve_json_config(CONFIG_PATH),
        data_root_override=None,
    )
    payloads = [
        cast(
            "JsonObject",
            {
                "status": PASS_STATUS,
                "rank": rank,
                "dynamo_counter_source_available": rank != unavailable_rank,
            },
        )
        for rank in range(EXPECTED_DUAL_WORLD_SIZE)
    ]

    row = _dual_row_from_rank_payloads(
        settings=settings,
        row_spec=_dual_step_row_spec(),
        accelerator=_dual_accelerator(),
        rank_payloads=payloads,
    )

    assert row["status"] != PASS_STATUS
    assert row["failure_kind"] == "compiled_dynamo_counter_source_unavailable"


def test_compiled_dual_row_fails_when_counter_schema_is_unavailable() -> None:
    """A counter mapping without expected schema cannot manufacture zero telemetry."""
    settings = _pretest_settings(
        resolve_json_config(CONFIG_PATH),
        data_root_override=None,
    )
    payloads = [
        cast(
            "JsonObject",
            {
                "status": PASS_STATUS,
                "rank": rank,
                "dynamo_counter_source_available": True,
                "dynamo_counter_schema_available": rank == 0,
            },
        )
        for rank in range(EXPECTED_DUAL_WORLD_SIZE)
    ]

    row = _dual_row_from_rank_payloads(
        settings=settings,
        row_spec=_dual_step_row_spec(),
        accelerator=_dual_accelerator(),
        rank_payloads=payloads,
    )

    assert row["status"] != PASS_STATUS
    assert row["failure_kind"] == "compiled_dynamo_counter_schema_unavailable"


@pytest.mark.parametrize("missing_proof_rank", [0, 1])
def test_compiled_dual_row_requires_real_execution_proof_from_every_rank(
    missing_proof_rank: int,
) -> None:
    """A timed compiled row cannot pass with eager-only proof payloads.

    One rank carries complete proof and the other omits it. The aggregator must fail
    before reading throughput fields in either rank order; replacing ``all`` with
    ``any`` would recreate the eager-only audit defect and fail one parameter case.
    """
    settings = _pretest_settings(
        resolve_json_config(CONFIG_PATH),
        data_root_override=None,
    )
    payloads = [
        cast(
            "JsonObject",
            {
                "status": PASS_STATUS,
                "rank": rank,
                "dynamo_counter_source_available": True,
                "dynamo_counter_schema_available": True,
                **(
                    {}
                    if rank == missing_proof_rank
                    else {
                        "compiled_execution_proof": {
                            "status": PASS_STATUS,
                            "outputs_finite": True,
                            "parameter_update_finite_nonzero": True,
                            "successful_optimizer_update_count": 1,
                            "ddp_parameters_in_sync": True,
                        },
                    }
                ),
            },
        )
        for rank in range(EXPECTED_DUAL_WORLD_SIZE)
    ]

    row = _dual_row_from_rank_payloads(
        settings=settings,
        row_spec=_dual_step_row_spec(),
        accelerator=_dual_accelerator(),
        rank_payloads=payloads,
    )

    assert row["status"] != PASS_STATUS
    assert row["failure_kind"] == _COMPILED_EXECUTION_PROOF_FAILURE_KIND


def test_failure_row_stamps_oom_only_when_requested() -> None:
    """``_failure_row`` sets the oom cell from its flag, overriding the base false.

    The base selection row hardcodes oom=false; a VRAM-infeasible failure must override
    it to true while every other failure stays false. Re-hardcoding the cell to a
    literal breaks one of the two assertions.
    """
    row_spec = _dual_step_row_spec()
    settings = _pretest_settings(
        resolve_json_config(CONFIG_PATH),
        data_root_override=None,
    )
    infeasible = _failure_row(
        settings=settings,
        row_spec=row_spec,
        accelerator=_dual_accelerator(),
        status="fail",
        failure_kind=_VRAM_INFEASIBLE_FAILURE_KIND,
        failure_message=_VRAM_INFEASIBLE_FAILURE_KIND,
        oom=True,
    )
    crashed = _failure_row(
        settings=settings,
        row_spec=row_spec,
        accelerator=_dual_accelerator(),
        status="fail",
        failure_kind="torchrun_failed",
        failure_message="boom",
    )

    assert infeasible["oom"] == "true"
    assert crashed["oom"] == "false"


def test_oom_row_is_never_a_selection_candidate() -> None:
    """A row flagged oom=true is excluded from selection even if otherwise passing.

    An infeasible batch must never win the runtime, defensively even if a future path
    leaves it status=pass with a positive throughput (Spec 0011 S14c). Dropping the oom
    gate in ``_runtime_row_candidate_pass`` lets the flipped row pass, failing the
    second assertion.
    """
    row = _runtime_row(
        accelerator_mode="dual_t4_ddp",
        per_device_batch_size=12,
        precision_policy="amp_off_fp32",
        compile_scope="none",
        corruption_strategy="indexed_masked",
        world_size=EXPECTED_DUAL_WORLD_SIZE,
        samples_sec=100.0,
    )

    assert _runtime_row_candidate_pass(row)
    row["oom"] = "true"
    assert not _runtime_row_candidate_pass(row)


def test_grid_retains_original_compiled_step_controls_at_bs48() -> None:
    """The grid emits fp32 and fp16 compiled bs48 rows against proof batch 12.

    A2 compares both precision recipes at the large timed batch while reusing the
    measured fp32 branchless batch-12 proof; exact policy membership and batch values
    are expected, so dropping fp16 or reverting same-batch proof must fail.
    """
    resolved = resolve_json_config(CONFIG_PATH)
    settings = _pretest_settings(resolved, data_root_override=None)
    stage = _selection_stage_settings(resolved.effective_config)

    assert _WINNER_BATCH_SIZE in stage.efficiency_batch_sizes
    step_policies = [
        policy
        for policy in stage.efficiency_policies
        if policy.runtime_policy_id in _ORIGINAL_COMPILED_POLICY_IDS
    ]
    assert len(step_policies) == len(_ORIGINAL_COMPILED_POLICY_IDS)
    policies_by_precision = {
        policy.precision_policy: policy for policy in step_policies
    }
    assert set(policies_by_precision) == {"amp_off_fp32", "amp_conservative"}
    for policy in policies_by_precision.values():
        assert policy.optimize_ddp == "ddp_optimizer"
        assert policy.fused_optimizer is True
        assert policy.ddp_gradient_as_bucket_view is True
        assert policy.ddp_bucket_cap_mb == _WINNER_BUCKET_CAP_MB
        assert policy.ddp_broadcast_buffers is False
        assert policy.compiled_autograd is False
        assert policy.ddp_static_graph is False
        assert policy.memory_format == "channels_last"
    assert policies_by_precision["amp_conservative"].autocast_dtype == "float16"
    assert policies_by_precision["amp_conservative"].grad_scaler_enabled is True
    assert stage.proof_reference_per_device_batch_size == _PROOF_REFERENCE_BATCH_SIZE

    specs = _dual_row_specs(settings=settings, stage=stage)
    step_bs48 = [
        spec
        for spec in specs
        if spec.compile_scope == COMPILE_STEP
        and spec.per_device_batch_size == _WINNER_BATCH_SIZE
    ]
    step_bs48 = [
        spec
        for spec in step_bs48
        if spec.runtime_policy_id in _ORIGINAL_COMPILED_POLICY_IDS
    ]
    assert len(step_bs48) == len(_ORIGINAL_COMPILED_POLICY_IDS)
    assert {spec.precision_policy for spec in step_bs48} == {
        "amp_off_fp32",
        "amp_conservative",
    }
    assert all(spec.fused_optimizer for spec in step_bs48)
    assert {spec.ddp_bucket_cap_mb for spec in step_bs48} == {
        _WINNER_BUCKET_CAP_MB,
    }
    # The relaxed eager AMP row remains same-batch gated and therefore stops at bs12.
    amp_bs48 = [
        spec
        for spec in specs
        if spec.precision_policy == "amp_scalar_gate_relaxed"
        and spec.per_device_batch_size == _WINNER_BATCH_SIZE
    ]
    assert amp_bs48 == []


def test_executor_scopes_cross_batch_proof_to_efficiency_rows() -> None:
    """Only efficiency candidates use the configured cross-batch proof control.

    This derived scoping keeps bs48 timing honest without rewriting unrelated future
    rows. Applying the proof batch globally or reverting efficiency rows to same-batch
    proof breaks one of the two exact batch assertions.
    """
    resolved = resolve_json_config(CONFIG_PATH)
    stage = _selection_stage_settings(resolved.effective_config)
    efficiency = _dual_step_row_spec()
    ordinary = replace(
        efficiency,
        candidate_role="future_non_efficiency_gate",
        per_device_batch_size=_FUTURE_NON_EFFICIENCY_BATCH_SIZE,
    )

    assert (
        _row_proof_reference_batch_size(
            row_spec=efficiency,
            stage=stage,
        )
        == _PROOF_REFERENCE_BATCH_SIZE
    )
    assert (
        _row_proof_reference_batch_size(
            row_spec=ordinary,
            stage=stage,
        )
        == _FUTURE_NON_EFFICIENCY_BATCH_SIZE
    )


def test_amp_efficiency_row_needs_an_fp32_companion_batch() -> None:
    """Only compiled AMP may reuse the configured fp32 proof at a larger batch.

    The new fp16 compiled row is expected at bs48 against proof batch 12, while the
    existing eager relaxed-AMP slice remains bs12-only; broadening or restoring the
    old universal same-batch guard must fail one exact polarity.
    """
    stage = _selection_stage_settings(
        resolve_json_config(CONFIG_PATH).effective_config,
    )
    relaxed_amp = next(
        policy
        for policy in stage.efficiency_policies
        if policy.precision_policy == "amp_scalar_gate_relaxed"
    )
    compiled_amp = next(
        policy
        for policy in stage.efficiency_policies
        if policy.precision_policy == "amp_conservative"
        and policy.compile_scope == COMPILE_STEP
    )
    companion_batch = stage.dual_batch_sizes[-1]

    assert _efficiency_row_enumerable(
        policy=relaxed_amp,
        batch_size=companion_batch,
        stage=stage,
    )
    assert not _efficiency_row_enumerable(
        policy=relaxed_amp,
        batch_size=_WINNER_BATCH_SIZE,
        stage=stage,
    )
    assert _efficiency_row_enumerable(
        policy=compiled_amp,
        batch_size=_WINNER_BATCH_SIZE,
        stage=stage,
    )


def test_oom_result_does_not_crash_dual_evidence_aggregation() -> None:
    """A returncode-0 oom result must not crash the dual evidence consumers (S14c).

    The oom skip is the first case where a returncode-0 child yields non-PASS rank
    payloads (missing ``dataloader``/``proof_step``). The PASS-status guards in
    ``_dual_dataloader_rows`` and ``_rank0_proof_steps_by_row_id`` skip it; removing
    either guard makes the corresponding consumer raise ``TypeError`` on the missing
    key.
    """
    settings = _pretest_settings(
        resolve_json_config(CONFIG_PATH),
        data_root_override=None,
    )
    oom_row = _failure_row(
        settings=settings,
        row_spec=_dual_step_row_spec(),
        accelerator=_dual_accelerator(),
        status="fail",
        failure_kind=_VRAM_INFEASIBLE_FAILURE_KIND,
        failure_message=_VRAM_INFEASIBLE_FAILURE_KIND,
        oom=True,
    )
    oom_payloads = tuple(
        _vram_infeasible_rank_payload(
            rank=rank,
            local_rank=rank,
            row_id=oom_row["row_id"],
            torch_module=_FakeTorchModule(),
            dist_module=_FakeDistModule(),
        )
        for rank in range(EXPECTED_DUAL_WORLD_SIZE)
    )
    # Directly carry the oom payloads (as a returncode-0 child would) to exercise the
    # consumer guards, not the _run_dual_row choke point that also empties them.
    oom_result = _DdpLaunchResult(
        row=oom_row,
        rank_payloads=oom_payloads,
        command_display="torchrun --standalone --nproc_per_node=2",
        returncode=0,
        failure_kind=_VRAM_INFEASIBLE_FAILURE_KIND,
        failure_message_hash="",
        reference_row_id=_row_id(
            accelerator_mode="dual_t4_ddp",
            batch_size=12,
            precision_policy="amp_off_fp32",
            compile_scope="none",
            corruption_strategy="branchless_all",
        ),
    )

    assert _dual_dataloader_rows(settings=settings, results=[oom_result]) == []
    assert isinstance(
        _dual_numerical_rows(settings=settings, results=[oom_result]),
        list,
    )
    assert isinstance(
        _dual_corruption_rows(settings=settings, results=[oom_result]),
        list,
    )


def test_runtime_selection_records_v8_shortlist_provenance(tmp_path: Path) -> None:
    """The selected-runtime path records v8 hashes without promoting v8 rows."""
    v8_dir = _write_fake_v8_artifacts(tmp_path / "v8")
    output_dir = tmp_path / "selection"

    artifacts = write_runtime_selection_benchmark(
        RuntimeSelectionBenchmarkRequest(
            config_path=CONFIG_PATH,
            output_dir=output_dir,
            v8_artifact_dir=v8_dir,
        ),
    )

    proof = _load_json(artifacts.runtime_proof)
    provenance = cast("dict[str, object]", proof["v8_provenance"])
    hashes = cast("dict[str, object]", provenance["artifact_hashes"])

    assert provenance["status"] == "pass"
    assert provenance["used_for"] == "candidate_shortlist_only"
    assert provenance["v8_artifacts_are_promotable"] is False
    assert hashes["benchmark/runtime_matrix.csv"] == _sha256_file(
        v8_dir / "benchmark" / "runtime_matrix.csv",
    )
    assert proof["status"] == "fail"
    assert proof["selected_runtime_written"] is False
    assert artifacts.selected_runtime is None
    assert not (output_dir / "benchmark" / "selected_runtime.json").exists()


def test_runtime_selection_blocks_without_dual_t4_train_step_gate(
    tmp_path: Path,
) -> None:
    """Local schema plumbing must not pass without real dual-T4 timing."""
    output_dir = tmp_path / "selection"

    artifacts = write_runtime_selection_benchmark(
        RuntimeSelectionBenchmarkRequest(
            config_path=CONFIG_PATH,
            output_dir=output_dir,
            v8_artifact_dir=_write_fake_v8_artifacts(tmp_path / "v8"),
        ),
    )

    proof = _load_json(artifacts.runtime_proof)
    dual_gate = cast("dict[str, object]", proof["dual_t4_train_step_gate"])
    decision = cast("dict[str, object]", proof["selected_runtime_write_decision"])
    blockers = cast("list[object]", decision["blockers"])
    runtime_rows = _load_csv(output_dir / "benchmark" / "runtime_matrix.csv")
    dual_rows = [
        row
        for row in runtime_rows
        if row["accelerator_mode"] == "dual_t4_ddp"
        and row["precision_policy"] == "amp_off_fp32"
        and row["compile_scope"] == "none"
    ]

    assert dual_gate["status"] == "skipped_unsupported"
    assert "missing_real_dual_t4_train_step_timing" in blockers
    assert len(dual_rows) == EXPECTED_DUAL_RUNTIME_ROWS
    assert {row["per_device_batch_size"] for row in dual_rows} == {"4", "8", "12"}
    assert {row["corruption_strategy"] for row in dual_rows} == set(
        CORRUPTION_STRATEGIES,
    )
    assert {row["status"] for row in dual_rows} == {"skipped_unsupported"}
    assert not (output_dir / "benchmark" / "selected_runtime.json").exists()


def test_runtime_selection_refuses_stale_selected_runtime_when_blocked(
    tmp_path: Path,
) -> None:
    """A blocked run fails if a stale selected_runtime.json is already present."""
    output_dir = tmp_path / "selection"
    write_json(output_dir / "benchmark" / "selected_runtime.json", {"stale": True})

    with pytest.raises(RuntimeError, match="selected_runtime"):
        write_runtime_selection_benchmark(
            RuntimeSelectionBenchmarkRequest(
                config_path=CONFIG_PATH,
                output_dir=output_dir,
                v8_artifact_dir=_write_fake_v8_artifacts(tmp_path / "v8"),
            ),
        )


def test_runtime_selection_writes_selected_runtime_after_full_local_proof(
    tmp_path: Path,
) -> None:
    """Successful injected compact follow-up evidence writes a pass payload."""
    output_dir = tmp_path / "selection"
    evidence = _passing_runtime_selection_evidence()
    _write_stain_qa(output_dir, evidence)

    artifacts = write_runtime_selection_benchmark(
        RuntimeSelectionBenchmarkRequest(
            config_path=CONFIG_PATH,
            output_dir=output_dir,
            v8_artifact_dir=_write_fake_v8_artifacts(tmp_path / "v8"),
            evidence=evidence,
        ),
    )

    assert artifacts.selected_runtime is not None
    proof = _load_json(artifacts.runtime_proof)
    selected = _load_json(artifacts.selected_runtime)
    runtime_rows = _load_csv(output_dir / "benchmark" / "runtime_matrix.csv")
    compiled_row = next(
        row for row in runtime_rows if row["compile_scope"] == "model_forward"
    )

    assert proof["status"] == "pass"
    assert proof["selection_ready"] is True
    assert proof["selected_runtime_written"] is True
    assert (
        cast("dict[str, object]", proof["dual_t4_train_step_gate"])["status"] == "pass"
    )
    assert selected["status"] == "pass"
    assert selected["benchmark_kind"] == "kaggle_runtime_selection"
    assert selected["world_size"] == EXPECTED_DUAL_WORLD_SIZE
    assert selected["nproc_per_node"] == EXPECTED_DUAL_WORLD_SIZE
    assert selected["selected_row_id"] == (
        "dual_t4_ddp__bs12__amp_scalar_gate_relaxed__compile_none__indexed_masked__"
        "policy_amp_fp16_scalar_gate_relaxed"
    )
    artifacts_payload = cast("dict[str, object]", selected["artifacts"])
    assert artifacts_payload["stain_corruptor_qa"] == (
        "benchmark/stain_corruptor_qa.json"
    )
    assert artifacts_payload["runtime_proof_sha256"] == _sha256_file(
        artifacts.runtime_proof,
    )
    assert compiled_row["status"] == "ineligible"
    assert compiled_row["failure_kind"] == (
        "compiled_rows_diagnostic_only_until_stable_settle_proof"
    )


def test_runtime_selection_allows_stable_compile_efficiency_row(
    tmp_path: Path,
) -> None:
    """A settled zero-break python-reducer row may replace the fallback baseline.

    Compiled selection needs explicit mode telemetry as well as throughput; a written
    artifact is expected, so blank-mode acceptance or universal rejection must fail.
    """
    output_dir = tmp_path / "selection"
    config_path = _write_config_with_efficiency_policies(
        tmp_path=tmp_path,
        policies=(
            {
                "runtime_policy_id": "compile_model_forward_fp32_channels_last",
                "precision_policy": "amp_off_fp32",
                "compile_scope": "model_forward",
                "optimize_ddp": "python_reducer",
                "compiled_autograd": True,
            },
        ),
    )
    compiled_row = _runtime_row(
        accelerator_mode="dual_t4_ddp",
        per_device_batch_size=12,
        precision_policy="amp_off_fp32",
        compile_scope="model_forward",
        corruption_strategy="indexed_masked",
        world_size=EXPECTED_DUAL_WORLD_SIZE,
        samples_sec=400.0,
        runtime_policy_id="compile_model_forward_fp32_channels_last",
        memory_format="channels_last",
    )
    compiled_row["optimize_ddp"] = "python_reducer"
    compiled_row["compiled_autograd"] = "true"
    runtime_rows = (
        *_passing_runtime_rows(),
        compiled_row,
    )
    evidence = _runtime_selection_evidence_from_rows(runtime_rows)
    _write_stain_qa(output_dir, evidence)

    artifacts = write_runtime_selection_benchmark(
        RuntimeSelectionBenchmarkRequest(
            config_path=config_path,
            output_dir=output_dir,
            v8_artifact_dir=_write_fake_v8_artifacts(tmp_path / "v8"),
            evidence=evidence,
        ),
    )

    assert artifacts.selected_runtime is not None
    selected = _load_json(artifacts.selected_runtime)
    assert selected["runtime_policy_id"] == "compile_model_forward_fp32_channels_last"
    assert cast("dict[str, object]", selected["torch_compile"])["enabled"] is True
    assert cast("dict[str, object]", selected["runtime_policy"])["memory_format"] == (
        "channels_last"
    )
    assert selected["full_training_launch_ready"] is False


def test_runtime_selection_excludes_amp_skip_rows_without_global_block(
    tmp_path: Path,
) -> None:
    """AMP skips block their own row, not an otherwise safe selected runtime."""
    output_dir = tmp_path / "selection"
    config_path = _write_config_with_efficiency_policies(
        tmp_path=tmp_path,
        policies=(
            {
                "runtime_policy_id": "amp_fp16_conservative",
                "precision_policy": "amp_conservative",
                "compile_scope": "none",
            },
        ),
    )
    runtime_rows = (
        *_passing_runtime_rows(),
        _runtime_row(
            accelerator_mode="dual_t4_ddp",
            per_device_batch_size=12,
            precision_policy="amp_conservative",
            compile_scope="none",
            corruption_strategy="indexed_masked",
            world_size=EXPECTED_DUAL_WORLD_SIZE,
            samples_sec=500.0,
            runtime_policy_id="amp_fp16_conservative",
            amp_step_skipped_count=2,
        ),
    )
    evidence = _runtime_selection_evidence_from_rows(runtime_rows)
    _write_stain_qa(output_dir, evidence)

    artifacts = write_runtime_selection_benchmark(
        RuntimeSelectionBenchmarkRequest(
            config_path=config_path,
            output_dir=output_dir,
            v8_artifact_dir=_write_fake_v8_artifacts(tmp_path / "v8"),
            evidence=evidence,
        ),
    )

    proof = _load_json(artifacts.runtime_proof)
    amp_policy = cast("dict[str, object]", proof["amp_followup_policy"])
    decision = cast("dict[str, object]", proof["selected_runtime_write_decision"])

    assert artifacts.selected_runtime is None
    assert amp_policy["status"] == "pass"
    assert runtime_rows[-1]["row_id"] in cast(
        "list[object]",
        amp_policy["amp_skipped_row_ids"],
    )
    assert "amp_followup_policy_not_pass" not in cast(
        "list[object]",
        decision["blockers"],
    )


def test_runtime_selection_accepts_large_amp_with_configured_reference(
    tmp_path: Path,
) -> None:
    """A bs48 fp16 compiled row may use the exact fp32 branchless bs12 proof.

    The configured proof batch is a deliberate cross-batch comparison contract;
    successful selection and exact proof metadata are expected, so same-batch AMP
    companionship or zero-break DDPOptimizer gating must fail.
    """
    output_dir = tmp_path / "selection"
    large_amp = _fp16_compiled_bs48_row()
    runtime_rows = (*_passing_runtime_rows(), large_amp)
    evidence = _runtime_selection_evidence_from_rows(runtime_rows)
    _write_stain_qa(output_dir, evidence)

    artifacts = write_runtime_selection_benchmark(
        RuntimeSelectionBenchmarkRequest(
            config_path=CONFIG_PATH,
            output_dir=output_dir,
            v8_artifact_dir=_write_fake_v8_artifacts(tmp_path / "v8"),
            evidence=evidence,
        ),
    )

    assert artifacts.selected_runtime is not None
    proof = _load_json(artifacts.runtime_proof)
    amp_policy = cast("dict[str, object]", proof["amp_followup_policy"])
    efficiency = cast("dict[str, object]", proof["efficiency_followup"])
    selected = _load_json(artifacts.selected_runtime)
    assert amp_policy["status"] == "pass"
    assert (
        efficiency["proof_reference_per_device_batch_size"]
        == _PROOF_REFERENCE_BATCH_SIZE
    )
    assert selected["selected_row_id"] == large_amp["row_id"]


@pytest.mark.parametrize(
    ("lane", "failure_prefix"),
    [
        ("numerical", "numerical_checks"),
        ("corruption", "corruption_checks"),
    ],
)
def test_runtime_selection_rejects_wrong_linked_reference_id(
    tmp_path: Path,
    lane: str,
    failure_prefix: str,
) -> None:
    """Numerical and corruption lanes must name the candidate's exact proof row.

    A same-batch bs48 reference is plausible but wrong when proof batch 12 is
    configured; exact linked failures are expected, so ignoring ``reference_row_id``
    or checking only candidate identity must fail each lane case.
    """
    output_dir = tmp_path / lane
    large_amp = _fp16_compiled_bs48_row()
    evidence = _runtime_selection_evidence_from_rows(
        (*_passing_runtime_rows(), large_amp),
    )
    wrong_reference = _row_id(
        accelerator_mode="dual_t4_ddp",
        batch_size=48,
        precision_policy="amp_off_fp32",
        compile_scope="none",
        corruption_strategy="branchless_all",
    )
    if lane == "numerical":
        evidence = replace(
            evidence,
            numerical_rows=tuple(
                {**row, "reference_row_id": wrong_reference}
                if row["candidate_row_id"] == large_amp["row_id"]
                else row
                for row in evidence.numerical_rows
            ),
        )
    else:
        evidence = replace(
            evidence,
            corruption_rows=tuple(
                {**row, "reference_row_id": wrong_reference}
                if row["candidate_row_id"] == large_amp["row_id"]
                else row
                for row in evidence.corruption_rows
            ),
        )
    _write_stain_qa(output_dir, evidence)

    artifacts = write_runtime_selection_benchmark(
        RuntimeSelectionBenchmarkRequest(
            config_path=CONFIG_PATH,
            output_dir=output_dir,
            v8_artifact_dir=_write_fake_v8_artifacts(tmp_path / f"v8-{lane}"),
            evidence=evidence,
        ),
    )

    assert artifacts.selected_runtime is None
    proof = _load_json(artifacts.runtime_proof)
    decision = cast("dict[str, object]", proof["selected_runtime_write_decision"])
    assert f"{failure_prefix}:{large_amp['row_id']}" in cast(
        "list[object]",
        decision["linked_pass_row_failures"],
    )


def test_runtime_selection_accepts_small_numerical_drift_for_faster_row(
    tmp_path: Path,
) -> None:
    """Performance-first selection accepts small finite numerical drift."""
    output_dir = tmp_path / "selection"
    config_path = _write_config_with_efficiency_policies(
        tmp_path=tmp_path,
        policies=(
            {
                "runtime_policy_id": "amp_fp16_conservative",
                "precision_policy": "amp_conservative",
                "compile_scope": "none",
            },
        ),
    )
    fast_amp = _runtime_row(
        accelerator_mode="dual_t4_ddp",
        per_device_batch_size=12,
        precision_policy="amp_conservative",
        compile_scope="none",
        corruption_strategy="indexed_masked",
        world_size=EXPECTED_DUAL_WORLD_SIZE,
        samples_sec=350.0,
        runtime_policy_id="amp_fp16_conservative",
    )
    runtime_rows = (*_passing_runtime_rows(), fast_amp)
    evidence = _runtime_selection_evidence_from_rows(runtime_rows)
    numerical_rows = tuple(
        _with_small_numerical_drift(row, candidate_row_id=fast_amp["row_id"])
        for row in evidence.numerical_rows
    )
    _write_stain_qa(output_dir, evidence)

    artifacts = write_runtime_selection_benchmark(
        RuntimeSelectionBenchmarkRequest(
            config_path=config_path,
            output_dir=output_dir,
            v8_artifact_dir=_write_fake_v8_artifacts(tmp_path / "v8"),
            evidence=RuntimeSelectionEvidence(
                runtime_rows=evidence.runtime_rows,
                dataloader_rows=evidence.dataloader_rows,
                numerical_rows=numerical_rows,
                corruption_rows=evidence.corruption_rows,
                gate_health_rows=evidence.gate_health_rows,
                gate_health_summary=evidence.gate_health_summary,
                runtime_environment=evidence.runtime_environment,
            ),
        ),
    )

    assert artifacts.selected_runtime is not None
    proof = _load_json(artifacts.runtime_proof)
    decision = cast("dict[str, object]", proof["selected_runtime_write_decision"])
    selected = _load_json(artifacts.selected_runtime)
    assert selected["runtime_policy_id"] == "amp_fp16_conservative"
    assert decision["allowed"] is True


def test_runtime_selection_can_select_relaxed_scalar_gate_amp_policy(
    tmp_path: Path,
) -> None:
    """A faster relaxed scalar-gate AMP row can replace the v5 fallback."""
    output_dir = tmp_path / "selection"
    runtime_rows = _runtime_rows_for_v5_followup(relaxed_samples_sec=350.0)
    relaxed_amp = runtime_rows[-1]
    evidence = _runtime_selection_evidence_from_rows(runtime_rows)
    numerical_rows = tuple(
        _with_small_numerical_drift(row, candidate_row_id=relaxed_amp["row_id"])
        for row in evidence.numerical_rows
    )
    _write_stain_qa(output_dir, evidence)

    artifacts = write_runtime_selection_benchmark(
        RuntimeSelectionBenchmarkRequest(
            config_path=CONFIG_PATH,
            output_dir=output_dir,
            v8_artifact_dir=_write_fake_v8_artifacts(tmp_path / "v8"),
            evidence=RuntimeSelectionEvidence(
                runtime_rows=evidence.runtime_rows,
                dataloader_rows=evidence.dataloader_rows,
                numerical_rows=numerical_rows,
                corruption_rows=evidence.corruption_rows,
                gate_health_rows=evidence.gate_health_rows,
                gate_health_summary=evidence.gate_health_summary,
                runtime_environment=evidence.runtime_environment,
            ),
        ),
    )

    assert artifacts.selected_runtime is not None
    proof = _load_json(artifacts.runtime_proof)
    selected = _load_json(artifacts.selected_runtime)
    decision = cast("dict[str, object]", proof["selected_runtime_write_decision"])

    assert decision["allowed"] is True
    assert selected["selected_row_id"] == relaxed_amp["row_id"]
    assert selected["runtime_policy_id"] == "amp_fp16_scalar_gate_relaxed"
    assert cast("dict[str, object]", selected["mixed_precision"])["policy"] == (
        "amp_scalar_gate_relaxed"
    )


def test_runtime_selection_blocks_relaxed_amp_without_gate_dtype_proof(
    tmp_path: Path,
) -> None:
    """A faster relaxed row must prove it really used relaxed gate math."""
    output_dir = tmp_path / "selection"
    runtime_rows = _runtime_rows_for_v5_followup(relaxed_samples_sec=350.0)
    relaxed_amp = runtime_rows[-1]
    evidence = _runtime_selection_evidence_from_rows(runtime_rows)
    gate_rows = tuple(
        {
            **row,
            "precision_proof_status": "",
            "gate_math_dtype": "float32",
        }
        if row["candidate_row_id"] == relaxed_amp["row_id"]
        else row
        for row in evidence.gate_health_rows
    )
    _write_stain_qa(output_dir, evidence)

    artifacts = write_runtime_selection_benchmark(
        RuntimeSelectionBenchmarkRequest(
            config_path=CONFIG_PATH,
            output_dir=output_dir,
            v8_artifact_dir=_write_fake_v8_artifacts(tmp_path / "v8"),
            evidence=RuntimeSelectionEvidence(
                runtime_rows=evidence.runtime_rows,
                dataloader_rows=evidence.dataloader_rows,
                numerical_rows=evidence.numerical_rows,
                corruption_rows=evidence.corruption_rows,
                gate_health_rows=gate_rows,
                gate_health_summary=evidence.gate_health_summary,
                runtime_environment=evidence.runtime_environment,
            ),
        ),
    )

    proof = _load_json(artifacts.runtime_proof)
    decision = cast("dict[str, object]", proof["selected_runtime_write_decision"])
    blockers = cast("list[object]", decision["blockers"])
    linked_failures = cast("list[object]", decision["linked_pass_row_failures"])
    assert artifacts.selected_runtime is None
    assert "runtime_pass_rows_linked_proof_not_pass" in blockers
    assert "selected_row_gate_health_not_pass" in blockers
    assert f"gate_health:{relaxed_amp['row_id']}" in linked_failures


def test_runtime_selection_blocks_partial_relaxed_gate_dtype_proof(
    tmp_path: Path,
) -> None:
    """Every scalar gate row for a relaxed candidate must prove relaxed math."""
    output_dir = tmp_path / "selection"
    runtime_rows = _runtime_rows_for_v5_followup(relaxed_samples_sec=350.0)
    relaxed_amp = runtime_rows[-1]
    evidence = _runtime_selection_evidence_from_rows(runtime_rows)
    relaxed_gate_rows = [
        row
        for row in evidence.gate_health_rows
        if row["candidate_row_id"] == relaxed_amp["row_id"]
    ]
    assert len(relaxed_gate_rows) == 1
    failing_second_gate = {
        **relaxed_gate_rows[0],
        "row_id": f"{relaxed_amp['row_id']}__gate__encoder_1",
        "module": "encoder.1",
        "gate_force_fp32": "true",
        "gate_math_dtype": "float32",
        "gate_tensor_dtype": "float32",
        "precision_proof_status": "pass",
    }
    _write_stain_qa(output_dir, evidence)

    artifacts = write_runtime_selection_benchmark(
        RuntimeSelectionBenchmarkRequest(
            config_path=CONFIG_PATH,
            output_dir=output_dir,
            v8_artifact_dir=_write_fake_v8_artifacts(tmp_path / "v8"),
            evidence=RuntimeSelectionEvidence(
                runtime_rows=evidence.runtime_rows,
                dataloader_rows=evidence.dataloader_rows,
                numerical_rows=evidence.numerical_rows,
                corruption_rows=evidence.corruption_rows,
                gate_health_rows=(
                    *evidence.gate_health_rows,
                    failing_second_gate,
                ),
                gate_health_summary=evidence.gate_health_summary,
                runtime_environment=evidence.runtime_environment,
            ),
        ),
    )

    proof = _load_json(artifacts.runtime_proof)
    decision = cast("dict[str, object]", proof["selected_runtime_write_decision"])
    blockers = cast("list[object]", decision["blockers"])
    linked_failures = cast("list[object]", decision["linked_pass_row_failures"])
    assert artifacts.selected_runtime is None
    assert "runtime_pass_rows_linked_proof_not_pass" in blockers
    assert "selected_row_gate_health_not_pass" in blockers
    assert f"gate_health:{relaxed_amp['row_id']}" in linked_failures


def test_runtime_selection_keeps_v5_fallback_when_relaxed_amp_is_not_material(
    tmp_path: Path,
) -> None:
    """A slower relaxed follow-up must not write a replacement artifact."""
    output_dir = tmp_path / "selection"
    runtime_rows = _runtime_rows_for_v5_followup(relaxed_samples_sec=27.5)
    evidence = _runtime_selection_evidence_from_rows(runtime_rows)
    _write_stain_qa(output_dir, evidence)

    artifacts = write_runtime_selection_benchmark(
        RuntimeSelectionBenchmarkRequest(
            config_path=CONFIG_PATH,
            output_dir=output_dir,
            v8_artifact_dir=_write_fake_v8_artifacts(tmp_path / "v8"),
            evidence=evidence,
        ),
    )

    proof = _load_json(artifacts.runtime_proof)
    decision = cast("dict[str, object]", proof["selected_runtime_write_decision"])
    efficiency = cast("dict[str, object]", proof["efficiency_followup"])
    blockers = cast("list[object]", decision["blockers"])

    assert artifacts.selected_runtime is None
    assert decision["allowed"] is False
    assert decision["selected_row_id"] == (
        "dual_t4_ddp__bs12__amp_conservative__compile_none__indexed_masked__"
        "policy_amp_fp16_conservative"
    )
    assert "selected_runtime_reuses_configured_baseline_no_replacement" in blockers
    assert efficiency["material_speedup_over_baseline"] is False
    assert not (output_dir / "benchmark" / "selected_runtime.json").exists()


def test_runtime_selection_ignores_fast_nonconfigured_row_for_v5_followup(
    tmp_path: Path,
) -> None:
    """Only the configured compact relaxed policy may replace the v5 fallback."""
    output_dir = tmp_path / "selection"
    runtime_rows = (
        *_runtime_rows_for_v5_followup(relaxed_samples_sec=27.5),
        _runtime_row(
            accelerator_mode="dual_t4_ddp",
            per_device_batch_size=12,
            precision_policy="amp_conservative",
            compile_scope="none",
            corruption_strategy="indexed_masked",
            world_size=EXPECTED_DUAL_WORLD_SIZE,
            samples_sec=350.0,
            runtime_policy_id="amp_fp16_channels_last_cudnn_ddpfast",
            memory_format="channels_last",
        ),
    )
    evidence = _runtime_selection_evidence_from_rows(runtime_rows)
    _write_stain_qa(output_dir, evidence)

    artifacts = write_runtime_selection_benchmark(
        RuntimeSelectionBenchmarkRequest(
            config_path=CONFIG_PATH,
            output_dir=output_dir,
            v8_artifact_dir=_write_fake_v8_artifacts(tmp_path / "v8"),
            evidence=evidence,
        ),
    )

    proof = _load_json(artifacts.runtime_proof)
    decision = cast("dict[str, object]", proof["selected_runtime_write_decision"])
    efficiency = cast("dict[str, object]", proof["efficiency_followup"])

    assert artifacts.selected_runtime is None
    assert decision["selected_row_id"] == (
        "dual_t4_ddp__bs12__amp_conservative__compile_none__indexed_masked__"
        "policy_amp_fp16_conservative"
    )
    assert efficiency["material_speedup_over_baseline"] is False
    ignored_count = efficiency["ignored_candidate_row_count"]
    assert isinstance(ignored_count, int)
    assert ignored_count >= 1


def test_runtime_selection_fails_closed_when_configured_v5_fallback_is_missing(
    tmp_path: Path,
) -> None:
    """A relaxed row cannot replace v5 if the configured fallback is absent."""
    output_dir = tmp_path / "selection"
    config_path = _write_config_with_baseline(
        tmp_path=tmp_path,
        baseline_path="runs/kaggle/missing_v5/benchmark/selected_runtime.json",
    )
    evidence = _runtime_selection_evidence_from_rows(
        _runtime_rows_for_v5_followup(relaxed_samples_sec=350.0),
    )
    _write_stain_qa(output_dir, evidence)

    artifacts = write_runtime_selection_benchmark(
        RuntimeSelectionBenchmarkRequest(
            config_path=config_path,
            output_dir=output_dir,
            v8_artifact_dir=_write_fake_v8_artifacts(tmp_path / "v8"),
            evidence=evidence,
        ),
    )

    proof = _load_json(artifacts.runtime_proof)
    decision = cast("dict[str, object]", proof["selected_runtime_write_decision"])
    efficiency = cast("dict[str, object]", proof["efficiency_followup"])
    blockers = cast("list[object]", decision["blockers"])

    assert artifacts.selected_runtime is None
    assert "baseline_selected_runtime_not_available" in blockers
    assert efficiency["baseline_available"] is False
    assert not decision["selected_row_id"]


def test_runtime_selection_fails_closed_when_v5_fallback_identity_mismatches(
    tmp_path: Path,
) -> None:
    """The fallback selected-runtime snapshot must match the configured v5 row."""
    output_dir = tmp_path / "selection"
    baseline_path = "runs/kaggle/mismatched_v5/benchmark/selected_runtime.json"
    mismatched = _load_json(
        Path("runs/kaggle/runtime_selection_v5/benchmark/selected_runtime.json"),
    )
    mismatched["runtime_policy_id"] = "wrong_policy"
    write_json(tmp_path / baseline_path, cast("JsonObject", mismatched))
    config_path = _write_config_with_baseline(
        tmp_path=tmp_path,
        baseline_path=baseline_path,
    )
    evidence = _runtime_selection_evidence_from_rows(
        _runtime_rows_for_v5_followup(relaxed_samples_sec=350.0),
    )
    _write_stain_qa(output_dir, evidence)

    artifacts = write_runtime_selection_benchmark(
        RuntimeSelectionBenchmarkRequest(
            config_path=config_path,
            output_dir=output_dir,
            v8_artifact_dir=_write_fake_v8_artifacts(tmp_path / "v8"),
            evidence=evidence,
        ),
    )

    proof = _load_json(artifacts.runtime_proof)
    decision = cast("dict[str, object]", proof["selected_runtime_write_decision"])
    blockers = cast("list[object]", decision["blockers"])

    assert artifacts.selected_runtime is None
    assert "baseline_selected_runtime_identity_mismatch" in blockers


def test_runtime_selection_executor_materializes_relaxed_amp_policy() -> None:
    """Executor policy parsing keeps relaxed eager AMP at bs12 only.

    The decoupled proof batch applies to the new compiled fp16 policy, not the existing
    eager relaxed slice; the exact singleton ID catches accidental broadening to bs48.
    """
    resolved = resolve_json_config(CONFIG_PATH)

    stage = runtime_selection_executor._selection_stage_settings(  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
        resolved.effective_config,
    )

    policies = {
        policy.runtime_policy_id: policy for policy in stage.efficiency_policies
    }
    relaxed = policies["amp_fp16_scalar_gate_relaxed"]
    row_specs = runtime_selection_executor._dual_row_specs(  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
        settings=cast(
            "runtime_selection_executor.pretest.RealDataRuntimePretestSettings",
            object(),
        ),
        stage=stage,
    )
    relaxed_rows = [
        row
        for row in row_specs
        if row.runtime_policy_id == "amp_fp16_scalar_gate_relaxed"
    ]

    assert len(stage.efficiency_policies) == _EFFICIENCY_POLICY_COUNT
    assert relaxed.precision_policy == "amp_scalar_gate_relaxed"
    assert relaxed.compile_scope == "none"
    assert relaxed.autocast_dtype == "float16"
    assert relaxed.fp32_loss is True
    assert relaxed.grad_scaler_enabled is True
    assert [row.row_id for row in relaxed_rows] == [
        (
            "dual_t4_ddp__bs12__amp_scalar_gate_relaxed__compile_none__"
            "indexed_masked__policy_amp_fp16_scalar_gate_relaxed"
        ),
    ]


def test_runtime_selection_blocks_clearly_invalid_numerical_drift(
    tmp_path: Path,
) -> None:
    """Large metric drift still blocks the faster row."""
    output_dir = tmp_path / "selection"
    config_path = _write_config_with_efficiency_policies(
        tmp_path=tmp_path,
        policies=(
            {
                "runtime_policy_id": "amp_fp16_conservative",
                "precision_policy": "amp_conservative",
                "compile_scope": "none",
            },
        ),
    )
    fast_amp = _runtime_row(
        accelerator_mode="dual_t4_ddp",
        per_device_batch_size=12,
        precision_policy="amp_conservative",
        compile_scope="none",
        corruption_strategy="indexed_masked",
        world_size=EXPECTED_DUAL_WORLD_SIZE,
        samples_sec=350.0,
        runtime_policy_id="amp_fp16_conservative",
    )
    runtime_rows = (*_passing_runtime_rows(), fast_amp)
    evidence = _runtime_selection_evidence_from_rows(runtime_rows)
    numerical_rows = tuple(
        _with_large_numerical_drift(row, candidate_row_id=fast_amp["row_id"])
        for row in evidence.numerical_rows
    )
    _write_stain_qa(output_dir, evidence)

    artifacts = write_runtime_selection_benchmark(
        RuntimeSelectionBenchmarkRequest(
            config_path=config_path,
            output_dir=output_dir,
            v8_artifact_dir=_write_fake_v8_artifacts(tmp_path / "v8"),
            evidence=RuntimeSelectionEvidence(
                runtime_rows=evidence.runtime_rows,
                dataloader_rows=evidence.dataloader_rows,
                numerical_rows=numerical_rows,
                corruption_rows=evidence.corruption_rows,
                gate_health_rows=evidence.gate_health_rows,
                gate_health_summary=evidence.gate_health_summary,
                runtime_environment=evidence.runtime_environment,
            ),
        ),
    )

    proof = _load_json(artifacts.runtime_proof)
    decision = cast("dict[str, object]", proof["selected_runtime_write_decision"])
    assert artifacts.selected_runtime is None
    assert "selected_row_numerical_not_pass" in cast(
        "list[object]",
        decision["blockers"],
    )


def test_runtime_selection_blocks_bounded_unrelated_numerical_failure(
    tmp_path: Path,
) -> None:
    """Only numerical-delta failures are eligible for relaxed drift handling."""
    output_dir = tmp_path / "selection"
    config_path = _write_config_with_efficiency_policies(
        tmp_path=tmp_path,
        policies=(
            {
                "runtime_policy_id": "amp_fp16_conservative",
                "precision_policy": "amp_conservative",
                "compile_scope": "none",
            },
        ),
    )
    fast_amp = _runtime_row(
        accelerator_mode="dual_t4_ddp",
        per_device_batch_size=12,
        precision_policy="amp_conservative",
        compile_scope="none",
        corruption_strategy="indexed_masked",
        world_size=EXPECTED_DUAL_WORLD_SIZE,
        samples_sec=350.0,
        runtime_policy_id="amp_fp16_conservative",
    )
    runtime_rows = (*_passing_runtime_rows(), fast_amp)
    evidence = _runtime_selection_evidence_from_rows(runtime_rows)
    numerical_rows = tuple(
        _with_small_numerical_drift(
            row,
            candidate_row_id=fast_amp["row_id"],
            failure_kind="candidate_train_step_RuntimeError",
        )
        for row in evidence.numerical_rows
    )
    _write_stain_qa(output_dir, evidence)

    artifacts = write_runtime_selection_benchmark(
        RuntimeSelectionBenchmarkRequest(
            config_path=config_path,
            output_dir=output_dir,
            v8_artifact_dir=_write_fake_v8_artifacts(tmp_path / "v8"),
            evidence=RuntimeSelectionEvidence(
                runtime_rows=evidence.runtime_rows,
                dataloader_rows=evidence.dataloader_rows,
                numerical_rows=numerical_rows,
                corruption_rows=evidence.corruption_rows,
                gate_health_rows=evidence.gate_health_rows,
                gate_health_summary=evidence.gate_health_summary,
                runtime_environment=evidence.runtime_environment,
            ),
        ),
    )

    proof = _load_json(artifacts.runtime_proof)
    decision = cast("dict[str, object]", proof["selected_runtime_write_decision"])
    assert artifacts.selected_runtime is None
    assert "selected_row_numerical_not_pass" in cast(
        "list[object]",
        decision["blockers"],
    )


def test_runtime_selection_ignores_nonselected_linked_proof_failures(
    tmp_path: Path,
) -> None:
    """A bad nonselected policy row cannot veto the selected safe faster row."""
    output_dir = tmp_path / "selection"
    config_path = _write_config_with_efficiency_policies(
        tmp_path=tmp_path,
        policies=(
            {
                "runtime_policy_id": "amp_fp16_conservative",
                "precision_policy": "amp_conservative",
                "compile_scope": "none",
            },
            {
                "runtime_policy_id": "fp32_channels_last_cudnn_ddpfast",
                "precision_policy": "amp_off_fp32",
                "compile_scope": "none",
            },
        ),
    )
    fast_amp = _runtime_row(
        accelerator_mode="dual_t4_ddp",
        per_device_batch_size=12,
        precision_policy="amp_conservative",
        compile_scope="none",
        corruption_strategy="indexed_masked",
        world_size=EXPECTED_DUAL_WORLD_SIZE,
        samples_sec=350.0,
        runtime_policy_id="amp_fp16_conservative",
    )
    bad_nonselected = _runtime_row(
        accelerator_mode="dual_t4_ddp",
        per_device_batch_size=12,
        precision_policy="amp_off_fp32",
        compile_scope="none",
        corruption_strategy="indexed_masked",
        world_size=EXPECTED_DUAL_WORLD_SIZE,
        samples_sec=300.0,
        runtime_policy_id="fp32_channels_last_cudnn_ddpfast",
        memory_format="channels_last",
    )
    runtime_rows = (*_passing_runtime_rows(), fast_amp, bad_nonselected)
    evidence = _runtime_selection_evidence_from_rows(runtime_rows)
    numerical_rows = tuple(
        row
        for row in evidence.numerical_rows
        if row["candidate_row_id"] != bad_nonselected["row_id"]
    )
    _write_stain_qa(output_dir, evidence)

    artifacts = write_runtime_selection_benchmark(
        RuntimeSelectionBenchmarkRequest(
            config_path=config_path,
            output_dir=output_dir,
            v8_artifact_dir=_write_fake_v8_artifacts(tmp_path / "v8"),
            evidence=RuntimeSelectionEvidence(
                runtime_rows=evidence.runtime_rows,
                dataloader_rows=evidence.dataloader_rows,
                numerical_rows=numerical_rows,
                corruption_rows=evidence.corruption_rows,
                gate_health_rows=evidence.gate_health_rows,
                gate_health_summary=evidence.gate_health_summary,
                runtime_environment=evidence.runtime_environment,
            ),
        ),
    )

    assert artifacts.selected_runtime is not None
    proof = _load_json(artifacts.runtime_proof)
    decision = cast("dict[str, object]", proof["selected_runtime_write_decision"])
    selected = _load_json(artifacts.selected_runtime)

    assert proof["status"] == "pass"
    assert decision["allowed"] is True
    assert decision["linked_pass_row_failures"] == []
    assert selected["selected_row_id"] == fast_amp["row_id"]
    assert selected["runtime_policy_id"] == "amp_fp16_conservative"


def test_runtime_selection_replays_downloaded_v5_artifacts_if_available(
    tmp_path: Path,
) -> None:
    """Downloaded v5 artifacts are the fallback selected-runtime fixture."""
    v5_dir = Path("runs/kaggle/runtime_selection_v5")
    benchmark_dir = v5_dir / "benchmark"
    gate_health_path = v5_dir / "metrics" / "gate_health.csv"
    required_paths = (
        benchmark_dir / "runtime_matrix.csv",
        benchmark_dir / "dataloader_matrix.csv",
        benchmark_dir / "numerical_checks.csv",
        benchmark_dir / "corruption_checks.csv",
        benchmark_dir / "gate_health_summary.json",
        benchmark_dir / "runtime_proof.json",
        benchmark_dir / "stain_corruptor_qa.json",
        gate_health_path,
    )
    if not all(path.exists() for path in required_paths):
        pytest.skip("Downloaded runtime-selection v5 artifacts are not present")
    output_dir = tmp_path / "selection"
    write_json(
        output_dir / "benchmark" / "stain_corruptor_qa.json",
        cast("JsonObject", _load_json(benchmark_dir / "stain_corruptor_qa.json")),
    )

    artifacts = write_runtime_selection_benchmark(
        RuntimeSelectionBenchmarkRequest(
            config_path=CONFIG_PATH,
            output_dir=output_dir,
            v8_artifact_dir=_write_fake_v8_artifacts(tmp_path / "v8"),
            evidence=load_runtime_selection_evidence(v5_dir),
        ),
    )

    assert artifacts.selected_runtime is not None
    proof = _load_json(artifacts.runtime_proof)
    selected = _load_json(artifacts.selected_runtime)
    assert proof["status"] == "pass"
    assert proof["selected_runtime_written"] is True
    assert selected["selected_row_id"] == (
        "dual_t4_ddp__bs12__amp_conservative__compile_none"
        "__indexed_masked__policy_amp_fp16_conservative"
    )
    assert selected["runtime_policy_id"] == "amp_fp16_conservative"


def test_runtime_selection_blocks_policy_mismatched_dataloader(
    tmp_path: Path,
) -> None:
    """Dataloader proof must be bound to the selected runtime policy id."""
    output_dir = tmp_path / "selection"
    fast_policy = "amp_fp16_channels_last_cudnn_ddpfast"
    config_path = _write_config_with_efficiency_policies(
        tmp_path=tmp_path,
        policies=(
            {
                "runtime_policy_id": fast_policy,
                "precision_policy": "amp_conservative",
                "compile_scope": "none",
            },
        ),
    )
    runtime_rows = (
        *_passing_runtime_rows(),
        _runtime_row(
            accelerator_mode="dual_t4_ddp",
            per_device_batch_size=12,
            precision_policy="amp_conservative",
            compile_scope="none",
            corruption_strategy="indexed_masked",
            world_size=EXPECTED_DUAL_WORLD_SIZE,
            samples_sec=500.0,
            runtime_policy_id=fast_policy,
            memory_format="channels_last",
        ),
    )
    evidence = _runtime_selection_evidence_from_rows(runtime_rows)
    dataloader_rows = tuple(
        {
            **row,
            "runtime_policy_id": "fp32_eager_default"
            if row["runtime_policy_id"] == fast_policy
            else row["runtime_policy_id"],
        }
        for row in evidence.dataloader_rows
    )
    evidence = RuntimeSelectionEvidence(
        runtime_rows=evidence.runtime_rows,
        dataloader_rows=dataloader_rows,
        numerical_rows=evidence.numerical_rows,
        corruption_rows=evidence.corruption_rows,
        gate_health_rows=evidence.gate_health_rows,
        gate_health_summary=evidence.gate_health_summary,
        runtime_environment=evidence.runtime_environment,
    )
    _write_stain_qa(output_dir, evidence)

    artifacts = write_runtime_selection_benchmark(
        RuntimeSelectionBenchmarkRequest(
            config_path=config_path,
            output_dir=output_dir,
            v8_artifact_dir=_write_fake_v8_artifacts(tmp_path / "v8"),
            evidence=evidence,
        ),
    )

    proof = _load_json(artifacts.runtime_proof)
    decision = cast("dict[str, object]", proof["selected_runtime_write_decision"])
    assert artifacts.selected_runtime is None
    assert "selected_row_dataloader_not_pass" in cast(
        "list[object]",
        decision["blockers"],
    )


def test_runtime_selection_accepts_train_corruption_without_validation_rng_flag(
    tmp_path: Path,
) -> None:
    """Train corruption proof rows do not carry the validation clean-RNG flag."""
    output_dir = tmp_path / "selection"
    evidence = _passing_runtime_selection_evidence()
    corruption_rows = tuple(
        {
            **row,
            "clean_validation_rng_advanced": ""
            if row["split"] == "train"
            else row["clean_validation_rng_advanced"],
        }
        for row in evidence.corruption_rows
    )
    evidence = RuntimeSelectionEvidence(
        runtime_rows=evidence.runtime_rows,
        dataloader_rows=evidence.dataloader_rows,
        numerical_rows=evidence.numerical_rows,
        corruption_rows=corruption_rows,
        gate_health_rows=evidence.gate_health_rows,
        gate_health_summary=evidence.gate_health_summary,
        runtime_environment=evidence.runtime_environment,
    )
    _write_stain_qa(output_dir, evidence)

    artifacts = write_runtime_selection_benchmark(
        RuntimeSelectionBenchmarkRequest(
            config_path=CONFIG_PATH,
            output_dir=output_dir,
            v8_artifact_dir=_write_fake_v8_artifacts(tmp_path / "v8"),
            evidence=evidence,
        ),
    )

    assert artifacts.selected_runtime is not None
    proof = _load_json(artifacts.runtime_proof)
    assert proof["status"] == "pass"
    assert proof["selected_runtime_written"] is True


def test_runtime_selection_executor_marks_local_pass_gate_rows_eligible() -> None:
    """Executor normalization must preserve local-pass gate-health proof rows."""
    gate_row = dict.fromkeys(GATE_HEALTH_COLUMNS, "")
    gate_row.update({
        "status": "",
        "gate_health_status": "local_pass",
        "full_run_eligible": "false",
        "module": "encoder.0",
        "gate_kind": "scalar",
    })
    failed_runtime_row = {
        "status": "fail",
        "gate_health_status": "local_pass",
        "full_run_eligible": "true",
    }

    normalized = runtime_selection_executor._rows_with_selection_scope(  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
        (gate_row, failed_runtime_row),
    )

    assert normalized[0]["benchmark_kind"] == "kaggle_runtime_selection"
    assert normalized[0]["benchmark_source"] == "kaggle_runtime_benchmark"
    assert normalized[0]["gate_health_status"] == "pass"
    assert normalized[0]["full_run_eligible"] == "true"
    assert not normalized[0]["status"]
    assert normalized[1]["gate_health_status"] == "pass"
    assert normalized[1]["full_run_eligible"] == "false"
    assert normalized[1]["status"] == "fail"


def test_runtime_selection_executor_expands_gate_rows_to_indexed_candidates() -> None:
    """Single-visible indexed rows need candidate-bound gate-health rows too."""
    branchless = _runtime_row(
        accelerator_mode="single_visible_t4",
        per_device_batch_size=8,
        precision_policy="amp_off_fp32",
        compile_scope="none",
        corruption_strategy="branchless_all",
        world_size=1,
        samples_sec=7.0,
    )
    indexed = _runtime_row(
        accelerator_mode="single_visible_t4",
        per_device_batch_size=8,
        precision_policy="amp_off_fp32",
        compile_scope="none",
        corruption_strategy="indexed_masked",
        world_size=1,
        samples_sec=7.1,
    )
    gate_row = dict.fromkeys(GATE_HEALTH_COLUMNS, "")
    gate_row.update({
        "candidate_row_id": branchless["row_id"],
        "row_id": f"{branchless['row_id']}__gate__encoder.0",
        "module": "encoder.0",
        "gate_kind": "scalar",
        "gate_health_status": "local_pass",
    })

    expanded = runtime_selection_executor._single_gate_rows(  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
        gate_rows=(gate_row,),
        runtime_rows=(branchless, indexed),
    )

    rows_by_candidate = {row["candidate_row_id"]: row for row in expanded}
    assert set(rows_by_candidate) == {branchless["row_id"], indexed["row_id"]}
    assert rows_by_candidate[indexed["row_id"]]["row_id"] == (
        f"{indexed['row_id']}__gate__encoder.0"
    )
    assert rows_by_candidate[indexed["row_id"]]["full_run_eligible"] == "true"
    assert rows_by_candidate[indexed["row_id"]]["gate_health_status"] == "pass"


def test_runtime_selection_executor_does_not_expand_gate_rows_to_other_rows() -> None:
    """Gate-health expansion is limited to passing FP32 eager indexed rows."""
    branchless = _runtime_row(
        accelerator_mode="single_visible_t4",
        per_device_batch_size=8,
        precision_policy="amp_off_fp32",
        compile_scope="none",
        corruption_strategy="branchless_all",
        world_size=1,
        samples_sec=7.0,
    )
    failed_indexed = _runtime_row(
        accelerator_mode="single_visible_t4",
        per_device_batch_size=8,
        precision_policy="amp_off_fp32",
        compile_scope="none",
        corruption_strategy="indexed_masked",
        world_size=1,
        samples_sec=7.1,
        status="fail",
    )
    amp_indexed = _runtime_row(
        accelerator_mode="single_visible_t4",
        per_device_batch_size=8,
        precision_policy="amp_conservative",
        compile_scope="none",
        corruption_strategy="indexed_masked",
        world_size=1,
        samples_sec=7.2,
    )
    compiled_indexed = _runtime_row(
        accelerator_mode="single_visible_t4",
        per_device_batch_size=8,
        precision_policy="amp_off_fp32",
        compile_scope="model_forward",
        corruption_strategy="indexed_masked",
        world_size=1,
        samples_sec=7.3,
    )
    gate_row = dict.fromkeys(GATE_HEALTH_COLUMNS, "")
    gate_row.update({
        "candidate_row_id": branchless["row_id"],
        "row_id": f"{branchless['row_id']}__gate__encoder.0",
        "module": "encoder.0",
        "gate_kind": "scalar",
        "gate_health_status": "local_pass",
    })

    expanded = runtime_selection_executor._single_gate_rows(  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
        gate_rows=(gate_row,),
        runtime_rows=(branchless, failed_indexed, amp_indexed, compiled_indexed),
    )

    assert {row["candidate_row_id"] for row in expanded} == {branchless["row_id"]}


def test_runtime_selection_executor_sets_scalar_gate_precision_policy() -> None:
    """Relaxed AMP toggles scalar gate math while conservative rows keep FP32."""
    model = build_non_equivariant_vae()

    updated = runtime_selection_executor._set_scalar_gate_precision(  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
        model=model,
        force_fp32=False,
    )

    gates = [
        module
        for module in model.modules()
        if isinstance(module, GatedScalarActivation)
    ]
    assert updated == len(gates)
    assert gates
    assert {gate.force_fp32 for gate in gates} == {False}

    runtime_selection_executor._set_scalar_gate_precision(  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
        model=model,
        force_fp32=True,
    )

    assert {gate.force_fp32 for gate in gates} == {True}


def test_gated_scalar_activation_records_precision_proof() -> None:
    """Gate precision proof is captured during the actual forward path."""
    gate = GatedScalarActivation(channels=2, force_fp32=False)
    inputs = torch.ones((1, 2, 4, 4), dtype=torch.float16)

    output = cast("torch.Tensor", gate(inputs))

    assert output.dtype == torch.float16
    assert gate.last_precision_proof_status == "pass"
    assert gate.last_input_dtype == "float16"
    assert gate.last_gate_math_dtype == "float16"
    assert gate.last_gate_tensor_dtype == "float16"
    assert gate.last_output_dtype == "float16"


def test_runtime_selection_blocks_train_only_dataloader_proof(
    tmp_path: Path,
) -> None:
    """Train-only dataloader proof cannot unlock selected runtime."""
    output_dir = tmp_path / "selection"
    evidence = _passing_runtime_selection_evidence()
    _write_stain_qa(output_dir, evidence)
    train_only_rows = tuple(
        row for row in evidence.dataloader_rows if row["split"] == "train"
    )

    artifacts = write_runtime_selection_benchmark(
        RuntimeSelectionBenchmarkRequest(
            config_path=CONFIG_PATH,
            output_dir=output_dir,
            v8_artifact_dir=_write_fake_v8_artifacts(tmp_path / "v8"),
            evidence=RuntimeSelectionEvidence(
                runtime_rows=evidence.runtime_rows,
                dataloader_rows=train_only_rows,
                numerical_rows=evidence.numerical_rows,
                corruption_rows=evidence.corruption_rows,
                gate_health_rows=evidence.gate_health_rows,
                gate_health_summary=evidence.gate_health_summary,
                runtime_environment=evidence.runtime_environment,
            ),
        ),
    )

    proof = _load_json(artifacts.runtime_proof)
    decision = cast("dict[str, object]", proof["selected_runtime_write_decision"])

    assert artifacts.selected_runtime is None
    assert "runtime_pass_rows_linked_proof_not_pass" in cast(
        "list[object]",
        decision["blockers"],
    )
    assert not (output_dir / "benchmark" / "selected_runtime.json").exists()


def test_runtime_selection_blocks_missing_child_launch_proof(tmp_path: Path) -> None:
    """Dual timing proof must include the configured child-process launch."""
    output_dir = tmp_path / "selection"
    evidence = _passing_runtime_selection_evidence()
    _write_stain_qa(output_dir, evidence)
    runtime_environment = dict(evidence.runtime_environment)
    runtime_environment.pop("child_process_launch_command")

    artifacts = write_runtime_selection_benchmark(
        RuntimeSelectionBenchmarkRequest(
            config_path=CONFIG_PATH,
            output_dir=output_dir,
            v8_artifact_dir=_write_fake_v8_artifacts(tmp_path / "v8"),
            evidence=RuntimeSelectionEvidence(
                runtime_rows=evidence.runtime_rows,
                dataloader_rows=evidence.dataloader_rows,
                numerical_rows=evidence.numerical_rows,
                corruption_rows=evidence.corruption_rows,
                gate_health_rows=evidence.gate_health_rows,
                gate_health_summary=evidence.gate_health_summary,
                runtime_environment=runtime_environment,
            ),
        ),
    )

    proof = _load_json(artifacts.runtime_proof)
    dual_gate = cast("dict[str, object]", proof["dual_t4_train_step_gate"])

    assert artifacts.selected_runtime is None
    assert dual_gate["child_process_launch_status"] == "skipped_unsupported"
    assert not (output_dir / "benchmark" / "selected_runtime.json").exists()


def test_runtime_selection_blocks_unbound_gate_health_rows(tmp_path: Path) -> None:
    """Gate health must be bound to each candidate runtime row."""
    output_dir = tmp_path / "selection"
    evidence = _passing_runtime_selection_evidence()
    _write_stain_qa(output_dir, evidence)
    gate_rows = tuple(
        {
            **row,
            "candidate_row_id": "",
        }
        for row in evidence.gate_health_rows
    )

    artifacts = write_runtime_selection_benchmark(
        RuntimeSelectionBenchmarkRequest(
            config_path=CONFIG_PATH,
            output_dir=output_dir,
            v8_artifact_dir=_write_fake_v8_artifacts(tmp_path / "v8"),
            evidence=RuntimeSelectionEvidence(
                runtime_rows=evidence.runtime_rows,
                dataloader_rows=evidence.dataloader_rows,
                numerical_rows=evidence.numerical_rows,
                corruption_rows=evidence.corruption_rows,
                gate_health_rows=gate_rows,
                gate_health_summary=evidence.gate_health_summary,
                runtime_environment=evidence.runtime_environment,
            ),
        ),
    )

    proof = _load_json(artifacts.runtime_proof)
    decision = cast("dict[str, object]", proof["selected_runtime_write_decision"])

    assert artifacts.selected_runtime is None
    assert "selected_row_gate_health_not_pass" in cast(
        "list[object]",
        decision["blockers"],
    )


def test_runtime_selection_blocks_shallow_dataloader_measurement(
    tmp_path: Path,
) -> None:
    """Dataloader rows must meet the configured measurement depth."""
    output_dir = tmp_path / "selection"
    evidence = _passing_runtime_selection_evidence()
    _write_stain_qa(output_dir, evidence)
    shallow_rows = tuple(
        {
            **row,
            "batches_measured": "3",
        }
        for row in evidence.dataloader_rows
    )

    artifacts = write_runtime_selection_benchmark(
        RuntimeSelectionBenchmarkRequest(
            config_path=CONFIG_PATH,
            output_dir=output_dir,
            v8_artifact_dir=_write_fake_v8_artifacts(tmp_path / "v8"),
            evidence=RuntimeSelectionEvidence(
                runtime_rows=evidence.runtime_rows,
                dataloader_rows=shallow_rows,
                numerical_rows=evidence.numerical_rows,
                corruption_rows=evidence.corruption_rows,
                gate_health_rows=evidence.gate_health_rows,
                gate_health_summary=evidence.gate_health_summary,
                runtime_environment=evidence.runtime_environment,
            ),
        ),
    )

    proof = _load_json(artifacts.runtime_proof)
    decision = cast("dict[str, object]", proof["selected_runtime_write_decision"])

    assert artifacts.selected_runtime is None
    assert "selected_row_dataloader_not_pass" in cast(
        "list[object]",
        decision["blockers"],
    )


def test_runtime_selection_blocks_train_only_corruption_proof(
    tmp_path: Path,
) -> None:
    """Training-only corruption evidence cannot certify checkpoint evaluation.

    Runtime selection depends on the clean validation RNG lane used for quality
    evidence; dropping those rows must block the winner even when training corruption
    passes. This catches a selector that checks only candidate presence or the train
    split.
    """
    output_dir = tmp_path / "selection"
    evidence = _passing_runtime_selection_evidence()
    _write_stain_qa(output_dir, evidence)
    train_only_corruption = tuple(
        row for row in evidence.corruption_rows if row["split"] == "train"
    )

    artifacts = write_runtime_selection_benchmark(
        RuntimeSelectionBenchmarkRequest(
            config_path=CONFIG_PATH,
            output_dir=output_dir,
            v8_artifact_dir=_write_fake_v8_artifacts(tmp_path / "v8"),
            evidence=RuntimeSelectionEvidence(
                runtime_rows=evidence.runtime_rows,
                dataloader_rows=evidence.dataloader_rows,
                numerical_rows=evidence.numerical_rows,
                corruption_rows=train_only_corruption,
                gate_health_rows=evidence.gate_health_rows,
                gate_health_summary=evidence.gate_health_summary,
                runtime_environment=evidence.runtime_environment,
            ),
        ),
    )

    proof = _load_json(artifacts.runtime_proof)
    decision = cast("dict[str, object]", proof["selected_runtime_write_decision"])

    assert artifacts.selected_runtime is None
    assert "selected_row_corruption_not_pass" in cast(
        "list[object]",
        decision["blockers"],
    )


def test_runtime_selection_blocks_shallow_numerical_proof(tmp_path: Path) -> None:
    """Numerical checks must cover the fixed three-batch grid."""
    output_dir = tmp_path / "selection"
    evidence = _passing_runtime_selection_evidence()
    _write_stain_qa(output_dir, evidence)
    batch_zero_only = tuple(
        row for row in evidence.numerical_rows if row["batch_index"] == "0"
    )

    artifacts = write_runtime_selection_benchmark(
        RuntimeSelectionBenchmarkRequest(
            config_path=CONFIG_PATH,
            output_dir=output_dir,
            v8_artifact_dir=_write_fake_v8_artifacts(tmp_path / "v8"),
            evidence=RuntimeSelectionEvidence(
                runtime_rows=evidence.runtime_rows,
                dataloader_rows=evidence.dataloader_rows,
                numerical_rows=batch_zero_only,
                corruption_rows=evidence.corruption_rows,
                gate_health_rows=evidence.gate_health_rows,
                gate_health_summary=evidence.gate_health_summary,
                runtime_environment=evidence.runtime_environment,
            ),
        ),
    )

    proof = _load_json(artifacts.runtime_proof)
    decision = cast("dict[str, object]", proof["selected_runtime_write_decision"])

    assert artifacts.selected_runtime is None
    assert "selected_row_numerical_not_pass" in cast(
        "list[object]",
        decision["blockers"],
    )


def test_runtime_selection_blocks_missing_stain_qa_link(tmp_path: Path) -> None:
    """A selected-runtime payload must include the required stain QA hash."""
    output_dir = tmp_path / "selection"
    artifacts = write_runtime_selection_benchmark(
        RuntimeSelectionBenchmarkRequest(
            config_path=CONFIG_PATH,
            output_dir=output_dir,
            v8_artifact_dir=_write_fake_v8_artifacts(tmp_path / "v8"),
            evidence=_passing_runtime_selection_evidence(),
        ),
    )

    proof = _load_json(artifacts.runtime_proof)
    decision = cast("dict[str, object]", proof["selected_runtime_write_decision"])

    assert artifacts.selected_runtime is None
    assert "stain_corruptor_qa_not_pass" in cast(
        "list[object]",
        decision["blockers"],
    )
    assert not (output_dir / "benchmark" / "selected_runtime.json").exists()


def _write_fake_v8_artifacts(root: Path) -> Path:
    rows = [
        _runtime_row(
            accelerator_mode="single_visible_t4",
            per_device_batch_size=batch_size,
            precision_policy="amp_off_fp32",
            compile_scope="none",
            corruption_strategy=corruption_strategy,
            world_size=1,
            samples_sec=120.0 + batch_size,
        )
        for batch_size in (4, 8, 12)
        for corruption_strategy in CORRUPTION_STRATEGIES
    ]
    write_json(
        root / "benchmark" / "runtime_proof.json",
        {
            "status": "pretest_incomplete",
            "full_run_eligible": False,
            "selected_runtime_written": False,
            "eligible_pass_row_count": len(rows),
        },
    )
    write_csv(root / "benchmark" / "runtime_matrix.csv", RUNTIME_MATRIX_COLUMNS, rows)
    write_csv(
        root / "benchmark" / "dataloader_matrix.csv",
        DATALOADER_MATRIX_COLUMNS,
        (),
    )
    write_csv(root / "benchmark" / "numerical_checks.csv", NUMERICAL_CHECK_COLUMNS, ())
    write_csv(
        root / "benchmark" / "corruption_checks.csv",
        CORRUPTION_CHECK_COLUMNS,
        (),
    )
    write_json(root / "benchmark" / "gate_health_summary.json", {"status": "fail"})
    write_csv(root / "metrics" / "gate_health.csv", GATE_HEALTH_COLUMNS, ())
    return root


def _passing_runtime_selection_evidence() -> RuntimeSelectionEvidence:
    runtime_rows = tuple(_passing_runtime_rows())
    return _runtime_selection_evidence_from_rows(runtime_rows)


def _runtime_rows_for_v5_followup(
    *,
    relaxed_samples_sec: float,
) -> tuple[dict[str, str], ...]:
    rows: list[dict[str, str]] = []
    for batch_size in (4, 8, 12):
        for corruption_strategy in CORRUPTION_STRATEGIES:
            rows.extend((
                _runtime_row(
                    accelerator_mode="single_visible_t4",
                    per_device_batch_size=batch_size,
                    precision_policy="amp_off_fp32",
                    compile_scope="none",
                    corruption_strategy=corruption_strategy,
                    world_size=1,
                    samples_sec=10.0,
                ),
                _runtime_row(
                    accelerator_mode="dual_t4_ddp",
                    per_device_batch_size=batch_size,
                    precision_policy="amp_off_fp32",
                    compile_scope="none",
                    corruption_strategy=corruption_strategy,
                    world_size=EXPECTED_DUAL_WORLD_SIZE,
                    samples_sec=10.0,
                ),
            ))
    rows.append(
        _runtime_row(
            accelerator_mode="dual_t4_ddp",
            per_device_batch_size=12,
            precision_policy="amp_scalar_gate_relaxed",
            compile_scope="none",
            corruption_strategy="indexed_masked",
            world_size=EXPECTED_DUAL_WORLD_SIZE,
            samples_sec=relaxed_samples_sec,
            runtime_policy_id="amp_fp16_scalar_gate_relaxed",
        ),
    )
    return tuple(rows)


def _fp16_compiled_bs48_row() -> dict[str, str]:
    row = _runtime_row(
        accelerator_mode="dual_t4_ddp",
        per_device_batch_size=48,
        precision_policy="amp_conservative",
        compile_scope=COMPILE_STEP,
        corruption_strategy="indexed_masked",
        world_size=EXPECTED_DUAL_WORLD_SIZE,
        samples_sec=400.0,
        runtime_policy_id="compile_step_ddp_optimizer_fp16_channels_last",
        memory_format="channels_last",
    )
    row["ddp_gradient_as_bucket_view"] = "true"
    row["optimize_ddp"] = "ddp_optimizer"
    row["compiled_autograd"] = "false"
    row["reorder_compute_comm_overlap"] = "false"
    row["ddp_broadcast_buffers"] = "false"
    row["ddp_find_unused_parameters"] = "false"
    row["ddp_bucket_cap_mb"] = str(_WINNER_BUCKET_CAP_MB)
    row["fused_optimizer"] = "true"
    row["gradient_clip_foreach"] = "true"
    row["graph_break_count"] = "0"
    row["recompile_count"] = "0"
    return row


def _write_config_with_baseline(*, tmp_path: Path, baseline_path: str) -> Path:
    return _write_config_with_efficiency_policies(
        tmp_path=tmp_path,
        baseline_path=baseline_path,
        policies=None,
    )


def _write_config_with_efficiency_policies(
    *,
    tmp_path: Path,
    policies: tuple[JsonObject, ...] | None,
    baseline_path: str | None = None,
) -> Path:
    baseline = baseline_path or str(
        Path(
            "runs/kaggle/runtime_selection_v5/benchmark/selected_runtime.json",
        ).resolve(),
    )
    payload = _load_json(CONFIG_PATH)
    payload["source_config"] = str(
        Path("configs/spec0001/non_eq_vae_model_base.json").resolve(),
    )
    runtime = cast("dict[str, object]", payload["runtime_matrix"])
    selection = cast("dict[str, object]", runtime["selection_benchmark_slice"])
    efficiency = cast("dict[str, object]", selection["efficiency_followup"])
    efficiency["baseline_selected_runtime"] = baseline
    if policies is not None:
        efficiency["policies"] = [dict(policy) for policy in policies]
    config_path = (
        tmp_path / "configs" / "spec0001" / "non_eq_vae_kaggle_runtime_benchmark.json"
    )
    write_json(config_path, cast("JsonObject", payload))
    return config_path


def _runtime_selection_evidence_from_rows(
    runtime_rows: tuple[dict[str, str], ...],
) -> RuntimeSelectionEvidence:
    pass_row_ids = [row["row_id"] for row in runtime_rows if row["status"] == "pass"]
    return RuntimeSelectionEvidence(
        runtime_rows=runtime_rows,
        dataloader_rows=tuple(_dataloader_rows(runtime_rows)),
        numerical_rows=tuple(
            _candidate_rows(
                runtime_rows=runtime_rows,
                columns=NUMERICAL_CHECK_COLUMNS,
                row_prefix="numerical",
            ),
        ),
        corruption_rows=tuple(
            _candidate_rows(
                runtime_rows=runtime_rows,
                columns=CORRUPTION_CHECK_COLUMNS,
                row_prefix="corruption",
            ),
        ),
        gate_health_rows=tuple(_gate_health_rows(runtime_rows)),
        gate_health_summary=cast(
            "JsonObject",
            {
                "status": "pass",
                "benchmark_kind": "kaggle_runtime_selection",
                "benchmark_source": "kaggle_runtime_benchmark",
                "overall_status": "pass",
                "full_run_eligible": True,
                "logged_intervals": 1,
                "module_count": 34,
                "nonfinite_count": 0,
                "candidate_row_ids": pass_row_ids,
                "failing_modules": [],
                "warning_modules": [],
            },
        ),
        runtime_environment={
            "status": "pass",
            "machine_shape": "NvidiaTeslaT4",
            "visible_device_count": EXPECTED_DUAL_WORLD_SIZE,
            "cuda_device_count": EXPECTED_DUAL_WORLD_SIZE,
            "gpu_names": ["Tesla T4", "Tesla T4"],
            "world_size": EXPECTED_DUAL_WORLD_SIZE,
            "nproc_per_node": EXPECTED_DUAL_WORLD_SIZE,
            "rank_assignments": [
                {"rank": 0, "local_rank": 0, "cuda_device": "cuda:0"},
                {"rank": 1, "local_rank": 1, "cuda_device": "cuda:1"},
            ],
            "child_process_launch_command": "torchrun --nproc_per_node=2",
        },
    )


def _passing_runtime_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    rows.extend(
        _runtime_row(
            accelerator_mode="single_visible_t4",
            per_device_batch_size=batch_size,
            precision_policy="amp_off_fp32",
            compile_scope="none",
            corruption_strategy=corruption_strategy,
            world_size=1,
            samples_sec=150.0 + batch_size,
        )
        for batch_size in (4, 8, 12)
        for corruption_strategy in CORRUPTION_STRATEGIES
    )
    rows.extend(
        _runtime_row(
            accelerator_mode="single_visible_t4",
            per_device_batch_size=batch_size,
            precision_policy="amp_conservative",
            compile_scope="none",
            corruption_strategy=corruption_strategy,
            world_size=1,
            samples_sec=175.0 + batch_size,
        )
        for batch_size in (8, 12)
        for corruption_strategy in CORRUPTION_STRATEGIES
    )
    for batch_size in (4, 8, 12):
        for index, corruption_strategy in enumerate(CORRUPTION_STRATEGIES):
            rows.append(
                _runtime_row(
                    accelerator_mode="dual_t4_ddp",
                    per_device_batch_size=batch_size,
                    precision_policy="amp_off_fp32",
                    compile_scope="none",
                    corruption_strategy=corruption_strategy,
                    world_size=EXPECTED_DUAL_WORLD_SIZE,
                    samples_sec=250.0 + batch_size + index,
                ),
            )
    rows.extend((
        _runtime_row(
            accelerator_mode="dual_t4_ddp",
            per_device_batch_size=12,
            precision_policy="amp_scalar_gate_relaxed",
            compile_scope="none",
            corruption_strategy="indexed_masked",
            world_size=EXPECTED_DUAL_WORLD_SIZE,
            samples_sec=350.0,
            runtime_policy_id="amp_fp16_scalar_gate_relaxed",
        ),
        _runtime_row(
            accelerator_mode="single_visible_t4",
            per_device_batch_size=8,
            precision_policy="amp_off_fp32",
            compile_scope="model_forward",
            corruption_strategy="branchless_all",
            world_size=1,
            samples_sec=999.0,
        ),
    ))
    return rows


def _runtime_row(  # noqa: PLR0913
    *,
    accelerator_mode: str,
    per_device_batch_size: int,
    precision_policy: str,
    compile_scope: str,
    corruption_strategy: str,
    world_size: int,
    samples_sec: float,
    status: str = "pass",
    runtime_policy_id: str = "fp32_eager_default",
    memory_format: str = "contiguous",
    amp_step_skipped_count: int = 0,
) -> dict[str, str]:
    row = dict.fromkeys(RUNTIME_MATRIX_COLUMNS, "")
    row.update({
        "run_name": RUN_NAME,
        "benchmark_kind": "kaggle_runtime_selection",
        "benchmark_source": "kaggle_runtime_benchmark",
        "full_run_eligible": "true" if status == "pass" else "false",
        "row_id": _row_id(
            accelerator_mode=accelerator_mode,
            batch_size=per_device_batch_size,
            precision_policy=precision_policy,
            compile_scope=compile_scope,
            corruption_strategy=corruption_strategy,
            runtime_policy_id=runtime_policy_id,
        ),
        "accelerator_mode": accelerator_mode,
        "machine_shape": "NvidiaTeslaT4",
        "visible_device_count": str(world_size),
        "cuda_device_count": str(world_size),
        "gpu_names": json.dumps(["Tesla T4"] * world_size),
        "ddp_backend": "nccl" if world_size == EXPECTED_DUAL_WORLD_SIZE else "",
        "world_size": str(world_size),
        "nproc_per_node": str(world_size),
        "precision_policy": precision_policy,
        "amp_enabled": "false" if precision_policy == "amp_off_fp32" else "true",
        "torch_compile_enabled": "false" if compile_scope == "none" else "true",
        "compile_scope": compile_scope,
        "runtime_policy_id": runtime_policy_id,
        "memory_format": memory_format,
        "autocast_dtype": "float16" if precision_policy != "amp_off_fp32" else "",
        "fp32_loss": "true",
        "grad_scaler_enabled": "false"
        if precision_policy == "amp_off_fp32"
        else "true",
        "cudnn_benchmark": "false",
        "cudnn_deterministic": "false",
        "deterministic_algorithms": "false",
        "tf32_enabled": "false",
        "matmul_precision": "highest",
        "ddp_static_graph": "false",
        "ddp_gradient_as_bucket_view": "false",
        "optimizer_implementation": "adamw_default",
        "zero_grad_set_to_none": "true",
        "gradient_clip_foreach": "true",
        "compile_dynamic": "false",
        # Spec 0011 S13: eager recipe knobs, matching the production row producers.
        **EAGER_RECIPE_KNOB_COLUMNS,
        "corruption_strategy": corruption_strategy,
        "per_device_batch_size": str(per_device_batch_size),
        "global_batch_size": str(per_device_batch_size * world_size),
        "gradient_accumulation_steps": "1",
        "warmup_steps": "5",
        "measured_steps": "25",
        "repeats": "3",
        "compile_startup_sec": "0.000000",
        "compile_settle_steps": "0" if compile_scope == "none" else "5",
        "steady_step_ms_p50": "25.000000",
        "steady_step_ms_p95": "30.000000",
        "samples_sec": f"{samples_sec:.6f}",
        "trainer_samples_sec": f"{samples_sec:.6f}",
        "max_vram_allocated_mb": "4000.000000",
        "max_vram_reserved_mb": "5000.000000",
        "vram_headroom_fraction": "0.500000",
        "amp_step_skipped_count": str(amp_step_skipped_count),
        "gate_health_status": "pass",
        "gate_health_warning_count": "0",
        "numerical_check_status": "pass",
        "data_wait_fraction_p95": "0.010000",
        "graph_break_count": "0",
        "recompile_count": "0",
        "oom": "false",
        "status": status,
        "failure_kind": "",
        "failure_message_hash": "",
    })
    return row


def _dataloader_rows(runtime_rows: tuple[dict[str, str], ...]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for runtime_row in runtime_rows:
        if runtime_row["status"] != "pass":
            continue
        key = (
            runtime_row["accelerator_mode"],
            runtime_row["machine_shape"],
            runtime_row["world_size"],
            runtime_row["per_device_batch_size"],
            runtime_row["runtime_policy_id"],
        )
        if key in seen:
            continue
        seen.add(key)
        for rank in range(int(runtime_row["world_size"])):
            for split in ("train", "validation"):
                row = dict.fromkeys(DATALOADER_MATRIX_COLUMNS, "")
                row.update({
                    "run_name": RUN_NAME,
                    "benchmark_kind": "kaggle_runtime_selection",
                    "benchmark_source": "kaggle_runtime_benchmark",
                    "full_run_eligible": "true",
                    "accelerator_mode": runtime_row["accelerator_mode"],
                    "machine_shape": runtime_row["machine_shape"],
                    "world_size": runtime_row["world_size"],
                    "runtime_policy_id": runtime_row["runtime_policy_id"],
                    "memory_format": runtime_row["memory_format"],
                    "rank": str(rank),
                    "split": split,
                    "num_workers": "1",
                    "prefetch_factor": "2",
                    "pin_memory": "true",
                    "persistent_workers": "true",
                    "non_blocking_h2d": "true",
                    "batch_size": runtime_row["per_device_batch_size"],
                    "batches_measured": "25",
                    "batch_fetch_ms_p50": "1.000000",
                    "batch_fetch_ms_p95": "2.000000",
                    "h2d_ms_p50": "1.000000",
                    "h2d_ms_p95": "2.000000",
                    "loader_samples_sec": "512.000000",
                    "trainer_samples_sec": runtime_row["trainer_samples_sec"],
                    "data_wait_fraction_p50": "0.005000",
                    "data_wait_fraction_p95": "0.010000",
                    "rank_sample_count": "25",
                    "dropped_sample_count": "0",
                    "status": "pass",
                    "failure_kind": "",
                })
                rows.append(row)
    return rows


def _candidate_rows(
    *,
    runtime_rows: tuple[dict[str, str], ...],
    columns: tuple[str, ...],
    row_prefix: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for runtime_row in runtime_rows:
        if runtime_row["status"] != "pass":
            continue
        if tuple(columns) == NUMERICAL_CHECK_COLUMNS:
            rows.extend(
                _candidate_row(
                    runtime_row=runtime_row,
                    columns=columns,
                    row_prefix=row_prefix,
                    batch_index=batch_index,
                    split="train",
                )
                for batch_index in range(3)
            )
        elif tuple(columns) == CORRUPTION_CHECK_COLUMNS:
            rows.extend(
                _candidate_row(
                    runtime_row=runtime_row,
                    columns=columns,
                    row_prefix=row_prefix,
                    batch_index=0,
                    split=split,
                )
                for split in ("train", "validation")
            )
        else:
            rows.append(
                _candidate_row(
                    runtime_row=runtime_row,
                    columns=columns,
                    row_prefix=row_prefix,
                    batch_index=0,
                    split="train",
                ),
            )
    return rows


def _candidate_row(
    *,
    runtime_row: dict[str, str],
    columns: tuple[str, ...],
    row_prefix: str,
    batch_index: int,
    split: str,
) -> dict[str, str]:
    row = dict.fromkeys(columns, "")
    row_id = f"{row_prefix}__{runtime_row['row_id']}__{split}__batch_{batch_index}"
    shared = {
        "run_name": RUN_NAME,
        "benchmark_kind": "kaggle_runtime_selection",
        "benchmark_source": "kaggle_runtime_benchmark",
        "full_run_eligible": "true",
        "accelerator_mode": runtime_row["accelerator_mode"],
        "machine_shape": runtime_row["machine_shape"],
        "row_id": row_id,
        "reference_row_id": _row_id(
            accelerator_mode=runtime_row["accelerator_mode"],
            batch_size=min(
                int(runtime_row["per_device_batch_size"]),
                _PROOF_REFERENCE_BATCH_SIZE,
            ),
            precision_policy="amp_off_fp32",
            compile_scope="none",
            corruption_strategy="branchless_all",
        ),
        "candidate_row_id": runtime_row["row_id"],
        "runtime_policy_id": runtime_row["runtime_policy_id"],
        "batch_index": str(batch_index),
        "precision_policy": runtime_row["precision_policy"],
        "torch_compile_enabled": runtime_row["torch_compile_enabled"],
        "compile_scope": runtime_row["compile_scope"],
        "corruption_strategy": runtime_row["corruption_strategy"],
        "rank": "0",
        "world_size": runtime_row["world_size"],
        "split": split,
        "status": "pass",
        "failure_kind": "",
    }
    for key, value in shared.items():
        if key in row:
            row[key] = value
    if "nonfinite_count" in row:
        row["nonfinite_count"] = "0"
    if "amp_step_skipped" in row:
        row["amp_step_skipped"] = "false"
    if "gate_health_status" in row:
        row["gate_health_status"] = "pass"
    for key in tuple(row):
        if key.endswith("_delta"):
            row[key] = "0.000000"
    if "corruption_version" in row:
        row["corruption_version"] = "test"
        row["profile_name"] = "test"
        row["corruption_view"] = (
            "validation_clean_no_corruption" if split == "validation" else "combined"
        )
        row["corruption_step"] = split
        row["semantic_sample_key_hash"] = f"semantic_{split}"
        row["binary_sample_id_hash"] = f"binary_{split}"
        row["applied_mask_hash"] = f"mask_{split}"
        row["stain_param_hash"] = f"stain_{split}"
        row["noise_std_hash"] = f"noise_std_{split}"
        row["noise_field_hash"] = f"noise_field_{split}"
        row["clean_sample_unchanged_count"] = "25" if split == "validation" else "0"
        row["clean_validation_rng_advanced"] = "false"
    return row


def _with_small_numerical_drift(
    row: Mapping[str, str],
    *,
    candidate_row_id: str,
    failure_kind: str = "dual_t4_numerical_delta_failed",
) -> dict[str, str]:
    if row["candidate_row_id"] != candidate_row_id or row["batch_index"] != "0":
        return dict(row)
    updated = dict(row)
    updated["status"] = "fail"
    updated["failure_kind"] = failure_kind
    updated["kl_loss_abs_delta"] = "0.000013"
    updated["kl_loss_rel_delta"] = "0.000050"
    updated["grad_norm_abs_delta"] = "0.000448"
    updated["grad_norm_rel_delta"] = "0.000068"
    updated["logvar_mean_abs_delta"] = "0.000045"
    updated["logvar_std_abs_delta"] = "0.000229"
    updated["mu_mean_abs_delta"] = "0.000009"
    updated["mu_std_abs_delta"] = "0.000124"
    return updated


def _with_large_numerical_drift(
    row: Mapping[str, str],
    *,
    candidate_row_id: str,
) -> dict[str, str]:
    if row["candidate_row_id"] != candidate_row_id or row["batch_index"] != "0":
        return dict(row)
    updated = dict(row)
    updated["status"] = "fail"
    updated["failure_kind"] = "dual_t4_numerical_delta_failed"
    updated["total_loss_rel_delta"] = "0.500000"
    updated["x_hat_max_abs_delta"] = "0.500000"
    return updated


def _gate_health_rows(
    runtime_rows: tuple[dict[str, str], ...],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for runtime_row in runtime_rows:
        if runtime_row["status"] != "pass":
            continue
        row = dict.fromkeys(GATE_HEALTH_COLUMNS, "0.000000")
        row.update({
            "run_name": RUN_NAME,
            "benchmark_kind": "kaggle_runtime_selection",
            "benchmark_source": "kaggle_runtime_benchmark",
            "full_run_eligible": "true",
            "accelerator_mode": runtime_row["accelerator_mode"],
            "machine_shape": "NvidiaTeslaT4",
            "row_id": f"{runtime_row['row_id']}__gate__encoder_0",
            "candidate_row_id": runtime_row["row_id"],
            "runtime_policy_id": runtime_row["runtime_policy_id"],
            "optimizer_step": "1",
            "module": "encoder.0",
            "gate_kind": "scalar",
            "num_channels": "64",
            "num_elements": "1024",
            **_gate_precision_proof(runtime_row),
            "gate_health_status": "pass",
        })
        rows.append(row)
    return rows


def _gate_precision_proof(runtime_row: dict[str, str]) -> dict[str, str]:
    if runtime_row["precision_policy"] == "amp_scalar_gate_relaxed":
        return {
            "gate_force_fp32": "false",
            "input_dtype": "float16",
            "gate_math_dtype": "float16",
            "gate_tensor_dtype": "float16",
            "output_dtype": "float16",
            "requested_autocast_dtype": runtime_row["autocast_dtype"],
            "precision_proof_status": "pass",
        }
    return {
        "gate_force_fp32": "true",
        "input_dtype": "float16"
        if runtime_row["precision_policy"] != "amp_off_fp32"
        else "float32",
        "gate_math_dtype": "float32",
        "gate_tensor_dtype": "float16"
        if runtime_row["precision_policy"] != "amp_off_fp32"
        else "float32",
        "output_dtype": "float16"
        if runtime_row["precision_policy"] != "amp_off_fp32"
        else "float32",
        "requested_autocast_dtype": runtime_row["autocast_dtype"],
        "precision_proof_status": "pass",
    }


def _write_stain_qa(output_dir: Path, evidence: RuntimeSelectionEvidence) -> None:
    pass_row_ids = [
        row["row_id"] for row in evidence.runtime_rows if row["status"] == "pass"
    ]
    payload = cast(
        "JsonObject",
        {
            "status": "pass",
            "benchmark_kind": "kaggle_runtime_selection",
            "benchmark_source": "kaggle_runtime_benchmark",
            "full_run_eligible": True,
            "candidate_row_ids": pass_row_ids,
            "missing_candidate_row_ids": [],
            "passing_corruption_row_count": len(evidence.corruption_rows),
            "runtime_pass_row_count": len(pass_row_ids),
            "runtime_row_count": len(evidence.runtime_rows),
            "proof_scope": "selected_runtime_stain_corruptor_row_linked_qa",
        },
    )
    write_json(output_dir / "benchmark" / "stain_corruptor_qa.json", payload)


def _row_id(  # noqa: PLR0913
    *,
    accelerator_mode: str,
    batch_size: int,
    precision_policy: str,
    compile_scope: str,
    corruption_strategy: str,
    runtime_policy_id: str = "fp32_eager_default",
) -> str:
    base = (
        f"{accelerator_mode}__bs{batch_size}__{precision_policy}"
        f"__compile_{compile_scope}__{corruption_strategy}"
    )
    if runtime_policy_id in {"", "fp32_eager_default"}:
        return base
    return f"{base}__policy_{runtime_policy_id}"


def _load_json(path: Path) -> dict[str, object]:
    return cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        return [{key: value or "" for key, value in row.items()} for row in reader]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_runtime_environment_stamps_torch_and_cuda_version() -> None:
    """The executor runtime-environment stamps the torch build and CUDA version."""
    environment = runtime_selection_executor._runtime_environment([])  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]

    assert environment["torch_version"] == str(torch.__version__)
    assert environment["cuda_version"] == torch.version.cuda
