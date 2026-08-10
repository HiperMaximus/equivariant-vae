# Copyright 2026 HiperMaximus
"""Tests for the capped real-data runtime pretest scaffold and guard."""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess  # noqa: S404
import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
import torch

from eqvae.benchmarking import real_data_runtime_pretest as pretest
from eqvae.benchmarking.real_data_runtime_pretest import (
    RealDataRuntimePretestRequest,
    write_real_data_runtime_pretest,
)
from eqvae.config import resolve_json_config

if TYPE_CHECKING:
    from eqvae.benchmarking.io import CsvRow, JsonObject
from eqvae.data.synthetic import SyntheticPatchSpec, write_synthetic_patch_shard

_TINY_IMAGE_SIZE = 16
_TINY_CHANNELS = 3
_TINY_TRAIN_PATCHES = 16
_TINY_VALIDATION_PATCHES = 14
_EXPECTED_FILE_HASH_COUNT = 4
_CANONICAL_REAL_TRAIN_PATCHES = 300_000
_CANONICAL_REAL_VALIDATION_PATCHES = 30_000
_CANONICAL_CAP_TRAIN_PATCHES = 8_192
_CANONICAL_CAP_VALIDATION_PATCHES = 2_048
_CANONICAL_WINDOW_PATCHES = 2_048
_CANONICAL_VALIDATION_WINDOW_PATCHES = 1_024
_TEST_GATE_QUANTILE_CAP = 4
_TEST_COMPILED_BATCH_SIZE = 2
_FLOAT_TOLERANCE = 1.0e-6
_RUNTIME_BENCHMARK_CONFIG = Path(
    "configs/spec0001/non_eq_vae_kaggle_runtime_benchmark.json",
)


def test_real_data_runtime_pretest_local_wrong_accelerator_artifacts(
    tmp_path: Path,
) -> None:
    """Local CPU runs write non-promotable artifacts and no selected runtime."""
    repo_root = Path(__file__).resolve().parents[1]
    recommendations_path = write_real_data_runtime_pretest(
        RealDataRuntimePretestRequest(
            config_path=(
                repo_root
                / "configs"
                / "spec0001"
                / "non_eq_vae_kaggle_runtime_benchmark.json"
            ),
            output_dir=tmp_path,
        ),
    )

    benchmark_dir = tmp_path / "benchmark"
    assert recommendations_path == (
        benchmark_dir / "real_data_runtime_pretest_recommendations.json"
    )
    assert not (benchmark_dir / "selected_runtime.json").exists()
    runtime_proof = _load_json(benchmark_dir / "runtime_proof.json")
    recommendations = _load_json(recommendations_path)
    assert runtime_proof["full_run_eligible"] is False
    assert runtime_proof["selection_ready"] is False
    assert runtime_proof["eligible_pass_row_count"] == 0
    assert "real-data identity" in cast("str", runtime_proof["evidence_gate"])
    manifest = _load_json(benchmark_dir / "real_data_runtime_pretest_manifest.json")
    phase_timings = _load_json(benchmark_dir / "phase_timings.json")
    _assert_phase_timings(
        phase_timings,
        required_names={
            "config_resolution",
            "real_data_identity_and_clean_path_proof",
            "stage1_runtime_rows",
            "linked_evidence_payload",
            "write_artifacts",
        },
    )
    assert (
        cast("dict[str, object]", manifest["phase_timings"])["schema_version"]
        == "eqvae.phase_timings.v1"
    )
    assert (
        cast("dict[str, object]", runtime_proof["phase_timings"])["schema_version"]
        == "eqvae.phase_timings.v1"
    )
    assert "phase_timings.json" in cast("list[str]", manifest["artifact_allowlist"])
    assert manifest["real_data_identity_proof_status"] == "skipped_unsupported"
    assert manifest["validation_windows_exercised"] is False
    assert manifest["timed_rows_eligible"] is False
    real_data_proof = cast("dict[str, object]", manifest["real_data_proof"])
    assert real_data_proof["failure_kind"] == "data_root_unavailable"
    assert not real_data_proof["resolved_data_root"]
    diagnostics = cast("dict[str, object]", real_data_proof["data_root_diagnostics"])
    assert diagnostics["requested_data_root"] == "auto"
    assert diagnostics["kaggle_input_exists"] is False
    assert diagnostics["candidate_count"]
    assert diagnostics["accepted_candidates"]
    assert diagnostics["complete_unaccepted_candidate_count"] == 0
    assert "env_value" not in diagnostics
    wrong_accelerator_count = cast("int", runtime_proof["wrong_accelerator_row_count"])
    assert wrong_accelerator_count > 0
    assert recommendations["writes_selected_runtime"] is False
    assert recommendations["status"] == "pretest_skipped"


def test_real_data_runtime_pretest_writes_identity_crc_and_clean_validation_proof(
    tmp_path: Path,
) -> None:
    """A tiny UBC-format root exercises the real-data proof lane locally."""
    repo_root = Path(__file__).resolve().parents[1]
    data_root = _write_tiny_patch_root(tmp_path)
    config_path = _write_tiny_runtime_pretest_config(
        tmp_path=tmp_path,
        repo_root=repo_root,
        data_root=data_root,
    )

    write_real_data_runtime_pretest(
        RealDataRuntimePretestRequest(
            config_path=config_path,
            output_dir=tmp_path / "run",
        ),
    )

    benchmark_dir = tmp_path / "run" / "benchmark"
    manifest = _load_json(benchmark_dir / "real_data_runtime_pretest_manifest.json")
    runtime_proof = _load_json(benchmark_dir / "runtime_proof.json")
    dataloader_rows = _load_csv(benchmark_dir / "dataloader_matrix.csv")
    numerical_rows = _load_csv(benchmark_dir / "numerical_checks.csv")
    corruption_rows = _load_csv(benchmark_dir / "corruption_checks.csv")
    gate_rows = _load_csv(tmp_path / "run" / "metrics" / "gate_health.csv")
    _assert_tiny_manifest_linked_evidence(manifest)
    _assert_tiny_runtime_proof_linked_evidence(runtime_proof)
    _assert_tiny_linked_csv_rows(
        dataloader_rows=dataloader_rows,
        numerical_rows=numerical_rows,
        corruption_rows=corruption_rows,
        gate_rows=gate_rows,
    )
    assert not (benchmark_dir / "selected_runtime.json").exists()


def test_real_data_runtime_pretest_rejects_prefix_only_real_window_contract(
    tmp_path: Path,
) -> None:
    """Canonical real-data configs must keep the locked spread windows."""
    repo_root = Path(__file__).resolve().parents[1]
    data_root = _write_tiny_patch_root(tmp_path)
    config_path = _write_tiny_runtime_pretest_config(
        tmp_path=tmp_path,
        repo_root=repo_root,
        data_root=data_root,
    )
    config = _load_json(config_path)
    data = cast("dict[str, object]", config["data"])
    data["real_train_patch_count"] = _CANONICAL_REAL_TRAIN_PATCHES
    data["real_validation_patch_count"] = _CANONICAL_REAL_VALIDATION_PATCHES
    benchmark_cap = cast("dict[str, object]", data["benchmark_cap"])
    benchmark_cap["train_patch_count"] = _CANONICAL_CAP_TRAIN_PATCHES
    benchmark_cap["validation_patch_count"] = _CANONICAL_CAP_VALIDATION_PATCHES
    benchmark_cap["train_windows"] = [
        {
            "name": "train_prefix_a",
            "start_row": 0,
            "patch_count": _CANONICAL_WINDOW_PATCHES,
        },
        {
            "name": "train_prefix_b",
            "start_row": _CANONICAL_WINDOW_PATCHES,
            "patch_count": _CANONICAL_WINDOW_PATCHES,
        },
        {
            "name": "train_prefix_c",
            "start_row": 2 * _CANONICAL_WINDOW_PATCHES,
            "patch_count": _CANONICAL_WINDOW_PATCHES,
        },
        {
            "name": "train_prefix_d",
            "start_row": 3 * _CANONICAL_WINDOW_PATCHES,
            "patch_count": _CANONICAL_WINDOW_PATCHES,
        },
    ]
    benchmark_cap["validation_windows"] = [
        {
            "name": "validation_prefix_a",
            "start_row": 0,
            "patch_count": _CANONICAL_VALIDATION_WINDOW_PATCHES,
        },
        {
            "name": "validation_prefix_b",
            "start_row": _CANONICAL_VALIDATION_WINDOW_PATCHES,
            "patch_count": _CANONICAL_VALIDATION_WINDOW_PATCHES,
        },
    ]
    config_path.write_text(
        f"{json.dumps(config, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )

    write_real_data_runtime_pretest(
        RealDataRuntimePretestRequest(
            config_path=config_path,
            output_dir=tmp_path / "run_prefix",
        ),
    )

    manifest = _load_json(
        tmp_path
        / "run_prefix"
        / "benchmark"
        / "real_data_runtime_pretest_manifest.json",
    )
    proof = cast("dict[str, object]", manifest["real_data_proof"])
    window_contract = cast("dict[str, object]", proof["window_contract"])
    assert manifest["real_data_identity_proof_status"] == "fail"
    assert window_contract["status"] == "fail"
    assert window_contract["train_windows_match_locked_real_contract"] is False
    assert window_contract["validation_windows_match_locked_real_contract"] is False


def test_real_data_runtime_pretest_rejects_stale_selected_runtime(
    tmp_path: Path,
) -> None:
    """The direct package writer refuses stale selected-runtime artifacts."""
    repo_root = Path(__file__).resolve().parents[1]
    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()
    (benchmark_dir / "selected_runtime.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="selected_runtime"):
        write_real_data_runtime_pretest(
            RealDataRuntimePretestRequest(
                config_path=(
                    repo_root
                    / "configs"
                    / "spec0001"
                    / "non_eq_vae_kaggle_runtime_benchmark.json"
                ),
                output_dir=tmp_path,
            ),
        )


def test_real_data_pretest_push_guard_requires_dataset_confirmation(
    tmp_path: Path,
) -> None:
    """Real-data pretest pushes require explicit dataset attachment approval."""
    repo_root = Path(__file__).resolve().parents[1]
    fake_bin = _fake_bin(tmp_path=tmp_path, repo_root=repo_root)
    kernel_dir = _generated_kernel_dir(
        tmp_path=tmp_path,
        repo_root=repo_root,
        fake_bin=fake_bin,
    )

    completed = subprocess.run(  # noqa: S603
        (
            _required_executable("bash"),
            str(repo_root / "scripts" / "kaggle_kernel.sh"),
            "push",
            str(kernel_dir),
        ),
        cwd=repo_root,
        env=_guard_environment(fake_bin=fake_bin, full_dataset_confirmed=False),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "KAGGLE_FULL_DATASET_CONFIRMED=1" in completed.stderr


def test_real_data_pretest_push_guard_rejects_wrong_dataset_sources(
    tmp_path: Path,
) -> None:
    """The guard rejects missing or drifted real-data source attachments."""
    repo_root = Path(__file__).resolve().parents[1]
    fake_bin = _fake_bin(tmp_path=tmp_path, repo_root=repo_root)
    kernel_dir = _generated_kernel_dir(
        tmp_path=tmp_path,
        repo_root=repo_root,
        fake_bin=fake_bin,
    )
    metadata_path = kernel_dir / "kernel-metadata.json"
    metadata = _load_json(metadata_path)
    metadata["dataset_sources"] = []
    metadata_path.write_text(
        f"{json.dumps(metadata, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )

    completed = subprocess.run(  # noqa: S603
        (
            _required_executable("bash"),
            str(repo_root / "scripts" / "kaggle_kernel.sh"),
            "push",
            str(kernel_dir),
        ),
        cwd=repo_root,
        env=_guard_environment(fake_bin=fake_bin),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "dataset_sources must be exactly" in completed.stderr


def test_real_data_pretest_push_guard_accepts_generated_kernel(
    tmp_path: Path,
) -> None:
    """The positive guard path reaches fake Kaggle without network access."""
    repo_root = Path(__file__).resolve().parents[1]
    fake_bin = _fake_bin(tmp_path=tmp_path, repo_root=repo_root)
    kernel_dir = _generated_kernel_dir(
        tmp_path=tmp_path,
        repo_root=repo_root,
        fake_bin=fake_bin,
    )

    completed = subprocess.run(  # noqa: S603
        (
            _required_executable("bash"),
            str(repo_root / "scripts" / "kaggle_kernel.sh"),
            "push",
            str(kernel_dir),
        ),
        cwd=repo_root,
        env=_guard_environment(fake_bin=fake_bin),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "fake kaggle kernels push" in completed.stdout


def test_real_data_pretest_validate_allows_current_worktree_payload() -> None:
    """Local validate accepts a payload freshly built from the current worktree."""
    repo_root = Path(__file__).resolve().parents[1]
    kernel_dir = "kaggle/kernels/real_data_runtime_pretest"

    build = subprocess.run(  # noqa: S603
        (
            _required_executable("bash"),
            str(repo_root / "scripts" / "kaggle_kernel.sh"),
            "build",
            kernel_dir,
        ),
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    validate = subprocess.run(  # noqa: S603
        (
            _required_executable("bash"),
            str(repo_root / "scripts" / "kaggle_kernel.sh"),
            "validate",
            kernel_dir,
        ),
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert build.returncode == 0, build.stderr
    assert validate.returncode == 0, validate.stderr
    assert "matches current worktree" in validate.stdout


def test_kaggle_pull_guard_requires_remote_confirmation(tmp_path: Path) -> None:
    """Pull is a remote read and refuses even pull-specific approval alone."""
    repo_root = Path(__file__).resolve().parents[1]
    fake_bin = _fake_bin(tmp_path=tmp_path, repo_root=repo_root)

    completed = subprocess.run(  # noqa: S603
        (
            _required_executable("bash"),
            str(repo_root / "scripts" / "kaggle_kernel.sh"),
            "pull",
            "maximusshtefan/eqvae-real-data-runtime-pretest",
            str(tmp_path / "pulled_kernel"),
        ),
        cwd=repo_root,
        env=_guard_environment(
            fake_bin=fake_bin,
            push_confirmed=False,
            full_dataset_confirmed=False,
            pull_confirmed=True,
            remote_confirmed=False,
        ),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "KAGGLE_REMOTE_CONFIRMED=1" in completed.stderr
    assert "fake kaggle" not in completed.stdout


def test_grid_step_scope_is_enumerated_into_row_specs() -> None:
    """The declared "step" scope is enumerated into row specs (S13 effectiveness).

    S13 adds "step" to the grid ``runtime_matrix.compile_scopes``, which the pretest
    enumerates for every seeded candidate, so the S12 selector can eventually see a step
    row. As of S14b the single-GPU pretest also *implements* the whole-step scope, so
    ``_assert_tiny_runtime_proof_linked_evidence`` asserts
    ``implemented_compile_scopes == ['model_forward', 'step']``; scopes outside that set
    (``model_loss``/``train_step_no_optimizer``) still fail closed to
    ``compile_scope_implementation_pending`` (see
    ``test_stage1_admits_single_gpu_step_row_and_rejects_unimplemented_scope``).
    """
    config_path = Path("configs/spec0001/non_eq_vae_kaggle_runtime_benchmark.json")
    settings = pretest._settings(  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
        resolve_json_config(config_path),
        data_root_override=None,
    )
    assert "step" in settings.compile_scopes
    specs = pretest._stage1_row_specs(  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
        settings,
    )
    assert any(spec.compile_scope == "step" for spec in specs), (
        "step must be enumerated for the seeded candidates"
    )


def test_train_step_target_rows_prioritize_eager_before_compiled() -> None:
    """Candidate evidence spends coverage on eager smaller batches first."""
    rows = [
        _train_step_target_row(
            row_id="compiled_bs4",
            batch_size=4,
            compile_scope="model_forward",
        ),
        _train_step_target_row(
            row_id="eager_bs12",
            batch_size=12,
            compile_scope="none",
        ),
        _train_step_target_row(
            row_id="eager_bs4",
            batch_size=4,
            compile_scope="none",
        ),
        _train_step_target_row(
            row_id="compiled_bs8",
            batch_size=8,
            compile_scope="model_forward",
        ),
        _train_step_target_row(
            row_id="eager_bs8",
            batch_size=8,
            compile_scope="none",
        ),
    ]

    ordered = pretest._unique_train_step_target_rows(rows)  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001

    assert [row["row_id"] for row in ordered] == [
        "eager_bs4",
        "eager_bs8",
        "eager_bs12",
        "compiled_bs4",
        "compiled_bs8",
    ]


def _first_single_visible_step_row_spec(
    settings: pretest.RealDataRuntimePretestSettings,
) -> pretest.RowSpec:
    for spec in pretest._stage1_row_specs(settings):  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
        if (
            spec.compile_scope == "step"
            and spec.accelerator_mode == "single_visible_t4"
        ):
            return spec
    message = "no single-GPU step row spec was enumerated from the grid"
    raise AssertionError(message)


def test_stage1_row_specs_enable_cudnn_benchmark_by_default() -> None:
    """Grid-enumerated pre-screen rows carry the speed-first cuDNN default (S17f).

    The single-GPU pre-screen times each row (and records what it applied) under the
    same cuDNN autotuning the dual-T4 search and the run use; the grid config sets no
    cuDNN axis, so the enumerated ``RowSpec`` defaults to ``benchmark=True`` and
    ``deterministic=False``.
    """
    settings = pretest._settings(  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
        resolve_json_config(_RUNTIME_BENCHMARK_CONFIG),
        data_root_override=None,
    )

    row_spec = _first_single_visible_step_row_spec(settings)

    assert row_spec.cudnn_benchmark is True
    assert row_spec.cudnn_deterministic is False


def test_model_for_compile_scope_leaves_whole_step_uncompiled() -> None:
    """The whole-step scope returns the model uncompiled; model_forward compiles it.

    The whole-step recipe compiles the train-*step* closure (``_build_compiled_step``),
    not the model object, so ``_model_for_compile_scope_name`` must return a step row's
    model untouched -- otherwise the paired numerical proof
    (``_one_strategy_train_step_evidence``) would run a compiled model whose
    ``FastpathStepOutput`` cannot emit the mu/logvar/hash telemetry that lane records.
    The ``model_forward`` case proves the pass-through is scope-specific: a mutation
    that also routed ``step`` into ``torch.compile`` would fail the identity check.
    """
    from eqvae.models.registry import (  # noqa: PLC0415
        MODEL_KIND_NON_EQ_TRANSLATABLE,
        build_model,
    )

    settings = pretest._settings(  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
        resolve_json_config(_RUNTIME_BENCHMARK_CONFIG),
        data_root_override=None,
    )
    model = build_model(
        MODEL_KIND_NON_EQ_TRANSLATABLE,
        model_config={"norm_groups": settings.norm_groups},
    )

    step_result = pretest._model_for_compile_scope_name(  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
        model=model,
        compile_scope="step",
    )
    forward_result = pretest._model_for_compile_scope_name(  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
        model=model,
        compile_scope="model_forward",
    )

    assert step_result is model
    assert forward_result is not model


@pytest.mark.parametrize(
    ("precision_policy", "autocast_dtype", "grad_scaler_enabled", "expected_dtype"),
    [
        ("amp_off_fp32", "float32", False, torch.float32),
        ("amp_conservative", "float16", True, torch.float16),
    ],
)
def test_build_compiled_step_wires_the_measured_recipe(
    monkeypatch: pytest.MonkeyPatch,
    precision_policy: str,
    autocast_dtype: str,
    grad_scaler_enabled: object,
    expected_dtype: torch.dtype,
) -> None:
    """_build_compiled_step threads fp32 and AMP recipes into the compiled step.

    Spies on the three recipe seams (dynamo config, the step-fn factory, torch.compile)
    prove the single-GPU builder wires the row's knobs and the requested autocast,
    returns the model unwrapped, and compiles ``dynamic=False`` on inductor. The AMP
    case replaces the stale rejection contract: deleting row-derived autocast now
    makes the float16 case fail instead of blessing a silently measured fp32 recipe.
    CPU-safe: ``torch.compile`` and the factory are stubbed, so nothing is traced.
    """
    from eqvae.corruption.stain import profile_from_config  # noqa: PLC0415
    from eqvae.models.registry import (  # noqa: PLC0415
        MODEL_KIND_NON_EQ_TRANSLATABLE,
        build_model,
    )
    from eqvae.training import fastpath_recipe, fastpath_step  # noqa: PLC0415

    settings = pretest._settings(  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
        resolve_json_config(_RUNTIME_BENCHMARK_CONFIG),
        data_root_override=None,
    )
    step_spec = replace(
        _first_single_visible_step_row_spec(settings),
        precision_policy=precision_policy,
        autocast_dtype=autocast_dtype,
        grad_scaler_enabled=bool(grad_scaler_enabled),
    )
    model = build_model(
        MODEL_KIND_NON_EQ_TRANSLATABLE,
        model_config={"norm_groups": settings.norm_groups},
    )
    profile = profile_from_config(settings.corruption_config)

    dynamo_calls: list[dict[str, object]] = []
    make_calls: list[dict[str, object]] = []
    compile_calls: list[tuple[object, object, object]] = []

    def spy_dynamo(**kwargs: object) -> None:
        dynamo_calls.append(kwargs)

    def spy_make(
        model_arg: object,
        corruptor_arg: object,
        *,
        ssim_weight: float,
        autocast_dtype: object,
        autocast_enabled: bool,
    ) -> str:
        make_calls.append({
            "model": model_arg,
            "corruptor": corruptor_arg,
            "ssim_weight": ssim_weight,
            "autocast_dtype": autocast_dtype,
            "autocast_enabled": autocast_enabled,
        })
        return "step_fn_sentinel"

    def spy_compile(fn: object, *, dynamic: object, backend: object) -> str:
        compile_calls.append((fn, dynamic, backend))
        return "compiled_sentinel"

    monkeypatch.setattr(fastpath_recipe, "apply_fastpath_dynamo_config", spy_dynamo)
    monkeypatch.setattr(fastpath_step, "make_fastpath_step_fn", spy_make)
    monkeypatch.setattr(torch, "compile", spy_compile)

    returned_model, optimizer, compiled_step_fn = pretest._build_compiled_step(  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
        raw_model=model,
        device=torch.device("cpu"),
        profile=profile,
        settings=settings,
        row_spec=step_spec,
    )

    assert returned_model is model
    assert isinstance(optimizer, torch.optim.AdamW)
    assert compiled_step_fn == "compiled_sentinel"
    assert dynamo_calls == [
        {
            "optimize_ddp": step_spec.optimize_ddp,
            "compiled_autograd": step_spec.compiled_autograd,
            "reorder_compute_comm_overlap": step_spec.reorder_compute_comm_overlap,
        },
    ]
    assert len(make_calls) == 1
    assert make_calls[0]["model"] is model
    assert make_calls[0]["autocast_enabled"] is bool(grad_scaler_enabled)
    assert make_calls[0]["autocast_dtype"] is expected_dtype
    assert make_calls[0]["ssim_weight"] == settings.ssim_weight
    assert compile_calls == [
        ("step_fn_sentinel", False, pretest._STEP_COMPILE_BACKEND),  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
    ]


def test_run_compiled_step_batch_forwards_amp_scaler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The compiled pretest step forwards its persistent scaler and AMP policy.

    Builder-level autocast proof is insufficient if the driver later performs an
    ordinary optimizer step. This spy guards the second A2 seam: the exact scaler built
    by the child reaches the shared scale/unscale/clip/update helper, its skip result is
    returned to telemetry, and the row's enabled flag is not replaced by a constant.
    """
    from types import SimpleNamespace  # noqa: PLC0415

    from eqvae.data.training_batches import PatchTrainingBatch  # noqa: PLC0415
    from eqvae.models.registry import (  # noqa: PLC0415
        MODEL_KIND_NON_EQ_TRANSLATABLE,
        build_model,
    )

    settings = pretest._settings(  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
        resolve_json_config(_RUNTIME_BENCHMARK_CONFIG),
        data_root_override=None,
    )
    row_spec = replace(
        _first_single_visible_step_row_spec(settings),
        precision_policy="amp_conservative",
        autocast_dtype="float16",
        grad_scaler_enabled=True,
    )
    model = build_model(
        MODEL_KIND_NON_EQ_TRANSLATABLE,
        model_config={"norm_groups": settings.norm_groups},
    )
    optimizer = torch.optim.AdamW(model.parameters())
    scaler = cast("torch.amp.GradScaler", object())
    calls: list[dict[str, object]] = []

    def fake_optimizer_step(**kwargs: object) -> bool:
        calls.append(kwargs)
        return True

    monkeypatch.setattr(pretest, "run_fastpath_optimizer_step", fake_optimizer_step)
    batch = PatchTrainingBatch(
        images_uint8=torch.zeros(
            (
                _TEST_COMPILED_BATCH_SIZE,
                settings.channels,
                settings.image_size,
                settings.image_size,
            ),
            dtype=torch.uint8,
        ),
        split="train",
        file_indices=(),
        row_indices=(),
        wsi_ids=(),
        labels=(),
        xs=(),
        ys=(),
        semantic_sample_keys=(),
        sample_ids=(),
    )
    output = SimpleNamespace(loss=torch.tensor(1.0))

    def fake_compiled_step(*args: torch.Tensor) -> object:
        del args
        return output

    def fake_beta(
        *,
        optimizer_step_index: int,
        max_optimizer_steps: int,
        target_beta: float,
        warmup_fraction: float,
    ) -> float:
        del optimizer_step_index, max_optimizer_steps, target_beta, warmup_fraction
        return 0.0

    batch_size, skipped = pretest._run_compiled_step_batch(  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
        iterator=iter([batch]),
        compiled_step_fn=fake_compiled_step,
        optimizer=optimizer,
        scaler=scaler,
        model=model,
        device=torch.device("cpu"),
        settings=settings,
        step_index=0,
        row_spec=row_spec,
        latent_channels=model.latent_channels,
        beta_for_step_fn=fake_beta,
    )

    assert batch_size == _TEST_COMPILED_BATCH_SIZE
    assert skipped is True
    assert len(calls) == 1
    assert calls[0]["scaler"] is scaler
    assert calls[0]["grad_scaler_enabled"] is True
    assert calls[0]["optimizer"] is optimizer


def test_step_compile_backend_matches_dual_t4_executor() -> None:
    """The pretest's compiled-step backend equals the dual-T4 executor's (S14b).

    Both the single-GPU pre-screen and the dual-T4 measurement must compile the step
    under the same backend the generator bakes into the plan
    (``_selected_runtime_payload``: any compiled scope -> ``"inductor"``), or the
    stability screen would exercise a different backend than the measured/consumed one.
    """
    from eqvae.benchmarking.runtime_selection_executor import (  # noqa: PLC0415
        _STEP_COMPILE_BACKEND as _EXECUTOR_STEP_COMPILE_BACKEND,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    )

    assert pretest._STEP_COMPILE_BACKEND == "inductor"  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
    assert (
        _EXECUTOR_STEP_COMPILE_BACKEND == pretest._STEP_COMPILE_BACKEND  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
    )


def test_unique_train_step_target_rows_admits_step_scope() -> None:
    """The candidate train-step evidence filter admits whole-step rows (S14b).

    Before S14b the filter dropped every scope but ``none``/``model_forward``, so a step
    row could never be joined to its numerical evidence. Eager rows still sort first.
    """
    rows = [
        _train_step_target_row(row_id="eager_bs8", batch_size=8, compile_scope="none"),
        _train_step_target_row(row_id="step_bs8", batch_size=8, compile_scope="step"),
    ]

    ordered = pretest._unique_train_step_target_rows(rows)  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001

    scopes = [row["compile_scope"] for row in ordered]
    assert "step" in scopes
    assert ordered[0]["compile_scope"] == "none"


def test_compile_evidence_pass_treats_step_like_model_forward() -> None:
    """A whole-step row passes compile evidence like ``model_forward`` (S14b).

    A step row clears the lane exactly when the shared ``compile_settle`` lane passes
    and it recorded zero post-settle graph breaks/recompiles; a graph break or a
    non-pass settle lane fails it -- identical to ``model_forward``.
    """
    settle_pass = cast("JsonObject", {"compile_settle": {"status": "pass"}})
    clean_step_row = cast(
        "CsvRow",
        {
            **_train_step_target_row(
                row_id="step_bs8",
                batch_size=8,
                compile_scope="step",
            ),
            "graph_break_count": "0",
            "recompile_count": "0",
        },
    )

    assert (
        pretest._compile_evidence_pass_for_row(  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
            row=clean_step_row,
            linked_evidence=settle_pass,
        )
        is True
    )
    broken_step_row = cast("CsvRow", {**clean_step_row, "graph_break_count": "1"})
    assert (
        pretest._compile_evidence_pass_for_row(  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
            row=broken_step_row,
            linked_evidence=settle_pass,
        )
        is False
    )
    recompiled_step_row = cast("CsvRow", {**clean_step_row, "recompile_count": "1"})
    assert (
        pretest._compile_evidence_pass_for_row(  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
            row=recompiled_step_row,
            linked_evidence=settle_pass,
        )
        is False
    )
    settle_fail = cast(
        "JsonObject",
        {"compile_settle": {"status": "skipped_unsupported"}},
    )
    assert (
        pretest._compile_evidence_pass_for_row(  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
            row=clean_step_row,
            linked_evidence=settle_fail,
        )
        is False
    )


def test_stage1_admits_single_gpu_step_row_and_rejects_unimplemented_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage 1 runs a single-GPU step row and still fails closed on other scopes (S14b).

    The widened ``compile_scope_implementation_pending`` guard admits ``step`` (now
    implemented single-GPU) while still screening out the scopes that are not
    (``model_loss``). ``_run_single_child_row`` is stubbed so the assertion is CPU-safe.
    """
    settings = pretest._settings(  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
        resolve_json_config(_RUNTIME_BENCHMARK_CONFIG),
        data_root_override=None,
    )
    specs = pretest._stage1_row_specs(settings)  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
    step_spec = _first_single_visible_step_row_spec(settings)
    pending_spec = next(
        spec
        for spec in specs
        if spec.compile_scope == "model_loss"
        and spec.accelerator_mode == "single_visible_t4"
    )

    reached: list[str] = []

    def fake_child(config: pretest.ChildRowConfig) -> CsvRow:
        reached.append(config.row_spec.row_id)
        return pretest._base_row(settings=settings, row_spec=config.row_spec)  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001

    monkeypatch.setattr(pretest, "_run_single_child_row", fake_child)

    rows = pretest._run_stage1_rows(  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
        request=RealDataRuntimePretestRequest(
            config_path=_RUNTIME_BENCHMARK_CONFIG,
            output_dir=tmp_path,
        ),
        settings=settings,
        row_specs=(step_spec, pending_spec),
        phase_timings=pretest.PhaseTimingRecorder(),
    )

    assert step_spec.row_id in reached
    assert pending_spec.row_id not in reached
    pending_row = next(row for row in rows if row["row_id"] == pending_spec.row_id)
    assert pending_row["failure_kind"] == "compile_scope_implementation_pending"


def test_gate_quantiles_use_exact_small_tensor_path() -> None:
    """Small gate tensors keep exact torch.quantile telemetry."""
    tensor = torch.tensor([0.0, 1.0, 2.0, 4.0], dtype=torch.float32)

    observed = pretest._tensor_quantile(tensor, 0.50)  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
    expected = float(torch.quantile(tensor.flatten(), 0.50).item())

    assert abs(observed - expected) <= _FLOAT_TOLERANCE


def test_gate_quantiles_sample_large_tensor_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Large gate quantiles are deterministic and bounded without huge tensors."""
    monkeypatch.setattr(
        pretest,
        "MAX_GATE_QUANTILE_ELEMENTS",
        _TEST_GATE_QUANTILE_CAP,
    )
    tensor = torch.arange(10, dtype=torch.float32)

    sampled = pretest._gate_quantile_values(tensor)  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
    first = pretest._tensor_quantile(tensor, 0.50)  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
    second = pretest._tensor_quantile(tensor, 0.50)  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001

    assert sampled.numel() == _TEST_GATE_QUANTILE_CAP
    assert torch.equal(sampled, torch.tensor([0.0, 3.0, 6.0, 9.0]))
    assert abs(first - float(torch.quantile(sampled, 0.50).item())) <= _FLOAT_TOLERANCE
    assert second == first


def test_gate_health_lane_pass_does_not_cover_missing_candidate_rows() -> None:
    """A lane-level pass cannot make uncovered runtime rows gate-health pass."""
    covered = _train_step_target_row(
        row_id="single_visible_t4__bs4__amp_off_fp32__compile_none__branchless_all",
        batch_size=4,
        compile_scope="none",
    )
    uncovered = _train_step_target_row(
        row_id="single_visible_t4__bs8__amp_off_fp32__compile_none__branchless_all",
        batch_size=8,
        compile_scope="none",
    )
    rows = [covered, uncovered]
    linked_evidence = cast(
        "JsonObject",
        {
            "ddp_launch": {"status": "pass"},
            "compile_settle": {"status": "skipped_unsupported"},
            "dataloader_throughput": {
                "status": "pass",
                "rows": [
                    _passing_dataloader_row(row=row, split=split)
                    for row in rows
                    for split in ("train", "validation")
                ],
            },
            "paired_numerical": {
                "status": "pass",
                "rows": [{"row_id": row["row_id"], "status": "pass"} for row in rows],
            },
            "corruption_equivalence": {
                "status": "pass",
                "rows": [{"row_id": row["row_id"], "status": "pass"} for row in rows],
            },
            "gate_health": {
                "status": "pass",
                "rows": [],
                "row_statuses": [
                    {
                        "row_id": covered["row_id"],
                        "accelerator_mode": covered["accelerator_mode"],
                        "world_size": int(covered["world_size"]),
                        "per_device_batch_size": int(covered["per_device_batch_size"]),
                        "precision_policy": covered["precision_policy"],
                        "compile_scope": covered["compile_scope"],
                        "status": "pass",
                    },
                ],
            },
        },
    )
    data_proof = cast(
        "JsonObject",
        {
            "identity_status": "pass",
            "row_count_status": "pass",
            "crc_validation_status": "pass",
            "window_status": "pass",
            "clean_validation_dataloader_status": "pass",
        },
    )

    updated = pretest._rows_with_linked_evidence(  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
        rows=rows,
        data_proof=data_proof,
        linked_evidence=linked_evidence,
    )
    by_id = {row["row_id"]: row for row in updated}

    assert by_id[covered["row_id"]]["status"] == "pass"
    assert by_id[covered["row_id"]]["gate_health_status"] == "pass"
    assert by_id[uncovered["row_id"]]["status"] == "ineligible"
    assert by_id[uncovered["row_id"]]["gate_health_status"] == "skipped_unsupported"
    assert by_id[uncovered["row_id"]]["failure_kind"] == (
        "gate_health_evidence_not_row_pass"
    )


def test_train_step_evidence_failure_preserves_candidate_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All-failed candidate evidence returns proof diagnostics instead of raising."""
    repo_root = Path(__file__).resolve().parents[1]
    data_root = _write_tiny_patch_root(tmp_path)
    config_path = _write_tiny_runtime_pretest_config(
        tmp_path=tmp_path,
        repo_root=repo_root,
        data_root=data_root,
    )
    settings = pretest._settings(  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
        resolve_json_config(config_path),
        data_root_override=None,
    )
    rows = [
        _train_step_target_row(
            row_id="single_visible_t4__bs4__amp_off_fp32__compile_none__branchless_all",
            batch_size=4,
            compile_scope="none",
        ),
        _train_step_target_row(
            row_id="single_visible_t4__bs4__amp_off_fp32__compile_none__indexed_masked",
            batch_size=4,
            compile_scope="none",
            corruption_strategy="indexed_masked",
        ),
    ]

    def fail_fixed_batch(*_args: object, **kwargs: object) -> object:
        target_row = cast("dict[str, str]", kwargs["target_row"])
        raise pretest._CandidateTrainStepEvidenceError(  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
            strategy_attempt="indexed_masked",
            target_corruption_strategy=target_row["corruption_strategy"],
            cause=RuntimeError("synthetic candidate boom"),
        )

    monkeypatch.setattr(
        pretest,
        "_paired_fixed_batch_train_step_evidence",
        fail_fixed_batch,
    )

    evidence = pretest._paired_train_step_evidence(  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
        settings=settings,
        data_proof={"identity_status": "local_pass"},
        rows=rows,
    )
    numerical = pretest._paired_numerical_proof(  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
        settings=settings,
        data_proof={"identity_status": "local_pass"},
        rows=rows,
        train_step_evidence=evidence,
    )
    corruption = pretest._corruption_equivalence_proof(  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
        settings=settings,
        data_proof={"identity_status": "local_pass"},
        rows=rows,
        train_step_evidence=evidence,
    )

    assert evidence["status"] == "fail"
    assert evidence["candidate_evidence_count"] == 0
    assert evidence["failed_candidate_evidence_count"] == 1
    failed = cast("list[dict[str, object]]", evidence["failed_candidate_evidence"])
    assert failed[0]["strategy_attempt"] == "indexed_masked"
    assert failed[0]["target_corruption_strategy"] == "branchless_all"
    assert failed[0]["failure_message_excerpt"] == "synthetic candidate boom"
    assert set(cast("list[str]", failed[0]["affected_row_ids"])) == {
        "single_visible_t4__bs4__amp_off_fp32__compile_none__branchless_all",
        "single_visible_t4__bs4__amp_off_fp32__compile_none__indexed_masked",
    }
    assert numerical["status"] == "fail"
    assert numerical["failed_candidate_evidence_count"] == 1
    numerical_failed = cast(
        "list[dict[str, object]]",
        numerical["failed_candidate_evidence"],
    )
    assert numerical_failed[0]["failure_message_excerpt"] == "synthetic candidate boom"
    assert corruption["status"] == "fail"
    assert corruption["failed_candidate_evidence_count"] == 1
    corruption_failed = cast(
        "list[dict[str, object]]",
        corruption["failed_candidate_evidence"],
    )
    assert corruption_failed[0]["failure_message_excerpt"] == "synthetic candidate boom"


def _generated_kernel_dir(
    *,
    tmp_path: Path,
    repo_root: Path,
    fake_bin: Path,
) -> Path:
    kernel_source = repo_root / "kaggle" / "kernels" / "real_data_runtime_pretest"
    kernel_dir = tmp_path / "real_data_runtime_pretest_kernel"
    kernel_dir.mkdir()
    shutil.copy2(kernel_source / "kernel-metadata.json", kernel_dir)
    subprocess.run(  # noqa: S603
        (
            sys.executable,
            str(repo_root / "scripts" / "build_kaggle_embedded_kernel.py"),
            "--repo-root",
            str(repo_root),
            "--kernel-dir",
            str(kernel_dir),
            "--template",
            str(kernel_source / "run_template.py"),
            "--ready-marker",
            "KAGGLE_REAL_DATA_RUNTIME_PRETEST_READY = True",
        ),
        cwd=repo_root,
        env=_guard_environment(
            fake_bin=fake_bin,
            push_confirmed=False,
            full_dataset_confirmed=False,
        ),
        check=True,
    )
    return kernel_dir


def _fake_bin(*, tmp_path: Path, repo_root: Path) -> Path:
    fake_bin = tmp_path / "fake_bin"
    fake_bin.mkdir(exist_ok=True)
    commit = subprocess.run(  # noqa: S603
        (_required_executable("git"), "rev-parse", "HEAD"),
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (fake_bin / "git").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'if [[ "$1" == "rev-parse" && "${2:-}" == "HEAD" ]]; then\n'
        f"  printf '%s\\n' '{commit}'\n"
        'elif [[ "$1" == "status" && "${2:-}" == "--short" ]]; then\n'
        "  exit 0\n"
        "else\n"
        '  command git "$@"\n'
        "fi\n",
        encoding="utf-8",
    )
    (fake_bin / "kaggle").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\nprintf 'fake kaggle %s\\n' \"$*\"\n",
        encoding="utf-8",
    )
    (fake_bin / "git").chmod(0o755)
    (fake_bin / "kaggle").chmod(0o755)
    return fake_bin


def _required_executable(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        message = f"missing executable: {name}"
        raise RuntimeError(message)
    return path


def _guard_environment(
    *,
    fake_bin: Path,
    push_confirmed: bool = True,
    full_dataset_confirmed: bool = True,
    pull_confirmed: bool = False,
    remote_confirmed: bool = False,
) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PATH"] = f"{fake_bin}{os.pathsep}{environment['PATH']}"
    environment["KAGGLE_DISABLE_FRESH_OAUTH"] = "1"
    if push_confirmed:
        environment["KAGGLE_PUSH_CONFIRMED"] = "1"
    else:
        environment.pop("KAGGLE_PUSH_CONFIRMED", None)
    if full_dataset_confirmed:
        environment["KAGGLE_FULL_DATASET_CONFIRMED"] = "1"
    else:
        environment.pop("KAGGLE_FULL_DATASET_CONFIRMED", None)
    if pull_confirmed:
        environment["KAGGLE_PULL_CONFIRMED"] = "1"
    else:
        environment.pop("KAGGLE_PULL_CONFIRMED", None)
    if remote_confirmed:
        environment["KAGGLE_REMOTE_CONFIRMED"] = "1"
    else:
        environment.pop("KAGGLE_REMOTE_CONFIRMED", None)
    return environment


def _train_step_target_row(
    *,
    row_id: str,
    batch_size: int,
    compile_scope: str,
    corruption_strategy: str = "branchless_all",
) -> CsvRow:
    return {
        "row_id": row_id,
        "accelerator_mode": "single_visible_t4",
        "world_size": "1",
        "per_device_batch_size": str(batch_size),
        "precision_policy": "amp_off_fp32",
        "compile_scope": compile_scope,
        "corruption_strategy": corruption_strategy,
        "status": "ineligible",
    }


def _passing_dataloader_row(*, row: CsvRow, split: str) -> CsvRow:
    return {
        "accelerator_mode": row["accelerator_mode"],
        "world_size": row["world_size"],
        "batch_size": row["per_device_batch_size"],
        "split": split,
        "status": "pass",
        "data_wait_fraction_p95": "0.010000",
        "loader_samples_sec": "100.000000",
        "trainer_samples_sec": "10.000000",
    }


def _load_json(path: Path) -> dict[str, object]:
    return cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def _assert_tiny_manifest_linked_evidence(manifest: dict[str, object]) -> None:
    assert manifest["real_data_identity_proof_status"] == "local_pass"
    assert manifest["row_count_proof_status"] == "pass"
    assert manifest["crc_validation_status"] == "pass"
    assert manifest["train_windows_exercised"] is True
    assert manifest["validation_windows_exercised"] is True
    assert manifest["linked_evidence_status"] == "skipped_unsupported"
    assert _object_status(manifest, "compile_settle_proof") == "skipped_unsupported"
    assert (
        _object_field(manifest, "compile_settle_proof", "contract_status")
        == "local_pass"
    )
    assert _object_status(manifest, "ddp_launch_proof") == "skipped_unsupported"
    assert _object_field(manifest, "ddp_launch_proof", "contract_status") == (
        "local_pass"
    )
    assert _object_status(manifest, "dataloader_throughput_proof") == "local_pass"
    assert _object_status(manifest, "paired_numerical_proof") == "local_pass"
    assert (
        _object_field(manifest, "paired_numerical_proof", "candidate_row_specific")
        is False
    )
    assert _object_field(manifest, "paired_numerical_proof", "candidate_evidence_count")
    assert (
        _object_field(
            manifest,
            "paired_numerical_proof",
            "failed_candidate_evidence_count",
        )
        == 0
    )
    assert _object_status(manifest, "corruption_equivalence_proof") == "local_pass"
    assert _object_field(
        manifest,
        "corruption_equivalence_proof",
        "candidate_evidence_count",
    )
    assert (
        _object_field(
            manifest,
            "corruption_equivalence_proof",
            "failed_candidate_evidence_count",
        )
        == 0
    )
    assert (
        _object_field(
            manifest,
            "corruption_equivalence_proof",
            "clean_validation_rng_status",
        )
        == "not_exercised_training_batch_only"
    )
    assert _object_status(manifest, "gate_health_proof") == "local_pass"
    assert (
        len(cast("list[object]", manifest["file_hashes"])) == _EXPECTED_FILE_HASH_COUNT
    )
    _assert_tiny_clean_validation_proof(
        cast("dict[str, object]", manifest["clean_validation_dataloader_proof"]),
    )
    _assert_tiny_real_data_proof(
        cast("dict[str, object]", manifest["real_data_proof"]),
    )


def _assert_tiny_clean_validation_proof(clean_proof: dict[str, object]) -> None:
    assert clean_proof["status"] == "pass"
    assert clean_proof["dataset_class"] == "PatchTrainingDataset"
    assert clean_proof["collate_fn"] == "collate_patch_training_samples"
    assert clean_proof["normalizer"] == "normalize_uint8_batch"
    assert clean_proof["corruption_called"] is False
    assert clean_proof["proof_scope"] == "validation_loader_clean_input_only"
    assert clean_proof["corruption_rng_instrumented"] is False
    assert clean_proof["clean_validation_rng_status"] == (
        "not_exercised_in_this_loader_lane"
    )
    assert clean_proof["clean_validation_rng_consumed"] is None
    assert clean_proof["sample_count"] == _TINY_VALIDATION_PATCHES
    assert clean_proof["partial_batch_observed"] is True


def _assert_tiny_real_data_proof(proof: dict[str, object]) -> None:
    splits = cast("dict[str, object]", proof["splits"])
    train = cast("dict[str, object]", splits["train"])
    validation = cast("dict[str, object]", splits["validation"])
    assert train["csv_row_count"] == _TINY_TRAIN_PATCHES
    assert validation["csv_row_count"] == _TINY_VALIDATION_PATCHES
    assert cast("dict[str, object]", train["windows"])["selected_patch_count"] == (
        _TINY_TRAIN_PATCHES
    )
    assert (
        cast("dict[str, object]", validation["windows"])["selected_patch_count"]
        == _TINY_VALIDATION_PATCHES
    )
    assert proof["status"] == "local_pass"
    assert cast("dict[str, object]", proof["window_contract"])["status"] == (
        "local_pass"
    )


def _assert_tiny_runtime_proof_linked_evidence(
    runtime_proof: dict[str, object],
) -> None:
    assert runtime_proof["real_data_identity_proof_status"] == "local_pass"
    assert runtime_proof["clean_validation_dataloader_status"] == "pass"
    assert runtime_proof["linked_evidence_status"] == "skipped_unsupported"
    compile_policy = cast("dict[str, object]", runtime_proof["compile_settle_policy"])
    assert compile_policy["implemented_in_this_runner"] is True
    assert compile_policy["implemented_compile_scopes"] == ["model_forward", "step"]
    assert compile_policy["contract_proof_available"] is True
    assert compile_policy["status"] == "skipped_unsupported"
    assert runtime_proof["paired_numerical_status"] == "local_pass"
    assert runtime_proof["corruption_equivalence_status"] == "local_pass"
    assert cast("int", runtime_proof["paired_numerical_candidate_evidence_count"]) >= 1
    assert runtime_proof["paired_numerical_failed_candidate_evidence_count"] == 0
    assert (
        cast("int", runtime_proof["corruption_equivalence_candidate_evidence_count"])
        >= 1
    )
    assert runtime_proof["corruption_equivalence_failed_candidate_evidence_count"] == 0
    assert runtime_proof["gate_health_status"] == "local_pass"
    assert runtime_proof["ddp_launch_status"] == "skipped_unsupported"
    assert "real-data identity" in cast("str", runtime_proof["evidence_gate"])


def _assert_phase_timings(
    payload: dict[str, object],
    *,
    required_names: set[str],
) -> None:
    assert payload["schema_version"] == "eqvae.phase_timings.v1"
    assert payload["recorded_phase_count"] == len(
        cast("list[object]", payload["phases"]),
    )
    assert cast("float", payload["total_elapsed_sec"]) >= 0.0
    phases = [
        cast("dict[str, object]", item)
        for item in cast("list[object]", payload["phases"])
    ]
    names = {cast("str", phase["name"]) for phase in phases}
    assert required_names.issubset(names)
    for phase in phases:
        assert phase["status"] in {"pass", "fail"}
        assert cast("float", phase["elapsed_sec"]) >= 0.0
        assert phase["started_at_utc"]
        assert phase["finished_at_utc"]


def _assert_tiny_linked_csv_rows(
    *,
    dataloader_rows: list[dict[str, str]],
    numerical_rows: list[dict[str, str]],
    corruption_rows: list[dict[str, str]],
    gate_rows: list[dict[str, str]],
) -> None:
    validation_rows = [row for row in dataloader_rows if row["split"] == "validation"]
    train_rows = [row for row in dataloader_rows if row["split"] == "train"]
    assert len(validation_rows) == 1
    assert len(train_rows) == 1
    assert validation_rows[0]["status"] == "local_pass"
    assert train_rows[0]["status"] == "local_pass"
    validation_measured = int(validation_rows[0]["rank_sample_count"])
    assert 0 < validation_measured <= _TINY_VALIDATION_PATCHES
    assert numerical_rows
    assert all(row["status"] == "skipped_unsupported" for row in numerical_rows)
    assert all(
        row["failure_kind"] == "compile_or_ddp_numerical_pending"
        for row in numerical_rows
    )
    assert corruption_rows
    assert all(row["status"] == "skipped_unsupported" for row in corruption_rows)
    assert all(not row["clean_validation_rng_advanced"] for row in corruption_rows)
    assert gate_rows
    assert all(row["gate_health_status"] == "local_pass" for row in gate_rows)


def _object_status(payload: dict[str, object], key: str) -> object:
    return cast("dict[str, object]", payload[key])["status"]


def _object_field(payload: dict[str, object], key: str, field: str) -> object:
    return cast("dict[str, object]", payload[key])[field]


def _write_tiny_patch_root(tmp_path: Path) -> Path:
    dataset_root = tmp_path / "patches-pre-shuffled-ubc-ocean" / "dataset"
    write_synthetic_patch_shard(
        bin_path=dataset_root / "ubc_train_shuffled.bin",
        csv_path=dataset_root / "ubc_train_shuffled.csv",
        spec=SyntheticPatchSpec(
            count=_TINY_TRAIN_PATCHES,
            image_size=_TINY_IMAGE_SIZE,
            channels=_TINY_CHANNELS,
            seed=20260619,
        ),
        include_idx=False,
    )
    write_synthetic_patch_shard(
        bin_path=dataset_root / "ubc_ocean_valid.bin",
        csv_path=dataset_root / "ubc_ocean_valid.csv",
        spec=SyntheticPatchSpec(
            count=_TINY_VALIDATION_PATCHES,
            image_size=_TINY_IMAGE_SIZE,
            channels=_TINY_CHANNELS,
            seed=20260620,
        ),
        include_idx=True,
    )
    validation_csv = dataset_root / "ubc_ocean_valid.csv"
    validation_csv.write_text(
        validation_csv.read_text(encoding="utf-8").replace(
            "synthetic_wsi_",
            "validation_synthetic_wsi_",
        ),
        encoding="utf-8",
    )
    return dataset_root.parent


def _write_tiny_runtime_pretest_config(
    *,
    tmp_path: Path,
    repo_root: Path,
    data_root: Path,
) -> Path:
    source = repo_root / "configs" / "spec0001" / "non_eq_vae_model_base.json"
    config = _load_json(
        repo_root / "configs" / "spec0001" / "non_eq_vae_kaggle_runtime_benchmark.json",
    )
    config["source_config"] = str(source)
    data = cast("dict[str, object]", config["data"])
    data["data_root"] = str(data_root)
    data["image_size"] = _TINY_IMAGE_SIZE
    data["channels"] = _TINY_CHANNELS
    data["real_train_patch_count"] = _TINY_TRAIN_PATCHES
    data["real_validation_patch_count"] = _TINY_VALIDATION_PATCHES
    data["benchmark_cap"] = {
        "enabled": True,
        "train_patch_count": _TINY_TRAIN_PATCHES,
        "validation_patch_count": _TINY_VALIDATION_PATCHES,
        "window_policy": "fixed_hashed_spread_windows",
        "train_windows": [
            {"name": "train_head", "start_row": 0, "patch_count": 8},
            {"name": "train_tail", "start_row": 8, "patch_count": 8},
        ],
        "validation_windows": [
            {"name": "validation_head", "start_row": 0, "patch_count": 8},
            {"name": "validation_tail", "start_row": 8, "patch_count": 6},
        ],
        "full_epoch_allowed": False,
        "purpose": "tiny_local_real_data_proof_test",
    }
    config_path = tmp_path / "tiny_real_data_runtime_pretest.json"
    config_path.write_text(
        f"{json.dumps(config, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    return config_path


def test_accelerator_observation_stamps_torch_and_cuda_version() -> None:
    """The pretest accelerator record stamps the torch build and CUDA version."""
    observation = pretest._accelerator_observation()  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]

    assert observation["torch_version"] == str(torch.__version__)
    assert observation["cuda_version"] == torch.version.cuda


def test_numerical_delta_fails_and_records_an_amp_skipped_proof_step() -> None:
    """A skipped candidate update cannot masquerade as successful numerical proof.

    The two steps are otherwise identical, so the skip bit is the only reason the
    delta fails. Hardcoding the emitted field or omitting it from ``passed`` breaks both
    assertions.
    """
    losses: JsonObject = {
        "loss": 1.0,
        "recon_loss": 1.0,
        "l1_loss": 1.0,
        "ssim_loss": 1.0,
        "kl_loss": 1.0,
    }
    reference: JsonObject = {
        "losses": losses,
        "grad_norm": 1.0,
        "param_update_norm": 1.0,
        "x_hat_min": -0.5,
        "x_hat_max": 0.5,
        "mu_mean": 0.0,
        "mu_std": 1.0,
        "logvar_mean": 0.0,
        "logvar_std": 1.0,
        "nonfinite_count": 0,
        "logvar_clamp_count": 0,
        "amp_step_skipped": False,
    }
    candidate = dict(reference)
    candidate["amp_step_skipped"] = True

    delta = pretest._numerical_delta_payload(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        reference=reference,
        candidate=candidate,
    )

    assert delta["passed"] is False
    assert delta["amp_step_skipped"] is True
