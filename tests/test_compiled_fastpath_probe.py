# Copyright 2026 HiperMaximus
"""CPU-only tests for the compiled fast-path probe (fast-path port, step 5b).

The NCCL/DDP/CUDA measurement core (`run_compiled_fastpath_probe`) only runs
under `torchrun` on GPU and is exercised on Kaggle, not here. These tests cover
the import-safe, CPU-runnable public surface: the dynamo-counter helpers, the
negative-control desync guard, the non-promotable payload/artifact builders, the
correctness-invariant verdict, and the eager step configuration the probe uses.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, cast

import pytest
import torch
import torch._dynamo as torch_dynamo  # noqa: PLC2701
from torch._dynamo.utils import counters  # noqa: PLC2701

from eqvae.benchmarking.compiled_fastpath_probe import (
    BLOCKED_CLAIM_KEYS,
    COMPILED_FASTPATH_PROBE_COMPILE_SCOPE,
    COMPILED_FASTPATH_PROBE_STATUS_SCOPE,
    PROBE_STATUS_FAIL,
    PROBE_STATUS_PASS,
    CompiledFastpathProbeEnvironment,
    CompiledFastpathProbeMeasurement,
    CompiledFastpathProbeRequest,
    build_compiled_fastpath_probe_matrix_rows,
    build_compiled_fastpath_probe_proof,
    graph_break_total,
    run_negative_control_desync,
    unique_graph_count,
    write_compiled_fastpath_probe_artifacts,
)
from eqvae.corruption.inline_stain import InlineStainCorruptor
from eqvae.corruption.stain import CONSERVATIVE_DEFAULT_PROFILE, profile_from_name
from eqvae.models.non_equivariant_vae import build_non_equivariant_vae
from eqvae.training import ddp_sync_guard
from eqvae.training.fastpath_step import make_fastpath_step_fn

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

_WORLD_SIZE = 2
_IN_SYNC_FINGERPRINT: tuple[float, float] = (0.0, 0.0)
_STEP_BATCH = 2
_STEP_SIZE = 64
_STEP_SSIM_WEIGHT = 0.1
_DEFAULT_SYNC_CHECKS = 3


def _request(output_dir: Path) -> CompiledFastpathProbeRequest:
    return CompiledFastpathProbeRequest(output_dir=output_dir)


def _environment() -> CompiledFastpathProbeEnvironment:
    return CompiledFastpathProbeEnvironment(
        world_size=_WORLD_SIZE,
        nproc_per_node=_WORLD_SIZE,
        gpu_names=("Tesla T4", "Tesla T4"),
    )


def _measurement(  # noqa: PLR0913
    *,
    graph_break_count: int = 0,
    recompile_count: int = 0,
    positive_sync_in_sync: bool = True,
    negative_control_fired: bool = True,
    nonfinite_loss_count: int = 0,
    sync_check_steps: int = _DEFAULT_SYNC_CHECKS,
) -> CompiledFastpathProbeMeasurement:
    return CompiledFastpathProbeMeasurement(
        graph_break_count=graph_break_count,
        recompile_count=recompile_count,
        positive_sync_in_sync=positive_sync_in_sync,
        negative_control_fired=negative_control_fired,
        sync_check_steps=sync_check_steps,
        compiled_step_ms_p50=10.0,
        eager_step_ms_p50=12.0,
        speedup=1.2,
        nonfinite_loss_count=nonfinite_loss_count,
    )


def _clean_fn(value: torch.Tensor) -> torch.Tensor:
    return value + 1.0


def _data_dependent_fn(value: torch.Tensor) -> torch.Tensor:
    result = value + 1.0
    if bool((result.sum() > 0).item()):
        result *= 2.0
    return result


def _load_json(path: Path) -> dict[str, object]:
    return cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))


def test_counters_report_clean_compile_and_graph_break() -> None:
    """A clean compile adds a graph with zero breaks; a data-dependent one breaks."""
    torch_dynamo.reset()
    counters.clear()
    clean = torch.compile(  # pyright: ignore[reportUnknownMemberType]
        _clean_fn,
        backend="eager",
    )
    clean(torch.zeros(3))
    assert graph_break_total() == 0
    assert unique_graph_count() >= 1

    torch_dynamo.reset()
    counters.clear()
    broken = torch.compile(  # pyright: ignore[reportUnknownMemberType]
        _data_dependent_fn,
        backend="eager",
    )
    broken(torch.ones(3))
    assert graph_break_total() > 0


def test_negative_control_raises_on_hand_desynced_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rank 0's desync makes the guard see divergence versus a pristine rank 1."""

    def gather(gathered: list[object], obj: object) -> None:
        gathered[0] = obj
        gathered[1] = _IN_SYNC_FINGERPRINT

    monkeypatch.setattr(ddp_sync_guard.dist, "all_gather_object", gather)
    with pytest.raises(RuntimeError, match="divergent parameters"):
        run_negative_control_desync(
            rank=0,
            world_size=_WORLD_SIZE,
            device=torch.device("cpu"),
        )


def test_negative_control_stays_silent_when_rank_not_desynced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only rank 0 desyncs, so a non-zero rank leaves the pair in sync (no raise)."""

    def gather(gathered: list[object], obj: object) -> None:
        gathered[0] = obj
        gathered[1] = _IN_SYNC_FINGERPRINT

    monkeypatch.setattr(ddp_sync_guard.dist, "all_gather_object", gather)
    run_negative_control_desync(
        rank=_WORLD_SIZE - 1,
        world_size=_WORLD_SIZE,
        device=torch.device("cpu"),
    )


def test_passed_requires_every_correctness_invariant() -> None:
    """The pass verdict flips when any settle/sync/finiteness invariant breaks."""
    assert _measurement().passed
    assert not _measurement(graph_break_count=1).passed
    assert not _measurement(recompile_count=1).passed
    assert not _measurement(positive_sync_in_sync=False).passed
    assert not _measurement(negative_control_fired=False).passed
    assert not _measurement(nonfinite_loss_count=1).passed
    assert not _measurement(sync_check_steps=0).passed


def test_proof_payload_marks_non_promotable(tmp_path: Path) -> None:
    """The proof blocks promotion: no eligibility, no sources, every claim blocked."""
    proof = build_compiled_fastpath_probe_proof(
        request=_request(tmp_path),
        measurement=_measurement(),
        environment=_environment(),
    )
    assert proof["full_run_eligible"] is False
    assert proof["full_training_launch_ready"] is False
    assert proof["dataset_sources"] == []
    assert proof["competition_sources"] == []
    assert proof["kernel_sources"] == []
    assert proof["model_sources"] == []
    assert proof["status_scope"] == COMPILED_FASTPATH_PROBE_STATUS_SCOPE
    assert proof["status"] == PROBE_STATUS_PASS
    blocked = cast("dict[str, object]", proof["blocked_claims"])
    assert set(blocked) == set(BLOCKED_CLAIM_KEYS)
    assert all(blocked.values())
    compile_block = cast("dict[str, object]", proof["compile"])
    assert compile_block["scope"] == COMPILED_FASTPATH_PROBE_COMPILE_SCOPE
    assert compile_block["dynamic"] is False


def test_proof_status_reflects_failed_measurement(tmp_path: Path) -> None:
    """A broken correctness invariant marks the proof status as failed."""
    proof = build_compiled_fastpath_probe_proof(
        request=_request(tmp_path),
        measurement=_measurement(graph_break_count=1),
        environment=_environment(),
    )
    assert proof["status"] == PROBE_STATUS_FAIL


def test_matrix_row_carries_probe_summary(tmp_path: Path) -> None:
    """The single matrix row echoes the non-promotable scope and probe status."""
    rows = build_compiled_fastpath_probe_matrix_rows(
        request=_request(tmp_path),
        measurement=_measurement(),
        environment=_environment(),
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["status_scope"] == COMPILED_FASTPATH_PROBE_STATUS_SCOPE
    assert row["full_run_eligible"] == "false"
    assert row["compile_scope"] == COMPILED_FASTPATH_PROBE_COMPILE_SCOPE
    assert row["status"] == PROBE_STATUS_PASS
    assert row["negative_control_fired"] == "true"


def test_write_artifacts_emits_three_non_promotable_files(tmp_path: Path) -> None:
    """Writing produces proof/matrix/manifest, and the manifest hashes match."""
    artifacts = write_compiled_fastpath_probe_artifacts(
        request=_request(tmp_path),
        measurement=_measurement(),
        environment=_environment(),
    )
    assert artifacts.proof.exists()
    assert artifacts.matrix.exists()
    assert artifacts.manifest.exists()
    proof = _load_json(artifacts.proof)
    manifest = _load_json(artifacts.manifest)
    assert proof["full_run_eligible"] is False
    assert manifest["full_run_eligible"] is False
    manifest_artifacts = cast("dict[str, object]", manifest["artifacts"])
    assert (
        manifest_artifacts["proof_sha256"]
        == hashlib.sha256(
            artifacts.proof.read_bytes(),
        ).hexdigest()
    )
    assert (
        manifest_artifacts["matrix_sha256"]
        == hashlib.sha256(
            artifacts.matrix.read_bytes(),
        ).hexdigest()
    )
    matrix_text = artifacts.matrix.read_text(encoding="utf-8")
    assert PROBE_STATUS_PASS in matrix_text
    assert "channels_last" in matrix_text


def test_probe_step_configuration_backprops_eagerly() -> None:
    """The probe's default corruptor + ssim weight give a finite backpropable loss."""
    model = build_non_equivariant_vae()
    corruptor = InlineStainCorruptor(profile_from_name(CONSERVATIVE_DEFAULT_PROFILE))
    step_fn = make_fastpath_step_fn(
        model,
        corruptor,
        ssim_weight=_STEP_SSIM_WEIGHT,
        autocast_dtype=torch.bfloat16,
    )
    generator = torch.Generator().manual_seed(0)
    x_clean = (
        torch.rand((_STEP_BATCH, 3, _STEP_SIZE, _STEP_SIZE), generator=generator) * 2.0
    ) - 1.0
    with torch.no_grad():
        mu_shape = model.forward(x_clean).mu.shape
    eps = torch.randn(mu_shape, generator=generator)

    output = step_fn(x_clean, eps, torch.tensor(1.0))

    assert bool(torch.isfinite(output.loss).item())
    assert output.loss.requires_grad
    cast("Callable[[], None]", output.loss.backward)()


def test_probe_step_stays_finite_on_degenerate_batch() -> None:
    """The probe's degenerate linspace input (all images identical) stays finite."""
    model = build_non_equivariant_vae()
    corruptor = InlineStainCorruptor(profile_from_name(CONSERVATIVE_DEFAULT_PROFILE))
    step_fn = make_fastpath_step_fn(
        model,
        corruptor,
        ssim_weight=_STEP_SSIM_WEIGHT,
        autocast_dtype=torch.bfloat16,
    )
    field = torch.linspace(-1.0, 1.0, steps=_STEP_SIZE * _STEP_SIZE).view(
        _STEP_SIZE,
        _STEP_SIZE,
    )
    x_clean = (
        field
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(_STEP_BATCH, 3, _STEP_SIZE, _STEP_SIZE)
        .contiguous()
    )
    with torch.no_grad():
        mu_shape = model.forward(x_clean).mu.shape
    eps = torch.zeros(mu_shape)

    output = step_fn(x_clean, eps, torch.tensor(1.0))

    assert bool(torch.isfinite(output.loss).item())
