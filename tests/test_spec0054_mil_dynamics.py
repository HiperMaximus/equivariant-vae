# Copyright 2026 HiperMaximus
"""Focused mathematical and resume seams for Spec 0054 telemetry."""

from __future__ import annotations

import math
import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional

from eqvae.models.local_global_mil import (
    EXPECTED_PARAMETER_COUNT,
    LocalGlobalMILClassifier,
    build_local_attention_graph,
)
from eqvae.models.local_attention_candidates import (
    WholeBagFixed25Attention,
    use_whole_bag_fixed25_attention,
)
from eqvae.training.mil_dynamics import (
    AdamWTelemetry,
    ExampleDynamicsTracker,
    T0ForwardWrapper,
    attention_distribution_summary,
    compact_tensor_summary,
    parameter_optimizer_records,
    representation_spectrum_summary,
    run_eager_layer_probe,
    t0_forward_records,
)
from eqvae.training.mil_dynamics_checkpoint import (
    build_dynamics_checkpoint,
    load_dynamics_checkpoint,
    restore_dynamics_checkpoint,
    save_dynamics_checkpoint,
    should_pause_for_session,
)
from eqvae.training.mil_dynamics_step import instrumented_adamw_attempt
from eqvae.training.mil_t2_lite import flatten_t2_records, run_t2_lite
from eqvae.training.mil_telemetry_io import (
    load_telemetry_table,
    write_telemetry_table,
)


@dataclass(frozen=True)
class _Instance:
    wsi_id: int
    x: int
    y: int


class _ToyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(3, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(inputs)


def _small_graph(count: int = 3):
    instances = tuple(_Instance(wsi_id=7, x=index * 256, y=0) for index in range(count))
    return build_local_attention_graph(instances, expected_instance_count=count)


def _toy_attempt(
    model: _ToyClassifier,
    optimizer: torch.optim.AdamW,
    telemetry: AdamWTelemetry,
    scaler: torch.amp.GradScaler,
    *,
    update: int,
) -> dict[str, float]:
    inputs = torch.tensor([[0.3, -0.2, 0.7], [-0.1, 0.5, 0.2]])
    targets = torch.tensor([1, 0])

    def closure():
        logits = model(inputs)
        loss = functional.cross_entropy(logits, targets)
        return logits.mean(dim=0), loss, loss

    result = instrumented_adamw_attempt(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        telemetry=telemetry,
        next_committed_update=update,
        forward_loss=closure,
    )
    assert result.committed
    return result.optimizer


def test_compact_tensor_summary_matches_direct_finite_reference() -> None:
    values = torch.tensor([0.0, 1.0, -2.0, float("nan"), float("inf")])
    record = compact_tensor_summary(values, near_zero=0.0).record()
    finite = torch.tensor([0.0, 1.0, -2.0])
    assert record["count"] == 5
    assert record["finite_fraction"] == pytest.approx(3 / 5)
    assert record["mean"] == pytest.approx(float(finite.mean()))
    assert record["mean_abs"] == pytest.approx(float(finite.abs().mean()))
    assert record["rms"] == pytest.approx(float(finite.square().mean().sqrt()))
    assert record["l2"] == pytest.approx(float(finite.norm()))
    assert record["minimum"] == -2
    assert record["maximum"] == 1


def test_t0_wrapper_preserves_logits_and_reduces_six_terminals() -> None:
    torch.manual_seed(11)
    model = LocalGlobalMILClassifier().eval()
    assert (
        sum(parameter.numel() for parameter in model.parameters())
        == EXPECTED_PARAMETER_COUNT
    )
    graph = _small_graph()
    latents = torch.randn(3, 16, 32, 32)
    expected = model(latents, graph)
    observed, summary = T0ForwardWrapper(model)(latents, graph)
    assert torch.equal(expected, observed)
    assert summary.shape == (6, 6)
    records = t0_forward_records(summary)
    assert tuple(records) == T0ForwardWrapper.capture_names
    assert records["patches_x0"]["count"] == 3 * 192
    assert all(record["finite_fraction"] == 1.0 for record in records.values())


def test_t0_wrapper_supports_one_full_compiled_graph() -> None:
    torch.manual_seed(13)
    model = LocalGlobalMILClassifier().eval()
    graph = _small_graph(2)
    latents = torch.randn(2, 16, 32, 32)
    compiled = torch.compile(T0ForwardWrapper(model), backend="eager", fullgraph=True)
    logits, summary = compiled(latents, graph)
    expected = model(latents, graph)
    assert torch.equal(logits, expected)
    assert summary.shape == (6, 6)


def test_t0_full_first_step_matches_uninstrumented_training() -> None:
    torch.manual_seed(14)
    plain = LocalGlobalMILClassifier()
    instrumented = copy.deepcopy(plain)
    use_whole_bag_fixed25_attention(plain)
    use_whole_bag_fixed25_attention(instrumented)
    plain_optimizer = torch.optim.AdamW(plain.parameters(), lr=2e-4, weight_decay=0.01)
    instrumented_optimizer = torch.optim.AdamW(
        instrumented.parameters(), lr=2e-4, weight_decay=0.01
    )
    graph = _small_graph(2)
    latents = torch.randn(2, 16, 32, 32)
    target = torch.tensor([3])

    plain_optimizer.zero_grad(set_to_none=True)
    plain_logits = plain(latents, graph)
    plain_loss = functional.cross_entropy(plain_logits[None], target)
    plain_loss.backward()
    plain_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in plain.named_parameters()
        if parameter.grad is not None
    }
    plain_optimizer.step()

    wrapper = T0ForwardWrapper(instrumented)
    telemetry = AdamWTelemetry()

    def closure():
        logits, _summary = wrapper(latents, graph)
        loss = functional.cross_entropy(logits[None], target)
        return logits, loss, loss

    result = instrumented_adamw_attempt(
        model=instrumented,
        optimizer=instrumented_optimizer,
        scaler=torch.amp.GradScaler("cpu", enabled=False),
        telemetry=telemetry,
        next_committed_update=1,
        forward_loss=closure,
    )
    assert result.committed
    assert torch.equal(torch.tensor(result.logits), plain_logits.detach())
    assert result.unweighted_loss == float(plain_loss.detach())
    assert set(telemetry.previous_gradients) == set(plain_gradients)
    assert all(
        torch.equal(telemetry.previous_gradients[name], value)
        for name, value in plain_gradients.items()
    )
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            plain.state_dict().values(), instrumented.state_dict().values(), strict=True
        )
    )
    plain_state = plain_optimizer.state_dict()
    instrumented_state = instrumented_optimizer.state_dict()
    assert plain_state["param_groups"] == instrumented_state["param_groups"]
    assert all(
        torch.equal(plain_state["state"][index][name], value)
        for index, state in instrumented_state["state"].items()
        for name, value in state.items()
    )


def test_accepted_whole_bag_attention_preserves_state_and_cpu_output() -> None:
    torch.manual_seed(15)
    reference = LocalGlobalMILClassifier().eval()
    candidate = copy.deepcopy(reference)
    state_before = {
        name: value.detach().clone() for name, value in candidate.state_dict().items()
    }
    use_whole_bag_fixed25_attention(candidate)
    assert all(
        isinstance(block.attention, WholeBagFixed25Attention)
        for block in candidate.local_blocks
    )
    assert all(
        torch.equal(value, state_before[name])
        for name, value in candidate.state_dict().items()
    )
    graph = _small_graph(3)
    latents = torch.randn(3, 16, 32, 32)
    assert torch.allclose(
        reference(latents, graph), candidate(latents, graph), atol=1e-6
    )


def test_actual_adamw_telemetry_matches_parameter_difference() -> None:
    torch.manual_seed(17)
    model = _ToyClassifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.07)
    telemetry = AdamWTelemetry()
    inputs = torch.tensor([[0.3, -0.2, 0.7], [-0.1, 0.5, 0.2]])
    targets = torch.tensor([1, 0])
    optimizer.zero_grad(set_to_none=True)
    loss = functional.cross_entropy(model(inputs), targets)
    loss.backward()
    before = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    telemetry.begin_step(model, optimizer)
    optimizer.step()
    record = telemetry.finish_step(model, next_committed_update=1, committed=True)
    direct = math.sqrt(
        sum(
            float((parameter.detach() - before[name]).double().square().sum())
            for name, parameter in model.named_parameters()
        )
    )
    assert record["global.update_l2"] == pytest.approx(direct, rel=1e-7, abs=1e-12)
    assert record["classifier_head.update_energy_fraction"] == pytest.approx(1.0)
    assert record["attempt.committed"] == 1.0


def test_skipped_attempt_does_not_advance_temporal_telemetry() -> None:
    model = _ToyClassifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    telemetry = AdamWTelemetry()
    functional.cross_entropy(model(torch.ones(1, 3)), torch.tensor([1])).backward()
    telemetry.begin_step(model, optimizer)
    record = telemetry.finish_step(model, next_committed_update=1, committed=False)
    assert record["attempt.committed"] == 0.0
    assert record["global.update_l2"] == 0.0
    assert telemetry.last_committed_update == 0
    assert telemetry.previous_gradients == {}


def test_real_grad_scaler_overflow_skips_then_retries_update_one() -> None:
    model = _ToyClassifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cpu", init_scale=65_536.0, enabled=True)
    telemetry = AdamWTelemetry()

    def overflowing_closure():
        logits = model(torch.zeros(1, 3))[0]
        finite_loss_with_overflowing_scaled_gradient = (
            model.classifier.weight[0, 0] * 1e38
        )
        return (
            logits,
            finite_loss_with_overflowing_scaled_gradient,
            finite_loss_with_overflowing_scaled_gradient,
        )

    skipped = instrumented_adamw_attempt(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        telemetry=telemetry,
        next_committed_update=1,
        forward_loss=overflowing_closure,
    )
    assert not skipped.committed
    assert skipped.final_scale < skipped.initial_scale
    assert skipped.optimizer["global.update_l2"] == 0
    assert telemetry.last_committed_update == 0

    def finite_closure():
        logits = model(torch.ones(1, 3))[0]
        loss = functional.cross_entropy(logits[None], torch.tensor([1]))
        return logits, loss, loss

    retried = instrumented_adamw_attempt(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        telemetry=telemetry,
        next_committed_update=1,
        forward_loss=finite_closure,
    )
    assert retried.committed
    assert telemetry.last_committed_update == 1


def test_t1_parameter_optimizer_records_include_adam_and_radial_geometry() -> None:
    model = _ToyClassifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.02)
    telemetry = AdamWTelemetry()
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    _toy_attempt(model, optimizer, telemetry, scaler, update=1)
    inputs = torch.tensor([[0.3, -0.2, 0.7], [-0.1, 0.5, 0.2]])
    functional.cross_entropy(model(inputs), torch.tensor([1, 0])).backward()
    records = parameter_optimizer_records(
        model,
        optimizer,
        last_updates=telemetry.previous_updates,
    )
    weight = records["classifier.weight"]
    assert weight["optimizer.step"] == 1
    assert "adam_effective_step.rms" in weight
    assert "radial_gradient_energy_fraction" in weight
    assert "perpendicular_gradient_ratio" in weight
    assert "last_update_to_weight" in weight
    optimizer.zero_grad(set_to_none=True)


def test_example_dynamics_tracks_learning_forgetting_and_roundtrips() -> None:
    tracker = ExampleDynamicsTracker((11, 22))
    tracker.update(
        wsi_id=11,
        boundary=1,
        correct=False,
        margin=-0.4,
        true_probability=0.2,
    )
    tracker.update(
        wsi_id=11,
        boundary=2,
        correct=True,
        margin=0.3,
        true_probability=0.6,
    )
    tracker.update(
        wsi_id=11,
        boundary=3,
        correct=False,
        margin=-0.1,
        true_probability=0.4,
    )
    record = tracker.record(11)
    assert record["learning_events"] == 1
    assert record["forgetting_events"] == 1
    assert record["first_learned_boundary"] == 2
    assert record["last_forgotten_boundary"] == 3
    restored = ExampleDynamicsTracker((11, 22))
    restored.load_state_dict(tracker.state_dict())
    assert restored.record(11) == record


def test_eager_layer_probe_is_non_intrusive_to_source_model() -> None:
    torch.manual_seed(23)
    model = LocalGlobalMILClassifier()
    graph = _small_graph(2)
    latents = torch.randn(2, 16, 32, 32)
    state_before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }

    def closure(candidate: LocalGlobalMILClassifier) -> torch.Tensor:
        return functional.cross_entropy(
            candidate(latents, graph)[None, :],
            torch.tensor([2]),
        )

    records = run_eager_layer_probe(model, closure)
    assert "patch_encoder.layers.0" in records
    assert "local_blocks.0.attention" in records
    assert "classifier_logits" in records
    assert any(
        key.startswith("backward_output.") for key in records["classifier_logits"]
    )
    assert all(
        torch.equal(value, state_before[name])
        for name, value in model.state_dict().items()
    )
    assert all(parameter.grad is None for parameter in model.parameters())


def test_light_t2_matches_ordinary_logits_and_reports_attention(tmp_path: Path) -> None:
    torch.manual_seed(31)
    model = LocalGlobalMILClassifier().eval()
    use_whole_bag_fixed25_attention(model)
    graph = _small_graph(3)
    latents = torch.randn(3, 16, 32, 32)
    expected = model(latents, graph)
    result = run_t2_lite(model, latents, graph, top_k_patches=2, local_chunk_size=2)
    assert torch.allclose(torch.tensor(result.logits), expected, rtol=1e-5, atol=1e-6)
    assert "local_0.attention" in result.records
    assert "global_summary.attention" in result.records
    assert "cls.attention" in result.records
    assert len(result.global_patch_top_indices) == 2
    assert all(0 <= index < 3 for index in result.global_patch_top_indices)
    assert len(result.global_patch_top_lattice_coordinates) == 2
    rows = flatten_t2_records(result, identity={"update": 7, "wsi_id": 19})
    assert any(
        row["record_kind"] == "summary"
        and row["capture_name"] == "local_0.attention"
        and row["metric_name"] == "entropy_mean"
        for row in rows
    )
    top_rows = [row for row in rows if row["record_kind"] == "global_gate_top_patch"]
    assert [row["patch_index"] for row in top_rows] == list(
        result.global_patch_top_indices
    )
    assert [row["value"] for row in top_rows] == list(result.global_patch_top_scores)
    assert [(row["lattice_x"], row["lattice_y"]) for row in top_rows] == list(
        result.global_patch_top_lattice_coordinates
    )
    path = tmp_path / "t2_00000007_00000007.npz"
    write_telemetry_table(path, rows)
    with np.load(path, allow_pickle=False) as archive:
        assert "local_0.attention" in archive["capture_name"].tolist()
        assert "mean_sigmoid_gate_score" in archive["metric_name"].tolist()


def test_light_t2_math_for_rank_and_attention() -> None:
    matrix = torch.eye(4)
    spectrum = representation_spectrum_summary(matrix)
    assert spectrum["stable_rank"] == pytest.approx(3.0)
    weights = torch.full((2, 4), 0.25)
    attention = attention_distribution_summary(weights)
    assert attention["normalized_entropy_mean"] == pytest.approx(1.0)
    assert attention["effective_instance_count_mean"] == pytest.approx(4.0)
    zero_spectrum = representation_spectrum_summary(torch.zeros(4, 3))
    assert zero_spectrum["spectrum_defined"] == 0
    assert math.isnan(zero_spectrum["effective_rank"])
    zero_attention = attention_distribution_summary(torch.zeros(2, 4))
    assert zero_attention["distribution_defined_fraction"] == 0
    assert math.isnan(zero_attention["entropy_mean"])


def test_cumulative_telemetry_table_is_atomic_and_resume_truncates(
    tmp_path: Path,
) -> None:
    path = tmp_path / "t0.npz"
    rows = (
        {"update": 1, "loss": 1.25, "committed": True},
        {"update": 2, "loss": 0.75, "committed": True},
    )
    write_telemetry_table(path, rows)
    loaded = load_telemetry_table(path)
    assert loaded["update"].tolist() == [1, 2]
    assert loaded["loss"].tolist() == [1.25, 0.75]
    truncated = load_telemetry_table(path, through_update=1)
    assert truncated["update"].tolist() == [1]


def test_atomic_checkpoint_resume_matches_uninterrupted_next_update(
    tmp_path: Path,
) -> None:
    torch.manual_seed(41)
    model = _ToyClassifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.03)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    telemetry = AdamWTelemetry()
    dynamics = ExampleDynamicsTracker((11, 22))
    dynamics.update(
        wsi_id=11,
        boundary=1,
        correct=True,
        margin=0.2,
        true_probability=0.6,
    )
    first = _toy_attempt(model, optimizer, telemetry, scaler, update=1)
    assert first["global.update_l2"] > 0
    scheduler.step()
    payload = build_dynamics_checkpoint(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        scheduler=scheduler,
        telemetry=telemetry,
        dynamics=dynamics,
        committed_update=1,
        exposure_count=1,
        epoch=0,
        within_epoch_cursor=1,
        current_order=(1, 0),
        effective_batch_size=1,
    )
    checkpoint_path = tmp_path / "latest.pt"
    save_dynamics_checkpoint(checkpoint_path, payload)

    resumed_model = _ToyClassifier()
    resumed_optimizer = torch.optim.AdamW(
        resumed_model.parameters(), lr=2e-3, weight_decay=0.03
    )
    resumed_scaler = torch.amp.GradScaler("cpu", enabled=False)
    resumed_scheduler = torch.optim.lr_scheduler.StepLR(
        resumed_optimizer, step_size=3, gamma=0.5
    )
    resumed_telemetry = AdamWTelemetry()
    resumed_dynamics = ExampleDynamicsTracker((11, 22))
    loaded = load_dynamics_checkpoint(checkpoint_path)
    progress = restore_dynamics_checkpoint(
        loaded,
        model=resumed_model,
        optimizer=resumed_optimizer,
        scaler=resumed_scaler,
        scheduler=resumed_scheduler,
        telemetry=resumed_telemetry,
        dynamics=resumed_dynamics,
    )
    assert progress.committed_update == 1
    assert progress.exposure_count == 1
    assert progress.epoch == 0
    assert progress.within_epoch_cursor == 1
    assert progress.current_order == (1, 0)
    assert progress.effective_batch_size == 1
    assert resumed_dynamics.record(11) == dynamics.record(11)
    uninterrupted_record = _toy_attempt(model, optimizer, telemetry, scaler, update=2)
    resumed_record = _toy_attempt(
        resumed_model,
        resumed_optimizer,
        resumed_telemetry,
        resumed_scaler,
        update=2,
    )
    assert uninterrupted_record == pytest.approx(resumed_record, rel=0, abs=0)
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            model.state_dict().values(),
            resumed_model.state_dict().values(),
            strict=True,
        )
    )


def test_session_guard_reserves_boundary_and_checkpoint_time() -> None:
    assert not should_pause_for_session(
        session_started_unix=0,
        now_unix=10_000,
    )
    assert should_pause_for_session(
        session_started_unix=0,
        now_unix=42_300,
    )
