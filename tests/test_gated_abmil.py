"""Direct ABMIL mathematics, non-invasive probes and the actual resume loop."""

import copy
import json
import math
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event, current_thread
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from experiments import spec0054_mil_fold0 as fold
from experiments.spec0054_shard_bags import ShardBags
from eqvae.models.gated_abmil import GatedABMILClassifier, GatedAttention
from eqvae.training.mil_dynamics import EagerLayerProbe, attention_distribution_summary
from eqvae.training.mil_dynamics_checkpoint import load_dynamics_checkpoint
from eqvae.training.mil_t2_lite import run_abmil_t2_lite, flatten_t2_records
from eqvae.training.mil_telemetry_io import load_telemetry_table


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _install_synthetic_coalesced_reads(monkeypatch, bags):
    """Route the production FIFO's binary-read seam to existing test arrays."""
    bags.descriptors["normal_vae", 1] = 12345
    for wsi_id, location in bags.locations.items():
        location.first_record = wsi_id

    def read(descriptor, first_record, indices, gap):
        assert descriptor == 12345
        values = bags.read("normal_vae", first_record, indices).copy()
        return values, len(indices) * fold.RECORD_BYTES, len(indices)

    monkeypatch.setattr(fold, "_read_coalesced", read)
    monkeypatch.setattr(fold.os, "posix_fadvise", lambda *args: None)


def test_attention_matches_formula_gradients_and_bag_symmetries():
    torch.manual_seed(7)
    attention = GatedAttention()
    direct = copy.deepcopy(attention)
    patches = torch.randn(5, 64, requires_grad=True)
    other = patches.detach().clone().requires_grad_(True)
    embedding, terminals = attention(patches)
    gated = torch.tanh(F.linear(other, direct.tanh_projection.weight, direct.tanh_projection.bias)) * torch.sigmoid(F.linear(other, direct.gate_projection.weight, direct.gate_projection.bias))
    probability = F.linear(gated, direct.score.weight, direct.score.bias).squeeze(-1).softmax(0)
    expected = (other * probability[:, None]).sum(0)
    torch.testing.assert_close(embedding, expected)
    embedding.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(patches.grad, other.grad)
    for a, b in zip(attention.parameters(), direct.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad)
    torch.testing.assert_close(terminals[-1].sum(), torch.tensor(1.0))
    torch.testing.assert_close(attention(patches.flip(0))[0], embedding)
    torch.testing.assert_close(attention(patches.repeat(2, 1))[0], embedding)
    torch.testing.assert_close(attention(patches[:1])[0], patches[0])


def test_spatial_projection_and_aem_match_direct_formulas():
    torch.manual_seed(19)
    model = GatedABMILClassifier()
    latent = torch.randn(5, 16, 32, 32)
    projection = model.patch_encoder.projection
    blocks = F.unfold(latent, kernel_size=4, stride=4).transpose(1, 2)
    expected = F.linear(blocks, projection.weight.flatten(1), projection.bias)
    torch.testing.assert_close(projection(latent).flatten(2).transpose(1, 2), expected)
    tokens = model.patch_encoder.tokens(latent)
    attention_block = model.patch_encoder.cls_attention
    q, k, v = attention_block.projections(tokens)
    probability = (q @ k.transpose(-2, -1) / math.sqrt(32)).softmax(-1)
    pooled = (probability @ v).transpose(1, 2).reshape(5, 1, 64)
    torch.testing.assert_close(attention_block(tokens), tokens[:, :1] + attention_block.output(pooled))
    # Make WSI attention nonuniform to check the entropy sign and gradient.
    with torch.no_grad():
        model.attention.score.weight.normal_()
        model.classifier.weight.normal_(std=0.01)
    direct = copy.deepcopy(model)
    target = torch.tensor([2])
    _, _, objective, _ = fold._forward_loss(model, latent, target, torch.tensor(1.7), torch.tensor(0.01))
    logits, representations = direct.forward_with_representations(latent)
    attention = representations[-3]
    entropy = -(attention * attention.log()).sum()
    expected_loss = F.cross_entropy(logits.unsqueeze(0), target) * 1.7 - 0.01 * entropy
    objective.backward()
    expected_loss.backward()
    torch.testing.assert_close(objective, expected_loss)
    for measured, expected_parameter in zip(model.parameters(), direct.parameters(), strict=True):
        torch.testing.assert_close(measured.grad, expected_parameter.grad)


def test_t0_compile_and_t1_preserve_loss_gradients():
    torch.manual_seed(12)
    model = GatedABMILClassifier()
    plain = copy.deepcopy(model)
    latent = torch.randn(3, 16, 32, 32)
    target = torch.tensor([2])
    expected = F.cross_entropy(plain(latent).unsqueeze(0), target) * 1.25
    expected.backward()
    compiled = torch.compile(lambda x: fold._forward_loss(model, x, target, torch.tensor(1.25)), backend="eager", fullgraph=True)
    torch._dynamo.maybe_mark_dynamic(latent, 0)
    logits, loss, weighted, summary = compiled(latent)
    weighted.backward()
    torch.testing.assert_close(weighted, expected)
    assert set(fold._forward_records(summary)) == set(model.capture_names)
    for a, b in zip(model.parameters(), plain.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad)
    second = torch.randn(4, 16, 32, 32)
    torch._dynamo.maybe_mark_dynamic(second, 0)
    torch.testing.assert_close(compiled(second)[0], model(second))
    model.zero_grad(set_to_none=True)
    with EagerLayerProbe(fold._capture_modules(model)) as probe:
        fold._forward_loss(model, latent, target, torch.tensor(1.25))[2].backward()
        records = probe.records()
    assert {"attention.gate", "attention.tanh", "attention.softmax", "attention.pool", "classifier", "patch_encoder"} <= set(records)
    for a, b in zip(model.parameters(), plain.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad)


def test_smoothing_preserves_ordinary_ce_and_matches_soft_target_gradients():
    torch.manual_seed(28)
    model = GatedABMILClassifier().eval()
    with torch.no_grad():
        model.classifier.weight.normal_(std=0.1)
    direct = copy.deepcopy(model)
    latents = torch.randn(4, 16, 32, 32)
    target = torch.tensor([2])
    logits, ordinary, objective, _ = fold._forward_loss(model, latents, target, 1.7, 0.05, 0.1)
    expected_logits, representations = direct.forward_with_representations(latents)
    soft_target = torch.full((5,), 0.02)
    soft_target[2] += 0.9
    entropy = -(representations[-3] * representations[-3].log()).sum()
    expected = -(soft_target * expected_logits.log_softmax(0)).sum() * 1.7 - 0.05 * entropy
    torch.testing.assert_close(ordinary, -logits.log_softmax(0)[2])
    torch.testing.assert_close(objective, expected)
    objective.backward()
    expected.backward()
    for measured, reference in zip(model.parameters(), direct.parameters(), strict=True):
        torch.testing.assert_close(measured.grad, reference.grad)


@pytest.mark.parametrize("gradient_scale", [1.0, 65536.0])
def test_classification_gradient_norm_matches_quadratic_hessian_vector(gradient_scale):
    model = torch.nn.Linear(2, 1, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.4, -0.7]], dtype=torch.float64))
        model.bias.fill_(0.2)
    inputs = torch.tensor([[1.0, -0.3], [0.2, 0.9]], dtype=torch.float64)
    targets = torch.tensor([0.6, -0.5], dtype=torch.float64)
    design = torch.cat((inputs, torch.ones(2, 1, dtype=torch.float64)), dim=1)
    theta = torch.cat((model.weight.detach().reshape(-1), model.bias.detach()))
    loss = 0.5 * (model(inputs).squeeze(-1) - targets).square().sum()
    gradient = design.T @ (design @ theta - targets)
    hessian = design.T @ design
    # Existing accumulated .grad must be unaffected by the inner differentiation.
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 0.125)
    squared = fold._classification_gradient_norm_squared(
        model, loss, gradient_scale=gradient_scale)
    torch.testing.assert_close(squared, gradient.square().sum(), rtol=1e-12, atol=1e-12)
    for parameter in model.parameters():
        torch.testing.assert_close(parameter.grad, torch.full_like(parameter, 0.125))
    squared.backward()
    outer = torch.cat([cast(torch.Tensor, parameter.grad).reshape(-1) for parameter in model.parameters()])
    torch.testing.assert_close(outer, 0.125 + 2 * hessian @ gradient, rtol=1e-12, atol=1e-12)
    # Independent central difference of the closed-form squared gradient norm.
    epsilon = 1e-6
    differences = []
    for axis in torch.eye(3, dtype=torch.float64):
        positive = design.T @ (design @ (theta + epsilon * axis) - targets)
        negative = design.T @ (design @ (theta - epsilon * axis) - targets)
        differences.append((positive.square().sum() - negative.square().sum()) / (2 * epsilon))
    torch.testing.assert_close(outer - 0.125, torch.stack(differences), rtol=1e-8, atol=1e-9)


def test_gradient_penalty_uses_smoothed_ce_without_class_weight_or_aem():
    torch.manual_seed(81)
    model = GatedABMILClassifier(8).eval()
    with torch.no_grad():
        model.classifier.weight.normal_(std=0.07)
        model.attention.score.weight.normal_(std=0.2)
    latents = torch.randn(3, 16, 32, 32)
    target = torch.tensor([3])
    measured = []
    for weight, aem, scale in ((1.0, 0.0, 1.0), (2.7, 0.05, 65536.0)):
        logits, ordinary, base, _, squared = fold._gradient_penalty_forward_loss(
            model, latents, target, weight, aem,
            label_smoothing=0.1, gradient_scale=scale)
        soft_target = torch.full((5,), 0.02)
        soft_target[3] += 0.9
        reference_ce = -(soft_target * logits.log_softmax(0)).sum()
        gradients = torch.autograd.grad(reference_ce, tuple(model.parameters()), retain_graph=True)
        reference_squared = sum(value.square().sum() for value in gradients)
        torch.testing.assert_close(squared, reference_squared, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(ordinary, -logits.log_softmax(0)[3])
        assert all(parameter.grad is None for parameter in model.parameters())
        # Nonzero head weights exercise second derivatives through the encoder/SDPA.
        (base + 0.5e-4 * squared).backward()
        projection_gradient = model.patch_encoder.projection.weight.grad
        assert projection_gradient is not None
        assert projection_gradient.abs().sum() > 0
        for parameter in model.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
        measured.append(squared.detach())
        model.zero_grad(set_to_none=True)
    torch.testing.assert_close(measured[0], measured[1], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("gradient_scale", [1.0, 65536.0])
def test_penalty_calibration_matches_quadratic_base_and_penalty_gradients(gradient_scale):
    model = torch.nn.Linear(2, 1, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.4, -0.7]], dtype=torch.float64))
        model.bias.fill_(0.2)
    inputs = torch.tensor([[1.0, -0.3], [0.2, 0.9]], dtype=torch.float64)
    targets = torch.tensor([0.6, -0.5], dtype=torch.float64)
    design = torch.cat((inputs, torch.ones(2, 1, dtype=torch.float64)), dim=1)
    theta = torch.cat((model.weight.reshape(-1), model.bias))
    reference_theta = theta.detach()
    loss = 0.5 * (model(inputs).squeeze(-1) - targets).square().sum()
    squared = fold._classification_gradient_norm_squared(model, loss, gradient_scale=gradient_scale)
    weight, alpha, coefficient = 1.7, 0.05, 1e-4
    base = weight * loss - alpha / 2 * theta.square().sum()
    penalty = coefficient / 2 * squared
    expected_gradient = design.T @ (design @ reference_theta - targets)
    expected_base = weight * expected_gradient - alpha * reference_theta
    expected_added = coefficient * design.T @ design @ expected_gradient
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 0.125)
    measured = fold._gradient_penalty_calibration(
        model, base, penalty, gradient_scale=gradient_scale)
    assert measured["base_objective_gradient_l2"] == pytest.approx(float(expected_base.norm()), rel=1e-12)
    assert measured["gradient_penalty_gradient_l2"] == pytest.approx(float(expected_added.norm()), rel=1e-12)
    expected_cosine = (expected_base @ expected_added) / (expected_base.norm() * expected_added.norm())
    assert measured["gradient_penalty_base_cosine"] == pytest.approx(float(expected_cosine), rel=1e-12)
    for parameter in model.parameters():
        torch.testing.assert_close(parameter.grad, torch.full_like(parameter, 0.125))
    # Diagnostic autograd must retain the training graph and leave .grad untouched.
    (base + penalty).backward()
    outer = torch.cat([cast(torch.Tensor, parameter.grad).reshape(-1) for parameter in model.parameters()])
    torch.testing.assert_close(outer, 0.125 + expected_base + expected_added, rtol=1e-12, atol=1e-12)


def test_gradient_penalty_is_averaged_per_example_before_accumulation():
    model = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.fill_(0.3)
    regularization = 0.2
    squared_norms = []
    for target in (-1.0, 1.0):
        loss = 0.5 * (model.weight.squeeze() - target).square()
        squared = fold._classification_gradient_norm_squared(model, loss)
        squared_norms.append(squared.detach())
        ((loss + regularization / 2 * squared) / 2).backward()
    # Gradients are 1.3 and -0.7: averaging first would erase their variance.
    torch.testing.assert_close(torch.stack(squared_norms).mean(), torch.tensor(1.09, dtype=torch.float64))
    torch.testing.assert_close(model.weight.grad, torch.tensor([[0.36]], dtype=torch.float64))
    assert torch.stack(squared_norms).mean() > torch.tensor(0.3 ** 2)


def test_gradient_penalty_reuses_dropout_forward_without_extra_rng():
    from torch.nn.attention import SDPBackend, sdpa_kernel

    torch.manual_seed(93)
    model = GatedABMILClassifier(8, cnn_dropout=0.1, mlp_dropout=0.2, bag_dropout=0.2).train()
    with torch.no_grad():
        model.classifier.weight.normal_(std=0.05)
    reference = copy.deepcopy(model)
    latents = torch.randn(2, 16, 32, 32)
    target = torch.tensor([1])
    rng = torch.get_rng_state()
    logits, _, base, _, squared = fold._gradient_penalty_forward_loss(
        model, latents, target, 1.7, 0.05, label_smoothing=0.1, gradient_scale=1.0)
    after = torch.get_rng_state()
    (base + 0.5e-4 * squared).backward()
    assert torch.equal(torch.get_rng_state(), after)
    torch.set_rng_state(rng)
    with sdpa_kernel(SDPBackend.MATH):
        expected = fold._forward_loss(reference, latents, target, 1.7, 0.05, 0.1)
    assert torch.equal(torch.get_rng_state(), after)
    torch.testing.assert_close(logits, expected[0], rtol=0, atol=0)
    torch.testing.assert_close(base, expected[2], rtol=0, atol=0)


def test_c4_is_exact_spatial_permutation_with_paired_independent_schedule():
    array = np.arange(8 * 16 * 32 * 32).reshape(8, 16, 32, 32)
    original = array.copy()
    rotations = np.tile(np.arange(4, dtype=np.uint8), 2)
    fold._apply_c4(array, rotations)
    expected = np.stack([np.rot90(patch, int(k), axes=(-2, -1))
                         for patch, k in zip(original, rotations, strict=True)])
    np.testing.assert_array_equal(array, expected)
    fold._apply_c4(array, (-rotations.astype(np.int16) % 4).astype(np.uint8))
    np.testing.assert_array_equal(array, original)
    indices = fold._sample_indices(100, 1701, 4, 123, 0.6)
    a = fold._sample_c4(len(indices), 1701, 4, 123)
    np.testing.assert_array_equal(a, fold._sample_c4(len(indices), 1701, 4, 123))
    assert not np.array_equal(a, fold._sample_c4(len(indices), 1701, 5, 123))
    np.testing.assert_array_equal(indices, fold._sample_indices(100, 1701, 4, 123, 0.6))


def test_dropout_preserves_initial_weights_eval_and_channel_masks():
    torch.manual_seed(17)
    baseline = GatedABMILClassifier().eval()
    torch.manual_seed(17)
    model = GatedABMILClassifier(cnn_dropout=0.1, mlp_dropout=0.2, bag_dropout=0.2)
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
    latents = torch.randn(6, 16, 32, 32)
    torch.testing.assert_close(model.eval()(latents), baseline(latents), rtol=0, atol=0)
    mask = model.train().patch_encoder.projection_dropout(torch.ones(64, 64, 8, 8))
    torch.testing.assert_close(mask, mask[:, :, :1, :1].expand_as(mask))
    assert (mask == 0).any() and (mask != 0).any()
    assert not torch.equal(model.patch_encoder(latents), model.patch_encoder(latents))
    with torch.no_grad():
        model.classifier.weight.normal_(std=0.1)
    target = torch.tensor([1])
    compiled = torch.compile(lambda x: fold._forward_loss(model, x, target, 1.2, 0.05, 0.1),
                             backend="eager", fullgraph=True)
    reference = copy.deepcopy(model)
    rng = torch.get_rng_state()
    measured = compiled(latents)[2]
    measured.backward()
    torch.set_rng_state(rng)
    expected = fold._forward_loss(reference, latents, target, 1.2, 0.05, 0.1)[2]
    expected.backward()
    torch.testing.assert_close(measured, expected)
    for actual, other in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, other.grad)


def test_t2_uniform_attention_and_nonfinite_observations():
    model = GatedABMILClassifier()
    with torch.no_grad():
        model.attention.score.weight.zero_()
        model.attention.score.bias.zero_()
    state = copy.deepcopy(model.state_dict())
    coordinates = tuple((i * 256, 0) for i in range(6))
    result = run_abmil_t2_lite(model, torch.randn(6, 16, 32, 32), coordinates)
    record = result.records["attention"]
    assert record["entropy_mean"] == pytest.approx(math.log(6))
    assert record["raw_mass_mean"] == pytest.approx(1)
    assert record["effective_instance_count_mean"] == pytest.approx(6)
    assert record["top_5_mass_mean"] == pytest.approx(5 / 6)
    assert result.global_patch_top_lattice_coordinates == tuple((index, 0) for index in result.global_patch_top_indices)
    rows = flatten_t2_records(result, identity={"update": 0}, attention_capture="attention", attention_metric="attention_probability")
    assert any(row["record_kind"] == "attention_top_patch" for row in rows)
    for name, value in model.state_dict().items():
        assert torch.equal(value, state[name])
    invalid = attention_distribution_summary(torch.tensor([float("nan"), 0.5]))
    assert invalid["distribution_defined_fraction"] == 0
    assert math.isnan(invalid["entropy_mean"])


def test_prefetch_overlaps_training_and_keeps_only_two_future_inputs(tmp_path, monkeypatch):
    config = json.loads(Path("docs/data/spec0054_fold0_run.json").read_text())
    config.update(effective_batch=1, t1_train_wsi=0)
    config["training_io"] = {**config["training_io"], "future_wsi": 2}
    bags = SimpleNamespace(files={"normal_vae": {}}, descriptors={}, locations={i: SimpleNamespace(
        fold=1, diagnosis_index=i % 5, count=6,
        coordinates=tuple((j * 256, 0) for j in range(6)), shard=1,
    ) for i in range(10)})
    generator = np.random.default_rng(43)
    arrays = {i: generator.standard_normal((6, 16, 32, 32)).astype(np.float32)
              for i in bags.locations}
    order = fold.paired_epoch_order(row_count=10, epoch=0, seed=config["seed"])
    two_ahead = Event()
    first_pair = Barrier(2)
    background_reads = []
    reader_threads = set()
    main_thread = current_thread()

    def read(branch, wsi_id, indices=None):
        if current_thread() is not main_thread:
            reader_threads.add(current_thread())
            background_reads.append(wsi_id)
            if len(background_reads) == 3:
                two_ahead.set()
            if wsi_id in order[:2]:
                first_pair.wait(timeout=10)
        return arrays[wsi_id] if indices is None else arrays[wsi_id][indices]

    bags.read = read
    _install_synthetic_coalesced_reads(monkeypatch, bags)
    forwards = 0

    def closure(model, *, label_smoothing=0.0):
        def forward(x, target, weight, aem):
            nonlocal forwards
            if forwards == 0:
                # Hold actual compute until the reader has filled its lookahead.
                assert two_ahead.wait(timeout=10)
                assert set(background_reads) == set(order[:3])
            wsi_id = order[forwards]
            indices = fold._sample_indices(6, config["seed"], 0, wsi_id, config["patch_retention"])
            expected = arrays[wsi_id][indices].copy()
            fold._apply_c4(expected, fold._sample_c4(len(indices), config["seed"], 0, wsi_id))
            torch.testing.assert_close(x, torch.from_numpy(expected), rtol=0, atol=0)
            assert x.is_contiguous(memory_format=torch.channels_last)
            forwards += 1
            return fold._forward_loss(model, x, target, weight, aem, label_smoothing)
        return forward

    monkeypatch.setattr(fold, "_compiled_closure", closure)
    monkeypatch.setattr(fold, "should_pause_for_session", lambda **kwargs: False)
    assert fold.run_branch("normal_vae", cast(ShardBags, bags), config, tmp_path,
                           None, {}, time.time(), device=torch.device("cpu"),
                           stop_after_updates=2)
    assert set(background_reads) == set(order[:4])  # Two consumed, two discarded on stop.
    assert len(background_reads) == 4
    assert len(reader_threads) == 2
    assert all(not thread.is_alive() for thread in reader_threads)
    checkpoint = load_dynamics_checkpoint(tmp_path / "normal_vae/latest.pt")
    assert checkpoint["exposure_count"] == checkpoint["within_epoch_cursor"] == 2
    t0 = load_telemetry_table(tmp_path / "normal_vae/t0.npz")
    assert t0["wsi_id"].tolist() == list(order[:2])
    np.testing.assert_allclose(t0["objective_loss"], t0["weighted_smoothed_ce"] - t0["aem_penalty"], atol=1e-7)
    np.testing.assert_allclose(t0["weighted_loss"], t0["unweighted_loss"] * t0["class_weight"], rtol=1e-6)


@pytest.mark.parametrize(("batch", "gradient_penalty_weight"), [(1, 0.0), (4, 0.0), (8, 0.0), (4, 1e-4)])
def test_runner_resume_matches_uninterrupted_after_interrupted_assessment(tmp_path: Path, monkeypatch, batch, gradient_penalty_weight):
    config = json.loads(Path("docs/data/spec0054_fold0_run.json").read_text())
    config.update(epochs=2, warmup_epochs=1, effective_batch=batch,
                  checkpoint_every_exposures=4, t1_train_wsi=0, t2_holdout_wsi=10,
                  gradient_probe_wsi_ids=list(range(10)), gradient_probe_epochs=[0, 1, 2],
                  gradient_penalty_weight=gradient_penalty_weight)
    options = {"calibrate_gradient_penalty": True, "class_weights": [0.8, 1.1, 0.6, 1.4, 1.6],
               "schedule_train_count": 288} if gradient_penalty_weight else {}
    # Ten train bags and five holdouts. A tail batch tests complete epoch coverage.
    bags = SimpleNamespace(files={"normal_vae": {}}, descriptors={}, locations={i: SimpleNamespace(
        fold=1 if i < 10 else 0, diagnosis_index=i % 5, count=5 + i % 2,
        coordinates=tuple((j * 256, 0) for j in range(5 + i % 2)), shard=1,
    ) for i in range(15)})
    generator = np.random.default_rng(32)
    arrays = {i: generator.standard_normal((bag.count, 16, 32, 32)).astype(np.float32) for i, bag in bags.locations.items()}
    bags.read = lambda branch, wsi_id, indices=None: arrays[wsi_id] if indices is None else arrays[wsi_id][indices]
    _install_synthetic_coalesced_reads(monkeypatch, bags)
    original_boundaries = fold._boundaries
    # Unlike the real 288-WSI cohort (half144), synthetic half5 is unreachable
    # by batch4/8. Keep epoch assessments and the two-WSI tail batch here.
    monkeypatch.setattr(fold, "_boundaries", lambda count, recipe:
                        original_boundaries(count, recipe) - ({count // 2} if batch > 1 else set()))
    monkeypatch.setattr(fold, "_compiled_closure", lambda model, *, label_smoothing=0.0: lambda x, y, w, aem=0.0: fold._forward_loss(model, x, y, w, aem, label_smoothing))
    monkeypatch.setattr(fold, "should_pause_for_session", lambda **kwargs: False)
    full = tmp_path / "full"
    class InlineReader(ThreadPoolExecutor):
        def submit(self, fn, /, *args, **kwargs):
            future = Future()
            future.set_result(fn(*args, **kwargs))
            return future
    # Synchronous reference versus asynchronous interruption + continuation.
    with monkeypatch.context() as sync:
        sync.setattr(fold, "ThreadPoolExecutor", InlineReader)
        assert fold.run_branch("normal_vae", cast(ShardBags, bags), config, full, None, {}, time.time(), device=torch.device("cpu"), **options)
    # Graceful session guard during evaluation, after the epoch's training save.
    part = tmp_path / "part"
    saved_exposure = -1
    assessment_reads = 0
    main_thread = current_thread()
    original_save = fold.save_dynamics_checkpoint
    def track_save(path, payload):
        nonlocal saved_exposure
        original_save(path, payload)
        if path.parent.parent == part:
            saved_exposure = payload["exposure_count"]
    monkeypatch.setattr(fold, "save_dynamics_checkpoint", track_save)
    def read(branch, wsi_id, indices=None):
        nonlocal assessment_reads
        if current_thread() is main_thread and saved_exposure == 10:
            assessment_reads += 1
        return arrays[wsi_id] if indices is None else arrays[wsi_id][indices]
    bags.read = read
    def pause(**kwargs):
        return assessment_reads >= 2  # Two full-bag reads into epoch-1 evaluation.
    monkeypatch.setattr(fold, "should_pause_for_session", pause)
    assert not fold.run_branch("normal_vae", cast(ShardBags, bags), config, part, None, {}, time.time(), device=torch.device("cpu"), **options)
    saved = load_dynamics_checkpoint(part / "normal_vae/latest.pt")
    assert saved["exposure_count"] == 10
    assert saved["scheduler"]["last_assessment_exposure"] == (5 if batch == 1 else 0)
    monkeypatch.setattr(fold, "should_pause_for_session", lambda **kwargs: False)
    resumed = tmp_path / "resumed"
    assert fold.run_branch("normal_vae", cast(ShardBags, bags), config, resumed, part, {}, time.time(), device=torch.device("cpu"), **options)
    a = load_dynamics_checkpoint(full / "normal_vae/final.pt")
    b = load_dynamics_checkpoint(resumed / "normal_vae/final.pt")
    def exact(a, b):
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                exact(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b)
            for av, bv in zip(a, b, strict=True):
                exact(av, bv)
        else:
            assert a == b
    exact(a, b)
    t0 = load_telemetry_table(resumed / "normal_vae/t0.npz")
    assert len(t0["wsi_id"]) == 20
    np.testing.assert_allclose(t0["gradient_penalty_weight"], gradient_penalty_weight)
    np.testing.assert_allclose(t0["objective_loss"],
                               t0["weighted_smoothed_ce"] - t0["aem_penalty"] + t0["gradient_penalty"], atol=2e-7)
    if gradient_penalty_weight:
        np.testing.assert_allclose(t0["gradient_penalty"],
                                   0.5 * gradient_penalty_weight * t0["classification_gradient_norm_squared"], rtol=1e-6)
        assert (t0["classification_gradient_norm_squared"] > 0).all()
        assert (t0["base_objective_gradient_l2"] > 0).all()
        assert (t0["gradient_penalty_gradient_l2"] >= 0).all()
        assert np.isfinite(t0["gradient_penalty_base_cosine"]).all()
        expected_weights = np.array([0.8, 1.1, 0.6, 1.4, 1.6])[t0["target"].astype(int)]
        np.testing.assert_array_equal(t0["class_weight"], expected_weights)
        warmup = 288 * config["warmup_epochs"]
        expected_lr = config["peak_lr"] * (0.1 + 0.9 * (t0["exposures"] - 1) / (warmup - 1))
        np.testing.assert_allclose(t0["lr"], expected_lr, rtol=1e-12)
    else:
        assert np.isnan(t0["classification_gradient_norm_squared"]).all()
        np.testing.assert_array_equal(t0["gradient_penalty"], np.zeros(20))
    assert t0["wsi_id"].tolist() == [*fold.paired_epoch_order(row_count=10, epoch=0, seed=config["seed"]), *fold.paired_epoch_order(row_count=10, epoch=1, seed=config["seed"])]
    for table in ("t1", "t2", "predictions"):
        before = load_telemetry_table(full / f"normal_vae/{table}.npz")
        after = load_telemetry_table(resumed / f"normal_vae/{table}.npz")
        assert before.keys() == after.keys()
        for key in before:
            np.testing.assert_array_equal(before[key], after[key])
    best = load_dynamics_checkpoint(resumed / "normal_vae/best.pt")
    exact(load_dynamics_checkpoint(full / "normal_vae/best.pt"), best)
    predictions = load_telemetry_table(resumed / "normal_vae/predictions.npz")
    validation = predictions["split"] == "validation"
    boundaries = np.unique(predictions["exposures"][validation])
    selected = min(boundaries, key=lambda boundary: float(np.mean(
        predictions["unweighted_loss"][validation & (predictions["exposures"] == boundary)])))
    assert best["exposure_count"] == selected


def test_submission_deadline_saves_before_reading_any_training_bag(tmp_path, monkeypatch):
    config = json.loads(Path("docs/data/spec0054_fold0_run.json").read_text())
    config.update(effective_batch=4)
    bags = SimpleNamespace(files={"normal_vae": {}}, descriptors={}, locations={i: SimpleNamespace(
        fold=1, diagnosis_index=i % 5, count=3,
        coordinates=((0, 0), (256, 0), (512, 0)), shard=1,
    ) for i in range(10)})
    monkeypatch.setattr(fold, "_compiled_closure", lambda model, **kwargs: None)
    # The old twelve-hour guard permits work; the submission deadline has expired.
    monkeypatch.setattr(fold, "should_pause_for_session", lambda **kwargs: False)
    assert not fold.run_branch("normal_vae", cast(ShardBags, bags), config, tmp_path,
                               None, {}, time.time(), device=torch.device("cpu"),
                               stop_after_updates=2, session_deadline_unix=time.time() - 1)
    checkpoint = load_dynamics_checkpoint(tmp_path / "normal_vae/latest.pt")
    assert checkpoint["committed_update"] == checkpoint["exposure_count"] == 0
    assert not (tmp_path / "normal_vae/final.pt").exists()


def test_lr_confirmation_resume_replays_pending_boundary_and_stops_at_eight_epochs(tmp_path, monkeypatch):
    config = json.loads(Path("docs/data/spec0054_fold0_run.json").read_text())
    config.update(run_mode="lr_confirmation", effective_batch=4, peak_lr=0.001,
                  t1_train_wsi=0, t2_holdout_wsi=10, gradient_probe_epochs=[])
    bags = SimpleNamespace(files={"normal_vae": {}}, descriptors={}, locations={i: SimpleNamespace(
        fold=1 if i < 10 else 0, diagnosis_index=i % 5, count=3,
        coordinates=((0, 0), (256, 0), (512, 0)), shard=1,
    ) for i in range(15)})
    generator = np.random.default_rng(74)
    # Holdouts have no array: accessing one fails naturally.
    arrays = {i: generator.standard_normal((3, 16, 32, 32)).astype(np.float32) for i in range(10)}
    bags.read = lambda branch, wsi_id, indices=None: arrays[wsi_id] if indices is None else arrays[wsi_id][indices]
    _install_synthetic_coalesced_reads(monkeypatch, bags)
    monkeypatch.setattr(fold, "_compiled_closure", lambda model, *, label_smoothing=0.0: lambda x, y, w, aem: fold._forward_loss(model, x, y, w, aem, label_smoothing))
    monkeypatch.setattr(fold, "should_pause_for_session", lambda **kwargs: False)
    full, part, resumed = (tmp_path / name for name in ("full", "part", "resumed"))
    def run(output, prior=None):
        return fold.run_branch("normal_vae", cast(ShardBags, bags), config, output, prior, {},
                               time.time(), device=torch.device("cpu"), stop_after_updates=24)
    assert run(full)
    paused = False
    original_save = fold.save_dynamics_checkpoint
    def save(path, payload):
        nonlocal paused
        original_save(path, payload)
        if path == part / "normal_vae/latest.pt" and payload["exposure_count"] == 50:
            paused = True
    monkeypatch.setattr(fold, "save_dynamics_checkpoint", save)
    monkeypatch.setattr(fold, "should_pause_for_session", lambda **kwargs: paused)
    assert not run(part)
    monkeypatch.setattr(fold, "should_pause_for_session", lambda **kwargs: False)
    assert run(resumed, part)
    for name in ("t0", "t1", "t2"):
        before = load_telemetry_table(full / f"normal_vae/{name}.npz")
        after = load_telemetry_table(resumed / f"normal_vae/{name}.npz")
        for key in before:
            if key != "step_seconds":
                np.testing.assert_array_equal(before[key], after[key])
    checkpoint = load_dynamics_checkpoint(resumed / "normal_vae/final.pt")
    assert checkpoint["committed_update"] == 24
    assert checkpoint["exposure_count"] == 80
    assert checkpoint["scheduler"]["last_assessment_exposure"] == 80
    t0 = load_telemetry_table(resumed / "normal_vae/t0.npz")
    assert set(t0["lr"][t0["exposures"] > 50]) == {0.001}
    assert not (resumed / "normal_vae/predictions.npz").exists()
    assert run(tmp_path / "completed", resumed)
    completed = load_dynamics_checkpoint(tmp_path / "completed/normal_vae/final.pt")
    for key in checkpoint["model"]:
        assert torch.equal(checkpoint["model"][key], completed["model"][key])


def test_lr_search_tail_batch_resume_and_completed_bound(tmp_path, monkeypatch):
    config = json.loads(Path("docs/data/spec0054_fold0_run.json").read_text())
    config.update(run_mode="lr_search", effective_batch=4, epochs=2,
                  checkpoint_every_exposures=4, t1_train_wsi=0, t2_holdout_wsi=10,
                  lr_search={"preparation_updates": 2, "sweep_updates": 4,
                             "min_lr": 1e-5, "max_lr": 0.1})
    # Ten train bags produce tail batches; five holdouts must never be read.
    bags = SimpleNamespace(files={"normal_vae": {}}, descriptors={}, locations={i: SimpleNamespace(
        fold=1 if i < 10 else 0, diagnosis_index=i % 5, count=3,
        coordinates=((0, 0), (256, 0), (512, 0)), shard=1,
    ) for i in range(15)})
    generator = np.random.default_rng(74)
    arrays = {i: generator.standard_normal((3, 16, 32, 32)).astype(np.float32) for i in range(10)}
    bags.read = lambda branch, wsi_id, indices=None: arrays[wsi_id] if indices is None else arrays[wsi_id][indices]
    _install_synthetic_coalesced_reads(monkeypatch, bags)
    calls = []
    def closure(model, *, label_smoothing=0.0):
        def forward(x, y, w, aem=0.0):
            calls.append(1)
            return fold._forward_loss(model, x, y, w, aem, label_smoothing)
        return forward
    monkeypatch.setattr(fold, "_compiled_closure", closure)
    monkeypatch.setattr(fold, "should_pause_for_session", lambda **kwargs: False)
    full = tmp_path / "full"
    assert fold.run_branch("normal_vae", cast(ShardBags, bags), config, full, None, {},
                           time.time(), device=torch.device("cpu"), stop_after_updates=6)
    part = tmp_path / "part"
    pause_requested = False
    original_save = fold.save_dynamics_checkpoint
    def save(path, payload):
        nonlocal pause_requested
        original_save(path, payload)
        if path == part / "normal_vae/preparation.pt":
            pause_requested = True
    monkeypatch.setattr(fold, "save_dynamics_checkpoint", save)
    monkeypatch.setattr(fold, "should_pause_for_session", lambda **kwargs: pause_requested)
    assert not fold.run_branch("normal_vae", cast(ShardBags, bags), config, part, None, {},
                               time.time(), device=torch.device("cpu"), stop_after_updates=6)
    checkpoint = load_dynamics_checkpoint(part / "normal_vae/latest.pt")
    assert checkpoint["committed_update"] == 2
    assert checkpoint["scheduler"]["last_assessment_exposure"] == 0
    monkeypatch.setattr(fold, "should_pause_for_session", lambda **kwargs: False)
    resumed = tmp_path / "resumed"
    assert fold.run_branch("normal_vae", cast(ShardBags, bags), config, resumed, part, {},
                           time.time(), device=torch.device("cpu"), stop_after_updates=6)
    for name in ("t0", "t1", "t2"):
        before = load_telemetry_table(full / f"normal_vae/{name}.npz")
        after = load_telemetry_table(resumed / f"normal_vae/{name}.npz")
        # Wall time is observational; order, loss, schedule and all probes match.
        for key in before:
            if key != "step_seconds":
                np.testing.assert_array_equal(before[key], after[key])
    t0 = load_telemetry_table(resumed / "normal_vae/t0.npz")
    assert len(t0["update"]) == 20
    lr = [t0["lr"][t0["update"] == update][0] for update in range(1, 7)]
    np.testing.assert_allclose(lr, [1e-5, 1e-5, *np.geomspace(1e-5, 0.1, 4)], rtol=1e-14)
    np.testing.assert_array_equal(t0["aem_coefficient"], np.full(20, config["aem_weight"]))
    assert set(t0["lr_search_phase"]) == {"preparation", "sweep"}
    assert not (resumed / "normal_vae/predictions.npz").exists()
    a = load_dynamics_checkpoint(full / "normal_vae/final.pt")
    b = load_dynamics_checkpoint(resumed / "normal_vae/final.pt")
    for key in a["model"]:
        assert torch.equal(a["model"][key], b["model"][key])
    calls.clear()
    completed = tmp_path / "completed"
    assert fold.run_branch("normal_vae", cast(ShardBags, bags), config, completed, resumed, {},
                           time.time(), device=torch.device("cpu"), stop_after_updates=6)
    assert not calls
    final = load_dynamics_checkpoint(completed / "normal_vae/final.pt")
    assert final["committed_update"] == 6
    assert final["exposure_count"] == 20
