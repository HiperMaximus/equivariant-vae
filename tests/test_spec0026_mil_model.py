# Copyright 2026 HiperMaximus
# ruff: noqa: PLR2004
"""Integrated contract tests for the Spec 0026 local-global MIL model."""

from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
from typing import cast

import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional

from eqvae.data.supervised_latents import LogicalPointer, WSIInstance
from eqvae.models.local_global_mil import (
    CLASS_ORDER,
    EXPECTED_PARAMETER_COUNT,
    LOCAL_GLOBAL_MIL_CONFIG,
    LocalAttentionGraph,
    LocalGlobalMILClassifier,
    LocalGlobalMILConfig,
    LocalTransformerBlock,
    build_local_attention_graph,
    local_global_mil_adamw_parameter_groups,
    make_paired_local_global_mil_models,
)


def _self_graph(node_count: int) -> LocalAttentionGraph:
    instances = tuple(
        WSIInstance(
            instance_row=index,
            atlas_row_index=index,
            wsi_id=1,
            x=index * 3 * 256,
            y=0,
            diagnosis_label="HGSC",
            diagnosis_index=2,
            split="train",
            pointer=LogicalPointer(part=1, file_index=index),
        )
        for index in range(node_count)
    )
    return build_local_attention_graph(
        instances,
        expected_instance_count=node_count,
    )


def test_locked_config_shapes_and_exact_parameter_count() -> None:
    """Invariant: implementation drift cannot silently recreate the larger design."""
    model = LocalGlobalMILClassifier()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    assert LOCAL_GLOBAL_MIL_CONFIG.token_width == 192
    assert LOCAL_GLOBAL_MIL_CONFIG.attention_heads == 6
    assert LOCAL_GLOBAL_MIL_CONFIG.global_registers == 16
    assert LOCAL_GLOBAL_MIL_CONFIG.local_layers == 2
    assert LOCAL_GLOBAL_MIL_CONFIG.local_attention_chunk_size == 2048
    assert LOCAL_GLOBAL_MIL_CONFIG.ffn_inner_width == 256
    assert CLASS_ORDER == ("CC", "EC", "HGSC", "LGSC", "MC")
    assert parameter_count == EXPECTED_PARAMETER_COUNT == 1_513_055

    assert asdict(LOCAL_GLOBAL_MIL_CONFIG) == {
        "latent_shape": (16, 32, 32),
        "patch_size_pixels": 256,
        "patch_encoder_channels": (64, 128, 192),
        "patch_encoder_kernels": (5, 3, 3),
        "patch_encoder_strides": (2, 2, 2),
        "group_norm_groups": 8,
        "token_width": 192,
        "attention_heads": 6,
        "head_width": 32,
        "local_layers": 2,
        "local_chebyshev_radius": 2,
        "local_max_degree": 25,
        "local_relative_bias": "radial_squared_distance",
        "global_registers": 16,
        "global_patch_reads": 1,
        "global_to_patch_feedback": False,
        "local_attention_activation": "softmax_with_static_zero_value_null",
        "local_attention_backend": "fixed26_fp16_efficient_sdpa",
        "local_attention_chunk_size": 2048,
        "global_patch_summary_activation": "sigmoid",
        "global_patch_summary_cardinality_bias": "negative_log_valid_keys",
        "global_patch_summary_precision": "fp32_scores_weights_reduction_output",
        "global_sequence_precision": "float32",
        "final_cls_attention_activation": "softmax",
        "final_cls_attention_mask": "none",
        "local_attention_cardinality_bias": "none",
        "global_patch_summary_head_offset_init": 0.0,
        "qkv_bias": False,
        "attention_output_bias": True,
        "ffn_kind": "swiglu",
        "ffn_inner_width": 256,
        "norm_order": "pre_norm",
        "layer_norm_epsilon": 1e-5,
        "dropout": 0.0,
        "layer_scale": False,
        "classifier_classes": 5,
        "classifier_precision": "float32",
    }
    with pytest.raises(TypeError):
        LocalGlobalMILConfig(token_width=128)  # pyright: ignore[reportCallIssue]


def test_forward_returns_raw_fp32_logits_and_every_parameter_learns() -> None:
    """Invariant: the complete architecture participates in one WSI loss graph."""
    torch.manual_seed(31)  # pyright: ignore[reportUnknownMemberType]
    model = LocalGlobalMILClassifier()
    latents = torch.randn(2, 16, 32, 32)
    logits = cast("Tensor", model(latents, _self_graph(2)))

    assert logits.shape == (5,)
    assert logits.dtype == torch.float32
    logits.square().sum().backward()  # pyright: ignore[reportUnknownMemberType]
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_final_norm_and_classifier_remain_fp32_inside_cpu_autocast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invariant: mixed precision cannot downcast the five-class loss boundary."""
    classifier_dtypes: list[tuple[torch.dtype, torch.dtype, torch.dtype]] = []
    original_linear = functional.linear

    def recording_linear(
        input_tensor: Tensor,
        weight: Tensor,
        bias: Tensor | None = None,
    ) -> Tensor:
        output = original_linear(input_tensor, weight, bias)
        if tuple(weight.shape) == (5, 192):
            classifier_dtypes.append((input_tensor.dtype, weight.dtype, output.dtype))
        return output

    monkeypatch.setattr(functional, "linear", recording_linear)
    model = LocalGlobalMILClassifier()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        logits = cast("Tensor", model(torch.randn(2, 16, 32, 32), _self_graph(2)))

    assert logits.dtype == torch.float32
    assert classifier_dtypes == [(torch.float32, torch.float32, torch.float32)]


def test_global_state_cannot_change_locally_contextualized_patches() -> None:
    """Invariant: global summaries never feed back and leak distant patch context."""
    torch.manual_seed(37)  # pyright: ignore[reportUnknownMemberType]
    baseline = LocalGlobalMILClassifier().eval()
    changed = copy.deepcopy(baseline)
    with torch.no_grad():
        changed.global_tokens.normal_(mean=8.0, std=2.0)
        for parameter in changed.global_summary.parameters():
            parameter.normal_(mean=3.0, std=1.0)
        for parameter in changed.cls_block.parameters():
            parameter.normal_(mean=-3.0, std=1.0)
        for parameter in changed.classifier.parameters():
            parameter.normal_(mean=5.0, std=1.0)
    latents = torch.randn(3, 16, 32, 32)
    graph = _self_graph(3)

    expected = baseline.encode_patches(latents, graph)
    actual = changed.encode_patches(latents, graph)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_initialization_and_optimizer_groups_follow_semantic_roles() -> None:
    """Learned tokens stay unshrunk so decay regularizes maps, not attention slots."""
    model = LocalGlobalMILClassifier()
    named_parameters = dict(model.named_parameters())

    assert model.global_tokens.std().item() > 0.01
    assert model.global_tokens.std().item() < 0.03
    for name, parameter in named_parameters.items():
        if name.endswith(("relative_bias", "null_bias", "head_offset")):
            assert torch.count_nonzero(parameter) == 0, name
        if name.endswith(".bias"):
            assert torch.count_nonzero(parameter) == 0, name

    decay_group, no_decay_group = local_global_mil_adamw_parameter_groups(model)
    decay_ids = {id(parameter) for parameter in decay_group["params"]}
    no_decay_ids = {id(parameter) for parameter in no_decay_group["params"]}
    assert decay_ids.isdisjoint(no_decay_ids)
    assert decay_ids | no_decay_ids == {
        id(parameter) for parameter in model.parameters()
    }
    assert id(model.global_tokens) in no_decay_ids
    assert id(model.classifier.weight) in decay_ids
    for raw_block in model.local_blocks:
        block = cast("LocalTransformerBlock", raw_block)
        assert id(block.attention.null_key) in no_decay_ids
        assert id(block.attention.relative_bias) in no_decay_ids
        assert id(block.attention.null_bias) in no_decay_ids
    assert id(model.global_summary.attention.head_offset) in no_decay_ids
    assert decay_group["weight_decay"] == pytest.approx(5e-3)
    assert no_decay_group["weight_decay"] == pytest.approx(0.0)


def test_paired_models_have_identical_but_independent_initial_states() -> None:
    """Invariant: latent branches begin matched without sharing trainable storage."""
    normal, so2 = make_paired_local_global_mil_models()
    so2_parameters = dict(so2.named_parameters())
    for name, normal_parameter in normal.named_parameters():
        so2_parameter = so2_parameters[name]
        torch.testing.assert_close(normal_parameter, so2_parameter)
        assert normal_parameter.data_ptr() != so2_parameter.data_ptr()


def test_model_source_is_account_and_runtime_neutral() -> None:
    """Invariant: a collaborator clone never inherits an author's remote identity."""
    source_path = (
        Path(__file__).parents[1] / "src" / "eqvae" / "models" / "local_global_mil.py"
    )
    source = source_path.read_text(encoding="utf-8").lower()

    forbidden_literals = ("/kaggle/", "kaggle_source", "dataset_slug", "credential")
    for literal in forbidden_literals:
        assert literal not in source


def test_patch_encoder_uses_the_locked_staged_channel_expansion() -> None:
    """Invariant: the first spatial activation remains width 64 for T4 headroom."""
    model = LocalGlobalMILClassifier()
    convolutions = [
        module
        for module in model.patch_encoder.modules()
        if isinstance(module, nn.Conv2d)
    ]
    assert [(layer.in_channels, layer.out_channels) for layer in convolutions] == [
        (16, 64),
        (64, 128),
        (128, 192),
    ]
    assert [layer.kernel_size for layer in convolutions] == [(5, 5), (3, 3), (3, 3)]
    assert all(layer.bias is None for layer in convolutions)


def test_model_uses_unmodified_pytorch_normalization() -> None:
    """Invariant: AMP, not custom wrappers, owns normalization precision."""
    model = LocalGlobalMILClassifier()
    group_norms = [
        module
        for module in model.patch_encoder.modules()
        if type(module) is nn.GroupNorm
    ]
    layer_norms = [module for module in model.modules() if type(module) is nn.LayerNorm]

    assert len(group_norms) == 3
    assert len(layer_norms) == 10


def test_graph_transfer_verifies_once_and_forward_does_not_rehash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invariant: trusted device graph derivatives remove hot-path host syncs."""
    graph = _self_graph(2)
    original_verify = LocalAttentionGraph.verify_integrity
    verification_count = 0

    def count_verification(candidate: LocalAttentionGraph) -> None:
        nonlocal verification_count
        verification_count += 1
        original_verify(candidate)

    monkeypatch.setattr(LocalAttentionGraph, "verify_integrity", count_verification)
    transferred = graph.to("cpu")
    assert verification_count == 1

    def reject_hot_path(_candidate: LocalAttentionGraph) -> None:
        message = "model forward attempted to rehash its graph"
        raise AssertionError(message)

    monkeypatch.setattr(LocalAttentionGraph, "verify_integrity", reject_hot_path)
    logits = cast(
        "Tensor",
        LocalGlobalMILClassifier()(torch.randn(2, 16, 32, 32), transferred),
    )
    assert logits.shape == (5,)
