# Copyright 2026 HiperMaximus
# ruff: noqa: PLR0914, SLF001
"""Tests for Spec 0033 exact native local-attention candidates."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import Tensor

from eqvae.data.supervised_latents import LogicalPointer, WSIInstance
from eqvae.models.local_attention_candidates import (
    Fixed25GraphTensors,
    LocalAttentionParameters,
    ProjectedQKV,
    WholeBagFixed25Attention,
    edge_attention_graph,
    edge_segment_inductor_attention,
    fixed25_inductor_attention,
    use_whole_bag_fixed25_attention,
)
from eqvae.models.local_global_mil import (
    ATTENTION_HEADS,
    HEAD_WIDTH,
    LOCAL_MAX_DEGREE,
    RADIAL_CODEBOOK,
    LocalAttentionGraph,
    LocalGlobalMILClassifier,
    build_local_attention_graph,
)


def _graph(coordinates: tuple[tuple[int, int], ...]) -> LocalAttentionGraph:
    instances = tuple(
        WSIInstance(
            instance_row=index,
            atlas_row_index=index,
            wsi_id=45630,
            x=x * 256,
            y=y * 256,
            diagnosis_label="HGSC",
            diagnosis_index=2,
            split="train",
            pointer=LogicalPointer(part=1, file_index=index),
        )
        for index, (x, y) in enumerate(coordinates)
    )
    return build_local_attention_graph(
        instances,
        expected_instance_count=len(instances),
    )


def _inputs(count: int) -> tuple[Tensor, ...]:
    generator = torch.Generator().manual_seed(3300 + count)
    query = torch.randn(
        count,
        ATTENTION_HEADS,
        HEAD_WIDTH,
        generator=generator,
        requires_grad=True,
    )
    key = torch.randn(
        count,
        ATTENTION_HEADS,
        HEAD_WIDTH,
        generator=generator,
        requires_grad=True,
    )
    value = torch.randn(
        count,
        ATTENTION_HEADS,
        HEAD_WIDTH,
        generator=generator,
        requires_grad=True,
    )
    relative_bias = torch.randn(
        ATTENTION_HEADS,
        len(RADIAL_CODEBOOK),
        generator=generator,
        requires_grad=True,
    )
    null_key = torch.randn(
        ATTENTION_HEADS,
        HEAD_WIDTH,
        generator=generator,
        requires_grad=True,
    )
    null_bias = torch.randn(
        ATTENTION_HEADS,
        generator=generator,
        requires_grad=True,
    )
    return query, key, value, relative_bias, null_key, null_bias


def _oracle(
    projected: ProjectedQKV,
    graph: Fixed25GraphTensors,
    parameters: LocalAttentionParameters,
) -> Tensor:
    query, key, value = projected
    neighbor_index, neighbor_valid, radial_code = graph
    relative_bias, null_key, null_bias = parameters
    rows: list[Tensor] = []
    for query_index in range(query.shape[0]):
        valid = neighbor_valid[query_index]
        indices = neighbor_index[query_index, valid]
        codes = radial_code[query_index, valid].long()
        head_rows: list[Tensor] = []
        for head in range(ATTENTION_HEADS):
            score = (key[indices, head] @ query[query_index, head]) / math.sqrt(
                HEAD_WIDTH,
            )
            score += relative_bias[head, codes]
            sink = (query[query_index, head] @ null_key[head]) / math.sqrt(HEAD_WIDTH)
            sink += null_bias[head]
            probability = torch.softmax(torch.cat((score, sink[None])), dim=0)
            head_rows.append(probability[:-1] @ value[indices, head])
        rows.append(torch.stack(head_rows))
    return torch.stack(rows)


def _gradients(
    output: Tensor,
    tensors: tuple[Tensor, ...],
    upstream: Tensor,
) -> tuple[Tensor, ...]:
    return torch.autograd.grad(
        (output * upstream).sum(),
        tensors,
        retain_graph=False,
    )


def test_native_candidates_match_independent_oracle_and_gradients() -> None:
    """Both tensor layouts reproduce one independent forward/backward oracle."""
    graph = _graph(((0, 0), (1, 0), (4, 0), (1, 1), (3, 2), (4, 2), (8, 8)))
    edge_graph = edge_attention_graph(graph)
    source = _inputs(graph.node_count)
    clones = tuple(tensor.detach().clone().requires_grad_() for tensor in source)
    oracle_clones = tuple(tensor.detach().clone().requires_grad_() for tensor in source)
    upstream = torch.randn_like(source[0])
    fixed_projected: ProjectedQKV = (source[0], source[1], source[2])
    fixed_parameters: LocalAttentionParameters = (source[3], source[4], source[5])
    edge_projected: ProjectedQKV = (clones[0], clones[1], clones[2])
    edge_parameters: LocalAttentionParameters = (clones[3], clones[4], clones[5])
    oracle_projected: ProjectedQKV = (
        oracle_clones[0],
        oracle_clones[1],
        oracle_clones[2],
    )
    oracle_parameters: LocalAttentionParameters = (
        oracle_clones[3],
        oracle_clones[4],
        oracle_clones[5],
    )
    fixed_graph: Fixed25GraphTensors = (
        graph.neighbor_index,
        graph.neighbor_valid,
        graph.radial_code,
    )
    edge_graph_tensors = (
        edge_graph.query_index,
        edge_graph.key_index,
        edge_graph.radial_code,
        edge_graph.row_lengths,
    )

    fixed_output = fixed25_inductor_attention(
        fixed_projected,
        fixed_graph,
        fixed_parameters,
    )
    fixed_gradients = _gradients(fixed_output, source, upstream)
    edge_output = edge_segment_inductor_attention(
        edge_projected,
        edge_graph_tensors,
        edge_parameters,
    )
    edge_gradients = _gradients(edge_output, clones, upstream)
    oracle_output = _oracle(oracle_projected, fixed_graph, oracle_parameters)
    oracle_gradients = _gradients(oracle_output, oracle_clones, upstream)

    torch.testing.assert_close(fixed_output, oracle_output, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(edge_output, oracle_output, rtol=1e-5, atol=1e-6)
    for fixed, edge, oracle in zip(
        fixed_gradients,
        edge_gradients,
        oracle_gradients,
        strict=True,
    ):
        torch.testing.assert_close(fixed, oracle, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(edge, oracle, rtol=2e-5, atol=2e-6)


def _compile_graph(count: int) -> LocalAttentionGraph:
    coordinates = tuple((index % 5, index // 5) for index in range(count))
    return _graph(coordinates)


def test_fixed_candidate_is_one_strict_dynamic_graph() -> None:
    """One strict fixed-width full graph serves two distinct node counts."""
    torch._dynamo.reset()  # pyright: ignore[reportPrivateUsage]
    torch._dynamo.utils.counters.clear()  # pyright: ignore[reportPrivateUsage]
    inputs = _inputs(1)
    params: LocalAttentionParameters = (inputs[3], inputs[4], inputs[5])
    compiled = torch.compile(  # pyright: ignore[reportUnknownMemberType]
        fixed25_inductor_attention,
        backend="inductor",
        fullgraph=True,
        dynamic=None,
    )

    for count in (7, 11):
        graph = _compile_graph(count)
        candidate_inputs = _inputs(count)
        qkv: ProjectedQKV = (
            candidate_inputs[0],
            candidate_inputs[1],
            candidate_inputs[2],
        )
        graph_arguments: Fixed25GraphTensors = (
            graph.neighbor_index,
            graph.neighbor_valid,
            graph.radial_code,
        )
        for tensor in (*qkv, *graph_arguments):
            torch._dynamo.mark_dynamic(  # pyright: ignore[reportPrivateUsage, reportAny]
                tensor,
                0,
                min=1,
                max=32,
            )
        output = compiled(qkv, graph_arguments, params)
        output.square().mean().backward()  # pyright: ignore[reportUnknownMemberType]
        assert output.shape == (count, ATTENTION_HEADS, HEAD_WIDTH)
        assert torch.isfinite(output).all()

    assert (
        torch._dynamo.utils.counters["stats"]["unique_graphs"]  # pyright: ignore[reportPrivateUsage]
        == 1
    )


def test_segment_candidate_is_one_strict_dynamic_graph() -> None:
    """One strict segment full graph serves distinct node and edge counts."""
    torch._dynamo.reset()  # pyright: ignore[reportPrivateUsage]
    torch._dynamo.utils.counters.clear()  # pyright: ignore[reportPrivateUsage]
    inputs = _inputs(1)
    params: LocalAttentionParameters = (inputs[3], inputs[4], inputs[5])
    compiled = torch.compile(  # pyright: ignore[reportUnknownMemberType]
        edge_segment_inductor_attention,
        backend="inductor",
        fullgraph=True,
        dynamic=None,
    )

    for count in (7, 11):
        graph = _compile_graph(count)
        edge_graph = edge_attention_graph(graph)
        candidate_inputs = _inputs(count)
        qkv: ProjectedQKV = (
            candidate_inputs[0],
            candidate_inputs[1],
            candidate_inputs[2],
        )
        graph_arguments = (
            edge_graph.query_index,
            edge_graph.key_index,
            edge_graph.radial_code,
            edge_graph.row_lengths,
        )
        for tensor in graph_arguments[:3]:
            torch._dynamo.mark_dynamic(  # pyright: ignore[reportPrivateUsage, reportAny]
                tensor,
                0,
                min=1,
                max=800,
            )
        for tensor in (*qkv, graph_arguments[3]):
            torch._dynamo.mark_dynamic(  # pyright: ignore[reportPrivateUsage, reportAny]
                tensor,
                0,
                min=1,
                max=32,
            )
        output = compiled(qkv, graph_arguments, params)
        output.square().mean().backward()  # pyright: ignore[reportUnknownMemberType]
        assert output.shape == (count, ATTENTION_HEADS, HEAD_WIDTH)
        assert torch.isfinite(output).all()

    assert (
        torch._dynamo.utils.counters["stats"]["unique_graphs"]  # pyright: ignore[reportPrivateUsage]
        == 1
    )


def test_fixed_candidate_preserves_static_degree_axis() -> None:
    """Only node count varies; fixed graph degree remains exactly 25."""
    graph = _compile_graph(9)
    assert graph.neighbor_index.shape == (9, LOCAL_MAX_DEGREE)
    assert graph.neighbor_valid.shape == (9, LOCAL_MAX_DEGREE)
    assert graph.radial_code.shape == (9, LOCAL_MAX_DEGREE)


def test_whole_bag_module_is_state_compatible_with_locked_model() -> None:
    """Candidate replacement preserves names, values, count and small output."""
    graph = _compile_graph(9)
    model = LocalGlobalMILClassifier()
    reference = LocalGlobalMILClassifier()
    reference.load_state_dict(model.state_dict())
    use_whole_bag_fixed25_attention(model)
    assert all(
        isinstance(block.attention, WholeBagFixed25Attention)
        for block in model.local_blocks
    )
    candidate_state = cast("dict[str, Tensor]", model.state_dict())
    reference_state = cast("dict[str, Tensor]", reference.state_dict())
    assert candidate_state.keys() == reference_state.keys()
    for name, tensor in candidate_state.items():
        torch.testing.assert_close(tensor, reference_state[name])

    latents = torch.randn(graph.node_count, 16, 32, 32)
    candidate = cast("Tensor", model(latents, graph))
    expected = cast("Tensor", reference(latents, graph))
    torch.testing.assert_close(candidate, expected, rtol=2e-5, atol=2e-6)


def test_whole_bag_model_is_one_strict_dynamic_trace() -> None:
    """The candidate removes the last full-model dynamic graph break."""
    torch._dynamo.reset()  # pyright: ignore[reportPrivateUsage]
    torch._dynamo.utils.counters.clear()  # pyright: ignore[reportPrivateUsage]
    model = LocalGlobalMILClassifier().eval()
    use_whole_bag_fixed25_attention(model)
    compiled = torch.compile(  # pyright: ignore[reportUnknownMemberType]
        model,
        backend="eager",
        fullgraph=True,
        dynamic=None,
    )
    for count in (7, 11):
        graph = _compile_graph(count)
        latents = torch.randn(count, 16, 32, 32, requires_grad=True)
        for tensor in (
            latents,
            graph.neighbor_index,
            graph.neighbor_valid,
            graph.radial_code,
        ):
            torch._dynamo.mark_dynamic(  # pyright: ignore[reportPrivateUsage, reportAny]
                tensor,
                0,
                min=1,
                max=32,
            )
        logits = cast("Tensor", compiled(latents, graph))
        logits.square().sum().backward()  # pyright: ignore[reportUnknownMemberType]
        assert logits.shape == (5,)
        assert torch.isfinite(logits).all()
    assert (
        torch._dynamo.utils.counters["stats"]["unique_graphs"]  # pyright: ignore[reportPrivateUsage]
        == 1
    )
