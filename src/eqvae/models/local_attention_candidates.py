# Copyright 2026 HiperMaximus
"""Exact native candidates for the fixed-degree local-attention bakeoff."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor

from eqvae.models.local_global_mil import (
    ATTENTION_HEADS,
    HEAD_WIDTH,
    LOCAL_MAX_DEGREE,
    RADIAL_CODEBOOK,
    TOKEN_WIDTH,
    LocalAttentionGraph,
    SparseLocalSoftmaxAttention,
)

if TYPE_CHECKING:
    from eqvae.models.local_global_mil import (
        LocalGlobalMILClassifier,
        LocalTransformerBlock,
    )

type ProjectedQKV = tuple[Tensor, Tensor, Tensor]
type Fixed25GraphTensors = tuple[Tensor, Tensor, Tensor]
type EdgeGraphTensors = tuple[Tensor, Tensor, Tensor, Tensor]
type LocalAttentionParameters = tuple[Tensor, Tensor, Tensor]


@dataclass(frozen=True)
class EdgeAttentionGraph:
    """CSR-style valid-edge view of one fixed-25 local graph."""

    query_index: Tensor
    key_index: Tensor
    radial_code: Tensor
    row_lengths: Tensor


class WholeBagFixed25Attention(SparseLocalSoftmaxAttention):
    """State-compatible local module using the loop-free native candidate."""

    def forward(self, tokens: Tensor, graph: LocalAttentionGraph) -> Tensor:
        """Project once and compute every query in one dynamic compiled graph.

        Returns:
            One local-attention update per patch token.

        """
        projected = cast("Tensor", self.qkv(tokens))
        query, key, value = (
            part.reshape(-1, ATTENTION_HEADS, HEAD_WIDTH)
            for part in self.qkv.split(projected)
        )
        context = fixed25_inductor_attention(
            (query, key, value),
            (graph.neighbor_index, graph.neighbor_valid, graph.radial_code),
            (self.relative_bias, self.null_key, self.null_bias),
        )
        return cast("Tensor", self.output(context.reshape(-1, TOKEN_WIDTH)))


def use_whole_bag_fixed25_attention(model: LocalGlobalMILClassifier) -> None:
    """Replace both local attention modules without changing learned state."""
    for item in model.local_blocks:
        block = cast("LocalTransformerBlock", item)
        replacement = WholeBagFixed25Attention()
        replacement.load_state_dict(block.attention.state_dict())
        block.attention = replacement


def edge_attention_graph(graph: LocalAttentionGraph) -> EdgeAttentionGraph:
    """Remove fixed-width padding and return query-major valid edge arrays.

    Returns:
        A compact edge view whose rows retain fixed-graph query order.

    """
    query_index, slot_index = torch.nonzero(
        graph.neighbor_valid,
        as_tuple=True,
    )
    return EdgeAttentionGraph(
        query_index=query_index,
        key_index=graph.neighbor_index[query_index, slot_index],
        radial_code=graph.radial_code[query_index, slot_index],
        row_lengths=graph.neighbor_valid.sum(dim=1, dtype=torch.int64),
    )


def fixed25_inductor_attention(  # noqa: PLR0914
    projected: ProjectedQKV,
    graph: Fixed25GraphTensors,
    parameters: LocalAttentionParameters,
) -> Tensor:
    """Compute exact local softmax with native fixed-width tensor reductions.

    Returns:
        Local context with the input value tensor's shape and dtype.

    """
    query, key, value = projected
    neighbor_index, neighbor_valid, radial_code = graph
    relative_bias, null_key, null_bias = parameters
    safe_index = neighbor_index.clamp_min(0)
    gathered_key = key[safe_index]
    gathered_value = value[safe_index]
    score = (query[:, None, :, :].float() * gathered_key.float()).sum(dim=-1) * (
        1.0 / math.sqrt(HEAD_WIDTH)
    )
    safe_code = radial_code.long().clamp_max(len(RADIAL_CODEBOOK) - 1)
    score = score.permute(0, 2, 1) + relative_bias[:, safe_code].permute(1, 0, 2)
    score = score.masked_fill(~neighbor_valid[:, None, :], -torch.inf)
    sink_score = (query.float() * null_key.float()[None, :, :]).sum(dim=-1) * (
        1.0 / math.sqrt(HEAD_WIDTH)
    )
    sink_score += null_bias.float()[None, :]
    probability = torch.softmax(
        torch.cat((score, sink_score[:, :, None]), dim=-1),
        dim=-1,
    )
    weighted_value = (
        probability[:, :, :LOCAL_MAX_DEGREE].permute(0, 2, 1)[:, :, :, None]
        * gathered_value.float()
    )
    return weighted_value.sum(dim=1).to(dtype=value.dtype)


def edge_segment_inductor_attention(  # noqa: PLR0914
    projected: ProjectedQKV,
    graph: EdgeGraphTensors,
    parameters: LocalAttentionParameters,
) -> Tensor:
    """Compute the same attention through edge scores and segment reductions.

    Returns:
        Local context with the input value tensor's shape and dtype.

    """
    query, key, value = projected
    query_index, key_index, radial_code, row_lengths = graph
    relative_bias, null_key, null_bias = parameters
    edge_score = (query[query_index].float() * key[key_index].float()).sum(dim=-1) * (
        1.0 / math.sqrt(HEAD_WIDTH)
    )
    edge_score += relative_bias[:, radial_code.long()].transpose(0, 1)
    sink_score = (query.float() * null_key.float()[None, :, :]).sum(dim=-1) * (
        1.0 / math.sqrt(HEAD_WIDTH)
    )
    sink_score += null_bias.float()[None, :]

    edge_max = torch.segment_reduce(edge_score, "max", lengths=row_lengths)
    row_max = torch.maximum(edge_max, sink_score)
    edge_exponential = torch.exp(edge_score - row_max[query_index])
    denominator = torch.segment_reduce(
        edge_exponential,
        "sum",
        lengths=row_lengths,
    ) + torch.exp(sink_score - row_max)
    edge_probability = edge_exponential / denominator[query_index]
    weighted_value = edge_probability[:, :, None] * value[key_index].float()
    context = torch.segment_reduce(weighted_value, "sum", lengths=row_lengths)
    return context.to(dtype=value.dtype)


__all__ = [
    "EdgeAttentionGraph",
    "EdgeGraphTensors",
    "Fixed25GraphTensors",
    "LocalAttentionParameters",
    "ProjectedQKV",
    "WholeBagFixed25Attention",
    "edge_attention_graph",
    "edge_segment_inductor_attention",
    "fixed25_inductor_attention",
    "use_whole_bag_fixed25_attention",
]
