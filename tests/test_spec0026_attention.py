# Copyright 2026 HiperMaximus
# ruff: noqa: COM812, FBT001, PLR0913, PLR0914, PLR2004, SLF001
"""Attention-algebra tests for the Spec 0026 local-global MIL model."""

from __future__ import annotations

import copy
import itertools
import math
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable

import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional

from eqvae.data.supervised_latents import LogicalPointer, WSIInstance
from eqvae.models import local_global_mil
from eqvae.models.local_global_mil import (
    ATTENTION_HEADS,
    GLOBAL_TOKENS,
    HEAD_WIDTH,
    TOKEN_WIDTH,
    CLSOnlyGlobalBlock,
    GlobalSummaryBlock,
    LocalAttentionGraph,
    PackedLinear,
    SigmoidPatchSummaryAttention,
    SparseLocalSoftmaxAttention,
    SwiGLU,
    build_local_attention_graph,
)


def _small_graph() -> LocalAttentionGraph:
    instances = tuple(
        WSIInstance(
            instance_row=index,
            atlas_row_index=index,
            wsi_id=1,
            x=grid_x * 256,
            y=0,
            diagnosis_label="HGSC",
            diagnosis_index=2,
            split="train",
            pointer=LogicalPointer(part=1, file_index=index),
        )
        for index, grid_x in enumerate((0, 1, 3, 4))
    )
    return build_local_attention_graph(
        instances,
        expected_instance_count=len(instances),
    )


def _grid_graph() -> LocalAttentionGraph:
    instances = tuple(
        WSIInstance(
            instance_row=index,
            atlas_row_index=index,
            wsi_id=2,
            x=grid_x * 256,
            y=grid_y * 256,
            diagnosis_label="HGSC",
            diagnosis_index=2,
            split="train",
            pointer=LogicalPointer(part=1, file_index=index),
        )
        for index, (grid_y, grid_x) in enumerate(
            (grid_y, grid_x) for grid_y in range(5) for grid_x in range(5)
        )
    )
    return build_local_attention_graph(
        instances,
        expected_instance_count=len(instances),
    )


def _manual_local_attention(
    module: SparseLocalSoftmaxAttention,
    tokens: Tensor,
    graph: LocalAttentionGraph,
) -> Tensor:
    query, key, value = (
        part.reshape(-1, ATTENTION_HEADS, HEAD_WIDTH)
        for part in module.qkv.split(cast("Tensor", module.qkv(tokens)))
    )
    rows: list[Tensor] = []
    for row in range(tokens.shape[0]):
        valid = graph.neighbor_valid[row]
        indices = graph.neighbor_index[row, valid]
        codes = graph.radial_code[row, valid].long()
        patch_scores = torch.einsum("hd,khd->hk", query[row], key[indices]) / math.sqrt(
            HEAD_WIDTH
        )
        patch_scores += module.relative_bias[:, codes]
        null_score = (
            torch.einsum("hd,hd->h", query[row], module.null_key)
            / math.sqrt(HEAD_WIDTH)
            + module.null_bias
        )
        weights = torch.softmax(
            torch.cat((patch_scores, null_score[:, None]), dim=1), dim=1
        )
        rows.append(torch.einsum("hk,khd->hd", weights[:, :-1], value[indices]))
    context = torch.stack(rows).reshape(-1, TOKEN_WIDTH)
    return cast("Tensor", module.output(context))


def test_sparse_local_softmax_matches_independent_null_oracle() -> None:
    """Invariant: sparse padding gets no mass, so only the valid null can abstain."""
    torch.manual_seed(11)  # pyright: ignore[reportUnknownMemberType]
    base = SparseLocalSoftmaxAttention()
    with torch.no_grad():
        base.relative_bias.normal_(mean=0.0, std=0.1)
        base.null_key.normal_(mean=0.0, std=0.1)
        base.null_bias.normal_(mean=0.0, std=0.1)
    for graph in (_small_graph(), _grid_graph()):
        optimized = copy.deepcopy(base)
        reference = copy.deepcopy(base)
        optimized_tokens = torch.randn(
            graph.node_count,
            TOKEN_WIDTH,
            requires_grad=True,
        )
        reference_tokens = optimized_tokens.detach().clone().requires_grad_()

        actual = cast("Tensor", optimized(optimized_tokens, graph))
        expected = _manual_local_attention(reference, reference_tokens, graph)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

        actual.square().sum().backward()  # pyright: ignore[reportUnknownMemberType]
        expected.square().sum().backward()  # pyright: ignore[reportUnknownMemberType]
        torch.testing.assert_close(
            optimized_tokens.grad,
            reference_tokens.grad,
            rtol=1e-5,
            atol=1e-6,
        )
        reference_parameters = dict(reference.named_parameters())
        for name, parameter in optimized.named_parameters():
            assert parameter.grad is not None, name
            expected_parameter = reference_parameters[name]
            assert expected_parameter.grad is not None, name
            torch.testing.assert_close(
                parameter.grad,
                expected_parameter.grad,
                rtol=1e-5,
                atol=2e-6,
            )

    grid = _grid_graph()
    codes = grid.radial_code[grid.neighbor_valid]
    code_values = cast(
        "list[int]",
        codes.tolist(),  # pyright: ignore[reportUnknownMemberType]
    )
    assert set(code_values) == set(range(6))


def test_sigmoid_summary_zero_logits_use_negative_log_count_calibration() -> None:
    """Invariant: initial global gate mass stays order one as complete bags grow."""
    module = SigmoidPatchSummaryAttention()
    with torch.no_grad():
        module.query.weight.zero_()
        key_weight, value_weight = module.kv.logical_weight_slices()
        key_weight.zero_()
        value_weight.copy_(torch.eye(TOKEN_WIDTH))
        module.output.weight.copy_(torch.eye(TOKEN_WIDTH))
        module.output.bias.zero_()
        module.head_offset.zero_()
    queries = torch.randn(GLOBAL_TOKENS, TOKEN_WIDTH)
    patches = torch.randn(7, TOKEN_WIDTH)

    actual = cast("Tensor", module(queries, patches))
    expected_row = patches.sum(dim=0) / 8.0

    torch.testing.assert_close(
        actual,
        expected_row.expand(GLOBAL_TOKENS, -1),
        rtol=1e-5,
        atol=1e-6,
    )


def test_sigmoid_summary_keeps_only_patch_count_dynamic_under_compile() -> None:
    """Invariant: cardinality calibration cannot specialize one graph per WSI."""
    module = SigmoidPatchSummaryAttention().eval()
    compile_fn = cast("Callable[..., Callable[..., Tensor]]", torch.compile)
    compiled = compile_fn(module, backend="eager", fullgraph=True)
    mark_dynamic = cast(
        "Callable[..., None]",
        torch._dynamo.mark_dynamic,  # pyright: ignore[reportAny, reportPrivateUsage]
    )
    queries = torch.randn(GLOBAL_TOKENS, TOKEN_WIDTH)

    for patch_count in (7, 11):
        patches = torch.randn(patch_count, TOKEN_WIDTH)
        mark_dynamic(patches, 0, min=1, max=32)
        with torch.no_grad():
            actual = compiled(queries, patches)
            expected = cast("Tensor", module(queries, patches))
        torch.testing.assert_close(actual, expected)


def test_sigmoid_summary_preserves_large_finite_context_in_fp32() -> None:
    """Invariant: unnormalized global evidence may exceed the FP16 finite range."""
    module = SigmoidPatchSummaryAttention().eval()
    with torch.no_grad():
        module.query.weight.zero_()
        key_weight, value_weight = module.kv.logical_weight_slices()
        key_weight.zero_()
        value_weight.copy_(torch.eye(TOKEN_WIDTH))
        module.output.weight.copy_(torch.eye(TOKEN_WIDTH))
        module.output.bias.zero_()
        module.head_offset.fill_(100.0)
    queries = torch.zeros(GLOBAL_TOKENS, TOKEN_WIDTH)
    patches = torch.full((128, TOKEN_WIDTH), 1_000.0)

    with torch.autocast(device_type="cpu", dtype=torch.float16):
        actual = cast("Tensor", module(queries, patches))

    assert actual.dtype == torch.float32
    assert torch.isfinite(actual).all()
    assert float(actual.detach().min()) > torch.finfo(torch.float16).max


def _full_global_reference(block: CLSOnlyGlobalBlock, tokens: Tensor) -> Tensor:
    normalized = cast("Tensor", block.attention_norm(tokens))
    query = cast("Tensor", block.query(normalized)).reshape(
        GLOBAL_TOKENS, ATTENTION_HEADS, HEAD_WIDTH
    )
    key, value = (
        part.reshape(GLOBAL_TOKENS, ATTENTION_HEADS, HEAD_WIDTH)
        for part in block.kv.split(cast("Tensor", block.kv(normalized)))
    )
    attended = functional.scaled_dot_product_attention(
        query.transpose(0, 1),
        key.transpose(0, 1),
        value.transpose(0, 1),
        dropout_p=0.0,
        is_causal=False,
        scale=1.0 / math.sqrt(HEAD_WIDTH),
    ).transpose(0, 1)
    projected = cast("Tensor", block.output(attended.reshape(GLOBAL_TOKENS, -1)))
    cls = tokens[0] + projected[0]
    ffn_input = cast("Tensor", block.ffn_norm(cls))
    return cls + cast("Tensor", block.ffn(ffn_input))


def test_cls_only_attention_matches_full_cls_output_and_gradients() -> None:
    """Invariant: omitting 16 unused query rows changes neither CLS nor its learning."""
    torch.manual_seed(23)  # pyright: ignore[reportUnknownMemberType]
    optimized = CLSOnlyGlobalBlock()
    reference = copy.deepcopy(optimized)
    optimized_tokens = torch.randn(GLOBAL_TOKENS, TOKEN_WIDTH, requires_grad=True)
    reference_tokens = optimized_tokens.detach().clone().requires_grad_()

    optimized_output = cast("Tensor", optimized(optimized_tokens))
    reference_output = _full_global_reference(reference, reference_tokens)
    torch.testing.assert_close(optimized_output, reference_output, rtol=1e-5, atol=1e-6)

    optimized_output.square().sum().backward()  # pyright: ignore[reportUnknownMemberType]
    reference_output.square().sum().backward()  # pyright: ignore[reportUnknownMemberType]
    torch.testing.assert_close(
        optimized_tokens.grad, reference_tokens.grad, rtol=1e-5, atol=1e-6
    )
    reference_parameters = dict(reference.named_parameters())
    for name, parameter in optimized.named_parameters():
        reference_parameter = reference_parameters[name]
        assert parameter.grad is not None
        assert reference_parameter.grad is not None
        torch.testing.assert_close(
            parameter.grad,
            reference_parameter.grad,
            rtol=1e-5,
            atol=1e-6,
        )


def test_all_attention_projections_follow_bias_contract() -> None:
    """Invariant: only output projections carry bias, keeping score offsets explicit."""
    local = SparseLocalSoftmaxAttention()
    summary = SigmoidPatchSummaryAttention()
    cls = CLSOnlyGlobalBlock()

    assert local.qkv.bias is None
    assert local.qkv.logical_out_features == (192, 192, 192)
    assert summary.query.bias is None
    assert summary.kv.bias is None
    assert summary.kv.logical_out_features == (192, 192)
    assert cls.query.bias is None
    assert cls.kv.bias is None
    assert cls.kv.logical_out_features == (192, 192)
    assert local.output.bias is not None
    assert summary.output.bias is not None
    assert cls.output.bias is not None


def test_attention_precision_boundaries_inside_cpu_autocast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invariant: autocast cannot underflow calibrated gates or sparse reductions."""
    einsum_dtypes: list[tuple[str, tuple[torch.dtype, ...]]] = []
    sdpa_dtypes: list[tuple[torch.dtype, ...]] = []
    sigmoid_dtypes: list[torch.dtype] = []
    original_einsum = torch.einsum
    original_sdpa = functional.scaled_dot_product_attention
    original_sigmoid = torch.sigmoid

    def recording_einsum(equation: str, *operands: Tensor) -> Tensor:
        einsum_dtypes.append((equation, tuple(value.dtype for value in operands)))
        return original_einsum(equation, *operands)

    def recording_sdpa(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        attn_mask: Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> Tensor:
        assert attn_mask is not None
        sdpa_dtypes.append((query.dtype, key.dtype, value.dtype, attn_mask.dtype))
        return original_sdpa(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )

    def recording_sigmoid(input_tensor: Tensor) -> Tensor:
        sigmoid_dtypes.append(input_tensor.dtype)
        return original_sigmoid(input_tensor)

    monkeypatch.setattr(torch, "einsum", recording_einsum)
    monkeypatch.setattr(functional, "scaled_dot_product_attention", recording_sdpa)
    monkeypatch.setattr(torch, "sigmoid", recording_sigmoid)
    local = SparseLocalSoftmaxAttention()
    summary = SigmoidPatchSummaryAttention()
    tokens = torch.randn(4, TOKEN_WIDTH)
    queries = torch.randn(GLOBAL_TOKENS, TOKEN_WIDTH)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        local_output = cast("Tensor", local(tokens, _small_graph()))
        summary_output = cast("Tensor", summary(queries, tokens))

    assert local_output.dtype == torch.bfloat16
    assert summary_output.dtype == torch.float32
    assert sdpa_dtypes == [(torch.bfloat16,) * 4]
    assert sigmoid_dtypes == [torch.float32]
    relevant_equations = {
        "mhd,nhd->mhn",
        "mhn,nhd->mhd",
    }
    recorded = {
        equation: dtypes
        for equation, dtypes in einsum_dtypes
        if equation in relevant_equations
    }
    assert set(recorded) == relevant_equations
    assert all(
        all(dtype == torch.float32 for dtype in dtypes) for dtypes in recorded.values()
    )


def test_complete_global_token_stream_remains_fp32_inside_autocast() -> None:
    """Invariant: tiny global residual, FFN and CLS paths never return to AMP dtype."""
    summary = GlobalSummaryBlock()
    cls = CLSOnlyGlobalBlock()
    global_tokens = torch.randn(GLOBAL_TOKENS, TOKEN_WIDTH)
    patches = torch.randn(7, TOKEN_WIDTH)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        summarized = cast("Tensor", summary(global_tokens, patches))
        consumed_cls = cast("Tensor", cls(summarized))

    assert summarized.dtype == torch.float32
    assert consumed_cls.dtype == torch.float32
    assert torch.isfinite(summarized).all()
    assert torch.isfinite(consumed_cls).all()


def _run_local_chunk(
    module: SparseLocalSoftmaxAttention,
    tokens: Tensor,
    graph: LocalAttentionGraph,
    *,
    pad_query_count: int | None,
) -> Tensor:
    query, key, value = (
        part.reshape(-1, ATTENTION_HEADS, HEAD_WIDTH)
        for part in module.qkv.split(cast("Tensor", module.qkv(tokens)))
    )
    context = module._attention_chunk(  # pyright: ignore[reportPrivateUsage]
        query,
        key,
        value,
        graph.neighbor_index,
        graph.neighbor_valid,
        graph.radial_code,
        pad_query_count=pad_query_count,
    )
    return cast("Tensor", module.output(context.reshape(-1, TOKEN_WIDTH)))


def test_final_query_chunk_padding_is_algebraically_inert() -> None:
    """Invariant: helper-level tail padding cannot affect any real query."""
    torch.manual_seed(29)  # pyright: ignore[reportUnknownMemberType]
    unpadded = SparseLocalSoftmaxAttention()
    padded = copy.deepcopy(unpadded)
    graph = _small_graph()
    unpadded_tokens = torch.randn(4, TOKEN_WIDTH, requires_grad=True)
    padded_tokens = unpadded_tokens.detach().clone().requires_grad_()

    expected = _run_local_chunk(
        unpadded,
        unpadded_tokens,
        graph,
        pad_query_count=None,
    )
    actual = _run_local_chunk(
        padded,
        padded_tokens,
        graph,
        pad_query_count=8,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    actual.square().sum().backward()  # pyright: ignore[reportUnknownMemberType]
    expected.square().sum().backward()  # pyright: ignore[reportUnknownMemberType]
    torch.testing.assert_close(padded_tokens.grad, unpadded_tokens.grad)
    expected_parameters = dict(unpadded.named_parameters())
    for name, parameter in padded.named_parameters():
        expected_parameter = expected_parameters[name]
        assert parameter.grad is not None, name
        assert expected_parameter.grad is not None, name
        torch.testing.assert_close(parameter.grad, expected_parameter.grad)


def test_public_local_attention_loop_visits_every_query_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invariant: the eager outer loop neither drops nor duplicates query rows."""
    module = SparseLocalSoftmaxAttention()
    graph = _grid_graph()
    tokens = torch.randn(graph.node_count, TOKEN_WIDTH)
    chunk_sizes: list[int] = []

    def return_query(
        query: Tensor,
        _key: Tensor,
        _value: Tensor,
        _neighbor_index: Tensor,
        _neighbor_valid: Tensor,
        _radial_code: Tensor,
        *,
        pad_query_count: int | None = None,
    ) -> Tensor:
        assert pad_query_count is None
        chunk_sizes.append(query.shape[0])
        return query

    monkeypatch.setattr(local_global_mil, "LOCAL_CHUNK_SIZE", 4)
    monkeypatch.setattr(module, "_attention_chunk", return_query)
    actual = cast("Tensor", module(tokens, graph))
    query, _key, _value = module.qkv.split(cast("Tensor", module.qkv(tokens)))
    expected = cast("Tensor", module.output(query))

    assert chunk_sizes == [4, 4, 4, 4, 4, 4, 1]
    torch.testing.assert_close(actual, expected)


def test_largest_bag_chunk_plan_has_the_exact_masked_tail() -> None:
    """Invariant: the real largest bag has 15 full chunks plus 1,875 real rows."""
    chunks = list(
        local_global_mil._local_query_chunks(32_595),  # pyright: ignore[reportPrivateUsage]
    )
    assert len(chunks) == 16
    assert chunks[-1] == (30_720, 32_595)
    assert 2048 - (chunks[-1][1] - chunks[-1][0]) == 173
    assert sum(stop - start for start, stop in chunks) == 32_595
    assert all(left[1] == right[0] for left, right in itertools.pairwise(chunks))


@pytest.mark.parametrize(
    ("logical_widths", "bias"),
    [
        ((192, 192, 192), False),
        ((192, 192), False),
        ((256, 256), True),
    ],
)
def test_every_packed_projection_layout_matches_independent_linears(
    logical_widths: tuple[int, ...],
    bias: bool,
) -> None:
    """Invariant: all selected packs preserve initialization and AdamW algebra."""
    seed = 47
    torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]
    packed = PackedLinear(TOKEN_WIDTH, logical_widths, bias=bias)
    torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]
    expected_weights = tuple(
        torch.empty(width, TOKEN_WIDTH) for width in logical_widths
    )
    for expected_weight in expected_weights:
        nn.init.xavier_uniform_(expected_weight, gain=1.0)
    for actual_weight, expected_weight in zip(
        packed.logical_weight_slices(),
        expected_weights,
        strict=True,
    ):
        torch.testing.assert_close(actual_weight, expected_weight)
    if bias:
        assert packed.bias is not None
        assert torch.count_nonzero(packed.bias) == 0

    references = nn.ModuleList(
        nn.Linear(TOKEN_WIDTH, width, bias=bias) for width in logical_widths
    )
    with torch.no_grad():
        start = 0
        for raw_reference, weight in zip(
            references,
            packed.logical_weight_slices(),
            strict=True,
        ):
            reference = cast("nn.Linear", raw_reference)
            reference.weight.copy_(weight)
            if bias:
                assert reference.bias is not None
                assert packed.bias is not None
                reference.bias.copy_(
                    packed.bias[start : start + reference.out_features],
                )
            start += reference.out_features

    packed_input = torch.randn(5, TOKEN_WIDTH, requires_grad=True)
    reference_input = packed_input.detach().clone().requires_grad_()
    packed_outputs = packed.split(cast("Tensor", packed(packed_input)))
    reference_outputs = tuple(
        cast("Tensor", cast("nn.Linear", reference)(reference_input))
        for reference in references
    )
    for actual, expected in zip(packed_outputs, reference_outputs, strict=True):
        torch.testing.assert_close(actual, expected)
    packed_loss = torch.stack(
        tuple(
            (index + 1) * output.square().mean()
            for index, output in enumerate(packed_outputs)
        )
    ).sum()
    reference_loss = torch.stack(
        tuple(
            (index + 1) * output.square().mean()
            for index, output in enumerate(reference_outputs)
        )
    ).sum()
    packed_loss.backward()  # pyright: ignore[reportUnknownMemberType]
    reference_loss.backward()  # pyright: ignore[reportUnknownMemberType]
    torch.testing.assert_close(packed_input.grad, reference_input.grad)
    assert packed.weight.grad is not None
    start = 0
    for raw_reference, width in zip(
        references,
        logical_widths,
        strict=True,
    ):
        reference = cast("nn.Linear", raw_reference)
        actual_grad = packed.weight.grad[start : start + width]
        torch.testing.assert_close(actual_grad, reference.weight.grad)
        start += width

    packed_optimizer = torch.optim.AdamW([packed.weight], lr=1e-3, foreach=False)
    reference_optimizer = torch.optim.AdamW(
        [cast("nn.Linear", reference).weight for reference in references],
        lr=1e-3,
        foreach=False,
    )
    packed_optimizer.step()  # pyright: ignore[reportUnknownMemberType]
    reference_optimizer.step()  # pyright: ignore[reportUnknownMemberType]
    for actual_weight, raw_reference in zip(
        packed.logical_weight_slices(),
        references,
        strict=True,
    ):
        reference = cast("nn.Linear", raw_reference)
        torch.testing.assert_close(actual_weight, reference.weight)


class _UnpackedSwiGLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate = nn.Linear(TOKEN_WIDTH, 256)
        self.value = nn.Linear(TOKEN_WIDTH, 256)
        self.output = nn.Linear(256, TOKEN_WIDTH)

    def forward(self, tokens: Tensor) -> Tensor:
        gate = cast("Tensor", self.gate(tokens))
        value = cast("Tensor", self.value(tokens))
        return cast(
            "Tensor",
            self.output(functional.silu(gate) * value),
        )


def _copy_swiglu_to_unpacked(packed: SwiGLU, unpacked: _UnpackedSwiGLU) -> None:
    gate_weight, value_weight = packed.input.logical_weight_slices()
    assert packed.input.bias is not None
    gate_bias = packed.input.bias[:256]
    value_bias = packed.input.bias[256:]
    with torch.no_grad():
        unpacked.gate.weight.copy_(gate_weight)
        unpacked.gate.bias.copy_(gate_bias)
        unpacked.value.weight.copy_(value_weight)
        unpacked.value.bias.copy_(value_bias)
        unpacked.output.load_state_dict(packed.output.state_dict())


def test_packed_swiglu_matches_unpacked_through_one_optimizer_step() -> None:
    """Invariant: projection packing changes layout, not logical optimization."""
    torch.manual_seed(31)  # pyright: ignore[reportUnknownMemberType]
    packed = SwiGLU()
    unpacked = _UnpackedSwiGLU()
    _copy_swiglu_to_unpacked(packed, unpacked)
    assert isinstance(packed.input, PackedLinear)
    assert packed.input.logical_out_features == (256, 256)
    assert packed.input.bias is not None
    gate_weight, value_weight = packed.input.logical_weight_slices()
    assert not torch.equal(gate_weight, value_weight)

    packed_tokens = torch.randn(7, TOKEN_WIDTH, requires_grad=True)
    unpacked_tokens = packed_tokens.detach().clone().requires_grad_()
    packed_output = cast("Tensor", packed(packed_tokens))
    unpacked_output = cast("Tensor", unpacked(unpacked_tokens))
    torch.testing.assert_close(packed_output, unpacked_output)

    packed_output.square().mean().backward()  # pyright: ignore[reportUnknownMemberType]
    unpacked_output.square().mean().backward()  # pyright: ignore[reportUnknownMemberType]
    torch.testing.assert_close(packed_tokens.grad, unpacked_tokens.grad)
    assert packed.input.weight.grad is not None
    packed_gate_grad = packed.input.weight.grad[:256]
    packed_value_grad = packed.input.weight.grad[256:]
    torch.testing.assert_close(packed_gate_grad, unpacked.gate.weight.grad)
    torch.testing.assert_close(packed_value_grad, unpacked.value.weight.grad)
    assert packed.input.bias.grad is not None
    packed_gate_bias_grad = packed.input.bias.grad[:256]
    packed_value_bias_grad = packed.input.bias.grad[256:]
    torch.testing.assert_close(packed_gate_bias_grad, unpacked.gate.bias.grad)
    torch.testing.assert_close(packed_value_bias_grad, unpacked.value.bias.grad)

    packed_optimizer = torch.optim.AdamW(packed.parameters(), lr=1e-3, foreach=False)
    unpacked_optimizer = torch.optim.AdamW(
        unpacked.parameters(),
        lr=1e-3,
        foreach=False,
    )
    packed_optimizer.step()  # pyright: ignore[reportUnknownMemberType]
    unpacked_optimizer.step()  # pyright: ignore[reportUnknownMemberType]
    gate_weight, value_weight = packed.input.logical_weight_slices()
    assert packed.input.bias is not None
    gate_bias = packed.input.bias[:256]
    value_bias = packed.input.bias[256:]
    torch.testing.assert_close(gate_weight, unpacked.gate.weight)
    torch.testing.assert_close(value_weight, unpacked.value.weight)
    torch.testing.assert_close(gate_bias, unpacked.gate.bias)
    torch.testing.assert_close(value_bias, unpacked.value.bias)
    torch.testing.assert_close(packed.output.weight, unpacked.output.weight)
    torch.testing.assert_close(packed.output.bias, unpacked.output.bias)
