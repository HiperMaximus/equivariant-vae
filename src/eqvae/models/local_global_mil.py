# Copyright 2026 HiperMaximus
# ruff: noqa: C901, COM812, DOC201, DOC501, EM101, PLC2801, PLR0912, PLR0913, PLR0914, PLR0915, PLR0917, PLR2004, SLF001, TRY003
"""The AMP-first fixed width-192 local-global MIL architecture."""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Self, TypedDict, cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.nn.attention import SDPBackend, sdpa_kernel

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from eqvae.data.supervised_latents import WSIInstance

LATENT_SHAPE: Final = (16, 32, 32)
PATCH_SIZE_PIXELS: Final = 256
TOKEN_WIDTH: Final = 192
ATTENTION_HEADS: Final = 6
HEAD_WIDTH: Final = 32
LOCAL_RADIUS: Final = 2
LOCAL_MAX_DEGREE: Final = 25
LOCAL_CHUNK_SIZE: Final = 2048
GLOBAL_REGISTERS: Final = 16
GLOBAL_TOKENS: Final = GLOBAL_REGISTERS + 1
FFN_WIDTH: Final = 256
RADIAL_CODEBOOK: Final = (0, 1, 2, 4, 5, 8)
RADIAL_PADDING_CODE: Final = 255
NEIGHBOR_PADDING_INDEX: Final = -1
CLASS_ORDER: Final = ("CC", "EC", "HGSC", "LGSC", "MC")
EXPECTED_PARAMETER_COUNT: Final = 1_513_055
_GRAPH_SCHEMA: Final = b"eqvae_spec0026_graph_v1"
_GRAPH_DTYPES: Final = (
    b"coordinates:<i8;neighbor_index:<i8;neighbor_valid:u1;radial_code:u1"
)


@dataclass(frozen=True, init=False)
class LocalGlobalMILConfig:
    """Expose the single locked architecture without alternative model knobs."""

    latent_shape: tuple[int, int, int] = LATENT_SHAPE
    patch_size_pixels: int = PATCH_SIZE_PIXELS
    patch_encoder_channels: tuple[int, int, int] = (64, 128, TOKEN_WIDTH)
    patch_encoder_kernels: tuple[int, int, int] = (5, 3, 3)
    patch_encoder_strides: tuple[int, int, int] = (2, 2, 2)
    group_norm_groups: int = 8
    token_width: int = TOKEN_WIDTH
    attention_heads: int = ATTENTION_HEADS
    head_width: int = HEAD_WIDTH
    local_layers: int = 2
    local_chebyshev_radius: int = LOCAL_RADIUS
    local_max_degree: int = LOCAL_MAX_DEGREE
    local_relative_bias: str = "radial_squared_distance"
    global_registers: int = GLOBAL_REGISTERS
    global_patch_reads: int = 1
    global_to_patch_feedback: bool = False
    local_attention_activation: str = "softmax_with_static_zero_value_null"
    local_attention_backend: str = "fixed26_fp16_efficient_sdpa"
    local_attention_chunk_size: int = LOCAL_CHUNK_SIZE
    global_patch_summary_activation: str = "sigmoid"
    global_patch_summary_cardinality_bias: str = "negative_log_valid_keys"
    global_patch_summary_precision: str = "fp32_scores_weights_reduction_output"
    global_sequence_precision: str = "float32"
    final_cls_attention_activation: str = "softmax"
    final_cls_attention_mask: str = "none"
    local_attention_cardinality_bias: str = "none"
    global_patch_summary_head_offset_init: float = 0.0
    qkv_bias: bool = False
    attention_output_bias: bool = True
    ffn_kind: str = "swiglu"
    ffn_inner_width: int = FFN_WIDTH
    norm_order: str = "pre_norm"
    layer_norm_epsilon: float = 1e-5
    dropout: float = 0.0
    layer_scale: bool = False
    classifier_classes: int = 5
    classifier_precision: str = "float32"


LOCAL_GLOBAL_MIL_CONFIG: Final = LocalGlobalMILConfig()


@dataclass(frozen=True, init=False)
class LocalAttentionGraph:
    """One immutable sparse graph derived from an ordered WSI instance list."""

    wsi_id: int
    expected_instance_count: int
    lattice_coordinates: Tensor
    neighbor_index: Tensor
    neighbor_valid: Tensor
    radial_code: Tensor
    identity_sha256: str

    @classmethod
    def _create(
        cls,
        *,
        wsi_id: int,
        expected_instance_count: int,
        lattice_coordinates: Tensor,
        neighbor_index: Tensor,
        neighbor_valid: Tensor,
        radial_code: Tensor,
        identity_sha256: str,
        verify: bool = True,
    ) -> Self:
        graph = object.__new__(cls)
        object.__setattr__(graph, "wsi_id", wsi_id)
        object.__setattr__(graph, "expected_instance_count", expected_instance_count)
        object.__setattr__(graph, "lattice_coordinates", lattice_coordinates)
        object.__setattr__(graph, "neighbor_index", neighbor_index)
        object.__setattr__(graph, "neighbor_valid", neighbor_valid)
        object.__setattr__(graph, "radial_code", radial_code)
        object.__setattr__(graph, "identity_sha256", identity_sha256)
        if verify:
            graph.verify_integrity()
        return graph

    @property
    def node_count(self) -> int:
        """Number of ordered graph nodes."""
        return int(self.neighbor_index.shape[0])

    def to(self, device: torch.device | str) -> Self:
        """Verify once, then return a trusted derivative on ``device``."""
        self.verify_integrity()
        return type(self)._create(
            wsi_id=self.wsi_id,
            expected_instance_count=self.expected_instance_count,
            lattice_coordinates=self.lattice_coordinates,
            neighbor_index=self.neighbor_index.to(device=device, copy=True),
            neighbor_valid=self.neighbor_valid.to(device=device, copy=True),
            radial_code=self.radial_code.to(device=device, copy=True),
            identity_sha256=self.identity_sha256,
            verify=False,
        )

    def verify_integrity(self) -> None:
        """Recompute the canonical digest so mutable tensor storage cannot drift."""
        _validate_graph_arrays(self)
        coordinates = tuple(
            tuple(pair)
            for pair in cast(
                "list[list[int]]",
                self.lattice_coordinates.tolist(),  # pyright: ignore[reportUnknownMemberType]
            )
        )
        actual = _graph_identity_sha256(
            wsi_id=self.wsi_id,
            coordinates=cast("tuple[tuple[int, int], ...]", coordinates),
            neighbor_index=self.neighbor_index,
            neighbor_valid=self.neighbor_valid,
            radial_code=self.radial_code,
        )
        if actual != self.identity_sha256:
            raise ValueError("Graph tensors do not match their canonical identity")


class AdamWParameterGroup(TypedDict):
    """One named AdamW group with its explicit semantic decay role."""

    name: str
    params: list[nn.Parameter]
    weight_decay: float


def build_local_attention_graph(
    instances: Sequence[WSIInstance],
    *,
    expected_instance_count: int,
) -> LocalAttentionGraph:
    """Build the locked radius-two lattice graph and its canonical identity."""
    if not instances:
        raise ValueError("A local-attention graph requires at least one instance")
    if expected_instance_count != len(instances):
        raise ValueError("Graph instances must match the complete bag count")
    wsi_id = instances[0].wsi_id
    coordinates: list[tuple[int, int]] = []
    for instance in instances:
        if instance.wsi_id != wsi_id:
            raise ValueError("Every graph instance must have the same WSI identity")
        if instance.x % PATCH_SIZE_PIXELS or instance.y % PATCH_SIZE_PIXELS:
            raise ValueError("WSI coordinates must lie on the 256-pixel lattice")
        coordinates.append((
            instance.x // PATCH_SIZE_PIXELS,
            instance.y // PATCH_SIZE_PIXELS,
        ))
    if len(set(coordinates)) != len(coordinates):
        raise ValueError("WSI lattice coordinates must be unique")

    coordinate_to_index = {
        coordinate: index for index, coordinate in enumerate(coordinates)
    }
    neighbor_index = torch.full(
        (len(instances), LOCAL_MAX_DEGREE),
        NEIGHBOR_PADDING_INDEX,
        dtype=torch.int64,
    )
    neighbor_valid = torch.zeros((len(instances), LOCAL_MAX_DEGREE), dtype=torch.bool)
    radial_code = torch.full(
        (len(instances), LOCAL_MAX_DEGREE),
        RADIAL_PADDING_CODE,
        dtype=torch.uint8,
    )
    radius_to_code = {radius: code for code, radius in enumerate(RADIAL_CODEBOOK)}
    offsets = tuple(
        (delta_x, delta_y)
        for delta_x in range(-LOCAL_RADIUS, LOCAL_RADIUS + 1)
        for delta_y in range(-LOCAL_RADIUS, LOCAL_RADIUS + 1)
    )
    for query_index, (grid_x, grid_y) in enumerate(coordinates):
        valid_slot = 0
        for delta_x, delta_y in offsets:
            key_index = coordinate_to_index.get((grid_x + delta_x, grid_y + delta_y))
            if key_index is None:
                continue
            neighbor_index[query_index, valid_slot] = key_index
            neighbor_valid[query_index, valid_slot] = True
            squared_radius = delta_x * delta_x + delta_y * delta_y
            radial_code[query_index, valid_slot] = radius_to_code[squared_radius]
            valid_slot += 1

    identity = _graph_identity_sha256(
        wsi_id=wsi_id,
        coordinates=coordinates,
        neighbor_index=neighbor_index,
        neighbor_valid=neighbor_valid,
        radial_code=radial_code,
    )
    lattice_coordinates = torch.tensor(coordinates, dtype=torch.int64)
    return LocalAttentionGraph._create(  # pyright: ignore[reportPrivateUsage]
        wsi_id=wsi_id,
        expected_instance_count=expected_instance_count,
        lattice_coordinates=lattice_coordinates,
        neighbor_index=neighbor_index,
        neighbor_valid=neighbor_valid,
        radial_code=radial_code,
        identity_sha256=identity,
    )


class PackedLinear(nn.Linear):
    """One fused projection whose rows preserve independent logical matrices."""

    logical_out_features: tuple[int, ...]

    def __init__(
        self,
        in_features: int,
        logical_out_features: tuple[int, ...],
        *,
        bias: bool,
    ) -> None:
        """Build one linear operator with an explicit logical row partition."""
        if not logical_out_features or any(width < 1 for width in logical_out_features):
            raise ValueError("Packed linear slices must all be nonempty")
        object.__setattr__(self, "logical_out_features", logical_out_features)
        super().__init__(  # pyright: ignore[reportUnknownMemberType]
            in_features,
            sum(logical_out_features),
            bias=bias,
        )

    def split(self, output: Tensor) -> tuple[Tensor, ...]:
        """Split a projected tensor into its canonical logical outputs."""
        outputs: list[Tensor] = []
        start = 0
        for width in self.logical_out_features:
            outputs.append(output[..., start : start + width])
            start += width
        return tuple(outputs)

    def logical_weight_slices(self) -> tuple[Tensor, ...]:
        """Return the row views corresponding to the logical projections."""
        outputs: list[Tensor] = []
        start = 0
        for width in self.logical_out_features:
            outputs.append(self.weight[start : start + width])
            start += width
        return tuple(outputs)

    def reset_parameters(self) -> None:
        """Initialize every logical matrix independently in canonical order."""
        for logical_weight in self.logical_weight_slices():
            nn.init.xavier_uniform_(logical_weight, gain=1.0)
        if self.bias is not None:  # pyright: ignore[reportUnnecessaryComparison]
            nn.init.zeros_(self.bias)


class LocalGlobalPatchEncoder(nn.Module):
    """Compress frozen posterior means into trainable width-192 patch tokens."""

    def __init__(self) -> None:
        """Build the locked staged convolutional encoder."""
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, TOKEN_WIDTH, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, TOKEN_WIDTH),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
        )

    def forward(self, latents: Tensor) -> Tensor:
        """Encode exactly one nonempty `[N,16,32,32]` complete WSI bag."""
        if latents.ndim != 4 or tuple(latents.shape[1:]) != LATENT_SHAPE:
            raise ValueError("Patch encoder requires [N,16,32,32] latents")
        if latents.shape[0] < 1:
            raise ValueError("A complete WSI bag may not be empty")
        return cast("Tensor", self.layers(latents))


class SwiGLU(nn.Module):
    """The fixed token-wise width-256 SwiGLU residual branch."""

    def __init__(self) -> None:
        """Build one packed gate/value input and one output projection."""
        super().__init__()
        self.input = PackedLinear(
            TOKEN_WIDTH,
            (FFN_WIDTH, FFN_WIDTH),
            bias=True,
        )
        self.output = nn.Linear(FFN_WIDTH, TOKEN_WIDTH)

    def forward(self, tokens: Tensor) -> Tensor:
        """Apply SiLU to the gate branch before elementwise modulation."""
        gate, value = self.input.split(cast("Tensor", self.input(tokens)))
        return cast(
            "Tensor",
            self.output(functional.silu(gate) * value),
        )


class SparseLocalSoftmaxAttention(nn.Module):
    """Fixed-26 gathered SDPA with one static zero-value null."""

    def __init__(self) -> None:
        """Build six projection heads and their layer-owned null parameters."""
        super().__init__()
        self.qkv = PackedLinear(
            TOKEN_WIDTH,
            (TOKEN_WIDTH, TOKEN_WIDTH, TOKEN_WIDTH),
            bias=False,
        )
        self.output = nn.Linear(TOKEN_WIDTH, TOKEN_WIDTH)
        self.relative_bias = nn.Parameter(torch.zeros(ATTENTION_HEADS, 6))
        self.null_key = nn.Parameter(torch.zeros(ATTENTION_HEADS, HEAD_WIDTH))
        self.null_bias = nn.Parameter(torch.zeros(ATTENTION_HEADS))

    def forward(self, tokens: Tensor, graph: LocalAttentionGraph) -> Tensor:
        """Attend to at most 25 graph neighbours plus one noncommunicating null."""
        _validate_tokens_and_graph(tokens, graph)
        projected = cast("Tensor", self.qkv(tokens))
        query, key, value = (
            part.reshape(-1, ATTENTION_HEADS, HEAD_WIDTH)
            for part in self.qkv.split(projected)
        )
        query = query.reshape(-1, ATTENTION_HEADS, HEAD_WIDTH)
        chunks: list[Tensor] = []
        for start, stop in _local_query_chunks(tokens.shape[0]):
            chunks.append(
                self._attention_chunk(
                    query[start:stop],
                    key,
                    value,
                    graph.neighbor_index[start:stop],
                    graph.neighbor_valid[start:stop],
                    graph.radial_code[start:stop],
                    pad_query_count=(
                        LOCAL_CHUNK_SIZE
                        if (
                            query.device.type == "cuda"
                            and stop - start < LOCAL_CHUNK_SIZE
                        )
                        else None
                    ),
                ),
            )
        context = torch.cat(chunks, dim=0).reshape(-1, TOKEN_WIDTH)
        return cast("Tensor", self.output(context))

    def _attention_chunk(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        neighbor_index: Tensor,
        neighbor_valid: Tensor,
        radial_code: Tensor,
        *,
        pad_query_count: int | None = None,
    ) -> Tensor:
        real_query_count = query.shape[0]
        if pad_query_count is not None:
            if pad_query_count < real_query_count:
                raise ValueError("Padded query count may not truncate real queries")
            padding = pad_query_count - real_query_count
            if padding:
                query = functional.pad(query, (0, 0, 0, 0, 0, padding))
                neighbor_index = functional.pad(
                    neighbor_index,
                    (0, 0, 0, padding),
                    value=NEIGHBOR_PADDING_INDEX,
                )
                neighbor_valid = functional.pad(
                    neighbor_valid,
                    (0, 0, 0, padding),
                    value=False,
                )
                radial_code = functional.pad(
                    radial_code,
                    (0, 0, 0, padding),
                    value=RADIAL_PADDING_CODE,
                )
        safe_index = neighbor_index.clamp_min(0)
        gathered_key = key[safe_index].permute(0, 2, 1, 3)
        gathered_value = value[safe_index].permute(0, 2, 1, 3)
        chunk_count = query.shape[0]
        null_key = self.null_key.to(dtype=query.dtype)[None, :, None, :].expand(
            chunk_count,
            -1,
            -1,
            -1,
        )
        null_value = torch.zeros_like(null_key)
        gathered_key = torch.cat((gathered_key, null_key), dim=2)
        gathered_value = torch.cat((gathered_value, null_value), dim=2)

        safe_code = radial_code.long().clamp_max(len(RADIAL_CODEBOOK) - 1)
        relative_bias = self.relative_bias[:, safe_code].permute(1, 0, 2)
        relative_bias = relative_bias.masked_fill(
            ~neighbor_valid[:, None, :],
            -torch.inf,
        )
        null_bias = self.null_bias[None, :, None].expand(chunk_count, -1, -1)
        attention_bias = torch.cat((relative_bias, null_bias), dim=2).to(
            dtype=query.dtype,
        )
        query = query[:, :, None, :]

        if query.device.type == "cuda":
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                context = functional.scaled_dot_product_attention(
                    query,
                    gathered_key,
                    gathered_value,
                    attn_mask=attention_bias[:, :, None, :],
                    dropout_p=0.0,
                    is_causal=False,
                    scale=1.0 / math.sqrt(HEAD_WIDTH),
                )
        else:
            context = functional.scaled_dot_product_attention(
                query,
                gathered_key,
                gathered_value,
                attn_mask=attention_bias[:, :, None, :],
                dropout_p=0.0,
                is_causal=False,
                scale=1.0 / math.sqrt(HEAD_WIDTH),
            )
        return context[:real_query_count, :, 0, :]


class LocalTransformerBlock(nn.Module):
    """One separate-residual pre-norm sparse local Transformer block."""

    def __init__(self) -> None:
        """Build the attention and SwiGLU sublayers."""
        super().__init__()
        self.attention_norm = nn.LayerNorm(TOKEN_WIDTH)
        self.attention = SparseLocalSoftmaxAttention()
        self.ffn_norm = nn.LayerNorm(TOKEN_WIDTH)
        self.ffn = SwiGLU()

    def forward(self, tokens: Tensor, graph: LocalAttentionGraph) -> Tensor:
        """Apply local attention then token-wise nonlinear mixing."""
        normalized = cast("Tensor", self.attention_norm(tokens))
        attention_update = cast("Tensor", self.attention(normalized, graph))
        attended = tokens + attention_update
        ffn_input = cast("Tensor", self.ffn_norm(attended))
        return attended + cast("Tensor", self.ffn(ffn_input))


class SigmoidPatchSummaryAttention(nn.Module):
    """One FP32 calibrated sigmoid read from global queries to all patches."""

    def __init__(self) -> None:
        """Build separate projections and zero-initialized head offsets."""
        super().__init__()
        self.query = nn.Linear(TOKEN_WIDTH, TOKEN_WIDTH, bias=False)
        self.kv = PackedLinear(
            TOKEN_WIDTH,
            (TOKEN_WIDTH, TOKEN_WIDTH),
            bias=False,
        )
        self.output = nn.Linear(TOKEN_WIDTH, TOKEN_WIDTH)
        self.head_offset = nn.Parameter(torch.zeros(ATTENTION_HEADS))

    def forward(self, queries: Tensor, patches: Tensor) -> Tensor:
        """Return 17 global updates without updating the patch sequence."""
        if queries.shape != (GLOBAL_TOKENS, TOKEN_WIDTH):
            raise ValueError("Global summary requires [17,192] queries")
        if patches.ndim != 2 or patches.shape[1] != TOKEN_WIDTH:
            raise ValueError("Global summary requires [N,192] patch tokens")
        if patches.shape[0] < 1:
            raise ValueError("Global summary requires at least one patch")
        query = cast("Tensor", self.query(queries)).reshape(
            GLOBAL_TOKENS, ATTENTION_HEADS, HEAD_WIDTH
        )
        key, value = (
            part.reshape(-1, ATTENTION_HEADS, HEAD_WIDTH)
            for part in self.kv.split(cast("Tensor", self.kv(patches)))
        )
        with torch.autocast(device_type=query.device.type, enabled=False):
            scores = torch.einsum(
                "mhd,nhd->mhn",
                query.float(),
                key.float(),
            ) / math.sqrt(HEAD_WIDTH)
            scores += self.head_offset.float()[None, :, None]
            patch_count = torch.scalar_tensor(
                patches.shape[0],
                dtype=torch.float32,
                device=scores.device,
            )
            scores -= patch_count.log()
            weights = torch.sigmoid(scores)
            context32 = torch.einsum("mhn,nhd->mhd", weights, value.float())
            context = context32.reshape(GLOBAL_TOKENS, TOKEN_WIDTH)
            return cast("Tensor", self.output(context))


class GlobalSummaryBlock(nn.Module):
    """The sole one-way all-patch read followed by global-token SwiGLU."""

    def __init__(self) -> None:
        """Build distinct query/patch norms and the global residual block."""
        super().__init__()
        self.query_norm = nn.LayerNorm(TOKEN_WIDTH)
        self.patch_norm = nn.LayerNorm(TOKEN_WIDTH)
        self.attention = SigmoidPatchSummaryAttention()
        self.ffn_norm = nn.LayerNorm(TOKEN_WIDTH)
        self.ffn = SwiGLU()

    def forward(self, global_tokens: Tensor, patches: Tensor) -> Tensor:
        """Update global tokens from patches without a patch-update return path."""
        normalized_queries = cast("Tensor", self.query_norm(global_tokens))
        normalized_patches = cast("Tensor", self.patch_norm(patches))
        attention_update = cast(
            "Tensor",
            self.attention(normalized_queries, normalized_patches),
        )
        with torch.autocast(device_type=global_tokens.device.type, enabled=False):
            attended = global_tokens.float() + attention_update.float()
            ffn_input = cast("Tensor", self.ffn_norm(attended))
            return attended + cast("Tensor", self.ffn(ffn_input))


class CLSOnlyGlobalBlock(nn.Module):
    """Compute only the consumed CLS row of global-token self-attention."""

    def __init__(self) -> None:
        """Build shared attention projections and the CLS-only FFN."""
        super().__init__()
        self.attention_norm = nn.LayerNorm(TOKEN_WIDTH)
        self.query = nn.Linear(TOKEN_WIDTH, TOKEN_WIDTH, bias=False)
        self.kv = PackedLinear(
            TOKEN_WIDTH,
            (TOKEN_WIDTH, TOKEN_WIDTH),
            bias=False,
        )
        self.output = nn.Linear(TOKEN_WIDTH, TOKEN_WIDTH)
        self.ffn_norm = nn.LayerNorm(TOKEN_WIDTH)
        self.ffn = SwiGLU()

    def forward(self, global_tokens: Tensor) -> Tensor:
        """Return one CLS vector while all 17 normalized tokens serve as K/V."""
        if global_tokens.shape != (GLOBAL_TOKENS, TOKEN_WIDTH):
            raise ValueError("CLS-only block requires [17,192] global tokens")
        with torch.autocast(device_type=global_tokens.device.type, enabled=False):
            tokens32 = global_tokens.float()
            normalized = cast("Tensor", self.attention_norm(tokens32))
            query = cast("Tensor", self.query(normalized[:1])).reshape(
                1, ATTENTION_HEADS, HEAD_WIDTH
            )
            key, value = (
                part.reshape(GLOBAL_TOKENS, ATTENTION_HEADS, HEAD_WIDTH)
                for part in self.kv.split(cast("Tensor", self.kv(normalized)))
            )
            attended = functional.scaled_dot_product_attention(
                query.transpose(0, 1),
                key.transpose(0, 1),
                value.transpose(0, 1),
                dropout_p=0.0,
                is_causal=False,
                scale=1.0 / math.sqrt(HEAD_WIDTH),
            ).transpose(0, 1)
            projected = cast(
                "Tensor",
                self.output(attended.reshape(1, TOKEN_WIDTH)),
            )
            cls = tokens32[0] + projected[0]
            return cast("Tensor", cls + self.ffn(self.ffn_norm(cls)))


class LocalGlobalMILClassifier(nn.Module):
    """Five-class complete-bag classifier with strict local/global directionality."""

    def __init__(self) -> None:
        """Build and initialize the single Spec 0026 model."""
        super().__init__()
        self.patch_encoder = LocalGlobalPatchEncoder()
        self.local_blocks = nn.ModuleList((
            LocalTransformerBlock(),
            LocalTransformerBlock(),
        ))
        self.global_tokens = nn.Parameter(torch.empty(GLOBAL_TOKENS, TOKEN_WIDTH))
        self.global_summary = GlobalSummaryBlock()
        self.cls_block = CLSOnlyGlobalBlock()
        self.final_norm = nn.LayerNorm(TOKEN_WIDTH)
        self.classifier = nn.Linear(TOKEN_WIDTH, len(CLASS_ORDER))
        self.apply(_initialize_module)
        nn.init.normal_(self.global_tokens, mean=0.0, std=0.02)

    def encode_patches(self, latents: Tensor, graph: LocalAttentionGraph) -> Tensor:
        """Return locally contextualized patches before any global-token read."""
        patches = cast("Tensor", self.patch_encoder(latents))
        _validate_tokens_and_graph(patches, graph)
        for block in self.local_blocks:
            patches = cast("Tensor", block(patches, graph))
        return patches

    def forward(self, latents: Tensor, graph: LocalAttentionGraph) -> Tensor:
        """Return raw `[5]` logits for one complete WSI bag."""
        patches = self.encode_patches(latents, graph)
        global_tokens = cast("Tensor", self.global_summary(self.global_tokens, patches))
        cls = cast("Tensor", self.cls_block(global_tokens))
        with torch.autocast(device_type=cls.device.type, enabled=False):
            normalized = cast("Tensor", self.final_norm(cls.float()))
            return functional.linear(
                normalized,
                self.classifier.weight.float(),
                self.classifier.bias.float(),
            )


def make_paired_local_global_mil_models() -> tuple[
    LocalGlobalMILClassifier,
    LocalGlobalMILClassifier,
]:
    """Construct independent normal/SO(2) branches with byte-identical states."""
    normal_model = LocalGlobalMILClassifier()
    so2_model = LocalGlobalMILClassifier()
    so2_model.load_state_dict(normal_model.state_dict())
    return normal_model, so2_model


def local_global_mil_adamw_parameter_groups(
    model: LocalGlobalMILClassifier,
    *,
    weight_decay: float = 5e-3,
) -> tuple[AdamWParameterGroup, AdamWParameterGroup]:
    """Partition parameters by the exact Spec 0026 semantic decay policy."""
    if weight_decay < 0:
        raise ValueError("weight_decay may not be negative")
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        is_content_token = name == "global_tokens" or name.endswith(".null_key")
        if (
            parameter.ndim >= 2
            and not name.endswith(".relative_bias")
            and not is_content_token
        ):
            decay.append(parameter)
        else:
            no_decay.append(parameter)
    return (
        AdamWParameterGroup(name="decay", params=decay, weight_decay=weight_decay),
        AdamWParameterGroup(name="no_decay", params=no_decay, weight_decay=0.0),
    )


def _initialize_module(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(module, PackedLinear):
        module.reset_parameters()
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=1.0)
        if module.bias is not None:  # pyright: ignore[reportUnnecessaryComparison]
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.GroupNorm, nn.LayerNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def _validate_tokens_and_graph(tokens: Tensor, graph: LocalAttentionGraph) -> None:
    if tokens.ndim != 2 or tokens.shape[1] != TOKEN_WIDTH:
        raise ValueError("Local attention requires [N,192] patch tokens")
    if tokens.shape[0] != graph.node_count:
        raise ValueError("Graph node count must match the patch-token count")
    expected_shape = (graph.node_count, LOCAL_MAX_DEGREE)
    if tuple(graph.neighbor_index.shape) != expected_shape:
        raise ValueError("neighbor_index must have shape [N,25]")
    if tuple(graph.neighbor_valid.shape) != expected_shape:
        raise ValueError("neighbor_valid must have shape [N,25]")
    if tuple(graph.radial_code.shape) != expected_shape:
        raise ValueError("radial_code must have shape [N,25]")
    if graph.neighbor_index.dtype != torch.int64:
        raise TypeError("neighbor_index must be int64")
    if graph.neighbor_valid.dtype != torch.bool:
        raise TypeError("neighbor_valid must be bool")
    if graph.radial_code.dtype != torch.uint8:
        raise TypeError("radial_code must be uint8")
    graph_devices = {
        graph.neighbor_index.device,
        graph.neighbor_valid.device,
        graph.radial_code.device,
    }
    if graph_devices != {tokens.device}:
        raise ValueError("Graph arrays and patch tokens must be on the same device")


def _local_query_chunks(node_count: int) -> tuple[tuple[int, int], ...]:
    """Return a complete, disjoint partition of local-attention queries."""
    return tuple(
        (start, min(start + LOCAL_CHUNK_SIZE, node_count))
        for start in range(0, node_count, LOCAL_CHUNK_SIZE)
    )


def _validate_graph_arrays(graph: LocalAttentionGraph) -> None:
    node_count = graph.node_count
    if graph.expected_instance_count != node_count or node_count < 1:
        raise ValueError("Graph node count must match its complete bag count")
    if graph.lattice_coordinates.dtype != torch.int64:
        raise TypeError("lattice_coordinates must be int64")
    if tuple(graph.lattice_coordinates.shape) != (node_count, 2):
        raise ValueError("lattice_coordinates must have shape [N,2]")
    if graph.lattice_coordinates.device.type != "cpu":
        raise ValueError("Canonical lattice coordinates must remain on CPU")
    expected_shape = (node_count, LOCAL_MAX_DEGREE)
    if tuple(graph.neighbor_index.shape) != expected_shape:
        raise ValueError("neighbor_index must have shape [N,25]")
    if tuple(graph.neighbor_valid.shape) != expected_shape:
        raise ValueError("neighbor_valid must have shape [N,25]")
    if tuple(graph.radial_code.shape) != expected_shape:
        raise ValueError("radial_code must have shape [N,25]")
    if graph.neighbor_index.dtype != torch.int64:
        raise TypeError("neighbor_index must be int64")
    if graph.neighbor_valid.dtype != torch.bool:
        raise TypeError("neighbor_valid must be bool")
    if graph.radial_code.dtype != torch.uint8:
        raise TypeError("radial_code must be uint8")

    neighbor_index = graph.neighbor_index.detach().cpu()
    neighbor_valid = graph.neighbor_valid.detach().cpu()
    radial_code = graph.radial_code.detach().cpu()
    invalid = ~neighbor_valid
    if not neighbor_index[invalid].eq(NEIGHBOR_PADDING_INDEX).all():
        raise ValueError("Invalid graph slots must use neighbor sentinel -1")
    if not radial_code[invalid].eq(RADIAL_PADDING_CODE).all():
        raise ValueError("Invalid graph slots must use radial sentinel 255")
    if neighbor_valid[:, 1:].logical_and(~neighbor_valid[:, :-1]).any():
        raise ValueError("Valid graph neighbours must precede padded slots")
    valid_indices = neighbor_index[neighbor_valid]
    if valid_indices.lt(0).any() or valid_indices.ge(node_count).any():
        raise ValueError("Valid graph neighbour indices are out of bounds")
    valid_codes = radial_code[neighbor_valid]
    if valid_codes.ge(len(RADIAL_CODEBOOK)).any():
        raise ValueError("Valid graph radial codes are out of bounds")
    rows = torch.arange(node_count)[:, None]
    has_self = neighbor_valid.logical_and(neighbor_index.eq(rows)).any(dim=1)
    if not has_self.all():
        raise ValueError("Every graph row must contain its self edge")
    coordinates = graph.lattice_coordinates
    gathered = coordinates[neighbor_index.clamp_min(0)]
    delta = gathered - coordinates[:, None, :]
    valid_delta = delta[neighbor_valid]
    if valid_delta.abs().amax(dim=1).gt(LOCAL_RADIUS).any():
        raise ValueError("Graph contains an edge outside the local radius")
    squared_radius = valid_delta.square().sum(dim=1)
    decoded_radius = torch.tensor(RADIAL_CODEBOOK)[valid_codes.long()]
    if not squared_radius.eq(decoded_radius).all():
        raise ValueError("Graph radial codes do not match coordinate offsets")


def _graph_identity_sha256(
    *,
    wsi_id: int,
    coordinates: Sequence[tuple[int, int]],
    neighbor_index: Tensor,
    neighbor_valid: Tensor,
    radial_code: Tensor,
) -> str:
    fields: tuple[tuple[str, bytes], ...] = (
        ("schema", _GRAPH_SCHEMA),
        ("wsi_id", struct.pack("<q", wsi_id)),
        ("shape", struct.pack("<QQ", len(coordinates), LOCAL_MAX_DEGREE)),
        ("dtypes", _GRAPH_DTYPES),
        ("sentinels", struct.pack("<qB", NEIGHBOR_PADDING_INDEX, RADIAL_PADDING_CODE)),
        ("codebook", struct.pack("<6Q", *RADIAL_CODEBOOK)),
        ("coordinates", _int64_le_bytes(coordinates)),
        ("neighbor_index", _tensor_bytes(neighbor_index, dtype="<i8")),
        ("neighbor_valid", _tensor_bytes(neighbor_valid, dtype="u1")),
        ("radial_code", _tensor_bytes(radial_code, dtype="u1")),
    )
    digest = hashlib.sha256()
    for tag, payload in fields:
        encoded_tag = tag.encode("utf-8")
        digest.update(struct.pack("<H", len(encoded_tag)))
        digest.update(encoded_tag)
        digest.update(struct.pack("<Q", len(payload)))
        digest.update(payload)
    return digest.hexdigest()


def _int64_le_bytes(values: Iterable[tuple[int, int]]) -> bytes:
    array = np.asarray(tuple(values), dtype="<i8")
    return array.tobytes(order="C")


def _tensor_bytes(tensor: Tensor, *, dtype: str) -> bytes:
    array = tensor.detach().cpu().contiguous().numpy().astype(dtype, copy=False)
    return array.tobytes(order="C")


__all__ = [
    "CLASS_ORDER",
    "EXPECTED_PARAMETER_COUNT",
    "LOCAL_GLOBAL_MIL_CONFIG",
    "AdamWParameterGroup",
    "CLSOnlyGlobalBlock",
    "GlobalSummaryBlock",
    "LocalAttentionGraph",
    "LocalGlobalMILClassifier",
    "LocalGlobalMILConfig",
    "LocalTransformerBlock",
    "PackedLinear",
    "SigmoidPatchSummaryAttention",
    "SparseLocalSoftmaxAttention",
    "SwiGLU",
    "build_local_attention_graph",
    "local_global_mil_adamw_parameter_groups",
    "make_paired_local_global_mil_models",
]
