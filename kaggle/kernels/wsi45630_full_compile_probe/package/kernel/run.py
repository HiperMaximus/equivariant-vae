# Copyright 2026 HiperMaximus
# ruff: noqa: ANN001, ANN202, BLE001, DOC201, DOC501, EM101, EM102, FBT003, INP001, PLC0415, PLR0911, PLR0913, PLR0914, PLR0915, PLR0916, PLR0917, PLR2004, PLW0717, S102, S404, SLF001, T201, TRY003
"""Run the exact Spec 0034 full compiled fixed-25 MIL probe."""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import math
import os
import statistics
import subprocess
import sys
import textwrap
import time
import traceback
from collections import Counter
from operator import itemgetter
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# fmt: off
# BEGIN EMBEDDED SPEC0032 MODEL
if False:
    # Copyright 2026 HiperMaximus
    # ruff: noqa: C901, COM812, DOC201, DOC501, EM101, PLC2801, PLR0912, PLR0913, PLR0914, PLR0915, PLR0917, PLR2004, SLF001, TRY003
    """The AMP-first fixed width-192 local-global MIL architecture."""

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
            projected_dtype = query.dtype
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
            context = context32.to(dtype=projected_dtype).reshape(
                GLOBAL_TOKENS,
                TOKEN_WIDTH,
            )
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
            attended = global_tokens + attention_update
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
            normalized = cast("Tensor", self.attention_norm(global_tokens))
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
            projected = cast("Tensor", self.output(attended.reshape(1, TOKEN_WIDTH)))
            cls = global_tokens[0] + projected[0]
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
        weight_decay: float = 1e-4,
    ) -> tuple[AdamWParameterGroup, AdamWParameterGroup]:
        """Partition parameters by the exact Spec 0026 semantic decay policy."""
        if weight_decay < 0:
            raise ValueError("weight_decay may not be negative")
        decay: list[nn.Parameter] = []
        no_decay: list[nn.Parameter] = []
        for name, parameter in model.named_parameters():
            if parameter.ndim >= 2 and not name.endswith(".relative_bias"):
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
# END EMBEDDED SPEC0032 MODEL

# BEGIN EMBEDDED SPEC0033 CANDIDATE
if False:
    # Copyright 2026 HiperMaximus
    """Exact native candidates for the fixed-degree local-attention bakeoff."""

    import math
    from dataclasses import dataclass
    from typing import TYPE_CHECKING, cast

    import torch
    from torch import Tensor

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

    def fixed25_inductor_attention(
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

    def edge_segment_inductor_attention(
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
# END EMBEDDED SPEC0033 CANDIDATE
# fmt: on

SPEC0034_FULL_COMPILED_FIXED25_PROBE_READY = True
INPUT_ROOT = Path("/kaggle/input")
OUTPUT_PATH = Path("/kaggle/working/spec0034_full_compiled_fixed25_mil_probe.json")
CONTRACT_JSON = r"""{"authorization":"spec0034_pinned_torch_retry_v7_authorized","candidate":{"backend":"whole_bag_fixed25_inductor","sha256":"bd16522fc5192ec330a0f46ba8f6653e6141f9c746f670c4dcbb3330d2f88a5c","source":"src/eqvae/models/local_attention_candidates.py"},"compile":{"backend":"inductor","dynamic_axes":"N_only_permissive_cached_specialization","fullgraph":true,"mode":"max-autotune-no-cudagraphs","optimizer":"grad_scaler_and_native_fused_adamw_eager","recompile_limit":3},"correctness":{"amp_skip_gate":"eager_replay_compiled_histories_must_match","criterion":"compiler_training_effect_lte_max_amp_or_repeat_effect","decision_units":"per_parameter_optimizer_state_and_behavior_per_step","precision_control":"same_fixed25_eager_fp32","repeat_control":"same_fixed25_eager_amp","steps":5},"diagnosis_index":1,"execution":{"benchmark_workload":"single_wsi_repeated_not_epoch_requeue","branch_devices":{"normal_vae":0,"so2_vae":1},"branch_failure_policy":"record_and_continue_other_branch","checkpointing":false,"compile_warmups":2,"cudagraphs":false,"dynamic_reuse_bag_size":64,"fallback":null,"measured_committed_steps_required":5,"measured_steps":5},"initialization_seed":3401,"input_dataset":{"contract_sha256":"99bb4d2f60558aee9691b67be4867ffae434bc306581a000fd5d72a6befac660","pointer_sha256":"08e461846bf16efebac707c82962762f49837916986b29aee0dcd6ca1fc31c6c","reference":"maximusshtefan/eqvae-wsi45630-capacity-inputs","version":1},"kernel_sources":[{"reference":"maximusshtefan/eqvae-ubc-ocean-latent-run-04","version":1},{"reference":"maximusshtefan/eqvae-ubc-ocean-cancer-latent-top-up","version":1},{"reference":"maximusshtefan/eqvae-wsi45630-completion","version":1}],"model":{"parameter_count":1513055,"sha256":"1609e56ed9175e6afb4cbd6db0d91ffb73ee5765390f28d1b6c230946c7e3955","source":"src/eqvae/models/local_global_mil.py"},"optimizer":{"capturable":true,"fused":true,"learning_rate":0.0002,"matrix_weight_decay":0.0001,"name":"AdamW"},"output":"spec0034_full_compiled_fixed25_mil_probe.json","patch_count":32595,"precision":{"autocast":"float16","classifier_loss":"float32","grad_scaler":{"api":"torch.amp.GradScaler","growth_interval":1000000,"init_scale":32768.0,"overflow_policy":"skip_update_and_continue"},"input":"float16_channels_last","normalization":"standard_pytorch_amp_policy"},"runtime_dependency":{"cuda_wheel":"cu130","index_url":"https://download.pytorch.org/whl/cu130","install_scope":"torch_only_no_domain_libraries","torch":"2.14.0"},"schema_version":"spec0034.full_compiled_fixed25_mil.v1","scope":"capacity_optimization_only_not_learning_or_evaluation","spec_sha256":"8dc014d204a7ccf627b90a58f0595cb4bad1045809aa0bc0844311fd5635cdb0","wsi_id":45630}"""
CONTRACT_SHA256 = "55a49b3682a6c624920b7b69edc4ce2cddfc476559da6722c2564a9cdad5a328"
EMBEDDED_MODEL_SHA256 = (
    "16b03163596ca070e276625657467a7507f617d6db216b1f00a3fcde15f1e9e3"
)
EMBEDDED_CANDIDATE_SHA256 = (
    "e432147c449f648332ac3f8df460c3c535b59b57aa9e2cfd5a8183bb74002ed2"
)
INPUT_CONTRACT_NAME = "wsi45630_capacity_input.json"
INPUT_CONTRACT_SHA256 = (
    "99bb4d2f60558aee9691b67be4867ffae434bc306581a000fd5d72a6befac660"
)
POINTER_SHA256 = "08e461846bf16efebac707c82962762f49837916986b29aee0dcd6ca1fc31c6c"
INPUT_DATASET_REFERENCE = "maximusshtefan/eqvae-wsi45630-capacity-inputs"
KERNEL_SOURCES = (
    "maximusshtefan/eqvae-ubc-ocean-latent-run-04",
    "maximusshtefan/eqvae-ubc-ocean-cancer-latent-top-up",
    "maximusshtefan/eqvae-wsi45630-completion",
)
WSI_ID = 45_630
PATCH_COUNT = 32_595
DIAGNOSIS_INDEX = 1
PARAMETER_COUNT = 1_513_055
GRAD_SCALER_INIT_SCALE = 32_768.0
GRAD_SCALER_GROWTH_INTERVAL = 1_000_000
DYNAMIC_REUSE_COUNT = 64
WARMUP_STEPS = 2
MEASURED_STEPS = 5
MIN_RESERVED_HEADROOM = 512 * 1024 * 1024
TRAINING_EQUIVALENCE_STEPS = 5
PINNED_TORCH_VERSION = "2.14.0"
PINNED_TORCH_CUDA = "13.0"
PINNED_TORCH_INDEX = "https://download.pytorch.org/whl/cu130"


def sha256(path):
    """Stream one file hash without copying large inputs."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def embedded_source(begin, end, expected):
    """Extract and authenticate one generated source block."""
    source = Path(__file__).read_text(encoding="utf-8")
    if source.count(begin) != 1 or source.count(end) != 1:
        raise RuntimeError("Embedded source markers differ")
    block = source.split(begin, 1)[1].split(end, 1)[0]
    if not block.startswith("if False:\n"):
        raise RuntimeError("Embedded source guard differs")
    value = textwrap.dedent(block.removeprefix("if False:\n"))
    if hashlib.sha256(value.encode()).hexdigest() != expected:
        raise RuntimeError("Embedded source digest differs")
    return value


def resolve_package():
    """Authenticate the exact execution contract and both source snapshots."""
    if hashlib.sha256(CONTRACT_JSON.encode()).hexdigest() != CONTRACT_SHA256:
        raise RuntimeError("Spec 0034 contract bytes differ")
    model = embedded_source(
        "# BEGIN EMBEDDED SPEC0032 MODEL\n",
        "# END EMBEDDED SPEC0032 MODEL\n",
        EMBEDDED_MODEL_SHA256,
    )
    candidate = embedded_source(
        "# BEGIN EMBEDDED SPEC0033 CANDIDATE\n",
        "# END EMBEDDED SPEC0033 CANDIDATE\n",
        EMBEDDED_CANDIDATE_SHA256,
    )
    contract = json.loads(CONTRACT_JSON)
    if (
        contract.get("schema_version") != "spec0034.full_compiled_fixed25_mil.v1"
        or contract.get("authorization") != "spec0034_pinned_torch_retry_v7_authorized"
        or contract.get("scope")
        != "capacity_optimization_only_not_learning_or_evaluation"
        or contract.get("model", {}).get("parameter_count") != PARAMETER_COUNT
        or contract.get("candidate", {}).get("backend") != "whole_bag_fixed25_inductor"
        or contract.get("input_dataset", {}).get("reference") != INPUT_DATASET_REFERENCE
        or contract.get("input_dataset", {}).get("version") != 1
        or contract.get("input_dataset", {}).get("contract_sha256")
        != INPUT_CONTRACT_SHA256
        or contract.get("input_dataset", {}).get("pointer_sha256") != POINTER_SHA256
        or contract.get("kernel_sources")
        != [{"reference": source, "version": 1} for source in KERNEL_SOURCES]
        or contract.get("wsi_id") != WSI_ID
        or contract.get("patch_count") != PATCH_COUNT
        or contract.get("execution", {}).get("fallback") is not None
    ):
        raise RuntimeError("Spec 0034 execution contract differs")
    return contract, model, candidate


def activate_sources(model, candidate):
    """Define the current model and fixed-25 candidate after torch bootstrap."""
    namespace = globals()
    exec(
        compile(
            "from __future__ import annotations\n" + model,
            "spec0032_model.py",
            "exec",
        ),
        namespace,
    )
    exec(
        compile(
            "from __future__ import annotations\n" + candidate,
            "spec0033_candidate.py",
            "exec",
        ),
        namespace,
    )


def resolve_input_bundle():
    """Authenticate every file in the existing private pointer/source bundle."""
    matches = list(INPUT_ROOT.rglob(INPUT_CONTRACT_NAME))
    if len(matches) != 1 or sha256(matches[0]) != INPUT_CONTRACT_SHA256:
        raise RuntimeError("Expected the exact WSI45630 capacity input contract")
    root = matches[0].parent
    contract = json.loads(matches[0].read_text(encoding="utf-8"))
    if (
        contract.get("dataset_reference") != INPUT_DATASET_REFERENCE
        or contract.get("wsi_id") != WSI_ID
        or contract.get("patch_count") != PATCH_COUNT
        or contract.get("diagnosis_index") != DIAGNOSIS_INDEX
        or contract.get("kernel_sources") != list(KERNEL_SOURCES)
    ):
        raise RuntimeError("Mounted WSI45630 input scope differs")
    for name, record in contract["files"].items():
        path = root / name
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise RuntimeError("Mounted WSI45630 input bytes differ")
    observed = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    if observed != {*contract["files"], INPUT_CONTRACT_NAME}:
        raise RuntimeError("Unexpected file in WSI45630 capacity input dataset")
    return root, contract


def resolve_sources(catalog):
    """Resolve attached latent outputs by owner-qualified sidecar hashes."""
    roots = {}
    with catalog.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            matches = [
                path.parent
                for path in INPUT_ROOT.rglob(row["sidecar_name"])
                if path.is_file()
                and path.stat().st_size == int(row["sidecar_bytes"])
                and sha256(path) == row["sidecar_sha256"]
            ]
            if len(matches) != 1:
                raise RuntimeError("Expected one hash-matched producer sidecar")
            source = row["kaggle_source"]
            if source in roots and roots[source] != matches[0]:
                raise RuntimeError("Paired producer files have different roots")
            roots[source] = matches[0]
    return roots


def load_inputs(torch, root, contract, result):
    """Load both complete aligned bags directly to FP16 channels-last CUDA."""
    from eqvae.data.supervised_latents import (
        LogicalPointer,
        SupervisedLatentStore,
        WSIInstance,
    )

    catalog = root / "probe/physical_parts.csv"
    source_roots = resolve_sources(catalog)
    if set(source_roots) != set(contract["kernel_sources"]):
        raise RuntimeError("Unexpected latent producer set")
    with (root / "probe/pointers.csv").open(newline="", encoding="utf-8") as handle:
        rows = [
            {key: int(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]
    identity = tuple(
        (
            row["atlas_row_index"],
            row["wsi_id"],
            row["x"],
            row["y"],
            row["part"],
            row["file_index"],
        )
        for row in rows
    )
    if (
        len(identity) != PATCH_COUNT
        or {row[1] for row in identity} != {WSI_ID}
        or len({(row[2], row[3]) for row in identity}) != PATCH_COUNT
        or identity != tuple(sorted(identity, key=itemgetter(3, 2)))
    ):
        raise RuntimeError("Expected every unique WSI45630 coordinate in y,x order")
    pointers = tuple(LogicalPointer(row[4], row[5]) for row in identity)
    instances = tuple(
        WSIInstance(
            instance_row=index,
            atlas_row_index=row[0],
            wsi_id=row[1],
            x=row[2],
            y=row[3],
            diagnosis_label="EC",
            diagnosis_index=DIAGNOSIS_INDEX,
            split="train",
            pointer=pointers[index],
        )
        for index, row in enumerate(identity)
    )
    bags = []
    shared_reads = None
    result["input_reads"] = {}
    for device, name in enumerate(("normal_vae", "so2_vae")):
        with SupervisedLatentStore(
            catalog_path=catalog,
            model_name=name,
            source_roots=source_roots,
        ) as store:
            latents, reads = store.read_rows(pointers)
        counts = Counter(str(read.part) for read in reads)
        if (
            tuple(latents.shape) != (PATCH_COUNT, 16, 32, 32)
            or dict(counts) != contract["part_counts"]
            or not torch.isfinite(latents).all().item()
            or (shared_reads is not None and reads != shared_reads)
        ):
            raise RuntimeError("Full paired bag identity, alignment, or values differ")
        shared_reads = reads
        bag = latents.to(
            device=f"cuda:{device}",
            dtype=torch.float16,
            memory_format=torch.channels_last,
        )
        if bag.dtype != torch.float16 or not bag.is_contiguous(
            memory_format=torch.channels_last,
        ):
            raise RuntimeError("CUDA bag is not direct FP16 channels-last")
        bags.append(bag)
        result["input_reads"][name] = {
            "rows": PATCH_COUNT,
            "part_counts": dict(counts),
            "cuda_dtype": str(bag.dtype),
            "cuda_stride": list(bag.stride()),
        }
        del latents
    return bags, instances


def make_instances(count):
    """Create a valid compact graph used only for correctness/dynamic reuse."""
    from eqvae.data.supervised_latents import LogicalPointer, WSIInstance

    width = 8
    coordinates = [(index % width, index // width) for index in range(count)]
    return tuple(
        WSIInstance(
            instance_row=index,
            atlas_row_index=index,
            wsi_id=WSI_ID,
            x=x * 256,
            y=y * 256,
            diagnosis_label="EC",
            diagnosis_index=DIAGNOSIS_INDEX,
            split="train",
            pointer=LogicalPointer(1, index),
        )
        for index, (x, y) in enumerate(coordinates)
    )


def configure_torch(torch):
    """Enable the fastest safe dynamic-shape T4 runtime flags."""
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)
    torch.set_float32_matmul_precision("high")
    torch._functorch.config.backward_pass_autocast = "off"  # noqa: S105 -- PyTorch mode, not a credential.
    if hasattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction"):
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True


def compile_callable(torch, function, *, fullgraph):
    """Compile with the selected dynamic no-CUDA-graphs recipe."""
    kwargs = {
        "backend": "inductor",
        "fullgraph": fullgraph,
        "dynamic": None,
        "mode": "max-autotune-no-cudagraphs",
    }
    signature = inspect.signature(torch.compile)
    if "recompile_limit" in signature.parameters:
        kwargs["recompile_limit"] = 3
    if "isolate_recompiles" in signature.parameters:
        kwargs["isolate_recompiles"] = True
    return torch.compile(function, **kwargs)


def hint_dynamic_bag_axis(torch, latents, graph):
    """Hint only the physical bag axis without forbidding specialization."""
    torch._dynamo.maybe_mark_dynamic(latents, 0)
    for tensor in (graph.neighbor_index, graph.neighbor_valid, graph.radial_code):
        torch._dynamo.maybe_mark_dynamic(tensor, 0)


def make_model(torch, state, device):
    """Build the exact state-compatible fixed-25 full network."""
    model = LocalGlobalMILClassifier()
    model.load_state_dict(state)
    before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    use_whole_bag_fixed25_attention(model)
    after = model.state_dict()
    if before.keys() != after.keys() or any(
        not torch.equal(value, after[name]) for name, value in before.items()
    ):
        raise RuntimeError("Fixed-25 replacement changed model state")
    if not all(
        isinstance(block.attention, WholeBagFixed25Attention)
        for block in model.local_blocks
    ):
        raise RuntimeError("Both local blocks must use whole-bag fixed-25")
    model = model.to(device=device, memory_format=torch.channels_last).train()
    if sum(parameter.numel() for parameter in model.parameters()) != PARAMETER_COUNT:
        raise RuntimeError("Model parameter count differs")
    return model


def initialize_nonzero_attention(torch, model):
    """Prevent a zero-initialization correctness test from becoming vacuous."""
    with torch.no_grad():
        for block_index, block in enumerate(model.local_blocks):
            attention = block.attention
            attention.relative_bias.copy_(
                torch.linspace(-0.2, 0.2, attention.relative_bias.numel()).reshape_as(
                    attention.relative_bias,
                )
                + block_index * 0.01,
            )
            attention.null_key.copy_(
                torch.linspace(-0.1, 0.1, attention.null_key.numel()).reshape_as(
                    attention.null_key,
                ),
            )
            attention.null_bias.copy_(
                torch.linspace(-0.05, 0.05, attention.null_bias.numel()),
            )


def tensor_metrics(torch, actual, reference):
    """Measure a tensor difference without assigning a pass threshold."""
    actual = actual.detach().float().cpu()
    reference = reference.detach().float().cpu()
    delta = actual - reference
    reference_norm = float(torch.linalg.vector_norm(reference).item())
    actual_norm = float(torch.linalg.vector_norm(actual).item())
    return {
        "finite": bool(torch.isfinite(actual).all().item()),
        "max_abs": float(delta.abs().max().item()),
        "relative_l2": float(torch.linalg.vector_norm(delta).item())
        / max(reference_norm, 1e-12),
        "cosine": float((actual.flatten() * reference.flatten()).sum().item())
        / max(reference_norm * actual_norm, 1e-12),
        "reference_norm": reference_norm,
    }


def semantic_parameter_group(name):
    """Map parameters into training-relevant blocks for effect accounting."""
    if name.startswith("patch_encoder."):
        return "patch_encoder"
    if name.startswith("local_blocks.0.attention.") and name.rsplit(".", 1)[-1] in {
        "relative_bias",
        "null_key",
        "null_bias",
    }:
        return "local_0_structure"
    if name.startswith("local_blocks.0."):
        return "local_0_remaining"
    if name.startswith("local_blocks.1.attention.") and name.rsplit(".", 1)[-1] in {
        "relative_bias",
        "null_key",
        "null_bias",
    }:
        return "local_1_structure"
    if name.startswith("local_blocks.1."):
        return "local_1_remaining"
    if name == "global_tokens" or name.startswith("global_summary."):
        return "global_summary"
    return "cls_and_head"


def grouped_vectors(torch, named_values):
    """Concatenate named tensors only within semantic model blocks."""
    grouped = {}
    for name, value in named_values.items():
        grouped.setdefault(semantic_parameter_group(name), []).append(
            value.detach().float().reshape(-1),
        )
    return {name: torch.cat(values).cpu() for name, values in sorted(grouped.items())}


def named_vectors(named_values):
    """Keep training-state effects separate for every named parameter."""
    return {
        name: value.detach().float().reshape(-1).cpu()
        for name, value in named_values.items()
    }


def accumulate_training_effect(
    torch,
    totals,
    key,
    fp32,
    eager,
    replay,
    compiled,
):
    """Measure compiler impact against AMP precision and repeat controls."""
    row = totals.setdefault(
        key,
        {
            "compiler_squared": 0.0,
            "amp_squared": 0.0,
            "replay_squared": 0.0,
            "compiled_fp32_squared": 0.0,
        },
    )
    for field, left, right in (
        ("compiler_squared", compiled, eager),
        ("amp_squared", eager, fp32),
        ("replay_squared", replay, eager),
        ("compiled_fp32_squared", compiled, fp32),
    ):
        delta = left.detach().double().cpu() - right.detach().double().cpu()
        row[field] += float(torch.dot(delta, delta).item())


def finalize_training_effects(totals):
    """Bound compiler-induced training impact by measured AMP/repeat impact."""
    rows = {}
    for key, source in sorted(totals.items()):
        compiler = source["compiler_squared"] ** 0.5
        amp = source["amp_squared"] ** 0.5
        replay = source["replay_squared"] ** 0.5
        compiled_fp32 = source["compiled_fp32_squared"] ** 0.5
        accepted = max(amp, replay)
        finite = all(
            math.isfinite(value)
            for value in (compiler, amp, replay, compiled_fp32, accepted)
        )
        passed = finite and (
            compiler == 0.0  # noqa: RUF069 -- zero envelope requires identity.
            if accepted == 0.0  # noqa: RUF069
            else compiler <= accepted
        )
        rows[key] = {
            "compiler_effect": compiler,
            "amp_effect": amp,
            "replay_amp_eager_effect": replay,
            "compiled_fp32_diagnostic": compiled_fp32,
            "accepted_training_effect": accepted,
            "finite": finite,
            "effect_ratio": (
                0.0
                if compiler == 0.0 and accepted == 0.0  # noqa: RUF069
                else None
                if accepted == 0.0  # noqa: RUF069
                else compiler / accepted
            ),
            "pass": passed,
        }
    return rows


def correctness_loss_function(torch, model, *, amp):
    """Return the unscaled loss path under the requested precision policy."""
    if amp:
        return make_loss_function(torch, model)

    def loss_function(latents, graph, target):
        logits = model(latents, graph)
        loss = torch.nn.functional.cross_entropy(
            logits.float().unsqueeze(0),
            target,
        )
        return loss, logits

    return loss_function


def correctness_optimizer(torch, model):
    """Construct the real fused AdamW path for one independent trajectory."""
    return torch.optim.AdamW(
        local_global_mil_adamw_parameter_groups(model, weight_decay=1e-4),
        lr=2e-4,
        fused=True,
        capturable=True,
    )


def make_grad_scaler(torch, *, enabled=True):
    """Construct the production AMP scaler with ordinary dynamic backoff."""
    return torch.amp.GradScaler(
        "cuda",
        init_scale=GRAD_SCALER_INIT_SCALE,
        growth_interval=GRAD_SCALER_GROWTH_INTERVAL,
        enabled=enabled,
    )


def gradient_diagnostics(torch, named_parameters):
    """Describe every missing or nonfinite unscaled parameter gradient."""
    rows = []
    for name, parameter in named_parameters.items():
        gradient = parameter.grad
        if gradient is None:
            rows.append({"parameter": name, "missing": True})
            continue
        finite = torch.isfinite(gradient)
        if bool(finite.all().item()):
            continue
        finite_values = gradient.detach()[finite]
        rows.append({
            "parameter": name,
            "missing": False,
            "dtype": str(gradient.dtype),
            "nan_count": int(torch.isnan(gradient).sum().item()),
            "positive_inf_count": int(torch.isposinf(gradient).sum().item()),
            "negative_inf_count": int(torch.isneginf(gradient).sum().item()),
            "finite_max_abs": (
                float(finite_values.abs().max().item())
                if finite_values.numel()
                else None
            ),
        })
    return rows


def correctness_training_step(
    torch,
    model,
    optimizer,
    scaler,
    function,
    bag,
    graph,
    target,
):
    """Execute one ordinary GradScaler attempt and expose its training effects."""
    named_parameters = dict(model.named_parameters())
    before = {
        name: parameter.detach().clone() for name, parameter in named_parameters.items()
    }
    optimizer.zero_grad(set_to_none=True)
    loss, logits = function(bag, graph, target)
    if not torch.isfinite(loss).item() or not torch.isfinite(logits).all().item():
        raise FloatingPointError("Correctness trajectory has nonfinite outputs")
    scale_before = float(scaler.get_scale())
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    bad_gradients = gradient_diagnostics(torch, named_parameters)
    missing = [row["parameter"] for row in bad_gradients if row["missing"]]
    if missing:
        raise RuntimeError(f"Correctness trajectory has missing gradients: {missing}")
    if bad_gradients and not scaler.is_enabled():
        raise FloatingPointError(
            f"FP32 correctness gradients are nonfinite: {bad_gradients}",
        )
    grouped_gradients = (
        None
        if bad_gradients
        else grouped_vectors(
            torch,
            {name: parameter.grad for name, parameter in named_parameters.items()},
        )
    )
    scaler.step(optimizer)
    scaler.update()
    scale_after = float(scaler.get_scale())
    step_skipped = scale_after < scale_before
    if scaler.is_enabled() and step_skipped != bool(bad_gradients):
        raise RuntimeError("GradScaler skip decision differs from gradient diagnostics")
    updates = named_vectors(
        {
            name: parameter.detach() - before[name]
            for name, parameter in named_parameters.items()
        },
    )
    trajectory = named_vectors(
        {name: parameter.detach() for name, parameter in named_parameters.items()},
    )
    first_moment = {}
    second_moment = {}
    steps = {}
    for name, parameter in named_parameters.items():
        state = optimizer.state.get(parameter, {})
        if "exp_avg" not in state or "exp_avg_sq" not in state or "step" not in state:
            raise RuntimeError(f"AdamW state is incomplete for {name}")
        if not torch.isfinite(parameter).all().item():
            raise FloatingPointError(f"Correctness parameter is nonfinite: {name}")
        if not torch.isfinite(updates[name]).all().item():
            raise FloatingPointError(f"Correctness update is nonfinite: {name}")
        if not torch.isfinite(state["exp_avg"]).all().item():
            raise FloatingPointError(f"Correctness first moment is nonfinite: {name}")
        if not torch.isfinite(state["exp_avg_sq"]).all().item():
            raise FloatingPointError(f"Correctness second moment is nonfinite: {name}")
        first_moment[name] = state["exp_avg"]
        second_moment[name] = state["exp_avg_sq"]
        steps[name] = int(state["step"].item())
    return {
        "loss": loss.detach().float().cpu(),
        "logits": logits.detach().float().cpu(),
        "gradient": grouped_gradients,
        "update": updates,
        "parameters": trajectory,
        "exp_avg": named_vectors(first_moment),
        "exp_avg_sq": named_vectors(second_moment),
        "steps": steps,
        "step_skipped": step_skipped,
        "scale_before": scale_before,
        "scale_after": scale_after,
        "bad_gradients": bad_gradients,
    }


def correctness_evaluation(torch, model, panel, graph):
    """Measure post-update behavior through the common eager-FP32 model path."""
    was_training = model.training
    model.eval()
    probabilities = []
    losses = []
    predictions = []
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        for bag, label in panel:
            logits = model(bag.float(), graph).float()
            sample_probabilities = torch.softmax(logits, dim=-1)
            loss = torch.nn.functional.cross_entropy(
                logits.unsqueeze(0),
                label,
            )
            probabilities.append(sample_probabilities)
            losses.append(loss.reshape(1))
            predictions.append(int(logits.argmax().item()))
    model.train(was_training)
    return {
        "probabilities": torch.cat(probabilities).cpu(),
        "losses": torch.cat(losses).cpu(),
        "predictions": predictions,
    }


def full_model_correctness(torch, base_state):
    """Compare actual eager/compiled AMP training effects with measured controls."""
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    instances = make_instances(25)
    graph = build_local_attention_graph(instances, expected_instance_count=25).to(
        "cuda:0",
    )
    torch.manual_seed(3402)
    base_latent = torch.randn(25, 16, 32, 32, dtype=torch.float32).contiguous(
        memory_format=torch.channels_last,
    )
    cpu_bags = [
        (
            torch.roll(base_latent, shifts=step, dims=0) * (1.0 + 0.05 * step)
            + 0.01 * step
        ).contiguous(memory_format=torch.channels_last)
        for step in range(TRAINING_EQUIVALENCE_STEPS)
    ]
    fp32_bags = [bag.to("cuda:0") for bag in cpu_bags]
    amp_bags = [bag.to("cuda:0", dtype=torch.float16) for bag in cpu_bags]
    targets = [
        torch.tensor([step % len(CLASS_ORDER)], device="cuda:0")
        for step in range(TRAINING_EQUIVALENCE_STEPS)
    ]
    panel = list(zip(fp32_bags, targets, strict=True))
    arm_names = ("fp32", "eager_amp", "eager_amp_replay", "compiled_amp")
    models = {name: make_model(torch, base_state, "cuda:0") for name in arm_names}
    optimizers = {
        name: correctness_optimizer(torch, models[name]) for name in arm_names
    }
    for name in arm_names:
        optimizer_state_ready(torch, models[name], optimizers[name])
    scalers = {
        name: make_grad_scaler(torch, enabled=name != "fp32") for name in arm_names
    }
    functions = {
        "fp32": correctness_loss_function(torch, models["fp32"], amp=False),
        "eager_amp": correctness_loss_function(torch, models["eager_amp"], amp=True),
        "eager_amp_replay": correctness_loss_function(
            torch,
            models["eager_amp_replay"],
            amp=True,
        ),
    }
    compiled_source = correctness_loss_function(
        torch,
        models["compiled_amp"],
        amp=True,
    )
    hint_dynamic_bag_axis(torch, amp_bags[0], graph)
    functions["compiled_amp"] = compile_callable(
        torch,
        compiled_source,
        fullgraph=True,
    )
    effect_totals = {}
    diagnostics = []
    prediction_diagnostics = []
    exact_steps = True
    scaler_histories_match = True
    committed_steps = dict.fromkeys(arm_names, 0)
    update_coverage = {
        arm: {name: False for name, _ in models[arm].named_parameters()}
        for arm in arm_names
    }
    for step in range(TRAINING_EQUIVALENCE_STEPS):
        rows = {}
        for name in arm_names:
            bag = fp32_bags[step] if name == "fp32" else amp_bags[step]
            rows[name] = correctness_training_step(
                torch,
                models[name],
                optimizers[name],
                scalers[name],
                functions[name],
                bag,
                graph,
                targets[step],
            )
            if not rows[name]["step_skipped"]:
                committed_steps[name] += 1
        expected_step = step + 1
        reference_step_keys = rows["fp32"]["steps"].keys()
        exact_steps &= all(
            row["steps"].keys() == reference_step_keys
            and set(row["steps"].values()) == {committed_steps[name]}
            for name, row in rows.items()
        )
        amp_signatures = {
            name: (
                rows[name]["step_skipped"],
                rows[name]["scale_before"],
                rows[name]["scale_after"],
            )
            for name in ("eager_amp", "eager_amp_replay", "compiled_amp")
        }
        scaler_histories_match &= len(set(amp_signatures.values())) == 1
        for metric in ("update", "parameters", "exp_avg", "exp_avg_sq"):
            parameter_names = rows["fp32"][metric].keys()
            if not all(row[metric].keys() == parameter_names for row in rows.values()):
                raise RuntimeError(f"Correctness parameter names differ for {metric}")
            for parameter_name in parameter_names:
                accumulate_training_effect(
                    torch,
                    effect_totals,
                    f"step_{expected_step}:{metric}:{parameter_name}",
                    rows["fp32"][metric][parameter_name],
                    rows["eager_amp"][metric][parameter_name],
                    rows["eager_amp_replay"][metric][parameter_name],
                    rows["compiled_amp"][metric][parameter_name],
                )
                if metric == "update":
                    for arm, row in rows.items():
                        update_coverage[arm][parameter_name] |= bool(
                            torch.count_nonzero(
                                row[metric][parameter_name],
                            ).item(),
                        )
        evaluations = {
            name: correctness_evaluation(torch, models[name], panel, graph)
            for name in arm_names
        }
        for metric in ("probabilities", "losses"):
            accumulate_training_effect(
                torch,
                effect_totals,
                f"step_{expected_step}:post_update_{metric}",
                evaluations["fp32"][metric],
                evaluations["eager_amp"][metric],
                evaluations["eager_amp_replay"][metric],
                evaluations["compiled_amp"][metric],
            )
        prediction_diagnostics.append({
            arm: evaluations[arm]["predictions"] for arm in arm_names
        })
        diagnostics.append({
            "step": expected_step,
            "label": int(targets[step].item()),
            "scalers": amp_signatures,
            "bad_gradients": {name: rows[name]["bad_gradients"] for name in arm_names},
            "eager_vs_compiled_gradients": (
                None
                if rows["eager_amp"]["gradient"] is None
                or rows["compiled_amp"]["gradient"] is None
                else {
                    group: tensor_metrics(
                        torch,
                        rows["compiled_amp"]["gradient"][group],
                        rows["eager_amp"]["gradient"][group],
                    )
                    for group in rows["eager_amp"]["gradient"]
                }
            ),
            "eager_vs_compiled_loss": tensor_metrics(
                torch,
                rows["compiled_amp"]["loss"],
                rows["eager_amp"]["loss"],
            ),
            "eager_vs_compiled_logits": tensor_metrics(
                torch,
                rows["compiled_amp"]["logits"],
                rows["eager_amp"]["logits"],
            ),
        })
    effects = finalize_training_effects(effect_totals)
    graph_count = int(torch._dynamo.utils.counters["stats"]["unique_graphs"])
    graph_breaks = int(sum(torch._dynamo.utils.counters["graph_break"].values()))
    complete_update_coverage = all(
        all(parameters.values()) for parameters in update_coverage.values()
    )
    correct = (
        all(row["pass"] for row in effects.values())
        and exact_steps
        and scaler_histories_match
        and complete_update_coverage
        and graph_count == 1
        and graph_breaks == 0
    )
    return {
        "correct": correct,
        "criterion": "compiler_training_effect_lte_max_amp_or_repeat_effect",
        "steps": TRAINING_EQUIVALENCE_STEPS,
        "effects": effects,
        "prediction_diagnostics": prediction_diagnostics,
        "optimizer_steps_exact": exact_steps,
        "scaler_histories_match": scaler_histories_match,
        "committed_steps": committed_steps,
        "per_parameter_update_coverage": update_coverage,
        "diagnostics": diagnostics,
        "compiled_unique_graphs": graph_count,
        "compiled_graph_breaks": graph_breaks,
    }


def optimizer_state_ready(torch, model, optimizer):
    """Materialize fused AdamW moments without retaining a parameter update."""
    model_before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    groups = [(group["lr"], group["weight_decay"]) for group in optimizer.param_groups]
    for group in optimizer.param_groups:
        group["lr"] = 0.0
        group["weight_decay"] = 0.0
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    model.load_state_dict(model_before)
    for state in optimizer.state.values():
        state["step"].zero_()
    optimizer.zero_grad(set_to_none=True)
    for group, values in zip(optimizer.param_groups, groups, strict=True):
        group["lr"], group["weight_decay"] = values


def gradients_are_finite(torch, model):
    """Require every parameter gradient to exist and remain finite."""
    checked = 0
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all().item():
            return False, checked, name
        checked += 1
    return True, checked, None


def make_loss_function(torch, model):
    """Create the autocast model plus FP32 loss numerical region."""

    def loss_function(latents, graph, target):
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(latents, graph)
            loss = torch.nn.functional.cross_entropy(
                logits.float().unsqueeze(0),
                target,
            )
        return loss, logits

    return loss_function


def validate_optimizer_state(torch, model, optimizer, expected_step):
    """Require complete finite AdamW state for every trainable parameter."""
    named_parameters = dict(model.named_parameters())
    if len(optimizer.state) != len(named_parameters):
        raise RuntimeError("AdamW state does not cover every parameter")
    for name, parameter in named_parameters.items():
        state = optimizer.state.get(parameter, {})
        if not {"step", "exp_avg", "exp_avg_sq"}.issubset(state):
            raise RuntimeError(f"AdamW state is incomplete for {name}")
        if int(state["step"].item()) != expected_step:
            raise RuntimeError(f"AdamW step differs for {name}")
        if not all(
            torch.isfinite(value).all().item()
            for value in (parameter, state["exp_avg"], state["exp_avg_sq"])
        ):
            raise FloatingPointError(f"AdamW parameter/state is nonfinite: {name}")


def timed_training_step(
    torch,
    model,
    optimizer,
    scaler,
    numerical,
    bag,
    graph,
    target,
):
    """Run one recommended AMP attempt and return synchronized timing."""
    optimizer.zero_grad(set_to_none=True)
    forward_start = torch.cuda.Event(enable_timing=True)
    forward_end = torch.cuda.Event(enable_timing=True)
    update_start = torch.cuda.Event(enable_timing=True)
    update_end = torch.cuda.Event(enable_timing=True)
    wall = time.perf_counter()
    forward_start.record()
    loss, logits = numerical(bag, graph, target)
    if not torch.isfinite(loss).item() or not torch.isfinite(logits).all().item():
        raise FloatingPointError("Nonfinite loss or logits")
    scale_before = float(scaler.get_scale())
    scaler.scale(loss).backward()
    forward_end.record()
    forward_end.synchronize()
    missing = [
        name for name, parameter in model.named_parameters() if parameter.grad is None
    ]
    if missing:
        raise RuntimeError(f"Training attempt has missing gradients: {missing}")
    update_start.record()
    scaler.step(optimizer)
    scaler.update()
    update_end.record()
    update_end.synchronize()
    scale_after = float(scaler.get_scale())
    step_skipped = scale_after < scale_before
    return {
        "loss": float(loss.detach().item()),
        "logits": [float(value) for value in logits.detach().float().cpu()],
        "forward_backward_ms": float(forward_start.elapsed_time(forward_end)),
        "optimizer_ms": float(update_start.elapsed_time(update_end)),
        "wall_ms": (time.perf_counter() - wall) * 1000,
        "gradient_parameter_count": len(tuple(model.parameters())),
        "amp_step_skipped": step_skipped,
        "scale_before": scale_before,
        "scale_after": scale_after,
    }


def summarize_steps(steps):
    """Return stable timing summaries without discarding raw samples."""
    output = {"samples": steps}
    for key in ("forward_backward_ms", "optimizer_ms", "wall_ms"):
        values = [step[key] for step in steps]
        output[key] = {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }
    return output


def run_branch(torch, name, device, bag, graph, small_graph, base_state):
    """Compile and measure one independent complete MIL branch."""
    with torch.cuda.device(device):
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        torch._inductor.metrics.reset()
        model = make_model(torch, base_state, device)
        optimizer = torch.optim.AdamW(
            local_global_mil_adamw_parameter_groups(model, weight_decay=1e-4),
            lr=2e-4,
            fused=True,
            capturable=True,
        )
        optimizer_state_ready(torch, model, optimizer)
        scaler = make_grad_scaler(torch)
        update_status = {
            "compiled": False,
            "fullgraph": False,
            "implementation": "torch_amp_grad_scaler_native_fused_adamw",
            "fallback": None,
        }
        torch._dynamo.utils.counters.clear()
        torch._inductor.metrics.reset()
        numerical = compile_callable(
            torch,
            make_loss_function(torch, model),
            fullgraph=True,
        )
        target = torch.tensor([DIAGNOSIS_INDEX], device=device)
        hint_dynamic_bag_axis(torch, bag, graph)
        small_bag = torch.randn(
            DYNAMIC_REUSE_COUNT,
            16,
            32,
            32,
            device=device,
            dtype=torch.float16,
        ).contiguous(memory_format=torch.channels_last)
        hint_dynamic_bag_axis(torch, small_bag, small_graph)
        total = torch.cuda.get_device_properties(device).total_memory
        free_before, _ = torch.cuda.mem_get_info(device)
        reserved_before = torch.cuda.memory_reserved(device)
        non_torch_baseline = max(0, total - free_before - reserved_before)
        compile_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss, _ = numerical(bag, graph, target)
        loss.backward()
        torch.cuda.synchronize(device)
        if not torch.isfinite(loss).item() or not gradients_are_finite(torch, model)[0]:
            raise FloatingPointError("Compile warmup produced nonfinite values")
        optimizer.zero_grad(set_to_none=True)
        compile_seconds = time.perf_counter() - compile_started
        numerical_graphs_after_real = int(
            torch._dynamo.utils.counters["stats"]["unique_graphs"],
        )
        small_loss, _ = numerical(small_bag, small_graph, target)
        small_loss.backward()
        torch.cuda.synchronize(device)
        if (
            not torch.isfinite(small_loss).item()
            or not gradients_are_finite(torch, model)[0]
        ):
            raise FloatingPointError("Dynamic reuse produced nonfinite values")
        optimizer.zero_grad(set_to_none=True)
        numerical_graphs_after_reuse = int(
            torch._dynamo.utils.counters["stats"]["unique_graphs"],
        )
        graph_breaks = int(sum(torch._dynamo.utils.counters["graph_break"].values()))
        specialization_count = (
            numerical_graphs_after_reuse - numerical_graphs_after_real
        )
        if specialization_count not in {0, 1} or graph_breaks:
            raise RuntimeError(
                "Numerical callable exceeded one cached specialization or broke",
            )
        del small_bag
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        warmup = [
            timed_training_step(
                torch,
                model,
                optimizer,
                scaler,
                numerical,
                bag,
                graph,
                target,
            )
            for _ in range(WARMUP_STEPS)
        ]
        measured = [
            timed_training_step(
                torch,
                model,
                optimizer,
                scaler,
                numerical,
                bag,
                graph,
                target,
            )
            for _ in range(MEASURED_STEPS)
        ]
        numerical_graphs_after_training = int(
            torch._dynamo.utils.counters["stats"]["unique_graphs"],
        )
        graph_breaks_after_training = int(
            sum(torch._dynamo.utils.counters["graph_break"].values()),
        )
        if (
            numerical_graphs_after_training != numerical_graphs_after_reuse
            or graph_breaks_after_training
        ):
            raise RuntimeError(
                "Numerical callable recompiled or broke during committed updates",
            )
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        conservative_headroom = total - peak_reserved - non_torch_baseline
        attempts = warmup + measured
        committed_steps = sum(not step["amp_step_skipped"] for step in attempts)
        measured_committed_steps = sum(
            not step["amp_step_skipped"] for step in measured
        )
        validate_optimizer_state(torch, model, optimizer, committed_steps)
        row = {
            "status": "ok",
            "branch": name,
            "device": device,
            "compile_seconds": compile_seconds,
            "numerical_unique_graphs": numerical_graphs_after_training,
            "dynamic_reuse_added_graph": specialization_count == 1,
            "graph_breaks": graph_breaks_after_training,
            "generated_kernel_count": int(
                getattr(torch._inductor.metrics, "generated_kernel_count", -1),
            ),
            "optimizer": update_status,
            "optimizer_state_entries": len(optimizer.state),
            "optimizer_step": committed_steps,
            "attempted_steps": len(attempts),
            "amp_step_skipped_count": len(attempts) - committed_steps,
            "measured_committed_steps": measured_committed_steps,
            "grad_scaler": {
                "initial_scale": GRAD_SCALER_INIT_SCALE,
                "growth_interval": GRAD_SCALER_GROWTH_INTERVAL,
                "final_scale": float(scaler.get_scale()),
                "policy": "skip_update_and_continue",
            },
            "warmup": warmup,
            "timing": summarize_steps(measured),
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "non_torch_baseline_bytes": non_torch_baseline,
            "conservative_reserved_headroom_bytes": conservative_headroom,
            "headroom_pass": conservative_headroom >= MIN_RESERVED_HEADROOM,
            "losses_finite": all(math.isfinite(step["loss"]) for step in attempts),
        }
        del numerical, scaler, optimizer, model
        torch.cuda.empty_cache()
        return row


def run_probe(torch, root, input_contract, execution_contract, result):
    """Execute correctness and both independent complete T4 branches."""
    if torch.cuda.device_count() != 2 or any(
        "T4" not in torch.cuda.get_device_name(index) for index in range(2)
    ):
        raise RuntimeError("Spec 0034 requires exactly two Tesla T4 GPUs")
    configure_torch(torch)
    result["runtime"] = {
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "devices": [
            {
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
                "total_bytes": torch.cuda.get_device_properties(index).total_memory,
            }
            for index in range(2)
        ],
        "mode_options": torch._inductor.list_mode_options(),
        "allocator_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "compiled_backward_autocast": str(
            torch._functorch.config.backward_pass_autocast,
        ),
    }
    torch.manual_seed(execution_contract["initialization_seed"])
    prototype = LocalGlobalMILClassifier()
    initialize_nonzero_attention(torch, prototype)
    base_state = {
        name: value.detach().clone() for name, value in prototype.state_dict().items()
    }
    result["phase"] = "small_full_model_correctness"
    result["correctness"] = full_model_correctness(torch, base_state)
    if not result["correctness"]["correct"]:
        raise RuntimeError(
            "Compiled training exceeded the measured AMP effect envelope",
        )
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    result["phase"] = "load_complete_bags"
    bags, instances = load_inputs(torch, root, input_contract, result)
    cpu_graph = build_local_attention_graph(
        instances,
        expected_instance_count=PATCH_COUNT,
    )
    cpu_graph.verify_integrity()
    if cpu_graph.node_count != PATCH_COUNT:
        raise RuntimeError("Real graph node count differs")
    graphs = [cpu_graph.to(f"cuda:{index}") for index in range(2)]
    small_cpu_graph = build_local_attention_graph(
        make_instances(DYNAMIC_REUSE_COUNT),
        expected_instance_count=DYNAMIC_REUSE_COUNT,
    )
    small_graphs = [small_cpu_graph.to(f"cuda:{index}") for index in range(2)]
    result["graph"] = {
        "identity_sha256": cpu_graph.identity_sha256,
        "nodes": cpu_graph.node_count,
        "edges": int(cpu_graph.neighbor_valid.sum().item()),
        "minimum_degree": int(cpu_graph.neighbor_valid.sum(dim=1).min().item()),
        "maximum_degree": int(cpu_graph.neighbor_valid.sum(dim=1).max().item()),
    }
    rows = []
    for index, name in enumerate(("normal_vae", "so2_vae")):
        result["phase"] = f"full_branch_{name}"
        try:
            row = run_branch(
                torch,
                name,
                index,
                bags[index],
                graphs[index],
                small_graphs[index],
                base_state,
            )
        except Exception as error:
            row = {
                "status": "failed",
                "branch": name,
                "device": index,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            with torch.cuda.device(index):
                torch.cuda.empty_cache()
        rows.append(row)
        print(json.dumps(row), flush=True)
    result["branches"] = rows
    result["accepted_capacity"] = all(
        row["status"] == "ok"
        and row["headroom_pass"]
        and row["numerical_unique_graphs"] in {1, 2}
        and row["graph_breaks"] == 0
        and row["optimizer"]["implementation"]
        == "torch_amp_grad_scaler_native_fused_adamw"
        and row["optimizer"]["fallback"] is None
        and row["measured_committed_steps"] == MEASURED_STEPS
        and row["losses_finite"]
        for row in rows
    )
    result["status"] = "complete" if result["accepted_capacity"] else "rejected"


def install_pinned_torch():
    """Install the exact latest stack already authenticated by version 5."""
    subprocess.check_call([  # noqa: S603 -- every argument is a fixed constant.
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        f"torch=={PINNED_TORCH_VERSION}",
        "--index-url",
        PINNED_TORCH_INDEX,
    ])


def validate_pinned_torch(torch):
    """Reject any imported runtime outside the exact wheel contract."""
    installed_torch = str(torch.__version__).split("+", maxsplit=1)[0]
    if (
        installed_torch != PINNED_TORCH_VERSION
        or torch.version.cuda != PINNED_TORCH_CUDA
    ):
        raise RuntimeError(
            "Installed PyTorch runtime differs from the pinned contract: "
            f"torch={torch.__version__}, cuda={torch.version.cuda}",
        )


def main():
    """Run once and preserve evidence after any compiler/allocation failure."""
    result = {
        "status": "failed",
        "spec": "0034",
        "scope": "capacity_optimization_only_not_learning_or_evaluation",
        "phase": "resolve_package",
        "wsi_id": WSI_ID,
        "patch_count": PATCH_COUNT,
        "input_dataset": f"{INPUT_DATASET_REFERENCE}/1",
        "kernel_sources": [f"{source}/1" for source in KERNEL_SOURCES],
        "contract_sha256": CONTRACT_SHA256,
        "started_unix": time.time(),
    }
    try:
        contract, model, candidate = resolve_package()
        root, input_contract = resolve_input_bundle()
        result["phase"] = "install_pinned_torch"
        install_pinned_torch()
        import torch

        validate_pinned_torch(torch)

        sys.dont_write_bytecode = True
        sys.path.insert(0, str(root / "src"))
        activate_sources(model, candidate)
        result["phase"] = "probe"
        run_probe(torch, root, input_contract, contract, result)
    except BaseException:
        result["traceback"] = traceback.format_exc()
        print(result["traceback"], file=sys.stderr, flush=True)
    finally:
        result["elapsed_seconds"] = time.time() - result["started_unix"]
        result["finished_unix"] = time.time()
        OUTPUT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result), flush=True)
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
