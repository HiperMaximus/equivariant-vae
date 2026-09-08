# Copyright 2026 HiperMaximus
# ruff: noqa: ANN001, ANN201, ANN202, BLE001, C901, COM812, DOC201, DOC501, E303, E402, E501, EM101, EM102, F811, F821, INP001, PERF401, PLC0415, PLC2801, PLR0912, PLR0913, PLR0914, PLR0915, PLR0916, PLR0917, PLR2004, PLW0717, RUF100, S102, S404, SLF001, T201, TRY003
"""Run the exact Spec 0026 model on both complete WSI45630 latent bags."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import subprocess
import sys
import textwrap
import time
import traceback
from collections import Counter
from operator import itemgetter
from pathlib import Path

# fmt: off

# BEGIN EMBEDDED SPEC0026 MODEL
if False:
    # Copyright 2026 HiperMaximus
    # ruff: noqa: C901, COM812, DOC201, DOC501, EM101, PLC2801, PLR0912, PLR0913, PLR0914, PLR0915, PLR0917, PLR2004, SLF001, TRY003
    """The fixed width-192 local-global MIL architecture from Spec 0026."""


    import hashlib
    import math
    import struct
    from dataclasses import dataclass
    from typing import TYPE_CHECKING, Final, Self, TypedDict, cast

    import numpy as np
    import torch
    from torch import Tensor, nn
    from torch.nn import functional

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
    LOCAL_CHUNK_SIZE: Final = 8192
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
        local_attention_backend: str = "explicit_sparse_fp32"
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
        ) -> Self:
            graph = object.__new__(cls)
            object.__setattr__(graph, "wsi_id", wsi_id)
            object.__setattr__(graph, "expected_instance_count", expected_instance_count)
            object.__setattr__(graph, "lattice_coordinates", lattice_coordinates)
            object.__setattr__(graph, "neighbor_index", neighbor_index)
            object.__setattr__(graph, "neighbor_valid", neighbor_valid)
            object.__setattr__(graph, "radial_code", radial_code)
            object.__setattr__(graph, "identity_sha256", identity_sha256)
            graph.verify_integrity()
            return graph

        @property
        def node_count(self) -> int:
            """Number of ordered graph nodes."""
            return int(self.neighbor_index.shape[0])

        def to(self, device: torch.device | str) -> Self:
            """Return the same graph identity with arrays copied to one device."""
            self.verify_integrity()
            return type(self)._create(
                wsi_id=self.wsi_id,
                expected_instance_count=self.expected_instance_count,
                lattice_coordinates=self.lattice_coordinates,
                neighbor_index=self.neighbor_index.to(device=device),
                neighbor_valid=self.neighbor_valid.to(device=device),
                radial_code=self.radial_code.to(device=device),
                identity_sha256=self.identity_sha256,
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
            """Build two input branches and one output projection."""
            super().__init__()
            self.gate = nn.Linear(TOKEN_WIDTH, FFN_WIDTH)
            self.value = nn.Linear(TOKEN_WIDTH, FFN_WIDTH)
            self.output = nn.Linear(FFN_WIDTH, TOKEN_WIDTH)

        def forward(self, tokens: Tensor) -> Tensor:
            """Apply SiLU to the gate branch before elementwise modulation."""
            gate = cast("Tensor", self.gate(tokens))
            value = cast("Tensor", self.value(tokens))
            return cast(
                "Tensor",
                self.output(functional.silu(gate) * value),
            )


    class SparseLocalSoftmaxAttention(nn.Module):
        """Explicit sparse FP32 local attention with one static zero-value null."""

        def __init__(self) -> None:
            """Build six projection heads and their layer-owned null parameters."""
            super().__init__()
            self.query = nn.Linear(TOKEN_WIDTH, TOKEN_WIDTH, bias=False)
            self.key = nn.Linear(TOKEN_WIDTH, TOKEN_WIDTH, bias=False)
            self.value = nn.Linear(TOKEN_WIDTH, TOKEN_WIDTH, bias=False)
            self.output = nn.Linear(TOKEN_WIDTH, TOKEN_WIDTH)
            self.relative_bias = nn.Parameter(torch.zeros(ATTENTION_HEADS, 6))
            self.null_key = nn.Parameter(torch.zeros(ATTENTION_HEADS, HEAD_WIDTH))
            self.null_bias = nn.Parameter(torch.zeros(ATTENTION_HEADS))

        def forward(self, tokens: Tensor, graph: LocalAttentionGraph) -> Tensor:
            """Attend to at most 25 graph neighbours plus one noncommunicating null."""
            _validate_tokens_and_graph(tokens, graph)
            query = cast("Tensor", self.query(tokens)).reshape(
                -1, ATTENTION_HEADS, HEAD_WIDTH
            )
            key = cast("Tensor", self.key(tokens)).reshape(-1, ATTENTION_HEADS, HEAD_WIDTH)
            value = cast("Tensor", self.value(tokens)).reshape(
                -1, ATTENTION_HEADS, HEAD_WIDTH
            )
            chunks: list[Tensor] = []
            for start in range(0, tokens.shape[0], LOCAL_CHUNK_SIZE):
                stop = min(start + LOCAL_CHUNK_SIZE, tokens.shape[0])
                chunks.append(
                    self._attention_chunk(
                        query[start:stop],
                        key,
                        value,
                        graph.neighbor_index[start:stop],
                        graph.neighbor_valid[start:stop],
                        graph.radial_code[start:stop],
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
        ) -> Tensor:
            safe_index = neighbor_index.clamp_min(0)
            gathered_key = key[safe_index].permute(0, 2, 1, 3)
            gathered_value = value[safe_index].permute(0, 2, 1, 3)
            projected_dtype = query.dtype
            with torch.autocast(device_type=query.device.type, enabled=False):
                query32 = query.float()
                key32 = gathered_key.float()
                value32 = gathered_value.float()
                scores = torch.einsum("chd,chkd->chk", query32, key32) / math.sqrt(
                    HEAD_WIDTH,
                )
                safe_code = radial_code.long().clamp_max(len(RADIAL_CODEBOOK) - 1)
                bias = self.relative_bias.float()[:, safe_code].permute(1, 0, 2)
                scores = (scores + bias).masked_fill(
                    ~neighbor_valid[:, None, :],
                    -torch.inf,
                )
                null_scores = (
                    torch.einsum("chd,hd->ch", query32, self.null_key.float())
                    / math.sqrt(HEAD_WIDTH)
                    + self.null_bias.float()[None, :]
                )
                weights = torch.softmax(
                    torch.cat((scores, null_scores[..., None]), dim=2),
                    dim=2,
                )
                context32 = torch.einsum("chk,chkd->chd", weights[:, :, :-1], value32)
            return context32.to(dtype=projected_dtype)


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
            self.key = nn.Linear(TOKEN_WIDTH, TOKEN_WIDTH, bias=False)
            self.value = nn.Linear(TOKEN_WIDTH, TOKEN_WIDTH, bias=False)
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
            key = cast("Tensor", self.key(patches)).reshape(-1, ATTENTION_HEADS, HEAD_WIDTH)
            value = cast("Tensor", self.value(patches)).reshape(
                -1, ATTENTION_HEADS, HEAD_WIDTH
            )
            projected_dtype = query.dtype
            with torch.autocast(device_type=query.device.type, enabled=False):
                scores = torch.einsum(
                    "mhd,nhd->mhn",
                    query.float(),
                    key.float(),
                ) / math.sqrt(HEAD_WIDTH)
                scores += self.head_offset.float()[None, :, None]
                scores -= math.log(patches.shape[0])
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
            self.key = nn.Linear(TOKEN_WIDTH, TOKEN_WIDTH, bias=False)
            self.value = nn.Linear(TOKEN_WIDTH, TOKEN_WIDTH, bias=False)
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
            key = cast("Tensor", self.key(normalized)).reshape(
                GLOBAL_TOKENS, ATTENTION_HEADS, HEAD_WIDTH
            )
            value = cast("Tensor", self.value(normalized)).reshape(
                GLOBAL_TOKENS,
                ATTENTION_HEADS,
                HEAD_WIDTH,
            )
            attended = functional.scaled_dot_product_attention(
                query.transpose(0, 1),
                key.transpose(0, 1),
                value.transpose(0, 1),
                dropout_p=0.0,
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
            graph.verify_integrity()
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
        if graph.neighbor_index.device != tokens.device:
            raise ValueError("Graph arrays and patch tokens must be on the same device")


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
        "SigmoidPatchSummaryAttention",
        "SparseLocalSoftmaxAttention",
        "SwiGLU",
        "build_local_attention_graph",
        "local_global_mil_adamw_parameter_groups",
        "make_paired_local_global_mil_models",
    ]
# END EMBEDDED SPEC0026 MODEL

KAGGLE_LOCAL_GLOBAL_CAPACITY_READY = True
INPUT_ROOT = Path("/kaggle/input")
OUTPUT_PATH = Path("/kaggle/working/spec0030_local_global_mil_capacity.json")
CAPACITY_CONTRACT_JSON = r"""{"authorization":"spec0030_local_global_capacity_shared_access_retry_authorized","diagnosis_index":1,"execution":{"checkpointing":false,"direct_complete_bag_only":true,"fallback":null,"model_devices":{"normal_vae":0,"so2_vae":1},"paired_step_atomicity":"both_finite_before_either_step"},"initialization_seed":1701,"input_dataset":{"contract_sha256":"99bb4d2f60558aee9691b67be4867ffae434bc306581a000fd5d72a6befac660","pointer_sha256":"08e461846bf16efebac707c82962762f49837916986b29aee0dcd6ca1fc31c6c","reference":"maximusshtefan/eqvae-wsi45630-capacity-inputs","version":1},"kernel_sources":[{"reference":"maximusshtefan/eqvae-ubc-ocean-latent-run-04","version":1},{"reference":"maximusshtefan/eqvae-ubc-ocean-cancer-latent-top-up","version":1},{"reference":"maximusshtefan/eqvae-wsi45630-completion","version":1}],"model":{"parameter_count":1513055,"sha256":"60e815cdd1136bc742dc35944bc7291303c5688f4b39696c065b0432e5ec09f3","source":"src/eqvae/models/local_global_mil.py"},"optimizer":{"learning_rate":0.0002,"matrix_weight_decay":0.0001,"name":"AdamW","semantic_no_decay":true},"output":"spec0030_local_global_mil_capacity.json","patch_count":32595,"precision":{"autocast":"float16","classifier_and_loss":"float32","grad_scaler_growth_interval":1000000,"grad_scaler_initial_scale":32768},"schema_version":"spec0030.local_global_mil_capacity.v1","scope":"capacity_only_not_learning_or_evaluation","spec_sha256":"aa1a1ff05548d30bc49d94bbe45d438e62b08f1ad9555dc960ed23af6f67cf99","steps":["warmup","measured"],"wsi_id":45630}"""
CAPACITY_CONTRACT_SHA256 = "326e9add9549aa98577effc5807e02dc9312327cf221183a4e6e06693be87a9b"
MODEL_SHA256 = "60e815cdd1136bc742dc35944bc7291303c5688f4b39696c065b0432e5ec09f3"
EMBEDDED_MODEL_SHA256 = "86802786a55f25038e04def92296cadc364a75aeae18c427fff878cf2440a94c"
INPUT_CONTRACT_NAME = "wsi45630_capacity_input.json"
INPUT_CONTRACT_SHA256 = (
    "99bb4d2f60558aee9691b67be4867ffae434bc306581a000fd5d72a6befac660"
)
INPUT_DATASET_REFERENCE = "maximusshtefan/eqvae-wsi45630-capacity-inputs"
KERNEL_SOURCES = (
    "maximusshtefan/eqvae-ubc-ocean-latent-run-04",
    "maximusshtefan/eqvae-ubc-ocean-cancer-latent-top-up",
    "maximusshtefan/eqvae-wsi45630-completion",
)
WSI_ID = 45_630
PATCH_COUNT = 32_595
DIAGNOSIS_INDEX = 1
EXPECTED_PARAMETER_COUNT = 1_513_055


def sha256(path):
    """Stream a file hash so multi-gigabyte latent payloads are never copied."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_package():
    """Authenticate the embedded execution contract and canonical model digest."""
    if (
        hashlib.sha256(CAPACITY_CONTRACT_JSON.encode()).hexdigest()
        != CAPACITY_CONTRACT_SHA256
    ):
        raise RuntimeError("Spec 0030 embedded contract bytes differ")
    run_source = Path(__file__).read_text(encoding="utf-8")
    begin = "# BEGIN EMBEDDED SPEC0026 MODEL\n"
    end = "# END EMBEDDED SPEC0026 MODEL\n"
    if run_source.count(begin) != 1 or run_source.count(end) != 1:
        raise RuntimeError("Spec 0030 embedded model markers differ")
    embedded_block = run_source.split(begin, 1)[1].split(end, 1)[0]
    prefix = "if False:\n"
    if not embedded_block.startswith(prefix):
        raise RuntimeError("Spec 0030 embedded model guard differs")
    embedded_model = textwrap.dedent(embedded_block.removeprefix(prefix))
    if hashlib.sha256(embedded_model.encode()).hexdigest() != EMBEDDED_MODEL_SHA256:
        raise RuntimeError("Spec 0030 embedded model source differs")
    contract = json.loads(CAPACITY_CONTRACT_JSON)
    expected_sources = [
        {"reference": source, "version": 1} for source in KERNEL_SOURCES
    ]
    if (
        contract.get("schema_version") != "spec0030.local_global_mil_capacity.v1"
        or contract.get("authorization")
        != "spec0030_local_global_capacity_shared_access_retry_authorized"
        or contract.get("scope") != "capacity_only_not_learning_or_evaluation"
        or contract.get("model", {}).get("sha256") != MODEL_SHA256
        or contract.get("model", {}).get("parameter_count") != EXPECTED_PARAMETER_COUNT
        or contract.get("input_dataset", {}).get("reference") != INPUT_DATASET_REFERENCE
        or contract.get("input_dataset", {}).get("version") != 1
        or contract.get("input_dataset", {}).get("contract_sha256")
        != INPUT_CONTRACT_SHA256
        or contract.get("kernel_sources") != expected_sources
        or contract.get("wsi_id") != WSI_ID
        or contract.get("patch_count") != PATCH_COUNT
        or contract.get("diagnosis_index") != DIAGNOSIS_INDEX
        or contract.get("execution", {}).get("direct_complete_bag_only") is not True
        or contract.get("execution", {}).get("fallback") is not None
    ):
        raise RuntimeError("Spec 0030 execution contract differs")
    return contract


def activate_embedded_model():
    """Define the canonical model only after the latest torch bootstrap."""
    run_source = Path(__file__).read_text(encoding="utf-8")
    begin = "# BEGIN EMBEDDED SPEC0026 MODEL\n"
    end = "# END EMBEDDED SPEC0026 MODEL\n"
    embedded_block = run_source.split(begin, 1)[1].split(end, 1)[0]
    embedded_model = textwrap.dedent(embedded_block.removeprefix("if False:\n"))
    executable_model = "from __future__ import annotations\n" + embedded_model
    exec(compile(executable_model, "embedded_local_global_mil.py", "exec"), globals())


def resolve_input_bundle():
    """Authenticate the existing immutable version-1 pointer/source dataset."""
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
    """Resolve same-named producer files by their authenticated sidecar hashes."""
    roots = {}
    with catalog.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            candidates = [
                path.parent
                for path in INPUT_ROOT.rglob(row["sidecar_name"])
                if path.is_file()
                and path.stat().st_size == int(row["sidecar_bytes"])
                and sha256(path) == row["sidecar_sha256"]
            ]
            if len(candidates) != 1:
                raise RuntimeError("Expected one hash-matched producer sidecar")
            source = row["kaggle_source"]
            if source in roots and roots[source] != candidates[0]:
                raise RuntimeError("Paired producer binaries have different roots")
            roots[source] = candidates[0]
    return roots


def load_bags_and_instances(root, input_contract, result):
    """Load both aligned complete bags and construct their shared graph instances."""
    import torch

    from eqvae.data.supervised_latents import (
        LogicalPointer,
        SupervisedLatentStore,
        WSIInstance,
    )

    catalog = root / "probe/physical_parts.csv"
    source_roots = resolve_sources(catalog)
    if set(source_roots) != set(input_contract["kernel_sources"]):
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
    result["input_reads"] = {}
    bags = []
    shared_reads = None
    started = time.monotonic()
    for device, model_name in enumerate(("normal_vae", "so2_vae")):
        result["phase"] = f"load_{model_name}"
        with SupervisedLatentStore(
            catalog_path=catalog,
            model_name=model_name,
            source_roots=source_roots,
        ) as store:
            latents, reads = store.read_rows(pointers)
        counts = Counter(str(read.part) for read in reads)
        if (
            tuple(latents.shape) != (PATCH_COUNT, 16, 32, 32)
            or dict(counts) != input_contract["part_counts"]
            or not torch.isfinite(latents).all().item()
            or (shared_reads is not None and shared_reads != reads)
        ):
            raise RuntimeError("Full paired bag identity, alignment, or values differ")
        shared_reads = reads
        bags.append(latents.to(f"cuda:{device}"))
        result["input_reads"][model_name] = {
            "rows": PATCH_COUNT,
            "part_counts": dict(counts),
            "identity_sha256": hashlib.sha256(
                json.dumps(identity).encode()
            ).hexdigest(),
            "input_finite": True,
        }
        del latents
    result["load_seconds"] = time.monotonic() - started
    return bags, instances


def model_state_sha256(model):
    """Hash the complete initialized state before either branch reaches a GPU."""
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(list(array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def register_stage_hooks(model, device, target, active_stage, branch):
    """Record the active stage and memory at every model stage boundary."""
    import torch

    handles = []

    def mark_active(name):
        def hook(_module, _inputs):
            active_stage[branch] = name

        return hook

    def capture(name):
        def hook(_module, _inputs, _output):
            target[name] = {
                "allocated_bytes": torch.cuda.memory_allocated(device),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "reserved_bytes": torch.cuda.memory_reserved(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            }

        return hook

    stages = {
        "patch_encoder": model.patch_encoder,
        "local_block_1": model.local_blocks[0],
        "local_block_2": model.local_blocks[1],
        "global_summary": model.global_summary,
        "cls_block": model.cls_block,
        "final_norm": model.final_norm,
    }
    for name, module in stages.items():
        handles.extend((
            module.register_forward_pre_hook(mark_active(name)),
            module.register_forward_hook(capture(name)),
        ))
    return handles


def gradients_are_finite(model):
    """Return complete gradient coverage and the first invalid parameter name."""
    import torch

    checked = 0
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all().item():
            return False, checked, name
        checked += 1
    return True, checked, None


def materialize_optimizer_state(model, optimizer):
    """Allocate AdamW state without changing parameters or step counters."""
    import torch

    groups = [(group["lr"], group["weight_decay"]) for group in optimizer.param_groups]
    try:
        for group in optimizer.param_groups:
            group["lr"] = 0.0
            group["weight_decay"] = 0.0
        for parameter in model.parameters():
            parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        for state in optimizer.state.values():
            state["step"].zero_()
    finally:
        optimizer.zero_grad(set_to_none=True)
        for group, (learning_rate, weight_decay) in zip(
            optimizer.param_groups, groups, strict=True
        ):
            group["lr"] = learning_rate
            group["weight_decay"] = weight_decay


def commit_paired_optimizer_steps(models, optimizers, scalers):
    """Commit both branches or restore both models and optimizer states."""
    model_states = [copy.deepcopy(model.state_dict()) for model in models]
    optimizer_states = [
        copy.deepcopy(optimizer.state_dict()) for optimizer in optimizers
    ]
    try:
        for scaler, optimizer in zip(scalers, optimizers, strict=True):
            scaler.step(optimizer)
    except BaseException:
        for model, state in zip(models, model_states, strict=True):
            model.load_state_dict(state)
        for optimizer, state in zip(optimizers, optimizer_states, strict=True):
            optimizer.load_state_dict(state)
        raise


def run_probe(root, input_contract, execution_contract, result):
    """Execute two atomic paired full-bag optimizer steps on two T4 GPUs."""
    import torch
    from torch.nn import functional

    result["runtime"] = {
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "devices": [
            {
                "name": torch.cuda.get_device_name(index),
                "total_bytes": torch.cuda.get_device_properties(index).total_memory,
                "capability": list(torch.cuda.get_device_capability(index)),
            }
            for index in range(torch.cuda.device_count())
        ],
    }
    if torch.cuda.device_count() != 2 or any(
        "T4" not in device["name"] for device in result["runtime"]["devices"]
    ):
        raise RuntimeError("Exactly two T4 GPUs are required")
    if LOCAL_CHUNK_SIZE != 8192:
        raise RuntimeError("Local attention chunk differs from Spec 0028 selection")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.manual_seed(execution_contract["initialization_seed"])
    models = list(make_paired_local_global_mil_models())
    parameter_count = sum(parameter.numel() for parameter in models[0].parameters())
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("Spec 0026 parameter count differs")
    states = [model_state_sha256(model) for model in models]
    independent = all(
        left.data_ptr() != right.data_ptr() and torch.equal(left, right)
        for left, right in zip(
            models[0].parameters(), models[1].parameters(), strict=True
        )
    )
    if len(set(states)) != 1 or not independent:
        raise RuntimeError("Paired model initialization differs or shares storage")
    result["parameter_count"] = parameter_count
    result["initialization_sha256"] = states[0]
    result["identical_independent_initialization"] = True
    models = [model.to(f"cuda:{index}") for index, model in enumerate(models)]
    bags, instances = load_bags_and_instances(root, input_contract, result)
    result["phase"] = "build_graph"
    cpu_graph = build_local_attention_graph(
        instances, expected_instance_count=PATCH_COUNT
    )
    graphs = [cpu_graph.to(f"cuda:{index}") for index in range(2)]
    if any(graph.identity_sha256 != cpu_graph.identity_sha256 for graph in graphs):
        raise RuntimeError("Paired graph identity differs")
    result["graph"] = {
        "identity_sha256": cpu_graph.identity_sha256,
        "nodes": cpu_graph.node_count,
        "max_neighbours": int(cpu_graph.neighbor_valid.sum(dim=1).max().item()),
        "min_neighbours": int(cpu_graph.neighbor_valid.sum(dim=1).min().item()),
    }
    optimizers = [
        torch.optim.AdamW(
            local_global_mil_adamw_parameter_groups(model, weight_decay=1e-4),
            lr=2e-4,
        )
        for model in models
    ]
    for model, optimizer in zip(models, optimizers, strict=True):
        materialize_optimizer_state(model, optimizer)
    scalers = [
        torch.amp.GradScaler("cuda", init_scale=32768, growth_interval=1_000_000)
        for _ in models
    ]
    targets = [
        torch.tensor([DIAGNOSIS_INDEX], device=f"cuda:{index}") for index in range(2)
    ]
    parameter_tensor_counts = [sum(1 for _ in model.parameters()) for model in models]
    result["steps"] = []
    for phase in ("warmup", "measured"):
        result["phase"] = phase
        stage_memory = [{}, {}]
        result["active_stage"] = {}
        result["live_stage_memory"] = stage_memory
        handles = [
            register_stage_hooks(
                model,
                index,
                stage_memory[index],
                result["active_stage"],
                ("normal_vae", "so2_vae")[index],
            )
            for index, model in enumerate(models)
        ]
        for index, optimizer in enumerate(optimizers):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(index)
            torch.cuda.reset_peak_memory_stats(index)
        started = time.perf_counter()
        losses = []
        for index in range(2):
            with torch.cuda.device(index), torch.autocast("cuda", dtype=torch.float16):
                logits = models[index](bags[index], graphs[index])
                loss = functional.cross_entropy(
                    logits.float().unsqueeze(0), targets[index]
                )
            scalers[index].scale(loss).backward()
            scalers[index].unscale_(optimizers[index])
            losses.append(loss.detach())
        finite_checks = [gradients_are_finite(model) for model in models]
        losses_finite = [bool(torch.isfinite(loss).item()) for loss in losses]
        if not all(losses_finite) or not all(check[0] for check in finite_checks):
            invalid = [check[2] for check in finite_checks]
            raise FloatingPointError(
                f"Nonfinite paired loss/gradient; neither optimizer stepped: {invalid}",
            )
        scales_before = [scaler.get_scale() for scaler in scalers]
        commit_paired_optimizer_steps(models, optimizers, scalers)
        for scaler in scalers:
            scaler.update()
        for index in range(2):
            torch.cuda.synchronize(index)
        scales_after = [scaler.get_scale() for scaler in scalers]
        optimizer_entries = [len(optimizer.state) for optimizer in optimizers]
        if scales_before != [32768.0, 32768.0] or scales_after != [32768.0, 32768.0]:
            raise FloatingPointError("GradScaler backoff or unexpected growth occurred")
        if optimizer_entries != parameter_tensor_counts:
            raise RuntimeError("An optimizer step omitted parameter state")
        step = {
            "phase": phase,
            "paired_wall_seconds": time.perf_counter() - started,
            "losses": [float(loss.item()) for loss in losses],
            "losses_finite": losses_finite,
            "gradient_parameter_counts": [check[1] for check in finite_checks],
            "all_gradients_finite": [check[0] for check in finite_checks],
            "scales_before": scales_before,
            "scales_after": scales_after,
            "optimizer_state_entries": optimizer_entries,
            "optimizer_steps_committed": [True, True],
            "stage_memory": stage_memory,
            "peak_allocated_bytes": [
                torch.cuda.max_memory_allocated(index) for index in range(2)
            ],
            "peak_reserved_bytes": [
                torch.cuda.max_memory_reserved(index) for index in range(2)
            ],
        }
        result["steps"].append(step)
        for device_handles in handles:
            for handle in device_handles:
                handle.remove()
        print(json.dumps(step), flush=True)
    result["status"] = "fits_exact_local_global_mil_full_bag"


def main():
    """Persist compact capacity evidence even after allocation/numerical failure."""
    result = {
        "status": "failed",
        "scope": "capacity_only_not_learning_or_evaluation",
        "wsi_id": WSI_ID,
        "patch_count": PATCH_COUNT,
        "model": "spec0026_local_global_mil",
        "parameter_count_expected": EXPECTED_PARAMETER_COUNT,
        "direct_complete_bag_only": True,
        "fallback": None,
        "input_dataset": f"{INPUT_DATASET_REFERENCE}/1",
        "kernel_sources": [f"{source}/1" for source in KERNEL_SOURCES],
        "capacity_contract_sha256": CAPACITY_CONTRACT_SHA256,
        "model_sha256": MODEL_SHA256,
        "model_devices": {"normal_vae": 0, "so2_vae": 1},
    }
    started = time.monotonic()
    try:
        _ensure_latest_torch()
        execution_contract = resolve_package()
        activate_embedded_model()
        root, input_contract = resolve_input_bundle()
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(root / "src"))
        run_probe(root, input_contract, execution_contract, result)
    except Exception as error:
        result["error_type"] = type(error).__name__
        result["error"] = str(error)
        traceback.print_exc()
    finally:
        result["session_seconds"] = time.monotonic() - started
        if "torch" in sys.modules:
            import torch

            result["final_peak_allocated_bytes"] = [
                torch.cuda.max_memory_allocated(index)
                for index in range(torch.cuda.device_count())
            ]
            result["final_peak_reserved_bytes"] = [
                torch.cuda.max_memory_reserved(index)
                for index in range(torch.cuda.device_count())
            ]
        OUTPUT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result), flush=True)
    return 0 if result["status"] == "fits_exact_local_global_mil_full_bag" else 1


def _ensure_latest_torch():
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "torch",
        "torchvision",
        "torchaudio",
    ])


if __name__ == "__main__":
    raise SystemExit(main())
