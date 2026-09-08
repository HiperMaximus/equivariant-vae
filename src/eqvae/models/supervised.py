# Copyright 2026 HiperMaximus
# ruff: noqa: DOC201, DOC501, EM101, PLR2004, TRY003
"""The two small classifier shapes locked by Spec 0023."""

from __future__ import annotations

from typing import Final, cast

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import (
    checkpoint,  # pyright: ignore[reportUnknownVariableType]
)

LATENT_CHANNELS: Final = 16
PATCH_VECTOR_DIM: Final = 128
ATTENTION_DIM: Final = 64
GROUPS: Final = 8


class SpatialPatchEncoder(nn.Module):
    """Compress one frozen spatial latent map to a learned 128-vector."""

    def __init__(self) -> None:
        """Build the single Spec 0023 spatial compressor."""
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(LATENT_CHANNELS, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(GROUPS, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(GROUPS, 64),
            nn.GELU(),
            nn.Conv2d(64, PATCH_VECTOR_DIM, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(GROUPS, PATCH_VECTOR_DIM),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
        )
        self.apply(_initialize_supervised_module)

    def forward(self, latents: Tensor) -> Tensor:
        """Encode a batch of `[N,16,32,32]` frozen latent maps."""
        if latents.ndim != 4 or tuple(latents.shape[1:]) != (16, 32, 32):
            raise ValueError("Patch encoder requires [N,16,32,32] latents")
        return cast("Tensor", self.layers(latents))


class AttentionMILClassifier(nn.Module):
    """Five-class complete-bag gated-attention classifier."""

    def __init__(self, *, attention_dim: int = ATTENTION_DIM) -> None:
        """Build one spatial encoder, global attention, and WSI head."""
        super().__init__()
        if attention_dim < 1:
            raise ValueError("attention_dim must be positive")
        self.patch_encoder = SpatialPatchEncoder()
        self.attention_v = nn.Linear(PATCH_VECTOR_DIM, attention_dim)
        self.attention_u = nn.Linear(PATCH_VECTOR_DIM, attention_dim)
        self.attention_w = nn.Linear(attention_dim, 1)
        self.head = nn.Linear(PATCH_VECTOR_DIM, 5)
        _initialize_supervised_module(self.attention_v)
        _initialize_supervised_module(self.attention_u)
        _initialize_supervised_module(self.attention_w)
        _initialize_supervised_module(self.head)

    def forward(
        self,
        latents: Tensor,
        *,
        checkpoint_chunk_size: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return one WSI logit vector and complete-bag attention weights."""
        if latents.shape[0] < 1:
            raise ValueError("An attention bag may not be empty")
        tokens = self._encode(latents, checkpoint_chunk_size)
        attention_v = cast("Tensor", self.attention_v(tokens))
        attention_u = cast("Tensor", self.attention_u(tokens))
        gated = torch.tanh(attention_v) * torch.sigmoid(attention_u)
        scores = cast("Tensor", self.attention_w(gated)).squeeze(1)
        attention = torch.softmax(scores, dim=0)
        wsi_vector = torch.sum(attention[:, None] * tokens, dim=0)
        return cast("Tensor", self.head(wsi_vector)), attention

    def _encode(self, latents: Tensor, chunk_size: int | None) -> Tensor:
        if chunk_size is None:
            return cast("Tensor", self.patch_encoder(latents))
        if chunk_size < 1:
            raise ValueError("checkpoint_chunk_size must be positive")
        chunks = torch.split(latents, chunk_size, dim=0)
        if not chunks:
            raise ValueError("An attention bag may not be empty")
        return torch.cat(
            tuple(
                cast(
                    "Tensor",
                    checkpoint(
                        self.patch_encoder,
                        chunk,
                        use_reentrant=False,
                    ),
                )
                for chunk in chunks
            ),
            dim=0,
        )


class ClassSpecificAttentionMILClassifier(nn.Module):
    """Five-class complete-bag gated attention with one map per diagnosis."""

    def __init__(self) -> None:
        """Build the fixed width-128 class-specific MIL diagnostic."""
        super().__init__()
        self.patch_encoder = SpatialPatchEncoder()
        self.attention_v = nn.Linear(PATCH_VECTOR_DIM, PATCH_VECTOR_DIM)
        self.attention_u = nn.Linear(PATCH_VECTOR_DIM, PATCH_VECTOR_DIM)
        self.attention_w = nn.Linear(PATCH_VECTOR_DIM, 5)
        self.head = nn.Linear(PATCH_VECTOR_DIM, 5)
        _initialize_supervised_module(self.attention_v)
        _initialize_supervised_module(self.attention_u)
        _initialize_supervised_module(self.attention_w)
        _initialize_supervised_module(self.head)

    def forward(
        self,
        latents: Tensor,
        *,
        checkpoint_chunk_size: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return five diagonal WSI logits and full-bag class attention maps."""
        if latents.shape[0] < 1:
            raise ValueError("An attention bag may not be empty")
        if checkpoint_chunk_size is not None:
            raise ValueError("Class-specific MIL must process complete bags unchunked")
        tokens = cast("Tensor", self.patch_encoder(latents))
        attention_v = cast("Tensor", self.attention_v(tokens))
        attention_u = cast("Tensor", self.attention_u(tokens))
        gated = torch.tanh(attention_v) * torch.sigmoid(attention_u)
        scores = cast("Tensor", self.attention_w(gated))
        attention = torch.softmax(scores, dim=0)
        pooled = attention.transpose(0, 1) @ tokens
        head_scores = cast("Tensor", self.head(pooled))
        return torch.diagonal(head_scores), attention


class TissueClassifier(nn.Module):
    """Three-class high-purity tissue patch classifier."""

    def __init__(self) -> None:
        """Build one spatial encoder and the fixed tissue head."""
        super().__init__()
        self.patch_encoder = SpatialPatchEncoder()
        self.head = nn.Linear(PATCH_VECTOR_DIM, 3)
        _initialize_supervised_module(self.head)

    def forward(self, latents: Tensor) -> Tensor:
        """Return one three-class logit vector per patch."""
        return cast("Tensor", self.head(self.patch_encoder(latents)))


def make_attention_mil_width_variant(*, attention_dim: int) -> AttentionMILClassifier:
    """Create the one width variant while preserving baseline common parameters."""
    baseline = AttentionMILClassifier()
    if attention_dim == ATTENTION_DIM:
        return baseline
    candidate = AttentionMILClassifier(attention_dim=attention_dim)
    candidate.patch_encoder.load_state_dict(baseline.patch_encoder.state_dict())
    candidate.head.load_state_dict(baseline.head.state_dict())
    return candidate


def make_class_specific_attention_mil_variant() -> ClassSpecificAttentionMILClassifier:
    """Lift the width-128 MIL state into five initially identical score maps."""
    width128 = make_attention_mil_width_variant(attention_dim=PATCH_VECTOR_DIM)
    candidate = ClassSpecificAttentionMILClassifier()
    candidate.patch_encoder.load_state_dict(width128.patch_encoder.state_dict())
    candidate.attention_v.load_state_dict(width128.attention_v.state_dict())
    candidate.attention_u.load_state_dict(width128.attention_u.state_dict())
    candidate.head.load_state_dict(width128.head.state_dict())
    candidate.attention_w.load_state_dict({
        "weight": width128.attention_w.weight.detach().repeat(5, 1),
        "bias": width128.attention_w.bias.detach().repeat(5),
    })
    return candidate


def _initialize_supervised_module(module: nn.Module) -> None:
    """Apply the single paired-classifier initialization policy."""
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(
            module.weight,
            mode="fan_out",
            nonlinearity="relu",
        )
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=1.0)
    elif isinstance(module, nn.GroupNorm):
        nn.init.ones_(module.weight)
    if (
        isinstance(module, (nn.Conv2d, nn.Linear, nn.GroupNorm))
        and module.bias is not None
    ):
        nn.init.zeros_(module.bias)


__all__ = [
    "AttentionMILClassifier",
    "ClassSpecificAttentionMILClassifier",
    "SpatialPatchEncoder",
    "TissueClassifier",
    "make_attention_mil_width_variant",
    "make_class_specific_attention_mil_variant",
]
