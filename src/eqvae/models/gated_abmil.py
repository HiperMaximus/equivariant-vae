# Copyright 2026 HiperMaximus
"""Gated ABMIL on the frozen VAE maps, with the common patch CNN."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from eqvae.models.local_global_mil import LocalGlobalPatchEncoder, TOKEN_WIDTH


class GatedAttention(nn.Module):
    """Ilse et al. gated scores followed by a softmax over the full WSI."""

    def __init__(self, hidden_width: int = 128) -> None:
        super().__init__()
        self.tanh_projection = nn.Linear(TOKEN_WIDTH, hidden_width)
        self.tanh = nn.Tanh()
        self.gate_projection = nn.Linear(TOKEN_WIDTH, hidden_width)
        self.gate = nn.Sigmoid()
        self.gated_features = nn.Identity()
        self.score = nn.Linear(hidden_width, 1)
        self.softmax = nn.Softmax(dim=0)
        self.pool = nn.Identity()

    def forward(self, patches: Tensor) -> tuple[Tensor, tuple[Tensor, ...]]:
        features = self.tanh(self.tanh_projection(patches))
        gates = self.gate(self.gate_projection(patches))
        gated_features = self.gated_features(features * gates)
        # Keep logits, softmax and the full-bag weighted reduction in FP32.
        # The CNN and both hidden attention projections still use AMP.
        with torch.autocast(device_type=patches.device.type, enabled=False):
            scores = self.score(gated_features.float()).squeeze(-1)
            weights = self.softmax(scores)
            embedding = self.pool((weights[:, None] * patches.float()).sum(dim=0))
        return embedding, (features, gates, gated_features, scores, weights)


class GatedABMILClassifier(nn.Module):
    """Three-convolution patch encoder, one gated pool, five-class linear head."""

    capture_names = (
        "patch_tokens", "attention_tanh", "attention_gate",
        "attention_features", "attention_scores", "attention_weights",
        "bag_embedding", "logits",
    )

    def __init__(self, hidden_width: int = 128) -> None:
        super().__init__()
        self.patch_encoder = LocalGlobalPatchEncoder()
        self.attention = GatedAttention(hidden_width)
        self.classifier = nn.Linear(TOKEN_WIDTH, 5)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.GroupNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward_with_representations(self, latents: Tensor) -> tuple[Tensor, tuple[Tensor, ...]]:
        patches = self.patch_encoder(latents)
        embedding, attention = self.attention(patches)
        with torch.autocast(device_type=latents.device.type, enabled=False):
            logits = self.classifier(embedding.float())
        return logits, (patches, *attention, embedding, logits)

    def forward(self, latents: Tensor) -> Tensor:
        return self.forward_with_representations(latents)[0]
