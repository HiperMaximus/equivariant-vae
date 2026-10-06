# Copyright 2026 HiperMaximus
"""Small spatial CLS encoder and gated ABMIL on the frozen VAE maps."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

TOKEN_WIDTH = 64


class ResidualMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(TOKEN_WIDTH)
        self.layers = nn.Sequential(nn.Linear(TOKEN_WIDTH, 128), nn.GELU(),
                                    nn.Linear(128, TOKEN_WIDTH))

    def forward(self, tokens: Tensor) -> Tensor:
        return tokens + self.layers(self.norm(tokens))


class CLSAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(TOKEN_WIDTH)
        self.query = nn.Linear(TOKEN_WIDTH, TOKEN_WIDTH)
        self.kv = nn.Linear(TOKEN_WIDTH, 2 * TOKEN_WIDTH)
        self.output = nn.Linear(TOKEN_WIDTH, TOKEN_WIDTH)

    def projections(self, tokens: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        normalized = self.norm(tokens)
        q = self.query(normalized[:, :1])
        k, v = self.kv(normalized).chunk(2, dim=-1)
        return tuple(value.reshape(value.shape[0], value.shape[1], 2, 32).transpose(1, 2)
                     for value in (q, k, v))

    def forward(self, tokens: Tensor) -> Tensor:
        q, k, v = self.projections(tokens)
        attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        attended = attended.transpose(1, 2).reshape(tokens.shape[0], 1, TOKEN_WIDTH)
        return tokens[:, :1] + self.output(attended)


class SpatialPatchEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(16, TOKEN_WIDTH, kernel_size=4, stride=4)
        self.position = nn.Parameter(torch.empty(1, 64, TOKEN_WIDTH))
        self.cls = nn.Parameter(torch.empty(1, 1, TOKEN_WIDTH))
        nn.init.normal_(self.position, std=0.02)
        nn.init.normal_(self.cls, std=0.02)
        self.token_mlp = ResidualMLP()
        self.cls_attention = CLSAttention()
        self.cls_mlp = ResidualMLP()
        self.final_norm = nn.LayerNorm(TOKEN_WIDTH)

    def tokens(self, maps: Tensor) -> Tensor:
        projected = self.projection(maps).flatten(2).transpose(1, 2)
        # Shared position/CLS parameters and residual paths stay FP32. AMP still
        # runs convolution, linear projections and native SDPA in FP16.
        tokens = self.token_mlp(projected.float() + self.position)
        return torch.cat((self.cls.expand(maps.shape[0], -1, -1), tokens), dim=1)

    def forward(self, maps: Tensor) -> Tensor:
        cls = self.cls_attention(self.tokens(maps))
        return self.final_norm(self.cls_mlp(cls))[:, 0]


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
        # The encoder and both hidden attention projections still use AMP.
        with torch.autocast(device_type=patches.device.type, enabled=False):
            scores = self.score(gated_features.float()).squeeze(-1)
            weights = self.softmax(scores)
            embedding = self.pool((weights[:, None] * patches.float()).sum(dim=0))
        return embedding, (features, gates, gated_features, scores, weights)


class GatedABMILClassifier(nn.Module):
    """One spatial CLS attention, one gated WSI pool and five-class head."""

    capture_names = (
        "patch_tokens", "attention_tanh", "attention_gate",
        "attention_features", "attention_scores", "attention_weights",
        "bag_embedding", "logits",
    )

    def __init__(self, hidden_width: int = 128) -> None:
        super().__init__()
        self.patch_encoder = SpatialPatchEncoder()
        self.attention = GatedAttention(hidden_width)
        self.classifier = nn.Linear(TOKEN_WIDTH, 5)
        self.apply(self._initialize)
        nn.init.xavier_uniform_(self.patch_encoder.projection.weight.flatten(1))
        nn.init.zeros_(self.attention.score.weight)
        nn.init.zeros_(self.classifier.weight)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
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
