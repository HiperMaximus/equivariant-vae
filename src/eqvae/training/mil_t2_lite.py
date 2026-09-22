# Copyright 2026 HiperMaximus
"""Direct, inference-only T2 diagnostics for the accepted local-global MIL."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import Tensor
from torch.nn import functional

from eqvae.models.local_global_mil import (
    ATTENTION_HEADS,
    GLOBAL_TOKENS,
    HEAD_WIDTH,
    LOCAL_MAX_DEGREE,
    RADIAL_CODEBOOK,
    TOKEN_WIDTH,
    LocalAttentionGraph,
    LocalGlobalMILClassifier,
    LocalTransformerBlock,
)
from eqvae.training.mil_dynamics import (
    attention_distribution_summary,
    compact_tensor_summary,
    representation_spectrum_summary,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from eqvae.training.mil_telemetry_io import TelemetryScalar


@dataclass(frozen=True)
class T2LiteResult:
    """Compact sentinel result; no full local attention map is retained."""

    logits: tuple[float, ...]
    records: dict[str, dict[str, float]]
    global_patch_top_indices: tuple[int, ...]
    global_patch_top_scores: tuple[float, ...]
    global_patch_top_lattice_coordinates: tuple[tuple[int, int], ...]


def run_t2_lite(
    model: LocalGlobalMILClassifier,
    latents: Tensor,
    graph: LocalAttentionGraph,
    *,
    top_k_patches: int = 100,
    local_chunk_size: int = 2_048,
) -> T2LiteResult:
    """Inspect one sentinel on a disposable eager model without training state."""
    if top_k_patches < 1 or local_chunk_size < 1:
        raise ValueError("T2-lite budgets must be positive")
    diagnostic = copy.deepcopy(model).eval()
    records: dict[str, dict[str, float]] = {}
    with torch.inference_mode():
        patches = cast("Tensor", diagnostic.patch_encoder(latents))
        _record_representation(records, "patches.x0", patches)
        for index, block_value in enumerate(diagnostic.local_blocks):
            block = cast("LocalTransformerBlock", block_value)
            normalized = cast("Tensor", block.attention_norm(patches))
            records[f"local_{index}.attention"] = local_attention_summary(
                block,
                normalized,
                graph,
                chunk_size=local_chunk_size,
            )
            attention_update = cast("Tensor", block.attention(normalized, graph))
            attended = patches + attention_update
            ffn_input = cast("Tensor", block.ffn_norm(attended))
            records[f"local_{index}.swiglu"] = swiglu_summary(
                block.ffn,
                ffn_input,
            )
            ffn_update = cast("Tensor", block.ffn(ffn_input))
            output = attended + ffn_update
            records[f"local_{index}.residuals"] = _residual_record(
                input_tensor=patches,
                attention_update=attention_update,
                attended=attended,
                ffn_update=ffn_update,
                output=output,
            )
            patches = output
            _record_representation(records, f"patches.x{index + 1}", patches)

        normalized_queries = cast(
            "Tensor", diagnostic.global_summary.query_norm(diagnostic.global_tokens)
        )
        normalized_patches = cast(
            "Tensor", diagnostic.global_summary.patch_norm(patches)
        )
        global_attention, patch_gate_score = global_sigmoid_attention_summary(
            diagnostic,
            normalized_queries,
            normalized_patches,
        )
        records["global_summary.attention"] = global_attention
        global_attention_update = cast(
            "Tensor",
            diagnostic.global_summary.attention(normalized_queries, normalized_patches),
        )
        global_attended = (
            diagnostic.global_tokens.float() + global_attention_update.float()
        )
        global_ffn_input = cast(
            "Tensor", diagnostic.global_summary.ffn_norm(global_attended)
        )
        records["global_summary.swiglu"] = swiglu_summary(
            diagnostic.global_summary.ffn,
            global_ffn_input,
        )
        global_ffn_update = cast(
            "Tensor", diagnostic.global_summary.ffn(global_ffn_input)
        )
        global_tokens = global_attended + global_ffn_update
        records["global_summary.residuals"] = _residual_record(
            input_tensor=diagnostic.global_tokens.float(),
            attention_update=global_attention_update.float(),
            attended=global_attended,
            ffn_update=global_ffn_update,
            output=global_tokens,
        )
        _record_representation(records, "global_tokens.t1", global_tokens)

        records["cls.attention"] = cls_attention_summary(diagnostic, global_tokens)
        cls_normalized = cast(
            "Tensor", diagnostic.cls_block.attention_norm(global_tokens.float())
        )
        cls_query = cast(
            "Tensor", diagnostic.cls_block.query(cls_normalized[:1])
        ).reshape(1, ATTENTION_HEADS, HEAD_WIDTH)
        cls_key, cls_value = (
            part.reshape(GLOBAL_TOKENS, ATTENTION_HEADS, HEAD_WIDTH)
            for part in diagnostic.cls_block.kv.split(
                cast("Tensor", diagnostic.cls_block.kv(cls_normalized))
            )
        )
        cls_context = functional.scaled_dot_product_attention(
            cls_query.transpose(0, 1),
            cls_key.transpose(0, 1),
            cls_value.transpose(0, 1),
            dropout_p=0.0,
            is_causal=False,
            scale=1.0 / math.sqrt(HEAD_WIDTH),
        ).transpose(0, 1)
        cls_projected = cast(
            "Tensor",
            diagnostic.cls_block.output(cls_context.reshape(1, TOKEN_WIDTH)),
        )[0]
        cls_attended = global_tokens.float()[0] + cls_projected
        cls_ffn_input = cast("Tensor", diagnostic.cls_block.ffn_norm(cls_attended))
        records["cls.swiglu"] = swiglu_summary(
            diagnostic.cls_block.ffn,
            cls_ffn_input,
        )
        cls_ffn_update = cast("Tensor", diagnostic.cls_block.ffn(cls_ffn_input))
        cls = cls_attended + cls_ffn_update
        records["cls.residuals"] = _residual_record(
            input_tensor=global_tokens.float()[0],
            attention_update=cls_projected,
            attended=cls_attended,
            ffn_update=cls_ffn_update,
            output=cls,
        )
        records["cls.embedding"] = compact_tensor_summary(cls).record()
        normalized = cast("Tensor", diagnostic.final_norm(cls.float()))
        logits = functional.linear(
            normalized,
            diagnostic.classifier.weight.float(),
            diagnostic.classifier.bias.float(),
        )
        records["classifier.logits"] = compact_tensor_summary(logits).record()
        top_k = min(top_k_patches, patch_gate_score.numel())
        top_scores, top_indices = torch.topk(patch_gate_score, k=top_k, sorted=True)
        top_coordinates = graph.lattice_coordinates[top_indices.detach().cpu()]
    return T2LiteResult(
        logits=tuple(float(value) for value in logits.detach().cpu().tolist()),
        records=records,
        global_patch_top_indices=tuple(
            int(value) for value in top_indices.cpu().tolist()
        ),
        global_patch_top_scores=tuple(
            float(value) for value in top_scores.cpu().tolist()
        ),
        global_patch_top_lattice_coordinates=tuple(
            (int(x), int(y))
            for x, y in cast("list[list[int]]", top_coordinates.tolist())
        ),
    )


def local_attention_summary(
    block: LocalTransformerBlock,
    normalized_tokens: Tensor,
    graph: LocalAttentionGraph,
    *,
    chunk_size: int,
) -> dict[str, float]:
    """Reconstruct exact fixed-25+null probabilities in chunks and reduce them."""
    projected = cast("Tensor", block.attention.qkv(normalized_tokens))
    query, key, _value = (
        part.reshape(-1, ATTENTION_HEADS, HEAD_WIDTH)
        for part in block.attention.qkv.split(projected)
    )
    entropy_sum = torch.zeros((), device=query.device, dtype=torch.float64)
    effective_sum = torch.zeros_like(entropy_sum)
    maximum_sum = torch.zeros_like(entropy_sum)
    null_sum = torch.zeros_like(entropy_sum)
    head_count = 0
    radial_mass = torch.zeros(
        (len(RADIAL_CODEBOOK),), device=query.device, dtype=torch.float64
    )
    for start in range(0, query.shape[0], chunk_size):
        stop = min(start + chunk_size, query.shape[0])
        neighbor_index = graph.neighbor_index[start:stop]
        neighbor_valid = graph.neighbor_valid[start:stop]
        radial_code = graph.radial_code[start:stop]
        safe_index = neighbor_index.clamp_min(0)
        gathered_key = key[safe_index]
        score = (query[start:stop, None, :, :].float() * gathered_key.float()).sum(
            dim=-1
        ) * (1.0 / math.sqrt(HEAD_WIDTH))
        safe_code = radial_code.long().clamp_max(len(RADIAL_CODEBOOK) - 1)
        score = score.permute(0, 2, 1) + block.attention.relative_bias[
            :, safe_code
        ].permute(1, 0, 2)
        score = score.masked_fill(~neighbor_valid[:, None, :], -torch.inf)
        sink_score = (
            query[start:stop].float() * block.attention.null_key.float()[None, :, :]
        ).sum(dim=-1) * (1.0 / math.sqrt(HEAD_WIDTH))
        sink_score += block.attention.null_bias.float()[None, :]
        probability = torch.softmax(
            torch.cat((score, sink_score[:, :, None]), dim=-1), dim=-1
        )
        entropy = -(probability * probability.clamp_min(1e-12).log()).sum(dim=-1)
        entropy_sum += entropy.double().sum()
        effective_sum += entropy.exp().double().sum()
        maximum_sum += probability.amax(dim=-1).double().sum()
        null_sum += probability[..., -1].double().sum()
        head_count += entropy.numel()
        real_probability = probability[..., :LOCAL_MAX_DEGREE].permute(0, 2, 1)
        for code in range(len(RADIAL_CODEBOOK)):
            radial_mass[code] += real_probability[safe_code == code].double().sum()
    denominator = max(head_count, 1)
    record = {
        "entropy_mean": float((entropy_sum / denominator).item()),
        "effective_neighbor_count_mean": float((effective_sum / denominator).item()),
        "maximum_mass_mean": float((maximum_sum / denominator).item()),
        "null_mass_mean": float((null_sum / denominator).item()),
    }
    for code, radius_squared in enumerate(RADIAL_CODEBOOK):
        record[f"radial_{radius_squared}_mass_per_query_head"] = float(
            (radial_mass[code] / denominator).item()
        )
    return record


def global_sigmoid_attention_summary(
    model: LocalGlobalMILClassifier,
    normalized_queries: Tensor,
    normalized_patches: Tensor,
) -> tuple[dict[str, float], Tensor]:
    """Reconstruct sigmoid gates and return their mean per-patch gate score."""
    attention = model.global_summary.attention
    query = cast("Tensor", attention.query(normalized_queries)).reshape(
        GLOBAL_TOKENS, ATTENTION_HEADS, HEAD_WIDTH
    )
    key, _value = (
        part.reshape(-1, ATTENTION_HEADS, HEAD_WIDTH)
        for part in attention.kv.split(cast("Tensor", attention.kv(normalized_patches)))
    )
    scores = torch.einsum("mhd,nhd->mhn", query.float(), key.float()) / math.sqrt(
        HEAD_WIDTH
    )
    scores += attention.head_offset.float()[None, :, None]
    scores -= math.log(normalized_patches.shape[0])
    weights = torch.sigmoid(scores)
    record = attention_distribution_summary(weights)
    record["gate_mean"] = float(weights.mean().item())
    record["gate_std"] = float(weights.std(unbiased=False).item())
    record["gate_near_zero_fraction"] = float((weights <= 1e-4).float().mean().item())
    record["gate_near_one_fraction"] = float(
        (weights >= 1 - 1e-4).float().mean().item()
    )
    patch_gate_score = weights.mean(dim=(0, 1))
    return record, patch_gate_score


def cls_attention_summary(
    model: LocalGlobalMILClassifier,
    global_tokens: Tensor,
) -> dict[str, float]:
    """Reconstruct the final CLS-to-17 attention distribution."""
    block = model.cls_block
    normalized = cast("Tensor", block.attention_norm(global_tokens.float()))
    query = cast("Tensor", block.query(normalized[:1])).reshape(
        1, ATTENTION_HEADS, HEAD_WIDTH
    )
    key, _value = (
        part.reshape(GLOBAL_TOKENS, ATTENTION_HEADS, HEAD_WIDTH)
        for part in block.kv.split(cast("Tensor", block.kv(normalized)))
    )
    scores = torch.einsum("qhd,khd->hqk", query.float(), key.float()) / math.sqrt(
        HEAD_WIDTH
    )
    weights = torch.softmax(scores, dim=-1)
    record = attention_distribution_summary(weights)
    record["cls_self_mass_mean"] = float(weights[..., 0].mean().item())
    record["register_mass_mean"] = float(weights[..., 1:].sum(dim=-1).mean().item())
    return record


def swiglu_summary(module: Any, inputs: Tensor) -> dict[str, float]:
    """Reconstruct gate/value/product scales without altering the module output."""
    gate, value = module.input.split(cast("Tensor", module.input(inputs)))
    activated_gate = functional.silu(gate)
    product = activated_gate * value
    output = cast("Tensor", module.output(product))
    record: dict[str, float] = {}
    for name, tensor in (
        ("gate", gate),
        ("activated_gate", activated_gate),
        ("value", value),
        ("product", product),
        ("output", output),
    ):
        record.update(compact_tensor_summary(tensor).record(prefix=f"{name}."))
    record["activated_gate_saturation_low"] = float(
        (activated_gate.abs() <= 1e-4).float().mean().item()
    )
    record["activated_gate_saturation_high"] = float(
        (activated_gate.abs() >= 10.0).float().mean().item()
    )
    return record


def flatten_t2_records(
    result: T2LiteResult,
    *,
    identity: Mapping[str, int | float | bool],
) -> list[dict[str, TelemetryScalar]]:
    """Flatten T2 while retaining self-describing capture and metric identities.

    The long-table schema also stores exact logits and ranked global gate
    scores. ``patch_index`` is the row index in the ordered WSI atlas; a caller
    that needs coordinates joins it to that checkpoint-bound atlas rather than
    treating the gate score as a causal attribution.
    """
    rows: list[dict[str, TelemetryScalar]] = []
    for capture_name, values in sorted(result.records.items()):
        for metric_name, value in sorted(values.items()):
            rows.append(
                {
                    **identity,
                    "record_kind": "summary",
                    "capture_name": capture_name,
                    "metric_name": metric_name,
                    "rank": -1,
                    "patch_index": -1,
                    "lattice_x": -1,
                    "lattice_y": -1,
                    "value": value,
                }
            )
    for class_index, value in enumerate(result.logits):
        rows.append(
            {
                **identity,
                "record_kind": "logit",
                "capture_name": "classifier.logits",
                "metric_name": f"logit_{class_index}",
                "rank": class_index,
                "patch_index": -1,
                "lattice_x": -1,
                "lattice_y": -1,
                "value": value,
            }
        )
    for rank, (patch_index, score, coordinates) in enumerate(
        zip(
            result.global_patch_top_indices,
            result.global_patch_top_scores,
            result.global_patch_top_lattice_coordinates,
            strict=True,
        ),
        start=1,
    ):
        rows.append(
            {
                **identity,
                "record_kind": "global_gate_top_patch",
                "capture_name": "global_summary.attention",
                "metric_name": "mean_sigmoid_gate_score",
                "rank": rank,
                "patch_index": patch_index,
                "lattice_x": coordinates[0],
                "lattice_y": coordinates[1],
                "value": score,
            }
        )
    return rows


def _record_representation(
    records: dict[str, dict[str, float]],
    name: str,
    tensor: Tensor,
) -> None:
    record = compact_tensor_summary(tensor).record()
    record.update(
        {
            f"spectrum.{key}": value
            for key, value in representation_spectrum_summary(tensor).items()
        }
    )
    records[name] = record


def _residual_record(
    *,
    input_tensor: Tensor,
    attention_update: Tensor,
    attended: Tensor,
    ffn_update: Tensor,
    output: Tensor,
) -> dict[str, float]:
    input32 = input_tensor.detach().float()
    attention32 = attention_update.detach().float()
    attended32 = attended.detach().float()
    ffn32 = ffn_update.detach().float()
    output32 = output.detach().float()
    return {
        "attention_update_to_input": float(
            (attention32.norm() / input32.norm().clamp_min(1e-12)).item()
        ),
        "attention_input_cosine": float(
            functional.cosine_similarity(
                input32.reshape(-1), attention32.reshape(-1), dim=0
            ).item()
        ),
        "ffn_update_to_attended": float(
            (ffn32.norm() / attended32.norm().clamp_min(1e-12)).item()
        ),
        "ffn_attended_cosine": float(
            functional.cosine_similarity(
                attended32.reshape(-1), ffn32.reshape(-1), dim=0
            ).item()
        ),
        "output_to_input": float(
            (output32.norm() / input32.norm().clamp_min(1e-12)).item()
        ),
    }


__all__ = [
    "T2LiteResult",
    "cls_attention_summary",
    "flatten_t2_records",
    "global_sigmoid_attention_summary",
    "local_attention_summary",
    "run_t2_lite",
    "swiglu_summary",
]
