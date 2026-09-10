# Copyright 2026 HiperMaximus
# ruff: noqa: ANN001, ANN201, ANN202, ANN204, BLE001, C901, DOC201, DOC501, E501, EM101, INP001, PLC0415, PLR0913, PLR0914, PLR0915, PLR0917, S404, TRY003
"""Probe exact coordinate-masked FlexAttention on the real WSI45630 graph."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import struct
import subprocess
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from operator import itemgetter
from pathlib import Path

KAGGLE_FLEX_ATTENTION_PROBE_READY = True
INPUT_ROOT = Path("/kaggle/input")
OUTPUT_PATH = Path("/kaggle/working/spec0031_flex_attention_probe.json")
INPUT_CONTRACT_NAME = "wsi45630_capacity_input.json"
INPUT_CONTRACT_SHA256 = (
    "99bb4d2f60558aee969b67be4867ffae434bc306581a000fd5d72a6befac660"
)
POINTER_SHA256 = "08e461846bf16efebac707c82962762f49837916986b29aee0dcd6ca1fc31c6c"
DATASET_REFERENCE = "maximusshtefan/eqvae-wsi45630-capacity-inputs"
WSI_ID = 45_630
PATCH_COUNT = 32_595
TOKEN_WIDTH = 192
HEADS = 6
HEAD_WIDTH = 32
MAX_NEIGHBOURS = 25
PATCH_SIZE_PIXELS = 256
LOCAL_RADIUS = 2
RADIAL_PADDING_CODE = 255
RADIAL_VALUES = (0, 1, 2, 4, 5, 8)
RADIAL_LOOKUP = (0, 1, 2, 0, 3, 4, 0, 0, 5)
BLOCK_SIZES = (16, 32)
METADATA_BYTE_LIMIT = 32 * 1024 * 1024
CORRECTNESS_RTOL = 5e-3
CORRECTNESS_ATOL = 5e-3
MAX_CORRECTNESS_RELATIVE_L2 = 2e-3
MIN_CORRECTNESS_COSINE = 0.999
MIN_REFERENCE_GRADIENT_NORM = 1e-6
WARMUP_STEPS = 2
MEASURED_STEPS = 5
MAX_FLEX_PEAK_BYTES = 1_181_116_006  # 1.10 GiB
MIN_MEMORY_REDUCTION = 0.35
MAX_TIME_MULTIPLIER = 2.0


@dataclass(frozen=True)
class SparseBlockMetadata:
    """Compact forward and transpose block lists for one exact graph."""

    block_size: int
    query_length: int
    kv_length: int
    kv_num_blocks: object
    kv_indices: object
    q_num_blocks: object
    q_indices: object
    total_bytes: int


def sha256(path):
    """Hash an input before it influences graph or timing evidence."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_coordinates():
    """Authenticate and read the existing complete WSI45630 coordinate list."""
    matches = list(INPUT_ROOT.rglob(INPUT_CONTRACT_NAME))
    if len(matches) != 1 or sha256(matches[0]) != INPUT_CONTRACT_SHA256:
        raise RuntimeError("Expected the exact WSI45630 capacity contract")
    root = matches[0].parent
    contract = json.loads(matches[0].read_text(encoding="utf-8"))
    if (
        contract.get("dataset_reference") != DATASET_REFERENCE
        or contract.get("wsi_id") != WSI_ID
        or contract.get("patch_count") != PATCH_COUNT
    ):
        raise RuntimeError("Mounted coordinate contract scope differs")
    pointer = root / "probe/pointers.csv"
    record = contract.get("files", {}).get("probe/pointers.csv", {})
    if (
        not pointer.is_file()
        or sha256(pointer) != POINTER_SHA256
        or record.get("sha256") != POINTER_SHA256
        or pointer.stat().st_size != record.get("bytes")
    ):
        raise RuntimeError("Mounted WSI45630 pointer bytes differ")
    with pointer.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    coordinates = tuple((int(row["x"]), int(row["y"])) for row in rows)
    if (
        len(coordinates) != PATCH_COUNT
        or any(int(row["wsi_id"]) != WSI_ID for row in rows)
        or len(set(coordinates)) != PATCH_COUNT
        or any(x % PATCH_SIZE_PIXELS or y % PATCH_SIZE_PIXELS for x, y in coordinates)
        or coordinates != tuple(sorted(coordinates, key=itemgetter(1, 0)))
    ):
        raise RuntimeError("Expected unique WSI45630 coordinates in numeric y,x order")
    return coordinates


def build_graph(coordinates):
    """Build physical radius-two edges without consulting sequence adjacency."""
    lattice = tuple(
        (x // PATCH_SIZE_PIXELS, y // PATCH_SIZE_PIXELS) for x, y in coordinates
    )
    lookup = {point: index for index, point in enumerate(lattice)}
    radius_to_code = {value: index for index, value in enumerate(RADIAL_VALUES)}
    indices = []
    valid = []
    radial = []
    degrees = []
    for gx, gy in lattice:
        neighbours = []
        for dx in range(-LOCAL_RADIUS, LOCAL_RADIUS + 1):
            for dy in range(-LOCAL_RADIUS, LOCAL_RADIUS + 1):
                index = lookup.get((gx + dx, gy + dy))
                if index is not None:
                    neighbours.append((dx, dy, index))
        neighbours.sort(key=itemgetter(0, 1))
        degrees.append(len(neighbours))
        padding = MAX_NEIGHBOURS - len(neighbours)
        indices.append([item[2] for item in neighbours] + [-1] * padding)
        valid.append([True] * len(neighbours) + [False] * padding)
        radial.append(
            [radius_to_code[item[0] ** 2 + item[1] ** 2] for item in neighbours]
            + [RADIAL_PADDING_CODE] * padding,
        )
    if min(degrees) < 1 or max(degrees) > MAX_NEIGHBOURS:
        raise RuntimeError("Local graph degree is outside the locked boundary")
    return indices, valid, radial, degrees


def graph_identity_sha256(coordinates, indices, valid, radial):
    """Reproduce the canonical Spec 0026 graph digest without repo imports."""
    digest = hashlib.sha256()

    def start_field(tag, length):
        encoded = tag.encode("utf-8")
        digest.update(struct.pack("<H", len(encoded)))
        digest.update(encoded)
        digest.update(struct.pack("<Q", length))

    fixed_fields = (
        ("schema", b"eqvae_spec0026_graph_v1"),
        ("wsi_id", struct.pack("<q", WSI_ID)),
        ("shape", struct.pack("<QQ", len(coordinates), MAX_NEIGHBOURS)),
        (
            "dtypes",
            b"coordinates:<i8;neighbor_index:<i8;neighbor_valid:u1;radial_code:u1",
        ),
        ("sentinels", struct.pack("<qB", -1, RADIAL_PADDING_CODE)),
        ("codebook", struct.pack("<6Q", *RADIAL_VALUES)),
    )
    for tag, payload in fixed_fields:
        start_field(tag, len(payload))
        digest.update(payload)
    start_field("coordinates", len(coordinates) * 2 * 8)
    for x, y in coordinates:
        digest.update(
            struct.pack("<qq", x // PATCH_SIZE_PIXELS, y // PATCH_SIZE_PIXELS),
        )
    start_field("neighbor_index", len(indices) * MAX_NEIGHBOURS * 8)
    for row in indices:
        digest.update(struct.pack(f"<{MAX_NEIGHBOURS}q", *row))
    start_field("neighbor_valid", len(valid) * MAX_NEIGHBOURS)
    for row in valid:
        digest.update(bytes(row))
    start_field("radial_code", len(radial) * MAX_NEIGHBOURS)
    for row in radial:
        digest.update(bytes(row))
    return digest.hexdigest()


def edge_allowed(coordinates, query_index, key_index):
    """Define one edge solely from physical lattice displacement or null identity."""
    if key_index == len(coordinates):
        return True
    if not 0 <= query_index < len(coordinates) or not 0 <= key_index < len(coordinates):
        return False
    qx, qy = coordinates[query_index]
    kx, ky = coordinates[key_index]
    pixel_radius = PATCH_SIZE_PIXELS * LOCAL_RADIUS
    return abs(qx - kx) <= pixel_radius and abs(qy - ky) <= pixel_radius


def build_sparse_block_metadata(torch, indices, valid, *, block_size):
    """Create bounded forward and transpose block lists directly from sparse edges."""
    if block_size not in BLOCK_SIZES:
        raise ValueError("block_size must be one of the precommitted candidates")
    query_length = len(indices)
    kv_length = query_length + 1
    query_block_count = math.ceil(query_length / block_size)
    kv_block_count = math.ceil(kv_length / block_size)
    null_block = query_length // block_size
    forward = []
    for query_block in range(query_block_count):
        blocks = {null_block}
        start = query_block * block_size
        stop = min(start + block_size, query_length)
        for query_index in range(start, stop):
            blocks.update(
                key_index // block_size
                for key_index, keep in zip(
                    indices[query_index],
                    valid[query_index],
                    strict=True,
                )
                if keep
            )
        forward.append(sorted(blocks))
    reverse = [set() for _ in range(kv_block_count)]
    for query_block, key_blocks in enumerate(forward):
        for key_block in key_blocks:
            reverse[key_block].add(query_block)
    reverse = [sorted(blocks) for blocks in reverse]
    max_forward = max(map(len, forward))
    max_reverse = max(map(len, reverse))
    kv_num_blocks = torch.tensor(
        [[[len(blocks) for blocks in forward]]],
        dtype=torch.int32,
    )
    kv_indices = torch.zeros((1, 1, query_block_count, max_forward), dtype=torch.int32)
    for row, blocks in enumerate(forward):
        kv_indices[0, 0, row, : len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
    q_num_blocks = torch.tensor(
        [[[len(blocks) for blocks in reverse]]],
        dtype=torch.int32,
    )
    q_indices = torch.zeros((1, 1, kv_block_count, max_reverse), dtype=torch.int32)
    for row, blocks in enumerate(reverse):
        if blocks:
            q_indices[0, 0, row, : len(blocks)] = torch.tensor(
                blocks,
                dtype=torch.int32,
            )
    tensors = (kv_num_blocks, kv_indices, q_num_blocks, q_indices)
    total_bytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
    if total_bytes > METADATA_BYTE_LIMIT:
        raise RuntimeError("Sparse block metadata exceeds the fixed 32 MiB budget")
    return SparseBlockMetadata(
        block_size=block_size,
        query_length=query_length,
        kv_length=kv_length,
        kv_num_blocks=kv_num_blocks,
        kv_indices=kv_indices,
        q_num_blocks=q_num_blocks,
        q_indices=q_indices,
        total_bytes=total_bytes,
    )


def block_metadata_summary(metadata):
    """Expose bounded metadata sizes without serializing tensor contents."""
    names = ("kv_num_blocks", "kv_indices", "q_num_blocks", "q_indices")
    result = {
        "block_size": metadata.block_size,
        "query_length": metadata.query_length,
        "kv_length": metadata.kv_length,
        "total_bytes": metadata.total_bytes,
    }
    for name in names:
        tensor = getattr(metadata, name)
        result[name] = {
            "shape": list(tensor.shape),
            "bytes": tensor.numel() * tensor.element_size(),
            "maximum": int(tensor.max().item()),
        }
    return result


def make_block_mask(torch, metadata, coordinates, device):
    """Assemble a BlockMask whose partial tiles recheck exact physical edges."""
    from torch.nn.attention.flex_attention import BlockMask

    lattice = torch.tensor(
        [(x // PATCH_SIZE_PIXELS, y // PATCH_SIZE_PIXELS) for x, y in coordinates],
        dtype=torch.int32,
        device=device,
    )
    qx = lattice[:, 0].contiguous()
    qy = lattice[:, 1].contiguous()
    kx = qx.clone()
    ky = qy.clone()
    count = len(coordinates)

    def mask_mod(_b, _h, q_idx, kv_idx):
        safe_k = torch.where(kv_idx < count, kv_idx, 0)
        local = (torch.abs(qx[q_idx] - kx[safe_k]) <= LOCAL_RADIUS) & (
            torch.abs(qy[q_idx] - ky[safe_k]) <= LOCAL_RADIUS
        )
        return (kv_idx == count) | ((kv_idx < count) & local)

    return BlockMask(
        seq_lengths=(metadata.query_length, metadata.kv_length),
        kv_num_blocks=metadata.kv_num_blocks.to(device),
        kv_indices=metadata.kv_indices.to(device),
        full_kv_num_blocks=None,
        full_kv_indices=None,
        q_num_blocks=metadata.q_num_blocks.to(device),
        q_indices=metadata.q_indices.to(device),
        full_q_num_blocks=None,
        full_q_indices=None,
        BLOCK_SIZE=(metadata.block_size, metadata.block_size),
        mask_mod=mask_mod,
    ), (qx, qy, kx, ky)


def make_module(torch):
    """Create one projection-identical module with explicit and Flex paths."""
    nn = torch.nn

    class LocalAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(TOKEN_WIDTH, TOKEN_WIDTH, bias=False)
            self.k_proj = nn.Linear(TOKEN_WIDTH, TOKEN_WIDTH, bias=False)
            self.v_proj = nn.Linear(TOKEN_WIDTH, TOKEN_WIDTH, bias=False)
            self.out_proj = nn.Linear(TOKEN_WIDTH, TOKEN_WIDTH, bias=True)
            self.relative_bias = nn.Parameter(torch.zeros(HEADS, len(RADIAL_VALUES)))
            self.null_key = nn.Parameter(torch.zeros(HEADS, HEAD_WIDTH))
            self.null_bias = nn.Parameter(torch.zeros(HEADS))
            for projection in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
                nn.init.xavier_uniform_(projection.weight)
            nn.init.zeros_(self.out_proj.bias)

        def project(self, x):
            with torch.autocast(
                device_type=x.device.type,
                dtype=torch.float16,
                enabled=x.is_cuda,
            ):
                q = self.q_proj(x).reshape(-1, HEADS, HEAD_WIDTH)
                k = self.k_proj(x).reshape(-1, HEADS, HEAD_WIDTH)
                v = self.v_proj(x).reshape(-1, HEADS, HEAD_WIDTH)
            return q, k, v

        def explicit(self, x, indices, valid, radial, *, chunk_size=8192):
            q, k, v = self.project(x)
            outputs = []
            for start in range(0, x.shape[0], chunk_size):
                stop = min(start + chunk_size, x.shape[0])
                safe = indices[start:stop].clamp_min(0)
                keys = k[safe].permute(0, 2, 1, 3)
                values = v[safe].permute(0, 2, 1, 3)
                with torch.autocast(device_type=x.device.type, enabled=False):
                    scores = torch.einsum(
                        "chd,chkd->chk",
                        q[start:stop].float(),
                        keys.float(),
                    ) / math.sqrt(HEAD_WIDTH)
                    codes = radial[start:stop].long().clamp_max(len(RADIAL_VALUES) - 1)
                    bias = self.relative_bias.float()[:, codes].permute(1, 0, 2)
                    scores = (scores + bias).masked_fill(
                        ~valid[start:stop, None, :],
                        -torch.inf,
                    )
                    null_score = (
                        torch.einsum(
                            "chd,hd->ch",
                            q[start:stop].float(),
                            self.null_key.float(),
                        )
                        / math.sqrt(HEAD_WIDTH)
                        + self.null_bias.float()[None, :]
                    )
                    weights = torch.softmax(
                        torch.cat((scores, null_score[..., None]), dim=2),
                        dim=2,
                    )
                    context = torch.einsum(
                        "chk,chkd->chd",
                        weights[:, :, :-1],
                        values.float(),
                    )
                outputs.append(context.to(q.dtype).reshape(stop - start, TOKEN_WIDTH))
            with torch.autocast(
                device_type=x.device.type,
                dtype=torch.float16,
                enabled=x.is_cuda,
            ):
                return self.out_proj(torch.cat(outputs, dim=0))

        def flex(self, x, block_mask, coordinate_tensors, compiled_flex, *, block_size):
            q, k, v = self.project(x)
            count = x.shape[0]
            qx, qy, kx, ky = coordinate_tensors
            radial_lookup = torch.tensor(
                RADIAL_LOOKUP,
                dtype=torch.int64,
                device=x.device,
            )

            def score_mod(score, _b, h, q_idx, kv_idx):
                safe_k = torch.where(kv_idx < count, kv_idx, 0)
                dx = qx[q_idx] - kx[safe_k]
                dy = qy[q_idx] - ky[safe_k]
                squared_radius = torch.clamp(dx * dx + dy * dy, 0, 8)
                code = radial_lookup[squared_radius]
                patch_score = score + self.relative_bias[h, code]
                return torch.where(
                    kv_idx == count,
                    score + self.null_bias[h],
                    patch_score,
                )

            null_key = self.null_key.float()[None, :, None, :]
            null_value = torch.zeros_like(null_key)
            with torch.autocast(device_type=x.device.type, enabled=False):
                query = q.float().permute(1, 0, 2)[None, ...]
                key = torch.cat(
                    (k.float().permute(1, 0, 2)[None, ...], null_key),
                    dim=2,
                )
                value = torch.cat(
                    (v.float().permute(1, 0, 2)[None, ...], null_value),
                    dim=2,
                )
                context = compiled_flex(
                    query,
                    key,
                    value,
                    score_mod=score_mod,
                    block_mask=block_mask,
                    kernel_options={
                        "BACKEND": "TRITON",
                        "BLOCK_M": block_size,
                        "BLOCK_N": block_size,
                    },
                )
            projected = (
                context[0].permute(1, 0, 2).reshape(count, TOKEN_WIDTH).to(q.dtype)
            )
            with torch.autocast(
                device_type=x.device.type,
                dtype=torch.float16,
                enabled=x.is_cuda,
            ):
                return self.out_proj(projected)

    return LocalAttention()


def oracle_forward(torch, module, x, indices, valid, radial):
    """Evaluate every sparse query independently for a trusted CPU/GPU oracle."""
    q, k, v = module.project(x)
    contexts = []
    with torch.autocast(device_type=x.device.type, enabled=False):
        for query_index in range(x.shape[0]):
            keep = valid[query_index]
            neighbours = indices[query_index][keep]
            codes = radial[query_index][keep].long()
            query = q[query_index].float()
            keys = k[neighbours].float()
            values = v[neighbours].float()
            patch_scores = torch.einsum("khd,hd->hk", keys, query) / math.sqrt(
                HEAD_WIDTH,
            )
            patch_scores += module.relative_bias.float()[:, codes]
            null_score = (
                torch.einsum("hd,hd->h", query, module.null_key.float())
                / math.sqrt(HEAD_WIDTH)
                + module.null_bias.float()
            )
            weights = torch.softmax(
                torch.cat((patch_scores, null_score[:, None]), dim=1),
                dim=1,
            )
            contexts.append(
                torch
                .einsum("hk,khd->hd", weights[:, :-1], values)
                .reshape(TOKEN_WIDTH)
                .to(q.dtype),
            )
    with torch.autocast(
        device_type=x.device.type,
        dtype=torch.float16,
        enabled=x.is_cuda,
    ):
        return module.out_proj(torch.stack(contexts))


def comparison_metrics(torch, actual, expected):
    """Report elementwise and normwise evidence for one tensor comparison."""
    actual32 = actual.float()
    expected32 = expected.float()
    delta = (actual32 - expected32).abs()
    expected_l2 = float(torch.linalg.vector_norm(expected32).item())
    actual_l2 = float(torch.linalg.vector_norm(actual32).item())
    delta_l2 = float(torch.linalg.vector_norm(delta).item())
    denominator = max(expected_l2 * actual_l2, 1e-12)
    tolerance = CORRECTNESS_ATOL + CORRECTNESS_RTOL * expected32.abs()
    violations = delta > tolerance
    return {
        "allclose": bool(
            torch.allclose(
                actual32,
                expected32,
                rtol=CORRECTNESS_RTOL,
                atol=CORRECTNESS_ATOL,
            ),
        ),
        "finite": bool(
            torch.isfinite(actual32).all().item()
            and torch.isfinite(expected32).all().item(),
        ),
        "violation_count": int(violations.sum().item()),
        "max_abs": float(delta.max().item()),
        "mean_abs": float(delta.mean().item()),
        "expected_l2": expected_l2,
        "relative_l2": delta_l2 / max(expected_l2, 1e-12),
        "cosine_similarity": float(
            (actual32.flatten() * expected32.flatten()).sum().item(),
        )
        / denominator,
    }


def correctness_gate(comparisons):
    """Require strict outputs and precommitted normwise gradient agreement."""
    return all(
        item["finite"]
        and (
            item["allclose"]
            if name.endswith("output")
            else item["relative_l2"] <= MAX_CORRECTNESS_RELATIVE_L2
            and item["cosine_similarity"] >= MIN_CORRECTNESS_COSINE
        )
        for name, item in comparisons.items()
    )


def clone_state(state):
    """Give each candidate independent parameter storage."""
    return {name: tensor.detach().clone() for name, tensor in state.items()}


def compile_counter_snapshot(counters):
    """Convert nonzero Torch compilation counters into stable JSON values."""
    return {
        str(section): {
            str(name): int(value) for name, value in values.items() if int(value) != 0
        }
        for section, values in counters.items()
        if any(int(value) != 0 for value in values.values())
    }


def gradient_snapshot(module, x):
    """Capture complete gradients after one candidate backward pass."""
    if x.grad is None:
        raise RuntimeError("Input gradient is missing")
    gradients = {"input": x.grad.detach().clone()}
    for name, parameter in module.named_parameters():
        if parameter.grad is None:
            message = f"Missing parameter gradient: {name}"
            raise RuntimeError(message)
        gradients[name] = parameter.grad.detach().clone()
    return gradients


def gradient_coverage(gradients):
    """Require every learned spatial/null component to receive signal."""
    names = ("relative_bias", "null_key", "null_bias")
    coverage = {}
    for name in names:
        tensor = gradients[name].float()
        minimum = float(tensor.abs().min().item())
        coverage[name] = {
            "component_count": tensor.numel(),
            "minimum_absolute_gradient": minimum,
            "all_components_nontrivial": minimum > MIN_REFERENCE_GRADIENT_NORM,
        }
    if not all(item["all_components_nontrivial"] for item in coverage.values()):
        message = f"Oracle gradient coverage is incomplete: {coverage}"
        raise RuntimeError(message)
    return coverage


def run_candidate(module, x, forward, upstream):
    """Run one differentiable candidate with a fixed full-output upstream."""
    output = forward(x)
    (output.float() * upstream).sum().backward()
    return output.detach(), gradient_snapshot(module, x)


def compare_candidate(torch, output, gradients, expected_output, expected_gradients):
    """Compare complete output, input gradient and every parameter gradient."""
    records = {"reference:output": comparison_metrics(torch, output, expected_output)}
    records.update({
        f"reference:grad:{name}": comparison_metrics(
            torch,
            tensor,
            expected_gradients[name],
        )
        for name, tensor in gradients.items()
    })
    return records


def evaluate_candidate_set(candidate_values, evaluator):
    """Evaluate every candidate independently and retain correct survivors."""
    records = {}
    survivors = []
    for value in candidate_values:
        try:
            record = evaluator(value)
            correct = record.get("correct") is True
            record["status"] = "supported" if correct else "numerical_mismatch"
        except BaseException as error:
            correct = False
            record = {
                "status": "unsupported",
                "correct": False,
                "error": f"{type(error).__name__}: {error}",
            }
        records[str(value)] = record
        if correct:
            survivors.append(value)
    return records, survivors


def hostile_correctness(torch, compiled_flex, state):
    """Exercise holes, row-wrap traps, every radius and excluded Jacobians."""
    coordinates = (
        (0, 0),
        (2560, 0),
        *tuple(
            (x * 256, y * 256)
            for x in range(-2, 3)
            for y in range(-2, 3)
            if (x, y) != (0, 0)
        ),
    )
    indices_list, valid_list, radial_list, _ = build_graph(coordinates)
    listed_edges = {
        (query, key)
        for query, (row, row_valid) in enumerate(
            zip(indices_list, valid_list, strict=True),
        )
        for key, keep in zip(row, row_valid, strict=True)
        if keep
    }
    predicate_edges = {
        (query, key)
        for query in range(len(coordinates))
        for key in range(len(coordinates))
        if edge_allowed(coordinates, query, key)
    }
    if listed_edges != predicate_edges or edge_allowed(coordinates, 0, 1):
        raise RuntimeError(
            "Hostile graph does not exactly match the coordinate predicate",
        )
    used_codes = {
        code for row in radial_list for code in row if code != RADIAL_PADDING_CODE
    }
    if used_codes != set(range(len(RADIAL_VALUES))):
        message = f"Hostile graph misses radial codes: {used_codes}"
        raise RuntimeError(message)
    indices = torch.tensor(indices_list, dtype=torch.int64, device="cuda")
    valid = torch.tensor(valid_list, dtype=torch.bool, device="cuda")
    radial = torch.tensor(radial_list, dtype=torch.uint8, device="cuda")
    torch.manual_seed(3103)
    base_x = torch.randn(
        (len(coordinates), TOKEN_WIDTH),
        device="cuda",
        dtype=torch.float16,
    )
    upstream = torch.cos(
        torch.arange(base_x.numel(), device="cuda", dtype=torch.float32) * 0.019,
    ).reshape_as(base_x)
    oracle_module = make_module(torch).to("cuda")
    oracle_module.load_state_dict(clone_state(state))
    oracle_x = base_x.detach().clone().requires_grad_()
    oracle_output, oracle_gradients = run_candidate(
        oracle_module,
        oracle_x,
        lambda value: oracle_forward(
            torch,
            oracle_module,
            value,
            indices,
            valid,
            radial,
        ),
        upstream,
    )
    coverage = gradient_coverage(oracle_gradients)
    explicit_module = make_module(torch).to("cuda")
    explicit_module.load_state_dict(clone_state(state))
    explicit_x = base_x.detach().clone().requires_grad_()
    explicit_output, explicit_gradients = run_candidate(
        explicit_module,
        explicit_x,
        lambda value: explicit_module.explicit(value, indices, valid, radial),
        upstream,
    )
    explicit_comparisons = compare_candidate(
        torch,
        explicit_output,
        explicit_gradients,
        oracle_output,
        oracle_gradients,
    )
    result = {
        "nodes": len(coordinates),
        "edges": len(listed_edges),
        "radial_codes": sorted(used_codes),
        "coordinate_far_sequence_adjacent_pair": [0, 1],
        "oracle_gradient_coverage": coverage,
        "explicit": {
            "correct": correctness_gate(explicit_comparisons),
            "comparisons": explicit_comparisons,
        },
        "flex": {},
    }
    if not result["explicit"]["correct"]:
        raise RuntimeError("Explicit attention failed the hostile oracle")

    def evaluate_block_size(block_size):
        construction_started = time.perf_counter()
        metadata = build_sparse_block_metadata(
            torch,
            indices_list,
            valid_list,
            block_size=block_size,
        )
        block_mask, coordinate_tensors = make_block_mask(
            torch,
            metadata,
            coordinates,
            "cuda",
        )
        torch.cuda.synchronize()
        construction_seconds = time.perf_counter() - construction_started
        module = make_module(torch).to("cuda")
        module.load_state_dict(clone_state(state))
        x = base_x.detach().clone().requires_grad_()
        first_execution_started = time.perf_counter()
        output, gradients = run_candidate(
            module,
            x,
            lambda value, module=module, block_mask=block_mask, coordinate_tensors=coordinate_tensors, block_size=block_size: (
                module.flex(
                    value,
                    block_mask,
                    coordinate_tensors,
                    compiled_flex,
                    block_size=block_size,
                )
            ),
            upstream,
        )
        torch.cuda.synchronize()
        first_execution_seconds = time.perf_counter() - first_execution_started
        comparisons = compare_candidate(
            torch,
            output,
            gradients,
            oracle_output,
            oracle_gradients,
        )
        jacobian = {}
        for query_index, excluded_index in ((0, 1), (1, 0)):
            jacobian_module = make_module(torch).to("cuda")
            jacobian_module.load_state_dict(clone_state(state))
            jacobian_x = base_x.detach().clone().requires_grad_()
            jacobian_output = jacobian_module.flex(
                jacobian_x,
                block_mask,
                coordinate_tensors,
                compiled_flex,
                block_size=block_size,
            )
            query_upstream = torch.zeros_like(jacobian_output, dtype=torch.float32)
            query_upstream[query_index] = torch.cos(
                torch.arange(TOKEN_WIDTH, device="cuda", dtype=torch.float32) * 0.023,
            )
            (jacobian_output.float() * query_upstream).sum().backward()
            maximum = float(jacobian_x.grad[excluded_index].abs().max().item())
            jacobian[f"query_{query_index}_to_excluded_{excluded_index}"] = {
                "maximum_absolute_gradient": maximum,
                "exact_zero": math.isclose(maximum, 0.0, rel_tol=0.0, abs_tol=0.0),
            }
        correct = correctness_gate(comparisons) and all(
            item["exact_zero"] for item in jacobian.values()
        )
        return {
            "correct": correct,
            "comparisons": comparisons,
            "excluded_pair_jacobians": jacobian,
            "block_metadata": block_metadata_summary(metadata),
            "construction_seconds": construction_seconds,
            "first_compiled_forward_backward_seconds": first_execution_seconds,
        }

    result["flex"], survivors = evaluate_candidate_set(
        BLOCK_SIZES,
        evaluate_block_size,
    )
    result["surviving_block_sizes"] = survivors
    if not survivors:
        raise RuntimeError("No FlexAttention block size passed hostile correctness")
    return result


def benchmark(torch, state, base_x, forward_factory, *, candidate, block_size=None):
    """Measure settled full-graph forward/backward time and peak memory."""
    module = make_module(torch).to("cuda")
    module.load_state_dict(clone_state(state))
    x = base_x.detach().clone().requires_grad_()
    forward = forward_factory(module)
    try:
        for _ in range(WARMUP_STEPS):
            module.zero_grad(set_to_none=True)
            x.grad = None
            forward(x).float().square().mean().backward()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        timings = []
        for _ in range(MEASURED_STEPS):
            module.zero_grad(set_to_none=True)
            x.grad = None
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = forward(x)
            output.float().square().mean().backward()
            end.record()
            end.synchronize()
            timings.append(float(start.elapsed_time(end)))
        finite = x.grad is not None and bool(torch.isfinite(x.grad).all().item())
        finite = finite and all(
            parameter.grad is not None
            and bool(torch.isfinite(parameter.grad).all().item())
            for parameter in module.parameters()
        )
        return {
            "status": "ok" if finite else "nonfinite",
            "candidate": candidate,
            "block_size": block_size,
            "mean_ms": statistics.fmean(timings),
            "min_ms": min(timings),
            "max_ms": max(timings),
            "timings_ms": timings,
            "finite_gradients": finite,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
    finally:
        del module, x
        torch.cuda.empty_cache()


def qualifying_flex_rows(rows):
    """Nominate only correct rows meeting fixed memory and time bounds."""
    explicit = next(
        row
        for row in rows
        if row.get("candidate") == "explicit" and row.get("status") == "ok"
    )
    result = []
    for row in rows:
        if row.get("candidate") != "flex" or row.get("status") != "ok":
            continue
        memory_ratio = row["peak_allocated_bytes"] / explicit["peak_allocated_bytes"]
        time_ratio = row["mean_ms"] / explicit["mean_ms"]
        row["memory_ratio_vs_explicit"] = memory_ratio
        row["time_ratio_vs_explicit"] = time_ratio
        row["qualifies"] = (
            row["peak_allocated_bytes"] <= MAX_FLEX_PEAK_BYTES
            and memory_ratio <= 1.0 - MIN_MEMORY_REDUCTION
            and time_ratio <= MAX_TIME_MULTIPLIER
        )
        if row["qualifies"]:
            result.append(row)
    return sorted(result, key=itemgetter("mean_ms"))


def run_probe(coordinates, result):
    """Run exact full-graph correctness and timing on one T4."""
    import torch
    import triton
    from torch._dynamo.utils import counters  # noqa: PLC2701
    from torch.nn.attention.flex_attention import flex_attention

    triton_version = getattr(triton, "__version__", None)
    if not isinstance(triton_version, str) or not triton_version:
        raise RuntimeError("Spec 0031 requires an identifiable Triton runtime")
    result["runtime"] = {
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "triton": triton_version,
        "devices": [
            {
                "name": torch.cuda.get_device_name(index),
                "capability": torch.cuda.get_device_capability(index),
                "total_bytes": torch.cuda.get_device_properties(index).total_memory,
            }
            for index in range(torch.cuda.device_count())
        ],
    }
    if (
        torch.cuda.device_count() < 1
        or "T4" not in torch.cuda.get_device_name(0)
        or torch.cuda.get_device_capability(0) != (7, 5)
    ):
        raise RuntimeError("Spec 0031 requires Tesla T4 / SM75 as CUDA device 0")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.manual_seed(1701)
    counters.clear()
    result["phase"] = "build_graph"
    indices_list, valid_list, radial_list, degrees = build_graph(coordinates)
    result["graph"] = {
        "nodes": len(coordinates),
        "edges": sum(degrees),
        "minimum_degree": min(degrees),
        "maximum_degree": max(degrees),
        "degree_histogram": {
            str(key): value for key, value in sorted(Counter(degrees).items())
        },
        "identity_sha256": graph_identity_sha256(
            coordinates,
            indices_list,
            valid_list,
            radial_list,
        ),
    }
    metadata_plans = {}
    construction_seconds = {}
    for block_size in BLOCK_SIZES:
        started = time.perf_counter()
        metadata = build_sparse_block_metadata(
            torch,
            indices_list,
            valid_list,
            block_size=block_size,
        )
        construction_seconds[str(block_size)] = time.perf_counter() - started
        metadata_plans[block_size] = metadata
    result["block_metadata"] = {
        str(size): block_metadata_summary(metadata)
        for size, metadata in metadata_plans.items()
    }
    result["block_metadata_construction_seconds"] = construction_seconds
    indices = torch.tensor(indices_list, dtype=torch.int64, device="cuda")
    valid = torch.tensor(valid_list, dtype=torch.bool, device="cuda")
    radial = torch.tensor(radial_list, dtype=torch.uint8, device="cuda")
    base_x = torch.randn((PATCH_COUNT, TOKEN_WIDTH), device="cuda", dtype=torch.float16)
    prototype = make_module(torch).to("cuda")
    state = clone_state(prototype.state_dict())
    state["relative_bias"] = torch.linspace(
        -0.3,
        0.3,
        HEADS * len(RADIAL_VALUES),
    ).reshape(HEADS, len(RADIAL_VALUES))
    state["null_key"] = torch.linspace(-0.2, 0.2, HEADS * HEAD_WIDTH).reshape(
        HEADS,
        HEAD_WIDTH,
    )
    state["null_bias"] = torch.linspace(-0.15, 0.15, HEADS)
    upstream = torch.cos(
        torch.arange(PATCH_COUNT * TOKEN_WIDTH, device="cuda", dtype=torch.float32)
        * 0.013,
    ).reshape(PATCH_COUNT, TOKEN_WIDTH)
    result["parameter_count"] = sum(p.numel() for p in prototype.parameters())
    compiled_flex = torch.compile(
        flex_attention,
        fullgraph=True,
        dynamic=False,
        mode="max-autotune",
    )

    result["phase"] = "hostile_correctness"
    result["hostile_correctness"] = hostile_correctness(torch, compiled_flex, state)
    hostile_survivors = set(result["hostile_correctness"]["surviving_block_sizes"])

    result["phase"] = "full_graph_correctness"
    explicit_module = make_module(torch).to("cuda")
    explicit_module.load_state_dict(clone_state(state))
    explicit_x = base_x.detach().clone().requires_grad_()
    explicit_output, explicit_gradients = run_candidate(
        explicit_module,
        explicit_x,
        lambda value, module=explicit_module: module.explicit(
            value,
            indices,
            valid,
            radial,
        ),
        upstream,
    )
    full_reference_norms = {
        name: float(torch.linalg.vector_norm(tensor.float()).item())
        for name, tensor in explicit_gradients.items()
    }
    if any(
        not math.isfinite(norm) or norm <= MIN_REFERENCE_GRADIENT_NORM
        for norm in full_reference_norms.values()
    ):
        message = (
            "Full explicit reference produced a trivial gradient: "
            f"{full_reference_norms}"
        )
        raise RuntimeError(message)
    result["full_graph_reference_gradient_norms"] = full_reference_norms
    result["full_graph_reference_gradient_coverage"] = gradient_coverage(
        explicit_gradients,
    )
    result["correctness"] = {}
    for block_size, metadata in metadata_plans.items():
        if block_size not in hostile_survivors:
            result["correctness"][str(block_size)] = {
                "status": "failed_hostile_correctness",
                "correct": False,
            }
            continue
        mask_started = time.perf_counter()
        block_mask, coordinate_tensors = make_block_mask(
            torch,
            metadata,
            coordinates,
            "cuda",
        )
        torch.cuda.synchronize()
        mask_seconds = time.perf_counter() - mask_started
        module = make_module(torch).to("cuda")
        module.load_state_dict(clone_state(state))
        x = base_x.detach().clone().requires_grad_()
        try:  # noqa: PLW0717
            first_execution_started = time.perf_counter()
            output, gradients = run_candidate(
                module,
                x,
                lambda value, module=module, block_size=block_size, block_mask=block_mask, coordinate_tensors=coordinate_tensors: (
                    module.flex(
                        value,
                        block_mask,
                        coordinate_tensors,
                        compiled_flex,
                        block_size=block_size,
                    )
                ),
                upstream,
            )
            torch.cuda.synchronize()
            first_execution_seconds = time.perf_counter() - first_execution_started
            comparisons = compare_candidate(
                torch,
                output,
                gradients,
                explicit_output,
                explicit_gradients,
            )
            result["correctness"][str(block_size)] = {
                "status": "supported"
                if correctness_gate(comparisons)
                else "numerical_mismatch",
                "correct": correctness_gate(comparisons),
                "comparisons": comparisons,
                "block_mask_construction_seconds": mask_seconds,
                "first_compiled_forward_backward_seconds": first_execution_seconds,
            }
            del output, gradients, comparisons
        except BaseException as error:
            result["correctness"][str(block_size)] = {
                "status": "unsupported",
                "correct": False,
                "error": f"{type(error).__name__}: {error}",
            }
        finally:
            del module, x, block_mask, coordinate_tensors
            torch.cuda.empty_cache()

    compile_counters = compile_counter_snapshot(counters)
    result["compile_counters_after_correctness"] = compile_counters
    graph_breaks = compile_counters.get("graph_break", {})
    if not compile_counters or any(int(value) for value in graph_breaks.values()):
        raise RuntimeError(
            "Compiled FlexAttention evidence is absent or contains a graph break",
        )

    del explicit_module, explicit_x, explicit_output, explicit_gradients, prototype
    del upstream
    torch.cuda.empty_cache()

    result["phase"] = "benchmark"
    rows = [
        benchmark(
            torch,
            state,
            base_x,
            lambda module: lambda value: module.explicit(value, indices, valid, radial),
            candidate="explicit",
        ),
    ]
    for block_size, metadata in metadata_plans.items():
        if result["correctness"][str(block_size)]["correct"] is not True:
            rows.append({
                "status": "unsupported_or_incorrect",
                "candidate": "flex",
                "block_size": block_size,
            })
            continue
        block_mask, coordinate_tensors = make_block_mask(
            torch,
            metadata,
            coordinates,
            "cuda",
        )
        rows.append(
            benchmark(
                torch,
                state,
                base_x,
                lambda module, block_size=block_size, block_mask=block_mask, coordinate_tensors=coordinate_tensors: (
                    lambda value: module.flex(
                        value,
                        block_mask,
                        coordinate_tensors,
                        compiled_flex,
                        block_size=block_size,
                    )
                ),
                candidate="flex",
                block_size=block_size,
            ),
        )
        del block_mask, coordinate_tensors
        torch.cuda.empty_cache()
    result["rows"] = rows
    result["compile_counters_final"] = compile_counter_snapshot(counters)
    qualified = qualifying_flex_rows(rows)
    result["nomination"] = (
        {
            key: qualified[0][key]
            for key in (
                "candidate",
                "block_size",
                "mean_ms",
                "peak_allocated_bytes",
                "peak_reserved_bytes",
                "memory_ratio_vs_explicit",
                "time_ratio_vs_explicit",
            )
        }
        if qualified
        else None
    )
    result["status"] = "complete"


def main():
    """Upgrade Torch, run once and always write one compact result."""
    result = {
        "status": "failed",
        "spec": "0031",
        "scope": "local_attention_kernel_only_not_model_capacity_or_learning",
        "phase": "resolve_coordinates",
        "input_contract_sha256": INPUT_CONTRACT_SHA256,
        "pointer_sha256": POINTER_SHA256,
        "block_sizes": list(BLOCK_SIZES),
        "backend": "forced_triton_flexattention",
        "started_unix": time.time(),
    }
    try:
        coordinates = resolve_coordinates()
        result["phase"] = "upgrade_torch"
        ensure_latest_torch()
        result["phase"] = "run_probe"
        run_probe(coordinates, result)
    except BaseException:
        result["traceback"] = traceback.format_exc()
        print(result["traceback"], file=sys.stderr, flush=True)
    finally:
        result["elapsed_seconds"] = time.time() - result["started_unix"]
        result["finished_unix"] = time.time()
        OUTPUT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result), flush=True)
    return 0 if result["status"] == "complete" else 1


def ensure_latest_torch():
    """Install the same latest Torch stack used by later model execution."""
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
