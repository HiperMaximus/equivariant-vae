# Copyright 2026 HiperMaximus
# ruff: noqa: ANN001, ANN201, ANN202, ANN204, BLE001, DOC201, DOC501, EM101, E501, INP001, PLC0415, PLW0717, S404, T201, TRY003
"""Benchmark exact local-softmax implementations on the real WSI45630 graph."""

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
from contextlib import nullcontext
from operator import itemgetter
from pathlib import Path

KAGGLE_LOCAL_ATTENTION_REPAIR_PROBE_READY = True
INPUT_ROOT = Path("/kaggle/input")
OUTPUT_PATH = Path("/kaggle/working/spec0028_local_attention_repair_probe.json")
INPUT_CONTRACT_NAME = "wsi45630_capacity_input.json"
INPUT_CONTRACT_SHA256 = (
    "99bb4d2f60558aee9691b67be4867ffae434bc306581a000fd5d72a6befac660"
)
POINTER_SHA256 = "08e461846bf16efebac707c82962762f49837916986b29aee0dcd6ca1fc31c6c"
DATASET_REFERENCE = "maximusshtefan/eqvae-wsi45630-capacity-inputs"
WSI_ID = 45_630
PATCH_COUNT = 32_595
TOKEN_WIDTH = 192
HEADS = 6
HEAD_WIDTH = 32
MAX_NEIGHBOURS = 25
RADIAL_VALUES = (0, 1, 2, 4, 5, 8)
CHUNK_SIZES = (2_048, 8_192, PATCH_COUNT)
BACKENDS = ("explicit", "sdpa_auto", "sdpa_efficient", "sdpa_flash")
CORRECTNESS_QUERIES = 257
CORRECTNESS_RTOL = 5e-3
CORRECTNESS_ATOL = 5e-3
MAX_CORRECTNESS_RELATIVE_L2 = 2e-3
MIN_CORRECTNESS_COSINE = 0.999
MIN_REFERENCE_GRADIENT_NORM = 1e-6
WARMUP_STEPS = 2
MEASURED_STEPS = 5


def sha256(path):
    """Hash a compact contract or coordinate file before it influences timing."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_coordinates():
    """Authenticate and read only coordinates from the existing private input."""
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
        or any(x % 256 or y % 256 for x, y in coordinates)
        or coordinates != tuple(sorted(coordinates, key=itemgetter(1, 0)))
    ):
        raise RuntimeError(
            "Expected all unique WSI45630 coordinates in numeric y,x order",
        )
    return coordinates


def build_graph(coordinates):
    """Build the exact sparse radius-2 graph without an all-pairs allocation."""
    lattice = tuple((x // 256, y // 256) for x, y in coordinates)
    lookup = {point: index for index, point in enumerate(lattice)}
    radial_code = {value: index for index, value in enumerate(RADIAL_VALUES)}
    indices = []
    valid = []
    radial = []
    degrees = []
    for gx, gy in lattice:
        neighbours = []
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                neighbour = lookup.get((gx + dx, gy + dy))
                if neighbour is not None:
                    neighbours.append((dx, dy, neighbour))
        neighbours.sort(key=itemgetter(0, 1))
        degrees.append(len(neighbours))
        row_indices = [item[2] for item in neighbours]
        row_radial = [radial_code[item[0] ** 2 + item[1] ** 2] for item in neighbours]
        padding = MAX_NEIGHBOURS - len(neighbours)
        indices.append(row_indices + [-1] * padding)
        valid.append([True] * len(neighbours) + [False] * padding)
        radial.append(row_radial + [255] * padding)
    if min(degrees) < 1 or max(degrees) > MAX_NEIGHBOURS:
        raise RuntimeError("Local graph degree is outside the locked boundary")
    return indices, valid, radial, degrees


def graph_sha256(coordinates, indices, valid, radial):
    """Hash the canonical Spec 0026 graph encoding for artifact identity."""
    digest = hashlib.sha256()
    metadata = {
        "format": "eqvae_spec0026_graph_v1",
        "wsi_id": WSI_ID,
        "coordinates": {
            "meaning": "ordered_patch_lattice_gx_gy",
            "shape": [len(coordinates), 2],
            "dtype": "little_endian_int64",
        },
        "neighbor_index": {
            "shape": [len(indices), MAX_NEIGHBOURS],
            "dtype": "little_endian_int64",
            "padding_sentinel": -1,
        },
        "neighbor_valid": {
            "shape": [len(valid), MAX_NEIGHBOURS],
            "dtype": "uint8_boolean_0_or_1",
        },
        "radial_code": {
            "shape": [len(radial), MAX_NEIGHBOURS],
            "dtype": "uint8",
            "padding_sentinel": 255,
            "codebook_squared_radius": list(RADIAL_VALUES),
        },
    }
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
    for x, y in coordinates:
        digest.update(struct.pack("<qq", x // 256, y // 256))
    for row in indices:
        digest.update(struct.pack(f"<{MAX_NEIGHBOURS}q", *row))
    for row in valid:
        digest.update(bytes(int(item) for item in row))
    for row in radial:
        digest.update(bytes(row))
    return digest.hexdigest()


def make_module(torch):
    """Create one local-attention sublayer shared by every backend candidate."""
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

        def forward(  # noqa: PLR0913, PLR0914
            self,
            x,
            neighbour_index,
            neighbour_valid,
            radial_code,
            *,
            backend,
            chunk_size,
            query_count=None,
        ):
            from torch.nn import functional

            count = x.shape[0] if query_count is None else query_count
            q = self.q_proj(x).reshape(-1, HEADS, HEAD_WIDTH)
            k = self.k_proj(x).reshape(-1, HEADS, HEAD_WIDTH)
            v = self.v_proj(x).reshape(-1, HEADS, HEAD_WIDTH)
            outputs = []
            backend_context = _backend_context(backend)
            with backend_context:
                for start in range(0, count, chunk_size):
                    stop = min(start + chunk_size, count)
                    safe = neighbour_index[start:stop].clamp_min(0)
                    valid_chunk = neighbour_valid[start:stop]
                    codes = radial_code[start:stop].clamp_max(len(RADIAL_VALUES) - 1)
                    keys = k[safe].permute(0, 2, 1, 3)
                    values = v[safe].permute(0, 2, 1, 3)
                    batch = stop - start
                    null_key = self.null_key.to(keys.dtype)[None, :, None, :].expand(
                        batch,
                        -1,
                        -1,
                        -1,
                    )
                    null_value = torch.zeros(
                        (batch, HEADS, 1, HEAD_WIDTH),
                        device=x.device,
                        dtype=values.dtype,
                    )
                    keys = torch.cat((keys, null_key), dim=2)
                    values = torch.cat((values, null_value), dim=2)
                    patch_bias = self.relative_bias[:, codes.long()].permute(1, 0, 2)
                    patch_bias = patch_bias.masked_fill(
                        ~valid_chunk[:, None, :],
                        -torch.inf,
                    )
                    sink_bias = self.null_bias[None, :, None].expand(batch, -1, -1)
                    attention_bias = torch.cat((patch_bias, sink_bias), dim=2)[
                        :,
                        :,
                        None,
                        :,
                    ]
                    queries = q[start:stop, :, None, :]
                    if backend == "explicit":
                        with torch.autocast(device_type=x.device.type, enabled=False):
                            scores = torch.matmul(
                                queries.float(),
                                keys.float().transpose(-2, -1),
                            ) / math.sqrt(HEAD_WIDTH)
                            weights = torch.softmax(
                                scores + attention_bias.float(),
                                dim=-1,
                            )
                            context = torch.matmul(weights, values.float()).to(x.dtype)
                    else:
                        context = functional.scaled_dot_product_attention(
                            queries,
                            keys,
                            values,
                            attn_mask=attention_bias.to(queries.dtype),
                            dropout_p=0.0,
                        )
                    outputs.append(context[:, :, 0, :].reshape(batch, TOKEN_WIDTH))
            return self.out_proj(torch.cat(outputs, dim=0))

    return LocalAttention()


def oracle_forward(  # noqa: PLR0913, PLR0917
    torch,
    module,
    x,
    neighbour_index,
    neighbour_valid,
    radial_code,
    query_count,
):
    """Independently evaluate the small correctness slice without padded gathers."""
    with torch.autocast(
        device_type=x.device.type,
        dtype=torch.float16,
        enabled=x.is_cuda,
    ):
        q = module.q_proj(x).reshape(-1, HEADS, HEAD_WIDTH)
        k = module.k_proj(x).reshape(-1, HEADS, HEAD_WIDTH)
        v = module.v_proj(x).reshape(-1, HEADS, HEAD_WIDTH)
    contexts = []
    scale = math.sqrt(HEAD_WIDTH)
    with torch.autocast(device_type=x.device.type, enabled=False):
        for query_index in range(query_count):
            keep = neighbour_valid[query_index]
            neighbours = neighbour_index[query_index][keep]
            codes = radial_code[query_index][keep].long()
            query = q[query_index]
            keys = k[neighbours]
            values = v[neighbours]
            patch_scores = (keys.float() * query[None, :, :].float()).sum(
                dim=-1,
            ).T / scale
            patch_scores += module.relative_bias[:, codes].float()
            null_score = (query.float() * module.null_key.float()).sum(
                dim=-1,
            ) / scale + module.null_bias.float()
            weights = torch.softmax(
                torch.cat((patch_scores, null_score[:, None]), dim=1),
                dim=1,
            )
            context = (weights[:, :-1].T[:, :, None] * values.float()).sum(dim=0)
            contexts.append(context.reshape(TOKEN_WIDTH).to(x.dtype))
    with torch.autocast(
        device_type=x.device.type,
        dtype=torch.float16,
        enabled=x.is_cuda,
    ):
        return module.out_proj(torch.stack(contexts))


def _backend_context(backend):
    """Force only named fused backends so rejection cannot silently fall back."""
    if backend in {"explicit", "sdpa_auto"}:
        return nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel

    selected = {
        "sdpa_efficient": SDPBackend.EFFICIENT_ATTENTION,
        "sdpa_flash": SDPBackend.FLASH_ATTENTION,
    }[backend]
    return sdpa_kernel(selected)


def _clone_state(state):
    """Create independent bytes so candidates cannot share mutable gradients."""
    return {name: tensor.detach().clone() for name, tensor in state.items()}


def _run_once(  # noqa: PLR0913
    torch,
    module,
    x,
    graph,
    *,
    backend,
    chunk_size,
    query_count=None,
    upstream=None,
):
    """Execute one differentiable candidate and return its output and scalar loss."""
    neighbour_index, neighbour_valid, radial_code = graph
    with torch.autocast("cuda", dtype=torch.float16):
        output = module(
            x,
            neighbour_index,
            neighbour_valid,
            radial_code,
            backend=backend,
            chunk_size=chunk_size,
            query_count=query_count,
        )
        loss = (
            output.float().square().mean()
            if upstream is None
            else (output.float() * upstream).sum()
        )
    loss.backward()
    return output.detach(), loss.detach()


def _gradient_snapshot(module, x):
    """Capture only finite comparison tensors after a completed backward pass."""
    gradients = {"input": x.grad.detach().clone()}
    gradients.update({
        name: parameter.grad.detach().clone()
        for name, parameter in module.named_parameters()
    })
    return gradients


def comparison_metrics(torch, actual, expected):
    """Report elementwise and normwise evidence for one comparison tensor."""
    actual32 = actual.float()
    expected32 = expected.float()
    delta = (actual32 - expected32).abs()
    expected_l2 = float(torch.linalg.vector_norm(expected32).item())
    actual_l2 = float(torch.linalg.vector_norm(actual32).item())
    delta_l2 = float(torch.linalg.vector_norm(delta).item())
    denominator = max(expected_l2 * actual_l2, 1e-12)
    cosine = (
        float((actual32.flatten() * expected32.flatten()).sum().item()) / denominator
    )
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
        "violation_fraction": float(violations.float().mean().item()),
        "max_abs": float(delta.max().item()),
        "mean_abs": float(delta.mean().item()),
        "expected_l2": expected_l2,
        "actual_l2": actual_l2,
        "relative_l2": delta_l2 / max(expected_l2, 1e-12),
        "cosine_similarity": cosine,
    }


def correctness_gate(comparisons):
    """Apply strict output and precommitted normwise gradient criteria."""
    for name, item in comparisons.items():
        if not item["finite"]:
            return False
        if name.endswith("output"):
            if not item["allclose"]:
                return False
        elif (
            item["relative_l2"] > MAX_CORRECTNESS_RELATIVE_L2
            or item["cosine_similarity"] < MIN_CORRECTNESS_COSINE
        ):
            return False
    return True


def _comparisons(  # noqa: PLR0913, PLR0917
    torch,
    actual_output,
    actual_gradients,
    expected_output,
    expected_gradients,
    prefix,
):
    """Compare one candidate with a named reference without hiding violations."""
    records = {
        f"{prefix}:output": comparison_metrics(
            torch,
            actual_output,
            expected_output,
        ),
    }
    records.update({
        f"{prefix}:grad:{name}": comparison_metrics(
            torch,
            tensor,
            expected_gradients[name],
        )
        for name, tensor in actual_gradients.items()
    })
    return records


def correctness_checks(torch, prototype, base_x, graph):
    """Reject backend drift before any full-graph performance measurement."""
    state = _clone_state(prototype.state_dict())
    state["relative_bias"] = torch.linspace(
        -0.3,
        0.3,
        HEADS * len(RADIAL_VALUES),
    ).reshape(
        HEADS,
        len(RADIAL_VALUES),
    )
    state["null_key"] = torch.linspace(-0.2, 0.2, HEADS * HEAD_WIDTH).reshape(
        HEADS,
        HEAD_WIDTH,
    )
    state["null_bias"] = torch.linspace(-0.15, 0.15, HEADS)
    records = {}
    reference_module = make_module(torch).to("cuda")
    reference_module.load_state_dict(_clone_state(state))
    reference_x = base_x.detach().clone().requires_grad_()
    upstream_phase = torch.arange(
        CORRECTNESS_QUERIES * TOKEN_WIDTH,
        device="cuda",
        dtype=torch.float32,
    )
    upstream = torch.cos(upstream_phase * 0.013).reshape(
        CORRECTNESS_QUERIES,
        TOKEN_WIDTH,
    )
    reference_output = oracle_forward(
        torch,
        reference_module,
        reference_x,
        *graph,
        CORRECTNESS_QUERIES,
    )
    (reference_output.float() * upstream).sum().backward()
    reference_gradients = _gradient_snapshot(reference_module, reference_x)
    reference_norms = {
        name: float(torch.linalg.vector_norm(tensor.float()).item())
        for name, tensor in reference_gradients.items()
    }
    if any(
        not math.isfinite(norm) or norm <= MIN_REFERENCE_GRADIENT_NORM
        for norm in reference_norms.values()
    ):
        message = f"Oracle produced a trivial gradient: {reference_norms}"
        raise RuntimeError(message)
    explicit_module = make_module(torch).to("cuda")
    explicit_module.load_state_dict(_clone_state(state))
    explicit_x = base_x.detach().clone().requires_grad_()
    explicit_output, _ = _run_once(
        torch,
        explicit_module,
        explicit_x,
        graph,
        backend="explicit",
        chunk_size=CORRECTNESS_QUERIES,
        query_count=CORRECTNESS_QUERIES,
        upstream=upstream,
    )
    explicit_gradients = _gradient_snapshot(explicit_module, explicit_x)
    explicit_comparisons = _comparisons(
        torch,
        explicit_output,
        explicit_gradients,
        reference_output,
        reference_gradients,
        "oracle",
    )
    explicit_correct = correctness_gate(explicit_comparisons)
    records["explicit"] = {
        "status": "supported" if explicit_correct else "numerical_mismatch",
        "correct": explicit_correct,
        "comparisons": explicit_comparisons,
    }
    for backend in BACKENDS[1:]:
        module = make_module(torch).to("cuda")
        module.load_state_dict(_clone_state(state))
        x = base_x.detach().clone().requires_grad_()
        try:
            output, _ = _run_once(
                torch,
                module,
                x,
                graph,
                backend=backend,
                chunk_size=CORRECTNESS_QUERIES,
                query_count=CORRECTNESS_QUERIES,
                upstream=upstream,
            )
            gradients = _gradient_snapshot(module, x)
            comparisons = _comparisons(
                torch,
                output,
                gradients,
                explicit_output,
                explicit_gradients,
                "explicit",
            )
            comparisons.update(
                _comparisons(
                    torch,
                    output,
                    gradients,
                    reference_output,
                    reference_gradients,
                    "oracle",
                ),
            )
            correct = explicit_correct and correctness_gate(comparisons)
            records[backend] = {
                "status": "supported" if correct else "numerical_mismatch",
                "correct": correct,
                "comparisons": comparisons,
            }
        except BaseException as error:
            records[backend] = {
                "status": "unsupported",
                "correct": False,
                "error": f"{type(error).__name__}: {error}",
            }
        finally:
            del module, x
            torch.cuda.empty_cache()
    del (
        reference_module,
        reference_x,
        reference_output,
        reference_gradients,
        explicit_module,
        explicit_x,
        explicit_output,
        explicit_gradients,
    )
    torch.cuda.empty_cache()
    return records


def benchmark_candidate(  # noqa: PLR0913
    torch,
    state,
    base_x,
    graph,
    *,
    backend,
    chunk_size,
):
    """Time full-graph forward/backward only after backend correctness passes."""
    module = make_module(torch).to("cuda")
    module.load_state_dict(_clone_state(state))
    x = base_x.detach().clone().requires_grad_()
    try:
        for _ in range(WARMUP_STEPS):
            module.zero_grad(set_to_none=True)
            x.grad = None
            _run_once(torch, module, x, graph, backend=backend, chunk_size=chunk_size)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        timings = []
        losses = []
        for _ in range(MEASURED_STEPS):
            module.zero_grad(set_to_none=True)
            x.grad = None
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _, loss = _run_once(
                torch,
                module,
                x,
                graph,
                backend=backend,
                chunk_size=chunk_size,
            )
            end.record()
            end.synchronize()
            timings.append(float(start.elapsed_time(end)))
            losses.append(float(loss.item()))
        finite = (
            all(
                parameter.grad is not None
                and bool(torch.isfinite(parameter.grad).all().item())
                for parameter in module.parameters()
            )
            and x.grad is not None
            and bool(torch.isfinite(x.grad).all().item())
        )
        return {
            "status": "ok" if finite else "nonfinite",
            "backend": backend,
            "chunk_size": chunk_size,
            "mean_ms": statistics.fmean(timings),
            "min_ms": min(timings),
            "max_ms": max(timings),
            "timings_ms": timings,
            "losses": losses,
            "finite_gradients": finite,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
    except BaseException as error:
        return {
            "status": "failed",
            "backend": backend,
            "chunk_size": chunk_size,
            "error": f"{type(error).__name__}: {error}",
        }
    finally:
        del module, x
        torch.cuda.empty_cache()


def select_winner(rows):
    """Select only a finite correctness-gated row by measured full-step time."""
    eligible = [
        row
        for row in rows
        if row.get("status") == "ok" and row.get("finite_gradients") is True
    ]
    if not eligible:
        return None
    winner = min(eligible, key=itemgetter("mean_ms"))
    return {
        key: winner[key]
        for key in (
            "backend",
            "chunk_size",
            "mean_ms",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
        )
    }


def run_probe(coordinates, result):
    """Execute the bounded one-GPU graph-kernel comparison on the target T4."""
    import torch

    result["runtime"] = {
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "devices": [
            {
                "name": torch.cuda.get_device_name(index),
                "capability": torch.cuda.get_device_capability(index),
                "total_bytes": torch.cuda.get_device_properties(index).total_memory,
            }
            for index in range(torch.cuda.device_count())
        ],
    }
    if torch.cuda.device_count() < 1 or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Spec 0027 requires a T4 as CUDA device 0")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.manual_seed(1701)
    result["phase"] = "build_graph"
    indices, valid, radial, degrees = build_graph(coordinates)
    result["graph"] = {
        "sha256": graph_sha256(coordinates, indices, valid, radial),
        "nodes": len(coordinates),
        "edges": sum(degrees),
        "minimum_degree": min(degrees),
        "maximum_degree": max(degrees),
        "degree_histogram": {
            str(key): value for key, value in sorted(Counter(degrees).items())
        },
        "radial_values": list(RADIAL_VALUES),
    }
    graph = (
        torch.tensor(indices, dtype=torch.int64, device="cuda"),
        torch.tensor(valid, dtype=torch.bool, device="cuda"),
        torch.tensor(radial, dtype=torch.uint8, device="cuda"),
    )
    base_x = torch.randn((PATCH_COUNT, TOKEN_WIDTH), device="cuda", dtype=torch.float16)
    prototype = make_module(torch).to("cuda")
    state = _clone_state(prototype.state_dict())
    result["parameter_count"] = sum(
        parameter.numel() for parameter in prototype.parameters()
    )
    result["phase"] = "correctness"
    result["correctness"] = correctness_checks(torch, prototype, base_x, graph)
    result["phase"] = "benchmark"
    rows = []
    for backend in BACKENDS:
        permitted = result["correctness"].get(backend, {}).get("correct") is True
        for chunk_size in CHUNK_SIZES:
            if not permitted:
                rows.append({
                    "status": "unsupported_or_incorrect",
                    "backend": backend,
                    "chunk_size": chunk_size,
                })
                continue
            row = benchmark_candidate(
                torch,
                state,
                base_x,
                graph,
                backend=backend,
                chunk_size=chunk_size,
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
    result["rows"] = rows
    result["selected"] = select_winner(rows)
    if len(rows) != len(BACKENDS) * len(CHUNK_SIZES) or result["selected"] is None:
        raise RuntimeError("No complete finite local-attention candidate survived")
    result["excluded"] = {
        "flex_attention": "block-sparse tiles do not represent the irregular degree-25 graph without substantial partial-block work",
        "custom_triton": "outside the bounded standard-PyTorch probe",
    }
    result["status"] = "complete"


def main():
    """Upgrade the runtime, execute once and always persist compact evidence."""
    result = {
        "status": "failed",
        "scope": "local_attention_kernel_only_not_model_capacity_or_learning",
        "spec": "0028",
        "phase": "resolve_coordinates",
        "input_contract_sha256": INPUT_CONTRACT_SHA256,
        "pointer_sha256": POINTER_SHA256,
        "backends": list(BACKENDS),
        "chunk_sizes": list(CHUNK_SIZES),
        "correctness_contract": {
            "queries": CORRECTNESS_QUERIES,
            "output_rtol": CORRECTNESS_RTOL,
            "output_atol": CORRECTNESS_ATOL,
            "gradient_max_relative_l2": MAX_CORRECTNESS_RELATIVE_L2,
            "gradient_min_cosine": MIN_CORRECTNESS_COSINE,
            "minimum_oracle_gradient_norm": MIN_REFERENCE_GRADIENT_NORM,
            "gradient_elementwise_allclose_is_diagnostic": True,
        },
        "started_unix": time.time(),
    }
    try:
        coordinates = resolve_coordinates()
        result["phase"] = "upgrade_torch"
        _ensure_latest_torch()
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


def _ensure_latest_torch():
    """Use the same latest-stack policy as the eventual implementation runtime."""
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
