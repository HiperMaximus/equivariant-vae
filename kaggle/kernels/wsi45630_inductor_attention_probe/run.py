# Copyright 2026 HiperMaximus
# ruff: noqa: ANN001, ANN201, ANN202, ANN204, BLE001, DOC201, DOC501, EM101, INP001, PLC0415, PLR0913, PLR0914, PLR0917, PLW0717, S404, SLF001, TRY003
"""Probe exact loop-free local attention on the real WSI45630 T4 graph."""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import math
import statistics
import subprocess
import sys
import time
import traceback
from collections import Counter
from operator import itemgetter
from pathlib import Path

SPEC0033_INDUCTOR_ATTENTION_PROBE_READY = True
INPUT_ROOT = Path("/kaggle/input")
OUTPUT_PATH = Path("/kaggle/working/spec0033_inductor_attention_probe.json")
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
LOCAL_CHUNK = 2_048
WARMUPS = 2
MEASURED_STEPS = 5
OUTPUT_RTOL = 5e-3
OUTPUT_ATOL = 5e-3
GRADIENT_RELATIVE_L2 = 2e-3
GRADIENT_COSINE = 0.999


def sha256(path):
    """Hash one mounted contract input."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_coordinates():
    """Authenticate and read only the exact existing WSI45630 coordinates."""
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
        or len(set(coordinates)) != PATCH_COUNT
        or any(int(row["wsi_id"]) != WSI_ID for row in rows)
        or any(x % 256 or y % 256 for x, y in coordinates)
        or coordinates != tuple(sorted(coordinates, key=itemgetter(1, 0)))
    ):
        raise RuntimeError("WSI45630 coordinate order or membership differs")
    return coordinates


def build_graph(coordinates):
    """Build fixed-25 and CSR views from physical coordinates without all-pairs work."""
    lattice = tuple((x // 256, y // 256) for x, y in coordinates)
    lookup = {point: index for index, point in enumerate(lattice)}
    radius_to_code = {value: index for index, value in enumerate(RADIAL_VALUES)}
    indices = []
    valid = []
    radial = []
    edge_query = []
    edge_key = []
    edge_radial = []
    lengths = []
    for query_index, (grid_x, grid_y) in enumerate(lattice):
        neighbours = []
        for delta_x in range(-2, 3):
            for delta_y in range(-2, 3):
                key_index = lookup.get((grid_x + delta_x, grid_y + delta_y))
                if key_index is not None:
                    neighbours.append((delta_x, delta_y, key_index))
        count = len(neighbours)
        if count < 1 or count > MAX_NEIGHBOURS:
            raise RuntimeError("Local graph degree is outside 1..25")
        row_indices = [item[2] for item in neighbours]
        row_radial = [
            radius_to_code[item[0] * item[0] + item[1] * item[1]] for item in neighbours
        ]
        padding = MAX_NEIGHBOURS - count
        indices.append(row_indices + [-1] * padding)
        valid.append([True] * count + [False] * padding)
        radial.append(row_radial + [255] * padding)
        edge_query.extend([query_index] * count)
        edge_key.extend(row_indices)
        edge_radial.extend(row_radial)
        lengths.append(count)
    return {
        "indices": indices,
        "valid": valid,
        "radial": radial,
        "edge_query": edge_query,
        "edge_key": edge_key,
        "edge_radial": edge_radial,
        "lengths": lengths,
    }


def make_module(torch):
    """Create one packed-QKV attention module shared by every candidate."""
    nn = torch.nn

    class ProbeAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(TOKEN_WIDTH, 3 * TOKEN_WIDTH, bias=False)
            self.output = nn.Linear(TOKEN_WIDTH, TOKEN_WIDTH)
            self.relative_bias = nn.Parameter(torch.zeros(HEADS, len(RADIAL_VALUES)))
            self.null_key = nn.Parameter(torch.zeros(HEADS, HEAD_WIDTH))
            self.null_bias = nn.Parameter(torch.zeros(HEADS))
            for weight in self.qkv.weight.split(TOKEN_WIDTH, dim=0):
                nn.init.xavier_uniform_(weight)
            nn.init.xavier_uniform_(self.output.weight)
            nn.init.zeros_(self.output.bias)

        def project(self, tokens):
            projected = self.qkv(tokens)
            return tuple(
                item.reshape(-1, HEADS, HEAD_WIDTH)
                for item in projected.split(TOKEN_WIDTH, dim=-1)
            )

        def fixed(self, tokens, indices, valid, radial):
            query, key, value = self.project(tokens)
            safe = indices.clamp_min(0)
            gathered_key = key[safe]
            gathered_value = value[safe]
            score = (query[:, None, :, :].float() * gathered_key.float()).sum(
                dim=-1,
            ) * (1.0 / math.sqrt(HEAD_WIDTH))
            codes = radial.long().clamp_max(len(RADIAL_VALUES) - 1)
            score = score.permute(0, 2, 1) + self.relative_bias[:, codes].permute(
                1,
                0,
                2,
            )
            score = score.masked_fill(~valid[:, None, :], -torch.inf)
            sink = (query.float() * self.null_key.float()[None, :, :]).sum(dim=-1) * (
                1.0 / math.sqrt(HEAD_WIDTH)
            )
            sink += self.null_bias.float()[None, :]
            probability = torch.softmax(
                torch.cat((score, sink[:, :, None]), dim=-1),
                dim=-1,
            )
            context = (
                probability[:, :, :MAX_NEIGHBOURS].permute(0, 2, 1)[:, :, :, None]
                * gathered_value.float()
            ).sum(dim=1)
            return self.output(context.to(value.dtype).reshape(-1, TOKEN_WIDTH))

        def segment(
            self,
            tokens,
            edge_query,
            edge_key,
            edge_radial,
            lengths,
        ):
            query, key, value = self.project(tokens)
            score = (query[edge_query].float() * key[edge_key].float()).sum(dim=-1) * (
                1.0 / math.sqrt(HEAD_WIDTH)
            )
            score += self.relative_bias[:, edge_radial.long()].transpose(0, 1)
            sink = (query.float() * self.null_key.float()[None, :, :]).sum(dim=-1) * (
                1.0 / math.sqrt(HEAD_WIDTH)
            )
            sink += self.null_bias.float()[None, :]
            edge_max = torch.segment_reduce(score, "max", lengths=lengths)
            row_max = torch.maximum(edge_max, sink)
            exponential = torch.exp(score - row_max[edge_query])
            denominator = torch.segment_reduce(
                exponential,
                "sum",
                lengths=lengths,
            ) + torch.exp(sink - row_max)
            probability = exponential / denominator[edge_query]
            messages = probability[:, :, None] * value[edge_key].float()
            context = torch.segment_reduce(messages, "sum", lengths=lengths)
            return self.output(context.to(value.dtype).reshape(-1, TOKEN_WIDTH))

        def sdpa(self, tokens, indices, valid, radial):
            from torch.nn import functional
            from torch.nn.attention import SDPBackend, sdpa_kernel

            query, key, value = self.project(tokens)
            outputs = []
            for start in range(0, tokens.shape[0], LOCAL_CHUNK):
                stop = min(start + LOCAL_CHUNK, tokens.shape[0])
                real = stop - start
                query_chunk = query[start:stop]
                index_chunk = indices[start:stop]
                valid_chunk = valid[start:stop]
                radial_chunk = radial[start:stop]
                if real < LOCAL_CHUNK:
                    padding = LOCAL_CHUNK - real
                    query_chunk = functional.pad(query_chunk, (0, 0, 0, 0, 0, padding))
                    index_chunk = functional.pad(
                        index_chunk,
                        (0, 0, 0, padding),
                        value=-1,
                    )
                    valid_chunk = functional.pad(
                        valid_chunk,
                        (0, 0, 0, padding),
                        value=False,
                    )
                    radial_chunk = functional.pad(
                        radial_chunk,
                        (0, 0, 0, padding),
                        value=255,
                    )
                safe = index_chunk.clamp_min(0)
                keys = key[safe].permute(0, 2, 1, 3)
                values = value[safe].permute(0, 2, 1, 3)
                batch = query_chunk.shape[0]
                null_key = self.null_key.to(query_chunk.dtype)[None, :, None, :].expand(
                    batch,
                    -1,
                    -1,
                    -1,
                )
                keys = torch.cat((keys, null_key), dim=2)
                values = torch.cat((values, torch.zeros_like(null_key)), dim=2)
                codes = radial_chunk.long().clamp_max(len(RADIAL_VALUES) - 1)
                bias = self.relative_bias[:, codes].permute(1, 0, 2)
                bias = bias.masked_fill(~valid_chunk[:, None, :], -torch.inf)
                bias = torch.cat(
                    (bias, self.null_bias[None, :, None].expand(batch, -1, -1)),
                    dim=2,
                ).to(query_chunk.dtype)
                with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                    context = functional.scaled_dot_product_attention(
                        query_chunk[:, :, None, :],
                        keys,
                        values,
                        attn_mask=bias[:, :, None, :],
                        dropout_p=0.0,
                        is_causal=False,
                        scale=1.0 / math.sqrt(HEAD_WIDTH),
                    )
                outputs.append(context[:real, :, 0, :])
            return self.output(torch.cat(outputs).reshape(-1, TOKEN_WIDTH))

    return ProbeAttention()


def oracle(torch, module, tokens, indices, valid, radial):
    """Independent query/head loop for the hostile small correctness graph."""
    query, key, value = module.project(tokens)
    rows = []
    with torch.autocast("cuda", enabled=False):
        for query_index in range(tokens.shape[0]):
            keep = valid[query_index]
            neighbours = indices[query_index, keep]
            codes = radial[query_index, keep].long()
            heads = []
            for head in range(HEADS):
                score = (
                    key[neighbours, head].float() @ query[query_index, head].float()
                ) / math.sqrt(HEAD_WIDTH)
                score += module.relative_bias[head, codes].float()
                sink = (
                    query[query_index, head].float() @ module.null_key[head].float()
                ) / math.sqrt(HEAD_WIDTH)
                sink += module.null_bias[head].float()
                probability = torch.softmax(torch.cat((score, sink[None])), dim=0)
                heads.append(probability[:-1] @ value[neighbours, head].float())
            rows.append(torch.stack(heads))
        context = torch.stack(rows).to(tokens.dtype).reshape(-1, TOKEN_WIDTH)
    return module.output(context)


def snapshot_gradients(module, tokens):
    """Clone input and parameter gradients for numerical comparison."""
    output = {"input": tokens.grad.detach().clone()}
    output.update({
        name: parameter.grad.detach().clone()
        for name, parameter in module.named_parameters()
    })
    return output


def metrics(torch, actual, expected):
    """Return elementwise and normwise comparison metrics."""
    actual = actual.float()
    expected = expected.float()
    delta = actual - expected
    expected_norm = float(torch.linalg.vector_norm(expected).item())
    actual_norm = float(torch.linalg.vector_norm(actual).item())
    denominator = max(expected_norm * actual_norm, 1e-12)
    return {
        "finite": bool(torch.isfinite(actual).all().item()),
        "allclose": bool(
            torch.allclose(actual, expected, rtol=OUTPUT_RTOL, atol=OUTPUT_ATOL),
        ),
        "max_abs": float(delta.abs().max().item()),
        "relative_l2": float(torch.linalg.vector_norm(delta).item())
        / max(expected_norm, 1e-12),
        "cosine": float((actual.flatten() * expected.flatten()).sum().item())
        / denominator,
        "reference_norm": expected_norm,
    }


def correctness(torch):
    """Compare eager fixed-25 forward and every gradient with an independent oracle."""
    coordinates = (
        (0, 0),
        (256, 0),
        (1024, 0),
        (256, 256),
        (768, 512),
        (1024, 512),
        (2048, 2048),
    )
    graph_lists = build_graph(coordinates)
    graph = tuple(
        torch.tensor(graph_lists[name], device="cuda", dtype=dtype)
        for name, dtype in (
            ("indices", torch.int64),
            ("valid", torch.bool),
            ("radial", torch.uint8),
        )
    )
    torch.manual_seed(3303)
    base = make_module(torch).to("cuda")
    with torch.no_grad():
        base.relative_bias.copy_(
            torch.linspace(
                -0.3,
                0.3,
                HEADS * len(RADIAL_VALUES),
                device="cuda",
            ).reshape(HEADS, -1),
        )
        base.null_key.copy_(
            torch.linspace(-0.2, 0.2, HEADS * HEAD_WIDTH, device="cuda").reshape(
                HEADS,
                HEAD_WIDTH,
            ),
        )
        base.null_bias.copy_(torch.linspace(-0.15, 0.15, HEADS, device="cuda"))
    state = {
        name: tensor.detach().clone() for name, tensor in base.state_dict().items()
    }
    upstream = torch.cos(
        torch.arange(len(coordinates) * TOKEN_WIDTH, device="cuda", dtype=torch.float32)
        * 0.013,
    ).reshape(len(coordinates), TOKEN_WIDTH)
    records = {}
    outputs = {}
    gradients = {}
    for name in ("oracle", "fixed", "sdpa"):
        module = make_module(torch).to("cuda")
        module.load_state_dict(state)
        tokens = torch.randn(
            len(coordinates),
            TOKEN_WIDTH,
            device="cuda",
            dtype=torch.float16,
        ).requires_grad_()
        if name != "oracle":
            tokens.data.copy_(records["oracle_tokens"])
        else:
            records["oracle_tokens"] = tokens.detach().clone()
        with torch.autocast("cuda", dtype=torch.float16):
            output = (
                oracle(torch, module, tokens, *graph)
                if name == "oracle"
                else getattr(module, name)(tokens, *graph)
            )
            loss = (output.float() * upstream).sum()
        loss.backward()
        outputs[name] = output.detach()
        gradients[name] = snapshot_gradients(module, tokens)
    comparisons = {}
    for name in ("fixed", "sdpa"):
        comparisons[f"{name}:output"] = metrics(torch, outputs[name], outputs["oracle"])
        for parameter_name, gradient in gradients[name].items():
            comparisons[f"{name}:grad:{parameter_name}"] = metrics(
                torch,
                gradient,
                gradients["oracle"][parameter_name],
            )
    correct = all(
        item["finite"]
        and (
            item["allclose"]
            if name.endswith(":output")
            else item["relative_l2"] <= GRADIENT_RELATIVE_L2
            and item["cosine"] >= GRADIENT_COSINE
        )
        for name, item in comparisons.items()
    )
    records.pop("oracle_tokens")
    records.update({"correct": correct, "comparisons": comparisons})
    return records


def device_graph(torch, graph_lists):
    """Create fixed and edge views on CUDA."""
    fixed = tuple(
        torch.tensor(graph_lists[name], device="cuda", dtype=dtype)
        for name, dtype in (
            ("indices", torch.int64),
            ("valid", torch.bool),
            ("radial", torch.uint8),
        )
    )
    edge = tuple(
        torch.tensor(graph_lists[name], device="cuda", dtype=dtype)
        for name, dtype in (
            ("edge_query", torch.int64),
            ("edge_key", torch.int64),
            ("edge_radial", torch.uint8),
            ("lengths", torch.int64),
        )
    )
    return fixed, edge


def mark_dynamic(torch, tokens, graph, *, edge=False):
    """Mark only node and optional edge axes dynamic with strict real bounds."""
    torch._dynamo.mark_dynamic(tokens, 0, min=1, max=PATCH_COUNT)
    if edge:
        for tensor in graph[:3]:
            torch._dynamo.mark_dynamic(
                tensor,
                0,
                min=1,
                max=PATCH_COUNT * MAX_NEIGHBOURS,
            )
        torch._dynamo.mark_dynamic(graph[3], 0, min=1, max=PATCH_COUNT)
    else:
        for tensor in graph:
            torch._dynamo.mark_dynamic(tensor, 0, min=1, max=PATCH_COUNT)


def compile_candidate(torch, method, profile):
    """Compile one isolated exact candidate under a named option profile."""
    kwargs = {
        "fullgraph": True,
        "dynamic": None,
        "backend": "inductor",
        "mode": profile.get("mode"),
        "options": profile.get("options"),
    }
    signature = inspect.signature(torch.compile)
    if "recompile_limit" in signature.parameters:
        kwargs["recompile_limit"] = 1
    if "isolate_recompiles" in signature.parameters:
        kwargs["isolate_recompiles"] = True
    return torch.compile(method, **kwargs)


def benchmark(torch, state, tokens, graph, *, backend, profile):
    """Measure one full-graph forward/backward row with finite-gradient checks."""
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    torch._inductor.metrics.reset()
    module = make_module(torch).to("cuda")
    module.load_state_dict(state)
    candidate_tokens = tokens.detach().clone().requires_grad_()
    method = getattr(module, backend)
    compiled = None
    if backend != "sdpa":
        mark_dynamic(torch, candidate_tokens, graph, edge=backend == "segment")
        compiled = compile_candidate(torch, method, profile)
        method = compiled

    def step():
        module.zero_grad(set_to_none=True)
        candidate_tokens.grad = None
        with torch.autocast("cuda", dtype=torch.float16):
            output = method(candidate_tokens, *graph)
            loss = output.float().square().mean()
        loss.backward()
        return loss

    try:
        for _ in range(WARMUPS):
            step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        timings = []
        losses = []
        for _ in range(MEASURED_STEPS):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            loss = step()
            end.record()
            end.synchronize()
            timings.append(float(start.elapsed_time(end)))
            losses.append(float(loss.detach().item()))
        finite = candidate_tokens.grad is not None and bool(
            torch.isfinite(candidate_tokens.grad).all().item(),
        )
        finite = finite and all(
            parameter.grad is not None
            and bool(torch.isfinite(parameter.grad).all().item())
            for parameter in module.parameters()
        )
        return {
            "status": "ok" if finite else "nonfinite",
            "backend": backend,
            "profile": profile["name"],
            "mean_ms": statistics.fmean(timings),
            "min_ms": min(timings),
            "max_ms": max(timings),
            "timings_ms": timings,
            "losses": losses,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "unique_graphs": int(
                torch._dynamo.utils.counters["stats"]["unique_graphs"],
            ),
            "graph_breaks": int(
                sum(torch._dynamo.utils.counters["graph_break"].values()),
            ),
            "generated_kernel_count": int(
                getattr(torch._inductor.metrics, "generated_kernel_count", -1),
            ),
            "finite_gradients": finite,
        }
    except BaseException as error:
        return {
            "status": "failed",
            "backend": backend,
            "profile": profile["name"],
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
    finally:
        torch.cuda.empty_cache()


def profiles(torch):
    """Feature-detect default, official max-autotune and experimental profiles."""
    available = set(torch._inductor.list_options())
    aggressive = {
        "max_autotune": True,
        "coordinate_descent_tuning": True,
        "max_autotune_pointwise": True,
        "combo_kernels": True,
        "benchmark_combo_kernel": True,
        "triton.cudagraphs": False,
    }
    output = [
        {"name": "default", "mode": "default", "options": None},
        {
            "name": "max_autotune_no_cudagraphs",
            "mode": "max-autotune-no-cudagraphs",
            "options": None,
        },
    ]
    if set(aggressive).issubset(available):
        output.append({
            "name": "aggressive_experimental",
            "mode": None,
            "options": aggressive,
        })
    return output


def run_probe(coordinates, result):
    """Run correctness and exact real-graph T4 measurements."""
    import torch

    if torch.cuda.device_count() < 1 or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Spec 0033 requires a Tesla T4 as CUDA device 0")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    result["runtime"] = {
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": torch.cuda.get_device_name(0),
        "capability": torch.cuda.get_device_capability(0),
        "total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "mode_options": torch._inductor.list_mode_options(),
    }
    result["phase"] = "correctness"
    result["correctness"] = correctness(torch)
    if not result["correctness"]["correct"]:
        raise RuntimeError("Fixed-25 or SDPA failed the independent oracle gate")
    result["phase"] = "real_graph"
    graph_lists = build_graph(coordinates)
    fixed, edge = device_graph(torch, graph_lists)
    degrees = graph_lists["lengths"]
    result["graph"] = {
        "nodes": PATCH_COUNT,
        "edges": len(graph_lists["edge_key"]),
        "minimum_degree": min(degrees),
        "maximum_degree": max(degrees),
        "degree_histogram": {
            str(key): value for key, value in sorted(Counter(degrees).items())
        },
        "fixed_slot_occupancy": len(graph_lists["edge_key"])
        / (PATCH_COUNT * MAX_NEIGHBOURS),
    }
    torch.manual_seed(3304)
    prototype = make_module(torch).to("cuda")
    state = {
        name: tensor.detach().clone() for name, tensor in prototype.state_dict().items()
    }
    tokens = torch.randn(PATCH_COUNT, TOKEN_WIDTH, device="cuda", dtype=torch.float16)
    rows = []
    baseline = benchmark(
        torch,
        state,
        tokens,
        fixed,
        backend="sdpa",
        profile={"name": "eager_chunk2048", "mode": None, "options": None},
    )
    rows.append(baseline)
    print(json.dumps(baseline), flush=True)
    for profile in profiles(torch):
        row = benchmark(torch, state, tokens, fixed, backend="fixed", profile=profile)
        rows.append(row)
        print(json.dumps(row), flush=True)
    segment_row = benchmark(
        torch,
        state,
        tokens,
        edge,
        backend="segment",
        profile={"name": "default_control", "mode": "default", "options": None},
    )
    rows.append(segment_row)
    print(json.dumps(segment_row), flush=True)
    result["rows"] = rows
    fixed_rows = [
        row
        for row in rows
        if row.get("backend") == "fixed" and row.get("status") == "ok"
    ]
    if not fixed_rows:
        raise RuntimeError("No loop-free fixed-25 T4 row completed")
    result["selected_fixed"] = min(fixed_rows, key=itemgetter("mean_ms"))
    result["baseline"] = baseline
    result["status"] = "complete"


def ensure_latest_torch():
    """Install the current PyTorch CUDA stack before its first import."""
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


def main():
    """Execute once and always preserve a compact result artifact."""
    result = {
        "status": "failed",
        "spec": "0033",
        "scope": "local_attention_backend_probe_not_learning_or_test_access",
        "phase": "resolve_coordinates",
        "input_contract_sha256": INPUT_CONTRACT_SHA256,
        "pointer_sha256": POINTER_SHA256,
        "started_unix": time.time(),
    }
    try:
        coordinates = resolve_coordinates()
        result["phase"] = "upgrade_torch"
        ensure_latest_torch()
        result["phase"] = "probe"
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


if __name__ == "__main__":
    raise SystemExit(main())
