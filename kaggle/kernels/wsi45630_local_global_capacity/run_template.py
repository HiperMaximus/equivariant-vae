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
# __EMBEDDED_SPEC0026_MODEL__
# END EMBEDDED SPEC0026 MODEL

KAGGLE_LOCAL_GLOBAL_CAPACITY_READY = True
INPUT_ROOT = Path("/kaggle/input")
OUTPUT_PATH = Path("/kaggle/working/spec0030_local_global_mil_capacity.json")
CAPACITY_CONTRACT_JSON = r"""$capacity_contract_json"""
CAPACITY_CONTRACT_SHA256 = "$capacity_contract_sha256"
MODEL_SHA256 = "$model_sha256"
EMBEDDED_MODEL_SHA256 = "$embedded_model_sha256"
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
