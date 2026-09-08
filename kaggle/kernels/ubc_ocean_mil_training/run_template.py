# Copyright 2026 HiperMaximus
# ruff: noqa: ANN001, ANN002, ANN201, ANN202, B023, BLE001, C901, D103, E501, EM101, EM102, FBT003, INP001, PLC0415, PLR0912, PLR0913, PLR0914, PLR0915, PLR0916, PLR0917, PLR2004, PLW0717, S404, S603, SLF001, T201, TRY003, TRY004, TRY301
"""Run the locked Spec 0036 independent dual-T4 MIL campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter
from dataclasses import asdict
from operator import itemgetter
from pathlib import Path

SPEC0036_LOCAL_GLOBAL_MIL_TRAINING_READY = True
INPUT_ROOT = Path("/kaggle/input")
WORKING_ROOT = Path("/kaggle/working")
OUTPUT_ROOT = WORKING_ROOT / "spec0036_mil_training"
INPUT_CONTRACT_NAME = "mil_training_input.json"
INPUT_CONTRACT_SHA256 = "$input_contract_sha256"
INPUT_DATASET_REFERENCE = "$input_dataset_reference"
RESUME_CONTRACT_NAME = "mil_training_resume.json"
RESUME_DATASET_REFERENCE = "$resume_dataset_reference"
RESUME_CONTRACT_SHA256 = "$resume_contract_sha256"
PINNED_TORCH_VERSION = "2.14.0"
PINNED_TORCH_CUDA = "13.0"
PINNED_TORCH_INDEX = "https://download.pytorch.org/whl/cu130"
DEVELOPMENT_CONTRACT_SHA256 = (
    "6aa25a3f56db62b903d3451e154c47dc0039233f88179c7bdab03b7a48319fe0"
)
INITIAL_STATE_HASH_SCHEMA = b"eqvae_spec0036_initial_state_v1"
BRANCH_DEVICES = {"normal_vae": 0, "so2_vae": 1}
KERNEL_SOURCES = (
    "maximusshtefan/eqvae-ubc-ocean-latent-run-01",
    "maximusshtefan/eqvae-ubc-ocean-latent-run-02",
    "maximusshtefan/eqvae-ubc-ocean-latent-run-03",
    "maximusshtefan/eqvae-ubc-ocean-latent-run-04",
    "maximusshtefan/eqvae-ubc-ocean-latent-run-05",
    "maximusshtefan/eqvae-ubc-ocean-cancer-latent-top-up",
    "maximusshtefan/eqvae-wsi45630-completion",
    "maximusshtefan/eqvae-full-foreground-01",
    "maximusshtefan/eqvae-full-foreground-02",
    "maximusshtefan/eqvae-full-foreground-03",
    "maximusshtefan/eqvae-full-foreground-04",
    "maximusshtefan/eqvae-full-foreground-05",
    "maximusshtefan/eqvae-full-foreground-06",
    "maximusshtefan/eqvae-full-foreground-07",
    "maximusshtefan/eqvae-full-foreground-08",
)
DEVELOPMENT_FILES = {
    "dataset.json",
    "physical_parts.csv",
    "wsi_cancer_train_bags.csv",
    "wsi_cancer_train_instances.csv",
    "wsi_cancer_validation_bags.csv",
    "wsi_cancer_validation_instances.csv",
}
SESSION_LIMIT_SECONDS = 12 * 60 * 60
SESSION_SAVE_MARGIN_SECONDS = 15 * 60
MINIMUM_NEXT_HALF_SECONDS = 10 * 60
GPU_GRAPH_CACHE_BUDGET_BYTES = 512 * 1024 * 1024
ACCEPTED_MINIMUM_HEADROOM_BYTES = 4_776_067_072
BOOTSTRAP_REPLICATES = 10_000
COMPILE_DISABLE_ENVIRONMENT = ("TORCH_COMPILE_DISABLE", "TORCHDYNAMO_DISABLE")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(payload):
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _canonical_sha256(payload):
    return hashlib.sha256(_canonical_json(_json_safe(payload))).hexdigest()


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(_canonical_json(_json_safe(payload)) + b"\n")
    temporary.replace(path)


def _json_safe(value):
    if hasattr(value, "detach"):
        tensor = value.detach().cpu()
        return tensor.item() if tensor.numel() == 1 else tensor.tolist()
    if hasattr(value, "__dataclass_fields__"):
        return {key: _json_safe(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": repr(value)}
    return value


def _read_object(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return payload


def _resolve_input_bundle():
    matches = [
        path
        for path in INPUT_ROOT.rglob(INPUT_CONTRACT_NAME)
        if path.is_file() and _sha256(path) == INPUT_CONTRACT_SHA256
    ]
    if len(matches) != 1:
        raise RuntimeError("Expected one exact Spec 0036 input contract")
    contract_path = matches[0]
    root = contract_path.parent
    contract = _read_object(contract_path)
    if (
        contract.get("schema_version") != "spec0036.mil_training_input.v1"
        or contract.get("scope") != "development_training_only_no_sealed_test"
        or contract.get("dataset_reference") != INPUT_DATASET_REFERENCE
        or contract.get("development_contract_sha256") != DEVELOPMENT_CONTRACT_SHA256
        or contract.get("kernel_sources") != list(KERNEL_SOURCES)
    ):
        raise RuntimeError("Spec 0036 mounted input identity differs")
    files = contract.get("files")
    if not isinstance(files, dict):
        raise RuntimeError("Spec 0036 file manifest is missing")
    observed = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    if observed != {*files, INPUT_CONTRACT_NAME}:
        raise RuntimeError("Spec 0036 mounted input allow-list differs")
    for name, record in files.items():
        if not isinstance(record, dict):
            raise RuntimeError("Spec 0036 file record is malformed")
        path = root / name
        if path.stat().st_size != record.get("bytes") or _sha256(path) != record.get(
            "sha256",
        ):
            raise RuntimeError(f"Spec 0036 mounted bytes differ: {name}")
    development = root / "development"
    if {path.name for path in development.iterdir()} != DEVELOPMENT_FILES or any(
        not path.is_file() or path.is_symlink() for path in development.iterdir()
    ):
        raise RuntimeError("Spec 0036 development-only allow-list differs")
    if any(
        "sealed" in path.name or "test" in path.name for path in development.iterdir()
    ):
        raise RuntimeError("Spec 0036 input exposes sealed-test logical data")
    declared_development = contract.get("development_files")
    if (
        not isinstance(declared_development, dict)
        or set(declared_development) != DEVELOPMENT_FILES
    ):
        raise RuntimeError("Spec 0036 development manifest differs")
    for name, record in declared_development.items():
        path = development / name
        if path.stat().st_size != record.get("bytes") or _sha256(path) != record.get(
            "sha256",
        ):
            raise RuntimeError(f"Spec 0036 development bytes differ: {name}")
    initial = contract.get("initial_state")
    if not isinstance(initial, dict):
        raise RuntimeError("Spec 0036 initial-state contract is missing")
    initial_path = root / str(initial.get("path"))
    if initial_path.stat().st_size != files[initial_path.relative_to(root).as_posix()][
        "bytes"
    ] or _sha256(initial_path) != initial.get("file_sha256"):
        raise RuntimeError("Spec 0036 initial-state file differs")
    return root, contract, contract_path


def _catalog_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 30:
        raise RuntimeError("Spec 0036 physical catalog must contain 30 rows")
    sources = tuple(dict.fromkeys(row["kaggle_source"] for row in rows))
    if sources != KERNEL_SOURCES:
        raise RuntimeError("Spec 0036 physical source order differs")
    return rows


def _resolve_source_roots(development, contract):
    rows = _catalog_rows(development / "physical_parts.csv")
    declared = contract.get("physical_sources")
    if not isinstance(declared, list) or rows != declared:
        raise RuntimeError("Spec 0036 physical catalog contract differs")
    roots = {}
    for row in rows:
        matches = [
            path.parent
            for path in INPUT_ROOT.rglob(row["sidecar_name"])
            if path.is_file()
            and path.stat().st_size == int(row["sidecar_bytes"])
            and _sha256(path) == row["sidecar_sha256"]
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one sidecar-hash match for {row['model_name']} part {row['part']}",
            )
        root = matches[0]
        binary = root / row["binary_name"]
        if not binary.is_file() or binary.stat().st_size != int(row["binary_bytes"]):
            raise RuntimeError("Spec 0036 mounted physical binary differs")
        source = row["kaggle_source"]
        if source in roots and roots[source] != root:
            raise RuntimeError("One physical producer resolved to multiple roots")
        roots[source] = root
    if set(roots) != set(KERNEL_SOURCES):
        raise RuntimeError("Spec 0036 mounted physical producer set differs")
    return roots


def _resolve_optional_resume(input_contract_sha256):
    candidates = tuple(INPUT_ROOT.rglob(RESUME_CONTRACT_NAME))
    if not RESUME_DATASET_REFERENCE and not RESUME_CONTRACT_SHA256:
        if candidates:
            raise RuntimeError("Fresh Spec 0036 kernel refuses an unbound resume")
        return None
    if not RESUME_DATASET_REFERENCE or not RESUME_CONTRACT_SHA256:
        raise RuntimeError("Spec 0036 resume binding is incomplete")
    if len(candidates) != 1:
        raise RuntimeError("Expected exactly one externally bound resume contract")
    unresolved_path = candidates[0]
    if unresolved_path.is_symlink():
        raise RuntimeError("Spec 0036 resume contract may not be a symlink")
    path = unresolved_path.resolve()
    input_root = INPUT_ROOT.resolve()
    root = path.parent
    if (
        not path.is_file()
        or not root.is_relative_to(input_root)
        or _sha256(path) != RESUME_CONTRACT_SHA256
    ):
        raise RuntimeError("Expected exact externally bound Spec 0036 resume contract")
    contract = _read_object(path)
    branches = contract.get("branches")
    carried_branches = contract.get("carried_branches")
    if (
        contract.get("schema_version") != "spec0036.mil_training_resume.v1"
        or contract.get("dataset_reference") != RESUME_DATASET_REFERENCE
        or contract.get("input_contract_sha256") != input_contract_sha256
        or not isinstance(branches, dict)
        or not branches
        or not isinstance(carried_branches, dict)
        or set(branches) | set(carried_branches) != set(BRANCH_DEVICES)
        or set(branches) & set(carried_branches)
    ):
        raise RuntimeError("Spec 0036 resume contract identity differs")
    files = contract.get("files")
    if not isinstance(files, dict):
        raise RuntimeError("Spec 0036 resume file manifest is missing")
    entries = tuple(root.rglob("*"))
    if any(item.is_symlink() for item in entries):
        raise RuntimeError("Spec 0036 resume mount contains a symlink")
    observed = {item.relative_to(root).as_posix() for item in entries if item.is_file()}
    if observed != {*files, RESUME_CONTRACT_NAME}:
        raise RuntimeError("Spec 0036 resume allow-list differs")
    for name, record in files.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("Spec 0036 resume file path escapes its mount")
        unresolved_item = root / relative
        item = unresolved_item.resolve()
        if (
            not isinstance(record, dict)
            or not item.is_relative_to(root)
            or unresolved_item.is_symlink()
            or item.stat().st_size != record.get("bytes")
            or _sha256(item) != record.get("sha256")
        ):
            raise RuntimeError(f"Spec 0036 resume bytes differ: {name}")
    branch_roots = {}
    for branch, relative in branches.items():
        relative_path = Path(str(relative))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise RuntimeError("Spec 0036 resume branch path escapes its mount")
        unresolved_branch_root = root / relative_path
        branch_root = unresolved_branch_root.resolve()
        if (
            not branch_root.is_relative_to(root)
            or not branch_root.is_dir()
            or unresolved_branch_root.is_symlink()
            or not (branch_root / "latest").is_file()
        ):
            raise RuntimeError(f"Spec 0036 resume branch is incomplete: {branch}")
        branch_roots[branch] = branch_root
    carried_summaries = {}
    for branch, relative in carried_branches.items():
        relative_path = Path(str(relative))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise RuntimeError("Spec 0036 carried branch path escapes its mount")
        unresolved_summary = root / relative_path
        summary_path = unresolved_summary.resolve()
        if (
            unresolved_summary.is_symlink()
            or not summary_path.is_relative_to(root)
            or not summary_path.is_file()
        ):
            raise RuntimeError(f"Spec 0036 carried branch is incomplete: {branch}")
        summary = _read_object(summary_path)
        if summary.get("branch") != branch or summary.get("status") != "failed":
            raise RuntimeError(f"Spec 0036 carried branch evidence differs: {branch}")
        carried_summaries[branch] = summary_path
    return {
        "contract": contract,
        "path": path,
        "branch_roots": branch_roots,
        "carried_summaries": carried_summaries,
    }


def install_pinned_torch():
    if not WORKING_ROOT.is_dir():
        return
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--no-cache-dir",
        f"torch=={PINNED_TORCH_VERSION}",
        "--index-url",
        PINNED_TORCH_INDEX,
    ])


def _driver_versions():
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise RuntimeError("Spec 0036 cannot resolve nvidia-smi for driver identity")
    output = subprocess.check_output(
        [
            executable,
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ],
        text=True,
    )
    versions = [line.strip() for line in output.splitlines() if line.strip()]
    if len(versions) != 2:
        raise RuntimeError("Spec 0036 requires two reported NVIDIA driver versions")
    return versions


def _validate_runtime(torch, *, driver_versions=None):
    version = str(torch.__version__).split("+", maxsplit=1)[0]
    if version != PINNED_TORCH_VERSION or torch.version.cuda != PINNED_TORCH_CUDA:
        raise RuntimeError(
            f"Spec 0036 runtime differs: torch={torch.__version__}, cuda={torch.version.cuda}",
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("Spec 0036 requires exactly two CUDA devices")
    names = [torch.cuda.get_device_name(index) for index in range(2)]
    if any("T4" not in name for name in names):
        raise RuntimeError(f"Spec 0036 requires exactly two T4 GPUs, found {names!r}")
    drivers = list(_driver_versions() if driver_versions is None else driver_versions)
    if len(drivers) != 2 or any(
        not isinstance(value, str) or not value for value in drivers
    ):
        raise RuntimeError("Spec 0036 NVIDIA driver fingerprint is incomplete")
    return {
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "devices": names,
        "capabilities": [
            list(torch.cuda.get_device_capability(index)) for index in range(2)
        ],
        "driver_versions": drivers,
    }


def _state_dict_sha256(state):
    digest = hashlib.sha256(INITIAL_STATE_HASH_SCHEMA)
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        metadata = json.dumps(
            {"dtype": str(tensor.dtype), "name": name, "shape": list(tensor.shape)},
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        payload = tensor.numpy().tobytes(order="C")
        digest.update(struct.pack("<Q", len(metadata)))
        digest.update(metadata)
        digest.update(struct.pack("<Q", len(payload)))
        digest.update(payload)
    return digest.hexdigest()


def _validate_initial_state(torch, root, contract):
    initial = contract["initial_state"]
    path = root / initial["path"]
    state = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(state, dict)
        or _state_dict_sha256(state) != initial["state_sha256"]
    ):
        raise RuntimeError("Spec 0036 initial model state identity differs")
    if sum(tensor.numel() for tensor in state.values()) != initial["parameter_count"]:
        raise RuntimeError("Spec 0036 initial-state tensor count differs")
    return path


def _contract_hashes(contract, runtime):
    return {
        "input_contract_sha256": INPUT_CONTRACT_SHA256,
        "spec_sha256": contract["spec_sha256"],
        "development_contract_sha256": contract["development_contract_sha256"],
        "training_config_sha256": contract["training_config_sha256"],
        "initial_state_file_sha256": contract["initial_state"]["file_sha256"],
        "initial_state_sha256": contract["initial_state"]["state_sha256"],
        "model_sha256": contract["model"]["sha256"],
        "candidate_sha256": contract["candidate"]["sha256"],
        "runtime_sha256": hashlib.sha256(_canonical_json(runtime)).hexdigest(),
    }


def _configure_torch(torch):
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)
    torch.set_float32_matmul_precision("high")


def _reject_compile_disable_environment():
    disabled = {
        name: value
        for name in COMPILE_DISABLE_ENVIRONMENT
        if (value := os.environ.get(name, "")).strip().lower()
        not in {"", "0", "false", "no", "off"}
    }
    if disabled:
        raise RuntimeError(
            f"Spec 0036 refuses compile-disabled execution: {sorted(disabled)!r}",
        )


def _make_model(torch, initial_state_path, device):
    from eqvae.models.local_attention_candidates import (
        WholeBagFixed25Attention,
        use_whole_bag_fixed25_attention,
    )
    from eqvae.models.local_global_mil import (
        EXPECTED_PARAMETER_COUNT,
        LocalGlobalMILClassifier,
    )

    model = LocalGlobalMILClassifier()
    state = torch.load(initial_state_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    before = _state_dict_sha256(model.state_dict())
    use_whole_bag_fixed25_attention(model)
    if before != _state_dict_sha256(model.state_dict()):
        raise RuntimeError("WholeBagFixed25Attention changed the staged model state")
    if not all(
        isinstance(block.attention, WholeBagFixed25Attention)
        for block in model.local_blocks
    ):
        raise RuntimeError("Both local blocks require WholeBagFixed25Attention")
    if (
        sum(parameter.numel() for parameter in model.parameters())
        != EXPECTED_PARAMETER_COUNT
    ):
        raise RuntimeError("Spec 0036 model parameter count differs")
    return model.to(device=device, memory_format=torch.channels_last).train()


def _make_numerical(torch, model):
    def numerical(latents, graph, target):
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(latents, graph)
            loss = torch.nn.functional.cross_entropy(
                logits.float().unsqueeze(0),
                target,
            )
        return loss, logits

    return torch.compile(
        numerical,
        backend="inductor",
        fullgraph=True,
        dynamic=None,
        mode="max-autotune-no-cudagraphs",
        recompile_limit=3,
        isolate_recompiles=True,
    )


def _compiler_counts(torch):
    counters = torch._dynamo.utils.counters
    return {
        "unique_graphs": int(counters["stats"]["unique_graphs"]),
        "graph_breaks": int(sum(counters["graph_break"].values())),
        "generated_kernels": int(
            getattr(torch._inductor.metrics, "generated_kernel_count", -1),
        ),
    }


def _exception_record(error):
    record = {
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
    }
    details = getattr(error, "details", None)
    if isinstance(details, dict):
        record["error_details"] = dict(details)
    return record


def _build_graph_cache(dataset, expected, build_graph):
    graphs = {}
    summaries = {}
    observed = set()
    for bag in dataset.bags:
        stop = bag.instance_start + bag.instance_count
        instances = dataset.instances[bag.instance_start : stop]
        graph = build_graph(instances, expected_instance_count=bag.instance_count)
        key = str(bag.wsi_id)
        if key in observed or graph.identity_sha256 != expected.get(key):
            raise RuntimeError(f"Spec 0036 graph identity differs for WSI {key}")
        observed.add(key)
        degrees = graph.neighbor_valid.sum(dim=1)
        graphs[bag.wsi_id] = graph
        summaries[bag.wsi_id] = {
            "minimum": int(degrees.min().item()),
            "maximum": int(degrees.max().item()),
            "mean": float(degrees.float().mean().item()),
        }
    if observed != set(expected):
        raise RuntimeError("Spec 0036 graph identity coverage differs")
    return graphs, summaries


def _graph_cache_bytes(*graph_caches):
    return sum(
        tensor.numel() * tensor.element_size()
        for cache in graph_caches
        for graph in cache.values()
        for tensor in (
            graph.neighbor_index,
            graph.neighbor_valid,
            graph.radial_code,
        )
    )


def _load_bag(torch, dataset, index, cpu_graphs, gpu_graphs, degree_summaries, device):
    bag = dataset.bags[index]
    graph = gpu_graphs.get(bag.wsi_id)
    graph_transfer_started = time.perf_counter()
    if graph is None:
        graph = cpu_graphs[bag.wsi_id].to(device)
        gpu_graphs[bag.wsi_id] = graph
    graph_transfer_seconds = time.perf_counter() - graph_transfer_started
    stop = bag.instance_start + bag.instance_count
    instances = dataset.instances[bag.instance_start : stop]
    load_started = time.perf_counter()
    latents, reads = dataset.store.read_rows(
        tuple(instance.pointer for instance in instances),
    )
    load_seconds = time.perf_counter() - load_started
    transfer_started = time.perf_counter()
    latents = latents.to(
        device=device,
        dtype=torch.float16,
        memory_format=torch.channels_last,
    )
    target = torch.tensor([bag.diagnosis_index], device=device, dtype=torch.long)
    transfer_seconds = time.perf_counter() - transfer_started
    if latents.shape[0] != graph.node_count or len(reads) != bag.instance_count:
        raise RuntimeError("Spec 0036 complete-bag load differs")
    access = {
        "wsi_id": bag.wsi_id,
        "diagnosis": bag.diagnosis_label,
        "bag_size": bag.instance_count,
        "parts": dict(Counter(str(read.part) for read in reads)),
        "load_seconds": load_seconds,
        "transfer_seconds": transfer_seconds,
        "graph_transfer_seconds": graph_transfer_seconds,
        "graph_identity": graph.identity_sha256,
        "graph_degree": degree_summaries[bag.wsi_id],
    }
    return bag, latents, graph, target, access


def _hint_dynamic(torch, latents, graph):
    torch._dynamo.maybe_mark_dynamic(latents, 0)
    for tensor in (graph.neighbor_index, graph.neighbor_valid, graph.radial_code):
        torch._dynamo.maybe_mark_dynamic(tensor, 0)


def _validate_split(torch, model, dataset, cpu_graphs, gpu_graphs, summaries, device):
    truths = []
    predictions = []
    losses = []
    rows = []
    access_rows = []
    started = time.perf_counter()
    with torch.no_grad():
        for index in range(len(dataset)):
            bag, latents, graph, target, access = _load_bag(
                torch,
                dataset,
                index,
                cpu_graphs,
                gpu_graphs,
                summaries,
                device,
            )
            compute_started = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(latents, graph)
                loss = torch.nn.functional.cross_entropy(
                    logits.float().unsqueeze(0),
                    target,
                )
            logits_cpu = logits.detach().float().cpu().tolist()
            loss_value = float(loss.detach().item())
            prediction = int(max(range(len(logits_cpu)), key=logits_cpu.__getitem__))
            truths.append(bag.diagnosis_index)
            predictions.append(prediction)
            losses.append(loss_value)
            rows.append({
                "wsi_id": bag.wsi_id,
                "truth": bag.diagnosis_index,
                "prediction": prediction,
                "logits": logits_cpu,
                "cross_entropy": loss_value,
                "bag_size": bag.instance_count,
                "graph_identity": graph.identity_sha256,
                "graph_degree": summaries[bag.wsi_id],
            })
            access["compute_seconds"] = time.perf_counter() - compute_started
            access_rows.append(access)
    return truths, predictions, losses, rows, access_rows, time.perf_counter() - started


def _best_revalidation_comparison(
    *,
    best_boundary,
    selection_metrics,
    validation_history,
    observed_rows,
    observed_metrics,
):
    selected = [
        row for row in validation_history if int(row["boundary"]) == best_boundary
    ]
    mismatches = []
    if len(selected) != 1:
        mismatches.append("selected_boundary_history")
        stored_rows = []
    else:
        stored_rows = selected[0].get("predictions", [])
    stable_fields = (
        "wsi_id",
        "truth",
        "prediction",
        "bag_size",
        "graph_identity",
    )
    if len(stored_rows) != len(observed_rows):
        mismatches.append("row_count")
    stored_ids = [row.get("wsi_id") for row in stored_rows]
    observed_ids = [row.get("wsi_id") for row in observed_rows]
    unique_wsi_rows = (
        len(set(stored_ids)) == len(stored_ids)
        and len(set(observed_ids)) == len(observed_ids)
        and stored_ids == observed_ids
    )
    if not unique_wsi_rows:
        mismatches.append("wsi_identity_or_order")
    stable_rows_match = len(stored_rows) == len(observed_rows) and all(
        all(stored.get(field) == observed.get(field) for field in stable_fields)
        for stored, observed in zip(stored_rows, observed_rows, strict=True)
    )
    if not stable_rows_match:
        mismatches.append("wsi_truth_prediction_or_graph")
    discrete_fields = (
        "macro_f1",
        "balanced_accuracy",
        "accuracy",
        "per_class",
        "confusion_matrix",
        "wsi_count",
    )
    discrete_metrics_match = all(
        selection_metrics.get(field) == observed_metrics.get(field)
        for field in discrete_fields
    )
    if not discrete_metrics_match:
        mismatches.append("discrete_metrics")
    finite_outputs = (
        bool(observed_rows)
        and all(
            bool(row.get("logits", []))
            and math.isfinite(float(row.get("cross_entropy", float("nan"))))
            and all(math.isfinite(float(value)) for value in row.get("logits", []))
            for row in observed_rows
        )
        and math.isfinite(float(observed_metrics.get("mean_ce", float("nan"))))
    )
    if not finite_outputs:
        mismatches.append("nonfinite_output")
    argmax_predictions_match = finite_outputs and all(
        int(row["prediction"])
        == max(
            range(len(row["logits"])),
            key=lambda index: float(row["logits"][index]),
        )
        for row in observed_rows
    )
    if not argmax_predictions_match:
        mismatches.append("prediction_is_not_logit_argmax")
    selected_primary = float(selection_metrics["macro_f1"])
    competitors = [
        (int(row["boundary"]), float(row["macro_f1"]))
        for row in validation_history
        if int(row["boundary"]) != best_boundary
    ]
    runner_up = max(competitors, key=itemgetter(1), default=None)
    selected_history_matches = len(selected) == 1 and _canonical_sha256(
        selected[0]["macro_f1"],
    ) == _canonical_sha256(selection_metrics["macro_f1"])
    unique_primary = selected_history_matches and all(
        selected_primary > value for _, value in competitors
    )
    if not unique_primary:
        mismatches.append("primary_metric_not_unique")
    prediction_changes = [
        observed.get("wsi_id")
        for stored, observed in zip(stored_rows, observed_rows, strict=False)
        if stored.get("prediction") != observed.get("prediction")
    ]
    row_diagnostics = []
    finite_logit_deltas = []
    finite_ce_deltas = []
    for stored, observed in zip(stored_rows, observed_rows, strict=False):
        stored_logits = [float(value) for value in stored.get("logits", [])]
        observed_logits = [float(value) for value in observed.get("logits", [])]
        deltas = [
            abs(new - old)
            for old, new in zip(stored_logits, observed_logits, strict=False)
            if math.isfinite(old) and math.isfinite(new)
        ]
        finite_logit_deltas.extend(deltas)
        stored_ce = float(stored.get("cross_entropy", float("nan")))
        observed_ce = float(observed.get("cross_entropy", float("nan")))
        ce_delta = observed_ce - stored_ce
        if math.isfinite(ce_delta):
            finite_ce_deltas.append(ce_delta)

        def top_margin(values):
            if len(values) < 2 or any(not math.isfinite(value) for value in values):
                return None
            top = sorted(values, reverse=True)
            return top[0] - top[1]

        row_diagnostics.append({
            "wsi_id": observed.get("wsi_id"),
            "stored_prediction": stored.get("prediction"),
            "observed_prediction": observed.get("prediction"),
            "stored_top1_top2_margin": top_margin(stored_logits),
            "observed_top1_top2_margin": top_margin(observed_logits),
            "cross_entropy_delta": ce_delta,
            "maximum_absolute_logit_delta": max(deltas, default=None),
        })
    return {
        "schema_version": "spec0036.best_revalidation_comparison.v1",
        "status": "passed" if not mismatches else "failed",
        "mismatches": mismatches,
        "stable_rows_match": stable_rows_match,
        "unique_wsi_rows_in_same_order": unique_wsi_rows,
        "discrete_metrics_match": discrete_metrics_match,
        "finite_outputs": finite_outputs,
        "argmax_predictions_match": argmax_predictions_match,
        "selected_primary_macro_f1_is_unique": unique_primary,
        "selected_primary_macro_f1": selected_primary,
        "runner_up_boundary": None if runner_up is None else runner_up[0],
        "runner_up_macro_f1": None if runner_up is None else runner_up[1],
        "primary_macro_f1_gap": (
            None if runner_up is None else selected_primary - runner_up[1]
        ),
        "prediction_changes": prediction_changes,
        "selection_mean_ce": float(selection_metrics["mean_ce"]),
        "observed_mean_ce": float(observed_metrics["mean_ce"]),
        "mean_ce_delta": float(observed_metrics["mean_ce"])
        - float(selection_metrics["mean_ce"]),
        "maximum_absolute_wsi_ce_delta": max(
            (abs(value) for value in finite_ce_deltas),
            default=0.0,
        ),
        "maximum_absolute_logit_delta": max(finite_logit_deltas, default=0.0),
        "rows": row_diagnostics,
        "stored_prediction_rows_sha256": _canonical_sha256(stored_rows),
        "observed_prediction_rows_sha256": _canonical_sha256(observed_rows),
    }


def _checkpoint_payload(
    training,
    *,
    branch,
    model,
    optimizer,
    scaler,
    committed,
    epoch,
    cursor,
    order,
    train_history,
    validation_history,
    selection,
    contract_hashes,
    access_transcript,
):
    return training.branch_checkpoint_payload(
        branch_name=branch,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        committed_update=committed,
        epoch=epoch,
        within_epoch_cursor=cursor,
        current_order=order,
        train_history=train_history,
        validation_history=validation_history,
        best_selection=selection,
        contract_hashes=contract_hashes,
        access_transcript=access_transcript,
    )


def _write_branch_checkpoint(training, checkpoints, *, slot, payload):
    return training.save_branch_checkpoint(
        checkpoints,
        slot=slot,
        payload=payload,
    )


def _prune_checkpoint_objects(checkpoints_root):
    referenced = set()
    for slot in ("latest", "best", "final"):
        pointer = checkpoints_root / slot
        if pointer.is_file():
            manifest = _read_object(pointer)["manifest"]
            referenced.add(Path(manifest).parts[1])
    objects = checkpoints_root / ".objects"
    if objects.is_dir():
        for path in objects.iterdir():
            if path.is_dir() and path.name not in referenced:
                shutil.rmtree(path)


def _write_metric_tables(branch_root, train_history, validation_history):
    metrics = branch_root / "metrics"
    metrics.mkdir(exist_ok=True)
    train_fields = (
        "committed_update",
        "epoch",
        "within_epoch_cursor",
        "wsi_id",
        "diagnosis",
        "bag_size",
        "learning_rate",
        "unweighted_loss",
        "weighted_loss",
        "attempts",
        "scale_backoffs",
        "initial_scale",
        "final_scale",
        "load_seconds",
        "transfer_seconds",
        "compute_seconds",
    )
    validation_fields = (
        "boundary",
        "epoch",
        "within_epoch_cursor",
        "macro_f1",
        "balanced_accuracy",
        "accuracy",
        "mean_ce",
        "wsi_count",
        "elapsed_seconds",
        "improved",
        "non_improving_checks",
    )
    for path, fields, rows in (
        (metrics / "train.csv", train_fields, train_history),
        (metrics / "validation.csv", validation_fields, validation_history),
    ):
        temporary = path.with_name(f".{path.name}.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)


def _copy_resume_tree(resume_root, destination):
    if destination.exists():
        raise RuntimeError("Spec 0036 output checkpoint tree already exists")
    shutil.copytree(resume_root, destination)


def _resume_branch(
    training,
    resume_root,
    checkpoints,
    branch,
    hashes,
    model,
    optimizer,
    scaler,
):
    if resume_root is None:
        return (
            0,
            0,
            0,
            training.training_epoch_order(0),
            [],
            [],
            training.BestSelectionState(),
            [],
        )
    _copy_resume_tree(resume_root, checkpoints)
    loaded = training.load_branch_checkpoint(
        checkpoints,
        slot="latest",
        expected_branch_name=branch,
        expected_contract_hashes=hashes,
    )
    committed, epoch, cursor = training.restore_branch_checkpoint(
        loaded,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
    )
    payload = loaded.payload
    return (
        committed,
        epoch,
        cursor,
        tuple(payload["current_order"]),
        [dict(row) for row in payload["train_history"]],
        [dict(row) for row in payload["validation_history"]],
        training.selection_from_checkpoint_payload(payload["best_selection"]),
        [dict(row) for row in payload["access_transcript"]],
    )


def _prevalidate_resume(training, resume, hashes):
    if resume is None:
        return
    for branch, branch_root in resume["branch_roots"].items():
        latest = training.load_branch_checkpoint(
            branch_root,
            slot="latest",
            expected_branch_name=branch,
            expected_contract_hashes=hashes,
        )
        best = training.load_branch_checkpoint(
            branch_root,
            slot="best",
            expected_branch_name=branch,
            expected_contract_hashes=hashes,
        )
        selection = training.selection_from_checkpoint_payload(
            latest.payload["best_selection"],
        )
        best_selection = training.selection_from_checkpoint_payload(
            best.payload["best_selection"],
        )
        if (
            best.payload["committed_update"] != selection.best_boundary
            or best_selection.best_metrics != selection.best_metrics
        ):
            raise RuntimeError(f"Spec 0036 resume best checkpoint differs: {branch}")
        completed = (
            latest.payload["committed_update"] == training.TOTAL_UPDATES
            or selection.stopped
        )
        if completed:
            final = training.load_branch_checkpoint(
                branch_root,
                slot="final",
                expected_branch_name=branch,
                expected_contract_hashes=hashes,
            )
            if (
                final.payload["committed_update"] != latest.payload["committed_update"]
                or training.selection_from_checkpoint_payload(
                    final.payload["best_selection"],
                )
                != selection
            ):
                raise RuntimeError(
                    f"Spec 0036 resume final checkpoint differs: {branch}",
                )


def _should_pause(session_started_unix, half_durations):
    elapsed = time.time() - session_started_unix
    estimate = (
        max(
            MINIMUM_NEXT_HALF_SECONDS,
            sum(half_durations[-4:]) / len(half_durations[-4:]),
        )
        if half_durations
        else MINIMUM_NEXT_HALF_SECONDS
    )
    return (
        elapsed + 1.25 * estimate + SESSION_SAVE_MARGIN_SECONDS >= SESSION_LIMIT_SECONDS
    )


def _run_branch(args):
    import torch

    _reject_compile_disable_environment()
    _configure_torch(torch)
    if torch.cuda.device_count() != 1 or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("A Spec 0036 child must see exactly its assigned T4")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    root = Path(args.bundle_root)
    contract = _read_object(Path(args.contract_path))
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root / "src"))
    from eqvae.data.full_foreground_latents import FullForegroundBagDataset
    from eqvae.models.local_global_mil import build_local_attention_graph
    from eqvae.training import mil_training as training

    branch = args.worker
    branch_root = Path(args.branch_root)
    branch_root.mkdir(parents=True, exist_ok=False)
    result_path = branch_root / "final_summary.json"
    result = {
        "schema_version": "spec0036.branch_summary.v1",
        "branch": branch,
        "status": "failed",
        "started_unix": time.time(),
        "device": torch.cuda.get_device_name(0),
    }
    train_dataset = None
    validation_dataset = None
    try:
        source_roots = {
            name: Path(path) for name, path in json.loads(args.source_roots).items()
        }
        development = root / "development"
        graph_contract = contract["graph_identities"]
        train_dataset = FullForegroundBagDataset(
            development_root=development,
            expected_contract_sha256=DEVELOPMENT_CONTRACT_SHA256,
            split="train",
            model_name=branch,
            source_roots=source_roots,
        )
        validation_dataset = FullForegroundBagDataset(
            development_root=development,
            expected_contract_sha256=DEVELOPMENT_CONTRACT_SHA256,
            split="validation",
            model_name=branch,
            source_roots=source_roots,
        )
        if (
            len(train_dataset) != training.TRAIN_WSI_COUNT
            or len(validation_dataset) != training.VALIDATION_WSI_COUNT
        ):
            raise RuntimeError("Spec 0036 split WSI counts differ")
        train_graphs, train_degrees = _build_graph_cache(
            train_dataset,
            graph_contract["train"],
            build_local_attention_graph,
        )
        validation_graphs, validation_degrees = _build_graph_cache(
            validation_dataset,
            graph_contract["validation"],
            build_local_attention_graph,
        )
        graph_cache_bytes = _graph_cache_bytes(train_graphs, validation_graphs)
        if graph_cache_bytes > min(
            GPU_GRAPH_CACHE_BUDGET_BYTES,
            ACCEPTED_MINIMUM_HEADROOM_BYTES,
        ):
            raise RuntimeError(
                "Spec 0036 authenticated GPU graph cache exceeds its 512 MiB budget",
            )
        result["gpu_graph_cache"] = {
            "policy": "lazy_all_wsi_reuse_no_epoch_rebuild",
            "projected_bytes": graph_cache_bytes,
            "budget_bytes": GPU_GRAPH_CACHE_BUDGET_BYTES,
            "accepted_minimum_headroom_bytes": ACCEPTED_MINIMUM_HEADROOM_BYTES,
        }
        train_gpu_graphs = {}
        validation_gpu_graphs = {}
        model = _make_model(torch, Path(args.initial_state), device)
        numerical = _make_numerical(torch, model)
        scaler = training.make_default_grad_scaler()
        hashes = json.loads(args.contract_hashes)
        optimizer = training.make_fused_adamw(model)
        resume_root = Path(args.resume_root) if args.resume_root else None
        (
            committed,
            epoch,
            cursor,
            order,
            train_history,
            validation_history,
            selection,
            access_transcript,
        ) = _resume_branch(
            training,
            resume_root,
            branch_root / "checkpoints",
            branch,
            hashes,
            model,
            optimizer,
            scaler,
        )
        resumed = resume_root is not None
        if not resumed:
            by_wsi = {bag.wsi_id: index for index, bag in enumerate(train_dataset.bags)}
            if set(training.CALIBRATION_WSI_IDS) - set(by_wsi):
                raise RuntimeError("Spec 0036 calibration WSI is absent from training")
            calibration_cache = {}
            calibration_access = []

            def calibration_loss(wsi_id):
                if calibration_cache.get("wsi_id") != wsi_id:
                    calibration_cache.clear()
                    loaded = _load_bag(
                        torch,
                        train_dataset,
                        by_wsi[wsi_id],
                        train_graphs,
                        train_gpu_graphs,
                        train_degrees,
                        device,
                    )
                    loaded[-1].update({"phase": "calibration"})
                    calibration_access.append(loaded[-1])
                    calibration_cache.update({"wsi_id": wsi_id, "loaded": loaded})
                _, latents, graph, target, _ = calibration_cache["loaded"]
                _hint_dynamic(torch, latents, graph)
                loss, _ = numerical(latents, graph, target)
                return loss

            class_weights_by_wsi = {
                bag.wsi_id: training.CLASS_WEIGHTS[bag.diagnosis_index]
                for bag in train_dataset.bags
                if bag.wsi_id in training.CALIBRATION_WSI_IDS
            }
            calibration = training.calibrate_grad_scaler(
                model,
                scaler,
                lambda: training.make_fused_adamw(model),
                calibration_loss,
                class_weights_by_wsi=class_weights_by_wsi,
            )
            _write_json(
                branch_root / "calibration.json",
                {
                    "schema_version": "spec0036.calibration.v1",
                    "branch": branch,
                    "cases": calibration.cases,
                    "final_scaler": scaler.state_dict(),
                    "compiler": _compiler_counts(torch),
                    "access": calibration_access,
                },
            )
            access_transcript.extend(calibration_access)
            calibration_cache.clear()
            optimizer = training.make_fused_adamw(model)
        else:
            _write_json(
                branch_root / "calibration.json",
                {
                    "schema_version": "spec0036.calibration.v1",
                    "branch": branch,
                    "status": "restored_from_authenticated_checkpoint",
                    "scaler": scaler.state_dict(),
                },
            )

        checkpoints = branch_root / "checkpoints"
        half_started = time.perf_counter()
        half_durations = []
        validation_records = [
            prediction
            for row in validation_history
            for prediction in row.get("predictions", [])
        ]
        already_terminal = resumed and (
            selection.stopped or committed == training.TOTAL_UPDATES
        )
        terminal_reason = "early_stopping" if selection.stopped else "maximum_epochs"
        while committed < training.TOTAL_UPDATES and not selection.stopped:
            index = order[cursor]
            bag, latents, graph, target, access = _load_bag(
                torch,
                train_dataset,
                index,
                train_graphs,
                train_gpu_graphs,
                train_degrees,
                device,
            )
            _hint_dynamic(torch, latents, graph)
            learning_rate = training.committed_update_learning_rate(committed + 1)
            compute_started = time.perf_counter()

            def loss_closure():
                loss, _ = numerical(latents, graph, target)
                return loss

            try:
                step = training.weighted_amp_step(
                    model,
                    optimizer,
                    scaler,
                    loss_closure,
                    class_weight=training.CLASS_WEIGHTS[bag.diagnosis_index],
                    learning_rate=learning_rate,
                )
            except training.BranchNumericalError as error:
                error.details.update({
                    "branch": branch,
                    "phase": "training",
                    "committed_update_before_attempt": committed,
                    "epoch": epoch,
                    "within_epoch_cursor": cursor,
                    "wsi_id": bag.wsi_id,
                    "diagnosis": bag.diagnosis_label,
                    "bag_size": bag.instance_count,
                    "scaler_scale": float(scaler.get_scale()),
                })
                raise
            compute_seconds = time.perf_counter() - compute_started
            committed += 1
            cursor += 1
            train_history.append({
                "committed_update": committed,
                "epoch": epoch,
                "within_epoch_cursor": cursor,
                "wsi_id": bag.wsi_id,
                "diagnosis": bag.diagnosis_label,
                "bag_size": bag.instance_count,
                **asdict(step),
                "load_seconds": access["load_seconds"],
                "transfer_seconds": access["transfer_seconds"],
                "compute_seconds": compute_seconds,
            })
            access.update({"phase": "train", "committed_update": committed})
            access_transcript.append(access)
            if not training.is_validation_boundary(committed):
                continue

            truths, predictions, losses, prediction_rows, validation_access, elapsed = (
                _validate_split(
                    torch,
                    model,
                    validation_dataset,
                    validation_graphs,
                    validation_gpu_graphs,
                    validation_degrees,
                    device,
                )
            )
            metrics = training.compute_validation_metrics(truths, predictions, losses)
            update = training.update_best_selection(
                selection,
                metrics,
                boundary=committed,
            )
            selection = update.state
            for row in prediction_rows:
                row["boundary"] = committed
                row["branch"] = branch
            validation_records.extend(prediction_rows)
            for access_row in validation_access:
                access_row.update({
                    "phase": "validation",
                    "committed_update": committed,
                })
            access_transcript.extend(validation_access)
            validation_history.append({
                "boundary": committed,
                "epoch": epoch,
                "within_epoch_cursor": cursor,
                **metrics.to_dict(),
                "elapsed_seconds": elapsed,
                "improved": update.improved,
                "non_improving_checks": selection.non_improving_checks,
                "predictions": prediction_rows,
            })
            if cursor == training.TRAIN_WSI_COUNT:
                epoch += 1
                cursor = 0
                order = training.training_epoch_order(epoch)
            payload = _checkpoint_payload(
                training,
                branch=branch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                committed=committed,
                epoch=epoch,
                cursor=cursor,
                order=order,
                train_history=train_history,
                validation_history=validation_history,
                selection=selection,
                contract_hashes=hashes,
                access_transcript=access_transcript,
            )
            latest = _write_branch_checkpoint(
                training,
                checkpoints,
                slot="latest",
                payload=payload,
            )
            if update.improved:
                _write_branch_checkpoint(
                    training,
                    checkpoints,
                    slot="best",
                    payload=payload,
                )
            _prune_checkpoint_objects(checkpoints)
            _write_metric_tables(branch_root, train_history, validation_history)
            _write_json(
                branch_root / "validation_predictions.json",
                {
                    "schema_version": "spec0036.validation_predictions.v1",
                    "rows": validation_records,
                },
            )
            half_durations.append(time.perf_counter() - half_started)
            half_started = time.perf_counter()
            _write_json(
                branch_root / "checkpoint_resume_proof.json",
                {
                    "schema_version": "spec0036.resume_proof.v1",
                    "branch": branch,
                    "committed_update": committed,
                    "checkpoint_sha256": latest.checkpoint_sha256,
                    "manifest_sha256": latest.manifest_sha256,
                    "contract_hashes": hashes,
                },
            )
            if selection.stopped:
                terminal_reason = "early_stopping"
                break
            if _should_pause(float(args.session_started_unix), half_durations):
                terminal_reason = "session_deadline"
                result["status"] = "paused"
                break

        if already_terminal:
            final_checkpoint = training.load_branch_checkpoint(
                checkpoints,
                slot="final",
                expected_branch_name=branch,
                expected_contract_hashes=hashes,
            )
        else:
            final_payload = _checkpoint_payload(
                training,
                branch=branch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                committed=committed,
                epoch=epoch,
                cursor=cursor,
                order=order,
                train_history=train_history,
                validation_history=validation_history,
                selection=selection,
                contract_hashes=hashes,
                access_transcript=access_transcript,
            )
            final_checkpoint = _write_branch_checkpoint(
                training,
                checkpoints,
                slot="final",
                payload=final_payload,
            )
            _prune_checkpoint_objects(checkpoints)
        _write_metric_tables(branch_root, train_history, validation_history)
        _write_json(
            branch_root / "validation_predictions.json",
            {
                "schema_version": "spec0036.validation_predictions.v1",
                "rows": validation_records,
            },
        )
        best = training.load_branch_checkpoint(
            checkpoints,
            slot="best",
            expected_branch_name=branch,
            expected_contract_hashes=hashes,
        )
        best_identity = {
            "branch": branch,
            "best_boundary": selection.best_boundary,
            "best_checkpoint_sha256": best.checkpoint_sha256,
            "best_manifest_sha256": best.manifest_sha256,
            "contract_hashes": hashes,
        }
        best_prediction_rows_sha256 = None
        best_revalidation_metrics = None
        best_revalidation_comparison = None
        if result["status"] != "paused":
            training.restore_branch_checkpoint(
                best,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
            )
            truths, predictions, losses, best_rows, best_access, elapsed = (
                _validate_split(
                    torch,
                    model,
                    validation_dataset,
                    validation_graphs,
                    validation_gpu_graphs,
                    validation_degrees,
                    device,
                )
            )
            selection_metrics = selection.best_metrics.to_dict()
            best_prediction_rows_sha256 = _canonical_sha256(best_rows)
            metric_error = None
            try:
                best_revalidation_metrics = training.compute_validation_metrics(
                    truths,
                    predictions,
                    losses,
                ).to_dict()
            except ValueError as error:
                best_revalidation_metrics = training.compute_validation_metrics(
                    truths,
                    predictions,
                    [0.0] * len(losses),
                ).to_dict()
                best_revalidation_metrics["mean_ce"] = float("nan")
                metric_error = {
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            best_revalidation_comparison = _best_revalidation_comparison(
                best_boundary=selection.best_boundary,
                selection_metrics=selection_metrics,
                validation_history=validation_history,
                observed_rows=best_rows,
                observed_metrics=best_revalidation_metrics,
            )
            best_revalidation_comparison["metric_error"] = metric_error
            result.update({
                "best_metrics": selection.best_metrics,
                "best_revalidation_metrics": best_revalidation_metrics,
                "best_revalidation_comparison": best_revalidation_comparison,
                **best_identity,
                "best_prediction_rows_sha256": best_prediction_rows_sha256,
            })
            _write_json(
                branch_root / "best_validation_predictions.json",
                {
                    "schema_version": "spec0036.best_validation_predictions.v1",
                    **best_identity,
                    "rows_sha256": best_prediction_rows_sha256,
                    "rows": best_rows,
                },
            )
            _write_json(
                branch_root / "best_revalidation.json",
                {
                    "schema_version": "spec0036.best_revalidation.v1",
                    **best_identity,
                    "prediction_rows_sha256": best_prediction_rows_sha256,
                    "phase": "revalidate_selected_best",
                    "selection_metrics": selection_metrics,
                    "metrics": best_revalidation_metrics,
                    "comparison": best_revalidation_comparison,
                    "metric_error": metric_error,
                    "elapsed_seconds": elapsed,
                    "access": best_access,
                },
            )
            if best_revalidation_comparison["status"] != "passed":
                raise RuntimeError(
                    "Selected best checkpoint changed classification semantics",
                )
            result["status"] = "complete"
        result.update({
            "terminal_reason": terminal_reason,
            "resumed": resumed,
            "committed_updates": committed,
            "epoch": epoch,
            "within_epoch_cursor": cursor,
            "best_metrics": selection.best_metrics,
            "best_revalidation_metrics": best_revalidation_metrics,
            "best_revalidation_comparison": best_revalidation_comparison,
            **best_identity,
            "best_prediction_rows_sha256": best_prediction_rows_sha256,
            "compiler": _compiler_counts(torch),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
            "final_checkpoint_sha256": final_checkpoint.checkpoint_sha256,
            "final_manifest_sha256": final_checkpoint.manifest_sha256,
        })
    except BaseException as error:
        result.update(_exception_record(error))
        print(result["traceback"], file=sys.stderr, flush=True)
    finally:
        if train_dataset is not None:
            train_dataset.store.close()
        if validation_dataset is not None:
            validation_dataset.store.close()
        result["finished_unix"] = time.time()
        result["elapsed_seconds"] = result["finished_unix"] - result["started_unix"]
        _write_json(result_path, result)
        print(json.dumps(_json_safe(result), sort_keys=True), flush=True)
    return 0 if result["status"] in {"complete", "paused"} else 1


def _launch_children(
    root,
    contract_path,
    source_roots,
    initial_state,
    hashes,
    resume,
    session_started,
):
    processes = {}
    active_branches = (
        BRANCH_DEVICES
        if resume is None
        else {branch: BRANCH_DEVICES[branch] for branch in resume["branch_roots"]}
    )
    for branch, device in active_branches.items():
        branch_root = OUTPUT_ROOT / branch
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(device)
        cache_root = Path(tempfile.gettempdir()) / "spec0036_compile_cache" / branch
        environment["TORCHINDUCTOR_CACHE_DIR"] = str(cache_root / "inductor")
        environment["TRITON_CACHE_DIR"] = str(cache_root / "triton")
        environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        resume_root = ""
        if resume is not None and branch in resume["branch_roots"]:
            resume_root = str(resume["branch_roots"][branch])
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            branch,
            "--bundle-root",
            str(root),
            "--contract-path",
            str(contract_path),
            "--branch-root",
            str(branch_root),
            "--initial-state",
            str(initial_state),
            "--source-roots",
            json.dumps({name: str(path) for name, path in source_roots.items()}),
            "--contract-hashes",
            json.dumps(hashes),
            "--resume-root",
            resume_root,
            "--session-started-unix",
            str(session_started),
        ]
        processes[branch] = subprocess.Popen(command, env=environment)
    return {branch: process.wait() for branch, process in processes.items()}


def _aggregate(return_codes, resume=None):
    summaries = {}
    for branch in BRANCH_DEVICES:
        path = OUTPUT_ROOT / branch / "final_summary.json"
        if (
            resume is not None
            and branch in resume["carried_summaries"]
            and not path.exists()
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(resume["carried_summaries"][branch], path)
        summaries[branch] = (
            _read_object(path)
            if path.is_file()
            else {
                "branch": branch,
                "status": "failed",
                "error": "child emitted no final summary",
            }
        )
    statuses = {branch: row.get("status") for branch, row in summaries.items()}
    paired = None
    if set(statuses.values()) == {"complete"}:
        normal = _read_object(
            OUTPUT_ROOT / "normal_vae/best_validation_predictions.json",
        )
        so2 = _read_object(OUTPUT_ROOT / "so2_vae/best_validation_predictions.json")
        for branch, artifact in (("normal_vae", normal), ("so2_vae", so2)):
            summary = summaries[branch]
            revalidation = _read_object(OUTPUT_ROOT / branch / "best_revalidation.json")
            rows = artifact.get("rows", [])
            identity = {
                "branch": branch,
                "best_boundary": summary.get("best_boundary"),
                "best_checkpoint_sha256": summary.get("best_checkpoint_sha256"),
                "best_manifest_sha256": summary.get("best_manifest_sha256"),
                "contract_hashes": summary.get("contract_hashes"),
            }
            rows_sha256 = _canonical_sha256(rows)
            comparison = revalidation.get("comparison")
            comparison_valid = (
                isinstance(comparison, dict)
                and comparison.get("schema_version")
                == "spec0036.best_revalidation_comparison.v1"
                and comparison.get("status") == "passed"
                and comparison.get("mismatches") == []
                and comparison.get("stable_rows_match") is True
                and comparison.get("unique_wsi_rows_in_same_order") is True
                and comparison.get("discrete_metrics_match") is True
                and comparison.get("finite_outputs") is True
                and comparison.get("argmax_predictions_match") is True
                and comparison.get("selected_primary_macro_f1_is_unique") is True
                and comparison.get("prediction_changes") == []
                and comparison.get("observed_prediction_rows_sha256") == rows_sha256
            )
            if (
                artifact.get("schema_version")
                != "spec0036.best_validation_predictions.v1"
                or any(artifact.get(key) != value for key, value in identity.items())
                or artifact.get("rows_sha256") != rows_sha256
                or summary.get("best_prediction_rows_sha256") != rows_sha256
                or revalidation.get("schema_version") != "spec0036.best_revalidation.v1"
                or any(
                    revalidation.get(key) != value for key, value in identity.items()
                )
                or revalidation.get("prediction_rows_sha256") != rows_sha256
                or revalidation.get("metrics")
                != summary.get("best_revalidation_metrics")
                or revalidation.get("selection_metrics") != summary.get("best_metrics")
                or revalidation.get("comparison")
                != summary.get("best_revalidation_comparison")
                or not comparison_valid
            ):
                raise RuntimeError(
                    f"Best-checkpoint evidence binding differs: {branch}",
                )
        normal_rows = normal.get("rows", [])
        so2_rows = so2.get("rows", [])
        normal_by_wsi = {row["wsi_id"]: row for row in normal_rows}
        so2_by_wsi = {row["wsi_id"]: row for row in so2_rows}
        if set(normal_by_wsi) != set(so2_by_wsi) or len(normal_by_wsi) != 23:
            raise RuntimeError("Best-checkpoint prediction identities do not align")
        keys = sorted(normal_by_wsi)
        truths = [normal_by_wsi[key]["truth"] for key in keys]
        if truths != [so2_by_wsi[key]["truth"] for key in keys]:
            raise RuntimeError("Best-checkpoint validation truths differ")
        from eqvae.training.mil_training import diagnosis_stratified_paired_bootstrap

        paired = diagnosis_stratified_paired_bootstrap(
            truths,
            [normal_by_wsi[key]["prediction"] for key in keys],
            [so2_by_wsi[key]["prediction"] for key in keys],
            normal_cross_entropies=[
                normal_by_wsi[key]["cross_entropy"] for key in keys
            ],
            so2_cross_entropies=[so2_by_wsi[key]["cross_entropy"] for key in keys],
            replicates=BOOTSTRAP_REPLICATES,
        )
        _write_json(OUTPUT_ROOT / "paired_bootstrap.json", paired)
    overall = (
        "complete"
        if set(statuses.values()) == {"complete"}
        else "failed"
        if "failed" in statuses.values()
        else "paused"
    )
    return {
        "schema_version": "spec0036.overall_status.v1",
        "status": overall,
        "branches": summaries,
        "child_return_codes": return_codes,
        "paired_bootstrap": paired,
        "finished_unix": time.time(),
    }


def _run_parent():
    started = time.time()
    _reject_compile_disable_environment()
    root, contract, contract_path = _resolve_input_bundle()
    resume = _resolve_optional_resume(INPUT_CONTRACT_SHA256)
    source_roots = _resolve_source_roots(root / "development", contract)
    install_pinned_torch()
    import torch

    runtime = _validate_runtime(torch)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root / "src"))
    initial_state = _validate_initial_state(torch, root, contract)
    hashes = _contract_hashes(contract, runtime)
    from eqvae.training import mil_training as training

    _prevalidate_resume(training, resume, hashes)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    run_contract = {
        "schema_version": "spec0036.run_contract.v1",
        "input_contract_sha256": INPUT_CONTRACT_SHA256,
        "input_dataset_reference": INPUT_DATASET_REFERENCE,
        "contract_hashes": hashes,
        "kernel_sources": list(KERNEL_SOURCES),
        "source_roots": {name: str(path) for name, path in source_roots.items()},
        "graph_identities": contract["graph_identities"],
        "training_config": contract["training_config"],
        "initial_state": contract["initial_state"],
        "resume_contract_sha256": _sha256(resume["path"]) if resume else None,
        "started_unix": started,
    }
    _write_json(OUTPUT_ROOT / "run_contract.json", run_contract)
    _write_json(OUTPUT_ROOT / "runtime.json", runtime)
    return_codes = _launch_children(
        root,
        contract_path,
        source_roots,
        initial_state,
        hashes,
        resume,
        started,
    )
    overall = _aggregate(return_codes, resume)
    overall["elapsed_seconds"] = time.time() - started
    _write_json(OUTPUT_ROOT / "overall_status.json", overall)
    print(json.dumps(_json_safe(overall), sort_keys=True), flush=True)
    return 0 if overall["status"] in {"complete", "paused"} else 1


def _parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=tuple(BRANCH_DEVICES))
    parser.add_argument("--bundle-root")
    parser.add_argument("--contract-path")
    parser.add_argument("--branch-root")
    parser.add_argument("--initial-state")
    parser.add_argument("--source-roots")
    parser.add_argument("--contract-hashes")
    parser.add_argument("--resume-root", default="")
    parser.add_argument("--session-started-unix")
    return parser


def main():
    args = _parser().parse_args()
    if args.worker:
        return _run_branch(args)
    try:
        return _run_parent()
    except BaseException as error:
        failure = {
            "schema_version": "spec0036.overall_status.v1",
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "finished_unix": time.time(),
        }
        if WORKING_ROOT.is_dir():
            OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
            _write_json(OUTPUT_ROOT / "overall_status.json", failure)
        print(failure["traceback"], file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
