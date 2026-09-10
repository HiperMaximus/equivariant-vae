# Copyright 2026 HiperMaximus
# ruff: noqa: ANN001, ANN201, ANN202, BLE001, D101, D103, E501, EM101, EM102, FBT003, INP001, PLC0415, PLR0914, PLR0916, PLR2004, S404, S603, S607, SLF001, TRY003, TRY004
"""Run the label-blind, inference-only Spec 0041 paired MIL evaluation."""

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
from dataclasses import dataclass
from pathlib import Path

SPEC0041_LABEL_BLIND_MIL_TEST_READY = True
INPUT_ROOT = Path("/kaggle/input")
WORKING_ROOT = Path("/kaggle/working")
OUTPUT_ROOT = WORKING_ROOT / "spec0041_mil_test_inference"
INPUT_CONTRACT_NAME = "mil_test_inference_input.json"
INPUT_CONTRACT_SHA256 = "$input_contract_sha256"
INPUT_DATASET_REFERENCE = "$input_dataset_reference"
PINNED_TORCH_VERSION = "2.14.0"
PINNED_TORCH_CUDA = "13.0"
PINNED_TORCH_INDEX = "https://download.pytorch.org/whl/cu130"
BRANCH_DEVICES = {"normal_vae": 0, "so2_vae": 1}
TEST_WSI_COUNT = 23
TEST_INSTANCE_COUNT = 261_168
MODEL_STATE_SCHEMA = b"eqvae_spec0041_model_state_v1"
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
FORBIDDEN_NAMES = ("label", "diagnosis", "truth", "target", "train", "validation")
COMPILE_DISABLE_ENVIRONMENT = ("TORCH_COMPILE_DISABLE", "TORCHDYNAMO_DISABLE")


@dataclass(frozen=True)
class TestInstance:
    instance_row: int
    atlas_row_index: int
    wsi_id: int
    x: int
    y: int
    split: str
    pointer: object


@dataclass(frozen=True)
class TestBag:
    bag_row: int
    wsi_id: int
    split: str
    instance_start: int
    instance_count: int


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value):
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _canonical_sha256(value):
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(_canonical_json(value) + b"\n")
    temporary.replace(path)


def _read_object(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def _resolve_bundle():
    matches = [
        path
        for path in INPUT_ROOT.rglob(INPUT_CONTRACT_NAME)
        if path.is_file() and _sha256(path) == INPUT_CONTRACT_SHA256
    ]
    if len(matches) != 1:
        raise RuntimeError("Expected one exact Spec 0041 input contract")
    contract_path = matches[0]
    root = contract_path.parent
    contract = _read_object(contract_path)
    if (
        contract.get("schema_version") != "spec0041.label_blind_input.v1"
        or contract.get("scope") != "sealed_test_label_blind_inference_only"
        or contract.get("dataset_reference") != INPUT_DATASET_REFERENCE
        or contract.get("kernel_sources") != list(KERNEL_SOURCES)
        or contract.get("producer_versions") != dict.fromkeys(KERNEL_SOURCES, 1)
    ):
        raise RuntimeError("Spec 0041 mounted contract identity differs")
    files = contract.get("files")
    if not isinstance(files, dict):
        raise RuntimeError("Spec 0041 input file manifest is missing")
    observed = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    if observed != {*files, INPUT_CONTRACT_NAME}:
        raise RuntimeError("Spec 0041 input allow-list differs")
    for name, record in files.items():
        path = root / name
        if (
            not isinstance(record, dict)
            or path.stat().st_size != record.get("bytes")
            or _sha256(path) != record.get("sha256")
        ):
            raise RuntimeError(f"Spec 0041 mounted byte differs: {name}")
    test_entries = {path.name for path in (root / "test").iterdir()}
    if test_entries != {
        "physical_parts.csv",
        "wsi_cancer_test_bags.csv",
        "wsi_cancer_test_instances.csv",
    }:
        raise RuntimeError("Spec 0041 test-only file set differs")
    allowed = {
        "test/physical_parts.csv",
        "test/wsi_cancer_test_bags.csv",
        "test/wsi_cancer_test_instances.csv",
        "weights/normal_vae.pt",
        "weights/so2_vae.pt",
    }
    non_source = {name for name in files if not name.startswith("src/")}
    if non_source != allowed or any(
        any(word in name.lower() for word in FORBIDDEN_NAMES)
        for name in non_source - allowed
    ):
        raise RuntimeError("Spec 0041 input exposes a forbidden logical file")
    return root, contract, contract_path


def _read_csv(path, expected_header):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != expected_header:
            raise RuntimeError(f"Spec 0041 CSV header differs: {path.name}")
        return list(reader)


def _load_test(root, logical_pointer):
    instance_rows = _read_csv(
        root / "test/wsi_cancer_test_instances.csv",
        (
            "instance_row",
            "atlas_row_index",
            "wsi_id",
            "x",
            "y",
            "split",
            "part",
            "file_index",
        ),
    )
    bag_rows = _read_csv(
        root / "test/wsi_cancer_test_bags.csv",
        ("bag_row", "wsi_id", "split", "instance_start", "instance_count"),
    )
    if len(instance_rows) != TEST_INSTANCE_COUNT or len(bag_rows) != TEST_WSI_COUNT:
        raise RuntimeError("Spec 0041 test row counts differ")
    instances = []
    previous = None
    for index, row in enumerate(instance_rows):
        instance = TestInstance(
            instance_row=int(row["instance_row"]),
            atlas_row_index=int(row["atlas_row_index"]),
            wsi_id=int(row["wsi_id"]),
            x=int(row["x"]),
            y=int(row["y"]),
            split=row["split"],
            pointer=logical_pointer(int(row["part"]), int(row["file_index"])),
        )
        order = (instance.wsi_id, instance.y, instance.x)
        if (
            instance.instance_row != index
            or instance.split != "test"
            or (previous is not None and order <= previous)
        ):
            raise RuntimeError("Spec 0041 test instance identity differs")
        instances.append(instance)
        previous = order
    bags = []
    cursor = 0
    seen = set()
    for index, row in enumerate(bag_rows):
        bag = TestBag(
            index,
            int(row["wsi_id"]),
            row["split"],
            int(row["instance_start"]),
            int(row["instance_count"]),
        )
        selected = instances[cursor : cursor + bag.instance_count]
        if (
            int(row["bag_row"]) != index
            or bag.split != "test"
            or bag.wsi_id in seen
            or bag.instance_start != cursor
            or len(selected) != bag.instance_count
            or any(item.wsi_id != bag.wsi_id for item in selected)
        ):
            raise RuntimeError("Spec 0041 test bag identity differs")
        bags.append(bag)
        seen.add(bag.wsi_id)
        cursor += bag.instance_count
    if cursor != len(instances):
        raise RuntimeError("Spec 0041 test bags do not consume all instances")
    return instances, bags


def _resolve_source_roots(root, contract):
    rows = _read_csv(
        root / "test/physical_parts.csv",
        (
            "part",
            "model_name",
            "kaggle_source",
            "binary_name",
            "row_count",
            "binary_bytes",
            "binary_sha256",
            "sidecar_name",
            "sidecar_bytes",
            "sidecar_sha256",
        ),
    )
    if len(rows) != 30 or rows != contract.get("physical_sources"):
        raise RuntimeError("Spec 0041 physical catalog differs")
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
                f"Spec 0041 expected one sidecar for {row['model_name']} part {row['part']}",
            )
        binary = matches[0] / row["binary_name"]
        if (
            not binary.is_file()
            or binary.stat().st_size != int(row["binary_bytes"])
            or _sha256(binary) != row["binary_sha256"]
        ):
            raise RuntimeError("Spec 0041 physical binary differs")
        source = row["kaggle_source"]
        if source in roots and roots[source] != matches[0]:
            raise RuntimeError("Spec 0041 physical source resolved twice")
        roots[source] = matches[0]
    if set(roots) != set(KERNEL_SOURCES):
        raise RuntimeError("Spec 0041 physical source set differs")
    return roots


def _state_dict_sha256(state):
    digest = hashlib.sha256(MODEL_STATE_SCHEMA)
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


def _configure_torch(torch):
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)
    torch.set_float32_matmul_precision("high")


def _make_model(torch, root, contract, branch, device):
    from eqvae.models.local_attention_candidates import (
        WholeBagFixed25Attention,
        use_whole_bag_fixed25_attention,
    )
    from eqvae.models.local_global_mil import (
        EXPECTED_PARAMETER_COUNT,
        LocalGlobalMILClassifier,
    )

    record = contract["branches"][branch]
    state_path = root / record["path"]
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    if (
        _sha256(state_path) != record["file_sha256"]
        or _state_dict_sha256(state) != record["state_sha256"]
    ):
        raise RuntimeError("Spec 0041 model-only state differs")
    model = LocalGlobalMILClassifier()
    model.load_state_dict(state)
    before = _state_dict_sha256(model.state_dict())
    use_whole_bag_fixed25_attention(model)
    if before != _state_dict_sha256(model.state_dict()) or not all(
        isinstance(block.attention, WholeBagFixed25Attention)
        for block in model.local_blocks
    ):
        raise RuntimeError("Spec 0041 attention replacement changed state")
    if (
        sum(parameter.numel() for parameter in model.parameters())
        != EXPECTED_PARAMETER_COUNT
    ):
        raise RuntimeError("Spec 0041 model parameter count differs")
    model = model.to(device=device, memory_format=torch.channels_last).eval()

    def inference(latents, graph):
        with torch.autocast("cuda", dtype=torch.float16):
            return model(latents, graph)

    return torch.compile(
        inference,
        backend="inductor",
        fullgraph=True,
        dynamic=None,
        mode="max-autotune-no-cudagraphs",
        recompile_limit=3,
        isolate_recompiles=True,
    )


def _build_graphs(instances, bags, expected, build_graph):
    graphs = {}
    summaries = {}
    for bag in bags:
        selected = instances[
            bag.instance_start : bag.instance_start + bag.instance_count
        ]
        graph = build_graph(selected, expected_instance_count=bag.instance_count)
        if graph.identity_sha256 != expected.get(str(bag.wsi_id)):
            raise RuntimeError(f"Spec 0041 graph identity differs: {bag.wsi_id}")
        degree = graph.neighbor_valid.sum(dim=1)
        graphs[bag.wsi_id] = graph
        summaries[bag.wsi_id] = {
            "minimum": int(degree.min().item()),
            "maximum": int(degree.max().item()),
            "mean": float(degree.float().mean().item()),
        }
    if set(map(str, graphs)) != set(expected):
        raise RuntimeError("Spec 0041 graph coverage differs")
    return graphs, summaries


def _run_worker(args):
    import torch

    _configure_torch(torch)
    if torch.cuda.device_count() != 1 or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Spec 0041 worker requires one assigned T4")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    root = Path(args.bundle_root)
    contract = _read_object(Path(args.contract_path))
    sys.path.insert(0, str(root / "src"))
    from eqvae.data.supervised_latents import LogicalPointer, SupervisedLatentStore
    from eqvae.models.local_global_mil import build_local_attention_graph

    instances, bags = _load_test(root, LogicalPointer)
    if {str(bag.wsi_id): bag.instance_count for bag in bags} != contract["test"][
        "bag_sizes"
    ]:
        raise RuntimeError("Spec 0041 bag-size identity differs")
    graphs, degrees = _build_graphs(
        instances,
        bags,
        contract["test"]["graph_identities"],
        build_local_attention_graph,
    )
    source_roots = {
        name: Path(path) for name, path in json.loads(args.source_roots).items()
    }
    store = SupervisedLatentStore(
        catalog_path=root / "test/physical_parts.csv",
        model_name=args.worker,
        source_roots=source_roots,
    )
    output = Path(args.scratch_root) / args.worker / "predictions.json"
    try:
        inference = _make_model(torch, root, contract, args.worker, device)
        gpu_graphs = {}
        rows = []
        with torch.inference_mode():
            for bag in bags:
                graph = gpu_graphs.get(bag.wsi_id)
                if graph is None:
                    graph = graphs[bag.wsi_id].to(device)
                    gpu_graphs[bag.wsi_id] = graph
                selected = instances[
                    bag.instance_start : bag.instance_start + bag.instance_count
                ]
                started = time.perf_counter()
                latents, reads = store.read_rows(
                    tuple(item.pointer for item in selected),
                )
                load_seconds = time.perf_counter() - started
                transfer_started = time.perf_counter()
                latents = latents.to(
                    device=device,
                    dtype=torch.float16,
                    memory_format=torch.channels_last,
                )
                transfer_seconds = time.perf_counter() - transfer_started
                torch._dynamo.maybe_mark_dynamic(latents, 0)
                for tensor in (
                    graph.neighbor_index,
                    graph.neighbor_valid,
                    graph.radial_code,
                ):
                    torch._dynamo.maybe_mark_dynamic(tensor, 0)
                compute_started = time.perf_counter()
                logits = inference(latents, graph)
                logits_cpu = [
                    float(value) for value in logits.detach().float().cpu().tolist()
                ]
                compute_seconds = time.perf_counter() - compute_started
                if len(logits_cpu) != 5 or not all(
                    math.isfinite(value) for value in logits_cpu
                ):
                    raise RuntimeError("Spec 0041 produced nonfinite logits")
                rows.append({
                    "wsi_id": bag.wsi_id,
                    "prediction": int(max(range(5), key=logits_cpu.__getitem__)),
                    "logits": logits_cpu,
                    "bag_size": bag.instance_count,
                    "graph_identity": graph.identity_sha256,
                    "graph_degree": degrees[bag.wsi_id],
                    "access": {
                        "parts": dict(Counter(str(read.part) for read in reads)),
                        "load_seconds": load_seconds,
                        "transfer_seconds": transfer_seconds,
                        "compute_seconds": compute_seconds,
                    },
                })
        artifact = {
            "schema_version": "spec0041.label_blind_predictions.v1",
            "branch": args.worker,
            "checkpoint": contract["branches"][args.worker]["source"],
            "optimizer_updates": 0,
            "rows_sha256": _canonical_sha256(rows),
            "rows": rows,
        }
        _write_json(output, artifact)
        return 0
    finally:
        store.close()


def _driver_versions():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        text=True,
    )
    values = [line.strip() for line in output.splitlines() if line.strip()]
    if len(values) != 2:
        raise RuntimeError("Spec 0041 driver fingerprint differs")
    return values


def _validate_runtime(torch):
    if (
        str(torch.__version__).split("+", maxsplit=1)[0] != PINNED_TORCH_VERSION
        or torch.version.cuda != PINNED_TORCH_CUDA
        or torch.cuda.device_count() != 2
    ):
        raise RuntimeError("Spec 0041 runtime differs")
    names = [torch.cuda.get_device_name(index) for index in range(2)]
    if any("T4" not in name for name in names):
        raise RuntimeError("Spec 0041 requires two T4 GPUs")
    return {
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "devices": names,
        "capabilities": [
            list(torch.cuda.get_device_capability(index)) for index in range(2)
        ],
        "driver_versions": _driver_versions(),
    }


def _install_torch():
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


def _run_parent():
    disabled = {
        name: os.environ[name]
        for name in COMPILE_DISABLE_ENVIRONMENT
        if os.environ.get(name, "").lower() not in {"", "0", "false", "no", "off"}
    }
    if disabled:
        raise RuntimeError("Spec 0041 refuses compile-disabled execution")
    root, contract, contract_path = _resolve_bundle()
    source_roots = _resolve_source_roots(root, contract)
    _install_torch()
    import torch

    runtime = _validate_runtime(torch)
    scratch = Path(tempfile.mkdtemp(prefix="spec0041_scratch_", dir=WORKING_ROOT))
    try:
        processes = {}
        for branch, device in BRANCH_DEVICES.items():
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(device)
            cache = Path(tempfile.gettempdir()) / "spec0041_compile" / branch
            environment["TORCHINDUCTOR_CACHE_DIR"] = str(cache / "inductor")
            environment["TRITON_CACHE_DIR"] = str(cache / "triton")
            environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                branch,
                "--bundle-root",
                str(root),
                "--contract-path",
                str(contract_path),
                "--scratch-root",
                str(scratch),
                "--source-roots",
                json.dumps({name: str(path) for name, path in source_roots.items()}),
            ]
            processes[branch] = subprocess.Popen(command, env=environment)
        codes = {branch: process.wait() for branch, process in processes.items()}
        artifacts = {}
        if set(codes.values()) == {0}:
            for branch in BRANCH_DEVICES:
                path = scratch / branch / "predictions.json"
                artifact = _read_object(path)
                rows = artifact.get("rows", [])
                if (
                    artifact.get("branch") != branch
                    or len(rows) != TEST_WSI_COUNT
                    or artifact.get("rows_sha256") != _canonical_sha256(rows)
                ):
                    raise RuntimeError("Spec 0041 branch prediction artifact differs")
                artifacts[branch] = artifact
            normal_rows = artifacts["normal_vae"]["rows"]
            so2_rows = artifacts["so2_vae"]["rows"]
            stable = ("wsi_id", "bag_size", "graph_identity", "graph_degree")
            if any(
                any(left[field] != right[field] for field in stable)
                for left, right in zip(normal_rows, so2_rows, strict=True)
            ):
                raise RuntimeError("Spec 0041 paired prediction identities differ")
            OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
            _write_json(
                OUTPUT_ROOT / "run_contract.json",
                {
                    "schema_version": "spec0041.remote_run_contract.v1",
                    "input_contract_sha256": INPUT_CONTRACT_SHA256,
                    "input_dataset_reference": INPUT_DATASET_REFERENCE,
                    "spec_sha256": contract["spec_sha256"],
                    "scorer_sha256": contract["scorer_sha256"],
                    "test_vector_sha256": contract["test_vector_sha256"],
                    "branches": contract["branches"],
                    "test": contract["test"],
                    "runtime": runtime,
                    "optimizer_updates": 0,
                },
            )
            _write_json(OUTPUT_ROOT / "runtime.json", runtime)
            for branch, artifact in artifacts.items():
                _write_json(OUTPUT_ROOT / branch / "predictions.json", artifact)
            published = {
                path.relative_to(OUTPUT_ROOT).as_posix(): {
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in sorted(OUTPUT_ROOT.rglob("*"))
                if path.is_file()
            }
            _write_json(
                OUTPUT_ROOT / "overall_status.json",
                {
                    "schema_version": "spec0041.remote_status.v1",
                    "status": "complete",
                    "branches": list(BRANCH_DEVICES),
                    "optimizer_updates": 0,
                    "artifacts": published,
                },
            )
            return 0
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
        _write_json(
            OUTPUT_ROOT / "overall_status.json",
            {
                "schema_version": "spec0041.remote_status.v1",
                "status": "failed",
                "child_return_codes": codes,
                "predictions_published": False,
            },
        )
        return 1
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=tuple(BRANCH_DEVICES))
    parser.add_argument("--bundle-root")
    parser.add_argument("--contract-path")
    parser.add_argument("--scratch-root")
    parser.add_argument("--source-roots")
    args = parser.parse_args()
    try:
        code = _run_worker(args) if args.worker else _run_parent()
    except BaseException as error:
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        if not args.worker and not OUTPUT_ROOT.exists():
            OUTPUT_ROOT.mkdir(parents=True)
            _write_json(
                OUTPUT_ROOT / "overall_status.json",
                {
                    "schema_version": "spec0041.remote_status.v1",
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "predictions_published": False,
                },
            )
        code = 1
    raise SystemExit(code)


if __name__ == "__main__":
    main()
