# Copyright 2026 HiperMaximus
# ruff: noqa: ANN001, ANN201, ANN202, BLE001, D101, D103, E501, EM101, EM102, FBT003, INP001, PLC0415, PLR0913, PLR0914, PLR0915, PLR0916, PLR0917, PLR2004, S404, S603, S607, SIM115, T201, TRY003, TRY004
"""Label-blind, all-or-nothing Spec 0043 tissue test inference."""

from __future__ import annotations

import argparse
import csv
import gzip
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
from itertools import starmap
from pathlib import Path

INPUT_ROOT = Path("/kaggle/input")
WORKING_ROOT = Path("/kaggle/working")
OUTPUT_ROOT = WORKING_ROOT / "tissue_test_evaluation"
INPUT_CONTRACT_NAME = "tissue_test_inference_input.json"
INPUT_CONTRACT_SHA256 = "$input_contract_sha256"
INPUT_DATASET_REFERENCE = "$input_dataset_reference"
PINNED_TORCH_VERSION = "2.14.0"
PINNED_TORCH_CUDA = "13.0"
PINNED_TORCH_INDEX = "https://download.pytorch.org/whl/cu130"
TEST_ROW_COUNT = 31_572
TEST_WSI_COUNT = 23
BATCH_SIZE = 159
BUDGETS = (250, 500, 1000, 2500, 5671)
BRANCH_DEVICES = {"normal_vae": 0, "so2_vae": 1}
TEST_HEADER = (
    "dataset_row",
    "atlas_row_index",
    "wsi_id",
    "x",
    "y",
    "split",
    "part",
    "file_index",
)
PREDICTION_HEADER = (
    "dataset_row",
    "atlas_row_index",
    "wsi_id",
    "x",
    "y",
    "part",
    "file_index",
    "logit_tumor",
    "logit_stroma",
    "logit_necrosis",
    "prediction",
)
CATALOG_HEADER = (
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
)
KERNEL_SOURCES = (
    "maximusshtefan/eqvae-ubc-ocean-latent-run-01",
    "maximusshtefan/eqvae-ubc-ocean-latent-run-02",
    "maximusshtefan/eqvae-ubc-ocean-latent-run-03",
    "maximusshtefan/eqvae-ubc-ocean-latent-run-04",
    "maximusshtefan/eqvae-ubc-ocean-latent-run-05",
    "maximusshtefan/eqvae-ubc-ocean-cancer-latent-top-up",
)
FORBIDDEN_NAMES = (
    "label",
    "diagnosis",
    "truth",
    "target",
    "selection",
    "train",
    "validation",
    "metric",
    "prediction",
)
MODEL_STATE_SCHEMA = b"eqvae_spec0043_tissue_model_state_v1"
ROW_IDENTITY_SCHEMA = b"eqvae_spec0043_tissue_row_identity_v1"
EXPECTED_PARAMETER_COUNT = 106_019


@dataclass(frozen=True)
class TestInstance:
    dataset_row: int
    atlas_row_index: int
    wsi_id: int
    x: int
    y: int
    split: str
    pointer: object


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


def _read_csv(path, expected_header):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != expected_header:
            raise RuntimeError(f"Spec 0043 CSV header differs: {path.name}")
        return list(reader)


def _file_record(path):
    return {"bytes": path.stat().st_size, "sha256": _sha256(path)}


def _row_identity_sha256(rows):
    digest = hashlib.sha256(ROW_IDENTITY_SCHEMA)
    for row in rows:
        for name in TEST_HEADER:
            digest.update(str(row[name]).encode())
            digest.update(b"\0")
        digest.update(b"\n")
    return digest.hexdigest()


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


def _resolve_bundle():
    matches = [
        path
        for path in INPUT_ROOT.rglob(INPUT_CONTRACT_NAME)
        if path.is_file() and _sha256(path) == INPUT_CONTRACT_SHA256
    ]
    if len(matches) != 1:
        raise RuntimeError("Expected one exact Spec 0043 input contract")
    contract_path = matches[0]
    root = contract_path.parent
    contract = _read_object(contract_path)
    if (
        contract.get("schema_version") != "spec0043.label_blind_input.v1"
        or contract.get("scope") != "sealed_tissue_test_label_blind_inference_only"
        or contract.get("dataset_reference") != INPUT_DATASET_REFERENCE
        or contract.get("kernel_sources") != list(KERNEL_SOURCES)
        or contract.get("producer_versions") != dict.fromkeys(KERNEL_SOURCES, 1)
        or contract.get("budgets_per_class") != list(BUDGETS)
    ):
        raise RuntimeError("Spec 0043 mounted contract identity differs")
    files = contract.get("files")
    if not isinstance(files, dict):
        raise RuntimeError("Spec 0043 input file manifest is missing")
    observed = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    if observed != {*files, INPUT_CONTRACT_NAME}:
        raise RuntimeError("Spec 0043 input allow-list differs")
    for name, record in files.items():
        path = root / name
        if (
            not isinstance(record, dict)
            or path.stat().st_size != record.get("bytes")
            or _sha256(path) != record.get("sha256")
        ):
            raise RuntimeError(f"Spec 0043 mounted byte differs: {name}")
    allowed = {"test/physical_parts.csv", "test/tissue_test_locations.csv"} | {
        f"weights/{branch}/budget_{budget:04d}_per_class.pt"
        for branch in BRANCH_DEVICES
        for budget in BUDGETS
    }
    non_source = {name for name in files if not name.startswith("src/")}
    if non_source != allowed or any(
        any(word in name.lower() for word in FORBIDDEN_NAMES)
        for name in non_source - {"test/tissue_test_locations.csv"}
    ):
        raise RuntimeError("Spec 0043 input exposes a forbidden logical file")
    if {path.name for path in (root / "test").iterdir()} != {
        "physical_parts.csv",
        "tissue_test_locations.csv",
    }:
        raise RuntimeError("Spec 0043 test-only file set differs")
    return root, contract, contract_path


def _load_test(root, logical_pointer, contract):
    rows = _read_csv(root / "test/tissue_test_locations.csv", TEST_HEADER)
    instances = []
    seen = set()
    for index, row in enumerate(rows):
        instance = TestInstance(
            dataset_row=int(row["dataset_row"]),
            atlas_row_index=int(row["atlas_row_index"]),
            wsi_id=int(row["wsi_id"]),
            x=int(row["x"]),
            y=int(row["y"]),
            split=row["split"],
            pointer=logical_pointer(int(row["part"]), int(row["file_index"])),
        )
        identity = (
            instance.dataset_row,
            instance.atlas_row_index,
            instance.wsi_id,
            instance.x,
            instance.y,
            instance.split,
            instance.pointer.part,
            instance.pointer.file_index,
        )
        if (
            instance.dataset_row != index
            or instance.split != "test"
            or identity in seen
        ):
            raise RuntimeError("Spec 0043 test row identity differs")
        seen.add(identity)
        instances.append(instance)
    identity_rows = [
        {
            "dataset_row": item.dataset_row,
            "atlas_row_index": item.atlas_row_index,
            "wsi_id": item.wsi_id,
            "x": item.x,
            "y": item.y,
            "split": item.split,
            "part": item.pointer.part,
            "file_index": item.pointer.file_index,
        }
        for item in instances
    ]
    expected = contract["test"]
    if (
        len(instances) != TEST_ROW_COUNT
        or len({item.wsi_id for item in instances}) != TEST_WSI_COUNT
        or _row_identity_sha256(identity_rows) != expected["row_identity_sha256"]
    ):
        raise RuntimeError("Spec 0043 test coverage differs")
    return instances


def _resolve_source_roots(root, contract):
    rows = _read_csv(root / "test/physical_parts.csv", CATALOG_HEADER)
    if len(rows) != 12 or rows != contract.get("physical_sources"):
        raise RuntimeError("Spec 0043 physical catalog differs")
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
                f"Spec 0043 expected one sidecar for {row['model_name']} part {row['part']}",
            )
        binary = matches[0] / row["binary_name"]
        if (
            not binary.is_file()
            or binary.stat().st_size != int(row["binary_bytes"])
            or _sha256(binary) != row["binary_sha256"]
        ):
            raise RuntimeError("Spec 0043 physical binary differs")
        source = row["kaggle_source"]
        if source in roots and roots[source] != matches[0]:
            raise RuntimeError("Spec 0043 physical source resolved twice")
        roots[source] = matches[0]
    if set(roots) != set(KERNEL_SOURCES):
        raise RuntimeError("Spec 0043 physical source set differs")
    return roots


def _load_models(torch, root, contract, branch, device, model_class):
    models = {}
    for budget in BUDGETS:
        record = contract["branches"][branch][str(budget)]
        path = root / record["path"]
        state = torch.load(path, map_location="cpu", weights_only=True)
        if (
            not isinstance(state, dict)
            or _sha256(path) != record["file_sha256"]
            or _state_dict_sha256(state) != record["state_sha256"]
        ):
            raise RuntimeError("Spec 0043 model-only state differs")
        model = model_class()
        model.load_state_dict(state, strict=True)
        if (
            sum(parameter.numel() for parameter in model.parameters())
            != EXPECTED_PARAMETER_COUNT
        ):
            raise RuntimeError("Spec 0043 model parameter count differs")
        models[budget] = model.to(
            device=device,
            memory_format=torch.channels_last,
        ).eval()
    return models


def _prediction_row(instance, logits):
    values = [float(value) for value in logits]
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise RuntimeError("Spec 0043 produced nonfinite logits")
    prediction = int(max(range(3), key=values.__getitem__))
    return {
        "dataset_row": instance.dataset_row,
        "atlas_row_index": instance.atlas_row_index,
        "wsi_id": instance.wsi_id,
        "x": instance.x,
        "y": instance.y,
        "part": instance.pointer.part,
        "file_index": instance.pointer.file_index,
        "logit_tumor": values[0],
        "logit_stroma": values[1],
        "logit_necrosis": values[2],
        "prediction": prediction,
    }


def _validate_prediction_file(path):
    identities = []
    content = hashlib.sha256(b"eqvae_spec0043_prediction_rows_v1")
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != PREDICTION_HEADER:
            raise RuntimeError("Spec 0043 prediction header differs")
        for index, row in enumerate(reader):
            logits = [
                float(row[f"logit_{name}"]) for name in ("tumor", "stroma", "necrosis")
            ]
            prediction = int(row["prediction"])
            identity = tuple(int(row[name]) for name in PREDICTION_HEADER[:7])
            if (
                int(row["dataset_row"]) != index
                or not all(math.isfinite(value) for value in logits)
                or prediction != max(range(3), key=logits.__getitem__)
            ):
                raise RuntimeError("Spec 0043 prediction row differs")
            identities.append(identity)
            content.update(_canonical_json({**row, "prediction": prediction}))
            content.update(b"\n")
    if len(identities) != TEST_ROW_COUNT or len(set(identities)) != TEST_ROW_COUNT:
        raise RuntimeError("Spec 0043 prediction coverage differs")
    return {
        "row_count": len(identities),
        "identity_sha256": hashlib.sha256(_canonical_json(identities)).hexdigest(),
        "content_sha256": content.hexdigest(),
    }


def _configure_torch(torch):
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)
    torch.set_float32_matmul_precision("high")


def _run_worker(args):
    import torch

    _configure_torch(torch)
    if torch.cuda.device_count() != 1 or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Spec 0043 worker requires one assigned T4")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    root = Path(args.bundle_root)
    contract = _read_object(Path(args.contract_path))
    sys.path.insert(0, str(root / "src"))
    from eqvae.data.supervised_latents import LogicalPointer, SupervisedLatentStore
    from eqvae.models.supervised import TissueClassifier

    instances = _load_test(root, LogicalPointer, contract)
    source_roots = {
        name: Path(path) for name, path in json.loads(args.source_roots).items()
    }
    store = SupervisedLatentStore(
        catalog_path=root / "test/physical_parts.csv",
        model_name=args.worker,
        source_roots=source_roots,
    )
    worker_root = Path(args.scratch_root) / args.worker
    worker_root.mkdir(parents=True, exist_ok=False)
    handles = {}
    writers = {}
    output_paths = {}
    try:
        models = _load_models(
            torch,
            root,
            contract,
            args.worker,
            device,
            TissueClassifier,
        )
        for budget in BUDGETS:
            path = worker_root / f"budget_{budget:04d}_per_class" / "predictions.csv.gz"
            path.parent.mkdir(parents=True)
            handle = gzip.open(
                path,
                "wt",
                newline="",
                encoding="utf-8",
                compresslevel=6,
            )
            writer = csv.DictWriter(handle, fieldnames=PREDICTION_HEADER)
            writer.writeheader()
            handles[budget] = handle
            writers[budget] = writer
            output_paths[budget] = path
        access = Counter()
        load_seconds = 0.0
        transfer_seconds = 0.0
        compute_seconds = dict.fromkeys(BUDGETS, 0.0)
        with torch.inference_mode():
            for start in range(0, len(instances), BATCH_SIZE):
                batch = instances[start : start + BATCH_SIZE]
                began = time.perf_counter()
                latents, reads = store.read_rows(tuple(item.pointer for item in batch))
                load_seconds += time.perf_counter() - began
                access.update(str(read.part) for read in reads)
                began = time.perf_counter()
                latents = latents.to(
                    device=device,
                    dtype=torch.float16,
                    memory_format=torch.channels_last,
                )
                transfer_seconds += time.perf_counter() - began
                for budget in BUDGETS:
                    began = time.perf_counter()
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits = models[budget](latents)
                    values = logits.detach().float().cpu().tolist()
                    compute_seconds[budget] += time.perf_counter() - began
                    if len(values) != len(batch):
                        raise RuntimeError("Spec 0043 model output batch differs")
                    writers[budget].writerows(
                        starmap(_prediction_row, zip(batch, values, strict=True)),
                    )
        for handle in handles.values():
            handle.close()
        handles.clear()
        prediction_records = {}
        for budget, path in output_paths.items():
            validation = _validate_prediction_file(path)
            prediction_records[str(budget)] = {
                "path": path.relative_to(worker_root).as_posix(),
                "file": _file_record(path),
                **validation,
                "checkpoint": contract["branches"][args.worker][str(budget)],
                "compute_seconds": compute_seconds[budget],
            }
        _write_json(
            worker_root / "worker_status.json",
            {
                "schema_version": "spec0043.worker_status.v1",
                "status": "complete",
                "branch": args.worker,
                "optimizer_updates": 0,
                "batch_size": BATCH_SIZE,
                "final_batch_size": len(instances) % BATCH_SIZE,
                "row_identity_sha256": contract["test"]["row_identity_sha256"],
                "access_by_part": dict(access),
                "load_seconds": load_seconds,
                "transfer_seconds": transfer_seconds,
                "predictions": prediction_records,
            },
        )
        return 0
    finally:
        for handle in handles.values():
            handle.close()
        store.close()


def _driver_versions():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        text=True,
    )
    values = [line.strip() for line in output.splitlines() if line.strip()]
    if len(values) != 2:
        raise RuntimeError("Spec 0043 driver fingerprint differs")
    return values


def _validate_runtime(torch):
    if (
        str(torch.__version__).split("+", maxsplit=1)[0] != PINNED_TORCH_VERSION
        or torch.version.cuda != PINNED_TORCH_CUDA
        or torch.cuda.device_count() != 2
    ):
        raise RuntimeError("Spec 0043 runtime differs")
    names = [torch.cuda.get_device_name(index) for index in range(2)]
    if any("T4" not in name for name in names):
        raise RuntimeError("Spec 0043 requires two T4 GPUs")
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
    root, contract, contract_path = _resolve_bundle()
    source_roots = _resolve_source_roots(root, contract)
    _install_torch()
    import torch

    runtime = _validate_runtime(torch)
    scratch = Path(tempfile.mkdtemp(prefix="spec0043_scratch_", dir=WORKING_ROOT))
    publish_root = WORKING_ROOT / ".tissue_test_evaluation.publishing"
    try:
        processes = {}
        for branch, device in BRANCH_DEVICES.items():
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(device)
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
        if set(codes.values()) != {0}:
            OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
            _write_json(
                OUTPUT_ROOT / "overall_status.json",
                {
                    "schema_version": "spec0043.remote_status.v1",
                    "status": "failed",
                    "child_return_codes": codes,
                    "predictions_published": False,
                },
            )
            return 1
        statuses = {
            branch: _read_object(scratch / branch / "worker_status.json")
            for branch in BRANCH_DEVICES
        }
        identity_hashes = set()
        for branch, status in statuses.items():
            if (
                status.get("status") != "complete"
                or status.get("branch") != branch
                or status.get("optimizer_updates") != 0
                or set(status.get("predictions", {}))
                != {str(value) for value in BUDGETS}
            ):
                raise RuntimeError("Spec 0043 worker completion differs")
            for budget in BUDGETS:
                record = status["predictions"][str(budget)]
                path = scratch / branch / record["path"]
                validation = _validate_prediction_file(path)
                if record["file"] != _file_record(path) or any(
                    record[key] != validation[key] for key in validation
                ):
                    raise RuntimeError("Spec 0043 prediction authentication differs")
                identity_hashes.add(validation["identity_sha256"])
        if len(identity_hashes) != 1:
            raise RuntimeError("Spec 0043 prediction identities differ")
        publish_root.mkdir(parents=True, exist_ok=False)
        _write_json(
            publish_root / "run_contract.json",
            {
                "schema_version": "spec0043.remote_run_contract.v1",
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
        _write_json(publish_root / "runtime.json", runtime)
        for branch in BRANCH_DEVICES:
            shutil.copytree(scratch / branch, publish_root / "branches" / branch)
        published = {
            path.relative_to(publish_root).as_posix(): _file_record(path)
            for path in sorted(publish_root.rglob("*"))
            if path.is_file()
        }
        _write_json(
            publish_root / "overall_status.json",
            {
                "schema_version": "spec0043.remote_status.v1",
                "status": "complete",
                "branches": list(BRANCH_DEVICES),
                "budgets_per_class": list(BUDGETS),
                "optimizer_updates": 0,
                "artifacts": published,
            },
        )
        publish_root.replace(OUTPUT_ROOT)
        return 0
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        shutil.rmtree(publish_root, ignore_errors=True)


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
                    "schema_version": "spec0043.remote_status.v1",
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
