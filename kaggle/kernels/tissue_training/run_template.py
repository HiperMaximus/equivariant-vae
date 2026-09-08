# Copyright 2026 HiperMaximus
# ruff: noqa: ANN001, ANN002, ANN003, ANN201, ANN202, ANN204, ARG001, BLE001, C901, D102, D103, D105, D107, E501, EM101, EM102, FBT003, INP001, N803, N818, PLC0415, PLR0912, PLR0913, PLR0914, PLR0915, PLR0916, PLR0917, PLR2004, PLR6301, PLW0603, PLW0717, S105, S311, S404, S603, SLF001, T201, TRY003, TRY301
"""Run the paired, development-only Spec 0039 tissue training campaign."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

SPEC0039_TISSUE_TRAINING_READY = True
INPUT_ROOT = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/working/tissue_label_efficiency_training")
OUTPUT_PATH = WORK_ROOT / "spec0039_tissue_training.json"
INPUT_CONTRACT_NAME = "tissue_training_input.json"
INPUT_CONTRACT_SHA256 = "$input_contract_sha256"
INPUT_DATASET_REFERENCE = "$input_dataset_reference"
PINNED_TORCH_VERSION = "2.14.0"
PINNED_TORCH_CUDA = "13.0"
PINNED_TORCH_INDEX = "https://download.pytorch.org/whl/cu130"
KERNEL_SOURCES = (
    "maximusshtefan/eqvae-ubc-ocean-latent-run-01",
    "maximusshtefan/eqvae-ubc-ocean-latent-run-02",
    "maximusshtefan/eqvae-ubc-ocean-latent-run-03",
    "maximusshtefan/eqvae-ubc-ocean-latent-run-04",
    "maximusshtefan/eqvae-ubc-ocean-latent-run-05",
    "maximusshtefan/eqvae-ubc-ocean-cancer-latent-top-up",
)
BRANCHES = (("normal_vae", 0), ("so2_vae", 1))
LABELS = ("tumor", "stroma", "necrosis")
DEFAULT_MINIMUM_COMPLETED_EPOCHS = 10


class CampaignFailure(RuntimeError):
    """A fail-closed campaign error retaining its exact phase."""


class InlineExecutor:
    """Keep each worker's compiled-autograd calls on its main thread."""

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def submit(self, function, *args, **kwargs):
        return InlineFuture(function(*args, **kwargs))


class InlineFuture:
    """Match the small future surface used by a one-worker branch."""

    def __init__(self, value):
        self.value = value

    def result(self):
        return self.value


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode(),
    ).hexdigest()


def read_object(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CampaignFailure(f"Expected a JSON object: {path}")
    return value


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def resolve_input_bundle():
    matches = [
        path
        for path in INPUT_ROOT.rglob(INPUT_CONTRACT_NAME)
        if path.is_file() and sha256(path) == INPUT_CONTRACT_SHA256
    ]
    if len(matches) != 1:
        raise CampaignFailure("Expected one exact Spec 0039 input contract")
    contract_path = matches[0]
    root = contract_path.parent
    contract = read_object(contract_path)
    if (
        contract.get("schema_version") != "spec0039.tissue_training_input.v1"
        or contract.get("dataset_reference") != INPUT_DATASET_REFERENCE
        or contract.get("scope") != "tissue_development_training_no_sealed_test"
        or contract.get("visibility") != "private"
        or contract.get("kernel_sources") != list(KERNEL_SOURCES)
    ):
        raise CampaignFailure("Spec 0039 mounted input identity differs")
    files = contract.get("files")
    if not isinstance(files, dict):
        raise CampaignFailure("Spec 0039 input file manifest is missing")
    observed = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    if observed != {*files, INPUT_CONTRACT_NAME}:
        raise CampaignFailure("Spec 0039 mounted input allow-list differs")
    exposed = [
        name
        for name in observed
        if not name.startswith("src/")
        and ("test" in name or name == "tissue/tissue_validation.csv")
    ]
    if exposed:
        raise CampaignFailure(
            f"Spec 0039 input exposes forbidden logical assets: {exposed}",
        )
    for name, record in files.items():
        if not isinstance(record, dict):
            raise CampaignFailure("Spec 0039 file record is malformed")
        path = root / name
        if path.stat().st_size != record.get("bytes") or sha256(path) != record.get(
            "sha256",
        ):
            raise CampaignFailure(f"Spec 0039 mounted file differs: {name}")
    config = read_object(root / "tissue_training_config.json")
    if sha256(root / "tissue_training_config.json") != contract.get(
        "configuration_sha256",
    ):
        raise CampaignFailure("Spec 0039 training configuration differs")
    model = contract.get("model")
    if not isinstance(model, dict) or sha256(
        root / "tissue_initial_state.pt",
    ) != model.get(
        "initial_state_file_sha256",
    ):
        raise CampaignFailure("Spec 0039 initial state binding differs")
    return root, contract, config


def resolve_source_roots(catalog_path, physical_sources):
    with catalog_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if rows != physical_sources:
        raise CampaignFailure("Spec 0039 physical catalog differs from its contract")
    roots = {}
    for row in rows:
        matches = [
            path.parent
            for path in INPUT_ROOT.rglob(row["sidecar_name"])
            if path.is_file()
            and path.stat().st_size == int(row["sidecar_bytes"])
            and sha256(path) == row["sidecar_sha256"]
        ]
        if len(matches) != 1:
            raise CampaignFailure(
                f"Expected one sidecar match for {row['model_name']} part {row['part']}",
            )
        binary = matches[0] / row["binary_name"]
        if not binary.is_file() or binary.stat().st_size != int(row["binary_bytes"]):
            raise CampaignFailure("Spec 0039 mounted latent binary differs")
        source = row["kaggle_source"]
        if source in roots and roots[source] != matches[0]:
            raise CampaignFailure("One Spec 0039 physical source resolved twice")
        roots[source] = matches[0]
    if tuple(roots) != KERNEL_SOURCES:
        raise CampaignFailure("Spec 0039 physical source order differs")
    return roots


def install_pinned_torch():
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        f"torch=={PINNED_TORCH_VERSION}",
        "--index-url",
        PINNED_TORCH_INDEX,
    ])


def validate_pinned_torch(torch):
    installed = str(torch.__version__).split("+", maxsplit=1)[0]
    if installed != PINNED_TORCH_VERSION or torch.version.cuda != PINNED_TORCH_CUDA:
        raise CampaignFailure(
            f"Pinned runtime differs: torch={torch.__version__}, cuda={torch.version.cuda}",
        )


def configure_torch(torch):
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(False)
    torch.set_float32_matmul_precision("high")
    functorch = getattr(torch, "_functorch", None)
    config = getattr(functorch, "config", None)
    if config is not None and hasattr(config, "backward_pass_autocast"):
        config.backward_pass_autocast = "off"
    dynamo = torch._dynamo.config
    if not hasattr(dynamo, "compiled_autograd"):
        raise CampaignFailure("Spec 0039 requires torch._dynamo.compiled_autograd")
    dynamo.compiled_autograd = True
    if not dynamo.compiled_autograd:
        raise CampaignFailure("Spec 0039 could not enable compiled autograd")
    return {
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "tf32": {"supported_on_t4": False, "enabled": False},
        "channels_last": True,
        "fused_adamw": True,
        "compile_mode": "max-autotune",
        "compiled_autograd": bool(getattr(dynamo, "compiled_autograd", False)),
    }


def cpu_state(torch, model):
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def state_sha256(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(repr(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def save_checkpoint(torch, path, *, state, manifest):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({"state_dict": state, "manifest": manifest}, temporary)
    temporary.replace(path)
    atomic_json(path.with_suffix(".json"), {**manifest, "file_sha256": sha256(path)})


def make_optimizer(torch, model, optimizer_config):
    matrix = [parameter for parameter in model.parameters() if parameter.ndim >= 2]
    vector = [parameter for parameter in model.parameters() if parameter.ndim < 2]
    if not matrix or not vector:
        raise CampaignFailure("Spec 0039 matrix/vector decay partition differs")
    return torch.optim.AdamW(
        [
            {"params": matrix, "weight_decay": optimizer_config["matrix_weight_decay"]},
            {"params": vector, "weight_decay": optimizer_config["vector_weight_decay"]},
        ],
        lr=optimizer_config["peak_lr"],
        betas=tuple(optimizer_config["betas"]),
        eps=optimizer_config["eps"],
        fused=optimizer_config["fused"],
        capturable=False,
    )


def make_branch(
    torch,
    TissueClassifier,
    *,
    name,
    device,
    initial_state,
    optimizer_config,
):
    model = TissueClassifier().to(device=device, memory_format=torch.channels_last)
    model.load_state_dict(initial_state)

    def loss_closure(inputs, labels):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(inputs)
        return torch.nn.functional.cross_entropy(logits.float(), labels)

    return {
        "name": name,
        "device": device,
        "model": model,
        "loss": torch.compile(
            loss_closure,
            mode="max-autotune",
            fullgraph=True,
            dynamic=False,
        ),
        "optimizer_config": optimizer_config,
        "optimizer": None,
        "scaler": None,
        "history": {"calibration": [], "updates": [], "validation": []},
        "best": None,
        "best_state": None,
        "last_improvement_check": 0,
        "patience_exhausted": False,
    }


def to_device(torch, dataset, indices, device):
    batch = dataset.read_batch(indices)
    inputs = batch.latents.pin_memory().to(device=device, non_blocking=True)
    inputs = inputs.contiguous(memory_format=torch.channels_last)
    labels = batch.labels.pin_memory().to(device=device, non_blocking=True)
    return batch, inputs, labels


def update_access_digest(digest, batch):
    """Append actual grouped physical reads, bound to their logical records."""
    for read in batch.physical_reads:
        instance = batch.instances[read.logical_index]
        digest.update(
            json.dumps(
                {
                    "dataset_row": instance.dataset_row,
                    "part": read.part,
                    "file_index": read.file_index,
                    "logical_index": read.logical_index,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode(),
        )


def gradients_are_finite(torch, model):
    return all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all().item())
        for parameter in model.parameters()
    )


def clear_gradients(branch):
    for parameter in branch["model"].parameters():
        parameter.grad = None


def calibrate_branch(
    torch,
    branch,
    dataset,
    *,
    calibration_batches,
    initial_state,
    config,
):
    """Settle the ordinary scaler without leaking calibration optimizer state.

    Raises:
        CampaignFailure: If default-scaler calibration cannot establish a finite update.

    """
    scaler = torch.amp.GradScaler("cuda")
    if not math.isclose(float(scaler.get_scale()), 65536.0):
        raise CampaignFailure("Spec 0039 no longer starts from the default GradScaler")
    base_rng = torch.random.get_rng_state()
    for ordinal, indices in enumerate(calibration_batches, start=1):
        backoffs = 0
        while True:
            branch["model"].load_state_dict(initial_state)
            clear_gradients(branch)
            torch.random.set_rng_state(base_rng)
            optimizer = make_optimizer(
                torch,
                branch["model"],
                branch["optimizer_config"],
            )
            _batch, inputs, labels = to_device(
                torch,
                dataset,
                indices,
                branch["device"],
            )
            loss = branch["loss"](inputs, labels)
            if loss.ndim != 0 or not bool(torch.isfinite(loss.detach()).item()):
                finite = False
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                finite = gradients_are_finite(torch, branch["model"])
            before = float(scaler.get_scale())
            if finite:
                scaler.step(optimizer)
                scaler.update()
                branch["history"]["calibration"].append({
                    "ordinal": ordinal,
                    "backoffs": backoffs,
                    "loss": float(loss.detach().item()),
                    "scale_before": before,
                    "scale_after": float(scaler.get_scale()),
                })
                break
            clear_gradients(branch)
            if backoffs >= config["runtime"]["max_scaler_backoffs"]:
                raise CampaignFailure(
                    f"{branch['name']} exceeded calibration scaler backoffs on batch {ordinal}",
                )
            scaler.update(new_scale=before * 0.5)
            backoffs += 1
    branch["model"].load_state_dict(initial_state)
    clear_gradients(branch)
    branch["optimizer"] = make_optimizer(
        torch,
        branch["model"],
        branch["optimizer_config"],
    )
    branch["scaler"] = scaler
    if branch["optimizer"].state:
        raise CampaignFailure("Spec 0039 real optimizer inherited calibration state")


def epoch_batches(dataset, *, batch_size, epoch, seed):
    groups = {
        label: [
            index
            for index, item in enumerate(dataset.instances)
            if item.tissue_label == label
        ]
        for label in LABELS
    }
    if len({len(rows) for rows in groups.values()}) != 1:
        raise CampaignFailure("Spec 0039 training pool is no longer class-balanced")
    per_class = len(groups[LABELS[0]])
    for label_index, label in enumerate(LABELS):
        random.Random(f"{seed}:{epoch}:{label_index}").shuffle(groups[label])
    if batch_size == 159:
        composition = ((53, 53, 53),)
    elif batch_size == 125:
        composition = ((42, 42, 41), (42, 41, 42), (41, 42, 42))
    else:
        raise CampaignFailure("Spec 0039 batch shape is not selected by Spec 0037")
    cursors = dict.fromkeys(LABELS, 0)
    batches = []
    while cursors[LABELS[0]] < per_class:
        amounts = composition[len(batches) % len(composition)]
        selected = []
        for label, amount in zip(LABELS, amounts, strict=True):
            start = cursors[label]
            stop = start + amount
            if stop > per_class:
                raise CampaignFailure("Spec 0039 static class batch has a tail")
            selected.extend(groups[label][start:stop])
            cursors[label] = stop
        random.Random(f"{seed}:{epoch}:batch:{len(batches)}").shuffle(selected)
        batches.append(tuple(selected))
    if any(cursor != per_class for cursor in cursors.values()):
        raise CampaignFailure("Spec 0039 epoch does not consume its train pool")
    return tuple(batches)


def validation_metrics(
    torch,
    branch,
    dataset,
    batch_size,
    *,
    access_digest=None,
    capture_predictions=False,
):
    model = branch["model"]
    device = branch["device"]
    model.eval()
    confusion = [[0, 0, 0] for _ in LABELS]
    total_ce = 0.0
    total = 0
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(dataset), batch_size):
            indices = tuple(range(start, min(start + batch_size, len(dataset))))
            batch, inputs, labels = to_device(torch, dataset, indices, device)
            if access_digest is not None:
                update_access_digest(access_digest, batch)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(inputs)
            logits = logits.float()
            losses = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
            classes = logits.argmax(dim=1)
            labels_cpu = labels.cpu().tolist()
            classes_cpu = classes.cpu().tolist()
            logits_cpu = logits.cpu().tolist()
            losses_cpu = losses.cpu().tolist()
            total_ce += float(losses.sum().item())
            total += len(labels_cpu)
            for offset, (truth, predicted) in enumerate(
                zip(labels_cpu, classes_cpu, strict=True),
            ):
                confusion[truth][predicted] += 1
                if capture_predictions:
                    instance = batch.instances[offset]
                    predictions.append({
                        "dataset_row": instance.dataset_row,
                        "atlas_row_index": instance.atlas_row_index,
                        "wsi_id": instance.wsi_id,
                        "x": instance.x,
                        "y": instance.y,
                        "part": instance.pointer.part,
                        "file_index": instance.pointer.file_index,
                        "truth": truth,
                        "logit_tumor": logits_cpu[offset][0],
                        "logit_stroma": logits_cpu[offset][1],
                        "logit_necrosis": logits_cpu[offset][2],
                        "prediction": predicted,
                        "cross_entropy": losses_cpu[offset],
                    })
    if total != len(dataset):
        raise CampaignFailure("Spec 0039 validation did not consume its selected set")
    f1 = []
    recalls = []
    for label in range(3):
        true_positive = confusion[label][label]
        false_positive = sum(confusion[row][label] for row in range(3) if row != label)
        false_negative = sum(
            confusion[label][column] for column in range(3) if column != label
        )
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1.append(
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0,
        )
        recalls.append(recall)
    metrics = {
        "rows": total,
        "macro_f1": sum(f1) / 3.0,
        "balanced_accuracy": sum(recalls) / 3.0,
        "accuracy": sum(confusion[index][index] for index in range(3)) / total,
        "mean_cross_entropy": total_ce / total,
        "per_class_f1": dict(zip(LABELS, f1, strict=True)),
        "confusion": confusion,
    }
    model.train()
    return metrics, predictions


def train_branch_step(
    torch,
    branch,
    dataset,
    indices,
    learning_rate,
    access_digest,
    *,
    epoch,
    step,
    update_index,
):
    """Issue one independent native-AMP update on this branch's T4."""
    optimizer = branch["optimizer"]
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    batch, inputs, labels = to_device(torch, dataset, indices, branch["device"])
    update_access_digest(access_digest, batch)
    loss = branch["loss"](inputs, labels)
    branch["scaler"].scale(loss).backward()
    branch["scaler"].step(optimizer)
    branch["scaler"].update()
    branch["history"]["updates"].append({
        "epoch": epoch,
        "step": step,
        "update_index": update_index,
        "learning_rate": learning_rate,
    })


def improves(candidate, incumbent):
    if incumbent is None:
        return True
    if candidate["macro_f1"] != incumbent["macro_f1"]:
        return candidate["macro_f1"] > incumbent["macro_f1"]
    return candidate["mean_cross_entropy"] < incumbent["mean_cross_entropy"]


def minimum_completed_epochs(config):
    value = config.get("minimum_completed_epochs", DEFAULT_MINIMUM_COMPLETED_EPOCHS)
    if type(value) is not int or value < 1 or value > config["maximum_epochs"]:
        raise CampaignFailure("Spec 0039 minimum completed epochs is malformed")
    return value


def patience_is_armed(config, *, epoch, step, end):
    return epoch >= minimum_completed_epochs(config) and step == end


def write_predictions(path, rows):
    if not rows:
        raise CampaignFailure("Spec 0039 selected validation produced no predictions")
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_branch(
    torch,
    budget_root,
    branches,
    *,
    cursor,
    order_hash,
    access_hashes,
    config_hash,
):
    manifests = {}
    for name, _device in BRANCHES:
        branch = branches[name]
        state = cpu_state(torch, branch["model"])
        manifest = {
            "branch": name,
            "cursor": cursor,
            "state_sha256": state_sha256(state),
            "order_sha256": order_hash,
            "access_sha256": {
                phase: access_hashes[phase][name] for phase in ("train", "validation")
            },
            "config_sha256": config_hash,
            "best": branch["best"],
            "patience_exhausted": branch["patience_exhausted"],
        }
        save_checkpoint(
            torch,
            budget_root / name / "latest.pt",
            state=state,
            manifest=manifest,
        )
        manifests[name] = manifest
    atomic_json(
        budget_root / "branch_progress.json",
        {
            "schema_version": "spec0039.branch_boundary.v1",
            "cursor": cursor,
            "order_sha256": order_hash,
            "access_sha256": access_hashes,
            "branches": manifests,
        },
    )


def run_budget(
    torch,
    root,
    config,
    *,
    source_roots,
    initial_state,
    budget,
):
    from eqvae.data.supervised_latents import SupervisedLatentStore, TissueDataset
    from eqvae.models.supervised import TissueClassifier
    from eqvae.training.supervised_pairing import (
        full_training_learning_rate,
        half_epoch_boundaries,
    )

    key = str(budget)
    geometry = config["budgets"][key]
    train_name = f"tissue/tissue_train_{budget:04d}_per_class.csv"
    validation_name = (
        f"tissue/tissue_validation_{geometry['validation_per_class']:04d}_per_class.csv"
    )
    budget_root = WORK_ROOT / f"budget_{budget:04d}_per_class"
    budget_root.mkdir(parents=True, exist_ok=False)
    config_hash = canonical_sha256(config)
    branches = {}
    datasets = {}
    for name, device_index in BRANCHES:
        device = f"cuda:{device_index}"
        store = SupervisedLatentStore(
            catalog_path=root / "physical_parts.csv",
            model_name=name,
            source_roots=source_roots,
        )
        datasets[name] = {
            "train": TissueDataset(path=root / train_name, split="train", store=store),
            "validation": TissueDataset(
                path=root / validation_name,
                split="validation",
                store=store,
            ),
        }
        branches[name] = make_branch(
            torch,
            TissueClassifier,
            name=name,
            device=device,
            initial_state=initial_state,
            optimizer_config=config["optimizer"],
        )
    initial_hashes = {
        name: state_sha256(cpu_state(torch, branches[name]["model"]))
        for name, _ in BRANCHES
    }
    if (
        len(set(initial_hashes.values())) != 1
        or next(iter(initial_hashes.values())) != config["initial_state_sha256"]
    ):
        raise CampaignFailure(
            "Spec 0039 branches do not start from the sealed paired state",
        )
    reference_branch = BRANCHES[0][0]
    calibration_batches = epoch_batches(
        datasets[reference_branch]["train"],
        batch_size=geometry["batch_size"],
        epoch=0,
        seed=config["initialization_seed"],
    )[: config["runtime"]["calibration_batches"]]
    if len(calibration_batches) != config["runtime"]["calibration_batches"]:
        raise CampaignFailure("Spec 0039 lacks three static calibration batches")
    for name, _device in BRANCHES:
        calibrate_branch(
            torch,
            branches[name],
            datasets[name]["train"],
            calibration_batches=calibration_batches,
            initial_state=initial_state,
            config=config,
        )
    check_index = 0
    update_index = 0
    order_hashes = []
    access_transcript = []
    stopped_early = False
    minimum_epochs = minimum_completed_epochs(config)
    with InlineExecutor() as executor:
        for epoch in range(1, config["maximum_epochs"] + 1):
            batches = epoch_batches(
                datasets[reference_branch]["train"],
                batch_size=geometry["batch_size"],
                epoch=epoch,
                seed=config["initialization_seed"],
            )
            if len(batches) != geometry["steps_per_epoch"]:
                raise CampaignFailure("Spec 0039 epoch geometry differs")
            flattened = [item for batch in batches for item in batch]
            order_hash = canonical_sha256(flattened)
            order_hashes.append({"epoch": epoch, "sha256": order_hash})
            train_access = {name: hashlib.sha256() for name, _ in BRANCHES}
            half, end = half_epoch_boundaries(geometry["steps_per_epoch"])
            for step, indices in enumerate(batches, start=1):
                update_index += 1
                learning_rate = full_training_learning_rate(
                    peak=config["optimizer"]["peak_lr"],
                    successful_update=update_index,
                    steps_per_epoch=geometry["steps_per_epoch"],
                    maximum_epochs=config["maximum_epochs"],
                    minimum_ratio=config["schedule"]["minimum_ratio"],
                )
                updates = [
                    executor.submit(
                        train_branch_step,
                        torch,
                        branches[name],
                        datasets[name]["train"],
                        indices,
                        learning_rate,
                        train_access[name],
                        epoch=epoch,
                        step=step,
                        update_index=update_index,
                    )
                    for name, _ in BRANCHES
                ]
                for update in updates:
                    update.result()
                if step not in {half, end}:
                    continue
                check_index += 1
                cursor = {
                    "epoch": epoch,
                    "step": step,
                    "update_index": update_index,
                    "check": check_index,
                }
                validation_access = {name: hashlib.sha256() for name, _ in BRANCHES}
                validations = {
                    name: executor.submit(
                        validation_metrics,
                        torch,
                        branches[name],
                        datasets[name]["validation"],
                        config["validation_batch_size"],
                        access_digest=validation_access[name],
                    )
                    for name, _ in BRANCHES
                }
                for name, _device in BRANCHES:
                    branch = branches[name]
                    metrics, _ = validations[name].result()
                    metrics = {
                        **metrics,
                        **cursor,
                        "scaler_scale": float(branch["scaler"].get_scale()),
                        "validation_access_sha256": validation_access[name].hexdigest(),
                    }
                    branch["history"]["validation"].append(metrics)
                    if improves(metrics, branch["best"]):
                        branch["best"] = metrics
                        branch["best_state"] = cpu_state(torch, branch["model"])
                        branch["last_improvement_check"] = check_index
                        save_checkpoint(
                            torch,
                            budget_root / name / "best.pt",
                            state=branch["best_state"],
                            manifest={
                                "branch": name,
                                "selection": metrics,
                                "config_sha256": config_hash,
                            },
                        )
                    branch["patience_exhausted"] = (
                        patience_is_armed(config, epoch=epoch, step=step, end=end)
                        and check_index - branch["last_improvement_check"]
                        >= config["patience_checks"]
                    )
                access_transcript.append({
                    **cursor,
                    "train_access_sha256": {
                        name: train_access[name].hexdigest() for name, _ in BRANCHES
                    },
                    "validation_access_sha256": {
                        name: validation_access[name].hexdigest()
                        for name, _ in BRANCHES
                    },
                })
                checkpoint_branch(
                    torch,
                    budget_root,
                    branches,
                    cursor=cursor,
                    order_hash=order_hash,
                    access_hashes={
                        "train": {
                            name: train_access[name].hexdigest() for name, _ in BRANCHES
                        },
                        "validation": {
                            name: validation_access[name].hexdigest()
                            for name, _ in BRANCHES
                        },
                    },
                    config_hash=config_hash,
                )
                atomic_json(
                    budget_root / "history.json",
                    {name: branches[name]["history"] for name, _ in BRANCHES},
                )
                if branches[reference_branch]["patience_exhausted"]:
                    stopped_early = True
                    break
            if stopped_early:
                break
    selected_access = {name: hashlib.sha256() for name, _ in BRANCHES}
    for name, _device in BRANCHES:
        branch = branches[name]
        final_state = cpu_state(torch, branch["model"])
        save_checkpoint(
            torch,
            budget_root / name / "final.pt",
            state=final_state,
            manifest={
                "branch": name,
                "final_update_index": update_index,
                "config_sha256": config_hash,
            },
        )
        if branch["best_state"] is None:
            raise CampaignFailure(
                "Spec 0039 branch lacks a selected validation checkpoint",
            )
        branch["model"].load_state_dict(branch["best_state"])
    with InlineExecutor() as executor:
        selected_validation = {
            name: executor.submit(
                validation_metrics,
                torch,
                branches[name],
                datasets[name]["validation"],
                config["validation_batch_size"],
                access_digest=selected_access[name],
                capture_predictions=True,
            )
            for name, _ in BRANCHES
        }
        for name, _device in BRANCHES:
            branch = branches[name]
            selected, predictions = selected_validation[name].result()
            if (
                abs(selected["macro_f1"] - branch["best"]["macro_f1"]) > 1e-6
                or abs(
                    selected["mean_cross_entropy"]
                    - branch["best"]["mean_cross_entropy"],
                )
                > 1e-6
            ):
                raise CampaignFailure(
                    "Spec 0039 best checkpoint cannot reproduce selection metrics",
                )
            write_predictions(
                budget_root / name / "selected_validation_predictions.csv.gz",
                predictions,
            )
    result = {
        "budget_per_class": budget,
        "train_rows": geometry["total_rows"],
        "validation_rows": geometry["validation_rows"],
        "batch_size": geometry["batch_size"],
        "steps_per_epoch": geometry["steps_per_epoch"],
        "minimum_completed_epochs": minimum_epochs,
        "scheduled_updates": update_index,
        "stopped_early": stopped_early,
        "order_hashes": order_hashes,
        "access_transcript": access_transcript,
        "selected_validation_access_sha256": {
            name: selected_access[name].hexdigest() for name, _ in BRANCHES
        },
        "configuration_sha256": config_hash,
        "initial_state_sha256": config["initial_state_sha256"],
        "branches": {
            name: {
                "best": branches[name]["best"],
                "calibration": branches[name]["history"]["calibration"],
                "patience_exhausted": branches[name]["patience_exhausted"],
            }
            for name, _ in BRANCHES
        },
    }
    atomic_json(budget_root / "result.json", result)
    return result


def validate_initial_state(torch, root, config):
    loaded = torch.load(
        root / "tissue_initial_state.pt",
        map_location="cpu",
        weights_only=True,
    )
    if (
        loaded.get("schema_version") != "spec0039.tissue_initial_state.v1"
        or loaded.get("seed") != config["initialization_seed"]
    ):
        raise CampaignFailure("Spec 0039 serialized initial state differs")
    initial_state = loaded.get("state_dict")
    if (
        not isinstance(initial_state, dict)
        or state_sha256(initial_state) != config["initial_state_sha256"]
    ):
        raise CampaignFailure("Spec 0039 initial state payload differs")
    return initial_state


def run_all_budgets(
    torch,
    root,
    config,
    *,
    source_roots,
    initial_state,
):
    return [
        run_budget(
            torch,
            root,
            config,
            source_roots=source_roots,
            initial_state=initial_state,
            budget=budget,
        )
        for budget in sorted(int(key) for key in config["budgets"])
    ]


def _worker_result(branch):
    return {
        "schema_version": "spec0039.branch_summary.v1",
        "branch": branch,
        "status": "failed",
        "spec": "0039",
        "scope": "tissue_development_training_no_sealed_test",
        "phase": "resolve_input",
        "input_dataset": f"{INPUT_DATASET_REFERENCE}/1",
        "kernel_sources": [f"{source}/1" for source in KERNEL_SOURCES],
        "input_contract_sha256": INPUT_CONTRACT_SHA256,
        "started_unix": time.time(),
    }


def _run_worker(args):
    global BRANCHES, OUTPUT_PATH, WORK_ROOT

    branch = args.worker
    BRANCHES = ((branch, 0),)
    WORK_ROOT = Path(args.branch_root)
    OUTPUT_PATH = WORK_ROOT / "branch_summary.json"
    WORK_ROOT.mkdir(parents=True, exist_ok=False)
    result = _worker_result(branch)
    try:
        root, contract, config = resolve_input_bundle()
        result["configuration_sha256"] = canonical_sha256(config)
        result["minimum_completed_epochs"] = minimum_completed_epochs(config)
        result["phase"] = "validate_runtime"
        import torch

        validate_pinned_torch(torch)
        if torch.cuda.device_count() != 1 or "T4" not in torch.cuda.get_device_name(0):
            raise CampaignFailure("A Spec 0039 child must see exactly its assigned T4")
        torch.cuda.set_device(0)
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(root / "src"))
        result["runtime"] = {
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "flags": configure_torch(torch),
        }
        source_roots = resolve_source_roots(
            root / "physical_parts.csv",
            contract["physical_sources"],
        )
        initial_state = validate_initial_state(torch, root, config)
        result["phase"] = "training"
        result["budgets"] = run_all_budgets(
            torch,
            root,
            config,
            source_roots=source_roots,
            initial_state=initial_state,
        )
        result["status"] = "complete"
    except BaseException:
        result["traceback"] = traceback.format_exc()
        print(result["traceback"], file=sys.stderr, flush=True)
    finally:
        result["finished_unix"] = time.time()
        result["elapsed_seconds"] = result["finished_unix"] - result["started_unix"]
        atomic_json(OUTPUT_PATH, result)
        print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["status"] == "complete" else 1


def _launch_children():
    processes = {}
    for branch, device in BRANCHES:
        branch_root = WORK_ROOT / "branches" / branch
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(device)
        cache_root = Path(tempfile.gettempdir()) / "spec0039_compile_cache" / branch
        environment["TORCHINDUCTOR_CACHE_DIR"] = str(cache_root / "inductor")
        environment["TRITON_CACHE_DIR"] = str(cache_root / "triton")
        environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            branch,
            "--branch-root",
            str(branch_root),
        ]
        processes[branch] = subprocess.Popen(command, env=environment)
    return {branch: process.wait() for branch, process in processes.items()}


def _aggregate(return_codes, config):
    summaries = {}
    expected_budgets = sorted(int(key) for key in config["budgets"])
    for branch, _device in BRANCHES:
        path = WORK_ROOT / "branches" / branch / "branch_summary.json"
        summary = (
            read_object(path)
            if path.is_file()
            else {
                "branch": branch,
                "status": "failed",
                "error": "child emitted no branch summary",
            }
        )
        if summary.get("branch") != branch:
            raise CampaignFailure("Spec 0039 child branch identity differs")
        summaries[branch] = summary
    statuses = {branch: summary.get("status") for branch, summary in summaries.items()}
    result = {
        "schema_version": "spec0039.overall_status.v1",
        "spec": "0039",
        "scope": "tissue_development_training_no_sealed_test",
        "status": "failed",
        "configuration_sha256": canonical_sha256(config),
        "minimum_completed_epochs": minimum_completed_epochs(config),
        "branches": summaries,
        "child_return_codes": return_codes,
    }
    if set(statuses.values()) != {"complete"} or any(return_codes.values()):
        return result
    budget_records = {}
    for branch, summary in summaries.items():
        records = summary.get("budgets")
        if not isinstance(records, list):
            raise CampaignFailure("Spec 0039 complete child lacks budget results")
        mapped = {row.get("budget_per_class"): row for row in records}
        if sorted(mapped) != expected_budgets:
            raise CampaignFailure("Spec 0039 child budget matrix differs")
        budget_records[branch] = mapped
    merged = []
    for budget in expected_budgets:
        rows = {branch: budget_records[branch][budget] for branch, _ in BRANCHES}
        reference = rows[BRANCHES[0][0]]
        for branch, row in rows.items():
            shared_epochs = min(
                len(row["order_hashes"]),
                len(reference["order_hashes"]),
            )
            if (
                row["train_rows"] != reference["train_rows"]
                or row["validation_rows"] != reference["validation_rows"]
                or row["batch_size"] != reference["batch_size"]
                or row["steps_per_epoch"] != reference["steps_per_epoch"]
                or row["minimum_completed_epochs"]
                != reference["minimum_completed_epochs"]
                or row["order_hashes"][:shared_epochs]
                != reference["order_hashes"][:shared_epochs]
                or row["initial_state_sha256"] != reference["initial_state_sha256"]
            ):
                raise CampaignFailure(
                    f"Spec 0039 branch protocol differs at budget {budget}: {branch}",
                )
        merged.append({
            **{
                key: reference[key]
                for key in (
                    "budget_per_class",
                    "train_rows",
                    "validation_rows",
                    "batch_size",
                    "steps_per_epoch",
                    "minimum_completed_epochs",
                    "initial_state_sha256",
                )
            },
            "branches": {
                branch: {
                    **rows[branch]["branches"][branch],
                    "scheduled_updates": rows[branch]["scheduled_updates"],
                    "stopped_early": rows[branch]["stopped_early"],
                    "order_hashes": rows[branch]["order_hashes"],
                }
                for branch, _ in BRANCHES
            },
        })
    result["budgets"] = merged
    result["aggregate_label_efficiency"] = [
        {
            "budget_per_class": row["budget_per_class"],
            **{
                f"{branch}_best_macro_f1": row["branches"][branch]["best"]["macro_f1"]
                for branch, _ in BRANCHES
            },
        }
        for row in merged
    ]
    result["status"] = "complete"
    return result


def _run_parent():
    started = time.time()
    root, contract, config = resolve_input_bundle()
    install_pinned_torch()
    import torch

    validate_pinned_torch(torch)
    if torch.cuda.device_count() != 2 or any(
        "T4" not in torch.cuda.get_device_name(index) for index in range(2)
    ):
        raise CampaignFailure("Spec 0039 parent requires exactly two Tesla T4 GPUs")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root / "src"))
    source_roots = resolve_source_roots(
        root / "physical_parts.csv",
        contract["physical_sources"],
    )
    validate_initial_state(torch, root, config)
    WORK_ROOT.mkdir(parents=True, exist_ok=False)
    atomic_json(
        WORK_ROOT / "run_contract.json",
        {
            "schema_version": "spec0039.process_dispatch.v1",
            "input_dataset": f"{INPUT_DATASET_REFERENCE}/1",
            "input_contract_sha256": INPUT_CONTRACT_SHA256,
            "configuration_sha256": canonical_sha256(config),
            "minimum_completed_epochs": minimum_completed_epochs(config),
            "kernel_sources": [f"{source}/1" for source in KERNEL_SOURCES],
            "branch_devices": dict(BRANCHES),
            "source_roots": {name: str(path) for name, path in source_roots.items()},
            "started_unix": started,
        },
    )
    return_codes = _launch_children()
    overall = _aggregate(return_codes, config)
    overall["started_unix"] = started
    overall["finished_unix"] = time.time()
    overall["elapsed_seconds"] = overall["finished_unix"] - started
    atomic_json(OUTPUT_PATH, overall)
    print(json.dumps(overall, sort_keys=True), flush=True)
    return 0 if overall["status"] == "complete" else 1


def _parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=tuple(name for name, _ in BRANCHES))
    parser.add_argument("--branch-root")
    return parser


def main():
    args = _parser().parse_args()
    if args.worker:
        if not args.branch_root:
            raise CampaignFailure("A Spec 0039 worker requires --branch-root")
        return _run_worker(args)
    try:
        return _run_parent()
    except BaseException:
        failure = {
            "schema_version": "spec0039.overall_status.v1",
            "status": "failed",
            "spec": "0039",
            "scope": "tissue_development_training_no_sealed_test",
            "traceback": traceback.format_exc(),
            "finished_unix": time.time(),
        }
        WORK_ROOT.mkdir(parents=True, exist_ok=True)
        atomic_json(OUTPUT_PATH, failure)
        print(failure["traceback"], file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
