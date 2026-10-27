"""First paired Spec 0054 fold, reading complete FP16 WSI bags in place."""

from __future__ import annotations

import argparse
import copy
import ctypes
import hashlib
import json
import math
import mmap
import os
import random
import shutil
import subprocess
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from pathlib import Path
from threading import Lock, local

import numpy as np
import torch
from torch.nn import functional

from experiments.spec0054_shard_bags import RECORD_BYTES, ShardBags
from eqvae.models.gated_abmil import GatedABMILClassifier
from eqvae.training.mil_dynamics import (
    AdamWTelemetry, EagerLayerProbe, ExampleDynamicsTracker,
    parameter_optimizer_records, t0_forward_records, t0_forward_summary,
)
from eqvae.training.mil_dynamics_checkpoint import (
    build_dynamics_checkpoint, load_dynamics_checkpoint,
    restore_dynamics_checkpoint, save_dynamics_checkpoint,
    should_pause_for_session,
)
from eqvae.training.mil_dynamics_step import classification_example_record
from eqvae.training.mil_t2_lite import flatten_t2_records, run_abmil_t2_lite
from eqvae.training.mil_telemetry_io import load_telemetry_table, write_telemetry_table
from eqvae.training.supervised_pairing import paired_epoch_order

BRANCHES = ("normal_vae", "so2_vae")


def _make_classifier(state: dict, device: torch.device, *, hidden_width: int = 128,
                     cnn_dropout: float = 0.0, mlp_dropout: float = 0.0,
                     bag_dropout: float = 0.0):
    model = GatedABMILClassifier(hidden_width, cnn_dropout=cnn_dropout,
                                 mlp_dropout=mlp_dropout, bag_dropout=bag_dropout)
    model.load_state_dict(state)
    model.to(device)
    # PyTorch supports memory_format here; its Module.to overload omits it.
    return model.to(memory_format=torch.channels_last).train()  # pyright: ignore[reportCallIssue]


def _make_optimizer(model, *, peak_lr: float = 2e-4, weight_decay: float = 5e-3):
    decay = [parameter for parameter in model.parameters() if parameter.ndim >= 2]
    no_decay = [parameter for parameter in model.parameters() if parameter.ndim < 2]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=peak_lr, betas=(0.9, 0.999), eps=1e-8,
        fused=next(model.parameters()).is_cuda,
    )


def _forward_loss(model, latents, target, class_weight, aem_weight: torch.Tensor | float = 0.0,
                  label_smoothing: float = 0.0):
    with torch.autocast(latents.device.type, dtype=torch.float16, enabled=latents.is_cuda):
        logits, representations = model.forward_with_representations(latents)
        loss = functional.cross_entropy(logits.float().unsqueeze(0), target)
        training_ce = functional.cross_entropy(
            logits.float().unsqueeze(0), target, label_smoothing=label_smoothing)
        summary = t0_forward_summary(representations)
        attention = representations[-3].float()
        entropy = -(attention * attention.clamp_min(1e-12).log()).sum()
    return logits, loss, training_ce * class_weight - aem_weight * entropy, summary


def _compiled_closure(model, *, label_smoothing: float = 0.0):
    def closure(latents, target, class_weight, aem_weight):
        return _forward_loss(model, latents, target, class_weight, aem_weight, label_smoothing)
    compiled = torch.compile(
        closure, backend="inductor", fullgraph=True, dynamic=None,
        mode="max-autotune-no-cudagraphs",
    )
    def forward(latents, target, class_weight, aem_weight: torch.Tensor | float = 0.0):
        torch._dynamo.maybe_mark_dynamic(latents, 0)
        return compiled(latents, target, class_weight, aem_weight)
    return forward


def _aem_weight(exposure: int, train_count: int, config: dict) -> float:
    """Keep entropy regularization constant throughout the selected recipe."""
    return config["aem_weight"]


def _sample_indices(count: int, seed: int, epoch: int, wsi_id: int, fraction: float) -> np.ndarray:
    generator = np.random.default_rng(np.random.SeedSequence([seed, epoch, wsi_id]))
    return np.sort(generator.choice(count, size=math.ceil(count * fraction), replace=False))


def _sample_c4(count: int, seed: int, epoch: int, wsi_id: int) -> np.ndarray:
    # Independent stream: augmentation never changes the paired patch subset.
    generator = np.random.default_rng(np.random.SeedSequence([seed, epoch, wsi_id, 0xC4]))
    return generator.integers(0, 4, size=count, dtype=np.uint8)


def _apply_c4(array: np.ndarray, rotations: np.ndarray) -> None:
    """Rotate owned patch buffers exactly on their last two spatial axes."""
    for quarter_turns in (1, 2, 3):
        selected = np.flatnonzero(rotations == quarter_turns)
        array[selected] = np.rot90(array[selected], quarter_turns, axes=(-2, -1))


def _capture_modules(model):
    # All parameter/activation layers plus the terminal patch-vector capture.
    # The attention container returns a tuple and is observed through its layers.
    return {
        name: module for name, module in model.named_modules()
        if name and (not list(module.children()) or name == "patch_encoder")
    }


def _forward_records(summary):
    return t0_forward_records(summary, capture_names=GatedABMILClassifier.capture_names)


class ExposureSchedule:
    """Exposure warmup/cosine, or a bounded committed-update LR sweep."""

    def __init__(self, optimizer: torch.optim.Optimizer, train_count: int, config: dict) -> None:
        self.optimizer = optimizer
        self.train_count = train_count
        self.config = config
        self.last_exposure = 0
        self.last_assessment_exposure = -1

    def set_next(self, exposure: int, *, next_update: int = 1) -> float:
        if self.config.get("run_mode") == "lr_search":
            search = self.config["lr_search"]
            # Index committed updates, so tail batches, AMP retries and
            # checkpoint restoration retain the same LR.
            sweep_index = max(next_update - search["preparation_updates"] - 1, 0)
            progress = min(sweep_index / (search["sweep_updates"] - 1), 1.0)
            lr = search["min_lr"] * (search["max_lr"] / search["min_lr"]) ** progress
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            return lr
        warmup = self.config["warmup_epochs"] * self.train_count
        horizon = self.config["epochs"] * self.train_count
        peak = self.config["peak_lr"]
        if exposure <= warmup:
            lr = peak * (0.1 + 0.9 * (exposure - 1) / (warmup - 1))
        elif self.config.get("run_mode") == "lr_confirmation":
            lr = peak
        else:
            progress = (exposure - warmup) / (horizon - warmup)
            minimum = self.config["min_lr"]
            lr = minimum + (peak - minimum) * (1 + math.cos(math.pi * progress)) / 2
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def state_dict(self) -> dict:
        return {"last_exposure": self.last_exposure, "last_assessment_exposure": self.last_assessment_exposure}

    def load_state_dict(self, state_dict: dict) -> None:
        self.last_exposure = int(state_dict["last_exposure"])
        self.last_assessment_exposure = int(state_dict["last_assessment_exposure"])


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _head_gradient_norms(logits: torch.Tensor, target: int, embedding: torch.Tensor) -> tuple[float, float]:
    """Exact unweighted CE head-gradient norms from a fixed-boundary forward."""
    residual = torch.softmax(logits.detach().float(), dim=0)
    residual[target] -= 1
    bias_norm = float(residual.norm().item())
    weight_norm = float((residual.norm() * embedding.detach().float().norm()).item())
    return bias_norm, weight_norm


def _gradient_noise_summary(gradients: torch.Tensor) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
    """Within-sample gradient energy and conflict on one fixed train-only panel."""
    gram = gradients.double() @ gradients.double().T
    norms = gram.diag().clamp_min(0).sqrt()
    denominator = norms[:, None] * norms[None, :]
    cosines = torch.where(denominator > 0, gram / denominator.clamp_min(1e-12), torch.nan)
    upper = torch.triu_indices(len(gradients), len(gradients), offset=1, device=gradients.device)
    pairs = cosines[upper[0], upper[1]]
    valid_pairs = pairs[torch.isfinite(pairs)]
    g2 = gram.diag().mean()
    mean_sq = gram.mean()
    variance = (g2 - mean_sq).clamp_min(0)
    defined = bool(mean_sq > 0)
    record = {
        "mean_gradient_squared_norm": float(mean_sq.item()),
        "mean_individual_squared_norm": float(g2.item()),
        "gradient_variance": float(variance.item()),
        "noise_scale_defined": float(defined),
        "gradient_noise_scale": float((variance / mean_sq).item()) if defined else float("nan"),
        "pairwise_defined_fraction": float(torch.isfinite(pairs).float().mean().item()),
        "pairwise_cosine_mean": float(valid_pairs.mean().item()) if valid_pairs.numel() else float("nan"),
        "pairwise_conflict_fraction": float((valid_pairs < 0).float().mean().item()) if valid_pairs.numel() else float("nan"),
    }
    return record, norms, cosines


def _restore_rows(path: Path, update: int) -> list[dict]:
    if not path.exists():
        return []
    columns = load_telemetry_table(path, through_update=update)
    return [
        {name: value.item() for name, value in zip(columns, values, strict=True)}
        for values in zip(*columns.values(), strict=True)
    ]


def _write_rows(directory: Path, tables: dict[str, list[dict]]) -> None:
    for name, rows in tables.items():
        if rows:
            write_telemetry_table(directory / f"{name}.npz", rows)


def _at_epoch(exposure: int, train_count: int, early: list[int], every: int) -> bool:
    return exposure > 0 and exposure % train_count == 0 and (
        exposure // train_count in early or exposure // train_count % every == 0
    )


def _boundaries(train_count: int, config: dict) -> set[int]:
    schedule = {0, train_count // 2}
    schedule |= {epoch * train_count for epoch in config["evaluation_early_epochs"]}
    schedule |= {
        epoch * train_count
        for epoch in range(config["evaluation_every_epochs"], config["epochs"] + 1, config["evaluation_every_epochs"])
    }
    return schedule


def run_branch(
    branch: str, bags: ShardBags, config: dict, output_root: Path,
    resume_root: Path | None, identity: dict, session_started: float,
    device: torch.device = torch.device("cuda:0"),
    stop_after_updates: int | None = None,
    session_deadline_unix: float | None = None,
) -> bool:
    torch.backends.cudnn.benchmark = True
    directory = output_root / branch
    directory.mkdir(parents=True, exist_ok=False)
    events = (directory / "events.jsonl").open("a", buffering=1)
    log_lock = Lock()

    def log(event: str, **details) -> None:
        candidate = {"candidate_peak_lr": config["peak_lr"]} if config.get("run_mode") == "lr_confirmation" else {}
        line = json.dumps({"branch": branch, "event": event, **candidate, **details}, sort_keys=True)
        with log_lock:
            print(line, flush=True)
            print(line, file=events, flush=True)

    def session_ending() -> bool:
        if session_deadline_unix is not None:
            return time.time() >= session_deadline_unix
        return should_pause_for_session(session_started_unix=session_started, now_unix=time.time())

    log("worker_start", physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
        device=str(device), device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        session_deadline_unix=session_deadline_unix)
    log("training_recipe", resume=resume_root is not None,
        epochs=config["epochs"], warmup_epochs=config["warmup_epochs"],
        peak_lr=config["peak_lr"], min_lr=config["min_lr"],
        effective_batch=config["effective_batch"], weight_decay=config["weight_decay"],
        patch_retention=config["patch_retention"], aem_weight=config["aem_weight"],
        aem_schedule=config["aem_schedule"],
        cnn_dropout=config["cnn_dropout"], mlp_dropout=config["mlp_dropout"],
        bag_dropout=config["bag_dropout"], label_smoothing=config["label_smoothing"],
        c4_augmentation=config["c4_augmentation"],
        checkpoint_every_exposures=config["checkpoint_every_exposures"])
    prior = resume_root / branch if resume_root is not None else None
    if prior is not None and prior.exists():
        if json.loads((prior / "identity.json").read_text()) != {**identity, "branch": branch}:
            raise ValueError("Resume identity differs")
        for name in ("latest.pt", "best.pt", "t0.npz", "t1.npz", "t2.npz", "predictions.npz"):
            if (prior / name).exists():
                shutil.copy2(prior / name, directory / name)
    (directory / "identity.json").write_text(json.dumps({**identity, "branch": branch}, sort_keys=True) + "\n")
    ids = sorted(bags.locations)
    train_ids = [wsi_id for wsi_id in ids if bags.locations[wsi_id].fold != config["fold"]]
    validation_ids = [wsi_id for wsi_id in ids if bags.locations[wsi_id].fold == config["fold"]]
    count = len(train_ids)
    lr_search = config.get("run_mode") == "lr_search"
    lr_confirmation = config.get("run_mode") == "lr_confirmation"
    counts = Counter(bags.locations[wsi_id].diagnosis_index for wsi_id in train_ids)
    weights = [count / (5 * counts[index]) for index in range(5)]
    covariates = {}
    for wsi_id in ids:
        coordinates = np.asarray(bags.locations[wsi_id].coordinates, dtype=np.int64) // 256
        extent = coordinates.max(axis=0) - coordinates.min(axis=0) + 1
        covariates[wsi_id] = {
            "lattice_occupancy": float(bags.locations[wsi_id].count / extent.prod()),
        }
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    initial_state = {name: value.detach().cpu().clone() for name, value in GatedABMILClassifier(config["attention_hidden_width"]).state_dict().items()}
    model = _make_classifier(initial_state, device, hidden_width=config["attention_hidden_width"],
                             cnn_dropout=config["cnn_dropout"], mlp_dropout=config["mlp_dropout"],
                             bag_dropout=config["bag_dropout"])
    optimizer = _make_optimizer(model, peak_lr=config["peak_lr"], weight_decay=config["weight_decay"])
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    telemetry = AdamWTelemetry()
    dynamics = ExampleDynamicsTracker(ids)
    scheduler = ExposureSchedule(optimizer, count, config)
    compiled = _compiled_closure(model, label_smoothing=config["label_smoothing"])
    tables = {name: [] for name in ("t0", "t1", "t2", "predictions")}
    epoch = 0
    cursor = 0
    update = 0
    exposure = 0
    order = paired_epoch_order(row_count=count, epoch=epoch, seed=config["seed"])
    checkpoint = directory / "latest.pt"
    resumed = checkpoint.exists()
    if resumed:
        progress = restore_dynamics_checkpoint(
            load_dynamics_checkpoint(checkpoint), model=model, optimizer=optimizer,
            scaler=scaler, scheduler=scheduler, telemetry=telemetry, dynamics=dynamics,
        )
        epoch, cursor = progress.epoch, progress.within_epoch_cursor
        update, exposure = progress.committed_update, progress.exposure_count
        order = progress.current_order
        if progress.effective_batch_size != config["effective_batch"]:
            raise ValueError("Resume effective batch differs")
        if order != paired_epoch_order(row_count=count, epoch=epoch, seed=config["seed"]):
            raise ValueError("Resume order differs from paired schedule")
        if exposure != epoch * count + cursor or scheduler.last_exposure != exposure:
            raise ValueError("Resume cursor/exposure differs")
        tables = {name: _restore_rows(directory / f"{name}.npz", update) for name in tables}
        if scheduler.last_assessment_exposure < exposure:
            for name in ("t1", "t2", "predictions"):
                tables[name] = [row for row in tables[name] if row["exposures"] < exposure]
        log("resume_done", epoch=epoch, cursor=cursor, update=update, exposures=exposure,
            last_assessment_exposure=scheduler.last_assessment_exposure)
    if lr_search:
        log("lr_search_start", train_wsi=count, effective_batch=config["effective_batch"],
            architecture=config["architecture"], weight_decay=config["weight_decay"],
            parameter_count=sum(parameter.numel() for parameter in model.parameters()),
            aem_weight=config["aem_weight"], **config["lr_search"])
    if lr_confirmation:
        log("lr_confirmation_start", train_wsi=count, effective_batch=config["effective_batch"],
            peak_lr=config["peak_lr"], epochs=config["lr_confirmation"]["epochs"],
            warmup_epochs=config["warmup_epochs"], aem_horizon_epochs=config["epochs"],
            architecture=config["architecture"], weight_decay=config["weight_decay"],
            parameter_count=sum(parameter.numel() for parameter in model.parameters()))

    def load(wsi_id: int, indices: np.ndarray | None = None) -> torch.Tensor:
        log("bag_read_start", wsi_id=wsi_id, shard=bags.locations[wsi_id].shard,
            bag_size=bags.locations[wsi_id].count)
        source = bags.read(branch, wsi_id)
        array = source.copy() if indices is None else source[indices]
        log("bag_transfer_start", wsi_id=wsi_id, sampled_bag_size=len(array),
            sample_indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest() if indices is not None else "full")
        latents = torch.from_numpy(array).to(device=device, memory_format=torch.channels_last)
        log("bag_load_done", wsi_id=wsi_id)
        return latents

    @contextmanager
    def training_inputs():
        # Two ready inputs per VAE worker, across accumulation/epoch boundaries.
        # Its private cursor is speculative; only the training loop commits it.
        def requests():
            read_epoch, read_cursor, read_order = epoch, cursor, order
            read_horizon = config["lr_confirmation"]["epochs"] if lr_confirmation else config["epochs"]
            while read_epoch < read_horizon:
                for index in read_order[read_cursor:]:
                    wsi_id = train_ids[index]
                    indices = _sample_indices(bags.locations[wsi_id].count,
                                              config["seed"], read_epoch, wsi_id,
                                              config["patch_retention"])
                    yield read_epoch, wsi_id, indices
                read_epoch += 1
                read_cursor = 0
                read_order = paired_epoch_order(row_count=count, epoch=read_epoch,
                                               seed=config["seed"])

        # Open on the main thread so two first reads cannot race to open a shard.
        for shard, binary in bags.files[branch].items():
            key = (branch, shard)
            if key not in bags.descriptors:
                bags.descriptors[key] = os.open(binary, os.O_RDONLY)
                log("shard_open", shard=shard)
        log("bag_prefetch_start", reader_workers=2, future_wsi=2)
        reader_state = local()

        def prepare(request):
            read_epoch, wsi_id, indices = request
            started = time.perf_counter()
            log("bag_read_start", wsi_id=wsi_id, shard=bags.locations[wsi_id].shard,
                bag_size=bags.locations[wsi_id].count, prefetch=True,
                requested_bytes=len(indices) * RECORD_BYTES,
                read_regions=int(np.count_nonzero(np.diff(indices) != 1)) + 1)
            read_started, read_cpu_started = time.perf_counter(), time.thread_time()
            array = bags.read(branch, wsi_id, indices)
            read_seconds = time.perf_counter() - read_started
            read_cpu_seconds = time.thread_time() - read_cpu_started
            host_started, host_cpu_started = time.perf_counter(), time.thread_time()
            rotations = _sample_c4(len(indices), config["seed"], read_epoch, wsi_id)
            if config["c4_augmentation"]:
                _apply_c4(array, rotations)
            source = torch.from_numpy(array)
            host = torch.empty_like(source, memory_format=torch.channels_last,
                                    pin_memory=device.type == "cuda")
            host.copy_(source)
            host_seconds = time.perf_counter() - host_started
            host_cpu_seconds = time.thread_time() - host_cpu_started
            log("bag_transfer_start", wsi_id=wsi_id, sampled_bag_size=len(indices),
                sample_indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
                read_prepare_seconds=time.perf_counter() - started,
                read_seconds=read_seconds, read_cpu_seconds=read_cpu_seconds,
                host_prepare_seconds=host_seconds, host_prepare_cpu_seconds=host_cpu_seconds,
                c4_rotations_sha256=hashlib.sha256(rotations.tobytes()).hexdigest()
                if config["c4_augmentation"] else "identity", prefetch=True)
            transfer_started = time.perf_counter()
            if device.type == "cuda":
                if not hasattr(reader_state, "copy_stream"):
                    reader_state.copy_stream = torch.cuda.Stream(device=device)
                copy_stream = reader_state.copy_stream
                with torch.cuda.device(device), torch.cuda.stream(copy_stream):
                    latents = host.to(device, non_blocking=True)
                    ready = torch.cuda.Event()
                    ready.record(copy_stream)
                # Wait in the reader thread: keep pinned source alive until DMA
                # completes, while the main stream trains on the previous WSI.
                ready.synchronize()
            else:
                latents = host
            log("bag_load_done", wsi_id=wsi_id, prefetch=True,
                transfer_seconds=time.perf_counter() - transfer_started)
            return wsi_id, indices, latents

        plan = iter(requests())
        with ThreadPoolExecutor(max_workers=2) as reader:
            pending = deque()
            for _ in range(2):
                request = next(plan, None)
                if request is not None:
                    pending.append(reader.submit(prepare, request))

            def consume():
                while pending:
                    wait_started = time.perf_counter()
                    wsi_id, indices, latents = pending.popleft().result()
                    if device.type == "cuda":
                        latents.record_stream(torch.cuda.current_stream(device))
                    request = next(plan, None)
                    if request is not None:
                        pending.append(reader.submit(prepare, request))
                    log("bag_prefetch_consume", wsi_id=wsi_id,
                        wait_seconds=time.perf_counter() - wait_started)
                    yield wsi_id, indices, latents
                    del latents

            prepared = consume()
            try:
                yield prepared
            finally:
                # Drain before full-bag probes or closing shard descriptors.
                prepared.close()
                for future in pending:
                    future.result()

    def evaluate() -> bool:
        log("evaluation_start", exposures=exposure)
        model.eval()
        with torch.inference_mode():
            for split, selected in (("train", train_ids), ("validation", validation_ids)):
                for wsi_id in selected:
                    if session_ending():
                        model.train()
                        return False
                    log("evaluation_wsi_start", split=split, wsi_id=wsi_id)
                    latents = load(wsi_id)
                    target_index = bags.locations[wsi_id].diagnosis_index
                    with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                        logits, representations = model.forward_with_representations(latents)
                        embedding = representations[-2].detach().float()
                        loss = functional.cross_entropy(
                            logits.float().unsqueeze(0), torch.tensor([target_index], device=device)
                        )
                    record = classification_example_record(
                        logits=logits, target=target_index, unweighted_loss=loss,
                        weighted_loss=loss * weights[target_index],
                    )
                    record["dynamics_observation_defined"] = (
                        math.isfinite(float(record["true_class_margin"]))
                        and math.isfinite(float(record["true_probability"]))
                    )
                    if record["dynamics_observation_defined"]:
                        dynamics.update(
                            wsi_id=wsi_id, boundary=exposure,
                            correct=bool(record["correct"]),
                            margin=float(record["true_class_margin"]),
                            true_probability=float(record["true_probability"]),
                        )
                    else:
                        log("dynamics_observation_undefined", wsi_id=wsi_id, exposures=exposure)
                    cumulative = dynamics.record(wsi_id)
                    head_bias_norm, head_weight_norm = _head_gradient_norms(
                        logits, target_index, embedding,
                    )
                    embedding_values = embedding.cpu().tolist()
                    tables["predictions"].append({
                        "update": update, "exposures": exposure, "split": split,
                        "wsi_id": wsi_id, "bag_size": bags.locations[wsi_id].count,
                        **covariates[wsi_id], "class_weight": weights[target_index],
                        "head_gradient_bias_l2": head_bias_norm,
                        "head_gradient_weight_l2": head_weight_norm,
                        **{f"embedding_{index}": float(value) for index, value in enumerate(embedding_values)},
                        **record,
                        **{f"dynamics.{key}": value for key, value in cumulative.items() if key != "wsi_id"},
                    })
                    del latents, representations, embedding
                    log("evaluation_wsi_done", split=split, wsi_id=wsi_id)
        model.train()
        log("evaluation_done", exposures=exposure)
        return True

    def diagnostics(force: bool = False) -> bool:
        log("diagnostics_start", exposures=exposure)
        if session_ending():
            return False
        if force or exposure == 0 or _at_epoch(exposure, count, config["t1_early_epochs"], config["t1_every_epochs"]):
            wsi_id = int(config["t1_train_wsi"])
            latents = load(wsi_id)
            target_index = bags.locations[wsi_id].diagnosis_index
            target = torch.tensor([target_index], device=device)
            def diagnostic_loss(probe: GatedABMILClassifier) -> torch.Tensor:
                return _forward_loss(probe, latents, target, weights[target_index],
                                     _aem_weight(exposure, count, config), config["label_smoothing"])[2]
            log("t1_start", wsi_id=wsi_id)
            probe_model = copy.deepcopy(model).train()
            probe_optimizer = _make_optimizer(probe_model, peak_lr=config["peak_lr"], weight_decay=config["weight_decay"])
            probe_optimizer.load_state_dict(optimizer.state_dict())
            probe_model.zero_grad(set_to_none=True)
            with EagerLayerProbe(_capture_modules(probe_model)) as probe:
                diagnostic_loss(probe_model).backward()
                layer_records = probe.records()
            parameter_records = parameter_optimizer_records(
                probe_model, probe_optimizer, last_updates=telemetry.previous_updates,
            )
            for kind, records in (("layer", layer_records), ("parameter", parameter_records)):
                for capture, values in records.items():
                    for metric, value in values.items():
                        tables["t1"].append({
                            "update": update, "exposures": exposure, "wsi_id": wsi_id,
                            "kind": kind, "capture_name": capture,
                            "metric_name": metric, "value": value,
                        })
            del probe_model, probe_optimizer, latents
            log("t1_done", wsi_id=wsi_id)
        if force or exposure == 0 or _at_epoch(exposure, count, config["t2_early_epochs"], config["t2_every_epochs"]):
            sentinel_ids = (int(config["t1_train_wsi"]),) if stop_after_updates is not None else (
                int(config["t1_train_wsi"]), int(config["t2_holdout_wsi"]))
            for wsi_id in sentinel_ids:
                if session_ending():
                    return False
                log("t2_start", wsi_id=wsi_id)
                latents = load(wsi_id)
                with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    result = run_abmil_t2_lite(model, latents, bags.locations[wsi_id].coordinates)
                tables["t2"].extend(flatten_t2_records(
                    result, identity={"update": update, "exposures": exposure, "wsi_id": wsi_id},
                    attention_capture="attention", attention_metric="attention_probability",
                ))
                del latents
                log("t2_done", wsi_id=wsi_id)
        log("diagnostics_done", exposures=exposure)
        return True

    snapshots = {epoch * count for epoch in config["snapshot_epochs"]}
    if config["snapshot_first_half_epoch"]:
        snapshots.add(count // 2)

    def gradient_probe() -> bool:
        if exposure // count not in config["gradient_probe_epochs"] or exposure % count:
            return True
        log("gradient_panel_start", exposures=exposure)
        sampled_ids = [int(value) for value in config["gradient_probe_wsi_ids"]]
        probe_model = copy.deepcopy(model).train()
        vectors = []
        for wsi_id in sampled_ids:
            if session_ending():
                return False
            log("gradient_panel_wsi_start", wsi_id=wsi_id)
            probe_model.zero_grad(set_to_none=True)
            latents = load(wsi_id)
            target_index = bags.locations[wsi_id].diagnosis_index
            loss = _forward_loss(probe_model, latents,
                                 torch.tensor([target_index], device=device), weights[target_index],
                                 _aem_weight(exposure, count, config), config["label_smoothing"])[2]
            loss.backward()
            vectors.append(torch.cat([
                parameter.grad.detach().float().reshape(-1)
                if parameter.grad is not None else torch.zeros_like(parameter).reshape(-1)
                for parameter in probe_model.parameters()
            ]))
            del latents
        summary, norms, cosines = _gradient_noise_summary(torch.stack(vectors))
        for metric, value in summary.items():
            tables["t1"].append({
                "update": update, "exposures": exposure, "wsi_id": -1,
                "kind": "gradient_probe", "capture_name": "panel",
                "metric_name": metric, "value": value,
            })
        for index, wsi_id in enumerate(sampled_ids):
            tables["t1"].append({
                "update": update, "exposures": exposure, "wsi_id": wsi_id,
                "kind": "gradient_probe", "capture_name": "individual",
                "metric_name": "gradient_l2", "value": float(norms[index].item()),
            })
            for second in range(index + 1, len(sampled_ids)):
                tables["t1"].append({
                    "update": update, "exposures": exposure, "wsi_id": wsi_id,
                    "kind": "gradient_probe", "capture_name": f"pair.{sampled_ids[second]}",
                    "metric_name": "cosine", "value": float(cosines[index, second].item()),
                })
        del probe_model, vectors
        log("gradient_panel_done", exposures=exposure)
        return True

    def save(name: str = "latest.pt") -> dict:
        log("checkpoint_start", name=name, update=update, exposures=exposure)
        _write_rows(directory, tables)
        payload = build_dynamics_checkpoint(
            model=model, optimizer=optimizer, scaler=scaler, scheduler=scheduler,
            telemetry=telemetry, dynamics=dynamics, committed_update=update,
            exposure_count=exposure, epoch=epoch, within_epoch_cursor=cursor,
            current_order=order, effective_batch_size=config["effective_batch"],
        )
        save_dynamics_checkpoint(directory / name, payload)
        if name == "latest.pt" and exposure in snapshots:
            save_dynamics_checkpoint(directory / f"boundary_{exposure:06d}.pt", payload)
        log("checkpoint_done", name=name, update=update, exposures=exposure)
        return payload

    horizon = config["epochs"] * count
    boundaries = _boundaries(count, config)
    confirmation_boundaries = {
        value * count for value in (*config["t1_early_epochs"], config["warmup_epochs"])
    } if lr_confirmation else set()
    def assess() -> bool:
        # latest.pt is the training state before this potentially long boundary.
        if not (evaluate() and diagnostics() and gradient_probe()):
            log("session_pause_during_assessment", exposures=exposure)
            return False
        scheduler.last_assessment_exposure = exposure
        validation = {}
        for row in tables["predictions"]:
            if row["split"] == "validation":
                validation.setdefault(row["exposures"], []).append(row["unweighted_loss"])
        validation_ce = float(np.mean(validation[exposure]))
        previous_best = min((float(np.mean(losses)) for boundary, losses in validation.items()
                             if boundary < exposure), default=math.inf)
        payload = save()
        if validation_ce < previous_best:
            save_dynamics_checkpoint(directory / "best.pt", payload)
            log("best_checkpoint_done", exposures=exposure, validation_ce=validation_ce,
                criterion=config["checkpoint_selection"])
        log("assessment_done", exposures=exposure, validation_ce=validation_ce)
        return True

    if (lr_search or lr_confirmation) and update == 0 and scheduler.last_assessment_exposure < 0:
        payload = save()
        if not resumed:
            save_dynamics_checkpoint(directory / "initial.pt", payload)
        if not diagnostics(force=True):
            return False
        if lr_confirmation and not gradient_probe():
            return False
        scheduler.last_assessment_exposure = exposure
        save()
    if lr_search and update == config["lr_search"]["preparation_updates"] and scheduler.last_assessment_exposure < exposure:
        if not diagnostics(force=True):
            return False
        scheduler.last_assessment_exposure = exposure
        save()
        log("lr_search_sweep_start", update=update, exposures=exposure)
    if lr_search and stop_after_updates is not None and update >= stop_after_updates:
        save("final.pt")
        log("lr_search_done", reason="sweep_complete", update=update, exposures=exposure)
        return True
    if lr_confirmation and stop_after_updates is not None and update >= stop_after_updates:
        if scheduler.last_assessment_exposure < exposure:
            if not (diagnostics(force=True) and gradient_probe()):
                return False
            scheduler.last_assessment_exposure = exposure
            save()
        save("final.pt")
        log("lr_confirmation_done", peak_lr=config["peak_lr"], update=update, exposures=exposure)
        return True
    if lr_confirmation and exposure in confirmation_boundaries and scheduler.last_assessment_exposure < exposure:
        if not (diagnostics(force=True) and gradient_probe()):
            return False
        scheduler.last_assessment_exposure = exposure
        save()
    if stop_after_updates is None and exposure in ({0} | boundaries) and scheduler.last_assessment_exposure < exposure:
        save()
        if not assess():
            return False
    log("epoch_start", epoch=epoch, cursor=cursor, exposures=exposure)
    with ExitStack() as input_stack:
        inputs = None
        while exposure < horizon:
            if session_ending():
                save()
                log("session_pause", exposures=exposure)
                return False
            previous_exposure = exposure
            selected = order[cursor:cursor + config["effective_batch"]]
            selected_ids = [train_ids[index] for index in selected]
            next_exposure = exposure + len(selected_ids)
            lr = scheduler.set_next(next_exposure, next_update=update + 1)
            aem = _aem_weight(next_exposure, count, config)
            aem_tensor = torch.tensor(aem, device=device)
            indices = {}
            if device.type == "cuda":
                torch.cuda.current_stream(device).synchronize()
                torch.cuda.reset_peak_memory_stats(device)
            step_started = time.perf_counter()
            skipped = 0
            while True:
                if session_ending():
                    save()
                    log("session_pause_during_amp_backoff", exposures=exposure)
                    return False
                optimizer.zero_grad(set_to_none=True)
                observations = []
                initial_scale = float(scaler.get_scale())
                for wsi_id in selected_ids:
                    if inputs is None:
                        inputs = input_stack.enter_context(training_inputs())
                    _, indices[wsi_id], latents = next(inputs)
                    target_index = bags.locations[wsi_id].diagnosis_index
                    target = torch.tensor([target_index], device=device)
                    weight = torch.tensor(weights[target_index], device=device)
                    log("forward_start", wsi_id=wsi_id, update=update + 1)
                    logits, unweighted, weighted, summary = compiled(latents, target, weight, aem_tensor)
                    training_ce = functional.cross_entropy(
                        logits.detach().float().unsqueeze(0), target,
                        label_smoothing=config["label_smoothing"])
                    if lr_search and not bool(torch.isfinite(torch.stack((unweighted, weighted))).all()):
                        # An actual nonfinite loss is a search observation; do
                        # not backward/step it or retry this forward indefinitely.
                        log("lr_search_nonfinite_loss", attempted_update=update + 1,
                            lr=lr, wsi_id=wsi_id, unweighted_loss=float(unweighted.item()),
                            objective_loss=float(weighted.item()),
                            weighted_ce=float((unweighted * weight).item()),
                            aem_coefficient=aem,
                            aem_penalty=float((training_ce * weight - weighted).item()),
                            sampled_bag_size=len(indices[wsi_id]),
                            sample_indices_sha256=hashlib.sha256(indices[wsi_id].tobytes()).hexdigest())
                        optimizer.zero_grad(set_to_none=True)
                        input_stack.close()
                        inputs = None
                        save()
                        log("lr_search_done", reason="nonfinite_loss", update=update,
                            exposures=exposure, attempted_lr=lr)
                        return True
                    log("forward_done_backward_start", wsi_id=wsi_id)
                    scaler.scale(weighted / len(selected_ids)).backward()
                    log("backward_done", wsi_id=wsi_id)
                    observations.append((wsi_id, logits.detach(), unweighted.detach(),
                                         weighted.detach(), training_ce, summary.detach()))
                    del latents
                log("optimizer_start", update=update + 1)
                scaler.unscale_(optimizer)
                telemetry.begin_step(model, optimizer)
                scaler.step(optimizer)
                scaler.update()
                committed = float(scaler.get_scale()) >= initial_scale
                optimizer_record = telemetry.finish_step(
                    model, next_committed_update=update + 1, committed=committed,
                )
                if not committed:
                    affected = [name for name, parameter in model.named_parameters()
                                if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())]
                    log("amp_backoff", update=update + 1, attempt=skipped + 1,
                        scale_before=initial_scale, scale_after=float(scaler.get_scale()),
                        nonfinite_gradient_parameters=affected)
                optimizer.zero_grad(set_to_none=True)
                log("optimizer_done", committed=committed, amp_scale=float(scaler.get_scale()))
                if committed:
                    break
                skipped += 1
                input_stack.close()
                inputs = None
            if device.type == "cuda":
                torch.cuda.current_stream(device).synchronize()
            step_seconds = time.perf_counter() - step_started
            peak_memory_bytes = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
            update += 1
            exposure = next_exposure
            cursor += len(selected_ids)
            scheduler.last_exposure = exposure
            for wsi_id, logits, unweighted, weighted, training_ce, summary in observations:
                target_index = bags.locations[wsi_id].diagnosis_index
                record = classification_example_record(
                    logits=logits, target=target_index,
                    unweighted_loss=unweighted, weighted_loss=unweighted * weights[target_index],
                )
                forward = {
                    f"forward.{capture}.{metric}": value
                    for capture, values in _forward_records(summary).items()
                    for metric, value in values.items()
                }
                tables["t0"].append({
                    "update": update, "exposures": exposure, "wsi_id": wsi_id,
                    "effective_batch": len(selected_ids), "lr": lr,
                    "bag_size": bags.locations[wsi_id].count,
                    "sampled_bag_size": len(indices[wsi_id]),
                    "sample_indices_sha256": hashlib.sha256(indices[wsi_id].tobytes()).hexdigest(),
                    "aem_coefficient": aem, "objective_loss": float(weighted.item()),
                    "class_weight": weights[target_index],
                    "smoothed_ce": float(training_ce.item()),
                    "weighted_smoothed_ce": float((training_ce * weights[target_index]).item()),
                    "label_smoothing": config["label_smoothing"],
                    "c4_rotations_sha256": hashlib.sha256(
                        _sample_c4(len(indices[wsi_id]), config["seed"], epoch, wsi_id).tobytes()
                    ).hexdigest() if config["c4_augmentation"] else "identity",
                    "aem_penalty": float((training_ce * weights[target_index] - weighted).item()),
                    **covariates[wsi_id],
                    "amp_scale_before": initial_scale,
                    "amp_scale_after": float(scaler.get_scale()),
                    "amp_skipped_attempts": skipped,
                    "step_seconds": step_seconds, "peak_allocated_bytes": peak_memory_bytes,
                    **record, **forward,
                    **{f"optimizer.{key}": value for key, value in optimizer_record.items()},
                    **({"lr_search_phase": "preparation" if update <= config["lr_search"]["preparation_updates"] else "sweep",
                        "lr_search_step": max(update - config["lr_search"]["preparation_updates"], 0)} if lr_search else {}),
                })
            if cursor == count:
                log("epoch_done", epoch=epoch, exposures=exposure)
                epoch += 1
                cursor = 0
                order = paired_epoch_order(row_count=count, epoch=epoch, seed=config["seed"])
            log("step_done", update=update, exposures=exposure, step_seconds=step_seconds,
                peak_allocated_bytes=peak_memory_bytes, amp_scale=float(scaler.get_scale()),
                amp_skipped_attempts=skipped, lr=lr,
                mean_weighted_ce=float(np.mean([row["weighted_loss"] for row in tables["t0"][-len(selected_ids):]])),
                mean_aem_contribution=-float(np.mean([row["aem_penalty"] for row in tables["t0"][-len(selected_ids):]])),
                mean_objective_loss=float(np.mean([row["objective_loss"] for row in tables["t0"][-len(selected_ids):]])))
            if lr_search and update == config["lr_search"]["preparation_updates"]:
                payload = save()
                save_dynamics_checkpoint(directory / "preparation.pt", payload)
                log("preparation_checkpoint_done", update=update, exposures=exposure)
                input_stack.close()
                inputs = None
                if not diagnostics(force=True):
                    return False
                scheduler.last_assessment_exposure = exposure
                save()
                log("lr_search_sweep_start", update=update, exposures=exposure)
            if stop_after_updates is not None and update >= stop_after_updates:
                save()
                input_stack.close()
                inputs = None
                if lr_search:
                    save("final.pt")
                    log("lr_search_done", reason="sweep_complete", update=update,
                        exposures=exposure, lr=lr)
                    return True
                if not diagnostics(force=True):
                    return False
                if lr_confirmation and not gradient_probe():
                    return False
                scheduler.last_assessment_exposure = exposure
                save()
                if lr_confirmation:
                    save("final.pt")
                    log("lr_confirmation_done", peak_lr=config["peak_lr"], update=update,
                        exposures=exposure, lr=lr)
                    return True
                log("smoke_segment_done", update=update, exposures=exposure)
                return True
            if exposure in confirmation_boundaries and scheduler.last_assessment_exposure < exposure:
                save()
                input_stack.close()
                inputs = None
                if not (diagnostics(force=True) and gradient_probe()):
                    return False
                scheduler.last_assessment_exposure = exposure
                save()
                log("lr_confirmation_boundary_done", peak_lr=config["peak_lr"], exposures=exposure)
            elif exposure in boundaries and stop_after_updates is None:
                save()
                input_stack.close()
                inputs = None
                if not assess():
                    return False
            elif (exposure // config["checkpoint_every_exposures"]
                  > previous_exposure // config["checkpoint_every_exposures"] or cursor == 0):
                save()
            if cursor == 0 and exposure < horizon:
                log("epoch_start", epoch=epoch, exposures=exposure)
    save("final.pt")
    return True


def run(repo_root: Path, latent_root: Path, output_root: Path, resume_root: Path | None, effective_batch: int, branch: str, *, lr_search: bool = False, lr_confirmation: bool = False, session_deadline_unix: float | None = None) -> None:
    session_started = time.time()
    config_path = repo_root / "docs/data/spec0054_fold0_run.json"
    config = json.loads(config_path.read_text())
    config["effective_batch"] = effective_batch
    config["run_mode"] = "lr_confirmation" if lr_confirmation else "lr_search" if lr_search else "fold"
    extraction_path = repo_root / "docs/data/spec0054_fp16_latent_extraction.json"
    cohort_path = repo_root / "docs/data/spec0054_cohort_folds.csv"
    contract = json.loads(extraction_path.read_text())
    expected_results = {int(item["shard"]): item["result_sha256"] for item in config["sources"]}
    bags = ShardBags(latent_root, contract, cohort_path, expected_results)
    identity = {
        "source_commit": subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "runner_sha256": _sha256(Path(__file__)),
        "reader_sha256": _sha256(Path(__file__).with_name("spec0054_shard_bags.py")),
        "model_sha256": _sha256(repo_root / "src/eqvae/models/gated_abmil.py"),
        "dynamics_sha256": _sha256(repo_root / "src/eqvae/training/mil_dynamics.py"),
        "t2_sha256": _sha256(repo_root / "src/eqvae/training/mil_t2_lite.py"),
        "config_sha256": _sha256(config_path), "cohort_sha256": _sha256(cohort_path),
        "extraction_sha256": _sha256(extraction_path), "effective_batch": effective_batch,
        "run_mode": config["run_mode"],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        if lr_confirmation:
            train_count = sum(bag.fold != config["fold"] for bag in bags.locations.values())
            stop = config["lr_confirmation"]["epochs"] * math.ceil(train_count / effective_batch)
            for peak in config["lr_confirmation"]["peak_lrs"]:
                candidate = f"lr_{peak:g}"
                candidate_config = {**config, "peak_lr": peak}
                completed = run_branch(
                    branch, bags, candidate_config, output_root / candidate,
                    resume_root / candidate if resume_root is not None else None,
                    {**identity, "peak_lr": peak}, session_started, stop_after_updates=stop,
                    session_deadline_unix=session_deadline_unix,
                )
                if not completed:
                    break
            return
        search = config["lr_search"]
        stop = search["preparation_updates"] + search["sweep_updates"] if lr_search else None
        run_branch(branch, bags, config, output_root, resume_root, identity, session_started,
                   stop_after_updates=stop, session_deadline_unix=session_deadline_unix)
    finally:
        bags.close()


def run_read_probe(repo_root: Path, latent_root: Path, output_root: Path, branch: str,
                   session_deadline_unix: float | None = None) -> None:
    """Compare combined I/O policies on the same training window with controlled client cache."""
    config_path = repo_root / "docs/data/spec0054_fold0_run.json"
    config = json.loads(config_path.read_text())
    probe = config["read_probe"]
    extraction = json.loads((repo_root / "docs/data/spec0054_fp16_latent_extraction.json").read_text())
    shards = [shard for shard in extraction["shards"] if shard["shard"] in probe["shards"]]
    contract = {**extraction, "shards": shards}
    directory = output_root / branch
    directory.mkdir(parents=True, exist_ok=False)
    lines = (repo_root / "docs/data/spec0054_cohort_folds.csv").read_text().splitlines(keepends=True)
    cohort_path = directory / "probe_cohort.csv"
    cohort_path.write_text(lines[0] + "".join(
        line for line in lines[1:]
        if any(shard["first_wsi"] <= int(line.split(",", 1)[0]) <= shard["last_wsi"] for shard in shards)
    ))
    print(json.dumps({"event": "read_probe_index_start", "branch": branch}), flush=True)
    bags = ShardBags(latent_root, contract, cohort_path,
                     {item["shard"]: item["result_sha256"] for item in config["sources"]
                      if item["shard"] in probe["shards"]})
    rows, blocks, hints = [], [], []
    lock = Lock()
    hardware = {"logical_cpu_count": os.cpu_count(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
                "torch_threads": torch.get_num_threads(), "gpu_used": False, "euid": os.geteuid(),
                "kernel_release": os.uname().release, "host_pinning": False, "h2d_measured": False,
                "working_free_bytes": shutil.disk_usage(output_root.parent).free,
                "working_device": output_root.parent.stat().st_dev,
                "scratch_free_bytes": shutil.disk_usage(repo_root.parent).free,
                "scratch_device": repo_root.parent.stat().st_dev}
    source = subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True).strip()
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mincore.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_ubyte))
    libc.mincore.restype = ctypes.c_int

    def snapshot(reader, wsi_id):
        location = reader.locations[wsi_id]
        offset = location.first_record * RECORD_BYTES
        aligned = offset - offset % mmap.PAGESIZE
        length = location.count * RECORD_BYTES + offset - aligned
        page_count = (length + mmap.PAGESIZE - 1) // mmap.PAGESIZE
        vector = (ctypes.c_ubyte * page_count)()
        with mmap.mmap(reader.descriptors[branch, location.shard], length,
                       access=mmap.ACCESS_COPY, offset=aligned) as mapping:
            pointer = ctypes.c_char.from_buffer(mapping)
            status = libc.mincore(ctypes.addressof(pointer), length, vector)
            error = ctypes.get_errno() if status != 0 else None
            del pointer
        resident = sum(value & 1 for value in vector) if status == 0 else None
        memory = {key: int(value.split()[0]) for key, value in
                  (line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
                  if key in ("MemTotal", "MemAvailable", "Cached", "Shmem", "SReclaimable", "Dirty", "Writeback")}
        process_memory = {key: int(value.split()[0]) * 1024 for key, value in
                          (line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines()
                           if ":" in line) if key in ("VmRSS", "VmHWM")}
        process_io = {key: int(value) for key, value in
                      (line.split(":", 1) for line in Path("/proc/self/io").read_text().splitlines())}
        vmstat = {key: int(value) for key, value in
                  (line.split() for line in Path("/proc/vmstat").read_text().splitlines())
                  if key.startswith(("workingset_refault", "pgscan", "pgsteal")) or key == "pgmajfault"}
        return {"at_unix": time.time(), "pages": page_count, "resident_pages": resident,
                "resident_fraction": resident / page_count if resident is not None else None,
                "mincore_status": status, "mincore_errno": error,
                "meminfo_kib": memory, "process_memory_bytes": process_memory,
                "process_io": process_io, "host_vmstat": vmstat}

    result = {"branch": branch, "source_commit": source,
              "config_sha256": _sha256(config_path),
              "reader_sha256": _sha256(Path(__file__).with_name("spec0054_shard_bags.py")),
              "seed": config["seed"], "patch_retention": config["patch_retention"],
              "probe": probe, "hardware": hardware, "completed": False,
              "prepared_hash_order": "NHWC_physical_channels_last",
              "cache_policy": "window_reset_before_each_arm_then_method_specific_discard_and_willneed",
              "counter_scope": "mincore_bag_snapshot; process_rss_peak_shared_by_threads; vmstat_host_wide; NFS_mount_shared_between_branches",
              "blocks": blocks, "rows": rows, "prefetch_hints": hints}

    def save_result():
        (directory / "read_probe.json").write_text(json.dumps(result, indent=2) + "\n")

    def synchronize(phase):
        # Both branches exercise the same condition together; waiting is not timed I/O.
        other = next(item for item in BRANCHES if item != branch)
        (output_root / f"{branch}.{phase}.ready").write_text(str(time.time()))
        waiting = time.perf_counter()
        while not (output_root / f"{other}.{phase}.ready").exists():
            if session_deadline_unix is not None and time.time() >= session_deadline_unix:
                return None
            time.sleep(0.1)
        return time.perf_counter() - waiting

    def nfs_read_counters():
        # Only retain counters for the mounted inputs, never server addresses.
        points = {item["mount"]["mount_point"] for item in result["files"].values()}
        observed, active = {}, None
        for line in Path("/proc/self/mountstats").read_text().splitlines():
            if line.startswith("device "):
                point = line.split(" mounted on ", 1)[1].split(" with fstype ", 1)[0]
                active = point if point in points and " with fstype nfs" in line else None
            elif active is not None and line.lstrip().startswith("READ:"):
                observed[active] = [int(value) for value in line.split()[1:]]
        return observed

    def advise_future(wsi_id, epoch, block_index, method):
        location = bags.locations[wsi_id]
        length = min(location.count * RECORD_BYTES, probe["prefetch_bytes_per_wsi"])
        at_unix, started = time.time(), time.perf_counter()
        os.posix_fadvise(bags.descriptors[branch, location.shard],
                        location.first_record * RECORD_BYTES, length, os.POSIX_FADV_WILLNEED)
        hint = {"epoch": epoch, "block_index": block_index, "method": method, "wsi_id": wsi_id,
                "requested_bytes": length, "started_unix": at_unix,
                "ended_unix": time.time(), "seconds": time.perf_counter() - started}
        with lock:
            hints.append(hint)
            with (directory / "read_probe_hints.jsonl").open("a") as handle:
                handle.write(json.dumps(hint) + "\n")
        return hint["seconds"]

    def read_bag(request):
        position, window, epoch, block_index, method = request
        if session_deadline_unix is not None and time.time() >= session_deadline_unix:
            return None
        wsi_id = window[position]
        location = bags.locations[wsi_id]
        indices = _sample_indices(location.count, config["seed"], epoch, wsi_id,
                                  config["patch_retention"])
        hint_seconds = 0.0
        if method in ("sampled_discard_prefetch", "mmap_discard_prefetch") and position + probe["lookahead_wsi"] < len(window):
            hint_seconds = advise_future(window[position + probe["lookahead_wsi"]], epoch, block_index, method)
        info = {"branch": branch, "shard": location.shard, "wsi_id": wsi_id,
                "method": method, "storage": "input", "epoch": epoch,
                "block_index": block_index}
        before = snapshot(bags, wsi_id)
        print(json.dumps({"event": "read_probe_read_start", **info,
                          "resident_fraction": before["resident_fraction"]}), flush=True)
        cpu_started, started = time.thread_time(), time.perf_counter()
        if method == "mmap_discard_prefetch":
            with mmap.mmap(bags.descriptors[branch, location.shard], location.count * RECORD_BYTES,
                           access=mmap.ACCESS_READ, offset=location.first_record * RECORD_BYTES) as mapping:
                mapping.madvise(getattr(mmap, probe["mmap_advice"]))
                view = np.ndarray((location.count, 16, 32, 32), dtype="<f2", buffer=mapping)
                selected = view[indices]  # Integer indexing copies into an owned, writable buffer.
                del view
        else:
            selected = bags.read(branch, wsi_id, indices)
        read_seconds = time.perf_counter() - started
        read_cpu_seconds = time.thread_time() - cpu_started
        after = snapshot(bags, wsi_id)
        started = time.perf_counter()
        if method in ("sampled_discard_prefetch", "mmap_discard_prefetch"):
            os.posix_fadvise(bags.descriptors[branch, location.shard],
                            location.first_record * RECORD_BYTES, location.count * RECORD_BYTES,
                            os.POSIX_FADV_DONTNEED)
        discard_seconds = time.perf_counter() - started
        after_discard = snapshot(bags, wsi_id)
        hash_started = time.perf_counter()
        digest = hashlib.sha256(memoryview(selected).cast("B")).hexdigest()
        hash_seconds = time.perf_counter() - hash_started
        host_started, host_cpu_started = time.perf_counter(), time.thread_time()
        rotations = _sample_c4(len(indices), config["seed"], epoch, wsi_id)
        if config["c4_augmentation"]:
            _apply_c4(selected, rotations)
        source = torch.from_numpy(selected)
        host = torch.empty_like(source, memory_format=torch.channels_last, pin_memory=False)
        host.copy_(source)
        host_seconds = time.perf_counter() - host_started
        host_cpu_seconds = time.thread_time() - host_cpu_started
        after_prepare = snapshot(bags, wsi_id)
        prepared_hash_started = time.perf_counter()
        prepared_digest = hashlib.sha256(memoryview(host.permute(0, 2, 3, 1).numpy()).cast("B")).hexdigest()
        prepared_hash_seconds = time.perf_counter() - prepared_hash_started
        row = {**info, "bag_size": location.count, "sampled_bag_size": len(indices),
               "requested_bytes": len(indices) * RECORD_BYTES,
               "read_regions": None if method == "mmap_discard_prefetch" else int(np.count_nonzero(np.diff(indices) != 1)) + 1,
               "sample_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
               "selected_values_sha256": digest, "prepared_values_sha256": prepared_digest,
               "c4_rotations_sha256": hashlib.sha256(rotations.tobytes()).hexdigest(),
               "read_seconds": read_seconds, "read_cpu_seconds": read_cpu_seconds,
               "read_plus_subset_seconds": read_seconds,
               "discard_seconds": discard_seconds, "future_hint_seconds": hint_seconds,
               "host_prepare_seconds": host_seconds, "host_prepare_cpu_seconds": host_cpu_seconds,
               "pipeline_seconds": hint_seconds + read_seconds + discard_seconds + host_seconds,
               "hash_seconds": hash_seconds, "prepared_hash_seconds": prepared_hash_seconds,
               "host_channels_last": host.is_contiguous(memory_format=torch.channels_last),
               "host_buffer_bytes": host.numel() * host.element_size(),
               "before_read": before, "after_read": after,
               "after_discard": after_discard, "after_prepare": after_prepare}
        del source, selected
        with lock:
            rows.append(row)
            with (directory / "read_probe_rows.jsonl").open("a") as partial:
                partial.write(json.dumps(row, sort_keys=True) + "\n")
            print(json.dumps({"event": "read_probe_read_done", **info,
                              "read_seconds": read_seconds, "pipeline_seconds": row["pipeline_seconds"],
                              "resident_before": before["resident_fraction"],
                              "resident_after_discard": after_discard["resident_fraction"],
                              "selected_values_sha256": digest}), flush=True)
        return host

    try:
        mounts = []
        for line in Path("/proc/self/mountinfo").read_text().splitlines():
            left, right = line.split(" - ", 1)
            point = left.split()[4].replace("\\040", " ")
            mounts.append({"mount_point": point, "filesystem_type": right.split()[0],
                           "device": left.split()[2]})

        def file_metadata(path):
            stat = path.stat()
            mount = max((m for m in mounts if path.resolve().is_relative_to(m["mount_point"])),
                        key=lambda m: len(m["mount_point"]))
            return {"path": str(path), "resolved_path": str(path.resolve()),
                    "bytes": stat.st_size, "device": stat.st_dev,
                    "owner_uid": stat.st_uid, "mount": mount}

        result["files"] = {str(shard): file_metadata(binary)
                           for shard, binary in bags.files[branch].items()}
        for shard, binary in bags.files[branch].items():
            bags.descriptors[branch, shard] = os.open(binary, os.O_RDONLY)
        panel = sorted(wsi for wsi, location in bags.locations.items()
                       if location.fold != config["fold"])
        epoch = probe["first_epoch"]
        epoch_order = [panel[i] for i in paired_epoch_order(row_count=len(panel), epoch=epoch,
                                                           seed=config["seed"])]
        epoch_plan = []
        for position, wsi in enumerate(epoch_order):
            location = bags.locations[wsi]
            indices = _sample_indices(location.count, config["seed"], epoch, wsi,
                                      config["patch_retention"])
            epoch_plan.append({"position": position, "wsi_id": wsi, "shard": location.shard,
                               "byte_offset": location.first_record * RECORD_BYTES,
                               "bag_bytes": location.count * RECORD_BYTES,
                               "sampled_bytes": len(indices) * RECORD_BYTES,
                               "sample_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
                               "contiguous_regions": int(np.count_nonzero(np.diff(indices) != 1)) + 1})
        result["epoch_read_plan"] = epoch_plan
        result["epoch_plan_scope"] = "training_only; assessments excluded; subsets reconstructed from seed/epoch/count"
        result["largest_planned_selected_bytes"] = max(item["sampled_bytes"] for item in epoch_plan)
        window = epoch_order[:probe["wsi_block_size"]]
        result["candidate_wsi_count"] = len(panel)
        result["wsi_ids"] = window
        result["paired_panel_full_bytes"] = 2 * sum(bags.locations[wsi].count * RECORD_BYTES
                                                   for wsi in window)
        save_result()
        print(json.dumps({"event": "read_probe_start", "branch": branch,
                          "hardware": hardware, "files": result["files"],
                          "wsi_count": len(window), "candidate_wsi_count": len(panel),
                          "probe": probe}), flush=True)
        with ThreadPoolExecutor(max_workers=probe["reader_workers"]) as pool:
            for block_index in range(probe["passes"]):
                method_order = probe["method_order_indices"][block_index]
                methods = [probe["methods"][i] for i in method_order]
                for position, method in enumerate(methods):
                    wait = synchronize(f"repeat{block_index}_condition{position}")
                    if wait is None:
                        return
                    reset_at, reset_started = time.time(), time.perf_counter()
                    for wsi in window:
                        location = bags.locations[wsi]
                        os.posix_fadvise(bags.descriptors[branch, location.shard],
                                        location.first_record * RECORD_BYTES,
                                        location.count * RECORD_BYTES, os.POSIX_FADV_DONTNEED)
                    residency = []
                    for wsi in window:
                        observed = snapshot(bags, wsi)
                        residency.append({"wsi_id": wsi, **{k: observed[k] for k in
                                          ("resident_fraction", "pages", "mincore_status", "mincore_errno")}})
                    block = {"epoch": epoch, "block_index": block_index,
                             "storage": "input", "method": method, "completed": False,
                             "barrier_wait_seconds": wait, "wsi_order": window,
                             "cache_reset_started_unix": reset_at,
                             "cache_reset_ended_unix": time.time(),
                             "cache_reset_seconds": time.perf_counter() - reset_started,
                             "residency_after_cache_reset": residency}
                    blocks.append(block)
                    save_result()
                    ready_wait = synchronize(f"repeat{block_index}_condition{position}_cache_ready")
                    if ready_wait is None:
                        return
                    block["cache_ready_wait_seconds"] = ready_wait
                    block["started_unix"] = time.time()
                    started = time.perf_counter()
                    block["nfs_read_before"] = nfs_read_counters()
                    if method in ("sampled_discard_prefetch", "mmap_discard_prefetch"):
                        for wsi in window[:probe["lookahead_wsi"]]:
                            if session_deadline_unix is not None and time.time() >= session_deadline_unix:
                                break
                            advise_future(wsi, epoch, block_index, method)
                    # Keep prepared CPU buffers alive in a bounded FIFO, without changing WSI order.
                    plan = iter((i, window, epoch, block_index, method) for i in range(len(window)))
                    request = next(plan, None)
                    pending = deque()
                    completed = True
                    reserved = max_reserved = max_pending = max_ready = 0
                    future_limit = probe["future_wsi"][method]
                    while request is not None or pending:
                        while request is not None and len(pending) < future_limit:
                            wsi = window[request[0]]
                            size = math.ceil(bags.locations[wsi].count * config["patch_retention"]) * RECORD_BYTES
                            if reserved + size > probe["max_ready_bytes_per_branch"]:
                                break
                            pending.append((pool.submit(read_bag, request), size))
                            reserved += size
                            max_reserved = max(max_reserved, reserved)
                            max_pending = max(max_pending, len(pending))
                            request = next(plan, None)
                        max_ready = max(max_ready, sum(f.done() and f.result() is not None for f, _ in pending))
                        future, size = pending.popleft()
                        prepared = future.result()
                        completed = (prepared is not None) and completed
                        reserved -= size
                        del prepared, future
                    block.update(future_wsi_limit=future_limit, max_reserved_host_bytes=max_reserved,
                                 max_pending_wsi=max_pending, max_ready_wsi_observed=max_ready)
                    block["nfs_read_after"] = nfs_read_counters()
                    block.update(ended_unix=time.time(), wall_seconds=time.perf_counter() - started,
                                 completed=completed)
                    save_result()
                    print(json.dumps({"event": "read_probe_block_done", "branch": branch, **block}), flush=True)
                    if not completed:
                        return
        result["completed"] = True
    finally:
        save_result()
        bags.close()
        print(json.dumps({"event": "read_probe_done", "branch": branch,
                          "completed": result["completed"], "reads": len(rows)}), flush=True)



def run_smoke(repo_root: Path, latent_root: Path, output_root: Path, branch: str, effective_batch: int) -> None:
    """Eight real optimizer updates, restoring after four; train-only T1/T2."""
    config = json.loads((repo_root / "docs/data/spec0054_fold0_run.json").read_text())
    config["effective_batch"] = effective_batch
    config["t1_train_wsi"] = 1080  # Shard-1 train sentinel, 4,505 patches.
    contract = json.loads((repo_root / "docs/data/spec0054_fp16_latent_extraction.json").read_text())
    shard = contract["shards"][0]
    contract = {**contract, "shards": [shard]}
    output_root.mkdir(parents=True, exist_ok=True)
    cohort_lines = (repo_root / "docs/data/spec0054_cohort_folds.csv").read_text().splitlines(keepends=True)
    cohort_path = output_root / f"smoke_cohort_{branch}.csv"
    cohort_path.write_text(cohort_lines[0] + "".join(
        line for line in cohort_lines[1:]
        if shard["first_wsi"] <= int(line.split(",", 1)[0]) <= shard["last_wsi"]
    ))
    print(json.dumps({"event": "smoke_index_start", "branch": branch, "shards": [1]}), flush=True)
    bags = ShardBags(
        latent_root, contract, cohort_path,
        {shard["shard"]: config["sources"][0]["result_sha256"]},
    )
    print(json.dumps({"event": "smoke_index_done", "branch": branch}), flush=True)
    session_started = time.time()
    identity = {"smoke": True, "effective_batch": effective_batch,
                "config": config, "source_commit": subprocess.check_output(
                    ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True).strip()}
    initial_root = output_root / "smoke_initial"
    try:
        run_branch(branch, bags, config, initial_root, None, identity, session_started,
                   stop_after_updates=4)
        checkpoint = load_dynamics_checkpoint(initial_root / branch / "latest.pt")
        print(json.dumps({"branch": branch, "event": "smoke_resume_start",
                          "update": checkpoint["committed_update"],
                          "exposures": checkpoint["exposure_count"],
                          "amp_scale": checkpoint["scaler"]["scale"]}), flush=True)
        run_branch(branch, bags, config, output_root, initial_root, identity, session_started,
                   stop_after_updates=8)
        branch_dir = output_root / branch
        checkpoint = load_dynamics_checkpoint(branch_dir / "latest.pt")
        rows = {name: len(load_telemetry_table(branch_dir / f"{name}.npz")["update"])
                for name in ("t0", "t1", "t2")}
        result = {"branch": branch, "effective_batch": effective_batch,
                  "committed_updates": checkpoint["committed_update"],
                  "exposures": checkpoint["exposure_count"],
                  "resume_after_update": 4, "amp_scale": checkpoint["scaler"]["scale"],
                  "saved_telemetry_rows": rows, "mounted_shards": [1],
                  "architecture": config["architecture"], "weight_decay": config["weight_decay"],
                  "train_wsi_count": sum(bag.fold != config["fold"] for bag in bags.locations.values())}
        (branch_dir / "smoke.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"event": "smoke_done", **result}), flush=True)
    finally:
        bags.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume-root", type=Path)
    parser.add_argument("--effective-batch", type=int, choices=(1, 4, 8), default=1)
    parser.add_argument("--branch", choices=BRANCHES, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--read-probe", action="store_true")
    parser.add_argument("--lr-search", action="store_true")
    parser.add_argument("--lr-confirmation", action="store_true")
    parser.add_argument("--session-deadline-unix", type=float)
    args = parser.parse_args()
    if args.read_probe:
        run_read_probe(args.repo_root, args.latent_root, args.output_root, args.branch, args.session_deadline_unix)
    elif args.smoke:
        run_smoke(args.repo_root, args.latent_root, args.output_root, args.branch, args.effective_batch)
    else:
        run(args.repo_root, args.latent_root, args.output_root, args.resume_root, args.effective_batch, args.branch,
            lr_search=args.lr_search, lr_confirmation=args.lr_confirmation,
            session_deadline_unix=args.session_deadline_unix)
