"""First paired Spec 0054 fold, reading complete FP16 WSI bags in place."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from pathlib import Path
from threading import Lock

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


def _make_classifier(state: dict, device: torch.device, *, hidden_width: int = 128):
    model = GatedABMILClassifier(hidden_width)
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


def _forward_loss(model, latents, target, class_weight, aem_weight: torch.Tensor | float = 0.0):
    with torch.autocast(latents.device.type, dtype=torch.float16, enabled=latents.is_cuda):
        logits, representations = model.forward_with_representations(latents)
        loss = functional.cross_entropy(logits.float().unsqueeze(0), target)
        summary = t0_forward_summary(representations)
        attention = representations[-3].float()
        entropy = -(attention * attention.clamp_min(1e-12).log()).sum()
    return logits, loss, loss * class_weight - aem_weight * entropy, summary


def _compiled_closure(model):
    def closure(latents, target, class_weight, aem_weight):
        return _forward_loss(model, latents, target, class_weight, aem_weight)
    compiled = torch.compile(
        closure, backend="inductor", fullgraph=True, dynamic=None,
        mode="max-autotune-no-cudagraphs",
    )
    def forward(latents, target, class_weight, aem_weight: torch.Tensor | float = 0.0):
        torch._dynamo.maybe_mark_dynamic(latents, 0)
        return compiled(latents, target, class_weight, aem_weight)
    return forward


def _aem_weight(exposure: int, train_count: int, config: dict) -> float:
    progress = min(exposure / (config["epochs"] * train_count), 1.0)
    return config["aem_weight"] * (1 + math.cos(math.pi * progress)) / 2


def _sample_indices(count: int, seed: int, epoch: int, wsi_id: int, fraction: float) -> np.ndarray:
    generator = np.random.default_rng(np.random.SeedSequence([seed, epoch, wsi_id]))
    return np.sort(generator.choice(count, size=math.ceil(count * fraction), replace=False))


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
    """The historical warmup/cosine curve indexed by WSI exposures."""

    def __init__(self, optimizer: torch.optim.Optimizer, train_count: int, config: dict) -> None:
        self.optimizer = optimizer
        self.train_count = train_count
        self.config = config
        self.last_exposure = 0
        self.last_assessment_exposure = -1

    def set_next(self, exposure: int) -> float:
        warmup = self.config["warmup_epochs"] * self.train_count
        horizon = self.config["epochs"] * self.train_count
        peak = self.config["peak_lr"]
        if exposure <= warmup:
            lr = peak * (0.1 + 0.9 * (exposure - 1) / (warmup - 1))
        else:
            progress = (exposure - warmup) / (horizon - warmup)
            lr = peak * (0.01 + 0.99 * (1 + math.cos(math.pi * progress)) / 2)
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
) -> bool:
    torch.backends.cudnn.benchmark = True
    directory = output_root / branch
    directory.mkdir(parents=True, exist_ok=False)
    events = (directory / "events.jsonl").open("a", buffering=1)
    log_lock = Lock()

    def log(event: str, **details) -> None:
        line = json.dumps({"branch": branch, "event": event, **details}, sort_keys=True)
        with log_lock:
            print(line, flush=True)
            print(line, file=events, flush=True)

    def session_ending() -> bool:
        return should_pause_for_session(session_started_unix=session_started, now_unix=time.time())

    log("worker_start", physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
        device=str(device), device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu")
    prior = resume_root / branch if resume_root is not None else None
    if prior is not None and prior.exists():
        if json.loads((prior / "identity.json").read_text()) != {**identity, "branch": branch}:
            raise ValueError("Resume identity differs")
        for name in ("latest.pt", "t0.npz", "t1.npz", "t2.npz", "predictions.npz"):
            if (prior / name).exists():
                shutil.copy2(prior / name, directory / name)
    (directory / "identity.json").write_text(json.dumps({**identity, "branch": branch}, sort_keys=True) + "\n")
    ids = sorted(bags.locations)
    train_ids = [wsi_id for wsi_id in ids if bags.locations[wsi_id].fold != config["fold"]]
    validation_ids = [wsi_id for wsi_id in ids if bags.locations[wsi_id].fold == config["fold"]]
    count = len(train_ids)
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
    model = _make_classifier(initial_state, device, hidden_width=config["attention_hidden_width"])
    optimizer = _make_optimizer(model, peak_lr=config["peak_lr"], weight_decay=config["weight_decay"])
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    telemetry = AdamWTelemetry()
    dynamics = ExampleDynamicsTracker(ids)
    scheduler = ExposureSchedule(optimizer, count, config)
    compiled = _compiled_closure(model)
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
            while read_epoch < config["epochs"]:
                for index in read_order[read_cursor:]:
                    wsi_id = train_ids[index]
                    indices = _sample_indices(bags.locations[wsi_id].count,
                                              config["seed"], read_epoch, wsi_id,
                                              config["patch_retention"])
                    yield wsi_id, indices
                read_epoch += 1
                read_cursor = 0
                read_order = paired_epoch_order(row_count=count, epoch=read_epoch,
                                               seed=config["seed"])

        copy_stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None

        def prepare(request):
            wsi_id, indices = request
            started = time.perf_counter()
            log("bag_read_start", wsi_id=wsi_id, shard=bags.locations[wsi_id].shard,
                bag_size=bags.locations[wsi_id].count, prefetch=True,
                requested_bytes=len(indices) * RECORD_BYTES,
                read_regions=int(np.count_nonzero(np.diff(indices) != 1)) + 1)
            array = bags.read(branch, wsi_id, indices)
            source = torch.from_numpy(array)
            host = torch.empty_like(source, memory_format=torch.channels_last,
                                    pin_memory=device.type == "cuda")
            host.copy_(source)
            read_seconds = time.perf_counter() - started
            log("bag_transfer_start", wsi_id=wsi_id, sampled_bag_size=len(indices),
                sample_indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
                read_prepare_seconds=read_seconds, prefetch=True)
            transfer_started = time.perf_counter()
            if copy_stream is not None:
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
        with ThreadPoolExecutor(max_workers=1) as reader:
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
                                     _aem_weight(exposure, count, config))[2]
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
                                 _aem_weight(exposure, count, config))[2]
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

    def save(name: str = "latest.pt") -> None:
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

    horizon = config["epochs"] * count
    boundaries = _boundaries(count, config)
    def assess() -> bool:
        # latest.pt is the training state before this potentially long boundary.
        if not (evaluate() and diagnostics() and gradient_probe()):
            log("session_pause_during_assessment", exposures=exposure)
            return False
        scheduler.last_assessment_exposure = exposure
        save()
        return True

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
            lr = scheduler.set_next(next_exposure)
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
                    log("forward_done_backward_start", wsi_id=wsi_id)
                    scaler.scale(weighted / len(selected_ids)).backward()
                    log("backward_done", wsi_id=wsi_id)
                    observations.append((wsi_id, logits.detach(), unweighted.detach(), weighted.detach(), summary.detach()))
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
            for wsi_id, logits, unweighted, weighted, summary in observations:
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
                    "aem_penalty": float((unweighted * weights[target_index] - weighted).item()),
                    **covariates[wsi_id],
                    "amp_scale_before": initial_scale,
                    "amp_scale_after": float(scaler.get_scale()),
                    "amp_skipped_attempts": skipped,
                    "step_seconds": step_seconds, "peak_allocated_bytes": peak_memory_bytes,
                    **record, **forward,
                    **{f"optimizer.{key}": value for key, value in optimizer_record.items()},
                })
            if cursor == count:
                log("epoch_done", epoch=epoch, exposures=exposure)
                epoch += 1
                cursor = 0
                order = paired_epoch_order(row_count=count, epoch=epoch, seed=config["seed"])
            log("step_done", update=update, exposures=exposure, step_seconds=step_seconds,
                peak_allocated_bytes=peak_memory_bytes, amp_scale=float(scaler.get_scale()),
                amp_skipped_attempts=skipped)
            if stop_after_updates is not None and update >= stop_after_updates:
                save()
                input_stack.close()
                inputs = None
                if not diagnostics(force=True):
                    return False
                scheduler.last_assessment_exposure = exposure
                save()
                log("smoke_segment_done", update=update, exposures=exposure)
                return True
            if exposure in boundaries:
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


def run(repo_root: Path, latent_root: Path, output_root: Path, resume_root: Path | None, effective_batch: int, branch: str) -> None:
    session_started = time.time()
    config_path = repo_root / "docs/data/spec0054_fold0_run.json"
    config = json.loads(config_path.read_text())
    config["effective_batch"] = effective_batch
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
    }
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        run_branch(branch, bags, config, output_root, resume_root, identity, session_started)
    finally:
        bags.close()


def run_read_probe(repo_root: Path, latent_root: Path, output_root: Path, branch: str,
                   device: torch.device = torch.device("cuda:0")) -> None:
    """One median train bag, one trial per read mode; no classifier execution."""
    config = json.loads((repo_root / "docs/data/spec0054_fold0_run.json").read_text())
    contract = json.loads((repo_root / "docs/data/spec0054_fp16_latent_extraction.json").read_text())
    print(json.dumps({"event": "read_probe_index_start", "branch": branch}), flush=True)
    bags = ShardBags(latent_root, contract, repo_root / "docs/data/spec0054_cohort_folds.csv",
                     {int(item["shard"]): item["result_sha256"] for item in config["sources"]})
    print(json.dumps({"event": "read_probe_index_done", "branch": branch}), flush=True)
    directory = output_root / branch
    directory.mkdir(parents=True, exist_ok=False)
    train = sorted((wsi for wsi in bags.locations if bags.locations[wsi].fold != config["fold"]),
                   key=lambda wsi: (bags.locations[wsi].count, wsi))
    selected = (train[len(train) // 2],)
    modes = ("full_then_sample", "sampled_only")
    sequence = (0, 1) if branch == "normal_vae" else (1, 0)
    print(json.dumps({"event": "read_probe_start", "branch": branch,
                      "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                      "device": str(device), "wsi_ids": selected}), flush=True)
    # Initialize pinned allocator/CUDA outside the timed reads.
    print(json.dumps({"event": "read_probe_cuda_start", "branch": branch}), flush=True)
    warmup = torch.empty(1, dtype=torch.float16, pin_memory=device.type == "cuda").to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    del warmup
    print(json.dumps({"event": "read_probe_cuda_done", "branch": branch}), flush=True)
    rows = []
    try:
        for wsi_id in selected:
            location = bags.locations[wsi_id]
            indices = _sample_indices(location.count, config["seed"], 0, wsi_id, config["patch_retention"])
            key = (branch, location.shard)
            if key not in bags.descriptors:
                bags.descriptors[key] = os.open(bags.files[branch][location.shard], os.O_RDONLY)
            for trial, mode_index in enumerate(sequence):
                mode = modes[mode_index]
                print(json.dumps({"event": "read_probe_trial_start", "branch": branch,
                                  "wsi_id": wsi_id, "trial": trial, "mode": mode}), flush=True)
                started = time.perf_counter()
                array = (bags.read(branch, wsi_id)[indices] if mode_index == 0
                         else bags.read(branch, wsi_id, indices))
                read_sample_seconds = time.perf_counter() - started
                print(json.dumps({"event": "read_probe_read_done", "branch": branch,
                                  "wsi_id": wsi_id, "trial": trial, "mode": mode,
                                  "read_sample_seconds": read_sample_seconds}), flush=True)
                host_started = time.perf_counter()
                source = torch.from_numpy(array)
                host = torch.empty_like(source, memory_format=torch.channels_last,
                                        pin_memory=device.type == "cuda")
                host.copy_(source)
                prepare_seconds = read_sample_seconds + time.perf_counter() - host_started
                print(json.dumps({"event": "read_probe_host_done", "branch": branch,
                                  "wsi_id": wsi_id, "trial": trial, "mode": mode}), flush=True)
                transfer_started = time.perf_counter()
                latents = host.to(device, non_blocking=True)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                row = {"branch": branch, "wsi_id": wsi_id, "trial": trial, "mode": mode,
                       "bag_size": location.count, "sampled_bag_size": len(indices),
                       "sample_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
                       "requested_bytes": (location.count if mode_index == 0 else len(indices)) * RECORD_BYTES,
                       "read_regions": 1 if mode_index == 0 else int(np.count_nonzero(np.diff(indices) != 1)) + 1,
                       "read_sample_seconds": read_sample_seconds, "prepare_seconds": prepare_seconds,
                       "transfer_seconds": time.perf_counter() - transfer_started}
                rows.append(row)
                with (directory / "read_probe_rows.jsonl").open("a") as partial:
                    partial.write(json.dumps(row) + "\n")
                print(json.dumps({"event": "read_probe_trial_done", **row}), flush=True)
                del array, source, host, latents
        result = {"branch": branch, "seed": config["seed"], "epoch": 0,
                  "patch_retention": config["patch_retention"], "wsi_ids": selected,
                  "cache_policy": "natural_cache_no_eviction_opposite_method_order_across_branches",
                  "source_commit": subprocess.check_output(
                      ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True).strip(),
                  "rows": rows}
        (directory / "read_probe.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"event": "read_probe_done", "branch": branch, "trials": len(rows)}), flush=True)
    finally:
        bags.close()


def run_smoke(repo_root: Path, latent_root: Path, output_root: Path, branch: str, effective_batch: int) -> None:
    """Eight real optimizer updates, restoring after four; train-only T1/T2."""
    config = json.loads((repo_root / "docs/data/spec0054_fold0_run.json").read_text())
    config["effective_batch"] = effective_batch
    config["t1_train_wsi"] = 26190  # Median-sized fold-train WSI, 7,869 patches.
    contract = json.loads((repo_root / "docs/data/spec0054_fp16_latent_extraction.json").read_text())
    bags = ShardBags(
        latent_root, contract, repo_root / "docs/data/spec0054_cohort_folds.csv",
        {int(item["shard"]): item["result_sha256"] for item in config["sources"]},
    )
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
                  "saved_telemetry_rows": rows}
        # One inference-only profile confirms the native CLS attention backend.
        model = _make_classifier(checkpoint["model"], torch.device("cuda:0"),
                                 hidden_width=config["attention_hidden_width"])
        array = bags.read(branch, config["t1_train_wsi"])[:16].copy()
        latents = torch.from_numpy(array).to("cuda", memory_format=torch.channels_last)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as profile:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                model(latents)
            torch.cuda.synchronize()
        result["attention_kernel_names"] = sorted({event.key for event in profile.key_averages()
                                                     if "attention" in event.key.lower() or "fmha" in event.key.lower()})
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
    args = parser.parse_args()
    if args.read_probe:
        run_read_probe(args.repo_root, args.latent_root, args.output_root, args.branch)
    elif args.smoke:
        run_smoke(args.repo_root, args.latent_root, args.output_root, args.branch, args.effective_batch)
    else:
        run(args.repo_root, args.latent_root, args.output_root, args.resume_root, args.effective_batch, args.branch)
