"""First paired Spec 0054 fold, reading complete FP16 WSI bags in place."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import shutil
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional

from experiments.spec0054_mil_a0_probe import _compiled_closure, _make_classifier, _make_optimizer
from experiments.spec0054_shard_bags import ShardBags
from eqvae.models.local_global_mil import LocalGlobalMILClassifier, build_local_attention_graph
from eqvae.training.mil_dynamics import (
    AdamWTelemetry, EagerLayerProbe, ExampleDynamicsTracker,
    parameter_optimizer_records, semantic_capture_modules, t0_forward_records,
)
from eqvae.training.mil_dynamics_checkpoint import (
    build_dynamics_checkpoint, load_dynamics_checkpoint,
    restore_dynamics_checkpoint, save_dynamics_checkpoint,
    should_pause_for_session,
)
from eqvae.training.mil_dynamics_step import classification_example_record
from eqvae.training.mil_t2_lite import flatten_t2_records, run_t2_lite
from eqvae.training.mil_telemetry_io import load_telemetry_table, write_telemetry_table
from eqvae.training.supervised_pairing import paired_epoch_order

BRANCHES = ("normal_vae", "so2_vae")


@dataclass(frozen=True)
class Instance:
    wsi_id: int
    x: int
    y: int


class ExposureSchedule:
    """The historical warmup/cosine curve indexed by WSI exposures."""

    def __init__(self, optimizer: torch.optim.Optimizer, train_count: int, config: dict) -> None:
        self.optimizer = optimizer
        self.train_count = train_count
        self.config = config
        self.last_exposure = 0

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
        return {"last_exposure": self.last_exposure}

    def load_state_dict(self, state: dict) -> None:
        self.last_exposure = int(state["last_exposure"])


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
) -> bool:
    device = torch.device("cuda:0")
    directory = output_root / branch
    directory.mkdir(parents=True, exist_ok=False)
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
    graphs = {
        wsi_id: build_local_attention_graph(
            [Instance(wsi_id, x, y) for x, y in bags.locations[wsi_id].coordinates],
            expected_instance_count=bags.locations[wsi_id].count,
        )
        for wsi_id in ids
    }
    covariates = {}
    for wsi_id, graph in graphs.items():
        lattice = graph.lattice_coordinates
        extent = lattice.amax(dim=0) - lattice.amin(dim=0) + 1
        covariates[wsi_id] = {
            "graph_degree_mean": float(graph.neighbor_valid.sum().item() / bags.locations[wsi_id].count),
            "lattice_occupancy": float(bags.locations[wsi_id].count / extent.prod().item()),
        }
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    initial_state = {name: value.detach().cpu().clone() for name, value in LocalGlobalMILClassifier().state_dict().items()}
    model = _make_classifier(initial_state, device, torch)
    optimizer = _make_optimizer(model, torch)
    scaler = torch.amp.GradScaler("cuda")
    telemetry = AdamWTelemetry()
    dynamics = ExampleDynamicsTracker(ids)
    scheduler = ExposureSchedule(optimizer, count, config)
    compiled = _compiled_closure(model, instrumented=True, torch=torch)
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

    def load(wsi_id: int) -> tuple[torch.Tensor, object]:
        array = bags.read(branch, wsi_id).copy()
        latents = torch.from_numpy(array).to(device=device, memory_format=torch.channels_last)
        return latents, graphs[wsi_id].to(device)

    def evaluate() -> None:
        model.eval()
        with torch.inference_mode():
            for split, selected in (("train", train_ids), ("validation", validation_ids)):
                for wsi_id in selected:
                    latents, graph = load(wsi_id)
                    target_index = bags.locations[wsi_id].diagnosis_index
                    with torch.autocast("cuda", dtype=torch.float16):
                        logits, representations = model.forward_with_representations(latents, graph)
                        embedding = representations[-1].detach().float()
                        loss = functional.cross_entropy(
                            logits.float().unsqueeze(0), torch.tensor([target_index], device=device)
                        )
                    record = classification_example_record(
                        logits=logits, target=target_index, unweighted_loss=loss,
                        weighted_loss=loss * weights[target_index],
                    )
                    dynamics.update(
                        wsi_id=wsi_id, boundary=exposure,
                        correct=bool(record["correct"]),
                        margin=float(record["true_class_margin"]),
                        true_probability=float(record["true_probability"]),
                    )
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
                    del latents, graph, representations, embedding
        model.train()

    def diagnostics() -> None:
        if exposure == 0 or _at_epoch(exposure, count, config["t1_early_epochs"], config["t1_every_epochs"]):
            wsi_id = int(config["t1_train_wsi"])
            latents, graph = load(wsi_id)
            target_index = bags.locations[wsi_id].diagnosis_index
            target = torch.tensor([target_index], device=device)
            def diagnostic_loss(probe: LocalGlobalMILClassifier) -> torch.Tensor:
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = probe(latents, graph)
                    return functional.cross_entropy(logits.float().unsqueeze(0), target) * weights[target_index]
            probe_model = copy.deepcopy(model).train()
            probe_optimizer = _make_optimizer(probe_model, torch)
            probe_optimizer.load_state_dict(optimizer.state_dict())
            probe_model.zero_grad(set_to_none=True)
            with EagerLayerProbe(semantic_capture_modules(probe_model)) as probe:
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
            del probe_model, probe_optimizer, latents, graph
        if exposure == 0 or _at_epoch(exposure, count, config["t2_early_epochs"], config["t2_every_epochs"]):
            for wsi_id in (int(config["t1_train_wsi"]), int(config["t2_holdout_wsi"])):
                latents, graph = load(wsi_id)
                with torch.autocast("cuda", dtype=torch.float16):
                    result = run_t2_lite(model, latents, graph)
                tables["t2"].extend(flatten_t2_records(
                    result, identity={"update": update, "exposures": exposure, "wsi_id": wsi_id},
                ))
                del latents, graph

    snapshots = {epoch * count for epoch in config["snapshot_epochs"]}
    if config["snapshot_first_half_epoch"]:
        snapshots.add(count // 2)

    def gradient_probe() -> None:
        if exposure // count not in config["gradient_probe_epochs"] or exposure % count:
            return
        sampled_ids = [int(value) for value in config["gradient_probe_wsi_ids"]]
        probe_model = copy.deepcopy(model).train()
        vectors = []
        for wsi_id in sampled_ids:
            probe_model.zero_grad(set_to_none=True)
            latents, graph = load(wsi_id)
            target_index = bags.locations[wsi_id].diagnosis_index
            with torch.autocast("cuda", dtype=torch.float16):
                logits = probe_model(latents, graph)
                loss = functional.cross_entropy(
                    logits.float().unsqueeze(0), torch.tensor([target_index], device=device)
                ) * weights[target_index]
            loss.backward()
            vectors.append(torch.cat([
                parameter.grad.detach().float().reshape(-1)
                if parameter.grad is not None else torch.zeros_like(parameter).reshape(-1)
                for parameter in probe_model.parameters()
            ]))
            del latents, graph
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

    def save(name: str = "latest.pt") -> None:
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

    horizon = config["epochs"] * count
    boundaries = _boundaries(count, config)
    if exposure == 0 and not resumed:
        evaluate()
        diagnostics()
        gradient_probe()
        save()
    while exposure < horizon:
        selected = order[cursor:cursor + config["effective_batch"]]
        selected_ids = [train_ids[index] for index in selected]
        next_exposure = exposure + len(selected_ids)
        lr = scheduler.set_next(next_exposure)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        step_started = time.perf_counter()
        skipped = 0
        while True:
            optimizer.zero_grad(set_to_none=True)
            observations = []
            initial_scale = float(scaler.get_scale())
            for wsi_id in selected_ids:
                latents, graph = load(wsi_id)
                target_index = bags.locations[wsi_id].diagnosis_index
                target = torch.tensor([target_index], device=device)
                weight = torch.tensor(weights[target_index], device=device)
                logits, unweighted, weighted, summary = compiled(latents, graph, target, weight)
                scaler.scale(weighted / len(selected_ids)).backward()
                observations.append((wsi_id, logits.detach(), unweighted.detach(), weighted.detach(), summary.detach()))
                del latents, graph
            scaler.unscale_(optimizer)
            telemetry.begin_step(model, optimizer)
            scaler.step(optimizer)
            scaler.update()
            committed = float(scaler.get_scale()) >= initial_scale
            optimizer_record = telemetry.finish_step(
                model, next_committed_update=update + 1, committed=committed,
            )
            optimizer.zero_grad(set_to_none=True)
            if committed:
                break
            skipped += 1
        torch.cuda.synchronize(device)
        step_seconds = time.perf_counter() - step_started
        peak_memory_bytes = torch.cuda.max_memory_allocated(device)
        update += 1
        exposure = next_exposure
        cursor += len(selected_ids)
        scheduler.last_exposure = exposure
        for wsi_id, logits, unweighted, weighted, summary in observations:
            target_index = bags.locations[wsi_id].diagnosis_index
            record = classification_example_record(
                logits=logits, target=target_index,
                unweighted_loss=unweighted, weighted_loss=weighted,
            )
            forward = {
                f"forward.{capture}.{metric}": value
                for capture, values in t0_forward_records(summary).items()
                for metric, value in values.items()
            }
            tables["t0"].append({
                "update": update, "exposures": exposure, "wsi_id": wsi_id,
                "effective_batch": len(selected_ids), "lr": lr,
                "bag_size": bags.locations[wsi_id].count,
                **covariates[wsi_id],
                "amp_scale_before": initial_scale,
                "amp_scale_after": float(scaler.get_scale()),
                "amp_skipped_attempts": skipped,
                "step_seconds": step_seconds, "peak_allocated_bytes": peak_memory_bytes,
                **record, **forward,
                **{f"optimizer.{key}": value for key, value in optimizer_record.items()},
            })
        if cursor == count:
            epoch += 1
            cursor = 0
            order = paired_epoch_order(row_count=count, epoch=epoch, seed=config["seed"])
        if exposure in boundaries:
            evaluate()
            diagnostics()
            gradient_probe()
        if (
            exposure % config["checkpoint_every_exposures"] == 0
            or exposure in boundaries
            or cursor == 0
            or should_pause_for_session(session_started_unix=session_started, now_unix=time.time())
        ):
            save()
        if update % 100 == 0 or exposure in boundaries:
            print(json.dumps({"branch": branch, "update": update, "exposures": exposure, "epoch": epoch}), flush=True)
        if should_pause_for_session(session_started_unix=session_started, now_unix=time.time()):
            return False
    save("final.pt")
    return True


def run(repo_root: Path, latent_root: Path, output_root: Path, resume_root: Path | None, effective_batch: int) -> None:
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
        "config_sha256": _sha256(config_path), "cohort_sha256": _sha256(cohort_path),
        "extraction_sha256": _sha256(extraction_path), "effective_batch": effective_batch,
    }
    output_root.mkdir(parents=True, exist_ok=False)
    for branch in BRANCHES:
        if not run_branch(branch, bags, config, output_root, resume_root, identity, session_started):
            break


def run_smoke(repo_root: Path, latent_root: Path, output_root: Path) -> None:
    """Two complete bags per branch: mounted input, AMP/T0, exact resume, T1/T2-lite."""
    config_path = repo_root / "docs/data/spec0054_fold0_run.json"
    config = json.loads(config_path.read_text())
    config["effective_batch"] = 1
    contract = json.loads((repo_root / "docs/data/spec0054_fp16_latent_extraction.json").read_text())
    bags = ShardBags(
        latent_root, contract, repo_root / "docs/data/spec0054_cohort_folds.csv",
        {int(item["shard"]): item["result_sha256"] for item in config["sources"]},
    )
    selected_ids = (3672, 61100)
    train_ids = [wsi_id for wsi_id, bag in bags.locations.items() if bag.fold != config["fold"]]
    counts = Counter(bags.locations[wsi_id].diagnosis_index for wsi_id in train_ids)
    weights = [len(train_ids) / (5 * counts[index]) for index in range(5)]
    output_root.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda:0")
    result = {
        "source_commit": subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True,
        ).strip(),
        "config_sha256": _sha256(config_path),
        "indexed_shards": len(contract["shards"]),
        "indexed_wsi": len(bags.locations),
        "selected_wsi": list(selected_ids),
        "branches": {},
    }
    for branch in BRANCHES:
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        initial = {
            name: value.detach().cpu().clone()
            for name, value in LocalGlobalMILClassifier().state_dict().items()
        }
        model = _make_classifier(initial, device, torch)
        optimizer = _make_optimizer(model, torch)
        scaler = torch.amp.GradScaler("cuda")
        scheduler = ExposureSchedule(optimizer, len(train_ids), config)
        telemetry = AdamWTelemetry()
        dynamics = ExampleDynamicsTracker(sorted(bags.locations))
        compiled = _compiled_closure(model, instrumented=True, torch=torch)
        branch_dir = output_root / branch
        branch_dir.mkdir()
        records = []
        resume_exact = None
        for step, wsi_id in enumerate(selected_ids, start=1):
            location = bags.locations[wsi_id]
            graph = build_local_attention_graph(
                [Instance(wsi_id, x, y) for x, y in location.coordinates],
                expected_instance_count=location.count,
            ).to(device)
            latents = torch.from_numpy(bags.read(branch, wsi_id).copy()).to(
                device=device, memory_format=torch.channels_last,
            )
            target = torch.tensor([location.diagnosis_index], device=device)
            weight = torch.tensor(weights[location.diagnosis_index], device=device)
            lr = scheduler.set_next(step)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            skipped = 0
            while True:
                optimizer.zero_grad(set_to_none=True)
                scale_before = float(scaler.get_scale())
                logits, unweighted, weighted, summary = compiled(latents, graph, target, weight)
                scaler.scale(weighted).backward()
                scaler.unscale_(optimizer)
                telemetry.begin_step(model, optimizer)
                scaler.step(optimizer)
                scaler.update()
                committed = float(scaler.get_scale()) >= scale_before
                optimizer_record = telemetry.finish_step(
                    model, next_committed_update=step, committed=committed,
                )
                if committed:
                    break
                skipped += 1
            torch.cuda.synchronize(device)
            scheduler.last_exposure = step
            records.append({
                "wsi_id": wsi_id,
                "shard": location.shard,
                "bag_size": location.count,
                "lr": lr,
                "unweighted_loss": float(unweighted.item()),
                "weighted_loss": float(weighted.item()),
                "amp_scale_before": scale_before,
                "amp_scale_after": float(scaler.get_scale()),
                "amp_skipped_attempts": skipped,
                "step_seconds": time.perf_counter() - started,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "t0_forward_captures": len(t0_forward_records(summary)),
                "t0_optimizer_fields": len(optimizer_record),
            })
            if step == 1:
                checkpoint = branch_dir / "latest.pt"
                save_dynamics_checkpoint(checkpoint, build_dynamics_checkpoint(
                    model=model, optimizer=optimizer, scaler=scaler, scheduler=scheduler,
                    telemetry=telemetry, dynamics=dynamics, committed_update=1,
                    exposure_count=1, epoch=0, within_epoch_cursor=1,
                    current_order=(0, 1), effective_batch_size=1,
                ))
                restored_model = _make_classifier(initial, device, torch)
                restored_optimizer = _make_optimizer(restored_model, torch)
                restored_scaler = torch.amp.GradScaler("cuda")
                restored_scheduler = ExposureSchedule(restored_optimizer, len(train_ids), config)
                restored_telemetry = AdamWTelemetry()
                restored_dynamics = ExampleDynamicsTracker(sorted(bags.locations))
                progress = restore_dynamics_checkpoint(
                    load_dynamics_checkpoint(checkpoint), model=restored_model,
                    optimizer=restored_optimizer, scaler=restored_scaler,
                    scheduler=restored_scheduler, telemetry=restored_telemetry,
                    dynamics=restored_dynamics,
                )
                resume_exact = (
                    progress.committed_update == 1 and progress.exposure_count == 1
                    and progress.within_epoch_cursor == 1
                    and all(torch.equal(value, restored_model.state_dict()[name])
                            for name, value in model.state_dict().items())
                )
                model, optimizer, scaler = restored_model, restored_optimizer, restored_scaler
                scheduler, telemetry, dynamics = restored_scheduler, restored_telemetry, restored_dynamics
                compiled = _compiled_closure(model, instrumented=True, torch=torch)
            del latents, graph
        probe_model = copy.deepcopy(model).train()
        location = bags.locations[selected_ids[-1]]
        graph = build_local_attention_graph(
            [Instance(selected_ids[-1], x, y) for x, y in location.coordinates],
            expected_instance_count=location.count,
        ).to(device)
        latents = torch.from_numpy(bags.read(branch, selected_ids[-1]).copy()).to(
            device=device, memory_format=torch.channels_last,
        )
        probe_model.zero_grad(set_to_none=True)
        with EagerLayerProbe(semantic_capture_modules(probe_model)) as probe:
            with torch.autocast("cuda", dtype=torch.float16):
                logits = probe_model(latents, graph)
                loss = functional.cross_entropy(
                    logits.float().unsqueeze(0),
                    torch.tensor([location.diagnosis_index], device=device),
                ) * weights[location.diagnosis_index]
            loss.backward()
            t1_captures = len(probe.records())
        with torch.autocast("cuda", dtype=torch.float16):
            t2 = run_t2_lite(model, latents, graph)
        result["branches"][branch] = {
            "updates": records,
            "checkpoint_resume_exact": resume_exact,
            "t1_layer_captures": t1_captures,
            "t2_lite_records": len(flatten_t2_records(t2, identity={
                "update": 2, "exposures": 2, "wsi_id": selected_ids[-1],
            })),
        }
        print(json.dumps({"branch": branch, **result["branches"][branch]}), flush=True)
    (output_root / "smoke.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume-root", type=Path)
    parser.add_argument("--effective-batch", type=int, choices=(1, 4, 8), default=1)
    args = parser.parse_args()
    run(args.repo_root, args.latent_root, args.output_root, args.resume_root, args.effective_batch)
