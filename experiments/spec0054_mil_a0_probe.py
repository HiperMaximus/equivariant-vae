# Copyright 2026 HiperMaximus
"""Run the bounded Spec 0054 cohort and telemetry A0 probe on Kaggle."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import struct
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

INPUT_ROOT = Path("/kaggle/input")
OUTPUT_ROOT = Path("/kaggle/working/spec0054_mil_a0")
CONTRACT_RELATIVE = Path("docs/data/spec0054_a0_contract.json")
COHORT_RELATIVE = Path("docs/data/spec0054_cohort_folds.csv")
HISTORICAL_RELATIVE = Path("docs/data/ubc_ocean_masked_holdout_ids.csv")
PATCH_HEADER_FORMAT = "<8sIQiiii3s25x"
PATCH_HEADER_BYTES = 64
PATCH_RECORD_BYTES = 3 * 256 * 256
ENCODER_BATCH_SIZE = 8
CLASSIFIER_SEED = 1701
PEAK_LR = 2e-4
MAX_AMP_RETRIES = 16


@dataclass(frozen=True)
class PatchRow:
    source_role: str
    source_row_index: int
    wsi_id: int
    label: int
    x: int
    y: int


@dataclass(frozen=True)
class GraphInstance:
    wsi_id: int
    x: int
    y: int


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _unique_input(name: str, expected_sha256: str | None = None) -> Path:
    matches = [path for path in INPUT_ROOT.rglob(name) if path.is_file()]
    if expected_sha256 is not None:
        matches = [path for path in matches if _sha256(path) == expected_sha256]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one mounted input named {name}, found {len(matches)}")
    return matches[0]


def _load_patch_rows(path: Path, role: str, expected_rows: int) -> list[PatchRow]:
    rows: list[PatchRow] = []
    identities: set[tuple[int, int, int]] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for source_row_index, row in enumerate(csv.DictReader(handle)):
            patch = PatchRow(
                source_role=role,
                source_row_index=source_row_index,
                wsi_id=int(row["wsi_id"]),
                label=int(row["label"]),
                x=int(row["x"]),
                y=int(row["y"]),
            )
            identity = (patch.wsi_id, patch.x, patch.y)
            if identity in identities:
                raise RuntimeError(f"Duplicate patch identity in {role}: {identity}")
            identities.add(identity)
            rows.append(patch)
    if len(rows) != expected_rows:
        raise RuntimeError(f"Unexpected {role} row count: {len(rows)}")
    return rows


def _validate_binary(
    path: Path,
    expected_rows: int,
    expected_bytes: int,
    expected_crc32: int,
) -> dict[str, Any]:
    if path.stat().st_size != expected_bytes:
        raise RuntimeError(f"Patch binary size differs: {path.name}")
    with path.open("rb") as handle:
        header = handle.read(PATCH_HEADER_BYTES)
    values = struct.unpack(PATCH_HEADER_FORMAT, header)
    magic, crc32, count, channels, height, width, version, layout = values
    if (
        magic != b"UBC_DATA"
        or count != expected_rows
        or crc32 != expected_crc32
        or (channels, height, width, version, layout) != (3, 256, 256, 1, b"CHW")
    ):
        raise RuntimeError(f"Patch binary header differs: {path.name}")
    return {
        "name": path.name,
        "bytes": path.stat().st_size,
        "crc32": int(crc32),
        "rows": int(count),
        "shape": [int(channels), int(height), int(width)],
        "layout": layout.decode(),
    }


def _validate_atlas(
    atlas_path: Path,
    train_rows: list[PatchRow],
    validation_rows: list[PatchRow],
) -> None:
    class_index = {"CC": 0, "EC": 1, "HGSC": 2, "LGSC": 3, "MC": 4}
    observed: dict[str, set[tuple[int, int, int, int]]] = {
        "train": set(),
        "valid": set(),
    }
    with atlas_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            observed[row["split"]].add(
                (
                    int(row["image_id"]),
                    class_index[row["label"]],
                    int(row["x"]),
                    int(row["y"]),
                )
            )
    expected = {
        "train": {(row.wsi_id, row.label, row.x, row.y) for row in train_rows},
        "valid": {
            (row.wsi_id, row.label, row.x, row.y) for row in validation_rows
        },
    }
    if observed != expected:
        raise RuntimeError("Mounted patch CSV identities differ from canonical atlas")


def _write_instance_manifest(rows: list[PatchRow]) -> dict[str, Any]:
    logical = sorted(
        range(len(rows)),
        key=lambda index: (
            rows[index].wsi_id,
            rows[index].y,
            rows[index].x,
            rows[index].source_role,
            rows[index].source_row_index,
        ),
    )
    logical_rank = np.empty(len(rows), dtype=np.int32)
    logical_rank[np.asarray(logical, dtype=np.int64)] = np.arange(
        len(rows), dtype=np.int32
    )
    path = OUTPUT_ROOT / "spec0054_patch_instances.npz"
    np.savez_compressed(
        path,
        source_role=np.asarray(
            [row.source_role == "vae_validation" for row in rows], dtype=np.bool_
        ),
        source_row_index=np.asarray(
            [row.source_row_index for row in rows], dtype=np.int32
        ),
        logical_rank=logical_rank,
        wsi_id=np.asarray([row.wsi_id for row in rows], dtype=np.int32),
        label=np.asarray([row.label for row in rows], dtype=np.uint8),
        x=np.asarray([row.x for row in rows], dtype=np.int32),
        y=np.asarray([row.y for row in rows], dtype=np.int32),
    )
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _validate_cohort(
    repo_root: Path,
    contract: dict[str, Any],
    rows: list[PatchRow],
) -> dict[int, dict[str, Any]]:
    cohort_path = repo_root / COHORT_RELATIVE
    if _sha256(cohort_path) != contract["cohort_csv_sha256"]:
        raise RuntimeError("Cohort/fold table hash differs")
    with cohort_path.open(newline="", encoding="utf-8") as handle:
        cohort = {int(row["wsi_id"]): row for row in csv.DictReader(handle)}
    grouped: dict[int, list[PatchRow]] = defaultdict(list)
    for row in rows:
        grouped[row.wsi_id].append(row)
    if set(cohort) != set(grouped) or len(cohort) != 361:
        raise RuntimeError("Cohort WSI identities differ")
    for wsi_id, patches in grouped.items():
        labels = {row.label for row in patches}
        if len(labels) != 1:
            raise RuntimeError(f"WSI label differs: {wsi_id}")
        row = cohort[wsi_id]
        if int(row["diagnosis_index"]) != next(iter(labels)) or int(
            row["patch_count"]
        ) != len(patches):
            raise RuntimeError(f"Cohort summary differs: {wsi_id}")
    fold_counts = Counter(int(row["fold"]) for row in cohort.values())
    if sorted(fold_counts.values()) != [72, 72, 72, 72, 73]:
        raise RuntimeError("Five-fold WSI sizes differ")
    with (repo_root / HISTORICAL_RELATIVE).open(newline="", encoding="utf-8") as handle:
        historical = {int(row["image_id"]) for row in csv.DictReader(handle)}
    if historical & set(cohort):
        raise RuntimeError("Development cohort overlaps historical 152 WSI")
    return cohort


def _read_patch_batch(binary: Path, rows: list[PatchRow]) -> np.ndarray:
    descriptor = os.open(binary, os.O_RDONLY)
    try:
        payloads = [
            os.pread(
                descriptor,
                PATCH_RECORD_BYTES,
                PATCH_HEADER_BYTES + row.source_row_index * PATCH_RECORD_BYTES,
            )
            for row in rows
        ]
    finally:
        os.close(descriptor)
    if any(len(payload) != PATCH_RECORD_BYTES for payload in payloads):
        raise RuntimeError("Incomplete random-access patch read")
    return np.frombuffer(b"".join(payloads), dtype=np.uint8).reshape(-1, 3, 256, 256).copy()


def _load_frozen_model(model_name: str, state_path: Path, device: Any, torch: Any) -> Any:
    from eqvae.models.registry import (
        MODEL_KIND_NON_EQ_TRANSLATABLE,
        MODEL_KIND_SO2_FIXED,
        build_model,
    )

    model_kind = (
        MODEL_KIND_NON_EQ_TRANSLATABLE
        if model_name == "normal_vae"
        else MODEL_KIND_SO2_FIXED
    )
    model = build_model(model_kind)
    model.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=True))
    return model.to(device).eval().requires_grad_(False)


def _encode_bags(
    selected_ids: set[int],
    rows: list[PatchRow],
    binary_by_role: dict[str, Path],
    state_paths: dict[str, Path],
    torch: Any,
) -> tuple[dict[str, dict[int, Any]], dict[str, float]]:
    by_wsi: dict[int, list[PatchRow]] = defaultdict(list)
    for row in rows:
        if row.wsi_id in selected_ids:
            by_wsi[row.wsi_id].append(row)
    if set(by_wsi) != selected_ids:
        raise RuntimeError("A0 sentinel selection is incomplete")
    for patches in by_wsi.values():
        patches.sort(key=lambda row: (row.y, row.x, row.source_row_index))
        if len({row.source_role for row in patches}) != 1:
            raise RuntimeError("One WSI spans physical source roles")
    models = {
        "normal_vae": _load_frozen_model(
            "normal_vae", state_paths["normal_vae"], torch.device("cuda:0"), torch
        ),
        "so2_vae": _load_frozen_model(
            "so2_vae", state_paths["so2_vae"], torch.device("cuda:1"), torch
        ),
    }
    output: dict[str, dict[int, Any]] = {name: {} for name in models}
    timings = {name: 0.0 for name in models}
    try:
        for wsi_id in sorted(by_wsi):
            patches = by_wsi[wsi_id]
            parts: dict[str, list[Any]] = {name: [] for name in models}
            for start in range(0, len(patches), ENCODER_BATCH_SIZE):
                batch_rows = patches[start : start + ENCODER_BATCH_SIZE]
                array = _read_patch_batch(
                    binary_by_role[batch_rows[0].source_role], batch_rows
                )
                images = torch.from_numpy(array).float().div_(255).mul_(2).sub_(1)
                for model_name, model in models.items():
                    device = next(model.parameters()).device
                    torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    with torch.inference_mode():
                        mean, _ = model.encode(images.to(device))
                    torch.cuda.synchronize(device)
                    timings[model_name] += time.perf_counter() - started
                    parts[model_name].append(mean.detach().cpu())
            for model_name in models:
                latent = torch.cat(parts[model_name])
                if tuple(latent.shape[1:]) != (16, 32, 32) or not torch.isfinite(
                    latent
                ).all():
                    raise RuntimeError(f"Nonfinite or malformed latent bag: {model_name}")
                output[model_name][wsi_id] = latent
    finally:
        del models
        torch.cuda.empty_cache()
    return output, timings


def _state_max_difference(left: Any, right: Any, torch: Any) -> float:
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or left.shape != right.shape:
            return math.inf
        if left.dtype == torch.bool or not left.dtype.is_floating_point:
            return 0.0 if torch.equal(left.cpu(), right.cpu()) else math.inf
        return float((left.detach().cpu().float() - right.detach().cpu().float()).abs().max())
    if isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):
            return math.inf
        return max((_state_max_difference(left[key], right[key], torch) for key in left), default=0.0)
    if isinstance(left, (list, tuple)):
        if not isinstance(right, type(left)) or len(left) != len(right):
            return math.inf
        return max((_state_max_difference(a, b, torch) for a, b in zip(left, right, strict=True)), default=0.0)
    return 0.0 if left == right else math.inf


def _make_classifier(state: dict[str, Any], device: Any, torch: Any) -> Any:
    from eqvae.models.local_attention_candidates import use_whole_bag_fixed25_attention
    from eqvae.models.local_global_mil import LocalGlobalMILClassifier

    model = LocalGlobalMILClassifier()
    model.load_state_dict(state)
    use_whole_bag_fixed25_attention(model)
    return model.to(  # pyright: ignore[reportCallIssue]
        device=device, memory_format=torch.channels_last
    ).train()


def _make_optimizer(model: Any, torch: Any) -> Any:
    from eqvae.models.local_global_mil import local_global_mil_adamw_parameter_groups

    return torch.optim.AdamW(
        local_global_mil_adamw_parameter_groups(model),
        lr=PEAK_LR,
        betas=(0.9, 0.999),
        eps=1e-8,
        fused=True,
    )


def _compiled_closure(model: Any, *, instrumented: bool, torch: Any) -> Any:
    from eqvae.training.mil_dynamics import T0ForwardWrapper

    module = T0ForwardWrapper(model) if instrumented else model

    def closure(latents: Any, graph: Any, target: Any, class_weight: Any) -> Any:
        with torch.autocast("cuda", dtype=torch.float16):
            if instrumented:
                logits, summary = module(latents, graph)
            else:
                logits = module(latents, graph)
            loss = torch.nn.functional.cross_entropy(logits.float().unsqueeze(0), target)
            weighted = loss * class_weight
        return (logits, loss, weighted, summary) if instrumented else (logits, loss, weighted)

    return torch.compile(
        closure,
        backend="inductor",
        fullgraph=True,
        dynamic=None,
        mode="max-autotune-no-cudagraphs",
    )


def _run_branch_probe(
    branch: str,
    device_index: int,
    classifier_state: dict[str, Any],
    latent_bags: dict[int, Any],
    rows_by_wsi: dict[int, list[PatchRow]],
    contract: dict[str, Any],
    class_weights: list[float],
    torch: Any,
) -> dict[str, Any]:
    from eqvae.models.local_global_mil import build_local_attention_graph
    from eqvae.training.mil_dynamics import (
        AdamWTelemetry,
        run_eager_layer_probe,
        t0_forward_records,
    )
    from eqvae.training.mil_dynamics_step import instrumented_adamw_attempt
    from eqvae.training.mil_dynamics_checkpoint import copy_state_to_cpu
    from eqvae.training.mil_t2_lite import flatten_t2_records, run_t2_lite
    from eqvae.training.mil_telemetry_io import write_telemetry_table

    device = torch.device(f"cuda:{device_index}")
    cost_ids = {
        name: int(record["wsi_id"]) for name, record in contract["cost_panel"].items()
    }
    median_id = cost_ids["median"]
    selected_ids = {
        int(record["wsi_id"]) for record in contract["sentinels"].values()
    } | set(cost_ids.values())
    graphs = {}
    for wsi_id in selected_ids:
        instances = [
            GraphInstance(row.wsi_id, row.x, row.y) for row in rows_by_wsi[wsi_id]
        ]
        graphs[wsi_id] = build_local_attention_graph(
            instances, expected_instance_count=len(instances)
        ).to(device)
    latents = {
        wsi_id: latent_bags[wsi_id].to(device=device, memory_format=torch.channels_last)
        for wsi_id in selected_ids
    }
    targets = {wsi_id: rows_by_wsi[wsi_id][0].label for wsi_id in selected_ids}

    target = torch.tensor([targets[median_id]], device=device)
    weight = torch.tensor(class_weights[targets[median_id]], device=device)
    median_latent, median_graph = latents[median_id], graphs[median_id]

    def run_baseline() -> dict[str, Any]:
        baseline = _make_classifier(classifier_state, device, torch)
        optimizer = _make_optimizer(baseline, torch)
        scaler = torch.amp.GradScaler("cuda")
        compiled = _compiled_closure(baseline, instrumented=False, torch=torch)

        def update() -> tuple[Any, Any, int]:
            skipped = 0
            while True:
                optimizer.zero_grad(set_to_none=True)
                logits, loss, weighted = compiled(
                    median_latent, median_graph, target, weight
                )
                initial_scale = float(scaler.get_scale())
                scaler.scale(weighted).backward()
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if float(scaler.get_scale()) >= initial_scale:
                    return logits, loss, skipped
                skipped += 1
                if skipped >= MAX_AMP_RETRIES:
                    raise RuntimeError("Baseline AMP retries exceeded the A0 limit")

        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        cold_started = time.perf_counter()
        logits, loss, first_skips = update()
        torch.cuda.synchronize(device)
        cold_seconds = time.perf_counter() - cold_started
        first_model_state = copy_state_to_cpu(baseline.state_dict())
        first_optimizer_state = copy_state_to_cpu(optimizer.state_dict())
        first_scaler_state = scaler.state_dict()
        steady_seconds: list[float] = []
        skipped_attempts = [first_skips]
        for _ in range(2, 5):
            torch.cuda.synchronize(device)
            step_started = time.perf_counter()
            _, _, skipped = update()
            torch.cuda.synchronize(device)
            steady_seconds.append(time.perf_counter() - step_started)
            skipped_attempts.append(skipped)
        return {
            "logits": logits.detach().float().cpu(),
            "loss": float(loss.detach().float().cpu()),
            "first_model_state": first_model_state,
            "first_optimizer_state": first_optimizer_state,
            "first_scaler_state": first_scaler_state,
            "final_model_state": copy_state_to_cpu(baseline.state_dict()),
            "cold_seconds": cold_seconds,
            "steady_seconds": steady_seconds,
            "skipped_attempts": skipped_attempts,
            "peak_allocated": torch.cuda.max_memory_allocated(device),
            "peak_reserved": torch.cuda.max_memory_reserved(device),
        }

    baseline_result = run_baseline()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    observed = _make_classifier(classifier_state, device, torch)
    observed_optimizer = _make_optimizer(observed, torch)
    observed_scaler = torch.amp.GradScaler("cuda")
    telemetry = AdamWTelemetry()
    observed_fn = _compiled_closure(observed, instrumented=True, torch=torch)

    summary_box: dict[str, Any] = {}

    def observed_loss() -> tuple[Any, Any, Any]:
        logits, loss, weighted, summary = observed_fn(
            median_latent, median_graph, target, weight
        )
        summary_box["value"] = summary
        return logits, loss, weighted

    torch.cuda.synchronize(device)
    cold_started = time.perf_counter()
    first_observed_skips = 0
    while True:
        first = instrumented_adamw_attempt(
            model=observed,
            optimizer=observed_optimizer,
            scaler=observed_scaler,
            telemetry=telemetry,
            next_committed_update=1,
            forward_loss=observed_loss,
        )
        if first.committed:
            break
        first_observed_skips += 1
        if first_observed_skips >= MAX_AMP_RETRIES:
            raise RuntimeError("T0 AMP retries exceeded the A0 limit")
    torch.cuda.synchronize(device)
    observed_cold = time.perf_counter() - cold_started
    observed_peak_allocated = torch.cuda.max_memory_allocated(device)
    observed_peak_reserved = torch.cuda.max_memory_reserved(device)
    logit_difference = float(
        (baseline_result["logits"] - torch.tensor(first.logits)).abs().max()
    )
    model_difference = _state_max_difference(
        baseline_result["first_model_state"], observed.state_dict(), torch
    )
    optimizer_difference = _state_max_difference(
        baseline_result["first_optimizer_state"], observed_optimizer.state_dict(), torch
    )
    scaler_equal = baseline_result["first_scaler_state"] == observed_scaler.state_dict()
    loss_difference = abs(baseline_result["loss"] - first.unweighted_loss)
    first_update_passed = not (
        logit_difference > 1e-6
        or loss_difference > 1e-6
        or model_difference > 1e-7
        or optimizer_difference > 1e-7
        or not scaler_equal
        or baseline_result["skipped_attempts"][0] != first_observed_skips
    )

    observed_times: list[float] = []
    observed_skips = [first_observed_skips]
    t0_rows: list[dict[str, Any]] = []
    for update in range(2, 5):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        skipped = 0
        while True:
            result = instrumented_adamw_attempt(
                model=observed,
                optimizer=observed_optimizer,
                scaler=observed_scaler,
                telemetry=telemetry,
                next_committed_update=update,
                forward_loss=observed_loss,
            )
            if result.committed:
                break
            skipped += 1
            if skipped >= MAX_AMP_RETRIES:
                raise RuntimeError("T0 AMP retries exceeded the A0 limit")
        torch.cuda.synchronize(device)
        observed_times.append(time.perf_counter() - started)
        observed_skips.append(skipped)
        t0_rows.append(
            {
                "update": update,
                "wsi_id": median_id,
                "forward": t0_forward_records(summary_box["value"]),
                "optimizer": result.optimizer,
            }
        )
        observed_peak_allocated = max(
            observed_peak_allocated, torch.cuda.max_memory_allocated(device)
        )
        observed_peak_reserved = max(
            observed_peak_reserved, torch.cuda.max_memory_reserved(device)
        )
    final_model_difference = _state_max_difference(
        baseline_result["final_model_state"], observed.state_dict(), torch
    )
    retry_trajectory_equal = baseline_result["skipped_attempts"] == observed_skips
    trajectory_passed = final_model_difference <= 1e-7 and retry_trajectory_equal
    equivalence = {
        "passed": first_update_passed and trajectory_passed,
        "first_update_passed": first_update_passed,
        "trajectory_passed": trajectory_passed,
        "thresholds": {
            "logit_max_abs_difference": 1e-6,
            "loss_abs_difference": 1e-6,
            "model_max_abs_difference": 1e-7,
            "optimizer_max_abs_difference": 1e-7,
            "trajectory_model_max_abs_difference": 1e-7,
        },
        "logit_max_abs_difference": logit_difference,
        "loss_abs_difference": loss_difference,
        "model_max_abs_difference": model_difference,
        "optimizer_max_abs_difference": optimizer_difference,
        "scaler_equal": scaler_equal,
        "measured_trajectory_model_max_abs_difference": final_model_difference,
        "retry_trajectory_equal": retry_trajectory_equal,
        "baseline_amp_skipped_attempts_by_update": baseline_result[
            "skipped_attempts"
        ],
        "t0_amp_skipped_attempts_by_update": observed_skips,
    }
    _write_json(OUTPUT_ROOT / f"equivalence_{branch}.json", equivalence)
    print(json.dumps({"branch": branch, "equivalence": equivalence}), flush=True)

    t1_results: dict[str, Any] = {}
    t2_rows: list[dict[str, Any]] = []
    diagnostic_ids = set(cost_ids.values()) | {
        int(record["wsi_id"]) for record in contract["sentinels"].values()
    }
    for wsi_id in sorted(diagnostic_ids):
        target_value = torch.tensor([targets[wsi_id]], device=device)
        class_weight = torch.tensor(class_weights[targets[wsi_id]], device=device)

        def t1_loss(model: Any) -> Any:
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(latents[wsi_id], graphs[wsi_id])
                loss = torch.nn.functional.cross_entropy(
                    logits.float().unsqueeze(0), target_value
                )
            return loss * class_weight

        if wsi_id in set(cost_ids.values()):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            layer_records = run_eager_layer_probe(observed, t1_loss)
            torch.cuda.synchronize(device)
            t1_results[str(wsi_id)] = {
                "seconds": time.perf_counter() - started,
                "records": layer_records,
            }
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        t2 = run_t2_lite(observed, latents[wsi_id], graphs[wsi_id])
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        t2_rows.extend(
            flatten_t2_records(
                t2,
                identity={
                    "update": 4,
                    "wsi_id": wsi_id,
                    "branch_index": 0 if branch == "normal_vae" else 1,
                    "seconds": elapsed,
                },
            )
        )
    write_telemetry_table(OUTPUT_ROOT / f"t2_{branch}.npz", t2_rows)
    _write_json(OUTPUT_ROOT / f"t0_{branch}.json", t0_rows)
    _write_json(OUTPUT_ROOT / f"t1_{branch}.json", t1_results)
    return {
        "branch": branch,
        "first_update_equivalence": equivalence,
        "cold_seconds": {
            "baseline": baseline_result["cold_seconds"],
            "t0": observed_cold,
        },
        "steady_seconds": {
            "baseline": baseline_result["steady_seconds"],
            "t0": observed_times,
            "t0_over_baseline_ratio": sum(observed_times)
            / sum(baseline_result["steady_seconds"]),
        },
        "peak_allocated_bytes": {
            "baseline": baseline_result["peak_allocated"],
            "t0": observed_peak_allocated,
        },
        "peak_reserved_bytes": {
            "baseline": baseline_result["peak_reserved"],
            "t0": observed_peak_reserved,
        },
        "graph_identities": {
            str(wsi_id): graphs[wsi_id].identity_sha256 for wsi_id in sorted(graphs)
        },
        "t1_wsi_ids": sorted(int(value) for value in t1_results),
        "t2_wsi_ids": sorted(diagnostic_ids),
    }


def run(*, repo_root: Path, source_commit: str) -> int:
    import torch

    started = time.time()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    contract = json.loads((repo_root / CONTRACT_RELATIVE).read_text(encoding="utf-8"))
    sources = contract["sources"]
    train_csv = _unique_input("ubc_train_shuffled.csv", sources["train_csv_sha256"])
    validation_csv = _unique_input(
        "ubc_ocean_valid.csv", sources["validation_csv_sha256"]
    )
    atlas_csv = _unique_input("train_val_atlas.csv", sources["atlas_csv_sha256"])
    train_binary = _unique_input("ubc_train_shuffled.bin")
    validation_binary = _unique_input("ubc_ocean_valid.bin")
    state_paths = {
        "normal_vae": _unique_input(
            "normal_vae_state.pt", sources["normal_state_file_sha256"]
        ),
        "so2_vae": _unique_input(
            "so2_vae_state.pt", sources["so2_state_file_sha256"]
        ),
    }
    train_rows = _load_patch_rows(train_csv, "vae_train", 300000)
    validation_rows = _load_patch_rows(validation_csv, "vae_validation", 30000)
    all_rows = train_rows + validation_rows
    binary_audit = {
        "vae_train": _validate_binary(
            train_binary,
            300000,
            contract["patch_sources"]["vae_train"]["binary_bytes"],
            contract["patch_sources"]["vae_train"]["binary_crc32"],
        ),
        "vae_validation": _validate_binary(
            validation_binary,
            30000,
            contract["patch_sources"]["vae_validation"]["binary_bytes"],
            contract["patch_sources"]["vae_validation"]["binary_crc32"],
        ),
    }
    _validate_atlas(atlas_csv, train_rows, validation_rows)
    cohort = _validate_cohort(repo_root, contract, all_rows)
    instance_manifest = _write_instance_manifest(all_rows)
    rows_by_wsi: dict[int, list[PatchRow]] = defaultdict(list)
    for row in all_rows:
        rows_by_wsi[row.wsi_id].append(row)
    for rows in rows_by_wsi.values():
        rows.sort(key=lambda row: (row.y, row.x, row.source_row_index))

    selected_ids = {
        int(record["wsi_id"]) for record in contract["sentinels"].values()
    } | {
        int(record["wsi_id"])
        for name, record in contract["cost_panel"].items()
        if name in {"median", "p99", "maximum"}
    }
    if torch.cuda.device_count() < 2:
        raise RuntimeError("A0 requires the matched two-GPU encoder pass")
    latent_bags, encoder_seconds = _encode_bags(
        selected_ids,
        all_rows,
        {"vae_train": train_binary, "vae_validation": validation_binary},
        state_paths,
        torch,
    )
    random.seed(CLASSIFIER_SEED)
    np.random.seed(CLASSIFIER_SEED)
    torch.manual_seed(CLASSIFIER_SEED)
    from eqvae.models.local_global_mil import LocalGlobalMILClassifier

    classifier_state = {
        name: value.detach().cpu().clone()
        for name, value in LocalGlobalMILClassifier().state_dict().items()
    }
    probe_fold = int(contract["probe_fold"])
    fold_counts = contract["fold_class_counts"][str(probe_fold)]
    train_counts = [
        contract["class_counts"][name] - fold_counts[name]
        for name in contract["class_order"]
    ]
    train_wsi_count = sum(train_counts)
    class_weights = [train_wsi_count / (5 * count) for count in train_counts]
    branches = {}
    for device_index, branch in enumerate(("normal_vae", "so2_vae")):
        branches[branch] = _run_branch_probe(
            branch,
            device_index,
            classifier_state,
            latent_bags[branch],
            rows_by_wsi,
            contract,
            class_weights,
            torch,
        )
    result = {
        "schema_version": "spec0054.a0_result.v1",
        "status": "complete",
        "source_commit": source_commit,
        "contract_sha256": _sha256(repo_root / CONTRACT_RELATIVE),
        "cohort_wsi_count": len(cohort),
        "patch_count": len(all_rows),
        "probe_fold": probe_fold,
        "probe_fold_train_class_counts": train_counts,
        "probe_fold_class_weights": class_weights,
        "binary_audit": binary_audit,
        "instance_manifest": instance_manifest,
        "encoder_seconds": encoder_seconds,
        "branches": branches,
        "all_equivalence_checks_passed": all(
            record["first_update_equivalence"]["passed"]
            for record in branches.values()
        ),
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        },
        "started_unix": started,
        "finished_unix": time.time(),
    }
    result["elapsed_seconds"] = result["finished_unix"] - started
    _write_json(OUTPUT_ROOT / "a0_result.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


__all__ = ["run"]
