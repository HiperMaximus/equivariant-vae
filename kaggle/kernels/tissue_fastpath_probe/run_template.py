# Copyright 2026 HiperMaximus
# ruff: noqa: ANN001, ANN201, ANN202, BLE001, C901, D103, E501, EM101, EM102, FBT003, INP001, PLC0415, PLR0913, PLR0914, PLR0917, PLR2004, S404, S603, SLF001, T201, TRY003
"""Run the train-only Spec 0037 tissue AMP/compile calibration probe."""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import os
import random
import statistics
import subprocess
import sys
import time
import traceback
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

SPEC0037_TISSUE_FASTPATH_PROBE_READY = True
INPUT_ROOT = Path("/kaggle/input")
OUTPUT_PATH = Path("/kaggle/working/spec0037_tissue_fastpath_probe.json")
INPUT_CONTRACT_NAME = "tissue_fastpath_probe_input.json"
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
LABELS = ("tumor", "stroma", "necrosis")
BRANCHES = (("normal_vae", 0), ("so2_vae", 1))
BATCH_SIZES = (125, 159)
CALIBRATION_BATCHES = 3
WARMUP_UPDATES = 2
TIMED_UPDATES = 10
MAX_OVERFLOW_BACKOFFS = 3
PEAK_LR = 1.5e-3
WEIGHT_DECAY = 1e-4
BETAS = (0.9, 0.999)
EPSILON = 1e-8
INITIALIZATION_SEED = 3407


class NumericalFailure(RuntimeError):
    """Attach retry evidence to a terminal update failure."""

    def __init__(self, message, attempts):
        super().__init__(message)
        self.attempts = attempts


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
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def resolve_input_bundle():
    matches = [
        path
        for path in INPUT_ROOT.rglob(INPUT_CONTRACT_NAME)
        if path.is_file() and sha256(path) == INPUT_CONTRACT_SHA256
    ]
    if len(matches) != 1:
        raise RuntimeError("Expected one exact Spec 0037 input contract")
    contract_path = matches[0]
    root = contract_path.parent
    contract = read_object(contract_path)
    if (
        contract.get("schema_version") != "spec0037.tissue_fastpath_probe_input.v1"
        or contract.get("dataset_reference") != INPUT_DATASET_REFERENCE
        or contract.get("scope") != "tissue_train_only_runtime_amp_probe_not_learning"
        or contract.get("visibility") != "private"
        or contract.get("kernel_sources") != list(KERNEL_SOURCES)
    ):
        raise RuntimeError("Spec 0037 mounted input identity differs")
    files = contract.get("files")
    if not isinstance(files, dict):
        raise RuntimeError("Spec 0037 input file manifest is missing")
    observed = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    if observed != {*files, INPUT_CONTRACT_NAME}:
        raise RuntimeError("Spec 0037 mounted input allow-list differs")
    if any(
        ("validation" in name or "test" in name) and not name.startswith("src/")
        for name in observed
    ):
        raise RuntimeError("Spec 0037 input exposes validation or sealed-test files")
    for name, record in files.items():
        if not isinstance(record, dict):
            raise RuntimeError("Spec 0037 file record is malformed")
        path = root / name
        if path.stat().st_size != record.get("bytes") or sha256(path) != record.get(
            "sha256",
        ):
            raise RuntimeError(f"Spec 0037 mounted file differs: {name}")
    audit = read_object(root / "train_manifest_audit.json")
    if audit != contract.get("train_manifest"):
        raise RuntimeError("Spec 0037 train-only manifest audit differs")
    batches = read_object(root / "probe_batches.json")
    if sha256(root / "probe_batches.json") != contract.get("probe_batches_sha256"):
        raise RuntimeError("Spec 0037 fixed batch selector differs")
    return root, contract, audit, batches


def resolve_source_roots(catalog_path, physical_sources):
    with catalog_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if rows != physical_sources:
        raise RuntimeError("Spec 0037 physical catalog differs from its contract")
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
            raise RuntimeError(
                f"Expected one sidecar-hash match for {row['model_name']} part {row['part']}",
            )
        binary = matches[0] / row["binary_name"]
        if not binary.is_file() or binary.stat().st_size != int(row["binary_bytes"]):
            raise RuntimeError("Spec 0037 mounted latent binary differs")
        source = row["kaggle_source"]
        if source in roots and roots[source] != matches[0]:
            raise RuntimeError("One Spec 0037 physical source resolved twice")
        roots[source] = matches[0]
    if tuple(roots) != KERNEL_SOURCES:
        raise RuntimeError("Spec 0037 physical source order differs")
    return roots


def configure_torch(torch):
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    functorch = getattr(torch, "_functorch", None)
    config = getattr(functorch, "config", None)
    if config is not None and hasattr(config, "backward_pass_autocast"):
        config.backward_pass_autocast = "off"
    return {
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "tf32": {"supported_on_t4": False, "enabled": False},
        "channels_last": True,
        "fused_adamw": True,
        "set_to_none": True,
    }


def make_optimizer(torch, model):
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=PEAK_LR,
        betas=BETAS,
        eps=EPSILON,
        fused=True,
        capturable=False,
    )


def make_scaler(torch):
    scaler = torch.amp.GradScaler("cuda")
    defaults = {
        "init_scale": scaler.get_scale(),
        "growth_factor": scaler.get_growth_factor(),
        "backoff_factor": scaler.get_backoff_factor(),
        "growth_interval": scaler.get_growth_interval(),
    }
    if defaults != {
        "init_scale": 65536.0,
        "growth_factor": 2.0,
        "backoff_factor": 0.5,
        "growth_interval": 2000,
    }:
        raise RuntimeError("Pinned PyTorch GradScaler defaults differ")
    return scaler


def make_numerical(torch, model, candidate):
    def numerical(latents, labels):
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(latents)
            loss = torch.nn.functional.cross_entropy(logits.float(), labels)
        return loss, logits

    if not candidate["compiled"]:
        return numerical
    kwargs = {
        "backend": "inductor",
        "fullgraph": True,
        "dynamic": False,
        "mode": candidate["mode"],
    }
    signature = inspect.signature(torch.compile)
    if "recompile_limit" in signature.parameters:
        kwargs["recompile_limit"] = 2
    if "isolate_recompiles" in signature.parameters:
        kwargs["isolate_recompiles"] = True
    return torch.compile(numerical, **kwargs)


def candidate_configs(torch):
    candidates = [
        {
            "name": "eager_fp16",
            "compiled": False,
            "mode": None,
            "compiled_autograd": False,
        },
        {
            "name": "compile_max_autotune_no_cudagraphs",
            "compiled": True,
            "mode": "max-autotune-no-cudagraphs",
            "compiled_autograd": False,
        },
        {
            "name": "compile_max_autotune",
            "compiled": True,
            "mode": "max-autotune",
            "compiled_autograd": False,
        },
    ]
    dynamo_config = getattr(getattr(torch, "_dynamo", None), "config", None)
    if dynamo_config is not None and hasattr(dynamo_config, "compiled_autograd"):
        candidates.extend(
            {
                "name": f"{candidate['name']}_compiled_autograd",
                "compiled": True,
                "mode": candidate["mode"],
                "compiled_autograd": True,
            }
            for candidate in candidates[1:]
        )
    return candidates


def model_state_sha256(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def capture_rng_state(torch):
    return {
        "python": random.getstate(),
        "cpu": torch.get_rng_state().clone(),
        "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()],
    }


def restore_rng_state(torch, state):
    random.setstate(state["python"])
    torch.set_rng_state(state["cpu"])
    torch.cuda.set_rng_state_all(state["cuda"])


def optimizer_step_value(optimizer):
    values = {
        int(state["step"].item())
        for state in optimizer.state.values()
        if "step" in state
    }
    if not values:
        return 0
    if len(values) != 1:
        raise RuntimeError("AdamW parameter step counters differ")
    return values.pop()


def tensor_state_finite(torch, model, optimizer):
    bad = []
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all().item():
            bad.append(f"parameter:{name}")
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    for parameter, state in optimizer.state.items():
        for key, value in state.items():
            if torch.is_tensor(value) and not torch.isfinite(value).all().item():
                bad.append(f"optimizer:{names[id(parameter)]}:{key}")
    return {"all_finite": not bad, "bad": bad}


def gradient_state(torch, model):
    missing, nonfinite, nonzero = [], [], 0
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            missing.append(name)
        elif not torch.isfinite(gradient).all().item():
            nonfinite.append(name)
        else:
            nonzero += int(torch.count_nonzero(gradient).item())
    return {
        "missing": missing,
        "nonfinite": nonfinite,
        "nonzero_elements": nonzero,
        "all_finite": not missing and not nonfinite,
    }


def run_update(
    torch,
    model,
    optimizer,
    scaler,
    numerical,
    batch,
    device,
    phase,
    ordinal,
):
    attempts = []
    for retry in range(MAX_OVERFLOW_BACKOFFS):
        optimizer.zero_grad(set_to_none=True)
        before_step = optimizer_step_value(optimizer)
        before_state = model_state_sha256(model)
        scale_before = float(scaler.get_scale())
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        loss, logits = numerical(batch["latents"], batch["labels"])
        if (
            not torch.isfinite(loss).all().item()
            or not torch.isfinite(logits).all().item()
        ):
            raise NumericalFailure("Nonfinite FP32 loss or logits", attempts)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradients = gradient_state(torch, model)
        if gradients["missing"]:
            raise NumericalFailure("Missing model gradient", attempts)
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        committed = optimizer_step_value(optimizer) == before_step + 1
        post = tensor_state_finite(torch, model, optimizer)
        row = {
            "phase": phase,
            "ordinal": ordinal,
            "retry": retry,
            "scale_before": scale_before,
            "scale_after": float(scaler.get_scale()),
            "loss": float(loss.detach().float().cpu()),
            "logits_finite": True,
            "gradients": gradients,
            "committed": committed,
            "elapsed_ms": elapsed_ms,
            "post_state": post,
        }
        attempts.append(row)
        if committed:
            if not gradients["all_finite"] or not post["all_finite"]:
                raise NumericalFailure("A committed update was nonfinite", attempts)
            return attempts
        if not gradients["nonfinite"]:
            raise NumericalFailure("Scaler skipped a finite-gradient update", attempts)
        if model_state_sha256(model) != before_state:
            raise NumericalFailure(
                "Overflowed scaler step changed model state",
                attempts,
            )
    raise NumericalFailure("Three consecutive scaler backoffs", attempts)


def graph_telemetry(torch, candidate):
    if not candidate["compiled"]:
        return {
            "compiled": False,
            "graph_breaks": None,
            "unique_graphs": None,
            "generated_kernels": None,
        }
    counters = torch._dynamo.utils.counters
    graph_breaks = sum(int(value) for value in counters.get("graph_break", {}).values())
    return {
        "compiled": True,
        "graph_breaks": graph_breaks,
        "unique_graphs": int(counters.get("stats", {}).get("unique_graphs", 0)),
        "generated_kernels": int(
            getattr(torch._inductor.metrics, "generated_kernel_count", 0),
        ),
    }


def load_batches(torch, dataset, indices, device):
    batches = []
    for row in indices:
        loaded = dataset.read_batch(row)
        latents = loaded.latents.to(
            device,
            non_blocking=False,
            memory_format=torch.channels_last,
        )
        labels = loaded.labels.to(device, non_blocking=False)
        batches.append({"latents": latents, "labels": labels})
    return batches


def run_branch(torch, root, source_roots, batch_indices, branch, device, candidate):
    from eqvae.data.supervised_latents import SupervisedLatentStore, TissueDataset
    from eqvae.models.supervised import TissueClassifier
    from eqvae.training.supervised_pairing import make_paired_models

    with torch.cuda.device(device):
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        torch._inductor.metrics.reset()
        normal, so2 = make_paired_models(TissueClassifier, seed=INITIALIZATION_SEED)
        model = (
            (normal if branch == "normal_vae" else so2)
            .to(
                device=device,
                memory_format=torch.channels_last,
            )
            .train()
        )
        base_state = deepcopy(model.state_dict())
        base_hash = model_state_sha256(model)
        numerical = make_numerical(torch, model, candidate)
        scaler = make_scaler(torch)
        torch.manual_seed(INITIALIZATION_SEED)
        torch.cuda.manual_seed_all(INITIALIZATION_SEED)
        base_rng = capture_rng_state(torch)
        calibration = []
        timed = []
        try:
            with SupervisedLatentStore(
                catalog_path=root / "physical_parts.csv",
                model_name=branch,
                source_roots=source_roots,
            ) as store:
                dataset = TissueDataset(
                    path=root / "tissue/tissue_train_5671_per_class.csv",
                    split="train",
                    store=store,
                )
                batches = load_batches(torch, dataset, batch_indices, device)
                for ordinal, batch in enumerate(batches, start=1):
                    model.load_state_dict(base_state)
                    for parameter in model.parameters():
                        parameter.grad = None
                    restore_rng_state(torch, base_rng)
                    optimizer = make_optimizer(torch, model)
                    attempts = run_update(
                        torch,
                        model,
                        optimizer,
                        scaler,
                        numerical,
                        batch,
                        device,
                        "calibration",
                        ordinal,
                    )
                    calibration.append(attempts)
                model.load_state_dict(base_state)
                for parameter in model.parameters():
                    parameter.grad = None
                restore_rng_state(torch, base_rng)
                if model_state_sha256(model) != base_hash:
                    raise RuntimeError("Model did not restore after calibration")
                optimizer = make_optimizer(torch, model)
                if optimizer.state:
                    raise RuntimeError("Training optimizer inherited calibration state")
                for ordinal in range(1, WARMUP_UPDATES + TIMED_UPDATES + 1):
                    attempts = run_update(
                        torch,
                        model,
                        optimizer,
                        scaler,
                        numerical,
                        batches[(ordinal - 1) % len(batches)],
                        device,
                        "warmup" if ordinal <= WARMUP_UPDATES else "timed",
                        ordinal,
                    )
                    if ordinal > WARMUP_UPDATES:
                        timed.append(attempts[-1])
        except NumericalFailure as error:
            return {
                "status": "rejected",
                "branch": branch,
                "device": device,
                "error": str(error),
                "attempts": error.attempts,
            }
        except Exception as error:
            return {
                "status": "rejected",
                "branch": branch,
                "device": device,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        telemetry = graph_telemetry(torch, candidate)
        accepted = (
            len(calibration) == CALIBRATION_BATCHES
            and len(timed) == TIMED_UPDATES
            and all(attempts[-1]["committed"] for attempts in calibration)
            and all(
                row["committed"] and row["post_state"]["all_finite"] for row in timed
            )
            and (
                not candidate["compiled"]
                or (
                    telemetry["graph_breaks"] == 0
                    and telemetry["unique_graphs"] >= 1
                    and telemetry["generated_kernels"] > 0
                )
            )
        )
        return {
            "status": "accepted" if accepted else "rejected",
            "branch": branch,
            "device": device,
            "base_state_sha256": base_hash,
            "scaler_after_calibration": float(scaler.get_scale()),
            "calibration": calibration,
            "timed_updates": timed,
            "settled_median_ms": statistics.median(row["elapsed_ms"] for row in timed),
            "telemetry": telemetry,
        }


def run_candidate(torch, root, contract, source_roots, batches, candidate):
    dynamo_config = torch._dynamo.config
    previous = getattr(dynamo_config, "compiled_autograd", None)
    if previous is not None:
        dynamo_config.compiled_autograd = candidate["compiled_autograd"]
    try:
        branches = [
            run_branch(torch, root, source_roots, batches, name, device, candidate)
            for name, device in BRANCHES
        ]
    finally:
        if previous is not None:
            dynamo_config.compiled_autograd = previous
    paired_initialization = (
        all(row["status"] == "accepted" for row in branches)
        and len({row["base_state_sha256"] for row in branches}) == 1
    )
    accepted = paired_initialization
    slowest = max(
        (row.get("settled_median_ms", float("inf")) for row in branches),
        default=float("inf"),
    )
    return {
        "candidate": candidate,
        "status": "accepted" if accepted else "rejected",
        "branches": branches,
        "paired_initialization": paired_initialization,
        "slower_branch_settled_median_ms": slowest,
    }


def choose_winner(results):
    per_shape = {}
    for batch_size, rows in results.items():
        accepted = sorted(
            (row for row in rows if row["status"] == "accepted"),
            key=lambda row: row["slower_branch_settled_median_ms"],
        )
        per_shape[batch_size] = [row["candidate"]["name"] for row in accepted]
    shared = set(per_shape["125"]) & set(per_shape["159"])
    if not shared:
        return {"per_shape_ranking": per_shape, "shared_winner": None}
    scores = {}
    for name in shared:
        score = 0.0
        for size in ("125", "159"):
            rows = {row["candidate"]["name"]: row for row in results[size]}
            eager = rows["eager_fp16"]["slower_branch_settled_median_ms"]
            score += rows[name]["slower_branch_settled_median_ms"] / eager
        scores[name] = score
    return {
        "per_shape_ranking": per_shape,
        "shared_winner": min(scores, key=scores.get),
        "normalized_score": scores,
    }


def run_probe(torch, root, contract, batches, result):
    if torch.cuda.device_count() != 2 or any(
        "T4" not in torch.cuda.get_device_name(index) for index in range(2)
    ):
        raise RuntimeError("Spec 0037 requires exactly two Tesla T4 GPUs")
    result["runtime"] = {
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "devices": [torch.cuda.get_device_name(index) for index in range(2)],
        "flags": configure_torch(torch),
        "grad_scaler_defaults": {
            "init_scale": 65536.0,
            "growth_factor": 2.0,
            "backoff_factor": 0.5,
            "growth_interval": 2000,
        },
    }
    source_roots = resolve_source_roots(
        root / "physical_parts.csv",
        contract["physical_sources"],
    )
    candidates = candidate_configs(torch)
    all_results = {}
    for batch_size in BATCH_SIZES:
        selected = batches["batch_sizes"][str(batch_size)]
        if len(selected) != CALIBRATION_BATCHES or any(
            len(row) != batch_size for row in selected
        ):
            raise RuntimeError("Spec 0037 fixed batch geometry differs")
        rows = []
        for candidate in candidates:
            result["phase"] = f"batch_{batch_size}_{candidate['name']}"
            row = run_candidate(
                torch,
                root,
                contract,
                source_roots,
                selected,
                candidate,
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
        all_results[str(batch_size)] = rows
    result["candidates"] = all_results
    result["selection"] = choose_winner(all_results)
    result["status"] = (
        "complete" if result["selection"]["shared_winner"] else "rejected"
    )


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
        raise RuntimeError(
            f"Pinned runtime differs: torch={torch.__version__}, cuda={torch.version.cuda}",
        )


def execute(result):
    root, contract, _audit, batches = resolve_input_bundle()
    result["phase"] = "install_pinned_torch"
    install_pinned_torch()
    import torch

    validate_pinned_torch(torch)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root / "src"))
    result["phase"] = "probe"
    run_probe(torch, root, contract, batches, result)


def main():
    result = {
        "status": "failed",
        "spec": "0037",
        "scope": "tissue_train_only_runtime_amp_probe_not_learning",
        "phase": "resolve_input",
        "input_dataset": f"{INPUT_DATASET_REFERENCE}/1",
        "kernel_sources": [f"{source}/1" for source in KERNEL_SOURCES],
        "input_contract_sha256": INPUT_CONTRACT_SHA256,
        "started_unix": time.time(),
    }
    try:
        execute(result)
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
