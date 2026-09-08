# Copyright 2026 HiperMaximus
# ruff: noqa: ANN001, ANN201, ANN202, BLE001, C901, D103, EM101, EM102, FBT003, INP001, NPY002, PLC0415, PLR0912, PLR0913, PLR0914, PLR0915, PLR0916, PLR0917, PLR2004, S105, S311, S404, S603, SLF001, T201, TRY003
"""Run the exact Spec 0035 largest-per-class weighted AMP probe."""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from collections import Counter
from copy import deepcopy
from itertools import starmap
from operator import itemgetter
from pathlib import Path

import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

SPEC0035_LARGEST_CLASS_WEIGHTED_AMP_READY = True
INPUT_ROOT = Path("/kaggle/input")
OUTPUT_PATH = Path("/kaggle/working/spec0035_largest_class_weighted_amp_probe.json")
INPUT_CONTRACT_NAME = "largest_class_weighted_amp_input.json"
INPUT_CONTRACT_SHA256 = "$input_contract_sha256"
INPUT_DATASET_REFERENCE = "$input_dataset_reference"
PARAMETER_COUNT = 1_513_055
INITIAL_SCALE = 65_536.0
GROWTH_FACTOR = 2.0
BACKOFF_FACTOR = 0.5
GROWTH_INTERVAL = 2_000
MAX_OVERFLOW_BACKOFFS = 3
CALIBRATION_ORDER = (45_630, 51_346, 57_162, 65_094, 35_239)
TRAINING_SHUFFLE_SEED = 3_501
TRAINING_SIMULATION_ORDER = (57_162, 45_630, 65_094, 35_239, 51_346)
MIN_RESERVED_HEADROOM = 512 * 1024 * 1024
PINNED_TORCH_VERSION = "2.14.0"
PINNED_TORCH_CUDA = "13.0"
PINNED_TORCH_INDEX = "https://download.pytorch.org/whl/cu130"
KERNEL_SOURCES = (
    "maximusshtefan/eqvae-ubc-ocean-latent-run-03",
    "maximusshtefan/eqvae-ubc-ocean-latent-run-04",
    "maximusshtefan/eqvae-ubc-ocean-latent-run-05",
    "maximusshtefan/eqvae-ubc-ocean-cancer-latent-top-up",
    "maximusshtefan/eqvae-wsi45630-completion",
    "maximusshtefan/eqvae-full-foreground-05",
    "maximusshtefan/eqvae-full-foreground-07",
    "maximusshtefan/eqvae-full-foreground-08",
)


class WeightedUpdateError(RuntimeError):
    """Preserve structured attempt diagnostics for one terminal case."""

    def __init__(self, message, attempts) -> None:
        """Attach every attempt made before the terminal failure."""
        super().__init__(message)
        self.attempts = attempts


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_input_bundle():
    matches = list(INPUT_ROOT.rglob(INPUT_CONTRACT_NAME))
    if len(matches) != 1 or sha256(matches[0]) != INPUT_CONTRACT_SHA256:
        raise RuntimeError("Expected the exact Spec 0035 input contract")
    root = matches[0].parent
    contract = json.loads(matches[0].read_text(encoding="utf-8"))
    if (
        contract.get("schema_version") != "spec0035.largest_class_weighted_amp_input.v1"
        or contract.get("dataset_reference") != INPUT_DATASET_REFERENCE
        or contract.get("scope") != "train_only_numerical_probe_not_learning"
        or contract.get("parameter_count") != PARAMETER_COUNT
        or contract.get("kernel_sources") != list(KERNEL_SOURCES)
        or len(contract.get("selected", ())) != 5
    ):
        raise RuntimeError("Spec 0035 mounted input scope differs")
    for name, record in contract["files"].items():
        path = root / name
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise RuntimeError(f"Spec 0035 mounted input bytes differ: {name}")
    observed = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    if observed != {*contract["files"], INPUT_CONTRACT_NAME}:
        raise RuntimeError("Unexpected file in Spec 0035 input dataset")
    return root, contract


def resolve_sources(catalog):
    roots = {}
    with catalog.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            matches = [
                path.parent
                for path in INPUT_ROOT.rglob(row["sidecar_name"])
                if path.is_file()
                and path.stat().st_size == int(row["sidecar_bytes"])
                and sha256(path) == row["sidecar_sha256"]
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"Expected one hash-matched sidecar for part {row['part']} "
                    f"and {row['model_name']}",
                )
            source = row["kaggle_source"]
            if source in roots and roots[source] != matches[0]:
                raise RuntimeError("Paired producer files have different roots")
            roots[source] = matches[0]
    return roots


def load_pointer_cases(root, contract):
    selected = {int(row["wsi_id"]): row for row in contract["selected"]}
    cases = {wsi_id: [] for wsi_id in selected}
    with (root / "probe/pointers.csv").open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row = {
                key: (value if key == "diagnosis_label" else int(value))
                for key, value in raw.items()
            }
            wsi_id = row["wsi_id"]
            if wsi_id not in cases:
                raise RuntimeError("Foreign WSI in Spec 0035 pointers")
            cases[wsi_id].append(row)
    for wsi_id, rows in cases.items():
        expected = selected[wsi_id]
        if (
            len(rows) != expected["patch_count"]
            or {row["diagnosis_label"] for row in rows} != {expected["diagnosis_label"]}
            or {row["diagnosis_index"] for row in rows} != {expected["diagnosis_index"]}
            or len({(row["x"], row["y"]) for row in rows}) != len(rows)
            or rows != sorted(rows, key=itemgetter("y", "x"))
            or dict(Counter(str(row["part"]) for row in rows))
            != expected["part_counts"]
        ):
            raise RuntimeError(f"Spec 0035 pointer identity differs for WSI {wsi_id}")
    return cases


def configure_torch(torch):
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)
    torch.set_float32_matmul_precision("high")
    torch._functorch.config.backward_pass_autocast = "off"
    if hasattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction"):
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True


def make_model(torch, base_state, device):
    from eqvae.models.local_attention_candidates import (
        WholeBagFixed25Attention,
        use_whole_bag_fixed25_attention,
    )
    from eqvae.models.local_global_mil import LocalGlobalMILClassifier

    model = LocalGlobalMILClassifier()
    model.load_state_dict(base_state)
    before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    use_whole_bag_fixed25_attention(model)
    if before.keys() != model.state_dict().keys() or any(
        not torch.equal(value, model.state_dict()[name])
        for name, value in before.items()
    ):
        raise RuntimeError("Fixed-25 replacement changed model state")
    if not all(
        isinstance(block.attention, WholeBagFixed25Attention)
        for block in model.local_blocks
    ):
        raise RuntimeError("Both local blocks must use whole-bag fixed-25 attention")
    model = model.to(device=device, memory_format=torch.channels_last).train()
    if sum(parameter.numel() for parameter in model.parameters()) != PARAMETER_COUNT:
        raise RuntimeError("Model parameter count differs")
    return model


def make_optimizer(torch, model):
    from eqvae.models.local_global_mil import local_global_mil_adamw_parameter_groups

    return torch.optim.AdamW(
        local_global_mil_adamw_parameter_groups(model, weight_decay=1e-4),
        lr=2e-4,
        fused=True,
        capturable=True,
    )


def make_scaler(torch):
    scaler = torch.amp.GradScaler("cuda")
    if (
        scaler.get_scale() != INITIAL_SCALE
        or scaler.get_growth_factor() != GROWTH_FACTOR
        or scaler.get_backoff_factor() != BACKOFF_FACTOR
        or scaler.get_growth_interval() != GROWTH_INTERVAL
    ):
        raise RuntimeError("Pinned PyTorch GradScaler defaults differ")
    return scaler


def make_numerical(torch, model):
    def numerical(latents, graph, target):
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(latents, graph)
            loss = torch.nn.functional.cross_entropy(
                logits.float().unsqueeze(0),
                target,
            )
        return loss, logits

    kwargs = {
        "backend": "inductor",
        "fullgraph": True,
        "dynamic": None,
        "mode": "max-autotune-no-cudagraphs",
    }
    signature = inspect.signature(torch.compile)
    if "recompile_limit" in signature.parameters:
        kwargs["recompile_limit"] = 3
    if "isolate_recompiles" in signature.parameters:
        kwargs["isolate_recompiles"] = True
    return torch.compile(numerical, **kwargs)


def hint_dynamic(torch, bag, graph):
    torch._dynamo.maybe_mark_dynamic(bag, 0)
    for tensor in (graph.neighbor_index, graph.neighbor_valid, graph.radial_code):
        torch._dynamo.maybe_mark_dynamic(tensor, 0)


def gradient_summary(torch, model):
    count = 0
    nonzero_elements = 0
    nonzero_tensors = 0
    squared_norm = 0.0
    max_abs = 0.0
    bad = []
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            bad.append({"parameter": name, "reason": "missing"})
            continue
        count += 1
        if gradient.dtype != torch.float32:
            bad.append({
                "parameter": name,
                "reason": "wrong_dtype",
                "dtype": str(gradient.dtype),
            })
        finite = torch.isfinite(gradient)
        if not finite.all().item():
            bad.append({
                "parameter": name,
                "reason": "nonfinite",
                "nonfinite": int((~finite).sum().item()),
            })
        finite_values = gradient.detach().float()[finite]
        if finite_values.numel():
            tensor_nonzero = int(torch.count_nonzero(finite_values).item())
            nonzero_elements += tensor_nonzero
            nonzero_tensors += int(tensor_nonzero > 0)
            squared_norm += float(torch.sum(finite_values.double().square()).item())
            max_abs = max(max_abs, float(finite_values.abs().max().item()))
    return {
        "parameter_count": count,
        "nonzero_element_count": nonzero_elements,
        "nonzero_tensor_count": nonzero_tensors,
        "all_finite": not bad,
        "l2_norm": math.sqrt(squared_norm),
        "max_abs": max_abs,
        "bad": bad,
    }


def training_state_summary(torch, model, optimizer):
    bad = []
    parameter_count = 0
    state_tensor_count = 0
    for name, parameter in model.named_parameters():
        parameter_count += 1
        if not torch.isfinite(parameter).all().item():
            bad.append({"tensor": f"parameter:{name}", "reason": "nonfinite"})
    parameter_names = {
        id(parameter): name for name, parameter in model.named_parameters()
    }
    for parameter, state in optimizer.state.items():
        name = parameter_names.get(id(parameter), "unknown")
        for key, value in state.items():
            if not torch.is_tensor(value):
                continue
            state_tensor_count += 1
            if not torch.isfinite(value).all().item():
                bad.append({
                    "tensor": f"optimizer:{name}:{key}",
                    "reason": "nonfinite",
                })
    return {
        "parameter_count": parameter_count,
        "optimizer_state_tensor_count": state_tensor_count,
        "all_finite": not bad,
        "bad": bad,
    }


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


def model_state_sha256(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def model_matches_state(torch, model, expected):
    observed = model.state_dict()
    return observed.keys() == expected.keys() and all(
        torch.equal(observed[name].detach().cpu(), value.detach().cpu())
        for name, value in expected.items()
    )


def restore_model_state(torch, model, expected, expected_sha256):
    model.load_state_dict(expected)
    for parameter in model.parameters():
        parameter.grad = None
    observed_sha256 = model_state_sha256(model)
    if (
        not model_matches_state(torch, model, expected)
        or observed_sha256 != expected_sha256
        or any(parameter.grad is not None for parameter in model.parameters())
    ):
        raise RuntimeError("Model did not restore exactly to calibration base state")
    return observed_sha256


def capture_rng_state(torch):
    numpy_state = np.random.get_state()
    return {
        "python": deepcopy(random.getstate()),
        "numpy": (
            numpy_state[0],
            numpy_state[1].copy(),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        ),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda_all": [state.clone() for state in torch.cuda.get_rng_state_all()],
    }


def restore_rng_state(torch, state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda_all"])


def rng_state_sha256(state):
    digest = hashlib.sha256()
    digest.update(repr(state["python"]).encode("ascii"))
    numpy_state = state["numpy"]
    digest.update(numpy_state[0].encode("ascii"))
    digest.update(numpy_state[1].tobytes())
    digest.update(repr(numpy_state[2:]).encode("ascii"))
    for name in ("torch_cpu",):
        digest.update(name.encode("ascii"))
        digest.update(state[name].detach().cpu().numpy().tobytes())
    for index, value in enumerate(state["torch_cuda_all"]):
        digest.update(f"torch_cuda_{index}".encode("ascii"))
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def rng_state_matches(torch, observed, expected):
    observed_numpy = observed["numpy"]
    expected_numpy = expected["numpy"]
    return (
        observed["python"] == expected["python"]
        and observed_numpy[0] == expected_numpy[0]
        and np.array_equal(observed_numpy[1], expected_numpy[1])
        and observed_numpy[2:] == expected_numpy[2:]
        and torch.equal(observed["torch_cpu"], expected["torch_cpu"])
        and len(observed["torch_cuda_all"]) == len(expected["torch_cuda_all"])
        and all(
            starmap(
                torch.equal,
                zip(
                    observed["torch_cuda_all"],
                    expected["torch_cuda_all"],
                    strict=True,
                ),
            ),
        )
    )


def scaler_state_record(scaler):
    state = scaler.state_dict()
    encoded = json.dumps(state, sort_keys=True, separators=(",", ":"))
    return {
        "state": state,
        "sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
    }


def run_weighted_update(torch, model, optimizer, scaler, numerical, case, branch):
    attempts = []
    retries = 0
    while True:
        optimizer.zero_grad(set_to_none=True)
        before_state = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        scale_before = float(scaler.get_scale())
        started = time.perf_counter()
        loss, logits = numerical(case["bag"], case["graph"], case["target"])
        if not torch.isfinite(loss).item() or not torch.isfinite(logits).all().item():
            raise FloatingPointError("Nonfinite unweighted loss or logits")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        raw = gradient_summary(torch, model)
        terminal_gradient_errors = [
            row for row in raw["bad"] if row["reason"] != "nonfinite"
        ]
        if terminal_gradient_errors:
            attempts.append({
                "committed": False,
                "retryable_overflow": False,
                "scale_before": scale_before,
                "class_weight": case["class_weight"],
                "raw_gradients": raw,
                "terminal_error": "missing_or_wrong_dtype_gradient",
            })
            raise WeightedUpdateError(
                f"Terminal gradient contract failure: branch={branch}, "
                f"WSI={case['wsi_id']}, diagnosis={case['diagnosis_label']}, "
                f"scale={scale_before}, bad={terminal_gradient_errors}",
                attempts,
            )
        if not raw["all_finite"]:
            step_before = optimizer_step_value(optimizer)
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            if (
                optimizer_step_value(optimizer) != step_before
                or scale_after >= scale_before
                or any(
                    not torch.equal(value, dict(model.named_parameters())[name])
                    for name, value in before_state.items()
                )
            ):
                raise RuntimeError("GradScaler overflow did not skip atomically")
            attempts.append({
                "committed": False,
                "retryable_overflow": True,
                "scale_before": scale_before,
                "scale_after": scale_after,
                "class_weight": case["class_weight"],
                "raw_gradients": raw,
                "elapsed_ms": (time.perf_counter() - started) * 1000,
            })
            retries += 1
            if retries >= MAX_OVERFLOW_BACKOFFS:
                raise WeightedUpdateError(
                    f"Overflow backoff budget exhausted: branch={branch}, "
                    f"WSI={case['wsi_id']}, diagnosis={case['diagnosis_label']}, "
                    f"scale={scale_before}, bad={raw['bad']}",
                    attempts,
                )
            continue
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.grad.mul_(case["class_weight"])
        weighted = gradient_summary(torch, model)
        if not weighted["all_finite"]:
            attempts.append({
                "committed": False,
                "retryable_overflow": False,
                "scale_before": scale_before,
                "class_weight": case["class_weight"],
                "raw_gradients": raw,
                "weighted_gradients": weighted,
                "terminal_error": "post_weight_nonfinite_gradient",
            })
            raise WeightedUpdateError(
                f"Post-weight gradient failure: branch={branch}, "
                f"WSI={case['wsi_id']}, diagnosis={case['diagnosis_label']}, "
                f"scale={scale_before}, bad={weighted['bad']}",
                attempts,
            )
        expected_weighted_norm = raw["l2_norm"] * case["class_weight"]
        if (
            raw["nonzero_element_count"] == 0
            or weighted["nonzero_element_count"] == 0
            or not math.isclose(
                weighted["l2_norm"],
                expected_weighted_norm,
                rel_tol=1e-6,
                abs_tol=1e-12,
            )
        ):
            attempts.append({
                "committed": False,
                "retryable_overflow": False,
                "scale_before": scale_before,
                "class_weight": case["class_weight"],
                "raw_gradients": raw,
                "weighted_gradients": weighted,
                "expected_weighted_l2_norm": expected_weighted_norm,
                "terminal_error": "zero_or_inexact_weighted_gradient",
            })
            raise WeightedUpdateError(
                f"Weighted-gradient signal failure: branch={branch}, "
                f"WSI={case['wsi_id']}, diagnosis={case['diagnosis_label']}, "
                f"scale={scale_before}, raw_l2={raw['l2_norm']}, "
                f"weighted_l2={weighted['l2_norm']}, "
                f"expected_l2={expected_weighted_norm}",
                attempts,
            )
        total_norm = torch.nn.utils.get_total_norm(
            (parameter.grad for parameter in model.parameters()),
            error_if_nonfinite=True,
            foreach=True,
        )
        step_before = optimizer_step_value(optimizer)
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize(case["device"])
        scale_after = float(scaler.get_scale())
        step_after = optimizer_step_value(optimizer)
        if step_after != step_before + 1 or scale_after != scale_before:
            raise RuntimeError("Finite weighted update did not commit exactly once")
        changed = sum(
            not torch.equal(value, dict(model.named_parameters())[name])
            for name, value in before_state.items()
        )
        if not changed:
            raise RuntimeError("Finite weighted update changed no parameter")
        post_step = training_state_summary(torch, model, optimizer)
        if not post_step["all_finite"]:
            attempts.append({
                "committed": False,
                "optimizer_step_applied": True,
                "retryable_overflow": False,
                "scale_before": scale_before,
                "scale_after": scale_after,
                "class_weight": case["class_weight"],
                "raw_gradients": raw,
                "weighted_gradients": weighted,
                "post_step_state": post_step,
                "terminal_error": "post_step_nonfinite_state",
            })
            raise WeightedUpdateError(
                f"Post-step state failure: branch={branch}, "
                f"WSI={case['wsi_id']}, diagnosis={case['diagnosis_label']}, "
                f"scale={scale_before}, bad={post_step['bad']}",
                attempts,
            )
        attempts.append({
            "committed": True,
            "retryable_overflow": False,
            "scale_before": scale_before,
            "scale_after": scale_after,
            "class_weight": case["class_weight"],
            "unweighted_loss": float(loss.detach().item()),
            "weighted_loss": float(loss.detach().item()) * case["class_weight"],
            "logits": [float(value) for value in logits.detach().float().cpu()],
            "raw_gradients": raw,
            "weighted_gradients": weighted,
            "weighted_total_norm": float(total_norm.detach().item()),
            "post_step_state": post_step,
            "changed_parameter_count": changed,
            "elapsed_ms": (time.perf_counter() - started) * 1000,
        })
        return attempts


def load_case(torch, store, rows, selected, device):
    from eqvae.data.supervised_latents import LogicalPointer, WSIInstance
    from eqvae.models.local_global_mil import build_local_attention_graph

    pointers = tuple(LogicalPointer(row["part"], row["file_index"]) for row in rows)
    instances = tuple(
        WSIInstance(
            instance_row=index,
            atlas_row_index=row["atlas_row_index"],
            wsi_id=row["wsi_id"],
            x=row["x"],
            y=row["y"],
            diagnosis_label=row["diagnosis_label"],
            diagnosis_index=row["diagnosis_index"],
            split="train",
            pointer=pointers[index],
        )
        for index, row in enumerate(rows)
    )
    latents, reads = store.read_rows(pointers)
    if (
        tuple(latents.shape) != (selected["patch_count"], 16, 32, 32)
        or dict(Counter(str(read.part) for read in reads)) != selected["part_counts"]
        or not torch.isfinite(latents).all().item()
    ):
        raise RuntimeError(f"Latent identity differs for WSI {selected['wsi_id']}")
    graph = build_local_attention_graph(
        instances,
        expected_instance_count=selected["patch_count"],
    ).to(device)
    bag = latents.to(
        device=device,
        dtype=torch.float16,
        memory_format=torch.channels_last,
    )
    del latents
    hint_dynamic(torch, bag, graph)
    return {
        **selected,
        "device": device,
        "bag": bag,
        "graph": graph,
        "target": torch.tensor([selected["diagnosis_index"]], device=device),
        "graph_sha256": graph.identity_sha256,
        "edges": int(graph.neighbor_valid.sum().item()),
    }


def case_record(row, case, attempts, failure=None, phase_evidence=None):
    record = {
        "wsi_id": row["wsi_id"],
        "diagnosis_label": row["diagnosis_label"],
        "diagnosis_index": row["diagnosis_index"],
        "patch_count": row["patch_count"],
        "class_weight": row["class_weight"],
        "graph_sha256": case["graph_sha256"],
        "edges": case["edges"],
        "attempts": attempts,
    }
    if failure is not None:
        record["failure"] = failure
    if phase_evidence is not None:
        record["phase_evidence"] = phase_evidence
    return record


def transcript_identity(rows):
    return [
        (
            row["wsi_id"],
            row["diagnosis_label"],
            row["diagnosis_index"],
            row["patch_count"],
            row["class_weight"],
            row["graph_sha256"],
            row["edges"],
        )
        for row in rows
    ]


def transcript_case_set_matches(left, right):
    return sorted(transcript_identity(left)) == sorted(transcript_identity(right))


def boundary_accepts(boundary):
    return (
        boundary is not None
        and boundary["model_matches_base_after_calibration"]
        and boundary["rng_matches_base_after_calibration"]
        and boundary["model_sha256_after_restore"] == boundary["base_model_sha256"]
        and boundary["rng_sha256_after_restore"] == boundary["base_rng_sha256"]
        and boundary["training_optimizer_state_entries_before"] == 0
        and boundary["training_optimizer_step_before"] == 0
        and boundary["model_gradients_cleared_after_calibration"]
        and boundary["scaler_state_preserved_across_boundary"]
        and boundary["single_scaler_reused"]
        and boundary["calibrated_scaler_before_training"]["sha256"]
        == boundary["scaler_after_training_optimizer_creation"]["sha256"]
    )


def scaler_chain_accepts(calibration, training_simulation, boundary):
    if boundary is None:
        return False
    scaler_object_id = boundary["scaler_object_id"]
    expected_sha256 = boundary["initial_scaler"]["sha256"]
    for row in calibration + training_simulation:
        evidence = row.get("phase_evidence", {})
        if (
            evidence.get("scaler_object_id") != scaler_object_id
            or evidence.get("scaler_before", {}).get("sha256") != expected_sha256
        ):
            return False
        expected_sha256 = evidence.get("scaler_after", {}).get("sha256")
    return expected_sha256 == boundary["final_scaler"]["sha256"]


def branch_accepts(
    calibration,
    training_simulation,
    terminal_failure,
    training_step,
    training_scale,
    unique_graphs,
    graph_breaks,
    generated_kernel_count,
    headroom,
    boundary=None,
    calibration_unique_graphs=None,
):
    expected_labels = {"CC", "EC", "HGSC", "LGSC", "MC"}
    calibration_fresh = all(
        row.get("phase_evidence", {}).get("phase") == "calibration"
        and row.get("phase_evidence", {}).get("disposable_optimizer_ordinal") == ordinal
        and row.get("phase_evidence", {}).get("model_matches_base_before", False)
        and row.get("phase_evidence", {}).get("rng_matches_base_before", False)
        and row.get("phase_evidence", {}).get("optimizer_state_entries_before", -1) == 0
        and row.get("phase_evidence", {}).get("optimizer_step_before", -1) == 0
        and row.get("phase_evidence", {}).get("optimizer_step_after", -1) == 1
        for ordinal, row in enumerate(calibration, start=1)
    )
    training_step_chain = all(
        row.get("phase_evidence", {}).get("phase") == "training_simulation"
        and row.get("phase_evidence", {}).get("training_step_ordinal") == ordinal
        and row.get("phase_evidence", {}).get("optimizer_step_before") == ordinal - 1
        and row.get("phase_evidence", {}).get("optimizer_step_after") == ordinal
        for ordinal, row in enumerate(training_simulation, start=1)
    )
    phase_contract = (
        [row["wsi_id"] for row in calibration] == list(CALIBRATION_ORDER)
        and [row["wsi_id"] for row in training_simulation]
        == list(TRAINING_SIMULATION_ORDER)
        and calibration_fresh
        and training_step_chain
        and boundary_accepts(boundary)
        and scaler_chain_accepts(calibration, training_simulation, boundary)
        and calibration_unique_graphs is not None
        and calibration_unique_graphs == unique_graphs
    )
    return (
        terminal_failure is None
        and {row["diagnosis_label"] for row in calibration} == expected_labels
        and {row["diagnosis_label"] for row in training_simulation} == expected_labels
        and transcript_case_set_matches(calibration, training_simulation)
        and phase_contract
        and all(
            sum(attempt["committed"] for attempt in row["attempts"]) == 1
            and row["attempts"][-1]["raw_gradients"]["all_finite"]
            and row["attempts"][-1]["weighted_gradients"]["all_finite"]
            and row["attempts"][-1]["raw_gradients"]["nonzero_element_count"] > 0
            and row["attempts"][-1]["weighted_gradients"]["nonzero_element_count"] > 0
            and math.isclose(
                row["attempts"][-1]["weighted_gradients"]["l2_norm"],
                row["attempts"][-1]["raw_gradients"]["l2_norm"] * row["class_weight"],
                rel_tol=1e-6,
                abs_tol=1e-12,
            )
            and row["attempts"][-1]["post_step_state"]["all_finite"]
            for row in calibration + training_simulation
        )
        and training_step == 5
        and training_scale is not None
        and training_scale <= INITIAL_SCALE
        and 1 <= unique_graphs <= 3
        and graph_breaks == 0
        and generated_kernel_count > 0
        and headroom >= MIN_RESERVED_HEADROOM
    )


def paired_transcripts_match(branches):
    if len(branches) != 2 or any(
        "calibration" not in branch or "training_simulation" not in branch
        for branch in branches
    ):
        return False
    return all(
        transcript_identity(branches[0][mode]) == transcript_identity(branches[1][mode])
        for mode in ("calibration", "training_simulation")
    )


def run_branch(
    torch,
    root,
    contract,
    pointer_cases,
    source_roots,
    name,
    device,
    base_state,
):
    from eqvae.data.supervised_latents import SupervisedLatentStore

    with torch.cuda.device(device):
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        torch._inductor.metrics.reset()
        model = make_model(torch, base_state, device)
        numerical = make_numerical(torch, model)
        selected = sorted(contract["selected"], key=lambda row: -row["patch_count"])
        if tuple(row["wsi_id"] for row in selected) != CALIBRATION_ORDER:
            raise RuntimeError("Largest-WSI calibration order differs")
        training_simulation_rows = list(selected)
        random.Random(TRAINING_SHUFFLE_SEED).shuffle(training_simulation_rows)
        if (
            tuple(row["wsi_id"] for row in training_simulation_rows)
            != TRAINING_SIMULATION_ORDER
        ):
            raise RuntimeError("Seeded training-simulation order differs")
        total = torch.cuda.get_device_properties(device).total_memory
        free_before, _ = torch.cuda.mem_get_info(device)
        reserved_before = torch.cuda.memory_reserved(device)
        non_torch_baseline = max(0, total - free_before - reserved_before)
        torch.cuda.reset_peak_memory_stats(device)
        calibration = []
        training_simulation = []
        training_optimizer = None
        scaler = make_scaler(torch)
        scaler_object_id = id(scaler)
        initial_scaler = scaler_state_record(scaler)
        torch.manual_seed(TRAINING_SHUFFLE_SEED)
        torch.cuda.manual_seed_all(TRAINING_SHUFFLE_SEED)
        base_rng_state = capture_rng_state(torch)
        base_rng_sha256 = rng_state_sha256(base_rng_state)
        base_model_sha256 = model_state_sha256(model)
        if not model_matches_state(torch, model, base_state):
            raise RuntimeError("Branch model does not match the shared initial state")
        boundary = None
        calibration_unique_graphs = None
        terminal_failure = None
        with SupervisedLatentStore(
            catalog_path=root / "probe/physical_parts.csv",
            model_name=name,
            source_roots=source_roots,
        ) as store:
            for ordinal, row in enumerate(selected, start=1):
                case = load_case(
                    torch,
                    store,
                    pointer_cases[row["wsi_id"]],
                    row,
                    device,
                )
                restored_model_sha256 = restore_model_state(
                    torch,
                    model,
                    base_state,
                    base_model_sha256,
                )
                restore_rng_state(torch, base_rng_state)
                observed_rng = capture_rng_state(torch)
                optimizer = make_optimizer(torch, model)
                phase_evidence = {
                    "phase": "calibration",
                    "disposable_optimizer_ordinal": ordinal,
                    "model_matches_base_before": model_matches_state(
                        torch,
                        model,
                        base_state,
                    ),
                    "model_sha256_before": restored_model_sha256,
                    "rng_matches_base_before": rng_state_matches(
                        torch,
                        observed_rng,
                        base_rng_state,
                    ),
                    "rng_sha256_before": rng_state_sha256(observed_rng),
                    "optimizer_state_entries_before": len(optimizer.state),
                    "optimizer_step_before": optimizer_step_value(optimizer),
                    "scaler_object_id": scaler_object_id,
                    "scaler_before": scaler_state_record(scaler),
                }
                if (
                    not phase_evidence["model_matches_base_before"]
                    or phase_evidence["model_sha256_before"] != base_model_sha256
                    or not phase_evidence["rng_matches_base_before"]
                    or phase_evidence["rng_sha256_before"] != base_rng_sha256
                    or phase_evidence["optimizer_state_entries_before"] != 0
                    or phase_evidence["optimizer_step_before"] != 0
                    or phase_evidence["scaler_object_id"] != id(scaler)
                ):
                    raise RuntimeError("Calibration reset boundary is not pristine")
                try:
                    attempts = run_weighted_update(
                        torch,
                        model,
                        optimizer,
                        scaler,
                        numerical,
                        case,
                        name,
                    )
                except WeightedUpdateError as error:
                    attempts = error.attempts
                    terminal_failure = str(error)
                phase_evidence["optimizer_step_after"] = optimizer_step_value(optimizer)
                phase_evidence["optimizer_state_entries_after"] = len(optimizer.state)
                phase_evidence["scaler_after"] = scaler_state_record(scaler)
                phase_evidence["successful_commit_count"] = sum(
                    attempt["committed"] for attempt in attempts
                )
                calibration.append(
                    case_record(
                        row,
                        case,
                        attempts,
                        terminal_failure,
                        phase_evidence,
                    ),
                )
                if terminal_failure is None and (
                    phase_evidence["optimizer_step_after"] != 1
                    or phase_evidence["successful_commit_count"] != 1
                ):
                    raise RuntimeError(
                        "Disposable calibration optimizer did not commit exactly once",
                    )
                del case, optimizer
                torch.cuda.empty_cache()
                if terminal_failure is not None:
                    break
            if terminal_failure is None:
                calibration_unique_graphs = int(
                    torch._dynamo.utils.counters["stats"]["unique_graphs"],
                )
                if not 1 <= calibration_unique_graphs <= 3:
                    raise RuntimeError(
                        "Calibration did not produce one to three compiled graphs",
                    )
                calibrated_scaler_before_training = scaler_state_record(scaler)
                model_sha256_before_restore = model_state_sha256(model)
                rng_sha256_before_restore = rng_state_sha256(capture_rng_state(torch))
                model_sha256_after_restore = restore_model_state(
                    torch,
                    model,
                    base_state,
                    base_model_sha256,
                )
                restore_rng_state(torch, base_rng_state)
                restored_rng = capture_rng_state(torch)
                training_optimizer = make_optimizer(torch, model)
                scaler_after_optimizer_creation = scaler_state_record(scaler)
                boundary = {
                    "initial_scaler": initial_scaler,
                    "base_model_sha256": base_model_sha256,
                    "model_sha256_before_restore": model_sha256_before_restore,
                    "model_sha256_after_restore": model_sha256_after_restore,
                    "model_matches_base_after_calibration": model_matches_state(
                        torch,
                        model,
                        base_state,
                    ),
                    "model_gradients_cleared_after_calibration": all(
                        parameter.grad is None for parameter in model.parameters()
                    ),
                    "base_rng_sha256": base_rng_sha256,
                    "rng_sha256_before_restore": rng_sha256_before_restore,
                    "rng_sha256_after_restore": rng_state_sha256(restored_rng),
                    "rng_matches_base_after_calibration": rng_state_matches(
                        torch,
                        restored_rng,
                        base_rng_state,
                    ),
                    "training_optimizer_state_entries_before": len(
                        training_optimizer.state,
                    ),
                    "training_optimizer_step_before": optimizer_step_value(
                        training_optimizer,
                    ),
                    "calibrated_scaler_before_training": (
                        calibrated_scaler_before_training
                    ),
                    "scaler_after_training_optimizer_creation": (
                        scaler_after_optimizer_creation
                    ),
                    "scaler_state_preserved_across_boundary": (
                        calibrated_scaler_before_training
                        == scaler_after_optimizer_creation
                    ),
                    "scaler_object_id": scaler_object_id,
                    "single_scaler_reused": scaler_object_id == id(scaler),
                    "calibration_unique_graphs": calibration_unique_graphs,
                }
                if (
                    not boundary_accepts(boundary)
                    or not boundary["model_gradients_cleared_after_calibration"]
                    or not boundary["single_scaler_reused"]
                ):
                    raise RuntimeError(
                        "Calibration-to-training boundary is not pristine",
                    )
                for ordinal, row in enumerate(training_simulation_rows, start=1):
                    case = load_case(
                        torch,
                        store,
                        pointer_cases[row["wsi_id"]],
                        row,
                        device,
                    )
                    phase_evidence = {
                        "phase": "training_simulation",
                        "training_step_ordinal": ordinal,
                        "optimizer_step_before": optimizer_step_value(
                            training_optimizer,
                        ),
                        "scaler_object_id": scaler_object_id,
                        "scaler_before": scaler_state_record(scaler),
                    }
                    try:
                        attempts = run_weighted_update(
                            torch,
                            model,
                            training_optimizer,
                            scaler,
                            numerical,
                            case,
                            name,
                        )
                    except WeightedUpdateError as error:
                        attempts = error.attempts
                        terminal_failure = str(error)
                    phase_evidence["optimizer_step_after"] = optimizer_step_value(
                        training_optimizer,
                    )
                    phase_evidence["scaler_after"] = scaler_state_record(scaler)
                    training_simulation.append(
                        case_record(
                            row,
                            case,
                            attempts,
                            terminal_failure,
                            phase_evidence,
                        ),
                    )
                    del case
                    torch.cuda.empty_cache()
                    if terminal_failure is not None:
                        break
        unique_graphs = int(torch._dynamo.utils.counters["stats"]["unique_graphs"])
        graph_breaks = int(sum(torch._dynamo.utils.counters["graph_break"].values()))
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        headroom = total - peak_reserved - non_torch_baseline
        generated_kernel_count = int(
            getattr(torch._inductor.metrics, "generated_kernel_count", -1),
        )
        training_step = (
            optimizer_step_value(training_optimizer)
            if training_optimizer is not None
            else None
        )
        training_scale = float(scaler.get_scale())
        final_scaler = scaler_state_record(scaler)
        if boundary is not None:
            boundary["final_scaler"] = final_scaler
        accepted = branch_accepts(
            calibration,
            training_simulation,
            terminal_failure,
            training_step,
            training_scale,
            unique_graphs,
            graph_breaks,
            generated_kernel_count,
            headroom,
            boundary,
            calibration_unique_graphs,
        )
        return {
            "status": "accepted" if accepted else "rejected",
            "branch": name,
            "device": device,
            "calibration_order": list(CALIBRATION_ORDER),
            "training_shuffle_seed": TRAINING_SHUFFLE_SEED,
            "training_simulation_order": list(TRAINING_SIMULATION_ORDER),
            "calibration": calibration,
            "training_simulation": training_simulation,
            "calibration_to_training_boundary": boundary,
            "initial_scaler": initial_scaler,
            "terminal_failure": terminal_failure,
            "training_optimizer_step": training_step,
            "training_final_scale": training_scale,
            "final_scaler": final_scaler,
            "numerical_unique_graphs": unique_graphs,
            "calibration_unique_graphs": calibration_unique_graphs,
            "training_added_unique_graphs": (
                unique_graphs - calibration_unique_graphs
                if calibration_unique_graphs is not None
                else None
            ),
            "graph_breaks": graph_breaks,
            "generated_kernel_count": generated_kernel_count,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "non_torch_baseline_bytes": non_torch_baseline,
            "conservative_reserved_headroom_bytes": headroom,
            "headroom_pass": headroom >= MIN_RESERVED_HEADROOM,
        }


def run_probe(torch, root, contract, result):
    from eqvae.models.local_global_mil import LocalGlobalMILClassifier

    if torch.cuda.device_count() != 2 or any(
        "T4" not in torch.cuda.get_device_name(index) for index in range(2)
    ):
        raise RuntimeError("Spec 0035 requires exactly two Tesla T4 GPUs")
    configure_torch(torch)
    catalog = root / "probe/physical_parts.csv"
    source_roots = resolve_sources(catalog)
    if set(source_roots) != set(contract["kernel_sources"]):
        raise RuntimeError("Spec 0035 producer set differs")
    pointer_cases = load_pointer_cases(root, contract)
    torch.manual_seed(3501)
    prototype = LocalGlobalMILClassifier()
    base_state = {
        name: value.detach().clone() for name, value in prototype.state_dict().items()
    }
    result["runtime"] = {
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "devices": [torch.cuda.get_device_name(index) for index in range(2)],
        "grad_scaler": {
            "initial_scale": INITIAL_SCALE,
            "growth_factor": GROWTH_FACTOR,
            "backoff_factor": BACKOFF_FACTOR,
            "growth_interval": GROWTH_INTERVAL,
            "retry_same_wsi": True,
            "max_overflow_backoffs": MAX_OVERFLOW_BACKOFFS,
        },
    }
    branches = []
    for device, name in enumerate(("normal_vae", "so2_vae")):
        result["phase"] = f"branch_{name}"
        try:
            row = run_branch(
                torch,
                root,
                contract,
                pointer_cases,
                source_roots,
                name,
                device,
                base_state,
            )
        except Exception as error:
            row = {
                "status": "failed",
                "branch": name,
                "device": device,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
        branches.append(row)
        print(json.dumps(row), flush=True)
    result["branches"] = branches
    paired_identity = all(row["status"] == "accepted" for row in branches) and (
        paired_transcripts_match(branches)
    )
    result["paired_transcript_identity"] = paired_identity
    result["accepted_weighted_amp"] = paired_identity
    committed_scales = [
        attempt["scale_before"]
        for branch in branches
        if branch["status"] == "accepted"
        for case_group in (branch["calibration"], branch["training_simulation"])
        for case in case_group
        for attempt in case["attempts"]
        if attempt["committed"]
    ]
    result["minimum_successful_probe_scale"] = (
        min(committed_scales) if result["accepted_weighted_amp"] else None
    )
    result["status"] = "complete" if result["accepted_weighted_amp"] else "rejected"


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
            "Pinned runtime differs: "
            f"torch={torch.__version__}, cuda={torch.version.cuda}",
        )


def execute(result):
    root, contract = resolve_input_bundle()
    result["phase"] = "install_pinned_torch"
    install_pinned_torch()
    import torch

    validate_pinned_torch(torch)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root / "src"))
    result["phase"] = "probe"
    run_probe(torch, root, contract, result)


def main():
    result = {
        "status": "failed",
        "spec": "0035",
        "scope": "train_only_numerical_probe_not_learning",
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
