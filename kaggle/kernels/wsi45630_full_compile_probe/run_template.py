# Copyright 2026 HiperMaximus
# ruff: noqa: ANN001, ANN201, ANN202, BLE001, DOC201, DOC501, EM101, EM102, F821, FBT003, INP001, PLC0415, PLR0911, PLR0913, PLR0914, PLR0915, PLR0916, PLR0917, PLR2004, PLW0717, S102, S404, SLF001, TRY003
"""Run the exact Spec 0034 full compiled fixed-25 MIL probe."""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import math
import os
import statistics
import subprocess
import sys
import textwrap
import time
import traceback
from collections import Counter
from operator import itemgetter
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# fmt: off
# BEGIN EMBEDDED SPEC0032 MODEL
# __EMBEDDED_MODEL__
# END EMBEDDED SPEC0032 MODEL

# BEGIN EMBEDDED SPEC0033 CANDIDATE
# __EMBEDDED_CANDIDATE__
# END EMBEDDED SPEC0033 CANDIDATE
# fmt: on

SPEC0034_FULL_COMPILED_FIXED25_PROBE_READY = True
INPUT_ROOT = Path("/kaggle/input")
OUTPUT_PATH = Path("/kaggle/working/spec0034_full_compiled_fixed25_mil_probe.json")
CONTRACT_JSON = r"""$contract_json"""
CONTRACT_SHA256 = "$contract_sha256"
EMBEDDED_MODEL_SHA256 = "$model_sha256"
EMBEDDED_CANDIDATE_SHA256 = "$candidate_sha256"
INPUT_CONTRACT_NAME = "wsi45630_capacity_input.json"
INPUT_CONTRACT_SHA256 = (
    "99bb4d2f60558aee9691b67be4867ffae434bc306581a000fd5d72a6befac660"
)
POINTER_SHA256 = "08e461846bf16efebac707c82962762f49837916986b29aee0dcd6ca1fc31c6c"
INPUT_DATASET_REFERENCE = "maximusshtefan/eqvae-wsi45630-capacity-inputs"
KERNEL_SOURCES = (
    "maximusshtefan/eqvae-ubc-ocean-latent-run-04",
    "maximusshtefan/eqvae-ubc-ocean-cancer-latent-top-up",
    "maximusshtefan/eqvae-wsi45630-completion",
)
WSI_ID = 45_630
PATCH_COUNT = 32_595
DIAGNOSIS_INDEX = 1
PARAMETER_COUNT = 1_513_055
GRAD_SCALER_INIT_SCALE = 32_768.0
GRAD_SCALER_GROWTH_INTERVAL = 1_000_000
DYNAMIC_REUSE_COUNT = 64
WARMUP_STEPS = 2
MEASURED_STEPS = 5
MIN_RESERVED_HEADROOM = 512 * 1024 * 1024
TRAINING_EQUIVALENCE_STEPS = 5
PINNED_TORCH_VERSION = "2.14.0"
PINNED_TORCH_CUDA = "13.0"
PINNED_TORCH_INDEX = "https://download.pytorch.org/whl/cu130"


def sha256(path):
    """Stream one file hash without copying large inputs."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def embedded_source(begin, end, expected):
    """Extract and authenticate one generated source block."""
    source = Path(__file__).read_text(encoding="utf-8")
    if source.count(begin) != 1 or source.count(end) != 1:
        raise RuntimeError("Embedded source markers differ")
    block = source.split(begin, 1)[1].split(end, 1)[0]
    if not block.startswith("if False:\n"):
        raise RuntimeError("Embedded source guard differs")
    value = textwrap.dedent(block.removeprefix("if False:\n"))
    if hashlib.sha256(value.encode()).hexdigest() != expected:
        raise RuntimeError("Embedded source digest differs")
    return value


def resolve_package():
    """Authenticate the exact execution contract and both source snapshots."""
    if hashlib.sha256(CONTRACT_JSON.encode()).hexdigest() != CONTRACT_SHA256:
        raise RuntimeError("Spec 0034 contract bytes differ")
    model = embedded_source(
        "# BEGIN EMBEDDED SPEC0032 MODEL\n",
        "# END EMBEDDED SPEC0032 MODEL\n",
        EMBEDDED_MODEL_SHA256,
    )
    candidate = embedded_source(
        "# BEGIN EMBEDDED SPEC0033 CANDIDATE\n",
        "# END EMBEDDED SPEC0033 CANDIDATE\n",
        EMBEDDED_CANDIDATE_SHA256,
    )
    contract = json.loads(CONTRACT_JSON)
    if (
        contract.get("schema_version") != "spec0034.full_compiled_fixed25_mil.v1"
        or contract.get("authorization") != "spec0034_pinned_torch_retry_v7_authorized"
        or contract.get("scope")
        != "capacity_optimization_only_not_learning_or_evaluation"
        or contract.get("model", {}).get("parameter_count") != PARAMETER_COUNT
        or contract.get("candidate", {}).get("backend") != "whole_bag_fixed25_inductor"
        or contract.get("input_dataset", {}).get("reference") != INPUT_DATASET_REFERENCE
        or contract.get("input_dataset", {}).get("version") != 1
        or contract.get("input_dataset", {}).get("contract_sha256")
        != INPUT_CONTRACT_SHA256
        or contract.get("input_dataset", {}).get("pointer_sha256") != POINTER_SHA256
        or contract.get("kernel_sources")
        != [{"reference": source, "version": 1} for source in KERNEL_SOURCES]
        or contract.get("wsi_id") != WSI_ID
        or contract.get("patch_count") != PATCH_COUNT
        or contract.get("execution", {}).get("fallback") is not None
    ):
        raise RuntimeError("Spec 0034 execution contract differs")
    return contract, model, candidate


def activate_sources(model, candidate):
    """Define the current model and fixed-25 candidate after torch bootstrap."""
    namespace = globals()
    exec(
        compile(
            "from __future__ import annotations\n" + model,
            "spec0032_model.py",
            "exec",
        ),
        namespace,
    )
    exec(
        compile(
            "from __future__ import annotations\n" + candidate,
            "spec0033_candidate.py",
            "exec",
        ),
        namespace,
    )


def resolve_input_bundle():
    """Authenticate every file in the existing private pointer/source bundle."""
    matches = list(INPUT_ROOT.rglob(INPUT_CONTRACT_NAME))
    if len(matches) != 1 or sha256(matches[0]) != INPUT_CONTRACT_SHA256:
        raise RuntimeError("Expected the exact WSI45630 capacity input contract")
    root = matches[0].parent
    contract = json.loads(matches[0].read_text(encoding="utf-8"))
    if (
        contract.get("dataset_reference") != INPUT_DATASET_REFERENCE
        or contract.get("wsi_id") != WSI_ID
        or contract.get("patch_count") != PATCH_COUNT
        or contract.get("diagnosis_index") != DIAGNOSIS_INDEX
        or contract.get("kernel_sources") != list(KERNEL_SOURCES)
    ):
        raise RuntimeError("Mounted WSI45630 input scope differs")
    for name, record in contract["files"].items():
        path = root / name
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise RuntimeError("Mounted WSI45630 input bytes differ")
    observed = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    if observed != {*contract["files"], INPUT_CONTRACT_NAME}:
        raise RuntimeError("Unexpected file in WSI45630 capacity input dataset")
    return root, contract


def resolve_sources(catalog):
    """Resolve attached latent outputs by owner-qualified sidecar hashes."""
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
                raise RuntimeError("Expected one hash-matched producer sidecar")
            source = row["kaggle_source"]
            if source in roots and roots[source] != matches[0]:
                raise RuntimeError("Paired producer files have different roots")
            roots[source] = matches[0]
    return roots


def load_inputs(torch, root, contract, result):
    """Load both complete aligned bags directly to FP16 channels-last CUDA."""
    from eqvae.data.supervised_latents import (
        LogicalPointer,
        SupervisedLatentStore,
        WSIInstance,
    )

    catalog = root / "probe/physical_parts.csv"
    source_roots = resolve_sources(catalog)
    if set(source_roots) != set(contract["kernel_sources"]):
        raise RuntimeError("Unexpected latent producer set")
    with (root / "probe/pointers.csv").open(newline="", encoding="utf-8") as handle:
        rows = [
            {key: int(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]
    identity = tuple(
        (
            row["atlas_row_index"],
            row["wsi_id"],
            row["x"],
            row["y"],
            row["part"],
            row["file_index"],
        )
        for row in rows
    )
    if (
        len(identity) != PATCH_COUNT
        or {row[1] for row in identity} != {WSI_ID}
        or len({(row[2], row[3]) for row in identity}) != PATCH_COUNT
        or identity != tuple(sorted(identity, key=itemgetter(3, 2)))
    ):
        raise RuntimeError("Expected every unique WSI45630 coordinate in y,x order")
    pointers = tuple(LogicalPointer(row[4], row[5]) for row in identity)
    instances = tuple(
        WSIInstance(
            instance_row=index,
            atlas_row_index=row[0],
            wsi_id=row[1],
            x=row[2],
            y=row[3],
            diagnosis_label="EC",
            diagnosis_index=DIAGNOSIS_INDEX,
            split="train",
            pointer=pointers[index],
        )
        for index, row in enumerate(identity)
    )
    bags = []
    shared_reads = None
    result["input_reads"] = {}
    for device, name in enumerate(("normal_vae", "so2_vae")):
        with SupervisedLatentStore(
            catalog_path=catalog,
            model_name=name,
            source_roots=source_roots,
        ) as store:
            latents, reads = store.read_rows(pointers)
        counts = Counter(str(read.part) for read in reads)
        if (
            tuple(latents.shape) != (PATCH_COUNT, 16, 32, 32)
            or dict(counts) != contract["part_counts"]
            or not torch.isfinite(latents).all().item()
            or (shared_reads is not None and reads != shared_reads)
        ):
            raise RuntimeError("Full paired bag identity, alignment, or values differ")
        shared_reads = reads
        bag = latents.to(
            device=f"cuda:{device}",
            dtype=torch.float16,
            memory_format=torch.channels_last,
        )
        if bag.dtype != torch.float16 or not bag.is_contiguous(
            memory_format=torch.channels_last,
        ):
            raise RuntimeError("CUDA bag is not direct FP16 channels-last")
        bags.append(bag)
        result["input_reads"][name] = {
            "rows": PATCH_COUNT,
            "part_counts": dict(counts),
            "cuda_dtype": str(bag.dtype),
            "cuda_stride": list(bag.stride()),
        }
        del latents
    return bags, instances


def make_instances(count):
    """Create a valid compact graph used only for correctness/dynamic reuse."""
    from eqvae.data.supervised_latents import LogicalPointer, WSIInstance

    width = 8
    coordinates = [(index % width, index // width) for index in range(count)]
    return tuple(
        WSIInstance(
            instance_row=index,
            atlas_row_index=index,
            wsi_id=WSI_ID,
            x=x * 256,
            y=y * 256,
            diagnosis_label="EC",
            diagnosis_index=DIAGNOSIS_INDEX,
            split="train",
            pointer=LogicalPointer(1, index),
        )
        for index, (x, y) in enumerate(coordinates)
    )


def configure_torch(torch):
    """Enable the fastest safe dynamic-shape T4 runtime flags."""
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)
    torch.set_float32_matmul_precision("high")
    torch._functorch.config.backward_pass_autocast = "off"  # noqa: S105 -- PyTorch mode, not a credential.
    if hasattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction"):
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True


def compile_callable(torch, function, *, fullgraph):
    """Compile with the selected dynamic no-CUDA-graphs recipe."""
    kwargs = {
        "backend": "inductor",
        "fullgraph": fullgraph,
        "dynamic": None,
        "mode": "max-autotune-no-cudagraphs",
    }
    signature = inspect.signature(torch.compile)
    if "recompile_limit" in signature.parameters:
        kwargs["recompile_limit"] = 3
    if "isolate_recompiles" in signature.parameters:
        kwargs["isolate_recompiles"] = True
    return torch.compile(function, **kwargs)


def hint_dynamic_bag_axis(torch, latents, graph):
    """Hint only the physical bag axis without forbidding specialization."""
    torch._dynamo.maybe_mark_dynamic(latents, 0)
    for tensor in (graph.neighbor_index, graph.neighbor_valid, graph.radial_code):
        torch._dynamo.maybe_mark_dynamic(tensor, 0)


def make_model(torch, state, device):
    """Build the exact state-compatible fixed-25 full network."""
    model = LocalGlobalMILClassifier()
    model.load_state_dict(state)
    before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    use_whole_bag_fixed25_attention(model)
    after = model.state_dict()
    if before.keys() != after.keys() or any(
        not torch.equal(value, after[name]) for name, value in before.items()
    ):
        raise RuntimeError("Fixed-25 replacement changed model state")
    if not all(
        isinstance(block.attention, WholeBagFixed25Attention)
        for block in model.local_blocks
    ):
        raise RuntimeError("Both local blocks must use whole-bag fixed-25")
    model = model.to(device=device, memory_format=torch.channels_last).train()
    if sum(parameter.numel() for parameter in model.parameters()) != PARAMETER_COUNT:
        raise RuntimeError("Model parameter count differs")
    return model


def initialize_nonzero_attention(torch, model):
    """Prevent a zero-initialization correctness test from becoming vacuous."""
    with torch.no_grad():
        for block_index, block in enumerate(model.local_blocks):
            attention = block.attention
            attention.relative_bias.copy_(
                torch.linspace(-0.2, 0.2, attention.relative_bias.numel()).reshape_as(
                    attention.relative_bias,
                )
                + block_index * 0.01,
            )
            attention.null_key.copy_(
                torch.linspace(-0.1, 0.1, attention.null_key.numel()).reshape_as(
                    attention.null_key,
                ),
            )
            attention.null_bias.copy_(
                torch.linspace(-0.05, 0.05, attention.null_bias.numel()),
            )


def tensor_metrics(torch, actual, reference):
    """Measure a tensor difference without assigning a pass threshold."""
    actual = actual.detach().float().cpu()
    reference = reference.detach().float().cpu()
    delta = actual - reference
    reference_norm = float(torch.linalg.vector_norm(reference).item())
    actual_norm = float(torch.linalg.vector_norm(actual).item())
    return {
        "finite": bool(torch.isfinite(actual).all().item()),
        "max_abs": float(delta.abs().max().item()),
        "relative_l2": float(torch.linalg.vector_norm(delta).item())
        / max(reference_norm, 1e-12),
        "cosine": float((actual.flatten() * reference.flatten()).sum().item())
        / max(reference_norm * actual_norm, 1e-12),
        "reference_norm": reference_norm,
    }


def semantic_parameter_group(name):
    """Map parameters into training-relevant blocks for effect accounting."""
    if name.startswith("patch_encoder."):
        return "patch_encoder"
    if name.startswith("local_blocks.0.attention.") and name.rsplit(".", 1)[-1] in {
        "relative_bias",
        "null_key",
        "null_bias",
    }:
        return "local_0_structure"
    if name.startswith("local_blocks.0."):
        return "local_0_remaining"
    if name.startswith("local_blocks.1.attention.") and name.rsplit(".", 1)[-1] in {
        "relative_bias",
        "null_key",
        "null_bias",
    }:
        return "local_1_structure"
    if name.startswith("local_blocks.1."):
        return "local_1_remaining"
    if name == "global_tokens" or name.startswith("global_summary."):
        return "global_summary"
    return "cls_and_head"


def grouped_vectors(torch, named_values):
    """Concatenate named tensors only within semantic model blocks."""
    grouped = {}
    for name, value in named_values.items():
        grouped.setdefault(semantic_parameter_group(name), []).append(
            value.detach().float().reshape(-1),
        )
    return {name: torch.cat(values).cpu() for name, values in sorted(grouped.items())}


def named_vectors(named_values):
    """Keep training-state effects separate for every named parameter."""
    return {
        name: value.detach().float().reshape(-1).cpu()
        for name, value in named_values.items()
    }


def accumulate_training_effect(
    torch,
    totals,
    key,
    fp32,
    eager,
    replay,
    compiled,
):
    """Measure compiler impact against AMP precision and repeat controls."""
    row = totals.setdefault(
        key,
        {
            "compiler_squared": 0.0,
            "amp_squared": 0.0,
            "replay_squared": 0.0,
            "compiled_fp32_squared": 0.0,
        },
    )
    for field, left, right in (
        ("compiler_squared", compiled, eager),
        ("amp_squared", eager, fp32),
        ("replay_squared", replay, eager),
        ("compiled_fp32_squared", compiled, fp32),
    ):
        delta = left.detach().double().cpu() - right.detach().double().cpu()
        row[field] += float(torch.dot(delta, delta).item())


def finalize_training_effects(totals):
    """Bound compiler-induced training impact by measured AMP/repeat impact."""
    rows = {}
    for key, source in sorted(totals.items()):
        compiler = source["compiler_squared"] ** 0.5
        amp = source["amp_squared"] ** 0.5
        replay = source["replay_squared"] ** 0.5
        compiled_fp32 = source["compiled_fp32_squared"] ** 0.5
        accepted = max(amp, replay)
        finite = all(
            math.isfinite(value)
            for value in (compiler, amp, replay, compiled_fp32, accepted)
        )
        passed = finite and (
            compiler == 0.0  # noqa: RUF069 -- zero envelope requires identity.
            if accepted == 0.0  # noqa: RUF069
            else compiler <= accepted
        )
        rows[key] = {
            "compiler_effect": compiler,
            "amp_effect": amp,
            "replay_amp_eager_effect": replay,
            "compiled_fp32_diagnostic": compiled_fp32,
            "accepted_training_effect": accepted,
            "finite": finite,
            "effect_ratio": (
                0.0
                if compiler == 0.0 and accepted == 0.0  # noqa: RUF069
                else None
                if accepted == 0.0  # noqa: RUF069
                else compiler / accepted
            ),
            "pass": passed,
        }
    return rows


def correctness_loss_function(torch, model, *, amp):
    """Return the unscaled loss path under the requested precision policy."""
    if amp:
        return make_loss_function(torch, model)

    def loss_function(latents, graph, target):
        logits = model(latents, graph)
        loss = torch.nn.functional.cross_entropy(
            logits.float().unsqueeze(0),
            target,
        )
        return loss, logits

    return loss_function


def correctness_optimizer(torch, model):
    """Construct the real fused AdamW path for one independent trajectory."""
    return torch.optim.AdamW(
        local_global_mil_adamw_parameter_groups(model, weight_decay=1e-4),
        lr=2e-4,
        fused=True,
        capturable=True,
    )


def make_grad_scaler(torch, *, enabled=True):
    """Construct the production AMP scaler with ordinary dynamic backoff."""
    return torch.amp.GradScaler(
        "cuda",
        init_scale=GRAD_SCALER_INIT_SCALE,
        growth_interval=GRAD_SCALER_GROWTH_INTERVAL,
        enabled=enabled,
    )


def gradient_diagnostics(torch, named_parameters):
    """Describe every missing or nonfinite unscaled parameter gradient."""
    rows = []
    for name, parameter in named_parameters.items():
        gradient = parameter.grad
        if gradient is None:
            rows.append({"parameter": name, "missing": True})
            continue
        finite = torch.isfinite(gradient)
        if bool(finite.all().item()):
            continue
        finite_values = gradient.detach()[finite]
        rows.append({
            "parameter": name,
            "missing": False,
            "dtype": str(gradient.dtype),
            "nan_count": int(torch.isnan(gradient).sum().item()),
            "positive_inf_count": int(torch.isposinf(gradient).sum().item()),
            "negative_inf_count": int(torch.isneginf(gradient).sum().item()),
            "finite_max_abs": (
                float(finite_values.abs().max().item())
                if finite_values.numel()
                else None
            ),
        })
    return rows


def correctness_training_step(  # noqa: C901 -- explicit diagnostics keep the probe auditable.
    torch,
    model,
    optimizer,
    scaler,
    function,
    bag,
    graph,
    target,
):
    """Execute one ordinary GradScaler attempt and expose its training effects."""
    named_parameters = dict(model.named_parameters())
    before = {
        name: parameter.detach().clone() for name, parameter in named_parameters.items()
    }
    optimizer.zero_grad(set_to_none=True)
    loss, logits = function(bag, graph, target)
    if not torch.isfinite(loss).item() or not torch.isfinite(logits).all().item():
        raise FloatingPointError("Correctness trajectory has nonfinite outputs")
    scale_before = float(scaler.get_scale())
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    bad_gradients = gradient_diagnostics(torch, named_parameters)
    missing = [row["parameter"] for row in bad_gradients if row["missing"]]
    if missing:
        raise RuntimeError(f"Correctness trajectory has missing gradients: {missing}")
    if bad_gradients and not scaler.is_enabled():
        raise FloatingPointError(
            f"FP32 correctness gradients are nonfinite: {bad_gradients}",
        )
    grouped_gradients = (
        None
        if bad_gradients
        else grouped_vectors(
            torch,
            {name: parameter.grad for name, parameter in named_parameters.items()},
        )
    )
    scaler.step(optimizer)
    scaler.update()
    scale_after = float(scaler.get_scale())
    step_skipped = scale_after < scale_before
    if scaler.is_enabled() and step_skipped != bool(bad_gradients):
        raise RuntimeError("GradScaler skip decision differs from gradient diagnostics")
    updates = named_vectors(
        {
            name: parameter.detach() - before[name]
            for name, parameter in named_parameters.items()
        },
    )
    trajectory = named_vectors(
        {name: parameter.detach() for name, parameter in named_parameters.items()},
    )
    first_moment = {}
    second_moment = {}
    steps = {}
    for name, parameter in named_parameters.items():
        state = optimizer.state.get(parameter, {})
        if "exp_avg" not in state or "exp_avg_sq" not in state or "step" not in state:
            raise RuntimeError(f"AdamW state is incomplete for {name}")
        if not torch.isfinite(parameter).all().item():
            raise FloatingPointError(f"Correctness parameter is nonfinite: {name}")
        if not torch.isfinite(updates[name]).all().item():
            raise FloatingPointError(f"Correctness update is nonfinite: {name}")
        if not torch.isfinite(state["exp_avg"]).all().item():
            raise FloatingPointError(f"Correctness first moment is nonfinite: {name}")
        if not torch.isfinite(state["exp_avg_sq"]).all().item():
            raise FloatingPointError(f"Correctness second moment is nonfinite: {name}")
        first_moment[name] = state["exp_avg"]
        second_moment[name] = state["exp_avg_sq"]
        steps[name] = int(state["step"].item())
    return {
        "loss": loss.detach().float().cpu(),
        "logits": logits.detach().float().cpu(),
        "gradient": grouped_gradients,
        "update": updates,
        "parameters": trajectory,
        "exp_avg": named_vectors(first_moment),
        "exp_avg_sq": named_vectors(second_moment),
        "steps": steps,
        "step_skipped": step_skipped,
        "scale_before": scale_before,
        "scale_after": scale_after,
        "bad_gradients": bad_gradients,
    }


def correctness_evaluation(torch, model, panel, graph):
    """Measure post-update behavior through the common eager-FP32 model path."""
    was_training = model.training
    model.eval()
    probabilities = []
    losses = []
    predictions = []
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        for bag, label in panel:
            logits = model(bag.float(), graph).float()
            sample_probabilities = torch.softmax(logits, dim=-1)
            loss = torch.nn.functional.cross_entropy(
                logits.unsqueeze(0),
                label,
            )
            probabilities.append(sample_probabilities)
            losses.append(loss.reshape(1))
            predictions.append(int(logits.argmax().item()))
    model.train(was_training)
    return {
        "probabilities": torch.cat(probabilities).cpu(),
        "losses": torch.cat(losses).cpu(),
        "predictions": predictions,
    }


def full_model_correctness(torch, base_state):  # noqa: C901 -- four comparison arms stay explicit.
    """Compare actual eager/compiled AMP training effects with measured controls."""
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    instances = make_instances(25)
    graph = build_local_attention_graph(instances, expected_instance_count=25).to(
        "cuda:0",
    )
    torch.manual_seed(3402)
    base_latent = torch.randn(25, 16, 32, 32, dtype=torch.float32).contiguous(
        memory_format=torch.channels_last,
    )
    cpu_bags = [
        (
            torch.roll(base_latent, shifts=step, dims=0) * (1.0 + 0.05 * step)
            + 0.01 * step
        ).contiguous(memory_format=torch.channels_last)
        for step in range(TRAINING_EQUIVALENCE_STEPS)
    ]
    fp32_bags = [bag.to("cuda:0") for bag in cpu_bags]
    amp_bags = [bag.to("cuda:0", dtype=torch.float16) for bag in cpu_bags]
    targets = [
        torch.tensor([step % len(CLASS_ORDER)], device="cuda:0")
        for step in range(TRAINING_EQUIVALENCE_STEPS)
    ]
    panel = list(zip(fp32_bags, targets, strict=True))
    arm_names = ("fp32", "eager_amp", "eager_amp_replay", "compiled_amp")
    models = {name: make_model(torch, base_state, "cuda:0") for name in arm_names}
    optimizers = {
        name: correctness_optimizer(torch, models[name]) for name in arm_names
    }
    for name in arm_names:
        optimizer_state_ready(torch, models[name], optimizers[name])
    scalers = {
        name: make_grad_scaler(torch, enabled=name != "fp32") for name in arm_names
    }
    functions = {
        "fp32": correctness_loss_function(torch, models["fp32"], amp=False),
        "eager_amp": correctness_loss_function(torch, models["eager_amp"], amp=True),
        "eager_amp_replay": correctness_loss_function(
            torch,
            models["eager_amp_replay"],
            amp=True,
        ),
    }
    compiled_source = correctness_loss_function(
        torch,
        models["compiled_amp"],
        amp=True,
    )
    hint_dynamic_bag_axis(torch, amp_bags[0], graph)
    functions["compiled_amp"] = compile_callable(
        torch,
        compiled_source,
        fullgraph=True,
    )
    effect_totals = {}
    diagnostics = []
    prediction_diagnostics = []
    exact_steps = True
    scaler_histories_match = True
    committed_steps = dict.fromkeys(arm_names, 0)
    update_coverage = {
        arm: {name: False for name, _ in models[arm].named_parameters()}
        for arm in arm_names
    }
    for step in range(TRAINING_EQUIVALENCE_STEPS):
        rows = {}
        for name in arm_names:
            bag = fp32_bags[step] if name == "fp32" else amp_bags[step]
            rows[name] = correctness_training_step(
                torch,
                models[name],
                optimizers[name],
                scalers[name],
                functions[name],
                bag,
                graph,
                targets[step],
            )
            if not rows[name]["step_skipped"]:
                committed_steps[name] += 1
        expected_step = step + 1
        reference_step_keys = rows["fp32"]["steps"].keys()
        exact_steps &= all(
            row["steps"].keys() == reference_step_keys
            and set(row["steps"].values()) == {committed_steps[name]}
            for name, row in rows.items()
        )
        amp_signatures = {
            name: (
                rows[name]["step_skipped"],
                rows[name]["scale_before"],
                rows[name]["scale_after"],
            )
            for name in ("eager_amp", "eager_amp_replay", "compiled_amp")
        }
        scaler_histories_match &= len(set(amp_signatures.values())) == 1
        for metric in ("update", "parameters", "exp_avg", "exp_avg_sq"):
            parameter_names = rows["fp32"][metric].keys()
            if not all(row[metric].keys() == parameter_names for row in rows.values()):
                raise RuntimeError(f"Correctness parameter names differ for {metric}")
            for parameter_name in parameter_names:
                accumulate_training_effect(
                    torch,
                    effect_totals,
                    f"step_{expected_step}:{metric}:{parameter_name}",
                    rows["fp32"][metric][parameter_name],
                    rows["eager_amp"][metric][parameter_name],
                    rows["eager_amp_replay"][metric][parameter_name],
                    rows["compiled_amp"][metric][parameter_name],
                )
                if metric == "update":
                    for arm, row in rows.items():
                        update_coverage[arm][parameter_name] |= bool(
                            torch.count_nonzero(
                                row[metric][parameter_name],
                            ).item(),
                        )
        evaluations = {
            name: correctness_evaluation(torch, models[name], panel, graph)
            for name in arm_names
        }
        for metric in ("probabilities", "losses"):
            accumulate_training_effect(
                torch,
                effect_totals,
                f"step_{expected_step}:post_update_{metric}",
                evaluations["fp32"][metric],
                evaluations["eager_amp"][metric],
                evaluations["eager_amp_replay"][metric],
                evaluations["compiled_amp"][metric],
            )
        prediction_diagnostics.append({
            arm: evaluations[arm]["predictions"] for arm in arm_names
        })
        diagnostics.append({
            "step": expected_step,
            "label": int(targets[step].item()),
            "scalers": amp_signatures,
            "bad_gradients": {name: rows[name]["bad_gradients"] for name in arm_names},
            "eager_vs_compiled_gradients": (
                None
                if rows["eager_amp"]["gradient"] is None
                or rows["compiled_amp"]["gradient"] is None
                else {
                    group: tensor_metrics(
                        torch,
                        rows["compiled_amp"]["gradient"][group],
                        rows["eager_amp"]["gradient"][group],
                    )
                    for group in rows["eager_amp"]["gradient"]
                }
            ),
            "eager_vs_compiled_loss": tensor_metrics(
                torch,
                rows["compiled_amp"]["loss"],
                rows["eager_amp"]["loss"],
            ),
            "eager_vs_compiled_logits": tensor_metrics(
                torch,
                rows["compiled_amp"]["logits"],
                rows["eager_amp"]["logits"],
            ),
        })
    effects = finalize_training_effects(effect_totals)
    graph_count = int(torch._dynamo.utils.counters["stats"]["unique_graphs"])
    graph_breaks = int(sum(torch._dynamo.utils.counters["graph_break"].values()))
    complete_update_coverage = all(
        all(parameters.values()) for parameters in update_coverage.values()
    )
    correct = (
        all(row["pass"] for row in effects.values())
        and exact_steps
        and scaler_histories_match
        and complete_update_coverage
        and graph_count == 1
        and graph_breaks == 0
    )
    return {
        "correct": correct,
        "criterion": "compiler_training_effect_lte_max_amp_or_repeat_effect",
        "steps": TRAINING_EQUIVALENCE_STEPS,
        "effects": effects,
        "prediction_diagnostics": prediction_diagnostics,
        "optimizer_steps_exact": exact_steps,
        "scaler_histories_match": scaler_histories_match,
        "committed_steps": committed_steps,
        "per_parameter_update_coverage": update_coverage,
        "diagnostics": diagnostics,
        "compiled_unique_graphs": graph_count,
        "compiled_graph_breaks": graph_breaks,
    }


def optimizer_state_ready(torch, model, optimizer):
    """Materialize fused AdamW moments without retaining a parameter update."""
    model_before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    groups = [(group["lr"], group["weight_decay"]) for group in optimizer.param_groups]
    for group in optimizer.param_groups:
        group["lr"] = 0.0
        group["weight_decay"] = 0.0
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    model.load_state_dict(model_before)
    for state in optimizer.state.values():
        state["step"].zero_()
    optimizer.zero_grad(set_to_none=True)
    for group, values in zip(optimizer.param_groups, groups, strict=True):
        group["lr"], group["weight_decay"] = values


def gradients_are_finite(torch, model):
    """Require every parameter gradient to exist and remain finite."""
    checked = 0
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all().item():
            return False, checked, name
        checked += 1
    return True, checked, None


def make_loss_function(torch, model):
    """Create the autocast model plus FP32 loss numerical region."""

    def loss_function(latents, graph, target):
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(latents, graph)
            loss = torch.nn.functional.cross_entropy(
                logits.float().unsqueeze(0),
                target,
            )
        return loss, logits

    return loss_function


def validate_optimizer_state(torch, model, optimizer, expected_step):
    """Require complete finite AdamW state for every trainable parameter."""
    named_parameters = dict(model.named_parameters())
    if len(optimizer.state) != len(named_parameters):
        raise RuntimeError("AdamW state does not cover every parameter")
    for name, parameter in named_parameters.items():
        state = optimizer.state.get(parameter, {})
        if not {"step", "exp_avg", "exp_avg_sq"}.issubset(state):
            raise RuntimeError(f"AdamW state is incomplete for {name}")
        if int(state["step"].item()) != expected_step:
            raise RuntimeError(f"AdamW step differs for {name}")
        if not all(
            torch.isfinite(value).all().item()
            for value in (parameter, state["exp_avg"], state["exp_avg_sq"])
        ):
            raise FloatingPointError(f"AdamW parameter/state is nonfinite: {name}")


def timed_training_step(
    torch,
    model,
    optimizer,
    scaler,
    numerical,
    bag,
    graph,
    target,
):
    """Run one recommended AMP attempt and return synchronized timing."""
    optimizer.zero_grad(set_to_none=True)
    forward_start = torch.cuda.Event(enable_timing=True)
    forward_end = torch.cuda.Event(enable_timing=True)
    update_start = torch.cuda.Event(enable_timing=True)
    update_end = torch.cuda.Event(enable_timing=True)
    wall = time.perf_counter()
    forward_start.record()
    loss, logits = numerical(bag, graph, target)
    if not torch.isfinite(loss).item() or not torch.isfinite(logits).all().item():
        raise FloatingPointError("Nonfinite loss or logits")
    scale_before = float(scaler.get_scale())
    scaler.scale(loss).backward()
    forward_end.record()
    forward_end.synchronize()
    missing = [
        name for name, parameter in model.named_parameters() if parameter.grad is None
    ]
    if missing:
        raise RuntimeError(f"Training attempt has missing gradients: {missing}")
    update_start.record()
    scaler.step(optimizer)
    scaler.update()
    update_end.record()
    update_end.synchronize()
    scale_after = float(scaler.get_scale())
    step_skipped = scale_after < scale_before
    return {
        "loss": float(loss.detach().item()),
        "logits": [float(value) for value in logits.detach().float().cpu()],
        "forward_backward_ms": float(forward_start.elapsed_time(forward_end)),
        "optimizer_ms": float(update_start.elapsed_time(update_end)),
        "wall_ms": (time.perf_counter() - wall) * 1000,
        "gradient_parameter_count": len(tuple(model.parameters())),
        "amp_step_skipped": step_skipped,
        "scale_before": scale_before,
        "scale_after": scale_after,
    }


def summarize_steps(steps):
    """Return stable timing summaries without discarding raw samples."""
    output = {"samples": steps}
    for key in ("forward_backward_ms", "optimizer_ms", "wall_ms"):
        values = [step[key] for step in steps]
        output[key] = {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }
    return output


def run_branch(torch, name, device, bag, graph, small_graph, base_state):
    """Compile and measure one independent complete MIL branch."""
    with torch.cuda.device(device):
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        torch._inductor.metrics.reset()
        model = make_model(torch, base_state, device)
        optimizer = torch.optim.AdamW(
            local_global_mil_adamw_parameter_groups(model, weight_decay=1e-4),
            lr=2e-4,
            fused=True,
            capturable=True,
        )
        optimizer_state_ready(torch, model, optimizer)
        scaler = make_grad_scaler(torch)
        update_status = {
            "compiled": False,
            "fullgraph": False,
            "implementation": "torch_amp_grad_scaler_native_fused_adamw",
            "fallback": None,
        }
        torch._dynamo.utils.counters.clear()
        torch._inductor.metrics.reset()
        numerical = compile_callable(
            torch,
            make_loss_function(torch, model),
            fullgraph=True,
        )
        target = torch.tensor([DIAGNOSIS_INDEX], device=device)
        hint_dynamic_bag_axis(torch, bag, graph)
        small_bag = torch.randn(
            DYNAMIC_REUSE_COUNT,
            16,
            32,
            32,
            device=device,
            dtype=torch.float16,
        ).contiguous(memory_format=torch.channels_last)
        hint_dynamic_bag_axis(torch, small_bag, small_graph)
        total = torch.cuda.get_device_properties(device).total_memory
        free_before, _ = torch.cuda.mem_get_info(device)
        reserved_before = torch.cuda.memory_reserved(device)
        non_torch_baseline = max(0, total - free_before - reserved_before)
        compile_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss, _ = numerical(bag, graph, target)
        loss.backward()
        torch.cuda.synchronize(device)
        if not torch.isfinite(loss).item() or not gradients_are_finite(torch, model)[0]:
            raise FloatingPointError("Compile warmup produced nonfinite values")
        optimizer.zero_grad(set_to_none=True)
        compile_seconds = time.perf_counter() - compile_started
        numerical_graphs_after_real = int(
            torch._dynamo.utils.counters["stats"]["unique_graphs"],
        )
        small_loss, _ = numerical(small_bag, small_graph, target)
        small_loss.backward()
        torch.cuda.synchronize(device)
        if (
            not torch.isfinite(small_loss).item()
            or not gradients_are_finite(torch, model)[0]
        ):
            raise FloatingPointError("Dynamic reuse produced nonfinite values")
        optimizer.zero_grad(set_to_none=True)
        numerical_graphs_after_reuse = int(
            torch._dynamo.utils.counters["stats"]["unique_graphs"],
        )
        graph_breaks = int(sum(torch._dynamo.utils.counters["graph_break"].values()))
        specialization_count = (
            numerical_graphs_after_reuse - numerical_graphs_after_real
        )
        if specialization_count not in {0, 1} or graph_breaks:
            raise RuntimeError(
                "Numerical callable exceeded one cached specialization or broke",
            )
        del small_bag
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        warmup = [
            timed_training_step(
                torch,
                model,
                optimizer,
                scaler,
                numerical,
                bag,
                graph,
                target,
            )
            for _ in range(WARMUP_STEPS)
        ]
        measured = [
            timed_training_step(
                torch,
                model,
                optimizer,
                scaler,
                numerical,
                bag,
                graph,
                target,
            )
            for _ in range(MEASURED_STEPS)
        ]
        numerical_graphs_after_training = int(
            torch._dynamo.utils.counters["stats"]["unique_graphs"],
        )
        graph_breaks_after_training = int(
            sum(torch._dynamo.utils.counters["graph_break"].values()),
        )
        if (
            numerical_graphs_after_training != numerical_graphs_after_reuse
            or graph_breaks_after_training
        ):
            raise RuntimeError(
                "Numerical callable recompiled or broke during committed updates",
            )
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        conservative_headroom = total - peak_reserved - non_torch_baseline
        attempts = warmup + measured
        committed_steps = sum(not step["amp_step_skipped"] for step in attempts)
        measured_committed_steps = sum(
            not step["amp_step_skipped"] for step in measured
        )
        validate_optimizer_state(torch, model, optimizer, committed_steps)
        row = {
            "status": "ok",
            "branch": name,
            "device": device,
            "compile_seconds": compile_seconds,
            "numerical_unique_graphs": numerical_graphs_after_training,
            "dynamic_reuse_added_graph": specialization_count == 1,
            "graph_breaks": graph_breaks_after_training,
            "generated_kernel_count": int(
                getattr(torch._inductor.metrics, "generated_kernel_count", -1),
            ),
            "optimizer": update_status,
            "optimizer_state_entries": len(optimizer.state),
            "optimizer_step": committed_steps,
            "attempted_steps": len(attempts),
            "amp_step_skipped_count": len(attempts) - committed_steps,
            "measured_committed_steps": measured_committed_steps,
            "grad_scaler": {
                "initial_scale": GRAD_SCALER_INIT_SCALE,
                "growth_interval": GRAD_SCALER_GROWTH_INTERVAL,
                "final_scale": float(scaler.get_scale()),
                "policy": "skip_update_and_continue",
            },
            "warmup": warmup,
            "timing": summarize_steps(measured),
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "non_torch_baseline_bytes": non_torch_baseline,
            "conservative_reserved_headroom_bytes": conservative_headroom,
            "headroom_pass": conservative_headroom >= MIN_RESERVED_HEADROOM,
            "losses_finite": all(math.isfinite(step["loss"]) for step in attempts),
        }
        del numerical, scaler, optimizer, model
        torch.cuda.empty_cache()
        return row


def run_probe(torch, root, input_contract, execution_contract, result):
    """Execute correctness and both independent complete T4 branches."""
    if torch.cuda.device_count() != 2 or any(
        "T4" not in torch.cuda.get_device_name(index) for index in range(2)
    ):
        raise RuntimeError("Spec 0034 requires exactly two Tesla T4 GPUs")
    configure_torch(torch)
    result["runtime"] = {
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "devices": [
            {
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
                "total_bytes": torch.cuda.get_device_properties(index).total_memory,
            }
            for index in range(2)
        ],
        "mode_options": torch._inductor.list_mode_options(),
        "allocator_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "compiled_backward_autocast": str(
            torch._functorch.config.backward_pass_autocast,
        ),
    }
    torch.manual_seed(execution_contract["initialization_seed"])
    prototype = LocalGlobalMILClassifier()
    initialize_nonzero_attention(torch, prototype)
    base_state = {
        name: value.detach().clone() for name, value in prototype.state_dict().items()
    }
    result["phase"] = "small_full_model_correctness"
    result["correctness"] = full_model_correctness(torch, base_state)
    if not result["correctness"]["correct"]:
        raise RuntimeError(
            "Compiled training exceeded the measured AMP effect envelope",
        )
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    result["phase"] = "load_complete_bags"
    bags, instances = load_inputs(torch, root, input_contract, result)
    cpu_graph = build_local_attention_graph(
        instances,
        expected_instance_count=PATCH_COUNT,
    )
    cpu_graph.verify_integrity()
    if cpu_graph.node_count != PATCH_COUNT:
        raise RuntimeError("Real graph node count differs")
    graphs = [cpu_graph.to(f"cuda:{index}") for index in range(2)]
    small_cpu_graph = build_local_attention_graph(
        make_instances(DYNAMIC_REUSE_COUNT),
        expected_instance_count=DYNAMIC_REUSE_COUNT,
    )
    small_graphs = [small_cpu_graph.to(f"cuda:{index}") for index in range(2)]
    result["graph"] = {
        "identity_sha256": cpu_graph.identity_sha256,
        "nodes": cpu_graph.node_count,
        "edges": int(cpu_graph.neighbor_valid.sum().item()),
        "minimum_degree": int(cpu_graph.neighbor_valid.sum(dim=1).min().item()),
        "maximum_degree": int(cpu_graph.neighbor_valid.sum(dim=1).max().item()),
    }
    rows = []
    for index, name in enumerate(("normal_vae", "so2_vae")):
        result["phase"] = f"full_branch_{name}"
        try:
            row = run_branch(
                torch,
                name,
                index,
                bags[index],
                graphs[index],
                small_graphs[index],
                base_state,
            )
        except Exception as error:
            row = {
                "status": "failed",
                "branch": name,
                "device": index,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            with torch.cuda.device(index):
                torch.cuda.empty_cache()
        rows.append(row)
        print(json.dumps(row), flush=True)
    result["branches"] = rows
    result["accepted_capacity"] = all(
        row["status"] == "ok"
        and row["headroom_pass"]
        and row["numerical_unique_graphs"] in {1, 2}
        and row["graph_breaks"] == 0
        and row["optimizer"]["implementation"]
        == "torch_amp_grad_scaler_native_fused_adamw"
        and row["optimizer"]["fallback"] is None
        and row["measured_committed_steps"] == MEASURED_STEPS
        and row["losses_finite"]
        for row in rows
    )
    result["status"] = "complete" if result["accepted_capacity"] else "rejected"


def install_pinned_torch():
    """Install the exact latest stack already authenticated by version 5."""
    subprocess.check_call([  # noqa: S603 -- every argument is a fixed constant.
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
    """Reject any imported runtime outside the exact wheel contract."""
    installed_torch = str(torch.__version__).split("+", maxsplit=1)[0]
    if (
        installed_torch != PINNED_TORCH_VERSION
        or torch.version.cuda != PINNED_TORCH_CUDA
    ):
        raise RuntimeError(
            "Installed PyTorch runtime differs from the pinned contract: "
            f"torch={torch.__version__}, cuda={torch.version.cuda}",
        )


def main():
    """Run once and preserve evidence after any compiler/allocation failure."""
    result = {
        "status": "failed",
        "spec": "0034",
        "scope": "capacity_optimization_only_not_learning_or_evaluation",
        "phase": "resolve_package",
        "wsi_id": WSI_ID,
        "patch_count": PATCH_COUNT,
        "input_dataset": f"{INPUT_DATASET_REFERENCE}/1",
        "kernel_sources": [f"{source}/1" for source in KERNEL_SOURCES],
        "contract_sha256": CONTRACT_SHA256,
        "started_unix": time.time(),
    }
    try:
        contract, model, candidate = resolve_package()
        root, input_contract = resolve_input_bundle()
        result["phase"] = "install_pinned_torch"
        install_pinned_torch()
        import torch

        validate_pinned_torch(torch)

        sys.dont_write_bytecode = True
        sys.path.insert(0, str(root / "src"))
        activate_sources(model, candidate)
        result["phase"] = "probe"
        run_probe(torch, root, input_contract, contract, result)
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
