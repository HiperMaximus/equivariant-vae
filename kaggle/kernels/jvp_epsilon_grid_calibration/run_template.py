# Copyright 2026 HiperMaximus
# ruff: noqa: ANN001, ANN202, BLE001, C901, COM812, D103, EM101, FBT003, INP001, PLC0415, PLR0912, PLR0913, PLR2004, PLW0603, TRY003
"""Private result-blind JVP finite-difference epsilon calibration for Spec 0058."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import secrets
import shutil
import sys
import time
import zipfile
from pathlib import Path

# fmt: off
KAGGLE_JVP_EPSILON_GRID_CALIBRATION_READY = True
EMBEDDED_PAYLOAD_B64 = """
$embedded_payload_b64
"""
EMBEDDED_PAYLOAD_ZIP_SHA256 = "$embedded_payload_zip_sha256"
EMBEDDED_PAYLOAD_MANIFEST_SHA256 = "$embedded_payload_manifest_sha256"
# fmt: on

INPUT_ROOT = Path("/kaggle/input")
WORKING_ROOT = Path("/kaggle/working")
PRIVATE_ROOT = WORKING_ROOT / ".spec0058_jvp_epsilon_grid_payload"
OUTPUT_ROOT = WORKING_ROOT / "jvp_epsilon_grid_calibration_v1"
PATCH_BYTES = 3 * 256 * 256
HEADER_BYTES = 64
CONTRACT_SHA256 = "397b2cd6efc4bb7e4b3d775b27e58c63d6d8a7f4f2c0a68fb600ba337d183e08"
SPEC_SHA256 = "037e6f74b3b0f06b8fab3b43da62a6505bbfdd51d98be8d913a04fa3836223fb"
WEIGHT_BUNDLE_CONTRACT_SHA256 = (
    "b5a32ebffd0d88a88d6f21b64ba5c9a23016f05d7a2db0546e442f12a0acecc1"
)
MODEL_KINDS = ("non_eq_vae_translatable", "so2_vae_fixed")
OUTPUT_LABELS = ("branch_a", "branch_b")
EPSILON_GRID = (0.002, 0.004, 0.008, 0.016)
SELECTION_MAX = 0.005
PROGRESS_SCHEMA = "spec0058.jvp_epsilon_grid_progress.v1"
PROGRESS_STARTED = time.perf_counter()
PROGRESS_SEQUENCE = 0
ACTIVE_STAGE = "bootstrap"
FORBIDDEN_PUBLIC_TOKENS = (
    "normal_vae",
    "so2_vae",
    "comparison",
    "difference",
    "ranking",
    "decision",
    "model_identity",
    "branch_permutation",
    "payload_manifest",
    "input_contract",
)
_EVENT_FIELDS = {
    "run_started": frozenset(),
    "stage_started": frozenset(),
    "payload_ready": frozenset(),
    "runtime_policy_ready": frozenset({"device"}),
    "contract_ready": frozenset(),
    "sample_ready": frozenset({"shape", "dtype"}),
    "frozen_bundle_ready": frozenset(),
    "frozen_state_ready": frozenset(),
    "branch_started": frozenset({"branch"}),
    "jvp_direction_measured": frozenset({
        "branch",
        "direction_index",
        "epsilon",
        "residual",
    }),
    "branch_complete": frozenset({"branch"}),
    "calibration_complete": frozenset({"selection_status", "selection"}),
    "output_written": frozenset(),
    "run_complete": frozenset(),
    "failed": frozenset({"failure_code", "exception_class", "gpu_counters"}),
}
_EVENT_STAGES = {
    "run_started": frozenset({"bootstrap"}),
    "stage_started": frozenset({
        "payload",
        "contract",
        "sample",
        "frozen_state",
        "branch",
        "output",
    }),
    "payload_ready": frozenset({"payload"}),
    "runtime_policy_ready": frozenset({"payload"}),
    "contract_ready": frozenset({"contract"}),
    "sample_ready": frozenset({"sample"}),
    "frozen_bundle_ready": frozenset({"frozen_state"}),
    "frozen_state_ready": frozenset({"frozen_state"}),
    "branch_started": frozenset({"branch"}),
    "jvp_direction_measured": frozenset({"branch"}),
    "branch_complete": frozenset({"branch"}),
    "calibration_complete": frozenset({"branch"}),
    "output_written": frozenset({"output"}),
    "run_complete": frozenset({"output"}),
    "failed": frozenset({
        "bootstrap",
        "payload",
        "contract",
        "sample",
        "frozen_state",
        "branch",
        "output",
    }),
}


def main() -> int:
    try:
        _emit_progress("run_started")
        return _run()
    except Exception as error:
        _emit_progress(
            "failed",
            failure_code="controlled_calibration_failure",
            exception_class=_controlled_exception_class(error),
            gpu_counters=_safe_gpu_counters(),
        )
        return 1
    finally:
        shutil.rmtree(PRIVATE_ROOT, ignore_errors=True)


def _run() -> int:
    _set_stage("payload")
    payload_root = _extract_payload(PRIVATE_ROOT)
    _emit_progress("payload_ready")
    sys.path.insert(0, str(payload_root / "src"))
    import numpy as np
    import torch

    from eqvae.evaluation.vae_test import sha256_file, state_dict_sha256
    from eqvae.models.registry import build_model

    if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("required GPU differs")
    _set_runtime_policy(torch)
    _emit_progress("runtime_policy_ready", device="Tesla T4")

    _set_stage("contract")
    contract_path = (
        payload_root / "docs/data/spec0058_jvp_epsilon_grid_calibration_contract.json"
    )
    spec_path = payload_root / "docs/specs/0058-jvp-epsilon-grid-calibration.md"
    _require_hash(contract_path, CONTRACT_SHA256)
    _require_hash(spec_path, SPEC_SHA256)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    _validate_contract(contract)
    _emit_progress("contract_ready")

    _set_stage("sample")
    sample = _load_rank_zero(_load_selector(payload_root, contract), np=np, torch=torch)
    sample = sample.to("cuda:0")
    _emit_progress("sample_ready", shape=list(sample.shape), dtype=str(sample.dtype))

    _set_stage("frozen_state")
    bundle_root, weight_contract = _find_weight_bundle(
        contract["inputs"]["weight_bundle_contract_sha256"]
    )
    _emit_progress("frozen_bundle_ready")
    models, before_hashes = _load_models(
        bundle_root,
        weight_contract,
        contract,
        build_model=build_model,
        sha256_file=sha256_file,
        state_dict_sha256=state_dict_sha256,
        torch=torch,
    )
    _emit_progress("frozen_state_ready")

    _set_stage("branch")
    branches = {}
    global_rows = {epsilon: [] for epsilon in EPSILON_GRID}
    for label, model in zip(OUTPUT_LABELS, _anonymous_order(models), strict=True):
        _emit_progress("branch_started", branch=label)
        summary, rows = _calibrate_branch(model, sample, label=label, torch=torch)
        branches[label] = summary
        for epsilon, values in rows.items():
            global_rows[epsilon].extend(values)
        if state_dict_sha256(model.state_dict()) != before_hashes[id(model)]:
            raise RuntimeError("frozen model state differs")
        _emit_progress("branch_complete", branch=label)

    global_summary = {
        str(epsilon): _summary(rows) for epsilon, rows in global_rows.items()
    }
    selection = _select_epsilon(global_summary)
    _emit_progress(
        "calibration_complete",
        selection_status=selection["status"],
        selection=selection["public"],
    )

    _set_stage("output")
    _write_output(
        contract=contract,
        branches=branches,
        global_summary=global_summary,
        selection=selection,
    )
    _emit_progress("output_written")
    _emit_progress("run_complete")
    return 0


def _set_runtime_policy(torch) -> None:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def _validate_contract(contract) -> None:
    if contract.get("schema") != "spec0058.jvp_epsilon_grid_calibration.v1":
        raise RuntimeError("contract schema differs")
    numerics = contract.get("numerics")
    if (
        not isinstance(numerics, dict)
        or tuple(numerics.get("jvp_epsilon_grid", ())) != EPSILON_GRID
    ):
        raise RuntimeError("epsilon grid differs")
    if numerics.get("tangent_count") != 8 or numerics.get("tangent_seed") != 5053:
        raise RuntimeError("direction contract differs")
    if not math.isclose(
        float(numerics.get("selection_max_relative_l2", -1.0)),
        SELECTION_MAX,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise RuntimeError("selection limit differs")
    if (
        contract.get("inputs", {}).get("weight_bundle_contract_sha256")
        != WEIGHT_BUNDLE_CONTRACT_SHA256
    ):
        raise RuntimeError("weight bundle contract differs")
    if tuple(contract.get("blindness", {}).get("output_labels", ())) != OUTPUT_LABELS:
        raise RuntimeError("output labels differ")


def _load_selector(payload_root, contract):
    path = payload_root / contract["inputs"]["selector_path"]
    _require_hash(path, contract["inputs"]["selector_sha256"])
    selector = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(selector.get("selectors"), list)
        or len(selector["selectors"]) != 25
    ):
        raise RuntimeError("fixed selector differs")
    return selector


def _load_rank_zero(selector, *, np, torch):
    candidates = list(INPUT_ROOT.rglob("ubc_ocean_valid.bin"))
    if len(candidates) != 1:
        raise RuntimeError("validation binary differs")
    row = selector["selectors"][0]
    if row.get("rank") != 0:
        raise RuntimeError("rank-zero selector differs")
    with candidates[0].open("rb") as handle:
        handle.seek(HEADER_BYTES + int(row["file_index"]) * PATCH_BYTES)
        raw = handle.read(PATCH_BYTES)
    if (
        len(raw) != PATCH_BYTES
        or hashlib.sha256(raw).hexdigest() != row["patch_sha256"]
    ):
        raise RuntimeError("rank-zero patch differs")
    array = np.frombuffer(raw, dtype=np.uint8).reshape(1, 3, 256, 256).copy()
    return torch.from_numpy(array).to(dtype=torch.float32).div(255).mul(2).sub(1)


def _find_weight_bundle(expected_contract_sha256):
    candidates = [
        path
        for path in INPUT_ROOT.rglob("spec0045_vae_test_input.json")
        if _sha256(path) == expected_contract_sha256
        and (path.parent / "normal_vae_state.pt").is_file()
        and (path.parent / "so2_vae_state.pt").is_file()
    ]
    if len(candidates) != 1:
        raise RuntimeError("frozen weight bundle differs")
    return candidates[0].parent, json.loads(candidates[0].read_text(encoding="utf-8"))


def _load_models(
    bundle_root,
    weight_contract,
    contract,
    *,
    build_model,
    sha256_file,
    state_dict_sha256,
    torch,
):
    names = ("normal_vae", "so2_vae")
    models, hashes = [], {}
    for name, kind in zip(names, MODEL_KINDS, strict=True):
        record = weight_contract["weights"][name]
        prefix = name.split("_")[0]
        path = bundle_root / f"{name}_state.pt"
        if sha256_file(path) != contract["inputs"][f"{prefix}_state_file_sha256"]:
            raise RuntimeError("state-file hash differs")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if (
            state_dict_sha256(state)
            != contract["inputs"][f"{prefix}_state_dict_sha256"]
        ):
            raise RuntimeError("state-dict hash differs")
        if (
            record["source_checkpoint_sha256"]
            != contract["inputs"][f"{prefix}_checkpoint_sha256"]
        ):
            raise RuntimeError("checkpoint provenance differs")
        model = build_model(kind).to("cuda:0").eval().requires_grad_(False)
        model.load_state_dict(state, strict=True)
        hashes[id(model)] = state_dict_sha256(model.state_dict())
        models.append(model)
    return models, hashes


def _anonymous_order(models):
    return (
        tuple(reversed(models))
        if secrets.SystemRandom().randrange(2)
        else tuple(models)
    )


def _calibrate_branch(model, sample, *, label, torch):
    with torch.inference_mode():
        mu, _ = model.encode(sample)
    z = mu[:1].detach().clone().requires_grad_(True)
    rows = {epsilon: [] for epsilon in EPSILON_GRID}
    for index in range(8):
        direction = _rademacher_direction(z, seed=5053 + index, torch=torch)
        with torch.enable_grad():
            _, jvp = torch.autograd.functional.jvp(
                model.decode, z, direction, strict=True
            )
        for epsilon in EPSILON_GRID:
            with torch.no_grad():
                finite = (
                    model.decode(z + epsilon * direction)
                    - model.decode(z - epsilon * direction)
                ) / (2 * epsilon)
            residual, numerator, denominator = _symmetric_l2(jvp, finite, torch=torch)
            rows[epsilon].append((residual, numerator, denominator))
            _emit_progress(
                "jvp_direction_measured",
                branch=label,
                direction_index=index,
                epsilon=epsilon,
                residual=residual,
            )
    return {str(epsilon): _summary(values) for epsilon, values in rows.items()}, rows


def _rademacher_direction(values, *, seed, torch):
    generator = torch.Generator(device=values.device).manual_seed(seed)
    return (
        torch
        .randint(
            0,
            2,
            values.shape,
            generator=generator,
            device=values.device,
            dtype=torch.int64,
        )
        .mul(2)
        .sub(1)
        .to(dtype=values.dtype)
    )


def _symmetric_l2(left, right, *, torch):
    left_cpu = left.detach().to(device="cpu", dtype=torch.float64)
    right_cpu = right.detach().to(device="cpu", dtype=torch.float64)
    numerator = float(torch.linalg.vector_norm(left_cpu - right_cpu))
    denominator = max(
        float(torch.linalg.vector_norm(left_cpu)),
        float(torch.linalg.vector_norm(right_cpu)),
        1e-12,
    )
    return numerator / denominator, numerator, denominator


def _summary(rows):
    ordered = sorted(item[0] for item in rows)
    middle = len(ordered) // 2
    return {
        "aggregate": math.sqrt(sum(item[1] ** 2 for item in rows))
        / max(math.sqrt(sum(item[2] ** 2 for item in rows)), 1e-12),
        "worst": max(item[0] for item in rows),
        "p50": (ordered[middle - 1] + ordered[middle]) / 2,
        "p95": ordered[-1],
    }


def _select_epsilon(global_summary):
    for epsilon in EPSILON_GRID:
        summary = global_summary[str(epsilon)]
        if summary["aggregate"] <= SELECTION_MAX and summary["worst"] <= SELECTION_MAX:
            return {"status": "selected", "public": {"epsilon": epsilon, **summary}}
    return {"status": "no_eligible_epsilon", "public": {}}


def _write_output(*, contract, branches, global_summary, selection):
    if OUTPUT_ROOT.exists():
        raise RuntimeError("output root already exists")
    temporary = OUTPUT_ROOT.with_name(f".{OUTPUT_ROOT.name}.tmp")
    temporary.mkdir(parents=True, exist_ok=False)
    payload = {
        "schema": "spec0058.jvp_epsilon_grid_calibration_output.v1",
        "direction_count": contract["numerics"]["tangent_count"],
        "epsilon_grid": list(EPSILON_GRID),
        "selection": {"status": selection["status"], **selection["public"]},
        "branches": branches,
        "global": global_summary,
    }
    _assert_blind_document(payload)
    _atomic_json(temporary / "calibration.json", payload)
    temporary.replace(OUTPUT_ROOT)


def _set_stage(stage):
    global ACTIVE_STAGE
    ACTIVE_STAGE = stage
    _emit_progress("stage_started")


def _emit_progress(event, **fields: object):
    global PROGRESS_SEQUENCE
    _validate_progress_event(event, fields)
    PROGRESS_SEQUENCE += 1
    payload = {
        "schema": PROGRESS_SCHEMA,
        "sequence": PROGRESS_SEQUENCE,
        "event": event,
        "stage": ACTIVE_STAGE,
        "elapsed_seconds": time.perf_counter() - PROGRESS_STARTED,
    } | fields
    _assert_blind_document(payload)
    print(json.dumps(payload, sort_keys=True, allow_nan=False), flush=True)


def _validate_progress_event(event, fields):
    if event not in _EVENT_FIELDS or ACTIVE_STAGE not in _EVENT_STAGES[event]:
        raise RuntimeError("progress event differs")
    if frozenset(fields) != _EVENT_FIELDS[event]:
        raise RuntimeError("progress event fields differ")
    if "branch" in fields and fields["branch"] not in OUTPUT_LABELS:
        raise RuntimeError("progress branch differs")
    if "device" in fields and fields["device"] != "Tesla T4":
        raise RuntimeError("progress device differs")
    if "shape" in fields and fields["shape"] != [1, 3, 256, 256]:
        raise RuntimeError("progress shape differs")
    if "dtype" in fields and fields["dtype"] != "torch.float32":
        raise RuntimeError("progress dtype differs")
    if "direction_index" in fields and (
        type(fields["direction_index"]) is not int
        or fields["direction_index"] not in range(8)
    ):
        raise RuntimeError("progress direction differs")
    if "epsilon" in fields and fields["epsilon"] not in EPSILON_GRID:
        raise RuntimeError("progress epsilon differs")
    if "residual" in fields and not _nonnegative_finite(fields["residual"]):
        raise RuntimeError("progress residual differs")
    if "selection_status" in fields and fields["selection_status"] not in {
        "selected",
        "no_eligible_epsilon",
    }:
        raise RuntimeError("progress selection status differs")
    if "selection" in fields:
        selection = fields["selection"]
        if selection == {}:
            return
        if not isinstance(selection, dict) or set(selection) != {
            "epsilon",
            "aggregate",
            "worst",
            "p50",
            "p95",
        }:
            raise RuntimeError("progress selection differs")
        if selection["epsilon"] not in EPSILON_GRID or not all(
            _nonnegative_finite(value)
            for key, value in selection.items()
            if key != "epsilon"
        ):
            raise RuntimeError("progress selection differs")
    if (
        "failure_code" in fields
        and fields["failure_code"] != "controlled_calibration_failure"
    ):
        raise RuntimeError("progress failure differs")
    if "exception_class" in fields and fields["exception_class"] not in {
        "AssertionError",
        "FileNotFoundError",
        "KeyError",
        "RuntimeError",
        "ValueError",
        "UnhandledError",
    }:
        raise RuntimeError("progress exception differs")
    if "gpu_counters" in fields:
        _validate_gpu_counters(fields["gpu_counters"])


def _nonnegative_finite(value):
    return (
        type(value) in {int, float}
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _validate_gpu_counters(value):
    if value in (
        {"available": False},
        {"available": True, "counters": "unavailable"},
    ):
        return
    if (
        not isinstance(value, dict)
        or set(value) != {"available", "allocated_bytes", "reserved_bytes"}
        or value["available"] is not True
        or any(
            type(value[key]) is not int or value[key] < 0
            for key in ("allocated_bytes", "reserved_bytes")
        )
    ):
        raise RuntimeError("gpu counters differ")


def _controlled_exception_class(error):
    known = {
        "AssertionError",
        "FileNotFoundError",
        "KeyError",
        "RuntimeError",
        "ValueError",
    }
    return type(error).__name__ if type(error).__name__ in known else "UnhandledError"


def _safe_gpu_counters():
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available():
        return {"available": False}
    try:
        return {
            "available": True,
            "allocated_bytes": int(torch.cuda.memory_allocated()),
            "reserved_bytes": int(torch.cuda.memory_reserved()),
        }
    except Exception:
        return {"available": True, "counters": "unavailable"}


def _assert_blind_document(value) -> None:
    encoded = json.dumps(value, sort_keys=True).lower()
    if any(forbidden in encoded for forbidden in FORBIDDEN_PUBLIC_TOKENS):
        raise RuntimeError("public calibration output is not blind")


def _atomic_json(path, value):
    pending = path.with_suffix(path.suffix + ".tmp")
    with pending.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
    pending.replace(path)


def _extract_payload(destination):
    payload = base64.b64decode(EMBEDDED_PAYLOAD_B64.encode("ascii"))
    if hashlib.sha256(payload).hexdigest() != EMBEDDED_PAYLOAD_ZIP_SHA256:
        raise RuntimeError("embedded payload differs")
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name in archive.namelist():
            candidate = Path(name)
            if candidate.is_absolute() or ".." in candidate.parts:
                raise RuntimeError("embedded payload path differs")
        archive.extractall(destination)
    manifest_path = destination / "payload_manifest.json"
    if _sha256(manifest_path) != EMBEDDED_PAYLOAD_MANIFEST_SHA256:
        raise RuntimeError("embedded payload manifest differs")
    return destination


def _require_hash(path, expected):
    if _sha256(path) != expected:
        raise RuntimeError("required payload hash differs")


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
