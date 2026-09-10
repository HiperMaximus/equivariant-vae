# Copyright 2026 HiperMaximus
# ruff: noqa: ANN001, ANN202, ARG001, BLE001, C901, COM812, D103, E501, EM101, EM102, FBT003, INP001, PLC0415, PLR0912, PLR0913, PLR0914, PLR0915, PLR2004, TRY003
"""Private, result-blind numerical retry preflight for Spec 0053."""

from __future__ import annotations

import base64
import hashlib
import io
import itertools
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
KAGGLE_FUNCTIONAL_GEOMETRY_JVP_LADDER_READY = True
EMBEDDED_PAYLOAD_B64 = """
$embedded_payload_b64
"""
EMBEDDED_PAYLOAD_ZIP_SHA256 = "$embedded_payload_zip_sha256"
EMBEDDED_PAYLOAD_MANIFEST_SHA256 = "$embedded_payload_manifest_sha256"
# fmt: on

INPUT_ROOT = Path("/kaggle/input")
WORKING_ROOT = Path("/kaggle/working")
PRIVATE_ROOT = WORKING_ROOT / ".spec0057_jvp_ladder_payload"
OUTPUT_ROOT = WORKING_ROOT / "preflight_jvp_ladder_parent_v1"
PATCH_BYTES = 3 * 256 * 256
HEADER_BYTES = 64
CONTRACT_SHA256 = "bc48f1f6d4a3088501054aa246cfa2785adec9947914faf76e9b44501e359482"
SPEC_SHA256 = "6821036a7b616f34ec5ddee339d13bc15b8167b0359e089dbc89a5eed30144b0"
WEIGHT_BUNDLE_CONTRACT_SHA256 = (
    "b5a32ebffd0d88a88d6f21b64ba5c9a23016f05d7a2db0546e442f12a0acecc1"
)
MODEL_KINDS = ("non_eq_vae_translatable", "so2_vae_fixed")
OUTPUT_LABELS = ("branch_a", "branch_b")
FORBIDDEN_PUBLIC_TOKENS = (
    "normal_vae",
    "so2_vae",
    "comparison",
    "difference",
    "ranking",
    "median",
    "hypothesis",
    "decision",
    "payload_manifest",
    "branch_permutation",
    "model_identity",
)
_D4_NAMES = (
    "identity",
    "rot90",
    "rot180",
    "rot270",
    "flip_h",
    "flip_v",
    "flip_diag",
    "flip_anti",
)
PROGRESS_SCHEMA = "spec0053.preflight_progress.v1"
PROGRESS_STARTED = time.perf_counter()
PROGRESS_SEQUENCE = 0
ACTIVE_STAGE = "bootstrap"
_PROGRESS_EVENT_FIELDSETS = {
    "run_started": (frozenset(),),
    "stage_started": (frozenset(),),
    "payload_ready": (frozenset(),),
    "runtime_policy_ready": (frozenset({"device"}),),
    "contract_ready": (frozenset(),),
    "fixtures_started": (frozenset(),),
    "fixtures_complete": (frozenset({"passed"}),),
    "sample_ready": (frozenset({"shape", "dtype"}),),
    "frozen_bundle_ready": (frozenset(),),
    "frozen_state_ready": (frozenset(),),
    "branch_started": (frozenset({"branch"}),),
    "branch_complete": (frozenset({"branch"}),),
    "workload_started": (frozenset({"branch", "workload"}),),
    "workload_complete": (frozenset({"branch", "workload", "profile"}),),
    "eager_direction_measured": (
        frozenset({
            "branch",
            "direction_index",
            "epsilon",
            "residuals",
            "limits",
            "failed_checks",
        }),
        frozenset({
            "branch",
            "direction_index",
            "diagnostic_only",
            "epsilon",
            "residuals",
            "limits",
            "failed_checks",
        }),
    ),
    "eager_ad_measured": (
        frozenset({
            "branch",
            "residuals",
            "diagnostic_residuals",
            "limits",
            "failed_checks",
        }),
    ),
    "compilation_started": (frozenset({"branch"}),),
    "compilation_complete": (frozenset({"branch", "mode", "passed"}),),
    "direction_ladder_started": (frozenset({"branch", "directions"}),),
    "direction_ladder_complete": (frozenset({"branch", "directions", "profile"}),),
    "direction_ladder_skipped": (frozenset({"branch", "directions", "reason"}),),
    "parent_output_started": (frozenset(),),
    "parent_output_complete": (frozenset(),),
    "run_complete": (frozenset(),),
    "failed": (frozenset({"failure_code", "exception_class", "gpu_counters"}),),
}
_PROGRESS_EVENT_STAGES = {
    "run_started": frozenset({"bootstrap"}),
    "stage_started": frozenset({
        "payload",
        "contract",
        "fixtures",
        "sample",
        "frozen_state",
        "branch",
        "output",
    }),
    "payload_ready": frozenset({"payload"}),
    "runtime_policy_ready": frozenset({"payload"}),
    "contract_ready": frozenset({"contract"}),
    "fixtures_started": frozenset({"fixtures"}),
    "fixtures_complete": frozenset({"fixtures"}),
    "sample_ready": frozenset({"sample"}),
    "frozen_bundle_ready": frozenset({"frozen_state"}),
    "frozen_state_ready": frozenset({"frozen_state"}),
    "branch_started": frozenset({"branch"}),
    "branch_complete": frozenset({"branch"}),
    "workload_started": frozenset({"branch"}),
    "workload_complete": frozenset({"branch"}),
    "eager_direction_measured": frozenset({"branch"}),
    "eager_ad_measured": frozenset({"branch"}),
    "compilation_started": frozenset({"branch"}),
    "compilation_complete": frozenset({"branch"}),
    "direction_ladder_started": frozenset({"branch"}),
    "direction_ladder_complete": frozenset({"branch"}),
    "direction_ladder_skipped": frozenset({"branch"}),
    "parent_output_started": frozenset({"output"}),
    "parent_output_complete": frozenset({"output"}),
    "run_complete": frozenset({"output"}),
    "failed": frozenset({
        "bootstrap",
        "payload",
        "contract",
        "fixtures",
        "sample",
        "frozen_state",
        "branch",
        "output",
    }),
}
_PROGRESS_WORKLOADS = frozenset({
    "c4",
    "padded_continuous",
    "differentiable_path",
    "eager_jvp_vjp_hvp",
    "compiled_jvp_hvp_closure",
    "jvp_direction_microbatch",
})
_PROGRESS_RESIDUALS = frozenset({"jvp_fd", "vjp_adjoint", "hvp_fd", "repeat"})


def main() -> int:
    try:
        _emit_progress("run_started")
        return _run()
    except Exception as error:
        _emit_progress(
            "failed",
            failure_code="controlled_preflight_failure",
            exception_class=_controlled_exception_class(error),
            gpu_counters=_safe_gpu_counters(),
        )
        return 1
    finally:
        shutil.rmtree(PRIVATE_ROOT, ignore_errors=True)


def _run() -> int:
    _set_stage("payload")
    payload_root, _ = _extract_payload(PRIVATE_ROOT)
    _emit_progress("payload_ready")
    sys.path.insert(0, str(payload_root / "src"))
    import numpy as np
    import torch

    from eqvae.artifacts.rotation_orbits import continuous_rotate
    from eqvae.evaluation.vae_test import sha256_file, state_dict_sha256
    from eqvae.models.registry import build_model

    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA GPU is required")
    device_name = torch.cuda.get_device_name(0)
    if "T4" not in device_name:
        raise RuntimeError(f"a Tesla T4 is required, found {device_name}")
    if OUTPUT_ROOT.exists():
        raise RuntimeError("output root already exists")
    _set_runtime_policy(torch)
    _emit_progress("runtime_policy_ready", device=device_name)
    _set_stage("contract")
    contract_path = payload_root / "docs/data/spec0057_jvp_epsilon_ladder_contract.json"
    spec_path = payload_root / "docs/specs/0057-jvp-epsilon-ladder-preflight.md"
    _require_hash(contract_path, CONTRACT_SHA256)
    _require_hash(spec_path, SPEC_SHA256)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    _validate_contract(contract)
    _emit_progress("contract_ready")
    _set_stage("fixtures")
    _emit_progress("fixtures_started")
    fixture = _run_analytic_fixtures(contract, np=np, torch=torch)
    if not fixture["passed"]:
        raise RuntimeError(f"analytic fixture failure: {fixture}")
    _emit_progress("fixtures_complete", passed=True)
    _set_stage("sample")
    selector = _load_selector(payload_root, contract)
    sample = _load_rank_zero(selector, np=np, torch=torch).to("cuda:0")
    _emit_progress(
        "sample_ready",
        shape=list(sample.shape),
        dtype=str(sample.dtype),
    )
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
    branches = {}
    for label, model in zip(OUTPUT_LABELS, _anonymous_order(models), strict=True):
        _set_stage("branch")
        _emit_progress("branch_started", branch=label)
        branches[label] = _measure_branch(
            model,
            sample,
            contract,
            label=label,
            continuous_rotate=continuous_rotate,
            torch=torch,
        )
        after_hash = state_dict_sha256(model.state_dict())
        if after_hash != before_hashes[id(model)]:
            raise RuntimeError("frozen model state changed during preflight")
        branches[label]["state_preserved"] = True
        torch.cuda.empty_cache()
        _emit_progress("branch_complete", branch=label)
    _assert_blind_document(branches)
    _set_stage("output")
    _emit_progress("parent_output_started")
    _write_parent_output(
        contract=contract,
        fixture=fixture,
        branches=branches,
        runtime=_runtime_identity(device_name, torch=torch, np=np),
    )
    _emit_progress("parent_output_complete")
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
    if contract.get("schema") != "spec0057.jvp_epsilon_ladder_preflight.v1":
        raise RuntimeError("unexpected preflight contract schema")
    if tuple(contract["blindness"]["output_labels"]) != OUTPUT_LABELS:
        raise RuntimeError("anonymous output labels differ")
    if (
        contract["blindness"].get("branch_permutation")
        != "secure_random_nonpersisted_per_run"
    ):
        raise RuntimeError("anonymous permutation policy differs")
    if tuple(contract["tensor_contract"]["direction_microbatches"]) != (
        1,
        2,
        4,
        8,
        16,
        32,
    ):
        raise RuntimeError("direction ladder differs")
    if contract["numerics"].get("tangent_count") != 8:
        raise RuntimeError("tangent count differs")
    if tuple(contract["numerics"].get("jvp_diagnostic_epsilons", ())) != (
        0.004,
        0.008,
    ):
        raise RuntimeError("JVP epsilon ladder differs")
    diagnostic_hvp_epsilon = contract["numerics"].get("diagnostic_hvp_epsilon")
    if type(diagnostic_hvp_epsilon) is not float or not math.isclose(
        diagnostic_hvp_epsilon,
        0.002,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise RuntimeError("diagnostic HVP epsilon differs")
    if (
        contract["inputs"].get("weight_bundle_contract_sha256")
        != WEIGHT_BUNDLE_CONTRACT_SHA256
    ):
        raise RuntimeError("frozen bundle contract hash differs")
    if contract["remote"]["parent_launch_count"] != 1:
        raise RuntimeError("parent launch ceiling differs")
    if contract["remote"]["maximum_child_continuations"] != 1:
        raise RuntimeError("continuation ceiling differs")
    if contract["outputs"].get("required_parent_download_receipt") != {
        "name": "kaggle_output_receipt.json",
        "resource_reference": "maximshtefan/eqvae-functional-geometry-preflight-04a08ab5/1",
        "schema": "eqvae.kaggle_download.v1",
    }:
        raise RuntimeError("parent download receipt contract differs")


def _run_analytic_fixtures(contract, *, np, torch) -> dict[str, object]:
    group = _d4_fixture(torch=torch)
    lie = _lie_fixture(torch=torch)
    metric = _metric_fixture(contract, torch=torch, np=np)
    work_ids = _work_id_fixture()
    return {
        "passed": bool(
            group["passed"] and lie["passed"] and metric["passed"] and work_ids
        ),
        "d4": group,
        "lie": lie,
        "metric": metric,
        "work_id_framing": work_ids,
    }


def _d4_fixture(*, torch) -> dict[str, object]:
    scalar = torch.arange(49, dtype=torch.float64).reshape(1, 1, 7, 7)
    vector = torch.stack((scalar[0, 0], scalar[0, 0].square()), dim=0)[None]
    orbit = {name: _exact_action(scalar, name, torch=torch) for name in _D4_NAMES}
    table: dict[tuple[str, str], str] = {}
    scalar_ok = True
    vector_ok = True
    pseudo_ok = True
    for left in _D4_NAMES:
        for right in _D4_NAMES:
            composed = _exact_action(
                _exact_action(scalar, right, torch=torch), left, torch=torch
            )
            targets = [
                name for name, value in orbit.items() if torch.equal(composed, value)
            ]
            if len(targets) != 1:
                scalar_ok = False
                continue
            target = targets[0]
            table[left, right] = target
            vector_ok &= torch.equal(
                _f1_action(_f1_action(vector, right, torch=torch), left, torch=torch),
                _f1_action(vector, target, torch=torch),
            )
            pseudo_ok &= torch.equal(
                _f1_action(
                    _f1_action(vector, right, pseudo=True, torch=torch),
                    left,
                    pseudo=True,
                    torch=torch,
                ),
                _f1_action(vector, target, pseudo=True, torch=torch),
            )
    identity_ok = all(
        table.get(("identity", name)) == name and table.get((name, "identity")) == name
        for name in _D4_NAMES
    )
    inverse_ok = all(
        any(table.get((left, right)) == "identity" for right in _D4_NAMES)
        for left in _D4_NAMES
    )
    return {
        "passed": bool(
            scalar_ok
            and vector_ok
            and pseudo_ok
            and identity_ok
            and inverse_ok
            and len(table) == 64
        ),
        "table_entries": len(table),
        "scalar_exact": bool(scalar_ok),
        "f1_vector_exact": bool(vector_ok),
        "f1_pseudovector_exact": bool(pseudo_ok),
        "identity_inverse": bool(identity_ok and inverse_ok),
    }


def _exact_action(values, name, *, torch):
    table = {
        "identity": lambda x: x,
        "rot90": lambda x: torch.rot90(x, 1, dims=(-2, -1)),
        "rot180": lambda x: torch.rot90(x, 2, dims=(-2, -1)),
        "rot270": lambda x: torch.rot90(x, 3, dims=(-2, -1)),
        "flip_h": lambda x: torch.flip(x, dims=(-1,)),
        "flip_v": lambda x: torch.flip(x, dims=(-2,)),
        "flip_diag": lambda x: x.transpose(-2, -1),
        "flip_anti": lambda x: torch.flip(x.transpose(-2, -1), dims=(-2, -1)),
    }
    try:
        return table[name](values)
    except KeyError as error:
        raise ValueError(f"unknown exact action: {name}") from error


def _f1_action(values, name, *, pseudo=False, torch):
    matrices = {
        "identity": ((1, 0), (0, 1)),
        "rot90": ((0, -1), (1, 0)),
        "rot180": ((-1, 0), (0, -1)),
        "rot270": ((0, 1), (-1, 0)),
        "flip_h": ((-1, 0), (0, 1)),
        "flip_v": ((1, 0), (0, -1)),
        "flip_diag": ((0, -1), (-1, 0)),
        "flip_anti": ((0, 1), (1, 0)),
    }
    matrix = torch.tensor(matrices[name], dtype=values.dtype, device=values.device)
    if pseudo and name.startswith("flip_"):
        matrix = -matrix
    return torch.einsum(
        "ij,bjhw->bihw", matrix, _exact_action(values, name, torch=torch)
    )


def _lie_fixture(*, torch) -> dict[str, object]:
    coordinates = torch.linspace(-0.8, 0.8, 41, dtype=torch.float64)
    y, x = torch.meshgrid(coordinates, coordinates, indexing="ij")

    def scalar(px, py):
        return px.square() + 2 * py

    def vector(px, py):
        return torch.stack((px.square() + py, px - py.square()), dim=0)

    scalar_exact = 2 * x * y - 2 * x
    vector_value = vector(x, y)
    vector_exact = torch.stack(
        (-vector_value[1] + 2 * x * y - x, vector_value[0] + y + 2 * x * y),
        dim=0,
    )
    errors = []
    for degrees in (1.0, 0.5, 0.25):
        theta = math.radians(degrees)

        def rotated(function, angle):
            px = math.cos(angle) * x + math.sin(angle) * y
            py = -math.sin(angle) * x + math.cos(angle) * y
            return function(px, py)

        scalar_fd = (rotated(scalar, theta) - rotated(scalar, -theta)) / (2 * theta)
        plus = rotated(vector, theta)
        minus = rotated(vector, -theta)
        rotation_plus = torch.stack(
            (
                math.cos(theta) * plus[0] - math.sin(theta) * plus[1],
                math.sin(theta) * plus[0] + math.cos(theta) * plus[1],
            ),
            dim=0,
        )
        rotation_minus = torch.stack(
            (
                math.cos(theta) * minus[0] + math.sin(theta) * minus[1],
                -math.sin(theta) * minus[0] + math.cos(theta) * minus[1],
            ),
            dim=0,
        )
        vector_fd = (rotation_plus - rotation_minus) / (2 * theta)
        errors.append(
            max(
                _relative_l2(scalar_fd, scalar_exact, torch=torch),
                _relative_l2(vector_fd, vector_exact, torch=torch),
            )
        )
    return {
        "passed": bool(
            errors[-1] <= 1e-2 and errors[1] < errors[0] and errors[2] < errors[1]
        ),
        "contraction": bool(errors[1] < errors[0] and errors[2] < errors[1]),
    }


def _metric_fixture(contract, *, torch, np) -> dict[str, object]:
    tolerance = float(contract["numerics"]["analytic_relative_max"])
    zero_tolerance = float(contract["numerics"]["analytic_zero_absolute_max"])
    nodes, weights = np.polynomial.legendre.leggauss(4)
    x = torch.tensor(nodes, dtype=torch.float64)
    w = torch.tensor(weights, dtype=torch.float64)
    yy, xx = torch.meshgrid(x, x, indexing="ij")
    weights_2d = torch.outer(w, w)
    field = xx + yy
    l2 = torch.sum(weights_2d * field.square())
    h1 = l2 + 0.5 * torch.sum(weights_2d * 2.0)
    radial_nodes, radial_weights = np.polynomial.legendre.leggauss(4)
    radius = torch.tensor((radial_nodes + 1) / 2, dtype=torch.float64)
    radial_weight = torch.tensor(radial_weights / 2, dtype=torch.float64)
    disk = torch.sum(radius.square() * radius * radial_weight) * (2 * math.pi)
    values = torch.tensor([1.0, 2.0, 0.0, -1.0], dtype=torch.complex128)
    indices = torch.arange(4, dtype=torch.float64)
    fourier = torch.stack([
        torch.sum(values * torch.exp(-2j * math.pi * mode * indices / 4)) / 4
        for mode in range(4)
    ])
    reconstructed = torch.stack([
        torch.sum(fourier * torch.exp(2j * math.pi * position * indices / 4))
        for position in range(4)
    ])
    phase = torch.exp(2j * math.pi * torch.outer(indices, indices) / 4)
    projector = phase[:, 1:2] @ phase[:, 1:2].conj().T / 4
    projected = projector @ values
    c_matrix = torch.diag(torch.tensor([4.0, 1.0], dtype=torch.float64))
    metric = torch.diag(torch.tensor([2.0, 1.0], dtype=torch.float64))
    metric_inverse_sqrt = torch.diag(
        torch.tensor([1 / math.sqrt(2), 1.0], dtype=torch.float64)
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(
        metric_inverse_sqrt @ c_matrix @ metric_inverse_sqrt
    )
    principal = metric_inverse_sqrt @ eigenvectors[:, 1]
    w2_squared = torch.sum(torch.tensor([1.0, 2.0], dtype=torch.float64).square()) * 2
    fisher = (
        torch.sum(torch.tensor([1.0, 1.0], dtype=torch.float64).square())
        + 4 * math.log(2) ** 2
    )
    path = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    path_energy = torch.sum(torch.diff(path).square() / 0.5)
    quotient_seed = torch.zeros((1, 1, 7, 7), dtype=torch.float64)
    quotient_seed[..., 1, 3] = 1.0
    quotient_target = _exact_action(quotient_seed, "rot90", torch=torch)
    quotient = min(
        float(
            torch.linalg.vector_norm(
                _exact_action(quotient_seed, name, torch=torch) - quotient_target
            )
        )
        for name in ("identity", "rot90", "rot180", "rot270")
    )
    edges = {(0, 1): 1, (1, 2): 2, (2, 0): 1}
    gauge = {0: 3, 1: 1, 2: 0}
    transformed = {
        (left, right): (gauge[left] + value - gauge[right]) % 4
        for (left, right), value in edges.items()
    }
    randomized = _randomized_range_fixture(contract, torch=torch)
    checks = (
        _close(l2, 8 / 3, relative=tolerance, zero=zero_tolerance, torch=torch),
        _close(h1, 20 / 3, relative=tolerance, zero=zero_tolerance, torch=torch),
        _close(disk, math.pi / 2, relative=tolerance, zero=zero_tolerance, torch=torch),
        _relative_l2(reconstructed, values, torch=torch) <= tolerance,
        _relative_l2(projector @ projected, projected, torch=torch) <= tolerance,
        _close(
            eigenvalues[1], 2.0, relative=tolerance, zero=zero_tolerance, torch=torch
        ),
        _close(
            torch.dot(principal, metric @ principal),
            1.0,
            relative=tolerance,
            zero=zero_tolerance,
            torch=torch,
        ),
        _close(w2_squared, 10.0, relative=tolerance, zero=zero_tolerance, torch=torch),
        _close(
            fisher,
            2 + 4 * math.log(2) ** 2,
            relative=tolerance,
            zero=zero_tolerance,
            torch=torch,
        ),
        _close(path_energy, 1.0, relative=tolerance, zero=zero_tolerance, torch=torch),
        quotient <= zero_tolerance,
        sum(edges.values()) % 4 == sum(transformed.values()) % 4 == 0,
        randomized["residual_passed"],
        randomized["trace_passed"],
    )
    return {
        "passed": bool(all(checks)),
        "checks": len(checks),
        "randomized_certificate_confidence": contract["numerics"][
            "randomized_certificate_confidence"
        ],
        "randomized": randomized,
    }


def _randomized_range_fixture(contract, *, torch) -> dict[str, object]:
    numerics = contract["numerics"]
    rows, columns = numerics["randomized_sketch_shape"]
    matrix = torch.tensor(
        (
            (9.0, 1.0, 0.0, 0.0),
            (0.0, 7.0, 1.0, 0.0),
            (0.0, 0.0, 4.0, 1.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=torch.float64,
    )
    if (rows, columns) != (4, 2) or columns >= torch.linalg.matrix_rank(matrix):
        raise RuntimeError("randomized sketch is not strictly truncated")
    singular_values = torch.linalg.svdvals(matrix)
    exact_tail = torch.linalg.vector_norm(singular_values[columns:])

    def residual_ratio(omega):
        basis, _ = torch.linalg.qr(matrix @ omega, mode="reduced")
        approximation = basis @ (basis.T @ matrix)
        return float(torch.linalg.vector_norm(matrix - approximation) / exact_tail)

    exhaustive = [
        residual_ratio(torch.tensor(signs, dtype=torch.float64).reshape(rows, columns))
        for signs in itertools.product((-1.0, 1.0), repeat=rows * columns)
    ]
    residual_limit = float(numerics["randomized_residual_ratio_99_max"])
    coverage = sum(value <= residual_limit for value in exhaustive) / len(exhaustive)
    generator = torch.Generator(device="cpu").manual_seed(5054)
    omega = (
        torch
        .randint(0, 2, (rows, columns), generator=generator, dtype=torch.int64)
        .mul(2)
        .sub(1)
        .to(torch.float64)
    )
    observed_residual = residual_ratio(omega)
    gram = matrix @ matrix.T
    probe_count = int(numerics["randomized_trace_samples"])
    probes = (
        torch
        .randint(0, 2, (probe_count, rows), generator=generator, dtype=torch.int64)
        .mul(2)
        .sub(1)
        .to(torch.float64)
    )
    trace_estimate = torch.mean(torch.sum((probes @ gram) * probes, dim=1))
    exact_trace = torch.trace(gram)
    all_probe_values = [
        float(probe @ gram @ probe)
        for probe in (
            torch.tensor(signs, dtype=torch.float64)
            for signs in itertools.product((-1.0, 1.0), repeat=rows)
        )
    ]
    value_range = max(all_probe_values) - min(all_probe_values)
    confidence = float(numerics["randomized_certificate_confidence"])
    hoeffding_bound = value_range * math.sqrt(
        math.log(2 / (1 - confidence)) / (2 * probe_count)
    )
    trace_error = abs(float(trace_estimate - exact_trace))
    return {
        "residual_passed": bool(
            coverage >= confidence and observed_residual <= residual_limit
        ),
        "trace_passed": bool(
            hoeffding_bound <= float(numerics["randomized_trace_absolute_99_max"])
            and trace_error <= hoeffding_bound
        ),
        "sketch_dimensions": [rows, columns],
        "exhaustive_sketches": len(exhaustive),
        "exact_svd_tail": bool(exact_tail > 0),
        "coverage_at_least_99": bool(coverage >= confidence),
        "trace_hoeffding_99": bool(trace_error <= hoeffding_bound),
    }


def _close(value, expected, *, relative, zero, torch) -> bool:
    difference = abs(
        float(value.detach().to(device="cpu", dtype=torch.float64) - expected)
    )
    return (
        difference <= zero if expected == 0 else difference / abs(expected) <= relative
    )


def _work_id_fixture() -> bool:
    def digest(parts):
        encoded = b"".join(
            len(part.encode("utf-8")).to_bytes(8, "big") + part.encode("utf-8")
            for part in parts
        )
        return hashlib.sha256(encoded).hexdigest()

    return digest(("ab", "c")) != digest(("a", "bc"))


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
        raise RuntimeError(f"expected one validation binary, found {candidates}")
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
        raise RuntimeError(f"expected one frozen weight bundle, found {candidates}")
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
        model = build_model(kind)
        model.load_state_dict(state, strict=True)
        model = model.to("cuda:0").eval().requires_grad_(False)
        hashes[id(model)] = state_dict_sha256(model.state_dict())
        models.append(model)
    return models, hashes


def _anonymous_order(models):
    return (
        tuple(reversed(models))
        if secrets.SystemRandom().randrange(2)
        else tuple(models)
    )


def _measure_branch(model, sample, contract, label, *, continuous_rotate, torch):
    c4_angles = contract["tensor_contract"]["c4_angles_degrees"]
    padded_angles = contract["tensor_contract"]["padded_angles_degrees"]

    def c4_workload():
        with torch.inference_mode():
            c4 = torch.cat([
                torch.rot90(sample, turns // 90, dims=(-2, -1)) for turns in c4_angles
            ])
            mu, logvar = model.encode(c4)
            return mu, logvar, model.decode(mu)

    _emit_progress("workload_started", branch=label, workload="c4")
    (mu, logvar, decoded), c4_profile = _profile(
        "c4", c4_workload, device=sample.device, torch=torch
    )
    _emit_progress("workload_complete", branch=label, workload="c4", profile=c4_profile)

    def padded_workload():
        with torch.inference_mode():
            padded = torch.nn.functional.pad(sample, (64, 64, 64, 64))
            continuous = torch.cat([
                continuous_rotate(padded, float(angle)) for angle in padded_angles
            ])
            padded_mu, padded_logvar = model.encode(continuous)
            return padded_mu, padded_logvar, model.decode(padded_mu)

    _emit_progress("workload_started", branch=label, workload="padded_continuous")
    (padded_mu, padded_logvar, padded_decoded), padded_profile = _profile(
        "padded_continuous", padded_workload, device=sample.device, torch=torch
    )
    _emit_progress(
        "workload_complete",
        branch=label,
        workload="padded_continuous",
        profile=padded_profile,
    )

    def path_workload():
        z0 = mu[:1].detach()
        z1 = torch.rot90(z0, 1, dims=(-2, -1))
        knots = torch.linspace(
            0,
            1,
            contract["tensor_contract"]["path_knots"],
            device=z0.device,
            dtype=z0.dtype,
        ).view(-1, 1, 1, 1)
        path = (z0 + knots * (z1 - z0)).detach().requires_grad_(True)
        with torch.enable_grad():
            path_decoded = model.decode(path)
            path_gradient = torch.autograd.grad(path_decoded.square().mean(), path)[0]
        return path, path_decoded, path_gradient

    _emit_progress("workload_started", branch=label, workload="differentiable_path")
    (_path, path_decoded, path_gradient), path_profile = _profile(
        "differentiable_path", path_workload, device=sample.device, torch=torch
    )
    _emit_progress(
        "workload_complete",
        branch=label,
        workload="differentiable_path",
        profile=path_profile,
    )
    compatibility = {
        "c4_mu_shape": list(mu.shape),
        "c4_logvar_shape": list(logvar.shape),
        "c4_decode_shape": list(decoded.shape),
        "padded_mu_shape": list(padded_mu.shape),
        "padded_logvar_shape": list(padded_logvar.shape),
        "padded_decode_shape": list(padded_decoded.shape),
        "path_decode_shape": list(path_decoded.shape),
        "path_gradient_shape": list(path_gradient.shape),
        "all_finite": bool(
            all(
                torch.isfinite(value).all()
                for value in (
                    mu,
                    logvar,
                    decoded,
                    padded_mu,
                    padded_logvar,
                    padded_decoded,
                    path_decoded,
                    path_gradient,
                )
            )
        ),
    }
    expected = {
        "c4_mu_shape": [4, 16, 32, 32],
        "c4_logvar_shape": [4, 16, 32, 32],
        "c4_decode_shape": [4, 3, 256, 256],
        "padded_mu_shape": [36, 16, 48, 48],
        "padded_logvar_shape": [36, 16, 48, 48],
        "padded_decode_shape": [36, 3, 384, 384],
        "path_decode_shape": [17, 3, 256, 256],
        "path_gradient_shape": [17, 16, 32, 32],
    }
    if not compatibility["all_finite"] or any(
        compatibility[key] != value for key, value in expected.items()
    ):
        raise RuntimeError("native/padded/path compatibility differs")
    z = mu[:1].detach().clone().requires_grad_(True)
    direction = _rademacher_direction(
        z,
        seed=int(contract["numerics"]["tangent_seed"]),
        torch=torch,
    )
    _emit_progress("workload_started", branch=label, workload="eager_jvp_vjp_hvp")
    eager, autodiff_profile = _profile(
        "eager_jvp_vjp_hvp",
        lambda: _eager_autodiff_set(
            model, z, contract["numerics"], label=label, torch=torch
        ),
        device=sample.device,
        torch=torch,
    )
    _emit_progress(
        "workload_complete",
        branch=label,
        workload="eager_jvp_vjp_hvp",
        profile=autodiff_profile,
    )
    _emit_progress(
        "eager_ad_measured",
        branch=label,
        residuals=eager["summary"],
        diagnostic_residuals=eager["diagnostic_summary"],
        limits=_eager_limits(contract["numerics"]),
        failed_checks=eager["failed_checks"],
    )
    _assert_eager_tolerances(eager, contract["numerics"])
    _emit_progress("compilation_started", branch=label)
    _emit_progress(
        "workload_started", branch=label, workload="compiled_jvp_hvp_closure"
    )
    compiled, compile_profile = _profile(
        "compiled_jvp_hvp_closure",
        lambda: _compile_closure(
            model,
            z.detach(),
            direction.detach(),
            eager["representative"],
            contract["numerics"],
            torch=torch,
        ),
        device=sample.device,
        torch=torch,
    )
    _emit_progress(
        "workload_complete",
        branch=label,
        workload="compiled_jvp_hvp_closure",
        profile=compile_profile,
    )
    _emit_progress(
        "compilation_complete",
        branch=label,
        mode=compiled["mode"],
        passed=compiled["passed"],
    )
    return {
        "compatibility": compatibility,
        "autodiff": _eager_summary(eager) | {"compiled": compiled},
        "workloads": {
            "c4": c4_profile,
            "padded_continuous": padded_profile,
            "differentiable_path": path_profile,
            "eager_jvp_vjp_hvp": autodiff_profile,
            "compiled_jvp_hvp_closure": compile_profile,
        },
        "direction_ladder": _direction_ladder(
            model, z.detach(), contract, label=label, torch=torch
        ),
    }


def _profile(name, operation, *, device, torch):
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    free_before, total = torch.cuda.mem_get_info(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    result = operation()
    torch.cuda.synchronize(device)
    free_after, _ = torch.cuda.mem_get_info(device)
    return result, {
        "workload": name,
        "seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "free_fraction_before": free_before / total,
        "free_fraction_after": free_after / total,
    }


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


def _eager_autodiff_single(
    model, z, direction, cotangent, limits, *, torch, jvp_epsilon=None, hvp_epsilon=None
):
    def decode(values):
        return model.decode(values)

    epsilon = float(limits["jvp_epsilon"] if jvp_epsilon is None else jvp_epsilon)
    with torch.enable_grad():
        base, jvp = torch.autograd.functional.jvp(decode, z, direction, strict=True)
        finite = (decode(z + epsilon * direction) - decode(z - epsilon * direction)) / (
            2 * epsilon
        )
        _, vjp = torch.autograd.functional.vjp(decode, z, cotangent, strict=True)
        energy = decode(z).square().mean()
        gradient = torch.autograd.grad(energy, z, create_graph=True)[0]
        hvp = torch.autograd.grad((gradient * direction).sum(), z)[0]
        hp_epsilon = float(
            limits["hvp_epsilon"] if hvp_epsilon is None else hvp_epsilon
        )
        plus = (z + hp_epsilon * direction).detach().requires_grad_(True)
        minus = (z - hp_epsilon * direction).detach().requires_grad_(True)
        grad_plus = torch.autograd.grad(decode(plus).square().mean(), plus)[0]
        grad_minus = torch.autograd.grad(decode(minus).square().mean(), minus)[0]
    jvp_error, jvp_numerator, jvp_denominator = _symmetric_l2(jvp, finite, torch=torch)
    hvp_error, hvp_numerator, hvp_denominator = _symmetric_l2(
        hvp, (grad_plus - grad_minus) / (2 * hp_epsilon), torch=torch
    )
    repeat_error, repeat_numerator, repeat_denominator = _symmetric_l2(
        base, decode(z).detach(), torch=torch
    )
    vjp_error, vjp_numerator, vjp_denominator = _adjoint_error(
        jvp, cotangent, direction, vjp, torch=torch
    )
    return {
        "forward": base.detach(),
        "jvp": jvp.detach(),
        "hvp": hvp.detach(),
        "details": {
            "jvp_fd": (jvp_error, jvp_numerator, jvp_denominator),
            "vjp_adjoint": (vjp_error, vjp_numerator, vjp_denominator),
            "hvp_fd": (hvp_error, hvp_numerator, hvp_denominator),
            "repeat": (repeat_error, repeat_numerator, repeat_denominator),
        },
    }


def _eager_limits(limits):
    return {
        "jvp_fd": limits["jvp_finite_difference_relative_l2_max"],
        "vjp_adjoint": limits["vjp_adjoint_relative_scalar_max"],
        "hvp_fd": limits["hvp_finite_difference_relative_l2_max"],
        "repeat": limits["fp32_repeat_relative_l2_max"],
    }


def _summary_from_details(values):
    def quantiles(rows):
        ordered = sorted(item[0] for item in rows)
        middle = len(ordered) // 2
        return (ordered[middle - 1] + ordered[middle]) / 2, ordered[-1]

    return {
        key: {
            "aggregate": math.sqrt(sum(item[1] ** 2 for item in rows))
            / max(math.sqrt(sum(item[2] ** 2 for item in rows)), 1e-12),
            "worst": max(item[0] for item in rows),
            "p50": quantiles(rows)[0],
            "p95": quantiles(rows)[1],
        }
        for key, rows in values.items()
    }


def _eager_autodiff_set(model, z, limits, *, label, torch):
    values = {key: [] for key in _eager_limits(limits)}
    diagnostic_values = {
        _diagnostic_key(epsilon): {key: [] for key in _eager_limits(limits)}
        for epsilon in limits["jvp_diagnostic_epsilons"]
    }
    representative = None
    for index in range(int(limits["tangent_count"])):
        direction = _rademacher_direction(
            z, seed=int(limits["tangent_seed"]) + index, torch=torch
        )
        cotangent = _rademacher_direction(
            model.decode(z).detach(),
            seed=int(limits["cotangent_seed"]) + index,
            torch=torch,
        )
        result = _eager_autodiff_single(
            model, z, direction, cotangent, limits, torch=torch
        )
        if representative is None:
            representative = result
        per_direction = {key: result["details"][key][0] for key in values}
        for key, value in result["details"].items():
            values[key].append(value)
        _emit_progress(
            "eager_direction_measured",
            branch=label,
            direction_index=index,
            epsilon={"jvp": limits["jvp_epsilon"], "hvp": limits["hvp_epsilon"]},
            residuals=per_direction,
            limits=_eager_limits(limits),
            failed_checks=[
                key
                for key, value in per_direction.items()
                if value > _eager_limits(limits)[key]
            ],
        )
        for diagnostic_epsilon in limits["jvp_diagnostic_epsilons"]:
            diagnostic = _eager_autodiff_single(
                model,
                z,
                direction,
                cotangent,
                limits,
                torch=torch,
                jvp_epsilon=diagnostic_epsilon,
                hvp_epsilon=limits["diagnostic_hvp_epsilon"],
            )
            key = _diagnostic_key(diagnostic_epsilon)
            diagnostic_residuals = {
                name: diagnostic["details"][name][0] for name in diagnostic_values[key]
            }
            for name, value in diagnostic["details"].items():
                diagnostic_values[key][name].append(value)
            _emit_progress(
                "eager_direction_measured",
                branch=label,
                direction_index=index,
                diagnostic_only=True,
                epsilon={
                    "jvp": diagnostic_epsilon,
                    "hvp": limits["diagnostic_hvp_epsilon"],
                },
                residuals=diagnostic_residuals,
                limits={},
                failed_checks=[],
            )
    summary = _summary_from_details(values)
    failed_checks = [
        key
        for key, item in summary.items()
        if item["aggregate"] > _eager_limits(limits)[key]
        or item["worst"] > _eager_limits(limits)[key]
    ]
    if representative is None:
        raise RuntimeError("no eager directions")
    return {
        "representative": representative,
        "summary": summary,
        "diagnostic_summary": {
            key: _summary_from_details(items)
            for key, items in diagnostic_values.items()
        },
        "failed_checks": failed_checks,
    }


def _diagnostic_key(epsilon):
    labels = {0.004: "jvp_epsilon_0_004", 0.008: "jvp_epsilon_0_008"}
    try:
        return labels[float(epsilon)]
    except KeyError as error:
        raise RuntimeError("unexpected diagnostic JVP epsilon") from error


def _assert_eager_tolerances(eager, limits) -> None:
    if eager["failed_checks"]:
        raise RuntimeError("eager autodiff tolerance failed")


def _eager_summary(eager):
    return {
        "passed": True,
        "directions": 8,
        "residuals": eager["summary"],
        "diagnostic_residuals": eager["diagnostic_summary"],
        "accumulation": "detached_cpu_fp64",
    }


def _compile_closure(model, z, direction, eager, limits, *, torch):
    try:  # noqa: PLW0717

        def decode(values):
            return model.decode(values)

        def jvp(values, tangent):
            return torch.func.jvp(decode, (values,), (tangent,))[1]

        def energy(values):
            return decode(values).square().mean()

        def hvp(values, tangent):
            return torch.func.jvp(torch.func.grad(energy), (values,), (tangent,))[1]

        started = time.perf_counter()
        compiled_forward = torch.compile(
            decode, backend="inductor", fullgraph=True, dynamic=False
        )
        compiled_jvp = torch.compile(
            jvp, backend="inductor", fullgraph=True, dynamic=False
        )
        compiled_hvp = torch.compile(
            hvp, backend="inductor", fullgraph=True, dynamic=False
        )
        forward_value = compiled_forward(z)
        jvp_value = compiled_jvp(z, direction)
        hvp_value = compiled_hvp(z, direction)
        errors = {
            "forward": _relative_l2(forward_value, eager["forward"], torch=torch),
            "jvp": _relative_l2(jvp_value, eager["jvp"], torch=torch),
            "hvp": _relative_l2(hvp_value, eager["hvp"], torch=torch),
        }
        passed = bool(
            all(
                value <= limits["compiled_relative_l2_max"] for value in errors.values()
            )
        )
        return {
            "mode": "compiled" if passed else "eager_only",
            "passed": passed,
            "backend": "inductor",
            "fullgraph": True,
            "dynamic": False,
            "warmup_seconds": time.perf_counter() - started,
            "closure": errors,
        }
    except Exception as error:
        return {
            "mode": "eager_only",
            "passed": False,
            "backend": "inductor",
            "fullgraph": True,
            "dynamic": False,
            "failure_class": _controlled_exception_class(error),
        }


def _direction_ladder(model, z, contract, *, label, torch):
    rows = []
    minimum = float(contract["numerics"]["minimum_free_vram_fraction"])
    previous_consumed = 0
    for count in contract["tensor_contract"]["direction_microbatches"]:
        _emit_progress("direction_ladder_started", branch=label, directions=count)
        torch.cuda.empty_cache()
        torch.cuda.synchronize(z.device)
        free_before, total = torch.cuda.mem_get_info(z.device)
        predicted_consumed = max(previous_consumed * 2, total // 4 if not rows else 0)
        if (free_before - predicted_consumed) / total < minimum:
            _emit_progress(
                "direction_ladder_skipped",
                branch=label,
                directions=count,
                reason="predicted_vram_headroom",
            )
            break

        def jvp_microbatch(direction_count=count):
            generator = torch.Generator(device=z.device).manual_seed(
                5053 + direction_count
            )
            directions = torch.randn(
                (direction_count, *z.shape[1:]),
                generator=generator,
                device=z.device,
                dtype=z.dtype,
            )
            directions /= (
                torch.linalg.vector_norm(directions.flatten(1), dim=1).view(
                    direction_count, 1, 1, 1
                )
                + 1e-12
            )
            primals = z.detach().expand(direction_count, -1, -1, -1).clone()
            _, tangents = torch.func.jvp(model.decode, (primals,), (directions,))
            if direction_count > 1 and torch.equal(directions[0], directions[1]):
                raise RuntimeError("direction microbatch is not distinct")
            if not torch.isfinite(tangents).all():
                raise RuntimeError("direction microbatch produced non-finite values")
            return tangents

        _, profile = _profile(
            "jvp_direction_microbatch", jvp_microbatch, device=z.device, torch=torch
        )
        _emit_progress(
            "direction_ladder_complete",
            branch=label,
            directions=count,
            profile=profile,
        )
        if (
            profile["free_fraction_before"] < minimum
            or profile["free_fraction_after"] < minimum
        ):
            raise RuntimeError("direction microbatch violated fixed VRAM headroom")
        consumed = max(
            0,
            int(
                (profile["free_fraction_before"] - profile["free_fraction_after"])
                * total
            ),
        )
        previous_consumed = max(previous_consumed, consumed)
        rows.append({"directions": count} | profile)
    if not rows:
        raise RuntimeError("no direction microbatch satisfied predicted VRAM headroom")
    return rows


def _symmetric_l2(left, right, *, torch):
    dtype = (
        torch.complex128 if left.is_complex() or right.is_complex() else torch.float64
    )
    left_cpu = left.detach().to(device="cpu", dtype=dtype)
    right_cpu = right.detach().to(device="cpu", dtype=dtype)
    numerator = float(torch.linalg.vector_norm(left_cpu - right_cpu))
    denominator = max(
        float(torch.linalg.vector_norm(left_cpu)),
        float(torch.linalg.vector_norm(right_cpu)),
        1e-12,
    )
    return numerator / denominator, numerator, denominator


def _relative_l2(left, right, *, torch):
    return _symmetric_l2(left, right, torch=torch)[0]


def _adjoint_error(jvp, cotangent, direction, vjp, *, torch):
    jvp_cpu = jvp.detach().to(device="cpu", dtype=torch.float64)
    cotangent_cpu = cotangent.detach().to(device="cpu", dtype=torch.float64)
    direction_cpu = direction.detach().to(device="cpu", dtype=torch.float64)
    vjp_cpu = vjp.detach().to(device="cpu", dtype=torch.float64)
    left = torch.sum(jvp_cpu * cotangent_cpu)
    right = torch.sum(direction_cpu * vjp_cpu)
    jvp_norm = float(torch.linalg.vector_norm(jvp_cpu))
    cotangent_norm = float(torch.linalg.vector_norm(cotangent_cpu))
    direction_norm = float(torch.linalg.vector_norm(direction_cpu))
    vjp_norm = float(torch.linalg.vector_norm(vjp_cpu))
    numerator = abs(float(left - right))
    denominator = max(jvp_norm * cotangent_norm, direction_norm * vjp_norm, 1e-12)
    return numerator / denominator, numerator, denominator


def _runtime_identity(device_name, *, torch, np):
    return {
        "device": device_name,
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda": torch.version.cuda,
        "deterministic_algorithms": True,
        "tf32": False,
        "precision": "fp32_forward_autodiff_cpu_fp64_detached_accumulation",
    }


def _write_parent_output(*, contract, fixture, branches, runtime):
    temporary = OUTPUT_ROOT.with_name(f".{OUTPUT_ROOT.name}.tmp")
    temporary.mkdir(parents=True, exist_ok=False)
    work = contract["work_units"]
    binding = {
        "schema": "spec0057.preflight_parent_binding.v1",
        "contract_sha256": CONTRACT_SHA256,
        "spec_sha256": SPEC_SHA256,
        "selector_rank": contract["inputs"]["selector_rank"],
        "work_unit_ids": [
            work["fixtures_and_telemetry"],
            work["authenticated_resume_fixture"],
        ],
    }
    _atomic_json(temporary / "binding.json", binding)
    _atomic_json(temporary / "runtime.json", runtime)
    _atomic_json(temporary / "fixtures.json", fixture)
    _atomic_json(
        temporary / "metrics_partial.json",
        {"schema": "spec0057.preflight_blind_telemetry.v1", "branches": branches},
    )
    ledger = (
        "\n".join(
            json.dumps(row, sort_keys=True)
            for row in (
                {"work_id": work["fixtures_and_telemetry"], "status": "complete"},
                {"work_id": work["authenticated_resume_fixture"], "status": "pending"},
            )
        )
        + "\n"
    )
    _atomic_text(temporary / "work_units.jsonl", ledger)
    _atomic_json(
        temporary / "status.json",
        {
            "status": "partial",
            "pending_work_ids": [work["authenticated_resume_fixture"]],
        },
    )
    manifest = {
        "schema": "spec0057.preflight_parent_manifest.v1",
        "files": {
            str(path.relative_to(temporary)): _sha256(path)
            for path in sorted(temporary.rglob("*"))
            if path.is_file()
        },
    }
    _atomic_json(temporary / "manifest.json", manifest)
    for path in temporary.glob("*.json"):
        _assert_blind_document(json.loads(path.read_text(encoding="utf-8")))
    temporary.replace(OUTPUT_ROOT)
    _fsync_directory(OUTPUT_ROOT.parent)


def _set_stage(stage):
    global ACTIVE_STAGE  # noqa: PLW0603
    ACTIVE_STAGE = stage
    _emit_progress("stage_started")


def _emit_progress(event, **fields: object):
    global PROGRESS_SEQUENCE  # noqa: PLW0603
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
    allowed_fieldsets = _PROGRESS_EVENT_FIELDSETS.get(event)
    allowed_stages = _PROGRESS_EVENT_STAGES.get(event)
    if allowed_fieldsets is None or allowed_stages is None:
        raise RuntimeError("unrecognized public progress event")
    if ACTIVE_STAGE not in allowed_stages:
        raise RuntimeError("public progress event emitted from an invalid stage")
    if frozenset(fields) not in allowed_fieldsets:
        raise RuntimeError("public progress event fields differ from the contract")
    if "branch" in fields and fields["branch"] not in OUTPUT_LABELS:
        raise RuntimeError("public progress branch differs")
    if "device" in fields and fields["device"] != "Tesla T4":
        raise RuntimeError("public progress device differs")
    if "shape" in fields and fields["shape"] != [1, 3, 256, 256]:
        raise RuntimeError("public progress shape differs")
    if "dtype" in fields and fields["dtype"] != "torch.float32":
        raise RuntimeError("public progress dtype differs")
    if "passed" in fields and type(fields["passed"]) is not bool:
        raise RuntimeError("public progress passed flag differs")
    if "workload" in fields and fields["workload"] not in _PROGRESS_WORKLOADS:
        raise RuntimeError("public progress workload differs")
    if "directions" in fields and (
        type(fields["directions"]) is not int
        or fields["directions"] not in {1, 2, 4, 8, 16, 32}
    ):
        raise RuntimeError("public progress direction count differs")
    if "direction_index" in fields and (
        type(fields["direction_index"]) is not int
        or fields["direction_index"] not in range(8)
    ):
        raise RuntimeError("public progress direction index differs")
    if "epsilon" in fields:
        epsilon = fields["epsilon"]
        if (
            not isinstance(epsilon, dict)
            or set(epsilon) != {"jvp", "hvp"}
            or not all(_is_nonnegative_finite(value) for value in epsilon.values())
        ):
            raise RuntimeError("public progress epsilon differs")
        epsilon_pair = (float(epsilon["jvp"]), float(epsilon["hvp"]))
        if "diagnostic_only" in fields:
            if fields["diagnostic_only"] is not True or epsilon_pair not in {
                (0.004, 0.002),
                (0.008, 0.002),
            }:
                raise RuntimeError("public diagnostic progress epsilon differs")
        elif epsilon_pair != (0.001, 0.002):
            raise RuntimeError("public primary progress epsilon differs")
    if "residuals" in fields:
        _validate_residuals(fields["residuals"])
    if "diagnostic_residuals" in fields:
        _validate_diagnostic_residuals(fields["diagnostic_residuals"])
    if "limits" in fields:
        limits = fields["limits"]
        if limits != {} and (
            not isinstance(limits, dict)
            or set(limits) != _PROGRESS_RESIDUALS
            or not all(_is_nonnegative_finite(value) for value in limits.values())
        ):
            raise RuntimeError("public progress limits differ")
    if "failed_checks" in fields and (
        not isinstance(fields["failed_checks"], list)
        or not set(fields["failed_checks"]) <= _PROGRESS_RESIDUALS
        or not all(type(value) is str for value in fields["failed_checks"])
        or len(set(fields["failed_checks"])) != len(fields["failed_checks"])
    ):
        raise RuntimeError("public progress failed checks differ")
    if "profile" in fields:
        _validate_profile(fields["profile"], fields.get("workload"))
    if "mode" in fields and fields["mode"] not in {"compiled", "eager_only"}:
        raise RuntimeError("public progress compilation mode differs")
    if "reason" in fields and fields["reason"] != "predicted_vram_headroom":
        raise RuntimeError("public progress ladder reason differs")
    if (
        "failure_code" in fields
        and fields["failure_code"] != "controlled_preflight_failure"
    ):
        raise RuntimeError("public progress failure code differs")
    if "exception_class" in fields and fields["exception_class"] not in {
        "AssertionError",
        "FileNotFoundError",
        "KeyError",
        "RuntimeError",
        "ValueError",
        "UnhandledError",
    }:
        raise RuntimeError("public progress exception class differs")
    if "gpu_counters" in fields:
        _validate_gpu_counters(fields["gpu_counters"])


def _validate_residuals(value):
    if not isinstance(value, dict) or set(value) != _PROGRESS_RESIDUALS:
        raise RuntimeError("public progress residual names differ")
    for item in value.values():
        if _is_nonnegative_finite(item):
            continue
        if not (
            isinstance(item, dict)
            and set(item) == {"aggregate", "worst", "p50", "p95"}
            and all(_is_nonnegative_finite(number) for number in item.values())
        ):
            raise RuntimeError("public progress residual values differ")


def _validate_diagnostic_residuals(value):
    if not isinstance(value, dict) or set(value) != {
        "jvp_epsilon_0_004",
        "jvp_epsilon_0_008",
    }:
        raise RuntimeError("public progress diagnostic residual names differ")
    for summary in value.values():
        _validate_residuals(summary)


def _validate_profile(value, workload):
    required = {
        "workload",
        "seconds",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "free_fraction_before",
        "free_fraction_after",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise RuntimeError("public progress profile differs")
    if (
        value["workload"] not in _PROGRESS_WORKLOADS
        or (workload is not None and value["workload"] != workload)
        or not _is_nonnegative_finite(value["seconds"])
    ):
        raise RuntimeError("public progress profile differs")
    if any(
        type(value[key]) is not int or value[key] < 0
        for key in ("peak_allocated_bytes", "peak_reserved_bytes")
    ):
        raise RuntimeError("public progress profile differs")
    if any(
        not _is_nonnegative_finite(value[key]) or value[key] > 1.0
        for key in ("free_fraction_before", "free_fraction_after")
    ):
        raise RuntimeError("public progress profile differs")


def _validate_gpu_counters(value):
    allowed = (
        {"available": False},
        {"available": True, "counters": "unavailable"},
    )
    if value in allowed:
        return
    if not (
        isinstance(value, dict)
        and set(value) == {"available", "allocated_bytes", "reserved_bytes"}
        and value["available"] is True
        and type(value["allocated_bytes"]) is int
        and value["allocated_bytes"] >= 0
        and type(value["reserved_bytes"]) is int
        and value["reserved_bytes"] >= 0
    ):
        raise RuntimeError("public progress GPU counters differ")


def _is_nonnegative_finite(value):
    return (
        type(value) in {int, float}
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _controlled_exception_class(error):
    known = {
        "AssertionError",
        "FileNotFoundError",
        "KeyError",
        "RuntimeError",
        "ValueError",
    }
    name = type(error).__name__
    return name if name in known else "UnhandledError"


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
    for forbidden in FORBIDDEN_PUBLIC_TOKENS:
        if forbidden in encoded:
            raise RuntimeError(
                "public preflight output contains a forbidden identity or comparison token"
            )


def _atomic_json(path, value):
    _atomic_text(
        path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def _atomic_text(path, payload):
    pending = path.with_suffix(path.suffix + ".tmp")
    with pending.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    pending.replace(path)
    _fsync_directory(path.parent)


def _fsync_directory(path):
    directory = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _extract_payload(destination):
    payload = base64.b64decode(EMBEDDED_PAYLOAD_B64.encode("ascii"))
    if hashlib.sha256(payload).hexdigest() != EMBEDDED_PAYLOAD_ZIP_SHA256:
        raise RuntimeError("embedded payload zip hash mismatch")
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name in archive.namelist():
            candidate = Path(name)
            if candidate.is_absolute() or ".." in candidate.parts:
                raise RuntimeError(f"unsafe embedded payload path: {name}")
        archive.extractall(destination)
    manifest_path = destination / "payload_manifest.json"
    if _sha256(manifest_path) != EMBEDDED_PAYLOAD_MANIFEST_SHA256:
        raise RuntimeError("embedded payload manifest hash mismatch")
    return destination, json.loads(manifest_path.read_text(encoding="utf-8"))


def _require_hash(path, expected):
    if _sha256(path) != expected:
        raise RuntimeError(f"hash differs for {path}")


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
