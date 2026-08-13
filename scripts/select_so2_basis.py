# Copyright 2026 HiperMaximus
"""Run Spec 0012's one-off bounded SO(2) basis-oracle selection."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import subprocess  # noqa: S404
import sys
import types
import warnings
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

import numpy as np
import torch
from numpy.typing import NDArray
from scipy import linalg
from scipy.ndimage import gaussian_filter
from scipy.optimize import Bounds, LinearConstraint, minimize
from scipy.special import logsumexp
from torch.nn import functional

from eqvae.models.so2_basis import (
    PAIR_FREQUENCIES,
    generalized_he_standard_deviation,
    irrep_dimension,
    orthonormalize_columns,
    pair_angular_orders,
    sample_pair_basis,
    stored_pair_basis,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject = dict[str, JsonValue]

REPO_ROOT: Final = Path(__file__).resolve().parents[1]
MANIFEST_PATH: Final = REPO_ROOT / "configs/spec0012/so2_basis_manifest.json"
AUDIT_PATH: Final = REPO_ROOT / "docs/data/spec0012_so2_basis_audit.json"
ESCNN_ROOT: Final = REPO_ROOT / "reference/escnn"
ESCNN_COMMIT: Final = "9ad44cc37d694d9c805c4fef4d16722a254b3bf2"
ESCNN_SOURCE_URL: Final = "https://github.com/QUVA-Lab/escnn"
PROFILE_ORDER: Final = ("7-low", "7-full", "9-low", "9-full")
PROFILE_ORDERS: Final = {
    "7-low": frozenset({0, 1, 2}),
    "7-full": frozenset({0, 1, 2, 3, 4}),
    "9-low": frozenset({0, 1, 2}),
    "9-full": frozenset({0, 1, 2, 3, 4}),
}
GRID_BY_SUPPORT: Final = {
    7: (1.0, math.sqrt(2.0), 2.0, math.sqrt(5.0), math.sqrt(8.0), 3.0),
    9: (
        1.0,
        math.sqrt(2.0),
        2.0,
        math.sqrt(5.0),
        math.sqrt(8.0),
        3.0,
        math.sqrt(10.0),
        math.sqrt(13.0),
        4.0,
    ),
}
SHELL_COUNT: Final = {7: 3, 9: 4}
COARSE_WIDTHS: Final = (0.4, 0.5, 0.6, 0.7)
ANGLES_DEGREES: Final = (15, 30, 45, 60, 90)
PAIR_NAMES: Final = {
    (input_frequency, output_frequency): f"F{input_frequency}->F{output_frequency}"
    for input_frequency, output_frequency in PAIR_FREQUENCIES
}
RANK_TOLERANCE: Final = 1e-10
NORM_TOLERANCE: Final = 1e-10
NOMINAL_KAPPA_LIMIT: Final = 10.0
PERTURBED_KAPPA_LIMIT: Final = 12.0
PARAMETER_CAP: Final = 3_958_435
MINIMUM_SPACING: Final = 0.25
MINIMUM_WIDTH: Final = 0.30
MAXIMUM_WIDTH: Final = 0.90
MINIMUM_SHELL_COVERAGE: Final = 2
FEASIBILITY_TOLERANCE: Final = 1e-10
SPAN_TOLERANCE: Final = 5e-5
CONVOLUTION_TOLERANCE: Final = 1e-4
ORIGIN_PROXY_TOLERANCE: Final = 1e-12
HIGH_DIMENSION_RATIO: Final = 0.75
INITIALIZATION_RATIO_MINIMUM: Final = 0.9
INITIALIZATION_RATIO_MAXIMUM: Final = 1.1
EXPECTED_CONV_COUNT: Final = 43
EXPECTED_NORM_COUNT: Final = 40
EXPECTED_GATE_COUNT: Final = 34


@dataclass(frozen=True)
class PairAudit:
    """Hard numerical audit for one sampled one-copy field pair."""

    dimension: int
    rank: int
    condition_number: float
    minimum_singular_value: float
    maximum_singular_value: float
    minimum_column_norm: float
    full_rank: bool


@dataclass(frozen=True)
class CandidateAudit:
    """Numerical audit of all nine locked field pairs."""

    pairs: dict[str, PairAudit]
    full_rank_pair_count: int
    rank_sum: int
    total_dimension: int
    worst_condition_number: float
    worst_minimum_singular_value: float
    objective: float
    passes: bool


@dataclass(frozen=True)
class Candidate:
    """One radial profile candidate and its hard audit."""

    centres: tuple[float, ...]
    widths: tuple[float, ...]
    qmax: tuple[int, ...]
    audit: CandidateAudit
    origin: str
    solver: JsonObject | None = None


@dataclass(frozen=True)
class FixedProfile:
    """Selected radial values loaded without rerunning the radial oracle."""

    centres: tuple[float, ...]
    widths: tuple[float, ...]
    qmax: tuple[int, ...]


@dataclass(frozen=True)
class FieldSpec:
    """One fixed F0/F1/F2 copy layout used only by the count oracle."""

    n0: int
    n1: int
    n2: int

    @property
    def copies(self) -> tuple[int, int, int]:
        """Copy counts in locked frequency order."""
        return (self.n0, self.n1, self.n2)

    @property
    def channels(self) -> int:
        """Physical packed channel count."""
        return self.n0 + 2 * self.n1 + 2 * self.n2


@dataclass(frozen=True)
class ConvPosition:
    """One of the fixed 43 learned-convolution positions."""

    name: str
    input_spec: FieldSpec
    output_spec: FieldSpec
    output_size: int
    followed_by_norm: bool = field(kw_only=True)
    bias: bool = field(kw_only=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Run, write, or reproduce the one-off basis oracle.

    Returns:
        Process exit status.

    """
    args = _parse_args(argv)
    payload = _refresh_layout_artifacts() if args.refresh_layout else run_oracle()
    manifest = cast("JsonObject", payload["manifest"])
    audit = cast("JsonObject", payload["audit"])
    if args.write:
        _write_json(MANIFEST_PATH, manifest)
        _write_json(AUDIT_PATH, audit)
        sys.stdout.write(f"wrote {MANIFEST_PATH.relative_to(REPO_ROOT)}\n")
        sys.stdout.write(f"wrote {AUDIT_PATH.relative_to(REPO_ROOT)}\n")
        return 0
    if args.check:
        expected_manifest = _read_json(MANIFEST_PATH)
        expected_audit = _read_json(AUDIT_PATH)
        if manifest != expected_manifest or audit != expected_audit:
            sys.stderr.write("Spec 0012 basis artifacts do not reproduce\n")
            return 1
        sys.stdout.write("Spec 0012 basis artifacts reproduce exactly\n")
        return 0
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    sys.stdout.write(rendered + "\n")
    return 0


def run_oracle() -> JsonObject:  # noqa: PLR0914
    """Execute the four-profile search and all locked post-selection gates.

    Returns:
        Manifest and full numerical audit payloads.

    Raises:
        RuntimeError: If the locked radial oracle cannot select an architecture.

    """
    profile_results: dict[str, Candidate | None] = {}
    profile_search: dict[str, JsonObject] = {}
    for profile_name in PROFILE_ORDER:
        selected, search_audit = _search_profile(profile_name)
        profile_results[profile_name] = selected
        profile_search[profile_name] = search_audit

    reference_audits: dict[str, JsonObject] = {}
    for profile_name, candidate in profile_results.items():
        if candidate is not None:
            reference_audits[profile_name] = _escnn_profile_audit(
                profile_name,
                candidate,
            )

    high_order = _high_order_decision(profile_results, reference_audits)
    counts = _architecture_counts(profile_results)
    provisional = _select_provisional_candidate(
        profile_results=profile_results,
        reference_audits=reference_audits,
        high_order=high_order,
        counts=counts,
    )
    initialization = _initialization_audit(
        profile_results=profile_results,
        provisional=provisional,
    )
    if initialization["status"] != "pass":
        provisional = {
            **provisional,
            "status": "blocked",
            "blockers": [
                *cast("list[JsonValue]", provisional["blockers"]),
                "initialization_variance_gate_failed",
            ],
        }

    premise_findings: JsonObject = {
        "status": "fail",
        "findings": [
            {
                "profile": "7-full",
                "finding": "locked_coarse_grid_has_no_legal_seed",
                "continuous_feasible_example": {
                    "centres": [1.0, 2.0, 2.75],
                    "widths": [0.3, 0.3, 0.3],
                    "qmax": [2, 4, 4],
                },
                "decision_impact": (
                    "none: 9-full fails the prerequisite high-order sampling gate, "
                    "so Spec 0012 forbids 7x7 adequacy regardless"
                ),
            },
        ],
    }

    selected_profile_names = cast("list[str]", provisional["reference_profiles"])
    manifest_profiles: JsonObject = {}
    for profile_name in selected_profile_names:
        candidate = profile_results[profile_name]
        if candidate is None:
            message = f"selected profile {profile_name} is missing"
            raise RuntimeError(message)
        manifest_profiles[profile_name] = _candidate_manifest(candidate)
    f01_counts = cast("JsonObject", counts["F01"])
    selected_architecture = (
        _selected_architecture_manifest(f01_counts)
        if provisional["name"] == "F01"
        else None
    )
    manifest: JsonObject = {
        "schema_version": 2,
        "spec": "0012-continuous-so2-vae-architecture",
        "scope": "fixed_equal_copy_f01_training_handoff",
        "escnn_commit": ESCNN_COMMIT,
        "profiles": manifest_profiles,
        "selected_architecture": selected_architecture,
    }
    audit: JsonObject = {
        "schema_version": 1,
        "spec": "0012-continuous-so2-vae-architecture",
        "status": provisional["status"],
        "escnn": {
            "source_url": ESCNN_SOURCE_URL,
            "commit": ESCNN_COMMIT,
            "runtime_dependency": False,
        },
        "search": profile_search,
        "profiles": {
            profile_name: (
                _candidate_audit_payload(candidate)
                if candidate is not None
                else {"status": "fail", "selected": None}
            )
            for profile_name, candidate in profile_results.items()
        },
        "escnn_reference": reference_audits,
        "high_order_gate": high_order,
        "architecture_counts": counts,
        "initialization_variance": initialization,
        "provisional_candidate": provisional,
        "selected_architecture": selected_architecture,
        "locked_premise_findings": premise_findings,
        "scope_guards": {
            "per_layer_search": False,
            "image_training": False,
            "full_vae_implemented": False,
            "dynamic_runtime_options": False,
            "final_convolution_layer_implemented": False,
        },
    }
    return {"manifest": manifest, "audit": audit}


def _refresh_layout_artifacts() -> JsonObject:
    """Refresh only evidence affected by the locked equal-copy F01 layout.

    Returns:
        Updated manifest and audit with radial/reference evidence preserved.

    Raises:
        RuntimeError: If the locked F01 handoff or refreshed initialization fails.

    """
    manifest = copy.deepcopy(_read_json(MANIFEST_PATH))
    audit = copy.deepcopy(_read_json(AUDIT_PATH))
    selected = cast("JsonObject", manifest["selected_architecture"])
    if selected.get("name") != "F01_equal_copy":
        message = "layout refresh requires the already selected passing F01 candidate"
        raise RuntimeError(message)
    provisional: JsonObject = {
        "name": "F01",
        "status": "pass",
        "blockers": [],
        "reference_profiles": ["7-low", "9-low"],
    }

    profiles = _fixed_profiles_from_manifest(manifest)
    f01_count = _count_architecture("F01", _equal_copy_f01_fields(), profiles)
    initialization = _initialization_audit(
        profile_results=profiles,
        provisional=provisional,
    )
    if initialization["status"] != "pass":
        message = "equal-copy initialization variance gate failed"
        raise RuntimeError(message)

    architecture_counts = cast("JsonObject", audit["architecture_counts"])
    architecture_counts["F01"] = f01_count
    audit["initialization_variance"] = initialization
    selected_architecture = _selected_architecture_manifest(f01_count)
    selected_profiles = cast("JsonObject", manifest["profiles"])
    manifest = {
        "schema_version": 2,
        "spec": "0012-continuous-so2-vae-architecture",
        "scope": "fixed_equal_copy_f01_training_handoff",
        "escnn_commit": manifest["escnn_commit"],
        "profiles": {name: selected_profiles[name] for name in ("7-low", "9-low")},
        "selected_architecture": selected_architecture,
    }
    audit["selected_architecture"] = selected_architecture
    return {"manifest": manifest, "audit": audit}


def _fixed_profiles_from_manifest(
    manifest: JsonObject,
) -> dict[str, Candidate | FixedProfile | None]:
    profiles_payload = cast("JsonObject", manifest["profiles"])
    profiles: dict[str, Candidate | FixedProfile | None] = {}
    for profile_name in PROFILE_ORDER:
        payload = profiles_payload.get(profile_name)
        if payload is None:
            profiles[profile_name] = None
            continue
        profile = cast("JsonObject", payload)
        profiles[profile_name] = FixedProfile(
            centres=tuple(cast("list[float]", profile["centres"])),
            widths=tuple(cast("list[float]", profile["widths"])),
            qmax=tuple(cast("list[int]", profile["qmax"])),
        )
    return profiles


def _selected_architecture_manifest(counts: JsonObject) -> JsonObject:
    fields = _equal_copy_f01_fields()
    return {
        "name": "F01_equal_copy",
        "status": "layout_evidence_pass",
        "profile_assignment": {
            "stem": "9-low",
            "all_other_convolutions": "7-low",
        },
        "field_layout": {
            name: {
                "copies": list(fields[name].copies),
                "packed_channels": fields[name].channels,
            }
            for name in ("R", "A", "B", "C", "D", "L")
        },
        "parameter_cap": PARAMETER_CAP,
        "total_learned_parameters": counts["total_learned_parameters"],
    }


def _equal_copy_f01_fields() -> dict[str, FieldSpec]:
    return {
        "R": FieldSpec(3, 0, 0),
        "A": FieldSpec(16, 16, 0),
        "B": FieldSpec(24, 24, 0),
        "C": FieldSpec(32, 32, 0),
        "D": FieldSpec(48, 48, 0),
        "L": FieldSpec(16, 0, 0),
    }


def _search_profile(profile_name: str) -> tuple[Candidate | None, JsonObject]:
    kernel_size = int(profile_name.split("-", maxsplit=1)[0])
    allowed_orders = PROFILE_ORDERS[profile_name]
    starts = list(_coarse_starts(kernel_size, allowed_orders))
    passing: list[Candidate] = []
    ranked: list[Candidate] = []
    for centres, widths, qmax in starts:
        audit = _audit_candidate(
            kernel_size=kernel_size,
            allowed_orders=allowed_orders,
            centres=centres,
            widths=widths,
            qmax=qmax,
        )
        candidate = Candidate(centres, widths, qmax, audit, "coarse")
        ranked.append(candidate)
        if audit.passes:
            passing.append(candidate)

    retained = sorted(ranked, key=_coarse_sort_key)[:16]
    refined: list[Candidate] = []
    solver_statuses: dict[str, int] = {}
    for candidate in retained:
        optimized = _refine_candidate(
            kernel_size=kernel_size,
            allowed_orders=allowed_orders,
            candidate=candidate,
        )
        status = str(cast("JsonValue", cast("JsonObject", optimized.solver)["status"]))
        solver_statuses[status] = solver_statuses.get(status, 0) + 1
        refined.append(optimized)
        if optimized.audit.passes:
            passing.append(optimized)

    selected = _select_candidate(passing)
    if selected is not None:
        rounded = Candidate(
            centres=tuple(round(value, 8) for value in selected.centres),
            widths=tuple(round(value, 8) for value in selected.widths),
            qmax=selected.qmax,
            audit=selected.audit,
            origin=selected.origin,
            solver=selected.solver,
        )
        selected = Candidate(
            centres=rounded.centres,
            widths=rounded.widths,
            qmax=rounded.qmax,
            audit=_audit_candidate(
                kernel_size=kernel_size,
                allowed_orders=allowed_orders,
                centres=rounded.centres,
                widths=rounded.widths,
                qmax=rounded.qmax,
            ),
            origin=rounded.origin,
            solver=rounded.solver,
        )
        if not selected.audit.passes:
            message = f"rounded {profile_name} selection no longer passes"
            raise RuntimeError(message)
    return selected, {
        "status": "pass" if selected is not None else "fail",
        "kernel_size": kernel_size,
        "allowed_orders": sorted(allowed_orders),
        "legal_coarse_start_count": len(starts),
        "passing_coarse_count": sum(candidate.audit.passes for candidate in ranked),
        "retained_refinement_start_count": len(retained),
        "passing_refined_count": sum(candidate.audit.passes for candidate in refined),
        "solver_status_counts": solver_statuses,
        "failure_reason": (
            None
            if selected is not None
            else (
                "locked_coarse_grid_has_no_legal_seed"
                if not starts
                else "no_candidate_passed_rank_condition_gate"
            )
        ),
    }


def _coarse_starts(
    kernel_size: int,
    allowed_orders: frozenset[int],
) -> Iterator[tuple[tuple[float, ...], tuple[float, ...], tuple[int, ...]]]:
    shell_count = SHELL_COUNT[kernel_size]
    radius_limit = (kernel_size - 1) / 2.0 - 0.25
    used_orders = tuple(sorted(allowed_orders - {0}))
    for centres in itertools.combinations(GRID_BY_SUPPORT[kernel_size], shell_count):
        if any(radius > radius_limit for radius in centres):
            continue
        if any(
            right - left < MINIMUM_SPACING
            for left, right in itertools.pairwise(centres)
        ):
            continue
        for inner_width, outer_width in itertools.product(COARSE_WIDTHS, repeat=2):
            widths = (inner_width,) * (shell_count - 1) + (outer_width,)
            for qmax in itertools.combinations_with_replacement(range(5), shell_count):
                if qmax[-1] < max(used_orders):
                    continue
                if any(
                    sum(value >= order for value in qmax) < MINIMUM_SHELL_COVERAGE
                    for order in used_orders
                ):
                    continue
                if any(
                    cutoff > min(4, math.floor(2.0 * radius))
                    for cutoff, radius in zip(qmax, centres, strict=True)
                ):
                    continue
                yield tuple(centres), tuple(widths), tuple(qmax)


def _audit_candidate(  # noqa: PLR0914
    *,
    kernel_size: int,
    allowed_orders: frozenset[int],
    centres: tuple[float, ...],
    widths: tuple[float, ...],
    qmax: tuple[int, ...],
) -> CandidateAudit:
    pairs: dict[str, PairAudit] = {}
    losses: list[float] = []
    for input_frequency, output_frequency in PAIR_FREQUENCIES:
        sampled = sample_pair_basis(
            input_frequency=input_frequency,
            output_frequency=output_frequency,
            kernel_size=kernel_size,
            centres=centres,
            widths=widths,
            qmax=qmax,
            allowed_orders=allowed_orders,
        )
        matrix = sampled.flat_columns()
        norms = np.linalg.norm(matrix, axis=0)
        if float(np.min(norms)) < NORM_TOLERANCE:
            singular_values = np.zeros(matrix.shape[1], dtype=np.float64)
            rank = 0
            condition_number = math.inf
            minimum_singular = 0.0
            maximum_singular = 0.0
            full_rank = False
            loss = math.inf
        else:
            normalized = matrix / norms
            singular_values = np.linalg.svd(normalized, compute_uv=False)
            maximum_singular = float(singular_values[0])
            minimum_singular = float(singular_values[-1])
            rank = int(
                np.count_nonzero(singular_values > RANK_TOLERANCE * maximum_singular),
            )
            full_rank = rank == matrix.shape[1]
            condition_number = (
                maximum_singular / minimum_singular
                if minimum_singular > 0.0
                else math.inf
            )
            sign, log_determinant = np.linalg.slogdet(
                normalized.T @ normalized
                + 1e-8 * np.eye(matrix.shape[1], dtype=np.float64),
            )
            loss = -float(log_determinant) / matrix.shape[1] if sign > 0.0 else math.inf
        pairs[PAIR_NAMES[input_frequency, output_frequency]] = PairAudit(
            dimension=matrix.shape[1],
            rank=rank,
            condition_number=condition_number,
            minimum_singular_value=minimum_singular,
            maximum_singular_value=maximum_singular,
            minimum_column_norm=float(np.min(norms)),
            full_rank=full_rank,
        )
        losses.append(loss)
    finite_conditions = [
        pair.condition_number
        for pair in pairs.values()
        if math.isfinite(pair.condition_number)
    ]
    worst_condition = max(finite_conditions, default=math.inf)
    minimum_singular = min(pair.minimum_singular_value for pair in pairs.values())
    return CandidateAudit(
        pairs=pairs,
        full_rank_pair_count=sum(pair.full_rank for pair in pairs.values()),
        rank_sum=sum(pair.rank for pair in pairs.values()),
        total_dimension=sum(pair.dimension for pair in pairs.values()),
        worst_condition_number=worst_condition,
        worst_minimum_singular_value=minimum_singular,
        objective=float(0.05 * logsumexp(np.asarray(losses) / 0.05)),
        passes=(
            all(pair.full_rank for pair in pairs.values())
            and worst_condition <= NOMINAL_KAPPA_LIMIT
        ),
    )


def _coarse_sort_key(candidate: Candidate) -> tuple[Any, ...]:
    audit = candidate.audit
    return (
        -audit.full_rank_pair_count,
        -audit.rank_sum,
        -audit.total_dimension,
        round(audit.worst_condition_number, 10),
        -round(audit.worst_minimum_singular_value, 10),
        candidate.centres,
        candidate.widths,
        candidate.qmax,
    )


def _refine_candidate(
    *,
    kernel_size: int,
    allowed_orders: frozenset[int],
    candidate: Candidate,
) -> Candidate:
    shell_count = len(candidate.centres)
    upper_radius = (kernel_size - 1) / 2.0 - 0.25
    lower = np.asarray(
        [max(0.25, cutoff / 2.0) for cutoff in candidate.qmax] + [0.30] * shell_count,
        dtype=np.float64,
    )
    upper = np.asarray(
        [upper_radius] * shell_count + [0.90] * shell_count,
        dtype=np.float64,
    )
    constraint_matrix = np.zeros((shell_count - 1, 2 * shell_count), dtype=np.float64)
    for row in range(shell_count - 1):
        constraint_matrix[row, row] = -1.0
        constraint_matrix[row, row + 1] = 1.0
    linear_constraint = LinearConstraint(
        constraint_matrix,
        np.full(shell_count - 1, 0.25),
        np.full(shell_count - 1, np.inf),
    )

    def objective(vector: FloatArray) -> float:
        centres = tuple(float(value) for value in vector[:shell_count])
        widths = tuple(float(value) for value in vector[shell_count:])
        return _audit_candidate(
            kernel_size=kernel_size,
            allowed_orders=allowed_orders,
            centres=centres,
            widths=widths,
            qmax=candidate.qmax,
        ).objective

    initial = np.asarray((*candidate.centres, *candidate.widths), dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = minimize(
            objective,
            initial,
            method="COBYQA",
            bounds=Bounds(lower, upper),
            constraints=(linear_constraint,),
            options={
                "maxfev": 4000,
                "initial_tr_radius": 0.1,
                "final_tr_radius": 1e-7,
                "feasibility_tol": 1e-10,
                "scale": True,
            },
        )
    vector = np.asarray(result.x, dtype=np.float64)
    maximum_violation = _maximum_violation(vector, lower, upper, shell_count)
    final_is_feasible = bool(
        np.isfinite(vector).all() and maximum_violation <= FEASIBILITY_TOLERANCE,
    )
    if not final_is_feasible:
        vector = initial
    centres = tuple(float(value) for value in vector[:shell_count])
    widths = tuple(float(value) for value in vector[shell_count:])
    final_audit = _audit_candidate(
        kernel_size=kernel_size,
        allowed_orders=allowed_orders,
        centres=centres,
        widths=widths,
        qmax=candidate.qmax,
    )
    if _hard_score(final_audit) < _hard_score(candidate.audit):
        centres = candidate.centres
        widths = candidate.widths
        final_audit = candidate.audit
    return Candidate(
        centres=centres,
        widths=widths,
        qmax=candidate.qmax,
        audit=final_audit,
        origin="refined"
        if (centres, widths) != (candidate.centres, candidate.widths)
        else "coarse",
        solver={
            "status": str(result.status),
            "success": bool(result.success),
            "message": str(result.message),
            "function_evaluations": int(result.nfev),
            "maximum_constraint_violation": float(maximum_violation),
            "final_point_used": final_is_feasible,
        },
    )


def _maximum_violation(
    vector: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
    shell_count: int,
) -> float:
    bound_violation = max(
        float(np.max(lower - vector)),
        float(np.max(vector - upper)),
        0.0,
    )
    spacing_violation = max(
        (
            MINIMUM_SPACING - float(vector[index + 1] - vector[index])
            for index in range(shell_count - 1)
        ),
        default=0.0,
    )
    return max(bound_violation, spacing_violation, 0.0)


def _hard_score(audit: CandidateAudit) -> tuple[int, int, int, float, float]:
    return (
        audit.full_rank_pair_count,
        audit.rank_sum,
        audit.total_dimension,
        -round(audit.worst_condition_number, 10),
        round(audit.worst_minimum_singular_value, 10),
    )


def _select_candidate(candidates: Sequence[Candidate]) -> Candidate | None:
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda candidate: (
            -candidate.audit.total_dimension,
            round(candidate.audit.worst_condition_number, 10),
            -round(candidate.audit.worst_minimum_singular_value, 10),
            tuple(round(value, 8) for value in candidate.centres),
            tuple(round(value, 8) for value in candidate.widths),
            candidate.qmax,
        ),
    )


def _escnn_profile_audit(  # noqa: PLR0914
    profile_name: str,
    candidate: Candidate,
) -> JsonObject:
    escnn = _load_escnn()
    kernel_size = int(profile_name.split("-", maxsplit=1)[0])
    allowed_orders = PROFILE_ORDERS[profile_name]
    pair_results: JsonObject = {}
    all_pass = True
    for input_frequency, output_frequency in PAIR_FREQUENCIES:
        ours = sample_pair_basis(
            input_frequency=input_frequency,
            output_frequency=output_frequency,
            kernel_size=kernel_size,
            centres=candidate.centres,
            widths=candidate.widths,
            qmax=candidate.qmax,
            allowed_orders=allowed_orders,
        ).flat_columns()
        reference = _sample_escnn_basis(
            escnn=escnn,
            input_frequency=input_frequency,
            output_frequency=output_frequency,
            kernel_size=kernel_size,
            centres=candidate.centres,
            widths=candidate.widths,
            qmax=candidate.qmax,
            allowed_orders=allowed_orders,
        )
        ours_q = orthonormalize_columns(ours)
        reference_q = orthonormalize_columns(reference)
        span_distance = float(
            np.linalg.norm(
                ours_q @ ours_q.T - reference_q @ reference_q.T,
                ord=2,
            ),
        )
        coefficients = np.linspace(0.25, 1.25, ours_q.shape[1], dtype=np.float64)
        kernel = ours_q @ coefficients
        projected = reference_q @ (reference_q.T @ kernel)
        kernel_residual = float(
            np.linalg.norm(kernel - projected) / max(np.linalg.norm(kernel), 1e-12),
        )
        convolution_residual = _convolution_output_residual(
            kernel=kernel,
            projected=projected,
            input_frequency=input_frequency,
            output_frequency=output_frequency,
            kernel_size=kernel_size,
        )
        equivariance = _pair_equivariance_audit(
            escnn=escnn,
            kernel=kernel,
            projected=projected,
            input_frequency=input_frequency,
            output_frequency=output_frequency,
            kernel_size=kernel_size,
        )
        pair_pass = bool(
            ours.shape[1] == reference.shape[1]
            and span_distance <= SPAN_TOLERANCE
            and kernel_residual <= SPAN_TOLERANCE
            and convolution_residual <= CONVOLUTION_TOLERANCE
            and equivariance["status"] == "pass",
        )
        all_pass &= pair_pass
        pair_results[PAIR_NAMES[input_frequency, output_frequency]] = {
            "status": "pass" if pair_pass else "fail",
            "ours_dimension": ours.shape[1],
            "escnn_dimension": reference.shape[1],
            "span_projector_distance": span_distance,
            "relative_kernel_residual": kernel_residual,
            "fp32_convolution_relative_rms": convolution_residual,
            "equivariance": equivariance,
        }
    return {
        "status": "pass" if all_pass else "fail",
        "pairs": pair_results,
        "origin_proxy_maximum_off_centre": 0.0,
    }


@cache
def _load_escnn() -> Any:  # noqa: ANN401
    if not ESCNN_ROOT.is_dir():
        message = f"missing local escnn checkout at {ESCNN_ROOT}"
        raise RuntimeError(message)
    revision = _read_git_revision(ESCNN_ROOT)
    if revision != ESCNN_COMMIT:
        message = f"escnn checkout is {revision}, expected {ESCNN_COMMIT}"
        raise RuntimeError(message)
    status = subprocess.run(
        ["/usr/bin/git", "status", "--porcelain"],
        cwd=ESCNN_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        message = "escnn checkout is dirty; reference evidence would be mislabeled"
        raise RuntimeError(message)

    class NoCacheMemory:
        """Minimal no-write replacement for escnn's import-time joblib cache."""

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def cache(  # noqa: PLR6301
            self,
            function: Callable[..., Any] | None = None,
            **_kwargs: object,
        ) -> Any:  # noqa: ANN401
            if function is None:
                return lambda wrapped: wrapped
            return function

    joblib = types.ModuleType("joblib")
    joblib.Memory = NoCacheMemory  # type: ignore[attr-defined]
    sys.modules["joblib"] = joblib
    module_names = (
        "lie_learn",
        "lie_learn.representations",
        "lie_learn.representations.SO3",
        "lie_learn.representations.SO3.wigner_d",
    )
    for module_name in module_names:
        sys.modules[module_name] = types.ModuleType(module_name)

    def reject_so3(*_args: object, **_kwargs: object) -> None:
        message = "Spec 0012 SO(2) oracle entered an SO(3) lie_learn path"
        raise RuntimeError(message)

    sys.modules[module_names[-1]].wigner_D_matrix = reject_so3  # type: ignore[attr-defined]
    sys.path.insert(0, str(ESCNN_ROOT))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import escnn  # type: ignore[import-not-found]  # noqa: PLC0415

    return escnn


def _read_git_revision(repository: Path) -> str:
    git_path = repository / ".git"
    if git_path.is_file():
        pointer = git_path.read_text(encoding="utf-8").strip()
        git_path = (repository / pointer.removeprefix("gitdir: ")).resolve()
    head = (git_path / "HEAD").read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head
    reference = head.removeprefix("ref: ")
    loose_reference = git_path / reference
    if loose_reference.is_file():
        return loose_reference.read_text(encoding="utf-8").strip()
    packed = (git_path / "packed-refs").read_text(encoding="utf-8").splitlines()
    for line in packed:
        if not line.startswith(("#", "^")):
            revision, name = line.split(" ", maxsplit=1)
            if name == reference:
                return revision
    message = f"cannot resolve {reference} in {git_path}"
    raise RuntimeError(message)


def _sample_escnn_basis(  # noqa: PLR0913
    *,
    escnn: Any,  # noqa: ANN401
    input_frequency: int,
    output_frequency: int,
    kernel_size: int,
    centres: tuple[float, ...],
    widths: tuple[float, ...],
    qmax: tuple[int, ...],
    allowed_orders: frozenset[int],
) -> FloatArray:
    group = escnn.group.so2_group(4)
    points = torch.from_numpy(_flat_kernel_points(kernel_size))
    centre_index = kernel_size * kernel_size // 2
    required_orders = pair_angular_orders(input_frequency, output_frequency)
    shell_samples: list[FloatArray] = []
    shells = ((0.0, 0.005, 0), *zip(centres, widths, qmax, strict=True))
    for radius, width, cutoff in shells:
        effective_cutoff = min(cutoff, max(allowed_orders))
        if not any(order <= effective_cutoff for order in required_orders):
            continue
        # escnn@9ad44cc misindexes a per-radius filter in
        # CircularShellsBasis.steerable_attrs_j_iter(). Constructing one shell
        # at a time gives the intended basis without patching the reference.
        basis = copy.deepcopy(
            escnn.kernels.kernels_SO2_act_R2(
                group.irrep(input_frequency),
                group.irrep(output_frequency),
                [radius],
                [width],
                maximum_frequency=effective_cutoff,
            ),
        ).to(dtype=torch.float64)
        sampled = basis.sample(points).detach().numpy()
        if math.isclose(radius, 0.0, abs_tol=1e-7):
            off_centre = np.delete(sampled, centre_index, axis=0)
            maximum = float(np.max(np.abs(off_centre), initial=0.0))
            if maximum >= ORIGIN_PROXY_TOLERANCE:
                message = "escnn origin proxy leaks beyond the centre pixel"
                raise RuntimeError(message)
            centre_value = sampled[centre_index].copy()
            sampled[:] = 0.0
            sampled[centre_index] = centre_value
        shell_samples.append(sampled)
    concatenated = np.concatenate(shell_samples, axis=1)
    return concatenated.transpose(2, 3, 0, 1).reshape(
        irrep_dimension(output_frequency)
        * irrep_dimension(input_frequency)
        * kernel_size
        * kernel_size,
        concatenated.shape[1],
    )


def _flat_kernel_points(kernel_size: int) -> FloatArray:
    centre = (kernel_size - 1) / 2.0
    return np.asarray(
        [
            (column - centre, centre - row)
            for row in range(kernel_size)
            for column in range(kernel_size)
        ],
        dtype=np.float64,
    )


def _convolution_output_residual(
    *,
    kernel: FloatArray,
    projected: FloatArray,
    input_frequency: int,
    output_frequency: int,
    kernel_size: int,
) -> float:
    generator = torch.Generator().manual_seed(
        12012 + 10 * input_frequency + output_frequency,
    )
    inputs = torch.randn(
        3,
        irrep_dimension(input_frequency),
        33,
        33,
        generator=generator,
    )
    shape = (
        irrep_dimension(output_frequency),
        irrep_dimension(input_frequency),
        kernel_size,
        kernel_size,
    )
    ours = functional.conv2d(inputs, torch.from_numpy(kernel.reshape(shape)).float())
    reference = functional.conv2d(
        inputs,
        torch.from_numpy(projected.reshape(shape)).float(),
    )
    return float(
        torch.sqrt(torch.mean((ours - reference) ** 2))
        / torch.sqrt(torch.mean(reference**2)).clamp_min(1e-8),
    )


def _pair_equivariance_audit(  # noqa: PLR0913, PLR0914
    *,
    escnn: Any,  # noqa: ANN401
    kernel: FloatArray,
    projected: FloatArray,
    input_frequency: int,
    output_frequency: int,
    kernel_size: int,
) -> JsonObject:
    group_space = escnn.gspaces.rot2dOnR2(N=-1, maximum_frequency=4)
    group = group_space.fibergroup
    input_type = escnn.nn.FieldType(group_space, [group.irrep(input_frequency)])
    output_type = escnn.nn.FieldType(group_space, [group.irrep(output_frequency)])
    inputs = _smooth_input_bank(input_frequency, count=2)
    shape = (
        irrep_dimension(output_frequency),
        irrep_dimension(input_frequency),
        kernel_size,
        kernel_size,
    )
    ours_kernel = torch.from_numpy(kernel.reshape(shape)).to(dtype=torch.float64)
    reference_kernel = torch.from_numpy(projected.reshape(shape)).to(
        dtype=torch.float64,
    )
    crop = kernel_size // 2 + 4
    rows: JsonObject = {}
    status = True
    for degrees in ANGLES_DEGREES:
        element = group.element(math.radians(degrees), "radians")
        transformed_inputs = _transform_field(
            input_type,
            inputs,
            element,
            degrees,
        )
        ours_left = functional.conv2d(
            transformed_inputs,
            ours_kernel,
            padding=kernel_size // 2,
        )
        ours_right = _transform_field(
            output_type,
            functional.conv2d(inputs, ours_kernel, padding=kernel_size // 2),
            element,
            degrees,
        )
        reference_left = functional.conv2d(
            transformed_inputs,
            reference_kernel,
            padding=kernel_size // 2,
        )
        reference_right = _transform_field(
            output_type,
            functional.conv2d(inputs, reference_kernel, padding=kernel_size // 2),
            element,
            degrees,
        )
        ours_error = _relative_cropped_error(ours_left, ours_right, crop)
        reference_error = _relative_cropped_error(reference_left, reference_right, crop)
        limit = max(5e-4, 1.10 * reference_error)
        angle_pass = ours_error <= limit
        status &= angle_pass
        rows[str(degrees)] = {
            "ours": ours_error,
            "escnn": reference_error,
            "limit": limit,
            "status": "pass" if angle_pass else "fail",
        }
    return {"status": "pass" if status else "fail", "angles_degrees": rows}


def _smooth_input_bank(frequency: int, *, count: int = 8) -> torch.Tensor:
    generator = np.random.default_rng(12012)
    values = generator.normal(
        size=(count, irrep_dimension(frequency), 65, 65),
    )
    for batch_index in range(count):
        for component_index in range(values.shape[1]):
            values[batch_index, component_index] = gaussian_filter(
                values[batch_index, component_index],
                sigma=2.0,
                radius=6,
            )
    rms = np.sqrt(np.mean(values**2, axis=(1, 2, 3), keepdims=True))
    return torch.from_numpy(values / rms)


def _transform_field(
    field_type: Any,  # noqa: ANN401
    values: torch.Tensor,
    element: Any,  # noqa: ANN401
    degrees: int,
) -> torch.Tensor:
    if degrees % 90 == 0:
        transformed_fibers = field_type.transform_fibers(values, element)
        return torch.rot90(transformed_fibers, k=degrees // 90, dims=(-2, -1))
    return field_type.transform(values, element, order=1)


def _relative_cropped_error(
    left: torch.Tensor,
    right: torch.Tensor,
    crop: int,
) -> float:
    difference = left[..., crop:-crop, crop:-crop] - right[..., crop:-crop, crop:-crop]
    denominator = torch.linalg.vector_norm(
        right[..., crop:-crop, crop:-crop],
    ).clamp_min(1e-8)
    return float(torch.linalg.vector_norm(difference) / denominator)


def _high_order_decision(
    profiles: dict[str, Candidate | None],
    reference_audits: dict[str, JsonObject],
) -> JsonObject:
    support_results: JsonObject = {}
    for kernel_size in (7, 9):
        profile_name = f"{kernel_size}-full"
        candidate = profiles[profile_name]
        if candidate is None:
            support_results[str(kernel_size)] = {
                "status": "fail",
                "reason": "full_profile_missing",
                "robust": False,
                "d_high": 0,
                "e_high": None,
            }
            continue
        robustness = _perturbation_audit(kernel_size, candidate)
        d_high = _high_order_dimension(kernel_size, candidate)
        e_high, e_floor, incremental_reference = _high_order_equivariance(
            kernel_size,
            candidate,
        )
        reference_pass = reference_audits[profile_name]["status"] == "pass"
        support_results[str(kernel_size)] = {
            "status": "measured",
            "reference_status": "pass" if reference_pass else "fail",
            "robust": robustness["status"] == "pass",
            "perturbations": robustness,
            "d_high": d_high,
            "e_high": e_high,
            "e_floor": e_floor,
            "e_limit": max(0.05, 1.5 * e_floor),
            "incremental_escnn_reference": incremental_reference,
        }
    nine = cast("JsonObject", support_results["9"])
    nine_adequate = bool(
        nine["status"] == "measured"
        and nine["reference_status"] == "pass"
        and cast("JsonObject", nine["incremental_escnn_reference"])["status"] == "pass"
        and nine["robust"] is True
        and cast("int", nine["d_high"]) > 0
        and cast("float", nine["e_high"]) <= cast("float", nine["e_limit"]),
    )
    seven = cast("JsonObject", support_results["7"])
    seven_adequate = False
    if nine_adequate and seven["status"] == "measured":
        dimension_ratio = cast("int", seven["d_high"]) / cast("int", nine["d_high"])
        error_difference = cast("float", seven["e_high"]) - cast(
            "float",
            nine["e_high"],
        )
        seven["dimension_ratio_to_9"] = dimension_ratio
        seven["error_difference_to_9"] = error_difference
        seven_adequate = bool(
            seven["reference_status"] == "pass"
            and cast("JsonObject", seven["incremental_escnn_reference"])["status"]
            == "pass"
            and seven["robust"] is True
            and dimension_ratio >= HIGH_DIMENSION_RATIO
            and cast("float", seven["e_high"]) <= cast("float", seven["e_limit"])
            and error_difference <= max(5e-4, 0.25 * cast("float", nine["e_high"])),
        )
    return {
        "supports": support_results,
        "nine_adequate": nine_adequate,
        "seven_adequate": seven_adequate,
    }


def _perturbation_audit(kernel_size: int, candidate: Candidate) -> JsonObject:
    shell_count = len(candidate.centres)
    nominal_high = _high_order_dimension(kernel_size, candidate)
    retained = 0
    changed_coordinates = [False] * (2 * shell_count)
    worst_condition = 0.0
    failures = 0
    nominal = (*candidate.centres, *candidate.widths)
    seen: set[tuple[float, ...]] = set()
    for delta in itertools.product((-0.02, 0.0, 0.02), repeat=2 * shell_count):
        if not any(delta):
            continue
        point = tuple(
            round(value + shift, 10)
            for value, shift in zip(nominal, delta, strict=True)
        )
        if point in seen:
            continue
        seen.add(point)
        centres = point[:shell_count]
        widths = point[shell_count:]
        if not _profile_point_feasible(kernel_size, centres, widths, candidate.qmax):
            continue
        retained += 1
        for index, shift in enumerate(delta):
            changed_coordinates[index] |= not math.isclose(shift, 0.0)
        audit = _audit_candidate(
            kernel_size=kernel_size,
            allowed_orders=PROFILE_ORDERS[f"{kernel_size}-full"],
            centres=centres,
            widths=widths,
            qmax=candidate.qmax,
        )
        high_dimension = _high_order_dimension_values(
            kernel_size,
            centres,
            widths,
            candidate.qmax,
        )
        worst_condition = max(worst_condition, audit.worst_condition_number)
        if (
            not all(pair.full_rank for pair in audit.pairs.values())
            or audit.worst_condition_number > PERTURBED_KAPPA_LIMIT
            or high_dimension < nominal_high
        ):
            failures += 1
    status = retained > 0 and all(changed_coordinates) and failures == 0
    return {
        "status": "pass" if status else "fail",
        "retained_count": retained,
        "failed_count": failures,
        "every_coordinate_changed": all(changed_coordinates),
        "worst_condition_number": worst_condition,
        "nominal_d_high": nominal_high,
    }


def _profile_point_feasible(
    kernel_size: int,
    centres: tuple[float, ...],
    widths: tuple[float, ...],
    qmax: tuple[int, ...],
) -> bool:
    upper = (kernel_size - 1) / 2.0 - MINIMUM_SPACING
    return bool(
        all(
            max(MINIMUM_SPACING, cutoff / 2.0) <= radius <= upper
            for radius, cutoff in zip(centres, qmax, strict=True)
        )
        and all(
            right - left >= MINIMUM_SPACING
            for left, right in itertools.pairwise(centres)
        )
        and all(MINIMUM_WIDTH <= width <= MAXIMUM_WIDTH for width in widths),
    )


def _high_order_dimension(kernel_size: int, candidate: Candidate) -> int:
    return _high_order_dimension_values(
        kernel_size,
        candidate.centres,
        candidate.widths,
        candidate.qmax,
    )


def _high_order_dimension_values(
    kernel_size: int,
    centres: tuple[float, ...],
    widths: tuple[float, ...],
    qmax: tuple[int, ...],
) -> int:
    total = 0
    for input_frequency, output_frequency in ((1, 2), (2, 1), (2, 2)):
        low = sample_pair_basis(
            input_frequency=input_frequency,
            output_frequency=output_frequency,
            kernel_size=kernel_size,
            centres=centres,
            widths=widths,
            qmax=qmax,
            allowed_orders=frozenset({0, 1, 2}),
        ).flat_columns()
        full = sample_pair_basis(
            input_frequency=input_frequency,
            output_frequency=output_frequency,
            kernel_size=kernel_size,
            centres=centres,
            widths=widths,
            qmax=qmax,
        ).flat_columns()
        total += _matrix_rank(full) - _matrix_rank(low)
    return total


def _matrix_rank(matrix: FloatArray) -> int:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    return int(np.count_nonzero(singular_values > RANK_TOLERANCE * singular_values[0]))


def _high_order_equivariance(  # noqa: PLR0914, PLR0915
    kernel_size: int,
    candidate: Candidate,
) -> tuple[float, float, JsonObject]:
    escnn = _load_escnn()
    group_space = escnn.gspaces.rot2dOnR2(N=-1, maximum_frequency=4)
    group = group_space.fibergroup
    crop = kernel_size // 2 + 4
    worst_error = 0.0
    floor_error = 0.0
    incremental_pairs: JsonObject = {}
    incremental_status = True
    subspace_errors: JsonObject = {}
    worst_case: JsonObject | None = None
    for input_frequency, output_frequency in ((1, 2), (2, 1), (2, 2)):
        inputs = _smooth_input_bank(input_frequency)
        input_type = escnn.nn.FieldType(group_space, [group.irrep(input_frequency)])
        output_type = escnn.nn.FieldType(group_space, [group.irrep(output_frequency)])
        low = sample_pair_basis(
            input_frequency=input_frequency,
            output_frequency=output_frequency,
            kernel_size=kernel_size,
            centres=candidate.centres,
            widths=candidate.widths,
            qmax=candidate.qmax,
            allowed_orders=frozenset({0, 1, 2}),
        ).flat_columns()
        full_sampled = sample_pair_basis(
            input_frequency=input_frequency,
            output_frequency=output_frequency,
            kernel_size=kernel_size,
            centres=candidate.centres,
            widths=candidate.widths,
            qmax=candidate.qmax,
        )
        high_indices = [
            index
            for index, column in enumerate(full_sampled.columns)
            if column.angular_order in {3, 4}
        ]
        high = full_sampled.flat_columns()[:, high_indices]
        low_q = orthonormalize_columns(low)
        residual = high - low_q @ (low_q.T @ high)
        left_singular, singular_values, _right = np.linalg.svd(
            residual,
            full_matrices=False,
        )
        rank = int(
            np.count_nonzero(singular_values > RANK_TOLERANCE * singular_values[0]),
        )
        incremental = left_singular[:, :rank]
        reference_low = _sample_escnn_basis(
            escnn=escnn,
            input_frequency=input_frequency,
            output_frequency=output_frequency,
            kernel_size=kernel_size,
            centres=candidate.centres,
            widths=candidate.widths,
            qmax=candidate.qmax,
            allowed_orders=frozenset({0, 1, 2}),
        )
        reference_full = _sample_escnn_basis(
            escnn=escnn,
            input_frequency=input_frequency,
            output_frequency=output_frequency,
            kernel_size=kernel_size,
            centres=candidate.centres,
            widths=candidate.widths,
            qmax=candidate.qmax,
            allowed_orders=frozenset({0, 1, 2, 3, 4}),
        )
        reference_low_q = orthonormalize_columns(reference_low)
        reference_residual = reference_full - reference_low_q @ (
            reference_low_q.T @ reference_full
        )
        reference_left, reference_singular, _reference_right = np.linalg.svd(
            reference_residual,
            full_matrices=False,
        )
        reference_rank = int(
            np.count_nonzero(
                reference_singular > RANK_TOLERANCE * reference_singular[0],
            ),
        )
        reference_incremental = reference_left[:, :reference_rank]
        projector_distance = float(
            np.linalg.norm(
                incremental @ incremental.T
                - reference_incremental @ reference_incremental.T,
                ord=2,
            ),
        )
        pair_reference_pass = bool(
            rank == reference_rank and projector_distance <= SPAN_TOLERANCE,
        )
        incremental_status &= pair_reference_pass
        incremental_pairs[PAIR_NAMES[input_frequency, output_frequency]] = {
            "status": "pass" if pair_reference_pass else "fail",
            "ours_dimension": rank,
            "escnn_dimension": reference_rank,
            "span_projector_distance": projector_distance,
        }
        pair_errors: JsonObject = {}
        kernel_shape = (
            rank,
            irrep_dimension(output_frequency),
            irrep_dimension(input_frequency),
            kernel_size,
            kernel_size,
        )
        kernels = torch.from_numpy(incremental.T.reshape(kernel_shape))
        for degrees in ANGLES_DEGREES:
            element = group.element(math.radians(degrees), "radians")
            transformed_inputs = _transform_field(
                input_type,
                inputs,
                element,
                degrees,
            )
            inverse_element = group.element(-math.radians(degrees), "radians")
            round_trip = _transform_field(
                input_type,
                transformed_inputs,
                inverse_element,
                -degrees,
            )
            floor_error = max(
                floor_error,
                _relative_cropped_error(round_trip, inputs, crop),
            )
            residual_columns: list[FloatArray] = []
            output_columns: list[FloatArray] = []
            for kernel in kernels:
                transformed_output = functional.conv2d(
                    transformed_inputs,
                    kernel,
                    padding=kernel_size // 2,
                )
                reference_output = _transform_field(
                    output_type,
                    functional.conv2d(
                        inputs,
                        kernel,
                        padding=kernel_size // 2,
                    ),
                    element,
                    degrees,
                )
                residual_columns.append(
                    (transformed_output - reference_output)[..., crop:-crop, crop:-crop]
                    .numpy()
                    .reshape(-1),
                )
                output_columns.append(
                    reference_output[..., crop:-crop, crop:-crop].numpy().reshape(-1),
                )
            residual_matrix = np.stack(residual_columns, axis=1)
            output_matrix = np.stack(output_columns, axis=1)
            residual_gram = residual_matrix.T @ residual_matrix
            output_gram = output_matrix.T @ output_matrix + 1e-8 * np.eye(rank)
            eigenvalues = linalg.eigvalsh(residual_gram, output_gram)
            subspace_error = math.sqrt(max(float(eigenvalues[-1]), 0.0))
            pair_errors[str(degrees)] = subspace_error
            if subspace_error > worst_error:
                worst_error = subspace_error
                worst_case = {
                    "pair": PAIR_NAMES[input_frequency, output_frequency],
                    "angle_degrees": degrees,
                    "error": subspace_error,
                }
        subspace_errors[PAIR_NAMES[input_frequency, output_frequency]] = pair_errors
    return (
        worst_error,
        floor_error,
        {
            "status": "pass" if incremental_status else "fail",
            "pairs": incremental_pairs,
            "subspace_errors_by_pair_and_angle": subspace_errors,
            "worst_case": worst_case,
        },
    )


def _architecture_counts(profiles: dict[str, Candidate | None]) -> JsonObject:
    primary = {
        "R": FieldSpec(3, 0, 0),
        "A": FieldSpec(16, 8, 0),
        "B": FieldSpec(16, 12, 4),
        "C": FieldSpec(24, 12, 8),
        "D": FieldSpec(32, 20, 12),
        "L": FieldSpec(16, 0, 0),
    }
    fallback = _equal_copy_f01_fields()
    return {
        "F012-7": _count_architecture("F012-7", primary, profiles),
        "F012-9": _count_architecture("F012-9", primary, profiles),
        "F01": _count_architecture("F01", fallback, profiles),
        "parameter_cap": PARAMETER_CAP,
    }


def _count_architecture(  # noqa: PLR0914
    candidate_name: str,
    fields: dict[str, FieldSpec],
    profiles: Mapping[str, Candidate | FixedProfile | None],
) -> JsonObject:
    positions, norm_specs, gate_specs = _topology(fields)
    coefficient_parameters = 0
    dense_macs = 0
    expansion_macs = 0
    missing_profiles: set[str] = set()
    for position in positions:
        profile_name = _position_profile(candidate_name, position)
        profile = profiles[profile_name]
        kernel_size = int(profile_name.split("-", maxsplit=1)[0])
        dense_macs += (
            position.input_spec.channels
            * position.output_spec.channels
            * kernel_size
            * kernel_size
            * position.output_size
            * position.output_size
        )
        if profile is None:
            missing_profiles.add(profile_name)
            continue
        allowed_orders = PROFILE_ORDERS[profile_name]
        for input_frequency, input_copies in enumerate(position.input_spec.copies):
            for output_frequency, output_copies in enumerate(
                position.output_spec.copies,
            ):
                if input_copies == 0 or output_copies == 0:
                    continue
                dimension = sample_pair_basis(
                    input_frequency=input_frequency,
                    output_frequency=output_frequency,
                    kernel_size=kernel_size,
                    centres=profile.centres,
                    widths=profile.widths,
                    qmax=profile.qmax,
                    allowed_orders=allowed_orders,
                ).values.shape[0]
                coefficients = input_copies * output_copies * dimension
                coefficient_parameters += coefficients
                expansion_macs += (
                    coefficients
                    * irrep_dimension(input_frequency)
                    * irrep_dimension(output_frequency)
                    * kernel_size
                    * kernel_size
                )
    bias_parameters = sum(
        position.output_spec.n0 for position in positions if position.bias
    )
    normalization_parameters = sum(
        2 * spec.n0 + spec.n1 + spec.n2 for spec in norm_specs
    )
    gate_parameters = sum(2 * sum(spec.copies) for spec in gate_specs)
    total = (
        coefficient_parameters
        + bias_parameters
        + normalization_parameters
        + gate_parameters
    )
    return {
        "status": "blocked_missing_profile" if missing_profiles else "pass",
        "missing_profiles": sorted(missing_profiles),
        "learned_convolution_count": len(positions),
        "normalization_module_count": len(norm_specs),
        "gate_module_count": len(gate_specs),
        "coefficient_parameters": None if missing_profiles else coefficient_parameters,
        "scalar_bias_parameters": bias_parameters,
        "normalization_parameters": normalization_parameters,
        "gate_parameters": gate_parameters,
        "total_learned_parameters": None if missing_profiles else total,
        "within_parameter_cap": not missing_profiles and total <= PARAMETER_CAP,
        "dense_convolution_macs_per_sample": dense_macs,
        "basis_expansion_macs_per_forward": None
        if missing_profiles
        else expansion_macs,
        "physical_widths": [fields[name].channels for name in ("A", "B", "C", "D")],
    }


def _topology(
    fields: dict[str, FieldSpec],
) -> tuple[list[ConvPosition], list[FieldSpec], list[FieldSpec]]:
    positions: list[ConvPosition] = [
        ConvPosition(
            "stem",
            fields["R"],
            fields["A"],
            256,
            followed_by_norm=True,
            bias=False,
        ),
    ]
    norm_specs = [fields["A"]]
    gate_specs = [fields["A"]]
    encoder = (
        ("A", "A", 256),
        ("A", "A", 256),
        ("A", "B", 128),
        ("B", "B", 128),
        ("B", "C", 64),
        ("C", "C", 64),
        ("C", "D", 32),
        ("D", "D", 32),
    )
    for index, (input_name, output_name, output_size) in enumerate(encoder):
        input_spec = fields[input_name]
        output_spec = fields[output_name]
        main1_size = output_size if input_name == output_name else output_size * 2
        positions.extend(
            (
                ConvPosition(
                    f"enc{index}.main1",
                    input_spec,
                    output_spec,
                    main1_size,
                    followed_by_norm=True,
                    bias=False,
                ),
                ConvPosition(
                    f"enc{index}.main2",
                    output_spec,
                    output_spec,
                    output_size,
                    followed_by_norm=True,
                    bias=False,
                ),
            ),
        )
        norm_specs.extend((output_spec, output_spec))
        gate_specs.extend((output_spec, output_spec))
        if input_name != output_name:
            positions.append(
                ConvPosition(
                    f"enc{index}.skip",
                    input_spec,
                    output_spec,
                    output_size,
                    followed_by_norm=True,
                    bias=False,
                ),
            )
            norm_specs.append(output_spec)
    positions.extend(
        (
            ConvPosition(
                "mu",
                fields["D"],
                fields["L"],
                32,
                followed_by_norm=False,
                bias=True,
            ),
            ConvPosition(
                "logvar",
                fields["D"],
                fields["L"],
                32,
                followed_by_norm=False,
                bias=True,
            ),
            ConvPosition(
                "latent_projection",
                fields["L"],
                fields["D"],
                32,
                followed_by_norm=True,
                bias=False,
            ),
        ),
    )
    norm_specs.append(fields["D"])
    gate_specs.append(fields["D"])
    decoder = (
        ("D", "D", 32),
        ("D", "D", 32),
        ("D", "C", 64),
        ("C", "C", 64),
        ("C", "B", 128),
        ("B", "B", 128),
        ("B", "A", 256),
        ("A", "A", 256),
    )
    for index, (input_name, output_name, output_size) in enumerate(decoder):
        input_spec = fields[input_name]
        output_spec = fields[output_name]
        positions.extend(
            (
                ConvPosition(
                    f"dec{index}.main1",
                    input_spec,
                    output_spec,
                    output_size,
                    followed_by_norm=True,
                    bias=False,
                ),
                ConvPosition(
                    f"dec{index}.main2",
                    output_spec,
                    output_spec,
                    output_size,
                    followed_by_norm=True,
                    bias=False,
                ),
            ),
        )
        norm_specs.extend((output_spec, output_spec))
        gate_specs.extend((output_spec, output_spec))
        if input_name != output_name:
            positions.append(
                ConvPosition(
                    f"dec{index}.skip",
                    input_spec,
                    output_spec,
                    output_size,
                    followed_by_norm=True,
                    bias=False,
                ),
            )
            norm_specs.append(output_spec)
    positions.append(
        ConvPosition(
            "rgb",
            fields["A"],
            fields["R"],
            256,
            followed_by_norm=False,
            bias=True,
        ),
    )
    if not (
        len(positions) == EXPECTED_CONV_COUNT
        and len(norm_specs) == EXPECTED_NORM_COUNT
        and len(gate_specs) == EXPECTED_GATE_COUNT
    ):
        message = (
            "fixed topology inventory no longer has 43 convs, 40 norms, and 34 gates"
        )
        raise RuntimeError(message)
    return positions, norm_specs, gate_specs


def _position_profile(candidate_name: str, position: ConvPosition) -> str:
    if position.name == "stem":
        return "9-low"
    if candidate_name == "F012-7":
        return "7-full"
    if candidate_name == "F01":
        return "7-low"
    contains_f2 = position.input_spec.n2 > 0 or position.output_spec.n2 > 0
    return "9-full" if contains_f2 else "7-low"


def _select_provisional_candidate(
    *,
    profile_results: dict[str, Candidate | None],
    reference_audits: dict[str, JsonObject],
    high_order: JsonObject,
    counts: JsonObject,
) -> JsonObject:
    blockers: list[JsonValue] = []
    if high_order["seven_adequate"] is True:
        count = cast("JsonObject", counts["F012-7"])
        reference_pass = all(
            reference_audits.get(name, {}).get("status") == "pass"
            for name in ("9-low", "7-full")
        )
        if count["within_parameter_cap"] is True and reference_pass:
            return {
                "status": "pass",
                "name": "F012-7",
                "blockers": blockers,
                "reference_profiles": ["7-full", "9-low"],
            }
        blockers.append("F012-7_reference_or_parameter_gate")
    if high_order["nine_adequate"] is True:
        count = cast("JsonObject", counts["F012-9"])
        reference_pass = all(
            reference_audits.get(name, {}).get("status") == "pass"
            for name in ("9-low", "9-full", "7-low")
        )
        if count["within_parameter_cap"] is True and reference_pass:
            return {
                "status": "pass",
                "name": "F012-9",
                "blockers": blockers,
                "reference_profiles": ["7-low", "9-full", "9-low"],
            }
        blockers.append("F012-9_reference_or_parameter_gate")
    fallback_count = cast("JsonObject", counts["F01"])
    fallback_profiles_exist = all(
        profile_results[name] is not None for name in ("9-low", "7-low")
    )
    fallback_reference_pass = all(
        reference_audits.get(name, {}).get("status") == "pass"
        for name in ("9-low", "7-low")
    )
    if (
        fallback_profiles_exist
        and fallback_reference_pass
        and fallback_count["within_parameter_cap"] is True
    ):
        return {
            "status": "pass",
            "name": "F01",
            "blockers": blockers,
            "reference_profiles": ["7-low", "9-low"],
        }
    blockers.append("F01_basis_reference_or_parameter_gate")
    return {
        "status": "blocked",
        "name": None,
        "blockers": blockers,
        "reference_profiles": [],
    }


def _initialization_audit(
    *,
    profile_results: Mapping[str, Candidate | FixedProfile | None],
    provisional: JsonObject,
) -> JsonObject:
    candidate_name = provisional["name"]
    if not isinstance(candidate_name, str):
        return {"status": "blocked", "reason": "no_provisional_candidate"}
    fields = (
        _equal_copy_f01_fields()
        if candidate_name == "F01"
        else {
            "R": FieldSpec(3, 0, 0),
            "A": FieldSpec(16, 8, 0),
            "B": FieldSpec(16, 12, 4),
            "C": FieldSpec(24, 12, 8),
            "D": FieldSpec(32, 20, 12),
            "L": FieldSpec(16, 0, 0),
        }
    )
    positions, _norms, _gates = _topology(fields)
    distinct = {
        (
            position.input_spec,
            position.output_spec,
            _position_profile(candidate_name, position),
        )
        for position in positions
        if position.name != "rgb"
    }
    rows: JsonObject = {}
    all_pass = True
    for index, (input_spec, output_spec, profile_name) in enumerate(
        sorted(distinct, key=lambda item: (item[2], item[0].copies, item[1].copies)),
    ):
        profile = profile_results[profile_name]
        if profile is None:
            all_pass = False
            continue
        ratios, ours_variances, escnn_variances = _initialization_variance_ratios(
            input_spec=input_spec,
            output_spec=output_spec,
            profile_name=profile_name,
            profile=profile,
        )
        row_pass = all(
            INITIALIZATION_RATIO_MINIMUM <= ratio <= INITIALIZATION_RATIO_MAXIMUM
            for ratio in ratios.values()
        )
        all_pass &= row_pass
        rows[str(index)] = {
            "status": "pass" if row_pass else "fail",
            "input": list(input_spec.copies),
            "output": list(output_spec.copies),
            "profile": profile_name,
            "mean_variance_ratio_by_frequency": ratios,
            "ours_mean_output_variance_by_frequency": ours_variances,
            "escnn_mean_output_variance_by_frequency": escnn_variances,
        }
    return {
        "status": "pass" if all_pass else "fail",
        "trial_count": 128,
        "distinct_layer_type_count": len(distinct),
        "layers": rows,
    }


def _initialization_variance_ratios(
    *,
    input_spec: FieldSpec,
    output_spec: FieldSpec,
    profile_name: str,
    profile: Candidate | FixedProfile,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    kernel_size = int(profile_name.split("-", maxsplit=1)[0])
    allowed_orders = PROFILE_ORDERS[profile_name]
    trial_count = 128
    ours_kernels = _assemble_ours_trial_kernels(
        input_spec=input_spec,
        output_spec=output_spec,
        kernel_size=kernel_size,
        profile=profile,
        allowed_orders=allowed_orders,
        trial_count=trial_count,
    )
    escnn = _load_escnn()
    reference_layer = _build_escnn_reference_layer(
        escnn=escnn,
        input_spec=input_spec,
        output_spec=output_spec,
        kernel_size=kernel_size,
        profile=profile,
        allowed_orders=allowed_orders,
    )
    reference_kernels: list[torch.Tensor] = []
    with torch.random.fork_rng(), torch.no_grad():
        torch.manual_seed(12012)
        for _trial in range(trial_count):
            escnn.nn.init.generalized_he_init(
                reference_layer.weights,
                reference_layer.basisexpansion,
            )
            reference_kernels.append(
                reference_layer
                .basisexpansion(reference_layer.weights)
                .detach()
                .clone()
                .reshape(
                    output_spec.channels,
                    input_spec.channels,
                    kernel_size,
                    kernel_size,
                ),
            )
    reference_kernel_bank = torch.stack(reference_kernels)
    input_generator = torch.Generator().manual_seed(22012)
    inputs = torch.randn(
        trial_count,
        input_spec.channels,
        33,
        33,
        generator=input_generator,
    )
    inputs -= inputs.mean(dim=(-2, -1), keepdim=True)
    inputs /= inputs.square().mean(dim=(-2, -1), keepdim=True).sqrt()
    ours_outputs = _trial_grouped_convolution(inputs, ours_kernels)
    reference_outputs = _trial_grouped_convolution(inputs, reference_kernel_bank)
    ours_variances = _frequency_output_variances(ours_outputs, output_spec)
    reference_variances = _frequency_output_variances(reference_outputs, output_spec)
    ratios = {
        name: ours_variances[name] / reference_variances[name]
        for name in ours_variances
    }
    return ratios, ours_variances, reference_variances


def _assemble_ours_trial_kernels(  # noqa: PLR0913
    *,
    input_spec: FieldSpec,
    output_spec: FieldSpec,
    kernel_size: int,
    profile: Candidate | FixedProfile,
    allowed_orders: frozenset[int],
    trial_count: int,
) -> torch.Tensor:
    kernels = torch.zeros(
        trial_count,
        output_spec.channels,
        input_spec.channels,
        kernel_size,
        kernel_size,
    )
    generator = torch.Generator().manual_seed(12012)
    input_offsets = _frequency_offsets(input_spec)
    output_offsets = _frequency_offsets(output_spec)
    present_input_count = sum(copies > 0 for copies in input_spec.copies)
    for input_frequency, input_copies in enumerate(input_spec.copies):
        if input_copies == 0:
            continue
        input_dimension = irrep_dimension(input_frequency)
        input_start = input_offsets[input_frequency]
        for output_frequency, output_copies in enumerate(output_spec.copies):
            if output_copies == 0:
                continue
            output_dimension = irrep_dimension(output_frequency)
            output_start = output_offsets[output_frequency]
            sampled = sample_pair_basis(
                input_frequency=input_frequency,
                output_frequency=output_frequency,
                kernel_size=kernel_size,
                centres=profile.centres,
                widths=profile.widths,
                qmax=profile.qmax,
                allowed_orders=allowed_orders,
            )
            basis = torch.from_numpy(stored_pair_basis(sampled)).float()
            basis_dimension = len(sampled.columns)
            scale = generalized_he_standard_deviation(
                present_input_frequency_count=present_input_count,
                input_copies=input_copies,
                basis_dimension=basis_dimension,
            )
            coefficients = (
                torch.randn(
                    trial_count,
                    output_copies,
                    input_copies,
                    basis_dimension,
                    generator=generator,
                )
                * scale
            )
            block = torch.einsum("toib,buvhw->touivhw", coefficients, basis)
            block = block.reshape(
                trial_count,
                output_copies * output_dimension,
                input_copies * input_dimension,
                kernel_size,
                kernel_size,
            )
            kernels[
                :,
                output_start : output_start + output_copies * output_dimension,
                input_start : input_start + input_copies * input_dimension,
            ] = block
    return kernels


def _build_escnn_reference_layer(  # noqa: PLR0913
    *,
    escnn: Any,  # noqa: ANN401
    input_spec: FieldSpec,
    output_spec: FieldSpec,
    kernel_size: int,
    profile: Candidate | FixedProfile,
    allowed_orders: frozenset[int],
) -> Any:  # noqa: ANN401
    group_space = escnn.gspaces.rot2dOnR2(N=-1, maximum_frequency=4)
    group = group_space.fibergroup
    input_representations = [
        group.irrep(frequency)
        for frequency, copies in enumerate(input_spec.copies)
        for _copy in range(copies)
    ]
    output_representations = [
        group.irrep(frequency)
        for frequency, copies in enumerate(output_spec.copies)
        for _copy in range(copies)
    ]
    radii = (0.0, *profile.centres)
    cutoffs = (0, *profile.qmax)

    def frequency_cutoff(radius: float) -> int:
        radial_index = min(
            range(len(radii)),
            key=lambda index: abs(radii[index] - radius),
        )
        return min(cutoffs[radial_index], max(allowed_orders))

    return escnn.nn.R2Conv(
        escnn.nn.FieldType(group_space, input_representations),
        escnn.nn.FieldType(group_space, output_representations),
        kernel_size,
        bias=False,
        sigma=[0.005, *profile.widths],
        frequencies_cutoff=frequency_cutoff,
        rings=list(radii),
        recompute=True,
        initialize=False,
    )


def _trial_grouped_convolution(
    inputs: torch.Tensor,
    kernels: torch.Tensor,
) -> torch.Tensor:
    trial_count, input_channels, height, width = inputs.shape
    output_channels = kernels.shape[1]
    output = functional.conv2d(
        inputs.reshape(1, trial_count * input_channels, height, width),
        kernels.reshape(
            trial_count * output_channels,
            input_channels,
            kernels.shape[-2],
            kernels.shape[-1],
        ),
        groups=trial_count,
    )
    return output.reshape(
        trial_count,
        output_channels,
        output.shape[-2],
        output.shape[-1],
    )


def _frequency_output_variances(
    outputs: torch.Tensor,
    spec: FieldSpec,
) -> dict[str, float]:
    offsets = _frequency_offsets(spec)
    variances: dict[str, float] = {}
    for frequency, copies in enumerate(spec.copies):
        if copies == 0:
            continue
        start = offsets[frequency]
        stop = start + copies * irrep_dimension(frequency)
        trial_variances = outputs[:, start:stop].var(
            dim=(1, 2, 3),
            correction=0,
        )
        variances[f"F{frequency}"] = float(trial_variances.mean())
    return variances


def _frequency_offsets(spec: FieldSpec) -> tuple[int, int, int]:
    return (
        0,
        spec.n0,
        spec.n0 + 2 * spec.n1,
    )


def _candidate_manifest(candidate: Candidate) -> JsonObject:
    return {
        "centres": list(candidate.centres),
        "widths": list(candidate.widths),
        "qmax": list(candidate.qmax),
    }


def _candidate_audit_payload(candidate: Candidate) -> JsonObject:
    return {
        "status": "pass" if candidate.audit.passes else "fail",
        "selected": _candidate_manifest(candidate),
        "origin": candidate.origin,
        "solver": candidate.solver,
        "total_dimension": candidate.audit.total_dimension,
        "worst_condition_number": candidate.audit.worst_condition_number,
        "worst_minimum_singular_value": candidate.audit.worst_minimum_singular_value,
        "objective": candidate.audit.objective,
        "pairs": {
            name: {
                "dimension": pair.dimension,
                "rank": pair.rank,
                "condition_number": pair.condition_number,
                "minimum_singular_value": pair.minimum_singular_value,
                "maximum_singular_value": pair.maximum_singular_value,
                "minimum_column_norm": pair.minimum_column_norm,
                "full_rank": pair.full_rank,
            }
            for name, pair in candidate.audit.pairs.items()
        },
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-layout",
        action="store_true",
        help="reuse the selected profiles and refresh only equal-copy evidence",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def _write_json(path: Path, payload: JsonObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> JsonObject:
    return cast("JsonObject", json.loads(path.read_text(encoding="utf-8")))


if __name__ == "__main__":
    raise SystemExit(main())
