# pyright: reportAny=false, reportArgumentType=false, reportAssignmentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportExplicitAny=false, reportIndexIssue=false, reportMissingTypeArgument=false, reportOptionalMemberAccess=false, reportReturnType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false
# Copyright 2026 HiperMaximus
# ruff: noqa: C901, COM812, DOC201, DOC501, E501, EM101, EM102, PLR0912, PLR0913, PLR0914, PLR0915, PLR0917, PLR2004, PLW0717, RUF001, TRY003, TRY301
"""Fail-closed local renderer for the Spec 0044 evidence package."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from eqvae.metrics.reconstruction import (
    mae_per_image,
    mse_per_image,
    normalized_to_image_domain,
    psnr_per_image,
    ssim_per_image,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from torch import Tensor


ArrayF64 = NDArray[np.float64]
ArrayU8 = NDArray[np.uint8]

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_CONTRACT: Final = (
    REPOSITORY_ROOT / "docs/data/spec0044_professor_metrics_inputs.json"
)
DEFAULT_OUTPUT_DIR: Final = REPOSITORY_ROOT / "runs/local/professor_metrics_v1"
EXPECTED_INPUT_CONTRACT_SHA256: Final = (
    "486627232fe843b537556264f1c81dd246fbd7564f589be7d9bc781e9ede96c8"
)
EXPECTED_MODELS: Final = ("normal", "so2")
EXPECTED_RANKS: Final = {0, 1}
EXPECTED_ANGLES: Final = (90, 180, 270)
FIXED25_COUNT: Final = 25
FINAL_COUNTER: Final = 60_000
IMAGE_CHANNELS: Final = 3
IMAGE_SPATIAL_SIZE: Final = 256
DEFAULT_SMOOTHING_WINDOW: Final = 501
PLOT_NORMAL: Final = "#2563eb"
PLOT_SO2: Final = "#ea580c"
PLOT_VALIDATION_NORMAL: Final = "#1e3a8a"
PLOT_VALIDATION_SO2: Final = "#9a3412"
BACKGROUND: Final = "#ffffff"
INK: Final = "#111827"
MUTED: Final = "#6b7280"
GRID: Final = "#e5e7eb"
PANEL: Final = "#f8fafc"
LABEL_NAMES: Final = {
    0: "HGSC",
    1: "LGSC",
    2: "EC",
    3: "CC",
    4: "MC",
}


@dataclass(frozen=True)
class Fixed25Data:
    """Validated final fixed-25 tensors and identities."""

    originals_uint8: Tensor
    target_norm: Tensor
    predictions: dict[str, Tensor]
    identities: list[dict[str, Any]]
    final_roots: dict[str, Path]
    source_hashes: dict[str, str]


@dataclass(frozen=True)
class DashboardData:
    """Validated full-history arrays used by the six-panel dashboard."""

    train: dict[str, dict[str, ArrayF64]]
    validation: dict[str, dict[str, ArrayF64]]
    fixed_psnr: dict[str, dict[str, ArrayF64]]
    equivariance: dict[str, dict[str, ArrayF64]]
    skipped_attempts: list[dict[str, str]]
    train_fieldnames: list[str]
    source_hashes: dict[str, str]
    physical_skip_disclosure: dict[str, Any]


@dataclass(frozen=True)
class PlotSeries:
    """One labeled line/marker series."""

    x: ArrayF64
    y: ArrayF64
    color: str
    label: str
    dashed: bool = False
    markers: bool = False


def sha256_file(path: Path) -> str:
    """Return one file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_input_contract(
    path: Path,
    *,
    require_canonical_hash: bool = True,
) -> dict[str, Any]:
    """Load and minimally validate the machine-readable input contract."""
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_hash = sha256_file(path)
    if require_canonical_hash and actual_hash != EXPECTED_INPUT_CONTRACT_SHA256:
        raise ValueError(
            f"Spec 0044 input contract hash mismatch: {actual_hash} != "
            f"{EXPECTED_INPUT_CONTRACT_SHA256}"
        )
    payload = cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported Spec 0044 input contract schema")
    sessions = payload.get("sessions")
    if not isinstance(sessions, list) or not sessions:
        raise ValueError("Spec 0044 requires source sessions")
    if require_canonical_hash and len(sessions) != 10:
        raise ValueError("Canonical Spec 0044 requires exactly ten source sessions")
    consumed = payload.get("consumed_artifacts_sha256")
    if not isinstance(consumed, dict) or set(
        consumed
    ) != expected_consumed_artifact_paths(payload):
        raise ValueError(
            "Consumed-artifact hash coverage is incomplete or contains extras"
        )
    return payload


def validate_session_layout(contract: Mapping[str, Any]) -> None:
    """Validate exact model/session coverage before reading scientific values."""
    sessions = cast("list[dict[str, Any]]", contract["sessions"])
    seen_paths: set[str] = set()
    for session in sessions:
        path = str(session["path"])
        if path in seen_paths:
            raise ValueError(f"Duplicate session path: {path}")
        if path.endswith("_remote"):
            raise ValueError(f"Mirror session is forbidden: {path}")
        seen_paths.add(path)
    expected_counter = contract["successful_update_counter"]
    expected_first = int(expected_counter["first"])
    expected_last = int(expected_counter["last"])
    session_counts = cast("dict[str, int]", contract["session_count_by_model"])
    if set(session_counts) != set(EXPECTED_MODELS):
        raise ValueError("Session-count contract must name exactly normal and so2")
    for model in EXPECTED_MODELS:
        expected_count = int(session_counts[model])
        model_sessions = sorted(
            (session for session in sessions if session["model"] == model),
            key=lambda session: int(session["counter_first"]),
        )
        if len(model_sessions) != expected_count:
            raise ValueError(f"Expected {expected_count} {model} sessions")
        cursor = expected_first
        for session in model_sessions:
            first = int(session["counter_first"])
            last = int(session["counter_last"])
            if first != cursor or last < first:
                raise ValueError(
                    f"Non-contiguous {model} session range: {first}..{last}; "
                    f"expected first {cursor}"
                )
            cursor = last + 1
        if cursor - 1 != expected_last:
            raise ValueError(f"Incomplete {model} history: ends at {cursor - 1}")


def expected_consumed_artifact_paths(contract: Mapping[str, Any]) -> set[str]:
    """Return the exact child artifacts consumed outside session-level hashes."""
    sessions = cast("list[dict[str, Any]]", contract["sessions"])
    boundaries = [
        int(value)
        for value in contract["successful_update_counter"]["validation_boundaries"]
    ]
    expected: set[str] = set()
    for session in sessions:
        root = Path(str(session["path"]))
        first = int(session["counter_first"])
        last = int(session["counter_last"])
        for counter in boundaries:
            if first <= counter <= last:
                expected.add(
                    (
                        root
                        / "artifacts/fixed25"
                        / f"boundary_{counter:06d}"
                        / "reconstruction_progress.pt"
                    ).as_posix()
                )
        if last == int(contract["successful_update_counter"]["last"]):
            expected.update(
                (
                    root
                    / "artifacts/fixed25"
                    / f"boundary_{last:06d}"
                    / f"rotated_angle_{angle}.pt"
                ).as_posix()
                for angle in EXPECTED_ANGLES
            )
    expected.add(str(contract["fixed25"]["pca_reference_path"]))
    return expected


def _repo_path(path_text: str, *, repository_root: Path) -> Path:
    path = Path(path_text)
    resolved = path if path.is_absolute() else repository_root / path
    return resolved.resolve()


def _verify_hash(path: Path, expected: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: {actual} != {expected}")
    return actual


def _verify_consumed_artifact(
    path: Path,
    contract: Mapping[str, Any],
    *,
    repository_root: Path,
) -> str:
    relative = _relative_or_absolute(path, repository_root)
    expected = cast("dict[str, str]", contract["consumed_artifacts_sha256"]).get(
        relative
    )
    if expected is None:
        raise ValueError(f"Consumed artifact lacks a pinned digest: {relative}")
    return _verify_hash(path, expected)


def _load_torch_mapping(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a tensor mapping in {path}")
    return cast("dict[str, Any]", payload)


def _as_tensor(payload: Mapping[str, Any], key: str, *, path: Path) -> Tensor:
    value = payload.get(key)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected tensor key {key!r} in {path}")
    return value


def _validate_image_tensor(
    tensor: Tensor,
    *,
    expected_dtype: torch.dtype,
    name: str,
) -> None:
    if tuple(tensor.shape) != (
        FIXED25_COUNT,
        IMAGE_CHANNELS,
        IMAGE_SPATIAL_SIZE,
        IMAGE_SPATIAL_SIZE,
    ):
        raise ValueError(f"{name} has unexpected shape {tuple(tensor.shape)}")
    if tensor.dtype != expected_dtype:
        raise ValueError(f"{name} has dtype {tensor.dtype}, expected {expected_dtype}")
    if not bool(torch.isfinite(tensor.to(torch.float32)).all().item()):
        raise ValueError(f"{name} contains non-finite values")


def _session_boundaries(
    session: Mapping[str, Any],
    *,
    expected_boundaries: Sequence[int],
) -> list[int]:
    first = int(session["counter_first"])
    last = int(session["counter_last"])
    return [value for value in expected_boundaries if first <= value <= last]


def load_fixed25_data(
    contract: Mapping[str, Any],
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> Fixed25Data:
    """Validate and load the final paired fixed-25 reconstruction tensors."""
    validate_session_layout(contract)
    fixed_contract = cast("dict[str, Any]", contract["fixed25"])
    sessions = cast("list[dict[str, Any]]", contract["sessions"])
    source_hashes: dict[str, str] = {}
    identities_reference: list[dict[str, Any]] | None = None
    final_roots: dict[str, Path] = {}
    originals_reference: Tensor | None = None
    expected_boundaries = [
        int(value)
        for value in contract["successful_update_counter"]["validation_boundaries"]
    ]

    selector = repository_root / "configs/spec0001/fixed_25_validation_patches.json"
    source_hashes[_relative_or_absolute(selector, repository_root)] = _verify_hash(
        selector,
        str(fixed_contract["selector_sha256"]),
    )

    model_boundaries: dict[str, list[int]] = {model: [] for model in EXPECTED_MODELS}
    for session in sessions:
        root = _repo_path(str(session["path"]), repository_root=repository_root)
        for relative, expected in cast("dict[str, str]", session["hashes"]).items():
            source = root / relative
            source_hashes[_relative_or_absolute(source, repository_root)] = (
                _verify_hash(
                    source,
                    expected,
                )
            )
        fixed_root = root / "artifacts/fixed25"
        manifest_path = fixed_root / "manifest.json"
        manifest = cast(
            "dict[str, Any]",
            json.loads(manifest_path.read_text(encoding="utf-8")),
        )
        if (
            manifest.get("schema") != "spec0010.fixed25_equivariance.manifest.v1"
            or manifest.get("data_source") != "real"
            or manifest.get("promotable") is not True
        ):
            raise ValueError(f"Non-promotable fixed-25 manifest: {manifest_path}")
        boundaries = [int(value) for value in manifest["boundary_optimizer_steps"]]
        if boundaries != _session_boundaries(
            session,
            expected_boundaries=expected_boundaries,
        ):
            raise ValueError(f"Unexpected boundary coverage in {manifest_path}")
        model = str(session["model"])
        model_boundaries[model].extend(boundaries)
        identities = cast("list[dict[str, Any]]", manifest["selector"]["identities"])
        if len(identities) != FIXED25_COUNT:
            raise ValueError(f"Expected 25 identities in {manifest_path}")
        if identities_reference is None:
            identities_reference = identities
        elif identities != identities_reference:
            raise ValueError(f"Fixed-25 identities differ in {manifest_path}")
        originals_path = fixed_root / "originals.pt"
        source_hashes[_relative_or_absolute(originals_path, repository_root)] = (
            _verify_hash(
                originals_path,
                str(fixed_contract["originals_sha256"]),
            )
        )
        originals_payload = _load_torch_mapping(originals_path)
        originals = _as_tensor(originals_payload, "images_uint8", path=originals_path)
        _validate_image_tensor(
            originals,
            expected_dtype=torch.uint8,
            name=str(originals_path),
        )
        if (
            originals_payload.get("selector_sha256")
            != fixed_contract["selector_sha256"]
        ):
            raise ValueError(f"Selector hash mismatch inside {originals_path}")
        if cast("list[dict[str, Any]]", originals_payload["identities"]) != identities:
            raise ValueError(f"Identity mismatch inside {originals_path}")
        if originals_reference is None:
            originals_reference = originals
        elif not torch.equal(originals_reference, originals):
            raise ValueError(f"Original tensors differ in {originals_path}")
        if int(session["counter_last"]) == FINAL_COUNTER:
            if model in final_roots:
                raise ValueError(f"Duplicate final root for {model}")
            final_roots[model] = root

    for model in EXPECTED_MODELS:
        if model_boundaries[model] != expected_boundaries:
            raise ValueError(f"Incomplete fixed-25 boundary coverage for {model}")
    if identities_reference is None or originals_reference is None:
        raise ValueError("No fixed-25 inputs were loaded")
    pinned_rank = int(fixed_contract["rotated_grid_selector_rank"])
    if (
        identities_reference[pinned_rank]["sample_id"]
        != fixed_contract["rotated_grid_sample_id"]
    ):
        raise ValueError("Pinned rotated-grid sample identity mismatch")

    predictions: dict[str, Tensor] = {}
    for model in EXPECTED_MODELS:
        root = final_roots[model]
        path = (
            root
            / "artifacts/fixed25"
            / f"boundary_{FINAL_COUNTER:06d}"
            / "reconstruction_progress.pt"
        )
        expected_key = f"{model}_final_reconstruction_sha256"
        source_hashes[_relative_or_absolute(path, repository_root)] = _verify_hash(
            path,
            str(fixed_contract[expected_key]),
        )
        payload = _load_torch_mapping(path)
        if int(payload.get("optimizer_step", -1)) != FINAL_COUNTER:
            raise ValueError(f"Final reconstruction has wrong counter in {path}")
        reconstruction = _as_tensor(payload, "reconstruction", path=path)
        _validate_image_tensor(
            reconstruction,
            expected_dtype=torch.float16,
            name=str(path),
        )
        predictions[model] = reconstruction.to(torch.float32)

    target_norm = originals_reference.to(torch.float32).div(255.0).mul(2.0).sub(1.0)
    return Fixed25Data(
        originals_uint8=originals_reference,
        target_norm=target_norm,
        predictions=predictions,
        identities=identities_reference,
        final_roots=final_roots,
        source_hashes=source_hashes,
    )


def compute_fixed25_metrics(
    data: Fixed25Data,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compute official paired fixed-25 reconstruction metrics and summaries."""
    metric_arrays: dict[str, dict[str, ArrayF64]] = {}
    rows: list[dict[str, Any]] = []
    for model in EXPECTED_MODELS:
        prediction = data.predictions[model]
        target = data.target_norm
        prediction_img = normalized_to_image_domain(prediction)
        target_img = normalized_to_image_domain(target)
        arrays = {
            "mae_norm": mae_per_image(prediction, target).numpy().astype(np.float64),
            "mse_norm": mse_per_image(prediction, target).numpy().astype(np.float64),
            "psnr_img_db": psnr_per_image(prediction_img, target_img)
            .numpy()
            .astype(np.float64),
            "ssim_img": ssim_per_image(prediction_img, target_img)
            .numpy()
            .astype(np.float64),
        }
        if not all(np.isfinite(values).all() for values in arrays.values()):
            raise ValueError(f"Non-finite fixed-25 metric for {model}")
        metric_arrays[model] = arrays
        for index, identity in enumerate(data.identities):
            rows.append({
                "model": model,
                "selector_rank": int(identity["rank"]),
                "sample_id": str(identity["sample_id"]),
                "wsi_id": str(identity["wsi_id"]),
                "source_label": int(identity["label"]),
                "source_label_name": LABEL_NAMES[int(identity["label"])],
                "x": int(identity["x"]),
                "y": int(identity["y"]),
                "mae_mse_domain": "normalized_model_domain_-1_1",
                "psnr_ssim_domain": "clamped_image_domain_0_1",
                **{name: float(values[index]) for name, values in arrays.items()},
            })

    summaries: dict[str, Any] = {}
    paired_deltas: dict[str, Any] = {}
    for model in EXPECTED_MODELS:
        summaries[model] = {
            metric: _descriptive_summary(values)
            for metric, values in metric_arrays[model].items()
        }
    for metric in metric_arrays["normal"]:
        deltas = metric_arrays["normal"][metric] - metric_arrays["so2"][metric]
        paired_deltas[f"normal_minus_so2_{metric}"] = {
            **_descriptive_summary(deltas),
            "interpretation": "paired descriptive delta only; no significance or superiority claim",
        }
    summary = {
        "schema": "spec0044.reconstruction_fixed25_summary.v1",
        "population": "predetermined balanced fixed validation 25; not full validation or sealed test",
        "metric_domains": {
            "mae_norm": "normalized model domain [-1,1]; lower is better",
            "mse_norm": "normalized model domain [-1,1]; lower is better",
            "psnr_img_db": "clamped image domain [0,1], data_range=1; higher is better",
            "ssim_img": "clamped image domain [0,1], 11x11 Gaussian sigma=1.5; higher is better",
        },
        "quantile_method": "NumPy linear",
        "standard_deviation": "population (ddof=0)",
        "archive_precision": "FP16 reconstruction archives converted to FP32 for metrics",
        "models": summaries,
        "paired_deltas": paired_deltas,
    }
    return rows, summary


def _descriptive_summary(values: ArrayF64) -> dict[str, int | float]:
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Summary values must be a finite nonempty vector")
    q1, median, q3 = np.quantile(values, (0.25, 0.5, 0.75), method="linear")
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "std_population": float(values.std(ddof=0)),
        "minimum": float(values.min()),
        "q1": float(q1),
        "median": float(median),
        "q3": float(q3),
        "maximum": float(values.max()),
    }


def assemble_dashboard_data(
    contract: Mapping[str, Any],
    fixed: Fixed25Data,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> DashboardData:
    """Assemble and validate full archived train/validation/fixed-25 histories."""
    sessions = cast("list[dict[str, Any]]", contract["sessions"])
    expected_boundaries = np.asarray(
        contract["successful_update_counter"]["validation_boundaries"],
        dtype=np.float64,
    )
    committed: dict[str, dict[int, list[dict[str, str]]]] = {
        model: {} for model in EXPECTED_MODELS
    }
    validation_rows: dict[str, dict[int, list[dict[str, str]]]] = {
        model: {} for model in EXPECTED_MODELS
    }
    equivariance_rows: dict[str, dict[int, list[dict[str, str]]]] = {
        model: {} for model in EXPECTED_MODELS
    }
    skipped_attempts: list[dict[str, str]] = []
    source_hashes = dict(fixed.source_hashes)
    train_fieldnames: list[str] | None = None
    physical_skip_rows: list[tuple[str, int, int, float, float, int]] = []

    for session in sessions:
        model = str(session["model"])
        root = _repo_path(str(session["path"]), repository_root=repository_root)
        train_path = root / "metrics/train_steps.csv"
        validation_path = root / "metrics/validation_metrics.csv"
        equivariance_path = root / "metrics/equivariance_25.csv"
        source_hashes[_relative_or_absolute(train_path, repository_root)] = sha256_file(
            train_path
        )
        source_hashes[_relative_or_absolute(validation_path, repository_root)] = (
            sha256_file(validation_path)
        )
        source_hashes[_relative_or_absolute(equivariance_path, repository_root)] = (
            sha256_file(equivariance_path)
        )
        with train_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"Missing train header in {train_path}")
            if train_fieldnames is None:
                train_fieldnames = list(reader.fieldnames)
            elif list(reader.fieldnames) != train_fieldnames:
                raise ValueError(f"Train schema differs in {train_path}")
            session_committed = 0
            session_skipped = 0
            for row in reader:
                counter = int(row["successful_optimizer_update_count"])
                rank = int(row["rank"])
                if (
                    not int(session["counter_first"])
                    <= counter
                    <= int(session["counter_last"])
                ):
                    raise ValueError(
                        f"Counter outside session contract in {train_path}"
                    )
                if row["amp_step_skipped"] == "0":
                    session_committed += 1
                    committed[model].setdefault(counter, []).append(row)
                    grad = float(row["grad_norm"])
                    update = float(row["param_update_norm"])
                    nonfinite = int(row["nonfinite_count"])
                    if (
                        not math.isfinite(grad)
                        or not math.isfinite(update)
                        or math.isclose(update, 0.0, rel_tol=0.0, abs_tol=0.0)
                    ):
                        physical_skip_rows.append((
                            model,
                            rank,
                            counter,
                            grad,
                            update,
                            nonfinite,
                        ))
                else:
                    session_skipped += 1
                    skipped_attempts.append({
                        "model": model,
                        "source_session": str(session["path"]),
                        **row,
                    })
            if session_committed != int(session["committed_rank_rows"]):
                raise ValueError(f"Committed row count mismatch in {train_path}")
            if session_skipped != int(session["amp_skipped_attempt_rows"]):
                raise ValueError(f"Skipped-attempt count mismatch in {train_path}")

        with validation_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["view"] == "clean":
                    validation_rows[model].setdefault(
                        int(row["optimizer_step"]), []
                    ).append(row)
        with equivariance_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["metric_name"] == "equivariance_error_25_patches":
                    equivariance_rows[model].setdefault(
                        int(row["optimizer_step"]), []
                    ).append(row)

    validate_physical_skip_rows(
        physical_skip_rows,
        expected=cast(
            "list[dict[str, Any]]",
            contract["accepted_committed_physical_skip"],
        ),
    )
    expected_skips = {
        model: sum(
            int(session["amp_skipped_attempt_rows"])
            for session in sessions
            if session["model"] == model
        )
        for model in EXPECTED_MODELS
    }
    for model, expected in expected_skips.items():
        actual = sum(row["model"] == model for row in skipped_attempts)
        if actual != expected:
            raise ValueError(
                f"Expected {expected} skipped attempts for {model}, got {actual}"
            )

    train: dict[str, dict[str, ArrayF64]] = {}
    validation: dict[str, dict[str, ArrayF64]] = {}
    fixed_psnr: dict[str, dict[str, ArrayF64]] = {}
    equivariance: dict[str, dict[str, ArrayF64]] = {}
    first_counter = int(contract["successful_update_counter"]["first"])
    last_counter = int(contract["successful_update_counter"]["last"])
    expected_counters = list(range(first_counter, last_counter + 1))
    for model in EXPECTED_MODELS:
        if sorted(committed[model]) != expected_counters:
            raise ValueError(f"Incomplete committed history for {model}")
        train_values = {
            "counter": np.asarray(expected_counters, dtype=np.float64),
            "objective": np.empty(len(expected_counters), dtype=np.float64),
            "l1": np.empty(len(expected_counters), dtype=np.float64),
            "one_minus_ssim": np.empty(len(expected_counters), dtype=np.float64),
            "learning_rate": np.empty(len(expected_counters), dtype=np.float64),
        }
        for index, counter in enumerate(expected_counters):
            rows = committed[model][counter]
            if {int(row["rank"]) for row in rows} != EXPECTED_RANKS or len(rows) != 2:
                raise ValueError(
                    f"Expected exactly two ranks at {model} counter {counter}"
                )
            train_values["objective"][index] = np.mean([
                float(row["loss"]) for row in rows
            ])
            train_values["l1"][index] = np.mean([float(row["l1_loss"]) for row in rows])
            train_values["one_minus_ssim"][index] = np.mean([
                float(row["ssim_loss"]) for row in rows
            ])
            train_values["learning_rate"][index] = np.mean([
                float(row["learning_rate"]) for row in rows
            ])
        if not all(np.isfinite(values).all() for values in train_values.values()):
            raise ValueError(f"Non-finite dashboard train values for {model}")
        train[model] = train_values

        if sorted(validation_rows[model]) != expected_boundaries.astype(int).tolist():
            raise ValueError(f"Incomplete clean validation history for {model}")
        validation_values = {
            "counter": expected_boundaries.copy(),
            "objective": np.empty(expected_boundaries.size, dtype=np.float64),
            "l1": np.empty(expected_boundaries.size, dtype=np.float64),
            "one_minus_ssim": np.empty(expected_boundaries.size, dtype=np.float64),
        }
        for index, counter_f64 in enumerate(expected_boundaries):
            counter = int(counter_f64)
            rows = validation_rows[model][counter]
            if {int(row["rank"]) for row in rows} != EXPECTED_RANKS or len(rows) != 2:
                raise ValueError(
                    f"Expected two clean validation ranks at {model} {counter}"
                )
            weights = np.asarray(
                [int(row["sample_count"]) for row in rows], dtype=np.float64
            )
            for source, target in (
                ("loss", "objective"),
                ("l1_loss", "l1"),
                ("ssim_loss", "one_minus_ssim"),
            ):
                validation_values[target][index] = np.average(
                    np.asarray([float(row[source]) for row in rows], dtype=np.float64),
                    weights=weights,
                )
        validation[model] = validation_values

        if sorted(equivariance_rows[model]) != expected_boundaries.astype(int).tolist():
            raise ValueError(f"Incomplete headline equivariance history for {model}")
        eq_values = np.empty(expected_boundaries.size, dtype=np.float64)
        for index, counter_f64 in enumerate(expected_boundaries):
            counter = int(counter_f64)
            rows = equivariance_rows[model][counter]
            if sorted(int(row["angle_degrees"]) for row in rows) != list(
                EXPECTED_ANGLES
            ):
                raise ValueError(f"Unexpected equivariance angles at {model} {counter}")
            if any(
                int(row["n"]) != FIXED25_COUNT
                or row["data_source"] != "real"
                or row["promotable"] != "true"
                for row in rows
            ):
                raise ValueError(f"Invalid equivariance metadata at {model} {counter}")
            eq_values[index] = np.mean([float(row["mean"]) for row in rows])
        equivariance[model] = {
            "counter": expected_boundaries.copy(),
            "ratio": eq_values,
        }

        psnr_values = np.empty(expected_boundaries.size, dtype=np.float64)
        model_sessions = [session for session in sessions if session["model"] == model]
        for index, counter_f64 in enumerate(expected_boundaries):
            counter = int(counter_f64)
            owner = next(
                session
                for session in model_sessions
                if int(session["counter_first"])
                <= counter
                <= int(session["counter_last"])
            )
            root = _repo_path(str(owner["path"]), repository_root=repository_root)
            path = (
                root
                / "artifacts/fixed25"
                / f"boundary_{counter:06d}"
                / "reconstruction_progress.pt"
            )
            source_hashes[_relative_or_absolute(path, repository_root)] = (
                _verify_consumed_artifact(
                    path,
                    contract,
                    repository_root=repository_root,
                )
            )
            payload = _load_torch_mapping(path)
            if int(payload.get("optimizer_step", -1)) != counter:
                raise ValueError(f"Reconstruction boundary mismatch in {path}")
            prediction = _as_tensor(payload, "reconstruction", path=path)
            _validate_image_tensor(
                prediction,
                expected_dtype=torch.float16,
                name=str(path),
            )
            psnr = psnr_per_image(
                normalized_to_image_domain(prediction),
                normalized_to_image_domain(fixed.target_norm),
            )
            if not bool(torch.isfinite(psnr).all().item()):
                raise ValueError(f"Non-finite fixed-25 PSNR in {path}")
            psnr_values[index] = float(psnr.mean().item())
        fixed_psnr[model] = {
            "counter": expected_boundaries.copy(),
            "psnr_db": psnr_values,
        }

    if train_fieldnames is None:
        raise ValueError("No train schema found")
    return DashboardData(
        train=train,
        validation=validation,
        fixed_psnr=fixed_psnr,
        equivariance=equivariance,
        skipped_attempts=skipped_attempts,
        train_fieldnames=train_fieldnames,
        source_hashes=source_hashes,
        physical_skip_disclosure={
            "events": contract["accepted_committed_physical_skip"],
            "interpretation": "accepted legacy committed physical-skip event(s); x-axis is a recorded counter",
        },
    )


def validate_physical_skip_rows(
    rows: Sequence[tuple[str, int, int, float, float, int]],
    *,
    expected: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    """Require exactly the accepted two-rank normal counter-14,007 signature."""
    expected_rows = (
        [
            ("normal", 0, 14_007, float("inf"), 0.0, 1),
            ("normal", 1, 14_007, float("inf"), 0.0, 1),
        ]
        if expected is None
        else [
            (
                str(row["model"]),
                int(row["rank"]),
                int(row["counter"]),
                float(row["grad_norm"]),
                float(row["param_update_norm"]),
                int(row["nonfinite_count"]),
            )
            for row in expected
        ]
    )
    if list(rows) != expected_rows:
        raise ValueError(f"Unexpected committed physical-skip signatures: {rows}")


def _relative_or_absolute(path: Path, repository_root: Path) -> str:
    try:
        return path.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def build_professor_metrics_package(
    *,
    input_contract: Path = DEFAULT_INPUT_CONTRACT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    repository_root: Path = REPOSITORY_ROOT,
    smoothing_window: int = DEFAULT_SMOOTHING_WINDOW,
    require_canonical_contract: bool = True,
) -> Path:
    """Build and atomically publish the complete local Spec 0044 package."""
    if smoothing_window < 3 or smoothing_window % 2 == 0:
        raise ValueError("Smoothing window must be an odd integer >= 3")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing package: {output_dir}")
    contract = load_input_contract(
        input_contract,
        require_canonical_hash=require_canonical_contract,
    )
    validate_session_layout(contract)
    fixed = load_fixed25_data(contract, repository_root=repository_root)
    metric_rows, metric_summary = compute_fixed25_metrics(fixed)
    dashboard = assemble_dashboard_data(
        contract,
        fixed,
        repository_root=repository_root,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        metrics_dir = temporary / "metrics"
        tables_dir = temporary / "tables"
        figures_dir = temporary / "figures"
        metrics_dir.mkdir()
        tables_dir.mkdir()
        figures_dir.mkdir()

        _write_csv(
            metrics_dir / "reconstruction_fixed25.csv",
            metric_rows,
            fieldnames=[
                "model",
                "selector_rank",
                "sample_id",
                "wsi_id",
                "source_label",
                "source_label_name",
                "x",
                "y",
                "mae_mse_domain",
                "psnr_ssim_domain",
                "mae_norm",
                "mse_norm",
                "psnr_img_db",
                "ssim_img",
            ],
        )
        _write_json(metrics_dir / "reconstruction_fixed25_summary.json", metric_summary)
        _write_dashboard_csv(metrics_dir / "training_dashboard_series.csv", dashboard)
        _write_csv(
            metrics_dir / "amp_skipped_attempts.csv",
            dashboard.skipped_attempts,
            fieldnames=["model", "source_session", *dashboard.train_fieldnames],
        )
        _write_tex_table(tables_dir / "metrics_summary.tex", metric_summary)
        render_metrics_boxplots(
            figures_dir / "metrics_boxplots_fixed25.png",
            metric_rows,
        )
        render_training_dashboard(
            figures_dir / "training_dashboard.png",
            dashboard,
            smoothing_window=smoothing_window,
        )
        render_reconstruction_comparison(
            figures_dir / "reconstructions_fixed25.png",
            fixed,
        )
        rotated_hashes = render_rotated_comparison(
            figures_dir / "rotated_input_vs_latent.png",
            fixed,
            contract,
            repository_root=repository_root,
        )
        dashboard.source_hashes.update(rotated_hashes)
        pca_reference = _repo_path(
            str(contract["fixed25"]["pca_reference_path"]),
            repository_root=repository_root,
        )
        pca_hash = _verify_consumed_artifact(
            pca_reference,
            contract,
            repository_root=repository_root,
        )
        (figures_dir / "latent_pca_reference.txt").write_text(
            "Existing paired Spec 0038/0040 PCA artifact; no new PCA fit.\n"
            f"path={_relative_or_absolute(pca_reference, repository_root)}\n"
            f"sha256={pca_hash}\n"
            "interpretation=qualitative spatial variation only; PCA colour/basis cannot rank models\n",
            encoding="utf-8",
        )
        dashboard.source_hashes[
            _relative_or_absolute(pca_reference, repository_root)
        ] = pca_hash

        output_hashes = {
            path.relative_to(temporary).as_posix(): sha256_file(path)
            for path in sorted(temporary.rglob("*"))
            if path.is_file() and path.name not in {"manifest.json", "status.json"}
        }
        manifest = {
            "schema": "spec0044.professor_metrics_package.v1",
            "status": "local_pass",
            "scope": "post-hoc local rendering from immutable archived VAE artifacts",
            "population_boundaries": {
                "boxplots": "predetermined balanced fixed validation 25",
                "validation_curves": "archived clean validation aggregate, 30000 patches per boundary",
                "psnr_curve": "same predetermined fixed validation 25 per boundary",
                "equivariance_curve": "same predetermined fixed validation 25, mean of 90/180/270 degrees",
                "sealed_test_use": "none",
            },
            "input_contract": {
                "path": _relative_or_absolute(input_contract, repository_root),
                "sha256": sha256_file(input_contract),
            },
            "metric_definitions": metric_summary["metric_domains"],
            "smoothing": {
                "window": smoothing_window,
                "method": "centered uniform moving average; display only",
                "raw_series_preserved": "metrics/training_dashboard_series.csv",
            },
            "boxplot_convention": "box=Q1..Q3, center=median, whiskers=observed minimum and maximum; all paired raw points shown",
            "physical_skip_disclosure": dashboard.physical_skip_disclosure,
            "amp_skipped_attempt_counts": {
                model: sum(row["model"] == model for row in dashboard.skipped_attempts)
                for model in EXPECTED_MODELS
            },
            "rotated_grid": {
                "selector_rank": int(contract["fixed25"]["rotated_grid_selector_rank"]),
                "sample_id": str(contract["fixed25"]["rotated_grid_sample_id"]),
            },
            "source_files_sha256": dict(sorted(dashboard.source_hashes.items())),
            "output_files_sha256": output_hashes,
            "generation_command": ".venv/bin/python -m eqvae.cli.render_professor_metrics",
            "limitations": [
                "Fixed-25 distributions are not full-validation or sealed-test distributions.",
                "FP16 reconstruction archives were converted to FP32 for metrics.",
                "Paired deltas are descriptive; no significance or superiority claim is made.",
                "PCA appearance is qualitative and basis/sign/scale dependent.",
            ],
        }
        manifest_path = temporary / "manifest.json"
        _write_json(manifest_path, manifest)
        manifest_hash = sha256_file(manifest_path)
        _write_json(
            temporary / "status.json",
            {
                "schema": "spec0044.professor_metrics_status.v1",
                "status": "local_pass",
                "manifest_sha256": manifest_hash,
                "output_file_count": len(output_hashes),
                "fixed25_rows": len(metric_rows),
                "dashboard_models": list(EXPECTED_MODELS),
                "validation_boundary_count_per_model": len(
                    contract["successful_update_counter"]["validation_boundaries"]
                ),
                "amp_skipped_attempt_rows": len(dashboard.skipped_attempts),
            },
        )
        if output_dir.exists():
            raise FileExistsError(f"Refusing concurrent overwrite: {output_dir}")
        temporary.rename(output_dir)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return output_dir


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _write_dashboard_csv(path: Path, dashboard: DashboardData) -> None:
    fieldnames = [
        "model",
        "series",
        "recorded_successful_update_counter",
        "objective",
        "l1_norm",
        "one_minus_ssim_img",
        "psnr_img_db",
        "latent_equivariance_ratio",
        "learning_rate",
        "n",
        "population",
    ]
    rows: list[dict[str, Any]] = []
    for model in EXPECTED_MODELS:
        train = dashboard.train[model]
        for index, counter in enumerate(train["counter"]):
            rows.append({
                "model": model,
                "series": "train_rank_mean",
                "recorded_successful_update_counter": int(counter),
                "objective": train["objective"][index],
                "l1_norm": train["l1"][index],
                "one_minus_ssim_img": train["one_minus_ssim"][index],
                "psnr_img_db": "",
                "latent_equivariance_ratio": "",
                "learning_rate": train["learning_rate"][index],
                "n": 50,
                "population": "two rank-local training batches; skipped attempts excluded",
            })
        validation = dashboard.validation[model]
        for index, counter in enumerate(validation["counter"]):
            rows.append({
                "model": model,
                "series": "clean_validation",
                "recorded_successful_update_counter": int(counter),
                "objective": validation["objective"][index],
                "l1_norm": validation["l1"][index],
                "one_minus_ssim_img": validation["one_minus_ssim"][index],
                "psnr_img_db": "",
                "latent_equivariance_ratio": "",
                "learning_rate": "",
                "n": 30_000,
                "population": "full clean validation aggregate",
            })
        for index, counter in enumerate(dashboard.fixed_psnr[model]["counter"]):
            rows.append({
                "model": model,
                "series": "fixed25_psnr",
                "recorded_successful_update_counter": int(counter),
                "objective": "",
                "l1_norm": "",
                "one_minus_ssim_img": "",
                "psnr_img_db": dashboard.fixed_psnr[model]["psnr_db"][index],
                "latent_equivariance_ratio": "",
                "learning_rate": "",
                "n": 25,
                "population": "predetermined balanced fixed validation 25",
            })
        for index, counter in enumerate(dashboard.equivariance[model]["counter"]):
            rows.append({
                "model": model,
                "series": "fixed25_headline_equivariance",
                "recorded_successful_update_counter": int(counter),
                "objective": "",
                "l1_norm": "",
                "one_minus_ssim_img": "",
                "psnr_img_db": "",
                "latent_equivariance_ratio": dashboard.equivariance[model]["ratio"][
                    index
                ],
                "learning_rate": "",
                "n": 25,
                "population": "fixed validation 25; mean of exact 90/180/270-degree ratios",
            })
    _write_csv(path, rows, fieldnames=fieldnames)


def _write_tex_table(path: Path, summary: Mapping[str, Any]) -> None:
    labels = {
        "mae_norm": "MAE (norm.) $\\downarrow$",
        "mse_norm": "MSE (norm.) $\\downarrow$",
        "psnr_img_db": "PSNR (dB) $\\uparrow$",
        "ssim_img": "SSIM $\\uparrow$",
    }
    lines = [
        "% Spec 0044: fixed validation 25 only; not full validation or sealed test.",
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "Model & Metric & Mean & Population SD & $n$ \\\\",
        "\\midrule",
    ]
    for model in EXPECTED_MODELS:
        for metric, label in labels.items():
            values = summary["models"][model][metric]
            lines.append(
                f"{model.upper()} & {label} & {values['mean']:.6f} & "
                f"{values['std_population']:.6f} & {values['n']} \\\\"
            )
    lines.extend([
        "\\midrule",
        "\\multicolumn{5}{l}{\\footnotesize MAE/MSE use normalized $[-1,1]$; PSNR/SSIM use clamped $[0,1]$.} \\\\",
        "\\multicolumn{5}{l}{\\footnotesize FP16 reconstruction archives were converted to FP32 for metric computation.} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _font(
    size: int, *, bold: bool = False
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size=size)
    except OSError:
        return ImageFont.load_default()


def _metric_arrays_from_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, ArrayF64]]:
    names = ("mae_norm", "mse_norm", "psnr_img_db", "ssim_img")
    return {
        model: {
            name: np.asarray(
                [float(row[name]) for row in rows if row["model"] == model],
                dtype=np.float64,
            )
            for name in names
        }
        for model in EXPECTED_MODELS
    }


def render_metrics_boxplots(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Render paired raw points and box summaries for the fixed validation 25."""
    arrays = _metric_arrays_from_rows(rows)
    canvas = Image.new("RGB", (1800, 1180), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (70, 40),
        "Reconstruction metrics — fixed validation 25",
        fill=INK,
        font=_font(40, bold=True),
    )
    draw.text(
        (70, 94),
        "Same 25 predetermined patches for both models • raw paired observations + population summaries",
        fill=MUTED,
        font=_font(23),
    )
    panels = [
        ("mae_norm", "MAE • normalized [-1,1] • lower is better"),
        ("mse_norm", "MSE • normalized [-1,1] • lower is better"),
        ("psnr_img_db", "PSNR (dB) • image [0,1] • higher is better"),
        ("ssim_img", "SSIM • image [0,1] • higher is better"),
    ]
    positions = [(60, 155), (920, 155), (60, 650), (920, 650)]
    for (metric, title), (left, top) in zip(panels, positions, strict=True):
        _draw_box_panel(
            draw,
            (left, top, left + 820, top + 445),
            title,
            arrays["normal"][metric],
            arrays["so2"][metric],
        )
    draw.text(
        (70, 1115),
        "Boxes: Q1–Q3; center: median; whiskers: observed min/max; every paired raw point is shown.",
        fill=MUTED,
        font=_font(17),
    )
    draw.text(
        (70, 1140),
        "Fixed-25 validation only — not a full-validation distribution or sealed-test comparison.",
        fill="#9f1239",
        font=_font(19, bold=True),
    )
    canvas.save(path, format="PNG", optimize=True)


def _draw_box_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    normal: ArrayF64,
    so2: ArrayF64,
) -> None:
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=18, fill=PANEL, outline="#d1d5db", width=2)
    draw.text((left + 24, top + 18), title, fill=INK, font=_font(22, bold=True))
    plot_left, plot_top, plot_right, plot_bottom = (
        left + 105,
        top + 80,
        right - 35,
        bottom - 65,
    )
    all_values = np.concatenate((normal, so2))
    value_min = float(all_values.min())
    value_max = float(all_values.max())
    padding = max((value_max - value_min) * 0.12, abs(value_max) * 0.02, 1e-6)
    low, high = value_min - padding, value_max + padding
    for tick in range(5):
        value = low + (high - low) * tick / 4.0
        y = _map_y(value, low, high, plot_top, plot_bottom)
        draw.line((plot_left, y, plot_right, y), fill=GRID, width=1)
        draw.text((left + 12, y - 10), f"{value:.3g}", fill=MUTED, font=_font(16))
    x_normal = plot_left + (plot_right - plot_left) * 0.32
    x_so2 = plot_left + (plot_right - plot_left) * 0.72
    for index, (a, b) in enumerate(zip(normal, so2, strict=True)):
        jitter = ((index * 17) % 19 - 9) * 1.5
        y_a = _map_y(float(a), low, high, plot_top, plot_bottom)
        y_b = _map_y(float(b), low, high, plot_top, plot_bottom)
        draw.line(
            (x_normal + jitter, y_a, x_so2 + jitter, y_b), fill="#cbd5e1", width=1
        )
        draw.ellipse(
            (x_normal + jitter - 3, y_a - 3, x_normal + jitter + 3, y_a + 3),
            fill=PLOT_NORMAL,
        )
        draw.ellipse(
            (x_so2 + jitter - 3, y_b - 3, x_so2 + jitter + 3, y_b + 3), fill=PLOT_SO2
        )
    _draw_one_box(draw, x_normal, normal, low, high, plot_top, plot_bottom, PLOT_NORMAL)
    _draw_one_box(draw, x_so2, so2, low, high, plot_top, plot_bottom, PLOT_SO2)
    draw.text(
        (round(x_normal) - 55, plot_bottom + 18),
        "Normal VAE",
        fill=PLOT_NORMAL,
        font=_font(18, bold=True),
    )
    draw.text(
        (round(x_so2) - 48, plot_bottom + 18),
        "SO(2) VAE",
        fill=PLOT_SO2,
        font=_font(18, bold=True),
    )
    draw.text((right - 165, top + 50), "n=25 paired", fill=MUTED, font=_font(16))


def _draw_one_box(
    draw: ImageDraw.ImageDraw,
    x: float,
    values: ArrayF64,
    low: float,
    high: float,
    top: int,
    bottom: int,
    color: str,
) -> None:
    minimum, q1, median, q3, maximum = np.quantile(
        values,
        (0.0, 0.25, 0.5, 0.75, 1.0),
        method="linear",
    )
    ys = [
        _map_y(float(value), low, high, top, bottom)
        for value in (minimum, q1, median, q3, maximum)
    ]
    y_min, y_q1, y_med, y_q3, y_max = ys
    draw.line((x, y_max, x, y_q3), fill=color, width=4)
    draw.line((x, y_q1, x, y_min), fill=color, width=4)
    draw.line((x - 22, y_max, x + 22, y_max), fill=color, width=4)
    draw.line((x - 22, y_min, x + 22, y_min), fill=color, width=4)
    draw.rectangle((x - 42, y_q3, x + 42, y_q1), outline=color, width=5)
    draw.line((x - 42, y_med, x + 42, y_med), fill=color, width=5)


def _map_y(value: float, low: float, high: float, top: int, bottom: int) -> int:
    return round(bottom - (value - low) / (high - low) * (bottom - top))


def centered_moving_average(
    x: ArrayF64, y: ArrayF64, window: int
) -> tuple[ArrayF64, ArrayF64]:
    """Return a centered uniform display smoother without changing source data."""
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError("Smoother requires equal one-dimensional x/y arrays")
    if window < 3 or window % 2 == 0 or window > y.size:
        raise ValueError("Smoothing window must be odd and fit the series")
    kernel = np.full(window, 1.0 / window, dtype=np.float64)
    smoothed = np.convolve(y, kernel, mode="valid")
    half = window // 2
    return x[half : x.size - half], smoothed


def render_training_dashboard(
    path: Path,
    dashboard: DashboardData,
    *,
    smoothing_window: int,
) -> None:
    """Render the six-panel professor-requested training/evaluation dashboard."""
    canvas = Image.new("RGB", (1900, 1320), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (65, 35),
        "Normal VAE vs continuous SO(2) VAE — archived training evidence",
        fill=INK,
        font=_font(39, bold=True),
    )
    draw.text(
        (65, 88),
        f"Train display smoother: centered {smoothing_window}-counter mean • validation markers: 30,000 patches • fixed metrics: 25 patches",
        fill=MUTED,
        font=_font(21),
    )
    panel_boxes = [
        (55, 145, 925, 500),
        (975, 145, 1845, 500),
        (55, 535, 925, 890),
        (975, 535, 1845, 890),
        (55, 925, 925, 1280),
        (975, 925, 1845, 1280),
    ]
    panels: list[tuple[str, str, str, bool]] = [
        ("Composite objective • lower is better", "objective", "validation", False),
        ("L1 • normalized domain • lower is better", "l1", "validation", False),
        (
            "1 − SSIM • image domain • lower is better",
            "one_minus_ssim",
            "validation",
            False,
        ),
        (
            "PSNR (dB) • fixed validation 25 • higher is better",
            "psnr_db",
            "fixed_psnr",
            False,
        ),
        (
            "Latent equivariance ratio • fixed 25 • log scale • lower is better",
            "ratio",
            "equivariance",
            True,
        ),
        ("Learning rate", "learning_rate", "train_only", False),
    ]
    for box, (title, metric, kind, log_y) in zip(panel_boxes, panels, strict=True):
        series: list[PlotSeries] = []
        if kind in {"validation", "train_only"}:
            for model, color in (("normal", PLOT_NORMAL), ("so2", PLOT_SO2)):
                x, y = centered_moving_average(
                    dashboard.train[model]["counter"],
                    dashboard.train[model][metric],
                    smoothing_window,
                )
                x, y = _decimate(x, y, max_points=1200)
                series.append(
                    PlotSeries(x, y, color, f"{model} train", dashed=model == "so2")
                )
            if kind == "validation":
                for model, color in (
                    ("normal", PLOT_VALIDATION_NORMAL),
                    ("so2", PLOT_VALIDATION_SO2),
                ):
                    series.append(
                        PlotSeries(
                            dashboard.validation[model]["counter"],
                            dashboard.validation[model][metric],
                            color,
                            f"{model} validation",
                            dashed=model == "so2",
                            markers=True,
                        )
                    )
        else:
            source = (
                dashboard.fixed_psnr if kind == "fixed_psnr" else dashboard.equivariance
            )
            for model, color in (("normal", PLOT_NORMAL), ("so2", PLOT_SO2)):
                series.append(
                    PlotSeries(
                        source[model]["counter"],
                        source[model][metric],
                        color,
                        model,
                        dashed=model == "so2",
                        markers=True,
                    )
                )
        _draw_line_panel(draw, box, title, series, log_y=log_y)
    draw.text(
        (70, 1288),
        "X-axis: recorded successful-update counter. Normal counter 14,007 is the disclosed accepted physical-skip event.",
        fill="#9f1239",
        font=_font(18, bold=True),
    )
    canvas.save(path, format="PNG", optimize=True)


def _decimate(
    x: ArrayF64, y: ArrayF64, *, max_points: int
) -> tuple[ArrayF64, ArrayF64]:
    if x.size <= max_points:
        return x, y
    indices = np.linspace(0, x.size - 1, max_points).round().astype(np.int64)
    return x[indices], y[indices]


def _draw_line_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    series: Sequence[PlotSeries],
    *,
    log_y: bool,
) -> None:
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=16, fill=PANEL, outline="#d1d5db", width=2)
    draw.text((left + 22, top + 15), title, fill=INK, font=_font(20, bold=True))
    plot_left, plot_top, plot_right, plot_bottom = (
        left + 92,
        top + 66,
        right - 26,
        bottom - 64,
    )
    transformed: list[ArrayF64] = []
    for item in series:
        if not np.isfinite(item.y).all():
            raise ValueError(f"Non-finite plot series: {item.label}")
        if log_y:
            if np.any(item.y <= 0.0):
                raise ValueError(f"Log plot series is not positive: {item.label}")
            transformed.append(np.log10(item.y))
        else:
            transformed.append(item.y)
    y_all = np.concatenate(transformed)
    y_low, y_high = float(y_all.min()), float(y_all.max())
    pad = max((y_high - y_low) * 0.10, abs(y_high) * 0.01, 1e-10)
    y_low -= pad
    y_high += pad
    if not log_y and float(y_all.min()) >= 0.0:
        y_low = max(0.0, y_low)
    x_low = min(float(item.x.min()) for item in series)
    x_high = max(float(item.x.max()) for item in series)
    if math.isclose(x_low, x_high, rel_tol=0.0, abs_tol=0.0):
        x_low -= 0.5
        x_high += 0.5
    for tick in range(5):
        y_value = y_low + (y_high - y_low) * tick / 4.0
        y = _map_y(y_value, y_low, y_high, plot_top, plot_bottom)
        draw.line((plot_left, y, plot_right, y), fill=GRID, width=1)
        label_value = 10.0**y_value if log_y else y_value
        draw.text((left + 8, y - 9), f"{label_value:.3g}", fill=MUTED, font=_font(14))
    for tick in range(5):
        counter = x_low + (x_high - x_low) * tick / 4.0
        x = round(
            plot_left + (counter - x_low) / (x_high - x_low) * (plot_right - plot_left)
        )
        draw.line((x, plot_top, x, plot_bottom), fill=GRID, width=1)
        draw.text(
            (x - 20, plot_bottom + 10),
            f"{counter / 1000:.0f}k",
            fill=MUTED,
            font=_font(14),
        )
    for item, y_values in zip(series, transformed, strict=True):
        points = [
            (
                round(
                    plot_left
                    + (float(x) - x_low) / (x_high - x_low) * (plot_right - plot_left)
                ),
                _map_y(float(y), y_low, y_high, plot_top, plot_bottom),
            )
            for x, y in zip(item.x, y_values, strict=True)
        ]
        if item.dashed:
            _draw_dashed_polyline(draw, points, fill=item.color, width=3)
        else:
            draw.line(points, fill=item.color, width=3)
        if item.markers:
            for x, y in points:
                draw.ellipse(
                    (x - 4, y - 4, x + 4, y + 4),
                    fill=BACKGROUND,
                    outline=item.color,
                    width=2,
                )
    legend_x = left + 105
    legend_y = bottom - 39
    for index, item in enumerate(series):
        x = legend_x + (index % 2) * 330
        y = legend_y + (index // 2) * 21
        if item.dashed:
            _draw_dashed_polyline(
                draw, [(x, y + 7), (x + 35, y + 7)], fill=item.color, width=3
            )
        else:
            draw.line((x, y + 7, x + 35, y + 7), fill=item.color, width=3)
        if item.markers:
            draw.ellipse(
                (x + 13, y + 3, x + 21, y + 11),
                fill=BACKGROUND,
                outline=item.color,
                width=2,
            )
        draw.text((x + 43, y), item.label, fill=INK, font=_font(14))


def _draw_dashed_polyline(
    draw: ImageDraw.ImageDraw,
    points: Sequence[tuple[int, int]],
    *,
    fill: str,
    width: int,
) -> None:
    dash_length = 9.0
    gap_length = 6.0
    for start, end in itertools.pairwise(points):
        dx = float(end[0] - start[0])
        dy = float(end[1] - start[1])
        length = math.hypot(dx, dy)
        if math.isclose(length, 0.0, rel_tol=0.0, abs_tol=0.0):
            continue
        offset = 0.0
        while offset < length:
            dash_end = min(offset + dash_length, length)
            first = (
                round(start[0] + dx * offset / length),
                round(start[1] + dy * offset / length),
            )
            second = (
                round(start[0] + dx * dash_end / length),
                round(start[1] + dy * dash_end / length),
            )
            draw.line((first, second), fill=fill, width=width)
            offset += dash_length + gap_length


def _tensor_to_rgb(tensor: Tensor, *, normalized: bool) -> Image.Image:
    values = (
        normalized_to_image_domain(tensor.unsqueeze(0))[0]
        if normalized
        else tensor.to(torch.float32).div(255.0)
    )
    array = (
        values
        .mul(255.0)
        .round()
        .clamp(0.0, 255.0)
        .to(torch.uint8)
        .permute(1, 2, 0)
        .numpy()
    )
    return Image.fromarray(cast("ArrayU8", array), mode="RGB")


def render_reconstruction_comparison(path: Path, fixed: Fixed25Data) -> None:
    """Render three aligned 5x5 grids: originals, normal, and SO(2)."""
    canvas = Image.new("RGB", (1840, 760), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (55, 30),
        "Final reconstructions — same predetermined fixed validation 25",
        fill=INK,
        font=_font(38, bold=True),
    )
    tile = 102
    gap = 7
    starts = [55, 645, 1235]
    titles = ["Original", "Normal VAE", "Continuous SO(2) VAE"]
    tensors = [
        fixed.originals_uint8,
        fixed.predictions["normal"],
        fixed.predictions["so2"],
    ]
    normalized = [False, True, True]
    for start, title, values, is_normalized in zip(
        starts, titles, tensors, normalized, strict=True
    ):
        draw.text((start, 92), title, fill=INK, font=_font(25, bold=True))
        for index in range(FIXED25_COUNT):
            row, column = divmod(index, 5)
            image = _tensor_to_rgb(values[index], normalized=is_normalized).resize(
                (tile, tile),
                Image.Resampling.LANCZOS,
            )
            x = start + column * (tile + gap)
            y = 132 + row * (tile + gap)
            canvas.paste(image, (x, y))
            draw.rectangle(
                (x, y, x + tile - 1, y + tile - 1), outline="#d1d5db", width=1
            )
    draw.text(
        (55, 710),
        "Ordered by frozen selector rank (five patches per source histotype); FP16 archives shown after [−1,1] → [0,1] clamping.",
        fill=MUTED,
        font=_font(19),
    )
    canvas.save(path, format="PNG", optimize=True)


def render_rotated_comparison(
    path: Path,
    fixed: Fixed25Data,
    contract: Mapping[str, Any],
    *,
    repository_root: Path,
) -> dict[str, str]:
    """Render the predetermined patch's rotated-input versus latent-action paths."""
    rank = int(contract["fixed25"]["rotated_grid_selector_rank"])
    angles = (0, *EXPECTED_ANGLES)
    source_hashes: dict[str, str] = {}
    values: dict[str, dict[int, tuple[Tensor, Tensor, Tensor]]] = {
        model: {} for model in EXPECTED_MODELS
    }
    for model in EXPECTED_MODELS:
        root = fixed.final_roots[model]
        clean = fixed.predictions[model][rank]
        values[model][0] = (fixed.target_norm[rank], clean, clean)
        for angle in EXPECTED_ANGLES:
            source = (
                root
                / "artifacts/fixed25"
                / f"boundary_{FINAL_COUNTER:06d}"
                / f"rotated_angle_{angle}.pt"
            )
            source_hashes[_relative_or_absolute(source, repository_root)] = (
                _verify_consumed_artifact(
                    source,
                    contract,
                    repository_root=repository_root,
                )
            )
            payload = _load_torch_mapping(source)
            if int(payload.get("angle_degrees", -1)) != angle:
                raise ValueError(f"Rotation angle mismatch in {source}")
            archived_tensors = tuple(
                _as_tensor(payload, key, path=source)
                for key in (
                    "ground_truth",
                    "rotated_input_reconstruction",
                    "rotated_embedding_reconstruction",
                )
            )
            for tensor in archived_tensors:
                _validate_image_tensor(
                    tensor,
                    expected_dtype=torch.float16,
                    name=str(source),
                )
            tensors = tuple(
                tensor.to(torch.float32)[rank] for tensor in archived_tensors
            )
            expected_ground_truth = torch.rot90(
                fixed.target_norm[rank], angle // 90, dims=(-2, -1)
            )
            if not torch.equal(
                tensors[0], expected_ground_truth.to(torch.float16).to(torch.float32)
            ):
                raise ValueError(f"Rotated ground truth mismatch in {source}")
            values[model][angle] = cast("tuple[Tensor, Tensor, Tensor]", tensors)
    for angle in angles:
        if not torch.equal(values["normal"][angle][0], values["so2"][angle][0]):
            raise ValueError(f"Model ground truths differ at {angle} degrees")

    canvas = Image.new("RGB", (1770, 1430), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    identity = fixed.identities[rank]
    draw.text(
        (45, 28),
        "Rotated input vs transformed latent — predetermined fixed patch",
        fill=INK,
        font=_font(36, bold=True),
    )
    draw.text(
        (45, 78),
        f"selector rank {rank} • {identity['sample_id']} • exact quarter turns",
        fill=MUTED,
        font=_font(20),
    )
    columns = [
        "Ground truth",
        "Normal\nrotated input",
        "Normal\nlatent action",
        "SO(2)\nrotated input",
        "SO(2)\nlatent action",
    ]
    tile = 248
    gap = 42
    start_x = 160
    for column, title in enumerate(columns):
        x = start_x + column * (tile + gap)
        for line_index, line in enumerate(title.split("\n")):
            draw.text(
                (x + 8, 122 + line_index * 25),
                line,
                fill=INK,
                font=_font(19, bold=True),
            )
    for row, angle in enumerate(angles):
        y = 188 + row * 298
        draw.text((45, y + 104), f"{angle}°", fill=INK, font=_font(25, bold=True))
        ground_truth = values["normal"][angle][0]
        row_tensors = [
            ground_truth,
            values["normal"][angle][1],
            values["normal"][angle][2],
            values["so2"][angle][1],
            values["so2"][angle][2],
        ]
        for column, tensor in enumerate(row_tensors):
            x = start_x + column * (tile + gap)
            image = _tensor_to_rgb(tensor, normalized=True).resize(
                (tile, tile), Image.Resampling.LANCZOS
            )
            canvas.paste(image, (x, y))
            draw.rectangle(
                (x, y, x + tile - 1, y + tile - 1), outline="#d1d5db", width=2
            )
            if column > 0:
                mae = float((tensor - ground_truth).abs().mean().item())
                draw.text(
                    (x + 5, y + tile + 5),
                    f"MAE to target {mae:.4f}",
                    fill=MUTED,
                    font=_font(15),
                )
                if column in {2, 4}:
                    input_column = 1 if column == 2 else 3
                    path_delta = float(
                        (tensor - row_tensors[input_column]).abs().mean().item()
                    )
                    draw.text(
                        (x + 5, y + tile + 24),
                        f"path Δ MAE {path_delta:.4f}",
                        fill="#7c3aed",
                        font=_font(15, bold=True),
                    )
    draw.text(
        (45, 1390),
        "Annotations are normalized-domain MAE. 0° paths coincide by definition; this is fixed validation, not sealed test.",
        fill="#9f1239",
        font=_font(19, bold=True),
    )
    canvas.save(path, format="PNG", optimize=True)
    return source_hashes
