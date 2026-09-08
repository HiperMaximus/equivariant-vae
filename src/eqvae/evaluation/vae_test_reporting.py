# Copyright 2026 HiperMaximus
# pyright: reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
# ruff: noqa: DOC201, DOC501, E501, EM101, EM102, PLR0913, PLR0914, PLR0916, PLW0717, TRY003
"""Receipt-authenticated post-retrieval reporting for Spec 0045."""

from __future__ import annotations

import csv
import gzip
import io
import json
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

from PIL import Image, ImageDraw, ImageFont

from eqvae.evaluation.vae_test import (
    REDACTED_LOCATION_HEADER,
    REMOTE_METRIC_HEADER,
    TEST_ROW_COUNT,
    TEST_WSI_COUNT,
    TestLocation,
    load_test_locations,
    sha256_file,
)
from eqvae.evaluation.vae_test_scoring import (
    BRANCHES,
    DIAGNOSES,
    METRICS,
    TEST_VECTOR_PATH,
    OracleLabel,
    load_oracle_labels,
    load_remote_metric_rows,
    score_vae_test_metrics,
    verify_test_vector,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

ORACLE_SHA256: Final = (
    "1d0e4059f469d350ff3960cc10208221548f6afdfc1788e40e6d5da7829806cc"
)
LOCATION_IDENTITY_SHA256: Final = (
    "0f7fc4f01961bdb4f2dd71d7c1b7cd0455e6bb175d7dff095a88bfaa27e34fd5"
)
LATENT_KERNEL_SOURCES: Final = tuple(
    f"maximusshtefan/eqvae-ubc-ocean-latent-run-{run:02d}" for run in range(1, 6)
)
REMOTE_OUTPUT_FILES: Final = {
    "per_patch_metrics.csv.gz",
    "run_contract.json",
    "runtime.json",
    "wsi_evidence.json",
}
FROZEN_SCORING_FILES: Final = {
    "spec_sha256": "docs/specs/0045-frozen-vae-test-reconstruction-evaluation.md",
    "scorer_sha256": "src/eqvae/evaluation/vae_test_scoring.py",
    "scorer_vector_sha256": "docs/data/spec0045_vae_test_scorer_vector.json",
    "evaluator_sha256": "src/eqvae/evaluation/vae_test_runtime.py",
    "evaluation_contract_sha256": "src/eqvae/evaluation/vae_test.py",
    "reporter_sha256": "src/eqvae/evaluation/vae_test_reporting.py",
}
KAGGLE_VERSIONED_REFERENCE_PARTS: Final = 3
INK: Final = "#111827"
MUTED: Final = "#64748b"
GRID: Final = "#e2e8f0"
NORMAL: Final = "#2563eb"
SO2: Final = "#ea580c"


def write_exclusive_json(path: Path, value: object) -> None:
    """Atomically create, but never replace, a pre-score authority claim."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            payload = json.dumps(
                value,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
            handle.write(payload + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def score_retrieved_vae_test_output(
    *,
    remote_output_root: Path,
    launch_receipt_path: Path,
    input_receipt_path: Path,
    input_contract_path: Path,
    kernel_root: Path,
    oracle_path: Path,
    output_root: Path,
    normalization_amendment_path: Path,
    expected_normalization_amendment_sha256: str,
) -> dict[str, object]:
    """Authenticate predictions, claim them exclusively, then open labels and score."""
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite {output_root}")
    staging = output_root.with_name(f".{output_root.name}.scoring")
    if staging.exists():
        raise FileExistsError(f"Stale scoring directory exists: {staging}")
    repository = _repo_root(Path(__file__).resolve())
    scorer_path = repository / "src/eqvae/evaluation/vae_test_scoring.py"
    reporter_path = Path(__file__).resolve()
    vector_path = repository / TEST_VECTOR_PATH
    spec_path = (
        repository / "docs/specs/0045-frozen-vae-test-reconstruction-evaluation.md"
    )
    input_contract = _read_object(input_contract_path)
    amendment = _validate_normalization_amendment(
        path=normalization_amendment_path,
        expected_sha256=expected_normalization_amendment_sha256,
        current_reporter_sha256=sha256_file(reporter_path),
        remote_reporter_sha256=cast("str", input_contract.get("reporter_sha256")),
        launch_receipt_sha256=sha256_file(launch_receipt_path),
    )
    verify_test_vector(vector_path)
    authenticated = _authenticate_remote_output(
        remote_output_root=remote_output_root,
        launch_receipt_path=launch_receipt_path,
        input_receipt_path=input_receipt_path,
        input_contract_path=input_contract_path,
        kernel_root=kernel_root,
        normalization_amendment=amendment,
    )
    metric_path = cast("Path", authenticated["metric_path"])
    input_contract = cast("Mapping[str, object]", authenticated["contract"])
    pre_score = {
        "schema_version": "spec0045.pre_score_claim.v1",
        "policy": "no_remote_retry_after_this_claim",
        "remote_launch_receipt_sha256": sha256_file(launch_receipt_path),
        "input_dataset_receipt_sha256": sha256_file(input_receipt_path),
        "input_contract_sha256": sha256_file(input_contract_path),
        "remote_output_receipt_sha256": sha256_file(
            remote_output_root / "kaggle_output_receipt.json",
        ),
        "remote_manifest_sha256": sha256_file(
            cast("Path", authenticated["payload_root"]) / "manifest.json",
        ),
        "remote_metric_sha256": sha256_file(metric_path),
        "remote_metric_bytes": metric_path.stat().st_size,
        "oracle_sha256": ORACLE_SHA256,
        "scorer_sha256": sha256_file(scorer_path),
        "scorer_vector_sha256": sha256_file(vector_path),
        "original_reporter_sha256": input_contract["reporter_sha256"],
        "amended_local_reporter_sha256": sha256_file(reporter_path),
        "normalization_amendment_sha256": (expected_normalization_amendment_sha256),
        "spec_sha256": sha256_file(spec_path),
        "kernel_files": _kernel_upload_records(kernel_root),
    }
    external_claim = output_root.with_name(f"{output_root.name}.pre_score_claim.json")
    write_exclusive_json(external_claim, pre_score)

    staging.mkdir(parents=True)
    try:
        _write_json(staging / "pre_score_claim.json", pre_score)
        remote_rows = load_remote_metric_rows(metric_path)
        population = cast("Mapping[str, object]", input_contract["population"])
        redacted = cast("Mapping[str, object]", population["redacted_location"])
        locations = load_test_locations(
            input_contract_path.parent / "vae_test_locations.csv",
            expected_sha256=cast("str", redacted["sha256"]),
        )
        _validate_metric_location_rows(remote_rows, locations)
        # This is deliberately the first read of the local diagnosis oracle. The
        # immutable prediction/scorer claim therefore exists before label access.
        labels = load_oracle_labels(oracle_path, expected_sha256=ORACLE_SHA256)
        scored = score_vae_test_metrics(
            remote_rows=remote_rows,
            labels_by_atlas_row=labels,
        )
        bootstrap_rows = cast("list[dict[str, object]]", scored.pop("bootstrap_rows"))
        _write_json(staging / "metrics/overall_summary.json", scored)
        _write_joined_metrics(
            staging / "metrics/per_patch_metrics_joined.csv.gz",
            remote_rows=remote_rows,
            labels=labels,
        )
        _write_per_wsi(staging / "metrics/per_wsi_metrics.csv", scored=scored)
        _write_per_diagnosis(
            staging / "metrics/per_diagnosis_summary.csv",
            scored=scored,
        )
        _write_bootstrap(
            staging / "metrics/paired_bootstrap.csv.gz",
            rows=bootstrap_rows,
        )
        _write_tex(staging / "tables/vae_test_metrics.tex", scored=scored)
        _render_paired_wsi(
            staging / "figures/paired_wsi_mae.png",
            scored=scored,
        )
        _render_patch_distributions(
            staging / "figures/patch_metric_boxplots.png",
            scored=scored,
        )
        _render_diagnosis_mae(
            staging / "figures/patch_mae_by_diagnosis.png",
            scored=scored,
        )
        artifacts = {
            path.relative_to(staging).as_posix(): _file_record(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        manifest = {
            "schema_version": "spec0045.scoring_manifest.v1",
            "status": "complete",
            "artifacts": artifacts,
        }
        _write_json(staging / "manifest.json", manifest)
        status = {
            "schema_version": "spec0045.scoring_status.v1",
            "status": "complete",
            "primary_endpoint": scored["primary_endpoint"],
            "row_count": TEST_ROW_COUNT,
            "wsi_count": TEST_WSI_COUNT,
            "manifest_sha256": sha256_file(staging / "manifest.json"),
            "limitations": scored["limitations"],
        }
        _write_json(staging / "status.json", status)
        staging.replace(output_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return status


def _authenticate_remote_output(
    *,
    remote_output_root: Path,
    launch_receipt_path: Path,
    input_receipt_path: Path,
    input_contract_path: Path,
    kernel_root: Path,
    normalization_amendment: Mapping[str, object],
) -> dict[str, object]:
    input_contract = _read_object(input_contract_path)
    input_sha256 = sha256_file(input_contract_path)
    dataset_reference = cast("str", input_contract.get("dataset_reference"))
    if (
        input_contract.get("schema_version") != "spec0045.vae_test_input.v1"
        or input_contract.get("status") != "complete"
        or input_contract.get("kernel_sources") != list(LATENT_KERNEL_SOURCES)
    ):
        raise ValueError("Local Spec 0045 input contract differs")
    _validate_frozen_scoring_files(
        input_contract,
        repository=_repo_root(Path(__file__)),
        normalization_amendment=normalization_amendment,
    )
    input_receipt = _read_object(input_receipt_path)
    if (
        input_receipt.get("schema_version") != "spec0045.input_dataset_receipt.v1"
        or input_receipt.get("status") != "verified"
        or input_receipt.get("visibility") != "private"
        or input_receipt.get("dataset_version") != 1
        or input_receipt.get("dataset_reference") != dataset_reference
        or input_receipt.get("input_contract_sha256") != input_sha256
    ):
        raise ValueError("Spec 0045 immutable input receipt differs")
    launch = _read_object(launch_receipt_path)
    requested_kernel_id = cast("str", input_contract["kernel_id"])
    expected_reference = cast(
        "str",
        normalization_amendment.get("accepted_kernel_reference"),
    )
    accepted_parts = expected_reference.split("/")
    if (
        len(accepted_parts) != KAGGLE_VERSIONED_REFERENCE_PARTS
        or not accepted_parts[2].isdigit()
    ):
        raise ValueError("Spec 0045 amended accepted reference differs")
    accepted_kernel_id = "/".join(accepted_parts[:2])
    accepted_version = int(accepted_parts[2])
    actor = dataset_reference.split("/", maxsplit=1)[0]
    expected_sources = {
        "competition_sources": ["UBC-OCEAN"],
        "dataset_sources": [dataset_reference],
        "kernel_sources": list(LATENT_KERNEL_SOURCES),
        "model_sources": [],
    }
    expected_core_files = _kernel_upload_records(kernel_root)
    source_files = cast("Mapping[str, object]", launch.get("source_files"))
    upload_files = cast("Mapping[str, object]", launch.get("upload_files"))
    metadata_sha256 = sha256_file(kernel_root / "kernel-metadata.json")
    if (
        set(launch)
        != {
            "schema_version",
            "actor",
            "original_kernel_id",
            "requested_kernel_id",
            "kernel_id",
            "accepted_version",
            "kernel_reference",
            "source_locators",
            "source_metadata_sha256",
            "source_files",
            "upload_metadata_sha256",
            "upload_files",
        }
        or launch.get("schema_version") != "eqvae.kaggle_kernel_launch.v1"
        or launch.get("actor") != actor
        or launch.get("original_kernel_id") != requested_kernel_id
        or launch.get("requested_kernel_id") != requested_kernel_id
        or launch.get("accepted_version") != accepted_version
        or launch.get("kernel_id") != accepted_kernel_id
        or launch.get("kernel_reference") != expected_reference
        or launch.get("source_locators") != expected_sources
        or launch.get("source_metadata_sha256") != metadata_sha256
        or launch.get("upload_metadata_sha256") != metadata_sha256
        or upload_files != source_files
        or any(
            source_files.get(name) != record
            for name, record in expected_core_files.items()
        )
    ):
        raise ValueError("Spec 0045 launch receipt differs")

    receipt_path = remote_output_root / "kaggle_output_receipt.json"
    receipt = _read_object(receipt_path)
    if (
        receipt.get("schema_version") != "eqvae.kaggle_download.v1"
        or receipt.get("resource_kind") != "kernel"
        or receipt.get("resource_reference") != launch["kernel_reference"]
    ):
        raise ValueError("Spec 0045 output receipt identity differs")
    receipt_files = cast("Mapping[str, object]", receipt.get("files"))
    observed = _file_records(remote_output_root, exclude={receipt_path.name})
    if receipt_files != observed:
        raise ValueError("Spec 0045 downloaded output bytes differ")
    payload_root = remote_output_root / "vae_test_reconstruction"
    manifest = _read_object(payload_root / "manifest.json")
    status = _read_object(payload_root / "status.json")
    runtime = _read_object(payload_root / "runtime.json")
    run_contract = _read_object(payload_root / "run_contract.json")
    if (
        manifest.get("schema_version") != "spec0045.remote_manifest.v1"
        or manifest.get("status") != "complete"
        or manifest.get("row_count") != TEST_ROW_COUNT
        or manifest.get("wsi_count") != TEST_WSI_COUNT
        or set(cast("Mapping[str, object]", manifest.get("output_files")))
        != REMOTE_OUTPUT_FILES
        or status.get("schema_version") != "spec0045.remote_status.v1"
        or status.get("status") != "pass"
        or status.get("optimizer_updates") != 0
        or status.get("manifest_sha256") != sha256_file(payload_root / "manifest.json")
        or runtime.get("precision") != "FP32"
        or runtime.get("optimizer_updates") != 0
        or runtime.get("patch_count") != TEST_ROW_COUNT
        or run_contract.get("input_contract_sha256") != input_sha256
        or run_contract.get("dataset_reference") != dataset_reference
        or run_contract.get("kernel_sources") != list(LATENT_KERNEL_SOURCES)
        or run_contract.get("location_identity_sha256") != LOCATION_IDENTITY_SHA256
        or run_contract.get("source_contract") != input_contract
    ):
        raise ValueError("Spec 0045 completed remote contract differs")
    output_files = cast("Mapping[str, object]", manifest["output_files"])
    for name in REMOTE_OUTPUT_FILES:
        if output_files[name] != _file_record(payload_root / name):
            raise ValueError(f"Spec 0045 remote artifact differs: {name}")
    return {
        "payload_root": payload_root,
        "metric_path": payload_root / "per_patch_metrics.csv.gz",
        "contract": input_contract,
    }


def _validate_frozen_scoring_files(
    input_contract: Mapping[str, object],
    *,
    repository: Path,
    normalization_amendment: Mapping[str, object],
) -> None:
    mismatches = [
        field
        for field, relative in FROZEN_SCORING_FILES.items()
        if field != "reporter_sha256"
        and input_contract.get(field) != sha256_file(repository / relative)
    ]
    reporter_path = repository / FROZEN_SCORING_FILES["reporter_sha256"]
    if normalization_amendment.get("original_reporter_sha256") != input_contract.get(
        "reporter_sha256",
    ) or normalization_amendment.get("amended_reporter_sha256") != sha256_file(
        reporter_path,
    ):
        mismatches.append("reporter_sha256")
    if mismatches:
        raise ValueError(
            "Spec 0045 scoring files changed after input freeze: "
            + ", ".join(mismatches),
        )


def _validate_normalization_amendment(
    *,
    path: Path,
    expected_sha256: str,
    current_reporter_sha256: str,
    remote_reporter_sha256: str,
    launch_receipt_sha256: str,
) -> dict[str, object]:
    if sha256_file(path) != expected_sha256:
        raise ValueError("Spec 0045 normalization amendment bytes differ")
    amendment = _read_object(path)
    if (
        set(amendment)
        != {
            "accepted_kernel_reference",
            "amended_reporter_sha256",
            "authorization",
            "input_contract_sha256",
            "launch_receipt_sha256",
            "original_reporter_sha256",
            "outputs_retrieved_before_amendment",
            "requested_kernel_id",
            "schema_version",
            "scientific_changes",
            "scope",
        }
        or amendment.get("schema_version")
        != "spec0046.spec0045_slug_normalization_amendment.v1"
        or amendment.get("scope") != "administrative_slug_normalization_only"
        or amendment.get("original_reporter_sha256") != remote_reporter_sha256
        or amendment.get("amended_reporter_sha256") != current_reporter_sha256
        or amendment.get("launch_receipt_sha256") != launch_receipt_sha256
        or amendment.get("input_contract_sha256")
        != "b5a32ebffd0d88a88d6f21b64ba5c9a23016f05d7a2db0546e442f12a0acecc1"
        or amendment.get("requested_kernel_id")
        != "maximshtefan/eqvae-frozen-vae-test-reconstruction"
        or amendment.get("accepted_kernel_reference")
        != "maximshtefan/eqvae-frozen-vae-full-test-reconstruction/1"
        or amendment.get("scientific_changes") != []
        or amendment.get("outputs_retrieved_before_amendment") is not False
    ):
        raise ValueError("Spec 0045 normalization amendment contract differs")
    return amendment


def _write_joined_metrics(
    path: Path,
    *,
    remote_rows: Sequence[Mapping[str, object]],
    labels: Mapping[int, OracleLabel],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with (
        path.open("xb") as binary,
        gzip.GzipFile(
            fileobj=binary,
            mode="wb",
            mtime=0,
        ) as compressed,
    ):
        handle = io.TextIOWrapper(compressed, encoding="utf-8", newline="")
        fieldnames = (*REMOTE_METRIC_HEADER, "diagnosis_label", "diagnosis_index")
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(fieldnames)
        for row in remote_rows:
            label = labels[int(cast("str", row["atlas_row_index"]))]
            writer.writerow(
                (
                    *(row[name] for name in REMOTE_METRIC_HEADER),
                    label.diagnosis_label,
                    label.diagnosis_index,
                ),
            )
        handle.flush()
        handle.detach()


def _validate_metric_location_rows(
    rows: Sequence[Mapping[str, object]],
    locations: Sequence[TestLocation],
) -> None:
    if len(rows) != len(locations) or any(
        tuple(str(row[name]) for name in REDACTED_LOCATION_HEADER)
        != tuple(str(value) for value in location.csv_values())
        for row, location in zip(rows, locations, strict=True)
    ):
        raise ValueError("Remote metric rows differ from the frozen input locations")


def _write_per_wsi(path: Path, *, scored: Mapping[str, object]) -> None:
    rows = cast("list[dict[str, object]]", scored["per_wsi"])
    _write_csv(path, rows=rows, fieldnames=tuple(rows[0]))


def _write_per_diagnosis(path: Path, *, scored: Mapping[str, object]) -> None:
    branches = cast("Mapping[str, object]", scored["branches"])
    rows: list[dict[str, object]] = []
    for diagnosis in DIAGNOSES:
        for branch in BRANCHES:
            summary = cast(
                "Mapping[str, object]",
                cast(
                    "Mapping[str, object]",
                    cast("Mapping[str, object]", branches[branch])["per_diagnosis"],
                )[diagnosis],
            )
            patch_distributions = cast(
                "Mapping[str, object]",
                summary["patch_distribution"],
            )
            wsi_macro = cast("Mapping[str, object]", summary["wsi_macro"])
            rows.extend(
                {
                    "diagnosis": diagnosis,
                    "branch": branch,
                    "metric": metric,
                    "wsi_count": summary["wsi_count"],
                    "patch_count": summary["patch_count"],
                    **dict(cast("Mapping[str, object]", patch_distributions[metric])),
                    "wsi_macro_mean": wsi_macro[metric],
                }
                for metric in METRICS
            )
    _write_csv(path, rows=rows, fieldnames=tuple(rows[0]))


def _write_bootstrap(path: Path, *, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with (
        path.open("xb") as binary,
        gzip.GzipFile(
            fileobj=binary,
            mode="wb",
            mtime=0,
        ) as compressed,
    ):
        handle = io.TextIOWrapper(compressed, encoding="utf-8", newline="")
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        handle.detach()


def _write_tex(path: Path, *, scored: Mapping[str, object]) -> None:
    branches = cast("Mapping[str, object]", scored["branches"])
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"Model & MAE $\downarrow$ & MSE $\downarrow$ & PSNR $\uparrow$ & SSIM $\uparrow$ \\",
        r"\hline",
    ]
    for branch, label in (("normal", "Normal VAE"), ("so2", r"$SO(2)$ VAE")):
        values = cast(
            "Mapping[str, float | None]",
            cast("Mapping[str, object]", branches[branch])["pooled_patch_mean"],
        )
        rendered = ["--" if values[m] is None else f"{values[m]:.4f}" for m in METRICS]
        lines.append(f"{label} & " + " & ".join(rendered) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_paired_wsi(path: Path, *, scored: Mapping[str, object]) -> None:
    rows = cast("list[Mapping[str, object]]", scored["per_wsi"])
    values = [
        float(cast("float", row[f"{branch}_mae_norm"]))
        for row in rows
        for branch in BRANCHES
    ]
    low, high = min(values), max(values)
    span = max(high - low, 1e-9)
    image = Image.new("RGB", (1200, 700), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((50, 25), "Full sealed test: paired WSI mean MAE", fill=INK, font=font)
    draw.text(
        (50, 48),
        "Each line is one WSI; lower is better. Robustness view for the patch-level result.",
        fill=MUTED,
        font=font,
    )
    x_normal, x_so2 = 420, 780
    draw.text((x_normal - 35, 665), "Normal", fill=NORMAL, font=font)
    draw.text((x_so2 - 25, 665), "SO(2)", fill=SO2, font=font)
    for tick in range(6):
        y = 620 - tick * 105
        value = low + span * tick / 5
        draw.line((130, y, 1050, y), fill=GRID, width=1)
        draw.text((55, y - 6), f"{value:.4f}", fill=MUTED, font=font)
    for row in rows:
        normal = float(cast("float", row["normal_mae_norm"]))
        so2 = float(cast("float", row["so2_mae_norm"]))
        y_normal = 620 - int((normal - low) / span * 525)
        y_so2 = 620 - int((so2 - low) / span * 525)
        draw.line((x_normal, y_normal, x_so2, y_so2), fill="#94a3b8", width=2)
        draw.ellipse(
            (x_normal - 4, y_normal - 4, x_normal + 4, y_normal + 4),
            fill=NORMAL,
        )
        draw.ellipse((x_so2 - 4, y_so2 - 4, x_so2 + 4, y_so2 + 4), fill=SO2)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _render_diagnosis_mae(path: Path, *, scored: Mapping[str, object]) -> None:
    branches = cast("Mapping[str, object]", scored["branches"])
    summaries: dict[tuple[str, str], Mapping[str, object]] = {}
    finite_values: list[float] = []
    for diagnosis in DIAGNOSES:
        for branch in BRANCHES:
            diagnosis_summary = cast(
                "Mapping[str, object]",
                cast(
                    "Mapping[str, object]",
                    cast("Mapping[str, object]", branches[branch])["per_diagnosis"],
                )[diagnosis],
            )
            metric_summary = cast(
                "Mapping[str, object]",
                cast("Mapping[str, object]", diagnosis_summary["patch_distribution"])[
                    "mae_norm"
                ],
            )
            summaries[diagnosis, branch] = metric_summary
            finite_values.extend(
                float(cast("float", metric_summary[key]))
                for key in ("q1", "median", "q3", "mean")
                if metric_summary[key] is not None
            )
    low, high = min(finite_values), max(finite_values)
    span = max(high - low, 1e-9)
    image = Image.new("RGB", (1200, 720), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text(
        (45, 22),
        "Full sealed test: patch MAE by diagnosis",
        fill=INK,
        font=font,
    )
    draw.text(
        (45, 45),
        "Exploratory diagnosis breakdown; pooled patch MAE across all diagnoses is primary.",
        fill=MUTED,
        font=font,
    )
    plot_left, plot_right = 190, 1130
    for tick in range(6):
        x = plot_left + int((plot_right - plot_left) * tick / 5)
        value = low + span * tick / 5
        draw.line((x, 85, x, 650), fill=GRID, width=1)
        draw.text((x - 22, 662), f"{value:.4f}", fill=MUTED, font=font)
    for diagnosis_index, diagnosis in enumerate(DIAGNOSES):
        center_y = 135 + diagnosis_index * 105
        draw.text((55, center_y - 6), diagnosis, fill=INK, font=font)
        for branch_index, branch in enumerate(BRANCHES):
            summary = summaries[diagnosis, branch]
            y = center_y + (-13 if branch_index == 0 else 13)
            color = NORMAL if branch == "normal" else SO2
            q1 = _diagnosis_plot_scale(
                cast("float", summary["q1"]),
                low=low,
                span=span,
                left=plot_left,
                right=plot_right,
            )
            median = _diagnosis_plot_scale(
                cast("float", summary["median"]),
                low=low,
                span=span,
                left=plot_left,
                right=plot_right,
            )
            q3 = _diagnosis_plot_scale(
                cast("float", summary["q3"]),
                low=low,
                span=span,
                left=plot_left,
                right=plot_right,
            )
            mean = _diagnosis_plot_scale(
                cast("float", summary["mean"]),
                low=low,
                span=span,
                left=plot_left,
                right=plot_right,
            )
            draw.line((q1, y, q3, y), fill=color, width=4)
            draw.ellipse((median - 4, y - 4, median + 4, y + 4), fill=color)
            draw.line((mean, y - 7, mean, y + 7), fill=color, width=2)
    draw.text((850, 45), "Normal", fill=NORMAL, font=font)
    draw.text((925, 45), "SO(2)", fill=SO2, font=font)
    draw.text((1000, 45), "dot=median | bar=IQR | tick=mean", fill=MUTED, font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _render_patch_distributions(path: Path, *, scored: Mapping[str, object]) -> None:
    branches = cast("Mapping[str, object]", scored["branches"])
    image = Image.new("RGB", (1200, 760), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text(
        (45, 22),
        "Full sealed test: descriptive patch distributions",
        fill=INK,
        font=font,
    )
    draw.text(
        (45, 45),
        "Boxes describe correlated patches; they are not confidence intervals.",
        fill=MUTED,
        font=font,
    )
    for panel, metric in enumerate(METRICS):
        left = 50 + (panel % 2) * 575
        top = 90 + (panel // 2) * 320
        summaries = {
            branch: cast(
                "Mapping[str, object]",
                cast("Mapping[str, object]", branches[branch])["patch_distribution"],
            )[metric]
            for branch in BRANCHES
        }
        finite_values = [
            float(cast("float", value))
            for summary in summaries.values()
            for key in ("min", "q1", "median", "q3", "max")
            if (value := cast("Mapping[str, object]", summary)[key]) is not None
        ]
        draw.rectangle((left, top, left + 525, top + 270), outline=GRID, width=1)
        draw.text((left + 10, top + 10), metric, fill=INK, font=font)
        if not finite_values:
            draw.text(
                (left + 10, top + 45),
                "undefined: positive infinity present",
                fill=MUTED,
                font=font,
            )
            continue
        low, high = min(finite_values), max(finite_values)
        span = max(high - low, 1e-9)
        for index, branch in enumerate(BRANCHES):
            summary = cast("Mapping[str, object]", summaries[branch])
            y = top + 100 + index * 95
            if any(
                summary[key] is None for key in ("min", "q1", "median", "q3", "max")
            ):
                draw.text(
                    (left + 10, y),
                    f"{branch}: +inf count={summary['inf_count']}",
                    fill=MUTED,
                    font=font,
                )
                continue
            x_min, x_q1, x_med, x_q3, x_max = (
                _plot_scale(
                    cast("float", summary[key]),
                    low=low,
                    span=span,
                    left=left,
                )
                for key in ("min", "q1", "median", "q3", "max")
            )
            color = NORMAL if branch == "normal" else SO2
            draw.line((x_min, y, x_max, y), fill=color, width=2)
            draw.rectangle((x_q1, y - 15, x_q3, y + 15), outline=color, width=3)
            draw.line((x_med, y - 15, x_med, y + 15), fill=color, width=3)
            draw.text((left + 5, y - 6), branch, fill=color, font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _write_csv(
    path: Path,
    *,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _plot_scale(value: float, *, low: float, span: float, left: int) -> int:
    return left + 50 + int((value - low) / span * 420)


def _diagnosis_plot_scale(
    value: float,
    *,
    low: float,
    span: float,
    left: int,
    right: int,
) -> int:
    return left + int((value - low) / span * (right - left))


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return cast("dict[str, object]", value)


def _file_record(path: Path) -> dict[str, int | str]:
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _file_records(
    root: Path,
    *,
    exclude: set[str] | None = None,
) -> dict[str, object]:
    excluded = exclude or set()
    return {
        path.relative_to(root).as_posix(): _file_record(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.relative_to(root).as_posix() not in excluded
    }


def _kernel_upload_records(root: Path) -> dict[str, object]:
    metadata = _read_object(root / "kernel-metadata.json")
    code_file = cast("str", metadata.get("code_file"))
    if not code_file or "/" in code_file:
        raise ValueError("Spec 0045 kernel code file differs")
    return {
        name: _file_record(root / name) for name in ("kernel-metadata.json", code_file)
    }


def _repo_root(path: Path) -> Path:
    for parent in path.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise ValueError("Could not locate repository root")


__all__ = [
    "LOCATION_IDENTITY_SHA256",
    "ORACLE_SHA256",
    "score_retrieved_vae_test_output",
    "write_exclusive_json",
]
