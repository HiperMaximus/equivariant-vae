# Copyright 2026 HiperMaximus
# pyright: reportPrivateUsage=false
# ruff: noqa: D103, PLC2701, PLR2004
"""Focused estimator and imbalance tests for the frozen Spec 0045 scorer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
import scripts.build_vae_test_evaluation as builder
from PIL import Image

from eqvae.evaluation.vae_test import canonical_json_sha256, sha256_file
from eqvae.evaluation.vae_test_reporting import (
    LATENT_KERNEL_SOURCES,
    REMOTE_OUTPUT_FILES,
    _authenticate_remote_output,
    _kernel_upload_records,
    _render_diagnosis_mae,
    _render_paired_wsi,
    _render_patch_distributions,
    _validate_frozen_scoring_files,
    _validate_normalization_amendment,
    _write_bootstrap,
    _write_per_diagnosis,
    _write_per_wsi,
    _write_tex,
    write_exclusive_json,
)
from eqvae.evaluation.vae_test_scoring import (
    BOOTSTRAP_SEED,
    EXPECTED_DIAGNOSIS_WSI_COUNTS,
    _build_vector_fixture,
    _score_vae_test_metrics,
    verify_test_vector,
)

VECTOR = Path("docs/data/spec0045_vae_test_scorer_vector.json")
AMENDMENT = Path("docs/data/spec0046_spec0045_slug_normalization_amendment.json")
REPORTER = Path("src/eqvae/evaluation/vae_test_reporting.py")
ACCEPTED_REFERENCE = "maximshtefan/eqvae-frozen-vae-full-test-reconstruction/1"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _tree_records(root: Path, *, exclude: set[str] | None = None) -> dict[str, object]:
    omitted = exclude or set()
    return {
        path.relative_to(root).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.relative_to(root).as_posix() not in omitted
    }


def _authenticated_remote_fixture(tmp_path: Path) -> dict[str, Path]:
    contract_source = Path(
        "runs/local/vae_test_evaluation_input/spec0045_vae_test_input.json",
    )
    contract_path = tmp_path / "input/spec0045_vae_test_input.json"
    contract_path.parent.mkdir(parents=True)
    contract_path.write_bytes(contract_source.read_bytes())
    contract = cast(
        "dict[str, object]",
        json.loads(contract_path.read_text(encoding="utf-8")),
    )
    dataset_reference = cast("str", contract["dataset_reference"])
    requested_kernel_id = cast("str", contract["kernel_id"])
    actor = dataset_reference.split("/", maxsplit=1)[0]
    accepted_kernel_id = ACCEPTED_REFERENCE.rsplit("/", maxsplit=1)[0]
    input_receipt = tmp_path / "input_receipt.json"
    _write_json(
        input_receipt,
        {
            "schema_version": "spec0045.input_dataset_receipt.v1",
            "status": "verified",
            "visibility": "private",
            "dataset_version": 1,
            "dataset_reference": dataset_reference,
            "input_contract_sha256": sha256_file(contract_path),
        },
    )
    kernel_root = Path("kaggle/kernels/vae_test_reconstruction")
    launch_receipt = tmp_path / "launch_receipt.json"
    _write_json(
        launch_receipt,
        {
            "schema_version": "eqvae.kaggle_kernel_launch.v1",
            "actor": actor,
            "original_kernel_id": requested_kernel_id,
            "requested_kernel_id": requested_kernel_id,
            "accepted_version": 1,
            "kernel_id": accepted_kernel_id,
            "kernel_reference": ACCEPTED_REFERENCE,
            "source_locators": {
                "competition_sources": ["UBC-OCEAN"],
                "dataset_sources": [dataset_reference],
                "kernel_sources": list(LATENT_KERNEL_SOURCES),
                "model_sources": [],
            },
            "source_metadata_sha256": sha256_file(
                kernel_root / "kernel-metadata.json",
            ),
            "source_files": _kernel_upload_records(kernel_root),
            "upload_metadata_sha256": sha256_file(
                kernel_root / "kernel-metadata.json",
            ),
            "upload_files": _kernel_upload_records(kernel_root),
        },
    )
    remote_root = tmp_path / "remote"
    payload = remote_root / "vae_test_reconstruction"
    payload.mkdir(parents=True)
    (payload / "per_patch_metrics.csv.gz").write_bytes(b"frozen paired rows")
    _write_json(
        payload / "runtime.json",
        {"precision": "FP32", "optimizer_updates": 0, "patch_count": 67_138},
    )
    _write_json(payload / "wsi_evidence.json", {"wsi_count": 23})
    _write_json(
        payload / "run_contract.json",
        {
            "input_contract_sha256": sha256_file(contract_path),
            "dataset_reference": dataset_reference,
            "kernel_sources": list(LATENT_KERNEL_SOURCES),
            "location_identity_sha256": (
                "0f7fc4f01961bdb4f2dd71d7c1b7cd0455e6bb175d7dff095a88bfaa27e34fd5"
            ),
            "source_contract": contract,
        },
    )
    output_files = {
        name: {
            "bytes": (payload / name).stat().st_size,
            "sha256": sha256_file(payload / name),
        }
        for name in REMOTE_OUTPUT_FILES
    }
    _write_json(
        payload / "manifest.json",
        {
            "schema_version": "spec0045.remote_manifest.v1",
            "status": "complete",
            "row_count": 67_138,
            "wsi_count": 23,
            "output_files": output_files,
        },
    )
    _write_json(
        payload / "status.json",
        {
            "schema_version": "spec0045.remote_status.v1",
            "status": "pass",
            "optimizer_updates": 0,
            "manifest_sha256": sha256_file(payload / "manifest.json"),
        },
    )
    _refresh_output_receipt(remote_root, kernel_reference=ACCEPTED_REFERENCE)
    return {
        "contract": contract_path,
        "input_receipt": input_receipt,
        "kernel_root": kernel_root,
        "launch_receipt": launch_receipt,
        "payload": payload,
        "remote_root": remote_root,
    }


def _refresh_output_receipt(remote_root: Path, *, kernel_reference: str) -> None:
    _write_json(
        remote_root / "kaggle_output_receipt.json",
        {
            "schema_version": "eqvae.kaggle_download.v1",
            "resource_kind": "kernel",
            "resource_reference": kernel_reference,
            "files": _tree_records(
                remote_root,
                exclude={"kaggle_output_receipt.json"},
            ),
        },
    )


def _score(*, replicates: int = 250) -> dict[str, object]:
    vector = cast("dict[str, object]", json.loads(VECTOR.read_text(encoding="utf-8")))
    rows, labels, patch_counts = _build_vector_fixture(
        cast("dict[str, object]", vector["fixture"]),
    )
    return _score_vae_test_metrics(
        remote_rows=rows,
        labels_by_atlas_row=labels,
        bootstrap_replicates=replicates,
        bootstrap_seed=BOOTSTRAP_SEED,
        expected_patch_counts=patch_counts,
        expected_wsi_counts=EXPECTED_DIAGNOSIS_WSI_COUNTS,
    )


def test_frozen_scorer_vector_reproduces_exact_result() -> None:
    vector = cast("dict[str, object]", json.loads(VECTOR.read_text(encoding="utf-8")))
    verify_test_vector(VECTOR)
    rows, labels, patch_counts = _build_vector_fixture(
        cast("dict[str, object]", vector["fixture"]),
    )
    result = _score_vae_test_metrics(
        remote_rows=rows,
        labels_by_atlas_row=labels,
        bootstrap_replicates=10_000,
        bootstrap_seed=BOOTSTRAP_SEED,
        expected_patch_counts=patch_counts,
        expected_wsi_counts=EXPECTED_DIAGNOSIS_WSI_COUNTS,
    )
    assert canonical_json_sha256(result) == vector["expected_result_sha256"]


def test_primary_is_pooled_patch_mean_and_diagnosis_is_secondary() -> None:
    result = _score()
    primary = cast("dict[str, object]", result["primary_endpoint"])
    paired = cast("dict[str, object]", result["paired_normal_minus_so2"])
    normal = cast(
        "dict[str, object]",
        cast("dict[str, object]", result["branches"])["normal"],
    )
    so2 = cast(
        "dict[str, object]",
        cast("dict[str, object]", result["branches"])["so2"],
    )
    assert primary["aggregation"] == "pooled_patch_mean"
    patch_difference = float(
        cast("dict[str, float]", normal["pooled_patch_mean"])["mae_norm"],
    ) - float(cast("dict[str, float]", so2["pooled_patch_mean"])["mae_norm"])
    assert patch_difference == pytest.approx(
        float(
            cast(
                "float",
                cast("dict[str, object]", paired["mae_norm"])["point_difference"],
            ),
        ),
    )
    assert (
        primary["normal_value"]
        == cast(
            "dict[str, float]",
            normal["pooled_patch_mean"],
        )["mae_norm"]
    )
    assert (
        primary["so2_value"]
        == cast(
            "dict[str, float]",
            so2["pooled_patch_mean"],
        )["mae_norm"]
    )
    assert "diagnosis_balanced_wsi_macro" not in normal
    for diagnosis_summary in cast(
        "dict[str, dict[str, object]]",
        normal["per_diagnosis"],
    ).values():
        distribution = cast(
            "dict[str, dict[str, object]]",
            diagnosis_summary["patch_distribution"],
        )["mae_norm"]
        assert distribution["n"] == diagnosis_summary["patch_count"]


def test_bootstrap_is_paired_unstratified_wsi_cluster_and_deterministic() -> None:
    first = _score(replicates=100)
    second = _score(replicates=100)
    assert first["bootstrap_rows"] == second["bootstrap_rows"]
    bootstrap = cast("dict[str, object]", first["bootstrap"])
    assert bootstrap["sampling_unit"] == "WSI"
    assert bootstrap["stratified_by"] is None
    assert bootstrap["cluster_count"] == sum(EXPECTED_DIAGNOSIS_WSI_COUNTS.values())
    assert bootstrap["estimand"] == "pooled_patch_mean_weighted_by_patch_count"
    assert bootstrap["same_draws_for_both_models_and_all_metrics"] is True


def test_positive_infinite_psnr_is_preserved_but_json_safe() -> None:
    vector = cast("dict[str, object]", json.loads(VECTOR.read_text(encoding="utf-8")))
    rows, labels, patch_counts = _build_vector_fixture(
        cast("dict[str, object]", vector["fixture"]),
    )
    rows[0]["normal_psnr_img"] = float("inf")
    result = _score_vae_test_metrics(
        remote_rows=rows,
        labels_by_atlas_row=labels,
        bootstrap_replicates=20,
        bootstrap_seed=BOOTSTRAP_SEED,
        expected_patch_counts=patch_counts,
        expected_wsi_counts=EXPECTED_DIAGNOSIS_WSI_COUNTS,
    )
    paired_psnr = cast(
        "dict[str, object]",
        cast("dict[str, object]", result["paired_normal_minus_so2"])["psnr_img"],
    )
    assert paired_psnr["point_difference"] is None
    assert cast("dict[str, object]", paired_psnr["interval"])["status"] == (
        "omitted_positive_infinity"
    )
    assert all(
        "normal_minus_so2_psnr_img" not in cast("dict[str, object]", row)
        for row in cast("list[object]", result["bootstrap_rows"])
    )
    json.dumps(result, allow_nan=False)


def test_scorer_rejects_row_and_support_mutations() -> None:
    vector = cast("dict[str, object]", json.loads(VECTOR.read_text(encoding="utf-8")))
    rows, labels, patch_counts = _build_vector_fixture(
        cast("dict[str, object]", vector["fixture"]),
    )
    reordered = [dict(row) for row in rows]
    reordered[0], reordered[1] = reordered[1], reordered[0]
    with pytest.raises(ValueError, match="row order"):
        _score_vae_test_metrics(
            remote_rows=reordered,
            labels_by_atlas_row=labels,
            bootstrap_replicates=10,
            bootstrap_seed=BOOTSTRAP_SEED,
            expected_patch_counts=patch_counts,
            expected_wsi_counts=EXPECTED_DIAGNOSIS_WSI_COUNTS,
        )

    missing = rows[:-1]
    with pytest.raises(ValueError, match="row counts"):
        _score_vae_test_metrics(
            remote_rows=missing,
            labels_by_atlas_row=labels,
            bootstrap_replicates=10,
            bootstrap_seed=BOOTSTRAP_SEED,
            expected_patch_counts=patch_counts,
            expected_wsi_counts=EXPECTED_DIAGNOSIS_WSI_COUNTS,
        )

    wrong_coordinate = [dict(row) for row in rows]
    wrong_coordinate[0]["x"] = 256
    with pytest.raises(ValueError, match="local oracle"):
        _score_vae_test_metrics(
            remote_rows=wrong_coordinate,
            labels_by_atlas_row=labels,
            bootstrap_replicates=10,
            bootstrap_seed=BOOTSTRAP_SEED,
            expected_patch_counts=patch_counts,
            expected_wsi_counts=EXPECTED_DIAGNOSIS_WSI_COUNTS,
        )


def test_post_launch_scoring_rejects_frozen_file_mutation() -> None:
    contract_path = Path(
        "runs/local/vae_test_evaluation_input/spec0045_vae_test_input.json",
    )
    contract = cast(
        "dict[str, object]",
        json.loads(contract_path.read_text(encoding="utf-8")),
    )
    amendment = cast(
        "dict[str, object]",
        json.loads(AMENDMENT.read_text(encoding="utf-8")),
    )
    _validate_frozen_scoring_files(
        contract,
        repository=Path.cwd(),
        normalization_amendment=amendment,
    )
    contract["reporter_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="reporter_sha256"):
        _validate_frozen_scoring_files(
            contract,
            repository=Path.cwd(),
            normalization_amendment=amendment,
        )
    contract["reporter_sha256"] = amendment["original_reporter_sha256"]
    contract["evaluation_contract_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="evaluation_contract_sha256"):
        _validate_frozen_scoring_files(
            contract,
            repository=Path.cwd(),
            normalization_amendment=amendment,
        )


def test_normalization_amendment_binds_exact_receipt_and_reporters() -> None:
    amendment = cast(
        "dict[str, object]",
        json.loads(AMENDMENT.read_text(encoding="utf-8")),
    )
    result = _validate_normalization_amendment(
        path=AMENDMENT,
        expected_sha256=builder.NORMALIZATION_AMENDMENT_SHA256,
        current_reporter_sha256=sha256_file(REPORTER),
        remote_reporter_sha256=cast("str", amendment["original_reporter_sha256"]),
        launch_receipt_sha256=cast("str", amendment["launch_receipt_sha256"]),
    )
    assert result["scientific_changes"] == []
    assert sha256_file(AMENDMENT) == builder.NORMALIZATION_AMENDMENT_SHA256
    with pytest.raises(ValueError, match="amendment contract"):
        _validate_normalization_amendment(
            path=AMENDMENT,
            expected_sha256=builder.NORMALIZATION_AMENDMENT_SHA256,
            current_reporter_sha256="0" * 64,
            remote_reporter_sha256=cast(
                "str",
                amendment["original_reporter_sha256"],
            ),
            launch_receipt_sha256=cast("str", amendment["launch_receipt_sha256"]),
        )


def test_remote_output_authentication_rejects_source_status_and_byte_mutations(
    tmp_path: Path,
) -> None:
    fixture = _authenticated_remote_fixture(tmp_path)

    def authenticate() -> dict[str, object]:
        contract = cast(
            "dict[str, object]",
            json.loads(fixture["contract"].read_text(encoding="utf-8")),
        )
        return _authenticate_remote_output(
            remote_output_root=fixture["remote_root"],
            launch_receipt_path=fixture["launch_receipt"],
            input_receipt_path=fixture["input_receipt"],
            input_contract_path=fixture["contract"],
            kernel_root=fixture["kernel_root"],
            normalization_amendment={
                "accepted_kernel_reference": ACCEPTED_REFERENCE,
                "original_reporter_sha256": contract["reporter_sha256"],
                "amended_reporter_sha256": sha256_file(REPORTER),
            },
        )

    assert authenticate()["metric_path"] == (
        fixture["payload"] / "per_patch_metrics.csv.gz"
    )

    launch = cast(
        "dict[str, object]",
        json.loads(fixture["launch_receipt"].read_text(encoding="utf-8")),
    )
    launch["accepted_version"] = 2
    _write_json(fixture["launch_receipt"], launch)
    with pytest.raises(ValueError, match="launch receipt"):
        authenticate()
    launch["accepted_version"] = 1
    _write_json(fixture["launch_receipt"], launch)

    launch["requested_kernel_id"] = "maximshtefan/wrong"
    _write_json(fixture["launch_receipt"], launch)
    with pytest.raises(ValueError, match="launch receipt"):
        authenticate()
    launch["requested_kernel_id"] = cast(
        "dict[str, object]",
        json.loads(fixture["contract"].read_text(encoding="utf-8")),
    )["kernel_id"]
    _write_json(fixture["launch_receipt"], launch)

    status_path = fixture["payload"] / "status.json"
    status = cast(
        "dict[str, object]",
        json.loads(status_path.read_text(encoding="utf-8")),
    )
    status["optimizer_updates"] = 1
    _write_json(status_path, status)
    kernel_reference = cast("str", launch["kernel_reference"])
    _refresh_output_receipt(
        fixture["remote_root"],
        kernel_reference=kernel_reference,
    )
    with pytest.raises(ValueError, match="completed remote contract"):
        authenticate()
    status["optimizer_updates"] = 0
    _write_json(status_path, status)

    metric_path = fixture["payload"] / "per_patch_metrics.csv.gz"
    metric_path.write_bytes(b"mutated paired rows")
    _refresh_output_receipt(
        fixture["remote_root"],
        kernel_reference=kernel_reference,
    )
    with pytest.raises(ValueError, match="remote artifact"):
        authenticate()


def test_reporting_tables_figures_and_claim_are_created_exclusively(
    tmp_path: Path,
) -> None:
    scored = _score(replicates=25)
    bootstrap = cast("list[dict[str, object]]", scored.pop("bootstrap_rows"))
    _write_per_wsi(tmp_path / "per_wsi.csv", scored=scored)
    _write_per_diagnosis(tmp_path / "per_diagnosis.csv", scored=scored)
    _write_bootstrap(tmp_path / "bootstrap.csv.gz", rows=bootstrap)
    _write_tex(tmp_path / "metrics.tex", scored=scored)
    _render_paired_wsi(tmp_path / "paired.png", scored=scored)
    _render_patch_distributions(tmp_path / "boxplots.png", scored=scored)
    _render_diagnosis_mae(tmp_path / "diagnosis.png", scored=scored)
    assert (tmp_path / "per_wsi.csv").read_text(encoding="utf-8").count("\n") == 24
    assert (tmp_path / "per_diagnosis.csv").read_text(encoding="utf-8").count(
        "\n",
    ) == 41
    assert "Normal VAE" in (tmp_path / "metrics.tex").read_text(encoding="utf-8")
    assert Image.open(tmp_path / "paired.png").size == (1200, 700)
    assert Image.open(tmp_path / "boxplots.png").size == (1200, 760)
    assert Image.open(tmp_path / "diagnosis.png").size == (1200, 720)

    claim = tmp_path / "claim.json"
    write_exclusive_json(claim, {"frozen": True})
    with pytest.raises(FileExistsError):
        write_exclusive_json(claim, {"frozen": False})
