# Copyright 2026 HiperMaximus
# pyright: reportAny=false, reportExplicitAny=false, reportPrivateUsage=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
# ruff: noqa: DOC201, DOC501, EM101, EM102, PLR0913, PLR0914, PLR0915, PLR0916, PLW0717, TRY003
"""One-use administrative scoring resume for Spec 0045 through Spec 0047."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

from eqvae.evaluation.vae_test import (
    TEST_ROW_COUNT,
    TEST_WSI_COUNT,
    load_test_locations,
    sha256_file,
)
from eqvae.evaluation.vae_test_index_adapter import (
    FROZEN_ORACLE_DIAGNOSIS_TO_INDEX,
    score_with_frozen_oracle_indices,
)
from eqvae.evaluation.vae_test_reporting import (
    ORACLE_SHA256,
    _file_record,
    _file_records,
    _kernel_upload_records,
    _read_object,
    _render_diagnosis_mae,
    _render_paired_wsi,
    _render_patch_distributions,
    _repo_root,
    _validate_metric_location_rows,
    _validate_normalization_amendment,
    _write_bootstrap,
    _write_joined_metrics,
    _write_json,
    _write_per_diagnosis,
    _write_per_wsi,
    _write_tex,
    write_exclusive_json,
)
from eqvae.evaluation.vae_test_scoring import (
    DIAGNOSIS_TO_INDEX,
    TEST_VECTOR_PATH,
    load_oracle_labels,
    load_remote_metric_rows,
    verify_test_vector,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

OLD_DIAGNOSIS_TO_INDEX: Final = dict(DIAGNOSIS_TO_INDEX)
FROZEN_SCORER_SHA256: Final = (
    "55a0249661d5f972e2be42724d5afac797997f26c75f57d14988da52b01936c1"
)
_PRIOR_CLAIM_KEYS: Final = {
    "amended_local_reporter_sha256",
    "input_contract_sha256",
    "input_dataset_receipt_sha256",
    "kernel_files",
    "normalization_amendment_sha256",
    "oracle_sha256",
    "original_reporter_sha256",
    "policy",
    "remote_launch_receipt_sha256",
    "remote_manifest_sha256",
    "remote_metric_bytes",
    "remote_metric_sha256",
    "remote_output_receipt_sha256",
    "schema_version",
    "scorer_sha256",
    "scorer_vector_sha256",
    "spec_sha256",
}


def resume_score_retrieved_vae_test_output(
    *,
    remote_output_root: Path,
    launch_receipt_path: Path,
    input_receipt_path: Path,
    input_contract_path: Path,
    kernel_root: Path,
    oracle_path: Path,
    output_root: Path,
    prior_pre_score_claim_path: Path,
    normalization_amendment_path: Path,
    expected_normalization_amendment_sha256: str,
    diagnosis_index_amendment_path: Path,
    expected_diagnosis_index_amendment_sha256: str,
) -> dict[str, object]:
    """Resume the exact claimed score once under the administrative amendment."""
    repository = _repo_root(Path(__file__).resolve())
    if (
        remote_output_root.resolve()
        != repository / "runs/kaggle/vae_test_reconstruction_v1"
        or launch_receipt_path.resolve()
        != repository
        / "runs/local/kaggle_launches/maximshtefan"
        / "eqvae-frozen-vae-full-test-reconstruction/v0001.json"
        or output_root.resolve()
        != repository / "runs/local/vae_test_reconstruction_scored_v1"
        or prior_pre_score_claim_path.resolve()
        != repository
        / "runs/local/vae_test_reconstruction_scored_v1.pre_score_claim.json"
    ):
        raise ValueError("Spec 0047 exact scoring-resume paths differ")
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite {output_root}")
    staging = output_root.with_name(f".{output_root.name}.scoring")
    if staging.exists():
        raise FileExistsError(f"Stale scoring directory exists: {staging}")
    resume_claim_path = output_root.with_name(f"{output_root.name}.resume_claim.json")
    if resume_claim_path.exists():
        raise FileExistsError("Spec 0047 scoring-resume authority is consumed")

    scorer_path = repository / "src/eqvae/evaluation/vae_test_scoring.py"
    index_adapter_path = repository / "src/eqvae/evaluation/vae_test_index_adapter.py"
    reporter_path = repository / "src/eqvae/evaluation/vae_test_reporting.py"
    resume_module_path = Path(__file__).resolve()
    vector_path = repository / TEST_VECTOR_PATH
    scorer_sha256 = _validate_frozen_scorer(scorer_path)
    input_contract = _read_object(input_contract_path)
    normalization_amendment = _validate_normalization_amendment(
        path=normalization_amendment_path,
        expected_sha256=expected_normalization_amendment_sha256,
        current_reporter_sha256=sha256_file(reporter_path),
        remote_reporter_sha256=cast("str", input_contract.get("reporter_sha256")),
        launch_receipt_sha256=sha256_file(launch_receipt_path),
    )
    diagnosis_amendment = validate_diagnosis_index_amendment(
        path=diagnosis_index_amendment_path,
        expected_sha256=expected_diagnosis_index_amendment_sha256,
        current_index_adapter_sha256=sha256_file(index_adapter_path),
        resume_module_sha256=sha256_file(resume_module_path),
        prior_pre_score_claim_sha256=sha256_file(prior_pre_score_claim_path),
    )
    prior_authority = _load_prior_authority_before_claim(
        path=prior_pre_score_claim_path,
        normalization_amendment=normalization_amendment,
        diagnosis_amendment=diagnosis_amendment,
    )
    verify_test_vector(vector_path)
    resume_claim = {
        "schema_version": "spec0047.resume_score_claim.v1",
        "policy": "one_administrative_resume_no_remote_retry",
        "prior_pre_score_claim_sha256": sha256_file(prior_pre_score_claim_path),
        "diagnosis_index_amendment_sha256": expected_diagnosis_index_amendment_sha256,
        "remote_launch_receipt_sha256": prior_authority["remote_launch_receipt_sha256"],
        "remote_output_receipt_sha256": prior_authority["remote_output_receipt_sha256"],
        "remote_metric_sha256": prior_authority["remote_metric_sha256"],
        "oracle_sha256": prior_authority["oracle_sha256"],
        "frozen_scorer_sha256": scorer_sha256,
        "diagnosis_index_adapter_sha256": sha256_file(index_adapter_path),
        "resume_module_sha256": sha256_file(resume_module_path),
        "scorer_vector_sha256": sha256_file(vector_path),
    }
    prior_claim = _consume_resume_authority(
        resume_claim_path=resume_claim_path,
        resume_claim=resume_claim,
        post_claim_validation=lambda: _validate_resume_inputs(
            remote_output_root=remote_output_root,
            launch_receipt_path=launch_receipt_path,
            input_receipt_path=input_receipt_path,
            input_contract_path=input_contract_path,
            kernel_root=kernel_root,
            oracle_path=oracle_path,
            prior_pre_score_claim_path=prior_pre_score_claim_path,
            normalization_amendment=normalization_amendment,
            diagnosis_amendment=diagnosis_amendment,
            vector_path=vector_path,
        ),
    )

    staging.mkdir(parents=True)
    try:
        _write_json(staging / "pre_score_claim.json", prior_claim)
        _write_json(staging / "resume_score_claim.json", resume_claim)
        metric_path = (
            remote_output_root / "vae_test_reconstruction/per_patch_metrics.csv.gz"
        )
        remote_rows = load_remote_metric_rows(metric_path)
        population = cast("Mapping[str, object]", input_contract["population"])
        redacted = cast("Mapping[str, object]", population["redacted_location"])
        locations = load_test_locations(
            input_contract_path.parent / "vae_test_locations.csv",
            expected_sha256=cast("str", redacted["sha256"]),
        )
        _validate_metric_location_rows(remote_rows, locations)
        labels = load_oracle_labels(oracle_path, expected_sha256=ORACLE_SHA256)
        scored = score_with_frozen_oracle_indices(
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
        _render_paired_wsi(staging / "figures/paired_wsi_mae.png", scored=scored)
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
            "administrative_resume": "spec0047_diagnosis_index_only",
        }
        _write_json(staging / "status.json", status)
        staging.replace(output_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return status


def validate_diagnosis_index_amendment(
    *,
    path: Path,
    expected_sha256: str,
    current_index_adapter_sha256: str,
    resume_module_sha256: str,
    prior_pre_score_claim_sha256: str,
) -> dict[str, object]:
    """Authenticate the exact non-scientific diagnosis-index repair."""
    if sha256_file(path) != expected_sha256:
        raise ValueError("Spec 0047 diagnosis-index amendment bytes differ")
    amendment = _read_object(path)
    if (
        set(amendment)
        != {
            "aggregate_metrics_observed_before_amendment",
            "authorization",
            "diagnosis_index_adapter_sha256",
            "frozen_oracle_mapping",
            "input_contract_sha256",
            "launch_receipt_sha256",
            "old_scorer_mapping",
            "oracle_sha256",
            "frozen_scorer_sha256",
            "output_receipt_sha256",
            "prior_pre_score_claim_sha256",
            "remote_metric_sha256",
            "resume_module_sha256",
            "schema_version",
            "scientific_changes",
            "scope",
            "scored_output_created_before_amendment",
            "scorer_vector_sha256",
        }
        or amendment.get("schema_version")
        != "spec0047.spec0045_diagnosis_index_resume_amendment.v1"
        or amendment.get("scope") != "administrative_diagnosis_index_only"
        or amendment.get("frozen_scorer_sha256") != FROZEN_SCORER_SHA256
        or amendment.get("diagnosis_index_adapter_sha256")
        != current_index_adapter_sha256
        or amendment.get("resume_module_sha256") != resume_module_sha256
        or amendment.get("prior_pre_score_claim_sha256") != prior_pre_score_claim_sha256
        or amendment.get("old_scorer_mapping") != OLD_DIAGNOSIS_TO_INDEX
        or amendment.get("frozen_oracle_mapping") != FROZEN_ORACLE_DIAGNOSIS_TO_INDEX
        or amendment.get("scientific_changes") != []
        or amendment.get("aggregate_metrics_observed_before_amendment") is not False
        or amendment.get("scored_output_created_before_amendment") is not False
    ):
        raise ValueError("Spec 0047 diagnosis-index amendment contract differs")
    return amendment


def _validate_frozen_scorer(path: Path) -> str:
    observed = sha256_file(path)
    if observed != FROZEN_SCORER_SHA256:
        raise ValueError("Spec 0047 frozen scorer bytes differ")
    return observed


def _load_prior_authority_before_claim(
    *,
    path: Path,
    normalization_amendment: Mapping[str, object],
    diagnosis_amendment: Mapping[str, object],
) -> dict[str, object]:
    """Validate recorded authority without reopening remote metrics or labels."""
    prior = _read_object(path)
    if (
        set(prior) != _PRIOR_CLAIM_KEYS
        or prior.get("schema_version") != "spec0045.pre_score_claim.v1"
        or prior.get("policy") != "no_remote_retry_after_this_claim"
        or sha256_file(path) != diagnosis_amendment.get("prior_pre_score_claim_sha256")
        or prior.get("normalization_amendment_sha256")
        != "4eb160368e8eb8413171e77a6c30edd172a13693cf9f1bda625c7b60a261e147"
        or prior.get("amended_local_reporter_sha256")
        != normalization_amendment.get("amended_reporter_sha256")
        or prior.get("original_reporter_sha256")
        != normalization_amendment.get("original_reporter_sha256")
        or prior.get("scorer_sha256") != FROZEN_SCORER_SHA256
        or diagnosis_amendment.get("input_contract_sha256")
        != prior.get("input_contract_sha256")
        or diagnosis_amendment.get("launch_receipt_sha256")
        != prior.get("remote_launch_receipt_sha256")
        or diagnosis_amendment.get("output_receipt_sha256")
        != prior.get("remote_output_receipt_sha256")
        or diagnosis_amendment.get("remote_metric_sha256")
        != prior.get("remote_metric_sha256")
        or diagnosis_amendment.get("oracle_sha256") != prior.get("oracle_sha256")
        or diagnosis_amendment.get("scorer_vector_sha256")
        != prior.get("scorer_vector_sha256")
    ):
        raise ValueError("Spec 0047 recorded scoring authority differs")
    return prior


def _consume_resume_authority(
    *,
    resume_claim_path: Path,
    resume_claim: Mapping[str, object],
    post_claim_validation: Callable[[], dict[str, object]],
) -> dict[str, object]:
    """Consume the resume before evidence validation; never remove the claim."""
    write_exclusive_json(resume_claim_path, resume_claim)
    return post_claim_validation()


def _validate_resume_inputs(
    *,
    remote_output_root: Path,
    launch_receipt_path: Path,
    input_receipt_path: Path,
    input_contract_path: Path,
    kernel_root: Path,
    oracle_path: Path,
    prior_pre_score_claim_path: Path,
    normalization_amendment: Mapping[str, object],
    diagnosis_amendment: Mapping[str, object],
    vector_path: Path,
) -> dict[str, object]:
    prior = _read_object(prior_pre_score_claim_path)
    output_receipt_path = remote_output_root / "kaggle_output_receipt.json"
    metric_path = (
        remote_output_root / "vae_test_reconstruction/per_patch_metrics.csv.gz"
    )
    manifest_path = remote_output_root / "vae_test_reconstruction/manifest.json"
    if (
        set(prior) != _PRIOR_CLAIM_KEYS
        or prior.get("schema_version") != "spec0045.pre_score_claim.v1"
        or prior.get("policy") != "no_remote_retry_after_this_claim"
        or sha256_file(prior_pre_score_claim_path)
        != diagnosis_amendment.get("prior_pre_score_claim_sha256")
        or sha256_file(input_contract_path) != prior.get("input_contract_sha256")
        or sha256_file(input_receipt_path) != prior.get("input_dataset_receipt_sha256")
        or sha256_file(launch_receipt_path) != prior.get("remote_launch_receipt_sha256")
        or sha256_file(output_receipt_path) != prior.get("remote_output_receipt_sha256")
        or sha256_file(manifest_path) != prior.get("remote_manifest_sha256")
        or sha256_file(metric_path) != prior.get("remote_metric_sha256")
        or metric_path.stat().st_size != prior.get("remote_metric_bytes")
        or sha256_file(oracle_path) != prior.get("oracle_sha256")
        or sha256_file(vector_path) != prior.get("scorer_vector_sha256")
        or prior.get("scorer_sha256") != diagnosis_amendment.get("frozen_scorer_sha256")
        or prior.get("normalization_amendment_sha256")
        != "4eb160368e8eb8413171e77a6c30edd172a13693cf9f1bda625c7b60a261e147"
        or prior.get("amended_local_reporter_sha256")
        != normalization_amendment.get("amended_reporter_sha256")
        or prior.get("original_reporter_sha256")
        != normalization_amendment.get("original_reporter_sha256")
        or prior.get("kernel_files") != _kernel_upload_records(kernel_root)
        or diagnosis_amendment.get("input_contract_sha256")
        != prior.get("input_contract_sha256")
        or diagnosis_amendment.get("launch_receipt_sha256")
        != prior.get("remote_launch_receipt_sha256")
        or diagnosis_amendment.get("output_receipt_sha256")
        != prior.get("remote_output_receipt_sha256")
        or diagnosis_amendment.get("remote_metric_sha256")
        != prior.get("remote_metric_sha256")
        or diagnosis_amendment.get("oracle_sha256") != prior.get("oracle_sha256")
        or diagnosis_amendment.get("scorer_vector_sha256")
        != prior.get("scorer_vector_sha256")
    ):
        raise ValueError("Spec 0047 prior scoring authority differs")
    output_receipt = _read_object(output_receipt_path)
    if (
        output_receipt.get("schema_version") != "eqvae.kaggle_download.v1"
        or output_receipt.get("resource_kind") != "kernel"
        or output_receipt.get("resource_reference")
        != "maximshtefan/eqvae-frozen-vae-full-test-reconstruction/1"
        or output_receipt.get("files")
        != _file_records(remote_output_root, exclude={output_receipt_path.name})
    ):
        raise ValueError("Spec 0047 remote output receipt differs")
    return prior


__all__ = [
    "FROZEN_ORACLE_DIAGNOSIS_TO_INDEX",
    "FROZEN_SCORER_SHA256",
    "OLD_DIAGNOSIS_TO_INDEX",
    "resume_score_retrieved_vae_test_output",
    "validate_diagnosis_index_amendment",
]
