# Copyright 2026 HiperMaximus
# pyright: reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
# ruff: noqa: COM812, DOC201, DOC501, E501, EM101, EM102, PLR0916, T201, TRY003
"""Build and validate the diagnosis-free Spec 0045 evaluation input."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import torch

from eqvae.evaluation.vae_test import (
    EXPECTED_WSI_PATCH_COUNTS,
    redact_test_locations,
    sha256_file,
    state_dict_sha256,
)
from eqvae.evaluation.vae_test_reporting import (
    score_retrieved_vae_test_output,
    write_exclusive_json,
)
from eqvae.evaluation.vae_test_resume import resume_score_retrieved_vae_test_output
from eqvae.evaluation.vae_test_scoring import verify_test_vector
from eqvae.inference.checkpoints import FrozenModelName, load_frozen_checkpoint

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

ROOT: Final = Path(__file__).resolve().parents[1]
OUTPUT_ROOT: Final = ROOT / "runs/local/vae_test_evaluation_input"
DATASET_SLUG: Final = "eqvae-vae-test-reconstruction-inputs-v1"
CONTRACT_NAME: Final = "spec0045_vae_test_input.json"
SPEC_PATH: Final = ROOT / "docs/specs/0045-frozen-vae-test-reconstruction-evaluation.md"
SCORER_PATH: Final = ROOT / "src/eqvae/evaluation/vae_test_scoring.py"
SCORER_VECTOR_PATH: Final = ROOT / "docs/data/spec0045_vae_test_scorer_vector.json"
EVALUATOR_PATH: Final = ROOT / "src/eqvae/evaluation/vae_test_runtime.py"
EVALUATION_CONTRACT_PATH: Final = ROOT / "src/eqvae/evaluation/vae_test.py"
REPORTER_PATH: Final = ROOT / "src/eqvae/evaluation/vae_test_reporting.py"
NORMALIZATION_AMENDMENT_PATH: Final = (
    ROOT / "docs/data/spec0046_spec0045_slug_normalization_amendment.json"
)
NORMALIZATION_AMENDMENT_SHA256: Final = (
    "4eb160368e8eb8413171e77a6c30edd172a13693cf9f1bda625c7b60a261e147"
)
DIAGNOSIS_INDEX_AMENDMENT_PATH: Final = (
    ROOT / "docs/data/spec0047_spec0045_diagnosis_index_resume_amendment.json"
)
DIAGNOSIS_INDEX_AMENDMENT_SHA256: Final = (
    "0641f572e497f44cec51565e7f0758fbc2065e9d56b59619e2acc7ee529b1a96"
)
SOURCE_LOCATION_PATH: Final = (
    ROOT
    / "runs/kaggle/ubc_ocean_latent_store_finalizer/dataset/views/cancer_test_locations.csv"
)
SOURCE_LOCATION_SHA256: Final = (
    "3599baeb4b70d0414e3f7fa9bf8a359c91b2433613dde186b58f6a711f95e66b"
)
ORACLE_PATH: Final = ROOT / "runs/local/ubc_ocean_eval_consumption/cancer_test.csv"
ORACLE_SHA256: Final = (
    "1d0e4059f469d350ff3960cc10208221548f6afdfc1788e40e6d5da7829806cc"
)
GLOBAL_AUDIT_PATH: Final = (
    ROOT
    / "runs/kaggle/ubc_ocean_latent_store_finalizer/dataset/spec0021_latent_store_global_audit.json"
)
GLOBAL_AUDIT_SHA256: Final = (
    "25dca0a379da88c41a99bf738908cbd22b89eed94b0193bc972832dde6e9889a"
)
PAIR_AUDIT_ROOT: Final = ROOT / "runs/kaggle/ubc_ocean_latents"
CHECKPOINTS: Final = {
    "normal_vae": {
        "path": ROOT
        / "runs/kaggle/selected_runtime_full_v4_session3/checkpoints/step_060000.pt",
        "sha256": "f733304e9178e468546113642bdf01e11348570b340c366cf148973083cb9075",
    },
    "so2_vae": {
        "path": ROOT
        / "runs/kaggle/so2_selected_runtime_full_session7_fresh_v1_retry1/checkpoints/step_060000.pt",
        "sha256": "041e0cd7483cb8642bb72eb1b63c3a36774bf9cadd0b659c9d1db6a813c8f4c7",
    },
}
LATENT_KERNEL_SOURCES: Final = tuple(
    f"maximusshtefan/eqvae-ubc-ocean-latent-run-{run:02d}" for run in range(1, 6)
)
ANALYSIS_CONTRACT: Final = {
    "primary_metric": "mae_norm",
    "primary_aggregation": "pooled_patch_mean",
    "paired_difference": "normal_minus_so2",
    "diagnosis_role": "secondary_exploratory_breakdown",
    "bootstrap": "paired_unstratified_wsi_cluster_10000_seed4501",
}
AUTHORITY_ROOT: Final = ROOT / "runs/local/vae_test_evaluation_authority"
INPUT_RECEIPT_PATH: Final = AUTHORITY_ROOT / "input_dataset_receipt.json"
LAUNCH_CLAIM_PATH: Final = AUTHORITY_ROOT / "launch_claim.json"
KERNEL_ROOT: Final = ROOT / "kaggle/kernels/vae_test_reconstruction"
PRIOR_PRE_SCORE_CLAIM_PATH: Final = (
    ROOT / "runs/local/vae_test_reconstruction_scored_v1.pre_score_claim.json"
)


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch the input builder or validator."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build-input")
    build_parser.add_argument("--actor", required=True)
    validate_parser = subparsers.add_parser("validate-input")
    validate_parser.add_argument("--actor", required=True)
    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--remote-output-root", type=Path, required=True)
    score_parser.add_argument("--launch-receipt", type=Path, required=True)
    score_parser.add_argument("--input-receipt", type=Path, required=True)
    score_parser.add_argument("--output-root", type=Path, required=True)
    resume_parser = subparsers.add_parser("resume-score")
    resume_parser.add_argument("--remote-output-root", type=Path, required=True)
    resume_parser.add_argument("--launch-receipt", type=Path, required=True)
    resume_parser.add_argument("--input-receipt", type=Path, required=True)
    resume_parser.add_argument("--output-root", type=Path, required=True)
    claim_parser = subparsers.add_parser("claim-launch")
    claim_parser.add_argument("--actor", required=True)
    validate_claim_parser = subparsers.add_parser("validate-claimed-launch")
    validate_claim_parser.add_argument("--actor", required=True)
    args = parser.parse_args(argv)
    command = cast("str", args.command)
    if command == "resume-score":
        result = resume_score_retrieved_vae_test_output(
            remote_output_root=cast("Path", args.remote_output_root),
            launch_receipt_path=cast("Path", args.launch_receipt),
            input_receipt_path=cast("Path", args.input_receipt),
            input_contract_path=OUTPUT_ROOT / CONTRACT_NAME,
            kernel_root=KERNEL_ROOT,
            oracle_path=ORACLE_PATH,
            output_root=cast("Path", args.output_root),
            prior_pre_score_claim_path=PRIOR_PRE_SCORE_CLAIM_PATH,
            normalization_amendment_path=NORMALIZATION_AMENDMENT_PATH,
            expected_normalization_amendment_sha256=(NORMALIZATION_AMENDMENT_SHA256),
            diagnosis_index_amendment_path=DIAGNOSIS_INDEX_AMENDMENT_PATH,
            expected_diagnosis_index_amendment_sha256=(
                DIAGNOSIS_INDEX_AMENDMENT_SHA256
            ),
        )
    elif command == "score":
        result = score_retrieved_vae_test_output(
            remote_output_root=cast("Path", args.remote_output_root),
            launch_receipt_path=cast("Path", args.launch_receipt),
            input_receipt_path=cast("Path", args.input_receipt),
            input_contract_path=OUTPUT_ROOT / CONTRACT_NAME,
            kernel_root=KERNEL_ROOT,
            oracle_path=ORACLE_PATH,
            output_root=cast("Path", args.output_root),
            normalization_amendment_path=NORMALIZATION_AMENDMENT_PATH,
            expected_normalization_amendment_sha256=(NORMALIZATION_AMENDMENT_SHA256),
        )
    elif command == "claim-launch":
        result = claim_launch(actor=cast("str", args.actor))
    elif command == "validate-claimed-launch":
        result = validate_claimed_launch(actor=cast("str", args.actor))
    elif command == "build-input":
        actor = cast("str", args.actor)
        result = build_input(actor=actor)
    else:
        actor = cast("str", args.actor)
        result = validate_input(actor=actor)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def build_input(*, actor: str) -> dict[str, object]:
    """Build the private label-free input atomically from immutable local evidence."""
    _validate_actor(actor)
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT_ROOT}")
    staging = OUTPUT_ROOT.with_name(f".{OUTPUT_ROOT.name}.building")
    if staging.exists():
        raise FileExistsError(f"Stale staging directory exists: {staging}")
    try:
        _build_staging(staging, actor=actor)
        staging.replace(OUTPUT_ROOT)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return validate_input(actor=actor)


def _build_staging(staging: Path, *, actor: str) -> None:
    verify_test_vector(SCORER_VECTOR_PATH)
    if sha256_file(GLOBAL_AUDIT_PATH) != GLOBAL_AUDIT_SHA256:
        raise ValueError("Spec 0021 global audit SHA-256 differs")
    audit = _read_object(GLOBAL_AUDIT_PATH)
    if (
        audit.get("schema_version") != "spec0021.latent_store_global_audit.v1"
        or audit.get("status") != "complete"
    ):
        raise ValueError("Spec 0021 global audit is not complete")
    staging.mkdir(parents=True)
    redacted_record = redact_test_locations(
        source_path=SOURCE_LOCATION_PATH,
        oracle_path=ORACLE_PATH,
        output_path=staging / "vae_test_locations.csv",
        expected_source_sha256=SOURCE_LOCATION_SHA256,
        expected_oracle_sha256=ORACLE_SHA256,
    )
    weights = {
        branch: _derive_state_dict(
            branch=branch,
            destination=staging / f"{branch}_state.pt",
        )
        for branch in CHECKPOINTS
    }
    latent_sources, wsi_evidence = _latent_source_contract(audit)
    dataset_reference = f"{actor}/{DATASET_SLUG}"
    files = {
        "vae_test_locations.csv": _file_record(staging / "vae_test_locations.csv"),
        **{
            f"{branch}_state.pt": _file_record(staging / f"{branch}_state.pt")
            for branch in weights
        },
    }
    contract: dict[str, object] = {
        "schema_version": "spec0045.vae_test_input.v1",
        "scope": "frozen_vae_full_test_reconstruction_label_free",
        "status": "complete",
        "visibility": "private",
        "dataset_reference": dataset_reference,
        "dataset_actor": actor,
        "kernel_id": f"{actor}/eqvae-frozen-vae-test-reconstruction",
        "kernel_sources": list(LATENT_KERNEL_SOURCES),
        "producer_versions": dict.fromkeys(LATENT_KERNEL_SOURCES, 1),
        "spec_sha256": sha256_file(SPEC_PATH),
        "scorer_sha256": sha256_file(SCORER_PATH),
        "scorer_vector_sha256": sha256_file(SCORER_VECTOR_PATH),
        "evaluator_sha256": sha256_file(EVALUATOR_PATH),
        "evaluation_contract_sha256": sha256_file(EVALUATION_CONTRACT_PATH),
        "reporter_sha256": sha256_file(REPORTER_PATH),
        "population": {
            "source_manifest_sha256": ORACLE_SHA256,
            "source_location_sha256": SOURCE_LOCATION_SHA256,
            "redacted_location": redacted_record,
            "wsi_patch_counts": {
                str(key): value for key, value in EXPECTED_WSI_PATCH_COUNTS.items()
            },
        },
        "weights": weights,
        "latent_global_audit_sha256": GLOBAL_AUDIT_SHA256,
        "latent_sources": latent_sources,
        "wsi_evidence": wsi_evidence,
        "metrics": {
            "mae_norm": "per_image_fp32_normalized_minus1_plus1",
            "mse_norm": "per_image_fp32_normalized_minus1_plus1",
            "psnr_img": "per_image_fp32_clamped_0_1",
            "ssim_img": "per_image_fp32_clamped_0_1_gaussian11_sigma1.5_reflect",
        },
        "analysis": ANALYSIS_CONTRACT,
        "execution": {
            "batch_size": 8,
            "decoder_input": "stored_fp32_posterior_mu",
            "decoder_output": "raw_fp32_no_tanh",
            "torch": "2.14.0+cu130",
            "optimizer_updates": 0,
            "devices": ["Tesla T4", "Tesla T4"],
        },
        "files": files,
    }
    _write_json(staging / CONTRACT_NAME, contract)
    _write_json(
        staging / "dataset-metadata.json",
        {
            "id": dataset_reference,
            "licenses": [{"name": "other"}],
            "title": "EQVAE VAE test reconstruction inputs",
        },
    )


def validate_input(*, actor: str) -> dict[str, object]:
    """Validate the complete input tree and reject any label-bearing path or field."""
    _validate_actor(actor)
    contract_path = OUTPUT_ROOT / CONTRACT_NAME
    contract = _read_object(contract_path)
    dataset_reference = f"{actor}/{DATASET_SLUG}"
    if (
        contract.get("schema_version") != "spec0045.vae_test_input.v1"
        or contract.get("scope") != "frozen_vae_full_test_reconstruction_label_free"
        or contract.get("status") != "complete"
        or contract.get("dataset_reference") != dataset_reference
        or contract.get("kernel_sources") != list(LATENT_KERNEL_SOURCES)
        or contract.get("producer_versions") != dict.fromkeys(LATENT_KERNEL_SOURCES, 1)
        or contract.get("spec_sha256") != sha256_file(SPEC_PATH)
        or contract.get("scorer_sha256") != sha256_file(SCORER_PATH)
        or contract.get("scorer_vector_sha256") != sha256_file(SCORER_VECTOR_PATH)
        or contract.get("evaluator_sha256") != sha256_file(EVALUATOR_PATH)
        or contract.get("evaluation_contract_sha256")
        != sha256_file(EVALUATION_CONTRACT_PATH)
        or contract.get("latent_global_audit_sha256") != GLOBAL_AUDIT_SHA256
        or contract.get("analysis") != ANALYSIS_CONTRACT
    ):
        raise ValueError("Spec 0045 input contract identity differs")
    current_reporter_sha256 = sha256_file(REPORTER_PATH)
    if contract.get("reporter_sha256") != current_reporter_sha256:
        amendment = _read_object(NORMALIZATION_AMENDMENT_PATH)
        if (
            sha256_file(NORMALIZATION_AMENDMENT_PATH) != NORMALIZATION_AMENDMENT_SHA256
            or amendment.get("original_reporter_sha256")
            != contract.get("reporter_sha256")
            or amendment.get("amended_reporter_sha256") != current_reporter_sha256
        ):
            raise ValueError("Spec 0045 reporter normalization amendment differs")
    files = cast("Mapping[str, object]", contract.get("files"))
    expected_paths = {*files, CONTRACT_NAME, "dataset-metadata.json"}
    observed_paths = {
        path.relative_to(OUTPUT_ROOT).as_posix()
        for path in OUTPUT_ROOT.rglob("*")
        if path.is_file()
    }
    if observed_paths != expected_paths:
        raise ValueError("Spec 0045 input file allow-list differs")
    forbidden = ("diagnosis", "tissue", "label", "truth", "target", "oracle")
    for relative, raw_record in files.items():
        if any(word in relative.lower() for word in forbidden):
            raise ValueError(f"Forbidden label-bearing path: {relative}")
        record = cast("Mapping[str, object]", raw_record)
        path = OUTPUT_ROOT / relative
        if path.stat().st_size != record.get("bytes") or sha256_file(
            path
        ) != record.get("sha256"):
            raise ValueError(f"Spec 0045 input byte differs: {relative}")
    header = (
        (OUTPUT_ROOT / "vae_test_locations.csv")
        .open(
            encoding="utf-8",
        )
        .readline()
    )
    if any(word in header.lower() for word in forbidden):
        raise ValueError("Spec 0045 redacted header exposes a forbidden field")
    metadata = _read_object(OUTPUT_ROOT / "dataset-metadata.json")
    if metadata.get("id") != dataset_reference:
        raise ValueError("Spec 0045 dataset metadata identity differs")
    return {
        "status": "pass",
        "dataset_reference": dataset_reference,
        "input_contract_sha256": sha256_file(contract_path),
        "input_contract_bytes": contract_path.stat().st_size,
        "redacted_location_sha256": cast(
            "Mapping[str, object]",
            cast("Mapping[str, object]", contract["population"])["redacted_location"],
        )["sha256"],
        "remote_file_count": len(expected_paths),
    }


def claim_launch(*, actor: str) -> dict[str, object]:
    """Write the exclusive launch claim only after input and kernel verification."""
    claim = _expected_launch_claim(actor=actor)
    write_exclusive_json(LAUNCH_CLAIM_PATH, claim)
    return claim


def validate_claimed_launch(*, actor: str) -> dict[str, object]:
    """Require the exact existing one-use claim immediately before upload."""
    expected = _expected_launch_claim(actor=actor)
    observed = _read_object(LAUNCH_CLAIM_PATH)
    if observed != expected:
        raise ValueError("Spec 0045 exclusive launch claim differs")
    return observed


def _expected_launch_claim(*, actor: str) -> dict[str, object]:
    validated = validate_input(actor=actor)
    receipt = _read_object(INPUT_RECEIPT_PATH)
    dataset_reference = f"{actor}/{DATASET_SLUG}"
    contract_sha256 = cast("str", validated["input_contract_sha256"])
    if (
        receipt.get("schema_version") != "spec0045.input_dataset_receipt.v1"
        or receipt.get("dataset_reference") != dataset_reference
        or receipt.get("dataset_version") != 1
        or receipt.get("visibility") != "private"
        or receipt.get("status") != "verified"
        or receipt.get("input_contract_sha256") != contract_sha256
        or receipt.get("remote_files")
        != _file_records(OUTPUT_ROOT, exclude={"dataset-metadata.json"})
    ):
        raise ValueError("Spec 0045 immutable input receipt differs")
    run_path = KERNEL_ROOT / "run.py"
    metadata_path = KERNEL_ROOT / "kernel-metadata.json"
    metadata = _read_object(metadata_path)
    if (
        metadata.get("id") != f"{actor}/eqvae-frozen-vae-test-reconstruction"
        or metadata.get("is_private") != "true"
        or metadata.get("machine_shape") != "NvidiaTeslaT4"
        or metadata.get("dataset_sources") != [dataset_reference]
        or metadata.get("kernel_sources") != list(LATENT_KERNEL_SOURCES)
        or f'INPUT_CONTRACT_SHA256 = "{contract_sha256}"'
        not in run_path.read_text(encoding="utf-8")
    ):
        raise ValueError("Spec 0045 uploadable kernel differs")
    return {
        "schema_version": "spec0045.exclusive_launch_claim.v1",
        "dataset_reference": dataset_reference,
        "dataset_version": 1,
        "input_contract_sha256": contract_sha256,
        "input_dataset_receipt_sha256": sha256_file(INPUT_RECEIPT_PATH),
        "kernel_id": metadata["id"],
        "kernel_files": {
            "kernel-metadata.json": _file_record(metadata_path),
            "run.py": _file_record(run_path),
        },
        "kernel_sources": list(LATENT_KERNEL_SOURCES),
        "policy": "one_complete_metric_result_only",
    }


def _derive_state_dict(*, branch: str, destination: Path) -> dict[str, object]:
    source = cast("Mapping[str, object]", CHECKPOINTS[branch])
    source_path = cast("Path", source["path"])
    source_sha256 = cast("str", source["sha256"])
    if sha256_file(source_path) != source_sha256:
        raise ValueError(f"Frozen {branch} checkpoint SHA-256 differs")
    model = load_frozen_checkpoint(
        source_path,
        model_name=cast("FrozenModelName", branch),
    )
    state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in model.state_dict().items()
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, destination)
    return {
        "source_checkpoint_sha256": source_sha256,
        "source_checkpoint_bytes": source_path.stat().st_size,
        "state_dict_sha256": state_dict_sha256(state),
        "state_file_sha256": sha256_file(destination),
        "state_file_bytes": destination.stat().st_size,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def _latent_source_contract(
    audit: Mapping[str, object],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    artifacts = cast("Mapping[str, object]", audit["artifacts"])
    pair_audits = cast("Mapping[str, object]", audit["pair_audits"])
    selected_wsi = set(EXPECTED_WSI_PATCH_COUNTS)
    wsi_records: dict[int, dict[str, object]] = {}
    sources: dict[str, object] = {}
    for run in range(1, 6):
        key = str(run)
        pair_record = cast("Mapping[str, object]", pair_audits[key])
        pair_path = next(
            iter(
                sorted(
                    PAIR_AUDIT_ROOT.glob(
                        f"run_{run:02d}_metadata/dataset/"
                        f"spec0021_pair_audit_run_{run:02d}_of_05.json",
                    ),
                ),
            ),
            None,
        )
        if pair_path is None:
            raise ValueError(f"Spec 0021 pair audit {run} is missing")
        observed_pair = {"name": pair_path.name, **_file_record(pair_path)}
        if observed_pair != dict(pair_record):
            raise ValueError(f"Spec 0021 pair audit {run} differs")
        pair = _read_object(pair_path)
        for raw in cast("list[object]", pair["completed_wsi_evidence"]):
            record = cast("Mapping[str, object]", raw)
            wsi_id = cast("int", record["wsi_id"])
            if wsi_id in selected_wsi:
                if wsi_id in wsi_records:
                    raise ValueError(f"Duplicate WSI evidence for {wsi_id}")
                wsi_records[wsi_id] = {
                    "wsi_id": wsi_id,
                    "png_bytes": record["png_bytes"],
                    "png_sha256": record["png_sha256"],
                }
        sources[key] = {
            "kernel_source": LATENT_KERNEL_SOURCES[run - 1],
            "producer_version": 1,
            "pair_audit": dict(pair_record),
            "normal_vae": cast("Mapping[str, object]", artifacts["normal_vae"])[key],
            "so2_vae": cast("Mapping[str, object]", artifacts["so2_vae"])[key],
        }
    if set(wsi_records) != selected_wsi:
        raise ValueError("Spec 0045 WSI PNG evidence is incomplete")
    return sources, [wsi_records[key] for key in sorted(wsi_records)]


def _file_record(path: Path) -> dict[str, int | str]:
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _file_records(root: Path, *, exclude: set[str]) -> dict[str, object]:
    return {
        path.relative_to(root).as_posix(): _file_record(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.relative_to(root).as_posix() not in exclude
    }


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return cast("dict[str, object]", value)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _validate_actor(actor: str) -> None:
    if not actor or "/" in actor or actor in {".", ".."}:
        raise ValueError("Kaggle actor must be one canonical username")


if __name__ == "__main__":
    raise SystemExit(main())
