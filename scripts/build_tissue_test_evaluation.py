# Copyright 2026 HiperMaximus
# pyright: reportAny=false
# ruff: noqa: C901, D103, EM101, EM102, PLR0912, PLR0914, PLR0916, PLR2004, T201, TRY003
"""Build, validate, claim and score the locked Spec 0043 tissue test."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import struct
import zipfile
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING, Final, cast

import torch
from torch import Tensor

from eqvae.evaluation.tissue_test_scoring import (
    score_retrieved_tissue_test_output,
)
from eqvae.kaggle_resources import KaggleResourceRef
from eqvae.models.supervised import TissueClassifier

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

ROOT: Final = Path.cwd()
OUTPUT_ROOT: Final = Path("runs/local/tissue_test_evaluation")
INPUT_RECEIPT_NAME: Final = "input_dataset_receipt.json"
LAUNCH_RECEIPT_ROOT: Final = Path("runs/local/kaggle_launches")
SPEC_PATH: Final = Path("docs/specs/0043-sealed-tissue-patch-test-evaluation.md")
SCORER_PATH: Final = Path("src/eqvae/evaluation/tissue_test_scoring.py")
TEST_VECTOR_PATH: Final = Path("docs/data/spec0043_tissue_test_scorer_vector.json")
SOURCE_ROOT: Final = Path("runs/local/tissue_label_efficiency_training_retry_v3/bundle")
SOURCE_CONTRACT: Final = SOURCE_ROOT / "tissue_training_input.json"
SOURCE_CONTRACT_SHA256: Final = (
    "154994e86cdc5c8b43536e2ad5cbe17390ae0591ee9484641b14084f967f14ca"
)
ORIGINAL_CONFIG_SHA256: Final = (
    "70a41cae61d1859bf2a9b603c32e0e7b60f5b6975bf8e57c9bbbfa2e332daeac"
)
EXECUTED_CONFIG_SHA256: Final = (
    "4c72e17c493b6beb2065d9b0828747caf46475af2ca75f33594014a446940b7e"
)
MANIFEST_ROOT: Final = Path("runs/local/ubc_ocean_supervised_manifests")
ORACLE_PATH: Final = MANIFEST_ROOT / "tissue/tissue_test.csv"
ORACLE_SHA256: Final = (
    "ff4183da11bdee4065a061410ef7ec13f45f1fe9791cdbe7139c664721b44082"
)
CATALOG_PATH: Final = MANIFEST_ROOT / "physical_parts.csv"
CATALOG_SHA256: Final = (
    "9bf2baa9c8dd8cfc079ff341f441207e3cf4e68843c7e926ea55934fdc0ef80e"
)
MANIFEST_AUDIT_PATH: Final = MANIFEST_ROOT / "spec0023_supervised_manifest_audit.json"
MANIFEST_AUDIT_SHA256: Final = (
    "805d0dc94b38b0a12a67779df30a17a6b531b99f9db6279f8b73f433d9d6de9e"
)
TRAINING_ROOT: Final = Path(
    "runs/kaggle/tissue_label_efficiency_training_v0003/tissue_label_efficiency_training",
)
TRAINING_LAUNCH_RECEIPT: Final = Path(
    "runs/local/kaggle_launches/maximshtefan/eqvae-tissue-label-efficiency-training/v0003.json",
)
TRAINING_LAUNCH_SHA256: Final = (
    "5f69bb777f80c9926f54387cfb7e767cb623bcd55e8d77d473362d4a5795198d"
)
TRAINING_OUTPUT_RECEIPT: Final = TRAINING_ROOT.parent / "kaggle_output_receipt.json"
TRAINING_OUTPUT_SHA256: Final = (
    "fc1c68b6aea1be90ca1842dba1625da35b1098e8fe68bf42cd47c599559743e7"
)
TRAINING_RUN_CONTRACT: Final = TRAINING_ROOT / "run_contract.json"
TRAINING_RUN_CONTRACT_SHA256: Final = (
    "09235066302f88086a7c55962445092216bd7c4797e1d808cbaad859c0479ecf"
)
TRAINING_RESULT: Final = TRAINING_ROOT / "spec0039_tissue_training.json"
TRAINING_RESULT_SHA256: Final = (
    "67eb1cd1a4a73dea5c6353af67b5cadc182dcbcf19b61617d806a97a7c04a9d6"
)
TRAINING_CLAIM: Final = Path(
    "runs/local/tissue_label_efficiency_training_retry_v3_authority/launch_claim.json",
)
TRAINING_CLAIM_SHA256: Final = (
    "a2332cbe9110d8205bcfaf2f5a75ffe30c80f7efd04bd1133254b8e9a9de6af2"
)
TRAINING_INPUT_RECEIPT: Final = Path(
    "runs/local/tissue_label_efficiency_training_authority/input_dataset_receipt.json",
)
TRAINING_INPUT_RECEIPT_SHA256: Final = (
    "1488baea9808baef186db7a0cb5646f1ee3aeb06275b3e312ce7078aba889a8b"
)
TEMPLATE_PATH: Final = Path(
    "kaggle/kernels/tissue_test_evaluation/run_template.py",
)
CONTRACT_NAME: Final = "tissue_test_inference_input.json"
METADATA_NAME: Final = "dataset-metadata.json"
DATASET_SLUG: Final = "eqvae-tissue-test-inputs-v1"
KERNEL_SLUG: Final = "eqvae-label-blind-tissue-test-evaluation"
TEST_ROW_COUNT: Final = 31_572
TEST_WSI_COUNT: Final = 23
BUDGETS: Final = (250, 500, 1000, 2500, 5671)
BRANCHES: Final = ("normal_vae", "so2_vae")
TEST_HEADER: Final = (
    "dataset_row",
    "atlas_row_index",
    "wsi_id",
    "x",
    "y",
    "split",
    "part",
    "file_index",
)
ORACLE_HEADER: Final = (
    "dataset_row",
    "atlas_row_index",
    "wsi_id",
    "x",
    "y",
    "tissue_label",
    "split",
    "selection_rank",
    "part",
    "file_index",
)
KERNEL_SOURCES: Final = (
    "maximusshtefan/eqvae-ubc-ocean-latent-run-01",
    "maximusshtefan/eqvae-ubc-ocean-latent-run-02",
    "maximusshtefan/eqvae-ubc-ocean-latent-run-03",
    "maximusshtefan/eqvae-ubc-ocean-latent-run-04",
    "maximusshtefan/eqvae-ubc-ocean-latent-run-05",
    "maximusshtefan/eqvae-ubc-ocean-cancer-latent-top-up",
)
MODEL_STATE_SCHEMA: Final = b"eqvae_spec0043_tissue_model_state_v1"
ROW_IDENTITY_SCHEMA: Final = b"eqvae_spec0043_tissue_row_identity_v1"
EXPECTED_PARAMETER_COUNT: Final = 106_019

CHECKPOINT_PROVENANCE: Final = {
    "normal_vae": {
        250: (
            39,
            "0367d3add18c4e41300a2a69c7d7dffa9b50d501659877dac42fa7af0e4b787e",
            "321c4ba7c90819d78652cb8c17ce9d9e422892ed1ae70f5780b1622b46aef2a4",
        ),
        500: (
            24,
            "eb9366cf870c71ea6beb743aa33c36ef3fabb37f7dbbff47a143b9a033ed0567",
            "5de6a137d6fca60fc330ee9e89aa25a1a95ba63a838d4d5c21bf87bbf5e10d43",
        ),
        1000: (
            240,
            "6ddc54185e749e0fef29ecbc6328c55679d931de02dbf25b829f013382b2a4f1",
            "09113b17b49263371ff310fdd4951c8c625afa671a73b3815b045ce8280c60e3",
        ),
        2500: (
            360,
            "2155dd5723acc7b9b96e2f1a19570a6a8ddbe6cfc9e847c32c4f8765556c350d",
            "21080c819aa1d908c13289b9b74c0620b583073dc9e63f110af893b780d2da22",
        ),
        5671: (
            642,
            "b8ea17f05cfd354545492a254410c9f0631d247208ffd46659053d279ed6cee6",
            "62d3b0f9fc3b7bd744947fb18bd6352447aac7e3309c331919a97900620f0d58",
        ),
    },
    "so2_vae": {
        250: (
            45,
            "beaa92f0b441ef1071aaa882d3e5019126ec848730d713d74c58a83022250297",
            "cd0873407c22a8df298f05dedf2ebff603760b77e69908a7a6e69bd68667568d",
        ),
        500: (
            60,
            "51830e80e5d18c47c834e153c6ea9b952cbc9d8892c3e32a6f3fa833c8d7a142",
            "5fa0e9c97b2476f10f324ae7bccdbb3b2b3a0099699bfa602a95c4168a64457a",
        ),
        1000: (
            108,
            "0203475529d512c50a2193a7c97838bfab781c2ef46300b42b4064a7ec7d4720",
            "7ca1ede8b0d8f75ede1e4f31ed571ca92338aee25d240f12be03af94025d8e84",
        ),
        2500: (
            240,
            "81cbc941c07d158c7f89c5dc8c0afffbd58453e6639facd0da1cb93ec76bf0ad",
            "749644887e3ebaf4104d67057ccd3d22c82982d9580fa04b99dbbb724f83be6d",
        ),
        5671: (
            802,
            "93d3848dfbdc962ad15fb7bd8344482d643a20958e787294d0228b30a60d9c7b",
            "f9cca25c9b9f482f341192843e470e0aa4b3da2eba1bef9a997c22b0ad83f273",
        ),
    },
}


def build(*, actor: str) -> dict[str, object]:
    dataset_reference = KaggleResourceRef(
        owner=actor,
        slug=DATASET_SLUG,
    ).canonical_id
    output = ROOT / OUTPUT_ROOT
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    staging = output.with_name(f".{output.name}.building")
    if staging.exists():
        raise FileExistsError(f"Stale staging directory exists: {staging}")
    try:
        _build_staging(staging, dataset_reference=dataset_reference, actor=actor)
        staging.replace(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return validate(expected_actor=actor)


def _build_staging(staging: Path, *, dataset_reference: str, actor: str) -> None:
    _validate_authorities()
    bundle = staging / "bundle"
    test = bundle / "test"
    test.mkdir(parents=True)
    shutil.copy2(ROOT / CATALOG_PATH, test / "physical_parts.csv")
    test_rows = _derive_label_free_test(test / "tissue_test_locations.csv")
    _copy_frozen_source(bundle / "src")
    weights = bundle / "weights"
    weights.mkdir()
    model_records = {
        branch: {
            str(budget): _derive_model_state(
                branch,
                budget,
                weights / branch / f"budget_{budget:04d}_per_class.pt",
            )
            for budget in BUDGETS
        }
        for branch in BRANCHES
    }
    scorer_sha256 = _sha256(ROOT / SCORER_PATH)
    vector_sha256 = _sha256(ROOT / TEST_VECTOR_PATH)
    contract: dict[str, object] = {
        "schema_version": "spec0043.label_blind_input.v1",
        "scope": "sealed_tissue_test_label_blind_inference_only",
        "visibility": "private",
        "dataset_reference": dataset_reference,
        "dataset_actor": actor,
        "kernel_id": f"{actor}/{KERNEL_SLUG}",
        "spec_sha256": _sha256(ROOT / SPEC_PATH),
        "scorer_sha256": scorer_sha256,
        "test_vector_sha256": vector_sha256,
        "source_contract_sha256": SOURCE_CONTRACT_SHA256,
        "training_provenance": {
            "kernel_reference": "maximshtefan/eqvae-tissue-label-efficiency-training/3",
            "launch_receipt_sha256": TRAINING_LAUNCH_SHA256,
            "output_receipt_sha256": TRAINING_OUTPUT_SHA256,
            "run_contract_sha256": TRAINING_RUN_CONTRACT_SHA256,
            "result_sha256": TRAINING_RESULT_SHA256,
            "launch_claim_sha256": TRAINING_CLAIM_SHA256,
            "input_receipt_sha256": TRAINING_INPUT_RECEIPT_SHA256,
            "original_configuration_sha256": ORIGINAL_CONFIG_SHA256,
            "executed_configuration_sha256": EXECUTED_CONFIG_SHA256,
        },
        "test": {
            "row_count": TEST_ROW_COUNT,
            "wsi_count": TEST_WSI_COUNT,
            "class_order_local_only": False,
            "locations": _file_record(test / "tissue_test_locations.csv"),
            "row_identity_sha256": _row_identity_sha256(test_rows),
            "label_fields_present": False,
        },
        "physical_sources": _read_csv(test / "physical_parts.csv"),
        "producer_versions": dict.fromkeys(KERNEL_SOURCES, 1),
        "kernel_sources": list(KERNEL_SOURCES),
        "budgets_per_class": list(BUDGETS),
        "branches": model_records,
        "runtime": {
            "torch": "2.14.0",
            "cuda": "13.0",
            "index": "https://download.pytorch.org/whl/cu130",
            "devices": {"normal_vae": 0, "so2_vae": 1},
            "batch_size": 159,
            "final_batch_size": 90,
            "compile": False,
            "optimizer_updates": 0,
        },
    }
    contract["files"] = _artifact_records(
        bundle,
        exclude={CONTRACT_NAME, METADATA_NAME},
    )
    _write_json(bundle / CONTRACT_NAME, contract)
    metadata = _dataset_metadata(dataset_reference)
    _write_json(bundle / METADATA_NAME, metadata)
    upload = staging / "upload"
    upload.mkdir()
    shutil.copy2(bundle / METADATA_NAME, upload / METADATA_NAME)
    with zipfile.ZipFile(upload / "bundle.zip", "w", zipfile.ZIP_STORED) as archive:
        for path in sorted(bundle.rglob("*")):
            if path.is_file() and path.name != METADATA_NAME:
                archive.write(path, path.relative_to(bundle).as_posix())
    kernel = staging / "kernel"
    kernel.mkdir()
    _write_json(
        kernel / "kernel-metadata.json",
        _kernel_metadata(dataset_reference, actor),
    )
    rendered = (
        Template((ROOT / TEMPLATE_PATH).read_text(encoding="utf-8"))
        .substitute(
            input_contract_sha256=_sha256(bundle / CONTRACT_NAME),
            input_dataset_reference=dataset_reference,
        )
        .encode()
    )
    compile(rendered, str(TEMPLATE_PATH), "exec")
    (kernel / "run.py").write_bytes(rendered)


def validate(*, expected_actor: str | None = None) -> dict[str, object]:
    output = ROOT / OUTPUT_ROOT
    bundle = output / "bundle"
    contract = _read_object(bundle / CONTRACT_NAME)
    dataset_reference = str(contract.get("dataset_reference"))
    actor = KaggleResourceRef.parse(dataset_reference, allow_version=False).owner
    if expected_actor is not None and actor != expected_actor:
        raise ValueError("Spec 0043 dataset actor differs")
    _validate_authorities()
    if (
        contract.get("schema_version") != "spec0043.label_blind_input.v1"
        or contract.get("scope") != "sealed_tissue_test_label_blind_inference_only"
        or contract.get("visibility") != "private"
        or contract.get("dataset_actor") != actor
        or contract.get("kernel_id") != f"{actor}/{KERNEL_SLUG}"
        or contract.get("spec_sha256") != _sha256(ROOT / SPEC_PATH)
        or contract.get("scorer_sha256") != _sha256(ROOT / SCORER_PATH)
        or contract.get("test_vector_sha256") != _sha256(ROOT / TEST_VECTOR_PATH)
        or contract.get("source_contract_sha256") != SOURCE_CONTRACT_SHA256
        or contract.get("kernel_sources") != list(KERNEL_SOURCES)
        or contract.get("producer_versions") != dict.fromkeys(KERNEL_SOURCES, 1)
        or contract.get("budgets_per_class") != list(BUDGETS)
    ):
        raise ValueError("Spec 0043 input contract identity differs")
    files = cast("dict[str, object]", contract.get("files"))
    if files != _artifact_records(bundle, exclude={CONTRACT_NAME, METADATA_NAME}):
        raise ValueError("Spec 0043 bundle bytes differ")
    _validate_tree(bundle, {*files, CONTRACT_NAME, METADATA_NAME})
    test_rows = _validate_test_stage(bundle / "test")
    test_contract = cast("dict[str, object]", contract["test"])
    if test_contract.get("locations") != _file_record(
        bundle / "test/tissue_test_locations.csv",
    ) or test_contract.get("row_identity_sha256") != _row_identity_sha256(test_rows):
        raise ValueError("Spec 0043 test identity differs")
    if contract.get("physical_sources") != _read_csv(
        bundle / "test/physical_parts.csv",
    ):
        raise ValueError("Spec 0043 physical catalog differs")
    _validate_source_stage(bundle / "src")
    branches = cast("dict[str, dict[str, object]]", contract["branches"])
    for branch in BRANCHES:
        if set(branches[branch]) != {str(value) for value in BUDGETS}:
            raise ValueError("Spec 0043 staged budget set differs")
        for budget in BUDGETS:
            record = cast("dict[str, object]", branches[branch][str(budget)])
            path = bundle / str(record["path"])
            state = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(state, dict):
                raise TypeError("Spec 0043 staged model state is not a mapping")
            provenance = _provenance_record(branch, budget)
            if (
                record.get("source") != provenance
                or record.get("file_sha256") != _sha256(path)
                or record.get("state_sha256")
                != _state_dict_sha256(cast("dict[str, Tensor]", state))
                or record.get("parameter_count") != EXPECTED_PARAMETER_COUNT
            ):
                raise ValueError(f"Spec 0043 model state differs: {branch}/{budget}")
            model = TissueClassifier()
            model.load_state_dict(cast("dict[str, Tensor]", state), strict=True)
    metadata = _dataset_metadata(dataset_reference)
    if _read_object(bundle / METADATA_NAME) != metadata:
        raise ValueError("Spec 0043 dataset metadata differs")
    _validate_upload(output / "upload", bundle, metadata)
    kernel = output / "kernel"
    if _read_object(kernel / "kernel-metadata.json") != _kernel_metadata(
        dataset_reference,
        actor,
    ):
        raise ValueError("Spec 0043 kernel metadata differs")
    expected_run = (
        Template((ROOT / TEMPLATE_PATH).read_text(encoding="utf-8"))
        .substitute(
            input_contract_sha256=_sha256(bundle / CONTRACT_NAME),
            input_dataset_reference=dataset_reference,
        )
        .encode()
    )
    run_path = kernel / "run.py"
    if run_path.read_bytes() != expected_run or run_path.stat().st_size >= 1_000_000:
        raise ValueError("Spec 0043 rendered kernel differs")
    compile(expected_run, str(TEMPLATE_PATH), "exec")
    return contract


def claim_launch(*, expected_actor: str) -> dict[str, object]:
    contract = validate(expected_actor=expected_actor)
    root = ROOT / OUTPUT_ROOT
    input_receipt = _validate_input_receipt(contract)
    claim = _expected_launch_claim(contract, input_receipt=input_receipt)
    _write_exclusive_json(root / "launch_claim.json", claim)
    return claim


def validate_claimed_launch(*, expected_actor: str) -> dict[str, object]:
    """Validate the sealed input and claim immediately before generic upload.

    Returns:
        The exact authenticated exclusive launch claim.

    Raises:
        ValueError: If the sealed input or launch claim differs.
        FileExistsError: If an accepted launch receipt already exists.

    """
    contract = validate(expected_actor=expected_actor)
    root = ROOT / OUTPUT_ROOT
    input_receipt = _validate_input_receipt(contract)
    claim = _expected_launch_claim(contract, input_receipt=input_receipt)
    if _read_object(root / "launch_claim.json") != claim:
        raise ValueError("Spec 0043 exclusive launch claim differs")
    receipt_dir = ROOT / LAUNCH_RECEIPT_ROOT / expected_actor / KERNEL_SLUG
    if receipt_dir.exists() and any(receipt_dir.glob("v*.json")):
        raise FileExistsError("Spec 0043 accepted launch receipt already exists")
    return claim


def _expected_launch_claim(
    contract: Mapping[str, object],
    *,
    input_receipt: Mapping[str, int | str],
) -> dict[str, object]:
    root = ROOT / OUTPUT_ROOT
    return {
        "schema_version": "spec0043.exclusive_launch_claim.v1",
        "status": "claimed_before_remote_push",
        "authorization": "ok let's do the patch tissue test evaluation",
        "authorization_date": "2026-09-06",
        "dataset_reference": contract["dataset_reference"],
        "kernel_id": contract["kernel_id"],
        "input_contract_sha256": _sha256(root / "bundle" / CONTRACT_NAME),
        "input_dataset_receipt": dict(input_receipt),
        "kernel_files": {
            name: _file_record(root / "kernel" / name)
            for name in ("kernel-metadata.json", "run.py")
        },
        "scientific_retries_authorized": 0,
    }


def _validate_input_receipt(
    contract: Mapping[str, object],
) -> dict[str, int | str]:
    root = ROOT / OUTPUT_ROOT
    receipt_path = root / INPUT_RECEIPT_NAME
    expected = {
        "schema_version": "spec0043.input_dataset_receipt.v1",
        "dataset_reference": contract["dataset_reference"],
        "dataset_version": 1,
        "visibility": "private",
        "status": "verified",
        "input_contract_sha256": _sha256(root / "bundle" / CONTRACT_NAME),
        "remote_files": _artifact_records(
            root / "bundle",
            exclude={METADATA_NAME},
        ),
    }
    if _read_object(receipt_path) != expected:
        raise ValueError("Spec 0043 verified input-dataset receipt differs")
    return _file_record(receipt_path)


def score(
    *,
    remote_output_root: Path,
    launch_receipt_path: Path,
    output_root: Path,
) -> dict[str, object]:
    contract = validate()
    root = ROOT / OUTPUT_ROOT
    input_receipt = _validate_input_receipt(contract)
    return score_retrieved_tissue_test_output(
        remote_output_root=remote_output_root,
        launch_receipt_path=launch_receipt_path,
        launch_claim_path=root / "launch_claim.json",
        label_oracle_path=ROOT / ORACLE_PATH,
        output_root=output_root,
        expected_contract=contract,
        expected_input_contract_sha256=_sha256(root / "bundle" / CONTRACT_NAME),
        expected_input_receipt_sha256=str(input_receipt["sha256"]),
        expected_input_receipt_bytes=int(input_receipt["bytes"]),
        expected_kernel_sha256=_sha256(root / "kernel/run.py"),
        expected_metadata_sha256=_sha256(root / "kernel/kernel-metadata.json"),
        expected_kernel_bytes=(root / "kernel/run.py").stat().st_size,
        expected_metadata_bytes=(root / "kernel/kernel-metadata.json").stat().st_size,
    )


def _validate_authorities() -> None:
    fixed = {
        ROOT / SOURCE_CONTRACT: SOURCE_CONTRACT_SHA256,
        ROOT / SOURCE_ROOT / "tissue_training_config.json": ORIGINAL_CONFIG_SHA256,
        ROOT / ORACLE_PATH: ORACLE_SHA256,
        ROOT / CATALOG_PATH: CATALOG_SHA256,
        ROOT / MANIFEST_AUDIT_PATH: MANIFEST_AUDIT_SHA256,
        ROOT / TRAINING_LAUNCH_RECEIPT: TRAINING_LAUNCH_SHA256,
        ROOT / TRAINING_OUTPUT_RECEIPT: TRAINING_OUTPUT_SHA256,
        ROOT / TRAINING_RUN_CONTRACT: TRAINING_RUN_CONTRACT_SHA256,
        ROOT / TRAINING_RESULT: TRAINING_RESULT_SHA256,
        ROOT / TRAINING_CLAIM: TRAINING_CLAIM_SHA256,
        ROOT / TRAINING_INPUT_RECEIPT: TRAINING_INPUT_RECEIPT_SHA256,
    }
    for path, expected in fixed.items():
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"Spec 0043 authority differs: {path}")
    receipt = _read_object(ROOT / TRAINING_OUTPUT_RECEIPT)
    declared = cast("dict[str, object]", receipt.get("files"))
    receipt_root = (ROOT / TRAINING_OUTPUT_RECEIPT).parent
    observed = {
        path.relative_to(receipt_root).as_posix(): _file_record(path)
        for path in sorted(receipt_root.rglob("*"))
        if path.is_file() and path != ROOT / TRAINING_OUTPUT_RECEIPT
    }
    if declared != observed:
        raise ValueError("Spec 0043 prior output receipt does not bind every byte")
    run_contract = _read_object(ROOT / TRAINING_RUN_CONTRACT)
    result = _read_object(ROOT / TRAINING_RESULT)
    if (
        run_contract.get("configuration_sha256") != EXECUTED_CONFIG_SHA256
        or run_contract.get("input_contract_sha256") != SOURCE_CONTRACT_SHA256
        or result.get("status") != "complete"
        or result.get("scope") != "tissue_development_training_no_sealed_test"
    ):
        raise ValueError("Spec 0043 prior training completion differs")
    _validate_source_authority()


def _validate_source_authority() -> None:
    contract = _read_object(ROOT / SOURCE_CONTRACT)
    files = cast("dict[str, dict[str, object]]", contract.get("files"))
    declared = {name for name in files if name.startswith("src/")}
    observed = {
        path.relative_to(ROOT / SOURCE_ROOT).as_posix()
        for path in (ROOT / SOURCE_ROOT / "src").rglob("*.py")
    }
    if (
        contract.get("schema_version") != "spec0039.tissue_training_input.v1"
        or contract.get("scope") != "tissue_development_training_no_sealed_test"
        or contract.get("configuration_sha256") != ORIGINAL_CONFIG_SHA256
        or declared != observed
    ):
        raise ValueError("Spec 0043 frozen source authority differs")
    for name in declared:
        path = ROOT / SOURCE_ROOT / name
        record = files[name]
        if path.stat().st_size != record["bytes"] or _sha256(path) != record["sha256"]:
            raise ValueError(f"Spec 0043 frozen source byte differs: {name}")


def _copy_frozen_source(destination: Path) -> None:
    _validate_source_authority()
    shutil.copytree(ROOT / SOURCE_ROOT / "src", destination)


def _derive_label_free_test(destination: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with (ROOT / ORACLE_PATH).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != ORACLE_HEADER:
            raise ValueError("Spec 0043 tissue oracle header differs")
        for index, row in enumerate(reader):
            selected = {name: row[name] for name in TEST_HEADER}
            if int(selected["dataset_row"]) != index or selected["split"] != "test":
                raise ValueError("Spec 0043 tissue oracle order/split differs")
            rows.append(selected)
    if len(rows) != TEST_ROW_COUNT or len({tuple(row.values()) for row in rows}) != len(
        rows,
    ):
        raise ValueError("Spec 0043 tissue oracle row identity differs")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=TEST_HEADER)
        writer.writeheader()
        writer.writerows(rows)  # pyright: ignore[reportArgumentType]
    return rows


def _validate_test_stage(test: Path) -> list[dict[str, str]]:
    if {path.name for path in test.iterdir()} != {
        "physical_parts.csv",
        "tissue_test_locations.csv",
    }:
        raise ValueError("Spec 0043 test stage file set differs")
    if _sha256(test / "physical_parts.csv") != CATALOG_SHA256:
        raise ValueError("Spec 0043 staged physical catalog differs")
    with (test / "tissue_test_locations.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != TEST_HEADER:
            raise ValueError("Spec 0043 label-free test header differs")
        rows = list(reader)
    if (
        len(rows) != TEST_ROW_COUNT
        or any(
            row["split"] != "test" or int(row["dataset_row"]) != index
            for index, row in enumerate(rows)
        )
        or any(
            any(
                word in name.lower()
                for word in ("label", "truth", "target", "diagnosis", "selection")
            )
            for name in TEST_HEADER
        )
    ):
        raise ValueError("Spec 0043 staged test rows differ")
    return rows


def _checkpoint_paths(branch: str, budget: int) -> tuple[Path, Path]:
    root = (
        TRAINING_ROOT / "branches" / branch / f"budget_{budget:04d}_per_class" / branch
    )
    return ROOT / root / "best.json", ROOT / root / "best.pt"


def _provenance_record(branch: str, budget: int) -> dict[str, object]:
    update, pointer_sha256, checkpoint_sha256 = CHECKPOINT_PROVENANCE[branch][budget]
    return {
        "branch": branch,
        "budget_per_class": budget,
        "selected_update_index": update,
        "pointer_sha256": pointer_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "configuration_sha256": EXECUTED_CONFIG_SHA256,
    }


def _derive_model_state(
    branch: str,
    budget: int,
    destination: Path,
) -> dict[str, object]:
    pointer_path, checkpoint_path = _checkpoint_paths(branch, budget)
    provenance = _provenance_record(branch, budget)
    if (
        _sha256(pointer_path) != provenance["pointer_sha256"]
        or _sha256(checkpoint_path) != provenance["checkpoint_sha256"]
    ):
        raise ValueError(
            f"Spec 0043 selected checkpoint byte differs: {branch}/{budget}",
        )
    pointer = _read_object(pointer_path)
    loaded = cast(
        "object",
        torch.load(checkpoint_path, map_location="cpu", weights_only=True),
    )
    if not isinstance(loaded, dict):
        raise TypeError("Spec 0043 selected checkpoint payload differs")
    payload = cast("dict[str, object]", loaded)
    if set(payload) != {"state_dict", "manifest"}:
        raise ValueError("Spec 0043 selected checkpoint payload differs")
    expected_manifest = {
        key: value for key, value in pointer.items() if key != "file_sha256"
    }
    selection = cast("dict[str, object]", pointer.get("selection"))
    if (
        pointer.get("branch") != branch
        or pointer.get("config_sha256") != EXECUTED_CONFIG_SHA256
        or pointer.get("file_sha256") != provenance["checkpoint_sha256"]
        or selection.get("update_index") != provenance["selected_update_index"]
        or payload.get("manifest") != expected_manifest
    ):
        raise ValueError(
            f"Spec 0043 selected checkpoint manifest differs: {branch}/{budget}",
        )
    state = cast("dict[str, Tensor]", payload["state_dict"])
    if len(state) != 14 or any(
        tensor.dtype != torch.float32 for tensor in state.values()
    ):
        raise ValueError("Spec 0043 selected state tensor contract differs")
    model = TissueClassifier()
    model.load_state_dict(state, strict=True)
    if (
        sum(parameter.numel() for parameter in model.parameters())
        != EXPECTED_PARAMETER_COUNT
    ):
        raise ValueError("Spec 0043 tissue classifier parameter count differs")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, destination)
    return {
        "path": destination.relative_to(destination.parents[2]).as_posix(),
        "file_sha256": _sha256(destination),
        "state_sha256": _state_dict_sha256(state),
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "source": provenance,
    }


def _state_dict_sha256(state: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256(MODEL_STATE_SCHEMA)
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        metadata = json.dumps(
            {"dtype": str(tensor.dtype), "name": name, "shape": list(tensor.shape)},
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        payload = tensor.numpy().tobytes(order="C")
        digest.update(struct.pack("<Q", len(metadata)))
        digest.update(metadata)
        digest.update(struct.pack("<Q", len(payload)))
        digest.update(payload)
    return digest.hexdigest()


def _row_identity_sha256(rows: Sequence[Mapping[str, str]]) -> str:
    digest = hashlib.sha256(ROW_IDENTITY_SCHEMA)
    for row in rows:
        for name in TEST_HEADER:
            digest.update(row[name].encode())
            digest.update(b"\0")
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_source_stage(destination: Path) -> None:
    contract = _read_object(ROOT / SOURCE_CONTRACT)
    files = cast("dict[str, dict[str, object]]", contract["files"])
    expected = {
        name[4:]: record for name, record in files.items() if name.startswith("src/")
    }
    observed = {
        path.relative_to(destination).as_posix(): _file_record(path)
        for path in sorted(destination.rglob("*"))
        if path.is_file()
    }
    if observed != expected or any(
        path.is_symlink() for path in destination.rglob("*")
    ):
        raise ValueError("Spec 0043 frozen source stage differs")


def _dataset_metadata(reference: str) -> dict[str, object]:
    return {
        "id": reference,
        "title": "eqvae label blind tissue test inputs",
        "licenses": [{"name": "other"}],
    }


def _kernel_metadata(reference: str, actor: str) -> dict[str, object]:
    return {
        "id": f"{actor}/{KERNEL_SLUG}",
        "title": "eqvae label blind tissue test evaluation",
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [reference],
        "competition_sources": [],
        "kernel_sources": list(KERNEL_SOURCES),
        "model_sources": [],
    }


def _validate_upload(upload: Path, bundle: Path, metadata: dict[str, object]) -> None:
    if _read_object(upload / METADATA_NAME) != metadata:
        raise ValueError("Spec 0043 upload metadata differs")
    with zipfile.ZipFile(upload / "bundle.zip") as archive:
        expected = {
            path.relative_to(bundle).as_posix(): path
            for path in bundle.rglob("*")
            if path.is_file() and path.name != METADATA_NAME
        }
        if set(archive.namelist()) != set(expected) or any(
            archive.read(name) != path.read_bytes() for name, path in expected.items()
        ):
            raise ValueError("Spec 0043 upload archive differs")


def _validate_tree(root: Path, expected_files: set[str]) -> None:
    observed = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    if observed != expected_files or any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("Spec 0043 bundle tree differs")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _artifact_records(root: Path, *, exclude: set[str]) -> dict[str, object]:
    return {
        path.relative_to(root).as_posix(): _file_record(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in exclude
    }


def _file_record(path: Path) -> dict[str, int | str]:
    return {"bytes": path.stat().st_size, "sha256": _sha256(path)}


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _read_object(path: Path) -> dict[str, object]:
    value = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return cast("dict[str, object]", value)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_exclusive_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", closefd=False) as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--actor", required=True)
    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("--actor")
    claim_parser = sub.add_parser("claim-launch")
    claim_parser.add_argument("--actor", required=True)
    claimed_parser = sub.add_parser("validate-claimed-launch")
    claimed_parser.add_argument("--actor", required=True)
    score_parser = sub.add_parser("score")
    score_parser.add_argument("--remote-output-root", type=Path, required=True)
    score_parser.add_argument("--launch-receipt", type=Path, required=True)
    score_parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        result = build(actor=args.actor)
    elif args.command == "validate":
        result = validate(expected_actor=args.actor)
    elif args.command == "claim-launch":
        result = claim_launch(expected_actor=args.actor)
    elif args.command == "validate-claimed-launch":
        result = validate_claimed_launch(expected_actor=args.actor)
    else:
        result = score(
            remote_output_root=args.remote_output_root,
            launch_receipt_path=args.launch_receipt,
            output_root=args.output_root,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
