# Copyright 2026 HiperMaximus
# ruff: noqa: C901, D103, E501, EM101, EM102, PLR0914, PLR0916, PLR2004, T201, TRY003
"""Build and score the locked label-blind Spec 0041 MIL test evaluation."""

from __future__ import annotations

# pyright: reportAny=false
import argparse
import csv
import hashlib
import json
import shutil
import struct
import zipfile
from pathlib import Path
from string import Template
from types import SimpleNamespace
from typing import TYPE_CHECKING, Final, cast

import torch
from torch import Tensor

from eqvae.evaluation.mil_test_scoring import score_retrieved_mil_test_output
from eqvae.kaggle_resources import KaggleResourceRef
from eqvae.models.local_global_mil import (
    EXPECTED_PARAMETER_COUNT,
    LocalGlobalMILClassifier,
    build_local_attention_graph,
)
from eqvae.training.mil_training import load_branch_checkpoint

if TYPE_CHECKING:
    from collections.abc import Sequence

    from eqvae.data.supervised_latents import WSIInstance

ROOT: Final = Path.cwd()
OUTPUT_ROOT: Final = Path("runs/local/ubc_ocean_mil_test_evaluation")
SPEC_PATH: Final = Path("docs/specs/0041-sealed-mil-test-evaluation.md")
SCORER_PATH: Final = Path("src/eqvae/evaluation/mil_test_scoring.py")
TEST_VECTOR_PATH: Final = Path("docs/data/spec0041_mil_test_scorer_vector.json")
NORMALIZATION_AMENDMENT_PATH: Final = Path(
    "docs/data/spec0042_spec0041_slug_normalization_amendment.json",
)
NORMALIZATION_AMENDMENT_SHA256: Final = (
    "0f63933d656cb9c8143762a6b9dabbf42d42b2b17c4670bd7f7f9a325826b3eb"
)
SOURCE_ROOT: Final = Path("runs/local/ubc_ocean_mil_training_v3/bundle")
SOURCE_CONTRACT: Final = SOURCE_ROOT / "mil_training_input.json"
SOURCE_CONTRACT_SHA256: Final = (
    "c6b1b2ace6ed6cc8de5b1856fde74348a4895ee91eb9a5bbc1a22ae6d4a9f4f9"
)
TEST_ROOT: Final = Path("runs/local/ubc_ocean_full_foreground_manifests/sealed_test")
CATALOG_PATH: Final = Path(
    "runs/local/ubc_ocean_full_foreground_manifests/development/physical_parts.csv",
)
INTEGRATION_AUDIT_PATH: Final = Path(
    "runs/local/ubc_ocean_full_foreground_manifests/integration_audit.json",
)
TRAINING_OUTPUT_ROOT: Final = Path("runs/kaggle/ubc_ocean_mil_training_v7")
TRAINING_LAUNCH_RECEIPT: Final = Path(
    "runs/local/kaggle_launches/maximshtefan/eqvae-local-global-mil-training/v0007.json",
)
TRAINING_OUTPUT_RECEIPT: Final = TRAINING_OUTPUT_ROOT / "kaggle_output_receipt.json"
TEMPLATE_PATH: Final = Path(
    "kaggle/kernels/ubc_ocean_mil_test_evaluation/run_template.py",
)
CONTRACT_NAME: Final = "mil_test_inference_input.json"
METADATA_NAME: Final = "dataset-metadata.json"
DATASET_SLUG: Final = "eqvae-local-global-mil-test-inputs-v1"
KERNEL_SLUG: Final = "eqvae-local-global-mil-test-evaluation"
TEST_WSI_COUNT: Final = 23
TEST_INSTANCE_COUNT: Final = 261_168
TEST_FILE_HASHES: Final = {
    "wsi_cancer_test_bags.csv": (
        "8243f52abb68bea57a4a98083467a8a883999117a44908d8592627c74c160601"
    ),
    "wsi_cancer_test_instances.csv": (
        "eda697c7f8a9408fef7a2d42f4aa1d74263323ae108cbe698a5c6c485d867f69"
    ),
    "physical_parts.csv": (
        "9303120aa99ab105eafdb14868bd8e1a1f785b643f149f103bb60beeabf8d92e"
    ),
}
INTEGRATION_AUDIT_SHA256: Final = (
    "9b22ef9e3c58f59b4d88f4518749b9eee3e585b5034c2c1440dc31695e10860a"
)
TRAINING_LAUNCH_SHA256: Final = (
    "e6e6e3ad366ccdc2eec7655e4576c76ed2c6c506da541ec6c421db736b674cb6"
)
TRAINING_OUTPUT_SHA256: Final = (
    "13ee0a2ccce4d9301777cd5e02bc0158d9f231843db178324137c58baf63db4a"
)
EXPECTED_TRAINING_HASHES: Final = {
    "candidate_sha256": "bd16522fc5192ec330a0f46ba8f6653e6141f9c746f670c4dcbb3330d2f88a5c",
    "development_contract_sha256": "6aa25a3f56db62b903d3451e154c47dc0039233f88179c7bdab03b7a48319fe0",
    "initial_state_file_sha256": "c3d5a46b704e46a8c56e420d2727b5c0842f9be9afedcac47dba76e341bf4be4",
    "initial_state_sha256": "4f667c26c68d6c64d84d0931a9995d7fa80ff49e968bdbaa162ae107210ee705",
    "input_contract_sha256": SOURCE_CONTRACT_SHA256,
    "model_sha256": "9d4513c6f7d63aeb13ffc29f7586d1b336daddba3fa2daca3c2a9c45e53a7c72",
    "runtime_sha256": "3d23f1ec9fa534aa24e9657113cdd5f23e10914be2c9649f63c778db412a0c24",
    "spec_sha256": "8c92e4eb7458bbff955d9a8c964ebabb1ff554f35c4d78384eed2749abc864e7",
    "training_config_sha256": "52cc294d009850f1dd623e189c241cde3a4017dce715133b36e19ec530d41eee",
}
BRANCH_PROVENANCE: Final = {
    "normal_vae": {
        "boundary": 3127,
        "checkpoint_sha256": "1dcd17cbebcb9bf97ca9492681b788c1cca2c170cf4189eefcd93da0106b4806",
        "manifest_sha256": "b9e7aee6b8a99ea6b80495ec026ed59231739a09aad5c8a4c8461dcda7fd1aef",
    },
    "so2_vae": {
        "boundary": 5671,
        "checkpoint_sha256": "47e82979dc2fcf8b3b1793e1687a3efc8591b9c815f92dac5c632845e681914a",
        "manifest_sha256": "f40747bd11ecae3c47092a9f8867f28db9c8240ce2e629ebd6b4440ca530b1fd",
    },
}
KERNEL_SOURCES: Final = (
    "maximusshtefan/eqvae-ubc-ocean-latent-run-01",
    "maximusshtefan/eqvae-ubc-ocean-latent-run-02",
    "maximusshtefan/eqvae-ubc-ocean-latent-run-03",
    "maximusshtefan/eqvae-ubc-ocean-latent-run-04",
    "maximusshtefan/eqvae-ubc-ocean-latent-run-05",
    "maximusshtefan/eqvae-ubc-ocean-cancer-latent-top-up",
    "maximusshtefan/eqvae-wsi45630-completion",
    "maximusshtefan/eqvae-full-foreground-01",
    "maximusshtefan/eqvae-full-foreground-02",
    "maximusshtefan/eqvae-full-foreground-03",
    "maximusshtefan/eqvae-full-foreground-04",
    "maximusshtefan/eqvae-full-foreground-05",
    "maximusshtefan/eqvae-full-foreground-06",
    "maximusshtefan/eqvae-full-foreground-07",
    "maximusshtefan/eqvae-full-foreground-08",
)
MODEL_STATE_SCHEMA: Final = b"eqvae_spec0041_model_state_v1"


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
    for name in ("wsi_cancer_test_bags.csv", "wsi_cancer_test_instances.csv"):
        shutil.copy2(ROOT / TEST_ROOT / name, test / name)
    _copy_frozen_source(bundle / "src")
    weights = bundle / "weights"
    weights.mkdir()
    branch_records = {
        branch: _derive_model_state(branch, weights / f"{branch}.pt")
        for branch in BRANCH_PROVENANCE
    }
    graph_identities = _test_graph_identities(test)
    bag_sizes = {
        row["wsi_id"]: int(row["instance_count"])
        for row in _read_csv(test / "wsi_cancer_test_bags.csv")
    }
    scorer_sha256 = _sha256(ROOT / SCORER_PATH)
    test_vector_sha256 = _sha256(ROOT / TEST_VECTOR_PATH)
    physical_rows = _read_csv(test / "physical_parts.csv")
    contract: dict[str, object] = {
        "schema_version": "spec0041.label_blind_input.v1",
        "scope": "sealed_test_label_blind_inference_only",
        "visibility": "private",
        "dataset_reference": dataset_reference,
        "dataset_actor": actor,
        "kernel_id": f"{actor}/{KERNEL_SLUG}",
        "spec_sha256": _sha256(ROOT / SPEC_PATH),
        "scorer_sha256": scorer_sha256,
        "test_vector_sha256": test_vector_sha256,
        "source_contract_sha256": SOURCE_CONTRACT_SHA256,
        "training_provenance": {
            "kernel_reference": "maximshtefan/eqvae-local-global-mil-training/7",
            "launch_receipt_sha256": TRAINING_LAUNCH_SHA256,
            "output_receipt_sha256": TRAINING_OUTPUT_SHA256,
            "contract_hashes": EXPECTED_TRAINING_HASHES,
        },
        "test": {
            "wsi_count": TEST_WSI_COUNT,
            "instance_count": TEST_INSTANCE_COUNT,
            "files": {name: _file_record(test / name) for name in TEST_FILE_HASHES},
            "graph_identities": graph_identities,
            "bag_sizes": bag_sizes,
            "label_fields_present": False,
        },
        "physical_sources": physical_rows,
        "producer_versions": dict.fromkeys(KERNEL_SOURCES, 1),
        "kernel_sources": list(KERNEL_SOURCES),
        "branches": branch_records,
        "runtime": {
            "torch": "2.14.0",
            "cuda": "13.0",
            "index": "https://download.pytorch.org/whl/cu130",
            "devices": {"normal_vae": 0, "so2_vae": 1},
            "compile": "inductor_fullgraph_max-autotune-no-cudagraphs",
            "optimizer_updates": 0,
        },
    }
    contract["files"] = _artifact_records(
        bundle,
        exclude={CONTRACT_NAME, METADATA_NAME},
    )
    _write_json(bundle / CONTRACT_NAME, contract)
    _write_json(bundle / METADATA_NAME, _dataset_metadata(dataset_reference))
    upload = staging / "upload"
    upload.mkdir()
    shutil.copy2(bundle / METADATA_NAME, upload / METADATA_NAME)
    with zipfile.ZipFile(
        upload / "bundle.zip",
        "w",
        compression=zipfile.ZIP_STORED,
    ) as archive:
        for path in sorted(bundle.rglob("*")):
            if path.is_file() and path.name != METADATA_NAME:
                archive.write(path, path.relative_to(bundle).as_posix())
    kernel = staging / "kernel"
    kernel.mkdir()
    _write_json(
        kernel / "kernel-metadata.json",
        _kernel_metadata(dataset_reference, actor),
    )
    template = (ROOT / TEMPLATE_PATH).read_text(encoding="utf-8")
    rendered = (
        Template(template)
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
        raise ValueError("Spec 0041 dataset actor differs")
    _validate_authorities()
    if (
        contract.get("schema_version") != "spec0041.label_blind_input.v1"
        or contract.get("scope") != "sealed_test_label_blind_inference_only"
        or contract.get("visibility") != "private"
        or contract.get("dataset_actor") != actor
        or contract.get("kernel_id") != f"{actor}/{KERNEL_SLUG}"
        or contract.get("spec_sha256") != _sha256(ROOT / SPEC_PATH)
        or contract.get("test_vector_sha256") != _sha256(ROOT / TEST_VECTOR_PATH)
        or contract.get("source_contract_sha256") != SOURCE_CONTRACT_SHA256
        or contract.get("kernel_sources") != list(KERNEL_SOURCES)
        or contract.get("producer_versions") != dict.fromkeys(KERNEL_SOURCES, 1)
    ):
        raise ValueError("Spec 0041 input contract identity differs")
    current_scorer_sha256 = _sha256(ROOT / SCORER_PATH)
    if contract.get("scorer_sha256") != current_scorer_sha256:
        amendment = _read_object(ROOT / NORMALIZATION_AMENDMENT_PATH)
        if (
            _sha256(ROOT / NORMALIZATION_AMENDMENT_PATH)
            != NORMALIZATION_AMENDMENT_SHA256
            or amendment.get("original_scorer_sha256") != contract.get("scorer_sha256")
            or amendment.get("amended_scorer_sha256") != current_scorer_sha256
        ):
            raise ValueError("Spec 0041 scorer normalization amendment differs")
    files = cast("dict[str, object]", contract.get("files"))
    if files != _artifact_records(bundle, exclude={CONTRACT_NAME, METADATA_NAME}):
        raise ValueError("Spec 0041 bundle bytes differ")
    _validate_tree(bundle, {*files, CONTRACT_NAME, METADATA_NAME})
    test = bundle / "test"
    _validate_test_stage(test)
    if contract.get("physical_sources") != _read_csv(test / "physical_parts.csv"):
        raise ValueError("Spec 0041 physical catalog differs")
    expected_graphs = _test_graph_identities(test)
    test_contract = cast("dict[str, object]", contract["test"])
    if test_contract.get("graph_identities") != expected_graphs:
        raise ValueError("Spec 0041 test graph identities differ")
    _validate_source_stage(bundle / "src")
    for branch, provenance in BRANCH_PROVENANCE.items():
        record = cast(
            "dict[str, object]",
            cast("dict[str, object]", contract["branches"])[branch],
        )
        state_path = bundle / str(record["path"])
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        if (
            record.get("source") != provenance
            or record.get("file_sha256") != _sha256(state_path)
            or record.get("state_sha256")
            != _state_dict_sha256(cast("dict[str, Tensor]", state))
            or record.get("parameter_count") != EXPECTED_PARAMETER_COUNT
        ):
            raise ValueError(f"Spec 0041 model-only state differs: {branch}")
    metadata = _dataset_metadata(dataset_reference)
    if _read_object(bundle / METADATA_NAME) != metadata:
        raise ValueError("Spec 0041 dataset metadata differs")
    _validate_upload(output / "upload", bundle, metadata)
    kernel = output / "kernel"
    if _read_object(kernel / "kernel-metadata.json") != _kernel_metadata(
        dataset_reference,
        actor,
    ):
        raise ValueError("Spec 0041 kernel metadata differs")
    run_path = kernel / "run.py"
    template = (ROOT / TEMPLATE_PATH).read_text(encoding="utf-8")
    expected_run = (
        Template(template)
        .substitute(
            input_contract_sha256=_sha256(bundle / CONTRACT_NAME),
            input_dataset_reference=dataset_reference,
        )
        .encode()
    )
    if run_path.read_bytes() != expected_run or run_path.stat().st_size >= 1_000_000:
        raise ValueError("Spec 0041 rendered kernel differs")
    compile(expected_run, str(TEMPLATE_PATH), "exec")
    return contract


def score(
    *,
    remote_output_root: Path,
    launch_receipt_path: Path,
    output_root: Path,
) -> dict[str, object]:
    contract = validate()
    return score_retrieved_mil_test_output(
        remote_output_root=remote_output_root,
        launch_receipt_path=launch_receipt_path,
        launch_claim_path=ROOT / OUTPUT_ROOT / "launch_claim.json",
        label_oracle_path=ROOT / "docs/data/ubc_ocean_eval_wsi_split.csv",
        output_root=output_root,
        expected_scorer_sha256=str(contract["scorer_sha256"]),
        expected_test_vector_sha256=str(contract["test_vector_sha256"]),
        expected_input_contract_sha256=_sha256(
            ROOT / OUTPUT_ROOT / "bundle" / CONTRACT_NAME,
        ),
        expected_dataset_reference=str(contract["dataset_reference"]),
        expected_spec_sha256=str(contract["spec_sha256"]),
        expected_branches=cast("dict[str, object]", contract["branches"]),
        expected_test=cast("dict[str, object]", contract["test"]),
        expected_kernel_sources=cast("list[str]", contract["kernel_sources"]),
        expected_kernel_sha256=_sha256(ROOT / OUTPUT_ROOT / "kernel/run.py"),
        expected_metadata_sha256=_sha256(
            ROOT / OUTPUT_ROOT / "kernel/kernel-metadata.json",
        ),
        expected_kernel_bytes=(ROOT / OUTPUT_ROOT / "kernel/run.py").stat().st_size,
        expected_metadata_bytes=(ROOT / OUTPUT_ROOT / "kernel/kernel-metadata.json")
        .stat()
        .st_size,
        normalization_amendment_path=ROOT / NORMALIZATION_AMENDMENT_PATH,
        expected_normalization_amendment_sha256=NORMALIZATION_AMENDMENT_SHA256,
    )


def _validate_authorities() -> None:
    fixed = {
        ROOT / SOURCE_CONTRACT: SOURCE_CONTRACT_SHA256,
        ROOT / INTEGRATION_AUDIT_PATH: INTEGRATION_AUDIT_SHA256,
        ROOT / TRAINING_LAUNCH_RECEIPT: TRAINING_LAUNCH_SHA256,
        ROOT / TRAINING_OUTPUT_RECEIPT: TRAINING_OUTPUT_SHA256,
        ROOT / CATALOG_PATH: TEST_FILE_HASHES["physical_parts.csv"],
    }
    fixed.update({
        ROOT / TEST_ROOT / name: digest
        for name, digest in TEST_FILE_HASHES.items()
        if name != "physical_parts.csv"
    })
    for path, expected in fixed.items():
        if _sha256(path) != expected:
            raise ValueError(f"Spec 0041 authority differs: {path}")
    receipt = _read_object(ROOT / TRAINING_OUTPUT_RECEIPT)
    declared = cast("dict[str, object]", receipt.get("files"))
    observed = {
        path.relative_to(ROOT / TRAINING_OUTPUT_ROOT).as_posix(): _file_record(path)
        for path in sorted((ROOT / TRAINING_OUTPUT_ROOT).rglob("*"))
        if path.is_file() and path != ROOT / TRAINING_OUTPUT_RECEIPT
    }
    if declared != observed:
        raise ValueError("Spec 0041 prior output receipt does not bind every byte")
    overall = _read_object(
        ROOT / TRAINING_OUTPUT_ROOT / "spec0036_mil_training/overall_status.json",
    )
    run_contract = _read_object(
        ROOT / TRAINING_OUTPUT_ROOT / "spec0036_mil_training/run_contract.json",
    )
    if (
        overall.get("status") != "complete"
        or run_contract.get("contract_hashes") != EXPECTED_TRAINING_HASHES
    ):
        raise ValueError("Spec 0041 prior training completion differs")
    _validate_source_authority()


def _validate_source_authority() -> None:
    contract = _read_object(ROOT / SOURCE_CONTRACT)
    files = cast("dict[str, dict[str, object]]", contract.get("files"))
    expected_names = {
        path.relative_to(ROOT / SOURCE_ROOT).as_posix()
        for path in (ROOT / SOURCE_ROOT / "src").rglob("*.py")
    }
    declared = {name for name in files if name.startswith("src/")}
    if (
        contract.get("schema_version") != "spec0036.mil_training_input.v1"
        or declared != expected_names
    ):
        raise ValueError("Spec 0041 frozen source authority differs")
    for name in declared:
        path = ROOT / SOURCE_ROOT / name
        record = files[name]
        if path.stat().st_size != record["bytes"] or _sha256(path) != record["sha256"]:
            raise ValueError(f"Spec 0041 frozen source bytes differ: {name}")


def _derive_model_state(branch: str, destination: Path) -> dict[str, object]:
    provenance = BRANCH_PROVENANCE[branch]
    checkpoints = (
        ROOT / TRAINING_OUTPUT_ROOT / "spec0036_mil_training" / branch / "checkpoints"
    )
    loaded = load_branch_checkpoint(
        checkpoints,
        slot="best",
        expected_branch_name=branch,
        expected_contract_hashes=EXPECTED_TRAINING_HASHES,
    )
    if (
        loaded.payload["committed_update"] != provenance["boundary"]
        or loaded.checkpoint_sha256 != provenance["checkpoint_sha256"]
        or loaded.manifest_sha256 != provenance["manifest_sha256"]
    ):
        raise ValueError(f"Spec 0041 selected checkpoint differs: {branch}")
    state = cast("dict[str, Tensor]", loaded.payload["model_state_dict"])
    model = LocalGlobalMILClassifier()
    model.load_state_dict(state)
    if (
        sum(parameter.numel() for parameter in model.parameters())
        != EXPECTED_PARAMETER_COUNT
    ):
        raise ValueError("Spec 0041 model parameter count differs")
    torch.save(state, destination)
    return {
        "path": destination.relative_to(destination.parents[1]).as_posix(),
        "file_sha256": _sha256(destination),
        "state_sha256": _state_dict_sha256(state),
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "source": provenance,
        "training_contract_hashes": EXPECTED_TRAINING_HASHES,
    }


def _test_graph_identities(test_root: Path) -> dict[str, str]:
    instances = _read_csv(test_root / "wsi_cancer_test_instances.csv")
    bags = _read_csv(test_root / "wsi_cancer_test_bags.csv")
    identities: dict[str, str] = {}
    for bag in bags:
        start = int(bag["instance_start"])
        count = int(bag["instance_count"])
        selected = instances[start : start + count]
        graph_rows = [
            SimpleNamespace(
                wsi_id=int(row["wsi_id"]),
                x=int(row["x"]),
                y=int(row["y"]),
            )
            for row in selected
        ]
        graph = build_local_attention_graph(
            cast("Sequence[WSIInstance]", graph_rows),
            expected_instance_count=count,
        )
        identities[bag["wsi_id"]] = graph.identity_sha256
    return identities


def _validate_test_stage(test: Path) -> None:
    if {path.name for path in test.iterdir()} != set(TEST_FILE_HASHES):
        raise ValueError("Spec 0041 test-only allow-list differs")
    for name, expected in TEST_FILE_HASHES.items():
        if _sha256(test / name) != expected:
            raise ValueError(f"Spec 0041 test file differs: {name}")
    instances = _read_csv(test / "wsi_cancer_test_instances.csv")
    bags = _read_csv(test / "wsi_cancer_test_bags.csv")
    forbidden = {
        "diagnosis",
        "diagnosis_label",
        "diagnosis_index",
        "truth",
        "target",
        "class",
    }
    if (
        len(instances) != TEST_INSTANCE_COUNT
        or len(bags) != TEST_WSI_COUNT
        or forbidden & set(instances[0])
        or forbidden & set(bags[0])
        or any(row["split"] != "test" for row in instances + bags)
        or sum(int(row["instance_count"]) for row in bags) != TEST_INSTANCE_COUNT
    ):
        raise ValueError("Spec 0041 label-free test rows differ")


def _copy_frozen_source(destination: Path) -> None:
    for source in sorted((ROOT / SOURCE_ROOT / "src").rglob("*.py")):
        target = destination / source.relative_to(ROOT / SOURCE_ROOT / "src")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _validate_source_stage(destination: Path) -> None:
    expected = ROOT / SOURCE_ROOT / "src"
    source_files = {
        path.relative_to(expected).as_posix(): path for path in expected.rglob("*.py")
    }
    staged = {
        path.relative_to(destination).as_posix(): path
        for path in destination.rglob("*.py")
    }
    if set(source_files) != set(staged) or any(
        staged[name].read_bytes() != source.read_bytes()
        for name, source in source_files.items()
    ):
        raise ValueError("Spec 0041 frozen source stage differs")


def _state_dict_sha256(state: dict[str, Tensor]) -> str:
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


def _dataset_metadata(reference: str) -> dict[str, object]:
    return {
        "id": reference,
        "title": "eqvae label blind MIL test inputs",
        "licenses": [{"name": "other"}],
    }


def _kernel_metadata(reference: str, actor: str) -> dict[str, object]:
    return {
        "id": f"{actor}/{KERNEL_SLUG}",
        "title": "eqvae label blind MIL test evaluation",
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
        raise ValueError("Spec 0041 upload metadata differs")
    with zipfile.ZipFile(upload / "bundle.zip") as archive:
        expected = {
            path.relative_to(bundle).as_posix(): path
            for path in bundle.rglob("*")
            if path.is_file() and path.name != METADATA_NAME
        }
        if set(archive.namelist()) != set(expected) or any(
            archive.read(name) != path.read_bytes() for name, path in expected.items()
        ):
            raise ValueError("Spec 0041 upload archive differs")


def _validate_tree(root: Path, expected_files: set[str]) -> None:
    observed = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    if observed != expected_files or any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("Spec 0041 bundle tree differs")


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
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--actor", required=True)
    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("--actor")
    score_parser = sub.add_parser("score")
    score_parser.add_argument("--remote-output-root", type=Path, required=True)
    score_parser.add_argument("--launch-receipt", type=Path, required=True)
    score_parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        print(json.dumps(build(actor=args.actor), indent=2, sort_keys=True))
    elif args.command == "validate":
        print(json.dumps(validate(expected_actor=args.actor), indent=2, sort_keys=True))
    else:
        print(
            json.dumps(
                score(
                    remote_output_root=args.remote_output_root,
                    launch_receipt_path=args.launch_receipt,
                    output_root=args.output_root,
                ),
                indent=2,
                sort_keys=True,
            ),
        )


if __name__ == "__main__":
    main()
