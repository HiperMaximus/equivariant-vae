# Copyright 2026 HiperMaximus
# ruff: noqa: DOC201, DOC501, EM101, PLR2004, T201, TRY003
"""Build the exact Spec 0030 full-model WSI45630 capacity kernel."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import textwrap
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING, Final, cast

from eqvae.models.local_global_mil import EXPECTED_PARAMETER_COUNT

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

ROOT: Final = Path.cwd()
DEFAULT_ROOT: Final = Path(
    "kaggle/kernels/wsi45630_local_global_capacity/package",
)
SPEC_PATH: Final = Path("docs/specs/0030-local-global-mil-capacity-probe.md")
TEMPLATE_PATH: Final = Path(
    "kaggle/kernels/wsi45630_local_global_capacity/run_template.py",
)
MODEL_PATH: Final = Path("src/eqvae/models/local_global_mil.py")
INPUT_ROOT: Final = Path("runs/local/wsi45630_capacity")
INPUT_RECEIPT_PATH: Final = INPUT_ROOT / "input_receipt.json"
INPUT_CONTRACT_PATH: Final = INPUT_ROOT / "bundle/wsi45630_capacity_input.json"
INPUT_DATASET_REFERENCE: Final = "maximusshtefan/eqvae-wsi45630-capacity-inputs"
INPUT_DATASET_VERSION: Final = 1
INPUT_CONTRACT_SHA256: Final = (
    "99bb4d2f60558aee9691b67be4867ffae434bc306581a000fd5d72a6befac660"
)
POINTER_SHA256: Final = (
    "08e461846bf16efebac707c82962762f49837916986b29aee0dcd6ca1fc31c6c"
)
KERNEL_ID: Final = "maximusshtefan/eqvae-wsi45630-local-global-mil-capacity"
KERNEL_SOURCES: Final = (
    "maximusshtefan/eqvae-ubc-ocean-latent-run-04",
    "maximusshtefan/eqvae-ubc-ocean-cancer-latent-top-up",
    "maximusshtefan/eqvae-wsi45630-completion",
)
SOURCE_VERSIONS: Final = dict.fromkeys(KERNEL_SOURCES, 1)
PACKAGE_FILES: Final = {
    "kernel-metadata.json",
    "run.py",
}
WSI_ID: Final = 45_630
PATCH_COUNT: Final = 32_595
DIAGNOSIS_INDEX: Final = 1


def build() -> dict[str, object]:
    """Create a new immutable local launch package."""
    output = ROOT / DEFAULT_ROOT
    if output.exists():
        message = f"Refusing to overwrite {output}"
        raise FileExistsError(message)
    _validate_input_authority(ROOT)
    staging = output.with_name(f".{output.name}.building")
    if staging.exists():
        message = f"Stale staging directory exists: {staging}"
        raise FileExistsError(message)
    kernel = staging / "kernel"
    try:
        _populate_kernel(kernel)
        staging.replace(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return validate()


def _populate_kernel(kernel: Path) -> None:
    kernel.mkdir(parents=True)
    model_source = ROOT / MODEL_PATH
    contract = _contract(ROOT)
    _write_json(kernel / "kernel-metadata.json", _kernel_metadata())
    rendered = _render(
        ROOT,
        contract=contract,
        model_sha256=_sha256(model_source),
    )
    (kernel / "run.py").write_bytes(rendered)


def validate() -> dict[str, object]:
    """Reject input, source, metadata, contract, or rendered-code drift."""
    _validate_input_authority(ROOT)
    kernel = ROOT / DEFAULT_ROOT / "kernel"
    entries = list(kernel.iterdir())
    observed = {path.name for path in entries}
    if observed != PACKAGE_FILES or any(
        not path.is_file() or path.is_symlink() for path in entries
    ):
        raise ValueError("Spec 0030 kernel package allow-list differs")
    contract = _contract(ROOT)
    if _read_object(kernel / "kernel-metadata.json") != _kernel_metadata():
        raise ValueError("Spec 0030 kernel metadata differs")
    model_source = ROOT / MODEL_PATH
    expected_run = _render(
        ROOT,
        contract=contract,
        model_sha256=_sha256(model_source),
    )
    run_path = kernel / "run.py"
    if run_path.read_bytes() != expected_run:
        raise ValueError("Rendered Spec 0030 launcher differs")
    if run_path.stat().st_size >= 1_000_000:
        raise ValueError("Spec 0030 launcher exceeds Kaggle's 1 MB limit")
    return contract


def _validate_input_authority(root: Path) -> None:
    receipt = _read_object(root / INPUT_RECEIPT_PATH)
    input_contract = root / INPUT_CONTRACT_PATH
    payload = _read_object(input_contract)
    if (
        receipt.get("status") != "verified"
        or receipt.get("visibility") != "private"
        or receipt.get("dataset_reference") != INPUT_DATASET_REFERENCE
        or receipt.get("dataset_version") != INPUT_DATASET_VERSION
        or receipt.get("input_contract_sha256") != INPUT_CONTRACT_SHA256
    ):
        raise ValueError("WSI45630 private input receipt differs")
    files = receipt.get("files")
    contract_files = payload.get("files")
    if not isinstance(files, dict) or not isinstance(contract_files, dict):
        raise TypeError("WSI45630 input receipt files must be an object")
    if files != contract_files:
        raise ValueError("WSI45630 input receipt file provenance differs")
    pointer = cast("dict[str, object]", files).get("probe/pointers.csv")
    if not isinstance(pointer, dict):
        raise TypeError("WSI45630 pointer receipt must be an object")
    pointer_record = cast("dict[str, object]", pointer)
    if pointer_record.get("sha256") != POINTER_SHA256:
        raise ValueError("WSI45630 pointer receipt differs")
    if _sha256(input_contract) != INPUT_CONTRACT_SHA256:
        raise ValueError("WSI45630 input contract hash differs")
    if (
        payload.get("dataset_reference") != INPUT_DATASET_REFERENCE
        or payload.get("wsi_id") != WSI_ID
        or payload.get("patch_count") != PATCH_COUNT
        or payload.get("diagnosis_index") != DIAGNOSIS_INDEX
        or payload.get("kernel_sources") != list(KERNEL_SOURCES)
    ):
        raise ValueError("WSI45630 input contract scope differs")


def _contract(root: Path) -> dict[str, object]:
    model_sha256 = _sha256(root / MODEL_PATH)
    return {
        "schema_version": "spec0030.local_global_mil_capacity.v1",
        "authorization": (
            "spec0030_local_global_capacity_shared_access_retry_authorized"
        ),
        "scope": "capacity_only_not_learning_or_evaluation",
        "spec_sha256": _sha256(root / SPEC_PATH),
        "model": {
            "source": MODEL_PATH.as_posix(),
            "sha256": model_sha256,
            "parameter_count": EXPECTED_PARAMETER_COUNT,
        },
        "input_dataset": {
            "reference": INPUT_DATASET_REFERENCE,
            "version": INPUT_DATASET_VERSION,
            "contract_sha256": INPUT_CONTRACT_SHA256,
            "pointer_sha256": POINTER_SHA256,
        },
        "kernel_sources": [
            {"reference": source, "version": SOURCE_VERSIONS[source]}
            for source in KERNEL_SOURCES
        ],
        "wsi_id": WSI_ID,
        "patch_count": PATCH_COUNT,
        "diagnosis_index": DIAGNOSIS_INDEX,
        "initialization_seed": 1701,
        "steps": ["warmup", "measured"],
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 2e-4,
            "matrix_weight_decay": 1e-4,
            "semantic_no_decay": True,
        },
        "precision": {
            "autocast": "float16",
            "classifier_and_loss": "float32",
            "grad_scaler_initial_scale": 32768,
            "grad_scaler_growth_interval": 1_000_000,
        },
        "execution": {
            "model_devices": {"normal_vae": 0, "so2_vae": 1},
            "direct_complete_bag_only": True,
            "checkpointing": False,
            "fallback": None,
            "paired_step_atomicity": "both_finite_before_either_step",
        },
        "output": "spec0030_local_global_mil_capacity.json",
    }


def _kernel_metadata() -> dict[str, object]:
    return {
        "id": KERNEL_ID,
        "title": "eqvae WSI45630 local global MIL capacity",
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [INPUT_DATASET_REFERENCE],
        "competition_sources": [],
        "kernel_sources": list(KERNEL_SOURCES),
        "model_sources": [],
    }


def _render(
    root: Path,
    *,
    contract: Mapping[str, object],
    model_sha256: str,
) -> bytes:
    template = (root / TEMPLATE_PATH).read_text(encoding="utf-8")
    contract_json = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    embedded_model = _embedded_model_source(root / MODEL_PATH)
    placeholders = {
        "capacity_contract_json": contract_json,
        "capacity_contract_sha256": hashlib.sha256(contract_json.encode()).hexdigest(),
        "embedded_model_sha256": hashlib.sha256(embedded_model.encode()).hexdigest(),
        "model_sha256": model_sha256,
    }
    for name in placeholders:
        if template.count(f"${name}") != 1:
            message = f"Spec 0030 template must contain one ${name}"
            raise ValueError(message)
    sentinel = "# __EMBEDDED_SPEC0026_MODEL__\n"
    if template.count(sentinel) != 1:
        raise ValueError("Spec 0030 template model sentinel differs")
    embedded_block = "if False:\n" + textwrap.indent(embedded_model, "    ")
    rendered_text = Template(template).substitute(placeholders)
    rendered = rendered_text.replace(sentinel, embedded_block, 1).encode()
    compile(rendered, str(root / TEMPLATE_PATH), "exec")
    return rendered


def _embedded_model_source(path: Path) -> str:
    source = path.read_text(encoding="utf-8")
    future = "from __future__ import annotations\n"
    if source.count(future) != 1:
        raise ValueError("Canonical MIL model future import differs")
    embedded = source.replace(future, "", 1)
    compile(embedded, str(path), "exec")
    return embedded


def _read_object(path: Path) -> dict[str, object]:
    payload = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(payload, dict):
        message = f"Expected a JSON object: {path}"
        raise TypeError(message)
    return cast("dict[str, object]", payload)


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    """Build or validate the immutable Spec 0030 kernel package."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("build", "validate"),
        nargs="?",
        default="build",
    )
    args = parser.parse_args(argv)
    action = cast("str", args.action)
    result = validate() if action == "validate" else build()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
