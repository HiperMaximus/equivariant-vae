# Copyright 2026 HiperMaximus
# ruff: noqa: DOC201, DOC501, EM101, EM102, PLR0916, PLR2004, PLW0717, TRY003, TRY300
"""Build the exact private dual-T4 kernel directory for Spec 0022."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import shutil
import textwrap
import zipfile
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING, Final, cast

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

KERNEL_ID: Final = "maximusshtefan/eqvae-ubc-ocean-cancer-topup"
INPUT_DATASET_ID: Final = "maximusshtefan/eqvae-ubc-ocean-cancer-topup-inputs"
CONFIG_SCHEMA: Final = "spec0022.cancer_topup_kernel_config.v1"
RECEIPT_SCHEMA: Final = "spec0022.input_dataset_receipt.v1"
TEMPLATE_PATH: Final = Path(
    "kaggle/kernels/ubc_ocean_cancer_topup/run_template.py",
)
SPEC_PATH: Final = Path("docs/specs/0022-additive-cancer-coverage-topup.md")
DEFAULT_PLAN_ROOT: Final = Path("runs/local/ubc_ocean_cancer_topup")
DEFAULT_RECEIPT: Final = DEFAULT_PLAN_ROOT / "input_dataset_receipt.json"
DEFAULT_OUTPUT: Final = Path("runs/local/ubc_ocean_cancer_topup_kernel")
UPLOAD_FILES: Final = frozenset({
    "kernel-metadata.json",
    "run.py",
    "spec0022_topup_kernel_config.json",
})


def build_kernel(
    *,
    repo_root: Path,
    plan_root: Path,
    receipt_path: Path,
    output_root: Path,
) -> dict[str, object]:
    """Build one hash-bound upload directory without contacting Kaggle."""
    contract_path = (
        plan_root / "inference_bundle/spec0022_topup_inference_contract.json"
    )
    plan_audit = plan_root / "spec0022_cancer_topup_plan_audit.json"
    logical_root = plan_root / "logical"
    if (
        not contract_path.is_file()
        or not plan_audit.is_file()
        or not logical_root.is_dir()
    ):
        raise FileNotFoundError(
            "Completed local Spec 0022 plan is required before kernel build",
        )
    contract = _read_object(contract_path)
    plan = _read_object(plan_audit)
    receipt = _read_object(receipt_path)
    if (
        contract.get("schema_version") != "spec0022.cancer_topup_inference_input.v1"
        or contract.get("status") != "complete"
    ):
        raise ValueError("Spec 0022 inference contract is not complete")
    inference_files = _artifact_records(plan_root / "inference_bundle")
    mounted_files = _mounted_input_records(inference_files)
    plan_files = plan.get("inference_bundle_files")
    _validate_receipt(
        receipt,
        receipt_path=receipt_path,
        contract_path=contract_path,
        input_files=mounted_files,
    )
    if plan_files != inference_files:
        raise ValueError("Spec 0022 plan input records changed")
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite {output_root}")
    output_root.mkdir(parents=True)
    try:
        config: dict[str, object] = {
            "schema_version": CONFIG_SCHEMA,
            "spec_sha256": _sha256(repo_root / SPEC_PATH),
            "input_contract_sha256": _sha256(contract_path),
            "input_dataset_receipt": receipt,
            "expected_binary_output_bytes": contract["expected_binary_output_bytes"],
            "saved_output_limit_bytes": contract["saved_output_limit_bytes"],
            "metadata_reserve_bytes": contract["metadata_reserve_bytes"],
            "deadline_projection": contract.get("deadline_projection"),
        }
        config_bytes = _canonical_json(config)
        (output_root / "spec0022_topup_kernel_config.json").write_bytes(config_bytes)
        metadata = {
            "id": KERNEL_ID,
            "title": "eqvae UBC-OCEAN cancer latent top-up",
            "code_file": "run.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_internet": "true",
            "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [INPUT_DATASET_ID],
            "competition_sources": ["UBC-OCEAN"],
            "kernel_sources": [],
            "model_sources": [],
        }
        (output_root / "kernel-metadata.json").write_text(
            f"{json.dumps(metadata, indent=2)}\n",
            encoding="utf-8",
        )
        (output_root / "run.py").write_bytes(
            _render_wrapper(repo_root, config_bytes),
        )
        validate_kernel(
            repo_root=repo_root,
            plan_root=plan_root,
            receipt_path=receipt_path,
            output_root=output_root,
        )
        return config
    except BaseException:
        shutil.rmtree(output_root, ignore_errors=True)
        raise


def validate_kernel(
    *,
    repo_root: Path,
    plan_root: Path,
    receipt_path: Path,
    output_root: Path,
) -> None:
    """Fail closed on stale config, metadata, source, or receipt bindings."""
    observed = {
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("*")
        if path.is_file()
    }
    if observed != set(UPLOAD_FILES):
        raise ValueError(f"Spec 0022 upload allow-list differs: {sorted(observed)!r}")
    config_path = output_root / "spec0022_topup_kernel_config.json"
    config = _read_object(config_path)
    receipt = _read_object(receipt_path)
    contract_path = (
        plan_root / "inference_bundle/spec0022_topup_inference_contract.json"
    )
    plan = _read_object(plan_root / "spec0022_cancer_topup_plan_audit.json")
    inference_files = _artifact_records(plan_root / "inference_bundle")
    mounted_files = _mounted_input_records(inference_files)
    _validate_receipt(
        receipt,
        receipt_path=receipt_path,
        contract_path=contract_path,
        input_files=mounted_files,
    )
    if plan.get("inference_bundle_files") != inference_files:
        raise ValueError("Spec 0022 plan input records changed")
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or config.get("spec_sha256") != _sha256(repo_root / SPEC_PATH)
        or config.get("input_contract_sha256") != _sha256(contract_path)
        or config.get("input_dataset_receipt") != receipt
        or config_path.read_bytes() != _canonical_json(config)
    ):
        raise ValueError("Spec 0022 kernel config is stale or noncanonical")
    expected_bytes = config.get("expected_binary_output_bytes")
    if (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes + 10_000_000 > 20_000_000_000
        or config.get("metadata_reserve_bytes") != 10_000_000
        or config.get("saved_output_limit_bytes") != 20_000_000_000
    ):
        raise ValueError("Spec 0022 kernel output projection is invalid")
    deadline_value = config.get("deadline_projection")
    deadline = (
        cast("dict[str, object]", deadline_value)
        if isinstance(deadline_value, dict)
        else None
    )
    if (
        not isinstance(deadline, dict)
        or deadline.get("method") != "max_observed_seconds_per_row_or_wsi_v1"
        or deadline.get("fits") is not True
        or deadline.get("validation_reserve_seconds") != 3_600
        or deadline.get("session_limit_seconds") != 28_800
        or isinstance(deadline.get("worst_observed_seconds_per_wsi"), bool)
        or not isinstance(deadline.get("worst_observed_seconds_per_wsi"), int)
        or cast("int", deadline["worst_observed_seconds_per_wsi"]) < 1
        or isinstance(deadline.get("projected_total_seconds"), bool)
        or not isinstance(deadline.get("projected_total_seconds"), int)
        or cast("int", deadline["projected_total_seconds"]) > 28_800
    ):
        raise ValueError("Spec 0022 kernel deadline projection is invalid")
    metadata = _read_object(output_root / "kernel-metadata.json")
    if metadata != {
        "id": KERNEL_ID,
        "title": "eqvae UBC-OCEAN cancer latent top-up",
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [INPUT_DATASET_ID],
        "competition_sources": ["UBC-OCEAN"],
        "kernel_sources": [],
        "model_sources": [],
    }:
        raise ValueError("Spec 0022 kernel metadata differs from the one-off contract")
    wrapper_path = output_root / "run.py"
    if wrapper_path.read_bytes() != _render_wrapper(
        repo_root,
        config_path.read_bytes(),
    ):
        raise ValueError("Spec 0022 wrapper differs from exact generated bytes")


def _render_wrapper(repo_root: Path, config_bytes: bytes) -> bytes:
    payload = _source_zip(repo_root)
    substitutions = {
        "embedded_payload_b64": "\n".join(
            textwrap.wrap(base64.b64encode(payload).decode("ascii"), 76),
        ),
        "embedded_payload_sha256": hashlib.sha256(payload).hexdigest(),
        "embedded_config_b64": base64.b64encode(config_bytes).decode("ascii"),
        "embedded_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
    }
    template = Template((repo_root / TEMPLATE_PATH).read_text(encoding="utf-8"))
    return template.substitute(substitutions).encode()


def _source_zip(repo_root: Path) -> bytes:
    buffer = io.BytesIO()
    source_root = repo_root / "src/eqvae"
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_root.rglob("*")):
            if (
                not path.is_file()
                or "__pycache__" in path.parts
                or path.suffix == ".pyc"
            ):
                continue
            relative = Path("src/eqvae") / path.relative_to(source_root)
            info = zipfile.ZipInfo(relative.as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    return buffer.getvalue()


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    return f"{json.dumps(payload, sort_keys=True, separators=(',', ':'))}\n".encode()


def _artifact_records(root: Path) -> dict[str, dict[str, object]]:
    return {
        path.relative_to(root).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _mounted_input_records(
    records: Mapping[str, Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    """Exclude Kaggle's upload-control metadata, which is never mounted as data."""
    return {
        name: record
        for name, record in records.items()
        if name != "dataset-metadata.json"
    }


def _validate_receipt(
    receipt: Mapping[str, object],
    *,
    receipt_path: Path,
    contract_path: Path,
    input_files: Mapping[str, Mapping[str, object]],
) -> None:
    expected_keys = {
        "schema_version",
        "status",
        "visibility",
        "dataset_reference",
        "dataset_version",
        "input_contract_sha256",
        "files",
        "remote_listing_sha256",
        "remote_status_sha256",
    }
    version = receipt.get("dataset_version")
    if (
        set(receipt) != expected_keys
        or receipt.get("schema_version") != RECEIPT_SCHEMA
        or receipt.get("status") != "verified"
        or receipt.get("visibility") != "private"
        or receipt.get("dataset_reference") != INPUT_DATASET_ID
        or isinstance(version, bool)
        or not isinstance(version, int)
        or version < 1
        or receipt.get("input_contract_sha256") != _sha256(contract_path)
        or receipt.get("files") != input_files
        or not _is_sha256(receipt.get("remote_listing_sha256"))
        or not _is_sha256(receipt.get("remote_status_sha256"))
        or receipt_path.read_bytes() != _canonical_json(receipt)
    ):
        raise ValueError("Spec 0022 input dataset receipt is missing or mismatched")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, object]:
    value = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain an object")
    return cast("dict[str, object]", value)


def main(argv: Sequence[str] | None = None) -> int:
    """Build or validate the local upload directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("build", "validate"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--plan-root", type=Path, default=DEFAULT_PLAN_ROOT)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    mode = cast("str", args.mode)
    repo_root = cast("Path", args.repo_root)
    plan_root = cast("Path", args.plan_root)
    receipt_path = cast("Path", args.receipt)
    output_root = cast("Path", args.output_root)
    if mode == "build":
        build_kernel(
            repo_root=repo_root,
            plan_root=plan_root,
            receipt_path=receipt_path,
            output_root=output_root,
        )
    else:
        validate_kernel(
            repo_root=repo_root,
            plan_root=plan_root,
            receipt_path=receipt_path,
            output_root=output_root,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
