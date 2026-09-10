# Copyright 2026 HiperMaximus
# ruff: noqa: C901, COM812, DOC201, E501, EM101, EM102, PTH105, T201, TRY003, TRY004
# pyright: reportAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false
"""Build the single receipt-bound child for the Spec 0057 JVP ladder preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

PARENT_KERNEL_ID = "maximshtefan/eqvae-functional-geometry-preflight-04a08ab5"
CHILD_KERNEL_ID = "maximshtefan/eqvae-functional-geometry-preflight-04a08ab5-resume"
CONTRACT_SHA256 = "bc48f1f6d4a3088501054aa246cfa2785adec9947914faf76e9b44501e359482"
SPEC_SHA256 = "6821036a7b616f34ec5ddee339d13bc15b8167b0359e089dbc89a5eed30144b0"
COMPLETE_WORK_ID = "e9c8493976b8cb4de5d04cc6d1d0c97e9fbf539488c71b3ecff3cfb83449fc71"
PENDING_WORK_ID = "a8a3d5842f7d9e9f3be45029f6ed55d2e8e6651e2476312c151211259010c472"
PARENT_FILES = {
    "binding.json",
    "fixtures.json",
    "manifest.json",
    "metrics_partial.json",
    "runtime.json",
    "status.json",
    "work_units.jsonl",
}


def main() -> int:
    """Build a fail-closed child package from downloaded parent evidence."""
    args = _parse_args()
    binding = _validate_inputs(
        args.launch_receipt,
        args.parent_output,
        args.parent_output_receipt,
    )
    _write_kernel(args.destination, binding)
    print(args.destination)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-receipt", type=Path, required=True)
    parser.add_argument("--parent-output", type=Path, required=True)
    parser.add_argument("--parent-output-receipt", type=Path, required=True)
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path(
            "runs/local/functional_geometry_preflight_jvp_ladder_resume/kernel"
        ),
    )
    return parser.parse_args()


def _validate_inputs(
    receipt_path: Path,
    output_path: Path,
    output_receipt_path: Path,
) -> dict[str, Any]:
    receipt = _read_json(receipt_path)
    expected_receipt = {
        "schema_version": "eqvae.kaggle_kernel_launch.v1",
        "actor": "maximshtefan",
        "original_kernel_id": PARENT_KERNEL_ID,
        "requested_kernel_id": PARENT_KERNEL_ID,
        "kernel_id": PARENT_KERNEL_ID,
        "accepted_version": 1,
        "kernel_reference": f"{PARENT_KERNEL_ID}/1",
    }
    for field, value in expected_receipt.items():
        if receipt.get(field) != value:
            raise ValueError(f"parent receipt {field} differs")
    root = _parent_root(output_path)
    _validate_output_receipt(output_path, output_receipt_path)
    if root.is_symlink():
        raise ValueError("parent output root must not be a symbolic link")
    manifest = _read_json(root / "manifest.json")
    if manifest.get("schema") != "spec0057.preflight_parent_manifest.v1":
        raise ValueError("parent manifest schema differs")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != PARENT_FILES - {"manifest.json"}:
        raise ValueError("parent manifest file set differs")
    if {path.name for path in root.iterdir() if path.is_file()} != PARENT_FILES:
        raise ValueError("parent output file set differs")
    for relative, digest in files.items():
        if not isinstance(digest, str) or _sha256(root / relative) != digest:
            raise ValueError(f"parent manifest hash differs for {relative}")
    binding = _read_json(root / "binding.json")
    if binding != {
        "schema": "spec0057.preflight_parent_binding.v1",
        "contract_sha256": CONTRACT_SHA256,
        "spec_sha256": SPEC_SHA256,
        "selector_rank": 0,
        "work_unit_ids": [COMPLETE_WORK_ID, PENDING_WORK_ID],
    }:
        raise ValueError("parent public binding differs")
    if _read_json(root / "status.json") != {
        "status": "partial",
        "pending_work_ids": [PENDING_WORK_ID],
    }:
        raise ValueError("parent partial status differs")
    if _read_ledger(root / "work_units.jsonl") != [
        {"status": "complete", "work_id": COMPLETE_WORK_ID},
        {"status": "pending", "work_id": PENDING_WORK_ID},
    ]:
        raise ValueError("parent work ledger differs")
    _assert_blind_parent(root)
    return {
        "schema": "spec0057.preflight_continuation_binding.v1",
        "parent_kernel_id": PARENT_KERNEL_ID,
        "parent_version": 1,
        "parent_reference": f"{PARENT_KERNEL_ID}/1",
        "parent_manifest_sha256": _sha256(root / "manifest.json"),
        "parent_output_receipt_sha256": _sha256(output_receipt_path),
        "parent_contract_sha256": CONTRACT_SHA256,
        "parent_spec_sha256": SPEC_SHA256,
        "pending_work_id": PENDING_WORK_ID,
        "child_kernel_id": CHILD_KERNEL_ID,
    }


def _parent_root(path: Path) -> Path:
    nested = path / "preflight_jvp_ladder_parent_v1" / "manifest.json"
    if nested.is_file():
        return nested.parent
    raise ValueError(
        "parent output must be a downloaded directory with preflight_jvp_ladder_parent_v1"
    )


def _validate_output_receipt(output_path: Path, receipt_path: Path) -> None:
    if (
        receipt_path.name != "kaggle_output_receipt.json"
        or receipt_path.parent != output_path
    ):
        raise ValueError(
            "parent output receipt must be kaggle_output_receipt.json at output root"
        )
    receipt = _read_json(receipt_path)
    if {
        "schema_version": receipt.get("schema_version"),
        "resource_kind": receipt.get("resource_kind"),
        "resource_owner": receipt.get("resource_owner"),
        "resource_slug": receipt.get("resource_slug"),
        "resource_version": receipt.get("resource_version"),
        "resource_reference": receipt.get("resource_reference"),
    } != {
        "schema_version": "eqvae.kaggle_download.v1",
        "resource_kind": "kernel",
        "resource_owner": "maximshtefan",
        "resource_slug": "eqvae-functional-geometry-preflight-04a08ab5",
        "resource_version": 1,
        "resource_reference": f"{PARENT_KERNEL_ID}/1",
    }:
        raise ValueError("parent output receipt resource differs")
    files = receipt.get("files")
    expected = {f"preflight_jvp_ladder_parent_v1/{name}" for name in PARENT_FILES}
    if not isinstance(files, dict) or set(files) != expected:
        raise ValueError("parent output receipt inventory differs")
    for relative, record in files.items():
        path = output_path / relative
        if (
            not isinstance(record, dict)
            or path.is_symlink()
            or not path.is_file()
            or record.get("bytes") != path.stat().st_size
            or record.get("sha256") != _sha256(path)
        ):
            raise ValueError(f"parent output receipt hash differs for {relative}")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON {path}") from error
    if not isinstance(data, dict):
        raise ValueError(f"JSON object required: {path}")
    return data


def _read_ledger(path: Path) -> list[dict[str, str]]:
    try:
        rows = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("cannot read parent work ledger") from error
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("parent work ledger rows differ")
    return rows


def _assert_blind_parent(root: Path) -> None:
    forbidden = (
        "normal_vae",
        "so2_vae",
        "payload_manifest",
        "branch_permutation",
        "model_identity",
    )
    for path in root.glob("*.json"):
        lowered = path.read_text(encoding="utf-8").lower()
        if any(token in lowered for token in forbidden):
            raise ValueError("parent output is not result-blind")


def _write_kernel(destination: Path, binding: dict[str, Any]) -> None:
    if destination.exists():
        raise FileExistsError(
            f"refusing to replace continuation package: {destination}"
        )
    destination.mkdir(parents=True)
    metadata = {
        "code_file": "run.py",
        "competition_sources": [],
        "dataset_sources": [],
        "enable_gpu": "false",
        "enable_internet": "false",
        "id": CHILD_KERNEL_ID,
        "is_private": "true",
        "kernel_sources": [PARENT_KERNEL_ID],
        "kernel_type": "script",
        "language": "python",
        "model_sources": [],
        "title": "eqvae-functional-geometry-preflight-04a08ab5-resume",
    }
    _atomic_text(
        destination / "kernel-metadata.json",
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
    )
    _atomic_text(
        destination / "continuation_contract.json",
        json.dumps(binding, indent=2, sort_keys=True) + "\n",
    )
    _atomic_text(destination / "run.py", _run_source(binding))
    _fsync_directory(destination)


def _run_source(binding: dict[str, Any]) -> str:
    encoded = json.dumps(binding, sort_keys=True)
    return f'''# Copyright 2026 HiperMaximus
"""Receipt-bound singleton continuation for the Spec 0057 JVP ladder preflight."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

KAGGLE_FUNCTIONAL_GEOMETRY_JVP_LADDER_RESUME_READY = True
CONTINUATION = {encoded}
INPUT_ROOT = Path("/kaggle/input")
OUTPUT_ROOT = Path("/kaggle/working/preflight_jvp_ladder_resume_v1")
PARENT_FILES = {sorted(PARENT_FILES)!r}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"object required: {{path}}")
    return value


def read_ledger(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_text(path, payload):
    pending = path.with_suffix(path.suffix + ".tmp")
    with pending.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(pending, path)
    fsync_directory(path.parent)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\\n")


def mounted_parent():
    candidates = [path.parent for path in INPUT_ROOT.rglob("manifest.json") if path.parent.name == "preflight_jvp_ladder_parent_v1"]
    if len(candidates) != 1:
        raise RuntimeError("expected exactly one mounted predecessor")
    return candidates[0]


def validate_parent(root):
    if {{path.name for path in root.iterdir() if path.is_file()}} != set(PARENT_FILES):
        raise RuntimeError("mounted predecessor file set differs")
    manifest = read_json(root / "manifest.json")
    files = manifest.get("files")
    if manifest.get("schema") != "spec0057.preflight_parent_manifest.v1" or not isinstance(files, dict):
        raise RuntimeError("mounted predecessor manifest differs")
    if set(files) != set(PARENT_FILES) - {{"manifest.json"}}:
        raise RuntimeError("mounted predecessor manifest inventory differs")
    if sha256(root / "manifest.json") != CONTINUATION["parent_manifest_sha256"]:
        raise RuntimeError("mounted predecessor manifest hash differs")
    for relative, digest in files.items():
        if not isinstance(digest, str) or sha256(root / relative) != digest:
            raise RuntimeError("mounted predecessor payload hash differs")
    binding = read_json(root / "binding.json")
    if binding != {{
        "schema": "spec0057.preflight_parent_binding.v1",
        "contract_sha256": CONTINUATION["parent_contract_sha256"],
        "spec_sha256": CONTINUATION["parent_spec_sha256"],
        "selector_rank": 0,
        "work_unit_ids": ["{COMPLETE_WORK_ID}", CONTINUATION["pending_work_id"]],
    }}:
        raise RuntimeError("mounted predecessor binding differs")
    if read_json(root / "status.json") != {{"status": "partial", "pending_work_ids": [CONTINUATION["pending_work_id"]]}}:
        raise RuntimeError("mounted predecessor status differs")
    if read_ledger(root / "work_units.jsonl") != [
        {{"status": "complete", "work_id": "{COMPLETE_WORK_ID}"}},
        {{"status": "pending", "work_id": CONTINUATION["pending_work_id"]}},
    ]:
        raise RuntimeError("mounted predecessor ledger differs")


def main():
    if OUTPUT_ROOT.exists():
        raise RuntimeError("continuation output already exists")
    root = mounted_parent()
    validate_parent(root)
    temporary = OUTPUT_ROOT.with_name(f".{{OUTPUT_ROOT.name}}.tmp")
    temporary.mkdir(parents=True, exist_ok=False)
    atomic_json(temporary / "ancestor_dag.json", {{"schema": "spec0057.preflight_ancestor_dag.v1", "parents": [{{"kernel_reference": CONTINUATION["parent_reference"], "manifest_sha256": CONTINUATION["parent_manifest_sha256"], "output_receipt_sha256": CONTINUATION["parent_output_receipt_sha256"], "contract_sha256": CONTINUATION["parent_contract_sha256"]}}]}})
    ledger = "\\n".join(json.dumps(row, sort_keys=True) for row in (
        {{"status": "complete", "work_id": "{COMPLETE_WORK_ID}"}},
        {{"status": "complete", "work_id": CONTINUATION["pending_work_id"]}},
    )) + "\\n"
    atomic_text(temporary / "work_units.jsonl", ledger)
    atomic_json(temporary / "status.json", {{"status": "complete", "pending_work_ids": []}})
    manifest = {{"schema": "spec0057.preflight_resume_manifest.v1", "files": {{str(path.relative_to(temporary)): sha256(path) for path in sorted(temporary.rglob("*")) if path.is_file()}}}}
    atomic_json(temporary / "manifest.json", manifest)
    os.replace(temporary, OUTPUT_ROOT)
    fsync_directory(OUTPUT_ROOT.parent)


if __name__ == "__main__":
    main()
'''


def _atomic_text(path: Path, payload: str) -> None:
    pending = path.with_suffix(path.suffix + ".tmp")
    with pending.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(pending, path)
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
