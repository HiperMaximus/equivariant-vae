# Copyright 2026 HiperMaximus
# ruff: noqa: DOC201, DOC501, EM101, EM102, PLR0914, PLR0915, PLR2004, PLW0717, TRY003, TRY300, TRY301
"""Run the exact paired dual-T4 Spec 0022 cancer top-up."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

from eqvae.data.latent_shards import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_UNION_SHA256,
    LATENT_RECORD_BYTES,
    LATENT_SHARD_HEADER_SIZE,
    LatentArtifact,
    LatentShardWriter,
    load_work_manifest,
)
from eqvae.inference.checkpoints import load_frozen_checkpoint
from eqvae.inference.dual_writer import DualLatentWriter, WorkerResumeBinding
from eqvae.inference.pilot import require_exact_dual_t4
from eqvae.inference.worker import (
    DualEncoderWorker,
    EncoderRecipe,
    FrozenEncoderRunner,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

CONTRACT_SCHEMA: Final = "spec0022.cancer_topup_inference_input.v1"
PAIR_AUDIT_SCHEMA: Final = "spec0022.cancer_topup_pair_audit.v1"
OUTPUT_ALLOWLIST: Final = frozenset({
    "normal_vae_mu_cancer_topup.bin",
    "normal_vae_mu_cancer_topup.json",
    "so2_vae_mu_cancer_topup.bin",
    "so2_vae_mu_cancer_topup.json",
    "spec0022_cancer_topup_pair_audit.json",
})
INPUT_ALLOWLIST: Final = frozenset({
    "cancer_topup_manifest.csv",
    "normal_vae_step_060000.pt",
    "so2_vae_step_060000.pt",
    "spec0022_topup_inference_contract.json",
})
METADATA_RESERVE_BYTES: Final = 10_000_000
SESSION_LIMIT_SECONDS: Final = 28_800
VALIDATION_RESERVE_SECONDS: Final = 3_600


def run_topup(
    *,
    input_root: Path,
    wsi_dir: Path,
    output_root: Path,
    scratch_root: Path,
) -> dict[str, object]:
    """Encode one sealed label-free manifest and publish only an aligned pair."""
    if os.environ.get("EQVAE_CANCER_TOPUP_CONFIRMED") != "1":
        raise RuntimeError("Spec 0022 top-up execution confirmation is missing")
    contract_path = input_root / "spec0022_topup_inference_contract.json"
    contract = _validate_input_tree(input_root, contract_path)
    raw_manifest = _mapping(contract, "supplement_manifest")
    manifest_path = input_root / _string(raw_manifest, "path")
    manifest_hash = _string(raw_manifest, "sha256")
    manifest = load_work_manifest(
        manifest_path,
        run_number=1,
        expected_sha256=manifest_hash,
        require_run_basename=False,
    )
    if len(manifest.rows) != _integer(raw_manifest, "row_count") or not manifest.rows:
        raise ValueError(
            "Top-up manifest row count is empty or disagrees with contract",
        )
    expected_bytes = 2 * (
        LATENT_SHARD_HEADER_SIZE + len(manifest.rows) * LATENT_RECORD_BYTES
    )
    if (
        contract.get("expected_binary_output_bytes") != expected_bytes
        or contract.get("saved_output_limit_bytes") != 20_000_000_000
        or contract.get("metadata_reserve_bytes") != METADATA_RESERVE_BYTES
        or expected_bytes + METADATA_RESERVE_BYTES > 20_000_000_000
    ):
        raise ValueError("Top-up saved-output projection is invalid")
    worst_wsi_seconds = _validate_deadline_projection(contract)
    checkpoints = _mapping(contract, "checkpoints")
    normal_record = _mapping(checkpoints, "normal_vae")
    so2_record = _mapping(checkpoints, "so2_vae")
    normal_path = input_root / _string(normal_record, "path")
    so2_path = input_root / _string(so2_record, "path")
    if (
        normal_record.get("sha256") != EXPECTED_CHECKPOINT_SHA256["normal_vae"]
        or so2_record.get("sha256") != EXPECTED_CHECKPOINT_SHA256["so2_vae"]
    ):
        raise ValueError(
            "Top-up contract checkpoint hashes differ from the frozen pair",
        )
    _require_hash(normal_path, EXPECTED_CHECKPOINT_SHA256["normal_vae"])
    _require_hash(so2_path, EXPECTED_CHECKPOINT_SHA256["so2_vae"])
    if contract.get("base_union_sha256") != EXPECTED_UNION_SHA256:
        raise ValueError("Top-up contract does not bind the frozen base union")
    if output_root.exists() or scratch_root.exists():
        raise FileExistsError("Top-up output and scratch roots must start absent")
    scratch_root.mkdir(parents=True)
    config_hash = _sha256(contract_path)
    normal_device, so2_device = require_exact_dual_t4()
    recipe = EncoderRecipe(
        batch_size=8,
        numeric_mode="FP32",
        execution="eager",
        d2h_mode="synchronous",
    )
    session_start = float(
        os.environ.get("EQVAE_SESSION_START_MONOTONIC", str(time.monotonic())),
    )

    def may_start_wsi(_wsi_id: int, _start: int, _end: int) -> bool:
        return time.monotonic() - session_start + worst_wsi_seconds <= (
            SESSION_LIMIT_SECONDS - VALIDATION_RESERVE_SECONDS
        )

    normal_model = load_frozen_checkpoint(normal_path, model_name="normal_vae")
    so2_model = load_frozen_checkpoint(so2_path, model_name="so2_vae")
    normal_writer = _writer(
        scratch_root,
        manifest_path,
        manifest_hash,
        "normal_vae",
    )
    so2_writer = _writer(
        scratch_root,
        manifest_path,
        manifest_hash,
        "so2_vae",
    )
    dual = DualLatentWriter(
        normal_writer=normal_writer,
        so2_writer=so2_writer,
        state_path=scratch_root / "spec0022_topup_worker.resume.json",
        binding=WorkerResumeBinding(
            input_bundle_sha256=config_hash,
            run_config_sha256=config_hash,
            work_manifest_sha256=manifest_hash,
            run_number=1,
        ),
    )
    try:
        worker = DualEncoderWorker(
            normal=FrozenEncoderRunner(
                normal_model,
                device=normal_device,
                recipe=recipe,
            ),
            so2=FrozenEncoderRunner(
                so2_model,
                device=so2_device,
                recipe=recipe,
            ),
        )
        result = worker.run(
            manifest=manifest,
            wsi_dir=wsi_dir,
            writers=dual,
            incomplete_path=scratch_root / "spec0022_topup_incomplete.json",
            may_start_wsi=may_start_wsi,
        )
        if result.stopped_for_deadline or result.row_count != len(manifest.rows):
            raise RuntimeError("Top-up job did not finish the sealed manifest")
        pair = _finalize_pair(
            writers=dual,
            audit_path=scratch_root / "spec0022_cancer_topup_pair_audit.json",
            manifest_sha256=manifest_hash,
            config_sha256=config_hash,
        )
        dual.state_path.unlink(missing_ok=True)
        _publish_complete_output(scratch_root, output_root)
        return pair
    except BaseException:
        shutil.rmtree(output_root, ignore_errors=True)
        raise
    finally:
        normal_writer.close()
        so2_writer.close()
        shutil.rmtree(scratch_root, ignore_errors=True)


def _writer(
    root: Path,
    manifest_path: Path,
    manifest_sha256: str,
    model_name: str,
) -> LatentShardWriter:
    typed = "normal_vae" if model_name == "normal_vae" else "so2_vae"
    return LatentShardWriter(
        bin_path=root / f"{typed}_mu_cancer_topup.bin",
        manifest_path=manifest_path,
        run_number=1,
        model_name=typed,
        checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256[typed],
        expected_checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256[typed],
        expected_manifest_sha256=manifest_sha256,
        expected_union_sha256=EXPECTED_UNION_SHA256,
        coordinate_only_manifest=True,
    )


def _finalize_pair(
    *,
    writers: DualLatentWriter,
    audit_path: Path,
    manifest_sha256: str,
    config_sha256: str,
) -> dict[str, object]:
    normal = writers.normal_writer.finalize()
    so2 = writers.so2_writer.finalize()
    if normal.manifest.rows != so2.manifest.rows:
        raise ValueError("Top-up model manifests differ")
    payload: dict[str, object] = {
        "schema_version": PAIR_AUDIT_SCHEMA,
        "status": "complete",
        "row_count": len(normal.manifest.rows),
        "supplement_manifest_sha256": manifest_sha256,
        "input_contract_sha256": config_sha256,
        "base_union_sha256": EXPECTED_UNION_SHA256,
        "normal_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256["normal_vae"],
        "so2_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256["so2_vae"],
        "completed_wsi_evidence": [
            asdict(evidence) for evidence in writers.completed_wsi_evidence
        ],
        "artifacts": {
            "normal_vae": _artifact(normal),
            "so2_vae": _artifact(so2),
        },
    }
    _atomic_write_json(audit_path, payload)
    return payload


def _artifact(artifact: LatentArtifact) -> dict[str, object]:
    if artifact.file_sha256 is None:
        raise ValueError("Top-up artifact lacks a full-file SHA-256")
    return {
        "bin_name": artifact.bin_path.name,
        "bin_bytes": artifact.bin_path.stat().st_size,
        "bin_sha256": artifact.file_sha256,
        "sidecar_name": artifact.sidecar_path.name,
        "sidecar_bytes": artifact.sidecar_path.stat().st_size,
        "sidecar_sha256": _sha256(artifact.sidecar_path),
    }


def _validate_input_tree(
    root: Path,
    contract_path: Path,
) -> dict[str, object]:
    observed = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    if observed != set(INPUT_ALLOWLIST):
        raise ValueError(f"Top-up input allow-list differs: {sorted(observed)!r}")
    contract = _read_object(contract_path)
    if (
        contract.get("schema_version") != CONTRACT_SCHEMA
        or contract.get("status") != "complete"
    ):
        raise ValueError("Top-up input contract is not complete")
    manifest = root / "cancer_topup_manifest.csv"
    with manifest.open(encoding="utf-8", newline="") as handle:
        header = handle.readline().rstrip("\r\n")
    if header != "atlas_row_index,wsi_id,x,y":
        raise ValueError("Mounted top-up manifest is not strictly label-free")
    forbidden = ("diagnosis", "split", "tissue", "mask")
    for relative in observed:
        if any(token in relative.lower() for token in forbidden):
            raise ValueError(f"Forbidden semantic filename in top-up input: {relative}")
    return contract


def _validate_deadline_projection(contract: Mapping[str, object]) -> int:
    projection = _mapping(contract, "deadline_projection")
    if (
        projection.get("method") != "max_observed_seconds_per_row_or_wsi_v1"
        or projection.get("fits") is not True
        or projection.get("validation_reserve_seconds") != VALIDATION_RESERVE_SECONDS
        or projection.get("session_limit_seconds") != SESSION_LIMIT_SECONDS
    ):
        raise ValueError(
            "Top-up deadline projection is missing or not the sealed method",
        )
    projected = _integer(projection, "projected_total_seconds")
    worst_wsi_seconds = _integer(projection, "worst_observed_seconds_per_wsi")
    if projected > SESSION_LIMIT_SECONDS:
        raise ValueError("Top-up deadline projection exceeds one Kaggle session")
    if worst_wsi_seconds < 1:
        raise ValueError("Top-up deadline projection lacks a WSI safety bound")
    return worst_wsi_seconds


def _publish_complete_output(scratch_root: Path, output_root: Path) -> None:
    """Atomically expose the exact complete pair; never move audit separately."""
    observed = {path.name for path in scratch_root.iterdir() if path.is_file()}
    if observed != set(OUTPUT_ALLOWLIST) or any(
        not path.is_file() for path in scratch_root.iterdir()
    ):
        missing = sorted(set(OUTPUT_ALLOWLIST) - observed)
        if missing:
            raise ValueError(f"Top-up completion is missing {missing!r}")
        raise ValueError("Top-up scratch output allow-list differs")
    _fsync_directory(scratch_root)
    scratch_root.replace(output_root)
    _fsync_directory(output_root.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    encoded = f"{json.dumps(payload, indent=2, sort_keys=True)}\n".encode()
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_object(path: Path) -> dict[str, object]:
    value = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return cast("dict[str, object]", value)


def _mapping(value: Mapping[str, object], key: str) -> Mapping[str, object]:
    item = value.get(key)
    if not isinstance(item, dict):
        raise TypeError(f"{key} must be an object")
    return cast("Mapping[str, object]", item)


def _string(value: Mapping[str, object], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str):
        raise TypeError(f"{key} must be a string")
    return item


def _integer(value: Mapping[str, object], key: str) -> int:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, int):
        raise TypeError(f"{key} must be an integer")
    return item


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hash(path: Path, expected: str) -> None:
    observed = _sha256(path)
    if observed != expected:
        raise ValueError(f"SHA-256 mismatch for {path.name}: {observed}")


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the exact one-off worker paths and run the top-up."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--wsi-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    args = parser.parse_args(argv)
    run_topup(
        input_root=cast("Path", args.input_root),
        wsi_dir=cast("Path", args.wsi_dir),
        output_root=cast("Path", args.output_root),
        scratch_root=cast("Path", args.scratch_root),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
