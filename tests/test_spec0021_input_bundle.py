# Copyright 2026 HiperMaximus
"""Focused immutable-bundle tests for the locked Spec 0021 contract."""

from __future__ import annotations

import csv
import hashlib
import json
import zipfile
import zlib
from typing import TYPE_CHECKING, cast

import pytest

from eqvae.data import latent_shards
from eqvae.data.latent_shards import (
    EXPECTED_TASK_MANIFEST_SHA256,
    UNION_HEADER,
    make_latent_shard_header,
)
from eqvae.inference import input_bundle

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

_SHA_A = "a" * 64
_SHA_B = "b" * 64
_FRESH_PAYLOAD_FILES = 15


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_work_manifest(path: Path, *, atlas_index: int, wsi_id: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=UNION_HEADER)
        writer.writeheader()
        writer.writerow(
            {
                "atlas_row_index": atlas_index,
                "wsi_id": wsi_id,
                "diagnosis_label": "CC",
                "diagnosis_index": 0,
                "x": 0,
                "y": 0,
                "split": "train",
                "cancer_ae_selected": "true",
                "tissue_selected": "true",
                "tissue_label": "tumor",
            },
        )


def _fresh_sources(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[int, Path], Path, dict[str, Path], Path, Path]:
    work: dict[int, Path] = {}
    for run in range(1, 6):
        path = root / "work_shards" / f"run_{run:02d}_of_05.csv"
        _write_work_manifest(path, atlas_index=run, wsi_id=100 + run)
        work[run] = path
    union = root / "union_patch_manifest.csv"
    _write_work_manifest(union, atlas_index=1, wsi_id=101)
    tasks: dict[str, Path] = {}
    for task_split in EXPECTED_TASK_MANIFEST_SHA256:
        path = root / "task_views" / f"{task_split}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture,{task_split}\n", encoding="utf-8")
        tasks[task_split] = path
    normal = root / "normal.pt"
    normal.write_bytes(b"normal checkpoint")
    so2 = root / "so2.pt"
    so2.write_bytes(b"so2 checkpoint")
    monkeypatch.setattr(
        input_bundle,
        "EXPECTED_WORK_MANIFEST_SHA256",
        {run: _sha256(path) for run, path in work.items()},
    )
    monkeypatch.setattr(input_bundle, "EXPECTED_UNION_SHA256", _sha256(union))
    monkeypatch.setattr(
        input_bundle,
        "EXPECTED_TASK_MANIFEST_SHA256",
        {name: _sha256(path) for name, path in tasks.items()},
    )
    monkeypatch.setattr(
        input_bundle,
        "EXPECTED_CHECKPOINT_SHA256",
        {"normal_vae": _sha256(normal), "so2_vae": _sha256(so2)},
    )
    return work, union, tasks, normal, so2


def _stage_fresh(root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    work, union, tasks, normal, so2 = _fresh_sources(root / "sources", monkeypatch)
    destination = root / "bundle"
    input_bundle.stage_fresh_input_bundle(
        destination,
        work_manifests=work,
        union_manifest=union,
        task_manifests=tasks,
        normal_checkpoint=normal,
        so2_checkpoint=so2,
        dataset_slug="owner/spec0021-inputs",
    )
    return destination


def test_fresh_bundle_is_exact_canonical_and_contract_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fresh contract is canonical, complete, and published last."""
    destination = _stage_fresh(tmp_path, monkeypatch)
    validated = input_bundle.validate_fresh_input_bundle(
        destination,
        expected_dataset_slug="owner/spec0021-inputs",
    )
    assert len(validated.files) == _FRESH_PAYLOAD_FILES
    contract_path = destination / input_bundle.FRESH_PROVENANCE_FILENAME
    contract = cast(
        "Mapping[str, object]",
        json.loads(contract_path.read_text(encoding="utf-8")),
    )
    assert contract["schema_version"] == "spec0021.input_bundle.v1"
    assert contract_path.stat().st_mtime_ns >= max(
        path.stat().st_mtime_ns
        for path in destination.rglob("*")
        if path.is_file() and path != contract_path
    )
    with pytest.raises(FileExistsError):
        _stage_fresh(tmp_path, monkeypatch)


def test_fresh_upload_archive_round_trips_the_sealed_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Kaggle envelope is flat while its archive restores exact paths."""
    bundle = _stage_fresh(tmp_path, monkeypatch)
    envelope = tmp_path / "upload"
    archive_path = input_bundle.stage_fresh_upload_archive(
        bundle,
        envelope,
        expected_dataset_slug="owner/spec0021-inputs",
    )
    assert {path.name for path in envelope.iterdir()} == {
        input_bundle.DATASET_METADATA_FILENAME,
        input_bundle.FRESH_UPLOAD_ARCHIVE_FILENAME,
    }
    contract = bundle / input_bundle.FRESH_PROVENANCE_FILENAME
    contract_sha256 = _sha256(contract)
    extracted = input_bundle.extract_fresh_upload_archive(
        archive_path,
        tmp_path / "extracted",
        expected_dataset_slug="owner/spec0021-inputs",
        expected_provenance_sha256=contract_sha256,
    )
    assert extracted.provenance_sha256 == contract_sha256
    assert {
        path.relative_to(extracted.root).as_posix()
        for path in extracted.root.rglob("*")
        if path.is_file()
    } == {
        path.relative_to(bundle).as_posix()
        for path in bundle.rglob("*")
        if path.is_file()
    }


def test_fresh_upload_archive_rejects_an_unsealed_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Archive extraction never accepts a file outside the contract allow-list."""
    bundle = _stage_fresh(tmp_path, monkeypatch)
    archive_path = input_bundle.stage_fresh_upload_archive(
        bundle,
        tmp_path / "upload",
        expected_dataset_slug="owner/spec0021-inputs",
    )
    with zipfile.ZipFile(archive_path, mode="a") as archive:
        archive.writestr("unexpected.txt", b"no")
    with pytest.raises(ValueError, match="member allow-list"):
        input_bundle.extract_fresh_upload_archive(
            archive_path,
            tmp_path / "extracted",
            expected_dataset_slug="owner/spec0021-inputs",
            expected_provenance_sha256=_sha256(
                bundle / input_bundle.FRESH_PROVENANCE_FILENAME,
            ),
        )


@pytest.mark.parametrize("failure", ["missing", "extra", "mutated"])
def test_fresh_bundle_rejects_every_allow_list_or_byte_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    """Missing, extra, or byte-mutated fresh inputs fail closed."""
    destination = _stage_fresh(tmp_path, monkeypatch)
    target = destination / input_bundle.NORMAL_CHECKPOINT_NAME
    if failure == "missing":
        target.unlink()
    elif failure == "extra":
        (destination / "extra.txt").write_text("unexpected", encoding="utf-8")
    else:
        target.write_bytes(b"mutated")
    with pytest.raises(ValueError, match=r"Bundle (contains|file bytes)"):
        input_bundle.validate_fresh_input_bundle(
            destination,
            expected_dataset_slug="owner/spec0021-inputs",
        )


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )


def _resume_artifacts(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, input_bundle.ResumeBundleAuthority]:
    manifest = root / "run_01_of_05.csv"
    _write_work_manifest(manifest, atlas_index=1, wsi_id=101)
    manifest_sha = _sha256(manifest)
    monkeypatch.setattr(
        input_bundle,
        "EXPECTED_WORK_MANIFEST_SHA256",
        {1: manifest_sha},
    )
    monkeypatch.setattr(
        input_bundle,
        "EXPECTED_CHECKPOINT_SHA256",
        {"normal_vae": _SHA_A, "so2_vae": _SHA_B},
    )
    monkeypatch.setattr(input_bundle, "EXPECTED_UNION_SHA256", "c" * 64)
    monkeypatch.setattr(
        latent_shards,
        "EXPECTED_CHECKPOINT_SHA256",
        {"normal_vae": _SHA_A, "so2_vae": _SHA_B},
    )
    monkeypatch.setattr(latent_shards, "EXPECTED_UNION_SHA256", "c" * 64)
    artifacts = root / "artifacts"
    artifacts.mkdir()
    empty_sha = hashlib.sha256(b"").hexdigest()
    for model, checkpoint in (("normal_vae", _SHA_A), ("so2_vae", _SHA_B)):
        stem = f"{model}_mu_run_01_of_05"
        (artifacts / f"{stem}.bin.partial").write_bytes(
            make_latent_shard_header(tensor_count=0, payload_crc32=0),
        )
        _write_json(
            artifacts / f"{stem}.resume.json",
            {
                "checkpoint_sha256": checkpoint,
                "committed_bytes": 64,
                "committed_rows": 0,
                "completed_wsi_ids": [],
                "model_name": model,
                "pinned_union_sha256": "c" * 64,
                "prefix_crc32": 0,
                "prefix_sha256": empty_sha,
                "schema_version": "spec0020.latent_resume.v1",
                "source_manifest": {
                    "logical_basename": manifest.name,
                    "row_count": 1,
                    "run_number": 1,
                    "sha256": manifest_sha,
                },
                "status": "in_progress",
                "tensor": {
                    "dtype": "float32_le",
                    "layout": "CHW",
                    "record_bytes": 65_536,
                    "shape": [16, 32, 32],
                },
            },
        )
    worker_name = "spec0021_worker_run_01.resume.json"
    _write_json(
        artifacts / worker_name,
        {
            "active_wsi": None,
            "completed_wsi_evidence": [],
            "input_bundle_sha256": "d" * 64,
            "normal_prefix": {"committed_rows": 0, "completed_wsi_ids": []},
            "run_config_sha256": "e" * 64,
            "run_number": 1,
            "schema_version": "spec0021.dual_worker_resume.v1",
            "so2_prefix": {"committed_rows": 0, "completed_wsi_ids": []},
            "status": "in_progress",
            "work_manifest_sha256": manifest_sha,
        },
    )
    _write_json(
        artifacts / "spec0021_incomplete_run_01.json",
        {
            "catch_up_model": None,
            "normal_completed_wsi_ids": [],
            "recovery_kind": None,
            "schema_version": "spec0021.dual_worker_incomplete.v1",
            "so2_completed_wsi_ids": [],
            "status": "incomplete",
            "worker_resume": worker_name,
        },
    )
    authority = input_bundle.ResumeBundleAuthority(
        provenance_sha256="0" * 64,
        dataset_slug="owner/run-01-resume",
        dataset_version=1,
        run_number=1,
        input_bundle_sha256="d" * 64,
        run_config_sha256="e" * 64,
        work_manifest_sha256=manifest_sha,
    )
    return artifacts, manifest, authority


def _read_mapping(path: Path) -> dict[str, object]:
    return cast(
        "dict[str, object]",
        json.loads(path.read_text(encoding="utf-8")),
    )


def _make_complete_sidecar(
    *,
    model: str,
    checkpoint: str,
    manifest: Path,
    payload: bytes,
) -> dict[str, object]:
    identity = {"atlas_row_index": 1, "wsi_id": 101, "x": 0, "y": 0}
    return {
        "checkpoint_sha256": checkpoint,
        "completed_wsi_count": 1,
        "completed_wsi_ids": [101],
        "file_size": 64 + len(payload),
        "model_name": model,
        "payload_bytes": len(payload),
        "payload_crc32": zlib.crc32(payload),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "pinned_union_sha256": "c" * 64,
        "schema_version": "spec0020.latent_shard.v1",
        "source_manifest": {
            "first_identity": identity,
            "last_identity": identity,
            "logical_basename": manifest.name,
            "row_count": 1,
            "run_number": 1,
            "sha256": _sha256(manifest),
        },
        "status": "complete",
        "tensor": {
            "count": 1,
            "dtype": "float32_le",
            "layout": "CHW",
            "record_bytes": 65_536,
            "shape": [16, 32, 32],
        },
    }


def _set_complete_writer_window(
    artifacts: Path,
    manifest: Path,
    *,
    window: str,
) -> None:
    payload = bytes(65_536)
    crc = zlib.crc32(payload)
    payload_sha = hashlib.sha256(payload).hexdigest()
    for model, checkpoint in (("normal_vae", _SHA_A), ("so2_vae", _SHA_B)):
        stem = f"{model}_mu_run_01_of_05"
        partial = artifacts / f"{stem}.bin.partial"
        state_path = artifacts / f"{stem}.resume.json"
        state = _read_mapping(state_path)
        state.update(
            {
                "committed_bytes": 64 + len(payload),
                "committed_rows": 1,
                "completed_wsi_ids": [101],
                "prefix_crc32": crc,
                "prefix_sha256": payload_sha,
            },
        )
        _write_json(state_path, state)
        final = artifacts / f"{stem}.bin"
        partial.write_bytes(
            make_latent_shard_header(tensor_count=1, payload_crc32=crc) + payload,
        )
        partial.rename(final)
        if window in {"complete_with_stale_state", "complete"}:
            _write_json(
                artifacts / f"{stem}.json",
                _make_complete_sidecar(
                    model=model,
                    checkpoint=checkpoint,
                    manifest=manifest,
                    payload=payload,
                ),
            )
        if window == "complete":
            state_path.unlink()
    worker_path = artifacts / "spec0021_worker_run_01.resume.json"
    worker = _read_mapping(worker_path)
    worker["active_wsi"] = {
        "evidence": {
            "png_bytes": 1,
            "png_sha256": "f" * 64,
            "transcript_sha256": "1" * 64,
            "wsi_id": 101,
        },
        "row_end": 1,
        "row_start": 0,
    }
    _write_json(worker_path, worker)
    marker_path = artifacts / "spec0021_incomplete_run_01.json"
    marker = _read_mapping(marker_path)
    marker.update(
        {
            "normal_completed_wsi_ids": [101],
            "recovery_kind": "verify_converged",
            "so2_completed_wsi_ids": [101],
        },
    )
    _write_json(marker_path, marker)


def test_resume_bundle_validates_and_copies_only_working_set_durably(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A validated attachment copies only its writer working set."""
    artifacts, manifest, binding = _resume_artifacts(tmp_path, monkeypatch)
    staged = tmp_path / "resume_bundle"
    authority = input_bundle.stage_resume_bundle(
        staged,
        artifacts_dir=artifacts,
        work_manifest_path=manifest,
        dataset_slug=binding.dataset_slug,
        dataset_version=binding.dataset_version,
        run_number=1,
        input_bundle_sha256=binding.input_bundle_sha256,
        run_config_sha256=binding.run_config_sha256,
    )
    validated = input_bundle.validate_resume_bundle(
        staged,
        authority=authority,
        work_manifest_path=manifest,
    )
    assert set(validated.writer_windows.values()) == {"partial_with_state"}
    working = input_bundle.copy_validated_resume_bundle(
        staged,
        tmp_path / "working",
        authority=authority,
        work_manifest_path=manifest,
    )
    assert {path.name for path in working.iterdir()} == {
        path.name for path in artifacts.iterdir()
    }


@pytest.mark.parametrize(
    "window",
    ["final_with_state", "complete_with_stale_state", "complete"],
)
def test_resume_bundle_accepts_each_final_spec0020_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    window: str,
) -> None:
    """Each allowed final/stale-state window passes full payload validation."""
    artifacts, manifest, binding = _resume_artifacts(tmp_path, monkeypatch)
    _set_complete_writer_window(artifacts, manifest, window=window)
    destination = tmp_path / "resume_bundle"
    authority = input_bundle.stage_resume_bundle(
        destination,
        artifacts_dir=artifacts,
        work_manifest_path=manifest,
        dataset_slug=binding.dataset_slug,
        dataset_version=1,
        run_number=1,
        input_bundle_sha256=binding.input_bundle_sha256,
        run_config_sha256=binding.run_config_sha256,
    )
    validated = input_bundle.validate_resume_bundle(
        destination,
        authority=authority,
        work_manifest_path=manifest,
    )
    assert set(validated.writer_windows.values()) == {window}


def test_resume_bundle_rejects_mutation_extra_and_invalid_writer_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resume mutation, extras, and incomplete writer windows fail closed."""
    artifacts, manifest, binding = _resume_artifacts(tmp_path, monkeypatch)
    staged = tmp_path / "resume_bundle"
    authority = input_bundle.stage_resume_bundle(
        staged,
        artifacts_dir=artifacts,
        work_manifest_path=manifest,
        dataset_slug=binding.dataset_slug,
        dataset_version=1,
        run_number=1,
        input_bundle_sha256=binding.input_bundle_sha256,
        run_config_sha256=binding.run_config_sha256,
    )
    (staged / "normal_vae_mu_run_01_of_05.bin.partial").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Bundle file bytes"):
        input_bundle.validate_resume_bundle(
            staged,
            authority=authority,
            work_manifest_path=manifest,
        )
    (artifacts / "unexpected.txt").write_text("no", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact directory"):
        input_bundle.stage_resume_bundle(
            tmp_path / "bad-extra",
            artifacts_dir=artifacts,
            work_manifest_path=manifest,
            dataset_slug=binding.dataset_slug,
            dataset_version=1,
            run_number=1,
            input_bundle_sha256=binding.input_bundle_sha256,
            run_config_sha256=binding.run_config_sha256,
        )
    (artifacts / "unexpected.txt").unlink()
    (artifacts / "so2_vae_mu_run_01_of_05.resume.json").unlink()
    with pytest.raises(ValueError, match="accepted Spec 0020 recovery window"):
        input_bundle.stage_resume_bundle(
            tmp_path / "bad-window",
            artifacts_dir=artifacts,
            work_manifest_path=manifest,
            dataset_slug=binding.dataset_slug,
            dataset_version=1,
            run_number=1,
            input_bundle_sha256=binding.input_bundle_sha256,
            run_config_sha256=binding.run_config_sha256,
        )


@pytest.mark.parametrize(
    ("target", "match"),
    [("writer", "WSI prefix"), ("marker", "Incomplete marker")],
)
def test_resume_bundle_rejects_tampered_completed_wsi_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    match: str,
) -> None:
    """Writer-state and incomplete-marker WSI IDs are independently checked."""
    artifacts, manifest, binding = _resume_artifacts(tmp_path, monkeypatch)
    if target == "writer":
        path = artifacts / "normal_vae_mu_run_01_of_05.resume.json"
        payload = _read_mapping(path)
        payload["completed_wsi_ids"] = [999]
    else:
        path = artifacts / "spec0021_incomplete_run_01.json"
        payload = _read_mapping(path)
        payload["normal_completed_wsi_ids"] = [999]
    _write_json(path, payload)
    with pytest.raises(ValueError, match=match):
        input_bundle.stage_resume_bundle(
            tmp_path / "bad-identities",
            artifacts_dir=artifacts,
            work_manifest_path=manifest,
            dataset_slug=binding.dataset_slug,
            dataset_version=1,
            run_number=1,
            input_bundle_sha256=binding.input_bundle_sha256,
            run_config_sha256=binding.run_config_sha256,
        )
