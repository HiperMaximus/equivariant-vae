# Copyright 2026 HiperMaximus
"""Tests for the Spec 0020 fixed-record FP32 latent storage contract."""

from __future__ import annotations

import csv
import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
import torch
from torch import Tensor

from eqvae.data import latent_shards
from eqvae.data.latent_shards import (
    EXPECTED_CHECKPOINT_SHA256,
    LATENT_SHARD_HEADER_SIZE,
    LATENT_SHARD_MAGIC,
    LatentRowIdentity,
    LatentShardWriter,
    LatentTensorDataset,
    make_latent_shard_header,
    parse_latent_shard_header,
    validate_latent_artifact,
    validate_latent_store_pair,
)

UNION_HEADER = (
    "atlas_row_index",
    "wsi_id",
    "diagnosis_label",
    "diagnosis_index",
    "x",
    "y",
    "split",
    "cancer_ae_selected",
    "tissue_selected",
    "tissue_label",
)
CANCER_TASK_HEADER = UNION_HEADER[:7]
TISSUE_TASK_HEADER = (
    *CANCER_TASK_HEADER,
    "tissue_label",
    "annotated_fraction",
    "dominant_fraction",
    "purity",
    "tumor_fraction",
    "stroma_fraction",
    "necrosis_fraction",
)
FAKE_UNION_SHA256 = "a" * 64
FIRST_FLOAT = 1.25
SECOND_FLOAT = -2.5
HEADER_TENSOR_COUNT = 7
HEADER_CRC32 = 123
EXPECTED_STORE_ROWS = 5
FIRST_WSI_ROWS = 2
HEADER_CHANNELS_OFFSET = 20
HEADER_VERSION_OFFSET = 32
HEADER_DTYPE_OFFSET = 36
HEADER_LAYOUT_OFFSET = 40
INJECTED_SIDECAR_ERROR = "injected sidecar failure"


@dataclass(frozen=True)
class ManifestFixture:
    """Paths and exact identities for one small ordered work manifest."""

    path: Path
    sha256: str
    rows: tuple[dict[str, str], ...]
    identities: tuple[LatentRowIdentity, ...]


def test_header_is_distinct_fixed_and_little_endian() -> None:
    """Use a distinct fixed header so patch readers fail closed."""
    header_bytes = make_latent_shard_header(
        tensor_count=HEADER_TENSOR_COUNT,
        payload_crc32=HEADER_CRC32,
    )
    header = parse_latent_shard_header(header_bytes)

    assert len(header_bytes) == LATENT_SHARD_HEADER_SIZE
    assert header_bytes[:8] == LATENT_SHARD_MAGIC
    assert struct.unpack_from("<Q", header_bytes, 12)[0] == HEADER_TENSOR_COUNT
    assert header.tensor_count == HEADER_TENSOR_COUNT
    assert header.payload_crc32 == HEADER_CRC32

    with pytest.raises(ValueError, match="Unsupported latent header"):
        parse_latent_shard_header(b"UBC_DATA" + header_bytes[8:])


@pytest.mark.parametrize(
    ("offset", "replacement"),
    [
        (HEADER_CHANNELS_OFFSET, struct.pack("<i", 15)),
        (HEADER_VERSION_OFFSET, struct.pack("<i", 2)),
        (HEADER_DTYPE_OFFSET, b"F16L"),
        (HEADER_LAYOUT_OFFSET, b"HWC"),
    ],
)
def test_header_rejects_each_fixed_contract_dimension(
    offset: int,
    replacement: bytes,
) -> None:
    """Reject compatible-size headers that describe a different tensor contract."""
    header = bytearray(make_latent_shard_header(tensor_count=1, payload_crc32=0))
    header[offset : offset + len(replacement)] = replacement

    with pytest.raises(ValueError, match="Unsupported latent header"):
        parse_latent_shard_header(bytes(header))


def test_writer_and_mmap_round_trip_exact_fp32_rows(tmp_path: Path) -> None:
    """Preserve full posterior maps without quantization or reordering."""
    fixture = _write_manifest(tmp_path, run_number=1, rows=_four_rows())
    bin_path = tmp_path / "normal_mu_run_01_of_05.bin"
    tensors = _known_tensors(len(fixture.rows)).transpose(2, 3)
    writer = _writer(bin_path, fixture)

    writer.append_batch(
        row_start=0,
        identities=fixture.identities[:2],
        tensors=tensors[:2],
    )
    writer.append_batch(
        row_start=2,
        identities=fixture.identities[2:],
        tensors=tensors[2:],
    )
    finalized = writer.finalize()

    artifact = _validate_artifact(bin_path, fixture)
    dataset = _dataset(bin_path, fixture, advice="sequential")
    first = dataset[0].clone()
    second = dataset[1].clone()
    last = dataset[-1].clone()
    state = dataset.__getstate__()
    del dataset

    assert artifact.header.tensor_count == len(fixture.rows)
    assert finalized.file_sha256 == _sha256(bin_path)
    assert torch.equal(first, tensors[0].contiguous())
    assert torch.equal(second, tensors[1].contiguous())
    assert torch.equal(last, tensors[-1].contiguous())
    assert state["_file"] is None
    assert state["_mmap"] is None
    with bin_path.open("rb") as handle:
        handle.seek(LATENT_SHARD_HEADER_SIZE)
        flat = tensors[0].contiguous().reshape(-1)
        expected = (float(flat[0].item()), float(flat[1].item()))
        assert struct.unpack("<ff", handle.read(8)) == pytest.approx(expected)


def test_writer_rejects_false_order_and_invalid_tensor_contract(tmp_path: Path) -> None:
    """Reject wrong semantic order, dtype, and nonfinite FP32 evidence."""
    fixture = _write_manifest(tmp_path, run_number=1, rows=_four_rows())
    writer = _writer(tmp_path / "latent.bin", fixture)
    tensors = _known_tensors(2)

    with pytest.raises(ValueError, match="identities do not match"):
        writer.append_batch(
            row_start=0,
            identities=tuple(reversed(fixture.identities[:2])),
            tensors=tensors,
        )
    with pytest.raises(TypeError, match="Expected FP32"):
        writer.append_batch(
            row_start=0,
            identities=fixture.identities[:2],
            tensors=tensors.to(torch.float16),
        )
    tensors[0, 0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="nonfinite"):
        writer.append_batch(
            row_start=0,
            identities=fixture.identities[:2],
            tensors=tensors,
        )
    writer.close()


def test_writer_rejects_wsi_crossing_overrun_and_incomplete_finalize(
    tmp_path: Path,
) -> None:
    """Keep WSI commits atomic and reject rows beyond the bound manifest."""
    fixture = _write_manifest(tmp_path, run_number=1, rows=_four_rows())
    writer = _writer(tmp_path / "latent.bin", fixture)

    with pytest.raises(ValueError, match="cannot cross a WSI boundary"):
        writer.append_batch(
            row_start=0,
            identities=fixture.identities,
            tensors=_known_tensors(len(fixture.identities)),
        )
    writer.append_batch(
        row_start=0,
        identities=fixture.identities[:1],
        tensors=_known_tensors(1),
    )
    with pytest.raises(RuntimeError, match="Cannot finalize"):
        writer.finalize()
    with pytest.raises(ValueError, match="overruns"):
        writer.append_batch(
            row_start=1,
            identities=(*fixture.identities[1:], fixture.identities[-1]),
            tensors=_known_tensors(len(fixture.identities)),
        )
    writer.close()


def test_resume_truncates_uncommitted_wsi_and_matches_clean_bytes(
    tmp_path: Path,
) -> None:
    """Drop the uncommitted WSI tail after a Kaggle interruption."""
    fixture = _write_manifest(tmp_path, run_number=1, rows=_four_rows())
    tensors = _known_tensors(4)
    resumed_path = tmp_path / "resumed.bin"
    interrupted = _writer(resumed_path, fixture)
    interrupted.append_batch(
        row_start=0,
        identities=fixture.identities[:2],
        tensors=tensors[:2],
    )
    interrupted.append_batch(
        row_start=2,
        identities=fixture.identities[2:3],
        tensors=tensors[2:3],
    )
    interrupted.close()
    with Path(f"{resumed_path}.partial").open("ab") as handle:
        handle.write(b"partial-record-tail")

    resumed = _writer(resumed_path, fixture)
    assert resumed.next_row_start == FIRST_WSI_ROWS
    resumed.append_batch(
        row_start=2,
        identities=fixture.identities[2:],
        tensors=tensors[2:],
    )
    resumed.finalize()

    clean_path = tmp_path / "clean.bin"
    clean = _writer(clean_path, fixture)
    clean.append_batch(
        row_start=0,
        identities=fixture.identities[:2],
        tensors=tensors[:2],
    )
    clean.append_batch(
        row_start=2,
        identities=fixture.identities[2:],
        tensors=tensors[2:],
    )
    clean.finalize()

    assert resumed_path.read_bytes() == clean_path.read_bytes()
    assert (
        resumed_path.with_suffix(".json").read_bytes()
        == clean_path.with_suffix(
            ".json",
        ).read_bytes()
    )


def test_resume_rejects_mutated_committed_payload_and_state(tmp_path: Path) -> None:
    """Trust resume state only when committed bytes and its WSI prefix verify."""
    fixture = _write_manifest(tmp_path, run_number=1, rows=_four_rows())
    bin_path = tmp_path / "latent.bin"
    writer = _writer(bin_path, fixture)
    writer.append_batch(
        row_start=0,
        identities=fixture.identities[:2],
        tensors=_known_tensors(2),
    )
    writer.close()
    partial_path = Path(f"{bin_path}.partial")
    with partial_path.open("r+b") as handle:
        handle.seek(LATENT_SHARD_HEADER_SIZE)
        value = handle.read(1)
        handle.seek(LATENT_SHARD_HEADER_SIZE)
        handle.write(bytes([value[0] ^ 1]))
    with pytest.raises(ValueError, match="prefix CRC32"):
        _writer(bin_path, fixture)

    with partial_path.open("r+b") as handle:
        handle.seek(LATENT_SHARD_HEADER_SIZE)
        value = handle.read(1)
        handle.seek(LATENT_SHARD_HEADER_SIZE)
        handle.write(bytes([value[0] ^ 1]))
    state_path = bin_path.with_suffix(".resume.json")
    state = cast(
        "dict[str, object]",
        json.loads(state_path.read_text(encoding="utf-8")),
    )
    state["completed_wsi_ids"] = [999]
    state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ValueError, match="completed-WSI prefix"):
        _writer(bin_path, fixture)


def test_finalization_recovery_requires_valid_state_and_publishes_json_last(
    tmp_path: Path,
) -> None:
    """Recover each crash window only from proven state."""
    fixture = _write_manifest(tmp_path, run_number=1, rows=_four_rows())
    tensors = _known_tensors(4)
    bin_path = tmp_path / "latent.bin"
    writer = _writer(bin_path, fixture)
    _append_all(writer, fixture, tensors)
    state_bytes = writer.state_path.read_bytes()
    writer.close()
    state = cast("dict[str, object]", json.loads(state_bytes))
    prefix_crc32 = state["prefix_crc32"]
    assert isinstance(prefix_crc32, int)
    assert not isinstance(prefix_crc32, bool)
    with writer.partial_path.open("r+b") as handle:
        handle.write(
            make_latent_shard_header(
                tensor_count=len(fixture.identities),
                payload_crc32=prefix_crc32,
            ),
        )

    recovered = _writer(bin_path, fixture)
    assert recovered.complete
    assert bin_path.is_file()
    assert bin_path.with_suffix(".json").is_file()

    recovered.state_path.write_bytes(state_bytes)
    reopened = _writer(bin_path, fixture)
    assert reopened.complete
    assert not reopened.state_path.exists()

    reopened.state_path.write_bytes(state_bytes.replace(b"normal_vae", b"normal_bad"))
    with pytest.raises(ValueError, match="Resume state provenance mismatch"):
        _writer(bin_path, fixture)

    reopened.state_path.unlink()
    bin_path.with_suffix(".json").unlink()
    with pytest.raises(ValueError, match="Orphan final"):
        _writer(bin_path, fixture)


def test_sidecar_publication_failure_leaves_recoverable_incomplete_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Treat the JSON as completion and recover a renamed binary after failure."""
    fixture = _write_manifest(tmp_path, run_number=1, rows=_four_rows())
    bin_path = tmp_path / "latent.bin"
    writer = _writer(bin_path, fixture)
    _append_all(writer, fixture, _known_tensors(len(fixture.identities)))

    def fail_publication(_writer_instance: LatentShardWriter) -> None:
        raise OSError(INJECTED_SIDECAR_ERROR)

    with monkeypatch.context() as context:
        context.setattr(LatentShardWriter, "_publish_sidecar", fail_publication)
        with pytest.raises(OSError, match="injected sidecar failure"):
            writer.finalize()

    assert bin_path.is_file()
    assert not bin_path.with_suffix(".json").exists()
    assert writer.state_path.is_file()
    recovered = _writer(bin_path, fixture)
    assert recovered.complete
    assert bin_path.with_suffix(".json").is_file()


def test_full_validation_detects_same_size_payload_mutation(tmp_path: Path) -> None:
    """Detect changed tensor bytes during full publication validation."""
    fixture = _write_manifest(tmp_path, run_number=1, rows=_four_rows())
    bin_path = tmp_path / "latent.bin"
    writer = _writer(bin_path, fixture)
    _append_all(writer, fixture, _known_tensors(4))
    writer.finalize()
    with bin_path.open("r+b") as handle:
        handle.seek(-1, 2)
        value = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([value[0] ^ 1]))

    _validate_artifact(bin_path, fixture, validate_payload=False)
    with pytest.raises(ValueError, match="payload checksum"):
        _validate_artifact(bin_path, fixture)
    with pytest.raises(ValueError, match="payload checksum"):
        _writer(bin_path, fixture)


def test_lightweight_open_rejects_sidecar_and_model_provenance_mutation(
    tmp_path: Path,
) -> None:
    """Reject altered metadata without needing a payload scan."""
    fixture = _write_manifest(tmp_path, run_number=1, rows=_four_rows())
    bin_path = tmp_path / "latent.bin"
    writer = _writer(bin_path, fixture)
    _append_all(writer, fixture, _known_tensors(len(fixture.identities)))
    writer.finalize()

    with pytest.raises(ValueError, match="checkpoint"):
        LatentTensorDataset(
            bin_path=bin_path,
            manifest_path=fixture.path,
            run_number=1,
            model_name="so2_vae",
            checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256["normal_vae"],
            expected_checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256["so2_vae"],
            expected_manifest_sha256=fixture.sha256,
            expected_union_sha256=FAKE_UNION_SHA256,
        )

    sidecar_path = bin_path.with_suffix(".json")
    sidecar = cast(
        "dict[str, object]",
        json.loads(sidecar_path.read_text(encoding="utf-8")),
    )
    sidecar["status"] = "in_progress"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(ValueError, match="sidecar mismatch"):
        _dataset(bin_path, fixture, advice="random")


def test_dual_store_validator_proves_twelve_views_and_publishes_audit_last(  # noqa: PLR0914, PLR0915
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bind six manifests into twelve logical views without copying tensors."""
    rows = _five_store_rows()
    union_path = tmp_path / "union_patch_manifest.csv"
    _write_rows(union_path, rows)
    union_sha = _sha256(union_path)
    work_paths: dict[int, Path] = {}
    work_hashes: dict[int, str] = {}
    normal_paths: dict[int, Path] = {}
    so2_paths: dict[int, Path] = {}
    for run_number, row in enumerate(rows, start=1):
        fixture = _write_manifest(tmp_path, run_number=run_number, rows=(row,))
        work_paths[run_number] = fixture.path
        work_hashes[run_number] = fixture.sha256
        model_destinations: tuple[
            tuple[latent_shards.ModelName, dict[int, Path]],
            ...,
        ] = (
            ("normal_vae", normal_paths),
            ("so2_vae", so2_paths),
        )
        for model_name, destination in model_destinations:
            bin_path = tmp_path / f"{model_name}_{run_number}.bin"
            writer = _writer(
                bin_path,
                fixture,
                model_name=model_name,
                union_sha256=union_sha,
            )
            writer.append_batch(
                row_start=0,
                identities=fixture.identities,
                tensors=_known_tensors(1),
            )
            writer.finalize()
            destination[run_number] = bin_path
    task_paths, task_hashes, task_counts = _write_task_manifests(tmp_path, rows)

    monkeypatch.setattr(latent_shards, "EXPECTED_UNION_SHA256", union_sha)
    monkeypatch.setattr(latent_shards, "EXPECTED_WORK_MANIFEST_SHA256", work_hashes)
    monkeypatch.setattr(
        latent_shards,
        "EXPECTED_TASK_MANIFEST_SHA256",
        task_hashes,
    )
    monkeypatch.setattr(latent_shards, "EXPECTED_TASK_VIEW_ROWS", task_counts)
    monkeypatch.setattr(latent_shards, "EXPECTED_UNION_ROWS", EXPECTED_STORE_ROWS)
    monkeypatch.setattr(latent_shards, "EXPECTED_CANCER_ROWS", 4)
    monkeypatch.setattr(latent_shards, "EXPECTED_TISSUE_ROWS", 4)
    monkeypatch.setattr(latent_shards, "EXPECTED_SHARED_ROWS", 3)
    audit_path = tmp_path / "global_audit.json"
    result = validate_latent_store_pair(
        union_manifest_path=union_path,
        expected_union_sha256=union_sha,
        work_manifest_paths=work_paths,
        expected_work_manifest_hashes=work_hashes,
        task_manifest_paths=task_paths,
        expected_task_manifest_hashes=task_hashes,
        normal_shards=normal_paths,
        so2_shards=so2_paths,
        normal_checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256["normal_vae"],
        expected_normal_checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256["normal_vae"],
        so2_checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256["so2_vae"],
        expected_so2_checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256["so2_vae"],
        global_audit_path=audit_path,
    )

    assert result["row_count"] == EXPECTED_STORE_ROWS
    assert result["task_counts_per_model"] == {
        "cancer": 4,
        "tissue": 4,
        "shared": 3,
    }
    expected_location_hashes = {
        "cancer_train": _location_hash((1, 4)),
        "cancer_validation": _location_hash((2,)),
        "cancer_test": _location_hash((3,)),
        "tissue_train": _location_hash((1,)),
        "tissue_validation": _location_hash((2, 5)),
        "tissue_test": _location_hash((3,)),
    }
    assert result["logical_view_counts"] == {
        f"{model}/{task_split.replace('_', '/')}": count
        for model in ("normal_vae", "so2_vae")
        for task_split, count in task_counts.items()
    }
    assert result["logical_view_location_sha256"] == {
        f"{model}/{task_split.replace('_', '/')}": digest
        for model in ("normal_vae", "so2_vae")
        for task_split, digest in expected_location_hashes.items()
    }
    audit = cast(
        "dict[str, object]",
        json.loads(audit_path.read_text(encoding="utf-8")),
    )
    assert audit["status"] == "complete"
    assert audit["validation"] == result
    missing_normal = dict(normal_paths)
    missing_normal.pop(5)
    with pytest.raises(ValueError, match="exactly runs 1 through 5"):
        validate_latent_store_pair(
            union_manifest_path=union_path,
            expected_union_sha256=union_sha,
            work_manifest_paths=work_paths,
            expected_work_manifest_hashes=work_hashes,
            task_manifest_paths=task_paths,
            expected_task_manifest_hashes=task_hashes,
            normal_shards=missing_normal,
            so2_shards=so2_paths,
            normal_checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256["normal_vae"],
            expected_normal_checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256["normal_vae"],
            so2_checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256["so2_vae"],
            expected_so2_checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256["so2_vae"],
            validate_payload=False,
        )

    cancer_train = task_paths["cancer_train"]
    canonical_cancer_train = cancer_train.read_text(encoding="utf-8")
    semantic_mutations = {
        "wrong_split": (
            canonical_cancer_train.replace(",train\n", ",test\n", 1),
            "wrong split",
        ),
        "wrong_identity": (
            canonical_cancer_train.replace("\n0,1,HGSC", "\n99,1,HGSC", 1),
            "absent from union",
        ),
        "wrong_header": (
            canonical_cancer_train.replace(
                "atlas_row_index",
                "atlas_index",
                1,
            ),
            "Unexpected task manifest header",
        ),
    }
    for mutation_name, (mutated_text, error_pattern) in semantic_mutations.items():
        cancer_train.write_text(
            mutated_text,
            encoding="utf-8",
            newline="",
        )
        mutated_hashes = {**task_hashes, "cancer_train": _sha256(cancer_train)}
        monkeypatch.setattr(
            latent_shards,
            "EXPECTED_TASK_MANIFEST_SHA256",
            mutated_hashes,
        )
        failed_audit_path = tmp_path / f"failed_{mutation_name}_audit.json"
        with pytest.raises(ValueError, match=error_pattern):
            validate_latent_store_pair(
                union_manifest_path=union_path,
                expected_union_sha256=union_sha,
                work_manifest_paths=work_paths,
                expected_work_manifest_hashes=work_hashes,
                task_manifest_paths=task_paths,
                expected_task_manifest_hashes=mutated_hashes,
                normal_shards=normal_paths,
                so2_shards=so2_paths,
                normal_checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256["normal_vae"],
                expected_normal_checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256[
                    "normal_vae"
                ],
                so2_checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256["so2_vae"],
                expected_so2_checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256["so2_vae"],
                validate_payload=False,
                global_audit_path=failed_audit_path,
            )
        assert not failed_audit_path.exists()

    cancer_train.write_text(canonical_cancer_train, encoding="utf-8", newline="")
    monkeypatch.setattr(latent_shards, "EXPECTED_TASK_MANIFEST_SHA256", task_hashes)
    wrong_counts = {**task_counts, "cancer_train": task_counts["cancer_train"] + 1}
    monkeypatch.setattr(latent_shards, "EXPECTED_TASK_VIEW_ROWS", wrong_counts)
    with pytest.raises(ValueError, match="Task view count disagrees"):
        validate_latent_store_pair(
            union_manifest_path=union_path,
            expected_union_sha256=union_sha,
            work_manifest_paths=work_paths,
            expected_work_manifest_hashes=work_hashes,
            task_manifest_paths=task_paths,
            expected_task_manifest_hashes=task_hashes,
            normal_shards=normal_paths,
            so2_shards=so2_paths,
            normal_checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256["normal_vae"],
            expected_normal_checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256["normal_vae"],
            so2_checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256["so2_vae"],
            expected_so2_checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256["so2_vae"],
            validate_payload=False,
            global_audit_path=tmp_path / "failed_count_audit.json",
        )
    assert not (tmp_path / "failed_count_audit.json").exists()


def _writer(
    bin_path: Path,
    fixture: ManifestFixture,
    *,
    model_name: latent_shards.ModelName = "normal_vae",
    union_sha256: str = FAKE_UNION_SHA256,
) -> LatentShardWriter:
    checkpoint = EXPECTED_CHECKPOINT_SHA256[model_name]
    return LatentShardWriter(
        bin_path=bin_path,
        manifest_path=fixture.path,
        run_number=int(fixture.path.name[4:6]),
        model_name=model_name,
        checkpoint_sha256=checkpoint,
        expected_checkpoint_sha256=checkpoint,
        expected_manifest_sha256=fixture.sha256,
        expected_union_sha256=union_sha256,
    )


def _validate_artifact(
    bin_path: Path,
    fixture: ManifestFixture,
    *,
    validate_payload: bool = True,
) -> latent_shards.LatentArtifact:
    checkpoint = EXPECTED_CHECKPOINT_SHA256["normal_vae"]
    return validate_latent_artifact(
        bin_path=bin_path,
        manifest_path=fixture.path,
        run_number=int(fixture.path.name[4:6]),
        model_name="normal_vae",
        checkpoint_sha256=checkpoint,
        expected_checkpoint_sha256=checkpoint,
        expected_manifest_sha256=fixture.sha256,
        expected_union_sha256=FAKE_UNION_SHA256,
        validate_payload=validate_payload,
    )


def _dataset(
    bin_path: Path,
    fixture: ManifestFixture,
    *,
    advice: latent_shards.MmapAdvice,
) -> LatentTensorDataset:
    checkpoint = EXPECTED_CHECKPOINT_SHA256["normal_vae"]
    return LatentTensorDataset(
        bin_path=bin_path,
        manifest_path=fixture.path,
        run_number=int(fixture.path.name[4:6]),
        model_name="normal_vae",
        checkpoint_sha256=checkpoint,
        expected_checkpoint_sha256=checkpoint,
        expected_manifest_sha256=fixture.sha256,
        expected_union_sha256=FAKE_UNION_SHA256,
        advice=advice,
        validate_payload=False,
    )


def _append_all(
    writer: LatentShardWriter,
    fixture: ManifestFixture,
    tensors: Tensor,
) -> None:
    writer.append_batch(
        row_start=0,
        identities=fixture.identities[:2],
        tensors=tensors[:2],
    )
    writer.append_batch(
        row_start=2,
        identities=fixture.identities[2:],
        tensors=tensors[2:],
    )


def _known_tensors(count: int) -> Tensor:
    tensors = torch.arange(
        count * 16 * 32 * 32,
        dtype=torch.float32,
    ).reshape(count, 16, 32, 32)
    tensors[0, 0, 0, 0] = FIRST_FLOAT
    tensors[0, 0, 0, 1] = SECOND_FLOAT
    return tensors


def _write_manifest(
    root: Path,
    *,
    run_number: int,
    rows: tuple[dict[str, str], ...],
) -> ManifestFixture:
    path = root / f"run_{run_number:02d}_of_05.csv"
    _write_rows(path, rows)
    identities = tuple(
        LatentRowIdentity(
            atlas_row_index=int(row["atlas_row_index"]),
            wsi_id=int(row["wsi_id"]),
            x=int(row["x"]),
            y=int(row["y"]),
        )
        for row in rows
    )
    return ManifestFixture(path, _sha256(path), rows, identities)


def _write_rows(path: Path, rows: tuple[dict[str, str], ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(UNION_HEADER)
        for row in rows:
            writer.writerow([row[column] for column in UNION_HEADER])


def _write_task_manifests(
    root: Path,
    rows: tuple[dict[str, str], ...],
) -> tuple[dict[str, Path], dict[str, str], dict[str, int]]:
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    counts: dict[str, int] = {}
    for task_name in ("cancer", "tissue"):
        for split_name in ("train", "validation", "test"):
            task_split = f"{task_name}_{split_name}"
            path = root / f"{task_split}.csv"
            selected = tuple(
                row
                for row in rows
                if row["split"] == split_name
                and row[
                    "cancer_ae_selected" if task_name == "cancer" else "tissue_selected"
                ]
                == "true"
            )
            header = CANCER_TASK_HEADER if task_name == "cancer" else TISSUE_TASK_HEADER
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(header)
                for row in selected:
                    values = [row[column] for column in CANCER_TASK_HEADER]
                    if task_name == "tissue":
                        values.extend(
                            (
                                row["tissue_label"],
                                "1.0",
                                "1.0",
                                "1.0",
                                "1.0",
                                "0.0",
                                "0.0",
                            ),
                        )
                    writer.writerow(values)
            paths[task_split] = path
            hashes[task_split] = _sha256(path)
            counts[task_split] = len(selected)
    return paths, hashes, counts


def _location_hash(run_numbers: tuple[int, ...]) -> str:
    digest = hashlib.sha256()
    for run_number in run_numbers:
        digest.update(struct.pack("<II", run_number, 0))
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(  # noqa: PLR0913
    index: int,
    wsi_id: int,
    coordinates: tuple[int, int],
    *,
    cancer: bool,
    tissue: bool,
    split: str = "train",
) -> dict[str, str]:
    x, y = coordinates
    return {
        "atlas_row_index": str(index),
        "wsi_id": str(wsi_id),
        "diagnosis_label": "HGSC",
        "diagnosis_index": "0",
        "x": str(x),
        "y": str(y),
        "split": split,
        "cancer_ae_selected": str(cancer).lower(),
        "tissue_selected": str(tissue).lower(),
        "tissue_label": "tumor" if tissue else "",
    }


def _four_rows() -> tuple[dict[str, str], ...]:
    return (
        _row(0, 10, (0, 0), cancer=True, tissue=False),
        _row(1, 10, (256, 0), cancer=True, tissue=True),
        _row(2, 20, (0, 256), cancer=False, tissue=True),
        _row(3, 20, (256, 256), cancer=True, tissue=False),
    )


def _five_store_rows() -> tuple[dict[str, str], ...]:
    return (
        _row(0, 1, (0, 0), cancer=True, tissue=True),
        _row(
            1,
            2,
            (0, 0),
            cancer=True,
            tissue=True,
            split="validation",
        ),
        _row(2, 3, (0, 0), cancer=True, tissue=True, split="test"),
        _row(3, 4, (0, 0), cancer=True, tissue=False),
        _row(
            4,
            5,
            (0, 0),
            cancer=False,
            tissue=True,
            split="validation",
        ),
    )
