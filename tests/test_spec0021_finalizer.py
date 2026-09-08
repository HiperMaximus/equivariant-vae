# Copyright 2026 HiperMaximus
# pyright: reportPrivateUsage=false
"""Focused fail-closed tests for Spec 0021 global latent-store closure."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import pytest
import torch

from eqvae.cli import finalize_ubc_latent_stores as finalizer
from eqvae.data import latent_shards
from eqvae.data.latent_shards import (
    EXPECTED_CHECKPOINT_SHA256,
    LATENT_LOCATION_HEADER,
    LatentRowIdentity,
    LatentShardWriter,
    LatentTaskView,
)

FAKE_UNION_SHA256 = "a" * 64
EXPECTED_LOGICAL_VIEWS = 12
FIRST_WSI_ID = 101

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class ClosureFixture:
    """Five complete pair directories and their canonical work rows."""

    pair_root: Path
    input_contract: Path
    input_receipt: Path
    input_receipt_sha256: str
    config_root: Path
    config_hashes: dict[int, str]
    work_paths: dict[int, Path]
    work_hashes: dict[int, str]
    normal_shards: dict[int, Path]
    so2_shards: dict[int, Path]
    pair_audits: dict[int, Path]
    rows: dict[int, dict[str, str]]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, object]:
    payload = cast("object", json.loads(path.read_text(encoding="utf-8")))
    assert isinstance(payload, dict)
    return cast("dict[str, object]", payload)


def _dict_field(payload: dict[str, object], key: str) -> dict[str, object]:
    value = payload[key]
    assert isinstance(value, dict)
    return cast("dict[str, object]", value)


def _row(run: int, split: str) -> dict[str, str]:
    return {
        "atlas_row_index": str(run),
        "wsi_id": str(100 + run),
        "diagnosis_label": "CC",
        "diagnosis_index": "0",
        "x": "0",
        "y": "0",
        "split": split,
        "cancer_ae_selected": "true",
        "tissue_selected": "true",
        "tissue_label": "tumor",
    }


def _write_manifest(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(latent_shards.UNION_HEADER)
        writer.writerow([row[name] for name in latent_shards.UNION_HEADER])


def _artifact_identity(bin_path: Path) -> dict[str, object]:
    sidecar = bin_path.with_suffix(".json")
    return {
        "bin_name": bin_path.name,
        "bin_bytes": bin_path.stat().st_size,
        "bin_sha256": _sha256(bin_path),
        "sidecar_name": sidecar.name,
        "sidecar_bytes": sidecar.stat().st_size,
        "sidecar_sha256": _sha256(sidecar),
    }


def _closure_fixture(  # noqa: PLR0914
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> ClosureFixture:
    input_contract = tmp_path / "spec0021_input_contract.json"
    input_contract.write_text("{}\n", encoding="utf-8")
    receipt_payload: dict[str, object] = {
        "schema_version": "spec0021.input_dataset_receipt.v1",
        "dataset_reference": "owner/immutable-inputs",
        "dataset_version": 1,
    }
    input_receipt = tmp_path / "input_dataset_receipt.json"
    _write_json(input_receipt, receipt_payload)
    input_receipt_sha256 = _canonical_hash(receipt_payload)
    config_root = tmp_path / "configs"
    config_hashes: dict[int, str] = {}
    for run in range(1, 6):
        config_path = config_root / f"run_{run:02d}" / "spec0021_inference_config.json"
        config_path.parent.mkdir(parents=True)
        _write_json(
            config_path,
            {
                "schema_version": "spec0021.inference_config.v2",
                "mode": "production",
                "run_number": run,
                "input_contract_sha256": _sha256(input_contract),
                "input_dataset_receipt": receipt_payload,
            },
        )
        config_hashes[run] = _sha256(config_path)
    splits = {1: "train", 2: "validation", 3: "test", 4: "train", 5: "validation"}
    rows = {run: _row(run, split) for run, split in splits.items()}
    work_paths: dict[int, Path] = {}
    work_hashes: dict[int, str] = {}
    for run, row in rows.items():
        path = tmp_path / "work" / f"run_{run:02d}_of_05.csv"
        _write_manifest(path, row)
        work_paths[run] = path
        work_hashes[run] = _sha256(path)
    monkeypatch.setattr(latent_shards, "EXPECTED_WORK_MANIFEST_SHA256", work_hashes)
    monkeypatch.setattr(latent_shards, "EXPECTED_UNION_SHA256", FAKE_UNION_SHA256)
    monkeypatch.setattr(finalizer, "EXPECTED_WORK_MANIFEST_SHA256", work_hashes)
    monkeypatch.setattr(finalizer, "EXPECTED_UNION_SHA256", FAKE_UNION_SHA256)

    pair_root = tmp_path / "pairs"
    normal_shards: dict[int, Path] = {}
    so2_shards: dict[int, Path] = {}
    pair_audits: dict[int, Path] = {}
    for run, row in rows.items():
        dataset = pair_root / f"run_{run:02d}_of_05" / "dataset"
        dataset.mkdir(parents=True)
        identities = (
            LatentRowIdentity(
                atlas_row_index=int(row["atlas_row_index"]),
                wsi_id=int(row["wsi_id"]),
                x=0,
                y=0,
            ),
        )
        for model_name, destinations in (
            ("normal_vae", normal_shards),
            ("so2_vae", so2_shards),
        ):
            typed_model = cast("latent_shards.ModelName", model_name)
            bin_path = dataset / f"{model_name}_mu_run_{run:02d}_of_05.bin"
            checkpoint = EXPECTED_CHECKPOINT_SHA256[typed_model]
            writer = LatentShardWriter(
                bin_path=bin_path,
                manifest_path=work_paths[run],
                run_number=run,
                model_name=typed_model,
                checkpoint_sha256=checkpoint,
                expected_checkpoint_sha256=checkpoint,
                expected_manifest_sha256=work_hashes[run],
                expected_union_sha256=FAKE_UNION_SHA256,
            )
            writer.append_batch(
                row_start=0,
                identities=identities,
                tensors=torch.full((1, 16, 32, 32), float(run)),
            )
            writer.finalize()
            destinations[run] = bin_path
        audit_path = dataset / f"spec0021_pair_audit_run_{run:02d}_of_05.json"
        _write_json(
            audit_path,
            {
                "schema_version": "spec0021.latent_pair_audit.v1",
                "status": "complete",
                "run_number": run,
                "row_count": 1,
                "work_manifest_sha256": work_hashes[run],
                "union_manifest_sha256": FAKE_UNION_SHA256,
                "run_config_sha256": config_hashes[run],
                "input_receipt_sha256": input_receipt_sha256,
                "completed_wsi_evidence": [
                    {
                        "wsi_id": 100 + run,
                        "png_bytes": 1000 + run,
                        "png_sha256": "d" * 64,
                        "transcript_sha256": "e" * 64,
                    },
                ],
                "artifacts": {
                    "normal_vae": _artifact_identity(normal_shards[run]),
                    "so2_vae": _artifact_identity(so2_shards[run]),
                },
            },
        )
        pair_audits[run] = audit_path
    return ClosureFixture(
        pair_root,
        input_contract,
        input_receipt,
        input_receipt_sha256,
        config_root,
        config_hashes,
        work_paths,
        work_hashes,
        normal_shards,
        so2_shards,
        pair_audits,
        rows,
    )


def _write_task(path: Path, row: dict[str, str], *, tissue: bool) -> None:
    header = (
        latent_shards.TISSUE_TASK_HEADER if tissue else latent_shards.CANCER_TASK_HEADER
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        values = [row[name] for name in latent_shards.CANCER_TASK_HEADER]
        if tissue:
            values.extend(("tumor", "1", "1", "1", "1", "0", "0"))
        writer.writerow(values)


def _expected_location_hash(row: dict[str, str], run: int, *, tissue: bool) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(LATENT_LOCATION_HEADER)
    writer.writerow(
        (
            run,
            0,
            int(row["atlas_row_index"]),
            int(row["wsi_id"]),
            0,
            0,
            row["split"],
            "CC",
            0,
            "tumor" if tissue else "",
        ),
    )
    return hashlib.sha256(buffer.getvalue().encode()).hexdigest()


def _validated_file_hashes(
    normal: dict[int, Path],
    so2: dict[int, Path],
) -> dict[str, object]:
    return {
        "normal_vae": {str(run): _sha256(path) for run, path in normal.items()},
        "so2_vae": {str(run): _sha256(path) for run, path in so2.items()},
    }


def test_finalizer_publishes_exact_locations_twelve_views_and_audit_last(  # noqa: PLR0914
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Complete closure binds sidecars, six files, and twelve usable views."""
    fixture = _closure_fixture(tmp_path, monkeypatch)
    normal, so2, audits = finalizer._pair_paths(fixture.pair_root)  # noqa: SLF001
    config_hashes, receipt_sha256 = finalizer._expected_production_bindings(  # noqa: SLF001
        config_root=fixture.config_root,
        input_receipt_path=fixture.input_receipt,
        input_contract_sha256=_sha256(fixture.input_contract),
    )
    validated_file_hashes = _validated_file_hashes(normal, so2)
    original_sha256 = finalizer._sha256  # noqa: SLF001

    def reject_binary_rehash(path: Path) -> str:
        assert path.suffix != ".bin"
        return original_sha256(path)

    monkeypatch.setattr(finalizer, "_sha256", reject_binary_rehash)
    records = finalizer._validate_pair_audits(  # noqa: SLF001
        work_paths=fixture.work_paths,
        pair_audits=audits,
        normal_shards=normal,
        so2_shards=so2,
        expected_run_config_hashes=config_hashes,
        expected_input_receipt_sha256=receipt_sha256,
        validated_file_hashes=validated_file_hashes,
    )
    task_runs = {
        "cancer_train": 1,
        "cancer_validation": 2,
        "cancer_test": 3,
        "tissue_train": 4,
        "tissue_validation": 5,
        "tissue_test": 3,
    }
    task_paths: dict[str, Path] = {}
    location_hashes: dict[str, str] = {}
    view_counts: dict[str, int] = {}
    for name, run in task_runs.items():
        path = tmp_path / f"{name}.csv"
        tissue = name.startswith("tissue_")
        _write_task(path, fixture.rows[run], tissue=tissue)
        task_paths[name] = path
        location_hashes[name] = _expected_location_hash(
            fixture.rows[run],
            run,
            tissue=tissue,
        )
        for model in ("normal_vae", "so2_vae"):
            view_counts[f"{model}/{name.replace('_', '/')}"] = 1
    output = tmp_path / "store"
    finalizer._publish_output(  # noqa: SLF001
        output_root=output,
        work_paths=fixture.work_paths,
        task_paths=task_paths,
        normal_shards=normal,
        so2_shards=so2,
        artifact_records=records,
        validation={
            "status": "pass",
            "location_file_sha256": location_hashes,
            "logical_view_counts": view_counts,
        },
        input_contract=fixture.input_contract,
    )
    audit_path = output / "spec0021_latent_store_global_audit.json"
    assert {path.name for path in output.iterdir()} == {
        "views",
        audit_path.name,
    }
    assert not tuple(output.rglob("*.bin"))
    audit = _read_json(audit_path)
    assert audit["status"] == "complete"
    runtime = _dict_field(audit, "runtime_environment")
    assert runtime == {
        "python_version": finalizer.platform.python_version(),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
    }
    logical_views = _dict_field(audit, "logical_views")
    assert len(logical_views) == EXPECTED_LOGICAL_VIEWS
    artifacts = _dict_field(audit, "artifacts")
    normal_artifacts = _dict_field(artifacts, "normal_vae")
    assert "sidecar" in _dict_field(normal_artifacts, "1")
    location = output / "views/cancer_train_locations.csv"
    assert location.read_text(encoding="utf-8").splitlines()[0] == ",".join(
        LATENT_LOCATION_HEADER,
    )
    view = LatentTaskView(
        model_name="normal_vae",
        shard_paths=normal,
        work_manifest_paths=fixture.work_paths,
        location_path=location,
        checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256["normal_vae"],
        expected_checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256["normal_vae"],
        expected_work_manifest_hashes=fixture.work_hashes,
        expected_union_sha256=FAKE_UNION_SHA256,
    )
    tensor, metadata = view[0]
    assert float(tensor[0, 0, 0]) == pytest.approx(1.0)
    assert metadata.identity.wsi_id == FIRST_WSI_ID
    del tensor
    view.close()
    assert audit_path.stat().st_mtime_ns >= max(
        path.stat().st_mtime_ns for path in (output / "views").iterdir()
    )


def test_pair_closure_rejects_extra_files_and_forged_artifact_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Contradictory output or a forged pair audit can never certify."""
    fixture = _closure_fixture(tmp_path, monkeypatch)
    extra = fixture.pair_root / "run_01_of_05/dataset/spec0021_incomplete_run_01.json"
    extra.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="allow-list differs"):
        finalizer._pair_paths(fixture.pair_root)  # noqa: SLF001
    extra.unlink()
    normal, so2, audits = finalizer._pair_paths(fixture.pair_root)  # noqa: SLF001
    validated_file_hashes = _validated_file_hashes(normal, so2)
    audit = _read_json(audits[1])
    audit["run_config_sha256"] = "f" * 64
    _write_json(audits[1], audit)
    with pytest.raises(ValueError, match="provenance differs"):
        finalizer._validate_pair_audits(  # noqa: SLF001
            work_paths=fixture.work_paths,
            pair_audits=audits,
            normal_shards=normal,
            so2_shards=so2,
            expected_run_config_hashes=fixture.config_hashes,
            expected_input_receipt_sha256=fixture.input_receipt_sha256,
            validated_file_hashes=validated_file_hashes,
        )
    audit["run_config_sha256"] = fixture.config_hashes[1]
    audit["input_receipt_sha256"] = "f" * 64
    _write_json(audits[1], audit)
    with pytest.raises(ValueError, match="provenance differs"):
        finalizer._validate_pair_audits(  # noqa: SLF001
            work_paths=fixture.work_paths,
            pair_audits=audits,
            normal_shards=normal,
            so2_shards=so2,
            expected_run_config_hashes=fixture.config_hashes,
            expected_input_receipt_sha256=fixture.input_receipt_sha256,
            validated_file_hashes=validated_file_hashes,
        )
    audit["input_receipt_sha256"] = fixture.input_receipt_sha256
    artifacts = _dict_field(audit, "artifacts")
    normal_artifact = _dict_field(artifacts, "normal_vae")
    normal_artifact["sidecar_sha256"] = "0" * 64
    _write_json(audits[1], audit)
    with pytest.raises(ValueError, match="normal_vae identity differs"):
        finalizer._validate_pair_audits(  # noqa: SLF001
            work_paths=fixture.work_paths,
            pair_audits=audits,
            normal_shards=normal,
            so2_shards=so2,
            expected_run_config_hashes=fixture.config_hashes,
            expected_input_receipt_sha256=fixture.input_receipt_sha256,
            validated_file_hashes=validated_file_hashes,
        )
