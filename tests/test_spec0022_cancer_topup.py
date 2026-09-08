# Copyright 2026 HiperMaximus
"""Focused tests for the leakage-free Spec 0022 additive top-up."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import pytest
import torch

from eqvae.cli import build_ubc_cancer_topup as builder
from eqvae.cli import build_ubc_cancer_topup_kernel as kernel_builder
from eqvae.cli import generate_ubc_cancer_topup as worker
from eqvae.data.latent_shards import (
    EXPECTED_CHECKPOINT_SHA256,
    UNION_HEADER,
    LatentRowIdentity,
    LatentShardWriter,
    ModelName,
)
from eqvae.inference.dual_writer import DualLatentWriter, WorkerResumeBinding

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

LOW_AVAILABLE = 999
TARGET_SIZE = 1_000
LARGE_AVAILABLE = 32_596
LARGE_TARGET = 8_149
SESSION_LIMIT_SECONDS = 28_800
type CandidateCsvRow = tuple[int, int, str, int, int, int, str]


@dataclass(frozen=True)
class PlanFixture:
    """All local inputs needed to materialize a small complete plan."""

    candidate: Path
    old: dict[str, Path]
    split: Path
    union: Path
    work: dict[int, Path]
    global_audit: Path
    checkpoints: dict[str, Path]
    checkpoint_hashes: dict[ModelName, str]
    pair_audits: dict[int, Path]
    run_logs: dict[int, Path]
    output: Path


def test_locked_bag_rule_and_independent_pcg64_streams() -> None:
    """Freeze exact per-WSI membership without consulting storage or model outputs."""
    candidates = _candidate_objects(wsi_id=17, count=1_200, split="train")

    under = builder.select_wsi_target(candidates, candidates[:800])
    repeated = builder.select_wsi_target(candidates, candidates[:800])
    over = builder.select_wsi_target(candidates, candidates[:1_100])
    equal = builder.select_wsi_target(candidates, candidates[:1_000])

    assert builder.target_bag_size(LOW_AVAILABLE) == LOW_AVAILABLE
    assert builder.target_bag_size(1_200) == TARGET_SIZE
    assert builder.target_bag_size(LARGE_AVAILABLE) == LARGE_TARGET
    assert under == repeated
    assert len(under) == len(over) == len(equal) == TARGET_SIZE
    assert set(candidates[:800]) < set(under)
    assert set(over) < set(candidates[:1_100])
    assert equal == candidates[:1_000]


def test_plan_materializes_label_free_topup_and_model_independent_bags(
    tmp_path: Path,
) -> None:
    """Keep labels out of inference while resolving frozen bags to physical rows."""
    fixture = _make_plan_fixture(tmp_path)

    audit = _materialize(fixture)

    inference = fixture.output / "inference_bundle"
    logical = fixture.output / "logical"
    manifest = inference / "cancer_topup_manifest.csv"
    with manifest.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    contract = _read_json(inference / "spec0022_topup_inference_contract.json")
    checkpoint_records = cast("dict[str, dict[str, object]]", contract["checkpoints"])
    projection = cast("dict[str, object]", contract["deadline_projection"])
    assert tuple(reader.fieldnames or ()) == builder.SUPPLEMENT_HEADER
    assert rows
    assert not ({"diagnosis_label", "diagnosis_index", "split", "mask"} & set(rows[0]))
    assert projection["fits"] is True
    assert cast("int", projection["projected_total_seconds"]) <= SESSION_LIMIT_SECONDS
    assert not any(path.is_dir() for path in inference.iterdir())
    assert checkpoint_records["normal_vae"]["path"] == "normal_vae_step_060000.pt"
    assert checkpoint_records["so2_vae"]["path"] == "so2_vae_step_060000.pt"
    assert cast("dict[str, object]", audit["acceptance"]) == {
        "base_global_audit_complete": True,
        "deadline_projection_fits": True,
        "inference_manifest_label_free": True,
        "model_independent_logical_files": True,
        "saved_output_fits": True,
        "supplement_disjoint_from_base": True,
        "target_frozen_before_union_resolution": True,
    }
    for split in builder.SPLITS:
        instances = _read_csv(logical / f"wsi_cancer_{split}_instances.csv")
        bags = _read_csv(logical / f"wsi_cancer_{split}_bags.csv")
        assert len(instances) == TARGET_SIZE
        assert len(bags) == 1
        assert bags[0]["instance_start"] == "0"
        assert bags[0]["instance_count"] == "1000"
        assert {row["store_source"] for row in instances} <= {"base", "topup"}
        assert not any("model" in key for key in instances[0])


def test_target_membership_is_unchanged_when_base_union_changes(tmp_path: Path) -> None:
    """Make physical reuse affect only T-minus-B, never scientific membership."""
    first = _make_plan_fixture(tmp_path / "first", base_stride=7)
    second = _make_plan_fixture(tmp_path / "second", base_stride=11)

    _materialize(first)
    _materialize(second)

    first_logical = _logical_identities(first.output)
    second_logical = _logical_identities(second.output)
    first_manifest = _read_csv(
        first.output / "inference_bundle/cancer_topup_manifest.csv",
    )
    second_manifest = _read_csv(
        second.output / "inference_bundle/cancer_topup_manifest.csv",
    )
    assert first_logical == second_logical
    assert len(first_manifest) != len(second_manifest)


def test_incomplete_spec0021_global_audit_blocks_plan_publication(
    tmp_path: Path,
) -> None:
    """Refuse derived artifacts before all ten base stores are accepted."""
    fixture = _make_plan_fixture(tmp_path)
    payload = _read_json(fixture.global_audit)
    payload["status"] = "incomplete"
    fixture.global_audit.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="global audit"):
        _materialize(fixture)

    assert not fixture.output.exists()


def test_mismatched_pair_cannot_publish_final_logical_audit(tmp_path: Path) -> None:
    """Require both top-up artifacts to match before logical completion."""
    plan_root, pair_path = _make_completed_topup(tmp_path)
    pair = _read_json(pair_path)
    pair["row_count"] = cast("int", pair["row_count"]) + 1
    pair_path.write_text(json.dumps(pair), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match"):
        builder.finalize_logical_audit(
            plan_root=plan_root,
            topup_pair_audit_path=pair_path,
        )

    assert not (plan_root / "spec0022_logical_dataset_audit.json").exists()


def test_final_logical_audit_is_written_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leave no completion claim when the last atomic logical-audit write fails."""
    plan_root, pair_path = _make_completed_topup(tmp_path)

    def fail_write(_path: Path, _payload: object) -> None:
        message = "injected final-audit failure"
        raise OSError(message)

    monkeypatch.setattr(builder, "_write_json_exclusive", fail_write)
    with pytest.raises(OSError, match="injected"):
        builder.finalize_logical_audit(
            plan_root=plan_root,
            topup_pair_audit_path=pair_path,
        )

    assert not (plan_root / "spec0022_logical_dataset_audit.json").exists()


def test_final_logical_audit_rejects_post_plan_bag_mutation(tmp_path: Path) -> None:
    """Reread all logical CSVs so a post-plan split or bag edit cannot be sealed."""
    plan_root, pair_path = _make_completed_topup(tmp_path)
    bag_path = plan_root / "logical/wsi_cancer_train_bags.csv"
    bag_path.write_text(
        "bag_row,wsi_id,diagnosis_label,diagnosis_index,split,instance_start,instance_count\n"
        "0,10,HGSC,2,train,0,1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"bag|file record"):
        builder.finalize_logical_audit(
            plan_root=plan_root,
            topup_pair_audit_path=pair_path,
        )

    assert not (plan_root / "spec0022_logical_dataset_audit.json").exists()


def test_compact_remote_metadata_seals_logical_audit_without_binaries(
    tmp_path: Path,
) -> None:
    """Accept the remote writer's final JSONs without downloading latent payloads."""
    plan_root, pair_path = _make_completed_topup(tmp_path)
    contract = plan_root / "inference_bundle/spec0022_topup_inference_contract.json"
    pair = _read_json(pair_path)
    pair["input_contract_sha256"] = _sha256(contract)
    pair["base_union_sha256"] = cast(
        "str",
        _read_json(contract)["base_union_sha256"],
    )
    pair["completed_wsi_evidence"] = [{"wsi_id": 10}]
    pair_path.write_text(json.dumps(pair), encoding="utf-8")
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    for path in pair_path.parent.glob("*.json"):
        if path.name != "worker.resume.json":
            shutil.copyfile(path, metadata / path.name)

    audit = builder.finalize_logical_audit_from_metadata(
        plan_root=plan_root,
        topup_metadata_root=metadata,
    )

    assert audit["status"] == "complete"
    assert cast("dict[str, object]", audit["acceptance"]) == {
        "latent_binaries_remain_remote": True,
        "remote_pair_audit_complete_and_aligned": True,
        "same_logical_files_for_both_models": True,
    }
    assert not any(metadata.glob("*.bin"))


def test_worker_rejects_any_labelled_or_extra_mounted_input(tmp_path: Path) -> None:
    """Prevent labels from entering the GPU job even as an unreferenced extra file."""
    root = tmp_path / "input"
    root.mkdir()
    for relative in worker.INPUT_ALLOWLIST:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    (root / "cancer_topup_manifest.csv").write_text(
        "atlas_row_index,wsi_id,x,y\n0,1,0,0\n",
        encoding="utf-8",
    )
    (root / "cancer_train.csv").write_text("diagnosis_label\nHGSC\n", encoding="utf-8")

    with pytest.raises(ValueError, match="allow-list"):
        worker._validate_input_tree(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
            root,
            root / "spec0022_topup_inference_contract.json",
        )


def test_pair_publication_renames_one_complete_directory_atomically(
    tmp_path: Path,
) -> None:
    """Expose the audit and four data files together, never one file at a time."""
    scratch = tmp_path / "scratch"
    output = tmp_path / "dataset"
    scratch.mkdir()
    for name in worker.OUTPUT_ALLOWLIST:
        (scratch / name).write_text(name, encoding="utf-8")

    worker._publish_complete_output(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        scratch,
        output,
    )

    assert not scratch.exists()
    assert {path.name for path in output.iterdir()} == set(worker.OUTPUT_ALLOWLIST)


def test_coordinate_only_writer_preserves_row_identity_and_pair_audit_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use four columns and omit pair completion when the final audit fails."""
    manifest = tmp_path / "cancer_topup_manifest.csv"
    manifest.write_text(
        "atlas_row_index,wsi_id,x,y\n0,10,0,0\n1,10,256,0\n",
        encoding="utf-8",
    )
    manifest_hash = _sha256(manifest)
    normal = _coordinate_writer(tmp_path, manifest, manifest_hash, "normal_vae")
    so2 = _coordinate_writer(tmp_path, manifest, manifest_hash, "so2_vae")
    dual = DualLatentWriter(
        normal_writer=normal,
        so2_writer=so2,
        state_path=tmp_path / "worker.resume.json",
        binding=WorkerResumeBinding(
            input_bundle_sha256="d" * 64,
            run_config_sha256="e" * 64,
            work_manifest_sha256=manifest_hash,
            run_number=1,
        ),
    )
    identities = tuple(row.identity for row in normal.manifest.rows)
    tensors = torch.zeros((2, 16, 32, 32), dtype=torch.float32)

    with pytest.raises(ValueError, match="identities do not match"):
        normal.append_batch(
            row_start=0,
            identities=tuple(reversed(identities)),
            tensors=tensors,
        )
    normal.append_batch(row_start=0, identities=identities, tensors=tensors)
    so2.append_batch(row_start=0, identities=identities, tensors=tensors)
    normal.finalize()
    so2.finalize()
    assert normal.manifest.rows == so2.manifest.rows

    audit = tmp_path / "spec0022_cancer_topup_pair_audit.json"

    def fail_atomic(_path: Path, _payload: object) -> None:
        message = "injected pair-audit failure"
        raise OSError(message)

    monkeypatch.setattr(worker, "_atomic_write_json", fail_atomic)
    with pytest.raises(OSError, match="injected"):
        worker._finalize_pair(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
            writers=dual,
            audit_path=audit,
            manifest_sha256=manifest_hash,
            config_sha256="e" * 64,
        )
    assert not audit.exists()
    assert (tmp_path / "normal_vae_mu_cancer_topup.bin").is_file()
    assert (tmp_path / "so2_vae_mu_cancer_topup.bin").is_file()


def test_kernel_build_binds_one_private_dual_t4_job_and_one_input_dataset(
    tmp_path: Path,
) -> None:
    """Build only the exact one-off GPU upload after a versioned input receipt."""
    fixture = _make_plan_fixture(tmp_path / "fixture")
    _materialize(fixture)
    contract = (
        fixture.output / "inference_bundle/spec0022_topup_inference_contract.json"
    )
    plan = _read_json(fixture.output / "spec0022_cancer_topup_plan_audit.json")
    receipt = tmp_path / "receipt.json"
    output = tmp_path / "kernel"
    receipt_payload = {
        "schema_version": kernel_builder.RECEIPT_SCHEMA,
        "status": "verified",
        "visibility": "private",
        "dataset_reference": kernel_builder.INPUT_DATASET_ID,
        "dataset_version": 1,
        "input_contract_sha256": _sha256(contract),
        "files": {
            name: record
            for name, record in cast(
                "dict[str, dict[str, object]]",
                plan["inference_bundle_files"],
            ).items()
            if name != "dataset-metadata.json"
        },
    }
    receipt.write_text(
        f"{json.dumps(receipt_payload, sort_keys=True, separators=(',', ':'))}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="receipt"):
        kernel_builder.build_kernel(
            repo_root=builder.SPEC_PATH.resolve().parents[2],
            plan_root=fixture.output,
            receipt_path=receipt,
            output_root=output,
        )
    assert not output.exists()

    receipt_payload.update({
        "remote_listing_sha256": "a" * 64,
        "remote_status_sha256": "b" * 64,
    })
    receipt.write_text(
        f"{json.dumps(receipt_payload, sort_keys=True, separators=(',', ':'))}\n",
        encoding="utf-8",
    )

    config = kernel_builder.build_kernel(
        repo_root=builder.SPEC_PATH.resolve().parents[2],
        plan_root=fixture.output,
        receipt_path=receipt,
        output_root=output,
    )

    metadata = _read_json(output / "kernel-metadata.json")
    assert {path.name for path in output.iterdir()} == kernel_builder.UPLOAD_FILES
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["dataset_sources"] == [kernel_builder.INPUT_DATASET_ID]
    assert metadata["competition_sources"] == ["UBC-OCEAN"]
    assert cast("dict[str, object]", config["deadline_projection"])["fits"] is True
    assert "KAGGLE_UBC_OCEAN_CANCER_TOPUP_READY = True" in (
        output / "run.py"
    ).read_text(encoding="utf-8")
    with (output / "run.py").open("a", encoding="utf-8") as handle:
        handle.write("# stale wrapper mutation\n")
    with pytest.raises(ValueError, match="generated bytes"):
        kernel_builder.validate_kernel(
            repo_root=builder.SPEC_PATH.resolve().parents[2],
            plan_root=fixture.output,
            receipt_path=receipt,
            output_root=output,
        )


def _materialize(fixture: PlanFixture) -> dict[str, object]:
    return builder.materialize_cancer_topup(
        candidate_path=fixture.candidate,
        old_cancer_paths=fixture.old,
        split_path=fixture.split,
        union_path=fixture.union,
        work_paths=fixture.work,
        global_audit_path=fixture.global_audit,
        normal_checkpoint_path=fixture.checkpoints["normal_vae"],
        so2_checkpoint_path=fixture.checkpoints["so2_vae"],
        output_root=fixture.output,
        expected_candidate_sha256=None,
        expected_old_sha256=None,
        expected_split_sha256=None,
        expected_union_sha256=None,
        expected_work_sha256=None,
        expected_checkpoint_sha256=fixture.checkpoint_hashes,
        pair_audit_paths=fixture.pair_audits,
        run_log_paths=fixture.run_logs,
        enforce_real_counts=False,
    )


def _make_plan_fixture(  # noqa: PLR0914
    root: Path,
    *,
    base_stride: int = 7,
) -> PlanFixture:
    root.mkdir(parents=True, exist_ok=True)
    split_rows = (
        (10, "CC", 0, "train"),
        (20, "EC", 1, "validation"),
        (30, "HGSC", 2, "test"),
    )
    split = root / "split.csv"
    _write_csv(
        split,
        (
            "wsi_id",
            "diagnosis_label",
            "diagnosis_index",
            "is_updated_image_id",
            "split",
        ),
        (
            (wsi, label, index, "false", split_name)
            for wsi, label, index, split_name in split_rows
        ),
    )
    candidate = root / "candidates.csv"
    candidate_rows: list[CandidateCsvRow] = []
    objects: dict[int, tuple[builder.CandidateRow, ...]] = {}
    atlas = 0
    for wsi, label, diagnosis_index, split_name in split_rows:
        group: list[builder.CandidateRow] = []
        for local in range(1_200):
            identity = LatentRowIdentity(atlas, wsi, local * 256, 0)
            group.append(
                builder.CandidateRow(identity, label, diagnosis_index, split_name),
            )
            candidate_rows.append((
                atlas,
                wsi,
                label,
                diagnosis_index,
                local * 256,
                0,
                split_name,
            ))
            atlas += 1
        objects[wsi] = tuple(group)
    _write_csv(candidate, builder.CANCER_HEADER, iter(candidate_rows))
    old_counts = {"train": 800, "validation": 1_000, "test": 1_100}
    old: dict[str, Path] = {}
    targets: list[builder.CandidateRow] = []
    for wsi, _label, _diagnosis_index, split_name in split_rows:
        path = root / f"cancer_{split_name}.csv"
        selected = objects[wsi][: old_counts[split_name]]
        _write_csv(
            path,
            builder.CANCER_HEADER,
            (_cancer_values(row) for row in selected),
        )
        old[split_name] = path
        targets.extend(builder.select_wsi_target(objects[wsi], selected))
    target_ids = {row.identity for row in targets}
    base_rows = [
        row
        for index, row in enumerate(candidate_rows)
        if LatentRowIdentity(int(row[0]), int(row[1]), int(row[4]), int(row[5]))
        in target_ids
        and index % base_stride == 0
    ]
    union = root / "union.csv"
    union_values = [_union_values(row) for row in base_rows]
    _write_csv(union, UNION_HEADER, iter(union_values))
    work: dict[int, Path] = {}
    chunk = (len(union_values) + 4) // 5
    for run in range(1, 6):
        path = root / f"run_{run:02d}_of_05.csv"
        start = (run - 1) * chunk
        _write_csv(path, UNION_HEADER, iter(union_values[start : start + chunk]))
        work[run] = path
    union_hash = _sha256(union)
    global_audit = root / "global.json"
    global_audit.write_text(
        json.dumps({
            "schema_version": "spec0021.latent_store_global_audit.v1",
            "status": "complete",
            "validation": {"union_sha256": union_hash},
            "artifacts": {"normal_vae": {}, "so2_vae": {}},
        }),
        encoding="utf-8",
    )
    checkpoints = {
        "normal_vae": root / "normal.pt",
        "so2_vae": root / "so2.pt",
    }
    checkpoints["normal_vae"].write_bytes(b"normal checkpoint")
    checkpoints["so2_vae"].write_bytes(b"so2 checkpoint")
    checkpoint_hashes: dict[ModelName, str] = {
        "normal_vae": _sha256(checkpoints["normal_vae"]),
        "so2_vae": _sha256(checkpoints["so2_vae"]),
    }
    pair_audits: dict[int, Path] = {}
    run_logs: dict[int, Path] = {}
    for run in range(1, 6):
        pair = root / f"pair_{run}.json"
        pair.write_text(
            json.dumps({
                "schema_version": "spec0021.latent_pair_audit.v1",
                "status": "complete",
                "run_number": run,
                "row_count": 1_000,
                "completed_wsi_evidence": [{"wsi_id": run}],
            }),
            encoding="utf-8",
        )
        log = root / f"run_{run}.log"
        log.write_text(json.dumps([{"time": 100.0 + run}]), encoding="utf-8")
        pair_audits[run] = pair
        run_logs[run] = log
    return PlanFixture(
        candidate,
        old,
        split,
        union,
        work,
        global_audit,
        checkpoints,
        checkpoint_hashes,
        pair_audits,
        run_logs,
        root / "plan",
    )


def _candidate_objects(
    *,
    wsi_id: int,
    count: int,
    split: str,
) -> tuple[builder.CandidateRow, ...]:
    return tuple(
        builder.CandidateRow(
            LatentRowIdentity(index, wsi_id, index * 256, 0),
            "HGSC",
            2,
            split,
        )
        for index in range(count)
    )


def _cancer_values(row: builder.CandidateRow) -> tuple[object, ...]:
    identity = row.identity
    return (
        identity.atlas_row_index,
        identity.wsi_id,
        row.diagnosis_label,
        row.diagnosis_index,
        identity.x,
        identity.y,
        row.split,
    )


def _union_values(row: CandidateCsvRow) -> tuple[object, ...]:
    atlas, wsi, label, diagnosis_index, x, y, split = row
    return (atlas, wsi, label, diagnosis_index, x, y, split, "true", "false", "")


def _logical_identities(root: Path) -> dict[str, tuple[tuple[str, ...], ...]]:
    return {
        split: tuple(
            (row["atlas_row_index"], row["wsi_id"], row["x"], row["y"])
            for row in _read_csv(root / f"logical/wsi_cancer_{split}_instances.csv")
        )
        for split in builder.SPLITS
    }


def _make_completed_topup(  # noqa: PLR0914
    root: Path,
) -> tuple[Path, Path]:
    plan_root = root / "plan"
    inference = plan_root / "inference_bundle"
    logical = plan_root / "logical"
    pair_root = root / "pair"
    inference.mkdir(parents=True)
    logical.mkdir()
    pair_root.mkdir()
    manifest = inference / "cancer_topup_manifest.csv"
    manifest.write_text(
        "atlas_row_index,wsi_id,x,y\n0,10,0,0\n1,10,256,0\n",
        encoding="utf-8",
    )
    manifest_hash = _sha256(manifest)
    normal = _coordinate_writer(pair_root, manifest, manifest_hash, "normal_vae")
    so2 = _coordinate_writer(pair_root, manifest, manifest_hash, "so2_vae")
    dual = DualLatentWriter(
        normal_writer=normal,
        so2_writer=so2,
        state_path=pair_root / "worker.resume.json",
        binding=WorkerResumeBinding(
            input_bundle_sha256="d" * 64,
            run_config_sha256="e" * 64,
            work_manifest_sha256=manifest_hash,
            run_number=1,
        ),
    )
    identities = tuple(row.identity for row in normal.manifest.rows)
    tensors = torch.zeros((2, 16, 32, 32), dtype=torch.float32)
    normal.append_batch(row_start=0, identities=identities, tensors=tensors)
    so2.append_batch(row_start=0, identities=identities, tensors=tensors)
    pair_path = pair_root / "spec0022_cancer_topup_pair_audit.json"
    worker._finalize_pair(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        writers=dual,
        audit_path=pair_path,
        manifest_sha256=manifest_hash,
        config_sha256="e" * 64,
    )
    (pair_root / "worker.resume.json").unlink()
    contract: dict[str, object] = {
        "schema_version": "spec0022.cancer_topup_inference_input.v1",
        "status": "complete",
        "base_union_sha256": "c" * 64,
        "supplement_manifest": {
            "path": "cancer_topup_manifest.csv",
            "sha256": manifest_hash,
            "row_count": 2,
        },
        "checkpoints": {
            "normal_vae": {"sha256": EXPECTED_CHECKPOINT_SHA256["normal_vae"]},
            "so2_vae": {"sha256": EXPECTED_CHECKPOINT_SHA256["so2_vae"]},
        },
    }
    (inference / "spec0022_topup_inference_contract.json").write_text(
        json.dumps(contract),
        encoding="utf-8",
    )
    logical_records: dict[str, dict[str, object]] = {}
    for split in builder.SPLITS:
        instance_path = logical / f"wsi_cancer_{split}_instances.csv"
        bag_path = logical / f"wsi_cancer_{split}_bags.csv"
        instance_rows: tuple[tuple[object, ...], ...] = ()
        bag_rows: tuple[tuple[object, ...], ...] = ()
        if split == "train":
            instance_rows = (
                (0, 0, 10, 0, 0, "HGSC", 2, "train", "topup", 1, 0),
                (1, 1, 10, 256, 0, "HGSC", 2, "train", "topup", 1, 1),
            )
            bag_rows = ((0, 10, "HGSC", 2, "train", 0, 2),)
        _write_csv(instance_path, builder.INSTANCE_HEADER, iter(instance_rows))
        _write_csv(bag_path, builder.BAG_HEADER, iter(bag_rows))
        for path, count in (
            (instance_path, len(instance_rows)),
            (bag_path, len(bag_rows)),
        ):
            logical_records[path.name] = {
                "path": path.name,
                "row_count": count,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
    (plan_root / "spec0022_cancer_topup_plan_audit.json").write_text(
        json.dumps({
            "schema_version": "spec0022.cancer_topup_plan_audit.v1",
            "status": "complete",
            "inputs": {"base_global_audit": "f" * 64},
            "logical_files": logical_records,
            "coverage_by_wsi": [
                {
                    "wsi_id": 10,
                    "split": "train",
                    "candidate_count": 2,
                    "target_count": 2,
                },
            ],
            "selection": {
                "target_counts": {"train": 2, "validation": 0, "test": 0},
                "target_total": 2,
            },
        }),
        encoding="utf-8",
    )
    (logical / "spec0022_logical_dataset_contract.json").write_text(
        json.dumps({
            "schema_version": "spec0022.cancer_logical_dataset.v1",
            "status": "planned",
            "files": logical_records,
        }),
        encoding="utf-8",
    )
    return plan_root, pair_path


def _coordinate_writer(
    root: Path,
    manifest: Path,
    manifest_hash: str,
    model: str,
) -> LatentShardWriter:
    typed = cast("ModelName", model)
    checkpoint = EXPECTED_CHECKPOINT_SHA256[typed]
    return LatentShardWriter(
        bin_path=root / f"{model}_mu_cancer_topup.bin",
        manifest_path=manifest,
        run_number=1,
        model_name=typed,
        checkpoint_sha256=checkpoint,
        expected_checkpoint_sha256=checkpoint,
        expected_manifest_sha256=manifest_hash,
        expected_union_sha256="c" * 64,
        coordinate_only_manifest=True,
    )


def _write_csv(
    path: Path,
    header: tuple[str, ...],
    rows: Iterable[Sequence[object]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, object]:
    value = cast("object", json.loads(path.read_text(encoding="utf-8")))
    assert isinstance(value, dict)
    return cast("dict[str, object]", value)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
