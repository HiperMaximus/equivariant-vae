# Copyright 2026 HiperMaximus
"""Focused local-fixture tests for the Spec 0021 inference core."""

from __future__ import annotations

import csv
import hashlib
import json
import struct
from dataclasses import replace
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
import torch
from torch import nn

from eqvae.data.latent_shards import (
    EXPECTED_CHECKPOINT_SHA256,
    LATENT_SHARD_HEADER_SIZE,
    LatentRowIdentity,
    LatentShardWriter,
    load_work_manifest,
)
from eqvae.data.wsi_batches import (
    ArrayWSIReader,
    WSIEvidence,
    iter_wsi_patch_batches,
    iter_wsi_patch_interval,
    read_wsi_patch,
)
from eqvae.inference import checkpoints
from eqvae.inference.checkpoints import FrozenCheckpointSpec, load_frozen_checkpoint
from eqvae.inference.dual_writer import (
    CatchUpPlan,
    DualLatentWriter,
    VerificationPlan,
    WorkerResumeBinding,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from numpy.typing import NDArray

_UNION_HEADER = (
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
_FAKE_UNION_SHA256 = "a" * 64
_INJECTED_FAILURE = "injected second-writer failure"


class _TinyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))


def test_frozen_loader_hashes_first_and_freezes_valid_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject bytes before deserialization, then strictly load and freeze."""
    bad_path = tmp_path / "bad.pt"
    bad_path.write_bytes(b"not a checkpoint")
    called = False
    original_load = torch.load

    def reject_load(*_args: object, **_kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError

    monkeypatch.setattr(torch, "load", reject_load)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_frozen_checkpoint(bad_path, model_name="normal_vae")
    assert called is False
    monkeypatch.setattr(torch, "load", original_load)

    model = _TinyEncoder()
    payload: dict[str, object] = {
        "schema_version": checkpoints.CHECKPOINT_SCHEMA_VERSION,
        "optimizer_step": checkpoints.FINAL_OPTIMIZER_STEP,
        "successful_optimizer_update_count": checkpoints.FINAL_OPTIMIZER_STEP,
        "model_state_dict": model.state_dict(),
    }
    good_path = tmp_path / "good.pt"
    torch.save(payload, good_path)
    digest = _sha256(good_path)
    spec = FrozenCheckpointSpec("normal_vae", "fixture", digest, 1)
    monkeypatch.setattr(
        checkpoints,
        "FROZEN_CHECKPOINT_SPECS",
        {**checkpoints.FROZEN_CHECKPOINT_SPECS, "normal_vae": spec},
    )
    monkeypatch.setattr(checkpoints, "build_model", _build_tiny)

    loaded = load_frozen_checkpoint(good_path, model_name="normal_vae")

    assert loaded.training is False
    assert all(not parameter.requires_grad for parameter in loaded.parameters())


def test_manifest_array_stream_preserves_sparse_rgb_identity_and_transcript(
    tmp_path: Path,
) -> None:
    """Prove sparse strip offsets, CHW order, batching, and transcript bytes."""
    rows = (
        _row(10, 7, x=0, y=0),
        _row(11, 7, x=512, y=0),
        _row(20, 9, x=256, y=256),
    )
    manifest_path = _write_manifest(tmp_path, rows)
    manifest = load_work_manifest(
        manifest_path,
        run_number=1,
        expected_sha256=_sha256(manifest_path),
    )
    images = {7: _coordinate_image(768, 256), 9: _coordinate_image(512, 512)}
    for wsi_id in images:
        (tmp_path / f"{wsi_id}.png").write_bytes(f"png-{wsi_id}".encode())
    reader = ArrayWSIReader(images)

    batches = tuple(
        iter_wsi_patch_batches(
            manifest=manifest,
            wsi_dir=tmp_path,
            batch_size=1,
            reader=reader,
        ),
    )

    assert [batch.row_start for batch in batches] == [0, 1, 2]
    assert [batch.identities[0] for batch in batches] == [
        row.identity for row in manifest.rows
    ]
    assert all(batch.images_uint8.dtype == torch.uint8 for batch in batches)
    for batch in batches:
        identity = batch.identities[0]
        expected_array = np.ascontiguousarray(
            images[identity.wsi_id][
                identity.y : identity.y + 256,
                identity.x : identity.x + 256,
            ].transpose(2, 0, 1),
        )
        assert np.array_equal(batch.images_uint8[0].numpy(), expected_array)
        assert torch.equal(
            batch.images_uint8[0],
            read_wsi_patch(identity=identity, wsi_dir=tmp_path, reader=reader),
        )
    assert batches[0].final_wsi_evidence is None
    first_evidence = batches[1].final_wsi_evidence
    assert first_evidence is not None
    assert first_evidence.transcript_sha256 == _transcript(
        tuple(row.identity for row in manifest.rows[:2]),
        images[7],
    )
    assert batches[2].final_wsi_evidence is not None

    with pytest.raises(ValueError, match="complete WSI boundaries"):
        tuple(
            iter_wsi_patch_batches(
                manifest=manifest,
                wsi_dir=tmp_path,
                batch_size=1,
                start_row=1,
                reader=reader,
            ),
        )

    smoke_batches = tuple(
        iter_wsi_patch_interval(
            manifest=manifest,
            wsi_dir=tmp_path,
            batch_size=1,
            start_row=1,
            stop_row=2,
            reader=reader,
        ),
    )
    assert len(smoke_batches) == 1
    assert smoke_batches[0].row_start == 1
    assert smoke_batches[0].identities == (manifest.rows[1].identity,)
    assert smoke_batches[0].final_wsi_evidence is None

    with pytest.raises(ValueError, match="within one WSI"):
        tuple(
            iter_wsi_patch_interval(
                manifest=manifest,
                wsi_dir=tmp_path,
                batch_size=2,
                start_row=1,
                stop_row=3,
                reader=reader,
            ),
        )


def test_dual_writer_journal_recovers_one_writer_ahead(  # noqa: PLR0914
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Journal evidence before commit and catch up only the lagging encoder."""
    rows = (_row(0, 7, x=0, y=0), _row(1, 9, x=0, y=0))
    manifest_path = _write_manifest(tmp_path, rows)
    manifest_sha256 = _sha256(manifest_path)
    normal, so2 = _writers(tmp_path, manifest_path, manifest_sha256)
    state_path = tmp_path / "worker.resume.json"
    binding = WorkerResumeBinding("b" * 64, "c" * 64, manifest_sha256, 1)
    dual = DualLatentWriter(
        normal_writer=normal,
        so2_writer=so2,
        state_path=state_path,
        binding=binding,
    )
    identities = (LatentRowIdentity(0, 7, 0, 0),)
    evidence = WSIEvidence(7, 5, "d" * 64, "e" * 64)
    tensors = torch.zeros((1, 16, 32, 32), dtype=torch.float32)

    def fail_before_so2_commit(**_kwargs: object) -> None:
        raise RuntimeError(_INJECTED_FAILURE)

    monkeypatch.setattr(so2, "append_batch", fail_before_so2_commit)
    with pytest.raises(RuntimeError, match="injected"):
        dual.append_lockstep(
            row_start=0,
            identities=identities,
            normal_tensors=tensors,
            so2_tensors=tensors,
            final_wsi_evidence=evidence,
        )
    journal = _json(state_path)
    assert cast("Mapping[str, object]", journal["active_wsi"])["row_end"] == 1
    normal.close()
    so2.close()
    monkeypatch.undo()

    resumed_normal, resumed_so2 = _writers(
        tmp_path,
        manifest_path,
        manifest_sha256,
    )
    resumed = DualLatentWriter(
        normal_writer=resumed_normal,
        so2_writer=resumed_so2,
        state_path=state_path,
        binding=binding,
    )
    plan = resumed.reconcile()
    assert isinstance(plan, CatchUpPlan)
    assert plan.model_name == "so2_vae"
    with pytest.raises(ValueError, match="evidence mismatch"):
        resumed.append_catch_up(
            row_start=0,
            identities=identities,
            tensors=tensors,
            final_wsi_evidence=replace(evidence, png_bytes=6),
        )
    resumed.append_catch_up(
        row_start=0,
        identities=identities,
        tensors=tensors,
        final_wsi_evidence=evidence,
    )

    assert resumed_normal.committed_rows == resumed_so2.committed_rows == 1
    converged = _json(state_path)
    assert converged["active_wsi"] is None
    assert len(cast("Sequence[object]", converged["completed_wsi_evidence"])) == 1
    resumed_normal.close()
    resumed_so2.close()


def test_equal_active_end_requires_reread_and_rejects_tampered_wsi_ids(  # noqa: PLR0914
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never auto-promote equal committed prefixes or trust journal-only IDs."""
    rows = (_row(0, 7, x=0, y=0), _row(1, 9, x=0, y=0))
    manifest_path = _write_manifest(tmp_path, rows)
    manifest_sha256 = _sha256(manifest_path)
    normal, so2 = _writers(tmp_path, manifest_path, manifest_sha256)
    state_path = tmp_path / "worker.resume.json"
    binding = WorkerResumeBinding("b" * 64, "c" * 64, manifest_sha256, 1)
    dual = DualLatentWriter(
        normal_writer=normal,
        so2_writer=so2,
        state_path=state_path,
        binding=binding,
    )
    evidence = WSIEvidence(7, 5, "d" * 64, "e" * 64)
    tensors = torch.zeros((1, 16, 32, 32), dtype=torch.float32)

    def fail_before_journal_promotion() -> None:
        raise RuntimeError(_INJECTED_FAILURE)

    monkeypatch.setattr(dual, "_promote_active", fail_before_journal_promotion)
    with pytest.raises(RuntimeError, match="injected"):
        dual.append_lockstep(
            row_start=0,
            identities=(LatentRowIdentity(0, 7, 0, 0),),
            normal_tensors=tensors,
            so2_tensors=tensors,
            final_wsi_evidence=evidence,
        )
    normal.close()
    so2.close()
    monkeypatch.undo()

    resumed_normal, resumed_so2 = _writers(
        tmp_path,
        manifest_path,
        manifest_sha256,
    )
    resumed = DualLatentWriter(
        normal_writer=resumed_normal,
        so2_writer=resumed_so2,
        state_path=state_path,
        binding=binding,
    )
    plan = resumed.reconcile()
    assert isinstance(plan, VerificationPlan)
    with pytest.raises(ValueError, match="reread evidence mismatch"):
        resumed.verify_converged_active(replace(evidence, png_bytes=6))
    assert _json(state_path)["active_wsi"] is not None
    resumed.verify_converged_active(evidence)
    assert _json(state_path)["active_wsi"] is None
    resumed_normal.close()
    resumed_so2.close()

    state = _json(state_path)
    for key in ("normal_prefix", "so2_prefix"):
        prefix = cast("dict[str, object]", state[key])
        prefix["completed_wsi_ids"] = [999]
    completed = cast("list[dict[str, object]]", state["completed_wsi_evidence"])
    completed[0]["wsi_id"] = 999
    state_path.write_text(json.dumps(state), encoding="utf-8")
    tampered_normal, tampered_so2 = _writers(
        tmp_path,
        manifest_path,
        manifest_sha256,
    )
    tampered = DualLatentWriter(
        normal_writer=tampered_normal,
        so2_writer=tampered_so2,
        state_path=state_path,
        binding=binding,
    )
    with pytest.raises(ValueError, match="rows/WSI IDs"):
        tampered.reconcile()
    tampered_normal.close()
    tampered_so2.close()


def test_incomplete_marker_rolls_back_and_rescans_active_wsi(
    tmp_path: Path,
) -> None:
    """Publish incomplete only after both partials match the durable boundary."""
    rows = (
        _row(0, 7, x=0, y=0),
        _row(1, 7, x=256, y=0),
        _row(2, 9, x=0, y=0),
    )
    manifest_path = _write_manifest(tmp_path, rows)
    manifest_sha256 = _sha256(manifest_path)
    normal, so2 = _writers(tmp_path, manifest_path, manifest_sha256)
    state_path = tmp_path / "worker.resume.json"
    dual = DualLatentWriter(
        normal_writer=normal,
        so2_writer=so2,
        state_path=state_path,
        binding=WorkerResumeBinding("b" * 64, "c" * 64, manifest_sha256, 1),
    )
    tensors = torch.zeros((1, 16, 32, 32), dtype=torch.float32)
    dual.append_lockstep(
        row_start=0,
        identities=(LatentRowIdentity(0, 7, 0, 0),),
        normal_tensors=tensors,
        so2_tensors=tensors,
    )

    marker = tmp_path / "incomplete.json"
    dual.publish_incomplete(marker)

    assert normal.next_row_start == so2.next_row_start == 0
    assert normal.partial_path.stat().st_size == LATENT_SHARD_HEADER_SIZE
    assert so2.partial_path.stat().st_size == LATENT_SHARD_HEADER_SIZE
    assert _json(marker)["status"] == "incomplete"
    normal.close()
    so2.close()
    reopened_normal, reopened_so2 = _writers(
        tmp_path,
        manifest_path,
        manifest_sha256,
    )
    assert reopened_normal.next_row_start == reopened_so2.next_row_start == 0
    reopened_normal.close()
    reopened_so2.close()


def _writers(
    root: Path,
    manifest_path: Path,
    manifest_sha256: str,
) -> tuple[LatentShardWriter, LatentShardWriter]:
    def make(model_name: str, filename: str) -> LatentShardWriter:
        typed_name = cast("checkpoints.FrozenModelName", model_name)
        checkpoint_sha256 = EXPECTED_CHECKPOINT_SHA256[typed_name]
        return LatentShardWriter(
            bin_path=root / filename,
            manifest_path=manifest_path,
            run_number=1,
            model_name=typed_name,
            checkpoint_sha256=checkpoint_sha256,
            expected_checkpoint_sha256=checkpoint_sha256,
            expected_manifest_sha256=manifest_sha256,
            expected_union_sha256=_FAKE_UNION_SHA256,
        )

    return make("normal_vae", "normal.bin"), make("so2_vae", "so2.bin")


def _build_tiny(_kind: str) -> _TinyEncoder:
    return _TinyEncoder()


def _write_manifest(root: Path, rows: tuple[dict[str, str], ...]) -> Path:
    path = root / "run_01_of_05.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(_UNION_HEADER)
        for row in rows:
            writer.writerow([row[column] for column in _UNION_HEADER])
    return path


def _row(index: int, wsi_id: int, *, x: int, y: int) -> dict[str, str]:
    return {
        "atlas_row_index": str(index),
        "wsi_id": str(wsi_id),
        "diagnosis_label": "HGSC",
        "diagnosis_index": "0",
        "x": str(x),
        "y": str(y),
        "split": "train",
        "cancer_ae_selected": "true",
        "tissue_selected": "false",
        "tissue_label": "",
    }


def _coordinate_image(width: int, height: int) -> NDArray[np.uint8]:
    y_values, x_values = np.indices((height, width), dtype=np.uint16)
    return np.stack(
        (
            x_values % 251,
            y_values % 253,
            (x_values + y_values) % 255,
        ),
        axis=2,
    ).astype(np.uint8)


def _transcript(
    identities: tuple[LatentRowIdentity, ...],
    image: NDArray[np.uint8],
) -> str:
    digest = hashlib.sha256()
    for identity in identities:
        crop = image[
            identity.y : identity.y + 256,
            identity.x : identity.x + 256,
        ]
        chw = np.ascontiguousarray(crop.transpose(2, 0, 1))
        digest.update(
            struct.pack(
                "<QQQQ",
                identity.atlas_row_index,
                identity.wsi_id,
                identity.x,
                identity.y,
            ),
        )
        digest.update(chw.tobytes(order="C"))
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, object]:
    value = cast("object", json.loads(path.read_text(encoding="utf-8")))
    assert isinstance(value, dict)
    return cast("dict[str, object]", value)
