# Copyright 2026 HiperMaximus
# ruff: noqa: EM101, FBT003, PLR2004, PLR6301, TRY003
"""Focused executable-worker and pilot-contract tests for Spec 0021."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from eqvae.cli import generate_ubc_latent_stores as stores
from eqvae.data.latent_shards import (
    EXPECTED_CHECKPOINT_SHA256,
    LatentRowIdentity,
    LatentShardWriter,
    WorkManifest,
    load_work_manifest,
)
from eqvae.data.wsi_batches import ArrayWSIReader
from eqvae.inference.dual_writer import DualLatentWriter, WorkerResumeBinding
from eqvae.inference.pilot import (
    PilotCandidateResult,
    pilot_recipes,
    remove_pilot_payload,
    select_pilot_candidate,
    write_pilot_payload,
)
from eqvae.inference.worker import (
    DualDeviceExecutor,
    DualEncoderWorker,
    EncoderRecipe,
    FrozenEncoderRunner,
    finalize_pair,
)

if TYPE_CHECKING:
    from collections.abc import Callable

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


class _EncodeOnlyVAE(nn.Module):
    def __init__(self, offset: float = 0.0, *, nonfinite: bool = False) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.offset = offset
        self.nonfinite = nonfinite
        self.encode_calls = 0
        self.batch_sizes: list[int] = []

    def forward(self, _inputs: Tensor) -> Tensor:
        raise AssertionError("worker must not call forward or decoder")

    def encode(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        self.encode_calls += 1
        self.batch_sizes.append(inputs.shape[0])
        base = inputs[:, :1, ::8, ::8].repeat(1, 16, 1, 1) + self.offset
        if self.nonfinite:
            base[0, 0, 0, 0] = torch.nan
        return base, torch.zeros_like(base)


def test_runner_calls_only_encode_trims_compiled_padding_and_rejects_nonfinite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep padded rows outside the valid posterior output contract."""

    def identity_compile(
        function: Callable[[Tensor], tuple[Tensor, Tensor]],
    ) -> Callable[[Tensor], tuple[Tensor, Tensor]]:
        return function

    monkeypatch.setattr(torch, "compile", identity_compile)
    model = _EncodeOnlyVAE()
    runner = FrozenEncoderRunner(
        model,
        device="cpu",
        recipe=EncoderRecipe(4, "FP32", "compiled-fixed"),
    )
    batch = torch.arange(2 * 3 * 256 * 256, dtype=torch.int64).reshape(
        2,
        3,
        256,
        256,
    )
    output = runner(batch.to(torch.uint8))

    assert output.shape == (2, 16, 32, 32)
    assert output.dtype == torch.float32
    assert model.batch_sizes == [4]

    bad = FrozenEncoderRunner(
        _EncodeOnlyVAE(nonfinite=True),
        device="cpu",
        recipe=EncoderRecipe(2, "FP32", "eager"),
    )
    with pytest.raises(ValueError, match="nonfinite"):
        bad(torch.zeros((2, 3, 256, 256), dtype=torch.uint8))


def test_dual_worker_feeds_one_stream_to_both_models_and_publishes_pair_last(
    tmp_path: Path,
) -> None:
    """Materialize two aligned fixture stores from one WSI stream."""
    manifest_path = _manifest(tmp_path)
    manifest_sha256 = _sha256(manifest_path)
    manifest = load_work_manifest(
        manifest_path,
        run_number=1,
        expected_sha256=manifest_sha256,
    )
    normal_writer = _writer(tmp_path, manifest_path, manifest_sha256, "normal_vae")
    so2_writer = _writer(tmp_path, manifest_path, manifest_sha256, "so2_vae")
    dual = DualLatentWriter(
        normal_writer=normal_writer,
        so2_writer=so2_writer,
        state_path=tmp_path / "worker.resume.json",
        binding=WorkerResumeBinding("b" * 64, "c" * 64, manifest_sha256, 1),
    )
    normal_model = _EncodeOnlyVAE()
    so2_model = _EncodeOnlyVAE(offset=0.25)
    recipe = EncoderRecipe(2, "FP32", "eager")
    worker = DualEncoderWorker(
        normal=FrozenEncoderRunner(normal_model, device="cpu", recipe=recipe),
        so2=FrozenEncoderRunner(so2_model, device="cpu", recipe=recipe),
    )
    images = {
        7: np.zeros((256, 512, 3), dtype=np.uint8),
        9: np.full((256, 256, 3), 255, dtype=np.uint8),
    }
    for wsi_id in images:
        (tmp_path / f"{wsi_id}.png").write_bytes(f"png-{wsi_id}".encode())

    result = worker.run(
        manifest=manifest,
        wsi_dir=tmp_path,
        writers=dual,
        incomplete_path=tmp_path / "incomplete.json",
        reader=ArrayWSIReader(images),
    )
    audit_path = tmp_path / "pair_audit.json"
    audit = finalize_pair(
        writers=dual,
        pair_audit_path=audit_path,
        expected_union_sha256=_FAKE_UNION_SHA256,
        run_config_sha256="c" * 64,
        input_receipt_sha256="b" * 64,
    )

    assert result.row_count == 3
    assert result.wsi_count == 2
    assert normal_model.encode_calls == so2_model.encode_calls == 2
    assert audit["status"] == "complete"
    assert audit_path.is_file()
    assert not (tmp_path / "incomplete.json").exists()


def test_receipt_mount_resolution_supports_nested_versions_and_rejects_ambiguity(
    tmp_path: Path,
) -> None:
    """Resolve the receipt's exact Kaggle dataset version, never an arbitrary copy."""
    input_root = tmp_path / "input"
    receipt: dict[str, object] = {
        "dataset_reference": "owner/immutable-inputs",
        "dataset_version": 3,
    }
    nested = input_root / "datasets/owner/immutable-inputs/versions/3"
    nested.mkdir(parents=True)
    (nested / "spec0021_input_contract.json").write_text("{}\n", encoding="utf-8")

    resolved = stores._resolve_receipt_mount_root(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        receipt,
        required_filenames=("spec0021_input_contract.json",),
        label="immutable input bundle",
        input_root=input_root,
    )
    assert resolved == nested

    flat = input_root / "immutable-inputs"
    flat.mkdir()
    (flat / "spec0021_input_contract.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="found 2"):
        stores._resolve_receipt_mount_root(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
            receipt,
            required_filenames=("spec0021_input_contract.json",),
            label="immutable input bundle",
            input_root=input_root,
        )


def test_dual_executor_keeps_two_slots_owned_until_ordered_consumption() -> None:
    """The buffered runtime must never reuse either model's unconsumed slot."""
    recipe = EncoderRecipe(
        1,
        "FP32",
        "eager",
        "bounded-pinned-double-buffer",
    )
    executor = DualDeviceExecutor(
        normal=FrozenEncoderRunner(_EncodeOnlyVAE(), device="cpu", recipe=recipe),
        so2=FrozenEncoderRunner(
            _EncodeOnlyVAE(offset=0.25),
            device="cpu",
            recipe=recipe,
        ),
    )
    images = torch.zeros((1, 3, 256, 256), dtype=torch.uint8)
    identities = (
        LatentRowIdentity(atlas_row_index=0, wsi_id=7, x=0, y=0),
        LatentRowIdentity(atlas_row_index=1, wsi_id=7, x=256, y=0),
        LatentRowIdentity(atlas_row_index=2, wsi_id=9, x=0, y=0),
    )
    for row_start in range(2):
        executor.submit(
            row_start=row_start,
            identities=(identities[row_start],),
            images_uint8=images,
            final_wsi_evidence=None,
        )
    assert executor.pending_count == executor.slot_count == 2
    first = executor.drain_one()
    assert first.identities == identities[:1]
    with pytest.raises(RuntimeError, match="slots are still occupied"):
        executor.submit(
            row_start=2,
            identities=(identities[2],),
            images_uint8=images,
            final_wsi_evidence=None,
        )
    executor.release(first)
    executor.submit(
        row_start=2,
        identities=(identities[2],),
        images_uint8=images,
        final_wsi_evidence=None,
    )
    second = executor.drain_one()
    assert second.identities == identities[1:2]
    executor.release(second)
    third = executor.drain_one()
    assert third.identities == identities[2:]
    assert torch.equal(third.so2_tensors, third.normal_tensors + 0.25)
    executor.release(third)


def test_tiny_pilot_pass_reads_and_encodes_exactly_two_batches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep hidden warmups or full-WSI work from returning to the smoke path."""
    manifest_path = tmp_path / "run_02_of_05.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(_UNION_HEADER)
        for index in range(16):
            writer.writerow(
                (
                    index,
                    15_188,
                    "HGSC",
                    2,
                    index * 256,
                    0,
                    "train",
                    "true",
                    "false",
                    "",
                ),
            )
    manifest = load_work_manifest(
        manifest_path,
        run_number=2,
        expected_sha256=_sha256(manifest_path),
    )

    def smoke_range(_manifest: WorkManifest) -> tuple[int, int]:
        return 0, 16

    def smoke_sentinels(_manifest: WorkManifest) -> tuple[int, int, int]:
        return 0, 8, 15

    monkeypatch.setattr(stores, "pilot_wsi_range", smoke_range)
    monkeypatch.setattr(stores, "pilot_sentinel_indices", smoke_sentinels)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    normal_model = _EncodeOnlyVAE()
    so2_model = _EncodeOnlyVAE(offset=0.25)
    recipe = EncoderRecipe(8, "FP32", "eager", "synchronous")
    executor = DualDeviceExecutor(
        normal=FrozenEncoderRunner(normal_model, device="cpu", recipe=recipe),
        so2=FrozenEncoderRunner(so2_model, device="cpu", recipe=recipe),
    )
    result = stores._pilot_pass(  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
        manifest,
        tmp_path,
        executor,
        scratch,
        measure=True,
        reader=ArrayWSIReader(
            {15_188: np.zeros((256, 16 * 256, 3), dtype=np.uint8)},
        ),
    )
    assert normal_model.batch_sizes == [8, 8]
    assert so2_model.batch_sizes == [8, 8]
    assert result["output_bytes"] == 2 * 16 * 65_536
    assert result["finite"] is True
    assert result["identity_match"] is True
    assert not any(scratch.iterdir())


def test_pilot_uses_one_fixed_recipe_and_ephemeral_payload(tmp_path: Path) -> None:
    """Lock the single conservative smoke recipe and scratch cleanup."""
    recipes = pilot_recipes()
    assert len(recipes) == 1
    assert recipes[0] == replace(
        recipes[0],
        batch_size=8,
        d2h_mode="synchronous",
        numeric_mode="FP32",
        execution="eager",
    )
    rows = (PilotCandidateResult(recipes[0], True, 10.0, 100, 100),)
    selected = select_pilot_candidate(rows)
    assert selected.recipe.batch_size == 8
    assert selected.recipe.d2h_mode == "synchronous"
    assert selected.recipe.numeric_mode == "FP32"
    assert selected.recipe.execution == "eager"

    scratch = tmp_path / "scratch" / "pilot.bin"
    tensors = torch.arange(16 * 32 * 32, dtype=torch.float32).reshape(1, 16, 32, 32)
    assert write_pilot_payload(scratch, tensors) == 65_536
    assert scratch.stat().st_size == 65_536
    remove_pilot_payload(scratch)
    assert not scratch.exists()


def test_publication_links_exact_files_with_completion_marker_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Publish only hard-linked allow-list files and put authority last."""
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    for name in ("normal.bin", "so2.bin", "complete.json"):
        (source / name).write_bytes(name.encode())
    observed: list[str] = []
    real_link = stores.os.link

    def record_link(source_path: Path, target_path: Path) -> None:
        observed.append(Path(target_path).name)
        real_link(source_path, target_path)

    monkeypatch.setattr(stores.os, "link", record_link)
    stores._publish_link_set(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        source_root=source,
        output_root=output,
        names=("complete.json", "normal.bin", "so2.bin"),
        final_name="complete.json",
    )
    assert observed == ["normal.bin", "so2.bin", "complete.json"]
    assert {path.name for path in output.iterdir()} == set(observed)
    assert all(
        (source / name).stat().st_ino == (output / name).stat().st_ino
        for name in observed
    )


def test_publication_failure_rolls_back_every_visible_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Remove every visible link if publication fails before its marker."""
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    for name in ("a.bin", "b.bin", "marker.json"):
        (source / name).write_bytes(b"payload")
    real_link = stores.os.link
    calls = 0

    def fail_second(source_path: Path, target_path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected publication failure")
        real_link(source_path, target_path)

    monkeypatch.setattr(stores.os, "link", fail_second)
    with pytest.raises(OSError, match="injected"):
        stores._publish_link_set(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
            source_root=source,
            output_root=output,
            names=("a.bin", "b.bin", "marker.json"),
            final_name="marker.json",
        )
    assert not any(output.iterdir())
    assert {path.name for path in source.iterdir()} == {
        "a.bin",
        "b.bin",
        "marker.json",
    }


def test_rebind_worker_resume_is_atomic_and_checks_previous_config(
    tmp_path: Path,
) -> None:
    """Rebind only a journal that still matches its validated attachment."""
    state = tmp_path / "worker.resume.json"
    state.write_text(
        json.dumps({"run_config_sha256": "a" * 64}) + "\n",
        encoding="utf-8",
    )
    stores._rebind_worker_resume(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        state,
        expected_previous_sha256="a" * 64,
        current_sha256="b" * 64,
    )
    assert json.loads(state.read_text(encoding="utf-8"))["run_config_sha256"] == (
        "b" * 64
    )
    with pytest.raises(ValueError, match="does not match"):
        stores._rebind_worker_resume(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
            state,
            expected_previous_sha256="a" * 64,
            current_sha256="c" * 64,
        )


def test_production_evidence_requires_exact_tiny_smoke_binding(
    tmp_path: Path,
) -> None:
    """Accept only the 16-patch smoke result bound to this input and recipe."""
    receipt: dict[str, object] = {"dataset_reference": stores.INPUT_DATASET_SLUG}
    selected = {
        "batch_size": 8,
        "d2h": "synchronous",
        "numeric": "FP32",
        "execution": "eager",
    }
    authority = {
        "schema_version": stores.PILOT_SUMMARY_SCHEMA,
        "status": "smoke_passed",
        "smoke_passed": True,
        "smoke_wsi_id": stores.PILOT_WSI_ID,
        "smoke_patch_count": stores.PILOT_SMOKE_PATCHES,
        "input_dataset_receipt_sha256": stores._canonical_hash(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
            receipt,
        ),
        "selected_recipe": selected,
    }
    authority_path = tmp_path / "authority.json"
    authority_path.write_text(json.dumps(authority), encoding="utf-8")
    config = {"run_number": 1, "selected_recipe": selected}
    stores._validate_production_evidence(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        config=config,
        input_receipt=receipt,
        pilot_path=authority_path,
    )
    authority["smoke_patch_count"] = 17
    authority_path.write_text(json.dumps(authority), encoding="utf-8")
    with pytest.raises(ValueError, match="Tiny pilot"):
        stores._validate_production_evidence(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
            config=config,
            input_receipt=receipt,
            pilot_path=authority_path,
        )


def test_download_validator_pins_exact_one_row_smoke_evidence(tmp_path: Path) -> None:
    """Reject a nominal pass that did hidden work or wrote the wrong row count."""
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    matrix = dataset / "spec0021_pilot_matrix.csv"
    row: dict[str, object] = dict.fromkeys(stores.MATRIX_HEADER, 0)
    row.update(
        {
            "candidate_index": 0,
            "batch_size": 8,
            "d2h": "synchronous",
            "numeric": "FP32",
            "execution": "eager",
            "status": "pass",
            "failure_kind": "",
            "warmup_passes": 0,
            "measured_passes": 1,
            "patch_count": 16,
            "output_bytes": 2 * 16 * 65_536,
            "finite": True,
            "identity_match": True,
            "eligible": True,
        },
    )
    stores._write_matrix(matrix, (row,))  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    authority = dataset / "spec0021_pilot_authority.json"
    selected = {
        "batch_size": 8,
        "d2h": "synchronous",
        "numeric": "FP32",
        "execution": "eager",
    }
    authority.write_text(
        json.dumps(
            {
                "schema_version": stores.PILOT_SUMMARY_SCHEMA,
                "status": "smoke_passed",
                "smoke_passed": True,
                "smoke_wsi_id": stores.PILOT_WSI_ID,
                "smoke_patch_count": stores.PILOT_SMOKE_PATCHES,
                "matrix_sha256": _sha256(matrix),
                "input_dataset_receipt_sha256": "b" * 64,
                "selected_recipe": selected,
            },
        ),
        encoding="utf-8",
    )
    receipt = tmp_path / "pilot_receipt.json"
    stores._validate_pilot_download(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        tmp_path,
        receipt,
    )
    assert json.loads(receipt.read_text(encoding="utf-8"))["status"] == ("smoke_passed")

    row["patch_count"] = 17
    matrix.unlink()
    stores._write_matrix(matrix, (row,))  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    authority_payload = cast(
        "dict[str, object]",
        json.loads(authority.read_text(encoding="utf-8")),
    )
    authority_payload["matrix_sha256"] = _sha256(matrix)
    authority.write_text(json.dumps(authority_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="exact tiny smoke"):
        stores._validate_pilot_download(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
            tmp_path,
            receipt,
        )


def _manifest(root: Path) -> Path:
    path = root / "run_01_of_05.csv"
    rows = (
        (0, 7, 0, 0),
        (1, 7, 256, 0),
        (2, 9, 0, 0),
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(_UNION_HEADER)
        for atlas, wsi_id, x, y in rows:
            writer.writerow(
                (atlas, wsi_id, "HGSC", 2, x, y, "train", "true", "false", ""),
            )
    return path


def _writer(
    root: Path,
    manifest_path: Path,
    manifest_sha256: str,
    model_name: str,
) -> LatentShardWriter:
    typed_name = "normal_vae" if model_name == "normal_vae" else "so2_vae"
    return LatentShardWriter(
        bin_path=root / f"{typed_name}.bin",
        manifest_path=manifest_path,
        run_number=1,
        model_name=typed_name,
        checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256[typed_name],
        expected_checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256[typed_name],
        expected_manifest_sha256=manifest_sha256,
        expected_union_sha256=_FAKE_UNION_SHA256,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
