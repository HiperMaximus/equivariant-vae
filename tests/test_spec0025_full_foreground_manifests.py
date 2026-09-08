# Copyright 2026 HiperMaximus
# pyright: reportPrivateUsage=false
# ruff: noqa: PLR2004, SLF001
"""Focused proofs for source-qualified full bags and development-only access."""

from __future__ import annotations

import json
import zlib
from collections import Counter
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from eqvae.cli import build_ubc_full_foreground_manifests as builder
from eqvae.cli import build_ubc_supervised_manifests as legacy
from eqvae.data import full_foreground_latents as reader
from eqvae.data.latent_shards import (
    EXPECTED_CHECKPOINT_SHA256,
    make_latent_shard_header,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from eqvae.data.supervised_latents import LearningSplit


def _write_csv(
    path: Path,
    header: Sequence[str],
    rows: Sequence[Sequence[object]],
) -> None:
    legacy._write_csv(path, header, rows)


def _fixture(
    tmp_path: Path,
) -> tuple[list[builder.SourcePart], Path, Path, list[tuple[object, ...]]]:
    rows = {
        1: [(0, 10, 0, 0), (2, 10, 512, 0), (3, 10, 768, 0)],
        11: [(1, 10, 256, 0), (5, 20, 0, 0)],
        13: [(4, 10, 1024, 0), (6, 30, 0, 0)],
    }
    sources: list[builder.SourcePart] = []
    catalog: list[tuple[object, ...]] = []
    for part, identities in rows.items():
        root = tmp_path / f"part-{part}"
        root.mkdir()
        manifest = root / "manifest.csv"
        _write_csv(manifest, builder.prior.MANIFEST_HEADER, identities)
        for model_index, model in enumerate(EXPECTED_CHECKPOINT_SHA256):
            name = f"{model}_mu.bin"
            payload = np.stack([
                np.full((16, 32, 32), part * 10 + i + model_index * 1000, dtype="<f4")
                for i in range(len(identities))
            ]).tobytes()
            binary = (
                make_latent_shard_header(
                    tensor_count=len(identities),
                    payload_crc32=zlib.crc32(payload),
                )
                + payload
            )
            (root / name).write_bytes(binary)
            sidecar = {
                "schema_version": "spec0020.latent_shard.v1",
                "status": "complete",
                "model_name": model,
                "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256[model],
                "file_size": len(binary),
                "payload_bytes": len(payload),
                "payload_crc32": zlib.crc32(payload),
                "source_manifest": {
                    "logical_basename": manifest.name,
                    "sha256": legacy._sha256(manifest),
                    "row_count": len(identities),
                },
                "tensor": {
                    "count": len(identities),
                    "dtype": "float32_le",
                    "layout": "CHW",
                    "shape": [16, 32, 32],
                    "record_bytes": 65_536,
                },
            }
            sidecar_path = root / f"{model}_mu.json"
            legacy._write_json(sidecar_path, sidecar)
            catalog.append((
                part,
                model,
                f"fixture/part-{part}",
                name,
                len(identities),
                len(binary),
                legacy._sha256(root / name),
                sidecar_path.name,
                sidecar_path.stat().st_size,
                legacy._sha256(sidecar_path),
            ))
        sources.append(
            builder.SourcePart(
                part,
                manifest,
                legacy._sha256(manifest),
                len(identities),
                root,
            ),
        )
    candidate = tmp_path / "candidates.csv"
    selected = sorted(r for group in rows.values() for r in group if r[0] != 2)
    _write_csv(
        candidate,
        (*builder.prior.MANIFEST_HEADER, "split"),
        [(*r, {10: "train", 20: "validation", 30: "test"}[r[1]]) for r in selected],
    )
    split_path = tmp_path / "split.csv"
    _write_csv(
        split_path,
        ("wsi_id", "diagnosis_label", "diagnosis_index", "split"),
        [
            (10, "CC", 0, "train"),
            (20, "EC", 1, "validation"),
            (30, "SEALED", "NOT_A_LABEL_INDEX", "test"),
        ],
    )
    return sources, candidate, split_path, catalog


def _development(tmp_path: Path) -> tuple[Path, list[builder.SourcePart], str]:
    sources, candidate, splits, catalog = _fixture(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    summary = builder.write_views(
        output=output,
        sources=sources,
        candidate=candidate,
        split_path=splits,
        expected_per_wsi={10: 4, 20: 1, 30: 1},
    )
    legacy._write_csv(
        output / "development/physical_parts.csv",
        legacy.CATALOG_HEADER,
        catalog,
    )
    builder.seal_development(output, summary)
    return output, sources, legacy._sha256(output / "development/dataset.json")


def test_full_foreground_join_preserves_gaps_pair_order_and_complete_bags(
    tmp_path: Path,
) -> None:
    """One shared mapping must restore all patches across repeated filenames.

    The omitted base record is deliberate mask-only policy; renumbering its
    successor, swapping producer roots or truncating a bag changes known values.
    An invalid sealed label proves test labels are not interpreted by the join.
    """
    output, sources, digest = _development(tmp_path)
    roots = {f"fixture/part-{s.part}": s.metadata_root for s in sources}
    observed: dict[str, list[float]] = {}
    for model in EXPECTED_CHECKPOINT_SHA256:
        with reader.FullForegroundBagDataset(
            development_root=output / "development",
            expected_contract_sha256=digest,
            split="train",
            model_name=model,
            source_roots=roots,
        ) as dataset:
            bag = dataset[0]
            assert len(dataset) == 1
            assert bag.bag.instance_count == len(bag.instances) == 4
            assert [r.atlas_row_index for r in bag.instances] == [0, 1, 3, 4]
            assert [(r.part, r.file_index) for r in bag.physical_reads] == [
                (1, 0),
                (1, 2),
                (11, 0),
                (13, 0),
            ]
            observed[model] = [float(value) for value in bag.latents[:, 0, 0, 0]]
    assert observed == {
        "normal_vae": [10.0, 110.0, 12.0, 130.0],
        "so2_vae": [1010.0, 1110.0, 1012.0, 1130.0],
    }
    for path in (output / "sealed_test").iterdir():
        assert "diagnosis" not in path.read_text()
        assert "SEALED" not in path.read_text()
    contract = legacy._read_object(output / "development/dataset.json")
    assert contract["bags"] == {"train": {"10": 4}, "validation": {"20": 1}}


@pytest.mark.parametrize(
    "mutation",
    ["duplicate", "missing", "tail", "atlas", "split", "extra"],
)
def test_full_foreground_merge_rejects_wrong_membership(
    tmp_path: Path,
    mutation: str,
) -> None:
    """Each corruption invalidates unique exact foreground coverage before publication.

    Distinct mutations catch overlapping parts, interior/end omissions, atlas
    aliasing, candidate split drift and unauthorized extra supplement records.
    """
    sources, candidate, splits, _ = _fixture(tmp_path)
    if mutation == "duplicate":
        sources.append(sources[0])
    elif mutation == "missing":
        sources = [s for s in sources if s.part != 11]
    elif mutation == "tail":
        candidate.write_text(candidate.read_text() + "7,30,256,0,test\n")
    elif mutation == "atlas":
        candidate.write_text(candidate.read_text().replace("3,10,768,0", "2,10,768,0"))
    elif mutation == "split":
        candidate.write_text(candidate.read_text().replace("validation", "train"))
    else:
        candidate.write_text(candidate.read_text().replace("4,10,1024,0,train\n", ""))
    with pytest.raises(ValueError, match=r"overlap|Missing|drift|split|nonforeground"):
        list(
            builder.resolve_foreground(
                sources,
                candidate,
                legacy._load_split(splits),
                Counter(),
            ),
        )


def test_full_foreground_checks_each_wsi_not_just_global_count(tmp_path: Path) -> None:
    """Equal total counts cannot excuse moving one patch from a bag to another."""
    sources, candidate, splits, _ = _fixture(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(ValueError, match="Per-WSI"):
        builder.write_views(
            output=output,
            sources=sources,
            candidate=candidate,
            split_path=splits,
            expected_per_wsi={10: 3, 20: 2, 30: 1},
        )


def test_full_foreground_source_binds_both_models_to_exact_manifest(
    tmp_path: Path,
) -> None:
    """Authenticated sidecars must bind the same source manifest for both models.

    Even self-consistent sidecar hashes cannot bless a SO(2) manifest swap.
    """
    sources, _, _, catalog = _fixture(tmp_path)
    for source in sources:
        builder._validate_source(source, catalog)
    sidecar = sources[-1].metadata_root / "so2_vae_mu.json"
    raw = legacy._read_object(sidecar)
    cast("dict[str, object]", raw["source_manifest"])["sha256"] = sources[
        1
    ].manifest_sha256
    sidecar.write_text(json.dumps(raw))
    catalog[-1] = (*catalog[-1][:8], sidecar.stat().st_size, legacy._sha256(sidecar))
    with pytest.raises(ValueError, match="manifest/sidecar pairing"):
        builder._validate_source(sources[-1], catalog)


@pytest.mark.parametrize(
    "mutation",
    ["test", "contract", "csv", "extra_file", "wrong_split", "wrong_wsi"],
)
def test_full_foreground_development_guards_before_latent_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    """Fail before store construction for unsafe metadata.

    The external JSON pin prevents self-attestation; explicit membership checks
    also reject a mislabelled requested split even with freshly sealed metadata.
    """
    output, sources, digest = _development(tmp_path)
    dev = output / "development"
    split = cast("LearningSplit", "test" if mutation == "test" else "train")
    if mutation == "contract":
        digest = "0" * 64
    elif mutation in {"csv", "wrong_split", "wrong_wsi"}:
        for name in ("wsi_cancer_train_instances.csv", "wsi_cancer_train_bags.csv"):
            path = dev / name
            path.write_text(
                path.read_text().replace(
                    "train" if mutation != "wrong_wsi" else ",10,",
                    "test" if mutation != "wrong_wsi" else ",30,",
                ),
            )
        if mutation != "csv":
            contract = legacy._read_object(dev / "dataset.json")
            files = cast("dict[str, dict[str, object]]", contract["files"])
            for name in files:
                files[name]["bytes"] = (dev / name).stat().st_size
                files[name]["sha256"] = legacy._sha256(dev / name)
            (dev / "dataset.json").write_text(json.dumps(contract))
            digest = legacy._sha256(dev / "dataset.json")
    elif mutation == "extra_file":
        (dev / "test.csv").write_text("not allowed")
    store_calls: list[object] = []

    def forbidden_store(**kwargs: object) -> None:
        store_calls.append(kwargs)
        pytest.fail("Invalid development metadata reached a physical store")

    monkeypatch.setattr(reader, "SupervisedLatentStore", forbidden_store)
    with pytest.raises(
        ValueError,
        match=r"sealed|hash pin|identity differs|allow-list|frozen split",
    ):
        reader.FullForegroundBagDataset(
            development_root=dev,
            expected_contract_sha256=digest,
            split=split,
            model_name="normal_vae",
            source_roots={f"fixture/part-{s.part}": s.metadata_root for s in sources},
        )
    assert not store_calls
