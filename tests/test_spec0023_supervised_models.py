# Copyright 2026 HiperMaximus
# ruff: noqa: PLR2004, TC003
# pyright: reportAny=false
# pyright: reportUnknownMemberType=false
"""Fast synthetic preflight for the exact Spec 0023 reader and models."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import zlib
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from torch import nn

from eqvae.data.latent_shards import (
    LATENT_RECORD_BYTES,
    LATENT_SHARD_HEADER_SIZE,
    make_latent_shard_header,
    parse_latent_shard_header,
)
from eqvae.data.supervised_latents import (
    CATALOG_HEADER,
    TISSUE_HEADER,
    WSI_BAG_HEADER,
    WSI_INSTANCE_HEADER,
    LogicalPointer,
    SupervisedLatentStore,
    TissueDataset,
    WSIBagDataset,
)
from eqvae.models.supervised import (
    AttentionMILClassifier,
    ClassSpecificAttentionMILClassifier,
    TissueClassifier,
    make_attention_mil_width_variant,
    make_class_specific_attention_mil_variant,
)
from eqvae.training.supervised_pairing import (
    make_paired_models,
    nested_tissue_epoch_order,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


def _write_binary(path: Path, values: Sequence[float]) -> str:
    tensors = np.stack(
        [np.full((16, 32, 32), value, dtype="<f4") for value in values],
    )
    payload = tensors.tobytes(order="C")
    path.write_bytes(
        make_latent_shard_header(
            tensor_count=len(values),
            payload_crc32=zlib.crc32(payload),
        )
        + payload,
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_sidecar(
    path: Path,
    *,
    binary_path: Path,
    model: str,
    rows: int,
) -> tuple[int, str]:
    binary = binary_path.read_bytes()
    header = parse_latent_shard_header(binary[:LATENT_SHARD_HEADER_SIZE])
    payload = {
        "file_size": len(binary),
        "model_name": model,
        "payload_bytes": len(binary) - LATENT_SHARD_HEADER_SIZE,
        "payload_crc32": header.payload_crc32,
        "schema_version": "spec0020.latent_shard.v1",
        "status": "complete",
        "tensor": {
            "count": rows,
            "dtype": "float32_le",
            "layout": "CHW",
            "shape": [16, 32, 32],
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(encoded)
    return len(encoded), hashlib.sha256(encoded).hexdigest()


def _write_csv(
    path: Path,
    header: Sequence[str],
    rows: Sequence[Sequence[object]],
) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _fixture_catalog(tmp_path: Path) -> tuple[Path, Mapping[str, Path]]:
    rows: list[tuple[object, ...]] = []
    roots: dict[str, Path] = {}
    for part in (1, 11):
        source = f"source-{part}"
        root = tmp_path / source
        root.mkdir()
        roots[source] = root
        for model_index, model in enumerate(("normal_vae", "so2_vae")):
            name = f"{model}_part_{part}.bin"
            path = root / name
            sha256 = _write_binary(
                path,
                [model_index * 100 + part * 10 + index for index in range(3)],
            )
            sidecar_name = f"{model}_part_{part}.json"
            sidecar_bytes, sidecar_sha256 = _write_sidecar(
                root / sidecar_name,
                binary_path=path,
                model=model,
                rows=3,
            )
            rows.append((
                part,
                model,
                source,
                name,
                3,
                LATENT_SHARD_HEADER_SIZE + 3 * LATENT_RECORD_BYTES,
                sha256,
                sidecar_name,
                sidecar_bytes,
                sidecar_sha256,
            ))
    catalog = tmp_path / "physical_parts.csv"
    _write_csv(catalog, CATALOG_HEADER, rows)
    return catalog, roots


def test_store_rejects_a_same_size_substituted_sidecar(tmp_path: Path) -> None:
    """Authenticate producer evidence before any latent row can be consumed."""
    catalog, roots = _fixture_catalog(tmp_path)
    sidecar = roots["source-1"] / "normal_vae_part_1.json"
    altered = sidecar.read_bytes().replace(b"normal_vae", b"xormal_vae")
    assert len(altered) == sidecar.stat().st_size
    sidecar.write_bytes(altered)
    with pytest.raises(ValueError, match="sidecar identity differs"):
        SupervisedLatentStore(
            catalog_path=catalog,
            model_name="normal_vae",
            source_roots=roots,
        )


def test_part_11_uses_the_same_grouped_reader_and_restores_logical_order(
    tmp_path: Path,
) -> None:
    """Treat part 11 ordinarily while preserving representation-independent order."""
    catalog, roots = _fixture_catalog(tmp_path)
    pointers = (
        LogicalPointer(11, 2),
        LogicalPointer(1, 1),
        LogicalPointer(11, 0),
    )
    with SupervisedLatentStore(
        catalog_path=catalog,
        model_name="normal_vae",
        source_roots=roots,
    ) as normal_store:
        normal, reads = normal_store.read_rows(pointers)
    with SupervisedLatentStore(
        catalog_path=catalog,
        model_name="so2_vae",
        source_roots=roots,
    ) as so2_store:
        so2, _ = so2_store.read_rows(pointers)

    assert [(read.part, read.file_index) for read in reads] == [
        (1, 1),
        (11, 0),
        (11, 2),
    ]
    assert normal[:, 0, 0, 0].tolist() == [112.0, 11.0, 110.0]
    assert so2[:, 0, 0, 0].tolist() == [212.0, 111.0, 210.0]


def test_wsi_dataset_reads_one_complete_contiguous_bag(tmp_path: Path) -> None:
    """Consume one WSI as the complete frozen bag instead of sampled patches."""
    catalog, roots = _fixture_catalog(tmp_path)
    instances = tmp_path / "instances.csv"
    bags = tmp_path / "bags.csv"
    _write_csv(
        instances,
        WSI_INSTANCE_HEADER,
        [
            (0, 1, 99, 0, 0, "HGSC", 2, "train", 11, 2),
            (1, 2, 99, 1, 0, "HGSC", 2, "train", 1, 1),
            (2, 3, 99, 2, 0, "HGSC", 2, "train", 11, 0),
        ],
    )
    _write_csv(bags, WSI_BAG_HEADER, [(0, 99, "HGSC", 2, "train", 0, 3)])
    with SupervisedLatentStore(
        catalog_path=catalog,
        model_name="normal_vae",
        source_roots=roots,
    ) as store:
        loaded = WSIBagDataset(
            instance_path=instances,
            bag_path=bags,
            store=store,
        )[0]

    assert loaded.bag.wsi_id == 99
    assert tuple(row.x for row in loaded.instances) == (0, 1, 2)
    assert loaded.latents[:, 0, 0, 0].tolist() == [112.0, 11.0, 110.0]
    assert len(loaded.physical_reads) == loaded.bag.instance_count


def test_wsi_dataset_rejects_one_wsi_split_into_multiple_bags(tmp_path: Path) -> None:
    """Prevent partial-bag attention from changing the WSI statistical unit."""
    catalog, roots = _fixture_catalog(tmp_path)
    instances = tmp_path / "instances.csv"
    bags = tmp_path / "bags.csv"
    _write_csv(
        instances,
        WSI_INSTANCE_HEADER,
        [
            (0, 1, 99, 0, 0, "HGSC", 2, "train", 1, 0),
            (1, 2, 99, 1, 0, "HGSC", 2, "train", 1, 1),
        ],
    )
    _write_csv(
        bags,
        WSI_BAG_HEADER,
        [
            (0, 99, "HGSC", 2, "train", 0, 1),
            (1, 99, "HGSC", 2, "train", 1, 1),
        ],
    )
    with (
        SupervisedLatentStore(
            catalog_path=catalog,
            model_name="normal_vae",
            source_roots=roots,
        ) as store,
        pytest.raises(ValueError, match="range or metadata"),
    ):
        WSIBagDataset(instance_path=instances, bag_path=bags, store=store)


def test_attention_is_global_and_tissue_shape_is_exact() -> None:
    """Normalize attention across every bag token and retain both task shapes."""
    torch.manual_seed(5)
    latents = torch.randn(7, 16, 32, 32)
    mil = AttentionMILClassifier().eval()
    logits, attention = mil(latents)
    tissue_logits = TissueClassifier().eval()(latents[:2])

    assert logits.shape == (5,)
    assert attention.shape == (7,)
    torch.testing.assert_close(attention.sum(), torch.tensor(1.0))
    assert bool(torch.all(attention > 0))
    assert tissue_logits.shape == (2, 3)


def test_width128_changes_only_the_gated_scorer_width() -> None:
    """Keep patch tokens, one score per patch, and the WSI head unchanged."""
    model = AttentionMILClassifier(attention_dim=128)
    assert (model.attention_v.in_features, model.attention_v.out_features) == (
        128,
        128,
    )
    assert (model.attention_u.in_features, model.attention_u.out_features) == (
        128,
        128,
    )
    assert (model.attention_w.in_features, model.attention_w.out_features) == (128, 1)
    assert (model.head.in_features, model.head.out_features) == (128, 5)

    baseline, _ = make_paired_models(AttentionMILClassifier, seed=1701)
    normal, so2 = make_paired_models(
        lambda: make_attention_mil_width_variant(attention_dim=128),
        seed=1701,
    )
    assert normal.attention_v.weight.shape == (128, 128)
    assert normal.attention_u.weight.shape == (128, 128)
    assert normal.attention_w.weight.shape == (1, 128)
    for prefix in ("patch_encoder.", "head."):
        for name, value in baseline.state_dict().items():
            if name.startswith(prefix):
                torch.testing.assert_close(normal.state_dict()[name], value)
    for name, value in normal.state_dict().items():
        torch.testing.assert_close(so2.state_dict()[name], value)


def test_class_specific_attention_starts_as_width128_then_can_specialize(  # noqa: PLR0914
) -> None:
    """Pin class maps so only learning, not initialization, adds capacity."""
    latents = torch.randn(7, 16, 32, 32)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1701)
        width128 = make_attention_mil_width_variant(attention_dim=128).eval()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1701)
        class_specific = make_class_specific_attention_mil_variant().eval()

    assert isinstance(class_specific, ClassSpecificAttentionMILClassifier)
    assert class_specific.attention_w.weight.shape == (5, 128)
    assert class_specific.attention_w.bias.shape == (5,)
    width_logits, width_attention = width128(latents)
    logits, attention = class_specific(latents)
    assert logits.shape == (5,)
    assert attention.shape == (7, 5)
    torch.testing.assert_close(attention.sum(dim=0), torch.ones(5))
    for class_index in range(5):
        torch.testing.assert_close(attention[:, class_index], width_attention)
    torch.testing.assert_close(logits, width_logits)

    with torch.no_grad():
        row_offsets = torch.arange(5, dtype=torch.float32)[:, None]
        feature_offsets = torch.linspace(-0.03, 0.03, 128)[None, :]
        class_specific.attention_w.weight.add_(row_offsets * feature_offsets)
    tokens = class_specific.patch_encoder(latents)
    gated = torch.tanh(class_specific.attention_v(tokens)) * torch.sigmoid(
        class_specific.attention_u(tokens),
    )
    manual_attention = torch.softmax(class_specific.attention_w(gated), dim=0)
    manual_pooled = manual_attention.transpose(0, 1) @ tokens
    manual_head_scores = class_specific.head(manual_pooled)
    logits, attention = class_specific(latents)
    torch.testing.assert_close(attention, manual_attention)
    torch.testing.assert_close(logits, torch.diagonal(manual_head_scores))
    assert not torch.equal(attention[:, 0], attention[:, 1])

    normal, so2 = make_paired_models(
        make_class_specific_attention_mil_variant,
        seed=1701,
    )
    for normal_parameter, so2_parameter in zip(
        normal.parameters(),
        so2.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(normal_parameter, so2_parameter)
        assert normal_parameter.data_ptr() != so2_parameter.data_ptr()

    class_specific.train()
    loss = nn.functional.cross_entropy(logits[None, :], torch.tensor([3]))
    loss.backward()
    gradients = class_specific.attention_w.weight.grad
    assert gradients is not None
    assert not torch.equal(gradients[0], gradients[1])


def test_checkpointed_complete_bag_matches_full_forward_and_gradients() -> None:
    """Prove chunk recomputation preserves global attention and all gradients."""
    torch.manual_seed(17)
    full = AttentionMILClassifier()
    chunked = copy.deepcopy(full)
    latents = torch.randn(5, 16, 32, 32)
    target = torch.tensor([3])

    full_logits, full_attention = full(latents)
    full_loss = nn.functional.cross_entropy(full_logits[None, :], target)
    full_loss.backward()
    chunk_logits, chunk_attention = chunked(latents, checkpoint_chunk_size=2)
    chunk_loss = nn.functional.cross_entropy(chunk_logits[None, :], target)
    chunk_loss.backward()

    torch.testing.assert_close(chunk_logits, full_logits, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(chunk_attention, full_attention, rtol=2e-5, atol=2e-6)
    full_parameters = dict(full.named_parameters())
    chunk_parameters = dict(chunked.named_parameters())
    assert full_parameters.keys() == chunk_parameters.keys()
    for name, full_parameter in full_parameters.items():
        full_gradient = full_parameter.grad
        chunk_gradient = chunk_parameters[name].grad
        assert full_gradient is not None
        assert chunk_gradient is not None
        torch.testing.assert_close(
            chunk_gradient,
            full_gradient,
            rtol=3e-5,
            atol=3e-6,
            msg=lambda message, name=name: f"{name}: {message}",
        )


def test_paired_initialization_and_nested_epoch_order_are_shared() -> None:
    """Keep initialization and nested-example order identical across branches."""
    normal, so2 = make_paired_models(TissueClassifier, seed=3407)
    for normal_parameter, so2_parameter in zip(
        normal.parameters(),
        so2.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(normal_parameter, so2_parameter)
        assert normal_parameter.data_ptr() != so2_parameter.data_ptr()

    full_rows = tuple(range(12))
    small_rows = {0, 2, 4, 6}
    large_rows = {*small_rows, 7, 9, 11}
    full_order = nested_tissue_epoch_order(full_rows, full_rows, epoch=3)
    small_order = nested_tissue_epoch_order(full_rows, small_rows, epoch=3)
    large_order = nested_tissue_epoch_order(full_rows, large_rows, epoch=3)
    assert small_order == tuple(row for row in full_order if row in small_rows)
    assert large_order == tuple(row for row in full_order if row in large_rows)
    assert small_order == tuple(row for row in large_order if row in small_rows)
    assert set(small_order) == small_rows
    assert nested_tissue_epoch_order(full_rows, small_rows, epoch=4) != small_order


def test_supervised_initialization_uses_nonzero_heads_and_zero_biases() -> None:
    """Pin the agreed Xavier/Kaiming/GroupNorm policy without a zero head."""
    model = AttentionMILClassifier()
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.GroupNorm)):
            assert module.bias is not None
            torch.testing.assert_close(module.bias, torch.zeros_like(module.bias))
        if isinstance(module, nn.GroupNorm):
            assert module.weight is not None
            torch.testing.assert_close(module.weight, torch.ones_like(module.weight))
        if isinstance(module, nn.Linear):
            fan_in, fan_out = module.weight.shape[1], module.weight.shape[0]
            bound = (6.0 / (fan_in + fan_out)) ** 0.5
            assert float(module.weight.detach().abs().max()) <= bound
            assert bool(torch.any(module.weight != 0))
    assert bool(torch.any(model.head.weight != 0))


def test_tissue_reader_preserves_supplied_order_and_rejects_test_rows(
    tmp_path: Path,
) -> None:
    """Read learning rows through ordinary parts without exposing sealed test data."""
    catalog, roots = _fixture_catalog(tmp_path)
    tissue = tmp_path / "tissue.csv"
    _write_csv(
        tissue,
        TISSUE_HEADER,
        [
            (0, 1, 20, 0, 0, "tumor", "train", 0, 11, 2),
            (1, 2, 21, 0, 0, "stroma", "train", 0, 1, 1),
            (2, 3, 22, 0, 0, "necrosis", "train", 0, 11, 0),
        ],
    )
    with SupervisedLatentStore(
        catalog_path=catalog,
        model_name="normal_vae",
        source_roots=roots,
    ) as store:
        dataset = TissueDataset(path=tissue, split="train", store=store)
        loaded = dataset.read_batch((2, 0, 1))
        with pytest.raises(IndexError, match="outside"):
            dataset.read_batch((-1,))

    assert loaded.labels.tolist() == [2, 0, 1]
    assert loaded.latents[:, 0, 0, 0].tolist() == [110.0, 112.0, 11.0]
    assert [(row.part, row.file_index) for row in loaded.physical_reads] == [
        (1, 1),
        (11, 0),
        (11, 2),
    ]

    test_tissue = tmp_path / "test_tissue.csv"
    _write_csv(
        test_tissue,
        TISSUE_HEADER,
        [(0, 4, 30, 0, 0, "tumor", "test", "", 1, 0)],
    )
    with (
        SupervisedLatentStore(
            catalog_path=catalog,
            model_name="normal_vae",
            source_roots=roots,
        ) as store,
        pytest.raises(ValueError, match="learning split"),
    ):
        TissueDataset(path=test_tissue, split="train", store=store)
