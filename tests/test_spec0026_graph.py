# Copyright 2026 HiperMaximus
# ruff: noqa: COM812, PLR2004
"""Graph-contract tests for the Spec 0026 local-global MIL model."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest
import torch

from eqvae.data.supervised_latents import LogicalPointer, WSIInstance
from eqvae.models.local_global_mil import build_local_attention_graph

if TYPE_CHECKING:
    from collections.abc import Sequence


def _instances(
    coordinates: Sequence[tuple[int, int]], *, wsi_id: int = 17
) -> tuple[WSIInstance, ...]:
    return tuple(
        WSIInstance(
            instance_row=index,
            atlas_row_index=100 + index,
            wsi_id=wsi_id,
            x=grid_x * 256,
            y=grid_y * 256,
            diagnosis_label="HGSC",
            diagnosis_index=2,
            split="train",
            pointer=LogicalPointer(part=1, file_index=index),
        )
        for index, (grid_x, grid_y) in enumerate(coordinates)
    )


def test_graph_preserves_order_holes_borders_self_and_radial_codes() -> None:
    """Invariant: coordinate adjacency prevents CSV order from inventing tissue."""
    instances = _instances(((1, 1), (0, 0), (2, 0), (0, 2), (2, 2), (4, 4)))
    graph = build_local_attention_graph(instances, expected_instance_count=6)

    assert graph.neighbor_index.shape == (6, 25)
    assert graph.neighbor_index.dtype == torch.int64
    assert graph.neighbor_valid.dtype == torch.bool
    assert graph.radial_code.dtype == torch.uint8

    center_valid = graph.neighbor_valid[0]
    assert cast(
        "list[int]",
        graph.neighbor_index[0, center_valid].tolist(),  # pyright: ignore[reportUnknownMemberType]
    ) == [
        1,
        3,
        0,
        2,
        4,
    ]
    assert cast(
        "list[int]",
        graph.radial_code[0, center_valid].tolist(),  # pyright: ignore[reportUnknownMemberType]
    ) == [
        2,
        2,
        0,
        2,
        2,
    ]
    assert graph.neighbor_index[0, ~center_valid].eq(-1).all()
    assert graph.radial_code[0, ~center_valid].eq(255).all()

    isolated_valid = graph.neighbor_valid[5]
    assert cast(
        "list[int]",
        graph.neighbor_index[5, isolated_valid].tolist(),  # pyright: ignore[reportUnknownMemberType]
    ) == [
        4,
        5,
    ]
    assert cast(
        "list[int]",
        graph.radial_code[5, isolated_valid].tolist(),  # pyright: ignore[reportUnknownMemberType]
    ) == [5, 0]


def test_graph_hash_is_stable_and_binds_order_wsi_and_topology() -> None:
    """Invariant: one graph identity binds both model branches to identical geometry."""
    coordinates = ((0, 0), (1, 0), (0, 1), (3, 3))
    graph = build_local_attention_graph(
        _instances(coordinates), expected_instance_count=4
    )
    rebuilt = build_local_attention_graph(
        _instances(coordinates), expected_instance_count=4
    )
    reordered = build_local_attention_graph(
        _instances(tuple(reversed(coordinates))), expected_instance_count=4
    )
    other_wsi = build_local_attention_graph(
        _instances(coordinates, wsi_id=18), expected_instance_count=4
    )

    assert graph.identity_sha256 == rebuilt.identity_sha256
    assert graph.identity_sha256 == (
        "64c55d7432a63cd4b58e0caffe3b80c4314db1e37db03b2fa9c8fdf1fc1be345"
    )
    assert graph.identity_sha256 != reordered.identity_sha256
    assert graph.identity_sha256 != other_wsi.identity_sha256
    assert len(graph.identity_sha256) == 64


@pytest.mark.parametrize(
    ("instances", "message"),
    [
        ((), "at least one"),
        (_instances(((0, 0), (0, 0))), "unique"),
        (
            (
                _instances(((0, 0),))[0],
                _instances(((1, 0),), wsi_id=18)[0],
            ),
            "same WSI",
        ),
    ],
)
def test_graph_rejects_ambiguous_instance_contracts(
    instances: tuple[WSIInstance, ...], message: str
) -> None:
    """Invariant: invalid metadata fails early, preventing silent graph drift."""
    with pytest.raises(ValueError, match=message):
        build_local_attention_graph(instances, expected_instance_count=len(instances))


def test_graph_rejects_off_lattice_coordinates() -> None:
    """Invariant: non-lattice pixels cannot create false neighbourhood edges."""
    instance = _instances(((0, 0),))[0]
    off_lattice = WSIInstance(
        instance_row=instance.instance_row,
        atlas_row_index=instance.atlas_row_index,
        wsi_id=instance.wsi_id,
        x=1,
        y=instance.y,
        diagnosis_label=instance.diagnosis_label,
        diagnosis_index=instance.diagnosis_index,
        split=instance.split,
        pointer=instance.pointer,
    )
    with pytest.raises(ValueError, match="256-pixel"):
        build_local_attention_graph((off_lattice,), expected_instance_count=1)


def test_graph_binds_the_complete_bag_instance_count() -> None:
    """Invariant: a truncated bag cannot masquerade as a complete WSI graph."""
    with pytest.raises(ValueError, match="complete bag count"):
        build_local_attention_graph(_instances(((0, 0),)), expected_instance_count=2)


def test_graph_detects_tensor_mutation_after_construction() -> None:
    """Invariant: paired branches cannot consume geometry under a stale digest."""
    graph = build_local_attention_graph(
        _instances(((0, 0), (1, 0))), expected_instance_count=2
    )
    graph.radial_code[0, 0] = 1

    with pytest.raises(ValueError, match=r"radial codes|canonical identity"):
        graph.verify_integrity()
