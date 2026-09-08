# Copyright 2026 HiperMaximus
"""Focused invariants for the coordinate-masked FlexAttention probe."""

from __future__ import annotations

import importlib.util
import json
import sys
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import torch
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from eqvae.models.local_global_mil import (
    _graph_identity_sha256 as canonical_graph_identity_sha256,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

KERNEL_ROOT = Path("kaggle/kernels/wsi45630_flex_attention_probe")
SOURCE_PATH = KERNEL_ROOT / "run.py"
METADATA_PATH = KERNEL_ROOT / "kernel-metadata.json"
MAX_SOURCE_BYTES = 1_000_000
RADIAL_PADDING_CODE = 255
INJECTED_FAILED_BLOCK_SIZE = 16
type GraphResult = tuple[
    list[list[int]],
    list[list[bool]],
    list[list[int]],
    list[int],
]


class LocalAttentionContract(Protocol):
    """Methods exercised on the standalone local-attention module."""

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return named parameter tensors."""
        ...

    def load_state_dict(self, state: Mapping[str, torch.Tensor]) -> object:
        """Load named parameter tensors."""
        ...

    def explicit(
        self,
        tokens: torch.Tensor,
        indices: torch.Tensor,
        valid: torch.Tensor,
        radial: torch.Tensor,
    ) -> torch.Tensor:
        """Run explicit sparse attention."""
        ...

    def flex(
        self,
        tokens: torch.Tensor,
        block_mask: object,
        coordinate_tensors: tuple[torch.Tensor, ...],
        compiled_flex: Callable[..., torch.Tensor],
        *,
        block_size: int,
    ) -> torch.Tensor:
        """Run the FlexAttention candidate."""
        ...


class ProbeContract(Protocol):
    """Typed helpers exposed by the intentionally standalone probe."""

    HEADS: int
    HEAD_WIDTH: int
    TOKEN_WIDTH: int
    BLOCK_SIZES: tuple[int, ...]
    METADATA_BYTE_LIMIT: int
    build_graph: Callable[[Sequence[tuple[int, int]]], GraphResult]
    graph_identity_sha256: Callable[..., str]
    edge_allowed: Callable[[Sequence[tuple[int, int]], int, int], bool]
    build_sparse_block_metadata: Callable[..., object]
    block_metadata_summary: Callable[[object], dict[str, object]]
    make_block_mask: Callable[..., tuple[object, tuple[torch.Tensor, ...]]]
    make_module: Callable[[object], LocalAttentionContract]
    evaluate_candidate_set: Callable[..., tuple[dict[str, object], list[int]]]


class SparseMetadataContract(Protocol):
    """Tensor members needed to prove the direct block-list construction."""

    kv_num_blocks: torch.Tensor
    kv_indices: torch.Tensor
    q_num_blocks: torch.Tensor
    q_indices: torch.Tensor


class BlockMaskContract(Protocol):
    """Callable exact mask member exposed by Torch BlockMask."""

    mask_mod: Callable[..., torch.Tensor]


def _load_probe() -> ProbeContract:
    """Import the shipped script so tests cover the bytes intended for Kaggle.

    Returns:
        The standalone probe module under its tested structural contract.

    """
    spec = importlib.util.spec_from_file_location("spec0031_probe", SOURCE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return cast("ProbeContract", module)


def _graph_tensors(
    graph: GraphResult,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Translate the public list contract without changing edge membership.

    Returns:
        Index, validity and radial-code tensors in the model dtypes.

    """
    indices, valid, radial, _ = graph
    return (
        torch.tensor(indices, dtype=torch.int64),
        torch.tensor(valid, dtype=torch.bool),
        torch.tensor(radial, dtype=torch.uint8),
    )


def test_metadata_is_private_coordinate_only_and_account_portable() -> None:
    """The probe may read only the shared coordinate contract under any actor."""
    metadata = cast("dict[str, object]", json.loads(METADATA_PATH.read_text()))
    assert metadata == {
        "id": "maximusshtefan/eqvae-wsi45630-flex-attention-probe",
        "title": "eqvae WSI45630 flex attention probe",
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": ["maximusshtefan/eqvae-wsi45630-capacity-inputs"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    assert {path.name for path in KERNEL_ROOT.iterdir()} == {
        "kernel-metadata.json",
        "run.py",
    }


def test_coordinate_graph_rejects_sequence_near_physical_far_edges() -> None:
    """Packed row adjacency must never masquerade as a physical WSI neighbour."""
    probe = _load_probe()
    coordinates = ((0, 0), (2560, 0), (0, 256), (512, 512))
    graph = probe.build_graph(coordinates)
    indices, valid, _, degrees = graph
    assert degrees == [3, 1, 3, 3]
    assert probe.edge_allowed(coordinates, 0, 1) is False
    assert probe.edge_allowed(coordinates, 1, 0) is False
    assert probe.edge_allowed(coordinates, 0, len(coordinates)) is True
    assert 1 not in {
        index for index, keep in zip(indices[0], valid[0], strict=True) if keep
    }
    assert 0 not in {
        index for index, keep in zip(indices[1], valid[1], strict=True) if keep
    }


def test_every_listed_partial_tile_rechecks_exact_edge_membership() -> None:  # noqa: PLR0914
    """Coarse block scheduling must not admit any false pair inside a tile."""
    probe = _load_probe()
    grid = tuple((x * 256, y * 256) for y in range(5) for x in range(8))
    coordinates = (*grid[::2], *grid[1::2])
    indices, valid, radial, _ = probe.build_graph(coordinates)
    graph_edges = {
        (query, key)
        for query, (row, row_valid) in enumerate(zip(indices, valid, strict=True))
        for key, keep in zip(row, row_valid, strict=True)
        if keep
    }
    predicate_edges = {
        (query, key)
        for query in range(len(coordinates))
        for key in range(len(coordinates))
        if probe.edge_allowed(coordinates, query, key)
    }
    assert predicate_edges == graph_edges
    assert {
        radial_code
        for row in radial
        for radial_code in row
        if radial_code != RADIAL_PADDING_CODE
    } == {
        0,
        1,
        2,
        3,
        4,
        5,
    }
    for block_size in probe.BLOCK_SIZES:
        metadata = probe.build_sparse_block_metadata(
            torch,
            indices,
            valid,
            block_size=block_size,
        )
        summary = probe.block_metadata_summary(metadata)
        assert cast("int", summary["total_bytes"]) < probe.METADATA_BYTE_LIMIT
        tensors = cast("SparseMetadataContract", metadata)
        query_blocks = (len(coordinates) + block_size - 1) // block_size
        kv_blocks = (len(coordinates) + 1 + block_size - 1) // block_size
        null_block = len(coordinates) // block_size
        expected_forward: list[list[int]] = []
        for query_block in range(query_blocks):
            scheduled = {null_block}
            for query in range(
                query_block * block_size,
                min((query_block + 1) * block_size, len(coordinates)),
            ):
                scheduled.update(
                    key // block_size
                    for key, keep in zip(indices[query], valid[query], strict=True)
                    if keep
                )
            expected_forward.append(sorted(scheduled))
        kv_counts = tensors.kv_num_blocks[0, 0]
        kv_indices = tensors.kv_indices[0, 0]
        actual_forward: list[list[int]] = [
            [int(item) for item in kv_indices[row, : int(kv_counts[row])]]
            for row in range(query_blocks)
        ]
        assert actual_forward == expected_forward
        expected_reverse: list[list[int]] = [[] for _ in range(kv_blocks)]
        for query_block, key_blocks in enumerate(expected_forward):
            for key_block in key_blocks:
                expected_reverse[key_block].append(query_block)
        q_counts = tensors.q_num_blocks[0, 0]
        q_indices = tensors.q_indices[0, 0]
        actual_reverse: list[list[int]] = [
            [int(item) for item in q_indices[row, : int(q_counts[row])]]
            for row in range(kv_blocks)
        ]
        assert actual_reverse == expected_reverse
        block_mask, _ = probe.make_block_mask(torch, metadata, coordinates, "cpu")
        mask_mod = cast("BlockMaskContract", block_mask).mask_mod
        for query_block, key_blocks in enumerate(actual_forward):
            for key_block in key_blocks:
                for query in range(
                    query_block * block_size,
                    min((query_block + 1) * block_size, len(coordinates)),
                ):
                    for key in range(
                        key_block * block_size,
                        min((key_block + 1) * block_size, len(coordinates) + 1),
                    ):
                        actual = bool(
                            mask_mod(
                                0,
                                0,
                                torch.tensor(query),
                                torch.tensor(key),
                            ).item(),
                        )
                        assert actual is probe.edge_allowed(coordinates, query, key)


def test_graph_identity_matches_the_canonical_model_contract() -> None:
    """The standalone artifact must identify the same graph as Spec 0026."""
    probe = _load_probe()
    coordinates = ((0, 0), (2560, 0), (0, 256), (512, 512), (512, 0))
    indices, valid, radial, _ = probe.build_graph(coordinates)
    tensors = _graph_tensors((indices, valid, radial, [0] * len(coordinates)))
    expected = canonical_graph_identity_sha256(
        wsi_id=45_630,
        coordinates=tuple((x // 256, y // 256) for x, y in coordinates),
        neighbor_index=tensors[0],
        neighbor_valid=tensors[1],
        radial_code=tensors[2],
    )
    assert probe.graph_identity_sha256(coordinates, indices, valid, radial) == expected


def test_direct_forward_and_transpose_metadata_stay_block_sparse() -> None:
    """Direct metadata must avoid both token-quadratic masks and auto-transpose."""
    probe = _load_probe()
    count = 257
    coordinates = tuple((index * 256, 0) for index in range(count))
    indices, valid, _, _ = probe.build_graph(coordinates)
    for block_size in probe.BLOCK_SIZES:
        metadata = probe.build_sparse_block_metadata(
            torch,
            indices,
            valid,
            block_size=block_size,
        )
        summary = probe.block_metadata_summary(metadata)
        assert cast("int", summary["total_bytes"]) < probe.METADATA_BYTE_LIMIT
        kv_indices = cast("dict[str, object]", summary["kv_indices"])
        q_indices = cast("dict[str, object]", summary["q_indices"])
        assert cast("list[int]", kv_indices["shape"])[-1] < count
        assert cast("list[int]", q_indices["shape"])[-1] < count


def test_one_unsupported_block_size_does_not_hide_the_other_candidate() -> None:
    """Candidate isolation must continue after one backend-specific failure."""
    probe = _load_probe()
    evaluated: list[int] = []

    def injected(candidate: int) -> dict[str, bool]:
        evaluated.append(candidate)
        if candidate == INJECTED_FAILED_BLOCK_SIZE:
            message = "injected unsupported block size"
            raise RuntimeError(message)
        return {"correct": True}

    records, survivors = probe.evaluate_candidate_set((16, 32), injected)
    assert evaluated == [16, 32]
    assert survivors == [32]
    assert cast("dict[str, object]", records["16"])["status"] == "unsupported"
    assert cast("dict[str, object]", records["32"])["status"] == "supported"


def test_small_cpu_flex_forward_matches_explicit_irregular_attention() -> None:
    """Mask, radial bias and one null sink must preserve the accepted algebra."""
    probe = _load_probe()
    coordinates = ((0, 0), (2560, 0), (0, 256), (512, 512), (512, 0))
    graph = probe.build_graph(coordinates)
    indices, valid, radial = _graph_tensors(graph)
    metadata = probe.build_sparse_block_metadata(
        torch,
        graph[0],
        graph[1],
        block_size=16,
    )
    block_mask, coordinate_tensors = probe.make_block_mask(
        torch,
        metadata,
        coordinates,
        "cpu",
    )
    torch.manual_seed(31)  # pyright: ignore[reportUnknownMemberType]
    explicit = probe.make_module(torch)
    candidate = probe.make_module(torch)
    state = explicit.state_dict()
    state["relative_bias"] = torch.linspace(-0.3, 0.3, probe.HEADS * 6).reshape(
        probe.HEADS,
        6,
    )
    state["null_key"] = torch.linspace(
        -0.2,
        0.2,
        probe.HEADS * probe.HEAD_WIDTH,
    ).reshape(probe.HEADS, probe.HEAD_WIDTH)
    state["null_bias"] = torch.linspace(-0.15, 0.15, probe.HEADS)
    explicit.load_state_dict(state)
    candidate.load_state_dict(state)
    tokens = torch.randn(len(coordinates), probe.TOKEN_WIDTH)

    def cpu_debug_flex(  # noqa: PLR0913
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        score_mod: Callable[..., torch.Tensor] | None = None,
        block_mask: BlockMask | None = None,
        kernel_options: dict[str, object] | None = None,
    ) -> torch.Tensor:
        assert kernel_options == {
            "BACKEND": "TRITON",
            "BLOCK_M": 16,
            "BLOCK_N": 16,
        }
        return flex_attention(
            query,
            key,
            value,
            score_mod=score_mod,
            block_mask=block_mask,
        )

    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        expected = explicit.explicit(tokens, indices, valid, radial)
        actual = candidate.flex(
            tokens,
            block_mask,
            coordinate_tensors,
            cpu_debug_flex,
            block_size=16,
        )
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_cuda_path_forces_compiled_triton_without_dense_mask_builder() -> None:
    """Performance evidence must fail closed instead of using dense Flex debug."""
    source = SOURCE_PATH.read_text(encoding="utf-8")
    assert SOURCE_PATH.stat().st_size < MAX_SOURCE_BYTES
    assert "create_block_mask" not in source
    assert '"BACKEND": "TRITON"' in source
    assert "fullgraph=True" in source
    assert 'mode="max-autotune"' in source
    assert "BlockMask(" in source
    assert "from_kv_blocks" not in source
    assert "spec0031_flex_attention_probe.json" in source
    launcher = Path("scripts/kaggle_kernel.sh").read_text(encoding="utf-8")
    assert 'flex_attention_probe_kernel_dir="kaggle/kernels/' in launcher
    assert 'flex_attention_probe_kernel_id="maximusshtefan/' in launcher
    assert "Spec 0031 FlexAttention remote launch is not authorized" in launcher
