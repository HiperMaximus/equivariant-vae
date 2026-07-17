# Copyright 2026 HiperMaximus
"""Instantiated spec 0001 model-count artifact writer."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, cast

import torch
from torch import nn

from eqvae.benchmarking.io import CsvRow, JsonObject, JsonValue, write_csv, write_json
from eqvae.config import resolve_json_config
from eqvae.models.activations import GatedScalarActivation
from eqvae.models.non_equivariant_vae import (
    DEFAULT_GROUPNORM_GROUPS,
    NonEquivariantVAE,
)
from eqvae.models.registry import MODEL_KIND_NON_EQ_TRANSLATABLE, build_model
from eqvae.models.resampling import (
    FieldwiseBilinearUpsample2x,
    FixedBinomialLowpassDownsample2x,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping
    from pathlib import Path

SPEC0001_MODEL_COUNT_INPUT_SHAPE: Final = (1, 3, 256, 256)
SPEC0001_MODEL_COUNT_TARGET: Final[JsonObject] = {
    "input_shape": [1, 3, 256, 256],
    "learned_convolution_count": 43,
    "normalization_module_count": 40,
    "gate_module_count": 34,
    "fixed_resampling_op_count": 12,
    "learned_convolution_parameters": 3_949_539,
    "groupnorm_affine_parameters": 4_800,
    "learned_gate_parameters": 4_096,
    "total_learned_parameters": 3_958_435,
    "learned_convolution_macs_per_sample": 36_471_046_144,
    "fixed_resampling_macs_per_sample": 85_032_960,
    "total_macs_per_sample_with_fixed_resampling": 36_556_079_104,
    "activation_output_elements_per_sample": 36_110_336,
}

MODEL_INVENTORY_COLUMNS: Final[tuple[str, ...]] = (
    "module_id",
    "module_type",
    "parent_path",
    "stage",
    "block",
    "branch",
    "op_index",
    "observed_forward_index",
    "input_shape",
    "output_shape",
    "kernel_size",
    "stride",
    "padding",
    "groups",
    "taps",
    "trainable",
    "learned_parameter_count",
    "macs_per_sample",
    "activation_output_elements",
    "in_channels",
    "out_channels",
    "has_bias",
    "followed_by_norm",
    "gate_channels",
    "resampling_kind",
    "count_category",
    "mac_formula",
)

_INPUT_SPATIAL_SIZE: Final = 256
_MODEL_INVENTORY_FILENAME: Final = "model_inventory.csv"
_BLOCK_MODULE_PATH_PARTS: Final = 3
_BCHW_RANK: Final = 4
_CHANNEL_DIMENSION: Final = 1
_HEIGHT_DIMENSION: Final = 2
_WIDTH_DIMENSION: Final = 3


@dataclass(frozen=True)
class ConvEntrySpec:
    """Shape and parameter metadata for one learned convolution row."""

    in_channels: int
    out_channels: int
    kernel_size: int
    spatial_size: int
    bias: bool
    followed_by_norm: bool


@dataclass(frozen=True)
class InventoryEntry:
    """One model inventory row used for count verification."""

    module_id: str
    module_type: str
    stage: str
    block: str
    branch: str
    op_index: int
    observed_forward_index: int
    input_shape: str
    output_shape: str
    kernel_size: str
    stride: str
    padding: str
    groups: int
    taps: int
    trainable: bool
    learned_parameter_count: int
    macs_per_sample: int
    activation_output_elements: int
    in_channels: int
    out_channels: int
    has_bias: bool
    followed_by_norm: bool
    gate_channels: int
    resampling_kind: str
    count_category: str
    mac_formula: str

    @property
    def parent_path(self) -> str:
        """The containing module path."""
        return self.module_id.rpartition(".")[0]

    def as_csv_row(self) -> CsvRow:
        """Convert the entry to a CSV-ready row.

        Returns:
            String-valued row with the exact inventory columns.

        """
        return {
            "module_id": self.module_id,
            "module_type": self.module_type,
            "parent_path": self.parent_path,
            "stage": self.stage,
            "block": self.block,
            "branch": self.branch,
            "op_index": str(self.op_index),
            "observed_forward_index": str(self.observed_forward_index),
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding,
            "groups": str(self.groups),
            "taps": str(self.taps),
            "trainable": _bool_csv(value=self.trainable),
            "learned_parameter_count": str(self.learned_parameter_count),
            "macs_per_sample": str(self.macs_per_sample),
            "activation_output_elements": str(self.activation_output_elements),
            "in_channels": str(self.in_channels),
            "out_channels": str(self.out_channels),
            "has_bias": _bool_csv(value=self.has_bias),
            "followed_by_norm": _bool_csv(value=self.followed_by_norm),
            "gate_channels": str(self.gate_channels),
            "resampling_kind": self.resampling_kind,
            "count_category": self.count_category,
            "mac_formula": self.mac_formula,
        }


@dataclass(frozen=True)
class ObservedCounts:
    """Observed counts from an instantiated spec 0001 model."""

    learned_convolution_count: int
    normalization_module_count: int
    gate_module_count: int
    fixed_resampling_op_count: int
    learned_convolution_parameters: int
    groupnorm_affine_parameters: int
    learned_gate_parameters: int
    total_learned_parameters: int
    learned_convolution_macs_per_sample: int
    fixed_resampling_macs_per_sample: int
    total_macs_per_sample_with_fixed_resampling: int
    activation_output_elements_per_sample: int

    def as_flat_payload(self) -> JsonObject:
        """Return flat JSON values matching the spec target keys.

        Returns:
            Flat JSON object with observed count fields.

        """
        return {
            "learned_convolution_count": self.learned_convolution_count,
            "normalization_module_count": self.normalization_module_count,
            "gate_module_count": self.gate_module_count,
            "fixed_resampling_op_count": self.fixed_resampling_op_count,
            "learned_convolution_parameters": self.learned_convolution_parameters,
            "groupnorm_affine_parameters": self.groupnorm_affine_parameters,
            "learned_gate_parameters": self.learned_gate_parameters,
            "total_learned_parameters": self.total_learned_parameters,
            "learned_convolution_macs_per_sample": (
                self.learned_convolution_macs_per_sample
            ),
            "fixed_resampling_macs_per_sample": (self.fixed_resampling_macs_per_sample),
            "total_macs_per_sample_with_fixed_resampling": (
                self.total_macs_per_sample_with_fixed_resampling
            ),
            "activation_output_elements_per_sample": (
                self.activation_output_elements_per_sample
            ),
        }

    def as_observed_payload(self) -> JsonObject:
        """Return the nested `observed` proof payload.

        Returns:
            JSON object for the model-count proof's `observed` field.

        """
        return {
            "total_learned_parameters": self.total_learned_parameters,
            "learned_convolution_macs_per_sample": (
                self.learned_convolution_macs_per_sample
            ),
            "fixed_resampling_macs_per_sample": (self.fixed_resampling_macs_per_sample),
        }


@dataclass(frozen=True)
class ModuleLocation:
    """Canonical inventory location metadata derived from module paths."""

    stage: str
    block: str
    branch: str
    op_index: int


@dataclass(frozen=True)
class ModuleObservation:
    """One live shape/execution observation from a meta forward pass."""

    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    observed_forward_index: int


@dataclass(frozen=True)
class ObservedEntryDetails:
    """Live module details used to build one observed inventory entry."""

    module_type: str
    input_shape: str
    output_shape: str
    kernel_size: str
    stride: str
    padding: str
    groups: int
    taps: int
    trainable: bool
    learned_parameter_count: int
    macs_per_sample: int
    activation_output_elements: int
    in_channels: int
    out_channels: int
    has_bias: bool
    followed_by_norm: bool
    gate_channels: int
    resampling_kind: str
    count_category: str
    mac_formula: str


@dataclass(frozen=True)
class InventoryProof:
    """Observed inventory rows plus row/order verification details."""

    entries: list[InventoryEntry]
    matches_expected_inventory: bool
    forward_order_verified: bool
    mismatch_count: int
    mismatches: list[str]


def build_model_count_payload(
    config_path: Path,
    *,
    module_inventory_path: str = f"benchmark/{_MODEL_INVENTORY_FILENAME}",
    model: NonEquivariantVAE | None = None,
) -> tuple[JsonObject, list[CsvRow]]:
    """Build the instantiated `benchmark/model_count.json` payload.

    Returns:
        JSON-ready model-count payload and CSV-ready inventory rows.

    """
    resolved_config = resolve_json_config(config_path)
    checked_model = model
    if checked_model is None:
        checked_model = build_model(
            MODEL_KIND_NON_EQ_TRANSLATABLE,
            model_config={
                "norm_groups": _read_norm_groups(
                    resolved_config.effective_config,
                    config_path=config_path,
                ),
            },
        )
    entries = _build_inventory(checked_model)
    observed = _count_inventory(entries=entries.entries, model=checked_model)
    expected = _expected_payload()
    model_config = _required_mapping(resolved_config.effective_config, "model")
    matches_spec_target = _matches_spec_target(observed)
    zero_head_verified = _zero_initialized_rgb_head_verified(checked_model)
    banned_operations_checked = _banned_operations_checked(checked_model)
    status = (
        "pass"
        if (
            matches_spec_target
            and entries.matches_expected_inventory
            and zero_head_verified
            and banned_operations_checked
        )
        else "fail"
    )
    payload: JsonObject = {
        "status": status,
        "benchmark_kind": "implementation_model_count",
        "benchmark_source": "instantiated_model",
        "full_run_eligible": status == "pass",
        "config": str(config_path),
        "architecture_id": _required_string(model_config, "architecture_id"),
        "topology_version": _required_string(model_config, "topology_version"),
        "config_resolution": "source_config_deep_merge_v1",
        "source_config_chain": [
            source_config.as_json()
            for source_config in resolved_config.source_config_chain
        ],
        "invoked_config_hash": resolved_config.invoked_config_hash,
        "effective_config_hash": resolved_config.effective_config_hash,
        "model_config_hash": resolved_config.effective_config_hash,
        "model_config_hash_source": "canonical_json_sorted_compact_effective_config",
        "count_source": "instantiated_model",
        "input_shape": SPEC0001_MODEL_COUNT_TARGET["input_shape"],
        **observed.as_flat_payload(),
        "implementation": {
            "model_factory": "eqvae.models.non_equivariant_vae",
            "instantiated_model": True,
            "uses_meta_device_or_real_cpu": "cpu",
            "zero_initialized_rgb_head_verified": zero_head_verified,
            "banned_operations_checked": banned_operations_checked,
            "inventory_matches_expected": entries.matches_expected_inventory,
            "forward_order_verified": entries.forward_order_verified,
            "shape_source": "meta_forward_hooks",
        },
        "inventory_mismatch_count": entries.mismatch_count,
        "inventory_mismatches": _json_string_list(entries.mismatches),
        "expected": expected,
        "observed": observed.as_observed_payload(),
        "resampling_macs": {
            "actual_implementation": observed.fixed_resampling_macs_per_sample,
            "conservative_dense_grouped_5x5_equivalent": (
                observed.fixed_resampling_macs_per_sample
            ),
        },
        "module_inventory_path": module_inventory_path,
        "tolerances": {
            "parameters_abs": 0,
            "macs_abs": 0,
            "activation_output_elements_abs": 0,
        },
        "matches_spec_target": matches_spec_target,
    }
    return payload, [entry.as_csv_row() for entry in entries.entries]


def write_model_count(config_path: Path, output_path: Path) -> JsonObject:
    """Write the spec 0001 instantiated model-count artifacts.

    Returns:
        JSON-ready payload that was written to disk.

    """
    inventory_path = output_path.with_name(_MODEL_INVENTORY_FILENAME)
    payload, inventory_rows = build_model_count_payload(
        config_path=config_path,
        module_inventory_path=_artifact_reference(output_path, inventory_path),
    )
    write_csv(inventory_path, MODEL_INVENTORY_COLUMNS, inventory_rows)
    write_json(output_path, payload)
    return payload


def _build_inventory(model: NonEquivariantVAE) -> InventoryProof:
    expected_entries = _expected_inventory_entries()
    module_by_path = dict(_named_modules(model))
    _verify_inventory_paths(module_by_path=module_by_path, entries=expected_entries)
    observations = _observe_model_shapes(model=model, expected_entries=expected_entries)
    observed_entries = [
        _observed_entry(
            expected=entry,
            module=module_by_path[entry.module_id],
            observation=observations[entry.module_id],
        )
        for entry in expected_entries
    ]
    mismatches = _inventory_mismatches(
        expected_entries=expected_entries,
        observed_entries=observed_entries,
    )
    unexpected_leaf_paths = _unexpected_inventory_leaf_paths(
        module_by_path=module_by_path,
        expected_entries=expected_entries,
    )
    if unexpected_leaf_paths:
        mismatches.append(
            f"Unexpected countable leaf modules: {', '.join(unexpected_leaf_paths)}",
        )
    forward_order_verified = _forward_order_verified(
        expected_entries=expected_entries,
        observed_entries=observed_entries,
    )
    if not forward_order_verified:
        mismatches.append("Observed module execution order differs from spec order.")
    return InventoryProof(
        entries=observed_entries,
        matches_expected_inventory=len(mismatches) == 0,
        forward_order_verified=forward_order_verified,
        mismatch_count=len(mismatches),
        mismatches=mismatches[:20],
    )


def _expected_inventory_entries() -> list[InventoryEntry]:
    entries: list[InventoryEntry] = []
    entries.extend(_stem_entries())
    entries.extend(_encoder_entries())
    entries.extend(
        [
            _conv_entry(
                "mu_head",
                ConvEntrySpec(
                    in_channels=96,
                    out_channels=16,
                    kernel_size=5,
                    spatial_size=32,
                    bias=True,
                    followed_by_norm=False,
                ),
            ),
            _conv_entry(
                "logvar_head",
                ConvEntrySpec(
                    in_channels=96,
                    out_channels=16,
                    kernel_size=5,
                    spatial_size=32,
                    bias=True,
                    followed_by_norm=False,
                ),
            ),
            _conv_entry(
                "latent_projection_conv",
                ConvEntrySpec(
                    in_channels=16,
                    out_channels=96,
                    kernel_size=5,
                    spatial_size=32,
                    bias=False,
                    followed_by_norm=True,
                ),
            ),
            _norm_entry("latent_projection_norm", 96, 32),
            _gate_entry("latent_projection_gate", 96, 32),
        ],
    )
    entries.extend(_decoder_entries())
    entries.append(
        _conv_entry(
            "output_head",
            ConvEntrySpec(
                in_channels=32,
                out_channels=3,
                kernel_size=5,
                spatial_size=256,
                bias=True,
                followed_by_norm=False,
            ),
        ),
    )
    return entries


def _stem_entries() -> list[InventoryEntry]:
    return [
        _conv_entry(
            "stem_conv",
            ConvEntrySpec(
                in_channels=3,
                out_channels=32,
                kernel_size=7,
                spatial_size=256,
                bias=False,
                followed_by_norm=True,
            ),
        ),
        _norm_entry("stem_norm", 32, 256),
        _gate_entry("stem_gate", 32, 256),
    ]


def _encoder_entries() -> list[InventoryEntry]:
    block_specs = (
        (0, 256, 32, 32, False),
        (1, 256, 32, 32, False),
        (2, 256, 32, 48, True),
        (3, 128, 48, 48, False),
        (4, 128, 48, 64, True),
        (5, 64, 64, 64, False),
        (6, 64, 64, 96, True),
        (7, 32, 96, 96, False),
    )
    entries: list[InventoryEntry] = []
    for block_index, spatial_size, in_channels, out_channels, downsample in block_specs:
        prefix = f"encoder_blocks.{block_index}"
        output_spatial_size = spatial_size // 2 if downsample else spatial_size
        entries.extend(
            [
                _conv_entry(
                    f"{prefix}.main_conv1",
                    ConvEntrySpec(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=5,
                        spatial_size=spatial_size,
                        bias=False,
                        followed_by_norm=True,
                    ),
                ),
                _norm_entry(f"{prefix}.main_norm1", out_channels, spatial_size),
                _gate_entry(f"{prefix}.main_gate", out_channels, spatial_size),
            ],
        )
        if downsample:
            entries.append(
                _downsample_entry(
                    f"{prefix}.main_downsample",
                    out_channels,
                    spatial_size,
                ),
            )
        entries.extend(
            [
                _conv_entry(
                    f"{prefix}.main_conv2",
                    ConvEntrySpec(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=5,
                        spatial_size=output_spatial_size,
                        bias=False,
                        followed_by_norm=True,
                    ),
                ),
                _norm_entry(f"{prefix}.main_norm2", out_channels, output_spatial_size),
            ],
        )
        if downsample:
            entries.append(
                _downsample_entry(
                    f"{prefix}.skip_downsample",
                    in_channels,
                    spatial_size,
                ),
            )
        if downsample or in_channels != out_channels:
            entries.extend(
                [
                    _conv_entry(
                        f"{prefix}.skip_conv",
                        ConvEntrySpec(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=5,
                            spatial_size=output_spatial_size,
                            bias=False,
                            followed_by_norm=True,
                        ),
                    ),
                    _norm_entry(
                        f"{prefix}.skip_norm",
                        out_channels,
                        output_spatial_size,
                    ),
                ],
            )
        entries.append(
            _gate_entry(f"{prefix}.output_gate", out_channels, output_spatial_size),
        )
    return entries


def _decoder_entries() -> list[InventoryEntry]:
    block_specs = (
        (0, 32, 96, 96, False),
        (1, 32, 96, 96, False),
        (2, 32, 96, 64, True),
        (3, 64, 64, 64, False),
        (4, 64, 64, 48, True),
        (5, 128, 48, 48, False),
        (6, 128, 48, 32, True),
        (7, 256, 32, 32, False),
    )
    entries: list[InventoryEntry] = []
    for (
        block_index,
        input_spatial_size,
        in_channels,
        out_channels,
        upsample,
    ) in block_specs:
        prefix = f"decoder_blocks.{block_index}"
        output_spatial_size = input_spatial_size * 2 if upsample else input_spatial_size
        if upsample:
            entries.append(
                _upsample_entry(
                    f"{prefix}.main_upsample",
                    in_channels,
                    input_spatial_size,
                ),
            )
        entries.extend(
            [
                _conv_entry(
                    f"{prefix}.main_conv1",
                    ConvEntrySpec(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=5,
                        spatial_size=output_spatial_size,
                        bias=False,
                        followed_by_norm=True,
                    ),
                ),
                _norm_entry(f"{prefix}.main_norm1", out_channels, output_spatial_size),
                _gate_entry(f"{prefix}.main_gate", out_channels, output_spatial_size),
                _conv_entry(
                    f"{prefix}.main_conv2",
                    ConvEntrySpec(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=5,
                        spatial_size=output_spatial_size,
                        bias=False,
                        followed_by_norm=True,
                    ),
                ),
                _norm_entry(f"{prefix}.main_norm2", out_channels, output_spatial_size),
            ],
        )
        if upsample:
            entries.append(
                _upsample_entry(
                    f"{prefix}.skip_upsample",
                    in_channels,
                    input_spatial_size,
                ),
            )
        if upsample or in_channels != out_channels:
            entries.extend(
                [
                    _conv_entry(
                        f"{prefix}.skip_conv",
                        ConvEntrySpec(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=5,
                            spatial_size=output_spatial_size,
                            bias=False,
                            followed_by_norm=True,
                        ),
                    ),
                    _norm_entry(
                        f"{prefix}.skip_norm",
                        out_channels,
                        output_spatial_size,
                    ),
                ],
            )
        entries.append(
            _gate_entry(f"{prefix}.output_gate", out_channels, output_spatial_size),
        )
    return entries


def _conv_entry(
    module_id: str,
    spec: ConvEntrySpec,
) -> InventoryEntry:
    parameter_count = (
        spec.in_channels * spec.out_channels * spec.kernel_size * spec.kernel_size
    ) + (spec.out_channels if spec.bias else 0)
    padding = spec.kernel_size // 2
    taps = spec.kernel_size * spec.kernel_size
    macs_per_sample = (
        spec.in_channels
        * spec.out_channels
        * spec.kernel_size
        * spec.kernel_size
        * spec.spatial_size
        * spec.spatial_size
    )
    location = _module_location(module_id)
    return InventoryEntry(
        module_id=module_id,
        module_type="Conv2d",
        stage=location.stage,
        block=location.block,
        branch=location.branch,
        op_index=location.op_index,
        observed_forward_index=-1,
        input_shape=_shape(spec.in_channels, spec.spatial_size),
        output_shape=_shape(spec.out_channels, spec.spatial_size),
        kernel_size=f"{spec.kernel_size}x{spec.kernel_size}",
        stride="1x1",
        padding=f"{padding}x{padding}",
        groups=1,
        taps=taps,
        trainable=True,
        learned_parameter_count=parameter_count,
        macs_per_sample=macs_per_sample,
        activation_output_elements=(
            spec.out_channels * spec.spatial_size * spec.spatial_size
        ),
        in_channels=spec.in_channels,
        out_channels=spec.out_channels,
        has_bias=spec.bias,
        followed_by_norm=spec.followed_by_norm,
        gate_channels=0,
        resampling_kind="",
        count_category="learned_convolution",
        mac_formula="in_channels*out_channels*kernel_h*kernel_w*out_h*out_w",
    )


def _module_location(module_id: str) -> ModuleLocation:
    if module_id.startswith("encoder_blocks."):
        return _block_location(module_id, stage="encoder", prefix="enc")
    if module_id.startswith("decoder_blocks."):
        return _block_location(module_id, stage="decoder", prefix="dec")
    if module_id.startswith("stem_"):
        return ModuleLocation(
            stage="stem",
            block="stem",
            branch="stem",
            op_index=_stem_op_index(module_id),
        )
    if module_id in {"mu_head", "logvar_head"}:
        return ModuleLocation(
            stage="heads",
            block="vae_heads",
            branch="head",
            op_index=0 if module_id == "mu_head" else 1,
        )
    if module_id.startswith("latent_projection_"):
        return ModuleLocation(
            stage="latent_projection",
            block="latent_projection",
            branch="main",
            op_index=_latent_projection_op_index(module_id),
        )
    if module_id == "output_head":
        return ModuleLocation(
            stage="output",
            block="rgb_head",
            branch="head",
            op_index=0,
        )
    message = f"Unknown spec 0001 module path: {module_id}"
    raise ValueError(message)


def _block_location(module_id: str, *, stage: str, prefix: str) -> ModuleLocation:
    parts = module_id.split(".")
    if len(parts) != _BLOCK_MODULE_PATH_PARTS:
        message = f"Unexpected block module path: {module_id}"
        raise ValueError(message)
    block_index = int(parts[1])
    op_name = parts[2]
    return ModuleLocation(
        stage=stage,
        block=f"{prefix}{block_index}",
        branch=_branch_for_op(op_name),
        op_index=_block_op_index(op_name),
    )


def _branch_for_op(op_name: str) -> str:
    if op_name.startswith("main_"):
        return "main"
    if op_name.startswith("skip_"):
        return "skip"
    if op_name == "output_gate":
        return "post_add"
    message = f"Unexpected residual op name: {op_name}"
    raise ValueError(message)


def _block_op_index(op_name: str) -> int:
    op_order = {
        "main_upsample": 0,
        "main_conv1": 1,
        "main_norm1": 2,
        "main_gate": 3,
        "main_downsample": 4,
        "main_conv2": 5,
        "main_norm2": 6,
        "skip_upsample": 7,
        "skip_downsample": 8,
        "skip_conv": 9,
        "skip_norm": 10,
        "output_gate": 11,
    }
    return op_order[op_name]


def _stem_op_index(module_id: str) -> int:
    return {
        "stem_conv": 0,
        "stem_norm": 1,
        "stem_gate": 2,
    }[module_id]


def _latent_projection_op_index(module_id: str) -> int:
    return {
        "latent_projection_conv": 0,
        "latent_projection_norm": 1,
        "latent_projection_gate": 2,
    }[module_id]


def _shape(channels: int, spatial_size: int) -> str:
    return f"1x{channels}x{spatial_size}x{spatial_size}"


def _norm_entry(module_id: str, channels: int, spatial_size: int) -> InventoryEntry:
    location = _module_location(module_id)
    return InventoryEntry(
        module_id=module_id,
        module_type="GroupNorm",
        stage=location.stage,
        block=location.block,
        branch=location.branch,
        op_index=location.op_index,
        observed_forward_index=-1,
        input_shape=_shape(channels, spatial_size),
        output_shape=_shape(channels, spatial_size),
        kernel_size="",
        stride="",
        padding="",
        groups=DEFAULT_GROUPNORM_GROUPS,
        taps=0,
        trainable=True,
        learned_parameter_count=channels * 2,
        macs_per_sample=0,
        activation_output_elements=0,
        in_channels=channels,
        out_channels=channels,
        has_bias=True,
        followed_by_norm=False,
        gate_channels=0,
        resampling_kind="",
        count_category="groupnorm_affine",
        mac_formula="",
    )


def _gate_entry(module_id: str, channels: int, spatial_size: int) -> InventoryEntry:
    location = _module_location(module_id)
    return InventoryEntry(
        module_id=module_id,
        module_type="GatedScalarActivation",
        stage=location.stage,
        block=location.block,
        branch=location.branch,
        op_index=location.op_index,
        observed_forward_index=-1,
        input_shape=_shape(channels, spatial_size),
        output_shape=_shape(channels, spatial_size),
        kernel_size="",
        stride="",
        padding="",
        groups=1,
        taps=0,
        trainable=True,
        learned_parameter_count=channels * 2,
        macs_per_sample=0,
        activation_output_elements=0,
        in_channels=channels,
        out_channels=channels,
        has_bias=False,
        followed_by_norm=False,
        gate_channels=channels,
        resampling_kind="",
        count_category="learned_gate",
        mac_formula="",
    )


def _downsample_entry(
    module_id: str,
    channels: int,
    input_spatial_size: int,
) -> InventoryEntry:
    output_spatial_size = input_spatial_size // 2
    location = _module_location(module_id)
    return InventoryEntry(
        module_id=module_id,
        module_type="FixedBinomialLowpassDownsample2x",
        stage=location.stage,
        block=location.block,
        branch=location.branch,
        op_index=location.op_index,
        observed_forward_index=-1,
        input_shape=_shape(channels, input_spatial_size),
        output_shape=_shape(channels, output_spatial_size),
        kernel_size="5x5",
        stride="2x2",
        padding="2x2",
        groups=channels,
        taps=25,
        trainable=False,
        learned_parameter_count=0,
        macs_per_sample=channels * 25 * output_spatial_size * output_spatial_size,
        activation_output_elements=0,
        in_channels=channels,
        out_channels=channels,
        has_bias=False,
        followed_by_norm=False,
        gate_channels=0,
        resampling_kind="fixed_binomial_lowpass_5x5_stride2",
        count_category="fixed_resampling",
        mac_formula="channels*25*out_h*out_w",
    )


def _upsample_entry(
    module_id: str,
    channels: int,
    input_spatial_size: int,
) -> InventoryEntry:
    output_spatial_size = input_spatial_size * 2
    location = _module_location(module_id)
    return InventoryEntry(
        module_id=module_id,
        module_type="FieldwiseBilinearUpsample2x",
        stage=location.stage,
        block=location.block,
        branch=location.branch,
        op_index=location.op_index,
        observed_forward_index=-1,
        input_shape=_shape(channels, input_spatial_size),
        output_shape=_shape(channels, output_spatial_size),
        kernel_size="bilinear_4tap",
        stride="scale_factor_2",
        padding="",
        groups=channels,
        taps=4,
        trainable=False,
        learned_parameter_count=0,
        macs_per_sample=channels * 4 * output_spatial_size * output_spatial_size,
        activation_output_elements=0,
        in_channels=channels,
        out_channels=channels,
        has_bias=False,
        followed_by_norm=False,
        gate_channels=0,
        resampling_kind="bilinear_scale_factor_2_align_corners_false",
        count_category="fixed_resampling",
        mac_formula="channels*4*out_h*out_w",
    )


def _verify_inventory_paths(
    *,
    module_by_path: Mapping[str, nn.Module],
    entries: Iterable[InventoryEntry],
) -> None:
    type_map: Mapping[str, type[nn.Module]] = {
        "Conv2d": nn.Conv2d,
        "GroupNorm": nn.GroupNorm,
        "GatedScalarActivation": GatedScalarActivation,
        "FixedBinomialLowpassDownsample2x": FixedBinomialLowpassDownsample2x,
        "FieldwiseBilinearUpsample2x": FieldwiseBilinearUpsample2x,
    }
    for entry in entries:
        module = module_by_path.get(entry.module_id)
        if module is None:
            message = f"Missing module in instantiated model: {entry.module_id}"
            raise RuntimeError(message)
        expected_type = type_map[entry.module_type]
        if not isinstance(module, expected_type):
            message = (
                f"Module {entry.module_id} has type {type(module).__name__}; "
                f"expected {entry.module_type}"
            )
            raise TypeError(message)


def _observe_model_shapes(
    *,
    model: NonEquivariantVAE,
    expected_entries: list[InventoryEntry],
) -> dict[str, ModuleObservation]:
    expected_paths = {entry.module_id for entry in expected_entries}
    shape_model = copy.deepcopy(model)
    shape_model.to(device=torch.device("meta"))
    observations: dict[str, ModuleObservation] = {}
    forward_index = 0
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(
        module_path: str,
    ) -> Callable[[nn.Module, tuple[object, ...], object], None]:
        def hook(
            _module: nn.Module,
            inputs: tuple[object, ...],
            output: object,
        ) -> None:
            nonlocal forward_index
            input_tensor = _first_tensor(inputs)
            output_tensor = _first_tensor(output)
            observations[module_path] = ModuleObservation(
                input_shape=_tensor_shape(input_tensor),
                output_shape=_tensor_shape(output_tensor),
                observed_forward_index=forward_index,
            )
            forward_index += 1

        return hook

    for module_path, module in _named_modules(shape_model):
        if module_path in expected_paths:
            handles.append(module.register_forward_hook(make_hook(module_path)))
    try:
        with torch.no_grad():
            shape_model.forward(
                torch.empty(
                    SPEC0001_MODEL_COUNT_INPUT_SHAPE,
                    device=torch.device("meta"),
                ),
            )
    finally:
        for handle in handles:
            handle.remove()

    missing_paths = sorted(expected_paths.difference(observations))
    if missing_paths:
        message = f"Inventory modules were not executed: {missing_paths}"
        raise RuntimeError(message)
    return observations


def _observed_entry(
    *,
    expected: InventoryEntry,
    module: nn.Module,
    observation: ModuleObservation,
) -> InventoryEntry:
    if isinstance(module, nn.Conv2d):
        return _observed_conv_entry(
            expected=expected,
            module=module,
            observation=observation,
        )
    if isinstance(module, nn.GroupNorm):
        return _observed_norm_entry(
            expected=expected,
            module=module,
            observation=observation,
        )
    if isinstance(module, GatedScalarActivation):
        return _observed_gate_entry(
            expected=expected,
            module=module,
            observation=observation,
        )
    if isinstance(module, FixedBinomialLowpassDownsample2x):
        return _observed_downsample_entry(
            expected=expected,
            module=module,
            observation=observation,
        )
    if isinstance(module, FieldwiseBilinearUpsample2x):
        return _observed_upsample_entry(
            expected=expected,
            module=module,
            observation=observation,
        )
    message = f"Unsupported inventory module type: {type(module).__name__}"
    raise TypeError(message)


def _observed_conv_entry(
    *,
    expected: InventoryEntry,
    module: nn.Conv2d,
    observation: ModuleObservation,
) -> InventoryEntry:
    kernel_h, kernel_w = module.kernel_size
    stride_h, stride_w = module.stride
    padding_h, padding_w = module.padding
    out_h, out_w = _spatial_dims(observation.output_shape)
    in_channels = module.in_channels
    out_channels = module.out_channels
    activation_elements = _activation_output_elements(observation.output_shape)
    macs_per_sample = (
        (in_channels // module.groups)
        * out_channels
        * kernel_h
        * kernel_w
        * out_h
        * out_w
    )
    return _with_observed_common(
        expected=expected,
        observation=observation,
        details=ObservedEntryDetails(
            module_type="Conv2d",
            input_shape=_shape_text(observation.input_shape),
            output_shape=_shape_text(observation.output_shape),
            kernel_size=f"{kernel_h}x{kernel_w}",
            stride=f"{stride_h}x{stride_w}",
            padding=f"{padding_h}x{padding_w}",
            groups=module.groups,
            taps=kernel_h * kernel_w,
            trainable=_has_trainable_direct_parameters(module),
            learned_parameter_count=_direct_trainable_parameter_count(module),
            macs_per_sample=macs_per_sample,
            activation_output_elements=activation_elements,
            in_channels=in_channels,
            out_channels=out_channels,
            has_bias=module.bias is not None,
            followed_by_norm=expected.followed_by_norm,
            gate_channels=0,
            resampling_kind="",
            count_category="learned_convolution",
            mac_formula="in_channels*out_channels*kernel_h*kernel_w*out_h*out_w",
        ),
    )


def _observed_norm_entry(
    *,
    expected: InventoryEntry,
    module: nn.GroupNorm,
    observation: ModuleObservation,
) -> InventoryEntry:
    return _with_observed_common(
        expected=expected,
        observation=observation,
        details=ObservedEntryDetails(
            module_type="GroupNorm",
            input_shape=_shape_text(observation.input_shape),
            output_shape=_shape_text(observation.output_shape),
            kernel_size="",
            stride="",
            padding="",
            groups=module.num_groups,
            taps=0,
            trainable=_has_trainable_direct_parameters(module),
            learned_parameter_count=_direct_trainable_parameter_count(module),
            macs_per_sample=0,
            activation_output_elements=0,
            in_channels=module.num_channels,
            out_channels=module.num_channels,
            has_bias=module.affine,
            followed_by_norm=False,
            gate_channels=0,
            resampling_kind="",
            count_category="groupnorm_affine",
            mac_formula="",
        ),
    )


def _observed_gate_entry(
    *,
    expected: InventoryEntry,
    module: GatedScalarActivation,
    observation: ModuleObservation,
) -> InventoryEntry:
    return _with_observed_common(
        expected=expected,
        observation=observation,
        details=ObservedEntryDetails(
            module_type="GatedScalarActivation",
            input_shape=_shape_text(observation.input_shape),
            output_shape=_shape_text(observation.output_shape),
            kernel_size="",
            stride="",
            padding="",
            groups=1,
            taps=0,
            trainable=_has_trainable_direct_parameters(module),
            learned_parameter_count=_direct_trainable_parameter_count(module),
            macs_per_sample=0,
            activation_output_elements=0,
            in_channels=module.channels,
            out_channels=module.channels,
            has_bias=False,
            followed_by_norm=False,
            gate_channels=module.channels,
            resampling_kind="",
            count_category="learned_gate",
            mac_formula="",
        ),
    )


def _observed_downsample_entry(
    *,
    expected: InventoryEntry,
    module: FixedBinomialLowpassDownsample2x,
    observation: ModuleObservation,
) -> InventoryEntry:
    out_h, out_w = _spatial_dims(observation.output_shape)
    kernel = cast("torch.Tensor", module.kernel)
    taps = kernel.shape[-2] * kernel.shape[-1]
    return _with_observed_common(
        expected=expected,
        observation=observation,
        details=ObservedEntryDetails(
            module_type="FixedBinomialLowpassDownsample2x",
            input_shape=_shape_text(observation.input_shape),
            output_shape=_shape_text(observation.output_shape),
            kernel_size=f"{kernel.shape[-2]}x{kernel.shape[-1]}",
            stride="2x2",
            padding="2x2",
            groups=module.channels,
            taps=taps,
            trainable=False,
            learned_parameter_count=0,
            macs_per_sample=module.channels * taps * out_h * out_w,
            activation_output_elements=0,
            in_channels=module.channels,
            out_channels=module.channels,
            has_bias=False,
            followed_by_norm=False,
            gate_channels=0,
            resampling_kind="fixed_binomial_lowpass_5x5_stride2",
            count_category="fixed_resampling",
            mac_formula="channels*25*out_h*out_w",
        ),
    )


def _observed_upsample_entry(
    *,
    expected: InventoryEntry,
    module: FieldwiseBilinearUpsample2x,
    observation: ModuleObservation,
) -> InventoryEntry:
    out_h, out_w = _spatial_dims(observation.output_shape)
    taps = 4
    return _with_observed_common(
        expected=expected,
        observation=observation,
        details=ObservedEntryDetails(
            module_type="FieldwiseBilinearUpsample2x",
            input_shape=_shape_text(observation.input_shape),
            output_shape=_shape_text(observation.output_shape),
            kernel_size="bilinear_4tap",
            stride="scale_factor_2",
            padding="",
            groups=module.channels,
            taps=taps,
            trainable=False,
            learned_parameter_count=0,
            macs_per_sample=module.channels * taps * out_h * out_w,
            activation_output_elements=0,
            in_channels=module.channels,
            out_channels=module.channels,
            has_bias=False,
            followed_by_norm=False,
            gate_channels=0,
            resampling_kind="bilinear_scale_factor_2_align_corners_false",
            count_category="fixed_resampling",
            mac_formula="channels*4*out_h*out_w",
        ),
    )


def _with_observed_common(
    *,
    expected: InventoryEntry,
    observation: ModuleObservation,
    details: ObservedEntryDetails,
) -> InventoryEntry:
    return InventoryEntry(
        module_id=expected.module_id,
        module_type=details.module_type,
        stage=expected.stage,
        block=expected.block,
        branch=expected.branch,
        op_index=expected.op_index,
        observed_forward_index=observation.observed_forward_index,
        input_shape=details.input_shape,
        output_shape=details.output_shape,
        kernel_size=details.kernel_size,
        stride=details.stride,
        padding=details.padding,
        groups=details.groups,
        taps=details.taps,
        trainable=details.trainable,
        learned_parameter_count=details.learned_parameter_count,
        macs_per_sample=details.macs_per_sample,
        activation_output_elements=details.activation_output_elements,
        in_channels=details.in_channels,
        out_channels=details.out_channels,
        has_bias=details.has_bias,
        followed_by_norm=details.followed_by_norm,
        gate_channels=details.gate_channels,
        resampling_kind=details.resampling_kind,
        count_category=details.count_category,
        mac_formula=details.mac_formula,
    )


def _inventory_mismatches(
    *,
    expected_entries: list[InventoryEntry],
    observed_entries: list[InventoryEntry],
) -> list[str]:
    mismatches: list[str] = []
    expected_rows = [_comparable_csv_row(entry) for entry in expected_entries]
    observed_rows = [_comparable_csv_row(entry) for entry in observed_entries]
    for expected_row, observed_row in zip(expected_rows, observed_rows, strict=True):
        module_id = expected_row["module_id"]
        for key, expected_value in expected_row.items():
            observed_value = observed_row[key]
            if observed_value != expected_value:
                mismatches.append(
                    (
                        f"{module_id}.{key}: expected {expected_value}, "
                        f"got {observed_value}"
                    ),
                )
    return mismatches


def _unexpected_inventory_leaf_paths(
    *,
    module_by_path: Mapping[str, nn.Module],
    expected_entries: Iterable[InventoryEntry],
) -> list[str]:
    expected_paths = {entry.module_id for entry in expected_entries}
    return sorted(
        module_path
        for module_path, module in module_by_path.items()
        if module_path
        and module_path not in expected_paths
        and not _has_child_modules(module)
        and _is_allowed_leaf_module(module)
    )


def _comparable_csv_row(entry: InventoryEntry) -> dict[str, str]:
    row = dict(entry.as_csv_row())
    del row["observed_forward_index"]
    return row


def _forward_order_verified(
    *,
    expected_entries: list[InventoryEntry],
    observed_entries: list[InventoryEntry],
) -> bool:
    expected_order = [entry.module_id for entry in expected_entries]
    observed_order = [
        entry.module_id
        for entry in sorted(
            observed_entries,
            key=lambda entry: entry.observed_forward_index,
        )
    ]
    return observed_order == expected_order


def _count_inventory(
    *,
    entries: list[InventoryEntry],
    model: NonEquivariantVAE,
) -> ObservedCounts:
    learned_convolution_entries = _entries_for(entries, "learned_convolution")
    norm_entries = _entries_for(entries, "groupnorm_affine")
    gate_entries = _entries_for(entries, "learned_gate")
    resampling_entries = _entries_for(entries, "fixed_resampling")
    return ObservedCounts(
        learned_convolution_count=len(learned_convolution_entries),
        normalization_module_count=len(norm_entries),
        gate_module_count=len(gate_entries),
        fixed_resampling_op_count=len(resampling_entries),
        learned_convolution_parameters=sum(
            entry.learned_parameter_count for entry in learned_convolution_entries
        ),
        groupnorm_affine_parameters=sum(
            entry.learned_parameter_count for entry in norm_entries
        ),
        learned_gate_parameters=sum(
            entry.learned_parameter_count for entry in gate_entries
        ),
        total_learned_parameters=sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        learned_convolution_macs_per_sample=sum(
            entry.macs_per_sample for entry in learned_convolution_entries
        ),
        fixed_resampling_macs_per_sample=sum(
            entry.macs_per_sample for entry in resampling_entries
        ),
        total_macs_per_sample_with_fixed_resampling=(
            sum(entry.macs_per_sample for entry in learned_convolution_entries)
            + sum(entry.macs_per_sample for entry in resampling_entries)
        ),
        activation_output_elements_per_sample=sum(
            entry.activation_output_elements for entry in learned_convolution_entries
        ),
    )


def _entries_for(
    entries: Iterable[InventoryEntry],
    counts_toward: str,
) -> list[InventoryEntry]:
    return [entry for entry in entries if entry.count_category == counts_toward]


def _matches_spec_target(observed: ObservedCounts) -> bool:
    target = SPEC0001_MODEL_COUNT_TARGET
    observed_payload = observed.as_flat_payload()
    expected_payload = {key: target[key] for key in observed_payload}
    return observed_payload == expected_payload


def _expected_payload() -> JsonObject:
    return {
        "total_learned_parameters": SPEC0001_MODEL_COUNT_TARGET[
            "total_learned_parameters"
        ],
        "learned_convolution_macs_per_sample": SPEC0001_MODEL_COUNT_TARGET[
            "learned_convolution_macs_per_sample"
        ],
        "fixed_resampling_macs_per_sample": SPEC0001_MODEL_COUNT_TARGET[
            "fixed_resampling_macs_per_sample"
        ],
    }


def _zero_initialized_rgb_head_verified(model: NonEquivariantVAE) -> bool:
    weight_is_zero = bool(torch.count_nonzero(model.output_head.weight).item() == 0)
    bias_is_zero = model.output_head.bias is not None and bool(
        torch.count_nonzero(model.output_head.bias).item() == 0,
    )
    return weight_is_zero and bias_is_zero


def _banned_operations_checked(model: NonEquivariantVAE) -> bool:
    return all(
        _module_allowed_for_spec0001_count(name=name, module=module)
        for name, module in _named_modules(model)
    )


def _module_allowed_for_spec0001_count(*, name: str, module: nn.Module) -> bool:
    if _is_banned_module_type(module):
        return False
    if name and not _has_child_modules(module) and not _is_allowed_leaf_module(module):
        return False
    allowed = True
    if isinstance(module, nn.Conv2d):
        allowed = _conv2d_allowed_for_count(name=name, module=module)
    elif isinstance(module, nn.GroupNorm):
        allowed = _groupnorm_allowed_for_count(module)
    elif isinstance(module, GatedScalarActivation):
        allowed = _gate_allowed_for_count(module)
    elif isinstance(module, FixedBinomialLowpassDownsample2x):
        allowed = _downsample_allowed_for_count(module)
    elif isinstance(module, FieldwiseBilinearUpsample2x):
        allowed = module.channels > 0
    return allowed


def _is_banned_module_type(module: nn.Module) -> bool:
    return isinstance(
        module,
        (
            nn.AdaptiveAvgPool2d,
            nn.AvgPool2d,
            nn.BatchNorm2d,
            nn.ConvTranspose2d,
            nn.Dropout,
            nn.Dropout2d,
            nn.Identity,
            nn.LayerNorm,
            nn.MaxPool2d,
            nn.PixelShuffle,
            nn.Upsample,
        ),
    )


def _has_child_modules(module: nn.Module) -> bool:
    return any(module.children())


def _is_allowed_leaf_module(module: nn.Module) -> bool:
    return isinstance(
        module,
        (
            nn.Conv2d,
            nn.GroupNorm,
            GatedScalarActivation,
            FixedBinomialLowpassDownsample2x,
            FieldwiseBilinearUpsample2x,
        ),
    )


def _conv2d_allowed_for_count(*, name: str, module: nn.Conv2d) -> bool:
    expected_kernel = (7, 7) if name == "stem_conv" else (5, 5)
    expected_padding = (module.kernel_size[0] // 2, module.kernel_size[1] // 2)
    return (
        module.groups == 1
        and module.kernel_size == expected_kernel
        and module.stride == (1, 1)
        and module.padding == expected_padding
        and module.dilation == (1, 1)
        and module.padding_mode == "zeros"
    )


def _groupnorm_allowed_for_count(module: nn.GroupNorm) -> bool:
    if module.num_groups != DEFAULT_GROUPNORM_GROUPS:
        return False
    return module.affine


def _gate_allowed_for_count(module: GatedScalarActivation) -> bool:
    return module.a.shape == (module.channels,) and module.b.shape == (module.channels,)


def _downsample_allowed_for_count(
    module: FixedBinomialLowpassDownsample2x,
) -> bool:
    kernel = cast("torch.Tensor", module.kernel)
    return module.channels > 0 and tuple(kernel.shape) == (1, 1, 5, 5)


def _read_norm_groups(config: Mapping[str, object], *, config_path: Path) -> int:
    model_config = _required_mapping(config, "model")
    if model_config.get("implementation_status") == "count_schema_only":
        message = (
            "Pass-mode model-count verification rejects configs whose "
            "`model.implementation_status` is still `count_schema_only`."
        )
        raise ValueError(message)
    if model_config.get("architecture_id") != "spec0001_non_eq_vae_translatable":
        message = "Missing spec0001 architecture_id in model config."
        raise ValueError(message)
    if model_config.get("topology_version") != "spec0001.count.v1":
        message = "Missing spec0001 topology_version in model config."
        raise ValueError(message)
    normalization = _required_mapping(model_config, "normalization")
    if normalization.get("num_groups") != DEFAULT_GROUPNORM_GROUPS:
        message = (
            "Spec 0001 model-count verification currently requires "
            f"`num_groups = {DEFAULT_GROUPNORM_GROUPS}` in {config_path}."
        )
        raise ValueError(message)
    return DEFAULT_GROUPNORM_GROUPS


def _required_mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, dict):
        message = f"Expected object field `{key}` in config."
        raise TypeError(message)
    return cast("Mapping[str, object]", value)


def _required_string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        message = f"Expected string field `{key}` in config."
        raise TypeError(message)
    return value


def _artifact_reference(output_path: Path, artifact_path: Path) -> str:
    if output_path.parent.name == "benchmark":
        return f"benchmark/{artifact_path.name}"
    return str(artifact_path)


def _named_modules(model: nn.Module) -> Iterable[tuple[str, nn.Module]]:
    return cast("Iterable[tuple[str, nn.Module]]", model.named_modules())


def _json_string_list(values: Iterable[str]) -> list[JsonValue]:
    return cast("list[JsonValue]", list(values))


def _first_tensor(value: object) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, tuple):
        tuple_value = cast("tuple[object, ...]", value)
        for item in tuple_value:
            if isinstance(item, torch.Tensor):
                return item
    message = "Expected tensor in hook value."
    raise TypeError(message)


def _tensor_shape(tensor: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(dimension) for dimension in tensor.shape)


def _shape_text(shape: tuple[int, ...]) -> str:
    return "x".join(str(dimension) for dimension in shape)


def _spatial_dims(shape: tuple[int, ...]) -> tuple[int, int]:
    if len(shape) == _BCHW_RANK:
        return shape[_HEIGHT_DIMENSION], shape[_WIDTH_DIMENSION]
    message = f"Expected BCHW tensor shape, got {_shape_text(shape)}"
    raise ValueError(message)


def _activation_output_elements(shape: tuple[int, ...]) -> int:
    if len(shape) != _BCHW_RANK:
        message = f"Expected BCHW tensor shape, got {_shape_text(shape)}"
        raise ValueError(message)
    return (
        shape[_CHANNEL_DIMENSION] * shape[_HEIGHT_DIMENSION] * shape[_WIDTH_DIMENSION]
    )


def _direct_trainable_parameter_count(module: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters(recurse=False)
        if parameter.requires_grad
    )


def _has_trainable_direct_parameters(module: nn.Module) -> bool:
    return _direct_trainable_parameter_count(module) > 0


def _bool_csv(*, value: bool) -> str:
    return "true" if value else "false"


__all__ = [
    "MODEL_INVENTORY_COLUMNS",
    "SPEC0001_MODEL_COUNT_TARGET",
    "build_model_count_payload",
    "write_model_count",
]
