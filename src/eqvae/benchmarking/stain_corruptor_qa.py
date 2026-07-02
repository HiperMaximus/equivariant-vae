# Copyright 2026 HiperMaximus
"""Local non-promotable QA artifact for the HED stain corruptor."""

from __future__ import annotations

import hashlib
import struct
import zlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor

from eqvae.benchmarking.io import write_json
from eqvae.config import resolve_json_config
from eqvae.corruption.stain import (
    CORRUPTION_VERSION,
    HED_FROM_RGB,
    OD_EPSILON,
    RGB_FROM_HED,
    SCIKIT_IMAGE_ORACLE_VERSION,
    SEMANTIC_SEED_FIELDS,
    StainCorruptionResult,
    StainCorruptor,
    clean_validation_passthrough,
    corrupt_normalized_batch,
    hed_to_rgb,
    profile_from_config,
    rgb_to_hed,
    semantic_sample_key,
)
from eqvae.data.synthetic import (
    SyntheticPatchSpec,
    make_synthetic_patches,
    synthetic_patch_records,
)

if TYPE_CHECKING:
    from pathlib import Path

    from eqvae.benchmarking.io import JsonObject

LOCAL_STAIN_QA_KIND = "local_synthetic_stain_corruptor_qa"
LOCAL_STAIN_QA_SOURCE = "local_cpu_synthetic_stain_corruptor_qa"
STAIN_QA_SCHEMA_VERSION = "spec0001.stain_corruptor_qa.v1"
STAIN_QA_CORRUPTION_VIEW = "train_corrupted_local_qa"
STAIN_QA_SPLIT = "train"
STAIN_QA_COUNT = 25
GRID_COLUMNS = 5
QA_VARIANT_COUNT = 4
ORACLE_TOLERANCE = 1.0e-6
HWC_NDIM = 3
HWC_RGB_CHANNELS = 3

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_ORACLE_RGB_FIXTURE = (
    ((0.2, 0.4, 0.8), (0.9, 0.7, 0.3)),
    ((1.0, 1.0, 1.0), (1.0e-6, 0.5, 0.25)),
)
_ORACLE_HED_FROM_RGB = (
    ((0.20468254940785954, 0.0, 0.0), (0.0, 0.0, 0.12939278335441723)),
    ((0.0, 0.0, 0.0), (1.814278629235823, 0.0, 0.0)),
)
_ORACLE_HED_FIXTURE = (
    ((0.0, 0.05, 0.02), (0.1, 0.0, 0.03)),
    ((0.2, 0.1, 0.0), (0.4, 0.05, 0.02)),
)
_ORACLE_RGB_FROM_HED = (
    (
        (0.884300763535516, 0.4311218160437418, 0.7471365798284315),
        (0.36425036866476335, 0.30019261222416926, 0.4848417387084463),
    ),
    (
        (0.1506607066186742, 0.03681289736425315, 0.3854783576657718),
        (0.02435566545313374, 0.00900741142186956, 0.15045270487597587),
    ),
)


@dataclass(frozen=True)
class LocalStainCorruptorQaRequest:
    """Inputs for the local stain-corruptor QA artifact."""

    config_path: Path
    output_dir: Path
    run_name: str


def write_local_stain_corruptor_qa(  # noqa: PLR0914
    request: LocalStainCorruptorQaRequest,
) -> Path:
    """Write local synthetic HED corruptor QA JSON and PNG artifacts.

    Returns:
        Path to `benchmark/stain_corruptor_qa.json`.

    """
    resolved = resolve_json_config(request.config_path)
    effective_config = resolved.effective_config
    seeds = _required_object(effective_config, "seeds")
    data_config = _required_object(effective_config, "data")
    corruption_config = _required_object(effective_config, "corruption")
    profile = profile_from_config(corruption_config)
    image_size = _required_int(data_config, "image_size")
    data_seed = _required_int(seeds, "data_seed")
    corruption_seed = _required_int(seeds, "corruption_seed")

    clean = _synthetic_clean_batch(image_size=image_size, seed=data_seed)
    sample_keys = _synthetic_semantic_keys()
    target = clean.clone()
    corruptor = StainCorruptor()
    result = corrupt_normalized_batch(
        clean,
        profile=profile,
        corruption_seed=corruption_seed,
        split=STAIN_QA_SPLIT,
        semantic_sample_keys=sample_keys,
        corruption_step=0,
        corruption_view=STAIN_QA_CORRUPTION_VIEW,
        corruptor=corruptor,
    )
    visual_dir = request.output_dir / "artifacts" / "stain_corruptor_qa"
    visual_path = visual_dir / "synthetic_grid.png"
    _write_variant_grid_png(
        path=visual_path,
        variants=(
            clean,
            result.stain_only,
            result.gaussian_only,
            result.combined,
        ),
    )
    payload = _payload(
        request=request,
        effective_config=effective_config,
        invoked_config_hash=resolved.invoked_config_hash,
        effective_config_hash=resolved.effective_config_hash,
        source_config_chain=[
            source_config.as_json() for source_config in resolved.source_config_chain
        ],
        profile_payload=profile.as_json(),
        corruption_seed=corruption_seed,
        clean=clean,
        target=target,
        result=result,
        visual_path=visual_path,
    )
    output_path = request.output_dir / "benchmark" / "stain_corruptor_qa.json"
    write_json(output_path, payload)
    return output_path


def _payload(  # noqa: PLR0913
    *,
    request: LocalStainCorruptorQaRequest,
    effective_config: JsonObject,
    invoked_config_hash: str,
    effective_config_hash: str,
    source_config_chain: list[JsonObject],
    profile_payload: JsonObject,
    corruption_seed: int,
    clean: Tensor,
    target: Tensor,
    result: StainCorruptionResult,
    visual_path: Path,
) -> JsonObject:
    oracle_checks = _oracle_fixture_checks()
    output_range_pass = bool(
        result.corrupted.min().item() >= -1.0 and result.corrupted.max().item() <= 1.0,
    )
    clean_validation_rng_advanced = _clean_validation_advances_rng(clean)
    visual_sha256 = _sha256_file(visual_path)
    visual_relative = visual_path.relative_to(request.output_dir)
    applied_count = sum(1 for item in result.metadata if item.applied)
    checks: JsonObject = {
        **oracle_checks,
        "finite_pass": bool(torch.isfinite(result.corrupted).all().item()),
        "output_range_pass": output_range_pass,
        "shape_preserved": list(result.corrupted.shape) == list(clean.shape),
        "dtype_preserved": str(result.corrupted.dtype) == str(clean.dtype),
        "target_preserved": bool(torch.equal(target, clean)),
        "clean_validation_rng_advanced": clean_validation_rng_advanced,
        "applied_count": applied_count,
        "sample_count": int(clean.shape[0]),
    }
    return cast(
        "JsonObject",
        {
            "schema_version": STAIN_QA_SCHEMA_VERSION,
            "status": "local_pass" if _checks_pass(checks) else "fail",
            "benchmark_kind": LOCAL_STAIN_QA_KIND,
            "benchmark_source": LOCAL_STAIN_QA_SOURCE,
            "full_run_eligible": False,
            "run_name": request.run_name,
            "config": {
                "path": str(request.config_path),
                "invoked_config_hash": invoked_config_hash,
                "effective_config_hash": effective_config_hash,
                "source_config_chain": source_config_chain,
            },
            "corruption_version": CORRUPTION_VERSION,
            "profile_name": _required_string(profile_payload, "name"),
            "reference_oracle": {
                "name": "scikit-image",
                "version": SCIKIT_IMAGE_ORACLE_VERSION,
                "source_url": (
                    "https://github.com/scikit-image/scikit-image/blob/v0.26.0/"
                    "skimage/color/colorconv.py"
                ),
                "runtime_dependency": False,
                "runtime_code_imports_scikit_image": False,
                "fixture_source": "checked-in scikit-image 0.26.0 oracle values",
            },
            "api_contract": {
                "input_shape": list(clean.shape),
                "output_shape": list(result.corrupted.shape),
                "input_domain": "normalized_rgb_minus1_1",
                "output_domain": "normalized_rgb_minus1_1",
                "channel_order": "NCHW_RGB",
                "dtype": str(clean.dtype),
                "target_preservation": "x_clean_unchanged",
                "mask_handling": "masks_not_modified_by_corruptor",
            },
            "hed_convention": {
                "rgb_from_hed": _matrix_json(RGB_FROM_HED),
                "hed_from_rgb": _matrix_json(HED_FROM_RGB),
                "od_epsilon": OD_EPSILON,
                "uses_srgb_gamma_decode": False,
                "channel_first_multiplication": "torch.einsum('bchw,cd->bdhw')",
                "arbitrary_rgb_roundtrip": (
                    "not_required_after_rgb2hed_clamps_negative_stain_channels"
                ),
            },
            "rng": {
                "policy": "semantic_stateless_v1",
                "corruption_seed": corruption_seed,
                "semantic_seed_fields": list(SEMANTIC_SEED_FIELDS),
                "rank_in_semantic_seed": False,
                "corruption_step": 0,
                "corruption_view": STAIN_QA_CORRUPTION_VIEW,
                "clean_validation_consumes_rng": False,
            },
            "profile": profile_payload,
            "checks": checks,
            "summary": {
                "clean": _tensor_summary(clean),
                "corrupted": _tensor_summary(result.corrupted),
                "stain_only": _tensor_summary(result.stain_only),
                "gaussian_only": _tensor_summary(result.gaussian_only),
                "combined": _tensor_summary(result.combined),
            },
            "clamp_fractions": _clamp_fractions(result),
            "sample_metadata": [
                item.as_json() for item in result.metadata[:STAIN_QA_COUNT]
            ],
            "visual_artifacts": {
                "directory": "artifacts/stain_corruptor_qa",
                "synthetic_grid_path": str(visual_relative),
                "synthetic_grid_sha256": visual_sha256,
                "grid_order": ["clean", "stain_only", "gaussian_only", "combined"],
                "fixed_real_25_status": "committed",
            },
            "effective_config_snapshot": {
                "schema_version": _optional_string(effective_config, "schema_version"),
                "status": _optional_string(effective_config, "status"),
            },
        },
    )


def _synthetic_clean_batch(*, image_size: int, seed: int) -> Tensor:
    patches = make_synthetic_patches(
        SyntheticPatchSpec(
            count=STAIN_QA_COUNT,
            image_size=image_size,
            channels=3,
            seed=seed,
        ),
    )
    return (patches.to(dtype=torch.float32) / 127.5) - 1.0


def _synthetic_semantic_keys() -> tuple[str, ...]:
    records = synthetic_patch_records(
        SyntheticPatchSpec(count=STAIN_QA_COUNT, image_size=1, channels=3),
    )
    return tuple(
        semantic_sample_key(
            split=STAIN_QA_SPLIT,
            wsi_id=record.wsi_id,
            label=record.label,
            x=record.x,
            y=record.y,
        )
        for record in records
    )


def _oracle_fixture_checks() -> JsonObject:
    rgb_hwc = torch.tensor(_ORACLE_RGB_FIXTURE, dtype=torch.float64)
    hed_expected_hwc = torch.tensor(_ORACLE_HED_FROM_RGB, dtype=torch.float64)
    hed_hwc = torch.tensor(_ORACLE_HED_FIXTURE, dtype=torch.float64)
    rgb_expected_hwc = torch.tensor(_ORACLE_RGB_FROM_HED, dtype=torch.float64)
    rgb_nchw = rgb_hwc.permute(2, 0, 1).unsqueeze(0)
    hed_nchw = hed_hwc.permute(2, 0, 1).unsqueeze(0)
    hed_actual_hwc = rgb_to_hed(rgb_nchw).squeeze(0).permute(1, 2, 0)
    rgb_actual_hwc = hed_to_rgb(hed_nchw).squeeze(0).permute(1, 2, 0)
    return {
        "oracle_rgb2hed_max_abs_error": _max_abs_error(
            hed_actual_hwc,
            hed_expected_hwc,
        ),
        "oracle_hed2rgb_max_abs_error": _max_abs_error(
            rgb_actual_hwc,
            rgb_expected_hwc,
        ),
    }


def _max_abs_error(left: Tensor, right: Tensor) -> float:
    return float((left - right).abs().max().item())


def _clean_validation_advances_rng(clean: Tensor) -> bool:
    state = torch.get_rng_state()
    output = clean_validation_passthrough(clean)
    advanced = not torch.equal(state, torch.get_rng_state())
    if output is not clean:
        return True
    return advanced


def _checks_pass(checks: JsonObject) -> bool:
    required_true = (
        "finite_pass",
        "output_range_pass",
        "shape_preserved",
        "dtype_preserved",
        "target_preserved",
    )
    if any(checks.get(key) is not True for key in required_true):
        return False
    if checks.get("clean_validation_rng_advanced") is not False:
        return False
    rgb_error = _required_float(checks, "oracle_rgb2hed_max_abs_error")
    hed_error = _required_float(checks, "oracle_hed2rgb_max_abs_error")
    return rgb_error <= ORACLE_TOLERANCE and hed_error <= ORACLE_TOLERANCE


def _tensor_summary(tensor: Tensor) -> JsonObject:
    work = tensor.detach().cpu().to(dtype=torch.float32)
    return {
        "min": float(work.min().item()),
        "max": float(work.max().item()),
        "mean": float(work.mean().item()),
        "std": float(work.std(unbiased=False).item()),
    }


def _clamp_fractions(result: StainCorruptionResult) -> JsonObject:
    return {
        "corrupted_low": _fraction(result.corrupted <= -1.0),
        "corrupted_high": _fraction(result.corrupted >= 1.0),
        "stain_only_low": _fraction(result.stain_only <= -1.0),
        "stain_only_high": _fraction(result.stain_only >= 1.0),
        "gaussian_only_low": _fraction(result.gaussian_only <= -1.0),
        "gaussian_only_high": _fraction(result.gaussian_only >= 1.0),
        "combined_low": _fraction(result.combined <= -1.0),
        "combined_high": _fraction(result.combined >= 1.0),
    }


def _fraction(mask: Tensor) -> float:
    return float(mask.to(dtype=torch.float32).mean().item())


def _write_variant_grid_png(
    *,
    path: Path,
    variants: tuple[Tensor, Tensor, Tensor, Tensor],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(variants) != QA_VARIANT_COUNT:
        message = "Expected four visual QA variants"
        raise ValueError(message)
    batch, _channels, height, width = variants[0].shape
    if batch != STAIN_QA_COUNT:
        message = f"Expected {STAIN_QA_COUNT} QA patches, got {batch}"
        raise ValueError(message)
    grid_rows_per_variant = _ceil_div(STAIN_QA_COUNT, GRID_COLUMNS)
    canvas = torch.zeros(
        (len(variants) * grid_rows_per_variant * height, GRID_COLUMNS * width, 3),
        dtype=torch.uint8,
    )
    for variant_index, variant in enumerate(variants):
        tiles = _normalized_to_uint8_hwc(variant)
        for sample_index in range(STAIN_QA_COUNT):
            row = (variant_index * grid_rows_per_variant) + (
                sample_index // GRID_COLUMNS
            )
            column = sample_index % GRID_COLUMNS
            y0 = row * height
            x0 = column * width
            canvas[y0 : y0 + height, x0 : x0 + width, :] = tiles[sample_index]
    _write_rgb_png(path, canvas)


def _ceil_div(value: int, divisor: int) -> int:
    if value <= 0 or divisor <= 0:
        message = "ceil division inputs must be positive"
        raise ValueError(message)
    return (value + divisor - 1) // divisor


def _normalized_to_uint8_hwc(images: Tensor) -> Tensor:
    rgb01 = ((images.detach().cpu().to(dtype=torch.float32) + 1.0) * 0.5).clamp(
        0.0,
        1.0,
    )
    return (rgb01 * 255.0).round().to(dtype=torch.uint8).permute(0, 2, 3, 1)


def _write_rgb_png(path: Path, image: Tensor) -> None:
    if image.ndim != HWC_NDIM or int(image.shape[2]) != HWC_RGB_CHANNELS:
        message = "PNG image must have HWC RGB shape"
        raise ValueError(message)
    height = int(image.shape[0])
    width = int(image.shape[1])
    raw = _png_scanlines(image)
    payload = (
        _PNG_SIGNATURE
        + _png_chunk(
            b"IHDR",
            struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0),
        )
        + _png_chunk(b"IDAT", zlib.compress(raw))
        + _png_chunk(b"IEND", b"")
    )
    path.write_bytes(payload)


def _png_scanlines(image: Tensor) -> bytes:
    contiguous = image.contiguous()
    row_length = int(contiguous.shape[1]) * int(contiguous.shape[2])
    payload = bytearray((row_length + 1) * int(contiguous.shape[0]))
    flat_buffer = bytearray(contiguous.numel())
    flat_tensor = torch.frombuffer(flat_buffer, dtype=torch.uint8)
    flat_tensor.copy_(contiguous.view(-1))
    source = bytes(flat_buffer)
    for row_index in range(int(contiguous.shape[0])):
        output_start = row_index * (row_length + 1)
        input_start = row_index * row_length
        payload[output_start] = 0
        payload[output_start + 1 : output_start + 1 + row_length] = source[
            input_start : input_start + row_length
        ]
    return bytes(payload)


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(
            ">I",
            checksum,
        )
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _matrix_json(
    matrix: tuple[tuple[float, float, float], ...],
) -> list[list[float]]:
    return [list(row) for row in matrix]


def _required_object(payload: JsonObject, key: str) -> JsonObject:
    value = payload.get(key)
    if isinstance(value, dict):
        return cast("JsonObject", value)
    message = f"Expected object config field: {key}"
    raise TypeError(message)


def _required_int(payload: JsonObject, key: str) -> int:
    value = payload.get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    message = f"Expected integer config field: {key}"
    raise TypeError(message)


def _required_float(payload: JsonObject, key: str) -> float:
    value = payload.get(key)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    message = f"Expected numeric config field: {key}"
    raise TypeError(message)


def _required_string(payload: JsonObject, key: str) -> str:
    value = payload.get(key)
    if isinstance(value, str):
        return value
    message = f"Expected string config field: {key}"
    raise TypeError(message)


def _optional_string(payload: JsonObject, key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        return value
    message = f"Expected optional string config field: {key}"
    raise TypeError(message)


__all__ = [
    "LOCAL_STAIN_QA_KIND",
    "LOCAL_STAIN_QA_SOURCE",
    "LocalStainCorruptorQaRequest",
    "write_local_stain_corruptor_qa",
]
