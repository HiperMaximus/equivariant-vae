# Copyright 2026 HiperMaximus
# ruff: noqa: COM812, DOC201, DOC501, EM101, EM102, PLR0916, TRY003
"""Shared contracts for the frozen-VAE full-test reconstruction evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, cast

import torch
from torch import Tensor

from eqvae.metrics.reconstruction import (
    mae_per_image,
    mse_per_image,
    normalized_to_image_domain,
    psnr_per_image,
    ssim_per_image,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

SOURCE_LOCATION_HEADER: Final = (
    "run_number",
    "file_index",
    "atlas_row_index",
    "wsi_id",
    "x",
    "y",
    "split",
    "diagnosis_label",
    "diagnosis_index",
    "tissue_label",
)
REDACTED_LOCATION_HEADER: Final = SOURCE_LOCATION_HEADER[:7]
ORACLE_HEADER: Final = (
    "atlas_row_index",
    "wsi_id",
    "diagnosis_label",
    "diagnosis_index",
    "x",
    "y",
    "split",
)
REMOTE_METRIC_HEADER: Final = (
    *REDACTED_LOCATION_HEADER,
    "normal_mae_norm",
    "normal_mse_norm",
    "normal_psnr_img",
    "normal_ssim_img",
    "so2_mae_norm",
    "so2_mse_norm",
    "so2_psnr_img",
    "so2_ssim_img",
)
TEST_ROW_COUNT: Final = 67_138
TEST_WSI_COUNT: Final = 23
EXPECTED_WSI_PATCH_COUNTS: Final = {
    1252: 3000,
    4211: 3000,
    4797: 3000,
    4963: 3000,
    6281: 3000,
    6898: 3000,
    10246: 3000,
    12522: 3000,
    15470: 3000,
    15486: 3000,
    16064: 3000,
    19255: 3000,
    22221: 3000,
    27315: 3000,
    27950: 3000,
    38019: 3000,
    39466: 3000,
    40888: 3000,
    43815: 3000,
    46139: 3000,
    49281: 3000,
    50048: 1138,
    63165: 3000,
}
_STATE_SCHEMA: Final = b"eqvae_spec0045_state_dict_v1"
_ROW_SCHEMA: Final = b"eqvae_spec0045_redacted_location_v1"
_HASH_CHUNK_BYTES: Final = 8 * 1024 * 1024


@dataclass(frozen=True)
class TestLocation:
    """One diagnosis-free pointer to an exact frozen latent and image patch."""

    run_number: int
    file_index: int
    atlas_row_index: int
    wsi_id: int
    x: int
    y: int
    split: str

    def csv_values(self) -> tuple[int | str, ...]:
        """Return values in the locked redacted CSV order."""
        return (
            self.run_number,
            self.file_index,
            self.atlas_row_index,
            self.wsi_id,
            self.x,
            self.y,
            self.split,
        )


def sha256_file(path: Path) -> str:
    """Return the streaming SHA-256 of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    """Hash one JSON value under the repo's compact canonical encoding."""
    encoded = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def state_dict_sha256(state: Mapping[str, Tensor]) -> str:
    """Hash tensor names, metadata, and bytes independently of Torch serialization."""
    digest = hashlib.sha256(_STATE_SCHEMA)
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        metadata = json.dumps(
            {"dtype": str(tensor.dtype), "name": name, "shape": list(tensor.shape)},
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        payload = tensor.numpy().tobytes(order="C")
        digest.update(struct.pack("<Q", len(metadata)))
        digest.update(metadata)
        digest.update(struct.pack("<Q", len(payload)))
        digest.update(payload)
    return digest.hexdigest()


def redact_test_locations(
    *,
    source_path: Path,
    oracle_path: Path,
    output_path: Path,
    expected_source_sha256: str,
    expected_oracle_sha256: str,
) -> dict[str, object]:
    """Write the exact test pointers after proving and removing all label fields."""
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    if sha256_file(source_path) != expected_source_sha256:
        raise ValueError("Cancer-test location source SHA-256 differs")
    if sha256_file(oracle_path) != expected_oracle_sha256:
        raise ValueError("Cancer-test oracle SHA-256 differs")
    source_rows = _read_csv(source_path, SOURCE_LOCATION_HEADER)
    oracle_rows = _read_csv(oracle_path, ORACLE_HEADER)
    if len(source_rows) != len(oracle_rows):
        raise ValueError("Cancer-test source and oracle row counts differ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(REDACTED_LOCATION_HEADER)
        for index, (source, oracle) in enumerate(
            zip(source_rows, oracle_rows, strict=True),
        ):
            source_identity = tuple(source[name] for name in ORACLE_HEADER)
            oracle_identity = tuple(oracle[name] for name in ORACLE_HEADER)
            if source_identity != oracle_identity:
                raise ValueError(f"Cancer-test identity differs at row {index}")
            if source["tissue_label"]:
                raise ValueError(
                    "Cancer-test location source unexpectedly has tissue labels"
                )
            writer.writerow(tuple(source[name] for name in REDACTED_LOCATION_HEADER))
    locations = load_test_locations(
        output_path,
        expected_sha256=sha256_file(output_path),
        expected_row_count=len(source_rows),
        expected_wsi_patch_counts=EXPECTED_WSI_PATCH_COUNTS,
    )
    return {
        "bytes": output_path.stat().st_size,
        "row_count": len(locations),
        "row_identity_sha256": location_identity_sha256(locations),
        "sha256": sha256_file(output_path),
        "wsi_count": len({row.wsi_id for row in locations}),
    }


def load_test_locations(
    path: Path,
    *,
    expected_sha256: str,
    expected_row_count: int = TEST_ROW_COUNT,
    expected_wsi_patch_counts: Mapping[int, int] = EXPECTED_WSI_PATCH_COUNTS,
) -> tuple[TestLocation, ...]:
    """Load and fail closed on the complete diagnosis-free location sequence."""
    observed_sha256 = sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise ValueError(f"Redacted location SHA-256 differs: {observed_sha256}")
    raw_rows = _read_csv(path, REDACTED_LOCATION_HEADER)
    if len(raw_rows) != expected_row_count:
        raise ValueError("Redacted location row count differs")
    locations: list[TestLocation] = []
    counts: dict[int, int] = {}
    previous: tuple[int, int, int] | None = None
    previous_atlas = -1
    seen_pointers: set[tuple[int, int]] = set()
    for index, raw in enumerate(raw_rows):
        location = TestLocation(
            run_number=_positive_int(raw["run_number"], "run_number"),
            file_index=_nonnegative_int(raw["file_index"], "file_index"),
            atlas_row_index=_nonnegative_int(
                raw["atlas_row_index"],
                "atlas_row_index",
            ),
            wsi_id=_positive_int(raw["wsi_id"], "wsi_id"),
            x=_nonnegative_int(raw["x"], "x"),
            y=_nonnegative_int(raw["y"], "y"),
            split=raw["split"],
        )
        key = (location.wsi_id, location.y, location.x)
        pointer = (location.run_number, location.file_index)
        if (
            location.run_number not in range(1, 6)
            or location.split != "test"
            or location.x % 256
            or location.y % 256
            or (previous is not None and key <= previous)
            or location.atlas_row_index <= previous_atlas
            or pointer in seen_pointers
        ):
            raise ValueError(f"Redacted location invariant fails at row {index}")
        previous = key
        previous_atlas = location.atlas_row_index
        seen_pointers.add(pointer)
        counts[location.wsi_id] = counts.get(location.wsi_id, 0) + 1
        locations.append(location)
    if counts != dict(expected_wsi_patch_counts):
        raise ValueError("Redacted location WSI support differs")
    return tuple(locations)


def location_identity_sha256(locations: Sequence[TestLocation]) -> str:
    """Hash ordered redacted row identities without relying on CSV serialization."""
    digest = hashlib.sha256(_ROW_SCHEMA)
    for location in locations:
        for value in location.csv_values():
            digest.update(str(value).encode())
            digest.update(b"\0")
        digest.update(b"\n")
    return digest.hexdigest()


def paired_reconstruction_metrics(
    *,
    normal_reconstruction: Tensor,
    so2_reconstruction: Tensor,
    target_normalized: Tensor,
) -> dict[str, Tensor]:
    """Compute the eight locked FP32 per-image reconstruction metric vectors."""
    normal = reconstruction_metrics(
        reconstruction=normal_reconstruction,
        target_normalized=target_normalized,
    )
    so2 = reconstruction_metrics(
        reconstruction=so2_reconstruction,
        target_normalized=target_normalized,
    )
    return {
        **{f"normal_{name}": values for name, values in normal.items()},
        **{f"so2_{name}": values for name, values in so2.items()},
    }


def reconstruction_metrics(
    *,
    reconstruction: Tensor,
    target_normalized: Tensor,
) -> dict[str, Tensor]:
    """Compute the four locked FP32 per-image metrics for one decoder branch."""
    target = target_normalized.to(dtype=torch.float32)
    prediction = reconstruction.to(dtype=torch.float32)
    prediction_img = normalized_to_image_domain(prediction)
    target_img = normalized_to_image_domain(target)
    result = {
        "mae_norm": mae_per_image(prediction, target),
        "mse_norm": mse_per_image(prediction, target),
        "psnr_img": psnr_per_image(prediction_img, target_img),
        "ssim_img": ssim_per_image(prediction_img, target_img),
    }
    expected = target.shape[0]
    for name, values in result.items():
        if values.shape != (expected,):
            raise ValueError(f"Metric vector shape differs for {name}")
        if "psnr" not in name and not bool(torch.isfinite(values).all().item()):
            raise ValueError(f"Nonfinite metric values for {name}")
        if "psnr" in name and bool(torch.isnan(values).any().item()):
            raise ValueError(f"NaN PSNR values for {name}")
    return result


def _read_csv(path: Path, expected_header: Sequence[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(expected_header):
            raise ValueError(f"CSV header differs: {path}")
        return [cast("dict[str, str]", row) for row in reader]


def _positive_int(value: str, name: str) -> int:
    parsed = _nonnegative_int(value, name)
    if parsed == 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _nonnegative_int(value: str, name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if parsed < 0 or str(parsed) != value:
        raise ValueError(f"{name} must be canonical and nonnegative")
    return parsed


__all__ = [
    "EXPECTED_WSI_PATCH_COUNTS",
    "ORACLE_HEADER",
    "REDACTED_LOCATION_HEADER",
    "REMOTE_METRIC_HEADER",
    "SOURCE_LOCATION_HEADER",
    "TEST_ROW_COUNT",
    "TEST_WSI_COUNT",
    "TestLocation",
    "canonical_json_sha256",
    "load_test_locations",
    "location_identity_sha256",
    "paired_reconstruction_metrics",
    "reconstruction_metrics",
    "redact_test_locations",
    "sha256_file",
    "state_dict_sha256",
]
